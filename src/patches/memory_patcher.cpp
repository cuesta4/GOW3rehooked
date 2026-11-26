// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#include "patches/memory_patcher.h"
#include "main/host.h"
#include "orbis/core/loader/elf.h"
#include "orbis/core/memory.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <format>
#include <stdexcept>

namespace Gow3 {
void MemoryPatcher::Apply(std::span<const MemoryPatch> patches) {
    for (size_t i = 0; i < patches.size(); ++i) {
        const auto& p = patches[i];
        if (p.expected.empty() || p.expected.size() != p.replacement.size() ||
            !module.Contains(p.rva, p.expected.size()))
            throw std::runtime_error("Invalid patch range");
        if (std::memcmp(reinterpret_cast<const void*>(module.GetBaseAddress() + p.rva),
                        p.expected.data(), p.expected.size()))
            throw std::runtime_error("Patch original bytes do not match");
        for (size_t j = 0; j < i; ++j) {
            const auto& other = patches[j];
            if (p.rva < other.rva + other.expected.size() && other.rva < p.rva + p.expected.size())
                throw std::runtime_error("Overlapping patches");
        }
    }
    // Startup-only: write the loader's backing allocation without changing guest permissions.
    try {
        for (const auto& p : patches)
            if (!module.memory->TryWriteBacking(reinterpret_cast<void*>(module.GetBaseAddress() + p.rva),
                                                 p.replacement.data(), p.replacement.size()))
                throw std::runtime_error("Patch range has no writable loader backing");
#ifdef _WIN32
        if (!FlushInstructionCache(GetCurrentProcess(), nullptr, 0))
            throw std::runtime_error(HostError("Publish memory patches"));
#endif
    } catch (...) {
        for (const auto& p : patches)
            module.memory->TryWriteBacking(reinterpret_cast<void*>(module.GetBaseAddress() + p.rva),
                                            p.expected.data(), p.expected.size());
#ifdef _WIN32
        FlushInstructionCache(GetCurrentProcess(), nullptr, 0);
#endif
        throw;
    }
}

bool IsSupportedEboot(const std::filesystem::path& path) {
    struct Site {
        u64 rva;
        std::array<u8, 5> bytes;
        u32 size;
    };
    // Three of ApplyGamePatches' originals: the heap size, the save count and the save table size.
    static constexpr Site sites[]{{0x000db, {0x00, 0x00, 0x00, 0xb0}, 4}, {0x210a76, {0x83, 0x7f, 0x20, 0x1f}, 4},
                                  {0x21e3cd, {0xbf, 0x40, 0x04, 0x00, 0x00}, 5}};
    Core::Loader::Elf elf;
    elf.Open(path);
    if (!elf.IsElfFile()) return false;
    const auto headers = elf.GetProgramHeader();
    return std::ranges::all_of(sites, [&](const Site& s) {
        const auto h = std::ranges::find_if(headers, [&](const auto& h) {
            return h.p_type == PT_LOAD && s.rva >= h.p_vaddr && s.rva + s.size <= h.p_vaddr + h.p_filesz;
        });
        std::array<u8, 5> got{};
        if (h == headers.end()) return false;
        elf.LoadSegment(reinterpret_cast<u64>(got.data()), h->p_offset + (s.rva - h->p_vaddr), s.size);
        return std::memcmp(got.data(), s.bytes.data(), s.size) == 0;
    });
}

void ApplyGamePatches(Core::Module& eboot, const Config& config) {
    std::vector<MemoryPatch> patches;
    const auto value = [&](u64 rva, u32 original, u32 replacement, u32 size = 4) {
        const auto bytes = [size](u32 v) { return std::vector<u8>(reinterpret_cast<const u8*>(&v), reinterpret_cast<const u8*>(&v) + size); };
        patches.push_back({rva, bytes(original), bytes(replacement)});
    };
    // 64 manual saves plus AUTOSAVE/USERDATA; each directory name remains 32 bytes.
    patches.push_back({0x210a76, {0x83, 0x7f, 0x20, 0x1f}, {0x83, 0x7f, 0x20, 0x3f}});
    for (const u64 rva : {0x21e3cd, 0x21e42f})
        patches.push_back({rva, {0xbf, 0x40, 0x04, 0x00, 0x00}, {0xbf, 0x40, 0x08, 0x00, 0x00}});
    for (const u64 rva : {0x21e3e8, 0x21e446})
        patches.push_back({rva, {0xb9, 0x22, 0x00, 0x00, 0x00}, {0xb9, 0x42, 0x00, 0x00, 0x00}});
    // User Malloc (General Heap) reduction from 2816 MiB (0xB0000000) to 1888 MiB (0x76000000).
    // Verified in Ghidra at FUN_010000a0 and FUN_010002f0 (allocate, map, mspace create).
    for (const u64 rva : {0x000db, 0x00115, 0x0013c, 0x0031a, 0x0034f, 0x0037a}) {
        value(rva, 0xb0000000, 0x76000000);
    }
    // Resolution patches (kvicken + cuesta4): output size, the fixed 1920x1080 pool render target and its users, and
    // the half-resolution luminance buffers; all of them below 1080p too, since effects sample them with the scene's
    // texel size. Every option includes the Texture Corruption Fix: Video Arena 1.25 GiB -> `arena`.
    const auto& resolution = Resolutions[config.resolution];
    if (config.resolution != NativeResolution) {
        struct Site { u64 rva; u32 size; bool height, half; };
        static constexpr Site sites[]{
            {0x1f0c5b, 4, false}, {0x1f0c63, 4, true}, {0x1f0c6c, 2, false}, {0x1f0c73, 2, true},
            {0x1643e3, 4, true}, {0x1643f0, 4, false}, {0x1ff889, 4, false}, {0x1ff88f, 4, true},
            {0x3c46b4, 4, false}, {0x3c46b8, 4, true}, {0x16bb72, 4, true}, {0x20965d, 4, true},
            {0x3f2a3c, 4, false, true}, {0x3f2a40, 4, true, true},
        };
        for (const auto& site : sites) {
            const u32 divisor = site.half ? 2 : 1;
            value(site.rva, (site.height ? 1080 : 1920) / divisor, u32(site.height ? resolution.height : resolution.width) / divisor, site.size);
        }
    }
    value(0x3cd918, 0x50000000, VideoArenaHighByte(config) << 24);
    if (config.msaa != 2) value(0x3cdcd0, 1, std::countr_zero(config.msaa));
    if (config.fps != 60) { // Keep swing normalization at the original 59.94 Hz for every other rate.
        patches.push_back({0x08ff0a, {0x48, 0x8d, 0x05, 0x6f, 0xdd, 0x33, 0x00}, {0x48, 0x8d, 0x05, 0xf3, 0xdd, 0x2c, 0x00}});
        // The timing helper alone reads these constants. Preserve its original maximum delta, 4/59.94 s.
        value(0x35dca8, 0x43c80000, std::bit_cast<u32>(400.f * (static_cast<float>(config.fps) / 60.f)));
        value(0x35dcac, 0x40800000, std::bit_cast<u32>(4.f * (static_cast<float>(config.fps) / 60.f)));
    }
    if (config.skip_intro) { // Skip Intro (cuesta4): skips the initial logos.
        patches.push_back({0x184940, {0x75, 0x34}, {0xeb, 0x34}});
        patches.push_back({0x0c8430, {0x48, 0x8d, 0x05, 0xa5, 0x5b, 0x38, 0x00, 0x8b, 0x00, 0xc3, 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00},
                                     {0xc7, 0x05, 0x1e, 0x0e, 0x9b, 0x01, 0x01, 0x00, 0x00, 0x00, 0xb8, 0x01, 0x00, 0x00, 0x00, 0xc3}});
        patches.push_back({0x01b36a, {0xe8, 0x11, 0x99, 0x09, 0x00}, {0xe8, 0x21, 0x6f, 0x1f, 0x00}});
        patches.push_back({0x212290, {0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90},
                                     {0xc7, 0x05, 0xbe, 0x6f, 0x86, 0x01, 0x01, 0x00, 0x00, 0x00, 0xe9, 0xe1, 0x29, 0xea, 0xff}});
    }
    if (config.skip_videos) { // Skip Any Video With X Button (cuesta4).
        patches.push_back({0x01b2a3, {0x75, 0x4a}, {0x75, 0x42}});
        patches.push_back({0x01b2e7, {0x85, 0xc0, 0x0f, 0x95, 0xc0, 0x0f, 0xb6, 0xc8}, {0xb9, 0x01, 0x00, 0x00, 0x00, 0x90, 0x90, 0x90}});
    }
    if (config.cinematics_4k) {
        // Frame queue capacity, decoder limits, and the Y/UV render height clamp.
        value(0x1f2d69, 1920, 3840);
        value(0x1f2d6e, 1080, 2160);
        value(0x2e1c53, 42, 51);
        value(0x2e1c67, 1920, 3840);
        value(0x2e1c71, 1088, 2160);
        value(0x1f24bb, 1080, 2160);
        value(0x1f24c0, 1080, 2160);
        // Retain two 512 KiB blocks when the compressed stream wraps; 4K access units can exceed one block.
        struct Wrap { u64 rva; u32 displacement; };
        for (const auto [rva, displacement] : {Wrap{0x1f2095, 0x0188b1e1}, Wrap{0x1f21f1, 0x0188b085}, Wrap{0x1f2cca, 0x0188a5ac}}) {
            std::vector<u8> before{0xc1, 0xe1, 0x13, 0x48, 0x8b, 0x1d};
            const auto* disp = reinterpret_cast<const u8*>(&displacement);
            before.insert(before.end(), disp, disp + 4);
            before.insert(before.end(), {0x48, 0x63, 0xf1, 0x48, 0x01, 0xde, 0xba, 0x00, 0x00, 0x08, 0x00});
            const u32 moved_displacement = displacement + 1;
            const auto* moved = reinterpret_cast<const u8*>(&moved_displacement);
            std::vector<u8> after{0x90, 0x90, 0x48, 0x8b, 0x1d};
            after.insert(after.end(), moved, moved + 4);
            after.insert(after.end(), {0x48, 0x8d, 0xb3, 0x00, 0x00, 0x90, 0x00, 0xba, 0x00, 0x00, 0x10, 0x00});
            patches.push_back({rva, std::move(before), std::move(after)});
            value(rva + 0x23, 1, 2);
            value(rva + 0x28, 0x80000, 0x100000);
        }
        value(0x1f2c07, 0x980000, 0x900000);
    }
    MemoryPatcher(eboot).Apply(patches);
    if (DebugEnabled()) Log(std::format("Applied {} game patch sites: {} at {:.2f} fps; 64 manual saves", patches.size(), resolution.name, GameFrameRate(config.fps)));
}
}
