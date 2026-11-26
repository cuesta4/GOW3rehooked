// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string_view>

namespace Gow3 {
enum class DisplayMode { Windowed, Borderless, Fullscreen };
enum class VSync { Fifo, Mailbox, Off }; // swapchain present mode
enum class AntiAliasing { None, Cmaa2 };
enum class ScalingFilter { Bilinear, Fsr1 };
enum class ShaderRecompilation { Sync, Async };
enum class ShaderMechanism { Jit, Aot };
// Output resolutions of the resolution patches (God_of_War_III_Remastered.xml, kvicken + cuesta4).
// Base video arena in the stock eboot is 0x50000000 (1280 MiB), high byte 0x50.
// Every option enlarges the Video Arena (`arena` is the high byte of its size, where 1 unit = 16 MiB).
// Extra DMEM is aligned strictly 1:1 with the additional video memory requested over base.
inline constexpr unsigned BaseVideoArenaHighByte = 0x50;
inline constexpr unsigned BaseVideoArenaMb = BaseVideoArenaHighByte * 16; // 1280 MiB

inline constexpr unsigned CalculateExtraDmemMb(unsigned arena_high_byte) {
    return arena_high_byte > BaseVideoArenaHighByte ? (arena_high_byte - BaseVideoArenaHighByte) * 16 : 0;
}

struct Resolution {
    std::string_view name;
    int width, height;
    unsigned arena;
    unsigned dmem_mb;

    constexpr Resolution(std::string_view n, int w, int h, unsigned a)
        : name(n), width(w), height(h), arena(a), dmem_mb(CalculateExtraDmemMb(a)) {}

    constexpr Resolution(std::string_view n, int w, int h, unsigned a, unsigned d)
        : name(n), width(w), height(h), arena(a), dmem_mb(d) {}
};
inline constexpr std::array<Resolution, 6> Resolutions{{
    {"480p", 854, 480, 0x70},
    {"720p", 1280, 720, 0x70},
    {"1080p", 1920, 1080, 0x70},
    {"1440p", 2560, 1440, 0x78},
    {"1800p", 3200, 1800, 0x82},
    {"2160p", 3840, 2160, 0x8e},
}};
inline constexpr unsigned NativeResolution = 2;
inline constexpr std::array<unsigned, 3> MsaaLevels{1, 2, 4};
inline constexpr std::array<unsigned, 4> AnisotropyLevels{0, 4, 8, 16}; // 0 keeps the game's samplers
struct FrameRate {
    int nominal;
    std::string_view name;
};
inline constexpr std::array<FrameRate, 6> FrameRates{{
    {30, "29.97"}, {40, "39.96"}, {60, "59.94"}, {120, "119.88"}, {240, "239.76"}, {480, "479.52"},
}};
// Match the eboot's original 0x426fc28f float, including its rounding.
inline constexpr float GameFrameRate(int nominal) { return 59.94f * (static_cast<float>(nominal) / 60.f); }
inline constexpr std::int64_t FramePeriodNs(int nominal) {
    return std::llround(1'000'000'000.0 / GameFrameRate(nominal));
}
struct Config {
    unsigned resolution = NativeResolution; // index into Resolutions
    int fps = 60;                           // nominal FrameRates selection; actual rate is GameFrameRate(fps)
    unsigned anisotropy = 0;
    unsigned msaa = 2;
    bool skip_intro = false;
    bool skip_videos = true;                // cinematics can be skipped with X
    bool cinematics_4k = false;
    DisplayMode display_mode = DisplayMode::Borderless;
    bool vrr = false;                       // full screen: the display at its highest refresh rate
    VSync vsync = VSync::Fifo;
    bool nvidia_reflex = false;
    AntiAliasing anti_aliasing = AntiAliasing::None;
    ScalingFilter scaling_filter = ScalingFilter::Bilinear;
    ShaderRecompilation shader_recompilation = ShaderRecompilation::Async;
    ShaderMechanism shader_mechanism = ShaderMechanism::Aot;
    bool lazy_aot = true;                   // AOT builds a fixed quarter of all WAD variants per eligible launch
    unsigned vibration = 100;               // 0..100 controller vibration strength percentage
    bool cheat_infinite_health = false;
    bool cheat_infinite_magic = false;
    bool cheat_infinite_item = false;
    bool cheat_infinite_rage = false;
    bool cheat_max_red_orbs = false;
    bool cheat_infinite_double_jumps = false;
    bool cheat_unlock_chaos = false;
    bool cheat_auto_qte = false;
    bool remove_motion_blur = false;
    bool remove_dof_spartan = false;
};
// MSAA above the game's 2x grows its two color targets and its depth/stencil (4 + 4 + 4 + 1 bytes a sample) inside the
// Video Arena, and the texture heap (the arena's largest free block - 256 MiB) loses them: at 2160p 4x it fell to 768 MiB
// and streaming overflowed it, so failed allocations left textures sharing one block (#97). Those bytes go back here.
inline constexpr unsigned VideoArenaHighByte(const Config& c) {
    const Resolution& r = Resolutions[c.resolution];
    const unsigned long long msaa = c.msaa > 2 ? 13ull * r.width * r.height * (c.msaa - 2) : 0;
    return r.arena + unsigned((msaa + (16ull << 20) - 1) >> 24);
}
Config LoadConfig(const std::filesystem::path& path);
void SaveMenuConfig(const std::filesystem::path& path, const Config& config);
void SaveExtrasConfig(const std::filesystem::path& path, const Config& config);
}
