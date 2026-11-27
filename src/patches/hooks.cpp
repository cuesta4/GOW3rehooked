// SPDX-License-Identifier: GPL-2.0-or-later
// Game engine detours: timing, movement, title menu and optional rendering diagnostics.
// EBOOT 6E72330E9092E5164049B865E5568F2D450904001A73FE0A2F4B39194C195B3C (VA = 0x01000000 + RVA).
#include "patches/hooks.h"
#include "ui/game_language.h"
#include "ui/translation.h"
#include "debug/dump_archive.h"
#include "gpu/device.h"
#include "gpu/gpu_thread.h"
#include "ui/controls_ui.h"
#include "patches/extras.h"
#include "patches/memory_patcher.h"
#include "main/host.h"
#ifndef _WIN32
#include <strings.h>
#include <sys/mman.h>
#endif
#include "present/presenter.h"
#include "present/splash.h"
#include "present/window.h"
#include "orbis/core/memory.h"
#include "orbis/internals/kernel/time.h"
#include "orbis/internals/save_data/dialog/savedatadialog.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <immintrin.h>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
#include <SDL3/SDL.h>
#ifdef _WIN32
#include <evntprov.h>
#pragma comment(lib, "advapi32.lib")
#endif

namespace Gow3 {
namespace {
[[gnu::always_inline]] inline int CompareCaseInsensitive(const char* lhs, const char* rhs) {
#ifdef _WIN32
    return _stricmp(lhs, rhs);
#else
    return strcasecmp(lhs, rhs);
#endif
}

// Shared hook machinery.
uintptr_t guest_base;
float configured_frame_rate;
using InputUpdate = void(PS4_SYSV_ABI*)();
InputUpdate original_input_update;
void PS4_SYSV_ABI UpdateInput() { Presenter::BeforeInput(); original_input_update(); }
// FUN_011ea670, the frame loop: its first step (011F0F00, FlipArenas) resets arenas earlier submits drew (GpuThread::AwaitCopies).
// Each start is an ETW event (provider E5B7A118-EAA4-4971-BCA2-413393ACFA70, id 1, the frame number as u64) that
// tools\gow3.wprp records beside the WPR CPU profile; etlprof counts frames with it. Nearly free while no session listens.
using FrameLoop = void(PS4_SYSV_ABI*)(u32);
FrameLoop original_frame_loop;
#ifdef _WIN32
REGHANDLE frame_events;
constexpr GUID kFrameProvider{0xe5b7a118, 0xeaa4, 0x4971, {0xbc, 0xa2, 0x41, 0x33, 0x93, 0xac, 0xfa, 0x70}};
constexpr EVENT_DESCRIPTOR kFrameEvent{1, 0, 0, 4, 0, 0, 0};  // level 4: information
#endif
void HoldSlowMotionOnset();
void PS4_SYSV_ABI RunFrame(u32 mode) {
#ifdef _WIN32
    static u64 frame;
    if (++frame; EventEnabled(frame_events, &kFrameEvent)) {
        EVENT_DATA_DESCRIPTOR data;
        EventDataDescCreate(&data, &frame, sizeof(frame));
        EventWrite(frame_events, &kFrameEvent, 1, &data);
    }
#endif
    GpuThread::AwaitCopies();
    HoldSlowMotionOnset();
    original_frame_loop(mode);
}
template <class T> T Guest(u64 rva) { return reinterpret_cast<T>(guest_base + rva); }
// 011F0F00 starts each pass on the other half of the double arena (02A78060/68, index 02A7805C) and resets it and the 1 MB
// arena (02A78070) with 011AABA0. Every user allocates from table[index] (57 sites, all through 011AABE0, none from the other
// half); 011F0FD0 tries the small one first. A pass's submit (011F09F0) draws both halves and its small arena, so the next
// pass reset what was just submitted and RunFrame waited for the GPU thread's copy (~0.45 ms a frame). A spare of each now
// rides through the slots: the slot being reset takes the spare and its arena rests one pass, so a submit's arenas outlive the
// next pass start and its copy runs beside that pass (GpuThread::AwaitCopies waits for the submit before).
u64 spare_arenas[2];  // descriptors resting this pass: the half, the 1 MB arena (game main thread)
bool MapSpareArenas() {
    constexpr u64 kHalf = 0xf80000, kSmall = 0x100000, kDescs = 0x10000, kSize = kDescs + kHalf + kSmall;  // sizes as 011F0DF0
    // A System mapping: guest memory inside the renderer's mirror (below 32 GB) outside the game's flexible and direct budgets.
    void* block = nullptr;
    if (Core::Memory::Instance()->MapMemory(&block, 0, kSize, Core::MemoryProt::CpuReadWrite | Core::MemoryProt::GpuReadWrite,
                                            Core::MemoryMapFlags::NoFlags, Core::VMAType::System, "gow3 spare arenas") ||
        reinterpret_cast<u64>(block) + kSize > (32ull << 30))
        return Log("Frame arenas: no guest memory for the spares; each pass waits for the last submit's copy"), false;
    const u64 b = reinterpret_cast<u64>(block);
    const auto init = Guest<void(PS4_SYSV_ABI*)(u64, u64, u32)>(0x1aab70);
    init(b, b + kDescs, kHalf), init(b + 0x80, b + kDescs + kHalf, kSmall);
    spare_arenas[0] = b, spare_arenas[1] = b + 0x80;
    Log(std::format("Frame arenas: spares at {:#x}; the frame loop no longer waits for the last submit's copy", b));
    return true;
}
void PS4_SYSV_ABI FlipArenas() {
    static const bool spares = MapSpareArenas();
    auto& index = *Guest<s32*>(0x1a7805c);
    auto* const table = Guest<u64*>(0x1a78060);  // the halves, then the 1 MB arena
    auto* const stats = Guest<s32*>(0x1a78078);  // used and peak of the half and the small arena reset; 011F0FD0's overflow, peak
    // Descriptor {u32 front, u32 back, u64 base, u64 end}: 011AAC40 (capacity) minus 011AAC30 (free).
    const auto used = [](u64 a) {
        const auto* d = reinterpret_cast<const u32*>(a);
        return s32(*reinterpret_cast<const u64*>(a + 16) - *reinterpret_cast<const u64*>(a + 8)) - s32(d[1] - d[0]);
    };
    index = 1 - index;
    stats[0] = used(table[index]), stats[1] = std::max(stats[1], stats[0]);
    stats[2] = used(table[2]), stats[3] = std::max(stats[3], stats[2]);
    stats[5] = std::max(stats[5], stats[4]), stats[4] = 0;
    if (spares) {
        std::swap(table[index], spare_arenas[0]), std::swap(table[2], spare_arenas[1]);
        GpuThread::SetSpareArenas(spare_arenas[0], spare_arenas[1]);
    }
    const auto reset = Guest<void(PS4_SYSV_ABI*)(u64)>(0x1aaba0);
    reset(table[2]), reset(table[index]);
}
using CinematicPathFn = char*(PS4_SYSV_ABI*)(const char*, char*, int, u64);
CinematicPathFn original_cinematic_path;
char* PS4_SYSV_ABI CinematicPath(const char* name, char* output, int size, u64 flags) {
    if (name && (flags & 0x80) && !(flags & 0x40) && CompareCaseInsensitive(name, "OPENINGINT.BSF") == 0)
        name = "4K/OPENINGINT_nomos8k_30.0fps_3840x2160.bsf";
    return original_cinematic_path(name, output, size, flags);
}
u32 Caller(void* address) { return u32(reinterpret_cast<uintptr_t>(address) - guest_base + 0x1000000); }

// Copies a verified prologue into executable memory followed by a jump back past it; `branch` is the offset of
// one rel32 call or conditional branch inside the prologue, retargeted through an absolute jump.
void* PrepareTrampoline(u64 rva, std::span<const u8> prologue, u32 branch = 0) {
    const auto* entry = Guest<const u8*>(rva);
    if (std::memcmp(entry, prologue.data(), prologue.size()))
        throw std::runtime_error(std::format("Engine prologue at RVA {:#x} does not match EBOOT", rva));
    std::vector<u8> code(prologue.size() + 14 + (branch ? 14 : 0));
    std::copy(prologue.begin(), prologue.end(), code.begin());
    const auto jump = [&](size_t at, uintptr_t target) {
        code[at] = 0xff; code[at + 1] = 0x25;
        std::memcpy(code.data() + at + 6, &target, sizeof(target));
    };
    jump(prologue.size(), reinterpret_cast<uintptr_t>(entry + prologue.size()));
    if (branch) {
        const u32 end = branch + (prologue[branch] == 0xe8 ? 5 : 6);
        s32 displacement;
        std::memcpy(&displacement, prologue.data() + end - 4, sizeof(displacement));
        jump(prologue.size() + 14, reinterpret_cast<uintptr_t>(entry + end + displacement));
        displacement = s32(prologue.size() + 14 - end);
        std::memcpy(code.data() + end - 4, &displacement, sizeof(displacement));
    }
#ifdef _WIN32
    auto* allocation = VirtualAlloc(nullptr, code.size(), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!allocation) throw std::runtime_error("Allocate engine trampoline");
    std::memcpy(allocation, code.data(), code.size());
    DWORD old{};
    if (!VirtualProtect(allocation, code.size(), PAGE_EXECUTE_READ, &old) ||
        !FlushInstructionCache(GetCurrentProcess(), allocation, code.size()))
        throw std::runtime_error("Publish engine trampoline");
#else
    auto* allocation = mmap(nullptr, code.size(), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (allocation == MAP_FAILED) throw std::runtime_error(HostError("Allocate engine trampoline"));
    std::memcpy(allocation, code.data(), code.size());
    if (mprotect(allocation, code.size(), PROT_READ | PROT_EXEC) != 0) {
        const auto error = HostError("Publish engine trampoline");
        munmap(allocation, code.size());
        throw std::runtime_error(error);
    }
#endif
    return allocation;
}

// 011A5840 (a worker takes a job and returns its node) and 011A5D00 (enqueue) guard the job queue with a bare
// `lock cmpxchg` loop on DAT_024ba6e0. On the console each of these threads owns a core; here the holder can lose its CPU
// and the others spin until it runs again. The loops call this instead: test before the exchange, pause between tries
// and, every ~16 us still held, hand the CPU to whatever is ready on it.
void PS4_SYSV_ABI LockJobQueue() {
    const std::atomic_ref lock(*Guest<u64*>(0x14ba6e0));
    for (u64 start = __rdtsc();;) {
        if (u64 free = 0; !lock.load(std::memory_order_relaxed) && lock.compare_exchange_strong(free, 1)) return;
        _mm_pause();
        if (__rdtsc() - start > 0x10000) {
#ifdef _WIN32
            SwitchToThread();
#else
            std::this_thread::yield();
#endif
            start = __rdtsc();
        }
    }
}

// 011A5E30 waits for a job group's count to reach zero in a bare loop, on a core of its own on the console. Here the
// waiting thread would keep a processor from the workers it waits for: the same pacing as the lock.
void PS4_SYSV_ABI WaitJobs(const std::atomic<s32>* pending) {
    for (u64 start = __rdtsc(); pending->load(std::memory_order_relaxed);) {
        _mm_pause();
        if (__rdtsc() - start > 0x10000) {
#ifdef _WIN32
            SwitchToThread();
#else
            std::this_thread::yield();
#endif
            start = __rdtsc();
        }
    }
}

// 01165AC0 refills the color correction LUT on every frame its blend stack is active, also while the stack holds still: it
// flips effect+0x160 to the other half of the 0x40000 buffer at effect+0x1C0 and starts the group at 0146A5C0 with 32 jobs of
// 011E08C0, one 4 KiB slice of the 32^3 RGBA8 lattice each (~1.5 ms of worker CPU per frame). 011E08C0 depends only on the
// 0xE0-byte block at effect+0x1B8 and the slice, the effect lives for the session (010B1F40) and only these jobs write the
// buffer. A half last filled from the same block (by a group the previous start completed) already holds those bytes: the
// group starts empty, and 01166090 draws without waiting.
void PS4_SYSV_ABI FillColorLattice(const u8* effect) {
    static struct { u64 lut; u8 block[0xe0]; } filled[2];
    const u64 base = *reinterpret_cast<const u64*>(effect + 0x1c0), lut = *reinterpret_cast<const u64*>(effect + 0x160);
    const auto* block = *reinterpret_cast<const u8* const*>(effect + 0x1b8);
    auto& half = filled[(lut - base) >> 17 & 1];
    const bool kept = half.lut == lut && !std::memcmp(half.block, block, sizeof(half.block));
    half.lut = lut, std::memcpy(half.block, block, sizeof(half.block));
    Guest<u8(PS4_SYSV_ABI*)(void*, s32)>(0x2c61f0)(Guest<void*>(0x46a5c0), kept ? 0 : 32);
}

// Gameplay and timing hooks.
struct alignas(16) CollisionHit {
    __m128 point, normal;
    float distance;
    u32 unknown;
    uintptr_t node, surface;
    u8 extra[40];
};
static_assert(sizeof(CollisionHit) == 0x60 && offsetof(CollisionHit, node) == 0x28);
using RayQuery = u32(PS4_SYSV_ABI*)(__m128, __m128, u64, u64, u64, u64, u64, u64);
using ContactQuery = u64(PS4_SYSV_ABI*)(__m128, __m128, u64, u64, u64, u64, u64, u64);
using ModeQuery = u64(PS4_SYSV_ABI*)(uintptr_t, u32, u32, uintptr_t, __m128*);
using MotionPhysics = __m128(PS4_SYSV_ABI*)(uintptr_t, uintptr_t);
RayQuery original_ray_query;
ContactQuery original_hang_query;
ModeQuery original_mode_query;
MotionPhysics original_motion_physics, original_air_motion;
struct CornerMotion { uintptr_t actor; __m128 offset; bool valid{}; };
thread_local CornerMotion* corner_motion{};
struct AirFrame { uintptr_t actor, context; __m128 offset; float weight; bool applied{}; };
thread_local AirFrame* air_frame{};

uintptr_t MainActor() {
    const auto table = *Guest<const uintptr_t*>(0x44c098);
    return table ? *reinterpret_cast<const uintptr_t*>(table + 0x18) : 0;
}
__m128 ActorCenterOffset(uintptr_t actor) {
    const auto offset = _mm_loadu_ps(reinterpret_cast<const float*>(actor + 0x180));
    return _mm_add_ps(_mm_add_ps(
        _mm_mul_ps(_mm_shuffle_ps(offset, offset, 0), _mm_loadu_ps(reinterpret_cast<const float*>(actor + 0x70))),
        _mm_mul_ps(_mm_shuffle_ps(offset, offset, 0x55), _mm_loadu_ps(reinterpret_cast<const float*>(actor + 0x80)))),
        _mm_mul_ps(_mm_shuffle_ps(offset, offset, 0xaa), _mm_loadu_ps(reinterpret_cast<const float*>(actor + 0x90))));
}
float Dot3(__m128 a, __m128 b) {
    const auto p = _mm_mul_ps(a, b);
    return _mm_cvtss_f32(_mm_add_ss(_mm_add_ss(p, _mm_shuffle_ps(p, p, 0x55)), _mm_shuffle_ps(p, p, 0xaa)));
}
u32 ContactFlags(uintptr_t surface) {
    if (!surface) return 0;
    const auto* p = reinterpret_cast<const u8*>(surface + *Guest<const u32*>(0x420500));
    return u32(p[0]) << 24 | u32(p[1]) << 16 | u32(p[2]) << 8 | p[3];
}

// Motion sounds compare per-frame movement with cutoffs authored for 59.94 Hz; above it they never start (#9).
struct MotionSoundOperands {
    float cutoff;                             // 0.01 scaled to the frame's gameplay step
    alignas(16) float displacement_scale[4];  // the actor's displacement scaled to its 59.94 Hz share
};
MotionSoundOperands* motion_sound;

void UpdateMotionSound() {
    if (!motion_sound) return;
    constexpr float reference_rate = GameFrameRate(60);
    const float gameplay_dt = *Guest<const float*>(0x3cdcb0), frame_dt = *Guest<const float*>(0x3cdca0);
    motion_sound->cutoff = gameplay_dt > 0.f ? 0.01f * gameplay_dt * reference_rate : 0.01f;
    std::fill_n(motion_sound->displacement_scale, 4, frame_dt > 0.f ? 1.f / (frame_dt * reference_rate) : 1.f);
}

// 011EFE10: the timing globals at the configured rate, without overwriting the 59.94 Hz constant.
void PS4_SYSV_ABI SetFrameRate(float) {
    *Guest<float*>(0x3cdc80) = configured_frame_rate;
    *Guest<s32*>(0x1a77f30) = s32(std::lround(configured_frame_rate));
    const float delta = 1.f / configured_frame_rate;
    *Guest<float*>(0x3cdca0) = delta;
    std::fill_n(Guest<float*>(0x3cdcb0), 4, delta);
    UpdateMotionSound();
}

// 011EF9E0: the frame's step, the TSC since its last call (02A77F40) in nominal frames, at least one and four above four.
// Short frames (catch-ups after a stall, FIFO jitter) counted whole ran the game ahead of the clock: ~2% on average and
// ~20% over the half second after a stall at 59.94 Hz FIFO. Their excess is now debt the next longer frames pay.
using FrameStepFn = void(PS4_SYSV_ABI*)();
FrameStepFn original_frame_step;
void PS4_SYSV_ABI StepFrame() {
    static const float tsc = float(Libraries::Kernel::sceKernelGetTscFrequency());
    static float debt;  // <= 0
    original_frame_step();
    // The variable step only: not the fixed step (02A77F50) or the averaged and doubled ones (02A77F34).
    if ((*Guest<const u32*>(0x3cdcc0) & 3) != 1 || *Guest<const s32*>(0x1a77f50) || *Guest<const s32*>(0x1a77f34)) {
        debt = 0;
        return UpdateMotionSound();
    }
    const float rate = *Guest<const float*>(0x3cdc80);
    const float want = std::min(float(*Guest<const s64*>(0x1a77f40)) * rate / tsc, 4.f) + debt, step = std::max(want, 1.f);
    debt = std::max(want - step, -4.f);
    float& dt = *Guest<float*>(0x3cdca0);
    const float k = step / (dt * rate);  // the channels (013CDCB0) scale with the step
    dt *= k;
    for (float& c : std::span(Guest<float*>(0x3cdcb0), 4)) c *= k;
    UpdateMotionSound();
}

void AddMotionSoundPatches(Core::Module& eboot, std::vector<MemoryPatch>& patches) {
    // The operands (first 16 KB) and the displacement stub (next 16 KB) within rel32 reach of the EBOOT.
    void* page{};
    if (eboot.memory->MapMemory(&page, guest_base, 0x8000, Core::MemoryProt::CpuReadWrite, Core::MemoryMapFlags::NoFlags,
                                Core::VMAType::System, "gow3 motion sound"))
        throw std::runtime_error("Map motion sound operands");
    motion_sound = static_cast<MotionSoundOperands*>(page);
    *motion_sound = {0.01f, {1.f, 1.f, 1.f, 1.f}};  // SetFrameRate supplies the delta before the first simulation step.
    const auto rel32 = [](uintptr_t end, const void* target) {
        const auto displacement = s64(reinterpret_cast<uintptr_t>(target)) - s64(end);
        if (!std::in_range<s32>(displacement)) throw std::runtime_error("Motion sound operand out of rel32 range");
        return s32(displacement);
    };
    // `vucomiss xmm, [0.01]` on the movement of the frame: the push/pull loop (01062810) and scrape (01062CD0), then
    // both start/stop pairs of the balance beam's loop (012B51C0, 012B5800).
    for (const auto& [rva, original] : std::initializer_list<std::pair<u64, std::vector<u8>>>{
             {0x62893, {0xc5, 0xf8, 0x2e, 0x0d, 0x29, 0x21, 0x29, 0x00}},
             {0x63032, {0xc5, 0xf8, 0x2e, 0x05, 0xaa, 0x19, 0x29, 0x00}},
             {0x2b539f, {0xc5, 0xf8, 0x2e, 0x05, 0xa5, 0xc0, 0x09, 0x00}},
             {0x2b5407, {0xc5, 0xf8, 0x2e, 0x05, 0x3d, 0xc0, 0x09, 0x00}},
             {0x2b58ad, {0xc5, 0xf8, 0x2e, 0x05, 0xab, 0xbb, 0x09, 0x00}},
             {0x2b5915, {0xc5, 0xf8, 0x2e, 0x05, 0x43, 0xbb, 0x09, 0x00}}}) {
        auto replacement = original;
        const s32 rel = rel32(guest_base + rva + original.size(), &motion_sound->cutoff);
        std::memcpy(replacement.data() + 4, &rel, sizeof(rel));
        patches.push_back({rva, original, std::move(replacement)});
    }
    // 0109E9FC, the actor's motion sounds (0109E820, the Gaia pillar's SND_PUSHPULL_LP): `vmovaps xmm0, [rdx+0x140]` loads
    // its displacement of the frame, whose length starts the loop above 0.02 and sets its intensity. It jumps to a copy
    // that scales it to its 59.94 Hz share and back.
    constexpr u64 load = 0x9e9fc;
    auto* const code = static_cast<u8*>(page) + 0x4000;
    const auto stub = reinterpret_cast<uintptr_t>(code);
    u8 scaled[]{0xc5, 0xf8, 0x28, 0x82, 0x40, 0x01, 0x00, 0x00,  // vmovaps xmm0, [rdx+0x140]
                0xc5, 0xf8, 0x59, 0x05, 0, 0, 0, 0,              // vmulps xmm0, xmm0, [displacement_scale]
                0xe9, 0, 0, 0, 0};                               // jmp 0109EA04
    const s32 scale = rel32(stub + 16, motion_sound->displacement_scale);
    const s32 back = rel32(stub + 21, reinterpret_cast<const void*>(guest_base + load + 8));
    std::memcpy(scaled + 12, &scale, sizeof(scale));
    std::memcpy(scaled + 17, &back, sizeof(back));
    std::memcpy(code, scaled, sizeof(scaled));
    if (eboot.memory->Protect(stub, 0x4000, Core::MemoryProt::CpuRead | Core::MemoryProt::CpuExec))
        throw std::runtime_error("Protect motion sound stub");
    std::vector<u8> jump{0xe9, 0, 0, 0, 0, 0x0f, 0x1f, 0x00};  // jmp stub; nop
    const s32 out = rel32(guest_base + load + 5, code);
    std::memcpy(jump.data() + 1, &out, sizeof(out));
    patches.push_back({load, {scaled, scaled + 8}, std::move(jump)});
}

// 010E7DF0: an animation event (FUN_010e1810 case 0x12) starts a slow motion, scale DAT_0144e9f4 for DAT_0144e9f8 s through
// DAT_013cdc8c, from the next frame. At 29.97 Hz the event's frame carried the animations up to 33 ms past it, so events
// authored just after it fired in the same frame: Hermes's leap starts a 1% slow motion ~17 ms of animation before its QTE.
// At 120 FPS the frame stops ~8 ms past it and the QTE waited for the 1% speed to cover the rest (its prompt showed after
// the 1.5 s slow motion). The slow motion now starts once the frames since its event add up to one 29.97 Hz frame; the
// held frames don't count in its duration.
void HoldSlowMotionOnset() {
    static float scale, hold, previous;
    static bool holding;
    float &left = *Guest<float*>(0x44e9f8), &target = *Guest<float*>(0x44e9f4);
    float &applied = *Guest<float*>(0x3cdc8c), &inverse = *Guest<float*>(0x3cdc9c);
    const float dt = *Guest<const float*>(0x3cdca0); // the frame that just ran
    if (holding) left += dt, hold -= dt;
    else if (previous <= 0 && left > 0 && target < 1) holding = true, hold = 1 / 29.97f - dt;
    previous = left;
    if (!holding) return;
    if (target < 1) scale = target; // its scale, or a new event's
    holding = hold > dt / 2;
    target = holding ? 1 : scale, applied = std::max(target, 0.001f), inverse = 1 / applied;
}

// 010970F0: blade air motion. At higher rates the pivot displacement repeats every frame; ProbeMotionPhysics
// keeps its 59.94 Hz share.
__m128 PS4_SYSV_ABI ResolveBladeAirMotion(uintptr_t actor, uintptr_t context) {
    const auto animation = *reinterpret_cast<const uintptr_t*>(actor + 0x508);
    const float delta = *reinterpret_cast<const float*>(context + 0x148);
    const float weight = 1.f - delta * 59.94f;
    const bool correct = actor == MainActor() && (*reinterpret_cast<const u32*>(actor + 0x1c0) & 4) &&
        !(*reinterpret_cast<const u32*>(actor + 0xb0c) & 0x2000) && animation &&
        *reinterpret_cast<const u16*>(animation + 0x164) == 0x3c && delta > 0 && weight > 0.0001f &&
        *reinterpret_cast<const float*>(actor + 0x18c) == 0;
    AirFrame frame{actor, context, correct ? ActorCenterOffset(actor) : _mm_setzero_ps(), weight};
    const auto previous = air_frame;
    air_frame = correct ? &frame : nullptr;
    const auto result = original_air_motion(actor, context);
    air_frame = previous;
    return result;
}
// 0109FD20
__m128 PS4_SYSV_ABI ProbeMotionPhysics(uintptr_t actor, uintptr_t context) {
    if (air_frame && air_frame->actor == actor && air_frame->context == context && !air_frame->applied) {
        air_frame->applied = true;
        alignas(16) float direction[4];
        _mm_store_ps(direction, _mm_loadu_ps(reinterpret_cast<const float*>(actor + 0x90)));
        const float length = std::hypot(direction[0], direction[2]);
        if (std::isfinite(length) && length > 0.0001f) {
            // Air control is horizontal; yaw convergence must not retain the wall's pitch.
            const float x = direction[0] / length, z = direction[2] / length;
            _mm_storeu_ps(reinterpret_cast<float*>(actor + 0x70), _mm_setr_ps(z, 0, -x, 0));
            _mm_storeu_ps(reinterpret_cast<float*>(actor + 0x80), _mm_setr_ps(0, 1, 0, 0));
            _mm_storeu_ps(reinterpret_cast<float*>(actor + 0x90), _mm_setr_ps(x, 0, z, 0));
            const auto offset = ActorCenterOffset(actor);
            // Retain the 59.94 Hz pivot motion instead of repeating its full displacement every frame.
            const auto correction = _mm_mul_ps(_mm_sub_ps(air_frame->offset, offset), _mm_set1_ps(air_frame->weight));
            _mm_storeu_ps(reinterpret_cast<float*>(actor + 0xa0), _mm_add_ps(
                _mm_loadu_ps(reinterpret_cast<const float*>(actor + 0xa0)), correction));
        }
    }
    return original_motion_physics(actor, context);
}

using PrepareActorMotion = void(PS4_SYSV_ABI*)(uintptr_t, u32, uintptr_t);
using RebaseActor = void(PS4_SYSV_ABI*)(uintptr_t, const __m128*);
PrepareActorMotion original_prepare_actor_motion;
RebaseActor original_rebase_actor;
struct GrappleRelease { uintptr_t object; bool hanging, blending; };
struct GrappleMotionStep { uintptr_t actor; float share; };
thread_local GrappleRelease grapple_release{};
thread_local GrappleMotionStep* grapple_motion_step{};

// 010800F0 sees the change from grapple motion to collision-only motion before the actor becomes airborne.
void PS4_SYSV_ABI PrepareGrappleRelease(uintptr_t actor, u32 mode, uintptr_t context) {
    if (actor != MainActor()) {
        original_prepare_actor_motion(actor, mode, context);
        return;
    }
    const auto object = *reinterpret_cast<const uintptr_t*>(actor + 8);
    const auto state = *reinterpret_cast<const u16*>(actor + 0x1c0);
    const auto motion = *reinterpret_cast<const u32*>(context + 0x16c);
    const float blend = *reinterpret_cast<const float*>(actor + 0x1a8);
    const bool releasing = object && grapple_release.object == object && motion == 1 &&
        ((state == 0x4000 && grapple_release.hanging && blend < 1.f) ||
         ((state == 0x4000 || state == 8) && grapple_release.blending));
    // Apply once more when the blend ends. The object can still hold the last blended pose.
    grapple_release = {object, state == 0x4000 && motion == 0x17, releasing && blend < 1.f};
    const float share = *reinterpret_cast<const float*>(context + 0x148) * GameFrameRate(60);
    GrappleMotionStep step{actor, share};
    const auto previous = grapple_motion_step;
    grapple_motion_step = releasing && share > 0.f && share < 1.f ? &step : nullptr;
    original_prepare_actor_motion(actor, mode, context);
    grapple_motion_step = previous;
}

// 010E8C99 feeds the actor's own visual origin back into physics. Keep its 59.94 Hz displacement share.
void PS4_SYSV_ABI RebaseGrappleRelease(uintptr_t actor, const __m128* pose) {
    const auto* step = grapple_motion_step;
    if (step && actor == step->actor && Caller(__builtin_return_address(0)) == 0x010e8c9c) {
        const auto object = *reinterpret_cast<const uintptr_t*>(actor + 8);
        if (object && *reinterpret_cast<const s16*>(object + 0x80) == -1 &&
            !*reinterpret_cast<const uintptr_t*>(object + 0x28)) {
            const auto position = _mm_load_ps(reinterpret_cast<const float*>(actor + 0xa0));
            const auto delta = _mm_mul_ps(_mm_sub_ps(pose[3], position), _mm_set1_ps(step->share));
            const __m128 adjusted[]{pose[0], pose[1], pose[2], _mm_blend_ps(_mm_add_ps(position, delta), pose[3], 8)};
            original_rebase_actor(actor, adjusted);
            return;
        }
    }
    original_rebase_actor(actor, pose);
}

// 010597A0: hang contact. During a ledge corner (ResolveLedgeMotion) a rejected hang retries pushed off the wall.
u64 PS4_SYSV_ABI HangContact(__m128 start, __m128 second, u64 a, u64 b, u64 c, u64 d, u64 e, u64 f) {
    const auto caller = Caller(__builtin_return_address(0));
    auto result = original_hang_query(start, second, a, b, c, d, e, f);
    if (caller == 0x1098cc1 && result == 8 && (d & 0x30) && corner_motion && corner_motion->actor == a &&
        (*reinterpret_cast<const u32*>(a + 0x1c0) & 0x14) == 4 && *reinterpret_cast<const float*>(a + 0x124) < 0) {
        CollisionHit wall;
        std::memcpy(&wall, Guest<const void*>(0x466330), sizeof(wall));
        const auto normal = _mm_loadu_ps(reinterpret_cast<const float*>(a + 0x310));
        const auto position = _mm_loadu_ps(reinterpret_cast<const float*>(b));
        const float radius = *reinterpret_cast<const float*>(a + 0x178) * 0.5f;
        const float distance = Dot3(_mm_sub_ps(position, wall.point), normal);
        const auto bits = *reinterpret_cast<const u32*>(c) & *reinterpret_cast<const u32*>(c + 4) & 31;
        bool supported = (bits & (bits - 1)) && wall.node &&
            wall.surface == *reinterpret_cast<const uintptr_t*>(a + 0x330) &&
            Dot3(wall.normal, normal) > 0.995f && Dot3(normal, normal) > 0.995f && Dot3(normal, normal) < 1.005f &&
            std::abs(Dot3(normal, _mm_loadu_ps(reinterpret_cast<const float*>(a + 0x80)))) < 0.1f;
        const auto* contacts = *reinterpret_cast<const CollisionHit* const*>(c + 8);
        for (u32 i = 0; supported && i < 5; ++i) {
            if (!(bits & (1u << i))) continue;
            supported = contacts[i].node &&
                *reinterpret_cast<const uintptr_t*>(contacts[i].node + 0x28) == *reinterpret_cast<const uintptr_t*>(wall.node + 0x28) &&
                (ContactFlags(contacts[i].surface) & 1) &&
                Dot3(contacts[i].normal, _mm_loadu_ps(reinterpret_cast<const float*>(a + 0x80))) > 0.85f;
        }
        if (supported && distance >= 0 && distance < radius) {
            // The mode dispatcher discards this local proposal on failure; publish the offset after it returns.
            const auto offset = _mm_mul_ps(normal, _mm_set1_ps(radius - distance));
            const auto retry = original_ray_query(position, _mm_add_ps(position, offset), a, b, c, d, e, f);
            std::memcpy(Guest<void*>(0x466330), &wall, sizeof(wall));
            if (!retry) { corner_motion->offset = offset; corner_motion->valid = true; }
        }
    }
    return result;
}
// 0109AC00
u64 PS4_SYSV_ABI ResolveLedgeMotion(uintptr_t actor, u32 mode, u32 flags, uintptr_t context, __m128* position) {
    if (Caller(__builtin_return_address(0)) != 0x109d038 || mode != 5 || !position || actor != MainActor())
        return original_mode_query(actor, mode, flags, context, position);
    CornerMotion motion{actor, _mm_setzero_ps()};
    const auto previous = corner_motion;
    corner_motion = &motion;
    const auto result = original_mode_query(actor, mode, flags, context, position);
    corner_motion = previous;
    if (!result && motion.valid)
        _mm_storeu_ps(reinterpret_cast<float*>(position),
                      _mm_add_ps(_mm_loadu_ps(reinterpret_cast<const float*>(position)), motion.offset));
    return result;
}
// 01156D10: ledge support ray. This call site already accepted both lower wall contacts; bridge only a small,
// continuous seam.
u32 PS4_SYSV_ABI QueryLedgeSupport(__m128 start, __m128 end, u64 a, u64 b, u64 c, u64 d, u64 e, u64 f) {
    auto result = original_ray_query(start, end, a, b, c, d, e, f);
    if (result || Caller(__builtin_return_address(0)) != 0x105bfa7) return result;
    auto actor = MainActor();
    if (!actor || *Guest<const uintptr_t*>(0x466300) != *reinterpret_cast<const uintptr_t*>(actor + 8) ||
        !(*reinterpret_cast<const u32*>(actor + 0x1c0) & 0x10) || !(*reinterpret_cast<const u32*>(actor + 0x1c4) & 0x200))
        return result;
    const float span = *reinterpret_cast<const float*>(actor + 0x170) / 60.f;
    const auto offset = _mm_mul_ps(_mm_loadu_ps(reinterpret_cast<const float*>(actor + 0x70)), _mm_set1_ps(span));
    CollisionHit original, first, second;
    std::memcpy(&original, Guest<const void*>(0x466330), sizeof(original));
    const u32 left = original_ray_query(_mm_add_ps(start, offset), _mm_add_ps(end, offset), a, b, c, d, e, f);
    std::memcpy(&first, Guest<const void*>(0x466330), sizeof(first));
    const u32 right = original_ray_query(_mm_sub_ps(start, offset), _mm_sub_ps(end, offset), a, b, c, d, e, f);
    std::memcpy(&second, Guest<const void*>(0x466330), sizeof(second));
    const auto wall_normal = _mm_loadu_ps(reinterpret_cast<const float*>(actor + 0x310));
    const auto residual = _mm_sub_ps(_mm_sub_ps(first.point, second.point), _mm_add_ps(offset, offset));
    const bool bridge = left && right && first.node && second.node &&
        *reinterpret_cast<const uintptr_t*>(first.node + 0x28) == *reinterpret_cast<const uintptr_t*>(second.node + 0x28) &&
        (ContactFlags(first.surface) & 0x80218) && (ContactFlags(second.surface) & 0x80218) &&
        Dot3(first.normal, wall_normal) > 0.995f && Dot3(second.normal, wall_normal) > 0.995f &&
        Dot3(first.normal, second.normal) > 0.995f && Dot3(residual, residual) <= 4.f * span * span;
    if (bridge) {
        first.point = _mm_mul_ps(_mm_add_ps(first.point, second.point), _mm_set1_ps(0.5f));
        first.normal = _mm_add_ps(first.normal, second.normal);
        first.normal = _mm_div_ps(first.normal, _mm_sqrt_ps(_mm_set1_ps(Dot3(first.normal, first.normal))));
        first.distance = (first.distance + second.distance) * 0.5f;
        std::memcpy(Guest<void*>(0x466330), &first, sizeof(first));
        return left;
    }
    std::memcpy(Guest<void*>(0x466330), &original, sizeof(original));
    return result;
}

// 011B6E60: the frame's light array (DAT_025c4810: base +0x180, count +0x174; 0x50 bytes per light, color and intensity at
// +0x20, the light object at +0x40), rebuilt from the light objects 60 times a second; every mesh copies its lights from
// it (FUN_012135e0). Some lights flicker by animation clip, jumping between dark and bright (the portal's and Pandora's
// torch's blue light, ~40% RMS around their average: harsh on the eyes). A light seen turning around 3+ times within
// ~0.6 s with a 20%+ swing is marked for good: its swing around its 2 s average is capped to 25% RMS and every rise and
// fall eased over ~100 ms. Gentle flicker and the flashes and fades of other lights pass unchanged.
using LightArrayFn = void(PS4_SYSV_ABI*)(uintptr_t);
LightArrayFn original_light_array;
void PS4_SYSV_ABI BuildLightArray(uintptr_t lights) {
    original_light_array(lights);
    struct Light { float mean[4], mid[4], out[4], level, average, var, turns; s32 trend; u32 seen; bool harsh; };
    static std::unordered_map<uintptr_t, Light> state;
    static std::mutex lock;
    static u32 pass;
    static auto last = std::chrono::steady_clock::now();
    std::scoped_lock hold(lock);
    const auto now = std::chrono::steady_clock::now();
    const float dt = std::min(std::chrono::duration<float>(now - last).count(), 0.1f);  // a pause or a load: as one step
    last = now, ++pass;
    const auto rate = [dt](float tau) { return 1 - std::exp(-dt / tau); };
    // The swing is measured over ~3 s, so a calm stretch doesn't let the next jump through. The ease: two low-passes in a
    // row (S-shaped, no sudden start), 10-90% in 3.36 time constants.
    const float slow = rate(2), spread = rate(3), ease = rate(0.1f / 3.36f), decay = std::exp(-dt / 0.6f);
    auto* record = *reinterpret_cast<u8**>(lights + 0x180);
    for (u32 i = 0, n = *reinterpret_cast<const u32*>(lights + 0x174); i < n; ++i, record += 0x50) {
        auto* color = reinterpret_cast<float*>(record + 0x20);
        const float level = color[3] * (color[0] + color[1] + color[2]);
        auto [it, fresh] = state.try_emplace(*reinterpret_cast<const uintptr_t*>(record + 0x40));
        Light& s = it->second;
        if (fresh) {
            for (float* v : {s.mean, s.mid, s.out}) std::copy_n(color, 4, v);
            s.level = s.average = level, s.seen = pass;
            continue;
        }
        // Turns of the intensity (a rise after a fall or the reverse), fading over ~0.6 s: a flicker keeps several.
        const float d = level - s.level;
        s.level = level, s.seen = pass, s.turns *= decay;
        if (const s32 trend = std::abs(d) > 1e-3f * std::max(std::abs(level), 1e-3f) ? (d > 0 ? 1 : -1) : 0) {
            if (s.trend && trend != s.trend) s.turns += 1;
            s.trend = trend;
        }
        s.average += (level - s.average) * slow;
        s.var += ((level - s.average) * (level - s.average) - s.var) * spread;
        const float swing = std::sqrt(s.var) / std::max(std::abs(s.average), 1e-6f);
        s.harsh |= s.turns >= 3 && swing > 0.2f;  // marked for good: its next jump eased too
        const float k = s.harsh ? ease : 1, gain = s.harsh ? std::min(0.25f / std::max(swing, 1e-6f), 1.f) : 1;
        for (int c = 0; c < 4; ++c) {
            s.mean[c] += (color[c] - s.mean[c]) * slow;
            s.mid[c] += (s.mean[c] + (color[c] - s.mean[c]) * gain - s.mid[c]) * k, color[c] = s.out[c] += (s.mid[c] - s.out[c]) * k;
        }
    }
    // Lights gone for ~1 s forgotten; marked ones after ~1 min, so a flickering light out of view stays marked.
    if (pass % 1024 == 0) std::erase_if(state, [](const auto& e) { return e.second.seen + (e.second.harsh ? 4096 : 64) < pass; });
}

// Rendering diagnostic hooks.
// TEMP shadow trace (GOW3_SHADOW=1): texture lifetime and level loading without touching the frame. Hooks append
// 64-byte records to a lock-free ring; a separate thread drains it to _rehooked_/shadow-<time>.bin every 5 ms
// and adds a tick every 100 ms (engine frame counter, section loading flag and depth).
namespace Shadow {
enum Kind : u32 { View = 1, Rebase, Release, GpuMove, SectionOpen, WadOpen, Tick };
struct Record {
    u64 time_ns;
    u32 kind, x;
    u64 a, b;
    u32 data[8];  // View: final T#; SectionOpen/WadOpen: name (truncated, NUL-padded)
};
static_assert(sizeof(Record) == 64);
constexpr u64 kSlots = 1 << 16;
struct Slot { std::atomic<u64> seq; Record r; };
Slot* ring;
std::atomic<u64> head, lost;
const auto start = std::chrono::steady_clock::now();
using ViewFn = u64(PS4_SYSV_ABI*)(uintptr_t, uintptr_t);
using RebaseFn = u64(PS4_SYSV_ABI*)(uintptr_t, u32);
using ReleaseFn = u64(PS4_SYSV_ABI*)(uintptr_t*);
using MoveFn = u64(PS4_SYSV_ABI*)(u64, u64, u32);
using SectionFn = u64(PS4_SYSV_ABI*)(u32, u32, const char*);
ViewFn original_view;
RebaseFn original_rebase;
ReleaseFn original_release;
MoveFn original_move;
SectionFn original_section;

void Push(Kind kind, u32 x, u64 a, u64 b, const void* data = nullptr, size_t bytes = 0) {
    const u64 i = head.fetch_add(1, std::memory_order_relaxed);
    Slot& s = ring[i & (kSlots - 1)];
    s.r = {u64((std::chrono::steady_clock::now() - start).count()), kind, x, a, b, {}};
    if (data) std::memcpy(s.r.data, data, std::min(bytes, sizeof(s.r.data)));
    s.seq.store(i + 1, std::memory_order_release);
}
void PushName(Kind kind, u32 x, u64 a, u64 b, const char* name) {
    char text[32]{};
    const size_t n = name ? strnlen(name, 256) : 0;
    if (n) std::memcpy(text, name + (n > 31 ? n - 31 : 0), std::min<size_t>(n, 31));  // keep the tail: the file name
    Push(kind, x, a, b, text, sizeof(text));
}

// 011C7D70(node, texture): builds the final T# at node+0x20 (and its sampler after it).
u64 PS4_SYSV_ABI BuildView(uintptr_t node, uintptr_t texture) {
    const u64 result = original_view(node, texture);
    Push(View, 0, node, texture, reinterpret_cast<const void*>(node + 0x20), 32);
    return result;
}
// 011C8280(block, delta): compaction moves every view of the block; base at block+0x20.
u64 PS4_SYSV_ABI RebaseBlock(uintptr_t block, u32 delta) {
    Push(Rebase, delta, block, *reinterpret_cast<const u64*>(block + 0x20));
    return original_rebase(block, delta);
}
// 011C8D80(&texture.block): drops one reference, frees the backing at zero.
u64 PS4_SYSV_ABI ReleaseBlock(uintptr_t* field) {
    const uintptr_t block = *field;
    Push(Release, block ? *reinterpret_cast<const u32*>(block + 0x2c) : 0, block, block ? *reinterpret_cast<const u64*>(block + 0x20) : 0);
    return original_release(field);
}
// 011F1540(dst, src, bytes): overlap-safe GPU memmove of the texture heap compaction.
u64 PS4_SYSV_ABI GpuMemmove(u64 dst, u64 src, u32 bytes) {
    Push(GpuMove, bytes, dst, src);
    return original_move(dst, src, bytes);
}
// 011F5B90(a, b, name): opens a level memory section (asset type 20).
u64 PS4_SYSV_ABI OpenSection(u32 a, u32 b, const char* name) {
    const u64 section = original_section(a, b, name);
    PushName(SectionOpen, a, b, section, name);
    return section;
}

std::filesystem::path file_path;

void Drain(const std::stop_token& token) {
#ifdef _WIN32
    FILE* f = _wfopen(file_path.c_str(), L"wb");
#else
    FILE* f = std::fopen(file_path.c_str(), "wb");
#endif
    if (!f) return;
    std::fwrite("GOW3SHD1", 1, 8, f);
    u64 tail = 0, next_tick = 0;
    for (;;) {
        const bool stop = token.stop_requested();
        const u64 now = u64((std::chrono::steady_clock::now() - start).count());
        if (now >= next_tick) {
            Push(Tick, *Guest<const u32*>(0x1a7d754), *Guest<const u32*>(0x1a780a4), *Guest<const s8*>(0x3cdd60));
            next_tick = now + 100'000'000;
        }
        for (;;) {
            Slot& s = ring[tail & (kSlots - 1)];
            const u64 seq = s.seq.load(std::memory_order_acquire);
            if (seq == tail + 1) { std::fwrite(&s.r, sizeof(Record), 1, f); ++tail; continue; }
            if (seq > tail + 1) { lost += seq - 1 - tail; tail = seq - 1; continue; }  // lapped: skip to the oldest kept
            break;
        }
        std::fflush(f);
        if (stop) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::fclose(f);
}
std::jthread drain_thread;
} // namespace Shadow

// TEMP draw shadow (GOW3_DRAWLOG=1), plan phase 1: per draw/dispatch, what the draw reader would read (ctx fields,
// program objects, stage meta) plus each ctx's user data and a few context registers applied from the PM4 that ctx
// wrote since its chain head. `framecap drawcheck` compares it with the GPU-order replay of a simultaneous recording.
namespace DrawShadow {
enum Kind : u32 { DrawIndex = 1, DrawAuto, Dispatch, Submit, Lost };
struct Record {
    u64 pkt, ctx;              // Submit: head before, head after
    u32 kind, tid, dirty, resync;  // resync: 1 stream lost (restarted at the current segment), 2 packet not found
    u64 isa[2];                // PS, VS from the ctx program objects
    u64 pgm[2];                // PS, VS (CS: VS slot) PGM_LO/HI from the ctx's own PM4
    u32 user[2][16];           // PS, VS (CS: VS slot) user data from the ctx's own PM4
    u32 set[2], used[2];       // user data written by the ctx / referenced by the stage meta
    u32 field[8];              // ctx +0x48 blend, +0x4C raster, +0x54 depth, +0x58/+0x5C stencil ref, +0x60, +0x68, +0x6C
    u32 reg[12], reg_set;      // kRegs from the ctx's own SET_CONTEXT_REG
    u32 pad[7];
};
static_assert(sizeof(Record) == 320);
constexpr u32 kRegs[12]{0x08e, 0x08f, 0x200, 0x202, 0x203, 0x205, 0x1e0, 0x10b, 0x10c, 0x10d, 0x318, 0x012};
constexpr u64 kSlots = 1 << 15, kMainCtx = 0x1a78b58;
struct Slot { std::atomic<u64> seq; Record r; };
Slot* ring;
std::atomic<u64> head, lost;
using DrawIndexFn = void(PS4_SYSV_ABI*)(uintptr_t, u32, u64);
using DrawAutoFn = void(PS4_SYSV_ABI*)(uintptr_t, u32);
using DispatchFn = void(PS4_SYSV_ABI*)(uintptr_t, u32, u32, u32);
using SubmitFn = u64(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64);
DrawIndexFn original_draw_index;
DrawAutoFn original_draw_auto;
DispatchFn original_dispatch;
SubmitFn original_submit;

void Push(const Record& r) {
    const u64 i = head.fetch_add(1, std::memory_order_relaxed);
    Slot& s = ring[i & (kSlots - 1)];
    s.r = r;
    s.seq.store(i + 1, std::memory_order_release);
}

// One ctx's own packet stream: segments are closed by 012122D0 as [u32 dwords to the link][packets][u64 next].
struct Stream {
    uintptr_t ctx;
    u64 head;
    const u32 *seg, *cursor;
    u32 sh[0x260], reg[12], reg_set;
    u64 sh_set[0x260 / 64];
};
thread_local Stream streams[4];
thread_local u32 victim;

bool InRing(const u32* p) {
    const u64 base = *Guest<const u64*>(kMainCtx + 0x528), size = *Guest<const u64*>(kMainCtx + 0x530);
    return u64(p) >= base && u64(p) < base + size;
}
void Apply(Stream& s, const u32* p, const u32* end) {
    while (p < end) {
        const u32 h = *p;
        if (h >> 30 != 3) { ++p; continue; }
        const u32 n = (h >> 16 & 0x3fff) + 1, op = h >> 8 & 0xff;
        if (p + 1 + n > end) break;
        const u32* b = p + 1;
        for (u32 k = 1; k < n; ++k) {
            const u32 r = (b[0] & 0xffff) + k - 1;
            if (op == 0x76 && r < 0x260) s.sh[r] = b[k], s.sh_set[r >> 6] |= 1ull << (r & 63);
            if (op == 0x69)
                for (u32 i = 0; i < 12; ++i)
                    if (kRegs[i] == r) s.reg[i] = b[k], s.reg_set |= 1u << i;
        }
        p += n + 1;
    }
}
Stream& Find(uintptr_t ctx) {
    const u64 chain = *reinterpret_cast<const u64*>(ctx + 0x40);
    Stream* s = nullptr;
    for (auto& t : streams)
        if (t.ctx == ctx) s = &t;
    if (!s) s = &streams[victim++ & 3];
    if (s->ctx != ctx || s->head != chain) {
        std::memset(s, 0, sizeof(*s));
        s->ctx = ctx, s->head = chain;
        s->seg = reinterpret_cast<const u32*>(chain), s->cursor = s->seg + 1;
    }
    return *s;
}
// Applies the ctx's packets from the cursor to `to`, following the closed segments in between.
u32 Advance(Stream& s, uintptr_t ctx, const u32* to) {
    const auto* seg = *reinterpret_cast<const u32* const*>(ctx);
    for (u32 guard = 0; s.seg != seg; ++guard) {
        if (guard == 4096 || !InRing(s.seg) || !*s.seg) return s.seg = seg, s.cursor = to, 1;
        const u32* end = s.seg + *s.seg;
        Apply(s, s.cursor, end);
        u64 next;
        std::memcpy(&next, end, 8);
        s.seg = reinterpret_cast<const u32*>(next), s.cursor = s.seg + 1;
    }
    Apply(s, s.cursor, to);
    return 0;
}
// User data registers the stage meta references: table pointer pairs, direct data, inline T#/S#/V# selectors.
u32 Used(const u8* stage) {
    u32 m = 0;
    for (u32 o : {0x0f, 0x10, 0x11, 0x12, 0x13, 0x15, 0x16, 0x17})
        if (stage[o] < 16) m |= 3u << stage[o];
    if (stage[0x1b] < 16) m |= ((1u << std::min<u32>(stage[0x1c], 16)) - 1) << stage[0x1b];
    for (auto [o, n] : {std::pair{0x2a, 8}, {0x6a, 8}, {0xaa, 4}, {0xea, 4}, {0x12a, 4}})
        for (u32 i = 0; i < 32; ++i) {
            const u16 sel = *reinterpret_cast<const u16*>(stage + o + 2 * i);
            if (sel != 0xffff && sel & 0x8000) m |= ((1u << n) - 1) << (sel & 0xf);
        }
    return m & 0xffff;
}
u64 Isa(uintptr_t program) {
    return program ? (u64(*reinterpret_cast<const u32*>(program + 8)) | u64(*reinterpret_cast<const u32*>(program + 0xc)) << 32) << 8 : 0;
}
// After the packet was written: `back` dwords before the cursor (packet + trailing NOP/event).
void Log(uintptr_t ctx, Kind kind, u32 back, u32 op) {
    const auto* cmd = *reinterpret_cast<const u32* const*>(ctx + 0x10);
    const u32* pkt = cmd - back;
    Record r{};
    if ((*pkt >> 8 & 0xff) != op) pkt = cmd, r.resync = 2;
    Stream& s = Find(ctx);
    r.resync |= Advance(s, ctx, pkt);
    s.cursor = cmd;
    r.pkt = u64(pkt), r.ctx = ctx, r.kind = kind, r.tid = Gow3::ThreadId(), r.dirty = *reinterpret_cast<const u32*>(ctx + 0x64);
    const bool compute = kind == Dispatch;
    for (u32 i = 0; i < 2; ++i) {
        const u32 user = compute ? 0x240 : i ? 0x4c : 0x0c, pgm = compute ? 0x20c : i ? 0x48 : 0x08;
        if (compute && !i) continue;
        std::memcpy(r.user[i], s.sh + user, 64);
        for (u32 k = 0; k < 16; ++k) r.set[i] |= u32(s.sh_set[(user + k) >> 6] >> ((user + k) & 63) & 1) << k;
        r.pgm[i] = (u64(s.sh[pgm]) | u64(s.sh[pgm + 1]) << 32) << 8;
        const auto* stage = reinterpret_cast<const u8*>(ctx + (i ? 0x260 : 0x98));
        if (!compute) r.isa[i] = Isa(*reinterpret_cast<const uintptr_t*>(stage)), r.used[i] = Used(stage);
    }
    for (u32 i = 0; auto o : {0x48, 0x4c, 0x54, 0x58, 0x5c, 0x60, 0x68, 0x6c}) r.field[i++] = *reinterpret_cast<const u32*>(ctx + o);
    std::memcpy(r.reg, s.reg, sizeof(r.reg)), r.reg_set = s.reg_set;
    Push(r);
}
// 0123A230(ctx, count, index address): DRAW_INDEX_2 + NOP/event (8 dwords).
void PS4_SYSV_ABI DrawIndex2(uintptr_t ctx, u32 count, u64 indices) {
    original_draw_index(ctx, count, indices);
    Log(ctx, DrawIndex, 8, 0x27);
}
// 0123A170(ctx, count): DRAW_INDEX_AUTO + NOP/event (5 dwords).
void PS4_SYSV_ABI DrawIndexAuto(uintptr_t ctx, u32 count) {
    original_draw_auto(ctx, count);
    Log(ctx, DrawAuto, 5, 0x2d);
}
// 0123A380(ctx, x, y, z): DISPATCH_DIRECT + NOP/event (7 dwords).
void PS4_SYSV_ABI DispatchDirect(uintptr_t ctx, u32 x, u32 y, u32 z) {
    original_dispatch(ctx, x, y, z);
    Log(ctx, Dispatch, 7, 0x15);
}
// 011A2AA0: frame submit; the main ctx chain head before/after keys the recording's frames.
u64 PS4_SYSV_ABI SubmitFrame(u64 a, u64 b, u64 c, u64 d, u64 e, u64 f) {
    const u64 before = *Guest<const u64*>(kMainCtx + 0x40);
    const u64 result = original_submit(a, b, c, d, e, f);
    Record r{};
    r.kind = Submit, r.pkt = before, r.ctx = *Guest<const u64*>(kMainCtx + 0x40), r.tid = Gow3::ThreadId();
    Push(r);
    return result;
}

std::filesystem::path file_path;

void Drain(const std::stop_token& token) {
#ifdef _WIN32
    FILE* f = _wfopen(file_path.c_str(), L"wb");
#else
    FILE* f = std::fopen(file_path.c_str(), "wb");
#endif
    if (!f) return;
    std::fwrite("GOW3DRW1", 1, 8, f);
    u64 tail = 0;
    for (;;) {
        const bool stop = token.stop_requested();
        for (;;) {
            Slot& s = ring[tail & (kSlots - 1)];
            const u64 seq = s.seq.load(std::memory_order_acquire);
            if (seq == tail + 1) { std::fwrite(&s.r, sizeof(Record), 1, f); ++tail; continue; }
            if (seq > tail + 1) { lost += seq - 1 - tail; tail = seq - 1; continue; }
            break;
        }
        std::fflush(f);
        if (stop) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    Record r{};
    r.kind = Lost, r.pkt = lost;
    std::fwrite(&r, sizeof(r), 1, f);
    std::fclose(f);
}
std::jthread drain_thread;
} // namespace DrawShadow

// Title menu hooks.
// Title menu: entries added to the `mainShell` script of SHELL.WAD in memory (research/menu-inicial-script.md). The script is a state
// machine run by a bytecode VM. The title menu and Rehooked submenu reuse the ten-row UIVideosSubmenuAG list group.
namespace TitleMenu {
constexpr u32 kQuitTarget = 0xffffff00, kBackTarget = 0xfffffe00, kPageNextTarget = 0xfffffd00, kHelpTarget = 0xfffffc00, kSettingTarget = 0xfffe0000,
              kMoveTarget = 0xfffd0000, kFirstItemTarget = 0xfffffb00, kBeforeNewGameTarget = 0xfffffa00, kContinueTarget = 0xfffff900,
              kRefreshContinueTarget = 0xfffff800;
constexpr u32 kMainShellSize = 0xc1e0, kRows = 10, kRowSize = 192, kLabelSize = 96;
constexpr u32 kResidentSize = 0x132fe, kBrightnessState = 0x193c;
constexpr u32 kTurnOnState = 0xe4, kMainState = 0xf8, kNewGameState = 0x114, kLoadState = 0x148, kTreasuresState = 0x1c8;
// Load: X requests the load; 0x14b comes with 1 when it starts and 0 when it ends (the body, from the jz on, runs only for 0);
// 0x149 when the loaded game starts.
constexpr u32 kLoadHandler = 0xe6b, kLoadClosedHandler = 0xe9a, kLoadClosedJump = 0xea8, kLoadClosedEnd = 0xece, kGameStartHandler = 0xecf;
constexpr u32 kMainItemSize = 0x34, kRehookedState = kMainShellSize - 4, kHelpState = kRehookedState + kMainItemSize,
              kQuitState = kHelpState + kMainItemSize, kContinueState = kQuitState + kMainItemSize,
              kContinueLoadState = kContinueState + kMainItemSize,
              kSettingsState = kContinueLoadState + 0x40, kFirstRow = kSettingsState + 0x1c;
enum MainRow : u32 { ContinueRow, NewGameRow, LoadRow, OptionsRow, TreasuresRow, RehookedRow, HelpRow, QuitRow, kMainRows };
constexpr const char* kMainLabels[kMainRows] = {
    "Continue", "[msg_A[8001]]", "[msg_A[8002]]", "[msg_A[8003]]", "[msg_A[8004]]", "Rehooked", "Controls and Extras", "[msg_A[4012]]",
};
constexpr char kSelectClip[] = "menuItem0Select";
// String pool offsets (from the code base) already in the script.
constexpr u32 kMainGroup = 0x8b5d, kVideosGroup = 0x8bfc, kHelpSelect = 0x973c, kHelpText = 0x974a; // the button help line: Select / Select + Back
// The game's rows are single strings with proportional text and scale as a whole on focus.
constexpr float kSpacePixels = 11.6f, kColumnPixels = 450.f, kMaxValueWidth = 344.f;
using InitFn = void(PS4_SYSV_ABI*)(u8*);
using GotoFn = void(PS4_SYSV_ABI*)(u8*, u32);
InitFn original_init;
GotoFn original_goto;
std::vector<u8> patched;
std::vector<u8> patched_options;
u32 main_labels_at{}, main_clips_at{}, main_initial_target{}, options_label_at{};
bool continue_available{};
UI::GameLanguage menu_language{};

enum Option : u32 { Resolution, Fps, DisplayModeOption, Vrr, Vsync, Reflex, Msaa, Filtering, PostEffects, Scaler,
                    SkipIntro, SkipVideos, Cinematics4K, Mechanism, Recompilation, LazyAot, OptionCount, Empty = OptionCount };
struct Setting {
    std::string_view label;
    std::span<const std::string_view> values;
    u32 index;
    float width;
};
constexpr std::string_view resolutions[]{"480p", "720p", "1080p", "1440p", "1800p", "4K"}, frame_rates[]{"30 FPS", "40 FPS", "60 FPS", "120 FPS", "240 FPS", "480 FPS"},
    display_modes[]{"Windowed", "Borderless", "Exclusive"}, vsync_modes[]{"FIFO", "Mailbox", "Off"}, on_off[]{"Off", "On"},
    msaa[]{"Off", "2x", "4x"}, filtering[]{"Trilinear", "4x", "8x", "16x"}, anti_aliasing[]{"Off", "CMAA2", "Fast Adaptive AA"},
    scalers[]{"Bilinear / Bilinear", "FSR 1 / Lanczos 2"},
    compilation[]{"Sync", "Async"}, mechanisms[]{"JIT", "AOT"};
Setting settings[OptionCount]{
    {"Resolution", resolutions, 2, 201}, {"Frame Rate", frame_rates, 2, 206}, {"Display Mode", display_modes, 1, 237},
    {"VRR", on_off, 0, 80}, {"VSync", vsync_modes, 0, 106}, {"NVIDIA Reflex", on_off, 0, 260},
    {"MSAA", msaa, 1, 106}, {"Texture Filtering", filtering, 0, 319}, {"Post Effects", anti_aliasing, 0, 235},
    {"Scaler", scalers, 0, 116},
    {"Skip Opening Logos", on_off, 0, 350}, {"Skip Videos with X", on_off, 0, 330}, {"4K Cinematics", on_off, 0, 260},
    {"Shader Mechanism", mechanisms, 0, 338}, {"Shader Recompilation", compilation, 0, 400}, {"Lazy AOT", on_off, 0, 175}
};
constexpr Option pages[][kRows - 1]{
    {Resolution, Fps, DisplayModeOption, Vrr, Vsync, Reflex, Empty, Empty, Empty},
    {Msaa, Filtering, PostEffects, Scaler, Empty, Empty, Empty, Empty, Empty},
    {SkipIntro, SkipVideos, Mechanism, Recompilation, LazyAot, Empty, Empty, Empty, Empty}
};
constexpr std::string_view page_names[]{"Display", "Graphics", "Other"};
constexpr float page_widths[]{198.f, 234.f, 181.f};
constexpr u32 kPageCount = std::size(pages);
u32 row_at[kRows], page{};
std::array<u32, OptionCount> initial_indices{};
Config initial_config;

bool Selectable(u32 row) {
    if (row == kRows - 1) return true;
    const Option option = pages[page][row];
    return option != Empty && (option != LazyAot || settings[Mechanism].index == u32(ShaderMechanism::Aot)) &&
           (option != Vrr || settings[DisplayModeOption].index != u32(Gow3::DisplayMode::Windowed)) &&
           (option != Reflex || Vulkan::Get().reflex);
}
float ValueWidth(std::string_view value) {
    constexpr std::pair<std::string_view, float> widths[]{
        {"480p", 84}, {"720p", 81}, {"1080p", 99}, {"1440p", 98}, {"1800p", 99}, {"4K", 48},
        {"30 FPS", 120}, {"40 FPS", 121}, {"60 FPS", 120}, {"120 FPS", 133}, {"240 FPS", 141}, {"480 FPS", 142},  // digits + an estimated 77 for " FPS"
        {"Windowed", 196}, {"Borderless", 209}, {"Exclusive", 180}, {"FIFO", 84}, {"Mailbox", 153},
        {"Off", 67}, {"On", 57}, {"2x", 42}, {"4x", 43}, {"8x", 43}, {"16x", 57},
        {"Trilinear", 178}, {"CMAA2", 129}, {"Fast Adaptive AA", 324}, {"Bilinear / Bilinear", 344}, {"FSR 1 / Lanczos 2", 305},
        {"Sync", 92}, {"Async", 114}, {"JIT", 50}, {"AOT", 77},
    };
    const auto it = std::ranges::find(widths, value, &std::pair<std::string_view, float>::first);
    return it != std::end(widths) ? it->second : kMaxValueWidth;
}
float LocalizedWidth(std::string_view localized, std::string_view source, float width) {
    if (localized == source) return width;
    float result{};
    for (unsigned char c : localized) {
        if (c < 0x80) result += c == ' ' ? kSpacePixels : 19.f;
        else if (c >= 0xe0) result += 31.f;
        else if (c >= 0xc0) result += 20.f;
    }
    return result;
}

void WriteText(u8* at, size_t capacity, std::string_view text) {
    size_t length = std::min(text.size(), capacity - 1);
    while (length && length < text.size() && (static_cast<unsigned char>(text[length]) & 0xc0) == 0x80) --length;
    std::memcpy(at, text.data(), length);
    std::memset(at + length, 0, capacity - length);
}

void WriteRow(u32 row) {
    std::string text;
    if (row == kRows - 1) {
        constexpr float page_center = 210.f, arrow_with_space = 45.f;
        const float left_arrow = page ? arrow_with_space : 0.f;
        const std::string_view page_name = UI::Tr(page_names[page].data());
        const float width = LocalizedWidth(page_name, page_names[page], page_widths[page]);
        const int padding = std::max(0, int(std::lround((page_center - width * 0.5f - left_arrow) / kSpacePixels)));
        text = std::format("{:{}}{}{} {}/{}{}", "", padding, page ? "[ArrowSelectionL] " : "", page_name, page + 1, kPageCount,
                           page + 1 < kPageCount ? " [ArrowSelectionR]" : "");
    } else if (const Option option = pages[page][row]; option != Empty) {
        const Setting& setting = settings[option];
        const std::string_view label = UI::Tr(setting.label.data());
        if (!Selectable(row)) {
            const char* reason = option == Reflex ? "Unavailable" : option == Vrr ? "Full Screen only" : "AOT only";
            text = std::format("{} ({})", label, UI::Tr(reason));
        }
        else {
            const auto value = setting.values[setting.index];
            const std::string_view localized = UI::Tr(value.data());
            const float offset = (kMaxValueWidth - LocalizedWidth(localized, value, ValueWidth(value))) * 0.5f;
            const float width = LocalizedWidth(label, setting.label, setting.width);
            text = std::format("{}{:{}}[ArrowSelectionL] {} [ArrowSelectionR]", label, "",
                std::max(1, int(std::lround((kColumnPixels + offset - width) / kSpacePixels))), localized);
        }
    }
    WriteText(&patched[row_at[row]], kRowSize, text);
}
void WriteRows() { for (u32 row = 0; row < kRows; ++row) WriteRow(row); }
Config SelectedConfig() {
    Config config = initial_config;
    config.resolution = settings[Resolution].index;
    config.fps = FrameRates[settings[Fps].index].nominal;
    config.display_mode = Gow3::DisplayMode(settings[DisplayModeOption].index);
    config.vrr = settings[Vrr].index;
    config.vsync = VSync(settings[Vsync].index);
    config.nvidia_reflex = settings[Reflex].index;
    config.msaa = MsaaLevels[settings[Msaa].index];
    config.anisotropy = AnisotropyLevels[settings[Filtering].index];
    config.anti_aliasing = AntiAliasing(settings[PostEffects].index);
    config.scaling_filter = ScalingFilter(settings[Scaler].index);
    config.skip_intro = settings[SkipIntro].index;
    config.skip_videos = settings[SkipVideos].index;
    config.cinematics_4k = settings[Cinematics4K].index;
    config.shader_recompilation = ShaderRecompilation(settings[Recompilation].index);
    config.shader_mechanism = ShaderMechanism(settings[Mechanism].index);
    config.lazy_aot = settings[LazyAot].index;
    Extras::WriteToConfig(config);
    return config;
}
void Configure(const Config& config) {
    initial_config = config;
    settings[Resolution].index = config.resolution;
    settings[Fps].index = u32(std::ranges::find(FrameRates, config.fps, &FrameRate::nominal) - FrameRates.begin());
    settings[DisplayModeOption].index = u32(config.display_mode);
    settings[Vrr].index = config.vrr;
    settings[Vsync].index = u32(config.vsync);
    settings[Reflex].index = config.nvidia_reflex;
    settings[Msaa].index = u32(std::ranges::find(MsaaLevels, config.msaa) - MsaaLevels.begin());
    settings[Filtering].index = u32(std::ranges::find(AnisotropyLevels, config.anisotropy) - AnisotropyLevels.begin());
    settings[PostEffects].index = u32(config.anti_aliasing);
    settings[Scaler].index = u32(config.scaling_filter);
    settings[SkipIntro].index = config.skip_intro;
    settings[SkipVideos].index = config.skip_videos;
    settings[Cinematics4K].index = config.cinematics_4k;
    settings[Recompilation].index = u32(config.shader_recompilation);
    settings[Mechanism].index = u32(config.shader_mechanism);
    settings[LazyAot].index = config.lazy_aot;
    for (u32 i = 0; i < OptionCount; ++i) initial_indices[i] = settings[i].index;
}

[[gnu::always_inline]] inline u32 SwapBytes(u32 value) {
#ifdef _WIN32
    return _byteswap_ulong(value);
#else
    return std::byteswap(value);
#endif
}
u32 Be32(const u8* p) {
    u32 v;
    std::memcpy(&v, p, sizeof(v));
    return SwapBytes(v);
}
void PutBe32(u8* p, u32 v) {
    v = SwapBytes(v);
    std::memcpy(p, &v, sizeof(v));
}

void WriteMainRows(bool available) {
    continue_available = available;
    const u32 skip = available ? 0 : 1;
    auto* const labels = patched.data() + main_labels_at;
    for (u32 i = 0; i < kMainRows; ++i) {
        const u32 item = i + skip;
        const char* label = item < kMainRows ? UI::Tr(kMainLabels[item]) : "";
        WriteText(labels + i * kLabelSize, kLabelSize, label);
    }
    for (u32 i = NewGameRow; i < kMainRows; ++i)
        patched[main_clips_at + i * sizeof(kSelectClip) + sizeof("menuItem") - 1] = u8('0' + i - skip);
    PutBe32(patched.data() + main_initial_target, available ? kContinueState : kNewGameState);
}

void RefreshLanguage() {
    const auto current = UI::TextLanguage();
    if (current == menu_language) return;
    menu_language = current;
    if (!patched.empty()) {
        WriteMainRows(continue_available);
        WriteRows();
    }
    if (!patched_options.empty())
        WriteText(patched_options.data() + options_label_at, kLabelSize, UI::Tr("Controls and Extras"));
}

bool RefreshContinue(u8* script, bool in_main_menu) {
    const bool available = Libraries::SaveData::Dialog::HasContinueSave();
    if (available == continue_available) return false;
    // Exit with the old clip names before assigning the rows their new positions.
    if (in_main_menu) original_goto(script, kTurnOnState);
    WriteMainRows(available);
    if (in_main_menu) original_goto(script, kMainState);
    return in_main_menu;
}

struct Code {
    std::vector<u8> bytes;
    Code& Raw(std::initializer_list<u8> v) { bytes.insert(bytes.end(), v); return *this; }
    Code& Str(u32 pool) { return Raw({0x16, u8(pool >> 8), u8(pool)}); }
    Code& Int(u32 v) { return Raw({0x01, u8(v >> 24), u8(v >> 16), u8(v >> 8), u8(v)}); }
    Code& Float(float v) { u32 b; std::memcpy(&b, &v, 4); return Raw({0x00, u8(b >> 24), u8(b >> 16), u8(b >> 8), u8(b)}); }
    Code& Native(u32 index) { return Raw({0x13, u8(index >> 8 & 15), u8(index)}); }
    Code& Pop() { return Raw({0x43}); }
    Code& End() { return Raw({0x49}); }
    Code& Goto(u32 state) { return Raw({0x47, u8(state >> 24), u8(state >> 16), u8(state >> 8), u8(state)}); }
    Code& Call(u32 at, u32 target) {
        const u32 relative = target - (at + 5);
        return Raw({0x4a, u8(relative >> 24), u8(relative >> 16), u8(relative >> 8), u8(relative)});
    }
};

// The script blob (big-endian records after a little-endian header) plus the new states; empty when the script is not the one
// this was written against.
std::vector<u8> Build(const u8* blob) {
    if (*reinterpret_cast<const u32*>(blob + 8) != kMainShellSize) return {};
    const u32 base = *reinterpret_cast<const u32*>(blob + 0x10), code_base = 4 + *reinterpret_cast<const u32*>(blob + 0xC);
    if (std::strcmp(reinterpret_cast<const char*>(blob + 4 + base + Be32(blob + 8 + base + Be32(blob + 0xC + 0x20))), "mainShell") ||
        std::strcmp(reinterpret_cast<const char*>(blob + code_base + kMainGroup), "UIMainMenuAG") ||
        std::strcmp(reinterpret_cast<const char*>(blob + code_base + kVideosGroup), "UIVideosSubmenuAG") ||
        std::strcmp(reinterpret_cast<const char*>(blob + code_base + kHelpSelect), "[msg_B[4000]]") ||
        std::strcmp(reinterpret_cast<const char*>(blob + code_base + kHelpText), "[msg_B[4000]] [msg_D[4000]]") ||
        Be32(blob + 4 + kLoadState + 0x2c) != (0x03000000 | kLoadHandler) ||
        Be32(blob + 4 + kLoadState + 0x34) != (0x03000000 | kLoadClosedHandler) ||
        Be32(blob + 4 + kLoadState + 0x3c) != (0x03000000 | kGameStartHandler) || blob[code_base + kLoadClosedEnd] != 0x49 ||
        !std::equal(blob + code_base + kLoadClosedJump, blob + code_base + kLoadClosedJump + 3,
                    std::array<u8, 3>{0x17, 0, kLoadClosedEnd - kLoadClosedJump - 3}.begin()))
        return {};
    // State offsets are relative to blob + 4. The last settings row has an extra handler.
    constexpr u32 records_end = kFirstRow + 0x44 * kRows + 4 + 8;
    std::vector<u8> out(blob, blob + kMainShellSize);
    out.resize(records_end);
    const auto string = [&](const std::string& s) {
        const u32 at = u32(out.size()) - code_base;
        out.insert(out.end(), s.c_str(), s.c_str() + s.size() + 1);
        return at;
    };
    const u32 empty = string("");
    main_labels_at = u32(out.size());
    out.resize(out.size() + kMainRows * kLabelSize);
    main_clips_at = u32(out.size());
    for (u32 i = 0; i < kMainRows; ++i) string(kSelectClip);
    const auto main_select = [&](u32 row) { return main_clips_at - code_base + row * u32(sizeof(kSelectClip)); };
    u32 enable[kRows], select[kRows], row_text[kRows];
    for (u32 i = 0; i < kRows; ++i) enable[i] = string(std::format("menuItem{}Enable", i)), select[i] = string(std::format("menuItem{}Select", i));
    for (u32 i = 0; i < kRows; ++i) { // row text buffers the host rewrites in place
        row_text[i] = u32(out.size()) - code_base;
        row_at[i] = u32(out.size());
        out.resize(out.size() + kRowSize);
    }
    out.resize((out.size() + 3) & ~3);
    const auto block = [&](const Code& c) {
        const u32 at = u32(out.size()) - code_base;
        out.insert(out.end(), c.bytes.begin(), c.bytes.end());
        return at;
    };
    const auto here = [&] { return u32(out.size()) - code_base; };
    // Row `i` of the list group: Enable shows it unfocused (red), Select plays the focus clip forward (mode 8) or backward (mode 9).
    const auto clip = [](Code& c, u32 mode, u32 name, float speed) {
        c.Str(kVideosGroup).Int(mode).Str(name).Float(1).Float(speed).Int(5).Native(0x5a).Pop();
    };
    const auto slot = [](Code& c, u32 text, u32 row) { c.Str(text).Int(row).Raw({0x11, 0xe0, 0xf0}).Pop(); }; // menuSlots[row] = text
    // Main menu enter (sub): fill menuSlots, enable the rows and focus the first entry.
    Code show;
    show.Raw({0x1a, 0x08, 0xe1, 0x41}).Pop();
    for (u32 i = 0; i < kRows; ++i)
        slot(show, i < kMainRows ? main_labels_at - code_base + i * kLabelSize : empty, i);
    for (u32 i = 0; i < kMainRows; ++i) clip(show, 8, enable[i], 1);
    main_initial_target = u32(out.size() + show.bytes.size() + 1);
    show.Goto(kNewGameState).Raw({0x4b});
    const u32 show_sub = block(show);
    // A row state focuses its list row on enter and releases it on exit.
    const auto focus = [&](u32 name, u32& on_enter, u32& on_leave) {
        Code enter, leave;
        clip(enter, 8, name, 0);
        clip(leave, 9, name, 0);
        on_enter = block(enter.End());
        on_leave = block(leave.End());
    };
    // Handlers: a sound sub (nav 0x1bb, select 0x1ab, back 0x1b3) then a goto.
    const auto go = [&](u32 sound, u32 target) { return block(Code().Call(here(), sound).Goto(target).End()); };
    // As the native Load handlers: nothing while the load they requested is pending (this[0x140]).
    const auto idle_go = [&](u32 sound, u32 target) {
        Code c;
        c.Raw({0x04, 0xe1, 0x40, 0x2f, 0x17, 0x00, u8(sound ? 10 : 5)});
        if (sound) c.Call(here() + 7, sound);
        c.Goto(target).End();
        return block(c);
    };
    const auto handler = [](std::vector<u32>& entries, u32 code, std::initializer_list<u32> events) {
        entries.push_back(0x03000000 | code);
        entries.insert(entries.end(), events);
    };
    const std::initializer_list<u32> up{0x800000e0, 0xec}, down{0x800000e4, 0xf0}, left{0x800000e6, 0xf2}, right{0x800000e2, 0xee};
    const auto record = [&](u32 state, u32 parent, const std::vector<u32>& entries) {
        u8* const at = &out[state + 4];
        PutBe32(at, 0x04000000 | (0xc + 4 * u32(entries.size())));
        PutBe32(at + 4, parent);
        PutBe32(at + 8, 0x28c);
        for (u32 i = 0; i < entries.size(); ++i) PutBe32(at + 0xc + 4 * i, entries[i]);
    };
    struct Item { u32 row, state, up, down, select; };
    for (const Item& item : {Item{RehookedRow, kRehookedState, kTreasuresState, kHelpState, kSettingsState},
                             Item{HelpRow, kHelpState, kRehookedState, kQuitState, kHelpTarget},
                             Item{QuitRow, kQuitState, kHelpState, kFirstItemTarget, kQuitTarget},
                             Item{ContinueRow, kContinueState, kQuitState, kNewGameState, kContinueTarget}}) {
        u32 on_enter, on_leave;
        focus(main_select(item.row), on_enter, on_leave);
        std::vector<u32> entries{0x01000000 | on_enter, 0x02000000 | on_leave};
        handler(entries, go(0x1bb, item.up), up);
        handler(entries, go(0x1bb, item.down), down);
        // The native Load handler plays its own confirmation sound.
        handler(entries, item.select == kContinueTarget ? block(Code().Goto(kContinueTarget).End()) : go(0x1ab, item.select), {0xe9});
        record(item.state, kMainState, entries);
    }
    // Load's end refreshes Continue: the jz now also skips the goto, which the start of the load used to reach and clear Continue's request.
    Code load_closed;
    load_closed.bytes.assign(blob + code_base + kLoadClosedHandler, blob + code_base + kLoadClosedEnd);
    load_closed.bytes[kLoadClosedJump - kLoadClosedHandler + 2] += 5;
    const u32 load_closed_code = block(load_closed.Goto(kRefreshContinueTarget).End());
    PutBe32(out.data() + 4 + kLoadState + 0x34, 0x03000000 | load_closed_code);
    // Continue's load enters through the native Load X code, under Continue so the row keeps its focus, with Load's other events.
    {
        std::vector<u32> entries{0x01000000 | kLoadHandler};
        handler(entries, idle_go(0x1bb, kQuitState), up);
        handler(entries, idle_go(0x1bb, kNewGameState), down);
        handler(entries, idle_go(0, kContinueTarget), {0xe9});
        handler(entries, load_closed_code, {0x14b});
        handler(entries, kGameStartHandler, {0x149});
        record(kContinueLoadState, kContinueState, entries);
    }
    // Submenu (child of turnOn, so MainMenu's exit hides the main group): fills the rows, enables them all, focuses row 0; Circle goes back.
    Code enter_menu, leave_menu;
    enter_menu.Str(kHelpText).Raw({0x09, 0xe0, 0x20}).Pop();
    for (u32 i = 0; i < kRows; ++i) slot(enter_menu, row_text[i], i);
    for (u32 i = 0; i < kRows; ++i) clip(enter_menu, 8, enable[i], 1);
    enter_menu.Str(kVideosGroup).Int(1).Native(0x58).Pop().Goto(kFirstRow).End();
    leave_menu.Str(kVideosGroup).Int(1).Native(0x59).Pop().Str(kHelpSelect).Raw({0x09, 0xe0, 0x20}).Pop().End();
    {
        const u32 on_enter = block(enter_menu), on_leave = block(leave_menu);
        std::vector<u32> entries{0x01000000 | on_enter, 0x02000000 | on_leave};
        handler(entries, go(0x1b3, kBackTarget), {0xeb});
        record(kSettingsState, kTurnOnState, entries);
    }
    for (u32 i = 0; i < kRows; ++i) {
        u32 on_enter, on_leave;
        focus(select[i], on_enter, on_leave);
        std::vector<u32> entries{0x01000000 | on_enter, 0x02000000 | on_leave};
        handler(entries, go(0x1bb, kMoveTarget | i << 4), up);
        handler(entries, go(0x1bb, kMoveTarget | i << 4 | 1), down);
        handler(entries, go(0x1bb, kSettingTarget | i << 4), left);
        handler(entries, go(0x1bb, kSettingTarget | i << 4 | 1), right);
        if (i == kRows - 1) handler(entries, go(0x1ab, kPageNextTarget), {0xe9});
        record(kFirstRow + 0x44 * i, kSettingsState, entries);
    }
    out.resize((out.size() + 3) & ~3);
    *reinterpret_cast<u32*>(&out[8]) = u32(out.size());
    // Keep the native handlers and replace just the two edges of the original navigation cycle.
    for (const auto& [at, old, target] : {std::tuple{0xdee + 6, kTreasuresState, kBeforeNewGameTarget}, std::tuple{0xf81 + 6, kNewGameState, kRehookedState}}) {
        if (Be32(&out[code_base + at]) != u32(old)) return {};
        PutBe32(&out[code_base + at], target);
    }
    const auto replace = [&](u32 at, std::initializer_list<u8> original, const Code& with) {
        if (!std::equal(original.begin(), original.end(), &out[code_base + at])) return false;
        std::copy(with.bytes.begin(), with.bytes.end(), &out[code_base + at]);
        return true;
    };
    const auto use_list_group = [&](u32 at) { return replace(at, {0x16, u8(kMainGroup >> 8), u8(kMainGroup)}, Code().Str(kVideosGroup)); };
    // Show, hide and focus the title rows through the reusable list, preserving the native animation calls.
    if (!replace(0xd6e, {0x47, 0, 0, 1, 0x14}, Code().Call(0xd6e, show_sub)) ||
        !use_list_group(0xd55) || !use_list_group(0xd96))
        return {};
    constexpr std::tuple<MainRow, u32, u32, u32> native_focus[]{
        {NewGameRow, 0xda9, 0xdcf, 0x9b82}, {LoadRow, 0xe14, 0xe32, 0x9c5d},
        {OptionsRow, 0xee3, 0xf01, 0x9cc1}, {TreasuresRow, 0xf43, 0xf61, 0x9d03},
    };
    for (const auto& [row, enter, leave, name] : native_focus) {
        for (const u32 at : {enter, leave})
            if (!use_list_group(at) || !replace(at + 8, {0x16, u8(name >> 8), u8(name)}, Code().Str(main_select(row))))
                return {};
    }
    return out;
}

bool IsResident(const u8* blob) {
    return blob && *reinterpret_cast<const u32*>(blob + 8) == kResidentSize &&
           std::memcmp(blob + 0xe94d, "BrightnessSetting", sizeof("BrightnessSetting")) == 0;
}

// R_UI.WAD's SMI_R_UI. Both the title and the in-game pause enter PauseOptions.
std::vector<u8> BuildPauseOptions(const u8* blob) {
    constexpr u32 code_base = 0x1de4, symbols = 0x11028;
    constexpr u32 options = 0x1424, controls = 0x1448, display = 0x14e4;
    constexpr u32 blank_line = 0xa2ff, controls_up = 0xa428, display_down = 0xa526;
    constexpr u32 pause_text = 0xd52f, select_clip = 0xeeb6, select_sound = 0xd688, nav_sound = 0x2a;
    constexpr u8 blank[]{0x16, 0xc9, 0x59, 0x09, 0xe0, 0x05, 0x43};
    const u8* const code = blob + code_base;
    if (*reinterpret_cast<const u32*>(blob + 0xc) != code_base - 4 ||
        *reinterpret_cast<const u32*>(blob + 0x10) != symbols || blob[symbols] != 0 ||
        Be32(blob + options + 4) != 0x040000f4 || Be32(blob + display + 8) != options ||
        Be32(blob + controls + 0x18) != 0x0300a423 || Be32(blob + display + 0x24) != 0x0300a521 ||
        std::memcmp(code + blank_line, blank, sizeof(blank)) != 0 ||
        code[controls_up] != 0x47 || Be32(code + controls_up + 1) != display ||
        code[display_down] != 0x47 || Be32(code + display_down + 1) != controls ||
        std::memcmp(code + pause_text, "PauseText", sizeof("PauseText")) != 0 ||
        std::memcmp(code + select_clip, "menuItem4Select", sizeof("menuItem4Select")) != 0 ||
        std::memcmp(code + select_sound, "SND_MM_SELECT", sizeof("SND_MM_SELECT")) != 0)
        return {};

    // Literals have 16-bit offsets. Insert before the symbols, keeping the last original literal's NUL.
    constexpr u32 label_ref = symbols + 1 - code_base;
    constexpr u32 label_bytes = (1 + kLabelSize + 3) & ~3u;
    static_assert(label_ref <= 0xffff);
    std::vector<u8> out(blob, blob + symbols);
    out.resize(symbols + label_bytes);
    options_label_at = symbols + 1;
    WriteText(out.data() + options_label_at, kLabelSize, UI::Tr("Controls and Extras"));
    out.insert(out.end(), blob + symbols, blob + 4 + kResidentSize);
    out.resize((out.size() + 3) & ~3u);
    const u32 row = u32(out.size()) - 4;
    out.resize(out.size() + 0x34);

    const auto block = [&](const Code& c) {
        const u32 at = u32(out.size()) - code_base;
        out.insert(out.end(), c.bytes.begin(), c.bytes.end());
        return at;
    };
    const auto focus = [&](bool entering) {
        Code c;
        if (entering) c.Native(0x6b).Pop();
        return block(c.Str(pause_text).Int(entering ? 8 : 9).Str(select_clip).Int(3).Native(0x5a).Pop().End());
    };
    const auto navigate = [&](u32 target) {
        return block(Code().Call(u32(out.size()) - code_base, nav_sound).Goto(target).End());
    };
    // Only the parent reference is needed for a Goto to enter the appended row, as in mainShell.
    const u32 record[]{0x04000034, options, Be32(blob + display + 0xc),
                       0x01000000 | focus(true), 0x02000000 | focus(false),
                       0x03000000 | navigate(display), 0x800000e0, 0xec,
                       0x03000000 | navigate(controls), 0x800000e4, 0xf0,
                       0x03000000 | block(Code().Str(select_sound).Native(0x5c).Pop().Goto(kHelpTarget).End()), 0xe9};
    for (u32 i = 0; i < std::size(record); ++i) PutBe32(out.data() + row + 4 + i * 4, record[i]);
    out[code_base + blank_line + 1] = u8(label_ref >> 8);
    out[code_base + blank_line + 2] = u8(label_ref);
    PutBe32(out.data() + code_base + controls_up + 1, row);
    PutBe32(out.data() + code_base + display_down + 1, row);
    *reinterpret_cast<u32*>(out.data() + 8) = u32(out.size()) - 4;
    *reinterpret_cast<u32*>(out.data() + 0x10) = symbols + label_bytes;
    return out;
}

// 01154C40(script): first-time init copies the template's blob pointer (template + 0x38 -> +8) into script + 0x50.
void PS4_SYSV_ABI Init(u8* script) {
    u8** const slot = reinterpret_cast<u8**>(*reinterpret_cast<u8**>(script + 0x38) + 8);
    u8* const original = *slot;
    u8* replacement = nullptr;
    if (!*reinterpret_cast<u8**>(script + 0x50) && original && *reinterpret_cast<const u32*>(original + 8) == kMainShellSize) {
        if (patched.empty()) {
            patched = Build(original);
            if (!patched.empty()) {
                WriteRows();
            }
            if (DebugEnabled()) Log(patched.empty() ? "Title menu: script not recognized" : "Title menu: Rehooked, Controls and Extras and Quit entries added");
        }
        if (!patched.empty()) {
            WriteMainRows(Libraries::SaveData::Dialog::HasContinueSave());
            replacement = patched.data();
        }
    } else if (!*reinterpret_cast<u8**>(script + 0x50) && IsResident(original)) {
        // All resident instances must use the same blob pointer: the VM keys shared variables by it.
        static const bool ready = [original] {
            patched_options = BuildPauseOptions(original);
            if (DebugEnabled()) Log(patched_options.empty() ? "Options menu: script not recognized" : "Options menu: Controls and Extras added");
            return !patched_options.empty();
        }();
        if (ready) replacement = patched_options.data();
    }
    if (replacement) *slot = replacement;
    original_init(script);
    if (replacement) *slot = original; // the template keeps owning its own blob
}
// 01154970(script, state): goes to a state; the reserved targets of the handlers above are host actions instead.
void PS4_SYSV_ABI Goto(u8* script, u32 state) {
    const auto* blob = *reinterpret_cast<const u8* const*>(script + 0x50);
    const bool title_script = blob && blob == patched.data();
    const bool options_script = blob && blob == patched_options.data();
    if (title_script || options_script) RefreshLanguage();
    if (title_script) {
        if (state == kMainState || state == kLoadState)
            Libraries::SaveData::Dialog::RequestContinue(false);
        if (state == kMainState) {
            const auto depth = *reinterpret_cast<const s8*>(script + 0xb0);
            const auto* state_stack = reinterpret_cast<const u32*>(script + 0x70);
            const bool in_main_menu = depth >= 0 && depth < 16 &&
                std::find(state_stack, state_stack + depth + 1, kMainState) != state_stack + depth + 1;
            if (RefreshContinue(script, in_main_menu)) return;
        }
    }
    if (state == kBrightnessState && (options_script || IsResident(blob))) {
        original_goto(script, state);
        Splash::Reveal();
        return;
    }
    if (state == kBackTarget) {
        bool changed = false;
        for (u32 i = 0; i < OptionCount; ++i) changed |= settings[i].index != initial_indices[i];
        if (changed) {
            if (!Window::RequestSettingsRestart(SelectedConfig())) Log("Cannot request settings restart");
        } else {
            page = 0;
            WriteRows();
            RefreshContinue(script, false);
            original_goto(script, kMainState);
        }
        return;
    }
    if (state == kHelpTarget && (title_script || options_script)) {
        ControlsUi::Open(title_script);
        return;
    }
    if (state == kQuitTarget) {
        SDL_Event quit{};
        quit.type = SDL_EVENT_QUIT;
        SDL_PushEvent(&quit);
        return;
    }
    if (state == kFirstItemTarget || state == kBeforeNewGameTarget) {
        original_goto(script, continue_available ? kContinueState : state == kFirstItemTarget ? kNewGameState : kQuitState);
        return;
    }
    if (state == kContinueTarget) {
        if (RefreshContinue(script, true)) return;
        Libraries::SaveData::Dialog::RequestContinue(true);
        original_goto(script, kContinueState); // leaves a finished attempt: a goto to the current state enters nothing
        original_goto(script, kContinueLoadState);
        return;
    }
    if (state == kRefreshContinueTarget) {
        Libraries::SaveData::Dialog::RequestContinue(false);
        RefreshContinue(script, true);
        return;
    }
    if (state >> 16 == kMoveTarget >> 16) {
        u32 row = state >> 4 & 15;
        do row = (row + (state & 1 ? 1 : kRows - 1)) % kRows; while (!Selectable(row));
        original_goto(script, kFirstRow + 0x44 * row);
        return;
    }
    if (state == kPageNextTarget || state >> 16 == kSettingTarget >> 16) {
        const u32 row = state == kPageNextTarget ? kRows - 1 : state >> 4 & 15;
        const bool forward = state == kPageNextTarget || (state & 1);
        if (row == kRows - 1) {
            if (forward && page + 1 < kPageCount) ++page;
            else if (!forward && page > 0) --page;
            WriteRows();
        } else if (row < kRows - 1 && Selectable(row)) {
            Setting& setting = settings[pages[page][row]];
            setting.index = (setting.index + (forward ? 1 : u32(setting.values.size()) - 1)) % u32(setting.values.size());
            WriteRow(row);
            if (pages[page][row] == Mechanism) WriteRow(row + 2); // Lazy AOT, greyed out unless AOT
            if (pages[page][row] == DisplayModeOption) WriteRow(row + 1); // VRR, greyed out when windowed
            if (DebugEnabled()) Log(std::format("Title menu: {} = {}", setting.label, setting.values[setting.index]));
        }
        return;
    }
    original_goto(script, state);
}
} // namespace TitleMenu

// Light log (GOW3_LIGHTLOG=1): the light lists the engine builds, for lighting bugs (an object too dark, too
// bright or flickering). Each distinct list is logged once, up to kLines per hook.
// 011A6080(item, set, filtered) copies an object's light set (offsets of pos/color/dir/flags at set[0..3], light
//    counts of the three types at set[4..6], capacity set[7]) into the VS's interleaved 48-byte records, dropping lights
//    whose flag byte has bit 7 when filtered; the item receives the counts at +0x1c8/+0x1cc/+0x1d0.
// 011C4860 builds the light set of one chunk of objects (Chunk).
// 012E2690 counts the lights that touch a dynamic chunk (Query); small results come with every nearby light's verdict.
namespace LightLog {
using CopyFn = void(PS4_SYSV_ABI*)(u64, u64, u64);
CopyFn original_copy;
std::atomic<u32> calls, copy_lines;
constexpr u32 kSeen = 1 << 16, kLines = 8192;  // signatures kept, lines per hook
std::atomic<u64> seen[kSeen]; // signatures already logged (open addressing, 0 = free)
// A signature not logged yet; a full neighbourhood counts as logged, so a hook costs a few probes even once the table fills.
bool Fresh(u64 key) {
    for (u32 n = 0, i = u32(key % kSeen); n < 16; ++n, i = (i + 1) % kSeen) {
        u64 entry = seen[i].load(std::memory_order_relaxed);
        if (!entry && seen[i].compare_exchange_strong(entry, key)) return true;
        if (entry == key) return false;
    }
    return false;
}
void PS4_SYSV_ABI Copy(u64 item, u64 set, u64 filtered) {
    original_copy(item, set, filtered);
    if (!set) return;
    const auto* h = reinterpret_cast<const u32*>(set);
    const auto* out = reinterpret_cast<const u32*>(item + 0x1c8);
    const auto* flags = reinterpret_cast<const signed char*>(set + h[3]);
    const u32 total = h[4] + h[5] + h[6];
    u32 flagged = 0;
    for (u32 i = 0; i < total && i < 64; ++i) flagged += flags[i] < 0;
    const u32 call = calls.fetch_add(1, std::memory_order_relaxed);
    u64 key = 0;
    for (const u32 v : {1u, h[4], h[5], h[6], h[7], u32(u8(filtered) != 0), flagged, out[0], out[1], out[2]}) key = key * 131 + v;
    if (!Fresh(key | 1ull << 63) || copy_lines.fetch_add(1, std::memory_order_relaxed) >= kLines) return;
    std::string text = std::format("Lights #{} set {:#x} raw {}/{}/{} cap {} filtered {} flagged {} out {}/{}/{}", call, set, h[4], h[5], h[6], h[7],
                                   u32(u8(filtered) != 0), flagged, out[0], out[1], out[2]);
    const auto* pos = reinterpret_cast<const float*>(set + h[0]);
    const auto* color = reinterpret_cast<const float*>(set + h[1]);
    for (u32 i = 0; i < total && i < 12; ++i)
        text += std::format("\nLights   light {} flag {:#04x} pos {:.1f} {:.1f} {:.1f} invr {:.4f} color {:.2f} {:.2f} {:.2f} i {:.2f}", i, u8(flags[i]), pos[i * 4],
                            pos[i * 4 + 1], pos[i * 4 + 2], pos[i * 4 + 3], color[i * 4], color[i * 4 + 1], color[i * 4 + 2], color[i * 4 + 3]);
    Log(text);
}
// 012E2690(xmm0 sphere, flags, mask, lights, count, max) counts the lights (0x50 bytes each: x y z radius at +0, flags at +0x30,
// group mask at +0x34) that touch a dynamic chunk's sphere and pass the type test (bit 24 of the light flags == bit 2 of
// `flags`), the 0x10/0x20/0xc bits and (light mask & mask). Small results are logged with the verdict of every nearby light.
using QueryFn = u32(PS4_SYSV_ABI*)(__m128, u32, u32, const u8*, u32, u32);
QueryFn original_query;
std::atomic<u32> query_lines;
u32 PS4_SYSV_ABI Query(__m128 sphere, u32 flags, u32 mask, const u8* lights, u32 count, u32 max) {
    const u32 result = original_query(sphere, flags, mask, lights, count, max);
    if (result > 8 || !lights) return result;
    alignas(16) float s[4];
    _mm_store_ps(s, sphere);
    u64 key = 0;
    for (const u32 v : {3u, u32(s[0] / 4), u32(s[1] / 4), u32(s[2] / 4), u32(s[3] / 2), flags, mask, result}) key = key * 131 + v;
    if (!Fresh(key | 1ull << 63) || query_lines.fetch_add(1, std::memory_order_relaxed) >= kLines) return result;
    std::string text = std::format("Lights query sphere {:.1f} {:.1f} {:.1f} r {:.1f} flags {:#x} mask {:#x} lights {} result {}", s[0], s[1], s[2], s[3], flags, mask, count, result);
    u32 shown = 0;
    for (u32 i = 0; i < count && shown < 14; ++i) {
        const auto* l = reinterpret_cast<const float*>(lights + u64(i) * 0x50);
        const float dx = l[0] - s[0], dy = l[1] - s[1], dz = l[2] - s[2], reach = l[3] + s[3];
        const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (distance > reach + 40.f) continue;
        const u32 light_flags = *reinterpret_cast<const u32*>(lights + u64(i) * 0x50 + 0x30), light_mask = *reinterpret_cast<const u32*>(lights + u64(i) * 0x50 + 0x34);
        ++shown;
        text += std::format("\nLights   light {} pos {:.1f} {:.1f} {:.1f} r {:.1f} dist {:.1f} touch {} flags {:#x} mask {:#x} type_ok {} mask_ok {} col {:.3f} {:.3f} {:.3f} i {:.3f}", i, l[0], l[1], l[2], l[3], distance,
                            distance * distance < reach * reach, light_flags, light_mask, ((light_flags >> 24) & 1) == ((flags & 4) != 0), (light_mask & mask) != 0, l[8], l[9], l[10], l[11]);
    }
    Log(text);
    return result;
}
// 011C4860(ctx, a2, chunk, spheres, visible, out; stack: a7) builds the light set of one chunk of objects (chunk+0xa8 objects
// from index chunk+0xaa) in thread-local scratch and returns it: the lights that overlap the sphere (x, y, z, radius) of
// every object whose bit is set in `visible`; chunk+0x60 holds the chunk sphere computed from the visible objects.
using ChunkFn = u64(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64, u64);
ChunkFn original_chunk;
std::atomic<u32> chunk_lines;
u64 PS4_SYSV_ABI Chunk(u64 ctx, u64 a2, u64 chunk, u64 spheres, u64 visible, u64 out, u64 a7) {
    const u64 set = original_chunk(ctx, a2, chunk, spheres, visible, out, a7);
    const auto* h = reinterpret_cast<const u32*>(set);
    const u32 objects = *reinterpret_cast<const u16*>(chunk + 0xa8), first = *reinterpret_cast<const u16*>(chunk + 0xaa);
    const auto* sphere = reinterpret_cast<const float*>(spheres);
    const auto* chunk_sphere = reinterpret_cast<const float*>(chunk + 0x60);
    const auto shown = [&](u32 i) { return reinterpret_cast<const u8*>(visible)[(first + i) >> 3] >> ((first + i) & 7) & 1; };
    u32 visible_count = 0;
    for (u32 i = 0; i < objects; ++i) visible_count += shown(i);
    u64 key = 0;
    for (const u32 v : {2u, h[4], h[5], h[6], objects, visible_count, u32(sphere[0] / 8), u32(sphere[3] / 4)}) key = key * 131 + v;
    if (!Fresh(key | 1ull << 63) || chunk_lines.fetch_add(1, std::memory_order_relaxed) >= kLines) return set;
    std::string list;
    for (u32 i = 0, n = 0; i < objects && n < 6; ++i)
        if (shown(i)) ++n, list += std::format(" ({:.1f} {:.1f} {:.1f} r {:.1f})", sphere[i * 4], sphere[i * 4 + 1], sphere[i * 4 + 2], sphere[i * 4 + 3]);
    std::string text = std::format("Lights chunk {:#x} objects {} visible {} chunk sphere {:.1f} {:.1f} {:.1f} r {:.1f} lights {}/{}/{} spheres{}", chunk, objects,
                                   visible_count, chunk_sphere[0], chunk_sphere[1], chunk_sphere[2], chunk_sphere[3], h[4], h[5], h[6], list);
    const auto* pos = reinterpret_cast<const float*>(set + h[0]);
    const auto* color = reinterpret_cast<const float*>(set + h[1]);
    const u32 total = h[4] + h[5] + h[6];
    if (total <= 8)
        for (u32 i = 0; i < total; ++i)
            text += std::format("\nLights   chunk light {} pos {:.1f} {:.1f} {:.1f} invr {:.4f} color {:.2f} {:.2f} {:.2f}", i, pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2],
                                pos[i * 4 + 3], color[i * 4], color[i * 4 + 1], color[i * 4 + 2]);
    Log(text);
    return set;
}
} // namespace LightLog
} // namespace

void ShadowFileOpen(const char* path) {
    if (!Shadow::ring || !path) return;
    const size_t n = strnlen(path, 256);
    if (n > 4 && !CompareCaseInsensitive(path + n - 4, ".wad")) Shadow::PushName(Shadow::WadOpen, 0, 0, 0, path);
}

void InstallHooks(Core::Module& eboot, const Config& config) {
    guest_base = eboot.GetBaseAddress();
#ifdef _WIN32
    EventRegister(&kFrameProvider, nullptr, nullptr, &frame_events);
#endif
    configured_frame_rate = GameFrameRate(config.fps);
    TitleMenu::Configure(config);
    // Ghidra: complete prologues, verified before the detours are installed.
    constexpr u8 ray[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54,
                       0x53, 0x48, 0x81, 0xec, 0xb8, 0x00, 0x00, 0x00};
    constexpr u8 hang[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54,
                        0x53, 0x48, 0x81, 0xec, 0x38, 0x03, 0x00, 0x00};
    constexpr u8 mode[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54,
                        0x53, 0x48, 0x83, 0xec, 0x78};
    constexpr u8 physics[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x83, 0xec, 0x10, 0x49, 0x89, 0xf6, 0x48, 0x89, 0xfb};
    constexpr u8 air[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x40};
    original_ray_query = reinterpret_cast<RayQuery>(PrepareTrampoline(0x156d10, ray));
    original_hang_query = reinterpret_cast<ContactQuery>(PrepareTrampoline(0x597a0, hang));
    original_mode_query = reinterpret_cast<ModeQuery>(PrepareTrampoline(0x9ac00, mode));
    original_motion_physics = reinterpret_cast<MotionPhysics>(PrepareTrampoline(0x9fd20, physics));
    original_air_motion = reinterpret_cast<MotionPhysics>(PrepareTrampoline(0x970f0, air));
    constexpr u8 frame_loop[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x41, 0x89, 0xfe, 0xe8, 0x1e, 0xf9, 0x0f, 0x00};
    original_frame_loop = reinterpret_cast<FrameLoop>(PrepareTrampoline(0x1ea670, frame_loop, 13));
    constexpr u8 light_array[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50, 0x49, 0x89, 0xfe};
    original_light_array = reinterpret_cast<LightArrayFn>(PrepareTrampoline(0x1b6e60, light_array));
    constexpr u8 frame_step[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x83, 0xec, 0x10, 0xe8, 0xb0, 0xa5, 0x0f, 0x00};
    original_frame_step = reinterpret_cast<FrameStepFn>(PrepareTrampoline(0x1ef9e0, frame_step, 11));
    std::vector<std::pair<u64, const void*>> detours{
        {0x1ef9e0, reinterpret_cast<const void*>(StepFrame)},
        {0x1b6e60, reinterpret_cast<const void*>(BuildLightArray)},
        {0x1ea670, reinterpret_cast<const void*>(RunFrame)},
        {0x1f0f00, reinterpret_cast<const void*>(FlipArenas)},
        {0x1efe10, reinterpret_cast<const void*>(SetFrameRate)},
        {0x156d10, reinterpret_cast<const void*>(QueryLedgeSupport)},
        {0x597a0, reinterpret_cast<const void*>(HangContact)},
        {0x9ac00, reinterpret_cast<const void*>(ResolveLedgeMotion)},
        {0x9fd20, reinterpret_cast<const void*>(ProbeMotionPhysics)},
        {0x970f0, reinterpret_cast<const void*>(ResolveBladeAirMotion)},
        {0x1a5e30, reinterpret_cast<const void*>(WaitJobs)},
    };
    if (config.fps > 60) {
        constexpr u8 prepare_motion[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54,
                                      0x53, 0x48, 0x81, 0xec, 0x88, 0x00, 0x00, 0x00};
        constexpr u8 rebase_actor[]{0xc5, 0xf8, 0x28, 0x46, 0x30, 0xc5, 0xf8, 0x5c, 0x87, 0xa0, 0x00, 0x00, 0x00,
                                   0xc5, 0xf8, 0x58, 0x8f, 0xe0, 0x02, 0x00, 0x00};
        original_prepare_actor_motion = reinterpret_cast<PrepareActorMotion>(PrepareTrampoline(0x800f0, prepare_motion));
        original_rebase_actor = reinterpret_cast<RebaseActor>(PrepareTrampoline(0x5f920, rebase_actor));
        detours.emplace_back(0x800f0, reinterpret_cast<const void*>(PrepareGrappleRelease));
        detours.emplace_back(0x5f920, reinterpret_cast<const void*>(RebaseGrappleRelease));
    }
    if (config.nvidia_reflex) {
        constexpr u8 input[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50};
        original_input_update = reinterpret_cast<InputUpdate>(PrepareTrampoline(0x1eec40, input));
        detours.emplace_back(0x1eec40, reinterpret_cast<const void*>(UpdateInput));
    }
    if (config.cinematics_4k) {
        constexpr u8 path[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x49, 0x89, 0xf6,
                            0x48, 0x89, 0xfb, 0x31, 0xc0};
        original_cinematic_path = reinterpret_cast<CinematicPathFn>(PrepareTrampoline(0x1eb770, path));
        detours.emplace_back(0x1eb770, reinterpret_cast<const void*>(CinematicPath));
    }
    if (DebugDeep("GOW3_SHADOW")) {
        constexpr u8 view[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50, 0x48, 0x89, 0xf3};
        constexpr u8 rebase[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x49, 0x89, 0xfe};
        constexpr u8 release[]{0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x03, 0xff, 0x48, 0x2c};
        constexpr u8 frame[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x18};
        Shadow::original_view = reinterpret_cast<Shadow::ViewFn>(PrepareTrampoline(0x1c7d70, view));
        Shadow::original_rebase = reinterpret_cast<Shadow::RebaseFn>(PrepareTrampoline(0x1c8280, rebase));
        Shadow::original_release = reinterpret_cast<Shadow::ReleaseFn>(PrepareTrampoline(0x1c8d80, release));
        Shadow::original_move = reinterpret_cast<Shadow::MoveFn>(PrepareTrampoline(0x1f1540, frame));
        Shadow::original_section = reinterpret_cast<Shadow::SectionFn>(PrepareTrampoline(0x1f5b90, frame));
        detours.insert(detours.end(), {{0x1c7d70, reinterpret_cast<const void*>(Shadow::BuildView)},
                                       {0x1c8280, reinterpret_cast<const void*>(Shadow::RebaseBlock)},
                                       {0x1c8d80, reinterpret_cast<const void*>(Shadow::ReleaseBlock)},
                                       {0x1f1540, reinterpret_cast<const void*>(Shadow::GpuMemmove)},
                                       {0x1f5b90, reinterpret_cast<const void*>(Shadow::OpenSection)}});
        Shadow::ring = new Shadow::Slot[Shadow::kSlots]();
        Shadow::file_path = PortDirectory() / std::format("shadow-{:%Y%m%d-%H%M%S}.bin",
                                                          std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
        Shadow::drain_thread = std::jthread(Shadow::Drain);
    }
    if (DebugDeep("GOW3_DRAWLOG")) {
        constexpr u8 draw_index[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x49, 0x89, 0xd7, 0x41, 0x89, 0xf6};
        constexpr u8 draw_auto[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41, 0x89, 0xf6, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08};
        constexpr u8 dispatch[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x41, 0x89, 0xce};
        constexpr u8 submit[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x48, 0x81, 0xec, 0x08, 0xc0, 0x00, 0x00};
        DrawShadow::original_draw_index = reinterpret_cast<DrawShadow::DrawIndexFn>(PrepareTrampoline(0x23a230, draw_index));
        DrawShadow::original_draw_auto = reinterpret_cast<DrawShadow::DrawAutoFn>(PrepareTrampoline(0x23a170, draw_auto));
        DrawShadow::original_dispatch = reinterpret_cast<DrawShadow::DispatchFn>(PrepareTrampoline(0x23a380, dispatch));
        DrawShadow::original_submit = reinterpret_cast<DrawShadow::SubmitFn>(PrepareTrampoline(0x1a2aa0, submit));
        detours.insert(detours.end(), {{0x23a230, reinterpret_cast<const void*>(DrawShadow::DrawIndex2)},
                                       {0x23a170, reinterpret_cast<const void*>(DrawShadow::DrawIndexAuto)},
                                       {0x23a380, reinterpret_cast<const void*>(DrawShadow::DispatchDirect)},
                                       {0x1a2aa0, reinterpret_cast<const void*>(DrawShadow::SubmitFrame)}});
        DrawShadow::ring = new DrawShadow::Slot[DrawShadow::kSlots]();
        DrawShadow::file_path = PortDirectory() / std::format("draws-{:%Y%m%d-%H%M%S}.bin",
                                                              std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
        DrawShadow::drain_thread = std::jthread(DrawShadow::Drain);
    }
    if (DebugDeep("GOW3_LIGHTLOG")) { // the per-object copy, the per-chunk build and the light query, together
        constexpr u8 light_query[]{0x55, 0x41, 0x56, 0x53, 0x31, 0xc0, 0x85, 0xc9, 0x0f, 0x84, 0x07, 0x02, 0x00, 0x00};
        constexpr u8 light_chunk[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x68};
        constexpr u8 light_copy[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x28};
        LightLog::original_query = reinterpret_cast<LightLog::QueryFn>(PrepareTrampoline(0x2e2690, light_query, 8));
        LightLog::original_chunk = reinterpret_cast<LightLog::ChunkFn>(PrepareTrampoline(0x1c4860, light_chunk));
        LightLog::original_copy = reinterpret_cast<LightLog::CopyFn>(PrepareTrampoline(0x1a6080, light_copy));
        detours.insert(detours.end(), {{0x2e2690, reinterpret_cast<const void*>(LightLog::Query)},
                                       {0x1c4860, reinterpret_cast<const void*>(LightLog::Chunk)},
                                       {0x1a6080, reinterpret_cast<const void*>(LightLog::Copy)}});
    }
    constexpr u8 shell_init[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x28, 0x49, 0x89, 0xfe};
    constexpr u8 shell_goto[]{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x68, 0x48, 0x89, 0xfb};
    TitleMenu::original_init = reinterpret_cast<TitleMenu::InitFn>(PrepareTrampoline(0x154c40, shell_init));
    TitleMenu::original_goto = reinterpret_cast<TitleMenu::GotoFn>(PrepareTrampoline(0x154970, shell_goto));
    detours.insert(detours.end(), {{0x154c40, reinterpret_cast<const void*>(TitleMenu::Init)},
                                   {0x154970, reinterpret_cast<const void*>(TitleMenu::Goto)}});
    std::vector<MemoryPatch> patches;
    for (const auto [rva, target] : detours) {
        // jmp qword ptr [rip+0]; dq target
        std::vector<u8> jump{0xff, 0x25, 0, 0, 0, 0};
        const auto address = reinterpret_cast<u64>(target);
        jump.insert(jump.end(), reinterpret_cast<const u8*>(&address), reinterpret_cast<const u8*>(&address) + 8);
        patches.push_back({rva, {Guest<const u8*>(rva), Guest<const u8*>(rva) + jump.size()}, std::move(jump)});
    }
    if (config.fps > 60) AddMotionSoundPatches(eboot, patches);
    // The job queue's three spin loops ([mov ecx, 1;] xor eax, eax; lock cmpxchg [DAT_024ba6e0], rcx; test rax, rax; jnz),
    // each replaced whole by `mov rax, LockJobQueue; call rax`: only callee-saved registers are live past them.
    for (const auto& [rva, loop] : std::initializer_list<std::pair<u64, std::vector<u8>>>{
             {0x1a5910, {0x31, 0xc0, 0xf0, 0x48, 0x0f, 0xb1, 0x0d, 0xc5, 0x4d, 0x31, 0x01, 0x48, 0x85, 0xc0, 0x75, 0xf0}},
             {0x1a59d0, {0xb9, 0x01, 0x00, 0x00, 0x00, 0x31, 0xc0, 0xf0, 0x48, 0x0f, 0xb1, 0x0d, 0x00, 0x4d, 0x31, 0x01, 0x48, 0x85, 0xc0, 0x75, 0xeb}},
             {0x1a5d70, {0xb9, 0x01, 0x00, 0x00, 0x00, 0x31, 0xc0, 0xf0, 0x48, 0x0f, 0xb1, 0x0d, 0x60, 0x49, 0x31, 0x01, 0x48, 0x85, 0xc0, 0x75, 0xeb}}}) {
        std::vector<u8> call(loop.size(), 0x90);
        const auto target = reinterpret_cast<u64>(&LockJobQueue);
        call[0] = 0x48, call[1] = 0xb8, std::memcpy(call.data() + 2, &target, sizeof(target)), call[10] = 0xff, call[11] = 0xd0;
        patches.push_back({rva, loop, std::move(call)});
    }
    // 011A5DF6, the enqueue's tail: sceKernelSignalSema(workers, 1) ran inside the lock, so the worker it woke met the lock
    // held across a system call. The same instructions with the unlock first (displacements moved): a job's token only
    // comes later than the job, so a woken worker still finds one.
    patches.push_back({0x1a5df6,
                       {0x48, 0x8b, 0x3d, 0xeb, 0x48, 0x31, 0x01, 0xbe, 0x01, 0x00, 0x00, 0x00, 0xe8, 0xb9, 0x45, 0x14, 0x00,
                        0x31, 0xc0, 0x48, 0x87, 0x05, 0xd0, 0x48, 0x31, 0x01, 0xb0, 0x01},
                       {0x31, 0xc0, 0x48, 0x87, 0x05, 0xe1, 0x48, 0x31, 0x01, 0x48, 0x8b, 0x3d, 0xe2, 0x48, 0x31, 0x01,
                        0xbe, 0x01, 0x00, 0x00, 0x00, 0xe8, 0xb0, 0x45, 0x14, 0x00, 0xb0, 0x01}});
    // 01166003, the LUT refill's `lea rdi, [0146A5C0]; mov esi, 32; call 012C61F0`: `mov rdi, rbx (the effect);
    // mov rax, FillColorLattice; call rax`. Only callee-saved registers are live past it.
    std::vector<u8> lattice{0x48, 0x89, 0xdf, 0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xd0, 0x66, 0x90};
    const auto lattice_target = reinterpret_cast<u64>(&FillColorLattice);
    std::memcpy(lattice.data() + 5, &lattice_target, sizeof(lattice_target));
    patches.push_back({0x166003, {0x48, 0x8d, 0x3d, 0xb6, 0x45, 0x30, 0x00, 0xbe, 0x20, 0x00, 0x00, 0x00, 0xe8, 0xdc, 0x01, 0x16, 0x00},
                       std::move(lattice)});
    MemoryPatcher(eboot).Apply(patches);
    Log("Job queue patched: spin loops at 011A5910, 011A59D0, 011A5D70 and 011A5E30 pause and yield; 011A5DF6 unlocks before "
        "signaling; 01166003 keeps unchanged color correction LUT halves");
    if (DebugEnabled()) Log(std::format("Installed {} game detours", detours.size()));
    Extras::InstallHooks(eboot);
}

void StopDiagnosticHooks() {
    auto stop_and_adopt = [](std::jthread& thread, std::filesystem::path& path) {
        if (!thread.joinable()) return;
        thread.request_stop();
        thread.join();
        if (!path.empty()) {
            DumpArchive::AdoptFile(path.filename().string(), path);
            path.clear();
        }
    };
    stop_and_adopt(Shadow::drain_thread, Shadow::file_path);
    stop_and_adopt(DrawShadow::drain_thread, DrawShadow::file_path);
}
} // namespace Gow3
