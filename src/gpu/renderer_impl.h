// SPDX-License-Identifier: GPL-2.0-or-later
// Private definition of Gpu::Renderer::Impl, shared by the renderer's translation units (vk_gpu, device, hazards, textures,
// buffers and the shaders/ strategies). The public interface is vk_gpu.h.
#pragma once
#include <vulkan/vulkan.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <source_location>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // TEMP fault (framecap builds without these)
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include "orbis/common/thread.h"
#include "gpu/device.h"
#include "gpu/hazards.h"
#include "gpu/vk_gpu.h"
#include "gpu/shaders/gcn_translator.h"
#include "gpu/shaders/shader_cache.h"
#include "gpu/shaders/shader_catalog.h"
#include "gpu/shaders/shader_contract.h"
#include "gpu/shaders/shader_runtime.h"
#define XXH_INLINE_ALL
#include "xxhash.h"
#include "gpu/piece_hash.h"
#include "debug/render_probe.h"

namespace Gpu {
namespace RenderProbe = Gow3::RenderProbe;
using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// The mirror holds guest addresses below 32 GB (heap, ring and arenas are below 16 GB). A driver that cannot reserve that
// much sparse address space at once gets 16 GB instead (SetupMirror; RADV on GFX8 with 16 GB of RAM, issue #35).
inline constexpr u64 kMirrorMax = 1ull << 35, kMirrorMin = 1ull << 34;
inline constexpr u64 kChunk = 256ull << 20;            // device memory behind the mirror, allocated in chunks
inline constexpr u64 kTableBytes = 64ull << 20;        // per-frame draw tables
inline constexpr u64 kTableRange = 16 << 10;           // a draw table's uniform buffer window (the prelude's Table0/Table1)
// Push constants of every translated program (the prelude's Push): the mirror delta and both stages' user data, which
// the program reads first (no memory load before it can start). The draw tables are dynamic uniform buffers (set 1).
struct Push {
    u64 mem_delta;
    u32 user[2][16];
};
// The set layout the cached driver binaries are made against (binary_seed holds the counts). kStorage: a view per T#, and
// Rage of Sparta's grab passes write one surface through ~6 new T# heights a frame (256 overflowed every ~50 frames).
inline constexpr u32 kTextures = 1024, kSamplers = 256, kStorage = 2048;
inline constexpr u32 kSamplerBinding = 18, kStorageBinding = 19;
inline thread_local const char* temp_call = nullptr;  // TEMP fault: the queue call this thread is in (address binding reports)
// TEMP fault: GPU address ranges bound and unbound, in order (VK_EXT_device_address_binding_report); device.cpp fills them.
struct TempBinding {
    u64 base, size, handle;
    u32 type, flags, frame;
    bool bound;
    const char* call;  // the queue call of this renderer the reporting thread was in
    u64 thread;
    u32 tid;
    double ms;        // since the first report
    std::string who;  // 8 MB and more: thread name, then the call stack as module+offset
};
extern std::mutex temp_bind_lock;
extern std::vector<TempBinding> temp_binds;
extern std::atomic<u32> temp_bind_frame;

inline float F(u32 v) {
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

// Open addressing over 64-bit keys other than ~0 (hashes, addresses), at most half full: the keys apart from the values (a
// probe reads one line), a multiplicative index (aligned addresses share their low bits). Values move when it grows.
template <class V> struct Flat {
    static constexpr u64 kFree = ~0ull;
    std::vector<u64> keys;
    std::vector<V> vals;
    u32 shift = 58, used = 0;  // 64 slots first
    size_t Home(u64 k) const { return size_t(k * 0x9E3779B97F4A7C15ull >> shift); }
    V* Find(u64 k) {
        if (keys.empty()) return nullptr;
        for (size_t i = Home(k);; i = (i + 1) & (keys.size() - 1))
            if (keys[i] == k || keys[i] == kFree) return keys[i] == k ? &vals[i] : nullptr;
    }
    V& operator[](u64 k) {
        if (V* v = Find(k)) return *v;
        if (2 * (used + 1) > keys.size()) {
            Flat n;
            n.shift = keys.empty() ? shift : shift - 1;
            n.keys.assign(size_t(1) << (64 - n.shift), kFree), n.vals.resize(n.keys.size());
            for (size_t i = 0; i < keys.size(); ++i)
                if (keys[i] != kFree) n[keys[i]] = std::move(vals[i]);
            *this = std::move(n);
        }
        size_t i = Home(k);
        while (keys[i] != kFree) i = (i + 1) & (keys.size() - 1);
        return ++used, keys[i] = k, vals[i] = V{};
    }
    void clear() { std::fill(keys.begin(), keys.end(), kFree), used = 0; }
};

struct Renderer::Impl {
    using Pos = std::pair<size_t, size_t>;  // segment, dword in PM4 chain

    // ---- Device, queues, memory allocation and extension entry points (device.cpp, vk_gpu.cpp, offline_device.cpp) ----
    VkInstance inst = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    float anisotropy = 0;
    float motion_blur_rate = 1;
    bool blend_interlock = false;
    DynamicCaps caps{};
    bool skip_motion_blur = false;
    bool skip_dof_spartan = false;
    bool mb_seen_blur = false;
    std::vector<Pos> skipped_draws;
    std::unordered_map<u64, u64> orb_hash_cache;
    static u64 OrbShaderHash(u64 prog, const CpuRead& cpu);
    bool ShouldSkipMotionBlur(const CpuRead& cpu);
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    u32 family = 0;
    bool owned = true;              // instance and device are ours (offline), not the game window's
    std::mutex own_queue_lock;
    std::mutex* queue_lock = &own_queue_lock;  // every queue operation, shared with the presenter when live

    VkPhysicalDeviceMemoryProperties mem{};
    // Integrated GPU with VK_EXT_memory_budget: its device-local heap is a carve-out of the same RAM (1 GiB by default on the
    // Steam Deck), so DeviceType takes the largest heap once that budget is spent.
    bool uma_budget = false;
    bool uma_single_heap = false;  // integrated GPU with one DEVICE_LOCAL+HOST_VISIBLE heap
    bool use_stage_dev = true;     // false on integrated GPUs (host and device heaps are the same RAM): stage -> mirror directly
    bool direct_tables = false;    // single-heap UMA: write draw tables directly into host-visible device memory
    // Heaps PressureAge watches (VK_EXT_memory_budget): all of an integrated GPU's (DeviceType spills between them), the
    // device-local one of a discrete GPU.
    u32 budget_heaps = 0;
    bool compact_memory = false;   // integrated GPU with <= 8 GiB total Vulkan heap (8 GB of RAM) or discrete with <= 4 GiB VRAM: smaller fixed chunks/reserves
    std::atomic<bool> spilled{false};
    std::atomic<bool> alloc_failed{false};
    std::atomic<bool> alloc_pressure{false};   // an allocation failed since the last Submit start (PressureAge)
    std::atomic<bool> budget_pressure{false};  // PressureAge saw every watched heap within an eighth of its budget
    u32 PressureAge();  // frames a texture may go unused under memory pressure, 0 without any
    // Under memory pressure the reserves (empty image blocks, prologue scratch) shrink to what is used.
    bool Pressed() const { return alloc_pressure.load(std::memory_order_relaxed) || budget_pressure.load(std::memory_order_relaxed); }
    // The mirror is larger than the driver's maxBufferSize (AMD, RADV): copies between it and an image go through a temp
    // buffer. Mesa 26.0 RADV read zeros from it in buffer-to-image copies past 4 GB (issue #24: green video, black frame);
    // buffer copies and storage buffer descriptors reach the whole mirror.
    bool image_copy_via_temp = false;
    u32 Type(u32 bits, VkMemoryPropertyFlags want) const {
        for (u32 i = 0; i < mem.memoryTypeCount; ++i)
            if (bits >> i & 1 && (mem.memoryTypes[i].propertyFlags & want) == want) return i;
        return ~0u;
    }
    u32 DeviceType(u32 bits, VkDeviceSize size);  // for `size` more bytes of device memory
    VkDeviceMemory Alloc(VkMemoryRequirements r, VkMemoryPropertyFlags want, bool address);
    // One-off command buffer, waited for.
    template <class F> void Once(F&& f) {
        std::scoped_lock res(res_lock);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cmd_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VkCommandBuffer c;
        vkAllocateCommandBuffers(dev, &ai, &c);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        vkBeginCommandBuffer(c, &bi);
        f(c);
        vkEndCommandBuffer(c);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &c};
        {
            std::scoped_lock lock(*queue_lock);
            vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue);
        }
        vkFreeCommandBuffers(dev, cmd_pool, 1, &c);
    }

    PFN_vkCreateShadersEXT CreateShaders = nullptr;
    PFN_vkGetShaderBinaryDataEXT GetShaderBinary = nullptr;
    u64 binary_seed = 0;  // what a driver binary is valid for: the GPU and driver (shaderBinaryUUID, version), mixed into its cache key
    PFN_vkCmdBindShadersEXT BindShaders = nullptr;
    PFN_vkCmdSetVertexInputEXT SetVertexInput = nullptr;
    PFN_vkCmdSetPolygonModeEXT SetPolygonMode = nullptr;
    PFN_vkCmdSetRasterizationSamplesEXT SetSamples = nullptr;
    PFN_vkCmdSetSampleMaskEXT SetSampleMask = nullptr;
    PFN_vkCmdSetAlphaToCoverageEnableEXT SetAlphaToCoverage = nullptr;
    PFN_vkCmdSetColorBlendEnableEXT SetBlendEnable = nullptr;
    PFN_vkCmdSetColorBlendEquationEXT SetBlendEquation = nullptr;
    PFN_vkCmdSetColorWriteMaskEXT SetWriteMask = nullptr;
    PFN_vkCmdSetDepthClampEnableEXT SetDepthClamp = nullptr;
    PFN_vkCmdSetLogicOpEnableEXT SetLogicOpEnable = nullptr;
    PFN_vkCmdSetDepthClipNegativeOneToOneEXT SetNegativeOneToOne = nullptr;
    PFN_vkCmdBeginConditionalRenderingEXT BeginConditional = nullptr;
    PFN_vkCmdEndConditionalRenderingEXT EndConditional = nullptr;
    PFN_vkDestroyShaderEXT DestroyShader = nullptr;
    // The mirror as index buffer, bound once per 4 GB window with the window's size (VK_KHR_maintenance5; null without it: bound to
    // the buffer's end). A driver that keeps the index count of a binding in 32 bits (AMD: its maxBufferSize is 2 GB; Intel's: 4 GB;
    // the mirror 16 or 32 GB) finds none from a multiple of 4 GB, and every index reads as 0: the windows start four bytes past
    // one (Draw).
    PFN_vkCmdBindIndexBuffer2KHR BindIndex2 = nullptr;
    static constexpr u64 kIndexWindow = 1ull << 32;
    // Diagnostic (debug runs, GOW3_TEMP_LABELS=0 off): pass, block and copy labels for Nsight GPU Trace; the entry points stay
    // null otherwise.
    PFN_vkCmdBeginDebugUtilsLabelEXT temp_label_begin = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT temp_label_end = nullptr;
    void TempLabel(VkCommandBuffer c, const char* s) {
        if (!temp_label_begin) return;
        const VkDebugUtilsLabelEXT l{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, s};
        temp_label_begin(c, &l);
    }
    void TempLabelEnd(VkCommandBuffer c) {
        if (temp_label_end) temp_label_end(c);
    }
    // Diagnostic (debug runs, GOW3_TEMP_FAULT=0 off): checkpoints (kind << 56 | frame << 32 | value) and, on the first lost device, the fault
    // report and each queue's last checkpoints in the notes.
    PFN_vkGetDeviceFaultInfoEXT temp_get_fault = nullptr;
    PFN_vkCmdSetCheckpointNV temp_set_check = nullptr;
    PFN_vkGetQueueCheckpointDataNV temp_get_checks = nullptr;
    std::atomic<bool> temp_lost{false};
    void TempCheck(VkCommandBuffer c, u64 kind, u64 value) {
        if (temp_set_check) temp_set_check(c, reinterpret_cast<void*>(kind << 56 | (frame & 0xFFFFFFull) << 32 | (value & 0xFFFFFFFF)));
    }
    void LoadDeviceFunctions();
    void TempDeviceReport(std::span<const VkQueueFamilyProperties> fp);
    void TempLost(const char* where, VkResult r);
    bool Init(bool validation, std::string& error);  // tools/framecap/offline_device.cpp
    bool Setup(std::string& error);                  // vk_gpu.cpp: everything after the device

    // ---- Locks, stats counters and stall tracking (vk_gpu.cpp) ----
    // Recording threads: one lock per group of shared state, taken in this order when nested (res, then the others).
    std::recursive_mutex res_lock;  // textures, samplers, targets, bindless elements, uploads, frame temporaries, Once
    std::shared_mutex prog_lock;    // shaders, slots, vs_outputs, rect_gs, compiling
    std::mutex note_lock, bind_lock, stage_lock;
    // Stats counters: a recording block counts in its thread and adds the counts once at its end (a counter line written
    // by every thread on every draw stalls them all).
    static inline thread_local u64 counts[9]{};
    void Count(u64& c, u64 n = 1) {
        if (const size_t k = &c - &stats.draws; in_worker && k < std::size(counts)) counts[k] += n;
        else std::atomic_ref(c).fetch_add(n, std::memory_order_relaxed);
    }
    void AddCounts() {
        for (size_t k = 0; k < std::size(counts); ++k)
            if (counts[k]) std::atomic_ref((&stats.draws)[k]).fetch_add(counts[k], std::memory_order_relaxed), counts[k] = 0;
    }
    // This Submit's stalls (Stats::compiles...), from any recording thread; `Stalled` times one.
    std::atomic<u32> stall_compiles{0}, stall_waits{0}, stall_binds{0}, stall_evicts{0};
    std::atomic<u64> stall_wait_ns{0}, stall_bind_ns{0}, stall_evict_ns{0};
    static void Stalled(std::atomic<u32>& n, std::atomic<u64>& ns, auto&& f) {
        if (!DebugEnabled()) return f();
        const auto t = std::chrono::steady_clock::now();
        f();
        n.fetch_add(1, std::memory_order_relaxed), ns.fetch_add(u64((std::chrono::steady_clock::now() - t).count()), std::memory_order_relaxed);
    }

    // ---- Mirror, guest tracking, staging, buffers and draw tables (buffers.cpp) ----
    std::function<bool(u64, u64)> track;                  // live: makes guest ranges the GPU reads resident (SetTracker), thread-safe
    std::function<u64(u64, const void*, u64)> readback;   // live: GPU-written ranges back to guest memory
    // Submit's deferred spans (the prepared arenas: the GPU reads about a third of their bytes), each with its first word in
    // the bitmaps every recording thread marks the 512-byte blocks of Track's ranges in; merged and staged after recording.
    struct Deferred { u64 lo, hi, word; };
    static constexpr u64 kDeferredWords = (17ull << 20 >> 15) + 2;  // 17 MB of pages in two spans
    Deferred deferred[2]{};
    u32 deferred_n = 0;
    u64 marked_lo = 0, marked_hi = 0;  // the spans' bounds (none: 0, 0)
    static inline std::mutex deferred_lock;
    static inline std::vector<u64*> deferred_maps;  // one per thread that tracked, alive with the process
    void MarkDeferred(u64 lo, u64 hi) {
        static thread_local u64* map = [] {
            auto* m = new u64[kDeferredWords]{};
            std::scoped_lock l(deferred_lock);
            deferred_maps.push_back(m);
            return m;
        }();
        for (u32 i = 0; i < deferred_n; ++i)
            if (const Deferred& s = deferred[i]; lo < s.hi && s.lo < hi)
                for (u64 k = s.word * 64 + ((std::max(lo, s.lo) - s.lo) >> 9), end = s.word * 64 + ((std::min(hi, s.hi) - s.lo + 511) >> 9), n; k < end; k += n)
                    n = std::min(64 - (k & 63), end - k), map[k >> 6] |= (n == 64 ? ~0ull : ((1ull << n) - 1)) << (k & 63);
    }
    bool Track(u64 lo, u64 hi) {
        if (!track || hi <= lo) return false;
        if (boom_on) NoteGpuRead(lo, hi);  // TEMP coverage of the offer
        // GuestMemory tracks whole pages, the deferred spans' bitmaps 512-byte blocks. A range this thread already tracked
        // this frame (in pages, in blocks over the spans) is skipped: its last four without a lookup (consecutive draws read
        // the same pages: the data ring), the others by hash (entries tagged with the frame).
        const bool marked = lo < marked_hi && marked_lo < hi;
        const u64 g = marked ? 0x1FF : 0xFFF;
        lo &= ~g, hi = (hi + g) & ~g;
        thread_local u64 seen[4096], last[4][2], last_frame = ~0ull;
        thread_local u32 last_at = 0;
        if (last_frame != frame) std::memset(last, 0, sizeof(last)), last_frame = frame;
        for (const auto& e : last)
            if (e[0] == lo && e[1] == hi) return false;
        last[last_at][0] = lo, last[last_at][1] = hi, last_at = (last_at + 1) & 3;
        const u64 h = ::Gpu::PieceHash::TrackingKey(&lo, 8, hi), tag = h ^ frame * 0x9E3779B97F4A7C15ull;
        if (u64& s = seen[h & 4095]; s == tag) return false;
        else s = tag;
        if (marked) MarkDeferred(lo, hi);
        return track(lo, hi);
    }

    // Guest memory mirror: one sparse buffer, device address of byte 0 = guest address 0.
    VkBuffer mirror = VK_NULL_HANDLE;
    VkDeviceAddress mirror_addr = 0;
    u64 mirror_size = 0;  // kMirrorMax, or kMirrorMin when the driver could not reserve that (set once by SetupMirror)
    u64 page = 0;
    u32 mirror_bits = 0;  // memory types its chunks may have (DeviceType picks one per chunk)
    std::vector<VkDeviceMemory> chunks;
    u64 chunk_size = kChunk;
    u64 chunk_used = kChunk;
    std::vector<u64> bound;  // bitmap of mirror pages with memory
    std::vector<VkSparseMemoryBind> pending_binds;
    std::atomic<u64> mirror_chunks{0}, device_buffers{0}, host_buffers{0};  // debug: Vulkan memory held (bytes)
    // 4 KB piece -> content hash | 1 (0: never written), to skip unchanged uploads. Guest addresses below 32 GB; the
    // table is demand-zero memory, so only the pieces the game uses cost anything.
    static constexpr u64 kPieces = 1ull << 23;
    std::unique_ptr<u64, decltype(&std::free)> written_table{(u64*)std::calloc(kPieces, sizeof(u64)), &std::free};
    u64* written = written_table.get();
    std::unordered_map<u64, u64> written_partial;  // pieces that do not cover their whole 4 KB page, by start address
    struct BlockHashes { u64 witness = 0, hashes[0x1000 / Gpu::PieceHash::kMemoryBlock]{}; };
    Flat<std::unique_ptr<BlockHashes[]>> written_blocks;  // dense page metadata within sparse 1 MB chunks
    void InvalidateBlocks(u64 a, u64 n = 1) {
        for (const u64 end = a + n; a < end;) {
            const u64 hi = std::min(end, (a | u64{0xFFFFF}) + 1);
            if (auto* b = written_blocks.Find(a >> 20))
                for (u64 p = a >> 12; p < ((hi + 0xFFF) >> 12); ++p) (*b)[p & 255].witness = 0;
            a = hi;
        }
    }
    static u64 PieceHash(const u8* p, u64 n) { return Gpu::PieceHash::Hash(p, n); }
    u64& WrittenSlot(u64 a, u64 n) {
        static thread_local u64 beyond;
        if (n == 0x1000 && a >> 12 < kPieces) return written[a >> 12];
        return a >> 12 < kPieces ? written_partial[a] : (beyond = 0);
    }

    // Guest bytes for the mirror, copied once into host-visible staging chunks (reused after the frame's fence).
    struct Pending {
        u64 addr, size, src;
        u32 chunk;
    };
    std::vector<Pending> pending;
    std::vector<Pending> stage_sends;  // extra staging slices copied by RecordSend to stage_dev, not flushed via pending
    size_t sent = 0;  // pending entries already copied to their chunk's VRAM twin (Send)
    u32 stage_at = 0;
    u64 stage_used = 0;
    const u8* Stage(u64 addr, const u8* src, u64 n, u64* hash = nullptr);  // the staged copy
    Pending AllocateStage(u64 n);  // stage_lock held; reserves bytes without scheduling an upload
    void QueueStage(Pending p);    // stage_lock held; schedules only changed bytes
    void RecordSend(VkCommandBuffer c);  // copies the bytes staged since the last send to their chunks' VRAM twins
    void Send();
    void BindPages(u64 lo, u64 hi);
    void Flush(VkCommandBuffer c);
    bool SetupMirror(std::string& error);
    void DestroyMirror();  // mirror, its memory, the draw tables' sets
    u64 RecordReadback(const VkCommandBufferBeginInfo& bi);  // Submit's readback copies, the bytes
    // Debug runs (boom_on), from any thread that stages: torn snapshots (pieces whose staged copy differs from the guest memory
    // right after it) and the ranges skipped because the frame in flight writes them; the end of each Submit moves them to Stats.
    std::mutex tear_lock;
    std::vector<Stats::Tear> tears;
    std::vector<std::pair<u64, u64>> skips;
    void NoteTear(u64 addr, const u8* staged, const u8* live, u64 n);
    // Fault diagnostics (debug runs): every buffer made and freed, by handle, to name a freed one the GPU still wrote.
    struct TempAlloc {
        u64 handle;
        const char* tag;
        u64 size;
        u32 made, freed, slot;
        bool busy;  // the frame in flight had not retired when it was freed
    };
    std::mutex temp_alloc_lock;
    std::vector<TempAlloc> temp_allocs;

    struct Buffer {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        u8* ptr = nullptr;
        VkDeviceAddress addr = 0;
        u64 size = 0;
    };
    Buffer MakeBuffer(u64 size, VkBufferUsageFlags usage, bool host, const char* temp_tag = "?", bool shared = false);
    void FreeBuffer(Buffer& b);
    Buffer tables_host, tables, rb, rb_dev;  // rb: readback of the frame's GPU writes (rb_dev: their VRAM copy, copy engine)
    static constexpr u64 kRbReserve = 16ull << 20;
    u64 rb_used = 0;
    std::vector<Buffer> stage, stage_dev;    // staging chunks of `pending`, and their VRAM twins
    u64 stage_chunk_size = 64ull << 20;
    // Draw tables: per frame slot dynamic uniform buffers over `tables`. Each thread takes 64 KB pieces and fills them alone.
    VkDescriptorSetLayout table_layout = VK_NULL_HANDLE;
    VkDescriptorPool table_pool = VK_NULL_HANDLE;
    VkDescriptorSet table_sets[2]{};  // per frame slot: the VS/CS and PS draw tables over that slot's `tables`
    u64 table_align = 16;             // minUniformBufferOffsetAlignment
    u64 table_bytes = kTableBytes;
    u64 tables_used = 0;
    bool SetupTables();
    u8* TablePtr() const { return (direct_tables ? tables : tables_host).ptr; }
    static inline thread_local u64 table_lo = 0, table_hi = 0;
    static inline thread_local u32 table_frame = ~0u;
    u64 TableAt(u64 n, u64 align) {
        if (table_frame != frame) table_lo = table_hi = 0, table_frame = frame;
        if (u64 at = (table_lo + align - 1) & ~(align - 1); at + n <= table_hi) return table_lo = at + n, at;
        const u64 piece = std::max<u64>(n + align, 64 << 10);
        const u64 lo = std::atomic_ref(tables_used).fetch_add(piece, std::memory_order_relaxed);
        if (lo + piece > table_bytes) return ~0ull;
        table_lo = lo, table_hi = lo + piece;
        const u64 at = (table_lo + align - 1) & ~(align - 1);
        return table_lo = at + n, at;
    }

    // ---- Bindless set, samplers, textures, render targets and tiling (textures.cpp) ----
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    struct Tex {
        VkImage img = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkImageUsageFlags usage = 0;
        VkDeviceSize mem_offset = 0, mem_size = 0;  // mem_size: a range of an image block (ImageMemory), else mem is its own
        VkDeviceSize alloc_size = 0;                // the size of that own allocation
    };
    std::vector<Tex> dummies;
    std::atomic<u64> dedicated_images{0};  // debug: device memory held outside image blocks (bytes)
    bool SetupTextures();
    void DestroyTextures();
    Tex MakeImage(VkImageType type, VkImageViewType vtype, VkFormat fmt, u32 w, u32 h, u32 depth, u32 layers, VkSampleCountFlagBits samples,
                  VkImageUsageFlags usage, VkImageAspectFlags aspect, u32 levels = 1, VkImageCreateFlags flags = 0, bool lasting = false);
    // Image memory, suballocated from 128 MB blocks (first fit, aligned; freed ranges merge): vkAllocateMemory is a kernel
    // call that zeroes the pages, which held the recording for up to milliseconds per new texture. The next block is made
    // on another thread when less than half a block is free; meanwhile, and above 32 MB, an image gets its own allocation.
    static constexpr VkDeviceSize kImageBlock = 128ull << 20;
    VkDeviceSize image_block_size = kImageBlock;
    u32 image_reserve_blocks = 2;
    struct ImageBlock {
        VkDeviceMemory mem;
        u32 type;
        std::map<VkDeviceSize, VkDeviceSize> free;  // offset -> size
        bool lasting = false;  // targets, their copies and the dummies: images no guest memory reloads
        bool draining = false;  // CompactImageBlocks empties it: no new image goes there
    };
    std::mutex image_lock;
    std::vector<ImageBlock> image_blocks;
    VkDeviceSize image_free[2]{};  // free bytes of the texture blocks [0] and the lasting ones [1]
    std::future<void> image_growing;
    void ImageMemory(const VkMemoryRequirements& r, Tex& t, bool lasting);
    void FreeImage(const Tex& t);
    bool ReclaimEmptyImageBlocks();
    // `usage`: the image's, when it is a storage image too: a view in a format without storage support (SRGB) leaves it out.
    VkImageView View(VkImage img, VkImageViewType vtype, VkFormat fmt, VkImageSubresourceRange range, VkComponentMapping map, VkImageUsageFlags usage = 0);
    // Next free element of a bindless array (element 0 holds the dummy image / default sampler).
    u32 Bindless(u32 binding, VkImageView view, VkSampler smp);

    struct Target {
        u64 base;
        u32 w, h, pitch, height, samples;
        VkFormat fmt;
        bool depth, stencil;
        Tex t;
        mutable u32 used = ~0u, rendered = ~0u;
        std::map<u64, u32> views;                        // (binding, view format) -> bindless element, when sampled as a texture
        mutable std::map<VkFormat, VkImageView> attach;  // attachment views by CB format (same memory written as UNORM and SRGB)
        mutable bool bgra = false;                       // last written with a B, G, R, A export swizzle (diagnostic dumps only)
        Tex stencil_copy{};
        Tex storage_copy{};  // a dispatch's storage image when the driver gives the target's format none (Intel SRGB): copied in and back
        std::vector<VkImageView> owned_views;            // image views created for `views`
    };
    std::mutex targets_lock;  // `targets` insertions vs FindTarget from other threads
    std::map<std::tuple<u64, u32, u32, VkFormat, u32>, Target> targets;
    // Bases with a target drawn since their memory was last rewritten (`rendered` not ~0u): a T# over one is looked up
    // among every target sharing it, the others are plain textures.
    std::unordered_set<u64> target_at;
    template <class F> void TargetsAt(u64 base, F&& f) {  // `targets` is ordered by base
        for (auto it = targets.lower_bound({base, 0u, 0u, VkFormat(0), 0u}); it != targets.end() && it->second.base == base; ++it) f(it->second);
    }
    std::unordered_map<u64, std::pair<Target*, u32>> stencil_at;
    // width | height << 16 from the NOP hint after the CB / DB setup (0: none)
    static inline thread_local u32 cb_extent[8]{}, db_extent = 0;
    VkSampleCountFlags msaa_samples = VK_SAMPLE_COUNT_1_BIT;  // sample counts supported across color, depth, stencil and sampled views
    u32 ClampSamples(u32 requested) const {
        u32 s = std::max(1u, requested);
        while (s > 1u && !(msaa_samples & s)) s >>= 1;
        return s;
    }
    const Target* GetTarget(u64 base, u32 pitch, u32 height, VkFormat fmt, bool depth, bool stencil, u32 extent, u32 samples, u32 tile);
    void RegisterStencil();
    VkImageView TargetAttachment(const Target& t, VkFormat fmt);
    u32 TargetBlendElement(const Target& t);
    u32 TargetView(Target& rt, u64 key, u32 binding, VkImage img, VkImageViewType vtype, VkFormat fmt, VkImageAspectFlags aspect,
                   VkComponentMapping map, VkImageUsageFlags usage, const char* fail_note = nullptr);
    u32 TargetTexture(Target& rt, const u32* t, int kind, u32 binding, u64& sub);
    u32 StencilTexture(Target& target, const u32* t, u32 binding);
    u32 StencilMsTexture(Target& target, u32 binding);
    void WriteBackTarget(const Target& t, u32 pitch, VkRect2D area);

    // Guest textures by T#: image filled from the mirror (detiled by tiling.comp when tiled) at the start of the frame.
    struct Layout {
        u32 size, pitch, height, offset;  // tiling.comp MipInfo
    };
    struct TilingInfo {
        u32 bank_swizzle, num_slices, num_mips;
        Layout mips[16];
        u32 image_width, image_height, range_begin, range_end, block_scale, is_volume;
    };
    // Bindless views of an image: image, binding, format, swizzle | base level << 12 | base layer << 16.
    using ViewKey = std::tuple<VkImage, u32, VkFormat, u32>;
    struct ViewEntry {
        VkImageView view = VK_NULL_HANDLE;
        u32 element = 0;
        std::shared_ptr<Tex> image;  // a surface's image, held until its views go (Reap destroys the views first)
    };
    using Views = std::map<ViewKey, ViewEntry>;
    // Image of the T#s over one memory layout (Describe): as big as the largest of them.
    struct Surface {
        std::weak_ptr<Tex> t;
        VkImage img = VK_NULL_HANDLE;
        u32 w = 0, h = 0, users = 0;
        bool cube = false;
        bool foreign = false;  // read through another layout once: its writes go back to the mirror at once
        bool growing = false;  // its image grew this frame: the same until Regrow
        u32 current = ~0u;     // the last frame a T# as big as its image was checked: the image holds the CPU writes staged until then
        u32 valid_frame = ~0u, valid_w = 0, valid_h = 0;  // top-left [0..valid_w) x [0..valid_h) current as of valid_frame
        Views views;           // its T#s' (PruneSurfaceViews drops those of images none of them holds)
    };
    std::map<std::tuple<u64, u32, u32, u32, u32, u32, u32>, Surface> surfaces;  // base, tile, bits, pitch, layers, levels, extent if mipped
    bool regrow = false;  // a surface's image grew this frame
    void Regrow();
    struct GuestTex {
        Tex t;  // its image's handles (shared: the surface's)
        std::shared_ptr<Tex> shared;
        Surface* surf = nullptr;
        VkFormat fmt = VK_FORMAT_UNDEFINED;
        u64 base = 0, size = 0;
        u32 w = 1, h = 1, d = 1, layers = 1, guest_layers = 1, levels = 1, img_levels = 1, tile = 0, bits = 0, checked = ~0u;
        u32 ActiveMipSize(u32 m) const {
            return guest_layers > layers ? u32(u64(mips[m].size) / guest_layers * layers) : mips[m].size;
        }
        u32 iw = 1, ih = 1;  // its image's extent
        u64 synced = 0;      // write_seq its contents reflect
        u64 video_serial = 0;
        bool block = false, volume = false, storage = false, fresh = true;
        bool flat = false;    // single-sampled 2D (array): can be the top-left part of a bigger image
        bool reload = false;  // in an image other T#s filled: its part uploaded at the next Check
        VkComponentMapping map{};
        Layout mips[16]{};
        VkImageView mip_views[16]{};
        Views views;  // block-coded and volume T#s' (the others use their surface's)
        std::vector<VkImageView> owned;  // mip views
    };
    VkImageView MipView(GuestTex& g, u32 m, bool storage);
    std::unordered_map<u64, GuestTex> textures;
    std::vector<GuestTex*> uploads;  // this frame
    std::unordered_map<u64, u32> samplers;  // S# -> bindless element
    std::vector<VkSampler> owned_samplers;
    std::vector<VkImageView> owned_views;
    u32 next_element[kStorageBinding + 18]{};
    std::vector<u32> free_elements[kStorageBinding + 18];  // released by EvictTextures
    std::vector<std::pair<u32, u32>> pending_elements;     // {binding, element} freed while `flight` may still use them; drained on Retire()
    void DrainRetiredElements();
    bool evict_soon = false;  // a bindless array (evict_binding) filled up on a worker thread: evictions at the next Submit start
    u32 evict_binding = 0;
    std::vector<std::pair<u64, u64>> dirty;  // guest ranges written for this frame, sorted
    // 4 KB piece -> the frame whose `dirty` last held it (demand-zero, as `written`): Check finds the writes of the frames a
    // texture skipped.
    std::unique_ptr<u32, decltype(&std::free)> staged_table{(u32*)std::calloc(kPieces, sizeof(u32)), &std::free};
    u32 frame = 0;
    // Guest textures unused for more than `age` frames: bindless elements freed (no frame samples them), views and images
    // destroyed by Reap once the frame being recorded has retired (it and the frame in flight may still copy from them).
    void EvictTextures(u32 age, bool defer_elements = false);
    // Texture blocks left mostly empty (a level's textures went, a few survivors hold each one): the textures of the emptiest
    // are evicted, loaded again from guest memory on their next use into the fuller blocks, and the emptied blocks go back.
    void CompactImageBlocks();
    std::vector<const GuestTex*> doomed;  // CompactImageBlocks' picks for EvictTextures, sorted
    void Bury(const ViewKey& k, ViewEntry& v, bool defer_elements = false);  // element freed (or deferred), view (then image) to Reap
    void PruneSurfaceViews(bool defer_elements = false);
    // Evicted textures' handles: this frame's, then older ones (`dying`), which go once their frames retired.
    struct Graves {
        std::vector<VkImageView> views;
        std::vector<Tex> own;
        std::vector<std::shared_ptr<Tex>> shared;  // the last owner's deleter destroys the surface's image
        bool empty() const { return views.empty() && own.empty() && shared.empty(); }
        void Append(Graves& o) {
            views.insert(views.end(), o.views.begin(), o.views.end()), own.insert(own.end(), o.own.begin(), o.own.end());
            shared.insert(shared.end(), std::make_move_iterator(o.shared.begin()), std::make_move_iterator(o.shared.end()));
            o = {};
        }
    } graves, dying;
    std::future<void> reaping;
    std::atomic<u32> reaped{0};        // objects destroyed (frame recorder)
    std::atomic<u64> reap_max_ns{0};   // the longest of them
    // After the wait for the frame in flight: the textures evicted while it was recorded go to another thread (destroying an
    // image, or freeing memory of its own, is a kernel call), one object at a time with a pause: destroyed at once, a batch
    // held the frame's queue submit or recording for up to 5 ms. While that thread is busy they wait for a later frame.
    void Reap();
    // Detiler: tiling.comp per (tile mode, bits per element), push descriptors 0 tiled, 1 linear, 2 TilingInfo.
    std::string tiling_src;
    VkDescriptorSetLayout tiling_set = VK_NULL_HANDLE;
    VkPipelineLayout tiling_layout = VK_NULL_HANDLE;
    std::unordered_map<u32, VkPipeline> detilers;
    PFN_vkCmdPushDescriptorSetKHR PushDescriptors = nullptr;
    Buffer scratch;  // detiled texels of the frame's prologue uploads, per frame slot: grown while the frame in flight writes the other
    u64 scratch_reserve = 128ull << 20, scratch_used = 0;
    // Storage images a dispatch or draw writes go back to the mirror; any other image over that memory is uploaded again
    // before its next use (mid-frame here, at the next frame's start through `dirty`).
    struct GpuWrite {
        u64 lo, hi, seq;
        GuestTex* by;
        bool lazy = false;          // left in its image (WriteBack): the mirror gets it only through Materialize
        bool guest_synced = false;  // guest RAM and WrittenSlot already match mirror (RecordReadback skips it)
    };
    std::vector<GpuWrite> gpu_writes;  // this frame
    std::vector<GpuWrite> last_writes;  // the last frame's: textures over them in another image are uploaded again
    std::vector<GpuWrite> lazy_writes;  // earlier frames' lazy ones not written back yet (one per image and range)
    std::vector<std::pair<GpuWrite, bool>> prologue_lazy;  // those the frame's prologue writes back before its uploads (Check; `learn`)
    void QueuePrologueLazy(const GpuWrite& w, bool learn) {
        for (auto& p : prologue_lazy)
            if (p.first.by == w.by && p.first.lo == w.lo) return void(p.second |= learn);
        prologue_lazy.push_back({w, learn});
    }
    u64 write_seq = 0, frame_seq = 0;  // frame_seq: write_seq when the frame started
    std::unordered_set<u64> storage_draw_progs;
    std::atomic<bool> has_storage_draw_progs{false};
    static inline thread_local u64 draw_vs_addr = 0, draw_ps_addr = 0;
    // Images made this frame (targets): to GENERAL and cleared in the frame's prologue, before any block
    // that uses them, whichever thread made them.
    std::vector<std::pair<VkImage, VkImageAspectFlags>> fresh;
    // Recording threads (Submit): a block sees the writes recorded before it was cut (`horizon` of `gpu_writes`) plus its
    // own (`own_writes`); `synced_here` tracks the images it uploaded again. The serial walk sees everything.
    static inline thread_local size_t horizon = ~size_t(0);
    static inline thread_local std::vector<GpuWrite> own_writes;
    static inline thread_local std::unordered_map<const GuestTex*, u64>* synced_here = nullptr;
    static inline thread_local std::vector<std::pair<GuestTex*, u32>> stores;  // storage images (and level) of the current dispatch or draw
    // Consecutive dispatches share one barrier and write-back: the storage images of those recorded since the last one, not
    // written back yet, and the guest textures they sample (`reads`: the current dispatch's). A slot reading their writes,
    // or writing over what they read or wrote, flushes them first (AgainstPending); the current command's own slots never do.
    static inline thread_local std::vector<std::pair<GuestTex*, u32>> pend_stores;
    static inline thread_local std::vector<const GuestTex*> pend_reads, reads;
    static inline thread_local bool dispatching = false;
    // Chunked device arena of mid-frame uploads and write-backs: slices from reusable chunks, trimmed once quiet after a spike.
    struct TempSlice {
        VkBuffer buf = VK_NULL_HANDLE;
        u64 offset = 0;
    };
    std::vector<Buffer> frame_temp;
    u32 temp_at = 0;
    u64 temp_used = 0, temp_chunk_size = 32ull << 20, temp_align = 256;
    u32 temp_quiet = 0, scratch_quiet = 0, stage_quiet = 0, rb_quiet = 0;
    TempSlice Temp(u64 size);
    void NoteFrameBuffers();  // before SwapFrame: updates the submitted slot's quiet counters
    void TrimFrameBuffers();  // after SwapFrame: frees the retired slot's oversized or extra transient buffers
    static inline thread_local bool disp_pending = false;
    void FlushStores() {
        if (!disp_pending) return;
        disp_pending = false, pend_reads.clear(), Barrier(cmd);
        std::swap(stores, pend_stores), WriteBack(), std::swap(stores, pend_stores);  // the current command's stay
    }
    void AgainstPending(const GuestTex& g, bool storage) {
        auto over = [&](const GuestTex* p) { return p->base < g.base + g.size && g.base < p->base + p->size; };
        if (std::any_of(pend_stores.begin(), pend_stores.end(), [&](auto& s) { return over(s.first); }) ||
            (storage && std::any_of(pend_reads.begin(), pend_reads.end(), over)))
            FlushStores();
        if (dispatching && !storage) reads.push_back(&g);
    }
    static inline const bool temp_copylog = DebugDeep("GOW3_TEMP_COPYLOG");
    static std::string TempWriter(const GpuWrite& w, const GuestTex& g);
    static void TempCopy(const char* what, const GuestTex& g, const std::string& why = {});
    u32 Texture(const u32* t, const Gcn::SlotSpec& spec, bool storage, bool stencil_resolve);
    u32 Sampler(const u32* s);
    bool Describe(GuestTex& g, const u32* t);
    void Resync(GuestTex& g);
    void Check(GuestTex& g);
    VkPipeline Detiler(u32 tile, u32 bits, u32 mode = 0);
    void DispatchDetile(VkCommandBuffer c, u32 mip, u32 bytes, u32 bits);
    void Upload(VkCommandBuffer c, std::vector<GuestTex*>& list, Buffer* buf, u64 reserve, bool frag = false);
    bool Store(VkCommandBuffer c, GuestTex& g, u32 m);
    void ImageToMirror(VkCommandBuffer c, VkImage img, VkBufferImageCopy bc, u64 to, u64 row, u64 pitch, u32 slices);
    void Materialize(VkCommandBuffer c, const GpuWrite& w, bool learn);
    void WriteBack();

    // ---- Per-thread lookup memo (textures.cpp, shader_runtime.cpp) ----
    // Per recording thread: lookups already made, so the hot path takes no lock. Targets and samplers live as long as the
    // renderer; textures last until an eviction or a new render target (which may cover a texture's memory) bumps `epoch`.
    struct MemoTex {
        GuestTex* g;  // null: a render target's view
        u32 e;        // bindless element
        VkImage rt;   // the render target's image
        u64 sub;      // tex_sub of the binding
        const Target* target;
    };
    // A program's last distinct inputs on this thread and what Shader made of each. The same inputs give the same
    // translation key, shader and slot elements (`epoch` is one of them): a draw that repeats one skips the specs, the keys
    // and the slots' lookups.
    struct Recent {
        std::vector<u32> in;  // Shader's `inputs` (its first n)
        u64 out_mask = 0;
        ShaderRef s{};
        std::vector<MemoTex> slots;  // per slot: Texture's `tex_used`, a sampler's element
    };
    struct ProgUse {
        const ProgSlots* ps = nullptr;
        u8 order[4]{0, 1, 2, 3};  // most recent first
        Recent recent[4];
    };
    struct TargetKey {  // GetTarget's: base, pitch, height, format (color: texel bytes), samples
        u64 base;
        u32 pitch, height, fmt, samples;
        bool operator==(const TargetKey&) const = default;
    };
    struct MemoTarget {
        TargetKey key;  // the whole key: entries are found by its hash
        const Target* t;
    };
    struct MemoAttach {
        const Target* t = nullptr;
        VkFormat fmt = VK_FORMAT_UNDEFINED;
        VkImageView view = VK_NULL_HANDLE;
    };
    struct Memo {
        const Impl* owner = nullptr;
        u64 epoch = ~0ull;
        u64 pipeline_epoch = ~0ull;
        Flat<MemoTarget> targets;
        Flat<MemoAttach> target_attach;
        Flat<u32> samplers;
        Flat<MemoTex> tex;                       // T# hash + binding
        Flat<std::pair<ShaderRef, u64>> shaders; // translation key -> shader, VS output mask
        Flat<CompiledPipeline> pipelines;        // GraphicsPipelineKey::Hash() of the draw's raw state -> main/square VkPipeline
        Flat<ProgUse> progs;                     // program key -> its slots and last inputs
    };
    static thread_local Memo memo;
    u64 epoch = 0, pipeline_epoch = 0;
    Memo& Memos() {
        if (memo.owner != this) memo = {}, memo.owner = this;
        if (const u64 e = std::atomic_ref(epoch).load(std::memory_order_acquire); memo.epoch != e)
            memo.tex.clear(), memo.targets.clear(), memo.target_attach.clear(), memo.epoch = e;
        if (const u64 pe = std::atomic_ref(pipeline_epoch).load(std::memory_order_acquire); memo.pipeline_epoch != pe)
            memo.pipelines.clear(), memo.pipeline_epoch = pe;
        return memo;
    }
    static inline thread_local bool table_ps = false;  // the table being built is a pixel shader's
    // Set by Texture when the T# is the top-left part of the image it binds: w | h << 16 | image w << 32 | image h << 48.
    static inline thread_local u64 tex_sub = 0;
    // Texture's last call, for Recent: the memo entry it used (all null: nothing to redo); `tex_keep` false: a result a
    // repeat must not skip (a Note after the checks).
    static inline thread_local MemoTex tex_used{};
    static inline thread_local bool tex_keep = true;
    u32 Reuse(const MemoTex& m, const u32* t, bool storage);

    // ---- Shaders: translation, compilation, caches and discovery (shaders/) ----
    Gcn::Compiler compiler;
    std::unordered_map<u64, ShaderRef> shaders;  // translation key -> shader/module/compute pipeline (empty: failed)
    std::unordered_map<u64, ProgSlots> slots;    // program key -> slots
    // Live: the content hashes of the 4 KB pieces under [a, a + n), tracked first so that a CPU write there reaches them
    // by the next frame; 0 when unknown (offline, or a piece not mirrored whole).
    u64 Version(u64 a, u64 n) {
        if (!track || !n) return 0;
        Track(a, a + n);
        u64 v = 0;
        for (u64 p = a >> 12; p <= (a + n - 1) >> 12; ++p)
            if (const u64 w = p < kPieces ? std::atomic_ref(written[p]).load(std::memory_order_relaxed) : 0; !w) return 0;
            else v = (v ^ w) * 0x9E3779B97F4A7C15ull;
        return v;
    }
    static inline thread_local Flat<ProgMemo> prog_memo;   // program address | stage -> this frame's code, hash, fetch SGPR
    static inline thread_local Flat<ProgMemo> fetch_memo;  // fetch shader address -> this frame's code and hash
    static inline thread_local u32 prog_memo_gen = ~0u;    // GuestMemory::Generation() when prog_memo/fetch_memo were populated
    std::unordered_map<u64, ShaderRef> rect_gs;            // VS output mask -> rect list geometry shader
    ShaderRef clear_rect_ref{};                            // the SDK clear's four strip vertices
    bool HasClearRectVs() const { return bool(clear_rect_ref); }
    std::vector<u32> clear_rect_spirv;
    std::unordered_map<u64, u64> vs_outputs;               // VS translation key -> output mask
    std::unordered_map<u64, Variant> variants;
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    std::atomic<u32> pipeline_cache_dirty{0};  // pipelines compiled since the last save
    std::mutex pipeline_cache_save_lock;
    size_t pipeline_cache_bytes = 0;  // the last save's (or load's) size, under pipeline_cache_save_lock
    std::unordered_map<u64, CompiledPipeline> pipelines;   // GraphicsPipelineKey::Hash() -> main/square VkPipeline (under prog_lock)
    std::unordered_set<u64> gpl_lto_queued;                // fast-linked GPL pipelines queued for background LTO (under prog_lock)
    std::vector<VkPipeline> retired_pipelines;             // fast-linked handles replaced by LTO, destroyed on shutdown (under prog_lock)
    // Tier 2 (Gpl): 4-part Graphics Pipeline Library caches (under prog_lock).
    VkPipeline gpl_vi[3]{};                              // Vertex Input Interface: POINT_LIST, LINE_LIST, TRIANGLE_LIST
    std::unordered_map<u64, VkPipeline> gpl_prerast;     // Pre-Rasterization library cache (VS+GS or non-default static flags)
    std::unordered_map<u64, VkPipeline> gpl_frag_msaa;   // Fragment Shader library cache when samples > 1 or static alphaToCoverageEnable is active
    std::unordered_map<u64, VkPipeline> gpl_foi;         // Fragment Output Interface library cache
    // A program's translation input and key under the current registers (no GPU work).
    struct Prepared {
        Gcn::Input in;
        bool motion_blur = false;
        bool stencil_resolve = false;
        bool dof_composite = false;
        bool linear_copy = false;
        std::vector<Gcn::SlotSpec> spec;
        std::vector<const u32*> desc;   // descriptor of each slot (null: unreadable)
        std::vector<const u32*> fdesc;  // V# of each format load (null: unreadable)
        const std::vector<Gcn::Slot>* slots = nullptr;
        const std::vector<Gcn::Path>* paths = nullptr;
        const std::vector<Gcn::Const>* consts = nullptr;
        ProgUse* use = nullptr;  // until the thread's next new program key
        u64 pkey = 0, key = 0;
        u64 stored = 0;  // bit k: the buffer of format V# k is written by the program
        int depth_resolve_slot = -1;
        bool rect = false;
    };
    static thread_local Prepared prep;  // Shader's, reused (its vectors keep their capacity)
    static inline thread_local std::vector<u32> vs_table, ps_table;  // Draw's, reused
    static constexpr u32 kInputs = 512;
    static inline thread_local u32 inputs[kInputs];                  // Shader's: what makes its variant and slot elements
    static inline thread_local std::vector<MemoTex> slot_used;       // Shader's: each slot's, for Recent
    ShaderRef Shader(Gcn::Stage stage, u64 prog, const u32* user, const CpuRead& cpu, u32 rsrc1, u32 rsrc2, u64& out_mask,
                     bool rect, std::vector<u32>& table, u32 blend = 0, bool identified = false);
    static const u32* Locate(const Gcn::Path& path, u32 n, const u32* user, const CpuRead& cpu) {
        const u32* w = path.reg + (path.depth ? 2 : n) <= 16 ? user + path.reg : nullptr;
        for (u32 d = 0; d < path.depth && w; ++d)
            w = (const u32*)cpu(((u64)(w[1] & 0xFFFF) << 32 | w[0]) + path.offset[d], d + 1 < path.depth ? 8 : n * 4);
        return w;
    }
    // Prepare in two halves: the program's code, key and slots with their descriptors (Identify), then the input's
    // registers, the slots' specs and the translation key (Specify), which a repeated input skips.
    bool Identify(Gcn::Stage stage, u64 prog, const u32* user, const CpuRead& cpu, u32 rsrc1, u32 rsrc2, bool rect, Prepared& p);
    void Specify(Gcn::Stage stage, Prepared& p);
    void SpecifySlots(Prepared& p);
    ShaderRef Translated(Gcn::Stage stage, u64 prog, const CpuRead& cpu, Prepared& p, u64& out_mask);  // compiled, or compiled now
    // Translation and compilation, on any thread: `code` and `fetch` are copies the input's spans point at.
    Compiled Compile(Gcn::Stage stage, u64 prog, u64 key, Gcn::Input in, std::vector<u32> code, std::vector<u32> fetch, std::vector<Gcn::SlotSpec> spec);
    // A shader the player's runs used, made from its driver binary when the game opens its WAD (ShaderCache::prepare).
    u32 Prepare(u64 key, u64 binary, u32 stage);
    // Parallel compilation: the first shader a frame lacks walks the rest of the frame (on a copy of the registers) and
    // compiles every program it will need on worker threads; each draw then waits only for its own.
    std::unordered_map<u64, std::shared_future<Compiled>> compiling;
    std::unordered_map<u64, std::shared_future<CompiledPipeline>> compiling_pipelines;
    // Compile on the workers; `miss`: a draw needs it now (else Discover's, queued behind).
    // `waited`: a draw waits for it (a driver binary's load), so it runs at normal priority even in Async mode.
    std::shared_future<Compiled> Async(Gcn::Stage stage, u64 prog, const Prepared& p, bool miss, bool waited = false);
    ShaderRef ResolveCachedModule(Gcn::Stage stage, u64 key);
    // `probe`: first look for it in the pipeline cache (a load, no compile); `urgent`: a draw needs it now (else Discover's).
    std::shared_future<CompiledPipeline> AsyncPipeline(u64 h, ShaderRef vs, ShaderRef ps, ShaderRef gs, FixedPipelineState state, bool probe, bool urgent);
    bool ExtractDiscoverPipelineState(bool rect, bool rect_vs, bool has_ps, FixedPipelineState& out) const;
    void EnqueuePipeline(u64 vs_key, u64 ps_key, bool has_gs, const FixedPipelineState& raw_state);
    // The compile workers: tasks by priority (0: a draw needs it now, 1: found ahead by Discover).
    std::mutex task_lock;
    std::condition_variable task_cv;
    std::deque<std::function<void()>> tasks[2];
    std::vector<std::thread> compilers;
    bool task_stop = false;
    void Queue(std::function<void()> task, u32 priority);
    void StopCompilers();
    void AotBuild();
    // shader_contract.cpp: the draw configurations the catalog implies, as recipes (translation keys the runtime would compute),
    // and the pipelines they imply (gated on PS programs the recipes must meet, scene sample count 0).
    void Contracts(const Catalog::Data& data, std::vector<Contracted>& out, std::vector<ContractedPipeline>* pipelines_out);
    static inline thread_local bool waiting = false;  // the shader a draw asked for is still compiling (Jit::Mode::Async)
    std::mutex compiled_lock;
    std::vector<Stats::Compile> compiled;
    static inline thread_local bool block_discovered = false;  // the parallel block this thread records
    // The programs from the current packet up to `to` go to the compile threads; `urgent`: only those with a driver binary
    // (a short load), ahead of the prefetches.
    void Discover(const CpuRead& cpu, Pos to, bool urgent);
    u64 Enqueue(Gcn::Stage stage, u64 prog, const u32* user, const CpuRead& cpu, u32 rsrc1, u32 rsrc2, bool rect, bool urgent);
    ShaderRef RectGs(u64 mask);
    static bool ClearVs(std::span<const u32> code);
    bool SetupClearRect(std::string& error);
    bool RectVs(u32 op, const u32* b, const CpuRead& cpu) const;
    VkShaderEXT Create(const std::vector<u32>& spirv, VkShaderStageFlagBits stage, VkShaderStageFlags next);
    VkShaderEXT CreateFrom(VkShaderCodeTypeEXT type, const void* code, size_t bytes, VkShaderStageFlagBits stage, VkShaderStageFlags next);
    u64 BinaryKey(u64 key) const { return ::Gpu::PieceHash::DriverBinaryKey(&key, sizeof(key), binary_seed); }
    std::vector<u8> Binary(VkShaderEXT s);  // the driver's binary of a shader object (empty: not available)
    bool ValidateNativePipelineCacheData(std::span<const u8> data) const;
    void CreatePipelineCache(std::span<const u8> initial_data = {});
    void LoadPipelineCache();
    void SavePipelineCache();  // at the exit and the startup AOT's end only
    void DestroyShaderRef(const ShaderRef& r);
    VkShaderModule CreateShaderModule(std::span<const u32> spirv);
    VkPipeline CreateComputePipeline(std::span<const u32> spirv, bool fail_on_compile = false);
    std::vector<VkDynamicState> BuildDynamicStates() const;
    VkPipeline CreateGraphicsPipelineRaw(const ShaderRef& vs, const ShaderRef& ps, const ShaderRef& gs,
                                         const FixedPipelineState& state, bool fail_on_compile, bool square_pass);
    CompiledPipeline CreateMonolithicPipeline(const ShaderRef& vs, const ShaderRef& ps, const ShaderRef& gs,
                                              const FixedPipelineState& state, bool fail_on_compile);
    void SetupGplGlobals();
    VkPipeline CreateGplPreRaster(VkShaderModule vs_mod, VkShaderModule gs_mod, bool neg_one, bool depth_clamp);
    VkPipeline CreateGplFragShader(VkShaderModule ps_mod, u32 samples, bool alpha_to_coverage = false);
    VkPipeline CreateGplFragOutput(const FixedPipelineState& state, bool square_pass);
    VkPipeline GetGplPreRaster(const ShaderRef& vs, const ShaderRef& gs, const FixedPipelineState& state);
    VkPipeline GetGplFragShader(const ShaderRef& ps, const FixedPipelineState& state);
    VkPipeline GetGplFragOutput(const FixedPipelineState& state, bool square_pass);
    VkPipeline LinkGplPipeline(const VkPipeline (&libs)[4], bool lto, bool fail_on_compile);
    CompiledPipeline LinkGplPair(const VkPipeline (&main_libs)[4], VkPipeline foi_square, bool lto, bool fail_on_compile);
    void EnqueueGplLto(u64 h, const VkPipeline (&libs)[4], VkPipeline foi_sq);
    CompiledPipeline GetGraphicsPipeline(const ShaderRef& vs, const ShaderRef& ps, const ShaderRef& gs,
                                         const FixedPipelineState& raw_state, const CpuRead& cpu);
    // TEMP pipeline log (debug runs, GOW3_TEMP_PIPELOG=0: off): every distinct draw state (programs, raw state) with its frames
    // and draws, and how each pipeline of this GPU came to be (origin, compile time, draws that waited for it). Written at exit
    // to pipelines-<time>.bin, read by tools\plstat.
    enum PlOrigin : u32 { kPlUnknown, kPlWarm, kPlHit, kPlAheadHit, kPlAhead, kPlMiss, kPlGplLto, kPlGplLink, kPlTrace };
    struct TempPlState {
        u64 vs, ps, gs, canon;
        FixedPipelineState raw;
        u32 first_frame, frames, last_frame, flags;  // flags: 1 VS, 2 PS in the shader cache at first use
        u64 draws;                                   // a frame's first draw at once, its others at the thread's next frame with it
    };
    struct TempPlMade { u32 origin = kPlUnknown, us = 0, made_frame = ~0u, waits = 0, wait_frame = ~0u, pad = 0; };
    struct TempPlLocal { TempPlState* s; u32 frame, draws; };
    static inline const bool temp_pipelog = DebugAuto("GOW3_TEMP_PIPELOG");
    static inline thread_local Flat<TempPlLocal> temp_pl_local;  // raw key -> its state, this frame's draws on this thread
    std::mutex temp_pl_lock;
    std::deque<TempPlState> temp_pl_states;
    std::unordered_map<u64, TempPlState*> temp_pl_by_raw;
    std::unordered_map<u64, TempPlMade> temp_pl_made;  // canonical key -> how it was made
    void TempPlUse(const GraphicsPipelineKey& k, u64 rh);  // k: the draw's raw state, rh its hash
    void TempPlOrigin(u64 h, PlOrigin origin, u32 us);
    void TempPlWait(u64 h);
    void TempPlWrite();
    // Keeps `pl` as the pipeline of `h`; one another thread kept first wins (this one is destroyed).
    CompiledPipeline KeepPipeline(u64 h, CompiledPipeline pl, PlOrigin origin, u32 us = 0);
    void WarmPipelines();
    std::once_flag warm_once;
    std::vector<GraphicsPipelineKey> warm_recipes;  // the pipeline recipes for this GPU, canonical
    VkPipeline IndexProbePipeline(const FixedPipelineState& state);

    // ---- Barriers (hazards.cpp) ----
    static thread_local Hazards hz;
    // Everything recorded so far before what follows. `frag`: before its fragment work, compute and transfers only (the
    // vertex stages may start earlier): for image hazards (Hazards tracks the vertex shaders' samples). Shaders write only
    // images (buffer stores are not translated): the mirror changes through write-back copies alone.
    static void Barrier(VkCommandBuffer c, bool frag = false, std::source_location temp_at = std::source_location::current());
    static void StageBarrier(VkCommandBuffer c, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                             VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access);
    // Between two parts of one pass recorded apart: attachment writes before attachment accesses, as within the pass (no
    // `flags`: between passes that share an attachment; `war`: also orders prior shader reads before attachment writes).
    static void PassBarrier(VkCommandBuffer c, VkDependencyFlags flags = VK_DEPENDENCY_BY_REGION_BIT, bool war = false);
    static void ToGeneral(VkCommandBuffer c, VkImage img, VkImageAspectFlags aspect);
    void EndRendering(bool barrier = true);
    // Diagnostics (debug runs): the frame's full and attachment barriers, the full ones by source line (GOW3_TEMP_BARSITES=1).
    static inline std::atomic<u64> temp_bars[2]{};
    static inline std::mutex temp_site_lock;
    static inline std::map<u32, u64> temp_sites;
    static inline const bool temp_barsites = DebugDeep("GOW3_TEMP_BARSITES");

    // ---- Frame: command buffers, queues and the frame in flight (vk_gpu.cpp) ----
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    // The frame's draws; its mirror and texture uploads; its staged bytes to VRAM (`send_cmd`) and its table copy, on the
    // second queue (`xfer`) while the frame before still renders (both per frame). The draws wait for `copied`.
    VkCommandBuffer primary = VK_NULL_HANDLE, copy_cmd = VK_NULL_HANDLE, xfer_cmd = VK_NULL_HANDLE, send_cmd = VK_NULL_HANDLE;
    static inline thread_local VkCommandBuffer cmd = VK_NULL_HANDLE;  // the one this thread records into
    static inline thread_local bool in_worker = false;                // recording a parallel block
    VkQueue xfer = VK_NULL_HANDLE;
    std::mutex own_xfer_lock;
    std::mutex* xfer_lock = &own_xfer_lock;
    VkSemaphore copied = VK_NULL_HANDLE, bound_sem = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    std::vector<VkBufferCopy> back;  // GPU writes copied to `rb`: guest address, rb offset, size
    // Copy engine (a transfer-only family): `xfer` is its first queue; on its second (`rbq`) the frame's GPU writes go
    // from `rb_dev` (copied there in VRAM at the frame's end) to the host once the frame signals `rendered`.
    u32 copy_family = ~0u;
    VkCommandPool copy_pool = VK_NULL_HANDLE;
    VkQueue rbq = VK_NULL_HANDLE;
    VkCommandBuffer rb_cmd = VK_NULL_HANDLE;
    VkSemaphore rendered = VK_NULL_HANDLE;
    // One frame stays in flight while the next one records: its share of the per-frame resources (the members above, from
    // `tables_host` on) is swapped into `flight` at submit and taken back once the frame after it has recorded.
    u32 slot = 0;  // which of the two per-frame sets this frame uses (worker command pools)
    struct Frame {
        VkCommandBuffer primary = VK_NULL_HANDLE, copy_cmd = VK_NULL_HANDLE, xfer_cmd = VK_NULL_HANDLE, send_cmd = VK_NULL_HANDLE, rb_cmd = VK_NULL_HANDLE;
        VkSemaphore copied = VK_NULL_HANDLE, rendered = VK_NULL_HANDLE, bound_sem = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        Buffer tables_host, tables, rb, rb_dev, scratch;
        std::vector<Buffer> stage, stage_dev, temp;
        u32 temp_quiet = 0, scratch_quiet = 0, stage_quiet = 0, rb_quiet = 0;
        std::vector<VkBufferCopy> back;
        bool busy = false;
        u64 frame = 0;              // its number (Submit's count)
        u32 slot = 1, queries = 0;  // queries: occlusion queries recorded (query_pool)
        u32 prof_base = kProfQueries, prof_n = 0;  // profile
        std::vector<std::string> prof_label;
    } flight;
    void SwapFrame() {
        std::swap(primary, flight.primary), std::swap(copy_cmd, flight.copy_cmd), std::swap(xfer_cmd, flight.xfer_cmd), std::swap(copied, flight.copied);
        std::swap(rb_cmd, flight.rb_cmd), std::swap(rendered, flight.rendered), std::swap(bound_sem, flight.bound_sem), std::swap(rb_dev, flight.rb_dev), std::swap(send_cmd, flight.send_cmd);
        std::swap(fence, flight.fence), std::swap(tables_host, flight.tables_host), std::swap(tables, flight.tables), std::swap(stage_dev, flight.stage_dev);
        std::swap(rb, flight.rb), std::swap(scratch, flight.scratch), std::swap(stage, flight.stage), std::swap(frame_temp, flight.temp), std::swap(back, flight.back), std::swap(slot, flight.slot);
        std::swap(temp_quiet, flight.temp_quiet), std::swap(scratch_quiet, flight.scratch_quiet), std::swap(stage_quiet, flight.stage_quiet), std::swap(rb_quiet, flight.rb_quiet);
        temp_at = 0, temp_used = 0, scratch_used = 0, rb_used = 0;  // the slot taken back belongs to a retired frame
        std::swap(prof_base, flight.prof_base), std::swap(prof_n, flight.prof_n), std::swap(prof_label, flight.prof_label);
        prof_n = 0, prof_label.clear();
    }
    void Retire();  // waits for the frame in flight and hands its GPU writes to the caller

    // ---- Occlusion queries, predication and GPU timestamps (vk_gpu.cpp) ----
    // Occlusion queries (PIXEL_PIPE_STAT_DUMP events into 128-byte blocks, the begin at +0, the end at +8), kQueries per
    // frame slot. The first kRays: blocks the CPU reads, in [query_lo, query_hi) (SetQueries): this frame's in begin order,
    // whether one is open, and the counts of the last frame retired. The rest: the game's mesh queries, which only feed
    // SET_PREDICATION (GOW3_PREDICATION=0 ignores both): their number by block and how many of them `cond` (u32 sample
    // counts per slot, for conditional rendering) holds, as the walk numbers and copies them.
    static constexpr u32 kRays = 32, kQueries = 4096;
    static inline const bool predication = [] { const char* e = std::getenv("GOW3_PREDICATION"); return !e || std::strcmp(e, "0"); }();
    VkQueryPool query_pool = VK_NULL_HANDLE;
    u64 query_lo = 0, query_hi = 0;
    std::vector<u64> query_blocks, query_counts;
    bool query_open = false;
    std::unordered_map<u64, u32> mesh;
    u32 mesh_copied = kRays;
    Buffer cond;
    u64 QueryBlock(const u32* b) const {  // the address an EVENT_WRITE dumps to when the CPU reads it, else 0
        const u64 a = ((u64)b[1] | (u64)(b[2] & 0xFFFF) << 32) & ~7ull;
        return (b[0] & 0x3F) == 0x39 && a - query_lo < query_hi - query_lo ? a : 0;
    }
    bool MeshDump(const u32* b) const { return predication && (b[0] & 0x3F) == 0x39 && !QueryBlock(b); }
    void Query(u64 a);
    void MeshQuery(const u32* b, bool record);
    void EndQuery() {
        if (query_on != ~0u) vkCmdEndQuery(cmd, query_pool, slot * kQueries + query_on), query_on = ~0u;
    }
    std::pair<u32, u32> CopyRange();
    void CopyQueries(std::pair<u32, u32> range);
    double temp_fence_ms = 0;  // debug: Retire's fence wait
    VkQueryPool temp_qp = VK_NULL_HANDLE;  // debug: per frame slot, 6 timestamps: xfer begin/end, copy begin/end, render end, readback end
    double tick_ms = 1e-6;                 // a GPU timestamp tick in ms: timestampPeriod ns (1 on NVIDIA, 40 on RADV Raven, issue #37)
    // TEMP perf: GPU timestamps (GOW3_GPU_PROF=1: copy, render and each rendering pass; 2: also each draw), read when the
    // frame retires. Queries per frame: copy start, copy end, frame end, then begin/end pairs.
    static constexpr u32 kProfQueries = 16384;
    const int prof = [] {
        const char* e = DebugOption("GOW3_GPU_PROF");
        if (e) return std::atoi(e);
        return DebugLevel() >= 2 ? 1 : 0;
    }();
    VkQueryPool prof_pool = VK_NULL_HANDLE;
    u32 prof_base = 0, prof_n = 0;
    static inline thread_local u32 prof_pass = ~0u;  // pair of the open rendering pass
    static inline thread_local u64 prof_draws = 0;
    std::vector<std::string> prof_label;
    u32 ProfBegin(std::string label) {  // serial recording only
        if (in_worker || prof_n + 5 > kProfQueries) return ~0u;
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, prof_pool, prof_base + 3 + prof_n);
        prof_label.push_back(std::move(label));
        return (prof_n += 2) - 2;
    }
    void ProfEnd(u32 pair) {
        if (pair != ~0u) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, prof_pool, prof_base + 4 + pair);
    }

    // ---- Draw state, registers, PM4 walk and parallel recording (vk_gpu.cpp) ----
    // Register files, per recording thread: the serial walk's carry over between frames; a block starts from a copy.
    static inline thread_local u32 ctx[0x400]{}, sh[0x400]{}, uc[0x400]{};
    // CLEAR_STATE: context registers back to the hardware defaults (only the nonzero ones the renderer reads).
    void ClearState() {
        std::fill(std::begin(ctx), std::end(ctx), 0u);
        ctx[0x00D] = ctx[0x082] = ctx[0x091] = 0x40004000;  // screen, window, generic scissor BR
        ctx[0x081] = ctx[0x090] = 0x80000000;                // window, generic scissor TL (window offset disabled)
        ctx[0x083] = 0x0000FFFF;                             // PA_SC_CLIPRECT_RULE
        ctx[0x08E] = ctx[0x08F] = 0xFFFFFFFF;                // CB_TARGET_MASK, CB_SHADER_MASK
        ctx[0x205] = 4;                                      // PA_SU_SC_MODE_CNTL (front face CW)
        ctx[0x30E] = ctx[0x30F] = 0xFFFFFFFF;                // PA_SC_AA_MASK
        for (u32 k = 0; k < 16; ++k) ctx[0x094 + 2 * k] = 0x80000000, ctx[0x095 + 2 * k] = 0x40004000, ctx[0x0B5 + 2 * k] = 0x3F800000;
    }
    static inline thread_local u32 index_size = 2, instances = 1;
    static inline thread_local u64 index_base = 0, draw_seq = 0;  // draws of the current Submit
    // Predication: the query block of the SET_PREDICATION in force (0: none); the next mesh query's number (walk order),
    // the one a PIXEL_PIPE_STAT_DUMP began for the next draw, and the one open in this command buffer (~0u: none).
    static inline thread_local u64 pred = 0;
    static inline thread_local u32 query_seq = 0, query_next = ~0u, query_on = ~0u;
    // Everything a recording block starts from: the context registers and the walk's state, with the SH and UC files
    // whole (Regs) or only what a parallel block's draws read of them (BlockRegs: PS and VS programs, primitive type).
    struct RegsBase {
        u32 ctx[0x400], cb_extent[8], db_extent, index_size, instances, query_seq, query_next;
        u64 index_base, pred;
    };
    struct Regs : RegsBase {
        u32 sh[0x400], uc[0x400];
    };
    struct BlockRegs : RegsBase {
        u32 ps[20], vs[20], prim;
    };
    static void SaveRegs(RegsBase& r) {
        std::copy_n(ctx, 0x400, r.ctx), std::copy_n(cb_extent, 8, r.cb_extent);
        r.db_extent = db_extent, r.index_size = index_size, r.instances = instances, r.index_base = index_base;
        r.pred = pred, r.query_seq = query_seq, r.query_next = query_next;
    }
    static void LoadRegs(const RegsBase& r) {
        std::copy_n(r.ctx, 0x400, ctx), std::copy_n(r.cb_extent, 8, cb_extent);
        db_extent = r.db_extent, index_size = r.index_size, instances = r.instances, index_base = r.index_base;
        pred = r.pred, query_seq = r.query_seq, query_next = r.query_next;
    }
    static void SaveRegs(Regs& r) { SaveRegs((RegsBase&)r), std::copy_n(sh, 0x400, r.sh), std::copy_n(uc, 0x400, r.uc); }
    static void LoadRegs(const Regs& r) { LoadRegs((const RegsBase&)r), std::copy_n(r.sh, 0x400, sh), std::copy_n(r.uc, 0x400, uc); }
    static void SaveRegs(BlockRegs& r) { SaveRegs((RegsBase&)r), std::copy_n(sh + 0x08, 20, r.ps), std::copy_n(sh + 0x48, 20, r.vs), r.prim = uc[0x242]; }
    static void LoadRegs(const BlockRegs& r) {
        LoadRegs((const RegsBase&)r), std::copy_n(r.ps, 20, sh + 0x08), std::copy_n(r.vs, 20, sh + 0x48), uc[0x242] = r.prim;
    }

    // Graphics dynamic state recorded in the current command buffer: Draw emits only what changed (reset per Submit).
    struct DynState {
        bool valid = false, neg_one = false;
        bool pushed = false;  // `push` holds the push constants (another layout's push clears it)
        Push push{};
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkShaderEXT shaders[5]{};
        VkViewport vp{};
        VkRect2D sc{};
        u32 raster[8]{};  // samples, alpha to coverage, depth clamp, cull, front face, bias enable, topology, restart
        float bias[3]{};
        u32 depth[5]{};  // test, write, compare op, stencil test, bounds test
        float bounds[2]{-1.f, -1.f};
        u32 stencil[14]{};
        u32 blend_count = 0;
        VkBool32 blend[8]{};
        VkColorBlendEquationEXT eq[8]{};
        VkColorComponentFlags mask[8]{};
        float constants[4]{};
        u64 index = ~0ull;
        VkIndexType index_type{};
    };
    static thread_local DynState dyn;
    // Draw's blend state, decoded again only when its registers and inputs change (consecutive draws mostly repeat it):
    // key = {which color targets are bound << 1 | a PS runs, CB_TARGET_MASK, CB_BLEND0..7_CONTROL, CB_COLOR0..7_INFO}.
    struct BlendState {
        u32 key[18];
        VkBool32 blend[8], square_blend[8];
        VkColorBlendEquationEXT eq[8], square_eq[8];
        VkColorComponentFlags mask[8];
        bool square;
        u8 square_color_mask, square_alpha_mask;
        u32 unsupported;  // targets whose scaled MIN/MAX blend is not representable
    };
    static inline thread_local BlendState blend_memo{{~0u}};  // key[0] above any real one: none yet
    // Current dynamic rendering: color attachment views, depth target.
    static inline thread_local VkImageView bound_color[8]{};
    static inline thread_local const Target* bound_depth = nullptr;
    static inline thread_local bool rendering = false;
    static inline thread_local std::vector<VkImage> pass_w;  // targets the open pass writes
    static inline thread_local std::vector<std::pair<VkImage, bool>> draw_rts;  // targets the current draw samples (false) or stores to
    static inline thread_local std::vector<const Target*> twins;  // targets the current dispatch stores to through their storage_copy
    static bool In(const std::vector<VkImage>& v, VkImage x) { return std::find(v.begin(), v.end(), x) != v.end(); }
    static void Add(std::vector<VkImage>& v, VkImage x) {
        if (!In(v, x)) v.push_back(x);
    }
    static bool Overlap(const std::vector<VkImage>& a, const std::vector<VkImage>& b) {
        return std::any_of(a.begin(), a.end(), [&](VkImage x) { return In(b, x); });
    }

    // PM4 walk and outside effects.
    template <class F> void Walk(std::span<const Segment> chain, Pos from, Pos to, F&& exec);
    std::span<const Segment> walk_chain;
    static inline thread_local size_t walk_seg = 0, walk_pos = 0;  // packet being executed
    bool discovered = false;                                     // this frame (the serial walk only)
    // Effects(): the serial walk's packets that act outside the GPU, by header; `effects_whole` false when an
    // INDIRECT_BUFFER was seen (0x3F is collected only to clear it).
    std::vector<const u32*> effects;
    bool effects_whole = false;
    static constexpr bool Effect(u32 op) { return op == 0x37 || op == 0x3C || op == 0x3F || (op >= 0x47 && op <= 0x49) || op == 0x50; }
    void Collect(u32 op, const u32* body) { op == 0x3F ? void(effects_whole = false) : effects.push_back(body - 1); }

    // Parallel recording (Submit). The serial walk applies the registers in chain order and cuts the frame into blocks of
    // draws, each recorded by any thread into its own secondary command buffer from a copy of the registers at its start;
    // the primary executes the blocks in chain order.
    struct Pass {  // attachments of a rendering instance
        VkImageView color[8]{};
        const Target* depth = nullptr;
        bool operator==(const Pass&) const = default;
    };
    struct Block {
        Pos from, to;
        u32 regs = 0;  // a parallel one's in `block_regs`
        size_t horizon = 0;  // gpu_writes recorded before it
        u64 draws = 0;
        bool serial = false;  // recorded by the walk itself (dispatches, draws to linear targets)
        VkCommandBuffer cb = VK_NULL_HANDLE;
        u64 first = 0;  // its first draw among the frame's parallel ones
        std::pair<u32, u32> copy;  // mesh queries it copies first (CopyQueries), for its predicated draws
        double us = 0;  // recording time
        // Its first command a rendering begin (`head`, clean) or anything else; a pass still open at its end (`tail`).
        bool started = false, clean = false, open = false;
        Pass head, tail;
        std::vector<std::pair<GuestTex*, u64>> resync;  // images it samples, with the GPU write they must reflect (Submit uploads)
        std::vector<GuestTex*> resync_vs;               // those its vertex shaders sample
        std::vector<std::pair<GpuWrite, bool>> lazy;    // writes those uploads need written back first (Materialize's `learn`)
        std::vector<GpuWrite> writes;                   // GuestTex storage writes recorded by this block (merged in chain order)
        // Hazards (lazy pass ends): targets written / sampled before its first full barrier (`all`: something there
        // needed every earlier write done) and since its last one.
        std::vector<VkImage> in_w, in_r, out_w, out_r, out_wv, vs_in;  // as in Hazards
        bool fenced = false, all = false, wide = false;
    };
    static void SaveHazards(Block& b) {
        b.fenced = hz.fenced, b.all = hz.all, b.out_w = hz.w, b.out_r = hz.r;
        b.in_w = hz.fenced ? hz.in_w : hz.w, b.in_r = hz.in_r;
        b.out_wv = hz.wv, b.vs_in = hz.vs_in, b.wide = hz.wide;
    }
    static inline thread_local Block* block_now = nullptr;  // the parallel block this thread records
    std::vector<Block> blocks;
    std::vector<BlockRegs> block_regs;  // kept across frames
    std::span<const Segment> chain_now;
    const CpuRead* cpu_now = nullptr;
    struct Recorder {  // per recording thread, per frame slot: secondaries reused once that frame retires
        VkCommandPool pool[2]{};
        std::vector<VkCommandBuffer> cbs[2];
        size_t used[2]{};
    };
    std::vector<Recorder> recorders;  // the workers', then the walking thread's
    std::vector<std::thread> threads;
    std::mutex pool_lock;
    std::condition_variable pool_cv;
    u64 pool_gen = 0;
    u32 pool_busy = 0;
    bool pool_stop = false;
    const std::function<void(u32)>* pool_job = nullptr;
    // Runs job(k) on every recording thread (k: its recorder, this thread the last) and returns once all are done.
    void Parallel(const std::function<void(u32)>& job) {
        {
            std::scoped_lock lock(pool_lock);
            pool_job = &job, ++pool_gen, pool_busy = (u32)threads.size();
        }
        pool_cv.notify_all();
        job((u32)threads.size());
        std::unique_lock lock(pool_lock);
        pool_cv.wait(lock, [&] { return pool_busy == 0; });
    }
    std::atomic<size_t> next_block{0};
    // Recording cost of the last frame's parallel draws, per block: {first draw, microseconds per draw}. Steady frames
    // repeat it, so the cuts aim at equal shares of it (draws differ by more than twice in cost).
    std::vector<std::pair<u64, double>> last_cost;
    double last_total = 0;
    VkCommandBuffer Secondary(u32 rec);  // begun, with the descriptor set bound
    void RecordBlocks(u32 rec);
    void Draw(u32 op, const u32* b, const CpuRead& cpu);
    void Dispatch(const u32* b, const CpuRead& cpu);
    bool TryLinearCopyDispatch(const u32* b, const CpuRead& cpu);
    struct LinSync { u64 dst, src, size; };
    std::vector<LinSync> lin_syncs;  // deferred BackingPages syncs when unrecorded parallel blocks precede a linear copy
    void SyncLinearBacking(u64 dst_lo, u64 src_lo, u64 copy_bytes);
    void FlushLinearSyncs();

    // ---- Diagnostics: stats, notes, traced probes and exploded vertices (vk_gpu.cpp) ----
    Stats stats;
    void Note(const std::string& s) {
        std::scoped_lock lock(note_lock);
        if (stats.trace && !stats.trace_lines.empty()) stats.trace_lines.back() += " !" + s.substr(0, 80);
        if (std::find(stats.notes.begin(), stats.notes.end(), s) == stats.notes.end() && stats.notes.size() < 64) stats.notes.push_back(s);
    }
    // TEMP diagnostics (debug runs): a line for the log, once per text (a failed Vulkan call, what the device answers, a probe's
    // verdict); Submit hands them to Stats::report.
    std::vector<std::string> report;
    std::unordered_set<std::string> reported;
    void Report(std::string s) {
        GOW3_DIAG {
            std::scoped_lock lock(note_lock);
            if (reported.insert(s).second) report.push_back(std::move(s));
        }
    }

    // Trace diagnostics of every draw: pipeline statistics, samples passed, and the first written color target at the
    // scissor centre, appended to its trace line after the Submit.
    static constexpr u32 kProbes = 4096;
    static constexpr VkQueryPipelineStatisticFlags kStats = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
        VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
        VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT | VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
    VkQueryPool q_stat = VK_NULL_HANDLE, q_occ = VK_NULL_HANDLE;
    // Texels; per draw {vec4 gl_Position, taken, gl_VertexIndex, pad[2], v0..v63} of the first VS invocation
    // (GOW3_PROBE_DRAW=<draw> prints the registers of that draw).
    static constexpr u32 kVertexProbe = 288;
    Buffer probe_buf, vertex_probe;
    std::map<VkFormat, Tex> probe_img;  // 16x16 resolve images of multisampled targets
    static constexpr u32 kProbeBytes = 16 * 16 * 16;  // per draw: up to 16x16 texels of up to 16 bytes
    struct Probe {
        size_t line;
        u32 bytes, rect = 0;  // texel size; bytes of the hashed rectangle (GOW3_PROBE_XY)
        bool px = true;       // samples passed and the texel: not for a draw inside one of the frame's own occlusion queries
    };
    std::vector<Probe> probes;
    // TEMP index probe (F9): each traced indexed draw runs again without rasterizing, once per way of binding the mirror as its
    // index buffer (0 the window to the buffer's end; 1 the draw's own address; 2 the window with its size, as the draw with
    // BindIndex2), with a vertex shader that records {invocations, lowest, highest gl_VertexIndex} per way, set beside what
    // the CPU reads at the index address.
    static constexpr u32 kIndexProbe = 3 * 12;
    Buffer index_probe;
    VkShaderEXT index_probe_vs = VK_NULL_HANDLE;
    VkShaderModule index_probe_mod = VK_NULL_HANDLE;
    std::unordered_map<u64, VkPipeline> index_probe_pls;
    struct IndexProbe {
        size_t line;
        u32 lo, hi;  // what the CPU reads, base vertex added (lo > hi: unreadable)
    };
    std::vector<IndexProbe> index_probes;
    std::unordered_set<u64> traced_tex;  // guest textures the traced frame bound (ReadTextures)
    void BeginTraceProbes();
    void ProbeColor(const Target* t, const VkRect2D& sc);
    void CollectTraceProbes();

    // Exploded vertices (GOW3_BOOM=1, +44% GPU at 4K; frames not traced): per draw {exploded vertices, first one's index, its load count,
    // pad, gl_Position, then its first 192 guest loads: pc[192], address / 4 [192], value[192]}, one half per frame slot;
    // the programs of each record; read when the frame retires. boom_up: each slot's mirror uploads (guest range, stage
    // chunk and offset), to tell which frame's upload the loaded bytes came from.
    static inline const bool boom_on = DebugDeep("GOW3_BOOM");
    static constexpr u32 kBoomDraws = 4096, kBoomLoads = 192, kBoomConsts = 96, kBoomRec = 32 + 12 * kBoomLoads + 4 * kBoomConsts;
    Buffer boom_buf;
    std::atomic<u32> boom_next{0};
    u32 boom_used[2]{};
    std::vector<Pending> boom_up[2];
    // BeginFrame's staged pieces by address (the snapshot the mirror gets), for BoomUser: a draw's user-data pointers whose block
    // changed in guest memory since the snapshot, or whose first dwords differ from what that draw's program found there before.
    struct BoomSnap {
        u64 lo, hi;
        const u8* p;
    };
    std::vector<BoomSnap> boom_snap;
    void BoomUser(const u32* user, const CpuRead& cpu);
    struct BoomDraw {
        u64 vs, ps;
        u32 vertices;
        u32 user[16];  // the vertex stage's user data (s0..s15)
    };
    std::vector<BoomDraw> boom_meta[2]{std::vector<BoomDraw>(kBoomDraws), std::vector<BoomDraw>(kBoomDraws)};
    // Debug runs (same switch): the guest ranges a frame's draws read (vertex and constant buffers, indices), by the draw's vertex
    // shader, kept by each recording thread; the end of each Submit moves them to Stats::reads.
    static inline thread_local u64 read_prog = 0;  // the vertex shader of the draw being recorded
    static inline std::mutex read_lock;
    static inline std::vector<std::vector<Stats::Read>*> read_lists;
    void NoteRead(u64 lo, u64 hi) {
        if (!boom_on || !track || hi <= lo) return;
        static thread_local std::vector<Stats::Read>* mine = [] {
            auto* v = new std::vector<Stats::Read>;  // lives with the process: Submit reads it from the GPU thread
            std::scoped_lock l(read_lock);
            read_lists.push_back(v);
            return v;
        }();
        if (mine->empty() || mine->back().lo != lo || mine->back().hi != hi || mine->back().prog != read_prog) mine->push_back({lo, hi, read_prog});
    }
    // TEMP debug runs: every range a frame's work declares to Track (an upper bound of what the GPU reads: it includes V#s
    // whose scalar loads the draw table carries), for FrameDump's share of the offered ring and arenas the GPU reads.
    static inline std::vector<std::vector<std::pair<u64, u64>>*> gpu_read_lists;
    static void NoteGpuRead(u64 lo, u64 hi) {
        static thread_local auto* mine = [] {
            auto* v = new std::vector<std::pair<u64, u64>>;  // lives with the process, like NoteRead's
            std::scoped_lock l(read_lock);
            gpu_read_lists.push_back(v);
            return v;
        }();
        if (mine->empty() || mine->back() != std::pair{lo, hi}) mine->push_back({lo, hi});
    }
};


// Register packets carry a few dwords: overlapping fixed-size moves inline, the CRT memcpy only past 16.
inline void CopyDwords(u32* d, const u32* s, u32 n) {
    if (n > 16) std::memcpy(d, s, n * 4);
    else if (n >= 8) std::memcpy(d, s, 32), std::memcpy(d + n - 8, s + n - 8, 32);
    else if (n >= 4) std::memcpy(d, s, 16), std::memcpy(d + n - 4, s + n - 4, 16);
    else if (n >= 2) std::memcpy(d, s, 8), std::memcpy(d + n - 2, s + n - 2, 8);
    else if (n) *d = *s;
}

// Packets in [from, to): register writes applied here, every other packet handed to `exec(op, body, segment, dword)`.
template <class F> void Renderer::Impl::Walk(std::span<const Segment> chain, Pos from, Pos to, F&& exec) {
    for (size_t si = from.first; si < chain.size() && Pos{si, 0} < to; ++si) {
        const Segment& s = chain[si];
        const size_t lim = si < to.first ? s.count : std::min<size_t>(s.count, to.second);
        for (size_t i = si == from.first ? from.second : 0; i < lim;) {
            const u32 hdr = s.dw[i];
            if (hdr >> 30 != 3) {
                ++i;
                continue;
            }
            const u32 cnt = (hdr >> 16 & 0x3FFF) + 1, op = hdr >> 8 & 0xFF;
            const u32* b = s.dw + i + 1;
            auto set = [&](u32* file) {
                if (b[0] < 0x400) CopyDwords(file + b[0], b + 1, std::min(cnt - 1, 0x400 - b[0]));
            };
            switch (op) {
            case 0x69:
                set(ctx);
                // Extent hint: a NOP {width, height} right after the CB setup / 20 dwords after the DB one (shadPS4
                // Liverpool::HandleContextRegisterHint).
                if (const u32 reg = b[0]; reg >= 0x318 && reg < 0x318 + 8 * 15 && (reg - 0x318) % 15 == 0)
                    cb_extent[(reg - 0x318) / 15] = (cnt == 0xE || cnt == 0xD || cnt == 0xB) && i + cnt + 3 <= s.count && b[cnt] == 0xC0001000 ? b[cnt + 1] : 0;
                else if (reg == 0x010)
                    db_extent = cnt == 8 && i + 23 <= s.count && b[20] == 0xC0001000 ? b[21] : 0;
                break;
            case 0x12: ClearState(); break;
            case 0x76: set(sh); break;
            case 0x79: set(uc); break;
            case 0x2F: instances = b[0]; break;
            case 0x26: index_base = (u64)b[0] | (u64)(b[1] & 0xFFFF) << 32; break;
            case 0x2A: index_size = b[0] & 1 ? 4 : 2; break;
            case 0x20: pred = predication && b[1] >> 8 & 1 ? (b[0] & ~0xFull) | (u64)(b[1] & 0xFF) << 32 : 0; break;
            case 0x3F: Note("PM4 INDIRECT_BUFFER not executed (the chain is the submitted buffers)"), exec(op, b, si, i); break;
            default: exec(op, b, si, i); break;
            }
            i += cnt + 1;
        }
    }
}
} // namespace Gpu
