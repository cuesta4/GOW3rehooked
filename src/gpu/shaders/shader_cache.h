// SPDX-License-Identifier: GPL-2.0-or-later
// Persistence: the on-disk cache, one file per strategy (jit.cache: the variants the JIT recorded; aot.cache: the AOT
// build), each holding its shaders (SPIR-V) and driver binaries. Records are zstd frames; once enough have been seen, a
// dictionary trained on them is stored in the file and used by the records that follow. A translation key points to a
// content record (index records), so variants with the same SPIR-V are stored, and compiled by the driver, once. The file is
// mapped and indexed when it is configured; a record is decompressed when it is asked for. When the game opens a WAD, the
// binaries its programs need (the AOT cache lists them per WAD, and stores them grouped by WAD) are read ahead in large
// pieces, and the shaders the player's runs used there are made ready (Warm).
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Gpu {
enum class CacheKind { Jit, Aot };

// Content version of each cache file: bump it when a change makes the SPIR-V (or driver binary) stored for a key stale -
// the translator's output, the shader layouts, the key (the AOT's also for its order). A file of another version is discarded
// (and rewritten) when the game starts; the recipes are kept apart. 1: the first public release. 2: the pipeline paths' release
// with SGPR spills kept per invocation and pruned shader contract (exact sample_log2).
// kAotPipelineVersion: the pipelines the AOT derives from the game's files (shader_contract.cpp's pipeline rules). Bumping it
// derives them again and keeps every cache: the driver judges its own pipeline cache. 2: pruned shader contract (exact sample_log2).
inline constexpr std::uint32_t kJitCacheVersion = 2, kAotCacheVersion = 2, kAotPipelineVersion = 2;

struct CachedShader {
    std::uint32_t stage = 0;
    std::uint64_t out_mask = 0;  // VS: exported parameters
    std::vector<std::uint32_t> spirv;
};

struct CachedBinary {
    std::uint64_t out_mask = 0;  // VS: exported parameters
    std::vector<std::uint8_t> data;
};

// Upper-case name of a WAD from a game path ("/app0/EXEC/WAD/PS4_LE/R_FX.WAD" -> "R_FX"); empty for other files.
std::string WadName(const char* path);

class ShaderCache {
public:
    struct Progress {
        std::uint32_t done, total, pipelines_done, pipelines_total;
        std::uint64_t done_weight, total_weight;  // the same jobs weighed by their expected cost (the remaining-time estimate)
    };

    static ShaderCache& Instance();
    // Loads the cache of `kind` from `dir` (created if missing). Without it the cache stays disabled.
    void Configure(const std::filesystem::path& dir, CacheKind kind);
    bool Enabled() const { return enabled; }
    CacheKind Kind() const { return kind; }
    const std::filesystem::path& Directory() const { return dir; }
    // A shader is a translation key -> content (the SPIR-V and its exports, identified by ContentHash) -> record. Variants
    // whose SPIR-V is the same share one record, and one driver binary (a binary depends only on the SPIR-V and its stage).
    static std::uint64_t ContentHash(std::span<const std::uint32_t> spirv, std::uint64_t out_mask);
    bool Contains(std::uint64_t key) const;
    std::uint64_t ContentOf(std::uint64_t key) const;  // 0: the key has no shader
    bool ContainsContent(std::uint64_t content) const;
    std::shared_ptr<const CachedShader> Find(std::uint64_t key);
    // A shader the JIT translated for a draw (jit.cache only: a miss of the AOT run becomes a recipe instead). It waits in
    // memory (the run keeps using it) and is hashed, compressed and written by Flush: the draw path and the compile threads
    // do no I/O and no hashing.
    void Record(std::uint64_t key, std::uint32_t stage, std::uint64_t out_mask, std::span<const std::uint32_t> spirv);
    // Deferred writes (shaders recorded by the JIT, the recipes): something is waiting; wakes the flush thread.
    void Dirty();
    // Bytes the game streamed into guest memory (file reads, mappings): a lot of them within a moment is a loading screen.
    void Streamed(std::uint64_t bytes) { streamed.fetch_add(bytes, std::memory_order_relaxed); }
    // Writes everything waiting now (`all_threads`: at exit, nothing else is running).
    void Flush(const char* why, bool all_threads);
    // A shader of the AOT build (aot.cache only): written once per content, linked to `key`; returns the content (0: failed).
    std::uint64_t Store(std::uint64_t key, std::uint32_t stage, std::uint64_t out_mask, std::span<const std::uint32_t> spirv);
    // `key` has the shader of `content` (already stored): the AOT build found it without translating again.
    void Link(std::uint64_t key, std::uint64_t content);
    // The AOT build's memory of the translator's text: the hash of the GLSL (and stage) -> the content it compiled to, so the
    // same text is not compiled by glslang again.
    std::uint64_t GlslContent(std::uint64_t glsl_hash) const;  // 0: unknown
    void LinkGlsl(std::uint64_t glsl_hash, std::uint64_t content);
    // Driver binaries (VK_EXT_shader_object): opaque data of one content for the GPU and driver, under a key the caller derives
    // from the content and the GPU/driver.
    bool ContainsBinary(std::uint64_t key) const;
    std::shared_ptr<const CachedBinary> FindBinary(std::uint64_t key);
    bool StoreBinary(std::uint64_t key, std::uint64_t out_mask, std::span<const std::uint8_t> data);
    // The translation keys (and binary keys) the programs of WAD `wad` need: their binaries are read ahead when the game opens it.
    void SetWadKeys(const std::string& wad, std::span<const std::uint64_t> keys, std::span<const std::uint64_t> binaries);
    // The shaders the player's runs used (the recipes) whose programs lie in WAD `wad`, in the order the runs first met them,
    // given by the AOT at every launch: `prepare` (the renderer: the shader object from the binary) makes them ready when the
    // game opens the WAD, on the warm thread. 0: it was there already, 1: made now, 2: failed.
    struct Hot {
        std::uint64_t key, binary;
        std::uint32_t stage;
    };
    void SetWadHot(const std::string& wad, std::vector<Hot> hot);
    void SetPrepare(std::function<std::uint32_t(std::uint64_t key, std::uint64_t binary, std::uint32_t stage)> fn);
    void SetWarmPipelines(std::function<void()> fn);
    void SetPipelineCacheFlusher(std::function<void(bool exit)> fn);  // called by every flush; `exit`: the last one
    void ClearRendererCallbacks();
    void WadOpened(const char* path);
    // End of the run: trains the dictionaries and flushes.
    void Finish();
    // Compilations queued and finished, for the loading screen.
    void Queued(std::uint32_t weight = 1) { total.fetch_add(1, std::memory_order_relaxed), total_weight.fetch_add(weight, std::memory_order_relaxed); }
    void Completed(std::uint32_t weight = 1) { done.fetch_add(1, std::memory_order_relaxed), done_weight.fetch_add(weight, std::memory_order_relaxed); }
    void PipelineQueued() { pipeline_total.fetch_add(1, std::memory_order_relaxed); }
    void PipelineCompleted() { pipeline_done.fetch_add(1, std::memory_order_relaxed); }
    Progress Snapshot() const {
        return {done.load(std::memory_order_relaxed), total.load(std::memory_order_relaxed), pipeline_done.load(std::memory_order_relaxed),
                pipeline_total.load(std::memory_order_relaxed), done_weight.load(std::memory_order_relaxed), total_weight.load(std::memory_order_relaxed)};
    }
    ~ShaderCache();

private:
    struct Loc {
        std::uint64_t offset, out_mask;  // offset of the packed data in the mapped file
        std::uint32_t packed, raw, stage;
    };
    struct View;
    struct Dict;  // zstd dictionary trained on the records of one type (shaders, binaries)
    bool Append(std::uint32_t type, std::uint32_t stage, std::uint64_t key, std::uint64_t out_mask, const void* data, std::size_t bytes);
    std::vector<std::uint8_t> Decode(std::uint32_t type, const Loc& loc);
    std::shared_ptr<View> Mapped(std::uint64_t end);  // the file mapped up to at least `end` (null: it is shorter)
    void Train(std::uint32_t type);
    void Warm(const std::string& wad);
    void Requeue(const std::string& wad, std::uint64_t h);
    std::shared_ptr<const CachedShader> FindContent(std::uint64_t content);
    // Hashes and writes a shader (once per content) and links the key to it; `keep` is the decoded shader to hold for the run.
    std::uint64_t Commit(std::uint64_t key, std::uint32_t stage, std::uint64_t out_mask, std::span<const std::uint32_t> spirv, const std::shared_ptr<const CachedShader>& keep);
    struct Pair {
        std::uint64_t a, b;
    };
    void PutPairs(std::uint32_t type, std::vector<Pair>& items);  // the lock is held
    bool enabled = false;
    CacheKind kind = CacheKind::Jit;
    std::filesystem::path dir;
    mutable std::shared_mutex lock;
    std::unordered_map<std::uint64_t, std::shared_ptr<const CachedShader>> shaders;   // decoded, by content
    std::unordered_map<std::uint64_t, Loc> locs[2];  // records of the file (shaders by content, binaries)
    std::unordered_set<std::uint64_t> writing[2];    // keys currently being compressed/appended (under lock)
    std::unordered_map<std::uint64_t, std::uint64_t> index, glsl_map;  // key -> content, GLSL hash -> content
    std::vector<Pair> index_new, glsl_new;                              // links not written yet (written in batches)
    std::unordered_map<std::uint64_t, std::shared_ptr<const CachedShader>> recent;  // recorded by the JIT, not written yet, by key
    struct WadKeys {
        std::vector<std::uint64_t> shaders, binaries;
    };
    std::unordered_map<std::uint64_t, WadKeys> wad_keys;  // by XXH3 of the WAD name
    std::unordered_map<std::uint64_t, std::vector<Hot>> wad_hot;  // the same
    std::shared_ptr<View> view;  // the file as mapped when last needed; a snapshot stays valid for a decode in progress
    std::uint64_t file_end = 0;
    std::ofstream out;
    std::unique_ptr<Dict> dicts[2];
    void Put(std::uint32_t type, std::uint32_t stage, std::uint64_t key, std::uint64_t out_mask, std::uint32_t raw, const void* packed, std::uint32_t n);
    // Deferred writes. The flush thread (low priority, started by the first Dirty) sleeps until something waits, then looks every
    // tick at what the game streamed and which WADs it opened: a loading screen (many WADs, or a lot of data read) is where
    // the writes go, as is a burst of compilations that has just finished with enough shaders waiting.
    struct Pending {
        std::uint64_t key, out_mask;
        std::uint32_t stage;
        std::shared_ptr<const CachedShader> shader;
    };
    void FlushLoop(const std::stop_token& stop);
    std::mutex pending_lock;
    std::vector<Pending> pending;
    std::atomic<std::uint64_t> streamed{0};
    std::atomic<std::uint32_t> wad_opens{0};
    std::atomic<bool> dirty{false};
    std::mutex flush_lock;
    std::condition_variable_any flush_cv;
    std::jthread flusher;
    // WADs the game opened, warmed by one low-priority thread.
    std::mutex wad_lock;
    std::condition_variable_any wad_cv;
    std::deque<std::string> wad_queue;
    std::unordered_set<std::uint64_t> wad_seen;
    std::jthread wad_thread;
    std::mutex prepare_cb_lock, flush_cb_lock;
    std::function<std::uint32_t(std::uint64_t key, std::uint64_t binary, std::uint32_t stage)> prepare_cb;
    std::function<void()> warm_pipelines_cb;
    std::function<void(bool exit)> flush_pipeline_cache_cb;
    std::atomic<std::uint32_t> done{0}, total{0}, pipeline_done{0}, pipeline_total{0};
    std::atomic<std::uint64_t> done_weight{0}, total_weight{0};
};
} // namespace Gpu
