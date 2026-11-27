// SPDX-License-Identifier: GPL-2.0-or-later
#include "gpu/piece_hash.h"
#include "gpu/shaders/shader_cache.h"
#include <zdict.h>
#include <zstd.h>
#include <xxhash.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <format>
#include "gpu/shaders/shader_recipes.h"
#include "main/cpu.h"
#include "main/host.h"
#ifdef _WIN32
#include <windows.h>
#else
#include "main/mapped_file.h"
#include "orbis/common/thread.h"
#endif

namespace Gpu {
namespace {
constexpr std::uint32_t kMagic = 0x43533347, kFormat = 3;   // "G3SC"; kFormat: the layout of the records below
const char* CacheName(CacheKind kind) {
    return kind == CacheKind::Jit ? (PieceHash::UsesAes() ? "jit-hash2-aes.cache" : "jit-hash2-xxh.cache")
                                  : (PieceHash::UsesAes() ? "aot-hash2-aes.cache" : "aot-hash2-xxh.cache");
}
constexpr std::uint32_t kMaxRecord = 256u << 20;
constexpr int kLevel = 3;                                  // zstd level: a third of level 6's time (0.3 ms for a 58 KB shader) for 10% more bytes
// kShader and kBinary records are keyed by content (the binary's key comes from the caller); kIndex holds (key, content) pairs and
// kGlslMap (GLSL hash, content) pairs, each record a compressed batch. A kDict record's `stage` is the type it serves.
enum Type : std::uint32_t { kShader, kBinary, kDict, kWadKeys, kIndex, kGlslMap };
struct Header {
    std::uint32_t magic = kMagic, format = kFormat, version;
    explicit Header(CacheKind kind = CacheKind::Jit) : version(kind == CacheKind::Jit ? kJitCacheVersion : kAotCacheVersion) {}
};
struct Entry {
    std::uint64_t key, out_mask;
    std::uint32_t type, stage, raw, packed;
};
} // namespace

std::string WadName(const char* path) {
    if (!path) return {};
    const std::filesystem::path p(path);
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (ext != ".wad") return {};
    std::string name = p.stem().string();
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::toupper(c)); });
    return name;
}

struct ShaderCache::View {
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE, map = nullptr;
#endif
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    ~View() { Close(); }
    void Close() {
#ifdef _WIN32
        if (data) UnmapViewOfFile(data);
        if (map) CloseHandle(map);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
        file = INVALID_HANDLE_VALUE, map = nullptr;
#else
        Gow3::UnmapReadOnly(data, size);
#endif
        data = nullptr, size = 0;
    }
    bool Open(const std::filesystem::path& path) {  // shared for writing: the cache appends to the file meanwhile
#ifdef _WIN32
        file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
        LARGE_INTEGER n{};
        if (file == INVALID_HANDLE_VALUE || !GetFileSizeEx(file, &n) || n.QuadPart <= 0) return Close(), false;
        size = std::size_t(n.QuadPart), map = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (map) data = static_cast<const std::uint8_t*>(MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0));
        return data ? true : (Close(), false);
#else
        return Gow3::MapReadOnly(path, data, size);
#endif
    }
};

struct ShaderCache::Dict {
    static constexpr std::size_t Budget = 32u << 20, Capacity = 256u << 10, MinSamples = 16, TrainAfter = 256;
    std::vector<std::uint8_t> samples;
    std::vector<std::size_t> sizes;
    ZSTD_CDict* c = nullptr;
    std::atomic<ZSTD_DDict*> d{nullptr};
    std::atomic<std::uint32_t> id{0};
    ~Dict() { ZSTD_freeCDict(c), ZSTD_freeDDict(d.load(std::memory_order_relaxed)); }
    void Load(const void* blob, std::size_t size) {
        c = ZSTD_createCDict(blob, size, kLevel);
        d.store(ZSTD_createDDict(blob, size), std::memory_order_relaxed);
        id.store(ZDICT_getDictID(blob, size), std::memory_order_release);
        samples = {}, sizes = {};
    }
    void Sample(const void* data, std::size_t size) {
        if (c || samples.size() >= Budget) return;
        const auto* p = static_cast<const std::uint8_t*>(data);
        samples.insert(samples.end(), p, p + size), sizes.push_back(size);
    }
};

ShaderCache::~ShaderCache() = default;

ShaderCache& ShaderCache::Instance() {
    static ShaderCache cache;
    return cache;
}

void ShaderCache::Configure(const std::filesystem::path& d, CacheKind k) {
    std::unique_lock l(lock);
    dir = d, kind = k;
    for (auto& dict : dicts) dict = std::make_unique<Dict>();
    std::filesystem::create_directories(dir);
    const auto file = dir / CacheName(k);
    view = std::make_shared<View>();
    bool valid = false;
    std::uint64_t good = sizeof(Header);  // end of the last whole record
    if (view->Open(file) && view->size >= sizeof(Header)) {
        const Header want(k);
        Header got;
        std::memcpy(&got, view->data, sizeof(got));
        valid = !std::memcmp(&want, &got, sizeof(got));
    }
    for (std::uint64_t at = good; valid;) {  // index the records; the data stays compressed
        Entry r;
        if (at + sizeof(r) > view->size) break;
        std::memcpy(&r, view->data + at, sizeof(r));
        if (r.packed > kMaxRecord || r.raw > kMaxRecord || at + sizeof(r) + r.packed > view->size) break;
        const std::uint8_t* body = view->data + at + sizeof(r);
        if (r.type == kDict) {
            if (r.stage < 2) dicts[r.stage]->Load(body, r.packed);
        } else if (r.type == kShader || r.type == kBinary) {
            locs[r.type][r.key] = {at + sizeof(r), r.out_mask, r.packed, r.raw, r.stage};
        } else if (r.type == kIndex || r.type == kGlslMap) {
            std::vector<Pair> pairs(r.raw / sizeof(Pair));
            if (r.raw % sizeof(Pair) || ZSTD_decompress(pairs.data(), r.raw, body, r.packed) != r.raw) break;
            for (const Pair& p : pairs) (r.type == kIndex ? index : glsl_map)[p.a] = p.b;
        } else if (r.type == kWadKeys) {
            std::vector<std::uint64_t> all(r.raw / 8);
            if (r.raw % 8 || ZSTD_decompress(all.data(), r.raw, body, r.packed) != r.raw || all.empty() || all[0] >= all.size()) break;
            wad_keys[r.key] = {{all.begin() + 1, all.begin() + 1 + all[0]}, {all.begin() + 1 + all[0], all.end()}};
        } else {
            break;
        }
        good = at += sizeof(r) + r.packed;
    }
    if (valid && good < view->size) {  // a killed run left a torn tail: cut it, or the records appended now would be lost behind it
        view.reset();
        std::error_code ec;
        std::filesystem::resize_file(file, good, ec);
        view = std::make_shared<View>();
        if (!view->Open(file)) valid = false;
    }
    if (!valid) {
        view.reset();
        locs[0].clear(), locs[1].clear(), writing[0].clear(), writing[1].clear(), index.clear(), glsl_map.clear(), wad_keys.clear();
        for (auto& dict : dicts) dict = std::make_unique<Dict>();
    }
    out.open(file, std::ios::binary | (valid ? std::ios::app : std::ios::trunc));
    if (!valid) {
        const Header h(k);
        out.write((const char*)&h, sizeof(h));
        good = sizeof(h);
    }
    file_end = good;
    enabled = out.good();
    if (enabled && k == CacheKind::Aot) {
        wad_seen.insert(::Gpu::PieceHash::WadName("EBOOT", 5)), wad_queue.push_back("EBOOT");  // the executable's programs are always needed
        wad_thread = std::jthread([this](const std::stop_token& stop) {
#ifdef _WIN32
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#else
            Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
#endif
            for (;;) {
                std::string wad;
                {
                    std::unique_lock q(wad_lock);
                    if (!wad_cv.wait(q, stop, [&] { return !wad_queue.empty(); })) return;
                    wad = wad_queue.front(), wad_queue.pop_front();
                }
                Warm(wad);
            }
        });
    }
}

std::uint64_t ShaderCache::ContentHash(std::span<const std::uint32_t> spirv, std::uint64_t out_mask) {
    return ::Gpu::PieceHash::ShaderContent(spirv.data(), spirv.size_bytes(), out_mask) | 1;  // never 0: that is "none"
}

bool ShaderCache::Contains(std::uint64_t key) const {
    if (!enabled) return false;
    std::shared_lock l(lock);
    return recent.contains(key) || index.contains(key);
}

std::uint64_t ShaderCache::ContentOf(std::uint64_t key) const {
    if (!enabled) return 0;
    std::shared_lock l(lock);
    const auto it = index.find(key);
    return it == index.end() ? 0 : it->second;
}

bool ShaderCache::ContainsContent(std::uint64_t content) const {
    if (!enabled) return false;
    std::shared_lock l(lock);
    return locs[0].contains(content);
}

std::uint64_t ShaderCache::GlslContent(std::uint64_t glsl_hash) const {
    if (!enabled) return 0;
    std::shared_lock l(lock);
    const auto it = glsl_map.find(glsl_hash);
    return it == glsl_map.end() ? 0 : it->second;
}

bool ShaderCache::ContainsBinary(std::uint64_t key) const {
    if (!enabled) return false;
    std::shared_lock l(lock);
    return locs[1].contains(key);
}

std::shared_ptr<ShaderCache::View> ShaderCache::Mapped(std::uint64_t end) {
    {
        std::shared_lock l(lock);
        if (view && end <= view->size) return view;
    }
    std::unique_lock l(lock);  // appended after the file was mapped: map it again
    if (view && end <= view->size) return view;
    out.flush();
    auto fresh = std::make_shared<View>();
    if (!fresh->Open(dir / CacheName(kind)) || end > fresh->size) return nullptr;
    return view = std::move(fresh);
}

std::vector<std::uint8_t> ShaderCache::Decode(std::uint32_t type, const Loc& loc) {
    const std::shared_ptr<View> v = Mapped(loc.offset + loc.packed);
    if (!v) return {};
    const std::uint8_t* src = v->data + loc.offset;
    const Dict& dict = *dicts[type];
    const std::uint32_t id = ZSTD_getDictID_fromFrame(src, loc.packed);
    ZSTD_DDict* ddict = nullptr;
    if (id) {
        if (id != dict.id.load(std::memory_order_acquire)) return {};
        ddict = dict.d.load(std::memory_order_relaxed);
        if (!ddict) return {};
    }
    std::vector<std::uint8_t> raw(loc.raw);
    thread_local const std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> context{ZSTD_createDCtx(), ZSTD_freeDCtx};
    const std::size_t n = ddict ? ZSTD_decompress_usingDDict(context.get(), raw.data(), raw.size(), src, loc.packed, ddict)
                                : ZSTD_decompressDCtx(context.get(), raw.data(), raw.size(), src, loc.packed);
    return ZSTD_isError(n) || n != raw.size() ? std::vector<std::uint8_t>{} : raw;
}

std::shared_ptr<const CachedShader> ShaderCache::Find(std::uint64_t key) {
    if (!enabled) return nullptr;
    std::uint64_t content;
    {
        std::shared_lock l(lock);
        if (const auto it = recent.find(key); it != recent.end()) return it->second;
        const auto it = index.find(key);
        if (it == index.end()) return nullptr;
        content = it->second;
    }
    return FindContent(content);
}

std::shared_ptr<const CachedShader> ShaderCache::FindContent(std::uint64_t key) {
    Loc loc;
    {
        std::shared_lock l(lock);
        if (const auto it = shaders.find(key); it != shaders.end()) return it->second;
        const auto it = locs[0].find(key);
        if (it == locs[0].end()) return nullptr;
        loc = it->second;
    }
    const auto raw = Decode(kShader, loc);
    if (raw.empty()) return nullptr;
    auto s = std::make_shared<CachedShader>();
    s->stage = loc.stage, s->out_mask = loc.out_mask, s->spirv.resize(raw.size() / 4);
    std::memcpy(s->spirv.data(), raw.data(), s->spirv.size() * 4);
    std::unique_lock l(lock);
    return shaders.try_emplace(key, std::move(s)).first->second;
}

std::shared_ptr<const CachedBinary> ShaderCache::FindBinary(std::uint64_t key) {
    if (!enabled) return nullptr;
    Loc loc;
    {
        std::shared_lock l(lock);
        const auto it = locs[1].find(key);
        if (it == locs[1].end()) return nullptr;
        loc = it->second;
    }
    auto raw = Decode(kBinary, loc);  // not kept: a binary is wanted once per run, for its shader object
    return raw.empty() ? nullptr : std::make_shared<const CachedBinary>(CachedBinary{loc.out_mask, std::move(raw)});
}

// Trains the dictionary of `type` on the records seen so far and stores it; the records written from now on use it.
void ShaderCache::Train(std::uint32_t type) {
    Dict& dict = *dicts[type];
    if (dict.c || dict.sizes.size() < Dict::MinSamples) return;
    std::vector<std::uint8_t> blob(Dict::Capacity);
    const std::size_t size = ZDICT_trainFromBuffer(blob.data(), blob.size(), dict.samples.data(), dict.sizes.data(), std::uint32_t(dict.sizes.size()));
    dict.samples = {}, dict.sizes = {};  // too few or too alike: the kind stays without dictionary
    if (ZDICT_isError(size)) return;
    blob.resize(size);
    dict.Load(blob.data(), size);
    Put(kDict, type, 0, 0, std::uint32_t(size), blob.data(), std::uint32_t(size));
}

// Writes one record at the end of the file (the lock is held) and returns where its data starts.
void ShaderCache::Put(std::uint32_t type, std::uint32_t stage, std::uint64_t key, std::uint64_t out_mask, std::uint32_t raw, const void* packed, std::uint32_t n) {
    const Entry r{key, out_mask, type, stage, raw, n};
    out.write((const char*)&r, sizeof(r));
    out.write((const char*)packed, n);
    file_end += sizeof(r) + n;
}

// Compresses (outside the lock: the AOT build runs many threads) and appends a record; false: not written.
bool ShaderCache::Append(std::uint32_t type, std::uint32_t stage, std::uint64_t key, std::uint64_t out_mask, const void* data, std::size_t bytes) {
    const ZSTD_CDict* cdict = nullptr;
    if (type < 2) {
        std::shared_lock l(lock);
        cdict = dicts[type]->c;
    }
    thread_local const std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)> context{ZSTD_createCCtx(), ZSTD_freeCCtx};
    std::vector<char> packed(ZSTD_compressBound(bytes));
    const std::size_t n = cdict ? ZSTD_compress_usingCDict(context.get(), packed.data(), packed.size(), data, bytes, cdict)
                                : ZSTD_compressCCtx(context.get(), packed.data(), packed.size(), data, bytes, kLevel);
    if (ZSTD_isError(n)) return false;
    std::unique_lock l(lock);
    if (type < 2) {
        Dict& dict = *dicts[type];
        dict.Sample(data, bytes);
        if (dict.sizes.size() >= Dict::TrainAfter) Train(type);
    }
    const std::uint64_t at = file_end + sizeof(Entry);
    Put(type, stage, key, out_mask, std::uint32_t(bytes), packed.data(), std::uint32_t(n));
    if (type < 2) locs[type][key] = {at, out_mask, std::uint32_t(n), std::uint32_t(bytes), stage}, writing[type].erase(key);
    return true;
}

void ShaderCache::Record(std::uint64_t key, std::uint32_t stage, std::uint64_t out_mask, std::span<const std::uint32_t> spirv) {
    if (!enabled || kind != CacheKind::Jit) return;
    // Kept decoded by key (the run asks for what it just translated); Flush hashes and writes it later.
    std::shared_ptr<const CachedShader> shader = std::make_shared<CachedShader>(CachedShader{stage, out_mask, {spirv.begin(), spirv.end()}});
    {
        std::unique_lock l(lock);
        if (index.contains(key) || !recent.try_emplace(key, shader).second) return;
    }
    {
        std::scoped_lock l(pending_lock);
        pending.push_back({key, out_mask, stage, std::move(shader)});
    }
    Dirty();
}

// The content is written once (the first key to reach it); every key is linked to it.
std::uint64_t ShaderCache::Commit(std::uint64_t key, std::uint32_t stage, std::uint64_t out_mask, std::span<const std::uint32_t> spirv,
                                  const std::shared_ptr<const CachedShader>& keep) {
    const std::uint64_t content = ContentHash(spirv, out_mask);
    bool write;
    {
        std::unique_lock l(lock);
        write = !locs[0].contains(content) && writing[0].insert(content).second;
        if (keep) shaders.try_emplace(content, keep);
    }
    if (write && !Append(kShader, stage, content, out_mask, spirv.data(), spirv.size() * 4)) {
        std::unique_lock l(lock);
        writing[0].erase(content), shaders.erase(content);
        return 0;
    }
    Link(key, content);
    return content;
}

std::uint64_t ShaderCache::Store(std::uint64_t key, std::uint32_t stage, std::uint64_t out_mask, std::span<const std::uint32_t> spirv) {
    return enabled ? Commit(key, stage, out_mask, spirv, nullptr) : 0;
}

void ShaderCache::Link(std::uint64_t key, std::uint64_t content) {
    if (!enabled) return;
    std::unique_lock l(lock);
    if (!index.try_emplace(key, content).second) return;
    index_new.push_back({key, content});
    if (index_new.size() >= 4096) PutPairs(kIndex, index_new);
}

void ShaderCache::LinkGlsl(std::uint64_t glsl_hash, std::uint64_t content) {
    if (!enabled) return;
    std::unique_lock l(lock);
    if (!glsl_map.try_emplace(glsl_hash, content).second) return;
    glsl_new.push_back({glsl_hash, content});
    if (glsl_new.size() >= 4096) PutPairs(kGlslMap, glsl_new);
}

void ShaderCache::PutPairs(std::uint32_t type, std::vector<Pair>& items) {
    if (items.empty()) return;
    const std::size_t bytes = items.size() * sizeof(Pair);
    std::vector<char> packed(ZSTD_compressBound(bytes));
    const std::size_t n = ZSTD_compress(packed.data(), packed.size(), items.data(), bytes, kLevel);
    if (!ZSTD_isError(n)) Put(type, 0, 0, 0, std::uint32_t(bytes), packed.data(), std::uint32_t(n));
    items.clear();
}

bool ShaderCache::StoreBinary(std::uint64_t key, std::uint64_t out_mask, std::span<const std::uint8_t> data) {
    if (!enabled) return false;
    {
        std::unique_lock l(lock);
        if (locs[1].contains(key) || !writing[1].insert(key).second) return false;
    }
    if (Append(kBinary, 0, key, out_mask, data.data(), data.size())) return true;
    std::unique_lock l(lock);
    writing[1].erase(key);
    return false;
}

void ShaderCache::SetWadKeys(const std::string& wad, std::span<const std::uint64_t> shader_keys, std::span<const std::uint64_t> binary_keys) {
    if (!enabled) return;
    const std::uint64_t h = ::Gpu::PieceHash::WadName(wad.data(), wad.size());
    std::vector<std::uint64_t> payload;
    {
        std::unique_lock l(lock);
        WadKeys& w = wad_keys[h];
        const std::size_t before = w.shaders.size() + w.binaries.size();
        for (auto [dst, src] : {std::pair{&w.shaders, shader_keys}, std::pair{&w.binaries, binary_keys}}) {
            dst->insert(dst->end(), src.begin(), src.end());
            std::sort(dst->begin(), dst->end());
            dst->erase(std::unique(dst->begin(), dst->end()), dst->end());
        }
        if (w.shaders.size() + w.binaries.size() == before) return;
        payload.push_back(w.shaders.size());
        payload.insert(payload.end(), w.shaders.begin(), w.shaders.end());
        payload.insert(payload.end(), w.binaries.begin(), w.binaries.end());
    }
    Append(kWadKeys, 0, h, 0, payload.data(), payload.size() * 8);
    Requeue(wad, h);
}

// The game opened the WAD before the AOT gave its lists: warmed again, once for the lists that arrive meanwhile.
void ShaderCache::Requeue(const std::string& wad, std::uint64_t h) {
    {
        std::scoped_lock q(wad_lock);
        if (!wad_seen.contains(h) || std::find(wad_queue.begin(), wad_queue.end(), wad) != wad_queue.end()) return;
        wad_queue.push_back(wad);
    }
    wad_cv.notify_one();
}

void ShaderCache::SetWadHot(const std::string& wad, std::vector<Hot> hot) {
    if (!enabled) return;
    const std::uint64_t h = ::Gpu::PieceHash::WadName(wad.data(), wad.size());
    {
        std::unique_lock l(lock);
        wad_hot[h] = std::move(hot);
    }
    Requeue(wad, h);
}

void ShaderCache::SetPrepare(std::function<std::uint32_t(std::uint64_t key, std::uint64_t binary, std::uint32_t stage)> fn) {
    std::scoped_lock l(prepare_cb_lock);
    prepare_cb = std::move(fn);
}

void ShaderCache::SetWarmPipelines(std::function<void()> fn) {
    std::scoped_lock l(prepare_cb_lock);
    warm_pipelines_cb = std::move(fn);
}

void ShaderCache::SetPipelineCacheFlusher(std::function<void(bool exit)> fn) {
    std::scoped_lock l(flush_cb_lock);
    flush_pipeline_cache_cb = std::move(fn);
}

void ShaderCache::ClearRendererCallbacks() {
    {
        std::scoped_lock l(prepare_cb_lock);
        prepare_cb = {};
        warm_pipelines_cb = {};
    }
    {
        std::scoped_lock l(flush_cb_lock);
        flush_pipeline_cache_cb = {};
    }
}

// The game opened the WAD. The binaries of its shaders are read ahead into the system's file cache (not decoded: the memory
// stays the system's), sorted by place in the file and joined into large pieces, one sequential read each on a hard disk:
// first those of the shaders the player's runs used there, which are then made ready in the order the runs met them, then
// the rest the WAD's programs can ask for.
void ShaderCache::Warm(const std::string& name) {
    const auto begin = std::chrono::steady_clock::now();
    const std::uint64_t wad = ::Gpu::PieceHash::WadName(name.data(), name.size());
    std::vector<Hot> hot;
    std::vector<Loc> used, rest;
    {
        std::shared_lock l(lock);
        const auto add_bin = [&](std::vector<Loc>& to, std::uint64_t k) {
            if (const auto it = locs[1].find(k); it != locs[1].end() && it->second.packed) to.push_back(it->second);
        };
        const auto add_spirv = [&](std::vector<Loc>& to, std::uint64_t key) {
            if (const auto idx = index.find(key); idx != index.end())
                if (const auto it = locs[0].find(idx->second); it != locs[0].end() && it->second.packed) to.push_back(it->second);
        };
        if (const auto it = wad_hot.find(wad); it != wad_hot.end()) hot = it->second;
        for (const Hot& h : hot) {
            if (h.binary != 0) add_bin(used, h.binary);
            else add_spirv(used, h.key);
        }
        if (const auto it = wad_keys.find(wad); it != wad_keys.end()) {
            if (!it->second.binaries.empty()) {
                for (std::uint64_t k : it->second.binaries) add_bin(rest, k);
            } else {
                for (std::uint64_t k : it->second.shaders) add_spirv(rest, k);
            }
        }
    }
    std::uint64_t bytes = 0, pieces = 0;
    const auto ahead = [&](std::vector<Loc>& at) {
        if (at.empty()) return;
        std::sort(at.begin(), at.end(), [](const Loc& a, const Loc& b) { return a.offset < b.offset; });
        const std::shared_ptr<View> v = Mapped(at.back().offset + at.back().packed);
        if (!v) return;
        constexpr std::uint64_t kGap = 512u << 10;  // read through a gap this small: cheaper than a seek
#ifdef _WIN32
        std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
#else
        const auto page_mask = std::uint64_t(getpagesize() - 1);
#endif
        for (std::uint64_t from = at[0].offset, to = from, k = 0; k <= at.size(); ++k) {
            if (k == at.size() || at[k].offset > to + kGap) {
#ifdef _WIN32
                ranges.push_back({const_cast<std::uint8_t*>(v->data) + from, SIZE_T(to - from)});
#else
                const auto start = from & ~page_mask;
                madvise(const_cast<std::uint8_t*>(v->data) + start, to - start, MADV_WILLNEED);
                ++pieces;
#endif
                bytes += to - from;
                if (k == at.size()) break;
                from = at[k].offset;
            }
            to = std::max(to, at[k].offset + at[k].packed);
        }
#ifdef _WIN32
        pieces += ranges.size();
        PrefetchVirtualMemory(GetCurrentProcess(), ranges.size(), ranges.data(), 0);
#endif
    };
    ahead(used);
    std::uint32_t made[3]{};  // already there (another WAD's or a draw's), made now, failed
    if (!hot.empty()) {
        for (const Hot& h : hot) {
            std::scoped_lock l(prepare_cb_lock);
            if (!prepare_cb) break;
            ++made[std::min(prepare_cb(h.key, h.binary, h.stage), 2u)];
        }
        std::scoped_lock l(prepare_cb_lock);
        if (warm_pipelines_cb) warm_pipelines_cb();
    }
    ahead(rest);
    if (!hot.empty() || !rest.empty())
        Gow3::Log(std::format("Shader cache: {}: {:.1f} MB of binaries read ahead in {} pieces; of {} used shaders {} made ready, {} were, {} failed ({:.0f} ms)", name,
                              bytes / 1048576.0, pieces, hot.size(), made[1], made[0], made[2],
                              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count()));
}

void ShaderCache::WadOpened(const char* path) {
    const std::string name = WadName(path);
    if (name.empty()) return;
    wad_opens.fetch_add(1, std::memory_order_relaxed);
    if (!enabled || kind != CacheKind::Aot) return;
    {
        std::scoped_lock q(wad_lock);
        if (!wad_seen.insert(::Gpu::PieceHash::WadName(name.data(), name.size())).second) return;
        wad_queue.push_back(name);
    }
    wad_cv.notify_one();
}

void ShaderCache::Dirty() {
    if (dirty.exchange(true, std::memory_order_acq_rel)) return;
    std::scoped_lock l(flush_lock);
    if (!flusher.joinable()) flusher = std::jthread([this](const std::stop_token& stop) { FlushLoop(stop); });
    flush_cv.notify_one();
}

void ShaderCache::Flush(const char* why, bool all_threads) {
    const auto begin = std::chrono::steady_clock::now();
    dirty.store(false, std::memory_order_release);  // what is noted from here on wakes the flush thread again
    std::vector<Pending> batch;
    {
        std::scoped_lock l(pending_lock);
        batch.swap(pending);
    }
    Recipes::Instance().Flush();
    std::atomic<std::size_t> next = 0;
    const auto work = [&] {
        for (std::size_t k; (k = next++) < batch.size();) {
            const Pending& p = batch[k];
            Commit(p.key, p.stage, p.out_mask, p.shader->spirv, p.shader);
            std::unique_lock l(lock);
            recent.erase(p.key);  // from here the index finds it
        }
    };
    if (all_threads) {
        std::vector<std::jthread> workers;
        for (std::size_t t = 0, n = std::min<std::size_t>(batch.size(), Gow3::Cpu::plan.shader_threads); t < n; ++t) workers.emplace_back(work);
    } else {
        work();
    }
    {
        std::unique_lock l(lock);
        PutPairs(kIndex, index_new), PutPairs(kGlslMap, glsl_new);
    }
    {
        std::scoped_lock l(flush_cb_lock);
        if (flush_pipeline_cache_cb) flush_pipeline_cache_cb(all_threads);
    }
    if (!batch.empty())
        std::fprintf(stderr, "Shader cache: wrote %zu shaders (%s) in %.0f ms\n", batch.size(), why,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
}

void ShaderCache::FlushLoop(const std::stop_token& stop) {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#else
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
#endif
    constexpr auto kTick = std::chrono::milliseconds(500);
    constexpr std::size_t kWindow = 4;                // ticks: what the game does within two seconds
    constexpr std::uint64_t kLoadBytes = 128u << 20;  // streamed into guest memory within the window: a loading screen
    constexpr std::uint32_t kLoadWads = 2;            // WADs opened within the window: the same
    constexpr std::size_t kBurst = 100;               // shaders waiting once the compilations are done
    for (;;) {
        {
            std::unique_lock q(flush_lock);
            if (!flush_cv.wait(q, stop, [&] { return dirty.load(std::memory_order_acquire); })) return;  // nothing waits: asleep
        }
        std::uint64_t bytes[kWindow]{};
        std::uint32_t wads[kWindow]{};
        streamed.store(0, std::memory_order_relaxed), wad_opens.store(0, std::memory_order_relaxed);  // only what happens from now on counts
        for (std::size_t tick = 0; dirty.load(std::memory_order_acquire); ++tick) {
            {
                std::unique_lock q(flush_lock);
                flush_cv.wait_for(q, stop, kTick, [] { return false; });
            }
            if (stop.stop_requested()) return;
            bytes[tick % kWindow] = streamed.exchange(0, std::memory_order_relaxed), wads[tick % kWindow] = wad_opens.exchange(0, std::memory_order_relaxed);
            std::uint64_t streamed_now = 0;
            std::uint32_t wads_now = 0;
            for (std::size_t k = 0; k < kWindow; ++k) streamed_now += bytes[k], wads_now += wads[k];
            std::size_t waiting;
            {
                std::scoped_lock l(pending_lock);
                waiting = pending.size();
            }
            const Progress progress = Snapshot();
            const bool loading = streamed_now >= kLoadBytes || wads_now >= kLoadWads;
            const bool burst = waiting >= kBurst && progress.done == progress.total && progress.pipelines_done == progress.pipelines_total;
            if (!loading && !burst) continue;
            char why[96];
            if (loading) std::snprintf(why, sizeof why, "loading: %llu MB streamed, %u WADs in 2 s", (unsigned long long)(streamed_now >> 20), wads_now);
            else std::snprintf(why, sizeof why, "compile burst: %zu waiting", waiting);
            Flush(why, false);
        }
    }
}

void ShaderCache::Finish() {
    Flush("exit", true);  // everything still waiting, on every thread: the game is over
    if (!enabled) return;
    std::unique_lock l(lock);
    Train(kShader), Train(kBinary);  // a short session still leaves its dictionary for the next one
    out.flush();
}
} // namespace Gpu
