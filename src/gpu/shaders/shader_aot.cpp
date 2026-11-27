// SPDX-License-Identifier: GPL-2.0-or-later
#include "gpu/shaders/shader_aot.h"
#include <cstdarg>
#include <deque>
#include <fstream>
#include <map>
#include "main/cpu.h"
#include "main/host.h"
#include "gpu/shaders/shader_cache.h"
#include "gpu/shaders/shader_catalog.h"
#include "gpu/shaders/shader_contract.h"
#include "gpu/shaders/shader_parser.h"
#include "gpu/shaders/shader_recipes.h"
#include "gpu/renderer_impl.h"

namespace Gpu {
namespace {
// The AOT report: the console and gow3.log.
[[gnu::format(printf, 1, 2)]] void Say(const char* format, ...) {
    char text[2048];
    va_list args;
    va_start(args, format);
    const int n = std::vsnprintf(text, sizeof text, format, args);
    va_end(args);
    std::fputs(text, stderr);
    std::string_view line(text, size_t(std::clamp(n, 0, int(sizeof text) - 1)));
    while (!line.empty() && line.back() == '\n') line.remove_suffix(1);
    if (!line.empty()) Gow3::Log(line);
}

struct Job {
    Recipe recipe;
    u64 key;
    bool spirv;                   // the key was in aot.cache when the job was made (only its binary was missing)
    const Parser::Program* prog;  // where its code is (null: not in the WADs)
    std::vector<u32> files;       // the parser's files that hold the program, with the same fetch shader
    std::span<const u8> code, fetch;  // contract: the program in the catalog's mappings (else found through `prog`)
};

// The contract of the game's data (what shader_contract.cpp derives from reading the EBOOT and every WAD), kept so the next
// launches do not read them again: each variant with where its code lies in the parser's files. Only a complete reading is
// kept, and only while the game's files, the sample class and the versions of what it derives are what it was made from.
struct Kept {
    Recipe recipe;
    u64 key, code_start, fetch_start;
    u32 file, paths, code_size, fetch_size;  // file: the parser's files
};
constexpr u32 kKeptMagic = 0x43433347, kKeptFormat = 2;  // "G3CC"; kKeptFormat: the layout below and the variant rules
struct KeptHead { u32 magic, format; u64 stamp; u32 count, reserved; };
struct KeptFixed { RecipeHead h; u64 key, code_start, fetch_start; u32 file, paths, code_size, fetch_size, fetch_words, reserved; };

// The scene's sample count as the contract sees it: x1 uses single-sampled views, x2/x4 multisampled ones, and the
// pipelines leave it 0. Switching between x1 and x2/x4 rebuilds the contract for the active sample count.
u32 SampleClass() { return Aot::sample_log2; }

// `version`: of the rules the kept list comes from (kKeptFormat, kAotPipelineVersion); its keys follow kAotCacheVersion.
u64 KeptStamp(u64 files_stamp, bool no_storage_read_nofmt, u32 version) {
    const u64 parts[]{files_stamp, SampleClass(), kAotCacheVersion, version};
    const u64 h = PieceHash::Stable64(parts, sizeof(parts));
    return no_storage_read_nofmt ? (h ^ 0x53544F524E4F5244ull) : h;
}

bool LoadKept(const std::filesystem::path& path, u64 stamp, std::vector<Kept>& out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    std::vector<u8> bytes(size_t(in.tellg()));
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()))) return false;
    KeptHead head;
    if (bytes.size() < sizeof(head)) return false;
    std::memcpy(&head, bytes.data(), sizeof(head));
    if (head.magic != kKeptMagic || head.format != kKeptFormat || head.stamp != stamp) return false;
    size_t at = sizeof(head);
    out.reserve(head.count);
    for (u32 k = 0; k < head.count; ++k) {
        KeptFixed fixed;
        if (bytes.size() - at < sizeof(fixed)) return out.clear(), false;
        std::memcpy(&fixed, bytes.data() + at, sizeof(fixed));
        at += sizeof(fixed);
        const size_t spec_bytes = fixed.h.nspec * 8ull, fetch_bytes = fixed.fetch_words * 4ull;
        if (bytes.size() - at < spec_bytes + fetch_bytes) return out.clear(), false;
        Kept& v = out.emplace_back();
        v.recipe.h = fixed.h;
        v.recipe.spec.resize(fixed.h.nspec), v.recipe.fetch.resize(fixed.fetch_words);
        std::memcpy(v.recipe.spec.data(), bytes.data() + at, spec_bytes);
        std::memcpy(v.recipe.fetch.data(), bytes.data() + at + spec_bytes, fetch_bytes);
        at += spec_bytes + fetch_bytes;
        v.key = fixed.key, v.code_start = fixed.code_start, v.fetch_start = fixed.fetch_start;
        v.file = fixed.file, v.paths = fixed.paths, v.code_size = fixed.code_size, v.fetch_size = fixed.fetch_size;
    }
    return at == bytes.size() || (out.clear(), false);
}

void SaveKept(const std::filesystem::path& path, u64 stamp, const std::vector<Kept>& list) {
    std::vector<u8> bytes;
    const auto append = [&](const void* data, size_t size) { bytes.insert(bytes.end(), static_cast<const u8*>(data), static_cast<const u8*>(data) + size); };
    const KeptHead head{kKeptMagic, kKeptFormat, stamp, u32(list.size()), 0};
    append(&head, sizeof(head));
    for (const Kept& v : list) {
        KeptFixed fixed{};
        fixed.h = v.recipe.h, fixed.key = v.key, fixed.code_start = v.code_start, fixed.fetch_start = v.fetch_start;
        fixed.file = v.file, fixed.paths = v.paths, fixed.code_size = v.code_size, fixed.fetch_size = v.fetch_size, fixed.fetch_words = u32(v.recipe.fetch.size());
        append(&fixed, sizeof(fixed)), append(v.recipe.spec.data(), v.recipe.spec.size() * 8), append(v.recipe.fetch.data(), v.recipe.fetch.size() * 4);
    }
    auto temporary = path;
    temporary += ".tmp";  // written whole, then put in place: a launch closed in the middle leaves no half file
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()))) return;
    }
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
}

constexpr u32 kKeptPlMagic = 0x50433347, kKeptPlFormat = 2;  // "G3CP"; 2: ContractedPipeline with its gate
struct KeptPlHead { u32 magic, format; u64 stamp; u32 count, reserved; };

bool LoadKeptPipelines(const std::filesystem::path& path, u64 stamp, std::vector<ContractedPipeline>& out) {
    out.clear();
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    const auto size = in.tellg();
    if (size < std::streamoff(sizeof(KeptPlHead))) return false;
    in.seekg(0);
    KeptPlHead head{};
    if (!in.read(reinterpret_cast<char*>(&head), sizeof(head))) return false;
    if (head.magic != kKeptPlMagic || head.format != kKeptPlFormat || head.stamp != stamp) return false;
    const size_t body = size_t(head.count) * sizeof(ContractedPipeline);
    if (size_t(size) != sizeof(head) + body) return false;
    out.resize(head.count);
    if (body && !in.read(reinterpret_cast<char*>(out.data()), std::streamsize(body))) return out.clear(), false;
    return true;
}

void SaveKeptPipelines(const std::filesystem::path& path, u64 stamp, const std::vector<ContractedPipeline>& list) {
    const KeptPlHead head{kKeptPlMagic, kKeptPlFormat, stamp, u32(list.size()), 0};
    auto temporary = path;
    temporary += ".tmp";
    std::error_code ec;
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&head), sizeof(head));
        if (!list.empty()) out.write(reinterpret_cast<const char*>(list.data()), std::streamsize(list.size() * sizeof(ContractedPipeline)));
        out.flush();
        if (!out.good()) {
            out.close();
            std::filesystem::remove(temporary, ec);
            return;
        }
    }
    std::filesystem::rename(temporary, path, ec);
    if (ec) std::filesystem::remove(temporary, ec);
}

} // namespace

// Gets every variant the game can ask for into the cache, in this order: the recipes (what the player's runs and the JIT
// misses of the gameplay met), then the programs of the game's data in the order of the parse (the files sorted by name,
// the executable first). Lazy AOT builds all embedded variants and up to a quarter of the full WAD total per eligible launch.
// Each becomes SPIR-V in aot.cache (translated from the code in the game's files, once) and a binary for this GPU and driver (made again after a
// driver update, from the SPIR-V). All CPU threads work on it. A launch in the cooldown after a build compiles nothing, but its
// WADs still make ready the shaders the player's runs used there.
void Renderer::Impl::AotBuild() {
    Aot::has_pipelines_phase.store(caps.path != ShaderPath::Objects, std::memory_order_release);
    struct FinishProgress { ~FinishProgress() { Aot::SetPhase(Aot::Phase::Finished); } } finish_progress;
    const bool build = Gow3::AotBuildAllowed();
    struct FinishBuild { ~FinishBuild() { Gow3::EndAotBuild(); } } finish_build;
    Aot::SetPhase(Aot::Phase::Index);
    static constexpr VkShaderStageFlagBits vk_stage[]{VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_COMPUTE_BIT};
    ShaderCache& cache = ShaderCache::Instance();
    const auto t0 = std::chrono::steady_clock::now();
    const auto seconds = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    const bool no_srnf = !caps.storage_image_read_without_format;
    // The recipes as this GPU's runtime keys them, whichever GPU met them (the release's come from another one).
    std::vector<Recipe> recipes = Recipes::Instance().All();
    // What the recipes met orders the lazy AOT's work: their variants' keys (with the contract's PS variants of the programs
    // they met, added below) and their PS programs' code.
    std::unordered_set<u64> met_keys, met_ps;
    std::erase_if(recipes, [&](Recipe& r) { return r.Adapt(no_srnf), !met_keys.insert(r.Key()).second; });
    for (const Recipe& r : recipes) if (r.h.stage == u8(Gcn::Stage::Pixel)) met_ps.insert(r.h.code_hash);
    const auto note_met = [&](const Recipe& r, u64 key) {
        if (r.h.stage == u8(Gcn::Stage::Pixel) && met_ps.contains(r.h.code_hash)) met_keys.insert(key);
    };
    const auto pending_of = [&](u64 key) {
        return !cache.Contains(key) || (caps.path == ShaderPath::Objects && !cache.ContainsBinary(BinaryKey(cache.ContentOf(key))));
    };
    // The built-in clear VS handles SDK clears; other RectLists use the ordinary VS and an on-demand RectGs.
    const auto rect_variant = [&](const Recipe& r) { return HasClearRectVs() && r.h.stage == u32(Gcn::Stage::Vertex) && r.h.rect; };

    Parser::Index index;
    try {
        index = Parser::Load(Aot::game_directory, cache.Directory() / (PieceHash::UsesAes() ? "wad-hash2-aes.index" : "wad-hash2-xxh.index"), [&](u32 n) { for (u32 k = 0; k < n; ++k) cache.Queued(); },
                             [&] { cache.Completed(); });
    } catch (const std::exception& e) {
        Say("AOT: %s\n", e.what());
        return;
    }
    Say("AOT: %zu programs in %zu files of the game (%.1f s)\n", index.programs.size(), index.files.size(), seconds());
    std::unordered_map<u64, std::vector<u32>> by_code;  // programs of the index by code hash
    for (u32 k = 0; k < index.programs.size(); ++k) by_code[index.programs[k].code_hash].push_back(k);

    std::vector<std::unique_ptr<Parser::Mapping>> maps(index.files.size());
    std::mutex lock;  // maps, wad_keys, reasons
    const auto source = [&](u32 file) -> const Parser::Mapping* {
        std::scoped_lock l(lock);
        if (!maps[file]) {
            maps[file] = std::make_unique<Parser::Mapping>();
            if (!maps[file]->Open(Aot::game_directory / index.files[file])) maps[file].reset();
        }
        return maps[file].get();
    };
    // A recipe's code: the parser's programs with its hashes, and the files that hold them.
    const auto files_of = [&](const Recipe& r, std::vector<u32>& files) {
        const Parser::Program* p = nullptr;
        if (const auto it = by_code.find(r.h.code_hash); it != by_code.end())
            for (u32 pi : it->second)
                // A recipe that carries its own fetch shader (the engine generated it) needs only the program's code.
                if (!r.fetch.empty() || index.programs[pi].fetch_hash == r.h.fetch_hash)
                    p = p ? p : &index.programs[pi], files.push_back(index.programs[pi].file);
        return p;
    };
    // What the player's runs used (with a binary), per file of its program, in the order they met it: made ready when the game
    // opens the file (or now, for a file already open).
    const auto give_hot = [&] {
        std::vector<std::vector<ShaderCache::Hot>> hot(index.files.size());
        std::vector<u32> files;
        for (const Recipe& r : recipes) {
            if (rect_variant(r)) continue;
            const u64 key = r.Key();
            files.clear();
            if (caps.path == ShaderPath::Objects) {
                const u64 binary = BinaryKey(cache.ContentOf(key));
                if (cache.ContainsBinary(binary) && files_of(r, files))
                    for (u32 f : files) hot[f].push_back({key, binary, r.h.stage});
            } else if (cache.Contains(key) && files_of(r, files)) {
                for (u32 f : files) hot[f].push_back({key, 0, r.h.stage});
            }
        }
        for (u32 f = 0; f < index.files.size(); ++f)
            if (const std::string name = f ? WadName(index.files[f].c_str()) : "EBOOT"; !name.empty() && !hot[f].empty()) cache.SetWadHot(name, std::move(hot[f]));
    };
    give_hot();
    if (!build) {
        Say("AOT: recent build; skipping all startup compilation for this launch (used shaders listed for their WADs, %.1f s)\n", seconds());
        return;
    }
    const auto place = [&](Job& job) { job.prog = files_of(job.recipe, job.files); };
    const auto parallel = [&](size_t count, auto&& body) {
        std::atomic<size_t> next = 0;
        std::vector<std::jthread> workers;
        for (unsigned t = 0; t < Gow3::Cpu::plan.shader_threads; ++t)
            workers.emplace_back([&] {
                for (size_t k; (k = next++) < count;) try {
                    body(k);
                } catch (const std::exception& e) {
                    Say("AOT: %s\n", e.what());
                }
            });
    };
    // The program's code and fetch shader as translator input (copied: the translator wants aligned words).
    const auto load = [&](const Job& job, std::vector<u32>& code, std::vector<u32>& fetch) {
        if (!job.code.empty()) {
            code.resize(job.code.size() / 4), fetch.resize(job.fetch.size() / 4);
            std::memcpy(code.data(), job.code.data(), code.size() * 4);
            if (!fetch.empty()) std::memcpy(fetch.data(), job.fetch.data(), fetch.size() * 4);
            if (!job.recipe.fetch.empty()) fetch = job.recipe.fetch;
            return true;
        }
        const Parser::Mapping* map = job.prog ? source(job.prog->file) : nullptr;
        if (!map) return false;
        code.resize(job.prog->size / 4);
        std::memcpy(code.data(), map->data + job.prog->start, code.size() * 4);
        if (!job.recipe.fetch.empty()) {
            fetch = job.recipe.fetch;
        } else {
            fetch.resize(job.prog->fetch_size / 4);
            if (!fetch.empty()) std::memcpy(fetch.data(), map->data + job.prog->fetch_start, fetch.size() * 4);
        }
        return true;
    };

    std::vector<std::vector<u64>> wad_keys(index.files.size()), wad_binaries(index.files.size());
    // emitted: SPIR-V made by the translator; stored: of those, the SPIR-V the cache did not hold; binaries: the driver's compiles
    // (a content is compiled once). The times are the TEMP split of a job.
    std::atomic<u32> emitted = 0, stored = 0, binaries = 0, missing = 0, failed = 0;
    std::atomic<u64> spirv_bytes = 0, binary_bytes = 0, emit_ns = 0, driver_ns = 0;
    std::unordered_set<u64> building;  // binary keys being compiled by a job (under `lock`)
    std::map<std::string, u32> reasons;  // why variants failed, under `lock`
    const auto fail = [&](const std::string& why) {
        std::scoped_lock l(lock);
        ++reasons[why.substr(0, 100)];
    };
    // What a job costs, for the remaining-time estimate: the driver's compile is ~98% of the time and follows the program's size.
    const auto weight = [](const Job& job) { return 64u + u32((job.code.size() ? job.code.size() : job.prog ? job.prog->size : 0) / 4); };
    // Builds the jobs. Prints what came of them (the first runs of a generator are read from this).
    const auto run = [&](const char* what, std::vector<Job>& jobs) {
        if (!jobs.empty()) Gow3::BeginAotBuild();
        emitted = stored = binaries = missing = failed = 0, spirv_bytes = binary_bytes = emit_ns = driver_ns = 0, reasons.clear();
        const auto start = std::chrono::steady_clock::now();
        const auto since = [](std::chrono::steady_clock::time_point t) { return u64((std::chrono::steady_clock::now() - t).count()); };
        for (const Job& job : jobs) cache.Queued(weight(job));
        // The cheapest way to a job's result first: the key is indexed (nothing to translate); else the translator emits the SPIR-V,
        // stored only if no variant made it already. The driver compiles a content once: a binary depends on the SPIR-V and its
        // stage only (Create's other inputs are fixed).
        parallel(jobs.size(), [&](size_t k) {
            Job& job = jobs[k];
            const Recipe& r = job.recipe;
            const auto stage = Gcn::Stage(r.h.stage);
            const u32 w = weight(job);
            u64 content = cache.ContentOf(job.key);
            if (!content) {
                std::vector<u32> code, fetch;
                if (!load(job, code, fetch)) return ++missing, cache.Completed(w);
                const auto t = std::chrono::steady_clock::now();
                Gcn::Input in = r.Input();
                in.code = code, in.fetch = fetch;
                std::string err;
                std::vector<u32> spirv;
                u64 out_mask = 0;
                const bool ok = Gcn::EmitSpirv(Gcn::Analyze(in), in, r.Specs(), spirv, out_mask, err, true);
                emit_ns += since(t), ++emitted;
                if (!ok) {
                    ++failed, fail(std::format("stage {}: {}", (int)stage, err));
                    return cache.Completed(w);
                }
                stored += !cache.ContainsContent(ShaderCache::ContentHash(spirv, out_mask)), spirv_bytes += spirv.size() * 4;
                if (!(content = cache.Store(job.key, r.h.stage, out_mask, spirv))) return ++failed, fail("cache write failed"), cache.Completed(w);
            }
            bool has_binary = false;
            u64 binary_key = 0;
            if (caps.path == ShaderPath::Objects) {
                binary_key = BinaryKey(content);
                has_binary = cache.ContainsBinary(binary_key);
                if (!has_binary) {
                    {
                        std::scoped_lock l(lock);
                        has_binary = !building.insert(binary_key).second;  // another job is making it: listed for the WAD as if done
                    }
                    if (!has_binary) {  // the driver's code for this GPU, so the game loads it without compiling
                        auto hit = cache.Find(job.key);
                        // Same content from another job: it is linked as soon as reserved, its writer may still be compressing it.
                        for (int wait = 0; !hit && wait < 2000; ++wait) std::this_thread::sleep_for(std::chrono::milliseconds(1)), hit = cache.Find(job.key);
                        const VkShaderStageFlags next_stage = stage == Gcn::Stage::Vertex ? VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_GEOMETRY_BIT : 0;
                        const auto t = std::chrono::steady_clock::now();
                        if (!hit) {
                            ++failed, fail("cached SPIR-V unreadable");
                        } else if (const VkShaderEXT s = Create(hit->spirv, vk_stage[r.h.stage], next_stage)) {
                            const auto data = Binary(s);
                            has_binary = !data.empty() && cache.StoreBinary(binary_key, hit->out_mask, data);
                            DestroyShader(dev, s, nullptr);
                            binaries += has_binary, binary_bytes += data.size();
                        } else {
                            fail(std::format("stage {}: the driver rejected the SPIR-V", (int)stage));
                        }
                        driver_ns += since(t);
                    }
                }
            }
            std::scoped_lock l(lock);
            for (u32 f : job.files) {
                wad_keys[f].push_back(job.key);
                if (has_binary) wad_binaries[f].push_back(binary_key);
            }
            cache.Completed(w);
        });
        const auto each = [](std::atomic<u64>& ns, u32 n) { return n ? ns.load() / 1e6 / n : 0.0; };
        Say(
                     "AOT: %s: %zu jobs: %u emitted as SPIR-V (%.3f ms each, %.1f KB), %u new SPIR-V stored, %u binaries (%.1f KB, %.2f ms each), "
                     "%u without code, %u failed (%.1f s)\n",
                     what, jobs.size(), emitted.load(), each(emit_ns, emitted), spirv_bytes.load() / 1024.0, stored.load(),
                     binaries.load(), binary_bytes.load() / 1024.0, each(driver_ns, binaries), missing.load(), failed.load(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        for (const auto& [why, count] : reasons) Say("AOT:   %u x %s\n", count, why.c_str());
    };

    // The recipes first, all of them: what the player's runs met and what the JIT had to translate during play.
    std::vector<Job> jobs;
    std::vector<u64> compute_keys;
    for (const Recipe& r : recipes) {
        const u64 key = r.Key();
        if (r.h.stage == u8(Gcn::Stage::Compute)) compute_keys.push_back(key);
        if (!rect_variant(r) && (!(r.h.reserved & 0xFFFFu) || blend_interlock) && pending_of(key)) jobs.push_back({r, key, cache.Contains(key), nullptr, {}});
    }
    for (Job& job : jobs) place(job);
    // Grouped by WAD (each by its first file, in the order the runs met them): the file keeps what a WAD's runs used together.
    std::stable_sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) { return (a.files.empty() ? ~0u : a.files[0]) < (b.files.empty() ? ~0u : b.files[0]); });
    {   // TEMP: why recipes have no code (stage: found by code hash only / not found at all)
        std::map<std::string, u32> lost;
        for (const Job& job : jobs)
            if (!job.prog) ++lost[std::format("stage {} fetch {} code {}", job.recipe.h.stage, job.recipe.h.fetch_hash ? "yes" : "none", by_code.contains(job.recipe.h.code_hash) ? "known" : "unknown")];
        for (const auto& [why, count] : lost) Say("AOT:   no code: %u %s\n", count, why.c_str());
    }
    Aot::SetPhase(Aot::Phase::Recipes);
    run("recipes", jobs);

    // Then everything the game's own data implies: every program of the catalog with the configurations the engine can
    // request, in the order of the parse. Cached WAD variants still count toward the fixed lazy budget; recipes and
    // embedded variants do not consume it. The catalog's mappings stay open while the jobs read their code from them.
    std::unordered_map<std::string, u32> file_of;  // WAD name -> index of the parser's files
    for (u32 f = 1; f < index.files.size(); ++f) file_of.emplace(WadName(index.files[f].c_str()), f);
    std::unique_ptr<Catalog::Data> catalog;
    jobs.clear();
    Aot::SetPhase(Aot::Phase::Catalog);
    const bool need_pl = caps.path != ShaderPath::Objects;
    const auto kept_path = cache.Directory() / (PieceHash::UsesAes() ? "contract-hash2-aes.cache" : "contract-hash2-xxh.cache");
    const auto kept_pl_path = cache.Directory() / (PieceHash::UsesAes() ? "pipeline-contract-hash2-aes.cache" : "pipeline-contract-hash2-xxh.cache");
    const u64 kept_stamp = KeptStamp(index.stamp, no_srnf, kKeptFormat), kept_pl_stamp = KeptStamp(index.stamp, no_srnf, kAotPipelineVersion);
    std::vector<Kept> kept;
    std::vector<ContractedPipeline> kept_pipelines;
    std::deque<Parser::Program> kept_code;  // where the code of the variants read from the file lies, for their jobs
    size_t wad_total = 0, embedded_pending = 0;
    if (LoadKept(kept_path, kept_stamp, kept) && (!need_pl || LoadKeptPipelines(kept_pl_path, kept_pl_stamp, kept_pipelines))) {
        Say("AOT: contract of %zu variants read from the cache (%.1f s)\n", kept.size(), seconds());
        for (Kept& v : kept) {
            note_met(v.recipe, v.key);
            if (v.recipe.h.stage == u8(Gcn::Stage::Compute)) compute_keys.push_back(v.key);
            if (rect_variant(v.recipe)) continue;
            wad_total += v.file != 0;
            if (!pending_of(v.key)) continue;
            embedded_pending += v.file == 0;
            Parser::Program& at = kept_code.emplace_back();
            at.file = v.file, at.start = v.code_start, at.size = v.code_size, at.fetch_start = v.fetch_start, at.fetch_size = v.fetch_size;
            jobs.push_back({std::move(v.recipe), v.key, cache.Contains(v.key), &at, {v.file}});
        }
    } else try {
        kept.clear(), kept_pipelines.clear();
        catalog = std::make_unique<Catalog::Data>(Catalog::Read(Aot::game_directory, Gow3::Cpu::plan.shader_threads, SampleClass()));
        Say("AOT: catalog of %zu WADs and %zu embedded programs (%.1f s)\n", catalog->files.size(), catalog->embedded.size(), seconds());
        for (const auto& file : catalog->files)
            if (!file.fatal.empty()) Say("AOT:   unreadable: %s\n", file.fatal.c_str());
        std::vector<Contracted> contracts;
        Contracts(*catalog, contracts, need_pl ? &kept_pipelines : nullptr);
        {   // TEMP: how many of the player's recipes the contract predicted (all / with the engine's own fetch shader), and per path
            std::unordered_map<u64, u32> have;  // key -> paths
            u32 made[kContractPaths]{}, hits[kContractPaths]{}, alone[kContractPaths]{};
            for (const Contracted& c : contracts) {
                have.emplace(c.key, c.paths);
                for (u32 p = 0; p < kContractPaths; ++p) made[p] += c.paths >> p & 1;
            }
            {   // how the variants spread over programs, per stage
                std::unordered_set<u64> programs[3];
                u32 variants[3]{};
                for (const Contracted& c : contracts) programs[c.recipe.h.stage].insert(c.recipe.h.code_hash), ++variants[c.recipe.h.stage];
                static constexpr const char* names[]{"vs", "ps", "cs"};
                for (u32 s = 0; s < 3; ++s) Say("AOT:   %s: %u variants over %zu programs\n", names[s], variants[s], programs[s].size());
            }
            u32 all = 0, hit = 0, with_fetch = 0, hit_fetch = 0;
            static const bool list_missed = DebugAuto("GOW3_TEMP_AOT_MISSED");  // thousands of lines: debug runs only
            for (const Recipe& r : recipes) {
                const auto found = have.find(r.Key());
                const bool in = found != have.end(), vs = !r.fetch.empty();
                ++all, hit += in, with_fetch += vs, hit_fetch += vs && in;
                if (in)
                    for (u32 p = 0; p < kContractPaths; ++p) hits[p] += found->second >> p & 1, alone[p] += found->second == 1u << p;
                if (in || !list_missed) continue;
                std::string line = std::format("AOT:   missed stage {} code {:016x} mrt0 {:02x} specs", r.h.stage, r.h.code_hash, r.h.mrt_map[0]);
                for (u64 s : r.spec) line += std::format(" {:x}", s);
                Say("%s\n", line.c_str());
            }
            Say("AOT: contract predicted %u of %u recipes (%u of %u with a generated fetch shader)\n", hit, all, hit_fetch, with_fetch);
            for (u32 p = 0; p < kContractPaths; ++p)
                Say("AOT:   path %-22s %7u variants, %5u recipes predicted (%u only by it)\n", kContractPathNames[p], made[p], hits[p], alone[p]);
        }
        // Kept for the next launches: a reading that got every file, each variant's code placed in the parser's files.
        bool complete = std::ranges::all_of(catalog->files, [](const auto& file) { return file.fatal.empty(); });
        std::vector<u32> parser_file(catalog->files.size(), ~0u);  // Catalog::Data::files -> the parser's files
        for (size_t f = 0; f < parser_file.size(); ++f)
            if (const auto it = file_of.find(WadName(catalog->files[f].map->path.string().c_str())); it != file_of.end()) parser_file[f] = it->second;
        kept.reserve(contracts.size());
        for (Contracted& c : contracts) {
            const u32 file = c.file == ~0u ? 0 : parser_file[c.file];
            const Catalog::Mapping& map = c.file == ~0u ? *catalog->eboot : *catalog->files[c.file].map;
            const auto inside = [&](std::span<const u8> s) { return s.empty() || (s.data() >= map.data && s.data() + s.size() <= map.data + map.size); };
            if (file == ~0u || !inside(c.code) || !inside(c.fetch)) complete = false;
            else if (complete) kept.push_back({c.recipe, c.key, u64(c.code.data() - map.data), c.fetch.empty() ? 0 : u64(c.fetch.data() - map.data), file, c.paths, u32(c.code.size()), u32(c.fetch.size())});
            note_met(c.recipe, c.key);
            if (c.recipe.h.stage == u8(Gcn::Stage::Compute)) compute_keys.push_back(c.key);
            if (rect_variant(c.recipe)) continue;
            wad_total += c.file != ~0u;
            if (!pending_of(c.key)) continue;
            embedded_pending += c.file == ~0u;
            Job job{std::move(c.recipe), c.key, cache.Contains(c.key), nullptr, {}, c.code, c.fetch};
            if (file != ~0u) job.files = {file};
            jobs.push_back(std::move(job));
        }
        if (complete) {
            SaveKept(kept_path, kept_stamp, kept);
            if (need_pl) SaveKeptPipelines(kept_pl_path, kept_pl_stamp, kept_pipelines);
            Say("AOT: contract of %zu variants kept for the next launches (%.1f s)\n", kept.size(), seconds());
        }
    } catch (const std::exception& e) {
        Say("AOT: catalog: %s\n", e.what());
    }
    const size_t pending = jobs.size();
    const size_t wad_budget = (wad_total + 3) / 4;
    // Lazy AOT builds a quarter of the WAD variants per launch, the likeliest first: the EBOOT's, the PS variants of programs the
    // recipes met, the other PS, then the VS (the recipes' own are built already); parse order within each.
    const auto rank = [&](const Job& j) {
        if (!j.files.empty() && j.files[0] == 0) return 0;
        const auto stage = Gcn::Stage(j.recipe.h.stage);
        return stage == Gcn::Stage::Vertex ? 3 : stage == Gcn::Stage::Pixel && met_ps.contains(j.recipe.h.code_hash) ? 1 : 2;
    };
    std::ranges::stable_sort(jobs, {}, rank);
    if (Aot::lazy) jobs.resize(std::min(pending, embedded_pending + wad_budget));
    Say("AOT: contract: %zu WAD variants total, %zu pending (%zu embedded), %zu started now%s\n",
        wad_total, pending, embedded_pending, jobs.size(), Aot::lazy ? " (lazy: 25% of WAD total)" : "");
    Aot::SetPhase(Aot::Phase::Variants);
    run("contract", jobs);

    if (caps.path != ShaderPath::Objects) {
        Aot::SetPhase(Aot::Phase::Pipelines);
        const auto pl_start = std::chrono::steady_clock::now();
        struct PlJob {
            GraphicsPipelineKey gkey;
            bool is_wad = false;
        };
        std::vector<PlJob> pl_jobs;
        std::unordered_set<u64> pl_seen;
        // 1: queued; 0: new, its programs not in the cache yet; -1: queued or counted before.
        auto push_pl = [&](u64 vs_key, u64 ps_key, u64 gs_mask, const FixedPipelineState& raw_state, bool is_wad) -> int {
            const bool has_ps = (raw_state.flags & FixedPipelineState::kHasPs) != 0;
            GraphicsPipelineKey gk{vs_key, has_ps ? ps_key : 0ull, (raw_state.flags & FixedPipelineState::kHasGs) ? gs_mask : 0ull, raw_state};
            gk.state.Canonicalize(caps);
            if (!pl_seen.insert(gk.Hash()).second) return -1;
            if (!vs_key || (vs_key != kClearRectVsKey && !cache.Contains(vs_key)) || (has_ps && (!ps_key || !cache.Contains(ps_key)))) return 0;
            pl_jobs.push_back({gk, is_wad});
            return 1;
        };
        for (const GraphicsPipelineKey& r : Recipes::Instance().Pipelines(no_srnf, blend_interlock)) push_pl(r.vs_key, r.ps_key, r.gs_mask, r.state, false);
        // The derived pipelines as this launch builds them: one gated on a PS program waits until the recipes meet its code; the
        // scene's sample count is this launch's.
        size_t gated = 0;
        const auto push_derived = [&](const ContractedPipeline& cp, bool is_wad) {
            if (cp.gate && !met_ps.contains(cp.gate)) return ++gated, -1;
            FixedPipelineState st = cp.state;
            if (!st.samples) st.samples = u8(1u << Aot::sample_log2);
            return push_pl(cp.vs_key, cp.ps_key, cp.gs_mask, st, is_wad);
        };
        for (const ContractedPipeline& cp : kept_pipelines)
            if (cp.file == ~0u) push_derived(cp, false);
        // Lazy AOT compiles a quarter of the WAD pipelines the gates let through (with their programs built or not yet) per
        // launch, the likeliest first: the VS variant and the PS program the recipes met, then a met VS with another PS, then
        // the rest; parse order within each.
        const auto group = [&](const ContractedPipeline& cp) {
            if (cp.vs_key != kClearRectVsKey && !met_keys.contains(cp.vs_key)) return 2;
            return !cp.ps_key || met_keys.contains(cp.ps_key) ? 0 : 1;
        };
        size_t by_group[3]{};
        for (int g = 0; g < 3; ++g)
            for (const ContractedPipeline& cp : kept_pipelines)
                if (cp.file != ~0u && group(cp) == g) by_group[g] += push_derived(cp, true) >= 0;
        const size_t wad_pl_total = by_group[0] + by_group[1] + by_group[2];
        const size_t wad_pl_budget = Aot::lazy ? (wad_pl_total + 3) / 4 : SIZE_MAX;
        Say("AOT: pipelines: %zu WAD derived (%zu of programs the recipes met, %zu of a met VS, %zu other), %zu wait for their PS in "
            "the recipes; %zu queued now (programs built)%s\n", wad_pl_total, by_group[0], by_group[1], by_group[2], gated,
            size_t(std::ranges::count(pl_jobs, true, &PlJob::is_wad)), Aot::lazy ? ", lazy: up to 25% of the WAD derived compiled" : "");

        {
            std::unordered_set<u64> cs_seen;
            std::vector<u64> cs_filtered;
            for (u64 k : compute_keys)
                if (cache.Contains(k) && cs_seen.insert(k).second) cs_filtered.push_back(k);
            compute_keys = std::move(cs_filtered);
        }
        for (size_t i = 0; i < compute_keys.size() + pl_jobs.size(); ++i) cache.PipelineQueued();

        struct AotCleanup {
            VkDevice dev;
            std::unordered_map<u64, VkPipeline> prerast, frag, foi;
            std::unordered_map<u64, VkShaderModule> mods;
            ~AotCleanup() {
                for (auto& [k, p] : prerast) if (p) vkDestroyPipeline(dev, p, nullptr);
                for (auto& [k, p] : frag) if (p) vkDestroyPipeline(dev, p, nullptr);
                for (auto& [k, p] : foi) if (p) vkDestroyPipeline(dev, p, nullptr);
                for (auto& [k, m] : mods) if (m) vkDestroyShaderModule(dev, m, nullptr);
            }
        } temp{dev};
        std::mutex aot_lock;
        std::once_flag aot_begin_once;
        std::atomic<u32> cs_cached{0}, cs_compiled{0}, cs_failed{0};
        std::atomic<u32> pl_cached{0}, pl_compiled{0}, pl_skipped{0}, pl_failed{0};
        std::atomic<size_t> wad_compiled{0};

        // A compute pipeline the renderer already holds (made ready for an open WAD) is not asked of the driver again: the Intel
        // UHD 620 driver (101.2145) crashes looking up a pipeline while one made from the same cache entry is alive.
        std::unordered_set<u64> held;
        {
            std::shared_lock lock(prog_lock);
            for (const auto& [k, ref] : shaders)
                if (ref.pl) held.insert(cache.ContentOf(k));
        }
        held.erase(0);
        parallel(compute_keys.size(), [&](size_t i) {
            const u64 key = compute_keys[i];
            if (held.contains(cache.ContentOf(key))) {
                ++cs_cached;
            } else if (const auto hit = cache.Find(key); hit && !hit->spirv.empty()) {
                VkPipeline pl = caps.cache_control ? CreateComputePipeline(hit->spirv, true) : VK_NULL_HANDLE;
                if (pl) {
                    vkDestroyPipeline(dev, pl, nullptr);
                    ++cs_cached;
                } else {
                    std::call_once(aot_begin_once, Gow3::BeginAotBuild);
                    pl = CreateComputePipeline(hit->spirv, false);
                    if (pl) {
                        vkDestroyPipeline(dev, pl, nullptr);
                        ++cs_compiled;
                    } else {
                        ++cs_failed;
                    }
                }
            } else {
                ++cs_failed;
            }
            cache.PipelineCompleted();
        });

        auto resolve_mod = [&](u64 key, u64& out_content) -> VkShaderModule {
            if (key == kClearRectVsKey) return out_content = kClearRectVsKey, clear_rect_ref.mod;
            out_content = cache.ContentOf(key);
            if (!out_content) return VK_NULL_HANDLE;
            {
                std::scoped_lock l(aot_lock);
                if (auto it = temp.mods.find(out_content); it != temp.mods.end()) return it->second;
            }
            const auto hit = cache.Find(key);
            if (!hit || hit->spirv.empty()) return VK_NULL_HANDLE;
            const VkShaderModule mod = CreateShaderModule(hit->spirv);
            if (!mod) return VK_NULL_HANDLE;
            std::scoped_lock l(aot_lock);
            auto [it, inserted] = temp.mods.try_emplace(out_content, mod);
            if (!inserted) vkDestroyShaderModule(dev, mod, nullptr);
            return it->second;
        };

        auto aot_gpl_prerast = [&](u64 vs_content, VkShaderModule vs_mod, VkShaderModule gs_mod, u64 gs_mask, const FixedPipelineState& st) -> VkPipeline {
            const bool neg_one = !caps.depth_clip_negative_one_to_one && (st.flags & FixedPipelineState::kNegOne);
            const bool depth_clamp = !caps.depth_clamp_enable && (st.flags & FixedPipelineState::kDepthClamp);
            const bool has_gs = (st.flags & FixedPipelineState::kHasGs) != 0;
            struct PreKey {
                u64 vs, gs;
                u32 flags, pad;
            } pk{vs_content, has_gs ? gs_mask : 0ull, u32(neg_one) | u32(depth_clamp) << 1 | u32(has_gs) << 2, 0};
            const u64 h = PieceHash::Stable64(&pk, sizeof(pk));
            {
                std::scoped_lock l(aot_lock);
                if (auto it = temp.prerast.find(h); it != temp.prerast.end()) return it->second;
            }
            const VkPipeline created = CreateGplPreRaster(vs_mod, has_gs ? gs_mod : VK_NULL_HANDLE, neg_one, depth_clamp);
            if (!created) return VK_NULL_HANDLE;
            std::scoped_lock l(aot_lock);
            auto [it, inserted] = temp.prerast.try_emplace(h, created);
            if (!inserted) vkDestroyPipeline(dev, created, nullptr);
            return it->second;
        };

        auto aot_gpl_frag = [&](u64 ps_content, VkShaderModule ps_mod, const FixedPipelineState& st) -> VkPipeline {
            if (!(st.flags & FixedPipelineState::kHasPs) || !ps_mod) return VK_NULL_HANDLE;
            const u32 samples = caps.rasterization_samples ? 1u : std::max<u32>(1, st.samples);
            const bool a2c = !caps.alpha_to_coverage_enable && (st.flags & FixedPipelineState::kAlphaToCoverage) != 0;
            struct FragKey {
                u64 ps;
                u32 samples, a2c;
            } fk{ps_content, samples, a2c ? 1u : 0u};
            const u64 h = PieceHash::Stable64(&fk, sizeof(fk));
            {
                std::scoped_lock l(aot_lock);
                if (auto it = temp.frag.find(h); it != temp.frag.end()) return it->second;
            }
            const VkPipeline created = CreateGplFragShader(ps_mod, samples, a2c);
            if (!created) return VK_NULL_HANDLE;
            std::scoped_lock l(aot_lock);
            auto [it, inserted] = temp.frag.try_emplace(h, created);
            if (!inserted) vkDestroyPipeline(dev, created, nullptr);
            return it->second;
        };

        auto aot_gpl_foi = [&](const FixedPipelineState& st, bool square_pass) -> VkPipeline {
            struct FoiKey {
                VkFormat color_formats[8];
                VkFormat depth_format;
                VkFormat stencil_format;
                u8 color_count;
                u8 samples;
                u8 a2c;
                u8 square_pass;
                u8 square_color_mask;
                u8 square_alpha_mask;
                u8 pad[2];
                FixedAttachmentBlend blend[8];
            } fk{};
            static_assert(sizeof(FoiKey) == 112);
            std::copy_n(st.color_formats, 8, fk.color_formats);
            fk.depth_format = st.depth_format;
            fk.stencil_format = st.stencil_format;
            fk.color_count = st.color_count;
            fk.samples = caps.rasterization_samples ? 1 : std::max<u8>(1, st.samples);
            fk.a2c = (!caps.alpha_to_coverage_enable && (st.flags & FixedPipelineState::kAlphaToCoverage)) ? 1 : 0;
            fk.square_pass = square_pass ? 1 : 0;
            fk.square_color_mask = square_pass ? st.square_color_mask : 0;
            fk.square_alpha_mask = square_pass ? st.square_alpha_mask : 0;
            std::copy_n(st.blend, 8, fk.blend);
            const u64 h = PieceHash::Stable64(&fk, sizeof(fk));
            {
                std::scoped_lock l(aot_lock);
                if (auto it = temp.foi.find(h); it != temp.foi.end()) return it->second;
            }
            const VkPipeline created = CreateGplFragOutput(st, square_pass);
            if (!created) return VK_NULL_HANDLE;
            std::scoped_lock l(aot_lock);
            auto [it, inserted] = temp.foi.try_emplace(h, created);
            if (!inserted) vkDestroyPipeline(dev, created, nullptr);
            return it->second;
        };

        parallel(pl_jobs.size(), [&](size_t i) {
            const PlJob& job = pl_jobs[i];
            const GraphicsPipelineKey& gk = job.gkey;
            u64 vs_content = 0, ps_content = 0;
            const VkShaderModule vs_mod = resolve_mod(gk.vs_key, vs_content);
            if (!vs_mod) {
                ++pl_failed;
                cache.PipelineCompleted();
                return;
            }
            const bool has_ps = (gk.state.flags & FixedPipelineState::kHasPs) != 0;
            const VkShaderModule ps_mod = has_ps ? resolve_mod(gk.ps_key, ps_content) : VK_NULL_HANDLE;
            if (has_ps && !ps_mod) {
                ++pl_failed;
                cache.PipelineCompleted();
                return;
            }
            const bool has_gs = (gk.state.flags & FixedPipelineState::kHasGs) != 0;
            const ShaderRef gs_ref = has_gs ? RectGs(gk.gs_mask) : ShaderRef{};
            if (has_gs && !gs_ref.mod) {
                ++pl_failed;
                cache.PipelineCompleted();
                return;
            }
            if (caps.path == ShaderPath::Gpl && has_ps) {
                const VkPipeline vi = gpl_vi[std::min<u8>(gk.state.topology_class, 2)];
                const VkPipeline prerast = aot_gpl_prerast(vs_content, vs_mod, gs_ref.mod, gk.gs_mask, gk.state);
                const VkPipeline fs = aot_gpl_frag(ps_content, ps_mod, gk.state);
                const VkPipeline foi = aot_gpl_foi(gk.state, false);
                const bool need_sq = gk.state.NeedsSquarePipeline(caps);
                const VkPipeline foi_sq = need_sq ? aot_gpl_foi(gk.state, true) : VK_NULL_HANDLE;
                if (!vi || !prerast || !fs || !foi || (need_sq && !foi_sq)) {
                    ++pl_failed;
                    cache.PipelineCompleted();
                    return;
                }
                const VkPipeline libs[4]{vi, prerast, fs, foi};
                if (const CompiledPipeline hit = LinkGplPair(libs, foi_sq, true, true); hit.main) {
                    vkDestroyPipeline(dev, hit.main, nullptr);
                    if (hit.square) vkDestroyPipeline(dev, hit.square, nullptr);
                    ++pl_cached;
                    cache.PipelineCompleted();
                    return;
                }
                if (job.is_wad && wad_compiled.fetch_add(1, std::memory_order_relaxed) >= wad_pl_budget) {
                    ++pl_skipped;
                    cache.PipelineCompleted();
                    return;
                }
                std::call_once(aot_begin_once, Gow3::BeginAotBuild);
                if (const CompiledPipeline built = LinkGplPair(libs, foi_sq, true, false); built.main) {
                    vkDestroyPipeline(dev, built.main, nullptr);
                    if (built.square) vkDestroyPipeline(dev, built.square, nullptr);
                    ++pl_compiled;
                } else {
                    ++pl_failed;
                }
                cache.PipelineCompleted();
                return;
            }
            const ShaderRef vs_ref{gk.vs_key, VK_NULL_HANDLE, vs_mod, VK_NULL_HANDLE};
            const ShaderRef ps_ref{gk.ps_key, VK_NULL_HANDLE, ps_mod, VK_NULL_HANDLE};
            if (const CompiledPipeline hit = CreateMonolithicPipeline(vs_ref, ps_ref, gs_ref, gk.state, true); hit.main) {
                vkDestroyPipeline(dev, hit.main, nullptr);
                if (hit.square) vkDestroyPipeline(dev, hit.square, nullptr);
                ++pl_cached;
                cache.PipelineCompleted();
                return;
            }
            if (job.is_wad && wad_compiled.fetch_add(1, std::memory_order_relaxed) >= wad_pl_budget) {
                ++pl_skipped;
                cache.PipelineCompleted();
                return;
            }
            std::call_once(aot_begin_once, Gow3::BeginAotBuild);
            if (const CompiledPipeline built = CreateMonolithicPipeline(vs_ref, ps_ref, gs_ref, gk.state, false); built.main) {
                vkDestroyPipeline(dev, built.main, nullptr);
                if (built.square) vkDestroyPipeline(dev, built.square, nullptr);
                ++pl_compiled;
            } else {
                ++pl_failed;
            }
            cache.PipelineCompleted();
        });

        Say("AOT: pipelines: %zu graphics (%u cached, %u compiled, %u skipped, %u failed), %zu compute (%u cached, %u compiled, %u failed) (%.1f s)\n",
            pl_jobs.size(), pl_cached.load(), pl_compiled.load(), pl_skipped.load(), pl_failed.load(),
            compute_keys.size(), cs_cached.load(), cs_compiled.load(), cs_failed.load(),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - pl_start).count());
    }

    Aot::SetPhase(Aot::Phase::Finalizing);
    for (u32 f = 0; f < index.files.size(); ++f)
        if (const std::string name = f ? WadName(index.files[f].c_str()) : "EBOOT"; !name.empty() && !wad_keys[f].empty())
            cache.SetWadKeys(name, wad_keys[f], wad_binaries[f]);
    give_hot();  // again, with the binaries this build made
    if (caps.path != ShaderPath::Objects) SavePipelineCache();
    cache.Finish();
    // The game waited for the build, so the WADs it opened meanwhile warmed their shaders but not their pipelines (the WAD
    // thread warms those only from here on): made now, before the first frames draw with them.
    if (caps.path != ShaderPath::Objects) {
        const auto warm_start = std::chrono::steady_clock::now();
        const auto made = [&] { std::shared_lock lock(prog_lock); return pipelines.size(); };
        const size_t before = made();
        WarmPipelines();
        Say("AOT: %zu pipelines made for the first frames (%.1f s)\n", made() - before,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_start).count());
    }
    Say("AOT: done (%.1f s)\n", seconds());
}
} // namespace Gpu
