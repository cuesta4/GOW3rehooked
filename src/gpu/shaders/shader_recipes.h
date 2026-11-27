// SPDX-License-Identifier: GPL-2.0-or-later
// Recipes: the exact shader variants the game met while played, kept as the fields their translation key hashes (not the
// key, so they outlive key and cache format changes). They come from the player's runs and from the shaders the JIT had to
// translate during play. The AOT translates them first; the recipes file only grows and holds no game data, the code
// is found by its hash in the user's WADs. The same file keeps the pipelines the draws used (pipeline paths), by the Id of
// their programs' recipes: a release ships both in one file, which earlier builds read without the pipelines.
#pragma once
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "gpu/shaders/gcn_translator.h"
#include "gpu/shaders/shader_runtime.h"

namespace Gpu {
struct RecipeHead {
    std::uint64_t code_hash, fetch_hash;  // hashes of the program and of the VS fetch shader (0: none)
    std::uint32_t rsrc1, rsrc2, ps_input_ena, ps_input_addr, col_format, threads[3], cntl[32];
    std::uint8_t mrt_class[8], mrt_map[8];
    std::uint16_t nspec;  // slots' Pack words follow
    std::uint8_t stage, rect;
    std::uint32_t reserved = 0;
};
struct Recipe {
    RecipeHead h;
    std::vector<std::uint64_t> spec;
    std::vector<std::uint32_t> fetch;  // the VS fetch shader the engine generated at run time (not in the program's file); not in the key
    Gcn::Input Input() const;  // everything of Gcn::Input but the code
    std::vector<Gcn::SlotSpec> Specs() const;
    std::uint64_t Key() const;  // the translation key the JIT computes for this variant
    // The variant on any GPU: its fields without the bit the runtime sets for the GPU in use (no storage reads without a format).
    std::uint64_t Id() const;
    void Adapt(bool no_storage_read_nofmt, bool no_float_controls = false) {
        h.reserved = (h.reserved & ~((1u << 16) | (1u << 17))) | (no_storage_read_nofmt ? 1u << 16 : 0u) | (no_float_controls ? 1u << 17 : 0u);
    }
    // The recipe of `in` (its code and fetch hashes given) with `spec`.
    static Recipe From(const Gcn::Input& in, std::uint64_t code_hash, std::uint64_t fetch_hash, std::span<const Gcn::SlotSpec> spec);
};

// A pipeline a draw used, for any GPU and sample count: its programs by their recipes' Id, its raw state (un-canonicalized:
// any set of dynamic states applies) with the scene's sample count as 0.
struct PipelineRecipe {
    std::uint64_t vs = 0, ps = 0;    // Recipe::Id (vs kClearRectVsKey: the built-in clear VS; ps 0: none)
    std::uint64_t gs_mask = 0;
    FixedPipelineState raw_state{};
    std::uint8_t scene_log2 = 0;     // the scene's sample count (log2) the draw was met with
    std::uint8_t pad[7]{};
};
static_assert(std::is_trivially_copyable_v<PipelineRecipe>);
static_assert(sizeof(PipelineRecipe) == 144);

class Recipes {
public:
    static Recipes& Instance();
    // Loads recipes from `dir` (created if missing); until then nothing is recorded. `sample_log2`: the scene's sample count.
    void Configure(const std::filesystem::path& dir, unsigned sample_log2);
    // A draw needs the translation `key` of `in` (code and fetch set) with `spec`: kept once, written at the next flush. `cached`: the
    // shader cache already holds it (a new recipe that is cached was built by the AOT's contract: its hit rate).
    void Note(std::uint64_t key, const Gcn::Input& in, std::span<const Gcn::SlotSpec> spec, bool cached);
    // A draw used the pipeline of these programs (translation keys of recipes noted before) with this raw state: kept once.
    void NotePipeline(std::uint64_t vs_key, std::uint64_t ps_key, std::uint64_t gs_mask, const FixedPipelineState& raw_state);
    void Report() const;  // run summary on stderr: new recipes and how many of them the AOT had already built
    void Flush();         // writes the recipes noted since the last flush (the ShaderCache's flush thread and the exit call it)
    std::vector<Recipe> All() const;
    // The pipeline recipes as this GPU and sample count draw them (translation keys, raw states). Left out: those whose
    // programs hold to another sample count they were met with (multisampled images, the interlock's blend) or need an interlock.
    std::vector<GraphicsPipelineKey> Pipelines(bool no_storage_read_nofmt, bool interlock, bool no_float_controls = false) const;

private:
    void Append(std::uint32_t type, const void* a, std::size_t na, const void* b = nullptr, std::size_t nb = 0);
    bool AddPipeline(std::uint64_t vs_key, std::uint64_t ps_key, std::uint64_t gs_mask, const FixedPipelineState& raw_state);  // under `lock`
    bool enabled = false;
    unsigned sample_log2 = 0;
    std::uint32_t fresh = 0, fresh_cached = 0, fresh_pipelines = 0;
    mutable std::mutex lock;
    FILE* file = nullptr;
    std::vector<std::uint8_t> staged;  // records not written yet
    std::vector<Recipe> recipes;
    std::unordered_set<std::uint64_t> keys;
    std::unordered_map<std::uint64_t, std::uint64_t> ids;  // translation key -> Recipe::Id
    std::vector<PipelineRecipe> pipelines;
    std::unordered_set<std::uint64_t> pipeline_hashes;
};
} // namespace Gpu
