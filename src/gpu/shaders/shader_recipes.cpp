// SPDX-License-Identifier: GPL-2.0-or-later
#include "gpu/piece_hash.h"
#include "gpu/shaders/shader_recipes.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <xxhash.h>
#include "gpu/shaders/shader_cache.h"
#include "gpu/shaders/shader_runtime.h"

namespace Gpu {
namespace {
constexpr std::uint32_t kMagic = 0x50523347, kFormat = 1;  // "G3RP"; kFormat: the layout of the records below
// kRecipeFetch: a recipe followed by its fetch shader words; kPipeline: a PipelineRecipe (readers skip record types they
// do not know). (2 was the WAD open order: skipped if a file still has such records.)
enum Type : std::uint32_t { kRecipe = 1, kRecipeFetch = 3, kPipeline = 4 };
struct Record {
    std::uint32_t type, size;
};
static_assert(sizeof(RecipeHead) == 200);
// Earlier builds kept the pipeline recipes in their own file, by translation key.
constexpr std::uint32_t kOldPlMagic = 0x52503347, kOldPlFormat = 1;  // "G3PR"
struct OldPipeline {
    std::uint64_t vs_key, ps_key, gs_mask;
    FixedPipelineState raw_state;
};
std::uint64_t Hash(const PipelineRecipe& p) { return PieceHash::Stable64(&p, sizeof(p)); }
std::uint64_t KeyOf(const Recipe& r, bool no_storage_read_nofmt) {
    Gcn::Input in = r.Input();
    in.no_storage_read_nofmt = no_storage_read_nofmt;
    return TranslationKey(ProgramKey(Gcn::Stage(r.h.stage), r.h.code_hash, r.h.fetch_hash, r.h.rsrc2, false, false), in, r.Specs());
}
// The variant holds to the sample count it was met with: it reads multisampled images or blends through the interlock (by that count).
bool HoldsSamples(const Recipe& r) {
    return (r.h.reserved & 0xFFFFu) || std::ranges::any_of(r.spec, [](std::uint64_t w) { return Gcn::Unpack(w).dim >= Gcn::Dim::D2Ms; });
}
} // namespace

Gcn::Input Recipe::Input() const {
    Gcn::Input in;
    in.stage = Gcn::Stage(h.stage), in.rsrc1 = h.rsrc1, in.rsrc2 = h.rsrc2, in.rect = h.rect;
    in.blend_interlock = h.reserved & 0xFFFFu;
    in.no_storage_read_nofmt = (h.reserved >> 16 & 1) != 0;
    in.ps_input_ena = h.ps_input_ena, in.ps_input_addr = h.ps_input_addr, in.col_format = h.col_format;
    std::memcpy(in.ps_input_cntl, h.cntl, sizeof(h.cntl)), std::memcpy(in.threads, h.threads, sizeof(h.threads));
    std::memcpy(in.mrt_class, h.mrt_class, 8), std::memcpy(in.mrt_map, h.mrt_map, 8);
    return in;
}

std::vector<Gcn::SlotSpec> Recipe::Specs() const {
    std::vector<Gcn::SlotSpec> s;
    s.reserve(spec.size());
    for (std::uint64_t w : spec) s.push_back(Gcn::Unpack(w));
    return s;
}

std::uint64_t Recipe::Key() const { return KeyOf(*this, (h.reserved >> 16 & 1) != 0); }

std::uint64_t Recipe::Id() const {
    RecipeHead head = h;
    head.reserved &= ~(1u << 16);
    return PieceHash::Stable64Seeded(spec.data(), spec.size() * 8, PieceHash::Stable64(&head, sizeof(head)));
}

Recipe Recipe::From(const Gcn::Input& in, std::uint64_t code_hash, std::uint64_t fetch_hash, std::span<const Gcn::SlotSpec> spec) {
    Recipe r;
    RecipeHead& h = r.h;
    h = {};
    h.code_hash = code_hash, h.fetch_hash = fetch_hash;
    h.rsrc1 = in.rsrc1, h.rsrc2 = in.rsrc2, h.ps_input_ena = in.ps_input_ena, h.ps_input_addr = in.ps_input_addr, h.col_format = in.col_format;
    std::memcpy(h.cntl, in.ps_input_cntl, sizeof(h.cntl)), std::memcpy(h.threads, in.threads, sizeof(h.threads));
    std::memcpy(h.mrt_class, in.mrt_class, 8), std::memcpy(h.mrt_map, in.mrt_map, 8);
    h.nspec = std::uint16_t(spec.size()), h.stage = std::uint8_t(in.stage), h.rect = in.rect;
    h.reserved = (in.blend_interlock & 0xFFFFu) | (in.no_storage_read_nofmt ? (1u << 16) : 0u);
    for (const Gcn::SlotSpec& s : spec) r.spec.push_back(Gcn::Pack(s));
    return r;
}

Recipes& Recipes::Instance() {
    static Recipes r;
    return r;
}

void Recipes::Configure(const std::filesystem::path& dir, unsigned scene_log2) {
    std::scoped_lock l(lock);
    sample_log2 = scene_log2;
    std::filesystem::create_directories(dir);
    const auto path = dir / (PieceHash::UsesAes() ? "recipes-hash2-aes.bin" : "recipes-hash2-xxh.bin");
    std::uintmax_t good = 8;  // end of the last whole record
    bool valid = false;
#ifdef _WIN32
    if (FILE* f = _wfopen(path.c_str(), L"rb")) {
#else
    if (FILE* f = std::fopen(path.c_str(), "rb")) {
#endif
        std::uint32_t head[2];
        valid = std::fread(head, 4, 2, f) == 2 && head[0] == kMagic && head[1] == kFormat;
        std::vector<std::uint8_t> body;
        for (Record r; valid && std::fread(&r, sizeof(r), 1, f) == 1 && r.size < (1u << 20); good += sizeof(r) + r.size) {
            body.resize(r.size);
            if (std::fread(body.data(), 1, r.size, f) != r.size) break;
            if ((r.type == kRecipe || r.type == kRecipeFetch) && r.size >= sizeof(RecipeHead)) {
                Recipe rc;
                std::memcpy(&rc.h, body.data(), sizeof(RecipeHead));
                const std::size_t spec_end = sizeof(RecipeHead) + rc.h.nspec * 8ull;
                if (r.size < spec_end || (r.type == kRecipe ? r.size != spec_end : (r.size - spec_end) % 4)) break;
                rc.spec.resize(rc.h.nspec), std::memcpy(rc.spec.data(), body.data() + sizeof(RecipeHead), rc.h.nspec * 8ull);
                rc.fetch.resize((r.size - spec_end) / 4), std::memcpy(rc.fetch.data(), body.data() + spec_end, r.size - spec_end);
                // Without the fetch shader's words a recipe of the engine's own fetch shaders cannot be built: noted again.
                if (r.type == kRecipe && rc.h.fetch_hash) continue;
                if (const std::uint64_t key = rc.Key(); keys.insert(key).second) ids.emplace(key, rc.Id()), recipes.push_back(std::move(rc));
            } else if (r.type == kPipeline && r.size >= sizeof(PipelineRecipe)) {  // a longer record: a later layout's fields follow
                PipelineRecipe p;
                std::memcpy(&p, body.data(), sizeof(p));
                if (pipeline_hashes.insert(Hash(p)).second) pipelines.push_back(p);
            }
        }
        std::fclose(f);
    }
    if (valid) {
        std::error_code ec;  // a killed run left a torn tail: cut it
        std::filesystem::resize_file(path, good, ec);
    }
#ifdef _WIN32
    file = _wfopen(path.c_str(), valid ? L"ab" : L"wb");
#else
    file = std::fopen(path.c_str(), valid ? "ab" : "wb");
#endif
    if (!file) return;
    if (!valid) {
        const std::uint32_t head[2]{kMagic, kFormat};
        std::fwrite(head, 4, 2, file);
        std::fflush(file);
    }
    enabled = true;
    // The pipeline recipes of earlier builds join this file; theirs goes once each of them found its programs' recipes.
    const auto old = dir / (PieceHash::UsesAes() ? "pipeline-recipes-hash2-aes.bin" : "pipeline-recipes-hash2-xxh.bin");
    if (std::ifstream in(old, std::ios::binary); in) {
        std::uint32_t head[2]{};
        const bool known = in.read(reinterpret_cast<char*>(head), sizeof(head)) && head[0] == kOldPlMagic && head[1] == kOldPlFormat;
        bool all = known;
        for (OldPipeline o; known && in.read(reinterpret_cast<char*>(&o), sizeof(o));) {
            all &= (o.vs_key == kClearRectVsKey || ids.contains(o.vs_key)) && (!o.ps_key || ids.contains(o.ps_key));
            AddPipeline(o.vs_key, o.ps_key, o.gs_mask, o.raw_state);
        }
        in.close();
        std::fwrite(staged.data(), 1, staged.size(), file), std::fflush(file), staged.clear();
        std::error_code ec;
        if (all) std::filesystem::remove(old, ec);
    }
}

// Staged in memory: the draw that met the variant does no I/O. Flush writes them (loading screens, exit).
void Recipes::Append(std::uint32_t type, const void* a, std::size_t na, const void* b, std::size_t nb) {
    const Record r{type, std::uint32_t(na + nb)};
    const auto bytes = [&](const void* p, std::size_t n) { staged.insert(staged.end(), static_cast<const std::uint8_t*>(p), static_cast<const std::uint8_t*>(p) + n); };
    bytes(&r, sizeof(r)), bytes(a, na);
    if (nb) bytes(b, nb);
}

void Recipes::Flush() {
    std::scoped_lock l(lock);
    if (staged.empty() || !file) return;
    std::fwrite(staged.data(), 1, staged.size(), file), std::fflush(file);
    staged.clear();
}

void Recipes::Report() const {
    std::scoped_lock l(lock);
    if (enabled)
        std::fprintf(stderr, "Recipes: %zu kept, %u new this run, %u of those already built by the AOT; %zu pipelines, %u new\n", recipes.size(), fresh,
                     fresh_cached, pipelines.size(), fresh_pipelines);
}

void Recipes::Note(std::uint64_t key, const Gcn::Input& in, std::span<const Gcn::SlotSpec> spec, bool cached) {
    if (!enabled) return;
    std::scoped_lock l(lock);
    if (!keys.insert(key).second) return;
    ++fresh, fresh_cached += cached;
    Recipe r = Recipe::From(in, ::Gpu::PieceHash::ShaderCode(in.code.data(), in.code.size_bytes()), in.fetch.empty() ? 0 : ::Gpu::PieceHash::ShaderCode(in.fetch.data(), in.fetch.size_bytes()), spec);
    RecipeHead& h = r.h;
    if (r.Key() != key) std::fprintf(stderr, "Recipe %016llx does not rebuild its key (%016llx)\n", (unsigned long long)key, (unsigned long long)r.Key());
    ids.emplace(key, r.Id());
    r.fetch.assign(in.fetch.begin(), in.fetch.end());
    std::vector<std::uint8_t> tail(r.spec.size() * 8 + r.fetch.size() * 4);
    std::memcpy(tail.data(), r.spec.data(), r.spec.size() * 8), std::memcpy(tail.data() + r.spec.size() * 8, r.fetch.data(), r.fetch.size() * 4);
    Append(r.fetch.empty() ? kRecipe : kRecipeFetch, &h, sizeof(h), tail.data(), tail.size());
    recipes.push_back(std::move(r));
    ShaderCache::Instance().Dirty();
}

std::vector<Recipe> Recipes::All() const {
    std::scoped_lock l(lock);
    return recipes;
}

void Recipes::NotePipeline(std::uint64_t vs_key, std::uint64_t ps_key, std::uint64_t gs_mask, const FixedPipelineState& raw_state) {
    {
        std::scoped_lock l(lock);
        if (!enabled || !AddPipeline(vs_key, ps_key, gs_mask, raw_state)) return;
        ++fresh_pipelines;
    }
    ShaderCache::Instance().Dirty();
}

bool Recipes::AddPipeline(std::uint64_t vs_key, std::uint64_t ps_key, std::uint64_t gs_mask, const FixedPipelineState& raw_state) {
    const auto id = [&](std::uint64_t key) { const auto it = ids.find(key); return it == ids.end() ? 0ull : it->second; };
    PipelineRecipe p{vs_key == kClearRectVsKey ? kClearRectVsKey : id(vs_key), ps_key ? id(ps_key) : 0ull,
                     (raw_state.flags & FixedPipelineState::kHasGs) ? gs_mask : 0ull, raw_state, std::uint8_t(sample_log2)};
    if (!p.vs || (ps_key && !p.ps)) return false;  // a program without a recipe (traced or exploded draws)
    FixedPipelineState& s = p.raw_state;
    s.pad[0] = s.pad[1] = 0;
    for (u32 k = s.color_count; k < 8; ++k) s.color_formats[k] = VK_FORMAT_UNDEFINED, s.blend[k] = {};
    if (sample_log2 && s.samples == 1u << sample_log2) s.samples = 0;
    if (!pipeline_hashes.insert(Hash(p)).second) return false;
    Append(kPipeline, &p, sizeof(p));
    pipelines.push_back(p);
    return true;
}

std::vector<GraphicsPipelineKey> Recipes::Pipelines(bool no_storage_read_nofmt, bool interlock) const {
    const std::vector<Recipe> variants = All();  // the keys are made without the lock: draws note recipes meanwhile
    std::vector<PipelineRecipe> kept;
    {
        std::scoped_lock l(lock);
        kept = pipelines;
    }
    struct Local {
        std::uint64_t key;
        bool holds;  // HoldsSamples
    };
    std::unordered_map<std::uint64_t, Local> local{{kClearRectVsKey, {kClearRectVsKey, false}}};  // Id -> this GPU's variant
    local.reserve(variants.size() + 1);
    for (const Recipe& r : variants)
        if (interlock || !(r.h.reserved & 0xFFFFu)) local.try_emplace(r.Id(), Local{KeyOf(r, no_storage_read_nofmt), HoldsSamples(r)});
    std::vector<GraphicsPipelineKey> out;
    out.reserve(kept.size());
    for (const PipelineRecipe& p : kept) {
        const auto vs = local.find(p.vs), ps = p.ps ? local.find(p.ps) : local.end();
        if (vs == local.end() || (p.ps && ps == local.end())) continue;
        if (p.scene_log2 != sample_log2 && (vs->second.holds || (p.ps && ps->second.holds))) continue;
        GraphicsPipelineKey& k = out.emplace_back(GraphicsPipelineKey{vs->second.key, p.ps ? ps->second.key : 0ull, p.gs_mask, p.raw_state});
        if (!k.state.samples) k.state.samples = std::uint8_t(1u << sample_log2);
    }
    return out;
}
} // namespace Gpu
