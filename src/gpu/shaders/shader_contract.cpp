// SPDX-License-Identifier: GPL-2.0-or-later
// The AOT's own universe: the draw configurations the engine can request, derived from the game's data (shader_catalog.h).
// Bindings come from the technique records, streams from the model meshes, textures from the materials, formats from the
// pass views and render resources. Each configuration is turned into the recipe (fields and slot specs) the runtime would
// produce for it, so its translation key is the one a draw computes.
#include <map>
#include <set>
#include "gpu/textures.h"
#include "gpu/renderer_impl.h"
#include "main/cpu.h"

namespace Gpu {
namespace {
using CatImage = Catalog::Image;
using Catalog::Mapping;
using Catalog::PassView;
using Catalog::Stream;
using CatRecipe = Catalog::Recipe;
using CatShader = Catalog::Shader;
constexpr u32 NullWord = 0x77200;

struct Job {
    const Mapping* file{};
    const CatShader* shader{};
    u32 source{};                        // Data::files index, ~0u: the EBOOT
    std::span<const u8> header;
    std::vector<u32> pixel_inputs;       // SPI_PS_INPUT_CNTL entries matched against the vertex exports
    std::vector<std::pair<u32, CatImage>> images;  // by texture slot
    u32 samplers{}, degamma{}, unnormalized{};
    bool vertex{}, rect{};
    u8 path{};                           // ContractPath that makes the configuration; paths: every path that made it
    u32 paths{};
    u8 mrt_map{0xE4};                    // export swizzle of the active targets: 0xE4 identity, 0xC6 BGRA
    std::array<u32, 16> words{};         // vertex buffer table: the V# word 3 of each slot
    std::vector<std::vector<std::pair<u32, CatImage>>> srts;  // per SRT root: T# by dword offset
    u32 pair_id = ~0u;
    std::vector<u32> pair_ids;
};
struct Key {
    u64 low, high;
    bool operator==(const Key&) const = default;
};
// How a material draws: translucent, then its CB_BLEND color and alpha functions (color 0xff: no blending), sources and destinations.
using MaterialClass = std::array<u8, 7>;
struct KeyHash { size_t operator()(const Key& key) const { return key.low; } };
template<class T> void Append(std::vector<u8>& out, const T& value) {
    const auto* p = reinterpret_cast<const u8*>(&value);
    out.insert(out.end(), p, p + sizeof(T));
}
template<class T> void AppendList(std::vector<u8>& out, const std::vector<T>& list) {
    Append(out, u32(list.size()));
    for (const auto& v : list) Append(out, v);
}
Key Signature(const Job& job) {
    std::vector<u8> b;
    Append(b, job.shader->identity);
    b.insert(b.end(), job.header.begin(), job.header.end());
    Append(b, job.mrt_map), AppendList(b, job.images);
    if (job.vertex) Append(b, job.words);
    for (const u32 v : {job.degamma, job.unnormalized, u32(job.rect)}) Append(b, v);
    for (const auto& root : job.srts) AppendList(b, root);
    const auto hash = ::Gpu::PieceHash::ShaderContract(b.data(), b.size());
    return {hash.low64, hash.high64};
}
const CatImage* Find(const Job& job, u32 slot) {
    const auto it = std::ranges::find(job.images, slot, &std::pair<u32, CatImage>::first);
    return it == job.images.end() ? nullptr : &it->second;
}
void Bind(Job& job, u32 slot, const CatImage& image) {
    if (slot >= 32) throw std::runtime_error("Texture slot outside the sampler table");
    const auto it = std::ranges::find(job.images, slot, &std::pair<u32, CatImage>::first);
    if (it != job.images.end()) it->second = image; else job.images.emplace_back(slot, image);
}
std::span<const u8> HeaderBytes(const Mapping& f, const CatShader& s) {
    const u8* h = f.data + s.header;
    return {h, (s.stage == 0 ? 60 : 40) + size_t(h[3]) * 4 + (s.stage == 1 ? size_t(h[36]) * 4 + size_t(h[37]) * 2 : s.stage == 0 ? size_t(h[56]) * 2 : 0)};
}
template<class T> T ReadHeader(std::span<const u8> header, size_t offset) {
    if (offset + sizeof(T) > header.size()) throw std::runtime_error("Truncated source header");
    T value;
    std::memcpy(&value, header.data() + offset, sizeof(T));
    return value;
}
// Classic SPI_PS_INPUT_CNTL matching of the pixel interpolants against the vertex exports.
std::vector<u32> PixelInputs(std::span<const u8> pixel, std::span<const u8> vertex) {
    std::map<u32, u32> exports;
    if (!vertex.empty()) {
        const size_t first = 40 + size_t(vertex[3]) * 4 + size_t(vertex[36]) * 4;
        for (u32 i = 0; i < vertex[37]; ++i) exports[vertex[first + i * 2]] = vertex[first + i * 2 + 1];
    }
    std::vector<u32> result;
    const size_t first = 60 + size_t(pixel[3]) * 4;
    for (u32 i = 0; i < pixel[56]; ++i) {
        const u32 semantic = pixel[first + i * 2] | pixel[first + i * 2 + 1] << 8;
        if (semantic & 0xe000) throw std::runtime_error("Packed/Neo pixel semantics need a separate contract");
        const auto found = exports.find(semantic & 255);
        if (found != exports.end() && found->second & 0xc0) throw std::runtime_error("Packed vertex exports need a separate contract");
        u32 offset = found != exports.end() ? found->second & 31 : 32;
        if (semantic & 0x1000) offset |= 32;
        result.push_back(offset | (semantic & 0x700));
    }
    return result;
}
// The fetch shader the engine builds at run time for a vertex program that does not carry one: from the program's vertex
// semantics {vertex buffer slot, destination VGPR, elements}, in chunks of as many V# loads as the SGPRs left after the user
// data allow (s_load_dwordx4 of each V# from the table at s[2:3], s_waitcnt, buffer_load_format_x..xyzw with an index in v0),
// ending in s_waitcnt 0 and s_setpc_b64 s[0:1]. Verified against the fetch shaders the JIT meets (recipes with their words).
std::vector<u32> GenerateFetch(std::span<const u8> header, u32 rsrc1, u32 rsrc2) {
    const u32 base = (rsrc2 >> 1 & 31) + 3 & ~3u, total = ((rsrc1 >> 6 & 15) + 1) * 8, chunk = std::max(1u, (total - std::min(base, total)) / 4);
    std::vector<u32> code;
    const u32 count = header[36], first = 40 + header[3] * 4;
    for (u32 begin = 0; begin < count; begin += chunk) {
        const u32 end = std::min(count, begin + chunk);
        for (u32 k = begin; k < end; ++k) code.push_back(0xC0800300 | (base + 4 * (k - begin)) << 15 | header[first + k * 4] * 4);
        code.push_back(0xBF8C007F);
        for (u32 k = begin; k < end; ++k)
            code.push_back(0xE0002000 | (header[first + k * 4 + 2] - 1u & 3) << 18), code.push_back(0x80000000 | (base / 4 + k - begin) << 16 | header[first + k * 4 + 1] << 8);
    }
    code.push_back(0xBF8C0000), code.push_back(0xBE802000);
    return code;
}
// The input-usage table as text (kind@register/flags), for the reasons a job cannot be resolved.
std::string Usages(const CatShader& s) {
    std::string text;
    for (const auto& u : s.usage) text += std::format(" {:x}@{}/{:x}", u[0], u[2], u[3]);
    return text;
}
Job Base(const Mapping& f, const CatShader& shader, u32 source, std::span<const u8> vertex = {}) {
    Job job{.file = &f, .shader = &shader, .source = source, .header = HeaderBytes(f, shader)};
    if (shader.stage == 0) job.pixel_inputs = PixelInputs(job.header, vertex);
    return job;
}
} // namespace

void Renderer::Impl::Contracts(const Catalog::Data& data, std::vector<Contracted>& out, std::vector<ContractedPipeline>* pipelines_out) {
    const auto begin = std::chrono::steady_clock::now();
    const u32 sample_log2 = data.sample_log2;
    CatImage fallback = Catalog::ApiFormat(0xfac00a);
    fallback.width = fallback.height = 8;
    const auto view_of = [&](u32 semantic) -> const PassView* {
        const auto it = std::ranges::find(data.views, semantic, &PassView::semantic);
        return it == data.views.end() ? nullptr : &*it;
    };

    std::vector<Job> jobs;
    std::unordered_map<Key, size_t, KeyHash> seen;  // configuration -> its place in `jobs`
    struct PairMeta {
        u32 file = ~0u;
        std::vector<MaterialClass> classes;  // empty: not a material's technique
        bool derive = true;                  // pipelines derived
        u64 code = 0, gate = 0;              // its PS program's code hash (0: none or unreadable); the gate of all its pipelines
    };
    std::vector<PairMeta> pair_metas;
    const auto add = [&](const Job& job) {
        const auto [at, fresh] = seen.emplace(Signature(job), jobs.size());
        if (fresh) jobs.push_back(job);
        jobs[at->second].paths |= 1u << job.path;
        if (job.pair_id != ~0u) {
            auto& ids = jobs[at->second].pair_ids;
            if (std::find(ids.begin(), ids.end(), job.pair_id) == ids.end()) ids.push_back(job.pair_id);
        }
    };
    // Pixel jobs vary where pass code decides at draw time: image type and the sRGB view of generic slots. Only the plain case
    // is made: the catalog's sampler flags and the BGRA scene target (0xC6, 99.8% of the pixel recipes met in play); other
    // flag sets and swizzles come from the recipes.
    // CatImage type: only MSAA-rule views (render_config.sample_log2 ? 14 : 9) switch, and together.
    // Material draws are exact: resolved masks, BGRA scene target, MSAA views only in the MSAA permutation (bit 1).
    const auto msaa_type = [&](Job& job, u32 msaa, u32 kind) {
        for (auto& [slot, image] : job.images) if (msaa >> slot & 1) { image.type = u8(kind); image.sample_log2 = u8(kind == 14 ? std::max(sample_log2, 1u) : 0); }
    };
    const auto pixel_variants = [&](const Job& job, u32 msaa, std::optional<u32> exact, u32 generic = 0) {
        if (exact) {
            // Flipbook layers (several textures on one semantic) are sampled without their degamma flag: those subsets made 84%
            // of the variants and predicted 4 recipes no other path did, so they are left to the recipes.
            Job variant = job;
            msaa_type(variant, msaa, *exact);
            variant.mrt_map = 0xC6;
            add(variant);
            return;
        }
        const auto emit = [&](Job shared) { shared.mrt_map = 0xC6, add(shared); };
        for (const u32 kind : {9u, 14u}) {
            if (kind == 14 && (!sample_log2 || !msaa)) break;
            Job shared = job;
            msaa_type(shared, msaa, kind);
            emit(shared);
        }
        // Generic slots may hold plain material textures while the MSAA views switch.
        if (const u32 views = msaa & ~generic; sample_log2 && views && generic) { Job shared = job; msaa_type(shared, views, 14); emit(shared); }
    };
    // Without vertex semantics the program builds a screen rectangle: the pass draws a RectList with clipping disabled.
    const auto vertex_variants = [&](Job job, u8 path) {
        job.path = path;
        for (const bool rect : {false, true}) {
            if (rect && job.header[36] != 0) break;
            job.rect = rect;
            add(job);
        }
    };
    // Procedural vertex declarations (kind, components) map to V# words through the engine's format table.
    const auto format_word = [&](u32 kind, u32 components) -> u32 {
        const u32 api = data.vertex_formats[(kind - 1) * 4 + components - 1];
        return api ? (api & 15) << 15 | (api >> 8 & 7) << 12 | (api >> 12 & 4095) : NullWord;
    };
    // Recipe streams the mesh does not supply stay on the engine's null V#; without a mesh every stream is bound.
    const auto vertex_job = [&](const Mapping& f, const CatRecipe& recipe, u32 source, const std::vector<Stream>* streams, u32 pair_id) {
        Job job = Base(f, *recipe.vs, source);
        job.vertex = true, job.words.fill(NullWord), job.pair_id = pair_id;
        for (const auto& binding : recipe.vertex) {
            if (streams && std::ranges::find(*streams, binding.source, &Stream::source) == streams->end()) continue;
            job.words[binding.slot] = binding.word;
        }
        return job;
    };
    // Pass views win for pass semantics; otherwise the material's texture, or nothing when no material is known.
    const auto pixel_jobs = [&](const Mapping& f, const CatRecipe& recipe, u32 source, const std::map<u32, CatImage>* material, u32 pair_id) {
        std::optional<u32> exact;
        if (material) exact = sample_log2 && ((recipe.permutation == ~u64{} ? 0 : u32(recipe.permutation)) & 2) ? 14u : 9u;
        Job job = Base(f, *recipe.ps, source, HeaderBytes(f, *recipe.vs));
        job.path = material ? kPsMaterial : kPsPass;
        job.pair_id = pair_id;
        u32 msaa{}, generic{};
        for (const auto& binding : recipe.textures) {
            const auto* view = view_of(binding.source);
            CatImage image = view ? view->image : fallback;
            if (!view && binding.source < data.semantic_types.size()) image.type = data.semantic_types[binding.source];
            if (material && !view) {
                const auto found = material->find(binding.source);
                if (found == material->end()) continue;
                image = found->second;
            }
            Bind(job, binding.slot, image);
            const u32 bit = 1u << binding.slot;
            // MSAA-rule views and unknown 2D semantics (default image or a pass target) follow the scene sample count.
            if (view ? !view->fixed_type : !material && image.type == 9) msaa |= bit;
            if (!view && !material && image.type == 9) generic |= bit;
            job.samplers |= bit;
            job.degamma |= image.degamma ? bit : 0, job.unnormalized |= image.unnormalized ? bit : 0;
        }
        // Only job-flag material permutations (bit 2: 0x4/0x6) ever sample a conditional pass view (FogTextureSampler) off.
        std::map<std::string, u32> conditions;
        if (material && recipe.permutation != ~u64{} && (recipe.permutation & 4))
            for (const auto& binding : recipe.textures)
                if (const auto* view = view_of(binding.source); view && !view->condition.empty() && Find(job, binding.slot)) conditions[view->condition] |= 1u << binding.slot;
        std::vector<u32> groups;
        groups.reserve(conditions.size());
        for (const auto& [_, group] : conditions) groups.push_back(group);
        for (u32 off = 0; off < 1u << groups.size(); ++off) {
            Job variant = job;
            u32 on = msaa;
            for (size_t g = 0; g < groups.size(); ++g) if (off >> g & 1) for (u32 bits = groups[g]; bits; bits &= bits - 1) {
                const u32 slot = u32(std::countr_zero(bits)), bit = 1u << slot;
                Bind(variant, slot, fallback), on &= ~bit, variant.degamma &= ~bit, variant.unnormalized &= ~bit;
            }
            pixel_variants(variant, on, exact, on & generic);
        }
    };
    // Pass-supplied shaders (velocity, shadow, ...) draw the model meshes with their own recipe: every distinct mesh stream set applies.
    std::set<std::vector<Stream>> layouts;
    for (const auto& file : data.files) for (const auto& geometry : file.geometry) for (const auto& mesh : geometry) layouts.insert(mesh);
    // The key sees which recipe streams a mesh supplies, not their strides: meshes with the same presence mask share a job.
    const auto presence = [](const CatRecipe& recipe, const std::vector<Stream>& streams) {
        u32 mask = 0, bit = 0;
        for (const auto& binding : recipe.vertex) mask |= u32(std::ranges::contains(streams, binding.source, &Stream::source)) << bit++;
        return mask;
    };
    size_t failed_setup{};
    std::map<std::string, u32> setup_reasons;
    const auto setup_failed = [&](const std::exception& error) { ++failed_setup, ++setup_reasons[std::string(error.what()).substr(0, 80)]; };
    for (u32 index_file = 0; index_file < data.files.size(); ++index_file) {
        const auto& file = data.files[index_file];
        if (!file.map) continue;
        const Mapping& f = *file.map;
        const auto& recipes = file.recipes;
        std::unordered_map<u32, std::vector<size_t>> families;
        std::unordered_map<u32, u8> family_perms;  // bit 0: has perm 0, bit 1: has perm 2
        for (size_t i = 0; i < recipes.size(); ++i) {
            families[recipes[i].shader_key].push_back(i);
            if (recipes[i].permutation == 0) family_perms[recipes[i].shader_key] |= 1;
            else if (recipes[i].permutation == 2) family_perms[recipes[i].shader_key] |= 2;
        }
        // Material draws select bit 1 (dynamic shadow) and bit 2 (render job flag): odd permutations are never chosen, and
        // the base fallback (~0) is unused once both 0 and 2 exist in the family.
        const auto skip_material_perm = [&](const CatRecipe& r) {
            const u64 p = r.permutation;
            return p == ~u64{} ? (family_perms[r.shader_key] & 3) == 3 : (p & 1) != 0;
        };
        std::unordered_set<u32> material_keys;
        for (const auto& material : file.materials) if (!material.pass_supplies_shader) for (const auto& layer : material.layers) material_keys.insert(layer.shader_key);
        const u32 pair_base = u32(pair_metas.size());
        if (pipelines_out) {
            // The distinct render classes of the materials drawing with each technique: translucent, then the blend (0xff: none).
            std::unordered_map<u32, std::vector<MaterialClass>> classes;
            for (const auto& material : file.materials) {
                if (material.pass_supplies_shader) continue;
                MaterialClass c{u8(material.Translucent()), 0xff};
                if (material.blend[0] != 0xff) std::ranges::copy(material.blend, c.begin() + 1);
                for (const auto& layer : material.layers)
                    if (auto& list = classes[layer.shader_key]; std::ranges::find(list, c) == list.end()) list.push_back(c);
            }
            // Shadow receivers and fallbacks of families with permutation 0 follow the scene at run time: their pipelines
            // are gated on the PS program, built once the recipes met its code.
            pair_metas.reserve(pair_base + recipes.size());
            for (const auto& r : recipes) {
                PairMeta& m = pair_metas.emplace_back(PairMeta{index_file, classes[r.shader_key]});
                if (m.classes.empty()) continue;
                if (r.ps && r.ps->code <= r.ps->footer && r.ps->footer <= f.size)
                    m.code = ::Gpu::PieceHash::ShaderCode(f.data + r.ps->code, r.ps->footer - r.ps->code);
                const u64 p = r.permutation;
                const bool follows_scene = p == ~u64{} ? (family_perms[r.shader_key] & 1) != 0 : (p & 2) != 0;
                m.gate = follows_scene ? m.code : 0;
                m.derive = !skip_material_perm(r) && !(follows_scene && r.ps && !m.code);  // an unreadable program is never met
            }
        }
        // Model meshes: the mesh streams bind for every layer recipe of the material.
        std::set<size_t> reached;
        std::set<std::pair<size_t, u64>> meshes_seen;
        for (const auto& edge : file.mesh_materials) try {
            const auto& streams = file.geometry.at(edge.geometry).at(edge.mesh);
            const auto& material = file.materials.at(edge.material);
            for (const auto& layer : material.layers) for (const size_t index : families[layer.shader_key]) {
                const auto& recipe = recipes[index];
                if (!recipe.vs || (!material.pass_supplies_shader && skip_material_perm(recipe))) continue;
                reached.insert(index);
                if (!meshes_seen.emplace(index, presence(recipe, streams) | u64(material.flags & 0x40) << 32).second) continue;
                const auto job = vertex_job(f, recipe, index_file, &streams, pipelines_out ? pair_base + u32(index) : ~0u);
                vertex_variants(job, kVsMesh);
                // Bit-0x40 materials render as TextFX glyphs with a fixed declaration.
                if (material.flags & 0x40) {
                    Job glyph = job;
                    glyph.words.fill(NullWord);
                    for (const auto [slot, count] : {std::pair{0u, 2u}, {1u, 3u}, {2u, 2u}, {3u, 2u}, {12u, 3u}}) glyph.words[slot] = format_word(2, count);
                    vertex_variants(glyph, kVsGlyph);
                }
            }
        } catch (const std::exception& error) { setup_failed(error); }
        // Material layers: each texture semantic samples the material's texture with its sRGB flag.
        for (const auto& material : file.materials) {
            if (material.pass_supplies_shader) continue;
            try {
                for (const auto& layer : material.layers) {
                    std::map<u32, CatImage> images;
                    for (const auto& [semantic, name] : layer.textures) {
                        const auto found = file.textures.find(name);
                        if (found == file.textures.end()) throw std::runtime_error("Material texture source unresolved: " + name);
                        CatImage image = found->second;
                        image.unnormalized = false;
                        // Bit-0x40 materials sample every texture degamma'd.
                        if (material.flags & 0x40) image.degamma = true;
                        images[semantic] = image;
                    }
                    for (const size_t index : families[layer.shader_key])
                        if (const auto& recipe = recipes[index]; recipe.ps && recipe.vs && !skip_material_perm(recipe))
                            pixel_jobs(f, recipe, index_file, &images, pipelines_out ? pair_base + u32(index) : ~0u);
                }
            } catch (const std::exception& error) { setup_failed(error); }
        }
        for (size_t index = 0; index < recipes.size(); ++index) try {
            const auto& recipe = recipes[index];
            const u32 pid = pipelines_out ? pair_base + u32(index) : ~0u;
            const bool pass_only = !material_keys.contains(recipe.shader_key);
            // Procedural recipes (pass code binds the streams): the recipe's own declaration, and each stream set the meshes
            // supply (pass-only techniques only: material VS not bound to a mesh in this WAD use their own declaration).
            if (recipe.vs && !reached.contains(index) && (pass_only || !skip_material_perm(recipe))) {
                const auto job = vertex_job(f, recipe, index_file, nullptr, pid);
                vertex_variants(job, kVsProcedural);
                if (pass_only) {
                    std::set<u32> masks;
                    for (const auto& streams : layouts)
                        if (masks.insert(presence(recipe, streams)).second) vertex_variants(vertex_job(f, recipe, index_file, &streams, pid), kVsLayout);
                }
            }
            if (recipe.ps && recipe.vs && pass_only) pixel_jobs(f, recipe, index_file, nullptr, pid);
        } catch (const std::exception& error) { setup_failed(error); }
    }
    const Mapping& eboot = *data.eboot;
    std::map<std::pair<u32, u32>, std::map<u32, const Catalog::EbootShader*>> groups;
    for (const auto& shader : data.embedded) {
        if (shader.stage == 0 || shader.stage == 1) groups[{shader.table, shader.group}][shader.stage] = &shader;
        else if (shader.stage == 4) try {
            // Compute images are bound by pass code: cover the default image as unorm and as uint (integer surfaces).
            Job job = Base(eboot, shader, ~0u);
            // SRT passes: fog passes {constants, fog volume T#}; group 0 writes that volume, group 1 reads it. The
            // spatial-lighting pass writes the cube faces through a 2D-array T# at dword 20. Views 4/5 are the fog and
            // spatial-lighting pass views.
            if (shader.table == 0x3c3508 && shader.group <= 1) job.srts = {{}, {{0, data.views.at(4).image}}};
            if (shader.table == 0x3c3b14 && shader.group == 0) {
                CatImage faces = data.views.at(5).image;
                faces.type = 13, faces.depth = 6;
                job.srts = {{{20, faces}}};
            }
            for (const u8 number : {u8(0), u8(4)}) {
                for (const auto& usage : shader.usage)
                    if ((usage[0] == 0 || usage[0] == 4) && usage[3] & 2) { CatImage image = fallback; image.num_format = number; Bind(job, usage[1], image); }
                job.path = kCsEmbedded;
                add(job);
            }
        } catch (const std::exception& error) { setup_failed(error); }
    }
    for (const auto& [_, pair] : groups) try {
        const u32 pid = pipelines_out ? u32(pair_metas.size()) : ~0u;
        if (pipelines_out) pair_metas.push_back({~0u, {}});
        const auto vs = pair.find(1), ps = pair.find(0);
        if (vs != pair.end()) {
            Job job = Base(eboot, *vs->second, ~0u);
            job.vertex = true, job.words.fill(NullWord), job.pair_id = pid;
            vertex_variants(job, kVsEmbedded);
        }
        if (ps != pair.end() && vs != pair.end()) {
            // Pass code binds the samplers the header declares; their slots need not match the image slots.
            Job job = Base(eboot, *ps->second, ~0u, HeaderBytes(eboot, *vs->second));
            job.path = kPsEmbedded, job.pair_id = pid;
            for (const auto& usage : ps->second->usage) if (usage[0] == 1 && usage[1] < 32) job.samplers |= 1u << usage[1];
            pixel_variants(job, 0, std::nullopt);
        }
    } catch (const std::exception& error) { setup_failed(error); }
    const auto prepared = std::chrono::steady_clock::now();
    std::fprintf(stderr, "AOT: contract: %zu configurations from %.1f s of setup (%zu setup failures)\n", jobs.size(),
                 std::chrono::duration<double>(prepared - begin).count(), failed_setup);
    for (const auto& [reason, count] : setup_reasons) std::fprintf(stderr, "AOT:   setup: %u %s\n", count, reason.c_str());

    // Each configuration as the recipe the runtime would produce: the program's slots (found by analysing its code) get the
    // descriptors the configuration implies, and the runtime's own SpecifySlots turns them into specs and the key.
    // Jobs of one program are adjacent, so a worker analyses a program once per chunk.
    std::ranges::sort(jobs, [](const Job& a, const Job& b) { return std::less<const void*>{}(a.shader, b.shader); });
    struct PairEntry {
        u32 pair_id;
        u64 key;
        u32 col_format;
        u32 paths;
        u8 stage;
        bool rect;
    };
    struct Mine {
        std::vector<Contracted> list;
        std::vector<PairEntry> pairs;
        std::map<std::string, u32> why;
    };
    const u32 threads = Gow3::Cpu::plan.shader_threads;
    std::vector<Mine> mine(threads);
    std::atomic<size_t> next = 0;
    {
        std::vector<std::jthread> workers;
        workers.reserve(threads);
        for (u32 t = 0; t < threads; ++t)
            workers.emplace_back([&, t] {
                struct Program {
                    const CatShader* shader = nullptr;
                    std::vector<u32> code, fetch;
                    u64 code_hash = 0, fetch_hash = 0;
                    Gcn::Analysis an;
                    bool generated = false;  // the fetch shader is the engine's run-time one (generated), not the file's
                    std::string error;  // why the program cannot be used: every job of it fails alike
                } cur;
                constexpr size_t kChunk = 64;
                for (size_t first; (first = next.fetch_add(kChunk)) < jobs.size();)
                    for (size_t i = first; i < std::min(first + kChunk, jobs.size()); ++i) try {
                        const Job& job = jobs[i];
                        const Mapping& f = *job.file;
                        const CatShader& s = *job.shader;
                        const std::span<const u8> header = job.header;
                        const Gcn::Stage stage = s.stage == 1 ? Gcn::Stage::Vertex : s.stage == 0 ? Gcn::Stage::Pixel : Gcn::Stage::Compute;
                        const u32 rsrc1 = ReadHeader<u32>(header, 16), rsrc2 = ReadHeader<u32>(header, 20);
                        if (cur.shader != &s) try {
                            cur = {};
                            cur.shader = &s;
                            const size_t bytes = s.footer - s.code;
                            f.Range(s.code, bytes, f.size);
                            cur.code.resize(bytes / 4);
                            std::memcpy(cur.code.data(), f.data + s.code, bytes);
                            if (cur.code.empty() || cur.code[0] != 0xBEEB03FF) throw std::runtime_error("Program without its header word");
                            if (stage == Gcn::Stage::Vertex && FetchSgpr(cur.code) >= 0) {  // the fetch shader follows the footer, up to its s_setpc_b64
                                const size_t at = s.footer + 28;
                                for (u32 k = 0; k < 0x100 && at + (k + 1) * 4ull <= f.size; ++k)
                                    if ((f.Read<u32>(at + k * 4) & 0xFF80FF00) == 0xBE802000) {
                                        cur.fetch.resize(k + 1);
                                        std::memcpy(cur.fetch.data(), f.data + at, (k + 1) * 4ull);
                                        break;
                                    }
                                if (cur.fetch.empty() && header[36]) cur.fetch = GenerateFetch(header, rsrc1, rsrc2), cur.generated = true;
                                if (cur.fetch.empty()) throw std::runtime_error("Vertex program without a fetch shader or vertex semantics");
                            }
                            cur.code_hash = ::Gpu::PieceHash::ShaderCode(cur.code.data(), cur.code.size() * 4);
                            cur.fetch_hash = cur.fetch.empty() ? 0 : ::Gpu::PieceHash::ShaderCode(cur.fetch.data(), cur.fetch.size() * 4);
                            Gcn::Input a;
                            a.stage = stage, a.code = cur.code, a.fetch = cur.fetch, a.rsrc1 = rsrc1, a.rsrc2 = rsrc2;
                            a.no_storage_read_nofmt = !caps.storage_image_read_without_format;
                            a.no_float_controls = !caps.signed_zero_inf_nan_preserve;
                            cur.an = Gcn::Analyze(a);
                            if (!cur.an.error.empty()) throw std::runtime_error("Analysis: " + cur.an.error);
                        } catch (const std::exception& error) {
                            cur.error = error.what();
                        }
                        if (!cur.error.empty()) throw std::runtime_error(cur.error);
                        Prepared p;
                        Gcn::Input& in = p.in;
                        in.stage = stage, in.code = cur.code, in.fetch = cur.fetch, in.rsrc1 = rsrc1, in.rsrc2 = rsrc2;
                        in.no_storage_read_nofmt = !caps.storage_image_read_without_format;
                        in.no_float_controls = !caps.signed_zero_inf_nan_preserve;
                        if (stage == Gcn::Stage::Vertex) {
                            in.rect = job.rect;
                        } else if (stage == Gcn::Stage::Pixel) {
                            in.ps_input_ena = ReadHeader<u32>(header, 32), in.ps_input_addr = ReadHeader<u32>(header, 36), in.col_format = ReadHeader<u32>(header, 28);
                            for (u32 k = 0; k < job.pixel_inputs.size() && k < 32; ++k)
                                if (cur.an.attrs >> k & 1) in.ps_input_cntl[k] = job.pixel_inputs[k];
                            for (u32 k = 0; k < 8; ++k)
                                if (in.col_format >> (4 * k) & 0xF) in.mrt_map[k] = job.mrt_map;
                        } else {
                            in.threads[0] = ReadHeader<u32>(header, 24), in.threads[1] = ReadHeader<u32>(header, 28), in.threads[2] = ReadHeader<u32>(header, 32);
                        }
                        p.slots = &cur.an.slots;
                        p.depth_resolve_slot = cur.an.depth_resolve_slot;
                        p.pkey = ProgramKey(stage, cur.code_hash, cur.fetch_hash, rsrc2, false, false);
                        const auto sharp = [&](const CatImage& im, u32* t) {
                            t[1] = u32(im.data_format) << 20 | u32(im.num_format) << 26;
                            t[2] = u32(im.width - 1) | u32(im.height - 1) << 14;
                            t[3] = u32(im.dst_select[0]) | u32(im.dst_select[1]) << 3 | u32(im.dst_select[2]) << 6 | u32(im.dst_select[3]) << 9 |
                                   (im.type == 14 || im.type == 15 ? u32(im.sample_log2) : 0) << 16 | u32(im.type) << 28;
                        };
                        const auto usage = [&](std::initializer_list<u8> kinds, u32 reg, u8 flags = 0) -> const std::array<u8, 4>* {
                            for (const auto& u : s.usage)
                                if (u[2] == reg && std::ranges::contains(kinds, u[0]) && (u[3] & flags) == flags) return &u;
                            return nullptr;
                        };
                        std::vector<std::array<u32, 8>> store(cur.an.slots.size() + cur.an.formats.size());
                        for (size_t k = 0; k < cur.an.slots.size(); ++k) {
                            const Gcn::Slot& sl = cur.an.slots[k];
                            const Gcn::Path& path = sl.path;
                            u32* w = store[k].data();
                            // A descriptor read through the extended user data pointer (usage 0x1b) is the usage register 16 + offset / 4.
                            const bool eud = path.depth == 1 && usage({0x1b}, path.reg);
                            const u32 reg = eud ? 16 + path.offset[0] / 4 : path.reg;
                            if (sl.kind == Gcn::SlotKind::Sampler) {
                                u32 slot = ~0u;
                                if (path.depth == 0 || eud) { if (const auto* u = usage({1}, reg)) slot = (*u)[1]; }
                                else if (path.depth == 1) { if (const auto* u = usage({0x15}, path.reg)) slot = (*u)[1] + path.offset[0] / 16; }
                                if (slot == ~0u) throw std::runtime_error(std::format("Sampler slot from an unknown source (path depth {} reg {}, usage{})", path.depth, path.reg, Usages(s)));
                                if (slot < 32) w[0] = (job.unnormalized >> slot & 1) << 15 | (job.degamma >> slot & 1) << 20;
                            } else {
                                const CatImage* im = nullptr;
                                bool known = false;
                                if (path.depth == 0 || eud) {
                                    if (const auto* u = usage({0, 4}, reg, 2)) known = true, im = Find(job, (*u)[1]);
                                } else if (path.depth == 1) {
                                    if (usage({0x13, 0x19}, path.reg)) known = true, im = Find(job, path.offset[0] / 32);
                                    else for (const auto& u : s.usage)  // SRT roots: pointer pairs from the usage register on
                                        if (u[0] == 0x0a && path.reg >= u[2] && (path.reg - u[2]) % 2 == 0 && (path.reg - u[2]) / 2 < job.srts.size()) {
                                            known = true;
                                            for (const auto& [offset, image] : job.srts[(path.reg - u[2]) / 2])
                                                if (offset * 4 == path.offset[0]) im = &image;
                                            break;
                                        }
                                }
                                if (!known) throw std::runtime_error(std::format("Image slot from an unknown source (path depth {} reg {}, usage{})", path.depth, path.reg, Usages(s)));
                                sharp(im ? *im : fallback, w);
                            }
                            p.desc.push_back(w);
                        }
                        for (size_t j = 0; j < cur.an.formats.size(); ++j) {
                            const Gcn::Path& path = cur.an.formats[j];
                            u32* w = store[cur.an.slots.size() + j].data();
                            w[3] = path.depth == 1 && path.offset[0] / 16 < 16 ? job.words[path.offset[0] / 16] : NullWord;
                            p.fdesc.push_back(w);
                        }
                        SpecifySlots(p);
                        const auto add_contract = [&] {
                            Contracted c{Recipe::From(in, cur.code_hash, cur.fetch_hash, p.spec), p.key, {f.data + s.code, s.footer - s.code}, {}, job.source, job.paths};
                            if (cur.generated) c.recipe.fetch = cur.fetch;
                            else if (!cur.fetch.empty()) c.fetch = {f.data + s.footer + 28, cur.fetch.size() * 4};
                            mine[t].list.push_back(std::move(c));
                            if (pipelines_out && !job.pair_ids.empty() && (stage == Gcn::Stage::Vertex || stage == Gcn::Stage::Pixel)) {
                                for (u32 pid : job.pair_ids)
                                    mine[t].pairs.push_back({pid, p.key, in.col_format, job.paths, u8(stage), job.rect});
                            }
                        };
                        add_contract();
                        if (cur.code_hash == Gcn::MotionVectorResolveCodeHash()) {
                            for (size_t k = 0; k < cur.an.slots.size(); ++k) {
                                const u32* w = p.desc[k];
                                auto& spec = p.spec[k];
                                if (!w || cur.an.slots[k].kind != Gcn::SlotKind::Texture || spec.dim != Gcn::Dim::D2Ms ||
                                    spec.cls != Gcn::Class::Float || (w[1] >> 20 & 0x3F) != 1 || (w[1] >> 26 & 0xF)) continue;
                                const auto sel = DstSel(w);
                                spec.stencil = true;
                                spec.dst_sel = u16(u32(sel.r) | u32(sel.g) << 3 | u32(sel.b) << 6 | u32(sel.a) << 9);
                                p.key = TranslationKey(p.pkey, in, p.spec);
                                add_contract();
                                spec.stencil = false, spec.dst_sel = 0;
                            }
                        }
                        if (p.depth_resolve_slot >= 0 && size_t(p.depth_resolve_slot) < p.spec.size() &&
                            p.spec[p.depth_resolve_slot].dim == Gcn::Dim::D2Ms && p.spec[p.depth_resolve_slot].cls == Gcn::Class::Float &&
                            !p.spec[p.depth_resolve_slot].msaa4) {
                            u32& word = store[p.depth_resolve_slot][3];
                            word = (word & ~(0xFu << 16)) | (2u << 16);
                            SpecifySlots(p);
                            add_contract();
                        }
                    } catch (const std::exception& error) { ++mine[t].why[std::string(error.what()).substr(0, 200)]; }
            });
    }
    // Parse order (the executable, then the WADs by name, the catalog sorted its files), then by key: the same on every launch.
    std::vector<Contracted> all;
    std::map<std::string, u32> why;
    for (Mine& m : mine) {
        std::ranges::move(m.list, std::back_inserter(all));
        for (const auto& [reason, count] : m.why) why[reason] += count;
    }
    std::ranges::sort(all, {}, [](const Contracted& c) { return std::pair{c.file + 1, c.key}; });
    std::unordered_map<u64, size_t> placed;  // key -> its place in `out`: configurations of several paths can share a variant
    for (Contracted& c : all)
        if (const auto [at, fresh] = placed.emplace(c.key, out.size()); fresh) out.push_back(std::move(c));
        else out[at->second].paths |= c.paths;
    std::fprintf(stderr, "AOT: contract: %zu variants from %zu configurations (%.1f s)\n", out.size(), jobs.size(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
    if (pipelines_out) {
        pipelines_out->clear();
        struct PsCand {
            u64 key;
            u32 col_format;
        };
        // VS variants: mesh, procedural or embedded (primary), a mesh stream set of a procedural recipe (layout), a TextFX
        // glyph declaration; PS variants: primary, or with sRGB views of generic slots only.
        struct Bucket {
            std::vector<u64> primary_vs, layout_vs, glyph_vs;
            std::vector<PsCand> primary_ps, srgb_ps;
        };
        std::vector<Bucket> buckets(pair_metas.size());
        for (const Mine& m : mine) {
            for (const PairEntry& e : m.pairs) {
                if (e.pair_id >= buckets.size()) continue;
                Bucket& b = buckets[e.pair_id];
                if (e.stage == u8(Gcn::Stage::Vertex) && !e.rect) {
                    auto& dst = e.paths & ((1u << kVsMesh) | (1u << kVsProcedural) | (1u << kVsEmbedded)) ? b.primary_vs : e.paths & (1u << kVsGlyph) ? b.glyph_vs : b.layout_vs;
                    if (std::find(dst.begin(), dst.end(), e.key) == dst.end()) dst.push_back(e.key);
                } else if (e.stage == u8(Gcn::Stage::Pixel)) {
                    const bool prim = (e.paths & ((1u << kPsMaterial) | (1u << kPsPass) | (1u << kPsEmbedded))) != 0;
                    auto& dst = prim ? b.primary_ps : b.srgb_ps;
                    if (std::none_of(dst.begin(), dst.end(), [&](const PsCand& c) { return c.key == e.key && c.col_format == e.col_format; }))
                        dst.push_back({e.key, e.col_format});
                }
            }
        }
        // Every draw of the engine uses the [-1, 1] clip space. The scene target is sampled as its UNORM image and rendered
        // through its sRGB view: opaque materials draw there multisampled; translucent ones (and screen passes) draw to
        // single-sampled UNORM targets. A run's pipelines confirmed each rule (pipelines-*.bin through tools\plstat).
        const VkFormat scene_fmt = SurfaceFormat(data.scene_color.data_format, data.scene_color.num_format);
        VkFormat scene_srgb = SurfaceFormat(data.scene_color.data_format, 9);
        if (scene_srgb == VK_FORMAT_UNDEFINED) scene_srgb = VK_FORMAT_R8G8B8A8_SRGB;
        constexpr u8 scene_samples = 0;  // the scene's sample count, set when the AOT builds
        constexpr u8 kClip = FixedPipelineState::kNegOne;
        // A material class's blend as the draw sets it: the runtime's MIN/MAX rewrite; no blending: the zero CB_BLEND_CONTROL.
        const auto blend_of = [](const MaterialClass& c) -> FixedAttachmentBlend {
            if (c[1] == 0xff) return {0, 0xF};
            u32 cf = c[1], cs = c[3], cd = c[5];
            u32 af = c[2] == 0xff ? cf : c[2], as = c[2] == 0xff ? cs : c[4], ad = c[2] == 0xff ? cd : c[6];
            RewriteMinMax(cs, cf, cd, true, false), RewriteMinMax(as, af, ad, true, true);
            return {1, 0xF, u8(BlendFactor(cs)), u8(BlendFactor(cd)), u8(BlendOp(cf)), u8(BlendFactor(as)), u8(BlendFactor(ad)), u8(BlendOp(af))};
        };

        std::vector<ContractedPipeline> derived;
        auto emit = [&](u64 vs_key, u64 ps_key, u32 file, const FixedPipelineState& st, u64 gate) {
            derived.push_back({vs_key, ps_key, 0ull, gate, file, 0u, st});
        };
        constexpr VkFormat kUnorm = VK_FORMAT_R8G8B8A8_UNORM;
        // What a pair draws: every target its recipe implies; pass code's model meshes; a screen pass sampling the scene through sRGB views.
        enum Draws : u8 { kAll, kMeshes, kScreenSrgb };
        for (size_t pid = 0; pid < pair_metas.size(); ++pid) {
            const PairMeta& meta = pair_metas[pid];
            if (!meta.derive) continue;
            const Bucket& b = buckets[pid];
            // A WAD technique no material draws with is drawn by pass code: depth and translucent passes over the model meshes
            // (every stream set), screen passes with the recipe's own declaration. The run's 25 such pipelines took these states.
            const bool pass = meta.classes.empty() && meta.file != ~0u;
            std::vector<u64> vs_list = b.primary_vs;
            if (!meta.classes.empty()) vs_list.insert(vs_list.end(), b.glyph_vs.begin(), b.glyph_vs.end());  // TextFX materials
            const bool fallback = vs_list.empty();
            if (fallback) vs_list.insert(vs_list.end(), b.layout_vs.begin(), b.layout_vs.end()), vs_list.insert(vs_list.end(), b.glyph_vs.begin(), b.glyph_vs.end());
            const auto& ps_list = !b.primary_ps.empty() ? b.primary_ps : b.srgb_ps;
            if (vs_list.empty()) continue;
            if (ps_list.empty()) {
                const size_t lim = std::min<size_t>(vs_list.size(), 8);
                for (size_t vi = 0; vi < lim; ++vi) {
                    FixedPipelineState z_ms{};
                    z_ms.depth_format = VK_FORMAT_D32_SFLOAT_S8_UINT;
                    z_ms.stencil_format = VK_FORMAT_D32_SFLOAT_S8_UINT;
                    z_ms.samples = scene_samples;
                    z_ms.topology_class = 2;
                    z_ms.flags = kClip;
                    emit(vs_list[vi], 0, meta.file, z_ms, meta.gate);

                    FixedPipelineState z_shadow{};
                    z_shadow.depth_format = VK_FORMAT_D32_SFLOAT;
                    z_shadow.samples = 1;
                    z_shadow.topology_class = 2;
                    z_shadow.flags = kClip;
                    emit(vs_list[vi], 0, meta.file, z_shadow, meta.gate);
                }
                continue;
            }
            u32 pair_count = 0;
            const auto pair = [&](u64 vs_key, const PsCand& pc, Draws draws) {
                if (pair_count++ >= 32) return;
                u32 count = 0;
                for (u32 k = 0; k < 8; ++k)
                    if (pc.col_format >> (4 * k) & 0xF) count = k + 1;
                u64 gate = meta.gate;
                // The pair on `n` color targets (MRT0 `first`, the others `rest`), with the D32S8 depth when `depth`.
                const auto put = [&](u32 n, VkFormat first, VkFormat rest, bool depth, u8 samples, const MaterialClass* c = nullptr) {
                    FixedPipelineState s{};
                    s.color_count = u8(n);
                    for (u32 k = 0; k < n; ++k) {
                        if (pc.col_format >> (4 * k) & 0xF) s.color_formats[k] = k ? rest : first;
                        s.blend[k] = c ? blend_of(*c) : FixedAttachmentBlend{0, 0xF};
                    }
                    if (depth) s.depth_format = s.stencil_format = VK_FORMAT_D32_SFLOAT_S8_UINT;
                    s.samples = samples, s.topology_class = 2, s.flags = FixedPipelineState::kHasPs | kClip;
                    emit(vs_key, pc.key, meta.file, s, gate);
                };
                // Depth only (a pass binds no color target whatever the PS exports): the scene's depth and the single-sampled one
                // (the same state without MSAA, built once).
                const auto depth_only = [&] { put(0, {}, {}, true, scene_samples), put(0, {}, {}, true, 1); };
                if (draws == kScreenSrgb) {
                    put(count, scene_srgb, scene_srgb, false, 1);
                } else if (!meta.classes.empty()) {
                    // Several targets are single-sampled UNORM ones; opaque materials were never seen there: gated on the PS.
                    for (const MaterialClass& c : meta.classes)
                        if (count < 2 && !c[0]) put(count, scene_srgb, kUnorm, true, scene_samples, &c);
                        else if (c[0]) put(count, kUnorm, kUnorm, true, 1, &c);
                        else if (meta.code) gate = meta.code, put(count, kUnorm, kUnorm, true, 1, &c), gate = meta.gate;
                } else if (count == 0) {
                    depth_only();
                } else if (pass) {
                    depth_only(), put(count, scene_fmt, kUnorm, true, 1);  // the translucent pass
                    if (draws == kAll) put(count, scene_fmt, kUnorm, false, 1), put(count, scene_srgb, scene_srgb, false, 1);
                } else {
                    put(count, scene_fmt, kUnorm, false, 1), put(count, scene_srgb, scene_srgb, false, 1);
                    put(count, scene_fmt, kUnorm, true, scene_samples);
                }
            };
            for (u64 vs_key : vs_list) for (const PsCand& pc : ps_list) pair(vs_key, pc, kAll);
            if (pass && !fallback) {
                for (u64 vs_key : b.layout_vs) for (const PsCand& pc : ps_list) pair(vs_key, pc, kMeshes);
                if (!b.primary_ps.empty()) for (u64 vs_key : vs_list) for (const PsCand& pc : b.srgb_ps) pair(vs_key, pc, kScreenSrgb);
            }
        }
        auto raw_hash = [](const ContractedPipeline& p) {
            return GraphicsPipelineKey{p.vs_key, p.ps_key, p.gs_mask, p.state}.Hash();
        };
        std::ranges::sort(derived, {}, [&](const ContractedPipeline& p) {
            return std::tuple{p.file + 1, p.vs_key, p.ps_key, p.gs_mask, raw_hash(p)};
        });
        std::unordered_multimap<u64, size_t> seen_pl;
        pipelines_out->reserve(derived.size());
        for (const ContractedPipeline& cp : derived) {
            const u64 h = raw_hash(cp);
            auto [lo, hi] = seen_pl.equal_range(h);
            bool dup = false;
            for (auto it = lo; it != hi; ++it) {
                ContractedPipeline& prev = (*pipelines_out)[it->second];
                if (prev.vs_key == cp.vs_key && prev.ps_key == cp.ps_key && prev.gs_mask == cp.gs_mask && prev.state == cp.state) {
                    dup = true;
                    if (!cp.gate) prev.gate = 0;  // another pair derives it without a gate
                    break;
                }
            }
            if (!dup) {
                seen_pl.emplace(h, pipelines_out->size());
                pipelines_out->push_back(cp);
            }
        }
        std::fprintf(stderr, "AOT: contract: %zu pipeline pairs derived\n", pipelines_out->size());
    }
    std::vector<std::pair<u32, std::string>> ranked;  // the most frequent reasons first
    ranked.reserve(why.size());
    for (const auto& [reason, count] : why) ranked.emplace_back(count, reason);
    std::ranges::sort(ranked, std::greater<>{});
    for (size_t k = 0; k < ranked.size() && k < 40; ++k) std::fprintf(stderr, "AOT:   convert: %u %s\n", ranked[k].first, ranked[k].second.c_str());
    if (ranked.size() > 40) std::fprintf(stderr, "AOT:   convert: ... and %zu more reasons\n", ranked.size() - 40);
}
} // namespace Gpu
