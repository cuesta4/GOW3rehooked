// SPDX-License-Identifier: GPL-2.0-or-later
// Shader runtime: identifies a draw's programs, builds the translation key and gets the shader object, translating on a miss.
#include "gpu/shaders/shader_runtime.h"
#include <bit>
#include <zstd.h>
#include "gpu/shaders/shader_jit.h"
#include "gpu/shaders/shader_recipes.h"
#include "gpu/textures.h"
#include "gpu/renderer_impl.h"
#include "main/host.h"

namespace Gpu {
namespace {
constexpr u32 kClearVs[]{0xBEEB03FF, 7, 0x36020081, 0x34020281, 0x360000C2, 0x4A0202C1, 0x4A0000C1, 0x7E020B01,
                         0x7E000B00, 0x7E040280, 0x7E0602F2, 0xF80008CF, 0x03020001, 0xF800020F, 0x03030303, 0xBF810000};
}

bool Renderer::Impl::ClearVs(std::span<const u32> code) {
    return code.size() >= std::size(kClearVs) && std::equal(std::begin(kClearVs), std::end(kClearVs), code.begin());
}

bool Renderer::Impl::SetupClearRect(std::string& error) {
    clear_rect_spirv = compiler.Spirv(R"(#version 460
layout(location = 0) out vec4 param0;
void main() {
    gl_Position = vec4(2.0 * float(gl_VertexIndex & 1) - 1.0,
                       2.0 * float((gl_VertexIndex >> 1) & 1) - 1.0, 0.0, 1.0);
    param0 = vec4(1.0);
}
)", shaderc_vertex_shader, error);
    if (clear_rect_spirv.empty()) return false;
    if (caps.path == ShaderPath::Objects) {
        clear_rect_ref = {kClearRectVsKey, Create(clear_rect_spirv, VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_GEOMETRY_BIT),
                          VK_NULL_HANDLE, VK_NULL_HANDLE};
    } else {
        const VkShaderModule mod = CreateShaderModule(clear_rect_spirv);
        const VkPipeline lib = (caps.path == ShaderPath::Gpl && mod) ? CreateGplPreRaster(mod, VK_NULL_HANDLE, !caps.depth_clip_negative_one_to_one, false) : VK_NULL_HANDLE;
        clear_rect_ref = {kClearRectVsKey, VK_NULL_HANDLE, mod, lib};
    }
    if (!clear_rect_ref || (caps.path == ShaderPath::Gpl && !clear_rect_ref.pl)) {
        DestroyShaderRef(clear_rect_ref);
        clear_rect_ref = {};
        return error = "SDK clear vertex shader creation failed", false;
    }
    return true;
}

bool Renderer::Impl::RectVs(u32 op, const u32* b, const CpuRead& cpu) const {
    if (uc[0x242] != 0x11 || op != 0x2D || b[0] != 3) return false;
    if (!HasClearRectVs()) return true;
    const u64 addr = (u64)(sh[0x48] | (u64)sh[0x49] << 32) << 8;
    const auto* code = static_cast<const u32*>(cpu(addr, sizeof(kClearVs)));
    return code && ClearVs({code, std::size(kClearVs)});
}

VkShaderEXT Renderer::Impl::Create(const std::vector<u32>& spirv, VkShaderStageFlagBits stage, VkShaderStageFlags next) {
    return CreateFrom(VK_SHADER_CODE_TYPE_SPIRV_EXT, spirv.data(), spirv.size() * 4, stage, next);
}

// A shader object from SPIR-V or from a driver binary (VK_INCOMPATIBLE_SHADER_BINARY_EXT, a binary of another driver: null).
VkShaderEXT Renderer::Impl::CreateFrom(VkShaderCodeTypeEXT type, const void* code, size_t bytes, VkShaderStageFlagBits stage, VkShaderStageFlags next) {
    VkPushConstantRange pcr{VK_SHADER_STAGE_ALL, 0, sizeof(Push)};
    const VkDescriptorSetLayout layouts[2]{set_layout, table_layout};
    VkShaderCreateInfoEXT ci{VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT};
    ci.stage = stage, ci.nextStage = next, ci.codeType = type, ci.codeSize = bytes, ci.pCode = code;
    ci.pName = "main", ci.setLayoutCount = 2, ci.pSetLayouts = layouts, ci.pushConstantRangeCount = 1, ci.pPushConstantRanges = &pcr;
    VkShaderEXT s = VK_NULL_HANDLE;
    if (const VkResult e = CreateShaders(dev, 1, &ci, nullptr, &s); e != VK_SUCCESS) {
        if (e != VK_INCOMPATIBLE_SHADER_BINARY_EXT) Report(std::format("vkCreateShadersEXT {}: stage {:x} next {:x} code type {} of {} bytes", (int)e, (u32)stage, (u32)next, (int)type, bytes));
        s = VK_NULL_HANDLE;
    }
    return s;
}

std::vector<u8> Renderer::Impl::Binary(VkShaderEXT s) {
    size_t size = 0;
    if (!GetShaderBinary || GetShaderBinary(dev, s, &size, nullptr) != VK_SUCCESS || !size) return {};
    std::vector<u8> data(size);
    if (GetShaderBinary(dev, s, &size, data.data()) != VK_SUCCESS) return {};
    data.resize(size);
    return data;
}

int FetchSgpr(std::span<const u32> code) {
    for (size_t i = 0; i < code.size() && code[i] != 0xBF810000;) {
        const u32 x = code[i];
        if (x >> 23 == 0x17D && (x >> 8 & 0xFF) == 33) return int(x & 0xFF);
        const u32 e = x >> 26;
        i += e == 0x34 || e == 0x36 || e == 0x38 || e == 0x3A || e == 0x3C || e == 0x3E ? 2
             : x >> 27 == 0x18                                                         ? 1 + ((x & 0x1FF) == 0xFF)
             : x >> 30 == 2                                                            ? 1 + ((x & 0xFF) == 0xFF || (x >> 8 & 0xFF) == 0xFF)
                                                                                       : 1 + ((x & 0x1FF) == 0xFF);
    }
    return -1;
}

// Program bytes up to its OrbShdr footer (the code size the engine stored after s_endpgm).
static std::span<const u32> Program(u64 addr, const CpuRead& cpu) {
    const u32* h = (const u32*)cpu(addr, 8);
    if (!h || h[0] != 0xBEEB03FF) return {};
    const u64 bytes = ((u64)h[1] + 1) * 8;
    const u32* code = (const u32*)cpu(addr, bytes);
    return code ? std::span<const u32>(code, bytes / 4) : std::span<const u32>{};
}

static u64 ReadOrbFooterHash(u64 addr, u64 code_bytes, const CpuRead& cpu) {
    const u8* f = code_bytes ? (const u8*)cpu(addr + code_bytes, 24) : nullptr;
    if (!f) return 0;
    constexpr char kMagic[] = "OrbShdr";
    for (size_t i = 0; i < 7; ++i)
        if (f[i] != (u8)kMagic[i]) return 0;
    u64 hash = 0;
    for (size_t i = 0; i < 8; ++i) hash |= (u64)f[16 + i] << (i * 8);
    return hash;
}

u64 Renderer::Impl::OrbShaderHash(u64 prog, const CpuRead& cpu) {
    const u32* h = (const u32*)cpu(prog, 8);
    if (!h || h[0] != 0xBEEB03FF) return 0;
    return ReadOrbFooterHash(prog, ((u64)h[1] + 1) * 8, cpu);
}

bool Renderer::Impl::ShouldSkipMotionBlur(const CpuRead& cpu) {
    if (!skip_motion_blur) {
        mb_seen_blur = false;
        return false;
    }
    const u32 mode = ctx[0x202] >> 4 & 7;
    const u64 ps_addr = (u64)(sh[0x08] | (u64)sh[0x09] << 32) << 8;
    if (mode != 1 || (ctx[0x000] & 3) || !ps_addr || query_open || query_next != ~0u || pred) {
        mb_seen_blur = false;
        return false;
    }
    auto [it, fresh] = orb_hash_cache.try_emplace(ps_addr, 0);
    if (fresh) it->second = OrbShaderHash(ps_addr, cpu);
    const u64 hash = it->second;
    if (hash == kMotionBlurPassOrbHash) {
        mb_seen_blur = true;
        return true;
    }
    const bool skip = hash == kMotionBlurCompositeOrbHash && mb_seen_blur && ((ctx[0x1E0] >> 30) & 1);
    mb_seen_blur = false;
    return skip;
}
// Where a descriptor or pointer of `n` dwords lies: Renderer::Impl::Locate (renderer_impl.h).
static std::span<const u32> FetchProgram(u64 addr, const CpuRead& cpu) {
    for (u32 i = 0; i < 0x100; ++i) {
        const u32* w = (const u32*)cpu(addr + i * 4ull, 4);
        if (!w) return {};
        if ((w[0] & 0xFF80FF00) == 0xBE802000) return {(const u32*)cpu(addr, (i + 1) * 4ull), i + 1};
    }
    return {};
}

// Translated program for a draw stage; fills `table` with user data and the slots' bindless indices.
ShaderRef Renderer::Impl::Shader(Gcn::Stage stage, u64 prog, const u32* user, const CpuRead& cpu, u32 rsrc1, u32 rsrc2, u64& out_mask,
                                 bool rect, std::vector<u32>& table, u32 blend, bool identified) {
    RenderProbe::Scope shader_time(RenderProbe::Shader);
    Prepared& p = prep;
    if (!identified && !Identify(stage, prog, user, cpu, rsrc1, rsrc2, rect, p)) return {};
    p.in.blend_interlock = blend;
    if (track)  // live: the memory of every V# the program loads through, and of the pointers leading to it
        for (const Gcn::Path& bp : *p.paths) {
            const u32* at = bp.reg + (bp.depth ? 2 : 4) <= 16 ? user + bp.reg : nullptr;
            for (u32 d = 0; d < bp.depth && at; ++d) {
                const u64 p = ((u64)(at[1] & 0xFFFF) << 32 | at[0]) + bp.offset[d];
                const u32 n = d + 1 < bp.depth ? 8 : 16;
                Track(p, p + n);
                at = (const u32*)cpu(p, n);
            }
            if (!at) continue;
            const u64 base = (u64)(at[1] & 0xFFFF) << 32 | at[0];
            const u32 stride = at[1] >> 16 & 0x3FFF;
            const u64 size = std::min<u64>(stride ? (u64)stride * at[2] : at[2], 64ull << 20);
            Track(base, base + size), NoteRead(base, base + size);
        }
    // Trace: every V# the program loads through, with what the GPU will find there.
    for (const Gcn::Path& bp : *p.paths) {
        if (!stats.trace || stats.trace_lines.empty()) break;
        const u32* at = bp.reg + (bp.depth ? 2 : 4) <= 16 ? user + bp.reg : nullptr;
        for (u32 d = 0; d < bp.depth && at; ++d)
            at = (const u32*)cpu(((u64)(at[1] & 0xFFFF) << 32 | at[0]) + bp.offset[d], d + 1 < bp.depth ? 8 : 16);
        if (!at) {
            stats.trace_lines.back() += std::format(" | V{} s{} unreadable", "vpc"[(int)stage], bp.reg);
            continue;
        }
        const u64 base = (u64)(at[1] & 0xFFFF) << 32 | at[0];
        const u32 stride = at[1] >> 16 & 0x3FFF, probe = std::min<u32>(stride ? stride * at[2] : at[2], 4096);
        const u8* m = (const u8*)cpu(base, probe);
        const bool mirrored = (base >> 12 < kPieces && written[base >> 12]) ||
                              std::any_of(pending.begin(), pending.end(), [&](const Pending& p) { return base >= p.addr && base < p.addr + p.size; });
        // TEMP: float elements that cannot be geometry (NaN, Inf, |x| >= 2^34): count, first byte offset and bits.
        std::string bad;
        if (const u32 df = at[3] >> 15 & 0xF, comps = df == 4 ? 1 : df == 11 ? 2 : df == 13 ? 3 : df == 14 ? 4 : 0; base && comps && (at[3] >> 12 & 7) == 7) {
            const u32 es = stride ? stride : comps * 4, n = std::min<u32>(stride ? at[2] : at[2] / es, (16u << 20) / es);
            if (const u8* all = n ? (const u8*)cpu(base, (u64)es * (n - 1) + comps * 4) : nullptr) {
                u32 count = 0, first = 0, bits = 0;
                for (u32 i = 0; i < n; ++i)
                    for (u32 c = 0; c < comps; ++c) {
                        u32 v;
                        std::memcpy(&v, all + (u64)i * es + c * 4, 4);
                        if ((v >> 23 & 0xFF) >= 127 + 34 && !count++) first = i * es + c * 4, bits = v;
                    }
                if (count) bad = std::format(" BAD {} of {} @{:x} {:08x}", count, n * comps, first, bits);
            }
        }
        stats.trace_lines.back() += std::format(" | V{} s{}{} {:x} {}x{} f{}/{} {}{}", "vpc"[(int)stage], bp.reg, bp.depth ? "*" : "", base, stride, at[2],
                                                at[3] >> 15 & 0xF, at[3] >> 12 & 7,
                                                !m ? "unreadable" : !mirrored ? "not-mirrored" : std::all_of(m, m + probe, [](u8 x) { return !x; }) ? "zero" : "ok", bad);
    }
    // Draw table (the user data goes as push constants): the scalar loads' constants as the GPU would read them now (past
    // the V#'s range: 0); the bindless element of each slot (0: dummy image / default sampler).
    table.clear();
    for (const Gcn::Const& c : *p.consts) {
        const size_t at = table.size();
        u64 n = (c.hi - c.lo) / 4;
        table.resize(at + n);
        const u32* w = Locate(c.path, c.buffer ? 4 : 2, user, cpu);
        if (!w) continue;
        const u64 buf_addr = (u64)(w[1] & 0xFFFF) << 32 | w[0];
        const u64 base = buf_addr + c.lo;
        u64 lim = 0;
        if (c.buffer) {
            const u64 stride = w[1] >> 16 & 0x3FFF;
            lim = stride ? stride * w[2] : w[2];
            n = c.lo >= lim ? 0 : std::min(n, (lim - c.lo + 3) / 4);
        }
        if (const u32* s = n ? (const u32*)cpu(base, n * 4) : nullptr) std::copy_n(s, n, table.data() + at);
        if (p.dof_composite && skip_dof_spartan && c.buffer && lim == 80 && n) {
            if (const auto* f = (const float*)cpu(buf_addr, 80);
                f && f[4] < 0.0f && f[5] < 0.0f && f[6] < 0.0f && f[7] > 0.0f && f[8] > 0.99f && f[8] < 1.01f) {
                for (const u64 d : {0ULL, 1ULL, 4ULL, 5ULL, 6ULL, 7ULL}) {
                    const u64 byte_off = d * 4;
                    if (byte_off >= c.lo && byte_off - c.lo < n * 4 && ((byte_off - c.lo) & 3) == 0)
                        table[at + (byte_off - c.lo) / 4] = 0;
                }
                if (static std::atomic<bool> logged{false};
                    Gow3::DebugEnabled() && !logged.exchange(true, std::memory_order_relaxed)) {
                    Gow3::Log("DOF composite CB0 constants zeroed (SpartanPort)");
                }
            }
        }
        if (stats.trace && n)  // TEMP: constants the GPU wrote (the CPU's copy would be stale)
            for (auto* list : {&gpu_writes, &last_writes})
                for (auto& g : *list)
                    if (g.lo < base + n * 4 && g.hi > base) Note(std::format("constants {:x} over a GPU write", base));
    }
    table_ps = stage == Gcn::Stage::Pixel;
    // Everything the variant and the slot elements come from besides the program key (Specify, Texture, Sampler): a
    // repeat of one of the program's last inputs reuses them and redoes only the slots' per-draw checks.
    const u64 ep = Memos().epoch;
    u32 n = 0;  // past kInputs: not kept
    auto put = [&](const u32* w, u32 k) {
        if (n + k <= kInputs) std::memcpy(inputs + n, w, k * 4);
        n += k;
    };
    const u32 head[7]{u32(p.pkey), u32(p.pkey >> 32), rsrc1, u32(rect), u32(ep), u32(ep >> 32), blend};
    put(head, 7);
    if (stage == Gcn::Stage::Pixel) {  // input enables, COL_FORMAT, the 8 CB_COLOR*_INFO (15 registers apart), input controls
        const u32 ps[11]{ctx[0x1B3], ctx[0x1B4], ctx[0x1C5], ctx[0x31C], ctx[0x32B], ctx[0x33A], ctx[0x349], ctx[0x358], ctx[0x367], ctx[0x376], ctx[0x385]};
        put(ps, 11), put(&ctx[0x191], 32);
    }
    if (stage == Gcn::Stage::Compute) put(&sh[0x207], 3);
    for (size_t k = 0; k < p.slots->size(); ++k) {
        const u32 present = p.desc[k] != nullptr;
        put(&present, 1);
        if (present) put(p.desc[k], (*p.slots)[k].kind == Gcn::SlotKind::Sampler ? 4 : 8);
    }
    for (const u32* v : p.fdesc) {
        const u32 w[2]{v != nullptr, v ? v[3] : 0};  // all BufferSpec reads
        put(w, v ? 2 : 1);
    }
    if (!p.stencil_resolve && !stats.trace && n <= kInputs)
        for (u32 i = 0; i < 4; ++i)
            if (const Recent& r = p.use->recent[p.use->order[i]]; r.in.size() == n && !std::memcmp(r.in.data(), inputs, n * 4)) {
                if (RenderProbe::current) ++RenderProbe::current->recent_hits;
                std::rotate(p.use->order, p.use->order + i, p.use->order + i + 1);
                for (size_t k = 0; k < p.slots->size(); ++k)
                    table.push_back(!p.desc[k] || (*p.slots)[k].kind == Gcn::SlotKind::Sampler
                                        ? r.slots[k].e
                                        : Reuse(r.slots[k], p.desc[k], (*p.slots)[k].kind == Gcn::SlotKind::Storage));
                if (p.motion_blur) table.push_back(std::bit_cast<u32>(motion_blur_rate));
                return out_mask = r.out_mask, r.s;
            }
    Specify(stage, p);
    bool subs = false, keep = !stats.trace && n <= kInputs;
    slot_used.clear();
    for (size_t k = 0; k < p.slots->size(); ++k) {
        const auto& slot = (*p.slots)[k];
        const bool tex = p.desc[k] && slot.kind != Gcn::SlotKind::Sampler;
        tex_sub = 0;
        const u32 e = !p.desc[k] ? 0 : tex ? Texture(p.desc[k], p.spec[k], slot.kind == Gcn::SlotKind::Storage, p.stencil_resolve) : Sampler(p.desc[k]);
        table.push_back(e), slot_used.push_back(tex ? tex_used : MemoTex{nullptr, e, VK_NULL_HANDLE, 0}), keep &= !tex || tex_keep;
        if (tex_sub)  // the slot samples part of its image: one variant of the program (the extents come from the T# and the image)
            p.spec[k].part = true, subs = true;
    }
    if (p.motion_blur) table.push_back(std::bit_cast<u32>(motion_blur_rate));
    if (subs) p.key = TranslationKey(p.pkey, p.in, p.spec);
    const ShaderRef s = Translated(stage, prog, cpu, p, out_mask);
    if (ProgUse* u = keep && s ? Memos().progs.Find(p.pkey) : nullptr) {  // not a shader still compiling  // found again: compiling may have added program keys
        Recent& r = u->recent[u->order[3]];  // the least recent gives way
        std::rotate(u->order, u->order + 3, u->order + 4);
        r.in.assign(inputs, inputs + n), r.out_mask = out_mask, r.s = s, r.slots = slot_used;
    }
    return s;
}

ShaderRef Renderer::Impl::Translated(Gcn::Stage stage, u64 prog, const CpuRead& cpu, Prepared& p, u64& out_mask) {
    const u64 key = p.key;
    const bool clear_rect = HasClearRectVs() && p.in.rect && ClearVs(p.in.code);
    auto found = [&] {
        auto it = shaders.find(key);
        if (it == shaders.end()) return false;
        if (auto o = vs_outputs.find(key); o != vs_outputs.end()) out_mask = o->second;
        return true;
    };
    // Traced frames keep the SPIR-V of every program they run, compiled before or not (written with the dump).
    if (stats.trace) {
        std::string name = std::format("{}-{:x}-{:x}", "vpc"[(int)stage], prog, key & 0xFFFF), err;
        if (std::none_of(stats.trace_spirv.begin(), stats.trace_spirv.end(), [&](auto& e) { return e.first == name; })) {
            std::vector<u32> words;
            u64 mask = 0;
            if (clear_rect) words = clear_rect_spirv;
            else Gcn::EmitSpirv(Gcn::Analyze(p.in), p.in, p.spec, words, mask, err);
            stats.trace_spirv.push_back({name, std::string((const char*)words.data(), words.size() * 4)});
        }
    }
    if (clear_rect) return out_mask = 1, clear_rect_ref;
    auto& memo = Memos().shaders;  // compiled shaders stay
    if (const auto* m = memo.Find(key)) return out_mask = m->second, m->first;
    // First use of this variant on the thread (translated now, found in the cache or made ready ahead by Discover or a
    // warm pass): kept as a recipe for the AOT.
    if (!stats.trace && !p.in.probe && !p.in.boom) Recipes::Instance().Note(key, p.in, p.spec, ShaderCache::Instance().Contains(key));
    {
        std::shared_lock lock(prog_lock);
        if (found()) return memo[key] = {shaders.find(key)->second, out_mask}, shaders.find(key)->second;
    }
    // First miss of the frame: the rest of the frame's programs go to the compile threads (traced frames compile here,
    // in order). A parallel block's first miss does the same for the rest of the block, so the variants a camera turn
    // reveals together load or translate at once on those threads, not one after another here; the block waits for each.
    if (!stats.trace && !in_worker && !discovered) discovered = true, Discover(cpu, {walk_chain.size(), 0}, false);
    else if (!stats.trace && in_worker && block_now && !block_discovered) block_discovered = true, Discover(cpu, block_now->to, true);
    Compiled c;
    if (stats.trace) {
        c = Compile(stage, prog, key, p.in,std::vector<u32>(p.in.code.begin(), p.in.code.end()), std::vector<u32>(p.in.fetch.begin(), p.in.fetch.end()),
                    p.spec);
    } else {
        // Sync waits for the compile. Async skips the draw until the workers finish it. Compute always waits.
        const Jit::Mode mode = Jit::mode;
        const bool compute = stage == Gcn::Stage::Compute;
        // Built by an earlier run or the AOT: the driver's binary only loads (no compile), here, so the draw is never skipped.
        const u64 content = mode == Jit::Mode::Async ? ShaderCache::Instance().ContentOf(key) : 0;
        const bool built = caps.path == ShaderPath::Objects
                               ? (content && ShaderCache::Instance().ContainsBinary(BinaryKey(content)))
                               : (content != 0);
        std::promise<Compiled> here;
        std::shared_future<Compiled> f;
        bool own = false;
        {
            std::unique_lock lock(prog_lock);
            if (found()) return shaders.find(key)->second;
            auto [it, fresh] = compiling.try_emplace(key);
            if (fresh) it->second = built ? here.get_future().share() : Async(stage, prog, p, true), own = built;
            f = it->second;
        }
        if (own) here.set_value(Compile(stage, prog, key, p.in, std::vector<u32>(p.in.code.begin(), p.in.code.end()),
                                        std::vector<u32>(p.in.fetch.begin(), p.in.fetch.end()), p.spec));
        if (mode == Jit::Mode::Async && !compute && !built && f.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return waiting = true, ShaderRef{};
        Stalled(stall_waits, stall_wait_ns, [&] { c = f.get(); });
    }
    std::unique_lock lock(prog_lock);
    if (found()) {  // another block or the WAD's warm-up (Prepare) finished it first: the first to end this compile drops its copy
        if (const auto it = compiling.find(key); it != compiling.end()) {
            compiling.erase(it);
            const ShaderRef cr = c.Ref(key);
            if (cr && cr != shaders.find(key)->second) DestroyShaderRef(cr);
        }
        return shaders.find(key)->second;
    }
    compiling.erase(key);
    if (stage == Gcn::Stage::Vertex) vs_outputs[key] = out_mask = c.out_mask;
    Count(stats.shaders);
    const ShaderRef ref = c.Ref(key);
    if (!c.Ready(caps.path, stage)) Count(stats.shader_fail), Note(std::format("shader {:x} stage {}: {}", prog, (int)stage, c.err.substr(0, 300)));
    return shaders[key] = ref;
}

bool Renderer::Impl::Identify(Gcn::Stage stage, u64 prog, const u32* user, const CpuRead& cpu, u32 rsrc1, u32 rsrc2, bool rect, Prepared& p) {
    RenderProbe::Scope identify_time(RenderProbe::Identify);
    // Code, content hash and fetch SGPR, checked once per program per frame: read again only when a page under the code
    // changed (Version), every frame when that is unknown.
    auto stale = [&](ProgMemo& m, u64 a) {
        if (m.frame == frame) return false;
        m.frame = frame;
        return !m.ver || m.ver != Version(a, m.code.size_bytes());
    };
    ProgMemo& pm = prog_memo[prog | (u64)stage];
    if (stale(pm, prog)) {
        pm.code = Program(prog, cpu);
        pm.hash = ::Gpu::PieceHash::ShaderCode(pm.code.data(), pm.code.size_bytes());
        pm.orb_hash = pm.code.empty() ? 0 : ReadOrbFooterHash(prog, pm.code.size_bytes(), cpu);
        pm.fetch_sgpr = stage == Gcn::Stage::Vertex ? FetchSgpr(pm.code) : -1;
        pm.ver = Version(prog, pm.code.size_bytes());
    }
    const auto code = pm.code;
    if (code.empty()) return Note(std::format("program {:x} unreadable", prog)), false;
    std::span<const u32> fetch;
    u64 fetch_hash = 0;
    if (const int s = pm.fetch_sgpr; s >= 0 && s < 15)
        if (const u64 fa = (u64)(user[s + 1] & 0xFFFF) << 32 | user[s]) {
            ProgMemo& fm = fetch_memo[fa];
            if (stale(fm, fa)) fm.code = FetchProgram(fa, cpu), fm.hash = ::Gpu::PieceHash::ShaderCode(fm.code.data(), fm.code.size_bytes()), fm.ver = Version(fa, fm.code.size_bytes());
            fetch = fm.code, fetch_hash = fm.hash;
        }
    p.spec.clear(), p.desc.clear(), p.fdesc.clear(), p.in = {}, p.rect = rect;
    p.motion_blur = stage == Gcn::Stage::Pixel && pm.hash == Gcn::MotionBlurCodeHash();
    p.stencil_resolve = stage == Gcn::Stage::Pixel && pm.hash == Gcn::MotionVectorResolveCodeHash();
    p.dof_composite = stage == Gcn::Stage::Pixel && pm.orb_hash == kDofCompositeOrbHash;
    p.linear_copy = stage == Gcn::Stage::Compute && pm.orb_hash == kLinearCopyCsOrbHash;
    Gcn::Input& in = p.in;
    in.stage = stage, in.code = code, in.fetch = fetch, in.rsrc1 = rsrc1, in.rsrc2 = rsrc2, in.probe = stats.trace && stage == Gcn::Stage::Vertex;
    in.boom = boom_on && !in.probe && stage == Gcn::Stage::Vertex;
    in.rect = rect && stage == Gcn::Stage::Vertex;
    in.no_storage_read_nofmt = !caps.storage_image_read_without_format;
    static const char* probe_pc = DebugOption("GOW3_PROBE_PC");  // e.g. 100017: after the fetch
    if (in.probe && probe_pc) in.probe_pc = (u32)std::strtoul(probe_pc, nullptr, 16);
    p.pkey = ProgramKey(stage, pm.hash, fetch_hash, rsrc2, in.probe, in.boom);
    // Elements of an unordered_map stay put: the entry outlives the lock. The thread's own entries start over past 4096
    // (programs come and go with the levels; each keeps up to 4 inputs).
    auto& progs = Memos().progs;
    if (progs.used >= 4096 && !progs.Find(p.pkey)) progs.clear();
    ProgUse& u = progs[p.pkey];
    if (!u.ps) {
        std::shared_lock lock(prog_lock);
        if (auto it = slots.find(p.pkey); it != slots.end()) u.ps = &it->second;
    }
    if (!u.ps) {
        Gcn::Analysis an = Gcn::Analyze(in);
        std::unique_lock lock(prog_lock);
        u.ps = &slots.emplace(p.pkey, ProgSlots{std::move(an.slots), std::move(an.buffers), std::move(an.formats), std::move(an.consts), an.attrs, an.stored, an.depth_resolve_slot}).first->second;
    }
    p.use = &u, p.slots = &u.ps->slots, p.paths = &u.ps->buffers, p.consts = &u.ps->consts, p.stored = u.ps->stored;
    p.depth_resolve_slot = u.ps->depth_resolve_slot;
    for (auto& s : *p.slots) p.desc.push_back(Locate(s.path, s.kind == Gcn::SlotKind::Sampler ? 4 : 8, user, cpu));
    for (auto& f : u.ps->formats) p.fdesc.push_back(Locate(f, 4, user, cpu));
    return true;
}

void Renderer::Impl::Specify(Gcn::Stage stage, Prepared& p) {
    Gcn::Input& in = p.in;
    if (stage == Gcn::Stage::Compute) std::copy_n(&sh[0x207], 3, in.threads);  // COMPUTE_NUM_THREAD_X/Y/Z
    if (stage == Gcn::Stage::Pixel) {
        in.ps_input_ena = ctx[0x1B3], in.ps_input_addr = ctx[0x1B4], in.col_format = ctx[0x1C5];
        std::memcpy(in.ps_input_cntl, &ctx[0x191], sizeof(in.ps_input_cntl));
        for (u32 k = 0; k < 32; ++k) in.ps_input_cntl[k] &= -u32(p.use->ps->attrs >> k & 1);  // only the interpolants the code reads count
        for (u32 k = 0; k < 8; ++k) {
            const u32 info = ctx[0x31C + k * 15], nt = info >> 8 & 7;
            if (!(in.col_format >> (4 * k) & 0xF)) continue;
            in.mrt_class[k] = nt == 4 ? 1 : nt == 5 ? 2 : 0;
            const auto swz = AmdGpu::MrtSwizzle(info >> 11 & 3, AmdGpu::DataFormat(info >> 2 & 0x1F));
            in.mrt_map[k] = u8(swz.Map(0) | swz.Map(1) << 2 | swz.Map(2) << 4 | swz.Map(3) << 6);
        }
    }
    SpecifySlots(p);
    if (!p.stencil_resolve) return;
    for (size_t k = 0; k < p.slots->size(); ++k) {
        const u32* w = p.desc[k];
        auto& spec = p.spec[k];
        if (!w || (*p.slots)[k].kind != Gcn::SlotKind::Texture || spec.dim != Gcn::Dim::D2Ms || spec.cls != Gcn::Class::Float ||
            (w[1] >> 20 & 0x3F) != 1 || (w[1] >> 26 & 0xF)) continue;
        const u64 base = ((u64)w[0] | (u64)(w[1] & 0x3F) << 32) << 8;
        if (auto it = stencil_at.find(base); it != stencil_at.end() && it->second.second == frame && it->second.first &&
            it->second.first->samples == (1u << (w[3] >> 16 & 0xF)) && (w[2] & 0x3FFF) + 1 == it->second.first->w &&
            (w[2] >> 14 & 0x3FFF) + 1 == it->second.first->h) {
            const auto sel = DstSel(w);
            spec.stencil = true;
            spec.dst_sel = u16(u32(sel.r) | u32(sel.g) << 3 | u32(sel.b) << 6 | u32(sel.a) << 9);
            p.key = TranslationKey(p.pkey, p.in, p.spec);
        }
    }
}

// The specs of the slots from their descriptors (p.desc, p.fdesc) and the translation key. The AOT calls it with descriptors it made.
void Renderer::Impl::SpecifySlots(Prepared& p) {
    Gcn::Input& in = p.in;
    p.spec.clear();
    for (size_t k = 0; k < p.slots->size(); ++k) {
        const auto& s = (*p.slots)[k];
        const u32* w = p.desc[k];
        p.spec.push_back(!w ? Gcn::SlotSpec{} : s.kind == Gcn::SlotKind::Sampler ? Gcn::SamplerSpec(w) : Gcn::TextureSpec(w));
        if (k != size_t(p.depth_resolve_slot) || p.spec.back().dim != Gcn::Dim::D2Ms || p.spec.back().cls != Gcn::Class::Float)
            p.spec.back().msaa4 = false;
        // Storage views cannot swizzle: the shader applies DST_SEL to its loads and the inverse to its stores (0: identity).
        if (const AmdGpu::CompMapping sel = w && s.kind == Gcn::SlotKind::Storage ? DstSel(w) : AmdGpu::IdentityMapping; sel != AmdGpu::IdentityMapping)
            p.spec.back().dst_sel = u16(u32(sel.r) | u32(sel.g) << 3 | u32(sel.b) << 6 | u32(sel.a) << 9);
    }
    // The engine pairs the fog composite with the fullscreen VS's normalized UV (SPI_PS_INPUT_CNTL_0 = 1: its param 1) yet binds
    // the scene with its unnormalized WhiteBuffer sampler: every pixel would read the corner texel (dark pit). The depth
    // sample at the same UV is normalized: the scene sample takes it as normalized too.
    if (in.stage == Gcn::Stage::Pixel && (in.ps_input_cntl[0] & 0x1F) == 1 && in.code.size() == 128 &&
        ::Gpu::PieceHash::ShaderCode(in.code.data(), in.code.size_bytes()) == Gcn::FogCompositeCodeHash())
        for (auto& s : p.spec) s.unnormalized = false;
    for (const u32* w : p.fdesc) p.spec.push_back(w ? Gcn::BufferSpec(w) : Gcn::SlotSpec{});
    p.key = TranslationKey(p.pkey, in, p.spec);
}

u64 ProgramKey(Gcn::Stage stage, u64 code_hash, u64 fetch_hash, u32 rsrc2, bool probe, bool boom) {
    if (stage == Gcn::Stage::Pixel && code_hash == Gcn::MotionBlurCodeHash()) code_hash ^= 0x6d6f74696f6e3031ull;
    const u64 hashes[2]{code_hash, fetch_hash};
    // Probe and exploded-vertex variants apart: each writes its own record, which only its frames push.
    return ::Gpu::PieceHash::ShaderKey(hashes, sizeof(hashes), (u64)stage << 32 | rsrc2 | (u64)probe << 62 | (boom ? (u64)(0x100 | Gcn::BoomClip()) << 40 | (u64)Gcn::BoomParamLimit() << 49 | 1ull << 63 | 1ull << 62 : 0));  // bit 62: the record carries the constant bank (never with a probe)
}

u64 TranslationKey(u64 program_key, const Gcn::Input& in, std::span<const Gcn::SlotSpec> spec) {
    u64 packed[64], *w = packed;  // the specs field by field
    std::vector<u64> more;
    if (spec.size() > std::size(packed)) more.resize(spec.size()), w = more.data();
    for (size_t k = 0; k < spec.size(); ++k) w[k] = Gcn::Pack(spec[k]);
    u64 key = ::Gpu::PieceHash::ShaderSpecKey(w, spec.size() * 8, program_key ^ in.rsrc1 ^ (in.rect ? 1ull << 63 : 0));
    if (in.stage == Gcn::Stage::Compute) key = ::Gpu::PieceHash::ShaderKey(in.threads, sizeof(in.threads), key);
    if (in.stage == Gcn::Stage::Pixel)
        key = ::Gpu::PieceHash::ShaderKey(in.ps_input_cntl, sizeof(in.ps_input_cntl),
                                   ::Gpu::PieceHash::ShaderKey(&in.mrt_class, 16, key ^ in.ps_input_ena ^ (u64)in.ps_input_addr << 32 ^ in.col_format));
    if (in.blend_interlock) key = ::Gpu::PieceHash::ShaderKey(&in.blend_interlock, sizeof(in.blend_interlock), key);
    if (in.no_storage_read_nofmt) key ^= 0x53544F524E4F5244ull;
    return key;
}

// Rect list: three corners of an axis-aligned rectangle; the corner with the right angle is opposite the missing one.
ShaderRef Renderer::Impl::RectGs(u64 mask) {
    std::unique_lock lock(prog_lock);
    if (auto it = rect_gs.find(mask); it != rect_gs.end()) return it->second;
    std::string s = "#version 460\nlayout(triangles) in;\nlayout(triangle_strip, max_vertices = 4) out;\n";
    for (u64 m = mask; m; m &= m - 1) s += std::format("layout(location = {0}) in vec4 i{0}[];\nlayout(location = {0}) out vec4 o{0};\n", std::countr_zero(m));
    s += R"(void Emit(int k, int a, int b, int c, bool far) {
    gl_Position = far ? gl_in[a].gl_Position + gl_in[b].gl_Position - gl_in[c].gl_Position : gl_in[k].gl_Position;
)";
    for (u64 m = mask; m; m &= m - 1) s += std::format("    o{0} = far ? i{0}[a] + i{0}[b] - i{0}[c] : i{0}[k];\n", std::countr_zero(m));
    s += R"(    EmitVertex();
}
void main() {
    vec2 p[3];
    for (int k = 0; k < 3; ++k) p[k] = gl_in[k].gl_Position.xy / gl_in[k].gl_Position.w;
    int c = 0;
    float best = 3.4e38;
    for (int k = 0; k < 3; ++k) {
        float d = abs(dot(p[(k + 1) % 3] - p[k], p[(k + 2) % 3] - p[k]));
        if (d < best) best = d, c = k;
    }
    int a = (c + 1) % 3, b = (c + 2) % 3;
    Emit(c, a, b, c, false);
    Emit(a, a, b, c, false);
    Emit(b, a, b, c, false);
    Emit(0, a, b, c, true);
}
)";
    std::string err;
    auto spirv = compiler.Spirv(s, shaderc_geometry_shader, err);
    ShaderRef gs{};
    if (!spirv.empty()) {
        if (caps.path == ShaderPath::Objects)
            gs = {mask, Create(spirv, VK_SHADER_STAGE_GEOMETRY_BIT, VK_SHADER_STAGE_FRAGMENT_BIT), VK_NULL_HANDLE, VK_NULL_HANDLE};
        else
            gs = {mask, VK_NULL_HANDLE, CreateShaderModule(spirv), VK_NULL_HANDLE};
    }
    if (!gs) Note("rect list geometry shader: " + err.substr(0, 300));
    return rect_gs[mask] = gs;
}

void FixedPipelineState::Canonicalize(const DynamicCaps& c) {
    if (c.depth_clip_negative_one_to_one) flags &= ~kNegOne;
    if (c.depth_clamp_enable) flags &= ~kDepthClamp;
    if (c.alpha_to_coverage_enable) flags &= ~kAlphaToCoverage;
    if (c.rasterization_samples) samples = 1;
    if (c.unrestricted_topology && !(flags & kHasGs)) topology_class = 2;
    if (c.color_blend_enable && c.color_blend_equation) square_color_mask = square_alpha_mask = 0;
    for (u32 k = color_count; k < 8; ++k) color_formats[k] = VK_FORMAT_UNDEFINED, blend[k] = {};
    for (u32 k = 0; k < color_count; ++k) {
        if (c.color_blend_enable) blend[k].blend_enable = 0;
        if (c.color_blend_equation || (!c.color_blend_enable && !blend[k].blend_enable))
            blend[k].src_color = blend[k].dst_color = blend[k].color_op =
                blend[k].src_alpha = blend[k].dst_alpha = blend[k].alpha_op = 0;
        if (c.color_write_mask) blend[k].write_mask = 0xF;
    }
    pad[0] = pad[1] = 0;
}

u64 GraphicsPipelineKey::Hash() const {
    const u64 head[3]{vs_key, ps_key, (state.flags & FixedPipelineState::kHasGs) ? gs_mask : 0ull};
    return ::Gpu::PieceHash::Stable64Seeded(&state, sizeof(state), ::Gpu::PieceHash::Stable64(head, sizeof(head)));
}

bool Renderer::Impl::ValidateNativePipelineCacheData(std::span<const u8> data) const {
    struct VulkanPipelineCacheHeader {
        u32 header_size, header_version, vendor_id, device_id;
        u8 uuid[VK_UUID_SIZE];
    };
    static_assert(sizeof(VulkanPipelineCacheHeader) == 32);
    if (data.size() < sizeof(VulkanPipelineCacheHeader)) return false;
    VulkanPipelineCacheHeader h{};
    std::memcpy(&h, data.data(), sizeof(h));
    if (h.header_size != sizeof(VulkanPipelineCacheHeader) || h.header_version != VK_PIPELINE_CACHE_HEADER_VERSION_ONE) return false;
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pd, &props);
    return h.vendor_id == props.vendorID && h.device_id == props.deviceID && std::memcmp(h.uuid, props.pipelineCacheUUID, VK_UUID_SIZE) == 0;
}

void Renderer::Impl::CreatePipelineCache(std::span<const u8> initial_data) {
    if (pipeline_cache) vkDestroyPipelineCache(dev, pipeline_cache, nullptr), pipeline_cache = VK_NULL_HANDLE;
    const bool valid = !initial_data.empty() && ValidateNativePipelineCacheData(initial_data);
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, nullptr, 0, valid ? initial_data.size() : 0,
                                 valid ? initial_data.data() : nullptr};
    VkResult e = vkCreatePipelineCache(dev, &ci, nullptr, &pipeline_cache);
    if (e != VK_SUCCESS && valid) {
        ci.initialDataSize = 0, ci.pInitialData = nullptr;
        e = vkCreatePipelineCache(dev, &ci, nullptr, &pipeline_cache);
    }
    if (e != VK_SUCCESS) Report(std::format("vkCreatePipelineCache {}", (int)e)), pipeline_cache = VK_NULL_HANDLE;
}

constexpr u64 kPipelineCacheMax = 1024ull << 20;  // the largest cache a load takes (a corrupt header's size is not allocated)

void Renderer::Impl::LoadPipelineCache() {
    if (caps.path == ShaderPath::Objects) return;
    ShaderCache& sc = ShaderCache::Instance();
    if (!sc.Enabled()) return CreatePipelineCache({});
    const auto path = sc.Directory() / (PieceHash::UsesAes() ? "pipeline-cache-hash2-aes.bin" : "pipeline-cache-hash2-xxh.bin");
    struct Header {
        u32 magic, format;
        u64 binary_seed, raw_size, packed_size, raw_hash;
    };
    static_assert(sizeof(Header) == 40);
    constexpr u32 kMagic = 0x43503347, kFormat = 1;  // "G3PC"
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return CreatePipelineCache({});
    const auto total = in.tellg();
    if (total < std::streamoff(sizeof(Header))) return CreatePipelineCache({});
    in.seekg(0);
    Header hdr{};
    if (!in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr))) return CreatePipelineCache({});
    if (hdr.magic != kMagic || hdr.format != kFormat || hdr.binary_seed != binary_seed || hdr.raw_size == 0 ||
        hdr.raw_size > kPipelineCacheMax || hdr.packed_size == 0 || u64(total) != sizeof(Header) + hdr.packed_size)
        return CreatePipelineCache({});
    std::vector<u8> packed(size_t(hdr.packed_size));
    if (!in.read(reinterpret_cast<char*>(packed.data()), std::streamsize(packed.size()))) return CreatePipelineCache({});
    std::vector<u8> raw(size_t(hdr.raw_size));
    const size_t got = ZSTD_decompress(raw.data(), raw.size(), packed.data(), packed.size());
    if (ZSTD_isError(got) || got != hdr.raw_size) return CreatePipelineCache({});
    if (PieceHash::Stable64(raw.data(), raw.size()) != hdr.raw_hash || !ValidateNativePipelineCacheData(raw))
        return CreatePipelineCache({});
    CreatePipelineCache(raw);
    pipeline_cache_bytes = raw.size();
    Gow3::Log(std::format("Pipeline cache: loaded {:.1f} MB ({:.1f} MB compressed)", raw.size() / 1048576.0, packed.size() / 1048576.0));
}

// A save serializes, hashes and compresses the whole cache while the driver holds it (its compiles wait): only at the exit
// and at the end of the startup AOT, never in play. A crash loses the session's pipelines, which the pipeline recipes rebuild.
void Renderer::Impl::SavePipelineCache() {
    ShaderCache& sc = ShaderCache::Instance();
    if (caps.path == ShaderPath::Objects || !pipeline_cache || !sc.Enabled()) return;
    std::scoped_lock l(pipeline_cache_save_lock);
    const u32 before = pipeline_cache_dirty.load(std::memory_order_acquire);
    if (before == 0) return;
    // One serialization when the last size is known (a guess with room), two otherwise or once it outgrew the guess.
    size_t size = pipeline_cache_bytes ? pipeline_cache_bytes + pipeline_cache_bytes / 8 + (1u << 20) : 0;
    std::unique_ptr<u8[]> raw;
    VkResult r = VK_INCOMPLETE;
    if (size) raw = std::make_unique_for_overwrite<u8[]>(size), r = vkGetPipelineCacheData(dev, pipeline_cache, &size, raw.get());
    if (r == VK_INCOMPLETE) {
        if (vkGetPipelineCacheData(dev, pipeline_cache, &size, nullptr) != VK_SUCCESS || !size) return;
        raw = std::make_unique_for_overwrite<u8[]>(size);
        r = vkGetPipelineCacheData(dev, pipeline_cache, &size, raw.get());
    }
    if (r != VK_SUCCESS || !size) return;
    if (size > kPipelineCacheMax)  // larger than a load takes: the file that loads stays
        return Gow3::Log(std::format("Pipeline cache: {:.1f} MB is past the {} MB limit, not saved", size / 1048576.0, kPipelineCacheMax >> 20));
    const size_t bound = ZSTD_compressBound(size);
    const auto packed = std::make_unique_for_overwrite<u8[]>(bound);
    const size_t packed_size = ZSTD_compress(packed.get(), bound, raw.get(), size, 3);
    if (ZSTD_isError(packed_size) || !packed_size) return;
    struct Header {
        u32 magic, format;
        u64 binary_seed, raw_size, packed_size, raw_hash;
    };
    static_assert(sizeof(Header) == 40);
    constexpr u32 kMagic = 0x43503347, kFormat = 1;  // "G3PC"
    const Header hdr{kMagic, kFormat, binary_seed, u64(size), u64(packed_size), PieceHash::Stable64(raw.get(), size)};
    const auto path = sc.Directory() / (PieceHash::UsesAes() ? "pipeline-cache-hash2-aes.bin" : "pipeline-cache-hash2-xxh.bin");
    auto tmp = path;
    tmp += ".tmp";
    std::error_code ec;
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        out.write(reinterpret_cast<const char*>(packed.get()), std::streamsize(packed_size));
        out.flush();
        if (!out.good()) {
            out.close();
            std::filesystem::remove(tmp, ec);
            return;
        }
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return;
    }
    pipeline_cache_bytes = size;
    pipeline_cache_dirty.fetch_sub(before, std::memory_order_acq_rel);
    Gow3::Log(std::format("Pipeline cache: saved {:.1f} MB ({:.1f} MB compressed)", size / 1048576.0, packed_size / 1048576.0));
}

void Renderer::Impl::DestroyShaderRef(const ShaderRef& r) {
    if (r.obj && DestroyShader) DestroyShader(dev, r.obj, nullptr);
    if (r.mod) vkDestroyShaderModule(dev, r.mod, nullptr);
    if (r.pl) vkDestroyPipeline(dev, r.pl, nullptr);
}

VkShaderModule Renderer::Impl::CreateShaderModule(std::span<const u32> spirv) {
    if (spirv.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, spirv.size_bytes(), spirv.data()};
    VkShaderModule mod = VK_NULL_HANDLE;
    if (const VkResult e = vkCreateShaderModule(dev, &ci, nullptr, &mod); e != VK_SUCCESS) {
        Report(std::format("vkCreateShaderModule {}: {} bytes", (int)e, spirv.size_bytes()));
        mod = VK_NULL_HANDLE;
    }
    return mod;
}

VkPipeline Renderer::Impl::CreateComputePipeline(std::span<const u32> spirv, bool fail_on_compile) {
    if (fail_on_compile && !caps.cache_control) return VK_NULL_HANDLE;
    if (spirv.empty()) return VK_NULL_HANDLE;
    VkShaderModule mod = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo mod_ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, spirv.size_bytes(), spirv.data()};
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, caps.maintenance5 ? &mod_ci : nullptr, 0,
                                          VK_SHADER_STAGE_COMPUTE_BIT, VK_NULL_HANDLE, "main"};
    if (!caps.maintenance5) {
        mod = CreateShaderModule(spirv);
        if (!mod) return VK_NULL_HANDLE;
        stage.module = mod;
    }
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr,
                                   fail_on_compile ? VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT : 0u, stage, layout};
    VkPipeline pl = VK_NULL_HANDLE;
    const VkResult e = vkCreateComputePipelines(dev, pipeline_cache, 1, &ci, nullptr, &pl);
    if (mod) vkDestroyShaderModule(dev, mod, nullptr);
    if (e == VK_SUCCESS) {
        if (!fail_on_compile) pipeline_cache_dirty.fetch_add(1, std::memory_order_relaxed);
        return pl;
    }
    if (e < 0) Report(std::format("vkCreateComputePipelines {}: {} bytes", (int)e, spirv.size_bytes()));
    return VK_NULL_HANDLE;
}

std::vector<VkDynamicState> Renderer::Impl::BuildDynamicStates() const {
    std::vector<VkDynamicState> list{
        VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT,
        VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT,
        VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE,
        VK_DYNAMIC_STATE_CULL_MODE,
        VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
        VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_OP,
        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS,
    };
    if (caps.depth_bounds) {
        list.push_back(VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE);
        list.push_back(VK_DYNAMIC_STATE_DEPTH_BOUNDS);
    }
    if (caps.depth_clip_negative_one_to_one) list.push_back(VK_DYNAMIC_STATE_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE_EXT);
    if (caps.depth_clamp_enable) list.push_back(VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT);
    if (caps.rasterization_samples) list.push_back(VK_DYNAMIC_STATE_RASTERIZATION_SAMPLES_EXT);
    if (caps.alpha_to_coverage_enable) list.push_back(VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT);
    if (caps.color_blend_enable) list.push_back(VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT);
    if (caps.color_blend_equation) list.push_back(VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT);
    if (caps.color_write_mask) list.push_back(VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT);
    return list;
}

VkPipeline Renderer::Impl::CreateGraphicsPipelineRaw(const ShaderRef& vs, const ShaderRef& ps, const ShaderRef& gs,
                                                     const FixedPipelineState& state, bool fail_on_compile, bool square_pass) {
    if (fail_on_compile && !caps.cache_control) return VK_NULL_HANDLE;
    if (!vs.mod) return VK_NULL_HANDLE;
    const bool has_gs = (state.flags & FixedPipelineState::kHasGs) != 0;
    const bool has_ps = (state.flags & FixedPipelineState::kHasPs) != 0;
    if (has_gs && !gs.mod) return VK_NULL_HANDLE;
    if (has_ps && !ps.mod) return VK_NULL_HANDLE;

    VkPipelineShaderStageCreateInfo stages[3]{};
    u32 stage_count = 0;
    stages[stage_count++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs.mod, "main"};
    if (has_gs) stages[stage_count++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_GEOMETRY_BIT, gs.mod, "main"};
    if (has_ps) stages[stage_count++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, ps.mod, "main"};

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    static constexpr VkPrimitiveTopology kTopos[]{VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0,
                                              kTopos[std::min<u8>(state.topology_class, 2)], VK_FALSE};
    VkPipelineViewportDepthClipControlCreateInfoEXT dcc{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, nullptr,
                                                       (state.flags & FixedPipelineState::kNegOne) ? VK_TRUE : VK_FALSE};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                          caps.depth_clip_negative_one_to_one ? nullptr : &dcc};
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.depthClampEnable = (!caps.depth_clamp_enable && (state.flags & FixedPipelineState::kDepthClamp)) ? VK_TRUE : VK_FALSE;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VkSampleCountFlagBits(std::max<u32>(1, state.samples));
    ms.alphaToCoverageEnable = (!caps.alpha_to_coverage_enable && (state.flags & FixedPipelineState::kAlphaToCoverage)) ? VK_TRUE : VK_FALSE;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

    VkPipelineColorBlendAttachmentState cba[8]{};
    const u32 count = std::min<u32>(state.color_count, 8);
    for (u32 k = 0; k < count; ++k) {
        auto& a = cba[k];
        if (square_pass) {
            const bool active = state.color_formats[k] != VK_FORMAT_UNDEFINED && state.blend[k].write_mask != 0;
            const bool sq_c = (state.square_color_mask >> k & 1) != 0;
            const bool sq_a = (state.square_alpha_mask >> k & 1) != 0;
            a.blendEnable = (!caps.color_blend_enable && active) ? VK_TRUE : VK_FALSE;
            a.srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            a.dstColorBlendFactor = sq_c ? VK_BLEND_FACTOR_DST_COLOR : VK_BLEND_FACTOR_ONE;
            a.colorBlendOp = VK_BLEND_OP_ADD;
            a.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            a.dstAlphaBlendFactor = sq_a ? VK_BLEND_FACTOR_DST_ALPHA : VK_BLEND_FACTOR_ONE;
            a.alphaBlendOp = VK_BLEND_OP_ADD;
        } else {
            const auto& b = state.blend[k];
            a.blendEnable = (!caps.color_blend_enable && b.blend_enable) ? VK_TRUE : VK_FALSE;
            a.srcColorBlendFactor = VkBlendFactor(b.src_color);
            a.dstColorBlendFactor = VkBlendFactor(b.dst_color);
            a.colorBlendOp = VkBlendOp(b.color_op);
            a.srcAlphaBlendFactor = VkBlendFactor(b.src_alpha);
            a.dstAlphaBlendFactor = VkBlendFactor(b.dst_alpha);
            a.alphaBlendOp = VkBlendOp(b.alpha_op);
        }
        a.colorWriteMask = caps.color_write_mask ? VkColorComponentFlags(0xF) : VkColorComponentFlags(state.blend[k].write_mask);
    }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY,
                                           count, count ? cba : nullptr};

    const auto dyns = BuildDynamicStates();
    VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, u32(dyns.size()), dyns.data()};
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, count,
                                      count ? state.color_formats : nullptr, state.depth_format, state.stencil_format};
    const VkPipelineCreateFlags flags = fail_on_compile ? VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT : 0;
    VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci, flags, stage_count, stages, &vi, &ia,
                                    nullptr, &vps, &rs, &ms, &ds, &cb, &dsi, layout};
    VkPipeline pl = VK_NULL_HANDLE;
    const VkResult e = vkCreateGraphicsPipelines(dev, pipeline_cache, 1, &gi, nullptr, &pl);
    if (e == VK_SUCCESS) {
        if (!fail_on_compile) pipeline_cache_dirty.fetch_add(1, std::memory_order_relaxed);
        return pl;
    }
    if (e < 0) Report(std::format("vkCreateGraphicsPipelines {}: vs {:x} ps {:x}", (int)e, vs.key, ps.key));
    return VK_NULL_HANDLE;
}

CompiledPipeline Renderer::Impl::CreateMonolithicPipeline(const ShaderRef& vs, const ShaderRef& ps, const ShaderRef& gs,
                                                          const FixedPipelineState& state, bool fail_on_compile) {
    CompiledPipeline pl{};
    pl.main = CreateGraphicsPipelineRaw(vs, ps, gs, state, fail_on_compile, false);
    if (!pl.main) return {};
    if (state.NeedsSquarePipeline(caps)) {
        pl.square = CreateGraphicsPipelineRaw(vs, ps, gs, state, fail_on_compile, true);
        if (!pl.square) {
            vkDestroyPipeline(dev, pl.main, nullptr);
            return {};
        }
    }
    return pl;
}

void Renderer::Impl::SetupGplGlobals() {
    static constexpr VkPipelineCreateFlags kLibFlags =
        VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT;
    static constexpr VkPrimitiveTopology kTopos[3]{
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    static constexpr VkDynamicState kViDyns[2]{
        VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 2, kViDyns};
    VkGraphicsPipelineLibraryCreateInfoEXT lib_ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, nullptr,
                                                  VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT};
    for (u32 k = 0; k < 3; ++k) {
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, kTopos[k], VK_FALSE};
        VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &lib_ci, kLibFlags};
        gi.pVertexInputState = &vi, gi.pInputAssemblyState = &ia, gi.pDynamicState = &dsi;
        if (const VkResult e = vkCreateGraphicsPipelines(dev, pipeline_cache, 1, &gi, nullptr, &gpl_vi[k]); e != VK_SUCCESS)
            Report(std::format("vkCreateGraphicsPipelines GPL VI[{}] {}", k, (int)e)), gpl_vi[k] = VK_NULL_HANDLE;
    }
}

VkPipeline Renderer::Impl::CreateGplPreRaster(VkShaderModule vs_mod, VkShaderModule gs_mod, bool neg_one, bool depth_clamp) {
    if (!vs_mod) return VK_NULL_HANDLE;
    VkPipelineShaderStageCreateInfo stages[2]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs_mod, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_GEOMETRY_BIT, gs_mod, "main"},
    };
    VkPipelineViewportDepthClipControlCreateInfoEXT dcc{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, nullptr,
                                                       neg_one ? VK_TRUE : VK_FALSE};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                          caps.depth_clip_negative_one_to_one ? nullptr : &dcc};
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.depthClampEnable = (!caps.depth_clamp_enable && depth_clamp) ? VK_TRUE : VK_FALSE;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.lineWidth = 1.0f;

    std::vector<VkDynamicState> dyns{
        VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT,
        VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT,
        VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE,
        VK_DYNAMIC_STATE_CULL_MODE,
        VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
    };
    if (caps.depth_clip_negative_one_to_one) dyns.push_back(VK_DYNAMIC_STATE_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE_EXT);
    if (caps.depth_clamp_enable) dyns.push_back(VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT);
    VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, u32(dyns.size()), dyns.data()};
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkGraphicsPipelineLibraryCreateInfoEXT lib_ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, &rci,
                                                  VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT};
    VkGraphicsPipelineCreateInfo gi{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &lib_ci,
        VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT,
        gs_mod ? 2u : 1u, stages};
    gi.pViewportState = &vps, gi.pRasterizationState = &rs, gi.pDynamicState = &dsi, gi.layout = layout;
    VkPipeline pl = VK_NULL_HANDLE;
    if (const VkResult e = vkCreateGraphicsPipelines(dev, pipeline_cache, 1, &gi, nullptr, &pl); e != VK_SUCCESS) {
        Report(std::format("vkCreateGraphicsPipelines GPL PreRaster {}", (int)e));
        return VK_NULL_HANDLE;
    }
    return pl;
}

VkPipeline Renderer::Impl::CreateGplFragShader(VkShaderModule ps_mod, u32 samples, bool alpha_to_coverage) {
    if (!ps_mod) return VK_NULL_HANDLE;
    VkPipelineShaderStageCreateInfo st{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, ps_mod, "main"};
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VkSampleCountFlagBits(std::max<u32>(1, samples));
    ms.alphaToCoverageEnable = (!caps.alpha_to_coverage_enable && alpha_to_coverage) ? VK_TRUE : VK_FALSE;

    std::vector<VkDynamicState> dyns{
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_OP,
        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    };
    if (caps.depth_bounds) {
        dyns.push_back(VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE);
        dyns.push_back(VK_DYNAMIC_STATE_DEPTH_BOUNDS);
    }
    if (caps.rasterization_samples) dyns.push_back(VK_DYNAMIC_STATE_RASTERIZATION_SAMPLES_EXT);
    VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, u32(dyns.size()), dyns.data()};
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkGraphicsPipelineLibraryCreateInfoEXT lib_ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, &rci,
                                                  VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT};
    VkGraphicsPipelineCreateInfo gi{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &lib_ci,
        VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT,
        1u, &st};
    gi.pMultisampleState = &ms, gi.pDepthStencilState = &ds, gi.pDynamicState = &dsi, gi.layout = layout;
    VkPipeline pl = VK_NULL_HANDLE;
    if (const VkResult e = vkCreateGraphicsPipelines(dev, pipeline_cache, 1, &gi, nullptr, &pl); e != VK_SUCCESS) {
        Report(std::format("vkCreateGraphicsPipelines GPL FragShader {}", (int)e));
        return VK_NULL_HANDLE;
    }
    return pl;
}

VkPipeline Renderer::Impl::CreateGplFragOutput(const FixedPipelineState& state, bool square_pass) {
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VkSampleCountFlagBits(std::max<u32>(1, state.samples));
    ms.alphaToCoverageEnable = (!caps.alpha_to_coverage_enable && (state.flags & FixedPipelineState::kAlphaToCoverage)) ? VK_TRUE : VK_FALSE;

    VkPipelineColorBlendAttachmentState cba[8]{};
    const u32 count = std::min<u32>(state.color_count, 8);
    for (u32 k = 0; k < count; ++k) {
        auto& a = cba[k];
        if (square_pass) {
            const bool active = state.color_formats[k] != VK_FORMAT_UNDEFINED && state.blend[k].write_mask != 0;
            const bool sq_c = (state.square_color_mask >> k & 1) != 0;
            const bool sq_a = (state.square_alpha_mask >> k & 1) != 0;
            a.blendEnable = (!caps.color_blend_enable && active) ? VK_TRUE : VK_FALSE;
            a.srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            a.dstColorBlendFactor = sq_c ? VK_BLEND_FACTOR_DST_COLOR : VK_BLEND_FACTOR_ONE;
            a.colorBlendOp = VK_BLEND_OP_ADD;
            a.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            a.dstAlphaBlendFactor = sq_a ? VK_BLEND_FACTOR_DST_ALPHA : VK_BLEND_FACTOR_ONE;
            a.alphaBlendOp = VK_BLEND_OP_ADD;
        } else {
            const auto& b = state.blend[k];
            a.blendEnable = (!caps.color_blend_enable && b.blend_enable) ? VK_TRUE : VK_FALSE;
            a.srcColorBlendFactor = VkBlendFactor(b.src_color);
            a.dstColorBlendFactor = VkBlendFactor(b.dst_color);
            a.colorBlendOp = VkBlendOp(b.color_op);
            a.srcAlphaBlendFactor = VkBlendFactor(b.src_alpha);
            a.dstAlphaBlendFactor = VkBlendFactor(b.dst_alpha);
            a.alphaBlendOp = VkBlendOp(b.alpha_op);
        }
        a.colorWriteMask = caps.color_write_mask ? VkColorComponentFlags(0xF) : VkColorComponentFlags(state.blend[k].write_mask);
    }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY,
                                           count, count ? cba : nullptr};

    std::vector<VkDynamicState> dyns{VK_DYNAMIC_STATE_BLEND_CONSTANTS};
    if (caps.rasterization_samples) dyns.push_back(VK_DYNAMIC_STATE_RASTERIZATION_SAMPLES_EXT);
    if (caps.alpha_to_coverage_enable) dyns.push_back(VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT);
    if (caps.color_blend_enable) dyns.push_back(VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT);
    if (caps.color_blend_equation) dyns.push_back(VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT);
    if (caps.color_write_mask) dyns.push_back(VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT);
    VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, u32(dyns.size()), dyns.data()};
    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, count,
                                      count ? state.color_formats : nullptr, state.depth_format, state.stencil_format};
    VkGraphicsPipelineLibraryCreateInfoEXT lib_ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, &rci,
                                                  VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT};
    VkGraphicsPipelineCreateInfo gi{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &lib_ci,
        VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT};
    gi.pMultisampleState = &ms, gi.pColorBlendState = &cb, gi.pDynamicState = &dsi;
    VkPipeline pl = VK_NULL_HANDLE;
    if (const VkResult e = vkCreateGraphicsPipelines(dev, pipeline_cache, 1, &gi, nullptr, &pl); e != VK_SUCCESS) {
        Report(std::format("vkCreateGraphicsPipelines GPL FragOutput {}", (int)e));
        return VK_NULL_HANDLE;
    }
    return pl;
}

VkPipeline Renderer::Impl::GetGplPreRaster(const ShaderRef& vs, const ShaderRef& gs, const FixedPipelineState& state) {
    const bool neg_one = !caps.depth_clip_negative_one_to_one && (state.flags & FixedPipelineState::kNegOne);
    const bool depth_clamp = !caps.depth_clamp_enable && (state.flags & FixedPipelineState::kDepthClamp);
    const bool has_gs = (state.flags & FixedPipelineState::kHasGs) != 0;
    if (!has_gs && neg_one == !caps.depth_clip_negative_one_to_one && !depth_clamp && vs.pl) return vs.pl;
    if (!vs.mod || (has_gs && !gs.mod)) return VK_NULL_HANDLE;
    struct PreKey {
        u64 vs, gs;
        u32 flags, pad;
    } pk{vs.key, has_gs ? gs.key : 0ull, u32(neg_one) | u32(depth_clamp) << 1 | u32(has_gs) << 2, 0};
    const u64 h = ::Gpu::PieceHash::Stable64(&pk, sizeof(pk));
    {
        std::shared_lock lock(prog_lock);
        if (auto it = gpl_prerast.find(h); it != gpl_prerast.end()) return it->second;
    }
    const VkPipeline created = CreateGplPreRaster(vs.mod, has_gs ? gs.mod : VK_NULL_HANDLE, neg_one, depth_clamp);
    if (!created) return VK_NULL_HANDLE;
    std::unique_lock lock(prog_lock);
    auto [it, inserted] = gpl_prerast.try_emplace(h, created);
    if (!inserted) vkDestroyPipeline(dev, created, nullptr);
    return it->second;
}

VkPipeline Renderer::Impl::GetGplFragShader(const ShaderRef& ps, const FixedPipelineState& state) {
    if (!(state.flags & FixedPipelineState::kHasPs) || !ps.mod) return VK_NULL_HANDLE;
    const u32 samples = caps.rasterization_samples ? 1u : std::max<u32>(1, state.samples);
    const bool a2c = !caps.alpha_to_coverage_enable && (state.flags & FixedPipelineState::kAlphaToCoverage) != 0;
    if (samples == 1 && !a2c && ps.pl) return ps.pl;
    struct FragKey {
        u64 ps;
        u32 samples, a2c;
    } fk{ps.key, samples, a2c ? 1u : 0u};
    const u64 h = ::Gpu::PieceHash::Stable64(&fk, sizeof(fk));
    {
        std::shared_lock lock(prog_lock);
        if (auto it = gpl_frag_msaa.find(h); it != gpl_frag_msaa.end()) return it->second;
    }
    const VkPipeline created = CreateGplFragShader(ps.mod, samples, a2c);
    if (!created) return VK_NULL_HANDLE;
    std::unique_lock lock(prog_lock);
    auto [it, inserted] = gpl_frag_msaa.try_emplace(h, created);
    if (!inserted) vkDestroyPipeline(dev, created, nullptr);
    return it->second;
}

VkPipeline Renderer::Impl::GetGplFragOutput(const FixedPipelineState& state, bool square_pass) {
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
    std::copy_n(state.color_formats, 8, fk.color_formats);
    fk.depth_format = state.depth_format;
    fk.stencil_format = state.stencil_format;
    fk.color_count = state.color_count;
    fk.samples = caps.rasterization_samples ? 1 : std::max<u8>(1, state.samples);
    fk.a2c = (!caps.alpha_to_coverage_enable && (state.flags & FixedPipelineState::kAlphaToCoverage)) ? 1 : 0;
    fk.square_pass = square_pass ? 1 : 0;
    fk.square_color_mask = square_pass ? state.square_color_mask : 0;
    fk.square_alpha_mask = square_pass ? state.square_alpha_mask : 0;
    std::copy_n(state.blend, 8, fk.blend);
    const u64 h = ::Gpu::PieceHash::Stable64(&fk, sizeof(fk));
    {
        std::shared_lock lock(prog_lock);
        if (auto it = gpl_foi.find(h); it != gpl_foi.end()) return it->second;
    }
    const VkPipeline created = CreateGplFragOutput(state, square_pass);
    if (!created) return VK_NULL_HANDLE;
    std::unique_lock lock(prog_lock);
    auto [it, inserted] = gpl_foi.try_emplace(h, created);
    if (!inserted) vkDestroyPipeline(dev, created, nullptr);
    return it->second;
}

VkPipeline Renderer::Impl::LinkGplPipeline(const VkPipeline (&libs)[4], bool lto, bool fail_on_compile) {
    if (fail_on_compile && !caps.cache_control) return VK_NULL_HANDLE;
    VkPipelineLibraryCreateInfoKHR lib_ci{VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, nullptr, 4, libs};
    VkPipelineCreateFlags flags = 0;
    if (lto) flags |= VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT;
    if (fail_on_compile) flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
    VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &lib_ci, flags};
    gi.layout = layout;
    VkPipeline pl = VK_NULL_HANDLE;
    const VkResult e = vkCreateGraphicsPipelines(dev, pipeline_cache, 1, &gi, nullptr, &pl);
    if (e == VK_SUCCESS) {
        if (lto && !fail_on_compile) pipeline_cache_dirty.fetch_add(1, std::memory_order_relaxed);
        return pl;
    }
    if (e < 0) Report(std::format("vkCreateGraphicsPipelines GPL Link(lto={}) {}", lto, (int)e));
    return VK_NULL_HANDLE;
}

CompiledPipeline Renderer::Impl::LinkGplPair(const VkPipeline (&main_libs)[4], VkPipeline foi_square, bool lto, bool fail_on_compile) {
    CompiledPipeline pl{};
    pl.main = LinkGplPipeline(main_libs, lto, fail_on_compile);
    if (!pl.main) return {};
    if (foi_square) {
        const VkPipeline sq_libs[4]{main_libs[0], main_libs[1], main_libs[2], foi_square};
        pl.square = LinkGplPipeline(sq_libs, lto, fail_on_compile);
        if (!pl.square) {
            vkDestroyPipeline(dev, pl.main, nullptr);
            return {};
        }
    }
    return pl;
}

CompiledPipeline Renderer::Impl::GetGraphicsPipeline(const ShaderRef& vs, const ShaderRef& ps, const ShaderRef& gs,
                                                     const FixedPipelineState& raw_state, const CpuRead& cpu) {
    // The thread's memo goes by the draw's own state; the first draw of one on this thread records it for the next launches'
    // AOT (whether or not its pipeline exists already) and finds the pipeline by the state with the dynamic parts zeroed.
    GraphicsPipelineKey gkey{vs.key, ps ? ps.key : 0ull, (raw_state.flags & FixedPipelineState::kHasGs) ? gs.key : 0ull, raw_state};
    const u64 rh = gkey.Hash();
    if (temp_pipelog) TempPlUse(gkey, rh);
    auto& pm = Memos().pipelines;
    if (const auto* m = pm.Find(rh)) return *m;
    if (!stats.trace && !boom_on) Recipes::Instance().NotePipeline(gkey.vs_key, gkey.ps_key, gkey.gs_mask, raw_state);
    gkey.state.Canonicalize(caps);
    const u64 h = gkey.Hash();
    {
        std::shared_lock lock(prog_lock);
        if (auto it = pipelines.find(h); it != pipelines.end()) return pm[rh] = it->second;
    }
    if (caps.path == ShaderPath::Gpl && (gkey.state.flags & FixedPipelineState::kHasPs)) {
        const VkPipeline vi = gpl_vi[std::min<u8>(gkey.state.topology_class, 2)];
        const VkPipeline prerast = GetGplPreRaster(vs, gs, gkey.state);
        const VkPipeline fs = GetGplFragShader(ps, gkey.state);
        const VkPipeline foi = GetGplFragOutput(gkey.state, false);
        const bool need_sq = gkey.state.NeedsSquarePipeline(caps);
        const VkPipeline foi_sq = need_sq ? GetGplFragOutput(gkey.state, true) : VK_NULL_HANDLE;
        if (vi && prerast && fs && foi && (!need_sq || foi_sq)) {
            const VkPipeline libs[4]{vi, prerast, fs, foi};
            CompiledPipeline pl = LinkGplPair(libs, foi_sq, true, true);
            const PlOrigin origin = pl.main ? kPlGplLto : kPlGplLink;
            if (!pl.main) pl = LinkGplPair(libs, foi_sq, false, false);
            if (pl.main) return pm[rh] = KeepPipeline(h, pl, origin);
        }
    }
    // A pipeline in the pipeline cache (built by the AOT or an earlier run) still costs the driver a load. Async: the load
    // runs on a worker ahead of the compiles, and the draw waits a frame or two instead of the recording; Sync: here.
    const bool probe_here = Jit::mode != Jit::Mode::Async;
    if (!stats.trace && caps.cache_control && probe_here)
        if (const CompiledPipeline hit = CreateMonolithicPipeline(vs, ps, gs, gkey.state, true); hit.main) return pm[rh] = KeepPipeline(h, hit, kPlHit);
    if (stats.trace) {
        const CompiledPipeline created = CreateMonolithicPipeline(vs, ps, gs, gkey.state, false);
        return created.main ? pm[rh] = KeepPipeline(h, created, kPlTrace) : CompiledPipeline{};
    }
    if (!in_worker && !discovered) discovered = true, Discover(cpu, {walk_chain.size(), 0}, false);
    else if (in_worker && block_now && !block_discovered) block_discovered = true, Discover(cpu, block_now->to, true);
    std::shared_future<CompiledPipeline> f;
    {
        std::unique_lock lock(prog_lock);
        if (auto it = pipelines.find(h); it != pipelines.end()) return pm[rh] = it->second;
        auto [it, fresh] = compiling_pipelines.try_emplace(h);
        if (fresh) it->second = AsyncPipeline(h, vs, ps, gs, gkey.state, !probe_here, true);
        f = it->second;
    }
    const bool ready = f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    if (!ready && temp_pipelog) TempPlWait(h);
    if (Jit::mode == Jit::Mode::Async && !ready) return waiting = true, CompiledPipeline{};
    CompiledPipeline pl{};
    Stalled(stall_waits, stall_wait_ns, [&] { pl = f.get(); });
    if (!pl.main) return {};
    return pm[rh] = pl;
}

CompiledPipeline Renderer::Impl::KeepPipeline(u64 h, CompiledPipeline pl, PlOrigin origin, u32 us) {
    std::unique_lock lock(prog_lock);
    auto [it, inserted] = pipelines.try_emplace(h, pl);
    if (!inserted) {
        vkDestroyPipeline(dev, pl.main, nullptr);
        if (pl.square) vkDestroyPipeline(dev, pl.square, nullptr);
    } else if (temp_pipelog) {
        TempPlOrigin(h, origin, us);
    }
    return it->second;
}

void Renderer::Impl::TempPlUse(const GraphicsPipelineKey& k, u64 rh) {
    TempPlLocal& l = temp_pl_local[rh];
    const u32 now = frame;
    if (!l.s) {
        const ShaderCache& sc = ShaderCache::Instance();
        const u32 flags = u32(sc.Contains(k.vs_key)) | u32(k.ps_key && sc.Contains(k.ps_key)) << 1;
        GraphicsPipelineKey canon = k;
        canon.state.Canonicalize(caps);
        std::scoped_lock lock(temp_pl_lock);
        TempPlState*& s = temp_pl_by_raw[rh];
        if (!s) s = &temp_pl_states.emplace_back(TempPlState{k.vs_key, k.ps_key, k.gs_mask, canon.Hash(), k.state, now, 0, ~0u, flags, 0});
        l = {s, ~0u, 0};
    }
    if (l.frame == now) return void(++l.draws);
    std::atomic_ref(l.s->draws).fetch_add(l.draws + 1, std::memory_order_relaxed);
    if (std::atomic_ref(l.s->last_frame).exchange(now, std::memory_order_relaxed) != now) std::atomic_ref(l.s->frames).fetch_add(1, std::memory_order_relaxed);
    l.frame = now, l.draws = 0;
}

void Renderer::Impl::TempPlOrigin(u64 h, PlOrigin origin, u32 us) {
    std::scoped_lock lock(temp_pl_lock);
    TempPlMade& m = temp_pl_made[h];
    m.origin = origin, m.us = us, m.made_frame = frame;
}

void Renderer::Impl::TempPlWait(u64 h) {
    std::scoped_lock lock(temp_pl_lock);
    if (TempPlMade& m = temp_pl_made[h]; !m.waits++) m.wait_frame = frame;
}

// Header, the draw states, then each made pipeline as its canonical key and TempPlMade; a summary in gow3.log.
void Renderer::Impl::TempPlWrite() {
    if (!temp_pipelog) return;
    std::scoped_lock lock(temp_pl_lock);
    if (temp_pl_states.empty()) return;
    static_assert(sizeof(TempPlState) == 168 && sizeof(TempPlMade) == 24);
    const u32 capbits = u32(caps.depth_clip_negative_one_to_one) | u32(caps.depth_clamp_enable) << 1 | u32(caps.rasterization_samples) << 2 |
                        u32(caps.alpha_to_coverage_enable) << 3 | u32(caps.color_blend_enable) << 4 | u32(caps.color_blend_equation) << 5 |
                        u32(caps.color_write_mask) << 6 | u32(caps.unrestricted_topology) << 7;
    // Format 2: the pipeline recipes as this GPU draws them follow (raw keys).
    const auto recipes = Recipes::Instance().Pipelines(!caps.storage_image_read_without_format, blend_interlock);
    const u32 head[8]{0x4C503347, 2, u32(caps.path), capbits, u32(temp_pl_states.size()), u32(temp_pl_made.size()), u32(recipes.size()), 0};  // "G3PL"
    std::vector<u8> out;
    const auto append = [&](const void* p, size_t n) { out.insert(out.end(), (const u8*)p, (const u8*)p + n); };
    append(head, sizeof(head));
    std::unordered_map<u64, u64> draws;  // canonical key -> draws
    u64 total = 0;
    for (const TempPlState& s : temp_pl_states) append(&s, sizeof(s)), draws[s.canon] += s.draws, total += s.draws;
    static constexpr const char* names[]{"unknown", "warm", "hit", "ahead-hit", "ahead", "miss", "gpl-lto", "gpl-link", "trace"};
    u32 made[9]{}, used[9]{}, waited = 0;
    u64 us[9]{}, waits = 0;
    for (const auto& [h, m] : temp_pl_made) {
        append(&h, 8), append(&m, sizeof(m));
        const u32 o = std::min<u32>(m.origin, 8);
        ++made[o], used[o] += draws.contains(h), us[o] += m.us, waited += m.waits != 0, waits += m.waits;
    }
    if (!recipes.empty()) append(recipes.data(), recipes.size() * sizeof(recipes[0]));
    std::string line = std::format("TEMP pipelines: {} draw states over {} pipelines in {} draws; made (used):", temp_pl_states.size(), draws.size(), total);
    for (u32 o = 0; o < 9; ++o)
        if (made[o]) line += std::format(" {} {} ({}{})", names[o], made[o], used[o], us[o] ? std::format(", {:.0f} ms", us[o] / 1e3) : "");
    std::tm tm{};
    Gow3::LocalTime(tm, std::time(nullptr));
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tm);
    const auto path = Gow3::PortDirectory() / std::format("pipelines-{}.bin", stamp);
    std::ofstream(path, std::ios::binary).write((const char*)out.data(), std::streamsize(out.size()));
    Gow3::Log(line + std::format("; {} draws waited for {} pipelines; {}", waits, waited, path.filename().string()));
}

void Renderer::Impl::WarmPipelines() {
    if (caps.path == ShaderPath::Objects || !caps.cache_control) return;
    std::call_once(warm_once, [&] {  // the recipes as this GPU draws them (those this run meets are made by then)
        warm_recipes = Recipes::Instance().Pipelines(!caps.storage_image_read_without_format, blend_interlock);
        for (GraphicsPipelineKey& k : warm_recipes) k.state.Canonicalize(caps);
    });
    for (const GraphicsPipelineKey& gk : warm_recipes) {
        const u64 h = gk.Hash();
        ShaderRef vs{}, ps{}, gs{};
        {
            std::shared_lock lock(prog_lock);
            if (pipelines.contains(h) || compiling_pipelines.contains(h)) continue;
            if (gk.vs_key == kClearRectVsKey) {
                vs = clear_rect_ref;
            } else if (auto it = shaders.find(gk.vs_key); it != shaders.end()) {
                vs = it->second;
            }
            if (!vs.mod) continue;
            if (gk.ps_key) {
                auto it = shaders.find(gk.ps_key);
                if (it == shaders.end() || !it->second.mod) continue;
                ps = it->second;
            }
            if (gk.state.flags & FixedPipelineState::kHasGs) {
                auto it = rect_gs.find(gk.gs_mask);
                if (it == rect_gs.end() || !it->second.mod) continue;
                gs = it->second;
            }
        }
        CompiledPipeline pl{};
        if (caps.path == ShaderPath::Gpl && (gk.state.flags & FixedPipelineState::kHasPs)) {
            const VkPipeline vi = gpl_vi[std::min<u8>(gk.state.topology_class, 2)];
            const VkPipeline prerast = GetGplPreRaster(vs, gs, gk.state);
            const VkPipeline fs = GetGplFragShader(ps, gk.state);
            const VkPipeline foi = GetGplFragOutput(gk.state, false);
            const bool need_sq = gk.state.NeedsSquarePipeline(caps);
            const VkPipeline foi_sq = need_sq ? GetGplFragOutput(gk.state, true) : VK_NULL_HANDLE;
            if (vi && prerast && fs && foi && (!need_sq || foi_sq)) {
                const VkPipeline libs[4]{vi, prerast, fs, foi};
                pl = LinkGplPair(libs, foi_sq, true, true);
            }
        } else {
            pl = CreateMonolithicPipeline(vs, ps, gs, gk.state, true);
        }
        if (pl.main) KeepPipeline(h, pl, kPlWarm);
    }
}

VkPipeline Renderer::Impl::IndexProbePipeline(const FixedPipelineState& raw_state) {
    if (!index_probe_mod) return VK_NULL_HANDLE;
    FixedPipelineState s = raw_state;
    s.flags &= ~(FixedPipelineState::kHasPs | FixedPipelineState::kHasGs);
    s.square_color_mask = s.square_alpha_mask = 0;
    s.Canonicalize(caps);
    const u64 h = GraphicsPipelineKey{0, 0, 0, s}.Hash();
    {
        std::shared_lock lock(prog_lock);
        if (auto it = index_probe_pls.find(h); it != index_probe_pls.end()) return it->second;
    }
    const VkPipeline pl = CreateGraphicsPipelineRaw({0, VK_NULL_HANDLE, index_probe_mod, VK_NULL_HANDLE}, {}, {}, s, false, false);
    if (!pl) return VK_NULL_HANDLE;
    std::unique_lock lock(prog_lock);
    auto [it, inserted] = index_probe_pls.try_emplace(h, pl);
    if (!inserted) vkDestroyPipeline(dev, pl, nullptr);
    return it->second;
}
} // namespace Gpu
