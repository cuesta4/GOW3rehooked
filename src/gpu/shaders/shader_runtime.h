// SPDX-License-Identifier: GPL-2.0-or-later
// Interface of the shader runtime the renderer consumes: what a program is made of once identified and translated (slots,
// memoized code, compiled shader, last variant). Deciding between a ready shader (AOT) and a translation now (JIT) is
// shader_runtime.cpp's.
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <iterator>
#include <span>
#include <string>
#include <type_traits>
#include <vector>
#include "gpu/device.h"
#include "gpu/shaders/gcn_translator.h"

namespace Gpu {
using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u64 kClearRectVsKey = 0x474F5733434C5231ull;

// Unified handle returned by Shader()/Translated()/RectGs() across all 3 ShaderPath tiers:
// - Objects: `obj` is VkShaderEXT.
// - Gpl/Monolithic: `mod` is VkShaderModule (VS/PS/GS), `pl` is Compute VkPipeline (CS).
struct ShaderRef {
    u64 key = 0;
    VkShaderEXT obj = VK_NULL_HANDLE;
    VkShaderModule mod = VK_NULL_HANDLE;
    VkPipeline pl = VK_NULL_HANDLE;
    explicit operator bool() const { return obj != VK_NULL_HANDLE || mod != VK_NULL_HANDLE || pl != VK_NULL_HANDLE; }
    bool operator==(const ShaderRef&) const = default;
};

struct ProgSlots {
    std::vector<Gcn::Slot> slots;
    std::vector<Gcn::Path> buffers;  // V# locations (tracking, trace)
    std::vector<Gcn::Path> formats;  // V# of format loads (key)
    std::vector<Gcn::Const> consts;  // scalar load constants of the draw table
    std::uint64_t attrs = 0;         // Analysis::attrs
    std::uint64_t stored = 0;        // Analysis::stored
    int depth_resolve_slot = -1;
};
struct ProgMemo {
    u32 frame = ~0u;
    std::span<const u32> code;
    u64 hash = 0;
    u64 orb_hash = 0;
    int fetch_sgpr = -1;
    u64 ver = 0;  // Version of the code's pages when it was read (0: read it again every frame)
};

inline constexpr u64 kMotionBlurPassOrbHash = 0x107d87f9d56574cfULL;
inline constexpr u64 kMotionBlurCompositeOrbHash = 0xabd418f2b4587d5dULL;
inline constexpr u64 kDofCompositeOrbHash = 0x1b4f84d1cc5f5f9eULL;
inline constexpr u64 kLinearCopyCsOrbHash = 0xccdebd80f02a77aaULL;
struct Compiled {
    VkShaderEXT s = VK_NULL_HANDLE;        // Tier 1 (Objects)
    VkShaderModule mod = VK_NULL_HANDLE;   // Tiers 2/3 graphics stage
    VkPipeline pl = VK_NULL_HANDLE;        // Tiers 2/3 compute pipeline
    u64 out_mask = 0;                      // VS: exported parameters
    std::string err;
    bool Ready(ShaderPath path, Gcn::Stage stage) const {
        if (path == ShaderPath::Objects) return s != VK_NULL_HANDLE;
        if (stage == Gcn::Stage::Compute) return pl != VK_NULL_HANDLE;
        return mod != VK_NULL_HANDLE;
    }
    ShaderRef Ref(u64 key) const { return {key, s, mod, pl}; }
};

inline VkBlendFactor BlendFactor(u32 f) {
    static constexpr VkBlendFactor t[]{VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_SRC_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                                       VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_DST_ALPHA,
                                       VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, VK_BLEND_FACTOR_DST_COLOR, VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
                                       VK_BLEND_FACTOR_SRC_ALPHA_SATURATE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_CONSTANT_COLOR,
                                       VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR, VK_BLEND_FACTOR_SRC1_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
                                       VK_BLEND_FACTOR_SRC1_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA, VK_BLEND_FACTOR_CONSTANT_ALPHA,
                                       VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA};
    return f < std::size(t) ? t[f] : VK_BLEND_FACTOR_ONE;
}
inline VkBlendOp BlendOp(u32 f) {
    static constexpr VkBlendOp t[]{VK_BLEND_OP_ADD, VK_BLEND_OP_SUBTRACT, VK_BLEND_OP_MIN, VK_BLEND_OP_MAX, VK_BLEND_OP_REVERSE_SUBTRACT};
    return f < std::size(t) ? t[f] : VK_BLEND_OP_ADD;
}
// GCN applies the blend factors before MIN/MAX, Vulkan ignores them (shadPS4 RewriteScaledMinMaxBlend). With one zero
// operand the equation is an exact ADD on a UNORM target; with both operands scaled by themselves MIN/MAX(s*s, d*d) =
// MIN/MAX(s, d)^2, kept native and squared by a second draw.
enum class MinMax { Native, Exact, Squared, Unsupported };
inline MinMax RewriteMinMax(u32& src, u32& fn, u32& dst, bool unorm, bool alpha) {
    if ((fn != 2 && fn != 3) || (src == 1 && dst == 1)) return MinMax::Native;  // MIN, MAX; factors ONE
    if (!unorm) return MinMax::Unsupported;
    if ((src == 2 || (alpha && src == 4)) && (dst == 8 || (alpha && dst == 6))) return MinMax::Squared;  // SRC_COLOR/ALPHA, DST_COLOR/ALPHA
    if (src && dst) return MinMax::Unsupported;
    if (fn == 2) src = dst = 0;
    fn = 0;
    return MinMax::Exact;
}

struct FixedAttachmentBlend {
    u8 blend_enable = 0;
    u8 write_mask = 0xF;
    u8 src_color = 0;
    u8 dst_color = 0;
    u8 color_op = 0;
    u8 src_alpha = 0;
    u8 dst_alpha = 0;
    u8 alpha_op = 0;
    bool operator==(const FixedAttachmentBlend&) const = default;
};
static_assert(sizeof(FixedAttachmentBlend) == 8);

// Non-dynamic graphics pipeline state. Raw (uncanonicalized) copies can be persisted as portable recipes;
// Canonicalize(caps) zeroes every field driven dynamically on this GPU to minimize pipeline permutations.
struct FixedPipelineState {
    static constexpr u8 kNegOne = 1u << 0;
    static constexpr u8 kDepthClamp = 1u << 1;
    static constexpr u8 kAlphaToCoverage = 1u << 2;
    static constexpr u8 kHasPs = 1u << 3;
    static constexpr u8 kHasGs = 1u << 4;

    VkFormat color_formats[8]{};
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    VkFormat stencil_format = VK_FORMAT_UNDEFINED;
    u8 color_count = 0;
    u8 samples = 1;            // 1..64
    u8 topology_class = 2;     // 0=point, 1=line, 2=triangle
    u8 flags = 0;              // kNegOne | kDepthClamp | kAlphaToCoverage | kHasPs | kHasGs
    u8 square_color_mask = 0;  // bit k: scaled MIN/MAX squared color channel on attachment k
    u8 square_alpha_mask = 0;  // bit k: scaled MIN/MAX squared alpha channel on attachment k
    u8 pad[2]{};
    FixedAttachmentBlend blend[8]{};

    void Canonicalize(const DynamicCaps& caps);
    bool NeedsSquarePipeline(const DynamicCaps& caps) const {
        return (square_color_mask || square_alpha_mask) && (!caps.color_blend_enable || !caps.color_blend_equation);
    }
    bool operator==(const FixedPipelineState&) const = default;
};
static_assert(std::is_trivially_copyable_v<FixedPipelineState>);
static_assert(sizeof(FixedPipelineState) == 112);

struct GraphicsPipelineKey {
    u64 vs_key = 0;
    u64 ps_key = 0;
    u64 gs_mask = 0;  // valid only when (state.flags & FixedPipelineState::kHasGs)
    FixedPipelineState state{};
    bool operator==(const GraphicsPipelineKey&) const = default;
    u64 Hash() const;
};

struct CompiledPipeline {
    VkPipeline main = VK_NULL_HANDLE;
    VkPipeline square = VK_NULL_HANDLE;  // coupled second-pass pipeline for scaled MIN/MAX squared blend
};
// Key of a program: its code, the fetch shader it calls and the stage state its translation depends on besides the draw's.
u64 ProgramKey(Gcn::Stage stage, u64 code_hash, u64 fetch_hash, u32 rsrc2, bool probe, bool boom);
// Key of one translation of a program: the program's, the stage registers in `in` (rsrc1, rect, pixel state, threads) and the slots' specs.
u64 TranslationKey(u64 program_key, const Gcn::Input& in, std::span<const Gcn::SlotSpec> spec);
// User SGPR holding the address of the fetch shader a VS calls through s_swappc_b64 (-1: none).
int FetchSgpr(std::span<const u32> code);
struct Variant {  // a program's last compiled variant (prog_lock), for Stats::compiled
    u64 code, fetch, ps, threads;
    u32 rsrc1, rsrc2;
    bool rect;
    std::vector<Gcn::SlotSpec> spec;
};
} // namespace Gpu
