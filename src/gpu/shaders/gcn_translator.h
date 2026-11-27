// SPDX-License-Identifier: GPL-2.0-or-later
// GCN (PS4, Sea Islands) program -> GLSL for Vulkan (plan 3.1). Registers are variables; EXEC and VCC are 64-bit
// subgroup ballots, uniform like on the hardware; branches run as a pc loop; the fetch shader is inlined. Memory reads
// follow the guest address (buffer device address), except scalar loads at fixed offsets from a known place, whose
// dwords the CPU copies into the draw table (`Const`); images and samplers go through the recipe: slot k of the draw
// table holds the bindless index of the descriptor found at `Slot::path`.
#pragma once
#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include "shaderc/shaderc.h"
#include "debug/debug.h"
#include "gpu/piece_hash.h"

namespace Gcn {
enum class Stage : std::uint8_t { Vertex, Pixel, Compute };
// Exploded vertices: a position in front of the camera more than this many half-screens off its centre (|x| or |y| > k * w, w > 0;
// default 8, GOW3_BOOM_CLIP 1 to 255), or with a NaN or Inf component, marks the vertex; it is part of the program key.
inline std::uint32_t BoomClip() {
    static const std::uint32_t k = [] { const char* v = Gow3::DebugOption("GOW3_BOOM_CLIP"); return v ? std::uint32_t(std::clamp(std::strtoul(v, nullptr, 10), 1ul, 255ul)) : 8u; }();
    return k;
}
// Exploded velocity (and any other interpolant): a param export with a NaN or Inf component, or |component| above this (default 512,
// GOW3_BOOM_PARAM 1 to 8191), marks the vertex like a position does; the velocity pass encodes 0.5 * velocity + 0.5 in a param.
inline std::uint32_t BoomParamLimit() {
    static const std::uint32_t k = [] { const char* v = Gow3::DebugOption("GOW3_BOOM_PARAM"); return v ? std::uint32_t(std::clamp(std::strtoul(v, nullptr, 10), 1ul, 8191ul)) : 512u; }();
    return k;
}
// motionblur.fx technique 7: this program consumes the game's clamped, encoded velocity.
inline std::uint64_t MotionBlurCodeHash() { return Gpu::PieceHash::UsesAes() ? 0x7110443d8f45cc4dull : 0xb452868e19ed3cceull; }
inline std::uint64_t MotionVectorResolveCodeHash() { return Gpu::PieceHash::UsesAes() ? 0xbbdcabca8b72215full : 0xa372e8396eff9499ull; }
// R_FX depth-fog composite: decodes the scene (WhiteBuffer, sampler FORCE_UNNORMALIZED) and fogs it by depth (normalized
// sampler), both at its one interpolant.
inline std::uint64_t FogCompositeCodeHash() { return Gpu::PieceHash::UsesAes() ? 0xad42431c3ee677e5ull : 0xbf5405573f1a0f3dull; }

// User data register `reg`, then `depth` loads through the 64-bit pointer at the current place plus `offset[i]` bytes.
struct Path {
    std::uint8_t reg = 0xff, depth = 0;
    std::uint32_t offset[3]{};
    bool operator==(const Path&) const = default;
};
enum class SlotKind : std::uint8_t { Texture, Storage, Sampler };
struct Slot {
    SlotKind kind;
    Path path;
    std::uint64_t samplers = 0;  // texture: bitmask of sampler slots that sample it
};
// Bytes [lo, hi) of scalar loads past the base of the V# at `path` (s_buffer_load, range checked like the hardware) or
// past the 64-bit pointer at `path` (s_load), as the draw table carries them (constant bank instead of memory loads).
struct Const {
    Path path;
    std::uint32_t lo = 0, hi = 0;
    bool buffer = false;
};

struct Input {
    Stage stage = Stage::Pixel;
    std::span<const std::uint32_t> code, fetch;  // program from its start; VS fetch shader
    std::uint32_t rsrc1 = 0, rsrc2 = 0;          // SPI_SHADER_PGM_RSRC1/2 (COMPUTE_PGM_RSRC1/2)
    std::uint32_t ps_input_ena = 0, ps_input_addr = 0, ps_input_cntl[32]{};
    std::uint32_t col_format = 0;                // SPI_SHADER_COL_FORMAT
    std::uint8_t mrt_class[8]{};                 // 0 float, 1 uint, 2 sint (CB_COLOR*_INFO number type)
    // Export swizzle per MRT (CB_COLOR*_INFO COMP_SWAP): output channel i takes component (mrt_map >> 2i) & 3. Follows
    // mrt_class: both are part of the pixel shader key.
    std::uint8_t mrt_map[8]{0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4};
    std::uint32_t threads[3]{1, 1, 1};           // COMPUTE_NUM_THREAD_X/Y/Z
    std::uint32_t blend_interlock = 0;            // samples in low byte; bit 8 selects MAX
    bool probe = false;  // VS diagnostics: the first invocation stores gl_Position and gl_VertexIndex at res[slots..+1]
    std::uint32_t probe_pc = ~0u;  // registers stored after this instruction instead of at the position export
    bool boom = false;  // VS counts exploded positions (first index, its loads, gl_Position) at res[slots..+1] (live)
    bool rect = false;  // VS expands a 3-vertex DI_PT_RECTLIST into a 4-vertex triangle strip: each vertex runs the program for the three corners
    bool no_storage_read_nofmt = false;  // GPU lacks shaderStorageImageReadWithoutFormat: keep image_load on Texture (OpImageFetch) instead of merging into Storage
    bool no_float_controls = false;      // GPU lacks shaderSignedZeroInfNanPreserveFloat32
};

// Per slot, from the descriptor at draw time (part of the translation key).
enum class Dim : std::uint8_t { D1, D2, D1Array, D2Array, D3, Cube, D2Ms, D2MsArray };  // 1D views are 2D, height 1
enum class Class : std::uint8_t { Float, Uint, Sint };
struct SlotSpec {
    Dim dim = Dim::D2;
    Class cls = Class::Float;
    bool unnormalized = false;  // sampler: FORCE_UNNORMALIZED
    bool degamma = false;       // sampler: FORCE_DEGAMMA
    bool srgb = false;          // texture: NUM_FORMAT SRGB (the view already linearizes)
    bool msaa4 = false;         // 2D multisampled texture with four samples
    bool stencil = false;       // multisampled stencil aspect read as normalized R8
    // Format load (buffer_load_format): the V# format, constant in the code instead of decoded per invocation.
    bool vfmt = false;
    std::uint8_t dfmt = 0, nfmt = 0;
    std::uint16_t dst_sel = 0;
    // 2D texture or storage slot bound to a larger image: the T# is its top-left part, as big as the T#'s extent, which the
    // program reads from the T# (it may change every frame) against the bound image's size (false: the whole image).
    bool part = false;
};
// The spec as one word, field by field (the struct has padding): what the translation key hashes.
inline std::uint64_t Pack(const SlotSpec& s) {
    return std::uint64_t(s.dim) | std::uint64_t(s.cls) << 4 | std::uint64_t(s.unnormalized) << 8 | std::uint64_t(s.degamma) << 9 | std::uint64_t(s.srgb) << 10 |
           std::uint64_t(s.vfmt) << 11 | std::uint64_t(s.part) << 12 | std::uint64_t(s.dfmt) << 16 | std::uint64_t(s.nfmt) << 24 | std::uint64_t(s.dst_sel) << 32 |
           std::uint64_t(s.msaa4) << 48 | std::uint64_t(s.stencil) << 49;
}
inline SlotSpec Unpack(std::uint64_t w) {  // Pack's inverse
    SlotSpec s;
    s.dim = Dim(w & 15), s.cls = Class(w >> 4 & 15), s.unnormalized = w >> 8 & 1, s.degamma = w >> 9 & 1, s.srgb = w >> 10 & 1, s.vfmt = w >> 11 & 1;
    s.part = w >> 12 & 1, s.dfmt = std::uint8_t(w >> 16), s.nfmt = std::uint8_t(w >> 24), s.dst_sel = std::uint16_t(w >> 32);
    s.msaa4 = w >> 48 & 1;
    s.stencil = w >> 49 & 1;
    return s;
}
SlotSpec TextureSpec(const std::uint32_t* t);  // T#
SlotSpec SamplerSpec(const std::uint32_t* s);  // S#
SlotSpec BufferSpec(const std::uint32_t* v);   // V#

struct Program;
struct Analysis {
    std::vector<Slot> slots;
    std::vector<Path> buffers;  // V# locations of buffer loads (diagnostics)
    std::vector<Path> formats;  // V# locations of format loads and stores: their specs follow the slots'
    std::uint64_t stored = 0;   // bit k: the program writes the buffer of formats[k] (the GPU needs its memory mapped)
    // Draw table: the constants' dwords (`const_dwords`, in this order), then the slots' bindless indices.
    std::vector<Const> consts;
    std::uint32_t const_dwords = 0;
    std::uint64_t attrs = 0;  // pixel: bit k = the program reads interpolant k (the other SPI_PS_INPUT_CNTL entries are stale state)
    int depth_resolve_slot = -1;
    std::string error;
    std::shared_ptr<const Program> program;
};
Analysis Analyze(const Input& in);
// The program as SPIR-V, written directly (gcn_spirv.cpp): false with `error` set. `out_mask` (vertex): the exported parameters
// (bit k: parameter k).
bool EmitSpirv(const Analysis& a, const Input& in, std::span<const SlotSpec> spec, std::vector<std::uint32_t>& words, std::uint64_t& out_mask,
              std::string& error, bool full_opt = false);

// GLSL -> SPIR-V for the renderer's auxiliary shaders. The compiler and options are shared read-only by compile threads.
struct Compiler {
    shaderc_compiler_t cc = shaderc_compiler_initialize();
    shaderc_compile_options_t opt = shaderc_compile_options_initialize();
    Compiler();
    ~Compiler();
    std::vector<std::uint32_t> Spirv(const std::string& glsl, shaderc_shader_kind kind, std::string& error,
                                     const std::vector<std::pair<std::string, std::string>>& macros = {});
};
} // namespace Gcn
