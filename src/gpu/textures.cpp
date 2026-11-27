// SPDX-License-Identifier: GPL-2.0-or-later
// Textures, render targets and tiling: adapted from shadPS4
#include "gpu/textures.h"
#include <algorithm>
#include <array>
#include <bit>
#include <magic_enum/magic_enum.hpp>
#include "gpu/renderer_impl.h"
#ifndef GOW3_OFFLINE
#include "gpu/gpu_thread.h"
#include "gpu/buffers.h"
#endif

namespace AmdGpu {

static constexpr u32 MICROTILE_SIZE = 8;
static constexpr u32 DRAM_ROW_SIZE = 1024;

std::string_view NameOf(TileMode tile_mode) {
    return magic_enum::enum_name(tile_mode);
}

ArrayMode GetArrayMode(TileMode tile_mode) {
    switch (tile_mode) {
    case TileMode::Depth1DThin:
    case TileMode::Display1DThin:
    case TileMode::Thin1DThin:
        return ArrayMode::Array1DTiledThin1;
    case TileMode::Depth2DThin64:
    case TileMode::Depth2DThin128:
    case TileMode::Depth2DThin256:
    case TileMode::Depth2DThin512:
    case TileMode::Depth2DThin1K:
    case TileMode::Display2DThin:
    case TileMode::Thin2DThin:
        return ArrayMode::Array2DTiledThin1;
    case TileMode::DisplayThinPrt:
    case TileMode::ThinThinPrt:
        return ArrayMode::ArrayPrtTiledThin1;
    case TileMode::Depth2DThinPrt256:
    case TileMode::Depth2DThinPrt1K:
    case TileMode::Display2DThinPrt:
    case TileMode::Thin2DThinPrt:
        return ArrayMode::ArrayPrt2DTiledThin1;
    case TileMode::Thin3DThin:
    case TileMode::Thin3DThinPrt:
        return ArrayMode::Array3DTiledThin1;
    case TileMode::Thick1DThick:
        return ArrayMode::Array1DTiledThick;
    case TileMode::Thick2DThick:
        return ArrayMode::Array2DTiledThick;
    case TileMode::Thick3DThick:
        return ArrayMode::Array3DTiledThick;
    case TileMode::ThickThickPrt:
        return ArrayMode::ArrayPrtTiledThick;
    case TileMode::Thick2DThickPrt:
        return ArrayMode::ArrayPrt2DTiledThick;
    case TileMode::Thick3DThickPrt:
        return ArrayMode::ArrayPrt3DTiledThick;
    case TileMode::Thick2DXThick:
        return ArrayMode::Array2DTiledXThick;
    case TileMode::Thick3DXThick:
        return ArrayMode::Array3DTiledXThick;
    case TileMode::DisplayLinearAligned:
        return ArrayMode::ArrayLinearAligned;
    case TileMode::DisplayLinearGeneral:
        return ArrayMode::ArrayLinearGeneral;
    default:
        UNREACHABLE_MSG("Unknown tile mode = {}", u32(tile_mode));
    }
}

MicroTileMode GetMicroTileMode(TileMode tile_mode) {
    switch (tile_mode) {
    case TileMode::Depth2DThin64:
    case TileMode::Depth2DThin128:
    case TileMode::Depth2DThin256:
    case TileMode::Depth2DThin512:
    case TileMode::Depth2DThin1K:
    case TileMode::Depth1DThin:
    case TileMode::Depth2DThinPrt256:
    case TileMode::Depth2DThinPrt1K:
        return MicroTileMode::Depth;
    case TileMode::DisplayLinearAligned:
    case TileMode::Display1DThin:
    case TileMode::Display2DThin:
    case TileMode::DisplayThinPrt:
    case TileMode::Display2DThinPrt:
    case TileMode::DisplayLinearGeneral:
        return MicroTileMode::Display;
    case TileMode::Thin1DThin:
    case TileMode::Thin2DThin:
    case TileMode::Thin3DThin:
    case TileMode::ThinThinPrt:
    case TileMode::Thin2DThinPrt:
    case TileMode::Thin3DThinPrt:
        return MicroTileMode::Thin;
    case TileMode::Thick1DThick:
    case TileMode::Thick2DThick:
    case TileMode::Thick3DThick:
    case TileMode::ThickThickPrt:
    case TileMode::Thick2DThickPrt:
    case TileMode::Thick3DThickPrt:
    case TileMode::Thick2DXThick:
    case TileMode::Thick3DXThick:
        return MicroTileMode::Thick;
    default:
        UNREACHABLE_MSG("Unknown tile mode = {}", u32(tile_mode));
    }
}

PipeConfig GetPipeConfig(TileMode tile_mode) {
    switch (tile_mode) {
    case TileMode::Depth2DThin64:
    case TileMode::Depth2DThin128:
    case TileMode::Depth2DThin256:
    case TileMode::Depth2DThin512:
    case TileMode::Depth2DThin1K:
    case TileMode::Depth1DThin:
    case TileMode::Depth2DThinPrt256:
    case TileMode::Depth2DThinPrt1K:
    case TileMode::DisplayLinearAligned:
    case TileMode::Display1DThin:
    case TileMode::Display2DThin:
    case TileMode::Display2DThinPrt:
    case TileMode::Thin1DThin:
    case TileMode::Thin2DThin:
    case TileMode::Thin2DThinPrt:
    case TileMode::Thin3DThinPrt:
    case TileMode::Thick1DThick:
    case TileMode::Thick2DThick:
    case TileMode::Thick2DThickPrt:
    case TileMode::Thick2DXThick:
        return PipeConfig::P8_32x32_16x16;
    case TileMode::DisplayThinPrt:
    case TileMode::Thin3DThin:
    case TileMode::ThinThinPrt:
    case TileMode::Thick3DThick:
    case TileMode::ThickThickPrt:
    case TileMode::Thick3DThickPrt:
    case TileMode::Thick3DXThick:
        return PipeConfig::P8_32x32_8x16;
    case TileMode::DisplayLinearGeneral:
        return PipeConfig::P2;
    default:
        UNREACHABLE_MSG("Unknown tile mode = {}", u32(tile_mode));
    }
}

PipeConfig GetAltPipeConfig(TileMode tile_mode) {
    switch (tile_mode) {
    case TileMode::Depth2DThin64:
    case TileMode::Depth2DThin128:
    case TileMode::Depth2DThin256:
    case TileMode::Depth2DThin512:
    case TileMode::Depth2DThin1K:
    case TileMode::Depth1DThin:
    case TileMode::Depth2DThinPrt256:
    case TileMode::Depth2DThinPrt1K:
    case TileMode::DisplayLinearAligned:
    case TileMode::Display1DThin:
    case TileMode::Display2DThin:
    case TileMode::DisplayThinPrt:
    case TileMode::Display2DThinPrt:
    case TileMode::Thin1DThin:
    case TileMode::Thin2DThin:
    case TileMode::Thin3DThin:
    case TileMode::ThinThinPrt:
    case TileMode::Thin2DThinPrt:
    case TileMode::Thin3DThinPrt:
    case TileMode::Thick1DThick:
    case TileMode::Thick2DThick:
    case TileMode::Thick3DThick:
    case TileMode::ThickThickPrt:
    case TileMode::Thick2DThickPrt:
    case TileMode::Thick3DThickPrt:
    case TileMode::Thick2DXThick:
    case TileMode::Thick3DXThick:
        return PipeConfig::P16_32x32_8x16;
    case TileMode::DisplayLinearGeneral:
        return PipeConfig::P2;
    default:
        UNREACHABLE_MSG("Unknown tile mode = {}", u32(tile_mode));
    }
}

u32 GetSampleSplit(TileMode tile_mode) {
    switch (tile_mode) {
    case TileMode::Depth2DThin64:
    case TileMode::Depth2DThin128:
    case TileMode::Depth2DThin256:
    case TileMode::Depth2DThin512:
    case TileMode::Depth2DThin1K:
    case TileMode::Depth1DThin:
    case TileMode::Depth2DThinPrt256:
    case TileMode::Depth2DThinPrt1K:
    case TileMode::DisplayLinearAligned:
    case TileMode::Display1DThin:
    case TileMode::Thin1DThin:
    case TileMode::Thick1DThick:
    case TileMode::Thick2DThick:
    case TileMode::Thick3DThick:
    case TileMode::ThickThickPrt:
    case TileMode::Thick2DThickPrt:
    case TileMode::Thick3DThickPrt:
    case TileMode::Thick2DXThick:
    case TileMode::Thick3DXThick:
    case TileMode::DisplayLinearGeneral:
        return 1;
    case TileMode::Display2DThin:
    case TileMode::DisplayThinPrt:
    case TileMode::Display2DThinPrt:
    case TileMode::Thin2DThin:
    case TileMode::Thin3DThin:
    case TileMode::ThinThinPrt:
    case TileMode::Thin2DThinPrt:
    case TileMode::Thin3DThinPrt:
        return 2;
    default:
        UNREACHABLE_MSG("Unknown tile mode = {}", u32(tile_mode));
    }
}

u32 GetTileSplitHw(TileMode tile_mode) {
    switch (tile_mode) {
    case TileMode::Depth2DThin64:
    case TileMode::Depth1DThin:
    case TileMode::DisplayLinearAligned:
    case TileMode::Display1DThin:
    case TileMode::Display2DThin:
    case TileMode::DisplayThinPrt:
    case TileMode::Display2DThinPrt:
    case TileMode::Thin1DThin:
    case TileMode::Thin2DThin:
    case TileMode::Thin3DThin:
    case TileMode::ThinThinPrt:
    case TileMode::Thin2DThinPrt:
    case TileMode::Thin3DThinPrt:
    case TileMode::Thick1DThick:
    case TileMode::Thick2DThick:
    case TileMode::Thick3DThick:
    case TileMode::ThickThickPrt:
    case TileMode::Thick2DThickPrt:
    case TileMode::Thick3DThickPrt:
    case TileMode::Thick2DXThick:
    case TileMode::Thick3DXThick:
    case TileMode::DisplayLinearGeneral:
        return 64;
    case TileMode::Depth2DThin128:
        return 128;
    case TileMode::Depth2DThin256:
    case TileMode::Depth2DThinPrt256:
        return 256;
    case TileMode::Depth2DThin512:
        return 512;
    case TileMode::Depth2DThin1K:
    case TileMode::Depth2DThinPrt1K:
        return 1024;
    default:
        UNREACHABLE_MSG("Unknown tile mode = {}", u32(tile_mode));
    }
}

u32 GetBankWidth(MacroTileMode mode) {
    switch (mode) {
    case MacroTileMode::Mode_1x4_16:
    case MacroTileMode::Mode_1x2_16:
    case MacroTileMode::Mode_1x1_16:
    case MacroTileMode::Mode_1x1_16_Dup:
    case MacroTileMode::Mode_1x1_8:
    case MacroTileMode::Mode_1x1_4:
    case MacroTileMode::Mode_1x1_2:
    case MacroTileMode::Mode_1x1_2_Dup:
    case MacroTileMode::Mode_1x8_16:
    case MacroTileMode::Mode_1x4_16_Dup:
    case MacroTileMode::Mode_1x2_16_Dup:
    case MacroTileMode::Mode_1x1_16_Dup2:
    case MacroTileMode::Mode_1x1_8_Dup:
    case MacroTileMode::Mode_1x1_4_Dup:
    case MacroTileMode::Mode_1x1_2_Dup2:
    case MacroTileMode::Mode_1x1_2_Dup3:
        return 1;
    default:
        UNREACHABLE_MSG("Unknown macro tile mode = {}", u32(mode));
    }
}

u32 GetBankHeight(MacroTileMode mode) {
    switch (mode) {
    case MacroTileMode::Mode_1x1_16:
    case MacroTileMode::Mode_1x1_16_Dup:
    case MacroTileMode::Mode_1x1_8:
    case MacroTileMode::Mode_1x1_4:
    case MacroTileMode::Mode_1x1_2:
    case MacroTileMode::Mode_1x1_2_Dup:
    case MacroTileMode::Mode_1x1_16_Dup2:
    case MacroTileMode::Mode_1x1_8_Dup:
    case MacroTileMode::Mode_1x1_4_Dup:
    case MacroTileMode::Mode_1x1_2_Dup2:
    case MacroTileMode::Mode_1x1_2_Dup3:
        return 1;
    case MacroTileMode::Mode_1x2_16:
    case MacroTileMode::Mode_1x2_16_Dup:
        return 2;
    case MacroTileMode::Mode_1x4_16:
    case MacroTileMode::Mode_1x4_16_Dup:
        return 4;
    case MacroTileMode::Mode_1x8_16:
        return 8;
    default:
        UNREACHABLE_MSG("Unknown macro tile mode = {}", u32(mode));
    }
}

u32 GetNumBanks(MacroTileMode mode) {
    switch (mode) {
    case MacroTileMode::Mode_1x1_2:
    case MacroTileMode::Mode_1x1_2_Dup:
    case MacroTileMode::Mode_1x1_2_Dup2:
    case MacroTileMode::Mode_1x1_2_Dup3:
        return 2;
    case MacroTileMode::Mode_1x1_4:
    case MacroTileMode::Mode_1x1_4_Dup:
        return 4;
    case MacroTileMode::Mode_1x1_8:
    case MacroTileMode::Mode_1x1_8_Dup:
        return 8;
    case MacroTileMode::Mode_1x4_16:
    case MacroTileMode::Mode_1x2_16:
    case MacroTileMode::Mode_1x1_16:
    case MacroTileMode::Mode_1x1_16_Dup:
    case MacroTileMode::Mode_1x8_16:
    case MacroTileMode::Mode_1x4_16_Dup:
    case MacroTileMode::Mode_1x2_16_Dup:
    case MacroTileMode::Mode_1x1_16_Dup2:
        return 16;
    default:
        UNREACHABLE_MSG("Unknown macro tile mode = {}", u32(mode));
    }
}

u32 GetMacrotileAspect(MacroTileMode mode) {
    switch (mode) {
    case MacroTileMode::Mode_1x1_8:
    case MacroTileMode::Mode_1x1_4:
    case MacroTileMode::Mode_1x1_2:
    case MacroTileMode::Mode_1x1_2_Dup:
    case MacroTileMode::Mode_1x1_8_Dup:
    case MacroTileMode::Mode_1x1_4_Dup:
    case MacroTileMode::Mode_1x1_2_Dup2:
    case MacroTileMode::Mode_1x1_2_Dup3:
        return 1;
    case MacroTileMode::Mode_1x2_16:
    case MacroTileMode::Mode_1x1_16:
    case MacroTileMode::Mode_1x1_16_Dup:
    case MacroTileMode::Mode_1x2_16_Dup:
    case MacroTileMode::Mode_1x1_16_Dup2:
        return 2;
    case MacroTileMode::Mode_1x4_16:
    case MacroTileMode::Mode_1x8_16:
    case MacroTileMode::Mode_1x4_16_Dup:
        return 4;
    default:
        UNREACHABLE_MSG("Unknown macro tile mode = {}", u32(mode));
    }
}

u32 GetAltBankHeight(MacroTileMode mode) {
    switch (mode) {
    case MacroTileMode::Mode_1x1_8:
    case MacroTileMode::Mode_1x1_4:
    case MacroTileMode::Mode_1x1_2:
    case MacroTileMode::Mode_1x1_2_Dup:
    case MacroTileMode::Mode_1x1_16_Dup2:
    case MacroTileMode::Mode_1x1_8_Dup:
    case MacroTileMode::Mode_1x1_4_Dup:
    case MacroTileMode::Mode_1x1_2_Dup2:
    case MacroTileMode::Mode_1x1_2_Dup3:
        return 1;
    case MacroTileMode::Mode_1x1_16:
    case MacroTileMode::Mode_1x1_16_Dup:
    case MacroTileMode::Mode_1x2_16_Dup:
        return 2;
    case MacroTileMode::Mode_1x4_16:
    case MacroTileMode::Mode_1x2_16:
    case MacroTileMode::Mode_1x8_16:
    case MacroTileMode::Mode_1x4_16_Dup:
        return 4;
    default:
        UNREACHABLE_MSG("Unknown macro tile mode = {}", u32(mode));
    }
}

u32 GetAltNumBanks(MacroTileMode mode) {
    switch (mode) {
    case MacroTileMode::Mode_1x1_2_Dup:
    case MacroTileMode::Mode_1x1_2_Dup2:
    case MacroTileMode::Mode_1x1_2_Dup3:
        return 2;
    case MacroTileMode::Mode_1x1_2:
    case MacroTileMode::Mode_1x1_8_Dup:
    case MacroTileMode::Mode_1x1_4_Dup:
        return 4;
    case MacroTileMode::Mode_1x4_16:
    case MacroTileMode::Mode_1x2_16:
    case MacroTileMode::Mode_1x1_16:
    case MacroTileMode::Mode_1x1_16_Dup:
    case MacroTileMode::Mode_1x1_8:
    case MacroTileMode::Mode_1x1_4:
    case MacroTileMode::Mode_1x4_16_Dup:
    case MacroTileMode::Mode_1x2_16_Dup:
    case MacroTileMode::Mode_1x1_16_Dup2:
        return 8;
    case MacroTileMode::Mode_1x8_16:
        return 16;
    default:
        UNREACHABLE_MSG("Unknown macro tile mode = {}", u32(mode));
    }
}

u32 GetAltMacrotileAspect(MacroTileMode mode) {
    switch (mode) {
    case MacroTileMode::Mode_1x1_16:
    case MacroTileMode::Mode_1x1_16_Dup:
    case MacroTileMode::Mode_1x1_8:
    case MacroTileMode::Mode_1x1_4:
    case MacroTileMode::Mode_1x1_2:
    case MacroTileMode::Mode_1x1_2_Dup:
    case MacroTileMode::Mode_1x2_16_Dup:
    case MacroTileMode::Mode_1x1_16_Dup2:
    case MacroTileMode::Mode_1x1_8_Dup:
    case MacroTileMode::Mode_1x1_4_Dup:
    case MacroTileMode::Mode_1x1_2_Dup2:
    case MacroTileMode::Mode_1x1_2_Dup3:
        return 1;
    case MacroTileMode::Mode_1x4_16:
    case MacroTileMode::Mode_1x2_16:
    case MacroTileMode::Mode_1x8_16:
    case MacroTileMode::Mode_1x4_16_Dup:
        return 2;
    default:
        UNREACHABLE_MSG("Unknown macro tile mode = {}", u32(mode));
    }
}

bool IsMacroTiled(ArrayMode array_mode) {
    switch (array_mode) {
    case ArrayMode::ArrayLinearGeneral:
    case ArrayMode::ArrayLinearAligned:
    case ArrayMode::Array1DTiledThin1:
    case ArrayMode::Array1DTiledThick:
        return false;
    case ArrayMode::Array2DTiledThin1:
    case ArrayMode::ArrayPrtTiledThin1:
    case ArrayMode::ArrayPrt2DTiledThin1:
    case ArrayMode::Array2DTiledThick:
    case ArrayMode::Array2DTiledXThick:
    case ArrayMode::ArrayPrtTiledThick:
    case ArrayMode::ArrayPrt2DTiledThick:
    case ArrayMode::ArrayPrt3DTiledThin1:
    case ArrayMode::Array3DTiledThin1:
    case ArrayMode::Array3DTiledThick:
    case ArrayMode::Array3DTiledXThick:
    case ArrayMode::ArrayPrt3DTiledThick:
        return true;
    default:
        UNREACHABLE_MSG("Unknown array mode = {}", u32(array_mode));
    }
}

bool IsPrt(ArrayMode array_mode) {
    switch (array_mode) {
    case ArrayMode::ArrayPrtTiledThin1:
    case ArrayMode::ArrayPrtTiledThick:
    case ArrayMode::ArrayPrt2DTiledThin1:
    case ArrayMode::ArrayPrt2DTiledThick:
    case ArrayMode::ArrayPrt3DTiledThin1:
    case ArrayMode::ArrayPrt3DTiledThick:
        return true;
    case ArrayMode::ArrayLinearGeneral:
    case ArrayMode::ArrayLinearAligned:
    case ArrayMode::Array1DTiledThin1:
    case ArrayMode::Array1DTiledThick:
    case ArrayMode::Array2DTiledThin1:
    case ArrayMode::Array2DTiledThick:
    case ArrayMode::Array2DTiledXThick:
    case ArrayMode::Array3DTiledThin1:
    case ArrayMode::Array3DTiledThick:
    case ArrayMode::Array3DTiledXThick:
        return false;
    default:
        UNREACHABLE_MSG("Unknown array mode = {}", u32(array_mode));
    }
}

u32 GetMicroTileThickness(ArrayMode array_mode) {
    switch (array_mode) {
    case ArrayMode::ArrayLinearGeneral:
    case ArrayMode::ArrayLinearAligned:
    case ArrayMode::Array1DTiledThin1:
    case ArrayMode::Array2DTiledThin1:
    case ArrayMode::ArrayPrtTiledThin1:
    case ArrayMode::ArrayPrt2DTiledThin1:
    case ArrayMode::ArrayPrt3DTiledThin1:
    case ArrayMode::Array3DTiledThin1:
        return 1;
    case ArrayMode::Array1DTiledThick:
    case ArrayMode::Array2DTiledThick:
    case ArrayMode::Array3DTiledThick:
    case ArrayMode::ArrayPrtTiledThick:
    case ArrayMode::ArrayPrt2DTiledThick:
    case ArrayMode::ArrayPrt3DTiledThick:
        return 4;
    case ArrayMode::Array2DTiledXThick:
    case ArrayMode::Array3DTiledXThick:
        return 8;
    default:
        UNREACHABLE_MSG("Unknown array mode = {}", u32(array_mode));
    }
}

u32 GetPipeCount(PipeConfig pipe_cfg) {
    switch (pipe_cfg) {
    case PipeConfig::P2:
        return 2;
    case PipeConfig::P8_32x32_8x16:
    case PipeConfig::P8_32x32_16x16:
        return 8;
    case PipeConfig::P16_32x32_8x16:
        return 16;
    default:
        UNREACHABLE_MSG("Unknown pipe config = {}", u32(pipe_cfg));
    }
}

u32 CalculateTileSplit(TileMode tile_mode, ArrayMode array_mode, MicroTileMode micro_tile_mode,
                       u32 bpp) {
    const u32 sample_split = GetSampleSplit(tile_mode);
    const u32 tile_split_hw = GetTileSplitHw(tile_mode);
    const u32 tile_thickness = GetMicroTileThickness(array_mode);
    const u32 tile_bytes_1x = (bpp * MICROTILE_SIZE * MICROTILE_SIZE * tile_thickness + 7) / 8;
    const u32 color_tile_split = std::max(256U, sample_split * tile_bytes_1x);
    const u32 tile_split =
        micro_tile_mode == MicroTileMode::Depth ? tile_split_hw : color_tile_split;
    return std::min(DRAM_ROW_SIZE, tile_split);
}

MacroTileMode CalculateMacrotileMode(TileMode tile_mode, u32 bpp, u32 num_samples) noexcept {
    ASSERT_MSG(std::has_single_bit(num_samples) && num_samples <= 16, "Invalid sample count {}",
               num_samples);
    ASSERT_MSG(bpp >= 1 && bpp <= 128, "Invalid bpp {}", bpp);

    const ArrayMode array_mode = GetArrayMode(tile_mode);
    ASSERT_MSG(IsMacroTiled(array_mode), "Tile mode not macro tiled");

    const MicroTileMode micro_tile_mode = GetMicroTileMode(tile_mode);
    const u32 tile_thickness = GetMicroTileThickness(array_mode);
    const u32 tile_bytes_1x = bpp * MICROTILE_SIZE * MICROTILE_SIZE * tile_thickness / 8;
    const u32 tilesplic = CalculateTileSplit(tile_mode, array_mode, micro_tile_mode, bpp);
    const u32 tile_bytes = std::min(tilesplic, num_samples * tile_bytes_1x);
    const u32 mtm_idx = std::bit_width(tile_bytes / 64) - 1;
    return IsPrt(array_mode) ? MacroTileMode(mtm_idx + 8) : MacroTileMode(mtm_idx);
}

} // namespace AmdGpu

namespace AmdGpu {

std::string_view NameOf(DataFormat fmt) {
    switch (fmt) {
    case DataFormat::FormatInvalid:
        return "FormatInvalid";
    case DataFormat::Format8:
        return "Format8";
    case DataFormat::Format16:
        return "Format16";
    case DataFormat::Format8_8:
        return "Format8_8";
    case DataFormat::Format32:
        return "Format32";
    case DataFormat::Format16_16:
        return "Format16_16";
    case DataFormat::Format10_11_11:
        return "Format10_11_11";
    case DataFormat::Format11_11_10:
        return "Format11_11_10";
    case DataFormat::Format10_10_10_2:
        return "Format10_10_10_2";
    case DataFormat::Format2_10_10_10:
        return "Format2_10_10_10";
    case DataFormat::Format8_8_8_8:
        return "Format8_8_8_8";
    case DataFormat::Format32_32:
        return "Format32_32";
    case DataFormat::Format16_16_16_16:
        return "Format16_16_16_16";
    case DataFormat::Format32_32_32:
        return "Format32_32_32";
    case DataFormat::Format32_32_32_32:
        return "Format32_32_32_32";
    case DataFormat::Format5_6_5:
        return "Format5_6_5";
    case DataFormat::Format1_5_5_5:
        return "Format1_5_5_5";
    case DataFormat::Format5_5_5_1:
        return "Format5_5_5_1";
    case DataFormat::Format4_4_4_4:
        return "Format4_4_4_4";
    case DataFormat::Format8_24:
        return "Format8_24";
    case DataFormat::Format24_8:
        return "Format24_8";
    case DataFormat::FormatX24_8_32:
        return "FormatX24_8_32";
    case DataFormat::FormatGB_GR:
        return "FormatGB_GR";
    case DataFormat::FormatBG_RG:
        return "FormatBG_RG";
    case DataFormat::Format5_9_9_9:
        return "Format5_9_9_9";
    case DataFormat::FormatBc1:
        return "FormatBc1";
    case DataFormat::FormatBc2:
        return "FormatBc2";
    case DataFormat::FormatBc3:
        return "FormatBc3";
    case DataFormat::FormatBc4:
        return "FormatBc4";
    case DataFormat::FormatBc5:
        return "FormatBc5";
    case DataFormat::FormatBc6:
        return "FormatBc6";
    case DataFormat::FormatBc7:
        return "FormatBc7";
    default:
        UNREACHABLE();
    }
}

std::string_view NameOf(NumberFormat fmt) {
    switch (fmt) {
    case NumberFormat::Unorm:
        return "Unorm";
    case NumberFormat::Snorm:
        return "Snorm";
    case NumberFormat::Uscaled:
        return "Uscaled";
    case NumberFormat::Sscaled:
        return "Sscaled";
    case NumberFormat::Uint:
        return "Uint";
    case NumberFormat::Sint:
        return "Sint";
    case NumberFormat::SnormNz:
        return "SnormNz";
    case NumberFormat::Float:
        return "Float";
    case NumberFormat::Srgb:
        return "Srgb";
    case NumberFormat::Ubnorm:
        return "Ubnorm";
    case NumberFormat::UbnormNz:
        return "UbnormNz";
    case NumberFormat::Ubint:
        return "Ubint";
    case NumberFormat::Ubscaled:
        return "Unscaled";
    default:
        UNREACHABLE();
    }
}

static constexpr std::array NUM_COMPONENTS = {
    0, //  0 FormatInvalid
    1, //  1 Format8
    1, //  2 Format16
    2, //  3 Format8_8
    1, //  4 Format32
    2, //  5 Format16_16
    3, //  6 Format10_11_11
    3, //  7 Format11_11_10
    4, //  8 Format10_10_10_2
    4, //  9 Format2_10_10_10
    4, // 10 Format8_8_8_8
    2, // 11 Format32_32
    4, // 12 Format16_16_16_16
    3, // 13 Format32_32_32
    4, // 14 Format32_32_32_32
    0, // 15
    3, // 16 Format5_6_5
    4, // 17 Format1_5_5_5
    4, // 18 Format5_5_5_1
    4, // 19 Format4_4_4_4
    2, // 20 Format8_24
    2, // 21 Format24_8
    2, // 22 FormatX24_8_32
    0, // 23
    0, // 24
    0, // 25
    0, // 26
    0, // 27
    0, // 28
    0, // 29
    0, // 30
    0, // 31
    3, // 32 FormatGB_GR
    3, // 33 FormatBG_RG
    4, // 34 Format5_9_9_9
    4, // 35 FormatBc1
    4, // 36 FormatBc2
    4, // 37 FormatBc3
    1, // 38 FormatBc4
    2, // 39 FormatBc5
    3, // 40 FormatBc6
    4, // 41 FormatBc7
};

u32 NumComponents(DataFormat format) {
    const u32 index = static_cast<u32>(format);
    ASSERT_MSG(index < NUM_COMPONENTS.size(), "Invalid data format = {}", format);
    return NUM_COMPONENTS[index];
}

CompMapping MrtSwizzle(u32 comp_swap, DataFormat format) {
    using enum CompSwizzle;
    // clang-format off
    static constexpr std::array<std::array<CompMapping, 4>, 4> mrt_swizzles{{
        // Standard
        std::array<CompMapping, 4>{{
            {.r = Red, .g = Zero, .b = Zero, .a = Zero},
            {.r = Red, .g = Green, .b = Zero, .a = Zero},
            {.r = Red, .g = Green, .b = Blue, .a = Zero},
            {.r = Red, .g = Green, .b = Blue, .a = Alpha},
        }},
        // Alternate
        std::array<CompMapping, 4>{{
            {.r = Green, .g = Zero, .b = Zero, .a = Zero},
            {.r = Red, .g = Alpha, .b = Zero, .a = Zero},
            {.r = Red, .g = Green, .b = Alpha, .a = Zero},
            {.r = Blue, .g = Green, .b = Red, .a = Alpha},
        }},
        // StandardReverse
        std::array<CompMapping, 4>{{
            {.r = Blue, .g = Zero, .b = Zero, .a = Zero},
            {.r = Green, .g = Red, .b = Zero, .a = Zero},
            {.r = Blue, .g = Green, .b = Red, .a = Zero},
            {.r = Alpha, .g = Blue, .b = Green, .a = Red},
        }},
        // AlternateReverse
        std::array<CompMapping, 4>{{
            {.r = Alpha, .g = Zero, .b = Zero, .a = Zero},
            {.r = Alpha, .g = Red, .b = Zero, .a = Zero},
            {.r = Alpha, .g = Green, .b = Red, .a = Zero},
            {.r = Alpha, .g = Red, .b = Green, .a = Blue},
        }},
    }};
    // clang-format on
    const u32 components = std::clamp<u32>(NumComponents(format), 1, 4);
    return RemapSwizzle(format, mrt_swizzles[comp_swap & 3][components - 1]);
}

static constexpr std::array BITS_PER_BLOCK = {
    0,   //  0 FormatInvalid
    8,   //  1 Format8
    16,  //  2 Format16
    16,  //  3 Format8_8
    32,  //  4 Format32
    32,  //  5 Format16_16
    32,  //  6 Format10_11_11
    32,  //  7 Format11_11_10
    32,  //  8 Format10_10_10_2
    32,  //  9 Format2_10_10_10
    32,  // 10 Format8_8_8_8
    64,  // 11 Format32_32
    64,  // 12 Format16_16_16_16
    96,  // 13 Format32_32_32
    128, // 14 Format32_32_32_32
    -1,  // 15
    16,  // 16 Format5_6_5
    16,  // 17 Format1_5_5_5
    16,  // 18 Format5_5_5_1
    16,  // 19 Format4_4_4_4
    32,  // 20 Format8_24
    32,  // 21 Format24_8
    64,  // 22 FormatX24_8_32
    -1,  // 23
    -1,  // 24
    -1,  // 25
    -1,  // 26
    -1,  // 27
    -1,  // 28
    -1,  // 29
    -1,  // 30
    -1,  // 31
    16,  // 32 FormatGB_GR
    16,  // 33 FormatBG_RG
    32,  // 34 Format5_9_9_9
    64,  // 35 FormatBc1
    128, // 36 FormatBc2
    128, // 37 FormatBc3
    64,  // 38 FormatBc4
    128, // 39 FormatBc5
    128, // 40 FormatBc6
    128, // 41 FormatBc7
};

u32 NumBitsPerBlock(DataFormat format) {
    const u32 index = static_cast<u32>(format);
    ASSERT_MSG(index < BITS_PER_BLOCK.size(), "Invalid data format = {}", format);
    return BITS_PER_BLOCK[index];
}

static constexpr std::array BITS_PER_ELEMENT = {
    0,   //  0 FormatInvalid
    8,   //  1 Format8
    16,  //  2 Format16
    16,  //  3 Format8_8
    32,  //  4 Format32
    32,  //  5 Format16_16
    32,  //  6 Format10_11_11
    32,  //  7 Format11_11_10
    32,  //  8 Format10_10_10_2
    32,  //  9 Format2_10_10_10
    32,  // 10 Format8_8_8_8
    64,  // 11 Format32_32
    64,  // 12 Format16_16_16_16
    96,  // 13 Format32_32_32
    128, // 14 Format32_32_32_32
    -1,  // 15
    16,  // 16 Format5_6_5
    16,  // 17 Format1_5_5_5
    16,  // 18 Format5_5_5_1
    16,  // 19 Format4_4_4_4
    32,  // 20 Format8_24
    32,  // 21 Format24_8
    64,  // 22 FormatX24_8_32
    -1,  // 23
    -1,  // 24
    -1,  // 25
    -1,  // 26
    -1,  // 27
    -1,  // 28
    -1,  // 29
    -1,  // 30
    -1,  // 31
    16,  // 32 FormatGB_GR
    16,  // 33 FormatBG_RG
    32,  // 34 Format5_9_9_9
    4,   // 35 FormatBc1
    8,   // 36 FormatBc2
    8,   // 37 FormatBc3
    4,   // 38 FormatBc4
    8,   // 39 FormatBc5
    8,   // 40 FormatBc6
    8,   // 41 FormatBc7
};

u32 NumBitsPerElement(DataFormat format) {
    const u32 index = static_cast<u32>(format);
    ASSERT_MSG(index < BITS_PER_ELEMENT.size(), "Invalid data format = {}", format);
    return BITS_PER_ELEMENT[index];
}

} // namespace AmdGpu

namespace Gpu {
// CB_COLOR*_INFO FORMAT and NUMBER_TYPE -> attachment format.
VkFormat ColorFormat(u32 info) {
    const u32 fmt = info >> 2 & 0x1F, nt = info >> 8 & 7;
    auto pick = [&](VkFormat un, VkFormat sn, VkFormat ui, VkFormat si, VkFormat srgb, VkFormat fl) {
        switch (nt) {
        case 0: return un;
        case 1: return sn;
        case 4: return ui;
        case 5: return si;
        case 6: return srgb ? srgb : un;
        case 7: return fl;
        default: return un;
        }
    };
    constexpr VkFormat X = VK_FORMAT_UNDEFINED;
    switch (fmt) {
    case 1: return pick(VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT, VK_FORMAT_R8_SRGB, X);
    case 2: return pick(VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT, X, VK_FORMAT_R16_SFLOAT);
    case 3: return pick(VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R8G8_SRGB, X);
    case 4: return pick(X, X, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT, X, VK_FORMAT_R32_SFLOAT);
    case 5: return pick(VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT, X, VK_FORMAT_R16G16_SFLOAT);
    case 6:
    case 7: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case 8:
    case 9: return nt == 4 ? VK_FORMAT_A2B10G10R10_UINT_PACK32 : VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case 10: return pick(VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R8G8B8A8_SRGB, X);
    case 11: return pick(X, X, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT, X, VK_FORMAT_R32G32_SFLOAT);
    case 12: return pick(VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_SINT, X,
                         VK_FORMAT_R16G16B16A16_SFLOAT);
    case 14: return pick(X, X, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT, X, VK_FORMAT_R32G32B32A32_SFLOAT);
    case 16: return VK_FORMAT_R5G6B5_UNORM_PACK16;
    case 17: return VK_FORMAT_A1R5G5B5_UNORM_PACK16;
    case 18: return VK_FORMAT_R5G5B5A1_UNORM_PACK16;
    case 19: return VK_FORMAT_R4G4B4A4_UNORM_PACK16;
    default: return X;
    }
}

// T# data/number format -> image format: shadPS4 liverpool_to_vk.cpp SurfaceFormats(), after its Remap* (pixel_format.h).
VkFormat SurfaceFormat(u32 dfmt, u32 nfmt) {
    struct Entry {
        u8 d, n;
        VkFormat f;
    };
    static constexpr Entry table[]{
        {1, 0, VK_FORMAT_R8_UNORM}, {1, 1, VK_FORMAT_R8_SNORM}, {1, 4, VK_FORMAT_R8_UINT}, {1, 5, VK_FORMAT_R8_SINT}, {1, 9, VK_FORMAT_R8_SRGB},
        {2, 0, VK_FORMAT_R16_UNORM}, {2, 1, VK_FORMAT_R16_SNORM}, {2, 4, VK_FORMAT_R16_UINT}, {2, 5, VK_FORMAT_R16_SINT}, {2, 7, VK_FORMAT_R16_SFLOAT},
        {3, 0, VK_FORMAT_R8G8_UNORM}, {3, 1, VK_FORMAT_R8G8_SNORM}, {3, 4, VK_FORMAT_R8G8_UINT}, {3, 5, VK_FORMAT_R8G8_SINT}, {3, 9, VK_FORMAT_R8G8_SRGB},
        {4, 4, VK_FORMAT_R32_UINT}, {4, 5, VK_FORMAT_R32_SINT}, {4, 7, VK_FORMAT_R32_SFLOAT},
        {5, 0, VK_FORMAT_R16G16_UNORM}, {5, 1, VK_FORMAT_R16G16_SNORM}, {5, 4, VK_FORMAT_R16G16_UINT}, {5, 5, VK_FORMAT_R16G16_SINT},
        {5, 7, VK_FORMAT_R16G16_SFLOAT}, {6, 7, VK_FORMAT_B10G11R11_UFLOAT_PACK32},
        {9, 0, VK_FORMAT_A2B10G10R10_UNORM_PACK32}, {9, 1, VK_FORMAT_A2B10G10R10_SNORM_PACK32}, {9, 4, VK_FORMAT_A2B10G10R10_UINT_PACK32},
        {9, 5, VK_FORMAT_A2B10G10R10_SINT_PACK32},
        {10, 0, VK_FORMAT_R8G8B8A8_UNORM}, {10, 1, VK_FORMAT_R8G8B8A8_SNORM}, {10, 4, VK_FORMAT_R8G8B8A8_UINT}, {10, 5, VK_FORMAT_R8G8B8A8_SINT},
        {10, 9, VK_FORMAT_R8G8B8A8_SRGB},
        {11, 4, VK_FORMAT_R32G32_UINT}, {11, 5, VK_FORMAT_R32G32_SINT}, {11, 7, VK_FORMAT_R32G32_SFLOAT},
        {12, 0, VK_FORMAT_R16G16B16A16_UNORM}, {12, 1, VK_FORMAT_R16G16B16A16_SNORM}, {12, 4, VK_FORMAT_R16G16B16A16_UINT},
        {12, 5, VK_FORMAT_R16G16B16A16_SINT}, {12, 6, VK_FORMAT_R16G16B16A16_SNORM}, {12, 7, VK_FORMAT_R16G16B16A16_SFLOAT},
        {13, 4, VK_FORMAT_R32G32B32_UINT}, {13, 5, VK_FORMAT_R32G32B32_SINT}, {13, 7, VK_FORMAT_R32G32B32_SFLOAT},
        {14, 4, VK_FORMAT_R32G32B32A32_UINT}, {14, 5, VK_FORMAT_R32G32B32A32_SINT}, {14, 7, VK_FORMAT_R32G32B32A32_SFLOAT},
        {16, 0, VK_FORMAT_R5G6B5_UNORM_PACK16}, {17, 0, VK_FORMAT_A1R5G5B5_UNORM_PACK16}, {18, 0, VK_FORMAT_R5G5B5A1_UNORM_PACK16},
        {19, 0, VK_FORMAT_B4G4R4A4_UNORM_PACK16}, {34, 7, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32},
        {35, 0, VK_FORMAT_BC1_RGBA_UNORM_BLOCK}, {35, 9, VK_FORMAT_BC1_RGBA_SRGB_BLOCK}, {36, 0, VK_FORMAT_BC2_UNORM_BLOCK}, {36, 9, VK_FORMAT_BC2_SRGB_BLOCK},
        {37, 0, VK_FORMAT_BC3_UNORM_BLOCK}, {37, 9, VK_FORMAT_BC3_SRGB_BLOCK}, {38, 0, VK_FORMAT_BC4_UNORM_BLOCK}, {38, 1, VK_FORMAT_BC4_SNORM_BLOCK},
        {39, 0, VK_FORMAT_BC5_UNORM_BLOCK}, {39, 1, VK_FORMAT_BC5_SNORM_BLOCK}, {40, 0, VK_FORMAT_BC6H_UFLOAT_BLOCK}, {40, 1, VK_FORMAT_BC6H_SFLOAT_BLOCK},
        {41, 0, VK_FORMAT_BC7_UNORM_BLOCK}, {41, 9, VK_FORMAT_BC7_SRGB_BLOCK}};
    for (auto& e : table)
        if (e.d == dfmt && e.n == nfmt) return e.f;
    return VK_FORMAT_UNDEFINED;
}

// Bytes per texel of the uncompressed formats render targets use (view reinterpretation of a target as a texture).
u32 TexelBytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8_SRGB: return 1;
    case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_SNORM: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R5G6B5_UNORM_PACK16:
    case VK_FORMAT_A1R5G5B5_UNORM_PACK16: case VK_FORMAT_R5G5B5A1_UNORM_PACK16: case VK_FORMAT_B4G4R4A4_UNORM_PACK16: return 2;
    case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_SNORM:
    case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R8G8B8A8_SRGB: return 4;
    case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16B16A16_SNORM: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    default: return 0;
    }
}

VkFormat UintFormat(u32 bits) {
    switch (bits) {
    case 8: return VK_FORMAT_R8_UINT;
    case 16: return VK_FORMAT_R16_UINT;
    case 32: return VK_FORMAT_R32_UINT;
    case 64: return VK_FORMAT_R32G32_UINT;
    case 128: return VK_FORMAT_R32G32B32A32_UINT;
    default: return VK_FORMAT_UNDEFINED;
    }
}

VkComponentSwizzle Swizzle(AmdGpu::CompSwizzle c) {
    switch (c) {
    case AmdGpu::CompSwizzle::Zero: return VK_COMPONENT_SWIZZLE_ZERO;
    case AmdGpu::CompSwizzle::One: return VK_COMPONENT_SWIZZLE_ONE;
    case AmdGpu::CompSwizzle::Red: return VK_COMPONENT_SWIZZLE_R;
    case AmdGpu::CompSwizzle::Green: return VK_COMPONENT_SWIZZLE_G;
    case AmdGpu::CompSwizzle::Blue: return VK_COMPONENT_SWIZZLE_B;
    case AmdGpu::CompSwizzle::Alpha: return VK_COMPONENT_SWIZZLE_A;
    default: return VK_COMPONENT_SWIZZLE_IDENTITY;
    }
}

// T# DST_SEL_X/Y/Z/W for the image's view format (images hold the guest memory component order, render targets included).
AmdGpu::CompMapping DstSel(const u32* t) {
    AmdGpu::CompMapping sel{};
    sel.r = AmdGpu::CompSwizzle(t[3] & 7), sel.g = AmdGpu::CompSwizzle(t[3] >> 3 & 7), sel.b = AmdGpu::CompSwizzle(t[3] >> 6 & 7),
    sel.a = AmdGpu::CompSwizzle(t[3] >> 9 & 7);
    return AmdGpu::RemapSwizzle(AmdGpu::DataFormat(t[1] >> 20 & 0x3F), sel);
}
VkComponentMapping DstSelect(const u32* t) {
    const AmdGpu::CompMapping sel = DstSel(t);
    return {Swizzle(sel.r), Swizzle(sel.g), Swizzle(sel.b), Swizzle(sel.a)};
}

// Bindless array of a texture slot: the translator's dimension kinds (gcn_translator.cpp Kind()).
int DimKind(Gcn::Dim d) {
    switch (d) {
    case Gcn::Dim::D1: case Gcn::Dim::D2: return 0;
    case Gcn::Dim::D1Array: case Gcn::Dim::D2Array: return 1;
    case Gcn::Dim::D3: return 2;
    case Gcn::Dim::Cube: return 3;
    case Gcn::Dim::D2Ms: return 4;
    default: return 5;
    }
}

std::string_view TilingSource() {
    return R"GLSL(
// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 450 core

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_samplerless_texture_functions : require

layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// #define BITS_PER_PIXEL
// #define NUM_SAMPLES
// #define MICRO_TILE_MODE
// #define ARRAY_MODE
// #define MICRO_TILE_THICKNESS
// #define PIPE_CONFIG
// #define BANK_WIDTH
// #define BANK_HEIGHT
// #define NUM_BANKS
// #define NUM_BANK_BITS
// #define TILE_SPLIT_BYTES
// #define MACRO_TILE_ASPECT
// #define TILE_FROM_IMAGE: the tiler reads the image through a uint view of its texels, rather
// than a linear copy of it.

#define BYTES_PER_PIXEL (BITS_PER_PIXEL / 8)

#if BITS_PER_PIXEL == 8
#define BLOCK_TYPE uint8_t
#elif BITS_PER_PIXEL == 16
#define BLOCK_TYPE uint16_t
#elif BITS_PER_PIXEL == 32
#define BLOCK_TYPE uint32_t
#elif BITS_PER_PIXEL == 64
#define BLOCK_TYPE u32vec2
#elif BITS_PER_PIXEL == 96
#define BLOCK_TYPE u32vec3
#else
#define BLOCK_TYPE u32vec4
#endif

#if PIPE_CONFIG == ADDR_SURF_P2
#define NUM_PIPES 2
#define NUM_PIPE_BITS 1
#else
#define NUM_PIPES 8
#define NUM_PIPE_BITS 3
#endif

#define MICRO_TILE_WIDTH 8
#define MICRO_TILE_HEIGHT 8
#define MICRO_TILE_PIXELS (MICRO_TILE_WIDTH * MICRO_TILE_HEIGHT)
#define MICRO_TILE_BITS (MICRO_TILE_PIXELS * MICRO_TILE_THICKNESS * BITS_PER_PIXEL * NUM_SAMPLES)
#define MICRO_TILE_BYTES (MICRO_TILE_BITS / 8)

#define NUM_PIPE_INTERLEAVE_BITS 8

#define ADDR_SURF_DISPLAY_MICRO_TILING 0
#define ADDR_SURF_THIN_MICRO_TILING 1
#define ADDR_SURF_DEPTH_MICRO_TILING 2
#define ADDR_SURF_ROTATED_MICRO_TILING 3

#define ARRAY_LINEAR_GENERAL 0
#define ARRAY_LINEAR_ALIGNED 1
#define ARRAY_1D_TILED_THIN1 2
#define ARRAY_1D_TILED_THICK 3
#define ARRAY_2D_TILED_THIN1 4
#define ARRAY_PRT_TILED_THIN1 5
#define ARRAY_PRT_2D_TILED_THIN1 6
#define ARRAY_2D_TILED_THICK 7
#define ARRAY_2D_TILED_XTHICK 8
#define ARRAY_PRT_TILED_THICK 9
#define ARRAY_PRT_2D_TILED_THICK 10
#define ARRAY_PRT_3D_TILED_THIN1 11
#define ARRAY_3D_TILED_THIN1 12
#define ARRAY_3D_TILED_THICK 13
#define ARRAY_3D_TILED_XTHICK 14
#define ARRAY_PRT_3D_TILED_THICK 15

#define	ADDR_SURF_P2 0
#define	ADDR_SURF_P8_32x32_8x16	10
#define	ADDR_SURF_P8_32x32_16x16 12

#define BITS_PER_BYTE 8
#define BITS_TO_BYTES(x) (((x) + (BITS_PER_BYTE-1)) / BITS_PER_BYTE)

#define _BIT(v, b) bitfieldExtract((v), (b), 1)

struct MipInfo {
    uint size;
    uint pitch;
    uint height;
    uint offset;
};

layout (set = 0, binding = 0, scalar) buffer InputBuf {
    BLOCK_TYPE tiled_data[];
};

#ifdef TILE_FROM_IMAGE
layout (set = 0, binding = 3) uniform utexture2DArray source_image;

#if BITS_PER_PIXEL == 8
#define TEXEL_TO_BLOCK(v) uint8_t((v).x)
#elif BITS_PER_PIXEL == 16
#define TEXEL_TO_BLOCK(v) uint16_t((v).x)
#elif BITS_PER_PIXEL == 32
#define TEXEL_TO_BLOCK(v) (v).x
#elif BITS_PER_PIXEL == 64
#define TEXEL_TO_BLOCK(v) (v).xy
#else
#define TEXEL_TO_BLOCK(v) (v)
#endif
#elif defined(DETILE_TO_IMAGE)
#if BITS_PER_PIXEL == 8
#define IMAGE_FORMAT r8ui
#define BLOCK_TO_TEXEL(v) uvec4(uint(v), 0u, 0u, 0u)
#elif BITS_PER_PIXEL == 16
#define IMAGE_FORMAT r16ui
#define BLOCK_TO_TEXEL(v) uvec4(uint(v), 0u, 0u, 0u)
#elif BITS_PER_PIXEL == 32
#define IMAGE_FORMAT r32ui
#define BLOCK_TO_TEXEL(v) uvec4(uint(v), 0u, 0u, 0u)
#elif BITS_PER_PIXEL == 64
#define IMAGE_FORMAT rg32ui
#define BLOCK_TO_TEXEL(v) uvec4((v).x, (v).y, 0u, 0u)
#else
#define IMAGE_FORMAT rgba32ui
#define BLOCK_TO_TEXEL(v) (v)
#endif
layout (set = 0, binding = 4, IMAGE_FORMAT) uniform writeonly uimage2DArray dest_image;
#else
layout (set = 0, binding = 1, scalar) buffer OutputBuf {
    BLOCK_TYPE linear_data[];
};
#endif

layout (set = 0, binding = 2, scalar) uniform TilingInfo {
    uint bank_swizzle;
    uint num_slices;
    uint num_mips;
    MipInfo mips[16];
    // Extent of mip 0, for the reads from the image.
    uint image_width;
    uint image_height;
    // Tiled bytes the tiler writes, from the start of the image.
    uint range_begin;
    uint range_end;
    uint block_scale;
    uint is_volume;
} info;

layout (push_constant) uniform TilingRegion {
    uint mip;
    uint first_tile;
    uint num_tiles;
    uint tile_width;
    uint tile_height;
} region;

uint32_t ComputePixelIndexWithinMicroTile(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t p0 = 0;
    uint32_t p1 = 0;
    uint32_t p2 = 0;
    uint32_t p3 = 0;
    uint32_t p4 = 0;
    uint32_t p5 = 0;
    uint32_t p6 = 0;
    uint32_t p7 = 0;
    uint32_t p8 = 0;

    uint32_t x0 = _BIT(x, 0);
    uint32_t x1 = _BIT(x, 1);
    uint32_t x2 = _BIT(x, 2);
    uint32_t y0 = _BIT(y, 0);
    uint32_t y1 = _BIT(y, 1);
    uint32_t y2 = _BIT(y, 2);
    uint32_t z0 = _BIT(z, 0);
    uint32_t z1 = _BIT(z, 1);
    uint32_t z2 = _BIT(z, 2);

#if MICRO_TILE_MODE == ADDR_SURF_DISPLAY_MICRO_TILING
    #if BITS_PER_PIXEL == 8
        p0 = x0;
        p1 = x1;
        p2 = x2;
        p3 = y1;
        p4 = y0;
        p5 = y2;
    #elif BITS_PER_PIXEL == 16
        p0 = x0;
        p1 = x1;
        p2 = x2;
        p3 = y0;
        p4 = y1;
        p5 = y2;
    #elif BITS_PER_PIXEL == 32
        p0 = x0;
        p1 = x1;
        p2 = y0;
        p3 = x2;
        p4 = y1;
        p5 = y2;
    #elif BITS_PER_PIXEL == 64
        p0 = x0;
        p1 = y0;
        p2 = x1;
        p3 = x2;
        p4 = y1;
        p5 = y2;
    #elif BITS_PER_PIXEL == 128
        p0 = y0;
        p1 = x0;
        p2 = x1;
        p3 = x2;
        p4 = y1;
        p5 = y2;
    #endif
#elif MICRO_TILE_MODE == ADDR_SURF_THIN_MICRO_TILING || MICRO_TILE_MODE == ADDR_SURF_DEPTH_MICRO_TILING
        p0 = x0;
        p1 = y0;
        p2 = x1;
        p3 = y1;
        p4 = x2;
        p5 = y2;
#else
    #if BITS_PER_PIXEL == 8 || BITS_PER_PIXEL == 16
        p0 = x0;
        p1 = y0;
        p2 = x1;
        p3 = y1;
        p4 = z0;
        p5 = z1;
    #elif BITS_PER_PIXEL == 32
        p0 = x0;
        p1 = y0;
        p2 = x1;
        p3 = z0;
        p4 = y1;
        p5 = z1;
    #elif BITS_PER_PIXEL == 64 || BITS_PER_PIXEL == 128
        p0 = x0;
        p1 = y0;
        p2 = z0;
        p3 = x1;
        p4 = y1;
        p5 = z1;
    #endif
        p6 = x2;
        p7 = y2;

    #if MICRO_TILE_THICKNESS == 8
        p8 = z2;
    #endif
#endif

    uint32_t pixel_number =
        ((p0) | (p1 << 1) | (p2 << 2) | (p3 << 3) | (p4 << 4) |
         (p5 << 5) | (p6 << 6) | (p7 << 7) | (p8 << 8));

    return pixel_number;
}

uint32_t ComputeSurfaceAddrFromCoordMicroTiled(uint32_t x, uint32_t y, uint32_t slice, uint32_t pitch, uint32_t height, uint32_t sample_index) {
    uint32_t slice_bytes = BITS_TO_BYTES(pitch * height * MICRO_TILE_THICKNESS * BITS_PER_PIXEL * NUM_SAMPLES);

    uint32_t micro_tiles_per_row = pitch / MICRO_TILE_WIDTH;
    uint32_t micro_tile_index_x = x / MICRO_TILE_WIDTH;
    uint32_t micro_tile_index_y = y / MICRO_TILE_HEIGHT;
    uint32_t micro_tile_index_z = slice / MICRO_TILE_THICKNESS;

    uint32_t slice_offset = micro_tile_index_z * slice_bytes;
    uint32_t micro_tile_offset = (micro_tile_index_y * micro_tiles_per_row + micro_tile_index_x) * MICRO_TILE_BYTES;

    uint32_t pixel_index = ComputePixelIndexWithinMicroTile(x, y, slice);

    uint32_t sample_offset;
    uint32_t pixel_offset;
#if MICRO_TILE_MODE == ADDR_SURF_DEPTH_MICRO_TILING
    sample_offset = sample_index * BITS_PER_PIXEL;
    pixel_offset = pixel_index * BITS_PER_PIXEL * NUM_SAMPLES;
#else
    sample_offset = sample_index * (MICRO_TILE_BYTES * 8 / NUM_SAMPLES);
    pixel_offset = pixel_index * BITS_PER_PIXEL;
#endif

    uint32_t elem_offset = (sample_offset + pixel_offset) / 8;
    return slice_offset + micro_tile_offset + elem_offset;
}
#if ARRAY_MODE != ARRAY_1D_TILED_THIN1 && ARRAY_MODE != ARRAY_1D_TILED_THICK
uint32_t ComputePipeFromCoord(uint32_t x, uint32_t y, uint32_t slice) {
    uint32_t p0 = 0;
    uint32_t p1 = 0;
    uint32_t p2 = 0;

    uint32_t tx = x / MICRO_TILE_WIDTH;
    uint32_t ty = y / MICRO_TILE_HEIGHT;
    uint32_t x3 = _BIT(tx, 0);
    uint32_t x4 = _BIT(tx, 1);
    uint32_t x5 = _BIT(tx, 2);
    uint32_t y3 = _BIT(ty, 0);
    uint32_t y4 = _BIT(ty, 1);
    uint32_t y5 = _BIT(ty, 2);

#if PIPE_CONFIG == ADDR_SURF_P2
    p0 = x3 ^ y3;
#elif PIPE_CONFIG == ADDR_SURF_P8_32x32_8x16
    p0 = x4 ^ y3 ^ x5;
    p1 = x3 ^ y4;
    p2 = x5 ^ y5;
#elif PIPE_CONFIG == ADDR_SURF_P8_32x32_16x16
    p0 = x3 ^ y3 ^ x4;
    p1 = x4 ^ y4;
    p2 = x5 ^ y5;
#endif

    uint32_t pipe = p0 | (p1 << 1) | (p2 << 2);

    uint32_t pipe_swizzle = 0;
#if ARRAY_MODE == ARRAY_3D_TILED_THIN1 || ARRAY_MODE == ARRAY_3D_TILED_THICK || ARRAY_MODE == ARRAY_3D_TILED_XTHICK
    pipe_swizzle += max(1, NUM_PIPES / 2 - 1) * (slice / MICRO_TILE_THICKNESS);
#endif
    pipe_swizzle &= (NUM_PIPES - 1);
    pipe = pipe ^ pipe_swizzle;
    return pipe;
}

uint32_t ComputeBankFromCoord(uint32_t x, uint32_t y, uint32_t slice, uint32_t tile_split_slice) {
    uint32_t b0 = 0;
    uint32_t b1 = 0;
    uint32_t b2 = 0;
    uint32_t b3 = 0;
    uint32_t slice_rotation = 0;
    uint32_t tile_split_rotation = 0;

    uint32_t tx = x / MICRO_TILE_WIDTH / (BANK_WIDTH * NUM_PIPES);
    uint32_t ty = y / MICRO_TILE_HEIGHT / BANK_HEIGHT;

    uint32_t x3 = _BIT(tx, 0);
    uint32_t x4 = _BIT(tx, 1);
    uint32_t x5 = _BIT(tx, 2);
    uint32_t x6 = _BIT(tx, 3);
    uint32_t y3 = _BIT(ty, 0);
    uint32_t y4 = _BIT(ty, 1);
    uint32_t y5 = _BIT(ty, 2);
    uint32_t y6 = _BIT(ty, 3);

#if NUM_BANKS == 16
    b0 = x3 ^ y6;
    b1 = x4 ^ y5 ^ y6;
    b2 = x5 ^ y4;
    b3 = x6 ^ y3;
#elif NUM_BANKS == 8
    b0 = x3 ^ y5;
    b1 = x4 ^ y4 ^ y5;
    b2 = x5 ^ y3;
#elif NUM_BANKS == 4
    b0 = x3 ^ y4;
    b1 = x4 ^ y3;
#elif NUM_BANKS == 2
    b0 = x3 ^ y3;
#endif

    uint32_t bank = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3);

#if ARRAY_MODE == ARRAY_2D_TILED_THIN1 || ARRAY_MODE == ARRAY_2D_TILED_THICK || ARRAY_MODE == ARRAY_2D_TILED_XTHICK
    slice_rotation = ((NUM_BANKS / 2) - 1) * (slice / MICRO_TILE_THICKNESS);
#elif ARRAY_MODE == ARRAY_3D_TILED_THIN1 || ARRAY_MODE == ARRAY_3D_TILED_THICK || ARRAY_MODE == ARRAY_3D_TILED_XTHICK
    slice_rotation = max(1u, (NUM_PIPES / 2) - 1) * (slice / MICRO_TILE_THICKNESS) / NUM_PIPES;
#endif

#if ARRAY_MODE == ARRAY_2D_TILED_THIN1 || ARRAY_MODE == ARRAY_3D_TILED_THIN1 || \
    ARRAY_MODE == ARRAY_PRT_2D_TILED_THIN1 || ARRAY_MODE == ARRAY_PRT_3D_TILED_THIN1
                                                                tile_split_rotation = ((NUM_BANKS / 2) + 1) * tile_split_slice;
#endif

    bank ^= info.bank_swizzle + slice_rotation;
    bank ^= tile_split_rotation;
    bank &= (NUM_BANKS - 1);

    return bank;
}

uint32_t ComputeSurfaceAddrFromCoordMacroTiled(uint32_t x, uint32_t y, uint32_t slice, uint32_t pitch, uint32_t height, uint32_t sample_index) {
    uint32_t pixel_index = ComputePixelIndexWithinMicroTile(x, y, slice);

    uint32_t sample_offset;
    uint32_t pixel_offset;
#if MICRO_TILE_MODE == ADDR_SURF_DEPTH_MICRO_TILING
    sample_offset = sample_index * BITS_PER_PIXEL;
    pixel_offset = pixel_index * BITS_PER_PIXEL * NUM_SAMPLES;
#else
    sample_offset = sample_index * (MICRO_TILE_BITS / NUM_SAMPLES);
    pixel_offset = pixel_index * BITS_PER_PIXEL;
#endif

    uint32_t element_offset = (pixel_offset + sample_offset) / 8;

    uint32_t slices_per_tile = 1;
    uint32_t tile_split_slice = 0;
#if MICRO_TILE_BYTES > TILE_SPLIT_BYTES && MICRO_TILE_THICKNESS == 1
    slices_per_tile = MICRO_TILE_BYTES / TILE_SPLIT_BYTES;
    tile_split_slice = element_offset / TILE_SPLIT_BYTES;
    element_offset %= TILE_SPLIT_BYTES;
    #undef MICRO_TILE_BYTES
    #define MICRO_TILE_BYTES TILE_SPLIT_BYTES
#endif

    uint32_t macro_tile_pitch = (MICRO_TILE_WIDTH * BANK_WIDTH * NUM_PIPES) * MACRO_TILE_ASPECT;
    uint32_t macro_tile_height = (MICRO_TILE_HEIGHT * BANK_HEIGHT * NUM_BANKS) / MACRO_TILE_ASPECT;

    uint32_t macro_tile_bytes = MICRO_TILE_BYTES *
                               (macro_tile_pitch / MICRO_TILE_WIDTH) *
                               (macro_tile_height / MICRO_TILE_HEIGHT) / (NUM_PIPES * NUM_BANKS);

    uint32_t macro_tiles_per_row = pitch / macro_tile_pitch;

    uint32_t macro_tile_index_x = x / macro_tile_pitch;
    uint32_t macro_tile_index_y = y / macro_tile_height;
    uint32_t macro_tile_offset =
        ((macro_tile_index_y * macro_tiles_per_row) + macro_tile_index_x) * macro_tile_bytes;
    uint32_t macro_tiles_per_slice = macro_tiles_per_row * (height / macro_tile_height);

    uint32_t slice_bytes = macro_tiles_per_slice * macro_tile_bytes;
    uint32_t slice_offset =
        slice_bytes * (tile_split_slice + slices_per_tile * (slice / MICRO_TILE_THICKNESS));

    uint32_t tile_row_index = (y / MICRO_TILE_HEIGHT) % BANK_HEIGHT;
    uint32_t tile_column_index = ((x / MICRO_TILE_WIDTH) / NUM_PIPES) % BANK_WIDTH;
    uint32_t tile_index = (tile_row_index * BANK_WIDTH) + tile_column_index;
    uint32_t tile_offset = tile_index * MICRO_TILE_BYTES;

    uint32_t total_offset = slice_offset + macro_tile_offset + element_offset + tile_offset;

#if ARRAY_MODE == ARRAY_PRT_TILED_THIN1 || ARRAY_MODE == ARRAY_PRT_TILED_THICK || \
    ARRAY_MODE == ARRAY_PRT_2D_TILED_THIN1 || ARRAY_MODE == ARRAY_PRT_2D_TILED_THICK || \
    ARRAY_MODE == ARRAY_PRT_3D_TILED_THIN1 || ARRAY_MODE == ARRAY_PRT_3D_TILED_THICK
    x %= macro_tile_pitch;
    y %= macro_tile_height;
#endif

    uint32_t pipe = ComputePipeFromCoord(x, y, slice);
    uint32_t bank = ComputeBankFromCoord(x, y, slice, tile_split_slice);

    uint32_t pipe_interleave_mask = (1 << NUM_PIPE_INTERLEAVE_BITS) - 1;
    uint32_t pipe_interleave_offset = total_offset & pipe_interleave_mask;
    uint32_t offset = total_offset >> NUM_PIPE_INTERLEAVE_BITS;

    uint32_t addr = pipe_interleave_offset;
    uint32_t pipe_bits = pipe << NUM_PIPE_INTERLEAVE_BITS;
    uint32_t bank_bits = bank << (NUM_PIPE_INTERLEAVE_BITS + NUM_PIPE_BITS);
    uint32_t offset_bits = offset << (NUM_PIPE_INTERLEAVE_BITS + NUM_PIPE_BITS + NUM_BANK_BITS);

    addr |= pipe_bits;
    addr |= bank_bits;
    addr |= offset_bits;

    return addr;
}
#endif

void main() {
    uint mip = region.mip;
    uint pitch = info.mips[mip].pitch;
    uint height = info.mips[mip].height;
    uint tiled_offset = info.mips[mip].offset;
    uint x;
    uint y;
    uint slice;
    uint texel = region.first_tile * 64u + gl_GlobalInvocationID.x;
    if (region.tile_width != 0u) {
        uint pixels_per_tile = region.tile_width * region.tile_height;
        uint tile = region.first_tile + gl_GlobalInvocationID.x / (pixels_per_tile * MICRO_TILE_THICKNESS);
        uint pixel = gl_GlobalInvocationID.x % (pixels_per_tile * MICRO_TILE_THICKNESS);
        uint tiles_x = pitch / region.tile_width;
        uint tiles_y = height / region.tile_height;
        uint block = (pixel % pixels_per_tile) / 64u;
        uint blocks_x = region.tile_width / 8u;
        x = (tile % tiles_x) * region.tile_width + (block % blocks_x) * 8u + (pixel & 7u);
        y = ((tile / tiles_x) % tiles_y) * region.tile_height + (block / blocks_x) * 8u + ((pixel / 8u) & 7u);
        slice = (tile / (tiles_x * tiles_y)) * MICRO_TILE_THICKNESS + pixel / pixels_per_tile;
    } else if ((pitch & 7u) == 0u && (height & 7u) == 0u) {
        // Each 64 threads take an 8x8 block, so a wave touches whole micro tiles on the tiled
        // side instead of a few bytes of many.
        uint block = texel >> 6u;
        uint blocks_x = pitch >> 3u;
        uint blocks_y = height >> 3u;
        uint rest = block / blocks_x;
        x = (block - rest * blocks_x) * 8u + (texel & 7u);
        uint by = rest / blocks_y;
        y = (rest - by * blocks_y) * 8u + ((texel >> 3u) & 7u);
        slice = by;
    } else {
        uint row = texel / pitch;
        x = texel - row * pitch;
        slice = row / height;
        y = row - slice * height;
    }
    uint linear_texel = info.mips[mip].offset / BYTES_PER_PIXEL + (slice * height + y) * pitch + x;
    if ((slice * height + y) * pitch + x >= info.mips[mip].size / BYTES_PER_PIXEL) {
        return;
    }

#if ARRAY_MODE == ARRAY_1D_TILED_THIN1 || ARRAY_MODE == ARRAY_1D_TILED_THICK
    tiled_offset += ComputeSurfaceAddrFromCoordMicroTiled(x, y, slice, pitch, height, 0);
#else
    if (pitch < (MICRO_TILE_WIDTH * BANK_WIDTH * NUM_PIPES) * MACRO_TILE_ASPECT ||
        height < (MICRO_TILE_HEIGHT * BANK_HEIGHT * NUM_BANKS) / MACRO_TILE_ASPECT) {
        tiled_offset += ComputeSurfaceAddrFromCoordMicroTiled(x, y, slice, pitch, height, 0);
    } else {
        tiled_offset += ComputeSurfaceAddrFromCoordMacroTiled(x, y, slice, pitch, height, 0);
    }
#endif

#if defined(IS_TILER) || defined(DETILE_TO_IMAGE)
    uint width = (max(info.image_width >> mip, 1u) + info.block_scale - 1u) / info.block_scale;
    uint image_height = (max(info.image_height >> mip, 1u) + info.block_scale - 1u) / info.block_scale;
    uint slices = max(info.num_slices >> (info.is_volume != 0u ? mip : 0u), 1u);
    if (x >= width || y >= image_height || slice >= slices) {
        return;
    }
#endif
#ifdef IS_TILER
    if (tiled_offset + BYTES_PER_PIXEL <= info.range_begin || tiled_offset >= info.range_end) {
        return;
    }
#ifdef TILE_FROM_IMAGE
    uvec4 value = texelFetch(source_image, ivec3(x, y, slice), 0);
    tiled_data[tiled_offset / BYTES_PER_PIXEL] = TEXEL_TO_BLOCK(value);
#else
    tiled_data[tiled_offset / BYTES_PER_PIXEL] = linear_data[linear_texel];
#endif
#elif defined(DETILE_TO_IMAGE)
    imageStore(dest_image, ivec3(x, y, slice), BLOCK_TO_TEXEL(tiled_data[tiled_offset / BYTES_PER_PIXEL]));
#else
    linear_data[linear_texel] = tiled_data[tiled_offset / BYTES_PER_PIXEL];
#endif
}
)GLSL";
}


// ---- Bindless set, image memory and texture lifecycle ----

// Detiler (tiling.comp, embedded above): descriptor layout and pipeline layout; the bindless set the programs sample
// textures, samplers and storage images through; its sampler and the 1x1 dummy images in every slot 0.
bool Renderer::Impl::SetupTextures() {
    // Detiler (tiling.comp, embedded in textures.cpp): bindings 0 tiled (mirror), 1 linear (scratch), 2 TilingInfo, 3 source_image, 4 dest_image; TilingRegion.
    tiling_src = TilingSource();
    const VkDescriptorSetLayoutBinding tb[]{{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                                            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                                            {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                                            {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                                            {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutCreateInfo tci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr,
                                        VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, 5, tb};
    vkCreateDescriptorSetLayout(dev, &tci, nullptr, &tiling_set);
    VkPushConstantRange tpc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 20};
    VkPipelineLayoutCreateInfo tpl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &tiling_set, 1, &tpc};
    vkCreatePipelineLayout(dev, &tpl, nullptr, &tiling_layout);
    std::fill(std::begin(next_element), std::end(next_element), 1u);
    // Bindless set: sampled images by (kind, class), samplers, storage images by (kind, class).
    std::vector<VkDescriptorSetLayoutBinding> b;
    b.reserve(28);
    for (u32 i = 0; i < 18; ++i) b.push_back({i, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kTextures, VK_SHADER_STAGE_ALL});
    b.push_back({kSamplerBinding, VK_DESCRIPTOR_TYPE_SAMPLER, kSamplers, VK_SHADER_STAGE_ALL});
    for (u32 i = 0; i < 9; ++i) b.push_back({kStorageBinding + i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kStorage, VK_SHADER_STAGE_ALL});
    std::vector<VkDescriptorBindingFlags> bf(b.size(), VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT);
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, nullptr, (u32)bf.size(), bf.data()};
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &bfi, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                                       (u32)b.size(), b.data()};
    const VkResult temp_layout = vkCreateDescriptorSetLayout(dev, &lci, nullptr, &set_layout);
    VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 18 * kTextures}, {VK_DESCRIPTOR_TYPE_SAMPLER, kSamplers},
                                 {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 9 * kStorage}};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, 1, 3, sizes};
    const VkResult temp_pool = vkCreateDescriptorPool(dev, &dpci, nullptr, &pool);
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool, 1, &set_layout};
    const VkResult temp_set = vkAllocateDescriptorSets(dev, &dsai, &set);
    VkDescriptorSetLayoutSupport sup{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT};  // TEMP diagnostics
    vkGetDescriptorSetLayoutSupport(dev, &lci, &sup);
    if (temp_layout || temp_pool || temp_set || !sup.supported)
        Report(std::format("Bindless set: layout {} pool {} set {} (VkResult), supported by the device {}", (int)temp_layout, (int)temp_pool, (int)temp_set, sup.supported));
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR, sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR, sci.maxLod = VK_LOD_CLAMP_NONE;
    vkCreateSampler(dev, &sci, nullptr, &sampler);

    // Slot 0 of every binding: a 1x1 white image (phase 3 has no guest textures yet).
    static constexpr VkFormat cls_fmt[]{VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT};
    struct Kind {
        VkImageType type;
        VkImageViewType view;
        u32 layers;
        VkSampleCountFlagBits samples;
    };
    static constexpr Kind kinds[]{{VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_1_BIT},
                                  {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT},
                                  {VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D, 1, VK_SAMPLE_COUNT_1_BIT},
                                  {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE, 6, VK_SAMPLE_COUNT_1_BIT},
                                  {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_4_BIT},
                                  {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_4_BIT}};
    std::vector<VkDescriptorImageInfo> infos;
    std::vector<VkWriteDescriptorSet> writes;
    infos.reserve(32);
    for (u32 k = 0; k < 6; ++k)
        for (u32 c = 0; c < 3; ++c) {
            const bool storage = k < 3 && kinds[k].samples == VK_SAMPLE_COUNT_1_BIT;
            auto t = MakeImage(kinds[k].type, kinds[k].view, cls_fmt[c], 1, 1, 1, kinds[k].layers, kinds[k].samples,
                               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0),
                               VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, true);
            if (!t.img) return false;
            dummies.push_back(t);
            infos.push_back({VK_NULL_HANDLE, t.view, VK_IMAGE_LAYOUT_GENERAL});
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, k * 3 + c, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &infos.back()});
            if (storage)
                writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, kStorageBinding + k * 3 + c, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                  &infos.back()});
        }
    infos.push_back({sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, kSamplerBinding, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER, &infos.back()});
    vkUpdateDescriptorSets(dev, (u32)writes.size(), writes.data(), 0, nullptr);
    Once([&](VkCommandBuffer c) {
        VkClearColorValue white{};
        for (size_t i = 0; i < dummies.size(); ++i) {
            ToGeneral(c, dummies[i].img, VK_IMAGE_ASPECT_COLOR_BIT);
            if (i % 3 == 0) white.float32[0] = white.float32[1] = white.float32[2] = white.float32[3] = 1.0f;
            else white.uint32[0] = white.uint32[1] = white.uint32[2] = white.uint32[3] = 1;
            VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS};
            vkCmdClearColorImage(c, dummies[i].img, VK_IMAGE_LAYOUT_GENERAL, &white, 1, &r);
        }
    });
    return true;
}

void Renderer::Impl::DestroyTextures() {
    auto free_tex = [&](Tex& t) { FreeImage(t); };
    for (auto& [k, t] : targets) {
        for (auto v : t.owned_views) vkDestroyImageView(dev, v, nullptr);
        for (auto& [f, v] : t.attach)
            if (v != t.t.view) vkDestroyImageView(dev, v, nullptr);
        free_tex(t.t);
        if (t.stencil_copy.img) free_tex(t.stencil_copy);
        if (t.storage_copy.img) free_tex(t.storage_copy);
    }
    for (auto& t : dummies) free_tex(t);
    for (auto& [f, t] : probe_img) free_tex(t);
    for (auto& [k, s] : surfaces) {
        for (auto& [vk, v] : s.views) vkDestroyImageView(dev, v.view, nullptr);
        s.views.clear();
    }
    for (auto& [k, g] : textures) {
        for (auto v : g.owned) vkDestroyImageView(dev, v, nullptr);
        for (auto& [vk, v] : g.views) vkDestroyImageView(dev, v.view, nullptr);
        if (g.surf) g.shared.reset();
        else if (g.t.img) free_tex(g.t);
    }
    for (auto v : owned_views) vkDestroyImageView(dev, v, nullptr);
    for (auto s : owned_samplers) vkDestroySampler(dev, s, nullptr);
    for (auto& [k, p] : detilers)
        if (p) vkDestroyPipeline(dev, p, nullptr);
    vkDestroyPipelineLayout(dev, tiling_layout, nullptr);
    vkDestroyDescriptorSetLayout(dev, tiling_set, nullptr);
    vkDestroySampler(dev, sampler, nullptr);
    vkDestroyDescriptorPool(dev, pool, nullptr), vkDestroyDescriptorSetLayout(dev, set_layout, nullptr);
    for (auto* f : {&reaping, &image_growing})
        if (f->valid()) f->wait();
    for (auto& b : image_blocks) vkFreeMemory(dev, b.mem, nullptr);
    image_blocks.clear();
}

Renderer::Impl::Tex Renderer::Impl::MakeImage(VkImageType type, VkImageViewType vtype, VkFormat fmt, u32 w, u32 h, u32 depth, u32 layers,
                                              VkSampleCountFlagBits samples, VkImageUsageFlags usage, VkImageAspectFlags aspect, u32 levels,
                                              VkImageCreateFlags flags, bool lasting) {
    TempTimer temp_timer{temp_make_ns, temp_makes};
    Tex t{.usage = usage};
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.flags = flags | (vtype == VK_IMAGE_VIEW_TYPE_CUBE ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0);
    ci.imageType = type, ci.format = fmt, ci.extent = {w, h, depth}, ci.mipLevels = levels, ci.arrayLayers = layers, ci.samples = samples;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL, ci.usage = usage, ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (const VkResult e = vkCreateImage(dev, &ci, nullptr, &t.img); e != VK_SUCCESS)
        return Report(std::format("vkCreateImage {}: format {} type {} {}x{}x{} layers {} levels {} samples {} usage {:x} flags {:x}", (int)e, (int)fmt, (int)type,
                                  w, h, depth, layers, levels, (int)samples, (u32)usage, (u32)ci.flags)),
               Tex{};
    VkMemoryRequirements r;
    vkGetImageMemoryRequirements(dev, t.img, &r);
    ImageMemory(r, t, lasting);
    if (!t.mem) {
        vkDestroyImage(dev, t.img, nullptr);
        return Tex{};
    }
    if (const VkResult e = vkBindImageMemory(dev, t.img, t.mem, t.mem_offset); e != VK_SUCCESS) {
        Report(std::format("vkBindImageMemory {}: {}x{}x{} fmt {}", (int)e, w, h, depth, (int)fmt));
        FreeImage(t);
        return Tex{};
    }
    t.view = View(t.img, vtype, fmt, {aspect, 0, levels, 0, layers}, {}, usage);
    if (!t.view) {
        FreeImage(t);
        return Tex{};
    }
    return t;
}

void Renderer::Impl::ImageMemory(const VkMemoryRequirements& r, Tex& t, bool lasting) {
    const u32 type = Type(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (r.size <= image_block_size / 4 && type != ~0u) {
        std::scoped_lock lock(image_lock);
        if (image_free[lasting] < image_block_size / 2 && (!image_growing.valid() || image_growing.wait_for(std::chrono::seconds(0)) == std::future_status::ready))
            image_growing = std::async(std::launch::async, [this, bits = r.memoryTypeBits, lasting] {
                const VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, image_block_size, DeviceType(bits, image_block_size)};
                VkDeviceMemory m = VK_NULL_HANDLE;
                if (vkAllocateMemory(dev, &ai, nullptr, &m) != VK_SUCCESS) return alloc_pressure.store(true, std::memory_order_relaxed);
                std::scoped_lock lock(image_lock);
                image_blocks.push_back({m, ai.memoryTypeIndex, {{0, image_block_size}}, lasting}), image_free[lasting] += image_block_size;
            });
        for (auto& b : image_blocks) {
            if (b.lasting != lasting || b.draining || !(r.memoryTypeBits >> b.type & 1)) continue;
            for (auto it = b.free.begin(); it != b.free.end(); ++it) {
                const auto [lo, n] = *it;
                const VkDeviceSize at = (lo + r.alignment - 1) / r.alignment * r.alignment;
                if (at + r.size > lo + n) continue;
                b.free.erase(it);
                if (at > lo) b.free.emplace(lo, at - lo);
                if (lo + n > at + r.size) b.free.emplace(at + r.size, lo + n - at - r.size);
                image_free[lasting] -= r.size, t.mem = b.mem, t.mem_offset = at, t.mem_size = r.size;
                return;
            }
        }
    }
    t.mem = Alloc(r, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
    if (t.mem) t.alloc_size = r.size, dedicated_images += r.size;
}

void Renderer::Impl::FreeImage(const Tex& t) {
    vkDestroyImageView(dev, t.view, nullptr), vkDestroyImage(dev, t.img, nullptr);
    if (!t.mem_size) return dedicated_images -= t.alloc_size, vkFreeMemory(dev, t.mem, nullptr);
    VkDeviceMemory empty = VK_NULL_HANDLE;
    {
        std::scoped_lock lock(image_lock);
        for (auto block = image_blocks.begin(); block != image_blocks.end(); ++block) {
            auto& b = *block;
            if (b.mem != t.mem) continue;
            auto it = b.free.emplace(t.mem_offset, t.mem_size).first;
            if (auto next = std::next(it); next != b.free.end() && it->first + it->second == next->first) it->second += next->second, b.free.erase(next);
            if (auto prev = it; it != b.free.begin() && (--prev)->first + prev->second == it->first) prev->second += it->second, b.free.erase(it);
            if (!b.draining) image_free[b.lasting] += t.mem_size;
            // A block free as a whole goes back to the driver (a level load's peak otherwise stays allocated for good),
            // keeping image_reserve_blocks of room for the next textures (none under memory pressure); a lasting or draining one always goes.
            if (b.free.size() == 1 && b.free.begin()->second == image_block_size &&
                (b.lasting || b.draining || image_free[0] - image_block_size >= (Pressed() ? 0 : image_reserve_blocks) * image_block_size))
                empty = b.mem, image_free[b.lasting] -= (b.draining ? 0 : image_block_size), image_blocks.erase(block);
            break;
        }
    }
    if (empty) vkFreeMemory(dev, empty, nullptr);  // a kernel call: outside the lock
}

bool Renderer::Impl::ReclaimEmptyImageBlocks() {
    std::vector<VkDeviceMemory> empty;
    {
        std::scoped_lock lock(image_lock);
        for (auto it = image_blocks.begin(); it != image_blocks.end();) {
            if (it->free.size() == 1 && it->free.begin()->second == image_block_size) {
                empty.push_back(it->mem);
                if (!it->draining) image_free[it->lasting] -= image_block_size;
                it = image_blocks.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (VkDeviceMemory m : empty) vkFreeMemory(dev, m, nullptr);
    return !empty.empty();
}

// `usage`: the image's, when it is a storage image too: a view in a format without storage support (SRGB) leaves it out.
VkImageView Renderer::Impl::View(VkImage img, VkImageViewType vtype, VkFormat fmt, VkImageSubresourceRange range, VkComponentMapping map,
                                 VkImageUsageFlags usage) {
    const VkImageViewUsageCreateInfo ui{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, nullptr, usage & ~VK_IMAGE_USAGE_STORAGE_BIT};
    VkFormatProperties fp{};
    if (usage & VK_IMAGE_USAGE_STORAGE_BIT) vkGetPhysicalDeviceFormatProperties(pd, fmt, &fp);
    const bool narrow = usage & VK_IMAGE_USAGE_STORAGE_BIT && !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, narrow ? &ui : nullptr, 0, img, vtype, fmt, map, range};
    VkImageView v = VK_NULL_HANDLE;
    vkCreateImageView(dev, &vi, nullptr, &v);
    return v;
}

// Next free element of a bindless array (element 0 holds the dummy image / default sampler).
u32 Renderer::Impl::Bindless(u32 binding, VkImageView view, VkSampler smp) {
    const bool storage = binding >= kStorageBinding, is_sampler = binding == kSamplerBinding;
    u32& n = next_element[binding];
    auto& freed = free_elements[binding];
    if (freed.empty() && n >= (is_sampler ? kSamplers : storage ? kStorage : kTextures)) {
        // Bursts of new images (texture-pool defrag moves): free everything this frame has not used.
        if (!is_sampler && !in_worker)
            Retire(), Stalled(stall_evicts, stall_evict_ns, [&] {
                if (freed.empty()) EvictTextures(0);
            });
        else if (!is_sampler) evict_soon = true, evict_binding = binding;
        if (freed.empty()) return Note(std::format("bindless array {} full", binding)), 0;
    }
    u32 e = n;
    if (freed.empty()) ++n;
    else e = freed.back(), freed.pop_back();
    VkDescriptorImageInfo ii{smp, view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, binding, e, 1,
                           is_sampler ? VK_DESCRIPTOR_TYPE_SAMPLER : storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ii};
    vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
    return e;
}

// Guest textures unused for more than `age` frames: bindless elements freed (no frame samples them), views and images
// destroyed by Reap once the frame being recorded has retired (it and the frame in flight may still copy from them).
void Renderer::Impl::EvictTextures(u32 age, bool defer_elements) {
    const auto raw_old = [&](const GuestTex& g) { return g.checked == ~0u || frame - g.checked > age || std::binary_search(doomed.begin(), doomed.end(), &g); };
    auto by_cpu = [&](const GpuWrite& w) {
        auto c = std::upper_bound(dirty.begin(), dirty.end(), std::pair{w.hi, 0ull});
        return c != dirty.begin() && std::prev(c)->second > w.lo;
    };
    std::unordered_set<const GuestTex*> keep_lazy;
    for (const auto& p : prologue_lazy)
        if (p.first.by) keep_lazy.insert(p.first.by);
    // Materialize expiring lazy writes at the frame's prologue instead of discarding GPU contents; delay evicting their owner by one pass.
    for (const GpuWrite& w : lazy_writes)
        if (w.by && raw_old(*w.by)) {
            if (!by_cpu(w)) QueuePrologueLazy(w, false), keep_lazy.insert(w.by);
            else if (w.by->surf) w.by->surf->foreign = true;
        }
    std::erase_if(lazy_writes, [&](const GpuWrite& w) { return w.by && raw_old(*w.by) && by_cpu(w); });
    const auto old = [&](const GuestTex& g) { return !keep_lazy.contains(&g) && raw_old(g); };
    std::vector<const GuestTex*> gone;
    for (auto& [h, g] : textures)
        if (old(g)) gone.push_back(&g);
    std::sort(gone.begin(), gone.end());
    const auto dead = [&](const GuestTex* g) { return std::binary_search(gone.begin(), gone.end(), g); };
    for (auto* list : {&gpu_writes, &last_writes})
        for (auto& w : *list)
            if (dead(w.by)) w.by = nullptr;
    std::erase_if(lazy_writes, [&](const GpuWrite& w) { return dead(w.by); });
    std::erase_if(prologue_lazy, [&](const auto& w) { return dead(w.first.by); });
    std::erase_if(textures, [&](auto& e) {
        GuestTex& g = e.second;
        if (!old(g)) return false;
        graves.views.insert(graves.views.end(), g.owned.begin(), g.owned.end());
        for (auto& [k, v] : g.views) Bury(k, v, defer_elements);
        if (g.surf) --g.surf->users, graves.shared.push_back(std::move(g.shared));
        else if (g.t.img) graves.own.push_back(g.t);
        return true;
    });
    doomed.clear();
    if (!gone.empty()) std::atomic_ref(epoch).fetch_add(1, std::memory_order_release), PruneSurfaceViews(defer_elements);
    std::erase_if(surfaces, [](auto& s) { return !s.second.users; });
    // Targets, at a frame's start only (never age 0: a frame being recorded holds its targets). A long age: a target's
    // content, unlike a texture's, cannot be loaded again. Under the lock: FindTarget marks the displayed one used.
    if (std::scoped_lock lock(targets_lock); age && std::erase_if(targets, [&](auto& e) {
            Target& t = e.second;
            if (frame - std::atomic_ref(t.used).load(std::memory_order_relaxed) <= 512) return false;
            std::erase_if(stencil_at, [&](const auto& s) { return s.second.first == &t; });
            graves.views.insert(graves.views.end(), t.owned_views.begin(), t.owned_views.end());
            for (auto& [f, v] : t.attach)
                if (v && v != t.t.view) graves.views.push_back(v);
            for (auto [key, el] : t.views)
                if (el) free_elements[key >> 44 & 0xFF].push_back(el);
            if (t.t.img) graves.own.push_back(t.t);
            if (t.stencil_copy.img) graves.own.push_back(t.stencil_copy);
            if (t.storage_copy.img) graves.own.push_back(t.storage_copy);
            return true;
        })) {
        std::erase_if(target_at, [&](u64 base) {
            bool drawn = false;
            TargetsAt(base, [&](Target& t) { drawn |= std::atomic_ref(t.rendered).load(std::memory_order_relaxed) != ~0u; });
            return !drawn;
        });
        std::atomic_ref(epoch).fetch_add(1, std::memory_order_release);
    }
}

void Renderer::Impl::CompactImageBlocks() {
    // Integrated GPUs, or a discrete one at its VRAM budget: otherwise a discrete GPU's idle blocks take no system RAM.
    if (!uma_single_heap && !uma_budget && !budget_pressure) return;
    constexpr u64 kBudget = 32ull << 20;  // texture bytes moved per call
    const bool pressed = Pressed();
    const auto idle = [](const ImageBlock& b) { VkDeviceSize n = 0; for (auto& [lo, s] : b.free) n += s; return n; };
    const auto candidate = [&](const ImageBlock& b) { const VkDeviceSize used = image_block_size - idle(b); return !b.lasting && !b.draining && used > 0 && used <= image_block_size / 4; };
    {
        std::scoped_lock lock(image_lock);
        if (image_free[0] < (pressed ? 2 : 4) * image_block_size || !std::any_of(image_blocks.begin(), image_blocks.end(), candidate)) return;
    }
    // Textures whose content may exist only in their image (a GPU write not in guest memory yet) stay.
    std::unordered_set<const GuestTex*> written;
    for (auto* list : {&gpu_writes, &last_writes, &lazy_writes})
        for (auto& w : *list)
            if (w.by) written.insert(w.by);
    for (auto& p : prologue_lazy)
        if (p.first.by) written.insert(p.first.by);
    // A surface's image moves only if none of its T#s is written: they all move with it.
    std::unordered_map<const Surface*, bool> surf_ok;
    for (auto& [h, g] : textures)
        if (g.surf) surf_ok.try_emplace(g.surf, !g.surf->growing).first->second &= !written.contains(&g);
    std::unordered_map<VkDeviceMemory, u64> movable;  // bytes of the images that can move, per block memory
    std::unordered_set<VkDeviceMemory> hot_blocks;    // blocks with recently checked textures (kept unless pressed or <= 1/8 full)
    std::unordered_set<VkImage> counted;              // a surface's image counts once
    for (auto& [h, g] : textures) {
        if (!pressed && g.t.img && g.t.mem_size && g.checked != ~0u && frame - g.checked <= 16)
            hot_blocks.insert(g.t.mem);
        if (g.t.img && g.t.mem_size && (g.surf ? surf_ok[g.surf] : !written.contains(&g)) && counted.insert(g.t.img).second)
            movable[g.t.mem] += g.t.mem_size;
    }
    std::vector<VkDeviceMemory> taken;
    {
        std::scoped_lock lock(image_lock);
        std::vector<std::pair<VkDeviceSize, ImageBlock*>> picks;  // emptiest first
        for (auto& b : image_blocks) {
            const VkDeviceSize used = image_block_size - idle(b);
            if (candidate(b) && movable[b.mem] == used && (pressed || used <= image_block_size / 8 || !hot_blocks.contains(b.mem)))
                picks.push_back({used, &b});  // every byte in it can move: it really empties
        }
        std::sort(picks.begin(), picks.end(), [](auto& x, auto& y) { return x.first < y.first; });
        u64 moved = 0;
        for (auto [used, b] : picks) {
            const VkDeviceSize vacant = idle(*b);
            // The moved textures fit in the other blocks with half a block to spare: no new block is made for them.
            if (moved + used > kBudget || image_free[0] - vacant < moved + used + image_block_size / 2) break;
            moved += used, image_free[0] -= vacant, b->draining = true, taken.push_back(b->mem);
        }
    }
    if (taken.empty()) return;
    doomed.clear();
    for (auto& [h, g] : textures)
        if (g.t.mem_size && std::find(taken.begin(), taken.end(), g.t.mem) != taken.end()) doomed.push_back(&g);
    std::sort(doomed.begin(), doomed.end());
    EvictTextures(~0u, true);  // only doomed and never-checked textures go; bindless slots wait for Retire
}

void Renderer::Impl::DrainRetiredElements() {
    if (pending_elements.empty()) return;
    std::scoped_lock res(res_lock);
    for (auto [binding, el] : pending_elements) free_elements[binding].push_back(el);
    pending_elements.clear();
}

void Renderer::Impl::Bury(const ViewKey& k, ViewEntry& v, bool defer_elements) {
    graves.views.push_back(v.view);
    const u32 binding = std::get<1>(k);
    if (defer_elements && flight.busy) {
        pending_elements.push_back({binding, v.element});
    } else {
        free_elements[binding].push_back(v.element);
    }
    if (v.image) graves.shared.push_back(std::move(v.image));
}

// Surface views of images none of its T#s holds any more (evicted, or moved to a grown image by Regrow).
void Renderer::Impl::PruneSurfaceViews(bool defer_elements) {
    std::vector<std::pair<const Surface*, VkImage>> live;
    for (const auto& [h, g] : textures)
        if (g.surf) live.push_back({g.surf, g.t.img});
    std::sort(live.begin(), live.end());
    for (auto& [key, s] : surfaces)
        std::erase_if(s.views, [&](auto& e) {
            if (std::binary_search(live.begin(), live.end(), std::pair<const Surface*, VkImage>{&s, std::get<0>(e.first)})) return false;
            return Bury(e.first, e.second, defer_elements), true;
        });
}

void Renderer::Impl::Reap() {
    if (!dying.empty() && (!reaping.valid() || reaping.wait_for(std::chrono::seconds(0)) == std::future_status::ready)) {
        reaping = std::async(std::launch::async, [this, g = std::move(dying)]() mutable {
            const auto each = [&](auto&& destroy) {
                const auto t = Gow3::DiagNow();
                destroy();
                GOW3_DIAG {
                    const u64 ns = u64((std::chrono::steady_clock::now() - t).count());
                    reaped.fetch_add(1, std::memory_order_relaxed);
                    for (u64 m = reap_max_ns.load(std::memory_order_relaxed); ns > m && !reap_max_ns.compare_exchange_weak(m, ns, std::memory_order_relaxed);) {}
                }
                Common::AccurateSleep(std::chrono::milliseconds(1), nullptr, false);
            };
            for (auto v : g.views) each([&] { vkDestroyImageView(dev, v, nullptr); });
            for (auto& t : g.own) each([&] { FreeImage(t); });
            // Only the last owner of a surface image destroys it: the other references just drop, unpaced (each evicted T#
            // of a surface holds one; Rage of Sparta's grab passes evict ~8 a frame, above the paced ~1000 a second).
            for (auto& s : g.shared)
                if (s.use_count() > 1) s.reset();
                else each([&] { s.reset(); });
        });
        dying = {};
    }
    dying.Append(graves);
}

std::string Renderer::Impl::TempWriter(const GpuWrite& w, const GuestTex& g) {
    return std::format(" | w {:x}+{:.1f}MB {}", w.lo, (w.hi - w.lo) / 1048576.0,
                       w.by == &g ? "self" : w.by ? std::format("by {:x} {}x{} fmt {} tile {}", w.by->base, w.by->w, w.by->h, (int)w.by->fmt, w.by->tile) : "target");
}

void Renderer::Impl::TempCopy(const char* what, const GuestTex& g, const std::string& why) {
    if (temp_copylog)
        std::fprintf(stderr, "COPY %s %llx %ux%ux%u lv%u fmt %d tile %u %.2f MB%s\n", what, (unsigned long long)g.base, g.w, g.h, g.layers, g.levels, (int)g.fmt, g.tile,
                     g.size / 1048576.0, why.c_str());
}

// ---- Render targets and target views ----

// The alignment a surface's size is padded to: its macro tile, the 8x8 micro tile or the linear pitch alignment (1x1 unknown).
static std::pair<u32, u32> PaddingAlign(u32 tile, u32 bits, u32 samples) {
    using namespace AmdGpu;
    const ArrayMode mode = tile == 31 ? ArrayMode::ArrayLinearAligned : tile <= 26 ? GetArrayMode(TileMode(tile)) : ArrayMode::ArrayLinearGeneral;
    if (mode == ArrayMode::ArrayLinearAligned) return {std::max(8u, 512u / std::max(bits, 8u)), 1u};
    if (mode == ArrayMode::Array1DTiledThin1 || mode == ArrayMode::Array1DTiledThick) return VideoCore::micro_tile_extent;
    if (!IsMacroTiled(mode) || !std::has_single_bit(bits) || bits < 8 || bits > 128) return {1u, 1u};
    const auto a = VideoCore::GetMacroTileExtents(TileMode(tile), bits, std::min(samples, 8u), false);
    return a.first ? a : std::pair{1u, 1u};
}

// Target by guest base and padded size. The image has the surface's real size: the extent hint when the command stream
// carries one (Gnm), else the screen scissor of the draw that creates it (GOW3 writes its own PM4 without hints) when it
// only trims the alignment padding (less than the tile alignment and half of each padded dimension). Anything more is a
// sub-rectangle pass: at 4K the shadow maps (2046x1732 of the 2048x2048 atlas, and of the 3840x2176 surfaces bound with
// it) cut the target they created, and every later pass over it lost the bottom and right.
const Renderer::Impl::Target* Renderer::Impl::GetTarget(u64 base, u32 pitch, u32 height, VkFormat fmt, bool depth, bool stencil, u32 extent,
                                                        u32 samples, u32 tile) {
    const u32 guest_samples = samples;
    samples = ClampSamples(samples);
    // Color formats of one texel size share the image (views per format); depth formats are distinct surfaces.
    const TargetKey k{base, pitch, height, u32(depth || !TexelBytes(fmt) ? fmt : TexelBytes(fmt)), samples};
    const u64 h_key = ::Gpu::PieceHash::TargetKey(&k, sizeof(k)) >> 1;
    if (MemoTarget& m = Memos().targets[h_key]; m.t && m.key == k) {
        if (std::atomic_ref(m.t->used).load(std::memory_order_relaxed) != frame)
            std::atomic_ref(m.t->used).store(frame, std::memory_order_relaxed);
        if (std::atomic_ref(m.t->rendered).load(std::memory_order_relaxed) != frame)
            std::atomic_ref(m.t->rendered).store(frame, std::memory_order_relaxed);
        return m.t;
    }
    const auto key = std::tuple{base, pitch, height, VkFormat(k.fmt), samples};
    std::scoped_lock res(res_lock);
    if (auto it = targets.find(key); it != targets.end()) {
        Target& t = it->second;
        std::atomic_ref(t.used).store(frame, std::memory_order_relaxed);
        // Drawn again after its memory was rewritten: the T#s over it are the target again. Another target already mapped
        // at the base changes nothing (Texture picks among them).
        if (const bool back = std::atomic_ref(t.rendered).exchange(frame, std::memory_order_relaxed) == ~0u; target_at.insert(base).second || back)
            std::atomic_ref(epoch).fetch_add(1, std::memory_order_release);
        return Memos().targets[h_key] = {k, &t}, &t;
    }
    u32 w = extent & 0xFFFF, h = extent >> 16;
    if (!extent) {
        const auto [aw, ah] = PaddingAlign(tile, depth ? (fmt == VK_FORMAT_D16_UNORM ? 16 : 32) : TexelBytes(fmt) * 8, guest_samples);
        w = ctx[0x00D] & 0x7FFF, h = ctx[0x00D] >> 16 & 0x7FFF;
        if (w * 2 <= pitch || h * 2 <= height || w > pitch || h > height || pitch - w >= aw || height - h >= ah) w = pitch, h = height;
    }
    w = std::clamp(w, 1u, pitch), h = std::clamp(h, 1u, height);
    const VkImageAspectFlags aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT | (stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0) : VK_IMAGE_ASPECT_COLOR_BIT;
    // Single-sampled color targets are storage images too where the format allows (compute passes write them in place).
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(pd, fmt, &fp);
    bool storage = !depth && samples == 1 && fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    VkImageCreateFlags flags = depth ? 0 : VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    VkImageUsageFlags usage = (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) | VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!depth && !storage && fmt == VK_FORMAT_R8G8B8A8_SRGB && (samples == 1 || blend_interlock)) {
        vkGetPhysicalDeviceFormatProperties(pd, VK_FORMAT_R8G8B8A8_UINT, &fp);
        VkImageFormatProperties image{};
        const auto extended = flags | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        if ((fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) &&
            vkGetPhysicalDeviceImageFormatProperties(pd, fmt, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                     usage | VK_IMAGE_USAGE_STORAGE_BIT, extended, &image) == VK_SUCCESS &&
            image.maxExtent.width >= w && image.maxExtent.height >= h && (image.sampleCounts & samples))
            storage = true, flags = extended;
    }
    if (storage) usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    Target t{base, w, h, pitch, height, samples, fmt, depth, stencil,
             MakeImage(VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, fmt, w, h, 1, 1, VkSampleCountFlagBits(samples), usage, aspect, 1,
                       flags, true),
             frame, frame};
    if (!t.t.img) return Note("render target allocation failed"), nullptr;
    // New targets start cleared, in GENERAL for their whole life (phase 3 simplicity).
    fresh.push_back({t.t.img, aspect});
    Count(stats.targets);
    Target* made;
    {
        std::scoped_lock lock(targets_lock);
        made = &targets.emplace(key, std::move(t)).first->second;
        target_at.insert(base);
        std::atomic_ref(epoch).fetch_add(1, std::memory_order_release);  // textures over this memory are the target now
    }
    return Memos().targets[h_key] = {k, made}, made;
}

VkImageView Renderer::Impl::TargetAttachment(const Target& t, VkFormat fmt) {
    std::scoped_lock res(res_lock);
    VkImageView& v = t.attach[fmt];
    if (!v) v = View(t.t.img, VK_IMAGE_VIEW_TYPE_2D, fmt, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, {}, t.t.usage);
    return v;
}

u32 Renderer::Impl::TargetView(Target& rt, u64 key, u32 binding, VkImage img, VkImageViewType vtype, VkFormat fmt, VkImageAspectFlags aspect,
                               VkComponentMapping map, VkImageUsageFlags usage, const char* fail_note) {
    if (auto it = rt.views.find(key); it != rt.views.end()) return it->second;
    const VkImageView view = View(img, vtype, fmt, {aspect, 0, 1, 0, 1}, map, usage);
    if (!view) {
        if (fail_note) Note(fail_note);
        return 0;
    }
    const u32 el = Bindless(binding, view, VK_NULL_HANDLE);
    if (!el) return vkDestroyImageView(dev, view, nullptr), 0;
    rt.owned_views.push_back(view);
    return rt.views[key] = el;
}

u32 Renderer::Impl::TargetBlendElement(const Target& t) {
    std::scoped_lock lock(res_lock);
    constexpr u32 kBinding = 20;
    constexpr u64 kKey = u64(kBinding) << 44 | VK_FORMAT_R8G8B8A8_UINT;
    return TargetView(const_cast<Target&>(t), kKey, kBinding, t.t.img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UINT,
                      VK_IMAGE_ASPECT_COLOR_BIT, {}, t.t.usage);
}

u32 Renderer::Impl::TargetTexture(Target& rt, const u32* t, int kind, u32 binding, u64& sub) {
    std::scoped_lock res(res_lock);
    const bool storage = binding >= kStorageBinding;
    const auto raw = AmdGpu::DataFormat(t[1] >> 20 & 0x3F);
    VkFormat f = SurfaceFormat(u32(AmdGpu::RemapDataFormat(raw)), u32(AmdGpu::RemapNumberFormat(AmdGpu::NumberFormat(t[1] >> 26 & 0xF), raw)));
    if (rt.depth || !f || TexelBytes(f) != TexelBytes(rt.fmt)) f = rt.fmt;
    // A storage slot writes the surface in place (a smaller T#, its top-left rectangle): identity swizzle, a view format
    // with storage support. A target the driver gave no storage usage (Intel's SRGB, even through EXTENDED_USAGE: GOW3's
    // grab copies into the scene target) is a dispatch's through a twin image in the view format, copied in and back.
    bool twin = false;
    if (storage) {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(pd, f, &fp);
        twin = !(rt.t.usage & VK_IMAGE_USAGE_STORAGE_BIT) && dispatching && !rt.depth && rt.samples == 1;
        if ((!(rt.t.usage & VK_IMAGE_USAGE_STORAGE_BIT) && !twin) || !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
            return Note(std::format("render target {:#x} (format {}, depth {}, usage {:#x}, {} samples, view format {}, features {:#x}) bound as storage image",
                                    rt.base, int(rt.fmt), rt.depth, rt.t.usage, rt.samples, int(f), fp.optimalTilingFeatures)), 0;
        if (twin && !rt.storage_copy.img) {
            rt.storage_copy = MakeImage(VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, f, rt.w, rt.h, 1, 1, VK_SAMPLE_COUNT_1_BIT,
                                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                        VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT, true);
            if (!rt.storage_copy.img) return Note("storage twin of a render target not creatable"), 0;
            fresh.push_back({rt.storage_copy.img, VK_IMAGE_ASPECT_COLOR_BIT});
        }
        if ((t[3] >> 20 & 0x1F) == 8) Note("linear render target written as storage image: not written back");
    }
    const Tex& img = twin ? rt.storage_copy : rt.t;
    const u64 key = (u64)binding << 44 | (u64)(rt.depth || storage ? 0 : t[3] & 0xFFF) << 32 | (u32)f;  // binding, DST_SEL, view format
    // A T# smaller than the surface's image is its top-left rectangle (sub-rectangle passes such as the luminance
    // reduction): the shader scales its coordinates into the image and bounds its texel accesses to the T#.
    const u32 tw = std::min((t[2] & 0x3FFF) + 1, rt.w), th = std::min((t[2] >> 14 & 0x3FFF) + 1, rt.h);
    if ((kind == 0 || kind == 1) && rt.samples == 1 && (tw < rt.w || th < rt.h)) sub = tw | th << 16 | (u64)rt.w << 32 | (u64)rt.h << 48;
    const u32 e = [&]() -> u32 {
        if (auto v = rt.views.find(key); v != rt.views.end()) return v->second;
        // Kinds 0/1 sample single-sampled surfaces, 4/5 multisampled ones (the T# says which the shader expects).
        if ((kind != 0 && kind != 1 && kind != 4 && kind != 5) || (kind >= 4) != (rt.samples > 1))
            return Note(std::format("render target ({} samples) sampled as view kind {}", rt.samples, kind)), 0;
        return TargetView(rt, key, binding, img.img, kind == 1 || kind == 5 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, f,
                          VkImageAspectFlags(rt.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT),
                          rt.depth || storage ? VkComponentMapping{} : DstSelect(t), img.usage, "render target image view creation failed");
    }();
    if (twin && e) {
        twins.push_back(&rt);
        if (stats.trace && !stats.trace_lines.empty()) stats.trace_lines.back() += " twin";
    }
    return e;
}

// The guest stores depth and stencil at separate addresses; Vulkan keeps both aspects in one image.
void Renderer::Impl::RegisterStencil() {
    const u32 zfmt = ctx[0x010] & 3;
    if (zfmt != 3 || !(ctx[0x011] & 1) || !ctx[0x014] || !ctx[0x015] ||
        (!(ctx[0x200] & 1) && !(ctx[0x000] & 2))) return;
    const u64 base = (u64)ctx[0x015] << 8;
    const u32 size = ctx[0x016], w = ((size & 0x7FF) + 1) * 8, h = ((size >> 11 & 0x7FF) + 1) * 8;
    Target* target = const_cast<Target*>(GetTarget((u64)ctx[0x014] << 8, w, h, VK_FORMAT_D32_SFLOAT_S8_UINT, true, true, db_extent,
                                                   1u << (ctx[0x010] >> 2 & 3), ctx[0x010] >> 20 & 7));
    if (!target) return;
    if (auto it = stencil_at.find(base); it == stencil_at.end() || it->second.first != target) {
        stencil_at[base] = {target, frame};
        std::atomic_ref(epoch).fetch_add(1, std::memory_order_release);
    } else it->second.second = frame;
}

u32 Renderer::Impl::StencilMsTexture(Target& target, u32 binding) {
    std::scoped_lock res(res_lock);
    const u64 key = 1ull << 62 | (u64)binding << 44;
    const u32 element = TargetView(target, key, binding, target.t.img, VK_IMAGE_VIEW_TYPE_2D, target.fmt,
                                   VK_IMAGE_ASPECT_STENCIL_BIT, {}, target.t.usage, "multisampled stencil view creation failed");
    if (!element) return 0;
    draw_rts.push_back({target.t.img, false});
    if (stats.trace && !stats.trace_lines.empty()) stats.trace_lines.back() += std::format(" | stencil-ms from {:x}", target.base);
    return element;
}

u32 Renderer::Impl::StencilTexture(Target& target, const u32* t, u32 binding) {
    std::scoped_lock res(res_lock);
    if (!target.stencil_copy.img) {
        target.stencil_copy = MakeImage(VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8_UNORM, target.w, target.h, 1, 1, VK_SAMPLE_COUNT_1_BIT,
                                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, true);
        if (!target.stencil_copy.img) return Note("stencil texture creation failed"), 0;
        fresh.push_back({target.stencil_copy.img, VK_IMAGE_ASPECT_COLOR_BIT});
    }
    const u64 key = 1ull << 63 | (u64)binding << 44 | (u64)(t[3] & 0xFFF) << 32;
    const u32 element = TargetView(target, key, binding, target.stencil_copy.img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8_UNORM,
                                   VK_IMAGE_ASPECT_COLOR_BIT, DstSelect(t), target.stencil_copy.usage, "stencil texture view creation failed");
    if (!element) return 0;
    EndRendering(false);
    Add(hz.r, target.t.img);
    if (!hz.fenced) Add(hz.in_r, target.t.img);
    Barrier(cmd, true);  // the stencil written, the copy sampled before
    const TempSlice tmp = Temp((u64)target.w * target.h);
    if (!tmp.buf) return Note("stencil temp buffer allocation failed"), 0;
    const VkBufferImageCopy src{tmp.offset, 0, 0, {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1}, {}, {target.w, target.h, 1}};
    const VkBufferImageCopy dst{tmp.offset, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {target.w, target.h, 1}};
    vkCmdCopyImageToBuffer(cmd, target.t.img, VK_IMAGE_LAYOUT_GENERAL, tmp.buf, 1, &src);
    StageBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    vkCmdCopyBufferToImage(cmd, tmp.buf, target.stencil_copy.img, VK_IMAGE_LAYOUT_GENERAL, 1, &dst);
    Add(hz.w, target.stencil_copy.img);
    draw_rts.push_back({target.stencil_copy.img, false});
    if (stats.trace && !stats.trace_lines.empty()) stats.trace_lines.back() += std::format(" | stencil {:x} from {:x}", (u64)t[0] << 8, target.base);
    return element;
}

// A linear color target (CB tile index 8: the 1x1 luminance the CPU reads) goes back to the mirror after each draw to it.
// Only the drawn rectangle: the surface's declared slice can span unrelated data (the 1x1 luminance target declares
// 128x64 texels over shader programs).
void Renderer::Impl::WriteBackTarget(const Target& t, u32 pitch, VkRect2D area) {
    const u32 x0 = (u32)area.offset.x, y0 = (u32)area.offset.y, x1 = std::min(t.w, x0 + area.extent.width), y1 = std::min(t.h, y0 + area.extent.height);
    if (t.samples > 1 || x1 <= x0 || y1 <= y0) return;
    const u64 tb = TexelBytes(t.fmt), lo = t.base + ((u64)y0 * pitch + x0) * tb, hi = t.base + ((u64)(y1 - 1) * pitch + x1) * tb;
    if (hi > mirror_size) return;
    EndRendering();
    BindPages(lo, hi);
    const VkBufferImageCopy bc{0, pitch, y1 - y0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {(int)x0, (int)y0, 0}, {x1 - x0, y1 - y0, 1}};
    ImageToMirror(cmd, t.t.img, bc, lo, (x1 - x0) * tb, pitch * tb, 1);
    if (temp_copylog) std::fprintf(stderr, "COPY WBT %llx %ux%u rect %ux%u\n", (unsigned long long)t.base, t.w, t.h, x1 - x0, y1 - y0);
    Barrier(cmd, true);  // as in WriteBack
    const u64 seq = std::atomic_ref(write_seq).fetch_add(1) + 1;
    (in_worker ? own_writes : gpu_writes).push_back({lo, hi, seq, nullptr});
}

bool Renderer::FindTarget(std::uint64_t base, Surface& out) {
    std::scoped_lock lock(d->targets_lock);
    const Impl::Target* shown = nullptr;
    d->TargetsAt(base, [&](const Impl::Target& t) {
        if (!t.depth && t.samples == 1 && t.w * t.h > out.w * out.h) out = {t.t.img, t.fmt, t.w, t.h}, shown = &t;
    });
    // On display: used, so EvictTextures keeps it while the game flips it without drawing it.
    if (shown) std::atomic_ref(shown->used).store(std::atomic_ref(d->frame).load(std::memory_order_relaxed), std::memory_order_relaxed);
    return out.image != VK_NULL_HANDLE;
}

// ---- Guest textures and samplers ----

// T# -> image description and guest mip layout (shadPS4 ImageInfo(const AmdGpu::Image&) and ImageInfo::UpdateSize()).
bool Renderer::Impl::Describe(GuestTex& g, const u32* t) {
    using namespace AmdGpu;
    const auto raw_dfmt = DataFormat(t[1] >> 20 & 0x3F);
    const auto dfmt = RemapDataFormat(raw_dfmt);
    const auto nfmt = RemapNumberFormat(NumberFormat(t[1] >> 26 & 0xF), raw_dfmt);
    const u32 type = t[3] >> 28, last_level = t[3] >> 16 & 0xF, depth = (t[4] & 0x1FFF) + 1, pitch = (t[4] >> 13 & 0x3FFF) + 1;
    const bool pow2 = t[3] >> 25 & 1, cube = type == 11;
    g.tile = t[3] >> 20 & 0x1F;
    if (g.tile > 26 && g.tile != 31) return Note(std::format("T# tile mode {}", g.tile)), false;
    if ((type == 14 || type == 15) && last_level) return Note("multisampled T# (texture from a MSAA surface)"), false;
    g.fmt = SurfaceFormat(u32(dfmt), u32(nfmt));
    if (!g.fmt) return Note(std::format("T# format {}/{} has no Vulkan format", u32(dfmt), u32(nfmt))), false;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(pd, g.fmt, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) return Note(std::format("format {} not sampleable", (int)g.fmt)), false;
    g.storage = fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    g.base = ((u64)t[0] | (u64)(t[1] & 0x3F) << 32) << 8;
    g.w = (t[2] & 0x3FFF) + 1, g.h = (t[2] >> 14 & 0x3FFF) + 1, g.volume = type == 10, g.d = g.volume ? depth : 1;
    // POW2_PAD pads guest mip slice stride to a power of two (6 -> 8 faces), while Vulkan CUBE_COMPATIBLE images take the
    // real 6*N faces (avoids 2 dead layers and an AMD PAL gfx6/Polaris AddrMgr1 mip-stride bug on arrayLayers = 8).
    const u32 raw_layers = g.volume ? 1 : cube ? depth * 6 : depth;
    const u32 guest_layers = pow2 ? std::bit_ceil(raw_layers) : raw_layers;
    g.layers = raw_layers;
    g.guest_layers = guest_layers;
    g.levels = type >= 14 ? 1 : last_level + 1;
    g.bits = NumBitsPerBlock(dfmt), g.block = IsBlockCoded(dfmt);
    g.map = DstSelect(t);
    const auto tile_mode = TileMode(g.tile);
    const auto array_mode = g.tile == 31 ? ArrayMode::ArrayLinearAligned : GetArrayMode(tile_mode);
    g.size = 0;
    for (u32 mip = 0; mip < g.levels; ++mip) {
        u32 mip_w = std::max(g.block ? ((pitch >> mip) + 3) / 4 : pitch >> mip, 1u);
        u32 mip_h = std::max(g.block ? ((g.h >> mip) + 3) / 4 : g.h >> mip, 1u);
        u32 mip_d = std::max(g.d >> mip, 1u), thickness = 1;
        if (pow2) mip_w = std::bit_ceil(mip_w), mip_h = std::bit_ceil(mip_h), mip_d = std::bit_ceil(mip_d);
        u32 p, h;
        size_t s;
        switch (array_mode) {
        case ArrayMode::ArrayLinearAligned: std::tie(p, h, s) = VideoCore::ImageSizeLinearAligned(mip_w, mip_h, g.bits, 1); break;
        case ArrayMode::Array1DTiledThick:
            thickness = 4, mip_d += (0u - mip_d) & 3;
            [[fallthrough]];
        case ArrayMode::Array1DTiledThin1: std::tie(p, h, s) = VideoCore::ImageSizeMicroTiled(mip_w, mip_h, thickness, g.bits, 1); break;
        case ArrayMode::Array2DTiledThick:
            thickness = 4, mip_d += (0u - mip_d) & 3;
            [[fallthrough]];
        case ArrayMode::Array2DTiledThin1:
            if (g.block) return Note("macro tiled block-compressed T#"), false;
            std::tie(p, h, s) = VideoCore::ImageSizeMacroTiled(mip_w, mip_h, thickness, g.bits, 1, tile_mode, mip, false);
            break;
        default: return Note(std::format("T# array mode {}", u32(array_mode))), false;
        }
        if (g.block) p = std::max(p * 4, 32u), h = std::max(h * 4, 32u);
        g.mips[mip] = {u32(s * mip_d * guest_layers), p, h, u32(g.size)};
        g.size += g.mips[mip].size;
    }
    if (g.base + g.size > mirror_size) return Note(std::format("T# beyond the mirror (guest addresses below {} GB)", mirror_size >> 30)), false;
    g.img_levels = std::min<u32>(g.levels, std::bit_width(std::max({g.w, g.h, g.d})));
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |  // src: ReadTextures
                                    (g.storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
    const bool cube_ok = cube && g.w == g.h && g.layers >= 6;
    auto fail = [&](u32 w, u32 h) { return Note(std::format("texture {}x{}x{} format {} not creatable", w, h, g.layers, (int)g.fmt)), false; };
    // Block-coded and volume textures own their image. The others share their surface's (storage and sampled T#s of one
    // memory layout, bands of one surface), as big as the largest T# seen; a smaller 2D T# is its top-left part.
    Surface* s = g.block || g.volume ? nullptr : &surfaces[{g.base, g.tile, g.bits, g.mips[0].pitch, g.layers | (g.guest_layers << 16), g.levels, g.levels > 1 ? g.w | g.h << 16 : 0}];
    std::shared_ptr<Tex> cur = s ? s->t.lock() : nullptr;
    g.flat = type == 9 || type >= 13;  // single-sampled 2D (array): views that take a sub-rectangle
    if (s && cur && !g.flat && (g.w < s->w || g.h < s->h)) s = nullptr;  // not representable in it
    const bool storable = g.storage || (!g.block && !g.volume && (g.bits == 8 || g.bits == 16 || g.bits == 32 || g.bits == 64 || g.bits == 128));
    if (!s) {
        const VkImageCreateFlags mut = g.block || g.volume ? 0 : VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        const VkImageCreateFlags cflags = mut | (cube_ok ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0);
        g.t = MakeImage(g.volume ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D, g.volume ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D_ARRAY, g.fmt, g.w, g.h, g.d,
                        g.layers, VK_SAMPLE_COUNT_1_BIT, usage | (storable ? VK_IMAGE_USAGE_STORAGE_BIT : 0), VK_IMAGE_ASPECT_COLOR_BIT, g.img_levels, cflags);
        if (!g.t.img && storable && !g.storage)
            g.t = MakeImage(g.volume ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D, g.volume ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D_ARRAY, g.fmt, g.w, g.h, g.d,
                            g.layers, VK_SAMPLE_COUNT_1_BIT, usage, VK_IMAGE_ASPECT_COLOR_BIT, g.img_levels, cflags);
        if (!g.t.img) return fail(g.w, g.h);
        g.iw = g.w, g.ih = g.h;
        Count(stats.textures);
        return true;
    }
    if (!cur || g.w > s->w || g.h > s->h || (cube_ok && !s->cube)) {
        // A bigger image: this T# takes it now, the surface's others at the next frame's start (Regrow).
        if (cur) regrow = s->growing = true;
        s->w = std::max(s->w, g.w), s->h = std::max(s->h, g.h), s->cube |= cube_ok;
        s->current = s->valid_frame = ~0u, s->valid_w = s->valid_h = 0;
        // Storage usage through a format of the same size when this one has none (MUTABLE + EXTENDED_USAGE).
        Tex* t = new Tex(MakeImage(VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, g.fmt, s->w, s->h, 1, g.layers, VK_SAMPLE_COUNT_1_BIT,
                                   usage | (storable ? VK_IMAGE_USAGE_STORAGE_BIT : 0), VK_IMAGE_ASPECT_COLOR_BIT, g.img_levels,
                                   VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT |
                                       (s->cube && s->w == s->h ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0)));
        if (!t->img) return delete t, fail(s->w, s->h);
        cur = std::shared_ptr<Tex>(t, [this](Tex* x) { FreeImage(*x), delete x; });
        s->t = cur, s->img = cur->img;
        Count(stats.textures);
    } else {
        // The image is live (no layout change) and holds this part unless the CPU wrote it since the surface was last current.
        // GOW3's distortion grab passes copy the screen into one surface through a T# of another height each time (~6 a
        // frame in Rage of Sparta): each reloaded its ~13 MB part, written by the GPU only.
        u32 known = s->current;
        if (g.w <= s->valid_w && g.h <= s->valid_h && s->valid_frame != ~0u && (known == ~0u || s->valid_frame > known))
            known = s->valid_frame;
        g.fresh = false, g.reload = known == ~0u;
        const u32* at = staged_table.get();
        for (u64 p = g.base >> 12, end = std::min((g.base + g.size + 0xFFF) >> 12, kPieces); !g.reload && p < end; ++p)
            g.reload = at[p] > known;
    }
    g.shared = cur, g.t = *cur, g.surf = s, g.iw = s->w, g.ih = s->h, ++s->users;
    return true;
}

// Start of a frame: the T#s of a surface whose image grew move to the new image (old views, images and bindless slots wait for Retire).
void Renderer::Impl::Regrow() {
    regrow = false;
    auto by_cpu = [&](const GpuWrite& w) {
        auto c = std::upper_bound(dirty.begin(), dirty.end(), std::pair{w.hi, 0ull});
        return c != dirty.begin() && std::prev(c)->second > w.lo;
    };
    std::vector<const GuestTex*> moved;
    for (auto& [h, g] : textures) {
        if (!g.surf) continue;
        Surface& s = *g.surf;
        s.growing = false;
        const std::shared_ptr<Tex> cur = s.t.lock();
        if (!cur) s.t = g.shared, s.img = g.shared ? g.shared->img : VK_NULL_HANDLE, s.w = g.iw, s.h = g.ih, s.current = s.valid_frame = ~0u, s.valid_w = s.valid_h = 0;  // the grown image's T#s are gone: this one's is the surface's again
        if (!cur || cur == g.shared || g.w > s.w || g.h > s.h || (!g.flat && (g.w != s.w || g.h != s.h))) continue;
        // Materialize pending lazy writes from the old image before switching `g` to the grown one.
        bool defer = false;
        std::erase_if(lazy_writes, [&](const GpuWrite& w) {
            if (w.by != &g) return false;
            if (by_cpu(w)) return true;
            QueuePrologueLazy(w, false);
            defer = true;
            return false;
        });
        if (defer) {
            regrow = s.growing = true;
            continue;
        }
        graves.views.insert(graves.views.end(), g.owned.begin(), g.owned.end());
        g.owned.clear(), std::fill_n(g.mip_views, 16, VK_NULL_HANDLE);
        if (g.shared) graves.shared.push_back(std::move(g.shared));
        g.shared = cur, g.t = *cur, g.iw = s.w, g.ih = s.h, g.fresh = false, g.reload = true;
        moved.push_back(&g);
    }
    if (!moved.empty()) {
        std::sort(moved.begin(), moved.end());
        for (auto* list : {&gpu_writes, &last_writes})
            for (auto& w : *list)
                if (w.by && std::binary_search(moved.begin(), moved.end(), w.by)) w.by = nullptr, w.lazy = false;
    }
    PruneSurfaceViews(true);
    std::atomic_ref(epoch).fetch_add(1, std::memory_order_release);
}

// Once per frame (res_lock held): queued for the prologue upload when its memory changed since the last frame.
void Renderer::Impl::Check(GuestTex& g) {
    if (g.checked == frame) return;
    TempTimer temp_timer{temp_check_ns, temp_checks};
#ifndef GOW3_OFFLINE
    const u64 video_serial = Gow3::GpuThread::VideoFrameVersion(g.base);
    const bool staged = video_serial ? false : Track(g.base, g.base + g.size);
#else
    const bool staged = Track(g.base, g.base + g.size);
#endif
    auto d = std::upper_bound(dirty.begin(), dirty.end(), std::pair{g.base + g.size, 0ull});
    // The last frame's GPU writes count when made in another image (its own already holds them), and the earlier ones
    // still only there.
    auto foreign = [&](const GpuWrite& w) { return w.lo < g.base + g.size && w.hi > g.base && (!w.by || w.by->t.img != g.t.img); };
    bool cpu = staged || (d != dirty.begin() && std::prev(d)->second > g.base);
    // Writes staged in frames it skipped are in no `dirty` it saw: the halves of the color correction LUT (01165AC0) are
    // drawn in turns, each written while the other is drawn, and one kept an old grade (the title menu's brightness flicker).
    if (const u32* at = staged_table.get(); !cpu && g.checked != ~0u && frame - g.checked > 1)
        for (u64 p = g.base >> 12, end = std::min((g.base + g.size + 0xFFF) >> 12, kPieces); !cpu && p < end; ++p) cpu = at[p] > g.checked;
#ifndef GOW3_OFFLINE
    if (video_serial && video_serial != g.video_serial && Gow3::GuestMemory::Readable(g.base, g.size)) {
        Gow3::GpuThread::TraceVideoCopy(g.base, frame, false);
        if (Stage(g.base, reinterpret_cast<const u8*>(g.base), g.size)) {
            Gow3::GpuThread::TraceVideoCopy(g.base, frame, true);
            g.video_serial = video_serial;
            cpu = true;
        }
    }
#endif
    const bool upload = g.fresh || g.reload || cpu || std::any_of(last_writes.begin(), last_writes.end(), foreign) || std::any_of(lazy_writes.begin(), lazy_writes.end(), foreign);
#ifndef GOW3_OFFLINE
    Gow3::GpuThread::TraceVideoTexture(g.base, frame, u32(upload) | (u32(staged) << 1) | (u32(cpu) << 2));
#endif
    if (upload) {
        uploads.push_back(&g), g.synced = frame_seq;
        // Those reach the mirror in the prologue, before the uploads; a CPU write over one replaces it.
        auto by_cpu = [&](const GpuWrite& w) {
            auto c = std::upper_bound(dirty.begin(), dirty.end(), std::pair{w.hi, 0ull});
            return c != dirty.begin() && std::prev(c)->second > w.lo;
        };
        for (auto& w : lazy_writes)
            if (foreign(w) && w.by) {
                if (by_cpu(w)) w.by->surf->foreign = true;
                else QueuePrologueLazy(w, !g.surf || w.by->surf != g.surf);
            }
        if (temp_copylog) {
            std::string why = g.fresh ? " fresh" : g.reload ? " reload" : staged ? " cpu" : " dirty";
            for (auto& w : last_writes)
                if (w.lo < g.base + g.size && w.hi > g.base) why += TempWriter(w, g);
            TempCopy("UP", g, why);
        }
        g.reload = false;
    }
    if (g.surf && g.t.img == g.surf->img) {
        Surface& s = *g.surf;
        if (g.w == s.w && g.h == s.h) {
            s.current = s.valid_frame = frame, s.valid_w = s.w, s.valid_h = s.h;
        } else if (s.valid_frame == ~0u || (g.w >= s.valid_w && g.h >= s.valid_h)) {
            s.valid_frame = frame, s.valid_w = g.w, s.valid_h = g.h;
        } else {
            const u32* at = staged_table.get();
            bool stale = false;
            for (u64 p = g.base >> 12, end = std::min((g.base + g.size + 0xFFF) >> 12, kPieces); !stale && p < end; ++p)
                stale = at[p] > s.valid_frame;
            if (stale) s.valid_frame = frame, s.valid_w = g.w, s.valid_h = g.h;
        }
    }
    std::atomic_ref(g.checked).store(frame, std::memory_order_release);
}

// Memory this frame's GPU work rewrote since the image was filled: upload it again before this use. A parallel block
// counts from the frame start (the image as the prologue uploads it) and only the writes recorded before it: it leaves
// the upload for those to Submit, which makes it once per version ahead of the blocks that need it.
void Renderer::Impl::Resync(GuestTex& g) {
    const size_t seen = std::min(horizon, gpu_writes.size());
    auto over = [&](const GpuWrite& w) { return w.lo < g.base + g.size && w.hi > g.base && (!w.by || w.by->t.img != g.t.img); };  // in another image
    if (std::none_of(gpu_writes.begin(), gpu_writes.begin() + seen, over) && std::none_of(own_writes.begin(), own_writes.end(), over)) return;
    u64& synced = synced_here ? synced_here->try_emplace(&g, std::min(g.synced, frame_seq)).first->second : g.synced;
    // Writes left in their image go to the mirror first; a reader outside their surface makes it write back at once.
    auto learn = [&](const GpuWrite& w) { return !g.surf || !w.by || w.by->surf != g.surf; };
    if (block_now) {
        u64 need = 0;
        for (size_t k = 0; k < seen; ++k)
            if (over(gpu_writes[k])) need = std::max(need, gpu_writes[k].seq);
        if (need > synced && temp_copylog)
            for (size_t k = 0; k < seen; ++k)
                if (over(gpu_writes[k]) && gpu_writes[k].seq > synced) TempCopy("RS", g, TempWriter(gpu_writes[k], g));
        if (need > synced) {  // the block's own writes come after these
            for (size_t k = 0; k < seen; ++k)
                if (const GpuWrite& w = gpu_writes[k]; w.lazy && over(w) && w.seq > synced) block_now->lazy.push_back({w, learn(w)});
            block_now->resync.push_back({&g, need}), synced = need;
        }
        if (need && !table_ps) block_now->resync_vs.push_back(&g);
    }
    for (size_t k = 0; k < seen + own_writes.size(); ++k)
        if (const GpuWrite& w = k < seen ? gpu_writes[k] : own_writes[k - seen]; over(w) && w.seq > synced) {
            if (temp_copylog) TempCopy("RS-inline", g, TempWriter(w, g));
            EndRendering();
            std::scoped_lock res(res_lock);
            bool fenced = false;
            for (size_t j = 0; j < seen + own_writes.size(); ++j)
                if (const GpuWrite& x = j < seen ? gpu_writes[j] : own_writes[j - seen]; x.lazy && over(x) && x.seq > synced) {
                    if (!fenced) Barrier(cmd), fenced = true;
                    Materialize(cmd, x, learn(x));
                }
            std::vector<GuestTex*> one{&g};
            Upload(cmd, one, nullptr, 0, table_ps);
            synced = write_seq;
            return;
        }
}

// A slot's memo entry again: a render target's view joins the draw's targets; a texture is checked once per frame and
// uploaded again when this frame's GPU work rewrote its memory.
u32 Renderer::Impl::Reuse(const MemoTex& m, const u32* t, bool storage) {
    GuestTex* g = m.g;
    tex_sub = m.sub;
    if (!g) {
        if (m.target && std::atomic_ref(m.target->used).load(std::memory_order_relaxed) != frame)
            std::atomic_ref(m.target->used).store(frame, std::memory_order_relaxed);
        if (storage && m.target && m.target->storage_copy.img) {  // the twin of TargetTexture: only a dispatch copies it back
            if (!dispatching) return 0;
            twins.push_back(m.target);
        }
        return m.rt ? (draw_rts.push_back({m.rt, storage}), m.e) : m.e;
    }
    AgainstPending(*g, storage);
    if (std::atomic_ref(g->checked).load(std::memory_order_acquire) != frame) {
        std::scoped_lock res(res_lock);
        Check(*g);
    }
    Resync(*g);
    if (storage && std::none_of(stores.begin(), stores.end(), [&](auto& s) { return s.first == g; }))
        stores.push_back({g, std::min<u32>(t[3] >> 12 & 0xF, g->img_levels - 1)});
    return m.e;
}

// Bindless element of a texture/storage slot; 0 (dummy) when the T# cannot be represented.
u32 Renderer::Impl::Texture(const u32* t, const Gcn::SlotSpec& spec, bool storage, bool stencil_resolve) {
    tex_used = {}, tex_keep = true;
    const u64 base = ((u64)t[0] | (u64)(t[1] & 0x3F) << 32) << 8;
    if (!(t[3] >> 31) || !base) return 0;
    const int kind = DimKind(spec.dim);
    const u32 binding = (storage ? kStorageBinding : 0) + kind * 3 + u32(spec.cls);
    if (spec.stencil) {
        if (auto it = stencil_at.find(base); it != stencil_at.end() && it->second.second == frame && it->second.first)
            return tex_keep = false, StencilMsTexture(*it->second.first, kind * 3 + u32(Gcn::Class::Uint));
        return tex_keep = false, 0;
    }
    if (stencil_resolve && !storage && kind == 0 && spec.cls == Gcn::Class::Float && (t[1] >> 20 & 0x3F) == 1 && !(t[1] >> 26 & 0xF))
        if (auto it = stencil_at.find(base); it != stencil_at.end() && it->second.second == frame && it->second.first && it->second.first->samples == 1 &&
            (t[2] & 0x3FFF) + 1 == it->second.first->w && (t[2] >> 14 & 0x3FFF) + 1 == it->second.first->h)
            return tex_keep = false, StencilTexture(*it->second.first, t, binding);
    const u64 hash = ::Gpu::PieceHash::TextureKey(t, 32), key = hash + binding * 0x9E3779B97F4A7C15ull;
    auto& memo = Memos().tex;
    if (const MemoTex* m = stats.trace ? nullptr : memo.Find(key)) return tex_used = *m, Reuse(*m, t, storage);
    std::scoped_lock res(res_lock);
    if (stats.trace && !stats.trace_lines.empty())
        stats.trace_lines.back() += std::format(" | T {:x} {}x{} fmt {}/{} tile {} type {} kind {}{}", base, (t[2] & 0x3FFF) + 1, (t[2] >> 14 & 0x3FFF) + 1,
                                                t[1] >> 20 & 0x3F, t[1] >> 26 & 0xF, t[3] >> 20 & 0x1F, t[3] >> 28, kind,
                                                std::any_of(targets.begin(), targets.end(), [&](auto& e) { return e.second.base == base; }) ? " RT" : "");
    // A render target the GPU wrote is the texture, not the stale guest memory under it (a storage slot writes it in place).
    // A multisampled one read or written single-sampled is another surface over its memory: GOW3 copies the resolved scene
    // into its MSAA scene target's memory for the refraction of water and glass. Of the targets sharing the base, the T#'s is
    // the tightest one it fits in, then the one drawn last (no thread's order of arrival decides).
    if (target_at.contains(base)) {
        const auto raw_dfmt = AmdGpu::DataFormat(t[1] >> 20 & 0x3F), dfmt = AmdGpu::RemapDataFormat(raw_dfmt);
        const u32 tw = (t[2] & 0x3FFF) + 1, th = (t[2] >> 14 & 0x3FFF) + 1;
        const bool viewable = (kind == 0 || kind == 1 || kind == 4 || kind == 5) && (kind < 4 || !storage) && !AmdGpu::IsBlockCoded(dfmt);
        Target* rt = nullptr;
        u64 best = ~0ull;
        bool theirs = false;  // the memory is still a target's: one drawn this frame or the last, or a multisampled one read single-sampled
        TargetsAt(base, [&](Target& c) {
            const u32 drawn = std::atomic_ref(c.rendered).load(std::memory_order_relaxed), texel = TexelBytes(c.fmt);
            if (drawn == ~0u) return;
            theirs |= frame - drawn <= 1 || (c.samples > 1 && kind < 4);
            // A color format without a texel size in TexelBytes is viewed only as itself (GetTarget's key).
            if (const u64 rank = (u64)c.pitch * c.height << 32 | (frame - drawn);
                viewable && (kind >= 4) == (c.samples > 1) && tw <= c.pitch && th <= c.height && rank < best &&
                (c.depth || (texel ? AmdGpu::NumBitsPerBlock(dfmt) == texel * 8
                                   : SurfaceFormat(u32(dfmt), u32(AmdGpu::RemapNumberFormat(AmdGpu::NumberFormat(t[1] >> 26 & 0xF), raw_dfmt))) == c.fmt)))
                rt = &c, best = rank;
        });
        if (rt) {
            if (std::atomic_ref(rt->used).load(std::memory_order_relaxed) != frame) std::atomic_ref(rt->used).store(frame, std::memory_order_relaxed);
            u64 sub = 0;
            const u32 e = TargetTexture(*rt, t, kind, binding, sub);
            if (!e) return tex_keep = false, 0;
            tex_used = memo[key] = {nullptr, e, rt->t.img, sub, rt}, draw_rts.push_back({rt->t.img, storage}), tex_sub = sub;
            return e;
        }
        if (!theirs) {  // the game reused the memory: textures, until a target is drawn there again
            TargetsAt(base, [](Target& c) { std::atomic_ref(c.rendered).store(~0u, std::memory_order_relaxed); });
            target_at.erase(base);
            std::atomic_ref(epoch).fetch_add(1, std::memory_order_release);
        }
    }
    auto [it, fresh] = textures.try_emplace(hash);
    GuestTex& g = it->second;
    if (fresh && !Describe(g, t)) g.t = {};
    if (!g.t.img) return 0;
    if (stats.trace && !stats.trace_lines.empty()) {  // share of the guest bytes (4 KB pieces) that reached the mirror
        u64 have = 0, n = 0;
        for (u64 a = g.base & ~0xFFFull; a < g.base + g.size; a += 0x1000, ++n) have += a >> 12 < kPieces && written[a >> 12];
        stats.trace_lines.back() += std::format(" lv{} mem {}%", g.levels, n ? have * 100 / n : 0);
        traced_tex.insert(hash);
    }
    AgainstPending(g, storage);
    Check(g);
    Resync(g);
    if (storage && !g.storage) return tex_keep = false, Note(std::format("format {} has no storage image support", (int)g.fmt)), 0;
    if (storage && std::none_of(stores.begin(), stores.end(), [&](auto& s) { return s.first == &g; }))
        stores.push_back({&g, std::min<u32>(t[3] >> 12 & 0xF, g.img_levels - 1)});
    static constexpr VkImageViewType vt[]{VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_VIEW_TYPE_CUBE};
    if (kind > 3 || (kind == 2) != g.volume || (kind == 3 && g.layers < 6)) return tex_keep = false, Note(std::format("T# view kind {} on its image", kind)), 0;
    const u32 base_level = std::min<u32>(t[3] >> 12 & 0xF, g.img_levels - 1), base_array = std::min<u32>(t[5] & 0x1FFF, g.layers - 1);
    const VkComponentMapping map = storage ? VkComponentMapping{} : g.map;
    // The T#s of one surface image share its views; a shorter one is its top-left part (`sub` below, bounded in the
    // shader). GOW3's grab passes copy each effect's screen rectangle through a T# as tall as the rectangle's bottom edge
    // (~8 heights a frame in Rage of Sparta), each one a view and a storage element of its own before.
    Views& views = g.surf ? g.surf->views : g.views;
    const ViewKey vk{g.t.img, binding, g.fmt, u32(map.r | map.g << 3 | map.b << 6 | map.a << 9) | base_level << 12 | base_array << 16};
    auto v = views.find(vk);
    if (v == views.end()) {
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, base_level, storage ? 1 : g.img_levels - base_level,
                                            kind == 0 ? base_array : 0, kind == 0 ? 1 : kind == 3 ? g.layers / 6 * 6 : g.layers};
        const VkImageView view = View(g.t.img, vt[kind], g.fmt, range, map, g.t.usage);
        if (!view) return tex_keep = false, 0;
        const u32 el = Bindless(binding, view, VK_NULL_HANDLE);
        if (!el) return vkDestroyImageView(dev, view, nullptr), tex_keep = false, 0;
        v = views.emplace(vk, ViewEntry{view, el, g.surf ? g.shared : nullptr}).first;
    }
    // Smaller than its surface's image (a flat T#): the top-left part.
    const u64 sub = g.w < g.iw || g.h < g.ih ? (u64)(g.w | g.h << 16) | (u64)g.iw << 32 | (u64)g.ih << 48 : 0;
    return tex_sub = sub, tex_used = memo[key] = {&g, v->second.element, VK_NULL_HANDLE, sub}, v->second.element;
}

// S# -> VkSampler (shadPS4 Sampler fields and liverpool_to_vk.cpp mappings); unnormalized coordinates are scaled in the shader.
u32 Renderer::Impl::Sampler(const u32* s) {
    if (stats.trace && !stats.trace_lines.empty()) stats.trace_lines.back() += std::format(" | S {:08x} {:08x} {:08x} {:08x}", s[0], s[1], s[2], s[3]);
    const u64 key = ::Gpu::PieceHash::SamplerKey(s, 16);
    auto& memo = Memos().samplers;
    if (const u32* e = memo.Find(key)) return *e;
    std::scoped_lock res(res_lock);
    if (auto it = samplers.find(key); it != samplers.end()) return memo[key] = it->second;
    const u64 r0 = (u64)s[1] << 32 | s[0], r1 = (u64)s[3] << 32 | s[2];
    auto clamp = [](u64 m) {
        static constexpr VkSamplerAddressMode t[]{VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
                                                  VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
                                                  VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
                                                  VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE};
        return t[m & 7];
    };
    const u32 aniso = r0 >> 9 & 7, compare = r0 >> 12 & 7, reduction = r0 >> 29 & 3, mag = r1 >> 20 & 3, min = r1 >> 22 & 3, mip = r1 >> 26 & 3;
    VkSamplerReductionModeCreateInfo rm{VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO, nullptr,
                                        reduction == 1 ? VK_SAMPLER_REDUCTION_MODE_MIN : reduction == 2 ? VK_SAMPLER_REDUCTION_MODE_MAX
                                                                                        : VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE};
    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, &rm};
    ci.magFilter = mag & 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST, ci.minFilter = min & 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.mipmapMode = mip == 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = clamp(r0), ci.addressModeV = clamp(r0 >> 3), ci.addressModeW = clamp(r0 >> 6);
    ci.mipLodBias = float(s16(((r1 & 0x3FFF) ^ 0x2000) - 0x2000)) / 256.0f;
    ci.anisotropyEnable = aniso && (mag >= 2 || min >= 2), ci.maxAnisotropy = float(1u << std::min(aniso, 4u));
    ci.compareEnable = compare != 0, ci.compareOp = VkCompareOp(compare);
    ci.minLod = float(r0 >> 32 & 0xFFF) / 256.0f, ci.maxLod = float(r0 >> 44 & 0xFFF) / 256.0f;
    if (anisotropy > 1 && (mag & 1) && (min & 1) && mip == 2 && !compare && !reduction && ci.maxLod > ci.minLod)
        ci.anisotropyEnable = VK_TRUE, ci.maxAnisotropy = anisotropy;
    if (const char* f = DebugOption("GOW3_FORCE_LOD")) ci.minLod = ci.maxLod = std::strtof(f, nullptr);  // diagnostic: one mip only
    const u32 border = r1 >> 62;
    ci.borderColor = border == 1 ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK : border == 2 ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    if (border == 3) Note("S# custom border color (transparent black used)");
    VkSampler smp = VK_NULL_HANDLE;
    vkCreateSampler(dev, &ci, nullptr, &smp);
    const u32 e = Bindless(kSamplerBinding, VK_NULL_HANDLE, smp);
    if (!e) return vkDestroySampler(dev, smp, nullptr), 0;
    owned_samplers.push_back(smp);
    return memo[key] = samplers[key] = e;
}

// ---- Detiling, uploads and write-backs ----

VkImageView Renderer::Impl::MipView(GuestTex& g, u32 m, bool storage) {
    if (m >= g.img_levels || g.block || g.volume) return VK_NULL_HANDLE;
    const VkImageUsageFlags need = storage ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!(g.t.usage & need)) return VK_NULL_HANDLE;
    const VkFormat fmt = UintFormat(g.bits);
    if (!fmt) return VK_NULL_HANDLE;
    if (!g.mip_views[m]) {
        g.mip_views[m] = View(g.t.img, VK_IMAGE_VIEW_TYPE_2D_ARRAY, fmt, {VK_IMAGE_ASPECT_COLOR_BIT, m, 1, 0, g.layers}, {},
                              g.t.usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT));
        if (g.mip_views[m]) g.owned.push_back(g.mip_views[m]);
    }
    return g.mip_views[m];
}

// tiling.comp specialized like shadPS4 TileManager::GetTilingPipeline (mode 0: buffer detiler, 1: buffer tiler, 2: image detiler, 3: image tiler).
VkPipeline Renderer::Impl::Detiler(u32 tile, u32 bits, u32 mode) {
    using namespace AmdGpu;
    const u32 key = tile << 8 | bits | mode << 16;
    if (auto it = detilers.find(key); it != detilers.end()) return it->second;
    const auto tile_mode = TileMode(tile);
    const auto array_mode = GetArrayMode(tile_mode);
    const auto micro = GetMicroTileMode(tile_mode);
    std::vector<std::pair<std::string, std::string>> m{{"BITS_PER_PIXEL", std::to_string(bits)},
                                                       {"NUM_SAMPLES", "1"},
                                                       {"ARRAY_MODE", std::to_string(u32(array_mode))},
                                                       {"MICRO_TILE_MODE", std::to_string(u32(micro))},
                                                       {"MICRO_TILE_THICKNESS", std::to_string(GetMicroTileThickness(array_mode))}};
    if (mode == 1 || mode == 3) m.push_back({"IS_TILER", "1"});
    if (mode == 2) m.push_back({"DETILE_TO_IMAGE", "1"});
    if (mode == 3) m.push_back({"TILE_FROM_IMAGE", "1"});
    if (IsMacroTiled(array_mode)) {
        const auto macro = CalculateMacrotileMode(tile_mode, bits, 1);
        const u32 banks = GetNumBanks(macro);
        m.insert(m.end(), {{"PIPE_CONFIG", std::to_string(u32(GetPipeConfig(tile_mode)))},
                           {"BANK_WIDTH", std::to_string(GetBankWidth(macro))},
                           {"BANK_HEIGHT", std::to_string(GetBankHeight(macro))},
                           {"NUM_BANKS", std::to_string(banks)},
                           {"NUM_BANK_BITS", std::to_string(std::bit_width(banks) - 1)},
                           {"TILE_SPLIT_BYTES", std::to_string(CalculateTileSplit(tile_mode, array_mode, micro, bits))},
                           {"MACRO_TILE_ASPECT", std::to_string(GetMacrotileAspect(macro))}});
    }
    std::string err;
    const auto spirv = compiler.Spirv(tiling_src, shaderc_compute_shader, err, m);
    VkPipeline p = VK_NULL_HANDLE;
    if (!spirv.empty()) {
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, spirv.size() * 4, spirv.data()};
        VkShaderModule mod;
        vkCreateShaderModule(dev, &mi, nullptr, &mod);
        VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main"};
        pci.layout = tiling_layout;
        vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pci, nullptr, &p);
        vkDestroyShaderModule(dev, mod, nullptr);
    }
    if (!p) Note(std::format("{} tile {} bits {}: {}", (mode & 1) ? "tiler" : "detiler", tile, bits, err.substr(0, 300)));
    return detilers[key] = p;
}

void Renderer::Impl::DispatchDetile(VkCommandBuffer c, u32 mip, u32 bytes, u32 bits) {
    // Whole mips through the generic texel walk (shadPS4 MakeTilingRegions without a tile size), 64 texels per group.
    for (u32 first = 0, groups = (bytes / (bits / 8) + 63) / 64; first < groups;) {
        const u32 n = std::min(groups - first, 65535u), region[5]{mip, first, n, 0, 0};
        vkCmdPushConstants(c, tiling_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(region), region), dyn.pushed = false;
        vkCmdDispatch(c, n, 1, 1);
        first += n;
    }
}

// Texture uploads: tiled non-block 2D/array textures are detiled directly into their image via `DETILE_TO_IMAGE`;
// block-compressed and volume tiled textures detile into `buf` (grown to at least `reserve`) and then copy into the image.
// Linear ones copy from the mirror, through `buf` (or a Temp slice when null) when `image_copy_via_temp`.
// `frag`: no vertex shader samples them before the next full barrier.
void Renderer::Impl::Upload(VkCommandBuffer c, std::vector<GuestTex*>& list, Buffer* buf, u64 reserve, bool frag) {
    struct Job {
        GuestTex* g;
        u64 scratch;   // offset of the detiled copy, or ~0 for a linear or direct-to-image texture
        u64 info;      // TilingInfo offset in the tables
        bool direct;   // detiled directly into g->t.img via DETILE_TO_IMAGE
    };
    std::vector<Job> jobs;
    u64 need = 0;
    const bool prologue = &list == &uploads;
    for (GuestTex* g : list) {
        Job j{g, ~0ull, 0, false};
        if (g->tile != 8 && g->tile != 31) {
            TilingInfo ti{0, g->volume ? g->d : g->layers, g->levels};
            std::copy_n(g->mips, 16, ti.mips);
            if (g->block)
                for (u32 m = 0; m < g->levels; ++m) ti.mips[m].pitch = std::max((ti.mips[m].pitch + 3) / 4, 1u), ti.mips[m].height = std::max((ti.mips[m].height + 3) / 4, 1u);
            ti.image_width = g->w, ti.image_height = g->h, ti.range_end = ~0u, ti.block_scale = g->block ? 4 : 1, ti.is_volume = g->volume;
            const u64 at = TableAt(sizeof(ti), 256);
            if (at == ~0ull) {
                g->reload = true;
                if (g->surf && g->t.img == g->surf->img) g->surf->current = g->surf->valid_frame = ~0u, g->surf->valid_w = g->surf->valid_h = 0;
                Note("draw tables full (tiling info)");
                continue;
            }
            std::memcpy(TablePtr() + at, &ti, sizeof(ti));
            j.info = at;
            j.direct = !g->block && !g->volume && (g->t.usage & VK_IMAGE_USAGE_STORAGE_BIT) && UintFormat(g->bits) && Detiler(g->tile, g->bits, 2);
            if (j.direct) {
                for (u32 m = 0; m < g->img_levels; ++m)
                    if (!MipView(*g, m, true)) {
                        j.direct = false;
                        break;
                    }
            }
            if (!j.direct) {
                const u64 scratch_bytes = g->levels == 1 ? g->ActiveMipSize(0) : g->size;
                j.scratch = need, need = (need + scratch_bytes + temp_align - 1) & ~(temp_align - 1);
            }
        } else if (image_copy_via_temp) {
            j.scratch = need, need = (need + g->size + temp_align - 1) & ~(temp_align - 1);
        }
        jobs.push_back(j);
    }
    // A new image still queued for the prologue gets its layout there (the prologue runs first, though recorded last).
    for (auto& j : jobs)
        if (j.g->fresh && (prologue || std::find(uploads.begin(), uploads.end(), j.g) == uploads.end()))
            ToGeneral(c, j.g->t.img, VK_IMAGE_ASPECT_COLOR_BIT), j.g->fresh = false;
    list.clear();
    if (jobs.empty()) return;
    if (prologue) scratch_used = need;
    VkBuffer dst_buf = VK_NULL_HANDLE;
    u64 dst_base = 0;
    if (need > 0) {
        if (buf) {
            if (buf->size < need) {
                FreeBuffer(*buf);
                const u64 alloc_size = std::max(need, Pressed() ? std::min(std::bit_ceil(need), reserve) : reserve);
                *buf = MakeBuffer(alloc_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false,
                                  prologue ? "scratch" : "upload temp");
                if (!buf->buf && alloc_size > need)
                    *buf = MakeBuffer(need, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false,
                                      prologue ? "scratch" : "upload temp");
            }
            dst_buf = buf->buf;
        } else {
            const TempSlice s = Temp(need);
            dst_buf = s.buf, dst_base = s.offset;
        }
        if (!dst_buf) Note("texture upload buffer allocation failed");
    }
    bool detiled_buf = false;
    std::vector<VkBufferCopy> linear;  // linear textures into `dst_buf`
    for (auto& j : jobs) {
        if (j.scratch == ~0ull && !j.direct) continue;
        if (j.scratch != ~0ull && !dst_buf) {
            j.g->reload = true;
            if (j.g->surf && j.g->t.img == j.g->surf->img) j.g->surf->current = j.g->surf->valid_frame = ~0u, j.g->surf->valid_w = j.g->surf->valid_h = 0;
            continue;
        }
        GuestTex& g = *j.g;
        if (g.tile == 8 || g.tile == 31) {
            linear.push_back({g.base, dst_base + j.scratch, g.size});
            continue;
        }
        const VkPipeline p = Detiler(g.tile, g.bits, j.direct ? 2 : 0);
        if (!p) continue;
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, p);
        if (j.direct) {  // T#s sharing a surface image write the same bytes of one memory: no order between them, as for copies
            const VkDescriptorBufferInfo in{mirror, g.base, g.size}, info{tables.buf, j.info, sizeof(TilingInfo)};
            for (u32 m = 0; m < g.img_levels; ++m) {
                const VkDescriptorImageInfo dst{VK_NULL_HANDLE, MipView(g, m, true), VK_IMAGE_LAYOUT_GENERAL};
                const VkWriteDescriptorSet w[]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &in},
                                               {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 2, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &info},
                                               {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dst}};
                PushDescriptors(c, VK_PIPELINE_BIND_POINT_COMPUTE, tiling_layout, 0, 3, w);
                DispatchDetile(c, m, g.ActiveMipSize(m), g.bits);
            }
            continue;
        }
        detiled_buf = true;
        const u64 scratch_bytes = g.levels == 1 ? g.ActiveMipSize(0) : g.size;
        const VkDescriptorBufferInfo in{mirror, g.base, g.size}, out{dst_buf, dst_base + j.scratch, scratch_bytes}, info{tables.buf, j.info, sizeof(TilingInfo)};
        const VkWriteDescriptorSet w[]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &in},
                                       {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &out},
                                       {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 2, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &info}};
        PushDescriptors(c, VK_PIPELINE_BIND_POINT_COMPUTE, tiling_layout, 0, 3, w);
        for (u32 m = 0; m < g.levels; ++m) DispatchDetile(c, m, g.ActiveMipSize(m), g.bits);
    }
    if (!linear.empty()) vkCmdCopyBuffer(c, mirror, dst_buf, (u32)linear.size(), linear.data());
    if (detiled_buf || !linear.empty())  // one barrier for both ways into `dst_buf`
        StageBarrier(c, (detiled_buf ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : 0) | (linear.empty() ? 0 : VK_PIPELINE_STAGE_2_COPY_BIT),
                     (detiled_buf ? VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT : 0) | (linear.empty() ? 0 : VK_ACCESS_2_TRANSFER_WRITE_BIT),
                     VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    for (auto& j : jobs) {
        if (j.direct) continue;
        if (j.scratch != ~0ull && !dst_buf) continue;
        GuestTex& g = *j.g;
        std::vector<VkBufferImageCopy> regions;
        regions.reserve(g.img_levels);
        for (u32 m = 0; m < g.img_levels; ++m)
            regions.push_back({(j.scratch == ~0ull ? g.base : dst_base + j.scratch) + g.mips[m].offset, g.mips[m].pitch, g.mips[m].height,
                               {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, g.layers},
                               {},
                               {std::max(g.w >> m, 1u), std::max(g.h >> m, 1u), std::max(g.d >> m, 1u)}});
        vkCmdCopyBufferToImage(c, j.scratch == ~0ull ? mirror : dst_buf, g.t.img, VK_IMAGE_LAYOUT_GENERAL, (u32)regions.size(), regions.data());
    }
    Barrier(c, frag);
}

// One mip of a storage image to its guest address in the mirror (linear: one copy; tiled: direct from image or through a buffer).
bool Renderer::Impl::Store(VkCommandBuffer c, GuestTex& g, u32 m) {
    const Layout& l = g.mips[m];
    const bool linear = g.tile == 8 || g.tile == 31;
    BindPages(g.base + l.offset, g.base + l.offset + l.size);
    const VkExtent3D ext{std::max(g.w >> m, 1u), std::max(g.h >> m, 1u), std::max(g.d >> m, 1u)};
    VkBufferImageCopy bc{0, l.pitch, l.height, {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, g.layers}, {}, ext};
    if (linear) return ImageToMirror(c, g.t.img, bc, g.base + l.offset, (u64)ext.width * g.bits / 8, (u64)l.pitch * g.bits / 8, ext.depth * g.layers), true;
    VkImageView view = MipView(g, m, false);
    VkPipeline p = view ? Detiler(g.tile, g.bits, 3) : VK_NULL_HANDLE;
    if (!p) view = VK_NULL_HANDLE, p = Detiler(g.tile, g.bits, 1);  // through a buffer when the image tiler is missing
    if (!p) return false;
    // TilingInfo of this one mip, rebased to offset 0 on both sides (the mirror binding starts at the mip).
    TilingInfo ti{0, g.layers, m + 1};
    ti.mips[m] = {l.size, l.pitch, l.height, 0};
    ti.image_width = g.w, ti.image_height = g.h, ti.range_end = ~0u, ti.block_scale = 1;
    const u64 at = TableAt(sizeof(ti), 256);
    if (at == ~0ull) return Note("draw tables full (tiling info)"), false;
    std::memcpy(TablePtr() + at, &ti, sizeof(ti));
    if (view) {
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, p);
        const VkDescriptorBufferInfo out{mirror, g.base + l.offset, l.size}, info{tables.buf, at, sizeof(ti)};
        const VkDescriptorImageInfo src{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
        const VkWriteDescriptorSet w[]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &out},
                                       {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 2, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &info},
                                       {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 3, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &src}};
        PushDescriptors(c, VK_PIPELINE_BIND_POINT_COMPUTE, tiling_layout, 0, 3, w);
    } else {
        const TempSlice tmp = Temp(l.size);
        if (!tmp.buf) return Note("store temp buffer allocation failed"), false;
        bc.bufferOffset = tmp.offset;
        vkCmdCopyImageToBuffer(c, g.t.img, VK_IMAGE_LAYOUT_GENERAL, tmp.buf, 1, &bc);
        StageBarrier(c, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, p);
        const VkDescriptorBufferInfo out{mirror, g.base + l.offset, l.size}, in{tmp.buf, tmp.offset, l.size}, info{tables.buf, at, sizeof(ti)};
        const VkWriteDescriptorSet w[]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &out},
                                       {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &in},
                                       {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE, 2, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &info}};
        PushDescriptors(c, VK_PIPELINE_BIND_POINT_COMPUTE, tiling_layout, 0, 3, w);
    }
    DispatchDetile(c, m, g.ActiveMipSize(m), g.bits);
    return true;
}

// An image copy to the mirror at `to` (`bc` laid out from offset 0: rows of `pitch` bytes, `bc.bufferImageHeight` rows a
// slice). Through a temp buffer (image_copy_via_temp) only the `row` bytes of each row the copy wrote go on to the mirror:
// what lies between them there stays.
void Renderer::Impl::ImageToMirror(VkCommandBuffer c, VkImage img, VkBufferImageCopy bc, u64 to, u64 row, u64 pitch, u32 slices) {
    if (!image_copy_via_temp) return bc.bufferOffset = to, vkCmdCopyImageToBuffer(c, img, VK_IMAGE_LAYOUT_GENERAL, mirror, 1, &bc);
    std::scoped_lock res(res_lock);
    const u32 rows = bc.imageExtent.height, height = bc.bufferImageHeight;
    const TempSlice tmp = Temp(((u64)(slices - 1) * height + rows - 1) * pitch + row);
    if (!tmp.buf) return Note("image-to-mirror temp buffer allocation failed");
    bc.bufferOffset = tmp.offset;
    vkCmdCopyImageToBuffer(c, img, VK_IMAGE_LAYOUT_GENERAL, tmp.buf, 1, &bc);
    StageBarrier(c, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    std::vector<VkBufferCopy> r;
    for (u32 s = 0; s < slices; ++s) {
        const u64 at = (u64)s * height * pitch;
        if (row == pitch) r.push_back({tmp.offset + at, to + at, (rows - 1) * pitch + row});
        else
            for (u32 y = 0; y < rows; ++y) r.push_back({tmp.offset + at + y * pitch, to + at + y * pitch, row});
    }
    vkCmdCopyBuffer(c, tmp.buf, mirror, (u32)r.size(), r.data());
}

// A write left in its image (WriteBack `lazy`) to the mirror, for a view that reads the memory another way: everything
// recorded before is done (the caller's barrier), the reads after wait for it. `learn`: the reader is not one of the
// surface's T#s, so its later writes go back at once.
void Renderer::Impl::Materialize(VkCommandBuffer c, const GpuWrite& w, bool learn) {
    if (!w.by) return Note("write left in an evicted image lost");
    GuestTex& g = *w.by;
    if (learn) g.surf->foreign = true;
    for (u32 m = 0; m < g.levels; ++m)
        if (g.base + g.mips[m].offset == w.lo) {
            Store(c, g, m);
            if (temp_copylog) TempCopy("MAT", g, std::format(" | mip {}", m));
        }
    Barrier(c, true);
}

// After a dispatch or draw: each storage image it wrote goes back to the mirror, so every other view of that memory sees
// the write. A tiled surface's stays in its image while only the surface's own T#s use it (they sample that image):
// Materialize writes it back when another view reads the memory.
void Renderer::Impl::WriteBack() {
    if (stores.empty()) return;
    EndRendering();
    std::scoped_lock res(res_lock);
    bool shared = false;  // other T#s sample this image directly, vertex shaders included
    for (auto [gp, m] : stores) {
        GuestTex& g = *gp;
        const Layout& l = g.mips[m];
        const bool linear = g.tile == 8 || g.tile == 31;
        if (g.block || (g.volume && !linear)) {
            Note("block or tiled volume storage image not written back");
            continue;
        }
        shared |= g.surf != nullptr;
        const bool lazy = g.surf && !linear && !g.surf->foreign && !g.surf->growing;
        if (!lazy && !Store(cmd, g, m)) continue;
        if (temp_copylog) TempCopy(lazy ? "WB-lazy" : linear ? "WB-linear" : "WB-tiled", g, std::format(" | mip {} {:.2f} MB", m, l.size / 1048576.0));
        const u64 seq = std::atomic_ref(write_seq).fetch_add(1) + 1;
        const GpuWrite w{g.base + l.offset, g.base + l.offset + l.size, seq, &g, lazy};
        if (in_worker) {
            own_writes.push_back(w), (*synced_here)[&g] = seq;
            std::scoped_lock nl(note_lock);
            storage_draw_progs.insert(draw_ps_addr ^ (draw_vs_addr * 0x9E3779B97F4A7C15ull));
            has_storage_draw_progs.store(true, std::memory_order_relaxed);
        } else {
            gpu_writes.push_back(w), g.synced = seq;
        }
    }
    stores.clear();
    // The copies write image memory in the mirror, which uploads (transfers, compute) and shaders sample as textures; the
    // shader's own writes are behind the barrier before, for the stages after the vertex ones (a draw's).
    Barrier(cmd, !shared);
}

void Renderer::Impl::SyncLinearBacking(u64 dst_lo, u64 src_lo, u64 copy_bytes) {
#ifndef GOW3_OFFLINE
    InvalidateBlocks(dst_lo, copy_bytes);
    const u8* src = reinterpret_cast<const u8*>(src_lo);
    for (u64 o = 0; o < copy_bytes;) {
        const u64 a = dst_lo + o, p_idx = a >> 12, page_addr = a & ~0xFFFull;
        const u64 n = std::min<u64>(0x1000 - (a & 0xFFFull), copy_bytes - o);
        if (n == 0x1000) {
            const u64 h = Gow3::GuestMemory::CopyBacking(a, src + o, 0x1000, true);
            if (p_idx < kPieces) written[p_idx] = h;
        } else {
            Gow3::GuestMemory::CopyBacking(a, src + o, n, false);
            if (p_idx < kPieces) {
                const char before = Gow3::GuestMemory::PageKind(page_addr);
                if (written[p_idx] && (before == 'C' || before == 'F')) {
                    const u64 h = PieceHash(reinterpret_cast<const u8*>(page_addr), 0x1000);
                    const char after = Gow3::GuestMemory::PageKind(page_addr);
                    written[p_idx] = (after == 'C' || after == 'F') ? h : 0;
                } else {
                    written[p_idx] = 0;
                }
            }
        }
        o += n;
    }
#else
    (void)dst_lo, (void)src_lo, (void)copy_bytes;
#endif
}

void Renderer::Impl::FlushLinearSyncs() {
    for (const LinSync& s : lin_syncs) SyncLinearBacking(s.dst, s.src, s.size);
    lin_syncs.clear();
}

bool Renderer::Impl::TryLinearCopyDispatch(const u32* b, const CpuRead& cpu) {
    Prepared& p = prep;
    if (sh[0x207] != 8 || sh[0x208] != 8 || sh[0x209] != 1) return false;
    if (!p.slots || p.slots->size() != 2 || (*p.slots)[0].kind != Gcn::SlotKind::Texture || (*p.slots)[1].kind != Gcn::SlotKind::Storage)
        return false;
    if (p.desc.size() != 2 || !p.desc[0] || !p.desc[1] || !p.fdesc.empty() || p.stored) return false;
    if (!p.consts || p.consts->empty()) return false;
    const Gcn::Const& c = (*p.consts)[0];
    if (!c.buffer || c.lo != 0 || c.hi < 24) return false;
    const u32* cb_v = Locate(c.path, 4, &sh[0x240], cpu);
    if (!cb_v) return false;
    const u64 cb_addr = (u64)(cb_v[1] & 0xFFFF) << 32 | cb_v[0], cb_stride = cb_v[1] >> 16 & 0x3FFF;
    const u64 cb_lim = cb_stride ? cb_stride * cb_v[2] : cb_v[2];
    if (cb_lim < 24) return false;
    const u32* cb = static_cast<const u32*>(cpu(cb_addr, 24));
    if (!cb) return false;
    const u32 dst_x = cb[0], dst_y = cb[1], src_x = cb[2], src_y = cb[3], copy_w = cb[4], copy_h = cb[5];
    if (!copy_w || !copy_h || b[2] != 1 || (u64)b[0] * 8 < copy_w || (u64)b[1] * 8 < copy_h) return false;

    struct LinearT {
        u64 base = 0;
        u32 w = 0, h = 0, pitch = 0, tile = 0, bits = 0, comps = 0;
        AmdGpu::DataFormat raw_dfmt{};
        AmdGpu::NumberFormat raw_nfmt{};
    };
    auto parse = [&](const u32* t, LinearT& out) -> bool {
        if (!(t[3] >> 31)) return false;
        out.base = ((u64)t[0] | (u64)(t[1] & 0x3F) << 32) << 8;
        if (!out.base) return false;
        const u32 type = t[3] >> 28, base_level = t[3] >> 12 & 0xF, last_level = t[3] >> 16 & 0xF;
        const u32 depth = (t[4] & 0x1FFF) + 1, base_array = t[5] & 0x1FFF;
        const bool pow2 = t[3] >> 25 & 1;
        out.tile = t[3] >> 20 & 0x1F;
        if (type != 9 || base_level || last_level || depth != 1 || base_array || pow2) return false;
        if (out.tile != 8 && out.tile != 31) return false;
        out.raw_dfmt = AmdGpu::DataFormat(t[1] >> 20 & 0x3F);
        out.raw_nfmt = AmdGpu::NumberFormat(t[1] >> 26 & 0xF);
        const auto dfmt = AmdGpu::RemapDataFormat(out.raw_dfmt);
        const auto nfmt = AmdGpu::RemapNumberFormat(out.raw_nfmt, out.raw_dfmt);
        if (out.raw_nfmt == AmdGpu::NumberFormat::Srgb || AmdGpu::IsBlockCoded(dfmt)) return false;
        if (SurfaceFormat(u32(dfmt), u32(nfmt)) == VK_FORMAT_UNDEFINED) return false;
        out.bits = AmdGpu::NumBitsPerBlock(dfmt);
        out.comps = AmdGpu::NumComponents(dfmt);
        if (out.bits < 8 || out.bits > 128 || (out.bits & 7) || !out.comps || out.comps > 4) return false;
        const auto sel = DstSel(t);
        for (u32 k = 0; k < out.comps; ++k)
            if (sel.array[k] != AmdGpu::CompSwizzle(u8(AmdGpu::CompSwizzle::Red) + k)) return false;
        out.w = (t[2] & 0x3FFF) + 1;
        out.h = (t[2] >> 14 & 0x3FFF) + 1;
        out.pitch = (t[4] >> 13 & 0x3FFF) + 1;
        return out.w <= out.pitch;
    };
    LinearT src, dst;
    if (!parse(p.desc[0], src) || !parse(p.desc[1], dst)) return false;
    if (src.raw_dfmt != dst.raw_dfmt || src.raw_nfmt != dst.raw_nfmt) return false;
    if ((u64)src_x + copy_w > src.w || (u64)src_y + copy_h > src.h || (u64)dst_x + copy_w > dst.w || (u64)dst_y + copy_h > dst.h)
        return false;
    if ((copy_h > 1 || src_y > 0) && std::get<0>(VideoCore::ImageSizeLinearAligned(src.pitch, src.h, src.bits, 1)) != src.pitch)
        return false;
    if ((copy_h > 1 || dst_y > 0) && std::get<0>(VideoCore::ImageSizeLinearAligned(dst.pitch, dst.h, dst.bits, 1)) != dst.pitch)
        return false;
    if (copy_h > 1 && copy_w != dst.pitch) return false;

    const u64 bpp = src.bits / 8, row_bytes = (u64)copy_w * bpp;
    const u64 src_stride = (u64)src.pitch * bpp, dst_stride = (u64)dst.pitch * bpp;
    const u64 src_lo = src.base + (u64)src_y * src_stride + (u64)src_x * bpp;
    const u64 src_hi = src.base + ((u64)(src_y + copy_h - 1) * src.pitch + src_x + copy_w) * bpp;
    const u64 dst_lo = dst.base + (u64)dst_y * dst_stride + (u64)dst_x * bpp;
    const u64 dst_hi = dst.base + ((u64)(dst_y + copy_h - 1) * dst.pitch + dst_x + copy_w) * bpp;
    if (src_hi <= src_lo || src_hi > mirror_size || dst_hi <= dst_lo || dst_hi > mirror_size) return false;
    if (src_lo < dst_hi && dst_lo < src_hi) return false;
#ifndef GOW3_OFFLINE
    if (Gow3::GpuThread::VideoFrameVersion(src.base) || Gow3::GpuThread::VideoFrameVersion(dst.base)) return false;
#endif

    FlushStores();
    {
        std::scoped_lock res(res_lock);
        auto blocked_range = [&](u64 a) { return (a >= src_lo && a < src_hi) || (a >= dst_lo && a < dst_hi); };
        if (target_at.contains(src.base) || target_at.contains(dst.base) || stencil_at.contains(src.base) || stencil_at.contains(dst.base) ||
            std::any_of(target_at.begin(), target_at.end(), blocked_range))
            return false;
        auto has_lazy = [&](const std::vector<GpuWrite>& list) {
            return std::any_of(list.begin(), list.end(), [&](const GpuWrite& w) {
                return w.lazy && ((w.lo < src_hi && w.hi > src_lo) || (w.lo < dst_hi && w.hi > dst_lo));
            });
        };
        if (has_lazy(lazy_writes) || has_lazy(gpu_writes) || has_lazy(own_writes)) return false;
    }

    const u64 copy_bytes = (u64)copy_h * row_bytes;
    const bool contiguous = copy_h == 1 || (copy_w == src.pitch && copy_w == dst.pitch);

#ifndef GOW3_OFFLINE
    auto over_writes = [&](const std::vector<GpuWrite>& list, u64 lo, u64 hi, bool unsynced_only) {
        return std::any_of(list.begin(), list.end(), [&](const GpuWrite& w) {
            return (!unsynced_only || !w.guest_synced) && w.lo < hi && w.hi > lo;
        });
    };
    const bool can_sync_guest =
        contiguous && !stats.trace && !boom_on && !in_worker &&
        Gow3::GpuThread::InTextureHeap(src_lo, src_hi) && Gow3::GpuThread::InTextureHeap(dst_lo, dst_hi) &&
        !(src_lo < marked_hi && marked_lo < src_hi) && !(dst_lo < marked_hi && marked_lo < dst_hi) &&
        std::none_of(flight.back.begin(), flight.back.end(), [&](const VkBufferCopy& cf) {
            const u64 clo = cf.srcOffset, chi = clo + cf.size;
            return (clo < src_hi && chi > src_lo) || (clo < dst_hi && chi > dst_lo);
        }) &&
        !over_writes(gpu_writes, src_lo, src_hi, true) && !over_writes(gpu_writes, dst_lo, dst_hi, true) &&
        !over_writes(last_writes, src_lo, src_hi, true) && !over_writes(last_writes, dst_lo, dst_hi, true) &&
        !over_writes(own_writes, src_lo, src_hi, true) && !over_writes(own_writes, dst_lo, dst_hi, true) &&
        Gow3::GuestMemory::Readable(src_lo, copy_bytes) && Gow3::GuestMemory::HasBacking(dst_lo, dst_hi);

    if (can_sync_guest) {
        const bool has_pending_parallel = std::any_of(blocks.begin(), blocks.end(), [](const Block& blk) { return !blk.serial; });
        const bool src_in_lin_syncs = std::any_of(lin_syncs.begin(), lin_syncs.end(), [&](const LinSync& s) {
            return s.dst < src_hi && src_lo < s.dst + s.size;
        });
        const bool src_unknown = !src_in_lin_syncs && Gow3::GuestMemory::AllUnknown(src_lo, src_hi);
        auto over_dirty = [&](u64 lo, u64 hi) {
            auto d = std::upper_bound(dirty.begin(), dirty.end(), std::pair{hi, 0ull});
            return d != dirty.begin() && std::prev(d)->second > lo;
        };

        // Sub-case A: both src and dst are untracked (Unknown), no parallel blocks precede us, and dst has no mirror writes.
        if (src_unknown && !has_pending_parallel && !over_dirty(dst_lo, dst_hi) && Gow3::GuestMemory::AllUnknown(dst_lo, dst_hi) &&
            !over_writes(gpu_writes, dst_lo, dst_hi, false) && !over_writes(last_writes, dst_lo, dst_hi, false) &&
            !over_writes(own_writes, dst_lo, dst_hi, false)) {
            Gow3::GuestMemory::CopyBacking(dst_lo, reinterpret_cast<const void*>(src_lo), copy_bytes, false);
            InvalidateBlocks(dst_lo, copy_bytes);
            for (u64 pg = dst_lo >> 12, end = std::min((dst_hi + 0xFFF) >> 12, kPieces); pg < end; ++pg) staged_table.get()[pg] = frame;
            Count(stats.dispatches);
            return true;
        }

        // Sub-case B: cold src (AllUnknown), dst already mirrored/tracked -> stage src directly for dst, sync BackingPages, skip readback.
        if (src_unknown) {
            Pending sp{};
            {
                std::scoped_lock lock(stage_lock);
                sp = AllocateStage(copy_bytes);
                if (sp.chunk != ~0u && use_stage_dev) stage_sends.push_back(sp);
            }
            if (sp.chunk != ~0u) {
                std::memcpy(stage[sp.chunk].ptr + sp.src, reinterpret_cast<const void*>(src_lo), copy_bytes);
                EndRendering();
                BindPages(dst_lo, dst_hi);
                Barrier(cmd);
                const VkBufferCopy region{sp.src, dst_lo, copy_bytes};
                vkCmdCopyBuffer(cmd, (use_stage_dev ? stage_dev : stage)[sp.chunk].buf, mirror, 1, &region);
                Barrier(cmd);
                const u64 seq = std::atomic_ref(write_seq).fetch_add(1) + 1;
                gpu_writes.push_back({dst_lo, dst_hi, seq, nullptr, false, true});
                if (has_pending_parallel) lin_syncs.push_back({dst_lo, src_lo, copy_bytes});
                else SyncLinearBacking(dst_lo, src_lo, copy_bytes);
                Count(stats.dispatches);
                return true;
            }
        }

        // Sub-case C: hot/mixed src -> mirror->mirror copy + BackingPages sync on dst, skip RecordReadback.
        Track(src_lo, src_hi), NoteRead(src_lo, src_hi);
        EndRendering();
        BindPages(src_lo, src_hi), BindPages(dst_lo, dst_hi);
        Barrier(cmd);
        const VkBufferCopy region{src_lo, dst_lo, copy_bytes};
        vkCmdCopyBuffer(cmd, mirror, mirror, 1, &region);
        Barrier(cmd);
        const u64 seq = std::atomic_ref(write_seq).fetch_add(1) + 1;
        gpu_writes.push_back({dst_lo, dst_hi, seq, nullptr, false, true});
        if (has_pending_parallel) lin_syncs.push_back({dst_lo, src_lo, copy_bytes});
        else SyncLinearBacking(dst_lo, src_lo, copy_bytes);
        Count(stats.dispatches);
        return true;
    }
#endif

    Track(src_lo, src_hi), NoteRead(src_lo, src_hi);
    EndRendering();
    BindPages(src_lo, src_hi), BindPages(dst_lo, dst_hi);
    Barrier(cmd);
    std::vector<VkBufferCopy> regions;
    if (contiguous) {
        regions.push_back({src_lo, dst_lo, copy_bytes});
    } else {
        regions.reserve(copy_h);
        for (u32 y = 0; y < copy_h; ++y) regions.push_back({src_lo + (u64)y * src_stride, dst_lo + (u64)y * dst_stride, row_bytes});
    }
    vkCmdCopyBuffer(cmd, mirror, mirror, (u32)regions.size(), regions.data());
    Barrier(cmd);
    const u64 seq = std::atomic_ref(write_seq).fetch_add(1) + 1;
    (in_worker ? own_writes : gpu_writes).push_back({dst_lo, dst_hi, seq, nullptr});
    Count(stats.dispatches);
    return true;
}

// ---- Debug and offline readback ----

std::vector<Image> Renderer::ReadTextures(bool traced) {
    Impl& r = *d;
    r.Retire();
    std::vector<Image> out;
    for (auto& [k, g] : r.textures) {
        if (traced && !r.traced_tex.contains(k)) continue;
        VkFormatProperties fp{};
        if (g.t.img) vkGetPhysicalDeviceFormatProperties(r.pd, g.fmt, &fp);
        // The blit converts to RGBA8, which an integer format cannot do: such a format has no linear filtering.
        const bool readable = g.t.img && !g.volume && fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT &&
                              (!traced || fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
        if (!readable && traced) out.push_back({g.base, g.w, g.h, (u32)g.fmt});
        if (!readable) continue;
        for (u32 m : {0u, 2u}) {
            if (m >= g.img_levels || (traced && m)) break;
            const u32 sw = std::max(g.w >> m, 1u), sh = std::max(g.h >> m, 1u), w = traced ? std::min(sw, 2048u) : sw, h = traced ? std::min(sh, 2048u) : sh;
            Impl::Buffer buf = r.MakeBuffer((u64)w * h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
            Impl::Tex rgba = r.MakeImage(VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, w, h, 1, 1, VK_SAMPLE_COUNT_1_BIT,
                                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            if (!buf.buf || !rgba.img) {
                if (rgba.img) r.FreeImage(rgba);
                r.FreeBuffer(buf);
                continue;
            }
            r.Once([&](VkCommandBuffer c) {
                r.ToGeneral(c, rgba.img, VK_IMAGE_ASPECT_COLOR_BIT);
                const VkImageBlit b{{VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1}, {{0, 0, 0}, {(int)sw, (int)sh, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {(int)w, (int)h, 1}}};
                vkCmdBlitImage(c, g.t.img, VK_IMAGE_LAYOUT_GENERAL, rgba.img, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_NEAREST);
                r.Barrier(c);
                const VkBufferImageCopy bc{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {w, h, 1}};
                vkCmdCopyImageToBuffer(c, rgba.img, VK_IMAGE_LAYOUT_GENERAL, buf.buf, 1, &bc);
            });
            Image img{g.base, w, h, (u32)g.fmt, false, std::vector<u8>(buf.ptr, buf.ptr + (size_t)w * h * 4), m};
            for (size_t p = 3; !traced && p < img.rgba.size(); p += 4) img.rgba[p] = 255;  // alpha kept out of the picture
            out.push_back(std::move(img));
            r.FreeImage(rgba);
            r.FreeBuffer(buf);
        }
    }
    return out;
}

std::vector<Image> Renderer::ReadTargets() {
    Impl& r = *d;
    r.Retire();
    std::vector<Image> out;
    for (auto& [key, t] : r.targets) {
        const bool f16 = t.fmt == VK_FORMAT_R16G16B16A16_SFLOAT || t.fmt == VK_FORMAT_R16G16_SFLOAT || t.fmt == VK_FORMAT_R16_SFLOAT;
        const bool f32 = t.fmt == VK_FORMAT_R32G32B32A32_SFLOAT || t.fmt == VK_FORMAT_R32G32_SFLOAT || t.fmt == VK_FORMAT_R32_SFLOAT || t.depth;
        u32 texel = 4;
        switch (t.fmt) {
        case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_R16G16B16A16_UNORM:
        case VK_FORMAT_R32G32_SFLOAT: texel = 8; break;
        case VK_FORMAT_R32G32B32A32_SFLOAT: texel = 16; break;
        case VK_FORMAT_R8_UNORM: texel = 1; break;
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R16_SFLOAT:
        case VK_FORMAT_R16_UNORM:
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_R5G6B5_UNORM_PACK16: texel = 2; break;
        default: break;
        }
        if (t.depth && t.samples > 1) continue;  // multisampled depth has no resolve copy here
        Impl::Buffer buf = r.MakeBuffer((u64)t.w * t.h * texel, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
        if (!buf.buf) continue;
        Impl::Tex single{};
        if (t.samples > 1) {
            single = r.MakeImage(VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, t.fmt, t.w, t.h, 1, 1, VK_SAMPLE_COUNT_1_BIT,
                                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            if (!single.img) {
                r.FreeBuffer(buf);
                continue;
            }
        }
        r.Once([&](VkCommandBuffer c) {
            VkImage src = t.t.img;
            if (single.img) {
                r.ToGeneral(c, single.img, VK_IMAGE_ASPECT_COLOR_BIT);
                const VkImageResolve rs{{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {}, {t.w, t.h, 1}};
                vkCmdResolveImage(c, t.t.img, VK_IMAGE_LAYOUT_GENERAL, single.img, VK_IMAGE_LAYOUT_GENERAL, 1, &rs);
                r.Barrier(c);
                src = single.img;
            }
            VkBufferImageCopy bc{0, 0, 0, {(VkImageAspectFlags)(t.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1}, {}, {t.w, t.h, 1}};
            vkCmdCopyImageToBuffer(c, src, VK_IMAGE_LAYOUT_GENERAL, buf.buf, 1, &bc);
        });
        if (single.img) r.FreeImage(single);
        Image img{t.base, t.w, t.h, (u32)t.fmt, t.depth, std::vector<u8>((size_t)t.w * t.h * 4)};
        auto half = [](uint16_t h) {
            const u32 e = h >> 10 & 0x1F, m = h & 0x3FF;
            return (h >> 15 ? -1.0f : 1.0f) * (e ? std::ldexp(1.0f + m / 1024.0f, (int)e - 15) : std::ldexp(m / 1024.0f, -14));
        };
        auto to8 = [](float v) { return (u8)std::clamp(std::isnan(v) ? 0.0f : v * 255.0f + 0.5f, 0.0f, 255.0f); };
        float zmin = 1, zmax = 0;
        if (t.depth && t.fmt != VK_FORMAT_D16_UNORM)
            for (size_t i = 0; i < (size_t)t.w * t.h; ++i) {
                const float z = ((const float*)buf.ptr)[i];
                if (z < 1.0f) zmin = std::min(zmin, z), zmax = std::max(zmax, z);
            }
        for (size_t i = 0; i < (size_t)t.w * t.h; ++i) {
            u8* o = &img.rgba[i * 4];
            const u8* s = buf.ptr + i * texel;
            float c[4]{0, 0, 0, 1};
            if (t.depth) {
                const float z = t.fmt == VK_FORMAT_D16_UNORM ? *(const uint16_t*)s / 65535.0f : *(const float*)s;
                c[0] = c[1] = c[2] = z >= 1.0f || zmax <= zmin ? 1.0f : (z - zmin) / (zmax - zmin);
            } else if (f16) {
                for (u32 k = 0; k < texel / 2; ++k) c[k] = half(((const uint16_t*)s)[k]);
            } else if (f32) {
                for (u32 k = 0; k < texel / 4; ++k) c[k] = ((const float*)s)[k];
            } else if (t.fmt == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
                const u32 v = *(const u32*)s;
                auto small = [](u32 x, u32 mbits) {
                    const u32 e = x >> mbits, m = x & ((1u << mbits) - 1);
                    return e ? std::ldexp(1.0f + (float)m / (1u << mbits), (int)e - 15) : std::ldexp((float)m / (1u << mbits), -14);
                };
                c[0] = small(v & 0x7FF, 6), c[1] = small(v >> 11 & 0x7FF, 6), c[2] = small(v >> 22, 5);
            } else if (t.fmt == VK_FORMAT_A2B10G10R10_UNORM_PACK32) {
                const u32 v = *(const u32*)s;
                c[0] = (v & 0x3FF) / 1023.0f, c[1] = (v >> 10 & 0x3FF) / 1023.0f, c[2] = (v >> 20 & 0x3FF) / 1023.0f;
            } else if (texel == 8) {
                for (u32 k = 0; k < 4; ++k) c[k] = ((const uint16_t*)s)[k] / 65535.0f;
            } else {
                for (u32 k = 0; k < std::min(texel, 4u); ++k) c[k] = s[k] / 255.0f;
            }
            if (t.bgra) std::swap(c[0], c[2]);  // memory order B, G, R, A (COMP_SWAP)
            for (u32 k = 0; k < 3; ++k) o[k] = to8(c[k]);
            o[3] = 255;
        }
        r.FreeBuffer(buf);
        out.push_back(std::move(img));
    }
    return out;
}
} // namespace Gpu
