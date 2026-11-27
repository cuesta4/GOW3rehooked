// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <utility>
#include <vector>
#include "orbis/common/types.h"

namespace Gpu {
class Renderer;
}

// Guest memory the GPU reads, kept in the renderer's mirror. The frame's own regions (data ring slice, CPU arenas) are
// offered every frame. Anything else is uploaded on first use and write-protected: the first CPU write to a page faults
// once, reopens it and queues it for the next frame. Pages written in consecutive frames stay open and are offered every
// frame (unchanged 4 KB pieces are skipped by the renderer's hash).
namespace Gow3::GuestMemory {
using Range = std::pair<u64, u64>;  // [lo, hi)
void Install();                                       // fault handler and memory observer
void SetRegions(const std::vector<Range>& regions);   // per-frame regions: never protected
// Queued pages and `offer` -> renderer. Returns the pages CPU writes queued (write faults) since the last frame.
u32 BeginFrame(Gpu::Renderer& renderer, const std::vector<Range>& offer);
bool Track(Gpu::Renderer& renderer, u64 lo, u64 hi);  // Renderer::SetTracker
u64 Store(u64 addr, const void* data, u64 size);      // GPU readback into guest memory, returns piece hash
bool Readable(u64 addr, u64 size);                    // committed and readable (descriptor and program reads)
char PageKind(u64 addr);                              // U unknown, C clean, D dirty, Y dynamic, F fixed, R region (diagnostics)
bool AllUnknown(u64 lo, u64 hi);                      // every 4 KB page in [lo, hi) is Unknown
bool HasBacking(u64 lo, u64 hi);                      // every 16 KB backing page in [lo, hi) has a physical alias
u64 CopyBacking(u64 dst, const void* src, u64 size, bool hash = false);  // write via physical alias (caller verified HasBacking & Readable)
} // namespace Gow3::GuestMemory
