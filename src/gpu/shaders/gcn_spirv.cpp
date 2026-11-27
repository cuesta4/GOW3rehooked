// SPDX-License-Identifier: GPL-2.0-or-later
// GCN program -> SPIR-V, written directly (no GLSL, no glslang). The same model as the GLSL it replaced: registers are
// variables of main, EXEC and VCC are subgroup ballots, branches nest as structured control flow or run as a pc loop,
// memory follows the guest address (PhysicalStorageBuffer), images and samplers come from the recipe's slots.
#include "gpu/piece_hash.h"
#include <spirv/unified1/GLSL.std.450.h>
#include <spirv-tools/optimizer.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <map>
#include <xxhash.h>
#include "gpu/shaders/gcn_internal.h"
#include "gpu/shaders/gcn_spirv.h"

namespace Gcn {
namespace {
using namespace Spv;
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using s32 = std::int32_t;
constexpr float kFltMax = 3.402823466e38f;

struct Gen : Ops {
    const Program& p;
    const Analysis& a;
    const Input& in;
    std::span<const SlotSpec> spec;
    std::string error;
    const bool ps = in.stage == Stage::Pixel;
    const u32 res = a.const_dwords;  // the slots' place in the draw table
    const bool motion_blur = ps && ::Gpu::PieceHash::ShaderCode(in.code.data(), in.code.size_bytes()) == MotionBlurCodeHash();


    // Registers: variables of main, created on first use.
    std::map<u32, Id> sreg, vreg, spill;  // spill: SGPRs spilled to VGPR lanes (Inst::spill), one copy per invocation
    // Per-lane compare results still held by their SGPR pair (VCC 106 included): this lane's bit of the mask, without the
    // ballot round trip. Valid within one block and one EXEC, until the pair is written again.
    std::map<u32, Id> lane_masks;
    std::vector<u32> lane_set;  // pairs the instruction being translated set
    std::vector<Id> touched;  // variables stored by the instruction being translated
    Id vcc_lo = Var(pFnU, CU(0)), vcc_hi = Var(pFnU, CU(0)), exec_lo = Var(pFnU, CU(0)), exec_hi = Var(pFnU, CU(0)), m0 = Var(pFnU, CU(0));
    Id scc = Var(pFnB, CB(false)), live = Var(pFnB, CB(true));
    Id Sr(u32 n) {
        auto [it, fresh] = sreg.try_emplace(n, 0);
        if (fresh) it->second = Var(pFnU, CU(0));
        return it->second;
    }
    Id Vr(u32 n) {
        auto [it, fresh] = vreg.try_emplace(n, 0);
        if (fresh) it->second = Var(pFnU, CU(0));
        return it->second;
    }
    Id Spill(u32 key) {
        auto [it, fresh] = spill.try_emplace(key, 0);
        if (fresh) it->second = Var(pFnU, CU(0));
        return it->second;
    }
    void St(Id var, Id v) {
        Ops::St(var, v);
        touched.push_back(var);
    }
    void Fail(const std::string& e) {
        if (error.empty()) error = e;
    }
    bool wrote(Id v) const { return std::find(touched.begin(), touched.end(), v) != touched.end(); }

    // Globals, created on first use.
    Id gSubSize, gSubId, gSubLt, gVertex, gInstance, gFragCoord, gFront, gSampleMask, gPush, gT0, gT1;
    Id gFragDepth, gPosition, gSamplerArr, gWorkgroup, gLocal, gRect;
    Id mem_delta;
    std::vector<Id> table_dwords, table_quads;
    std::map<std::pair<u32, u32>, Id> image_ids;
    std::map<u32, Id> sampler_ids;
    std::map<u32, Id> image_sizes;
    const Id pInU = Ptr(spv::StorageClassInput, tU32);
    Id gTex[6][3]{}, gImg[3][3]{};
    bool storage_read[3][3]{};
    std::map<u32, Id> gOut, gParam, gAttr;
    std::map<u32, u32> attrs;  // PS attribute -> interpolation qualifier (1 sample, 2 centroid, 4 noperspective, 8 flat), decorated at the end
    bool samplers = false, depth_out = false;
    DepthResolvePattern depth_resolve;
    Id extra_depth;
    u32 mrt_mask = 0;
    std::set<u32> params;

    Id Builtin(spv::StorageClass sc, Id pointee, spv::BuiltIn bi, bool flat) {
        const Id v = b.Global(Ptr(sc, pointee), sc);
        b.Decorate(v, spv::DecorationBuiltIn, {u32(bi)});
        if (flat && ps && sc == spv::StorageClassInput) b.Decorate(v, spv::DecorationFlat);
        return v;
    }
    Id SubSize() {
        if (!gSubSize) gSubSize = Builtin(spv::StorageClassInput, tU32, spv::BuiltInSubgroupSize, true), b.Capability(spv::CapabilityGroupNonUniform);
        return LdU(gSubSize);
    }
    Id SubId() {
        if (!gSubId) gSubId = Builtin(spv::StorageClassInput, tU32, spv::BuiltInSubgroupLocalInvocationId, true), b.Capability(spv::CapabilityGroupNonUniform);
        return LdU(gSubId);
    }
    Id SubLt() {
        if (!gSubLt) gSubLt = Builtin(spv::StorageClassInput, tV4U, spv::BuiltInSubgroupLtMask, true), b.Capability(spv::CapabilityGroupNonUniformBallot);
        return Ld(gSubLt, tV4U);
    }
    Id Wide() { return Bool(spv::OpUGreaterThan, SubSize(), CU(32)); }  // 64 lanes: the hardware's wave

    // Subgroup operations.
    Id Ballot(Id pred) {
        b.Capability(spv::CapabilityGroupNonUniformBallot);
        return R(spv::OpGroupNonUniformBallot, tV4U, CU(3), pred);
    }
    Id Shuffle(Id t, Id v, Id lane) {
        b.Capability(spv::CapabilityGroupNonUniformShuffle);
        return R(spv::OpGroupNonUniformShuffle, t, CU(3), v, lane);
    }
    Id ShuffleU(Id v, Id lane) { return Shuffle(tU32, v, R(spv::OpUMod, tU32, lane, SubSize())); }
    // EXEC as this lane sees it: bit `lane` of the 64-bit mask.
    Id LB(Id lo, Id hi) {
        const Id id = SubId();
        const Id small = Ne(And(Shr(lo, id), 1u), 0);
        const Id big = R(spv::OpINotEqual, tBool, R(spv::OpBitwiseAnd, tU64, R(spv::OpShiftRightLogical, tU64, U64(lo, hi), id), CU64(1)), CU64(0));
        return Sel(tBool, Wide(), big, small);
    }
    // Active lanes: all, unless EXEC changes per lane here.
    bool full = true;
    bool AllLive() const { return full || !p.exec_dynamic; }
    Id Live() { return AllLive() ? CB(true) : Ld(live, tBool); }
    void SetLive() { St(live, full ? CB(true) : LB(LdU(exec_lo), LdU(exec_hi))); }
    // Lane mask into an SGPR pair: inactive lanes read 0, like V_CMP.
    // A lane reading its own bit of the mask reads its compare result, helper lanes included (they are in the WQM EXEC that
    // V_CMP writes under): the predicate serves those reads, the ballot only the scalar ones (unread, the driver drops it).
    void WM(u32 sdst, Id cond, bool lane_only = false) {
        auto [dl, dh] = D64(sdst);
        const Id pred = AllLive() ? cond : LAnd(Ld(live, tBool), cond);
        if (sdst != 126) lane_masks[sdst] = pred, lane_set.push_back(sdst);
        if (lane_only) return;  // no scalar reader before the mask is overwritten
        const Id v = Ballot(pred);
        const Id lo = Extract(tU32, v, 0), hi = SelU(Wide(), Extract(tU32, v, 1), CU(0));
        St(dl, lo), St(dh, hi);
    }
    void InvalidateMask(u32 r) {
        std::erase_if(lane_masks, [&](const auto& e) { return e.first == r || e.first + 1 == r; });
    }
    Id LaneMask(u32 s, u32 lit) {
        if (const auto it = lane_masks.find(s); it != lane_masks.end()) return it->second;
        const auto [lo, hi] = U64p(s, lit);
        return LB(lo, hi);
    }

    // Operands.
    Id Sg(u32 n) { return LdU(Sr(n)); }
    Id U(u32 c, u32 lit) {
        if (c < 106 || (c >= 108 && c <= 123)) return LdU(Sr(c));
        switch (c) {
        case 106: return LdU(vcc_lo);
        case 107: return LdU(vcc_hi);
        case 124: return LdU(m0);
        case 126: return LdU(exec_lo);
        case 127: return LdU(exec_hi);
        case 251: return SelU(LAnd(Eq(LdU(vcc_lo), 0), Eq(LdU(vcc_hi), 0)), CU(1), CU(0));
        case 252: return SelU(LAnd(Eq(LdU(exec_lo), 0), Eq(LdU(exec_hi), 0)), CU(1), CU(0));
        case 253: return SelU(Ld(scc, tBool), CU(1), CU(0));
        case 255: return CU(lit);
        }
        if (c >= 128 && c <= 192) return CU(c - 128);
        if (c >= 193 && c <= 208) return CU(u32(-s32(c - 192)));
        if (c >= 240 && c <= 247) return CU(kFloatBits[c - 240]);
        if (c >= 256) return LdU(Vr(c - 256));
        Fail(std::format("operand {}", c));
        return CU(0);
    }
    std::pair<Id, Id> U64p(u32 c, u32 lit) {
        if (c < 104 || (c >= 108 && c <= 122)) return {LdU(Sr(c)), LdU(Sr(c + 1))};
        if (c == 106) return {LdU(vcc_lo), LdU(vcc_hi)};
        if (c == 126) return {LdU(exec_lo), LdU(exec_hi)};
        if (c >= 128 && c <= 192) return {CU(c - 128), CU(0)};
        if (c >= 193 && c <= 208) return {CU(u32(-s32(c - 192))), CU(0xFFFFFFFFu)};
        if (c >= 240 && c <= 247) return {CU(0), CU(kDoubleHi[c - 240])};
        if (c == 255) return {CU(lit), CU(0)};
        if (c == 253) return {SelU(Ld(scc, tBool), CU(1), CU(0)), CU(0)};
        Fail(std::format("64-bit operand {}", c));
        return {CU(0), CU(0)};
    }
    Id P64(u32 c, u32 lit) {
        auto [lo, hi] = U64p(c, lit);
        return U64(lo, hi);
    }
    Id D(u32 c) {
        InvalidateMask(c);
        switch (c) {
        case 106: return vcc_lo;
        case 107: return vcc_hi;
        case 124: return m0;
        case 126: return exec_lo;
        case 127: return exec_hi;
        }
        if (c < 106 || (c >= 108 && c <= 123)) return Sr(c);
        Fail(std::format("destination {}", c));
        return m0;
    }
    std::pair<Id, Id> D64(u32 c) {
        InvalidateMask(c), InvalidateMask(c + 1);
        if (c == 106) return {vcc_lo, vcc_hi};
        if (c == 126) return {exec_lo, exec_hi};
        if (c < 104 || (c >= 108 && c <= 122)) return {Sr(c), Sr(c + 1)};
        Fail(std::format("64-bit destination {}", c));
        return {m0, m0};
    }
    void St64(u32 c, Id v) {
        auto [lo, hi] = D64(c);
        St(lo, Lo(v)), St(hi, Hi(v));
    }
    // VALU result for this lane: predicated by EXEC unless EXEC never changes per lane.
    void WV(u32 v, Id e) {
        const Id n = Vr(v);
        if (!AllLive()) St(n, SelU(Ld(live, tBool), e, LdU(n)));
        else St(n, e);
    }
    Id Fv(const Inst& i, int k) {
        const u32 c = i.src[k];
        Id e = c >= 240 && c <= 247 ? CFb(kFloatBits[c - 240]) : Fu(U(c, i.lit));
        if (i.abs >> k & 1) e = FAbs(e);
        if (i.neg >> k & 1) e = FNeg(e);
        return e;
    }
    Id Uv(const Inst& i, int k) { return U(i.src[k], i.lit); }
    Id Iv(const Inst& i, int k) { return Iu(Uv(i, k)); }
    Id FR(const Inst& i, Id e) {  // output modifiers, as the raw bits
        static constexpr float omod[]{1.f, 2.f, 4.f, 0.5f};
        if (i.omod) e = FMul(e, CF(omod[i.omod]));
        if (i.clamp) e = E(tF32, GLSLstd450FClamp, e, CF(0.f), CF(1.f));
        return Uf(e);
    }

    // Arrays of dwords (push constants, draw tables): one type, one stride decoration, per length.
    std::set<Id> strided;
    Id Arr(u32 n) {
        const Id t = b.Type(spv::OpTypeArray, {tU32, CU(n)});
        if (strided.insert(t).second) b.Decorate(t, spv::DecorationArrayStride, {4});
        return t;
    }
    Id VertexIndex() {
        if (!gVertex) gVertex = Builtin(spv::StorageClassInput, tU32, spv::BuiltInVertexIndex, false);
        return LdU(gVertex);
    }
    Id InstanceIndex() {
        if (!gInstance) gInstance = Builtin(spv::StorageClassInput, tU32, spv::BuiltInInstanceIndex, false);
        return LdU(gInstance);
    }

    // Guest memory.
    Id Ld32(Id addr) {
        const Id ptr = R(spv::OpConvertUToPtr, Ptr(spv::StorageClassPhysicalStorageBuffer, tU32), addr);
        return R(spv::OpLoad, tU32, ptr, u32(spv::MemoryAccessAlignedMask), 4u);
    }
    Id MemDelta() {
        if (!mem_delta)
            mem_delta = b.EntryRes(spv::OpLoad, tU64, b.EntryRes(spv::OpAccessChain, Ptr(spv::StorageClassPushConstant, tU64), Push(), CI(0)));
        return mem_delta;
    }
    Id Push() {
        if (!gPush) {
            const Id arr = Arr(16);
            const Id st = b.Struct({tU64, arr, arr});
            b.Decorate(st, spv::DecorationBlock);
            b.DecorateMember(st, 0, spv::DecorationOffset, {0}), b.DecorateMember(st, 1, spv::DecorationOffset, {8}), b.DecorateMember(st, 2, spv::DecorationOffset, {72});
            gPush = b.Global(Ptr(spv::StorageClassPushConstant, st), spv::StorageClassPushConstant);
        }
        return gPush;
    }
    Id PushU(u32 stage_ps, u32 k) {
        return R(spv::OpLoad, tU32, R(spv::OpAccessChain, Ptr(spv::StorageClassPushConstant, tU32), Push(), CI(1 + stage_ps), CU(k)));
    }
    Id LD(Id addr) {
        const Id v = Ld32(R(spv::OpIAdd, tU64, addr, MemDelta()));
        if (in.boom) BoomLoad(addr, v);
        return v;
    }
    void LDVec(Id addr, u32 count, Id* out) {
        const Id delta = R(spv::OpIAdd, tU64, addr, MemDelta());
        if (count == 1) {
            out[0] = Ld32(delta);
        } else {
            const Id vt = count == 2 ? tV2U : count == 3 ? Vec(tU32, 3) : tV4U;
            const Id ptr = R(spv::OpConvertUToPtr, Ptr(spv::StorageClassPhysicalStorageBuffer, vt), delta);
            const Id vec = R(spv::OpLoad, vt, ptr, u32(spv::MemoryAccessAlignedMask), 4u);
            for (u32 k = 0; k < count; ++k) out[k] = Extract(tU32, vec, k);
        }
        if (in.boom)
            for (u32 k = 0; k < count; ++k) BoomLoad(k ? R(spv::OpIAdd, tU64, addr, CU64(4 * k)) : addr, out[k]);
    }

    // Exploded vertices (in.boom, VS; GOW3_BOOM): this invocation keeps its first kBoomLoads guest loads (pc, address / 4, value);
    // when the position has a NaN or Inf component or lies BoomClip() half-screens off the screen's centre in front of the camera, the
    // first such invocation of the draw writes them to the record at table dword res + slots (Impl::kBoomRec: count, index, loads,
    // param + 1 (0: the position is the one exploded), the exploded position or param, then the three arrays). A param export with a
    // NaN, an Inf or a component above BoomParamLimit() marks the vertex too (the velocity pass's encoded velocity); each vertex counts once.
    static constexpr u32 kBoomLoads = 192, kBoomConsts = 96;  // Impl::kBoomLoads, kBoomConsts: the record ends with the constant bank's first dwords
    u32 boom_pc = 0;
    Id boom_n, boom_hit, boom_log[3];
    void BoomInit() {
        if (boom_n) return;
        boom_n = Var(pFnU, CU(0));
        boom_hit = Var(pFnU, CU(0));
        const Id arr = Ptr(spv::StorageClassFunction, b.Type(spv::OpTypeArray, {tU32, CU(kBoomLoads)}));
        for (Id& l : boom_log) l = b.Local(arr, Id{});
    }
    void BoomLoad(Id addr, Id v) {
        BoomInit();
        const Id n = LdU(boom_n);
        const IfCtx room = IfBegin(Bool(spv::OpULessThan, n, CU(kBoomLoads)));
        const Id val[3]{CU(boom_pc), Lo(R(spv::OpShiftRightLogical, tU64, addr, CU(2))), v};
        for (u32 j = 0; j < 3; ++j) Ops::St(R(spv::OpAccessChain, pFnU, boom_log[j], n), val[j]);
        IfEnd(room);
        Ops::St(boom_n, Add(n, CU(1)));
    }
    void BoomExport(const Id* c) {
        BoomInit();
        Id bad = CB(false);
        for (u32 k = 0; k < 4; ++k) bad = LOr(bad, Eq(And(Shr(Uf(c[k]), 23), 0xFFu), 0xFF));
        const Id reach = FMul(c[3], CF(float(BoomClip())));
        bad = LOr(bad, LAnd(Bool(spv::OpFOrdGreaterThan, c[3], CF(0)),
                            LOr(Bool(spv::OpFOrdGreaterThan, FAbs(c[0]), reach), Bool(spv::OpFOrdGreaterThan, FAbs(c[1]), reach))));
        BoomRecord(c, bad, 0);
    }
    void BoomParam(const Id* c, u32 param) {
        BoomInit();
        Id bad = CB(false);
        for (u32 k = 0; k < 4; ++k)
            bad = LOr(bad, LOr(Eq(And(Shr(Uf(c[k]), 23), 0xFFu), 0xFF), Bool(spv::OpFOrdGreaterThan, FAbs(c[k]), CF(float(BoomParamLimit())))));
        BoomRecord(c, bad, param + 1);
    }
    void BoomRecord(const Id* c, Id bad, u32 param) {
        bad = LAnd(bad, Eq(LdU(boom_hit), 0));
        const IfCtx hit = IfBegin(bad);
        Ops::St(boom_hit, CU(1));
        const Id rec = U64(TableAt(res + u32(a.slots.size())), TableAt(res + u32(a.slots.size()) + 1));
        auto at = [&](Id off) { return R(spv::OpConvertUToPtr, Ptr(spv::StorageClassPhysicalStorageBuffer, tU32), R(spv::OpIAdd, tU64, rec, off)); };
        auto put = [&](Id off, Id val) { b.Do(spv::OpStore, at(off), val, u32(spv::MemoryAccessAlignedMask), 4u); };
        const IfCtx first = IfBegin(Eq(R(spv::OpAtomicIAdd, tU32, at(CU64(0)), CU(1), CU(0), CU(1)), 0));
        put(CU64(4), VertexIndex()), put(CU64(8), LdU(boom_n)), put(CU64(12), CU(param));
        for (u32 k = 0; k < 4; ++k) put(CU64(16 + 4 * k), Uf(c[k]));
        const Id kv = Var(pFnU, CU(0)), header = b.NewId(), body = b.NewId(), cont = b.NewId(), merge = b.NewId();
        Ops::St(kv, CU(0));
        b.Br(header);
        b.Label(header);
        const Id k = LdU(kv);
        b.Do(spv::OpLoopMerge, merge, cont, 0u);
        b.BrCond(Bool(spv::OpULessThan, k, E(tU32, GLSLstd450UMin, LdU(boom_n), CU(kBoomLoads))), body, merge);
        b.Label(body);
        const Id off = W64(Shl(k, 2));
        for (u32 j = 0; j < 3; ++j) put(R(spv::OpIAdd, tU64, CU64(32 + 4ull * kBoomLoads * j), off), Ld(R(spv::OpAccessChain, pFnU, boom_log[j], k), tU32));
        b.Br(cont);
        b.Label(cont);
        Ops::St(kv, Add(k, CU(1)));
        b.Br(header);
        b.Label(merge);
        for (u32 i = 0; i < std::min(res, kBoomConsts); ++i) put(CU64(32 + 12ull * kBoomLoads + 4 * i), TableAt(i));  // the draw's constant bank
        IfEnd(first);
        IfEnd(hit);
    }
    Id LD8(Id a64) {
        const Id w = LD(R(spv::OpBitwiseAnd, tU64, a64, CU64(~3ull)));
        return And(Shr(w, Shl(And(Lo(a64), 3u), 3u)), 0xFFu);
    }
    Id LD16(Id a64) {
        const Id w = LD(R(spv::OpBitwiseAnd, tU64, a64, CU64(~3ull)));
        return And(Shr(w, Shl(And(Lo(a64), 2u), 3u)), 0xFFFFu);
    }
    Id Table(bool pixel) {
        Id& g = pixel ? gT1 : gT0;
        if (!g) {
            const Id arr = b.Type(spv::OpTypeArray, {tV4U, CU(1024)});
            if (strided.insert(arr).second) b.Decorate(arr, spv::DecorationArrayStride, {16});
            const Id st = b.Struct({arr});
            b.Decorate(st, spv::DecorationBlock);
            b.DecorateMember(st, 0, spv::DecorationOffset, {0});
            g = b.Global(Ptr(spv::StorageClassUniform, st), spv::StorageClassUniform);
            b.Decorate(g, spv::DecorationDescriptorSet, {1}), b.Decorate(g, spv::DecorationBinding, {pixel ? 1u : 0u});
        }
        return g;
    }
    Id TableAt(u32 dword) {
        if (dword >= table_dwords.size()) table_dwords.resize(dword + 1);
        if (table_dwords[dword]) return table_dwords[dword];
        const u32 q = dword / 4;
        if (q >= table_quads.size()) table_quads.resize(q + 1);
        if (!table_quads[q])
            table_quads[q] = b.EntryRes(spv::OpLoad, tV4U, b.EntryRes(spv::OpAccessChain, Ptr(spv::StorageClassUniform, tV4U), Table(ps), CI(0), CU(q)));
        return table_dwords[dword] = b.EntryRes(spv::OpCompositeExtract, tU32, table_quads[q], dword % 4);
    }

    // --- Scalar ALU ---
    void Salu(const Inst& i);
    void Smrd(const Inst& i);
    // --- Vector ALU ---
    void Valu(const Inst& i);
    Id Bfe(Id v, Id o, Id w) { return And(Shr(v, And(o, 31u)), Sub(Shl(CU(1), And(w, 31u)), CU(1))); }
    Id BfeI(Id v, Id o, Id w) {
        const Id wm = And(w, 31u), sh = Sub(CU(32), wm);
        const Id x = Ui(R(spv::OpShiftRightArithmetic, tI32, Iu(Shl(Shr(v, And(o, 31u)), sh)), sh));
        return SelU(Eq(wm, 0), CU(0), x);
    }
    Id FMin(Id x, Id y) { return E(tF32, GLSLstd450FMin, x, y); }
    Id FMax(Id x, Id y) { return E(tF32, GLSLstd450FMax, x, y); }
    Id Med3(Id x, Id y, Id z) { return FMax(FMin(x, y), FMin(FMax(x, y), z)); }
    Id LMul(Id x, Id y) { return SelF(LOr(Bool(spv::OpFOrdEqual, x, CF(0.f)), Bool(spv::OpFOrdEqual, y, CF(0.f))), CF(0.f), FMul(x, y)); }
    Id FClass(Id x, Id m);
    Id Wqm(Id x);  // 64-bit
    Id Wqm32(Id x);
    Id CvtU(Id f) {
        const Id r = R(spv::OpConvertFToU, tU32, f);
        return SelU(FGt(f, CF(0.f)), SelU(FGe(f, CF(4294967040.f)), CU(0xFFFFFFFFu), r), CU(0));
    }
    Id CvtI(Id f) {
        const Id c = E(tF32, GLSLstd450FClamp, f, CF(-2147483648.f), CF(2147483520.f));
        return SelU(R(spv::OpIsNan, tBool, f), CU(0), Ui(R(spv::OpConvertFToS, tI32, c)));
    }
    Id CubeZ(Id x, Id y, Id z) { return LAnd(FGe(FAbs(z), FAbs(x)), FGe(FAbs(z), FAbs(y))); }
    Id CubeY(Id x, Id y) { return FGe(FAbs(y), FAbs(x)); }
    Id Cube(int which, Id x, Id y, Id z);
    Id CubeDir(Id s, Id t, Id f);
    Id SmallFloat(Id v, u32 m);
    Id CvtComp(Id v, u32 bits, u32 nfmt, bool packed_float);
    Id Fetch(Id addr, u32 dfmt, u32 nfmt);
    Id TBuf(Id d0, Id d1, Id d2, u32 d3, Id index, Id offset, u32 dfmt, u32 nfmt);
    Id RBuf(Id d0, Id d1, Id d2, Id index, Id offset);
    Id EncComp(Id v, u32 bits, u32 nfmt);
    void St32(Id addr, Id v);
    void StMasked(Id addr, Id v, u32 mask);
    void StFields(Id addr, std::initializer_list<std::pair<u32, u32>> fields, u32 k0, u32 nfmt, const Id* c, u32 n);
    void Put(Id addr, u32 dfmt, u32 nfmt, const Id* c, u32 n);
    void TStore(Id d0, Id d1, Id d2, Id index, Id offset, u32 dfmt, u32 nfmt, const Id* c, u32 n);
    Id Swizzle(Id v, u32 off);
    // --- Images ---
    void Mimg(const Inst& i);
    Id ImageType(int kind, int cls, bool storage) {
        static constexpr spv::Dim dims[]{spv::Dim2D, spv::Dim2D, spv::Dim3D, spv::DimCube, spv::Dim2D, spv::Dim2D};
        static constexpr u32 arrayed[]{0, 1, 0, 0, 0, 1}, ms[]{0, 0, 0, 0, 1, 1};
        return b.Type(spv::OpTypeImage, {Sc(cls), u32(dims[kind]), 0, arrayed[kind], ms[kind], storage ? 2u : 1u, u32(spv::ImageFormatUnknown)});
    }
    Id ImageVar(int kind, int cls, bool storage) {
        Id& g = storage ? gImg[kind][cls] : gTex[kind][cls];
        if (!g) {
            const Id arr = b.Type(spv::OpTypeRuntimeArray, {ImageType(kind, cls, storage)});
            g = b.Global(Ptr(spv::StorageClassUniformConstant, arr), spv::StorageClassUniformConstant);
            b.Decorate(g, spv::DecorationDescriptorSet, {0}), b.Decorate(g, spv::DecorationBinding, {u32(storage ? kStorageBinding + kind * 3 + cls : kind * 3 + cls)});
            b.Capability(spv::CapabilityRuntimeDescriptorArray);
            if (storage) b.Capability(spv::CapabilityStorageImageWriteWithoutFormat);
        }
        return g;
    }
    Id LoadImage(int kind, int cls, bool storage, Id index) {
        const u32 binding = storage ? kStorageBinding + kind * 3 + cls : kind * 3 + cls;
        auto [it, fresh] = image_ids.try_emplace({binding, index.v}, Id{});
        if (!fresh) return it->second;
        const Id t = ImageType(kind, cls, storage);
        return it->second = b.EntryRes(spv::OpLoad, t, b.EntryRes(spv::OpAccessChain, Ptr(spv::StorageClassUniformConstant, t), ImageVar(kind, cls, storage), index));
    }
    Id LoadSampler(Id index) {
        auto [it, fresh] = sampler_ids.try_emplace(index.v, Id{});
        if (!fresh) return it->second;
        const Id t = b.Type(spv::OpTypeSampler);
        if (!gSamplerArr) {
            gSamplerArr = b.Global(Ptr(spv::StorageClassUniformConstant, b.Type(spv::OpTypeRuntimeArray, {t})), spv::StorageClassUniformConstant);
            b.Decorate(gSamplerArr, spv::DecorationDescriptorSet, {0}), b.Decorate(gSamplerArr, spv::DecorationBinding, {kSamplerBinding});
            b.Capability(spv::CapabilityRuntimeDescriptorArray);
        }
        return it->second = b.EntryRes(spv::OpLoad, t, b.EntryRes(spv::OpAccessChain, Ptr(spv::StorageClassUniformConstant, t), gSamplerArr, index));
    }
    Id ImageSize(Id img, int kind, bool storage) {
        auto [it, fresh] = image_sizes.try_emplace(img.v, Id{});
        if (fresh) {
            b.Capability(spv::CapabilityImageQuery);
            const Id t = kind == 0 || kind == 3 || kind == 4 ? tV2I : tV3I;
            it->second = storage || kind >= 4 ? b.EntryRes(spv::OpImageQuerySize, t, img) : b.EntryRes(spv::OpImageQuerySizeLod, t, img, CI(0));
        }
        return it->second;
    }
    // --- Exports, interpolation, buffers ---
    void Exp(const Inst& i);
    void Vintrp(const Inst& i);
    void Buffer(const Inst& i);
    Id OutVar(std::map<u32, Id>& m, u32 loc, Id t) {
        auto [it, fresh] = m.try_emplace(loc, 0);
        if (fresh) {
            it->second = b.Global(Ptr(spv::StorageClassOutput, t), spv::StorageClassOutput);
            b.Decorate(it->second, spv::DecorationLocation, {loc});
        }
        return it->second;
    }
    Id Position() {
        if (!gPosition) {
            const Id st = b.Struct({tV4F});
            b.Decorate(st, spv::DecorationBlock);
            b.DecorateMember(st, 0, spv::DecorationBuiltIn, {u32(spv::BuiltInPosition)});
            gPosition = b.Global(Ptr(spv::StorageClassOutput, st), spv::StorageClassOutput);
        }
        return gPosition;
    }
    Id PositionPtr() { return R(spv::OpAccessChain, Ptr(spv::StorageClassOutput, tV4F), Position(), CI(0)); }
    void RectOut();
    void BlendAlpha();

    // --- Program structure ---
    std::map<u32, size_t> index;  // pc -> instruction
    size_t exit = ~size_t(0);     // instruction after the innermost loop being structured
    Id exit_merge;                // the merge block of that loop (the break target)
    // EXEC known all-ones here (structured code in program order): at the start, and again once restored from the copy an
    // s_and_saveexec_b64 made of it (`saved`), the end of a divergent if. Loops and ifs that write EXEC end the knowledge.
    u32 saved = ~0u, entry = ~0u;  // entry: an s_mov_b64 copy of the all-ones EXEC (pixel shader prologue)
    void Forget(size_t from, size_t to) {
        lane_masks.clear();
        if (std::any_of(p.insts.begin() + from, p.insts.begin() + to, WritesExec)) full = false, saved = ~0u;
    }
    Id Cond(u32 op) {  // branch condition of a SOPP branch
        switch (op) {
        case 2: return CB(true);
        case 4: return LNot(Ld(scc, tBool));
        case 5: return Ld(scc, tBool);
        case 6: return LAnd(Eq(LdU(vcc_lo), 0), Eq(LdU(vcc_hi), 0));
        case 7: return LNot(LAnd(Eq(LdU(vcc_lo), 0), Eq(LdU(vcc_hi), 0)));
        case 8: return LAnd(Eq(LdU(exec_lo), 0), Eq(LdU(exec_hi), 0));
        default: return LNot(LAnd(Eq(LdU(exec_lo), 0), Eq(LdU(exec_hi), 0)));
        }
    }
    bool Block(size_t from, size_t to, bool emit);  // emit false: only whether the branches nest
    void OneInst(const Inst& i, bool last, bool loop);
    void Op(const Inst& i, bool last, bool loop);
    Id pc_var, switch_merge;

    Gen(const Program& p_, const Analysis& a_, const Input& in_, std::span<const SlotSpec> spec_) : p(p_), a(a_), in(in_), spec(spec_) {
        b.Capability(spv::CapabilityShader), b.Capability(spv::CapabilityInt64), b.Capability(spv::CapabilityPhysicalStorageBufferAddresses);
        if (in.stage == Stage::Pixel) {
            const auto match = FindDepthResolve(p);
            if (match.slot >= 0 && size_t(match.slot) < spec.size() && spec[match.slot].dim == Dim::D2Ms &&
                spec[match.slot].cls == Class::Float && spec[match.slot].msaa4)
                depth_resolve = match;
        }
    }
    bool Run(std::vector<u32>& words, u64& out_mask);
};

// ---------------------------------------------------------------------------------------------------------------------
// Helpers of the instruction set.

Id Gen::FClass(Id x, Id m) {
    const Id bits = Uf(x), e = And(Shr(bits, 23), 0xFFu), f = And(bits, 0x7FFFFFu), n = Ne(Shr(bits, 31), 0);
    const Id fz = Eq(f, 0), nz = LNot(fz);
    const Id e255 = Eq(e, 255), e0 = Eq(e, 0);
    // NaN: quiet (2) or signaling (1); infinity: -inf (4) or +inf (512)
    const Id nan = SelU(Ne(And(f, 0x400000u), 0), CU(2), CU(1));
    const Id inf = SelU(n, CU(4), CU(512));
    const Id den = SelU(nz, SelU(n, CU(16), CU(128)), SelU(n, CU(32), CU(64)));
    const Id norm = SelU(n, CU(8), CU(256));
    const Id c = SelU(e255, SelU(nz, nan, inf), SelU(e0, den, norm));
    return Ne(And(m, c), 0);
}

Id Gen::Wqm32(Id x) { return Mul(And(Or(Or(x, Shr(x, 1)), Or(Shr(x, 2), Shr(x, 3))), 0x11111111u), CU(15)); }

Id Gen::Wqm(Id x) {
    const Id r32 = W64(Wqm32(Lo(x)));
    auto sh = [&](u32 n) { return R(spv::OpShiftRightLogical, tU64, x, CU(n)); };
    const Id q64 = R(spv::OpBitwiseAnd, tU64, R(spv::OpBitwiseOr, tU64, R(spv::OpBitwiseOr, tU64, x, sh(1)), R(spv::OpBitwiseOr, tU64, sh(2), sh(3))), CU64(0x1111111111111111ull));
    return Sel(tU64, Wide(), R(spv::OpIMul, tU64, q64, CU64(15)), r32);
}

Id Gen::Cube(int which, Id x, Id y, Id z) {
    const Id cz = CubeZ(x, y, z), cy = CubeY(x, y);
    const Id zneg = FLt(z, CF(0.f)), yneg = FLt(y, CF(0.f)), xneg = FLt(x, CF(0.f));
    switch (which) {
    case 0: return SelF(cz, SelF(zneg, CF(5.f), CF(4.f)), SelF(cy, SelF(yneg, CF(3.f), CF(2.f)), SelF(xneg, CF(1.f), CF(0.f))));
    case 1: return SelF(cz, SelF(zneg, FNeg(x), x), SelF(cy, x, SelF(xneg, z, FNeg(z))));
    case 2: return SelF(cz, FNeg(y), SelF(cy, SelF(yneg, FNeg(z), z), FNeg(y)));
    default: return FMul(CF(2.f), SelF(cz, z, SelF(cy, y, x)));
    }
}

// (s, t, face) as produced by v_cube* and the 1.5 bias -> direction.
Id Gen::CubeDir(Id s, Id t, Id f) {
    const Id u = FMul(FSub(s, CF(1.5f)), CF(2.f)), w = FMul(FSub(t, CF(1.5f)), CF(2.f));
    const Id face = R(spv::OpConvertFToS, tI32, E(tF32, GLSLstd450Round, f));
    const Id one = CF(1.f), mone = CF(-1.f), nu = FNeg(u), nw = FNeg(w);
    auto pick = [&](Id c0, Id c1, Id c2, Id c3, Id c4, Id c5) {
        Id r = c5;
        const Id c[]{c0, c1, c2, c3, c4};
        for (int k = 4; k >= 0; --k) r = SelF(Bool(spv::OpIEqual, face, CI(k)), c[k], r);
        return r;
    };
    const Id x = pick(one, mone, u, u, u, nu), y = pick(nw, nw, one, mone, nw, nw), z = pick(nu, u, w, nw, one, mone);
    return R(spv::OpCompositeConstruct, tV3F, x, y, z);
}

// Unsigned 5-bit exponent, m-bit mantissa (10/11-bit packed floats).
Id Gen::SmallFloat(Id v, u32 m) {
    const Id e = Shr(v, m), f = And(v, (1u << m) - 1u);
    const Id denorm = FMul(R(spv::OpConvertUToF, tF32, f), CF(std::exp2(-14.f - float(m))));
    const Id special = SelF(Ne(f, 0), Fu(CU(0x7FC00000u)), Fu(CU(0x7F800000u)));
    const Id norm = FMul(E(tF32, GLSLstd450Exp2, FSub(R(spv::OpConvertUToF, tF32, e), CF(15.f))),
                         FAdd(CF(1.f), FDiv(R(spv::OpConvertUToF, tF32, f), CF(float(1u << m)))));
    return SelF(Eq(e, 0), denorm, SelF(Eq(e, 31), special, norm));
}

// One raw component of `bits` bits in number format `nfmt` (the format is constant: the V# or the instruction's).
Id Gen::CvtComp(Id v, u32 bits, u32 nfmt, bool packed_float) {
    const u32 sh = 32 - bits;
    const Id sv = bits == 32 ? Iu(v) : R(spv::OpShiftRightArithmetic, tI32, Iu(Shl(v, sh)), CU(sh));
    const float mx = bits == 32 ? 4294967295.f : float((1u << bits) - 1u), smx = bits == 32 ? 2147483647.f : float((1u << (bits - 1)) - 1u);
    const Id fv = R(spv::OpConvertUToF, tF32, v), fs = R(spv::OpConvertSToF, tF32, sv);
    switch (nfmt) {
    case 0: return Uf(FDiv(fv, CF(mx)));
    case 1: return Uf(FMax(FDiv(fs, CF(smx)), CF(-1.f)));
    case 2: return Uf(fv);
    case 3: return Uf(fs);
    case 4: return v;
    case 5: return Ui(sv);
    case 6: return Uf(FDiv(FAdd(FMul(CF(2.f), fs), CF(1.f)), CF(mx)));
    default:
        if (packed_float) return Uf(SmallFloat(v, bits - 5));
        if (bits == 16) return Uf(Extract(tF32, E(tV2F, GLSLstd450UnpackHalf2x16, v), 0));
        return v;
    }
}

// Typed buffer element at `addr`: components by constants only (a dynamic index into a vector spills it to local memory).
Id Gen::Fetch(Id addr, u32 dfmt, u32 nfmt) {
    Id r[4]{CU(0), CU(0), CU(0), CU(0)};
    u32 bits[4]{}, n = 0;
    bool packed_float = false;
    switch (dfmt) {
    case 1: r[0] = LD8(addr), bits[0] = 8, n = 1; break;
    case 2: r[0] = LD16(addr), bits[0] = 16, n = 1; break;
    case 3: { const Id w = LD16(addr); r[0] = And(w, 0xFFu), r[1] = Shr(w, 8); bits[0] = bits[1] = 8, n = 2; break; }
    case 4: r[0] = LD(addr), bits[0] = 32, n = 1; break;
    case 5: { const Id w = LD(addr); r[0] = And(w, 0xFFFFu), r[1] = Shr(w, 16); bits[0] = bits[1] = 16, n = 2; break; }
    case 6: { const Id w = LD(addr); r[0] = And(w, 0x7FFu), r[1] = And(Shr(w, 11), 0x7FFu), r[2] = Shr(w, 22); bits[0] = 11, bits[1] = 11, bits[2] = 10, n = 3, packed_float = nfmt == 7; break; }
    case 7: { const Id w = LD(addr); r[0] = And(w, 0x3FFu), r[1] = And(Shr(w, 10), 0x7FFu), r[2] = Shr(w, 21); bits[0] = 10, bits[1] = 11, bits[2] = 11, n = 3, packed_float = nfmt == 7; break; }
    case 8: { const Id w = LD(addr); r[0] = And(w, 3u), r[1] = And(Shr(w, 2), 0x3FFu), r[2] = And(Shr(w, 12), 0x3FFu), r[3] = Shr(w, 22); bits[0] = 2, bits[1] = bits[2] = bits[3] = 10, n = 4; break; }
    case 9: { const Id w = LD(addr); r[0] = And(w, 0x3FFu), r[1] = And(Shr(w, 10), 0x3FFu), r[2] = And(Shr(w, 20), 0x3FFu), r[3] = Shr(w, 30); bits[0] = bits[1] = bits[2] = 10, bits[3] = 2, n = 4; break; }
    case 10: { const Id w = LD(addr); r[0] = And(w, 0xFFu), r[1] = And(Shr(w, 8), 0xFFu), r[2] = And(Shr(w, 16), 0xFFu), r[3] = Shr(w, 24); bits[0] = bits[1] = bits[2] = bits[3] = 8, n = 4; break; }
    case 11: LDVec(addr, 2, r); bits[0] = bits[1] = 32, n = 2; break;
    case 12: { Id w[2]; LDVec(addr, 2, w); r[0] = And(w[0], 0xFFFFu), r[1] = Shr(w[0], 16), r[2] = And(w[1], 0xFFFFu), r[3] = Shr(w[1], 16); bits[0] = bits[1] = bits[2] = bits[3] = 16, n = 4; break; }
    case 13: LDVec(addr, 3, r); bits[0] = bits[1] = bits[2] = 32, n = 3; break;
    case 14: LDVec(addr, 4, r); bits[0] = bits[1] = bits[2] = bits[3] = 32, n = 4; break;
    }
    const bool is_int = nfmt == 4 || nfmt == 5;
    Id c[4];
    for (u32 k = 0; k < 4; ++k) c[k] = k < n ? CvtComp(r[k], bits[k], nfmt, packed_float) : k == 3 && n > 0 ? CU(is_int ? 1u : 0x3F800000u) : CU(0);
    if (n == 0) c[3] = CU(is_int ? 1u : 0x3F800000u);  // (the original returned 0 only for unknown formats: x, y, z zero, w one)
    return R(spv::OpCompositeConstruct, tV4U, c[0], c[1], c[2], c[3]);
}

Id Gen::TBuf(Id d0, Id d1, Id d2, u32 d3, Id index, Id offset, u32 dfmt, u32 nfmt) {
    const Id stride = And(Shr(d1, 16), 0x3FFFu);
    const Id oor = Sel(tBool, Ne(stride, 0), Bool(spv::OpUGreaterThanEqual, index, d2), Bool(spv::OpUGreaterThanEqual, offset, d2));
    const Id zero4 = R(spv::OpCompositeConstruct, tV4U, CU(0), CU(0), CU(0), CU(0));
    const Id v = Guard(tV4U, LNot(oor), zero4, [&] {
        const Id addr = R(spv::OpIAdd, tU64, R(spv::OpIAdd, tU64, Ptr64(d0, d1), R(spv::OpIMul, tU64, W64(index), W64(stride))), W64(offset));
        return Fetch(addr, dfmt, nfmt);
    });
    const bool is_int = nfmt == 4 || nfmt == 5;
    Id c[4];
    for (u32 k = 0; k < 4; ++k) {
        const u32 s = d3 >> (3 * k) & 7;
        c[k] = s >= 4 ? Extract(tU32, v, s - 4) : s == 1 ? CU(is_int ? 1u : 0x3F800000u) : CU(0);
    }
    return R(spv::OpCompositeConstruct, tV4U, c[0], c[1], c[2], c[3]);
}

Id Gen::RBuf(Id d0, Id d1, Id d2, Id index, Id offset) {
    const Id stride = And(Shr(d1, 16), 0x3FFFu);
    const Id oor = Sel(tBool, Ne(stride, 0), Bool(spv::OpUGreaterThanEqual, index, d2), Bool(spv::OpUGreaterThanEqual, offset, d2));
    return Guard(tU32, LNot(oor), CU(0), [&] {
        return LD(R(spv::OpIAdd, tU64, R(spv::OpIAdd, tU64, Ptr64(d0, d1), R(spv::OpIMul, tU64, W64(index), W64(stride))), W64(offset)));
    });
}

// One component for storing: the raw `bits` bits of the register in number format `nfmt` (the inverse of CvtComp).
Id Gen::EncComp(Id v, u32 bits, u32 nfmt) {
    const u32 mask = bits == 32 ? 0xFFFFFFFFu : (1u << bits) - 1u;
    const float mx = bits == 32 ? 4294967295.f : float(mask), smx = bits == 32 ? 2147483647.f : float((1u << (bits - 1)) - 1u);
    const Id f = Fu(v);
    switch (nfmt) {
    case 0: return And(CvtU(E(tF32, GLSLstd450RoundEven, FMul(FMax(FMin(f, CF(1.f)), CF(0.f)), CF(mx)))), mask);
    case 1: case 6: return And(CvtI(E(tF32, GLSLstd450RoundEven, FMul(FMax(FMin(f, CF(1.f)), CF(-1.f)), CF(smx)))), mask);
    case 2: return And(CvtU(FMin(f, CF(mx))), mask);
    case 3: return And(CvtI(E(tF32, GLSLstd450FClamp, f, CF(-smx - 1.f), CF(smx))), mask);
    case 4: return bits == 32 ? v : E(tU32, GLSLstd450UMin, v, CU(mask));
    case 5: return bits == 32 ? v : And(Ui(E(tI32, GLSLstd450SClamp, Iu(v), CI(-(1 << (bits - 1))), CI((1 << (bits - 1)) - 1))), mask);
    default:
        if (bits == 16) return And(E(tU32, GLSLstd450PackHalf2x16, R(spv::OpCompositeConstruct, tV2F, f, CF(0.f))), 0xFFFFu);
        if (bits != 32) Fail("store of a packed float format");
        return v;
    }
}

void Gen::St32(Id addr, Id v) {
    const Id ptr = R(spv::OpConvertUToPtr, Ptr(spv::StorageClassPhysicalStorageBuffer, tU32), R(spv::OpIAdd, tU64, addr, MemDelta()));
    b.Do(spv::OpStore, ptr, v, u32(spv::MemoryAccessAlignedMask), 4u);
}

// The bits of `mask` (already in place in v) into the dword holding the byte address; the other bits stay (atomic and/or).
void Gen::StMasked(Id addr, Id v, u32 mask) {
    const Id ptr = R(spv::OpConvertUToPtr, Ptr(spv::StorageClassPhysicalStorageBuffer, tU32),
                     R(spv::OpIAdd, tU64, R(spv::OpBitwiseAnd, tU64, addr, CU64(~3ull)), MemDelta()));
    const Id sh = Shl(And(Lo(addr), 3u), 3u), m = Shl(CU(mask), sh);
    R(spv::OpAtomicAnd, tU32, ptr, CU(1), CU(0), R(spv::OpNot, tU32, m));
    R(spv::OpAtomicOr, tU32, ptr, CU(1), CU(0), And(Shl(v, sh), m));
}

// Components c[k0 + j] (those below n) into the fields of the dword at `addr`: (shift, bits) for j = 0.. A whole dword is written
// plainly, a part of one leaves its other bits as they are.
void Gen::StFields(Id addr, std::initializer_list<std::pair<u32, u32>> fields, u32 k0, u32 nfmt, const Id* c, u32 n) {
    Id v = CU(0);
    u32 mask = 0, j = 0;
    for (auto [shift, bits] : fields) {
        if (k0 + j < n) v = Or(v, Shl(EncComp(c[k0 + j], bits, nfmt), shift)), mask |= (bits == 32 ? 0xFFFFFFFFu : (1u << bits) - 1u) << shift;
        ++j;
    }
    if (mask == 0xFFFFFFFFu) St32(addr, v);
    else if (mask) StMasked(addr, v, mask);
}

// Components c[0..n) into the element at `addr` (guest address) of format dfmt/nfmt: the components the register data holds.
void Gen::Put(Id addr, u32 dfmt, u32 nfmt, const Id* c, u32 n) {
    auto a4 = [&](u32 k) { return R(spv::OpIAdd, tU64, addr, CU64(k)); };
    switch (dfmt) {
    case 1: return StFields(addr, {{0, 8}}, 0, nfmt, c, n);
    case 2: return StFields(addr, {{0, 16}}, 0, nfmt, c, n);
    case 3: return StFields(addr, {{0, 8}, {8, 8}}, 0, nfmt, c, n);
    case 4: case 11: case 13: case 14:
        for (u32 k = 0, count = dfmt == 4 ? 1 : dfmt == 11 ? 2 : dfmt == 13 ? 3 : 4; k < std::min(n, count); ++k) St32(a4(4 * k), EncComp(c[k], 32, nfmt));
        return;
    case 5: return StFields(addr, {{0, 16}, {16, 16}}, 0, nfmt, c, n);
    case 12: return StFields(addr, {{0, 16}, {16, 16}}, 0, nfmt, c, n), StFields(a4(4), {{0, 16}, {16, 16}}, 2, nfmt, c, n);
    case 8: return StFields(addr, {{0, 2}, {2, 10}, {12, 10}, {22, 10}}, 0, nfmt, c, n);
    case 9: return StFields(addr, {{0, 10}, {10, 10}, {20, 10}, {30, 2}}, 0, nfmt, c, n);
    case 10: return StFields(addr, {{0, 8}, {8, 8}, {16, 8}, {24, 8}}, 0, nfmt, c, n);
    default: Fail(std::format("store of buffer format {}", dfmt));
    }
}

// Active lanes only (and in range, like the loads).
void Gen::TStore(Id d0, Id d1, Id d2, Id index, Id offset, u32 dfmt, u32 nfmt, const Id* c, u32 n) {
    const Id stride = And(Shr(d1, 16), 0x3FFFu);
    const Id oor = Sel(tBool, Ne(stride, 0), Bool(spv::OpUGreaterThanEqual, index, d2), Bool(spv::OpUGreaterThanEqual, offset, d2));
    const IfCtx ctx = IfBegin(LAnd(Live(), LNot(oor)));
    Put(R(spv::OpIAdd, tU64, R(spv::OpIAdd, tU64, Ptr64(d0, d1), R(spv::OpIMul, tU64, W64(index), W64(stride))), W64(offset)), dfmt, nfmt, c, n);
    IfEnd(ctx);
}

Id Gen::Swizzle(Id v, u32 off) {
    // A pixel quad broadcast must include helper lanes, which a subgroup shuffle can skip.
    const u32 lane = off & 3u;
    if (in.stage == Stage::Pixel && (off & 0x8000) && (off & 0xFF) == lane * 0x55u) {
        b.Capability(spv::CapabilityGroupNonUniformQuad);
        return R(spv::OpGroupNonUniformQuadBroadcast, tU32, CU(spv::ScopeSubgroup), v, CU(lane));
    }
    const Id l = SubId();
    Id src;
    if (off & 0x8000) {
        src = Or(And(l, CU(~3u)), And(Shr(CU(off), Shl(And(l, 3u), 1)), 3u));
    } else {
        const Id m = Or(And(And(l, 31u), CU(off & 0x1F)), CU(off >> 5 & 0x1F));
        src = Or(And(l, CU(~31u)), Xor(m, CU(off >> 10 & 0x1F)));
    }
    return ShuffleU(v, src);
}

// ---------------------------------------------------------------------------------------------------------------------
// Scalar instructions.

void Gen::Salu(const Inst& i) {
    const u32 op = i.op;
    const Id tr = tBool;
    if (i.enc == SOP2) {
        const Id a = Uv(i, 0), b2 = Uv(i, 1);
        auto done32 = [&](Id r, Id c) {
            if (c) St(scc, c);
            St(D(i.dst), r);
        };
        auto done64 = [&](Id r, Id c) {
            if (c) St(scc, c);
            St64(i.dst, r);
        };
        auto nz32 = [&](Id r) { return Ne(r, 0); };
        auto nz64 = [&](Id r) { return R(spv::OpINotEqual, tr, r, CU64(0)); };
        auto a64 = [&] { return P64(i.src[0], i.lit); };
        auto b64 = [&] { return P64(i.src[1], i.lit); };
        auto sgn = [&](spv::Op o, Id x, Id y) { return Bool(o, Iu(x), Iu(y)); };
        switch (op) {
        case 0: { const Id r = Add(a, b2); return done32(r, Bool(spv::OpULessThan, r, a)); }
        case 1: { const Id r = Sub(a, b2); return done32(r, Bool(spv::OpUGreaterThan, b2, a)); }
        case 2: { const Id r = Add(a, b2); return done32(r, Ne(Shr(And(Not(Xor(a, b2)), Xor(a, r)), 31), 0)); }
        case 3: { const Id r = Sub(a, b2); return done32(r, Ne(Shr(And(Xor(a, b2), Xor(a, r)), 31), 0)); }
        case 4: {
            const Id c = SelU(Ld(scc, tBool), CU(1), CU(0)), r = Add(Add(a, b2), c);
            const Id s = R(spv::OpIAdd, tU64, R(spv::OpIAdd, tU64, W64(a), W64(b2)), W64(c));
            return done32(r, R(spv::OpINotEqual, tr, R(spv::OpShiftRightLogical, tU64, s, CU(32)), CU64(0)));
        }
        case 5: {
            const Id c = SelU(Ld(scc, tBool), CU(1), CU(0)), r = Sub(Sub(a, b2), c);
            return done32(r, Bool(spv::OpUGreaterThan, R(spv::OpIAdd, tU64, W64(b2), W64(c)), W64(a)));
        }
        case 6: return done32(Ui(E(tI32, GLSLstd450SMin, Iu(a), Iu(b2))), sgn(spv::OpSLessThan, a, b2));
        case 7: return done32(E(tU32, GLSLstd450UMin, a, b2), Bool(spv::OpULessThan, a, b2));
        case 8: return done32(Ui(E(tI32, GLSLstd450SMax, Iu(a), Iu(b2))), sgn(spv::OpSGreaterThan, a, b2));
        case 9: return done32(E(tU32, GLSLstd450UMax, a, b2), Bool(spv::OpUGreaterThan, a, b2));
        case 10: return done32(SelU(Ld(scc, tBool), a, b2), Id());
        case 11: return done64(Sel(tU64, Ld(scc, tBool), a64(), b64()), Id());
        case 30: { const Id r = Shl(a, And(b2, 31u)); return done32(r, nz32(r)); }
        case 31: { const Id r = R(spv::OpShiftLeftLogical, tU64, a64(), And(b2, 63u)); return done64(r, nz64(r)); }
        case 32: { const Id r = Shr(a, And(b2, 31u)); return done32(r, nz32(r)); }
        case 33: { const Id r = R(spv::OpShiftRightLogical, tU64, a64(), And(b2, 63u)); return done64(r, nz64(r)); }
        case 34: { const Id r = Sar(a, And(b2, 31u)); return done32(r, nz32(r)); }
        case 35: { const Id r = R(spv::OpBitcast, tU64, R(spv::OpShiftRightArithmetic, tI64, R(spv::OpBitcast, tI64, a64()), And(b2, 63u))); return done64(r, nz64(r)); }
        case 36: return done32(Shl(Sub(Shl(CU(1), And(a, 31u)), CU(1)), And(b2, 31u)), Id());
        case 38: return done32(Mul(a, b2), Id());
        case 39: { const Id r = Bfe(a, b2, And(Shr(b2, 16), 0x7Fu)); return done32(r, nz32(r)); }
        case 40: { const Id r = BfeI(a, b2, And(Shr(b2, 16), 0x7Fu)); return done32(r, nz32(r)); }
        case 44: { const Id r = Ui(E(tI32, GLSLstd450SAbs, Iu(Sub(a, b2)))); return done32(r, nz32(r)); }
        }
        if (op >= 14 && op <= 29) {
            const u32 k = (op - 14) / 2;
            auto logic = [&](Id x, Id y, Id t) {
                auto bit = [&](spv::Op o, Id u, Id v) { return R(o, t, u, v); };
                auto inv = [&](Id v) { return R(spv::OpNot, t, v); };
                switch (k) {
                case 0: return bit(spv::OpBitwiseAnd, x, y);
                case 1: return bit(spv::OpBitwiseOr, x, y);
                case 2: return bit(spv::OpBitwiseXor, x, y);
                case 3: return bit(spv::OpBitwiseAnd, x, inv(y));
                case 4: return bit(spv::OpBitwiseOr, x, inv(y));
                case 5: return inv(bit(spv::OpBitwiseAnd, x, y));
                case 6: return inv(bit(spv::OpBitwiseOr, x, y));
                default: return inv(bit(spv::OpBitwiseXor, x, y));
                }
            };
            if (op & 1) { const Id r = logic(a64(), b64(), tU64); return done64(r, nz64(r)); }
            const Id r = logic(a, b2, tU32);
            return done32(r, nz32(r));
        }
        return Fail(std::format("sop2 {} at {:#x}", op, i.pc));
    }
    if (i.enc == SOPK) {
        const u32 imm = u32(s32(std::int16_t(i.lit)));
        const Id d = U(i.dst, 0);
        static constexpr spv::Op scmp[]{spv::OpIEqual, spv::OpINotEqual, spv::OpSGreaterThan, spv::OpSGreaterThanEqual, spv::OpSLessThan, spv::OpSLessThanEqual};
        static constexpr spv::Op ucmp[]{spv::OpIEqual, spv::OpINotEqual, spv::OpUGreaterThan, spv::OpUGreaterThanEqual, spv::OpULessThan, spv::OpULessThanEqual};
        if (op == 0) return St(D(i.dst), CU(imm));
        if (op == 2) return St(D(i.dst), SelU(Ld(scc, tBool), CU(imm), d));
        if (op >= 3 && op <= 8) return St(scc, Bool(scmp[op - 3], Iu(d), CI(s32(imm))));
        if (op >= 9 && op <= 14) return St(scc, Bool(ucmp[op - 9], d, CU(i.lit)));
        if (op == 15) {
            const Id r = Add(d, CU(imm));
            St(scc, Ne(Shr(And(Not(Xor(d, CU(imm))), Xor(d, r)), 31), 0));
            return St(D(i.dst), r);
        }
        if (op == 16) return St(D(i.dst), Mul(d, CU(imm)));
        if (op == 18) return St(D(i.dst), CU(0));  // s_getreg: hardware state is not modelled
        if (op == 19 || op == 21) return;          // s_setreg: mode bits
        return Fail(std::format("sopk {} at {:#x}", op, i.pc));
    }
    if (i.enc == SOP1) {
        auto nz32 = [&](Id r) { return Ne(r, 0); };
        auto nz64 = [&](Id r) { return R(spv::OpINotEqual, tr, r, CU64(0)); };
        auto one = [&](Id r, bool flag) {
            if (flag) St(scc, nz32(r));
            St(D(i.dst), r);
        };
        auto wide = [&](Id r, bool flag) {
            if (flag) St(scc, nz64(r));
            St64(i.dst, r);
        };
        switch (op) {
        case 3: return St(D(i.dst), Uv(i, 0));
        case 4: return wide(P64(i.src[0], i.lit), false);
        case 5: return St(D(i.dst), SelU(Ld(scc, tBool), Uv(i, 0), U(i.dst, 0)));
        case 6: {
            auto [dl, dh] = D64(i.dst);
            return wide(Sel(tU64, Ld(scc, tBool), P64(i.src[0], i.lit), U64(LdU(dl), LdU(dh))), false);
        }
        case 7: return one(Not(Uv(i, 0)), true);
        case 8: return wide(R(spv::OpNot, tU64, P64(i.src[0], i.lit)), true);
        case 9: return one(Wqm32(Uv(i, 0)), true);
        case 10: return wide(Wqm(P64(i.src[0], i.lit)), true);
        case 11: return one(R(spv::OpBitReverse, tU32, Uv(i, 0)), false);
        case 15: return one(R(spv::OpBitCount, tU32, Uv(i, 0)), true);
        case 16: {
            auto [lo, hi] = U64p(i.src[0], i.lit);
            return one(Add(R(spv::OpBitCount, tU32, lo), R(spv::OpBitCount, tU32, hi)), true);
        }
        case 19: return one(Ui(E(tI32, GLSLstd450FindILsb, Uv(i, 0))), false);
        case 21: {
            const Id a = Uv(i, 0);
            return one(SelU(Eq(a, 0), CU(0xFFFFFFFFu), Sub(CU(31), Ui(E(tI32, GLSLstd450FindUMsb, a)))), false);
        }
        case 25: return one(Sar(Shl(Uv(i, 0), 24), CU(24)), false);
        case 26: return one(Sar(Shl(Uv(i, 0), 16), CU(16)), false);
        case 27: return St(D(i.dst), And(U(i.dst, 0), Not(Shl(CU(1), And(Uv(i, 0), 31u)))));
        case 29: return St(D(i.dst), Or(U(i.dst, 0), Shl(CU(1), And(Uv(i, 0), 31u))));
        case 52: return one(Ui(E(tI32, GLSLstd450SAbs, Iu(Uv(i, 0)))), true);
        }
        if (op >= 36 && op <= 43) {
            const Id a = P64(i.src[0], i.lit), e = U64(LdU(exec_lo), LdU(exec_hi));
            auto bit = [&](spv::Op o, Id u, Id v) { return R(o, tU64, u, v); };
            auto inv = [&](Id v) { return R(spv::OpNot, tU64, v); };
            Id r;
            switch (op - 36) {
            case 0: r = bit(spv::OpBitwiseAnd, a, e); break;
            case 1: r = bit(spv::OpBitwiseOr, a, e); break;
            case 2: r = bit(spv::OpBitwiseXor, a, e); break;
            case 3: r = bit(spv::OpBitwiseAnd, a, inv(e)); break;
            case 4: r = bit(spv::OpBitwiseOr, a, inv(e)); break;
            case 5: r = inv(bit(spv::OpBitwiseAnd, a, e)); break;
            case 6: r = inv(bit(spv::OpBitwiseOr, a, e)); break;
            default: r = inv(bit(spv::OpBitwiseXor, a, e)); break;
            }
            St64(i.dst, e);
            St(exec_lo, Lo(r)), St(exec_hi, Hi(r));
            return St(scc, nz64(r));
        }
        return Fail(std::format("sop1 {} at {:#x}", op, i.pc));
    }
    if (i.enc == SOPC) {
        static constexpr spv::Op scmp[]{spv::OpIEqual, spv::OpINotEqual, spv::OpSGreaterThan, spv::OpSGreaterThanEqual, spv::OpSLessThan, spv::OpSLessThanEqual};
        static constexpr spv::Op ucmp[]{spv::OpIEqual, spv::OpINotEqual, spv::OpUGreaterThan, spv::OpUGreaterThanEqual, spv::OpULessThan, spv::OpULessThanEqual};
        const Id a = Uv(i, 0), b2 = Uv(i, 1);
        if (op <= 5) return St(scc, Bool(scmp[op], Iu(a), Iu(b2)));
        if (op <= 11) return St(scc, Bool(ucmp[op - 6], a, b2));
        if (op == 12 || op == 13) return St(scc, Eq(And(Shr(a, And(b2, 31u)), 1u), op - 12));
        if (op == 14 || op == 15)
            return St(scc, R(spv::OpIEqual, tBool, R(spv::OpBitwiseAnd, tU64, R(spv::OpShiftRightLogical, tU64, P64(i.src[0], i.lit), And(b2, 63u)), CU64(1)), CU64(op - 14)));
        if (op == 16) return;  // s_setvskip
        return Fail(std::format("sopc {} at {:#x}", op, i.pc));
    }
}

void Gen::Smrd(const Inst& i) {
    const u32 base = i.src[0];
    if (i.tex >= 0) {  // constants the draw table carries
        for (u32 k = 0, n = 1u << (i.op <= 4 ? i.op : i.op - 8); k < n; ++k) St(D(i.dst + k), TableAt(i.tex + k));
        return;
    }
    const Id off = i.imm ? CU(i.src[1] * 4) : i.src[1] == 255 ? CU(i.lit * 4) : U(i.src[1], 0);
    if (i.op <= 4) {
        const u32 n = 1u << i.op;
        const Id addr = R(spv::OpIAdd, tU64, Ptr64(Sg(base), Sg(base + 1)), W64(off));
        Id v[16];
        for (u32 k = 0; k < n; k += 4) LDVec(k ? R(spv::OpIAdd, tU64, addr, CU64(4 * k)) : addr, std::min(n - k, 4u), &v[k]);
        for (u32 k = 0; k < n; ++k) St(D(i.dst + k), v[k]);
        return;
    }
    if (i.op >= 8 && i.op <= 12) {
        // Range checked like the hardware: num_records bytes, or records of `stride` bytes.
        const u32 n = 1u << (i.op - 8);
        const Id a64 = Ptr64(Sg(base), Sg(base + 1)), st = And(Shr(Sg(base + 1), 16), 0x3FFFu);
        const Id lim = SelU(Ne(st, 0), Mul(Sg(base + 2), st), Sg(base + 2));
        Id v[16], fast[16], fast_block;
        IfCtx range{};
        if (n > 1) {
            // Vector loads require a complete range without 32-bit offset wraparound.
            range = IfBegin(LAnd(Bool(spv::OpULessThanEqual, off, lim), Bool(spv::OpUGreaterThanEqual, Sub(lim, off), CU(4 * n))), true);
            const Id addr = R(spv::OpIAdd, tU64, a64, W64(off));
            for (u32 k = 0; k < n; k += 4) LDVec(k ? R(spv::OpIAdd, tU64, addr, CU64(4 * k)) : addr, std::min(n - k, 4u), &fast[k]);
            fast_block = b.Cur();
            IfElse(range);
        }
        for (u32 k = 0; k < n; ++k) {
            const Id o = Add(off, CU(4 * k));
            v[k] = Guard(tU32, Bool(spv::OpULessThan, o, lim), CU(0), [&] { return LD(R(spv::OpIAdd, tU64, a64, W64(o))); });
        }
        if (n > 1) {
            const Id slow_block = b.Cur();
            IfEnd(range);
            for (u32 k = 0; k < n; ++k) v[k] = R(spv::OpPhi, tU32, fast[k], fast_block, v[k], slow_block);
        }
        for (u32 k = 0; k < n; ++k) St(D(i.dst + k), v[k]);
        return;
    }
    if (i.op == 30) {  // s_memtime
        auto [lo, hi] = D64(i.dst);
        St(lo, CU(0)), St(hi, CU(0));
        return;
    }
    if (i.op == 29 || i.op == 31) return;  // cache invalidation
    Fail(std::format("smrd {} at {:#x}", i.op, i.pc));
}

// ---------------------------------------------------------------------------------------------------------------------
// Vector instructions.

void Gen::Valu(const Inst& i) {
    const u32 op = i.op;
    auto f = [&](int k) { return Fv(i, k); };
    auto u = [&](int k) { return Uv(i, k); };
    auto si = [&](int k) { return Iv(i, k); };
    auto F = [&](Id e) { WV(i.dst, FR(i, e)); };
    auto Uw = [&](Id e) { WV(i.dst, e); };
    if (op < 256) {
        // f32 (0x00, cmpx 0x10, cmps 0x40/0x50), i32 (0x80/0x90), u32 (0xC0/0xD0), class (0x88/0x98)
        const u32 g = op & 0xF0, cc = op & 0xF;
        const bool cls = op == 0x88 || op == 0x98, fp = g == 0x00 || g == 0x10 || g == 0x40 || g == 0x50;
        const bool sgn = g == 0x80 || g == 0x90, uns = g == 0xC0 || g == 0xD0;
        if (!cls && !fp && !((sgn || uns) && cc < 8)) return Fail(std::format("vopc {:#x} at {:#x}", op, i.pc));
        Id c;
        if (cls) {
            c = FClass(f(0), u(1));
        } else if (fp) {
            const Id x = f(0), y = f(1);
            auto lt = [&] { return FLt(x, y); };
            auto gt = [&] { return FGt(x, y); };
            auto le = [&] { return Bool(spv::OpFOrdLessThanEqual, x, y); };
            auto ge = [&] { return FGe(x, y); };
            auto eq = [&] { return Bool(spv::OpFOrdEqual, x, y); };
            auto nan = [&] { return LOr(R(spv::OpIsNan, tBool, x), R(spv::OpIsNan, tBool, y)); };
            switch (cc) {
            case 0: c = CB(false); break;
            case 1: c = lt(); break;
            case 2: c = eq(); break;
            case 3: c = le(); break;
            case 4: c = gt(); break;
            case 5: c = LOr(lt(), gt()); break;
            case 6: c = ge(); break;
            case 7: c = LNot(nan()); break;
            case 8: c = nan(); break;
            case 9: c = LNot(ge()); break;
            case 10: c = LNot(LOr(lt(), gt())); break;
            case 11: c = LNot(gt()); break;
            case 12: c = LNot(le()); break;
            case 13: c = LNot(eq()); break;
            case 14: c = LNot(lt()); break;
            default: c = CB(true); break;
            }
        } else {
            const Id x = sgn ? si(0) : u(0), y = sgn ? si(1) : u(1);
            auto cmp = [&](spv::Op s, spv::Op un) { return Bool(sgn ? s : un, x, y); };
            switch (cc) {
            case 0: c = CB(false); break;
            case 1: c = cmp(spv::OpSLessThan, spv::OpULessThan); break;
            case 2: c = Bool(spv::OpIEqual, x, y); break;
            case 3: c = cmp(spv::OpSLessThanEqual, spv::OpULessThanEqual); break;
            case 4: c = cmp(spv::OpSGreaterThan, spv::OpUGreaterThan); break;
            case 5: c = Bool(spv::OpINotEqual, x, y); break;
            case 6: c = cmp(spv::OpSGreaterThanEqual, spv::OpUGreaterThanEqual); break;
            default: c = CB(true); break;
            }
        }
        WM(i.sdst, c, i.lane_only);
        if (g == 0x10 || g == 0x50 || g == 0x90 || g == 0xD0) WM(126, c);
        return;
    }
    switch (op) {
    case 256: {
        const Id x = u(0), y = u(1);
        return Uw(SelU(LaneMask(i.src[2], i.lit), y, x));
    }
    case 257: return St(D(i.dst), i.spill >= 0 ? LdU(Spill(i.spill)) : ShuffleU(u(0), u(1)));
    case 258: {
        if (i.spill >= 0) St(Spill(i.spill), u(0));
        const Id n = Vr(i.dst);
        return St(n, SelU(Bool(spv::OpIEqual, SubId(), R(spv::OpUMod, tU32, u(1), SubSize())), u(0), LdU(n)));
    }
    case 259: return F(FAdd(f(0), f(1)));
    case 260: return F(FSub(f(0), f(1)));
    case 261: return F(FSub(f(1), f(0)));
    case 262: return F(FAdd(LMul(f(0), f(1)), Fu(LdU(Vr(i.dst)))));
    case 263: return F(LMul(f(0), f(1)));
    case 264: return F(FMul(f(0), f(1)));
    case 265: {
        auto s24 = [&](Id x) { return Iu(Sar(Shl(x, 8), CU(8))); };
        return Uw(Ui(R(spv::OpIMul, tI32, s24(u(0)), s24(u(1)))));
    }
    case 266: {
        auto s24 = [&](Id x) { return R(spv::OpSConvert, tI64, Iu(Sar(Shl(x, 8), CU(8)))); };
        return Uw(Lo(R(spv::OpShiftRightArithmetic, tI64, R(spv::OpIMul, tI64, s24(u(0)), s24(u(1))), CU(32))));
    }
    case 267: return Uw(Mul(And(u(0), 0xFFFFFFu), And(u(1), 0xFFFFFFu)));
    case 268: return Uw(Hi(R(spv::OpIMul, tU64, W64(And(u(0), 0xFFFFFFu)), W64(And(u(1), 0xFFFFFFu)))));
    case 269: { const Id x = f(0), y = f(1); return F(SelF(FLt(x, y), x, y)); }
    case 270: { const Id x = f(0), y = f(1); return F(SelF(FGe(x, y), x, y)); }
    case 271: return F(FMin(f(0), f(1)));
    case 272: {
        const Id result = FMax(f(0), f(1));
        return F(i.pc == depth_resolve.max_pc && extra_depth ? FMax(result, extra_depth) : result);
    }
    case 273: return Uw(Ui(E(tI32, GLSLstd450SMin, si(0), si(1))));
    case 274: return Uw(Ui(E(tI32, GLSLstd450SMax, si(0), si(1))));
    case 275: return Uw(E(tU32, GLSLstd450UMin, u(0), u(1)));
    case 276: return Uw(E(tU32, GLSLstd450UMax, u(0), u(1)));
    case 277: return Uw(Shr(u(0), And(u(1), 31u)));
    case 278: return Uw(Shr(u(1), And(u(0), 31u)));
    case 279: return Uw(Sar(u(0), And(u(1), 31u)));
    case 280: return Uw(Sar(u(1), And(u(0), 31u)));
    case 281: return Uw(Shl(u(0), And(u(1), 31u)));
    case 282: return Uw(Shl(u(1), And(u(0), 31u)));
    case 283: return Uw(And(u(0), u(1)));
    case 284: return Uw(Or(u(0), u(1)));
    case 285: return Uw(Xor(u(0), u(1)));
    case 286: return Uw(Shl(Sub(Shl(CU(1), And(u(0), 31u)), CU(1)), And(u(1), 31u)));
    case 287: return F(FAdd(FMul(f(0), f(1)), Fu(LdU(Vr(i.dst)))));
    case 288: return F(FAdd(FMul(f(0), CFb(i.lit)), f(1)));
    case 289: return F(FAdd(FMul(f(0), f(1)), CFb(i.lit)));
    case 290: return Uw(Add(R(spv::OpBitCount, tU32, u(0)), u(1)));
    case 291: return Uw(Add(R(spv::OpBitCount, tU32, And(u(0), Extract(tU32, SubLt(), 0))), u(1)));
    case 292: return Uw(Add(R(spv::OpBitCount, tU32, And(u(0), Extract(tU32, SubLt(), 1))), u(1)));
    case 293: case 294: case 295: case 296: case 297: case 298: {
        const Id a = u(0), b2 = u(1), c = op >= 296 ? SelU(LaneMask(i.src[2], i.lit), CU(1), CU(0)) : CU(0);
        Id r, carry;
        switch (op - 293) {
        case 0: r = Add(a, b2), carry = Bool(spv::OpULessThan, r, a); break;
        case 1: r = Sub(a, b2), carry = Bool(spv::OpUGreaterThan, b2, a); break;
        case 2: r = Sub(b2, a), carry = Bool(spv::OpUGreaterThan, a, b2); break;
        case 3: r = Add(Add(a, b2), c), carry = R(spv::OpINotEqual, tBool, R(spv::OpShiftRightLogical, tU64, R(spv::OpIAdd, tU64, R(spv::OpIAdd, tU64, W64(a), W64(b2)), W64(c)), CU(32)), CU64(0)); break;
        case 4: r = Sub(Sub(a, b2), c), carry = Bool(spv::OpUGreaterThan, R(spv::OpIAdd, tU64, W64(b2), W64(c)), W64(a)); break;
        default: r = Sub(Sub(b2, a), c), carry = Bool(spv::OpUGreaterThan, R(spv::OpIAdd, tU64, W64(a), W64(c)), W64(b2)); break;
        }
        WM(i.sdst, carry);
        return WV(i.dst, r);
    }
    case 299: return F(E(tF32, GLSLstd450Ldexp, f(0), si(1)));
    case 301: return Uw(E(tU32, GLSLstd450PackSnorm2x16, R(spv::OpCompositeConstruct, tV2F, f(0), f(1))));
    case 302: return Uw(E(tU32, GLSLstd450PackUnorm2x16, R(spv::OpCompositeConstruct, tV2F, f(0), f(1))));
    case 303: return Uw(E(tU32, GLSLstd450PackHalf2x16, R(spv::OpCompositeConstruct, tV2F, f(0), f(1))));
    case 304: case 305: return Uw(Or(And(u(0), 0xFFFFu), Shl(u(1), 16)));
    case 320: return F(FAdd(LMul(f(0), f(1)), f(2)));
    case 321: return F(FAdd(FMul(f(0), f(1)), f(2)));
    case 322: {
        auto s24 = [&](Id x) { return Iu(Sar(Shl(x, 8), CU(8))); };
        return Uw(Ui(R(spv::OpIAdd, tI32, R(spv::OpIMul, tI32, s24(u(0)), s24(u(1))), si(2))));
    }
    case 323: return Uw(Add(Mul(And(u(0), 0xFFFFFFu), And(u(1), 0xFFFFFFu)), u(2)));
    case 324: case 325: case 326: case 327: return F(Cube(int(op - 324), f(0), f(1), f(2)));
    case 328: return Uw(Bfe(u(0), u(1), u(2)));
    case 329: return Uw(BfeI(u(0), u(1), u(2)));
    case 330: return Uw(Or(And(u(0), u(1)), And(Not(u(0)), u(2))));
    case 331: return F(E(tF32, GLSLstd450Fma, f(0), f(1), f(2)));
    case 334: return Uw(Lo(R(spv::OpShiftRightLogical, tU64, U64(u(1), u(0)), And(u(2), 31u))));
    case 335: return Uw(Lo(R(spv::OpShiftRightLogical, tU64, U64(u(1), u(0)), Shl(And(u(2), 3u), 3u))));
    case 337: return F(FMin(FMin(f(0), f(1)), f(2)));
    case 338: return Uw(Ui(E(tI32, GLSLstd450SMin, E(tI32, GLSLstd450SMin, si(0), si(1)), si(2))));
    case 339: return Uw(E(tU32, GLSLstd450UMin, E(tU32, GLSLstd450UMin, u(0), u(1)), u(2)));
    case 340: return F(FMax(FMax(f(0), f(1)), f(2)));
    case 341: return Uw(Ui(E(tI32, GLSLstd450SMax, E(tI32, GLSLstd450SMax, si(0), si(1)), si(2))));
    case 342: return Uw(E(tU32, GLSLstd450UMax, E(tU32, GLSLstd450UMax, u(0), u(1)), u(2)));
    case 343: return F(Med3(f(0), f(1), f(2)));
    case 344: {
        const Id x = si(0), y = si(1), z = si(2);
        auto mn = [&](Id p1, Id p2) { return E(tI32, GLSLstd450SMin, p1, p2); };
        auto mx = [&](Id p1, Id p2) { return E(tI32, GLSLstd450SMax, p1, p2); };
        return Uw(Ui(mx(mn(x, y), mn(mx(x, y), z))));
    }
    case 345: {
        const Id x = u(0), y = u(1), z = u(2);
        auto mn = [&](Id p1, Id p2) { return E(tU32, GLSLstd450UMin, p1, p2); };
        auto mx = [&](Id p1, Id p2) { return E(tU32, GLSLstd450UMax, p1, p2); };
        return Uw(mx(mn(x, y), mn(mx(x, y), z)));
    }
    case 349: { const Id x = u(0), y = u(1); return Uw(Add(SelU(Bool(spv::OpUGreaterThan, x, y), Sub(x, y), Sub(y, x)), u(2))); }
    case 361: case 363: return Uw(Mul(u(0), u(1)));
    case 362: return Uw(Hi(R(spv::OpIMul, tU64, W64(u(0)), W64(u(1)))));
    case 364: return Uw(Lo(R(spv::OpShiftRightArithmetic, tI64, R(spv::OpIMul, tI64, R(spv::OpSConvert, tI64, si(0)), R(spv::OpSConvert, tI64, si(1))), CU(32))));
    case 384: return;
    case 385: return Uw(u(0));
    case 386: {
        const Id first = SelU(LOr(Bool(spv::OpULessThanEqual, SubSize(), CU(32)), Ne(LdU(exec_lo), 0)), Ui(E(tI32, GLSLstd450FindILsb, LdU(exec_lo))),
                              Add(CU(32), Ui(E(tI32, GLSLstd450FindILsb, LdU(exec_hi)))));
        return St(D(i.dst), ShuffleU(u(0), first));
    }
    case 389: return F(R(spv::OpConvertSToF, tF32, si(0)));
    case 390: return F(R(spv::OpConvertUToF, tF32, u(0)));
    case 391: return Uw(CvtU(f(0)));
    case 392: return Uw(CvtI(f(0)));
    case 394: return Uw(E(tU32, GLSLstd450PackHalf2x16, R(spv::OpCompositeConstruct, tV2F, f(0), CF(0.f))));
    case 395: return F(Extract(tF32, E(tV2F, GLSLstd450UnpackHalf2x16, u(0)), 0));
    case 396: return Uw(CvtI(E(tF32, GLSLstd450Floor, FAdd(f(0), CF(0.5f)))));
    case 397: return Uw(CvtI(E(tF32, GLSLstd450Floor, f(0))));
    case 401: case 402: case 403: case 404: return F(R(spv::OpConvertUToF, tF32, And(Shr(u(0), (op - 401) * 8), 0xFFu)));
    case 416: return F(E(tF32, GLSLstd450Fract, f(0)));
    case 417: return F(E(tF32, GLSLstd450Trunc, f(0)));
    case 418: return F(E(tF32, GLSLstd450Ceil, f(0)));
    case 419: return F(E(tF32, GLSLstd450RoundEven, f(0)));
    case 420: return F(E(tF32, GLSLstd450Floor, f(0)));
    case 421: return F(E(tF32, GLSLstd450Exp2, f(0)));
    case 422: return F(FMax(E(tF32, GLSLstd450Log2, f(0)), CF(-kFltMax)));
    case 423: return F(E(tF32, GLSLstd450Log2, f(0)));
    case 424: return F(E(tF32, GLSLstd450FClamp, FDiv(CF(1.f), f(0)), CF(-kFltMax), CF(kFltMax)));
    case 425: case 426: case 427: return F(FDiv(CF(1.f), f(0)));
    case 428: return F(E(tF32, GLSLstd450FClamp, E(tF32, GLSLstd450InverseSqrt, f(0)), CF(-kFltMax), CF(kFltMax)));
    case 429: case 430: return F(E(tF32, GLSLstd450InverseSqrt, f(0)));
    case 435: return F(E(tF32, GLSLstd450Sqrt, f(0)));
    case 437: return F(E(tF32, GLSLstd450Sin, FMul(f(0), CF(6.283185307f))));  // the hardware takes revolutions
    case 438: return F(E(tF32, GLSLstd450Cos, FMul(f(0), CF(6.283185307f))));
    case 439: return Uw(Not(u(0)));
    case 440: return Uw(R(spv::OpBitReverse, tU32, u(0)));
    case 441: { const Id x = u(0); return Uw(SelU(Eq(x, 0), CU(0xFFFFFFFFu), Sub(CU(31), Ui(E(tI32, GLSLstd450FindUMsb, x))))); }
    case 442: return Uw(Ui(E(tI32, GLSLstd450FindILsb, u(0))));
    case 443: {
        const Id x = si(0);
        const Id bad = LOr(Bool(spv::OpIEqual, x, CI(0)), Bool(spv::OpIEqual, x, CI(-1)));
        return Uw(SelU(bad, CU(0xFFFFFFFFu), Sub(CU(31), Ui(E(tI32, GLSLstd450FindSMsb, x)))));
    }
    }
    Fail(std::format("valu {} at {:#x}", op, i.pc));
}

// ---------------------------------------------------------------------------------------------------------------------
// Images.

void Gen::Mimg(const Inst& i) {
    const u32 dmask = i.w0 >> 8 & 0xF, unorm = i.w0 >> 12 & 1, op = i.op, vaddr = i.w1 & 0xFF, vdata = i.w1 >> 8 & 0xFF;
    if (size_t(i.tex) >= spec.size() || (i.smp >= 0 && size_t(i.smp) >= spec.size())) return Fail("slot without spec");
    const SlotSpec& t = spec[i.tex];
    const Slot& slot = a.slots[i.tex];
    const int kind = Kind(t.dim), cls = t.stencil ? int(Class::Uint) : int(t.cls);
    const bool storage = slot.kind == SlotKind::Storage;
    if (storage && kind > 2) return Fail("storage image dimension");
    const Id slot_index = TableAt(res + i.tex);
    u32 a_next = vaddr;
    auto addr = [&] { return LdU(Vr(a_next++)); };
    auto fa = [&] { return Fu(addr()); };
    auto ia = [&] { return Iu(addr()); };
    const bool d1 = t.dim == Dim::D1 || t.dim == Dim::D1Array;
    const Id vec4t = V4(cls), zero = Zero(cls), one = One(cls);
    const Id img = LoadImage(kind, cls, storage, slot_index);
    auto c_is2 = [&](Id) { return kind == 0 || kind == 4 || (t.dim == Dim::D1); };  // coordinates already two-component
    // Sub-rectangle of the bound image: texel accesses outside the T# read zero / are dropped, like the hardware's bounds.
    // Its extent comes from the T# in the SGPRs (word 2), at most the image's (read from the bound image: not part of the key).
    const bool sub = t.part && (kind == 0 || kind == 1) && !d1;
    auto size_xy = [&]() -> Id {  // the image's size, x and y
        const Id s = ImageSize(img, kind, storage);
        if (kind == 0) return s;
        return R(spv::OpVectorShuffle, tV2I, s, s, 0u, 1u);
    };
    Id isize, tsize;
    if (sub) {
        isize = size_xy();
        const Id w = Sg((i.w1 >> 16 & 0x1F) * 4 + 2);
        const Id ext = R(spv::OpIAdd, tV2U, R(spv::OpCompositeConstruct, tV2U, And(w, 0x3FFFu), And(Shr(w, 14), 0x3FFFu)), R(spv::OpCompositeConstruct, tV2U, CU(1), CU(1)));
        const Id ei = R(spv::OpBitcast, tV2I, ext);
        tsize = E(tV2I, GLSLstd450SMin, ei, isize);
    }
    auto inside = [&](Id c) {  // c: ivec2 or ivec3
        const Id cxy = c_is2(c) ? c : R(spv::OpVectorShuffle, tV2I, c, c, 0u, 1u);
        return R(spv::OpAll, tBool, R(spv::OpSLessThan, tV2B, cxy, tsize));
    };
    auto write = [&](Id r) {  // r: vec4/uvec4/ivec4
        u32 n = vdata;
        for (u32 c = 0; c < 4; ++c) {
            if (!(dmask >> c & 1)) continue;
            if (t.stencil) {
                const u32 sel = t.dst_sel >> (3 * c) & 7;
                const Id v = sel == 4 ? FDiv(R(spv::OpConvertUToF, tF32, Extract(tU32, r, 0)), CF(255.f)) : CF(sel == 1 || sel == 7 ? 1.f : 0.f);
                WV(n++, Uf(v));
            } else WV(n++, cls == 0 ? Uf(Extract(tF32, r, c)) : cls == 1 ? Extract(tU32, r, c) : Ui(Extract(tI32, r, c)));
        }
    };
    auto v2i = [&](Id x, Id y) { return R(spv::OpCompositeConstruct, tV2I, x, y); };
    auto v3i = [&](Id x, Id y, Id z) { return R(spv::OpCompositeConstruct, tV3I, x, y, z); };
    // Integer texel coordinates: x[, y][, z | slice]
    auto icoords = [&]() -> Id {
        switch (t.dim) {
        case Dim::D1: return v2i(ia(), CI(0));
        case Dim::D1Array: { const Id x = ia(); return v3i(x, CI(0), ia()); }
        case Dim::D2: case Dim::D2Ms: { const Id x = ia(); return v2i(x, ia()); }
        default: { const Id x = ia(), y = ia(); return v3i(x, y, ia()); }
        }
    };
    if (op == 0 || op == 1) {  // image_load / image_load_mip
        if (t.dim == Dim::Cube) return Fail("image_load on a cube");
        const Id c = icoords();
        Id e;
        if (storage) {
            storage_read[kind][cls] = true;
            b.Capability(spv::CapabilityStorageImageReadWithoutFormat);
            e = R(spv::OpImageRead, vec4t, img, c);
        }
        else if (t.dim == Dim::D2Ms || t.dim == Dim::D2MsArray) {
            e = R(spv::OpImageFetch, vec4t, img, c, 0x40u, ia());
            if (i.pc == depth_resolve.load_pc) {
                const Id s2 = R(spv::OpImageFetch, vec4t, img, c, 0x40u, CI(2));
                const Id s3 = R(spv::OpImageFetch, vec4t, img, c, 0x40u, CI(3));
                extra_depth = FMax(Extract(tF32, s2, 0), Extract(tF32, s3, 0));
            }
        }
        else e = R(spv::OpImageFetch, vec4t, img, c, 0x2u, op == 1 ? ia() : CI(0));
        if (sub) e = Sel(vec4t, inside(c), e, R(spv::OpCompositeConstruct, vec4t, zero, zero, zero, zero));
        if (storage && t.dst_sel) {  // the view cannot swizzle: DST_SEL here (0, 1 or a texel channel per component)
            Id s[4];
            for (u32 k = 0; k < 4; ++k) s[k] = (t.dst_sel >> 3 * k & 7) >= 4 ? Extract(Sc(cls), e, (t.dst_sel >> 3 * k & 7) - 4) : t.dst_sel >> 3 * k & 1 ? one : zero;
            e = R(spv::OpCompositeConstruct, vec4t, s[0], s[1], s[2], s[3]);
        }
        return write(e);
    }
    if (op == 8) {  // image_store
        if (!storage) return Fail("image_store without storage slot");
        const Id c = icoords();
        Id v[4];
        u32 n = vdata;
        for (u32 k = 0; k < 4; ++k) v[k] = dmask >> k & 1 ? (cls == 0 ? Fu(LdU(Vr(n++))) : cls == 1 ? LdU(Vr(n++)) : Iu(LdU(Vr(n++)))) : zero;
        if (t.dst_sel) {  // the inverse of DST_SEL: each texel channel takes the first component selecting it, others zero
            Id w[4]{zero, zero, zero, zero};
            for (u32 k = 4; k--;)
                if (const u32 s = t.dst_sel >> 3 * k & 7; s >= 4) w[s - 4] = v[k];
            std::copy_n(w, 4, v);
        }
        const Id texel = R(spv::OpCompositeConstruct, vec4t, v[0], v[1], v[2], v[3]);
        Id cond = Live();
        if (sub) cond = LAnd(cond, inside(c));
        const bool guarded = AllLive() ? sub : true;
        if (!guarded) return b.Do(spv::OpImageWrite, img, c, texel);
        const IfCtx f2 = IfBegin(cond);
        b.Do(spv::OpImageWrite, img, c, texel);
        return IfEnd(f2);
    }
    if (op == 14) {  // image_get_resinfo: width, height, depth or layers, levels
        b.Capability(spv::CapabilityImageQuery);
        const Id lod = ia();
        Id r[4];
        auto comp = [&](Id v, u32 k) { return Extract(tI32, v, k); };
        if (storage) {
            const Id s = ImageSize(img, kind, storage);
            r[0] = comp(s, 0), r[1] = comp(s, 1), r[2] = kind == 0 ? CI(1) : comp(s, 2), r[3] = CI(1);
        } else if (t.dim == Dim::D2Ms || t.dim == Dim::D2MsArray) {
            const Id s = ImageSize(img, kind, storage);
            r[0] = comp(s, 0), r[1] = comp(s, 1), r[2] = kind == 4 ? CI(1) : comp(s, 2), r[3] = CI(1);
        } else {
            const bool two = kind == 0 || kind == 3;
            const Id s = R(spv::OpImageQuerySizeLod, two ? tV2I : tV3I, img, lod);
            r[0] = comp(s, 0), r[1] = comp(s, 1), r[2] = two ? CI(1) : comp(s, 2), r[3] = R(spv::OpImageQueryLevels, tI32, img);
        }
        if (sub) r[0] = Extract(tI32, tsize, 0), r[1] = Extract(tI32, tsize, 1);
        u32 n = vdata;
        for (u32 c = 0; c < 4; ++c)
            if (dmask >> c & 1) WV(n++, Ui(r[c]));
        return;
    }
    if (op < 32 || op > 47 || storage) return Fail(std::format("mimg {} at {:#x}{}", op, i.pc, storage ? " on a storage slot" : ""));
    // image_sample*: [bias][compare][coordinates][lod]
    const bool bias = op == 37 || op == 38 || op == 45 || op == 46, compare = op >= 40, lod = op == 36 || op == 44, lz = op == 39 || op == 47;
    const bool clampv = op == 33 || op == 35 || op == 38 || op == 41 || op == 43 || op == 46, grad = op == 34 || op == 35 || op == 42 || op == 43;
    if (grad) return Fail(std::format("mimg {} at {:#x}", op, i.pc));
    const Id bias_v = bias ? fa() : Id(), ref = compare ? fa() : Id();
    if (t.dim == Dim::D2Ms || t.dim == Dim::D2MsArray) {
        // HYPOTHESIS: a sample instruction on a multisampled T# reads sample 0 of the texel under the coordinate.
        if (compare) return Fail(std::format("mimg {} at {:#x} compare on multisampled", op, i.pc));
        const Id x = fa(), y = fa();
        Id c;
        if (unorm || spec[i.smp].unnormalized) {
            c = v2i(R(spv::OpConvertFToS, tI32, x), R(spv::OpConvertFToS, tI32, y));
        } else {
            const Id sz = ImageSize(img, kind, storage);
            const Id sz2 = kind == 4 ? sz : R(spv::OpVectorShuffle, tV2I, sz, sz, 0u, 1u);
            const Id sf = R(spv::OpConvertSToF, tV2F, sz2);
            const Id m = R(spv::OpFMul, tV2F, R(spv::OpCompositeConstruct, tV2F, x, y), sf);
            c = R(spv::OpConvertFToS, tV2I, m);
        }
        if (t.dim == Dim::D2MsArray)
            c = v3i(Extract(tI32, c, 0), Extract(tI32, c, 1), R(spv::OpConvertFToS, tI32, E(tF32, GLSLstd450Round, fa())));
        return write(R(spv::OpImageFetch, vec4t, img, c, 0x40u, CI(0)));
    }
    samplers = true;
    Id coords;
    auto v2 = [&](Id x, Id y) { return R(spv::OpCompositeConstruct, tV2F, x, y); };
    auto v3 = [&](Id x, Id y, Id z) { return R(spv::OpCompositeConstruct, tV3F, x, y, z); };
    auto rnd = [&](Id x) { return E(tF32, GLSLstd450Round, x); };
    switch (t.dim) {
    case Dim::D1: coords = v2(fa(), CF(0.5f)); break;
    case Dim::D1Array: { const Id x = fa(); coords = v3(x, CF(0.5f), rnd(fa())); break; }
    case Dim::D2: { const Id x = fa(); coords = v2(x, fa()); break; }
    case Dim::D2Array: { const Id x = fa(), y = fa(); coords = v3(x, y, rnd(fa())); break; }
    case Dim::D3: { const Id x = fa(), y = fa(); coords = v3(x, y, fa()); break; }
    default: { const Id x = fa(), y = fa(); coords = CubeDir(x, y, fa()); break; }
    }
    const Id level = lod ? fa() : lz ? CF(0.f) : Id(0);
    if (clampv) a_next++;
    if ((unorm || spec[i.smp].unnormalized) && kind <= 2) {
        if (kind <= 1) {
            Id denom;
            if (sub) {
                denom = R(spv::OpConvertSToF, tV2F, tsize);
            } else if (d1) {
                const Id sz = ImageSize(img, kind, storage);
                denom = v2(R(spv::OpConvertSToF, tF32, Extract(tI32, sz, 0)), CF(1.f));
            } else {
                const Id sz = ImageSize(img, kind, storage);
                denom = R(spv::OpConvertSToF, tV2F, kind == 0 ? sz : R(spv::OpVectorShuffle, tV2I, sz, sz, 0u, 1u));
            }
            if (kind == 0) {
                coords = R(spv::OpFDiv, tV2F, coords, denom);
            } else {
                const Id xy = R(spv::OpFDiv, tV2F, R(spv::OpVectorShuffle, tV2F, coords, coords, 0u, 1u), denom);
                coords = v3(Extract(tF32, xy, 0), Extract(tF32, xy, 1), Extract(tF32, coords, 2));
            }
        } else {
            const Id denom = R(spv::OpConvertSToF, tV3F, ImageSize(img, kind, storage));
            coords = R(spv::OpFDiv, tV3F, coords, denom);
        }
    }
    if (sub) {  // T# coordinates -> the image's, clamped to the T#'s edge texels
        const Id ts = R(spv::OpConvertSToF, tV2F, tsize), half = R(spv::OpCompositeConstruct, tV2F, CF(0.5f), CF(0.5f)), onev = R(spv::OpCompositeConstruct, tV2F, CF(1.f), CF(1.f));
        const Id lo = R(spv::OpFDiv, tV2F, half, ts), hi = R(spv::OpFSub, tV2F, onev, R(spv::OpFDiv, tV2F, half, ts));
        const Id sc = R(spv::OpFDiv, tV2F, ts, R(spv::OpConvertSToF, tV2F, isize));
        if (kind == 0) {
            coords = R(spv::OpFMul, tV2F, E(tV2F, GLSLstd450FClamp, coords, lo, hi), sc);
        } else {
            const Id xy = R(spv::OpVectorShuffle, tV2F, coords, coords, 0u, 1u);
            const Id cl = R(spv::OpFMul, tV2F, E(tV2F, GLSLstd450FClamp, xy, lo, hi), sc);
            coords = v3(Extract(tF32, cl, 0), Extract(tF32, cl, 1), Extract(tF32, coords, 2));
        }
    }
    const Id smp = LoadSampler(TableAt(res + i.smp));
    const Id sampled = R(spv::OpSampledImage, b.Type(spv::OpTypeSampledImage, {ImageType(kind, cls, false)}), img, smp);
    const bool frag = ps && !level;
    if (compare) {
        Id e;
        if (frag) {
            e = bias ? R(spv::OpImageSampleDrefImplicitLod, tF32, sampled, coords, ref, 0x1u, bias_v) : R(spv::OpImageSampleDrefImplicitLod, tF32, sampled, coords, ref);
        } else {
            e = R(spv::OpImageSampleDrefExplicitLod, tF32, sampled, coords, ref, 0x2u, level ? level : CF(0.f));
        }
        const Id r = R(spv::OpCompositeConstruct, tV4F, e, CF(0.f), CF(0.f), CF(0.f));
        return write(r);
    }
    Id e;
    if (level) e = R(spv::OpImageSampleExplicitLod, vec4t, sampled, coords, 0x2u, level);
    else if (ps) e = bias ? R(spv::OpImageSampleImplicitLod, vec4t, sampled, coords, 0x1u, bias_v) : R(spv::OpImageSampleImplicitLod, vec4t, sampled, coords);
    else e = R(spv::OpImageSampleExplicitLod, vec4t, sampled, coords, 0x2u, CF(0.f));
    // FORCE_DEGAMMA (shadPS4 ApplyForceDegamma): after the filter, the swizzled texel's RGB goes sRGB -> linear, alpha untouched.
    if (cls == 0 && spec[i.smp].degamma && !t.srgb && (dmask & 7)) {
        Id comps[4];
        for (u32 c = 0; c < 4; ++c) comps[c] = Extract(tF32, e, c);
        // Above the linear piece, a minimax polynomial of the sRGB curve: under 0.15% off, less than one 8-bit step, without the pow.
        static constexpr float k[]{0.000953839544f, 0.0311568489f, 0.540141674f, 0.571112918f, -0.144809437f};
        for (u32 c = 0; c < 3; ++c)
            if (dmask >> c & 1) {
                const Id x = comps[c];
                Id hi = CF(k[4]);
                for (int n = 3; n >= 0; --n) hi = FAdd(FMul(hi, x), CF(k[n]));
                comps[c] = SelF(FGt(x, CF(0.04045f)), hi, FMul(x, CF(1.f / 12.92f)));
            }
        e = R(spv::OpCompositeConstruct, tV4F, comps[0], comps[1], comps[2], comps[3]);
    }
    if (motion_blur && i.pc == 14) {
        // Move the frame-rate compensation ahead of the velocity clamp; preserve the original sample at 60 Hz.
        const Id rate = Fu(TableAt(res + u32(a.slots.size())));
        const Id ex = Extract(tF32, e, 0), ey = Extract(tF32, e, 1);
        const Id x = FSub(ex, CF(0.5f)), y = FSub(ey, CF(0.5f));
        const Id length = E(tF32, GLSLstd450Sqrt, FAdd(FMul(x, x), FMul(y, y)));
        const Id scaled = FMul(FMul(CF(2.f), length), rate), factor = FDiv(rate, FMax(CF(1.f), scaled)), active = FGt(rate, CF(1.f));
        // The two UNORM codes nearest zero carry no reliable sub-step direction; do not amplify their bias when clamping.
        const auto component = [&](Id original, Id v) {
            const Id neutral = Bool(spv::OpFOrdLessThanEqual, FAbs(v), CF(0.5f / 255.f + 1e-7f));
            return SelF(LAnd(active, LNot(LAnd(FGt(scaled, CF(1.f)), neutral))), FAdd(FMul(v, factor), CF(0.5f)), original);
        };
        e = R(spv::OpCompositeConstruct, tV4F, component(ex, x), component(ey, y),
              Extract(tF32, e, 2), Extract(tF32, e, 3));
    }
    write(e);
}

// ---------------------------------------------------------------------------------------------------------------------
// Exports, interpolation, buffers.

void Gen::Exp(const Inst& i) {
    const u32 tgt = i.w0 >> 4 & 0x3F, en = i.w0 & 0xF, compr = i.w0 >> 10 & 1, done = i.w0 >> 11 & 1, vm = i.w0 >> 12 & 1;
    const u32 src[4]{i.w1 & 0xFF, i.w1 >> 8 & 0xFF, i.w1 >> 16 & 0xFF, i.w1 >> 24};
    auto fv = [&](u32 k) { return Fu(LdU(Vr(src[k]))); };
    if (tgt < 8) {
        const u32 fmt = in.col_format >> (4 * tgt) & 0xF;
        const int cls = in.mrt_class[tgt];
        Id c[4]{};
        auto conv = [&](Id f) { return cls == 0 ? f : cls == 1 ? R(spv::OpConvertFToU, tU32, f) : R(spv::OpConvertFToS, tI32, f); };
        if (compr) {
            for (u32 k = 0; k < 2; ++k) {
                if (!(en >> (2 * k) & 3)) continue;
                const Id v = LdU(Vr(src[k]));
                Id x, y;
                if (fmt >= 4 && fmt <= 6) {
                    const Id up = E(tV2F, fmt == 4 ? GLSLstd450UnpackHalf2x16 : fmt == 5 ? GLSLstd450UnpackUnorm2x16 : GLSLstd450UnpackSnorm2x16, v);
                    x = Extract(tF32, up, 0), y = Extract(tF32, up, 1);
                } else if (fmt == 8) {
                    x = R(spv::OpConvertSToF, tF32, R(spv::OpShiftRightArithmetic, tI32, Iu(Shl(v, 16)), CU(16)));
                    y = R(spv::OpConvertSToF, tF32, R(spv::OpShiftRightArithmetic, tI32, Iu(v), CU(16)));
                } else {
                    x = R(spv::OpConvertUToF, tF32, And(v, 0xFFFFu)), y = R(spv::OpConvertUToF, tF32, Shr(v, 16));
                }
                c[2 * k] = conv(x), c[2 * k + 1] = conv(y);
            }
        } else {
            for (u32 k = 0; k < 4; ++k)
                if (en >> k & 1) c[k] = cls == 0 ? fv(k) : cls == 1 ? LdU(Vr(src[k])) : Iu(LdU(Vr(src[k])));
        }
        for (Id& x : c)
            if (!x) x = Zero(cls);
        const u32 m = in.mrt_map[tgt];
        b.Do(spv::OpStore, OutVar(gOut, tgt, V4(cls)), R(spv::OpCompositeConstruct, V4(cls), c[m & 3], c[m >> 2 & 3], c[m >> 4 & 3], c[m >> 6 & 3]));
        mrt_mask |= 1u << tgt;
    } else if (tgt == 8) {
        if (en & 1) {
            if (!gFragDepth) gFragDepth = Builtin(spv::StorageClassOutput, tF32, spv::BuiltInFragDepth, false);
            b.Do(spv::OpStore, gFragDepth, fv(0));
            depth_out = true;
        }
    } else if (tgt == 12) {
        Id c[4];
        for (u32 k = 0; k < 4; ++k) c[k] = en >> k & 1 ? fv(k) : CF(k == 3 ? 1.f : 0.f);
        b.Do(spv::OpStore, PositionPtr(), R(spv::OpCompositeConstruct, tV4F, c[0], c[1], c[2], c[3]));
        if (in.boom) BoomExport(c);
    } else if (tgt >= 32 && tgt < 64) {
        Id c[4];
        for (u32 k = 0; k < 4; ++k) c[k] = en >> k & 1 ? fv(k) : CF(0.f);
        b.Do(spv::OpStore, OutVar(gParam, tgt - 32, tV4F), R(spv::OpCompositeConstruct, tV4F, c[0], c[1], c[2], c[3]));
        params.insert(tgt - 32);
        if (in.boom) BoomParam(c, tgt - 32);
    } else if (tgt != 9 && (tgt < 13 || tgt > 15)) {
        return Fail(std::format("export target {} at {:#x}", tgt, i.pc));
    }
    if (ps && done && vm && p.exec_dynamic && !full) {
        const IfCtx c = IfBegin(LNot(Ld(live, tBool)));
        b.Kill();
        IfEnd(c);
    }
}

void Gen::Vintrp(const Inst& i) {
    const u32 vdst = i.w0 >> 18 & 0xFF, attr = i.w0 >> 10 & 0x3F, chan = i.w0 >> 8 & 3, vsrc = i.w0 & 0xFF;
    // The i/j registers of the interpolation pick the qualifier (PS VGPR layout from SPI_PS_INPUT_ADDR).
    static constexpr u32 sizes[]{2, 2, 2, 3, 2, 2, 2, 1};
    static constexpr u32 qual[]{1, 0, 2, 0, 5, 4, 6, 0, 0};
    const u32 layout = in.ps_input_addr ? in.ps_input_addr : in.ps_input_ena;
    u32 base = 0, mode = 8;
    for (u32 g = 0; g < 8; ++g)
        if (layout >> g & 1) {
            if (vsrc >= base && vsrc < base + sizes[g]) mode = g;
            base += sizes[g];
        }
    if (!attrs.count(attr) || i.op == 2) attrs[attr] = i.op == 2 ? 8 : qual[mode];  // v_interp_mov reads one vertex
    if (i.op == 0) return;  // v_interp_p1: the input already holds the interpolated value
    const u32 cntl = in.ps_input_cntl[attr];
    Id v;
    if (cntl & 0x20) {
        static constexpr float def[]{0.f, 0.f, 1.f, 1.f};
        const u32 d = cntl >> 8 & 3;
        v = CF(chan == 3 ? (d & 1 ? 1.f : 0.f) : def[d]);
    } else {
        auto [it, fresh] = gAttr.try_emplace(attr, 0);
        if (fresh) {
            it->second = b.Global(Ptr(spv::StorageClassInput, tV4F), spv::StorageClassInput);
            b.Decorate(it->second, spv::DecorationLocation, {cntl & 0x1F});
        }
        v = R(spv::OpLoad, tF32, R(spv::OpAccessChain, Ptr(spv::StorageClassInput, tF32), it->second, CU(chan)));
    }
    WV(vdst, Uf(v));
}

void Gen::Buffer(const Inst& i) {
    // Typed (format from the instruction or, for MUBUF, the V#) loads and stores, or raw dword loads (MUBUF 12..15).
    const bool typed = i.enc == MTBUF, raw = !typed && i.op >= 12, store = i.op >= 4 && i.op < 8;
    const u32 n = i.op < 4 ? i.op + 1 : store ? i.op - 3 : !raw ? 0 : i.op == 12 ? 1 : i.op == 13 ? 2 : i.op == 14 ? 4 : i.op == 15 ? 3 : 0;
    if (!n) return Fail(std::format("buffer op {} at {:#x}", i.op, i.pc));
    const u32 offset = i.w0 & 0xFFF, offen = i.w0 >> 12 & 1, idxen = i.w0 >> 13 & 1;
    const u32 vaddr = i.w1 & 0xFF, vdata = i.w1 >> 8 & 0xFF, r = (i.w1 >> 16 & 0x1F) * 4, soff = i.w1 >> 24;
    const Id index = idxen ? LdU(Vr(vaddr)) : CU(0);
    const Id off = Add(Add(CU(offset), U(soff, 0)), offen ? LdU(Vr(vaddr + idxen)) : CU(0));
    const Id d0 = Sg(r), d1 = Sg(r + 1), d2 = Sg(r + 2);
    if (raw) {
        if (n == 1) {
            WV(vdata, RBuf(d0, d1, d2, index, off));
            return;
        }
        const Id stride = And(Shr(d1, 16), 0x3FFFu), strided = Ne(stride, 0);
        const Id oor = Sel(tBool, strided, Bool(spv::OpUGreaterThanEqual, index, d2), Bool(spv::OpUGreaterThanEqual, off, d2));
        const Id vt = n == 2 ? tV2U : n == 3 ? Vec(tU32, 3) : tV4U;
        const Id zero = n == 2 ? R(spv::OpCompositeConstruct, vt, CU(0), CU(0))
                      : n == 3 ? R(spv::OpCompositeConstruct, vt, CU(0), CU(0), CU(0))
                               : R(spv::OpCompositeConstruct, vt, CU(0), CU(0), CU(0), CU(0));
        const Id vec = Guard(vt, LNot(oor), zero, [&] {
            const Id addr = R(spv::OpIAdd, tU64, R(spv::OpIAdd, tU64, Ptr64(d0, d1), R(spv::OpIMul, tU64, W64(index), W64(stride))), W64(off));
            Id v[4];
            LDVec(addr, n, v);
            return n == 2 ? R(spv::OpCompositeConstruct, vt, v[0], v[1])
                 : n == 3 ? R(spv::OpCompositeConstruct, vt, v[0], v[1], v[2])
                          : R(spv::OpCompositeConstruct, vt, v[0], v[1], v[2], v[3]);
        });
        // As RBuf: a strided buffer checks the index, a raw one each dword's offset (the vector loads from the first in range).
        for (u32 c = 0; c < n; ++c) {
            const Id v = Extract(tU32, vec, c);
            WV(vdata + c, c ? Sel(tU32, LOr(strided, Bool(spv::OpULessThan, Add(off, CU(4 * c)), d2)), v, CU(0)) : v);
        }
        return;
    }
    // V# format known from the draw (spec): constants, so the code holds only that format's decode.
    const SlotSpec* vf = !typed && i.tex >= 0 && a.slots.size() + i.tex < spec.size() && spec[a.slots.size() + i.tex].vfmt ? &spec[a.slots.size() + i.tex] : nullptr;
    if (!typed && !vf) return Fail(std::format("format load or store without a known V# format at {:#x}", i.pc));
    if (store) {
        Id c[4];
        for (u32 k = 0; k < n; ++k) c[k] = LdU(Vr(vdata + k));
        return TStore(d0, d1, d2, index, off, typed ? i.w0 >> 19 & 0xF : vf->dfmt, typed ? i.w0 >> 23 & 7 : vf->nfmt, c, n);
    }
    // The V# DST_SEL applies to its own format only: an instruction format reads XYZW as is (shadPS4 lower_buffer_format_to_raw).
    const Id t = TBuf(d0, d1, d2, typed ? 0xFACu : vf->dst_sel, index, off, typed ? i.w0 >> 19 & 0xF : vf->dfmt, typed ? i.w0 >> 23 & 7 : vf->nfmt);
    for (u32 c = 0; c < n; ++c) WV(vdata + c, Extract(tU32, t, c));
}

// A 3-vertex rect list drawn as a 4-vertex strip. Nothing places a draw's vertices in the lanes of one subgroup, so each
// strip vertex computes the three corners itself: the program becomes a function, run once for each (gRect), and main
// emits, in strip order, the right-angle corner, the other two and the fourth (as RectGs).
void Gen::RectOut() {
    if (b.Open()) b.Return();
    const Id prog = b.Helper(b.Type(spv::OpTypeFunction, {tVoid}), tVoid);
    mem_delta = {};
    table_dwords.clear(), table_quads.clear(), image_ids.clear(), sampler_ids.clear();
    image_sizes.clear();
    std::vector<std::array<Id, 3>> q(1 + params.size());  // the position, then each parameter, at the three corners
    Id pxy[3];
    for (u32 k = 0; k < 3; ++k) {
        Ops::St(gRect, CU(k));
        R(spv::OpFunctionCall, tVoid, prog);
        size_t n = 0;
        q[n++][k] = R(spv::OpLoad, tV4F, PositionPtr());
        for (u32 loc : params) q[n++][k] = R(spv::OpLoad, tV4F, OutVar(gParam, loc, tV4F));
        const Id xy = R(spv::OpVectorShuffle, tV2F, q[0][k], q[0][k], 0u, 1u), w = Extract(tF32, q[0][k], 3);
        pxy[k] = R(spv::OpFDiv, tV2F, xy, R(spv::OpCompositeConstruct, tV2F, w, w));
    }
    Id c = CU(0), best = CF(3.4e38f);
    for (u32 k = 0; k < 3; ++k) {
        const Id d = FAbs(R(spv::OpDot, tF32, R(spv::OpFSub, tV2F, pxy[(k + 1) % 3], pxy[k]), R(spv::OpFSub, tV2F, pxy[(k + 2) % 3], pxy[k])));
        const Id lt = FLt(d, best);
        best = SelF(lt, d, best), c = SelU(lt, CU(k), c);
    }
    const Id ia = R(spv::OpUMod, tU32, Add(c, CU(1)), CU(3)), ib = R(spv::OpUMod, tU32, Add(c, CU(2)), CU(3)), v = And(VertexIndex(), 3u);
    auto strip = [&](const std::array<Id, 3>& x) {
        auto pick = [&](Id idx) { return Sel(tV4F, Eq(idx, 0), x[0], Sel(tV4F, Eq(idx, 1), x[1], x[2])); };
        const Id qc = pick(c), qa = pick(ia), qb = pick(ib), fourth = R(spv::OpFSub, tV4F, R(spv::OpFAdd, tV4F, qa, qb), qc);
        return Sel(tV4F, Eq(v, 0), qc, Sel(tV4F, Eq(v, 1), qa, Sel(tV4F, Eq(v, 2), qb, fourth)));
    };
    size_t n = 0;
    b.Do(spv::OpStore, PositionPtr(), strip(q[n++]));
    for (u32 loc : params) b.Do(spv::OpStore, OutVar(gParam, loc, tV4F), strip(q[n++]));
}

// ---------------------------------------------------------------------------------------------------------------------
// Program structure.

void Gen::Op(const Inst& i, bool last, bool loop) {
    boom_pc = i.pc;
    switch (i.enc) {
    case SOP2: case SOPK: case SOP1: case SOPC: return Salu(i);
    case SMRD: return Smrd(i);
    case VALU: return Valu(i);
    case MIMG: return Mimg(i);
    case EXP: return Exp(i);
    case VINTRP: return Vintrp(i);
    case SOPP:
        if (i.op == 1) {
            if (loop) {
                St(pc_var, CU(0xFFFFFFFFu));
                b.Br(switch_merge);
            } else if (!last) {
                b.Return();
            }
        } else if (IsBranch(i)) {
            const IfCtx c = IfBegin(Cond(i.op));
            St(pc_var, CU(Target(i)));
            b.Br(switch_merge);
            IfEnd(c);
        } else if (i.op == 10 && in.stage == Stage::Compute) {
            b.Do(spv::OpControlBarrier, CU(2), CU(2), CU(0x108));
        }
        return;
    case DS:
        if (i.op == 53) return WV(i.w1 >> 24, Swizzle(LdU(Vr(i.w1 & 0xFF)), i.w0 & 0xFFFF));  // source in the addr field
        return Fail(std::format("ds {} at {:#x}", i.op, i.pc));
    case MTBUF: case MUBUF: return Buffer(i);
    default: return Fail(std::format("encoding {} at {:#x}", int(i.enc), i.pc));
    }
}

void Gen::OneInst(const Inst& i, bool last, bool loop) {
    if (!lane_masks.empty() && (p.labels.count(i.pc) || WritesExec(i) || IsBranch(i))) lane_masks.clear();
    touched.clear(), lane_set.clear();
    Op(i, last, loop);
    // A pair stored by any other path (scalar loads, moves, lane reads) holds something else now.
    if (!lane_masks.empty())
        std::erase_if(lane_masks, [&](const auto& e) {
            if (std::find(lane_set.begin(), lane_set.end(), e.first) != lane_set.end()) return false;
            const auto reg = [&](u32 r) { return r == 106 ? vcc_lo : r == 107 ? vcc_hi : sreg.contains(r) ? sreg.at(r) : Id{}; };
            const Id lo = reg(e.first), hi = reg(e.first + 1);
            return (lo && wrote(lo)) || (hi && wrote(hi));
        });
    // The companion change removes that compensation from the blur strength in the same draw.
    if (motion_blur && i.pc == 16)
        St(Sr(0), Uf(FDiv(Fu(LdU(Sr(0))), Fu(TableAt(res + u32(a.slots.size()))))));
    auto w = [&](u32 r) {
        const auto it = sreg.find(r);
        return it != sreg.end() && wrote(it->second);
    };
    if (saved != ~0u && (w(saved) || w(saved + 1)) && !(i.enc == SOP1 && i.op == 36 && i.dst == saved)) saved = ~0u;
    if (entry != ~0u && (w(entry) || w(entry + 1))) entry = ~0u;
    if (wrote(exec_lo)) {
        // Not from `entry` inside a loop: the body may overwrite the copy after the restore, stale on the next pass.
        const bool restore = !loop && i.enc == SOP1 && i.op == 4 && i.dst == 126 && (i.src[0] == saved || (i.src[0] == entry && exit == ~size_t(0)));
        const bool wqm = full && i.enc == SOP1 && i.op == 10 && i.dst == 126 && i.src[0] == 126;  // whole quads of all-ones
        // The copy stays all-ones through the EXEC writes of the if/else it brackets (s_andn2_b64 exec, saved, exec).
        if (!loop && full && i.enc == SOP1 && i.op == 36 && i.dst < 104) saved = i.dst;
        full = restore || wqm;
        SetLive();
    } else if (!loop && full && i.enc == SOP1 && i.op == 4 && i.src[0] == 126 && i.dst < 104) {
        entry = i.dst;  // copy of the all-ones EXEC
    }
}

// Structured control flow for [from, to): a backward branch closes a do-while from its target, a forward branch skips an
// if (with an else when the skipped part ends in an unconditional forward branch), a branch to just after the innermost
// loop breaks out of it. Crossing branches return false.
bool Gen::Block(size_t from, size_t to, bool emit) {
    const auto& I = p.insts;
    for (size_t k = from; k < to && error.empty();) {
        size_t j = to;
        for (size_t x = to; x-- > k;)
            if (IsBranch(I[x]) && index[Target(I[x])] == k) {
                j = x;
                break;
            }
        if (j < to) {
            Id header, merge, cont;
            const size_t outer = exit;
            const Id outer_merge = exit_merge;
            if (emit) {
                Forget(k, j);
                header = b.NewId(), merge = b.NewId(), cont = b.NewId();
                const Id body = b.NewId();
                b.Br(header);
                b.Label(header);
                b.Do(spv::OpLoopMerge, merge, cont, 0u);
                b.Br(body);
                b.Label(body);
                exit_merge = merge;
            }
            exit = j + 1;
            if (!Block(k, j, emit)) return false;
            exit = outer;
            if (emit) {
                exit_merge = outer_merge;
                Forget(k, j);
                if (b.Open()) b.Br(cont);
                b.Label(cont);
                b.BrCond(Cond(I[j].op), header, merge);
                b.Label(merge);
            }
            k = j + 1;
            continue;
        }
        const Inst& i = I[k];
        if (!IsBranch(i)) {
            if (emit) OneInst(i, k + 1 == I.size(), false);
            ++k;
            continue;
        }
        const size_t t = index[Target(i)];
        if (t == exit) {  // out of the innermost loop (a while loop: test at the top, unconditional branch back)
            if (emit) {
                if (i.op == 2) {
                    b.Br(exit_merge);
                } else {
                    const IfCtx c = IfBegin(Cond(i.op));
                    b.Br(exit_merge);
                    IfEnd(c);
                }
            }
            ++k;
            continue;
        }
        if (t <= k || t > to) return false;
        size_t mid = t, end = t;
        if (t > k + 1 && I[t - 1].enc == SOPP && I[t - 1].op == 2) {
            const size_t x = index[Target(I[t - 1])];
            if (x > t && x <= to) mid = t - 1, end = x;
        }
        IfCtx c{};
        if (emit) {
            c = IfBegin(i.op == 2 ? CB(false) : LNot(Cond(i.op)), end != t);
            Forget(k + 1, end);
        }
        if (!Block(k + 1, mid, emit)) return false;
        if (end != t) {
            if (emit) {
                IfElse(c);
                Forget(k + 1, end);
            }
            if (!Block(t, end, emit)) return false;
        }
        if (emit) {
            Forget(k + 1, end);
            IfEnd(c);
        }
        k = end;
    }
    return true;
}

// Alpha MIN/MAX on RGBA8: one ordered update for each covered sample, after the guest pixel shader.
void Gen::BlendAlpha() {
    const u32 samples = in.blend_interlock & 0xff;
    b.Extension("SPV_EXT_fragment_shader_interlock"), b.Extension("SPV_KHR_post_depth_coverage");
    b.Capability(spv::CapabilityFragmentShaderPixelInterlockEXT), b.Capability(spv::CapabilitySampleMaskPostDepthCoverage);
    if (samples > 1) b.Capability(spv::CapabilityStorageImageMultisample);
    b.Mode(spv::ExecutionModeEarlyFragmentTests), b.Mode(spv::ExecutionModePostDepthCoverage), b.Mode(spv::ExecutionModePixelInterlockOrderedEXT);
    const Id img_type = b.Type(spv::OpTypeImage, {tU32, u32(spv::Dim2D), 0, 0, u32(samples > 1), 2, u32(spv::ImageFormatRgba8ui)});
    const Id arr = b.Type(spv::OpTypeRuntimeArray, {img_type});
    const Id g = b.Global(Ptr(spv::StorageClassUniformConstant, arr), spv::StorageClassUniformConstant);
    b.Decorate(g, spv::DecorationDescriptorSet, {0}), b.Decorate(g, spv::DecorationBinding, {20});
    b.Decorate(g, spv::DecorationCoherent), b.Capability(spv::CapabilityRuntimeDescriptorArray), b.Capability(spv::CapabilityStorageImageArrayDynamicIndexing);
    const Id index = TableAt(res + u32(a.slots.size()) + u32(motion_blur));
    const Id img = R(spv::OpLoad, img_type, R(spv::OpAccessChain, Ptr(spv::StorageClassUniformConstant, img_type), g, index));
    if (!gFragCoord) gFragCoord = Builtin(spv::StorageClassInput, tV4F, spv::BuiltInFragCoord, false);
    if (!gSampleMask) gSampleMask = Builtin(spv::StorageClassInput, b.Type(spv::OpTypeArray, {tU32, CU(1)}), spv::BuiltInSampleMask, true);
    const Id fc = Ld(gFragCoord, tV4F), x = R(spv::OpConvertFToS, tI32, Extract(tF32, fc, 0)), y = R(spv::OpConvertFToS, tI32, Extract(tF32, fc, 1));
    const Id pos = R(spv::OpCompositeConstruct, tV2I, x, y), mask = LdU(R(spv::OpAccessChain, pInU, gSampleMask, CU(0)));
    const Id src = E(tF32, GLSLstd450FClamp, Extract(tF32, Ld(gOut.at(0), tV4F), 3), CF(0.f), CF(1.f));
    const Id src2 = FMul(src, src);
    b.Do(spv::OpBeginInvocationInterlockEXT);
    for (u32 s = 0; s < samples; ++s) {
        const IfCtx covered = IfBegin(Ne(And(mask, 1u << s), 0));
        const Id rgba = samples > 1 ? R(spv::OpImageRead, tV4U, img, pos, u32(spv::ImageOperandsSampleMask), CI(s)) : R(spv::OpImageRead, tV4U, img, pos);
        const Id dst = FDiv(R(spv::OpConvertUToF, tF32, Extract(tU32, rgba, 3)), CF(255.f));
        const Id alpha = E(tF32, (in.blend_interlock & 0x100) ? GLSLstd450FMax : GLSLstd450FMin, src2, FMul(dst, dst));
        const Id quant = R(spv::OpConvertFToU, tU32, E(tF32, GLSLstd450RoundEven, FMul(alpha, CF(255.f))));
        const Id result = R(spv::OpCompositeInsert, tV4U, quant, rgba, 3u);
        if (samples > 1) b.Do(spv::OpImageWrite, img, pos, result, u32(spv::ImageOperandsSampleMask), CI(s));
        else b.Do(spv::OpImageWrite, img, pos, result);
        IfEnd(covered);
    }
    b.Do(spv::OpEndInvocationInterlockEXT);
}

bool Gen::Run(std::vector<u32>& words, u64& out_mask) {
    b.Start();  // (in.probe, the vertex probe of the GLSL path, is not translated: such a program is the plain one; in.boom is)
    const auto& I = p.insts;
    for (size_t k = 0; k < I.size(); ++k) index[I[k].pc] = k;

    // Prologue: EXEC is the subgroup, the user data registers come from the push constants, the stage's inputs from builtins.
    if (p.exec_dynamic || std::any_of(I.begin(), I.end(), [](const Inst& i) {
            return (i.enc == SOPP && (i.op == 8 || i.op == 9)) || (i.enc == VALU && i.op == 386) ||
                   std::any_of(std::begin(i.src), std::end(i.src), [](u32 s) { return s == 126 || s == 127 || s == 252; });
        })) {
        const Id bl = Ballot(CB(true));
        St(exec_lo, Extract(tU32, bl, 0)), St(exec_hi, SelU(Wide(), Extract(tU32, bl, 1), CU(0)));
    }
    for (u32 r = 0; r < p.user_sgprs; ++r) St(Sr(r), PushU(ps ? 1 : 0, r));
    u32 sgpr = p.user_sgprs;
    switch (in.stage) {
    case Stage::Vertex: {
        if (in.rect) gRect = b.Global(Ptr(spv::StorageClassPrivate, tU32), spv::StorageClassPrivate);  // the corner this run computes (RectOut)
        St(Vr(0), in.rect ? LdU(gRect) : VertexIndex());
        const u32 comp = in.rsrc1 >> 24 & 3;
        if (comp) {
            const Id ii = InstanceIndex();
            for (u32 k = 1; k <= comp; ++k) St(Vr(k), ii);
        }
        break;
    }
    case Stage::Pixel: {
        static constexpr u32 sizes[]{2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        const u32 layout = in.ps_input_addr ? in.ps_input_addr : in.ps_input_ena;
        Id frag;
        auto fc = [&] {
            if (!frag) frag = R(spv::OpLoad, tV4F, gFragCoord ? gFragCoord : (gFragCoord = Builtin(spv::StorageClassInput, tV4F, spv::BuiltInFragCoord, false)));
            return frag;
        };
        u32 v = 0;
        for (u32 g = 0; g < 16; ++g) {
            if (!(layout >> g & 1)) continue;
            for (u32 k = 0; k < sizes[g]; ++k) {
                Id val;
                switch (g) {
                case 8: case 9: case 10: case 11: val = Uf(Extract(tF32, fc(), g - 8)); break;
                case 12:
                    if (!gFront) gFront = Builtin(spv::StorageClassInput, tBool, spv::BuiltInFrontFacing, false);
                    val = Uf(SelF(R(spv::OpLoad, tBool, gFront), CF(1.f), CF(-1.f)));
                    break;
                case 14:
                    if (!gSampleMask) gSampleMask = Builtin(spv::StorageClassInput, b.Type(spv::OpTypeArray, {tU32, CU(1)}), spv::BuiltInSampleMask, true);
                    val = R(spv::OpLoad, tU32, R(spv::OpAccessChain, pInU, gSampleMask, CU(0)));
                    break;
                case 15: val = Or(R(spv::OpConvertFToU, tU32, Extract(tF32, fc(), 0)), Shl(R(spv::OpConvertFToU, tU32, Extract(tF32, fc(), 1)), 16)); break;
                default: val = CU(0);
                }
                St(Vr(v + k), val);
            }
            v += sizes[g];
        }
        St(Sr(sgpr), CU(0));  // primitive mask
        break;
    }
    case Stage::Compute: {
        auto id3 = [&](Id& g, spv::BuiltIn bi) {
            if (!g) g = Builtin(spv::StorageClassInput, Vec(tU32, 3), bi, false);
            return R(spv::OpLoad, Vec(tU32, 3), g);
        };
        for (u32 k = 0; k < 3; ++k)
            if (in.rsrc2 >> (7 + k) & 1) St(Sr(sgpr++), Extract(tU32, id3(gWorkgroup, spv::BuiltInWorkgroupId), k));
        if (in.rsrc2 >> 10 & 1) St(Sr(sgpr++), CU(0));
        const u32 comp = in.rsrc2 >> 11 & 3;
        for (u32 k = 0; k <= comp; ++k) St(Vr(k), Extract(tU32, id3(gLocal, spv::BuiltInLocalInvocationId), k));
        break;
    }
    }
    touched.clear();

    // Structured when the branches nest (SSA-friendly for the optimizers); else a pc dispatch loop.
    const bool structured = p.labels.empty() || Block(0, I.size(), false);
    if (structured) {
        if (p.labels.empty()) {
            for (size_t k = 0; k < I.size() && error.empty(); ++k) OneInst(I[k], k + 1 == I.size(), false);
        } else {
            Block(0, I.size(), true);
        }
    } else {
        full = false, saved = entry = ~0u;  // pc order is not program order
        pc_var = Var(pFnU, CU(0));
        std::map<u32, Id> cases{{0, b.NewId()}};
        for (u32 l : p.labels) cases.try_emplace(l, b.NewId());
        const Id header = b.NewId(), body = b.NewId(), cont = b.NewId(), merge = b.NewId(), dflt = b.NewId();
        switch_merge = b.NewId();
        b.Br(header);
        b.Label(header);
        b.Do(spv::OpLoopMerge, merge, cont, 0u);
        b.Br(body);
        b.Label(body);
        const Id sel = LdU(pc_var);
        b.Do(spv::OpSelectionMerge, switch_merge, 0u);
        b.Switch(sel, dflt, cases);
        b.Label(cases[0]);
        for (size_t k = 0; k < I.size() && error.empty(); ++k) {
            if (const auto it = cases.find(I[k].pc); it != cases.end() && I[k].pc) {
                if (b.Open()) b.Br(it->second);
                b.Label(it->second);
            }
            OneInst(I[k], k + 1 == I.size(), true);
        }
        St(pc_var, CU(0xFFFFFFFFu));
        b.Br(switch_merge);
        b.Label(dflt);
        St(pc_var, CU(0xFFFFFFFFu));
        b.Br(switch_merge);
        b.Label(switch_merge);
        b.BrCond(Bool(spv::OpIEqual, LdU(pc_var), CU(0xFFFFFFFFu)), merge, cont);
        b.Label(cont);
        b.Br(header);
        b.Label(merge);
    }
    if (!error.empty()) return false;
    if (in.rect) RectOut();
    if (in.blend_interlock && b.Open()) {
        if (!gOut.contains(0)) return error = "interlock shader has no color export", false;
        BlendAlpha();
    }
    if (b.Open()) b.Return();

    // Module: capabilities and modes of the stage, interpolation qualifiers of the inputs.
    for (int k = 0; k < 3; ++k)
        for (int c = 0; c < 3; ++c)
            if (gImg[k][c].v && !storage_read[k][c]) b.Decorate(gImg[k][c], spv::DecorationNonReadable);
    if (!in.no_float_controls) {
        b.Capability(spv::CapabilitySignedZeroInfNanPreserve);  // GCN floats keep inf, NaN and -0 (GOW3 multiplies by +inf literals)
        b.Mode(spv::ExecutionModeSignedZeroInfNanPreserve, {32});
    }
    spv::ExecutionModel model = spv::ExecutionModelVertex;
    switch (in.stage) {
    case Stage::Vertex:
        for (u32 n : params) out_mask |= 1ull << n;
        break;
    case Stage::Pixel:
        model = spv::ExecutionModelFragment;
        b.Mode(spv::ExecutionModeOriginUpperLeft);
        if (depth_out) b.Mode(spv::ExecutionModeDepthReplacing);
        for (auto& [attr, var] : gAttr) {
            const u32 q = in.ps_input_cntl[attr] >> 10 & 1 ? 8u : attrs[attr];
            if (q & 8) b.Decorate(var, spv::DecorationFlat);
            if (q & 1) b.Decorate(var, spv::DecorationSample), b.Capability(spv::CapabilitySampleRateShading);
            if (q & 2) b.Decorate(var, spv::DecorationCentroid);
            if (q & 4) b.Decorate(var, spv::DecorationNoPerspective);
        }
        break;
    case Stage::Compute:
        model = spv::ExecutionModelGLCompute;
        b.Mode(spv::ExecutionModeLocalSize, {std::max(1u, in.threads[0]), std::max(1u, in.threads[1]), std::max(1u, in.threads[2])});
        break;
    }
    const Id main = b.NewId();
    words = b.Finish(model, main, b.Type(spv::OpTypeFunction, {tVoid}), tVoid);
    return true;
}
}  // namespace

bool EmitSpirv(const Analysis& a, const Input& in, std::span<const SlotSpec> spec, std::vector<std::uint32_t>& words, std::uint64_t& out_mask,
              std::string& error, bool full_opt) {
    if (!a.error.empty() || !a.program) return error = a.error.empty() ? "no program" : a.error, false;
    Gen g(*a.program, a, in, spec);
    out_mask = 0;
    if (!g.Run(words, out_mask) || !g.error.empty()) return error = g.error.empty() ? "emit failed" : g.error, false;
    // Passes are consumed by Run; each translation needs its own optimizer and pass registrations.
    spvtools::Optimizer opt(SPV_ENV_VULKAN_1_3);
    opt.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char*) {});
    if (full_opt) opt.RegisterPerformancePasses();
    else {
        opt.RegisterPass(spvtools::CreateLocalAccessChainConvertPass());
        opt.RegisterPass(spvtools::CreateLocalSingleBlockLoadStoreElimPass());
        opt.RegisterPass(spvtools::CreateLocalSingleStoreElimPass());
        opt.RegisterPass(spvtools::CreateSSARewritePass());
        opt.RegisterPass(spvtools::CreateAggressiveDCEPass());
    }
    spvtools::ValidatorOptions validator;
    validator.SetScalarBlockLayout(true);
    spvtools::OptimizerOptions options;
    options.set_validator_options(validator);
    std::vector<u32> optimized;
    if (opt.Run(words.data(), words.size(), &optimized, options)) words.swap(optimized);
    return true;
}
}  // namespace Gcn
