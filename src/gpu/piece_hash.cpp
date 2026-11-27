// SPDX-License-Identifier: GPL-2.0-or-later
#include "gpu/piece_hash.h"
#include <algorithm>
#include <cstring>
#include <immintrin.h>
#ifdef _WIN32
#include <intrin.h>
#endif
#include <xxhash.h>

namespace Gpu::PieceHash {
namespace {
using u64 = std::uint64_t;

u64 Mix64(u64 x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}

template <bool Copy>
__attribute__((target("aes"))) void Block(const unsigned char* p, unsigned char* q,
                                          __m128i& a, __m128i& b, __m128i& c, __m128i& d) {
    const auto w = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const auto x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16));
    const auto y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 32));
    const auto z = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 48));
    if constexpr (Copy) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(q), w);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(q + 16), x);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(q + 32), y);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(q + 48), z);
    }
    a = _mm_aesenc_si128(a, w); b = _mm_aesenc_si128(b, x);
    c = _mm_aesenc_si128(c, y); d = _mm_aesenc_si128(d, z);
}

template <bool Copy, bool Avalanche = true>
__attribute__((target("aes"))) u64 Aes(void* dst, const void* data, std::size_t size) {
    const auto* src = static_cast<const unsigned char*>(data);
    auto* out = static_cast<unsigned char*>(dst);
    __m128i a = _mm_set_epi64x(0x123456789abcdef0ull, 0xfedcba9876543210ull);
    __m128i b = _mm_set_epi64x(0x243f6a8885a308d3ull, 0x13198a2e03707344ull);
    __m128i c = _mm_set_epi64x(0xa4093822299f31d0ull, 0x082efa98ec4e6c89ull);
    __m128i d = _mm_set_epi64x(0x452821e638d01377ull, 0xbe5466cf34e90c6cull);
    std::size_t i = 0;
    for (; size - i >= 64; i += 64) Block<Copy>(src + i, Copy ? out + i : nullptr, a, b, c, d);
    if (i != size) {
        alignas(16) unsigned char tail[64]{};
        std::memcpy(tail, src + i, size - i);
        if constexpr (Copy) std::memcpy(out + i, tail, size - i);
        Block<Copy>(tail, Copy ? tail : nullptr, a, b, c, d);
    }
    a = _mm_aesenc_si128(a, b); c = _mm_aesenc_si128(c, d); a = _mm_aesenc_si128(a, c);
    const u64 x = static_cast<u64>(_mm_cvtsi128_si64(a)) ^ static_cast<u64>(_mm_extract_epi64(a, 1)) ^ size;
    if constexpr (Avalanche) return Mix64(x) | 1;
    else return x;
}

u64 AesHash(const void* src, std::size_t n) { return Aes<false>(nullptr, src, n); }
u64 AesCopy(void* dst, const void* src, std::size_t n) { return Aes<true>(dst, src, n); }

u64 XxhHash(const void* src, std::size_t n) { return (Stable64(src, n) ^ n) | 1; }
u64 XxhCopy(void* dst, const void* src, std::size_t n) {
    if (n <= 4096) {
        alignas(64) unsigned char tmp[4096];
        std::memcpy(tmp, src, n);
        std::memcpy(dst, tmp, n);
        return XxhHash(tmp, n);
    }
    std::memcpy(dst, src, n);
    return XxhHash(src, n);
}

template <u64 (*CopyFn)(void*, const void*, std::size_t)>
__attribute__((always_inline)) inline u64 BlockCopyLoop(void* dst, const void* data, std::size_t size, u64* hashes, unsigned want) {
    auto* out = static_cast<unsigned char*>(dst);
    const auto* src = static_cast<const unsigned char*>(data);
    std::size_t count = 0;
    for (std::size_t i = 0; i < size; i += kMemoryBlock, ++count) {
        if (!(want >> count & 1)) continue;
        const std::size_t n = std::min(kMemoryBlock, size - i);
        // The final avalanche is a bijection before truncation; the page root needs it only once.
        hashes[count] = CopyFn(out + i, src + i, n);
    }
    return detail::Rapid(hashes, count * sizeof(u64), size) | 1;
}

__attribute__((target("aes"))) u64 BlockCopyAes(void* dst, const void* data, std::size_t size, u64* hashes, unsigned want) {
    return BlockCopyLoop<Aes<true, false>>(dst, data, size, hashes, want);
}

u64 BlockCopyXxh(void* dst, const void* data, std::size_t size, u64* hashes, unsigned want) {
    return BlockCopyLoop<XxhCopy>(dst, data, size, hashes, want);
}

struct Dispatch {
    decltype(&XxhHash) hash;
    decltype(&XxhCopy) copy;
    bool aes;
};
const Dispatch& Selected() {
    static const Dispatch d = HasAes() ? Dispatch{AesHash, AesCopy, true} : Dispatch{XxhHash, XxhCopy, false};
    return d;
}
} // namespace

bool HasAes() {
#ifdef _WIN32
    int regs[4];
    __cpuid(regs, 1);
    return (regs[2] & (1 << 25)) != 0;
#else
    return __builtin_cpu_supports("aes");
#endif
}
bool UsesAes() { return Selected().aes; }
std::uint64_t Hash(const void* data, std::size_t size) { return Selected().hash(data, size); }
std::uint64_t Copy(void* dst, const void* data, std::size_t size) { return Selected().copy(dst, data, size); }
std::uint64_t CopyBlocks(void* dst, const void* data, std::size_t size, std::uint64_t* hashes, unsigned want) {
    return UsesAes() ? BlockCopyAes(dst, data, size, hashes, want) : BlockCopyXxh(dst, data, size, hashes, want);
}
std::uint64_t ShaderCode(const void* p, std::size_t n) { return UsesAes() ? Hash(p, n) : Stable64(p, n); }
std::uint64_t ShaderContent(const void* p, std::size_t n, std::uint64_t seed) {
    return UsesAes() ? Mix64(Hash(p, n) ^ seed) : Stable64Seeded(p, n, seed);
}
} // namespace Gpu::PieceHash
