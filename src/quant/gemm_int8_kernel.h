#pragma once
// Internal: packing and micro-kernels for the int8 GEMM. Same panel structure
// as the float kernel (gemm_kernel.h); the difference is that the hardware
// dot-product instructions consume 4 consecutive k values per lane, so both
// packed operands are laid out in groups of 4 k:
//   packed B strip: [k/4][NR][4]   one group = NR columns x 4 k, NR*4 bytes
//   packed A strip: [k/4][MR][4]   one group = MR rows x 4 k, 4 bytes per row
// and the micro-kernel does, per group, MR*NR*4 multiply-adds from NR/4 + MR/4
// vector loads. kc is padded to a multiple of 4 with zeros, which is exact.
//
// Backends:
//   NEON dotprod   vdotq_laneq_s32: signed x signed, 16 MACs per instruction.
//   AVX-VNNI       vpdpbusd: UNSIGNED x signed, 32 MACs per instruction. The
//                  activations are stored as a+128 (unsigned) and the kernel
//                  starts each accumulator at -128 * sum_k(B[k][n]), which
//                  cancels the offset exactly: sum((a+128) b) - 128 sum(b).
//   AVX2           vpmaddwd on sign-extended int16: 16 MACs per instruction.
//                  Each 32-bit lane holds the sum of 2 adjacent k products, so
//                  a column's result is spread over 2 lanes until the end.
//   scalar         plain loops over the same packed layout.

#include <algorithm>
#include <cstdint>
#include <cstring>

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define TINYINFER_I8_NEON_DOT 1
#elif defined(__AVX512VNNI__) && defined(__AVX512VL__)
#include <immintrin.h>
#define TINYINFER_I8_VNNI 1
#define TINYINFER_DPBUSD(acc, a, b) _mm256_dpbusd_epi32((acc), (a), (b))
#elif defined(__AVXVNNI__)
#include <immintrin.h>
#define TINYINFER_I8_VNNI 1
#define TINYINFER_DPBUSD(acc, a, b) _mm256_dpbusd_avx_epi32((acc), (a), (b))
#elif defined(__AVX2__)
#include <immintrin.h>
#define TINYINFER_I8_AVX2 1
#else
#define TINYINFER_I8_SCALAR 1
#endif

namespace tinyinfer {
namespace kernel_i8 {

constexpr int KP = 4;  // k values per dot-product lane

#if defined(TINYINFER_I8_NEON_DOT)
constexpr int MR = 8, NR = 12;
using APacked = int8_t;
constexpr const char* kBackendName = "NEON dotprod (sdot), 8x12 micro-kernel";
#elif defined(TINYINFER_I8_VNNI)
constexpr int MR = 6, NR = 16;
using APacked = uint8_t;
constexpr const char* kBackendName = "AVX-VNNI (vpdpbusd), 6x16 micro-kernel";
#elif defined(TINYINFER_I8_AVX2)
constexpr int MR = 6, NR = 8;
using APacked = int16_t;
constexpr const char* kBackendName = "AVX2 (vpmaddwd on int16), 6x8 micro-kernel";
#else
constexpr int MR = 4, NR = 8;
using APacked = int8_t;
constexpr const char* kBackendName = "scalar fallback, 4x8 micro-kernel";
#endif

constexpr int MC = 72;
constexpr int KC = 256;  // multiple of KP
constexpr int NC = 3072;

inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

// B[kc][nc] (stride ldb) -> strips of NR columns in [kc/4][NR][4] order, zero
// padded in both k and n. colsum[strip*NR + t] = sum over k of the packed
// column (used only by the VNNI backend, computed for all for simplicity).
inline void pack_b(int kc, int nc, const int8_t* B, int ldb, int8_t* out, int32_t* colsum) {
    const int groups = round_up(kc, KP) / KP;
    for (int j = 0; j < nc; j += NR) {
        const int w = std::min(NR, nc - j);
        int8_t* strip = out + static_cast<size_t>(j / NR) * groups * NR * KP;
        int32_t* cs = colsum + j;
        for (int t = 0; t < NR; ++t) cs[t] = 0;
        for (int g = 0; g < groups; ++g) {
            int8_t* dst = strip + static_cast<size_t>(g) * NR * KP;
            for (int t = 0; t < NR; ++t) {
                for (int q = 0; q < KP; ++q) {
                    const int k = g * KP + q;
                    const int8_t v = (t < w && k < kc) ? B[static_cast<size_t>(k) * ldb + j + t] : 0;
                    dst[t * KP + q] = v;
                    cs[t] += v;
                }
            }
        }
    }
}

// A[mc][kc] (stride lda) -> strips of MR rows in [kc/4][MR][4] order, zero padded.
// The 4 k values of one group are consecutive bytes of a row of A, so for the
// 1-byte packed types a group is one 32-bit copy. (The first version copied
// byte by byte with a bounds test per byte and was 41% of INT8 inference time.)
inline void pack_a(int mc, int kc, const int8_t* A, int lda, APacked* out) {
    const int groups = round_up(kc, KP) / KP;
    const int full_groups = kc / KP;
    for (int i = 0; i < mc; i += MR) {
        const int h = std::min(MR, mc - i);
        APacked* strip = out + static_cast<size_t>(i / MR) * groups * MR * KP;
        for (int r = 0; r < MR; ++r) {
            APacked* dst = strip + static_cast<size_t>(r) * KP;  // then + g * MR * KP per group
            if (r >= h) {
                for (int g = 0; g < groups; ++g) {
                    for (int q = 0; q < KP; ++q) dst[static_cast<size_t>(g) * MR * KP + q] = 0;
                }
                continue;
            }
            const int8_t* src = A + static_cast<size_t>(i + r) * lda;
            if constexpr (sizeof(APacked) == 1) {
                for (int g = 0; g < full_groups; ++g) {
                    uint32_t v;
                    std::memcpy(&v, src + static_cast<size_t>(g) * KP, 4);
#if defined(TINYINFER_I8_VNNI)
                    v ^= 0x80808080u;  // s8 + 128 == s8 ^ 0x80, byte-wise
#endif
                    std::memcpy(dst + static_cast<size_t>(g) * MR * KP, &v, 4);
                }
            } else {
                for (int g = 0; g < full_groups; ++g) {
                    for (int q = 0; q < KP; ++q) {
                        dst[static_cast<size_t>(g) * MR * KP + q] = static_cast<APacked>(src[static_cast<size_t>(g) * KP + q]);
                    }
                }
            }
            if (full_groups < groups) {  // tail group: kc not a multiple of 4
                for (int q = 0; q < KP; ++q) {
                    const int k = full_groups * KP + q;
                    const int8_t v = k < kc ? src[k] : 0;
#if defined(TINYINFER_I8_VNNI)
                    dst[static_cast<size_t>(full_groups) * MR * KP + q] = static_cast<uint8_t>(static_cast<int>(v) + 128);
#else
                    dst[static_cast<size_t>(full_groups) * MR * KP + q] = static_cast<APacked>(v);
#endif
                }
            }
        }
    }
}

// C[MR][NR] (stride ldc) = (accumulate ? C : 0) + A strip x B strip over `groups` k-groups.
#if defined(TINYINFER_I8_NEON_DOT)
inline void micro_kernel(int groups, const APacked* Ap, const int8_t* Bp, const int32_t* /*colsum*/, int32_t* C, int ldc,
                         bool accumulate) {
    int32x4_t c[MR][3];
    for (int r = 0; r < MR; ++r) {
        for (int j = 0; j < 3; ++j) {
            c[r][j] = accumulate ? vld1q_s32(C + static_cast<size_t>(r) * ldc + 4 * j) : vdupq_n_s32(0);
        }
    }
    for (int g = 0; g < groups; ++g) {
        const int8_t* b = Bp + static_cast<size_t>(g) * NR * KP;
        const int8x16_t b0 = vld1q_s8(b), b1 = vld1q_s8(b + 16), b2 = vld1q_s8(b + 32);
        const int8_t* a = Ap + static_cast<size_t>(g) * MR * KP;
        const int8x16_t a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16);
        // vdotq_laneq_s32(acc, x, y, lane): acc[i] += dot(x[4i..4i+3], y[4*lane..4*lane+3]).
#define TINYINFER_SDOT_ROW(r, av, lane)                 \
    c[r][0] = vdotq_laneq_s32(c[r][0], b0, av, lane); \
    c[r][1] = vdotq_laneq_s32(c[r][1], b1, av, lane); \
    c[r][2] = vdotq_laneq_s32(c[r][2], b2, av, lane);
        TINYINFER_SDOT_ROW(0, a0, 0)
        TINYINFER_SDOT_ROW(1, a0, 1)
        TINYINFER_SDOT_ROW(2, a0, 2)
        TINYINFER_SDOT_ROW(3, a0, 3)
        TINYINFER_SDOT_ROW(4, a1, 0)
        TINYINFER_SDOT_ROW(5, a1, 1)
        TINYINFER_SDOT_ROW(6, a1, 2)
        TINYINFER_SDOT_ROW(7, a1, 3)
#undef TINYINFER_SDOT_ROW
    }
    for (int r = 0; r < MR; ++r) {
        for (int j = 0; j < 3; ++j) vst1q_s32(C + static_cast<size_t>(r) * ldc + 4 * j, c[r][j]);
    }
}
#elif defined(TINYINFER_I8_VNNI)
inline __m256i load_u32_bcast(const uint8_t* p) {
    int32_t v;
    std::memcpy(&v, p, 4);
    return _mm256_set1_epi32(v);
}
inline void micro_kernel(int groups, const APacked* Ap, const int8_t* Bp, const int32_t* colsum, int32_t* C, int ldc,
                         bool accumulate) {
    // Start from -128 * colsum so the unsigned offset on A cancels out.
    const __m256i m128 = _mm256_set1_epi32(-128);
    const __m256i corr0 = _mm256_mullo_epi32(m128, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(colsum)));
    const __m256i corr1 = _mm256_mullo_epi32(m128, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(colsum + 8)));
    __m256i c[MR][2];
    for (int r = 0; r < MR; ++r) {
        if (accumulate) {
            c[r][0] = _mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + static_cast<size_t>(r) * ldc)), corr0);
            c[r][1] = _mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + static_cast<size_t>(r) * ldc + 8)), corr1);
        } else {
            c[r][0] = corr0;
            c[r][1] = corr1;
        }
    }
    for (int g = 0; g < groups; ++g) {
        const int8_t* b = Bp + static_cast<size_t>(g) * NR * KP;
        const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
        const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 32));
        const uint8_t* a = Ap + static_cast<size_t>(g) * MR * KP;
        __m256i av;
        av = load_u32_bcast(a + 0 * KP); c[0][0] = TINYINFER_DPBUSD(c[0][0], av, b0); c[0][1] = TINYINFER_DPBUSD(c[0][1], av, b1);
        av = load_u32_bcast(a + 1 * KP); c[1][0] = TINYINFER_DPBUSD(c[1][0], av, b0); c[1][1] = TINYINFER_DPBUSD(c[1][1], av, b1);
        av = load_u32_bcast(a + 2 * KP); c[2][0] = TINYINFER_DPBUSD(c[2][0], av, b0); c[2][1] = TINYINFER_DPBUSD(c[2][1], av, b1);
        av = load_u32_bcast(a + 3 * KP); c[3][0] = TINYINFER_DPBUSD(c[3][0], av, b0); c[3][1] = TINYINFER_DPBUSD(c[3][1], av, b1);
        av = load_u32_bcast(a + 4 * KP); c[4][0] = TINYINFER_DPBUSD(c[4][0], av, b0); c[4][1] = TINYINFER_DPBUSD(c[4][1], av, b1);
        av = load_u32_bcast(a + 5 * KP); c[5][0] = TINYINFER_DPBUSD(c[5][0], av, b0); c[5][1] = TINYINFER_DPBUSD(c[5][1], av, b1);
    }
    for (int r = 0; r < MR; ++r) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(C + static_cast<size_t>(r) * ldc), c[r][0]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(C + static_cast<size_t>(r) * ldc + 8), c[r][1]);
    }
}
#elif defined(TINYINFER_I8_AVX2)
inline __m256i load_i16x4_bcast(const int16_t* p) {
    int64_t v;
    std::memcpy(&v, p, 8);
    return _mm256_set1_epi64x(v);
}
inline void micro_kernel(int groups, const APacked* Ap, const int8_t* Bp, const int32_t* /*colsum*/, int32_t* C, int ldc,
                         bool accumulate) {
    // Lane layout of each accumulator: [n0 k01, n0 k23, n1 k01, n1 k23, n2 .., n2 .., n3 .., n3 ..]
    __m256i c[MR][2];
    for (int r = 0; r < MR; ++r) c[r][0] = c[r][1] = _mm256_setzero_si256();
    for (int g = 0; g < groups; ++g) {
        const int8_t* b = Bp + static_cast<size_t>(g) * NR * KP;  // 8 columns x 4 k = 32 bytes
        const __m256i b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b)));       // cols 0-3
        const __m256i b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 16)));  // cols 4-7
        const int16_t* a = Ap + static_cast<size_t>(g) * MR * KP;
        for (int r = 0; r < MR; ++r) {
            const __m256i av = load_i16x4_bcast(a + r * KP);
            c[r][0] = _mm256_add_epi32(c[r][0], _mm256_madd_epi16(b0, av));
            c[r][1] = _mm256_add_epi32(c[r][1], _mm256_madd_epi16(b1, av));
        }
    }
    alignas(32) int32_t lanes[16];
    for (int r = 0; r < MR; ++r) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), c[r][0]);
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes + 8), c[r][1]);
        int32_t* out = C + static_cast<size_t>(r) * ldc;
        for (int t = 0; t < NR; ++t) {
            const int32_t v = lanes[2 * t] + lanes[2 * t + 1];
            out[t] = accumulate ? out[t] + v : v;
        }
    }
}
#else
inline void micro_kernel(int groups, const APacked* Ap, const int8_t* Bp, const int32_t* /*colsum*/, int32_t* C, int ldc,
                         bool accumulate) {
    int32_t c[MR][NR];
    for (int r = 0; r < MR; ++r) {
        for (int t = 0; t < NR; ++t) c[r][t] = accumulate ? C[static_cast<size_t>(r) * ldc + t] : 0;
    }
    for (int g = 0; g < groups; ++g) {
        const int8_t* a = Ap + static_cast<size_t>(g) * MR * KP;
        const int8_t* b = Bp + static_cast<size_t>(g) * NR * KP;
        for (int r = 0; r < MR; ++r) {
            for (int t = 0; t < NR; ++t) {
                int32_t s = 0;
                for (int q = 0; q < KP; ++q) s += static_cast<int32_t>(a[r * KP + q]) * b[t * KP + q];
                c[r][t] += s;
            }
        }
    }
    for (int r = 0; r < MR; ++r) {
        for (int t = 0; t < NR; ++t) C[static_cast<size_t>(r) * ldc + t] = c[r][t];
    }
}
#endif

inline void run_panel(int mc, int nc, int kc, const APacked* Ap, const int8_t* Bp, const int32_t* colsum, int32_t* C,
                      int ldc, bool accumulate) {
    const int groups = round_up(kc, KP) / KP;
    alignas(64) int32_t tmp[MR * NR];
    for (int jr = 0; jr < nc; jr += NR) {
        const int w = std::min(NR, nc - jr);
        const int8_t* bs = Bp + static_cast<size_t>(jr / NR) * groups * NR * KP;
        const int32_t* cs = colsum + jr;
        for (int ir = 0; ir < mc; ir += MR) {
            const int h = std::min(MR, mc - ir);
            const APacked* as = Ap + static_cast<size_t>(ir / MR) * groups * MR * KP;
            int32_t* c = C + static_cast<size_t>(ir) * ldc + jr;
            if (h == MR && w == NR) {
                micro_kernel(groups, as, bs, cs, c, ldc, accumulate);
            } else {
                micro_kernel(groups, as, bs, cs, tmp, NR, false);
                for (int r = 0; r < h; ++r) {
                    for (int t = 0; t < w; ++t) {
                        if (accumulate) {
                            c[static_cast<size_t>(r) * ldc + t] += tmp[r * NR + t];
                        } else {
                            c[static_cast<size_t>(r) * ldc + t] = tmp[r * NR + t];
                        }
                    }
                }
            }
        }
    }
}

}  // namespace kernel_i8
}  // namespace tinyinfer
