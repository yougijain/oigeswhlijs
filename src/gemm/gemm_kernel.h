#pragma once
// Internal to the GEMM implementation: panel packing and the register-blocked
// micro-kernel shared by gemm_simd (one thread) and gemm_threaded (OpenMP).
//
// The structure is the classic Goto/BLIS one:
//   - B is cut into kc x nc panels and re-laid out ("packed") into strips of
//     NR columns so the micro-kernel reads it as a contiguous stream.
//   - A is cut into mc x kc panels and packed into strips of MR rows.
//   - The micro-kernel keeps an MR x NR tile of C in registers and, for each k,
//     loads NR floats of B and broadcasts MR floats of A, doing MR*NR FMAs per
//     (MR + NR) loads. That ratio is what makes it fast: the naive loop does
//     one FMA per two loads.
// Edge tiles are handled by zero-padding in the packing step and by computing
// into a scratch tile that is copied out, so the micro-kernel itself never
// needs a bounds check.

#include <algorithm>
#include <cstddef>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define TINYINFER_KERNEL_AVX2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define TINYINFER_KERNEL_NEON 1
#else
#define TINYINFER_KERNEL_SCALAR 1
#endif

namespace tinyinfer {
namespace kernel {

#if defined(TINYINFER_KERNEL_AVX2)
// 6 rows x 16 columns = 12 ymm accumulators + 2 for B + 1 broadcast = 15 of 16 registers.
constexpr int MR = 6;
constexpr int NR = 16;
constexpr const char* kBackendName = "AVX2+FMA, 6x16 micro-kernel";
#elif defined(TINYINFER_KERNEL_NEON)
// 8 rows x 12 columns = 24 q accumulators + 3 for B + 2 for A = 29 of 32 registers.
// FMLA by-element means A needs no broadcast: 5 loads per 24 FMAs.
constexpr int MR = 8;
constexpr int NR = 12;
constexpr const char* kBackendName = "NEON, 8x12 micro-kernel";
#else
constexpr int MR = 4;
constexpr int NR = 8;
constexpr const char* kBackendName = "scalar fallback, 4x8 micro-kernel";
#endif

// Panel sizes. mc is a multiple of both 6 and 8, nc of both 16 and 12.
// Packed A panel: MC*KC*4 = 72 KiB, lives in L2. One B strip: KC*NR*4 = 16 KiB
// (AVX2) or 12 KiB (NEON), lives in L1 while all MC/MR tiles of A sweep it.
constexpr int MC = 72;
constexpr int KC = 256;
constexpr int NC = 3072;

inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

// B[kc rows][nc cols], row stride ldb -> strips of NR columns. Strip s holds
// rows 0..kc-1 of columns s*NR..s*NR+NR-1 as kc consecutive groups of NR floats.
inline void pack_b(int kc, int nc, const float* B, int ldb, float* out) {
    for (int j = 0; j < nc; j += NR) {
        const int w = std::min(NR, nc - j);
        float* strip = out + static_cast<size_t>(j / NR) * kc * NR;
        for (int k = 0; k < kc; ++k) {
            const float* src = B + static_cast<size_t>(k) * ldb + j;
            float* dst = strip + static_cast<size_t>(k) * NR;
            for (int t = 0; t < w; ++t) dst[t] = src[t];
            for (int t = w; t < NR; ++t) dst[t] = 0.0f;
        }
    }
}

// A[mc rows][kc cols], row stride lda -> strips of MR rows. Strip s holds
// columns 0..kc-1 of rows s*MR..s*MR+MR-1 as kc consecutive groups of MR floats.
inline void pack_a(int mc, int kc, const float* A, int lda, float* out) {
    for (int i = 0; i < mc; i += MR) {
        const int h = std::min(MR, mc - i);
        float* strip = out + static_cast<size_t>(i / MR) * kc * MR;
        for (int r = 0; r < h; ++r) {
            const float* src = A + static_cast<size_t>(i + r) * lda;
            for (int k = 0; k < kc; ++k) strip[static_cast<size_t>(k) * MR + r] = src[k];
        }
        for (int r = h; r < MR; ++r) {
            for (int k = 0; k < kc; ++k) strip[static_cast<size_t>(k) * MR + r] = 0.0f;
        }
    }
}

// C[MR][NR] (row stride ldc) = (accumulate ? C : 0) + Ap[kc][MR]^T * Bp[kc][NR]
#if defined(TINYINFER_KERNEL_AVX2)
inline void micro_kernel(int kc, const float* Ap, const float* Bp, float* C, int ldc, bool accumulate) {
    __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
    if (accumulate) {
        c00 = _mm256_loadu_ps(C + 0 * static_cast<size_t>(ldc));
        c01 = _mm256_loadu_ps(C + 0 * static_cast<size_t>(ldc) + 8);
        c10 = _mm256_loadu_ps(C + 1 * static_cast<size_t>(ldc));
        c11 = _mm256_loadu_ps(C + 1 * static_cast<size_t>(ldc) + 8);
        c20 = _mm256_loadu_ps(C + 2 * static_cast<size_t>(ldc));
        c21 = _mm256_loadu_ps(C + 2 * static_cast<size_t>(ldc) + 8);
        c30 = _mm256_loadu_ps(C + 3 * static_cast<size_t>(ldc));
        c31 = _mm256_loadu_ps(C + 3 * static_cast<size_t>(ldc) + 8);
        c40 = _mm256_loadu_ps(C + 4 * static_cast<size_t>(ldc));
        c41 = _mm256_loadu_ps(C + 4 * static_cast<size_t>(ldc) + 8);
        c50 = _mm256_loadu_ps(C + 5 * static_cast<size_t>(ldc));
        c51 = _mm256_loadu_ps(C + 5 * static_cast<size_t>(ldc) + 8);
    } else {
        c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_ps();
    }
    for (int k = 0; k < kc; ++k) {
        const __m256 b0 = _mm256_loadu_ps(Bp + static_cast<size_t>(k) * NR);
        const __m256 b1 = _mm256_loadu_ps(Bp + static_cast<size_t>(k) * NR + 8);
        const float* a = Ap + static_cast<size_t>(k) * MR;
        __m256 av;
        av = _mm256_broadcast_ss(a + 0); c00 = _mm256_fmadd_ps(av, b0, c00); c01 = _mm256_fmadd_ps(av, b1, c01);
        av = _mm256_broadcast_ss(a + 1); c10 = _mm256_fmadd_ps(av, b0, c10); c11 = _mm256_fmadd_ps(av, b1, c11);
        av = _mm256_broadcast_ss(a + 2); c20 = _mm256_fmadd_ps(av, b0, c20); c21 = _mm256_fmadd_ps(av, b1, c21);
        av = _mm256_broadcast_ss(a + 3); c30 = _mm256_fmadd_ps(av, b0, c30); c31 = _mm256_fmadd_ps(av, b1, c31);
        av = _mm256_broadcast_ss(a + 4); c40 = _mm256_fmadd_ps(av, b0, c40); c41 = _mm256_fmadd_ps(av, b1, c41);
        av = _mm256_broadcast_ss(a + 5); c50 = _mm256_fmadd_ps(av, b0, c50); c51 = _mm256_fmadd_ps(av, b1, c51);
    }
    _mm256_storeu_ps(C + 0 * static_cast<size_t>(ldc), c00);
    _mm256_storeu_ps(C + 0 * static_cast<size_t>(ldc) + 8, c01);
    _mm256_storeu_ps(C + 1 * static_cast<size_t>(ldc), c10);
    _mm256_storeu_ps(C + 1 * static_cast<size_t>(ldc) + 8, c11);
    _mm256_storeu_ps(C + 2 * static_cast<size_t>(ldc), c20);
    _mm256_storeu_ps(C + 2 * static_cast<size_t>(ldc) + 8, c21);
    _mm256_storeu_ps(C + 3 * static_cast<size_t>(ldc), c30);
    _mm256_storeu_ps(C + 3 * static_cast<size_t>(ldc) + 8, c31);
    _mm256_storeu_ps(C + 4 * static_cast<size_t>(ldc), c40);
    _mm256_storeu_ps(C + 4 * static_cast<size_t>(ldc) + 8, c41);
    _mm256_storeu_ps(C + 5 * static_cast<size_t>(ldc), c50);
    _mm256_storeu_ps(C + 5 * static_cast<size_t>(ldc) + 8, c51);
}
#elif defined(TINYINFER_KERNEL_NEON)
inline void micro_kernel(int kc, const float* Ap, const float* Bp, float* C, int ldc, bool accumulate) {
    float32x4_t c[MR][3];
    for (int r = 0; r < MR; ++r) {
        if (accumulate) {
            c[r][0] = vld1q_f32(C + static_cast<size_t>(r) * ldc);
            c[r][1] = vld1q_f32(C + static_cast<size_t>(r) * ldc + 4);
            c[r][2] = vld1q_f32(C + static_cast<size_t>(r) * ldc + 8);
        } else {
            c[r][0] = c[r][1] = c[r][2] = vdupq_n_f32(0.0f);
        }
    }
    for (int k = 0; k < kc; ++k) {
        const float32x4_t b0 = vld1q_f32(Bp + static_cast<size_t>(k) * NR);
        const float32x4_t b1 = vld1q_f32(Bp + static_cast<size_t>(k) * NR + 4);
        const float32x4_t b2 = vld1q_f32(Bp + static_cast<size_t>(k) * NR + 8);
        const float32x4_t a0 = vld1q_f32(Ap + static_cast<size_t>(k) * MR);
        const float32x4_t a1 = vld1q_f32(Ap + static_cast<size_t>(k) * MR + 4);
        // vfmaq_laneq_f32(acc, b, a, lane): acc += b * a[lane]; the lane index must be a constant.
#define TINYINFER_NEON_ROW(r, av, lane)                     \
    c[r][0] = vfmaq_laneq_f32(c[r][0], b0, av, lane);      \
    c[r][1] = vfmaq_laneq_f32(c[r][1], b1, av, lane);      \
    c[r][2] = vfmaq_laneq_f32(c[r][2], b2, av, lane);
        TINYINFER_NEON_ROW(0, a0, 0)
        TINYINFER_NEON_ROW(1, a0, 1)
        TINYINFER_NEON_ROW(2, a0, 2)
        TINYINFER_NEON_ROW(3, a0, 3)
        TINYINFER_NEON_ROW(4, a1, 0)
        TINYINFER_NEON_ROW(5, a1, 1)
        TINYINFER_NEON_ROW(6, a1, 2)
        TINYINFER_NEON_ROW(7, a1, 3)
#undef TINYINFER_NEON_ROW
    }
    for (int r = 0; r < MR; ++r) {
        vst1q_f32(C + static_cast<size_t>(r) * ldc, c[r][0]);
        vst1q_f32(C + static_cast<size_t>(r) * ldc + 4, c[r][1]);
        vst1q_f32(C + static_cast<size_t>(r) * ldc + 8, c[r][2]);
    }
}
#else
inline void micro_kernel(int kc, const float* Ap, const float* Bp, float* C, int ldc, bool accumulate) {
    float c[MR][NR];
    for (int r = 0; r < MR; ++r) {
        for (int t = 0; t < NR; ++t) c[r][t] = accumulate ? C[static_cast<size_t>(r) * ldc + t] : 0.0f;
    }
    for (int k = 0; k < kc; ++k) {
        const float* a = Ap + static_cast<size_t>(k) * MR;
        const float* b = Bp + static_cast<size_t>(k) * NR;
        for (int r = 0; r < MR; ++r) {
            for (int t = 0; t < NR; ++t) c[r][t] += a[r] * b[t];
        }
    }
    for (int r = 0; r < MR; ++r) {
        for (int t = 0; t < NR; ++t) C[static_cast<size_t>(r) * ldc + t] = c[r][t];
    }
}
#endif

// Run the micro-kernel over one packed A panel (mc rows) against one packed B
// panel (nc columns), writing C[mc][nc]. Full tiles go straight to C; edge
// tiles go through a scratch tile.
inline void run_panel(int mc, int nc, int kc, const float* Ap, const float* Bp, float* C, int ldc, bool accumulate) {
    alignas(64) float tmp[MR * NR];
    for (int jr = 0; jr < nc; jr += NR) {
        const int w = std::min(NR, nc - jr);
        const float* bs = Bp + static_cast<size_t>(jr / NR) * kc * NR;
        for (int ir = 0; ir < mc; ir += MR) {
            const int h = std::min(MR, mc - ir);
            const float* as = Ap + static_cast<size_t>(ir / MR) * kc * MR;
            float* c = C + static_cast<size_t>(ir) * ldc + jr;
            if (h == MR && w == NR) {
                micro_kernel(kc, as, bs, c, ldc, accumulate);
            } else {
                micro_kernel(kc, as, bs, tmp, NR, false);
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

}  // namespace kernel
}  // namespace tinyinfer
