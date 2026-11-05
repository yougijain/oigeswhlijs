#pragma once
// Single-precision GEMM: C[M][N] = A[M][K] * B[K][N].
//
// All matrices are row-major. lda/ldb/ldc are leading dimensions in elements
// (the distance between consecutive rows), so sub-matrices can be passed
// without copying. C is overwritten, not accumulated into.
//
// Five implementations of the same contract, in the order they were written.
// Each one is tested against the naive version; bench/bench_gemm.cpp measures them.

#include <cstdint>
#include <string>

namespace tinyinfer {

enum class GemmKind {
    Naive,      // i-j-k loops, B walked down a column in the inner loop
    Reordered,  // i-k-j loops, inner loop streams a row of B into a row of C
    Tiled,      // reordered loops inside cache blocks so B stays resident
    Simd,       // packed panels + register-blocked FMA micro-kernel (AVX2 / NEON / scalar)
    Threaded,   // simd with the M dimension split across OpenMP threads
};
constexpr int kGemmKindCount = 5;

const char* gemm_kind_name(GemmKind kind);
bool parse_gemm_kind(const std::string& name, GemmKind& out);
GemmKind default_gemm_kind();   // the fastest one available in this build
const char* gemm_simd_backend();  // which micro-kernel the simd/threaded kinds use
int gemm_thread_count();          // OpenMP threads the threaded kind will use (1 without OpenMP)

void gemm(GemmKind kind, int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);

void gemm_naive(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
void gemm_reordered(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
void gemm_tiled(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
void gemm_simd(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
void gemm_threaded(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);

// Cache block sizes used by the tiled kind (rows of A, depth, columns of B).
struct TileConfig {
    int mc, kc, nc;
};
TileConfig gemm_tiled_config();
void set_gemm_tiled_config(TileConfig t);

// Pre-packed A. The simd and threaded kinds copy A into strips of MR rows per
// KC-deep block before the micro-kernel reads it (gemm_kernel.h). A producer
// that already walks every element of A, such as im2col, can write that layout
// directly and skip the copy. Element (m, k) of an M x K matrix lives at
//     b * round_up(M, mr) * kc + (m / mr) * kc_b * mr + (k % kc) * mr + (m % mr)
// with b = k / kc the block and kc_b = min(kc, K - b * kc) its depth; rows
// M..round_up(M, mr)-1 of the last strip must be zero. Total size is
// round_up(M, mr) * K floats.
struct PackedALayout {
    int mr;
    int kc;
};
PackedALayout gemm_packed_a_layout();
int64_t gemm_packed_a_size(int M, int K);
bool gemm_kind_uses_packed_a(GemmKind kind);  // true for Simd and Threaded
// Same contract as gemm() for those two kinds, with A already packed.
void gemm_packed_a(GemmKind kind, int M, int N, int K, const float* Apacked, const float* B, int ldb, float* C,
                   int ldc);
void gemm_simd_packed_a(int M, int N, int K, const float* Apacked, const float* B, int ldb, float* C, int ldc);
void gemm_threaded_packed_a(int M, int N, int K, const float* Apacked, const float* B, int ldb, float* C, int ldc);

}  // namespace tinyinfer
