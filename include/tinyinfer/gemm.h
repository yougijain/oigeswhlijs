#pragma once
// Single-precision GEMM: C[M][N] = A[M][K] * B[K][N].
//
// All matrices are row-major. lda/ldb/ldc are leading dimensions in elements
// (the distance between consecutive rows), so sub-matrices can be passed
// without copying. C is overwritten, not accumulated into.

#include <string>

namespace tinyinfer {

enum class GemmKind {
    Naive,  // i-j-k loops, B walked down a column in the inner loop
};
constexpr int kGemmKindCount = 1;

const char* gemm_kind_name(GemmKind kind);
bool parse_gemm_kind(const std::string& name, GemmKind& out);
GemmKind default_gemm_kind();

void gemm(GemmKind kind, int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);

void gemm_naive(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);

}  // namespace tinyinfer
