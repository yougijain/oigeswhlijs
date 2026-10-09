#pragma once
// Integer GEMM: C[M][N] (int32) = A[M][K] (int8) * B[K][N] (int8).
//
// Exact integer arithmetic, so every implementation must produce identical
// output; the test checks equality, not tolerance. Row-major, leading
// dimensions in elements, C overwritten.

#include <cstdint>
#include <string>

namespace tinyinfer {

enum class GemmInt8Kind {
    Naive,     // i-j-k reference
    Simd,      // packed panels + 4-wide dot-product micro-kernel (NEON sdot / AVX-VNNI / AVX2 / scalar)
    Threaded,  // simd with M split across OpenMP threads
};
constexpr int kGemmInt8KindCount = 3;

const char* gemm_int8_kind_name(GemmInt8Kind kind);
bool parse_gemm_int8_kind(const std::string& name, GemmInt8Kind& out);
GemmInt8Kind default_gemm_int8_kind();
const char* gemm_int8_backend();

void gemm_s8(GemmInt8Kind kind, int M, int N, int K, const int8_t* A, int lda, const int8_t* B, int ldb, int32_t* C,
             int ldc);

void gemm_s8_naive(int M, int N, int K, const int8_t* A, int lda, const int8_t* B, int ldb, int32_t* C, int ldc);
void gemm_s8_simd(int M, int N, int K, const int8_t* A, int lda, const int8_t* B, int ldb, int32_t* C, int ldc);
void gemm_s8_threaded(int M, int N, int K, const int8_t* A, int lda, const int8_t* B, int ldb, int32_t* C, int ldc);

}  // namespace tinyinfer
