#include "tinyinfer/gemm_int8.h"

#include <stdexcept>

namespace tinyinfer {

const char* gemm_int8_kind_name(GemmInt8Kind kind) {
    switch (kind) {
        case GemmInt8Kind::Naive: return "naive";
        case GemmInt8Kind::Simd: return "simd";
        case GemmInt8Kind::Threaded: return "threaded";
    }
    return "?";
}

bool parse_gemm_int8_kind(const std::string& name, GemmInt8Kind& out) {
    for (int i = 0; i < kGemmInt8KindCount; ++i) {
        const GemmInt8Kind k = static_cast<GemmInt8Kind>(i);
        if (name == gemm_int8_kind_name(k)) {
            out = k;
            return true;
        }
    }
    return false;
}

GemmInt8Kind default_gemm_int8_kind() { return GemmInt8Kind::Threaded; }

bool gemm_int8_kind_uses_packed_a(GemmInt8Kind kind) {
    return (kind == GemmInt8Kind::Simd || kind == GemmInt8Kind::Threaded) && gemm_int8_packed_a_layout().elem_bytes == 1;
}

void gemm_s8_packed_a(GemmInt8Kind kind, int M, int N, int K, const uint8_t* Apacked, const int8_t* B, int ldb,
                      int32_t* C, int ldc) {
    if (M < 0 || N < 0 || K < 0) throw std::invalid_argument("gemm_s8_packed_a: negative dimension");
    if (ldb < N || ldc < N) throw std::invalid_argument("gemm_s8_packed_a: leading dimension smaller than row");
    if (!gemm_int8_kind_uses_packed_a(kind)) throw std::invalid_argument("gemm_s8_packed_a: kind does not take a packed A");
    if (M == 0 || N == 0) return;
    if (kind == GemmInt8Kind::Simd) {
        gemm_s8_simd_packed_a(M, N, K, Apacked, B, ldb, C, ldc);
    } else {
        gemm_s8_threaded_packed_a(M, N, K, Apacked, B, ldb, C, ldc);
    }
}

void gemm_s8(GemmInt8Kind kind, int M, int N, int K, const int8_t* A, int lda, const int8_t* B, int ldb, int32_t* C,
             int ldc) {
    if (M < 0 || N < 0 || K < 0) throw std::invalid_argument("gemm_s8: negative dimension");
    if (lda < K || ldb < N || ldc < N) throw std::invalid_argument("gemm_s8: leading dimension smaller than row");
    if (M == 0 || N == 0) return;
    switch (kind) {
        case GemmInt8Kind::Naive: gemm_s8_naive(M, N, K, A, lda, B, ldb, C, ldc); return;
        case GemmInt8Kind::Simd: gemm_s8_simd(M, N, K, A, lda, B, ldb, C, ldc); return;
        case GemmInt8Kind::Threaded: gemm_s8_threaded(M, N, K, A, lda, B, ldb, C, ldc); return;
    }
    throw std::invalid_argument("gemm_s8: unknown kind");
}

// Reference. int8 * int8 products are at most 2^14 in magnitude, so K up to
// 2^17 accumulates without overflowing int32.
void gemm_s8_naive(int M, int N, int K, const int8_t* A, int lda, const int8_t* B, int ldb, int32_t* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(A[static_cast<size_t>(i) * lda + k]) *
                       static_cast<int32_t>(B[static_cast<size_t>(k) * ldb + j]);
            }
            C[static_cast<size_t>(i) * ldc + j] = acc;
        }
    }
}

}  // namespace tinyinfer
