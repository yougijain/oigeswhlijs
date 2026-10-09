#include "tinyinfer/gemm.h"

#include <stdexcept>

namespace tinyinfer {

const char* gemm_kind_name(GemmKind kind) {
    switch (kind) {
        case GemmKind::Naive: return "naive";
        case GemmKind::Reordered: return "reordered";
        case GemmKind::Tiled: return "tiled";
        case GemmKind::Simd: return "simd";
        case GemmKind::Threaded: return "threaded";
    }
    return "?";
}

bool parse_gemm_kind(const std::string& name, GemmKind& out) {
    for (int i = 0; i < kGemmKindCount; ++i) {
        const GemmKind k = static_cast<GemmKind>(i);
        if (name == gemm_kind_name(k)) {
            out = k;
            return true;
        }
    }
    return false;
}

GemmKind default_gemm_kind() { return GemmKind::Threaded; }

bool gemm_kind_uses_packed_a(GemmKind kind) { return kind == GemmKind::Simd || kind == GemmKind::Threaded; }

void gemm_packed_a(GemmKind kind, int M, int N, int K, const float* Apacked, const float* B, int ldb, float* C,
                   int ldc) {
    if (M < 0 || N < 0 || K < 0) throw std::invalid_argument("gemm_packed_a: negative dimension");
    if (ldb < N || ldc < N) throw std::invalid_argument("gemm_packed_a: leading dimension smaller than row");
    if (M == 0 || N == 0) return;
    switch (kind) {
        case GemmKind::Simd: gemm_simd_packed_a(M, N, K, Apacked, B, ldb, C, ldc); return;
        case GemmKind::Threaded: gemm_threaded_packed_a(M, N, K, Apacked, B, ldb, C, ldc); return;
        default: break;
    }
    throw std::invalid_argument("gemm_packed_a: kind does not take a packed A");
}

void gemm(GemmKind kind, int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    if (M < 0 || N < 0 || K < 0) throw std::invalid_argument("gemm: negative dimension");
    if (lda < K || ldb < N || ldc < N) throw std::invalid_argument("gemm: leading dimension smaller than row");
    if (M == 0 || N == 0) return;
    switch (kind) {
        case GemmKind::Naive: gemm_naive(M, N, K, A, lda, B, ldb, C, ldc); return;
        case GemmKind::Reordered: gemm_reordered(M, N, K, A, lda, B, ldb, C, ldc); return;
        case GemmKind::Tiled: gemm_tiled(M, N, K, A, lda, B, ldb, C, ldc); return;
        case GemmKind::Simd: gemm_simd(M, N, K, A, lda, B, ldb, C, ldc); return;
        case GemmKind::Threaded: gemm_threaded(M, N, K, A, lda, B, ldb, C, ldc); return;
    }
    throw std::invalid_argument("gemm: unknown kind");
}

}  // namespace tinyinfer
