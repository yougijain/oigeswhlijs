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

// Pre-packed A for the simd and threaded kinds (the int8 analogue of
// gemm_packed_a in gemm.h). Blocks of kc along K, strips of mr rows, k in
// groups of 4 so one dot-product lane reads 4 consecutive k:
//     offset(m, k) = b * round_up(M, mr) * kc            (block b = k / kc)
//                  + (m / mr) * groups_b * mr * 4         (groups_b = ceil(kc_b / 4))
//                  + ((k % kc) / 4) * mr * 4 + (m % mr) * 4 + (k % 4)
// in bytes. The stored byte is a ^ xor_mask (0x80 on the VNNI backend, which
// multiplies unsigned bytes; 0 elsewhere). Padding (rows beyond M, k beyond
// kc_b in the last group) can hold anything: the matching B entries are zero.
// Only backends whose packed element is one byte support this; the AVX2
// fallback packs int16 and reports elem_bytes = 2, so callers fall back to
// gemm_s8 with a row-major A.
struct PackedALayoutI8 {
    int mr;
    int kc;
    int group;  // 4
    int elem_bytes;
    uint8_t xor_mask;
};
PackedALayoutI8 gemm_int8_packed_a_layout();
int64_t gemm_int8_packed_a_size(int M, int K);  // bytes
bool gemm_int8_kind_uses_packed_a(GemmInt8Kind kind);
void gemm_s8_packed_a(GemmInt8Kind kind, int M, int N, int K, const uint8_t* Apacked, const int8_t* B, int ldb,
                      int32_t* C, int ldc);
void gemm_s8_simd_packed_a(int M, int N, int K, const uint8_t* Apacked, const int8_t* B, int ldb, int32_t* C, int ldc);
void gemm_s8_threaded_packed_a(int M, int N, int K, const uint8_t* Apacked, const int8_t* B, int ldb, int32_t* C,
                               int ldc);

}  // namespace tinyinfer
