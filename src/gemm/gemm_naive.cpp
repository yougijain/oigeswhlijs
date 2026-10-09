#include "tinyinfer/gemm.h"

namespace tinyinfer {

// The textbook triple loop. For every output element it walks a row of A
// (contiguous, fine) and a column of B (stride ldb floats, so each step is a
// different cache line). That column walk is what every later version fixes.
void gemm_naive(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[static_cast<size_t>(i) * lda + k] * B[static_cast<size_t>(k) * ldb + j];
            }
            C[static_cast<size_t>(i) * ldc + j] = acc;
        }
    }
}

}  // namespace tinyinfer
