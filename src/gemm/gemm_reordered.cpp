#include "tinyinfer/gemm.h"

namespace tinyinfer {

// Same arithmetic as naive, different loop order: i-k-j. The inner loop now
// walks a row of B and a row of C, both contiguous, so every cache line that
// is fetched gets fully used and the loop is a plain streaming AXPY that the
// compiler can vectorise. The scalar A[i][k] is hoisted out of the inner loop.
void gemm_reordered(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        float* c = C + static_cast<size_t>(i) * ldc;
        for (int j = 0; j < N; ++j) c[j] = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float a = A[static_cast<size_t>(i) * lda + k];
            const float* b = B + static_cast<size_t>(k) * ldb;
            for (int j = 0; j < N; ++j) c[j] += a * b[j];
        }
    }
}

}  // namespace tinyinfer
