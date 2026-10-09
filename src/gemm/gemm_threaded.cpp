#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "gemm_kernel.h"
#include "tinyinfer/gemm.h"

namespace tinyinfer {

int gemm_thread_count() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

#ifdef _OPENMP
// The simd version with the M dimension split across OpenMP threads. Each
// thread packs its own A panel into thread-local scratch; the B panel is
// packed once per (jc, pc) block and shared read-only. Splitting M (not N)
// suits this engine: the conv GEMMs have M = batch*pixels in the thousands
// and N = out_channels at most 128, so M is where the parallelism is.
void gemm_threaded(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    using namespace kernel;
    static thread_local std::vector<float> packed_b;
    packed_b.resize(static_cast<size_t>(round_up(std::min(N, NC), NR)) * KC);
    const int m_blocks = (M + MC - 1) / MC;

    for (int jc = 0; jc < N; jc += NC) {
        const int nc = std::min(NC, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            const int kc = std::min(KC, K - pc);
            pack_b(kc, nc, B + static_cast<size_t>(pc) * ldb + jc, ldb, packed_b.data());
            const float* bp = packed_b.data();
#pragma omp parallel for schedule(dynamic, 1)
            for (int blk = 0; blk < m_blocks; ++blk) {
                static thread_local std::vector<float> packed_a;
                packed_a.resize(static_cast<size_t>(MC) * KC);
                const int ic = blk * MC;
                const int mc = std::min(MC, M - ic);
                pack_a(mc, kc, A + static_cast<size_t>(ic) * lda + pc, lda, packed_a.data());
                run_panel(mc, nc, kc, packed_a.data(), bp, C + static_cast<size_t>(ic) * ldc + jc, ldc, pc > 0);
            }
        }
    }
}
#else
void gemm_threaded(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    gemm_simd(M, N, K, A, lda, B, ldb, C, ldc);
}
#endif

}  // namespace tinyinfer
