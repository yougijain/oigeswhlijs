#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "gemm_int8_kernel.h"
#include "tinyinfer/gemm_int8.h"

namespace tinyinfer {

const char* gemm_int8_backend() { return kernel_i8::kBackendName; }

void gemm_s8_simd(int M, int N, int K, const int8_t* A, int lda, const int8_t* B, int ldb, int32_t* C, int ldc) {
    using namespace kernel_i8;
    static thread_local std::vector<APacked> packed_a;
    static thread_local std::vector<int8_t> packed_b;
    static thread_local std::vector<int32_t> colsum;
    const int nc_max = round_up(std::min(N, NC), NR);
    packed_a.resize(static_cast<size_t>(MC) * KC);
    packed_b.resize(static_cast<size_t>(nc_max) * KC);
    colsum.resize(static_cast<size_t>(nc_max));

    for (int jc = 0; jc < N; jc += NC) {
        const int nc = std::min(NC, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            const int kc = std::min(KC, K - pc);
            pack_b(kc, nc, B + static_cast<size_t>(pc) * ldb + jc, ldb, packed_b.data(), colsum.data());
            for (int ic = 0; ic < M; ic += MC) {
                const int mc = std::min(MC, M - ic);
                pack_a(mc, kc, A + static_cast<size_t>(ic) * lda + pc, lda, packed_a.data());
                run_panel(mc, nc, kc, packed_a.data(), packed_b.data(), colsum.data(),
                          C + static_cast<size_t>(ic) * ldc + jc, ldc, pc > 0);
            }
        }
    }
}

#ifdef _OPENMP
void gemm_s8_threaded(int M, int N, int K, const int8_t* A, int lda, const int8_t* B, int ldb, int32_t* C, int ldc) {
    using namespace kernel_i8;
    if (M <= MC) {  // one block: skip the parallel region
        gemm_s8_simd(M, N, K, A, lda, B, ldb, C, ldc);
        return;
    }
    static thread_local std::vector<int8_t> packed_b;
    static thread_local std::vector<int32_t> colsum;
    const int nc_max = round_up(std::min(N, NC), NR);
    packed_b.resize(static_cast<size_t>(nc_max) * KC);
    colsum.resize(static_cast<size_t>(nc_max));
    const int m_blocks = (M + MC - 1) / MC;

    for (int jc = 0; jc < N; jc += NC) {
        const int nc = std::min(NC, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            const int kc = std::min(KC, K - pc);
            pack_b(kc, nc, B + static_cast<size_t>(pc) * ldb + jc, ldb, packed_b.data(), colsum.data());
            const int8_t* bp = packed_b.data();
            const int32_t* cs = colsum.data();
#pragma omp parallel for schedule(dynamic, 1)
            for (int blk = 0; blk < m_blocks; ++blk) {
                static thread_local std::vector<APacked> packed_a;
                packed_a.resize(static_cast<size_t>(MC) * KC);
                const int ic = blk * MC;
                const int mc = std::min(MC, M - ic);
                pack_a(mc, kc, A + static_cast<size_t>(ic) * lda + pc, lda, packed_a.data());
                run_panel(mc, nc, kc, packed_a.data(), bp, cs, C + static_cast<size_t>(ic) * ldc + jc, ldc, pc > 0);
            }
        }
    }
}
#else
void gemm_s8_threaded(int M, int N, int K, const int8_t* A, int lda, const int8_t* B, int ldb, int32_t* C, int ldc) {
    gemm_s8_simd(M, N, K, A, lda, B, ldb, C, ldc);
}
#endif

}  // namespace tinyinfer
