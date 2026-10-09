#include <vector>

#include "gemm_kernel.h"
#include "tinyinfer/gemm.h"

namespace tinyinfer {

const char* gemm_simd_backend() { return kernel::kBackendName; }

// Packed panels + register-blocked micro-kernel, one thread. See gemm_kernel.h.
void gemm_simd(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    using namespace kernel;
    // Scratch persists across calls so steady-state inference does not allocate.
    static thread_local std::vector<float> packed_a, packed_b;
    packed_a.resize(static_cast<size_t>(MC) * KC);
    packed_b.resize(static_cast<size_t>(round_up(std::min(N, NC), NR)) * KC);

    for (int jc = 0; jc < N; jc += NC) {
        const int nc = std::min(NC, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            const int kc = std::min(KC, K - pc);
            pack_b(kc, nc, B + static_cast<size_t>(pc) * ldb + jc, ldb, packed_b.data());
            for (int ic = 0; ic < M; ic += MC) {
                const int mc = std::min(MC, M - ic);
                pack_a(mc, kc, A + static_cast<size_t>(ic) * lda + pc, lda, packed_a.data());
                run_panel(mc, nc, kc, packed_a.data(), packed_b.data(), C + static_cast<size_t>(ic) * ldc + jc, ldc,
                          pc > 0);
            }
        }
    }
}

}  // namespace tinyinfer
