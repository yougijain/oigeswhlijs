#include <vector>

#include "gemm_kernel.h"
#include "tinyinfer/gemm.h"

namespace tinyinfer {

const char* gemm_simd_backend() { return kernel::kBackendName; }

PackedALayout gemm_packed_a_layout() { return {kernel::MR, kernel::KC}; }

int64_t gemm_packed_a_size(int M, int K) {
    return static_cast<int64_t>(kernel::round_up(M, kernel::MR)) * K;
}

// A arrives already in strip layout (see gemm.h), so each (block, row-block)
// panel is a pointer computation instead of a copy.
void gemm_simd_packed_a(int M, int N, int K, const float* Ap, const float* B, int ldb, float* C, int ldc) {
    using namespace kernel;
    static thread_local std::vector<float> packed_b;
    packed_b.resize(static_cast<size_t>(round_up(std::min(N, NC), NR)) * KC);
    const int64_t m_pad = round_up(M, MR);

    for (int jc = 0; jc < N; jc += NC) {
        const int nc = std::min(NC, N - jc);
        for (int pc = 0, b = 0; pc < K; pc += KC, ++b) {
            const int kc = std::min(KC, K - pc);
            pack_b(kc, nc, B + static_cast<size_t>(pc) * ldb + jc, ldb, packed_b.data());
            const float* block = Ap + static_cast<size_t>(b) * m_pad * KC;
            for (int ic = 0; ic < M; ic += MC) {
                const int mc = std::min(MC, M - ic);
                const float* panel = block + static_cast<size_t>(ic / MR) * kc * MR;
                run_panel(mc, nc, kc, panel, packed_b.data(), C + static_cast<size_t>(ic) * ldc + jc, ldc, pc > 0);
            }
        }
    }
}

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
