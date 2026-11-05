#include <algorithm>
#include <stdexcept>

#include "tinyinfer/gemm.h"

namespace tinyinfer {

namespace {
// Defaults chosen by the sweep in results/gemm.md. The B block (kc x nc floats)
// is what has to stay resident while the mc rows of A stream past it.
TileConfig g_tiles = {64, 256, 256};
}  // namespace

TileConfig gemm_tiled_config() { return g_tiles; }

void set_gemm_tiled_config(TileConfig t) {
    if (t.mc <= 0 || t.kc <= 0 || t.nc <= 0) throw std::invalid_argument("tile sizes must be positive");
    g_tiles = t;
}

// Cache blocking on top of the reordered loops. The reordered version streams
// the whole of B (K x N floats) once per row of A, so for anything bigger than
// the cache every row of A re-fetches B from memory. Blocking cuts B into
// kc x nc pieces that fit in cache and runs mc rows of A against each piece
// before moving on, so each piece is loaded once and reused mc times.
void gemm_tiled(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    const int MC = g_tiles.mc, KC = g_tiles.kc, NC = g_tiles.nc;
    for (int jc = 0; jc < N; jc += NC) {
        const int nc = std::min(NC, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            const int kc = std::min(KC, K - pc);
            for (int ic = 0; ic < M; ic += MC) {
                const int mc = std::min(MC, M - ic);
                for (int i = ic; i < ic + mc; ++i) {
                    float* c = C + static_cast<size_t>(i) * ldc + jc;
                    if (pc == 0) {
                        for (int j = 0; j < nc; ++j) c[j] = 0.0f;
                    }
                    for (int k = pc; k < pc + kc; ++k) {
                        const float a = A[static_cast<size_t>(i) * lda + k];
                        const float* b = B + static_cast<size_t>(k) * ldb + jc;
                        for (int j = 0; j < nc; ++j) c[j] += a * b[j];
                    }
                }
            }
        }
    }
}

}  // namespace tinyinfer
