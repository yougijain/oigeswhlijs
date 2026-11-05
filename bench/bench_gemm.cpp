// GEMM ladder benchmark: every kind on every shape, median of repeated runs,
// in GFLOP/s, with a correctness check against the naive kernel first.
//
//   bench_gemm [--runs 7] [--warmup 1] [--csv results/gemm.csv]
//              [--shapes conv,square] [--sweep] [--sweep-runs 3]
//
// Shapes are the three convolution GEMMs of TinyCNN at batch 64 (M = batch
// times output pixels, N = output channels, K = 3*3*input channels) plus
// square matrices for the usual view.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "bench_util.h"
#include "tinyinfer/gemm.h"
#include "tinyinfer/sysinfo.h"

using namespace tinyinfer;

namespace {

struct Shape {
    const char* name;
    int M, N, K;
};

const Shape kConvShapes[] = {{"conv1 b64", 64 * 32 * 32, 32, 27}, {"conv2 b64", 64 * 16 * 16, 64, 288},
                             {"conv3 b64", 64 * 8 * 8, 128, 576}};
const Shape kSquareShapes[] = {{"square 256", 256, 256, 256}, {"square 512", 512, 512, 512}, {"square 1024", 1024, 1024, 1024}};

std::vector<float> random_matrix(size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    return m;
}

double gflops(const Shape& s, double ms) { return 2.0 * s.M * s.N * s.K / (ms * 1e-3) / 1e9; }

void bench_shape(const Shape& s, int runs, int warmup, bench::Csv& csv, std::mt19937& rng) {
    const std::vector<float> A = random_matrix(static_cast<size_t>(s.M) * s.K, rng);
    const std::vector<float> B = random_matrix(static_cast<size_t>(s.K) * s.N, rng);
    std::vector<float> C(static_cast<size_t>(s.M) * s.N), ref;
    gemm(GemmKind::Naive, s.M, s.N, s.K, A.data(), s.K, B.data(), s.N, C.data(), s.N);
    ref = C;
    const double tol = 1e-5 * std::sqrt(static_cast<double>(s.K)) + 1e-6;

    std::printf("| %-12s |", s.name);
    double naive_gf = 0, last_gf = 0;
    for (int ki = 0; ki < kGemmKindCount; ++ki) {
        const GemmKind kind = static_cast<GemmKind>(ki);
        std::fill(C.begin(), C.end(), -1.0f);
        gemm(kind, s.M, s.N, s.K, A.data(), s.K, B.data(), s.N, C.data(), s.N);
        const double diff = max_abs_diff(C, ref);
        if (diff > tol) {
            std::printf("\nWRONG RESULT: %s on %s differs from naive by %g\n", gemm_kind_name(kind), s.name, diff);
            std::exit(1);
        }
        const bench::Timing t = bench::time_repeated(
            [&] { gemm(kind, s.M, s.N, s.K, A.data(), s.K, B.data(), s.N, C.data(), s.N); }, warmup, runs);
        const double gf = gflops(s, t.median_ms);
        if (kind == GemmKind::Naive) naive_gf = gf;
        last_gf = gf;
        std::printf(" %8.2f |", gf);
        std::fflush(stdout);
        const int threads = kind == GemmKind::Threaded ? gemm_thread_count() : 1;
        csv.row(std::string(s.name) + "," + std::to_string(s.M) + "," + std::to_string(s.N) + "," + std::to_string(s.K) +
                "," + gemm_kind_name(kind) + "," + std::to_string(threads) + "," + std::to_string(t.median_ms) + "," +
                std::to_string(t.min_ms) + "," + std::to_string(t.max_ms) + "," + std::to_string(gf) + "," +
                std::to_string(diff));
    }
    std::printf(" %6.1fx |\n", last_gf / naive_gf);
}

void tile_sweep(const Shape& s, int runs, bench::Csv& csv, std::mt19937& rng) {
    const std::vector<float> A = random_matrix(static_cast<size_t>(s.M) * s.K, rng);
    const std::vector<float> B = random_matrix(static_cast<size_t>(s.K) * s.N, rng);
    std::vector<float> C(static_cast<size_t>(s.M) * s.N);
    const TileConfig saved = gemm_tiled_config();
    const int mcs[] = {16, 32, 64, 128, 256};
    const int kcs[] = {64, 128, 256, 512};
    const int ncs[] = {64, 128, 256, 512, 1024};
    std::printf("\ntile sweep on %s (M=%d N=%d K=%d), GFLOP/s, median of %d\n", s.name, s.M, s.N, s.K, runs);
    std::printf("| mc | kc | nc=64 | nc=128 | nc=256 | nc=512 | nc=1024 |\n|---:|---:|---:|---:|---:|---:|---:|\n");
    double best = 0;
    TileConfig best_cfg = saved;
    for (int mc : mcs) {
        for (int kc : kcs) {
            std::printf("| %d | %d |", mc, kc);
            for (int nc : ncs) {
                set_gemm_tiled_config({mc, kc, nc});
                const bench::Timing t = bench::time_repeated(
                    [&] { gemm(GemmKind::Tiled, s.M, s.N, s.K, A.data(), s.K, B.data(), s.N, C.data(), s.N); }, 1, runs);
                const double gf = gflops(s, t.median_ms);
                if (gf > best) {
                    best = gf;
                    best_cfg = {mc, kc, nc};
                }
                std::printf(" %.2f |", gf);
                std::fflush(stdout);
                csv.row(std::string("sweep ") + s.name + "," + std::to_string(s.M) + "," + std::to_string(s.N) + "," +
                        std::to_string(s.K) + ",tiled mc=" + std::to_string(mc) + " kc=" + std::to_string(kc) +
                        " nc=" + std::to_string(nc) + ",1," + std::to_string(t.median_ms) + "," +
                        std::to_string(t.min_ms) + "," + std::to_string(t.max_ms) + "," + std::to_string(gf) + ",0");
            }
            std::printf("\n");
        }
    }
    std::printf("best: mc=%d kc=%d nc=%d at %.2f GFLOP/s\n", best_cfg.mc, best_cfg.kc, best_cfg.nc, best);
    set_gemm_tiled_config(saved);
}

}  // namespace

int main(int argc, char** argv) {
    const bench::Args args = bench::Args::parse(argc, argv);
    const int runs = static_cast<int>(args.num("runs", 7));
    const int warmup = static_cast<int>(args.num("warmup", 1));
    const std::string shapes = args.str("shapes", "conv,square");
    std::mt19937 rng(123);

    std::printf("cpu: %s | %s | %s | simd backend: %s | threads: %d | median of %d runs\n\n", cpu_name().c_str(),
                compiler_name().c_str(), build_flags().c_str(), gemm_simd_backend(), gemm_thread_count(), runs);
    bench::Csv csv(args.str("csv", ""), "shape,M,N,K,kind,threads,median_ms,min_ms,max_ms,gflops,max_abs_diff_vs_naive");

    if (args.flag("sweep")) {
        const int sweep_runs = static_cast<int>(args.num("sweep-runs", 3));
        tile_sweep(kSquareShapes[2], sweep_runs, csv, rng);
        tile_sweep(kConvShapes[1], sweep_runs, csv, rng);
        return 0;
    }

    std::printf("| shape (GFLOP/s) |");
    for (int ki = 0; ki < kGemmKindCount; ++ki) std::printf(" %8s |", gemm_kind_name(static_cast<GemmKind>(ki)));
    std::printf(" speedup |\n|---|");
    for (int ki = 0; ki < kGemmKindCount; ++ki) std::printf("---:|");
    std::printf("---:|\n");
    if (shapes.find("conv") != std::string::npos) {
        for (const Shape& s : kConvShapes) bench_shape(s, runs, warmup, csv, rng);
    }
    if (shapes.find("square") != std::string::npos) {
        for (const Shape& s : kSquareShapes) bench_shape(s, runs, warmup, csv, rng);
    }
    return 0;
}
