// Peak floating-point throughput: a register-only FMA loop with enough
// independent accumulators to cover the FMA latency, run on one core and on
// every core. This is the vertical (compute) ceiling of the roofline. No
// memory traffic, so it measures the arithmetic units and nothing else.
//
//   bench_peak [--iters 200000000] [--runs 5] [--csv results/peak.csv]

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "bench_util.h"
#include "tinyinfer/sysinfo.h"

namespace {

// Independent accumulator chains per kernel. Enough to cover FMA latency x
// issue width (4 x 2 = 8 on Intel, 4 x 4 = 16 on Apple M-series), but never
// more than the register file holds with the two constants: AVX2 has 16 ymm
// registers, so 12 chains; AVX-512 and NEON have 32, so 16 and 24. (A first
// version used 16 everywhere; the AVX2 loop spilled and measured below the
// GEMM micro-kernel it was supposed to bound.)
[[maybe_unused]] constexpr int kAccScalar = 8;
[[maybe_unused]] constexpr int kAccAvx2 = 12;
[[maybe_unused]] constexpr int kAccAvx512 = 16;
[[maybe_unused]] constexpr int kAccNeon = 24;

struct Kernel {
    const char* name;
    int lanes;
    int acc;
    void (*run)(int64_t iters, float* sink);
};

// Each kernel: acc accumulators, each doing a = a * b + c per iteration.
// Two things the compiler must be told at -O2, or the loop measures the store
// port instead of the FMA units: the chain loop has to be fully unrolled so
// the accumulator array lives in registers, and the scalar loop must not be
// auto-vectorised (GCC turned it into AVX-512 and reported 22 GFLOP/s).
#if defined(__clang__)
#define TINYINFER_NO_VECTORIZE _Pragma("clang loop vectorize(disable)")
#define TINYINFER_UNROLL _Pragma("clang loop unroll(full)")
#define TINYINFER_SCALAR_ATTR
#elif defined(__GNUC__)
#define TINYINFER_NO_VECTORIZE
#define TINYINFER_UNROLL _Pragma("GCC unroll 32")
#define TINYINFER_SCALAR_ATTR __attribute__((optimize("no-tree-vectorize")))
#else
#define TINYINFER_NO_VECTORIZE
#define TINYINFER_UNROLL
#define TINYINFER_SCALAR_ATTR
#endif
TINYINFER_SCALAR_ATTR void scalar_fma(int64_t iters, float* sink) {
    float acc[kAccScalar];
    for (int i = 0; i < kAccScalar; ++i) acc[i] = 1.0f + 0.001f * static_cast<float>(i);
    const float b = 1.000001f, c = 0.000001f;
    for (int64_t it = 0; it < iters; ++it) {
        TINYINFER_NO_VECTORIZE
        TINYINFER_UNROLL
        for (int i = 0; i < kAccScalar; ++i) acc[i] = acc[i] * b + c;
    }
    float s = 0;
    for (int i = 0; i < kAccScalar; ++i) s += acc[i];
    *sink = s;
}

#if defined(__AVX2__) && defined(__FMA__)
void avx2_fma(int64_t iters, float* sink) {
    __m256 acc[kAccAvx2];
    for (int i = 0; i < kAccAvx2; ++i) acc[i] = _mm256_set1_ps(1.0f + 0.001f * static_cast<float>(i));
    const __m256 b = _mm256_set1_ps(1.000001f), c = _mm256_set1_ps(0.000001f);
    for (int64_t it = 0; it < iters; ++it) {
        TINYINFER_UNROLL
        for (int i = 0; i < kAccAvx2; ++i) acc[i] = _mm256_fmadd_ps(acc[i], b, c);
    }
    __m256 s = acc[0];
    for (int i = 1; i < kAccAvx2; ++i) s = _mm256_add_ps(s, acc[i]);
    float out[8];
    _mm256_storeu_ps(out, s);
    *sink = out[0] + out[7];
}
#endif

#if defined(__AVX512F__)
void avx512_fma(int64_t iters, float* sink) {
    __m512 acc[kAccAvx512];
    for (int i = 0; i < kAccAvx512; ++i) acc[i] = _mm512_set1_ps(1.0f + 0.001f * static_cast<float>(i));
    const __m512 b = _mm512_set1_ps(1.000001f), c = _mm512_set1_ps(0.000001f);
    for (int64_t it = 0; it < iters; ++it) {
        TINYINFER_UNROLL
        for (int i = 0; i < kAccAvx512; ++i) acc[i] = _mm512_fmadd_ps(acc[i], b, c);
    }
    __m512 s = acc[0];
    for (int i = 1; i < kAccAvx512; ++i) s = _mm512_add_ps(s, acc[i]);
    *sink = _mm512_reduce_add_ps(s);
}
#endif

#if defined(__ARM_NEON)
void neon_fma(int64_t iters, float* sink) {
    float32x4_t acc[kAccNeon];
    for (int i = 0; i < kAccNeon; ++i) acc[i] = vdupq_n_f32(1.0f + 0.001f * static_cast<float>(i));
    const float32x4_t b = vdupq_n_f32(1.000001f), c = vdupq_n_f32(0.000001f);
    for (int64_t it = 0; it < iters; ++it) {
        TINYINFER_UNROLL
        for (int i = 0; i < kAccNeon; ++i) acc[i] = vfmaq_f32(c, acc[i], b);
    }
    float32x4_t s = acc[0];
    for (int i = 1; i < kAccNeon; ++i) s = vaddq_f32(s, acc[i]);
    *sink = vaddvq_f32(s);
}
#endif

}  // namespace

int main(int argc, char** argv) {
    const bench::Args args = bench::Args::parse(argc, argv);
    const int64_t iters = args.num("iters", 200000000);
    const int runs = static_cast<int>(args.num("runs", 5));

    std::vector<Kernel> kernels = {{"scalar", 1, kAccScalar, scalar_fma}};
#if defined(__ARM_NEON)
    kernels.push_back({"neon", 4, kAccNeon, neon_fma});
#endif
#if defined(__AVX2__) && defined(__FMA__)
    kernels.push_back({"avx2", 8, kAccAvx2, avx2_fma});
#endif
#if defined(__AVX512F__)
    kernels.push_back({"avx512", 16, kAccAvx512, avx512_fma});
#endif

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::printf("iters: %lld, runs: %d, threads: %d, cpu: %s\n", static_cast<long long>(iters), runs, threads,
                tinyinfer::cpu_name().c_str());
    bench::Csv csv(args.str("csv", ""), "kernel,lanes,accumulators,threads,median_ms,gflops");

    for (const Kernel& k : kernels) {
        // 2 FLOPs per lane per FMA.
        const double flops_1t = 2.0 * k.lanes * k.acc * static_cast<double>(iters);
        float sink = 0;
        bench::Timing t1 = bench::time_repeated([&] { k.run(iters, &sink); bench::keep(sink); }, 1, runs);
        const double g1 = flops_1t / (t1.median_ms * 1e-3) / 1e9;
        std::printf("%-7s x%-2d (%2d acc)  1 thread   %8.2f ms  %8.2f GFLOP/s\n", k.name, k.lanes, k.acc, t1.median_ms, g1);
        csv.row(std::string(k.name) + "," + std::to_string(k.lanes) + "," + std::to_string(k.acc) + ",1," +
                std::to_string(t1.median_ms) + "," + std::to_string(g1));
#ifdef _OPENMP
        if (threads > 1) {
            std::vector<float> sinks(static_cast<size_t>(threads));
            bench::Timing tn = bench::time_repeated(
                [&] {
#pragma omp parallel
                    {
                        k.run(iters, &sinks[static_cast<size_t>(omp_get_thread_num())]);
                    }
                    bench::keep(sinks[0]);
                },
                1, runs);
            const double gn = flops_1t * threads / (tn.median_ms * 1e-3) / 1e9;
            std::printf("%-7s x%-2d (%2d acc) %2d threads  %8.2f ms  %8.2f GFLOP/s\n", k.name, k.lanes, k.acc, threads,
                        tn.median_ms, gn);
            csv.row(std::string(k.name) + "," + std::to_string(k.lanes) + "," + std::to_string(k.acc) + "," +
                    std::to_string(threads) + "," + std::to_string(tn.median_ms) + "," + std::to_string(gn));
        }
#endif
    }
    return 0;
}
