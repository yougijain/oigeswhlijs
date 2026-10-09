// Memory bandwidth: STREAM-style copy and triad over arrays far larger than
// the last-level cache, single-threaded and with every core. This is the
// horizontal (memory) ceiling of the roofline.
//
//   bench_bandwidth [--mb 512] [--runs 10] [--csv results/bandwidth.csv]

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bench_util.h"
#include "tinyinfer/sysinfo.h"

namespace {

void copy_st(float* dst, const float* src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}
void triad_st(float* a, const float* b, const float* c, float s, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = b[i] + s * c[i];
}

#ifdef _OPENMP
void copy_mt(float* dst, const float* src, size_t n) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) dst[i] = src[i];
}
void triad_mt(float* a, const float* b, const float* c, float s, size_t n) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) a[i] = b[i] + s * c[i];
}
#endif

}  // namespace

int main(int argc, char** argv) {
    const bench::Args args = bench::Args::parse(argc, argv);
    const size_t mb = static_cast<size_t>(args.num("mb", 512));
    const int runs = static_cast<int>(args.num("runs", 10));
    const size_t n = mb * 1024 * 1024 / sizeof(float);

    std::vector<float> a(n), b(n), c(n);
    for (size_t i = 0; i < n; ++i) {  // first touch, so pages are mapped before timing
        a[i] = 1.0f;
        b[i] = static_cast<float>(i & 1023);
        c[i] = 0.5f;
    }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    std::printf("arrays: 3 x %zu MiB, runs: %d, threads: %d, cpu: %s\n", mb, runs, threads, tinyinfer::cpu_name().c_str());
    bench::Csv csv(args.str("csv", ""), "kernel,threads,array_mib,median_ms,gbps");

    auto report = [&](const char* name, int th, double bytes, const bench::Timing& t) {
        const double gbps = bytes / (t.median_ms * 1e-3) / 1e9;
        std::printf("%-8s %2d thread%s  median %8.2f ms  %7.2f GB/s  (min %.2f, max %.2f)\n", name, th,
                    th == 1 ? " " : "s", t.median_ms, gbps, t.min_ms, t.max_ms);
        csv.row(std::string(name) + "," + std::to_string(th) + "," + std::to_string(mb) + "," +
                std::to_string(t.median_ms) + "," + std::to_string(gbps));
    };

    const double copy_bytes = 2.0 * static_cast<double>(n) * sizeof(float);
    const double triad_bytes = 3.0 * static_cast<double>(n) * sizeof(float);
    report("copy", 1, copy_bytes, bench::time_repeated([&] { copy_st(a.data(), b.data(), n); bench::keep(a[n / 2]); }, 1, runs));
    report("triad", 1, triad_bytes, bench::time_repeated([&] { triad_st(a.data(), b.data(), c.data(), 3.0f, n); bench::keep(a[n / 2]); }, 1, runs));
#ifdef _OPENMP
    if (threads > 1) {
        report("copy", threads, copy_bytes, bench::time_repeated([&] { copy_mt(a.data(), b.data(), n); bench::keep(a[n / 2]); }, 1, runs));
        report("triad", threads, triad_bytes, bench::time_repeated([&] { triad_mt(a.data(), b.data(), c.data(), 3.0f, n); bench::keep(a[n / 2]); }, 1, runs));
    }
#endif
    return 0;
}
