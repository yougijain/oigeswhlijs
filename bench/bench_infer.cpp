// End-to-end inference latency for every GEMM kind at several batch sizes.
// Median of repeated forward passes over the same images.
//
//   bench_infer --weights W --images I [--int8 W8] [--batches 1,16,64] [--runs 10]
//               [--seconds 5] [--csv results/infer.csv]

#include <cstdio>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "bench_util.h"
#include "tinyinfer/loader.h"
#include "tinyinfer/model.h"
#include "tinyinfer/model_int8.h"
#include "tinyinfer/sysinfo.h"

using namespace tinyinfer;

namespace {

std::vector<int> parse_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::stoi(item));
    return out;
}

TensorU8 first_images(const TensorU8& images, int count) {
    TensorU8 out({count, images.dim(1), images.dim(2), images.dim(3)});
    const size_t per = static_cast<size_t>(images.dim(1) * images.dim(2) * images.dim(3));
    std::copy(images.data.begin(), images.data.begin() + static_cast<std::ptrdiff_t>(count * per), out.data.begin());
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const bench::Args args = bench::Args::parse(argc, argv);
    const int runs = static_cast<int>(args.num("runs", 10));
    const double budget_s = static_cast<double>(args.num("seconds", 5));
    const std::vector<int> batches = parse_list(args.str("batches", "1,16,64"));

    const TensorFile wf = TensorFile::read(args.str("weights", "data/export/weights.bin"));
    const TensorFile imf = TensorFile::read(args.str("images", "data/export/test_images.bin"));
    TinyCNN model = TinyCNN::load(wf);
    const TensorU8 images = imf.u8("images");

    std::printf("cpu: %s | %s | %s | simd: %s | int8: %s | threads: %d\n\n", cpu_name().c_str(),
                compiler_name().c_str(), build_flags().c_str(), gemm_simd_backend(), gemm_int8_backend(),
                gemm_thread_count());
    std::printf("| engine | gemm | batch | ms / batch | ms / image | images / s | runs |\n|---|---|---:|---:|---:|---:|---:|\n");
    bench::Csv csv(args.str("csv", ""), "engine,kind,batch,runs,median_ms_per_batch,ms_per_image,images_per_s");

    auto measure = [&](const char* engine, const char* kind_name, int batch, const std::function<void()>& once) {
        // Warm up once, then size the run count so slow kinds stay within the time budget (min 3).
        const bench::Timing probe = bench::time_repeated(once, 1, 1);
        const int n = std::max(3, std::min(runs, static_cast<int>(budget_s * 1000.0 / std::max(probe.median_ms, 1e-3))));
        const bench::Timing t = bench::time_repeated(once, 0, n);
        const double per_image = t.median_ms / batch;
        std::printf("| %s | %s | %d | %.3f | %.4f | %.0f | %d |\n", engine, kind_name, batch, t.median_ms, per_image,
                    1000.0 / per_image, n);
        std::fflush(stdout);
        csv.row(std::string(engine) + "," + kind_name + "," + std::to_string(batch) + "," + std::to_string(n) + "," +
                std::to_string(t.median_ms) + "," + std::to_string(per_image) + "," + std::to_string(1000.0 / per_image));
    };

    for (int ki = 0; ki < kGemmKindCount; ++ki) {
        const GemmKind kind = static_cast<GemmKind>(ki);
        for (int batch : batches) {
            if (batch <= 0 || batch > images.dim(0)) continue;
            const TensorU8 chunk = first_images(images, batch);
            measure("fp32", gemm_kind_name(kind), batch, [&] { bench::keep(model.forward(chunk, kind).data[0]); });
        }
    }
    if (args.flag("int8")) {
        TinyCNNInt8 model8 = TinyCNNInt8::load(TensorFile::read(args.str("int8", "")));
        for (int ki = 0; ki < kGemmInt8KindCount; ++ki) {
            const GemmInt8Kind kind = static_cast<GemmInt8Kind>(ki);
            for (int batch : batches) {
                if (batch <= 0 || batch > images.dim(0)) continue;
                const TensorU8 chunk = first_images(images, batch);
                measure("int8", gemm_int8_kind_name(kind), batch, [&] { bench::keep(model8.forward(chunk, kind).data[0]); });
            }
        }
    }
    return 0;
}
