// tinyinfer command line.
//
//   tinyinfer parity   --weights W --images I --logits L [--gemm KIND] [--batch N]
//                      [--limit N] [--profile] [--md FILE]
//   tinyinfer parity   --int8 W8 --images I --logits L [--gemm-int8 KIND] ...   (same, INT8 engine)
//   tinyinfer quantize --weights W --calib C --out W8 [--gemm KIND]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinyinfer/loader.h"
#include "tinyinfer/model.h"
#include "tinyinfer/model_int8.h"
#include "tinyinfer/sysinfo.h"

using namespace tinyinfer;

namespace {

struct Args {
    std::string cmd;
    std::map<std::string, std::string> opts;
    bool flag(const std::string& k) const { return opts.count(k) != 0; }
    std::string str(const std::string& k, const std::string& def = "") const {
        auto it = opts.find(k);
        return it == opts.end() ? def : it->second;
    }
    std::string required(const std::string& k) const {
        if (!flag(k)) throw std::runtime_error("missing --" + k);
        return opts.at(k);
    }
    int num(const std::string& k, int def) const {
        auto it = opts.find(k);
        if (it == opts.end()) return def;
        const long v = std::stol(it->second);
        if (v < 0 || v > (1L << 30)) throw std::runtime_error("--" + k + " out of range");
        return static_cast<int>(v);
    }
};

Args parse(int argc, char** argv) {
    Args a;
    if (argc < 2) throw std::runtime_error("no command given");
    a.cmd = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument " + s);
        s = s.substr(2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            a.opts[s] = argv[++i];
        } else {
            a.opts[s] = "1";
        }
    }
    return a;
}

void usage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  tinyinfer parity   --weights W --images I --logits L [--gemm KIND] [--batch N]\n"
                 "                     [--limit N] [--profile] [--md FILE]\n"
                 "  tinyinfer parity   --int8 W8 --images I --logits L [--gemm-int8 KIND] [--batch N]\n"
                 "                     [--limit N] [--profile] [--md FILE]\n"
                 "  tinyinfer quantize --weights W --calib C --out W8 [--gemm KIND]\n"
                 "gemm kinds:");
    for (int i = 0; i < kGemmKindCount; ++i) std::fprintf(stderr, " %s", gemm_kind_name(static_cast<GemmKind>(i)));
    std::fprintf(stderr, " (default %s)\nint8 gemm kinds:", gemm_kind_name(default_gemm_kind()));
    for (int i = 0; i < kGemmInt8KindCount; ++i) std::fprintf(stderr, " %s", gemm_int8_kind_name(static_cast<GemmInt8Kind>(i)));
    std::fprintf(stderr, " (default %s)\n", gemm_int8_kind_name(default_gemm_int8_kind()));
}

int64_t file_size(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    return in ? static_cast<int64_t>(in.tellg()) : -1;
}

TensorU8 slice_images(const TensorU8& images, int start, int count) {
    TensorU8 out({count, images.dim(1), images.dim(2), images.dim(3)});
    const size_t per = static_cast<size_t>(images.dim(1) * images.dim(2) * images.dim(3));
    std::copy(images.data.begin() + static_cast<std::ptrdiff_t>(start * per),
              images.data.begin() + static_cast<std::ptrdiff_t>((start + count) * per), out.data.begin());
    return out;
}

int cmd_parity(const Args& a) {
    const bool use_int8 = a.flag("int8");
    GemmKind kind = default_gemm_kind();
    if (a.flag("gemm") && !parse_gemm_kind(a.str("gemm"), kind)) throw std::runtime_error("unknown gemm kind " + a.str("gemm"));
    GemmInt8Kind kind8 = default_gemm_int8_kind();
    if (a.flag("gemm-int8") && !parse_gemm_int8_kind(a.str("gemm-int8"), kind8)) {
        throw std::runtime_error("unknown int8 gemm kind " + a.str("gemm-int8"));
    }
    const int batch = std::max(1, a.num("batch", 64));

    TinyCNN model;
    TinyCNNInt8 model8;
    if (use_int8) {
        model8 = TinyCNNInt8::load(TensorFile::read(a.str("int8")));
    } else {
        model = TinyCNN::load(TensorFile::read(a.required("weights")));
    }
    const TensorFile imf = TensorFile::read(a.required("images"));
    const TensorFile lf = TensorFile::read(a.required("logits"));
    const TensorU8 images = imf.u8("images");
    const TensorU8 labels = imf.u8("labels");
    const Tensor ref = lf.f32("logits");
    if (ref.dim(0) != images.dim(0) || labels.dim(0) != images.dim(0)) throw std::runtime_error("image/logit/label counts differ");
    const int k = static_cast<int>(ref.dim(1));
    const int total = std::min(a.num("limit", static_cast<int>(images.dim(0))), static_cast<int>(images.dim(0)));
    const std::string engine = use_int8 ? std::string("INT8, gemm ") + gemm_int8_kind_name(kind8) + " (" + gemm_int8_backend() + ")"
                                        : std::string("FP32, gemm ") + gemm_kind_name(kind) + " (" + gemm_simd_backend() + ")";

    Profile prof;
    ProfileInt8 prof8;
    double max_err = 0, err_sum = 0;
    int agree = 0, correct_cpp = 0, correct_ref = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int start = 0; start < total; start += batch) {
        const int b = std::min(batch, total - start);
        const TensorU8 chunk = slice_images(images, start, b);
        const Tensor logits = use_int8 ? model8.forward(chunk, kind8, a.flag("profile") ? &prof8 : nullptr)
                                       : model.forward(chunk, kind, a.flag("profile") ? &prof : nullptr);
        for (int i = 0; i < b; ++i) {
            int arg_cpp = 0, arg_ref = 0;
            for (int c = 0; c < k; ++c) {
                const float v = logits.data[static_cast<size_t>(i) * k + c];
                const float r = ref.data[static_cast<size_t>(start + i) * k + c];
                const double e = std::fabs(static_cast<double>(v) - r);
                max_err = std::max(max_err, e);
                err_sum += e;
                if (v > logits.data[static_cast<size_t>(i) * k + arg_cpp]) arg_cpp = c;
                if (r > ref.data[static_cast<size_t>(start + i) * k + arg_ref]) arg_ref = c;
            }
            const int label = labels.data[static_cast<size_t>(start + i)];
            agree += arg_cpp == arg_ref;
            correct_cpp += arg_cpp == label;
            correct_ref += arg_ref == label;
        }
    }
    const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double mean_err = err_sum / (static_cast<double>(total) * k);
    const std::string profile_table = a.flag("profile") ? (use_int8 ? prof8.table() : prof.table()) : std::string();

    std::printf("engine=%s batch=%d images=%d threads=%d\n", engine.c_str(), batch, total, gemm_thread_count());
    std::printf("max abs error vs PyTorch:  %.3g\n", max_err);
    std::printf("mean abs error vs PyTorch: %.3g\n", mean_err);
    std::printf("argmax agreement:          %d/%d (%.2f%%)\n", agree, total, 100.0 * agree / total);
    std::printf("accuracy C++ / PyTorch:    %.2f%% / %.2f%%\n", 100.0 * correct_cpp / total, 100.0 * correct_ref / total);
    std::printf("wall time:                 %.1f ms total, %.3f ms/image\n", wall_ms, wall_ms / total);
    if (a.flag("profile")) std::printf("\n%s", profile_table.c_str());

    if (a.flag("md")) {
        std::ofstream md(a.str("md"));
        if (!md) throw std::runtime_error("cannot write " + a.str("md"));
        md << (use_int8 ? "# INT8 parity with PyTorch\n\n" : "# M2: parity with PyTorch\n\n")
           << "The C++ " << (use_int8 ? "INT8" : "FP32") << " forward pass was run on the exported CIFAR-10 test images and its\n"
           << "logits compared element-wise with the logits PyTorch (FP32) produced from the same\n"
           << "weights and pixels (`data/export/ref_logits.bin`).\n\n"
           << "| Quantity | Value |\n|---|---:|\n"
           << "| Test images | " << total << " |\n"
           << "| Max abs error vs PyTorch | " << max_err << " |\n"
           << "| Mean abs error vs PyTorch | " << mean_err << " |\n"
           << "| Argmax agreement | " << agree << " / " << total << " (" << 100.0 * agree / total << "%) |\n"
           << "| Accuracy, C++ engine | " << 100.0 * correct_cpp / total << "% |\n"
           << "| Accuracy, PyTorch, same images | " << 100.0 * correct_ref / total << "% |\n"
           << "| Engine | " << engine << " |\n"
           << "| Batch size | " << batch << " |\n"
           << "| Wall time | " << wall_ms << " ms (" << wall_ms / total << " ms/image) |\n\n"
           << "Run on " << cpu_name() << ", " << compiler_name() << ", flags `" << build_flags() << "`, "
           << gemm_thread_count() << " threads.\n";
        if (a.flag("profile")) md << "\n## Where the time goes\n\n" << profile_table;
        std::printf("wrote %s\n", a.str("md").c_str());
    }
    return 0;
}

int cmd_quantize(const Args& a) {
    GemmKind kind = default_gemm_kind();
    if (a.flag("gemm") && !parse_gemm_kind(a.str("gemm"), kind)) throw std::runtime_error("unknown gemm kind " + a.str("gemm"));
    const std::string wpath = a.required("weights");
    const std::string out = a.required("out");
    TinyCNN model = TinyCNN::load(TensorFile::read(wpath));
    const TensorU8 calib = TensorFile::read(a.required("calib")).u8("images");
    TinyCNNInt8 q = TinyCNNInt8::quantize(model, calib, kind);
    q.save(out);
    const int64_t fp32_bytes = file_size(wpath), int8_bytes = file_size(out);
    std::printf("calibration images: %lld\n", static_cast<long long>(calib.dim(0)));
    const char* names[3] = {"conv1", "conv2", "conv3"};
    for (int i = 0; i < 3; ++i) {
        const QConv2d& c = q.conv(i);
        float smin = c.scale_w[0], smax = c.scale_w[0];
        for (float s : c.scale_w) {
            smin = std::min(smin, s);
            smax = std::max(smax, s);
        }
        std::printf("%s: input scale %.6f (max |x| %.3f), weight scales %.6f .. %.6f (%zu channels)\n", names[i],
                    c.scale_in, c.scale_in * 127, smin, smax, c.scale_w.size());
    }
    std::printf("fc: input scale %.6f (max |x| %.3f)\n", q.fc().scale_in, q.fc().scale_in * 127);
    std::printf("weights: fp32 %lld bytes -> int8 %lld bytes (%.2fx smaller); in-memory int8 params+scales %lld bytes\n",
                static_cast<long long>(fp32_bytes), static_cast<long long>(int8_bytes),
                static_cast<double>(fp32_bytes) / static_cast<double>(int8_bytes), static_cast<long long>(q.weight_bytes()));
    std::printf("wrote %s\n", out.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args a = parse(argc, argv);
        if (a.cmd == "parity") return cmd_parity(a);
        if (a.cmd == "quantize") return cmd_quantize(a);
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        usage();
        return 1;
    }
}
