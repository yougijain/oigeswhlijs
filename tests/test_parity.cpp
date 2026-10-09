// End-to-end parity against PyTorch: run the exported test images through the
// C++ engine and compare logits with the ones PyTorch produced for the same
// weights and pixels. Usage: test_parity <data/export dir>

#include <cmath>
#include <cstdio>

#include "check.h"
#include "tinyinfer/loader.h"
#include "tinyinfer/model.h"

using namespace tinyinfer;

namespace {

struct ParityStats {
    double max_abs_err = 0;
    double mean_abs_err = 0;
    int agree = 0;
    int correct_cpp = 0;
    int correct_ref = 0;
    int n = 0;
};

ParityStats run(TinyCNN& model, const TensorU8& images, const TensorU8& labels, const Tensor& ref, GemmKind kind,
                int limit, int batch) {
    ParityStats s;
    const int total = std::min(limit, static_cast<int>(images.dim(0)));
    const int k = static_cast<int>(ref.dim(1));
    const size_t img_bytes = static_cast<size_t>(images.dim(1) * images.dim(2) * images.dim(3));
    double err_sum = 0;
    for (int start = 0; start < total; start += batch) {
        const int b = std::min(batch, total - start);
        TensorU8 chunk({b, images.dim(1), images.dim(2), images.dim(3)});
        std::copy(images.data.begin() + static_cast<std::ptrdiff_t>(start * img_bytes),
                  images.data.begin() + static_cast<std::ptrdiff_t>((start + b) * img_bytes), chunk.data.begin());
        const Tensor logits = model.forward(chunk, kind);
        CHECK(logits.shape == std::vector<int64_t>({b, k}));
        for (int i = 0; i < b; ++i) {
            int arg_cpp = 0, arg_ref = 0;
            for (int c = 0; c < k; ++c) {
                const float v = logits.data[static_cast<size_t>(i) * k + c];
                const float r = ref.data[static_cast<size_t>(start + i) * k + c];
                CHECK(std::isfinite(v));
                const double e = std::fabs(static_cast<double>(v) - r);
                s.max_abs_err = std::max(s.max_abs_err, e);
                err_sum += e;
                if (v > logits.data[static_cast<size_t>(i) * k + arg_cpp]) arg_cpp = c;
                if (r > ref.data[static_cast<size_t>(start + i) * k + arg_ref]) arg_ref = c;
            }
            const int label = labels.data[static_cast<size_t>(start + i)];
            s.agree += arg_cpp == arg_ref;
            s.correct_cpp += arg_cpp == label;
            s.correct_ref += arg_ref == label;
        }
    }
    s.n = total;
    s.mean_abs_err = err_sum / (static_cast<double>(total) * k);
    return s;
}

}  // namespace

TEST(logits_match_pytorch) {
    if (test_args().empty()) {
        ::tinytest::fail(__FILE__, __LINE__, "usage: test_parity <data/export dir>");
        return;
    }
    const std::string dir = test_args()[0];
    const TensorFile wf = TensorFile::read(dir + "/weights.bin");
    const TensorFile imf = TensorFile::read(dir + "/test_images.bin");
    const TensorFile lf = TensorFile::read(dir + "/ref_logits.bin");
    TinyCNN model = TinyCNN::load(wf);
    CHECK_EQ(model.param_count(), 113738);
    const TensorU8 images = imf.u8("images");
    const TensorU8 labels = imf.u8("labels");
    const Tensor ref = lf.f32("logits");
    CHECK_EQ(ref.dim(0), images.dim(0));
    CHECK_EQ(labels.dim(0), images.dim(0));

    for (int ki = 0; ki < kGemmKindCount; ++ki) {
        const GemmKind kind = static_cast<GemmKind>(ki);
        // Every kind runs on a slice; the default kind runs on everything.
        const int limit = kind == default_gemm_kind() ? static_cast<int>(images.dim(0)) : 128;
        const ParityStats s = run(model, images, labels, ref, kind, limit, 64);
        std::printf("  %-10s n=%d max_abs_err=%.3g mean_abs_err=%.3g argmax_agree=%d/%d acc_cpp=%.2f%% acc_ref=%.2f%%\n",
                    gemm_kind_name(kind), s.n, s.max_abs_err, s.mean_abs_err, s.agree, s.n,
                    100.0 * s.correct_cpp / s.n, 100.0 * s.correct_ref / s.n);
        // float32 summation-order noise is ~1e-5 on logits of magnitude ~10; anything near 1e-3 is a bug.
        CHECK(s.max_abs_err < 1e-4);
        // A logit gap smaller than the error above would flip an argmax. Allow 1 in 500, report the rest.
        CHECK(s.agree >= s.n - (s.n / 500));
    }
}
