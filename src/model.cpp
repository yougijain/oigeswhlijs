#include "tinyinfer/model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace tinyinfer {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
}  // namespace

double Profile::total_ms() const {
    double t = preprocess_ms + pool_ms + fc_ms;
    for (const ConvTimes& c : conv) t += c.im2col_ms + c.gemm_ms + c.bias_ms;
    return t;
}

std::string Profile::table() const {
    const double total = total_ms();
    const double n = images > 0 ? static_cast<double>(images) : 1.0;
    char buf[160];
    std::string s = "| Stage | Total ms | Per image ms | Share |\n|---|---:|---:|---:|\n";
    auto row = [&](const char* name, double ms) {
        std::snprintf(buf, sizeof buf, "| %s | %.2f | %.4f | %.1f%% |\n", name, ms, ms / n, 100.0 * ms / total);
        s += buf;
    };
    row("preprocess", preprocess_ms);
    const char* names[3][3] = {{"conv1 im2col", "conv1 gemm", "conv1 bias+relu"},
                               {"conv2 im2col", "conv2 gemm", "conv2 bias+relu"},
                               {"conv3 im2col", "conv3 gemm", "conv3 bias+relu"}};
    for (int i = 0; i < 3; ++i) {
        row(names[i][0], conv[i].im2col_ms);
        row(names[i][1], conv[i].gemm_ms);
        row(names[i][2], conv[i].bias_ms);
    }
    row("maxpool (x3)", pool_ms);
    row("fc", fc_ms);
    row("total", total);
    return s;
}

TinyCNN TinyCNN::load(const TensorFile& f) {
    TinyCNN m;
    const char* names[3] = {"conv1", "conv2", "conv3"};
    for (int i = 0; i < 3; ++i) {
        const std::string n = names[i];
        m.convs_[i] = Conv2d::from_pytorch(f.f32(n + ".weight"), f.f32(n + ".bias"), 1);
    }
    m.fc_ = Linear::from_pytorch(f.f32("fc.weight"), f.f32("fc.bias"));
    const Tensor mean = f.f32("norm.mean"), sd = f.f32("norm.std");
    if (mean.numel() != 3 || sd.numel() != 3) throw std::runtime_error("norm.mean/norm.std must have 3 entries");
    for (int c = 0; c < 3; ++c) {
        m.mean_[c] = mean.data[static_cast<size_t>(c)];
        m.std_[c] = sd.data[static_cast<size_t>(c)];
        if (m.std_[c] <= 0) throw std::runtime_error("norm.std must be positive");
    }
    // Shape checks: the engine is specialised to this topology.
    if (m.convs_[0].cin != kChannels || m.convs_[0].k != 3) throw std::runtime_error("conv1 shape unexpected");
    for (int i = 1; i < 3; ++i) {
        if (m.convs_[i].cin != m.convs_[i - 1].cout || m.convs_[i].k != 3) {
            throw std::runtime_error(std::string(names[i]) + " shape does not chain");
        }
    }
    const int final_hw = kImageSize / 8;
    if (m.fc_.in != m.convs_[2].cout * final_hw * final_hw || m.fc_.out != kClasses) {
        throw std::runtime_error("fc shape does not match conv3 output");
    }
    // PyTorch flattens NCHW, index c*16 + h*4 + w. We flatten NHWC, index (h*4 + w)*C + c.
    // Permute the rows of the [in][out] weight so both orders give the same logits.
    const int c3 = m.convs_[2].cout;
    Tensor permuted({m.fc_.in, m.fc_.out});
    for (int c = 0; c < c3; ++c) {
        for (int h = 0; h < final_hw; ++h) {
            for (int w = 0; w < final_hw; ++w) {
                const size_t src_row = (static_cast<size_t>(c) * final_hw + h) * final_hw + w;
                const size_t dst_row = (static_cast<size_t>(h) * final_hw + w) * c3 + c;
                for (int n = 0; n < m.fc_.out; ++n) {
                    permuted.data[dst_row * m.fc_.out + n] = m.fc_.w_kn.data[src_row * m.fc_.out + n];
                }
            }
        }
    }
    m.fc_.w_kn = std::move(permuted);
    return m;
}

void TinyCNN::preprocess(const TensorU8& images, Tensor& out) const { preprocess_cifar(images, mean_, std_, out); }

namespace {
float max_abs_of(const Tensor& t) {
    float m = 0.0f;
    for (float v : t.data) m = std::max(m, std::fabs(v));
    return m;
}
}  // namespace

Tensor TinyCNN::forward(const TensorU8& images, GemmKind kind, Profile* prof, LayerInputStats* stats) {
    auto t0 = Clock::now();
    preprocess(images, x_);
    if (prof) prof->preprocess_ms += ms_since(t0);

    for (int i = 0; i < 3; ++i) {
        if (stats) stats->max_abs[i] = std::max(stats->max_abs[i], max_abs_of(x_));
        convs_[i].forward(x_, y_, col_, kind, prof ? &prof->conv[i] : nullptr, /*relu=*/true);
        t0 = Clock::now();
        maxpool2x2_nhwc(y_, x_);
        if (prof) prof->pool_ms += ms_since(t0);
    }

    t0 = Clock::now();
    if (stats) stats->max_abs[3] = std::max(stats->max_abs[3], max_abs_of(x_));
    const int batch = static_cast<int>(x_.dim(0));
    flat_.resize({batch, x_.numel() / batch});
    flat_.data = x_.data;  // flatten is a copy of the contiguous buffer under a 2-D shape
    fc_.forward(flat_, logits_, kind);
    if (prof) {
        prof->fc_ms += ms_since(t0);
        prof->images += batch;
    }
    return logits_;
}

int64_t TinyCNN::param_count() const {
    int64_t n = 0;
    for (const Conv2d& c : convs_) n += c.w_kn.numel() + c.bias.numel();
    return n + fc_.w_kn.numel() + fc_.bias.numel();
}

}  // namespace tinyinfer
