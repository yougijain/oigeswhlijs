#include "tinyinfer/model_int8.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace tinyinfer {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
const char* const kConvNames[3] = {"conv1", "conv2", "conv3"};
}  // namespace

double ProfileInt8::total_ms() const {
    double t = preprocess_ms + pool_ms + fc_ms;
    for (const QConvTimes& c : conv) t += c.quantize_ms + c.im2col_ms + c.gemm_ms + c.dequant_ms;
    return t;
}

std::string ProfileInt8::table() const {
    const double total = total_ms();
    const double n = images > 0 ? static_cast<double>(images) : 1.0;
    char buf[160];
    std::string s = "| Stage | Total ms | Per image ms | Share |\n|---|---:|---:|---:|\n";
    auto row = [&](const std::string& name, double ms) {
        std::snprintf(buf, sizeof buf, "| %s | %.2f | %.4f | %.1f%% |\n", name.c_str(), ms, ms / n, 100.0 * ms / total);
        s += buf;
    };
    row("preprocess", preprocess_ms);
    for (int i = 0; i < 3; ++i) {
        const std::string c = kConvNames[i];
        row(c + " quantize", conv[i].quantize_ms);
        row(c + " im2col (int8)", conv[i].im2col_ms);
        row(c + " gemm (int8)", conv[i].gemm_ms);
        row(c + " dequant+bias+relu", conv[i].dequant_ms);
    }
    row("maxpool (x3)", pool_ms);
    row("fc (quantize+gemm+dequant)", fc_ms);
    row("total", total);
    return s;
}

TinyCNNInt8 TinyCNNInt8::quantize(TinyCNN& fp32, const TensorU8& calib, GemmKind calib_kind) {
    if (calib.ndim() != 4 || calib.dim(0) < 1) throw std::invalid_argument("calibration set must be [n][3][32][32]");
    LayerInputStats stats;
    const int n = static_cast<int>(calib.dim(0));
    const size_t per = static_cast<size_t>(calib.dim(1) * calib.dim(2) * calib.dim(3));
    for (int start = 0; start < n; start += 100) {
        const int b = std::min(100, n - start);
        TensorU8 chunk({b, calib.dim(1), calib.dim(2), calib.dim(3)});
        std::copy(calib.data.begin() + static_cast<std::ptrdiff_t>(start * per),
                  calib.data.begin() + static_cast<std::ptrdiff_t>((start + b) * per), chunk.data.begin());
        fp32.forward(chunk, calib_kind, nullptr, &stats);
    }
    TinyCNNInt8 m;
    for (int i = 0; i < 3; ++i) {
        if (!(stats.max_abs[i] > 0.0f)) throw std::runtime_error("calibration produced a zero activation range");
        m.convs_[i] = QConv2d::from(fp32.conv(i), stats.max_abs[i] / static_cast<float>(kQMax));
    }
    if (!(stats.max_abs[3] > 0.0f)) throw std::runtime_error("calibration produced a zero activation range");
    m.fc_ = QLinear::from(fp32.fc(), stats.max_abs[3] / static_cast<float>(kQMax));
    for (int c = 0; c < 3; ++c) {
        m.mean_[c] = fp32.mean(c);
        m.std_[c] = fp32.stddev(c);
    }
    return m;
}

std::vector<RawTensor> TinyCNNInt8::to_raw() const {
    std::vector<RawTensor> out;
    auto scalar = [](const std::string& name, float v) {
        Tensor t({1});
        t.data[0] = v;
        return raw_from(name, t);
    };
    for (int i = 0; i < 3; ++i) {
        const std::string c = kConvNames[i];
        const QConv2d& q = convs_[i];
        TensorI32 shape({4});
        shape.data = {q.cin, q.cout, q.k, q.pad};
        out.push_back(raw_from(c + ".shape", shape));
        out.push_back(raw_from(c + ".weight_q", q.wq));
        Tensor sw({static_cast<int64_t>(q.scale_w.size())});
        sw.data.assign(q.scale_w.begin(), q.scale_w.end());
        out.push_back(raw_from(c + ".weight_scale", sw));
        out.push_back(scalar(c + ".input_scale", q.scale_in));
        out.push_back(raw_from(c + ".bias", q.bias));
    }
    out.push_back(raw_from("fc.weight_q", fc_.wq));
    Tensor fsw({static_cast<int64_t>(fc_.scale_w.size())});
    fsw.data.assign(fc_.scale_w.begin(), fc_.scale_w.end());
    out.push_back(raw_from("fc.weight_scale", fsw));
    out.push_back(scalar("fc.input_scale", fc_.scale_in));
    out.push_back(raw_from("fc.bias", fc_.bias));
    Tensor mean({3}), sd({3});
    for (int c = 0; c < 3; ++c) {
        mean.data[static_cast<size_t>(c)] = mean_[c];
        sd.data[static_cast<size_t>(c)] = std_[c];
    }
    out.push_back(raw_from("norm.mean", mean));
    out.push_back(raw_from("norm.std", sd));
    return out;
}

void TinyCNNInt8::save(const std::string& path) const { write_tensor_file(path, to_raw()); }

TinyCNNInt8 TinyCNNInt8::load(const TensorFile& f) {
    TinyCNNInt8 m;
    auto scalar = [&](const std::string& name) {
        const Tensor t = f.f32(name);
        if (t.numel() != 1) throw std::runtime_error(name + " must have one element");
        return t.data[0];
    };
    for (int i = 0; i < 3; ++i) {
        const std::string c = kConvNames[i];
        QConv2d& q = m.convs_[i];
        const TensorI32 shape = f.i32(c + ".shape");
        if (shape.numel() != 4) throw std::runtime_error(c + ".shape must have 4 entries");
        q.cin = shape.data[0];
        q.cout = shape.data[1];
        q.k = shape.data[2];
        q.pad = shape.data[3];
        if (q.cin <= 0 || q.cout <= 0 || q.k <= 0 || q.pad < 0) throw std::runtime_error(c + ".shape is invalid");
        q.wq = f.i8(c + ".weight_q");
        if (q.wq.ndim() != 2 || q.wq.dim(0) != static_cast<int64_t>(q.k) * q.k * q.cin || q.wq.dim(1) != q.cout) {
            throw std::runtime_error(c + ".weight_q has shape " + shape_str(q.wq.shape));
        }
        const Tensor sw = f.f32(c + ".weight_scale");
        q.scale_w.assign(sw.data.begin(), sw.data.end());
        if (q.scale_w.size() != static_cast<size_t>(q.cout)) throw std::runtime_error(c + ".weight_scale size");
        q.scale_in = scalar(c + ".input_scale");
        if (!(q.scale_in > 0.0f)) throw std::runtime_error(c + ".input_scale must be positive");
        q.bias = f.f32(c + ".bias");
        if (q.bias.numel() != q.cout) throw std::runtime_error(c + ".bias size");
    }
    m.fc_.wq = f.i8("fc.weight_q");
    if (m.fc_.wq.ndim() != 2) throw std::runtime_error("fc.weight_q must be 2-D");
    m.fc_.in = static_cast<int>(m.fc_.wq.dim(0));
    m.fc_.out = static_cast<int>(m.fc_.wq.dim(1));
    const Tensor fsw = f.f32("fc.weight_scale");
    m.fc_.scale_w.assign(fsw.data.begin(), fsw.data.end());
    if (m.fc_.scale_w.size() != static_cast<size_t>(m.fc_.out)) throw std::runtime_error("fc.weight_scale size");
    m.fc_.scale_in = scalar("fc.input_scale");
    if (!(m.fc_.scale_in > 0.0f)) throw std::runtime_error("fc.input_scale must be positive");
    m.fc_.bias = f.f32("fc.bias");
    if (m.fc_.bias.numel() != m.fc_.out) throw std::runtime_error("fc.bias size");
    const Tensor mean = f.f32("norm.mean"), sd = f.f32("norm.std");
    if (mean.numel() != 3 || sd.numel() != 3) throw std::runtime_error("norm.mean/norm.std must have 3 entries");
    for (int c = 0; c < 3; ++c) {
        m.mean_[c] = mean.data[static_cast<size_t>(c)];
        m.std_[c] = sd.data[static_cast<size_t>(c)];
        if (!(m.std_[c] > 0.0f)) throw std::runtime_error("norm.std must be positive");
    }
    // Topology checks, as for the float model.
    if (m.convs_[0].cin != TinyCNN::kChannels) throw std::runtime_error("conv1 input channels");
    for (int i = 1; i < 3; ++i) {
        if (m.convs_[i].cin != m.convs_[i - 1].cout) throw std::runtime_error(std::string(kConvNames[i]) + " does not chain");
    }
    const int hw = TinyCNN::kImageSize / 8;
    if (m.fc_.in != m.convs_[2].cout * hw * hw || m.fc_.out != TinyCNN::kClasses) throw std::runtime_error("fc shape");
    return m;
}

Tensor TinyCNNInt8::forward(const TensorU8& images, GemmInt8Kind kind, ProfileInt8* prof) {
    auto t0 = Clock::now();
    preprocess_cifar(images, mean_, std_, x_);
    if (prof) prof->preprocess_ms += ms_since(t0);

    for (int i = 0; i < 3; ++i) {
        convs_[i].forward(x_, y_, xq_, col_, acc_, kind, prof ? &prof->conv[i] : nullptr, /*relu=*/true);
        t0 = Clock::now();
        maxpool2x2_nhwc(y_, x_);
        if (prof) prof->pool_ms += ms_since(t0);
    }

    t0 = Clock::now();
    const int batch = static_cast<int>(x_.dim(0));
    flat_.resize({batch, x_.numel() / batch});
    flat_.data = x_.data;
    fc_.forward(flat_, logits_, xq_, acc_, kind);
    if (prof) {
        prof->fc_ms += ms_since(t0);
        prof->images += batch;
    }
    return logits_;
}

int64_t TinyCNNInt8::weight_bytes() const {
    int64_t n = 0;
    for (const QConv2d& c : convs_) n += c.weight_bytes() + c.bias.numel() * 4;
    return n + fc_.weight_bytes() + fc_.bias.numel() * 4;
}

}  // namespace tinyinfer
