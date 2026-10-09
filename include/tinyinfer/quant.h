#pragma once
// INT8 inference pieces.
//
// Scheme: symmetric quantisation everywhere (no zero points).
//   weights      per output channel: wq[k][n] = round(w[k][n] / sw[n]), sw[n] = max_k |w[k][n]| / 127
//   activations  per tensor, static: xq = clamp(round(x / sa), -127, 127), sa calibrated once
//                as max |x| over a calibration set, divided by 127
//   compute      int8 x int8 products accumulated in int32, then
//                y[m][n] = acc[m][n] * (sa * sw[n]) + bias[n]  in float
// Per-channel weight scales matter because output channels of a conv differ in
// magnitude by 10x or more; one scale for all of them would waste most of the
// 8 bits on the small channels.

#include <cstdint>
#include <vector>

#include "tinyinfer/gemm_int8.h"
#include "tinyinfer/layers.h"
#include "tinyinfer/tensor.h"

namespace tinyinfer {

constexpr int kQMax = 127;

// Symmetric per-column quantisation of a [K][N] matrix. scale[n] = max_k |w[k][n]| / 127.
struct QuantizedMatrix {
    TensorI8 q;                // [K][N]
    std::vector<float> scale;  // [N]
};
QuantizedMatrix quantize_per_column(const Tensor& w_kn);

// Largest-magnitude entry, the input to activation scale calibration.
float max_abs(const float* x, int64_t n);

// xq = clamp(round(x / scale), -127, 127), round half to even.
void quantize_activations(const float* x, int64_t n, float scale, int8_t* out);

// int8 im2col with the same layout as im2col_nhwc. Zero padding is exact in a symmetric scheme.
void im2col_nhwc_i8(const int8_t* x, int batch, int h, int w, int cin, int k, int pad, int8_t* col);

// int8 im2col straight into the packed-A layout of gemm_s8_packed_a (gemm_int8.h).
// `packed` must hold gemm_int8_packed_a_size(batch*oh*ow, k*k*cin) bytes.
void im2col_nhwc_i8_packed(const int8_t* x, int batch, int h, int w, int cin, int k, int pad, bool parallel,
                           uint8_t* packed);

struct QConvTimes {
    double quantize_ms = 0;
    double im2col_ms = 0;
    double gemm_ms = 0;
    double dequant_ms = 0;
};

struct QConv2d {
    int cin = 0, cout = 0, k = 0, pad = 0;
    TensorI8 wq;                 // [k*k*cin][cout]
    std::vector<float> scale_w;  // [cout]
    float scale_in = 1.0f;
    Tensor bias;                 // [cout]

    static QConv2d from(const Conv2d& c, float scale_in);
    // x float NHWC in, y float NHWC out. xq, col, acc are scratch. relu fuses the activation
    // into the dequantisation pass.
    void forward(const Tensor& x, Tensor& y, TensorI8& xq, TensorI8& col, TensorI32& acc, GemmInt8Kind kind,
                 QConvTimes* times = nullptr, bool relu = false) const;
    int64_t weight_bytes() const { return wq.numel() + static_cast<int64_t>(scale_w.size()) * 4 + 4; }
};

struct QLinear {
    int in = 0, out = 0;
    TensorI8 wq;  // [in][out]
    std::vector<float> scale_w;
    float scale_in = 1.0f;
    Tensor bias;

    static QLinear from(const Linear& l, float scale_in);
    void forward(const Tensor& x, Tensor& y, TensorI8& xq, TensorI32& acc, GemmInt8Kind kind) const;
    int64_t weight_bytes() const { return wq.numel() + static_cast<int64_t>(scale_w.size()) * 4 + 4; }
};

}  // namespace tinyinfer
