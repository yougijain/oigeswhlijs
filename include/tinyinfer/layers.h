#pragma once
// The five ops TinyCNN needs: conv2d, relu, maxpool, flatten (free), linear.
//
// Activations are NHWC, [batch, height, width, channels], so that im2col can
// copy whole channel runs and the convolution becomes
//   col[batch*oh*ow][k*k*cin]  x  w[k*k*cin][cout]  ->  y[batch*oh*ow][cout]
// whose output is already NHWC. Weights stay in the exported PyTorch layout
// on disk and are re-packed once at load time.

#include "tinyinfer/gemm.h"
#include "tinyinfer/tensor.h"

namespace tinyinfer {

// Accumulated wall time per stage, in milliseconds. Optional; pass nullptr to skip timing.
struct ConvTimes {
    double im2col_ms = 0;
    double gemm_ms = 0;
    double bias_ms = 0;
};

// Output size for a stride-1 convolution.
inline int conv_out_size(int in, int k, int pad) { return in + 2 * pad - k + 1; }

// im2col for NHWC input, stride 1. x is [batch][h][w][cin]; col is
// [batch*oh*ow][k*k*cin] with column index (kh*k + kw)*cin + c. Padding reads as 0.
void im2col_nhwc(const float* x, int batch, int h, int w, int cin, int k, int pad, float* col);

// im2col written straight into the packed-A layout of gemm_packed_a (gemm.h), so
// the GEMM does not pack. `packed` must hold gemm_packed_a_size(batch*oh*ow, k*k*cin)
// floats. With parallel=true the pixels are split across OpenMP threads.
void im2col_nhwc_packed(const float* x, int batch, int h, int w, int cin, int k, int pad, bool parallel, float* packed);

// Reference convolution in seven plain loops, used only to test the GEMM path.
// w is PyTorch layout [cout][cin][k][k]. y is resized to [batch][oh][ow][cout].
void conv2d_direct_nhwc(const Tensor& x, const Tensor& w_oihw, const Tensor& bias, int pad, Tensor& y);

struct Conv2d {
    int cin = 0, cout = 0, k = 0, pad = 0;
    Tensor w_kn;  // [k*k*cin][cout]
    Tensor bias;  // [cout]

    static Conv2d from_pytorch(const Tensor& w_oihw, const Tensor& bias, int pad);
    // y is resized to [batch][oh][ow][cout]. col is scratch, grown as needed and reused across calls.
    // With relu=true the activation is applied in the same pass as the bias (one less sweep over y).
    void forward(const Tensor& x, Tensor& y, Tensor& col, GemmKind kind, ConvTimes* times = nullptr,
                 bool relu = false) const;
};

void relu_(Tensor& x);

// uint8 NCHW [batch][3][32][32] -> normalised float NHWC [batch][32][32][3]:
// (pixel / 255 - mean[c]) / std[c], the arithmetic of python/model.py normalize().
void preprocess_cifar(const TensorU8& images, const float mean[3], const float stddev[3], Tensor& out);

// 2x2 window, stride 2, floor on odd sizes (PyTorch default). y resized to [batch][h/2][w/2][c].
void maxpool2x2_nhwc(const Tensor& x, Tensor& y);

struct Linear {
    int in = 0, out = 0;
    Tensor w_kn;  // [in][out]
    Tensor bias;  // [out]

    static Linear from_pytorch(const Tensor& w_nk, const Tensor& bias);
    // x is [batch][in]; y resized to [batch][out].
    void forward(const Tensor& x, Tensor& y, GemmKind kind) const;
};

}  // namespace tinyinfer
