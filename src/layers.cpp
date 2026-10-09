#include "tinyinfer/layers.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace tinyinfer {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void check_rank(const Tensor& t, size_t rank, const char* what) {
    if (t.ndim() != rank) {
        throw std::invalid_argument(std::string(what) + ": expected rank " + std::to_string(rank) + ", got " +
                                    shape_str(t.shape));
    }
}

}  // namespace

void im2col_nhwc(const float* x, int batch, int h, int w, int cin, int k, int pad, float* col) {
    const int oh = conv_out_size(h, k, pad);
    const int ow = conv_out_size(w, k, pad);
    const size_t run = static_cast<size_t>(cin) * sizeof(float);
    float* out = col;
    for (int b = 0; b < batch; ++b) {
        const float* img = x + static_cast<size_t>(b) * h * w * cin;
        for (int oy = 0; oy < oh; ++oy) {
            for (int ox = 0; ox < ow; ++ox) {
                // One output pixel: k rows of k taps, each tap a run of cin floats. In NHWC the
                // k taps of one row are contiguous in the input, so an interior pixel copies
                // k*cin floats per row in one go; only border pixels go tap by tap.
                const int ix0 = ox - pad;
                const bool row_inside = ix0 >= 0 && ix0 + k <= w;
                for (int ky = 0; ky < k; ++ky) {
                    const int iy = oy + ky - pad;
                    if (iy < 0 || iy >= h) {
                        std::memset(out, 0, run * static_cast<size_t>(k));
                    } else if (row_inside) {
                        std::memcpy(out, img + (static_cast<size_t>(iy) * w + ix0) * cin, run * static_cast<size_t>(k));
                    } else {
                        for (int kx = 0; kx < k; ++kx) {
                            const int ix = ix0 + kx;
                            if (ix < 0 || ix >= w) {
                                std::memset(out + static_cast<size_t>(kx) * cin, 0, run);
                            } else {
                                std::memcpy(out + static_cast<size_t>(kx) * cin, img + (static_cast<size_t>(iy) * w + ix) * cin, run);
                            }
                        }
                    }
                    out += static_cast<size_t>(k) * cin;
                }
            }
        }
    }
}

void conv2d_direct_nhwc(const Tensor& x, const Tensor& w_oihw, const Tensor& bias, int pad, Tensor& y) {
    check_rank(x, 4, "conv2d_direct input");
    check_rank(w_oihw, 4, "conv2d_direct weight");
    const int batch = static_cast<int>(x.dim(0)), h = static_cast<int>(x.dim(1)), w = static_cast<int>(x.dim(2));
    const int cin = static_cast<int>(x.dim(3));
    const int cout = static_cast<int>(w_oihw.dim(0)), k = static_cast<int>(w_oihw.dim(2));
    if (w_oihw.dim(1) != cin || w_oihw.dim(3) != k) throw std::invalid_argument("conv2d_direct: weight/input mismatch");
    if (bias.numel() != cout) throw std::invalid_argument("conv2d_direct: bias size");
    const int oh = conv_out_size(h, k, pad), ow = conv_out_size(w, k, pad);
    y.resize({batch, oh, ow, cout});
    for (int b = 0; b < batch; ++b) {
        for (int oy = 0; oy < oh; ++oy) {
            for (int ox = 0; ox < ow; ++ox) {
                for (int co = 0; co < cout; ++co) {
                    float acc = bias.data[static_cast<size_t>(co)];
                    for (int ci = 0; ci < cin; ++ci) {
                        for (int ky = 0; ky < k; ++ky) {
                            const int iy = oy + ky - pad;
                            if (iy < 0 || iy >= h) continue;
                            for (int kx = 0; kx < k; ++kx) {
                                const int ix = ox + kx - pad;
                                if (ix < 0 || ix >= w) continue;
                                const float xv = x.data[((static_cast<size_t>(b) * h + iy) * w + ix) * cin + ci];
                                const float wv = w_oihw.data[((static_cast<size_t>(co) * cin + ci) * k + ky) * k + kx];
                                acc += xv * wv;
                            }
                        }
                    }
                    y.data[((static_cast<size_t>(b) * oh + oy) * ow + ox) * cout + co] = acc;
                }
            }
        }
    }
}

Conv2d Conv2d::from_pytorch(const Tensor& w_oihw, const Tensor& b, int pad) {
    check_rank(w_oihw, 4, "conv weight");
    Conv2d c;
    c.cout = static_cast<int>(w_oihw.dim(0));
    c.cin = static_cast<int>(w_oihw.dim(1));
    c.k = static_cast<int>(w_oihw.dim(2));
    c.pad = pad;
    if (w_oihw.dim(3) != c.k) throw std::invalid_argument("conv weight must be square: " + shape_str(w_oihw.shape));
    if (b.numel() != c.cout) throw std::invalid_argument("conv bias size does not match out channels");
    // [cout][cin][k][k] -> [(ky*k + kx)*cin + ci][cout], the column order im2col produces.
    c.w_kn = Tensor({static_cast<int64_t>(c.k) * c.k * c.cin, c.cout});
    for (int co = 0; co < c.cout; ++co) {
        for (int ci = 0; ci < c.cin; ++ci) {
            for (int ky = 0; ky < c.k; ++ky) {
                for (int kx = 0; kx < c.k; ++kx) {
                    const size_t src = ((static_cast<size_t>(co) * c.cin + ci) * c.k + ky) * c.k + kx;
                    const size_t row = (static_cast<size_t>(ky) * c.k + kx) * c.cin + ci;
                    c.w_kn.data[row * c.cout + co] = w_oihw.data[src];
                }
            }
        }
    }
    c.bias = b;
    return c;
}

void Conv2d::forward(const Tensor& x, Tensor& y, Tensor& col, GemmKind kind, ConvTimes* times, bool relu) const {
    check_rank(x, 4, "conv input");
    if (x.dim(3) != cin) throw std::invalid_argument("conv input channels " + std::to_string(x.dim(3)) + " != " + std::to_string(cin));
    const int batch = static_cast<int>(x.dim(0)), h = static_cast<int>(x.dim(1)), w = static_cast<int>(x.dim(2));
    const int oh = conv_out_size(h, k, pad), ow = conv_out_size(w, k, pad);
    const int64_t M = static_cast<int64_t>(batch) * oh * ow;
    const int K = k * k * cin;
    if (M > 1 << 30) throw std::invalid_argument("conv batch too large");

    auto t0 = Clock::now();
    col.resize({M, K});
    im2col_nhwc(x.ptr(), batch, h, w, cin, k, pad, col.ptr());
    if (times) times->im2col_ms += ms_since(t0);

    t0 = Clock::now();
    y.resize({batch, oh, ow, cout});
    gemm(kind, static_cast<int>(M), cout, K, col.ptr(), K, w_kn.ptr(), cout, y.ptr(), cout);
    if (times) times->gemm_ms += ms_since(t0);

    t0 = Clock::now();
    float* yp = y.ptr();
    const float* bp = bias.ptr();
    if (relu) {
        for (int64_t m = 0; m < M; ++m) {
            float* row = yp + static_cast<size_t>(m) * cout;
#pragma omp simd
            for (int n = 0; n < cout; ++n) row[n] = std::max(row[n] + bp[n], 0.0f);
        }
    } else {
        for (int64_t m = 0; m < M; ++m) {
            float* row = yp + static_cast<size_t>(m) * cout;
#pragma omp simd
            for (int n = 0; n < cout; ++n) row[n] += bp[n];
        }
    }
    if (times) times->bias_ms += ms_since(t0);
}

void relu_(Tensor& x) {
    float* p = x.ptr();
    const int64_t n = x.numel();
#pragma omp simd
    for (int64_t i = 0; i < n; ++i) p[i] = std::max(p[i], 0.0f);
}

void preprocess_cifar(const TensorU8& images, const float mean[3], const float stddev[3], Tensor& out) {
    constexpr int kSize = 32, kCh = 3;
    if (images.ndim() != 4 || images.dim(1) != kCh || images.dim(2) != kSize || images.dim(3) != kSize) {
        throw std::invalid_argument("images must be [batch][3][32][32] u8, got " + shape_str(images.shape));
    }
    const int batch = static_cast<int>(images.dim(0));
    const int hw = kSize * kSize;
    out.resize({batch, kSize, kSize, kCh});
    const uint8_t* src = images.ptr();
    float* dst = out.ptr();
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < kCh; ++c) {
            const uint8_t* plane = src + (static_cast<size_t>(b) * kCh + c) * hw;
            float* o = dst + static_cast<size_t>(b) * hw * kCh + c;
            for (int p = 0; p < hw; ++p) {
                o[static_cast<size_t>(p) * kCh] = (static_cast<float>(plane[p]) / 255.0f - mean[c]) / stddev[c];
            }
        }
    }
}

void maxpool2x2_nhwc(const Tensor& x, Tensor& y) {
    check_rank(x, 4, "maxpool input");
    const int batch = static_cast<int>(x.dim(0)), h = static_cast<int>(x.dim(1)), w = static_cast<int>(x.dim(2));
    const int c = static_cast<int>(x.dim(3));
    const int oh = h / 2, ow = w / 2;
    if (oh == 0 || ow == 0) throw std::invalid_argument("maxpool input smaller than the window");
    y.resize({batch, oh, ow, c});
    const float* xp = x.ptr();
    float* yp = y.ptr();
    for (int b = 0; b < batch; ++b) {
        for (int oy = 0; oy < oh; ++oy) {
            for (int ox = 0; ox < ow; ++ox) {
                const float* p00 = xp + ((static_cast<size_t>(b) * h + 2 * oy) * w + 2 * ox) * c;
                const float* p01 = p00 + c;
                const float* p10 = p00 + static_cast<size_t>(w) * c;
                const float* p11 = p10 + c;
                float* out = yp + ((static_cast<size_t>(b) * oh + oy) * ow + ox) * c;
#pragma omp simd
                for (int ch = 0; ch < c; ++ch) {
                    out[ch] = std::max(std::max(p00[ch], p01[ch]), std::max(p10[ch], p11[ch]));
                }
            }
        }
    }
}

Linear Linear::from_pytorch(const Tensor& w_nk, const Tensor& b) {
    check_rank(w_nk, 2, "linear weight");
    Linear l;
    l.out = static_cast<int>(w_nk.dim(0));
    l.in = static_cast<int>(w_nk.dim(1));
    if (b.numel() != l.out) throw std::invalid_argument("linear bias size does not match out features");
    l.w_kn = Tensor({l.in, l.out});
    for (int n = 0; n < l.out; ++n) {
        for (int kk = 0; kk < l.in; ++kk) {
            l.w_kn.data[static_cast<size_t>(kk) * l.out + n] = w_nk.data[static_cast<size_t>(n) * l.in + kk];
        }
    }
    l.bias = b;
    return l;
}

void Linear::forward(const Tensor& x, Tensor& y, GemmKind kind) const {
    check_rank(x, 2, "linear input");
    if (x.dim(1) != in) throw std::invalid_argument("linear input features " + std::to_string(x.dim(1)) + " != " + std::to_string(in));
    const int batch = static_cast<int>(x.dim(0));
    y.resize({batch, out});
    gemm(kind, batch, out, in, x.ptr(), in, w_kn.ptr(), out, y.ptr(), out);
    for (int b = 0; b < batch; ++b) {
        for (int n = 0; n < out; ++n) y.data[static_cast<size_t>(b) * out + n] += bias.data[static_cast<size_t>(n)];
    }
}

}  // namespace tinyinfer
