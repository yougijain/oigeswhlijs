#include "tinyinfer/quant.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace tinyinfer {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
}  // namespace

QuantizedMatrix quantize_per_column(const Tensor& w_kn) {
    if (w_kn.ndim() != 2) throw std::invalid_argument("quantize_per_column expects a [K][N] matrix");
    const int64_t K = w_kn.dim(0), N = w_kn.dim(1);
    QuantizedMatrix out;
    out.q = TensorI8({K, N});
    out.scale.assign(static_cast<size_t>(N), 0.0f);
    for (int64_t n = 0; n < N; ++n) {
        float m = 0.0f;
        for (int64_t k = 0; k < K; ++k) m = std::max(m, std::fabs(w_kn.data[static_cast<size_t>(k * N + n)]));
        // A column of zeros gets scale 1 so dequantisation stays finite.
        const float s = m > 0.0f ? m / static_cast<float>(kQMax) : 1.0f;
        out.scale[static_cast<size_t>(n)] = s;
        for (int64_t k = 0; k < K; ++k) {
            const float v = std::nearbyint(w_kn.data[static_cast<size_t>(k * N + n)] / s);
            out.q.data[static_cast<size_t>(k * N + n)] = static_cast<int8_t>(std::clamp(v, -127.0f, 127.0f));
        }
    }
    return out;
}

float max_abs(const float* x, int64_t n) {
    float m = 0.0f;
    for (int64_t i = 0; i < n; ++i) m = std::max(m, std::fabs(x[i]));
    return m;
}

void quantize_activations(const float* x, int64_t n, float scale, int8_t* out) {
    if (!(scale > 0.0f)) throw std::invalid_argument("activation scale must be positive");
    const float inv = 1.0f / scale;
    // Round half to even without touching the FP environment, so the loop vectorises:
    // adding and subtracting 1.5 * 2^23 rounds any |v| <= 2^22 to an integer in float.
    constexpr float kMagic = 12582912.0f;
#pragma omp simd
    for (int64_t i = 0; i < n; ++i) {
        float v = std::min(std::max(x[i] * inv, -127.0f), 127.0f);
        v = (v + kMagic) - kMagic;
        out[i] = static_cast<int8_t>(v);
    }
}

void im2col_nhwc_i8(const int8_t* x, int batch, int h, int w, int cin, int k, int pad, int8_t* col) {
    // Same structure as im2col_nhwc: one copy per kernel row for interior pixels.
    const int oh = conv_out_size(h, k, pad);
    const int ow = conv_out_size(w, k, pad);
    const size_t run = static_cast<size_t>(cin);
    int8_t* out = col;
    for (int b = 0; b < batch; ++b) {
        const int8_t* img = x + static_cast<size_t>(b) * h * w * cin;
        for (int oy = 0; oy < oh; ++oy) {
            for (int ox = 0; ox < ow; ++ox) {
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

namespace {

inline void im2col_row_i8(const int8_t* img, int h, int w, int cin, int k, int pad, int oy, int ox, int8_t* row) {
    const size_t run = static_cast<size_t>(cin);
    const int ix0 = ox - pad;
    const bool row_inside = ix0 >= 0 && ix0 + k <= w;
    int8_t* out = row;
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

// Scatter row m into the packed layout of gemm_int8.h. The 4 k of a group are
// consecutive bytes of the row, so a group is one 32-bit copy (xor mask applied
// to all four bytes at once); only a partial last group goes byte by byte.
inline void scatter_packed_row_i8(const int8_t* row, int64_t m, int64_t m_pad, int K, const PackedALayoutI8& L,
                                  uint8_t* packed) {
    const int64_t s = m / L.mr, r = m % L.mr;
    const uint32_t mask4 = static_cast<uint32_t>(L.xor_mask) * 0x01010101u;
    const size_t gstride = static_cast<size_t>(L.mr) * 4;
    for (int b = 0, k0 = 0; k0 < K; ++b, k0 += L.kc) {
        const int kc_b = std::min(L.kc, K - k0);
        const int groups = (kc_b + 3) / 4, full = kc_b / 4;
        uint8_t* strip = packed + static_cast<size_t>(b) * m_pad * L.kc + static_cast<size_t>(s) * groups * gstride +
                         static_cast<size_t>(r) * 4;
        const int8_t* src = row + k0;
        for (int g = 0; g < full; ++g) {
            uint32_t v;
            std::memcpy(&v, src + static_cast<size_t>(g) * 4, 4);
            v ^= mask4;
            std::memcpy(strip + static_cast<size_t>(g) * gstride, &v, 4);
        }
        if (full < groups) {
            uint8_t tail[4] = {0, 0, 0, 0};
            for (int q = 0; q < 4; ++q) {
                const int kk = full * 4 + q;
                if (kk < kc_b) tail[q] = static_cast<uint8_t>(static_cast<uint8_t>(src[kk]) ^ L.xor_mask);
            }
            std::memcpy(strip + static_cast<size_t>(full) * gstride, tail, 4);
        }
    }
}

}  // namespace

void im2col_nhwc_i8_packed(const int8_t* x, int batch, int h, int w, int cin, int k, int pad, bool parallel,
                           uint8_t* packed) {
    const int oh = conv_out_size(h, k, pad);
    const int ow = conv_out_size(w, k, pad);
    const int K = k * k * cin;
    const int64_t M = static_cast<int64_t>(batch) * oh * ow;
    const PackedALayoutI8 L = gemm_int8_packed_a_layout();
    if (L.elem_bytes != 1 || L.group != 4) throw std::logic_error("im2col_nhwc_i8_packed: unsupported packed layout");
    const int64_t m_pad = (M + L.mr - 1) / L.mr * L.mr;
    (void)parallel;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel)
#endif
    for (int64_t m = 0; m < M; ++m) {
        static thread_local std::vector<int8_t> row;
        row.resize(static_cast<size_t>(K));
        const int b = static_cast<int>(m / (oh * ow));
        const int pix = static_cast<int>(m % (oh * ow));
        im2col_row_i8(x + static_cast<size_t>(b) * h * w * cin, h, w, cin, k, pad, pix / ow, pix % ow, row.data());
        scatter_packed_row_i8(row.data(), m, m_pad, K, L, packed);
    }
    if (m_pad > M) {
        std::vector<int8_t> zeros(static_cast<size_t>(K), 0);
        for (int64_t m = M; m < m_pad; ++m) scatter_packed_row_i8(zeros.data(), m, m_pad, K, L, packed);
    }
}

QConv2d QConv2d::from(const Conv2d& c, float scale_in) {
    if (!(scale_in > 0.0f)) throw std::invalid_argument("conv input scale must be positive");
    QConv2d q;
    q.cin = c.cin;
    q.cout = c.cout;
    q.k = c.k;
    q.pad = c.pad;
    QuantizedMatrix qm = quantize_per_column(c.w_kn);
    q.wq = std::move(qm.q);
    q.scale_w = std::move(qm.scale);
    q.scale_in = scale_in;
    q.bias = c.bias;
    return q;
}

void QConv2d::forward(const Tensor& x, Tensor& y, TensorI8& xq, TensorI8& col, TensorI32& acc, GemmInt8Kind kind,
                      QConvTimes* times, bool relu) const {
    if (x.ndim() != 4 || x.dim(3) != cin) throw std::invalid_argument("qconv input shape " + shape_str(x.shape));
    const int batch = static_cast<int>(x.dim(0)), h = static_cast<int>(x.dim(1)), w = static_cast<int>(x.dim(2));
    const int oh = conv_out_size(h, k, pad), ow = conv_out_size(w, k, pad);
    const int64_t M = static_cast<int64_t>(batch) * oh * ow;
    const int K = k * k * cin;
    if (M > 1 << 30) throw std::invalid_argument("qconv batch too large");

    auto t0 = Clock::now();
    xq.resize(x.shape);
    quantize_activations(x.ptr(), x.numel(), scale_in, xq.ptr());
    if (times) times->quantize_ms += ms_since(t0);

    t0 = Clock::now();
    const bool packed = gemm_int8_kind_uses_packed_a(kind);
    if (packed) {
        col.resize({gemm_int8_packed_a_size(static_cast<int>(M), K)});
        im2col_nhwc_i8_packed(xq.ptr(), batch, h, w, cin, k, pad, kind == GemmInt8Kind::Threaded,
                              reinterpret_cast<uint8_t*>(col.ptr()));
    } else {
        col.resize({M, K});
        im2col_nhwc_i8(xq.ptr(), batch, h, w, cin, k, pad, col.ptr());
    }
    if (times) times->im2col_ms += ms_since(t0);

    t0 = Clock::now();
    acc.resize({M, cout});
    if (packed) {
        gemm_s8_packed_a(kind, static_cast<int>(M), cout, K, reinterpret_cast<const uint8_t*>(col.ptr()), wq.ptr(), cout,
                         acc.ptr(), cout);
    } else {
        gemm_s8(kind, static_cast<int>(M), cout, K, col.ptr(), K, wq.ptr(), cout, acc.ptr(), cout);
    }
    if (times) times->gemm_ms += ms_since(t0);

    t0 = Clock::now();
    y.resize({batch, oh, ow, cout});
    // Per-channel multiplier scale_in * scale_w[n], precomputed once per call.
    static thread_local std::vector<float> mult;
    mult.resize(static_cast<size_t>(cout));
    for (int n = 0; n < cout; ++n) mult[static_cast<size_t>(n)] = scale_in * scale_w[static_cast<size_t>(n)];
    const int32_t* ap = acc.ptr();
    float* yp = y.ptr();
    const float* mp = mult.data();
    const float* bp = bias.ptr();
    if (relu) {
        for (int64_t m = 0; m < M; ++m) {
            const int32_t* arow = ap + static_cast<size_t>(m) * cout;
            float* yrow = yp + static_cast<size_t>(m) * cout;
#pragma omp simd
            for (int n = 0; n < cout; ++n) yrow[n] = std::max(static_cast<float>(arow[n]) * mp[n] + bp[n], 0.0f);
        }
    } else {
        for (int64_t m = 0; m < M; ++m) {
            const int32_t* arow = ap + static_cast<size_t>(m) * cout;
            float* yrow = yp + static_cast<size_t>(m) * cout;
#pragma omp simd
            for (int n = 0; n < cout; ++n) yrow[n] = static_cast<float>(arow[n]) * mp[n] + bp[n];
        }
    }
    if (times) times->dequant_ms += ms_since(t0);
}

QLinear QLinear::from(const Linear& l, float scale_in) {
    if (!(scale_in > 0.0f)) throw std::invalid_argument("linear input scale must be positive");
    QLinear q;
    q.in = l.in;
    q.out = l.out;
    QuantizedMatrix qm = quantize_per_column(l.w_kn);
    q.wq = std::move(qm.q);
    q.scale_w = std::move(qm.scale);
    q.scale_in = scale_in;
    q.bias = l.bias;
    return q;
}

void QLinear::forward(const Tensor& x, Tensor& y, TensorI8& xq, TensorI32& acc, GemmInt8Kind kind) const {
    if (x.ndim() != 2 || x.dim(1) != in) throw std::invalid_argument("qlinear input shape " + shape_str(x.shape));
    const int batch = static_cast<int>(x.dim(0));
    xq.resize(x.shape);
    quantize_activations(x.ptr(), x.numel(), scale_in, xq.ptr());
    acc.resize({batch, out});
    gemm_s8(kind, batch, out, in, xq.ptr(), in, wq.ptr(), out, acc.ptr(), out);
    y.resize({batch, out});
    for (int b = 0; b < batch; ++b) {
        for (int n = 0; n < out; ++n) {
            const size_t i = static_cast<size_t>(b) * out + n;
            y.data[i] = static_cast<float>(acc.data[i]) * (scale_in * scale_w[static_cast<size_t>(n)]) + bias.data[static_cast<size_t>(n)];
        }
    }
}

}  // namespace tinyinfer
