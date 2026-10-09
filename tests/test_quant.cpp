// INT8 path: quantisation primitives, the int8 GEMM kinds (exact equality),
// an int8 conv against the float conv on exactly representable data, and the
// end-to-end int8 model on the exported images. Usage: test_quant <data/export dir>

#include <cmath>
#include <cstdio>
#include <random>

#include "check.h"
#include "tinyinfer/gemm_int8.h"
#include "tinyinfer/loader.h"
#include "tinyinfer/model_int8.h"
#include "tinyinfer/quant.h"

using namespace tinyinfer;

namespace {

std::mt19937& rng() {
    static std::mt19937 r(99);
    return r;
}

Tensor random_tensor(std::vector<int64_t> shape, float lo = -1.0f, float hi = 1.0f) {
    Tensor t(std::move(shape));
    std::uniform_real_distribution<float> d(lo, hi);
    for (float& v : t.data) v = d(rng());
    return t;
}

std::vector<int8_t> random_i8(size_t n, bool extremes) {
    std::uniform_int_distribution<int> d(-128, 127);
    std::vector<int8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<int8_t>(extremes ? ((i % 3 == 0) ? -128 : 127) : d(rng()));
    return v;
}

}  // namespace

TEST(per_column_quantisation_bounds) {
    const Tensor w = random_tensor({40, 7}, -3.0f, 3.0f);
    const QuantizedMatrix q = quantize_per_column(w);
    CHECK_EQ(q.scale.size(), size_t{7});
    for (int n = 0; n < 7; ++n) {
        float colmax = 0;
        for (int k = 0; k < 40; ++k) colmax = std::max(colmax, std::fabs(w.data[static_cast<size_t>(k) * 7 + n]));
        CHECK_NEAR(q.scale[static_cast<size_t>(n)], colmax / 127.0f, 1e-7);
        for (int k = 0; k < 40; ++k) {
            const int v = q.q.data[static_cast<size_t>(k) * 7 + n];
            CHECK(v >= -127 && v <= 127);
            // Dequantised value is within half a step of the original.
            CHECK_NEAR(v * q.scale[static_cast<size_t>(n)], w.data[static_cast<size_t>(k) * 7 + n], q.scale[static_cast<size_t>(n)] * 0.5f + 1e-6f);
        }
    }
    Tensor zeros({3, 2});
    const QuantizedMatrix qz = quantize_per_column(zeros);
    CHECK_EQ(qz.scale[0], 1.0f);  // a zero column must not produce a zero scale
}

TEST(activation_quantisation_rounds_and_clamps) {
    const float x[] = {0.26f, -0.26f, 0.05f, 12.7f, -12.7f, 100.0f, -100.0f, 0.0f};
    int8_t q[8];
    quantize_activations(x, 8, 0.1f, q);
    const int expect[] = {3, -3, 0, 127, -127, 127, -127, 0};  // 0.05/0.1 = 0.5 rounds to even
    for (int i = 0; i < 8; ++i) CHECK_EQ(q[i], expect[i]);
    CHECK_THROWS(quantize_activations(x, 8, 0.0f, q));
}

TEST(int8_im2col_matches_float_im2col) {
    const int batch = 2, h = 5, w = 7, cin = 3, k = 3, pad = 1;
    std::vector<int8_t> xi = random_i8(static_cast<size_t>(batch * h * w * cin), false);
    std::vector<float> xf(xi.begin(), xi.end());
    const int M = batch * h * w, K = k * k * cin;
    std::vector<int8_t> ci(static_cast<size_t>(M) * K);
    std::vector<float> cf(static_cast<size_t>(M) * K);
    im2col_nhwc_i8(xi.data(), batch, h, w, cin, k, pad, ci.data());
    im2col_nhwc(xf.data(), batch, h, w, cin, k, pad, cf.data());
    for (size_t i = 0; i < ci.size(); ++i) CHECK_EQ(static_cast<float>(ci[i]), cf[i]);
}

TEST(int8_gemm_kinds_are_exact) {
    const int sizes[][3] = {{1, 1, 1},   {1, 17, 5},   {13, 1, 9},   {7, 7, 7},     {6, 16, 8},      {12, 32, 16},
                            {33, 37, 29}, {64, 64, 64}, {100, 130, 70}, {5, 16, 27}, {128, 10, 2048}, {97, 200, 288}};
    for (const auto& s : sizes) {
        const int M = s[0], N = s[1], K = s[2];
        for (int extremes = 0; extremes < 2; ++extremes) {
            const std::vector<int8_t> A = random_i8(static_cast<size_t>(M) * K, extremes != 0);
            const std::vector<int8_t> B = random_i8(static_cast<size_t>(K) * N, extremes != 0);
            std::vector<int32_t> ref(static_cast<size_t>(M) * N);
            gemm_s8(GemmInt8Kind::Naive, M, N, K, A.data(), K, B.data(), N, ref.data(), N);
            for (int ki = 1; ki < kGemmInt8KindCount; ++ki) {
                std::vector<int32_t> C(static_cast<size_t>(M) * N, 77);
                gemm_s8(static_cast<GemmInt8Kind>(ki), M, N, K, A.data(), K, B.data(), N, C.data(), N);
                if (C != ref) {
                    ::tinytest::fail(__FILE__, __LINE__, std::string(gemm_int8_kind_name(static_cast<GemmInt8Kind>(ki))) +
                                                             " M=" + std::to_string(M) + " N=" + std::to_string(N) +
                                                             " K=" + std::to_string(K) + " differs from naive");
                }
            }
        }
    }
}

TEST(int8_gemm_respects_leading_dimensions) {
    const int8_t A[] = {1, 2, 99, 3, 4, 99};
    const int8_t B[] = {5, 6, 99, 99, 7, 8, 99, 99};
    for (int ki = 0; ki < kGemmInt8KindCount; ++ki) {
        int32_t C[10];
        for (int32_t& v : C) v = -1;
        gemm_s8(static_cast<GemmInt8Kind>(ki), 2, 2, 2, A, 3, B, 4, C, 5);
        CHECK_EQ(C[0], 19);
        CHECK_EQ(C[1], 22);
        CHECK_EQ(C[5], 43);
        CHECK_EQ(C[6], 50);
        CHECK_EQ(C[2], -1);
        CHECK_EQ(C[9], -1);
    }
}

TEST(int8_conv_matches_float_conv_on_representable_data) {
    // Inputs and weights are exact multiples of their scales, so the int8 path computes the
    // same products as the float path and must agree to float rounding.
    const int batch = 2, h = 6, w = 6, cin = 5, cout = 7, k = 3, pad = 1;
    const float scale_in = 0.01f;
    std::uniform_int_distribution<int> d(-127, 127);
    Tensor x({batch, h, w, cin});
    for (float& v : x.data) v = scale_in * static_cast<float>(d(rng()));
    Tensor w_oihw({cout, cin, k, k});
    for (int co = 0; co < cout; ++co) {
        const float sw = 0.002f * static_cast<float>(co + 1);
        float* col = w_oihw.data.data() + static_cast<size_t>(co) * cin * k * k;
        for (int i = 0; i < cin * k * k; ++i) col[i] = sw * static_cast<float>(d(rng()));
        col[0] = sw * 127.0f;  // make sure the column max is exactly 127 steps
    }
    const Tensor b = random_tensor({cout});
    const Conv2d conv = Conv2d::from_pytorch(w_oihw, b, pad);
    Tensor yf, colf;
    conv.forward(x, yf, colf, GemmKind::Naive);
    const QConv2d qconv = QConv2d::from(conv, scale_in);
    for (int ki = 0; ki < kGemmInt8KindCount; ++ki) {
        Tensor yq;
        TensorI8 xq, col;
        TensorI32 acc;
        qconv.forward(x, yq, xq, col, acc, static_cast<GemmInt8Kind>(ki));
        CHECK(yq.shape == yf.shape);
        double max_err = 0, max_val = 0;
        for (size_t i = 0; i < yq.data.size(); ++i) {
            max_err = std::max(max_err, std::fabs(static_cast<double>(yq.data[i]) - yf.data[i]));
            max_val = std::max(max_val, std::fabs(static_cast<double>(yf.data[i])));
        }
        CHECK(max_err <= 1e-4 * max_val + 1e-5);
    }
}

TEST(int8_conv_fused_relu_matches_separate_relu) {
    const Tensor x = random_tensor({1, 5, 5, 3});
    const Conv2d conv = Conv2d::from_pytorch(random_tensor({4, 3, 3, 3}), random_tensor({4}), 1);
    const QConv2d q = QConv2d::from(conv, 0.01f);
    Tensor y1, y2;
    TensorI8 xq, col;
    TensorI32 acc;
    q.forward(x, y1, xq, col, acc, GemmInt8Kind::Naive);
    relu_(y1);
    q.forward(x, y2, xq, col, acc, GemmInt8Kind::Naive, nullptr, /*relu=*/true);
    CHECK(y1.data == y2.data);
}

TEST(int8_model_end_to_end) {
    if (test_args().empty()) {
        ::tinytest::fail(__FILE__, __LINE__, "usage: test_quant <data/export dir>");
        return;
    }
    const std::string dir = test_args()[0];
    TinyCNN fp32 = TinyCNN::load(TensorFile::read(dir + "/weights.bin"));
    const TensorFile imf = TensorFile::read(dir + "/test_images.bin");
    const TensorU8 images = imf.u8("images");
    const TensorU8 labels = imf.u8("labels");
    const Tensor ref = TensorFile::read(dir + "/ref_logits.bin").f32("logits");
    const TensorU8 calib = TensorFile::read(dir + "/calib_images.bin").u8("images");

    TinyCNNInt8 q = TinyCNNInt8::quantize(fp32, calib);
    CHECK(q.weight_bytes() < fp32.param_count() * 4 / 3);  // roughly a quarter of the float size
    for (int i = 0; i < 3; ++i) CHECK(q.conv(i).scale_in > 0.0f);

    // Save and reload must be lossless.
    const std::vector<uint8_t> buf = serialize_tensors(q.to_raw());
    TinyCNNInt8 q2 = TinyCNNInt8::load(TensorFile::parse(buf, "mem"));

    const int n = static_cast<int>(images.dim(0));
    const int k = static_cast<int>(ref.dim(1));
    const size_t per = static_cast<size_t>(images.dim(1) * images.dim(2) * images.dim(3));
    int correct_q = 0, correct_ref = 0, agree = 0;
    double max_err = 0;
    for (int start = 0; start < n; start += 100) {
        const int b = std::min(100, n - start);
        TensorU8 chunk({b, images.dim(1), images.dim(2), images.dim(3)});
        std::copy(images.data.begin() + static_cast<std::ptrdiff_t>(start * per),
                  images.data.begin() + static_cast<std::ptrdiff_t>((start + b) * per), chunk.data.begin());
        const Tensor l1 = q.forward(chunk, default_gemm_int8_kind());
        const Tensor l2 = q2.forward(chunk, GemmInt8Kind::Naive);
        CHECK(l1.data == l2.data);  // same int8 weights, exact integer GEMM, same float epilogue
        for (int i = 0; i < b; ++i) {
            int aq = 0, ar = 0;
            for (int c = 0; c < k; ++c) {
                const float v = l1.data[static_cast<size_t>(i) * k + c];
                const float r = ref.data[static_cast<size_t>(start + i) * k + c];
                CHECK(std::isfinite(v));
                max_err = std::max(max_err, std::fabs(static_cast<double>(v) - r));
                if (v > l1.data[static_cast<size_t>(i) * k + aq]) aq = c;
                if (r > ref.data[static_cast<size_t>(start + i) * k + ar]) ar = c;
            }
            const int label = labels.data[static_cast<size_t>(start + i)];
            correct_q += aq == label;
            correct_ref += ar == label;
            agree += aq == ar;
        }
    }
    std::printf("  int8: acc %.2f%% (fp32 %.2f%%), argmax agreement %d/%d, max abs logit err %.3f\n",
                100.0 * correct_q / n, 100.0 * correct_ref / n, agree, n, max_err);
    // INT8 must stay within 2 points of FP32 on these 1,000 images and keep logits in the same range.
    CHECK(correct_q >= correct_ref - n / 50);
    CHECK(max_err < 2.0);
}
