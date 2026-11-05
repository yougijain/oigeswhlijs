#include "tinyinfer/layers.h"

#include <random>

#include "check.h"

using namespace tinyinfer;

namespace {

std::mt19937& rng() {
    static std::mt19937 r(42);
    return r;
}

Tensor random_tensor(std::vector<int64_t> shape, float lo = -1.0f, float hi = 1.0f) {
    Tensor t(std::move(shape));
    std::uniform_real_distribution<float> d(lo, hi);
    for (float& v : t.data) v = d(rng());
    return t;
}

double max_abs_diff(const Tensor& a, const Tensor& b) {
    CHECK(a.shape == b.shape);
    double m = 0;
    for (size_t i = 0; i < a.data.size(); ++i) m = std::max(m, std::fabs(static_cast<double>(a.data[i]) - b.data[i]));
    return m;
}

}  // namespace

TEST(im2col_hand_checked) {
    // 1 image, 2x2, 1 channel, k=3 pad=1. Pixel values 1..4.
    const float x[] = {1, 2, 3, 4};
    float col[4 * 9];
    im2col_nhwc(x, 1, 2, 2, 1, 3, 1, col);
    // Output pixel (0,0): window rows -1..1, cols -1..1 -> only (0,0),(0,1),(1,0),(1,1) inside.
    const float expect00[] = {0, 0, 0, 0, 1, 2, 0, 3, 4};
    for (int i = 0; i < 9; ++i) CHECK_EQ(col[i], expect00[i]);
    // Output pixel (1,1): window rows 0..2, cols 0..2.
    const float expect11[] = {1, 2, 0, 3, 4, 0, 0, 0, 0};
    for (int i = 0; i < 9; ++i) CHECK_EQ(col[27 + i], expect11[i]);
}

TEST(im2col_channel_runs_are_contiguous) {
    // 1x1 image with 3 channels, k=1, pad=0: col row is just the pixel.
    const float x[] = {7, 8, 9};
    float col[3];
    im2col_nhwc(x, 1, 1, 1, 3, 1, 0, col);
    CHECK_EQ(col[0], 7.0f);
    CHECK_EQ(col[2], 9.0f);
}

TEST(conv_gemm_matches_direct) {
    const struct {
        int batch, h, w, cin, cout, k, pad;
    } cases[] = {{1, 4, 4, 1, 1, 3, 1}, {2, 5, 7, 3, 4, 3, 1}, {3, 8, 8, 6, 5, 3, 0}, {1, 6, 6, 2, 3, 1, 0},
                 {2, 32, 32, 3, 32, 3, 1}, {1, 8, 8, 64, 128, 3, 1}};
    for (const auto& c : cases) {
        const Tensor x = random_tensor({c.batch, c.h, c.w, c.cin});
        const Tensor w = random_tensor({c.cout, c.cin, c.k, c.k});
        const Tensor b = random_tensor({c.cout});
        Tensor ref;
        conv2d_direct_nhwc(x, w, b, c.pad, ref);
        const Conv2d conv = Conv2d::from_pytorch(w, b, c.pad);
        const double tol = 1e-5 * std::sqrt(static_cast<double>(c.k * c.k * c.cin)) + 1e-6;
        Tensor col;  // one scratch shared across kinds, as the model does
        for (int ki = 0; ki < kGemmKindCount; ++ki) {
            const GemmKind kind = static_cast<GemmKind>(ki);
            Tensor y;
            conv.forward(x, y, col, kind);
            CHECK(y.shape == ref.shape);
            const double err = max_abs_diff(y, ref);
            if (err > tol) {
                ::tinytest::fail(__FILE__, __LINE__, std::string("conv mismatch with ") + gemm_kind_name(kind) + ": " + std::to_string(err));
            }
            // Running forward twice with the same scratch must give the same answer (no stale state).
            Tensor y2;
            conv.forward(x, y2, col, kind);
            CHECK_EQ(max_abs_diff(y, y2), 0.0);
        }
    }
}

TEST(conv_fused_relu_matches_separate_relu) {
    const Tensor x = random_tensor({2, 6, 6, 4});
    const Conv2d conv = Conv2d::from_pytorch(random_tensor({5, 4, 3, 3}), random_tensor({5}), 1);
    Tensor y1, y2, col;
    conv.forward(x, y1, col, GemmKind::Naive);
    relu_(y1);
    conv.forward(x, y2, col, GemmKind::Naive, nullptr, /*relu=*/true);
    CHECK(y1.data == y2.data);
}

TEST(conv_rejects_channel_mismatch) {
    const Conv2d conv = Conv2d::from_pytorch(random_tensor({4, 3, 3, 3}), random_tensor({4}), 1);
    Tensor y, col;
    CHECK_THROWS(conv.forward(random_tensor({1, 4, 4, 2}), y, col, GemmKind::Naive));
}

TEST(relu_clamps_negatives) {
    Tensor t({5});
    t.data = {-2, -0.0f, 0, 0.5f, 3};
    relu_(t);
    CHECK_EQ(t.data[0], 0.0f);
    CHECK_EQ(t.data[3], 0.5f);
    CHECK_EQ(t.data[4], 3.0f);
}

TEST(maxpool_hand_checked_and_floor) {
    // 1 image, 3x5, 2 channels: pooled to 1x2, dropping the last row and column.
    Tensor x({1, 3, 5, 2});
    for (size_t i = 0; i < x.data.size(); ++i) x.data[i] = static_cast<float>(i);
    Tensor y;
    maxpool2x2_nhwc(x, y);
    CHECK(y.shape == std::vector<int64_t>({1, 1, 2, 2}));
    // Window rows 0..1, cols 0..1, channel 0: indices (r*5+c)*2 -> max at r=1,c=1 -> (1*5+1)*2 = 12.
    CHECK_EQ(y.data[0], 12.0f);
    CHECK_EQ(y.data[1], 13.0f);
    // cols 2..3: (1*5+3)*2 = 16.
    CHECK_EQ(y.data[2], 16.0f);
    CHECK_EQ(y.data[3], 17.0f);
}

TEST(maxpool_matches_loop_reference) {
    const Tensor x = random_tensor({2, 8, 6, 5});
    Tensor y;
    maxpool2x2_nhwc(x, y);
    for (int b = 0; b < 2; ++b)
        for (int oy = 0; oy < 4; ++oy)
            for (int ox = 0; ox < 3; ++ox)
                for (int c = 0; c < 5; ++c) {
                    float m = -1e9f;
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx)
                            m = std::max(m, x.data[((static_cast<size_t>(b) * 8 + 2 * oy + dy) * 6 + 2 * ox + dx) * 5 + c]);
                    CHECK_EQ(y.data[((static_cast<size_t>(b) * 4 + oy) * 3 + ox) * 5 + c], m);
                }
}

TEST(linear_matches_loop_reference) {
    const Tensor w = random_tensor({10, 24});  // PyTorch [out][in]
    const Tensor b = random_tensor({10});
    const Tensor x = random_tensor({3, 24});
    const Linear fc = Linear::from_pytorch(w, b);
    Tensor y;
    fc.forward(x, y, GemmKind::Naive);
    CHECK(y.shape == std::vector<int64_t>({3, 10}));
    for (int i = 0; i < 3; ++i)
        for (int n = 0; n < 10; ++n) {
            double acc = b.data[static_cast<size_t>(n)];
            for (int k = 0; k < 24; ++k) acc += static_cast<double>(x.data[static_cast<size_t>(i) * 24 + k]) * w.data[static_cast<size_t>(n) * 24 + k];
            CHECK_NEAR(y.data[static_cast<size_t>(i) * 10 + n], acc, 1e-5);
        }
}
