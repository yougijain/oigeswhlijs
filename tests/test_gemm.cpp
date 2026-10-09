#include "tinyinfer/gemm.h"

#include <random>
#include <vector>

#include "check.h"

using namespace tinyinfer;

namespace {

// Reference in double precision: catches both wrong indexing and accumulated float error.
void gemm_ref(int M, int N, int K, const std::vector<float>& A, const std::vector<float>& B, std::vector<double>& C) {
    C.assign(static_cast<size_t>(M) * N, 0.0);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                C[static_cast<size_t>(i) * N + j] +=
                    static_cast<double>(A[static_cast<size_t>(i) * K + k]) * B[static_cast<size_t>(k) * N + j];
}

std::vector<float> random_matrix(size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float& x : v) x = dist(rng);
    return v;
}

void check_kind_against_ref(GemmKind kind, int M, int N, int K, std::mt19937& rng) {
    const std::vector<float> A = random_matrix(static_cast<size_t>(M) * K, rng);
    const std::vector<float> B = random_matrix(static_cast<size_t>(K) * N, rng);
    std::vector<double> ref;
    gemm_ref(M, N, K, A, B, ref);
    std::vector<float> C(static_cast<size_t>(M) * N, 12345.0f);  // garbage: gemm must overwrite, not accumulate
    gemm(kind, M, N, K, A.data(), K, B.data(), N, C.data(), N);
    double max_err = 0;
    for (size_t i = 0; i < C.size(); ++i) max_err = std::max(max_err, std::fabs(C[i] - ref[i]));
    // Each output is a sum of K products in [-1,1]; float32 error grows ~ sqrt(K) * eps * |sum|.
    const double tol = 1e-5 * std::sqrt(static_cast<double>(K)) + 1e-6;
    if (max_err > tol) {
        ::tinytest::fail(__FILE__, __LINE__, std::string(gemm_kind_name(kind)) + " M=" + std::to_string(M) +
                                                 " N=" + std::to_string(N) + " K=" + std::to_string(K) +
                                                 " max err " + std::to_string(max_err));
    }
}

}  // namespace

TEST(hand_checked_2x2) {
    const float A[] = {1, 2, 3, 4};
    const float B[] = {5, 6, 7, 8};
    for (int ki = 0; ki < kGemmKindCount; ++ki) {
        float C[4] = {0, 0, 0, 0};
        gemm(static_cast<GemmKind>(ki), 2, 2, 2, A, 2, B, 2, C, 2);
        CHECK_EQ(C[0], 19.0f);
        CHECK_EQ(C[1], 22.0f);
        CHECK_EQ(C[2], 43.0f);
        CHECK_EQ(C[3], 50.0f);
    }
}

TEST(leading_dimensions_are_respected) {
    // A is the top-left 2x2 of a 2x3 buffer, B the top-left 2x2 of a 2x4 buffer, C written into a 2x5 buffer.
    const float A[] = {1, 2, 99, 3, 4, 99};
    const float B[] = {5, 6, 99, 99, 7, 8, 99, 99};
    for (int ki = 0; ki < kGemmKindCount; ++ki) {
        float C[10];
        for (float& v : C) v = -1.0f;
        gemm(static_cast<GemmKind>(ki), 2, 2, 2, A, 3, B, 4, C, 5);
        CHECK_EQ(C[0], 19.0f);
        CHECK_EQ(C[1], 22.0f);
        CHECK_EQ(C[5], 43.0f);
        CHECK_EQ(C[6], 50.0f);
        CHECK_EQ(C[2], -1.0f);  // untouched padding
        CHECK_EQ(C[9], -1.0f);
    }
}

TEST(rejects_bad_arguments) {
    float x[4] = {0, 0, 0, 0};
    CHECK_THROWS(gemm(GemmKind::Naive, 2, 2, 2, x, 1, x, 2, x, 2));  // lda < K
    CHECK_THROWS(gemm(GemmKind::Naive, -1, 2, 2, x, 2, x, 2, x, 2));
}

TEST(all_kinds_match_reference_on_awkward_sizes) {
    std::mt19937 rng(7);
    const int sizes[][3] = {{1, 1, 1},   {1, 17, 5},  {13, 1, 9},   {7, 7, 7},    {6, 16, 8},   {12, 32, 16},
                            {33, 37, 29}, {64, 64, 64}, {100, 130, 70}, {5, 16, 27}, {128, 10, 2048}, {97, 200, 288}};
    for (int kind = 0; kind < kGemmKindCount; ++kind) {
        for (const auto& s : sizes) check_kind_against_ref(static_cast<GemmKind>(kind), s[0], s[1], s[2], rng);
    }
}

TEST(tiled_is_correct_for_odd_tile_sizes) {
    std::mt19937 rng(11);
    const TileConfig saved = gemm_tiled_config();
    const TileConfig configs[] = {{1, 1, 1}, {5, 7, 3}, {64, 256, 256}, {1000, 1000, 1000}};
    for (const TileConfig& t : configs) {
        set_gemm_tiled_config(t);
        check_kind_against_ref(GemmKind::Tiled, 33, 37, 29, rng);
        check_kind_against_ref(GemmKind::Tiled, 100, 130, 70, rng);
    }
    set_gemm_tiled_config(saved);
    CHECK_THROWS(set_gemm_tiled_config({0, 1, 1}));
}

TEST(threaded_is_bit_identical_to_simd) {
    // Same packing, same micro-kernel, same k order per element: splitting M across threads
    // must not change a single bit.
    std::mt19937 rng(5);
    const int M = 500, N = 70, K = 300;
    const std::vector<float> A = random_matrix(static_cast<size_t>(M) * K, rng);
    const std::vector<float> B = random_matrix(static_cast<size_t>(K) * N, rng);
    std::vector<float> c1(static_cast<size_t>(M) * N), c2(static_cast<size_t>(M) * N);
    gemm(GemmKind::Simd, M, N, K, A.data(), K, B.data(), N, c1.data(), N);
    gemm(GemmKind::Threaded, M, N, K, A.data(), K, B.data(), N, c2.data(), N);
    CHECK(c1 == c2);
}

namespace {
// Independent implementation of the packed-A layout documented in gemm.h.
std::vector<float> pack_a_by_formula(const std::vector<float>& A, int M, int K) {
    const PackedALayout L = gemm_packed_a_layout();
    const int m_pad = (M + L.mr - 1) / L.mr * L.mr;
    std::vector<float> out(static_cast<size_t>(gemm_packed_a_size(M, K)), 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            const int b = k / L.kc;
            const int kc_b = std::min(L.kc, K - b * L.kc);
            const size_t off = static_cast<size_t>(b) * m_pad * L.kc + static_cast<size_t>(m / L.mr) * kc_b * L.mr +
                               static_cast<size_t>(k % L.kc) * L.mr + static_cast<size_t>(m % L.mr);
            out[off] = A[static_cast<size_t>(m) * K + k];
        }
    }
    return out;
}
}  // namespace

TEST(prepacked_a_is_bit_identical_to_simd) {
    std::mt19937 rng(21);
    const int sizes[][3] = {{7, 7, 7}, {100, 130, 70}, {150, 40, 600}, {72, 16, 256}, {73, 17, 257}, {6, 16, 27}};
    for (const auto& s : sizes) {
        const int M = s[0], N = s[1], K = s[2];
        const std::vector<float> A = random_matrix(static_cast<size_t>(M) * K, rng);
        const std::vector<float> B = random_matrix(static_cast<size_t>(K) * N, rng);
        const std::vector<float> Ap = pack_a_by_formula(A, M, K);
        std::vector<float> ref(static_cast<size_t>(M) * N), c1(ref.size(), -1.0f), c2(ref.size(), -1.0f);
        gemm(GemmKind::Simd, M, N, K, A.data(), K, B.data(), N, ref.data(), N);
        gemm_packed_a(GemmKind::Simd, M, N, K, Ap.data(), B.data(), N, c1.data(), N);
        gemm_packed_a(GemmKind::Threaded, M, N, K, Ap.data(), B.data(), N, c2.data(), N);
        CHECK(c1 == ref);
        CHECK(c2 == ref);
    }
    float x[4] = {0, 0, 0, 0};
    CHECK_THROWS(gemm_packed_a(GemmKind::Naive, 2, 2, 2, x, x, 2, x, 2));
    CHECK(gemm_kind_uses_packed_a(GemmKind::Simd));
    CHECK(!gemm_kind_uses_packed_a(GemmKind::Tiled));
}

TEST(kind_names_round_trip) {
    for (int i = 0; i < kGemmKindCount; ++i) {
        GemmKind k;
        CHECK(parse_gemm_kind(gemm_kind_name(static_cast<GemmKind>(i)), k));
        CHECK(k == static_cast<GemmKind>(i));
    }
    GemmKind k;
    CHECK(!parse_gemm_kind("bogus", k));
}
