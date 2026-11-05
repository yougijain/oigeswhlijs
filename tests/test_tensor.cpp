#include "tinyinfer/tensor.h"

#include <stdexcept>

#include "check.h"

using namespace tinyinfer;

TEST(numel_and_shape) {
    Tensor t({2, 3, 4});
    CHECK_EQ(t.numel(), 24);
    CHECK_EQ(t.ndim(), size_t{3});
    CHECK_EQ(t.dim(2), 4);
    CHECK_EQ(shape_str(t.shape), std::string("[2, 3, 4]"));
    for (float v : t.data) CHECK_EQ(v, 0.0f);
}

TEST(reshape_keeps_data) {
    Tensor t({2, 6});
    for (size_t i = 0; i < t.data.size(); ++i) t.data[i] = static_cast<float>(i);
    t.reshape({3, 4});
    CHECK_EQ(t.dim(0), 3);
    CHECK_EQ(t.data[11], 11.0f);
    CHECK_THROWS(t.reshape({5, 5}));
}

TEST(bad_shapes_throw) {
    CHECK_THROWS(checked_numel({}));
    CHECK_THROWS(checked_numel({0}));
    CHECK_THROWS(checked_numel({3, -1}));
    CHECK_THROWS(checked_numel({int64_t{1} << 40, int64_t{1} << 40}));
    CHECK_EQ(checked_numel({7, 11}), 77);
}

TEST(resize_grows_and_shrinks) {
    Tensor t({4});
    t.resize({2, 8});
    CHECK_EQ(t.numel(), 16);
    t.resize({3});
    CHECK_EQ(t.numel(), 3);
}
