#include "tinyinfer/loader.h"

#include <cstring>
#include <stdexcept>

#include "check.h"

using namespace tinyinfer;

namespace {

std::vector<RawTensor> sample() {
    Tensor a({2, 3});
    for (size_t i = 0; i < 6; ++i) a.data[i] = static_cast<float>(i) * 0.5f;
    TensorU8 b({4});
    b.data = {1, 2, 3, 250};
    TensorI8 c({1, 2});
    c.data = {-128, 127};
    TensorI32 d({1});
    d.data[0] = -123456;
    return {raw_from("a.weight", a), raw_from("b", b), raw_from("c", c), raw_from("d", d)};
}

}  // namespace

TEST(round_trip) {
    const std::vector<uint8_t> buf = serialize_tensors(sample());
    const TensorFile f = TensorFile::parse(buf, "mem");
    CHECK_EQ(f.size(), size_t{4});
    CHECK(f.has("a.weight"));
    CHECK(!f.has("nope"));
    const Tensor a = f.f32("a.weight");
    CHECK_EQ(a.dim(0), 2);
    CHECK_EQ(a.dim(1), 3);
    CHECK_EQ(a.data[5], 2.5f);
    CHECK_EQ(f.u8("b").data[3], uint8_t{250});
    CHECK_EQ(f.i8("c").data[0], int8_t{-128});
    CHECK_EQ(f.i32("d").data[0], -123456);
    CHECK_THROWS(f.f32("b"));  // wrong dtype
    CHECK_THROWS(f.get("nope"));
}

TEST(header_is_as_specified) {
    const std::vector<uint8_t> buf = serialize_tensors(sample());
    CHECK(std::memcmp(buf.data(), "TINF", 4) == 0);
    CHECK_EQ(buf[4], 1);  // version
    CHECK_EQ(buf[8], 4);  // count
    CHECK_EQ(buf[12], 8);  // name_len of "a.weight"
}

TEST(rejects_bad_magic) {
    std::vector<uint8_t> buf = serialize_tensors(sample());
    buf[0] = 'X';
    CHECK_THROWS(TensorFile::parse(buf, "mem"));
}

TEST(rejects_bad_version) {
    std::vector<uint8_t> buf = serialize_tensors(sample());
    buf[4] = 2;
    CHECK_THROWS(TensorFile::parse(buf, "mem"));
}

TEST(rejects_truncation_anywhere) {
    const std::vector<uint8_t> full = serialize_tensors(sample());
    for (size_t n = 0; n < full.size(); n += 7) {
        std::vector<uint8_t> cut(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(n));
        CHECK_THROWS(TensorFile::parse(cut, "mem"));
    }
}

TEST(rejects_trailing_bytes) {
    std::vector<uint8_t> buf = serialize_tensors(sample());
    buf.push_back(0);
    CHECK_THROWS(TensorFile::parse(buf, "mem"));
}

TEST(rejects_huge_count) {
    std::vector<uint8_t> buf = serialize_tensors(sample());
    buf[8] = 0xFF;
    buf[9] = 0xFF;
    buf[10] = 0xFF;
    buf[11] = 0x7F;
    CHECK_THROWS(TensorFile::parse(buf, "mem"));
}

TEST(rejects_oversized_dims) {
    // One tensor "x", f32, shape [0xFFFFFFFF, 0xFFFFFFFF]: must fail on the element count, not allocate.
    std::vector<uint8_t> buf = {'T', 'I', 'N', 'F', 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 'x', 0, 0, 0, 0, 2, 0, 0, 0,
                                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    CHECK_THROWS(TensorFile::parse(buf, "mem"));
    // Single dim 0x40000000 f32 = 4 GiB of data claimed: must fail on bounds, not allocate.
    buf = {'T', 'I', 'N', 'F', 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 'x', 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0x40};
    CHECK_THROWS(TensorFile::parse(buf, "mem"));
}

TEST(rejects_zero_dim_and_bad_ndim) {
    std::vector<uint8_t> buf = {'T', 'I', 'N', 'F', 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 'x', 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    CHECK_THROWS(TensorFile::parse(buf, "mem"));
    buf = {'T', 'I', 'N', 'F', 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 'x', 0, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0,
           1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    CHECK_THROWS(TensorFile::parse(buf, "mem"));
}

TEST(rejects_bad_names) {
    std::vector<RawTensor> t = sample();
    t[0].name = "has space";
    CHECK_THROWS(serialize_tensors(t));
    t = sample();
    t[1].name = t[0].name;  // duplicate
    CHECK_THROWS(serialize_tensors(t));
    t = sample();
    t[0].name.clear();
    CHECK_THROWS(serialize_tensors(t));
    // And the same duplicate/invalid-name cases coming from the wire.
    std::vector<uint8_t> buf = serialize_tensors(sample());
    buf[16] = ' ';  // first byte of "a.weight"
    CHECK_THROWS(TensorFile::parse(buf, "mem"));
}

TEST(rejects_payload_size_mismatch_on_write) {
    std::vector<RawTensor> t = sample();
    t[0].bytes.pop_back();
    CHECK_THROWS(serialize_tensors(t));
}

TEST(read_missing_file_throws) { CHECK_THROWS(TensorFile::read("/nonexistent/definitely/not/here.bin")); }
