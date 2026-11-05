#pragma once
// Reader and writer for the TINF tensor container (docs/format.md).
//
// The reader treats the file as untrusted input: every length is bounds
// checked against the buffer before it is used, and any violation throws
// std::runtime_error with the file name and the reason.

#include <cstdint>
#include <string>
#include <vector>

#include "tinyinfer/tensor.h"

namespace tinyinfer {

enum class DType : uint32_t { F32 = 0, I8 = 1, U8 = 2, I32 = 3 };

const char* dtype_name(DType dt);
size_t dtype_size(DType dt);

struct RawTensor {
    std::string name;
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;  // little-endian payload, exactly dtype_size * numel bytes
};

class TensorFile {
public:
    static constexpr uint32_t kVersion = 1;
    static constexpr size_t kMaxFileBytes = size_t{1} << 30;  // 1 GiB
    static constexpr size_t kMaxNameLen = 256;
    static constexpr size_t kMaxNdim = 4;

    // Read and parse a file from disk.
    static TensorFile read(const std::string& path);
    // Parse an in-memory buffer. `what` names the source in error messages.
    static TensorFile parse(const std::vector<uint8_t>& buf, const std::string& what);

    bool has(const std::string& name) const;
    const RawTensor& get(const std::string& name) const;  // throws if missing
    std::vector<std::string> names() const;
    size_t size() const { return tensors_.size(); }

    // Typed copies. Throw if the dtype does not match.
    Tensor f32(const std::string& name) const;
    TensorU8 u8(const std::string& name) const;
    TensorI8 i8(const std::string& name) const;
    TensorI32 i32(const std::string& name) const;

private:
    std::vector<RawTensor> tensors_;
};

// Serialise tensors in the given order. Validates names and shapes the same
// way the reader does, so a file this writes always loads.
std::vector<uint8_t> serialize_tensors(const std::vector<RawTensor>& tensors);
void write_tensor_file(const std::string& path, const std::vector<RawTensor>& tensors);

// Helpers to build RawTensors from typed tensors.
RawTensor raw_from(const std::string& name, const Tensor& t);
RawTensor raw_from(const std::string& name, const TensorI8& t);
RawTensor raw_from(const std::string& name, const TensorU8& t);
RawTensor raw_from(const std::string& name, const TensorI32& t);

}  // namespace tinyinfer
