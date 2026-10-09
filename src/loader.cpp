#include "tinyinfer/loader.h"

#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace tinyinfer {

namespace {

// Little-endian decode without relying on host byte order or alignment.
uint32_t load_u32(const uint8_t* p) {
    return uint32_t{p[0]} | (uint32_t{p[1]} << 8) | (uint32_t{p[2]} << 16) | (uint32_t{p[3]} << 24);
}
void store_u32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

bool host_is_little_endian() {
    const uint32_t one = 1;
    uint8_t first;
    std::memcpy(&first, &one, 1);
    return first == 1;
}

// Bounds-checked cursor over an untrusted byte buffer.
class Cursor {
public:
    Cursor(const std::vector<uint8_t>& buf, const std::string& what) : buf_(buf), what_(what) {}

    uint32_t u32(const char* field) {
        need(4, field);
        uint32_t v = load_u32(buf_.data() + pos_);
        pos_ += 4;
        return v;
    }
    const uint8_t* bytes(size_t n, const char* field) {
        need(n, field);
        const uint8_t* p = buf_.data() + pos_;
        pos_ += n;
        return p;
    }
    size_t remaining() const { return buf_.size() - pos_; }
    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error(what_ + ": " + msg + " (at byte " + std::to_string(pos_) + ")");
    }

private:
    void need(size_t n, const char* field) const {
        if (n > remaining()) fail(std::string("truncated while reading ") + field);
    }
    const std::vector<uint8_t>& buf_;
    const std::string& what_;
    size_t pos_ = 0;
};

bool valid_name_byte(uint8_t b) { return b >= 0x21 && b <= 0x7E; }

void validate_name(const std::string& name) {
    if (name.empty() || name.size() > TensorFile::kMaxNameLen) {
        throw std::runtime_error("tensor name length out of range: " + std::to_string(name.size()));
    }
    for (unsigned char c : name) {
        if (!valid_name_byte(c)) throw std::runtime_error("tensor name has an invalid byte: " + name);
    }
}

template <typename T>
TensorT<T> typed_copy(const RawTensor& r, DType expect) {
    if (r.dtype != expect) {
        throw std::runtime_error("tensor " + r.name + " has dtype " + dtype_name(r.dtype) + ", expected " +
                                 dtype_name(expect));
    }
    TensorT<T> t(r.shape);
    std::memcpy(t.data.data(), r.bytes.data(), r.bytes.size());
    return t;
}

template <typename T>
RawTensor raw_from_typed(const std::string& name, const TensorT<T>& t, DType dt) {
    RawTensor r;
    r.name = name;
    r.dtype = dt;
    r.shape = t.shape;
    r.bytes.resize(t.data.size() * sizeof(T));
    if (!r.bytes.empty()) std::memcpy(r.bytes.data(), t.data.data(), r.bytes.size());
    return r;
}

}  // namespace

const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::F32: return "f32";
        case DType::I8: return "i8";
        case DType::U8: return "u8";
        case DType::I32: return "i32";
    }
    return "?";
}

size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F32: return 4;
        case DType::I8: return 1;
        case DType::U8: return 1;
        case DType::I32: return 4;
    }
    return 0;
}

TensorFile TensorFile::read(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error(path + ": cannot open");
    const std::streamoff size = in.tellg();
    if (size < 0) throw std::runtime_error(path + ": cannot determine size");
    if (static_cast<uint64_t>(size) > kMaxFileBytes) {
        throw std::runtime_error(path + ": file larger than the 1 GiB limit");
    }
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    in.seekg(0);
    if (!buf.empty() && !in.read(reinterpret_cast<char*>(buf.data()), size)) {
        throw std::runtime_error(path + ": short read");
    }
    return parse(buf, path);
}

TensorFile TensorFile::parse(const std::vector<uint8_t>& buf, const std::string& what) {
    if (!host_is_little_endian()) throw std::runtime_error("TINF loader requires a little-endian host");
    Cursor cur(buf, what);
    const uint8_t* magic = cur.bytes(4, "magic");
    if (std::memcmp(magic, "TINF", 4) != 0) cur.fail("bad magic, not a TINF file");
    const uint32_t version = cur.u32("version");
    if (version != kVersion) cur.fail("unsupported version " + std::to_string(version));
    const uint32_t count = cur.u32("tensor count");
    // Each record needs at least name_len(4)+name(1)+dtype(4)+ndim(4)+dim(4)+data(1) bytes.
    if (count > cur.remaining() / 18) cur.fail("tensor count " + std::to_string(count) + " exceeds file size");

    TensorFile tf;
    tf.tensors_.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        RawTensor r;
        const uint32_t name_len = cur.u32("name length");
        if (name_len == 0 || name_len > kMaxNameLen) cur.fail("name length " + std::to_string(name_len) + " out of range");
        const uint8_t* name = cur.bytes(name_len, "name");
        r.name.assign(reinterpret_cast<const char*>(name), name_len);
        for (unsigned char c : r.name) {
            if (!valid_name_byte(c)) cur.fail("tensor name contains an invalid byte");
        }
        const uint32_t dtype = cur.u32("dtype");
        if (dtype > static_cast<uint32_t>(DType::I32)) cur.fail("unknown dtype " + std::to_string(dtype) + " for " + r.name);
        r.dtype = static_cast<DType>(dtype);
        const uint32_t ndim = cur.u32("ndim");
        if (ndim == 0 || ndim > kMaxNdim) cur.fail("ndim " + std::to_string(ndim) + " out of range for " + r.name);
        uint64_t numel = 1;
        for (uint32_t d = 0; d < ndim; ++d) {
            const uint32_t dim = cur.u32("dim");
            if (dim == 0) cur.fail("zero dim in " + r.name);
            numel *= dim;  // numel < 2^32 and dim < 2^32 here, so this cannot wrap uint64
            if (numel > std::numeric_limits<uint32_t>::max()) cur.fail("element count overflows 32 bits in " + r.name);
            r.shape.push_back(dim);
        }
        const uint64_t nbytes = numel * dtype_size(r.dtype);
        if (nbytes > cur.remaining()) cur.fail("data for " + r.name + " runs past end of file");
        const uint8_t* data = cur.bytes(static_cast<size_t>(nbytes), "tensor data");
        r.bytes.assign(data, data + nbytes);
        for (const RawTensor& prev : tf.tensors_) {
            if (prev.name == r.name) cur.fail("duplicate tensor name " + r.name);
        }
        tf.tensors_.push_back(std::move(r));
    }
    if (cur.remaining() != 0) cur.fail(std::to_string(cur.remaining()) + " trailing bytes after last tensor");
    return tf;
}

bool TensorFile::has(const std::string& name) const {
    for (const RawTensor& t : tensors_) {
        if (t.name == name) return true;
    }
    return false;
}

const RawTensor& TensorFile::get(const std::string& name) const {
    for (const RawTensor& t : tensors_) {
        if (t.name == name) return t;
    }
    throw std::runtime_error("tensor not found: " + name);
}

std::vector<std::string> TensorFile::names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const RawTensor& t : tensors_) out.push_back(t.name);
    return out;
}

Tensor TensorFile::f32(const std::string& name) const { return typed_copy<float>(get(name), DType::F32); }
TensorU8 TensorFile::u8(const std::string& name) const { return typed_copy<uint8_t>(get(name), DType::U8); }
TensorI8 TensorFile::i8(const std::string& name) const { return typed_copy<int8_t>(get(name), DType::I8); }
TensorI32 TensorFile::i32(const std::string& name) const { return typed_copy<int32_t>(get(name), DType::I32); }

std::vector<uint8_t> serialize_tensors(const std::vector<RawTensor>& tensors) {
    if (!host_is_little_endian()) throw std::runtime_error("TINF writer requires a little-endian host");
    std::vector<uint8_t> out;
    out.insert(out.end(), {'T', 'I', 'N', 'F'});
    store_u32(out, TensorFile::kVersion);
    store_u32(out, static_cast<uint32_t>(tensors.size()));
    for (size_t i = 0; i < tensors.size(); ++i) {
        const RawTensor& t = tensors[i];
        validate_name(t.name);
        for (size_t j = 0; j < i; ++j) {
            if (tensors[j].name == t.name) throw std::runtime_error("duplicate tensor name " + t.name);
        }
        if (t.shape.empty() || t.shape.size() > TensorFile::kMaxNdim) {
            throw std::runtime_error("tensor " + t.name + ": ndim out of range");
        }
        const int64_t numel = checked_numel(t.shape);
        if (numel > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("tensor " + t.name + " too large");
        for (int64_t d : t.shape) {
            if (d > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("tensor " + t.name + ": dim too large");
        }
        if (t.bytes.size() != static_cast<size_t>(numel) * dtype_size(t.dtype)) {
            throw std::runtime_error("tensor " + t.name + ": payload size does not match shape");
        }
        store_u32(out, static_cast<uint32_t>(t.name.size()));
        out.insert(out.end(), t.name.begin(), t.name.end());
        store_u32(out, static_cast<uint32_t>(t.dtype));
        store_u32(out, static_cast<uint32_t>(t.shape.size()));
        for (int64_t d : t.shape) store_u32(out, static_cast<uint32_t>(d));
        out.insert(out.end(), t.bytes.begin(), t.bytes.end());
    }
    return out;
}

void write_tensor_file(const std::string& path, const std::vector<RawTensor>& tensors) {
    const std::vector<uint8_t> buf = serialize_tensors(tensors);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error(path + ": cannot open for writing");
    out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (!out) throw std::runtime_error(path + ": write failed");
}

RawTensor raw_from(const std::string& name, const Tensor& t) { return raw_from_typed(name, t, DType::F32); }
RawTensor raw_from(const std::string& name, const TensorI8& t) { return raw_from_typed(name, t, DType::I8); }
RawTensor raw_from(const std::string& name, const TensorU8& t) { return raw_from_typed(name, t, DType::U8); }
RawTensor raw_from(const std::string& name, const TensorI32& t) { return raw_from_typed(name, t, DType::I32); }

}  // namespace tinyinfer
