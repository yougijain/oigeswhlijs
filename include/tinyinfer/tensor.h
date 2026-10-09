#pragma once
// Dense, contiguous, row-major tensors. Memory is owned by std::vector so a
// Tensor cleans up after itself (RAII) and copies/moves behave like values.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tinyinfer {

// std::allocator whose value-initialisation is default-initialisation: for
// trivial T, vector::resize() then leaves new elements uninitialised instead of
// zero-filling them. The engine's scratch tensors shrink and regrow between
// layers of different sizes on every forward pass; with std::vector that was
// a memset of every regrown byte (7% of all instructions under callgrind).
template <typename T, typename Base = std::allocator<T>>
struct DefaultInitAllocator : Base {
    using Base::Base;
    template <typename U>
    struct rebind {
        using other = DefaultInitAllocator<U, typename std::allocator_traits<Base>::template rebind_alloc<U>>;
    };
    template <typename U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value) {
        ::new (static_cast<void*>(p)) U;
    }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        std::allocator_traits<Base>::construct(static_cast<Base&>(*this), p, std::forward<Args>(args)...);
    }
};

template <typename T>
using Buffer = std::vector<T, DefaultInitAllocator<T>>;

// Checked product of dims. Throws std::invalid_argument on a zero or negative
// dim, or if the element count would not fit in int64_t.
int64_t checked_numel(const std::vector<int64_t>& shape);

std::string shape_str(const std::vector<int64_t>& shape);

template <typename T>
struct TensorT {
    std::vector<int64_t> shape;
    Buffer<T> data;

    TensorT() = default;
    // A freshly constructed tensor is zero-filled; only resize() leaves new elements undefined.
    explicit TensorT(std::vector<int64_t> s) : shape(std::move(s)), data(static_cast<size_t>(checked_numel(shape))) {
        std::fill(data.begin(), data.end(), T{});
    }

    int64_t numel() const { return static_cast<int64_t>(data.size()); }
    size_t ndim() const { return shape.size(); }
    int64_t dim(size_t i) const { return shape.at(i); }
    T* ptr() { return data.data(); }
    const T* ptr() const { return data.data(); }

    // Reshape without copying. The element count must match.
    void reshape(std::vector<int64_t> s) {
        if (checked_numel(s) != numel()) {
            throw std::invalid_argument("reshape: element count mismatch " + shape_str(shape) + " -> " +
                                        shape_str(s));
        }
        shape = std::move(s);
    }
    // Resize to a new shape, reallocating only if the element count grows. New elements are
    // not initialised: every caller writes the whole tensor before reading it.
    void resize(std::vector<int64_t> s) {
        data.resize(static_cast<size_t>(checked_numel(s)));
        shape = std::move(s);
    }
};

using Tensor = TensorT<float>;
using TensorI8 = TensorT<int8_t>;
using TensorU8 = TensorT<uint8_t>;
using TensorI32 = TensorT<int32_t>;

}  // namespace tinyinfer
