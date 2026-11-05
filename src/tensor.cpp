#include "tinyinfer/tensor.h"

#include <limits>
#include <stdexcept>

namespace tinyinfer {

int64_t checked_numel(const std::vector<int64_t>& shape) {
    if (shape.empty()) throw std::invalid_argument("tensor shape must have at least one dim");
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d <= 0) throw std::invalid_argument("tensor dim must be positive, got " + shape_str(shape));
        if (d > std::numeric_limits<int64_t>::max() / n) {
            throw std::invalid_argument("tensor too large: " + shape_str(shape));
        }
        n *= d;
    }
    return n;
}

std::string shape_str(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(shape[i]);
    }
    return s + "]";
}

}  // namespace tinyinfer
