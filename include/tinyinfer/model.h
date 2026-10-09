#pragma once
// TinyCNN: the one network this engine runs.
//   conv3x3(3->32)  relu pool
//   conv3x3(32->64) relu pool
//   conv3x3(64->128) relu pool
//   flatten -> linear(2048->10)

#include <cstdint>
#include <string>

#include "tinyinfer/gemm.h"
#include "tinyinfer/layers.h"
#include "tinyinfer/loader.h"
#include "tinyinfer/tensor.h"

namespace tinyinfer {

// Per-stage wall time accumulated over forward() calls, milliseconds.
struct Profile {
    double preprocess_ms = 0;
    ConvTimes conv[3];  // bias_ms includes the fused ReLU
    double pool_ms = 0;
    double fc_ms = 0;
    int64_t images = 0;
    double total_ms() const;
    std::string table() const;  // markdown
};

// Largest |value| seen at each quantised layer's input: conv1, conv2, conv3, fc.
// Filled by forward() when requested; INT8 calibration reads it.
struct LayerInputStats {
    float max_abs[4] = {0, 0, 0, 0};
};

class TinyCNN {
public:
    static constexpr int kImageSize = 32;
    static constexpr int kChannels = 3;
    static constexpr int kClasses = 10;

    // Reads conv{1,2,3}.{weight,bias}, fc.{weight,bias}, norm.mean, norm.std.
    static TinyCNN load(const TensorFile& file);

    // images: u8 NCHW [batch][3][32][32], the raw CIFAR-10 pixels. Returns logits [batch][10].
    Tensor forward(const TensorU8& images, GemmKind kind, Profile* prof = nullptr, LayerInputStats* stats = nullptr);

    // uint8 NCHW -> normalised float NHWC, the same arithmetic as python/model.py normalize().
    void preprocess(const TensorU8& images, Tensor& out) const;

    int64_t param_count() const;
    const Conv2d& conv(int i) const { return convs_[i]; }
    const Linear& fc() const { return fc_; }
    float mean(int c) const { return mean_[c]; }
    float stddev(int c) const { return std_[c]; }

private:
    Conv2d convs_[3];
    Linear fc_;
    float mean_[3] = {0, 0, 0};
    float std_[3] = {1, 1, 1};
    // Scratch, reused across calls so steady-state inference does not allocate.
    Tensor x_, y_, col_, flat_, logits_;
};

}  // namespace tinyinfer
