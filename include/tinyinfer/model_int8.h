#pragma once
// TinyCNN with INT8 convolutions and linear layer. Same topology as TinyCNN;
// every GEMM runs in int8 x int8 -> int32 and is dequantised to float right
// after, so relu, max-pool and the hand-off between layers stay in float.

#include <cstdint>
#include <string>
#include <vector>

#include "tinyinfer/gemm_int8.h"
#include "tinyinfer/loader.h"
#include "tinyinfer/model.h"
#include "tinyinfer/quant.h"
#include "tinyinfer/tensor.h"

namespace tinyinfer {

struct ProfileInt8 {
    double preprocess_ms = 0;
    QConvTimes conv[3];  // dequant_ms includes the fused ReLU
    double pool_ms = 0;
    double fc_ms = 0;
    int64_t images = 0;
    double total_ms() const;
    std::string table() const;
};

class TinyCNNInt8 {
public:
    // Quantise a float model. Each layer's activation scale is max |input| / 127 over the
    // calibration images, measured by running them through the float model.
    static TinyCNNInt8 quantize(TinyCNN& fp32, const TensorU8& calib, GemmKind calib_kind = default_gemm_kind());

    // weights_int8.bin round trip. The file holds int8 weights in the engine's [K][N]
    // layout plus per-channel weight scales, per-layer input scales, float biases and
    // the normalisation constants (docs/format.md lists the tensor names).
    static TinyCNNInt8 load(const TensorFile& file);
    std::vector<RawTensor> to_raw() const;
    void save(const std::string& path) const;

    Tensor forward(const TensorU8& images, GemmInt8Kind kind, ProfileInt8* prof = nullptr);

    int64_t weight_bytes() const;  // int8 weights + scales + float biases
    const QConv2d& conv(int i) const { return convs_[i]; }
    const QLinear& fc() const { return fc_; }

private:
    QConv2d convs_[3];
    QLinear fc_;
    float mean_[3] = {0, 0, 0};
    float std_[3] = {1, 1, 1};
    Tensor x_, y_, flat_, logits_;
    TensorI8 xq_, col_;
    TensorI32 acc_;
};

}  // namespace tinyinfer
