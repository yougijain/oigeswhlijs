# M1: model

Three 3x3 convolutions (ReLU, 2x2 max-pool after each) and one linear layer,
trained on CIFAR-10 in PyTorch. No batch norm, no dropout: the C++ engine
implements exactly the five ops below and nothing else.

| Layer | Output (CxHxW) | Params | MACs / image |
|---|---|---:|---:|
| conv1 3x3, pad 1 | 32x32x32 | 896 | 884,736 |
| relu + maxpool 2x2 | 32x16x16 | 0 | 0 |
| conv2 3x3, pad 1 | 64x16x16 | 18,496 | 4,718,592 |
| relu + maxpool 2x2 | 64x8x8 | 0 | 0 |
| conv3 3x3, pad 1 | 128x8x8 | 73,856 | 4,718,592 |
| relu + maxpool 2x2 | 128x4x4 | 0 | 0 |
| flatten | 2048 | 0 | 0 |
| fc | 10 | 20,490 | 20,480 |

- Parameters: **113,738** (454,952 bytes as float32)
- Forward cost: 20,684,800 FLOPs per image (multiply-add = 2 FLOPs)

## Parameter tensors (PyTorch layout, as exported)

| Name | Shape | Count |
|---|---|---:|
| `conv1.weight` | [32, 3, 3, 3] | 864 |
| `conv1.bias` | [32] | 32 |
| `conv2.weight` | [64, 32, 3, 3] | 18,432 |
| `conv2.bias` | [64] | 64 |
| `conv3.weight` | [128, 64, 3, 3] | 73,728 |
| `conv3.bias` | [128] | 128 |
| `fc.weight` | [10, 2048] | 20,480 |
| `fc.bias` | [10] | 10 |

## Training

- Data: 50,000 train / 10,000 test images, random-crop (pad 4) + horizontal flip
- Normalisation: per-channel mean/std from `python/model.py`
- Optimiser: SGD, momentum 0.9 (Nesterov), weight decay 0.0005, batch 128
- Schedule: OneCycle, peak LR 0.05, 20 epochs, seed 0
- Wall time: 7.1 min on Intel(R) Xeon(R) Processor @ 2.10GHz (4 threads), torch 2.14.1+cu130

## Result

- **FP32 test accuracy: 83.71%** on all 10,000 test images (final epoch)
- Best epoch: 20 at 83.71% (the final-epoch weights are the ones exported)
- Per-epoch log: `results/train_log.csv`

This is a deliberately small model. The point of the project is the engine
that runs it, so accuracy is recorded, not tuned.
