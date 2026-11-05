# INT8 parity with PyTorch

The C++ INT8 forward pass was run on the exported CIFAR-10 test images and its
logits compared element-wise with the logits PyTorch (FP32) produced from the same
weights and pixels (`data/export/ref_logits.bin`).

| Quantity | Value |
|---|---:|
| Test images | 1000 |
| Max abs error vs PyTorch | 1.29475 |
| Mean abs error vs PyTorch | 0.0705123 |
| Argmax agreement | 993 / 1000 (99.3%) |
| Accuracy, C++ engine | 83.2% |
| Accuracy, PyTorch, same images | 83.2% |
| Engine | INT8, gemm threaded (AVX-VNNI (vpdpbusd), 6x16 micro-kernel) |
| Batch size | 64 |
| Wall time | 105.854 ms (0.105854 ms/image) |

Run on Intel(R) Xeon(R) Processor @ 2.10GHz, GCC 13.3.0, flags `-Wall -Wextra -Wpedantic -Wshadow -Werror -march=native -fopenmp-simd -fopenmp`, 4 threads.

## Where the time goes

| Stage | Total ms | Per image ms | Share |
|---|---:|---:|---:|
| preprocess | 7.20 | 0.0072 | 6.8% |
| conv1 quantize | 0.95 | 0.0010 | 0.9% |
| conv1 im2col (int8) | 8.81 | 0.0088 | 8.4% |
| conv1 gemm (int8) | 7.71 | 0.0077 | 7.3% |
| conv1 dequant+bias+relu | 15.33 | 0.0153 | 14.6% |
| conv2 quantize | 2.63 | 0.0026 | 2.5% |
| conv2 im2col (int8) | 7.35 | 0.0074 | 7.0% |
| conv2 gemm (int8) | 13.60 | 0.0136 | 12.9% |
| conv2 dequant+bias+relu | 5.66 | 0.0057 | 5.4% |
| conv3 quantize | 1.21 | 0.0012 | 1.1% |
| conv3 im2col (int8) | 2.14 | 0.0021 | 2.0% |
| conv3 gemm (int8) | 15.22 | 0.0152 | 14.5% |
| conv3 dequant+bias+relu | 2.55 | 0.0026 | 2.4% |
| maxpool (x3) | 12.28 | 0.0123 | 11.7% |
| fc (quantize+gemm+dequant) | 2.60 | 0.0026 | 2.5% |
| total | 105.24 | 0.1052 | 100.0% |
