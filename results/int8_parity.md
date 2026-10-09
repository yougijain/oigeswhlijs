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
| Wall time | 119.031 ms (0.119031 ms/image) |

Run on Intel(R) Xeon(R) Processor @ 2.10GHz, GCC 13.3.0, flags `-Wall -Wextra -Wpedantic -Wshadow -Werror -march=native -fopenmp-simd -fopenmp`, 4 threads.

## Where the time goes

| Stage | Total ms | Per image ms | Share |
|---|---:|---:|---:|
| preprocess | 6.77 | 0.0068 | 5.7% |
| conv1 quantize | 0.86 | 0.0009 | 0.7% |
| conv1 im2col (int8) | 16.91 | 0.0169 | 14.3% |
| conv1 gemm (int8) | 9.47 | 0.0095 | 8.0% |
| conv1 dequant+bias+relu | 13.72 | 0.0137 | 11.6% |
| conv2 quantize | 2.72 | 0.0027 | 2.3% |
| conv2 im2col (int8) | 9.59 | 0.0096 | 8.1% |
| conv2 gemm (int8) | 16.52 | 0.0165 | 13.9% |
| conv2 dequant+bias+relu | 5.33 | 0.0053 | 4.5% |
| conv3 quantize | 1.21 | 0.0012 | 1.0% |
| conv3 im2col (int8) | 2.98 | 0.0030 | 2.5% |
| conv3 gemm (int8) | 16.43 | 0.0164 | 13.9% |
| conv3 dequant+bias+relu | 2.44 | 0.0024 | 2.1% |
| maxpool (x3) | 11.38 | 0.0114 | 9.6% |
| fc (quantize+gemm+dequant) | 2.14 | 0.0021 | 1.8% |
| total | 118.48 | 0.1185 | 100.0% |
