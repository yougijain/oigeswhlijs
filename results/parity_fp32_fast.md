# M2: parity with PyTorch

The C++ FP32 forward pass was run on the exported CIFAR-10 test images and its
logits compared element-wise with the logits PyTorch (FP32) produced from the same
weights and pixels (`data/export/ref_logits.bin`).

| Quantity | Value |
|---|---:|
| Test images | 1000 |
| Max abs error vs PyTorch | 7.7188e-06 |
| Mean abs error vs PyTorch | 8.42034e-07 |
| Argmax agreement | 1000 / 1000 (100%) |
| Accuracy, C++ engine | 83.2% |
| Accuracy, PyTorch, same images | 83.2% |
| Engine | FP32, gemm threaded (AVX2+FMA, 6x16 micro-kernel) |
| Batch size | 64 |
| Wall time | 163.532 ms (0.163532 ms/image) |

Run on Intel(R) Xeon(R) Processor @ 2.10GHz, GCC 13.3.0, flags `-Wall -Wextra -Wpedantic -Wshadow -Werror -march=native -fopenmp-simd -fopenmp`, 4 threads.

## Where the time goes

| Stage | Total ms | Per image ms | Share |
|---|---:|---:|---:|
| preprocess | 7.48 | 0.0075 | 4.6% |
| conv1 im2col | 11.89 | 0.0119 | 7.3% |
| conv1 gemm | 10.22 | 0.0102 | 6.3% |
| conv1 bias+relu | 8.25 | 0.0082 | 5.1% |
| conv2 im2col | 23.92 | 0.0239 | 14.7% |
| conv2 gemm | 35.53 | 0.0355 | 21.8% |
| conv2 bias+relu | 4.01 | 0.0040 | 2.5% |
| conv3 im2col | 9.22 | 0.0092 | 5.7% |
| conv3 gemm | 34.20 | 0.0342 | 21.0% |
| conv3 bias+relu | 2.05 | 0.0020 | 1.3% |
| maxpool (x3) | 12.98 | 0.0130 | 8.0% |
| fc | 3.18 | 0.0032 | 2.0% |
| total | 162.92 | 0.1629 | 100.0% |
