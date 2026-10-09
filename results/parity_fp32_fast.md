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
| Wall time | 193.299 ms (0.193299 ms/image) |

Run on Intel(R) Xeon(R) Processor @ 2.10GHz, GCC 13.3.0, flags `-Wall -Wextra -Wpedantic -Wshadow -Werror -march=native -fopenmp-simd -fopenmp`, 4 threads.

## Where the time goes

| Stage | Total ms | Per image ms | Share |
|---|---:|---:|---:|
| preprocess | 7.04 | 0.0070 | 3.7% |
| conv1 im2col | 21.02 | 0.0210 | 10.9% |
| conv1 gemm | 13.14 | 0.0131 | 6.8% |
| conv1 bias+relu | 8.41 | 0.0084 | 4.4% |
| conv2 im2col | 29.82 | 0.0298 | 15.5% |
| conv2 gemm | 42.06 | 0.0421 | 21.8% |
| conv2 bias+relu | 4.00 | 0.0040 | 2.1% |
| conv3 im2col | 11.24 | 0.0112 | 5.8% |
| conv3 gemm | 38.53 | 0.0385 | 20.0% |
| conv3 bias+relu | 1.98 | 0.0020 | 1.0% |
| maxpool (x3) | 12.32 | 0.0123 | 6.4% |
| fc | 3.16 | 0.0032 | 1.6% |
| total | 192.71 | 0.1927 | 100.0% |
