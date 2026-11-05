# FP32 parity and profile on the macOS arm64 runner

| Quantity | Value |
|---|---:|
| Test images | 1000 |
| Max abs error vs PyTorch | 7.62939e-06 |
| Mean abs error vs PyTorch | 8.36246e-07 |
| Argmax agreement | 1000 / 1000 (100%) |
| Accuracy, C++ engine | 83.2% |
| Accuracy, PyTorch, same images | 83.2% |
| Engine | FP32, gemm threaded (NEON, 8x12 micro-kernel) |
| Batch size | 64 |
| Wall time | 252.86 ms (0.25286 ms/image) |

Run on Apple M1 (Virtual), Clang 15.0.0, flags `-Wall -Wextra -Wpedantic -Wshadow -Werror -march=native -fopenmp-simd -Xclang -fopenmp`, 3 threads.

| Stage | Total ms | Per image ms | Share |
|---|---:|---:|---:|
| preprocess | 3.82 | 0.0038 | 1.5% |
| conv1 im2col | 38.61 | 0.0386 | 15.3% |
| conv1 gemm | 21.94 | 0.0219 | 8.7% |
| conv1 bias+relu | 6.96 | 0.0070 | 2.8% |
| conv2 im2col | 21.02 | 0.0210 | 8.3% |
| conv2 gemm | 72.41 | 0.0724 | 28.7% |
| conv2 bias+relu | 3.06 | 0.0031 | 1.2% |
| conv3 im2col | 8.15 | 0.0082 | 3.2% |
| conv3 gemm | 61.97 | 0.0620 | 24.6% |
| conv3 bias+relu | 1.68 | 0.0017 | 0.7% |
| maxpool (x3) | 9.24 | 0.0092 | 3.7% |
| fc | 3.44 | 0.0034 | 1.4% |
| total | 252.30 | 0.2523 | 100.0% |
