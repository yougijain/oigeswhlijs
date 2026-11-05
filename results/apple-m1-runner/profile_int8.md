# INT8 parity and profile on the macOS arm64 runner

| Quantity | Value |
|---|---:|
| Test images | 1000 |
| Max abs error vs PyTorch | 1.29475 |
| Mean abs error vs PyTorch | 0.0705123 |
| Argmax agreement | 993 / 1000 (99.3%) |
| Accuracy, C++ engine | 83.2% |
| Accuracy, PyTorch, same images | 83.2% |
| Engine | INT8, gemm threaded (NEON dotprod (sdot), 8x12 micro-kernel) |
| Batch size | 64 |
| Wall time | 137.543 ms (0.137543 ms/image) |

Run on Apple M1 (Virtual), Clang 15.0.0, flags `-Wall -Wextra -Wpedantic -Wshadow -Werror -march=native -fopenmp-simd -Xclang -fopenmp`, 3 threads.

| Stage | Total ms | Per image ms | Share |
|---|---:|---:|---:|
| preprocess | 4.62 | 0.0046 | 3.4% |
| conv1 quantize | 1.34 | 0.0013 | 1.0% |
| conv1 im2col (int8) | 21.81 | 0.0218 | 15.9% |
| conv1 gemm (int8) | 14.76 | 0.0148 | 10.8% |
| conv1 dequant+bias+relu | 11.58 | 0.0116 | 8.5% |
| conv2 quantize | 3.43 | 0.0034 | 2.5% |
| conv2 im2col (int8) | 8.31 | 0.0083 | 6.1% |
| conv2 gemm (int8) | 21.32 | 0.0213 | 15.6% |
| conv2 dequant+bias+relu | 4.77 | 0.0048 | 3.5% |
| conv3 quantize | 1.56 | 0.0016 | 1.1% |
| conv3 im2col (int8) | 2.48 | 0.0025 | 1.8% |
| conv3 gemm (int8) | 19.78 | 0.0198 | 14.4% |
| conv3 dequant+bias+relu | 2.69 | 0.0027 | 2.0% |
| maxpool (x3) | 15.73 | 0.0157 | 11.5% |
| fc (quantize+gemm+dequant) | 2.78 | 0.0028 | 2.0% |
| total | 136.96 | 0.1370 | 100.0% |
