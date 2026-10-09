# M2: parity with PyTorch

The C++ forward pass was run on the exported CIFAR-10 test images and its
logits compared element-wise with the logits PyTorch produced from the same
weights and pixels (`data/export/ref_logits.bin`).

| Quantity | Value |
|---|---:|
| Test images | 1000 |
| Max abs error vs PyTorch | 2.47955e-05 |
| Mean abs error vs PyTorch | 1.48774e-06 |
| Argmax agreement | 1000 / 1000 (100%) |
| Accuracy, C++ engine | 83.2% |
| Accuracy, PyTorch, same images | 83.2% |
| GEMM kind | naive |
| Batch size | 64 |
| Wall time | 12012 ms (12.012 ms/image) |

Run on Intel(R) Xeon(R) Processor @ 2.10GHz, GCC 13.3.0, flags `-Wall -Wextra -Wpedantic -Wshadow -Werror -march=native`.

## Where the time goes

| Stage | Total ms | Per image ms | Share |
|---|---:|---:|---:|
| preprocess | 7.45 | 0.0074 | 0.1% |
| conv1 im2col | 45.96 | 0.0460 | 0.4% |
| conv1 gemm | 600.56 | 0.6006 | 5.0% |
| conv1 bias | 27.89 | 0.0279 | 0.2% |
| conv2 im2col | 73.74 | 0.0737 | 0.6% |
| conv2 gemm | 5567.02 | 5.5670 | 46.3% |
| conv2 bias | 13.62 | 0.0136 | 0.1% |
| conv3 im2col | 22.35 | 0.0223 | 0.2% |
| conv3 gemm | 5569.13 | 5.5691 | 46.4% |
| conv3 bias | 6.97 | 0.0070 | 0.1% |
| relu (x3) | 40.43 | 0.0404 | 0.3% |
| maxpool (x3) | 19.73 | 0.0197 | 0.2% |
| fc | 16.29 | 0.0163 | 0.1% |
| total | 12011.14 | 12.0111 | 100.0% |
