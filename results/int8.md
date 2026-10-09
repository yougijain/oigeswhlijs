# M4: INT8 inference and profiling

The same network with every GEMM in int8 x int8 -> int32, compared with the
FP32 engine on speed, size and accuracy, then both paths profiled and the
profile acted on. Every number below is from this machine (Intel Xeon @ 2.10
GHz, 4 vCPUs, GCC 13.3 `-O2 -march=native -fopenmp`), medians of repeated
runs; raw data in `results/infer.csv`, `results/int8_parity.md`,
`results/parity_fp32_fast.md` and the `gprof_*` / `callgrind_*` files.

## Scheme

- Weights: symmetric int8, one scale per output channel,
  `scale[n] = max_k |w[k][n]| / 127`. Per channel because the channels differ
  a lot: conv3's per-channel scales span 0.00025 to 0.0035, a 14x range, so a
  single scale would leave the small channels with about 4 useful bits.
- Activations: symmetric int8, one static scale per layer input, calibrated
  as `max |x| / 127` over 500 training images run through the FP32 model.
- Compute: int8 products accumulated in int32, then dequantised in float
  with the fused bias and ReLU: `y = acc * (sa * sw[n]) + b[n]`, max 0.
  Pooling and the hand-off between layers stay in float.
- Kernels: packed panels like the float GEMM, with both operands laid out in
  groups of 4 k so one dot-product instruction consumes 4 multiplies per
  lane. NEON `sdot` (signed x signed, 16 MACs per instruction, 8x12 tile),
  AVX-VNNI `vpdpbusd` (unsigned x signed, 32 MACs, 6x16 tile; activations
  are stored as a+128 and the kernel starts each accumulator at
  -128 * colsum(B), which cancels the offset exactly), an AVX2 `vpmaddwd`
  fallback and a scalar one. All four give bit-identical int32 results and
  the test checks that against the naive loop on awkward sizes with -128 and
  127 everywhere.

Calibration output (`tinyinfer quantize`):

| Layer | Input scale | max abs input | Weight scales (min .. max) |
|---|---:|---:|---|
| conv1 | 0.016744 | 2.126 | 0.002173 .. 0.006894 (32 ch) |
| conv2 | 0.072642 | 9.226 | 0.001333 .. 0.004167 (64 ch) |
| conv3 | 0.092803 | 11.786 | 0.000246 .. 0.003498 (128 ch) |
| fc | 0.081365 | 10.333 | per output |

## FP32 vs INT8

| | FP32 | INT8 | INT8 / FP32 |
|---|---:|---:|---:|
| Latency, batch 64 (ms / image) | 0.1745 | 0.1072 | **1.63x faster** |
| Latency, batch 16 (ms / image) | 0.191 | 0.110 | 1.74x faster |
| Latency, batch 1 (ms / image) | 0.373 | 0.315 | 1.18x faster |
| Throughput, batch 64 (images / s) | 5,732 | 9,330 | 1.63x |
| GEMM time, conv2, 1,000 images (ms) | 42.1 | 16.5 | 2.5x faster |
| GEMM time, conv3, 1,000 images (ms) | 38.5 | 16.4 | 2.3x faster |
| GEMM time, conv1, 1,000 images (ms) | 13.1 | 9.5 | 1.4x faster |
| Weights on disk (bytes) | 455,287 | 116,102 | **3.92x smaller** |
| Accuracy, 10,000 test images | 83.71% | 83.73% | **+0.02 points** |
| Accuracy, the 1,000 committed images | 83.20% | 83.20% | 0 |
| Predictions equal to PyTorch, 10,000 images | 10,000 | 9,914 (99.14%) | |
| Max / mean abs logit error vs PyTorch | 9.5e-06 / 8.5e-07 | 1.29 / 0.070 | |

The 86 images out of 10,000 whose prediction changed net out to +2 correct:
INT8 flips a few near-ties either way and the accuracy is unchanged within
noise. The logits move by 0.07 on average against a typical gap of several
units between the top two classes.

The GEMMs got 2.3x to 2.5x faster on the two layers with real K, and the
end-to-end result is 1.63x because the GEMMs are now only 43% of the FP32
time (next section). conv1 gains little: with K = 27 its cost was never the
arithmetic.

Measured on the Mac this project is aimed at, the ratio will differ: Apple
Silicon has no VNNI but has `sdot` with 4 NEON pipes against 4 FMA pipes, so
the int8 GEMM should again be 2x to 4x the fp32 one; the NEON path was
validated for correctness under qemu, not timed.

## Where the time goes (final, `tinyinfer parity --profile`, 1,000 images, batch 64, 4 threads)

FP32, 0.193 ms per image:

| Stage | Per image ms | Share |
|---|---:|---:|
| preprocess | 0.0070 | 3.7% |
| conv1 im2col | 0.0210 | 10.9% |
| conv1 gemm | 0.0131 | 6.8% |
| conv1 bias+relu | 0.0084 | 4.4% |
| conv2 im2col | 0.0298 | 15.5% |
| conv2 gemm | 0.0421 | 21.8% |
| conv2 bias+relu | 0.0040 | 2.1% |
| conv3 im2col | 0.0112 | 5.8% |
| conv3 gemm | 0.0385 | 20.0% |
| conv3 bias+relu | 0.0020 | 1.0% |
| maxpool (x3) | 0.0123 | 6.4% |
| fc | 0.0032 | 1.6% |

INT8, 0.119 ms per image:

| Stage | Per image ms | Share |
|---|---:|---:|
| preprocess | 0.0068 | 5.7% |
| conv1 quantize | 0.0009 | 0.7% |
| conv1 im2col (int8) | 0.0169 | 14.3% |
| conv1 gemm (int8) | 0.0095 | 8.0% |
| conv1 dequant+bias+relu | 0.0137 | 11.6% |
| conv2 quantize | 0.0027 | 2.3% |
| conv2 im2col (int8) | 0.0096 | 8.1% |
| conv2 gemm (int8) | 0.0165 | 13.9% |
| conv2 dequant+bias+relu | 0.0053 | 4.5% |
| conv3 quantize | 0.0012 | 1.0% |
| conv3 im2col (int8) | 0.0030 | 2.5% |
| conv3 gemm (int8) | 0.0164 | 13.9% |
| conv3 dequant+bias+relu | 0.0024 | 2.1% |
| maxpool (x3) | 0.0114 | 9.6% |
| fc | 0.0021 | 1.8% |

## Profiling: what the tools showed, and what was done about it

Three sources were used, because each one lies in its own way:

- The engine's own stage timers (above): wall time per stage, with threads.
- `gprof` on a `-pg` build of the real `-march=native` binary, single
  thread, 10,000 images: time per function. Caveat: it attributes the
  outlined OpenMP body of `gemm_s8_threaded` (which has the A packing inlined
  into it) to the nearest preceding symbol, `gemm_int8_backend()`, so that
  line in the int8 profile is the packing.
- `callgrind` for instruction counts per function and source line. Valgrind
  cannot execute AVX-512, so the native binary died after 2.6 million
  instructions without saying so; the callgrind numbers come from a separate
  `-march=haswell` build (AVX2 kernels, the `vpmaddwd` int8 fallback). They
  are the same source and the same stages, not the same kernels.

This is where `perf` or Instruments would be used on a machine that has
them; on this VM neither is available.

### Before

The first INT8 version ran at 0.264 ms per image against FP32's 0.272: a
1.07x gain from a 4x-wider dot product. The profiles said why.

gprof, FP32 (self time, 1 thread):

| Function | Share |
|---|---:|
| `kernel::micro_kernel` | 54.6% |
| `gemm_threaded` body (= packing A and B) | 18.0% |
| `relu_` | 7.5% |
| `Conv2d::forward` (bias loop) | 6.3% |
| `im2col_nhwc` | 5.8% |
| `maxpool2x2_nhwc` | 3.2% |

gprof, INT8:

| Function | Share |
|---|---:|
| `gemm_s8_threaded` body (= packing A, mislabelled `gemm_int8_backend`) | 40.6% |
| `kernel_i8::micro_kernel` | 25.1% |
| `relu_` | 12.2% |
| `QConv2d::forward` (dequant + bias loop) | 9.1% |
| `quantize_activations` | 4.7% |
| `im2col_nhwc_i8` | 3.6% |

callgrind, FP32, added one thing gprof could not see: 7.2% of all
instructions were in `memset`, called from `std::vector::resize`. The scratch
tensors shrink and regrow between layers of different sizes on every forward
pass, and every regrowth zero-filled the new bytes.

So the real bottlenecks were not the dot products. In INT8 the scalar,
byte-at-a-time packing of A, with a bounds test per byte, cost 1.6x the
kernel it fed. In both paths, the elementwise passes around the GEMM (ReLU,
bias or dequant, zero-fill, im2col) added up to more than the GEMM in INT8.

### Fixes, each from a line in the profile

| Fix | Evidence it addressed | Effect |
|---|---|---|
| Pack A 4 bytes at a time: the 4 k of a group are consecutive in a row of A, so a group is one 32-bit copy (xor 0x80808080 for the VNNI offset) | gprof INT8: packing 40.6% | packing 40.6% -> 24.9% of a run that is itself 2x shorter |
| `Tensor::resize` no longer zero-fills (default-init allocator) | callgrind: memset 7.2% | memset 7.2% -> 1.1% of instructions |
| ReLU fused into the bias / dequant pass | gprof: `relu_` 7.5% (FP32), 12.2% (INT8) | the pass is gone |
| `#pragma omp simd` on the epilogue, quantise, pool loops (`-fopenmp-simd`, which GCC needs at -O2) | gprof: `QConv2d::forward` 9.1% | dequant pass 134.5M -> 17.3M instructions (callgrind) |
| im2col copies the k taps of one kernel row in one `memcpy` for interior pixels | stage timer: conv1 im2col 0.041 ms, 15% | conv1 im2col 0.041 -> 0.021 ms per image |
| Single-block GEMMs (M <= 72: batch-1 conv3, the fc layer) skip the OpenMP region | bench_infer: threaded slower than simd at batch 1 | batch-1 FP32 0.434 -> 0.373 ms, INT8 0.478 -> 0.315 ms |

Net: FP32 0.272 -> 0.193 ms per image (1.41x), INT8 0.264 -> 0.119 ms per
image (2.2x), accuracy and parity results bit-for-bit unchanged (the tests
check the int8 kernels for exact equality and the fused ReLU against the
separate one).

### After

gprof, FP32 (1 thread): `micro_kernel` 66.1%, packing 21.5%, `im2col` 3.9%,
`maxpool` 3.0%, bias+relu 2.5%, preprocess 1.7%.

gprof, INT8 (1 thread): `micro_kernel` 40.7%, packing 24.9%, dequant+bias+relu
9.6%, `maxpool` 7.3%, `im2col_i8` 5.7%, preprocess 4.0%, `quantize` 2.8%.

### The bottleneck, named

- **FP32: packing A is 21% of the time and is the next target.** The kernel
  is compute-bound at two thirds of the profile, which is where it should
  be. The packing exists because im2col writes the column matrix row-major
  and the kernel wants it in MR-row strips; im2col could write the strips
  directly and the pass would disappear. That is the one change left in
  this engine with a plausible 20% in it.
- **INT8: the float passes around the GEMM.** Packing is 25% (same fix as
  above), and dequant, max-pool, im2col and quantise are another 25%: every
  one of them walks the activations in float, and for conv1 the dequant pass
  costs more than the GEMM it follows (0.0137 vs 0.0095 ms). The fix is to
  keep activations in int8 between layers: requantise in the GEMM epilogue
  to the next layer's scale, pool in int8 (max commutes with a monotonic
  scale), and im2col from int8 directly. That quarters the bytes every one
  of those passes touches. It was not done here because it changes the
  numerics (one more rounding per layer) and this milestone was about
  measuring a clean scheme first.

## What did not work

- **The first INT8 engine: 1.07x.** A 4x dot-product instruction bought 7%,
  because the GEMM was already fast enough that packing and the elementwise
  passes dominated (Amdahl), and because the int8 packing was written as the
  float one was, one element at a time, which is four times more elements
  per byte of useful work.
- **callgrind on the native binary.** It exited cleanly after 2.6 million
  instructions with an empty-looking profile. Valgrind does not implement
  AVX-512 and the program had died on the first such instruction. The
  profile had to come from a Haswell build.
- **Reading gprof literally.** `gemm_int8_backend()`, a one-line function
  that returns a string, showed at 40% of runtime. It was the OpenMP
  outlined function next to it in the object file.
- **Threads at batch 1.** Until single-block layers bypassed the parallel
  region, the threaded INT8 engine was slower at batch 1 than its own
  single-threaded kernel (0.478 vs 0.342 ms).
- **`omp simd` on max-pool.** It was marked like the other loops and did not
  move (0.0123 before, 0.0114 after): four strided loads per output over 32
  to 128 channels is bound by the loads, not the max.
