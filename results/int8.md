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

Final engine (im2col writing the packed layout, see the last fix below):

| | FP32 | INT8 | INT8 / FP32 |
|---|---:|---:|---:|
| Latency, batch 64, same 64 images repeated (ms / image) | 0.142 | 0.106 | **1.34x faster** |
| Latency, batch 16 (ms / image) | 0.149 | 0.120 | 1.25x faster |
| Latency, batch 1 (ms / image) | 0.359 | 0.339 | 1.06x faster |
| Latency, 10,000 different images streamed, batch 100 (ms / image) | 0.149 | 0.097 | **1.54x faster** |
| Latency, 1,000 images streamed, batch 64 (ms / image) | 0.164 | 0.105 | 1.56x faster |
| Throughput, batch 64 (images / s) | 7,046 | 9,418 | 1.34x |
| GEMM time, conv2, 1,000 images (ms) | 35.5 | 13.6 | 2.6x faster |
| GEMM time, conv3, 1,000 images (ms) | 34.2 | 15.2 | 2.2x faster |
| GEMM time, conv1, 1,000 images (ms) | 10.2 | 7.7 | 1.3x faster |
| Weights on disk (bytes) | 455,287 | 116,102 | **3.92x smaller** |
| Accuracy, 10,000 test images | 83.71% | 83.73% | **+0.02 points** |
| Accuracy, the 1,000 committed images | 83.20% | 83.20% | 0 |
| Predictions equal to PyTorch, 10,000 images | 10,000 | 9,914 (99.14%) | |
| Max / mean abs logit error vs PyTorch | 9.5e-06 / 8.5e-07 | 1.29 / 0.070 | |

The 86 images out of 10,000 whose prediction changed net out to +2 correct:
INT8 flips a few near-ties either way and the accuracy is unchanged within
noise. The logits move by 0.07 on average against a typical gap of several
units between the top two classes.

The GEMMs got 2.2x to 2.6x faster on the two layers with real K; end to end
INT8 is 1.3x faster when the same 64 images sit hot in cache and 1.5x when
images stream through, because the GEMMs are now about half of the FP32 time
(next section) and the rest is passes whose cost barely depends on the data
type. conv1 gains little: with K = 27 its cost was never the arithmetic.
Earlier in this milestone the ratio read 1.63x; the last fix sped FP32 up by
more than INT8 and narrowed it.

Measured on the Mac this project is aimed at, the ratio will differ: Apple
Silicon has no VNNI but has `sdot` with 4 NEON pipes against 4 FMA pipes, so
the int8 GEMM should again be 2x to 4x the fp32 one; the NEON path was
validated for correctness under qemu, not timed.

## Where the time goes (final, `tinyinfer parity --profile`, 1,000 images, batch 64, 4 threads)

"im2col" here includes writing the GEMM's packed layout; "gemm" is the
micro-kernel plus packing B.

FP32, 0.163 ms per image:

| Stage | Per image ms | Share |
|---|---:|---:|
| preprocess | 0.0075 | 4.6% |
| conv1 im2col + pack | 0.0119 | 7.3% |
| conv1 gemm | 0.0102 | 6.3% |
| conv1 bias+relu | 0.0082 | 5.1% |
| conv2 im2col + pack | 0.0239 | 14.7% |
| conv2 gemm | 0.0355 | 21.8% |
| conv2 bias+relu | 0.0040 | 2.5% |
| conv3 im2col + pack | 0.0092 | 5.7% |
| conv3 gemm | 0.0342 | 21.0% |
| conv3 bias+relu | 0.0020 | 1.3% |
| maxpool (x3) | 0.0130 | 8.0% |
| fc | 0.0032 | 2.0% |

INT8, 0.105 ms per image:

| Stage | Per image ms | Share |
|---|---:|---:|
| preprocess | 0.0072 | 6.8% |
| conv1 quantize | 0.0010 | 0.9% |
| conv1 im2col + pack (int8) | 0.0088 | 8.4% |
| conv1 gemm (int8) | 0.0077 | 7.3% |
| conv1 dequant+bias+relu | 0.0153 | 14.6% |
| conv2 quantize | 0.0026 | 2.5% |
| conv2 im2col + pack (int8) | 0.0074 | 7.0% |
| conv2 gemm (int8) | 0.0136 | 12.9% |
| conv2 dequant+bias+relu | 0.0057 | 5.4% |
| conv3 quantize | 0.0012 | 1.1% |
| conv3 im2col + pack (int8) | 0.0021 | 2.0% |
| conv3 gemm (int8) | 0.0152 | 14.5% |
| conv3 dequant+bias+relu | 0.0026 | 2.4% |
| maxpool (x3) | 0.0123 | 11.7% |
| fc | 0.0026 | 2.5% |

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
| im2col writes the GEMM's packed-A layout directly (`im2col_nhwc_packed`, `gemm_packed_a`; int8 likewise), so the GEMM's own packing pass is gone | gprof FP32: packing 21.5% of time after the fixes above | FP32 0.193 -> 0.164 ms per image (1,000 images), 0.175 -> 0.142 (bench, batch 64); INT8 0.119 -> 0.105 |

Net: FP32 0.272 -> 0.164 ms per image (1.66x), INT8 0.264 -> 0.105 ms per
image (2.5x), accuracy and parity results bit-for-bit unchanged (the tests
check the int8 kernels for exact equality, the pre-packed GEMMs against the
packing-on-the-fly ones, and the fused ReLU against the separate one).

The last fix had its own false start: the first INT8 version of the packed
im2col scattered one byte at a time with a divide and a modulo per byte and
made the INT8 engine slower than before (0.107 -> 0.180 ms per image; gprof
put 67% of the time in that loop). Writing each group of 4 k as one 32-bit
word, as `pack_a` already did, turned it into the gain above. Same lesson as
the first INT8 packing, learned twice.

### After the first six fixes (before the packed im2col)

gprof, FP32 (1 thread): `micro_kernel` 66.1%, packing 21.5%, `im2col` 3.9%,
`maxpool` 3.0%, bias+relu 2.5%, preprocess 1.7%.

gprof, INT8 (1 thread): `micro_kernel` 40.7%, packing 24.9%, dequant+bias+relu
9.6%, `maxpool` 7.3%, `im2col_i8` 5.7%, preprocess 4.0%, `quantize` 2.8%.

That 21.5% packing line is what the seventh fix removed. (gprof on the final
binary is not quoted: the packed im2col is an OpenMP-outlined function and
gprof files its time under whatever symbol precedes it, so the final
breakdown above comes from the stage timers.)

### The bottleneck, named (final engine)

- **FP32: the GEMM is half the time and im2col-with-packing is 28%.** The
  micro-kernel is compute-bound at 62% to 70% of the AVX2 roofline on these
  shapes; what is left around it is one pass that expands the input 9x and
  writes it in strip order. The next step would be to not materialise that
  matrix at all: have the micro-kernel read its A rows straight from the
  image through a per-row pointer table (the "indirect GEMM" used by XNNPACK
  and friends). That trades the 28% for a pointer load per row per k-step.
- **INT8: the float passes around the GEMM, 40% of the time.** Dequant +
  bias + ReLU (22%), max-pool (12%), quantise (5%) all walk the activations
  in float, and for conv1 the dequant pass costs twice the GEMM it follows
  (0.0153 vs 0.0077 ms). The fix is to keep activations in int8 between
  layers: requantise in the GEMM epilogue to the next layer's scale, pool in
  int8 (max commutes with a monotonic scale), and im2col from int8 directly.
  That quarters the bytes every one of those passes touches. It was not done
  here because it changes the numerics (one more rounding per layer) and the
  milestone was about measuring a clean scheme first.

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
- **The first packed INT8 im2col** (above): byte-wise scatter with a divide
  per byte, 67% of the INT8 time, slower than no fusion at all. Fixed with
  32-bit group copies.
