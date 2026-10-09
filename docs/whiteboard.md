# Whiteboard guide

What to be able to write and explain from memory, in the order an interviewer
would ask for it, with the numbers from this project to back each point.
Numbers are from the 4-vCPU Xeon VM in `results/` unless marked M1.

## 1. Convolution as a matrix multiply

For NHWC input `x[b][y][x][c]` and a 3x3 kernel with pad 1:

```
col[(b, oy, ox)][(ky, kx, ci)] = x[b][oy+ky-1][ox+kx-1][ci]   (0 outside the image)
y[(b, oy, ox)][co]             = sum over (ky, kx, ci) of col[..][..] * w[(ky, kx, ci)][co] + bias[co]
```

So `Y[M][N] = COL[M][K] x W[K][N]` with `M = batch * oh * ow`, `K = 9 * cin`,
`N = cout`. TinyCNN at batch 64: conv1 65536 x 32 x 27, conv2 16384 x 64 x 288,
conv3 4096 x 128 x 576.

What im2col costs: the column matrix is `k*k = 9x` the input in memory, and
each input value is copied 9 times. For conv1 that is 65536 x 27 floats = 7 MB
per batch of 64. The payoff is that all the awkwardness (padding, the 3x3
window) is in the copy, and the arithmetic is one dense GEMM that can be
optimised once. In NHWC the copy is `memcpy` of whole channel runs, and for
interior pixels one `memcpy` of 3 x cin floats per kernel row.

Where it ends up: after optimisation the GEMM is 43% of FP32 inference time
and im2col 30%; the final version has im2col write the GEMM's packed layout
directly so the GEMM's own copy of A disappears (`results/int8.md`).

## 2. The naive loop and why loop order matters

```
for i in M: for j in N: for k in K: C[i][j] += A[i][k] * B[k][j]
```

The inner loop reads `B[k][j]` with stride N: a new cache line every k. For
1024 x 1024 (4 MB per matrix) the column walk evicts itself before the next j
reuses it: 0.5 GFLOP/s.

```
for i in M: for k in K: a = A[i][k]; for j in N: C[i][j] += a * B[k][j]
```

Same multiplies, but the inner loop streams a row of B into a row of C:
contiguous, prefetchable, vectorisable. 8x faster at 1024 (4.2 GFLOP/s with
GCC, 18 with Clang, which vectorises it). The honest footnote: it was slower on
conv1 (4.8 to 3.0), because with K = 27 the column walk already fit in L1 and
the reordered loop trades a register accumulator for a store per multiply.

## 3. Tiling, as you would write it

```
for jc in 0..N step NC:            # columns of B
  for pc in 0..K step KC:          # depth
    for ic in 0..M step MC:        # rows of A
      for i in ic..ic+MC:
        for k in pc..pc+KC:
          a = A[i][k]
          for j in jc..jc+NC: C[i][j] += a * B[k][j]
```

The KC x NC block of B is loaded once and reused MC times instead of being
re-streamed per row of A. Choose it so KC x NC x 4 bytes sits in L2
(256 KB here) and the inner loop is long enough to amortise overhead.

What tiling did here: +10% to +50% with GCC, nothing with Clang except on the
1024 square (18 to 23). A 100-point sweep varied less than run-to-run noise.
Reason: the inner loop was bound by the store port, not by misses, so there
were no misses to remove. Tiling matters when the arithmetic is fast enough
to be starved, which is the next step.

## 4. The micro-kernel (the thing to be able to draw)

Keep an MR x NR tile of C in vector registers. Per k: load NR floats of B,
broadcast MR floats of A, do MR x NR / width FMAs.

AVX2, 6 x 16:
```
c[6][2] ymm accumulators (12), b0 b1 (2), a broadcast (1) = 15 of 16 registers
for k in kc:
    b0 = load(Bp + k*16); b1 = load(Bp + k*16 + 8)
    for r in 0..6: a = broadcast(Ap[k*6 + r]); c[r][0] += a*b0; c[r][1] += a*b1
```
12 FMAs per 2 loads + 6 broadcasts; 12 independent chains cover 4-cycle
latency x 2 ports. NEON is 8 x 12 with `fmla` by element: 24 FMAs per 5 loads,
no broadcasts.

Packing: B is copied into strips of NR columns (`[k][NR]`), A into strips of
MR rows (`[k][MR]`), zero-padded, so the kernel reads two contiguous streams
and never sees an edge. Panels: mc = 72 rows of A (72 KB in L2), kc = 256, one
B strip of kc x NR floats (16 KB) stays in L1 while all 12 A tiles sweep it.

Numbers: 50 to 79 GFLOP/s single-thread (GCC), 43 to 95 (Clang), against a
measured AVX2 FMA peak of 95 to 102. 84% of the ceiling on the 1024 square.

Why SIMD is not a flat 8x: the scalar loops were store-bound, not FMA-bound.
Register blocking (one store per MR x NR x K multiplies) and the 8 lanes
together gave 10x to 18x over tiled.

## 5. Threads

Split M across threads (the conv GEMMs have M in the thousands, N at most
128). B packed once per block and shared read-only; each thread packs or
reads its own A panel. 3.2x to 4.2x on 4 threads, bit-identical to one
thread. Nothing at batch 1 (conv3 has 64 rows, one block) until single-block
layers skip the fork.

## 6. Roofline

```
attainable GFLOP/s = min(peak FLOP/s, bandwidth x arithmetic intensity)
arithmetic intensity of GEMM (compulsory traffic) = 2MNK / 4(MK + KN + MN) FLOP/byte
```

Measured ceilings here: AVX2 FMA 95 / 373 GFLOP/s (1 / 4 threads), triad 10.4
/ 37.6 GB/s, ridge at 9 to 10 FLOP/B. conv1 is at 7.3 FLOP/B: memory-bound
on 4 threads (64% of 274, its bandwidth ceiling). Everything else is
compute-bound at 62% to 70% of the AVX2 roofline. The chip's AVX-512 ceiling
is twice higher; the kernel is AVX2 because the target machine is NEON.

How the peak was measured, and the two bugs: a register-only FMA loop with
enough independent chains. 16 chains on an ISA with 16 registers spilled;
12 chains in an array GCC did not unroll lived on the stack. Both read a
"peak" below the GEMM. A ceiling is a program too.

## 7. INT8

- Weights: symmetric, per output channel, `scale[n] = max_k |w[k][n]| / 127`.
  Per channel because conv3's channel ranges differ 14x; one scale would leave
  the small channels about 4 bits.
- Activations: symmetric, per tensor, static: `max |x| / 127` over 500
  training images through the float model.
- Compute: int8 x int8 -> int32 accumulate; `y = acc * (sa * sw[n]) + bias[n]`
  in float, ReLU fused.
- Instructions: 4 k per lane. NEON `sdot` (s8 x s8, 16 MACs). AVX-VNNI
  `vpdpbusd` (u8 x s8, 32 MACs): store `a + 128`, start the accumulator at
  `-128 * sum_k B[k][n]`. Both operands packed in groups of 4 k so one lane
  reads 4 consecutive k.
- Results: weights 3.92x smaller; accuracy 83.73% vs 83.71% on 10,000 images;
  GEMMs 2.3 to 2.5x faster; end to end 1.63x (x86 VNNI) and 1.8x (M1 runner,
  `sdot`).
- Where accuracy goes: the logits move 0.07 on average and 1.29 at most;
  86 of 10,000 predictions flip, netting +2 correct. The errors are the
  activation rounding (one step of `sa`, up to 0.09 here) times the weights,
  summed over K, plus per-channel weight rounding.

## 8. What the profiler showed (the surprise)

The first INT8 engine was 1.07x faster than FP32 despite a 4x dot product.
gprof: scalar byte-by-byte packing of A was 41% of INT8 time, the VNNI
kernel 25%. callgrind: 7% of all instructions in `memset` from scratch
tensors that shrank and regrew between layers. Six fixes from those lines
(32-bit packing copies, non-zeroing resize, ReLU fused into the epilogue,
`omp simd` on elementwise loops, row-run im2col, no fork for single-block
GEMMs) took INT8 to 1.63x and FP32 to 1.4x its own earlier time. Then im2col
was taught to write the packed layout so the GEMM's packing pass disappeared.

Two profiler lessons: valgrind cannot run AVX-512 and fails silently, and
gprof attributes an outlined OpenMP body to the preceding symbol.

## Self-check

Can you, on a whiteboard, without the repo:

1. Write im2col for NHWC and give M, N, K for conv2 at batch 64.
2. Write the i-j-k and i-k-j loops and say which access pattern each has.
3. Write the three blocking loops and say what fits in which cache.
4. Draw the 6 x 16 tile, count registers, count loads per FMA.
5. Derive arithmetic intensity for a 1024 square and place it on the roofline.
6. Explain `-128 * colsum` for the unsigned-times-signed instruction.
7. Say why INT8 was 1.07x at first and what the profile said.

If any of these needs the code open, that is the one to retype by hand.
