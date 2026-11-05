# NEON numbers from a GitHub macOS arm64 runner

Produced by `.github/workflows/bench-macos.yml` (run 37890199056, commit
e43dbc0, before the fused im2col+packing change) on a GitHub-hosted
`macos-14` runner: "Apple M1 (Virtual)", 3 cores, Apple Clang 15.0.0,
`-O2 -march=native -fopenmp-simd -Xclang -fopenmp` with libomp, 3 threads.

This is a shared, virtualised machine. It shows the NEON kernels working and
their rough shape; it is not the machine the project's numbers are meant to
come from. Run the same commands on a real Mac (README, "Reproducing on
Apple Silicon") for the write-up.

| | Value |
|---|---|
| Kernels selected | FP32: NEON 8x12 `fmla` micro-kernel. INT8: NEON dotprod (`sdot`) 8x12 |
| Parity | bit-identical to the x86 run for INT8 (993/1000 argmax agreement, max logit error 1.29475); FP32 max abs error 7.63e-06, 1000/1000 |
| FP32 inference, batch 64, 1,000 images (`profile_fp32.md`) | 0.253 ms per image, 3 threads |
| INT8 inference, batch 64, 1,000 images (`profile_int8.md`) | 0.138 ms per image, **1.8x faster than FP32** |
| GEMM, single thread, 1024 square | naive 1.13, reordered 18.3, tiled 15.3, simd 62.4 GFLOP/s |
| GEMM, 3 threads | conv3 149, 1024 square 158 GFLOP/s |
| Triad bandwidth | 37 GB/s on 1 thread, 49 GB/s on 3 |

Files: `peak.csv`, `bandwidth.csv`, `gemm.csv`, `infer.csv`, the two profile
tables, `roofline_table.md` and `roofline.png` (drawn locally from these CSVs
with `roofline.py --isa neon`).

## Read the roofline with care

`bench_peak` read 45.8 GFLOP/s for one NEON thread, yet the GEMM micro-kernel
itself reached 62 to 79 GFLOP/s single-threaded, so the "ceiling" is below
the kernel and the percentages in `roofline_table.md` (97% to 103%) mean
nothing. A real M1 performance core does 4 FMA pipes x 4 lanes x 2 FLOP x
3.2 GHz, about 100 GFLOP/s, and the kernel's 62 to 79 is the plausible
number; the peak loop is what underperformed here. Whether that is the
virtualised runner or an unrolling problem in Apple Clang 15 with 24
accumulators needs a real machine to tell apart. The bench_infer table in
`infer.csv` is also noisier than the x86 one (FP32 threaded reads 0.19 ms at
batch 16 and 0.31 at batch 64), which is why the 1,000-image profile runs
are quoted above.
