| shape | M x N x K | FLOP/B | naive | reordered | tiled | simd | threaded | best, % of neon roofline | % of chip roofline |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conv1 b64 | 65536x32x27 | 7.3 | 4.30 | 9.89 | 12.73 | 41.77 | 103.32 | 67% (threaded) | 67% |
| conv2 b64 | 16384x64x288 | 26.1 | 1.77 | 17.33 | 18.73 | 48.10 | 100.49 | 65% (threaded) | 65% |
| conv3 b64 | 4096x128x576 | 51.1 | 1.38 | 21.70 | 21.15 | 62.45 | 148.93 | 97% (threaded) | 97% |
| square 256 | 256x256x256 | 42.7 | 1.90 | 24.30 | 24.02 | 78.98 | 156.95 | 102% (threaded) | 102% |
| square 512 | 512x512x512 | 85.3 | 1.50 | 22.30 | 18.69 | 74.46 | 137.62 | 90% (threaded) | 90% |
| square 1024 | 1024x1024x1024 | 170.7 | 1.13 | 18.31 | 15.31 | 62.44 | 158.07 | 103% (threaded) | 103% |

ceilings (neon): 1 thread 45.8 GFLOP/s, 37.2 GB/s triad, ridge 1.2 FLOP/B; 3 threads 153.7 GFLOP/s, 49.1 GB/s triad, ridge 3.1 FLOP/B
chip peak (best ISA measured): 1 thread 45.8, 3 threads 153.7 GFLOP/s
