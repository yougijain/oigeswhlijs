#!/usr/bin/env bash
# Build, test and benchmark tinyinfer on this machine, and write every result
# into results/<label>/ so it can be committed as-is.
#
#   tools/run_benchmarks.sh [--label NAME] [--quick] [--no-perf]
#
#   --label NAME  directory name under results/ (default: derived from the CPU name)
#   --quick       a few runs per benchmark, to check the script works (about 2 minutes);
#                 numbers from a quick run are not for the write-up
#   --no-perf     skip the perf profiles even if perf is available
#
# Works on Linux (x86-64 or AArch64), including WSL2 on Windows. Needs cmake,
# ninja, a C++17 compiler and OpenMP (GCC has it built in). Optional: perf for
# profiles, python3 with numpy+matplotlib for the roofline chart.
#
# Before a real run: plug the laptop in, set Windows to its best-performance
# power mode, close other programs. On Intel CPUs with P- and E-cores, set
# OMP_NUM_THREADS to the number of P-cores, or threads land on E-cores and the
# medians wander.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL=""
QUICK=0
PERF=1
while [ $# -gt 0 ]; do
    case "$1" in
        --label) LABEL="$2"; shift 2 ;;
        --quick) QUICK=1; shift ;;
        --no-perf) PERF=0; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cpu_name() {
    if [ -r /proc/cpuinfo ]; then
        local n
        n="$(grep -m1 -E '^(model name|Model)' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
        if [ -z "$n" ] && command -v lscpu >/dev/null; then
            n="$(lscpu | sed -n 's/^Model name: *//p' | head -1)"
        fi
        echo "${n:-unknown}"
    else
        echo unknown
    fi
}

if [ -z "$LABEL" ]; then
    LABEL="$(cpu_name | tr '[:upper:]' '[:lower:]' | sed -E 's/\((r|tm)\)//g; s/(cpu|processor|with radeon graphics|[0-9]+-core)//g; s/@.*//; s/[^a-z0-9]+/-/g; s/^-+|-+$//g')"
    [ -n "$LABEL" ] || LABEL="this-machine"
fi
[ "$QUICK" = 1 ] && LABEL="${LABEL}-quick"

OUT="$ROOT/results/$LABEL"
BUILD="$ROOT/build-bench"
DATA="$ROOT/data/export"
mkdir -p "$OUT"
# The benchmark CSV writer appends; start every file fresh.
rm -f "$OUT"/{peak,bandwidth,gemm,infer}.csv

case "$(uname -m)" in
    aarch64|arm64) ISA=neon ;;
    *) ISA=avx2 ;;
esac

if [ "$QUICK" = 1 ]; then
    RUNS_GEMM=2; RUNS_PEAK=2; ITERS=20000000; BW_MB=128; RUNS_BW=3; RUNS_INFER=3; SECONDS_INFER=1; LIMIT=256
else
    RUNS_GEMM=7; RUNS_PEAK=5; ITERS=200000000; BW_MB=512; RUNS_BW=10; RUNS_INFER=10; SECONDS_INFER=5; LIMIT=1000
fi

step() { printf '\n== %s\n' "$*"; }

step "machine"
{
    echo "cpu: $(cpu_name)"
    echo "arch: $(uname -m)"
    echo "kernel: $(uname -r)"
    if grep -qi microsoft /proc/version 2>/dev/null; then echo "environment: WSL2 on Windows"; fi
    echo "logical cpus: $(nproc)"
    echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-unset (all logical cpus)}"
    echo "compiler: $(${CXX:-c++} --version | head -1)"
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "quick run: $([ "$QUICK" = 1 ] && echo yes || echo no)"
    if command -v lscpu >/dev/null; then
        echo
        lscpu | grep -E '^(Model name|Thread|Core|Socket|L1d|L2|L3|CPU max MHz|Flags)' | cut -c1-200
    fi
} | tee "$OUT/machine.txt"

step "configure and build (Release, -march=native)"
cmake -S "$ROOT" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DTINYINFER_NATIVE=ON >/dev/null
cmake --build "$BUILD"

step "tests"
ctest --test-dir "$BUILD" --output-on-failure | tail -3

step "quantize INT8 weights"
"$BUILD/tinyinfer" quantize --weights "$DATA/weights.bin" --calib "$DATA/calib_images.bin" \
    --out "$BUILD/weights_int8.bin" | tee "$OUT/quantize.txt"

step "kernels selected"
"$BUILD/bench_gemm" --runs 1 --shapes none | head -1 | tee "$OUT/backends.txt"

step "peak FMA throughput"
"$BUILD/bench_peak" --iters "$ITERS" --runs "$RUNS_PEAK" --csv "$OUT/peak.csv" | tee "$OUT/peak.txt"

step "memory bandwidth"
"$BUILD/bench_bandwidth" --mb "$BW_MB" --runs "$RUNS_BW" --csv "$OUT/bandwidth.csv" | tee "$OUT/bandwidth.txt"

step "GEMM ladder"
"$BUILD/bench_gemm" --runs "$RUNS_GEMM" --csv "$OUT/gemm.csv" | tee "$OUT/gemm.txt"

step "end-to-end inference, FP32 and INT8"
"$BUILD/bench_infer" --weights "$DATA/weights.bin" --images "$DATA/test_images.bin" \
    --int8 "$BUILD/weights_int8.bin" --runs "$RUNS_INFER" --seconds "$SECONDS_INFER" \
    --csv "$OUT/infer.csv" | tee "$OUT/infer.txt"

step "parity and stage profile, FP32"
"$BUILD/tinyinfer" parity --weights "$DATA/weights.bin" --images "$DATA/test_images.bin" \
    --logits "$DATA/ref_logits.bin" --limit "$LIMIT" --batch 64 --profile --md "$OUT/profile_fp32.md" \
    | head -6
step "parity and stage profile, INT8"
"$BUILD/tinyinfer" parity --int8 "$BUILD/weights_int8.bin" --images "$DATA/test_images.bin" \
    --logits "$DATA/ref_logits.bin" --limit "$LIMIT" --batch 64 --profile --md "$OUT/profile_int8.md" \
    | head -6

# perf: on WSL2 the /usr/bin/perf wrapper refuses to run because the WSL kernel has
# no matching linux-tools package, but the binary from linux-tools-generic works.
find_perf() {
    if perf --version >/dev/null 2>&1; then echo perf; return; fi
    local p
    for p in /usr/lib/linux-tools/*/perf /usr/lib/linux-tools-*/perf; do
        if [ -x "$p" ] && "$p" --version >/dev/null 2>&1; then echo "$p"; return; fi
    done
}
if [ "$PERF" = 1 ]; then
    PERF_BIN="$(find_perf || true)"
    if [ -n "${PERF_BIN:-}" ]; then
        # Hardware events need a PMU; WSL2 and most VMs do not expose one, so fall back
        # to timer-based sampling, which still gives a correct per-function profile.
        # perf stat exits 0 even when it prints "<not supported>", so check the output.
        if "$PERF_BIN" stat -x, -e cycles -- true 2>&1 | grep -qE '^[0-9]+,'; then EVENT=cycles; else EVENT=cpu-clock; fi
        for eng in fp32 int8; do
            step "perf profile, $eng (1 thread, event $EVENT)"
            if [ "$eng" = fp32 ]; then
                ARGS=(--weights "$DATA/weights.bin")
            else
                ARGS=(--int8 "$BUILD/weights_int8.bin")
            fi
            if OMP_NUM_THREADS=1 "$PERF_BIN" record -q -e "$EVENT" -o "$BUILD/perf.$eng.data" -- \
                "$BUILD/tinyinfer" parity "${ARGS[@]}" --images "$DATA/test_images.bin" \
                --logits "$DATA/ref_logits.bin" --limit "$LIMIT" --batch 64 >/dev/null 2>&1; then
                { echo "# perf report, $eng, event $EVENT, OMP_NUM_THREADS=1"
                  "$PERF_BIN" report -i "$BUILD/perf.$eng.data" --stdio --no-children --percent-limit 0.5 \
                      --sort symbol 2>/dev/null | grep -vE '^#|^$' | sed -E 's/[[:space:]-]+$//' | head -40; } > "$OUT/perf_$eng.txt"
                head -15 "$OUT/perf_$eng.txt"
                if [ "$EVENT" = cycles ]; then
                    "$PERF_BIN" stat -e cycles,instructions,cache-references,cache-misses -o "$OUT/perf_stat_$eng.txt" -- \
                        env OMP_NUM_THREADS=1 "$BUILD/tinyinfer" parity "${ARGS[@]}" --images "$DATA/test_images.bin" \
                        --logits "$DATA/ref_logits.bin" --limit "$LIMIT" --batch 64 >/dev/null 2>&1 || true
                fi
            else
                echo "perf record failed (check /proc/sys/kernel/perf_event_paranoid); skipping" | tee "$OUT/perf_$eng.txt"
            fi
        done
    else
        step "perf not found; skipping profiles (sudo apt install linux-tools-generic)"
    fi
fi

step "roofline chart"
if command -v python3 >/dev/null && python3 -c "import numpy, matplotlib" >/dev/null 2>&1; then
    python3 "$ROOT/python/roofline.py" --isa "$ISA" --gemm "$OUT/gemm.csv" --peak "$OUT/peak.csv" \
        --bandwidth "$OUT/bandwidth.csv" --out "$OUT/roofline.png" | tee "$OUT/roofline_table.md"
else
    echo "python3 with numpy and matplotlib not found; skipping the chart (the CSVs are enough to draw it later)"
fi

step "done"
echo "results written to results/$LABEL/"
ls -1 "$OUT"
