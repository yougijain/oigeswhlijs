"""Draw the roofline from the three benchmark CSVs and print the GEMM table as markdown.

Usage: python python/roofline.py [--gemm results/gemm.csv] [--peak results/peak.csv]
                                 [--bandwidth results/bandwidth.csv] [--out results/roofline.png]
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from typing import Optional
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KIND_ORDER = ["naive", "reordered", "tiled", "simd", "threaded"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def intensity(m: int, n: int, k: int) -> float:
    """FLOP per byte if A, B and C each touch memory exactly once (compulsory traffic)."""
    return 2.0 * m * n * k / (4.0 * (m * k + k * n + m * n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemm", type=Path, default=Path("results/gemm.csv"))
    ap.add_argument("--peak", type=Path, default=Path("results/peak.csv"))
    ap.add_argument("--bandwidth", type=Path, default=Path("results/bandwidth.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/roofline.png"))
    ap.add_argument("--isa", default="avx2", help="peak kernel the GEMM micro-kernel is written in (avx2 or neon)")
    args = ap.parse_args()

    gemm = [r for r in read_csv(args.gemm) if not r["shape"].startswith("sweep")]
    peak = read_csv(args.peak)
    bw = read_csv(args.bandwidth)

    # Ceilings. One compute line per vector ISA the peak benchmark measured (the scalar
    # loop measures the compiler, not the chip), at 1 thread and at all threads. The
    # percentage columns use the ISA the GEMM kernel is written in (--isa).
    isas = []
    for r in peak:
        if r["kernel"] != "scalar" and r["kernel"] not in isas:
            isas.append(r["kernel"])
    if not isas:
        isas = ["scalar"]
    isa = args.isa if args.isa in isas else isas[0]

    def peak_for(kernel: str, threads: str) -> Optional[float]:
        rows = [float(r["gflops"]) for r in peak if r["kernel"] == kernel and r["threads"] == threads]
        return max(rows) if rows else None

    threads_n = max(int(r["threads"]) for r in peak)
    peak_1 = peak_for(isa, "1") or 0.0
    peak_n = peak_for(isa, str(threads_n)) or peak_1
    chip_1 = max(peak_for(k, "1") or 0.0 for k in isas)
    chip_n = max(peak_for(k, str(threads_n)) or 0.0 for k in isas) or chip_1
    # Memory: triad, which has the read/write mix of a streaming kernel.
    bw_1 = max(float(r["gbps"]) for r in bw if r["kernel"] == "triad" and r["threads"] == "1")
    bw_n_rows = [r for r in bw if r["kernel"] == "triad" and r["threads"] != "1"]
    bw_n = max(float(r["gbps"]) for r in bw_n_rows) if bw_n_rows else bw_1

    # Points: one per (kind, shape).
    shapes: "OrderedDict[str, tuple[int, int, int]]" = OrderedDict()
    points: dict[str, list[tuple[float, float, str]]] = {k: [] for k in KIND_ORDER}
    for r in gemm:
        m, n, k = int(r["M"]), int(r["N"]), int(r["K"])
        shapes.setdefault(r["shape"], (m, n, k))
        if r["kind"] in points:
            points[r["kind"]].append((intensity(m, n, k), float(r["gflops"]), r["shape"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [2 ** (i / 4) for i in range(-8, 44)]
    for k in isas:
        p1 = peak_for(k, "1")
        if p1 is None:
            continue
        style = "k-" if k == isa else "k:"
        ax.plot(xs, [min(p1, bw_1 * x) for x in xs], style, lw=1.5,
                label=f"1 thread, {k} FMA peak {p1:.0f} GFLOP/s, triad {bw_1:.0f} GB/s")
        if threads_n > 1:
            pn = peak_for(k, str(threads_n))
            if pn is not None:
                ax.plot(xs, [min(pn, bw_n * x) for x in xs], "k--" if k == isa else "k-.", lw=1.5,
                        label=f"{threads_n} threads, {k} FMA peak {pn:.0f} GFLOP/s, triad {bw_n:.0f} GB/s")
    markers = {"naive": "x", "reordered": "+", "tiled": "s", "simd": "o", "threaded": "^"}
    for kind in KIND_ORDER:
        pts = points[kind]
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=markers[kind], s=60, label=kind, zorder=3)
    # Label the shapes once, along the top-most kind.
    top_kind = next((k for k in reversed(KIND_ORDER) if points[k]), None)
    if top_kind:
        for x, y, name in points[top_kind]:
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("arithmetic intensity (FLOP / byte, compulsory traffic)")
    ax.set_ylabel("GFLOP/s (median of repeated runs)")
    ax.set_title("GEMM roofline")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)

    # Markdown table: rows = shapes, columns = kinds, plus intensity and % of the matching roofline.
    kinds = [k for k in KIND_ORDER if points[k]]
    print(f"| shape | M x N x K | FLOP/B | " + " | ".join(kinds) + f" | best, % of {isa} roofline | % of chip roofline |")
    print("|---|---|---:|" + "---:|" * len(kinds) + "---:|---:|")
    for name, (m, n, k) in shapes.items():
        ai = intensity(m, n, k)
        row = [name, f"{m}x{n}x{k}", f"{ai:.1f}"]
        best = 0.0
        best_kind = ""
        for kind in kinds:
            g = next((p[1] for p in points[kind] if p[2] == name), None)
            row.append(f"{g:.2f}" if g is not None else "")
            if g is not None and g > best:
                best, best_kind = g, kind
        if best_kind == "threaded":
            ceiling, chip = min(peak_n, bw_n * ai), min(chip_n, bw_n * ai)
        else:
            ceiling, chip = min(peak_1, bw_1 * ai), min(chip_1, bw_1 * ai)
        row.append(f"{100 * best / ceiling:.0f}% ({best_kind})")
        row.append(f"{100 * best / chip:.0f}%")
        print("| " + " | ".join(row) + " |")
    print()
    print(f"ceilings ({isa}): 1 thread {peak_1:.1f} GFLOP/s, {bw_1:.1f} GB/s triad, ridge {peak_1 / bw_1:.1f} FLOP/B; "
          f"{threads_n} threads {peak_n:.1f} GFLOP/s, {bw_n:.1f} GB/s triad, ridge {peak_n / bw_n:.1f} FLOP/B")
    print(f"chip peak (best ISA measured): 1 thread {chip_1:.1f}, {threads_n} threads {chip_n:.1f} GFLOP/s")


if __name__ == "__main__":
    main()
