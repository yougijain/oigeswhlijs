"""Export the trained model, a test subset, PyTorch's reference logits, and a
calibration set to TINF files under data/export/.

Usage: python python/export.py [--ckpt checkpoints/tinycnn.pt] [--n-test 1000] [--n-calib 500]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cifar10 import load_split  # noqa: E402
from model import NORM_MEAN, NORM_STD, TinyCNN, normalize  # noqa: E402
from tinyfmt import read_tensors, write_tensors  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/tinycnn.pt"))
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/export"))
    ap.add_argument("--n-test", type=int, default=1000, help="first N test images are exported")
    ap.add_argument("--n-calib", type=int, default=500, help="last M train images (never in the test set)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = TinyCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 1. Weights, exactly the PyTorch state dict plus the normalisation constants.
    weights = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    weights["norm.mean"] = np.array(NORM_MEAN, dtype=np.float32)
    weights["norm.std"] = np.array(NORM_STD, dtype=np.float32)
    write_tensors(args.out / "weights.bin", weights)

    # 2. Test subset: raw uint8 pixels and labels. The engine does its own normalisation.
    xte, yte = load_split(args.data, "test")
    xs, ys = xte[: args.n_test], yte[: args.n_test]
    write_tensors(args.out / "test_images.bin", {"images": xs, "labels": ys})

    # 3. PyTorch's logits for those images, float32, eval mode, one batch.
    with torch.no_grad():
        logits = model(normalize(torch.from_numpy(xs))).numpy().astype(np.float32)
    write_tensors(args.out / "ref_logits.bin", {"logits": logits})
    acc = float((logits.argmax(1) == ys).mean())

    # 4. Calibration images for INT8 activation scales, drawn from the training set.
    xtr, _ = load_split(args.data, "train")
    write_tensors(args.out / "calib_images.bin", {"images": xtr[-args.n_calib :]})

    # Self-check: everything reads back bit-identical.
    for name, expected in (("weights.bin", weights), ("test_images.bin", {"images": xs, "labels": ys}),
                           ("ref_logits.bin", {"logits": logits}),
                           ("calib_images.bin", {"images": xtr[-args.n_calib :]})):
        got = read_tensors(args.out / name)
        assert list(got) == list(expected), name
        for k in expected:
            assert got[k].dtype == expected[k].dtype and np.array_equal(got[k], expected[k]), f"{name}:{k}"

    lines = ["# file  bytes  sha256"]
    for name in ("weights.bin", "test_images.bin", "ref_logits.bin", "calib_images.bin"):
        p = args.out / name
        lines.append(f"{name}  {p.stat().st_size}  {sha256(p)}")
    (args.out / "manifest.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"PyTorch accuracy on the {args.n_test} exported test images: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
