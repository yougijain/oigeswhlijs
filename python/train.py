"""Train TinyCNN on CIFAR-10 and write results/model.md.

Usage: python python/train.py [--epochs 20] [--batch 128] [--lr 0.05] [--seed 0]
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cifar10 import load_split  # noqa: E402
from model import TinyCNN, describe, forward_flops, normalize, param_count  # noqa: E402


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random 32x32 crop from a 4-pixel zero-padded image, then random horizontal flip.

    x: uint8 [B,3,32,32]. Vectorised with advanced indexing so there is no per-image loop.
    """
    b = x.shape[0]
    padded = F.pad(x, (4, 4, 4, 4))
    dy = torch.randint(0, 9, (b,), generator=gen)
    dx = torch.randint(0, 9, (b,), generator=gen)
    ar = torch.arange(32)
    rows = (dy[:, None] + ar)[:, None, :, None]  # [B,1,32,1]
    cols = (dx[:, None] + ar)[:, None, None, :]  # [B,1,1,32]
    bi = torch.arange(b)[:, None, None, None]
    ci = torch.arange(3)[None, :, None, None]
    out = padded[bi, ci, rows, cols]
    flip = torch.rand(b, generator=gen) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int = 1000) -> float:
    model.eval()
    correct = 0
    for i in range(0, x.shape[0], batch):
        logits = model(normalize(x[i : i + batch]))
        correct += (logits.argmax(1) == y[i : i + batch]).sum().item()
    return correct / x.shape[0]


def write_model_md(path: Path, model: TinyCNN, args: argparse.Namespace, test_acc: float,
                   best_acc: float, best_epoch: int, wall_s: float, n_train: int, n_test: int) -> None:
    rows = []
    h = 32
    for name, conv in (("conv1", model.conv1), ("conv2", model.conv2), ("conv3", model.conv3)):
        k = conv.in_channels * 9
        macs = conv.out_channels * k * h * h
        rows.append((f"{name} 3x3, pad 1", f"{conv.out_channels}x{h}x{h}", conv.weight.numel() + conv.bias.numel(), macs))
        rows.append(("relu + maxpool 2x2", f"{conv.out_channels}x{h // 2}x{h // 2}", 0, 0))
        h //= 2
    rows.append(("flatten", str(model.fc.in_features), 0, 0))
    rows.append(("fc", str(model.fc.out_features), model.fc.weight.numel() + model.fc.bias.numel(),
                 model.fc.in_features * model.fc.out_features))
    n_params = param_count(model)
    lines = [
        "# M1: model",
        "",
        "Three 3x3 convolutions (ReLU, 2x2 max-pool after each) and one linear layer,",
        "trained on CIFAR-10 in PyTorch. No batch norm, no dropout: the C++ engine",
        "implements exactly the five ops below and nothing else.",
        "",
        "| Layer | Output (CxHxW) | Params | MACs / image |",
        "|---|---|---:|---:|",
    ]
    for name, shape, p, macs in rows:
        lines.append(f"| {name} | {shape} | {p:,} | {macs:,} |")
    lines += [
        "",
        f"- Parameters: **{n_params:,}** ({n_params * 4:,} bytes as float32)",
        f"- Forward cost: {forward_flops(model):,} FLOPs per image (multiply-add = 2 FLOPs)",
        "",
        "## Parameter tensors (PyTorch layout, as exported)",
        "",
        "| Name | Shape | Count |",
        "|---|---|---:|",
    ]
    for name, shape, n in describe(model):
        lines.append(f"| `{name}` | {list(shape)} | {n:,} |")
    lines += [
        "",
        "## Training",
        "",
        f"- Data: {n_train:,} train / {n_test:,} test images, random-crop (pad 4) + horizontal flip",
        f"- Normalisation: per-channel mean/std from `python/model.py`",
        f"- Optimiser: SGD, momentum 0.9 (Nesterov), weight decay {args.wd}, batch {args.batch}",
        f"- Schedule: OneCycle, peak LR {args.lr}, {args.epochs} epochs, seed {args.seed}",
        f"- Wall time: {wall_s / 60:.1f} min on {platform.processor() or platform.machine()} "
        f"({torch.get_num_threads()} threads), torch {torch.__version__}",
        "",
        "## Result",
        "",
        f"- **FP32 test accuracy: {test_acc * 100:.2f}%** on all {n_test:,} test images (final epoch)",
        f"- Best epoch: {best_epoch} at {best_acc * 100:.2f}% (the final-epoch weights are the ones exported)",
        f"- Per-epoch log: `results/train_log.csv`",
        "",
        "This is a deliberately small model. The point of the project is the engine",
        "that runs it, so accuracy is recorded, not tuned.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N train images (smoke test)")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/tinycnn.pt"))
    ap.add_argument("--results", type=Path, default=Path("results/model.md"))
    ap.add_argument("--log", type=Path, default=Path("results/train_log.csv"))
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    xtr, ytr = load_split(args.data, "train")
    xte, yte = load_split(args.data, "test")
    if args.limit:
        xtr, ytr = xtr[: args.limit], ytr[: args.limit]
    xtr_t = torch.from_numpy(xtr)
    ytr_t = torch.from_numpy(ytr).long()
    xte_t = torch.from_numpy(xte)
    yte_t = torch.from_numpy(yte).long()

    model = TinyCNN()
    print(f"params: {param_count(model):,}")
    steps_per_epoch = (xtr_t.shape[0] + args.batch - 1) // args.batch
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_acc, best_epoch = 0.0, 0
    t0 = time.time()
    with args.log.open("w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "train_acc", "test_acc", "lr", "seconds"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            te = time.time()
            perm = torch.randperm(xtr_t.shape[0], generator=gen)
            loss_sum, correct = 0.0, 0
            for i in range(0, perm.shape[0], args.batch):
                idx = perm[i : i + args.batch]
                xb = normalize(augment(xtr_t[idx], gen))
                yb = ytr_t[idx]
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                loss_sum += loss.item() * yb.shape[0]
                correct += (logits.argmax(1) == yb).sum().item()
            train_loss = loss_sum / perm.shape[0]
            train_acc = correct / perm.shape[0]
            test_acc = evaluate(model, xte_t, yte_t)
            if test_acc > best_acc:
                best_acc, best_epoch = test_acc, epoch
            dt = time.time() - te
            log.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{test_acc:.4f}",
                          f"{sched.get_last_lr()[0]:.5f}", f"{dt:.1f}"])
            f.flush()
            print(f"epoch {epoch:2d}/{args.epochs} loss {train_loss:.4f} train {train_acc * 100:5.2f}% "
                  f"test {test_acc * 100:5.2f}% ({dt:.0f}s)", flush=True)
    wall = time.time() - t0

    final_acc = evaluate(model, xte_t, yte_t)
    torch.save({"state_dict": model.state_dict(), "test_acc": final_acc, "epochs": args.epochs,
                "seed": args.seed, "torch": str(torch.__version__)}, args.out)
    write_model_md(args.results, model, args, final_acc, best_acc, best_epoch, wall,
                   xtr_t.shape[0], xte_t.shape[0])
    print(f"final test accuracy {final_acc * 100:.2f}%  ->  {args.out}, {args.results}")


if __name__ == "__main__":
    main()
