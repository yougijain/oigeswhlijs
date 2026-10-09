"""CIFAR-10 (binary version) download, verification, and loading.

Each batch file is 10,000 records of 3,073 bytes: one label byte followed by
3,072 pixel bytes laid out as [3][32][32] (channel-major, red first).
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

ARCHIVE = "cifar-10-binary.tar.gz"
BATCH_DIR = "cifar-10-batches-bin"
RECORD_BYTES = 3073
RECORDS_PER_BATCH = 10_000
TRAIN_BATCHES = [f"data_batch_{i}.bin" for i in range(1, 6)]
TEST_BATCH = "test_batch.bin"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Official source first; the mirror is a byte-identical repack of the batch files.
MIRRORS = [
    "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
    "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/cifar10/cifar-10-binary.tar.gz",
]
KNOWN_SHA256 = {
    "5c42663433fe855e7227a0c588428176a867e401971f6e0be5fc37ca1b885452": "apache-mxnet mirror",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in MIRRORS:
        try:
            print(f"downloading {url}", file=sys.stderr)
            with urllib.request.urlopen(url, timeout=60) as resp, dest.open("wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            return
        except Exception as e:  # noqa: BLE001 - try the next mirror
            last_err = e
            print(f"  failed: {e}", file=sys.stderr)
    raise RuntimeError(f"could not download CIFAR-10 from any mirror: {last_err}")


def _extract(archive: Path, root: Path) -> None:
    """Extract only the batch files, flat, with path traversal blocked."""
    out = root / BATCH_DIR
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(TRAIN_BATCHES + [TEST_BATCH, "batches.meta.txt"])
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            base = Path(member.name).name
            if base not in wanted or not member.isfile():
                continue
            with tf.extractfile(member) as src, (out / base).open("wb") as dst:
                dst.write(src.read())


def _verify_batch(path: Path) -> None:
    size = path.stat().st_size
    if size != RECORD_BYTES * RECORDS_PER_BATCH:
        raise ValueError(f"{path}: {size} bytes, expected {RECORD_BYTES * RECORDS_PER_BATCH}")
    labels = np.fromfile(path, dtype=np.uint8).reshape(RECORDS_PER_BATCH, RECORD_BYTES)[:, 0]
    if labels.max() > 9:
        raise ValueError(f"{path}: label {labels.max()} out of range")


def ensure_dataset(root: Path) -> Path:
    """Return the batch directory, downloading and extracting if needed."""
    batch_dir = root / BATCH_DIR
    files = TRAIN_BATCHES + [TEST_BATCH]
    if not all((batch_dir / f).exists() for f in files):
        archive = root / ARCHIVE
        if not archive.exists():
            _download(archive)
        digest = sha256_file(archive)
        origin = KNOWN_SHA256.get(digest)
        print(f"archive sha256 {digest} ({origin or 'unrecognised; relying on structural checks'})",
              file=sys.stderr)
        _extract(archive, root)
    for f in files:
        _verify_batch(batch_dir / f)
    test_labels = np.fromfile(batch_dir / TEST_BATCH, dtype=np.uint8).reshape(-1, RECORD_BYTES)[:, 0]
    if not np.array_equal(np.bincount(test_labels, minlength=10), np.full(10, 1000)):
        raise ValueError("test batch is not 1,000 images per class")
    return batch_dir


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (images uint8 [N,3,32,32], labels uint8 [N]) for 'train' or 'test'."""
    batch_dir = ensure_dataset(root)
    names = TRAIN_BATCHES if split == "train" else [TEST_BATCH]
    raw = np.concatenate([np.fromfile(batch_dir / n, dtype=np.uint8).reshape(-1, RECORD_BYTES) for n in names])
    labels = raw[:, 0].copy()
    images = raw[:, 1:].reshape(-1, 3, 32, 32).copy()
    return images, labels


if __name__ == "__main__":
    d = ensure_dataset(Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw"))
    print(f"ok: {d}")
