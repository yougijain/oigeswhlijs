# Data

Nothing under `data/raw/` is committed. `python/cifar10.py` downloads and
extracts CIFAR-10 (binary version) into `data/raw/cifar-10-batches-bin/`.

| File | Source | Bytes | SHA-256 |
|---|---|---|---|
| `cifar-10-binary.tar.gz` | `https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz` (official) | 170052171 | not recorded here; see the CIFAR-10 page |
| `cifar-10-binary.tar.gz` | `https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/cifar10/cifar-10-binary.tar.gz` (mirror used for the committed results) | 169570669 | `5c42663433fe855e7227a0c588428176a867e401971f6e0be5fc37ca1b885452` |

The mirror is a repack of the same six batch files (the archive also carries
macOS `._*` resource-fork entries, which the loader ignores). Whichever archive
is used, `cifar10.py` checks the extracted batches structurally: each batch is
exactly 10,000 records of 3,073 bytes, every label is in 0..9, and the test
batch holds 1,000 images per class.

`data/export/` holds the small artifacts the C++ tests need, written by
`python/export.py`. They are committed on purpose so CI can run the parity
test without PyTorch. `data/export/manifest.txt` lists their sizes and hashes.
