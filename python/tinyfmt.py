"""Reference writer and reader for the TINF tensor container (docs/format.md)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = b"TINF"
VERSION = 1
MAX_NAME = 256
MAX_NDIM = 4

_DTYPE_CODES = {
    np.dtype("float32"): 0,
    np.dtype("int8"): 1,
    np.dtype("uint8"): 2,
    np.dtype("int32"): 3,
}
_CODE_DTYPES = {code: dt for dt, code in _DTYPE_CODES.items()}


def _check_name(name: str) -> bytes:
    raw = name.encode("ascii")
    if not 1 <= len(raw) <= MAX_NAME:
        raise ValueError(f"tensor name length {len(raw)} outside 1..{MAX_NAME}: {name!r}")
    if any(b < 0x21 or b > 0x7E for b in raw):
        raise ValueError(f"tensor name has a byte outside 0x21..0x7E: {name!r}")
    return raw


def write_tensors(path: str | Path, tensors: dict[str, np.ndarray]) -> None:
    """Write `tensors` (insertion order preserved) to `path`."""
    chunks = [MAGIC, struct.pack("<II", VERSION, len(tensors))]
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype not in _DTYPE_CODES:
            raise TypeError(f"{name}: unsupported dtype {arr.dtype}")
        if not 1 <= arr.ndim <= MAX_NDIM:
            raise ValueError(f"{name}: ndim {arr.ndim} outside 1..{MAX_NDIM}")
        if any(d < 1 for d in arr.shape):
            raise ValueError(f"{name}: zero-sized dimension in shape {arr.shape}")
        if arr.size >= 2**32:
            raise ValueError(f"{name}: too many elements ({arr.size})")
        raw = _check_name(name)
        chunks.append(struct.pack("<I", len(raw)))
        chunks.append(raw)
        chunks.append(struct.pack("<II", _DTYPE_CODES[arr.dtype], arr.ndim))
        chunks.append(struct.pack(f"<{arr.ndim}I", *arr.shape))
        chunks.append(arr.astype(arr.dtype, order="C", copy=False).tobytes(order="C"))
    Path(path).write_bytes(b"".join(chunks))


def read_tensors(path: str | Path) -> dict[str, np.ndarray]:
    """Parse a TINF file. Mirrors the C++ loader's checks so export can self-verify."""
    buf = Path(path).read_bytes()
    if len(buf) < 12 or buf[:4] != MAGIC:
        raise ValueError(f"{path}: not a TINF file")
    version, count = struct.unpack_from("<II", buf, 4)
    if version != VERSION:
        raise ValueError(f"{path}: version {version}, expected {VERSION}")
    pos = 12
    out: dict[str, np.ndarray] = {}
    for _ in range(count):
        (name_len,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        if not 1 <= name_len <= MAX_NAME or pos + name_len > len(buf):
            raise ValueError(f"{path}: bad name length {name_len}")
        name = buf[pos : pos + name_len].decode("ascii")
        pos += name_len
        code, ndim = struct.unpack_from("<II", buf, pos)
        pos += 8
        if code not in _CODE_DTYPES or not 1 <= ndim <= MAX_NDIM:
            raise ValueError(f"{path}: bad dtype/ndim for {name}")
        dims = struct.unpack_from(f"<{ndim}I", buf, pos)
        pos += 4 * ndim
        if any(d == 0 for d in dims):
            raise ValueError(f"{path}: zero dim in {name}")
        dt = _CODE_DTYPES[code]
        nbytes = int(np.prod(dims, dtype=np.uint64)) * dt.itemsize
        if pos + nbytes > len(buf):
            raise ValueError(f"{path}: {name} runs past end of file")
        if name in out:
            raise ValueError(f"{path}: duplicate tensor {name}")
        out[name] = np.frombuffer(buf, dtype=dt, count=int(np.prod(dims)), offset=pos).reshape(dims).copy()
        pos += nbytes
    if pos != len(buf):
        raise ValueError(f"{path}: {len(buf) - pos} trailing bytes")
    return out
