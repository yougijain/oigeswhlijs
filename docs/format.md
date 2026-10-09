# TINF tensor container

Every binary file the engine reads (`weights.bin`, `test_images.bin`,
`ref_logits.bin`, `calib_images.bin`, `weights_int8.bin`) uses one format: a
flat list of named, typed, shaped tensors. All integers are little-endian.

```
offset  size        field
0       4           magic, ASCII "TINF"
4       u32         version, currently 1
8       u32         count, number of tensor records that follow
12      ...         count records, back to back:

        u32         name_len, 1..256
        u8[name_len] name, ASCII 0x21..0x7E only (no spaces, no NUL)
        u32         dtype: 0 = f32, 1 = i8, 2 = u8, 3 = i32
        u32         ndim, 1..4
        u32[ndim]   dims, each >= 1
        bytes       dtype_size * dims[0] * ... * dims[ndim-1], row-major
```

Layouts follow PyTorch: conv weights are `[out_ch, in_ch, kh, kw]`, linear
weights are `[out_features, in_features]`, images are `[n, 3, 32, 32]`
(NCHW). Any re-packing the engine wants for speed (for example NHWC
activations) happens inside the engine at load time, so the file always
mirrors the PyTorch state dict and the parity test has nothing to argue about.

## Loader rules

The C++ loader treats every file as untrusted input and rejects it with an
error (never undefined behaviour) when:

- the file is larger than 1 GiB or smaller than the 12-byte header,
- the magic or version does not match,
- a name is empty, longer than 256 bytes, or contains a byte outside 0x21..0x7E,
- `ndim` is 0 or greater than 4, or any dim is 0,
- the element count would overflow 32 bits, or the data would run past the end
  of the file,
- two tensors share a name.

## Writer

`python/tinyfmt.py` is the reference writer and a reader used by the export
self-check.
