#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Generate golden reference output for InplacePartialRotaryMul.

Calculates: x = x * cos + x_rotate * sin  (interleave mode, inplace)
  where x_rotate = concat(-x[...,1::2], x[...,::2])

Usage: python3 gen_golden.py <slice_start> <slice_end>
  Reads x.bin, cos.bin, sin.bin (from gen_data.py output)
  Writes golden.bin
"""

import sys
import numpy as np


def partial_rotary_mul_golden(x, cos, sin, slice_start, slice_end):
    """
    Inplace partial rotary mul — interleave mode.

    x: [B, S, N, D] input/output
    cos: broadcast-compatible with x
    sin: broadcast-compatible with x
    slice_start, slice_end: range on last dim
    """
    # Empty slice: no rope computation, return x unchanged
    if slice_start == slice_end:
        return x

    d_slice = slice(slice_start, slice_end)
    x_slice = x[..., d_slice]

    # interleave: x1 = x[...,::2], x2 = x[...,1::2]
    x1 = x_slice[..., ::2]
    x2 = x_slice[..., 1::2]
    x_rotate = np.concatenate((-x2, x1), axis=-1)

    # Broadcast cos/sin to match x_slice shape
    cos_b = np.broadcast_to(cos[..., : slice_end - slice_start], x_slice.shape)
    sin_b = np.broadcast_to(sin[..., : slice_end - slice_start], x_slice.shape)

    # Inplace update
    x[..., d_slice] = x_slice * cos_b + x_rotate * sin_b
    return x


def load_bin(filename, shape, dtype):
    """Load binary file into numpy array."""
    data = np.fromfile(filename, dtype=dtype)
    return data.reshape(shape)


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python3 gen_golden.py <slice_start> <slice_end> [B S N D dtype]",
            file=sys.stderr,
        )
        sys.exit(1)

    slice_start = int(sys.argv[1])
    slice_end = int(sys.argv[2])

    # Try to read shape info from params or from x_shape.txt
    try:
        with open("x_shape.txt", "r") as f:
            parts = f.read().strip().split()
            b, s, n, d = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            cos_shape = list(map(int, parts[4:8])) if len(parts) >= 8 else [b, s, n, d]
            dtype_str = parts[8] if len(parts) >= 9 else "float16"
    except FileNotFoundError:
        if len(sys.argv) < 7:
            print(
                "Error: need B S N D dtype when x_shape.txt is missing", file=sys.stderr
            )
            sys.exit(1)
        b, s, n, d = (
            int(sys.argv[3]),
            int(sys.argv[4]),
            int(sys.argv[5]),
            int(sys.argv[6]),
        )
        cos_shape = [b, s, n, d]
        dtype_str = sys.argv[7] if len(sys.argv) >= 8 else "float16"

    dtype_map = {"float16": np.float16, "bfloat16": np.float32, "float32": np.float32}
    np_dtype = dtype_map.get(dtype_str, np.float16)

    x = load_bin("x.bin", [b, s, n, d], np_dtype)
    cos = load_bin("cos.bin", cos_shape, np_dtype)
    sin = load_bin("sin.bin", cos_shape, np_dtype)

    # Copy x since it's modified inplace (no-op when slice_start==slice_end)
    x_golden = partial_rotary_mul_golden(x.copy(), cos, sin, slice_start, slice_end)

    x_golden.astype(np_dtype).tofile("golden.bin")
    print(f"Generated golden: x{x.shape} slice=[{slice_start},{slice_end})")


if __name__ == "__main__":
    main()
