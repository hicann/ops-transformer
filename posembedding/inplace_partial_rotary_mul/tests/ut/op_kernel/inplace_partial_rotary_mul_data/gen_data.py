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
Generate random test data for InplacePartialRotaryMul kernel UT.

Usage: python3 gen_data.py <B> <S> <N> <D> <dtype>
  dtype: float16 | bfloat16 | float32

Outputs:
  x.bin — input tensor shaped [B, S, N, D]
  cos.bin — cos tensor, shape depends on broadcast pattern
  sin.bin — sin tensor, shape depends on broadcast pattern
"""

import sys
import struct
import numpy as np


def generate_data(b, s, n, d, dtype_str, cos_shape=None):
    """Generate random test data for InplacePartialRotaryMul."""

    dtype_map = {
        "float16": np.float16,
        "bfloat16": np.float32,  # numpy doesn't have bfloat16 natively
        "float32": np.float32,
    }

    if dtype_str not in dtype_map:
        print(f"Error: unsupported dtype {dtype_str}", file=sys.stderr)
        sys.exit(1)

    np_dtype = dtype_map[dtype_str]
    x = np.random.randn(b, s, n, d).astype(np_dtype)

    # Default cos/sin shape same as x (NO_BROADCAST)
    if cos_shape is None:
        cos_shape = [b, s, n, d]

    cos = np.random.randn(*cos_shape).astype(np_dtype)
    sin = np.random.randn(*cos_shape).astype(np_dtype)

    # interleave mode: D must be multiple of 2
    assert d % 2 == 0, f"D={d} must be multiple of 2 for interleave mode"

    # Write binary files
    x.astype(np_dtype).tofile("x.bin")
    cos.astype(np_dtype).tofile("cos.bin")
    sin.astype(np_dtype).tofile("sin.bin")

    print(f"Generated: x{x.shape} cos{cos.shape} sin{sin.shape} dtype={dtype_str}")


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python3 gen_data.py <B> <S> <N> <D> <dtype>", file=sys.stderr)
        sys.exit(1)

    b, s, n, d = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    dtype = sys.argv[5]
    generate_data(b, s, n, d, dtype)
