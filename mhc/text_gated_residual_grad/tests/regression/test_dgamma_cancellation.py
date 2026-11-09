#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run the ACLNN example against an exactly representable Kahan cancellation case.

Requires numpy, an Ascend device, and the example linked to the package under test.
All inputs are powers of two or zero; the expected dgamma is exactly one.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def run_case(runner, device, directory):
    n, c, h, r = 2048, 4, 16, 16
    d = c * h
    (directory / "meta.txt").write_text(f"bf16 {n} {c} {h} {r} 1 0\n")

    def save(name, values, bf16=True):
        values = np.asarray(values, dtype=np.float32)
        data = (values.view(np.uint32) >> 16).astype(np.uint16) if bf16 else values
        data.tofile(directory / f"{name}.bin")

    grad = np.zeros((n, h), dtype=np.float32)
    # direct = grad * sigmoid(0) / C = grad / 8.
    # Separate tasks (1024 rows each) force distinct pending-buffer flushes.
    # Lane 0 accumulates 2**24 then 1; another lane contributes -2**24.
    grad[0] = 2**27
    grad[1024] = 8
    grad[512] = -(2**27)
    save("grad_mixed", grad)
    save("x", np.ones((n, d)))
    save("up", np.zeros((n, d)))
    save("down_pre_act", np.zeros((n, r)))
    save("w_down", np.zeros((r, d)))
    save("w_up", np.zeros((d, r)))
    save("gamma", np.zeros(d), False)
    save("inv_rms", np.ones((n, c)), False)
    subprocess.run([str(runner), str(directory), str(device)], check=True, timeout=60)
    got = np.fromfile(directory / "dgamma.bin", dtype=np.float32)
    np.testing.assert_array_equal(got, np.ones(d, dtype=np.float32))
    print("PASS: dgamma cancellation, all 64 elements equal 1 exactly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--device", required=True, type=int)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tgrg-dgamma-") as temporary:
        run_case(args.runner.resolve(), args.device, Path(temporary))
