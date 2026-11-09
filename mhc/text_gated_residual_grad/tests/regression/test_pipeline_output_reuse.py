# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check that V6 cannot clear V1 output buffers before their MTE3 stores finish.

On the tested 56-AIV A5 tiling, H=1280 selects five-row blocks: V6's pending
buffer overlaps the last xn row and first dup row in a V1 stage-0 output block.
The mathematical result is exact for every supported core configuration, though
reproducing the original race depends on the platform's tiling and scheduling.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def run_case(runner, device, directory):
    n, c, h, r = 3072, 2, 1280, 160
    d = c * h
    (directory / "meta.txt").write_text(f"bf16 {n} {c} {h} {r} 1 1\n")

    def save(name, shape, value=0, bf16=True):
        values = np.full(shape, value, dtype=np.float32)
        data = (values.view(np.uint32) >> 16).astype(np.uint16) if bf16 else values
        data.tofile(directory / f"{name}.bin")

    save("grad_mixed", (n, h))
    save("grad_inject", (n, c), 1)
    save("x", (n, d), 1)
    save("up", (n, d))
    save("down_pre_act", (n, r))
    save("w_down", (r, d))
    save("w_up", (d, r))
    save("w_gate", (c, d))
    save("gamma", d, bf16=False)
    save("inv_rms", (n, c), 1, bf16=False)
    save("inject", (n, c), 1)
    subprocess.run([str(runner), str(directory), str(device)], check=True, timeout=60)
    raw = np.fromfile(directory / "dw_gate.bin", dtype=np.uint16)
    got = (raw.astype(np.uint32) << 16).view(np.float32)
    # xn=1, gpre=1 * (1 - 1/2) / C=1/4; sum over all 3072 rows.
    np.testing.assert_array_equal(got, np.full(c * d, n / 4, dtype=np.float32))
    print("PASS: every dw_gate element is exactly 768; no xn output row was lost")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--device", required=True, type=int)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tgrg-output-reuse-") as temporary:
        run_case(args.runner.resolve(), args.device, Path(temporary))
