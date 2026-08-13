#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import argparse
import math

import torch

from cann_ops_transformer.ops import chunk_kda_fwd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--check-determinism", action="store_true")
    parser.add_argument("--determinism-repeats", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.npu.set_device(args.device)
    batch, tokens, heads, dim = 1, 16384, 96, 128
    shape = (batch, tokens, heads, dim)
    normalized = 1.0 / math.sqrt(dim)
    q = torch.full(shape, normalized, dtype=torch.bfloat16, device="npu")
    k = torch.full_like(q, normalized)
    v = torch.full_like(q, 0.01)
    raw_g = torch.full(shape, -1.0, dtype=torch.float32, device="npu")
    beta = torch.full(shape[:-1], 0.5, dtype=torch.bfloat16, device="npu")
    a_log = torch.zeros((heads,), dtype=torch.float32, device="npu")
    dt_bias = torch.zeros((heads * dim,), dtype=torch.float32, device="npu")

    def launch():
        return chunk_kda_fwd(
            q,
            k,
            v,
            raw_g,
            beta,
            normalized,
            output_final_state=True,
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            chunk_size=64,
            layout="BSND",
        )

    baseline = launch()
    torch.npu.synchronize()
    if args.check_determinism:
        if args.determinism_repeats < 2:
            raise ValueError("--determinism-repeats must be at least 2")
        for repeat in range(1, args.determinism_repeats):
            repeated = launch()
            torch.npu.synchronize()
            for index in (0, 1, 3, 4):
                assert torch.equal(baseline[index], repeated[index]), (
                    f"output {index} differs at repeat {repeat}"
                )
            if (repeat + 1) % 10 == 0:
                print(f"determinism_progress={repeat + 1}", flush=True)
        print(
            f"binary_determinism=PASS repeats={args.determinism_repeats}",
            flush=True,
        )

    for _ in range(args.repeats):
        launch()
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
