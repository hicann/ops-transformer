#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check all four hRes rows through the arch22 four-head ACLNN path.

Run with the isolated MhcPost OPP and mhc_post_adapter on PYTHONPATH:
    python3 test_mhc_post_four_hres_layout.py --device 3
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu
from mhc_post_adapter import mhc_post


def inputs(dtype, batch, d, four_dimensional):
    prefix = (2, 2) if four_dimensional else (batch,)
    positions = torch.arange(d, dtype=torch.int32)
    signs = torch.where(positions % 2 == 0, 1.0, -1.0)
    head_values = torch.tensor([1.0, 2.0, 4.0, 8.0])
    x = (head_values[:, None] * signs[None, :]).expand(*prefix, 4, d).clone().to(dtype)
    h_res = (
        (torch.arange(1, 17, dtype=torch.float32).reshape(4, 4) / 16)
        .expand(*prefix, 4, 4)
        .clone()
    )
    h_out = (
        (torch.where(positions % 4 < 2, 1.0, -1.0) / 2)
        .expand(*prefix, d)
        .clone()
        .to(dtype)
    )
    h_post = (torch.arange(1, 5, dtype=torch.float32) / 8).expand(*prefix, 4).clone()
    assert torch.unique(h_res.reshape(-1, 16)[0]).numel() == 16
    assert torch.all(x != 0)
    return x, h_res, h_out, h_post


def golden(x, h_res, h_out, h_post):
    # Dyadic inputs make the float64 reference exact before the final dtype cast.
    x64, res64, out64, post64 = (v.to(torch.float64) for v in (x, h_res, h_out, h_post))
    result = post64.unsqueeze(-1) * out64.unsqueeze(-2)
    for input_head in range(4):
        result = result + res64[..., input_head, :].unsqueeze(-1) * x64[
            ..., input_head, :
        ].unsqueeze(-2)
    return result.to(x.dtype)


def assert_equal(actual, expected, label):
    if not torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ):
        delta = (actual.float() - expected.float()).abs()
        raise AssertionError(
            f"{label}: mismatched={(actual != expected).sum().item()}, max_abs={delta.max().item()}"
        )


def run_case(dtype, batch, d, four_dimensional):
    x, h_res, h_out, h_post = inputs(dtype, batch, d, four_dimensional)
    x_npu, out_npu, post_npu = x.npu(), h_out.npu(), h_post.npu()
    expected = golden(x, h_res, h_out, h_post)
    actual = mhc_post(x_npu, h_res.npu(), out_npu, post_npu).cpu()
    assert_equal(actual, expected, "base")
    assert actual.dtype == dtype and actual.shape == x.shape

    for input_head in range(4):
        for output_head in range(4):
            changed_res = h_res.clone()
            changed_res[..., input_head, output_head] += 0.5
            changed_expected = golden(x, changed_res, h_out, h_post)
            changed = mhc_post(x_npu, changed_res.npu(), out_npu, post_npu).cpu()
            label = f"input_head={input_head} output_head={output_head}"
            assert_equal(changed, changed_expected, label)
            assert torch.all(
                changed[..., output_head, :] != actual[..., output_head, :]
            ), label
            for other in range(4):
                if other != output_head:
                    assert_equal(changed[..., other, :], actual[..., other, :], label)
    return {
        "dtype": str(dtype),
        "shape": list(x.shape),
        "d": d,
        "calls": 17,
        "pass": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", nargs="?", default="manual")
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch_npu.npu.set_device(args.device)
    cases = []
    for dtype in (torch.float16, torch.bfloat16):
        for d in (2048, 2176, 5120):
            for batch, four_dimensional in ((1, False), (4, True)):
                case = run_case(dtype, batch, d, four_dimensional)
                cases.append(case)
                print(json.dumps(case), flush=True)
    torch.npu.synchronize()
    loaded = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libcust_op" in line or "liboptiling" in line
        }
    )
    result = {
        "label": args.label,
        "opp": os.environ.get("MHC_POST_OPP"),
        "loaded_libs": loaded,
        "device": args.device,
        "device_name": torch_npu.npu.get_device_name(args.device),
        "cases": cases,
        "calls": sum(case["calls"] for case in cases),
        "pass": True,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
