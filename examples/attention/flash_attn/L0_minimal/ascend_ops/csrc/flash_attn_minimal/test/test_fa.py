#!/usr/bin/python3
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import math
import sys
from pathlib import Path

import pytest
import torch
import torch_npu

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ascend_ops

from compare import compare

FIXED_TEST_CASE = (1, 32, 1, 8192, 8192, 128)

TEST_CASES = [
    (1, 32, 1, 8192, 8192, 128),
    (2, 8, 2, 1024, 1024, 128),
    (1, 16, 16, 512, 512, 128),
    (1, 32, 1, 256, 256, 128),
    (1, 32, 4, 2048, 2048, 128),
    (1, 4, 2, 384, 384, 128),
]

@pytest.mark.parametrize("b, n1, n2, sq, skv, d", TEST_CASES)
def test_fa_general(b, n1, n2, sq, skv, d):
    if torch.npu.device_count() == 0:
        pytest.skip("NPU not available")

    torch.manual_seed(0)

    q = torch.rand((b, sq, n1, d), dtype=torch.bfloat16)
    k = torch.rand((b, skv, n2, d), dtype=torch.bfloat16)
    v = torch.rand((b, skv, n2, d), dtype=torch.bfloat16)
    scale_value = 1 / math.sqrt(d)
    enable_gqa = n1 != n2

    q_bnsd = q.transpose(1, 2)
    k_bnsd = k.transpose(1, 2)
    v_bnsd = v.transpose(1, 2)
    cpu_out = torch.nn.functional.scaled_dot_product_attention(
        q_bnsd, k_bnsd, v_bnsd, scale=scale_value, enable_gqa=enable_gqa)
    cpu_out = cpu_out.transpose(1, 2).contiguous()

    q_npu = q.to("npu")
    k_npu = k.to("npu")
    v_npu = v.to("npu")
    npu_out = ascend_ops.ops.flash_attn_minimal(q_npu, k_npu, v_npu, scale_value)
    torch.npu.synchronize()
    npu_out_cpu = npu_out.to("cpu")

    assert npu_out.is_npu
    assert npu_out_cpu.shape == cpu_out.shape
    assert npu_out_cpu.dtype == torch.bfloat16

    compare_passed = compare(cpu_out.to(torch.float).numpy().flatten(),
                             npu_out_cpu.to(torch.float).numpy().flatten())
    if compare_passed:
        print(f"testcase b={b}, n1={n1}, n2={n2}, sq={sq}, skv={skv}, d={d} Pass")
    else:
        print(f"testcase b={b}, n1={n1}, n2={n2}, sq={sq}, skv={skv}, d={d} Failed")

def test_fa_fixed():
    b, n1, n2, sq, skv, d = FIXED_TEST_CASE
    print(f"testcase b={b}, n1={n1}, n2={n2}, sq={sq}, skv={skv}, d={d} start")
    if torch.npu.device_count() == 0:
        pytest.skip("NPU not available")

    torch.manual_seed(0)

    q = torch.rand((b, sq, n1, d), dtype=torch.bfloat16)
    k = torch.rand((b, skv, n2, d), dtype=torch.bfloat16)
    v = torch.rand((b, skv, n2, d), dtype=torch.bfloat16)
    scale_value = 1 / math.sqrt(d)
    enable_gqa = n1 != n2

    q_bnsd = q.transpose(1, 2)
    k_bnsd = k.transpose(1, 2)
    v_bnsd = v.transpose(1, 2)
    cpu_out = torch.nn.functional.scaled_dot_product_attention(
        q_bnsd, k_bnsd, v_bnsd, scale=scale_value, enable_gqa=enable_gqa)
    cpu_out = cpu_out.transpose(1, 2).contiguous()

    q_npu = q.to("npu")
    k_npu = k.to("npu")
    v_npu = v.to("npu")
    npu_out = ascend_ops.ops.flash_attn_minimal(q_npu, k_npu, v_npu, scale_value)
    torch.npu.synchronize()
    npu_out_cpu = npu_out.to("cpu")

    assert npu_out.is_npu
    assert npu_out_cpu.shape == cpu_out.shape
    assert npu_out_cpu.dtype == torch.bfloat16

    compare_passed = compare(cpu_out.to(torch.float).numpy().flatten(),
                             npu_out_cpu.to(torch.float).numpy().flatten())
    if compare_passed:
        print(f"testcase b={b}, n1={n1}, n2={n2}, sq={sq}, skv={skv}, d={d} Pass")
    else:
        print(f"testcase b={b}, n1={n1}, n2={n2}, sq={sq}, skv={skv}, d={d} Failed")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
