# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""experimental chunk_gated_delta_rule 精度测试。

调用路径: torch.ops.cann_ops_transformer.chunk_gated_delta_rule (torch_extension JIT binding)
真值参考: attention/chunk_gated_delta_rule/tests/pytest 下的纯 torch fp32 参考实现
对比指标: max_re / avg_re / rmse，并对 out(bf16) 与 final_state(fp32) 分别校验
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch_npu
import cann_ops_transformer

# 复用同目录下 test_chunk_gated_delta_rule_float 提供的 bit-close golden
# （bf16 中间量模拟 NPU，输出布局与 NPU 算子一致，无需转置）
_TEST_DIR = os.path.dirname(__file__)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)
from test_chunk_gated_delta_rule_float import chunk_gdn_benchmark, print_compare  # noqa: E402

DEVICE_ID = 0
# bf16 输出相对 fp32 真值的经验阈值（chunk 递推会累积误差，适当放宽）
MAX_RE_THRESH = 0.05
AVG_RE_THRESH = 1e-2
RMSE_THRESH = 1e-2
MIN_ERR = 1e-3


def _max_re(golden, actual):
    return torch.max(torch.abs(actual - golden) / (torch.abs(golden) + MIN_ERR)).item()


def _avg_re(golden, actual):
    return torch.mean(torch.abs(actual - golden) / (torch.abs(golden) + MIN_ERR)).item()


def _rmse(golden, actual):
    return torch.sqrt(torch.mean(torch.pow(actual - golden, 2))).item()


def _compare(golden, actual, name):
    golden = golden.to(torch.float32)
    actual = actual.to(torch.float32)
    max_re = _max_re(golden, actual)
    avg_re = _avg_re(golden, actual)
    rmse = _rmse(golden, actual)
    print(
        f"[{name}] max_re={max_re:.3e} (thr {MAX_RE_THRESH}), "
        f"avg_re={avg_re:.3e} (thr {AVG_RE_THRESH}), rmse={rmse:.3e} (thr {RMSE_THRESH})"
    )
    assert max_re < MAX_RE_THRESH, f"{name} max_re {max_re} >= {MAX_RE_THRESH}"
    assert avg_re < AVG_RE_THRESH, f"{name} avg_re {avg_re} >= {AVG_RE_THRESH}"


@pytest.mark.ci
@pytest.mark.parametrize(
    "B, seqlen, nk, nv, dk, dv, chunk_size, has_g",
    [
        (1, 128, 4, 4, 128, 128, 64, True),
        (2, 256, 4, 4, 128, 128, 64, True),
        (1, 128, 4, 4, 128, 128, 64, False),  # 不带 g 门控
    ],
)
def test_chunk_gated_delta_rule_acc(B, seqlen, nk, nv, dk, dv, chunk_size, has_g):
    torch_npu.npu.set_device(DEVICE_ID)
    np.random.seed(21)
    torch.npu.config.allow_internal_format = True
    dev = f"npu:{DEVICE_ID}"

    # ============ 构造输入（与 experimental 算子 dtype 约定一致）============
    q = torch.rand((seqlen * B, nk, dk), dtype=torch.bfloat16, device=dev)
    k = torch.rand((seqlen * B, nk, dk), dtype=torch.bfloat16, device=dev)
    v = torch.rand((seqlen * B, nv, dv), dtype=torch.bfloat16, device=dev)
    beta = torch.rand((seqlen * B, nv), dtype=torch.bfloat16, device=dev)
    q = torch.nn.functional.normalize(q, p=2, dim=-1)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    scale = 1.0 / (dk**0.5)
    # experimental: initial_state / final_state / g 均为 fp32
    initial_state = torch.rand((B, nv, dv, dk), dtype=torch.float32, device=dev)
    actual_seq_lengths = torch.full((B,), seqlen, dtype=torch.int32, device=dev)
    g = None
    if has_g:
        g = torch.rand((seqlen * B, nv), dtype=torch.float32, device=dev) * -1.0

    # ============ NPU 算子直调（torch_extension binding，JIT 首次编译）============
    o_npu, state_npu = cann_ops_transformer.chunk_gated_delta_rule(
        q, k, v, beta, initial_state, actual_seq_lengths, scale_value=scale, g=g
    )
    o_npu = o_npu.cpu().to(torch.float32)
    state_npu = state_npu.cpu().to(torch.float32)

    # ============ CPU golden（bit-close，bf16 中间量模拟 NPU）============
    # golden 约定: q/k/v/beta=bf16, initial_state=fp32, g=fp32, actual_seq_lengths=int32
    # 输出 attn_out=(T,Nv,Dv) bf16, final_state=(B,Nv,Dv,Dk) fp32，与 NPU 算子布局一致，无需转置
    g_cpu = None if g is None else g.cpu()
    o_golden, state_golden = chunk_gdn_benchmark(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        beta.cpu(),
        scale,
        initial_state.cpu(),
        actual_seq_lengths.cpu(),
        g_cpu,
    )
    o_golden = o_golden.to(torch.float32)
    state_golden = state_golden.to(torch.float32)

    # ============ 精度对比 ============
    print(f"out:         npu{tuple(o_npu.shape)} golden{tuple(o_golden.shape)}")
    print(f"final_state: npu{tuple(state_npu.shape)} golden{tuple(state_golden.shape)}")
    print_compare("out", o_npu, o_golden)
    _compare(o_golden, o_npu, "out")
    print_compare("final_state", state_npu, state_golden)
    _compare(state_golden, state_npu, "final_state")
