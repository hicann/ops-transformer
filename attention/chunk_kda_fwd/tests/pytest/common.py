# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from cann_ops_transformer.ops import chunk_kda_fwd


PRESENT_MINIMAL = (0, 1, 3, 4)


@dataclass
class Inputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor


def set_device(device: int = 0) -> None:
    torch.npu.set_device(device)


def make_inputs(tokens: int, heads: int = 2, seed: int = 20260808) -> Inputs:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shape = (1, tokens, heads, 128)
    q = (torch.randn(shape, generator=generator) * 0.05).to(torch.bfloat16).npu()
    k = (torch.randn(shape, generator=generator) * 0.05).to(torch.bfloat16).npu()
    v = (torch.randn(shape, generator=generator) * 0.05).to(torch.bfloat16).npu()
    g = (torch.randn(shape, generator=generator) * 0.02 - 0.5).float().npu()
    beta = (
        torch.sigmoid(torch.randn(shape[:-1], generator=generator))
        .to(torch.bfloat16)
        .npu()
    )
    a_log = torch.linspace(-0.5, 0.5, heads, dtype=torch.float32).npu()
    dt_bias = torch.linspace(-0.1, 0.1, heads * 128, dtype=torch.float32).npu()
    return Inputs(q, k, v, g, beta, a_log, dt_bias)


def as_layout(tensor: torch.Tensor, layout: str, has_dim: bool = True) -> torch.Tensor:
    if layout == "BSND":
        return tensor.contiguous()
    if layout == "BNSD":
        if has_dim:
            return tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.permute(0, 2, 1).contiguous()
    if layout == "TND":
        return tensor.squeeze(0).contiguous()
    if layout == "NTD":
        if has_dim:
            return tensor.squeeze(0).permute(1, 0, 2).contiguous()
        return tensor.squeeze(0).permute(1, 0).contiguous()
    raise ValueError(f"unsupported layout: {layout}")


def run(inputs: Inputs, layout: str = "BSND", **kwargs):
    return chunk_kda_fwd(
        as_layout(inputs.q, layout),
        as_layout(inputs.k, layout),
        as_layout(inputs.v, layout),
        as_layout(inputs.g, layout),
        as_layout(inputs.beta, layout, has_dim=False),
        1.0 / math.sqrt(128),
        output_final_state=True,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=inputs.a_log,
        dt_bias=inputs.dt_bias,
        chunk_size=64,
        layout=layout,
        **kwargs,
    )


def synchronize() -> None:
    torch.npu.synchronize()


def assert_binary_equal(first, second) -> None:
    assert len(first) == len(second) == 12
    for index, (actual, repeated) in enumerate(zip(first, second)):
        if actual is None or repeated is None:
            assert actual is None and repeated is None, (
                f"output {index} optionality changed"
            )
            continue
        assert torch.equal(actual, repeated), (
            f"output {index} is not binary deterministic"
        )


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=2e-3
        )
