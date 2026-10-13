# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
import torch
import torch_npu
from cann_ops_transformer.ops import chunk_gated_delta_rule_compute_wy


def golden_compute_wy(q, k, v, g, beta, chunk_size=64):
    """CPU FP64 reference using forward substitution instead of doubling."""
    batch, length, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    groups = value_heads // key_heads
    k_heads = k.double().permute(0, 2, 1, 3).repeat_interleave(groups, dim=1)
    v_heads = v.double().permute(0, 2, 1, 3)
    beta_heads = beta.double().permute(0, 2, 1)
    g_heads = g.double().permute(0, 2, 1)
    w = torch.empty(batch, value_heads, length, key_dim, dtype=torch.float64)
    u = torch.empty(batch, value_heads, length, value_dim, dtype=torch.float64)
    g_out = torch.empty_like(g_heads)
    eye = torch.eye(chunk_size, dtype=torch.float64)
    for start in range(0, length, chunk_size):
        sl = slice(start, start + chunk_size)
        kc, vc = k_heads[:, :, sl], v_heads[:, :, sl]
        bc = beta_heads[:, :, sl].unsqueeze(-1)
        prefix = g_heads[:, :, sl].cumsum(-1)
        decay = torch.exp(prefix.unsqueeze(-1) - prefix.unsqueeze(-2))
        # I - A = I + strictlower((beta*K) @ K.T * decay).
        lower = torch.tril((bc * kc) @ kc.transpose(-1, -2) * decay, diagonal=-1)
        matrix = eye + lower
        rhs = torch.cat((prefix.exp().unsqueeze(-1) * bc * kc, bc * vc), dim=-1)
        solved = torch.linalg.solve_triangular(
            matrix, rhs, upper=False, unitriangular=True
        )
        w[:, :, sl], u[:, :, sl] = solved.split((key_dim, value_dim), dim=-1)
        g_out[:, :, sl] = prefix
    return (
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        w.to(q.dtype),
        u.to(v.dtype),
        g_out.to(g.dtype),
    )


@pytest.mark.ci
@pytest.mark.parametrize(
    "shape",
    [
        (1, 64, 2, 2, 64, 64),
        (1, 128, 2, 4, 64, 32),
        (2, 64, 1, 2, 32, 64),
    ],
    ids=["single_chunk", "multi_chunk_gqa", "batch_gqa_unequal_dims"],
)
def test_chunk_gated_delta_rule_compute_wy_precision(shape):
    if "310p" not in torch_npu.npu.get_device_name(0).lower():
        pytest.skip("This operator currently supports 310P only")
    torch_npu.npu.set_device(0)
    batch, length, key_heads, value_heads, key_dim, value_dim = shape
    torch.manual_seed(42)
    q = torch.randn(batch, length, key_heads, key_dim, dtype=torch.float16)
    # Normalized keys match the GDN input convention and keep the solve well conditioned.
    k = torch.nn.functional.normalize(
        torch.randn(batch, length, key_heads, key_dim), dim=-1
    ).half()
    v = torch.randn(batch, length, value_heads, value_dim, dtype=torch.float16)
    g = -torch.rand(batch, length, value_heads, dtype=torch.float32) * 0.1
    beta = torch.rand(batch, length, value_heads, dtype=torch.float16)
    inputs = (q, k, v, g, beta)
    expected = golden_compute_wy(*inputs)
    actual = chunk_gated_delta_rule_compute_wy(
        *(x.npu() for x in inputs), chunk_size=64
    )
    torch_npu.npu.synchronize()
    assert len(actual) == 5
    for name, result, reference in zip(("q", "k", "w", "u", "g"), actual, expected):
        result = result.cpu()
        assert result.shape == reference.shape, name
        assert result.dtype == reference.dtype, name
        assert torch.isfinite(result).all(), name
        if name in ("q", "k"):
            torch.testing.assert_close(result, reference, rtol=0, atol=0, msg=name)
        elif name == "g":
            torch.testing.assert_close(
                result, reference, rtol=1e-5, atol=1e-6, msg=name
            )
        else:
            # FP16 Cube intermediate rounding; compare every element, including near zero.
            torch.testing.assert_close(
                result, reference, rtol=5e-3, atol=1e-3, msg=name
            )
