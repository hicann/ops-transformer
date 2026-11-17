# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""NPU integration checks for the FA4 compatibility adapter."""

import pytest
import torch
import torch_npu  # noqa: F401
from experimental.attention.flash_attn_exp.torch_extension import flash_attn_func


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,sq,sk,hq,hk,scale",
    [
        (1, 64, 64, 4, 2, None),
        (2, 32, 128, 4, 4, 0.2),
        (1, 128, 32, 8, 1, None),
    ],
)
def test_fa4_dense_forward(dtype, batch, sq, sk, hq, hk, scale):
    torch.manual_seed(123)
    q = torch.randn(batch, sq, hq, 64, device="npu", dtype=dtype)
    k = torch.randn(batch, sk, hk, 64, device="npu", dtype=dtype)
    v = torch.randn_like(k)
    out, lse = flash_attn_func(q, k, v, softmax_scale=scale, return_lse=True)
    qf = q.cpu().float().transpose(1, 2)
    kf = k.cpu().float().transpose(1, 2).repeat_interleave(hq // hk, dim=1)
    vf = v.cpu().float().transpose(1, 2).repeat_interleave(hq // hk, dim=1)
    scores = (qf @ kf.transpose(-1, -2)) * (0.125 if scale is None else scale)
    reference = (scores.softmax(-1) @ vf).transpose(1, 2)
    torch.testing.assert_close(out.cpu().float(), reference, atol=0.025, rtol=0.025)
    torch.testing.assert_close(
        lse.cpu().float(), scores.logsumexp(-1), atol=0.025, rtol=0.025
    )
    out_without_lse, absent_lse = flash_attn_func(q, k, v, softmax_scale=scale)
    assert absent_lse is None
    torch.testing.assert_close(out_without_lse, out, atol=0, rtol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"causal": True},
        {"window_size": (3, 3)},
        {"softcap": 1.0},
        {"num_splits": 2},
        {"pack_gqa": True},
        {"qv": object()},
        {"gather_kv_indices": object()},
        {"learnable_sink": object()},
        {"score_mod": lambda *args: None},
        {"mask_mod": lambda *args: None},
        {"block_sparse_tensors": object()},
        {"config": object()},
        {"scales": object()},
    ],
)
def test_fa4_rejects_unsupported(kwargs):
    q = torch.empty(1, 4, 1, 64, device="npu", dtype=torch.float16)
    with pytest.raises(NotImplementedError):
        flash_attn_func(q, q, q, **kwargs)


def test_fa4_rejects_backward():
    q = torch.empty(1, 4, 1, 64, device="npu", dtype=torch.float16, requires_grad=True)
    with pytest.raises(NotImplementedError, match="backward"):
        flash_attn_func(q, q, q)
