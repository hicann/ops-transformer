# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU golden tests for indexer_prologue_qw. No NPU required."""

from __future__ import annotations

import pytest
import torch

from golden import (  # noqa: E402
    E2M1_POS,
    E8M0_BIAS,
    apply_rope_inplace,
    indexer_prologue_qw_golden,
    dequant_mx_last,
    e2m1_nibble_to_fp32,
    fp32_to_e2m1_nibble,
    make_inputs,
    mx_matmul_fp32,
    pack_e2m1_nibbles,
    paired_scale_groups,
    rotate_half,
    unpack_e2m1_nibbles,
)


CASES = [
    pytest.param(16, 64, 64, 2, 64, 32, 1.0, False, id="aligned"),
    pytest.param(8, 32, 64, 3, 64, 64, 0.125, False, id="head-tail"),
    pytest.param(4, 32, 96, 1, 96, 32, 1.0, False, id="mx-tail"),
    pytest.param(17, 32, 64, 1, 64, 32, 2.0, False, id="t-tail"),
    pytest.param(8, 32, 64, 2, 64, 32, 1.0, True, id="rope-identity"),
]


def test_e2m1_codebook_roundtrip():
    pos = torch.tensor(list(E2M1_POS), dtype=torch.float32)
    codes = fp32_to_e2m1_nibble(pos)
    assert torch.equal(codes, torch.arange(8, dtype=torch.uint8))
    packed = pack_e2m1_nibbles(torch.cat([codes, codes]).unsqueeze(0))
    assert packed.shape[-1] == 8
    assert torch.equal(unpack_e2m1_nibbles(packed)[0, :8], codes)
    assert torch.allclose(e2m1_nibble_to_fp32(codes), pos)


def test_rotate_half_mapping():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = rotate_half(x)
    assert torch.equal(y, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))
    assert torch.equal(rotate_half(rotate_half(x)), -x)
    assert torch.equal(rotate_half(rotate_half(rotate_half(rotate_half(x)))), x)


def test_rope_identity_is_noop():
    q = torch.randn(4, 2, 8)
    ref = q.clone()
    sin = torch.zeros(4, 4)
    cos = torch.ones(4, 4)
    apply_rope_inplace(q, sin, cos)
    assert torch.allclose(q, ref)


@pytest.mark.parametrize("t,dim,q_lora,n,d,dr,scale,rope_identity", CASES)
def test_output_contracts(t, dim, q_lora, n, d, dr, scale, rope_identity):
    inputs = make_inputs(
        t, dim, q_lora, n, d, dr, softmax_scale=scale, rope_identity=rope_identity
    )
    out = indexer_prologue_qw_golden(inputs)
    assert out.q.shape == (t, n, d // 2)
    assert out.q.dtype == torch.uint8
    assert out.descale_q.shape == (t, n, paired_scale_groups(d), 2)
    assert out.descale_q.dtype == torch.uint8
    assert out.w.shape == (t, n)
    assert out.w.dtype == torch.float32
    assert out.q_fp32.shape == (t, n, d)


@pytest.mark.parametrize("t,dim,q_lora,n,d,dr,scale,rope_identity", CASES)
def test_w_path_matches_naive(t, dim, q_lora, n, d, dr, scale, rope_identity):
    inputs = make_inputs(
        t,
        dim,
        q_lora,
        n,
        d,
        dr,
        softmax_scale=scale,
        rope_identity=rope_identity,
        seed=7,
    )
    out = indexer_prologue_qw_golden(inputs)
    naive = scale * (inputs.x.float() @ inputs.ww.float().T)
    assert torch.allclose(out.w, naive, atol=0.0, rtol=0.0)


def test_q_gemm_matches_mx_formula():
    inputs = make_inputs(8, 32, 64, 2, 64, 32, rope_identity=True, seed=3)
    y = mx_matmul_fp32(inputs.qr, inputs.wqb, inputs.descale_qr, inputs.descale_wqb)
    out = indexer_prologue_qw_golden(inputs)
    assert torch.allclose(out.q_fp32.reshape(8, 2 * 64), y, atol=0.0, rtol=0.0)


def test_quant_roundtrip_bounded_by_fp4_step():
    inputs = make_inputs(4, 32, 64, 1, 64, 32, rope_identity=True, seed=11)
    out = indexer_prologue_qw_golden(inputs)
    recovered = dequant_mx_last(out.q, out.descale_q, d=64)
    nibbles = unpack_e2m1_nibbles(out.q)
    assert torch.equal(pack_e2m1_nibbles(nibbles), out.q)
    assert torch.isfinite(recovered).all()
    err = (recovered - out.q_fp32).abs()
    grouped = out.q_fp32.reshape(4, 1, 2, 32)
    grouped_err = err.reshape(4, 1, 2, 32)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    assert float((grouped_err / amax).median()) < 0.25
    assert out.descale_q.min() >= 0
    assert out.descale_q.max() <= 255


def test_identity_scale_bytes():
    """Zero activations encode e8m0=127 (2^0) after padding unused pair slots."""
    inputs = make_inputs(2, 32, 64, 1, 96, 32, rope_identity=True, seed=0)
    out = indexer_prologue_qw_golden(inputs)
    assert out.descale_q.shape == (2, 1, 2, 2)
    pad_slot = out.descale_q[:, :, 1, 1]
    assert torch.equal(pad_slot, torch.full_like(pad_slot, E8M0_BIAS))
