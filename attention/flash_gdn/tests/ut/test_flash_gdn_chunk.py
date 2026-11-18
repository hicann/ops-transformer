# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Chunk forward contracts and independent FP64 recurrent NPU comparisons."""

import os
import itertools
import pytest
import torch
from cann_ops_transformer import flash_gdn_fwd, flash_gdn_metadata


@pytest.mark.parametrize("layout", ["TND", "BSND", "BNSD"])
@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=3)))
def test_meta_fusions_gva_and_states(layout, flags):
    shape = (
        (9, 1, 128)
        if layout == "TND"
        else ((2, 9, 1, 128) if layout == "BSND" else (2, 1, 9, 128))
    )
    vshape = (
        (9, 2, 128)
        if layout == "TND"
        else ((2, 9, 2, 128) if layout == "BSND" else (2, 2, 9, 128))
    )
    q = torch.empty(shape, dtype=torch.bfloat16, device="meta")
    v = torch.empty(vshape, dtype=torch.bfloat16, device="meta")
    g = torch.empty(vshape[:-1], device="meta")
    cu = torch.tensor([0, 4, 9], dtype=torch.int32) if layout == "TND" else None
    md = flash_gdn_metadata(
        torch.empty(3, dtype=torch.int32, device="meta"),
        batch_size=2,
        nheads_qk=1,
        nheads_v=2,
        seqlens=9,
        layout=layout,
    )
    result = flash_gdn_fwd(
        q,
        q,
        v,
        g,
        g,
        md,
        cu_seqlens=cu,
        A_log=torch.empty(2, device="meta"),
        layout=layout,
        use_qk_l2norm_in_kernel=flags[0],
        use_gate_in_kernel=flags[1],
        use_beta_sigmoid_in_kernel=flags[2],
        return_intermediate_states=True,
        output_final_state=True,
    )
    assert len(result) == 16 and result[0].shape == vshape
    assert result[1].shape == (2, 2, 128, 128)
    assert result[2].shape == (
        (3, 2, 128, 128) if layout == "TND" else (2, 1, 2, 128, 128)
    )
    assert all(x is None for x in result[3:])


@pytest.mark.parametrize(
    "cu,used",
    [([1, 4, 9], None), ([0, 5, 4], None), ([0, 4, 8], None), ([0, 4, 9], [3, 6])],
)
def test_invalid_sequence_descriptions(cu, used):
    with pytest.raises(ValueError):
        flash_gdn_metadata(
            torch.tensor(cu, dtype=torch.int32),
            None if used is None else torch.tensor(used, dtype=torch.int32),
            batch_size=2,
            nheads_qk=1,
            nheads_v=2,
            seqlens=9,
        )


def recurrent(q, k, v, g, beta, h0, scale):
    # Inputs [T,H,D], state [Hv,V,K]; reference is never used by the operator.
    q, k, v, g, beta = [x.double() for x in (q, k, v, g, beta)]
    state = h0.double().clone()
    values = []
    states = []
    group = v.shape[1] // q.shape[1]
    for i in range(len(q)):
        if i % 64 == 0:
            states.append(state.clone())
        ki = k[i].repeat_interleave(group, 0)
        qi = q[i].repeat_interleave(group, 0)
        state = state * g[i].exp()[:, None, None]
        delta = beta[i, :, None] * (v[i] - (state * ki[:, None, :]).sum(-1))
        state = state + delta[:, :, None] * ki[:, None, :]
        values.append((state * qi[:, None, :]).sum(-1) * scale)
    return torch.stack(values), state, states


NPU = pytest.mark.skipif(
    os.environ.get("RUN_GDN_NPU") != "1", reason="requires an available Ascend NPU"
)


@pytest.mark.npu
@NPU
@pytest.mark.parametrize(
    "layout,length,flags",
    [
        ("TND", 3, (False, False, False)),
        ("TND", 65, (True, True, True)),
        ("TND", 129, (False, True, True)),
        ("BSND", 65, (True, False, True)),
        ("BNSD", 129, (True, True, False)),
        ("BNSD", 3, (False, False, True)),
        ("BSND", 3, (True, True, True)),
        ("BSND", 65, (True, False, False)),
        ("BNSD", 65, (False, True, False)),
    ],
)
def test_chunk_npu_matches_recurrence(layout, length, flags):
    import torch_npu

    torch.manual_seed(35)
    batch, hq, hv = 2, 1, 2
    lengths = [length, max(1, length - 2)]
    # Canonical CPU inputs stay FP32 for preprocessing, then compare the exact
    # BF16-rounded q/k/v values seen by the device.
    q = (torch.randn(batch, length, hq, 128) * 0.08).bfloat16()
    k = (torch.randn_like(q.float()) * 0.08).bfloat16()
    v = torch.randn(batch, length, hv, 128).bfloat16()
    g = (
        torch.randn(batch, length, hv)
        if flags[1]
        else -torch.rand(batch, length, hv) * 0.03
    )
    beta = torch.randn(batch, length, hv) if flags[2] else torch.rand(batch, length, hv)
    al = torch.tensor([-0.7, -1.2])
    bias = torch.tensor([0.1, -0.2])
    initial = torch.randn(batch, hv, 128, 128) * 0.02
    qref, kref = q.float(), k.float()
    if flags[0]:
        qref = qref * torch.rsqrt((qref * qref).sum(-1, keepdim=True) + 1e-6)
        kref = kref * torch.rsqrt((kref * kref).sum(-1, keepdim=True) + 1e-6)
    gref = -al.exp() * torch.nn.functional.softplus(g + bias) if flags[1] else g
    bref = beta.sigmoid() * 2 if flags[2] else beta
    refs = [
        recurrent(
            qref[b, :n],
            kref[b, :n],
            v[b, :n],
            gref[b, :n],
            bref[b, :n],
            initial[b],
            0.7,
        )
        for b, n in enumerate(lengths)
    ]

    def logical(t):
        if layout == "TND":
            return torch.cat([t[b, :n] for b, n in enumerate(lengths)], 0).contiguous()
        if layout == "BNSD":
            return t.transpose(1, 2).contiguous()
        return t.contiguous()

    cu = (
        torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
        if layout == "TND"
        else None
    )
    used = None if layout == "TND" else torch.tensor(lengths, dtype=torch.int32)
    md = flash_gdn_metadata(
        cu,
        used,
        batch_size=batch,
        nheads_qk=hq,
        nheads_v=hv,
        seqlens=sum(lengths) if layout == "TND" else length,
        layout=layout,
    )
    values = [logical(t).npu() for t in (q, k, v, g, beta)]
    init = initial.npu()
    before = init.clone()
    with torch.inference_mode():
        result = flash_gdn_fwd(
            *values,
            md,
            initial_state=init,
            A_log=al.npu(),
            dt_bias=bias.npu(),
            cu_seqlens=cu,
            seqused=used,
            layout=layout,
            scale=0.7,
            output_final_state=True,
            return_intermediate_states=True,
            use_qk_l2norm_in_kernel=flags[0],
            use_gate_in_kernel=flags[1],
            use_beta_sigmoid_in_kernel=flags[2],
            allow_neg_eigval=True,
        )
    torch.npu.synchronize()
    expected = torch.zeros_like(v, dtype=torch.float64)
    for b, n in enumerate(lengths):
        expected[b, :n] = refs[b][0]
    torch.testing.assert_close(
        result[0].cpu().float(), logical(expected).float(), atol=0.01, rtol=0.01
    )
    torch.testing.assert_close(
        result[1].cpu(), torch.stack([r[1] for r in refs]).float(), atol=1e-4, rtol=1e-3
    )
    prefix = 0
    for b, n in enumerate(lengths):
        for c, state in enumerate(refs[b][2]):
            actual = result[2][prefix + c] if layout == "TND" else result[2][b, c]
            torch.testing.assert_close(
                actual.cpu(), state.float(), atol=1e-4, rtol=1e-3
            )
        prefix += len(refs[b][2])
    assert torch.equal(init, before)
