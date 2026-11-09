# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Arch22 C-template Split-KV, small TopK, and empty-row regression tests."""

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

torch.set_num_threads(8)
LAYOUTS = [("TND", "PA_BSND"), ("BSND", "PA_BSND"), ("TND", "TND"), ("BSND", "BSND")]
SPECS = []
for bs in [8, 16, 32, 64, 128]:
    for kind in ["full", "partial_block", "tail_half", "tail_one", "mixed"]:
        topk = max(17, 1024 // bs + 1)
        SPECS.append(
            dict(
                bs=bs,
                topk=topk,
                kind=kind,
                mode=3 if kind in ["partial_block", "mixed"] else 0,
            )
        )
for topk in [127, 128, 129, 192, 255, 256]:
    for mode in [0, 3]:
        SPECS.append(dict(bs=4, topk=topk, kind="full", mode=mode))
for bs in [1, 2, 4]:
    for kind in ["tail_half", "tail_one", "mixed"]:
        SPECS.append(dict(bs=bs, topk=2048, kind=kind, mode=0))


# Q=1 and 48 inner tiles exercise more than 16 partitions, including G128 statistics.
SPECS.append(dict(bs=1, topk=48 * 512, b=1, q=1, g=128, kind="full", mode=0, rope=True))


# 32 inner tiles: fewer equally balanced partitions, empty tails, and mixed query rows.
for g, kind, q in [(128, "full", 1), (8, "tail_one", 1), (8, "mixed", 5)]:
    SPECS.append(dict(bs=128, topk=128, g=g, q=q, kind=kind, mode=0))


def make_case(spec, layouts=("TND", "PA_BSND"), dtype=torch.bfloat16):
    bs, topk = (spec["bs"], spec["topk"])
    mode = spec.get("mode", 0)
    kind = spec.get("kind", "full")
    b, q, g = (spec.get("b", 1), spec.get("q", 1), spec.get("g", 8))
    rope = spec.get("rope", True)
    ql, kl = layouts
    page = 128
    capacity = bs * topk
    actual_kv = spec.get(
        "actual_kv", capacity + (1 if kind == "partial_block" else 128)
    )
    kv_lengths = spec.get("kv_lengths", [actual_kv] * b)
    actual_kv = max(kv_lengths)
    storage = (actual_kv + page - 1) // page * page
    gen = torch.Generator().manual_seed(1341300 + bs + topk + b + q + g)
    qc = torch.randn(b, q, g, 512, generator=gen).to(dtype)
    kc = torch.randn(b, storage, 1, 512, generator=gen).to(dtype)
    qr = torch.randn(b, q, g, 64, generator=gen).to(dtype) if rope else None
    kr = torch.randn(b, storage, 1, 64, generator=gen).to(dtype) if rope else None
    idx = torch.full((b, q, 1, topk), -1, dtype=torch.int32)
    valid = spec.get(
        "valid",
        topk
        if kind in ["full", "partial_block"]
        else max(1, topk // 2)
        if kind == "tail_half"
        else 1,
    )
    for bi in range(b):
        for qi in range(q):
            n = [topk, max(1, topk // 2), 0, 1][qi % 4] if kind == "mixed" else valid
            selected = torch.randperm((actual_kv + bs - 1) // bs, generator=gen)[:n]
            if spec.get("ordered_indices", False):
                selected = torch.arange(n)
            if kind == "partial_block":
                selected = torch.arange(topk)
                selected[-1] = (actual_kv - 1) // bs
            idx[bi, qi, 0, :n] = selected.int()
    qd = (qc.reshape(-1, g, 512) if ql == "TND" else qc).npu()
    qrd = (qr.reshape(-1, g, 64) if ql == "TND" else qr).npu() if rope else None
    aq = torch.tensor(
        [q * (i + 1) for i in range(b)] if ql == "TND" else [q] * b, dtype=torch.int32
    ).npu()
    ak = torch.tensor(
        [sum(kv_lengths[: i + 1]) for i in range(b)] if kl == "TND" else kv_lengths,
        dtype=torch.int32,
    ).npu()
    if kl == "TND":
        kd = torch.cat([kc[i, : kv_lengths[i]] for i in range(b)]).npu()
        krd = (
            torch.cat([kr[i, : kv_lengths[i]] for i in range(b)]).npu()
            if rope
            else None
        )
    elif kl == "PA_BSND":
        kd = kc.reshape(-1, page, 1, 512).npu()
        krd = kr.reshape(-1, page, 1, 64).npu() if rope else None
    else:
        kd = kc.npu()
        krd = kr.npu() if rope else None
    ix = (idx.reshape(-1, 1, topk) if ql == "TND" else idx).npu()
    bt = (
        torch.arange(b * storage // page, dtype=torch.int32).reshape(b, -1).npu()
        if kl == "PA_BSND"
        else None
    )
    scale = 1 / math.sqrt(576 if rope else 512)
    lse = kl != "PA_BSND"

    def run():
        return torch_npu.npu_sparse_flash_attention(
            qd,
            kd,
            kd,
            ix,
            scale,
            block_table=bt,
            actual_seq_lengths_query=aq,
            actual_seq_lengths_kv=ak,
            query_rope=qrd,
            key_rope=krd,
            sparse_block_size=bs,
            sparse_mode=mode,
            attention_mode=2,
            layout_query=ql,
            layout_kv=kl,
            return_softmax_lse=lse,
        )

    def verify(result, sampled_rows=None):
        actual = result[0].cpu().float().reshape(b, q, g, 512)
        assert torch.isfinite(actual).all(), (spec, layouts, "nonfinite")
        mx, sm = (
            [t.cpu().float().reshape(b, q, g) for t in result[1:]]
            if lse
            else [None, None]
        )
        errs = []
        for bi in range(b):
            for qi in range(q) if sampled_rows is None else sampled_rows:
                visible = kv_lengths[bi] - q + qi + 1 if mode == 3 else kv_lengths[bi]
                # Match the native bound on the number of visited sparse-index slots.
                selected = idx[
                    bi, qi, 0, : min(topk, max(0, (visible + bs - 1) // bs))
                ].long()
                pos = (selected[:, None] * bs + torch.arange(bs)[None, :]).flatten()
                pos = pos[(pos >= 0) & (pos < visible)]
                if not pos.numel():
                    assert actual[bi, qi].abs().max() < 1e-20, (
                        spec,
                        layouts,
                        bi,
                        qi,
                        "empty",
                    )
                    if lse and bs > 4:
                        assert torch.isfinite(mx[bi, qi]).all()
                        assert torch.equal(sm[bi, qi], torch.zeros_like(sm[bi, qi]))
                    continue
                scores = qc[bi, qi].float() @ kc[bi, pos, 0].float().T
                if rope:
                    scores += qr[bi, qi].float() @ kr[bi, pos, 0].float().T
                scores *= scale
                ref = scores.softmax(-1) @ kc[bi, pos, 0].float()
                err = float((actual[bi, qi] - ref).norm() / ref.norm().clamp_min(1e-20))
                errs.append(err)
                assert err < (0.02 if dtype == torch.bfloat16 else 0.005), (
                    spec,
                    layouts,
                    bi,
                    qi,
                    err,
                )
                if lse:
                    rm = scores.amax(-1)
                    rs = (scores - rm[:, None]).exp().sum(-1)
                    torch.testing.assert_close(mx[bi, qi], rm, rtol=0.001, atol=0.03)
                    torch.testing.assert_close(
                        sm[bi, qi],
                        rs,
                        rtol=0.02 if dtype == torch.bfloat16 else 0.005,
                        atol=0.03,
                    )
        return max(errs, default=0.0)

    return (run, verify)


@pytest.mark.parametrize("case_id", range(len(SPECS)))
@pytest.mark.parametrize("layouts", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_ctemplate(case_id, layouts, dtype, graph_mode):
    spec = dict(b=2, q=4, g=[1, 8, 128][case_id % 3], rope=case_id % 2 == 0)
    spec.update(SPECS[case_id])
    run, verify = make_case(spec, layouts, dtype)
    if graph_mode:
        run()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out = run()
        graph.replay()
        graph.replay()
    else:
        out = run()
    torch.npu.synchronize()
    verify(out)


@pytest.mark.parametrize(
    "layouts", [("TND", "PA_BSND"), ("TND", "TND"), ("BSND", "BSND")]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bs", [8, 128])
@pytest.mark.parametrize("kind", ["partial_block", "tail_one", "mixed"])
def test_c_empty_rows_without_split(layouts, dtype, bs, kind):
    spec = dict(
        bs=bs,
        topk=max(17, 1024 // bs + 1),
        kind=kind,
        mode=3 if kind in ["partial_block", "mixed"] else 0,
        b=1,
        q=32,
        g=8,
        rope=True,
    )
    run, verify = make_case(spec, layouts, dtype)
    verify(run())


@pytest.mark.parametrize("layouts", [("TND", "PA_BSND"), ("BSND", "BSND")])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_c_empty_rows_long_prefill(layouts, dtype):
    spec = dict(bs=64, topk=17, kind="mixed", mode=0, b=1, q=8192, g=8, rope=True)
    run, verify = make_case(spec, layouts, dtype)
    rows = [
        0,
        1,
        2,
        3,
        31,
        32,
        33,
        34,
        35,
        511,
        512,
        513,
        514,
        515,
        8188,
        8189,
        8190,
        8191,
    ]
    verify(run(), sampled_rows=rows)


@pytest.mark.parametrize("layouts", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bs", [8, 64, 128])
@pytest.mark.parametrize("mode", [0, 3])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_c_split_short_visible_kv(layouts, dtype, bs, mode, graph_mode):
    topk = max(17, 1024 // bs + 1)
    capacity = topk * bs
    spec = dict(
        bs=bs,
        topk=topk,
        kind="full",
        mode=mode,
        b=2,
        q=4,
        g=8,
        kv_lengths=[capacity + 128, capacity // 2 + 3],
        ordered_indices=True,
    )
    run, verify = make_case(spec, layouts, dtype)
    if graph_mode:
        run()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = run()
        graph.replay()
        graph.replay()
    else:
        result = run()
    torch.npu.synchronize()
    verify(result)
