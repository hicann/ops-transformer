#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
非量化 kvcache 非连续 stride 传递测试

模板分发 (IsCapable → isReconstructTemp):
  新模板 (FiaTilingNonQuantArch35):
    D=128 且无 pse/sink/padding/prefix/后量化, sparseMode=0/3
    支持 ROPE_COMBINE (D=192)
  老模板 (兼容): 其余场景 (D≠128 非MLA, 非PA等) 由 OpAPI 连续化回退

非连续规格: BnNBsD/NZ: dim0/dim01; BnBsH: dim0; 非PA: OpAPI 连续化回退
"""

import logging
import math
import random
import pytest
import torch
import torch_npu

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

DEVICE_ID = 0
DTYPE = torch.float16


def _block_table(batch, seqs_kv, bs):
    vn = [math.ceil(s / bs) + 1 for s in seqs_kv]
    total, mx = sum(vn), max(vn)
    ids = list(range(total))
    random.shuffle(ids)
    tbl = -torch.ones(batch, mx, dtype=torch.int32)
    off = 0
    for b in range(batch):
        tbl[b, : vn[b]] = torch.tensor(ids[off : off + vn[b]], dtype=torch.int32)
        off += vn[b]
    return tbl, total


def _kv_shape(fmt, total_blocks, bs, d, N_kv, B, kv_seq, q_layout, act_kv):
    if fmt == "BnNBsD":
        return (total_blocks, N_kv, bs, d)
    if fmt == "BnBsH":
        return (total_blocks, bs, N_kv * d)
    if fmt == "NZ":
        return (total_blocks, N_kv, d // 16, bs, 16)
    if fmt == "none":
        if q_layout == "BSND":
            return (B, kv_seq, N_kv, d)
        if q_layout == "TND":
            return (sum(act_kv), N_kv, d)
        return (B, N_kv, kv_seq, d)
    raise ValueError(fmt)


def _make_kv(shape):
    torch.manual_seed(100)
    k = torch.randn(*shape, dtype=DTYPE)
    torch.manual_seed(200)
    v = torch.randn(*shape, dtype=DTYPE)
    return k, v


def _make_nc(tensor, nc):
    if nc == "none":
        return tensor
    pad_dim = 1 if nc == "dim0" else 2
    Bv = tensor.shape[pad_dim]
    shp = list(tensor.shape)
    shp[pad_dim] = Bv + 1
    p = torch.zeros(shp, dtype=tensor.dtype, device=tensor.device)
    ws = [slice(None)] * len(tensor.shape)
    ws[pad_dim] = slice(None, Bv)
    p[tuple(ws)] = tensor
    return p[tuple(ws)]


def _cstrides(shape):
    s = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        s[i] = s[i + 1] * shape[i + 1]
    return tuple(s)


def _accum(seqs):
    out, acc = [], 0
    for s in seqs:
        acc += s
        out.append(acc)
    return out


def _run(
    q_npu,
    k_npu,
    v_npu,
    pa,
    q_layout,
    act_q,
    act_kv,
    bt_npu,
    bs,
    scale,
    q_n,
    kv_n,
    q_rope_npu=None,
    k_rope_npu=None,
    d=128,
    dr=0,
    pre_tok=65535,
    next_tok=65535,
    mask_npu=None,
    sparse_mode=0,
):
    torch_npu.npu.set_device(DEVICE_ID)
    torch.npu.synchronize()
    if q_layout == "TND":
        aq = torch.tensor(_accum(act_q), dtype=torch.int64)
        akv = torch.tensor(act_kv if pa else _accum(act_kv), dtype=torch.int64)
    else:
        aq = torch.tensor(act_q, dtype=torch.int64)
        akv = torch.tensor(act_kv, dtype=torch.int64)
    if dr > 0:
        r, _ = torch_npu.npu_fused_infer_attention_score_v2(
            q_npu,
            k_npu,
            v_npu,
            query_rope=q_rope_npu,
            key_rope=k_rope_npu,
            actual_seq_qlen=aq,
            actual_seq_kvlen=akv,
            num_query_heads=q_n,
            num_key_value_heads=kv_n,
            input_layout=q_layout,
            softmax_scale=scale,
            pre_tokens=pre_tok,
            next_tokens=next_tok,
            block_table=bt_npu if pa else None,
            block_size=bs if pa else 0,
        )
    else:
        r, _ = torch_npu.npu_fused_infer_attention_score(
            q_npu,
            k_npu,
            v_npu,
            atten_mask=mask_npu,
            actual_seq_lengths=aq,
            actual_seq_lengths_kv=akv,
            num_heads=q_n,
            num_key_value_heads=kv_n,
            input_layout=q_layout,
            scale=scale,
            pre_tokens=pre_tok,
            next_tokens=next_tok,
            block_table=bt_npu if pa else None,
            block_size=bs if pa else 0,
            sparse_mode=sparse_mode,
        )
    torch.npu.synchronize()
    return r


def _check(ref, out, tag):
    d = (out.cpu().float() - ref.cpu().float()).abs()
    mx, mn = d.max().item(), d.mean().item()
    ok = (d <= 0.05).sum().item()
    t = d.numel()
    logger.info(f"  [{tag}] max={mx:.6f} mean={mn:.6f} pass={ok}/{t} {ok / t:.4f}")
    assert torch.allclose(out.cpu().float(), ref.cpu().float(), rtol=0.00, atol=0.00), (
        f"[{tag}] FAIL max={mx}"
    )


def _run_one(cfg, expect_pass):
    B = cfg["B"]
    N_q = cfg["N_q"]
    N_kv = cfg["N_kv"]
    pa = cfg["pa"]
    fmt = cfg["fmt"]
    ql = cfg["ql"]
    nc = cfg["nc"]
    kv_seq = cfg["kv_seq"]
    bs = cfg["bs"]
    q_seq = cfg["q_seq"]
    d = cfg["d"]
    dr = cfg.get("dr", 0)
    dv = cfg.get("dv", d)
    act_q = cfg.get("act_q", [q_seq] * B)
    act_kv = cfg.get("act_kv", [kv_seq] * B)
    sparse_mode = cfg.get("sparse_mode", 0)
    pre_tok = cfg.get("pre_tok", 65535)
    next_tok = cfg.get("next_tok", 65535)
    mask_shape = cfg.get("mask_shape", None)
    scale = 1.0 / math.sqrt(d + dr)
    bt, total_b = None, 0
    if pa:
        bt, total_b = _block_table(B, act_kv, bs)
    else:
        total_b = B
    bt_npu = bt.npu() if bt is not None else None

    torch.manual_seed(42)
    q_bnsd = torch.randn(B, N_q, q_seq, d, dtype=DTYPE)
    if ql == "BSND":
        q_bnsd = q_bnsd.permute(0, 2, 1, 3).contiguous()
    elif ql == "TND":
        T = sum(act_q)
        q_tnd = torch.zeros(T, N_q, d, dtype=DTYPE)
        t = 0
        for b, s in enumerate(act_q):
            q_tnd[t : t + s] = q_bnsd[b, :, :s, :].transpose(0, 1)
            t += s
        q_bnsd = q_tnd
    q_npu = q_bnsd.npu()

    kv_shape = (
        _kv_shape(fmt, total_b, bs, d, N_kv, B, kv_seq, ql, act_kv)
        if pa
        else _kv_shape("none", total_b, bs, d, N_kv, B, kv_seq, ql, act_kv)
    )

    q_rope, k_rope = None, None
    if dr > 0:
        torch.manual_seed(300)
        qr = torch.randn(B, N_q, q_seq, dr, dtype=DTYPE)
        if ql == "BSND":
            qr = qr.permute(0, 2, 1, 3).contiguous()
        elif ql == "TND":
            T = sum(act_q)
            qr_tnd = torch.zeros(T, N_q, dr, dtype=DTYPE)
            t = 0
            for b, s in enumerate(act_q):
                qr_tnd[t : t + s] = qr[b, :, :s, :].transpose(0, 1)
                t += s
            qr = qr_tnd
        q_rope = qr.npu()
        torch.manual_seed(400)
        if pa:
            kr_shape = kv_shape[:-1] + (dr,)  # PA: keyRope 与 key 同格式, 仅 D=64
            k_rope = torch.randn(*kr_shape, dtype=DTYPE).npu()
        else:
            k_rope = torch.randn(B, N_kv, kv_seq, dr, dtype=DTYPE).npu()

    k_ref, v_ref = _make_kv(kv_shape)
    mask_npu = None
    if mask_shape:
        mask_npu = torch.ones(mask_shape, dtype=torch.bool).npu()
    ref_out = _run(
        q_npu,
        k_ref.npu(),
        v_ref.npu(),
        pa,
        ql,
        act_q,
        act_kv,
        bt_npu,
        bs,
        scale,
        N_q,
        N_kv,
        q_rope,
        k_rope,
        d,
        dr,
        pre_tok,
        next_tok,
        mask_npu,
        sparse_mode,
    )

    k_nc = _make_nc(k_ref.npu(), nc)
    v_nc = _make_nc(v_ref.npu(), nc)
    k_rope_nc = _make_nc(k_rope, nc) if dr > 0 and nc != "none" else k_rope
    if nc != "none":
        cs = _cstrides(k_nc.shape)
        assert k_nc.stride(0) != cs[0], f"{cfg['id']}:dim0"
        if nc == "dim01":
            assert k_nc.stride(1) != cs[1], f"{cfg['id']}:dim1"
        logger.info(f"  k contig={k_nc.is_contiguous()} stride={k_nc.stride()}")

    if not expect_pass:
        with pytest.raises(Exception) as ei:
            _run(
                q_npu,
                k_nc,
                v_nc,
                pa,
                ql,
                act_q,
                act_kv,
                bt_npu,
                bs,
                scale,
                N_q,
                N_kv,
                q_rope,
                k_rope_nc,
                d,
                dr,
                pre_tok,
                next_tok,
                mask_npu,
                sparse_mode,
            )
        logger.error(f"  [{cfg['id']}] ERROR: {ei.value}")
        return
    out = _run(
        q_npu,
        k_nc,
        v_nc,
        pa,
        ql,
        act_q,
        act_kv,
        bt_npu,
        bs,
        scale,
        N_q,
        N_kv,
        q_rope,
        k_rope_nc,
        d,
        dr,
        pre_tok,
        next_tok,
        mask_npu,
        sparse_mode,
    )
    _check(ref_out, out, cfg["id"])


# ==============================================================================
# 新模板 (isReconstructTemp=true): D=128 GQA / D=192 ROPE_COMBINE, sparse=0/3,
#   无 pse/sink/padding/prefix/后量化, layout BSH/BSND/BNSD/TND
# ==============================================================================
NEW_NORMAL = [
    # BnNBsD
    {
        "id": "NEW-BnNBsD-contig",
        "B": 1,
        "N_q": 4,
        "N_kv": 2,
        "pa": True,
        "fmt": "BnNBsD",
        "ql": "BNSD",
        "nc": "none",
        "kv_seq": 2048,
        "bs": 512,
        "q_seq": 4,
        "d": 128,
    },
    {
        "id": "NEW-BnNBsD-dim0",
        "B": 2,
        "N_q": 2,
        "N_kv": 1,
        "pa": True,
        "fmt": "BnNBsD",
        "ql": "TND",
        "nc": "dim0",
        "kv_seq": 1536,
        "bs": 256,
        "q_seq": 6,
        "d": 128,
        "act_q": [3, 6],
    },
    {
        "id": "NEW-BnNBsD-dim01",
        "B": 1,
        "N_q": 4,
        "N_kv": 2,
        "pa": True,
        "fmt": "BnNBsD",
        "ql": "BSND",
        "nc": "dim01",
        "kv_seq": 2048,
        "bs": 512,
        "q_seq": 4,
        "d": 128,
    },
    # BnBsH
    {
        "id": "NEW-BnBsH-contig",
        "B": 2,
        "N_q": 4,
        "N_kv": 2,
        "pa": True,
        "fmt": "BnBsH",
        "ql": "BSND",
        "nc": "none",
        "kv_seq": 2048,
        "bs": 512,
        "q_seq": 4,
        "d": 128,
    },
    {
        "id": "NEW-BnBsH-dim0",
        "B": 3,
        "N_q": 6,
        "N_kv": 2,
        "pa": True,
        "fmt": "BnBsH",
        "ql": "TND",
        "nc": "dim0",
        "kv_seq": 2304,
        "bs": 384,
        "q_seq": 4,
        "d": 128,
        "act_q": [2, 4, 3],
    },
    # NZ
    {
        "id": "NEW-NZ-contig",
        "B": 1,
        "N_q": 2,
        "N_kv": 1,
        "pa": True,
        "fmt": "NZ",
        "ql": "BNSD",
        "nc": "none",
        "kv_seq": 2560,
        "bs": 512,
        "q_seq": 4,
        "d": 128,
    },
    {
        "id": "NEW-NZ-dim0",
        "B": 2,
        "N_q": 4,
        "N_kv": 4,
        "pa": True,
        "fmt": "NZ",
        "ql": "BSND",
        "nc": "dim0",
        "kv_seq": 1536,
        "bs": 256,
        "q_seq": 3,
        "d": 128,
    },
    {
        "id": "NEW-NZ-dim01",
        "B": 1,
        "N_q": 1,
        "N_kv": 1,
        "pa": True,
        "fmt": "NZ",
        "ql": "TND",
        "nc": "dim01",
        "kv_seq": 1024,
        "bs": 128,
        "q_seq": 5,
        "d": 128,
    },
    # ROPE_COMBINE: D=192, V=128
    {
        "id": "NEW-d192-dim0",
        "B": 1,
        "N_q": 2,
        "N_kv": 1,
        "pa": True,
        "fmt": "BnNBsD",
        "ql": "TND",
        "nc": "dim0",
        "kv_seq": 1536,
        "bs": 384,
        "q_seq": 4,
        "d": 192,
        "dv": 128,
    },
]

NEW_COMPAT_FALLBACK = [
    {
        "id": "NEW-BnBsH-dim01-FALLBACK",
        "B": 2,
        "N_q": 2,
        "N_kv": 1,
        "pa": True,
        "fmt": "BnBsH",
        "ql": "BNSD",
        "nc": "dim01",
        "kv_seq": 1024,
        "bs": 256,
        "q_seq": 5,
        "d": 128,
    },
]

# ==============================================================================
# 老模板 (isReconstructTemp=false): 非PA, ROPE_SPLIT, D≠128, 等
# ==============================================================================
OLD_NORMAL = [
    # ROPE_SPLIT: D=128+rope64, V=128
    {
        "id": "OLD-ROPE-dim0",
        "B": 1,
        "N_q": 4,
        "N_kv": 1,
        "pa": True,
        "fmt": "BnNBsD",
        "ql": "TND",
        "nc": "dim0",
        "kv_seq": 2048,
        "bs": 512,
        "q_seq": 4,
        "d": 128,
        "dr": 64,
        "dv": 128,
    },
    {
        "id": "OLD-ROPE-dim01",
        "B": 2,
        "N_q": 8,
        "N_kv": 4,
        "pa": True,
        "fmt": "BnNBsD",
        "ql": "BNSD",
        "nc": "dim01",
        "kv_seq": 1024,
        "bs": 256,
        "q_seq": 6,
        "d": 128,
        "dr": 64,
        "dv": 128,
    },
    # D=256 (D≠128 且非 ROPE_COMBINE, 路由至老模板)
    {
        "id": "OLD-d256-contig",
        "B": 1,
        "N_q": 8,
        "N_kv": 4,
        "pa": True,
        "fmt": "BnNBsD",
        "ql": "BNSD",
        "nc": "none",
        "kv_seq": 1024,
        "bs": 128,
        "q_seq": 8,
        "d": 256,
    },
    {
        "id": "OLD-d256-dim0",
        "B": 2,
        "N_q": 4,
        "N_kv": 2,
        "pa": True,
        "fmt": "BnNBsD",
        "ql": "BSND",
        "nc": "dim0",
        "kv_seq": 1024,
        "bs": 128,
        "q_seq": 8,
        "d": 256,
    },
    {
        "id": "OLD-d256-dim01",
        "B": 1,
        "N_q": 4,
        "N_kv": 2,
        "pa": True,
        "fmt": "BnNBsD",
        "ql": "BNSD",
        "nc": "dim01",
        "kv_seq": 1024,
        "bs": 128,
        "q_seq": 8,
        "d": 256,
    },
    # BnBsH + D=64
    {
        "id": "OLD-d64BnBsH-contig",
        "B": 2,
        "N_q": 2,
        "N_kv": 1,
        "pa": True,
        "fmt": "BnBsH",
        "ql": "BNSD",
        "nc": "none",
        "kv_seq": 1536,
        "bs": 256,
        "q_seq": 4,
        "d": 64,
    },
    {
        "id": "OLD-d64BnBsH-dim0",
        "B": 1,
        "N_q": 4,
        "N_kv": 2,
        "pa": True,
        "fmt": "BnBsH",
        "ql": "BNSD",
        "nc": "dim0",
        "kv_seq": 1536,
        "bs": 256,
        "q_seq": 4,
        "d": 64,
    },
    # 非PA
    {
        "id": "OLD-NonPA-contig",
        "B": 3,
        "N_q": 3,
        "N_kv": 1,
        "pa": False,
        "fmt": "none",
        "ql": "TND",
        "nc": "none",
        "kv_seq": 800,
        "bs": 0,
        "q_seq": 7,
        "d": 64,
        "act_q": [3, 5, 7],
    },
    # NZ + D=64 (D≠128 路由至老模板)
    {
        "id": "OLD-NZ-d64-contig",
        "B": 1,
        "N_q": 2,
        "N_kv": 1,
        "pa": True,
        "fmt": "NZ",
        "ql": "BNSD",
        "nc": "none",
        "kv_seq": 1024,
        "bs": 128,
        "q_seq": 4,
        "d": 64,
    },
    {
        "id": "OLD-NZ-d64-dim0",
        "B": 2,
        "N_q": 2,
        "N_kv": 2,
        "pa": True,
        "fmt": "NZ",
        "ql": "BNSD",
        "nc": "dim0",
        "kv_seq": 1024,
        "bs": 128,
        "q_seq": 3,
        "d": 64,
    },
    {
        "id": "OLD-NZ-d64-dim01",
        "B": 1,
        "N_q": 4,
        "N_kv": 2,
        "pa": True,
        "fmt": "NZ",
        "ql": "BNSD",
        "nc": "dim01",
        "kv_seq": 1024,
        "bs": 128,
        "q_seq": 5,
        "d": 64,
    },
]

COMPAT_FALLBACK = [
    {
        "id": "OLD-BnBsH-dim01-FALLBACK",
        "B": 2,
        "N_q": 4,
        "N_kv": 2,
        "pa": True,
        "fmt": "BnBsH",
        "ql": "BNSD",
        "nc": "dim01",
        "kv_seq": 1024,
        "bs": 128,
        "q_seq": 4,
        "d": 256,
    },
    {
        "id": "OLD-NonPA-dim0-FALLBACK",
        "B": 4,
        "N_q": 4,
        "N_kv": 2,
        "pa": False,
        "fmt": "none",
        "ql": "BNSD",
        "nc": "dim0",
        "kv_seq": 640,
        "bs": 0,
        "q_seq": 3,
        "d": 64,
    },
    {
        "id": "OLD-NonPA-dim01-FALLBACK",
        "B": 2,
        "N_q": 6,
        "N_kv": 2,
        "pa": False,
        "fmt": "none",
        "ql": "BNSD",
        "nc": "dim01",
        "kv_seq": 512,
        "bs": 0,
        "q_seq": 6,
        "d": 128,
    },
    # 非PA + rope: keyRope 非连续由 OpAPI 连续化回退
    {
        "id": "OLD-NonPA-rope-dim0-FALLBACK",
        "B": 1,
        "N_q": 2,
        "N_kv": 1,
        "pa": False,
        "fmt": "none",
        "ql": "BNSD",
        "nc": "dim0",
        "kv_seq": 512,
        "bs": 0,
        "q_seq": 4,
        "d": 128,
        "dr": 64,
        "dv": 128,
    },
]


# ==============================================================================
# pytest
# ==============================================================================
ALL_NORMAL = NEW_NORMAL + OLD_NORMAL + NEW_COMPAT_FALLBACK + COMPAT_FALLBACK


@pytest.mark.parametrize("cfg", ALL_NORMAL, ids=[c["id"] for c in ALL_NORMAL])
def test_normal(cfg):
    logger.info(f"\n[NORMAL] {cfg['id']}")
    _run_one(cfg, expect_pass=True)


if __name__ == "__main__":
    logger.info(
        f"normal_path={len(NEW_NORMAL) + len(OLD_NORMAL)} "
        f"compat_fallback={len(NEW_COMPAT_FALLBACK) + len(COMPAT_FALLBACK)}"
    )
    ok = True
    for c in ALL_NORMAL:
        try:
            _run_one(c, True)
            logger.info("  PASS")
        except Exception as e:
            logger.error(f"  FAIL: {e}")
            ok = False
    logger.info("\nAll passed!" if ok else "\nFailures exist!")
