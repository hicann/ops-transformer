#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import torch
import numpy as np
import math
import random
import ml_dtypes
from einops import rearrange
from common import (
    CalculationContext,
    gtype,
    PER_BLOCK_SIZE,
    PER_VCORE_BLOCK_SIZE,
    EPSILON,
    log,
    print_log,
)

PHILOX_M4_32 = [0xD2511F53, 0xCD9E8D57]
PHILOX_W_32 = [0x9E3779B9, 0xBB67AE85]

VAL_1 = 0
VAL_2 = 1
VAL_3 = 2
VAL_4 = 3

MASK_32 = 0xFFFFFFFF


def attentionScoreWithGrad(ctx: CalculationContext):
    B = ctx.input_case["B"]
    Nq = ctx.input_case["N1"]
    Nkv = ctx.input_case["N2"]
    S = ctx.input_case["S1"]
    Skv = ctx.input_case["S2"]
    D = ctx.input_case["D"]
    D_V = ctx.input_case["D_V"]
    scalar = 0
    if D != 0:
        scalar = 1 / math.sqrt(D)  # D=64时，为0.125
    if Nkv != 0:
        G = Nq // Nkv
    else:
        G = 0
    AlibiH = 1024
    dtype = ctx.input_case["dtype"]
    input_layout = (
        ctx.input_case["input_layout"] if "input_layout" in ctx.input_case else "SBH"
    )
    pse_shape = ctx.input_case["pse_shape"] if "pse_shape" in ctx.input_case else "NONE"
    atten_mask_shape = (
        ctx.input_case["atten_mask_shape"]
        if "atten_mask_shape" in ctx.input_case
        else "SS"
    )
    pre_tockens = (
        ctx.input_case["pre_tockens_input"]
        if "pre_tockens_input" in ctx.input_case
        else 65536
    )
    next_tockens = (
        ctx.input_case["next_tockens_input"]
        if "next_tockens_input" in ctx.input_case
        else 65536
    )
    keep_prob = (
        ctx.input_case["drop_out_possibility"]
        if "drop_out_possibility" in ctx.input_case
        else 0.8
    )
    pse_type = ctx.input_case["pse_type"]

    (
        q,
        k,
        v,
        dx,
        pse,
        pse_golden,
        prefix,
        atten_masks_golden,
        atten_masks,
        scale_q,
        scale_k,
        scale_v,
        scale_dx,
    ) = DataGen(ctx.input_case)

    # drop_masks = makeDropMask(case, seed=2)
    drop_masks = get_drop_mask(ctx.input_case, 1 - keep_prob)
    drop_maskn = (
        drop_masks if drop_masks.numel() == 1 else drop_masks.reshape(B, Nkv, G, S, Skv)
    )

    # 无效行适配
    is_skip_invalid_row = False
    if ctx.input_case["sparse_mode"] == 3:
        is_skip_invalid_row = S > Skv
    if ctx.input_case["sparse_mode"] == 4:
        is_skip_invalid_row = pre_tockens < 0 or next_tockens + Skv < S
    if ctx.input_case["sparse_mode"] == 0 and atten_masks is not None:
        is_skip_invalid_row = pre_tockens < S or next_tockens < Skv
    if ctx.input_case["sparse_mode"] in (5, 6):
        is_skip_invalid_row = True if 0 in prefix else False

    out, softmax_res, x_max, x_sum = tforward(
        ctx.input_case,
        q.cpu().to(gtype),
        k.cpu().to(gtype),
        v.cpu().to(gtype),
        drop_maskn.cpu().to(gtype),
        atten_masks_golden.to(gtype),
        pse_golden.cpu().to(gtype),
        scalar,
        keep_prob,
        is_skip_invalid_row,
        pse_type,
        scale_q,
        scale_k,
        scale_v,
    )

    dq_golden, dk_golden, dv_golden = tbackward(
        ctx.input_case,
        dx.cpu().to(gtype),
        q.cpu().to(gtype),
        k.cpu().to(gtype),
        v.cpu().to(gtype),
        softmax_res.cpu().to(gtype),
        out.cpu().to(gtype),
        drop_maskn.cpu().to(gtype),
        pse_golden.cpu().to(gtype),
        scalar,
        keep_prob,
        scale_q,
        scale_k,
        scale_v,
        scale_dx,
    )

    # kernel input should be scale quant
    dscale_q = 1 / scale_q
    dscale_k = 1 / scale_k
    dscale_v = 1 / scale_v
    dscale_dx = 1 / scale_dx
    if ctx.input_case["dtype"] == "fp8_e4m3fn" or ctx.input_case["dtype"] == "fp8_e5m2":
        q = block_quant_or_dequant_single(q, scale_q)
        k = block_quant_or_dequant_single(k, scale_k)
        v = block_quant_or_dequant_single(v, scale_v)
        dx = block_quant_or_dequant_single(dx, scale_dx)

    q_rope = q
    k_rope = k
    dq_rope_golden = dq_golden
    dk_rope_golden = dk_golden
    if ctx.input_case["rope"] == 1:
        q_rope = q[:, :, :, :, 128:]
        q = q[:, :, :, :, :128]
        k_rope = k[:, :, :, :, 128:]
        k = k[:, :, :, :, :128]
        dq_rope_golden = dq_golden[:, :, :, :, 128:]
        dk_rope_golden = dk_golden[:, :, :, :, 128:]
        dq_golden = dq_golden[:, :, :, :, :128]
        dk_golden = dk_golden[:, :, :, :, :128]
    else:
        q_rope = torch.tensor([0])
        k_rope = torch.tensor([0])
        dq_rope_golden = torch.tensor([0])
        dk_rope_golden = torch.tensor([0])
        q_rope_np = q_rope.float().numpy()
        k_rope_np = k_rope.float().numpy()
        dq_rope_np = dq_rope_golden.float().numpy()
        dk_rope_np = dk_rope_golden.float().numpy()

    x_max = x_max.reshape(B, Nq, S, 1).expand(B, Nq, S, 8)
    x_sum = x_sum.reshape(B, Nq, S, 1).expand(B, Nq, S, 8)
    if pse_type in (0, 1):
        pse = rearrange(pse, "b n g s d -> b (n g) s d").contiguous()
    if atten_mask_shape in ["B1SS", "BNSS"] and ctx.input_case["sparse_mode"] not in (
        2,
        3,
        4,
        6,
    ):
        atten_masks = rearrange(atten_masks, "b n g s d -> b (n g) s d").contiguous()
    if input_layout == "BSH":
        # (B,N,S,D)=>(B,S,N*D)
        if ctx.input_case["rope"] == 1:
            q_rope = rearrange(q_rope, "b n g s d -> b s (n g d)").contiguous()
            k_rope = rearrange(k_rope, "b n g s d -> b s (n g d)").contiguous()
            dq_rope_golden = rearrange(
                dq_rope_golden, "b n g s d -> b s (n g d)"
            ).contiguous()
            dk_rope_golden = rearrange(
                dk_rope_golden, "b n g s d -> b s (n g d)"
            ).contiguous()
        q = rearrange(q, "b n g s d -> b s (n g d)").contiguous()
        k = rearrange(k, "b n g s d -> b s (n g d)").contiguous()
        v = rearrange(v, "b n g s d -> b s (n g d)").contiguous()
        dx = rearrange(dx, "b n g s d -> b s (n g d)").contiguous()
        out = rearrange(out, "b n g s d -> b s (n g d)").contiguous()
        dq_golden = rearrange(dq_golden, "b n g s d -> b s (n g d)").contiguous()
        dk_golden = rearrange(dk_golden, "b n g s d -> b s (n g d)").contiguous()
        dv_golden = rearrange(dv_golden, "b n g s d -> b s (n g d)").contiguous()
    elif input_layout == "SBH":
        # (B,N,S,D)=>(S,B,N*D)
        if ctx.input_case["rope"] == 1:
            q_rope = rearrange(q_rope, "b n g s d -> s b (n g d)").contiguous()
            k_rope = rearrange(k_rope, "b n g s d -> s b (n g d)").contiguous()
            dq_rope_golden = rearrange(
                dq_rope_golden, "b n g s d -> s b (n g d)"
            ).contiguous()
            dk_rope_golden = rearrange(
                dk_rope_golden, "b n g s d -> s b (n g d)"
            ).contiguous()
        q = rearrange(q, "b n g s d -> s b (n g d)").contiguous()
        k = rearrange(k, "b n g s d -> s b (n g d)").contiguous()
        v = rearrange(v, "b n g s d -> s b (n g d)").contiguous()
        dx = rearrange(dx, "b n g s d -> s b (n g d)").contiguous()
        out = rearrange(out, "b n g s d -> s b (n g d)").contiguous()
        dq_golden = rearrange(dq_golden, "b n g s d -> s b (n g d)").contiguous()
        dk_golden = rearrange(dk_golden, "b n g s d -> s b (n g d)").contiguous()
        dv_golden = rearrange(dv_golden, "b n g s d -> s b (n g d)").contiguous()
    elif input_layout == "BSND" or input_layout == "TND":
        # (B,N,S,D)=>(B,S,N,D)
        if ctx.input_case["rope"] == 1:
            q_rope = rearrange(q_rope, "b n g s d -> b s (n g) d").contiguous()
            k_rope = rearrange(k_rope, "b n g s d -> b s (n g) d").contiguous()
            dq_rope_golden = rearrange(
                dq_rope_golden, "b n g s d -> b s (n g) d"
            ).contiguous()
            dk_rope_golden = rearrange(
                dk_rope_golden, "b n g s d -> b s (n g) d"
            ).contiguous()
        q = rearrange(q, "b n g s d -> b s (n g) d").contiguous()
        k = rearrange(k, "b n g s d -> b s (n g) d").contiguous()
        v = rearrange(v, "b n g s d -> b s (n g) d").contiguous()
        dx = rearrange(dx, "b n g s d -> b s (n g) d").contiguous()
        out = rearrange(out, "b n g s d -> b s (n g) d").contiguous()
        dq_golden = rearrange(dq_golden, "b n g s d -> b s (n g) d").contiguous()
        dk_golden = rearrange(dk_golden, "b n g s d -> b s (n g) d").contiguous()
        dv_golden = rearrange(dv_golden, "b n g s d -> b s (n g) d").contiguous()
    else:  # BNSD
        # (B,N,G,S,D)=>(B,N,S,D)
        if ctx.input_case["rope"] == 1:
            q_rope = rearrange(q_rope, "b n g s d -> b (n g) s d").contiguous()
            k_rope = rearrange(k_rope, "b n g s d -> b (n g) s d").contiguous()
            dq_rope_golden = rearrange(
                dq_rope_golden, "b n g s d -> b (n g) s d"
            ).contiguous()
            dk_rope_golden = rearrange(
                dk_rope_golden, "b n g s d -> b (n g) s d"
            ).contiguous()
        q = rearrange(q, "b n g s d -> b (n g) s d").contiguous()
        k = rearrange(k, "b n g s d -> b (n g) s d").contiguous()
        v = rearrange(v, "b n g s d -> b (n g) s d").contiguous()
        dx = rearrange(dx, "b n g s d -> b (n g) s d").contiguous()
        out = rearrange(out, "b n g s d -> b (n g) s d").contiguous()
        dq_golden = rearrange(dq_golden, "b n g s d -> b (n g) s d").contiguous()
        dk_golden = rearrange(dk_golden, "b n g s d -> b (n g) s d").contiguous()
        dv_golden = rearrange(dv_golden, "b n g s d -> b (n g) s d").contiguous()

    q_np = q.float().numpy()
    k_np = k.float().numpy()
    if ctx.input_case["rope"] == 1:
        q_rope_np = q_rope.float().numpy()
        k_rope_np = k_rope.float().numpy()
        dq_rope_np = dq_rope_golden.numpy()
        dk_rope_np = dk_rope_golden.numpy()
    v_np = v.float().numpy()
    dx_np = dx.float().numpy()
    out_np = out.float().numpy()
    x_max_np_b = x_max.float().numpy()
    x_sum_np_b = x_sum.float().numpy()
    pse_np = pse.float().numpy()
    atten_masks_np = atten_masks.numpy()
    dq_np = dq_golden.numpy()
    dk_np = dk_golden.numpy()
    dv_np = dv_golden.numpy()

    dscale_q_np = dscale_q.float().numpy()
    dscale_k_np = dscale_k.float().numpy()
    dscale_v_np = dscale_v.float().numpy()
    dscale_dx_np = dscale_dx.float().numpy()

    pttype = torch.float
    if input_layout == "TND":
        cu_seqlens_qInc = [
            sum(ctx.input_case["actual_seq_qlen"][: x + 1])
            for x in range(len(ctx.input_case["actual_seq_qlen"]))
        ]
        cu_seqlens_kvInc = [
            sum(ctx.input_case["actual_seq_kvlen"][: x + 1])
            for x in range(len(ctx.input_case["actual_seq_kvlen"]))
        ]
        np.array(cu_seqlens_qInc, dtype="int64")
        np.array(cu_seqlens_kvInc, dtype="int64")

        dq_np = unpad_input(dq_np, ctx.input_case["actual_seq_qlen"])
        q_np = unpad_input(q_np, ctx.input_case["actual_seq_qlen"])
        dx_np = unpad_input(dx_np, ctx.input_case["actual_seq_qlen"])
        out_np = unpad_input(out_np, ctx.input_case["actual_seq_qlen"])
        k_np = unpad_input(k_np, ctx.input_case["actual_seq_kvlen"])
        v_np = unpad_input(v_np, ctx.input_case["actual_seq_kvlen"])
        dk_np = unpad_input(dk_np, ctx.input_case["actual_seq_kvlen"])
        dv_np = unpad_input(dv_np, ctx.input_case["actual_seq_kvlen"])
        x_max_np_b = unpad_input(
            x_max_np_b, ctx.input_case["actual_seq_qlen"], softmax=True
        )
        x_sum_np_b = unpad_input(
            x_sum_np_b, ctx.input_case["actual_seq_qlen"], softmax=True
        )

        if atten_mask_shape in ["B1SS", "BNSS"]:
            atten_masks_np = unpad_input(
                atten_masks_np,
                ctx.input_case["actual_seq_qlen"],
                ctx.input_case["actual_seq_kvlen"],
                atten=True,
            )

        out_type = getOutDtype(ctx.input_case["out_dtype"])
        if ctx.input_case["dtype"] == "fp16":
            pttype = torch.float16
            out_type = pttype
        if ctx.input_case["dtype"] == "bf16":
            pttype = torch.bfloat16
            out_type = pttype
        if ctx.input_case["dtype"] == "fp32":
            pttype = torch.float
            out_type = pttype
        if ctx.input_case["dtype"] == "fp8_e4m3fn":
            pttype = torch.fp8_e4m3fn
        if ctx.input_case["dtype"] == "fp8_e5m2":
            pttype = torch.fp8_e5m2
        ctx.cu_seqlens_q_npu = cu_seqlens_qInc
        ctx.cu_seqlens_kv_npu = cu_seqlens_kvInc
        q = torch.from_numpy(q_np).to(pttype)
        k = torch.from_numpy(k_np).to(pttype)
        v = torch.from_numpy(v_np).to(pttype)
        dx = torch.from_numpy(dx_np).to(pttype)
        out = torch.from_numpy(out_np).to(pttype)
        dq_golden = torch.from_numpy(dq_np).to(pttype)
        dk_golden = torch.from_numpy(dk_np).to(pttype)
        dv_golden = torch.from_numpy(dv_np).to(pttype)
        x_max = torch.from_numpy(x_max_np_b)
        x_sum = torch.from_numpy(x_sum_np_b)
        ctx.x_max_npu = torch.from_numpy(x_max_np_b).detach()
        ctx.x_sum_npu = torch.from_numpy(x_sum_np_b).detach()

    ctx.q_npu = q.detach()
    ctx.k_npu = k.detach()
    ctx.v_npu = v.detach()
    ctx.q_rope_npu = q_rope.detach()
    ctx.k_rope_npu = k_rope.detach()
    ctx.dx_npu = dx.detach()
    ctx.atten_masks_npu = atten_masks.detach()

    if ctx.input_case["sparse_mode"] in (5, 6):
        prefix_np = np.array(prefix)
        prefix_torch = torch.from_numpy(prefix_np)
        ctx.prefix_npu = prefix_torch

    ctx.out_npu = out.detach()
    if not pse_shape == "NONE":
        ctx.pse_npu = pse.detach()

    ctx.x_max_npu = x_max.detach()
    ctx.x_sum_npu = x_sum.detach()
    ctx.dscale_q_npu = dscale_q.detach()
    ctx.dscale_k_npu = dscale_k.detach()
    ctx.dscale_v_npu = dscale_v.detach()
    ctx.dscale_dx_npu = dscale_dx.detach()
    ctx.dscale_o_npu = dscale_q.detach()

    if ctx.input_case["rope"] == 0:
        return dq_golden, dk_golden, dv_golden
    else:
        return dq_golden, dk_golden, dv_golden, dq_rope_golden, dk_rope_golden


def run_unpad(ctx: CalculationContext):
    B = ctx.input_case["B"]
    Nq = ctx.input_case["N1"]
    Nkv = ctx.input_case["N2"]
    S = ctx.input_case["S1"]
    Skv = ctx.input_case["S2"]
    D = ctx.input_case["D"]
    D_V = ctx.input_case["D_V"]
    dtype = ctx.input_case["dtype"].lower()  # 'fp16' & 'bf16' & 'fp32'

    N1 = ctx.input_case["N1"]
    N2 = ctx.input_case["N2"] if ctx.input_case["N2"] is not None else N1
    # S1 = case["S1"]
    # S2 = case["S2"] if not (case["S2"] is None) else S1
    G = int(N1 / N2)
    H = N1 * D
    scale = 1 / (D**0.5)
    keep_prob = ctx.input_case["drop_out_possibility"]
    dropout_p = 1 - keep_prob
    # padding_mask_drop_prob = 0.05  # padding_mask丢弃概率

    sparse_mode = int(
        ctx.input_case["sparse_mode"] if (ctx.input_case["sparse_mode"] != "") else 0
    )
    # 'BSH' & 'SBH' & 'BNSD' & 'BSND'
    input_layout = ctx.input_case["input_layout"].upper()
    # 'none' & 'bool' & 'qkv'
    atten_mask_dtype = ctx.input_case["atten_mask_dtype"].lower()
    # 'SS' & 'B1SS' & 'BNSS'
    atten_mask_shape = ctx.input_case["atten_mask_shape"].upper()
    # padding_mask_type = case["padding_mask"].upper()  # 'NONE' & 'SS' & 'B1SS' & 'BNSS'
    pse_layout = (
        ctx.input_case["pse_shape"].upper()
        if (ctx.input_case["pse_shape"] != "")
        else "NONE"
    )  # '1NHS' & 'BNHS' & 'None'
    pse_shape = pse_layout
    pse_type = int(
        ctx.input_case["pse_type"] if (ctx.input_case["pse_type"] != "") else 0
    )
    case_pre_tockens = ctx.input_case["pre_tockens_input"]
    case_next_tockens = ctx.input_case["next_tockens_input"]
    pre_tocken = case_pre_tockens
    next_tocken = case_next_tockens
    seqlens_list_q = np.array(ctx.input_case["actual_seq_qlen"])
    seqlens_list_k = (
        np.array(ctx.input_case["actual_seq_kvlen"])
        if ctx.input_case["actual_seq_kvlen"] is not None
        else seqlens_list_q
    )
    B = len(seqlens_list_q)
    mask_dtype = torch.bool
    if atten_mask_dtype.lower() == "bool":
        mask_dtype = torch.bool
    if atten_mask_dtype.lower() == "uint8":
        mask_dtype = torch.uint8

    cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
    cu_seqlens_k = get_cu_seqlens(seqlens_list_k)
    max_seqlen_q = seqlens_list_q.max()
    max_seqlen_k = seqlens_list_k.max()
    S1 = seqlens_list_q.sum()
    S2 = seqlens_list_k.sum()
    qk_size = seqlens_list_q * seqlens_list_k
    qk_pointer = get_cu_seqlens(qk_size)
    pttype = torch.float
    if ctx.input_case["dtype"] == "fp16":
        pttype = torch.float16
    if ctx.input_case["dtype"] == "bf16":
        pttype = torch.bfloat16
    if ctx.input_case["dtype"] == "fp32":
        pttype = torch.float
    # 设置qkv
    q = torch.from_numpy(np.random.randn(S1, N1, D)).to(pttype)
    k = torch.from_numpy(np.random.randn(S2, N2, D)).to(pttype)
    v = torch.from_numpy(np.random.randn(S2, N2, D_V)).to(pttype)
    dx = torch.from_numpy(np.random.randn(S1, N1, D_V)).to(pttype)

    # 生成量化参数
    scale_q = np.ones([1])
    scale_k = np.ones([1])
    scale_v = np.ones([1])
    scale_dx = np.ones([1])
    if ctx.input_case["dtype"] == "fp8_e4m3fn" or ctx.input_case["dtype"] == "fp8_e5m2":
        scale_q = get_block_quant_scale(q, ctx.input_case["dtype"])
        scale_k = get_block_quant_scale(k, ctx.input_case["dtype"])
        scale_v = get_block_quant_scale(v, ctx.input_case["dtype"])
        scale_dx = get_block_quant_scale(dx, ctx.input_case["dtype"])
    dscale_q = 1 / scale_q
    dscale_k = 1 / scale_k
    dscale_v = 1 / scale_v
    dscale_dx = 1 / scale_dx

    atten_mask, atten_mask_npu, prefix, pre_tocken, next_tocken = get_atten_mask_tnd(
        sparse_mode=sparse_mode,
        atten_mask_shape=atten_mask_shape,
        atten_mask_dtype=atten_mask_dtype,
        B=B,
        N1=N1,
        N2=N2,
        seqlens_list_q=seqlens_list_q,
        seqlens_list_k=seqlens_list_k,
        D=D,
        case_pre_tockens=case_pre_tockens,
        case_next_tockens=case_next_tockens,
        dtype=mask_dtype,
    )
    atten_masks_np = atten_mask_npu.numpy()

    is_skip_invalid_row = False
    if ctx.input_case["sparse_mode"] == 3:
        is_skip_invalid_row = S > Skv
    if ctx.input_case["sparse_mode"] == 4:
        is_skip_invalid_row = pre_tocken < 0 or next_tocken + Skv < S
    if ctx.input_case["sparse_mode"] == 0 and atten_mask is not None:
        is_skip_invalid_row = pre_tocken < 0 or next_tocken < 0
    if ctx.input_case["sparse_mode"] in (5, 6):
        is_skip_invalid_row = True if 0 in prefix else False

    AlibiH = 1024
    STmp = max(S, AlibiH)
    SkvTmp = max(Skv, AlibiH)

    if pse_type in (0, 1):
        if pse_shape == "BN1S":
            pse = torch.from_numpy(np.random.randn(B, Nkv, G, 1, Skv)).to(pttype)
            pse_golden = pse
        elif pse_shape == "1NSS":
            pse = torch.from_numpy(np.random.randn(1, Nkv, G, S, Skv)).to(pttype)
            pse_golden = pse
        elif pse_shape == "BNSS":
            pse = torch.from_numpy(np.random.randn(B, Nkv, G, S, Skv)).to(pttype)
            pse_golden = pse
        elif pse_shape == "BNHS" or pse_shape == "1NHS":
            pse_orgshape = (B, Nkv, G, STmp, Skv)
            if pse_shape == "1NHS":
                pse_orgshape = (1, Nkv, G, STmp, Skv)

            x, y = np.meshgrid(range(Skv), range(STmp))
            tensor = torch.from_numpy(np.tril(x - y))
            pse_golden = torch.zeros(pse_orgshape)
            m = get_slopes(Nq)
            for i in range(Nkv):
                for j in range(G):
                    pse_golden[:, i : i + 1, j : j + 1, :, :] = tensor * m[i * G + j]
            pse_golden = pse_golden.to(pttype)
            pse = pse_golden[:, :, :, (STmp - AlibiH) :, :]
            # pse_golden = pse_golden[:,:,:,(STmp-S):,(SkvTmp-Skv):]
            pse = rearrange(pse, "b n g s d -> b (n g) s d").contiguous()
            pse_golden = rearrange(pse_golden, "b n g s d -> b (n g) s d").contiguous()
        elif pse_shape.upper() == "NONE":
            pse = torch.tensor([0])
            pse_golden = pse
    elif pse_type in (2, 3):
        pse_golden, pse = get_all_alibi(
            B,
            Nq,
            S,
            Skv,
            pse_layout=pse_shape,
            pse_dtype=torch.float,
            pse_mode=pse_type,
        )

    pse_np = pse.float().numpy()

    # 设置drop_mask
    if keep_prob == 1:
        drop_mask = torch.tensor(1)
    else:
        drop_mask = get_drop_mask_tnd(ctx.input_case, 1 - keep_prob)

    # 运算golden
    out_golden = torch.zeros_like(dx)
    dq_golden = torch.zeros_like(q)
    dk_golden = torch.zeros_like(k)
    dv_golden = torch.zeros_like(v)
    x_max = torch.empty(0)
    x_sum = torch.empty(0)
    atten_masks = torch.empty(0)

    for i in range(B):
        if seqlens_list_q[i] != 0 and seqlens_list_k[i] != 0:
            qi = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]
            ki = k[cu_seqlens_k[i] : cu_seqlens_k[i + 1]]
            vi = v[cu_seqlens_k[i] : cu_seqlens_k[i + 1]]

            qi = rearrange(qi, "s n d -> 1 n s d")
            ki = rearrange(ki, "s n d -> 1 n s d")
            vi = rearrange(vi, "s n d -> 1 n s d")

            # N不等长适配by cdy
            if not (N1 == N2):
                ki = broadcastKV_sigle(N1, N2, ki, ki.dtype)
                vi = broadcastKV_sigle(N1, N2, vi, vi.dtype)
            qk_pointer = np.zeros(B + 1, dtype=np.int32)
            qk_pointer[0] = 0
            for j in range(B):
                qk_pointer[j + 1] = (
                    qk_pointer[j]
                    + seqlens_list_q[j] * math.ceil(seqlens_list_k[j] / 16) * 16
                )
            if drop_mask.numel() > 1:
                drop_maski = drop_mask[
                    (qk_pointer[i] * N1) : (qk_pointer[i + 1] * N1)
                ].reshape(
                    N1, seqlens_list_q[i], math.ceil(seqlens_list_k[i] / 16) * 16
                )[:, :, : seqlens_list_k[i]]
            else:
                drop_maski = drop_mask

            if sparse_mode in [5, 6, 7, 8]:
                atten_maski = get_unpad_atten_mask(
                    sparse_mode=sparse_mode,
                    N1=N1,
                    seqlens_list_q=seqlens_list_q,
                    seqlens_list_k=seqlens_list_k,
                    atten_mask_shape=atten_mask_shape,
                    atten_mask_dtype=atten_mask_dtype,
                    pre_tocken=pre_tocken,
                    next_tocken=next_tocken,
                    mask_dtype=mask_dtype,
                    t=i,
                    prefix=prefix,
                )
            else:
                if atten_mask.numel() > 1:
                    if sparse_mode in [3, 4, 5]:
                        if atten_mask_shape == "SS":
                            atten_maski = atten_mask[
                                max_seqlen_q - seqlens_list_q[i] :,
                                max_seqlen_k - seqlens_list_k[i] :,
                            ]
                        else:
                            atten_maski = atten_mask[
                                :,
                                :,
                                max_seqlen_q - seqlens_list_q[i] :,
                                max_seqlen_k - seqlens_list_k[i] :,
                            ]
                    else:
                        if atten_mask_shape == "SS":
                            atten_maski = atten_mask[
                                : seqlens_list_q[i], : seqlens_list_k[i]
                            ]
                        else:
                            atten_maski = atten_mask[
                                :, :, : seqlens_list_q[i], : seqlens_list_k[i]
                            ]
                else:
                    atten_maski = atten_mask

            if pse_golden.numel() > 1 and pse_layout != "NONE":
                if pse_layout == "1NHS":
                    psei = pse_golden[
                        :,
                        :,
                        (STmp - seqlens_list_q[i]) : STmp,
                        (max_seqlen_k - seqlens_list_k[i]) : max_seqlen_k,
                    ]
                    # psei = pse_golden[:, :, :seqlens_list_q[i], :seqlens_list_k[i]]
                elif pse_layout == "BNHS":
                    psei = pse_golden[
                        i : i + 1,
                        :,
                        (STmp - seqlens_list_q[i]) : STmp,
                        (max_seqlen_k - seqlens_list_k[i]) : max_seqlen_k,
                    ]
                    # psei = pse_golden[i:i+1, :, :seqlens_list_q[i], :seqlens_list_k[i]]
                elif pse_layout == "BN":
                    psei = pse_golden[
                        i : i + 1,
                        :,
                        (max_seqlen_q - seqlens_list_q[i]) : max_seqlen_q,
                        (max_seqlen_k - seqlens_list_k[i]) : max_seqlen_k,
                    ]
                elif pse_layout == "N":
                    psei = pse_golden[
                        :,
                        :,
                        (max_seqlen_q - seqlens_list_q[i]) : max_seqlen_q,
                        (max_seqlen_k - seqlens_list_k[i]) : max_seqlen_k,
                    ]

            else:
                psei = torch.tensor([0])

            # 正向golden运算
            outi_golden, softmax_resi, x_maxi, x_sumi = tforward_i(
                qi.to(gtype),
                ki.to(gtype),
                vi.to(gtype),
                drop_maski,
                atten_maski,
                psei.to(gtype),
                scale,
                keep_prob,
                is_skip_invalid_row,
                pse_type,
            )

            out_golden[cu_seqlens_q[i] : cu_seqlens_q[i + 1]] = rearrange(
                outi_golden, "1 n s d -> s n d"
            )

            # 记录max,sum
            x_maxi = x_maxi.broadcast_to(1, N1, seqlens_list_q[i], 8)
            x_maxi = x_maxi.contiguous().view(-1)
            x_sumi = x_sumi.broadcast_to(1, N1, seqlens_list_q[i], 8)
            x_sumi = x_sumi.contiguous().view(-1)
            x_max = torch.cat([x_max, x_maxi], dim=0)
            x_sum = torch.cat([x_sum, x_sumi], dim=0)

            dxi = dx[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]
            dxi = rearrange(dxi, "s n d -> 1 n s d")

            # 反向golden运算
            dqi_golden, dki_golden, dvi_golden = tbackward_i(
                dxi.to(gtype),
                qi.to(gtype),
                ki.to(gtype),
                vi.to(gtype),
                softmax_resi.to(gtype),
                drop_maski.to(gtype),
                psei.to(gtype),
                scale,
                keep_prob,
            )

            # N不等长适配by cdy
            if not (N1 == N2):
                G = int(N1 / N2)
                s2i = seqlens_list_k[i]
                dki_golden = torch.sum(
                    dki_golden.reshape(1, N2, G, s2i, D), dim=2, keepdim=True
                ).reshape(1, N2, s2i, D)
                dvi_golden = torch.sum(
                    dvi_golden.reshape(1, N2, G, s2i, D_V), dim=2, keepdim=True
                ).reshape(1, N2, s2i, D_V)

            # 记录dqkv
            dq_golden[cu_seqlens_q[i] : cu_seqlens_q[i + 1]] = rearrange(
                dqi_golden, "1 n s d -> s n d"
            )
            dk_golden[cu_seqlens_k[i] : cu_seqlens_k[i + 1]] = rearrange(
                dki_golden, "1 n s d -> s n d"
            )
            dv_golden[cu_seqlens_k[i] : cu_seqlens_k[i + 1]] = rearrange(
                dvi_golden, "1 n s d -> s n d"
            )
        else:
            print_log(f"s_{i}为空，sq={seqlens_list_q[i]}，skv={seqlens_list_k[i]}")

    q_rope = q
    k_rope = k
    dq_rope_golden = dq_golden
    dk_rope_golden = dk_golden
    if ctx.input_case["rope"] == 1:
        q_rope = q[:, :, 128:]
        q = q[:, :, :128]
        k_rope = k[:, :, 128:]
        k = k[:, :, :128]
        dq_rope_golden = dq_golden[:, :, 128:]
        dk_rope_golden = dk_golden[:, :, 128:]
        dq_golden = dq_golden[:, :, :128]
        dk_golden = dk_golden[:, :, :128]
        q_rope_np = q_rope.float().numpy()
        k_rope_np = k_rope.float().numpy()
        dq_rope_np = dq_rope_golden.float().numpy()
        dk_rope_np = dk_rope_golden.float().numpy()
    else:
        q_rope = torch.tensor([0])
        k_rope = torch.tensor([0])
        dq_rope_golden = torch.tensor([0])
        dk_rope_golden = torch.tensor([0])
        q_rope_np = q_rope.float().numpy()
        k_rope_np = k_rope.float().numpy()
        dq_rope_np = dq_rope_golden.float().numpy()
        dk_rope_np = dk_rope_golden.float().numpy()

    save_dtype = np.float16
    if dtype == "bf16":
        # save_dtype = np.float16
        save_dtype = ml_dtypes.bfloat16
    elif dtype == "fp32":
        save_dtype = np.float32

    out_np = out_golden.float().numpy()

    dq_np = dq_golden.float().numpy()
    dk_np = dk_golden.float().numpy()
    dv_np = dv_golden.float().numpy()

    x_max_np_b = x_max.float().numpy()
    x_sum_np_b = x_sum.float().numpy()
    q_np = q.float().numpy()
    k_np = k.float().numpy()
    v_np = v.float().numpy()
    dx_np = dx.float().numpy()

    if ctx.input_case["sparse_mode"] in (5, 6):
        prefix_np = np.array(prefix)
        prefix_torch = torch.from_numpy(prefix_np)
        ctx.prefix_npu = prefix_torch

    pse_save_dtype = save_dtype if pse_type in (0, 1) else np.float32

    cu_seqlens_qInc = [
        sum(ctx.input_case["actual_seq_qlen"][: x + 1])
        for x in range(len(ctx.input_case["actual_seq_qlen"]))
    ]
    cu_seqlens_kvInc = [
        sum(ctx.input_case["actual_seq_kvlen"][: x + 1])
        for x in range(len(ctx.input_case["actual_seq_kvlen"]))
    ]

    ctx.cu_seqlens_q_npu = cu_seqlens_qInc
    ctx.cu_seqlens_kv_npu = cu_seqlens_kvInc
    ctx.x_max_npu = x_max.reshape(S1, N1, 8).detach()
    ctx.x_sum_npu = x_sum.reshape(S1, N1, 8).detach()
    ctx.atten_masks_npu = atten_mask_npu.detach()
    out = torch.from_numpy(out_np).to(pttype)
    dq_golden = torch.from_numpy(dq_np).to(pttype)
    dk_golden = torch.from_numpy(dk_np).to(pttype)
    dv_golden = torch.from_numpy(dv_np).to(pttype)
    ctx.out_npu = out.detach()
    ctx.pse_npu = pse.detach()
    ctx.q_npu = q.detach()
    ctx.k_npu = k.detach()
    ctx.v_npu = v.detach()
    ctx.dx_npu = dx.detach()

    # save rope
    ctx.q_rope_npu = q_rope.detach()
    ctx.k_rope_npu = k_rope.detach()
    if ctx.input_case["rope"] == 0:
        return dq_golden, dk_golden, dv_golden
    else:
        return dq_golden, dk_golden, dv_golden, dq_rope_golden, dk_rope_golden


def DataGen(case):
    B = case["B"]
    Nq = case["N1"]
    Nkv = case["N2"]
    S = case["S1"]
    Skv = case["S2"]
    D = case["D"]
    D_V = case["D_V"]
    pse_type = case["pse_type"]
    outDtype = case["out_dtype"]
    AlibiH = 1024
    pre_tockens = case["pre_tockens_input"] if "pre_tockens_input" in case else 65536
    next_tockens = case["next_tockens_input"] if "next_tockens_input" in case else 65536
    input_layout = case["input_layout"] if "input_layout" in case else "SBH"
    if Nkv != 0:
        G = Nq // Nkv
    else:
        G = 0
    pttype = torch.float
    pse_dtype = torch.float16
    if case["dtype"] == "fp16":
        pttype = torch.float16
        pse_dtype = torch.float16
    if case["dtype"] == "bf16":
        pttype = torch.bfloat16
        pse_dtype = torch.bfloat16
    if case["dtype"] == "fp32":
        pttype = torch.float
        pse_dtype = torch.float
    if case["dtype"] == "fp8_e4m3fn" or case["dtype"] == "fp8_e5m2":
        pttype = torch.float
        if outDtype == "fp16":
            pse_dtype = torch.float16
        else:
            pse_dtype = torch.bfloat16
    pse_shape = case["pse_shape"] if "pse_shape" in case else "NONE"
    atten_mask_shape = case["atten_mask_shape"] if "atten_mask_shape" in case else "SS"

    # random data,shape [BNGSD]
    q = np.random.randn(B, Nkv, G, S, D)
    k = np.random.randn(B, Nkv, 1, Skv, D)
    v = np.random.randn(B, Nkv, 1, Skv, D_V)
    dx = np.random.randn(B, Nkv, G, S, D_V)

    # 生成量化参数
    scale_q = np.zeros([1])
    scale_k = np.zeros([1])
    scale_v = np.zeros([1])
    scale_dx = np.zeros([1])
    if case["dtype"] == "fp8_e4m3fn" or case["dtype"] == "fp8_e5m2":
        scale_q = get_block_quant_scale(q, case["dtype"])
        scale_k = get_block_quant_scale(k, case["dtype"])
        scale_v = get_block_quant_scale(v, case["dtype"])
        scale_dx = get_block_quant_scale(dx, case["dtype"])

    scale_q = torch.from_numpy(scale_q).to(torch.float32)
    scale_k = torch.from_numpy(scale_k).to(torch.float32)
    scale_v = torch.from_numpy(scale_v).to(torch.float32)
    scale_dx = torch.from_numpy(scale_dx).to(torch.float32)

    q = torch.from_numpy(q).to(pttype)
    k = torch.from_numpy(k).to(pttype)
    v = torch.from_numpy(v).to(pttype)
    dx = torch.from_numpy(dx).to(pttype)
    # Pse data,shape [BNGSD]
    if pse_type in (0, 1):
        if pse_shape == "BN1S":
            pse = torch.from_numpy(np.random.randn(B, Nkv, G, 1, Skv)).to(pse_dtype)
            pse_golden = pse
        elif pse_shape == "1NSS":
            pse = torch.from_numpy(np.random.randn(1, Nkv, G, S, Skv)).to(pse_dtype)
            pse_golden = pse
        elif pse_shape == "BNSS":
            pse = torch.from_numpy(np.random.randn(B, Nkv, G, S, Skv)).to(pse_dtype)
            pse_golden = pse
        elif pse_shape == "BNHS" or pse_shape == "1NHS":
            STmp = max(S, AlibiH)
            SkvTmp = max(Skv, AlibiH)
            pse_orgshape = (B, Nkv, G, STmp, SkvTmp)
            if pse_shape == "1NHS":
                pse_orgshape = (1, Nkv, G, STmp, SkvTmp)

            x, y = np.meshgrid(range(SkvTmp), range(STmp))
            tensor = torch.from_numpy(np.tril(x - y))
            pse_golden = torch.zeros(pse_orgshape)
            m = get_slopes(Nq)
            for i in range(Nkv):
                for j in range(G):
                    pse_golden[:, i : i + 1, j : j + 1, :, :] = tensor * m[i * G + j]
            pse_golden = pse_golden.to(pse_dtype)
            pse = pse_golden[:, :, :, (STmp - AlibiH) :, (SkvTmp - Skv) :]
            pse_golden = pse_golden[:, :, :, (STmp - S) :, (SkvTmp - Skv) :]
        elif pse_shape.upper() == "NONE":
            pse = torch.from_numpy(np.zeros([B, Nkv, G, S, Skv])).to(pse_dtype)
            pse_golden = pse
    else:
        pse_golden, pse = get_all_alibi(
            B,
            Nq,
            S,
            Skv,
            pse_layout=pse_shape,
            pse_dtype=torch.float,
            pse_mode=pse_type,
        )
        if pse_shape == "BN":
            pse_golden = pse_golden.reshape(B, Nkv, G, S, Skv)
        elif pse_shape == "N":
            pse_golden = pse_golden.reshape(Nkv, G, S, Skv).broadcast_to(
                B, Nkv, G, S, Skv
            )

    if atten_mask_shape.upper() == "NONE":
        atten_masks_golden = (
            torch.from_numpy(np.zeros([S, Skv])).to(torch.uint8).numpy()
        )
        atten_mask_npu = atten_masks_golden
        prefix = None
    else:
        atten_masks_golden, atten_mask_npu, prefix = get_atten_mask(
            case["sparse_mode"],
            atten_mask_shape,
            B,
            Nkv,
            G,
            S,
            Skv,
            pre_tockens,
            next_tockens,
        )
    return (
        q,
        k,
        v,
        dx,
        pse,
        pse_golden,
        prefix,
        torch.from_numpy(atten_masks_golden).to(torch.uint8),
        torch.from_numpy(atten_mask_npu).to(torch.uint8),
        scale_q,
        scale_k,
        scale_v,
        scale_dx,
    )


def get_block_quant_scale(tensor, data_type, is_scale_ds=False, is_vcore_split=False):
    if data_type == "fp8_e5m2":
        FP8_MAX = 57344.0
    elif data_type == "fp8_e4m3fn":
        FP8_MAX = 448.0
    else:
        FP8_MAX = -1
        print(f"{data_type} Not support block quant")
        return None

    dim1 = tensor.shape[0]  # b
    dim2 = tensor.shape[1]  # n2
    dim3 = tensor.shape[2]  # g
    dim4 = tensor.shape[3]  # s1
    dim5 = tensor.shape[4]  # d or s2
    if not is_scale_ds:
        scale_for_tensor = np.ones(
            [dim1, dim2, dim3, math.ceil(dim4 / PER_BLOCK_SIZE), 1]
        )
        for b in range(dim1):
            for n in range(dim2):
                for g in range(dim3):
                    for s in range(0, dim4, PER_BLOCK_SIZE):
                        # 取出128*D个数
                        s_start = s // PER_BLOCK_SIZE * PER_BLOCK_SIZE
                        s_end = min(
                            s // PER_BLOCK_SIZE * PER_BLOCK_SIZE + PER_BLOCK_SIZE, dim4
                        )
                        chunk = tensor[b, n, g, s_start:s_end, :]
                        max_val = np.max(np.abs(chunk))
                        scale_for_tensor[b, n, g, s // PER_BLOCK_SIZE, 0] = (
                            FP8_MAX / max_val
                        )
    else:
        if is_vcore_split:
            if dim4 % PER_VCORE_BLOCK_SIZE == 0:
                scale_for_tensor = np.ones(
                    [
                        dim1,
                        dim2,
                        dim3,
                        math.ceil(dim4 / PER_VCORE_BLOCK_SIZE),
                        math.ceil(dim5 / PER_BLOCK_SIZE),
                    ]
                )
            else:
                scale_for_tensor = np.ones(
                    [
                        dim1,
                        dim2,
                        dim3,
                        math.ceil(dim4 / PER_VCORE_BLOCK_SIZE) + 1,
                        math.ceil(dim5 / PER_BLOCK_SIZE),
                    ]
                )
            s1_tail = (
                PER_BLOCK_SIZE if dim4 % PER_BLOCK_SIZE == 0 else dim4 % PER_BLOCK_SIZE
            )
            for b in range(dim1):
                for n in range(dim2):
                    for g in range(dim3):
                        for s2 in range(0, dim5, PER_BLOCK_SIZE):
                            s2_start = s2 // PER_BLOCK_SIZE * PER_BLOCK_SIZE
                            s2_end = min(
                                s2 // PER_BLOCK_SIZE * PER_BLOCK_SIZE + PER_BLOCK_SIZE,
                                dim5,
                            )
                            for s1 in range(0, dim4 - s1_tail, PER_VCORE_BLOCK_SIZE):
                                s1_start = (
                                    s1 // PER_VCORE_BLOCK_SIZE * PER_VCORE_BLOCK_SIZE
                                )
                                s1_end = min(s1_start + PER_VCORE_BLOCK_SIZE, dim4)

                                chunk = tensor[
                                    b, n, g, s1_start:s1_end, s2_start:s2_end
                                ]
                                max_val = np.max(np.abs(chunk.numpy()))
                                if np.abs(max_val) < EPSILON:
                                    scale_for_tensor[
                                        b,
                                        n,
                                        g,
                                        s1 // PER_VCORE_BLOCK_SIZE,
                                        s2 // PER_BLOCK_SIZE,
                                    ] = 0
                                else:
                                    scale_for_tensor[
                                        b,
                                        n,
                                        g,
                                        s1 // PER_VCORE_BLOCK_SIZE,
                                        s2 // PER_BLOCK_SIZE,
                                    ] = FP8_MAX / max_val

                            # 处理尾块 v0
                            s1_start = dim4 - s1_tail
                            s1_end = s1_start + math.ceil(s1_tail / 2)
                            chunk = tensor[b, n, g, s1_start:s1_end, s2_start:s2_end]
                            max_val = np.max(np.abs(chunk.numpy()))
                            if np.abs(max_val) < EPSILON:
                                scale_for_tensor[
                                    b,
                                    n,
                                    g,
                                    s1_start // PER_VCORE_BLOCK_SIZE,
                                    s2 // PER_BLOCK_SIZE,
                                ] = 0
                            else:
                                scale_for_tensor[
                                    b,
                                    n,
                                    g,
                                    s1_start // PER_VCORE_BLOCK_SIZE,
                                    s2 // PER_BLOCK_SIZE,
                                ] = FP8_MAX / max_val
                            v0_s1_start_idx = s1_start // PER_VCORE_BLOCK_SIZE
                            # 处理尾块 v1
                            s1_start = s1_end
                            s1_end = dim4
                            chunk = tensor[b, n, g, s1_start:s1_end, s2_start:s2_end]
                            max_val = np.max(np.abs(chunk.numpy()))
                            if np.abs(max_val) < EPSILON:
                                scale_for_tensor[
                                    b, n, g, v0_s1_start_idx + 1, s2 // PER_BLOCK_SIZE
                                ] = 0
                            else:
                                scale_for_tensor[
                                    b, n, g, v0_s1_start_idx + 1, s2 // PER_BLOCK_SIZE
                                ] = FP8_MAX / max_val
        else:
            if dim4 % PER_VCORE_BLOCK_SIZE == 0:
                scale_for_tensor = np.ones(
                    [
                        dim1,
                        dim2,
                        dim3,
                        math.ceil(dim4 / PER_BLOCK_SIZE),
                        math.ceil(dim5 / PER_BLOCK_SIZE),
                    ]
                )
            else:
                scale_for_tensor = np.ones(
                    [
                        dim1,
                        dim2,
                        dim3,
                        math.ceil(dim4 / PER_BLOCK_SIZE) + 1,
                        math.ceil(dim5 / PER_BLOCK_SIZE),
                    ]
                )
            s1_tail = (
                PER_BLOCK_SIZE if dim4 % PER_BLOCK_SIZE == 0 else dim4 % PER_BLOCK_SIZE
            )
            for b in range(dim1):
                for n in range(dim2):
                    for g in range(dim3):
                        for s2 in range(0, dim5, PER_BLOCK_SIZE):
                            s2_start = s2 // PER_BLOCK_SIZE * PER_BLOCK_SIZE
                            s2_end = min(
                                s2 // PER_BLOCK_SIZE * PER_BLOCK_SIZE + PER_BLOCK_SIZE,
                                dim5,
                            )
                            for s1 in range(0, dim4 - s1_tail, PER_BLOCK_SIZE):
                                s1_start = s1 // PER_BLOCK_SIZE * PER_BLOCK_SIZE
                                s1_end = min(s1_start + PER_BLOCK_SIZE, dim4)

                                chunk = tensor[
                                    b, n, g, s1_start:s1_end, s2_start:s2_end
                                ]
                                max_val = np.max(np.abs(chunk.numpy()))
                                if np.abs(max_val) < EPSILON:
                                    scale_for_tensor[
                                        b,
                                        n,
                                        g,
                                        s1 // PER_BLOCK_SIZE,
                                        s2 // PER_BLOCK_SIZE,
                                    ] = 0
                                else:
                                    scale_for_tensor[
                                        b,
                                        n,
                                        g,
                                        s1 // PER_BLOCK_SIZE,
                                        s2 // PER_BLOCK_SIZE,
                                    ] = FP8_MAX / max_val

                            # 处理尾块
                            s1_start = dim4 - s1_tail
                            s1_end = dim4
                            chunk = tensor[b, n, g, s1_start:s1_end, s2_start:s2_end]
                            max_val = np.max(np.abs(chunk.numpy()))
                            if np.abs(max_val) < EPSILON:
                                scale_for_tensor[
                                    b,
                                    n,
                                    g,
                                    s1_start // PER_BLOCK_SIZE,
                                    s2 // PER_BLOCK_SIZE,
                                ] = 0
                            else:
                                scale_for_tensor[
                                    b,
                                    n,
                                    g,
                                    s1_start // PER_BLOCK_SIZE,
                                    s2 // PER_BLOCK_SIZE,
                                ] = FP8_MAX / max_val
    return scale_for_tensor


def unpad_input(
    single_input,
    actual_seq_lengths,
    actual_seq_lengths_kv=None,
    softmax=False,
    atten=False,
    drop=False,
):
    b = single_input.shape[0]
    N = single_input.shape[1]
    if actual_seq_lengths_kv is None:
        if softmax:
            q = single_input[0, :, : actual_seq_lengths[0], :8]
            q = q.reshape(actual_seq_lengths[0] * 8 * single_input.shape[1])
        else:
            q = single_input[0][: actual_seq_lengths[0]]
    else:
        if atten:
            q = single_input[0, :, : actual_seq_lengths[0], : actual_seq_lengths_kv[0]]
            q = q.reshape([N * actual_seq_lengths[0] * actual_seq_lengths_kv[0]])
        if drop:
            q = single_input[
                0, :, :, : actual_seq_lengths[0], : actual_seq_lengths_kv[0]
            ]
            q = q.reshape(
                [
                    single_input.shape[1]
                    * single_input.shape[2]
                    * actual_seq_lengths[0]
                    * actual_seq_lengths_kv[0]
                ]
            )

    for i in range(b - 1):
        if actual_seq_lengths_kv is None:
            if softmax:
                tmp = single_input[i + 1, :, : actual_seq_lengths[i + 1], :8]
                tmp = tmp.reshape(actual_seq_lengths[i + 1] * 8 * single_input.shape[1])
            else:
                tmp = single_input[i + 1][: actual_seq_lengths[i + 1]]
        else:
            if atten:  # BNSS or B1SS
                tmp = single_input[
                    i + 1,
                    :,
                    : actual_seq_lengths[i + 1],
                    : actual_seq_lengths_kv[i + 1],
                ]
                tmp = tmp.reshape(
                    [N * actual_seq_lengths[i + 1] * actual_seq_lengths_kv[i + 1]]
                )
            if drop:
                tmp = single_input[
                    i + 1,
                    :,
                    :,
                    : actual_seq_lengths[i + 1],
                    : actual_seq_lengths_kv[i + 1],
                ]
                tmp = tmp.reshape(
                    [
                        single_input.shape[1]
                        * single_input.shape[2]
                        * actual_seq_lengths[i + 1]
                        * actual_seq_lengths_kv[i + 1]
                    ]
                )
        q = np.concatenate((q, tmp), axis=0)
    if softmax:
        q = q.reshape([sum(actual_seq_lengths), N, 8])
    return q


def get_drop_mask(case, gen_p=0.2, pttype=torch.float16):
    B = case["B"]
    Nq = case["N1"]
    Nkv = case["N2"]
    S = case["S1"]
    Skv = case["S2"]
    D = case["D"]
    D_V = case["D_V"]
    if Nkv != 0:
        G = Nq // Nkv
    else:
        G = 0
    dtype = case["dtype"]
    keep_prob = 1 - gen_p
    if keep_prob == 1:
        return torch.tensor(1)
    rounds = 7
    seed_high32 = case["seed"] // 0x100000000
    seed_low32 = case["seed"] - seed_high32 * 0x100000000
    key = [seed_low32, seed_high32]
    counter = [math.ceil(case["offset"] / 16), 0, 0, 0]
    S2aligned = math.ceil(Skv / 16) * 16
    count = B * Nq * S * S2aligned
    golden_uint8 = gen_dropmask_uint8(rounds, counter, key, count, keep_prob)
    uint8_num_list = bin_to_uint8(golden_uint8)
    drop_masks_uint8_np = (
        np.array(golden_uint8)
        .reshape([B, Nq, S, S2aligned])[:, :, :, : -(S2aligned - Skv)]
        .astype(np.uint8)
        if S2aligned != Skv
        else np.array(golden_uint8).reshape([B, Nq, S, Skv]).astype(np.uint8)
    )
    drop_masks_bit = torch.from_numpy(drop_masks_uint8_np)
    drop_masks_bit = torch.tensor(drop_masks_bit, dtype=torch.uint8)
    drop_masks_uint8_np_1 = np.packbits(drop_masks_bit, bitorder="little")
    return drop_masks_bit.cpu()


def get_drop_mask_tnd(case, gen_p=0.2, pttype=torch.float16):
    B = case["B"]
    Nq = case["N1"]
    Nkv = case["N2"]
    S = case["S1"]
    Skv = case["S2"]
    D = case["D"]
    D_V = case["D_V"]
    G = Nq // Nkv
    dtype = case["dtype"]
    keep_prob = 1 - gen_p
    rounds = 7
    seed_high32 = case["seed"] // 0x100000000
    seed_low32 = case["seed"] - seed_high32 * 0x100000000
    key = [seed_low32, seed_high32]
    counter = [math.ceil(case["offset"] / 16), 0, 0, 0]
    seqlens_list_q = np.array(case["actual_seq_qlen"])
    seqlens_list_kv = np.array(case["actual_seq_kvlen"])
    qk_pointer = np.zeros(B + 1, dtype=np.int32)
    qk_pointer[0] = 0
    for i in range(B):
        qk_pointer[i + 1] = (
            qk_pointer[i] + seqlens_list_q[i] * math.ceil(seqlens_list_kv[i] / 16) * 16
        )
    count = Nq * int(qk_pointer[B])

    golden_uint8 = gen_dropmask_uint8(rounds, counter, key, count, keep_prob)
    uint8_num_list = bin_to_uint8(golden_uint8)
    drop_mask1 = np.zeros([B, Nq, S, Skv])
    drop_masks_bit1 = torch.from_numpy(np.array(golden_uint8).astype(np.uint8))
    for i in range(B):
        S2aligned = math.ceil(seqlens_list_kv[i] / 16) * 16
        drop_mask1[i, :, : seqlens_list_q[i], : seqlens_list_kv[i]] = (
            np.array(golden_uint8)[Nq * qk_pointer[i] : Nq * qk_pointer[i + 1]]
            .reshape(Nq, seqlens_list_q[i], S2aligned)[:, :, : seqlens_list_kv[i]]
            .astype(np.uint8)
        )

    drop_masks_uint8_np = drop_mask1
    drop_masks_bit = torch.from_numpy(drop_masks_uint8_np)
    drop_masks_bit = drop_masks_bit.to(dtype=torch.uint8)
    drop_masks_uint8_np_1 = np.packbits(drop_masks_bit, bitorder="little")
    return drop_masks_bit1.cpu()


def tforward_i(
    q, k, v, drop_mask, atten_mask, pse, scale, keep_prob, is_skip_invalid_row, pse_type
):
    qk = None
    if pse_type == 1:
        qk = torch.matmul(q, k.permute(0, 1, 3, 2)).add(pse).mul(scale)
    else:
        qk = torch.matmul(q, k.permute(0, 1, 3, 2)).mul(scale).add(pse)
    if atten_mask is not None:
        # qk = qk + atten_mask * (-40000.0)
        qk = qk.masked_fill(atten_mask.bool(), value=torch.tensor(-40000.0))

    softmax_res, x_max, x_sum = tsoftmax(qk)
    softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0
    # softmax_res = torch.softmax(qk, dim=-1)
    if is_skip_invalid_row:
        print("tforward need skip invalid row")
        # print(softmax_res.shape)
        b, n, s1, s2 = softmax_res.shape
        softmax_res, x_max, x_sum = skip_zero_row_tnd(
            b, n, s1, x_max, x_sum, softmax_res
        )
        softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0
    drop_res = softmax_res * drop_mask * (1.0 / (keep_prob))
    y = torch.matmul(drop_res, v)
    return y, softmax_res, x_max, x_sum


def tbackward_i(dx, q, k, v, softmax_res, drop_mask, pse, scale, keep_prob):
    drop_res = softmax_res.mul(drop_mask).mul(1.0 / (keep_prob))
    dv = torch.matmul(drop_res.permute(0, 1, 3, 2), dx)
    dp = torch.matmul(dx, v.permute(0, 1, 3, 2))
    dp_drop = dp * drop_mask * (1.0 / (keep_prob))
    softmax_grad_res = tsoftmax_grad(dp_drop, softmax_res) * scale
    dq = torch.matmul(softmax_grad_res, k)
    dk = torch.matmul(softmax_grad_res.permute(0, 1, 3, 2), q)
    return dq, dk, dv


def get_unpad_atten_mask(
    sparse_mode,
    N1,
    seqlens_list_q,
    seqlens_list_k,
    atten_mask_shape,
    atten_mask_dtype,
    pre_tocken,
    next_tocken,
    mask_dtype,
    t,
    prefix,
):
    band_index = 0
    max_seqlen_q = seqlens_list_q.max()
    max_seqlen_k = seqlens_list_k.max()
    B = len(seqlens_list_q)
    if atten_mask_shape == "SS":
        shape = [seqlens_list_q[t], seqlens_list_k[t]]
    elif atten_mask_shape == "11SS":
        shape = [1, 1, seqlens_list_q[t], seqlens_list_k[t]]
    elif atten_mask_shape == "B1SS":
        shape = [1, 1, seqlens_list_q[t], seqlens_list_k[t]]
    elif atten_mask_shape == "BNSS":
        shape = [1, N1, seqlens_list_q[t], seqlens_list_k[t]]
    elif atten_mask_dtype.lower() == "none" or atten_mask_shape.lower() == "none":
        atten_mask = torch.tensor(0)
        prefix = None
        pre_tocken = 65536
        next_tocken = 65536
        return atten_mask
    else:
        raise RuntimeError(
            f"not support shape of atten_mask {atten_mask_shape}, only support [S1,S2], [B, 1, S1, S2], [B, N1, S1, S2] or None"
        )

    if sparse_mode == 5 or sparse_mode == 6:  # prefix,shape只能是BNSS或者B1SS
        opname = "sparse_flash_attention"
        atten_mask = (
            torch.triu(
                torch.ones(shape), diagonal=seqlens_list_k[t] - seqlens_list_q[t] + 1
            )
        ).to(mask_dtype)
        if len(shape) > 2:
            atten_mask[:, :, :, 0 : prefix[t]] = 0
        else:
            atten_mask[:, 0 : prefix[t]] = 0
    elif sparse_mode == 7:
        prefix = None
        for index in range(B - 1, -1, -1):
            if seqlens_list_q[index] != 0:
                band_index = index
                break
        if t == band_index:
            atten_mask_u = torch.triu(
                torch.ones(shape),
                diagonal=next_tocken + 1 + seqlens_list_k[t] - seqlens_list_q[t],
            )
            atten_mask_l = torch.tril(
                torch.ones(shape),
                diagonal=-pre_tocken - 1 + seqlens_list_k[t] - seqlens_list_q[t],
            )
            atten_mask = (atten_mask_u + atten_mask_l).to(mask_dtype)

        else:
            atten_mask = (
                torch.triu(
                    torch.ones(shape),
                    diagonal=seqlens_list_k[t] - seqlens_list_q[t] + 1,
                )
            ).to(mask_dtype)
    elif sparse_mode == 8:
        prefix = None
        for index in range(B):
            if seqlens_list_q[index] != 0:
                band_index = index
                break
        if t == band_index:
            atten_mask_u = torch.triu(
                torch.ones(shape),
                diagonal=next_tocken + 1 + seqlens_list_k[t] - seqlens_list_q[t],
            )
            atten_mask_l = torch.tril(
                torch.ones(shape),
                diagonal=-pre_tocken - 1 + seqlens_list_k[t] - seqlens_list_q[t],
            )
            atten_mask = (atten_mask_u + atten_mask_l).to(mask_dtype)
        else:
            atten_mask = (torch.triu(torch.ones(shape), diagonal=1)).to(mask_dtype)
    return atten_mask


def get_atten_mask_tnd(
    sparse_mode,
    atten_mask_shape,
    atten_mask_dtype,
    B,
    N1,
    N2,
    seqlens_list_q,
    seqlens_list_k,
    D,
    case_pre_tockens,
    case_next_tockens,
    dtype,
):
    S1 = seqlens_list_q.max()
    S2 = seqlens_list_k.max()
    G = int(N1 / N2)
    prefix = []
    pre_tocken = case_pre_tockens
    next_tocken = case_next_tockens
    if atten_mask_shape == "SS":
        shape = [S1, S2]
    elif atten_mask_shape == "11SS":
        shape = [1, 1, S1, S2]
    elif atten_mask_shape == "B1SS":
        shape = [1, 1, S1, S2]
    elif atten_mask_shape == "BNSS":
        shape = [1, N1, S1, S2]
    # TND场景下，即使外部传入attenMask为空，也要
    elif atten_mask_dtype.lower() == "none" or atten_mask_shape.lower() == "none":
        atten_mask = torch.tensor(0)
        prefix = None
        pre_tocken = 65536
        next_tocken = 65536
        atten_mask_npu = atten_mask
        return atten_mask, atten_mask_npu, prefix, pre_tocken, next_tocken
        # shape = [1, N1, S1, S2]
    else:
        raise RuntimeError(
            f"not support shape of atten_mask {atten_mask_shape}, only support [S1,S2], [B, 1, S1, S2], [B, N1, S1, S2] or None"
        )

    if sparse_mode == 0:
        pre_tocken = case_pre_tockens
        next_tocken = case_next_tockens
        atten_mask_u = torch.triu(torch.ones(shape), diagonal=next_tocken + 1)
        atten_mask_l = torch.tril(torch.ones(shape), diagonal=-pre_tocken - 1)
        atten_mask = (atten_mask_u + atten_mask_l).to(dtype)
        atten_mask_npu = atten_mask

    elif sparse_mode == 1:  # no sparse
        atten_mask = (torch.zeros(shape)).to(dtype)
        pre_tocken = S1
        next_tocken = S2
        atten_mask_npu = atten_mask
    elif sparse_mode == 2:  # 用于生成golden数据,上三角
        atten_mask = (torch.triu(torch.ones(shape), diagonal=1)).to(dtype)
        pre_tocken = S1
        next_tocken = 0
        atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(dtype)
    elif sparse_mode == 3:  # 下三角
        atten_mask = (torch.triu(torch.ones(shape), diagonal=S2 - S1 + 1)).to(dtype)
        pre_tocken = S1
        next_tocken = 0
        atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(dtype)
    elif sparse_mode == 4:  # band
        opname = "sparse_band_flash_attention"
        pre_tocken = case_pre_tockens
        next_tocken = case_next_tockens
        atten_mask_u = torch.triu(torch.ones(shape), diagonal=next_tocken + 1 + S2 - S1)
        atten_mask_l = torch.tril(torch.ones(shape), diagonal=-pre_tocken - 1 + S2 - S1)
        atten_mask = (atten_mask_u + atten_mask_l).to(dtype)
        atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(dtype)

    elif sparse_mode == 5:  # prefix,shape只能是BNSS或者B1SS
        atten_mask = (torch.triu(torch.ones(shape), diagonal=S2 - S1 + 1)).to(dtype)
        prefix = get_prefix_tnd(
            seqlens_list_q=seqlens_list_q, seqlens_list_k=seqlens_list_k
        )
        for i in range(0, B):
            if len(shape) > 2:
                atten_mask[:, :, :, 0 : prefix[i]] = 0
            else:
                atten_mask[:, 0 : prefix[i]] = 0
        atten_mask_npu = atten_mask

    elif sparse_mode == 6:
        atten_mask = torch.triu(torch.ones(shape), diagonal=S2 - S1 + 1)
        prefix = get_prefix_tnd(
            seqlens_list_q=seqlens_list_q, seqlens_list_k=seqlens_list_k
        )
        upper = torch.triu(torch.ones(2048, 2048), diagonal=1)
        lower = torch.cat((torch.zeros(1024, 1024), torch.ones(1024, 1024)), dim=1)
        atten_mask_npu = torch.cat((upper, lower), dim=0).to(dtype)
    elif sparse_mode == 7 or sparse_mode == 8:
        atten_mask = torch.tensor(0)
        atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(dtype)
    return atten_mask, atten_mask_npu, prefix, pre_tocken, next_tocken


def get_slopes(n_heads: int):
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m


def get_all_alibi(
    B, Nq, Sq, Skv, pse_layout, pse_dtype, pse_mode=1, mask_value=-np.inf
):
    m = get_slopes(Nq)
    if pse_mode in (2, 3):
        if pse_layout.lower() == "bnhs":
            npu_pse = (
                1 / (torch.arange(B) + 1).unsqueeze(1) * 1 / (torch.arange(Nq) + 1)
            )
            bn11 = npu_pse.reshape(B, Nq, 1, 1)

        elif pse_layout.lower() == "1nhs":
            npu_pse = 1 / (torch.arange(Nq) + 1)
            bn11 = npu_pse.reshape(1, Nq, 1, 1)

        elif pse_layout.lower() == "1nss":
            npu_pse = 1 / (torch.arange(Nq) + 1)
            bn11 = npu_pse.reshape(1, Nq, 1, 1)

        elif pse_layout.lower() == "bnss":
            npu_pse = (
                1 / (torch.arange(B) + 1).unsqueeze(1) * 1 / (torch.arange(Nq) + 1)
            )
            bn11 = npu_pse.reshape(B, Nq, 1, 1)

        elif pse_layout.lower() == "n":
            npu_pse = 1 / (torch.arange(Nq) + 1)
            bn11 = npu_pse.reshape(1, Nq, 1, 1)
        elif pse_layout.lower() == "bn":
            npu_pse = (
                1 / (torch.arange(B) + 1).unsqueeze(1) * 1 / (torch.arange(Nq) + 1)
            )
            bn11 = npu_pse.reshape(B, Nq, 1, 1)
        else:
            print_log(f"[get_all_alibi] type:{pse_mode} lay_out error:{pse_layout}")
            return None, None
        if pse_layout.lower() == "n" or pse_layout.lower() == "bn":
            # SS，0线为右下对角线，对称排布与GPU一致
            # tensor([[-5., -4., -3., -2., -1.,  0., -1., -2., -3., -4.],
            #         [-6., -5., -4., -3., -2., -1.,  0., -1., -2., -3.],
            #         [-7., -6., -5., -4., -3., -2., -1.,  0., -1., -2.],
            #         [-8., -7., -6., -5., -4., -3., -2., -1.,  0., -1.],
            #         [-9., -8., -7., -6., -5., -4., -3., -2., -1.,  0.]])
            if pse_mode != 3:
                alibi_base = -torch.abs(
                    torch.arange(Sq).unsqueeze(1) - torch.arange(Skv).unsqueeze(0)
                )
                # alibi_base = -torch.abs(torch.arange(Sq).unsqueeze(1) - torch.arange(Skv).unsqueeze(0) - (Sq - Skv))
            else:
                alibi_base = -(
                    torch.abs(
                        torch.arange(Sq).unsqueeze(1) - torch.arange(Skv).unsqueeze(0)
                    ).sqrt()
                )
                # torch.abs(torch.arange(Sq).unsqueeze(1) - torch.arange(Skv).unsqueeze(0) - (Sq - Skv)).sqrt())
            print(alibi_base)
            pse_all = bn11 * alibi_base  # 1NSS or BNSS
        else:
            alibi_base = -torch.abs(
                torch.arange(Sq).unsqueeze(1)
                - torch.arange(Skv).unsqueeze(0)
                - (Sq - Skv)
            )
            pse_all = bn11 * alibi_base  # 1NSS or BNSS

        if pse_layout.lower() in ["bnhs", "1nhs"]:
            npu_pse = pse_all[:, :, -1024:, :]
            return pse_all.to(pse_dtype), npu_pse.to(pse_dtype)
        elif pse_layout.lower() in ["n", "bn"]:
            print(f"pse_dtype:{pse_all.dtype},{npu_pse.dtype}")
            return pse_all.to(pse_dtype), npu_pse
        else:
            return pse_all.to(pse_dtype), None
    else:
        print_log("[get_all_alibi] only support type 2 and 3")
        raise RuntimeError(
            f"[get_all_alibi] type:{pse_mode} lay_out error:{pse_layout}"
        )


def tforward(
    case,
    q,
    k,
    v,
    drop_mask,
    atten_mask,
    pse,
    scale,
    keep_prob,
    is_skip_invalid_row,
    pse_type,
    scale_q,
    scale_k,
    scale_v,
):
    if case["dtype"] == "fp16" or case["dtype"] == "bf16" or case["dtype"] == "fp32":
        qk = None
        if pse_type == 1:
            qk = torch.matmul(q, k.permute(0, 1, 2, 4, 3)).add(pse).mul(scale)
        else:
            qk = torch.matmul(q, k.permute(0, 1, 2, 4, 3)).mul(scale).add(pse)
        if atten_mask is not None:
            # qk = qk + atten_mask * (-40000.0)
            qk = qk.masked_fill(atten_mask.bool(), value=torch.tensor(-40000.0))
        print("qk shape:", qk.shape)
        print("q shape:", q.shape)
        print("k shape:", k.shape)
        if case["S2"] == 0 and case["N1"] != 0:
            y = torch.from_numpy(
                np.zeros(
                    [
                        case["B"],
                        case["N2"],
                        case["N1"] // case["N1"],
                        case["S1"],
                        case["D_V"],
                    ]
                )
            )
            softmax_res = torch.from_numpy(
                np.zeros(
                    [case["B"], case["N2"], case["N1"] // case["N1"], case["S1"], 1]
                )
            )
            x_max = torch.from_numpy(
                np.zeros(
                    [case["B"], case["N2"], case["N1"] // case["N2"], case["S1"], 1]
                )
            )
            x_sum = torch.from_numpy(
                np.zeros(
                    [case["B"], case["N2"], case["N1"] // case["N2"], case["S1"], 1]
                )
            )
            print("out shape:", y.shape)
            return y, softmax_res, x_max, x_sum
        if case["input_layout"] == "TND":
            for i in range(case["B"]):
                qk[i, :, :, :, case["actual_seq_kvlen"][i] :] = -40000.0
        softmax_res, x_max, x_sum = tsoftmax(qk)

        if case["input_layout"] == "TND":
            for i in range(case["B"]):
                softmax_res[i, :, :, case["actual_seq_qlen"][i] :, :] = 0

        if is_skip_invalid_row:
            print("tforward need skip invalid row")
            b, n2, g, s1, s2 = softmax_res.shape
            softmax_res, x_max, x_sum = skip_zero_row(
                b, n2, g, s1, x_max, x_sum, softmax_res
            )
            softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0

        drop_res = softmax_res * drop_mask * (1.0 / (keep_prob))
        y = torch.matmul(drop_res, v)
    elif case["dtype"] == "fp8_e4m3fn" or case["dtype"] == "fp8_e5m2":
        dtype = "float8_e5m2" if case["dtype"] == "fp8_e5m2" else "float8_e4m3fn"
        qk = None
        dscale_q = 1 / scale_q
        dscale_k = 1 / scale_k
        dscale_v = 1 / scale_v
        q = block_quant_or_dequant_single(q, scale_q)
        k = block_quant_or_dequant_single(k, scale_k)
        v = block_quant_or_dequant_single(v, scale_v)
        q = torch.from_numpy(q.cpu().numpy().astype(dtype).astype(np.float32))
        k = torch.from_numpy(k.cpu().numpy().astype(dtype).astype(np.float32))
        v = torch.from_numpy(v.cpu().numpy().astype(dtype).astype(np.float32))
        # 2 - matmul
        qkk = torch.matmul(q, k.permute(0, 1, 2, 4, 3))
        # 3 - dequant
        qk = block_dequant(qkk, dscale_q, dscale_k)
        if pse_type == 1:
            qk = qk.add(pse).mul(scale)
        else:
            qk = qk.mul(scale).add(pse)
        if atten_mask is not None:
            # qk = qk + atten_mask * (-40000.0)
            qk = qk.masked_fill(atten_mask.bool(), value=torch.tensor(-40000.0))
        # qk = qkk * dscale_q * dscale_k
        # 4 - softmax
        softmax_res, x_max, x_sum = tsoftmax(qk)

        if is_skip_invalid_row:
            print("tforward need skip invalid row")
            b, n2, g, s1, s2 = softmax_res.shape
            softmax_res, x_max, x_sum = skip_zero_row(
                b, n2, g, s1, x_max, x_sum, softmax_res
            )
            softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0

        # 5 - drop_res
        drop_res = softmax_res * drop_mask * (1.0 / (keep_prob))

        drop_res = torch.from_numpy(
            drop_res.cpu().numpy().astype(dtype).astype(np.float32)
        )  # cast to fp8
        y = chunked_matmul_with_quant(drop_res, v, dscale_v)
    return y, softmax_res, x_max, x_sum


def tbackward(
    case,
    dx,
    q,
    k,
    v,
    softmax_res,
    y,
    drop_mask,
    pse,
    scale,
    keep_prob,
    scale_q,
    scale_k,
    scale_v,
    scale_dx,
):
    is_flash = True
    if case["dtype"] == "fp16" or case["dtype"] == "bf16" or case["dtype"] == "fp32":
        drop_res = softmax_res.mul(drop_mask).mul(1.0 / (keep_prob))
        dv = torch.matmul(drop_res.permute(0, 1, 2, 4, 3), dx)
        dp = torch.matmul(dx, v.permute(0, 1, 2, 4, 3))
        dp_drop = dp * drop_mask * (1.0 / (keep_prob))
        softmax_grad_res = tsoftmax_grad(dp_drop, softmax_res) * scale
        dq = torch.matmul(softmax_grad_res, k)
        dk = torch.matmul(softmax_grad_res.permute(0, 1, 2, 4, 3), q)
        dk = torch.sum(dk, dim=2, keepdim=True)
        dv = torch.sum(dv, dim=2, keepdim=True)
    elif case["dtype"] == "fp8_e4m3fn" or case["dtype"] == "fp8_e5m2":
        dtype = "float8_e5m2" if case["dtype"] == "fp8_e5m2" else "float8_e4m3fn"
        # 1 - get dscale
        dscale_q = 1 / scale_q
        dscale_k = 1 / scale_k
        dscale_v = 1 / scale_v
        dscale_dx = 1 / scale_dx
        # 2 - quant
        q = block_quant_or_dequant_single(q, scale_q)
        k = block_quant_or_dequant_single(k, scale_k)
        v = block_quant_or_dequant_single(v, scale_v)
        dx = block_quant_or_dequant_single(dx, scale_dx)
        q = torch.from_numpy(q.cpu().numpy().astype(dtype).astype(np.float32))
        k = torch.from_numpy(k.cpu().numpy().astype(dtype).astype(np.float32))
        v = torch.from_numpy(v.cpu().numpy().astype(dtype).astype(np.float32))
        dx = torch.from_numpy(dx.cpu().numpy().astype(dtype).astype(np.float32))
        # drop_res
        drop_res = softmax_res.mul(drop_mask).mul(1.0 / (keep_prob))
        drop_res = torch.from_numpy(
            drop_res.cpu().numpy().astype(dtype).astype(np.float32)
        )  # cast to fp8
        dv = chunked_matmul_with_quant(drop_res.permute(0, 1, 2, 4, 3), dx, dscale_dx)
        # dp
        dpp = torch.matmul(dx, v.permute(0, 1, 2, 4, 3))
        dp = block_dequant(dpp, dscale_dx, dscale_v)
        dp_drop = dp * drop_mask * (1.0 / (keep_prob))
        if not is_flash:
            # softmax
            softmax_grad_res = tsoftmax_grad(dp_drop, softmax_res) * scale
            # dq
            dqq = torch.matmul(softmax_grad_res, k)
            dq = block_quant_or_dequant_single(dqq, dscale_k)
            # dk
            dkk = torch.matmul(softmax_grad_res.permute(0, 1, 2, 4, 3), q)
            dk = block_quant_or_dequant_single(dkk, dscale_q)
        else:
            # softmax grad
            dx_dequanted = block_quant_or_dequant_single(dx, dscale_dx)
            softmax_grad_res = (y * dx_dequanted).sum(dim=-1, keepdim=True)
            softmax_grad_res = (dp_drop - softmax_grad_res) * softmax_res
            # 模拟V核内quant与deQuant
            q_scale_ds = get_block_quant_scale(
                softmax_grad_res, case["dtype"], True, False
            )
            # quant
            softmax_grad_res_quanted = block_quant_or_dequant_single(
                softmax_grad_res, q_scale_ds, True
            )
            # cast to fp8 and cast to fp32
            softmax_grad_res_fp8 = torch.from_numpy(
                softmax_grad_res_quanted.cpu().numpy().astype(dtype).astype(np.float32)
            )
            # dq
            mask = q_scale_ds != 0
            deq_scale_ds = np.zeros_like(q_scale_ds)
            deq_scale_ds[mask] = 1 / q_scale_ds[mask]
            dq = (
                chunked_matmul_with_dq_quant(
                    softmax_grad_res_fp8, k, deq_scale_ds, dscale_k
                )
                * scale
            )
            # dk
            dk = (
                chunked_matmul_with_dk_quant(
                    softmax_grad_res_fp8.permute(0, 1, 2, 4, 3),
                    q,
                    deq_scale_ds,
                    dscale_q,
                )
                * scale
            )
        dk = torch.sum(dk, dim=2, keepdim=True)
        dv = torch.sum(dv, dim=2, keepdim=True)

    return dq, dk, dv


def block_quant_or_dequant_single(tensor, scale, is_scale_ds=False):
    dim1 = tensor.shape[0]  # b
    dim2 = tensor.shape[1]  # n2
    dim3 = tensor.shape[2]  # g
    dim4 = tensor.shape[3]  # s1
    dim5 = tensor.shape[4]  # d or s2
    quanted_tensor = torch.zeros([dim1, dim2, dim3, dim4, dim5])
    if not is_scale_ds:
        for b in range(dim1):
            for n in range(dim2):
                for g in range(dim3):
                    for s in range(0, dim4, PER_BLOCK_SIZE):
                        # 取出128*D个数
                        s_start = s // PER_BLOCK_SIZE * PER_BLOCK_SIZE
                        s_end = min(
                            s // PER_BLOCK_SIZE * PER_BLOCK_SIZE + PER_BLOCK_SIZE, dim4
                        )
                        scale_g_idx = 0 if scale.shape[2] == 1 else g
                        quanted_tensor[b, n, g, s_start:s_end, :] = (
                            tensor[b, n, g, s_start:s_end, :]
                            * scale[b, n, scale_g_idx, s // PER_BLOCK_SIZE, 0]
                        )
    else:
        s1_tail = (
            PER_BLOCK_SIZE if dim4 % PER_BLOCK_SIZE == 0 else dim4 % PER_BLOCK_SIZE
        )
        for b in range(dim1):
            for n in range(dim2):
                for g in range(dim3):
                    for s2 in range(0, dim5, PER_BLOCK_SIZE):
                        s2_start = s2 // PER_BLOCK_SIZE * PER_BLOCK_SIZE
                        s2_end = min(s2_start + PER_BLOCK_SIZE, dim5)
                        scale_g_idx = 0 if scale.shape[2] == 1 else g
                        for s1 in range(0, dim4 - s1_tail, PER_BLOCK_SIZE):
                            # 取出64*128个数
                            s1_start = s1 // PER_BLOCK_SIZE * PER_BLOCK_SIZE
                            s1_end = min(s1_start + PER_BLOCK_SIZE, dim4)
                            quanted_tensor[
                                b, n, g, s1_start:s1_end, s2_start:s2_end
                            ] = (
                                tensor[b, n, g, s1_start:s1_end, s2_start:s2_end]
                                * scale[
                                    b,
                                    n,
                                    scale_g_idx,
                                    s1 // PER_BLOCK_SIZE,
                                    s2 // PER_BLOCK_SIZE,
                                ]
                            )
                        # 处理尾块
                        s1_start = dim4 - s1_tail
                        s1_end = dim4
                        quanted_tensor[b, n, g, s1_start:s1_end, s2_start:s2_end] = (
                            tensor[b, n, g, s1_start:s1_end, s2_start:s2_end]
                            * scale[
                                b,
                                n,
                                scale_g_idx,
                                s1_start // PER_BLOCK_SIZE,
                                s2 // PER_BLOCK_SIZE,
                            ]
                        )
    return quanted_tensor


def get_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int64)
    for i in range(len(seqlens_list) + 1):
        cu[i] = sum(seqlens_list[:i])
    return cu


def broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor, dtype):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = torch.zeros([B, numHeads, S, D]).to(dtype)
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i : i + 1, :, :] = kv_tensor[:, j : j + 1, :, :]
    return kv_res


def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    softmax_res = y.div(x_sum)

    return softmax_res, x_max, x_sum


def tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdims=True)
    sub_r = dp - muls_r
    res = sub_r * softmax_res
    return res


def block_dequant(res, dequant_matrix_s1, dequant_matrix_s2):
    # 获取输入矩阵 res 的形状，分别为 B、N、S1、S2 四个维度的大小
    B, Nkv, G, S1, S2 = res.shape
    # 定义块大小为 128
    block_size = 128
    # 初始化结果矩阵，形状与输入矩阵 res 相同，元素全为 0
    result = np.zeros_like(res)
    # 遍历B维度
    for b in range(B):
        # 遍历N维度
        for n in range(Nkv):
            # 遍历G维度
            for g in range(G):
                # 遍历S1方向的块
                for i in range((S1 + block_size - 1) // block_size):
                    # 计算当前块在S1维度上的起始和结束索引，防止超出S1的范围
                    start_s1 = i * block_size
                    end_s1 = min(start_s1 + block_size, S1)
                    # 遍历S2方向的块
                    for j in range((S2 + block_size - 1) // block_size):
                        # 计算当前块在S2维度上的起始和结束索引，防止超出S2的范围
                        start_s2 = j * block_size
                        end_s2 = min(start_s2 + block_size, S2)
                        # 从输入矩阵res中提取当前数据块
                        block = res[b, n, g, start_s1:end_s1, start_s2:end_s2]
                        # 获取量化值&反量化运算
                        s1_g_idx = 0 if dequant_matrix_s1.shape[2] == 1 else g
                        s2_g_idx = 0 if dequant_matrix_s2.shape[2] == 1 else g
                        dequant_value_s1 = dequant_matrix_s1[b, n, s1_g_idx, i, 0]
                        dequant_value_s2 = dequant_matrix_s2[b, n, s2_g_idx, j, 0]
                        dequant_kv = dequant_value_s1 * dequant_value_s2
                        weighted_block = block * dequant_kv
                        # 将加权后的块放回结果矩阵
                        result[b, n, g, start_s1:end_s1, start_s2:end_s2] = (
                            weighted_block
                        )
    return torch.from_numpy(result)


def chunked_matmul_with_quant(
    left: torch.Tensor, right: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """
    分块矩阵乘法实现，支持reduce轴未对齐场景

    Args:
        left:  输入左矩阵 [B, N2, G, S1, S2]
        right: 输入右矩阵 [B, N2, G, S2, D]
        chunk_size: 分块大小，默认128

    Returns:
        乘积结果 [B, N2, G, S1, D]
    """
    B, N2, G1, S1, S2 = left.shape
    B_r, N2_r, G2, S2_r, D = right.shape
    assert B == B_r and N2 == N2_r and S2 == S2_r, "维度不匹配"
    assert G1 == G2 or G1 == 1 or G2 == 1, f"G轴无法广播: G1={G1}, G2={G2}"

    G_out = max(G1, G2)
    result = torch.zeros((B, N2, G_out, S1, D), dtype=left.dtype, device=left.device)

    for b in range(B):
        for n in range(N2):
            for g in range(G_out):
                g_left = 0 if G1 == 1 else g
                g_right = 0 if G2 == 1 else g

                for start in range(0, S2, PER_BLOCK_SIZE):
                    end = start + PER_BLOCK_SIZE
                    end = min(end, S2)
                    current_chunk = min(PER_BLOCK_SIZE, S2 - start)

                    left_block = left[b, n, g_left, :, start:end]  # [S1, chunk]
                    right_block = right[b, n, g_right, start:end, :]  # [chunk, D]

                    scale_g_idx = 0 if scale.shape[2] == 1 else g
                    result[b, n, g] += (
                        torch.matmul(left_block, right_block)
                        * scale[b, n, scale_g_idx, start // PER_BLOCK_SIZE, 0]
                    )
    return result


def chunked_matmul_with_dk_quant(
    left: torch.Tensor,
    right: torch.Tensor,
    scale_left: torch.Tensor,
    scale_right=torch.Tensor,
    is_vcore_split=False,
) -> torch.Tensor:
    # left： S2*S1; Right: S1*D
    B, N2, G1, S2, S1 = left.shape
    B_r, N2_r, G2, S1_r, D = right.shape
    assert B == B_r and N2 == N2_r and S1 == S1_r, "维度不匹配"
    assert G1 == G2 or G1 == 1 or G2 == 1, f"G轴无法广播: G1={G1}, G2={G2}"

    G_out = max(G1, G2)
    result = torch.zeros((B, N2, G_out, S2, D), dtype=left.dtype, device=left.device)
    s1_tail = PER_BLOCK_SIZE if S1 % PER_BLOCK_SIZE == 0 else S1 % PER_BLOCK_SIZE
    if is_vcore_split:
        for b in range(B):
            for n in range(N2):
                for g in range(G_out):
                    g_left = 0 if G1 == 1 else g
                    g_right = 0 if G2 == 1 else g
                    for s2 in range(0, S2, PER_BLOCK_SIZE):
                        s2_start = s2
                        s2_end = s2_start + PER_BLOCK_SIZE
                        s2_end = min(s2_end, S2)
                        scale_g_idx = 0 if scale_left.shape[2] == 1 else g
                        for s1 in range(0, S1 - s1_tail, PER_VCORE_BLOCK_SIZE):
                            s1_start = s1
                            s1_end = s1_start + PER_VCORE_BLOCK_SIZE
                            s1_end = min(s1_end, S1)

                            left_block = left[
                                b, n, g_left, s2_start:s2_end, s1_start:s1_end
                            ]
                            scale_left_value = scale_left[
                                b,
                                n,
                                scale_g_idx,
                                s1 // PER_VCORE_BLOCK_SIZE,
                                s2 // PER_BLOCK_SIZE,
                            ]
                            right_block = right[b, n, g_right, s1_start:s1_end, :]
                            scale_right_value = scale_right[
                                b, n, scale_g_idx, s1_start // PER_BLOCK_SIZE, 0
                            ]
                            result[b, n, g, s2_start:s2_end] += (
                                torch.matmul(left_block, right_block)
                                * scale_left_value
                                * scale_right_value
                            )
                        # 处理尾块 v0
                        s1_start = S1 - s1_tail
                        s1_end = s1_start + math.ceil(s1_tail / 2)
                        left_block = left[
                            b, n, g_left, s2_start:s2_end, s1_start:s1_end
                        ]
                        scale_left_value = scale_left[
                            b,
                            n,
                            scale_g_idx,
                            s1_start // PER_VCORE_BLOCK_SIZE,
                            s2 // PER_BLOCK_SIZE,
                        ]
                        right_block = right[b, n, g_right, s1_start:s1_end, :]
                        scale_right_value = scale_right[
                            b, n, scale_g_idx, s1_start // PER_BLOCK_SIZE, 0
                        ]
                        result[b, n, g, s2_start:s2_end] += (
                            torch.matmul(left_block, right_block)
                            * scale_left_value
                            * scale_right_value
                        )
                        v0_s1_start_idx = s1_start // PER_VCORE_BLOCK_SIZE
                        # 处理尾块 v1
                        s1_start = s1_end
                        s1_end = S1
                        left_block = left[
                            b, n, g_left, s2_start:s2_end, s1_start:s1_end
                        ]
                        scale_left_value = scale_left[
                            b, n, scale_g_idx, v0_s1_start_idx + 1, s2 // PER_BLOCK_SIZE
                        ]
                        right_block = right[b, n, g_right, s1_start:s1_end, :]
                        scale_right_value = scale_right[
                            b, n, scale_g_idx, s1_start // PER_BLOCK_SIZE, 0
                        ]
                        result[b, n, g, s2_start:s2_end] += (
                            torch.matmul(left_block, right_block)
                            * scale_left_value
                            * scale_right_value
                        )
    else:
        for b in range(B):
            for n in range(N2):
                for g in range(G_out):
                    g_left = 0 if G1 == 1 else g
                    g_right = 0 if G2 == 1 else g
                    for s2 in range(0, S2, PER_BLOCK_SIZE):
                        s2_start = s2
                        s2_end = s2_start + PER_BLOCK_SIZE
                        s2_end = min(s2_end, S2)
                        scale_g_idx = 0 if scale_left.shape[2] == 1 else g
                        for s1 in range(0, S1 - s1_tail, PER_BLOCK_SIZE):
                            s1_start = s1
                            s1_end = s1_start + PER_BLOCK_SIZE
                            s1_end = min(s1_end, S1)

                            left_block = left[
                                b, n, g_left, s2_start:s2_end, s1_start:s1_end
                            ]
                            scale_left_value = scale_left[
                                b,
                                n,
                                scale_g_idx,
                                s1 // PER_BLOCK_SIZE,
                                s2 // PER_BLOCK_SIZE,
                            ]
                            right_block = right[b, n, g_right, s1_start:s1_end, :]
                            scale_right_value = scale_right[
                                b, n, scale_g_idx, s1_start // PER_BLOCK_SIZE, 0
                            ]
                            result[b, n, g, s2_start:s2_end] += (
                                torch.matmul(left_block, right_block)
                                * scale_left_value
                                * scale_right_value
                            )
                        # 处理尾块 v0
                        s1_start = S1 - s1_tail
                        s1_end = S1
                        left_block = left[
                            b, n, g_left, s2_start:s2_end, s1_start:s1_end
                        ]
                        scale_left_value = scale_left[
                            b,
                            n,
                            scale_g_idx,
                            s1_start // PER_BLOCK_SIZE,
                            s2 // PER_BLOCK_SIZE,
                        ]
                        right_block = right[b, n, g_right, s1_start:s1_end, :]
                        scale_right_value = scale_right[
                            b, n, scale_g_idx, s1_start // PER_BLOCK_SIZE, 0
                        ]
                        result[b, n, g, s2_start:s2_end] += (
                            torch.matmul(left_block, right_block)
                            * scale_left_value
                            * scale_right_value
                        )
    return result


def chunked_matmul_with_dq_quant(
    left: torch.Tensor,
    right: torch.Tensor,
    scale_left: torch.Tensor,
    scale_right=torch.Tensor,
    is_vcore_split=False,
) -> torch.Tensor:
    B, N2, G1, S1, S2 = left.shape
    B_r, N2_r, G2, S2_r, D = right.shape
    assert B == B_r and N2 == N2_r and S2 == S2_r, "维度不匹配"
    assert G1 == G2 or G1 == 1 or G2 == 1, f"G轴无法广播: G1={G1}, G2={G2}"

    G_out = max(G1, G2)
    result = torch.zeros((B, N2, G_out, S1, D), dtype=left.dtype, device=left.device)
    s1_tail = PER_BLOCK_SIZE if S1 % PER_BLOCK_SIZE == 0 else S1 % PER_BLOCK_SIZE
    if is_vcore_split:
        for b in range(B):
            for n in range(N2):
                for g in range(G_out):
                    g_left = 0 if G1 == 1 else g
                    g_right = 0 if G2 == 1 else g
                    for s2 in range(0, S2, PER_BLOCK_SIZE):
                        s2_start = s2
                        s2_end = s2_start + PER_BLOCK_SIZE
                        s2_end = min(s2_end, S2)
                        scale_g_idx = 0 if scale_left.shape[2] == 1 else g
                        scale_right_value = scale_right[
                            b, n, 0, s2_start // PER_BLOCK_SIZE, 0
                        ]
                        right_block = right[b, n, g_right, s2_start:s2_end, :]
                        for s1 in range(0, S1 - s1_tail, PER_VCORE_BLOCK_SIZE):
                            s1_start = s1
                            s1_end = s1_start + PER_VCORE_BLOCK_SIZE
                            s1_end = min(s1_end, S1)
                            left_block = left[
                                b, n, g_left, s1_start:s1_end, s2_start:s2_end
                            ]
                            scale_left_value = scale_left[
                                b,
                                n,
                                scale_g_idx,
                                s1 // PER_VCORE_BLOCK_SIZE,
                                s2 // PER_BLOCK_SIZE,
                            ]
                            result[b, n, g, s1_start:s1_end, :] += (
                                torch.matmul(left_block, right_block)
                                * scale_left_value
                                * scale_right_value
                            )
                        # 处理尾块 v0
                        s1_start = S1 - s1_tail
                        s1_end = s1_start + math.ceil(s1_tail / 2)
                        left_block = left[
                            b, n, g_left, s1_start:s1_end, s2_start:s2_end
                        ]
                        scale_left_value = scale_left[
                            b,
                            n,
                            scale_g_idx,
                            s1_start // PER_VCORE_BLOCK_SIZE,
                            s2 // PER_BLOCK_SIZE,
                        ]
                        result[b, n, g, s1_start:s1_end, :] += (
                            torch.matmul(left_block, right_block)
                            * scale_left_value
                            * scale_right_value
                        )
                        v0_s1_start_idx = s1_start // PER_VCORE_BLOCK_SIZE
                        # 处理尾块 v1
                        s1_start = s1_end
                        s1_end = S1
                        left_block = left[
                            b, n, g_left, s1_start:s1_end, s2_start:s2_end
                        ]
                        scale_left_value = scale_left[
                            b, n, scale_g_idx, v0_s1_start_idx + 1, s2 // PER_BLOCK_SIZE
                        ]
                        result[b, n, g, s1_start:s1_end, :] += (
                            torch.matmul(left_block, right_block)
                            * scale_left_value
                            * scale_right_value
                        )
    else:
        for b in range(B):
            for n in range(N2):
                for g in range(G_out):
                    g_left = 0 if G1 == 1 else g
                    g_right = 0 if G2 == 1 else g
                    for s2 in range(0, S2, PER_BLOCK_SIZE):
                        s2_start = s2
                        s2_end = s2_start + PER_BLOCK_SIZE
                        s2_end = min(s2_end, S2)
                        scale_g_idx = 0 if scale_left.shape[2] == 1 else g
                        scale_right_value = scale_right[
                            b, n, 0, s2_start // PER_BLOCK_SIZE, 0
                        ]
                        right_block = right[b, n, g_right, s2_start:s2_end, :]
                        for s1 in range(0, S1 - s1_tail, PER_BLOCK_SIZE):
                            s1_start = s1
                            s1_end = s1_start + PER_BLOCK_SIZE
                            s1_end = min(s1_end, S1)
                            left_block = left[
                                b, n, g_left, s1_start:s1_end, s2_start:s2_end
                            ]
                            scale_left_value = scale_left[
                                b,
                                n,
                                scale_g_idx,
                                s1 // PER_BLOCK_SIZE,
                                s2 // PER_BLOCK_SIZE,
                            ]
                            result[b, n, g, s1_start:s1_end, :] += (
                                torch.matmul(left_block, right_block)
                                * scale_left_value
                                * scale_right_value
                            )
                        # 处理尾块
                        s1_start = S1 - s1_tail
                        s1_end = S1
                        left_block = left[
                            b, n, g_left, s1_start:s1_end, s2_start:s2_end
                        ]
                        scale_left_value = scale_left[
                            b,
                            n,
                            scale_g_idx,
                            s1_start // PER_BLOCK_SIZE,
                            s2 // PER_BLOCK_SIZE,
                        ]
                        result[b, n, g, s1_start:s1_end, :] += (
                            torch.matmul(left_block, right_block)
                            * scale_left_value
                            * scale_right_value
                        )
    return result


def get_atten_mask(
    sparse_mode, atten_mask_shape, B, N2, G, S1, S2, pre_tocken, next_tocken
):
    cpu_shape = [B, N2, G, S1, S2]
    if atten_mask_shape == "SS" or atten_mask_shape == "11SS":
        mask_shape = [S1, S2]
    elif atten_mask_shape == "B1SS":
        mask_shape = [B, 1, 1, S1, S2]
    elif atten_mask_shape == "BNSS":
        mask_shape = [B, N2, G, S1, S2]
    elif atten_mask_shape == "01SS":
        mask_shape = [0, 1, 1, S1, S2]
    elif atten_mask_shape == "BN0S":
        mask_shape = [B, N2, G, 0, S2]
    elif atten_mask_shape == "BNS0":
        mask_shape = [B, N2, G, S1, 0]
    else:
        raise RuntimeError(
            f"not support shape of atten_mask {atten_mask_shape}, only support SS or B1SS or BNSS."
        )

    prefix = None
    if sparse_mode == 0:
        atten_mask_u = np.triu(np.ones(mask_shape), k=next_tocken + 1)
        atten_mask_l = np.tril(np.ones(mask_shape), k=-pre_tocken - 1)
        atten_masks = atten_mask_u + atten_mask_l
        # atten_masks = np.zeros(mask_shape).astype(np.bool_)
        # for i in range(S2):
        #     atten_masks[3][i] = True
        atten_mask_npu = atten_masks
    elif sparse_mode == 1:
        atten_masks = np.random.randint(0, 2, size=mask_shape).astype(np.bool_)
        atten_mask_npu = atten_masks
    elif sparse_mode == 2:
        atten_masks = np.triu(np.ones(mask_shape), k=1)
        atten_mask_npu = np.triu(np.ones([2048, 2048]), k=1).astype(np.bool_)
    elif sparse_mode == 3:
        atten_masks = np.triu(np.ones(mask_shape), k=S2 - S1 + 1)
        atten_mask_npu = np.triu(np.ones([2048, 2048]), k=1).astype(np.bool_)
    elif sparse_mode == 4:
        atten_mask_u = np.triu(np.ones(mask_shape), k=next_tocken + S2 - S1 + 1)
        atten_mask_l = np.tril(np.ones(mask_shape), k=-pre_tocken + S2 - S1 - 1)
        atten_masks = atten_mask_u + atten_mask_l
        atten_mask_npu = np.triu(np.ones([2048, 2048]), k=1).astype(np.bool_)
    elif sparse_mode == 5:
        if atten_mask_shape == "SS":
            raise RuntimeError(
                f"prefix not support shape of atten_mask {atten_mask_shape}, only support BNSS and B1SS"
            )
        atten_masks = np.triu(np.ones(mask_shape), k=S2 - S1 + 1)
        prefix = get_prefix(B, S1, S2)
        for i in range(0, B):
            atten_masks[i, :, :, :, 0 : prefix[i]] = 0
        atten_mask_npu = atten_masks
    elif sparse_mode == 6:
        # not check atten mask shape
        atten_masks = np.triu(np.ones(cpu_shape), k=S2 - S1 + 1)
        prefix = get_prefix(B, S1, S2)
        for i in range(0, B):
            atten_masks[i, :, :, :, 0 : prefix[i]] = 0
        upper = np.triu(np.ones([2048, 2048]), k=1)
        lower = np.concatenate(
            (np.zeros([1024, 1024]), np.ones([1024, 1024])), axis=1
        ).astype(np.bool_)
        atten_mask_npu = np.concatenate((upper, lower), axis=0).reshape(3072, 2048)
    else:
        raise RuntimeError(
            f"Not supported sparse_mode {sparse_mode}, only support [0,1,2,3,4,5,6]"
        )
    return atten_masks, atten_mask_npu, prefix


def skip_zero_row(b, n2, g, s1, x_max, x_sum, softmax_res):
    for i in range(b):
        for j in range(n2):
            for k in range(g):
                for l in range(s1):
                    if x_max[i, j, k, l, :] == -40000.0:
                        softmax_res[i, j, k, l, :] = 0
                        x_max[i, j, k, l, :] = torch.finfo(torch.float).min
                        x_sum[i, j, k, l, :] = torch.finfo(torch.float).max
    return softmax_res, x_max, x_sum


def gen_dropmask_uint8(rounds, counter, key, uint8_count, keep_prob):
    keep_prob_uint8 = keep_prob * 255
    uint32_count = math.ceil(uint8_count / 4)
    philox_uint32 = philox_random(rounds, counter, key, uint32_count)
    philox_uint8 = []
    golden_uint8 = []
    for each_uint32 in philox_uint32:
        four_uint8 = uint32_to_uint8_array(each_uint32)
        philox_uint8.extend(four_uint8)
    for each_philox_uint8 in philox_uint8:
        # print(each_philox_uint8)
        compare_res = 1 if each_philox_uint8 <= keep_prob_uint8 else 0
        golden_uint8.append(compare_res)
    return golden_uint8[:uint8_count]


def bin_to_uint8(golden_uint8):
    uint8_num_list = []
    for i, each_num in enumerate(golden_uint8):
        if i % 8 == 0:
            num_str = str(each_num)
        else:
            num_str += str(each_num)
        if (i + 1) % 8 == 0:
            uint8_num = int(num_str[::-1], 2)
            # print(uint8_num)
            uint8_num_list.append(uint8_num)
    return uint8_num_list


def inc_counter(counter):
    for i in range(4):
        counter[i] = (counter[i] + 1) & (MASK_32)
        if counter[i] != 0:
            return counter
    return counter


def philox_random(rounds, counter, key, count):
    ret = list()
    for i in range((count + 255) // 256 * 256 // 4):
        ret.extend(philox4_32(counter[:], key[:], rounds))
        counter = inc_counter(counter[:])
    return np.array(ret)[:count]


def philox4_round(counter, key, philox_m, len_w, mask_w):
    prod = philox_m[VAL_1] * counter[VAL_1]
    hi_1 = prod >> len_w
    lo_1 = prod & mask_w
    prod = philox_m[VAL_2] * counter[VAL_3]
    hi_2 = prod >> len_w
    lo_2 = prod & mask_w
    counter[VAL_1] = hi_2 ^ counter[VAL_2] ^ key[VAL_1]
    counter[VAL_2] = lo_2
    counter[VAL_3] = hi_1 ^ counter[VAL_4] ^ key[VAL_2]
    counter[VAL_4] = lo_1


def philox4_32(counter, key, rounds):
    return philox(
        counter,
        key,
        philox4_round,
        PHILOX_M4_32,
        philox4_bumpkey,
        PHILOX_W_32,
        32,
        MASK_32,
        rounds,
    )


def philox(
    counter,
    key,
    philox_round,
    philox_m,
    philox_bumpkey,
    philox_w,
    len_w,
    mask_w,
    rounds,
):
    for i in range(rounds - 1):
        philox_round(counter, key, philox_m, len_w, mask_w)
        philox_bumpkey(key, philox_w, mask_w)
    philox_round(counter, key, philox_m, len_w, mask_w)
    return counter


def philox4_bumpkey(key, philox_w, mask_w):
    key[VAL_1] = (key[VAL_1] + philox_w[VAL_1]) & mask_w
    key[VAL_2] = (key[VAL_2] + philox_w[VAL_2]) & mask_w


def uint32_to_uint8_array(uint32_data):
    # 判断系统大小端
    little_endian = np.array([1], dtype=np.uint32).view(np.uint8)[0] == 1
    # 创建 uint8 数组
    uint8_array = np.zeros((4), dtype=np.uint8)
    if little_endian:
        # 如果系统是小端的，直接将 uint32 转为 uint8
        uint8_array[0] = uint32_data & 0xFF
        uint8_array[1] = (uint32_data >> 8) & 0xFF
        uint8_array[2] = (uint32_data >> 16) & 0xFF
        uint8_array[3] = (uint32_data >> 24) & 0xFF
    else:
        # 如果系统是大端的，需要交换字节顺序后再转为 uint8
        uint32_data = np.array(uint32_data, dtype=np.dtype(">u4"))
        uint8_array[0] = uint32_data.view(np.uint8)[3]
        uint8_array[1] = uint32_data.view(np.uint8)[2]
        uint8_array[2] = uint32_data.view(np.uint8)[1]
        uint8_array[3] = uint32_data.view(np.uint8)[0]
    return uint8_array
