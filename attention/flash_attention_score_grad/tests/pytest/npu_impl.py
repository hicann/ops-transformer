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
import torch_npu
import numpy as np
import math
import ml_dtypes
from einops import rearrange
from common import CalculationContext, DEVICE_ID, pta_mode, log


def PTAtest(ctx: CalculationContext):
    B = ctx.input_case["B"]
    Nq = ctx.input_case["N1"]
    Nkv = ctx.input_case["N2"] if ctx.input_case["N2"] else Nq
    S = ctx.input_case["S1"]
    Skv = ctx.input_case["S2"]
    D = ctx.input_case["D"]
    D_V = ctx.input_case["D_V"]
    sparse_mode = ctx.input_case["sparse_mode"]
    scalar = 0
    if D != 0:
        scalar = 1 / math.sqrt(D)  # D=64时，为0.125
    dtype = ctx.input_case["dtype"]
    pttype = torch.float16
    if dtype == "bf16" or dtype == "BF16":
        pttype = torch.bfloat16
    if dtype == "fp32" or dtype == "FP32":
        pttype = torch.float
    # FP8 return np.float16(when out_dtype=fp16) or tf.bfloat16.as_numpy_dtype(when out_dtype=bf16)
    out_dtype_np = getOutDtypeNp(ctx.input_case["out_dtype"])
    # FP8 return 0(when out_dtype=fp16) or 1(when out_dtype=bf16)
    out_dtype_num = getOutDtypeZeroOrOne(ctx.input_case["out_dtype"])
    input_layout = (
        ctx.input_case["input_layout"] if "input_layout" in ctx.input_case else "SBH"
    )
    pse_layout = (
        ctx.input_case["pse_shape"] if "pse_shape" in ctx.input_case else "NONE"
    )
    pse_type = ctx.input_case["pse_type"] if "pse_type" in ctx.input_case else 1
    atten_mask_shape = (
        ctx.input_case["atten_mask_shape"]
        if "atten_mask_shape" in ctx.input_case
        else "SS"
    )
    keep_prob = (
        ctx.input_case["drop_out_possibility"]
        if "drop_out_possibility" in ctx.input_case
        else 0.8
    )
    seed = ctx.input_case["seed"]
    offset = ctx.input_case["offset"]

    if input_layout == "TND":
        cu_seqlens_q = ctx.cu_seqlens_q_npu
        cu_seqlens_k = ctx.cu_seqlens_kv_npu

        seqlens_list_q = np.array(ctx.input_case["actual_seq_qlen"])
        seqlens_list_kv = np.array(ctx.input_case["actual_seq_kvlen"])

    torch.npu.synchronize()
    q = ctx.q_npu
    k = ctx.k_npu
    v = ctx.v_npu
    dx = ctx.dx_npu
    atten_masks = ctx.atten_masks_npu
    out = ctx.out_npu
    q_rope = ctx.q_rope_npu
    k_rope = ctx.k_rope_npu

    prefix = None
    if sparse_mode in (5, 6):
        prefix = ctx.prefix_npu

    x_max = ctx.x_max_npu
    x_sum = ctx.x_sum_npu
    if ctx.input_case["dtype"] == "fp8_e4m3fn" or ctx.input_case["dtype"] == "fp8_e5m2":
        d_scale_q = ctx.dscale_q_npu
        d_scale_k = ctx.dscale_k_npu
        d_scale_v = ctx.dscale_v_npu
        d_scale_dy = ctx.dscale_dx_npu
        d_scale_o = ctx.dscale_o_npu

        # 量化参数固定为BNSD
        d_scale_q = rearrange(d_scale_q, "b n g s d -> b (n g) s d")
        d_scale_dy = rearrange(d_scale_dy, "b n g s d -> b (n g) s d")
        d_scale_o = rearrange(d_scale_o, "b n g s d -> b (n g) s d")
        d_scale_k = rearrange(d_scale_k, "b n g s d -> b (n g) s d")
        d_scale_v = rearrange(d_scale_v, "b n g s d -> b (n g) s d")

    pre_tockens = (
        ctx.input_case["pre_tockens"] if "pre_tockens" in ctx.input_case else 65536
    )
    next_tockens = (
        ctx.input_case["next_tockens"] if "next_tockens" in ctx.input_case else 65536
    )
    pse = None if (pse_layout == "NONE") else ctx.pse_npu

    torch.npu.synchronize()
    device = torch.device(f"npu:{DEVICE_ID}")
    if ctx.input_case["dtype"] == "fp8_e5m2":
        q = q.to(torch.float8_e5m2).to(device)
        k = k.to(torch.float8_e5m2).to(device)
        v = v.to(torch.float8_e5m2).to(device)
        dx = dx.to(torch.float8_e5m2).to(device)
        out = out.to(getOutDtype(ctx.input_case["out_dtype"])).to(device)
    elif ctx.input_case["dtype"] == "fp8_e4m3fn":
        q = q.to(torch.float8_e4m3fn).to(device)
        k = k.to(torch.float8_e4m3fn).to(device)
        v = v.to(torch.float8_e4m3fn).to(device)
        dx = dx.to(torch.float8_e4m3fn).to(device)
        out = out.to(getOutDtype(ctx.input_case["out_dtype"])).to(device)
    else:
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        dx = dx.to(device)
        q_rope = q_rope.to(device)
        k_rope = k_rope.to(device)

    pse1 = pse.clone().to(device) if (not pse_layout.upper() == "NONE") else None
    atten_masks = atten_masks.bool().to(device)
    if atten_mask_shape == "NONE":
        atten_masks = None
    torch.npu.synchronize()
    # DSA_gen_bit_mask
    if pta_mode == "auto_grad":
        if (
            ctx.input_case["dtype"] == "fp8_e4m3fn"
            or ctx.input_case["dtype"] == "fp8_e5m2"
        ):
            log("FP8 场景暂不支持 auto_grad 模式，请使用 only_grad 模式")
            return
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True
        if ctx.input_case["rope"] == 1:
            q_rope.requires_grad = True
            k_rope.requires_grad = True
        torch.npu.manual_seed(2)
        if input_layout == "TND":
            if ctx.input_case["rope"]:
                npu_rst = torch_npu.npu_fusion_attention_v2(
                    q,
                    k,
                    v,
                    Nq,
                    query_rope=q_rope,
                    key_rope=k_rope,
                    pse=pse1,
                    padding_mask=None,
                    atten_mask=atten_masks,
                    scale=scalar,
                    keep_prob=keep_prob,
                    input_layout=input_layout,
                    actual_seq_qlen=tuple(cu_seqlens_q),
                    actual_seq_kvlen=tuple(cu_seqlens_k),
                    pre_tokens=pre_tockens,
                    next_tokens=next_tockens,
                    inner_precise=0,
                    prefix=prefix,
                    sparse_mode=ctx.input_case["sparse_mode"],
                )
            else:
                npu_rst = torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    Nq,
                    pse=pse1,
                    padding_mask=None,
                    atten_mask=atten_masks,
                    scale=scalar,
                    keep_prob=keep_prob,
                    input_layout=input_layout,
                    actual_seq_qlen=tuple(cu_seqlens_q),
                    actual_seq_kvlen=tuple(cu_seqlens_k),
                    pre_tockens=pre_tockens,
                    next_tockens=next_tockens,
                    inner_precise=0,
                    prefix=prefix,
                    sparse_mode=ctx.input_case["sparse_mode"],
                )
        else:
            if ctx.input_case["rope"]:
                npu_rst = torch_npu.npu_fusion_attention_v2(
                    query=q,
                    query_rope=q_rope,
                    key=k,
                    key_rope=k_rope,
                    value=v,
                    head_num=Nq,
                    pse=pse1,
                    padding_mask=None,
                    atten_mask=atten_masks,
                    scale=scalar,
                    keep_prob=keep_prob,
                    input_layout=input_layout,
                    pre_tokens=pre_tockens,
                    next_tokens=next_tockens,
                    inner_precise=0,
                    prefix=prefix,
                    sparse_mode=ctx.input_case["sparse_mode"],
                )
            else:
                npu_rst = torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    Nq,
                    pse=pse1,
                    padding_mask=None,
                    atten_mask=atten_masks,
                    scale=scalar,
                    keep_prob=keep_prob,
                    input_layout=input_layout,
                    pre_tockens=pre_tockens,
                    next_tockens=next_tockens,
                    inner_precise=0,
                    prefix=prefix,
                    sparse_mode=ctx.input_case["sparse_mode"],
                )
        torch.npu.synchronize()
        npu_out = npu_rst[0]
        npu_max = npu_rst[1]
        npu_sum = npu_rst[2]
        npu_out.backward(dx)
        dq = q.grad
        dk = k.grad
        dv = v.grad
        if ctx.input_case["rope"] == 1:
            dq_rope = q_rope.grad
            dk_rope = k_rope.grad
    elif pta_mode == "only_grad":
        torch.npu.synchronize()
        for i in range(5):
            torch.npu.manual_seed(2)
            if input_layout == "TND":
                if (
                    ctx.input_case["dtype"] == "fp8_e4m3fn"
                    or ctx.input_case["dtype"] == "fp8_e5m2"
                ):
                    log("FP8 场景暂不支持 TND layout")
                    return
                if ctx.input_case["rope"]:
                    dq, dk, dv, _, dq_rope, dk_rope = (
                        torch_npu.npu_fusion_attention_grad_v2(
                            q,
                            k,
                            v,
                            dx,
                            Nq,
                            query_rope=q_rope,
                            key_rope=k_rope,
                            pse=pse1,
                            padding_mask=None,
                            atten_mask=atten_masks,
                            softmax_max=x_max.float().to(device),
                            softmax_sum=x_sum.float().to(device),
                            softmax_in=None,
                            attention_in=out.to(pttype).to(device),
                            scale_value=scalar,
                            keep_prob=keep_prob,
                            input_layout=input_layout,
                            actual_seq_qlen=tuple(cu_seqlens_q),
                            actual_seq_kvlen=tuple(cu_seqlens_k),
                            pre_tokens=pre_tockens,
                            next_tokens=next_tockens,
                            seed=seed,
                            offset=offset,
                            numels=(seqlens_list_q * seqlens_list_kv).sum() * Nq,
                            inner_precise=0,
                            prefix=prefix,
                            sparse_mode=ctx.input_case["sparse_mode"],
                            pse_type=pse_type,
                        )
                    )
                else:
                    dq, dk, dv, _, dq_rope, dk_rope, _ = (
                        torch_npu.npu_fusion_attention_grad_v2(
                            q,
                            k,
                            v,
                            dx,
                            Nq,
                            pse=pse1,
                            padding_mask=None,
                            atten_mask=atten_masks,
                            softmax_max=x_max.float().to(device),
                            softmax_sum=x_sum.float().to(device),
                            softmax_in=None,
                            attention_in=out.to(pttype).to(device),
                            scale_value=scalar,
                            keep_prob=keep_prob,
                            input_layout=input_layout,
                            actual_seq_qlen=tuple(cu_seqlens_q),
                            actual_seq_kvlen=tuple(cu_seqlens_k),
                            pre_tokens=pre_tockens,
                            next_tokens=next_tockens,
                            seed=seed,
                            offset=offset,
                            numels=(seqlens_list_q * seqlens_list_kv).sum() * Nq,
                            inner_precise=0,
                            prefix=prefix,
                            sparse_mode=ctx.input_case["sparse_mode"],
                            pse_type=pse_type,
                        )
                    )
            else:
                if (
                    ctx.input_case["dtype"] == "fp8_e4m3fn"
                    or ctx.input_case["dtype"] == "fp8_e5m2"
                ):
                    dq, dk, dv, *_ = torch_npu.npu_fusion_attention_grad_v2(
                        q,
                        k,
                        v,
                        dx,
                        Nq,
                        pse=pse1,
                        padding_mask=None,
                        atten_mask=atten_masks,
                        softmax_max=x_max.float().to(device),
                        softmax_sum=x_sum.float().to(device),
                        softmax_in=None,
                        attention_in=out,
                        scale_value=scalar,
                        keep_prob=keep_prob,
                        input_layout=input_layout,
                        pre_tokens=pre_tockens,
                        next_tokens=next_tockens,
                        seed=seed,
                        offset=offset,
                        numels=B * Nq * S * Skv,
                        inner_precise=0,
                        prefix=prefix,
                        sparse_mode=ctx.input_case["sparse_mode"],
                        d_scale_q=d_scale_q.float().to(device),
                        d_scale_k=d_scale_k.float().to(device),
                        d_scale_v=d_scale_v.float().to(device),
                        d_scale_dy=d_scale_dy.float().to(device),
                        d_scale_o=d_scale_o.float().to(device),
                        out_dtype=out_dtype_num,
                        pse_type=pse_type,
                    )
                else:
                    if ctx.input_case["rope"]:
                        dq, dk, dv, _, dq_rope, dk_rope, _ = (
                            torch_npu.npu_fusion_attention_grad_v2(
                                q,
                                k,
                                v,
                                dx,
                                Nq,
                                query_rope=q_rope,
                                key_rope=k_rope,
                                pse=pse1,
                                padding_mask=None,
                                atten_mask=atten_masks,
                                softmax_max=x_max.float().to(device),
                                softmax_sum=x_sum.float().to(device),
                                softmax_in=None,
                                attention_in=out.to(pttype).to(device),
                                scale_value=scalar,
                                keep_prob=keep_prob,
                                input_layout=input_layout,
                                pre_tokens=pre_tockens,
                                next_tokens=next_tockens,
                                seed=seed,
                                offset=offset,
                                numels=B * Nq * S * Skv,
                                inner_precise=0,
                                prefix=prefix,
                                sparse_mode=ctx.input_case["sparse_mode"],
                            )
                        )
                    else:
                        dq, dk, dv, *_ = torch_npu.npu_fusion_attention_grad_v2(
                            q,
                            k,
                            v,
                            dx,
                            Nq,
                            pse=pse1,
                            padding_mask=None,
                            atten_mask=atten_masks,
                            softmax_max=x_max.float().to(device),
                            softmax_sum=x_sum.float().to(device),
                            softmax_in=None,
                            attention_in=out.to(pttype).to(device),
                            scale_value=scalar,
                            keep_prob=keep_prob,
                            input_layout=input_layout,
                            pre_tokens=pre_tockens,
                            next_tokens=next_tockens,
                            seed=seed,
                            offset=offset,
                            numels=B * Nq * S * Skv,
                            inner_precise=0,
                            prefix=prefix,
                            sparse_mode=ctx.input_case["sparse_mode"],
                            pse_type=pse_type,
                        )
        torch.npu.synchronize()
    torch.npu.synchronize()

    dq = dq.cpu()
    dk = dk.cpu()
    dv = dv.cpu()
    if ctx.input_case["rope"] == 1:
        dq_rope = dq_rope.cpu()
        dk_rope = dk_rope.cpu()
    torch.npu.synchronize()
    if ctx.input_case["rope"] == 0:
        return dq, dk, dv
    else:
        return dq, dk, dv, dq_rope, dk_rope


def getOutDtypeNp(out_dtype):
    if out_dtype == "fp16":
        return np.float16
    elif out_dtype == "bf16":
        # return np.float16
        return ml_dtypes.bfloat16
    elif out_dtype == "fp32":
        return np.float32


def getOutDtypeZeroOrOne(out_dtype):
    if out_dtype == "fp16":
        return 0
    elif out_dtype == "bf16":
        return 1


def getOutDtype(out_dtype):
    if out_dtype == "fp16":
        return torch.float16
    elif out_dtype == "bf16":
        return torch.bfloat16
    elif out_dtype == "fp32":
        return torch.float32
