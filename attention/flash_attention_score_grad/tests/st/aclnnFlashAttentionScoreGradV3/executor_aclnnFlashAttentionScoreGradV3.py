#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
import ctypes
import numpy as np


def tsoftmax_sink(x, sink=None):
    softmax_max = torch.max(x, dim=-1, keepdims=True)[0]
    if sink is not None and len(sink.shape) != 0:
        sink_expanded = sink.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, n, 1, 1)
        sink_expanded = sink_expanded.expand(
            1, x.shape[1], x.shape[2], 1
        )  # (b, n, s, 1)
        softmax_max = torch.maximum(softmax_max, sink_expanded)
    x_sub = x.sub(softmax_max)
    y = torch.exp(x_sub)
    softmax_sum = y.sum(dim=-1, keepdims=True)
    if sink is not None and len(sink.shape) != 0:
        softmax_sum = softmax_sum + torch.exp(sink_expanded - softmax_max)
    softmax_res = y.div(softmax_sum)
    return softmax_res, softmax_max, softmax_sum


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


def _get_layout_dims(q, k, v, headNum, inputLayout):
    if inputLayout == "BSH":
        B = q.shape[0]
        S1 = q.shape[1]
        N1 = headNum
        D1 = q.shape[2] // N1
        S2 = v.shape[1]
        N2 = k.shape[2] // D1
        D = D1
    elif inputLayout == "SBH":
        B = q.shape[1]
        S1 = q.shape[0]
        N1 = headNum
        D1 = q.shape[2] // N1
        S2 = v.shape[0]
        N2 = k.shape[2] // D1
        D = D1
    elif inputLayout == "BSND":
        B = q.shape[0]
        S1 = q.shape[1]
        N1 = headNum
        N2 = k.shape[2]
        S2 = k.shape[1]
        D = q.shape[3]
    elif inputLayout == "BNSD":
        B = q.shape[0]
        N1 = headNum
        N2 = k.shape[1]
        S1 = q.shape[2]
        S2 = k.shape[2]
        D = q.shape[3]
    else:
        raise RuntimeError(f"not support inputLayout {inputLayout}")
    return B, N1, N2, S1, S2, D


def _get_atten_mask_shape_name(atten_mask):
    if atten_mask is None or len(atten_mask.shape) == 0:
        return "NONE"
    shape = list(atten_mask.shape)
    if len(shape) == 2:
        return "SS"
    if len(shape) == 4:
        if shape[0] == 1 and shape[1] == 1:
            return "11SS"
        if shape[1] == 1:
            return "B1SS"
        return "BNSS"
    raise RuntimeError(f"not support shape of atten_mask {shape}")


def _make_atten_mask_from_sparse_mode(
    input_data: InputDataset, use_npu_mask: bool = False
):
    q = input_data.kwargs["query"]
    k = input_data.kwargs["keyIn"]
    v = input_data.kwargs["value"]
    atten_mask = input_data.kwargs["attenMaskOptional"]
    headNum = input_data.kwargs["headNum"]
    inputLayout = input_data.kwargs["inputLayout"]
    sparseMode = input_data.kwargs["sparseMode"]
    pre_tocken = input_data.kwargs["preTokens"]
    next_tocken = input_data.kwargs["nextTokens"]

    B, N1, N2, S1, S2, D = _get_layout_dims(q, k, v, headNum, inputLayout)
    mask_dtype = atten_mask.dtype if atten_mask is not None else torch.bool
    mask_shape_name = _get_atten_mask_shape_name(atten_mask)

    if mask_shape_name == "SS":
        shape = [S1, S2]
    elif mask_shape_name == "B1SS":
        shape = [B, 1, S1, S2]
    elif mask_shape_name == "BNSS":
        shape = [B, N1, S1, S2]
    elif mask_shape_name == "11SS":
        shape = [1, 1, S1, S2]
    else:
        new_mask = torch.tensor(0, dtype=mask_dtype)
        input_data.kwargs["attenMaskOptional"] = new_mask
        input_data.kwargs["preTokens"] = 65536
        input_data.kwargs["nextTokens"] = 65536
        return new_mask

    if sparseMode == 0:
        atten_mask_u = torch.triu(torch.ones(shape), diagonal=next_tocken + 1)
        atten_mask_l = torch.tril(torch.ones(shape), diagonal=-pre_tocken - 1)
        new_mask = (atten_mask_u + atten_mask_l).to(mask_dtype)
    elif sparseMode == 1:
        new_mask = torch.zeros(shape).to(mask_dtype)
        pre_tocken = S1
        next_tocken = S2
    elif sparseMode == 2:
        new_mask = torch.triu(torch.ones(shape), diagonal=1).to(mask_dtype)
        if use_npu_mask:
            new_mask = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(mask_dtype)
        pre_tocken = S1
        next_tocken = 0
    elif sparseMode == 3:
        new_mask = torch.triu(torch.ones(shape), diagonal=S2 - S1 + 1).to(mask_dtype)
        if use_npu_mask:
            new_mask = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(mask_dtype)
        pre_tocken = S2
        next_tocken = 0
    elif sparseMode == 4:
        atten_mask_u = torch.triu(torch.ones(shape), diagonal=next_tocken + 1 + S2 - S1)
        atten_mask_l = torch.tril(torch.ones(shape), diagonal=-pre_tocken - 1 + S2 - S1)
        new_mask = (atten_mask_u + atten_mask_l).to(mask_dtype)
        if use_npu_mask:
            new_mask = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(mask_dtype)
    elif sparseMode == 5:
        if mask_shape_name in ["SS", "11SS"]:
            raise RuntimeError(
                f"prefix not support shape of atten_mask {mask_shape_name}, only support BNSS and B1SS"
            )
        new_mask = torch.triu(torch.ones(shape), diagonal=S2 - S1 + 1).to(mask_dtype)
    elif sparseMode == 6:
        if mask_shape_name in ["SS", "11SS"]:
            raise RuntimeError(
                f"prefix not support shape of atten_mask {mask_shape_name}, only support BNSS and B1SS"
            )
        new_mask = torch.triu(torch.ones(shape), diagonal=S2 - S1 + 1).to(mask_dtype)
        if use_npu_mask:
            upper = torch.triu(torch.ones(2048, 2048), diagonal=1)
            lower = torch.cat((torch.zeros(1024, 1024), torch.ones(1024, 1024)), dim=1)
            new_mask = torch.cat((upper, lower), dim=0).to(mask_dtype)
    else:
        new_mask = atten_mask

    input_data.kwargs["attenMaskOptional"] = new_mask
    input_data.kwargs["preTokens"] = pre_tocken
    input_data.kwargs["nextTokens"] = next_tocken
    return new_mask


def tforward_sink(
    q, k, v, drop_mask, atten_mask, pse, sink, headNum, inputLayout, scale, keep_prob
):
    if inputLayout == "BSH":
        # (B,S,N*D) => (B,N,S,D)
        B = q.shape[0]
        S1 = q.shape[1]
        N1 = headNum
        D1 = q.shape[2] // N1
        S2 = v.shape[1]
        N2 = k.shape[2] // D1
        D2 = v.shape[2] // N2
        q = q.view(B, S1, N1, D1).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, S2, N2, D1).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, S2, N2, D2).permute(0, 2, 1, 3).contiguous()
    elif inputLayout == "SBH":
        # (S,B,N*D) => (B,N,S,D)
        B = q.shape[1]
        S1 = q.shape[0]
        N1 = headNum
        D1 = q.shape[2] // N1
        S2 = v.shape[0]
        N2 = k.shape[2] // D1
        D2 = v.shape[2] // N2
        q = q.view(S1, B, N1, D1).permute(1, 2, 0, 3).contiguous()
        k = k.view(S2, B, N2, D1).permute(1, 2, 0, 3).contiguous()
        v = v.view(S2, B, N2, D2).permute(1, 2, 0, 3).contiguous()
    elif inputLayout == "BSND":
        # (B,S,N,D) => (B,N,S,D)
        B = q.shape[0]
        N1 = headNum
        N2 = k.shape[2]
        S2 = k.shape[1]
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
    elif inputLayout == "BNSD":
        B = q.shape[0]
        N1 = headNum
        N2 = k.shape[1]
        S2 = k.shape[2]
        D1 = q.shape[3]
        D2 = v.shape[3]
    q_ori_dtype = q.dtype
    if q_ori_dtype == torch.float64:
        gtype = torch.float64
    elif q_ori_dtype != torch.float32:
        gtype = torch.float32
    else:
        gtype = torch.float64
    q = q.to(gtype)
    k = k.to(gtype)
    v = v.to(gtype)
    atten_mask = atten_mask.to(gtype)
    if not (N1 == N2):
        k = broadcastKV_sigle(N1, N2, k, k.dtype)
        v = broadcastKV_sigle(N1, N2, v, v.dtype)
    if pse is None or len(pse.shape) == 0:
        qk = torch.matmul(q, k.permute(0, 1, 3, 2)).mul(scale)
    else:
        qk = (torch.matmul(q, k.permute(0, 1, 3, 2)) + pse).mul(scale)
    if atten_mask is None or len(atten_mask.shape) == 0:
        qk = qk
    else:
        qk = qk + atten_mask * (-40000.0)  # -10000
    softmax_res, softmax_max, softmax_sum = tsoftmax_sink(qk, sink)
    if len(atten_mask.shape) != 0:
        softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = (
            0  # 要斟酌一下，部分场景存在计算异常精度问题
        )
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res
    else:
        drop_res = softmax_res * drop_mask * (1.0 / (keep_prob))
    attention_out = torch.matmul(drop_res, v)

    if inputLayout == "BSND":
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
    elif inputLayout == "BSH":
        attention_out = (
            attention_out.permute(0, 2, 1, 3).reshape(B, S1, -1).contiguous()
        )
    elif inputLayout == "SBH":
        attention_out = (
            attention_out.permute(2, 0, 1, 3).reshape(S1, B, -1).contiguous()
        )

    if q_ori_dtype == torch.float32 and gtype == torch.float64:
        softmax_sum = softmax_sum.to(q_ori_dtype)
        softmax_max = softmax_max.to(q_ori_dtype)
        attention_out = attention_out.to(q_ori_dtype)
    elif q_ori_dtype == torch.float16 or q_ori_dtype == torch.bfloat16:
        attention_out = attention_out.to(q_ori_dtype)
    return (
        attention_out,
        softmax_res,
        softmax_max,
        softmax_sum,
    )


def tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdims=True)
    sub_r = dp - muls_r
    res = sub_r * softmax_res
    return res


def simple_softmax(src, max, sum):
    dst = np.exp(src - max) / sum
    return dst


def tbackward_sink(
    dx,
    q,
    k,
    v,
    pse,
    x_max,
    x_sum,
    atten_in,
    drop_mask,
    atten_mask,
    scale,
    keep_prob,
    sink,
):
    if sink is not None and len(sink.shape) != 0:
        sink = sink.view(-1, 1, 1)
    # cube 1
    if pse is None or len(pse.shape) == 0:
        qk = torch.matmul(q, k.permute(0, 1, 3, 2)).mul(scale)
    else:
        qk = (torch.matmul(q, k.permute(0, 1, 3, 2)) + pse).mul(scale)
    # cube 2
    dyv = torch.matmul(dx, v.permute(0, 1, 3, 2))

    # SubGraphA
    qk = qk + atten_mask.bool() * (-4000000000.0)
    p = simple_softmax(qk, x_max[:, :, :, 0:1], x_sum[:, :, :, 0:1])
    if drop_mask is None or len(drop_mask.shape) == 0:
        p_drop = p.permute(0, 1, 3, 2)
    else:
        p_drop = p.mul(drop_mask).mul(1.0 / (keep_prob)).permute(0, 1, 3, 2)
    # SubGraphB
    print(f"dx is : {dx.shape}")
    print(f"atten_in is : {atten_in.shape}")

    softmax_grad = (dx * atten_in).sum(dim=-1, keepdims=True)
    if drop_mask is None or len(drop_mask.shape) == 0:
        dp = dyv
    else:
        dp = dyv.mul(drop_mask).mul(1.0 / (keep_prob))
    ds = p * (dp - softmax_grad)
    dsink = None
    if sink is not None and len(sink.shape) != 0:
        dsink_sum = p * dp
        dsink_sum = dsink_sum * simple_softmax(sink, x_max, x_sum)
        dsink = -dsink_sum.sum(dim=(0, 2, 3))

    # cube 345
    dv = torch.matmul(p_drop, dx)
    dq = torch.matmul(ds, k).mul(scale)
    dk = torch.matmul(ds.permute(0, 1, 3, 2), q).mul(scale)
    return dq, dk, dv, dsink


def tbackward(
    dx,
    q,
    k,
    v,
    softmax_max,
    softmax_sum,
    attention_out,
    drop_mask,
    pse,
    atten_mask,
    sink,
    headNum,
    inputLayout,
    scale,
    keep_prob,
):
    if inputLayout == "BSH":
        # (B,S,N*D) => (B,N,S,D)
        B = q.shape[0]
        S1 = q.shape[1]
        N1 = headNum
        D1 = q.shape[2] // N1
        S2 = v.shape[1]
        N2 = k.shape[2] // D1
        D2 = v.shape[2] // N2
        q = q.view(B, S1, N1, D1).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, S2, N2, D1).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, S2, N2, D2).permute(0, 2, 1, 3).contiguous()
        dx = dx.view(B, S1, N1, D2).permute(0, 2, 1, 3).contiguous()
        attention_out = (
            attention_out.view(B, S1, N1, D2).permute(0, 2, 1, 3).contiguous()
        )
    elif inputLayout == "SBH":
        # (S,B,N*D) => (B,N,S,D)
        B = q.shape[1]
        S1 = q.shape[0]
        N1 = headNum
        D1 = q.shape[2] // N1
        S2 = v.shape[0]
        N2 = k.shape[2] // D1
        D2 = v.shape[2] // N2
        q = q.view(S1, B, N1, D1).permute(1, 2, 0, 3).contiguous()
        k = k.view(S2, B, N2, D1).permute(1, 2, 0, 3).contiguous()
        v = v.view(S2, B, N2, D2).permute(1, 2, 0, 3).contiguous()
        dx = dx.view(S1, B, N1, D2).permute(1, 2, 0, 3).contiguous()
        attention_out = (
            attention_out.view(S1, B, N1, D2).permute(1, 2, 0, 3).contiguous()
        )
    elif inputLayout == "BSND":
        # (B,S,N,D) => (B,N,S,D)
        B = q.shape[0]
        N1 = headNum
        N2 = k.shape[2]
        S2 = k.shape[1]
        G = N1 // N2
        D1 = q.shape[3]
        D2 = v.shape[3]
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        dx = dx.permute(0, 2, 1, 3).contiguous()
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
    elif inputLayout == "BNSD":
        B = q.shape[0]
        N1 = headNum
        N2 = k.shape[1]
        S2 = k.shape[2]
        G = N1 // N2
        D1 = q.shape[3]
        D2 = v.shape[3]
    # if inputLayout == "BSND":
    #     attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
    # elif inputLayout == "BSH":
    #     attention_out = attention_out.permute(0, 2, 1, 3).reshape(B, S1, -1).contiguous()
    # elif inputLayout == "SBH":
    #     attention_out = attention_out.permute(2, 0, 1, 3).reshape(S1, B, -1).contiguous()
    q_ori_dtype = q.dtype
    if q_ori_dtype == torch.float64:
        gtype = torch.float64
    elif q_ori_dtype != torch.float32:
        gtype = torch.float32
    else:
        gtype = torch.float64
    q = q.to(gtype)
    k = k.to(gtype)
    v = v.to(gtype)
    dx = dx.to(gtype)
    pse = pse.to(gtype)
    softmax_max = softmax_max.to(gtype)
    softmax_sum = softmax_sum.to(gtype)
    attention_out = attention_out.to(gtype)
    if not (N1 == N2):
        k = broadcastKV_sigle(N1, N2, k, k.dtype)
        v = broadcastKV_sigle(N1, N2, v, v.dtype)
    dq, dk, dv, dsink = tbackward_sink(
        dx,
        q,
        k,
        v,
        pse,
        softmax_max,
        softmax_sum,
        attention_out,
        drop_mask,
        atten_mask,
        scale,
        keep_prob,
        sink.to(gtype) if sink is not None else None,
    )
    if q_ori_dtype == torch.float32 and gtype == torch.float64:
        dq = dq.to(q_ori_dtype)
        dk = dk.to(q_ori_dtype)
        dv = dv.to(q_ori_dtype)
    elif q_ori_dtype == torch.float16 or q_ori_dtype == torch.bfloat16:
        dq = dq.to(q_ori_dtype)
        dk = dk.to(q_ori_dtype)
        dv = dv.to(q_ori_dtype)
    if not (N1 == N2):
        G = int(N1 // N2)
        dk = torch.sum(dk.reshape(B, N2, G, S2, D1), dim=2, keepdim=True).reshape(
            B, N2, S2, D1
        )
        dv = torch.sum(dv.reshape(B, N2, G, S2, D2), dim=2, keepdim=True).reshape(
            B, N2, S2, D2
        )
    if inputLayout == "BSND":
        dq = dq.permute(0, 2, 1, 3).contiguous()
        dk = dk.permute(0, 2, 1, 3).contiguous()
        dv = dv.permute(0, 2, 1, 3).contiguous()
    elif inputLayout == "BSH":
        dq = dq.permute(0, 2, 1, 3).reshape(B, S1, -1).contiguous()
        dk = dk.permute(0, 2, 1, 3).reshape(B, S2, -1).contiguous()
        dv = dv.permute(0, 2, 1, 3).reshape(B, S2, -1).contiguous()
    elif inputLayout == "SBH":
        dq = dq.permute(2, 0, 1, 3).reshape(S1, B, -1).contiguous()
        dk = dk.permute(2, 0, 1, 3).reshape(S2, B, -1).contiguous()
        dv = dv.permute(2, 0, 1, 3).reshape(S2, B, -1).contiguous()

    return dq, dk, dv, torch.Tensor(0), torch.Tensor(0)


def aclnn_op_func_fag_cpu(input_data: InputDataset):
    q = input_data.kwargs["query"]
    k = input_data.kwargs["keyIn"]
    v = input_data.kwargs["value"]
    dy = input_data.kwargs["dy"]
    attention_in = input_data.kwargs["attentionInOptional"]
    softmax_max = input_data.kwargs["softmaxMaxOptional"]
    softmax_sum = input_data.kwargs["softmaxSumOptional"]
    real_shift = input_data.kwargs["pseShiftOptional"]
    drop_mask = input_data.kwargs["dropMaskOptional"]
    atten_mask = input_data.kwargs["attenMaskOptional"]
    sink = input_data.kwargs["sinkInOptional"]
    headNum = input_data.kwargs["headNum"]
    inputLayout = input_data.kwargs["inputLayout"]
    scaleValue = input_data.kwargs["scaleValue"]
    keep_prob = input_data.kwargs["keepProb"]

    return tbackward(
        dy,
        q,
        k,
        v,
        softmax_max,
        softmax_sum,
        attention_in,
        drop_mask,
        real_shift,
        atten_mask,
        sink,
        headNum,
        inputLayout,
        scaleValue,
        keep_prob,
    )


@register("executor_flash_attention_score_grad_v3")
class FlashAttentionScoreGradApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FlashAttentionScoreGradApi, self).__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        _make_atten_mask_from_sparse_mode(input_data, use_npu_mask=False)
        q = input_data.kwargs["query"]
        k = input_data.kwargs["keyIn"]
        v = input_data.kwargs["value"]
        softmax_max = input_data.kwargs["softmaxMaxOptional"]
        softmax_sum = input_data.kwargs["softmaxSumOptional"]
        real_shift = input_data.kwargs["pseShiftOptional"]
        drop_mask = input_data.kwargs["dropMaskOptional"]
        atten_mask = input_data.kwargs["attenMaskOptional"]
        sink = input_data.kwargs["sinkInOptional"]
        headNum = input_data.kwargs["headNum"]
        inputLayout = input_data.kwargs["inputLayout"]
        scaleValue = input_data.kwargs["scaleValue"]
        keep_prob = input_data.kwargs["keepProb"]

        if self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"

        attention_out, softmax_res, softmax_max, softmax_sum = tforward_sink(
            q.to(device),
            k.to(device),
            v.to(device),
            drop_mask.to(device),
            atten_mask.to(device),
            real_shift.to(device),
            sink.to(device) if sink is not None else None,
            headNum,
            inputLayout,
            scaleValue,
            keep_prob,
        )
        print(f"337 attention_out is :{attention_out.shape}")
        softmax_max = softmax_max.repeat(1, 1, 1, 8)
        softmax_sum = softmax_sum.repeat(1, 1, 1, 8)
        input_data.kwargs["softmaxMaxOptional"] = softmax_max
        input_data.kwargs["softmaxSumOptional"] = softmax_sum
        input_data.kwargs["softmaxInOptional"] = softmax_res
        input_data.kwargs["attentionInOptional"] = attention_out

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        return aclnn_op_func_fag_cpu(input_data)


@register("aclnn_flash_attention_score_grad_v3")
class aclnnFlashAttentionScoreApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        _make_atten_mask_from_sparse_mode(input_data, use_npu_mask=True)
        input_args, output_packages = super().init_by_input_data(input_data)
        AclTensorPtr = ctypes.POINTER(AclTensor)  # tensor指针类型
        null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
        null_tensor_ptr = ctypes.cast(
            null_void_ptr, AclTensorPtr
        )  # 把这个空指针类型转换为tensor指针类型
        input_args[5] = null_tensor_ptr
        input_args[6] = null_tensor_ptr
        input_args[12] = TensorPtr()
        return input_args, output_packages

    def after_call(self, output_packages):
        return super().after_call(output_packages)
