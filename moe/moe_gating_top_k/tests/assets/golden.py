#!/usr/bin/python
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

import numpy
import torch

__spec__ = {
    "moe_gating_top_k": "MoeGatingTopKTestSpec",
    "aclnnMoeGatingTopK": "AclnnMoeGatingTopKTestSpec",
    "aclnnMoeGatingTopKV2": "AclnnMoeGatingTopKV2TestSpec",
    "torch_npu.npu_moe_gating_top_k": "E2eMoeGatingTopKTestSpec",
}


def _softmax_torch(x, axis=-1):
    if x.dtype == torch.float16:
        x = x.to(torch.float32)
    x_max = x.max(dim=axis, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=axis, keepdim=True)
    return y / x_sum


def _pre_compare_topk(*arrays):
    """topk 输出顺序对排序键(normValue)ULP 级差异敏感: 相邻键不可区分时, kernel 与
    golden 的浮点路径微差会使相邻元素交换; 并列值(输入量化/下溢产生完全相同的键)时
    选取的专家集合也可能不同, 属于实现自由度。比对前逐行按 (值降序, 索引升序) 规范
    化; golden 侧值完全相同的并列段(长度>=2 且输出侧同段值也完全一致)内, 以及值差
    在输出精度 4 ULP 内的位置, 索引替换为输出侧索引——不可区分值下任意等值选取均
    合法, 值本身的精度由 out0 的 stat_rel_err 保证(阈值远大于 4 ULP, 不会掩盖真实
    值错误)。normOut 为按位置对齐的全量输出, 不做处理。kernel 通路输出为 numpy,
    aclnn/e2e 通路为 torch。"""

    def _as_numpy(a):
        if torch.is_tensor(a):
            if a.dtype == torch.bfloat16:
                return a.detach().cpu().to(torch.float32).numpy()
            return a.detach().cpu().numpy()
        return numpy.asarray(a)

    def _write_back(slot, arr):
        # 长度为 1 的轴反转后 strides 为负但 numpy 仍视为 C 连续(ascontiguousarray 不拷贝),
        # torch.from_numpy 却拒绝负 stride, 这里显式拷贝成正 stride
        if any(stride < 0 for stride in arr.strides):
            arr = arr.copy()
        if torch.is_tensor(slot):
            slot.copy_(torch.from_numpy(arr).to(slot.dtype))
        else:
            slot[:] = arr

    half = len(arrays) // 2
    if half < 2:
        return
    y_out = _as_numpy(arrays[0]).copy()
    y_gold = _as_numpy(arrays[half]).copy()
    if y_out.ndim != 2 or y_out.shape != y_gold.shape:
        return
    # golden 侧 idx 可能为 None(aclnn golden 不建模 expertIdxOut, 该输出比对自动抑制):
    # 仅对 y 两侧做行降序规范化(消除 topk 顺序敏感性)
    if arrays[half + 1] is None or arrays[1] is None:
        y_out = numpy.sort(y_out, axis=1)[:, ::-1].copy()
        y_gold = numpy.sort(y_gold, axis=1)[:, ::-1].copy()
        _write_back(arrays[0], y_out)
        _write_back(arrays[half], y_gold)
        return
    idx_out = _as_numpy(arrays[1]).copy()
    idx_gold = _as_numpy(arrays[half + 1]).copy()
    if idx_out.shape != idx_gold.shape or idx_out.ndim != 2:
        return
    # 输出精度下的不可区分容差(4 ULP): 覆盖 kernel 与 golden 浮点路径微差引起的
    # 相邻交换与 k 边界并列, 远小于 out0 的 stat_rel_err 阈值, 不掩盖真实值错误
    # (bfloat16 无原生 numpy finfo, eps 固定为 2^-7; 需在 bf16->fp32 转换前判断)
    y_raw = arrays[0]
    y_dtype_str = str(
        y_raw.dtype if torch.is_tensor(y_raw) else numpy.asarray(y_raw).dtype
    )
    if "bfloat16" in y_dtype_str:
        ulp_tol = 4.0 * (2.0**-7)
    else:
        ulp_tol = 4.0 * numpy.finfo(y_out.dtype).eps

    for r in range(y_out.shape[0]):
        yo = y_out[r].astype(numpy.float64)
        io = idx_out[r]
        yg = y_gold[r].astype(numpy.float64)
        ig = idx_gold[r]
        oo = numpy.lexsort((io, -yo))
        yo, io = yo[oo], io[oo]
        og = numpy.lexsort((ig, -yg))
        yg, ig = yg[og], ig[og]
        k = yo.shape[0]
        j = 0
        while j < k:
            j2 = j + 1
            while j2 < k and yg[j2] == yg[j]:
                j2 += 1
            if j2 > j + 1 and numpy.all(yo[j:j2] == yo[j]):
                ig[j:j2] = io[j:j2]
            j = j2
        # k 边界并列/相邻交换: 值在输出精度(4 ULP)内不可区分, 或绝对差低于
        # stat_rel_err 的 FLOOR(1e-7, 如 exp 下溢 denormal 被 FTZ 为 0 的场景)时,
        # 被选专家视为等价——与 out0 的值判定口径保持一致
        scale = numpy.maximum(numpy.maximum(numpy.abs(yg), numpy.abs(yo)), 1e-30)
        diff = numpy.abs(yo - yg)
        tie = (ig != io) & ((diff <= ulp_tol * scale) | (diff <= 1e-7))
        ig[tie] = io[tie]
        y_out[r] = yo
        idx_out[r] = io
        y_gold[r] = yg
        idx_gold[r] = ig

    _write_back(arrays[0], y_out)
    _write_back(arrays[1], idx_out)
    _write_back(arrays[half], y_gold)
    _write_back(arrays[half + 1], idx_gold)


def _softmax_numpy(x, axis=-1):
    if "float16" in x.dtype.name:
        x = x.astype(numpy.float32)
    x_max = x.max(axis=axis, keepdims=True)
    x_sub = x - x_max
    y = numpy.exp(x_sub)
    x_sum = y.sum(axis=axis, keepdims=True)
    return y / x_sum


class AclnnMoeGatingTopKTestSpec:
    @staticmethod
    def golden(
        x,
        biasOptional,
        k=1,
        kGroup=1,
        groupCount=1,
        groupSelectMode=0,
        renorm=0,
        normType=0,
        outFlag=False,
        routedScalingFactor=1.0,
        eps=1e-20,
        yOut=None,
        expertIdxOut=None,
        outOut=None,
        *args,
        **kwargs,
    ):
        inputIdsOptional = kwargs.get("inputIdsOptional", None)
        tid2eidOptional = kwargs.get("tid2eidOptional", None)
        ori_dtype = x.dtype
        x = x.to(torch.float32)
        if biasOptional is not None:
            biasOptional = biasOptional.to(torch.float32)

        if normType == 0:
            x = _softmax_torch(x, -1)
        elif normType == 1:
            x = 1 / (1 + numpy.exp(-x.numpy()))
            x = torch.from_numpy(x).to(torch.float32)
        elif normType == 2:
            x = torch.sqrt(torch.nn.functional.softplus(x))

        original_x = x
        if biasOptional is not None:
            x = x + biasOptional

        hashFlag = inputIdsOptional is not None and tid2eidOptional is not None
        if hashFlag:
            indices = tid2eidOptional[inputIdsOptional].to(torch.int64)
        else:
            if groupCount > 1:
                x_reshaped = x.reshape(x.shape[0], groupCount, -1)
                if groupSelectMode == 0:
                    group_x = torch.amax(x_reshaped, dim=-1)
                else:
                    top2 = torch.topk(x_reshaped, 2, dim=-1).values
                    group_x = top2.sum(dim=-1)
                _, group_indices = torch.sort(
                    group_x, dim=-1, descending=True, stable=True
                )
                group_indices = group_indices[:, :kGroup]

                mask = torch.ones((x_reshaped.shape[0], groupCount), dtype=torch.bool)
                mask[torch.arange(x_reshaped.shape[0])[:, None], group_indices] = False
                x = torch.where(mask.unsqueeze(-1), float("-inf"), x_reshaped)
                x = x.reshape(x.shape[0], -1)

            _, indices = torch.sort(x, dim=-1, stable=True, descending=True)
            indices = indices[:, :k].to(torch.int64)

        y = torch.gather(original_x, 1, indices)

        if normType != 0 or renorm != 0:
            y = y / (y.sum(dim=-1, keepdim=True) + eps)
        y = y * routedScalingFactor

        if outFlag:
            out = original_x
        else:
            out = None

        return [y.to(ori_dtype), None, out]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    pre_compare = _pre_compare_topk


class AclnnMoeGatingTopKV2TestSpec:
    @staticmethod
    def golden(
        x,
        biasOptional,
        inputIdsOptional=None,
        tid2eidOptional=None,
        k=1,
        kGroup=1,
        groupCount=1,
        groupSelectMode=0,
        renorm=0,
        normType=0,
        outFlag=False,
        routedScalingFactor=1.0,
        eps=1e-20,
        yOut=None,
        expertIdxOut=None,
        outOut=None,
        *args,
        **kwargs,
    ):
        ori_dtype = x.dtype
        x = x.to(torch.float32)
        if biasOptional is not None:
            biasOptional = biasOptional.to(torch.float32)

        if normType == 0:
            x = _softmax_torch(x, -1)
        elif normType == 1:
            x = 1 / (1 + torch.exp(-x))
        elif normType == 2:
            x = torch.sqrt(torch.nn.functional.softplus(x))

        original_x = x
        if biasOptional is not None:
            x = x + biasOptional

        hashFlag = inputIdsOptional is not None and tid2eidOptional is not None
        if hashFlag:
            indices = tid2eidOptional[inputIdsOptional].to(torch.int64)
        else:
            if groupCount > 1:
                x_reshaped = x.reshape(x.shape[0], groupCount, -1)
                if groupSelectMode == 0:
                    group_x = torch.amax(x_reshaped, dim=-1)
                else:
                    top2 = torch.topk(x_reshaped, 2, dim=-1).values
                    group_x = top2.sum(dim=-1)
                _, group_indices = torch.sort(
                    group_x, dim=-1, descending=True, stable=True
                )
                group_indices = group_indices[:, :kGroup]

                mask = torch.ones((x_reshaped.shape[0], groupCount), dtype=torch.bool)
                mask[torch.arange(x_reshaped.shape[0])[:, None], group_indices] = False
                x = torch.where(mask.unsqueeze(-1), float("-inf"), x_reshaped)
                x = x.reshape(x.shape[0], -1)

            _, indices = torch.sort(x, dim=-1, stable=True, descending=True)
            indices = indices[:, :k].to(torch.int64)

        y = torch.gather(original_x, 1, indices)

        if normType != 0 or renorm != 0:
            y = y / (y.sum(dim=-1, keepdim=True) + eps)
        y = y * routedScalingFactor

        if outFlag:
            out = original_x
        else:
            out = None

        return [y.to(ori_dtype), None, out]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    pre_compare = _pre_compare_topk


class MoeGatingTopKTestSpec:
    @staticmethod
    def golden(
        x,
        bias,
        input_ids,
        tid2eid,
        *,
        k,
        k_group=1,
        group_count=1,
        group_select_mode=0,
        renorm=0,
        norm_type=0,
        out_flag=False,
        routed_scaling_factor=1.0,
        eps=1e-20,
        **kwargs,
    ):
        ori_dtype = x.dtype
        x = x.astype(numpy.float32)
        if bias is not None:
            bias = bias.astype(numpy.float32)

        if norm_type == 0:
            x = _softmax_numpy(x, -1)
        elif norm_type == 1:
            x = 1 / (1 + numpy.exp(-x))
        elif norm_type == 2:
            # 数值稳定 softplus: sqrt(max(x,0) + log1p(exp(-|x|))), 避免大 |x| 溢出/吸收
            x = numpy.sqrt(
                numpy.maximum(x, 0.0) + numpy.log1p(numpy.exp(-numpy.abs(x)))
            )

        original_x = x
        if bias is not None:
            x = x + bias

        hashFlag = input_ids is not None and tid2eid is not None
        if hashFlag:
            indices = tid2eid[input_ids]
        else:
            if group_count > 1:
                x = x.reshape(x.shape[0], group_count, -1)
                if group_select_mode == 0:
                    group_x = numpy.amax(x, axis=-1)
                else:
                    group_x = numpy.partition(x, -2, axis=-1)[..., -2:].sum(axis=-1)
                indices = numpy.argsort(-group_x, axis=-1, kind="stable")[:, :k_group]

                mask = numpy.ones((x.shape[0], group_count), dtype=bool)
                mask[numpy.arange(x.shape[0])[:, None], indices] = False
                x = numpy.where(mask[..., None], float("-inf"), x)
                x = x.reshape(x.shape[0], -1)

            _, indices = torch.sort(
                torch.from_numpy(x), dim=-1, stable=True, descending=True
            )
            indices = numpy.asarray(indices[:, :k])

        y = numpy.take_along_axis(original_x, indices, axis=1)

        if norm_type != 0 or renorm != 0:
            y = y / (numpy.sum(y, axis=-1, keepdims=True) + eps)
        y = y * routed_scaling_factor

        if out_flag:
            out = original_x.astype(numpy.float32)
        else:
            out = None

        return [y.astype(ori_dtype), None, out]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    pre_compare = _pre_compare_topk


class E2eMoeGatingTopKTestSpec:
    """E2E spec for torch_npu.npu_moe_gating_top_k — snake_case params, torch CPU golden.

    Signature mirrors the torch_npu API: golden(x, k, bias=..., input_ids=...,
    tid2eid=..., k_group=..., ...). Inputs/outputs are torch tensors; the golden
    runs the norm -> (group select ->) topk / hash lookup -> gather -> renorm ->
    scale pipeline in float32 and returns [yOut, expertIdxOut, normOut], where
    normOut is None when out_flag is False (content not guaranteed by the API).
    """

    @staticmethod
    def golden(
        x,
        k,
        bias=None,
        input_ids=None,
        tid2eid=None,
        k_group=1,
        group_count=1,
        group_select_mode=0,
        renorm=0,
        norm_type=0,
        out_flag=False,
        routed_scaling_factor=1.0,
        eps=1e-20,
        **kwargs,
    ):
        ori_dtype = x.dtype
        x = x.to(torch.float32)
        if bias is not None:
            bias = bias.to(torch.float32)

        if norm_type == 0:
            x = _softmax_torch(x, -1)
        elif norm_type == 1:
            x = 1 / (1 + torch.exp(-x))
        elif norm_type == 2:
            x = torch.sqrt(torch.nn.functional.softplus(x))

        original_x = x
        if bias is not None:
            x = x + bias

        hash_flag = input_ids is not None and tid2eid is not None
        if hash_flag:
            indices = tid2eid[input_ids.to(torch.int64)].to(torch.int64)
        else:
            if group_count > 1:
                x_reshaped = x.reshape(x.shape[0], group_count, -1)
                if group_select_mode == 0:
                    group_x = torch.amax(x_reshaped, dim=-1)
                else:
                    top2 = torch.topk(x_reshaped, 2, dim=-1).values
                    group_x = top2.sum(dim=-1)
                _, group_indices = torch.sort(
                    group_x, dim=-1, descending=True, stable=True
                )
                group_indices = group_indices[:, :k_group]

                mask = torch.ones((x_reshaped.shape[0], group_count), dtype=torch.bool)
                mask[torch.arange(x_reshaped.shape[0])[:, None], group_indices] = False
                x = torch.where(mask.unsqueeze(-1), float("-inf"), x_reshaped)
                x = x.reshape(x.shape[0], -1)

            _, indices = torch.sort(x, dim=-1, stable=True, descending=True)
            indices = indices[:, :k].to(torch.int64)

        y = torch.gather(original_x, 1, indices)

        if norm_type != 0 or renorm != 0:
            y = y / (y.sum(dim=-1, keepdim=True) + eps)
        y = y * routed_scaling_factor

        if out_flag:
            out = original_x.to(torch.float32)
        else:
            out = None

        return [y.to(ori_dtype), indices.to(torch.int32), out]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    pre_compare = _pre_compare_topk
