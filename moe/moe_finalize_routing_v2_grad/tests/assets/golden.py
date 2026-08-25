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
    "moe_finalize_routing_v2_grad": "MoeFinalizeRoutingV2GradTestSpec",
    "aclnnMoeFinalizeRoutingV2Grad": "AclnnMoeFinalizeRoutingV2GradTestSpec",
}


class AclnnMoeFinalizeRoutingV2GradTestSpec:
    @staticmethod
    def golden(*args, **kwargs):
        gradY = args[0] if len(args) > 0 else kwargs.get("gradY")
        expandedRowIdx = args[1] if len(args) > 1 else kwargs.get("expandedRowIdx")
        expandedXOptional = (
            args[2] if len(args) > 2 else kwargs.get("expandedXOptional", None)
        )
        scalesOptional = (
            args[3] if len(args) > 3 else kwargs.get("scalesOptional", None)
        )
        expertIdxOptional = (
            args[4] if len(args) > 4 else kwargs.get("expertIdxOptional", None)
        )
        biasOptional = args[5] if len(args) > 5 else kwargs.get("biasOptional", None)
        dropPadMode = args[6] if len(args) > 6 else kwargs.get("dropPadMode", 0)
        activeNum = args[7] if len(args) > 7 else kwargs.get("activeNum", 0)
        expertNum = args[8] if len(args) > 8 else kwargs.get("expertNum", 0)
        expertCapacity = args[9] if len(args) > 9 else kwargs.get("expertCapacity", 0)

        if (
            expandedXOptional is not None
            and hasattr(expandedXOptional, "shape")
            and expandedXOptional.shape is not None
            and expandedXOptional.shape[0] == 0
        ):
            expandedXOptional = None
        if (
            scalesOptional is not None
            and hasattr(scalesOptional, "shape")
            and scalesOptional.shape is not None
            and scalesOptional.shape[0] == 0
        ):
            scalesOptional = None
        grad_y_dtype = gradY.dtype
        grad_y = gradY.to(torch.float32)
        BS, H = grad_y.shape
        BSK = expandedRowIdx.shape[0]
        eri = expandedRowIdx.to(torch.int64)

        topK = 1
        if scalesOptional is not None and scalesOptional.dim() >= 2:
            topK = scalesOptional.shape[1]
        N = BSK // topK

        if dropPadMode == 1:
            if expandedXOptional is not None and expandedXOptional.dim() == 3:
                E, C = expandedXOptional.shape[0], expandedXOptional.shape[1]
                H = expandedXOptional.shape[2]
            else:
                E = expertNum if expertNum > 0 else 1
                C = expertCapacity if expertCapacity > 0 else 1
            out_rows = E * C
            out_shape = (E, C, H)
        else:
            if activeNum > 0 and activeNum < BSK:
                out_rows = activeNum
            else:
                out_rows = BSK
            out_shape = (out_rows, H)

        if scalesOptional is None:
            grad_expanded_x = torch.zeros((out_rows, H), dtype=torch.float32)
            for i in range(BSK):
                src_row = eri[i].item()
                if 0 <= src_row < out_rows:
                    grad_row_idx = i // topK
                    grad_expanded_x[src_row] = grad_y[grad_row_idx]
            grad_expanded_x = grad_expanded_x.reshape(out_shape)
            grad_scales = torch.zeros((BSK, 1), dtype=grad_y_dtype)
            return [grad_expanded_x.to(grad_y_dtype), grad_scales]
        else:
            scales_dtype = scalesOptional.dtype
            scales = scalesOptional.reshape(BSK, topK).to(torch.float32)
            grad_expanded_x = torch.zeros((out_rows, H), dtype=torch.float32)
            grad_scales = torch.zeros((BSK, topK), dtype=torch.float32)

            expanded_x = (
                expandedXOptional.to(torch.float32)
                if expandedXOptional is not None
                else None
            )
            if expanded_x is not None and expanded_x.dim() == 3:
                expanded_x = expanded_x.reshape(-1, H)

            for i in range(BSK):
                src_row = eri[i].item()
                k_idx = i % topK
                grad_row_idx = i // topK
                if 0 <= src_row < out_rows:
                    scale_val = scales[grad_row_idx, k_idx]
                    grad_expanded_x[src_row] = grad_y[grad_row_idx] * scale_val
                    if expanded_x is not None:
                        grad_scales[grad_row_idx, k_idx] = torch.sum(
                            expanded_x[src_row] * grad_y[grad_row_idx]
                        )
                    else:
                        grad_scales[grad_row_idx, k_idx] = 0.0
            grad_expanded_x = grad_expanded_x.reshape(out_shape)
            return [
                grad_expanded_x.to(grad_y_dtype),
                grad_scales.reshape(N, topK).to(scales_dtype),
            ]

    @staticmethod
    def customize_inputs(*args, **kwargs):
        expandedRowIdx = args[1] if len(args) > 1 else kwargs.get("expandedRowIdx")
        dropPadMode = args[6] if len(args) > 6 else kwargs.get("dropPadMode", 0)
        activeNum = args[7] if len(args) > 7 else kwargs.get("activeNum", 0)
        expertNum = args[8] if len(args) > 8 else kwargs.get("expertNum", 0)
        expertCapacity = args[9] if len(args) > 9 else kwargs.get("expertCapacity", 0)
        BSK = expandedRowIdx.shape[0]

        if dropPadMode == 1:
            E = expertNum if expertNum > 0 else 1
            C = expertCapacity if expertCapacity > 0 else 1
            out_rows = E * C
        else:
            out_rows = activeNum if activeNum > 0 else BSK

        new_eri = torch.full((BSK,), -1, dtype=torch.int32)
        perm = torch.randperm(min(out_rows, BSK), dtype=torch.int32)
        new_eri[: len(perm)] = perm
        expandedRowIdx.copy_(new_eri)
        for idx in [10, 11]:
            if (
                len(args) > idx
                and args[idx] is not None
                and hasattr(args[idx], "zero_")
            ):
                args[idx].zero_()

    def pre_compare(*outputs, **kwargs):
        import numpy as _np

        try:
            for i in range(0, len(outputs) - 1, 2):
                npu_out = outputs[i]
                golden_out = outputs[i + 1]
                if npu_out is None or golden_out is None:
                    continue
                if not isinstance(npu_out, _np.ndarray) or not isinstance(
                    golden_out, _np.ndarray
                ):
                    continue
                if npu_out.shape != golden_out.shape:
                    continue
                mask = golden_out == 0
                if mask.any():
                    npu_out[mask] = 0
        except Exception:
            pass

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


class MoeFinalizeRoutingV2GradTestSpec:
    @staticmethod
    def golden(
        grad_y,
        expanded_row_idx,
        expanded_x=None,
        scales=None,
        expert_idx=None,
        bias=None,
        *,
        drop_pad_mode=0,
        active_num=0,
        expert_num=0,
        expert_capacity=0,
        **kwargs,
    ):
        ori_dtype = grad_y.dtype
        grad_y = grad_y.astype(numpy.float32)
        BS, H = grad_y.shape
        BSK = expanded_row_idx.shape[0]

        if drop_pad_mode == 1:
            if expanded_x is not None and expanded_x.ndim == 3:
                E, C = expanded_x.shape[0], expanded_x.shape[1]
                H = expanded_x.shape[2]
            else:
                E = expert_num if expert_num > 0 else 1
                C = expert_capacity if expert_capacity > 0 else 1
            out_rows = E * C
            out_shape = (E, C, H)
        else:
            out_rows = active_num if active_num > 0 else BSK
            out_shape = (out_rows, H)

        if scales is None:
            grad_expanded_x = numpy.zeros((out_rows, H), dtype=numpy.float32)
            for i in range(BSK):
                src_row = int(expanded_row_idx[i])
                if 0 <= src_row < out_rows:
                    grad_expanded_x[src_row] = grad_y[i]
            grad_expanded_x = grad_expanded_x.reshape(out_shape)
            grad_scales = numpy.zeros((BSK, 1), dtype=ori_dtype)
            return [grad_expanded_x.astype(ori_dtype), grad_scales]
        else:
            topK = 1
            if scales.ndim >= 2:
                topK = scales.shape[1]
            N = BSK // topK
            scales_np = scales.reshape(N, topK).astype(numpy.float32)
            grad_expanded_x = numpy.zeros((out_rows, H), dtype=numpy.float32)
            grad_scales = numpy.zeros((N, topK), dtype=numpy.float32)

            expanded_x_np = (
                expanded_x.astype(numpy.float32) if expanded_x is not None else None
            )
            if expanded_x_np is not None and expanded_x_np.ndim == 3:
                expanded_x_np = expanded_x_np.reshape(-1, H)
            for i in range(BSK):
                src_row = int(expanded_row_idx[i])
                k_idx = i % topK
                grad_row_idx = i // topK
                if 0 <= src_row < out_rows:
                    scale_val = scales_np[grad_row_idx, k_idx]
                    grad_expanded_x[src_row] = grad_y[grad_row_idx] * scale_val
                    if expanded_x_np is not None:
                        grad_scales[grad_row_idx, k_idx] = numpy.sum(
                            expanded_x_np[src_row] * grad_y[grad_row_idx]
                        )
                    else:
                        grad_scales[grad_row_idx, k_idx] = 0.0
            grad_expanded_x = grad_expanded_x.reshape(out_shape)
            return [
                grad_expanded_x.astype(ori_dtype),
                grad_scales.reshape(BSK, 1).astype(scales.dtype),
            ]

    @staticmethod
    def customize_inputs(
        grad_y,
        expanded_row_idx,
        expanded_x=None,
        scales=None,
        expert_idx=None,
        bias=None,
        **kwargs,
    ):
        BSK = expanded_row_idx.shape[0]
        drop_pad_mode = kwargs.get("drop_pad_mode", 0)
        active_num = kwargs.get("active_num", 0)
        expert_num = kwargs.get("expert_num", 0)
        expert_capacity = kwargs.get("expert_capacity", 0)

        if drop_pad_mode == 1:
            E = expert_num if expert_num > 0 else 1
            C = expert_capacity if expert_capacity > 0 else 1
            out_rows = E * C
        else:
            out_rows = active_num if active_num > 0 else BSK

        new_eri = numpy.full(BSK, -1, dtype=numpy.int32)
        perm = numpy.random.permutation(min(out_rows, BSK)).astype(numpy.int32)
        new_eri[: len(perm)] = perm
        return (grad_y, new_eri, expanded_x, scales, expert_idx, bias)

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
