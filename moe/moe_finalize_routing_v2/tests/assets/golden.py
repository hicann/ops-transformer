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
    "moe_finalize_routing_v2": "MoeFinalizeRoutingV2TestSpec",
    "aclnnMoeFinalizeRoutingV2": "AclnnMoeFinalizeRoutingV2TestSpec",
}


def _to_numpy(value):
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu()
        try:
            return value.numpy()
        except TypeError:
            import ml_dtypes

            return value.to(torch.float32).numpy().astype(ml_dtypes.bfloat16)
    return value


def _restore_type(value, template):
    if (
        template is not None
        and hasattr(template, "cpu")
        and hasattr(template, "numpy")
        and value is not None
    ):
        try:
            return torch.from_numpy(value)
        except TypeError:
            return torch.from_numpy(value.astype(numpy.float32)).to(torch.bfloat16)
    return value


# Default to local comparison; --compare cross_check selects the L2 preset.
_TOLERANCE = {
    "float16": {"standard": "stat_rel_err", "level": "L2"},
    "bfloat16": {"standard": "stat_rel_err", "level": "L2"},
    "float32": {"standard": "stat_rel_err", "level": "L2"},
}


class _MoeFinalizeRoutingV2TorchImpl:
    """Independent Torch reference used by the remote GPU cross-check."""

    def __call__(
        self,
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x=None,
        alpha1=None,
        alpha2=None,
        v=None,
        *,
        drop_pad_mode=0,
        zero_expert_range=None,
        copy_expert_range=None,
        constant_expert_range=None,
        k=1,
        **kwargs,
    ):
        zero_expert_range = zero_expert_range or []
        copy_expert_range = copy_expert_range or []
        constant_expert_range = constant_expert_range or []
        output_dtype = expanded_x.dtype
        compute_dtype = (
            torch.float64 if output_dtype == torch.float32 else torch.float32
        )
        bsk = expanded_row_idx.shape[0]
        hidden_size = expanded_x.shape[-1]
        if hidden_size == 0:
            return [torch.empty((0,), dtype=output_dtype, device=expanded_x.device)]

        expanded_x_2d = expanded_x.reshape(-1, hidden_size)
        topk = scales.shape[1] if scales is not None else int(k)
        num_rows = bsk // topk
        out = torch.zeros(
            (num_rows, hidden_size), dtype=compute_dtype, device=expanded_x.device
        )
        if x1 is not None:
            out += x1.to(compute_dtype)
        if x2 is not None:
            out += x2.to(compute_dtype)

        for row in range(num_rows):
            for topk_idx in range(topk):
                if drop_pad_mode in (0, 1):
                    route_pos = topk_idx * num_rows + row
                else:
                    route_pos = row * topk + topk_idx
                expanded_idx = int(expanded_row_idx[route_pos].item())
                if drop_pad_mode in (1, 3):
                    if expanded_idx == -1:
                        continue
                elif expanded_idx >= expanded_x_2d.shape[0]:
                    continue

                if x is not None and expert_idx is not None:
                    zero_start, zero_end = zero_expert_range
                    copy_start, copy_end = copy_expert_range
                    const_start, const_end = constant_expert_range
                    expert_id = int(expert_idx[row, topk_idx].item())
                    if zero_start <= expert_id < zero_end:
                        continue
                    if copy_start <= expert_id < copy_end:
                        dst_row = x[row].to(compute_dtype)
                    elif (
                        v is not None
                        and alpha1 is not None
                        and alpha2 is not None
                        and const_start <= expert_id < const_end
                    ):
                        const_row = expert_id - const_start
                        dst_row = (
                            alpha1[const_row] * x[row]
                            + alpha2[const_row] * v[const_row]
                        ).to(compute_dtype)
                    else:
                        dst_row = expanded_x_2d[expanded_idx].to(compute_dtype)
                else:
                    dst_row = expanded_x_2d[expanded_idx].to(compute_dtype)

                if bias is not None and expert_idx is not None:
                    expert_id = int(expert_idx[row, topk_idx].item())
                    if expert_id < 0 or expert_id >= bias.shape[0]:
                        continue
                    dst_row = dst_row + bias[expert_id].to(compute_dtype)
                if scales is not None:
                    dst_row = dst_row * scales[row, topk_idx].to(compute_dtype)
                out[row] += dst_row

        return [out.to(output_dtype)]


class _AclnnMoeFinalizeRoutingV2TorchImpl(_MoeFinalizeRoutingV2TorchImpl):
    def __call__(
        self,
        expandedX,
        expandedRowIdx,
        x1Optional=None,
        x2Optional=None,
        biasOptional=None,
        scalesOptional=None,
        expertIdxOptional=None,
        out=None,
        *,
        dropPadMode=0,
        k=1,
        **kwargs,
    ):
        return super().__call__(
            expandedX,
            expandedRowIdx,
            x1Optional,
            x2Optional,
            biasOptional,
            scalesOptional,
            expertIdxOptional,
            drop_pad_mode=dropPadMode,
            k=k,
        )


class MoeFinalizeRoutingV2TestSpec:
    third_party = {"torch": _MoeFinalizeRoutingV2TorchImpl}
    tolerance = _TOLERANCE

    @staticmethod
    def golden(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x=None,
        alpha1=None,
        alpha2=None,
        v=None,
        *,
        drop_pad_mode=0,
        zero_expert_range=None,
        copy_expert_range=None,
        constant_expert_range=None,
        k=1,
        **kwargs,
    ):
        if zero_expert_range is None:
            zero_expert_range = []
        if copy_expert_range is None:
            copy_expert_range = []
        if constant_expert_range is None:
            constant_expert_range = []

        ori_dtype = expanded_x.dtype
        out_type = numpy.float32
        if expanded_x.dtype == numpy.float32:
            out_type = numpy.float64

        bsk = expanded_row_idx.shape[0]
        h = expanded_x.shape[-1]
        if h == 0:
            return [numpy.array([], dtype=out_type)]
        expanded_x = expanded_x.reshape(-1, h)
        K = k
        if scales is not None:
            K = scales.shape[1]
        num_rows = bsk // K

        out = numpy.zeros((num_rows, h), dtype=out_type)
        if x1 is not None:
            out = out + x1
        if x2 is not None:
            out = out + x2

        for i in range(num_rows):
            for kk in range(K):
                if drop_pad_mode == 0 or drop_pad_mode == 1:
                    expanded_row_idx_idx = kk * num_rows + i
                else:
                    expanded_row_idx_idx = i * K + kk
                expanded_row_idx_value = expanded_row_idx[expanded_row_idx_idx]
                if drop_pad_mode == 1 or drop_pad_mode == 3:
                    if expanded_row_idx_value == -1:
                        continue
                else:
                    if expanded_row_idx_value >= expanded_x.shape[0]:
                        continue

                if x is not None and expert_idx is not None:
                    zero_start, zero_end = zero_expert_range
                    copy_start, copy_end = copy_expert_range
                    const_start, const_end = constant_expert_range
                    expert_id = expert_idx[i, kk]
                    if zero_start <= expert_id < zero_end:
                        continue
                    elif copy_start <= expert_id < copy_end:
                        dst_row = x[i, :].astype(out_type)
                    elif (
                        x is not None
                        and v is not None
                        and alpha1 is not None
                        and alpha2 is not None
                        and const_start <= expert_id < const_end
                    ):
                        row_idx = expert_id - const_start
                        dst_row = (
                            alpha1[row_idx, :] * x[i, :]
                            + alpha2[row_idx, :] * v[row_idx, :]
                        ).astype(out_type)
                    else:
                        dst_row = expanded_x[
                            expanded_row_idx[expanded_row_idx_idx], :
                        ].astype(out_type)
                else:
                    dst_row = expanded_x[
                        expanded_row_idx[expanded_row_idx_idx], :
                    ].astype(out_type)
                if bias is not None and expert_idx is not None:
                    expert_id = expert_idx[i, kk]
                    if expert_id < 0 or expert_id >= bias.shape[0]:
                        continue
                    dst_row = dst_row + bias[expert_id, :].astype(out_type)
                if scales is not None:
                    dst_row = dst_row * scales[i, kk].astype(out_type)
                out[i, :] = out[i, :] + dst_row

        if ori_dtype == numpy.float32:
            return [out.astype(numpy.float32)]
        return [out.astype(ori_dtype)]

    @staticmethod
    def customize_inputs(
        expanded_x,
        expanded_row_idx,
        x1,
        x2,
        bias,
        scales,
        expert_idx,
        x=None,
        alpha1=None,
        alpha2=None,
        v=None,
        **kwargs,
    ):
        drop_pad_mode = kwargs.get("drop_pad_mode", 0)
        expand_n = expanded_row_idx.shape[0]
        expanded_row_idx = numpy.random.choice(
            expand_n, expand_n, replace=False
        ).astype(numpy.int32)
        if drop_pad_mode == 1 or drop_pad_mode == 3:
            e = expanded_x.shape[0]
            c = expanded_x.shape[1] if len(expanded_x.shape) > 2 else expand_n
            expanded_row_idx[expanded_row_idx >= e * c] = -1
        if bias is not None:
            e = bias.shape[0]
            expert_idx = (
                numpy.random.randint(0, e, size=expert_idx.size)
                .reshape(expert_idx.shape)
                .astype(numpy.int32)
            )
        return (
            expanded_x,
            expanded_row_idx,
            x1,
            x2,
            bias,
            scales,
            expert_idx,
            x,
            alpha1,
            alpha2,
            v,
        )


class AclnnMoeFinalizeRoutingV2TestSpec:
    third_party = {"torch": _AclnnMoeFinalizeRoutingV2TorchImpl}
    tolerance = _TOLERANCE

    @staticmethod
    def golden(
        expandedX,
        expandedRowIdx,
        x1=None,
        x2=None,
        bias=None,
        scales=None,
        expertIdx=None,
        dropPadMode=0,
        out=None,
        *args,
        **kwargs,
    ):
        return MoeFinalizeRoutingV2TestSpec.golden(
            _to_numpy(expandedX),
            _to_numpy(expandedRowIdx),
            _to_numpy(x1),
            _to_numpy(x2),
            _to_numpy(bias),
            _to_numpy(scales),
            _to_numpy(expertIdx),
            x=_to_numpy(kwargs.get("x")),
            alpha1=_to_numpy(kwargs.get("alpha1")),
            alpha2=_to_numpy(kwargs.get("alpha2")),
            v=_to_numpy(kwargs.get("v")),
            drop_pad_mode=dropPadMode,
            zero_expert_range=kwargs.get("zeroExpertRange"),
            copy_expert_range=kwargs.get("copyExpertRange"),
            constant_expert_range=kwargs.get("constantExpertRange"),
            k=kwargs.get("k", 1),
        )

    @staticmethod
    def customize_inputs(
        expandedX,
        expandedRowIdx,
        x1=None,
        x2=None,
        bias=None,
        scales=None,
        expertIdx=None,
        dropPadMode=0,
        out=None,
        *args,
        **kwargs,
    ):
        templates = (expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx)
        results = MoeFinalizeRoutingV2TestSpec.customize_inputs(
            *[_to_numpy(t) for t in templates],
            x=None,
            alpha1=None,
            alpha2=None,
            v=None,
            **kwargs,
        )[:7]
        return tuple(
            _restore_type(value, template)
            for value, template in zip(results, templates)
        )
