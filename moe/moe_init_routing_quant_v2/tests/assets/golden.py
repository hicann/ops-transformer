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
    "moe_init_routing_quant_v2": "MoeInitRoutingQuantV2TestSpec",
    "aclnnMoeInitRoutingQuantV2": "AclnnMoeInitRoutingQuantV2TestSpec",
}


_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
    "int8": {"standard": "quant", "ptol": 0.001},
}


def _torch_dynamic_scale(
    x,
    expert_idx,
    scale,
    active_num,
    expert_capacity,
    expert_num,
    drop_pad_mode,
    quant_mode,
):
    if quant_mode == 0:
        return [None, None, None, None, None]
    rows, hidden_size = x.shape[0], x.shape[-1]
    topk = expert_idx.shape[-1]
    flat_experts = expert_idx.reshape(-1).to(torch.int64)
    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[order]
    if drop_pad_mode == 0:
        count = rows * topk if active_num == 0 else min(int(active_num), rows * topk)
        selected_experts = sorted_experts[:count]
        values = x[order[:count] // topk].to(torch.float32)
        if scale is not None:
            values = values * (
                scale if scale.shape[0] == 1 else scale[selected_experts]
            ).to(torch.float32)
    else:
        slots = int(expert_num) * int(expert_capacity)
        values = torch.zeros((slots, hidden_size), dtype=torch.float32, device=x.device)
        if order.numel() and slots:
            positions = torch.arange(order.numel(), device=x.device)
            starts = torch.ones_like(sorted_experts, dtype=torch.bool)
            starts[1:] = sorted_experts[1:] != sorted_experts[:-1]
            offsets = torch.where(starts, positions, 0)
            ranks = positions - torch.cummax(offsets, dim=0).values
            valid = (
                (sorted_experts >= 0)
                & (sorted_experts < int(expert_num))
                & (ranks < int(expert_capacity))
            )
            target = sorted_experts[valid] * int(expert_capacity) + ranks[valid]
            source = order[valid] // topk
            selected = x[source].to(torch.float32)
            if scale is not None:
                selected = selected * (
                    scale if scale.shape[0] == 1 else scale[sorted_experts[valid]]
                ).to(torch.float32)
            values[target] = selected
    dynamic_scale = torch.amax(torch.abs(values), dim=-1, keepdim=True) / 127
    return [None, None, None, None, dynamic_scale]


class _KernelTorchImpl:
    def __call__(
        self,
        x,
        expert_idx,
        scale,
        offset,
        *,
        active_num=0,
        expert_capacity=0,
        expert_num=0,
        drop_pad_mode=0,
        quant_mode=0,
        **kwargs,
    ):
        return _torch_dynamic_scale(
            x,
            expert_idx,
            scale,
            active_num,
            expert_capacity,
            expert_num,
            drop_pad_mode,
            quant_mode,
        )


class _AclnnTorchImpl:
    def __call__(
        self,
        x,
        expertIdx,
        scale,
        offset,
        activeNum=0,
        expertCapacity=0,
        expertNum=0,
        dropPadMode=0,
        quantMode=0,
        *args,
        **kwargs,
    ):
        return _torch_dynamic_scale(
            x,
            expertIdx,
            scale,
            activeNum,
            expertCapacity,
            expertNum,
            dropPadMode,
            quantMode,
        )


class AclnnMoeInitRoutingQuantV2TestSpec:
    third_party = {"torch": _AclnnTorchImpl}
    tolerance = _TOLERANCE

    @staticmethod
    def golden(
        x,
        expertIdx,
        scaleOptional=None,
        offsetOptional=None,
        activeNum=0,
        expertCapacity=0,
        expertNum=0,
        dropPadMode=0,
        expertTokensCountOrCumsumFlag=0,
        expertTokensBeforeCapacityFlag=False,
        quantMode=0,
        *args,
        **kwargs,
    ):
        input_x = x
        expert_idx = expertIdx
        scale = scaleOptional
        offset = offsetOptional

        num_rows = input_x.shape[0]
        hidden_size = input_x.shape[-1]
        k = expert_idx.shape[-1]

        sorted_row_idx = torch.argsort(expert_idx.reshape(-1), stable=True)
        sorted_expert_idx = torch.sort(expert_idx.reshape(-1)).values

        if dropPadMode == 1 and expertNum <= 0:
            return [None, None, None, None, None]

        expert_tokens_count_or_cumsum = None
        expert_tokens_before_capacity = None

        if expertTokensBeforeCapacityFlag or expertTokensCountOrCumsumFlag > 0:
            expert_idx_hist = torch.bincount(sorted_expert_idx, minlength=expertNum)
            expert_token_idx = torch.cumsum(expert_idx_hist, dim=0)
            if dropPadMode == 1 and expertTokensBeforeCapacityFlag:
                expert_tokens_before_capacity = expert_idx_hist.to(torch.int32)
            if dropPadMode == 0:
                if expertTokensCountOrCumsumFlag == 1:
                    expert_tokens_count_or_cumsum = expert_token_idx.to(torch.int32)
                elif expertTokensCountOrCumsumFlag == 2:
                    expert_tokens_count_or_cumsum = expert_idx_hist.to(torch.int32)

        if dropPadMode == 0:
            expanded_row_idx = torch.zeros_like(sorted_row_idx, dtype=torch.int32)
            expanded_row_idx[sorted_row_idx] = torch.arange(
                sorted_row_idx.shape[-1], dtype=torch.int32, device=input_x.device
            )
            active_num = (
                num_rows * k if activeNum == 0 else min(activeNum, num_rows * k)
            )
            expanded_x = input_x[sorted_row_idx[:active_num] // k]
        else:
            sort_row_tmp = torch.full(
                (expertNum * expertCapacity,),
                -1,
                dtype=torch.int32,
                device=input_x.device,
            )
            offset_tmp = 0
            last_expert_id = -1
            for i, val in enumerate(sorted_row_idx):
                if val != -1:
                    current_expert_id = sorted_expert_idx[i]
                    if last_expert_id != current_expert_id:
                        offset_tmp = 0
                        last_expert_id = current_expert_id
                    idx = current_expert_id * expertCapacity + offset_tmp
                    if idx < sort_row_tmp.numel():
                        sort_row_tmp[idx] = val
                        offset_tmp += 1

            expanded_row_idx = torch.full_like(sorted_row_idx, -1)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_row_idx[val] = i

            expanded_x_mask = torch.ones(
                (expertNum * expertCapacity, hidden_size),
                dtype=torch.int32,
                device=input_x.device,
            )
            expanded_x = torch.zeros(
                (expertNum * expertCapacity, hidden_size),
                dtype=input_x.dtype,
                device=input_x.device,
            )
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_x[i] = input_x[val // k]
                    expanded_x_mask[i] = 0

        dynamic_scale = None
        if quantMode == 0:
            expanded_x_fp16 = expanded_x.to(torch.float16)
            scale_val = scale.to(torch.float16) if scale is not None else None
            offset_val = offset.to(torch.float16) if offset is not None else None
            scale_rst = expanded_x_fp16 * scale_val[0]
            add_offset = scale_rst + offset_val[0]
            round_data = torch.round(add_offset)
            round_data = torch.clamp(round_data, -128, 127)
            expanded_x = round_data.to(torch.int8)
        else:
            x_final = expanded_x.to(torch.float32)
            if scale is not None:
                if scale.shape[0] == 1:
                    x_final = x_final * scale
                else:
                    if dropPadMode == 0:
                        x_final = x_final * scale[sorted_expert_idx[:active_num]]
                    else:
                        for i, val in enumerate(sort_row_tmp):
                            if val != -1:
                                x_final[i] = x_final[i] * scale[i // expertCapacity]
            x_abs = torch.abs(x_final)
            x_max = torch.amax(x_abs, dim=-1, keepdim=True)
            dynamic_scale = x_max / 127
            expanded_x = x_final / dynamic_scale
            expanded_x = torch.round(expanded_x).to(torch.int8)

        if dropPadMode == 1:
            mask_bool = expanded_x_mask.bool()
            expanded_x = expanded_x.masked_fill(mask_bool, 0)
            expanded_x = expanded_x.reshape(expertNum, expertCapacity, hidden_size)

        return [
            expanded_x,
            expanded_row_idx.to(torch.int32),
            expert_tokens_count_or_cumsum,
            expert_tokens_before_capacity,
            dynamic_scale,
        ]

    @staticmethod
    def customize_inputs(x, expertIdx, *args, **kwargs):
        expertNum = kwargs.get("expertNum", 0)
        if not expertNum and len(args) >= 3:
            expertNum = args[2]
        if not expertNum or expertNum <= 0:
            expertNum = 100
        expertIdx.copy_(torch.randint(0, expertNum, expertIdx.shape, dtype=torch.int32))


class MoeInitRoutingQuantV2TestSpec:
    third_party = {"torch": _KernelTorchImpl}
    tolerance = _TOLERANCE

    @staticmethod
    def golden(
        x,
        expert_idx,
        scale,
        offset,
        *,
        active_num=0,
        expert_capacity=0,
        expert_num=0,
        drop_pad_mode=0,
        expert_tokens_count_or_cumsum_flag=0,
        expert_tokens_before_capacity_flag=False,
        quant_mode=0,
        **kwargs,
    ):
        ori_dtype = x.dtype
        input_x = x.astype(numpy.float32)
        num_rows = input_x.shape[0]
        hidden_size = input_x.shape[-1]
        k = expert_idx.shape[-1]

        sorted_row_idx = numpy.argsort(
            expert_idx.reshape((-1,)), axis=-1, kind="stable"
        )
        sorted_expert_idx = numpy.sort(expert_idx.reshape((-1,)), axis=-1)

        if drop_pad_mode == 1 and expert_num <= 0:
            return [None, None, None, None, None]

        expert_tokens_count_or_cumsum = None
        expert_tokens_before_capacity = None

        if expert_tokens_before_capacity_flag or expert_tokens_count_or_cumsum_flag > 0:
            expert_idx_hist = numpy.bincount(sorted_expert_idx, minlength=expert_num)
            expert_token_idx = numpy.cumsum(expert_idx_hist)
            if drop_pad_mode == 1 and expert_tokens_before_capacity_flag:
                expert_tokens_before_capacity = expert_idx_hist.astype(numpy.int32)
            if drop_pad_mode == 0:
                if expert_tokens_count_or_cumsum_flag == 1:
                    expert_tokens_count_or_cumsum = expert_token_idx.astype(numpy.int32)
                elif expert_tokens_count_or_cumsum_flag == 2:
                    expert_tokens_count_or_cumsum = expert_idx_hist.astype(numpy.int32)

        if drop_pad_mode == 0:
            expanded_row_idx = numpy.zeros(sorted_row_idx.shape, dtype=numpy.int32)
            expanded_row_idx[sorted_row_idx] = numpy.arange(
                sorted_row_idx.shape[-1], dtype=numpy.int32
            )
            active_num = (
                num_rows * k if active_num == 0 else min(active_num, num_rows * k)
            )
            expanded_x = input_x[sorted_row_idx[:active_num] // k]
        else:
            sort_row_tmp = numpy.full((expert_num * expert_capacity), -1, dtype=int)
            offset_tmp = 0
            last_expert_id = -1
            for i, val in enumerate(sorted_row_idx):
                if val != -1:
                    current_expert_id = sorted_expert_idx[i]
                    if last_expert_id != current_expert_id:
                        offset_tmp = 0
                        last_expert_id = current_expert_id
                    idx = current_expert_id * expert_capacity + offset_tmp
                    if idx < sort_row_tmp.shape[0]:
                        sort_row_tmp[idx] = val
                        offset_tmp += 1

            expanded_row_idx = numpy.full(sorted_row_idx.shape, -1)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_row_idx[val] = i

            expanded_x_mask = numpy.ones(
                (expert_num * expert_capacity, hidden_size), dtype=numpy.int32
            )
            expanded_x = numpy.zeros(
                (expert_num * expert_capacity, hidden_size), dtype=input_x.dtype
            )
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_x[i] = input_x[val // k]
                    expanded_x_mask[i] = 0

        dynamic_scale = None
        if quant_mode == 0:
            expanded_x_fp16 = expanded_x.astype(numpy.float16)
            scale_val = scale.astype(numpy.float16) if scale is not None else None
            offset_val = offset.astype(numpy.float16) if offset is not None else None
            scale_rst = expanded_x_fp16 * scale_val[0]
            add_offset = scale_rst + offset_val[0]
            round_data = numpy.round(add_offset)
            round_data = numpy.clip(round_data, -128, 127)
            expanded_x = round_data.astype(numpy.int8)
        else:
            x_final = expanded_x.astype(numpy.float32)
            if scale is not None:
                if scale.shape[0] == 1:
                    x_final = x_final * scale
                else:
                    if drop_pad_mode == 0:
                        x_final = x_final * scale[sorted_expert_idx[:active_num]]
                    else:
                        for i, val in enumerate(sort_row_tmp):
                            if val != -1:
                                x_final[i] = x_final[i] * scale[i // expert_capacity]
            x_abs = numpy.abs(x_final)
            x_max = numpy.amax(x_abs, axis=-1, keepdims=True)
            dynamic_scale = x_max / 127
            expanded_x = x_final / dynamic_scale
            expanded_x = numpy.round(expanded_x).astype(numpy.int8)

        if drop_pad_mode == 1:
            mask_bool = expanded_x_mask.astype(bool)
            expanded_x = numpy.where(mask_bool, 0, expanded_x)
            expanded_x = expanded_x.reshape(expert_num, expert_capacity, hidden_size)

        return [
            expanded_x,
            expanded_row_idx.astype(numpy.int32),
            expert_tokens_count_or_cumsum,
            expert_tokens_before_capacity,
            dynamic_scale,
        ]

    @staticmethod
    def customize_inputs(x, expert_idx, scale=None, offset=None, *args, **kwargs):
        expert_num = kwargs.get("expert_num", 0)
        if not expert_num or expert_num <= 0:
            expert_num = 100
        expert_idx = numpy.random.randint(
            0, expert_num, size=expert_idx.size, dtype=numpy.int32
        ).reshape(expert_idx.shape)
        return (x, expert_idx, scale, offset)
