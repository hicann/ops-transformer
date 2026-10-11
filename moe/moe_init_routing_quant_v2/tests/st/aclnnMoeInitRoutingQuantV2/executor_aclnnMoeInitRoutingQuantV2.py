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
import numpy
from atk.configs.dataset_config import InputDataset

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
    count = 0
    last = sorted_expert_idx[0]
    for i, val in enumerate(sorted_expert_idx):
        if last != val:
            count = 1
            last = val
        else:
            count += 1
            if count > capacity:
                sorted_expert_idx[i] = -1
                sorted_row_idx[i] = -1


@register("function_aclnn_moe_init_routing_quant_v2")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_x = input_data.kwargs.get("x").cpu().numpy()
        expert_idx = input_data.kwargs.get("expert_idx").cpu().numpy()

        scale = input_data.kwargs.get("scale").cpu().numpy()
        offset_t = input_data.kwargs.get("offset").cpu().numpy()

        active_num = input_data.kwargs.get("active_num")
        expert_num = input_data.kwargs.get("expert_num")
        capacity = input_data.kwargs.get("expert_capacity")
        drop_pad_mode = input_data.kwargs.get("drop_pad_mode")

        quant_mode = input_data.kwargs.get("quant_mode")
        expert_tokens_before_capacity_flag = input_data.kwargs.get(
            "expert_tokens_before_capacity_flag"
        )
        expert_tokens_count_or_cumsum_flag = input_data.kwargs.get(
            "expert_tokens_count_or_cumsum_flag"
        )

        num_rows = input_x.shape[0]
        hidden_size = input_x.shape[-1]
        k = expert_idx.shape[-1]
        sorted_row_idx = numpy.argsort(
            expert_idx.reshape((-1,)), axis=-1, kind="stable"
        )
        sorted_expert_idx = numpy.sort(expert_idx.reshape((-1,)), axis=-1)
        if drop_pad_mode == 1 and expert_num <= 0:
            return

        expert_tokens_count_or_cumsum = None
        expert_tokens_before_capacity = None
        # expert_token_idx
        expert_idx_hist, bins = numpy.histogram(
            sorted_expert_idx, bins=expert_num, range=(0, expert_num - 1)
        )
        expert_token_idx = numpy.cumsum(expert_idx_hist)
        if drop_pad_mode == 1 and expert_tokens_before_capacity_flag:
            expert_tokens_before_capacity = expert_idx_hist.astype("int32")
        if drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 1:
            expert_tokens_count_or_cumsum = expert_token_idx.astype("int32")
        elif drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 2:
            expert_tokens_count_or_cumsum = expert_idx_hist.astype("int32")

        if drop_pad_mode == 0:
            expanded_row_idx = numpy.zeros(sorted_row_idx.shape, dtype=numpy.int32)
            expanded_row_idx[sorted_row_idx] = numpy.arange(
                sorted_row_idx.shape[-1], dtype=numpy.int32
            )

            if active_num == 0:
                active_num = num_rows * k
            else:
                active_num = min(active_num, num_rows * k)
            expanded_x = input_x[sorted_row_idx[:active_num] // k, :]
        else:
            adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity)

            sort_row_tmp = numpy.full((expert_num * capacity), -1, dtype=int)
            offset = 0
            lastExpertId = 0
            for i, val in enumerate(sorted_row_idx):
                if val != -1:
                    if lastExpertId != sorted_expert_idx[i]:
                        offset = 0
                        lastExpertId = sorted_expert_idx[i]
                    sort_row_tmp[sorted_expert_idx[i] * capacity + offset] = (
                        sorted_row_idx[i]
                    )
                    offset = offset + 1

            # expand_row_idx
            expanded_row_idx = numpy.full(sorted_row_idx.shape, -1)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_row_idx[val] = i

            # expanded_x
            expanded_x = numpy.full(
                (expert_num * capacity, hidden_size), 0, dtype=input_x.dtype
            )
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_x[i] = input_x[val // k]
            expanded_x = expanded_x.reshape(expert_num, capacity, hidden_size)
        if expert_tokens_count_or_cumsum is None:
            expert_tokens_count_or_cumsum = torch.tensor([])
        else:
            expert_tokens_count_or_cumsum = torch.from_numpy(
                expert_tokens_count_or_cumsum
            )
        ds = torch.tensor([])
        if quant_mode == 0:
            expanded_x = numpy.clip(
                expanded_x,
                numpy.finfo(numpy.float16).min,
                numpy.finfo(numpy.float16).max,
            )
            expanded_x = expanded_x.astype("float16")
            scale_v = numpy.clip(
                scale[0], numpy.finfo(numpy.float16).min, numpy.finfo(numpy.float16).max
            )
            offset_v = offset_t.astype("float16")
            rr = expanded_x * scale_v + offset_v
            roundd = numpy.rint(rr)
            roundd = numpy.clip(roundd, -128, 127)
            expanded_x = roundd.astype("int8")
        else:
            xf = expanded_x.astype("float32")
            xa = numpy.abs(xf)
            xm = numpy.max(xa, axis=-1, keepdims=True)
            ds = xm / 127
            expanded_x = numpy.round(xf / ds).astype("int8")
            ds = torch.from_numpy(ds).reshape(-1)
        t_expanded_x = torch.from_numpy(expanded_x)
        if expert_tokens_before_capacity is None:
            expert_tokens_before_capacity = torch.tensor([])
        else:
            expert_tokens_before_capacity = torch.from_numpy(
                expert_tokens_before_capacity
            )

        return (
            t_expanded_x,
            torch.from_numpy(expanded_row_idx.astype("int32")),
            expert_tokens_count_or_cumsum,
            expert_tokens_before_capacity,
            ds,
        )
