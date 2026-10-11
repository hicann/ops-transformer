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


@register("function_aclnn_moe_init_routing_V2")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_x = input_data.kwargs.get("x").cpu().numpy()
        expert_idx = input_data.kwargs.get("expert_idx").cpu().numpy()

        active_num = input_data.kwargs.get("active_num")
        expert_num = input_data.kwargs.get("expert_num")
        capacity = input_data.kwargs.get("expert_capacity")
        drop_pad_mode = input_data.kwargs.get("drop_pad_mode")
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

        return (
            torch.from_numpy(expanded_x),
            torch.from_numpy(expanded_row_idx.astype("int32")),
            expert_tokens_count_or_cumsum,
            torch.from_numpy(expert_tokens_before_capacity),
        )

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnMoeInitRoutingV2WorkspaceSize(const aclTensor *x, const aclTensor *expertIdx, int64_t activeNum, int64_t expertCapacity, int64_t expertNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag, bool expertTokensBeforeCapacityFlag, aclTensor *expandedXOut, aclTensor *expandedRowIdxOut, aclTensor *expertTokensCountOrCumsumOut, aclTensor *expertTokensBeforeCapacityOut, uint64_t *workspaceSize, aclOpExecutor **executor)"
