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
    "moe_init_routing_v2": "MoeInitRoutingV2TestSpec",
    "aclnnMoeInitRoutingV2": "AclnnMoeInitRoutingV2TestSpec",
}


def _adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
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


class AclnnMoeInitRoutingV2TestSpec:
    @staticmethod
    def golden(
        x,
        expertIdx,
        activeNumOptional=0,
        expertCapacityOptional=0,
        expertNumOptional=0,
        dropPadModeOptional=0,
        expertTokensCountOrCumsumFlagOptional=0,
        expertTokensBeforeCapacityFlagOptional=False,
        expandedXOut=None,
        expandedRowIdxOut=None,
        expertTokensCountOrCumsumOutOptional=None,
        expertTokensBeforeCapacityOutOptional=None,
        *args,
        **kwargs,
    ):
        ori_dtype = x.dtype
        input_x = x
        num_rows = input_x.shape[0]
        hidden_size = input_x.shape[-1]
        k = expertIdx.shape[-1] if len(expertIdx.shape) == 2 else 1
        expert_idx_np = expertIdx.reshape(-1).to(torch.int64).numpy()
        sorted_row_idx_np = numpy.argsort(expert_idx_np, axis=-1, kind="stable")
        sorted_expert_idx_np = numpy.sort(expert_idx_np, axis=-1, kind="stable")
        sorted_row_idx = torch.from_numpy(sorted_row_idx_np.copy())
        sorted_expert_idx = torch.from_numpy(sorted_expert_idx_np.copy())

        if dropPadModeOptional == 1 and expertNumOptional <= 0:
            return [None, None, None, None]

        expert_tokens_count_or_cumsum = None
        expert_tokens_before_capacity = None

        expert_idx_hist_np, _ = numpy.histogram(
            sorted_expert_idx_np,
            bins=expertNumOptional,
            range=(0, expertNumOptional - 1),
        )
        expert_idx_hist = torch.from_numpy(expert_idx_hist_np.astype(numpy.int32))
        expert_token_idx = torch.from_numpy(
            numpy.cumsum(expert_idx_hist_np).astype(numpy.int32)
        )

        if dropPadModeOptional == 1 and expertTokensBeforeCapacityFlagOptional:
            expert_tokens_before_capacity = expert_idx_hist
        if dropPadModeOptional == 0 and expertTokensCountOrCumsumFlagOptional == 1:
            expert_tokens_count_or_cumsum = expert_token_idx
        elif dropPadModeOptional == 0 and expertTokensCountOrCumsumFlagOptional == 2:
            expert_tokens_count_or_cumsum = expert_idx_hist

        if dropPadModeOptional == 0:
            expanded_row_idx = torch.zeros_like(sorted_row_idx, dtype=torch.int32)
            expanded_row_idx[sorted_row_idx] = torch.arange(
                sorted_row_idx.shape[0], dtype=torch.int32, device=sorted_row_idx.device
            )
            if activeNumOptional == 0:
                active_num = num_rows * k
            else:
                active_num = min(activeNumOptional, num_rows * k)
            selected_indices = sorted_row_idx[:active_num] // k
            expanded_x = input_x[selected_indices]
        else:
            sort_row_tmp = torch.full(
                (expertNumOptional * expertCapacityOptional,),
                -1,
                dtype=torch.int32,
                device=input_x.device,
            )
            offset = 0
            last_expert_id = -1
            for i in range(sorted_row_idx.shape[0]):
                val = sorted_row_idx[i].item()
                if val != -1:
                    current_expert = sorted_expert_idx[i].item()
                    if last_expert_id != current_expert:
                        offset = 0
                        last_expert_id = current_expert
                    index = current_expert * expertCapacityOptional + offset
                    if index < sort_row_tmp.shape[0]:
                        sort_row_tmp[index] = val
                        offset += 1

            expanded_row_idx = torch.full_like(sorted_row_idx, -1, dtype=torch.int32)
            for i in range(sort_row_tmp.shape[0]):
                val = sort_row_tmp[i].item()
                if val != -1:
                    expanded_row_idx[val] = i

            expanded_x = torch.zeros(
                (expertNumOptional * expertCapacityOptional, hidden_size),
                dtype=input_x.dtype,
                device=input_x.device,
            )
            for i in range(sort_row_tmp.shape[0]):
                val = sort_row_tmp[i].item()
                if val != -1:
                    expanded_x[i] = input_x[val // k]
            expanded_x = expanded_x.view(
                expertNumOptional, expertCapacityOptional, hidden_size
            )

        return [
            expanded_x.to(ori_dtype),
            expanded_row_idx,
            expert_tokens_count_or_cumsum,
            expert_tokens_before_capacity,
        ]

    @staticmethod
    def customize_inputs(x, expertIdx, *args, **kwargs):
        expert_num = kwargs.get("expertNumOptional", 0)
        if not expert_num and len(args) >= 3:
            expert_num = args[2]
        if not expert_num or expert_num <= 0:
            expert_num = 100
        expertIdx.copy_(
            torch.randint(0, expert_num, expertIdx.shape, dtype=torch.int32)
        )

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


class MoeInitRoutingV2TestSpec:
    @staticmethod
    def golden(
        x,
        expert_idx,
        *,
        active_num=0,
        expert_capacity=0,
        expert_num=0,
        drop_pad_mode=0,
        expert_tokens_count_or_cumsum_flag=0,
        expert_tokens_before_capacity_flag=False,
        **kwargs,
    ):
        ori_dtype = x.dtype
        input_x = x
        num_rows = input_x.shape[0]
        hidden_size = input_x.shape[-1]
        k = expert_idx.shape[-1] if len(expert_idx.shape) == 2 else 1
        sorted_row_idx = numpy.argsort(
            expert_idx.reshape((-1,)), axis=-1, kind="stable"
        )
        sorted_expert_idx = numpy.sort(expert_idx.reshape((-1,)), axis=-1)
        if drop_pad_mode == 1 and expert_num <= 0:
            return [None, None, None, None]

        expert_tokens_count_or_cumsum = None
        expert_tokens_before_capacity = None
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
            _adapter_capacity(sorted_row_idx, sorted_expert_idx, expert_capacity)
            sort_row_tmp = numpy.full((expert_num * expert_capacity), -1, dtype=int)
            offset = 0
            lastExpertId = 0
            for i, val in enumerate(sorted_row_idx):
                if val != -1:
                    if lastExpertId != sorted_expert_idx[i]:
                        offset = 0
                        lastExpertId = sorted_expert_idx[i]
                    sort_row_tmp[sorted_expert_idx[i] * expert_capacity + offset] = (
                        sorted_row_idx[i]
                    )
                    offset = offset + 1

            expanded_row_idx = numpy.full(sorted_row_idx.shape, -1)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_row_idx[val] = i

            expanded_x = numpy.full(
                (expert_num * expert_capacity, hidden_size), 0, dtype=input_x.dtype
            )
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_x[i] = input_x[val // k]
            expanded_x = expanded_x.reshape((expert_num, expert_capacity, hidden_size))

        return [
            expanded_x.astype(ori_dtype),
            expanded_row_idx.astype("int32"),
            expert_tokens_count_or_cumsum,
            expert_tokens_before_capacity,
        ]

    @staticmethod
    def customize_inputs(x, expert_idx, **kwargs):
        expert_num = kwargs.get("expert_num", 100)
        expert_idx = numpy.random.randint(
            0, expert_num, size=expert_idx.size, dtype=numpy.int32
        ).reshape(expert_idx.shape)
        return (x, expert_idx)

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
