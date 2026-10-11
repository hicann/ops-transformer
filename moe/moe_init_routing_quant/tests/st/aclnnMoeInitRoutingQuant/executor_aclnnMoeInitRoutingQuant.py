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
import numpy as np
import numpy

from atk.configs.dataset_config import InputDataset

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def moe_init_routing(x, row_idx, expert_idx, active_num):
    num_rows = x.shape[0]
    hidden_size = x.shape[-1]
    k = expert_idx.shape[-1]
    sort_expert_for_source_row = np.argsort(
        expert_idx.reshape((-1,)), axis=-1, kind="stable"
    )
    expanded_expert_idx = np.sort(expert_idx.reshape((-1,)), axis=-1)

    expanded_dst_to_src_row = np.take_along_axis(
        row_idx.reshape((-1,)), sort_expert_for_source_row, axis=-1
    )
    expanded_row_idx = np.zeros(expanded_dst_to_src_row.shape).astype(np.int32)
    expanded_row_idx[expanded_dst_to_src_row] = np.arange(
        expanded_dst_to_src_row.shape[-1]
    )
    active_num = min(active_num, num_rows) * k
    expanded_x = x[expanded_dst_to_src_row[:active_num] % num_rows, :]
    return expanded_x, expanded_row_idx, expanded_expert_idx


def quant_a(x, scale, offset):
    sr = x.astype("float16") * scale + offset
    roundd = np.rint(sr)
    roundd = np.clip(roundd, -128, 127)
    output = roundd.astype("int8")
    return output


@register("function_aclnn_moe_init_routing_quant")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_x = input_data.kwargs.get("x").cpu().numpy()
        row_idx = input_data.kwargs.get("rowIdx").cpu().numpy()

        expert_idx = input_data.kwargs.get("expertIdx").cpu().numpy()

        active_num = input_data.kwargs.get("active_num")
        scale = input_data.kwargs.get("scale")
        offset = input_data.kwargs.get("offset")
        expanded_x, expanded_row_idx, expanded_expert_idx = moe_init_routing(
            input_x, row_idx, expert_idx, active_num
        )
        expanded_x = quant_a(expanded_x, scale, offset)
        return (
            torch.from_numpy(expanded_x),
            torch.from_numpy(expanded_row_idx.astype("int32")),
            torch.from_numpy(expanded_expert_idx),
        )

    def init_by_input_data(self, input_data: InputDataset):
        input_data.kwargs["rowIdx"] = (
            torch.arange(input_data.kwargs["expertIdx"].size(0))
            .reshape(-1, 1)
            .expand(
                input_data.kwargs["expertIdx"].size(0),
                input_data.kwargs["expertIdx"].size(1),
            )
            .to(torch.int32)
            .to(input_data.kwargs["expertIdx"].device)
        )
