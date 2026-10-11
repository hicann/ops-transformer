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


@register("function_aclnn_moe_init_routing")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs.get("x").cpu().numpy()
        expert_idx = input_data.kwargs.get("expert_idx").cpu().numpy()
        row_idx = input_data.kwargs.get("row_idx").cpu().numpy()
        active_num = input_data.kwargs.get("active_num")
        num_rows = x.shape[0]
        k = expert_idx.shape[-1]
        sort_expert_idx = numpy.argsort(
            expert_idx.reshape((-1,)), axis=-1, kind="stable"
        )
        expanded_expert_idx = numpy.sort(
            expert_idx.reshape((-1,)), axis=-1, kind="stable"
        )

        expanded_dst_to_src_row = numpy.take_along_axis(
            row_idx.reshape((-1,)), sort_expert_idx, axis=-1
        )
        expanded_row_idx = numpy.zeros(expanded_dst_to_src_row.shape, dtype=numpy.int32)
        expanded_row_idx[expanded_dst_to_src_row] = numpy.arange(
            expanded_dst_to_src_row.shape[-1], dtype=numpy.int32
        )

        active_num = min(active_num, num_rows) * k
        expanded_x = x[expanded_dst_to_src_row[:active_num] % num_rows, :]
        return (
            torch.from_numpy(expanded_x),
            torch.from_numpy(expanded_row_idx.astype("int32")),
            torch.from_numpy(expanded_expert_idx),
        )

    def init_by_input_data(self, input_data: InputDataset):
        input_data.kwargs["row_idx"] = (
            torch.arange(
                input_data.kwargs["expert_idx"].size(0)
                * input_data.kwargs["expert_idx"].size(1)
            )
            .reshape(
                input_data.kwargs["expert_idx"].size(0),
                input_data.kwargs["expert_idx"].size(1),
            )
            .to(torch.int32)
            .to(input_data.kwargs["expert_idx"].device)
        )
