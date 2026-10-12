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
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import OutputData
from einops import rearrange


# 下面为mrope实现，rope的positions为一维，不需要使用mrope_section；


def mRopeNpu(
    positions,
    query,
    key,
    cos_sin_cache,
    mrope_section,
    head_size,
    num_q_heads,
    num_kv_heads,
    is_neox_style=True,
):
    num_tokens = positions.shape[-1]
    rotary_dim = cos_sin_cache.shape[-1]

    positions_flatten = positions.flatten()  # [3, num_tokens] -> [3*num_tokens]
    cos_sin = cos_sin_cache.index_select(
        0, positions_flatten
    )  # cos_sin_cache: [num_tokens, rotary_dim] -> [3*num_tokens, rotary_dim]
    cos_sin = cos_sin.reshape(-1, rotary_dim)

    cos, sin = cos_sin.chunk(
        2, dim=-1
    )  # 1D: [num_tokens, rotary_dim // 2]  2D: [3 * num_tokens, rotary_dim // 2]
    if positions.ndim == 2:
        cos = cos.reshape(3, -1, rotary_dim // 2)  # [3, num_tokens, rotary_dim // 2]
        cos_0 = cos[0, :, : mrope_section[0]]
        cos_1 = cos[1, :, mrope_section[0] : (mrope_section[0] + mrope_section[1])]
        cos_2 = cos[
            2,
            :,
            (mrope_section[0] + mrope_section[1]) : (
                mrope_section[0] + mrope_section[1] + mrope_section[2]
            ),
        ]
        cos = torch.concat((cos_0, cos_1, cos_2), dim=-1)

        sin = sin.reshape(3, -1, rotary_dim // 2)  # [3, num_tokens, rotary_dim // 2]
        sin_0 = sin[0, :, : mrope_section[0]]
        sin_1 = sin[1, :, mrope_section[0] : (mrope_section[0] + mrope_section[1])]
        sin_2 = sin[
            2,
            :,
            (mrope_section[0] + mrope_section[1]) : (
                mrope_section[0] + mrope_section[1] + mrope_section[2]
            ),
        ]
        sin = torch.concat((sin_0, sin_1, sin_2), dim=-1)

    cos = cos.unsqueeze(-2)  # [num_tokens, 1, rotary_dim // 2]
    sin = sin.unsqueeze(-2)  # [num_tokens, 1, rotary_dim // 2]
    query_shape = query.shape
    query = query.view(num_tokens, -1, head_size)  # [num_tokens, num_heads, head_size]
    query_rot = query[..., :rotary_dim]  # [num_tokens, num_heads, rotary_dim]
    query_pass = query[
        ..., rotary_dim:
    ]  # [num_tokens, num_heads, head_size - rotary_dim]
    if is_neox_style:
        x1, x2 = torch.chunk(
            query_rot, 2, dim=-1
        )  # [num_tokens, num_heads, rotary_dim // 2]
    else:
        x1 = query_rot[..., ::2]
        x2 = query_rot[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        query_rot = torch.cat((o1, o2), dim=-1)
    else:
        query_rot = torch.stack((o1, o2), dim=-1).flatten(-2)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, head_size)  # [num_tokens, num_kv_heads, head_size]
    key_rot = key[..., :rotary_dim]  # [num_tokens, num_kv_heads, rotary_dim]
    key_pass = key[
        ..., rotary_dim:
    ]  # [num_tokens, num_kv_heads, head_size - rotary_dim]
    if is_neox_style:
        x1, x2 = torch.chunk(key_rot, 2, dim=-1)
    else:
        x1 = key_rot[..., ::2]
        x2 = key_rot[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        key_rot = torch.cat((o1, o2), dim=-1)
    else:
        key_rot = torch.stack((o1, o2), dim=-1).flatten(-2)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(
        key_shape
    )  # [num_tokens, num_kv_heads * head_size]
    return query, key


@register("executor_function_ascend_rope_with_sin_cos_cache")
class aclnnRopeWithSinCosCachesApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        positions = input_data.kwargs["positions"]
        query = input_data.kwargs["queryIn"]
        key = input_data.kwargs["keyIn"]
        cos_sin_cache = input_data.kwargs["cosSinCache"]
        mrope_section = input_data.kwargs["mropeSection"]
        head_size = input_data.kwargs["headSize"]
        num_q_heads = query.shape[1] / head_size
        num_kv_heads = key.shape[1] / head_size
        is_neox_style = input_data.kwargs["isNeoxStyle"]
        if positions.ndim == 2:
            mrope_section[0] = 16
        if self.device == "cpu":
            query = query.to(torch.float32)
            key = key.to(torch.float32)
            cos_sin_cache = cos_sin_cache.to(torch.float32)
        output = mRopeNpu(
            positions,
            query,
            key,
            cos_sin_cache,
            mrope_section,
            head_size,
            num_q_heads,
            num_kv_heads,
            is_neox_style,
        )
        return output

    def init_by_input_data(self, input_data: InputDataset):
        if self.device == "pyaclnn":
            if input_data.kwargs["positions"].ndim == 2:
                input_data.kwargs["mropeSection"][0] = 16
            queryIn_input_dtype = input_data.kwargs["queryIn"].dtype
            keyIn_input_dtype = input_data.kwargs["keyIn"].dtype
            input_data.kwargs["positions"] = input_data.kwargs["positions"].npu()
            input_data.kwargs["queryIn"] = input_data.kwargs["queryIn"].npu()
            input_data.kwargs["keyIn"] = input_data.kwargs["keyIn"].npu()
            input_data.kwargs["cosSinCache"] = input_data.kwargs["cosSinCache"].npu()

            self.task_result.output_info_list[0].dtype = str(queryIn_input_dtype)
            self.task_result.output_info_list[1].dtype = str(keyIn_input_dtype)
