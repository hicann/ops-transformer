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

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from einops import rearrange


def create_random_integer_array(len, max):
    np.random.seed(33)
    # Generate a sorted array of random integers between 0 and b
    try:
        random_points = np.sort(np.random.choice(range(1, max), len - 2, replace=False))
        # Concatenate 0 at the beginning and b at the end
        array = np.concatenate(([0], random_points, [max]))
    except:
        array = np.array([0, max])
    return array


class RingAttentionUpdate:
    def __init__(
        self,
        prev_attn_out,
        prev_softmax_max,
        prev_softmax_sum,
        cur_attn_out,
        cur_softmax_max,
        cur_softmax_sum,
        actual_seq_qlen,
        input_layout,
    ):
        self.prev_attn_out = prev_attn_out
        self.prev_softmax_max = prev_softmax_max
        self.prev_softmax_sum = prev_softmax_sum
        self.cur_attn_out = cur_attn_out
        self.cur_softmax_max = cur_softmax_max
        self.cur_softmax_sum = cur_softmax_sum
        self.actual_seq_qlen = actual_seq_qlen
        self.input_layout = input_layout


def flatten_softmax(x, sub_seq_len):
    orig_shape = x.shape  # TN8
    section_len = [s * orig_shape[1] for s in sub_seq_len]
    splits = x.view(-1, orig_shape[-1]).split(
        section_len, dim=0
    )  # (NS1 + NS2 + NS3..) 8 FA 算子目前排布是NS8NS8NS8
    merged = [
        item.view(orig_shape[1], -1, orig_shape[-1]).transpose(0, 1) for item in splits
    ]
    merged = torch.cat(merged, dim=0)
    return merged


def unflatten_softmax(x, sub_seq_len):
    orig_shape = x.shape  # TN8: (S1N + S2N + S3N) 8
    section_len = [s * orig_shape[1] for s in sub_seq_len]
    splits = x.view(-1, orig_shape[-1]).split(
        section_len, dim=0
    )  # (S1N + S2N + S3N..)8
    merged = [
        item.reshape(-1, orig_shape[1], orig_shape[-1])
        .transpose(0, 1)
        .reshape(-1, orig_shape[-1])
        for item in splits
    ]
    merged = torch.cat(merged, dim=0)  # (NS1+NS2+NS3)8
    return merged.view(*orig_shape)


def ring_attention_update(params):
    # 使用params对象来访问参数
    prev_attn_out = params.prev_attn_out
    prev_softmax_max = params.prev_softmax_max
    prev_softmax_sum = params.prev_softmax_sum
    cur_attn_out = params.cur_attn_out
    cur_softmax_max = params.cur_softmax_max
    cur_softmax_sum = params.cur_softmax_sum
    actual_seq_qlen = params.actual_seq_qlen
    input_layout = params.input_layout

    cur_sub_out_seq_len = (
        torch.cat([torch.tensor([0]), actual_seq_qlen])[1:]
        - torch.cat([torch.tensor([0]), actual_seq_qlen])[:-1]
    ).tolist()

    # cur_sub_out_seq_len = (torch.tensor([0] + actual_seq_qlen)[1:] - torch.tensor([0] + actual_seq_qlen)[:-1]).tolist()
    if input_layout == "TND":
        cur_softmax_max = flatten_softmax(cur_softmax_max, cur_sub_out_seq_len)
        cur_softmax_sum = flatten_softmax(cur_softmax_sum, cur_sub_out_seq_len)
        prev_softmax_max = flatten_softmax(prev_softmax_max, cur_sub_out_seq_len)
        prev_softmax_sum = flatten_softmax(prev_softmax_sum, cur_sub_out_seq_len)

    # update softmax_max
    origin_dtype = prev_attn_out.dtype
    prev_attn_out = prev_attn_out.to(torch.float32)
    cur_attn_out = cur_attn_out.to(torch.float32)
    softmax_max = torch.maximum(prev_softmax_max, cur_softmax_max)
    prev_scale = torch.exp(prev_softmax_max - softmax_max)
    cur_scale = torch.exp(cur_softmax_max - softmax_max)

    # update softmax_sum
    prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
    cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
    softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

    # out updating scale
    prev_out_scale = prev_softmax_sum_scaled / softmax_sum
    cur_out_scale = cur_softmax_sum_scaled / softmax_sum

    # [b, n, s, 8] -> [s, b, h]
    if input_layout == "SBH":
        num_channels = prev_out_scale.shape[1]
        hidden_size = prev_attn_out.shape[-1]
        division_result = hidden_size // num_channels
        prev_out_scale = (
            prev_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, division_result)
        )
        prev_out_scale = rearrange(prev_out_scale, "b n s d -> s b (n d)").contiguous()
        cur_out_scale = (
            cur_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, division_result)
        )
        cur_out_scale = rearrange(cur_out_scale, "b n s d -> s b (n d)").contiguous()
    elif input_layout == "TND":
        division_result = prev_attn_out.shape[-1]
        prev_out_scale = (
            prev_out_scale[..., 0].unsqueeze(2).repeat(1, 1, division_result)
        )  # [sb, n, d]
        cur_out_scale = cur_out_scale[..., 0].unsqueeze(2).repeat(1, 1, division_result)

    # update output
    attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale
    attn_out = attn_out.to(origin_dtype)

    if input_layout == "TND":
        softmax_max = unflatten_softmax(softmax_max, cur_sub_out_seq_len)
        softmax_sum = unflatten_softmax(softmax_sum, cur_sub_out_seq_len)

    return attn_out, softmax_max, softmax_sum


@register("aclnn_ring_attention_update")
class AclnnRingAttentionUpdateApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = None
        if not with_output:
            if self.device == "npu":
                if self.device_id == 0:
                    eval(self.api_name)(*input_data.args, **input_data.kwargs)
                elif self.device_id == 1:
                    params = RingAttentionUpdate(**input_data.kwargs)
                    output = ring_attention_update(params)
            return output

        if self.device == "cpu":
            params = RingAttentionUpdate(**input_data.kwargs)
            output = ring_attention_update(params)
            return output
        elif self.device == "npu":
            if self.device_id == 0:
                torch.npu.set_compile_mode(jit_compile=False)
            elif self.device_id == 1:
                torch.npu.set_compile_mode(jit_compile=True)
            output = eval(self.api_name)(*input_data.args, **input_data.kwargs)
        if self.output is None:
            output = eval(self.api_name)(*input_data.args, **input_data.kwargs)
        else:
            if isinstance(self.output, int):
                output = input_data.args[self.output]
            elif isinstance(self.output, str):
                output = input_data.kwargs[self.output]
            else:
                raise ValueError(f"self.output {self.output} value is error")
        return output

    def init_by_input_data(self, input_data: InputDataset):
        input_data.kwargs["prev_softmax_max"][..., :] = input_data.kwargs[
            "prev_softmax_max"
        ][..., 0:1]
        input_data.kwargs["prev_softmax_sum"][..., :] = input_data.kwargs[
            "prev_softmax_sum"
        ][..., 0:1]
        input_data.kwargs["cur_softmax_max"][..., :] = input_data.kwargs[
            "cur_softmax_max"
        ][..., 0:1]
        input_data.kwargs["cur_softmax_sum"][..., :] = input_data.kwargs[
            "cur_softmax_sum"
        ][..., 0:1]
        actualseq = create_random_integer_array(
            input_data.kwargs["actual_seq_qlen"].shape[0],
            input_data.kwargs["prev_attn_out"].shape[0],
        )
        input_data.kwargs["actual_seq_qlen"][..., :] = torch.tensor(actualseq)
