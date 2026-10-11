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
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


@register("function_aclnn_moe_token_unpermute")
class FunctionUnpermuteApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionUnpermuteApi, self).__init__(task_result)
        self.origin_sorted_indices = None
        self.case_dtype = None

    def init_by_input_data(self, input_data: InputDataset):
        sorted_indices = input_data.kwargs["sorted_indices"]
        permutedTokens = input_data.kwargs["permutedTokens"]
        self.case_dtype = permutedTokens.dtype
        OpsDataset.seed_everything()
        sorted_indices_after = (
            torch.arange(sorted_indices.shape[0])
            .reshape(sorted_indices.shape)
            .to(torch.int32)
        )
        idx = torch.randperm(sorted_indices_after.nelement())
        sorted_indices_after = sorted_indices_after.view(-1)[idx].view(
            sorted_indices_after.size()
        )
        permuted_tokens, sorted_indices_tmp, origin_sorted_indices = self.mPermute(
            permutedTokens, sorted_indices_after
        )
        input_data.kwargs["sorted_indices"] = sorted_indices_tmp.to(torch.int32)
        input_data.kwargs["permutedTokens"] = permuted_tokens
        self.origin_sorted_indices = origin_sorted_indices

    def mPermute(
        self, tokens, indices, num_out_tokens: int = None, padded_mode: bool = False
    ):
        device = "cpu"
        if indices.dim() == 1:
            topk = 1
        else:
            topk = indices.size(1)
        flatten_indices = indices.view(-1)
        sorted_indices = torch.argsort(flatten_indices, stable=True)
        orgin_sorted_indices = sorted_indices
        sorted_indices = torch.argsort(sorted_indices, stable=True)
        if num_out_tokens is not None:
            sorted_indices = sorted_indices[:num_out_tokens]
        sorted_indices = sorted_indices.to(device)
        tokens = tokens.to(device)
        permuted_tokens = tokens.index_select(0, sorted_indices // topk)
        return permuted_tokens, sorted_indices, orgin_sorted_indices

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        permutedTokens = input_data.kwargs["permutedTokens"]
        sorted_indices = self.origin_sorted_indices
        probs = input_data.kwargs["probs"]

        permuted_tokens = permutedTokens.float()
        if probs is not None:
            probs = probs.float()
            num_unpermuted_tokens = probs.numel()
            topk = probs.size(1)
        else:
            num_unpermuted_tokens = permuted_tokens.size(0)
            topk = 1

        unpermuted_tokens = torch.zeros(
            [num_unpermuted_tokens, permuted_tokens.shape[-1]],
            dtype=permuted_tokens.dtype,
            device=permuted_tokens.device,
        )
        unpermuted_tokens.index_copy_(
            0, sorted_indices.to(torch.int64), permuted_tokens
        )
        unpermuted_tokens = unpermuted_tokens.reshape(
            -1, topk, permuted_tokens.size(-1)
        )
        if probs is not None:
            unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.sum(dim=1)

        return unpermuted_tokens.to(self.case_dtype)


@register("pyaclnn_aclnn_moe_token_unpermute")
class SampleAclnnApi(AclnnBaseApi):  # SampleApi类型仅需设置唯一即可。
    def init_by_input_data(self, input_data):
        """参数处理流同步报错问题"""
        input_data.kwargs["permutedTokens"] = input_data.kwargs["permutedTokens"].to(
            "npu"
        )
        input_data.kwargs["sorted_indices"] = input_data.kwargs["sorted_indices"].to(
            "npu"
        )
        input_data.kwargs["probs"] = input_data.kwargs["probs"].to("npu")
        return super().init_by_input_data(input_data)
