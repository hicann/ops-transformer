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


@register("ascend_aclnn_moe_token_unpermute_with_ep")
class FunctionUnpermuteApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        def unpermute(
            permuted_tokens,
            sorted_indices,
            probs,
            topK,
            range,
            padded_mode: bool = False,
            restore_shape: list = None,
        ):
            start = range[0]
            end = range[1]
            num_unpermuted_tokens = sorted_indices.numel() // topK
            sorted_indices = torch.sort(sorted_indices.float(), stable=True)[1]
            sorted_indices = sorted_indices[start:end]

            if probs is not None:
                flatten_probs = probs.view(-1)
                permuted_probs = flatten_probs.index_select(0, sorted_indices)
                permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

            unpermuted_tokens = torch.zeros(
                num_unpermuted_tokens,
                permuted_tokens.size(-1),
                dtype=permuted_tokens.dtype,
                device=permuted_tokens.device,
            )

            sorted_indices = sorted_indices // topK
            unpermuted_tokens = unpermuted_tokens.scatter_add_(
                0,
                sorted_indices.unsqueeze(1).expand(-1, permuted_tokens.shape[1]),
                permuted_tokens,
            )
            return unpermuted_tokens

        if self.device == "npu":
            device = f"{self.device}:{self.device_id}"
            unpermuted_tokens = unpermute(
                input_data.kwargs["permutedTokens"],
                input_data.kwargs["sorted_indices"],
                input_data.kwargs["probs"],
                input_data.kwargs["topKOptional"],
                input_data.kwargs["rangeOptional"],
                input_data.kwargs["padded_mode"],
                input_data.kwargs["restore_shape"],
            )
            return unpermuted_tokens
        else:
            device = "cpu"
            unpermuted_tokens = unpermute(
                input_data.kwargs["permutedTokens"],
                input_data.kwargs["sorted_indices"],
                input_data.kwargs["probs"],
                input_data.kwargs["topKOptional"],
                input_data.kwargs["rangeOptional"],
                input_data.kwargs["padded_mode"],
                input_data.kwargs["restore_shape"],
            )
            return unpermuted_tokens

    def init_by_input_data(self, input_data: InputDataset):
        tokens = input_data.kwargs["permutedTokens"]
        indices = input_data.kwargs["sorted_indices"]

        if "probs" in input_data.kwargs:
            probs = input_data.kwargs["probs"]
            topk = probs.shape[1]
        else:
            probs = None
            topk = input_data.kwargs["topKOptional"]

        rangeOptional = [
            input_data.kwargs["rangeOptional"][0],
            input_data.kwargs["rangeOptional"][1],
        ]
        total_len = indices.shape[0]
        # reshape indices
        indices = (
            indices.reshape(int(total_len / topk), topk)
            .to(torch.int32)
            .to(tokens.device)
        )

        # 生成unpermute输入
        permuted_tokens, sorted_indices1, permuted_probs = self.permute(
            tokens, indices, probs, rangeOptional
        )
        input_data.kwargs["permutedTokens"] = permuted_tokens
        input_data.kwargs["sorted_indices"] = sorted_indices1
        input_data.kwargs["probs"] = None
        input_data.kwargs["topKOptional"] = topk

        OpsDataset.seed_everything()

    def permute(
        self,
        tokens,
        indices,
        probs,
        rangeOptional: list = None,
        num_out_tokens: int = None,
        padded_mode: bool = False,
    ):
        if indices.dim() == 1:
            topk = 1
        else:
            topk = indices.size(1)
        flatten_indices = indices.view(-1)
        sorted_indices = torch.argsort(flatten_indices.float(), stable=True)
        sorted_indices1 = torch.argsort(sorted_indices.float(), stable=True)
        sorted_indices1 = sorted_indices1.to(torch.int32)

        if rangeOptional is not None:
            sorted_indices = sorted_indices[rangeOptional[0] : rangeOptional[1]]
        permuted_tokens = tokens.index_select(0, sorted_indices // topk)

        if probs is not None:
            flatten_probs = probs.view(-1)
            permuted_probs = flatten_probs.index_select(0, sorted_indices)
        else:
            permuted_probs = None

        return permuted_tokens, sorted_indices1, permuted_probs


@register("AclnnBaseApi_aclnn_moe_token_unpermute_with_ep")
class MoeFunctionApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        torch.npu.synchronize()
        input_args, output_packages = super(MoeFunctionApi, self).init_by_input_data(
            input_data
        )
        return input_args, output_packages
