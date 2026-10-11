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


@register("function_aclnn_moe_token_permute_with_routing_map")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        def permute(
            tokens,  # [num_tokens, hidden]
            routing_map,  # [num_tokens, num_experts], bool
            probs,
            num_out_tokens,
            drop_and_pad,
        ):
            num_tokens, hidden = tokens.shape
            num_experts = routing_map.shape[1]

            cap = num_out_tokens // num_experts
            sorted_indices2 = None
            if drop_and_pad:
                permuted_probs = None
                routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
                sorted_indices = routing_map.sort(dim=-1, descending=True, stable=True)[
                    1
                ]
                sorted_indices = sorted_indices[:, :cap].contiguous()
                sorted_indices = sorted_indices.view(-1)
                if probs is not None:
                    probs_T_1D = probs.T
                    probs_T_1D = probs_T_1D.contiguous().view(-1)
                    indices_dim0 = torch.arange(
                        num_experts, device=routing_map.device
                    ).unsqueeze(-1)
                    indices_1D = (
                        indices_dim0 * num_tokens
                        + sorted_indices.view(num_experts, cap)
                    ).view(-1)
                    permuted_probs = probs_T_1D.index_select(0, indices_1D)
                permuted_input = tokens.index_select(0, sorted_indices)
                return permuted_input, permuted_probs, sorted_indices.to(torch.int32)
            else:
                routing_map = routing_map.bool().T.contiguous()
                token_indices = (
                    torch.arange(num_tokens, device=routing_map.device)
                    .unsqueeze(0)
                    .expand(num_experts, -1)
                )
                sorted_indices = token_indices.masked_select(routing_map)
                sorted_indices2 = torch.sort(sorted_indices.float(), stable=True)[1]

                if probs is not None:
                    permuted_probs = probs.T.masked_select(routing_map)
                else:
                    permuted_probs = None
                permuted_input = tokens.index_select(0, sorted_indices)
                return permuted_input, permuted_probs, sorted_indices2.to(torch.int32)

        if self.device == "npu":
            device = f"{self.device}:{self.device_id}"
            permuted_tokens, permuted_probs, sorted_indices = permute(
                input_data.kwargs["tokens"],
                input_data.kwargs["routingMap"],
                input_data.kwargs["probsOptional"],
                input_data.kwargs["numOutTokens"],
                input_data.kwargs["dropAndPad"],
            )
            return permuted_tokens, permuted_probs, sorted_indices
        else:
            device = "cpu"
            permuted_tokens, permuted_probs, sorted_indices = permute(
                input_data.kwargs["tokens"],
                input_data.kwargs["routingMap"],
                input_data.kwargs["probsOptional"],
                input_data.kwargs["numOutTokens"],
                input_data.kwargs["dropAndPad"],
            )
            return permuted_tokens, permuted_probs, sorted_indices

    def init_by_input_data(self, input_data: InputDataset):
        OpsDataset.seed_everything()
        print(input_data.kwargs["routingMap"])

        def generate_map(m, n, k):
            rmap = torch.zeros((m, n), dtype=torch.bool)
            for i in range(m):
                ind = torch.randperm(n)[:k]
                rmap[i, ind] = True
            return rmap

        routingMap = input_data.kwargs["routingMap"]
        probsOptional = input_data.kwargs["probsOptional"]
        tokenNum = routingMap.shape[0]
        expertNum = routingMap.shape[1]
        numOutTokens = input_data.kwargs["numOutTokens"]
        dropAndPad = input_data.kwargs["dropAndPad"]
        cap = 0
        topk = 0
        numOutTokens = min(tokenNum * expertNum, numOutTokens)
        numOutTokens = max(tokenNum, numOutTokens)
        input_data.kwargs["numOutTokens"] = numOutTokens
        if dropAndPad:
            cap = numOutTokens // expertNum
            rmap = torch.rand(tokenNum, expertNum)
            rmap = rmap > 0.5
        else:
            topk = numOutTokens // tokenNum
            if topk >= 512:
                input_data.kwargs["numOutTokens"] = numOutTokens * 512
                topk = 512
            rmap = generate_map(tokenNum, expertNum, topk)
        print(input_data.kwargs["routingMap"])

        if self.device == "npu":
            rmap = rmap.npu()
        input_data.kwargs["routingMap"] = rmap.to(
            input_data.kwargs["routingMap"].device
        )
        print(input_data.kwargs["routingMap"])


@register("AclnnBaseApi_aclnn_moe_token_permute_with_routing_map")
class MoeFunctionApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super(MoeFunctionApi, self).init_by_input_data(
            input_data
        )
        return input_args, output_packages
