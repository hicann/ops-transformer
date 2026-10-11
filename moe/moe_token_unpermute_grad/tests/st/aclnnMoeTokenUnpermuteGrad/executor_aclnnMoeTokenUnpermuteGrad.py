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


@register("ascend_aclnn_moe_token_unpermute_grad_probs_none")
class FunctionUnpermuteGradApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionUnpermuteGradApi, self).__init__(task_result)
        self.origin_sorted_indices = None
        self.permuted_tokens_grad1 = None
        self.unpermuted_tokens = None
        self.probs = None
        self.permuted_tokens = None
        self.output_grad = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        permutedtokens = input_data.kwargs["permutedTokens"]
        output_grad = self.output_grad
        sorted_indices = self.origin_sorted_indices
        permuted_tokens = self.permuted_tokens

        topk = 1
        d = output_grad.size(1)
        n = sorted_indices.size(0)
        unpermuted_tokens = self.unpermuted_tokens
        unpermuted_tokens = permuted_tokens.index_select(0, sorted_indices)
        unpermuted_tokens = unpermuted_tokens.reshape(
            -1, topk, permuted_tokens.size(-1)
        )

        self.permuted_tokens_grad1.index_copy_(0, sorted_indices, output_grad)

        return self.permuted_tokens_grad1.bfloat16(), torch.tensor([1]).bfloat16()

    def init_by_input_data(self, input_data: InputDataset):
        sorted_indices = input_data.kwargs["sorted_indices"]
        permutedtokens = input_data.kwargs["permutedTokens"]
        probs = input_data.kwargs["probs"]
        OpsDataset.seed_everything()
        sorted_indices_after = torch.randint(
            low=0, high=sorted_indices.shape[0], size=(probs.shape[0], probs.shape[1])
        ).to(torch.int32)
        sorted_indices_after = sorted_indices_after.to(sorted_indices.device)
        permuted_tokens, sorted_indices_tmp, origin_sorted_indices = self.mpermute(
            permutedtokens, sorted_indices_after
        )
        input_data.kwargs["sorted_indices"] = origin_sorted_indices.to(torch.int32)
        input_data.kwargs["permutedTokens"] = permuted_tokens
        input_data.kwargs["probs"] = None
        self.origin_sorted_indices = origin_sorted_indices

        output_grad = input_data.kwargs["unpermutedTokensGrad"]
        d = output_grad.size(1)
        n = sorted_indices.size(0)
        self.permuted_tokens_grad1 = torch.zeros(n, d, dtype=torch.float32)
        unpermuted_tokens = torch.zeros(n, d, dtype=torch.float32)
        self.permuted_tokens_grad1 = self.permuted_tokens_grad1.to(
            sorted_indices.device
        )
        self.probs = probs.float()
        self.permuted_tokens = permuted_tokens.float()
        self.output_grad = output_grad.float()

    def mpermute(
        self, tokens, indices, num_out_tokens: int = None, padded_mode: bool = False
    ):
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
        permuted_tokens = tokens.index_select(0, sorted_indices // topk)
        return permuted_tokens, sorted_indices, orgin_sorted_indices


@register("AclnnBaseApi_aclnn_moe_token_unpermute_grad_probs_none")
class MoeFunctionApi(AclnnBaseApi):
    def after_call(self, output_packages):
        output = []
        torch_tensor = self.acl_tensor_to_torch(output_packages[0])
        output.append(torch_tensor)
        output.append(torch.tensor([1]).bfloat16().to(torch_tensor.device))
        return output
