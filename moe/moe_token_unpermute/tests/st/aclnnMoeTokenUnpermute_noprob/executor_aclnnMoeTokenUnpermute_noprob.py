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


@register("function_aclnn_moe_token_unpermute_probs_none")
class FunctionUnpermuteApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionUnpermuteApi, self).__init__(task_result)
        self.origin_sorted_indices = None
        self.unpermuted_tokens = None
        self.permuted_tokens = None
        self.probs = None
        self.case_dtype = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        sorted_indices = self.origin_sorted_indices

        permuted_tokens = self.permuted_tokens
        num_unpermuted_tokens = permuted_tokens.size(0)
        topk = 1

        unpermuted_tokens = self.unpermuted_tokens
        unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
        unpermuted_tokens = unpermuted_tokens.reshape(
            -1, topk, permuted_tokens.size(-1)
        )
        unpermuted_tokens = unpermuted_tokens.sum(dim=1)
        return unpermuted_tokens.to(self.case_dtype)

    def init_by_input_data(self, input_data: InputDataset):
        if self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"
        sorted_indices = input_data.kwargs["sorted_indices"]
        permutedtokens = input_data.kwargs["permutedTokens"]
        probs = input_data.kwargs["probs"]
        self.case_dtype = permutedtokens.dtype
        OpsDataset.seed_everything()
        sorted_indices_after = torch.randint(
            low=0, high=sorted_indices.shape[0], size=(probs.shape[0], probs.shape[1])
        ).to(torch.int32)
        if self.device == "npu":
            sorted_indices_after = sorted_indices_after.npu()
        sorted_indices_after = sorted_indices_after.to(device)
        permuted_tokens, sorted_indices_tmp, origin_sorted_indices = self.mpermute(
            permutedtokens, sorted_indices_after
        )
        input_data.kwargs["sorted_indices"] = sorted_indices_tmp.to(torch.int32)
        input_data.kwargs["permutedTokens"] = permuted_tokens
        self.origin_sorted_indices = origin_sorted_indices.to(torch.int64)
        self.unpermuted_tokens = torch.zeros(
            [probs.numel(), permuted_tokens.shape[-1]],
            dtype=permuted_tokens.dtype,
            device=permuted_tokens.device,
        ).float()
        self.unpermuted_tokens = self.unpermuted_tokens.to(device)
        self.permuted_tokens = permuted_tokens.float()
        self.probs = probs.float()

    def mpermute(
        self, tokens, indices, num_out_tokens: int = None, padded_mode: bool = False
    ):
        device = tokens.device
        if indices.dim() == 1:
            topk = 1
        else:
            topk = indices.size(1)
        flatten_indices = indices.view(-1)
        sorted_indices = torch.argsort(flatten_indices, stable=True)
        orgin_sorted_indices = sorted_indices
        orgin_sorted_indices = orgin_sorted_indices.to(device)
        sorted_indices = torch.argsort(sorted_indices, stable=True)
        if num_out_tokens is not None:
            sorted_indices = sorted_indices[:num_out_tokens]
        sorted_indices = sorted_indices.to(device)
        s_k = sorted_indices // topk
        tokens_tmp = tokens.cpu()
        s_k_tmp = s_k.cpu()
        permuted_tokens = tokens_tmp.index_select(0, s_k_tmp).to(device)
        return permuted_tokens, sorted_indices, orgin_sorted_indices


@register("pyaclnn_aclnn_moe_token_unpermute_probs_none")
class AclnnFunctionalMultiLabelMarginLossApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        # probs设置为空指针
        from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

        input_args[2] = TensorPtr()
        return input_args, output_packages
