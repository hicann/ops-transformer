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


@register("function_aclnn_moe_token_permute_with_ep")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        def permute(
            tokens,
            indices,
            probs,
            range: list = None,
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

            if range is not None:
                sorted_indices = sorted_indices[range[0] : range[1]]
            permuted_tokens = tokens.index_select(0, sorted_indices // topk)

            if probs is None:
                permuted_probs = None
            else:
                flatten_probs = probs.view(-1)
                permuted_probs = flatten_probs.index_select(0, sorted_indices)
            return permuted_tokens, sorted_indices1, permuted_probs

        if self.device == "npu":
            device = f"{self.device}:{self.device_id}"
            permuted_tokens, sorted_indices, permuted_probs = permute(
                input_data.kwargs["tokens"],
                input_data.kwargs["indices"],
                input_data.kwargs["probsOptional"],
                input_data.kwargs["rangeOptional"],
                input_data.kwargs["numOutTokensOptional"],
                input_data.kwargs["paddedModeOptional"],
            )
            return permuted_tokens, sorted_indices, permuted_probs
        else:
            device = "cpu"
            permuted_tokens, sorted_indices, permuted_probs = permute(
                input_data.kwargs["tokens"],
                input_data.kwargs["indices"],
                input_data.kwargs["probsOptional"],
                input_data.kwargs["rangeOptional"],
                input_data.kwargs["numOutTokensOptional"],
                input_data.kwargs["paddedModeOptional"],
            )
            return permuted_tokens, sorted_indices, permuted_probs


@register("AclnnBaseApi_aclnn_moe_token_permute_with_ep")
class MoeFunctionApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super(MoeFunctionApi, self).init_by_input_data(
            input_data
        )
        return input_args, output_packages
