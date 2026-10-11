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
from atk.configs.results_config import TaskResult

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def permute(tokens, indices, num_out_tokens: int = None, padded_mode: bool = False):
    if padded_mode:
        return permute_with_padded_tokens(tokens, indices)

    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)

    sorted_indices = torch.argsort(flatten_indices, stable=True)
    sorted_indices1 = torch.argsort(sorted_indices, stable=True)

    if num_out_tokens is not None and num_out_tokens != 0:
        sorted_indices = sorted_indices[:num_out_tokens]
    s_k = sorted_indices // topk
    permuted_tokens = tokens.index_select(0, s_k)
    return permuted_tokens, sorted_indices1


@register("function_aclnn_moe_token_permute_grad")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.numOutTokensOptional = 0
        self.index = None
        self.token = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        def permute_with_padded_tokens(tokens, indices):
            permuted_tokens = tokens.index_select(dim=0, index=indices.view(-1))
            return permuted_tokens, indices

        if self.device == "npu":
            device = f"{self.device}:{self.device_id}"
            from mindspeed.ops.npu_moe_token_permute import npu_moe_token_permute

            input_data.kwargs["tokens"].requires_grad_(True)
            permuted_tokens, sorted_indices = npu_moe_token_permute(
                input_data.kwargs["tokens"],
                input_data.kwargs["indices"],
                self.numOutTokensOptional,
                input_data.kwargs["paddedModeOptional"],
            )
            permuted_tokens.backward(
                torch.ones(permuted_tokens.shape).to(torch.bfloat16).npu()
            )
            out = input_data.kwargs["tokens"].grad
            return out

        else:
            device = "cpu"
            self.token.requires_grad_(True)
            permuted_tokens, sorted_indices = permute(
                self.token,
                self.index,
                self.numOutTokensOptional,
                input_data.kwargs["paddedModeOptional"],
            )
            permute_bwd_input = torch.ones(permuted_tokens.shape)
            print(permute_bwd_input.dtype)
            permuted_tokens.backward(permute_bwd_input)
            out = self.token.grad
            return out

    def init_by_input_data(self, input_data: InputDataset):
        self.token = input_data.kwargs["tokens"]
        tmp, indice = permute(
            input_data.kwargs["tokens"],
            input_data.kwargs["indices"],
            input_data.kwargs["numOutTokensOptional"],
            input_data.kwargs["paddedModeOptional"],
        )
        self.index = input_data.kwargs["indices"]

        input_data.kwargs["indices"] = indice.to(torch.int32)
        input_data.kwargs["tokens"] = (
            torch.ones(tmp.shape)
            .to(torch.bfloat16)
            .to(input_data.kwargs["tokens"].device)
        )
        self.numOutTokensOptional = input_data.kwargs["numOutTokensOptional"]
        input_data.kwargs["numOutTokensOptional"] = 1
        if len(self.index.shape) == 2:
            input_data.kwargs["numOutTokensOptional"] = self.index.shape[1]
