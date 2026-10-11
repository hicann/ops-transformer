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


@register("function_aclnn_moe_token_permute")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        def permute_with_padded_tokens(tokens, indices):
            permuted_tokens = tokens.index_select(dim=0, index=indices.view(-1))
            return permuted_tokens, indices

        def permute(
            tokens, indices, num_out_tokens: int = None, padded_mode: bool = False
        ):
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

        if self.device == "npu":
            device = f"{self.device}:{self.device_id}"
            permuted_tokens, sorted_indices = permute(
                input_data.kwargs["tokens"],
                input_data.kwargs["indices"],
                input_data.kwargs["numOutTokensOptional"],
                input_data.kwargs["paddedModeOptional"],
            )
            sorted_indices = sorted_indices.to(torch.int32)
            return permuted_tokens, sorted_indices
        else:
            device = "cpu"
            permuted_tokens, sorted_indices = permute(
                input_data.kwargs["tokens"],
                input_data.kwargs["indices"],
                input_data.kwargs["numOutTokensOptional"],
                input_data.kwargs["paddedModeOptional"],
            )
            sorted_indices = sorted_indices.to(torch.int32)
            return permuted_tokens, sorted_indices
