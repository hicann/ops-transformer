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
from atk.tasks.backends.lib_interface.acl_wrapper import (
    TensorPtr,
    AclFormat,
    AclTensorStruct,
)


@register("function_aclnn_moe_token_permute_with_ep_grad")
class FunctionPermuteApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionPermuteApi, self).__init__(task_result)
        self.tokens = None
        self.probs = None
        self.permuted_tokens = None
        self.permuted_probs = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.probs is None:
            gradt = input_data.kwargs["tokens"]
            self.permuted_tokens.backward(gradt)
            tokens_grad_out = self.tokens.grad
            probs_grad_out = None
        else:
            gradt1 = input_data.kwargs["tokens"]
            gradt2 = input_data.kwargs["probs"]
            gradt = torch.cat([gradt1.view(-1), gradt2.view(-1)])
            y = torch.cat([self.permuted_tokens.view(-1), self.permuted_probs.view(-1)])
            y.backward(gradt)
            tokens_grad_out = self.tokens.grad
            probs_grad_out = self.probs.grad
        return tokens_grad_out, probs_grad_out

    def init_by_input_data(self, input_data: InputDataset):
        def permute(
            tokens,
            indices,
            probs,
            rangeOptional: list = None,
            padded_mode: bool = False,
        ):
            if indices.dim() == 1:
                topk = 1
            else:
                topk = indices.size(1)
            flatten_indices = indices.view(-1)
            sorted_indices = torch.argsort(flatten_indices.float(), stable=True)
            sorted_indices1 = torch.argsort(sorted_indices.float(), stable=True)

            if rangeOptional is not None:
                sorted_indices = sorted_indices[rangeOptional[0] : rangeOptional[1]]
            permuted_tokens = tokens.index_select(0, sorted_indices // topk)

            if probs is not None:
                flatten_probs = probs.view(-1)
                permuted_probs = flatten_probs.index_select(0, sorted_indices)
            else:
                permuted_probs = None

            return permuted_tokens, sorted_indices1, permuted_probs

        tokens = input_data.kwargs["tokens"]
        indices = input_data.kwargs["indices"]
        topk = input_data.kwargs["topK"]
        total_len = indices.shape[0]
        # reshape indices
        indices = (
            indices.reshape(int(total_len / topk), topk)
            .to(torch.int32)
            .to(tokens.device)
        )
        input_data.kwargs["indices"] = indices

        # save forward tensor
        self.tokens = input_data.kwargs["tokens"].clone().to(tokens.device)
        self.tokens.requires_grad_(True)
        if input_data.kwargs["probs"] is not None:
            self.probs = input_data.kwargs["probs"].clone().to(tokens.device)
            self.probs.requires_grad_(True)
        else:
            self.probs = None

        if self.device == "npu":
            device = f"{self.device}:{self.device_id}"
            self.permuted_tokens, sorted_indices1, self.permuted_probs = permute(
                self.tokens,
                input_data.kwargs["indices"],
                self.probs,
                input_data.kwargs["rangeOptional"],
                input_data.kwargs["padded_mode"],
            )
        else:
            device = "cpu"
            self.permuted_tokens, sorted_indices1, self.permuted_probs = permute(
                self.tokens,
                input_data.kwargs["indices"],
                self.probs,
                input_data.kwargs["rangeOptional"],
                input_data.kwargs["padded_mode"],
            )

        input_data.kwargs["tokens"] = (
            torch.ones(self.permuted_tokens.shape)
            .to(self.permuted_tokens.dtype)
            .to(self.permuted_tokens.device)
        )
        input_data.kwargs["indices"] = sorted_indices1.to(torch.int32)
        if self.permuted_probs is None:
            input_data.kwargs["probs"] = None
        else:
            input_data.kwargs["probs"] = (
                torch.ones(self.permuted_probs.shape)
                .to(self.permuted_probs.dtype)
                .to(self.permuted_probs.device)
            )

        OpsDataset.seed_everything()


@register("AclnnBaseApi_aclnn_moe_token_permute_with_ep_grad")
class MoeFunctionApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        torch.npu.synchronize()
        input_args, output_packages = super().init_by_input_data(input_data)

        if input_data.kwargs["probs"] is None:
            torch_tensor = torch.tensor([]).to(input_data.kwargs["tokens"].dtype)
            acl_struct = self.torch_tensor_to_acl(torch_tensor, AclFormat.ACL_FORMAT_ND)
            input_args[2] = acl_struct
            input_args.append(acl_struct)

        return input_args, output_packages
