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
import random
import ctypes
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from typing import Optional


@register("ascend_aclnn_moe_token_permute_with_routing_map_grad")
class FunctionMoeTokenPermuteWithRoutingMapGradApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionMoeTokenPermuteWithRoutingMapGradApi, self).__init__(task_result)

    def safeDiv(self, a, b):
        if b == 0:
            return 0
        else:
            return a // b

    def init_by_input_data(self, input_data: InputDataset):
        num_tokens = input_data.kwargs["tokensNum"]
        hidden_size = input_data.kwargs["permutedTokenOutputGrad"].shape[1]
        num_experts = input_data.kwargs["numExperts"]
        drop_and_pad = input_data.kwargs["padded_mode"]
        num_out_tokens = input_data.kwargs["permutedTokenOutputGrad"].shape[0]
        self.torch_type = input_data.kwargs["permutedTokenOutputGrad"].dtype
        case_id = self.task_result.case_config.id
        prob_is_none_list = [True, False]
        self.prob_is_none = prob_is_none_list[case_id % 2]
        if not self.prob_is_none:
            self.prob_type = input_data.kwargs["permutedProbsOutputGradOptional"].dtype
        else:
            self.prob_type = self.torch_type
        input_data.kwargs["tokensNum"] = num_tokens
        input_data.kwargs["padded_mode"] = drop_and_pad
        input_data.kwargs["numExperts"] = num_experts
        sortedIndices = input_data.kwargs["sortedIndices"]

        OpsDataset.seed_everything(case_id)
        # self.prob_is_none = False

        routing_map = None
        if drop_and_pad:
            capacity = self.safeDiv(num_out_tokens, num_experts)
            routing_map = self.generate_routing_map_drop_pad_true(
                capacity, num_tokens, num_experts
            )
        else:
            topK = self.safeDiv(num_out_tokens, num_tokens)
            routing_map = self.generate_routing_map_drop_pad_false(
                topK, num_tokens, num_experts
            )
        tokens = self.generate_tensor((num_tokens, hidden_size), torch.bfloat16, 5)
        probs = None
        if not self.prob_is_none:
            probs = self.generate_tensor((num_tokens, num_experts), torch.bfloat16, 5)
        permuted_tokens, permuted_probs, sorted_indices = self.permute(
            tokens, routing_map, probs, num_out_tokens, False, drop_and_pad
        )
        if not drop_and_pad:
            sorted_twice_indices = torch.arange(
                num_tokens, dtype=sorted_indices.dtype
            ).repeat_interleave(topK)
            sorted_twice_index = torch.full(
                (num_tokens * topK,), -1, dtype=sorted_indices.dtype
            )
            sorted_vals, sort_idx = torch.sort(sorted_indices, stable=True)
            group_starts = torch.searchsorted(
                sorted_vals, torch.arange(num_tokens, dtype=sorted_vals.dtype)
            )
            within_offsets = (
                torch.arange(len(sorted_vals), dtype=torch.long)
                - group_starts[sorted_vals.long()]
            )
            target_pos = sorted_vals.long() * topK + within_offsets
            sorted_twice_index.scatter_(0, target_pos, sort_idx)
            padded_permuted_tokens = torch.zeros(
                (num_tokens * topK, hidden_size), dtype=permuted_tokens.dtype
            )
            padded_permuted_tokens[: permuted_tokens.shape[0]] = permuted_tokens
            permuted_tokens = padded_permuted_tokens
            if not self.prob_is_none:
                padded_permuted_probs = torch.zeros(
                    num_tokens * topK, dtype=permuted_probs.dtype
                )
                padded_permuted_probs[: permuted_probs.shape[0]] = permuted_probs
                permuted_probs = padded_permuted_probs
        else:
            sorted_twice_index = sorted_indices

        if self.device == "pyaclnn":
            input_data.kwargs["permutedTokenOutputGrad"] = permuted_tokens.to(
                self.torch_type
            ).npu()
            if permuted_probs is None:
                input_data.kwargs["permutedProbsOutputGradOptional"] = ctypes.POINTER(
                    AclTensor
                )()
            else:
                input_data.kwargs["permutedProbsOutputGradOptional"] = (
                    permuted_probs.to(self.prob_type).npu()
                )
            input_data.kwargs["sortedIndices"] = sorted_twice_index.to(
                torch.int32
            ).npu()
            input_data.kwargs["routingMapOptional"] = routing_map.npu()
        else:
            input_data.kwargs["permutedTokenOutputGrad"] = permuted_tokens.to(
                self.torch_type
            )
            if permuted_probs is None:
                input_data.kwargs["permutedProbsOutputGradOptional"] = ctypes.POINTER(
                    AclTensor
                )()
            else:
                input_data.kwargs["permutedProbsOutputGradOptional"] = (
                    permuted_probs.to(self.prob_type)
                )
            input_data.kwargs["sortedIndices"] = sorted_twice_index.to(torch.int32)
            input_data.kwargs["routingMapOptional"] = routing_map

    def generate_routing_map_drop_pad_false(self, topk, num_tokens, num_experts):
        tensor = torch.zeros((num_tokens, num_experts), dtype=torch.bool)
        for i in range(num_tokens):
            true_num = random.randint(1, topk)
            indices = torch.randperm(num_experts)[:true_num]
            tensor[i, indices] = True
        return tensor

    def generate_routing_map_drop_pad_true(self, capacity, num_tokens, num_experts):
        tensor = torch.zeros((num_tokens, num_experts), dtype=torch.bool)
        for i in range(num_experts):
            indices = torch.randperm(num_tokens)[:capacity]
            tensor[indices, i] = True
        return tensor

    # 生成值域在 [-data_max，data_max]之间，数据类型为data_type的torch tensor
    def generate_tensor(self, shape, data_type, data_max):
        tensor = torch.rand(shape) * (data_max * 2) - data_max
        return tensor.to(data_type)

    def permute(
        self,
        tokens,  # [num_tokens, hidden]
        routing_map,  # [num_tokens, num_experts], bool
        probs: Optional[torch.Tensor] = None,
        num_out_tokens: Optional[int] = None,
        fused: bool = False,
        drop_and_pad: bool = False,
    ):
        num_tokens, hidden = tokens.shape
        num_experts = routing_map.shape[1]
        if drop_and_pad and num_out_tokens is not None:
            capacity = num_out_tokens // num_experts
            assert not routing_map.requires_grad
            routing_map = routing_map.to(
                dtype=torch.int8
            ).T.contiguous()  # [num_experts, num_tokens]
            sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[
                :, :capacity
            ].contiguous()  # [num_experts, capacity]
            sorted_indices = sorted_indices.view(-1)  # [num_experts * capacity,]
            if not self.prob_is_none:
                routing_map = routing_map.bool()  # [num_experts, num_tokens]
        else:
            routing_map = routing_map.bool().T.contiguous()  # [num_experts, num_tokens]
            token_indices = (
                torch.arange(num_tokens, device=routing_map.device)
                .unsqueeze(0)
                .expand(num_experts, -1)
            )  # [num_experts, num_tokens]
            sorted_indices = token_indices.masked_select(
                routing_map
            )  # 1-D [routing_map.sum(0).sum(-1),]
        permuted_input = tokens.index_select(0, sorted_indices)
        if not self.prob_is_none:
            permuted_probs = probs.T.contiguous().masked_select(routing_map)
        else:
            permuted_probs = None
        return permuted_input, permuted_probs, sorted_indices

    def permute_grad_with_routing_map(
        self, permuted_tokens, sorted_indices: torch.Tensor, num_tokens, hidden, topK
    ):
        # sorted_index, sorted_indices1 = sorted_indices.sort(dim=-1, descending=False)  # 升序排序
        output_tokens = torch.zeros(
            (num_tokens, hidden), dtype=torch.float32, device=permuted_tokens.device
        )
        for i in range(sorted_indices.size(-1)):
            if sorted_indices[i] >= 0:
                output_tokens[i // topK] += permuted_tokens[sorted_indices[i]]

        return output_tokens.to(dtype=permuted_tokens.dtype)

    def permute_grad_with_routing_map_drop(
        self,
        permuted_tokens,
        sorted_indices: torch.Tensor,
        num_tokens,
        hidden,
        capacity,
    ):
        sorted_values, sorted_indices = torch.sort(sorted_indices, stable=True)
        output_tokens = torch.zeros(
            (num_tokens, hidden),
            dtype=permuted_tokens.dtype,
            device=permuted_tokens.device,
        )
        for i in range(sorted_indices.size(-1)):
            output_tokens[sorted_values[i]] += permuted_tokens[sorted_indices[i]]

        return output_tokens.to(dtype=permuted_tokens.dtype)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        permutedTokenOutputGrad = input_data.kwargs["permutedTokenOutputGrad"]
        permutedProbsOutputGradOptional = input_data.kwargs[
            "permutedProbsOutputGradOptional"
        ]
        sortedIndices = input_data.kwargs["sortedIndices"]
        routingMapOptional = input_data.kwargs["routingMapOptional"]
        num_experts = input_data.kwargs["numExperts"]
        # self.torch_type = input_data.kwargs['permutedTokenOutputGrad'].dtype
        num_tokens = input_data.kwargs["tokensNum"]
        hidden_size = input_data.kwargs["permutedTokenOutputGrad"].shape[1]
        num_experts = input_data.kwargs["numExperts"]
        drop_and_pad = input_data.kwargs["padded_mode"]
        num_out_tokens = input_data.kwargs["permutedTokenOutputGrad"].shape[0]
        self.torch_type = input_data.kwargs["permutedTokenOutputGrad"].dtype
        if not self.prob_is_none:
            self.prob_type = input_data.kwargs["permutedProbsOutputGradOptional"].dtype
        else:
            self.prob_type = self.torch_type

        if self.device == "npu":
            permutedTokenOutputGrad = permutedTokenOutputGrad.npu()
            permutedProbsOutputGradOptional = permutedProbsOutputGradOptional.npu()
            sortedIndices = sortedIndices.npu()
            routingMapOptional = routingMapOptional.npu()
        if drop_and_pad is False:
            topK = self.safeDiv(num_out_tokens, num_tokens)

            permuted_tokens_grad = self.permute_grad_with_routing_map(
                permutedTokenOutputGrad, sortedIndices, num_tokens, hidden_size, topK
            )
            if not self.prob_is_none:
                probs_grad_permute = torch.zeros(
                    (num_experts, num_tokens), dtype=self.prob_type
                )
                probs_grad_permute.masked_scatter_(
                    routingMapOptional.T, permutedProbsOutputGradOptional
                )
                probs_grad_permute = probs_grad_permute.T
                return permuted_tokens_grad, probs_grad_permute
            else:
                return permuted_tokens_grad, torch.tensor([0])
        else:
            capacity = self.safeDiv(num_out_tokens, num_experts)
            permuted_tokens_grad = self.permute_grad_with_routing_map_drop(
                permutedTokenOutputGrad,
                sortedIndices,
                num_tokens,
                hidden_size,
                capacity,
            )
            if not self.prob_is_none:
                probs_grad_permute = torch.zeros(
                    (num_tokens, num_experts), dtype=self.prob_type
                )
                for i in range(num_experts * capacity):
                    dim0 = i // capacity
                    dim1 = sortedIndices[i]
                    probs_grad_permute[dim1, dim0] = permutedProbsOutputGradOptional[i]
                return permuted_tokens_grad, probs_grad_permute
            else:
                return permuted_tokens_grad, torch.tensor([0])


@register("AclnnBaseApi_aclnn_moe_token_permute_with_routing_map_grad")
class AclnnMoeTokenPermuteWithRoutingMapGradApi(AclnnBaseApi):
    def after_call(self, output_packages):
        case_id = self.task_result.case_config.id
        prob_is_none_list = [True, False]
        self.prob_is_none = prob_is_none_list[case_id % 2]
        # self.prob_is_none = False
        output = []
        torch_tensor = self.acl_tensor_to_torch(output_packages[0])
        output.append(torch_tensor)
        if not self.prob_is_none:
            output.append(self.acl_tensor_to_torch(output_packages[1]))
        else:
            output.append(torch.tensor([0]))
        return output
