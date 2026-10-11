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
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


PADDED = "padded"
NON_PADDED = "non_padded"
NON_PADDED_MINUS_ONE = "non_padded_minus_one"
CASE_MODES = (PADDED, NON_PADDED, NON_PADDED_MINUS_ONE)


def case_mode(case_id):
    return CASE_MODES[case_id % len(CASE_MODES)]


def safe_div(dividend, divisor):
    return 0 if divisor == 0 else dividend // divisor


def generate_tensor(shape, dtype, data_max=5):
    return (torch.rand(shape) * (2 * data_max) - data_max).to(dtype)


def has_special_values(tensor):
    if tensor is None or tensor.device.type != "cpu":
        return False
    return bool(torch.any(torch.isinf(tensor)) or torch.any(torch.isnan(tensor)))


def target_device(api):
    if api.device in ("pyaclnn", "npu"):
        return f"npu:{api.device_id}"
    if api.device == "gpu":
        return f"cuda:{api.device_id}"
    return None


def to_target(tensor, device, dtype=None):
    if dtype is not None:
        return tensor.to(device=device, dtype=dtype) if device else tensor.to(dtype)
    return tensor.to(device=device) if device else tensor


def make_padded_routing_map(capacity, num_tokens, num_experts):
    routing_map = torch.zeros((num_tokens, num_experts), dtype=torch.bool)
    if capacity <= 0 or num_tokens <= 0 or num_experts <= 0:
        return routing_map

    token_indices = torch.arange(num_experts * capacity, dtype=torch.long) % num_tokens
    expert_indices = torch.arange(num_experts, dtype=torch.long).repeat_interleave(
        capacity
    )
    routing_map[token_indices, expert_indices] = True
    return routing_map


def first_route_mask(routing_map, dtype):
    mask = torch.zeros_like(routing_map, dtype=torch.bool)
    has_route = routing_map.any(dim=1)
    if torch.any(has_route):
        rows = torch.arange(routing_map.shape[0], dtype=torch.long)[has_route]
        cols = routing_map.to(torch.int32).argmax(dim=1)[has_route].to(torch.long)
        mask[rows, cols] = True
    return mask.to(dtype)


def make_non_padded_inputs(num_tokens, num_experts, topk, prob_dtype, with_minus_one):
    routing_map = torch.zeros((num_tokens, num_experts), dtype=torch.bool)
    probs = torch.zeros((num_tokens, num_experts), dtype=prob_dtype)
    sorted_indices = torch.full((num_tokens * topk,), -1, dtype=torch.int32)

    permuted_row = 0
    for token_idx in range(num_tokens):
        experts = torch.randperm(num_experts)[:topk].sort().values
        for topk_idx, expert_tensor in enumerate(experts):
            expert_idx = int(expert_tensor.item())
            slot = token_idx * topk + topk_idx
            routing_map[token_idx, expert_idx] = True

            invalid_slot = with_minus_one and (
                (token_idx % 4 == 0 and topk_idx == 0)
                or (token_idx % 4 == 1 and topk_idx == topk - 1)
                or (token_idx % 4 == 3 and topk_idx % 2 == 0)
            )
            if invalid_slot:
                continue

            sorted_indices[slot] = permuted_row
            probs[token_idx, expert_idx] = (
                torch.rand((), dtype=torch.float32) * 2 - 1
            ).to(prob_dtype)
            permuted_row += 1

    return routing_map, probs, sorted_indices


def permute_padded(tokens, routing_map, probs, num_out_tokens):
    num_experts = routing_map.shape[1]
    capacity = safe_div(num_out_tokens, num_experts)
    routing_map_t = routing_map.to(dtype=torch.int8).T.contiguous()
    sorted_indices = (
        routing_map_t.argsort(dim=-1, descending=True, stable=True)[:, :capacity]
        .contiguous()
        .view(-1)
    )
    permuted_tokens = tokens.index_select(0, sorted_indices)
    return permuted_tokens, sorted_indices, routing_map


def normalize_flat_probs(probs, expected_size):
    if probs.numel() < expected_size:
        padding = torch.zeros(
            (expected_size - probs.numel(),), dtype=probs.dtype, device=probs.device
        )
        return torch.cat((probs, padding), dim=0)
    return probs[:expected_size]


def unpermute_padded(
    permuted_tokens,
    sorted_indices,
    probs,
    restore_shape,
    is_mixed,
    compute_dtype,
    output_dtype,
):
    original_dtype = permuted_tokens.dtype
    if is_mixed:
        permuted_tokens = permuted_tokens.to(compute_dtype)
        probs = probs.to(compute_dtype)

    num_tokens, _ = restore_shape
    num_experts = probs.size(1)
    capacity = safe_div(sorted_indices.size(0), num_experts)

    expert_indices = torch.arange(num_experts, device=permuted_tokens.device).unsqueeze(
        -1
    )
    token_indices = sorted_indices.view(num_experts, capacity)
    prob_indices = (
        (expert_indices * num_tokens + token_indices)
        .view(-1)
        .clamp(0, num_experts * num_tokens - 1)
    )
    permuted_probs = probs.T.contiguous().view(-1).index_select(0, prob_indices)
    weighted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    sorted_values, sorted_order = sorted_indices.sort(
        dim=-1, stable=True, descending=False
    )
    output_tokens = torch.zeros(
        restore_shape, dtype=compute_dtype, device=permuted_tokens.device
    )
    output_tokens.index_add_(0, sorted_values, weighted_tokens[sorted_order])

    if is_mixed:
        output_tokens = output_tokens.to(output_dtype)
        permuted_probs = permuted_probs.to(output_dtype)

    return (
        output_tokens.to(original_dtype),
        sorted_order.to(sorted_indices.dtype),
        sorted_values.to(sorted_indices.dtype),
        permuted_probs.to(probs.dtype),
    )


def unpermute_non_padded(
    permuted_tokens, sorted_indices, routing_map, probs, restore_shape
):
    num_tokens = restore_shape[0]
    topk = safe_div(sorted_indices.numel(), num_tokens)
    compute_dtype = (
        torch.float32
        if permuted_tokens.dtype in (torch.float16, torch.bfloat16)
        else permuted_tokens.dtype
    )
    permuted_probs = probs.masked_select(routing_map.to(torch.bool)).to(compute_dtype)

    if num_tokens == 0 or topk == 0 or permuted_tokens.numel() == 0:
        output_tokens = torch.zeros(
            restore_shape, dtype=compute_dtype, device=permuted_tokens.device
        )
        dummy_index = torch.tensor(
            [1], dtype=sorted_indices.dtype, device=sorted_indices.device
        )
        return (
            output_tokens.to(permuted_tokens.dtype),
            dummy_index,
            dummy_index,
            permuted_probs.to(probs.dtype),
        )

    permuted_probs = normalize_flat_probs(permuted_probs, num_tokens * topk)
    valid_mask = (sorted_indices >= 0) & (sorted_indices < permuted_tokens.shape[0])
    gather_indices = sorted_indices.clamp(0, permuted_tokens.shape[0] - 1).to(
        torch.long
    )
    gathered_tokens = permuted_tokens.index_select(0, gather_indices).reshape(
        num_tokens, topk, permuted_tokens.shape[1]
    )

    weights = permuted_probs.reshape(num_tokens, topk).to(compute_dtype)
    valid_weights = valid_mask.reshape(num_tokens, topk).to(compute_dtype)
    output_tokens = (
        gathered_tokens.to(compute_dtype)
        * weights.unsqueeze(-1)
        * valid_weights.unsqueeze(-1)
    ).sum(dim=1)

    dummy_index = torch.tensor(
        [1], dtype=sorted_indices.dtype, device=permuted_tokens.device
    )
    return (
        output_tokens.to(permuted_tokens.dtype),
        dummy_index,
        dummy_index,
        permuted_probs.to(probs.dtype),
    )


@register("ascend_function_moe_token_unpermute_with_routing_map")
class FunctionMoeTokenUnpermuteWithRoutingMapApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.mode = PADDED
        self.is_mixed = False
        self.compute_dtype = None
        self.output_dtype = None

    def init_by_input_data(self, input_data: InputDataset):
        case_id = self.task_result.case_config.id
        OpsDataset.seed_everything(case_id)
        self.mode = case_mode(case_id)

        if self.mode == PADDED:
            self.init_padded(input_data)
        else:
            self.init_non_padded(input_data, self.mode == NON_PADDED_MINUS_ONE)

    def init_padded(self, input_data: InputDataset):
        token_input = input_data.kwargs["permutedTokens"]
        probs_input = input_data.kwargs["probsOptional"]
        sorted_indices_input = input_data.kwargs["sortedIndices"]
        restore_shape_input = input_data.kwargs["restoreShapeOptional"]

        token_dtype = token_input.dtype
        prob_dtype = probs_input.dtype
        hidden_size = restore_shape_input[1]
        num_tokens, num_experts = probs_input.shape
        num_out_tokens = sorted_indices_input.shape[0]
        capacity = safe_div(num_out_tokens, num_experts)

        self.is_mixed = (
            token_dtype in (torch.bfloat16, torch.float16)
            and prob_dtype == torch.float32
        )
        self.compute_dtype = torch.float32 if self.is_mixed else token_dtype
        self.output_dtype = token_dtype

        if has_special_values(token_input) or has_special_values(probs_input):
            permuted_tokens = token_input
            probs = probs_input.to(torch.float32 if self.is_mixed else token_dtype)
            sorted_indices = sorted_indices_input.to(torch.int32)
            routing_map = input_data.kwargs["routingMapOptional"]
            restore_shape = restore_shape_input
        else:
            original_tokens = generate_tensor((num_tokens, hidden_size), token_dtype)
            routing_map = make_padded_routing_map(capacity, num_tokens, num_experts)
            probs = generate_tensor(
                (num_tokens, num_experts),
                torch.float32 if self.is_mixed else prob_dtype,
                1.0,
            )
            probs = probs * first_route_mask(routing_map, probs.dtype)
            permuted_tokens, sorted_indices, routing_map = permute_padded(
                original_tokens, routing_map, probs, num_out_tokens
            )
            restore_shape = [num_tokens, hidden_size]

        device = target_device(self)
        probs_dtype = torch.float32 if self.is_mixed else token_dtype
        input_data.kwargs["permutedTokens"] = to_target(
            permuted_tokens, device, token_dtype
        )
        input_data.kwargs["sortedIndices"] = to_target(
            sorted_indices, device, torch.int32
        )
        input_data.kwargs["routingMapOptional"] = to_target(routing_map, device)
        input_data.kwargs["probsOptional"] = to_target(probs, device, probs_dtype)
        input_data.kwargs["paddedMode"] = True
        input_data.kwargs["restoreShapeOptional"] = restore_shape

    def init_non_padded(self, input_data: InputDataset, with_minus_one):
        token_input = input_data.kwargs["permutedTokens"]
        routing_input = input_data.kwargs["routingMapOptional"]
        probs_input = input_data.kwargs["probsOptional"]

        token_dtype = token_input.dtype
        num_tokens = max(1, min(int(routing_input.shape[0]), 128))
        num_experts = max(1, min(int(routing_input.shape[1]), 64))
        hidden_size = max(1, min(int(token_input.shape[1]), 512))
        topk = min(16, num_experts)

        self.compute_dtype = (
            torch.float32
            if token_dtype in (torch.float16, torch.bfloat16)
            else token_dtype
        )
        routing_map, probs, sorted_indices = make_non_padded_inputs(
            num_tokens, num_experts, topk, probs_input.dtype, with_minus_one
        )
        permuted_tokens = generate_tensor((num_tokens * topk, hidden_size), token_dtype)

        device = target_device(self)
        input_data.kwargs["permutedTokens"] = to_target(permuted_tokens, device)
        input_data.kwargs["sortedIndices"] = to_target(sorted_indices, device)
        input_data.kwargs["routingMapOptional"] = to_target(
            routing_map, device, routing_input.dtype
        )
        input_data.kwargs["probsOptional"] = to_target(probs, device)
        input_data.kwargs["paddedMode"] = False
        input_data.kwargs["restoreShapeOptional"] = [num_tokens, hidden_size]

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        device = target_device(self)
        permuted_tokens = to_target(input_data.kwargs["permutedTokens"], device)
        sorted_indices = to_target(input_data.kwargs["sortedIndices"], device)
        routing_map = to_target(input_data.kwargs["routingMapOptional"], device)
        probs = to_target(input_data.kwargs["probsOptional"], device)
        restore_shape = input_data.kwargs["restoreShapeOptional"]

        if self.mode == PADDED:
            return unpermute_padded(
                permuted_tokens,
                sorted_indices,
                probs,
                restore_shape,
                self.is_mixed,
                self.compute_dtype,
                self.output_dtype,
            )
        return unpermute_non_padded(
            permuted_tokens, sorted_indices, routing_map, probs, restore_shape
        )


@register("aclnn_function")
class AclnnMoeTokenUnpermuteWithRoutingMapApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        self.mode = case_mode(self.task_result.case_config.id)
        return super().init_by_input_data(input_data)

    def after_call(self, output_packages):
        if self.mode == PADDED:
            return [
                self.acl_tensor_to_torch(output_package)
                for output_package in output_packages
            ]

        output_tokens = self.acl_tensor_to_torch(output_packages[0])
        return [
            output_tokens,
            torch.tensor([1], dtype=torch.int32, device=output_tokens.device),
            torch.tensor([1], dtype=torch.int32, device=output_tokens.device),
            self.acl_tensor_to_torch(output_packages[3]),
        ]
