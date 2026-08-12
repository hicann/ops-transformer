#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
import torch_npu
import ctypes
import math
import struct
import copy

from typing import Optional
from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


def topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = False,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
    norm_topk_prob: bool = False,
):
    assert logits.dim() == 2, (
        f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
    )
    num_tokens, num_experts = logits.shape

    def compute_topk(scores, topk):
        return torch.topk(scores, k=topk, dim=1)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            if expert_bias is not None:
                scores += expert_bias
            probs, top_indices = compute_topk(scores, topk)
        else:
            if expert_bias is not None:
                logits += expert_bias
            scores, top_indices = compute_topk(logits, topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float())
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk)
            scores = torch.gather(scores, dim=1, index=top_indices)
        else:
            scores, top_indices = compute_topk(scores, topk)
        probs = scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if topk >= 1 and norm_topk_prob:
        denominator = probs.sum(dim=-1, keepdim=True) + 1e-20
        probs = probs / denominator

    if scaling_factor:
        probs = probs * scaling_factor

    probs = probs.type_as(logits)
    return probs, top_indices, scores


@register("aclnn_moe_gating_top_k_backward")
class FunctionAclnnMoeGatingTopKBackward(AclnnBaseApi):
    def _to_int(self, val, default=0):
        if val is None:
            return default
        if isinstance(val, (list, tuple)):
            return int(val[0]) if val else default
        if isinstance(val, torch.Tensor):
            return int(val.item())
        return int(val)

    def _to_float(self, val, default=1.0):
        if val is None:
            return default
        if isinstance(val, (list, tuple)):
            return float(val[0]) if val else default
        if isinstance(val, torch.Tensor):
            return float(val.item())
        return float(val)

    def get_input_data(self, input_data: InputDataset):
        kw = input_data.kwargs
        x = kw["x"]
        print("get_input_data x.device", x.device)
        print("self.device", self.device)
        k = self._to_int(kw["k"])
        M = x.shape[0]
        N = x.shape[-1]
        gen = torch.Generator().manual_seed(k)
        bias = torch.randn(N, generator=gen, dtype=x.dtype) * 0.1
        grad_y = torch.randn((M, k), generator=gen, dtype=x.dtype)
        rsf = self._to_float(kw.get("routed_scaling_factor", 1.0), default=1.0)
        eps = self._to_float(kw.get("eps", 1e-20), default=1e-20)
        return x, k, bias, rsf, eps, grad_y

    def init_by_input_data(self, input_data):
        import random
        import ctypes
        from atk.tasks.backends.lib_interface.acl_wrapper import (
            nnopbase,
            AclDataType,
            AclFormat,
        )

        x, k, bias, rsf, eps, grad_y = self.get_input_data(input_data)
        x_npu = x.clone().npu().requires_grad_(True)
        bias_npu = bias.clone().npu()
        y_out, expert_idx_out_npu, scores_npu = topk_routing_with_score_function(
            x_npu,
            topk=k,
            score_function="sigmoid",
            expert_bias=bias_npu,
            norm_topk_prob=True,
            scaling_factor=rsf,
        )
        x_norm = torch.sigmoid(x_npu.float())  # 构造x_norm
        expert_idx_out_npu = expert_idx_out_npu.to(torch.int32)
        grad_y_npu = grad_y.clone().npu().to(y_out.dtype)

        input_args, output_packages = super().init_by_input_data(input_data)
        input_args[3] = copy.deepcopy(input_args[6])  # renorm
        input_args[4] = copy.deepcopy(input_args[7])  # normtype
        input_args[5] = copy.deepcopy(input_args[9])  # rsf
        input_args[6] = copy.deepcopy(input_args[10])  # eps

        input_args.pop(10)
        input_args.pop(9)
        input_args.pop(8)
        input_args.pop(7)

        input_args[0] = self.backend.torch_tensor_to_acl(x_norm)
        input_args[1] = self.backend.torch_tensor_to_acl(grad_y_npu)
        input_args[2] = self.backend.torch_tensor_to_acl(expert_idx_out_npu)

        return input_args, output_packages


@register("execute_torch_npu_moe_gating_top_k_fwd_bwd")
class FunctionMoeGatingTopKFwdBwd(BaseApi):
    def _to_int(self, val, default=0):
        if val is None:
            return default
        if isinstance(val, (list, tuple)):
            return int(val[0]) if val else default
        if isinstance(val, torch.Tensor):
            return int(val.item())
        return int(val)

    def _to_float(self, val, default=1.0):
        if val is None:
            return default
        if isinstance(val, (list, tuple)):
            return float(val[0]) if val else default
        if isinstance(val, torch.Tensor):
            return float(val.item())
        return float(val)

    def get_input_data(self, input_data: InputDataset):
        kw = input_data.kwargs
        x = kw["x"]
        k = self._to_int(kw["k"])
        M = x.shape[0]
        N = x.shape[-1]
        gen = torch.Generator().manual_seed(k)
        bias = torch.randn(N, generator=gen, dtype=x.dtype) * 0.1
        grad_y = torch.randn((M, k), generator=gen, dtype=x.dtype)
        rsf = self._to_float(kw.get("routed_scaling_factor", 1.0), default=1.0)
        eps = self._to_float(kw.get("eps", 1e-20), default=1e-20)
        return x, k, bias, rsf, eps, grad_y

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x, k, bias, rsf, eps, grad_y = self.get_input_data(input_data)
        if self.device == "cpu":
            print(
                "=================================== run cpu ================================="
            )
            x_cpu = x.clone().requires_grad_(True)
            bias_cpu = bias.clone()
            y_out, _, _ = topk_routing_with_score_function(
                x_cpu,
                topk=k,
                score_function="sigmoid",
                expert_bias=bias_cpu,
                norm_topk_prob=True,
                scaling_factor=rsf,
            )
            y_out.backward(grad_y.clone().to(y_out.dtype))
            grad_x_cpu = x_cpu.grad.detach().clone()
            return grad_x_cpu
        elif self.device == "npu" and self.name == "bench":
            print(
                "=================================== run npu bench ================================="
            )
            x_bench = x.clone().npu().requires_grad_(True)
            bias_bench = bias.clone().npu()
            y_out, expert_idx_out_bench, scores_bench = (
                topk_routing_with_score_function(
                    x_bench,
                    topk=k,
                    score_function="sigmoid",
                    expert_bias=bias_bench,
                    norm_topk_prob=True,
                    scaling_factor=rsf,
                )
            )
            y_out.backward(grad_y.clone().npu().to(y_out.dtype))
            grad_x_bench = x_bench.grad.detach().clone()
            return grad_x_bench
