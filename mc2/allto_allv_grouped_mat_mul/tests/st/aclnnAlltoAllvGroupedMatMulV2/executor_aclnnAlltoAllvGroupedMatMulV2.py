# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A3 grouped matmul and communication, preserving the frozen eight-rank layout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import torch
import torch.distributed as dist
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
from allto_allv_grouped_mat_mul_test_options import OPTION_NAMES, OPTIONS


SIGNATURE = (
    "aclnnStatus aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize("
    "const aclTensor* gmmX, const aclTensor* gmmWeight, const aclTensor* sendCountsTensorOptional, "
    "const aclTensor* recvCountsTensorOptional, const aclTensor* mmXOptional, const aclTensor* mmWeightOptional, "
    "const char* group, const char* commMode, int64_t epWorldSize, "
    "const aclIntArray* sendCounts, const aclIntArray* recvCounts, bool transGmmWeight, "
    "bool transMmWeight, bool permuteOutFlag, aclTensor* gmmY, "
    "aclTensor* mmYOptional, aclTensor* permuteOutOptional, uint64_t* workspaceSize, "
    "aclOpExecutor** executor)"
)


def rank_inputs(kwargs, task_result):
    values = dict(kwargs)
    options = dict(zip(OPTION_NAMES, OPTIONS[values.pop("test_options")]))
    x = values["gmmX"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        20261119
        + int(task_result.case_config.id) * 1000003
        + int(task_result.dist_task_info.rank) * 1009
    )
    indices = torch.randperm(x.shape[-1], generator=generator, device="cpu")
    values["gmmX"] = x.index_select(-1, indices.to(x.device))
    return values, options


def gather_experts(values, rank):
    x, weights = values["gmmX"], values["gmmWeight"]
    world, experts = values["epWorldSize"], weights.shape[0]
    counts = values["sendCounts"]
    received = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(received, x)
    # Every source uses the frozen peer-major count vector.
    offset = sum(counts[: rank * experts])
    expert_inputs = []
    for expert in range(experts):
        count = counts[rank * experts + expert]
        expert_inputs.append(
            torch.cat([source[offset : offset + count] for source in received], dim=0)
        )
        offset += count
    return expert_inputs


def golden(values, rank):
    expert_inputs = gather_experts(values, rank)
    weights = values["gmmWeight"]
    result = torch.cat(
        [
            tokens.float() @ weights[expert].float()
            for expert, tokens in enumerate(expert_inputs)
        ],
        dim=0,
    )
    shared = values["mmXOptional"].float() @ values["mmWeightOptional"].float()
    return result, shared, torch.cat(expert_inputs, dim=0)


def benchmark(values, rank):
    expert_inputs = gather_experts(values, rank)
    weights = values["gmmWeight"]
    result = torch.cat(
        [tokens @ weights[expert] for expert, tokens in enumerate(expert_inputs)], dim=0
    )
    shared = values["mmXOptional"] @ values["mmWeightOptional"]
    return result, shared, torch.cat(expert_inputs, dim=0)


@register("a3_function_allto_allv_gmm_v2")
class Reference(BaseApi):
    signature = SIGNATURE

    def get_cpp_func_signature_type(self):
        return self.signature

    def __call__(self, input_data, with_output=False):
        values, options = rank_inputs(input_data.kwargs, self.task_result)
        if self.name == "cpu":
            return golden(values, self.task_result.dist_task_info.rank)
        return benchmark(values, self.task_result.dist_task_info.rank)


@register("a3_aclnn_allto_allv_gmm_v2")
class AclnnExecutor(AclnnBaseApi):
    signature = SIGNATURE
    input_count = 14

    def get_cpp_func_signature_type(self):
        return self.signature

    def init_by_input_data(self, input_data):
        values, options = rank_inputs(input_data.kwargs, self.task_result)
        if values.get("commMode") == "aiv":
            os.environ["HCCL_BUFFSIZE"] = "2048"
        from torch.distributed.distributed_c10d import _get_default_group

        self.group = _get_default_group()
        values["group"] = self.group._get_backend(
            torch.device("npu")
        ).get_hccl_comm_name(dist.get_rank())
        if values["transGmmWeight"]:
            values["gmmWeight"] = values["gmmWeight"].transpose(1, 2).contiguous()
        if values["transMmWeight"]:
            values["mmWeightOptional"] = values["mmWeightOptional"].T.contiguous()
        input_data.kwargs = values
        args, _ = super().init_by_input_data(input_data)
        # Output shapes follow the native API, independent of CPU metadata containers.
        rows = sum(values["recvCounts"])
        width = values["gmmWeight"].shape[1 if values["transGmmWeight"] else 2]
        shapes = [(rows, width)]
        shared_width = values["mmWeightOptional"].shape[
            0 if values["transMmWeight"] else 1
        ]
        shapes += [
            (values["mmXOptional"].shape[0], shared_width),
            (rows, values["gmmX"].shape[1]),
        ]
        outputs = [
            self.torch_tensor_to_acl(
                torch.empty(
                    shape, dtype=values["gmmX"].dtype, device=values["gmmX"].device
                )
            )
            for shape in shapes
        ]
        args = args[: self.input_count] + outputs
        args[2:4] = [TensorPtr(), TensorPtr()]
        return args, outputs

    def after_call(self, output_packages):
        return [self.acl_tensor_to_torch(output) for output in output_packages]

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()
