# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ctypes

import torch
import torch.distributed as dist
import torch_npu

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from matmul_allto_all_test_options import OPTION_NAMES, OPTIONS

SIGNATURE = """aclnnStatus aclnnMatmulAlltoAllGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
    const aclIntArray* alltoAllAxesOptional, const char* group,
    bool transposeX1, bool transposeX2, const aclTensor* output,
    uint64_t* workspaceSize, aclOpExecutor** executor)"""


def rank_inputs(kwargs, task_result):
    data = dict(kwargs)
    options = dict(zip(OPTION_NAMES, OPTIONS[data.pop("test_options")]))
    if options["deterministic"] is not None:
        torch.use_deterministic_algorithms(options["deterministic"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        20261119
        + int(task_result.case_config.id) * 1000003
        + int(task_result.dist_task_info.rank) * 1009
    )
    x = data["x1"]
    indices = torch.randperm(x.shape[-1], generator=generator, device="cpu")
    data["x1"] = x.index_select(-1, indices.to(x.device))
    return data, options


def exchange(output, info, cpu):
    if cpu:
        ranks = [torch.empty_like(output) for _ in range(info.world_size)]
        dist.all_gather(ranks, output)
        received = [value.chunk(info.world_size, dim=1)[info.rank] for value in ranks]
    else:
        sent = [value.contiguous() for value in output.chunk(info.world_size, dim=1)]
        received = [torch.empty_like(sent[0]) for _ in range(info.world_size)]
        dist.all_to_all(received, sent)
    return torch.cat(received, dim=0)


def golden(data, options, info):
    x = data["x1"]
    weight = data["x2"].T if data["transposeX2"] else data["x2"]
    output = x.float() @ weight.float()
    if options["use_bias"]:
        bias = data["biasOptional"]
        if x.dtype == torch.bfloat16:
            output = output.double() + bias.double()
        else:
            output = output + bias.float()
    return exchange(output, info, cpu=True)


def benchmark(data, options, info):
    x = data["x1"]
    weight = data["x2"].T if data["transposeX2"] else data["x2"]
    output = x @ weight
    if options["use_bias"]:
        bias = data["biasOptional"]
        output = (output + bias).to(x.dtype)
    return exchange(output, info, cpu=False)


@register("function_aclnn_matmul_allto_all")
class Reference(BaseApi):
    def get_cpp_func_signature_type(self):
        return SIGNATURE

    def __call__(self, input_data, with_output=False):
        data, options = rank_inputs(input_data.kwargs, self.task_result)
        if self.name == "cpu":
            return golden(data, options, self.task_result.dist_task_info)
        return benchmark(data, options, self.task_result.dist_task_info)


@register("pyaclnn_aclnn_matmul_allto_all")
class AclnnExecutor(AclnnBaseApi):
    def get_cpp_func_signature_type(self):
        return SIGNATURE

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def init_by_input_data(self, input_data):
        data, options = rank_inputs(input_data.kwargs, self.task_result)
        from torch.distributed.distributed_c10d import _get_default_group

        data["group"] = (
            _get_default_group()
            ._get_backend(torch.device("npu"))
            .get_hccl_comm_name(self.task_result.dist_task_info.rank)
        )
        input_data.kwargs = data
        args, outputs = super().init_by_input_data(input_data)
        if not options["use_bias"]:
            args[2] = ctypes.POINTER(AclTensor)()
        x = data["x1"]
        size = self.task_result.dist_task_info.world_size
        n = data["x2"].shape[0 if data["transposeX2"] else 1]
        output = torch.empty(
            (x.shape[0] * size, n // size), dtype=x.dtype, device=x.device
        )
        outputs[0] = self.torch_tensor_to_acl(output)
        args[7] = outputs[0]
        return args, outputs

    def after_call(self, output_packages):
        return [self.acl_tensor_to_torch(package) for package in output_packages]
