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
from matmul_reduce_scatter_test_options import OPTION_NAMES, OPTIONS

SIGNATURE = """aclnnStatus aclnnMatmulReduceScatterGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const char* group,
    const char* reduceOp, int64_t commTurn, int64_t streamMode, const aclTensor* output,
    uint64_t* workspaceSize, aclOpExecutor** executor);"""

INPUT_NAMES = ("x1", "x2", "bias", "group", "reduceOp", "commTurn", "streamMode")


def rank_inputs(data, task_result, options):
    data = dict(data)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        20261119
        + int(task_result.case_config.id) * 1000003
        + int(task_result.dist_task_info.rank) * 1009
    )
    x = data["x1"]
    indices = torch.randperm(x.shape[-1], generator=generator, device="cpu")
    data["x1"] = x.index_select(-1, indices.to(x.device))
    if options["transpose_x2"]:
        data["x2"] = data["x2"].T
    return data


def golden(data, options, info):
    output = data["x1"].float() @ data["x2"].float()
    if options["use_bias"] and data["bias"].numel():
        output = output.double() + data["bias"].double()
    dist.all_reduce(output)
    return output.chunk(info.world_size, dim=0)[info.rank].contiguous()


def benchmark(data, options, info):
    output = data["x1"] @ data["x2"]
    if options["use_bias"] and data["bias"].numel():
        output = (output.float() + data["bias"].float()).to(output.dtype)
    result = torch.empty(
        (output.shape[0] // info.world_size, output.shape[1]),
        dtype=output.dtype,
        device=output.device,
    )
    dist.reduce_scatter_tensor(result, output.contiguous())
    return result


@register("matmul_reduce_scatter_reference")
class Reference(BaseApi):
    @staticmethod
    def get_cpp_func_signature_type():
        return SIGNATURE

    def __call__(self, input_data, with_output=False):
        options = dict(zip(OPTION_NAMES, OPTIONS[input_data.kwargs["test_options"]]))
        data = rank_inputs(input_data.kwargs, self.task_result, options)
        reference = golden if self.name == "cpu" else benchmark
        return reference(data, options, self.task_result.dist_task_info)


@register("matmul_reduce_scatter_aclnn")
class AclnnExecutor(AclnnBaseApi):
    @staticmethod
    def get_cpp_func_signature_type():
        return SIGNATURE

    def init_by_input_data(self, input_data):
        data = dict(input_data.kwargs)
        options = dict(zip(OPTION_NAMES, OPTIONS[data.pop("test_options")]))
        data = rank_inputs(data, self.task_result, options)
        info = self.task_result.dist_task_info
        data["group"] = dist.group.WORLD._get_backend(
            torch.device("npu")
        ).get_hccl_comm_name(info.rank)
        input_data.kwargs = {name: data[name] for name in INPUT_NAMES}
        args, outputs = super().init_by_input_data(input_data)
        if not options["use_bias"] or not data["bias"].numel():
            args[2] = ctypes.POINTER(AclTensor)()
        x, weight = data["x1"], data["x2"]
        output = self.torch_tensor_to_acl(
            torch.empty(
                (x.shape[0] // info.world_size, weight.shape[1]),
                dtype=x.dtype,
                device=x.device,
            )
        )
        args[-1] = output
        outputs[:] = [output]
        return args, outputs

    def after_call(self, output_packages):
        return [self.acl_tensor_to_torch(output) for output in output_packages]

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()
