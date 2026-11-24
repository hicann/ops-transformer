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
from all_gather_matmul_test_options import OPTION_NAMES, OPTIONS

SIGNATURE = """aclnnStatus aclnnAllGatherMatmulGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const char* group,
    int64_t gatherIndex, int64_t commTurn, int64_t streamMode, const aclTensor* output,
    const aclTensor* gatherOut, uint64_t* workspaceSize, aclOpExecutor** executor);"""

INPUT_NAMES = ("x1", "x2", "bias", "group", "gatherIndex", "commTurn", "streamMode")


def rank_inputs(data, task_result, options):
    data = dict(data)
    info = task_result.dist_task_info
    rows = data["x1"].shape[0] // info.world_size
    start = info.rank * rows
    data["x1"] = data["x1"][start : start + rows].contiguous()
    if options["transpose_x2"]:
        data["x2"] = data["x2"].T
    return data


def golden(data, options, info):
    pieces = [torch.empty_like(data["x1"]) for _ in range(info.world_size)]
    dist.all_gather(pieces, data["x1"])
    gathered = torch.cat(pieces)
    output = gathered.float() @ data["x2"].float()
    if options["use_bias"] and data["bias"].numel():
        output = output.double() + data["bias"].double()
    return output, gathered if options["comm_output"] else None


def benchmark(data, options, info):
    pieces = [torch.empty_like(data["x1"]) for _ in range(info.world_size)]
    dist.all_gather(pieces, data["x1"])
    gathered = torch.cat(pieces)
    output = gathered @ data["x2"]
    if options["use_bias"] and data["bias"].numel():
        output = (output.float() + data["bias"].float()).to(output.dtype)
    return output, gathered if options["comm_output"] else None


@register("all_gather_matmul_reference")
class Reference(BaseApi):
    @staticmethod
    def get_cpp_func_signature_type():
        return SIGNATURE

    def __call__(self, input_data, with_output=False):
        options = dict(zip(OPTION_NAMES, OPTIONS[input_data.kwargs["test_options"]]))
        data = rank_inputs(input_data.kwargs, self.task_result, options)
        reference = golden if self.name == "cpu" else benchmark
        return reference(data, options, self.task_result.dist_task_info)


@register("all_gather_matmul_aclnn")
class AclnnExecutor(AclnnBaseApi):
    @staticmethod
    def get_cpp_func_signature_type():
        return SIGNATURE

    def init_by_input_data(self, input_data):
        data = dict(input_data.kwargs)
        options = dict(zip(OPTION_NAMES, OPTIONS[data.pop("test_options")]))
        data = rank_inputs(data, self.task_result, options)
        info = self.task_result.dist_task_info
        self.comm_output = options["comm_output"]
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
                (x.shape[0] * info.world_size, weight.shape[1]),
                dtype=x.dtype,
                device=x.device,
            )
        )
        # V1 documents an empty tensor to disable gatherOut; it is not a nullptr ABI.
        gather_shape = (
            (x.shape[0] * info.world_size, x.shape[1]) if self.comm_output else (0, 0)
        )
        gathered = self.torch_tensor_to_acl(
            torch.empty(gather_shape, dtype=x.dtype, device=x.device)
        )
        args = args[: len(INPUT_NAMES)] + [output, gathered]
        outputs[:] = [output, gathered] if self.comm_output else [output]
        return args, outputs

    def after_call(self, output_packages):
        output = self.acl_tensor_to_torch(output_packages[0])
        gathered = (
            self.acl_tensor_to_torch(output_packages[1]) if self.comm_output else None
        )
        return output, gathered

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()
