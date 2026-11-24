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

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from matmul_all_reduce_test_options import OPTION_NAMES, OPTIONS


SIGNATURE = (
    "aclnnStatus aclnnMatmulAllReduceV2GetWorkspaceSize("
    "const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3, "
    "const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode, "
    "const aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)"
)


def rank_inputs(kwargs, task_result):
    data = dict(kwargs)
    options = dict(zip(OPTION_NAMES, OPTIONS[data.pop("test_options")]))
    x = data["x1"]
    # The frozen CSV describes eight shards even for the two-rank task.
    rows = x.shape[0] // 8
    if options["input_distribution"] == "rank_shards" and rows:
        start = task_result.dist_task_info.rank * rows
        x = x[start : start + rows]
    data["x1"] = x
    if options["transpose_x2"]:
        data["x2"] = data["x2"].T
    return data


def golden(data):
    output = data["x1"].float() @ data["x2"].float()
    if data["bias"].numel():
        output = output + data["bias"].float()
    if data["x3"].numel():
        output = output + data["x3"].float()
    dist.all_reduce(output)
    return output


def benchmark(data):
    output = data["x1"] @ data["x2"]
    if data["bias"].numel():
        output = output + data["bias"]
    if data["x3"].numel():
        output = output + data["x3"]
    dist.all_reduce(output)
    return output


@register("execute_aclnnMatmulAllReduceV2")
class Reference(BaseApi):
    def get_cpp_func_signature_type(self):
        return SIGNATURE

    def __call__(self, input_data, with_output=False):
        data = rank_inputs(input_data.kwargs, self.task_result)
        return golden(data) if self.name == "cpu" else benchmark(data)


@register("pyaclnn_aclnnMatmulAllReduceV2")
class AclnnExecutor(AclnnBaseApi):
    def get_cpp_func_signature_type(self):
        return SIGNATURE

    def init_by_input_data(self, input_data):
        data = rank_inputs(input_data.kwargs, self.task_result)
        data["group"] = dist.group.WORLD._get_backend(
            torch.device("npu")
        ).get_hccl_comm_name(self.task_result.dist_task_info.rank)
        input_data.kwargs = {
            name: data[name]
            for name in (
                "x1",
                "x2",
                "bias",
                "x3",
                "group",
                "reduceOp",
                "commTurn",
                "streamMode",
            )
        }
        args, outputs = super().init_by_input_data(input_data)
        null = ctypes.POINTER(AclTensor)()
        if not data["bias"].numel():
            args[2] = null
        if not data["x3"].numel():
            args[3] = null
        output = torch.empty(
            (data["x1"].shape[0], data["x2"].shape[1]),
            dtype=data["x1"].dtype,
            device=data["x1"].device,
        )
        outputs[0] = self.torch_tensor_to_acl(output)
        args[-1] = outputs[0]
        return args, outputs

    def after_call(self, output_packages):
        return [self.acl_tensor_to_torch(output) for output in output_packages]

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()
