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
from atk.tasks.backends.lib_interface.acl_wrapper import (
    AclTensor,
    AclFormat,
    AclDataType,
    nnopbase,
)
from matmul_all_reduce_test_options import OPTION_NAMES, OPTIONS


SIGNATURE = (
    "aclnnStatus aclnnWeightQuantMatmulAllReduceGetWorkspaceSize("
    "const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* antiquantScale, "
    "const aclTensor* antiquantOffset, const aclTensor* x3, const char* group, const char* reduceOp, "
    "int64_t commTurn, int64_t streamMode, int64_t antiquantGroupSize, const aclTensor* output, "
    "uint64_t* workspaceSize, aclOpExecutor** executor)"
)


def rank_inputs(kwargs, task_result):
    data = dict(kwargs)
    options = dict(zip(OPTION_NAMES, OPTIONS[data.pop("test_options")]))
    x, weight = data["x1"], data["x2"]
    # Keep the frozen eight-shard input description and the task's local shape.
    rows = x.shape[0] // 8
    if options["input_distribution"] == "rank_shards" and rows:
        start = task_result.dist_task_info.rank * rows
        x = x[start : start + rows]
    if options["x2_nz"] and weight.device.type != "cpu":
        torch_npu.npu.config.allow_internal_format = True
        if weight.numel():
            weight = torch_npu.npu_format_cast(weight, 29).contiguous()
        else:
            weight = torch_npu.empty_with_format(
                weight.shape, dtype=weight.dtype, device=weight.device, acl_format=29
            )
    if options["transpose_x2"]:
        weight = weight.T
    scale, offset = data["antiquantScale"], data["antiquantOffset"]
    if scale.shape[-1] != weight.shape[-1] and scale.numel() != 1:
        scale, offset = scale.T, offset.T
    data.update(x1=x, x2=weight, antiquantScale=scale, antiquantOffset=offset)
    return data, options


def int4_tensor(logical, transpose):
    physical = logical.T if transpose else logical
    package = nnopbase.create_int4_acl_tensor(
        physical, fmt=AclFormat.ACL_FORMAT_ND, storage_shape=None
    )
    if transpose:
        nnopbase.aclDestroyTensor(package.tensor)
        dims = ctypes.c_int64 * 2
        package.tensor = nnopbase.aclCreateTensor(
            dims(*logical.shape),
            2,
            AclDataType.ACL_INT4,
            dims(*logical.stride()),
            0,
            AclFormat.ACL_FORMAT_ND,
            dims(*physical.shape),
            2,
            ctypes.c_void_p(package.addr),
        )
    return package


def golden(data, options):
    x, weight = data["x1"], data["x2"]
    # The existing empty-K benchmark performs independent FP32 arithmetic.
    if x.shape[-1] == 0:
        output = x.double() @ weight.double()
        if options["use_bias"]:
            output = output + data["bias"].double()
        if options["use_x3"]:
            output = output + data["x3"].double()
    else:
        weight = weight.float()
        scale = data["antiquantScale"].float()
        offset = data["antiquantOffset"].float()
        group_size = data["antiquantGroupSize"]
        if group_size:
            groups = torch.arange(weight.shape[0]) // group_size
            scale = scale[groups]
            offset = offset[groups]
        if options["use_antiquant_offset"]:
            weight = weight + offset
        weight = weight * scale
        output = x.float() @ weight
        if options["use_bias"]:
            output = output + data["bias"].float()
        if options["use_x3"]:
            output = output + data["x3"].float()
    dist.all_reduce(output)
    return output


def benchmark(data, options):
    x, weight = data["x1"], data["x2"]
    bias = data["bias"] if options["use_bias"] else None
    x3 = data["x3"] if options["use_x3"] else None
    offset = data["antiquantOffset"] if options["use_antiquant_offset"] else None
    if x.shape[-1] == 0:
        # Preserve the established empty-K benchmark, including its FP32 reduction.
        output = x.float().cpu() @ weight.float().cpu()
        if bias is not None:
            output = output + bias.float().cpu()
        if x3 is not None:
            output = output + x3.float().cpu()
        output = output.to(x.device)
        dist.all_reduce(output)
        return output.to(x.dtype)
    if bias is not None and bias.dtype == torch.bfloat16:
        bias = bias.float()
    output = torch_npu.npu_weight_quant_batchmatmul(
        x=x,
        weight=weight,
        bias=bias,
        antiquant_scale=data["antiquantScale"],
        antiquant_offset=offset,
        antiquant_group_size=data["antiquantGroupSize"],
    )
    if x3 is not None:
        output = output + x3
    dist.all_reduce(output)
    return output


@register("execute_aclnnWeightQuantMatmulAllReduce")
class Reference(BaseApi):
    def get_cpp_func_signature_type(self):
        return SIGNATURE

    def __call__(self, input_data, with_output=False):
        data, options = rank_inputs(input_data.kwargs, self.task_result)
        return golden(data, options) if self.name == "cpu" else benchmark(data, options)


@register("pyaclnn_aclnnWeightQuantMatmulAllReduce")
class AclnnExecutor(AclnnBaseApi):
    def get_cpp_func_signature_type(self):
        return SIGNATURE

    def init_by_input_data(self, input_data):
        data, options = rank_inputs(input_data.kwargs, self.task_result)
        self.nz = options["x2_nz"]
        # NZ storage describes the original physical matrix, before its transpose view.
        physical_rows, physical_columns = input_data.kwargs["x2"].shape
        self.weight_storage_shape = (
            ((physical_columns + 31) // 32, (physical_rows + 15) // 16, 16, 32)
            if self.nz
            else None
        )
        data["group"] = dist.group.WORLD._get_backend(
            torch.device("npu")
        ).get_hccl_comm_name(self.task_result.dist_task_info.rank)
        input_data.kwargs = {
            name: data[name]
            for name in (
                "x1",
                "x2",
                "bias",
                "antiquantScale",
                "antiquantOffset",
                "x3",
                "group",
                "reduceOp",
                "commTurn",
                "streamMode",
                "antiquantGroupSize",
            )
        }
        args, outputs = super().init_by_input_data(input_data)
        if options["x2_int4"]:
            args[1] = int4_tensor(data["x2"], options["transpose_x2"])
        null = ctypes.POINTER(AclTensor)()
        if not options["use_bias"]:
            args[2] = null
        if not options["use_antiquant_offset"]:
            args[4] = null
        if not options["use_x3"]:
            args[5] = null
        output = torch.empty(
            (data["x1"].shape[0], data["x2"].shape[1]),
            dtype=data["x1"].dtype,
            device=data["x1"].device,
        )
        outputs[0] = self.torch_tensor_to_acl(output)
        args[-1] = outputs[0]
        return args, outputs

    def get_format(self, input_data, index=None, name=None):
        return (
            AclFormat.ACL_FORMAT_FRACTAL_NZ
            if name == "x2" and self.nz
            else AclFormat.ACL_FORMAT_ND
        )

    def get_storage_shape(self, input_data, index=None, name=None):
        return self.weight_storage_shape if name == "x2" else None

    def after_call(self, output_packages):
        return [self.acl_tensor_to_torch(output) for output in output_packages]

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()
