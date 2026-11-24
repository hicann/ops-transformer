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
import os

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
from all_gather_matmul_v2_test_options import OPTION_NAMES, OPTIONS


torch_npu.npu.config.allow_internal_format = True

SIGNATURE = """aclnnStatus aclnnAllGatherMatmulV2GetWorkspaceSize(
     const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x1Scale,
    const aclTensor* x2Scale, const aclTensor* quantScale, int64_t blockSize, const char* group,
    int64_t gatherIndex, int64_t commTurn, int64_t streamMode, int64_t groupSize, const char* commMode,
    aclTensor* output, aclTensor* gatherOut, aclTensor* amaxOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);"""

INPUT_NAMES = (
    "x1",
    "x2",
    "bias",
    "x1Scale",
    "x2Scale",
    "quantScale",
    "blockSize",
    "group",
    "gatherIndex",
    "commTurn",
    "streamMode",
    "groupSize",
    "commMode",
)


def rank_inputs(data, task_result, options):
    data = dict(data)
    info = task_result.dist_task_info
    x = data["x1"]
    if options["input_distribution"] == "rank_shards":
        data["x1"] = x.chunk(info.world_size, dim=0)[info.rank].contiguous()
        if x.dtype == torch.int8 and options["use_x1_scale"]:
            data["x1Scale"] = (
                data["x1Scale"].chunk(info.world_size, dim=0)[info.rank].contiguous()
            )
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            20261119 + int(task_result.case_config.id) * 1000003 + int(info.rank) * 1009
        )
        indices = torch.randperm(x.shape[-1], generator=generator, device="cpu")
        data["x1"] = x.index_select(-1, indices.to(x.device))
    if options["transpose_x2"]:
        data["x2"] = data["x2"].T
    return data


def int4_tensor(logical, transpose=False):
    # Pack physical storage, then restore the logical transpose descriptor.
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


def gather(tensor, world_size):
    pieces = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(pieces, tensor)
    return torch.cat(pieces)


def effective_packed_scale(scale):
    # Match TransQuantParamV2 round_mode=1: the encoded input stores an RNE FP19 value.
    bits = scale.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exponent = (bits >> 23) & 0xFF
    rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & 0xFFFFE000
    rounded = torch.where(exponent == 0, bits & 0x80000000, rounded)
    rounded = torch.where(exponent == 0xFF, bits & 0xFFFFE000, rounded)
    return rounded.to(torch.int32).view(torch.float32)


def golden(data, options, info):
    gathered = gather(data["x1"], info.world_size)
    if gathered.dtype != torch.int8:
        return gathered.float() @ data["x2"].float(), gathered
    scale = data["x2Scale"].float()
    if options["packed_scale"]:
        scale = effective_packed_scale(scale)
    output = (gathered.int() @ data["x2"].int()).float() * scale
    if options["use_x1_scale"]:
        scale = gather(data["x1Scale"], info.world_size)
        if options["packed_scale"]:
            # The benchmark applies per-token scale as an independent FP32 multiply.
            output = output.double() * scale.double()
        else:
            output = output * scale.float()
    return output, gathered


def benchmark(data, options, info):
    gathered = gather(data["x1"], info.world_size)
    if gathered.dtype != torch.int8:
        return gathered @ data["x2"], gathered
    output_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[
        options["output_dtype"]
    ]
    scale = data["x2Scale"].flatten()
    pertoken = (
        gather(data["x1Scale"], info.world_size).flatten()
        if options["use_x1_scale"]
        else None
    )
    if options["packed_scale"]:
        scale = torch_npu.npu_trans_quant_param(scale, round_mode=1)
        output = torch_npu.npu_quant_matmul(
            gathered, data["x2"], scale=scale, output_dtype=output_dtype
        )
        if pertoken is not None:
            output = (output.float() * pertoken[:, None]).to(output_dtype)
    else:
        output = torch_npu.npu_quant_matmul(
            gathered,
            data["x2"],
            scale=scale,
            pertoken_scale=pertoken,
            output_dtype=output_dtype,
        )
    return output, gathered


@register("all_gather_matmul_v2_reference")
class Reference(BaseApi):
    @staticmethod
    def get_cpp_func_signature_type():
        return SIGNATURE

    def __call__(self, input_data, with_output=False):
        options = dict(zip(OPTION_NAMES, OPTIONS[input_data.kwargs["test_options"]]))
        data = rank_inputs(input_data.kwargs, self.task_result, options)
        reference = golden if self.name == "cpu" else benchmark
        return reference(data, options, self.task_result.dist_task_info)


@register("all_gather_matmul_v2_aclnn")
class AclnnExecutor(AclnnBaseApi):
    @staticmethod
    def get_cpp_func_signature_type():
        return SIGNATURE

    def init_by_input_data(self, input_data):
        data = dict(input_data.kwargs)
        options = dict(zip(OPTION_NAMES, OPTIONS[data.pop("test_options")]))
        data = rank_inputs(data, self.task_result, options)
        info = self.task_result.dist_task_info
        self.nz = options["x2_nz"]
        self.x2_storage_shape = None
        self.int4 = options["x1_int4"]
        if self.nz:
            # Cast physical storage before applying the requested logical transpose.
            physical = data["x2"].T if options["transpose_x2"] else data["x2"]
            # torch_npu FormatHelper::InferShapeNDToNZ uses the physical base shape.
            rows, columns = physical.shape
            c0 = 32 // physical.element_size()
            self.x2_storage_shape = [
                (columns + c0 - 1) // c0,
                (rows + 15) // 16,
                16,
                c0,
            ]
            physical = torch_npu.npu_format_cast(physical, 29)
            data["x2"] = physical.T if options["transpose_x2"] else physical
        if options["packed_scale"]:
            data["x2Scale"] = torch_npu.npu_trans_quant_param(
                data["x2Scale"], round_mode=1
            )
        data["group"] = dist.group.WORLD._get_backend(
            torch.device("npu")
        ).get_hccl_comm_name(info.rank)
        input_data.kwargs = {name: data[name] for name in INPUT_NAMES}
        args, outputs = super().init_by_input_data(input_data)
        null = ctypes.POINTER(AclTensor)()
        args[2] = null  # A2/A3 bias is fixed nullptr for this interface.
        args[5] = null
        x, weight = data["x1"], data["x2"]
        if x.dtype != torch.int8:
            args[3:5] = [null, null]
        elif not options["use_x1_scale"]:
            args[3] = null
        output_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
            options["output_dtype"], x.dtype
        )
        output = self.torch_tensor_to_acl(
            torch.empty(
                (x.shape[0] * info.world_size, weight.shape[1]),
                dtype=output_dtype,
                device=x.device,
            )
        )
        gathered_tensor = torch.empty(
            (x.shape[0] * info.world_size, x.shape[1]), dtype=x.dtype, device=x.device
        )
        if self.int4:
            args[0] = int4_tensor(x)
            gathered = int4_tensor(gathered_tensor.zero_())
        else:
            gathered = self.torch_tensor_to_acl(gathered_tensor)
        if options["x2_int4"]:
            if os.environ["MC2_PLATFORM"] == "A2":
                # A2's AIV interface infers transpose from the physical [N, K] shape.
                physical = weight.T if options["transpose_x2"] else weight
                args[1] = int4_tensor(physical)
            else:
                args[1] = int4_tensor(weight, options["transpose_x2"])
        args[-2:] = [output, gathered]
        outputs[:] = [output, gathered]
        args.append(null)  # amaxOut is reserved on A2/A3.
        return args, outputs

    def get_format(self, input_data, index=None, name=None):
        return (
            AclFormat.ACL_FORMAT_FRACTAL_NZ
            if name == "x2" and self.nz
            else AclFormat.ACL_FORMAT_ND
        )

    def get_storage_shape(self, input_data, index=None, name=None):
        if name == "x2" and self.nz:
            return self.x2_storage_shape
        return super().get_storage_shape(input_data, index=index, name=name)

    def after_call(self, output_packages):
        output = self.acl_tensor_to_torch(output_packages[0])
        convert = (
            self.int4_acl_tensor_to_torch if self.int4 else self.acl_tensor_to_torch
        )
        return output, convert(output_packages[1])

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()
