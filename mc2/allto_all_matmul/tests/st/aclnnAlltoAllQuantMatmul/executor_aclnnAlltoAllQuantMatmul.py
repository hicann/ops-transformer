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
from atk.tasks.backends.lib_interface.acl_wrapper import nnopbase, AclFormat
from allto_all_matmul_test_options import OPTION_NAMES, OPTIONS

SIGNATURE = """aclnnStatus aclnnAlltoAllQuantMatmulGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
    const aclTensor* x1ScaleOptional, const aclTensor* x2Scale,
    const aclTensor* commScaleOptional, const aclTensor* x1OffsetOptional,
    const aclTensor* x2OffsetOptional, const char* group, const aclIntArray* alltoAllAxesOptional,
    int64_t x1QuantMode, int64_t x2QuantMode, int64_t commQuantMode,
    int64_t commQuantDtype, int64_t x1QuantDtype, int64_t groupSize,
    bool transposeX1, bool transposeX2, const aclTensor* output,
    const aclTensor* alltoAllOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor)"""


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


def exchange(x, info, cpu):
    if cpu:
        ranks = [torch.empty_like(x) for _ in range(info.world_size)]
        dist.all_gather(ranks, x)
        received = [value.chunk(info.world_size, dim=0)[info.rank] for value in ranks]
    else:
        sent = [value.contiguous() for value in x.chunk(info.world_size, dim=0)]
        received = [torch.empty_like(sent[0]) for _ in range(info.world_size)]
        dist.all_to_all(received, sent)
    return torch.cat(received, dim=1)


def unpack_int4(packed, shape):
    values = packed.cpu().reshape(-1).to(torch.int64)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    values = ((values[:, None] >> shifts) & 15).reshape(-1)
    values = torch.where(values >= 8, values - 16, values)
    return values[: shape[0] * shape[1]].to(torch.int8).reshape(shape).to(packed.device)


def golden(data, options, info):
    gathered = exchange(data["x1"], info, cpu=True)
    weight = data["x2"].T if data["transposeX2"] else data["x2"]
    if data["x1QuantMode"] == 3:
        rows = gathered.shape[0]
        token_scale = data["x1ScaleOptional"].reshape(-1, 1)[
            info.rank * rows : (info.rank + 1) * rows
        ]
        output = (gathered.int() @ weight.int()).float()
        output = output * data["x2Scale"].float() * token_scale.float()
    else:
        x = gathered.float()
        if options["use_smooth"]:
            x = x * data["x1ScaleOptional"].float()
        maximum = 7 if options["x2_int4"] else 127
        token_scale = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / maximum
        quantized = (x / token_scale).round().clamp(-maximum, maximum).int()
        output = (quantized @ weight.int()).float()
        output = output * data["x2Scale"].float() * token_scale
    if options["use_bias"]:
        output = output + data["biasOptional"].float()
    return (output, gathered) if options["comm_output"] else output


def benchmark(data, options, info):
    gathered = exchange(data["x1"], info, cpu=False)
    weight = data["x2"].T if data["transposeX2"] else data["x2"]
    if data["x1QuantMode"] == 3:
        rows = gathered.shape[0]
        token_scale = data["x1ScaleOptional"][
            info.rank * rows : (info.rank + 1) * rows
        ].clone()
        output_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[
            options["output_dtype"]
        ]
        quantized = gathered
    else:
        smooth = data["x1ScaleOptional"] if options["use_smooth"] else None
        if options["x2_int4"]:
            packed, token_scale = torch_npu.npu_dynamic_quant(
                gathered,
                smooth_scales=smooth,
                dst_type=torch.quint4x2,
            )
            quantized = unpack_int4(packed, gathered.shape)
        else:
            quantized, token_scale = torch_npu.npu_dynamic_quant(
                gathered, smooth_scales=smooth
            )
        output_dtype = gathered.dtype
    bias = data["biasOptional"] if options["use_bias"] else None
    output = torch_npu.npu_quant_matmul(
        quantized,
        weight,
        scale=data["x2Scale"],
        pertoken_scale=token_scale,
        bias=bias,
        output_dtype=output_dtype,
    )
    return (output, gathered) if options["comm_output"] else output


@register("function_aclnn_alltoall_quant_matmul")
class Reference(BaseApi):
    def get_cpp_func_signature_type(self):
        return SIGNATURE

    def __call__(self, input_data, with_output=False):
        data, options = rank_inputs(input_data.kwargs, self.task_result)
        if self.name == "cpu":
            return golden(data, options, self.task_result.dist_task_info)
        return benchmark(data, options, self.task_result.dist_task_info)


@register("pyaclnn_aclnn_alltoall_quant_matmul")
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
        if data["x1QuantMode"] == 7 and not options["use_smooth"]:
            args[3] = ctypes.POINTER(AclTensor)()
        for index in (5, 6, 7):
            args[index] = ctypes.POINTER(AclTensor)()
        for index, name in enumerate(("x1", "x2")):
            if options[name + "_int4"]:
                args[index] = nnopbase.create_int4_acl_tensor(
                    data[name], fmt=AclFormat.ACL_FORMAT_ND, storage_shape=None
                )
        x = data["x1"]
        size = self.task_result.dist_task_info.world_size
        n = data["x2"].shape[0 if data["transposeX2"] else 1]
        output_dtype = x.dtype
        if data["x1QuantMode"] == 3:
            output_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[
                options["output_dtype"]
            ]
        outputs[0] = self.torch_tensor_to_acl(
            torch.empty((x.shape[0] // size, n), dtype=output_dtype, device=x.device)
        )
        args[18] = outputs[0]
        self.gather_int4 = options["x1_int4"] and options["comm_output"]
        if options["comm_output"]:
            gathered = torch.zeros(
                (x.shape[0] // size, x.shape[1] * size), dtype=x.dtype, device=x.device
            )
            if self.gather_int4:
                outputs[1] = nnopbase.create_int4_acl_tensor(
                    gathered, fmt=AclFormat.ACL_FORMAT_ND, storage_shape=None
                )
            else:
                outputs[1] = self.torch_tensor_to_acl(gathered)
            args[19] = outputs[1]
        else:
            args.append(ctypes.POINTER(AclTensor)())
        return args, outputs

    def after_call(self, output_packages):
        outputs = [self.acl_tensor_to_torch(output_packages[0])]
        if self.gather_int4:
            outputs.append(self.int4_acl_tensor_to_torch(output_packages[1]))
        else:
            outputs.extend(
                self.acl_tensor_to_torch(package) for package in output_packages[1:]
            )
        return outputs
