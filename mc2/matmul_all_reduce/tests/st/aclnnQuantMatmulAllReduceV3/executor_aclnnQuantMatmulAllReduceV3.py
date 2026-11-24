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

import numpy as np
import torch
import torch.distributed as dist
import torch_npu

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor, AclFormat
from matmul_all_reduce_test_options import OPTION_NAMES, OPTIONS


SIGNATURE = (
    "aclnnStatus aclnnQuantMatmulAllReduceV3GetWorkspaceSize("
    "const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional, const aclTensor* x3Optional, "
    "const aclTensor* dequantScale, const aclTensor* pertokenScaleOptional, "
    "const aclTensor* commQuantScale1Optional, const aclTensor* commQuantScale2Optional, "
    "const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode, "
    "const aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)"
)


def rank_inputs(kwargs, task_result):
    data = dict(kwargs)
    options = dict(zip(OPTION_NAMES, OPTIONS[data.pop("test_options")]))
    x, weight = data["x1"], data["x2"]
    # Preserve the original eight-shard description, not eight launched ranks.
    rows = x.shape[0] // 8
    if options["input_distribution"] == "rank_shards" and rows:
        start = task_result.dist_task_info.rank * rows
        x = x[start : start + rows]
    if options["x2_nz"] and weight.device.type != "cpu":
        torch_npu.npu.config.allow_internal_format = True
        weight = torch_npu.npu_format_cast(weight, 29).contiguous()
    if options["transpose_x2"]:
        weight = weight.T
    data.update(x1=x, x2=weight)
    return data, options


def prepare_comm_scale(value, stage, output_dtype, task_result):
    # Both stages use the preceding reference result, then share one stored input.
    scale = value.reshape(-1, value.shape[-1]).abs().amax(dim=0).div(127).float()
    if torch.all(scale == 0):
        scale = torch.ones_like(scale)
    dist.all_reduce(scale, op=dist.ReduceOp.MAX)
    if task_result.dist_task_info.rank == 0:
        folder = Path("comm_path") / str(task_result.case_config.id)
        folder.mkdir(parents=True, exist_ok=True)
        scale.numpy().tofile(folder / f"fp32_comm_quant_scale_{stage}_0.bin")
    return scale.to(output_dtype)


def load_comm_scales(data, task_result, output_dtype):
    folder = Path("comm_path") / str(task_result.case_config.id)
    for stage in (1, 2):
        name = f"commQuantScale{stage}Optional"
        values = np.fromfile(
            folder / f"fp32_comm_quant_scale_{stage}_0.bin", dtype=np.float32
        )
        data[name] = torch.from_numpy(values.reshape(data[name].shape)).to(
            device=data["x1"].device, dtype=output_dtype
        )


def golden_communication(output, output_dtype, task_result):
    size = task_result.dist_task_info.world_size
    rows, columns = output.shape
    scale1 = prepare_comm_scale(output, 1, output_dtype, task_result)
    # Benchmark first divides half-precision MM output by the stored half scale.
    quant1 = output.float() / scale1.float()
    padding = torch.zeros(((-rows) % size, columns), dtype=quant1.dtype)
    quant1 = torch.cat((quant1, padding)).reshape(size, -1, columns)
    quant1 = quant1.round().clamp(-128, 127).to(torch.int8)
    exchanged = torch.empty_like(quant1)
    dist.all_to_all_single(exchanged, quant1)
    # Benchmark's independent multiplication and division both run in FP32.
    reduced = exchanged.sum(dim=0, keepdim=True).double() * scale1.double()
    scale2 = prepare_comm_scale(reduced, 2, output_dtype, task_result)
    quant2 = (reduced / scale2.double()).round().clamp(-128, 127).to(torch.int8)
    gathered = [torch.empty_like(quant2) for _ in range(size)]
    dist.all_gather(gathered, quant2)
    # The last benchmark step multiplies INT8 by a half-precision external scale.
    output = torch.cat(gathered).float() * scale2.float()
    return output.reshape(-1, columns)[:rows]


def benchmark_communication(output, data, task_result):
    size = task_result.dist_task_info.world_size
    rows, columns = output.shape
    scale1, scale2 = data["commQuantScale1Optional"], data["commQuantScale2Optional"]
    quant1 = output / scale1
    padding = torch.zeros(
        ((-rows) % size, columns), dtype=torch.float32, device=output.device
    )
    quant1 = torch.cat((quant1, padding)).reshape(size, -1, columns)
    quant1 = quant1.round().clamp(-128, 127).to(torch.int8)
    exchanged = torch.empty_like(quant1)
    dist.all_to_all_single(exchanged, quant1)
    reduced = exchanged.sum(dim=0, keepdim=True).float() * scale1
    quant2 = (reduced / scale2).round().clamp(-128, 127).to(torch.int8)
    gathered = torch.empty_like(quant1)
    dist._all_gather_base(gathered, quant2)
    output = gathered * scale2
    return output.reshape(-1, columns)[:rows]


def golden(data, options, task_result):
    output = data["x1"].int() @ data["x2"].int()
    if options["use_bias"]:
        output = output + data["biasOptional"]
    output = output.float()
    if options["use_x1_scale"]:
        output = output * data["pertokenScaleOptional"].float().reshape(-1, 1)
    output = output * data["dequantScale"].float()
    if options["use_x3"]:
        output = output + data["x3Optional"].float()
    if options["use_comm_quant"]:
        output_dtype = (
            torch.bfloat16
            if data["dequantScale"].dtype == torch.bfloat16
            else torch.float16
        )
        return golden_communication(output, output_dtype, task_result)
    dist.all_reduce(output)
    return output


def benchmark(data, options, task_result):
    output_dtype = (
        torch.bfloat16
        if data["dequantScale"].dtype == torch.bfloat16
        else torch.float16
    )
    scale = data["dequantScale"].flatten()
    if not options["use_x1_scale"] and scale.dtype == torch.float32:
        scale = torch_npu.npu_trans_quant_param(scale)
    output = torch_npu.npu_quant_matmul(
        x1=data["x1"],
        x2=data["x2"],
        scale=scale,
        bias=data["biasOptional"] if options["use_bias"] else None,
        pertoken_scale=data["pertokenScaleOptional"]
        if options["use_x1_scale"]
        else None,
        output_dtype=output_dtype,
    )
    if options["use_x3"]:
        output = output + data["x3Optional"]
    if options["use_comm_quant"]:
        load_comm_scales(data, task_result, output_dtype)
        return benchmark_communication(output, data, task_result)
    dist.all_reduce(output)
    return output


@register("execute_aclnnQuantMatmulAllReduceV3")
class Reference(BaseApi):
    def get_cpp_func_signature_type(self):
        return SIGNATURE

    def __call__(self, input_data, with_output=False):
        data, options = rank_inputs(input_data.kwargs, self.task_result)
        if self.name == "cpu":
            return golden(data, options, self.task_result)
        return benchmark(data, options, self.task_result)


@register("pyaclnn_aclnnQuantMatmulAllReduceV3")
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
        output_dtype = (
            torch.bfloat16
            if data["dequantScale"].dtype == torch.bfloat16
            else torch.float16
        )
        if options["use_comm_quant"]:
            load_comm_scales(data, self.task_result, output_dtype)
        if not options["use_x1_scale"] and data["dequantScale"].dtype == torch.float32:
            data["dequantScale"] = torch_npu.npu_trans_quant_param(data["dequantScale"])
        data["group"] = dist.group.WORLD._get_backend(
            torch.device("npu")
        ).get_hccl_comm_name(self.task_result.dist_task_info.rank)
        input_data.kwargs = {
            name: data[name]
            for name in (
                "x1",
                "x2",
                "biasOptional",
                "x3Optional",
                "dequantScale",
                "pertokenScaleOptional",
                "commQuantScale1Optional",
                "commQuantScale2Optional",
                "group",
                "reduceOp",
                "commTurn",
                "streamMode",
            )
        }
        args, outputs = super().init_by_input_data(input_data)
        null = ctypes.POINTER(AclTensor)()
        if not options["use_bias"]:
            args[2] = null
        if not options["use_x3"]:
            args[3] = null
        if not options["use_x1_scale"]:
            args[5] = null
        if not options["use_comm_quant"]:
            args[6:8] = [null, null]
        output = torch.empty(
            (data["x1"].shape[0], data["x2"].shape[1]),
            dtype=output_dtype,
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
