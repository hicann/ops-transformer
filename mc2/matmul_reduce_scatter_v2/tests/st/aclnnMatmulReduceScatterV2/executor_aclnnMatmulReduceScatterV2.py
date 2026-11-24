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
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor, AclFormat
from matmul_reduce_scatter_v2_test_options import OPTION_NAMES, OPTIONS


torch_npu.npu.config.allow_internal_format = True

SIGNATURE = """aclnnStatus aclnnMatmulReduceScatterV2GetWorkspaceSize(
     const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x1Scale,
    const aclTensor* x2Scale, const aclTensor* quantScale, int64_t blockSize, const char* group,
    const char* reduceOp, int64_t commTurn, int64_t streamMode, int64_t groupSize, const char* commMode,
    aclTensor* output, aclTensor* amaxOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor);"""

INPUT_NAMES = (
    "x1",
    "x2",
    "bias",
    "x1Scale",
    "x2Scale",
    "quantScale",
    "blockSize",
    "group",
    "reduceOp",
    "commTurn",
    "streamMode",
    "groupSize",
    "commMode",
)


def rank_inputs(data, task_result, options):
    data = dict(data)
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
    if options["transpose_x2"]:
        data["x2"] = data["x2"].T
    return data


def effective_packed_scale(scale):
    # Match TransQuantParamV2 round_mode=1: the encoded input stores an RNE FP19 value.
    bits = scale.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exponent = (bits >> 23) & 0xFF
    rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & 0xFFFFE000
    rounded = torch.where(exponent == 0, bits & 0x80000000, rounded)
    rounded = torch.where(exponent == 0xFF, bits & 0xFFFFE000, rounded)
    return rounded.to(torch.int32).view(torch.float32)


def golden(data, options, info):
    x, weight = data["x1"], data["x2"]
    bias = data["bias"] if options["use_bias"] and data["bias"].numel() else None
    if x.dtype == torch.int8:
        scale = data["x2Scale"].float()
        if options["packed_scale"]:
            scale = effective_packed_scale(scale)
        output = (x.int() @ weight.int()).float() * scale
        if options["use_x1_scale"]:
            if options["packed_scale"]:
                # Packed-scale MM is followed by an independent FP32 per-token multiply.
                output = output.double() * data["x1Scale"].double()
            else:
                output = output * data["x1Scale"].float()
        if bias is not None:
            output = output + bias.float()
    else:
        output = x.float() @ weight.float()
        if bias is not None:
            output = output.double() + bias.double()
    dist.all_reduce(output)
    return output.chunk(info.world_size, dim=0)[info.rank].contiguous()


def benchmark(data, options, info):
    x, weight = data["x1"], data["x2"]
    bias = data["bias"] if options["use_bias"] and data["bias"].numel() else None
    if x.dtype == torch.int8:
        output_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[
            options["output_dtype"]
        ]
        if options["packed_scale"]:
            # A2 fixpipe emits FP16 before the optional per-token AIV step.
            scale = torch_npu.npu_trans_quant_param(
                data["x2Scale"].flatten(), round_mode=1
            )
            output = torch_npu.npu_quant_matmul(
                x, weight, scale=scale, bias=bias, output_dtype=output_dtype
            )
            if options["use_x1_scale"]:
                output = (output.float() * data["x1Scale"]).to(output_dtype)
        else:
            pertoken = data["x1Scale"].flatten() if options["use_x1_scale"] else None
            if (
                pertoken is None
                and data["x2Scale"].dtype == torch.float32
                and output_dtype == torch.float16
            ):
                # Keep raw FP32 scale: the no-token small-op route truncates it to FP19.
                pertoken = torch.ones(x.shape[0], dtype=torch.float32, device=x.device)
            output = torch_npu.npu_quant_matmul(
                x,
                weight,
                scale=data["x2Scale"].flatten(),
                pertoken_scale=pertoken,
                bias=bias,
                output_dtype=output_dtype,
            )
    else:
        output = x @ weight
        if bias is not None:
            output = (output.float() + bias.float()).to(output.dtype)
    result = torch.empty(
        (output.shape[0] // info.world_size, output.shape[1]),
        dtype=output.dtype,
        device=output.device,
    )
    dist.reduce_scatter_tensor(result, output.contiguous())
    return result


@register("matmul_reduce_scatter_v2_reference")
class Reference(BaseApi):
    @staticmethod
    def get_cpp_func_signature_type():
        return SIGNATURE

    def __call__(self, input_data, with_output=False):
        options = dict(zip(OPTION_NAMES, OPTIONS[input_data.kwargs["test_options"]]))
        data = rank_inputs(input_data.kwargs, self.task_result, options)
        reference = golden if self.name == "cpu" else benchmark
        return reference(data, options, self.task_result.dist_task_info)


@register("matmul_reduce_scatter_v2_aclnn")
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
        if self.nz:
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
        if not options["use_bias"] or not data["bias"].numel():
            args[2] = null
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
                (x.shape[0] // info.world_size, weight.shape[1]),
                dtype=output_dtype,
                device=x.device,
            )
        )
        args[-1] = output
        outputs[:] = [output]
        args.append(null)  # amaxOutOptional is reserved on A2/A3.
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
        return [self.acl_tensor_to_torch(output) for output in output_packages]

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()
