#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import ctypes
import logging
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
from atk.tasks.backends.lib_interface.acl_wrapper import (
    AclFormat,
    AclTensorList,
    AclTensorStruct,
    TORCH_TO_ACLTYPE,
    ascendcl,
    Int64,
    nnopbase,
)


def create_acl_tensor(
    tensor: torch.Tensor, fmt: AclFormat = AclFormat.ACL_FORMAT_ND
) -> AclTensorStruct:
    """使用 aclCreateTensor 创建 tensor 对象。"""
    # 数据类型大小 data数据的大小
    if tensor.device.type != "npu":
        logging.warning("The tensor is not on NPU!")
        tensor = tensor.npu()

    dtype = TORCH_TO_ACLTYPE[str(tensor.dtype)]
    dtype_size = ascendcl.aclDataTypeSize(dtype)
    data_size = 1
    for dim in tensor.shape:
        data_size *= dim
    data_size *= dtype_size

    # 通过torch的api，不再需要调用aclrtMalloc申请device侧内存
    device_addr = ctypes.c_void_p(tensor.data_ptr())

    # 计算连续tensor的strides
    strides = (Int64 * len(tensor.shape))(*[1] * len(tensor.shape))
    for i in range(len(tensor.shape) - 2, -1, -1):
        strides[i] = tensor.shape[i + 1] * strides[i + 1]
    strides[0], strides[1] = strides[1], strides[0]
    view_tensor = tensor.transpose(0, 1)
    acl_tensor = nnopbase.aclCreateTensor(
        (Int64 * len(view_tensor.shape))(*view_tensor.shape),
        len(view_tensor.shape),
        dtype,
        strides,
        0,
        fmt,
        (Int64 * len(tensor.shape))(*tensor.shape),
        len(tensor.shape),
        device_addr,
    )

    return AclTensorStruct(acl_tensor, device_addr, tensor.shape, dtype)


@register("function_aclnn_grouped_matmul_V4_single_mulit")
class AclnnGroupedMatmulV4SingleMulit(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        """
        该接口可实现部门场景下api的初始化需要依赖于当前的输入数据，且不希望计入耗时，
        可以在此接口实现
        :param input_data:
        :return:
        """

        self.x = input_data.kwargs["x"].copy()
        self.KTling = input_data.kwargs["KTiling"]
        input_data.kwargs.pop("KTiling")
        x = input_data.kwargs["x"][0]
        weight = input_data.kwargs["weight"][0]
        # 单单单场景
        if x.shape[0] == weight.shape[0]:
            self.singleY = True
            if self.device == "pyaclnn":
                input_data.kwargs["groupListOptional"] = torch.tensor(
                    self.KTling, dtype=torch.int64
                ).npu()
            else:
                input_data.kwargs["groupListOptional"] = torch.tensor(
                    self.KTling, dtype=torch.int64
                ).cpu()
        else:
            self.singleY = False
        if input_data.kwargs["outType"] == "fp16":
            self.outType = torch.float16
        elif input_data.kwargs["outType"] == "bf16":
            self.outType = torch.bfloat16
        elif input_data.kwargs["outType"] == "fp32":
            self.outType = torch.float32
        input_data.kwargs.pop("outType")

    def cpu(self, input_data: InputDataset, with_output: bool = True):
        x = self.x
        weight = input_data.kwargs["weight"]
        biasOptional = input_data.kwargs["biasOptional"]
        Tiling = self.KTling
        if self.outType == torch.float32:
            self.mmType = torch.float32
        elif self.outType == torch.float16:
            self.mmType = torch.float32
        elif self.outType == torch.bfloat16:
            self.mmType = torch.float32

        # 升精度
        x_type = x[0].dtype
        x[0] = x[0].to(self.mmType)
        if self.outType == torch.float32:
            x[0] = x[0].to(torch.float16).to(torch.float32)
        # for i in range(len(Tiling)):

        output = []
        last = 0
        cur = 0
        if self.singleY != False:
            weight[0] = weight[0].to(self.mmType)
            for i in range(len(Tiling)):
                cur += Tiling[i]
                res = (
                    torch.matmul(
                        x[0][last:cur, :].transpose(0, 1), weight[0][last:cur, :]
                    )
                ).to(x_type)
                output.append(res)
                last = cur
            output = torch.stack(output, dim=0)
            return output
        else:
            for i in range(len(Tiling)):
                weight[i] = weight[i].to(self.mmType)
                cur += Tiling[i]
                res = (torch.matmul(x[0][last:cur, :].transpose(0, 1), weight[i])).to(
                    x_type
                )
                last = cur
                output.append(res)

        return (*output,)

    def cpu_benchmark(self, input_data: InputDataset, with_output: bool = True):
        x = self.x
        weight = input_data.kwargs["weight"]
        biasOptional = input_data.kwargs["biasOptional"]
        Tiling = self.KTling

        # 升精度
        x[0] = x[0].to(torch.float64)

        output = []
        last = 0
        cur = 0
        if self.singleY != False:
            weight[0] = weight[0].to(torch.float64)
            for i in range(len(Tiling)):
                cur += Tiling[i]
                res = (
                    torch.matmul(
                        x[0][last:cur, :].transpose(0, 1).to(torch.float64),
                        weight[0][last:cur, :].to(torch.float64),
                    )
                ).to(torch.float64)
                output.append(res)
                last = cur
            output = torch.stack(output, dim=0)
            return output
        else:
            for i in range(len(Tiling)):
                weight[i] = weight[i].to(torch.float64)
                cur += Tiling[i]
                res = (
                    torch.matmul(
                        x[0][last:cur, :].transpose(0, 1).to(torch.float64),
                        weight[i].to(torch.float64),
                    )
                ).to(torch.float64)
                last = cur
                output.append(res)

        return (*output,)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"][0]
        if self.outType == x.dtype:
            return self.cpu(input_data, with_output)
        else:
            return self.cpu_benchmark(input_data, with_output)

    # def get_cpp_func_signature_type(self):
    #     return "aclnnStatus aclnnGroupedMatmulV4GetWorkspaceSize(const aclTensorList *x, const aclTensorList *weight, \
    #     const aclTensorList *biasOptional, const aclTensorList *scaleOptional, const aclTensorList *offsetOptional, \
    #     const aclTensorList *antiquantScaleOptional, const aclTensorList *antiquantOffsetOptional, const aclTensorList *perTokenScaleOptional, \
    #     const aclTensor *groupListOptional, const aclTensorList *activationInputOptional, const aclTensorList *activationQuantScaleOptional,  \
    #     const aclTensorList *activationQuantOffsetOptional, int64_t splitItem, int64_t groupType, int64_t groupListType, int64_t actType, \
    #     aclTensorList *out, aclTensorList *activationFeatureOutOptional, aclTensorList *dynQuantScaleOutOptional, uint64_t *workspaceSize, \
    #     aclOpExecutor **executor)"


@register("function_pyaclnn_grouped_matmul_V4_single_mulit")
class PyaclnnGroupedMatmulV4SingleMulti(
    AclnnBaseApi
):  # SampleApi类型仅需设置唯一即可。
    def __init__(self, task_result: TaskResult, backend):
        super().__init__(task_result, backend)
        self.input_args = None

    def init_by_input_data(self, input_data: InputDataset):
        self.task_result.output_info_list = [self.task_result.output_info_list]
        input_args, output_packages = super().init_by_input_data(input_data)
        x = input_data.kwargs["x"][0]
        weight = input_data.kwargs["weight"][0]
        # 单单单场景
        # if x.shape[0] == weight.shape[0]:
        #     self.singleY = True
        #     input_data.kwargs['groupListOptional'] = torch.tensor(self.KTling, dtype=torch.int64)
        x_py_tensor = input_data.kwargs["x"][0]
        x_acl_tensor = create_acl_tensor(x_py_tensor)
        input_args[0] = nnopbase.create_x_list([x_acl_tensor])

        input_args[2] = ctypes.POINTER(AclTensorList)()
        input_args[3] = ctypes.POINTER(AclTensorList)()
        input_args[4] = ctypes.POINTER(AclTensorList)()
        input_args[5] = ctypes.POINTER(AclTensorList)()
        input_args[6] = ctypes.POINTER(AclTensorList)()
        input_args[7] = ctypes.POINTER(AclTensorList)()
        if x.shape[0] != weight.shape[0]:
            input_args[8] = TensorPtr()
        input_args[9] = ctypes.POINTER(AclTensorList)()
        input_args[10] = ctypes.POINTER(AclTensorList)()
        input_args[11] = ctypes.POINTER(AclTensorList)()

        input_args.append(ctypes.POINTER(AclTensorList)())
        input_args.append(ctypes.POINTER(AclTensorList)())

        return input_args, output_packages
