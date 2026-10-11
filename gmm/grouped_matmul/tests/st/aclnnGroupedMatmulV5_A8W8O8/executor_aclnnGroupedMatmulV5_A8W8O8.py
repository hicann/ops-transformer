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
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat


@register("ascend_function_grouped_matmul")
class MethodTorchGroupedMatmulApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchGroupedMatmulApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = True):
        x = input_data.kwargs["x"][0]
        weight = input_data.kwargs["weight"][0]
        scale = input_data.kwargs["scaleOptional"][0]
        groupList = input_data.kwargs["groupList"]
        wformat = input_data.kwargs["wformat"]
        E, N, K = weight.shape[0], weight.shape[2], weight.shape[1]
        bias = torch.tensor([])
        pertokenScale = torch.tensor([])
        hasBias = False
        hasPertokenScale = False
        if not input_data.kwargs["biasOptional"]:
            hasBias = False
        else:
            hasBias = True
            bias = input_data.kwargs["biasOptional"][0]

        if not input_data.kwargs["perTokenScaleOptional"]:
            hasPertokenScale = False
        else:
            hasPertokenScale = True
            pertokenScale = input_data.kwargs["perTokenScaleOptional"][0]

        n = weight.shape[2]
        scale_dtype = scale.dtype
        if scale_dtype == torch.float32:
            output_dtype = torch.float16
        else:
            output_dtype = torch.bfloat16
        # grouped matmul
        result = torch.empty(0, n, dtype=torch.float32)
        last = 0
        index = 0
        for i in groupList.tolist():
            x_tensor = x[last:i, :].to(torch.int32)
            if wformat == "FRACTAL_NZ":
                weight[index] = (
                    weight[index]
                    .reshape(N // 32, 16, K // 16, 32)
                    .permute(1, 2, 0, 3)
                    .reshape(K, N)
                    .contiguous()
                )
            weight_tensor = weight[index].to(torch.int32)
            scale_tensor = scale[index].clone().detach().to(torch.float32)
            matmul_res = torch.matmul(x_tensor, weight_tensor)
            if not hasBias:
                cur_res = matmul_res.to(torch.float32) * scale_tensor.unsqueeze(0)
            else:
                bias_tensor = bias[index].clone().detach()
                cur_res = (matmul_res + bias_tensor.unsqueeze(0)).to(
                    torch.float32
                ) * scale_tensor.unsqueeze(0)
            if hasPertokenScale:
                pertokenScale_tensor = pertokenScale[last:i]
                cur_res = cur_res * pertokenScale_tensor.reshape(-1, 1)
            index += 1
            result = torch.cat([result, cur_res], dim=0)
            last = i
        return [result.to(output_dtype)]

    def init_by_input_data(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"][0]
        groupList = input_data.kwargs["groupList"]
        if self.device == "pyaclnn":
            input_data.kwargs["groupList"] = (
                torch.linspace(
                    x.shape[0] // groupList.shape[0],
                    x.shape[0],
                    steps=int(groupList.size(0)),
                )
                .round()
                .to(torch.int64)
                .npu()
            )
        else:
            input_data.kwargs["groupList"] = (
                torch.linspace(
                    x.shape[0] // groupList.shape[0],
                    x.shape[0],
                    steps=int(groupList.size(0)),
                )
                .round()
                .to(torch.int64)
                .cpu()
            )


@register("aclnn_grouped_matmul")
class AclnnFunctionApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def get_format(self, input_data, index=None, name=None):
        if name == "weight" and self.wformat == "FRACTAL_NZ":
            return AclFormat.ACL_FORMAT_FRACTAL_NZ
        return AclFormat.ACL_FORMAT_ND

    def init_by_input_data(self, input_data):
        print(input_data.kwargs["wformat"])
        if input_data.kwargs["wformat"] == "FRACTAL_NZ":
            self.wformat = "FRACTAL_NZ"
        else:
            self.wformat = "ND"
        if self.device == "pyaclnn":
            input_data.kwargs.pop("wformat")
        import ctypes

        # self.task_result.output_info_list = [self.task_result.output_info_list]
        input_args, output_pac = super().init_by_input_data(input_data)
        # input_args.insert(2, ctypes.c_void_p(0)) # bias
        # input_args.insert(3, ctypes.c_void_p(0)) # scale
        input_args.insert(4, ctypes.c_void_p(0))  # offset
        input_args.insert(5, ctypes.c_void_p(0))  # antiquantscale
        input_args.insert(6, ctypes.c_void_p(0))  # antiquantoffset
        # input_args.insert(7, ctypes.c_void_p(0)) # pertokenScale
        # 8 : groupList
        input_args.insert(9, ctypes.c_void_p(0))  # activationInputOptional
        input_args.insert(10, ctypes.c_void_p(0))  # activationQuantScaleOptional
        input_args.insert(11, ctypes.c_void_p(0))  # activationQuantOffsetOptional
        input_args[12] = ctypes.c_long(2)  # split item = 2
        # input_args.insert(15, ctypes.c_long(0)) # actType
        # input_args.insert(16, ctypes.c_void_p(0)) #tuningConfig
        # nullptr output
        input_args.insert(18, ctypes.c_void_p(0))
        input_args.insert(19, ctypes.c_void_p(0))

        return input_args, output_pac

    def after_call(self, output_packages):
        output = super().after_call(output_packages)
        return output
