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
import numpy as np
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat, AclTensorList


@register("ascend_function_grouped_matmul")
class MethodTorchGroupedMatmulApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchGroupedMatmulApi, self).__init__(task_result)

    # open it when debug, while skip interface check when running
    # def get_cpp_func_signature_type(self):
    #    return "aclnnStatus aclnnGroupedMatmulV4GetWorkspaceSize(const aclTensorList *x, const aclTensorList *weight, const aclTensorList *biasOptional, \
    #         const aclTensorList *scaleOptional, const aclTensorList *offsetOptional, \
    #         const aclTensorList *antiquantScaleOptional, const aclTensorList *antiquantOffsetOptional, \
    #         const aclTensorList *perTokenScaleOptional, const aclTensor *groupListOptional, \
    #         const aclTensorList *activationInputOptional, const aclTensorList *activationQuantScaleOptional, \
    #         const aclTensorList *activationQuantOffsetOptional,  int64_t splitItem, int64_t groupType, \
    #         int64_t groupListType, int64_t actType, aclTensorList *out, aclTensorList *activationFeatureOutOptional, \
    #         aclTensorList *dynQuantScaleOutOptional, uint64_t *workspaceSize, \
    #         aclOpExecutor **executor)"
    def cpu(self, input_data: InputDataset, with_output: bool = True):
        if self.outType == torch.float32:
            self.mmType = torch.float32
        elif self.outType == torch.float16:
            self.mmType = torch.float16
        elif self.outType == torch.bfloat16:
            self.mmType = torch.float16

        x = input_data.kwargs["x"][0]
        weight = input_data.kwargs["weight"][0]
        groupList = input_data.kwargs["groupList"]
        n = weight.shape[2]
        input_dtype = x.dtype
        result = torch.empty(0, n, dtype=self.outType)
        last = 0
        index = 0
        for i in groupList.tolist():
            x_tensor = x[last:i, :].to(self.mmType).numpy()
            weight_tensor = weight[index].to(self.mmType).numpy()
            index += 1
            cur_res = np.matmul(x_tensor, weight_tensor)
            result = torch.cat([result, torch.from_numpy(cur_res)], dim=0)
            last = i
        return [result.to(self.outType)]

    def cpu_benchmark(self, input_data: InputDataset, with_output: bool = True):
        if self.outType == torch.float32:
            self.mmType = torch.float64
        elif self.outType == torch.float16:
            self.mmType = torch.float64
        elif self.outType == torch.bfloat16:
            self.mmType = torch.float64
        x = input_data.kwargs["x"][0]
        weight = input_data.kwargs["weight"][0]
        groupList = input_data.kwargs["groupList"]
        n = weight.shape[2]
        input_dtype = x.dtype
        result = torch.empty(0, n, dtype=self.mmType)
        last = 0
        index = 0
        for i in groupList.tolist():
            x_tensor = x[last:i, :].to(self.mmType).numpy()
            weight_tensor = weight[index].to(self.mmType).numpy()
            index += 1
            cur_res = np.matmul(x_tensor, weight_tensor)
            result = torch.cat([result, torch.from_numpy(cur_res)], dim=0)
            last = i
        return [result.to(self.mmType)]

    def __call__(self, input_data: InputDataset, with_output: bool = True):
        x = input_data.kwargs["x"][0]

        if self.outType == x.dtype:
            return self.cpu(input_data, with_output)
        else:
            return self.cpu_benchmark(input_data, with_output)
        # if self.outType == torch.float16:
        #     if x.dtype == torch.float16:
        #         return self.cpu_benchmark(input_data, with_output)
        #     else:
        #         return self.cpu(input_data, with_output)
        # elif self.outType == torch.bfloat16:
        #     if scale.dtype == torch.float16:
        #         return self.cpu_benchmark(input_data, with_output)
        #     else:
        #         return self.cpu(input_data, with_output)
        # elif self.outType == torch.float32:

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
            )
        if input_data.kwargs["outType"] == "fp16":
            self.outType = torch.float16
        elif input_data.kwargs["outType"] == "bf16":
            self.outType = torch.bfloat16
        elif input_data.kwargs["outType"] == "fp32":
            self.outType = torch.float32
        input_data.kwargs.pop("outType")


@register("aclnn_grouped_matmul")
class AclnnFunctionApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data):
        import ctypes

        # self.task_result.output_info_list = [self.task_result.output_info_list]
        input_args, output_pac = super().init_by_input_data(input_data)
        input_args.insert(2, ctypes.c_void_p(0))  # bias
        input_args.insert(3, ctypes.c_void_p(0))  # scale
        input_args.insert(4, ctypes.c_void_p(0))  # offset
        input_args.insert(5, ctypes.c_void_p(0))  # antiquantscale
        input_args.insert(6, ctypes.c_void_p(0))  # antiquantoffset
        input_args.insert(7, ctypes.c_void_p(0))  # pertokenscale
        # 8 : groupList
        input_args.insert(9, ctypes.c_void_p(0))  # activationInputOptional
        input_args.insert(10, ctypes.c_void_p(0))  # activationQuantScaleOptional
        input_args.insert(11, ctypes.c_void_p(0))  # activationQuantOffsetOptional
        input_args[12] = ctypes.c_long(2)  # split item = 2
        input_args.insert(15, ctypes.c_long(0))  #  actType= 0
        # 16 gmm output
        # 17 18 nullptr output
        input_args.insert(17, ctypes.c_void_p(0))
        input_args.insert(18, ctypes.c_void_p(0))
        return input_args, output_pac

    def after_call(self, output_packages):
        output = super().after_call(output_packages)
        return output
