#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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


@register("ascend_function_grouped_matmul_add")
class MethodTorchGroupedMatmulAddApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchGroupedMatmulAddApi, self).__init__(task_result)

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnGroupedMatmulAddGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *groupList, aclTensor *yRef, bool transposeX, bool transposeWeight, int64_t groupType, uint64_t *workspaceSize, aclOpExecutor **executor)"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        weight = input_data.kwargs["weight"]
        groupList = input_data.kwargs["groupList"]
        y = input_data.kwargs["y"]
        y_dtype = y.dtype
        # 分组矩阵乘法
        result = []
        last = 0
        for i in groupList.tolist():
            x_tensor = x[last:i, :].to(torch.float64)
            weight_tensor = weight[last:i, :].to(torch.float64)
            result.append(torch.matmul(x_tensor.t(), weight_tensor))
            last = i

        result = torch.stack(result).reshape(y.shape) + y.cpu()
        return result.to(y_dtype)

    def init_by_input_data(self, input_data: InputDataset, with_output: bool = False):
        # print("before----------------------------------------------------")
        # print(input_data.kwargs)
        # print("before----------------------------------------------------")
        x = input_data.kwargs["x"]
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
        # print("----------------------------------------------------")
        # print(input_data.kwargs)
        # print("----------------------------------------------------")


@register("aclnn_grouped_matmul_add")
class AclnnFunctionApi(AclnnBaseApi):
    # def __call__(self):
    #     super().__call__()

    def init_by_input_data(self, input_data):
        input_args, output_packages = super().init_by_input_data(input_data)
        # 把最后一个入参去除（默认处理会把标杆的输出拼到入参后，但实际上算子并无此参数）
        for i in range(len(output_packages)):
            input_args.pop()
        # 将前1个输入参数标记为输出
        output_packages[:] = [input_args[3]]
        return input_args, output_packages

    def after_call(self, output_packages):
        output = super().after_call(output_packages)
        # logging.info(f"after_call output :{output}")
        return output
