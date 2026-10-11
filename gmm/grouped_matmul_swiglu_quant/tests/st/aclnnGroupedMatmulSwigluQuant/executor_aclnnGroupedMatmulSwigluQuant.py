#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
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
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat


def GMM_Swiglu_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    perChannelScale: torch.Tensor,
    perTokenScale: torch.Tensor,
    m: int,
):
    """
    执行量化的 GMM(通用矩阵乘法)操作,并使用 SwiGLU 激活函数。

    参数:
        x (torch.Tensor): 输入张量,形状为 (m, k)。
        weight (torch.Tensor): 权重张量,形状为 (k, n)。
        perChannelScale (torch.Tensor): 每个通道的缩放因子,形状为 (n,)。
        perTokenScale (torch.Tensor): 每个 token 的缩放因子,形状为 (m,)。
        m (int): token 的数量(x 的行数)。

    返回:
        quantOutput (torch.Tensor): 量化后的输出张量,形状为 (m, k // 2)。
        quantScaleOutput (torch.Tensor): 量化缩放因子,形状为 (m,)。
    """
    # 使用 int32 精度执行矩阵乘法
    c_temp1 = torch.matmul(x.to(torch.int32), weight.to(torch.int32))
    c_temp1 = c_temp1.to(torch.float32)  # 转换回 float32 以便进行缩放

    # 应用每个通道和每个 token 的缩放
    c_temp2 = torch.mul(c_temp1, perChannelScale)
    c_temp3 = torch.mul(c_temp2, perTokenScale.reshape(m, 1))

    # 将结果分成两部分以应用 SwiGLU 激活函数
    c_temp4, gate = c_temp3.chunk(2, dim=-1)
    c_temp5 = c_temp4 * torch.sigmoid(c_temp4)  # SwiGLU 激活
    c_temp6 = c_temp5 * gate  # 与门控值进行逐元素相乘

    # 对输出进行量化
    max = torch.max(torch.abs(c_temp6), -1).values  # 找到最大绝对值以计算缩放因子
    quantScaleOutput = 127 / max  # 计算量化缩放因子
    quantOutput = torch.round(c_temp6 * quantScaleOutput.reshape(m, 1)).to(
        torch.int8
    )  # 量化为 int8
    quantScaleOutput = 1 / quantScaleOutput  # 反向量化缩放因子以便后续反量化

    return quantOutput, quantScaleOutput


def process_groups(
    x: torch.Tensor,
    weight: torch.Tensor,
    perChannelScale: torch.Tensor,
    perTokenScale: torch.Tensor,
    groupList: torch.Tensor,
):
    """
    按组处理输入数据,并调用 GMM_Swiglu_quant 函数进行量化计算。

    参数:
        x (torch.Tensor): 输入张量,形状为 (M, K)。
        weight (torch.Tensor): 权重张量列表,每个元素的形状为 (E, K, N)。
        perChannelScale (torch.Tensor): 每个通道的缩放因子列表,每个元素的形状为 (E, N)。
        perTokenScale (torch.Tensor): 每个 token 的缩放因子,形状为 (M,)。
        groupList (list): 定义每个组的 token 数量的列表。

    返回:
        quantOutput (torch.Tensor): 量化后的输出张量,形状为 (M, N // 2)。
        quantScaleOutput (torch.Tensor): 量化缩放因子,形状为 (M,)。
    """
    M, N = x.shape[0], weight.shape[2]  # 获取输入张量的形状
    quantOutput = torch.zeros(M, N // 2).to(torch.int8)  # 初始化量化输出张量
    quantScaleOutput = torch.zeros(M).to(torch.float32)  # 初始化量化缩放因子张量

    start_idx = 0  # 起始索引
    preV = 0  # 前一个组的 token 数量
    groupList = groupList.tolist()
    # 遍历 groupList,按组处理数据
    for i, v in enumerate(groupList):
        currV = v
        tempV = currV - preV  # 计算当前组的 token 数量
        preV = currV  # 更新前一个组的 token 数量
        if tempV > 0:
            # 调用 GMM_Swiglu_quant 处理当前组
            (
                quantOutput[start_idx : start_idx + tempV],
                quantScaleOutput[start_idx : start_idx + tempV],
            ) = GMM_Swiglu_quant(
                x[start_idx : start_idx + tempV],
                weight[i],
                perChannelScale[i],
                perTokenScale[start_idx : start_idx + tempV],
                tempV,
            )

        start_idx += tempV  # 更新起始索引以处理下一组
    return quantOutput, quantScaleOutput


def generate_non_decreasing_sequence(length, upper_limit, seed: int):
    """
    生成一个随机非减的一维 Tensor,且最后一个值小于上限。

    参数:
        length (int): 序列的长度。 第二个输入weight 的shape[0]
        upper_limit (int): 最后一个值的上限。第一个输入x 的shape[0]

    返回:
        torch.Tensor: 生成的一维 Tensor。
    """
    # 指定随机种子
    torch.manual_seed(seed)
    # 生成随机递增序列
    random_increments = torch.randint(0, 128, (length,))  # 随机增量,范围 0~9
    sequence = torch.cumsum(random_increments, dim=0)  # 累加生成非减序列

    # 确保最后一个值小于上限
    if sequence[-1] >= upper_limit:
        scale_factor = upper_limit / sequence[-1]  # 计算缩放因子
        sequence = (sequence * scale_factor).to(torch.int64)  # 缩放并转换为整数
    return sequence


@register("function_aclnn_grouped_matmul_swiglu_quant")
class AclnnGroupedMatmulSwigluQuant(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        """
        该接口可实现部门场景下api的初始化需要依赖于当前的输入数据,且不希望计入耗时,
        可以在此接口实现
        :param input_data:
        :return:
        """
        x = input_data.kwargs["x"]
        weight = input_data.kwargs["weight"]
        seed = input_data.kwargs["seed"]
        groupList = generate_non_decreasing_sequence(weight.shape[0], x.shape[0], seed)
        if self.device == "pyaclnn":
            input_data.kwargs["groupList"] = groupList.npu()
        else:
            input_data.kwargs["groupList"] = groupList.cpu()
        input_data.kwargs.pop("seed")
        input_data.kwargs.pop("case")

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"

        x = input_data.kwargs["x"]
        weight = input_data.kwargs["weight"]
        E, N, K = (
            weight.shape[0],
            weight.shape[1] * weight.shape[4],
            weight.shape[2] * weight.shape[3],
        )
        weight = weight.permute(0, 2, 3, 1, 4).reshape(E, K, N).contiguous()
        weightScale = input_data.kwargs["weightScale"]
        xScale = input_data.kwargs["xScale"]
        groupList = input_data.kwargs["groupList"]

        output0, output1 = process_groups(x, weight, weightScale, xScale, groupList)

        real0, real1 = output0[: groupList[-1], :], output1[: groupList[-1]]
        # 将标杆的脏数据部分改成全0
        group_index = groupList[-1]
        output0[group_index:, :] = 0
        output1[group_index:] = 0
        output2 = torch.zeros(1)

        return output0, output1

    def get_format(self, input_data: InputDataset, index=None, name=None):
        """
        :param input_data: 参数列表
        :param index: 参数位置
        :param name: 参数名字
        :return:
        format at this index or name
        """
        if name == "weight":
            if input_data.kwargs["format"] == "FRACTAL_NZ":
                return AclFormat.ACL_FORMAT_FRACTAL_NZ

        return AclFormat.ACL_FORMAT_ND


@register("function_pyaclnn_grouped_matmul_swiglu_quant")
class AclnnGroupedMatmulSwigluQuant(AclnnBaseApi):  # SampleApi类型仅需设置唯一即可。
    def __init__(self, task_result: TaskResult, backend):
        super().__init__(task_result, backend)
        self.input_args = None
        self.groupList = None

    # 默认调用,可省略
    def init_by_input_data(self, input_data: InputDataset):
        self.groupList = input_data.kwargs["groupList"]
        self.input_args, output_packages = super().init_by_input_data(input_data)
        self.input_args[2] = TensorPtr()
        self.input_args[3] = TensorPtr()
        self.input_args.append(TensorPtr())

        return self.input_args, output_packages

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        groupindex = self.groupList[-1].item()
        for idx, output_tmp in enumerate(output):
            padded_tensor = torch.zeros_like(output_tmp)
            if idx == 0:
                padded_tensor[:groupindex, :] = output_tmp[:groupindex, :]
            elif idx == 1:
                padded_tensor[:groupindex] = output_tmp[:groupindex]
            output[idx] = padded_tensor  # 回填！

        return output
