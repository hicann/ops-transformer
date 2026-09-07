#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import os
import torch
import torch_npu
import ctypes
import logging
import numpy as np
import random
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat, AclTensorList

os.environ["PYTORCH_NO_NPU_MEMORY_CACHING"] = "1"


def Custom_MM(x: torch.Tensor, weight: torch.Tensor, weightScale: torch.Tensor, m: int):
    """
    执行量化的 GMM（通用矩阵乘法）操作，并使用 SwiGLU 激活函数。

    参数:
        x (torch.Tensor): 输入张量，形状为 (m, k)。
        weight (torch.Tensor): 权重张量，形状为 (k, n)。
        weightScale (torch.Tensor): 每个通道的缩放因子，
         - perGroup 场景: 形状为 (k_group_num, n)。注：当k_group_num==1时为perChannel场景
         - perChannel 场景: 形状为 (n)。
        perTokenScale (torch.Tensor): 每个 token 的缩放因子，形状为 (m,)。
        m (int): token 的数量（x 的行数）。

    返回:
        MMOut(fp16): MatMul + perGroup或perChannel反量化结果
    """
    # 使用 int32 精度执行矩阵乘法
    k, n = weight.shape
    MMOut = torch.zeros((m, n), dtype=torch.float16)
    # perGroup 场景
    if len(weightScale.shape) == 2 and weightScale.shape[0] != 1:
        K_group = weightScale.shape[0]
        per_group_ele = k // K_group
        x_grouped = x.view(-1, K_group, per_group_ele).transpose(0, 1)
        weight_grouped = weight.view(K_group, per_group_ele, n)
        c_temp = torch.bmm(
            x_grouped.to(torch.int32), weight_grouped.to(torch.int32)
        ).to(torch.float16)
        for k_idx in range(K_group):
            MMOut += (
                c_temp[k_idx] * weightScale[k_idx].view(1, -1).to(torch.float16)
            ).to(torch.float16)
    # perChannel 场景
    elif len(weightScale.shape) == 1 or (
        len(weightScale.shape) == 2 and weightScale.shape[0] == 1
    ):
        c_temp = torch.matmul(x.to(torch.int32), weight.to(torch.int32)).to(
            torch.float32
        )
        MMOut = (c_temp * weightScale.view(1, -1)).to(torch.float16)
    return MMOut.to(torch.float32)


def x_INT8_to_x_INT4(x: torch.Tensor):
    M, K = x.shape
    x_High_4bit = torch.floor(x.to(torch.float16) // 16).to(torch.int8)
    x_Low_4bit = torch.bitwise_and(x.view(torch.int16), 0x0F0F).view(torch.int8) - 8
    x_Int4 = torch.empty((2 * M, K), dtype=torch.int8)
    x_Int4[::2, :] = x_High_4bit
    x_Int4[1::2, :] = x_Low_4bit
    return x_Int4


def process_groups(
    x: torch.Tensor,
    weight: torch.Tensor,
    weightScale: torch.Tensor,
    perTokenScale: torch.Tensor,
    weightAssistanceMatrix: torch.Tensor,
    groupList: torch.Tensor,
):
    """
    按组处理输入数据，并调用 GMM_Swiglu_quant 函数进行量化计算。

    参数:
        x (torch.Tensor): 输入张量，形状为 (M, K) INT8。
        weight (torch.Tensor): 权重张量列表，每个元素的形状为 (E, K, N)数据类型 INT8 但数据范围为INT4, 实际代表INT4。
        weightScale (torch.Tensor): 每个通道的缩放因子，
         - perGroup 场景: 形状为 (E, k_group_num, N)。
         - perChannel 场景: 形状为 (E, N)。
        perTokenScale (torch.Tensor): 每个 token 的缩放因子，形状为 (M,)。
        groupList (list): 定义每个组的 token 数量的列表。

    返回:
        quantOutput (torch.Tensor): 量化后的输出张量，形状为 (M, N // 2)。
        quantScaleOutput (torch.Tensor): 量化缩放因子，形状为 (M,)。
    """
    M, N = x.shape[0], weight.shape[2]  # 获取输入张量的形状
    quantOutput = torch.zeros(M, N // 2).to(torch.int8)  # 初始化量化输出张量
    quantScaleOutput = torch.zeros(M).to(torch.float32)  # 初始化量化缩放因子张量
    # 前处理 X_INT8 -> X_INT4
    x_INT4 = x_INT8_to_x_INT4(x)
    start_idx = 0  # 起始索引
    preV = 0  # 前一个组的 token 数量
    groupList = groupList.tolist()
    # 遍历 groupList，按组处理数据
    for i, v in enumerate(groupList):
        currV = v
        tempV = int(
            (currV - preV) * 2
        )  # 计算当前组的 token 数量 “ * 2 ”是因为 1行Int8--> 2行Int4
        preV = currV  # 更新前一个组的 token 数量
        if tempV > 0:
            # 调用 MM_Swiglu_quant 处理当前组

            MMOut = Custom_MM(
                x_INT4[int(start_idx) : int(start_idx + tempV)],
                weight[i],
                weightScale[i],
                tempV,
            )
            MM_Num_Concat = (MMOut[::2] * 16 + MMOut[1::2]) + weightAssistanceMatrix[
                i
            ].view(1, -1)

            PerToken_Quant = MM_Num_Concat * perTokenScale[
                start_idx // 2 : (start_idx + tempV) // 2
            ].view(-1, 1)
            # 将结果分成两部分以应用 SwiGLU 激活函数
            SwiGLU, gate = PerToken_Quant.chunk(2, dim=-1)
            temp = SwiGLU * torch.sigmoid(SwiGLU)  # SwiGLU 激活
            temp = temp * gate  # 与门控值进行逐元素相乘
            # 对输出进行量化
            max_value = torch.max(
                torch.abs(temp), dim=-1
            ).values  # 找到最大绝对值以计算缩放因子
            quantScaleOutput_temp = 127 / max_value  # 计算量化缩放因子
            quantOutput[start_idx // 2 : (start_idx + tempV) // 2] = torch.round(
                temp * quantScaleOutput_temp.reshape(tempV // 2, 1)
            ).to(torch.int8)  # 量化为 int8
            quantScaleOutput[start_idx // 2 : (start_idx + tempV) // 2] = (
                max_value / 127
            )  # 反向量化缩放因子以便后续反量化
        start_idx += tempV  # 更新起始索引以处理下一组
    return quantOutput, quantScaleOutput


def generate_non_decreasing_sequence(length, upper_limit, seed):
    """
    生成一个随机非减的一维 Tensor，且最后一个值小于上限。

    参数:
        length (int): 序列的长度。
        upper_limit (int): 最后一个值的上限。

    返回:
        torch.Tensor: 生成的一维 Tensor。
    """
    # 生成随机递增序列
    torch.manual_seed(seed)
    random_increments = torch.randint(0, 512, (length,))  # 随机增量，范围 0~9
    sequence = torch.cumsum(random_increments, dim=0).to(
        torch.int64
    )  # 累加生成非减序列

    # 确保最后一个值小于上限
    if sequence[-1] >= upper_limit:
        scale_factor = upper_limit / sequence[-1]  # 计算缩放因子
        sequence = (sequence * scale_factor).to(torch.int64)  # 缩放并转换为整数
        random_increments = (random_increments * scale_factor).to(torch.int64)
    return sequence


def cumsum_to_count(cumsum_tensor):
    """
    将累积和模式的Tensor转换为计数模式

    参数:
        cumsum_tensor: 一维Tensor，表示累积和

    返回:
        count_tensor: 一维Tensor，表示原始计数
    """
    # 在开头添加0，然后计算相邻元素的差值
    padded = torch.cat(
        [torch.tensor([0], device=cumsum_tensor.device), cumsum_tensor[:-1]]
    )
    count_tensor = cumsum_tensor - padded
    return count_tensor


def gen_input_data(E, M, K, N, KNum_per_group, dequantMode):
    x = torch.randint(-128, 127, (M, K), dtype=torch.int8)
    weight = torch.randint(-5, 5, (E, K, N), dtype=torch.int8)
    if dequantMode == 1:
        assert K % KNum_per_group == 0, (
            "per-channel&&per-group模式下， K必须为KNum_per_group的整数倍"
        )
        weightScale = (
            torch.randint(-2, 2, (E, K // KNum_per_group, N))
            .to(torch.bfloat16)
            .to(torch.float32)
        )
    elif dequantMode == 0:
        weightScale = torch.randint(-2, 2, (E, N)).to(torch.bfloat16).to(torch.float32)
    xScale = 0.1 * torch.randn(M)
    repeat_times = K // weightScale.shape[1] if dequantMode == 1 else 1
    expanded_scale = weightScale.view(E, -1, N).repeat_interleave(repeat_times, dim=1)
    weightAssistanceMatrix = (8 * weight * expanded_scale).sum(dim=1)
    groupList = generate_non_decreasing_sequence(E, M, 42)
    return x, weight, weightScale, xScale, weightAssistanceMatrix, groupList


@register("function_aclnn_grouped_matmul_swiglu_quant_v2")
class AclnnGroupedMatmulSwigluQuantA8W4(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        """
        该接口可实现部门场景下api的初始化需要依赖于当前的输入数据，且不希望计入耗时，
        可以在此接口实现
        :param input_data:
        :return:
        """
        dequantMode = input_data.kwargs["dequantMode"]
        self.x = input_data.kwargs["x"].clone()
        self.weight = input_data.kwargs["weight"][0].clone()

        self.weightScale = (
            input_data.kwargs["weightScale"][0]
            .to(torch.bfloat16)
            .to(torch.float32)
            .clone()
        )
        self.xScale = (
            input_data.kwargs["xScale"].to(torch.bfloat16).to(torch.float32).clone()
        )
        E, M, K, N, K_group = (
            self.weight.shape[0],
            self.x.shape[0],
            self.x.shape[1],
            self.weightScale.shape[-1],
            (self.weightScale.shape[1] if dequantMode == 1 else 1),
        )
        print(
            f">>>>>>>>>>>>>>>>>E:{E}, M:{M}, K:{K}, N:{N}, K_group:{K_group}<<<<<<<<<<<<<<<<<"
        )  # 7 20 16 256 1
        seed = input_data.kwargs["seed"]
        torch.manual_seed(seed)
        (
            self.x,
            self.weight,
            self.weightScale,
            self.xScale,
            self.weightAssistanceMatrix,
            self.grouplist,
        ) = gen_input_data(E, M, K, N, K // K_group, dequantMode)
        x = self.x.clone()
        weight = self.weight.clone()
        weightScale = self.weightScale.clone()
        xScale = self.xScale.clone()
        weightAssistanceMatrix = self.weightAssistanceMatrix.clone()
        grouplist = self.grouplist.clone()

        self.weight_format = 1
        input_data.kwargs.pop("seed")
        input_data.kwargs.pop("case")

        if self.device == "pyaclnn":
            # NPU上板执行
            if self.weight_format == 0:  # ND
                weight_quant = torch_npu.npu_quantize(
                    weight.to(torch.float32).npu(),
                    torch.tensor([1.0], device="npu"),
                    None,
                    torch.quint4x2,
                    -1,
                    False,
                )
            if self.weight_format == 1:  # NZ
                weight_quant = (
                    weight.reshape(E, K // 16, 16, N // 64, 64)
                    .permute(0, 3, 1, 2, 4)
                    .contiguous()
                )
                weight_quant = weight_quant.npu()
                weight_quant = torch_npu.npu_quantize(
                    weight_quant.to(torch.float32),
                    torch.tensor([1.0], device="npu"),
                    None,
                    torch.quint4x2,
                    -1,
                    False,
                )

            if dequantMode == 1:
                weightScale = weightScale.view(E, -1, N)
                KGroup = weightScale.shape[1]
                scale_np = weightScale.cpu().numpy()
                scaleUint32 = scale_np.astype(np.float32)
                scaleUint32.dtype = np.uint32
                scaleUint64 = np.zeros((E, KGroup, N * 2), dtype=np.uint32)
                scaleUint64[..., ::2] = scaleUint32
                scaleUint64.dtype = np.int64
                scale = torch.from_numpy(scaleUint64)
            else:
                weightScale = weightScale.view(E, N)  # 直接保持 (E, N) 形状
                scale_np = weightScale.cpu().numpy()
                scaleUint32 = scale_np.astype(np.float32)
                scaleUint32.dtype = np.uint32
                scaleUint64 = np.zeros((E, N * 2), dtype=np.uint32)
                scaleUint64[..., ::2] = scaleUint32
                scaleUint64.dtype = np.int64
                scale = torch.from_numpy(scaleUint64.reshape(E, N))

            input_data.kwargs["x"] = x.npu()
            input_data.kwargs["weight"][0] = weight_quant
            input_data.kwargs["weightScale"][0] = scale.npu()
            input_data.kwargs["xScale"] = xScale.npu()
            input_data.kwargs["smoothScale"] = TensorPtr()
            input_data.kwargs["weightAssistMatrix"][0] = weightAssistanceMatrix.npu()

            if input_data.kwargs["groupListType"] == 0:
                input_data.kwargs["groupList"] = grouplist.npu()
            else:
                groupList_cumsum = self.grouplist.clone()
                groupList_count = cumsum_to_count(groupList_cumsum)
                input_data.kwargs["groupList"] = groupList_count.npu()

    def get_format(self, input_data: InputDataset, index=None, name=None):
        """
        :param input_data: 参数列表
        :param index: 参数位置
        :param name: 参数名字
        :return:
        format at this index or name
        """
        if name == "weight" and self.weight_format:
            return AclFormat.ACL_FORMAT_FRACTAL_NZ

        return AclFormat.ACL_FORMAT_ND

    def __call__(self, input_data: InputDataset, with_output: bool = True):
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"
        x_in = self.x.clone()
        weight_in = self.weight.clone()
        weightScale_in = self.weightScale.clone()
        xScale_in = self.xScale.clone()
        weightAssistanceMatrix_in = self.weightAssistanceMatrix.clone()
        groupList_in = self.grouplist.clone()
        quantOutput, quantScaleOutput = process_groups(
            x_in,
            weight_in,
            weightScale_in,
            xScale_in,
            weightAssistanceMatrix_in,
            groupList_in,
        )
        E = groupList_in[-1]

        t2 = torch.zeros_like(quantOutput)
        t1 = torch.zeros_like(quantScaleOutput)
        quantOutput[E:, :] = t2[E:, :]
        quantScaleOutput[E:] = t1[E:]
        return quantOutput, quantScaleOutput


@register("function_pyaclnn_grouped_matmul_swiglu_quant_v2")
class PyaclnnGroupedMatmulSwigluQuantA8W4(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super().__init__(task_result, backend)
        self.input_args = None
        self.groupList = None
        self.isCount = None

    def init_by_input_data(self, input_data: InputDataset):
        self.groupList = input_data.kwargs["groupList"]
        self.isCount = input_data.kwargs["groupListType"]

        self.input_args, output_packages = super().init_by_input_data(input_data)

        return self.input_args, output_packages

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        if self.isCount:
            groupindex = 0
            for item in self.groupList:
                groupindex += item
        else:
            groupindex = self.groupList[-1].item()
        for idx, output_tmp in enumerate(output):
            padded_tensor = torch.zeros_like(output_tmp)
            if idx == 0:
                padded_tensor[:groupindex, :] = output_tmp[:groupindex, :]
            elif idx == 1:
                padded_tensor[:groupindex] = output_tmp[:groupindex]
            output[idx] = padded_tensor  # 回填！

        return output
