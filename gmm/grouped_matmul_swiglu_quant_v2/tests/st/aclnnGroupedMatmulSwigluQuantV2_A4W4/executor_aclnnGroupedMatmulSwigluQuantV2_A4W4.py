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


def MM(
    x: torch.Tensor,
    weight: torch.Tensor,
    perChannelScale: torch.Tensor,
    perTokenScale: torch.Tensor,
    m: int,
    outDtype: torch.dtype,
    KNum_per_group: int,
    groupListType: int,
    dequantMode: int,
):
    """
    执行量化的 GMM（通用矩阵乘法）操作，并使用 SwiGLU 激活函数。

    参数:
        x (torch.Tensor): 输入张量，形状为 (m, k)。
        weight (torch.Tensor): 权重张量，形状为 (k, n)。
        perChannelScale (torch.Tensor): 每个通道的缩放因子。
        -    当dequantMode为1：per-GroupScale.shape为(k // KNum_per_group, n)
        -    当dequantMode为0：per-ChannelScale.shape为(1, n)
        perTokenScale (torch.Tensor): 每个 token 的缩放因子，形状为 (m,)。
        m (int): token 的数量（x 的行数）。

    返回:
        Output (torch.Tensor): 输出张量，形状为 (m, n)。
    """
    K, N = x.shape[1], weight.shape[1]
    if dequantMode == 1:
        c_temp1 = torch.zeros(m, N).type(torch.float16)
        for k_idx in range(K // KNum_per_group):
            c_temp1 = c_temp1 + (
                perChannelScale[k_idx].reshape(1, N)
                * torch.matmul(
                    x[:, k_idx * KNum_per_group : (k_idx + 1) * KNum_per_group].type(
                        torch.int32
                    ),
                    weight[
                        k_idx * KNum_per_group : (k_idx + 1) * KNum_per_group, :
                    ].type(torch.int32),
                ).to(torch.float16)
            ).type(torch.float16)
        c_temp2 = perTokenScale.reshape(m, 1) * c_temp1.to(torch.float32)
        return c_temp2.type(outDtype).to(torch.float32)
    elif dequantMode == 0:
        # 使用 int32 精度执行矩阵乘法
        c_temp1 = torch.matmul(x.type(torch.int32), weight.type(torch.int32))
        c_temp1 = c_temp1.type(torch.float32)  # 转换回 float32 以便进行缩放
        # 应用每个通道和每个 token 的缩放
        c_temp2 = torch.mul(c_temp1, perChannelScale)
        c_temp3 = torch.mul(c_temp2, perTokenScale.reshape(m, 1))
        return c_temp3.type(outDtype).to(torch.float32)


def SwiGLUQuant(MMOut: torch.Tensor, smoothScale: torch.Tensor, m: int):
    # 将结果分成两部分以应用 SwiGLU 激活函数
    c_temp4, gate = MMOut.chunk(2, dim=-1)
    c_temp5 = c_temp4 * torch.sigmoid(c_temp4)  # SwiGLU 激活
    c_temp6 = c_temp5 * gate  # 与门控值进行逐元素相乘
    c_temp7 = c_temp6 * smoothScale
    # 对输出进行量化
    max = torch.max(torch.abs(c_temp7), -1).values  # 找到最大绝对值以计算缩放因子
    quantScaleOutput = 127 / max  # 计算量化缩放因子
    quantOutput = torch.round(c_temp7 * quantScaleOutput.reshape(m, 1)).to(
        torch.int8
    )  # 量化为 int8
    quantScaleOutput = max / 127  # 反向量化缩放因子以便后续反量化
    return quantOutput, quantScaleOutput


def GMMA4W4(
    x: torch.Tensor,
    weight: torch.Tensor,
    perChannelScale: torch.Tensor,
    perTokenScale: torch.Tensor,
    smoothScale: torch.Tensor,
    groupList: torch.Tensor,
    MMOutDtype: torch.dtype,
    KNum_per_group: int,
    groupListType: int,
    dequantMode: int,
):
    """
    按组处理输入数据，并调用 GMM_Swiglu_quant 函数进行量化计算。

    参数:
        x (torch.Tensor): 输入张量，形状为 (M, K)。
        weight (torch.Tensor): 权重张量列表，每个元素的形状为 (E, K, N)。
        perChannelScale (torch.Tensor): 每个通道的缩放因子列表。
        -    当dequantMode为1时，shape为 (E,K//KNum_per_group,N);
        -    当dequantMode为0时，shape为 (E,N);
        perTokenScale (torch.Tensor): 每个 token 的缩放因子，形状为 (M,)。
        groupList (list): 定义每个组的 token 数量的列表。
        -    当groupListType为0时，List语义为cumsum模式;
        -    当groupListType为1时，List语义为count模式;

    返回:
        Output (torch.Tensor): 输出张量，形状为 (M, N)。
    """
    M, N = x.shape[0], weight.shape[2]  # 获取输入张量的形状
    MMOut = torch.zeros(M, N).type(MMOutDtype)  # 初始化量化输出张量
    quantOutput = torch.zeros(M, N // 2).type(torch.int8)
    quantScaleOutput = torch.zeros(M).type(torch.float32)
    start_idx = 0  # 起始索引
    preV = 0  # 前一个组的 token 数量
    groupList = groupList.tolist()
    # 遍历 groupList，按组处理数据
    for i, v in enumerate(groupList):
        if groupListType == 0:
            currV = v
            tempV = currV - preV  # 计算当前组的 token 数量
            preV = currV  # 更新前一个组的 token 数量
        elif groupListType == 1:
            tempV = v  # 计算当前组的 token 数量
        if tempV > 0:
            # 调用 GMM_Swiglu_quant 处理当前组
            MMOut[start_idx : start_idx + tempV] = MM(
                x[start_idx : start_idx + tempV],
                weight[i],
                perChannelScale[i],
                perTokenScale[start_idx : start_idx + tempV],
                tempV,
                MMOutDtype,
                KNum_per_group,
                groupListType,
                dequantMode,
            ).to(torch.float32)
            (
                quantOutput[start_idx : start_idx + tempV],
                quantScaleOutput[start_idx : start_idx + tempV],
            ) = SwiGLUQuant(
                MMOut[start_idx : start_idx + tempV].to(torch.float32),
                smoothScale[i],
                tempV,
            )

        start_idx += tempV  # 更新起始索引以处理下一组
    return quantOutput, quantScaleOutput


def generate_non_decreasing_sequence(length, upper_limit, groupListType):
    """
    生成一个随机非减的一维 Tensor，且最后一个值小于上限。

    参数:
        length (int): 序列的长度。
        upper_limit (int): 最后一个值的上限。

    返回:
        torch.Tensor: 生成的一维 Tensor。
    """
    # 生成随机递增序列
    torch.manual_seed(42)
    random_increments = torch.randint(0, 128, (length,))  # 随机增量，范围 0~9
    sequence = torch.cumsum(random_increments, dim=0).to(
        torch.int64
    )  # 累加生成非减序列

    # 确保最后一个值小于上限
    if sequence[-1] >= upper_limit:
        scale_factor = upper_limit / sequence[-1]  # 计算缩放因子
        sequence = (sequence * scale_factor).to(torch.int64)  # 缩放并转换为整数
        random_increments = (random_increments * scale_factor).to(torch.int64)
    if groupListType == 0:
        return sequence
    else:
        return cumsum_to_count(sequence)


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


def gen_input_data(E, M, K, N, KNum_per_group, groupListType, dequantMode):
    x = torch.randint(-8, 7, (M, K), dtype=torch.int8)
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
    smoothScale = torch.randn(E)
    groupList = generate_non_decreasing_sequence(E, M, groupListType)
    return x, weight, weightScale, xScale, smoothScale, groupList


@register("function_aclnn_grouped_matmul_swiglu_quant_v2")
class AclnnGroupedMatmulSwigluQuantA4W4(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        """
        该接口可实现部门场景下api的初始化需要依赖于当前的输入数据，且不希望计入耗时，
        可以在此接口实现
        :param input_data:
        :return:
        """
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
        self.smoothScale = (
            input_data.kwargs["smoothScale"]
            .to(torch.bfloat16)
            .to(torch.float32)
            .clone()
        )
        self.dequantMode = input_data.kwargs["dequantMode"]
        self.groupListType = input_data.kwargs["groupListType"]
        E, M, K, N, K_group = (
            self.weight.shape[0],
            self.x.shape[0],
            self.x.shape[1],
            self.weightScale.shape[-1],
            (self.weightScale.shape[1] if self.dequantMode == 1 else 1),
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
            self.smoothScale,
            self.grouplist,
        ) = gen_input_data(
            E, M, K, N, K // K_group, self.groupListType, self.dequantMode
        )
        x = self.x.clone()
        weight = self.weight.clone()
        weightScale = self.weightScale.clone()
        xScale = self.xScale.clone()
        smoothScale = self.smoothScale.clone()
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

            if self.dequantMode == 1:
                weightScale = weightScale.view(E, -1, N)
                KGroup = weightScale.shape[1]
                scale_np = weightScale.cpu().numpy()
                scaleUint32 = scale_np.astype(np.float32)
                scaleUint32.dtype = np.uint32
                scaleUint64 = np.zeros((E, KGroup, N * 2), dtype=np.uint32)
                scaleUint64[..., ::2] = scaleUint32
                scaleUint64.dtype = np.int64
                scale = torch.from_numpy(scaleUint64)
                print("PerGroup dequantMode: ", self.dequantMode)
            else:
                weightScale = weightScale.view(E, N)  # 直接保持 (E, N) 形状
                scale_np = weightScale.cpu().numpy()
                scaleUint32 = scale_np.astype(np.float32)
                scaleUint32.dtype = np.uint32
                scaleUint64 = np.zeros((E, N * 2), dtype=np.uint32)
                scaleUint64[..., ::2] = scaleUint32
                scaleUint64.dtype = np.int64
                scale = torch.from_numpy(scaleUint64.reshape(E, N))
                print("PerChannel dequantMode: ", self.dequantMode)

            x_quant = torch_npu.npu_quantize(
                x.to(torch.float32).npu(),
                torch.tensor([1.0], device="npu"),
                None,
                torch.quint4x2,
                -1,
                False,
            )
            input_data.kwargs["x"] = x_quant
            input_data.kwargs["weight"][0] = weight_quant
            input_data.kwargs["weightScale"][0] = scale.npu()
            input_data.kwargs["weightAssistMatrix"] = ctypes.POINTER(AclTensorList)()
            input_data.kwargs["xScale"] = xScale.npu()
            input_data.kwargs["smoothScale"] = smoothScale.npu()
            input_data.kwargs["groupList"] = grouplist.npu()

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
        smoothScale_in = self.smoothScale.clone()
        groupList_in = self.grouplist.clone()
        groupListType = (
            self.groupListType
        )  # 0 :groupList是cumsum模式； 1 :groupList是count模式
        dequantMode = (
            self.dequantMode
        )  # 0 :纯per-channel模式 1 :per-channel&&per-group模式
        K, K_group = (
            self.x.shape[1],
            (self.weightScale.shape[1] if dequantMode == 1 else 1),
        )
        KNum_per_group = K // K_group
        MMOutDtype = torch.float32

        quantOutput, quantScaleOutput = GMMA4W4(
            x_in,
            weight_in,
            weightScale_in,
            xScale_in,
            smoothScale_in,
            groupList_in,
            MMOutDtype,
            KNum_per_group,
            groupListType,
            dequantMode,
        )

        E = groupList_in[-1] if groupListType == 0 else sum(groupList_in)

        t2 = torch.zeros_like(quantOutput)
        t1 = torch.zeros_like(quantScaleOutput)
        quantOutput[E:, :] = t2[E:, :]
        quantScaleOutput[E:] = t1[E:]
        return quantOutput, quantScaleOutput


@register("function_pyaclnn_grouped_matmul_swiglu_quant_v2")
class PyaclnnGroupedMatmulSwigluQuantA4W4(AclnnBaseApi):
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
