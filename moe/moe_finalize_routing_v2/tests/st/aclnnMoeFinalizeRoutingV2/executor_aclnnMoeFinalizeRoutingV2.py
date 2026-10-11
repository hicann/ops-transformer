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
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat


def MoeFinalizeRoutingV2Golden(
    expandedX: torch.Tensor,
    expandedRowIdx: torch.Tensor,
    x1Optional: torch.Tensor,
    x2Optional: torch.Tensor,
    biasOptional: torch.Tensor,
    scalesOptional: torch.Tensor,
    expertIdxOptional: torch.Tensor,
    dropPadMode: int,
):
    """
    按组处理输入数据，并调用 GMM_Swiglu_quant 函数进行量化计算。

    参数:
        expandedX (torch.Tensor): 输入张量，形状为 (M, N)。
        expandedRowIdx (torch.Tensor): 权重张量列表，每个元素的形状为 (n, k)。
        x1Optional (torch.Tensor): 每个通道的缩放因子列表，每个元素的形状为 (k,)。
        x2Optional (torch.Tensor): 每个 token 的缩放因子，形状为 (M,)。
        biasOptional (torch.Tensor): 定义每个组的 token 数量的列表。
        scalesOptional (torch.Tensor): 定义每个组的 token 数量的列表。
        expertIdxOptional (torch.Tensor): 定义每个组的 token 数量的列表。
        dropPadMode (int): 定义每个组的 token 数量的列表。

    返回:
        Output (torch.Tensor): 量化后的输出张量，形状为 (M, N // 2)。
    """
    if len(expandedX.shape) == 2:
        Num_Rows = expertIdxOptional.shape[0]
        K = expertIdxOptional.shape[1]
        H = expandedX.shape[-1]
        output = torch.empty(Num_Rows, H)
        for i in range(Num_Rows):
            temp_sum = torch.zeros(H)
            for k in range(K):
                temp_sum += scalesOptional[i, k] * (
                    expandedX[expandedRowIdx[i + k * Num_Rows]]
                    + biasOptional[expertIdxOptional[i, k]]
                )
            output[i] = x1Optional[i] + x2Optional[i] + temp_sum
        # print(output.dtype)
        # print(output.shape)
        return output.to(expandedX.dtype)

    if len(expandedX.shape) == 3:
        pass
        # E,C,H = expandedX.shape
        # K = scalesOptional.shape[1]
        # Num_Rows = expandedRowIdx.shape[0] // K
        # output = torch.empty(Num_Rows, H)
        # for i in range(Num_Rows):
        #     temp_sum = torch.zeros(H)
        #     for k in range(K):
        #         temp_sum += scalesOptional[i, k] * (expandedX[expandedRowIdx[i + k * Num_Rows]] + biasOptional[expertIdxOptional[i, k]])
        #     output[i] = x1Optional[i] + x2Optional[i] + temp_sum
        # print(output.dtype)
        # print(output.shape)
        return output.to(expandedX.dtype)


@register("executor_function_aclnn_moe_finalize_routing_v2")
class AclnnMoeFinalizeRoutingV2(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        """
        该接口可实现部门场景下api的初始化需要依赖于当前的输入数据，且不希望计入耗时，
        可以在此接口实现
        :param input_data:
        :return:
        """
        seed = input_data.kwargs["seed"]
        # input_data.kwargs['groupList'] = generate_non_decreasing_sequence(weight.shape[0], x.shape[0], seed)
        input_data.kwargs.pop("seed")
        input_data.kwargs.pop("case")
        input_data.kwargs.pop("out")

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        expandedX = input_data.kwargs["expandedX"]
        expandedRowIdx = input_data.kwargs["expandedRowIdx"]
        x1Optional = input_data.kwargs["x1Optional"]
        x2Optional = input_data.kwargs["x2Optional"]
        biasOptional = input_data.kwargs["biasOptional"]
        scalesOptional = input_data.kwargs["scalesOptional"]
        expertIdxOptional = input_data.kwargs["expertIdxOptional"]
        dropPadMode = input_data.kwargs["dropPadMode"]

        output = MoeFinalizeRoutingV2Golden(
            expandedX,
            expandedRowIdx,
            x1Optional,
            x2Optional,
            biasOptional,
            scalesOptional,
            expertIdxOptional,
            dropPadMode,
        )

        return output
