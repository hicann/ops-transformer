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
import numpy
import numpy as np
import math
from atk.configs.dataset_config import InputDataset

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.dataset.base_dataset import OpsDataset
from typing import Union, List


# golden
def adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
    count = 0
    last = sorted_expert_idx[0]
    for i, val in enumerate(sorted_expert_idx):
        if last != val:
            count = 1
            last = val
        else:
            count += 1
            if count > capacity:
                sorted_expert_idx[i] = -1
                sorted_row_idx[i] = -1


@register("function_aclnn_moe_init_routing_v3")
class FunctionMoeInitRoutingV3Api(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x_dtype = input_data.kwargs.get("x").dtype
        active_expert_range = input_data.kwargs["active_expert_range"]
        expert_start = active_expert_range[0]
        expert_end = active_expert_range[1]
        active_num = input_data.kwargs["active_num"]
        expert_num = input_data.kwargs["expert_num"]
        expert_tokens_num_type = input_data.kwargs["expert_tokens_num_type"]
        expert_tokens_num_flag = input_data.kwargs["expert_tokens_num_flag"]
        expert_capacity = input_data.kwargs["expert_capacity"]
        scale = input_data.kwargs["scale"].cpu().numpy().copy()
        offset = input_data.kwargs["offset"].cpu().numpy().copy()
        x = input_data.kwargs["x"].cpu()
        if x.dtype == torch.bfloat16:
            x = x.float()
        elif x.dtype == torch.float16:
            x = x.float()
        x = x.numpy()
        num_rows = x.shape[0]
        h = x.shape[1]
        expert_idx = input_data.kwargs["expert_idx"].cpu().numpy().copy()
        k = expert_idx.shape[1]
        row_idx_type = input_data.kwargs["row_idx_type"]
        drop_pad_mode = input_data.kwargs["drop_pad_mode"]
        quant_mode = input_data.kwargs["quant_mode"]
        expert_idx_in = expert_idx.copy().reshape(-1)
        actual_expert_total_num = np.sum(
            (expert_idx >= expert_start) & (expert_idx < expert_end)
        )

        expert_idx_in[(expert_idx_in < expert_start)] = numpy.int32(
            numpy.iinfo(numpy.int32).max
        )
        sorted_expert_indices = numpy.argsort(expert_idx_in, axis=-1, kind="stable")
        sorted_expert_idx = expert_idx_in[sorted_expert_indices]

        if row_idx_type == 1:
            expanded_row_idx = sorted_expert_indices
        else:
            # gather
            expanded_row_idx = numpy.ones(num_rows * k).astype(numpy.int32) * -1
            tmp_indices = numpy.arange(actual_expert_total_num)
            expanded_row_idx[sorted_expert_indices[:actual_expert_total_num]] = (
                tmp_indices
            )
        # 计算直方图
        if (
            not expert_tokens_num_flag
        ):  # 注意：新加expert_tokens_num_flag的false模式，此时不输出expert_tokens_count
            expert_tokens_count = None
        else:
            if drop_pad_mode == 0:
                if expert_tokens_num_type == 1:
                    # 计算直方图
                    expert_tokens_count = numpy.bincount(
                        sorted_expert_idx[:actual_expert_total_num] - expert_start
                    )
                    expert_tokens_count = numpy.concatenate(
                        [
                            expert_tokens_count,
                            numpy.zeros(
                                (expert_end - expert_start) - len(expert_tokens_count)
                            ).astype(numpy.int64),
                        ]
                    )
                elif (
                    expert_tokens_num_type == 0
                ):  # 注意：新加cumsum模式，沿用直方图计算逻辑，最后对直方图结果进行累加得到
                    expert_tokens_count = numpy.bincount(
                        sorted_expert_idx[:actual_expert_total_num] - expert_start
                    )
                    expert_tokens_count = numpy.concatenate(
                        [
                            expert_tokens_count,
                            numpy.zeros(
                                (expert_end - expert_start) - len(expert_tokens_count)
                            ).astype(numpy.int64),
                        ]
                    )
                    expert_tokens_count = numpy.cumsum(expert_tokens_count)
                elif expert_tokens_num_type == 2:
                    # key-value
                    expert_id, counts = numpy.unique(
                        sorted_expert_idx[:actual_expert_total_num], return_counts=True
                    )
                    expert_tokens_count = numpy.column_stack((expert_id, counts))
                    if expert_tokens_count.shape[0] < expert_num:
                        expert_tokens_count = numpy.concatenate(
                            (
                                expert_tokens_count,
                                [
                                    [0, 0],
                                ],
                            ),
                            axis=0,
                        )
                    expert_tokens_count = numpy.concatenate(
                        [
                            expert_tokens_count,
                            numpy.zeros((expert_num - expert_tokens_count.shape[0]) * 2)
                            .reshape(-1, 2)
                            .astype(numpy.int64),
                        ]
                    )
            else:  # 注意：新加droppad为1时的expert_tokens_count，为count模式
                expert_tokens_count = numpy.bincount(
                    sorted_expert_idx[:actual_expert_total_num] - expert_start
                )
                expert_tokens_count = numpy.concatenate(
                    [
                        expert_tokens_count,
                        numpy.zeros(
                            (expert_end - expert_start) - len(expert_tokens_count)
                        ).astype(numpy.int64),
                    ]
                )
            expert_tokens_count = expert_tokens_count.astype(numpy.int64)

        if (
            drop_pad_mode == 0
        ):  # 注意：droppad为0时通过active_num区分dropless和active场景，droppad为1时进行截断和补零
            if active_num == 0:
                active_num = actual_expert_total_num
            else:
                active_num = min(active_num, actual_expert_total_num)
            expanded_scale = None
            expanded_x = x[sorted_expert_indices[:active_num] // k, :]
            if scale is not None and quant_mode == -1:
                expanded_scale = scale[sorted_expert_indices[:active_num] // k]
        else:
            # 注意：droppad=1时计算逻辑，沿用quantV2
            adapter_capacity(sorted_expert_indices, sorted_expert_idx, expert_capacity)

            sort_row_tmp = numpy.full((expert_num * expert_capacity), -1, dtype=int)
            offset_tmp = 0
            lastExpertId = 0
            for i, val in enumerate(sorted_expert_indices):
                if val != -1:
                    if lastExpertId != sorted_expert_idx[i]:
                        offset_tmp = 0
                        lastExpertId = sorted_expert_idx[i]

                    sort_row_tmp[
                        sorted_expert_idx[i] * expert_capacity + offset_tmp
                    ] = sorted_expert_indices[i]
                    offset_tmp = offset_tmp + 1

            # expand_row_idx
            expanded_row_idx = numpy.full(sorted_expert_indices.shape, -1)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_row_idx[val] = i

            # expanded_x
            expanded_x_mask = numpy.full(
                (expert_num * expert_capacity, h), 1, dtype=int
            )
            expanded_x = numpy.full((expert_num * expert_capacity, h), 0, dtype=x.dtype)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_x[i] = x[val // k]
                    expanded_x_mask[i] = numpy.full((h,), 0, dtype=int)

        # 非量化
        if quant_mode == -1:
            expanded_x = expanded_x
            expanded_row_idx = expanded_row_idx
            if scale is None or drop_pad_mode == 1:
                expanded_scale = None

        # 静态量化
        if quant_mode == 0:
            expanded_scale = None
            expanded_x_fp32 = expanded_x.astype(numpy.float32)
            scale_val = scale.astype(numpy.float32)
            offset_val = offset.astype(numpy.float32)
            scale_rst = expanded_x_fp32 * scale_val[0]
            add_offset = scale_rst + offset_val[0]
            round_data = numpy.rint(add_offset)
            round_data = numpy.clip(round_data, -128, 127)
            expanded_x = round_data.astype(numpy.int8)

        # 动态量化
        if quant_mode == 1:
            x_final = expanded_x.astype(numpy.float32)
            if scale is None:
                x_abs = numpy.abs(x_final)
                x_max = numpy.max(x_abs, axis=-1, keepdims=True)
                expanded_scale = x_max / 127
                expanded_x = x_final / expanded_scale
                expanded_x = numpy.round(expanded_x).astype(numpy.int8)
            else:
                scale = scale.astype(numpy.float32)
                if scale.shape[0] == 1:
                    x_final = x_final * scale
                else:
                    if drop_pad_mode == 0:
                        x_final = (
                            x_final
                            * scale[sorted_expert_idx[:active_num] - expert_start]
                        )

                    else:
                        for i, val in enumerate(sort_row_tmp):
                            if val != -1:
                                x_final[i] = x_final[i] * scale[i // expert_capacity]

                x_abs = numpy.abs(x_final)
                x_max = numpy.max(x_abs, axis=-1, keepdims=True)
                expanded_scale = x_max / 127
                expanded_x = x_final / expanded_scale
                expanded_x = numpy.round(expanded_x).astype(numpy.int8)
        if drop_pad_mode == 1:
            expanded_x = numpy.ma.array(expanded_x, mask=expanded_x_mask).filled(0)
            expanded_x = expanded_x.reshape(expert_num, expert_capacity, h)

        if expert_tokens_count is None:
            expert_tokens_count = torch.tensor([], dtype=torch.int64)
        else:
            expert_tokens_count = torch.from_numpy(expert_tokens_count.astype("int64"))
        if expanded_scale is None:
            expanded_scale = torch.tensor([], dtype=torch.float32)
        else:
            expanded_scale = torch.from_numpy(expanded_scale.astype("float32")).reshape(
                -1
            )

        expanded_row_idx = torch.tensor(expanded_row_idx, dtype=torch.int32)
        if quant_mode == -1:
            expanded_x = torch.from_numpy(expanded_x).to(x_dtype)
        else:
            expanded_x = torch.from_numpy(expanded_x)

        return expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale


@register("function_pyaclnn_moe_init_routing_v3")
class PyAclnnMoeInitRoutingV3(AclnnBaseApi):  # SampleApi类型仅需设置唯一即可。
    def __init__(self, task_result: TaskResult, backend):
        super().__init__(task_result, backend)

    def init_by_input_data(self, input_data: InputDataset):
        # 如果某个输入参数想要传入为空，yaml和json文件目前不支持，可以通过以下过程实现
        import ctypes
        from atk.tasks.backends.lib_interface.acl_wrapper import (
            nnopbase,
            AclDataType,
            AclFormat,
            AclTensor,
        )

        input_args, output_packages = super().init_by_input_data(input_data)
        AclTensorPtr = ctypes.POINTER(AclTensor)  # tensor指针类型
        null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
        null_tensor_ptr = ctypes.cast(
            null_void_ptr, AclTensorPtr
        )  # 把这个空指针类型转换为tensor指针类型
        if (
            input_data.kwargs["quant_mode"] == -1
            and input_data.kwargs["drop_pad_mode"] == 1
        ):
            input_args[2] = null_tensor_ptr  # 赋值给第3个参数
        return input_args, output_packages

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnMoeInitRoutingV3GetWorkspaceSize("
            "const aclTensor *x, "
            "const aclTensor *expertIdx, "
            "const aclTensor *scaleOptional, "
            "const aclTensor *offsetOptional, "
            "int64_t activeNum, "
            "int64_t expertCapacity, "
            "int64_t expertNum, "
            "int64_t dropPadMode, "
            "int64_t expertTokensNumType, "
            "bool expertTokensNumFlag, "
            "int64_t quantMode, "
            "const aclIntArray *activeExpertRangeOptional, "
            "int64_t rowIdxType, "
            "const aclTensor *expandedXOut, "
            "const aclTensor *expandedRowIdxOut, "
            "const aclTensor *expertTokensCountOrCumsumOut, "
            "const aclTensor *expandedScaleOut, "
            "uint64_t *workspaceSize, "
            "aclOpExecutor **executor)"
        )
