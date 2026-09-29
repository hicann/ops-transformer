#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


@register("function_MoeGatingTopK")
class MoeGatingTopKMethod(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MoeGatingTopKMethod, self).__init__(task_result)

    @staticmethod
    def softmax_func(x, axis=None):
        # print("before softmax:",x)
        if "float16" in x.dtype.name:
            x = x.astype(np.float32)
        x_max = x.max(axis=axis, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=axis, keepdims=True)
        # print(" x_sum:",x_sum)
        ans = y / x_sum
        # print(" ans:",ans)
        return ans, x_max, x_sum

    def single_golden(
        self,
        x,
        k,
        bias,
        k_group,
        group_count,
        group_select_mode,
        renorm,
        norm_type,
        y2_flag,
        routed_scaling_factor,
        eps,
    ):
        dtype = x.dtype
        # Convert to float32 if needed
        if dtype != torch.float32:
            x = x.to(torch.float32)
            if bias is not None:
                bias = bias.to(torch.float32)

        # Convert to numpy and process
        x = x.numpy()
        if bias is not None:
            bias = bias.numpy()

        # softmax/ sigmoid processing
        if norm_type == 0:
            x, _, _ = self.softmax_func(x, axis=-1)
        else:
            x = 1 / (1 + np.exp(-x))

        original_x = x
        if bias is not None:
            x = x + bias

        # group count processing
        if group_count > 1:
            x = x.reshape(x.shape[0], group_count, -1)
            if group_select_mode == 0:
                group_x = np.amax(x, axis=-1)
            else:
                group_x = np.partition(x, -2, axis=-1)[..., -2:].sum(axis=-1)

            indices = np.argsort(-group_x, axis=-1, kind="stable")[:, :k_group]

            mask = np.ones((x.shape[0], group_count), dtype=bool)
            mask[np.arange(x.shape[0])[:, None], indices] = False
            x = np.where(mask[..., None], float("-inf"), x).reshape(x.shape[0], -1)

        # Topk indices and gather
        _, indices = torch.sort(
            torch.from_numpy(x), dim=-1, stable=True, descending=True
        )
        indices = np.asarray(indices[:, :k]).astype(np.int32)

        y = np.take_along_axis(original_x, indices, axis=1)

        # Renorm and scaling
        if norm_type == 1 or renorm == 1:
            temp = np.sum(y, axis=-1, keepdims=True) + eps
            y /= temp
        y *= routed_scaling_factor

        # Process y2 if needed
        if y2_flag:
            y2 = torch.from_numpy(original_x).to(dtype).to(torch.float32)
        else:
            y2 = None

        # Convert back to torch tensors
        y = torch.from_numpy(np.asarray(y)).to(dtype)
        indices = torch.from_numpy(indices).to(torch.int32)

        return y, indices, y2

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            output = self.single_golden(
                input_data.kwargs["x"],
                input_data.kwargs["k"],
                input_data.kwargs["bias"],
                input_data.kwargs["k_group"],
                input_data.kwargs["group_count"],
                input_data.kwargs["group_select_mode"],
                input_data.kwargs["renorm"],
                input_data.kwargs["norm_type"],
                input_data.kwargs["out_flag"],
                input_data.kwargs["routed_scaling_factor"],
                input_data.kwargs["eps"],
            )
            return output


@register("function_aclnnMoeGatingTopK")
class MoeGatingTopKAclnnApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))

        return output
