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
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import numpy as np
import tensorflow as tf


def moe_init_routing_v2_grad(
    grad_expanded_x: np.ndarray,
    expanded_row_idx: np.ndarray,
    top_k,
    drop_pad_mode,
    active_num,
) -> np.ndarray:
    num_rows = grad_expanded_x.shape[0] // top_k
    hidden_size = grad_expanded_x.shape[-1]

    grad_x = np.zeros((num_rows, hidden_size), dtype=np.float32)
    for i in range(num_rows):
        for j in range(i * top_k, i * top_k + top_k, 1):
            expanded_x_idx = expanded_row_idx[j]

            if drop_pad_mode == 1:
                if expanded_x_idx == -1:
                    continue
            elif active_num > 0:
                if expanded_x_idx >= active_num:
                    continue
            grad_x[i] = np.add(
                grad_x[i], grad_expanded_x[expanded_x_idx].astype(np.float32)
            )

    return grad_x


@register("ascend_method_torch_nn_moe_init_routing_v2_grad")
class MethodTorchNnMIRGApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchNnMIRGApi, self).__init__(task_result)
        OpsDataset.seed_everything()

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # if self.device == 'cpu':
        if input_data.kwargs["grad_expanded_x"].dtype == torch.bfloat16:
            grad_expanded_x = input_data.kwargs["grad_expanded_x"].float().cpu().numpy()
            grad_expanded_x = grad_expanded_x.astype(tf.bfloat16.as_numpy_dtype)
        else:
            grad_expanded_x = input_data.kwargs["grad_expanded_x"].cpu().numpy()
        expanded_row_idx = input_data.kwargs["expanded_row_idx"].cpu().numpy()
        top_k = int(input_data.kwargs.get("top_k"))
        drop_pad_mode = int(input_data.kwargs.get("drop_pad_mode"))
        active_num = int(input_data.kwargs.get("active_num"))
        print(
            ">> run top_k",
            top_k,
            "drop_pad_mode",
            drop_pad_mode,
            "active_num",
            active_num,
        )
        output = moe_init_routing_v2_grad(
            grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode, active_num
        )
        return torch.from_numpy(output)


@register("aclnn_method_torch_nn_moe_init_routing_v2_grad")
class MIRGAclnnApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        input_args = []
        output_packages = []

        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)

        for name, kwarg in input_data.kwargs.items():
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)

        dtype = input_data.kwargs["grad_expanded_x"].dtype
        for index, output_data in enumerate(self.task_result.output_info_list):
            output_data.dtype = str(dtype)
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)
        input_args.extend(output_packages)

        return input_args, output_packages
