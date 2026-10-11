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
import numpy as np
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_function_moe_gating_topK_softmax")
class MethodTorchGroupNormApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchGroupNormApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        def softmax_func(x, axis=None):
            x = x.astype(np.float32)
            x_max = x.max(axis=axis, keepdims=True)
            x_sub = x - x_max
            y = np.exp(x_sub)
            x_sum = y.sum(axis=axis, keepdims=True)
            ans = y / x_sum
            return ans, x_max, x_sum

        gating = input_data.kwargs["x"]
        finished_optional = input_data.kwargs["finishedOptional"]
        k = input_data.kwargs["k"]
        gating_np = gating.to(torch.float32).cpu().numpy()

        num_expert = gating.shape[-1]

        softmax, _, _ = softmax_func(gating_np, -1)

        indices = np.argsort(-softmax, axis=-1, kind="stable")
        indices = indices[:, :k]
        out = np.take_along_axis(softmax, indices, axis=-1)

        if finished_optional is not None:
            finished_optional_np = finished_optional.cpu().numpy()
            finished_optional_np = finished_optional_np.reshape(
                finished_optional_np.shape[0], 1
            )
            finished_optional_np = np.tile(finished_optional_np, (1, k))
            indices = np.where(finished_optional_np, num_expert, indices)

        source_row_out = (
            np.arange(out.shape[0] * out.shape[1], dtype=np.int32)
            .reshape([out.shape[1], out.shape[0]])
            .transpose(1, 0)
        )

        return (
            torch.from_numpy(out).to(gating.device, dtype=gating.dtype),
            torch.from_numpy(indices).to(gating.device, dtype=torch.int32),
            torch.from_numpy(source_row_out).to(gating.device).contiguous(),
        )
