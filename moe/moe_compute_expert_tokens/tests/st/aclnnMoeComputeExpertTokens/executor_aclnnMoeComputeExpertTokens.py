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
from atk.configs.dataset_config import InputDataset

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

import os

os.environ["PYTORCH_NO_NPU_MEMORY_CACHING"] = "1"


@register("function_aclnn_moe_compute_expert_tokens")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        sorted_experts = input_data.kwargs.get("sortedExperts").cpu().numpy()
        num_experts = input_data.kwargs.get("numExperts")
        arr_length = sorted_experts.shape[-1]
        res = np.arange(num_experts)
        for i in range(num_experts):
            target = i
            low = 0
            high = arr_length - 1
            target_location = -1
            while low <= high:
                mid = (low + high) // 2
                if sorted_experts[mid] > target:
                    high = mid - 1
                else:
                    low = mid + 1
                    target_location = mid
            res[i] = target_location + 1
        res = res.astype(np.int32)
        return torch.from_numpy(res)

    def init_by_input_data(self, input_data: InputDataset):
        sorted_experts = input_data.kwargs["sortedExperts"]

        input_data.kwargs["sortedExperts"] = torch.sort(sorted_experts, stable=True)[0]

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnMoeComputeExpertTokensGetWorkspaceSize(const aclTensor* sortedExperts, int64_t numExperts, const aclTensor* out, uint64_t *workspaceSize, aclOpExecutor **executor)"
