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

import random
import torch
import ctypes

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor


@register("ascend_attention_update")
class AclnnAttentionUpdateApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            all_lse = input_data.kwargs["lse"]
            all_out = input_data.kwargs["localOut"]
            all_lse = torch.stack(all_lse)
            all_out = torch.stack(all_out)
            sp = all_out.shape[0]
            hd = all_out.shape[-1]

            all_lse = all_lse.transpose(0, 1)
            all_out = all_out.permute(1, 2, 0).reshape(-1, sp * hd)

            # (b * s * hc, sp)
            lse_exp = torch.exp(all_lse)
            # (b * s * hc, 1)
            sum_lse_exp = torch.sum(lse_exp, dim=-1, keepdim=True)
            # (b * s * hc, sp)
            sum_lse_exp = sum_lse_exp.repeat(1, sp)
            lse_exp = lse_exp / sum_lse_exp

            # oi = lse_exp*oi (b * s * hc, hd, sp)*(b*s*hc, hd, sp)
            lse_exp = lse_exp.unsqueeze(1)
            lse_exp = lse_exp.repeat(1, hd, 1)
            all_out = all_out.reshape(-1, hd, sp)
            all_out = all_out * lse_exp

            # o = sum(oi) (b*s*hc, hd)
            all_out = torch.sum(all_out, dim=-1)
            lse_out = torch.ones(0)
            return all_out, lse_out


@register("ascend_attention_update_npu")
class AclnnAttentionUpdateApiNPU(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(AclnnAttentionUpdateApiNPU, self).__init__(task_result, backend)
        self.input = None

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        self.input = input_data

        acl_tensor_ptr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, acl_tensor_ptr)
        input_args[4] = null_tensor_ptr
        output_packages.pop()

        return input_args, output_packages

    def after_call(self, output_packages):
        all_out = super().after_call(output_packages)

        if self.input.kwargs["updateType"] == 0:
            lse_out = torch.ones(0)

        return all_out, lse_out

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnAttentionUpdateGetWorkspaceSize(const aclTensorList* lse, const aclTensorList* localOut, int64_t updateType, aclTensor* out, aclTensor* lseOut, uint64_t* workspaceSize, aclOpExecutor** executor)"
