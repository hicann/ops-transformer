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
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor, nnopbase, AclFormat


@register("ascend_attention_update")
class AclnnAttentionUpdateApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            all_lse = input_data.kwargs["lse"]
            all_out = input_data.kwargs["localOut"]
            all_lse = torch.stack(all_lse)
            all_out = torch.stack(all_out)

            dtype = all_out.dtype
            if dtype != torch.float32:
                all_out = all_out.to(torch.float32)

            sp = all_out.shape[0]
            hd = all_out.shape[-1]

            total_length = all_out.shape[1]

            lse_max, _ = torch.max(all_lse, dim=0)
            lse_max_expand = lse_max.unsqueeze(0)
            lse_sub = all_lse - lse_max_expand
            lse_sub_exp = torch.exp(lse_sub)
            lse_sub_exp_sum = torch.sum(lse_sub_exp, dim=0)
            lse_out = lse_max + torch.log(lse_sub_exp_sum)

            lse_out_expand = lse_out.unsqueeze(0)
            lse_out_expand = all_lse - lse_out_expand

            lse_out_expand = lse_out_expand.unsqueeze(2)
            lse_out_expand = lse_out_expand.expand(sp, total_length, hd)
            lse_out_expand = torch.exp(lse_out_expand)

            prod_per_sp = all_out * lse_out_expand
            all_out = torch.sum(prod_per_sp, dim=0)

            if input_data.kwargs["updateType"] == 0:
                lse_out = torch.ones(0)
            if dtype != torch.float32:
                all_out = all_out.to(dtype)

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

        if self.input.kwargs["updateType"] == 0:
            input_args[4] = null_tensor_ptr
            output_packages.pop()
        else:
            total_length = input_data.kwargs["lse"][0].shape[0]
            lse_out_tensor = torch.randn(total_length, dtype=torch.float32).npu()
            acl_tensor = nnopbase.create_acl_tensor(
                lse_out_tensor, AclFormat.ACL_FORMAT_ND
            )
            output_packages[1] = acl_tensor
            input_args[4] = acl_tensor

        return input_args, output_packages

    def after_call(self, output_packages):
        if self.input.kwargs["updateType"] == 0:
            all_out = super().after_call(output_packages)
            out = all_out[0]
            lse_out = torch.ones(0)
        else:
            all_out = super().after_call(output_packages)
            out = all_out[0]
            lse_out = all_out[1]

        return out, lse_out

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnAttentionUpdateGetWorkspaceSize(const aclTensorList* lse, const aclTensorList* localOut, int64_t updateType, aclTensor* out, aclTensor* lseOut, uint64_t* workspaceSize, aclOpExecutor** executor)"
