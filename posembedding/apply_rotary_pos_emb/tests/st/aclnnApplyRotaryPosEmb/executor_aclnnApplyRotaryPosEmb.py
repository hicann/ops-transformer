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
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


@register("ascend_method_apply_rotary_pos_emb")
class MethodApplyRotaryPosEmb(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodApplyRotaryPosEmb, self).__init__(task_result)
        self.session = None

    def single_golden(self, x, cos, sin):
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.float()
            cos = cos.float()
            sin = sin.float()
        x1, x2 = torch.chunk(x, 2, dim=-1)
        x_new = torch.cat((-x2, x1), dim=-1)
        res = x * cos + x_new * sin
        return res.to(dtype)

    def golden(self, q, k, cos, sin):
        q_res = self.single_golden(q, cos, sin)
        k_res = self.single_golden(k, cos, sin)
        return q_res, k_res

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            output = self.golden(
                input_data.kwargs["queryRef"],
                input_data.kwargs["keyRef"],
                input_data.kwargs["cos"],
                input_data.kwargs["sin"],
            )
            return output

    def get_aclnn_func_signature_type(self):
        return "aclnnStatus aclnnApplyRotaryPosEmbGetWorkspaceSize(aclTensor* queryRef, aclTensor* keyRef, const aclTensor* cos, const aclTensor* sin, int64_t layout, uint64_t* workspaceSize, aclOpExecutor** executor)"


@register("aclnn_apply_rotary_pos_emb")
class ApplyRotaryPosEmbAclnnApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        input_args.pop()
        output_packages[:] = [input_args[0], input_args[1]]
        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output
