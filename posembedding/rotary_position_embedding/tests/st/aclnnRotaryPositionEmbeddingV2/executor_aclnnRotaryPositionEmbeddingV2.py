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

import torch
import torch_npu

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from einops import rearrange
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


@register("function_aclnn_rotary_position_embedding_api")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        torch.use_deterministic_algorithms(True)
        input_x = input_data.kwargs["x"]
        cos = input_data.kwargs["cos"]
        sin = input_data.kwargs["sin"]
        mode = int(input_data.kwargs.get("mode"))
        if input_x.shape == torch.Size([]):
            return (
                input_data.kwargs["x"],
                input_data.kwargs["x"],
                input_data.kwargs["x"],
            )

        def rotate_every_two(x):
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return rearrange(
                torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
            )

        def fused_rotary_position_embedding(x, cos, sin):
            return x * cos + rotate_every_two(x) * sin

        def golden_res(x, cos, sin):
            x = x.float()
            cos = cos.float()
            sin = sin.float()
            res = fused_rotary_position_embedding(x, cos, sin)
            return res

        try:
            if self.output is None:
                if mode == 0:
                    x_dtype = input_x.dtype
                    if x_dtype == torch.bfloat16:
                        input_x = input_x.float()
                        cos = cos.float()
                        sin = sin.float()
                    input_x_1, x_r = torch.chunk(input_x, 2, -1)
                    x_new = torch.cat((-x_r, input_x_1), dim=-1)
                    output = input_x * cos + sin * x_new
                    if x_dtype == torch.bfloat16:
                        output = output.to(x_dtype)
                elif mode == 1:
                    output = golden_res(input_x, cos, sin).to(input_x.dtype)
            else:
                if isinstance(self.output, int):
                    output = input_data.args[self.output]
                elif isinstance(self.output, str):
                    output = input_data.kwargs[self.output]
                else:
                    raise ValueError(f"self.output {self.output} value is error")
            return output
        except:
            return output

    def get_format(self, input_data: InputDataset, index=None, name=None):
        """
        :param input_data: 参数列表
        :param index: 参数位置
        :param name: 参数名字
        :return:
        format at this index or name
        """
        from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

        return AclFormat.ACL_FORMAT_ND


@register("aclnn_rotary_position_embedding_api")
class AclnnRotaryPositionEmbedding(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)

        return input_args, output_packages

    def get_format(self, input_data: InputDataset, index=None, name=None):
        """
        :param input_data: 参数列表
        :param index: 参数位置
        :param name: 参数名字
        :return:
        format at this index or name
        """
        from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

        return AclFormat.ACL_FORMAT_ND
