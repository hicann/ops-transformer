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
import numpy as np

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("ascend_moe_fused_topk")
class AclnnMoeFusedTopkApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            input_dic = input_data.kwargs
            y, indices = self.calc_golden(**input_dic)

            return y, indices

    def calc_golden(
        self,
        x,
        add_num,
        mapping_num,
        mapping_table,
        group_num,
        group_topk,
        top_n,
        top_k,
        activate_type,
        is_norm,
        scale,
        enable_expert_mapping,
    ):
        x = x.to(torch.float32)
        add_num = add_num.to(torch.float32)
        # sigmoid
        m = torch.nn.Sigmoid()
        output_sig = m(x)
        # add
        input0 = torch.add(output_sig, add_num)
        # group_topk
        token_num, expert_num = input0.shape
        group_eles = expert_num // group_num
        input0 = torch.reshape(input0, (token_num, group_num, group_eles))
        output = input0.clone()
        group_tensor = torch.topk(input0, top_n).values
        group_tensor = torch.sum(group_tensor, dim=-1)
        sort_index = torch.from_numpy(np.argsort(-group_tensor.numpy(), kind="stable"))
        cols_to_use = torch.arange(group_topk, group_num, dtype=torch.long)
        row_indices = torch.arange(sort_index.shape[0]).repeat_interleave(
            cols_to_use.shape[0]
        )
        col_indices = sort_index.index_select(1, cols_to_use).view(-1)
        output[row_indices, col_indices] = float(0)
        group_top_k_res = torch.reshape(output, (token_num, expert_num))
        # topk
        sort_res = torch.sort(group_top_k_res, descending=True, stable=True)
        # gather
        gather_res = torch.gather(output_sig, -1, sort_res.indices[:, 0:top_k])
        if is_norm:
            # reduce_sum
            sum_res = torch.sum(gather_res, -1, keepdim=True)
            # div
            res = torch.div(gather_res, sum_res)
            # mul
            res = res * torch.tensor(scale, dtype=torch.float32)
        else:
            res = gather_res

        out_indices = sort_res.indices.to(torch.int32)
        if enable_expert_mapping:
            out_indices_clone = out_indices.clone().detach()
            for bs in range(token_num):
                for ki in range(top_k):
                    expert_id = out_indices_clone[bs][ki]
                    redundantOffset = bs % max(1, mapping_num[expert_id])
                    out_indices[bs][ki] = mapping_table[expert_id][redundantOffset]

        return res, out_indices[:, 0:top_k]
