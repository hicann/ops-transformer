#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import itertools
import torch
import torch_npu
from testcases import ENABLED_PARAMS
from testcases import GRAPH_MODE_PARAMS
import check_valid_param
import gqa_no_quant_bnsd_bsnd
import gqa_no_quant_bnsd_bsnd_ge
import pytest

DEBUG_ON = 0

for _, params in enumerate(ENABLED_PARAMS):
    # 将params的所有字段注册为局部变量
    for key, value in params.items():
        locals()[f"param_{key}"] = value

    # 生成所有参数组合
    param_names = [
        "batch_size",
        "q_head_num",
        "kv_head_num",
        "q_seq",
        "kv_seq",
        "head_dim",
        "dtype",
        "in_layout",
        "block_size",
        "cache_layout",
    ]

    param_values = [
        locals()["param_batch_size"],
        locals()["param_q_head_num"],
        locals()["param_kv_head_num"],
        locals()["param_q_seq"],
        locals()["param_kv_seq"],
        locals()["param_head_dim"],
        locals()["param_dtype"],
        locals()["param_in_layout"],
        locals()["param_block_size"],
        locals()["param_cache_layout"],
    ]

    # 生成所有的组合，并转换为字典列表
    locals()["param_combinations"] = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        locals()["param_combinations"].append(param_dict)

    @pytest.mark.ci
    @pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
    def test_flash_attention(param_combinations):  # 初始化参数和tensor
        batch_size = param_combinations["batch_size"]
        q_head_num = param_combinations["q_head_num"]
        kv_head_num = param_combinations["kv_head_num"]
        q_seq = param_combinations["q_seq"]
        kv_seq = param_combinations["kv_seq"]
        head_dim = param_combinations["head_dim"]
        dtype = param_combinations["dtype"]
        in_layout = param_combinations["in_layout"]
        block_size = param_combinations["block_size"]
        cache_layout = param_combinations["cache_layout"]

        act_seq_len = [q_seq] * batch_size
        act_seq_len_kv = torch.randint(
            low=1, high=kv_seq + 1, size=(batch_size,), dtype=torch.int32
        )

        torch_npu.npu.set_device(0)

        scaled_value = 1 / (head_dim**0.5)

        try:
            test_data = (
                batch_size,
                q_head_num,
                kv_head_num,
                q_seq,
                kv_seq,
                head_dim,
                dtype,
                in_layout,
            )
            check_valid_param.validate_config(test_data)
        except ValueError as e:
            pytest.skip(f"参数校验失败: {e}")
        if in_layout == "BNSD" or in_layout == "BSND":
            test_data = (
                batch_size,
                q_head_num,
                kv_head_num,
                q_seq,
                kv_seq,
                head_dim,
                dtype,
                in_layout,
                act_seq_len,
                act_seq_len_kv,
                scaled_value,
                block_size,
                cache_layout,
            )
            expect, result = gqa_no_quant_bnsd_bsnd.test_gqa_no_quant(test_data)

        check_valid_param.check_result(expect, result)

    @pytest.mark.graph
    @pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
    def test_flash_attention_ge(param_combinations):  # 初始化参数和tensor
        batch_size = param_combinations["batch_size"]
        q_head_num = param_combinations["q_head_num"]
        kv_head_num = param_combinations["kv_head_num"]
        q_seq = param_combinations["q_seq"]
        kv_seq = param_combinations["kv_seq"]
        head_dim = param_combinations["head_dim"]
        dtype = param_combinations["dtype"]
        in_layout = param_combinations["in_layout"]
        block_size = param_combinations["block_size"]
        cache_layout = param_combinations["cache_layout"]

        act_seq_len = [q_seq] * batch_size
        act_seq_len_kv = torch.randint(
            low=1, high=kv_seq + 1, size=(batch_size,), dtype=torch.int32
        )
        if isinstance(act_seq_len, torch.Tensor):
            act_seq_len = act_seq_len.cpu().tolist()
        if isinstance(act_seq_len_kv, torch.Tensor):
            act_seq_len_kv = act_seq_len_kv.cpu().tolist()

        torch_npu.npu.set_device(0)

        scaled_value = 1 / (head_dim**0.5)

        try:
            test_data = (
                batch_size,
                q_head_num,
                kv_head_num,
                q_seq,
                kv_seq,
                head_dim,
                dtype,
                in_layout,
            )
            check_valid_param.validate_config(test_data)
        except ValueError as e:
            pytest.skip(f"参数校验失败: {e}")
        if in_layout == "BNSD" or in_layout == "BSND":
            test_data = (
                batch_size,
                q_head_num,
                kv_head_num,
                q_seq,
                kv_seq,
                head_dim,
                dtype,
                in_layout,
                act_seq_len,
                act_seq_len_kv,
                scaled_value,
                block_size,
                cache_layout,
            )
            expect, result = gqa_no_quant_bnsd_bsnd_ge.test_gqa_no_quant_ge(test_data)

        check_valid_param.check_result(expect, result)
