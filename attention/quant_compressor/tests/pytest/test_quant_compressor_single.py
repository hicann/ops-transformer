# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import itertools
import torch_npu

# ******入参调用
from test_quant_compressor_paramset import ENABLED_PARAMS

# ******CPU侧算子逻辑实现获取golden与npu算子直调结果
import quant_compressor_operator_single
import pytest

locals()["param_combinations"] = []

for _, params in enumerate(ENABLED_PARAMS):
    # 将params的所有字段注册为局部变量
    for key, value in params.items():
        locals()[f"param_{key}"] = value

    # 生成所有参数组合
    param_names = [
        "batch_size",
        "hidden_size",
        "Seq_len",
        "head_dim",
        "block_size",
        "cmp_ratio",
        "coff",
        "start_p",
        "cache_mode",
        "layout_x",
        "data_type",
        "cu_seqlens",
        "seqused",
        "start_pos",
        "x_datarange",
        "wkv_datarange",
        "wgate_datarange",
        "ape_datarange",
        "kv_state_datarange",
        "score_state_datarange",
        "x_descale_datarange",
        "wkv_descale_datarange",
        "wgate_descale_datarange",
        "quant_mode",
    ]

    param_values = [
        locals()["param_batch_size"],
        locals()["param_hidden_size"],
        locals()["param_Seq_len"],
        locals()["param_head_dim"],
        locals()["param_block_size"],
        locals()["param_cmp_ratio"],
        locals()["param_coff"],
        locals()["param_start_p"],
        locals()["param_cache_mode"],
        locals()["param_layout_x"],
        locals()["param_data_type"],
        locals()["param_cu_seqlens"],
        locals()["param_seqused"],
        locals()["param_start_pos"],
        locals()["param_x_datarange"],
        locals()["param_wkv_datarange"],
        locals()["param_wgate_datarange"],
        locals()["param_ape_datarange"],
        locals()["param_kv_state_datarange"],
        locals()["param_score_state_datarange"],
        locals()["param_x_descale_datarange"],
        locals()["param_wkv_descale_datarange"],
        locals()["param_wgate_descale_datarange"],
        locals()["param_quant_mode"],
    ]

    # 生成所有的组合，并转换为字典列表
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        locals()["param_combinations"].append(param_dict)

    @pytest.mark.ci
    @pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
    def test_compressor(param_combinations):  # 初始化参数和tensor
        batch_size = param_combinations["batch_size"]
        hidden_size = param_combinations["hidden_size"]
        Seq_len = param_combinations["Seq_len"]
        head_dim = param_combinations["head_dim"]
        block_size = param_combinations["block_size"]
        cmp_ratio = param_combinations["cmp_ratio"]
        coff = param_combinations["coff"]
        start_p = param_combinations["start_p"]
        cache_mode = param_combinations["cache_mode"]
        layout_x = param_combinations["layout_x"]
        data_type = param_combinations["data_type"]
        cu_seqlens = param_combinations["cu_seqlens"]
        seqused = param_combinations["seqused"]
        start_pos = param_combinations["start_pos"]
        x_datarange = param_combinations["x_datarange"]
        wkv_datarange = param_combinations["wkv_datarange"]
        wgate_datarange = param_combinations["wgate_datarange"]
        ape_datarange = param_combinations["ape_datarange"]
        kv_state_datarange = param_combinations["kv_state_datarange"]
        score_state_datarange = param_combinations["score_state_datarange"]
        x_descale_datarange = param_combinations["x_descale_datarange"]
        wkv_descale_datarange = param_combinations["wkv_descale_datarange"]
        wgate_descale_datarange = param_combinations["wgate_descale_datarange"]
        quant_mode = param_combinations["quant_mode"]

        test_data = (
            batch_size,
            hidden_size,
            Seq_len,
            head_dim,
            block_size,
            cmp_ratio,
            coff,
            start_p,
            cache_mode,
            layout_x,
            data_type,
            cu_seqlens,
            seqused,
            start_pos,
            x_datarange,
            wkv_datarange,
            wgate_datarange,
            ape_datarange,
            kv_state_datarange,
            score_state_datarange,
            x_descale_datarange,
            wkv_descale_datarange,
            wgate_descale_datarange,
            quant_mode,
        )

        torch_npu.npu.set_device(0)

        # 获得cpu结果(真值)和算子结果（测试值）
        quant_compressor_operator_single.output_operator(test_data)
