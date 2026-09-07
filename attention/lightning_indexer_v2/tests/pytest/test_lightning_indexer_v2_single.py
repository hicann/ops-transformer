#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib
import itertools
import os
import torch
import torch_npu
import result_compare_method
import lightning_indexer_v2_golden
import pytest
import lightning_indexer_v2_acl_graph

paramset = os.environ.get("LIV2_PARAMSET", "default")
if paramset not in ("default", "stc"):
    raise ValueError(f"unsupported paramset: {paramset}")
paramset_module = (
    "test_lightning_indexer_v2_stc"
    if paramset == "stc"
    else "test_lightning_indexer_v2_paramset"
)
paramset_data = importlib.import_module(paramset_module)
ENABLED_PARAMS = paramset_data.ENABLED_PARAMS
ENABLED_PARAMSETS = getattr(
    paramset_data,
    "ENABLED_PARAMSETS",
    [
        (f"default_{index:03d}", params)
        for index, params in enumerate(ENABLED_PARAMS, start=1)
    ],
)
requested_names = {
    name.strip()
    for name in os.environ.get("LIV2_CASE_NAMES", "").split(",")
    if name.strip()
}
matched_names = set()
test_created = False

for paramset_index, (paramset_name, params) in enumerate(ENABLED_PARAMSETS):
    # 将params的所有字段注册为局部变量
    for key, value in params.items():
        locals()[f"param_{key}"] = value
    # 生成所有参数组合
    param_names = [
        "batch_size",
        "q_seq",
        "k_seq",
        "q_t_size",
        "k_t_size",
        "q_head_num",
        "k_head_num",
        "head_dim",
        "block_size",
        "block_num",
        "qk_dtype",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "seqused_q",
        "seqused_k",
        "cmp_residual_k",
        "output_idx_offset",
        "layout_q",
        "layout_k",
        "topk",
        "mask_mode",
        "query_datarange",
        "key_datarange",
        "weights_datarange",
        "cmp_ratio",
        "return_value",
        "max_seqlen_q",
    ]
    param_values = [
        locals()["param_batch_size"],
        locals()["param_q_seq"],
        locals()["param_k_seq"],
        locals()["param_q_t_size"],
        locals()["param_k_t_size"],
        locals()["param_q_head_num"],
        locals()["param_k_head_num"],
        locals()["param_head_dim"],
        locals()["param_block_size"],
        locals()["param_block_num"],
        locals()["param_qk_dtype"],
        locals()["param_cu_seqlens_q"],
        locals()["param_cu_seqlens_k"],
        locals()["param_seqused_q"],
        locals()["param_seqused_k"],
        locals()["param_cmp_residual_k"],
        locals()["param_output_idx_offset"],
        locals()["param_layout_q"],
        locals()["param_layout_k"],
        locals()["param_topk"],
        locals()["param_mask_mode"],
        locals()["param_query_datarange"],
        locals()["param_key_datarange"],
        locals()["param_weights_datarange"],
        locals()["param_cmp_ratio"],
        locals()["param_return_value"],
        locals()["param_max_seqlen_q"],
    ]

    # 生成所有的组合，并转换为字典列表
    combinations = list(itertools.product(*param_values))
    locals()["param_combinations"] = []
    for combo_index, combo in enumerate(combinations, start=1):
        case_name = (
            paramset_name
            if len(combinations) == 1
            else f"{paramset_name}_{combo_index:03d}"
        )
        if requested_names and not (
            paramset_name in requested_names or case_name in requested_names
        ):
            continue
        matched_names.update((paramset_name, case_name))
        param_dict = dict(zip(param_names, combo))
        locals()["param_combinations"].append(pytest.param(param_dict, id=case_name))
    if not locals()["param_combinations"]:
        continue

    @pytest.mark.ci
    @pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
    def test_liv2(param_combinations):  # 初始化参数和tensor
        batch_size = param_combinations["batch_size"]
        q_seq = param_combinations["q_seq"]
        k_seq = param_combinations["k_seq"]
        q_t_size = param_combinations["q_t_size"]
        k_t_size = param_combinations["k_t_size"]
        q_head_num = param_combinations["q_head_num"]
        k_head_num = param_combinations["k_head_num"]
        head_dim = param_combinations["head_dim"]
        block_size = param_combinations["block_size"]
        block_num = param_combinations["block_num"]
        qk_dtype = param_combinations["qk_dtype"]
        cu_seqlens_q = param_combinations["cu_seqlens_q"]
        cu_seqlens_k = param_combinations["cu_seqlens_k"]
        seqused_q = param_combinations["seqused_q"]
        seqused_k = param_combinations["seqused_k"]
        cmp_residual_k = param_combinations["cmp_residual_k"]
        output_idx_offset = param_combinations["output_idx_offset"]
        layout_query = param_combinations["layout_q"]
        layout_key = param_combinations["layout_k"]
        sparse_count = param_combinations["topk"]
        sparse_mode = param_combinations["mask_mode"]
        query_datarange = param_combinations["query_datarange"]
        key_datarange = param_combinations["key_datarange"]
        weights_datarange = param_combinations["weights_datarange"]
        cmp_ratio = param_combinations["cmp_ratio"]
        return_value = param_combinations["return_value"]
        max_seqlen_q = param_combinations["max_seqlen_q"]
        torch_npu.npu.set_device(0)
        test_data = (
            batch_size,
            q_seq,
            k_seq,
            q_t_size,
            k_t_size,
            q_head_num,
            k_head_num,
            head_dim,
            block_size,
            block_num,
            qk_dtype,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            output_idx_offset,
            layout_query,
            layout_key,
            sparse_count,
            sparse_mode,
            query_datarange,
            key_datarange,
            weights_datarange,
            cmp_ratio,
            return_value,
            max_seqlen_q,
        )

        # 获得cpu结果(真值)和算子结果（测试值）
        run_mode = os.environ.get("LIV2_RUN_MODE", "both")
        if run_mode not in ("eager", "graph", "both"):
            raise ValueError(f"unsupported run_mode: {run_mode}")
        if run_mode != "graph":
            cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value = (
                lightning_indexer_v2_golden.liv2_output_single(test_data)
            )
        if run_mode != "eager":
            cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value = (
                lightning_indexer_v2_acl_graph.liv2_output_acl_graph(test_data)
            )
        print("npu_result", npu_result)
        print("cpu_result:", cpu_result)
        # 结果精度对比
        result, fulfill_percent = result_compare_method.check_result(
            cpu_result,
            npu_result,
            topk_value,
            output_idx_offset,
            test_data,
            cpu_topk_value,
            npu_topk_value,
        )
        print("result", result)
        print("result", fulfill_percent)
        assert result == "Pass"

        if return_value:
            result_return_value, fulfill_precent_return_value = (
                result_compare_method.check_result_return_value(
                    cpu_topk_value, npu_topk_value, test_data
                )
            )
            print(f"result_return_value: {result_return_value}")
            print(f"result_return_value: {fulfill_precent_return_value}")
            assert result_return_value == "Pass"

    globals()[f"test_liv2_{paramset_index:03d}"] = test_liv2
    test_created = True

unknown_names = sorted(requested_names - matched_names)
if unknown_names:
    raise ValueError(f"unknown case(s) in {paramset_module}: {unknown_names}")

if test_created:
    del test_liv2
