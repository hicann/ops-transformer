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

import itertools
import torch
import check_valid_param
import quant_sparse_flash_mla_golden
import pytest
import os
import utils
import concurrent.futures
import logging

# 读取表格（支持通过环境变量传入）
save_path = os.environ.get("QSMLA_PT_DIR", "qsmla_testcase")
excel_file = os.environ.get("QSMLA_EXCEL", "./excel/example.xlsx")
sheet_name = os.environ.get("QSMLA_SHEET", "decode")
ENABLED_PARAMS = utils.load_excel_test_cases(excel_file, sheet_name)

param_combinations = []
for _, params in enumerate(ENABLED_PARAMS):
    normalized_params = {}
    for key, value in params.items():
        if value == "torch.bfloat16":
            value = torch.bfloat16
        elif value == "torch.float8_e4m3fn":
            value = torch.float8_e4m3fn
        elif value == "torch.uint8":
            value = torch.uint8
        elif value == "FALSE":
            value = False
        elif value == "TRUE":
            value = True
        normalized_params[key] = value
    normalized_params["topk_value_mode"] = params.get("topk_value_mode", 1)
    normalized_params["return_softmax_lse"] = params.get("return_softmax_lse", False)
    isSink_raw = params.get("isSink", True)
    if isinstance(isSink_raw, str):
        isSink_raw = isSink_raw.upper() == "TRUE"
    normalized_params["isSink"] = bool(isSink_raw)
    for key in [
        "seqused_q",
        "cu_seqlens_q",
        "seqused_ori_kv",
        "seqused_cmp_kv",
        "cu_seqlens_ori_kv",
        "cu_seqlens_cmp_kv",
        "cmp_residual_kv",
    ]:
        normalized_params[key] = utils.parse_list_param(params.get(key))
    template_run_mode = normalized_params["template_run_mode"]
    if isinstance(template_run_mode, list):
        template_run_mode = template_run_mode[0]

    param_names = [
        "Testcase_Name",
        "layout_q",
        "layout_kv",
        "q_type",
        "ori_kv_type",
        "cmp_kv_type",
        "B",
        "S1",
        "S2",
        "N1",
        "N2",
        "D",
        "K",
        "block_size1",
        "block_size2",
        "softmax_scale",
        "cmp_ratio",
        "ori_mask_mode",
        "cmp_mask_mode",
        "ori_win_left",
        "ori_win_right",
        "quant_mode",
        "template_run_mode",
        "actlen_mode",
        "S1EQS2",
        "topk_value_mode",
        "return_softmax_lse",
        "isSink",
        "seqused_q",
        "cu_seqlens_q",
        "seqused_ori_kv",
        "seqused_cmp_kv",
        "cu_seqlens_ori_kv",
        "cu_seqlens_cmp_kv",
        "cmp_residual_kv",
    ]

    param_values = [
        [normalized_params["Testcase_Name"]],
        [normalized_params["layout_q"]],
        [normalized_params["layout_kv"]],
        [normalized_params["q_type"]],
        [normalized_params["ori_kv_type"]],
        [normalized_params["cmp_kv_type"]],
        [normalized_params["B"]],
        [normalized_params["S1"]],
        [normalized_params["S2"]],
        [normalized_params["N1"]],
        [normalized_params["N2"]],
        [normalized_params["D"]],
        [normalized_params["K"]],
        [normalized_params["block_size1"]],
        [normalized_params["block_size2"]],
        [normalized_params["softmax_scale"]],
        [normalized_params["cmp_ratio"]],
        [normalized_params["ori_mask_mode"]],
        [normalized_params["cmp_mask_mode"]],
        [normalized_params["ori_win_left"]],
        [normalized_params["ori_win_right"]],
        [normalized_params["quant_mode"]],
        [normalized_params["template_run_mode"]],
        [normalized_params["actlen_mode"]],
        [normalized_params["S1EQS2"]],
        [normalized_params["topk_value_mode"]],
        [normalized_params["return_softmax_lse"]],
        [normalized_params["isSink"]],
        [normalized_params["seqused_q"]],
        [normalized_params["cu_seqlens_q"]],
        [normalized_params["seqused_ori_kv"]],
        [normalized_params["seqused_cmp_kv"]],
        [normalized_params["cu_seqlens_ori_kv"]],
        [normalized_params["cu_seqlens_cmp_kv"]],
        [normalized_params["cmp_residual_kv"]],
    ]

    # 生成所有的组合，并转换为字典列表
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        param_combinations.append(param_dict)
    logging.info(param_combinations)

case_id = 0


def qsmla(param_combinations):
    global case_id
    params = utils.fill_none_params(param_combinations)

    Testcase_Name = params["Testcase_Name"]
    if Testcase_Name is None:
        ops_mode = "prefill" if params["S1"] > 4 else "decode"
        q_type_str = "BF16" if params["q_type"] == torch.bfloat16 else "FP16"
        kv_type_str = "HIF8" if params["ori_kv_type"] == torch.uint8 else "FP8_E4M3FN"
        Testcase_Name = f"quantSparseFlashMla_{params['template_run_mode']}_{ops_mode}_{params['layout_q']}_{q_type_str}_{params['layout_kv']}_{kv_type_str}_{params['B']}_{params['N1']}_{params['N2']}_{params['S1']}_{params['S2']}_{params['D']}_{params['K']}_{case_id:06d}"
        params["Testcase_Name"] = Testcase_Name
    logging.info(f"input_params: {params}")

    # 输入参数的合法性校验
    try:
        check_valid_param.check_valid_param(params)
    except ValueError as e:
        pytest.skip(f"输入参数校验失败:{e}")

    # 生成测试数据
    quant_sparse_flash_mla_golden.generate_and_save_testdata(
        params, save_pt=True, save_path=save_path
    )
    case_id += 1


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_quant_sparse_flash_mla(param_combinations):  # 初始化参数和tensor
    # 线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = executor.submit(qsmla, param_combinations)
        # 等待并获取结果
        for future in concurrent.futures.as_completed([futures]):
            try:
                future.result()
            except Exception:
                pytest.fail("当前用例线程执行失败")
