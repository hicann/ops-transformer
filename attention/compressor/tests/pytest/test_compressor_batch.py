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
from batch import compressor_pt_loadprocess
import pytest
import random
import pandas as pd
from pathlib import Path
import numpy as np
import math
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from compressor_golden import check_result
import concurrent.futures

TEST_INPUT_PATH = "./pt_path"
pt_dir = TEST_INPUT_PATH
result_path = Path("result.xlsx")  # 或使用传入的result_path

# 支持通过环境变量 COMPRESSOR_TESTCASE_PATH 指定单条用例文件，实现进程级隔离执行：
#   - 设置时：仅运行该条用例（配合 batch_isolated_run.sh 每条用例拉起独立进程）
#   - 未设置：回退为原有行为，一次性加载目录下全部用例
_single_case_path = os.environ.get("COMPRESSOR_TESTCASE_PATH", "").strip()
_is_isolated_mode = bool(_single_case_path)

excel_path = os.environ.get("TEST_CASE_EXCEL", "")
allowed_names = set()
if excel_path:
    try:
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path, sheet_name=0)
            df = df.replace({np.nan: None, pd.NA: None})
            if "Testcase_Name" in df.columns:
                allowed_names = set(df["Testcase_Name"].dropna().astype(str))
                print(f"从Excel读取了 {len(allowed_names)} 个用例名: {excel_path}")
            else:
                print("警告: Excel中没有Testcase_Name列，将运行所有pt文件")
        else:
            print(f"警告: 未找到Excel文件: {excel_path}，将运行所有pt文件")
    except Exception as e:
        print(f"警告: 读取Excel失败 ({e})，将运行所有pt文件")
else:
    print(f"未设置TEST_CASE_EXCEL，将运行 {pt_dir} 下所有pt文件")

# 生成所有的组合，并转换为字典列表
locals()["testcase_files"] = []
if _single_case_path:
    if not os.path.isfile(_single_case_path):
        print(
            f"错误: 环境变量 COMPRESSOR_TESTCASE_PATH 指定的用例文件不存在: {_single_case_path}"
        )
    else:
        print(f"单用例隔离模式, 仅执行: {_single_case_path}")
        locals()["testcase_files"].append(_single_case_path)
elif os.path.isdir(pt_dir):
    pt_files = [f for f in os.listdir(pt_dir) if f.endswith(".pt")]
    if not pt_files:
        print(f"错误: 目录中没有找到.pt文件: {pt_dir}")
    else:
        if allowed_names:
            pt_files = [f for f in pt_files if Path(f).stem in allowed_names]
        print(f"找到 {len(pt_files)} 个测试用例文件")
        for pt_file in pt_files:
            filepath = os.path.join(pt_dir, pt_file)
            locals()["testcase_files"].append(filepath)
else:
    print(f"错误: 输出目录不存在: {pt_dir}")


def compressor(testcase_files):  # 初始化参数和tensor
    (
        cpu_result,
        kv_mask_result,
        npu_result,
        cpu_kv_state,
        mask_cpu_kv_state,
        cpu_score_state,
        mask_cpu_score_state,
        state_cache,
        params,
    ) = compressor_pt_loadprocess.test_compressor_process(testcase_files, device_id=0)
    if npu_result is not None:
        cpu_kv_state_update = cpu_kv_state[mask_cpu_kv_state]
        cpu_kv_state_origin = cpu_kv_state[~mask_cpu_kv_state]
        cpu_score_state_update = cpu_score_state[mask_cpu_score_state]
        cpu_score_state_origin = cpu_score_state[~mask_cpu_score_state]
        data_type = str(npu_result.dtype).replace("torch.", "")
        print(
            "--------------------------------------------------------------check result-------------------------------------------------------------"
        )
        result_percent, result = check_result(
            cpu_result[kv_mask_result].to(torch.float32),
            npu_result.cpu()[kv_mask_result].to(torch.float32),
            data_type,
        )
        print(
            "--------------------------------------------------------------check kv state update-------------------------------------------------------------"
        )
        kv_state_result_percent, kv_state_result = check_result(
            cpu_kv_state_update.to(torch.float32),
            state_cache.cpu()[:, :, : state_cache.shape[2] // 2][mask_cpu_kv_state].to(
                torch.float32
            ),
            data_type,
        )
        print(
            "--------------------------------------------------------------check score state update-------------------------------------------------------------"
        )
        score_state_result_percent, score_state_result = check_result(
            cpu_score_state_update.to(torch.float32),
            state_cache.cpu()[:, :, state_cache.shape[2] // 2 :][
                mask_cpu_score_state
            ].to(torch.float32),
            data_type,
        )
        print(
            "--------------------------------------------------------------check kv state origin-------------------------------------------------------------"
        )
        kv_state_origin_result_percent, kv_state_origin_result = check_result(
            cpu_kv_state_origin.to(torch.float32),
            state_cache.cpu()[:, :, : state_cache.shape[2] // 2][~mask_cpu_kv_state].to(
                torch.float32
            ),
            data_type,
            0.0,
        )
        print(
            "--------------------------------------------------------------check score state origin-------------------------------------------------------------"
        )
        score_state_origin_result_percent, score_state_origin_result = check_result(
            cpu_score_state_origin.to(torch.float32),
            state_cache.cpu()[:, :, state_cache.shape[2] // 2 :][
                ~mask_cpu_score_state
            ].to(torch.float32),
            data_type,
            0.0,
        )
    else:
        result = "Failed"
        result_percent = 0
        kv_state_result = "Failed"
        kv_state_result_percent = 0
        score_state_result = "Failed"
        score_state_result_percent = 0
        kv_state_origin_result = "Failed"
        kv_state_origin_result_percent = 0
        score_state_origin_result = "Failed"
        score_state_origin_result_percent = 0

    row_data = {
        "Testcase_Name": Path(testcase_files).stem,
        "batch_size": params[0],
        "hidden_size": params[1],
        "Seq_len": params[2],
        "head_dim": params[3],
        "block_size": params[4],
        "cmp_ratio": params[5],
        "coff": params[6],
        "start_p": params[7],
        "cache_mode": params[8],
        "layout_x": params[9],
        "data_type": params[10],
        "cu_seqlens": params[11],
        "seqused": params[12],
        "start_pos": params[13],
        "x_datarange": params[14],
        "wkv_datarange": params[15],
        "wgate_datarange": params[16],
        "ape_datarange": params[17],
        "kv_state_datarange": params[18],
        "score_state_datarange": params[19],
        "result": result,
        "result_percent": result_percent,
        "kv_state_update": kv_state_result,
        "kv_state_update_percent": kv_state_result_percent,
        "score_state_update": score_state_result,
        "score_state_update_percent": score_state_result_percent,
        "kv_state_origin": kv_state_origin_result,
        "kv_state_origin_percent": kv_state_origin_result_percent,
        "score_state_origin": score_state_origin_result,
        "score_state_origin_percent": score_state_origin_result_percent,
    }

    # 检查文件是否存在
    if result_path.exists():
        # 读取现有数据
        df = pd.read_excel(result_path)

        # 检查列名是否一致
        if set(df.columns) != set(row_data.keys()):
            print("警告：变量名与Excel列名不匹配！")
            print(f"Excel列名: {list(df.columns)}")
            print(f"变量名: {list(row_data.keys())}")
            print("请检查变量名或Excel文件")
            return False

        # 追加新行
        new_df = pd.DataFrame([row_data])
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        # 文件不存在，创建新的DataFrame
        df = pd.DataFrame([row_data])

    # 保存到Excel
    df.to_excel(result_path, index=False)


@pytest.mark.ci
@pytest.mark.parametrize("testcase_files", locals()["testcase_files"])
def test_compressor(testcase_files):  # 初始化参数和tensor
    if _is_isolated_mode:
        # 批量隔离模式：shell 层已通过独立 pytest 进程提供进程隔离，内部使用线程池即可
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = executor.submit(compressor, testcase_files)
            for future in concurrent.futures.as_completed([futures]):
                try:
                    result = future.result()
                except Exception as e:
                    pytest.fail(f"当前用例线程执行失败：{e}")
    else:
        # 非隔离模式（直接 pytest）：使用子进程隔离，防止单条用例崩溃影响整体
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future1 = executor.submit(compressor, testcase_files)
            for future in concurrent.futures.as_completed([future1]):
                try:
                    result = future.result()
                except Exception as e:
                    pytest.fail(f"当前用例子进程执行失败：{e}")
