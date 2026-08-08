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

import os
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch_npu")

from batch import stem_indexer_pt_loadprocess
import result_compare_method


TEST_INPUT_PATH = os.environ.get("STEM_INDEXER_PT_DIR", "./pt_path")
PT_DIR = TEST_INPUT_PATH
RESULT_PATH = Path(os.environ.get("STEM_INDEXER_RESULT_PATH", "result.csv"))
MAX_RESULT_DETAIL_LEN = 2048
RESULT_COLUMNS = ["case_id", "testcase_name", "expected_result", "result", "detail"]

# 通过环境变量 STEM_INDEXER_MODE 切换执行模式：eager（默认）或 graph
# 例: STEM_INDEXER_MODE=graph python -m pytest test_stem_indexer_batch.py -m graph
EXEC_MODE = os.environ.get("STEM_INDEXER_MODE", "eager").strip().lower()
if EXEC_MODE not in ("eager", "graph"):
    raise ValueError(
        f"Unsupported STEM_INDEXER_MODE: {EXEC_MODE!r}. "
        "Expected 'eager' or 'graph'."
    )
_IS_GRAPH_MODE = EXEC_MODE == "graph"

# 支持通过环境变量 STEM_INDEXER_CASE_ID 指定只跑特定 case_id（逗号分隔多个）
# 例: STEM_INDEXER_CASE_ID=SI_WB_002 python -m pytest test_stem_indexer_batch.py
_FILTER_IDS_RAW = os.environ.get("STEM_INDEXER_CASE_ID", "").strip()
_FILTER_IDS = set(x.strip() for x in _FILTER_IDS_RAW.split(",") if x.strip())


def collect_testcase_files(pt_dir):
    if not os.path.isdir(pt_dir):
        print(f"StemIndexer pt directory does not exist: {pt_dir}")
        return []
    testcase_files = [
        os.path.join(pt_dir, pt_file)
        for pt_file in sorted(os.listdir(pt_dir))
        if pt_file.endswith(".pt")
    ]
    if _FILTER_IDS:
        filtered = []
        for fp in testcase_files:
            try:
                case = stem_indexer_pt_loadprocess.torch_load_cpu(fp)["case"]
            except Exception:
                continue
            if case.get("case_id", "") in _FILTER_IDS:
                filtered.append(fp)
        print(
            f"Filter by STEM_INDEXER_CASE_ID={_FILTER_IDS_RAW}: "
            f"{len(filtered)}/{len(testcase_files)} matched."
        )
        testcase_files = filtered
    print(f"Found {len(testcase_files)} StemIndexer pt testcase files.")
    return testcase_files


TESTCASE_FILES = collect_testcase_files(PT_DIR)


def append_result(case, result, detail=""):
    row_data = {
        "case_id": case["case_id"],
        "testcase_name": case["testcase_name"],
        "expected_result": case["expected_result"],
        "result": result,
        "detail": detail,
    }
    if RESULT_PATH.exists():
        df = pd.read_csv(RESULT_PATH, encoding="utf-8-sig")
        if list(df.columns) != RESULT_COLUMNS:
            print(
                f"Warning: {RESULT_PATH} columns mismatch, skip appending StemIndexer result."
            )
            print(f"Existing columns: {list(df.columns)}")
            print(f"Expected columns: {RESULT_COLUMNS}")
            return False
        df = pd.concat(
            [df, pd.DataFrame([row_data], columns=RESULT_COLUMNS)], ignore_index=True
        )
    else:
        df = pd.DataFrame([row_data], columns=RESULT_COLUMNS)
    df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    return True


def format_error_detail(err):
    detail = f"{type(err).__name__}: {err}"
    if len(detail) > MAX_RESULT_DETAIL_LEN:
        return detail[:MAX_RESULT_DETAIL_LEN] + "...[truncated]"
    return detail


def load_case(filepath):
    return stem_indexer_pt_loadprocess.torch_load_cpu(filepath)["case"]


@pytest.mark.ci
@pytest.mark.graph
@pytest.mark.parametrize("testcase_file", TESTCASE_FILES)
def test_stem_indexer_batch(testcase_file):
    case = load_case(testcase_file)

    if case["testcase_name"] == "invalid_sparse_indices_shape":
        pytest.skip(
            "Torch custom op API does not expose output tensor shape injection."
        )

    if case["expected_result"] == "FAIL":
        if _IS_GRAPH_MODE:
            pytest.skip("Graph mode does not support FAIL test cases.")
        try:
            with pytest.raises(Exception):
                stem_indexer_pt_loadprocess.stem_indexer_process(
                    testcase_file, device_id=0, mode=EXEC_MODE
                )
        except pytest.fail.Exception as err:
            append_result(case, "FAIL", format_error_detail(err))
            raise
        append_result(case, "PASS", "Expected failure was raised.")
        return

    try:
        expected_indices, expected_seq_len, npu_result, case, test_data = (
            stem_indexer_pt_loadprocess.stem_indexer_process(
                testcase_file, device_id=0, return_test_data=True, mode=EXEC_MODE
            )
        )
        result_compare_method.assert_stem_indexer_result(
            expected_indices, expected_seq_len, npu_result, case, test_data
        )
        append_result(case, "PASS")
    except Exception as err:
        append_result(case, "FAIL", format_error_detail(err))
        raise
