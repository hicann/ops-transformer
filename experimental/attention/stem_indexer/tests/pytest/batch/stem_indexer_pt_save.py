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

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import multiprocessing
import os
import re
import sys

import pandas as pd
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTEST_DIR = os.path.dirname(CURRENT_DIR)
if PYTEST_DIR not in sys.path:
    sys.path.insert(0, PYTEST_DIR)

import stem_indexer_golden  # noqa: E402


LIST_COLUMNS = {
    "q_seq_lens",
    "kv_seq_lens",
    "num_prompt_tokens",
}
INT_COLUMNS = {
    "batch_size",
    "q_heads",
    "kv_heads",
    "stem_block_size",
    "stem_stride",
    "initial_blocks",
    "window_size",
    "topk_score_precision",
}
REQUIRED_COLUMNS = [
    "case_id",
    "testcase_name",
    "expected_result",
    "batch_size",
    "q_heads",
    "kv_heads",
    "q_seq_lens",
    "kv_seq_lens",
    "num_prompt_tokens",
    "causal",
    "alpha",
    "stem_block_size",
    "stem_stride",
    "initial_blocks",
    "window_size",
    "topk_score_precision",
    "qflat_dtype",
    "kflat_dtype",
    "vbias_dtype",
    "special_setting",
]

# 支持通过环境变量 STEM_INDEXER_CASE_ID 指定只生成特定 case_id（逗号分隔多个）
# 例: STEM_INDEXER_CASE_ID=SI_WB_001_1,SI_WB_101_1 python3 batch/stem_indexer_pt_save.py ...
_FILTER_IDS_RAW = os.environ.get("STEM_INDEXER_CASE_ID", "").strip()
_FILTER_IDS = {item.strip() for item in _FILTER_IDS_RAW.split(",") if item.strip()}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes")


def parse_list(value):
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, int):
        return [value]
    parsed = ast.literal_eval(str(value))
    if isinstance(parsed, int):
        return [parsed]
    return [int(item) for item in parsed]


def normalize_cell(value):
    if pd.isna(value):
        return ""
    return value


def row_to_case(row):
    case = {}
    for column in REQUIRED_COLUMNS:
        value = normalize_cell(row[column])
        if column in LIST_COLUMNS:
            case[column] = parse_list(value)
        elif column in INT_COLUMNS:
            case[column] = int(value)
        elif column == "causal":
            case[column] = parse_bool(value)
        elif column == "alpha":
            case[column] = float(value)
        elif column == "expected_result":
            case[column] = str(value).strip().upper()
        else:
            case[column] = str(value).strip()
    if case["topk_score_precision"] not in (1, 2):
        raise ValueError(
            f"{case['case_id']} topk_score_precision must be 1(uint32) or 2(uint16), "
            f"but got {case['topk_score_precision']}"
        )
    if "description" in row.index:
        case["description"] = str(normalize_cell(row["description"])).strip()
    return case


def find_csv_files(path_pattern):
    paths = sorted(glob.glob(path_pattern))
    if not paths and os.path.isfile(path_pattern):
        paths = [path_pattern]
    return [path for path in paths if path.lower().endswith(".csv")]


def load_csv_test_cases(path_pattern):
    csv_files = find_csv_files(path_pattern)
    if not csv_files:
        raise FileNotFoundError(f"No CSV testcase file found from path: {path_pattern}")

    test_cases = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding="utf-8-sig")
        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"{csv_file} missing columns: {missing_columns}")
        for _, row in df.iterrows():
            if pd.isna(row["case_id"]):
                continue
            test_cases.append(row_to_case(row))
    return test_cases


def select_test_cases(test_cases):
    if not _FILTER_IDS:
        return test_cases

    filtered = [case for case in test_cases if case.get("case_id") in _FILTER_IDS]
    print(
        f"Filter by STEM_INDEXER_CASE_ID={_FILTER_IDS_RAW}: "
        f"{len(filtered)}/{len(test_cases)} matched."
    )
    return filtered


def sanitize_case_name(case):
    name = f"{case['case_id']}_{case['testcase_name']}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def build_pt_payload(case):
    inputs = stem_indexer_golden.build_case_inputs(case)
    expected_indices = None
    expected_seq_len = None
    if case["expected_result"] == "PASS":
        expected_indices, expected_seq_len = stem_indexer_golden.stem_indexer_golden(
            case, inputs
        )

    return {
        "case": case,
        "qflat": inputs["qflat"],
        "kflat": inputs["kflat"],
        "vbias": inputs["vbias"],
        "q_seq_lens": inputs["q_seq_lens"],
        "kv_seq_lens": inputs["kv_seq_lens"],
        "num_prompt_tokens": inputs["num_prompt_tokens"],
        "expected_sparse_indices": expected_indices,
        "expected_sparse_seq_len": expected_seq_len,
    }


def init_worker():
    """Keep each case worker single-threaded to avoid CPU oversubscription."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def generate_and_save_case(case, output_path):
    payload = build_pt_payload(case)
    torch.save(payload, output_path)
    return output_path


def resolve_worker_count(workers, case_count):
    if workers < 0:
        raise ValueError(f"workers must be greater than or equal to 0, but got {workers}")
    available_cpu_count = os.cpu_count() or 1
    requested_worker_count = available_cpu_count if workers == 0 else workers
    return max(1, min(requested_worker_count, max(case_count, 1)))


def save_test_cases(test_cases, output_dir, workers):
    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0
    skipped_count = 0
    pending_cases = []
    pending_paths = set()
    for case in test_cases:
        output_path = os.path.join(output_dir, f"{sanitize_case_name(case)}.pt")
        if os.path.exists(output_path) or output_path in pending_paths:
            skipped_count += 1
            print(f"Skipped existing StemIndexer testcase pt: {output_path}")
            continue
        pending_paths.add(output_path)
        pending_cases.append((case, output_path))

    worker_count = resolve_worker_count(workers, len(pending_cases))
    print(
        f"Generating {len(pending_cases)} StemIndexer testcase pt files "
        f"with {worker_count} worker process(es)."
    )
    if worker_count == 1:
        for case, output_path in pending_cases:
            try:
                print(
                    f"Generating StemIndexer testcase pt for {case.get('case_id', '<unknown>')}: {output_path}"
                )
                generate_and_save_case(case, output_path)
                saved_count += 1
                print(f"Saved StemIndexer testcase pt: {output_path}")
            except Exception as err:
                print(f"[FAILED] Generate pt for {case.get('case_id', '<unknown>')}: {err}")
                raise
    else:
        multiprocessing_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing_context,
            initializer=init_worker,
        ) as executor:
            future_to_case = {}
            for case, output_path in pending_cases:
                print(
                    f"Generating StemIndexer testcase pt for {case.get('case_id', '<unknown>')}: {output_path}"
                )
                future = executor.submit(generate_and_save_case, case, output_path)
                future_to_case[future] = case

            for future in as_completed(future_to_case):
                case = future_to_case[future]
                try:
                    output_path = future.result()
                    saved_count += 1
                    print(f"Saved StemIndexer testcase pt: {output_path}")
                except Exception as err:
                    print(f"[FAILED] Generate pt for {case.get('case_id', '<unknown>')}: {err}")
                    raise
    print(
        f"Saved {saved_count}, skipped {skipped_count} existing StemIndexer testcase pt files."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate StemIndexer pt cases from CSV."
    )
    parser.add_argument("csv_path", type=str, help="CSV file path or glob pattern.")
    parser.add_argument(
        "pt_output_dir", type=str, help="Output directory for pt files."
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of case worker processes. 0 uses all available CPU cores.",
    )
    args = parser.parse_args()

    test_cases = load_csv_test_cases(args.csv_path)
    test_cases = select_test_cases(test_cases)
    save_test_cases(test_cases, args.pt_output_dir, args.workers)


if __name__ == "__main__":
    main()
