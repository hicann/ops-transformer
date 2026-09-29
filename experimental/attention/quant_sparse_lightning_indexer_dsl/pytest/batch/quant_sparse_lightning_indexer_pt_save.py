# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import ast
from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsli_test_utils import (
    QsliCaseSelector,
    convert_qsli_case_to_tnd,
    generate_qsli_test_data,
)


def _value(value):
    if hasattr(value, "item"):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text in ("True", "False", "None") or text.startswith(("[", "(", "{")):
            return ast.literal_eval(text)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("excel_path")
    parser.add_argument("pt_dir")
    parser.add_argument("--sheet", "-S", default="TestCases")
    parser.add_argument("--indexes", default="")
    parser.add_argument("--paired-layouts", action="store_true")
    parser.add_argument("--defer-golden", action="store_true")
    args = parser.parse_args()
    frame = pd.read_excel(args.excel_path, sheet_name=args.sheet)
    required = [
        "Testcase_Name",
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
        "weight_dtype",
        "dequant_dtype",
        "actual_seq_dtype",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "seqused_q",
        "seqused_k",
        "cmp_residual_k",
        "max_seqlen_q",
        "quant_mode",
        "layout_query",
        "layout_key",
        "sparse_count",
        "sparse_mode",
        "query_datarange",
        "key_datarange",
        "weights_datarange",
        "q_scale_datarange",
        "k_scale_datarange",
        "cmp_ratio",
        "return_value",
        "output_idx_offset",
        "candidate_block_indices",
        "candidate_block_length",
        "candidate_block_size",
        "block_table",
        "storage_layout",
        "seed",
    ]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    output = Path(args.pt_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.indexes:
        selected = QsliCaseSelector.parse_indexes(args.indexes, len(frame))
        frame = frame.iloc[[index - 1 for index in selected]]
    for source_index, row in frame.iterrows():
        params = {
            name: _value(row[name]) for name in required if name != "Testcase_Name"
        }
        target = output / f"{row['Testcase_Name']}.pt"
        case_data = generate_qsli_test_data(
            params, defer_golden=args.defer_golden or args.paired_layouts
        )
        case_data["source"] = dict(
            workbook=str(Path(args.excel_path).resolve()),
            sheet=args.sheet,
            row=int(source_index) + 2,
            case_name=str(row["Testcase_Name"]),
        )
        if args.paired_layouts:
            for layout, data in (
                ("PA_BBND", case_data),
                ("TND", convert_qsli_case_to_tnd(case_data)),
            ):
                target = output / f"{row['Testcase_Name']}__{layout}.pt"
                torch.save(data, target)
                print(f"saved: {target}")
        else:
            torch.save(case_data, target)
            print(f"saved: {target}")


if __name__ == "__main__":
    main()
