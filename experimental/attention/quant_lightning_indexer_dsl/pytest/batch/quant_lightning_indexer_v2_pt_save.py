# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from pathlib import Path
import argparse

import torch

import quant_lightning_indexer_v2_golden
from qliv2_parameter_normalization import normalize_cell
from qliv2_test_utils import PARAM_NAMES, QliV2CaseSelector, QliV2ResultWriter


def load_excel_test_cases(excel_file_path: str, sheetname: str):
    import pandas as pd

    frame = pd.read_excel(excel_file_path, sheet_name=sheetname)
    # block_table was added after the original transfer workbook.  Keep the
    # old sheets runnable by treating a missing column as AUTO.
    required = [
        "Testcase_Name",
        *(name for name in PARAM_NAMES if name != "block_table"),
    ]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Excel is missing QLI columns: {missing}")
    cases = []
    for _, row in frame.iterrows():
        values = tuple(
            "AUTO"
            if name == "block_table" and name not in frame.columns
            else normalize_cell(row[name])
            for name in PARAM_NAMES
        )
        explicit_name = normalize_cell(row.get("Testcase_Name"))
        cases.append((QliV2ResultWriter.case_name(values, explicit_name), values))
    return cases


def save_test_case(test_cases, file_path):
    output = Path(file_path)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for case_name, test_data in test_cases:
        case = quant_lightning_indexer_v2_golden.generate_qliv2_test_data(test_data)
        path = output / f"{case_name}.pt"
        torch.save(case, path)
        paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("excel_file")
    parser.add_argument("pt_dir")
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument("--indexes", default="")
    args = parser.parse_args()
    cases = load_excel_test_cases(args.excel_file, args.sheet)
    if args.indexes:
        selected = QliV2CaseSelector.parse_indexes(args.indexes, len(cases))
        cases = [cases[index - 1] for index in selected]
    paths = save_test_case(cases, args.pt_dir)
    print(f"saved {len(paths)} PT cases to {args.pt_dir}")


if __name__ == "__main__":
    main()
