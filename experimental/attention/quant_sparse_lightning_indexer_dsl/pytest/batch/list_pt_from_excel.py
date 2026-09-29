# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("excel_path")
    parser.add_argument("pt_dir")
    parser.add_argument("--sheet", "-S", default="TestCases")
    args = parser.parse_args()
    frame = pd.read_excel(args.excel_path, sheet_name=args.sheet)
    if "Testcase_Name" not in frame.columns:
        raise ValueError("Column 'Testcase_Name' is required")
    root = Path(args.pt_dir)
    files = [root / f"{name}.pt" for name in frame["Testcase_Name"]]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing PT cases: {missing}")
    print(",".join(str(path) for path in files))


if __name__ == "__main__":
    main()
