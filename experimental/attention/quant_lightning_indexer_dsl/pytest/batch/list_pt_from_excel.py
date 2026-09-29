# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import argparse

from batch.quant_lightning_indexer_v2_pt_save import load_excel_test_cases


def list_pt_from_excel(excel_path, sheet_name, pt_dir):
    return [
        str(Path(pt_dir) / f"{case_name}.pt")
        for case_name, _ in load_excel_test_cases(excel_path, sheet_name)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("excel_path")
    parser.add_argument("pt_dir")
    parser.add_argument("--sheet", default="Sheet1")
    args = parser.parse_args()
    print(",".join(list_pt_from_excel(args.excel_path, args.sheet, args.pt_dir)))


if __name__ == "__main__":
    main()
