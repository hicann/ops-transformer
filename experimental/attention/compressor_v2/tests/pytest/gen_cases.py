#!/usr/bin/env python
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
"""Generate compressor_v2 test case Excel from test_compressor_paramset.py ENABLED_PARAMS."""

import itertools
import sys
import pandas as pd

sys.path.insert(
    0,
    "/home/m00946864/ops-transformer/experimental/attention/compressor_v2/tests/pytest",
)
import test_compressor_paramset as p  # noqa: E402

PARAM_NAMES = [
    "batch_size",
    "hidden_size",
    "Seq_len",
    "head_dim",
    "block_size",
    "cmp_ratio",
    "start_p",
    "layout_x",
    "data_type",
    "cu_seqlens",
    "seqused",
    "start_pos",
    "x_datarange",
    "wkv_datarange",
    "wgate_datarange",
    "kv_state_datarange",
    "score_state_datarange",
]

# pt_save.py 需要的列（多出 coff/cache_mode/ape_datarange，用默认值）
OUT_COLUMNS = [
    "Testcase_Name",
    "batch_size",
    "hidden_size",
    "Seq_len",
    "head_dim",
    "block_size",
    "cmp_ratio",
    "start_p",
    "layout_x",
    "data_type",
    "cu_seqlens",
    "seqused",
    "start_pos",
    "x_datarange",
    "wkv_datarange",
    "wgate_datarange",
    "kv_state_datarange",
    "score_state_datarange",
    "is_contiguous",
]


def fmt_range(r):
    return f"{r[0]},{r[1]}"


rows = []
for group_name, params in p.TEST_PARAMS.items():
    vals = [params[k] for k in PARAM_NAMES]
    for i, combo in enumerate(itertools.product(*vals)):
        d = dict(zip(PARAM_NAMES, combo))
        row = {
            "Testcase_Name": f"{group_name}_{i}"
            if len(list(itertools.product(*vals))) > 1
            else group_name,
            "batch_size": d["batch_size"],
            "hidden_size": d["hidden_size"],
            "Seq_len": d["Seq_len"],
            "head_dim": d["head_dim"],
            "block_size": d["block_size"],
            "cmp_ratio": d["cmp_ratio"],
            "start_p": d["start_p"],
            "layout_x": d["layout_x"],
            "data_type": "BF16" if "bfloat" in str(d["data_type"]) else "FP16",
            "cu_seqlens": d["cu_seqlens"],
            "seqused": d["seqused"],
            "start_pos": d["start_pos"],
            "x_datarange": fmt_range(d["x_datarange"]),
            "wkv_datarange": fmt_range(d["wkv_datarange"]),
            "wgate_datarange": fmt_range(d["wgate_datarange"]),
            "kv_state_datarange": fmt_range(d["kv_state_datarange"]),
            "score_state_datarange": fmt_range(d["score_state_datarange"]),
            "is_contiguous": params.get("is_contiguous", [False])[0],
        }
        rows.append(row)

df = pd.DataFrame(rows, columns=OUT_COLUMNS)
out = "/home/m00946864/ops-transformer/experimental/attention/compressor_v2/tests/pytest/excel/test_cases.csv"
import os

os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print(f"written {len(rows)} rows -> {out}")
print(df.to_string())
