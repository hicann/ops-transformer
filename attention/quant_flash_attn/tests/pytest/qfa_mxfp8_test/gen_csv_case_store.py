#!/usr/bin/python3
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
"""
python3 gen_csv_case_store.py [--paramset quant_flash_attn_paramset_func_rdv]
                              [--output qfa_mxfp8.csv]
"""

import argparse
import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

API_NAME = "qfa_mxfp8_wrapper.npu_qfa_mxfp8"


def case_to_csv_row(case):
    """把一个 case dict 转成 CSV 行(list,由 csv.writer 负责转义)。"""
    name = case["name"]
    B = case["B"]
    N_q = case["N_q"]
    N_kv = case["N_kv"]
    D = case["D"]
    cu_seqlens_q = (
        list(case["cu_seqlens_q"]) if case["cu_seqlens_q"] is not None else []
    )
    cu_seqlens_kv = (
        list(case["cu_seqlens_kv"]) if case["cu_seqlens_kv"] is not None else []
    )
    seqused_q = case["seqused_q"]
    seqused_kv = case["seqused_kv"]
    max_seqlen_q = case["max_seqlen_q"]
    max_seqlen_kv = case["max_seqlen_kv"]
    enable_pa = case["enable_pa"]
    block_size = case["block_size"]
    enable_lse = case["enable_lse"]

    # cu_seqlens → actual_seq (差分还原,用于计算 shape)
    actual_seq_q = (
        [cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q) - 1)]
        if len(cu_seqlens_q) > 1
        else [0]
    )
    actual_seq_kv = (
        [cu_seqlens_kv[i + 1] - cu_seqlens_kv[i] for i in range(len(cu_seqlens_kv) - 1)]
        if len(cu_seqlens_kv) > 1
        else [0]
    )
    max_sq = max(actual_seq_q) if actual_seq_q else D
    max_skv = max(actual_seq_kv) if actual_seq_kv else D

    num_groups = math.ceil(D / 32)
    max_blocks = (
        max(math.ceil(s / block_size) for s in actual_seq_kv)
        if actual_seq_kv and enable_pa
        else 0
    )

    shapes = [
        f"({B},{N_q},{max_sq},{D})",
        f"({B},{N_kv},{max_skv},{D})",
        f"({B},{N_kv},{max_skv},{D})",
        f"({B},{N_q},{max_sq},{num_groups})",
        f"({B},{N_kv},{max_skv},{num_groups})",
        f"({B},{N_kv},{math.ceil(max_skv / 32)},{D})",
        "(1,)",
        f"({B},{max_blocks})" if enable_pa and max_blocks > 0 else "(0,)",
    ]
    tensor_view_shapes = "(" + ",".join(f"({s})" for s in shapes) + ")"

    dtypes = [
        "'float16'",
        "'float16'",
        "'float16'",
        "'float32'",
        "'float32'",
        "'float32'",
        "'float32'",
        "'int32'",
    ]
    tensor_dtypes = "(" + ",".join(dtypes) + ")"

    attrs = {
        "B": B,
        "N_q": N_q,
        "N_kv": N_kv,
        "D": D,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_kv": cu_seqlens_kv,
        "seqused_q": seqused_q,
        "seqused_kv": seqused_kv,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
        "enable_pa": enable_pa,
        "kv_cache_layout": case["kv_cache_layout"],
        "block_size": block_size,
        "mask_mode": case["mask_mode"],
        "q_scale_layout": case["q_scale_layout"],
        "quant_mode": case.get("quant_mode", 1),
        "enable_lse": enable_lse,
        "graph_path": case.get("graph_path", 0),
        "input_layout": "TND",
        "is_contiguous": case.get("is_contiguous", True),
        "device_id": case.get("device_id", 0),
        "softmax_scale": case.get("softmax_scale"),
        "data_range_q": case.get("data_range_q", 1.0),
        "data_range_k": case.get("data_range_k", 1.0),
        "data_range_v": case.get("data_range_v", 1.0),
        "data_range_qr": case.get("data_range_qr", 1.0),
        "data_range_kr": case.get("data_range_kr", 1.0),
    }
    # TTK process_dict 用 eval() 解析,需 Python dict 字面量(单引号)
    attributes = repr(attrs)

    # wrapper 的 8 个 tensor 全是输入,输出是函数返回值(非 tensor 参数)
    # 所以 output_tensor_indexes 为空
    output_tensor_indexes = ""

    return [
        name,
        API_NAME,
        tensor_view_shapes,
        tensor_dtypes,
        "",
        attributes,
        output_tensor_indexes,
        "",
    ]


def main():
    parser = argparse.ArgumentParser(description="paramset → TTK e2e CSV")
    parser.add_argument(
        "--paramset",
        default="quant_flash_attn_paramset_func_rdv",
        help="paramset 模块名(不含.py)",
    )
    parser.add_argument("--output", default="qfa_mxfp8.csv", help="输出 CSV 路径")
    args = parser.parse_args()

    paramset_mod = __import__(args.paramset)
    skip = getattr(paramset_mod, "SKIP_CASES", set())

    # paramset 文件可能直接导出 CASES,也可能只导出 TEST_PARAMS(需要 expand)
    if hasattr(paramset_mod, "CASES") and paramset_mod.CASES is not None:
        cases = paramset_mod.CASES
    elif hasattr(paramset_mod, "TEST_PARAMS"):
        from quant_flash_attn_paramset_common import expand_paramset_to_cases

        cases = expand_paramset_to_cases(paramset_mod.TEST_PARAMS)
    else:
        raise ValueError(
            f"paramset module '{args.paramset}' has neither CASES nor TEST_PARAMS"
        )

    header = [
        "testcase_name",
        "api_name",
        "tensor_view_shapes",
        "tensor_dtypes",
        "tensor_formats",
        "attributes",
        "output_tensor_indexes",
        "golden_api",
    ]
    out_path = (
        args.output if os.path.isabs(args.output) else os.path.join(_HERE, args.output)
    )
    written = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for case in cases:
            if case["name"] in skip:
                skipped += 1
                continue
            writer.writerow(case_to_csv_row(case))
            written += 1

    print(f"生成 {written} 个 case (跳过 {skipped} 个 SKIP) → {out_path}")


if __name__ == "__main__":
    main()
