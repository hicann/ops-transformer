# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import torch


def _case(
    batch,
    s1,
    s2,
    *,
    ratio=1,
    storage="contiguous",
    candidate=2048,
    candidate_block=8,
    seed=1,
):
    pages = (s2 + 127) // 128
    return {
        "batch_size": [batch],
        "q_seq": [s1],
        "k_seq": [s2],
        "q_t_size": [batch * s1],
        "k_t_size": [s2],
        "q_head_num": [32],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128],
        "block_num": [pages],
        "qk_dtype": [torch.float4_e2m1fn_x2],
        "weight_dtype": [torch.float32],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": ["AUTO"],
        "cu_seqlens_k": [None],
        "seqused_q": ["AUTO"],
        "seqused_k": ["AUTO"],
        "cmp_residual_k": [
            None if ratio == 1 else [index % ratio for index in range(batch)]
        ],
        "block_table": ["AUTO"],
        "max_seqlen_q": [s1],
        "quant_mode": [1],
        "layout_query": ["TND"],
        "layout_key": ["PA_BBND"],
        "sparse_count": [512],
        "sparse_mode": [3],
        "query_datarange": [[-6.0, 6.0]],
        "key_datarange": [[-6.0, 6.0]],
        "weights_datarange": [[0.0, 1.0]],
        "q_scale_datarange": [[0.5, 2.0]],
        "k_scale_datarange": [[0.5, 2.0]],
        "cmp_ratio": [ratio],
        "return_value": [1],
        "output_idx_offset": [None],
        "candidate_topk_blocks": [candidate],
        "candidate_block_size": [candidate_block],
        "storage_layout": [storage],
        "seed": [seed],
        "run_mode": ["eager"],
    }


TEST_PARAMS = {
    "cleancode_b1_s1_4_s2_8192": _case(1, 4, 8192, ratio=2, seed=301),
    "decode_b8_s2_8k": _case(8, 1, 8192, seed=101),
    "prefill_b1_s1_8_s2_4k": _case(1, 8, 4096, seed=102),
    "cross_01_tiny_s2_5": _case(1, 1, 5, seed=201),
    "cross_02_s2_tail_127": _case(2, 1, 127, seed=202),
    "cross_03_query_tail_s1_3": _case(4, 3, 128, seed=203),
    "cross_04_full_query_tile_s1_4": _case(8, 4, 129, seed=204),
    "cross_05_cmp_ratio_2": _case(16, 1, 1024, ratio=2, seed=205),
    "cross_06_max_batch_32": _case(32, 1, 2048, seed=206),
    "cross_07_multi_query_task": _case(3, 5, 8192, ratio=2, seed=207),
    "cross_08_topk_trunk_merge": _case(1, 1, 16385, seed=208),
    "cross_09_pa_axis0_noncontiguous": _case(
        2, 1, 32768, storage="axis0_noncontiguous", seed=209
    ),
    "cross_10_candidate_disabled": _case(
        4, 1, 8192, candidate=-1, candidate_block=-1, seed=210
    ),
}


def _excel_cases():
    from pathlib import Path
    import openpyxl
    from qliv2_parameter_normalization import normalize_cell

    selections = (
        ("qli_0909.xlsx", (28, 31, 32, 37, 45, 48, 54, 59)),
        ("qli_decode_prefill_20cases.xlsx", (2, 3, 7, 8, 12, 13, 18, 22)),
    )
    cases = []
    for filename, row_numbers in selections:
        workbook = openpyxl.load_workbook(
            Path(__file__).parent / "excel" / filename, data_only=True
        )
        rows = list(workbook.active.values)
        workbook.close()
        for row_number in row_numbers:
            values = {
                key: normalize_cell(value)
                for key, value in zip(rows[0], rows[row_number - 1])
            }
            name = values.pop("Testcase_Name")
            values["layout_key"] = "TND"
            values["block_table"] = None
            values["block_num"] = (values["k_seq"] + 127) // 128
            values["output_idx_offset"] = None
            values["run_mode"] = "eager"
            index = len(cases)
            if index == 0:
                values["storage_layout"] = "key_axis0_noncontiguous"
            elif index == 1:
                values["storage_layout"] = "axis0_noncontiguous"
                values["cmp_ratio"] = 1
                values["cmp_residual_k"] = None
            elif index == 2:
                values["storage_layout"] = "scale_axis0_noncontiguous"
                values["max_seqlen_q"] = -1
            elif index == 3:
                values["sparse_mode"] = 0
                values["cmp_residual_k"] = None
            elif index == 4:
                values["candidate_topk_blocks"] = -1
                values["candidate_block_size"] = -1
            elif index == 5:
                values["seqused_q"] = None
            elif index == 8:
                values["seqused_k"] = None
            elif index == 9:
                lengths = [4096, 4060, 4030, 3999]
                bounds = [0]
                for length in lengths:
                    bounds.append(bounds[-1] + length)
                values["cu_seqlens_k"] = bounds
            case_name = f"excel_{index + 1:02d}_{name}"
            cases.append((case_name, {key: [value] for key, value in values.items()}))
    return cases


ENABLED_PARAMSETS = list(TEST_PARAMS.items()) + _excel_cases()
