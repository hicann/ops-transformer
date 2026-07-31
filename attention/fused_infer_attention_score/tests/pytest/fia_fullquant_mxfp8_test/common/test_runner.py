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

import pytest
import torch

from common import fia_fullquant_mxfp8_golden as golden
from common import golden_cache
from common import result_compare_method

PARAM_MAP = {
    "B": "B",
    "N_q": "N_q",
    "N_kv": "N_kv",
    "D": "D",
    "actual_seq_q": "ACTUAL_SEQ_Q",
    "actual_seq_kv": "ACTUAL_SEQ_KV",
    "enable_pa": "ENABLE_PA",
    "kv_cache_layout": "KV_CACHE_LAYOUT",
    "block_size": "BLOCK_SIZE",
    "sparse_mode": "SPARSE_MODE",
    "q_scale_layout": "Q_SCALE_LAYOUT",
    "p_scale": "P_SCALE",
    "data_range_q": "DATA_RANGE_Q",
    "data_range_k": "DATA_RANGE_K",
    "data_range_v": "DATA_RANGE_V",
    "data_range_qr": "DATA_RANGE_QR",
    "data_range_kr": "DATA_RANGE_KR",
    "enable_lse": "ENABLE_LSE",
    "enable_rope": "ENABLE_ROPE",
    "D_rope": "D_rope",
    "graph_path": "GRAPH_PATH",
    "device_id": "DEVICE_ID",
    "is_contiguous": "IS_CONTIGUOUS",
}


def apply_params(params):
    for param_key, golden_attr in PARAM_MAP.items():
        if param_key in params:
            setattr(golden, golden_attr, params[param_key])


def execute_test(params, mode, cdir=None):
    apply_params(params)
    case_name = params["name"]

    if "gen" in mode:
        data = golden.generate_data()
        q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, qr_bf16, kr_bf16, block_table_torch = data
        golden_cache.save_input(case_name, golden_cache.build_input_dict(
            q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, qr_bf16, kr_bf16, block_table_torch
        ), cache_dir=cdir)
    else:
        q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, qr_bf16, kr_bf16, block_table_torch = \
            golden_cache.load_input(case_name, cache_dir=cdir)

    if "gen" in mode and "cpu" not in mode and "npu" not in mode and "compare" not in mode:
        return None, None

    if "cpu" in mode:
        cpu_out, cpu_lse = golden.cpu_mxfp8_golden(
            q_fp8, k_fp8, v_fp8,
            deq_q, deq_k, deq_v, p_scale,
            golden.ACTUAL_SEQ_Q, golden.ACTUAL_SEQ_KV,
            qr_bf16, kr_bf16,
        )
        golden_cache.save_cpu_output(case_name, cpu_out, cpu_lse, cache_dir=cdir)

    if "npu" in mode and "cpu" not in mode and "compare" not in mode:
        return None, None

    if "cpu" not in mode:
        cpu_out, cpu_lse = golden_cache.load_cpu_output(case_name, cache_dir=cdir)

    if "cpu" in mode and "npu" not in mode and "compare" not in mode:
        return None, None

    if "npu" in mode:
        with torch.profiler.record_function(f"mxfp8_fa::{case_name}"):
            npu_out, lse_out = golden.npu_mxfp8_fa(
                q_fp8, k_fp8, v_fp8,
                deq_q, deq_k, deq_v, p_scale,
                golden.ACTUAL_SEQ_Q, golden.ACTUAL_SEQ_KV,
                block_table_torch, qr_bf16, kr_bf16,
            )
        golden_cache.save_npu_output(case_name, npu_out, lse_out, cache_dir=cdir)
    else:
        npu_out, lse_out = golden_cache.load_npu_output(case_name, cache_dir=cdir)

    if "npu" in mode and "compare" not in mode:
        return None, None

    compare_layout = "TND" if golden.ENABLE_PA else golden.INPUT_LAYOUT
    cpu_cmp = golden.convert_q_bnsd_to_layout(cpu_out, golden.ACTUAL_SEQ_Q, compare_layout)

    atten_result = result_compare_method.check_result(cpu_cmp, npu_out)

    lse_result = None
    if golden.ENABLE_LSE:
        lse_cmp = golden.convert_q_bnsd_to_layout(cpu_lse, golden.ACTUAL_SEQ_Q, compare_layout)
        lse_result = result_compare_method.check_result(lse_cmp, lse_out)

    return atten_result, lse_result


def check_results(atten_result, lse_result):
    if atten_result is None:
        return

    atten_status = atten_result[0] if isinstance(atten_result, tuple) else atten_result
    if atten_status != "Pass":
        pct = atten_result[1] if isinstance(atten_result, tuple) and len(atten_result) > 1 else "N/A"
        pytest.fail(f"Attention output compare failed: result={atten_status}, PctRlt={pct}")

    if lse_result is not None:
        lse_status = lse_result[0] if isinstance(lse_result, tuple) else lse_result
        if lse_status != "Pass":
            pct = lse_result[1] if isinstance(lse_result, tuple) and len(lse_result) > 1 else "N/A"
            pytest.fail(f"LSE compare failed: result={lse_status}, PctRlt={pct}")
