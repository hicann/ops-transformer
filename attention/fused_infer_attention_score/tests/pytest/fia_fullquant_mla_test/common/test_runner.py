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

from common import fia_fullquant_mla_golden as golden
from common import golden_cache
from common import result_compare_method

# paramset 字段 → golden 模块级变量
PARAM_MAP = {
    "B": "B",
    "N_q": "N_q",
    "N_kv": "N_kv",
    "D": "D",
    "D_v": "D_V",
    "D_rope": "D_rope",
    "actual_seq_q": "ACTUAL_SEQ_Q",
    "actual_seq_kv": "ACTUAL_SEQ_KV",
    "enable_pa": "ENABLE_PA",
    "enable_lse": "ENABLE_LSE",
    "golden_mode": "GOLDEN_MODE",
    "block_size": "BLOCK_SIZE",
    "sparse_mode": "SPARSE_MODE",
    "input_layout": "INPUT_LAYOUT",
    "output_layout": "OUTPUT_LAYOUT",
    "kv_cache_layout": "KV_CACHE_LAYOUT",
    "p_scale": "P_SCALE",
    "scale_value": "SCALE_VALUE",
    "is_contiguous": "IS_CONTIGUOUS",
    "num_blocks": "NUM_BLOCKS",
    "graph_path": "GRAPH_PATH",
    "device_id": "DEVICE_ID",
    "q_data_range": "Q_DATA_RANGE",
    "k_data_range": "K_DATA_RANGE",
    "v_data_range": "V_DATA_RANGE",
    "seed_q": "SEED_Q",
    "seed_k": "SEED_K",
    "seed_v": "SEED_V",
    "seed_qr": "SEED_QR",
    "seed_kr": "SEED_KR",
    "seed_block_table": "SEED_BLOCK_TABLE",
}


def apply_params(params):
    for param_key, golden_attr in PARAM_MAP.items():
        if param_key in params:
            setattr(golden, golden_attr, params[param_key])


def execute_test(params, mode, cdir=None):
    """执行单个 case 的全流程

    mode: {"gen","cpu","npu","compare"} 的子集
    cdir: golden 缓存目录
    返回 (atten_result, lse_result)；当 mode 只含 gen/cpu/npu 之一时返回 (None, None)
    """
    apply_params(params)
    case_name = params["name"]

    # ---- Step 1: 输入数据 ----
    block_table_torch = None
    if "gen" in mode:
        (q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, qr_bf16, kr_bf16) = (
            golden.generate_data()
        )
        golden_cache.save_input(
            case_name,
            golden_cache.build_input_dict(
                q_fp8,
                k_fp8,
                v_fp8,
                deq_q,
                deq_k,
                deq_v,
                p_scale,
                qr_bf16,
                kr_bf16,
                None,
                golden.NUM_BLOCKS,
                golden.KV_CACHE_LAYOUT,
            ),
            cache_dir=cdir,
        )
    else:
        (
            q_fp8,
            k_fp8,
            v_fp8,
            deq_q,
            deq_k,
            deq_v,
            p_scale,
            qr_bf16,
            kr_bf16,
            block_table_torch,
            num_blocks_loaded,
            kv_layout_loaded,
        ) = golden_cache.load_input(case_name, cache_dir=cdir)
        golden.NUM_BLOCKS = num_blocks_loaded
        golden.KV_CACHE_LAYOUT = kv_layout_loaded

    if "gen" in mode and not (mode & {"cpu", "npu", "compare"}):
        return None, None

    # ---- Step 2: CPU Golden ----
    if "cpu" in mode:
        if golden.NUM_BLOCKS != 0:
            # NUM_BLOCKS 非 0 时 CPU golden 需要从 cache 还原，留到 npu 阶段拿到 cache 后处理
            cpu_out, cpu_lse = None, None
        else:
            cpu_out, cpu_lse = golden.cpu_fp8_fullquant_mla_golden(
                q_fp8,
                k_fp8,
                v_fp8,
                deq_q,
                deq_k,
                deq_v,
                p_scale,
                golden.ACTUAL_SEQ_Q,
                golden.ACTUAL_SEQ_KV,
                qr_bf16,
                kr_bf16,
            )
            golden_cache.save_cpu_output(case_name, cpu_out, cpu_lse, cache_dir=cdir)
    else:
        cpu_out, cpu_lse = golden_cache.load_cpu_output(case_name, cache_dir=cdir)

    if "cpu" in mode and not (mode & {"npu", "compare"}):
        return None, None

    # ---- Step 3: NPU ----
    cache_info = None
    if "npu" in mode:
        with torch.profiler.record_function(f"mla_fa::{case_name}"):
            output, cache_info = golden.npu_fp8_full_quant_mla(
                q_fp8,
                k_fp8,
                v_fp8,
                deq_q,
                deq_k,
                deq_v,
                p_scale,
                golden.ACTUAL_SEQ_Q,
                golden.ACTUAL_SEQ_KV,
                block_table_torch,
                qr_bf16,
                kr_bf16,
            )
        npu_out, lse_out = output
        golden_cache.save_npu_output(case_name, npu_out, lse_out, cache_dir=cdir)
    else:
        npu_out, lse_out = golden_cache.load_npu_output(case_name, cache_dir=cdir)

    if "npu" in mode and "compare" not in mode:
        return None, None

    # ---- Step 3.5: NUM_BLOCKS != 0 时重建 CPU golden ----
    if cache_info is not None and cpu_out is None:
        k_pa_cache, v_pa_cache, bt_cache = cache_info
        k_bnsd_recon, v_bnsd_recon = golden.pa_cache_to_bnsd(
            k_pa_cache,
            v_pa_cache,
            bt_cache,
            golden.ACTUAL_SEQ_KV,
            golden.BLOCK_SIZE,
            kv_layout=golden.KV_CACHE_LAYOUT,
            n_kv=golden.N_kv,
        )
        cpu_out, cpu_lse = golden.cpu_fp8_fullquant_mla_golden(
            q_fp8,
            k_bnsd_recon,
            v_bnsd_recon,
            deq_q,
            deq_k,
            deq_v,
            p_scale,
            golden.ACTUAL_SEQ_Q,
            golden.ACTUAL_SEQ_KV,
            qr_bf16,
            kr_bf16,
        )
        if "cpu" in mode:
            golden_cache.save_cpu_output(case_name, cpu_out, cpu_lse, cache_dir=cdir)

    # ---- Step 4/5: 精度对比 ----
    compare_layout = "TND" if golden.ENABLE_PA else golden.INPUT_LAYOUT
    cpu_cmp = golden.convert_q_bnsd_to_layout(
        cpu_out, golden.ACTUAL_SEQ_Q, compare_layout
    )
    atten_result = result_compare_method.check_result(cpu_cmp, npu_out)

    lse_result = None
    if golden.ENABLE_LSE:
        lse_cmp = golden.convert_q_bnsd_to_layout(
            cpu_lse, golden.ACTUAL_SEQ_Q, compare_layout
        )
        lse_result = result_compare_method.check_result(lse_cmp, lse_out)

    return atten_result, lse_result


def check_results(atten_result, lse_result):
    if atten_result is None:
        return

    atten_status = atten_result[0] if isinstance(atten_result, tuple) else atten_result
    if atten_status != "Pass":
        pct = (
            atten_result[1]
            if isinstance(atten_result, tuple) and len(atten_result) > 1
            else "N/A"
        )
        pytest.fail(
            f"Attention output compare failed: result={atten_status}, PctRlt={pct}"
        )

    if lse_result is not None:
        lse_status = lse_result[0] if isinstance(lse_result, tuple) else lse_result
        if lse_status != "Pass":
            pct = (
                lse_result[1]
                if isinstance(lse_result, tuple) and len(lse_result) > 1
                else "N/A"
            )
            pytest.fail(f"LSE compare failed: result={lse_status}, PctRlt={pct}")
