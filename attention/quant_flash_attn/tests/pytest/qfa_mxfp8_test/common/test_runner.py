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

from common import quant_flash_attn_golden as golden
from common import golden_cache
from common import result_compare_method
from common.precision_compare_v2 import cross_check_compare

PARAM_MAP = {
    "B": "B",
    "N_q": "N_q",
    "N_kv": "N_kv",
    "D": "D",
    "cu_seqlens_q": "CU_SEQLENS_Q",
    "cu_seqlens_kv": "CU_SEQLENS_KV",
    "seqused_q": "SEQUSED_Q",
    "seqused_kv": "SEQUSED_KV",
    "batch_size": "BATCH_SIZE",
    "max_seqlen_q": "MAX_SEQLEN_Q",
    "max_seqlen_kv": "MAX_SEQLEN_KV",
    "enable_pa": "ENABLE_PA",
    "kv_cache_layout": "KV_CACHE_LAYOUT",
    "input_layout": "INPUT_LAYOUT",
    "block_size": "BLOCK_SIZE",
    "mask_mode": "SPARSE_MODE",
    "q_scale_layout": "Q_SCALE_LAYOUT",
    "p_scale": "P_SCALE",
    "softmax_scale": "SOFTMAX_SCALE",
    "data_range_q": "DATA_RANGE_Q",
    "data_range_k": "DATA_RANGE_K",
    "data_range_v": "DATA_RANGE_V",
    "data_range_qr": "DATA_RANGE_QR",
    "data_range_kr": "DATA_RANGE_KR",
    "enable_lse": "ENABLE_LSE",
    "enable_v_tail": "ENABLE_V_TAIL",
    "use_fp64_golden": "USE_FP64_GOLDEN",
    "quant_mode": None,  # quant_mode 用于新接口，golden 中不需要设置全局变量
    "device_id": "DEVICE_ID",
    "is_contiguous": "IS_CONTIGUOUS",
}


def apply_params(params):
    for param_key, golden_attr in PARAM_MAP.items():
        if param_key in params and golden_attr is not None:
            setattr(golden, golden_attr, params[param_key])


def execute_test(params, mode, cdir=None):
    apply_params(params)
    case_name = params["name"]

    if "gen" in mode:
        data = golden.generate_data()
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
        ) = data
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
                block_table_torch,
                vtail_bundle=golden._VTAIL_BUNDLE,
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
        ) = golden_cache.load_input(case_name, cache_dir=cdir)
        golden._VTAIL_BUNDLE = golden_cache.load_vtail_bundle(case_name, cache_dir=cdir)

    if (
        "gen" in mode
        and "cpu" not in mode
        and "npu" not in mode
        and "compare" not in mode
    ):
        return None, None

    if "cpu" in mode:
        # 两方比对的基准; 三方 cross_check 的 golden(bench) 在 compare 阶段现算
        cpu_out, cpu_lse = golden.cpu_mxfp8_golden(
            q_fp8,
            k_fp8,
            v_fp8,
            deq_q,
            deq_k,
            deq_v,
            p_scale,
            golden.SEQUSED_Q,
            golden.SEQUSED_KV,
            softmax_scale=golden.SOFTMAX_SCALE,
            qr_bf16=qr_bf16,
            kr_bf16=kr_bf16,
        )
        golden_cache.save_cpu_output(case_name, cpu_out, cpu_lse, cache_dir=cdir)
    else:
        cpu_out, cpu_lse = golden_cache.load_cpu_output(case_name, cache_dir=cdir)

    if "cpu" in mode and "npu" not in mode and "compare" not in mode:
        return None, None

    if "npu" in mode:
        with torch.profiler.record_function(f"mxfp8_fa::{case_name}"):
            npu_out, lse_out = golden.npu_mxfp8_fa(
                q_fp8,
                k_fp8,
                v_fp8,
                deq_q,
                deq_k,
                deq_v,
                p_scale,
                golden.CU_SEQLENS_Q,
                golden.CU_SEQLENS_KV,
                golden.SEQUSED_Q,
                golden.SEQUSED_KV,
                golden.MAX_SEQLEN_Q,
                golden.MAX_SEQLEN_KV,
                block_table_torch,
                qr_bf16,
                kr_bf16,
                batch_size=golden.BATCH_SIZE,
            )
        golden_cache.save_npu_output(case_name, npu_out, lse_out, cache_dir=cdir)
    else:
        npu_out, lse_out = golden_cache.load_npu_output(case_name, cache_dir=cdir)

    if "npu" in mode and "compare" not in mode:
        return None, None

    # NPU op 约束: layout_out == layout_q, 故输出排布始终为 INPUT_LAYOUT
    compare_layout = golden.INPUT_LAYOUT
    act_seqused_q = golden._get_act_seq_q()

    def _to_cmp(tensor):
        if compare_layout == "TND":
            return golden.convert_q_bnsd_to_layout(
                tensor, act_seqused_q, compare_layout, cu_seqlens=golden.CU_SEQLENS_Q
            )
        return golden.convert_q_bnsd_to_layout(tensor, act_seqused_q, compare_layout)

    def _lse_to_cmp(lse_bnsd):
        lse_cmp = _to_cmp(lse_bnsd)
        if compare_layout == "TND":
            # TND padding 位置填 inf 以匹配 NPU 行为：NPU 对超出实际序列长度的 Q 位置输出 inf LSE
            golden.fill_tnd_padding(
                lse_cmp, act_seqused_q, golden.CU_SEQLENS_Q, fill_value=float("inf")
            )
            # NPU LSE 输出为 N-major 排布 (N, T): golden 经 convert 后是 [T, N, 1] (T-major)
            lse_cmp = lse_cmp.squeeze(-1).permute(1, 0).contiguous()
        else:
            # BNSD/BSND: LSE 输出固定为 (B, N, S) 排布 (与 layout_out 无关, 由 infershape 决定)
            # CPU golden lse 为 [B, N, S, 1], 转成 BNSD 后 squeeze 掉末维对齐
            lse_cmp = (
                golden.convert_q_bnsd_to_layout(lse_bnsd, act_seqused_q, "BNSD")
                .squeeze(-1)
                .contiguous()
            )
        return lse_cmp

    if golden.USE_FP64_GOLDEN:
        # 三方 cross_check 仅做归因报告(与 master 的 CV 双标杆语义一致):
        # 结果只打印 [CROSS_CHECK] 指标, 不参与 pass/fail —— 比值判据取最差元素、
        # 零容忍, 会把 <=0.05% 的近零离群元素判成失败; 判定仍由两方 check_result 决定。
        #   golden      = FP64 bench(保留 P 的 FP8 量化, 作为误差分母)
        #   third_party = 分段量化参考(竞品噪声上界)
        #   actual      = NPU
        # 序列长度统一走 _get_act_seq_*: BNSD/BSND 仅给 batch_size 的用例
        # SEQUSED_* 为 None, 直接用会触发 len(None)。
        act_seq_q = golden._get_act_seq_q()
        act_seq_kv = golden._get_act_seq_kv()
        bench_out, bench_lse = golden.cpu_mxfp8_bench(
            q_fp8,
            k_fp8,
            v_fp8,
            deq_q,
            deq_k,
            deq_v,
            p_scale,
            act_seq_q,
            act_seq_kv,
            softmax_scale=golden.SOFTMAX_SCALE,
            qr_bf16=qr_bf16,
            kr_bf16=kr_bf16,
        )
        ref_out, ref_lse = golden.cpu_mxfp8_sectioned_golden(
            q_fp8,
            k_fp8,
            v_fp8,
            deq_q,
            deq_k,
            deq_v,
            p_scale,
            act_seq_q,
            act_seq_kv,
            softmax_scale=golden.SOFTMAX_SCALE,
            v_tail_bnsd=(
                golden._VTAIL_BUNDLE["v_tail_bnsd_fp16"]
                if golden.ENABLE_V_TAIL and golden._VTAIL_BUNDLE is not None
                else None
            ),
            tail_lens=(
                golden._VTAIL_BUNDLE["seqused_v_tail"].tolist()
                if golden.ENABLE_V_TAIL and golden._VTAIL_BUNDLE is not None
                else None
            ),
            qr_bf16=qr_bf16,
            kr_bf16=kr_bf16,
        )
        cross_check_compare(
            npu_out,
            _to_cmp(bench_out),
            # 与 TTK 一致: 分段参考 cast bf16, 使其量化噪声水平与 target 可比
            _to_cmp(ref_out).to(torch.bfloat16),
            # 容差档按 NPU 输出 dtype 取(与 quant_flash_mla / flash_attn 口径一致):
            # bf16 输出 → small_value=2^-8/atol=2^-16; 不能按 fp64 bench 取档
            output_dtype=npu_out.dtype,
            test_name=f"qfa_mxfp8_attention_out::{case_name}",
        )

    # pass/fail 恒定走两方判据(FP32 golden vs NPU), 三方结果只用于归因
    cpu_cmp = _to_cmp(cpu_out)
    atten_result = result_compare_method.check_result(cpu_cmp, npu_out)

    lse_result = None
    if golden.ENABLE_LSE:
        if golden.USE_FP64_GOLDEN:
            cross_check_compare(
                lse_out,
                _lse_to_cmp(bench_lse),
                _lse_to_cmp(ref_lse),
                output_dtype=lse_out.dtype,
                test_name=f"qfa_mxfp8_lse::{case_name}",
            )
        lse_cmp = _lse_to_cmp(cpu_lse)
        lse_result = result_compare_method.check_result(
            lse_cmp, lse_out, max_diff_hd=2000
        )

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
