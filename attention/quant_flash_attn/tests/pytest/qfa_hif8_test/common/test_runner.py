# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging

import pytest
import torch

from common import golden_cache, precision_compare_v2 as _CV, result_compare_method
from common import quant_flash_attn_golden as golden

logger = logging.getLogger(__name__)

PARAM_MAP = {
    "B": "B",
    "N_q": "N_q",
    "N_kv": "N_kv",
    "D": "D",
    "cu_seqlens_q": "CU_SEQLENS_Q",
    "cu_seqlens_kv": "CU_SEQLENS_KV",
    "seqused_q": "SEQUSED_Q",
    "seqused_kv": "SEQUSED_KV",
    "max_seqlen_q": "MAX_SEQLEN_Q",
    "max_seqlen_kv": "MAX_SEQLEN_KV",
    "batch_size": "BATCH_SIZE",
    "enable_pa": None,
    "kv_cache_layout": None,
    "block_size": None,
    "mask_mode": "SPARSE_MODE",
    "q_scale_layout": "Q_SCALE_LAYOUT",
    "p_scale": "P_SCALE",
    "softmax_scale": "SOFTMAX_SCALE",
    "data_range_q": "DATA_RANGE_Q",
    "data_range_k": "DATA_RANGE_K",
    "data_range_v": "DATA_RANGE_V",
    "enable_lse": "ENABLE_LSE",
    "input_layout": "INPUT_LAYOUT",
    "quant_mode": None,
    "device_id": "DEVICE_ID",
    "is_contiguous": "IS_CONTIGUOUS",
    "use_fp64_golden": "USE_FP64_GOLDEN",
}


def apply_params(params):
    for param_key, golden_attr in PARAM_MAP.items():
        if param_key in params and golden_attr is not None:
            setattr(golden, golden_attr, params[param_key])


def execute_test(params, mode, cdir=None):
    apply_params(params)
    case_name = params["name"]
    perf_only = "perf" in mode

    generate_input = "gen" in mode or (
        perf_only and not golden_cache.has_input(case_name, cache_dir=cdir)
    )
    if generate_input:
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

    if (
        "gen" in mode
        and "cpu" not in mode
        and "npu" not in mode
        and "compare" not in mode
    ):
        return None, None

    if "cpu" in mode:
        cpu_out, cpu_lse = golden.cpu_hif8_golden(
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
    elif "compare" in mode:
        cpu_out, cpu_lse = golden_cache.load_cpu_output(case_name, cache_dir=cdir)

    if "cpu" in mode and "npu" not in mode and "compare" not in mode:
        return None, None

    if "npu" in mode or perf_only:
        with torch.profiler.record_function(f"hif8_fa::{case_name}"):
            npu_out, lse_out = golden.npu_hif8_fa(
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
            )
        if perf_only:
            # Surface asynchronous device failures without copying outputs to CPU.
            torch.npu.synchronize()
            return None, None
        golden_cache.save_npu_output(case_name, npu_out, lse_out, cache_dir=cdir)
    else:
        npu_out, lse_out = golden_cache.load_npu_output(case_name, cache_dir=cdir)

    if "npu" in mode and "compare" not in mode:
        return None, None

    compare_layout = golden.INPUT_LAYOUT
    act_seqused_q = golden._get_seqused_q()

    def _to_cmp(tensor):
        if compare_layout == "TND":
            return golden.convert_q_bnsd_to_layout(
                tensor, act_seqused_q, compare_layout, cu_seqlens=golden.CU_SEQLENS_Q
            )
        return golden.convert_q_bnsd_to_layout(tensor, act_seqused_q, compare_layout)

    def _lse_to_cmp(lse_bnsd):
        if compare_layout == "TND":
            lse_cmp = _to_cmp(lse_bnsd)
            golden.fill_tnd_padding(
                lse_cmp, act_seqused_q, golden.CU_SEQLENS_Q, fill_value=float("inf")
            )
            # NPU LSE 输出为 N-major 排布 (N, T): N 在外, T 在内
            # CPU golden 经 convert 后是 [T, N, 1] (T-major), 需转成 [N, T] 对齐
            return lse_cmp.squeeze(-1).permute(1, 0).contiguous()
        # NPU LSE 非 TND 时固定为 (B,N,S) 3D (与 layout_out 无关),
        # BSND 也按 BNSD 对齐, 不随 compare_layout 转置
        lse_cmp = golden.convert_q_bnsd_to_layout(lse_bnsd, act_seqused_q, "BNSD")
        # golden LSE 带附加尾维 1 ((B,N,S,1)), 剥掉附加维;
        # S=1 时保留 (B,N,1) 与 NPU infershape 一致
        if lse_cmp.ndim > 1 and lse_cmp.shape[-1] == 1:
            lse_cmp = lse_cmp.squeeze(-1)
        return lse_cmp.contiguous()

    if golden.USE_FP64_GOLDEN:
        # 三方 cross_check(与 TTK 判据一致):
        #   golden      = FP64 bench(保留 P 的量化, 作为误差分母)
        #   third_party = 分段量化参考(竞品噪声上界)
        #   actual      = NPU
        # 序列长度统一走 _get_act_seq_*: BNSD/BSND 仅给 batch_size 的用例
        # SEQUSED_* 为 None, 直接用会触发 len(None)。
        act_seq_q = golden._get_act_seq_q()
        act_seq_kv = golden._get_act_seq_kv()
        bench_out, bench_lse = golden.cpu_hif8_bench(
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
        # 分段参考复刻 kernel 的 P 量化分段行为, 作为分母基准
        ref_out, ref_lse = golden.cpu_hif8_sectioned_golden(
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
        golden_cmp = _to_cmp(bench_out)
        # 与 TTK 一致: 分段参考 cast bf16, 使其量化噪声水平与 target 可比
        benchmark_cmp = _to_cmp(ref_out).to(torch.bfloat16)
        atten_result = _CV.cross_check_compare(
            npu_out,
            golden_cmp,
            benchmark_cmp,
            # 容差档按 NPU 输出 dtype 取(与 quant_flash_mla / flash_attn 口径一致):
            # bf16 输出 → small_value=2^-8/atol=2^-16; 不能按 fp64 bench 取档
            output_dtype=npu_out.dtype,
            test_name=f"qfa_hif8_attention_out::{case_name}",
        )
    else:
        cpu_cmp = _to_cmp(cpu_out)
        atten_result = result_compare_method.check_result(cpu_cmp, npu_out)

    lse_result = None
    if golden.ENABLE_LSE:
        if golden.USE_FP64_GOLDEN:
            lse_result = _CV.cross_check_compare(
                lse_out,
                _lse_to_cmp(bench_lse),
                _lse_to_cmp(ref_lse),
                output_dtype=lse_out.dtype,
                test_name=f"qfa_hif8_lse::{case_name}",
            )
        else:
            lse_cmp = _lse_to_cmp(cpu_lse)
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
