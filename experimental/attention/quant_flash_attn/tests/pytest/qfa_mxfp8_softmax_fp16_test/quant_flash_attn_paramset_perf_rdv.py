#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 性能/压力用例集：参考主线 qfa_fp8_test/quant_flash_attn_fp8_paramset_perf_rdv.py 的
# 6 个大序列 serving 场景迁移而来（B1 G8 Nq16 Nkv2 D128，Q/KVS 16K~61K，p_scale=256）。
#
# 布局适配（fp8 → 本算子 quant_mode=3）：
#   fp8 原版: PA 分页 + TND/NTD varlen（cu_seqlens + block_table + mask_mode=3）
#   本  版:   BNSD + seqused（golden 单模板只支持 BNSD；变长语义由 seqused 承载，
#             mask_mode=0，enable_pa=False）。计算规模（B/N_q/N_kv/序列长度/p_scale）
#             与 fp8 原版逐项一致，GQA G=8。
#
# 运行方式（典型）：
#   全流程（含 CPU golden 逐位对比，首次运行建议执行以填充缓存）:
#     pytest -v -m perf_rdv
#   纯性能采集（跳过 CPU golden，msprof 包裹 + 自动解析 Duration 报告；需先跑过全流程）:
#     pytest -v -m perf_rdv --golden-mode=npu --msprof
#   与基线比较（劣化超 8% 判 FAILED）:
#     pytest -v -m perf_rdv --golden-mode=npu --msprof --perf-baseline=./perf_output/xxx.log

from quant_flash_attn_paramset_common import expand_paramset_to_cases

# 6 个用例共享的固定骨架（仅 seqused_q/seqused_kv 逐例不同）
_PERF_COMMON = {
    "B": [1],
    "N_q": [16],
    "N_kv": [2],
    "D": [128],
    "cu_seqlens_q": [None],
    "cu_seqlens_kv": [None],
    "enable_pa": [False],
    "kv_cache_layout": [None],
    "block_size": [None],
    "mask_mode": [0],
    "q_scale_layout": ["BNSD"],
    "p_scale": [256.0],
    "enable_lse": [False],
}

TEST_PARAMS = {
    "B1_G8_Nq16_Nkv2_D128_SM0_LSE0_Q16384_KVS16384_P256_Perf": {
        **_PERF_COMMON,
        "seqused_q": [[16384]],
        "seqused_kv": [[16384]],
    },
    "B1_G8_Nq16_Nkv2_D128_SM0_LSE0_Q12544_KVS28928_P256_Perf": {
        **_PERF_COMMON,
        "seqused_q": [[12544]],
        "seqused_kv": [[28928]],
    },
    "B1_G8_Nq16_Nkv2_D128_SM0_LSE0_Q10496_KVS39424_P256_Perf": {
        **_PERF_COMMON,
        "seqused_q": [[10496]],
        "seqused_kv": [[39424]],
    },
    "B1_G8_Nq16_Nkv2_D128_SM0_LSE0_Q9216_KVS48640_P256_Perf": {
        **_PERF_COMMON,
        "seqused_q": [[9216]],
        "seqused_kv": [[48640]],
    },
    "B1_G8_Nq16_Nkv2_D128_SM0_LSE0_Q8320_KVS56960_P256_Perf": {
        **_PERF_COMMON,
        "seqused_q": [[8320]],
        "seqused_kv": [[56960]],
    },
    "B1_G8_Nq16_Nkv2_D128_SM0_LSE0_Q4040_KVS61000_P256_Perf": {
        **_PERF_COMMON,
        "seqused_q": [[4040]],
        "seqused_kv": [[61000]],
    },
}

CASES = expand_paramset_to_cases(TEST_PARAMS)
SKIP_CASES = set()
