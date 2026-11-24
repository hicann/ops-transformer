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

# 仿真专用用例集: 只为看流水图, 不参与性能统计.
# 组织方式: 同形状下 mask3 / mask0 成对, layout 为 TND(目标是 TND sparse3);
# 另留一条同形状 BNSD causal 作 layout 对照. 每次仿真选一条跑.
#   形状A QS=1024 KVS=8192 (nxt=7168): 因果只裁掉 3% 的 KV 块 -> 量 mask 自身开销
#   形状B QS=KVS=4096  (nxt=0)    : 因果裁掉约一半 KV 块 -> 量任务变短对流水填排的影响

from quant_flash_attn_paramset_common import expand_paramset_to_cases

_SIM_COMMON = {
    "D": [128],
    "cu_seqlens_q": [None],
    "cu_seqlens_kv": [None],
    "enable_pa": [False],
    "kv_cache_layout": [None],
    "block_size": [None],
    "q_scale_layout": ["BNSD"],
    "p_scale": [1.0],
    "enable_lse": [True],
}

TEST_PARAMS = {
    # ---- 形状A: QS=1024 / KVS=8192 (nxt=7168) ----
    # 因果几乎裁不掉工作(124 vs 128 个 KV 块), 用来量 mask 本身的额外开销.
    "MxFP8_Sim_TND_QS1024_KVS8192_Causal": {
        **_SIM_COMMON,
        "input_layout": ["TND"],
        "q_scale_layout": ["TND"],
        "mask_mode": [3],
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "seqused_q": [[1024]],
        "seqused_kv": [[8192]],
    },
    "MxFP8_Sim_TND_QS1024_KVS8192_Mask0": {
        **_SIM_COMMON,
        "input_layout": ["TND"],
        "q_scale_layout": ["TND"],
        "mask_mode": [0],
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "seqused_q": [[1024]],
        "seqused_kv": [[8192]],
    },
    # ---- 形状B: QS=KVS=4096 (nxt=0) ----
    # 因果真裁掉约一半 KV 块, 用来量"任务变短"对流水填排的影响.
    "MxFP8_Sim_TND_QS4096_KVS4096_Causal": {
        **_SIM_COMMON,
        "input_layout": ["TND"],
        "q_scale_layout": ["TND"],
        "mask_mode": [3],
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "seqused_q": [[4096]],
        "seqused_kv": [[4096]],
    },
    "MxFP8_Sim_TND_QS4096_KVS4096_Mask0": {
        **_SIM_COMMON,
        "input_layout": ["TND"],
        "q_scale_layout": ["TND"],
        "mask_mode": [0],
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "seqused_q": [[4096]],
        "seqused_kv": [[4096]],
    },
    # ---- BNSD 对照: 同形状A, 用来看 layout 是否引入额外停顿 ----
    "MxFP8_Sim_BNSD_QS1024_KVS8192_Causal": {
        **_SIM_COMMON,
        "mask_mode": [3],
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "seqused_q": [[1024]],
        "seqused_kv": [[8192]],
    },
}

CASES = expand_paramset_to_cases(TEST_PARAMS)
SKIP_CASES = set()
