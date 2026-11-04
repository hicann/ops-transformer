#!/usr/bin/python
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

"""Q_N 1~128 大交叉泛化用例集（1024 条）。

覆盖：
  - layout_query/layout_kv: BSND/BSND、TND/TND、BSND/PA_BSND、TND/PA_BSND
  - Q_N: 1~128 全区间采样，含奇数、2 的幂、非对齐值
  - q_type: bfloat16 / float16
  - kv_type: float8_e4m3fn（None）/ hifloat8
  - return_softmax_lse: True / False（覆盖 softmax_max / softmax_sum 输出）
  - sinks: True / False
  - PA_BSND 下 key/value 0 轴非连续（key_stride）
  - sparse_mode: 0 / 3
  - shape 非对齐：S1/S2 取奇数与非 16 对齐值，block_size 取 16/32/128

执行：
  bash test_run.sh single -P sparse_flash_attention_antiquant_v2_paramset_general
"""

import torch

LAYOUTS = [
    ("BSND", "BSND"),
    ("TND", "TND"),
    ("BSND", "PA_BSND"),
    ("TND", "PA_BSND"),
]

# 覆盖 1~128 边界、2 的幂、奇数、非对齐值
N1_VALUES = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    15,
    16,
    17,
    31,
    32,
    33,
    47,
    48,
    49,
    63,
    64,
    65,
    66,
    67,
    79,
    80,
    81,
    95,
    96,
    97,
    111,
    127,
    128,
]

# 每类 profile 交叉组合 dtype / sinks / lse / sparse_mode / shape / key_stride
PROFILES = [
    dict(
        q_type=torch.bfloat16,
        kv_type=None,
        sinks=False,
        lse=False,
        sparse_mode=3,
        batch=2,
        s1=3,
        s2=192,
        k=None,
        block_size=128,
        key_stride=0,
    ),
    dict(
        q_type=torch.bfloat16,
        kv_type=None,
        sinks=True,
        lse=True,
        sparse_mode=0,
        batch=2,
        s1=7,
        s2=255,
        k=None,
        block_size=16,
        key_stride=0,
    ),
    dict(
        q_type=torch.float16,
        kv_type=None,
        sinks=False,
        lse=True,
        sparse_mode=3,
        batch=1,
        s1=1,
        s2=128,
        k=None,
        block_size=128,
        key_stride=0,
    ),
    dict(
        q_type=torch.float16,
        kv_type=None,
        sinks=True,
        lse=False,
        sparse_mode=0,
        batch=2,
        s1=5,
        s2=129,
        k=None,
        block_size=16,
        key_stride=0,
    ),
    dict(
        q_type=torch.bfloat16,
        kv_type="hifloat8",
        sinks=False,
        lse=True,
        sparse_mode=3,
        batch=2,
        s1=3,
        s2=192,
        k=2048,
        block_size=128,
        key_stride=0,
    ),
    dict(
        q_type=torch.float16,
        kv_type="hifloat8",
        sinks=True,
        lse=False,
        sparse_mode=0,
        batch=1,
        s1=7,
        s2=255,
        k=2048,
        block_size=16,
        key_stride=0,
    ),
    dict(
        q_type=torch.bfloat16,
        kv_type=None,
        sinks=False,
        lse=False,
        sparse_mode=0,
        batch=2,
        s1=9,
        s2=63,
        k=None,
        block_size=16,
        key_stride=16,
    ),
    dict(
        q_type=torch.float16,
        kv_type=None,
        sinks=True,
        lse=True,
        sparse_mode=3,
        batch=2,
        s1=2,
        s2=257,
        k=None,
        block_size=32,
        key_stride=32,
    ),
]


def _seq_lengths(layout_query, layout_kv, batch, s1, s2):
    """生成与 layout 匹配的 actual_seq_q / actual_seq_kv。"""
    if layout_query == "TND":
        q_lens = [1] + [s1] * (batch - 1)
        actual_seq_q = []
        acc = 0
        for length in q_lens:
            acc += length
            actual_seq_q.append(acc)
    else:
        actual_seq_q = [1] + [s1] * (batch - 1)

    kv_lens = [max(1, s2 // 2)] + [s2] * (batch - 1)
    if layout_kv == "TND":
        actual_seq_kv = []
        acc = 0
        for length in kv_lens:
            acc += length
            actual_seq_kv.append(acc)
    else:
        actual_seq_kv = kv_lens
    return actual_seq_q, actual_seq_kv


def _make_case(name, n1, layout_query, layout_kv, profile, k, key_stride):
    batch = profile["batch"]
    s1 = profile["s1"]
    s2 = profile["s2"]
    actual_seq_q, actual_seq_kv = _seq_lengths(layout_query, layout_kv, batch, s1, s2)
    return {
        "Testcase_Prefix": [name],
        "layout_query": [layout_query],
        "layout_kv": [layout_kv],
        "q_type": [profile["q_type"]],
        "kv_type": [profile["kv_type"]],
        "B": [batch],
        "T1": [None],
        "T2": [None],
        "S1": [s1],
        "S2": [s2],
        "N1": [n1],
        "N2": [1],
        "D": [512],
        "K": [k],
        "scale_value": [0.041666666666666664],
        "key_quant_mode": [2],
        "value_quant_mode": [2],
        "sparse_block_size": [1],
        "tile_size": [128],
        "rope_head_dim": [64],
        "sparse_mode": [profile["sparse_mode"]],
        "attention_mode": [2],
        "quant_scale_repo_mode": [1],
        "block_size": [profile["block_size"]],
        "block_num": [None],
        "actual_seq_q": [actual_seq_q],
        "actual_seq_kv": [actual_seq_kv],
        "enable_sinks": [profile["sinks"]],
        "return_softmax_lse": [profile["lse"]],
        "key_stride": [key_stride],
    }


def build_cases():
    cases = []
    idx = 0
    for layout_id, (layout_query, layout_kv) in enumerate(LAYOUTS):
        for n1 in N1_VALUES:
            for profile_id, profile in enumerate(PROFILES):
                k = profile["k"] if profile["k"] is not None else profile["s2"]
                # key_stride 仅在 PA_BSND 场景生效
                key_stride = profile["key_stride"] if layout_kv == "PA_BSND" else 0
                cases.append(
                    _make_case(
                        f"gen_l{layout_id}_p{profile_id}",
                        n1,
                        layout_query,
                        layout_kv,
                        profile,
                        k,
                        key_stride,
                    )
                )
                idx += 1
    print(f"[general paramset] built {idx} cases")
    return cases


ENABLED_PARAMS = build_cases()
