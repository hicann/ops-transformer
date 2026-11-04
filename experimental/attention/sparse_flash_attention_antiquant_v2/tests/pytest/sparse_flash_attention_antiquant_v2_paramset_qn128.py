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

"""Q_N 1~128 泛化验证用例集。

覆盖:
  - layout_query/layout_kv: BSND/BSND、TND/TND、BSND/PA_BSND、TND/PA_BSND
  - Q_N: 1~128，含奇数、2 的幂、非对齐值（如 3/17/33/49/65/95/127）
  - q_type: bfloat16 / float16
  - kv_type: float8_e4m3fn / hifloat8
  - return_softmax_lse: True / False
  - sinks: True / False
  - PA_BSND 下 key/value 0 轴非连续（key_stride）
  - sparse_mode: 0 / 3
  - shape 非对齐: B/S1/S2/K/block_size 组合

ENABLED_PARAMS 中的每一项都会被 utils.combin_params 展成一个用例。
"""

import torch

LAYOUTS = [
    ("BSND", "BSND"),
    ("BSND", "PA_BSND"),
    ("TND", "TND"),
    ("TND", "PA_BSND"),
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


def _make_case(
    name,
    n1,
    layout_query,
    layout_kv,
    q_type,
    kv_type,
    batch,
    s1,
    s2,
    k,
    block_size,
    sinks,
    lse,
    sparse_mode,
    key_stride,
):
    actual_seq_q, actual_seq_kv = _seq_lengths(layout_query, layout_kv, batch, s1, s2)
    return {
        "Testcase_Prefix": [name],
        "layout_query": [layout_query],
        "layout_kv": [layout_kv],
        "q_type": [q_type],
        "kv_type": [kv_type],
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
        "sparse_mode": [sparse_mode],
        "attention_mode": [2],
        "quant_scale_repo_mode": [1],
        "block_size": [block_size],
        "block_num": [None],
        "actual_seq_q": [actual_seq_q],
        "actual_seq_kv": [actual_seq_kv],
        "enable_sinks": [sinks],
        "return_softmax_lse": [lse],
        "key_stride": [key_stride],
    }


def build_cases():
    cases = []
    idx = 0

    # Group A: Q_N 全量扫描（含奇数、非对齐、2 的幂），四种 layout 交叉。
    n1_values = [
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
        112,
        113,
        126,
        127,
        128,
    ]
    for n1 in n1_values:
        for layout_id, (layout_query, layout_kv) in enumerate(LAYOUTS):
            q_type = torch.bfloat16 if idx % 2 == 0 else torch.float16
            sinks = layout_id % 2 == 1
            lse = layout_id % 2 == 0
            sparse_mode = 3 if idx % 4 else 0
            batch = 2
            s1 = 7 if idx % 3 == 0 else 3
            s2 = 192 if idx % 2 == 0 else 255
            block_size = 16 if idx % 2 == 0 else 128
            key_stride = (
                16 if layout_kv == "PA_BSND" and (idx + layout_id) % 2 == 1 else 0
            )
            cases.append(
                _make_case(
                    f"qn128_a{idx:04d}",
                    n1,
                    layout_query,
                    layout_kv,
                    q_type,
                    None,
                    batch,
                    s1,
                    s2,
                    s2,
                    block_size,
                    sinks,
                    lse,
                    sparse_mode,
                    key_stride,
                )
            )
            idx += 1

    # Group B: 功能交叉（layout × Q_N 边界 × dtype × sinks × lse × sparse_mode）。
    n1_cross = [3, 65, 95, 127]
    for layout_id, (layout_query, layout_kv) in enumerate(LAYOUTS):
        for n1 in n1_cross:
            for q_type in (torch.bfloat16, torch.float16):
                for kv_type in (None, "float8_e4m3fn"):
                    for sinks in (False, True):
                        for lse in (False, True):
                            for sparse_mode in (0, 3):
                                batch = 1 if idx % 3 == 0 else 2
                                s1 = (1, 3, 5)[idx % 3]
                                s2 = (128, 192, 255)[idx % 3]
                                block_size = (16, 128)[idx % 2]
                                key_stride = (
                                    16 if layout_kv == "PA_BSND" and idx % 2 == 1 else 0
                                )
                                cases.append(
                                    _make_case(
                                        f"qn128_b{idx:04d}",
                                        n1,
                                        layout_query,
                                        layout_kv,
                                        q_type,
                                        kv_type,
                                        batch,
                                        s1,
                                        s2,
                                        s2,
                                        block_size,
                                        sinks,
                                        lse,
                                        sparse_mode,
                                        key_stride,
                                    )
                                )
                                idx += 1

    # Group C: hifloat8 输入（sparse_block_count 固定 2048）。
    for layout_id, (layout_query, layout_kv) in enumerate(LAYOUTS):
        for n1 in (1, 65, 95, 127):
            sinks = layout_id % 2 == 1
            lse = layout_id % 2 == 0
            key_stride = 16 if layout_kv == "PA_BSND" and n1 in (65, 127) else 0
            cases.append(
                _make_case(
                    f"qn128_c{idx:04d}",
                    n1,
                    layout_query,
                    layout_kv,
                    torch.bfloat16,
                    "hifloat8",
                    1,
                    1,
                    64,
                    2048,
                    16,
                    sinks,
                    lse,
                    3,
                    key_stride,
                )
            )
            idx += 1

    return cases


ENABLED_PARAMS = build_cases()
