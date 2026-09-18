# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DN mask 边界扩展: G=1, 整行不可见(空行), 零/负 scale."""

TestCases = {}


def add(name, **overrides):
    case = dict(
        B=[2],
        N1=[8],
        N2=[2],
        S1=[145],
        S2=[257],
        D=[128],
        DV=[128],
        Dtype=["fp16", "bf16"],
        layout_q=["BSND"],
        layout_kv=["BSND"],
        layout_out=["BSND"],
        mask_mode=[3],
        win_left=[-1],
        win_right=[-1],
        return_softmax_lse=[True],
        q_range=[(-1.0, 1.0)],
        k_range=[(-1.0, 1.0)],
        v_range=[(-1.0, 1.0)],
    )
    case.update(overrides)
    TestCases[name] = case


# G=1: 无 GQA 合并收益, 直接命中单头
add("DNM_EXT_G1", N1=[2], N2=[2])
# 整行不可见: S1 > S2 的因果 mask 上三角行全部掩蔽
add("DNM_EXT_EMPTY_CAUSAL_ROWS", S1=[257], S2=[129])
# 整行不可见: 窄 band 窗口
add(
    "DNM_EXT_EMPTY_BAND_ROWS",
    S1=[257],
    S2=[129],
    mask_mode=[4],
    win_left=[17],
    win_right=[9],
)
# 零/负 scale: 必须先施加 scale 再写 padding 哨兵
add("DNM_EXT_ZERO_SCALE", scale=[0.0])
add("DNM_EXT_NEGATIVE_SCALE", scale=[-0.125])
