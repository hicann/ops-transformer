# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DN 动态 M=64(mBase=64, s2Base=256) 专项: 全注册 D/DV 组合 x mask, 变长尾块."""

TestCases = {}
for d, dv in ((64, 64), (72, 72), (128, 128), (192, 128), (256, 256)):
    for mode in (0, 3, 4):
        TestCases[f"DNM_M64_D{d}_MASK{mode}"] = dict(
            B=[1],
            N1=[8],
            N2=[2],
            S1=[64],
            S2=[257],
            D=[d],
            DV=[dv],
            Dtype=["fp16", "bf16"],
            layout_q=["BSND"],
            layout_kv=["BSND"],
            layout_out=["BSND"],
            mask_mode=[mode],
            win_left=[133 if mode == 4 else -1],
            win_right=[17 if mode == 4 else -1],
            return_softmax_lse=[True],
            q_range=[(-1.0, 1.0)],
            k_range=[(-1.0, 1.0)],
            v_range=[(-1.0, 1.0)],
        )

for mode in (0, 3, 4):
    TestCases[f"DNM_M64_VARLEN_MASK{mode}"] = dict(
        B=[4],
        N1=[8],
        N2=[2],
        S1=[115],
        S2=[1028],
        D=[128],
        DV=[128],
        Dtype=["fp16", "bf16"],
        layout_q=["TND"],
        layout_kv=["TND"],
        layout_out=["TND"],
        cu_seqlens_q=[[0, 1, 18, 51, 115]],
        cu_seqlens_kv=[[0, 257, 514, 771, 1028]],
        mask_mode=[mode],
        win_left=[133 if mode == 4 else -1],
        win_right=[17 if mode == 4 else -1],
        return_softmax_lse=[True],
        q_range=[(-1.0, 1.0)],
        k_range=[(-1.0, 1.0)],
        v_range=[(-1.0, 1.0)],
    )
