# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DN 全路由 mask 专项: 压缩mask(3/4) x 布局, 全维度注册表(D/DV), FD, TND变长, 输出布局转换."""

TestCases = {}


def add(name, s1=128, s2=257, layout="BSND", dim=64, dv=None, mask=0, **kwargs):
    case = dict(
        B=[1],
        N1=[8],
        N2=[2],
        S1=[s1],
        S2=[s2],
        D=[dim],
        DV=[dv or dim],
        Dtype=["fp16", "bf16"],
        layout_q=[layout],
        layout_kv=[layout],
        layout_out=[layout],
        mask_mode=[mask],
        win_left=[-1],
        win_right=[-1],
        return_softmax_lse=[True],
        q_range=[(-1.0, 1.0)],
        k_range=[(-1.0, 1.0)],
        v_range=[(-1.0, 1.0)],
    )
    case.update(kwargs)
    TestCases[name] = case


# BSND 边界(UNMV 已覆盖 BNSD): mBase=64/128 边界与尾块
for s1 in (63, 64, 127, 128, 129, 145, 257):
    add(f"DNM_BOUNDARY_BSND_S{s1}", s1=s1)

# 压缩 mask: causal(3) 与 band(4) x 布局
for layout in ("BSND", "BNSD"):
    for mode in (3, 4):
        add(
            f"DNM_MASK_{layout}_M{mode}",
            s1=145,
            layout=layout,
            mask=mode,
            win_left=[33] if mode == 4 else [-1],
            win_right=[17] if mode == 4 else [-1],
        )

# 全维度注册表 x mask: 72/128/192-128/256(覆盖 config 2/4/6 + 72 pad 路径)
for layout in ("BSND", "BNSD"):
    for d, dv in ((72, 72), (128, 128), (192, 128), (256, 256)):
        add(f"DNM_DIM_{layout}_D{d}", s1=145, layout=layout, dim=d, dv=dv, mask=3)

# FlashDecode(S2 外切): 含 mask 变体
for layout in ("BSND", "BNSD"):
    add(f"DNM_FD_{layout}", s1=129, s2=8193, layout=layout, N1=[2], N2=[1])
add("DNM_FD_MASK_BSND", s1=129, s2=8193, N1=[2], N2=[1], mask=3)
add(
    "DNM_FD_LSE_BNSD",
    s1=129,
    s2=8193,
    layout="BNSD",
    N1=[4],
    N2=[2],
    mask=4,
    win_left=[65],
    win_right=[33],
)

# TND 变长: 无 mask / causal
add(
    "DNM_TND_VARLEN",
    s1=290,
    s2=601,
    layout="TND",
    B=[3],
    cu_seqlens_q=[[0, 1, 146, 290]],
    cu_seqlens_kv=[[0, 63, 320, 601]],
)
add(
    "DNM_TND_VARLEN_MASK",
    s1=290,
    s2=601,
    layout="TND",
    B=[3],
    mask=3,
    cu_seqlens_q=[[0, 1, 146, 290]],
    cu_seqlens_kv=[[0, 63, 320, 601]],
)

# BNSD 输入 -> BSND 输出(布局转换)
add("DNM_BNSD_TO_BSND", s1=145, layout="BNSD", layout_out=["BSND"], mask=3)

# sink(修复 GS 重编码): BNSD g=4, 无 mask, LSE 开
add(
    "DNM_SINK_BNSD_G4",
    s1=145,
    layout="BNSD",
    N1=[8],
    N2=[2],
    enable_learnable_sink=[True],
)
