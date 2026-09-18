# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DN 全量路由覆盖矩阵: 维度注册表 x 布局 x mask(0/3/4) x GQA x S1边界 x M64 x varlen x PA x FD x sink x scale.

DN 路由条件(与 host/metadata 共享判定一致): 全量路由, 所有 case 均走 DN 模板,
维度注册表(等维 64/72/128/256 或 192/128)仅作未注册维度的防御性回退。
历史版本曾有 maxSeqQ<mBase 短Q回退 ND(用例名中 _NDFB 后缀为当时对照标记, 现同样路由 DN)。
"""

DIMS = ((64, 64), (72, 72), (128, 128), (192, 128), (256, 256))
TestCases = {}


def add(
    name,
    s1=145,
    s2=257,
    layout="BSND",
    dim=128,
    dv=None,
    mask=0,
    b=1,
    n1=8,
    n2=2,
    **kwargs,
):
    case = dict(
        B=[b],
        N1=[n1],
        N2=[n2],
        S1=[s1],
        S2=[s2],
        D=[dim],
        DV=[dv or dim],
        Dtype=["fp16", "bf16"],
        layout_q=[layout],
        layout_kv=[kwargs.pop("layout_kv", layout)],
        layout_out=[kwargs.pop("layout_out", layout)],
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


# A. 布局 x 全维度注册表 (mask0, S1=145 >= mBase=128 -> DN)
for layout in ("BSND", "BNSD"):
    for d, dv in DIMS:
        add(f"FULL_A_DIM_{layout}_D{d}", s1=145, s2=257, layout=layout, dim=d, dv=dv)

# B. 压缩 mask(3/4) x 布局 x 维度 (S1=145 -> DN)
for layout in ("BSND", "BNSD"):
    for d, dv in ((64, 64), (128, 128), (192, 128), (256, 256)):
        for mode in (3, 4):
            add(
                f"FULL_B_MASK{mode}_{layout}_D{d}",
                s1=145,
                s2=257,
                layout=layout,
                dim=d,
                dv=dv,
                mask=mode,
                win_left=[33] if mode == 4 else [-1],
                win_right=[17] if mode == 4 else [-1],
            )

# C. band 窗口宽度变化 (D128, mask4, S1=145 -> DN)
for wl, wr in ((129, 1), (65, 33), (133, 17), (257, 65), (1, 1)):
    add(
        f"FULL_C_BAND_W{wl}_{wr}", s1=145, s2=300, mask=4, win_left=[wl], win_right=[wr]
    )

# D. GQA g=1/2/4/8 x 布局 x mask (S1=145 -> DN)
for n1, n2, g in ((4, 4, 1), (8, 4, 2), (8, 2, 4), (16, 2, 8)):
    for layout in ("BSND", "BNSD"):
        for mode in (0, 3):
            add(
                f"FULL_D_G{g}_{layout}_M{mode}",
                s1=145,
                s2=257,
                layout=layout,
                n1=n1,
                n2=n2,
                mask=mode,
            )

# E. S1 边界扫描 (mask3; S1=64 -> sOuter=32 -> mBase=64 -> DN, 65..127 -> mBase=128 -> NDFB 对照, >=128 -> DN)
for s1 in (64, 65, 96, 127, 128, 129, 160, 191, 192, 255, 256, 257, 320, 384, 512):
    tag = "DN" if (s1 == 64 or s1 >= 128) else "NDFB"
    add(f"FULL_E_S1_{s1}_{tag}", s1=s1, s2=300, mask=3)

# F. M=64 动态 (S1=64, S2=257; mask3/4 -> sOuter=32 -> mBase=64 -> DN; D256 因 g*S1>=64 走 mBase=128 -> NDFB)
for d, dv in DIMS:
    for mode in (3, 4):
        tag = "NDFB" if d == 256 else "DN"
        add(
            f"FULL_F_M64_{tag}_D{d}_M{mode}",
            s1=64,
            s2=257,
            dim=d,
            dv=dv,
            mask=mode,
            win_left=[133] if mode == 4 else [-1],
            win_right=[17] if mode == 4 else [-1],
        )

# G. TND varlen x mask (maxSeqQ = max(段长) >= 128 -> DN)
VARLEN = (
    ([0, 1, 146, 290], [0, 63, 320, 601]),
    ([0, 128, 256, 384, 512], [0, 257, 514, 771, 1028]),
    ([0, 1, 512], [0, 1024, 1025]),
    ([0, 130, 260, 390, 520], [0, 5, 129, 777, 1500]),
)
for i, (cu_q, cu_kv) in enumerate(VARLEN, 1):
    for mode in (0, 3, 4):
        add(
            f"FULL_G_VARLEN{i}_M{mode}",
            layout="TND",
            b=len(cu_q) - 1,
            cu_seqlens_q=[cu_q],
            cu_seqlens_kv=[cu_kv],
            mask=mode,
            win_left=[33] if mode == 4 else [-1],
            win_right=[17] if mode == 4 else [-1],
        )

# H. PA 分页 KV x 布局 x 维度 x mask (S1=128 == mBase -> DN; S2=1024 整块对齐; NZ 仅 D64 已知合法)
for pa, pa_dims in (("PA_BBND", (64, 128)), ("PA_BNBD", (64, 128)), ("PA_NZ", (64,))):
    for d in pa_dims:
        for mode in (0, 3):
            add(
                f"FULL_H_{pa}_D{d}_M{mode}",
                s1=128,
                s2=1024,
                layout="BNSD",
                layout_kv=pa,
                dim=d,
                dv=d,
                mask=mode,
                block_size=[128],
            )

# I. FlashDecode 长序列 (S2=4096/8193, S1=129 -> DN)
for s2 in (4096, 8193):
    for layout in ("BSND", "BNSD"):
        for mode in (0, 3, 4):
            add(
                f"FULL_I_FD_S2_{s2}_{layout}_M{mode}",
                s1=129,
                s2=s2,
                layout=layout,
                n1=2,
                n2=1,
                mask=mode,
                win_left=[65] if mode == 4 else [-1],
                win_right=[33] if mode == 4 else [-1],
            )
add(
    "FULL_I_FD_TND_M3",
    s1=129,
    s2=6144,
    layout="TND",
    b=2,
    n1=2,
    n2=1,
    mask=3,
    cu_seqlens_q=[[0, 129, 258]],
    cu_seqlens_kv=[[0, 3072, 6144]],
)

# J. learnable sink x 布局 x GQA x mask (S1=145 -> DN)
for layout in ("BSND", "BNSD"):
    for n1, n2, g in ((4, 4, 1), (8, 2, 4)):
        for mode in (0, 3):
            add(
                f"FULL_J_SINK_G{g}_{layout}_M{mode}",
                s1=145,
                s2=257,
                layout=layout,
                n1=n1,
                n2=n2,
                mask=mode,
                enable_learnable_sink=[True],
            )

# K. scale 特殊值 x mask (S1=145 -> DN; 零/负 scale 走先缩放后 padding 路径)
for scale in (0.0, -0.125, 2.0):
    for mode in (0, 3, 4):
        add(
            f"FULL_K_SCALE{scale}_M{mode}",
            s1=145,
            s2=257,
            mask=mode,
            scale=[scale],
            win_left=[33] if mode == 4 else [-1],
            win_right=[17] if mode == 4 else [-1],
        )

# L. 输出布局转换 (q BNSD -> out BSND 及反向; S1=145 -> DN)
for mode in (0, 3, 4):
    add(
        f"FULL_L_BNSD_TO_BSND_M{mode}",
        s1=145,
        layout="BNSD",
        layout_out="BSND",
        mask=mode,
        win_left=[33] if mode == 4 else [-1],
        win_right=[17] if mode == 4 else [-1],
    )
for mode in (0, 3):
    add(
        f"FULL_L_BSND_TO_BNSD_M{mode}",
        s1=145,
        layout="BSND",
        layout_out="BNSD",
        mask=mode,
    )

# M. 整行不可见 + 多 batch (S1>S2 上三角空行; B>1 跨 batch 元数据; S1=257 -> DN)
add("FULL_M_EMPTY_CAUSAL", s1=257, s2=129, mask=3)
add("FULL_M_EMPTY_BAND", s1=257, s2=129, mask=4, win_left=[17], win_right=[9])
for mode in (0, 3):
    for b in (3, 5):
        add(f"FULL_M_B{b}_M{mode}", s1=145, s2=257, b=b, mask=mode)
