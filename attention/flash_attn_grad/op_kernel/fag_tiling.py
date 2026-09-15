# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""host <-> kernel 契约。

两个类各自钉住一份跨语言契约，改动必须与 host 同步：

* :class:`FlashAttnGradTilingData` —— 字段**顺序**即 GM 上的内存布局，
  与 ``op_host/flash_attn_grad_tiling.h`` 的 ``TILING_DATA_FIELD_DEF`` 逐一对应。
* :class:`FlashAttnGradTilingKey` —— 位域编码，与
  ``op_host/plan/flash_attn_grad_tiling_key.cpp`` 对应；``is_valid`` 决定
  最终编出多少份二进制。
"""

from dataclasses import dataclass

from pypto_pro.runtime.tilingkey import TilingKeyField


@dataclass
class FlashAttnGradTilingData:
    """对齐 op_host/flash_attn_grad_tiling.h 的 TILING_DATA_FIELD_DEF 顺序。

    字段名与顺序必须与 host 侧逐一对应 —— codegen 出的
    FlashAttnGradTilingData_tiling.h 是按这里的声明顺序排布内存的。
    """

    b: int
    s1: int
    s2: int
    n1: int
    n2: int
    d: int
    dv: int
    scaleValue: float
    # ---- 布局视图参数 ----
    # BSND 与 BNSD 统一成同一种四维紧凑视图 [viewD0, S, viewD2, D]，
    # 这样 kernel 只需一份 pl.load(order=...) 代码，布局差异全部退化成数值：
    #   BSND [B,S,N,D] -> 视图 [B,   S, N, D]，索引 [b,     s, n, 0]
    #   BNSD [B,N,S,D] -> 视图 [B*N, S, 1, D]，索引 [b*N+n, s, 0, 0]
    #   idx0 = b*coefB0 + n*coefN0 ; idx2 = n*coefN2
    #   BSND: coefB0=1, coefN0=0, coefN2=1
    #   BNSD: coefB0=N, coefN0=1, coefN2=0
    # 运行期纯算术、无分支，故不占 tilingkey 位(不增大算子二进制)。
    # 注意不能改用「倒置 stride」的办法：make_tensor 对非紧凑 stride 会
    # 静默忽略并按紧凑跨距去读，不报错(实测见 FAG_PYPTO_DEV.md)。
    # GQA 组大小：n1 = n2 * gSize。gSize=1 即普通 MHA。
    # Q/dout/attn_out/dq 有 n1 个 head，K/V/dk/dv 只有 n2 个，
    # 故视图参数分 Q / KV 两套（coefN0/coefN2 只取决于 layout，两套共用）。
    gSize: int
    viewD0Q: int
    viewD2Q: int
    coefB0Q: int
    viewD0KV: int
    viewD2KV: int
    coefB0KV: int
    coefN0: int
    coefN2: int
    maskMode: int
    winLeft: int
    winRight: int
    sparseType: int
    s1Token: int
    s2Token: int
    totalPerBatchNum: int


# 与 op_kernel/flash_attn_grad.py 的 FlashAttnGradTilingKey 一致
class FlashAttnGradTilingKey:
    # bit[1:0]：0=BN2GS1S2，1=BN2（含 MultiBlk，bit9 区分），
    # 2=未用，3=BN2S2 预留（本轮不编、不进入口）。
    template = TilingKeyField(bits=2, values=[0, 1, 2, 3])
    layout = TilingKeyField(bits=1, values=[0, 1])
    # 0=无 attenmask，3=causal，4=band。host 存 0/1/2 索引。
    mask_mode = TilingKeyField(bits=2, values=[0, 3, 4])
    # 分核方式：0=线性均分，1=swizzle(按 s2 列跨核发放)。
    # 由 host DecideSwizzle() 依输入 shape 判定「同时在跑的核其工作集
    # 是否超 L2」后置位，两种分核各编译一份二进制，运行期零分支开销。
    # 判据与实测见 op_host/plan/flash_attn_grad_tiling_swizzle.cpp。
    swizzle = TilingKeyField(bits=1, values=[0, 1])
    # D 的分配宽度分档。**取值就是实际列数**(不是 0/1 索引)，故可直接当
    # tile 的 shape 用 —— 已实测两档均 bit-exact，且在被内联的 helper 里
    # 直接引用该名字也成立(无需逐层传参)。
    #
    # 为什么 D 必须占 tilingkey 位，而 layout/GQA 不用：后两者只改索引算术
    # (运行期数值)，而 D 改的是**每块片上 tile 的宽度**，即分配尺寸，
    # 而 pypto 的 tile shape 是 trace 期常量，所以 D 必须做成编译期分档，
    # 不能当运行期值。
    # 且「一律按最大 192 分配 + 运行期裁列」并非只是慢：L0C 会要 288KB，
    # 超出 256KB 预算(即便 acc 降到单缓冲)，物理上放不下。
    #
    # Dv 不占 tilingkey：它只决定已分配 tile 里填多少列，是纯运行期数值。
    d_align = TilingKeyField(bits=2, values=[64, 128, 192])
    # Dv 的**分配**宽度分档。Dv 本身是运行期值(填多少列)，但 y/dy/prod/tmp
    # 这四块 UB 的宽度是 Dv 派生的，而 tile shape 必须是 trace 期常量，
    # 故同样要分档。(64, 128) 是例外：d_align=64 时 dv_align 强制 128。
    dv_align = TilingKeyField(bits=1, values=[128, 192])
    # BN2 MultiBlk：S 在 (128, 640] 仍走 BN2 函数。bit9，不打乱已有 key。
    is_bn2_multiblk = TilingKeyField(bits=1, values=[0, 1])
    # BN2 MultiBlk 且存在整行或整列无效 tile：pre 把 dq/dk/dv out 清零。
    # 一行/一列共用这一位，kernel 不区分。
    bn2_need_zero = TilingKeyField(bits=1, values=[0, 1])

    def is_valid(self, key):
        """剪掉本实现未走到的组合，控制二进制数量。"""
        (
            template,
            layout,
            mask_mode,
            swizzle,
            d_align,
            dv_align,
            is_bn2_multiblk,
            bn2_need_zero,
        ) = key
        # 2 未用；3 预留给 BN2S2。不剪会当 GS1S2 多编死二进制。
        if template != 0 and template != 1:
            return False
        if template == 1:
            if d_align == 192 or swizzle != 0 or layout != 0:
                return False
            if d_align == 64 and dv_align != 128:
                return False
            if bn2_need_zero == 1:
                return is_bn2_multiblk == 1 and mask_mode != 0
            return True
        if is_bn2_multiblk == 1 or bn2_need_zero == 1:
            return False
        # GS1S2 + d_align=64：tilingkey 的 dv_align 没有 64 档，host 仍发 128。
        if d_align == 64:
            return dv_align == 128
        if dv_align > d_align:
            return False
        # D=192 只编非 TND。mask 3/4 与 dense 共用 GS1S2 D-split；
        # swizzle+mask 的 Group B 未迁，不编那几份。
        if d_align == 192:
            return layout == 0 and (mask_mode == 0 or swizzle == 0)
        return True
