# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FAG kernel 的公共定义。

这里只放**跨 cube / vector / schedule 三侧共享**的东西：基本块尺寸、D 轴分块
策略表、NZ 分形常量、CV 跨核同步 flag，以及两个贯穿全流程的结构体
``FagConstInfo`` / ``FagRunInfo``。

不要往这里放地址(见 :mod:`fag_mem`)或任何带 ``pl.load`` / ``pl.matmul`` 的
计算(见 :mod:`fag_block_cube` / :mod:`fag_block_vec`)。
"""

import pypto_pro.language as pl

# ---- 基本块尺寸 (对应 s1/s2/dTemplateType) ----
CUBE_BASEM = 128  # s1 对齐 128
CUBE_BASEN = 128  # s2 对齐 128
# D 的分配宽度。这不是普通常量，而是 tilingkey 字段 d_align 的取值之一 ——
# 见 FlashAttnGradTilingKey.d_align 与下方 D_CHUNKS 的说明。
# 需要按档变化的站点一律用 d_align(tilingkey 字段，trace 期常量)；这个名字只
# 在「明确指 128 这一档」时用，例如 fag_mem 里 BN2 复用的那张图。
HEAD_DIM_ALIGN = 128  # D 对齐 128 档
BLOCK_ELEMS_F16 = 16

# ---- 每档的 D 轴分块表 ----
# D<=128 只有一块，展开后就是一次完整的 matmul，不产生分块开销。
# D=192 切成等宽两块 96+96 而非 128+64：实测(bench_dsplit.py/bench_dsplit_n.py，
# 双点斜率法 + 5 次中位数，单位 us/次完整 D 遍历)
#     方案            MM2(D 在 K 轴)   dQ/dK(D 在 N 轴)
#     96+96  (DB)         0.466            0.738
#     128+64 (DB)         0.549 (+15.2%)   0.818 (+9.8%)
#     192 单发(无 DB)      0.781            1.123
# 两段等宽时 mac 与 MTE1 的节拍整齐；128+64 的第二段只有一半宽，盖不住前段。
# 分块的首要目的是保住 L0A/L0B 双缓冲：[128,96]fp16=24KB，x2=48KB<=64KB。
# 若直接按 192 单发，L0 tile 变 48KB，双缓冲放不下 —— 那是之前 319->184us
# 的最大一项优化。96=6*16 满足 fp16 的 C0=16 分形对齐。
#
# 表拆成几个**平铺的标量表**而不是一个嵌套元组表：把 tuple-of-tuples 赋给
# 局部变量(哪怕只是 `chunks = TABLE[d_align]` 之后再下标取)会让 codegen 报
# "Array tuple assignment ... has elements without a C++ name"。
D_NCHUNKS = {64: 1, 128: 1, 192: 2}  # 块数(当前最多 2)
D_CHUNK_OFF0 = {64: 0, 128: 0, 192: 0}  # 第 0 块的列偏移
D_CHUNK_W0 = {64: 128, 128: 128, 192: 96}  # D<=128 一块 128；尾列靠 valid_shape
D_CHUNK_OFF1 = {64: 0, 128: 0, 192: 96}  # 第 1 块(仅 192 档)
D_CHUNK_W1 = {64: 0, 128: 0, 192: 96}
# L0A/L0B/L0C 按最宽的一块声明，足以装下任一块
D_CHUNK_MAXW = {64: 128, 128: 128, 192: 96}

# Cube L1/L0 物理宽钉在 128（d_align=64 只是 tilingkey 分档，Acc/L1 仍按
# 128 开）。pre/post 已改成 1D 元素切，不再用列宽表。V1 列宽见 v1w。
D_PHYS_W = {64: 128, 128: 128, 192: 192}

CV_CORE_RATIO = 2
VECTOR_BASEM = CUBE_BASEM // CV_CORE_RATIO  # 64，每个 vector 子核处理半个 S1
VECTOR_BASEN = CUBE_BASEN

# 单个 VF 寄存器的 fp32 / fp16 lane 数
VF_LANES = 64
VF_LANES_FP16 = 128
# 压缩 mask 边长 2048，minValue = 0xFF7FFFFF
ATTEN_MASK_COMPRESS = 2048
ATTEN_MASK_MIN = -3.4028234663852886e38

# 兼容测试脚本导出的别名
TILE_M = CUBE_BASEM
TILE_N = CUBE_BASEN
TILE_D = HEAD_DIM_ALIGN


# ---- NZ 分形排布常量 ----
# v3/v4 的融合 VF 直接以 NZ 写出，随后 insert 进 CV 共享 L1；
# 搬运在 fag_block_vec.copy_ub2l1，这里只放几何。
# NZ 分形：fp16 的 C0 = 16 元素 = 32B(一个 DataBlock)
FRACTAL_C0 = 16
UB_BLOCK_BYTES = 32
# 一个分形列占 VECTOR_BASEM*C0 个 fp16 = (64*16)*2/32 = 64 个 32B block，
# 末尾 +1 垫一个 block 错开 bank：
#   blockStride = (srcM * blockN) * sizeof(T1) / blockSize + 1
NZP_BLOCK_STRIDE = (VECTOR_BASEM * FRACTAL_C0) * 2 // UB_BLOCK_BYTES + 1  # 65
NZP_REPEAT_STRIDE = 1
NZP_ROWS = VECTOR_BASEM + 1  # 65

# ---- cross core sync flags ----
SYNC_C1_TO_V2_FLAG = [0, 1]  # mm1(dO@V^T) 结果就绪，按 taskId&1 ping-pong
SYNC_C2_TO_V2_FLAG = [2, 3]  # mm2(Q@K^T)  结果就绪，按 taskId&1 ping-pong
SYNC_V3_TO_C3_FLAG = 4  # dS 已由 UB 落入 L1，dq/dk 可以开算
SYNC_V4_TO_C5_FLAG = 5  # P  已由 UB 落入 L1，dv 可以开算
SYNC_C4_TO_V3_FLAG = 8  # 反向：dqk 已读完 dS L1，vector 可覆写
SYNC_C5_TO_V4_FLAG = 9  # 反向：dv  已读完 P  L1，vector 可覆写
# 反向：vector 已读完 mm1/mm2 的 UB 槽位，cube 可以覆写。
# mm1/mm2 各只有 2 个槽位，
# 而 cube 领先 vector 两个 task，故 task N 要写的槽位正是 vector 刚读完
# 的 task N-2 那块，必须显式等这个 flag 才能覆写。
SYNC_V2_TO_C1_FLAG = [10, 11]
SYNC_V2_TO_C2_FLAG = [12, 13]
# BN2 1-ahead：mm3/mm4 的 fp32 fixpipe → 独立 Muls+Cast
SYNC_C3_TO_V5_FLAG = 6  # mm3 fp32 已进 mm1 UB，v5 可 Muls+Cast dQ
SYNC_C4_TO_V6_FLAG = 7  # mm4 fp32 已进 mm2 UB，v6 可 Muls+Cast dK
# cube 在下一轮 mm1/mm2 前等 v6 写完，避免覆写仍在用的 mm1/mm2 槽
SYNC_DETER_FIX_FLAG = 14

# cube 领先 vector 的 task 数。PRELOAD_TIMES=3：首轮连发两组
# mm2+mm1(共四次 matmul)填满两个 UB 槽位，之后第 N 组的 cube 计算与
# 第 N-2 组的 vector 计算重叠。runInfos 需要 3 份(N / N-1 / N-2)。
PRELOAD_TIMES = 3


# ==================================================================
#  SetConstInfo
# ==================================================================
def set_const_info(tiling, coreNum, enableSwizzle):
    """由 tiling + 运行期核数推导常量。

    coreNum 不在 tiling 里(TilingData 没有该字段)，改由调用方从
    pl.get_block_num() 取：这是 host SetBlockDim 启动的 cube 数，dense 下
    等于 metadata blockOuter。cube/vector 两侧同一口径，set/wait 才能配对。

    enableSwizzle 同理由 host 判据算好后经 tilingkey/常量传入。
    """
    s1Outer = (tiling.s1 + CUBE_BASEM - 1) // CUBE_BASEM
    s2Outer = (tiling.s2 + CUBE_BASEN - 1) // CUBE_BASEN
    return pl.struct(
        "FagConstInfo",
        bSize=tiling.b,
        n2Size=tiling.n2,
        s1Size=tiling.s1,
        s2Size=tiling.s2,
        dSize=tiling.d,
        # V/dO/attn_out/dv 侧的 head dim。D 与 Dv 可以不等长(如 D=192/Dv=128)。
        # Dv 是纯运行期数值：tile 一律按 d_align 分配，Dv 只决定填多少列。
        dvSize=tiling.dv,
        dAlign=(tiling.d + BLOCK_ELEMS_F16 - 1) // BLOCK_ELEMS_F16 * BLOCK_ELEMS_F16,
        dEvenSize=(tiling.d + 1) // 2,
        dOddSize=tiling.d // 2,
        dvEvenSize=(tiling.dv + 1) // 2,
        dvOddSize=tiling.dv // 2,
        scaleValue=tiling.scaleValue,
        s1Outer=s1Outer,
        s2Outer=s2Outer,
        s1CvTail=tiling.s1 - (s1Outer - 1) * CUBE_BASEM,
        s2CvTail=tiling.s2 - (s2Outer - 1) * CUBE_BASEN,
        # BSND: [B,S,N,D] 的行距
        n2D=tiling.n2 * tiling.d,
        s1oS2o=s1Outer * s2Outer,
        # 块网格为 b -> n2 -> g -> s2o -> s1o（g 在 n2 之内、s2o 之外）。
        # 下面三个常量是对应层级的块数：
        #   s1oS2o     : 一个 (b,n2,g) 内的块数
        #   gS1oS2o    : 一个 (b,n2)   内的块数
        #   n2GS1oS2o  : 一个 b        内的块数
        gS1oS2o=tiling.gSize * s1Outer * s2Outer,
        n2GS1oS2o=tiling.n2 * tiling.gSize * s1Outer * s2Outer,
        totalBlockNum=tiling.b * tiling.n2 * tiling.gSize * s1Outer * s2Outer,
        gSize=tiling.gSize,
        coreNum=coreNum,
        enableSwizzle=enableSwizzle,
        # 布局视图参数(BSND/BNSD 统一四维视图)，见 FlashAttnGradTilingData 注释。
        # Q 侧 n1=n2*G 个 head，KV 侧 n2 个，故两套。
        viewD0Q=tiling.viewD0Q,
        viewD2Q=tiling.viewD2Q,
        coefB0Q=tiling.coefB0Q,
        viewD0KV=tiling.viewD0KV,
        viewD2KV=tiling.viewD2KV,
        coefB0KV=tiling.coefB0KV,
        coefN0=tiling.coefN0,
        coefN2=tiling.coefN2,
        maskMode=tiling.maskMode,
        winLeft=tiling.winLeft,
        winRight=tiling.winRight,
        sparseType=tiling.sparseType,
        s1Token=tiling.s1Token,
        s2Token=tiling.s2Token,
        totalPerBatchNum=tiling.totalPerBatchNum,
        bandP=(pl.max(tiling.s1Token, 0) + CUBE_BASEM - 1) // CUBE_BASEM,
        bandQ=(pl.max(tiling.s2Token, 0) + CUBE_BASEN - 1) // CUBE_BASEN,
    )


def make_run_infos():
    """FagRunInfo runInfos[PRELOAD_TIMES]，用于 cv ping pong。

    3 份而非 2 份：cube 领先 vector 两个 task，同一时刻 N / N-1 / N-2
    三组 runInfo 都还活着。

    必须以字面量构造后再逐字段赋值：struct 字段按 int64_t 生成，
    直接用派生表达式(uint64_t)初始化会触发 -Wc++11-narrowing。
    """
    return pl.struct_array(
        3,  # == PRELOAD_TIMES，parser 要求字面量整数
        "FagRunInfo",
        boIdx=0,
        n2oIdx=0,
        s1oIdx=0,
        s2oIdx=0,
        s1RealSize=0,
        s2RealSize=0,
        halfS1RealSize=0,
        queryOffset=0,
        keyOffset=0,
        # GQA 的组内索引 g，取值 [0, gSize)
        goIdx=0,
        # 统一四维视图上的第 0 / 第 2 轴索引，见 set_run_info()。
        # Q 侧(q/dout/attn_out/dq)用 head = n2oIdx*gSize + goIdx；
        # KV 侧(k/v/dk/dv)用 head = n2oIdx。
        idx0Q=0,
        idx2Q=0,
        idx0KV=0,
        idx2KV=0,
        loadBase=0,
        rowShift=0,
        loadKV=0,
        dkvAcc=0,
        flushDkv=0,
        gIdx=0,
        dqFirst=1,
        dqLast=1,
    )
