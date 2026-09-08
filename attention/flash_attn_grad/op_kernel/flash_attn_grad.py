# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FlashAttentionScoreGrad kernel (PyPTO Pro).

一比一复刻 AscendC 实现 arch35/flash_attention_score_grad_kernel.h，
本文件对应 SPLIT_AXIS == BN2GS1S2 模板，且为下述场景裁剪：

  非 TND(layoutType == BSNGD)、无 sparse / attenMask / pse / dropout /
  rope / sink / 确定性计算(DETER_SPARSE_TYPE == NO_DETER)、输入 FP16。

在该配置下 AscendC 的编译期谓词取值为：
  IS_DQ_WRITE_UB = IS_DK_WRITE_UB = IS_DV_WRITE_UB = false
      -> dQ/dK/dV 均由 fixpipe 经 GM workspace 做 atomicAdd 累加
  IS_DKV_RESIDENT_L0C = true (mm4+mm5=131072，max(mm1..3)=65536，
      合计 196608 <= 262144 字节 L0C)
      -> AscendC 下 dK/dV 可驻留 L0C 累加。本文件当前仍走
         「fixpipe 直出 GM + atomicAdd」，L0C 驻留作为后续优化单独叠加
  IS_DROP = false
      -> ComputeDqkvBn2gs1s2 走 ProcessBn2gs1s2LastVec 的 else 分支
  IsValid() 恒为 true(无 attenMask)
      -> GetNextValidIdx 退化为边界判断，无需扫描
本文件实现 Process() 的两槽 taskId ping-pong。PRELOAD_TIMES=3 的
ProcessPreloadTwoTimes 预取环按需求暂不实现。

CV 通信全部使用手动 set/wait cross-core，且正反向成对：
  正向 C->V : SYNC_C2_TO_V2_FLAG[2] / SYNC_C1_TO_V2_FLAG[2]  (mm2/mm1 结果就绪)
  正向 V->C : SYNC_V3_TO_C3_FLAG / SYNC_V4_TO_C5_FLAG        (dS/P 已落 L1)
  反向 C->V : SYNC_C4_TO_V3_FLAG / SYNC_C5_TO_V4_FLAG        (L1 已被 cube 读完，可覆写)
反向同步由 needSyncDkMM 控制：首轮 L1 尚无人读，无需等待。
AscendC 中 AIC 侧对每个 V->C flag 额外等待 `16 + flag`(两个 vector 子核各发一次)，
在 pypto 中等价于 sync_mode=INTRA_BLOCK 的单次 wait。

数据流与 AscendC 对齐：
  MM2(Q@K^T)->UB -> V2(softmax->P) -> V4(P cast+ND2NZ->L1)
  MM1(dO@V^T)->UB -> V3(dS=(dP-sfmg)*P cast+ND2NZ->L1)
  -> MM5(P^T@dO->dV) -> MM3(dS@K->dQ) -> MM4(dS^T@Q->dK)
V1 计算 softmaxGradFront sfmg = rowsum(dy * y)，取自前向输出 y，
对应 vector_api/cast_softmax_grad.h 的 CopyInSoftmaxGrad + MySoftmaxGradFrontCast。

每个 GM tensor 只推导出一种 Layout：tensor_k 因 MM2 的转置读(order=[3,1])
被定为 Layout::DN，MM3 需要非转置的 K，若共用同一 tensor，plain 读会被按 DN
解释而静默拿到 K^T(dQ 幅度正确但数值全错)。故 MM3 用独立的 tensor_k_nt 视图
指向同一 buffer，各自推导 layout。平台仅支持
ND2NZ/DN2NZ/ND2ND/DN2DN/NZ2NZ/DN2ZN，不支持 ND2ZN。

MM4/MM5 的左矩阵需要转置(dS^T / P^T)。AscendC 通过 L1->L0A 载入时置
isLeftTranspose=true 免费得到；pypto 无此能力(NZ 版本静默算错、ZZ 版本
无法编译，L0A 要求非 row-major fractal)，因此显式走
transpose(UB) -> ND2NZ -> insert 到 [N,M] NZ 的 L1，再常规载入 L0A。
"""

from dataclasses import dataclass

import os

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField

# `vf` 由 parser 按语法识别(_call_parser.py: attrs[0] == "vf")，无需 import


# ---- 基本块尺寸 (对应 s1/s2/dTemplateType) ----
CUBE_BASEM = 128  # s1TemplateType::Aligned128
CUBE_BASEN = 128  # s2TemplateType::Aligned128
# D 的分配宽度。这不是普通常量，而是 tilingkey 字段 d_align 的取值之一 ——
# 见 FlashAttnGradTilingKey.d_align 与下方 D_CHUNKS 的说明。
# 保留这个名字是为了让「D=128 档」的代码与改动前逐字一致(零回归)；
# 需要按档变化的站点改用 d_align(tilingkey 字段，trace 期常量)。
HEAD_DIM_ALIGN = 128  # dTemplateType::Aligned128
# ---- 每档的 D 轴分块表 ----
# D=128 只有一块 -> 展开后与改动前生成同样的代码，故 D=128 档天然零回归。
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
D_NCHUNKS = {128: 1, 192: 2}  # 块数(当前最多 2)
D_CHUNK_OFF0 = {128: 0, 192: 0}  # 第 0 块的列偏移
D_CHUNK_W0 = {128: 128, 192: 96}  # 第 0 块的列宽
D_CHUNK_OFF1 = {128: 0, 192: 96}  # 第 1 块(仅 192 档)
D_CHUNK_W1 = {128: 0, 192: 96}
# L0A/L0B/L0C 按最宽的一块声明，足以装下任一块
D_CHUNK_MAXW = {128: 128, 192: 96}
CV_CORE_RATIO = 2
VECTOR_BASEM = CUBE_BASEM // CV_CORE_RATIO  # 64，每个 vector 子核处理半个 S1
VECTOR_BASEN = CUBE_BASEN

# 单个 VF 寄存器的 fp32 lane 数
VF_LANES = 64

# V1 的 sfmg 分段行数：prod/tmp 两块 fp32 缓冲按此尺寸开，控制 UB 占用
SFMG_LOOP_SIZE = VECTOR_BASEM // 2  # 32

# 兼容测试脚本导出的别名
TILE_M = CUBE_BASEM
TILE_N = CUBE_BASEN
TILE_D = HEAD_DIM_ALIGN

# ---- cross core sync flags，取值与 flash_attention_score_grad_common.h 一致 ----
SYNC_C1_TO_V2_FLAG = [0, 1]  # mm1(dO@V^T) 结果就绪，按 taskId&1 ping-pong
SYNC_C2_TO_V2_FLAG = [2, 3]  # mm2(Q@K^T)  结果就绪，按 taskId&1 ping-pong
SYNC_V3_TO_C3_FLAG = 4  # dS 已由 UB 落入 L1，dq/dk 可以开算
SYNC_V4_TO_C5_FLAG = 5  # P  已由 UB 落入 L1，dv 可以开算
SYNC_C4_TO_V3_FLAG = 8  # 反向：dqk 已读完 dS L1，vector 可覆写
SYNC_C5_TO_V4_FLAG = 9  # 反向：dv  已读完 P  L1，vector 可覆写
# 反向：vector 已读完 mm1/mm2 的 UB 槽位，cube 可以覆写。
# mm1/mm2 各只有 2 个槽位(与 ASC 的 mm1ResBuf[2]/mm2ResBuf[2] 一致)，
# 而 cube 领先 vector 两个 task，故 task N 要写的槽位正是 vector 刚读完
# 的 task N-2 那块，必须显式等这个 flag 才能覆写。
SYNC_V2_TO_C1_FLAG = [10, 11]
SYNC_V2_TO_C2_FLAG = [12, 13]

# cube 领先 vector 的 task 数。ASC 的 PRELOAD_TIMES=3：首轮连发两组
# mm2+mm1(共四次 matmul)填满两个 UB 槽位，之后第 N 组的 cube 计算与
# 第 N-2 组的 vector 计算重叠。runInfos 需要 3 份(N / N-1 / N-2)。
PRELOAD_TIMES = 3

# ---- L1 地址 (MutexBufferManager<BufferType::L1>，总量 512KB) ----
# dSL1Buf / pL1Buf 对应 kernel_base.h InitCVCommonBuffer 中的同名 buffer
L1_Q = 0x00000  # [128,128] fp16 = 32KB
L1_K = 0x08000
L1_V = 0x10000
L1_DO = 0x18000
L1_DS = 0x20000  # dSL1Buf : [M,N] 供 MM3 左矩阵
L1_P = 0x28000  # pL1Buf
# 0x30000-0x3FFFF 空闲：dS^T/P^T 不再单独存储(改为 L1_DS/L1_P 的 ZN 视图)
# MM3(dQ = dS@K) 的右矩阵要非转置的 K[S2,D]，必须独占一块 L1：
# 同一地址上叠加两种 shape/layout 视图(ZN 的 [D,S2] 与 ND 的 [S2,D])
# 能编过但结果静默错误。
L1_K_ND = 0x40000
# L1_PRELOAD：Q/K/V/dO/K_ND 各再开一块，构成 double buffer。
# tile_group 填两个 addrs + 两个 mutex_id 后，框架会自动插入核内同步，
# 使下一轮的 load 与本轮的 matmul 重叠(对应 AscendC 的 IS_L1_PRELOAD)。
# 合计 14 块 x 32KB = 448KB <= 512KB。
# Q/dO 各 3 个槽位(而非 2)：cube 领先 vector 两个 task，mm2 在 task N
# 载入 Q_N，mm4 要的是 Q_{N-2}。3 槽位时 N%3 != (N-2)%3，两者不会互相
# 覆写，于是 mm4/mm5 可以直接复用 mm2/mm1 搬好的数据，无需再读 GM。
L1_Q2 = 0x48000
L1_Q3 = 0x50000
L1_DO2 = 0x58000
L1_DO3 = 0x60000
L1_Q4 = 0x68000
L1_DO4 = 0x70000


# ---- 按 d_align 分档的 L1 布局 ----
# 128 档沿用上面那套写死的地址(与本改动前逐字一致，保证零回归)；
# 192 档每块是 48KB 而不是 32KB，512KB 装不下 14 块，故 Q/dO 各留 3 槽
# (mm4/mm5 复用 mm2/mm1 数据所需的最小槽位数，见 q_l1_dk 的说明)：
#   q x3 + dO x3 + k + k_nd + v = 9 x 48KB = 432KB，再加 ds/p 各 32KB
#   (它们是 [M,N]，与 D 无关) = 496KB <= 512KB。
# dict 查表在 addrs= 位置可用，且槽位数允许随档不同(已实测)。
# 地址表在**模块级**按档算好，kernel 里只做 `TABLE[d_align]` 这一步下标取值：
# 不能把 dict 赋给 kernel 内的局部变量(parser 报 "Unsupported closure
# variable type: dict"，只接受 int/float/bool/list/tuple)，但 dict 下标取出的
# list/int 可以直接用在 addrs= / mutex_ids= 位置(已实测，且槽位数可随档不同)。
def _build_l1(width):
    """铺排一档的 L1，返回 (q槽位, dO槽位, k, k_nd, v, ds, p)。"""
    if width == 128:
        # 沿用上面写死的地址，保证 128 档生成代码与改动前逐字一致
        return (
            [L1_Q, L1_Q2, L1_Q3, L1_Q4],
            [L1_DO, L1_DO2, L1_DO3, L1_DO4],
            L1_K,
            L1_K_ND,
            L1_V,
            L1_DS,
            L1_P,
        )
    blk = CUBE_BASEM * width * 2  # 一块 [128,width] fp16 = 48KB
    dsp = CUBE_BASEM * CUBE_BASEN * 2  # ds/p 是 [M,N]，与 D 无关 = 32KB
    a = 0
    qs = []
    for _ in range(3):
        qs.append(a)
        a += blk
    dos = []
    for _ in range(3):
        dos.append(a)
        a += blk
    k = a
    a += blk
    k_nd = a
    a += blk
    v = a
    a += blk
    ds = a
    a += dsp
    p = a
    a += dsp
    assert a <= 512 * 1024, f"L1 overflow for d_align={width}: {a} bytes"
    return (qs, dos, k, k_nd, v, ds, p)


_L1_TAB = {w: _build_l1(w) for w in (128, 192)}
L1_Q_ADDRS = {w: _L1_TAB[w][0] for w in _L1_TAB}
L1_DO_ADDRS = {w: _L1_TAB[w][1] for w in _L1_TAB}
L1_K_ADDR = {w: _L1_TAB[w][2] for w in _L1_TAB}
L1_KND_ADDR = {w: _L1_TAB[w][3] for w in _L1_TAB}
L1_V_ADDR = {w: _L1_TAB[w][4] for w in _L1_TAB}
L1_DS_ADDR = {w: _L1_TAB[w][5] for w in _L1_TAB}
L1_P_ADDR = {w: _L1_TAB[w][6] for w in _L1_TAB}
# mutex_id 跟着槽位数走。128 档沿用原有分配(4 槽)，192 档 3 槽。
L1_Q_MUTEX = {128: [4, 31, 21, 18], 192: [4, 31, 21]}
L1_DO_MUTEX = {128: [7, 15, 22, 19], 192: [7, 15, 22]}

# ---- L0A / L0B / L0C ----
# L0A/L0B 各 64KB，[128,128]fp16 占 32KB，故各放 2 块做双缓冲：
# 5 个 matmul 串行复用同一块 L0A/L0B 时，pl.move 填充期间 mac 只能干等
# (实测 mte1 占用率 0.309 vs ASC 0.584)。双缓冲让下一个 matmul 的
# L1->L0 搬运与当前 matmul 的计算重叠。
L0A_ADDR = 0x0000
L0A_ADDR2 = 0x8000
L0B_ADDR = 0x0000
L0B_ADDR2 = 0x8000
L0C_ADDR = 0x0000
# L0C 驻留累加(IS_DKV_RESIDENT_L0C=true)：dK/dV 各占一块独立 L0C，
# 在同一 s2 块内跨 s1 累加，s2 切换时才 fixpipe 出 GM。
# acc 64KB + dk 64KB + dv 64KB = 192KB <= 256KB
L0C_DK = 0x10000
L0C_DV = 0x20000
# acc 的第二块(L0C 共 256KB: acc x2 + dk + dv = 4 x 64KB 正好用满)。
# mm2/mm1/mm3 共用 acc，单缓冲时 fixpipe 把结果写出期间下一个 matmul
# 只能干等(实测 fixpipe 0.458 而 mac 仅 0.783)。
L0C_ADDR2 = 0x30000


# ---- 按 d_align 分档的 L0 布局 ----
# L0A/L0B 按「最宽的一个 D 分块」声明，故 192 档是 [128,96]=24KB 而不是
# 48KB，双缓冲(2x24=48KB<=64KB)得以保留 —— 这正是切 D 轴的首要目的。
def _build_l0ab(width):
    """L0A/L0B 的两个槽位地址(双缓冲)。

    间距必须按 **tile 实际声明的宽度** ACC_COLS 算，而不是 D 分块宽度：
    L0A/L0B 要同时装下 MM3/MM4/MM5 的 [M,N] 左矩阵，故声明成 ACC_COLS 列。
    若按分块宽 96 算间距(24KB)，两个槽位会重叠(每块实际占 32KB)。
    """
    if width == 128:
        return [L0A_ADDR, L0A_ADDR2]
    blk = CUBE_BASEM * max(CUBE_BASEN, D_CHUNK_MAXW[width]) * 2
    return [0x0000, blk]


def _build_l0c(width):
    """返回 (acc 槽位, dk 各分块地址, dv 各分块地址)。

    dK/dV 要在同一 s2 块内跨 s1 轮次累加(IS_DKV_RESIDENT_L0C)，故每个 D
    分块都得有自己的常驻 L0C —— 不能像 acc 那样复用。
      128 档: acc x2 + dk x1 + dv x1 = 4 x 64KB = 256KB(与改动前一致)
      192 档: acc x1 + dk x2 + dv x2 = 5 x 48KB = 240KB <= 256KB
              (dv 也按 D 的分块数分块；Dv<=D 时后一块可能整块无效，
               由运行期宽度判断跳过，见 dv_chunk_w)
    """
    if width == 128:
        return ([L0C_ADDR, L0C_ADDR2], [L0C_DK], [L0C_DV])
    n = D_NCHUNKS[width]
    # acc 声明成 ACC_COLS 列(要装 MM2/MM1 的 [M,s2] 输出)，故按它算大小；
    # dK/dV 只装 D 分块，按分块宽即可。间距一律用「tile 实际声明的宽度」，
    # 否则槽位会重叠。
    accBlk = CUBE_BASEM * max(CUBE_BASEN, D_CHUNK_MAXW[width]) * 4
    dkvBlk = CUBE_BASEN * D_CHUNK_MAXW[width] * 4
    a = 0
    acc = [a]
    a += accBlk
    dk = []
    for _ in range(n):
        dk.append(a)
        a += dkvBlk
    dv = []
    for _ in range(n):
        dv.append(a)
        a += dkvBlk
    assert a <= 256 * 1024, f"L0C overflow for d_align={width}: {a} bytes"
    return (acc, dk, dv)


L0AB_ADDRS = {w: _build_l0ab(w) for w in (128, 192)}
_L0C_TAB = {w: _build_l0c(w) for w in (128, 192)}
L0C_ACC_ADDRS = {w: _L0C_TAB[w][0] for w in _L0C_TAB}
L0C_ACC_MUTEX = {128: [14, 28], 192: [14]}
L0C_DK_ADDRS = {w: _L0C_TAB[w][1] for w in _L0C_TAB}
L0C_DV_ADDRS = {w: _L0C_TAB[w][2] for w in _L0C_TAB}
# 每个分块单独取一个标量地址表。不能在 kernel 里写 `TABLE[d_align][-1]`：
# parser 不支持负下标(报 "GetItemExpr index -1 out of bounds")。
L0C_DK0_ADDR = {w: L0C_DK_ADDRS[w][0] for w in _L0C_TAB}
L0C_DV0_ADDR = {w: L0C_DV_ADDRS[w][0] for w in _L0C_TAB}
# 第二个分块的地址/mutex。单块档位下这些 tile 永不被真的使用(调用路径被
# trace 期的 `if D_NCHUNKS > 1` 整块剪掉)，故让它别名到第 0 块即可。
L0C_DK1_ADDR = {w: L0C_DK_ADDRS[w][D_NCHUNKS[w] - 1] for w in _L0C_TAB}
L0C_DV1_ADDR = {w: L0C_DV_ADDRS[w][D_NCHUNKS[w] - 1] for w in _L0C_TAB}
L0C_DK1_MUTEX = {128: 16, 192: 29}
L0C_DV1_MUTEX = {128: 17, 192: 25}
# acc/L0A/L0B 的列宽：MM2/MM1 输出 [M,s2]，MM3 输出 dQ 的一个 D 分块。
# 取两者的较大值，一块 tile 同时够用。128 档等于 128(与改动前一致)。
ACC_COLS = {w: max(CUBE_BASEN, D_CHUNK_MAXW[w]) for w in (128, 192)}

# ---- pre/post 阶段的 UB ----
# 这两个阶段与主循环靠 sync_all 完全隔开，故直接从 0 起借用整块 UB
# (mm1/mm2 结果区等)，不与主循环争空间；mutex_ids 同样可复用。
#
# chunk 大小实测结论：把 UB 吃满反而更慢 —— 并行度比单次搬运大小重要。
# 8192 行按 chunk 跨核分配(核 i 领第 i, i+vecCoreNum, ... 块)时：
#
#   pre chunk  UB      块数   耗时      post chunk  UB       耗时
#   496 行     248KB    17    201.7us   165 行      247.5KB  (见左)
#   128 行      64KB    64    193.6us    64 行       96.0KB  193.6us
#    48 行      24KB   171    192.8us    32 行       48.0KB  193.5us
#
# pre=496 时 8192 行只切出 17 块，72 个 vector 核有 55 个空跑，并行度
# 从 72 塌到 17；搬运总量不变但摊到的核少了，单核串行时间变长，用满 UB
# 换来的「单次 MTE3 更大」补不回这个损失。post chunk 影响很小
# (165->32 仅差 1.3us)，因为它每轮都有 load/cast/store 三段可重叠。
# 取 128/64：块数远多于核数，每核多轮，depth 2 的重叠才真正生效。
# S=8192 下扫 N=1/32/64 三档，pre=128/post=64 在大 N 下最优、N=1 时
# 仅比最佳(48)慢 0.4%，故取它。大 N 时 pre/post 只占总时间约 1.5%，
# chunk 选值影响很小 —— 主导项是 matmul。
PRE_CHUNK_ROWS = 128
UB_PRE_ZERO = 0x00000

# post 需要 fp32 输入 + fp16 输出各两槽(depth 2)，让 MTE2/V/MTE3 并行：
#   D=128: 2 * 64 * 128 * (4 + 2) =  96KB
#   D=192: 2 * 64 * 192 * (4 + 2) = 144KB   都在 248KB 以内
POST_CHUNK_ROWS = 64
UB_POST_F32_0 = 0x00000


def post_ub_addrs(width):
    """按分配宽度算 post 的四个 UB 槽位地址。

    地址随档位变化，故不能写成模块级常量 —— 但 width 是 trace 期常量
    (tilingkey 字段 d_align)，在 kernel 体内算出来即可。
    D=128 时得到的四个地址与本改动前完全一致，故该档零回归。
    """
    f32Bytes = POST_CHUNK_ROWS * width * 4
    f16Bytes = POST_CHUNK_ROWS * width * 2
    f32_1 = UB_POST_F32_0 + f32Bytes
    f16_0 = f32_1 + f32Bytes
    return UB_POST_F32_0, f32_1, f16_0, f16_0 + f16Bytes


# ---- UB 地址，硬上限 248KB；实测总占用 242.5KB ----
# mm1ResBuf[2] / mm2ResBuf[2] 与 AscendC 同名同尺寸(VECTOR_BASEM*VECTOR_BASEN*4)
UB_MM1_0 = 0x00000  # fp32[64,128] 32KB
UB_MM1_1 = 0x08000
UB_MM2_0 = 0x10000
UB_MM2_1 = 0x18000
# V1 按 SFMG_LOOP_SIZE 行分段，避免 prod+tmp 两块 [64,128] fp32 撑爆 UB。
# 这四块的宽度是 **Dv**(y/dy 是 Dv 宽)，故按 dv_align 分档：
#   dv_align=128: y/dy 各 8KB，prod/tmp 各 16KB，合计 48KB
#   dv_align=192: y/dy 各12KB，prod/tmp 各 24KB，合计 72KB
# 后面几块的地址跟着顺移，故整段按档算(见 _build_ub)。
UB_Y = 0x20000  # fp16[32,dv] V1 前向输出 y
UB_DY = 0x22000  # fp16[32,dv] V1 的 dy
UB_PROD = 0x24000  # fp32[32,dv] dy*y 的逐元素积
UB_TMP = 0x28000  # fp32[32,dv] pl.sum 要求的同尺寸 workspace
# cast+ND2NZ 融合 VF 的输出缓冲：声明 65 行而非 64。
# VF 用 block_stride=65 写出，每个分形列比紧凑排布多垫一个 32B block，
# 把相邻分形列错开到不同 bank(对应 ASC vf_cast_transdata_deconflict.h 的 +1)。
UB_NZP = 0x2C000  # fp16[65,128] 16.25KB
# 0x30400-0x3BFFF 空闲
# sfmg 两个分段结果地址连续，故可用一个 [64,1] 视图整体读取
UB_SFMG = 0x3C000  # fp32[64,1]  256B  softmaxGradResBuf
UB_SFMG_HI = 0x3C080  # 后 32 行 (32*4 = 128B 偏移)
# lseQue[2] 双缓冲。原先是 max/sum 两组共四块，改用 LSE 后合成一组：
# lse = max + log(sum(exp(s-max)))，前向已把两者合进一个张量。
UB_LSE_0 = 0x3C200  # fp32[64,1] 256B
UB_LSE_1 = 0x3C400
# 末尾 0x3C600 = 241.5KB，未超 248KB 上限


def _build_ub(dvw):
    """按 Dv 的分配宽度铺排 V1 之后的那几块 UB。

    dv_align=128 时返回上面那套写死的地址(零回归)；192 时 y/dy/prod/tmp
    各变宽 1.5 倍，其后的 nzp/sfmg/lse 顺移。
    返回 (y, dy, prod, tmp, nzp, sfmg, sfmg_hi, lse0, lse1)。
    """
    if dvw == 128:
        return (
            UB_Y,
            UB_DY,
            UB_PROD,
            UB_TMP,
            UB_NZP,
            UB_SFMG,
            UB_SFMG_HI,
            UB_LSE_0,
            UB_LSE_1,
        )
    a = 0x20000
    y = a
    a += SFMG_LOOP_SIZE * dvw * 2
    dy = a
    a += SFMG_LOOP_SIZE * dvw * 2
    prod = a
    a += SFMG_LOOP_SIZE * dvw * 4
    tmp = a
    a += SFMG_LOOP_SIZE * dvw * 4
    nzp = a
    a += NZP_ROWS * VECTOR_BASEN * 2
    # 后面几块都很小，按 512B 对齐依次排
    a = (a + 0x1FF) & ~0x1FF
    sfmg = a
    sfmg_hi = a + SFMG_LOOP_SIZE * 4
    a += 0x200
    lse0 = a
    lse1 = a + 0x200
    a += 0x400
    assert a <= 248 * 1024, f"UB overflow for dv_align={dvw}: {a} bytes"
    return (y, dy, prod, tmp, nzp, sfmg, sfmg_hi, lse0, lse1)


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


# 与 op_kernel/flash_attn_grad.py 的 FlashAttnGradTilingKey 一致
class FlashAttnGradTilingKey:
    template = TilingKeyField(bits=2, values=[0, 1, 2, 3])
    layout = TilingKeyField(bits=1, values=[0, 1])
    has_atten_mask = TilingKeyField(bits=1, values=[0, 1])
    # 分核方式：0=线性均分，1=swizzle(按 s2 列跨核发放)。
    # 由 host 的 decide_swizzle() 依输入 shape 判定「同时在跑的核其工作集
    # 是否超 L2」后置位，两种分核各编译一份二进制，运行期零分支开销。
    # 判据与实测见下方 decide_swizzle()。
    swizzle = TilingKeyField(bits=1, values=[0, 1])
    # D 的分配宽度分档。**取值就是实际列数**(不是 0/1 索引)，故可直接当
    # tile 的 shape 用 —— 已实测两档均 bit-exact，且在被内联的 helper 里
    # 直接引用该名字也成立(无需逐层传参)。
    #
    # 为什么 D 必须占 tilingkey 位，而 layout/GQA 不用：后两者只改索引算术
    # (运行期数值)，而 D 改的是**每块片上 tile 的宽度**，即分配尺寸，
    # 而 pypto 的 tile shape 是 trace 期常量。ASC 同样把 D 做成模板参数
    # DTemplateType::Aligned128/Aligned192，不是运行期值。
    # 且「一律按最大 192 分配 + 运行期裁列」并非只是慢：L0C 会要 288KB，
    # 超出 256KB 预算(即便 acc 降到单缓冲)，物理上放不下。
    #
    # Dv 不占 tilingkey：它只决定已分配 tile 里填多少列，是纯运行期数值。
    d_align = TilingKeyField(bits=1, values=[128, 192])
    # Dv 的**分配**宽度分档。Dv 本身是运行期值(填多少列)，但 y/dy/prod/tmp
    # 这四块 UB 的宽度是 Dv 派生的，而 tile shape 必须是 trace 期常量，
    # 故同样要分档。dv_align <= d_align 恒成立(host 已校验 Dv<=D)。
    dv_align = TilingKeyField(bits=1, values=[128, 192])

    def is_valid(self, key):
        """剪掉本实现未走到的组合，控制二进制数量。

        key 顺序与字段声明顺序一致。不剪枝时加一位会让组合数 32 -> 64；
        只给 D=192 编译本实现真正走的那条路径(template=0 / 非 TND /
        无 mask)，可压到 34 个(+6%)。D=128 档的全部 32 个组合保持不变。
        """
        template, layout, has_atten_mask, swizzle, d_align, dv_align = key
        # Dv <= D，故 dv_align 不可能大于 d_align
        if dv_align > d_align:
            return False
        if d_align == 128:
            return True
        # D=192 只给本实现真正走的那条路径编译
        return template == 0 and layout == 0 and has_atten_mask == 0


# ==================================================================
#  SetConstInfo —— 对应 kernel_base.h::SetConstInfo
# ==================================================================
def set_const_info(tiling, coreNum, enableSwizzle):
    """由 tiling + 运行期核数推导常量。

    coreNum 不在 tiling 里(新接口的 TilingData 没有该字段)，改由调用方从
    pl.get_block_num() 取。已实测 cube/vector 两侧该 API 都返回物理核数
    (36)，口径一致，故 CV 两侧算出的 numBlocks 必然相同 —— 这是 set/wait
    能配对的前提。

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
        scaleValue=tiling.scaleValue,
        s1Outer=s1Outer,
        s2Outer=s2Outer,
        s1CvTail=tiling.s1 - (s1Outer - 1) * CUBE_BASEM,
        s2CvTail=tiling.s2 - (s2Outer - 1) * CUBE_BASEN,
        # BSND: [B,S,N,D] 的行距，mm2Ka/mm2Kb 即 AscendC 同名字段
        n2D=tiling.n2 * tiling.d,
        s1oS2o=s1Outer * s2Outer,
        # 块网格为 b -> n2 -> g -> s2o -> s1o（与 ASC 的维度顺序一致，
        # g 在 n2 之内、s2o 之外）。下面三个常量是对应层级的块数：
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
    )


def make_run_infos():
    """FagRunInfo runInfos[PRELOAD_TIMES]，用于 cv ping pong。

    3 份而非 2 份：cube 领先 vector 两个 task，同一时刻 N / N-1 / N-2
    三组 runInfo 都还活着(对应 ASC ProcessPreloadTwoTimes 的
    FagRunInfo runInfos[PRELOAD_TIMES])。

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
        firstHalfS1RealSize=0,
        taskIdMod2=0,
        queryOffset=0,
        keyOffset=0,
        isS2IdxNoChange=0,
        # GQA 的组内索引 g，取值 [0, gSize)
        goIdx=0,
        # 统一四维视图上的第 0 / 第 2 轴索引，见 set_run_info()。
        # Q 侧(q/dout/attn_out/dq)用 head = n2oIdx*gSize + goIdx；
        # KV 侧(k/v/dk/dv)用 head = n2oIdx。
        idx0Q=0,
        idx2Q=0,
        idx0KV=0,
        idx2KV=0,
    )


# ==================================================================
#  IsValid + GetNextValidIdx —— 无 attenMask 时 IsValid 恒真，
#  故仅剩 blockEnds 边界判断（对应 kernel_base.h 的 else 分支）
# ==================================================================
# ==================================================================
#  Swizzle 分核 —— 让同时在跑的核落在同一 head 内，命中 L2
# ==================================================================
# 线性切分(核 c 领连续的 [c*blockPerCore, ...))在大 N 下会让 36 个核同时
# 落在 32~36 个不同 head 上。实测 S=8192：
#   N=1  36核跨 1 head，K+V 工作集   4MB -> L2(128MB) 内，0.94x ASC
#   N=32 36核跨 32 head，          128MB -> 刚撑满，    1.46x
#   N=64 36核跨 36 head，          144MB -> 溢出，      1.54x
# 归一化后 mte2 从 93.9 涨到 227.5us，成为新瓶颈(占用率 0.72)，mac 反而
# 从 0.938 掉到 0.733 —— 即 L2 未命中退化成 HBM 带宽受限。
#
# 对标 ASC 的 swizzle(M/N_SWIZZLE_SIZE=32768, 触发条件 isExceedL2Cache,
# common.h:67)：把块网格按 tile 分组，同时在跑的核锁在同一 tile 内。
#
# 这里的做法：以「s2 列」为分配单位(一列 = 一个 (b,n2,s2o) 组 = s1Outer
# 个连续块)，按核跨步发放 —— 核 c 领第 c, c+coreNum, ... 列。
#   * 同核在一列内连续处理 s1Outer 个块 -> s2 不变
#     => K/V 只载一次、dK/dV 在 L0C 连续累加，这两项优化都保住
#   * 第 k 轮时 36 核处理列 k*coreNum+c，几乎总在同一 head 内
#     => K+V 工作集回落到 4MB
#
# 代价：分配粒度变粗(一列 s1Outer 块)。N=1 时总列数仅 64 < 2*coreNum，
# 核 0 拿 2 列而核 35 只拿 1 列，负载失衡 2 倍 —— 实测 N=1 由 193.8
# 慢到 220.8us。故与 ASC 一样按「工作集是否超 L2」开关。
#
# 实测(S=8192, 36 核, ASC 为基准):
#   N     线性切分         swizzle          选择
#   1     193.8 (0.94x)   220.8 (1.07x)    线性
#   32   10017 (1.46x)    6650 (0.97x)     swizzle
#   64   21993 (1.54x)   13434 (0.94x)     swizzle
L2_CACHE_BYTES = 128 * 1024 * 1024


def decide_swizzle(b, n, s1, s2, d, core_num, dtype_bytes=2, g_size=1):
    """**tiling.cpp 中 host 判据的 Python 镜像**，仅供 JIT 脚本使用。

    生产路径上这个判据由 host 承担，权威实现与全部标定依据都在
    `op_host/flash_attn_grad_tiling.cpp` 的 `FlashAttnGradTilingFunc()`
    (搜 `int64_t swizzle = 0;`)。本函数只是把那段整数运算逐行照抄成
    Python，让 JIT 脚本(test/prof/bench)能自己拼出 tilingkey ——
    JIT 不走 host，拿不到 host 算好的值。

    **改判据必须同时改两处，且以 tiling.cpp 为准。** 两边算得不一样不会
    报任何错，只会让 JIT 验证的二进制与 host 实际派发的那份不是同一个
    (GQA 下曾因此有 19 个 shape 静默不一致)。
    `test_..._pypto.py` 里有 `test_gate_matches_host()` 守护这件事。

    入参口径与 host 一一对应：
      n       <- n1Size (Q 侧 head 数)，KV 侧 N2 = n // g_size
      g_size  <- gSize
      core_num<- aicNum (物理 cube 核数，**不是** blockDim)
    """
    s1_outer = (s1 + CUBE_BASEM - 1) // CUBE_BASEM
    s2_outer = (s2 + CUBE_BASEN - 1) // CUBE_BASEN
    per_head_blocks = s1_outer * s2_outer
    total = b * n * per_head_blocks
    per_core = (total + core_num - 1) // core_num
    live_heads = len({(c * per_core) // per_head_blocks for c in range(core_num)})

    fp32 = 4
    # q + dy + y 读 (fp16) + dq 写 (fp32) —— 这些是本 g 私有的
    per_head = 3 * s1 * d * dtype_bytes + s1 * d * fp32
    # k + v 读 (fp16) + dk + dv 写 (fp32) —— GQA 下同一个 KV head 被 G 个
    # Q head 共享，故 KV 侧字节按 G 摊薄，只记一份(与 host 的 `/ gSize` 一致)
    per_head += (2 * s2 * d * dtype_bytes + 2 * s2 * d * fp32) // g_size
    # 判据依赖 S1/S2 的**方向**，不只看工作集大小：S2 更长时恒不开、
    # S1 明显更长时拐点更早(1.25x L2)、对称时保持 2x L2。
    # 三档阈值的实测标定数据(29 个点)全部记录在 tiling.cpp 的同一段里，
    # 此处不复制 —— 复制两份必然漂移，要看依据请看那边。
    if s2_outer > s1_outer:
        return 0
    threshold = (1.25 if s1_outer >= 2 * s2_outer else 2.0) * L2_CACHE_BYTES
    if live_heads * per_head <= threshold:
        return 0
    # swizzle 的分配粒度是「一整列 s1Outer 块」，列数太少会失衡到不划算
    # (N=1 时仅 64 列 < 2*36 核，核0 拿 2 列而核35 拿 1 列，实测反慢 14%)
    if b * n * s2_outer < 2 * core_num:
        return 0
    return 1


def swizzle_idx(constInfo, cBlockIdx, coreNum, localIdx):
    """把本核的局部块序号映射成全局块下标。

    局部序号 localIdx 先拆成「第几列 k」与「列内第几个 s1 块」，
    列号再按跨步还原成全局列 t = cBlockIdx + k * coreNum，
    最后由 t 解出 (head, s2Idx) 拼回全局下标。
    """
    k = localIdx // constInfo.s1Outer
    s1Idx = localIdx % constInfo.s1Outer
    t = cBlockIdx + k * coreNum
    head = t // constInfo.s2Outer
    s2Idx = t % constInfo.s2Outer
    return head * constInfo.s1oS2o + s2Idx * constInfo.s1Outer + s1Idx


def sel(flag, whenOne, whenZero):
    """flag∈{0,1} 的算术选择：flag=1 取 whenOne，flag=0 取 whenZero。

    不用 if：解析器要求内联函数只能在末尾的顶层语句 return，且分支赋值
    无法 phi 合并。两路都求值再线性插值，等价于 select 且无运行期跳转。
    """
    return whenZero + flag * (whenOne - whenZero)


def core_block_range(constInfo, cBlockIdx, coreNum):
    """本核负责的块数。

    swizzle: 按「s2 列」跨核发放，本核列数 = [0, numColTasks) 中满足
             t ≡ cBlockIdx (mod coreNum) 的个数，每列 s1Outer 块。
    线性:    连续均分 [cBlockIdx*blockPerCore, +blockPerCore)。
    """
    # 列数 = 融合 batch 维(b*n2*G) x s2Outer。G 必须计入，否则 swizzle 下
    # 会漏掉 (G-1)/G 的列(已验证：并入 G 后覆盖无重复无遗漏)。
    numColTasks = (
        constInfo.bSize * constInfo.n2Size * constInfo.gSize * constInfo.s2Outer
    )
    rem = pl.max(numColTasks - cBlockIdx, 0)
    swz = ((rem + coreNum - 1) // coreNum) * constInfo.s1Outer

    blockPerCore = (constInfo.totalBlockNum + coreNum - 1) // coreNum
    start = cBlockIdx * blockPerCore
    lin = pl.max(pl.min(start + blockPerCore, constInfo.totalBlockNum) - start, 0)
    return sel(constInfo.enableSwizzle, swz, lin)


def global_idx(constInfo, cBlockIdx, coreNum, localIdx):
    """局部块序号 -> 全局块下标。"""
    swz = swizzle_idx(constInfo, cBlockIdx, coreNum, localIdx)
    blockPerCore = (constInfo.totalBlockNum + coreNum - 1) // coreNum
    lin = cBlockIdx * blockPerCore + localIdx
    return sel(constInfo.enableSwizzle, swz, lin)


def s2_group_of(constInfo, idx):
    """(b, n2, g, s2o) 的组合编号：同一编号内 dK/dV 可在 L0C 上持续累加。

    块网格是 b -> n2 -> g -> s2o -> s1o，一组恰为「连续 s1Outer 个块」，
    故组号就是 idx // s1Outer —— 已融合 (b,n2,g,s2o) 四维，无需再逐维还原。
    GQA 下跨 g 即换组(g 在 s2o 外侧)，dK/dV 会先 flush 再重新累加；G 个 Q
    head 对同一 KV head 的贡献靠 GM 上的 atomicAdd 合并 —— 与 ASC 的
    BN2GS1S2 模板 needAtomic 恒真一致。
    """
    return idx // constInfo.s1Outer


def dkv_acc_flag(constInfo, gIdx, localIdx):
    """本轮的 dK/dV 是否应在 L0C 上累加(1)还是覆写(0)。

    swizzle: 本核连续处理同一列(同一 (b,n2,s2o) 组)的 s1Outer 个块，故
             localIdx % s1Outer == 0 即换组那一块，必须覆写；其余累加。
             这天然覆盖「本核第一块」(localIdx==0 也是列首) —— 漏掉这条
             时 L0C 残留值会污染 dK/dV(实测 max_diff 达 57)。
    线性:    需同时满足「非本核首块」与「与前一块同组」。
    """
    swz = pl.min(localIdx % constInfo.s1Outer, 1)
    notFirst = pl.min(localIdx, 1)
    gCur = s2_group_of(constInfo, gIdx)
    gBefore = s2_group_of(constInfo, pl.max(gIdx - 1, 0))
    sameGroup = 1 - pl.min(gCur - gBefore, 1)
    lin = pl.min(notFirst, sameGroup)
    return sel(constInfo.enableSwizzle, swz, lin)


def need_load_kv(constInfo, gIdx, localIdx):
    """是否需要重新从 GM 载入 K/V(1=需要，0=沿用 L1 里的)。

    与 dkv_acc_flag 互补：只有换组那一块要重载，组内其余块 K/V 相同。
    对应 ASC block_cube.h:529 的 isCopyRight。8192 场景 s1Outer=64，
    K/V 的 GM 读次数降到 1/64。
    """
    swz = 1 - pl.min(localIdx % constInfo.s1Outer, 1)
    isFirst = 1 - pl.min(localIdx, 1)
    gCur = s2_group_of(constInfo, gIdx)
    gPrev = s2_group_of(constInfo, pl.max(gIdx - 1, 0))
    lin = pl.max(isFirst, pl.min(gCur - gPrev, 1))
    return sel(constInfo.enableSwizzle, swz, lin)


def next_s2_same(constInfo, gIdx, localIdx):
    """下一块是否仍落在同一个 s2 块内(1=是，0=否)。"""
    swz = pl.min((localIdx + 1) % constInfo.s1Outer, 1)
    lin = 1 - pl.min(s2_group_of(constInfo, gIdx + 1) - s2_group_of(constInfo, gIdx), 1)
    return sel(constInfo.enableSwizzle, swz, lin)


def set_run_info(constInfo, runInfo, taskId, index, subIdx):
    # index -> (boIdx, n2oIdx, goIdx, s2oIdx, s1oIdx)，S1 为最快轴。
    # 维度顺序 b -> n2 -> g -> s2o -> s1o，与 ASC 的
    # GetNextValidIdx 一致（g 在 n2 之内、s2o 之外）。
    bDimTail = index % constInfo.n2GS1oS2o
    n2DimTail = bDimTail % constInfo.gS1oS2o
    gDimTail = index % constInfo.s1oS2o
    runInfo.boIdx = index // constInfo.n2GS1oS2o
    runInfo.n2oIdx = bDimTail // constInfo.gS1oS2o
    runInfo.goIdx = n2DimTail // constInfo.s1oS2o
    runInfo.s2oIdx = gDimTail // constInfo.s1Outer
    runInfo.s1oIdx = gDimTail % constInfo.s1Outer

    # 尾块真实大小。不能用分支赋值(IfStmt 无法 phi 合并)，也不能用比较结果
    # 参与算术(bool dtype 不被 mul 接受)，故写成 min 截断的形式
    s1Real = pl.min(CUBE_BASEM, constInfo.s1Size - runInfo.s1oIdx * CUBE_BASEM)
    s2Real = pl.min(CUBE_BASEN, constInfo.s2Size - runInfo.s2oIdx * CUBE_BASEN)
    runInfo.s1RealSize = s1Real
    runInfo.s2RealSize = s2Real
    runInfo.taskIdMod2 = taskId % 2

    # vector 侧两个子核各承担半个 S1：
    # subIdx=0 取 firstHalf，subIdx=1 取剩余部分
    # 两个 vector 子核的分界固定为 VECTOR_BASEM(64)，不能按 s1Real 折半：
    # mm1/mm2 结果由 fixpipe 以 DualModeSplitM 写出，硬件在第 64 行处切分，
    # 子核 0 恒拿块内 0..63 行、子核 1 恒拿 64..127 行。若分界跟着尾块变小，
    # UB 里的数据与 VF 的行号就错位(实测尾块场景 dQ/dK/dV 全错)。
    # 固定 64 同时天然满足 dS^T/P^T 的 NZ 分形 16 对齐要求。
    firstHalf = VECTOR_BASEM
    runInfo.firstHalfS1RealSize = firstHalf
    # 本子核真实行数：尾块可能不足，甚至为 0(s1Real <= 64 时子核 1 无事可做)
    runInfo.halfS1RealSize = pl.max(pl.min(s1Real - subIdx * firstHalf, firstHalf), 0)

    # BSND 布局的 GM 偏移
    runInfo.queryOffset = runInfo.s1oIdx * CUBE_BASEM
    runInfo.keyOffset = runInfo.s2oIdx * CUBE_BASEN

    # isS2IdxNoChange 决定 dK/dV 能否在 L0C 上累加。本配置下
    # IS_DKV_RESIDENT_L0C 为 true，但当前实现仍走 atomicAdd，
    # 故该字段暂只对齐结构，待 L0C 驻留优化时启用。
    # 由 index-1 直接推出上一块的 s2oIdx，避免跨迭代的可变量
    # (循环里带条件赋值的标量会触发 IfStmt phi 合并失败)
    prevS2oIdx = (index - 1) % constInfo.s1oS2o // constInfo.s1Outer
    runInfo.isS2IdxNoChange = runInfo.s2oIdx == prevS2oIdx

    # 统一四维视图上的两个非 S 轴索引，纯算术无分支。
    # Q 侧的 head 是 n2oIdx*gSize + goIdx（共 n1 个），KV 侧是 n2oIdx（共 n2 个）
    # —— 对应 ASC 里 GetQueryOffset 含 gOffset 而 GetKeyOffset 不含。
    headQ = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx
    runInfo.idx0Q = runInfo.boIdx * constInfo.coefB0Q + headQ * constInfo.coefN0
    runInfo.idx2Q = headQ * constInfo.coefN2
    runInfo.idx0KV = (
        runInfo.boIdx * constInfo.coefB0KV + runInfo.n2oIdx * constInfo.coefN0
    )
    runInfo.idx2KV = runInfo.n2oIdx * constInfo.coefN2


# ==================================================================
#  IterateMmQK —— mm2: S = Q @ K^T   (block_cube.h::IterateMmQK)
# ==================================================================
def _mm_k_chunk(lt, rt, at, srcL, srcR, dOff, dW, s2Real, isFirst):
    """MM2/MM1 的一个 K 分块：D 落在 K 轴，故各块要累加到同一个 acc。

    dOff/dW 是 trace 期常量；L1 tile 不能切片，取子块用 pl.move 的 offset。
    M 方向不裁剪的理由见调用处。
    """
    pl.set_validshape(lt, [CUBE_BASEM, dW])
    pl.set_validshape(rt, [dW, s2Real])
    pl.set_validshape(at, [CUBE_BASEM, s2Real])
    pl.move(lt, srcL, [0, dOff])  # [M,D] 的列子块
    pl.move(rt, srcR, [dOff, 0])  # [D,S2] 的行子块
    if isFirst:
        pl.matmul(at, lt, rt)
    else:
        pl.matmul_acc(at, at, lt, rt)


def iterate_mm_qk(
    constInfo, runInfo, tensor_q, tensor_k, q_t, k_l1, left, right, acc, mm2_ub, loadKV
):
    k_t = k_l1.current()
    # 尾块：load 允许 offset+shape 越过 tensor 边界，但必须用 valid_shape
    # 把真实行列数告知硬件，否则读到界外触发 507015
    # 列方向按真实 D(dSize)裁剪：tile 按 d_align 分配，D 可能更短。
    pl.set_validshape(q_t, [runInfo.s1RealSize, constInfo.dSize])
    # k_l1 以 [D,S2] 载入，故 S2 是它的列
    pl.set_validshape(k_t, [constInfo.dSize, runInfo.s2RealSize])
    # BSND -> tile[S,D]: order=[1,3] 取 (S,D) 两轴
    pl.load(
        q_t,
        tensor_q,
        [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q, 0],
        order=[1, 3],
    )
    # 右矩阵转置(isRightTranspose=true)：K[S2,D] 以 [D,S2] 载入。
    # 仅在 s2 组切换时重载：S1 是最快轴，同组内连续 s1Outer 个 task 的 K
    # 完全相同(对应 ASC isCopyRight = !isS2IdxNoChange)。
    if loadKV == 1:
        pl.load(
            k_t,
            tensor_k,
            [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV, 0],
            order=[3, 1],
        )
    lt = left.next()
    rt = right.next()
    at = acc.next()
    # M 方向不裁剪(保持 CUBE_BASEM)：mm1/mm2 的结果经 DualModeSplitM 写出，
    # 硬件在「有效行数的一半」处切分。若把 M 裁到 s1Real，切分点就会跟着
    # 尾块漂移，与 vector 侧固定 64 的分界错位(实测尾块结果全错)。
    # 保持 M=128 则切分点恒为 64；越界行只是 L1 中的陈旧数据，
    # 既不写回 GM(store 按 s1Real 裁剪)，也不影响有效行的结果。
    #
    # D 轴分块：D=128 档只有一块(等价于改动前的单次 matmul)；D=192 档两块
    # 96+96，第二块用 matmul_acc 累加到同一个 acc。第二块的 lt/rt 走另一个
    # L0 槽位(next() 轮转)，故它的 L1->L0 搬运与第一块的 matmul 重叠。
    _mm_k_chunk(
        lt,
        rt,
        at,
        q_t,
        k_t,
        D_CHUNK_OFF0[d_align],
        D_CHUNK_W0[d_align],
        runInfo.s2RealSize,
        True,
    )
    if D_NCHUNKS[d_align] > 1:
        _mm_k_chunk(
            left.next(),
            right.next(),
            at,
            q_t,
            k_t,
            D_CHUNK_OFF1[d_align],
            D_CHUNK_W1[d_align],
            runInfo.s2RealSize,
            False,
        )
    # Fixpipe L0C->UB，dualDstCtl=1：按 M 切分写入两个 vector 子核
    pl.move(mm2_ub, at, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


# ==================================================================
#  IterateMmDyV —— mm1: dP = dO @ V^T  (block_cube.h::IterateMmDyV)
# ==================================================================
def iterate_mm_dyv(
    constInfo,
    runInfo,
    tensor_dy,
    tensor_v,
    do_t,
    v_l1,
    left,
    right,
    acc,
    mm1_ub,
    loadKV,
):
    v_t = v_l1.current()
    # 这里的 K 轴是 **Dv** 而不是 D：dO/V 都是 Dv 宽。
    pl.set_validshape(do_t, [runInfo.s1RealSize, constInfo.dvSize])
    pl.set_validshape(v_t, [constInfo.dvSize, runInfo.s2RealSize])
    pl.load(
        do_t,
        tensor_dy,
        [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q, 0],
        order=[1, 3],
    )
    # 同 mm2 的 K：仅在 s2 组切换时重载
    if loadKV == 1:
        pl.load(
            v_t,
            tensor_v,
            [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV, 0],
            order=[3, 1],
        )
    lt = left.next()
    rt = right.next()
    at = acc.next()
    # M 方向不裁剪(保持 CUBE_BASEM)：mm1/mm2 的结果经 DualModeSplitM 写出，
    # 硬件在「有效行数的一半」处切分。若把 M 裁到 s1Real，切分点就会跟着
    # 尾块漂移，与 vector 侧固定 64 的分界错位(实测尾块结果全错)。
    # 保持 M=128 则切分点恒为 64；越界行只是 L1 中的陈旧数据，
    # 既不写回 GM(store 按 s1Real 裁剪)，也不影响有效行的结果。
    #
    # 分块宽度按 Dv 截断：Dv 可以短于 D，甚至短于第一块的宽度。
    # 第二块可能整块落在 Dv 之外(宽度 <= 0)，此时整块跳过 —— 这个判断的
    # 取值在整个 kernel 内不变(dvSize 是常量)，故各 task 的 next() 调用
    # 次数一致，游标记账不会错位。
    _mm_k_chunk(
        lt,
        rt,
        at,
        do_t,
        v_t,
        D_CHUNK_OFF0[d_align],
        pl.min(D_CHUNK_W0[d_align], constInfo.dvSize),
        runInfo.s2RealSize,
        True,
    )
    if D_NCHUNKS[d_align] > 1:
        # next() 必须无条件调用：把它放进运行期 if 里会让游标推进次数随分支
        # 变化，current() 就会指错槽位(process_post 的注释记录过这个坑)。
        # 故先无条件轮转，再用运行期宽度决定这一块是否真的参与计算。
        lt1 = left.next()
        rt1 = right.next()
        # Dv 可能短到让第二块整块落在有效列之外(如 D=192/Dv=96)，此时宽度
        # 截到 0，整块跳过 —— 宽度 0 的 matmul 既非法，也会把 padding 里的
        # 陈旧数据算进结果。判断只依赖 dvSize，在整个 kernel 内是同一个值。
        dvRem = constInfo.dvSize - D_CHUNK_OFF1[d_align]
        dvW1 = pl.max(pl.min(D_CHUNK_W1[d_align], dvRem), 0)
        if dvW1 > 0:
            _mm_k_chunk(
                lt1,
                rt1,
                at,
                do_t,
                v_t,
                D_CHUNK_OFF1[d_align],
                dvW1,
                runInfo.s2RealSize,
                False,
            )
    pl.move(mm1_ub, at, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


# ==================================================================
#  IterateMmDsK —— mm3: dQ = dS @ K    (block_cube.h::IterateMmDsK)
#  IS_DQ_WRITE_UB=false -> fixpipe 直出 GM，needAtomic=true(BN2GS1S2)
# ==================================================================
def _mm_n_chunk(
    left, right, acc, srcL, srcR, tensor_out, outIdx, mReal, kReal, dOff, dW
):
    """MM3 的一个输出列分块：D 在 N 轴，各块列区间不相交，无需跨块累加。

    outIdx 是统一四维视图上的前三个索引；第 4 维(D 轴)用 dOff 定位本块。
    """
    lt = left.next()
    rt = right.next()
    at = acc.next()
    pl.set_validshape(lt, [mReal, kReal])
    pl.set_validshape(rt, [kReal, dW])
    pl.set_validshape(at, [mReal, dW])
    pl.move(lt, srcL)
    pl.move(rt, srcR, [0, dOff])  # K[S2,D] 的列子块
    pl.matmul(at, lt, rt)
    pl.store(
        tensor_out,
        at,
        [outIdx[0], outIdx[1], outIdx[2], dOff],
        order=[1, 3],
        atomic=pl.AtomicType.AtomicAdd,
    )


def iterate_mm_dsk(
    constInfo, runInfo, tensor_k, tensor_dq_ws, loadKV, ds_l1, k_l1, left, right, acc
):
    k_t = k_l1.current()
    # 右矩阵不转置(isRightTranspose=false)：K 以 [S2,D] 载入。
    # order 仍取 (S,D) 两轴；此前 dQ 算成 dS@K^T 的根因是 k_l1_n 的 layout
    # 与 L0B tile 不一致(见其声明处注释)，改 order 无效。
    # MM3 的右矩阵是未转置的 K[S2,D]，S2 是它的行
    pl.set_validshape(k_t, [runInfo.s2RealSize, constInfo.dSize])
    # 同 mm2/mm1：仅在 s2 组切换时重载(mm3 的 K 是非转置视图)
    if loadKV == 1:
        pl.load(
            k_t,
            tensor_k,
            [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV, 0],
            order=[1, 3],
        )
    # dS[s1Real, s2Real] @ K[s2Real, D] -> dQ[s1Real, D]
    #
    # 这里 D 落在 **N 轴**(输出列)上，与 mm2/mm1 相反：各 D 分块算出的是
    # dQ 互不相交的列区间，故**不需要跨块累加**，每块独立算、独立写回自己
    # 的列范围(store 的第 4 个索引即列偏移)。
    _mm_n_chunk(
        left,
        right,
        acc,
        ds_l1,
        k_t,
        tensor_dq_ws,
        [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q],
        runInfo.s1RealSize,
        runInfo.s2RealSize,
        D_CHUNK_OFF0[d_align],
        D_CHUNK_W0[d_align],
    )
    if D_NCHUNKS[d_align] > 1:
        _mm_n_chunk(
            left,
            right,
            acc,
            ds_l1,
            k_t,
            tensor_dq_ws,
            [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q],
            runInfo.s1RealSize,
            runInfo.s2RealSize,
            D_CHUNK_OFF1[d_align],
            D_CHUNK_W1[d_align],
        )


# ==================================================================
#  IterateMmDsQ —— mm4: dK = dS^T @ Q  (block_cube.h::IterateMmDsQ)
#  左矩阵取已在 vector 侧转置好的 dS^T L1
# ==================================================================
def _mm_resident_chunk(left, right, at, srcL, srcR, mReal, kReal, dOff, dW, isAcc):
    """MM4/MM5 的一个输出列分块，累加进常驻 L0C 的对应那一块。

    D 在 N 轴上，故各块写 dK/dV 互不相交的列区间；但每块自己要跨 s1 轮次
    累加，所以每块各有一块常驻 L0C(由调用方按分块 next() 取到 at)。
    """
    lt = left.next()
    rt = right.next()
    pl.set_validshape(lt, [mReal, kReal])
    pl.set_validshape(rt, [kReal, dW])
    pl.set_validshape(at, [mReal, dW])
    pl.move(lt, srcL)
    pl.move(rt, srcR, [0, dOff])
    # 同一 s2 块的后续 s1 轮次在 L0C 上累加，首轮直接覆写
    if isAcc == 1:
        pl.matmul_acc(at, at, lt, rt)
    else:
        pl.matmul(at, lt, rt)


def iterate_mm_dsq(constInfo, runInfo, ds_t_l1, q_t, left, right, acc0, acc1, isAcc):
    # q_t 来自 q_l1_dk(延后两个 task 的游标视图)，即 mm2 在 task N-2
    # already 载入好的那一份 Q，无需再读 GM
    # dS^T[s2Real, s1Real] @ Q[s1Real, D] -> dK[s2Real, D]
    # dK 的宽度是 D(与 Q 同宽)，按 D 分块，每块累加进自己的常驻 L0C。
    _mm_resident_chunk(
        left,
        right,
        acc0.current(),
        ds_t_l1,
        q_t,
        runInfo.s2RealSize,
        runInfo.s1RealSize,
        D_CHUNK_OFF0[d_align],
        D_CHUNK_W0[d_align],
        isAcc,
    )
    if D_NCHUNKS[d_align] > 1:
        _mm_resident_chunk(
            left,
            right,
            acc1.current(),
            ds_t_l1,
            q_t,
            runInfo.s2RealSize,
            runInfo.s1RealSize,
            D_CHUNK_OFF1[d_align],
            D_CHUNK_W1[d_align],
            isAcc,
        )


# ==================================================================
#  IterateMmPDy —— mm5: dV = P^T @ dO  (block_cube.h::IterateMmPDy)
# ==================================================================
def dv_chunk_w(constInfo, chunkIdx):
    """dV 第 chunkIdx 块的有效列宽(按 Dv 截断，可能为 0)。

    dV 的宽度是 Dv 而不是 D，故最后一块可能部分甚至整块落在有效列之外。
    只依赖 dvSize，在整个 kernel 内是同一个值。
    """
    if chunkIdx == 0:
        return pl.max(
            pl.min(D_CHUNK_W0[d_align], constInfo.dvSize - D_CHUNK_OFF0[d_align]), 0
        )
    return pl.max(
        pl.min(D_CHUNK_W1[d_align], constInfo.dvSize - D_CHUNK_OFF1[d_align]), 0
    )


def iterate_mm_pdy(constInfo, runInfo, p_t_l1, do_t, left, right, acc0, acc1, isAcc):
    # do_t 同理来自 do_l1_dv，复用 mm1 在 task N-2 载入的 dO
    # P^T[s2Real, s1Real] @ dO[s1Real, Dv] -> dV[s2Real, Dv]
    # 注意 dV 的宽度是 **Dv**，按 D 的分块表切但每块宽度要按 Dv 截断；
    # 某块整块落在 Dv 之外时宽度为 0，跳过(见 dv_chunk_w)。
    w0 = dv_chunk_w(constInfo, 0)
    if w0 > 0:
        _mm_resident_chunk(
            left,
            right,
            acc0.current(),
            p_t_l1,
            do_t,
            runInfo.s2RealSize,
            runInfo.s1RealSize,
            D_CHUNK_OFF0[d_align],
            w0,
            isAcc,
        )
    if D_NCHUNKS[d_align] > 1:
        w1 = dv_chunk_w(constInfo, 1)
        if w1 > 0:
            _mm_resident_chunk(
                left,
                right,
                acc1.current(),
                p_t_l1,
                do_t,
                runInfo.s2RealSize,
                runInfo.s1RealSize,
                D_CHUNK_OFF1[d_align],
                w1,
                isAcc,
            )


# ==================================================================
#  CopyUB2L1 —— 对应 block_vec.h::CopyUB2L1
#  pypto 无 Vec->Mat 的 move，必须 cast -> ND2NZ(Vec->Vec) -> insert
#  insert 的源必须已是 NZ，否则能编过但结果错
# ==================================================================
# NZ 分形：fp16 的 C0 = 16 元素 = 32B(一个 DataBlock)
FRACTAL_C0 = 16
UB_BLOCK_BYTES = 32
# 一个分形列占 VECTOR_BASEM*C0 个 fp16 = (64*16)*2/32 = 64 个 32B block，
# 末尾 +1 垫一个 block 错开 bank。对应 ASC:
#   blockStride = (srcM * blockN) * sizeof(T1) / blockSize + 1
NZP_BLOCK_STRIDE = (VECTOR_BASEM * FRACTAL_C0) * 2 // UB_BLOCK_BYTES + 1  # 65
NZP_REPEAT_STRIDE = 1
NZP_ROWS = VECTOR_BASEM + 1  # 65

# V1 之后那几块 UB 的地址表，按 dv_align 分档(见 _build_ub)。
# 放在这里而不是 _build_ub 旁边：它要用上面的 NZP_ROWS。
_UB_TAB = {w: _build_ub(w) for w in (128, 192)}
UB_Y_A = {w: _UB_TAB[w][0] for w in _UB_TAB}
UB_DY_A = {w: _UB_TAB[w][1] for w in _UB_TAB}
UB_PROD_A = {w: _UB_TAB[w][2] for w in _UB_TAB}
UB_TMP_A = {w: _UB_TAB[w][3] for w in _UB_TAB}
UB_NZP_A = {w: _UB_TAB[w][4] for w in _UB_TAB}
UB_SFMG_A = {w: _UB_TAB[w][5] for w in _UB_TAB}
UB_SFMG_HI_A = {w: _UB_TAB[w][6] for w in _UB_TAB}
UB_LSE0_A = {w: _UB_TAB[w][7] for w in _UB_TAB}
UB_LSE1_A = {w: _UB_TAB[w][8] for w in _UB_TAB}


def copy_ub2l1(l1_tile, nzp_ub, row_offset):
    """把 NZ+1 缓冲搬进 CV 共享 L1，跳过 bank 错开用的填充块。

    valid_shape 固定设为 [VECTOR_BASEM, N]=[64,128]，框架据此算出源跨距
    65-64=1，正好跳过每个分形列末尾那一个填充 block。

    为什么不按尾块收窄到 halfS1RealSize：两个 vector 子核分别 insert 到
    rowOff=0 与 64，各占 L1 的 64 行、互不重叠，故超出 s1Real 的脏行只落在
    本子核自己那半区内，随后由 matmul 的 valid_shape 与 store 裁掉。

    搬完把 valid_shape 恢复成 [65, N]，让下一轮 VF 仍按 65 行跨距写入。
    """
    pl.set_validshape(nzp_ub, [VECTOR_BASEM, VECTOR_BASEN])
    pl.insert(l1_tile, nzp_ub, [row_offset, 0])
    pl.set_validshape(nzp_ub, [NZP_ROWS, VECTOR_BASEN])


# 注意：CV 共享的 dS/P L1(NZ) 一律不设 valid_shape。
# NZ 是分形排布，缩小行数会改变分形打包的跨距，导致 vector 侧 insert 与
# cube 侧读取对不上 —— 实测 S2 尾块下 dQ(按列裁剪的 ds_l1)正确，
# 而 dK/dV(按行裁剪的 ds_t_l1/p_t_l1)错误(max 0.68/0.90)。
# 矩阵乘的实际计算范围由 L0A/L0B/acc 的 valid_shape 决定，输出再由 store
# 按 s1Real/s2Real 裁剪，故界外脏数据不会流入 GM，L1 侧无需裁剪。


# ==================================================================
#  ProcessVec1 —— v1: sfmg = rowsum(dy * y)
#  对应 vector_api/cast_softmax_grad.h::CopyInSoftmaxGrad +
#  MySoftmaxGradFrontCast。y 为前向输出，按 SFMG_LOOP_SIZE 分段。
# ==================================================================
@pl.vector_function
def softmax_grad_front_mul_vf(
    prod_ub, y_ub, dy_ub, srcM: pl.DT_INT64, rowShift: pl.DT_INT64
):
    """prod = cast(dy) * cast(y)，逐行做逐元素乘（不含 reduce）。

    对应 AscendC CastAligned256F16VF128 的乘法部分：fp16 一次载入 128 个数，
    按 Even/Odd 拆成两个 64 lane 的 fp32 寄存器分别相乘。
    行内求和交给 tile 级 pl.sum：VF 的 store 目标地址不能随运行期循环变量
    变化（实测 507035），而逐行标量写回正需要这种寻址。
    注意 astype(ZERO)/astype(ONE) 得到的是「去交错」的偶/奇两半，
    所以两个乘积寄存器要按 even/odd 半区写回，保持与源同序。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    # 按「平铺的 128 元素段」遍历整块 tile，而不是按行 —— 一次 load_align +
    # Even/Odd 拆分正好覆盖 128 个 fp16，而 dv_align=192 时一行是 1.5 段，
    # 按行发固定条数的指令覆盖不全。
    #
    # 关键：源(y/dy)与目的(prod)的**行距都是 dv_align**，所以平铺段号在两边
    # 是同一套坐标，逐段搬运天然保持行内/行间位置一致 —— 不需要知道每行多宽。
    # 总段数按整块 tile 算(不是按 srcM 行)：多算的越界行由 pl.sum 前的
    # valid_shape 裁掉(见 process_vec1_chunk)，比在 VF 里按行判断便宜。
    # 两档都整除：32*128/128=32，32*192/128=48。
    nsegs = SFMG_LOOP_SIZE * dv_align // (2 * VF_LANES)
    # rowShift 是「源被钳位后要补偿的行数」，换算成元素偏移
    base = rowShift * dv_align
    for t in pl.range(0, nsegs):
        # rowShift：y/dy 的 GM 读取起点被钳位到界内(见 process_vec1_chunk)
        vregYHalf = vf.load_align(y_ub, base + t * (2 * VF_LANES))
        vregDyHalf = vf.load_align(dy_ub, base + t * (2 * VF_LANES))
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        # 用 INTLV_B32 一条指令把偶/奇两个寄存器**交错**写回，还原成与源
        # 完全相同的元素顺序。
        #
        # 这里必须还原顺序，不能像原先那样把两半分别落在本段的前/后 64 个
        # 位置。原因是 astype(ZERO/ONE) 的去交错是**跨整段 128 个元素**的，
        # 而一个段并不总等于一行：
        #   dv_align=128 时一段恰好一行，打乱只发生在行内，而后面按行求和
        #   对行内顺序不敏感，故看不出问题(这就是 Dv=128 一直是对的原因)；
        #   dv_align=192 时一行是 1.5 段，段跨越行边界，去交错会把元素搬到
        #   **另一行**去，行和随之全错 —— 且 pl.sum 前按 dvSize 裁列也裁错
        #   了位置。实测 Dv=96/160/192 三档 dQ/dK/dV 全错，唯 Dv=128 全对。
        # 不能用 post_update：它跨迭代累加地址，实测得到量级失控的脏数据。
        # 注意 prod 的写回**不带 rowShift**：rowShift 只是补偿源(y/dy)被
        # 钳位的读取起点，prod/sum 始终从第 0 行开始用。
        vf.store_align(
            prod_ub + t * (2 * VF_LANES),
            vregMul,
            vregMul2,
            pregFullExe,
            dist=pl.StoreDist.INTLV_B32,
        )


def process_vec1_chunk(
    constInfo,
    runInfo,
    s1Base,
    rowOff,
    rows,
    tensor_y,
    tensor_dy,
    y_ub,
    dy_ub,
    prod_ub,
    tmp_ub,
    sfmg_chunk_ub,
):
    # y/dy 的 tile 是 SFMG_LOOP_SIZE 行，load 总是搬满这么多行。
    # S1 非对齐时起点 + 32 会越过 s1Size，故同 process_vec2 的做法：
    # 把起点钳位到界内，再用 rowShift 在 VF 内补偿索引。
    rowBase = s1Base + rowOff
    loadBase = pl.max(pl.min(rowBase, constInfo.s1Size - SFMG_LOOP_SIZE), 0)
    rowShift = rowBase - loadBase
    pl.load(y_ub, tensor_y, [runInfo.idx0Q, loadBase, runInfo.idx2Q, 0], order=[1, 3])
    pl.load(dy_ub, tensor_dy, [runInfo.idx0Q, loadBase, runInfo.idx2Q, 0], order=[1, 3])
    softmax_grad_front_mul_vf(prod_ub, y_ub, dy_ub, rows, rowShift)
    # dim=0：沿最后一轴(Dv)做行向 reduce，得到 [rows, 1]。
    # 必须把有效宽度裁到真实的 Dv：tile 按 dv_align 声明，Dv 更短时右侧
    # 是 padding，pl.sum 会把它一起加进 sfmg。
    # 行数也裁到 rows —— 越界行的乘积是钳位读来的陈旧数据。
    pl.set_validshape(prod_ub, [rows, constInfo.dvSize])
    pl.sum(sfmg_chunk_ub, prod_ub, tmp_ub, dim=0)
    pl.set_validshape(prod_ub, [SFMG_LOOP_SIZE, dv_align])


def process_vec1(
    constInfo,
    runInfo,
    subIdx,
    tensor_y,
    tensor_dy,
    y_ub,
    dy_ub,
    prod_ub,
    tmp_ub,
    sfmg_lo_ub,
    sfmg_hi_ub,
):
    """softmaxGradFront: sfmg = rowsum(dy * y)。

    按 SFMG_LOOP_SIZE 行分两段：两段的 sfmg 结果地址连续，
    上层用一个 [VECTOR_BASEM,1] 视图整体参与后续 broadcast。
    分段数是编译期常量，但 kernel 内 for 必须用 pl.range，
    故直接展开两次调用。
    """
    s1Base = runInfo.queryOffset + subIdx * runInfo.firstHalfS1RealSize
    rowsLo = pl.min(SFMG_LOOP_SIZE, runInfo.halfS1RealSize)
    rowsHi = runInfo.halfS1RealSize - rowsLo
    process_vec1_chunk(
        constInfo,
        runInfo,
        s1Base,
        0,
        rowsLo,
        tensor_y,
        tensor_dy,
        y_ub,
        dy_ub,
        prod_ub,
        tmp_ub,
        sfmg_lo_ub,
    )
    process_vec1_chunk(
        constInfo,
        runInfo,
        s1Base,
        SFMG_LOOP_SIZE,
        rowsHi,
        tensor_y,
        tensor_dy,
        y_ub,
        dy_ub,
        prod_ub,
        tmp_ub,
        sfmg_hi_ub,
    )


# ==================================================================
#  ProcessVec2 —— v2: P = simpleSoftmax(S * scale, max, sum)
#  对应 vector_api/pse_atten_mask_muls_simple_softmax.h
#  无 pse / attenMask，仅 muls + simpleSoftmax
# ==================================================================
@pl.vector_function
def muls_simple_softmax_vf(
    mm2_ub, lse_ub, scale: pl.DT_FP32, srcM: pl.DT_INT64, rowShift: pl.DT_INT64
):
    """P = exp(S * scale - lse)，逐行处理。

    对标 Dao-AILab/flash-attention 的 LSE 约定：前向输出
      lse = max + log(sum(exp(s - max)))            (自然对数)
    故反向只需一次 exp_sub 就得到归一化后的 P，行和恒为 1 ——
    相比原先的 exp(S*scale - max) / sum 省掉一条 vf.div，
    也少读一个 GM 张量。两式数值差异约 6e-08。

    lse 用 BRC_B32 广播载入，S 按 VECTOR_BASEN=128 分两个 64 lane 寄存器。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for m in pl.range(0, srcM):
        # rowShift：lse 的 GM 读取起点被钳位到界内，逻辑第 m 行
        # 实际落在 tile 的 m + rowShift 处（见 process_vec2）
        vregLse = vf.load_align(lse_ub, m + rowShift, dist=pl.LoadDist.BRC_B32)
        for n in pl.range(0, VECTOR_BASEN // VF_LANES):
            vregSrc = vf.load_align(mm2_ub, m * VECTOR_BASEN + n * VF_LANES)
            vregMuls = vf.muls(vregSrc, scale, pregFullExe)
            vregExp = vf.exp_sub(vregMuls, vregLse, pregFullExe)
            vf.store_align(
                mm2_ub + m * VECTOR_BASEN + n * VF_LANES, vregExp, pregFullExe
            )


def process_vec2(constInfo, runInfo, subIdx, tensor_lse, mm2_ub, lse_ub):
    s1Base = runInfo.queryOffset + subIdx * runInfo.firstHalfS1RealSize
    # lse 的 tile 是 [VECTOR_BASEM,1]，load 总是搬 VECTOR_BASEM 行。
    # S1 非对齐时 s1Base + 64 会越过 s1Size(实测 AIV 507015)，故把起点
    # 钳位到界内，再把差值 rowShift 传给 VF 做索引补偿 —— 这样搬入的行
    # 一定在界内，且逻辑行号仍对得上。
    loadBase = pl.min(s1Base, constInfo.s1Size - VECTOR_BASEM)
    loadBase = pl.max(loadBase, 0)
    rowShift = s1Base - loadBase
    # CopyLse: 前向的 lse 直接复用，无需再求极值或行和
    # lse 形状 [B,N1,S1]，按 (b, n2*G+g, s1) 三维索引 —— head 维用 Q 侧的
    # n1，与 ASC 的 ((b*n2+n2o)*G+g)*S1 一致；order=[2,0] 指明搬运沿
    # S1(轴 2) 展开、落到 tile 的行方向
    headQ = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx
    pl.load(lse_ub, tensor_lse, [runInfo.boIdx, headQ, loadBase], order=[2, 0])
    muls_simple_softmax_vf(
        mm2_ub, lse_ub, constInfo.scaleValue, runInfo.halfS1RealSize, rowShift
    )


# ==================================================================
#  ProcessVec3 —— v3: dS = (dP - sfmg) * P，随后 cast + ND2NZ 落 L1
#  对应 block_vec.h::ProcessVec3 的 BroadcastSubMul +
#  CastTransdataDeconflict + CopyUBToL1Vec3
#  同时产出 dS^T L1，供 MM4 使用
# ==================================================================
@pl.vector_function
def broadcast_sub_mul_cast_vf(
    nzp_ub, mm1_ub, mm2_ub, sfmg_ub, scale: pl.DT_FP32, srcM: pl.DT_INT64
):
    """dS = (dP - broadcast(sfmg)) * P，并 cast 成 fp16。

    对应 AscendC BroadcastSubMulVF128 + CastTransdataDeconflict 的算术部分：
    sfmg 按行 BRC_B32 广播，128 列拆成两个 64 lane 寄存器；
    两个 fp32 寄存器 cast 后交错写回一个 fp16 行。
    注意此处不乘 scale，与 AscendC 一致(scale 在 dQ/dK 后处理施加)。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for m in pl.range(0, srcM):
        vregSfmg = vf.load_align(sfmg_ub, m, dist=pl.LoadDist.BRC_B32)
        # 用 DINTLV_B32 一次载入 128 个 fp32 并拆成 even/odd，
        # 使后续 Even/Odd cast + Or 的合并顺序正确；
        # sub/mul 为逐元素运算，只要 dP/P 拆分方式一致即可
        vregDpEven, vregDpOdd = vf.load_align(
            mm1_ub, m * VECTOR_BASEN, dist=pl.LoadDist.DINTLV_B32
        )
        vregPEven, vregPOdd = vf.load_align(
            mm2_ub, m * VECTOR_BASEN, dist=pl.LoadDist.DINTLV_B32
        )
        vregSub0 = vf.sub(vregDpEven, vregSfmg, pregFullExe)
        vregDs0 = vf.mul(vregSub0, vregPEven, pregFullExe)
        vregSub1 = vf.sub(vregDpOdd, vregSfmg, pregFullExe)
        vregDs1 = vf.mul(vregSub1, vregPOdd, pregFullExe)
        # 这里不乘 scale：scale 必须在矩阵乘之后、fp32 结果上施加
        # (post 阶段 muls_cast_dqkv)。虽然 scale*(dS@K)==(scale*dS)@K
        # 数学等价，但先乘会让 fp16 舍入发生在不同量级上，训练中误差会累积。
        # ASC 同样放在 post 的 fp32 workspace 上做(post_regbase.h:238)。
        # fp32 -> fp16：Even/Odd 分别 cast 进 b16 寄存器的两个半区再 Or 合并，
        # 与 vf_cast_transdata_deconflict.h 一致（probe 验证 MAXDIFF 0.0）
        vregCastEven = vf.astype(
            vregDs0, pregFullExeB16, dtype=pl.DT_FP16, layout=pl.CastLayout.ZERO
        )
        vregCastOdd = vf.astype(
            vregDs1, pregFullExeB16, dtype=pl.DT_FP16, layout=pl.CastLayout.ONE
        )
        vregCastRes = vf.or_(vregCastEven, vregCastOdd, pregFullExeB16)
        # DataBlock 拷贝模式(vsstb)：一条 store 同时完成 ND->NZ 重排，
        # 省掉独立的 pl.move —— 后者跑在 vector 计算流水上，正是
        # 与 ASC 差距最大的那一项(+127us)。
        # block_stride=65 带 +1 把相邻分形列错开到不同 bank。
        vf.store_align(
            nzp_ub,
            vregCastRes,
            pregFullExeB16,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=NZP_BLOCK_STRIDE,
            repeat_stride=NZP_REPEAT_STRIDE,
            post_update=True,
        )


# ==================================================================
#  Pre 阶段 —— dq workspace 清零 (presfmg_regbase.h:258 InitOutput)
# ==================================================================
# dQ 用 atomicAdd across s2Outer 累加，故进主循环前必须清零。
# ASC 只在 s2Outer>1 时清(同一处判断)，因为 s2Outer==1 时每个位置只写一次。
# dK/dV 走 L0C 驻留累加、每个 (b,n2,s2) 组只 store 一次，但仍用 atomicAdd
# (多核可能分到同一 s2 块的不同 s1 区间)，故同样需要清零。


@pl.vector_function
def fill_zero_vf(dst_ub, nvecs: pl.DT_INT64):
    """把 UB tile 的前 nvecs 个 vreg(每个 VF_LANES 个 fp32)填 0。

    按「平铺的 vreg 个数」而不是「行数」计数，故与行宽解耦 —— D 与 Dv 不等长
    时行宽会随档位变化(128 或 192)，按行写死 2 条 store 的老写法在 192 下
    只会清到 2/3。

    注意要按 tile 的**分配宽度 d_align** 平铺遍历整块，而不是按运行期的
    有效宽度：valid_shape 收窄只影响 load/store 搬多少，tile 自身的行距
    仍是声明时的 d_align，行与行之间是**跨距排布而非紧凑**。多清的那些
    padding 列由 store 的 valid_shape 挡掉，无副作用。
    调用方传 rows * d_align // VF_LANES，两档都整除(128*128/64=256,
    128*192/64=384)。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vregZero = vf.full(0.0, pregFullExe, dtype=pl.DT_FP32)
    for i in pl.range(0, nvecs):
        vf.store_align(dst_ub + i * VF_LANES, vregZero, pregFullExe)


def process_pre(tensor_ws, zero_ub, totalRows, width, vBlockIdx, vecCoreNum):
    """把一个 fp32 workspace 清零。

    workspace 形状 [B,S,N,D] 已展平成 [totalRows, width]，其中 width 是该
    张量真实的 head dim —— dq/dk 是 D，dv 是 Dv，两者可以不等长。
    按 chunk 跨核分配：核 i 领第 i, i+vecCoreNum, ... 块。chunk 取 128 行
    而非吃满 UB —— 实测吃满(496 行)会把 8192 行切成仅 17 块、55 个核
    空跑，反而慢 9us。零值在 UB 里只生成一次，之后反复 store，故单缓冲
    足够，无需 depth 2。
    """
    numChunks = (totalRows + PRE_CHUNK_ROWS - 1) // PRE_CHUNK_ROWS
    zt = zero_ub.current()
    # 清零按 tile 的分配宽度 d_align 平铺整块(见 fill_zero_vf 的说明)，
    # store 时才用 valid_shape 裁到真实宽度 width。
    pl.set_validshape(zt, [PRE_CHUNK_ROWS, width])
    fill_zero_vf(zt, PRE_CHUNK_ROWS * d_align // VF_LANES)
    c = vBlockIdx
    for _ in pl.range(0, numChunks):
        if c >= numChunks:
            break
        r = c * PRE_CHUNK_ROWS
        n = pl.min(PRE_CHUNK_ROWS, totalRows - r)
        pl.set_validshape(zt, [n, width])
        pl.store(tensor_ws, zt, [r, 0])
        pl.set_validshape(zt, [PRE_CHUNK_ROWS, width])
        c = c + vecCoreNum


# ==================================================================
#  Post 阶段 —— dQ/dK/dV 的 scale + cast (post_regbase.h:238-241)
# ==================================================================
# scale 必须在矩阵乘之后、fp32 结果上施加：先乘会让 fp16 舍入发生在
# 不同量级上，训练中误差会累积(用户明确要求)。
# 只有 dQ/dK 乘 scale，dV 不乘 —— ASC 的 `qkvIdx < 2` 判断。


@pl.vector_function
def muls_cast_vf(dst_ub, src_ub, scale: pl.DT_FP32, nsegs: pl.DT_INT64):
    """fp32 -> muls(scale) -> fp16。

    Even/Odd 双路 cast 再 Or 合并，与主循环里的 cast 路径一致。
    dV 不需要 scale，调用方传 1.0（fp32 乘 1.0 精确，不改变数值），
    这样避免在 VF 内做标量分支。

    按「128 元素段数」而不是行数计数：一次 DINTLV_B32 恰好覆盖 128 个 fp32，
    与行宽解耦 —— D 与 Dv 不等长时行宽会是 128 或 192，按行写死一次 load
    的老写法在 192 下只会算 2/3。
    源与目的的行距都是各自 tile 的分配宽度 d_align(相同)，故同一个段号在
    两边都对得上；调用方按 rows * d_align // VECTOR_BASEN 传段数，
    多算的 padding 列由 store 的 valid_shape 挡掉。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for m in pl.range(0, nsegs):
        # DINTLV_B32 一次取到 even/odd 两路，各 64 个 fp32，合起来正好
        # 覆盖 128 个连续元素。
        vregEven, vregOdd = vf.load_align(
            src_ub, m * VECTOR_BASEN, dist=pl.LoadDist.DINTLV_B32
        )
        vregEven = vf.muls(vregEven, scale, pregFullExe)
        vregOdd = vf.muls(vregOdd, scale, pregFullExe)
        vregCastEven = vf.astype(
            vregEven, pregB16, dtype=pl.DT_FP16, layout=pl.CastLayout.ZERO
        )
        vregCastOdd = vf.astype(
            vregOdd, pregB16, dtype=pl.DT_FP16, layout=pl.CastLayout.ONE
        )
        vregRes = vf.or_(vregCastEven, vregCastOdd, pregB16)
        # Or 合并后是完整的 128 个 fp16，一次 store 写完
        vf.store_align(dst_ub + m * VECTOR_BASEN, vregRes, pregB16)


def process_post(
    tensor_ws,
    tensor_out,
    f32_ub,
    f16_ub,
    totalRows,
    width,
    scale,
    vBlockIdx,
    vecCoreNum,
):
    """fp32 workspace -> muls(scale) -> cast fp16 -> 输出。

    width 是该张量真实的 head dim：dq/dk 是 D，dv 是 Dv(可以短于 D)。

    按 chunk 跨核分配：核 i 领第 i, i+vecCoreNum, ... 块，chunk 取满
    247.5KB UB(165 行)。块数少于核数时尾部核空跑 —— 这是「UB 用满优先」
    的取舍。

    depth 2 靠每轮无条件 next() 轮转两个槽位实现：相邻两轮落在不同 UB
    地址、各自独立 mutex_id，框架据此判无冲突，于是第 i+1 轮的 load(MTE2)
    可与第 i 轮的 cast(V)、store(MTE3) 重叠。

    不要手工做「先发下一块 load」的软流水：预取是条件执行的，next() 的
    调用次数随分支变化，current() 指向的槽位就会错位 —— 实测多块的核
    在最后一轮拿到上一块的旧数据(B2N2/S1=384 用例 dQ 错到 1.3/4.2)。
    无条件轮转把游标记账变成确定的，重叠交给框架，与 L0A/L0B 双缓冲同理。
    """
    numChunks = (totalRows + POST_CHUNK_ROWS - 1) // POST_CHUNK_ROWS
    c = vBlockIdx
    for _ in pl.range(0, numChunks):
        if c >= numChunks:
            break
        r = c * POST_CHUNK_ROWS
        n = pl.min(POST_CHUNK_ROWS, totalRows - r)
        ft = f32_ub.next()
        ht = f16_ub.next()
        # tile 按 d_align 声明，这里裁到该张量真实的宽度 width
        # (dq/dk 是 D，dv 是 Dv)。VF 按平铺段数走，与行宽解耦。
        pl.set_validshape(ft, [n, width])
        pl.set_validshape(ht, [n, width])
        pl.load(ft, tensor_ws, [r, 0])
        # 段数按分配宽度 d_align 算(两个 tile 行距相同)，见 muls_cast_vf
        muls_cast_vf(ht, ft, scale, n * d_align // VECTOR_BASEN)
        pl.store(tensor_out, ht, [r, 0])
        pl.set_validshape(ft, [POST_CHUNK_ROWS, width])
        pl.set_validshape(ht, [POST_CHUNK_ROWS, width])
        c = c + vecCoreNum


def process_vec3(constInfo, runInfo, subIdx, ds_l1, mm1_ub, mm2_ub, sfmg_ub, nzp_ub):
    rowOff = subIdx * runInfo.firstHalfS1RealSize
    rows = runInfo.halfS1RealSize
    # nd/nz 不裁剪：列方向的脏数据由 matmul 的 valid_shape 与 store 挡掉。
    # VF 固定写满 VECTOR_BASEM 行：NZ+1 的分形列间距由行数决定，按尾块
    # 缩小会让间距漂移。尾块只在 insert 时用 valid_shape 裁掉。
    # ASC 同样传常量 VECTOR_BASEM(block_vec.h:624)。
    broadcast_sub_mul_cast_vf(
        nzp_ub, mm1_ub, mm2_ub, sfmg_ub, constInfo.scaleValue, VECTOR_BASEM
    )
    # 只写一次 L1：mm4 通过 ds_t_l1(同地址的 ZN 视图)直接读到 dS^T，
    # 无需再做 transpose/insert
    copy_ub2l1(ds_l1, nzp_ub, rowOff)


# ==================================================================
#  ProcessVec4 —— v4: P cast + ND2NZ 落 L1 (block_vec.h::ProcessVec4)
#  同时产出 P^T L1，供 MM5 使用
# ==================================================================
@pl.vector_function
def cast_p_vf(nzp_ub, mm2_ub, srcM: pl.DT_INT64):
    """P: fp32 -> fp16，对应 AscendC ProcessVec4 中的 CastTransdataDeconflict。"""
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for m in pl.range(0, srcM):
        vregPEven, vregPOdd = vf.load_align(
            mm2_ub, m * VECTOR_BASEN, dist=pl.LoadDist.DINTLV_B32
        )
        vregCastEven = vf.astype(
            vregPEven, pregFullExeB16, dtype=pl.DT_FP16, layout=pl.CastLayout.ZERO
        )
        vregCastOdd = vf.astype(
            vregPOdd, pregFullExeB16, dtype=pl.DT_FP16, layout=pl.CastLayout.ONE
        )
        vregCastRes = vf.or_(vregCastEven, vregCastOdd, pregFullExeB16)
        # DataBlock 拷贝模式(vsstb)：一条 store 同时完成 ND->NZ 重排，
        # 省掉独立的 pl.move —— 后者跑在 vector 计算流水上，正是
        # 与 ASC 差距最大的那一项(+127us)。
        # block_stride=65 带 +1 把相邻分形列错开到不同 bank。
        vf.store_align(
            nzp_ub,
            vregCastRes,
            pregFullExeB16,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=NZP_BLOCK_STRIDE,
            repeat_stride=NZP_REPEAT_STRIDE,
            post_update=True,
        )


def process_vec4(constInfo, runInfo, subIdx, p_l1, mm2_ub, nzp_ub):
    rowOff = subIdx * runInfo.firstHalfS1RealSize
    rows = runInfo.halfS1RealSize
    cast_p_vf(nzp_ub, mm2_ub, VECTOR_BASEM)
    # 同 process_vec3：mm5 从 p_t_l1(同地址 ZN 视图)读 P^T
    copy_ub2l1(p_l1, nzp_ub, rowOff)


def flush_dkv(
    constInfo, runInfo, tensor_dk_ws, tensor_dv_ws, dk_acc0, dk_acc1, dv_acc0, dv_acc1
):
    """把 L0C 上累加完的 dK/dV 写回 GM。

    同一 s2 块内跨 s1 的部分和已在 L0C 累加完毕，故这里每个 (b,n2,s2) 组
    只写一次。仍用 atomicAdd：多核可能分到同一 s2 块的不同 s1 区间。

    每个 D 分块各有一块常驻 L0C，故要逐块写回自己的列区间(store 的第 4 个
    索引即列偏移)。dV 的宽度按 Dv 截断，整块无效时跳过。
    current() 而非 next()：本轮的 matmul 已经把游标推进到位。
    """
    i0 = runInfo.idx0KV
    i1 = runInfo.keyOffset
    i2 = runInfo.idx2KV
    # ---- dK：按 D 分块，每块写自己的列区间 ----
    pl.store(
        tensor_dk_ws,
        dk_acc0.current(),
        [i0, i1, i2, D_CHUNK_OFF0[d_align]],
        order=[1, 3],
        atomic=pl.AtomicType.AtomicAdd,
    )
    if D_NCHUNKS[d_align] > 1:
        pl.store(
            tensor_dk_ws,
            dk_acc1.current(),
            [i0, i1, i2, D_CHUNK_OFF1[d_align]],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicAdd,
        )
    # ---- dV：同样按 D 的分块表切，但宽度按 Dv 截断，整块无效则跳过 ----
    if dv_chunk_w(constInfo, 0) > 0:
        pl.store(
            tensor_dv_ws,
            dv_acc0.current(),
            [i0, i1, i2, D_CHUNK_OFF0[d_align]],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicAdd,
        )
    if D_NCHUNKS[d_align] > 1:
        if dv_chunk_w(constInfo, 1) > 0:
            pl.store(
                tensor_dv_ws,
                dv_acc1.current(),
                [i0, i1, i2, D_CHUNK_OFF1[d_align]],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicAdd,
            )


@pl.jit(
    auto_mutex=True,
    tiling_key=FlashAttnGradTilingKey,
    datatype={
        "q": "input_dtype",
    },
)
def flash_attn_grad(
    # 13 个输入，顺序严格对齐 op_host/flash_attn_grad_def.cpp 的 Input() 声明
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    v: pl.Ptr[pl.DT_UINT8],
    dout: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    # 以下 6 个为可选输入。本实现只覆盖 BSND 非 sparse 无 mask 路径，
    # 故 cu_seqlens/seqused(TND 变长)、sinks、attn_mask 均不读取；
    # metadata 预留给 FAG 元数据(head[4] + 116 int32)，当前未用。
    # 保留形参是为了与 host 的 13 输入按位对齐 —— 少一个都会错位。
    cu_seqlens_q: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_kv: pl.Ptr[pl.DT_UINT8],
    seqused_q: pl.Ptr[pl.DT_UINT8],
    seqused_kv: pl.Ptr[pl.DT_UINT8],
    sinks: pl.Ptr[pl.DT_UINT8],
    attn_mask: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    # 3 个输出
    dq: pl.Ptr[pl.DT_UINT8],
    dk: pl.Ptr[pl.DT_UINT8],
    dv: pl.Ptr[pl.DT_UINT8],
    # 单块 workspace，内部再切三份 fp32 累加区（见下方 make_ptr 分区）
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FlashAttnGradTilingData,
):
    # coreNum 从内建 API 取而非 tiling：新接口的 TilingData 无此字段。
    # 已实测 cube/vector 两侧 get_block_num() 都返回物理核数(36)，
    # 故两侧算出的 numBlocks 一致，set/wait 可配对。
    coreNum = pl.get_block_num()
    # swizzle 来自 tilingkey，是编译期常量 —— 每种分核各生成一份二进制，
    # sel() 的两路分支在编译期被折叠掉，运行期没有判断开销。
    constInfo = set_const_info(tiling, coreNum, swizzle)

    # ---- workspace 分区 ----
    # host(GetWorkspaceSizes) 给的是单块连续内存，这里按 fp32 元素切成
    # dq/dk/dv 三段累加区。三段大小必须与 host 的 CalcWorkSpace 逐一对齐。
    # dq 段按 n1(=n2*G) 个 head，dk/dv 段按 n2 —— 与 host 的 dqElems/dkvElems
    # 一致。GQA 下若这里误用 n2，dk/dv 的起始偏移就会算错。
    wsRowsQ = tiling.b * tiling.s1 * tiling.n1
    wsRowsKV = tiling.b * tiling.s2 * tiling.n2
    ws_ptr = pl.make_ptr(workspace, dtype=pl.DT_FP32)
    dq_workspace = ws_ptr
    # 段宽按各自的 head dim：dq/dk 是 D，dv 是 Dv(可以短于 D)。
    # 必须与 host 的 dqElems/dkElems/dvElems 逐一对齐。
    dk_workspace = pl.addptr(dq_workspace, wsRowsQ * tiling.d)
    dv_workspace = pl.addptr(dk_workspace, wsRowsKV * tiling.d)

    # ---- 统一四维视图 [viewD0, S, viewD2, D] ----
    # BSND 与 BNSD 共用同一套 make_tensor / pl.load(order=...) 代码，
    # 布局差异只体现在 viewD0/viewD2 与索引系数上(见 set_run_info 的 idx0/idx2)：
    #   BSND: [B,   S, N, D]，idx = [b,     s, n, 0]
    #   BNSD: [B*N, S, 1, D]，idx = [b*N+n, s, 0, 0]
    # 两者都是紧凑排布 —— 这一点是必须的：make_tensor 对非紧凑 stride 会静默
    # 忽略并按紧凑跨距去读(不报错)，所以不能用「倒置 stride」表达 BNSD。
    # Q 侧(n1 个 head)与 KV 侧(n2 个 head)各一套视图参数
    # 最内轴宽度取各张量**真实的** head dim，不能用分配宽度 d_align：
    # 这是 GM 上的实际排布，Q/K/dq/dk 侧是 D，V/dO/attn_out/dv 侧是 Dv。
    # (片上 tile 才按 d_align 分配、按真实宽度填。)
    dQ = constInfo.dSize
    dV = constInfo.dvSize
    viewD2Q = constInfo.viewD2Q
    d2dQ = viewD2Q * dQ
    s1ndQ = constInfo.s1Size * d2dQ
    viewD2KV = constInfo.viewD2KV
    d2dKV = viewD2KV * dQ
    s2ndKV = constInfo.s2Size * d2dKV
    # Dv 宽的那几个张量另有一套行距
    d2dQv = viewD2Q * dV
    s1ndQv = constInfo.s1Size * d2dQv
    d2dKVv = viewD2KV * dV
    s2ndKVv = constInfo.s2Size * d2dKVv

    tensor_q = pl.make_tensor(
        q,
        [constInfo.viewD0Q, constInfo.s1Size, viewD2Q, dQ],
        [s1ndQ, d2dQ, dQ, 1],
        dtype=pl.DT_FP16,
    )
    tensor_k = pl.make_tensor(
        k,
        [constInfo.viewD0KV, constInfo.s2Size, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=pl.DT_FP16,
    )
    # MM3(dQ = dS@K) 需要「未转置」的 K，而 MM2 需要 K^T。
    # 每个 GM tensor 只会推导出一种 Layout：tensor_k 因 MM2 的转置读被定成
    # Layout::DN，MM3 若共用它，plain 读也会被按 DN 解释，实测等价于拿到 K^T
    # (partial 与 K[s2o]^T 吻合到 1e-4)。故为 MM3 单独建一份同 buffer 的
    # tensor 视图，让它独立推导 layout。
    tensor_k_nt = pl.make_tensor(
        k,
        [constInfo.viewD0KV, constInfo.s2Size, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=pl.DT_FP16,
    )
    # V/dO/attn_out 是 Dv 宽
    tensor_v = pl.make_tensor(
        v,
        [constInfo.viewD0KV, constInfo.s2Size, viewD2KV, dV],
        [s2ndKVv, d2dKVv, dV, 1],
        dtype=pl.DT_FP16,
    )
    tensor_dy = pl.make_tensor(
        dout,
        [constInfo.viewD0Q, constInfo.s1Size, viewD2Q, dV],
        [s1ndQv, d2dQv, dV, 1],
        dtype=pl.DT_FP16,
    )
    tensor_y = pl.make_tensor(
        attn_out,
        [constInfo.viewD0Q, constInfo.s1Size, viewD2Q, dV],
        [s1ndQv, d2dQv, dV, 1],
        dtype=pl.DT_FP16,
    )
    # softmax_lse : [B,N1,S1] —— 按 Q 的 head 数(n1 = n2*G)存，因为每个 Q head
    # 的 softmax 归一化因子不同，G 个 Q head 不能共享同一份 lse。
    # 对齐 ASC 的 softmax_max/sum：offset = ((b*n2 + n2o)*G + g)*S1。
    n1Size = constInfo.n2Size * constInfo.gSize
    tensor_lse = pl.make_tensor(
        softmax_lse,
        [constInfo.bSize, n1Size, constInfo.s1Size],
        [n1Size * constInfo.s1Size, constInfo.s1Size, 1],
        dtype=pl.DT_FP32,
    )
    # dq/dk/dv workspace 为 fp32，供 atomicAdd 累加
    # dq/dk 是 D 宽，dv 是 Dv 宽 —— 与 workspace 的分段宽度一致
    tensor_dq_ws = pl.make_tensor(
        dq_workspace,
        [constInfo.viewD0Q, constInfo.s1Size, viewD2Q, dQ],
        [s1ndQ, d2dQ, dQ, 1],
        dtype=pl.DT_FP32,
    )
    tensor_dk_ws = pl.make_tensor(
        dk_workspace,
        [constInfo.viewD0KV, constInfo.s2Size, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=pl.DT_FP32,
    )
    tensor_dv_ws = pl.make_tensor(
        dv_workspace,
        [constInfo.viewD0KV, constInfo.s2Size, viewD2KV, dV],
        [s2ndKVv, d2dKVv, dV, 1],
        dtype=pl.DT_FP32,
    )

    # ---- pre/post 用的展平视图 ----
    # workspace 逻辑形状 [B,S,N,D]，pre/post 只需按行线性遍历，
    # 故另建 [B*S*N, D] 的二维视图(同一 buffer，独立推导 layout)。
    # dq 侧的 head 数是 n1(=n2*G)，dk/dv 侧是 n2 —— GQA 下两者不同。
    dqRows = constInfo.bSize * constInfo.s1Size * n1Size
    dkvRows = constInfo.bSize * constInfo.s2Size * constInfo.n2Size
    # 行宽同样按各自的 head dim：dq/dk 用 D，dv 用 Dv。
    tensor_dq_ws_flat = pl.make_tensor(
        dq_workspace, [dqRows, dQ], [dQ, 1], dtype=pl.DT_FP32
    )
    tensor_dk_ws_flat = pl.make_tensor(
        dk_workspace, [dkvRows, dQ], [dQ, 1], dtype=pl.DT_FP32
    )
    tensor_dv_ws_flat = pl.make_tensor(
        dv_workspace, [dkvRows, dV], [dV, 1], dtype=pl.DT_FP32
    )
    tensor_dq_out = pl.make_tensor(dq, [dqRows, dQ], [dQ, 1], dtype=pl.DT_FP16)
    tensor_dk_out = pl.make_tensor(dk, [dkvRows, dQ], [dQ, 1], dtype=pl.DT_FP16)
    tensor_dv_out = pl.make_tensor(dv, [dkvRows, dV], [dV, 1], dtype=pl.DT_FP16)

    # cBlockIdx 必须按物理核取：AIV 侧 get_block_idx() 按 vector 子核计数
    # (0..2*coreNum-1)，AIC 侧按物理核计数。两侧口径不一致会算出不同的
    # numBlocks 不同 -> 循环次数不同 -> set/wait 配不上 -> 死锁。
    # 核数直接用 tiling 传入，避免再依赖 get_block_num 的口径。
    cBlockIdx = pl.get_block_idx() // pl.get_subblock_num()
    coreNum = constInfo.coreNum

    # 本核负责的块数。swizzle 下按「s2 列」跨核发放(核 c 领第
    # c, c+coreNum, ... 列，每列 s1Outer 个块)，故不再是连续区间 —— 循环
    # 改用局部序号 0..numBlocks-1，再经 global_idx() 映射成全局块下标。
    # 两侧(cube/vector)必须算出同一个 numBlocks，否则循环次数不同、
    # set/wait 配不上就死锁。
    numBlocks = core_block_range(constInfo, cBlockIdx, coreNum)

    # ---- CV 共享 UB：mm1ResBuf[2] / mm2ResBuf[2]，按 taskIdMod2 ping-pong ----
    mm1_res = pl.make_tile_group(
        type=pl.TileType(
            shape=[VECTOR_BASEM, VECTOR_BASEN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[UB_MM1_0, UB_MM1_1],
        mutex_ids=[0, 1],
    )
    mm2_res = pl.make_tile_group(
        type=pl.TileType(
            shape=[VECTOR_BASEM, VECTOR_BASEN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[UB_MM2_0, UB_MM2_1],
        mutex_ids=[2, 3],
    )

    # ---- CV 共享 L1：dSL1Buf / pL1Buf 及其转置副本 ----
    # 对应 kernel_base.h 的「CV核间共享Buffer」，vector 侧写、cube 侧读，
    # 故必须声明在两个 section 之外
    ds_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEM, CUBE_BASEN],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=[L1_DS_ADDR[d_align]],
        mutex_ids=[8],
    )
    p_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEM, CUBE_BASEN],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=[L1_P_ADDR[d_align]],
        mutex_ids=[9],
    )
    # dS^T / P^T 不再单独存一份：[M,N] 按 NZ 存放的字节布局，与 [N,M] 按 ZN
    # 存放完全一致，所以对 ds_l1/p_l1 的同一地址再建一个 ZN 视图，就等于
    # 零指令拿到转置。vector 侧只 insert 一次(NZ 视图)，cube 侧 mm4/mm5
    # 直接按 ZN 视图 move 到 L0A —— shape 与 L0A 一致，TMov 通过。
    #
    # 这替掉了原先的 pl.transpose：实测单次 24.9us(44864 cycles)，
    # 整核估算 5672us，占 8192 场景实测 6011us 的 94%，是性能差距主因。
    # 注意 L1->L0 无法随路转置(TMov static_assert 要求 shape 相同)，
    # Python validator 里「允许 2D 转置」的注释能过 parse 但 codegen 失败。
    ds_t_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEN, CUBE_BASEM],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=[L1_DS_ADDR[d_align]],
        mutex_ids=[10],
    )
    p_t_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEN, CUBE_BASEM],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=[L1_P_ADDR[d_align]],
        mutex_ids=[11],
    )

    # ==== Pre 阶段：清零 dq/dk/dv workspace ====
    # 三个 workspace 都用 atomicAdd 累加，进主循环前必须为 0。
    # 对应 ASC presfmg_regbase.h:258 的 InitOutput / Duplicate。
    # 清完用 sync_all 隔开：主循环的 atomicAdd 必须看到已清零的 GM。
    with pl.section_vector():
        vBlockIdx = pl.get_block_idx()
        vecCoreNum = coreNum * pl.get_subblock_num()
        # 单缓冲吃满 248KB(496 行)。mutex_ids 复用主循环的 id：
        # pre 与主循环被 sync_all 完全隔开，不会同时活跃。
        zero_ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[PRE_CHUNK_ROWS, d_align],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
            ),
            addrs=[UB_PRE_ZERO],
            mutex_ids=[19],
        )
        # dq/dk 段是 D 宽，dv 段是 Dv 宽
        process_pre(tensor_dq_ws_flat, zero_ub, dqRows, dQ, vBlockIdx, vecCoreNum)
        process_pre(tensor_dk_ws_flat, zero_ub, dkvRows, dQ, vBlockIdx, vecCoreNum)
        process_pre(tensor_dv_ws_flat, zero_ub, dkvRows, dV, vBlockIdx, vecCoreNum)
    pl.system.sync_all()

    with pl.section_cube():
        # 地址/mutex 表按档下标取(表在模块级算好，见 _build_l1)。
        # 128 档取到的就是原有那套写死的值，故该档生成代码不变。
        q_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[CUBE_BASEM, d_align],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                compact=1,
            ),
            addrs=L1_Q_ADDRS[d_align],
            mutex_ids=L1_Q_MUTEX[d_align],
        )
        k_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[d_align, CUBE_BASEN],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
                compact=1,
            ),
            addrs=[L1_K_ADDR[d_align]],
            mutex_ids=[5],
        )
        # MM3 的右矩阵：非转置 K[S2,D]，默认(ND) layout，独占地址与 mutex。
        # 源必须是 ND layout 的 tensor_k_nt(ND2ND)；若源是 DN 的 tensor_k，
        # 载入会按 DN 解释而静默得到 K^T。ZN 版本虽能编过(DN2ZN 合法)但
        # 同样不对：平台只支持 ND2NZ/DN2NZ/ND2ND/DN2DN/NZ2NZ/DN2ZN。
        k_l1_n = pl.make_tile_group(
            type=pl.TileType(
                shape=[CUBE_BASEN, d_align],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                compact=1,
            ),
            addrs=[L1_KND_ADDR[d_align]],
            mutex_ids=[30],
        )
        # V/dO 是 Dv 宽，但 tile 仍按 d_align 分配(Dv<=D)：多出来的列是
        # padding，由 load/matmul 的 valid_shape 挡掉。这样两侧共用一套
        # 地址步长，L1 布局不必再分 D/Dv 两种块大小。
        v_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[d_align, CUBE_BASEN],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
                compact=1,
            ),
            addrs=[L1_V_ADDR[d_align]],
            mutex_ids=[6],
        )
        do_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[CUBE_BASEM, d_align],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                compact=1,
            ),
            addrs=L1_DO_ADDRS[d_align],
            mutex_ids=L1_DO_MUTEX[d_align],
        )
        # mm4/mm5 用的 Q/dO 视图：地址与 mutex_id 和上面完全相同(共享同一
        # 块 L1 与同一套同步跟踪)，但游标独立。它从 taskId==2 才开始
        # next()，因此永远落后两个 task —— current() 恰好指向 task N-2
        # 载入的槽位，正是 mm4/mm5 需要的那一份。
        # 这样 Q/dO 每个 (b,n,s1) 块只从 GM 读一次，mm4/mm5 直接复用。
        # mutex_ids 必须与 writer 视图完全相同：这对应 ASC 的
        # qL1Buf.Get() / qL1Buf.GetPre() —— 同一个 MutexBuffer 的两个访问器，
        # 靠同一套 LockProd/UnlockProd 与 LockCons/UnlockCons 把「mm2 写」
        # 与「mm4 读」排序。若给 reader 另一套 id，这层互锁就没了，
        # mm2(task N)会在 mm4 读完前覆写，实测 dK/dV 错(0.61/0.57)。
        q_l1_dk = pl.make_tile_group(
            type=pl.TileType(
                shape=[CUBE_BASEM, d_align],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                compact=1,
            ),
            addrs=L1_Q_ADDRS[d_align],
            mutex_ids=L1_Q_MUTEX[d_align],
        )
        do_l1_dv = pl.make_tile_group(
            type=pl.TileType(
                shape=[CUBE_BASEM, d_align],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                compact=1,
            ),
            addrs=L1_DO_ADDRS[d_align],
            mutex_ids=L1_DO_MUTEX[d_align],
        )
        # ---- L0A/L0B/L0C ----
        # 一律按「最宽的一个 D 分块」声明：192 档是 96 列而不是 192，
        # 于是 L0A/L0B 每块 24KB、双缓冲 48KB<=64KB 得以保留。
        # 128 档分块宽度就是 128，与改动前完全一致。
        # 左矩阵最宽的一维是 ACC_COLS = max(CUBE_BASEN, 分块宽)：MM2/MM1 的
        # 左矩阵是 Q/dO 的 D 分块，MM3/MM4/MM5 的左矩阵是 dS/P([M,N])。
        left = pl.make_tile_group(
            type=pl.TileType(
                shape=[CUBE_BASEM, ACC_COLS[d_align]],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Left,
                compact=1,
            ),
            addrs=L0AB_ADDRS[d_align],
            mutex_ids=[12, 26],
        )
        right = pl.make_tile_group(
            type=pl.TileType(
                shape=[ACC_COLS[d_align], CUBE_BASEN],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Right,
                compact=1,
            ),
            addrs=L0AB_ADDRS[d_align],
            mutex_ids=[13, 27],
        )
        # acc 的列宽取 ACC_COLS：MM2/MM1 输出 [M,s2]，MM3 输出 dQ 的一个
        # D 分块。
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[CUBE_BASEM, ACC_COLS[d_align]],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=L0C_ACC_ADDRS[d_align],
            mutex_ids=L0C_ACC_MUTEX[d_align],
        )
        # L0C 驻留的 dK/dV 累加器，跨 s1 轮次保持不变。
        # 每个 D 分块**各建一个独立的 tile_group**(而不是一个多槽位 group)：
        # 这些累加器必须在同一 s2 块内持续存在，访问方式是 current() 而非
        # next()。若做成多槽位 group 再靠 next() 轮换，D=128 档(单块)的
        # 游标语义就和改动前不同了，零回归就不成立。
        dk_acc0 = pl.make_tile_group(
            type=pl.TileType(
                shape=[CUBE_BASEN, D_CHUNK_MAXW[d_align]],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=[L0C_DK0_ADDR[d_align]],
            mutex_ids=[16],
        )
        dv_acc0 = pl.make_tile_group(
            type=pl.TileType(
                shape=[CUBE_BASEN, D_CHUNK_MAXW[d_align]],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=[L0C_DV0_ADDR[d_align]],
            mutex_ids=[17],
        )
        # 第二块只在 192 档存在。为了让下面的调用点无需分支，单块档位下把
        # 它指向第 0 块的同一地址/同一 mutex —— 那些代码路径被 trace 期的
        # `if D_NCHUNKS > 1` 完全剪掉，永远不会真的用到这个别名。
        dk_acc1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[CUBE_BASEN, D_CHUNK_MAXW[d_align]],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=[L0C_DK1_ADDR[d_align]],
            mutex_ids=[L0C_DK1_MUTEX[d_align]],
        )
        dv_acc1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[CUBE_BASEN, D_CHUNK_MAXW[d_align]],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=[L0C_DV1_ADDR[d_align]],
            mutex_ids=[L0C_DV1_MUTEX[d_align]],
        )

        runInfos = make_run_infos()
        # ---------------- Process() 主循环 (AIC 侧) ----------------
        taskId = 0
        localIdx = 0
        # cube 领先 vector 两个 task，故循环要多跑两轮把流水排空：
        # 前 numBlocks 轮发 mm1/mm2，后两轮只做 mm3/mm4/mm5。
        for _ in pl.range(0, numBlocks + 2):
            isLastLoop = localIdx >= numBlocks
            # 无条件轮转，保证与 vector 侧步调一致（末轮不做 mm 也要转）
            mm2_cur = mm2_res.next()
            mm1_cur = mm1_res.next()
            # L1_PRELOAD：Q/dO 的 double buffer 无条件轮转，供本轮 mm2/mm1。
            # mm4/mm5 用 q_l1_dk/do_l1_dv —— 同地址、同 mutex_ids 的延后
            # 游标视图，直接复用这里载入的数据(见其声明处说明)。
            q_cur = q_l1.next()
            do_cur = do_l1.next()
            if not isLastLoop:
                curInfo = runInfos[taskId % PRELOAD_TIMES]
                gIdx = global_idx(constInfo, cBlockIdx, coreNum, localIdx)
                set_run_info(constInfo, curInfo, taskId, gIdx, 0)

                # K/V 只在 s2 组切换时重载(S1 为最快轴，同组内 K/V 不变)
                loadKV = need_load_kv(constInfo, gIdx, localIdx)

                # ---- mm2: Q@K^T ----
                # 反向等待：mm2 只有 2 个 UB 槽位，task N 复用 task N-2 的
                # 那块，必须等 vector 读完(ASC ProcessPreloadTwoTimes 里
                # taskId>=1 时 wait SYNC_V2_TO_C2_FLAG)。
                # taskId>=2 才有 N-2 存在；首两轮槽位是干净的，直接发。
                # 这就是「首轮连发四次 matmul」：taskId 0/1 两轮各发
                # mm2+mm1，累计 4 次，把两个槽位填满后 vector 才开始消费，
                # 于是第 N 组 cube 计算与第 N-2 组 vector 计算重叠。
                if taskId > 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX, event_id=SYNC_V2_TO_C2_FLAG[taskId % 2]
                    )
                iterate_mm_qk(
                    constInfo,
                    curInfo,
                    tensor_q,
                    tensor_k,
                    q_cur,
                    k_l1,
                    left,
                    right,
                    acc,
                    mm2_cur,
                    loadKV,
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.FIX, event_id=SYNC_C2_TO_V2_FLAG[taskId % 2]
                )
                # ---- mm1: dO@V^T ----
                if taskId > 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX, event_id=SYNC_V2_TO_C1_FLAG[taskId % 2]
                    )
                iterate_mm_dyv(
                    constInfo,
                    curInfo,
                    tensor_dy,
                    tensor_v,
                    do_cur,
                    v_l1,
                    left,
                    right,
                    acc,
                    mm1_cur,
                    loadKV,
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.FIX, event_id=SYNC_C1_TO_V2_FLAG[taskId % 2]
                )

            # mm3/mm4/mm5 消费的是 vector 刚产出的 dS/P，对应 task N-2
            if taskId > 1:
                prevInfo = runInfos[(taskId + 1) % PRELOAD_TIMES]
                # 正向：等 vector 把 P / dS 写入 L1
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE1, event_id=SYNC_V4_TO_C5_FLAG
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE1, event_id=SYNC_V3_TO_C3_FLAG
                )
                # ---- mm5: dV = P^T@dO (L0C 驻留累加) ----
                # isS2IdxNoChange: 与上一轮同一个 s2 块 -> 在 L0C 上累加
                # prevInfo 对应 task N-2 的 block 下标。
                # task N 处理局部序号 N，故 task N-2 是 taskId-2。
                # 不能用 localIdx-2：排空阶段 localIdx 已停在 numBlocks。
                prevIdx = taskId - 2
                prevG = global_idx(constInfo, cBlockIdx, coreNum, prevIdx)
                dkvAcc = dkv_acc_flag(constInfo, prevG, prevIdx)
                # 延后游标在此推进：taskId>1 才走到这里，故它恒落后两拍
                q_prev = q_l1_dk.next()
                do_prev = do_l1_dv.next()
                iterate_mm_pdy(
                    constInfo,
                    prevInfo,
                    p_t_l1.current(),
                    do_prev,
                    left,
                    right,
                    dv_acc0,
                    dv_acc1,
                    dkvAcc,
                )
                # 反向：P 的 L1 已读完，放行 vector 覆写
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE1, event_id=SYNC_C5_TO_V4_FLAG
                )
                # ---- mm3: dQ = dS@K (仍直出 GM，dQ 按 s1 分块无法驻留) ----
                # mm3 对应 task N-2，其 K_ND 是否需重载按 prevIdx 判断
                loadKVPrev = need_load_kv(constInfo, prevG, prevIdx)
                iterate_mm_dsk(
                    constInfo,
                    prevInfo,
                    tensor_k_nt,
                    tensor_dq_ws,
                    loadKVPrev,
                    ds_l1.current(),
                    k_l1_n,
                    left,
                    right,
                    acc,
                )

                # ---- mm4: dK = dS^T@Q (L0C 驻留累加) ----
                iterate_mm_dsq(
                    constInfo,
                    prevInfo,
                    ds_t_l1.current(),
                    q_prev,
                    left,
                    right,
                    dk_acc0,
                    dk_acc1,
                    dkvAcc,
                )

                # s2 块即将切换(或已是最后一轮)时，把 L0C 上的 dK/dV 落 GM
                # 注意：这里判断的是「上一轮(prevInfo)所属的 s2 块是否即将
                # 结束」：比较 prevIdx 与其后继是否同组；
                # 末轮无后继，必须无条件落盘。
                if prevIdx >= numBlocks - 1 or not next_s2_same(
                    constInfo, prevG, prevIdx
                ):
                    flush_dkv(
                        constInfo,
                        prevInfo,
                        tensor_dk_ws,
                        tensor_dv_ws,
                        dk_acc0,
                        dk_acc1,
                        dv_acc0,
                        dv_acc1,
                    )
                # 反向：dS 的 L1 已读完
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE1, event_id=SYNC_C4_TO_V3_FLAG
                )

            # 处理完最后一个 block(task numBlocks-1，即 taskId==numBlocks+1)
            # 才退出；isLastLoop 只用来停发 mm1/mm2
            if taskId >= numBlocks + 1:
                break
            taskId = taskId + 1
            localIdx = pl.min(localIdx + 1, numBlocks)

    with pl.section_vector():
        subIdx = pl.get_subblock_idx()
        # y/dy/prod/tmp 的宽度是 **Dv**(y/dy 就是 Dv 宽)，故按 dv_align 声明。
        # pl.sum 沿最后一轴 reduce，宽度错了会把 padding 列也加进 sfmg。
        y_ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[SFMG_LOOP_SIZE, dv_align],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[UB_Y_A[dv_align]],
            mutex_ids=[15],
        )
        dy_ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[SFMG_LOOP_SIZE, dv_align],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[UB_DY_A[dv_align]],
            mutex_ids=[16],
        )
        prod_ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[SFMG_LOOP_SIZE, dv_align],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[UB_PROD_A[dv_align]],
            mutex_ids=[17],
        )
        tmp_ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[SFMG_LOOP_SIZE, dv_align],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=[UB_TMP_A[dv_align]],
            mutex_ids=[18],
        )
        # 融合 VF 的唯一输出缓冲：65 行 NZ，多出的一行是 bank 错开填充。
        # valid_shape 动态设置：insert 前收窄到 rows(框架据此算 65-rows
        # 的源跨距)，insert 后必须恢复成 65(见 copy_ub2l1)。
        nzp_ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[NZP_ROWS, VECTOR_BASEN],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=2,
            ),
            addrs=[UB_NZP_A[dv_align]],
            mutex_ids=[19],
        )
        # sfmg 整体视图([64,1])供 V3 broadcast；lo/hi 两个分段视图供 V1 写入
        sfmg_ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[VECTOR_BASEM, 1],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.DN,
            ),
            addrs=[UB_SFMG_A[dv_align]],
            mutex_ids=[23],
        )
        sfmg_lo_ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[SFMG_LOOP_SIZE, 1],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.DN,
            ),
            addrs=[UB_SFMG_A[dv_align]],
            mutex_ids=[28],
        )
        sfmg_hi_ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[SFMG_LOOP_SIZE, 1],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.DN,
            ),
            addrs=[UB_SFMG_HI_A[dv_align]],
            mutex_ids=[29],
        )
        lse_ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[VECTOR_BASEM, 1],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.DN,
            ),
            addrs=[UB_LSE0_A[dv_align], UB_LSE1_A[dv_align]],
            mutex_ids=[24, 25],
        )

        runInfos = make_run_infos()
        # ---------------- Process() 主循环 (AIV 侧) ----------------
        taskId = 0
        localIdx = 0
        # 与 cube 侧同样多跑两轮排空(见 cube 侧说明)，保证两侧 taskId
        # 逐轮一一对应，set/wait 才能配对
        for _ in pl.range(0, numBlocks + 2):
            isLastLoop = localIdx >= numBlocks
            # 无条件轮转，与 cube 侧一一对应
            mm2_res.next()
            mm1_res.next()
            # vector 落后 cube 两个 task：消费 runInfos[(taskId+1)%3] 即 N-2
            if taskId > 1:
                prevInfo = runInfos[(taskId + 1) % PRELOAD_TIMES]
                # ---- v1: softmaxGrad sfmg = rowsum(dy*y) ----
                process_vec1(
                    constInfo,
                    prevInfo,
                    subIdx,
                    tensor_y,
                    tensor_dy,
                    y_ub.current(),
                    dy_ub.current(),
                    prod_ub.current(),
                    tmp_ub.current(),
                    sfmg_lo_ub.current(),
                    sfmg_hi_ub.current(),
                )

            if not isLastLoop:
                curInfo = runInfos[taskId % PRELOAD_TIMES]
                gIdx = global_idx(constInfo, cBlockIdx, coreNum, localIdx)
                set_run_info(constInfo, curInfo, taskId, gIdx, subIdx)
                # CopyMaxSum：max/sum 双缓冲随 taskId 轮转
                lse_ub.next()

            if taskId > 1:
                prevInfo = runInfos[(taskId + 1) % PRELOAD_TIMES]
                # 消费 cube 在 task N-2 写入的槽位。mm1/mm2 只有 2 槽，
                # task N 与 N-2 同奇偶 -> 同一块物理槽位，故这里要取
                # current()(与 cube 侧 next() 返回的同一块)，
                # 而不是 lag=1 时用的 previous()
                mm1_t = mm1_res.current()
                mm2_t = mm2_res.current()
                # ---- ComputeDqkvBn2gs1s2 ----
                # 正向：等 mm2 结果
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_C2_TO_V2_FLAG[taskId % 2]
                )
                process_vec2(
                    constInfo, prevInfo, subIdx, tensor_lse, mm2_t, lse_ub.previous()
                )
                # 正向：等 mm1 结果
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_C1_TO_V2_FLAG[taskId % 2]
                )

                # ---- ProcessBn2gs1s2LastVec (非 IS_DROP 分支) ----
                # 反向：等 cube 读完上一轮 P L1 再覆写。
                # AscendC 用 needSyncDkMM 标记，其语义等价于「已经历过一轮
                # ComputeDqkv」，即 taskId > 1；此处直接用条件表达以避免
                # 跨迭代可变标量
                needSyncDkMM = taskId > 2
                if needSyncDkMM:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=SYNC_C5_TO_V4_FLAG
                    )
                process_vec4(
                    constInfo, prevInfo, subIdx, p_l1.current(), mm2_t, nzp_ub.current()
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3, event_id=SYNC_V4_TO_C5_FLAG
                )
                # 反向：等 cube 读完上一轮 dS L1
                if needSyncDkMM:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=SYNC_C4_TO_V3_FLAG
                    )
                process_vec3(
                    constInfo,
                    prevInfo,
                    subIdx,
                    ds_l1.current(),
                    mm1_t,
                    mm2_t,
                    sfmg_ub.current(),
                    nzp_ub.current(),
                )
                # 反向：mm1/mm2 的 UB 槽位已读完，放行 cube 覆写。
                # 下标与刚消费的 task N-2 一致，cube 侧 task N 等的正是它
                # (ASC kernel.h:709-710 同一位置 set 这两个 flag)。
                pl.system.set_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_V2_TO_C1_FLAG[taskId % 2]
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_V2_TO_C2_FLAG[taskId % 2]
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3, event_id=SYNC_V3_TO_C3_FLAG
                )

            if taskId >= numBlocks + 1:
                break
            taskId = taskId + 1
            localIdx = pl.min(localIdx + 1, numBlocks)

    # ==== Post 阶段：dQ/dK/dV 的 scale + cast fp32->fp16 ====
    # scale 在矩阵乘之后、fp32 结果上施加(而非提前乘进 dS)：先乘会让
    # fp16 舍入发生在不同量级上，网络训练中误差会累积。
    # 只有 dQ/dK 乘 scale，dV 不乘 —— 对应 ASC 的 `qkvIdx < 2`
    # (post_regbase.h:238)。dV 传 1.0，fp32 乘 1.0 精确。
    # 进 post 前必须 sync_all：要等所有核的 atomicAdd 都落盘。
    pl.system.sync_all()

    with pl.section_vector():
        vBlockIdxP = pl.get_block_idx()
        vecCoreNumP = coreNum * pl.get_subblock_num()
        # depth 2：两个 f32 槽 + 两个 f16 槽，各自独立 mutex_ids，
        # 框架据此判定「下一块的 load」与「本块的 cast/store」无冲突，
        # 从而让 MTE2 / V / MTE3 三条流水重叠。
        # 槽位地址随 d_align 变化，见 post_ub_addrs
        postF32_0, postF32_1, postF16_0, postF16_1 = post_ub_addrs(d_align)
        post_f32 = pl.make_tile_group(
            type=pl.TileType(
                shape=[POST_CHUNK_ROWS, d_align],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
            ),
            addrs=[postF32_0, postF32_1],
            mutex_ids=[19, 20],
        )
        post_f16 = pl.make_tile_group(
            type=pl.TileType(
                shape=[POST_CHUNK_ROWS, d_align],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
            ),
            addrs=[postF16_0, postF16_1],
            mutex_ids=[21, 22],
        )
        # dq/dk 是 D 宽，dv 是 Dv 宽；dv 不乘 scale(对应 ASC 的 qkvIdx < 2)
        process_post(
            tensor_dq_ws_flat,
            tensor_dq_out,
            post_f32,
            post_f16,
            dqRows,
            dQ,
            constInfo.scaleValue,
            vBlockIdxP,
            vecCoreNumP,
        )
        process_post(
            tensor_dk_ws_flat,
            tensor_dk_out,
            post_f32,
            post_f16,
            dkvRows,
            dQ,
            constInfo.scaleValue,
            vBlockIdxP,
            vecCoreNumP,
        )
        process_post(
            tensor_dv_ws_flat,
            tensor_dv_out,
            post_f32,
            post_f16,
            dkvRows,
            dV,
            1.0,
            vBlockIdxP,
            vecCoreNumP,
        )
