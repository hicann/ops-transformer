# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""片上 tile 的声明。

pypto 的 tile 必须在 ``with pl.section_*()`` 里就地声明，而且 tracer
不接受自定义 Python 类（``CubeBlock(...)`` 会直接报 Unsupported function
call），所以这里是**一个 buffer 一个工厂函数**：调用点写
``q_l1 = make_q_l1()``，声明本身连同它的全部理由留在本文件。

这样拆的收益：

* shape / layout / 地址 / mutex id 只写一处，BN2 与 BN2GS1S2 共用；
* ``fag_kernel`` 的主循环里不再出现 ``pl.make_tile_group`` 与任何裸地址；
* 想知道「某块 buffer 多大、放哪、和谁互斥」，只看本文件与 :mod:`fag_mem`。

**约定：一个工厂只返回一个 tile_group。**
不要把多个 tile_group 打包成元组返回 —— codegen 对「元组里的 tile 元素」有
额外要求（别名视图会被发成 C++ 引用而没有独立名字），实测会报
``Array tuple assignment ... has elements without a C++ name``，而且触发条件
并不直观。逐个返回没有这个坑，代价只是调用点多几行。

**声明顺序有意义**：tile 的声明顺序决定 codegen 里 buffer 的初始化顺序，
调用点请按本文件的书写顺序调用。

BN2 与 BN2GS1S2 共用 CV 共享区、cube L1 与 L0 工作集（窄档 ``d_align`` 64/128
``D_NCHUNKS`` 为 1；192/256 档差别在 L0C：GS1S2 192/256 与 BN2 256 的 dK 复用
acc 槽位作 mm4 scratch、只把 dV 两块常驻，BN2 192 则 dK/dV 都常驻（acc 退回
单槽）；768 档 dK/dV 都是独立单 scratch）。
"""

import pypto_pro.language as pl

# 地址 / mutex 走 fag_mem 别名，避免把整张地址表摊在文件头。
# shape 常量仍 from-import：工厂里到处写 shape=[CUBE_BASEM, ...]，裸名更干净。
import fag_mem as mem
from fag_tiling import Tmpl
from fag_common import (
    CUBE_BASEM,
    CUBE_BASEN,
    D_CHUNK_MAXW,
    D_PHYS_W,
    HEAD_DIM_ALIGN,
    NZP_ROWS,
    SP_ROWS,
    SP_WORK_W,
    VECTOR_BASEM,
    VECTOR_BASEN,
    SINK_REDUCE_BATCH,
)


# 192 档 L0C 按模板分叉（BN2 用 dK/dV 常驻表 _B2，其余用 acc 双槽表）。
# 判定写在工厂**自己体内**，调用点不传标志位：tilingkey 常量（template / d_align）
# 由 pypto 在 trace 时注入被内联函数的闭包，函数体里可以直接引用。
#
# 两条硬约束（都是 pypto 前端实测出来的，别绕）：
#
# 1. **不能把这个判定再包成一个小函数去调用**。当它出现在 ``addrs=`` /
#    ``mutex_ids=`` 这类编译期表达式位置时，被调函数由 Python 直接执行、用
#    **自己的模块 globals** 解析名字，看不到注入的 tilingkey 常量（实测报
#    ``NameError: name 'template' is not defined``）。
# 2. **条件表达式只能放在「选择项」位置**（list 元素、kwarg 值），不能塞进下标：
#    ``mem.UB_Y_A[A if cond else v1w]`` 会让 parser 解析不出被内联的形参
#    （``NameNotFound: v1w``，见 commit d5cbd345d）。正确写法是
#    ``mem.UB_Y_A[A] if cond else mem.UB_Y_A[v1w]``。
#
# 所以这里一律写成内联条件表达式，不用辅助函数、也不用体内局部量。


# ==================================================================
#  CV 共享区 —— 必须声明在两个 section 之外
# ==================================================================
def make_mm1_res():
    """mm1(dP = dO@V^T) 的结果 UB，双槽按 taskId % 2 ping-pong。

    cube 侧 fixpipe 写、vector 侧读，故声明在两个 section 之外。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[VECTOR_BASEM, VECTOR_BASEN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            compact=0,
        ),
        addrs=[mem.UB_MM1_0, mem.UB_MM1_1],
        mutex_ids=mem.MTX_MM1_RES,
    )


def make_mm2_res():
    """mm2(S = Q@K^T) 的结果 UB，双槽 ping-pong。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[VECTOR_BASEM, VECTOR_BASEN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            compact=0,
        ),
        addrs=[mem.UB_MM2_0, mem.UB_MM2_1],
        mutex_ids=mem.MTX_MM2_RES,
    )


def make_ds_l1():
    """dS 的 CV 共享 L1（NZ）：v3 写、mm3/mm4 读。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEM, CUBE_BASEN],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=[mem.L1_DS_ADDR[d_align]],
        mutex_ids=mem.MTX_L1_DS,
    )


def make_p_l1():
    """P 的 CV 共享 L1（NZ）：v4 写、mm5 读。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEM, CUBE_BASEN],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=[mem.L1_P_ADDR[d_align]],
        mutex_ids=mem.MTX_L1_P,
    )


def make_ds_t_l1():
    """dS^T：ds_l1 同一地址上的 ZN 视图，供 mm4 当左矩阵。

    dS^T / P^T 不再单独存一份：[M,N] 按 NZ 存放的字节布局，与 [N,M] 按 ZN
    存放完全一致，所以对同一地址再建一个 ZN 视图，就等于零指令拿到转置。
    vector 侧只 insert 一次(NZ 视图)，cube 侧 mm4/mm5 直接按 ZN 视图 move 到
    L0A —— shape 与 L0A 一致，TMov 通过。

    这替掉了原先的 pl.transpose：实测单次 24.9us(44864 cycles)，整核估算
    5672us，占 8192 场景实测 6011us 的 94%，是性能差距主因。注意 L1->L0
    无法随路转置(TMov static_assert 要求 shape 相同)，Python validator 里
    「允许 2D 转置」的注释能过 parse 但 codegen 失败。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEN, CUBE_BASEM],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=[mem.L1_DS_ADDR[d_align]],
        mutex_ids=mem.MTX_L1_DS_T,
    )


def make_p_t_l1():
    """P^T：p_l1 同一地址上的 ZN 视图，供 mm5 当左矩阵。见 :func:`make_ds_t_l1`。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEN, CUBE_BASEM],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=[mem.L1_P_ADDR[d_align]],
        mutex_ids=mem.MTX_L1_P_T,
    )


# ==================================================================
#  Cube 侧 L1
# ==================================================================
def make_q_l1():
    """Q 的 L1，多槽 double buffer（mm2 写入，mm4 经延后视图复用）。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEM, D_PHYS_W[d_align]],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            compact=1,
        ),
        addrs=mem.L1_Q_ADDRS[d_align],
        mutex_ids=mem.L1_Q_MUTEX[d_align],
    )


def make_k_l1():
    """K^T 的 L1（ZN，即以 [D,S2] 载入），供 mm2 当右矩阵。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[D_PHYS_W[d_align], CUBE_BASEN],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=[mem.L1_K_ADDR[d_align]],
        mutex_ids=mem.MTX_L1_K[d_align],
    )


def make_k_nd_l1():
    """未转置 K[S2,D] 的 L1，供 mm3 当右矩阵 —— 必须独占地址与 mutex。

    源必须是 ND layout 的 tensor_k_nt(ND2ND)；若源是 DN 的 tensor_k，载入会
    按 DN 解释而静默得到 K^T。ZN 版本虽能编过(DN2ZN 合法)但同样不对：
    平台只支持 ND2NZ/DN2NZ/ND2ND/DN2DN/NZ2NZ/DN2ZN。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEN, D_PHYS_W[d_align]],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            compact=1,
        ),
        addrs=[mem.L1_KND_ADDR[d_align]],
        mutex_ids=mem.MTX_L1_K_ND[d_align],
    )


def make_v_l1():
    """V^T 的 L1（ZN），供 mm1 当右矩阵。

    V/dO 是 Dv 宽，但 tile 仍按 d_align 分配(Dv<=D)：多出来的列是 padding，
    由 load/matmul 的 valid_shape 挡掉。这样两侧共用一套地址步长，L1 布局
    不必再分 D/Dv 两种块大小。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[D_PHYS_W[d_align], CUBE_BASEN],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=[mem.L1_V_ADDR[d_align]],
        mutex_ids=mem.MTX_L1_V[d_align],
    )


def make_do_l1():
    """dO 的 L1，多槽 double buffer（mm1 写入，mm5 经延后视图复用）。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEM, D_PHYS_W[d_align]],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            compact=1,
        ),
        addrs=mem.L1_DO_ADDRS[d_align],
        mutex_ids=mem.L1_DO_MUTEX[d_align],
    )


def make_q_lag_view():
    """mm4 读 Q 用的**延后游标视图**（dK = dS^T @ Q 复用 mm2 搬好的 Q）。

    地址与 mutex_id 和 :func:`make_q_l1` 完全相同(共享同一块 L1 与同一套同步
    跟踪)，但游标独立。它从 taskId==2 才开始 next()，因此永远落后两个 task
    —— current() 恰好指向 task N-2 载入的槽位，正是 mm4 需要的那一份。
    这样 Q 每个 (b,n,s1) 块只从 GM 读一次。

    mutex_ids 必须与 writer 视图完全相同：同一块 L1 的两个访问器靠同一套
    id 把「mm2 写」与「mm4 读」排序。
    若给 reader 另一套 id，这层互锁就没了，mm2(task N) 会在 mm4 读完前覆写，
    实测 dK/dV 错(0.61/0.57)。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEM, D_PHYS_W[d_align]],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            compact=1,
        ),
        addrs=mem.L1_Q_ADDRS[d_align],
        mutex_ids=mem.L1_Q_MUTEX[d_align],
    )


def make_do_lag_view():
    """mm5 读 dO 用的延后游标视图。语义同 :func:`make_q_lag_view`。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEM, D_PHYS_W[d_align]],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            compact=1,
        ),
        addrs=mem.L1_DO_ADDRS[d_align],
        mutex_ids=mem.L1_DO_MUTEX[d_align],
    )


# ==================================================================
#  Cube 侧 L0 —— 五个 matmul 共用的工作集
# ==================================================================
# L0A/L0B 一律按「最宽的一个 D 分块」声明：192 档是 96 列而不是 192，于是每块
# 24KB、双缓冲 2x24=48KB<=64KB 得以保留 —— 这正是切 D 轴的首要目的。
# 列宽取 mem.ACC_COLS = max(CUBE_BASEN, 分块宽)：MM2/MM1 的左矩阵是 Q/dO 的 D
# 分块，MM3/MM4/MM5 的左矩阵是 dS/P([M,N])。
def make_l0a():
    """L0A（左矩阵），双缓冲。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEM, mem.ACC_COLS[d_align]],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Left,
            compact=1,
        ),
        addrs=mem.L0AB_ADDRS[d_align],
        mutex_ids=mem.MTX_L0A[d_align],
    )


def make_l0b():
    """L0B（右矩阵），双缓冲。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[mem.ACC_COLS[d_align], CUBE_BASEN],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Right,
            compact=1,
        ),
        addrs=mem.L0AB_ADDRS[d_align],
        mutex_ids=mem.MTX_L0B[d_align],
    )


def make_l0c_acc():
    """mm1/mm2/mm3 轮流使用的 L0C 累加器。

    单缓冲时 fixpipe 写出期间下一个 matmul 只能干等(实测 fixpipe 0.458 而
    mac 仅 0.783)，故所有档位都开两槽；唯一例外是 BN2 192（P2 用 dK/dV 常驻
    的布局，acc 退回单槽，见 :mod:`fag_mem` 的 L0C 布局）。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEM, mem.ACC_COLS[d_align]],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=(
            mem.L0C_ACC_ADDRS_B2[d_align]
            if template == Tmpl.BN2 and d_align == 192
            else mem.L0C_ACC_ADDRS[d_align]
        ),
        mutex_ids=(
            mem.L0C_ACC_MUTEX_B2[d_align]
            if template == Tmpl.BN2 and d_align == 192
            else mem.L0C_ACC_MUTEX[d_align]
        ),
    )


# ==================================================================
#  dK / dV 的 L0C 分块（常驻累加 / mm4 scratch）
# ==================================================================
# dK：192/256 档不驻留，走 fp32 ws（GS1S2 192/256、BN2 256）；下面两个 chunk 与
# acc 同址，只作 mm4 输出 scratch，按 next() 轮转。BN2 192 例外，dK 常驻两槽（B2）。
# dV：常驻，需在同一 s2 块内跨 s1 轮次存活，按 current() 而非 next() 访问；故每个
# D 分块各建独立 tile_group，不做多槽位 group。
def make_dk_l0c_chunk0():
    """dK 第 0 个 D 分块的 L0C 槽。

    64/128 是 dK 的常驻累加器；GS1S2 192/256 与 BN2 256 与 acc 第 0 槽同址（只
    作 mm4 输出 scratch）；BN2 192 有独立常驻槽；768 是独立的单 scratch。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEN, D_CHUNK_MAXW[d_align]],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[
            mem.L0C_DK0_ADDR_B2[d_align]
            if template == Tmpl.BN2 and d_align == 192
            else mem.L0C_DK0_ADDR[d_align]
        ],
        mutex_ids=[
            mem.MTX_L0C_DK0_B2[d_align]
            if template == Tmpl.BN2 and d_align == 192
            else mem.MTX_L0C_DK0[d_align]
        ],
    )


def make_dv_l0c_chunk0():
    """dV 第 0 个 D 分块的常驻 L0C。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEN, D_CHUNK_MAXW[d_align]],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[
            mem.L0C_DV0_ADDR_B2[d_align]
            if template == Tmpl.BN2 and d_align == 192
            else mem.L0C_DV0_ADDR[d_align]
        ],
        mutex_ids=[
            mem.MTX_L0C_DV0_B2[d_align]
            if template == Tmpl.BN2 and d_align == 192
            else mem.MTX_L0C_DV0[d_align]
        ],
    )


def make_dk_l0c_chunk1():
    """dK 第 1 个 D 分块的 L0C 槽。

    GS1S2 192/256 与 BN2 256 与 acc 第 1 槽同址（dK 走 ws，该路径不会用到它）；
    BN2 192 有独立常驻槽；64/128/768 别名到第 0 块 —— ``D_NCHUNKS == 1`` 时
    该形参被 trace 期整块剪掉。该工厂主要为 ``iterate_mm_n_l0c`` 的 ``acc1``
    形参保留。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEN, D_CHUNK_MAXW[d_align]],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[
            mem.L0C_DK1_ADDR_B2[d_align]
            if template == Tmpl.BN2 and d_align == 192
            else mem.L0C_DK1_ADDR[d_align]
        ],
        mutex_ids=[
            mem.MTX_L0C_DK1_B2[d_align]
            if template == Tmpl.BN2 and d_align == 192
            else mem.MTX_L0C_DK1[d_align]
        ],
    )


def make_dv_l0c_chunk1():
    """dV 第 1 个 D 分块的常驻 L0C。见 :func:`make_dk_l0c_chunk1`。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[CUBE_BASEN, D_CHUNK_MAXW[d_align]],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=[
            mem.L0C_DV1_ADDR_B2[d_align]
            if template == Tmpl.BN2 and d_align == 192
            else mem.L0C_DV1_ADDR[d_align]
        ],
        mutex_ids=[
            mem.MTX_L0C_DV1_B2[d_align]
            if template == Tmpl.BN2 and d_align == 192
            else mem.MTX_L0C_DV1[d_align]
        ],
    )


# 64/128 不切 D（D_NCHUNKS=1，D_CHUNK_MAXW==D_PHYS_W==128），BN2 narrow 直接复用
# 上面的 chunk0，无需单独工厂；其余档位见 fag_mem._build_l0c 的预算表。


# ==================================================================
#  Vector 侧 UB（BN2GS1S2）
# ==================================================================
# `v1w` 是 V1 的列宽：一般等于 `dv_align`，768 档为 256。
def make_y_ub(v1w):
    """前向输出 y 的 UB（V1 的输入之一）。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[VECTOR_BASEM, v1w],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=0,
        ),
        addrs=[
            mem.UB_Y_A[HEAD_DIM_ALIGN]
            if template == Tmpl.BN2 and d_align <= 128
            else mem.UB_Y_A[v1w]
        ],
        mutex_ids=mem.MTX_UB_Y,
    )


def make_dy_ub(v1w):
    """上游梯度 dO 的 UB（V1 的输入之一）。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[VECTOR_BASEM, v1w],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=0,
        ),
        addrs=[
            mem.UB_DY_A[HEAD_DIM_ALIGN]
            if template == Tmpl.BN2 and d_align <= 128
            else mem.UB_DY_A[v1w]
        ],
        mutex_ids=mem.MTX_UB_DY,
    )


def make_nzp_ub(v1w):
    """v3/v4 融合 VF 的唯一输出缓冲：65 行 NZ，多出的一行是 bank 错开填充。

    valid_shape 动态设置：insert 前收窄到 rows(框架据此算 65-rows 的源跨距)，
    insert 后必须恢复成 65(见 ``fag_block_vec.copy_ub2l1``)。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[NZP_ROWS, VECTOR_BASEN],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.NZ,
            valid_shape=[-1, -1],
            compact=2,
        ),
        addrs=[
            mem.UB_NZP_A[HEAD_DIM_ALIGN]
            if template == Tmpl.BN2 and d_align <= 128
            else mem.UB_NZP_A[v1w]
        ],
        mutex_ids=mem.MTX_UB_NZP,
    )


def make_sfmg_ub(v1w):
    """softmaxGradFront 的按行结果 sfmg（V1 出、V3 入）。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[VECTOR_BASEM, 1],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.DN,
            compact=0,
        ),
        addrs=[
            mem.UB_SFMG_A[HEAD_DIM_ALIGN]
            if template == Tmpl.BN2 and d_align <= 128
            else mem.UB_SFMG_A[v1w]
        ],
        mutex_ids=mem.MTX_UB_SFMG,
    )


def make_sp_y_ub(v1w):
    """sfmg 前移独立 y 缓冲（pre section 专用）。

    窗口 [0x9000,0x40000) 分时复用 mm1 次槽、mm2_res 与主图，mutex 号
    复用主图 tile 的号（见 fag_mem 的 sfmg_pre 注释），使 pre 最后一次
    store 与主循环首次 load 有显式依赖。

    形状按档取窗口规格；双槽（slots=2，对齐 1102 已上板验证的设计）——
    循环内 .next() 乒乓轮转：load(N+1) 写下一槽、VF(N) 读本槽，物理
    隔离无需跨队列 wait。**不能用单槽 .current()**：pypto 对同一 tile
    的跨队列操作不插 wait，load 会覆盖 VF 正在读的窗口（上板 sfmg 错）。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[SP_ROWS[v1w], SP_WORK_W[v1w]],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=mem.UB_SP_Y_A[v1w],
        mutex_ids=mem.MTX_UB_SP_Y[v1w],
    )


def make_sp_dy_ub(v1w):
    """sfmg 前移独立 dy 缓冲（双槽口径与 make_sp_y_ub 一致）。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[SP_ROWS[v1w], SP_WORK_W[v1w]],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=mem.UB_SP_DY_A[v1w],
        mutex_ids=mem.MTX_UB_SP_DY[v1w],
    )


def make_sp_out_ub(v1w):
    """sfmg 前移独立输出缓冲，depth 2。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[SP_ROWS[v1w], 1],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.DN,
        ),
        addrs=mem.UB_SP_OUT_A[v1w],
        mutex_ids=mem.MTX_UB_SP_OUT[v1w],
    )


def make_sp_out_flat(v1w):
    """sp_out 的 [1,SP_ROWS] 扁平视图：同址同号（mask view 先例），depth 2。

    ws 是 [1,N] 扁平 tensor，而 pl.store 的 order 必须升序——[rows,1] DN tile
    无法用 order=[1,0] 落扁平行。改用本视图走 zt/sink 的恒等映射口径：
    validshape [1,rows] + 无 order，语义为「线性 rows 个 fp32」。
    VF 写 DN 列 tile、store 读本视图，同 mutex 保证先后序。
    与 sp_out 共用同一 (addr, mutex) 槽位对——循环内同步 .next() 保持轮转一致。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[1, SP_ROWS[v1w]],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=mem.UB_SP_OUT_A[v1w],
        mutex_ids=mem.MTX_UB_SP_OUT[v1w],
    )


def make_lse_ub(v1w):
    """softmax_lse 的 UB，双槽 ping-pong。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[VECTOR_BASEM, 1],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            layout=pl.DN,
            valid_shape=[-1, -1],
            compact=0,
        ),
        addrs=[
            mem.UB_LSE0_A[HEAD_DIM_ALIGN]
            if template == Tmpl.BN2 and d_align <= 128
            else mem.UB_LSE0_A[v1w],
            mem.UB_LSE1_A[HEAD_DIM_ALIGN]
            if template == Tmpl.BN2 and d_align <= 128
            else mem.UB_LSE1_A[v1w],
        ],
        mutex_ids=mem.MTX_UB_LSE,
    )


def make_mask_view(v1w):
    """attenmask 的 UB 视图 —— **故意**与 y_ub 同址同 mutex。

    v1 用完 y 之后 v2 才搬 mask，两者复用同一块 UB。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[VECTOR_BASEM, VECTOR_BASEN],
            dtype=pl.DT_UINT8,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=0,
        ),
        addrs=[
            mem.UB_MASK_A[HEAD_DIM_ALIGN]
            if template == Tmpl.BN2 and d_align <= 128
            else mem.UB_MASK_A[v1w]
        ],
        mutex_ids=mem.MTX_UB_Y,
    )


def make_mask_pre_view(v1w):
    """band mask 的第二块（前向窗口），与 dy_ub 同址同 mutex。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[VECTOR_BASEM, VECTOR_BASEN],
            dtype=pl.DT_UINT8,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=0,
        ),
        addrs=[
            mem.UB_MASK_PRE_A[HEAD_DIM_ALIGN]
            if template == Tmpl.BN2 and d_align <= 128
            else mem.UB_MASK_PRE_A[v1w]
        ],
        mutex_ids=mem.MTX_UB_DY,
    )


# ==================================================================
#  Vector 侧 UB（BN2）
# ==================================================================
# narrow BN2（d_align 64/128）没有 pre/post，整张 UB 图沿用 ``HEAD_DIM_ALIGN``
# 那一份（见 fag_mem 的 BN2 槽位说明），上面对应的工厂已按 template/d_align 自己
# 换档，调用点不需要区分；wide BN2（d_align>128）按档走下面的 workspace / L0C
# 两条直出路径。
def make_ws_load_f32_ub():
    """wide（d_align>128）：从 4D fp32 workspace 读回的 [64, chunk] load tile。

    列宽取该档的 D 分块宽（192→96，256/768→128），整块一次 cast，避免 64 列
    的短 burst。地址 alias 到 y（V2 之后已死），尺寸正好相等（24KB/32KB）。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[VECTOR_BASEM, D_CHUNK_MAXW[d_align]],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=[mem.UB_WSF32[d_align]],
        mutex_ids=mem.MTX_UB_WSF32,
    )


def make_ws_cast_f16_ub():
    """wide 的 [64, chunk] 16-bit 出口，与 :func:`make_ws_load_f32_ub` stride 一致。

    alias 到 cast（16KB，chunk=96/128 时占 12/16KB）
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[VECTOR_BASEM, D_CHUNK_MAXW[d_align]],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=[mem.UB_WSF16[d_align]],
        mutex_ids=mem.MTX_UB_WSF16,
    )


def make_l0c_cast_ub():
    """
    v5/v6 出口的 16-bit 缓冲，地址档由工厂自己按 template/d_align 判。
    BN2 narrow 与它整张 UB 图同档（HEAD_DIM_ALIGN）；其余用当前 d_align 档。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[VECTOR_BASEM, mem.CAST_W],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=[
            mem.UB_CAST_A[HEAD_DIM_ALIGN]
            if template == Tmpl.BN2 and d_align <= 128
            else mem.UB_CAST_A[d_align]
        ],
        mutex_ids=mem.MTX_UB_CAST,
    )


def make_l0c_load_ub():
    """
    （BN2 192）：dK chunk1 的 L0C→UB 出口，[64,128] fp32。
     与 mm2_res 同宽同 stride，供 ``muls_cast_vf`` 一次搬完
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[VECTOR_BASEM, VECTOR_BASEN if d_align == 192 else VECTOR_BASEN // 2],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=[mem.UB_L0C_LOAD_A[d_align]],
        mutex_ids=mem.MTX_UB_L0C_LOAD,
    )


def make_sink_part_ub(v1w):
    """dsink 单任务部分和：[1, VECTOR_BASEN] fp32，只写 lane 0。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[1, VECTOR_BASEN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=0,
        ),
        addrs=[
            mem.UB_SINK_PART_A[HEAD_DIM_ALIGN]
            if template == Tmpl.BN2 and d_align <= 128
            else mem.UB_SINK_PART_A[v1w]
        ],
        mutex_ids=mem.MTX_UB_SINK_PART,
    )


def make_sink_row_post_ub(v1w):
    """dsink 收尾归约的批量行 tile：返回 SINK_REDUCE_BATCH 个 [1, VECTOR_BASEN]。

    每个 tile 对应一条 AIV 槽位行；一批里发多次 load、只等一次 bar_mte2。
    内核会被 trace：推导式 / tuple(...) / 嵌套 def 都不支持，故显式展开。
    """
    t0 = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, VECTOR_BASEN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=[mem.UB_SINK_ROW_POST_A[v1w][0]],
        mutex_ids=[mem.MTX_UB_SINK_ROW_POST[0]],
    )
    t1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, VECTOR_BASEN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=[mem.UB_SINK_ROW_POST_A[v1w][1]],
        mutex_ids=[mem.MTX_UB_SINK_ROW_POST[1]],
    )
    t2 = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, VECTOR_BASEN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=[mem.UB_SINK_ROW_POST_A[v1w][2]],
        mutex_ids=[mem.MTX_UB_SINK_ROW_POST[2]],
    )
    t3 = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, VECTOR_BASEN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=0,
        ),
        addrs=[mem.UB_SINK_ROW_POST_A[v1w][3]],
        mutex_ids=[mem.MTX_UB_SINK_ROW_POST[3]],
    )
    return t0, t1, t2, t3


def make_sink_red_post_ub(v1w):
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[1, VECTOR_BASEN],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=0,
        ),
        addrs=[mem.UB_SINK_RED_POST_A[v1w]],
        mutex_ids=mem.MTX_UB_SINK_RED_POST,
    )


# ==================================================================
#  pre / post —— 与主循环被 sync_all 完全隔开，故可整块借用 UB
# ==================================================================
def make_pre_zero_buffer_f32():
    """BN2GS1S2 的清零缓冲（清 fp32 workspace）。单槽位。

    mutex_ids 复用主循环的 id：pre 与主循环被 sync_all 完全隔开，不会同时
    活跃。列宽/行数按 d_align 查表。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[1, mem.PRE_ELEMS[d_align]],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=0,
        ),
        addrs=[mem.UB_PRE_ZERO],
        mutex_ids=mem.MTX_UB_ZERO,
    )


def make_pre_zero_buffer_f16():
    """BN2 ``bn2_need_zero`` 支的清零缓冲（直接清 16-bit out，无 workspace）。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[1, mem.PRE_ELEMS[d_align]],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=0,
        ),
        addrs=[mem.UB_PRE_ZERO],
        mutex_ids=mem.MTX_UB_ZERO,
    )


def make_post_f32_buffer():
    """post 的 fp32 输入缓冲，depth 2。

    与 :func:`make_post_f16_buffer` 各用一套 mutex_ids，框架据此判定「下一块
    的 load」与「本块的 cast/store」无冲突，从而让 MTE2 / V / MTE3 三条流水
    重叠。
    """
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[1, mem.POST_ELEMS[d_align]],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=0,
        ),
        addrs=mem.POST_F32_ADDRS[d_align],
        mutex_ids=mem.MTX_UB_POST_F32,
    )


def make_post_f16_buffer():
    """post 的 16-bit 输出缓冲，depth 2。见 :func:`make_post_f32_buffer`。"""
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[1, mem.POST_ELEMS[d_align]],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=0,
        ),
        addrs=mem.POST_F16_ADDRS[d_align],
        mutex_ids=mem.MTX_UB_POST_F16,
    )
