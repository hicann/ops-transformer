# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cube 侧计算。

五个 matmul 阶段，编号与 CV 同步 flag 一致::

    C1  mm1 : dP = dO @ V^T
    C2  mm2 : S  = Q  @ K^T
    C3  mm3 : dQ = dS @ K
    C4  mm4 : dK = dS^T @ Q
    C5  mm5 : dV = P^T  @ dO

每个阶段只做「载入 -> matmul -> 输出」，不含分核逻辑、不含跨核同步 ——
同步全部由 :mod:`fag_kernel` 的主循环在阶段之间插入。

D 轴分块（``d_align=192`` 档切成 96+96）是本文件的主要复杂度来源，三种
分块形态各由一个 ``_mm_*_chunk`` 承担：

* ``_mm_k_chunk``        D 在 K 轴，各块累加到同一 acc（C1/C2）
* ``_mm_n_chunk``        D 在 N 轴，各块写不相交的输出列（C3）
* ``_mm_resident_chunk`` D 在 N 轴且每块有各自的常驻 L0C（C4/C5）
"""

import pypto_pro.language as pl

# --- generated imports (fag_fiximports.py) ---
from fag_common import (
    CUBE_BASEM,
    CUBE_BASEN,
    D_CHUNK_OFF0,
    D_CHUNK_OFF1,
    D_CHUNK_W0,
    D_CHUNK_W1,
    D_NCHUNKS,
    VECTOR_BASEM,
    VECTOR_BASEN,
)
# --- end generated imports ---


# ==================================================================
#  C2 —— mm2: S = Q @ K^T
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
    # 完全相同。
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
        pl.min(D_CHUNK_W0[d_align], constInfo.dSize),
        runInfo.s2RealSize,
        True,
    )
    if D_NCHUNKS[d_align] > 1:
        lt1 = left.next()
        rt1 = right.next()
        dRem = constInfo.dSize - D_CHUNK_OFF1[d_align]
        dW1 = pl.max(pl.min(D_CHUNK_W1[d_align], dRem), 0)
        if dW1 > 0:
            _mm_k_chunk(
                lt1,
                rt1,
                at,
                q_t,
                k_t,
                D_CHUNK_OFF1[d_align],
                dW1,
                runInfo.s2RealSize,
                False,
            )
    # Fixpipe L0C->UB，dualDstCtl=1：按 M 切分写入两个 vector 子核
    pl.move(mm2_ub, at, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


# ==================================================================
#  C1 —— mm1: dP = dO @ V^T
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
#  C3 —— mm3: dQ = dS @ K
#  BN2GS1S2：fixpipe 直出 GM，除首块外 AtomicAdd
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
        pl.min(D_CHUNK_W0[d_align], constInfo.dSize),
    )
    if D_NCHUNKS[d_align] > 1:
        dRem = constInfo.dSize - D_CHUNK_OFF1[d_align]
        dW1 = pl.max(pl.min(D_CHUNK_W1[d_align], dRem), 0)
        if dW1 > 0:
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
                dW1,
            )


# ==================================================================
#  C4 —— mm4: dK = dS^T @ Q
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
        pl.min(D_CHUNK_W0[d_align], constInfo.dSize),
        isAcc,
    )
    if D_NCHUNKS[d_align] > 1:
        dRem = constInfo.dSize - D_CHUNK_OFF1[d_align]
        dW1 = pl.max(pl.min(D_CHUNK_W1[d_align], dRem), 0)
        if dW1 > 0:
            _mm_resident_chunk(
                left,
                right,
                acc1.current(),
                ds_t_l1,
                q_t,
                runInfo.s2RealSize,
                runInfo.s1RealSize,
                D_CHUNK_OFF1[d_align],
                dW1,
                isAcc,
            )


# ==================================================================
#  C5 —— mm5: dV = P^T @ dO
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


# ==================================================================
#  BN2 模板的 matmul —— 一个核独占整个 (b, n2) head。
#  L1/L0 物理宽固定 128、不切 D，故不复用上面的 _mm_*_chunk。
# ==================================================================


def iterate_mm_qk_bn2(
    constInfo, runInfo, tensor_q, tensor_k, q_t, k_l1, left, right, acc, mm2_ub, loadKV
):
    """BN2: L1/L0 物理仍是 [128,128]，K 轴用真实 dSize，不分 D 块。"""
    k_t = k_l1.current()
    pl.set_validshape(q_t, [runInfo.s1RealSize, constInfo.dSize])
    pl.set_validshape(k_t, [constInfo.dSize, runInfo.s2RealSize])
    pl.load(
        q_t,
        tensor_q,
        [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q, 0],
        order=[1, 3],
    )
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
    pl.set_validshape(lt, [CUBE_BASEM, constInfo.dSize])
    pl.set_validshape(rt, [constInfo.dSize, runInfo.s2RealSize])
    pl.set_validshape(at, [CUBE_BASEM, runInfo.s2RealSize])
    pl.move(lt, q_t)
    pl.move(rt, k_t)
    pl.matmul(at, lt, rt)
    pl.move(mm2_ub, at, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


def iterate_mm_dyv_bn2(
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
    pl.set_validshape(do_t, [runInfo.s1RealSize, constInfo.dSize])
    pl.set_validshape(v_t, [constInfo.dSize, runInfo.s2RealSize])
    pl.load(
        do_t,
        tensor_dy,
        [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q, 0],
        order=[1, 3],
    )
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
    pl.set_validshape(lt, [CUBE_BASEM, constInfo.dSize])
    pl.set_validshape(rt, [constInfo.dSize, runInfo.s2RealSize])
    pl.set_validshape(at, [CUBE_BASEM, runInfo.s2RealSize])
    pl.move(lt, do_t)
    pl.move(rt, v_t)
    pl.matmul(at, lt, rt)
    pl.move(mm1_ub, at, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


def iterate_mm_dsk_bn2(
    constInfo,
    runInfo,
    tensor_k_nt,
    ds_l1,
    k_l1_n,
    left,
    right,
    acc,
    mm1_ub,
    tensor_dq_ws,
    cBlockIdx,
    loadKV,
):
    """MM3：dQ = dS @ K。写 UB 还是写 workspace 由 is_bn2_multiblk 切。

    单块：Fixpipe DualModeSplitM 进 mm1 UB。
    MultiBlk：Fixpipe 到 workspace；needAtomic = !dqFirst。
    dK/dV flush 与 SYNC_C3_TO_V5 仍在 caller。
    """
    k_t3 = k_l1_n.current()
    pl.set_validshape(k_t3, [runInfo.s2RealSize, constInfo.dSize])
    if loadKV == 1:
        pl.load(
            k_t3,
            tensor_k_nt,
            [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV, 0],
            order=[1, 3],
        )
    lt3 = left.next()
    rt3 = right.next()
    at3 = acc.next()
    # M 用满 CUBE_BASEM，DualModeSplitM 固定在第 64 行切开。
    pl.set_validshape(lt3, [CUBE_BASEM, runInfo.s2RealSize])
    pl.set_validshape(rt3, [runInfo.s2RealSize, constInfo.dSize])
    pl.set_validshape(at3, [CUBE_BASEM, constInfo.dSize])
    pl.move(lt3, ds_l1.current())
    pl.move(rt3, k_t3)
    pl.matmul(at3, lt3, rt3)
    if is_bn2_multiblk == 1:
        rowOff = (cBlockIdx * constInfo.s1Outer + runInfo.s1oIdx) * CUBE_BASEM
        # 不要把 acc 收到 s1Real 再 store。L1/MM3 按 CUBE_BASEM 的 NZ 打包，
        # L0C srcStride = AlignTo16(M)=128；valid_shape 改成 s1Real 后
        # srcStride 变成 AlignTo16(s1Real)，尾块（64/72/1）和 L0C 错位。
        # S=255 尾块 127 能过是因为 AlignTo16(127)=128 碰巧一致。
        # copy_ub2l1 / DualMode 固定切在 64，L1 恒 128 行，所以 store 跟
        # L0C 走 128；输出行数由 last FromGM 的 halfS1RealSize 裁。
        if runInfo.dqFirst == 1:
            pl.store(tensor_dq_ws, at3, [rowOff, 0], atomic=pl.AtomicType.AtomicNone)
        else:
            pl.store(tensor_dq_ws, at3, [rowOff, 0], atomic=pl.AtomicType.AtomicAdd)
    else:
        pl.set_validshape(mm1_ub, [VECTOR_BASEM, VECTOR_BASEN])
        pl.move(mm1_ub, at3, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


def iterate_mm_dsq_bn2(constInfo, runInfo, ds_t_l1, q_t, left, right, acc, isAcc):
    lt = left.next()
    rt = right.next()
    at = acc.current()
    # M 用满 CUBE_BASEN，让 DualModeSplitM 仍在第 64 行切开（与 mm1/mm2 一致）。
    # 若写成 s2RealSize，S2=96 会变成 48+48，和 v6 的 64 行切分错位。
    pl.set_validshape(lt, [CUBE_BASEN, runInfo.s1RealSize])
    pl.set_validshape(rt, [runInfo.s1RealSize, constInfo.dSize])
    pl.set_validshape(at, [CUBE_BASEN, constInfo.dSize])
    pl.move(lt, ds_t_l1)
    pl.move(rt, q_t)
    if isAcc == 1:
        pl.matmul_acc(at, at, lt, rt)
    else:
        pl.matmul(at, lt, rt)


def iterate_mm_pdy_bn2(constInfo, runInfo, p_t_l1, do_t, left, right, acc, isAcc):
    lt = left.next()
    rt = right.next()
    at = acc.current()
    pl.set_validshape(lt, [runInfo.s2RealSize, runInfo.s1RealSize])
    pl.set_validshape(rt, [runInfo.s1RealSize, constInfo.dSize])
    pl.set_validshape(at, [runInfo.s2RealSize, constInfo.dSize])
    pl.move(lt, p_t_l1)
    pl.move(rt, do_t)
    if isAcc == 1:
        pl.matmul_acc(at, at, lt, rt)
    else:
        pl.matmul(at, lt, rt)
