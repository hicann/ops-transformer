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

    C1  mm1 : dP = dO @ V^T     → ``iterate_mm_k``（Dy 宽）
    C2  mm2 : S  = Q  @ K^T     → ``iterate_mm_k``（D 宽）
    C3  mm3 : dQ = dS @ K       → ``iterate_mm_dsk``
    C4  mm4 : dK = dS^T @ Q     → ``iterate_mm_n``（D 宽）
    C5  mm5 : dV = P^T  @ dO    → ``iterate_mm_n``（Dv 宽）

每个阶段只做「载入 -> matmul -> 输出」，不含分核逻辑、不含跨核同步 ——
同步全部由 :mod:`fag_kernel` 的主循环在阶段之间插入。

D 轴分块：64/128 一块，192/256 两块，768 是 3 窗 × 128+128。
C4/C5 驻留 L0C（``d_align<=192``）必须编译期拆 ``acc0`` / ``acc1``，
不能在 ``pl.range`` 里用 ``i==0`` 选 tensor —— PyPTO 会把 ``at`` 绑成
同一块，192 的第二段 96 列写进 acc0，flush 的 acc1 是垃圾。

其余编译期分流：

* ``d_align==768``：每窗从 GM 重装，空块 ``next()`` 仍推进
* ``d_align>192``：C4/C5 每块 atomic 出 GM，否则驻留 L0C
* ``template==BN2``：C3 写出 UB / 2D workspace；C4 的 M 钉 ``CUBE_BASEN``
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
from fag_tiling import Tmpl
# --- end generated imports ---

# 768 档：物理窗 256、窗内 128+128。
_D768_STEP = 128
_D768_PHYS = 256


def _chunk_step():
    return _D768_STEP if d_align == 768 else D_CHUNK_W0[d_align]


def _chunk_off(i):
    return i * _chunk_step()


def _chunk_local(i):
    return (i % 2) * _D768_STEP if d_align == 768 else _chunk_off(i)


def _chunk_win(i):
    return (i // 2) * _D768_PHYS if d_align == 768 else 0


def _chunk_valid_d(dimSize, i):
    """第 i 块沿 D/Dv 的有效列数：按真实 dim 截断，空块为 0。"""
    return pl.max(pl.min(_chunk_step(), dimSize - _chunk_off(i)), 0)


def dv_chunk_valid(constInfo, chunkIdx):
    return _chunk_valid_d(constInfo.dvSize, chunkIdx)


# ==================================================================
#  三种片上 matmul：K 轴累加 / N 轴写出 / N 轴驻留
# ==================================================================
def _mm_k_chunk(lt, rt, at, srcL, srcR, dOff, validD, s2Real, isFirst):
    """MM2/MM1 的一个 K 分块：D 落在 K 轴，各块累加到同一个 acc。"""
    pl.set_validshape(lt, [CUBE_BASEM, validD])
    pl.set_validshape(rt, [validD, s2Real])
    pl.set_validshape(at, [CUBE_BASEM, s2Real])
    pl.move(lt, srcL, [0, dOff])
    pl.move(rt, srcR, [dOff, 0])
    if isFirst:
        pl.matmul(at, lt, rt)
    else:
        pl.matmul_acc(at, at, lt, rt)


def _mm_n_chunk(
    left, right, acc, srcL, srcR, tensor_out, outIdx, mReal, kReal, dOff, validD, l1Off
):
    """N 轴一块：next() 必推进；validD==0 不算（768 空块 / Dv 截断）。"""
    lt = left.next()
    rt = right.next()
    at = acc.next()
    if validD > 0:
        pl.set_validshape(lt, [mReal, kReal])
        pl.set_validshape(rt, [kReal, validD])
        pl.set_validshape(at, [mReal, validD])
        pl.move(lt, srcL)
        pl.move(rt, srcR, [0, l1Off])
        pl.matmul(at, lt, rt)
        pl.store(
            tensor_out,
            at,
            [outIdx[0], outIdx[1], outIdx[2], dOff],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicAdd,
        )


def _mm_resident_chunk(left, right, at, srcL, srcR, mReal, kReal, dOff, validD, isAcc):
    """C4/C5 驻留 L0C：D 在 N 轴，跨 s1 累加进同一块。"""
    lt = left.next()
    rt = right.next()
    pl.set_validshape(lt, [mReal, kReal])
    pl.set_validshape(rt, [kReal, validD])
    pl.set_validshape(at, [mReal, validD])
    pl.move(lt, srcL)
    pl.move(rt, srcR, [0, dOff])
    if isAcc == 1:
        pl.matmul_acc(at, at, lt, rt)
    else:
        pl.matmul(at, lt, rt)


# ==================================================================
#  C1 / C2 —— D 在 K 轴，一块 acc，Fixpipe DualModeSplitM
# ==================================================================
def iterate_mm_k(
    constInfo,
    runInfo,
    tensor_l,
    tensor_r,
    l_t,
    r_l1,
    left,
    right,
    acc,
    mm_ub,
    loadKV,
    dimSize,
):
    """C1/C2 共用：``[S1, dim] @ [dim, S2]``，dim 是 D 或 Dv。

    M 保持 CUBE_BASEM：DualModeSplitM 切在第 64 行。若把 M 裁到 s1Real，
    尾块切分点会漂，和 vector 侧固定 64 错位。
    """
    r_t = r_l1.current()
    if d_align != 768:
        pl.set_validshape(l_t, [runInfo.s1RealSize, dimSize])
        pl.set_validshape(r_t, [dimSize, runInfo.s2RealSize])
        pl.load(
            l_t,
            tensor_l,
            [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q, 0],
            order=[1, 3],
        )
        if loadKV == 1:
            pl.load(
                r_t,
                tensor_r,
                [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV, 0],
                order=[3, 1],
            )
    at = acc.next()
    if d_align == 768:
        for i in pl.range(0, D_NCHUNKS[d_align]):
            dOff = _chunk_off(i)
            localOff = _chunk_local(i)
            validD = _chunk_valid_d(dimSize, i)
            if i % 2 == 0:
                winOff = _chunk_win(i)
                winW = pl.max(pl.min(_D768_PHYS, dimSize - winOff), 0)
                pl.set_validshape(l_t, [runInfo.s1RealSize, winW])
                pl.set_validshape(r_t, [winW, runInfo.s2RealSize])
                if winW > 0:
                    pl.load(
                        l_t,
                        tensor_l,
                        [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q, winOff],
                        order=[1, 3],
                    )
                    pl.load(
                        r_t,
                        tensor_r,
                        [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV, winOff],
                        order=[3, 1],
                    )
            lt = left.next()
            rt = right.next()
            if validD > 0:
                _mm_k_chunk(
                    lt, rt, at, l_t, r_t, localOff, validD, runInfo.s2RealSize, i == 0
                )
    else:
        # 64/128/192/256：编译期一块或两块。isFirst 不能绑 pl.range 的 i。
        lt = left.next()
        rt = right.next()
        w0 = pl.min(D_CHUNK_W0[d_align], dimSize)
        _mm_k_chunk(
            lt,
            rt,
            at,
            l_t,
            r_t,
            D_CHUNK_OFF0[d_align],
            w0,
            runInfo.s2RealSize,
            True,
        )
        if D_NCHUNKS[d_align] > 1:
            lt1 = left.next()
            rt1 = right.next()
            w1 = pl.max(pl.min(D_CHUNK_W1[d_align], dimSize - D_CHUNK_OFF1[d_align]), 0)
            if w1 > 0:
                _mm_k_chunk(
                    lt1,
                    rt1,
                    at,
                    l_t,
                    r_t,
                    D_CHUNK_OFF1[d_align],
                    w1,
                    runInfo.s2RealSize,
                    False,
                )
    pl.move(mm_ub, at, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


def iterate_mm_qk(
    constInfo, runInfo, tensor_q, tensor_k, q_t, k_l1, left, right, acc, mm2_ub, loadKV
):
    iterate_mm_k(
        constInfo,
        runInfo,
        tensor_q,
        tensor_k,
        q_t,
        k_l1,
        left,
        right,
        acc,
        mm2_ub,
        loadKV,
        constInfo.dSize,
    )


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
    iterate_mm_k(
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
        constInfo.dvSize,
    )


# ==================================================================
#  C3 —— D 在 N 轴，各块 atomic 出不相交的 dQ 列
# ==================================================================
def iterate_mm_dsk(
    constInfo,
    runInfo,
    tensor_k,
    tensor_dq_ws,
    loadKV,
    ds_l1,
    k_l1,
    left,
    right,
    acc,
    mm1_ub,
    cBlockIdx,
):
    """C3：``dQ = dS @ K``。

    GS1S2 各块 atomic 出 4D workspace；BN2 单块后写 UB 或 2D workspace
    （``is_bn2_multiblk``）。``mm1_ub`` / ``cBlockIdx`` 只在 BN2 用到。
    """
    k_t = k_l1.current()
    outIdx = [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q]
    if d_align != 768:
        pl.set_validshape(k_t, [runInfo.s2RealSize, constInfo.dSize])
        if loadKV == 1:
            pl.load(
                k_t,
                tensor_k,
                [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV, 0],
                order=[1, 3],
            )
    for i in pl.range(0, D_NCHUNKS[d_align]):
        dOff = _chunk_off(i)
        localOff = _chunk_local(i)
        validD = _chunk_valid_d(constInfo.dSize, i)
        if d_align == 768:
            if i % 2 == 0:
                winOff = _chunk_win(i)
                winW = pl.max(pl.min(_D768_PHYS, constInfo.dSize - winOff), 0)
                pl.set_validshape(k_t, [runInfo.s2RealSize, winW])
                if winW > 0:
                    pl.load(
                        k_t,
                        tensor_k,
                        [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV, winOff],
                        order=[1, 3],
                    )
        if template == Tmpl.BN2:
            lt = left.next()
            rt = right.next()
            at = acc.next()
            if validD > 0:
                pl.set_validshape(lt, [CUBE_BASEM, runInfo.s2RealSize])
                pl.set_validshape(rt, [runInfo.s2RealSize, validD])
                pl.set_validshape(at, [CUBE_BASEM, validD])
                pl.move(lt, ds_l1)
                pl.move(rt, k_t, [0, localOff])
                pl.matmul(at, lt, rt)
                if is_bn2_multiblk == 1:
                    rowOff = (
                        cBlockIdx * constInfo.s1Outer + runInfo.s1oIdx
                    ) * CUBE_BASEM
                    if runInfo.dqFirst == 1:
                        pl.store(
                            tensor_dq_ws,
                            at,
                            [rowOff, 0],
                            atomic=pl.AtomicType.AtomicNone,
                        )
                    else:
                        pl.store(
                            tensor_dq_ws,
                            at,
                            [rowOff, 0],
                            atomic=pl.AtomicType.AtomicAdd,
                        )
                else:
                    pl.set_validshape(mm1_ub, [VECTOR_BASEM, VECTOR_BASEN])
                    pl.move(
                        mm1_ub,
                        at,
                        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
                    )
        elif d_align == 768 or validD > 0:
            _mm_n_chunk(
                left,
                right,
                acc,
                ds_l1,
                k_t,
                tensor_dq_ws,
                outIdx,
                runInfo.s1RealSize,
                runInfo.s2RealSize,
                dOff,
                validD,
                localOff,
            )


# ==================================================================
#  C4 / C5 —— D 在 N 轴：≤192 驻留 L0C，>192 atomic 出 GM
# ==================================================================
def iterate_mm_n(
    constInfo,
    runInfo,
    srcL,
    srcR,
    left,
    right,
    acc0,
    tensor_out,
    tensor_src,
    dimSize,
    srcIdx,
    outIdx,
    mReal,
):
    """C4/C5 写出路径（``d_align>192``）：每块 atomic 出 GM。

    ``tensor_out`` / ``tensor_src`` 必须是真 Tensor。新前端不允许把
    ``None`` 默认值再往下传 —— 驻留路径走 ``iterate_mm_n_l0c``。
    """
    kReal = runInfo.s1RealSize
    for i in pl.range(0, D_NCHUNKS[d_align]):
        dOff = _chunk_off(i)
        localOff = _chunk_local(i)
        validD = _chunk_valid_d(dimSize, i)
        if d_align == 768:
            if i % 2 == 0:
                winOff = _chunk_win(i)
                winW = pl.max(pl.min(_D768_PHYS, dimSize - winOff), 0)
                pl.set_validshape(srcR, [runInfo.s1RealSize, winW])
                if winW > 0:
                    pl.load(
                        srcR,
                        tensor_src,
                        [srcIdx[0], srcIdx[1], srcIdx[2], winOff],
                        order=[1, 3],
                    )
        if d_align == 768 or validD > 0:
            _mm_n_chunk(
                left,
                right,
                acc0,
                srcL,
                srcR,
                tensor_out,
                outIdx,
                mReal,
                kReal,
                dOff,
                validD,
                localOff,
            )


def iterate_mm_n_l0c(
    constInfo,
    runInfo,
    srcL,
    srcR,
    left,
    right,
    acc0,
    acc1,
    isAcc,
    dimSize,
    mReal,
):
    """C4/C5 驻留 L0C（``d_align<=192``）。不接收 Tensor 参数。

    acc0/acc1 必须是编译期两路。D=64/128 一块；D=192 是 96+96
    （Dv=128 时 C5 第二块 validD=32）。
    """
    kReal = runInfo.s1RealSize
    off0 = D_CHUNK_OFF0[d_align]
    w0 = pl.max(pl.min(D_CHUNK_W0[d_align], dimSize - off0), 0)
    if w0 > 0:
        _mm_resident_chunk(
            left,
            right,
            acc0.current(),
            srcL,
            srcR,
            mReal,
            kReal,
            off0,
            w0,
            isAcc,
        )
    if D_NCHUNKS[d_align] > 1:
        off1 = D_CHUNK_OFF1[d_align]
        w1 = pl.max(pl.min(D_CHUNK_W1[d_align], dimSize - off1), 0)
        if w1 > 0:
            _mm_resident_chunk(
                left,
                right,
                acc1.current(),
                srcL,
                srcR,
                mReal,
                kReal,
                off1,
                w1,
                isAcc,
            )


def iterate_mm_dsq(
    constInfo,
    runInfo,
    ds_t_l1,
    q_t,
    left,
    right,
    acc0,
    acc1=None,
    isAcc=0,
    tensor_out=None,
    tensor_src=None,
):
    if template == Tmpl.BN2:
        mReal = CUBE_BASEN
    else:
        mReal = runInfo.s2RealSize
    if d_align > 192:
        iterate_mm_n(
            constInfo,
            runInfo,
            ds_t_l1,
            q_t,
            left,
            right,
            acc0,
            tensor_out,
            tensor_src,
            constInfo.dSize,
            [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q],
            [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV],
            mReal,
        )
    else:
        iterate_mm_n_l0c(
            constInfo,
            runInfo,
            ds_t_l1,
            q_t,
            left,
            right,
            acc0,
            acc1,
            isAcc,
            constInfo.dSize,
            mReal,
        )


def iterate_mm_pdy(
    constInfo,
    runInfo,
    p_t_l1,
    do_t,
    left,
    right,
    acc0,
    acc1=None,
    isAcc=0,
    tensor_out=None,
    tensor_src=None,
):
    if d_align > 192:
        iterate_mm_n(
            constInfo,
            runInfo,
            p_t_l1,
            do_t,
            left,
            right,
            acc0,
            tensor_out,
            tensor_src,
            constInfo.dvSize,
            [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q],
            [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV],
            runInfo.s2RealSize,
        )
    else:
        iterate_mm_n_l0c(
            constInfo,
            runInfo,
            p_t_l1,
            do_t,
            left,
            right,
            acc0,
            acc1,
            isAcc,
            constInfo.dvSize,
            runInfo.s2RealSize,
        )


def flush_dk(constInfo, runInfo, tensor_dk_ws, dk_acc0, dk_acc1):
    """把 L0C 上累加完的 dK 写回 GM。只用于 d_align≤192。"""
    i0 = runInfo.idx0KV
    i1 = runInfo.keyOffset
    i2 = runInfo.idx2KV
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


def flush_dv(constInfo, runInfo, tensor_dv_ws, dv_acc0, dv_acc1):
    """把 L0C 上累加完的 dV 写回 GM。只用于 d_align≤192。"""
    i0 = runInfo.idx0KV
    i1 = runInfo.keyOffset
    i2 = runInfo.idx2KV
    if dv_chunk_valid(constInfo, 0) > 0:
        pl.store(
            tensor_dv_ws,
            dv_acc0.current(),
            [i0, i1, i2, D_CHUNK_OFF0[d_align]],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicAdd,
        )
    if D_NCHUNKS[d_align] > 1:
        if dv_chunk_valid(constInfo, 1) > 0:
            pl.store(
                tensor_dv_ws,
                dv_acc1.current(),
                [i0, i1, i2, D_CHUNK_OFF1[d_align]],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicAdd,
            )


def flush_dkv(
    constInfo, runInfo, tensor_dk_ws, tensor_dv_ws, dk_acc0, dk_acc1, dv_acc0, dv_acc1
):
    """普通路径兼容入口；newDeter 在各自 FIX token 内分开 flush。"""
    flush_dk(constInfo, runInfo, tensor_dk_ws, dk_acc0, dk_acc1)
    flush_dv(constInfo, runInfo, tensor_dv_ws, dv_acc0, dv_acc1)
