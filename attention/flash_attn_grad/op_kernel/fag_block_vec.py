# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Vector 侧编排。

阶段编号与 CV 同步 flag 一致::

    V1  sfmg = rowsum(dy * y)
    V2  P    = softmax(S*scale - lse) [+mask]
    V3  dS   = (dP - sfmg) * P -> cast -> L1
    V4  P    -> cast -> L1
    V5/V6    dQ/dK 的 muls(scale) + cast

本文件只做**搬运与编排**：GM->UB 的 load、UB->L1 的 insert、选哪一档 VF；
真正的寄存器级计算在 :mod:`fag_vector_api` 与 :mod:`attenmask`。

pre / post 两个阶段也放在这里：它们同样是纯 vector、同样按 metadata 给的
行区间切，只是被 ``sync_all`` 与主循环完全隔开。
"""

import pypto_pro.language as pl

# --- generated imports (fag_fiximports.py) ---
from fag_common import (
    CUBE_BASEM,
    NZP_ROWS,
    VECTOR_BASEM,
    VECTOR_BASEN,
    VF_LANES,
    VF_LANES_FP16,
)
from fag_mem import BN2_CAST_W, BN2_V1_ROWS, POST_ELEMS, PRE_ELEMS
from fag_vector_api import (
    broadcast_sub_mul_cast_vf,
    cast_p_vf,
    fill_zero_fp16_vf,
    fill_zero_vf,
    muls_cast_vf,
    softmax_grad_front_row192_vf,
    softmax_grad_front_row_vf,
)
from fag_schedule import (
    META_FAG_DKV_ROW_ENDS,
    META_FAG_DKV_ROW_STARTS,
    META_FAG_DQ_ROW_ENDS,
    META_FAG_DQ_ROW_STARTS,
)
from attenmask import copy_in_atten_mask, muls_sel_simple_softmax_vf
# --- end generated imports ---

# ==================================================================
#  UB -> CV 共享 L1
# ==================================================================


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


# ==================================================================
#  V1 —— sfmg = rowsum(dy * y)
# ==================================================================


def process_vec1(
    constInfo, runInfo, subIdx, tensor_y, tensor_dy, y_ub, dy_ub, sfmg_ub, rowStride
):
    """softmaxGradFront: sfmg = rowsum(dy * y)。一趟 VECTOR_BASEM 行。"""
    pl.set_validshape(y_ub, [VECTOR_BASEM, constInfo.dvSize])
    pl.set_validshape(dy_ub, [VECTOR_BASEM, constInfo.dvSize])
    pl.load(
        y_ub,
        tensor_y,
        [runInfo.idx0Q, runInfo.loadBase, runInfo.idx2Q, 0],
        order=[1, 3],
    )
    pl.load(
        dy_ub,
        tensor_dy,
        [runInfo.idx0Q, runInfo.loadBase, runInfo.idx2Q, 0],
        order=[1, 3],
    )
    w0 = pl.min(constInfo.dvSize, 128)
    if dv_align == 192:
        w1 = pl.max(constInfo.dvSize - 128, 0)
        softmax_grad_front_row192_vf(
            sfmg_ub,
            y_ub,
            dy_ub,
            runInfo.halfS1RealSize,
            runInfo.rowShift,
            rowStride,
            (w0 + 1) // 2,
            w0 // 2,
            (w1 + 1) // 2,
            w1 // 2,
        )
    else:
        softmax_grad_front_row_vf(
            sfmg_ub,
            y_ub,
            dy_ub,
            runInfo.halfS1RealSize,
            runInfo.rowShift,
            rowStride,
            (w0 + 1) // 2,
            w0 // 2,
        )


# ==================================================================
#  V2 —— P = simpleSoftmax(S * scale, lse) [+ atten mask]
# ==================================================================


def process_vec2(
    constInfo,
    runInfo,
    subIdx,
    tensor_lse,
    mm2_ub,
    lse_ub,
    tensor_mask,
    mask_ub,
    mask_pre_ub,
):
    headQ = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx
    pl.load(lse_ub, tensor_lse, [runInfo.boIdx, headQ, runInfo.loadBase], order=[2, 0])
    if mask_mode != 0:
        copy_in_atten_mask(
            constInfo, runInfo, subIdx, tensor_mask, mask_ub, mask_pre_ub
        )
    muls_sel_simple_softmax_vf(
        mm2_ub,
        lse_ub,
        mask_ub,
        constInfo.scaleValue,
        runInfo.halfS1RealSize,
        runInfo.rowShift,
    )


# ==================================================================
#  V3 —— dS = (dP - sfmg) * P，cast + ND2NZ 落 L1（mm4 读同址 ZN 视图）
# ==================================================================


def process_vec3(constInfo, runInfo, subIdx, ds_l1, mm1_ub, mm2_ub, sfmg_ub, nzp_ub):
    rowOff = subIdx * VECTOR_BASEM
    # nd/nz 不裁剪：列方向的脏数据由 matmul 的 valid_shape 与 store 挡掉。
    # VF 固定写满 VECTOR_BASEM 行：NZ+1 的分形列间距由行数决定，按尾块
    # 缩小会让间距漂移。尾块只在 insert 时用 valid_shape 裁掉。
    broadcast_sub_mul_cast_vf(
        nzp_ub, mm1_ub, mm2_ub, sfmg_ub, constInfo.scaleValue, VECTOR_BASEM
    )
    # 只写一次 L1：mm4 通过 ds_t_l1(同地址的 ZN 视图)直接读到 dS^T，
    # 无需再做 transpose/insert
    copy_ub2l1(ds_l1, nzp_ub, rowOff)


# ==================================================================
#  V4 —— P cast + ND2NZ 落 L1（mm5 读同址 ZN 视图）
# ==================================================================


def process_vec4(constInfo, runInfo, subIdx, p_l1, mm2_ub, nzp_ub):
    rowOff = subIdx * VECTOR_BASEM
    cast_p_vf(nzp_ub, mm2_ub, VECTOR_BASEM)
    # 同 process_vec3：mm5 从 p_t_l1(同地址 ZN 视图)读 P^T
    copy_ub2l1(p_l1, nzp_ub, rowOff)


# ==================================================================
#  Pre —— dq/dk/dv 累加区清零
# ==================================================================


def process_pre(tensor_ws, zt, elemStart, nElems):
    """把一段连续 GM 清零，按一维元素切。

    tensor_ws 是 [1, totalElems] 的扁平视图（B*S*N*D 紧凑）。本核负责
    [elemStart, elemStart+nElems)。UB 按 PRE_ELEMS 切 full tile；尾块用
    trip count 0/1 的第二个 pl.range。
    PyPTO tile 必须是二维，故写成 [1, N]，语义仍是一维元素。
    zt 已由调用方填零，本函数只 store。dtype 跟 zt / tensor_ws（fp32 ws
    或 fp16 out）。
    """
    chunk = PRE_ELEMS[d_align]
    nFull = nElems // chunk
    rem = nElems - nFull * chunk
    off = elemStart
    pl.set_validshape(zt, [1, chunk])
    for _ in pl.range(0, nFull):
        pl.store(tensor_ws, zt, [0, off])
        off = off + chunk
    nTail = (rem + chunk - 1) // chunk
    for _ in pl.range(0, nTail):
        pl.set_validshape(zt, [1, rem])
        pl.store(tensor_ws, zt, [0, off])


def process_pre_from_meta(meta, fagOff, tensor_dq, tensor_dk, tensor_dv, dQ, dV, zt):
    """按 metadata 行区间清 dq/dk/dv。GS1S2 清 fp32 workspace，bn2_need_zero 清 fp16 out。

    AICPU FagSplitRows：dqRows=B*S1*N1、dkvRows=B*S2*N2，按 AIV 均分行。
    元素起点 = rowStart * 行宽。填零 VF 跟 tilingkey：bn2_need_zero==1 走 fp16。
    """
    vBlockIdx = pl.get_block_idx()
    dqRowStart = pl.astype(
        pl.getval(meta, fagOff + META_FAG_DQ_ROW_STARTS + vBlockIdx), pl.DT_INT64
    )
    dqRowEnd = pl.astype(
        pl.getval(meta, fagOff + META_FAG_DQ_ROW_ENDS + vBlockIdx), pl.DT_INT64
    )
    dkvRowStart = pl.astype(
        pl.getval(meta, fagOff + META_FAG_DKV_ROW_STARTS + vBlockIdx), pl.DT_INT64
    )
    dkvRowEnd = pl.astype(
        pl.getval(meta, fagOff + META_FAG_DKV_ROW_ENDS + vBlockIdx), pl.DT_INT64
    )
    dqNElems = (dqRowEnd - dqRowStart) * dQ
    dkvNElems = (dkvRowEnd - dkvRowStart) * dQ
    dvNElems = (dkvRowEnd - dkvRowStart) * dV
    preElems = pl.min(PRE_ELEMS[d_align], pl.max(dqNElems, pl.max(dkvNElems, dvNElems)))
    pl.set_validshape(zt, [1, PRE_ELEMS[d_align]])
    if bn2_need_zero == 1:
        fill_zero_fp16_vf(zt, (preElems + VF_LANES_FP16 - 1) // VF_LANES_FP16)
    else:
        fill_zero_vf(zt, (preElems + VF_LANES - 1) // VF_LANES)
    process_pre(tensor_dq, zt, dqRowStart * dQ, dqNElems)
    process_pre(tensor_dk, zt, dkvRowStart * dQ, dkvNElems)
    process_pre(tensor_dv, zt, dkvRowStart * dV, dvNElems)


# ==================================================================
#  Post —— dQ/dK/dV 的 scale + cast
# ==================================================================


def process_post(tensor_ws, tensor_out, f32_ub, f16_ub, scale, elemStart, nElems):
    """fp32 workspace -> muls(scale) -> cast fp16，按一维元素切。

    本核 [elemStart, elemStart+nElems)。full tile 的 VF 段数是编译期常量；
    尾块用 trip count 0/1 的第二个 pl.range。
    """
    chunk = POST_ELEMS[d_align]
    nFull = nElems // chunk
    rem = nElems - nFull * chunk
    nsegsFull = chunk // VECTOR_BASEN
    off = elemStart
    for _ in pl.range(0, nFull):
        ft = f32_ub.next()
        ht = f16_ub.next()
        pl.set_validshape(ft, [1, chunk])
        pl.set_validshape(ht, [1, chunk])
        pl.load(ft, tensor_ws, [0, off])
        muls_cast_vf(ht, ft, scale, nsegsFull)
        pl.store(tensor_out, ht, [0, off])
        off = off + chunk
    nTail = (rem + chunk - 1) // chunk
    nsegsTail = (rem + VECTOR_BASEN - 1) // VECTOR_BASEN
    for _ in pl.range(0, nTail):
        ft = f32_ub.next()
        ht = f16_ub.next()
        pl.set_validshape(ft, [1, rem])
        pl.set_validshape(ht, [1, rem])
        pl.load(ft, tensor_ws, [0, off])
        muls_cast_vf(ht, ft, scale, nsegsTail)
        pl.store(tensor_out, ht, [0, off])


# ==================================================================
#  BN2 模板的 vector 阶段
# ==================================================================


def process_vec1_bn2(
    constInfo, runInfo, subIdx, tensor_y, tensor_dy, y_ub, dy_ub, sfmg_ub, rowStride
):
    pl.set_validshape(y_ub, [BN2_V1_ROWS, constInfo.dSize])
    pl.set_validshape(dy_ub, [BN2_V1_ROWS, constInfo.dSize])
    pl.load(
        y_ub,
        tensor_y,
        [runInfo.idx0Q, runInfo.loadBase, runInfo.idx2Q, 0],
        order=[1, 3],
    )
    pl.load(
        dy_ub,
        tensor_dy,
        [runInfo.idx0Q, runInfo.loadBase, runInfo.idx2Q, 0],
        order=[1, 3],
    )
    softmax_grad_front_row_vf(
        sfmg_ub,
        y_ub,
        dy_ub,
        runInfo.halfS1RealSize,
        runInfo.rowShift,
        rowStride,
        constInfo.dEvenSize,
        constInfo.dOddSize,
    )


def process_vec3_bn2(
    constInfo, runInfo, subIdx, ds_l1, mm1_ub, mm2_ub, sfmg_ub, nzp_ub
):
    """dS 与 BN2GS1S2 用同一个 VF：都不乘 scale。

    scale 在 mm3/mm4 之后由 v5/v6 的 Muls+Cast 施加。dV 不经过 dS，不乘 scale。
    """
    rowOff = subIdx * VECTOR_BASEM
    broadcast_sub_mul_cast_vf(
        nzp_ub, mm1_ub, mm2_ub, sfmg_ub, constInfo.scaleValue, VECTOR_BASEM
    )
    copy_ub2l1(ds_l1, nzp_ub, rowOff)


def process_muls_cast_dq_bn2(tensor_dq, src_ub, dst_ub, constInfo, runInfo, subIdx):
    """v5：mm3 的 fp32 UB → muls(scale) → cast fp16 → dQ GM。

    DualModeSplitM 在第 64 行切开，子核 0/1 各拿半块 S1。
    """
    rows = runInfo.halfS1RealSize
    pl.set_validshape(src_ub, [VECTOR_BASEM, BN2_CAST_W])
    pl.set_validshape(dst_ub, [VECTOR_BASEM, BN2_CAST_W])
    muls_cast_vf(
        dst_ub, src_ub, constInfo.scaleValue, VECTOR_BASEM * BN2_CAST_W // VECTOR_BASEN
    )
    if rows > 0:
        seqOff = runInfo.queryOffset + subIdx * VECTOR_BASEM
        pl.set_validshape(dst_ub, [rows, constInfo.dSize])
        pl.store(
            tensor_dq,
            dst_ub,
            [runInfo.idx0Q, seqOff, runInfo.idx2Q, 0],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicNone,
        )
        pl.set_validshape(dst_ub, [VECTOR_BASEM, BN2_CAST_W])


def process_muls_cast_dk_bn2(tensor_dk, src_ub, dst_ub, constInfo, runInfo, subIdx):
    """v6：mm4 的 fp32 UB → muls(scale) → cast fp16 → dK GM。"""
    rowOff = subIdx * VECTOR_BASEM
    rows = pl.max(pl.min(runInfo.s2RealSize - rowOff, VECTOR_BASEM), 0)
    pl.set_validshape(src_ub, [VECTOR_BASEM, BN2_CAST_W])
    pl.set_validshape(dst_ub, [VECTOR_BASEM, BN2_CAST_W])
    muls_cast_vf(
        dst_ub, src_ub, constInfo.scaleValue, VECTOR_BASEM * BN2_CAST_W // VECTOR_BASEN
    )
    if rows > 0:
        seqOff = runInfo.keyOffset + rowOff
        pl.set_validshape(dst_ub, [rows, constInfo.dSize])
        pl.store(
            tensor_dk,
            dst_ub,
            [runInfo.idx0KV, seqOff, runInfo.idx2KV, 0],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicNone,
        )
        pl.set_validshape(dst_ub, [VECTOR_BASEM, BN2_CAST_W])


def process_dq_multiblk(
    tensor_dq, tensor_dq_ws, src_ub, dst_ub, constInfo, runInfo, subIdx, cBlockIdx
):
    """只在 isLastS1Outer 从 workspace load。

    累加已由 cube 按 needAtomic 写在 GM 上。src_ub 是 load 缓冲。
    """
    rows = runInfo.halfS1RealSize
    rowOff = (
        cBlockIdx * constInfo.s1Outer + runInfo.s1oIdx
    ) * CUBE_BASEM + subIdx * VECTOR_BASEM
    pl.set_validshape(src_ub, [VECTOR_BASEM, BN2_CAST_W])
    if rows > 0:
        pl.set_validshape(src_ub, [rows, constInfo.dSize])
        pl.load(src_ub, tensor_dq_ws, [rowOff, 0])
        pl.set_validshape(src_ub, [VECTOR_BASEM, BN2_CAST_W])
    process_muls_cast_dq_bn2(tensor_dq, src_ub, dst_ub, constInfo, runInfo, subIdx)
