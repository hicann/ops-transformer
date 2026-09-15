# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""VF 微内核。

本文件只放 ``@pl.vector_function`` 的**纯寄存器级**实现，一律「进 UB
地址、出 UB 地址」，不碰 GM、不碰 L1、不读 runInfo；搬运与编排在上层的
:mod:`fag_block_vec`。

VF 可以按 dtype / 列宽分档写多份（如 128 与 192 两档 softmaxGradFront），
调用方只需在一处选档。
"""

import pypto_pro.language as pl

# --- generated imports (fag_fiximports.py) ---
from fag_common import (
    NZP_BLOCK_STRIDE,
    NZP_REPEAT_STRIDE,
    VECTOR_BASEN,
    VF_LANES,
    VF_LANES_FP16,
)
# --- end generated imports ---


# `vf` 由 parser 按语法识别(_call_parser.py: attrs[0] == "vf")，无需 import

# ---- V1: softmaxGradFront  sfmg = rowsum(dy * y) ----


@pl.vector_function
def softmax_grad_front_row_vf(
    sfmg_ub,
    y_ub,
    dy_ub,
    srcM: pl.DT_INT64,
    rowShift: pl.DT_INT64,
    rowStride: pl.DT_INT64,
    evenSize: pl.DT_INT64,
    oddSize: pl.DT_INT64,
):
    """sfmg：Dv≤128，整行一次 128-fp16 load，even/odd mask 盖真实宽度。"""
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    pregEven = vf.update_mask(evenSize, dtype=pl.DT_FP32)
    pregOdd = vf.update_mask(oddSize, dtype=pl.DT_FP32)
    aregSfmg = vf.unalign_reg_for_store()
    for m in pl.range(0, srcM):
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride)
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
        vregMul = vf.mul(vregDy, vregY, pregEven)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd)
        vregRed = vf.reduce_sum(vregMul, pregEven)
        vregRed2 = vf.reduce_sum(vregMul2, pregOdd)
        vregSum = vf.add(vregRed, vregRed2, pregFullExe)
        vf.store_unalign(sfmg_ub, vregSum, aregSfmg, 1, post_update=True)
    vf.store_unalign_post(sfmg_ub, aregSfmg, 0, post_update=True)


@pl.vector_function
def softmax_grad_front_row192_vf(
    sfmg_ub,
    y_ub,
    dy_ub,
    srcM: pl.DT_INT64,
    rowShift: pl.DT_INT64,
    rowStride: pl.DT_INT64,
    evenSize: pl.DT_INT64,
    oddSize: pl.DT_INT64,
    evenSize2: pl.DT_INT64,
    oddSize2: pl.DT_INT64,
):
    """sfmg：Dv=192 档。先收前 128 列，再 colOff=128 收剩余，两趟 reduce 相加。

    对标老 FAG `MySoftmaxGradFrontCast<..., 384, HEAD_DIM_ALIGN>`：独立档位
    函数，不把 128/192 写进同一个 VF。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    pregEven = vf.update_mask(evenSize, dtype=pl.DT_FP32)
    pregOdd = vf.update_mask(oddSize, dtype=pl.DT_FP32)
    pregEven2 = vf.update_mask(evenSize2, dtype=pl.DT_FP32)
    pregOdd2 = vf.update_mask(oddSize2, dtype=pl.DT_FP32)
    aregSfmg = vf.unalign_reg_for_store()
    for m in pl.range(0, srcM):
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride)
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
        vregMul = vf.mul(vregDy, vregY, pregEven)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd)
        vregRed = vf.reduce_sum(vregMul, pregEven)
        vregRed2 = vf.reduce_sum(vregMul2, pregOdd)
        vregSum = vf.add(vregRed, vregRed2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 128)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 128)
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
        vregMul = vf.mul(vregDy, vregY, pregEven2)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd2)
        vregRed = vf.reduce_sum(vregMul, pregEven2)
        vregRed2 = vf.reduce_sum(vregMul2, pregOdd2)
        vregSum2 = vf.add(vregRed, vregRed2, pregFullExe)
        vregSum = vf.add(vregSum, vregSum2, pregFullExe)
        vf.store_unalign(sfmg_ub, vregSum, aregSfmg, 1, post_update=True)
    vf.store_unalign_post(sfmg_ub, aregSfmg, 0, post_update=True)


# ---- V3: dS = (dP - sfmg) * P，随后 cast + ND2NZ ----


@pl.vector_function
def broadcast_sub_mul_cast_vf(
    nzp_ub, mm1_ub, mm2_ub, sfmg_ub, scale: pl.DT_FP32, srcM: pl.DT_INT64
):
    """dS = (dP - broadcast(sfmg)) * P，并 cast 成 fp16。

    sfmg 按行 BRC_B32 广播，128 列拆成两个 64 lane 寄存器；
    两个 fp32 寄存器 cast 后交错写回一个 fp16 行。
    此处不乘 scale（scale 在 dQ/dK 后处理施加）。
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
        # fp32 -> fp16：Even/Odd 分别 cast 进 b16 寄存器的两个半区再 Or 合并。
        vregCastEven = vf.astype(
            vregDs0, pregFullExeB16, dtype=pl.DT_FP16, layout=pl.CastLayout.ZERO
        )
        vregCastOdd = vf.astype(
            vregDs1, pregFullExeB16, dtype=pl.DT_FP16, layout=pl.CastLayout.ONE
        )
        vregCastRes = vf.or_(vregCastEven, vregCastOdd, pregFullExeB16)
        # DataBlock 拷贝模式(vsstb)：一条 store 同时完成 ND->NZ 重排，
        # 省掉独立的 pl.move —— 后者跑在 vector 计算流水上。
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


# ---- V4: P cast + ND2NZ ----


@pl.vector_function
def cast_p_vf(nzp_ub, mm2_ub, srcM: pl.DT_INT64):
    """P: fp32 -> fp16，并按 DataBlock 模式写成 NZ。"""
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
        # 省掉独立的 pl.move —— 后者跑在 vector 计算流水上。
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


# ---- pre: 清零 ----


@pl.vector_function
def fill_zero_vf(dst_ub, nvecs: pl.DT_INT64):
    """把 UB tile 的前 nvecs 个 vreg(每个 VF_LANES 个 fp32)填 0。

    1D tile 连续排布。调用方传 ceil(nElems / VF_LANES)。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vregZero = vf.full(0.0, pregFullExe, dtype=pl.DT_FP32)
    for i in pl.range(0, nvecs):
        vf.store_align(dst_ub + i * VF_LANES, vregZero, pregFullExe)


@pl.vector_function
def fill_zero_fp16_vf(dst_ub, nvecs: pl.DT_INT64):
    """把 UB tile 的前 nvecs 个 vreg(每个 VF_LANES_FP16 个 fp16)填 0。"""
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    vregZero = vf.full(0.0, pregFullExe, dtype=pl.DT_FP16)
    for i in pl.range(0, nvecs):
        vf.store_align(dst_ub + i * VF_LANES_FP16, vregZero, pregFullExe)


# ---- post / V5 / V6: fp32 -> muls(scale) -> fp16 ----


@pl.vector_function
def muls_cast_vf(dst_ub, src_ub, scale: pl.DT_FP32, nsegs: pl.DT_INT64):
    """fp32 -> muls(scale) -> fp16，按 128 个元素一段扁平遍历整块 tile。

    Muls/Cast 直接吃整块元素，完全不看行宽。Vec tile 是紧凑 RowMajor，行距就等于
    声明宽度 PHYS_W，整块在 UB 里连续，可以当一维数组遍历。

    按行走的老写法在 D≤64 时很吃亏：一次 DINTLV_B32 能覆盖 128 个 fp32，
    但一行只有 64 个，even/odd 掩码各只开一半 lane，VF 利用率 50%。扁平后
    三条掩码恒为全满，既省掉每次调用的 plt_b32/plt_b16 标量配置，vmuls 也
    跑在满 lane 上。

    掩码全满会连 padding 列一起算(D=47 时每行多算 17 列的残值)，但那些列
    被 store 的 valid_shape 挡在 UB 里，出不到 GM。调用方保证
    rows*PHYS_W % 128 == 0，故段边界不会越出 tile。
    """
    pregFull32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFull16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    for m in pl.range(0, nsegs):
        vregEven, vregOdd = vf.load_align(
            src_ub, m * VECTOR_BASEN, dist=pl.LoadDist.DINTLV_B32
        )
        vregEven = vf.muls(vregEven, scale, pregFull32)
        vregOdd = vf.muls(vregOdd, scale, pregFull32)
        vregCastEven = vf.astype(
            vregEven, pregFull16, dtype=pl.DT_FP16, layout=pl.CastLayout.ZERO
        )
        vregCastOdd = vf.astype(
            vregOdd, pregFull16, dtype=pl.DT_FP16, layout=pl.CastLayout.ONE
        )
        vregRes = vf.or_(vregCastEven, vregCastOdd, pregFull16)
        vf.store_align(dst_ub + m * VECTOR_BASEN, vregRes, pregFull16)
