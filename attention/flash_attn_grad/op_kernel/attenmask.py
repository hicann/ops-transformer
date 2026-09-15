# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Attenmask copy-in and V2 softmax VF."""

import pypto_pro.language as pl

CUBE_BASEM = 128
CUBE_BASEN = 128
VECTOR_BASEM = 64
VECTOR_BASEN = 128
VF_LANES = 64
ATTEN_MASK_COMPRESS = 2048
ATTEN_MASK_MIN = -3.4028234663852886e38


@pl.vector_function
def muls_sel_simple_softmax_vf(
    mm2_ub, lse_ub, mask_ub, scale: pl.DT_FP32, srcM: pl.DT_INT64, rowShift: pl.DT_INT64
):
    """P = exp(S * scale - lse)。

    mask_mode 是 tilingkey，与 d_align 一样在内联 VF 里直接引用，dense
    不编 DIST_DS / select。mask 字节 1=masked。minValue = 0xFF7FFFFF。
    DIST_DS 把 uint8 载入 predicate。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    if mask_mode != 0:
        pregMasked = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
        vregMin = vf.full(ATTEN_MASK_MIN, pregFullExe, dtype=pl.DT_FP32)
    for m in pl.range(0, srcM):
        vregLse = vf.load_align(lse_ub, m + rowShift, dist=pl.LoadDist.BRC_B32)
        for n in pl.range(0, VECTOR_BASEN // VF_LANES):
            off = m * VECTOR_BASEN + n * VF_LANES
            vregSrc = vf.load_align(mm2_ub, off)
            vregMuls = vf.muls(vregSrc, scale, pregFullExe)
            if mask_mode != 0:
                pregMasked = vf.load_align(mask_ub, off, dist=pl.LoadDist.DS)
                vregSel = vf.select(vregMin, vregMuls, pregMasked)
                vregExp = vf.exp_sub(vregSel, vregLse, pregFullExe)
            else:
                vregExp = vf.exp_sub(vregMuls, vregLse, pregFullExe)
            vf.store_align(mm2_ub + off, vregExp, pregFullExe)


@pl.vector_function
def merge_band_mask_vf(mask_next_ub, mask_pre_ub, srcM: pl.DT_INT64):
    """band mask 合并：(~pre | next)，xor 常数 0x01 逐字节。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    vxor = vf.full(1, preg, dtype=pl.DT_UINT8)
    for m in pl.range(0, srcM):
        for n in pl.range(0, VECTOR_BASEN // VF_LANES):
            off = m * VECTOR_BASEN + n * VF_LANES
            vpre = vf.load_align(mask_pre_ub, off)
            vnext = vf.load_align(mask_next_ub, off)
            vnot = vf.xor(vpre, vxor, preg)
            vor = vf.or_(vnot, vnext, preg)
            vf.store_align(mask_next_ub + off, vor, preg)


def compute_offset_for_causal(delta, vec_off):
    """causal mask 偏移：delta<=0 走列偏移，否则走行偏移。"""
    mask_s2 = ATTEN_MASK_COMPRESS
    off_le0 = pl.min(0 - delta, CUBE_BASEM) + vec_off * mask_s2
    off_gt0 = (pl.min(delta, CUBE_BASEN) + vec_off) * mask_s2
    if delta < 1:
        return off_le0
    return off_gt0


def _load_compress_mask(tensor_mask, mask_ub, off, halfS1, s2Real):
    row = off // ATTEN_MASK_COMPRESS
    col = off % ATTEN_MASK_COMPRESS
    pl.set_validshape(mask_ub, [halfS1, s2Real])
    pl.load(mask_ub, tensor_mask, [row, col], order=[0, 1])
    pl.set_validshape(mask_ub, [VECTOR_BASEM, VECTOR_BASEN])


def copy_in_atten_mask(constInfo, runInfo, subIdx, tensor_mask, mask_ub, mask_pre_ub):
    """按 mask_mode 把压缩 attenmask 搬进 UB。"""
    if runInfo.halfS1RealSize == 0:
        return
    s1Offset = runInfo.queryOffset
    s2Offset = runInfo.keyOffset
    vec_off = subIdx * VECTOR_BASEM
    halfS1 = runInfo.halfS1RealSize
    s2Real = runInfo.s2RealSize
    if mask_mode == 3:
        deltaN = constInfo.s1Size - constInfo.s2Size
        delta = s1Offset - s2Offset - deltaN
        off = compute_offset_for_causal(delta, vec_off)
        _load_compress_mask(tensor_mask, mask_ub, off, halfS1, s2Real)
        return
    preT = constInfo.s1Token
    nextT = constInfo.s2Token
    deltaPre = s1Offset - s2Offset - preT - 1
    deltaNext = s1Offset - s2Offset + nextT
    off = compute_offset_for_causal(deltaNext, vec_off)
    _load_compress_mask(tensor_mask, mask_ub, off, halfS1, s2Real)
    preFactor = deltaPre + 1 + CUBE_BASEM
    if preFactor > 0:
        off_pre = compute_offset_for_causal(deltaPre, vec_off)
        _load_compress_mask(tensor_mask, mask_pre_ub, off_pre, halfS1, s2Real)
        merge_band_mask_vf(mask_ub, mask_pre_ub, halfS1)
