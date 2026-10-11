/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_grouped_matmul_dequant_utils.h
 * \brief __aicore__ inline helpers used by the unified qgmmd kernel.
 *
 *  Scalar (filled here):
 *      ScaleFromMax(v)  -> 127/v  with EPS guard
 *      RecipScale(v)    -> 1/v    with EPS guard
 *
 *  Pipeline helpers:
 *      ScatterZZToL1            UB int8 ND -> L1 ZZ fractal via MTE3
 *      PrefetchWeightToL1<B>    GM int8 ZN/blocked ZN -> L1 via MTE2
 *      EncodeFp32ScaleToUint64  V-pipe scatter: fp32 scale -> VDEQ16 uint64
 */
#ifndef QUANT_GROUPED_MATMUL_DEQUANT_UTILS_H
#define QUANT_GROUPED_MATMUL_DEQUANT_UTILS_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "quant_grouped_matmul_dequant_config.h"

namespace AscendC {

// ============================================================
// DynamicQuant scalars - single-rounding S-register math.
// Kernel-side: quantize multiplier stays in S-register and is consumed
// immediately by Muls; it never crosses UB->GM.
// ============================================================

/** 127 / v. EPS guard matches DynamicQuant (0 for degenerate max). */
__aicore__ inline float ScaleFromMax(float v)
{
    return (v <= DQ_MAX_EPS) ? 0.0f : 127.0f / v;
}

/** 1 / v. Used when the forward scale (v = 127/max) has been materialized
 *  and we need its reciprocal (second-rounding path). */
__aicore__ inline float RecipScale(float v)
{
    return (v <= DQ_MAX_EPS) ? 0.0f : 1.0f / v;
}

// ============================================================
// Pipeline helpers - declared with full signature so includers can
// resolve overloads; body filled in the corresponding stage.
// These are free functions (not class methods) so the caller hands in
// every buffer + stride it needs; no reliance on member-variable state.
// ============================================================

/**
 * MTE3 scatter: linear-K UB bytes -> ZZ fractal L1 layout.
 * ZZ = [mAligned/16 turns][16 rows][K/32 fractals][32 bytes].
 *
 * Per-row MTE3 DataCopy with dstStride = (M_FRACTAL - 1) blocks walks the
 * ZZ layout: row r of turn t lands at L1 offset
 *     l1TurnStride * t + UB_BLOCK_SIZE * r + kFracOff
 * where kFracOff = kByteOff * M_FRACTAL.
 *
 * Caller supplies mAligned (rounded to M_FRACTAL), kByteOff (K bytes
 * already written), chunkSize (K bytes this call), and l1TurnStride
 * (bytes from turn t to t+1 within L1; typically K_padded * M_FRACTAL).
 */
__aicore__ inline void ScatterZZToL1(LocalTensor<int8_t> xL1Dst, LocalTensor<int8_t> ubSrc, uint32_t mAligned,
                                     uint32_t kByteOff, uint32_t chunkSize, uint32_t l1TurnStride, uint32_t mValid)
{
    constexpr uint32_t UB_BLOCK_SIZE = 32;
    constexpr uint32_t M_FRAC = M_FRACTAL;
    const uint16_t chunkBlocks = static_cast<uint16_t>(chunkSize / UB_BLOCK_SIZE);
    const DataCopyParams zzP = {chunkBlocks, 1, 0, static_cast<uint16_t>(M_FRAC - 1)};
    const uint32_t kFracOff = kByteOff * M_FRAC;
    const uint32_t m0Chunk = M_FRAC * chunkSize;

    // Only scatter mValid rows. Padding rows (mValid..mAligned) are left
    // as-is in L1 - Mmad reads them as L0A padding whose output is
    // discarded downstream (CopyOut writes mCount rows only). This avoids
    // reading past the ubSrc allocation when mValid < M_FRAC * zzTurns.
    uint32_t zzTurns = mAligned / M_FRAC;
    uint32_t l1Base = kFracOff;
    uint32_t ubBase = 0;
    uint32_t rowsLeft = mValid;
    for (uint32_t t = 0; t < zzTurns; ++t) {
        uint32_t l1r = l1Base;
        uint32_t ubr = ubBase;
        uint32_t rowsThis = (rowsLeft < M_FRAC) ? rowsLeft : M_FRAC;
        for (uint32_t r = 0; r < rowsThis; ++r) {
            DataCopy(xL1Dst[l1r], ubSrc[ubr], zzP);
            l1r += UB_BLOCK_SIZE;
            ubr += chunkSize;
        }
        rowsLeft -= rowsThis;
        l1Base += l1TurnStride;
        ubBase += m0Chunk;
        if (rowsLeft == 0)
            break;
    }
}

/**
 * MTE2 prefetch: weight GM -> L1. BlockedZN template selects between
 * the flat-ZN stride math and the 6-D blocked-ZN stride math.
 *
 * ZN fractal = 16 rows * 32 cols = 512 bytes = 16 blocks of 32 B.
 * Per K1 row the N dimension holds N1 fractals (flat) or blockN1 per block
 * (blocked). DataCopy pulls ngCount fractals per K1 row, stepping srcStride
 * 32-B blocks between rows.
 *
 * Offset math:
 *   flat:    base + ngStart * 512
 *   blocked: base + blockIdx * (K1_total * blockN1 * 512) + ngInBlock * 512
 *
 * Stride math:
 *   flat:    srcStride = (N1             - ngCount) * 16
 *   blocked: srcStride = (actualBlockNG  - ngCount) * 16
 *
 * actualBlockNG = min(blockN1, N1 - blockIdx * blockN1) accounts for the
 * last block that may have fewer N-groups than blockN1.
 */
template <bool BlockedZN>
__aicore__ inline void PrefetchWeightToL1(LocalTensor<int8_t> wL1, const GlobalTensor<int8_t>& weightGm,
                                          uint64_t expertBaseOff, uint32_t colBatch, uint32_t nSize, uint32_t baseNG,
                                          uint32_t N1, uint32_t K1q, uint32_t K, uint32_t weightBlockN1,
                                          uint32_t k1Start = 0)
{
    (void)colBatch;
    constexpr uint32_t K0_N0 = 512;      // bytes per ZN fractal (16 * 32)
    constexpr uint32_t FRAC_BLOCKS = 16; // K0_N0 / UB_BLOCK_SIZE

    const uint32_t ngCount = nSize / N_FRACTAL;
    const uint32_t blockLen = ngCount * FRAC_BLOCKS;
    const uint32_t ngStart = baseNG;

    uint64_t wGmOff;
    uint32_t effN1;

    if constexpr (!BlockedZN) {
        // Flat ZN: rows of K1 each hold N1 fractals of K0_N0 bytes.
        // K-pass kp starts at K1-row `k1Start`; per-row stride = N1 * K0_N0.
        // Compute uint32 fractal offsets first, widen to uint64 only for the
        // final base sum. (310P scalar unit can't lower uint64 * uint64.)
        const uint32_t kpFracOff = k1Start * N1;        // uint32
        const uint32_t fracTotal = kpFracOff + ngStart; // uint32
        wGmOff = expertBaseOff + static_cast<uint64_t>(fracTotal) * K0_N0;
        effN1 = N1;
        (void)K;
        (void)weightBlockN1;
    } else {
        const uint32_t blockNG = weightBlockN1;
        const uint32_t blockIdx = ngStart / blockNG;
        const uint32_t ngInBlock = ngStart - blockIdx * blockNG;
        const uint32_t remN1 = N1 - blockIdx * blockNG;
        const uint32_t actualBlockNG = (blockNG < remN1) ? blockNG : remN1;
        const uint32_t K1Total = K / K_FRACTAL_INT8_BYTES;
        // Same uint32-first pattern: compute per-block fractal count as
        // uint32 (K1Total * blockNG fits for practical shapes), widen once.
        const uint32_t blockFracs = K1Total * blockNG;
        const uint64_t totalBlockSize = static_cast<uint64_t>(blockFracs) * K0_N0;
        const uint32_t kpFracOff = k1Start * actualBlockNG;
        const uint32_t innerFracs = kpFracOff + ngInBlock;
        wGmOff = expertBaseOff + static_cast<uint64_t>(blockIdx) * totalBlockSize +
                 static_cast<uint64_t>(innerFracs) * K0_N0;
        effN1 = actualBlockNG;
    }

    const uint32_t srcStride = (effN1 - ngCount) * FRAC_BLOCKS;
    const DataCopyParams wP = {static_cast<uint16_t>(K1q), static_cast<uint16_t>(blockLen),
                               static_cast<uint16_t>(srcStride), 0};
    DataCopy(wL1, weightGm[wGmOff], wP);
}

/**
 * V-pipe scatter-encode: fp32 scale buffer -> uint64 VDEQ16 buffer.
 * Layout invariant: for each scale s_i in [0, nSize), the uint64 slot i
 * holds (0x00004000ULL << 32) | reinterpret<uint32>(s_i). Even int32 slots
 * hold the fp32 bits; odd slots hold the VDEQ16 marker 0x00004000.
 *
 * scatterIdx must be preseeded as byte offsets idx[i] = i * 8 (one uint64
 * is 8 bytes). CreateVecIndex + Muls(8) in InitUbLayout produces this
 * once per kernel instance (nSize persistent, independent of cb).
 */
__aicore__ inline void EncodeFp32ScaleToUint64(LocalTensor<uint64_t> dstU64, LocalTensor<float> srcFp32,
                                               LocalTensor<uint32_t> scatterIdx, uint32_t nSize)
{
    LocalTensor<int32_t> dstI32 = dstU64.ReinterpretCast<int32_t>();
    LocalTensor<uint32_t> dstU32 = dstU64.ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> srcU32 = srcFp32.ReinterpretCast<uint32_t>();
    // Fill every int32 lane (even + odd) with the 0x00004000 marker; then
    // Scatter rewrites the even lanes with the fp32 bits.
    Duplicate<int32_t>(dstI32, static_cast<int32_t>(0x00004000), nSize * 2);
    PipeBarrier<PIPE_V>();
    Scatter<uint32_t>(dstU32, srcU32, scatterIdx, 0, nSize);
    PipeBarrier<PIPE_V>();
}

} // namespace AscendC

#endif // QUANT_GROUPED_MATMUL_DEQUANT_UTILS_H
