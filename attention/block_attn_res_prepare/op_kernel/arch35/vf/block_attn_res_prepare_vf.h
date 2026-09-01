/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_VF_H
#define BLOCK_ATTN_RES_PREPARE_VF_H

#include "kernel_operator.h"

namespace BlockAttnResPrepareVF {

using HighLowPart = AscendC::Reg::HighLowPart;
using LoadDist = AscendC::Reg::LoadDist;
using MaskMergeMode = AscendC::Reg::MaskMergeMode;
using MaskPattern = AscendC::Reg::MaskPattern;
using MaskReg = AscendC::Reg::MaskReg;
using MemType = AscendC::Reg::MemType;
using ReduceType = AscendC::Reg::ReduceType;
using StoreDist = AscendC::Reg::StoreDist;

template <typename T>
using RegTensor = AscendC::Reg::RegTensor<T>;

using AscendC::Reg::Add;
using AscendC::Reg::Adds;
using AscendC::Reg::CreateMask;
using AscendC::Reg::Div;
using AscendC::Reg::Duplicate;
using AscendC::Reg::Exp;
using AscendC::Reg::LoadAlign;
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::Mul;
using AscendC::Reg::MulAddDst;
using AscendC::Reg::Muls;
using AscendC::Reg::Reduce;
using AscendC::Reg::RegTraitNumOne;
using AscendC::Reg::Sqrt;
using AscendC::Reg::StoreAlign;
using AscendC::Reg::Sub;
using AscendC::Reg::UpdateMask;

constexpr uint32_t FP32_REG_ELEMS = 64U;
constexpr uint32_t STAT_VECTOR_COUNT = 2U;
constexpr uint32_t SCALAR_BLOCK_ELEMS = 8U;
constexpr uint32_t SUM_SQUARE_OFFSET = 0U;
constexpr uint32_t DOT_OFFSET = FP32_REG_ELEMS;
constexpr uint32_t MAX_OFFSET = STAT_VECTOR_COUNT * FP32_REG_ELEMS;
constexpr uint32_t SUM_OFFSET = MAX_OFFSET + SCALAR_BLOCK_ELEMS;
constexpr float FP32_LOWEST_FINITE = -3.4028234663852886e+38F;

__simd_vf__ inline void InitializeEmptyOnlineSoftmax(__ubuf__ float *statAddr)
{
    RegTensor<float> maxReg;
    RegTensor<float> sumReg;
    MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();

    Duplicate(maxReg, FP32_LOWEST_FINITE, oneMask);
    Duplicate(sumReg, 0.0F, oneMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + MAX_OFFSET, maxReg, oneMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + SUM_OFFSET, sumReg, oneMask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_vf__ inline void FillZero(__ubuf__ float *outputAddr, uint32_t validD)
{
    RegTensor<float> zeroReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    Duplicate(zeroReg, 0.0F, allMask);

    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        StoreAlign<float, StoreDist::DIST_NORM>(outputAddr + offset, zeroReg, validMask);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

// These raw UB arguments are internal static byte offsets. UB offset zero is valid and must not be treated as nullptr.
template <bool FIRST_D_TILE>
__simd_vf__ inline void AccumulateSquareDot(__ubuf__ float *qAddr, __ubuf__ float *vAddr, __ubuf__ float *statAddr,
                                            uint32_t validD, uint32_t nIndex)
{
    RegTensor<float> qReg;
    RegTensor<float> vReg;
    RegTensor<float> squareReg;
    RegTensor<float> dotReg;
    RegTensor<float> squareAccReg;
    RegTensor<float> dotAccReg;
    RegTensor<float> squareReduceReg;
    RegTensor<float> dotReduceReg;

    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();
    Duplicate(squareAccReg, 0.0F, allMask);
    Duplicate(dotAccReg, 0.0F, allMask);

    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadAlign<float, LoadDist::DIST_NORM>(qReg, qAddr + offset);
        LoadAlign<float, LoadDist::DIST_NORM>(vReg, vAddr + offset);
        Mul<float, MaskMergeMode::ZEROING>(squareReg, vReg, vReg, validMask);
        Mul<float, MaskMergeMode::ZEROING>(dotReg, qReg, vReg, validMask);
        Add<float, MaskMergeMode::ZEROING>(squareAccReg, squareAccReg, squareReg, allMask);
        Add<float, MaskMergeMode::ZEROING>(dotAccReg, dotAccReg, dotReg, allMask);
    }

    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(squareReduceReg, squareAccReg, allMask);
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(dotReduceReg, dotAccReg, allMask);

    if constexpr (!FIRST_D_TILE) {
        RegTensor<float> oldValueReg;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(oldValueReg, statAddr + SUM_SQUARE_OFFSET + nIndex);
        Add<float, MaskMergeMode::ZEROING>(squareReduceReg, squareReduceReg, oldValueReg, oneMask);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(oldValueReg, statAddr + DOT_OFFSET + nIndex);
        Add<float, MaskMergeMode::ZEROING>(dotReduceReg, dotReduceReg, oldValueReg, oneMask);
    }

    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + SUM_SQUARE_OFFSET + nIndex, squareReduceReg,
                                                         oneMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + DOT_OFFSET + nIndex, dotReduceReg, oneMask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_vf__ inline void FinalizeSingleBlock(__ubuf__ float *statAddr, float reciprocalD, float eps)
{
    RegTensor<float> sumSquareReg;
    RegTensor<float> dotReg;
    RegTensor<float> rmsReg;
    RegTensor<float> zReg;
    RegTensor<float> oneReg;
    MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();

    LoadAlign<float, LoadDist::DIST_NORM>(sumSquareReg, statAddr + SUM_SQUARE_OFFSET);
    LoadAlign<float, LoadDist::DIST_NORM>(dotReg, statAddr + DOT_OFFSET);
    Muls<float, float, MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, reciprocalD, oneMask);
    Adds<float, float, MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, eps, oneMask);
    Sqrt<float, MaskMergeMode::ZEROING>(rmsReg, sumSquareReg, oneMask);
    Div<float, MaskMergeMode::ZEROING>(zReg, dotReg, rmsReg, oneMask);
    Duplicate(oneReg, 1.0F, oneMask);

    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + MAX_OFFSET, zReg, oneMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + SUM_OFFSET, oneReg, oneMask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_vf__ inline void FinalizeSoftmax(__ubuf__ float *statAddr, uint32_t validN, float reciprocalD, float eps)
{
    RegTensor<float> sumSquareReg;
    RegTensor<float> dotReg;
    RegTensor<float> rmsReg;
    RegTensor<float> zReg;
    RegTensor<float> maxReg;
    RegTensor<float> maxBroadcastReg;
    RegTensor<float> expReg;
    RegTensor<float> expSumReg;

    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();
    uint32_t remaining = validN;
    MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);

    LoadAlign<float, LoadDist::DIST_NORM>(sumSquareReg, statAddr + SUM_SQUARE_OFFSET);
    LoadAlign<float, LoadDist::DIST_NORM>(dotReg, statAddr + DOT_OFFSET);
    Muls<float, float, MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, reciprocalD, validMask);
    Adds<float, float, MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, eps, validMask);
    Sqrt<float, MaskMergeMode::ZEROING>(rmsReg, sumSquareReg, validMask);
    Div<float, MaskMergeMode::ZEROING>(zReg, dotReg, rmsReg, validMask);

    Reduce<ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(maxReg, zReg, validMask);
    Duplicate<float, HighLowPart::LOWEST, MaskMergeMode::ZEROING>(maxBroadcastReg, maxReg, allMask);
    Sub<float, MaskMergeMode::ZEROING>(expReg, zReg, maxBroadcastReg, validMask);
    Exp<float, MaskMergeMode::ZEROING>(expReg, expReg, validMask);
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(expSumReg, expReg, validMask);

    // E reuses the physical dot buffer after Z no longer needs to be retained.
    StoreAlign<float, StoreDist::DIST_NORM>(statAddr + DOT_OFFSET, expReg, validMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + MAX_OFFSET, maxReg, oneMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + SUM_OFFSET, expSumReg, oneMask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <bool FIRST_BLOCK>
__simd_vf__ inline void WeightedAccumulate(__ubuf__ float *vAddr, __ubuf__ float *weightAddr,
                                           __ubuf__ float *outputAddr, uint32_t validD)
{
    RegTensor<float> vReg;
    RegTensor<float> weightReg;
    RegTensor<float> outputReg;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(weightReg, weightAddr);

    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadAlign<float, LoadDist::DIST_NORM>(vReg, vAddr + offset);
        if constexpr (FIRST_BLOCK) {
            Mul<float, MaskMergeMode::ZEROING>(outputReg, vReg, weightReg, validMask);
        } else {
            LoadAlign<float, LoadDist::DIST_NORM>(outputReg, outputAddr + offset);
            MulAddDst<float, MaskMergeMode::ZEROING>(outputReg, vReg, weightReg, validMask);
        }
        StoreAlign<float, StoreDist::DIST_NORM>(outputAddr + offset, outputReg, validMask);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_vf__ inline void CopySingleBlock(__ubuf__ float *vAddr, __ubuf__ float *outputAddr, uint32_t validD)
{
    RegTensor<float> vReg;
    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadAlign<float, LoadDist::DIST_NORM>(vReg, vAddr + offset);
        StoreAlign<float, StoreDist::DIST_NORM>(outputAddr + offset, vReg, validMask);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

} // namespace BlockAttnResPrepareVF

#endif // BLOCK_ATTN_RES_PREPARE_VF_H
