/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_partial_rotary_mul_grad_common.h
 * \brief Common VF functions for InplacePartialRotaryMulGrad, shared across all templates.
 */

#ifndef __INPLACE_PARTIAL_ROTARY_MUL_GRAD_COMMON_H__
#define __INPLACE_PARTIAL_ROTARY_MUL_GRAD_COMMON_H__

#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"
#include "op_kernel/math_util.h"

namespace InplacePartialRotaryMulGrad {
using namespace AscendC;

constexpr uint32_t HALF_INTERLEAVE_COEF = 2;
constexpr uint32_t QUARTER_MODE_COEF = 4;

__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

constexpr uint32_t BLOCK_TYPE_SIZE = GetUbBlockSize();
constexpr uint32_t VREG_SIZE_BITS = 256U;                            // Ascend950 vector register width
constexpr uint32_t VL_FLOAT32_SIZE = VREG_SIZE_BITS / sizeof(float); // 256/4 = 64 floats per vector

// mode 常量，与 InplacePartialRotaryMulGradMode 对齐
constexpr int64_t ROTARY_MODE_HALF = 0;
constexpr int64_t ROTARY_MODE_INTERLEAVE = 1;
constexpr int64_t ROTARY_MODE_QUARTER = 2;
constexpr int64_t ROTARY_MODE_INTERLEAVE_HALF = 3;

// ---- InterleaveModeGradVF ----
// interleave mode (rotary_mode=1):
//   dx[2k]   = cos[2k]*dy[2k]   + sin[2k+1]*dy[2k+1]
//   dx[2k+1] = cos[2k+1]*dy[2k+1] - sin[2k]*dy[2k]
// Matches reference in rotary_position_embedding_grad.
template <typename TDY, typename TCOS>
__aicore__ inline void InterleaveModeGradVF(const LocalTensor<TCOS> &sinTensor, const LocalTensor<TCOS> &cosTensor,
                                            const LocalTensor<TDY> &inTensor, const LocalTensor<TDY> &outTensor,
                                            uint16_t dLen, uint16_t dSplitCoef_, uint16_t currSNum, uint16_t currNNum)
{
    __ubuf__ TCOS *sinUb = (__ubuf__ TCOS *)sinTensor.GetPhyAddr();
    __ubuf__ TCOS *cosUb = (__ubuf__ TCOS *)cosTensor.GetPhyAddr();
    __ubuf__ TDY *inUb = (__ubuf__ TDY *)inTensor.GetPhyAddr();
    __ubuf__ TDY *outUb = (__ubuf__ TDY *)outTensor.GetPhyAddr();
    uint16_t loopSize = 2 * VL_FLOAT32_SIZE;
    uint16_t loopNum = (dLen + loopSize - 1) / (2 * VL_FLOAT32_SIZE);
    uint16_t dAlignLenDy = Ops::Base::CeilAlign(static_cast<uint16_t>(dLen / dSplitCoef_),
                                                static_cast<uint16_t>(BLOCK_TYPE_SIZE / sizeof(TDY))) *
                           dSplitCoef_;
    uint16_t dAlignLenCos = Ops::Base::CeilAlign(static_cast<uint16_t>(dLen / dSplitCoef_),
                                                 static_cast<uint16_t>(BLOCK_TYPE_SIZE / sizeof(TCOS))) *
                            dSplitCoef_;

    uint32_t halfNum = dLen / 2;
    uint32_t part1Num = (loopNum - 1) * VL_FLOAT32_SIZE;
    uint32_t part2Num = part1Num;
    uint32_t tailNum = dLen - part1Num - part2Num;
    if (tailNum > VL_FLOAT32_SIZE) {
        part1Num += VL_FLOAT32_SIZE;
        part2Num += (tailNum - VL_FLOAT32_SIZE);
    } else {
        part1Num += tailNum;
    }

    __ubuf__ TDY *currInUb, *currOutUb;
    __ubuf__ TCOS *currSinUb, *currCosUb;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregFormerCos, vregLatterCos, vregFormerSin, vregLatterSin, vregFormerIn,
            vregLatterIn;
        MicroAPI::MaskReg pregLoop, pregPart1, pregPart2;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlignLenCos;
            currCosUb = cosUb + sIdx * dAlignLenCos;
            for (uint16_t idxD = 0; idxD < currNNum; idxD++) {
                uint32_t halfCnt = halfNum;
                uint32_t part1Cnt = part1Num;
                uint32_t part2Cnt = part2Num;
                currInUb = inUb + (sIdx * currNNum + idxD) * dAlignLenDy;
                currOutUb = outUb + (sIdx * currNNum + idxD) * dAlignLenDy;
                for (uint16_t i = 0; i < loopNum; i++) {
                    pregLoop = MicroAPI::UpdateMask<float>(halfCnt);
                    pregPart1 = MicroAPI::UpdateMask<float>(part1Cnt);
                    pregPart2 = MicroAPI::UpdateMask<float>(part2Cnt);
                    int32_t evenOffSet = (i * 2) * VL_FLOAT32_SIZE;
                    int32_t oddOffset = evenOffSet + VL_FLOAT32_SIZE;
                    ops::LoadOneTensorForDtypeT<TDY>(currInUb, vregFormerIn, pregPart1, evenOffSet);
                    ops::LoadOneTensorForDtypeT<TDY>(currInUb, vregLatterIn, pregPart2, oddOffset);
                    ops::LoadOneTensorForDtypeT<TCOS>(currCosUb, vregFormerCos, pregPart1, evenOffSet);
                    ops::LoadOneTensorForDtypeT<TCOS>(currCosUb, vregLatterCos, pregPart2, oddOffset);
                    ops::LoadOneTensorForDtypeT<TCOS>(currSinUb, vregFormerSin, pregPart1, evenOffSet);
                    ops::LoadOneTensorForDtypeT<TCOS>(currSinUb, vregLatterSin, pregPart2, oddOffset);
                    Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregPart1);
                    Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregPart2);
                    Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregPart1);
                    Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregPart2);
                    MicroAPI::DeInterleave<float>(vregFormerSin, vregLatterSin, vregFormerSin, vregLatterSin);
                    Muls(vregFormerSin, vregFormerSin, float(-1.0), pregLoop);
                    MicroAPI::Interleave<float>(vregFormerSin, vregLatterSin, vregLatterSin, vregFormerSin);
                    Add(vregFormerCos, vregFormerSin, vregFormerCos, pregPart1);
                    Add(vregLatterCos, vregLatterSin, vregLatterCos, pregPart2);
                    ops::StoreOneTensorForDtypeT<TDY>(currOutUb, vregFormerCos, pregPart1, evenOffSet);
                    ops::StoreOneTensorForDtypeT<TDY>(currOutUb, vregLatterCos, pregPart2, oddOffset);
                }
            }
        }
    }
}

// ---- BatchInterleaveModeGradVF ----
// Batch version supporting mixed precision (TDY for dy/dx, TCOS for cos/sin).
// UB layout: in/out are [B][N][S][dAlignDy], cos/sin are [B or 1][S][dAlignCos].
template <typename TDY, typename TCOS, bool IsBroadCast>
__aicore__ inline void BatchInterleaveModeGradVF(__ubuf__ TDY *in, __ubuf__ TCOS *cos, __ubuf__ TCOS *sin,
                                                 __ubuf__ TDY *out, uint16_t sLength, uint16_t bLength,
                                                 uint16_t nLength, int64_t dLen, int64_t dAlignDy, int64_t dAlignCos,
                                                 int64_t ubFactorS, int64_t ubFactorN)
{
    uint32_t loopSize = 2 * VL_FLOAT32_SIZE;
    uint16_t dLoopCount = (dLen + loopSize - 1) / loopSize;

    // Mask parameters
    uint32_t halfNum = dLen / 2;
    uint32_t part1Num = (dLoopCount - 1) * VL_FLOAT32_SIZE;
    uint32_t part2Num = part1Num;
    uint32_t tailNum = dLen - part1Num - part2Num;
    if (tailNum > VL_FLOAT32_SIZE) {
        part1Num += VL_FLOAT32_SIZE;
        part2Num += (tailNum - VL_FLOAT32_SIZE);
    } else {
        part1Num += tailNum;
    }

    // UB stride parameters
    int32_t bStepUb = ubFactorN * ubFactorS * dAlignDy; // in/out B stride
    int32_t nStepUb = ubFactorS * dAlignDy;             // in/out N stride
    int32_t cosBStepUb = ubFactorS * dAlignCos;         // cos/sin B stride
    int32_t cosSStepUb = dAlignCos;                     // cos/sin S stride

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inPart1Reg, inPart2Reg;
        MicroAPI::RegTensor<float> cosPart1Reg, cosPart2Reg;
        MicroAPI::RegTensor<float> sinPart1Reg, sinPart2Reg;
        MicroAPI::MaskReg pregLoop, pregPart1, pregPart2;
        __ubuf__ TDY *currInUb, *currOutUb;
        __ubuf__ TCOS *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t halfCnt = halfNum;
                    uint32_t part1Cnt = part1Num;
                    uint32_t part2Cnt = part2Num;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlignDy;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlignDy;
                    if constexpr (IsBroadCast) {
                        currCosUb = cos + sIdx * cosSStepUb;
                        currSinUb = sin + sIdx * cosSStepUb;
                    } else {
                        currCosUb = cos + bIdx * cosBStepUb + sIdx * cosSStepUb;
                        currSinUb = sin + bIdx * cosBStepUb + sIdx * cosSStepUb;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(halfCnt);
                        pregPart1 = MicroAPI::UpdateMask<float>(part1Cnt);
                        pregPart2 = MicroAPI::UpdateMask<float>(part2Cnt);
                        ops::LoadOneTensorForDtypeT<TDY>(currInUb, inPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<TDY>(currInUb, inPart2Reg, pregPart2,
                                                         i * loopSize + VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<TCOS>(currCosUb, cosPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<TCOS>(currCosUb, cosPart2Reg, pregPart2,
                                                          i * loopSize + VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<TCOS>(currSinUb, sinPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<TCOS>(currSinUb, sinPart2Reg, pregPart2,
                                                          i * loopSize + VL_FLOAT32_SIZE);
                        Mul(cosPart1Reg, cosPart1Reg, inPart1Reg, pregPart1);
                        Mul(cosPart2Reg, cosPart2Reg, inPart2Reg, pregPart2);
                        Mul(sinPart1Reg, sinPart1Reg, inPart1Reg, pregPart1);
                        Mul(sinPart2Reg, sinPart2Reg, inPart2Reg, pregPart2);
                        MicroAPI::DeInterleave<float>(sinPart1Reg, sinPart2Reg, sinPart1Reg, sinPart2Reg);
                        Muls(sinPart1Reg, sinPart1Reg, float(-1.0), pregLoop);
                        MicroAPI::Interleave<float>(sinPart1Reg, sinPart2Reg, sinPart2Reg, sinPart1Reg);
                        Add(cosPart1Reg, sinPart1Reg, cosPart1Reg, pregPart1);
                        Add(cosPart2Reg, sinPart2Reg, cosPart2Reg, pregPart2);
                        ops::StoreOneTensorForDtypeT<TDY>(currOutUb, cosPart1Reg, pregPart1, i * loopSize);
                        ops::StoreOneTensorForDtypeT<TDY>(currOutUb, cosPart2Reg, pregPart2,
                                                          i * loopSize + VL_FLOAT32_SIZE);
                    }
                }
            }
        }
    }
}

} // namespace InplacePartialRotaryMulGrad
#endif
