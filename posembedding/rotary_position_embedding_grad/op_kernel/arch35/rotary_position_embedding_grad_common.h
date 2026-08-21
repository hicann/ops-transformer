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
 * \file rotary_position_embedding_grad_common.h
 * \brief
 */
#ifndef ROTARY_POSITION_EMBEDDING_COMMON_H
#define ROTARY_POSITION_EMBEDDING_COMMON_H

#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"
#if __has_include("../../apply_rotary_pos_emb/arch35/apply_rotary_pos_emb_common.h")
#include "../../apply_rotary_pos_emb/arch35/apply_rotary_pos_emb_common.h"
#else
#include "../../apply_rotary_pos_emb/op_kernel/arch35/apply_rotary_pos_emb_common.h"
#endif

using namespace AscendC;

namespace RotaryPositionEmbeddingGrad {
/*
    qOut[0] = q[0] * cos[0] + q[1] * sin[1]
    qOut[1] = q[1] * cos[1] - q[0] * sin[0]
*/
template <typename T>
__aicore__ inline void HalfGradAlignVF(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                       const LocalTensor<T> &inTensor, const LocalTensor<T> &outTensor, uint32_t dLen,
                                       uint32_t dAlign, uint16_t currSNum, uint16_t currDNum)
{
    __ubuf__ T *sinUb = (__ubuf__ T *)sinTensor.GetPhyAddr();
    __ubuf__ T *cosUb = (__ubuf__ T *)cosTensor.GetPhyAddr();
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ T *outUb = (__ubuf__ T *)outTensor.GetPhyAddr();
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint32_t halfDAlign = Ops::Base::CeilAlign(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t repeatTimes = Ops::Base::CeilDiv(halfD, VL_FLOAT32_SIZE);
    __ubuf__ T *currInUb, *currOutUb, *currSinUb, *currCosUb;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregIn, vregHalfIn, vregSin, vregHalfSin, vregCos, vregHalfCos, vregOut, vregHalfOut;
        MicroAPI::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlign;
            currCosUb = cosUb + sIdx * dAlign;
            for (uint16_t row = 0; row < currDNum; row++) {
                currInUb = inUb + (sIdx * currDNum + row) * dAlign;
                currOutUb = outUb + (sIdx * currDNum + row) * dAlign;
                uint32_t updateCnt = halfD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = MicroAPI::UpdateMask<float>(updateCnt);
                    int32_t offset = i * VL_FLOAT32_SIZE;
                    int32_t halfOffset = offset + halfDAlign;
                    ops::LoadTwoTensorForDtypeT<T>(currInUb, currInUb, vregIn, vregHalfIn, preg, preg, offset,
                                                   halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currSinUb, currSinUb, vregSin, vregHalfSin, preg, preg, offset,
                                                   halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currCosUb, currCosUb, vregCos, vregHalfCos, preg, preg, offset,
                                                   halfOffset);

                    Mul(vregOut, vregCos, vregIn, preg);
                    Mul(vregHalfSin, vregHalfSin, vregHalfIn, preg);
                    Add(vregOut, vregOut, vregHalfSin, preg);
                    Mul(vregSin, vregSin, vregIn, preg);
                    Mul(vregHalfCos, vregHalfCos, vregHalfIn, preg);
                    Sub(vregHalfOut, vregHalfCos, vregSin, preg);

                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregOut, preg, offset);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregHalfOut, preg, halfOffset);
                }
            }
        }
    }
}

/*
    qOut[0] = q[0] * cos[0] + q[1] * sin[1]
    qOut[1] = q[1] * cos[1] - q[0] * sin[0]
    qOut[2] = q[2] * cos[2] + q[3] * sin[3]
    qOut[3] = q[3] * cos[3] - q[2] * sin[2]
*/
template <typename T>
__aicore__ inline void QuarterGradAlignVF(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                          const LocalTensor<T> &inTensor, const LocalTensor<T> &outTensor,
                                          uint32_t dLen, uint32_t dAlign, uint16_t currSNum, uint16_t currDNum)
{
    __ubuf__ T *sinUb = (__ubuf__ T *)sinTensor.GetPhyAddr();
    __ubuf__ T *cosUb = (__ubuf__ T *)cosTensor.GetPhyAddr();
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ T *outUb = (__ubuf__ T *)outTensor.GetPhyAddr();
    uint32_t quarterD = dLen / QUARTER_MODE_COEF;
    uint32_t quarterDAlign = Ops::Base::CeilAlign(quarterD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t repeatTimes = Ops::Base::CeilDiv(quarterD, VL_FLOAT32_SIZE);
    __ubuf__ T *currInUb, *currOutUb, *currSinUb, *currCosUb;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregIn, vregQ1In, vregQ2In, vregQ3In, vregSin, vregQ1Sin, vregQ2Sin, vregQ3Sin,
            vregCos, vregQ1Cos, vregQ2Cos, vregQ3Cos, vregOut, vregQ1Out, vregQ2Out, vregQ3Out;
        MicroAPI::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlign;
            currCosUb = cosUb + sIdx * dAlign;
            for (uint16_t row = 0; row < currDNum; row++) {
                currInUb = inUb + (sIdx * currDNum + row) * dAlign;
                currOutUb = outUb + (sIdx * currDNum + row) * dAlign;
                uint32_t updateCnt = quarterD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = MicroAPI::UpdateMask<float>(updateCnt);
                    int32_t offset = i * VL_FLOAT32_SIZE;
                    int32_t q1Offset = offset + quarterDAlign;
                    int32_t q2Offset = q1Offset + quarterDAlign;
                    int32_t q3Offset = q2Offset + quarterDAlign;
                    ops::LoadTwoTensorForDtypeT<T>(currInUb, currInUb, vregIn, vregQ1In, preg, preg, offset, q1Offset);
                    ops::LoadTwoTensorForDtypeT<T>(currInUb, currInUb, vregQ2In, vregQ3In, preg, preg, q2Offset,
                                                   q3Offset);
                    ops::LoadTwoTensorForDtypeT<T>(currSinUb, currSinUb, vregSin, vregQ1Sin, preg, preg, offset,
                                                   q1Offset);
                    ops::LoadTwoTensorForDtypeT<T>(currSinUb, currSinUb, vregQ2Sin, vregQ3Sin, preg, preg, q2Offset,
                                                   q3Offset);
                    ops::LoadTwoTensorForDtypeT<T>(currCosUb, currCosUb, vregCos, vregQ1Cos, preg, preg, offset,
                                                   q1Offset);
                    ops::LoadTwoTensorForDtypeT<T>(currCosUb, currCosUb, vregQ2Cos, vregQ3Cos, preg, preg, q2Offset,
                                                   q3Offset);

                    Mul(vregOut, vregCos, vregIn, preg);
                    Mul(vregQ1Sin, vregQ1Sin, vregQ1In, preg);
                    Add(vregOut, vregOut, vregQ1Sin, preg);
                    Mul(vregSin, vregSin, vregIn, preg);
                    Mul(vregQ1Cos, vregQ1Cos, vregQ1In, preg);
                    Sub(vregQ1Out, vregQ1Cos, vregSin, preg);
                    Mul(vregQ2Out, vregQ2Cos, vregQ2In, preg);
                    Mul(vregQ3Sin, vregQ3Sin, vregQ3In, preg);
                    Add(vregQ2Out, vregQ2Out, vregQ3Sin, preg);
                    Mul(vregQ2Sin, vregQ2Sin, vregQ2In, preg);
                    Mul(vregQ3Cos, vregQ3Cos, vregQ3In, preg);
                    Sub(vregQ3Out, vregQ3Cos, vregQ2Sin, preg);

                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregOut, preg, offset);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregQ1Out, preg, q1Offset);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregQ2Out, preg, q2Offset);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregQ3Out, preg, q3Offset);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void InterleaveModeGradVF(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                            const LocalTensor<T> &inTensor, const LocalTensor<T> &outTensor,
                                            uint16_t dLen, uint16_t currSNum, uint16_t currDNum)
{
    __ubuf__ T *sinUb = (__ubuf__ T *)sinTensor.GetPhyAddr();
    __ubuf__ T *cosUb = (__ubuf__ T *)cosTensor.GetPhyAddr();
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ T *outUb = (__ubuf__ T *)outTensor.GetPhyAddr();
    uint16_t loopSize = 2 * VL_FLOAT32_SIZE;
    uint16_t loopNum = (dLen + loopSize - 1) / (2 * VL_FLOAT32_SIZE);
    uint16_t dAlignLen = Ops::Base::CeilAlign(dLen, static_cast<uint16_t>(BLOCK_TYPE_SIZE / sizeof(T)));

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

    __ubuf__ T *currInUb, *currOutUb, *currSinUb, *currCosUb, *tailSinUb, *tailCosUb;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregFormerCos, vregLatterCos, vregFormerSin, vregLatterSin, vregFormerIn,
            vregLatterIn, vregOdd, vregEven;
        MicroAPI::MaskReg pregLoop, pregPart1, pregPart2;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlignLen;
            currCosUb = cosUb + sIdx * dAlignLen;
            for (uint16_t idxD = 0; idxD < currDNum; idxD++) {
                uint32_t halfCnt = halfNum;
                uint32_t part1Cnt = part1Num;
                uint32_t part2Cnt = part2Num;
                currInUb = inUb + (sIdx * currDNum + idxD) * dAlignLen;
                currOutUb = outUb + (sIdx * currDNum + idxD) * dAlignLen;
                for (uint16_t i = 0; i < loopNum; i++) {
                    pregLoop = MicroAPI::UpdateMask<float>(halfCnt);
                    pregPart1 = MicroAPI::UpdateMask<float>(part1Cnt);
                    pregPart2 = MicroAPI::UpdateMask<float>(part2Cnt);
                    int32_t evenOffSet = (i * 2) * VL_FLOAT32_SIZE;
                    int32_t oddOffset = evenOffSet + VL_FLOAT32_SIZE;
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregFormerIn, pregPart1, evenOffSet);
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregLatterIn, pregPart2, oddOffset);
                    ops::LoadOneTensorForDtypeT<T>(currCosUb, vregFormerCos, pregPart1, evenOffSet);
                    ops::LoadOneTensorForDtypeT<T>(currCosUb, vregLatterCos, pregPart2, oddOffset);
                    ops::LoadOneTensorForDtypeT<T>(currSinUb, vregFormerSin, pregPart1, evenOffSet);
                    ops::LoadOneTensorForDtypeT<T>(currSinUb, vregLatterSin, pregPart2, oddOffset);
                    Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregPart1);
                    Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregPart2);
                    Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregPart1);
                    Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregPart2);
                    MicroAPI::DeInterleave<float>(vregFormerSin, vregLatterSin, vregFormerSin, vregLatterSin);
                    Muls(vregFormerSin, vregFormerSin, float(-1.0), pregLoop);
                    MicroAPI::Interleave<float>(vregFormerSin, vregLatterSin, vregLatterSin, vregFormerSin);
                    Add(vregFormerCos, vregFormerSin, vregFormerCos, pregPart1);
                    Add(vregLatterCos, vregLatterSin, vregLatterCos, pregPart2);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregFormerCos, pregPart1, evenOffSet);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregLatterCos, pregPart2, oddOffset);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void DeepSeekInterleaveModeGradVF(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                                    const LocalTensor<T> &inTensor, const LocalTensor<T> &outTensor,
                                                    uint32_t dLen, uint16_t currSNum, uint16_t currDNum)
{
    __ubuf__ T *sinUb = (__ubuf__ T *)sinTensor.GetPhyAddr();
    __ubuf__ T *cosUb = (__ubuf__ T *)cosTensor.GetPhyAddr();
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ T *outUb = (__ubuf__ T *)outTensor.GetPhyAddr();
    uint32_t dAlign = Ops::Base::CeilAlign(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint32_t loopSize = 2 * VL_FLOAT32_SIZE;
    uint16_t repeatTimes = (dLen + loopSize - 1) / loopSize;

    // 计算Mask参数
    uint32_t halfD = dLen / 2;
    uint32_t halfDAlign = Ops::Base::CeilAlign(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint32_t part1Num = (repeatTimes - 1) * VL_FLOAT32_SIZE;
    uint32_t part2Num = part1Num;
    uint32_t tailNum = dLen - part1Num - part2Num;
    if (tailNum > VL_FLOAT32_SIZE) {
        part1Num += VL_FLOAT32_SIZE;
        part2Num += (tailNum - VL_FLOAT32_SIZE);
    } else {
        part1Num += tailNum;
    }

    __ubuf__ T *currInUb, *currOutUb, *currSinUb, *currCosUb;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregIn, vregHalfIn, vregSin, vregHalfSin, vregCos, vregHalfCos, vregOut, vregHalfOut;
        MicroAPI::MaskReg pregLoop, pregPart1, pregPart2;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * halfDAlign * HALF_INTERLEAVE_COEF;
            currCosUb = cosUb + sIdx * halfDAlign * HALF_INTERLEAVE_COEF;
            for (uint16_t row = 0; row < currDNum; row++) {
                currInUb = inUb + (sIdx * currDNum + row) * halfDAlign * HALF_INTERLEAVE_COEF;
                currOutUb = outUb + (sIdx * currDNum + row) * dAlign;
                uint32_t halfCnt = halfD;
                uint32_t part1Cnt = part1Num;
                uint32_t part2Cnt = part2Num;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    pregLoop = MicroAPI::UpdateMask<float>(halfCnt);
                    pregPart1 = MicroAPI::UpdateMask<float>(part1Cnt);
                    pregPart2 = MicroAPI::UpdateMask<float>(part2Cnt);
                    int32_t offset = i * VL_FLOAT32_SIZE;
                    int32_t halfOffset = offset + halfDAlign;
                    int32_t outOffset = offset * HALF_INTERLEAVE_COEF;
                    ops::LoadTwoTensorForDtypeT<T>(currInUb, currInUb, vregIn, vregHalfIn, pregLoop, pregLoop, offset,
                                                   halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currSinUb, currSinUb, vregSin, vregHalfSin, pregLoop, pregLoop,
                                                   offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currCosUb, currCosUb, vregCos, vregHalfCos, pregLoop, pregLoop,
                                                   offset, halfOffset);
                    Mul(vregCos, vregIn, vregCos, pregLoop);
                    Mul(vregHalfSin, vregHalfSin, vregHalfIn, pregLoop);
                    Add(vregCos, vregCos, vregHalfSin, pregLoop);
                    Mul(vregHalfCos, vregHalfIn, vregHalfCos, pregLoop);
                    Mul(vregSin, vregIn, vregSin, pregLoop);
                    Sub(vregHalfCos, vregHalfCos, vregSin, pregLoop);
                    MicroAPI::Interleave<float>(vregOut, vregHalfOut, vregCos, vregHalfCos);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregOut, pregPart1, outOffset);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregHalfOut, pregPart2, outOffset + VL_FLOAT32_SIZE);
                }
            }
        }
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void BatchHalfGradAlignVF(__ubuf__ T *in, __ubuf__ T *cos, __ubuf__ T *sin, __ubuf__ T *out,
                                            uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t d,
                                            int64_t dAlign, int64_t ubFactorS, int64_t ubFactorN)
{
    uint32_t dHalfSize = d / HALF_INTERLEAVE_COEF;
    uint16_t dLoopCount = (dHalfSize + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    uint32_t dHalfOffset = dAlign / HALF_INTERLEAVE_COEF;

    // 计算循环参数
    int32_t bStepUb = ubFactorN * ubFactorS * dAlign;
    int32_t nStepUb = ubFactorS * dAlign;

    __VEC_SCOPE__
    {
        // 定义相关寄存器
        MicroAPI::RegTensor<float> inPart1Reg, inPart2Reg, cosPart1Reg, cosPart2Reg, sinPart1Reg, sinPart2Reg;
        MicroAPI::MaskReg pregLoop;
        __ubuf__ T *currInUb, *currOutUb, *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t count = dHalfSize;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * dAlign;
                        currSinUb = sin + sIdx * dAlign;
                    } else {
                        currCosUb = cos + bIdx * nStepUb + sIdx * dAlign;
                        currSinUb = sin + bIdx * nStepUb + sIdx * dAlign;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(count);
                        // 拷贝到RegBase内
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart2Reg, pregLoop,
                                                       i * VL_FLOAT32_SIZE + dHalfOffset);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart2Reg, pregLoop,
                                                       i * VL_FLOAT32_SIZE + dHalfOffset);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart2Reg, pregLoop,
                                                       i * VL_FLOAT32_SIZE + dHalfOffset);
                        // 计算
                        Mul(cosPart1Reg, inPart1Reg, cosPart1Reg, pregLoop);
                        Mul(sinPart2Reg, sinPart2Reg, inPart2Reg, pregLoop);
                        Add(cosPart1Reg, cosPart1Reg, sinPart2Reg, pregLoop);
                        Mul(cosPart2Reg, inPart2Reg, cosPart2Reg, pregLoop);
                        Mul(sinPart1Reg, inPart1Reg, sinPart1Reg, pregLoop);
                        Sub(cosPart2Reg, cosPart2Reg, sinPart1Reg, pregLoop);
                        // 拷贝回UB
                        ops::StoreOneTensorForDtypeT<T>(currOutUb, cosPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::StoreOneTensorForDtypeT<T>(currOutUb, cosPart2Reg, pregLoop,
                                                        i * VL_FLOAT32_SIZE + dHalfOffset);
                    }
                }
            }
        }
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void BatchQuarterGradAlignVF(__ubuf__ T *in, __ubuf__ T *cos, __ubuf__ T *sin, __ubuf__ T *out,
                                               uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t d,
                                               int64_t dAlign, int64_t ubFactorS, int64_t ubFactorN)
{
    uint32_t dQuarterSize = d / QUARTER_MODE_COEF;
    uint16_t dLoopCount = (dQuarterSize + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    uint32_t dQuarterOffset = dAlign / QUARTER_MODE_COEF;
    uint32_t dHalfOffset = dAlign / HALF_INTERLEAVE_COEF;
    uint32_t dThreeQuarterOffset = dQuarterOffset + dHalfOffset;

    // 计算循环参数
    int32_t bStepUb = ubFactorN * ubFactorS * dAlign;
    int32_t nStepUb = ubFactorS * dAlign;

    __VEC_SCOPE__
    {
        // 定义相关寄存器
        MicroAPI::RegTensor<float> inPart1Reg, inPart2Reg, inPart3Reg, inPart4Reg, cosPart1Reg, cosPart2Reg,
            cosPart3Reg, cosPart4Reg, sinPart1Reg, sinPart2Reg, sinPart3Reg, sinPart4Reg;
        MicroAPI::MaskReg pregLoop;
        __ubuf__ T *currInUb, *currOutUb, *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t count = dQuarterSize;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * dAlign;
                        currSinUb = sin + sIdx * dAlign;
                    } else {
                        currCosUb = cos + bIdx * nStepUb + sIdx * dAlign;
                        currSinUb = sin + bIdx * nStepUb + sIdx * dAlign;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(count);
                        // 拷贝到RegBase内
                        ops::LoadTwoTensorForDtypeT<T>(currInUb, currInUb, inPart1Reg, inPart2Reg, pregLoop, pregLoop,
                                                       i * VL_FLOAT32_SIZE, i * VL_FLOAT32_SIZE + dQuarterOffset);
                        ops::LoadTwoTensorForDtypeT<T>(currInUb, currInUb, inPart3Reg, inPart4Reg, pregLoop, pregLoop,
                                                       i * VL_FLOAT32_SIZE + dHalfOffset,
                                                       i * VL_FLOAT32_SIZE + dThreeQuarterOffset);
                        ops::LoadTwoTensorForDtypeT<T>(currCosUb, currCosUb, cosPart1Reg, cosPart2Reg, pregLoop,
                                                       pregLoop, i * VL_FLOAT32_SIZE,
                                                       i * VL_FLOAT32_SIZE + dQuarterOffset);
                        ops::LoadTwoTensorForDtypeT<T>(currCosUb, currCosUb, cosPart3Reg, cosPart4Reg, pregLoop,
                                                       pregLoop, i * VL_FLOAT32_SIZE + dHalfOffset,
                                                       i * VL_FLOAT32_SIZE + dThreeQuarterOffset);
                        ops::LoadTwoTensorForDtypeT<T>(currSinUb, currSinUb, sinPart1Reg, sinPart2Reg, pregLoop,
                                                       pregLoop, i * VL_FLOAT32_SIZE,
                                                       i * VL_FLOAT32_SIZE + dQuarterOffset);
                        ops::LoadTwoTensorForDtypeT<T>(currSinUb, currSinUb, sinPart3Reg, sinPart4Reg, pregLoop,
                                                       pregLoop, i * VL_FLOAT32_SIZE + dHalfOffset,
                                                       i * VL_FLOAT32_SIZE + dThreeQuarterOffset);
                        // 计算
                        Mul(cosPart1Reg, inPart1Reg, cosPart1Reg, pregLoop);
                        Mul(sinPart2Reg, sinPart2Reg, inPart2Reg, pregLoop);
                        Add(cosPart1Reg, cosPart1Reg, sinPart2Reg, pregLoop);
                        Mul(cosPart2Reg, inPart2Reg, cosPart2Reg, pregLoop);
                        Mul(sinPart1Reg, inPart1Reg, sinPart1Reg, pregLoop);
                        Sub(cosPart2Reg, cosPart2Reg, sinPart1Reg, pregLoop);
                        Mul(cosPart3Reg, inPart3Reg, cosPart3Reg, pregLoop);
                        Mul(sinPart4Reg, sinPart4Reg, inPart4Reg, pregLoop);
                        Add(cosPart3Reg, cosPart3Reg, sinPart4Reg, pregLoop);
                        Mul(cosPart4Reg, inPart4Reg, cosPart4Reg, pregLoop);
                        Mul(sinPart3Reg, inPart3Reg, sinPart3Reg, pregLoop);
                        Sub(cosPart4Reg, cosPart4Reg, sinPart3Reg, pregLoop);
                        // 拷贝回UB
                        ops::StoreOneTensorForDtypeT<T>(currOutUb, cosPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::StoreOneTensorForDtypeT<T>(currOutUb, cosPart2Reg, pregLoop,
                                                        i * VL_FLOAT32_SIZE + dQuarterOffset);
                        ops::StoreOneTensorForDtypeT<T>(currOutUb, cosPart3Reg, pregLoop,
                                                        i * VL_FLOAT32_SIZE + dHalfOffset);
                        ops::StoreOneTensorForDtypeT<T>(currOutUb, cosPart4Reg, pregLoop,
                                                        i * VL_FLOAT32_SIZE + dThreeQuarterOffset);
                    }
                }
            }
        }
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void BatchInterleaveModeGradVF(__ubuf__ T *in, __ubuf__ T *cos, __ubuf__ T *sin, __ubuf__ T *out,
                                                 uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t d,
                                                 int64_t dAlign, int64_t ubFactorS, int64_t ubFactorN)
{
    uint32_t loopSize = 2 * VL_FLOAT32_SIZE;
    uint16_t dLoopCount = (d + loopSize - 1) / loopSize;

    // 计算Mask参数
    uint32_t halfNum = d / 2;
    uint32_t part1Num = (dLoopCount - 1) * VL_FLOAT32_SIZE;
    uint32_t part2Num = part1Num;
    uint32_t tailNum = d - part1Num - part2Num;
    if (tailNum > VL_FLOAT32_SIZE) {
        part1Num += VL_FLOAT32_SIZE;
        part2Num += (tailNum - VL_FLOAT32_SIZE);
    } else {
        part1Num += tailNum;
    }

    // 计算循环参数
    int32_t bStepUb = ubFactorN * ubFactorS * dAlign;
    int32_t nStepUb = ubFactorS * dAlign;

    __VEC_SCOPE__
    {
        // 定义相关寄存器
        MicroAPI::RegTensor<float> inPart1Reg;
        MicroAPI::RegTensor<float> inPart2Reg;
        MicroAPI::RegTensor<float> cosPart1Reg;
        MicroAPI::RegTensor<float> cosPart2Reg;
        MicroAPI::RegTensor<float> sinPart1Reg;
        MicroAPI::RegTensor<float> sinPart2Reg;
        MicroAPI::MaskReg pregLoop;
        MicroAPI::MaskReg pregPart1;
        MicroAPI::MaskReg pregPart2;
        __ubuf__ T *currInUb, *currOutUb, *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t halfCnt = halfNum;
                    uint32_t part1Cnt = part1Num;
                    uint32_t part2Cnt = part2Num;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * dAlign;
                        currSinUb = sin + sIdx * dAlign;
                    } else {
                        currCosUb = cos + bIdx * nStepUb + sIdx * dAlign;
                        currSinUb = sin + bIdx * nStepUb + sIdx * dAlign;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(halfCnt);
                        pregPart1 = MicroAPI::UpdateMask<float>(part1Cnt);
                        pregPart2 = MicroAPI::UpdateMask<float>(part2Cnt);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart2Reg, pregPart2, i * loopSize + VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart2Reg, pregPart2,
                                                       i * loopSize + VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart2Reg, pregPart2,
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
                        ops::StoreOneTensorForDtypeT<T>(currOutUb, cosPart1Reg, pregPart1, i * loopSize);
                        ops::StoreOneTensorForDtypeT<T>(currOutUb, cosPart2Reg, pregPart2,
                                                        i * loopSize + VL_FLOAT32_SIZE);
                    }
                }
            }
        }
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void BatchDeepSeekInterleaveModeGradVF(__ubuf__ T *in, __ubuf__ T *cos, __ubuf__ T *sin,
                                                         __ubuf__ T *out, uint16_t sLength, uint16_t bLength,
                                                         uint16_t nLength, int64_t d, int64_t dAlign, int64_t ubFactorS,
                                                         int64_t ubFactorN)
{
    uint32_t loopSize = 2 * VL_FLOAT32_SIZE;
    uint16_t dLoopCount = (d + loopSize - 1) / loopSize;
    uint32_t dHalfOffset = dAlign / HALF_INTERLEAVE_COEF;

    // 计算Mask参数
    uint32_t halfNum = d / 2;
    uint32_t part1Num = (dLoopCount - 1) * VL_FLOAT32_SIZE;
    uint32_t part2Num = part1Num;
    uint32_t tailNum = d - part1Num - part2Num;
    if (tailNum > VL_FLOAT32_SIZE) {
        part1Num += VL_FLOAT32_SIZE;
        part2Num += (tailNum - VL_FLOAT32_SIZE);
    } else {
        part1Num += tailNum;
    }

    // 计算循环参数
    int32_t bStepUb = ubFactorN * ubFactorS * dAlign;
    int32_t nStepUb = ubFactorS * dAlign;

    __VEC_SCOPE__
    {
        // 定义相关寄存器
        MicroAPI::RegTensor<float> inPart1Reg;
        MicroAPI::RegTensor<float> inPart2Reg;
        MicroAPI::RegTensor<float> cosPart1Reg;
        MicroAPI::RegTensor<float> cosPart2Reg;
        MicroAPI::RegTensor<float> sinPart1Reg;
        MicroAPI::RegTensor<float> sinPart2Reg;
        MicroAPI::MaskReg pregLoop;
        MicroAPI::MaskReg pregPart1;
        MicroAPI::MaskReg pregPart2;
        __ubuf__ T *currInUb, *currOutUb, *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t halfCnt = halfNum;
                    uint32_t part1Cnt = part1Num;
                    uint32_t part2Cnt = part2Num;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * dAlign;
                        currSinUb = sin + sIdx * dAlign;
                    } else {
                        currCosUb = cos + bIdx * nStepUb + sIdx * dAlign;
                        currSinUb = sin + bIdx * nStepUb + sIdx * dAlign;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(halfCnt);
                        pregPart1 = MicroAPI::UpdateMask<float>(part1Cnt);
                        pregPart2 = MicroAPI::UpdateMask<float>(part2Cnt);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart2Reg, pregLoop,
                                                       i * VL_FLOAT32_SIZE + dHalfOffset);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart2Reg, pregLoop,
                                                       i * VL_FLOAT32_SIZE + dHalfOffset);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart2Reg, pregLoop,
                                                       i * VL_FLOAT32_SIZE + dHalfOffset);
                        Mul(cosPart1Reg, inPart1Reg, cosPart1Reg, pregLoop);
                        Mul(sinPart2Reg, sinPart2Reg, inPart2Reg, pregLoop);
                        Add(cosPart1Reg, cosPart1Reg, sinPart2Reg, pregLoop);
                        Mul(cosPart2Reg, inPart2Reg, cosPart2Reg, pregLoop);
                        Mul(sinPart1Reg, inPart1Reg, sinPart1Reg, pregLoop);
                        Sub(cosPart2Reg, cosPart2Reg, sinPart1Reg, pregLoop);
                        MicroAPI::Interleave<float>(cosPart1Reg, cosPart2Reg, cosPart1Reg, cosPart2Reg);
                        ops::StoreOneTensorForDtypeT<T>(currOutUb, cosPart1Reg, pregPart1, i * loopSize);
                        ops::StoreOneTensorForDtypeT<T>(currOutUb, cosPart2Reg, pregPart2,
                                                        i * loopSize + VL_FLOAT32_SIZE);
                    }
                }
            }
        }
    }
}

} // namespace RotaryPositionEmbeddingGrad

#endif // ROTARY_POSITION_EMBEDDING_COMMON_H
