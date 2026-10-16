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
 * \file partial_rotary_mul_quant_hif4_common.h
 * \brief  All data (in/cos/sin) use type T, output is hifloat8 (uint8_t* + dAlignH8Out stride).
 *         VF handles T->f32 and f32->hifloat8 internally.
 */
#ifndef PARTIAL_ROTARY_MUL_QUANT_HIF4_COMMON_H
#define PARTIAL_ROTARY_MUL_QUANT_HIF4_COMMON_H

#include "partial_rotary_mul_quant_common.h"

using namespace AscendC;
using namespace MicroAPI;

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if defined(__DAV_C310__)
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

constexpr uint32_t VL_FLOAT32_SIZE = GetVRegSize() / sizeof(float);
constexpr uint32_t VL_FLOAT16_SIZE = GetVRegSize() / sizeof(half);
constexpr uint32_t BLOCK_TYPE_SIZE = GetUbBlockSize();
constexpr uint32_t HALF_INTERLEAVE_COEF = 2;
constexpr uint32_t QUARTER_MODE_COEF = 4;
constexpr uint32_t DOUBLE_BUFFER = 2;

enum class RotaryPosEmbeddingMode : int64_t {
    HALF = 0,
    INTERLEAVE = 1,
    QUARTER = 2,
    DEEPSEEK_INTERLEAVE = 3
};

/*
    qOut[0] = q[0] * cos[0] - q[1] * sin[0]
    qOut[1] = q[1] * cos[1] + q[0] * sin[1]
*/
template <typename T>
__aicore__ inline void HalfAlignVF(__local_mem__ T* sin, __local_mem__ T* cos, __local_mem__ T* in,
                                   __local_mem__ bfloat16_t* outUb, uint32_t dLen, uint32_t dAlign,
                                   uint32_t cosSinAlign, uint32_t dAlignH8Out, uint16_t currSNum, uint16_t currDNum,
                                   uint32_t sliceStart)
{
    __local_mem__ T* sinUb = sin;
    __local_mem__ T* cosUb = cos;
    __local_mem__ T* inUb = in;
    __local_mem__ T* ropeInUb = inUb + sliceStart;
    __local_mem__ bfloat16_t* ropeOutUb = outUb + sliceStart;
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint32_t halfDAlign = ops::CeilAlign(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t repeatTimes = ops::CeilDiv(halfD, VL_FLOAT32_SIZE);
    uint16_t nopeRepeatTimes = ops::CeilDiv(sliceStart, VL_FLOAT32_SIZE);
    __local_mem__ T* currInUb;
    __local_mem__ bfloat16_t* currOutUb;
    __local_mem__ T* currSinUb;
    __local_mem__ T* currCosUb;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregIn;
        MicroAPI::RegTensor<float> vregHalfIn;
        MicroAPI::RegTensor<float> vregSin;
        MicroAPI::RegTensor<float> vregHalfSin;
        MicroAPI::RegTensor<float> vregCos;
        MicroAPI::RegTensor<float> vregHalfCos;
        MicroAPI::RegTensor<float> vregOut;
        MicroAPI::RegTensor<float> vregHalfOut;
        MicroAPI::RegTensor<bfloat16_t> tmp1;
        MicroAPI::RegTensor<bfloat16_t> tmp2;
        MicroAPI::RegTensor<float> vregNope;
        MicroAPI::MaskReg preg;

        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * cosSinAlign;
            currCosUb = cosUb + sIdx * cosSinAlign;
            for (uint16_t row = 0; row < currDNum; row++) {
                if constexpr (!IsSameType<T, bfloat16_t>::value) {
                    currInUb = inUb + (sIdx * currDNum + row) * dAlign;
                    currOutUb = outUb + (sIdx * currDNum + row) * dAlign;
                    uint32_t updateCnt = sliceStart;
                    for (uint16_t i = 0; i < nopeRepeatTimes; i++) {
                        preg = MicroAPI::UpdateMask<float>(updateCnt);
                        uint32_t offset = i * VL_FLOAT32_SIZE;
                        ops::LoadOneTensorForDtypeT(currInUb, vregNope, preg, offset);
                        Cast<bfloat16_t, float, ops::castTraitB322B16>(tmp1, vregNope, preg);
                        DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb + offset, tmp1, preg);
                    }
                }
                currInUb = ropeInUb + (sIdx * currDNum + row) * dAlign;
                currOutUb = ropeOutUb + (sIdx * currDNum + row) * dAlignH8Out;
                uint32_t updateCnt = halfD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = MicroAPI::UpdateMask<float>(updateCnt);
                    uint32_t offset = i * VL_FLOAT32_SIZE;
                    uint32_t halfOffset = offset + halfDAlign;
                    ops::LoadTwoTensorForDtypeT<T>(currInUb, currInUb, vregIn, vregHalfIn, preg, preg, offset,
                                                   halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currSinUb, currSinUb, vregSin, vregHalfSin, preg, preg, offset,
                                                   halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currCosUb, currCosUb, vregCos, vregHalfCos, preg, preg, offset,
                                                   halfOffset);

                    Mul(vregSin, vregSin, vregHalfIn, preg);
                    Mul(vregHalfOut, vregHalfSin, vregIn, preg);
                    Mul(vregCos, vregCos, vregIn, preg);
                    Sub(vregOut, vregCos, vregSin, preg);
                    Mul(vregHalfCos, vregHalfCos, vregHalfIn, preg);
                    Add(vregHalfOut, vregHalfOut, vregHalfCos, preg);

                    Cast<bfloat16_t, float, ops::castTraitB322B16>(tmp1, vregOut, preg);
                    Cast<bfloat16_t, float, ops::castTraitB322B16>(tmp2, vregHalfOut, preg);
                    DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb + offset, tmp1, preg);
                    DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb + halfOffset, tmp2,
                                                                                      preg);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void InterleaveModeVF(__local_mem__ T* sin, __local_mem__ T* cos, __local_mem__ T* in,
                                        __local_mem__ bfloat16_t* outUb, uint32_t dLen, uint32_t dAlign,
                                        uint32_t cosSinAlign, uint32_t dAlignH8Out, uint16_t currSNum,
                                        uint16_t currDNum, uint32_t sliceStart)
{
    __local_mem__ T* sinUb = sin;
    __local_mem__ T* cosUb = cos;
    __local_mem__ T* inUb = in;
    __local_mem__ T* ropeInUb = inUb + sliceStart;
    __local_mem__ bfloat16_t* ropeOutUb = outUb + sliceStart;
    uint16_t repeatTimes = dLen / VL_FLOAT32_SIZE;
    uint32_t dAlignLen = dAlign;
    uint16_t loopNum = repeatTimes / 2;
    uint32_t tailNum = dLen - loopNum * 2 * VL_FLOAT32_SIZE;
    uint16_t tailTwoVL = tailNum / VL_FLOAT32_SIZE;
    uint16_t tailOneVL = (tailTwoVL == 1) ? 0 : 1;
    uint32_t tailLen = tailNum % VL_FLOAT32_SIZE;
    uint16_t nopeRepeatTimes = ops::CeilDiv(sliceStart, VL_FLOAT32_SIZE);
    __local_mem__ T* currInUb;
    __local_mem__ bfloat16_t* currOutUb;
    __local_mem__ T* currSinUb;
    __local_mem__ T* currCosUb;
    __local_mem__ T* tailSinUb;
    __local_mem__ T* tailCosUb;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregFormerCos;
        MicroAPI::RegTensor<float> vregLatterCos;
        MicroAPI::RegTensor<float> vregFormerSin;
        MicroAPI::RegTensor<float> vregLatterSin;
        MicroAPI::RegTensor<float> vregFormerIn;
        MicroAPI::RegTensor<float> vregLatterIn;
        MicroAPI::RegTensor<float> vregOdd;
        MicroAPI::RegTensor<float> vregEven;
        MicroAPI::RegTensor<float> vregFormerOut;
        MicroAPI::RegTensor<float> vregLatterOut;
        MicroAPI::RegTensor<bfloat16_t> tmp1;
        MicroAPI::RegTensor<bfloat16_t> tmp2;
        MicroAPI::RegTensor<float> vregNope;
        MicroAPI::MaskReg pregLoop;
        MicroAPI::MaskReg pregTail;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * cosSinAlign;
            currCosUb = cosUb + sIdx * cosSinAlign;
            for (uint16_t idxD = 0; idxD < currDNum; idxD++) {
                if constexpr (!IsSameType<T, bfloat16_t>::value) {
                    currInUb = inUb + (sIdx * currDNum + idxD) * dAlignLen;
                    currOutUb = outUb + (sIdx * currDNum + idxD) * dAlignLen;
                    uint32_t updateCnt = sliceStart;
                    for (uint16_t i = 0; i < nopeRepeatTimes; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(updateCnt);
                        uint32_t offset = i * VL_FLOAT32_SIZE;
                        ops::LoadOneTensorForDtypeT(currInUb, vregNope, pregLoop, offset);
                        Cast<bfloat16_t, float, ops::castTraitB322B16>(tmp1, vregNope, pregLoop);
                        DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb + offset, tmp1, pregLoop);
                    }
                }
                uint32_t updateCnt = dLen;
                currInUb = ropeInUb + (sIdx * currDNum + idxD) * dAlignLen;
                currOutUb = ropeOutUb + (sIdx * currDNum + idxD) * dAlignH8Out;
                pregLoop = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
                for (uint16_t i = 0; i < loopNum; i++) {
                    uint32_t evenOffSet = (i * 2) * VL_FLOAT32_SIZE;
                    uint32_t oddOffset = evenOffSet + VL_FLOAT32_SIZE;
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregFormerIn, pregLoop, evenOffSet);
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregLatterIn, pregLoop, oddOffset);
                    ops::LoadOneTensorForDtypeT<T>(currCosUb, vregFormerCos, pregLoop, evenOffSet);
                    ops::LoadOneTensorForDtypeT<T>(currCosUb, vregLatterCos, pregLoop, oddOffset);
                    ops::LoadOneTensorForDtypeT<T>(currSinUb, vregFormerSin, pregLoop, evenOffSet);
                    ops::LoadOneTensorForDtypeT<T>(currSinUb, vregLatterSin, pregLoop, oddOffset);
                    Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregLoop);
                    Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregLoop);
                    MicroAPI::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                    Muls(vregOdd, vregOdd, float(-1.0), pregLoop);
                    MicroAPI::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                    Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregLoop);
                    Add(vregFormerCos, vregFormerCos, vregFormerSin, pregLoop);
                    Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregLoop);
                    Add(vregLatterCos, vregLatterCos, vregLatterSin, pregLoop);
                    Cast<bfloat16_t, float, ops::castTraitB322B16>(tmp1, vregFormerCos, pregLoop);
                    Cast<bfloat16_t, float, ops::castTraitB322B16>(tmp2, vregLatterCos, pregLoop);
                    DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb + evenOffSet, tmp1,
                                                                                      pregLoop);
                    DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb + oddOffset, tmp2,
                                                                                      pregLoop);
                }

                currInUb = ropeInUb + (sIdx * currDNum + idxD) * dAlignLen + (loopNum * 2 * VL_FLOAT32_SIZE);
                currOutUb = ropeOutUb + (sIdx * currDNum + idxD) * dAlignH8Out + (loopNum * 2 * VL_FLOAT32_SIZE);
                tailSinUb = sinUb + sIdx * cosSinAlign + loopNum * 2 * VL_FLOAT32_SIZE;
                tailCosUb = cosUb + sIdx * cosSinAlign + loopNum * 2 * VL_FLOAT32_SIZE;
                for (uint16_t i = 0; i < tailTwoVL; i++) {
                    uint32_t updateCnt = tailLen;
                    pregTail = MicroAPI::UpdateMask<float>(updateCnt);
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregFormerIn, pregLoop, 0);
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregLatterIn, pregTail, VL_FLOAT32_SIZE);
                    ops::LoadOneTensorForDtypeT<T>(tailCosUb, vregFormerCos, pregLoop, 0);
                    ops::LoadOneTensorForDtypeT<T>(tailCosUb, vregLatterCos, pregTail, VL_FLOAT32_SIZE);
                    ops::LoadOneTensorForDtypeT<T>(tailSinUb, vregFormerSin, pregLoop, 0);
                    ops::LoadOneTensorForDtypeT<T>(tailSinUb, vregLatterSin, pregTail, VL_FLOAT32_SIZE);
                    Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregLoop);
                    Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregTail);
                    MicroAPI::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                    Muls(vregOdd, vregOdd, float(-1.0), pregLoop);
                    MicroAPI::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                    Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregLoop);
                    Add(vregFormerCos, vregFormerCos, vregFormerSin, pregLoop);
                    Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregTail);
                    Add(vregLatterCos, vregLatterCos, vregLatterSin, pregTail);
                    Cast<bfloat16_t, float, ops::castTraitB322B16>(tmp1, vregFormerCos, pregLoop);
                    Cast<bfloat16_t, float, ops::castTraitB322B16>(tmp2, vregLatterCos, pregTail);
                    DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb, tmp1, pregLoop);
                    DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb + VL_FLOAT32_SIZE, tmp2,
                                                                                      pregTail);
                }

                for (uint16_t i = 0; i < tailOneVL; i++) {
                    uint32_t updateCnt = tailLen;
                    pregTail = MicroAPI::UpdateMask<float>(updateCnt);
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregFormerIn, pregTail, 0);
                    ops::LoadOneTensorForDtypeT<T>(tailCosUb, vregFormerCos, pregTail, 0);
                    ops::LoadOneTensorForDtypeT<T>(tailSinUb, vregFormerSin, pregTail, 0);
                    Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregTail);
                    MicroAPI::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                    Muls(vregOdd, vregOdd, float(-1.0), pregTail);
                    MicroAPI::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                    Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregTail);
                    Add(vregFormerCos, vregFormerCos, vregFormerSin, pregTail);
                    Cast<bfloat16_t, float, ops::castTraitB322B16>(tmp1, vregFormerCos, pregTail);
                    DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb, tmp1, pregTail);
                }
            }
        }
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void BatchHalfAlignVF(__local_mem__ T* in, __local_mem__ T* cos, __local_mem__ T* sin,
                                        __local_mem__ bfloat16_t* outUb, uint16_t sLength, uint16_t bLength,
                                        uint16_t nLength, int64_t d, int64_t dAlign, int64_t cosSinAlign,
                                        int64_t dAlignH8Out, int64_t ubFactorS, int64_t ubFactorN, uint32_t sliceStart)
{
    uint32_t dHalfSize = d / HALF_INTERLEAVE_COEF;
    uint16_t dLoopCount = (dHalfSize + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    uint32_t dHalfOffset = cosSinAlign / HALF_INTERLEAVE_COEF;
    uint32_t cosHalfOffset = cosSinAlign / HALF_INTERLEAVE_COEF;
    uint16_t nopeRepeatTimes = ops::CeilDiv(sliceStart, VL_FLOAT32_SIZE);

    int32_t bStepUb = ubFactorN * ubFactorS * dAlign;
    int32_t nStepUb = ubFactorS * dAlign;
    int32_t bStepOut = ubFactorN * ubFactorS * dAlignH8Out;
    int32_t nStepOut = ubFactorS * dAlignH8Out;
    int32_t nStepCosSin = ubFactorS * cosSinAlign;

    __local_mem__ T* ropeInUb = in + sliceStart;
    __local_mem__ bfloat16_t* ropeOutUb = outUb + sliceStart;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inPart1Reg;
        MicroAPI::RegTensor<float> inPart2Reg;
        MicroAPI::RegTensor<float> cosPart1Reg;
        MicroAPI::RegTensor<float> cosPart2Reg;
        MicroAPI::RegTensor<float> sinPart1Reg;
        MicroAPI::RegTensor<float> sinPart2Reg;
        MicroAPI::RegTensor<bfloat16_t> tmp1;
        MicroAPI::RegTensor<bfloat16_t> tmp2;
        MicroAPI::RegTensor<float> vregNope;
        MicroAPI::MaskReg pregLoop;
        __local_mem__ T* currInUb;
        __local_mem__ bfloat16_t* currOutUb;
        __local_mem__ T *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    if constexpr (!IsSameType<T, bfloat16_t>::value) {
                        currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                        currOutUb = outUb + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                        uint32_t updateCnt = sliceStart;
                        for (uint16_t i = 0; i < nopeRepeatTimes; i++) {
                            pregLoop = MicroAPI::UpdateMask<float>(updateCnt);
                            uint32_t offset = i * VL_FLOAT32_SIZE;
                            ops::LoadOneTensorForDtypeT(currInUb, vregNope, pregLoop, offset);
                            Cast<bfloat16_t, float, ops::castTraitB322B16>(tmp1, vregNope, pregLoop);
                            DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb + offset, tmp1,
                                                                                     pregLoop);
                        }
                    }
                    uint32_t count = dHalfSize;
                    currInUb = ropeInUb + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    currOutUb = ropeOutUb + bIdx * bStepOut + nIdx * nStepOut + sIdx * dAlignH8Out;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * cosSinAlign;
                        currSinUb = sin + sIdx * cosSinAlign;
                    } else {
                        currCosUb = cos + bIdx * nStepCosSin + sIdx * cosSinAlign;
                        currSinUb = sin + bIdx * nStepCosSin + sIdx * cosSinAlign;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(count);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart2Reg, pregLoop,
                                                       i * VL_FLOAT32_SIZE + dHalfOffset);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart2Reg, pregLoop,
                                                       i * VL_FLOAT32_SIZE + cosHalfOffset);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart2Reg, pregLoop,
                                                       i * VL_FLOAT32_SIZE + cosHalfOffset);
                        Mul(cosPart1Reg, inPart1Reg, cosPart1Reg, pregLoop);
                        Mul(sinPart1Reg, inPart2Reg, sinPart1Reg, pregLoop);
                        Sub(cosPart1Reg, cosPart1Reg, sinPart1Reg, pregLoop);
                        Mul(cosPart2Reg, inPart2Reg, cosPart2Reg, pregLoop);
                        Mul(sinPart2Reg, sinPart2Reg, inPart1Reg, pregLoop);
                        Add(cosPart2Reg, cosPart2Reg, sinPart2Reg, pregLoop);

                        Cast<bfloat16_t, float, ops::castTraitB322B16>(tmp1, cosPart1Reg, pregLoop);
                        Cast<bfloat16_t, float, ops::castTraitB322B16>(tmp2, cosPart2Reg, pregLoop);
                        DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                            currOutUb + i * VL_FLOAT32_SIZE, tmp1, pregLoop);
                        DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                            currOutUb + i * VL_FLOAT32_SIZE + dHalfOffset, tmp2, pregLoop);
                    }
                }
            }
        }
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void BatchInterleaveModeVF(__local_mem__ T* in, __local_mem__ T* cos, __local_mem__ T* sin,
                                             __local_mem__ bfloat16_t* outUb, uint16_t sLength, uint16_t bLength,
                                             uint16_t nLength, int64_t d, int64_t dAlign, int64_t cosSinAlign,
                                             int64_t dAlignH8Out, int64_t ubFactorS, int64_t ubFactorN,
                                             uint32_t sliceStart)
{
    uint32_t loopSize = 2 * VL_FLOAT32_SIZE;
    uint16_t dLoopCount = (d + loopSize - 1) / loopSize;
    uint16_t nopeRepeatTimes = ops::CeilDiv(sliceStart, VL_FLOAT32_SIZE);

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

    int32_t bStepUb = ubFactorN * ubFactorS * dAlign;
    int32_t nStepUb = ubFactorS * dAlign;
    int32_t bStepOut = ubFactorN * ubFactorS * dAlignH8Out;
    int32_t nStepOut = ubFactorS * dAlignH8Out;
    int32_t nStepCosSin = ubFactorS * cosSinAlign;

    __local_mem__ T* ropeInUb = in + sliceStart;
    __local_mem__ bfloat16_t* ropeOutUb = outUb + sliceStart;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inPart1Reg;
        MicroAPI::RegTensor<float> inPart2Reg;
        MicroAPI::RegTensor<float> cosPart1Reg;
        MicroAPI::RegTensor<float> cosPart2Reg;
        MicroAPI::RegTensor<float> sinPart1Reg;
        MicroAPI::RegTensor<float> sinPart2Reg;
        MicroAPI::RegTensor<bfloat16_t> tmp1;
        MicroAPI::RegTensor<bfloat16_t> tmp2;
        MicroAPI::RegTensor<float> vregNope;
        MicroAPI::MaskReg pregLoop;
        MicroAPI::MaskReg pregPart1;
        MicroAPI::MaskReg pregPart2;
        __local_mem__ T* currInUb;
        __local_mem__ bfloat16_t* currOutUb;
        __local_mem__ T *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    if constexpr (!IsSameType<T, bfloat16_t>::value) {
                        currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                        currOutUb = outUb + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                        uint32_t updateCnt = sliceStart;
                        for (uint16_t i = 0; i < nopeRepeatTimes; i++) {
                            pregLoop = MicroAPI::UpdateMask<float>(updateCnt);
                            uint32_t offset = i * VL_FLOAT32_SIZE;
                            ops::LoadOneTensorForDtypeT(currInUb, vregNope, pregLoop, offset);
                            Cast<bfloat16_t, float, ops::castTraitB322B16>(tmp1, vregNope, pregLoop);
                            DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb + offset, tmp1,
                                                                                     pregLoop);
                        }
                    }
                    uint32_t halfCnt = halfNum;
                    uint32_t part1Cnt = part1Num;
                    uint32_t part2Cnt = part2Num;
                    currInUb = ropeInUb + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    currOutUb = ropeOutUb + bIdx * bStepOut + nIdx * nStepOut + sIdx * dAlignH8Out;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * cosSinAlign;
                        currSinUb = sin + sIdx * cosSinAlign;
                    } else {
                        currCosUb = cos + bIdx * nStepCosSin + sIdx * cosSinAlign;
                        currSinUb = sin + bIdx * nStepCosSin + sIdx * cosSinAlign;
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
                        MicroAPI::DeInterleave<float>(inPart1Reg, inPart2Reg, inPart1Reg, inPart2Reg);
                        Muls(inPart2Reg, inPart2Reg, float(-1.0), pregLoop);
                        MicroAPI::Interleave<float>(inPart1Reg, inPart2Reg, inPart2Reg, inPart1Reg);
                        Mul(sinPart1Reg, sinPart1Reg, inPart1Reg, pregPart1);
                        Add(cosPart1Reg, cosPart1Reg, sinPart1Reg, pregPart1);
                        Mul(sinPart2Reg, sinPart2Reg, inPart2Reg, pregPart2);
                        Add(cosPart2Reg, cosPart2Reg, sinPart2Reg, pregPart2);

                        Cast<bfloat16_t, float, ops::castTraitB322B16>(tmp1, cosPart1Reg, pregPart1);
                        Cast<bfloat16_t, float, ops::castTraitB322B16>(tmp2, cosPart2Reg, pregPart2);
                        DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(currOutUb + i * loopSize,
                                                                                          tmp1, pregPart1);
                        DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                            currOutUb + i * loopSize + VL_FLOAT32_SIZE, tmp2, pregPart2);
                    }
                }
            }
        }
    }
}

__aicore__ inline void VFProcessSwigluHiF4InvScaleAndQuant(
    const LocalTensor<int32_t>& yQuantLocal, const LocalTensor<float>& xLocal, const LocalTensor<uint8_t>& eaLocal,
    const LocalTensor<uint32_t>& ebLocal, const LocalTensor<uint8_t>& ecLocal, const uint32_t curColNum)
{
    // 512个数 -128个ec 64个eb和8个ea  1 0 1 0
    __local_mem__ int32_t* yQuantLocalAddr = (__local_mem__ int32_t*)yQuantLocal.GetPhyAddr();
    __local_mem__ float* xLocalAddr = (__local_mem__ float*)xLocal.GetPhyAddr();
    __local_mem__ uint8_t* eaLocalAddr = (__local_mem__ uint8_t*)eaLocal.GetPhyAddr();
    __local_mem__ uint32_t* ebLocalAddr = (__local_mem__ uint32_t*)ebLocal.GetPhyAddr();
    __local_mem__ uint8_t* ecLocalAddr = (__local_mem__ uint8_t*)ecLocal.GetPhyAddr();
    const uint16_t VL_LEN = 256;
    uint16_t vlForB16 = VL_LEN / sizeof(bfloat16_t);
    uint16_t vlForB32 = VL_LEN / sizeof(float);
    uint16_t loopCount = ops::CeilDiv(curColNum, vlForB16 * 4); // 一次搬入4个VL的输入，也就是512个b16数据
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<bfloat16_t> x1;
        MicroAPI::RegTensor<bfloat16_t> x2;
        MicroAPI::RegTensor<bfloat16_t> x3;
        MicroAPI::RegTensor<bfloat16_t> x4;

        MicroAPI::RegTensor<bfloat16_t> x10;
        MicroAPI::RegTensor<bfloat16_t> x20;
        MicroAPI::RegTensor<bfloat16_t> x30;
        MicroAPI::RegTensor<bfloat16_t> x40;

        MicroAPI::RegTensor<bfloat16_t> max1;
        MicroAPI::RegTensor<bfloat16_t> max2;
        MicroAPI::RegTensor<bfloat16_t> maxPer4;
        MicroAPI::RegTensor<bfloat16_t> maxPer4Tmp1;
        MicroAPI::RegTensor<bfloat16_t> maxPer4Tmp2;
        MicroAPI::RegTensor<bfloat16_t> vmaxPer4;
        MicroAPI::RegTensor<bfloat16_t> maxPer4Coeff;
        MicroAPI::RegTensor<bfloat16_t> maxPer8;
        MicroAPI::RegTensor<bfloat16_t> maxPer8Tmp;
        MicroAPI::RegTensor<bfloat16_t> maxPer8Tmp1;
        MicroAPI::RegTensor<bfloat16_t> vmaxPer8;
        MicroAPI::RegTensor<bfloat16_t> maxPer8Coeff;
        MicroAPI::RegTensor<bfloat16_t> vb16EaRecBf16;
        MicroAPI::RegTensor<bfloat16_t> tmp;
        MicroAPI::RegTensor<bfloat16_t> numOne;
        MicroAPI::RegTensor<bfloat16_t> numHalf;
        MicroAPI::RegTensor<bfloat16_t> invScale0;
        MicroAPI::RegTensor<bfloat16_t> invScale1;
        MicroAPI::RegTensor<bfloat16_t> invScale2;
        MicroAPI::RegTensor<bfloat16_t> invScale3;
        MicroAPI::RegTensor<bfloat16_t> invScale4;
        MicroAPI::RegTensor<fp4x2_e1m2_t> x1Out;
        MicroAPI::RegTensor<fp4x2_e1m2_t> x2Out;
        MicroAPI::RegTensor<fp4x2_e1m2_t> x3Out;
        MicroAPI::RegTensor<fp4x2_e1m2_t> x4Out;
        MicroAPI::RegTensor<bfloat16_t> maxPer64;
        MicroAPI::RegTensor<uint16_t> expMaskBF16;
        MicroAPI::RegTensor<uint16_t> numScale;
        MicroAPI::RegTensor<uint16_t> e6m2;
        MicroAPI::RegTensor<uint8_t> e6m2Res;
        MicroAPI::RegTensor<uint8_t> tmp1;
        MicroAPI::UnalignReg uregEa;
        MicroAPI::UnalignReg uregEb;
        Duplicate(expMaskBF16, 0x7fff);
        Duplicate(numScale, 0x3e12);
        Duplicate(numOne, 1.0);
        float numberhalf = 0.5;
        Duplicate(numHalf, static_cast<bfloat16_t>(numberhalf));
        MicroAPI::MaskReg ebMask;
        MicroAPI::MaskReg ecMask;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitE6M2 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::CastTrait castTraitE2M1 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        //
        for (uint16_t i = 0; i < loopCount; i++) {
            DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                (MicroAPI::RegTensor<float>&)x1, (MicroAPI::RegTensor<float>&)x2, xLocalAddr, vlForB32 * 2);
            DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                (MicroAPI::RegTensor<float>&)x3, (MicroAPI::RegTensor<float>&)x4, xLocalAddr, vlForB32 * 2);
            And((MicroAPI::RegTensor<uint16_t>&)x10, (MicroAPI::RegTensor<uint16_t>&)x1, expMaskBF16, maskAll);
            And((MicroAPI::RegTensor<uint16_t>&)x20, (MicroAPI::RegTensor<uint16_t>&)x2, expMaskBF16, maskAll);
            And((MicroAPI::RegTensor<uint16_t>&)x30, (MicroAPI::RegTensor<uint16_t>&)x3, expMaskBF16, maskAll);
            And((MicroAPI::RegTensor<uint16_t>&)x40, (MicroAPI::RegTensor<uint16_t>&)x4, expMaskBF16, maskAll);
            Max(max1, x10, x20, maskAll);
            Max(max2, x30, x40, maskAll);

            DeInterleave(max1, max2, max1, max2);
            Max(maxPer4, max1, max2, maskAll); // 512--> 128

            DeInterleave(maxPer4Tmp1, maxPer4Tmp2, maxPer4, maxPer4);
            Max(maxPer8, maxPer4Tmp1, maxPer4Tmp2, maskAll); // 128 -> 64

            ReduceMaxWithDataBlock<uint16_t>((MicroAPI::RegTensor<uint16_t>&)maxPer64,
                                             (MicroAPI::RegTensor<uint16_t>&)maxPer4,
                                             maskAll); // (128, 1) -> (8, 16) -> (8, 1)

            // Vmax * Const_rec
            Mul(maxPer64, maxPer64, (MicroAPI::RegTensor<bfloat16_t>&)numScale, maskAll);

            Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((MicroAPI::RegTensor<fp8_e6m2_t>&)e6m2, maxPer64, maskAll);
            Pack(e6m2Res, (MicroAPI::RegTensor<uint16_t>&)e6m2);
            DataCopyUnAlign<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(eaLocalAddr, e6m2Res, uregEa, 8);

            ReciprocalCast<bfloat16_t, fp8_e6m2_t, ops::castTraitReciprocal>(
                vb16EaRecBf16, (MicroAPI::RegTensor<fp8_e6m2_t>&)e6m2, maskAll);

            Interleave(vb16EaRecBf16, tmp, vb16EaRecBf16, vb16EaRecBf16); // 8 -> 16
            Interleave(vb16EaRecBf16, tmp, vb16EaRecBf16, vb16EaRecBf16); // 16 -> 32
            Interleave(vb16EaRecBf16, tmp, vb16EaRecBf16, vb16EaRecBf16); // 32 -> 64
            Interleave(vb16EaRecBf16, tmp, vb16EaRecBf16, vb16EaRecBf16); // 64 -> 128

            Interleave(maxPer8, tmp, maxPer8, maxPer8); // 64 -> 128

            Mul(maxPer8, maxPer8, vb16EaRecBf16, maskAll);
            CompareScalar<bfloat16_t, CMPMODE::GE>(ebMask, maxPer8, 4, maskAll);
            DataCopyUnAlign<uint32_t>(ebLocalAddr, ebMask, uregEb);
            Select(maxPer8Coeff, numHalf, numOne, ebMask);

            Mul(maxPer4, maxPer4, vb16EaRecBf16, maskAll);
            Mul(maxPer4, maxPer4, maxPer8Coeff, maskAll);
            CompareScalar<bfloat16_t, CMPMODE::GE>(ecMask, maxPer4, 2, maskAll);
            Select(maxPer4Coeff, numHalf, numOne, ecMask);
            DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE, MaskDist::DIST_PACK>(ecLocalAddr, ecMask, 16);

            Mul(invScale0, vb16EaRecBf16, maxPer8Coeff, maskAll);
            Mul(invScale0, invScale0, maxPer4Coeff, maskAll); // 这一步算出来了nan

            Interleave(invScale1, invScale3, invScale0, invScale0);
            Interleave(invScale1, invScale2, invScale1, invScale1);
            Interleave(invScale3, invScale4, invScale3, invScale3);

            Interleave((MicroAPI::RegTensor<float>&)x1, (MicroAPI::RegTensor<float>&)x2,
                       (MicroAPI::RegTensor<float>&)x1, (MicroAPI::RegTensor<float>&)x2);
            Interleave((MicroAPI::RegTensor<float>&)x3, (MicroAPI::RegTensor<float>&)x4,
                       (MicroAPI::RegTensor<float>&)x3, (MicroAPI::RegTensor<float>&)x4);

            Mul(x1, x1, invScale1, maskAll);
            Mul(x2, x2, invScale2, maskAll);
            Mul(x3, x3, invScale3, maskAll);
            Mul(x4, x4, invScale4, maskAll);

            Cast<fp4x2_e1m2_t, bfloat16_t, castTraitE2M1>(x1Out, x1, maskAll);
            Cast<fp4x2_e1m2_t, bfloat16_t, castTraitE2M1>(x2Out, x2, maskAll);
            Cast<fp4x2_e1m2_t, bfloat16_t, castTraitE2M1>(x3Out, x3, maskAll);
            Cast<fp4x2_e1m2_t, bfloat16_t, castTraitE2M1>(x4Out, x4, maskAll);

            DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                yQuantLocalAddr, (MicroAPI::RegTensor<int32_t>&)x1Out, 16, maskAll);
            DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                yQuantLocalAddr, (MicroAPI::RegTensor<int32_t>&)x2Out, 16, maskAll);
            DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                yQuantLocalAddr, (MicroAPI::RegTensor<int32_t>&)x3Out, 16, maskAll);
            DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                yQuantLocalAddr, (MicroAPI::RegTensor<int32_t>&)x4Out, 16, maskAll);
        }
        DataCopyUnAlignPost(eaLocalAddr, uregEa, 0);
        DataCopyUnAlignPost<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ebLocalAddr, uregEb, 0);
    }
}

// Ea:Eb:Ec=1:1:2 最终合并的Scale为32bits
__aicore__ inline void VFProcessSwigluGroupQuantScale(const LocalTensor<uint16_t>& scaleLocal,
                                                      const LocalTensor<uint8_t>& eaLocal,
                                                      const LocalTensor<uint8_t>& ebLocal,
                                                      const LocalTensor<uint16_t>& ecLocal, const uint32_t curColNum)
{
    __local_mem__ uint16_t* scaleLocalAddr = (__local_mem__ uint16_t*)scaleLocal.GetPhyAddr();
    __local_mem__ uint8_t* eaLocalAddr = (__local_mem__ uint8_t*)eaLocal.GetPhyAddr();
    __local_mem__ uint8_t* ebLocalAddr = (__local_mem__ uint8_t*)ebLocal.GetPhyAddr();
    __local_mem__ uint16_t* ecLocalAddr = (__local_mem__ uint16_t*)ecLocal.GetPhyAddr();
    const uint16_t VL_LEN = 256;
    uint16_t vlForB8 = VL_LEN / sizeof(uint8_t);
    uint16_t vlForB16 = VL_LEN / sizeof(uint16_t);
    uint16_t eaNum = ops::CeilDiv(curColNum, 64);
    uint16_t loopCount = ops::CeilDiv(eaNum, vlForB8);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint8_t> eA;
        MicroAPI::RegTensor<uint8_t> eB;
        MicroAPI::RegTensor<uint16_t> eC0;
        MicroAPI::RegTensor<uint16_t> eC1;
        MicroAPI::RegTensor<uint8_t> tmp;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopCount; i++) {
            DataCopy<uint8_t, AscendC::MicroAPI::LoadDist::DIST_NORM>(eA, eaLocalAddr + i * vlForB8);
            DataCopy<uint8_t, AscendC::MicroAPI::LoadDist::DIST_NORM>(eB, ebLocalAddr + i * vlForB8);
            DataCopy<uint16_t, AscendC::MicroAPI::LoadDist::DIST_NORM>(eC0, ecLocalAddr + i * vlForB16 * 2);
            DataCopy<uint16_t, AscendC::MicroAPI::LoadDist::DIST_NORM>(eC1, ecLocalAddr + i * vlForB16 * 2 + vlForB16);

            Interleave(eA, eB, eA, eB);
            DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(scaleLocalAddr + i * vlForB16 * 4,
                                                                    (MicroAPI::RegTensor<uint16_t>&)eA, eC0, maskAll);
            DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(scaleLocalAddr + i * vlForB16 * 4 + vlForB16 * 2,
                                                                    (MicroAPI::RegTensor<uint16_t>&)eB, eC1, maskAll);
        }
    }
}

#endif // PARTIAL_ROTARY_MUL_QUANT_HIF4_COMMON_H
