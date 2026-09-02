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
 * \file inplace_partial_rotary_mul_arpe_common.h
 * \brief
 */
#ifndef INPLACE_PARTIAL_ROTARY_MUL_ARPE_COMMON_H
#define INPLACE_PARTIAL_ROTARY_MUL_ARPE_COMMON_H

#include "inplace_partial_rotary_mul_common.h"

using namespace AscendC;

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

struct InplacePartialRotaryPosEmbeddingMode {
    static constexpr int64_t HALF = 0;
    static constexpr int64_t INTERLEAVE = 1;
    static constexpr int64_t QUARTER = 2;
    static constexpr int64_t DEEPSEEK_INTERLEAVE = 3;
};

/*
    qOut[0] = q[0] * cos[0] - q[1] * sin[0]
    qOut[1] = q[1] * cos[1] + q[0] * sin[1]
*/
template <typename T>
__aicore__ inline void HalfAlignVF(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                   const LocalTensor<T> &inTensor, const LocalTensor<T> &outTensor, uint32_t dLen,
                                   uint32_t dAlign, uint16_t currSNum, uint16_t currDNum)
{
    __ubuf__ T *sinUb = (__ubuf__ T *)sinTensor.GetPhyAddr();
    __ubuf__ T *cosUb = (__ubuf__ T *)cosTensor.GetPhyAddr();
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ T *outUb = (__ubuf__ T *)outTensor.GetPhyAddr();
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint32_t halfDAlign = ops::CeilAlign(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t repeatTimes = ops::CeilDiv(halfD, VL_FLOAT32_SIZE);
    __ubuf__ T *currInUb;
    __ubuf__ T *currOutUb;
    __ubuf__ T *currSinUb;
    __ubuf__ T *currCosUb;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vregIn;
        Reg::RegTensor<float> vregHalfIn;
        Reg::RegTensor<float> vregSin;
        Reg::RegTensor<float> vregHalfSin;
        Reg::RegTensor<float> vregCos;
        Reg::RegTensor<float> vregHalfCos;
        Reg::RegTensor<float> vregOut;
        Reg::RegTensor<float> vregHalfOut;
        Reg::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlign;
            currCosUb = cosUb + sIdx * dAlign;
            for (uint16_t row = 0; row < currDNum; row++) {
                currInUb = inUb + (sIdx * currDNum + row) * dAlign;
                currOutUb = outUb + (sIdx * currDNum + row) * dAlign;
                uint32_t updateCnt = halfD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = Reg::UpdateMask<float>(updateCnt);
                    uint32_t offset = i * VL_FLOAT32_SIZE;
                    uint32_t halfOffset = offset + halfDAlign;
                    ops::LoadTwoTensorForDtypeT<T>(currInUb, currInUb, vregIn, vregHalfIn, preg, preg, offset,
                                                   halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currSinUb, currSinUb, vregSin, vregHalfSin, preg, preg, offset,
                                                   halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currCosUb, currCosUb, vregCos, vregHalfCos, preg, preg, offset,
                                                   halfOffset);

                    Reg::Mul(vregSin, vregSin, vregHalfIn, preg);
                    Reg::Mul(vregHalfOut, vregHalfSin, vregIn, preg);
                    Reg::Mul(vregCos, vregCos, vregIn, preg);
                    Reg::Sub(vregOut, vregCos, vregSin, preg);
                    Reg::Mul(vregHalfCos, vregHalfCos, vregHalfIn, preg);
                    Reg::Add(vregHalfOut, vregHalfOut, vregHalfCos, preg);

                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregOut, preg, offset);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregHalfOut, preg, halfOffset);
                }
            }
        }
    }
}

/*
    qOut[0] = q[0] * cos[0] - q[1] * sin[0]
    qOut[1] = q[1] * cos[1] + q[0] * sin[1]
    qOut[2] = q[2] * cos[2] - q[3] * sin[2]
    qOut[3] = q[3] * cos[3] + q[2] * sin[3]
*/
template <typename T>
__aicore__ inline void QuarterAlignVF(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                      const LocalTensor<T> &inTensor, const LocalTensor<T> &outTensor, uint32_t dLen,
                                      uint32_t dAlign, uint16_t currSNum, uint16_t currDNum)
{
    __ubuf__ T *sinUb = (__ubuf__ T *)sinTensor.GetPhyAddr();
    __ubuf__ T *cosUb = (__ubuf__ T *)cosTensor.GetPhyAddr();
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ T *outUb = (__ubuf__ T *)outTensor.GetPhyAddr();
    uint32_t quarterD = dLen / QUARTER_MODE_COEF;
    uint32_t quarterDAlign = ops::CeilAlign(quarterD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t repeatTimes = ops::CeilDiv(quarterD, VL_FLOAT32_SIZE);
    __ubuf__ T *currInUb;
    __ubuf__ T *currOutUb;
    __ubuf__ T *currSinUb;
    __ubuf__ T *currCosUb;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vregIn;
        Reg::RegTensor<float> vregQ1In;
        Reg::RegTensor<float> vregQ2In;
        Reg::RegTensor<float> vregQ3In;
        Reg::RegTensor<float> vregSin;
        Reg::RegTensor<float> vregQ1Sin;
        Reg::RegTensor<float> vregQ2Sin;
        Reg::RegTensor<float> vregQ3Sin;
        Reg::RegTensor<float> vregCos;
        Reg::RegTensor<float> vregQ1Cos;
        Reg::RegTensor<float> vregQ2Cos;
        Reg::RegTensor<float> vregQ3Cos;
        Reg::RegTensor<float> vregOut;
        Reg::RegTensor<float> vregQ1Out;
        Reg::RegTensor<float> vregQ2Out;
        Reg::RegTensor<float> vregQ3Out;
        Reg::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlign;
            currCosUb = cosUb + sIdx * dAlign;
            for (uint16_t row = 0; row < currDNum; row++) {
                currInUb = inUb + (sIdx * currDNum + row) * dAlign;
                currOutUb = outUb + (sIdx * currDNum + row) * dAlign;
                uint32_t updateCnt = quarterD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = Reg::UpdateMask<float>(updateCnt);
                    uint32_t offset = i * VL_FLOAT32_SIZE;
                    uint32_t q1Offset = offset + quarterDAlign;
                    uint32_t q2Offset = q1Offset + quarterDAlign;
                    uint32_t q3Offset = q2Offset + quarterDAlign;
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

                    Reg::Mul(vregSin, vregSin, vregQ1In, preg);
                    Reg::Mul(vregQ1Out, vregQ1Sin, vregIn, preg);
                    Reg::Mul(vregQ2Sin, vregQ2Sin, vregQ3In, preg);
                    Reg::Mul(vregQ3Out, vregQ3Sin, vregQ2In, preg);
                    Reg::Mul(vregCos, vregCos, vregIn, preg);
                    Reg::Sub(vregOut, vregCos, vregSin, preg);
                    Reg::Mul(vregQ1Cos, vregQ1Cos, vregQ1In, preg);
                    Reg::Add(vregQ1Out, vregQ1Out, vregQ1Cos, preg);
                    Reg::Mul(vregQ2Cos, vregQ2Cos, vregQ2In, preg);
                    Reg::Sub(vregQ2Out, vregQ2Cos, vregQ2Sin, preg);
                    Reg::Mul(vregQ3Cos, vregQ3Cos, vregQ3In, preg);
                    Reg::Add(vregQ3Out, vregQ3Out, vregQ3Cos, preg);

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
__aicore__ inline void InterleaveModeVF(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                        const LocalTensor<T> &inTensor, const LocalTensor<T> &outTensor, uint32_t dLen,
                                        uint16_t currSNum, uint16_t currDNum)
{
    __ubuf__ T *sinUb = (__ubuf__ T *)sinTensor.GetPhyAddr();
    __ubuf__ T *cosUb = (__ubuf__ T *)cosTensor.GetPhyAddr();
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ T *outUb = (__ubuf__ T *)outTensor.GetPhyAddr();
    uint16_t repeatTimes = dLen / VL_FLOAT32_SIZE;
    uint32_t dAlignLen = ops::CeilAlign(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t loopNum = repeatTimes / 2;
    uint32_t tailNum = dLen - loopNum * 2 * VL_FLOAT32_SIZE;
    uint16_t tailTwoVL = tailNum / VL_FLOAT32_SIZE;
    uint16_t tailOneVL = (tailTwoVL == 1) ? 0 : 1;
    uint32_t tailLen = tailNum % VL_FLOAT32_SIZE;
    __ubuf__ T *currInUb;
    __ubuf__ T *currOutUb;
    __ubuf__ T *currSinUb;
    __ubuf__ T *currCosUb;
    __ubuf__ T *tailSinUb;
    __ubuf__ T *tailCosUb;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vregFormerCos;
        Reg::RegTensor<float> vregLatterCos;
        Reg::RegTensor<float> vregFormerSin;
        Reg::RegTensor<float> vregLatterSin;
        Reg::RegTensor<float> vregFormerIn;
        Reg::RegTensor<float> vregLatterIn;
        Reg::RegTensor<float> vregOdd;
        Reg::RegTensor<float> vregEven;
        Reg::RegTensor<float> vregFormerOut;
        Reg::RegTensor<float> vregLatterOut;
        Reg::MaskReg pregLoop;
        Reg::MaskReg pregTail;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlignLen;
            currCosUb = cosUb + sIdx * dAlignLen;
            for (uint16_t idxD = 0; idxD < currDNum; idxD++) {
                uint32_t updateCnt = dLen;
                currInUb = inUb + (sIdx * currDNum + idxD) * dAlignLen;
                currOutUb = outUb + (sIdx * currDNum + idxD) * dAlignLen;
                pregLoop = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
                for (uint16_t i = 0; i < loopNum; i++) {
                    uint32_t evenOffSet = (i * 2) * VL_FLOAT32_SIZE;
                    uint32_t oddOffset = evenOffSet + VL_FLOAT32_SIZE;
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregFormerIn, pregLoop, evenOffSet);
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregLatterIn, pregLoop, oddOffset);
                    ops::LoadOneTensorForDtypeT<T>(currCosUb, vregFormerCos, pregLoop, evenOffSet);
                    ops::LoadOneTensorForDtypeT<T>(currCosUb, vregLatterCos, pregLoop, oddOffset);
                    ops::LoadOneTensorForDtypeT<T>(currSinUb, vregFormerSin, pregLoop, evenOffSet);
                    ops::LoadOneTensorForDtypeT<T>(currSinUb, vregLatterSin, pregLoop, oddOffset);
                    Reg::Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregLoop);
                    Reg::Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregLoop);
                    Reg::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                    Reg::Muls(vregOdd, vregOdd, float(-1.0), pregLoop);
                    Reg::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                    Reg::Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregLoop);
                    Reg::Add(vregFormerCos, vregFormerCos, vregFormerSin, pregLoop);
                    Reg::Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregLoop);
                    Reg::Add(vregLatterCos, vregLatterCos, vregLatterSin, pregLoop);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregFormerCos, pregLoop, evenOffSet);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregLatterCos, pregLoop, oddOffset);
                }

                currInUb = inUb + (sIdx * currDNum + idxD) * dAlignLen + (loopNum * 2 * VL_FLOAT32_SIZE);
                currOutUb = outUb + (sIdx * currDNum + idxD) * dAlignLen + (loopNum * 2 * VL_FLOAT32_SIZE);
                tailSinUb = currSinUb + loopNum * 2 * VL_FLOAT32_SIZE;
                tailCosUb = currCosUb + loopNum * 2 * VL_FLOAT32_SIZE;
                // 尾块大于VL时,读取一个VL，读取尾块
                for (uint16_t i = 0; i < tailTwoVL; i++) {
                    uint32_t updateCnt = tailLen;
                    pregTail = Reg::UpdateMask<float>(updateCnt);
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregFormerIn, pregLoop, 0);
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregLatterIn, pregTail, VL_FLOAT32_SIZE);
                    ops::LoadOneTensorForDtypeT<T>(tailCosUb, vregFormerCos, pregLoop, 0);
                    ops::LoadOneTensorForDtypeT<T>(tailCosUb, vregLatterCos, pregTail, VL_FLOAT32_SIZE);
                    ops::LoadOneTensorForDtypeT<T>(tailSinUb, vregFormerSin, pregLoop, 0);
                    ops::LoadOneTensorForDtypeT<T>(tailSinUb, vregLatterSin, pregTail, VL_FLOAT32_SIZE);
                    Reg::Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregLoop);
                    Reg::Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregTail);
                    Reg::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                    Reg::Muls(vregOdd, vregOdd, float(-1.0), pregLoop);
                    Reg::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                    Reg::Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregLoop);
                    Reg::Add(vregFormerCos, vregFormerCos, vregFormerSin, pregLoop);
                    Reg::Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregTail);
                    Reg::Add(vregLatterCos, vregLatterCos, vregLatterSin, pregTail);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregFormerCos, pregLoop, 0);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregLatterCos, pregTail, VL_FLOAT32_SIZE);
                }

                // 尾块小于VL时,只读取VL
                for (uint16_t i = 0; i < tailOneVL; i++) {
                    uint32_t updateCnt = tailLen;
                    pregTail = Reg::UpdateMask<float>(updateCnt);
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregFormerIn, pregTail, 0);
                    ops::LoadOneTensorForDtypeT<T>(tailCosUb, vregFormerCos, pregTail, 0);
                    ops::LoadOneTensorForDtypeT<T>(tailSinUb, vregFormerSin, pregTail, 0);
                    Reg::Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregTail);
                    Reg::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                    Reg::Muls(vregOdd, vregOdd, float(-1.0), pregTail);
                    Reg::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                    Reg::Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregTail);
                    Reg::Add(vregFormerCos, vregFormerCos, vregFormerSin, pregTail);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregFormerCos, pregTail, 0);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void DeepSeekInterleaveModeVF(const LocalTensor<T> &sinTensor, const LocalTensor<T> &cosTensor,
                                                const LocalTensor<T> &inTensor, const LocalTensor<T> &outTensor,
                                                uint32_t dLen, uint16_t currSNum, uint16_t currDNum)
{
    __ubuf__ T *sinUb = (__ubuf__ T *)sinTensor.GetPhyAddr();
    __ubuf__ T *cosUb = (__ubuf__ T *)cosTensor.GetPhyAddr();
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ T *outUb = (__ubuf__ T *)outTensor.GetPhyAddr();
    uint32_t dAlign = ops::CeilAlign(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint32_t halfDAlign = ops::CeilAlign(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t repeatTimes = halfD / VL_FLOAT32_SIZE;
    uint32_t tailTwoNum = dLen - repeatTimes * VL_FLOAT32_SIZE * HALF_INTERLEAVE_COEF;
    uint16_t tailTwoVL = tailTwoNum > VL_FLOAT32_SIZE ? 1 : 0;
    uint16_t tailOneVL = tailTwoNum > 0 ? (1 - tailTwoVL) : 0;
    uint32_t halfTailNum = tailTwoNum / HALF_INTERLEAVE_COEF;
    uint32_t tailNum = tailTwoNum - tailTwoVL * VL_FLOAT32_SIZE;
    __ubuf__ T *currInUb;
    __ubuf__ T *currOutUb;
    __ubuf__ T *currSinUb;
    __ubuf__ T *currCosUb;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vregIn;
        Reg::RegTensor<float> vregHalfIn;
        Reg::RegTensor<float> vregSin;
        Reg::RegTensor<float> vregHalfSin;
        Reg::RegTensor<float> vregCos;
        Reg::RegTensor<float> vregHalfCos;
        Reg::RegTensor<float> vregOut;
        Reg::RegTensor<float> vregHalfOut;
        Reg::MaskReg pregTail;
        Reg::MaskReg pregHalfTail;
        Reg::MaskReg pregFull = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * halfDAlign * HALF_INTERLEAVE_COEF;
            currCosUb = cosUb + sIdx * halfDAlign * HALF_INTERLEAVE_COEF;
            uint32_t updateTailNum = tailNum;
            uint32_t updateHalfTailNum = halfTailNum;
            pregTail = Reg::UpdateMask<float>(updateTailNum);
            pregHalfTail = Reg::UpdateMask<float>(updateHalfTailNum);
            for (uint16_t row = 0; row < currDNum; row++) {
                currInUb = inUb + (sIdx * currDNum + row) * dAlign;
                currOutUb = outUb + (sIdx * currDNum + row) * halfDAlign * HALF_INTERLEAVE_COEF;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    uint32_t offset = i * VL_FLOAT32_SIZE;
                    uint32_t halfOffset = offset + halfDAlign;
                    uint32_t inOffset = offset * HALF_INTERLEAVE_COEF;
                    ops::LoadTwoTensorForDtypeT<T>(currInUb, currInUb, vregIn, vregHalfIn, pregFull, pregFull, inOffset,
                                                   inOffset + VL_FLOAT32_SIZE);
                    ops::LoadTwoTensorForDtypeT<T>(currSinUb, currSinUb, vregSin, vregHalfSin, pregFull, pregFull,
                                                   offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currCosUb, currCosUb, vregCos, vregHalfCos, pregFull, pregFull,
                                                   offset, halfOffset);

                    Reg::DeInterleave<float>(vregIn, vregHalfIn, vregIn, vregHalfIn);

                    Reg::Mul(vregOut, vregCos, vregIn, pregFull);
                    Reg::Mul(vregHalfOut, vregHalfCos, vregHalfIn, pregFull);
                    Reg::Muls(vregHalfIn, vregHalfIn, float(-1.0), pregFull);
                    Reg::Mul(vregSin, vregSin, vregHalfIn, pregFull);
                    Reg::Add(vregOut, vregOut, vregSin, pregFull);
                    Reg::Mul(vregHalfSin, vregHalfSin, vregIn, pregFull);
                    Reg::Add(vregHalfOut, vregHalfOut, vregHalfSin, pregFull);

                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregOut, pregFull, offset);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregHalfOut, pregFull, halfOffset);
                }

                for (uint16_t i = 0; i < tailTwoVL; i++) {
                    uint32_t offset = repeatTimes * VL_FLOAT32_SIZE;
                    uint32_t halfOffset = offset + halfDAlign;
                    uint32_t inOffset = offset * HALF_INTERLEAVE_COEF;
                    ops::LoadTwoTensorForDtypeT<T>(currInUb, currInUb, vregIn, vregHalfIn, pregFull, pregTail, inOffset,
                                                   inOffset + VL_FLOAT32_SIZE);
                    ops::LoadTwoTensorForDtypeT<T>(currSinUb, currSinUb, vregSin, vregHalfSin, pregHalfTail,
                                                   pregHalfTail, offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currCosUb, currCosUb, vregCos, vregHalfCos, pregHalfTail,
                                                   pregHalfTail, offset, halfOffset);

                    Reg::DeInterleave<float>(vregIn, vregHalfIn, vregIn, vregHalfIn);

                    Reg::Mul(vregOut, vregCos, vregIn, pregHalfTail);
                    Reg::Mul(vregHalfOut, vregHalfCos, vregHalfIn, pregHalfTail);
                    Reg::Muls(vregHalfIn, vregHalfIn, float(-1.0), pregHalfTail);
                    Reg::Mul(vregSin, vregSin, vregHalfIn, pregHalfTail);
                    Reg::Add(vregOut, vregOut, vregSin, pregHalfTail);
                    Reg::Mul(vregHalfSin, vregHalfSin, vregIn, pregHalfTail);
                    Reg::Add(vregHalfOut, vregHalfOut, vregHalfSin, pregHalfTail);

                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregOut, pregHalfTail, offset);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregHalfOut, pregHalfTail, halfOffset);
                }

                for (uint16_t i = 0; i < tailOneVL; i++) {
                    uint32_t offset = repeatTimes * VL_FLOAT32_SIZE;
                    uint32_t halfOffset = offset + halfDAlign;
                    uint32_t inOffset = offset * HALF_INTERLEAVE_COEF;
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregIn, pregTail, inOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currSinUb, currSinUb, vregSin, vregHalfSin, pregHalfTail,
                                                   pregHalfTail, offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currCosUb, currCosUb, vregCos, vregHalfCos, pregHalfTail,
                                                   pregHalfTail, offset, halfOffset);

                    Reg::DeInterleave<float>(vregIn, vregHalfIn, vregIn, vregHalfIn);
                    Reg::Mul(vregOut, vregCos, vregIn, pregHalfTail);
                    Reg::Mul(vregHalfOut, vregHalfCos, vregHalfIn, pregHalfTail);
                    Reg::Muls(vregHalfIn, vregHalfIn, float(-1.0), pregHalfTail);
                    Reg::Mul(vregSin, vregSin, vregHalfIn, pregHalfTail);
                    Reg::Add(vregOut, vregOut, vregSin, pregHalfTail);
                    Reg::Mul(vregHalfSin, vregHalfSin, vregIn, pregHalfTail);
                    Reg::Add(vregHalfOut, vregHalfOut, vregHalfSin, pregHalfTail);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregOut, pregHalfTail, offset);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregHalfOut, pregHalfTail, halfOffset);
                }
            }
        }
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void BatchHalfAlignVF(__ubuf__ T *in, __ubuf__ T *cos, __ubuf__ T *sin, __ubuf__ T *out,
                                        uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t d, int64_t dAlign,
                                        int64_t ubFactorS, int64_t ubFactorN)
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
        Reg::RegTensor<float> inPart1Reg;
        Reg::RegTensor<float> inPart2Reg;
        Reg::RegTensor<float> cosPart1Reg;
        Reg::RegTensor<float> cosPart2Reg;
        Reg::RegTensor<float> sinPart1Reg;
        Reg::RegTensor<float> sinPart2Reg;
        Reg::MaskReg pregLoop;
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
                        pregLoop = Reg::UpdateMask<float>(count);
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
                        Reg::Mul(cosPart1Reg, inPart1Reg, cosPart1Reg, pregLoop);
                        Reg::Mul(sinPart1Reg, inPart2Reg, sinPart1Reg, pregLoop);
                        Reg::Sub(cosPart1Reg, cosPart1Reg, sinPart1Reg, pregLoop);
                        Reg::Mul(cosPart2Reg, inPart2Reg, cosPart2Reg, pregLoop);
                        Reg::Mul(sinPart2Reg, sinPart2Reg, inPart1Reg, pregLoop);
                        Reg::Add(cosPart2Reg, cosPart2Reg, sinPart2Reg, pregLoop);
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
__aicore__ inline void BatchQuarterAlignVF(__ubuf__ T *in, __ubuf__ T *cos, __ubuf__ T *sin, __ubuf__ T *out,
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
        Reg::RegTensor<float> inPart1Reg;
        Reg::RegTensor<float> inPart2Reg;
        Reg::RegTensor<float> inPart3Reg;
        Reg::RegTensor<float> inPart4Reg;
        Reg::RegTensor<float> cosPart1Reg;
        Reg::RegTensor<float> cosPart2Reg;
        Reg::RegTensor<float> cosPart3Reg;
        Reg::RegTensor<float> cosPart4Reg;
        Reg::RegTensor<float> sinPart1Reg;
        Reg::RegTensor<float> sinPart2Reg;
        Reg::RegTensor<float> sinPart3Reg;
        Reg::RegTensor<float> sinPart4Reg;
        Reg::MaskReg pregLoop;
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
                        pregLoop = Reg::UpdateMask<float>(count);
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
                        Reg::Mul(cosPart1Reg, inPart1Reg, cosPart1Reg, pregLoop);
                        Reg::Mul(sinPart1Reg, inPart2Reg, sinPart1Reg, pregLoop);
                        Reg::Sub(cosPart1Reg, cosPart1Reg, sinPart1Reg, pregLoop);
                        Reg::Mul(cosPart2Reg, inPart2Reg, cosPart2Reg, pregLoop);
                        Reg::Mul(sinPart2Reg, sinPart2Reg, inPart1Reg, pregLoop);
                        Reg::Add(cosPart2Reg, cosPart2Reg, sinPart2Reg, pregLoop);
                        Reg::Mul(cosPart3Reg, inPart3Reg, cosPart3Reg, pregLoop);
                        Reg::Mul(sinPart3Reg, inPart4Reg, sinPart3Reg, pregLoop);
                        Reg::Sub(cosPart3Reg, cosPart3Reg, sinPart3Reg, pregLoop);
                        Reg::Mul(cosPart4Reg, inPart4Reg, cosPart4Reg, pregLoop);
                        Reg::Mul(sinPart4Reg, sinPart4Reg, inPart3Reg, pregLoop);
                        Reg::Add(cosPart4Reg, cosPart4Reg, sinPart4Reg, pregLoop);
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
__aicore__ inline void BatchInterleaveModeVF(__ubuf__ T *in, __ubuf__ T *cos, __ubuf__ T *sin, __ubuf__ T *out,
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
        Reg::RegTensor<float> inPart1Reg;
        Reg::RegTensor<float> inPart2Reg;
        Reg::RegTensor<float> cosPart1Reg;
        Reg::RegTensor<float> cosPart2Reg;
        Reg::RegTensor<float> sinPart1Reg;
        Reg::RegTensor<float> sinPart2Reg;
        Reg::MaskReg pregLoop;
        Reg::MaskReg pregPart1;
        Reg::MaskReg pregPart2;
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
                        pregLoop = Reg::UpdateMask<float>(halfCnt);
                        pregPart1 = Reg::UpdateMask<float>(part1Cnt);
                        pregPart2 = Reg::UpdateMask<float>(part2Cnt);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart2Reg, pregPart2, i * loopSize + VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart2Reg, pregPart2,
                                                       i * loopSize + VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart2Reg, pregPart2,
                                                       i * loopSize + VL_FLOAT32_SIZE);
                        Reg::Mul(cosPart1Reg, cosPart1Reg, inPart1Reg, pregPart1);
                        Reg::Mul(cosPart2Reg, cosPart2Reg, inPart2Reg, pregPart2);
                        Reg::DeInterleave<float>(inPart1Reg, inPart2Reg, inPart1Reg, inPart2Reg);
                        Reg::Muls(inPart2Reg, inPart2Reg, float(-1.0), pregLoop);
                        Reg::Interleave<float>(inPart1Reg, inPart2Reg, inPart2Reg, inPart1Reg);
                        Reg::Mul(sinPart1Reg, sinPart1Reg, inPart1Reg, pregPart1);
                        Reg::Add(cosPart1Reg, cosPart1Reg, sinPart1Reg, pregPart1);
                        Reg::Mul(sinPart2Reg, sinPart2Reg, inPart2Reg, pregPart2);
                        Reg::Add(cosPart2Reg, cosPart2Reg, sinPart2Reg, pregPart2);
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
__aicore__ inline void BatchDeepSeekInterleaveModeVF(__ubuf__ T *in, __ubuf__ T *cos, __ubuf__ T *sin, __ubuf__ T *out,
                                                     uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t d,
                                                     int64_t dAlign, int64_t ubFactorS, int64_t ubFactorN)
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
        Reg::RegTensor<float> inPart1Reg;
        Reg::RegTensor<float> inPart2Reg;
        Reg::RegTensor<float> cosPart1Reg;
        Reg::RegTensor<float> cosPart2Reg;
        Reg::RegTensor<float> sinPart1Reg;
        Reg::RegTensor<float> sinPart2Reg;
        Reg::MaskReg pregLoop;
        Reg::MaskReg pregPart1;
        Reg::MaskReg pregPart2;
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
                        pregLoop = Reg::UpdateMask<float>(halfCnt);
                        pregPart1 = Reg::UpdateMask<float>(part1Cnt);
                        pregPart2 = Reg::UpdateMask<float>(part2Cnt);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart2Reg, pregPart2, i * loopSize + VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart2Reg, pregLoop,
                                                       i * VL_FLOAT32_SIZE + dHalfOffset);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart2Reg, pregLoop,
                                                       i * VL_FLOAT32_SIZE + dHalfOffset);
                        Reg::DeInterleave<float>(inPart1Reg, inPart2Reg, inPart1Reg, inPart2Reg);
                        Reg::Mul(cosPart1Reg, cosPart1Reg, inPart1Reg, pregLoop);
                        Reg::Mul(cosPart2Reg, cosPart2Reg, inPart2Reg, pregLoop);
                        Reg::Muls(inPart2Reg, inPart2Reg, float(-1.0), pregLoop);
                        Reg::Mul(sinPart1Reg, sinPart1Reg, inPart2Reg, pregLoop);
                        Reg::Add(cosPart1Reg, cosPart1Reg, sinPart1Reg, pregLoop);
                        Reg::Mul(sinPart2Reg, sinPart2Reg, inPart1Reg, pregLoop);
                        Reg::Add(cosPart2Reg, cosPart2Reg, sinPart2Reg, pregLoop);
                        ops::StoreOneTensorForDtypeT<T>(currOutUb, cosPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::StoreOneTensorForDtypeT<T>(currOutUb, cosPart2Reg, pregLoop,
                                                        i * VL_FLOAT32_SIZE + dHalfOffset);
                    }
                }
            }
        }
    }
}

// Mixed precision: TX is half/bfloat16 for input, cos/sin are float
template <typename TX>
__aicore__ inline void InterleaveModeVFMixed(const LocalTensor<TX> &inTensor, const LocalTensor<float> &cosTensor,
                                             const LocalTensor<float> &sinTensor, const LocalTensor<TX> &outTensor,
                                             uint32_t dLen, uint16_t currSNum, uint16_t currDNum)
{
    __ubuf__ TX *inUb = (__ubuf__ TX *)inTensor.GetPhyAddr();
    __ubuf__ float *cosUb = (__ubuf__ float *)cosTensor.GetPhyAddr();
    __ubuf__ float *sinUb = (__ubuf__ float *)sinTensor.GetPhyAddr();
    __ubuf__ TX *outUb = (__ubuf__ TX *)outTensor.GetPhyAddr();
    uint16_t repeatTimes = dLen / VL_FLOAT32_SIZE;
    uint32_t dAlignLen = ops::CeilAlign(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(TX)));
    uint32_t dAlignLenFloat = ops::CeilAlign(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(float)));
    uint16_t loopNum = repeatTimes / 2;
    uint32_t tailNum = dLen - loopNum * 2 * VL_FLOAT32_SIZE;
    uint16_t tailTwoVL = tailNum / VL_FLOAT32_SIZE;
    uint16_t tailOneVL = (tailTwoVL == 1) ? 0 : 1;
    uint32_t tailLen = tailNum % VL_FLOAT32_SIZE;
    __ubuf__ TX *currInUb;
    __ubuf__ TX *currOutUb;
    __ubuf__ float *currSinUb;
    __ubuf__ float *currCosUb;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vregFormerCos;
        Reg::RegTensor<float> vregLatterCos;
        Reg::RegTensor<float> vregFormerSin;
        Reg::RegTensor<float> vregLatterSin;
        Reg::RegTensor<float> vregFormerIn;
        Reg::RegTensor<float> vregLatterIn;
        Reg::RegTensor<float> vregOdd;
        Reg::RegTensor<float> vregEven;
        Reg::MaskReg pregLoop;
        Reg::MaskReg pregTail;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlignLenFloat;
            currCosUb = cosUb + sIdx * dAlignLenFloat;
            for (uint16_t idxD = 0; idxD < currDNum; idxD++) {
                currInUb = inUb + (sIdx * currDNum + idxD) * dAlignLen;
                currOutUb = outUb + (sIdx * currDNum + idxD) * dAlignLen;
                pregLoop = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
                for (uint16_t i = 0; i < loopNum; i++) {
                    uint32_t evenOffSet = (i * 2) * VL_FLOAT32_SIZE;
                    uint32_t oddOffset = evenOffSet + VL_FLOAT32_SIZE;
                    ops::LoadOneTensorForDtypeT<TX>(currInUb, vregFormerIn, pregLoop, evenOffSet);
                    ops::LoadOneTensorForDtypeT<TX>(currInUb, vregLatterIn, pregLoop, oddOffset);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vregFormerCos, currCosUb + evenOffSet);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vregLatterCos, currCosUb + oddOffset);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vregFormerSin, currSinUb + evenOffSet);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vregLatterSin, currSinUb + oddOffset);
                    Reg::Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregLoop);
                    Reg::Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregLoop);
                    Reg::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                    Reg::Muls(vregOdd, vregOdd, float(-1.0), pregLoop);
                    Reg::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                    Reg::Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregLoop);
                    Reg::Add(vregFormerCos, vregFormerCos, vregFormerSin, pregLoop);
                    Reg::Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregLoop);
                    Reg::Add(vregLatterCos, vregLatterCos, vregLatterSin, pregLoop);
                    ops::StoreOneTensorForDtypeT<TX>(currOutUb, vregFormerCos, pregLoop, evenOffSet);
                    ops::StoreOneTensorForDtypeT<TX>(currOutUb, vregLatterCos, pregLoop, oddOffset);
                }

                currInUb = inUb + (sIdx * currDNum + idxD) * dAlignLen + (loopNum * 2 * VL_FLOAT32_SIZE);
                currOutUb = outUb + (sIdx * currDNum + idxD) * dAlignLen + (loopNum * 2 * VL_FLOAT32_SIZE);
                __ubuf__ float *tailSinUb = currSinUb + loopNum * 2 * VL_FLOAT32_SIZE;
                __ubuf__ float *tailCosUb = currCosUb + loopNum * 2 * VL_FLOAT32_SIZE;
                for (uint16_t i = 0; i < tailTwoVL; i++) {
                    uint32_t updateCnt = tailLen;
                    pregTail = Reg::UpdateMask<float>(updateCnt);
                    ops::LoadOneTensorForDtypeT<TX>(currInUb, vregFormerIn, pregLoop, 0);
                    ops::LoadOneTensorForDtypeT<TX>(currInUb, vregLatterIn, pregTail, VL_FLOAT32_SIZE);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vregFormerCos, tailCosUb);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vregLatterCos, tailCosUb + VL_FLOAT32_SIZE);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vregFormerSin, tailSinUb);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vregLatterSin, tailSinUb + VL_FLOAT32_SIZE);
                    Reg::Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregLoop);
                    Reg::Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregTail);
                    Reg::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                    Reg::Muls(vregOdd, vregOdd, float(-1.0), pregLoop);
                    Reg::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                    Reg::Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregLoop);
                    Reg::Add(vregFormerCos, vregFormerCos, vregFormerSin, pregLoop);
                    Reg::Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregTail);
                    Reg::Add(vregLatterCos, vregLatterCos, vregLatterSin, pregTail);
                    ops::StoreOneTensorForDtypeT<TX>(currOutUb, vregFormerCos, pregLoop, 0);
                    ops::StoreOneTensorForDtypeT<TX>(currOutUb, vregLatterCos, pregTail, VL_FLOAT32_SIZE);
                }

                for (uint16_t i = 0; i < tailOneVL; i++) {
                    uint32_t updateCnt = tailLen;
                    pregTail = Reg::UpdateMask<float>(updateCnt);
                    ops::LoadOneTensorForDtypeT<TX>(currInUb, vregFormerIn, pregTail, 0);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vregFormerCos, tailCosUb);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vregFormerSin, tailSinUb);
                    Reg::Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregTail);
                    Reg::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                    Reg::Muls(vregOdd, vregOdd, float(-1.0), pregTail);
                    Reg::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                    Reg::Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregTail);
                    Reg::Add(vregFormerCos, vregFormerCos, vregFormerSin, pregTail);
                    ops::StoreOneTensorForDtypeT<TX>(currOutUb, vregFormerCos, pregTail, 0);
                }
            }
        }
    }
}

// Mixed precision BatchInterleaveModeVF for ABA layout
template <typename TX, bool IsBBoardcast>
__aicore__ inline void BatchInterleaveModeVFMixed(__ubuf__ TX *in, __ubuf__ float *cos, __ubuf__ float *sin,
                                                  __ubuf__ TX *out, uint16_t sLength, uint16_t bLength,
                                                  uint16_t nLength, int64_t d, int64_t dAlign, int64_t dAlignFloat,
                                                  int64_t ubFactorS, int64_t ubFactorN)
{
    uint32_t loopSize = 2 * VL_FLOAT32_SIZE;
    uint16_t dLoopCount = (d + loopSize - 1) / loopSize;

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
    int32_t cosSinBStepUb = ubFactorS * dAlignFloat;
    int32_t cosSinSStepUb = dAlignFloat;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> inPart1Reg;
        Reg::RegTensor<float> inPart2Reg;
        Reg::RegTensor<float> cosPart1Reg;
        Reg::RegTensor<float> cosPart2Reg;
        Reg::RegTensor<float> sinPart1Reg;
        Reg::RegTensor<float> sinPart2Reg;
        Reg::MaskReg pregLoop;
        Reg::MaskReg pregPart1;
        Reg::MaskReg pregPart2;
        __ubuf__ TX *currInUb, *currOutUb;
        __ubuf__ float *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t halfCnt = halfNum;
                    uint32_t part1Cnt = part1Num;
                    uint32_t part2Cnt = part2Num;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    currOutUb = out + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * cosSinSStepUb;
                        currSinUb = sin + sIdx * cosSinSStepUb;
                    } else {
                        currCosUb = cos + bIdx * cosSinBStepUb + sIdx * cosSinSStepUb;
                        currSinUb = sin + bIdx * cosSinBStepUb + sIdx * cosSinSStepUb;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = Reg::UpdateMask<float>(halfCnt);
                        pregPart1 = Reg::UpdateMask<float>(part1Cnt);
                        pregPart2 = Reg::UpdateMask<float>(part2Cnt);
                        // Load/Store的offset是TX元素偏移，需与cos/sin逻辑下标对齐，不能按dtype字节数换算。
                        ops::LoadOneTensorForDtypeT<TX>(currInUb, inPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<TX>(currInUb, inPart2Reg, pregPart2,
                                                        i * loopSize + VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<float>(currCosUb, cosPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<float>(currCosUb, cosPart2Reg, pregPart2,
                                                           i * loopSize + VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<float>(currSinUb, sinPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<float>(currSinUb, sinPart2Reg, pregPart2,
                                                           i * loopSize + VL_FLOAT32_SIZE);
                        Reg::Mul(cosPart1Reg, cosPart1Reg, inPart1Reg, pregPart1);
                        Reg::Mul(cosPart2Reg, cosPart2Reg, inPart2Reg, pregPart2);
                        Reg::DeInterleave<float>(inPart1Reg, inPart2Reg, inPart1Reg, inPart2Reg);
                        Reg::Muls(inPart2Reg, inPart2Reg, float(-1.0), pregLoop);
                        Reg::Interleave<float>(inPart1Reg, inPart2Reg, inPart2Reg, inPart1Reg);
                        Reg::Mul(sinPart1Reg, sinPart1Reg, inPart1Reg, pregPart1);
                        Reg::Add(cosPart1Reg, cosPart1Reg, sinPart1Reg, pregPart1);
                        Reg::Mul(sinPart2Reg, sinPart2Reg, inPart2Reg, pregPart2);
                        Reg::Add(cosPart2Reg, cosPart2Reg, sinPart2Reg, pregPart2);
                        ops::StoreOneTensorForDtypeT<TX>(currOutUb, cosPart1Reg, pregPart1, i * loopSize);
                        ops::StoreOneTensorForDtypeT<TX>(currOutUb, cosPart2Reg, pregPart2,
                                                         i * loopSize + VL_FLOAT32_SIZE);
                    }
                }
            }
        }
    }
}

#endif // APPLY_ROTARY_POS_EMB_COMMON_H
