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
 * \file rotary_position_embedding_grad_rotary_x_vf.h
 * \brief
 */
#ifndef __ROTARY_POSITION_EMBEDDING_GRAD_ROTARY_X_VF__
#define __ROTARY_POSITION_EMBEDDING_GRAD_ROTARY_X_VF__

#include "kernel_operator.h"
#include "op_kernel/load_store_utils.h"

using namespace AscendC;
namespace RotaryPositionEmbeddingGrad {

constexpr uint32_t BLOCK_TYPE_SIZE = Ops::Base::GetUbBlockSize();
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
    x = [-x[1], x[0]]
*/
template <typename T>
__aicore__ inline void HalfRotaryVF(const LocalTensor<T> &inTensor, const LocalTensor<T> &rotaryTensor,
                                    const uint32_t dLen, const uint32_t dAlign, const uint16_t currDNum)
{
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ T *outUb = (__ubuf__ T *)rotaryTensor.GetPhyAddr();
    __ubuf__ T *currInUb;
    __ubuf__ T *currOutUb;
    uint32_t vecLen = Ops::Base::GetVRegSize() / sizeof(T);
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint32_t halfDAlign = Ops::Base::CeilAlign(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t repeatTimes = Ops::Base::CeilDiv(halfD, vecLen);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregIn;
        MicroAPI::RegTensor<T> vregHalfIn;
        MicroAPI::RegTensor<T> vregOut;
        MicroAPI::RegTensor<T> vregHalfOut;
        MicroAPI::RegTensor<T> vregNeg;
        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg;
        Duplicate(vregNeg, static_cast<T>(-1.0), pregAll);
        for (uint16_t idxD = 0; idxD < currDNum; idxD++) {
            currInUb = inUb + idxD * dAlign;
            currOutUb = outUb + idxD * dAlign;
            uint32_t updateCnt = halfD;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = MicroAPI::UpdateMask<T>(updateCnt);
                int32_t offset = i * vecLen;
                int32_t halfOffset = offset + halfDAlign;
                MicroAPI::LoadAlign(vregIn, currInUb + offset);
                MicroAPI::LoadAlign(vregHalfIn, currInUb + halfOffset);
                MicroAPI::Mul(vregHalfIn, vregHalfIn, vregNeg, preg);
                MicroAPI::StoreAlign(currOutUb + offset, vregHalfIn, preg);
                MicroAPI::StoreAlign(currOutUb + halfOffset, vregIn, preg);
            }
        }
    }
}

/*
    x = [-q1, q0, -q3, q2]
*/
template <typename T>
__aicore__ inline void QuarterRotaryVF(const LocalTensor<T> &inTensor, const LocalTensor<T> &rotaryTensor,
                                       const uint32_t dLen, const uint32_t dAlign, const uint16_t currDNum)
{
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ T *outUb = (__ubuf__ T *)rotaryTensor.GetPhyAddr();
    __ubuf__ T *currInUb;
    __ubuf__ T *currOutUb;
    uint32_t vecLen = Ops::Base::GetVRegSize() / sizeof(T);
    uint32_t quarterD = dLen / QUARTER_MODE_COEF;
    uint32_t quarterDAlign = Ops::Base::CeilAlign(quarterD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t repeatTimes = Ops::Base::CeilDiv(quarterD, vecLen);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregIn;
        MicroAPI::RegTensor<T> vregQ1In;
        MicroAPI::RegTensor<T> vregQ2In;
        MicroAPI::RegTensor<T> vregQ3In;
        MicroAPI::RegTensor<T> vregOut;
        MicroAPI::RegTensor<T> vregQ1Out;
        MicroAPI::RegTensor<T> vregQ2Out;
        MicroAPI::RegTensor<T> vregQ3Out;
        MicroAPI::RegTensor<T> vregNeg;
        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg preg;
        Duplicate(vregNeg, static_cast<T>(-1.0), pregAll);
        for (uint16_t idxD = 0; idxD < currDNum; idxD++) {
            currInUb = inUb + idxD * dAlign;
            currOutUb = outUb + idxD * dAlign;
            uint32_t updateCnt = quarterD;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = MicroAPI::UpdateMask<T>(updateCnt);
                int32_t offset = i * vecLen;
                int32_t q1Offset = offset + quarterDAlign;
                int32_t q2Offset = q1Offset + quarterDAlign;
                int32_t q3Offset = q2Offset + quarterDAlign;
                MicroAPI::LoadAlign(vregIn, currInUb + offset);
                MicroAPI::LoadAlign(vregQ1In, currInUb + q1Offset);
                MicroAPI::LoadAlign(vregQ2In, currInUb + q2Offset);
                MicroAPI::LoadAlign(vregQ3In, currInUb + q3Offset);
                MicroAPI::Mul(vregQ1In, vregQ1In, vregNeg, preg);
                MicroAPI::Mul(vregQ3In, vregQ3In, vregNeg, preg);
                MicroAPI::StoreAlign(currOutUb + offset, vregQ1In, preg);
                MicroAPI::StoreAlign(currOutUb + q1Offset, vregIn, preg);
                MicroAPI::StoreAlign(currOutUb + q2Offset, vregQ3In, preg);
                MicroAPI::StoreAlign(currOutUb + q3Offset, vregQ2In, preg);
            }
        }
    }
}

/*
    x_even = x[...,::2]
    x_odd = x[..., 1::2]
    x = [x_odd, - x_even]
*/
template <typename T>
__aicore__ inline void InterleaveRotaryVF(const LocalTensor<T> &inTensor, const LocalTensor<T> &rotaryTensor,
                                          const uint32_t dLen, const uint16_t currDNum)
{
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ T *outUb = (__ubuf__ T *)rotaryTensor.GetPhyAddr();
    __ubuf__ T *currInUb;
    __ubuf__ T *currOutUb;
    uint32_t vecLen = Ops::Base::GetVRegSize() / sizeof(T);
    uint32_t dAlignLen = Ops::Base::CeilAlign(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint32_t loopSize = vecLen * HALF_INTERLEAVE_COEF;
    uint16_t dLoopCnt = Ops::Base::CeilDiv(dLen, loopSize);
    // 计算Mask参数
    uint32_t halfNum = dLen / HALF_INTERLEAVE_COEF;
    uint32_t part1Num = static_cast<uint32_t>(dLoopCnt - 1) * vecLen;
    uint32_t part2Num = part1Num;
    uint32_t tailNum = dLen - part1Num - part2Num;
    if (tailNum > vecLen) {
        part1Num += vecLen;
        part2Num += (tailNum - vecLen);
    } else {
        part1Num += tailNum;
    }

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregFormerIn;
        MicroAPI::RegTensor<T> vregLatterIn;
        MicroAPI::RegTensor<T> vregOdd;
        MicroAPI::RegTensor<T> vregEven;
        MicroAPI::RegTensor<T> vregNeg;
        MicroAPI::MaskReg pregLoop = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregPart1;
        MicroAPI::MaskReg pregPart2;
        Duplicate(vregNeg, static_cast<T>(-1.0), pregLoop);
        for (uint16_t idxD = 0; idxD < currDNum; idxD++) {
            int32_t dOffset = idxD * dAlignLen;
            currInUb = inUb + dOffset;
            currOutUb = outUb + dOffset;
            uint32_t part1Cnt = part1Num;
            uint32_t part2Cnt = part2Num;
            for (uint16_t i = 0; i < dLoopCnt; i++) {
                int32_t offset = i * loopSize;
                pregPart1 = MicroAPI::UpdateMask<T>(part1Cnt);
                pregPart2 = MicroAPI::UpdateMask<T>(part2Cnt);
                MicroAPI::LoadAlign(vregFormerIn, currInUb + offset);
                MicroAPI::LoadAlign(vregLatterIn, currInUb + offset + vecLen);
                MicroAPI::DeInterleave<T>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                MicroAPI::Mul(vregOdd, vregOdd, vregNeg, pregLoop);
                MicroAPI::Interleave<T>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                MicroAPI::StoreAlign(currOutUb + offset, vregFormerIn, pregPart1);
                MicroAPI::StoreAlign(currOutUb + offset + vecLen, vregLatterIn, pregPart2);
            }
        }
    }
}

template <typename T>
__aicore__ inline void DSDSinInterleaveHalfVF(const LocalTensor<T> &inTensor, const LocalTensor<T> &rotaryTensor,
                                              const uint32_t dLen, const uint16_t currDNum)
{
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ T *outUb = (__ubuf__ T *)rotaryTensor.GetPhyAddr();
    __ubuf__ T *currInUb;
    __ubuf__ T *currOutUb;

    uint32_t vecLen = Ops::Base::GetVRegSize() / sizeof(T);
    uint32_t dAlignLen = Ops::Base::CeilAlign(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint32_t halfDAlign = Ops::Base::CeilAlign(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint32_t loopSize = vecLen * HALF_INTERLEAVE_COEF;
    uint16_t dLoopCnt = Ops::Base::CeilDiv(dLen, loopSize);

    // 计算Mask参数
    uint32_t part1Num = static_cast<uint32_t>(dLoopCnt - 1) * vecLen;
    uint32_t part2Num = part1Num;
    uint32_t tailNum = dLen - part1Num - part2Num;
    part1Num += tailNum / HALF_INTERLEAVE_COEF;
    part2Num += tailNum / HALF_INTERLEAVE_COEF;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregFormerIn;
        MicroAPI::RegTensor<T> vregLatterIn;
        MicroAPI::RegTensor<T> vregOdd;
        MicroAPI::RegTensor<T> vregEven;
        MicroAPI::RegTensor<T> vregNeg;
        MicroAPI::MaskReg pregLoop = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregPart1;
        MicroAPI::MaskReg pregPart2;
        Duplicate(vregNeg, static_cast<T>(-1.0), pregLoop);
        for (uint16_t idxD = 0; idxD < currDNum; idxD++) {
            currInUb = inUb + idxD * dAlignLen;
            currOutUb = outUb + idxD * halfDAlign * HALF_INTERLEAVE_COEF;
            uint32_t part1Cnt = part1Num;
            uint32_t part2Cnt = part2Num;
            for (uint16_t i = 0; i < dLoopCnt; i++) {
                int32_t inOffset = i * loopSize;
                int32_t outOffset = i * vecLen;
                pregPart1 = MicroAPI::UpdateMask<T>(part1Cnt);
                pregPart2 = MicroAPI::UpdateMask<T>(part2Cnt);
                MicroAPI::LoadAlign(vregFormerIn, currInUb + inOffset);
                MicroAPI::LoadAlign(vregLatterIn, currInUb + inOffset + vecLen);
                MicroAPI::DeInterleave<T>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                MicroAPI::Mul(vregOdd, vregOdd, vregNeg, pregLoop);
                MicroAPI::StoreAlign(currOutUb + outOffset, vregOdd, pregPart1);
                MicroAPI::StoreAlign(currOutUb + outOffset + halfDAlign, vregEven, pregPart2);
            }
        }
    }
}

template <typename T>
__aicore__ inline void DSCosInterleaveHalfVF(const LocalTensor<T> &inTensor, const LocalTensor<T> &rotaryTensor,
                                             const uint32_t dLen, const uint16_t currDNum)
{
    __ubuf__ T *inUb = (__ubuf__ T *)inTensor.GetPhyAddr();
    __ubuf__ T *outUb = (__ubuf__ T *)rotaryTensor.GetPhyAddr();
    __ubuf__ T *currInUb;
    __ubuf__ T *currOutUb;

    uint32_t vecLen = Ops::Base::GetVRegSize() / sizeof(T);
    uint32_t dAlignLen = Ops::Base::CeilAlign(dLen, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint32_t halfDAlign = Ops::Base::CeilAlign(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint32_t loopSize = vecLen * HALF_INTERLEAVE_COEF;
    uint16_t dLoopCnt = Ops::Base::CeilDiv(dLen, loopSize);

    // 计算Mask参数
    uint32_t part1Num = static_cast<uint32_t>(dLoopCnt - 1) * vecLen;
    uint32_t part2Num = part1Num;
    uint32_t tailNum = dLen - part1Num - part2Num;
    part1Num += tailNum / HALF_INTERLEAVE_COEF;
    part2Num += tailNum / HALF_INTERLEAVE_COEF;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vregFormerIn;
        MicroAPI::RegTensor<T> vregLatterIn;
        MicroAPI::RegTensor<T> vregOdd;
        MicroAPI::RegTensor<T> vregEven;
        MicroAPI::MaskReg pregPart1;
        MicroAPI::MaskReg pregPart2;

        for (uint16_t idxD = 0; idxD < currDNum; idxD++) {
            currInUb = inUb + idxD * dAlignLen;
            currOutUb = outUb + idxD * halfDAlign * HALF_INTERLEAVE_COEF;
            uint32_t part1Cnt = part1Num;
            uint32_t part2Cnt = part2Num;
            for (uint16_t i = 0; i < dLoopCnt; i++) {
                int32_t inOffset = i * loopSize;
                int32_t outOffset = i * vecLen;
                pregPart1 = MicroAPI::UpdateMask<T>(part1Cnt);
                pregPart2 = MicroAPI::UpdateMask<T>(part2Cnt);
                MicroAPI::LoadAlign(vregFormerIn, currInUb + inOffset);
                MicroAPI::LoadAlign(vregLatterIn, currInUb + inOffset + vecLen);
                MicroAPI::DeInterleave<T>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                MicroAPI::StoreAlign(currOutUb + outOffset, vregEven, pregPart1);
                MicroAPI::StoreAlign(currOutUb + outOffset + halfDAlign, vregOdd, pregPart2);
            }
        }
    }
}

} // namespace RotaryPositionEmbeddingGrad

#endif // __ROTARY_POSITION_EMBEDDING_GRAD_ROTARY_X_VF__
