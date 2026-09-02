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
 * \file dense_lightning_indexer_softmax_lse_vector.h
 * \brief
 */

#ifndef __DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_VECTOR_H__
#define __DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_VECTOR_H__

#include "kernel_operator.h"

namespace DenseLISoftmaxLseServiceVec {
using namespace AscendC;

constexpr uint8_t B32_BLOCK_ALIGN_NUM = 8;
constexpr uint8_t B32_VEC_REPEAT_STRIDE = 8;
constexpr uint64_t VEC_REPEAT_BYTES = 256;
constexpr int32_t CONST_TWO = 2;

template <typename T>
__aicore__ inline void CopyIn(LocalTensor<float> &mmOutUb, LocalTensor<T> &weightsUb, GlobalTensor<float> &mMoutGm,
                              GlobalTensor<T> &weightScaleGm, int64_t mm1ResGmOffset, int64_t weights_gmoffset,
                              int64_t groupInner, int64_t s2Inner, int64_t mmUbStride)
{
    // 将mm1ResGmOffset copy到UB上
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams dataCopymMoutParams;
    dataCopymMoutParams.blockCount = groupInner;
    dataCopymMoutParams.blockLen = s2Inner * sizeof(float);
    dataCopymMoutParams.srcStride = 0;
    dataCopymMoutParams.dstStride = mmUbStride;
    dataCopymMoutParams.rsv = 0;
    AscendC::DataCopyPad(mmOutUb, mMoutGm[mm1ResGmOffset], dataCopymMoutParams, padParams);

    // 将weights_gmoffset copy到UB
    AscendC::DataCopyPadExtParams<T> padTParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams dataCopyweightParams;
    dataCopyweightParams.blockCount = 1;
    dataCopyweightParams.blockLen = groupInner * sizeof(T);
    dataCopyweightParams.srcStride = 0;
    dataCopyweightParams.dstStride = 0;
    dataCopyweightParams.rsv = 0;
    AscendC::DataCopyPad(weightsUb, weightScaleGm[weights_gmoffset], dataCopyweightParams, padTParams);
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T> &dstGm, const LocalTensor<T> &srcUb, int64_t copyCount)
{
    AscendC::DataCopyParams dataCopyOutyParams;
    dataCopyOutyParams.blockCount = 1;
    dataCopyOutyParams.blockLen = copyCount * sizeof(T);
    dataCopyOutyParams.srcStride = 0;
    dataCopyOutyParams.dstStride = 0;
    AscendC::DataCopyPad(dstGm, srcUb, dataCopyOutyParams);
}

template <typename T>
__aicore__ inline void DoScale(const LocalTensor<float> &reduceCacheBuf, LocalTensor<float> &mmOutUb,
                               LocalTensor<float> &weightsUb, LocalTensor<T> &weightsTUb, LocalTensor<float> &tmpBuff,
                               int64_t groupInner, int64_t s2Inner, int32_t outerGidx)
{
    // cast bfloat16_t to float
    if constexpr (!IsSameType<T, float>::value) {
        AscendC::Cast(weightsUb, weightsTUb, RoundMode::CAST_NONE, groupInner);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // weight broadcast: [groupInner, 1] -> [groupInner, 8]
    AscendC::Brcb(tmpBuff, weightsUb,
                  DenseLISoftmaxLseCommon::CeilDiv(groupInner, static_cast<int64_t>(B32_BLOCK_ALIGN_NUM)),
                  {1, B32_VEC_REPEAT_STRIDE});
    AscendC::PipeBarrier<PIPE_V>();

    // do scale: [groupInner, 8] * [groupInner, s2Inner]
    uint64_t countPerRepeat = VEC_REPEAT_BYTES / sizeof(float);
    uint64_t repeatTimes = s2Inner / countPerRepeat;
    for (int32_t i = 0; i < groupInner; i++) {
        if (outerGidx == 0) {
            AscendC::Mul(reduceCacheBuf[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM],
                         countPerRepeat, repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        } else {
            AscendC::Mul(mmOutUb[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM], countPerRepeat,
                         repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        }
    }

    if (outerGidx != 0) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(reduceCacheBuf, mmOutUb, reduceCacheBuf, groupInner * s2Inner);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline uint64_t FindNearestPower2(uint64_t value)
{
    if (value <= CONST_TWO) {
        return value;
    } else {
        // 返回前导0的个数，对于64位整数，最大有效位位置 = 63 - 前导0个数
        const uint64_t pow = 63 - AscendC::ScalarCountLeadingZero(value);
        return (1 << pow);
    }
}

__aicore__ inline void DoReduce(const LocalTensor<float> &srcTensor, LocalTensor<float> &dstTensor, int32_t rNum,
                                int32_t aNum)
{
    if (rNum == 1) {
        AscendC::Adds<float>(dstTensor, srcTensor, 0, aNum);
        AscendC::PipeBarrier<PIPE_V>();
        return;
    }

    uint32_t dichotomizeAddPow = FindNearestPower2(rNum);
    uint32_t dichotomizeAddDiffSize = rNum - dichotomizeAddPow;
    if (dichotomizeAddDiffSize != 0) {
        AscendC::Add(srcTensor, srcTensor, srcTensor[dichotomizeAddPow * aNum], dichotomizeAddDiffSize * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    int32_t nowRows = dichotomizeAddPow;
    while (nowRows > CONST_TWO) {
        nowRows = nowRows / CONST_TWO;
        AscendC::Add(srcTensor, srcTensor, srcTensor[nowRows * aNum], nowRows * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Add(dstTensor, srcTensor, srcTensor[aNum], aNum);
    AscendC::PipeBarrier<PIPE_V>();
}
} // namespace DenseLISoftmaxLseServiceVec

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_VECTOR_H
