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
 * \file vf_muls_sel_simple_softmax_sfmg.h
 */
#ifndef VF_MULS_SEL_SIMPLE_SOFTMAX_SFMG_H
#define VF_MULS_SEL_SIMPLE_SOFTMAX_SFMG_H
#include "kernel_tensor.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
using namespace MicroAPI;
/* **************************************************************************************************

MulsSelSimpleSoftMax *

************************************************************************************************* /

@INGROUP MulsSelSimpleSoftMax
brief compute exp(sel(x * scale, mask) - max) / expSum
param [out] dst output LocalTensor
param [in] inSumTensor input expsum of last axis
param [in] inMaxTensor input max value of last axis
param [in] src input LocalTensor
param [in] tiling softmaxtiling
*/
template <typename T1, typename T, uint32_t srcN, bool hasAtten, bool hasPse, bool isDeter, bool IsSparseIndicesExist>
__simd_vf__ inline void MulsSelVFPseType1(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t softmaxL1LocalInt,
                                          uint64_t sumLocalInt, uint64_t maxLocalInt, uint64_t maskLocalInt,
                                          uint64_t srcLocalIntTail, uint64_t dstLocalIntTail,
                                          uint64_t softmaxL1LocalIntTail, uint64_t dstLocalIntTailZero,
                                          uint64_t maskLocalIntTail, uint64_t pseLocalInt, uint64_t pseLocalIntTail,
                                          uint32_t tailSize, uint32_t pseStride, uint16_t repeatTimes, const T scale,
                                          const T minValue, uint32_t pseTailStride, uint32_t srcM, int64_t lseLocalInt,
                                          float pScalar)
{
    RegTensor<float> vregSrc;
    RegTensor<float> vregMax;
    RegTensor<float> vregSum;
    RegTensor<float> vregLse;
    RegTensor<float> vregPse;
    RegTensor<half> vregPseHalf;
    RegTensor<bfloat16_t> vregPseBf;
    RegTensor<half> vregPseHalfEven;
    RegTensor<half> vregPseHalfOdd;
    RegTensor<bfloat16_t> vregPseBfEven;
    RegTensor<bfloat16_t> vregPseBfOdd;
    RegTensor<float> vregSub;
    RegTensor<float> vregExp;
    RegTensor<float> vregDiv;
    RegTensor<float> vregMuls;
    RegTensor<float> vregMin;
    RegTensor<float> vregReduceSumMuls;
    // tail
    RegTensor<float> vregSrcTail;
    RegTensor<float> vregSubTail;
    RegTensor<float> vregExpTail;
    RegTensor<float> vregDivTail;
    RegTensor<float> vregReduceSumMulsTail;

    RegTensor<float> vregSel;
    RegTensor<float> vregSelTail;
    RegTensor<float> vregPseTail;
    RegTensor<half> vregPseHalfTail;
    RegTensor<bfloat16_t> vregPseBfTail;
    RegTensor<half> vregPseHalfTailEven;
    RegTensor<half> vregPseHalfTailOdd;
    RegTensor<bfloat16_t> vregPseBfTailEven;
    RegTensor<bfloat16_t> vregPseBfTailOdd;
    RegTensor<float> vregMulsTail;
    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregFullExeB16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg pregTailExe = UpdateMask<float>(tailSize);
    MaskReg pregCompare;
    MaskReg pregCompareTail;

    Duplicate(vregMin, minValue);
    Duplicate(vregReduceSumMuls, 0.0f);
    Duplicate(vregReduceSumMulsTail, 0.0f);

    // 64 * 128 max/sum: 64 * 8
    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregLse, ((__ubuf__ float *&)lseLocalInt), 1);
        // 1 -> 64 【64，128】 【64，2，64】 64
        for (uint16_t n = 0; n < repeatTimes; n++) {
            LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float *&)srcLocalInt), srcN);
            Muls(vregMuls, vregSrc, scale, pregFullExe);
            Sub(vregSub, vregMuls, vregLse, pregFullExe);
            Exp(vregExp, vregSub, pregFullExe);
            if constexpr (IsSparseIndicesExist) {
                Add(vregReduceSumMuls, vregReduceSumMuls, vregExp, pregFullExe);
            }
            StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                ((__ubuf__ float *&)dstLocalInt), vregExp, srcN, pregFullExe);
        }
        // tailLoop
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ float *&)srcLocalIntTail),
                                                                  128);
        Muls(vregMulsTail, vregSrcTail, scale, pregTailExe);
        Sub(vregSubTail, vregMulsTail, vregLse, pregTailExe);
        Exp(vregExpTail, vregSubTail, pregTailExe);
        if constexpr (IsSparseIndicesExist) {
            Add(vregReduceSumMulsTail, vregReduceSumMulsTail, vregExpTail, pregTailExe);
        }
        StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
            ((__ubuf__ float *&)dstLocalIntTail), vregExpTail, 128, pregFullExe);
        if constexpr (isDeter == 1 && srcN == 64) { // 确定性计算需要将64~128的数据补零， 否则会有脏数据inf
            Duplicate(vregExp, 0);
            StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ float *&)dstLocalIntTailZero, vregExp, 128, pregFullExe);
        }
    }
    if constexpr (IsSparseIndicesExist) {
        for (uint16_t n = 0; n < repeatTimes; n++) {
            Muls(vregReduceSumMuls, vregReduceSumMuls, pScalar, pregFullExe);
            StoreAlign<float>(((__ubuf__ float *&)softmaxL1LocalInt), vregReduceSumMuls, pregFullExe);
        }
        Muls(vregReduceSumMulsTail, vregReduceSumMulsTail, pScalar, pregTailExe);
        StoreAlign<float>(((__ubuf__ float *&)softmaxL1LocalIntTail), vregReduceSumMulsTail, pregFullExe);
    }
}

template <typename T1, typename T, uint32_t srcN, bool hasAtten = 0, bool hasPse = 0, bool isDeter = 0,
          bool IsSparseIndicesExist = 0>
__aicore__ inline void MulsSelSimpleSoftMax(const LocalTensor<float> &dstTensor, const LocalTensor<float> &inLseTensor,
                                            const LocalTensor<float> &inMaxTensor, const LocalTensor<float> &srcTensor,
                                            const LocalTensor<float> &softmaxL1Tensor, const LocalTensor<T1> &pseTensor,
                                            const LocalTensor<uint8_t> &maskTensor, const T scale, const T minValue,
                                            uint32_t srcM, uint32_t realN = srcN, float pScalar = 1.0,
                                            uint32_t pseType = 1, uint32_t pse_layout_type = 1, float posShift = -1.0,
                                            float slopes = -1.0)
{
    const uint32_t fullExeSize = 64;
    uint16_t repeatTimes = srcN / fullExeSize - 1;
    uint32_t tailSize = realN % fullExeSize == 0 ? fullExeSize : realN % fullExeSize;
    uint32_t pseStride = pse_layout_type == 1 ? 0 : srcN;
    uint32_t pseTailStride = pse_layout_type == 1 ? 0 : 128;
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t softmaxL1LocalInt = softmaxL1Tensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t lseLocalInt = inLseTensor.GetPhyAddr();
    uint64_t sumLocalInt = inLseTensor.GetPhyAddr();
    uint64_t maxLocalInt = inMaxTensor.GetPhyAddr();
    uint64_t maskLocalInt = maskTensor.GetPhyAddr();
    uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t dstLocalIntTail = dstTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t softmaxL1LocalIntTail = softmaxL1Tensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t dstLocalIntTailZero = dstTensor.GetPhyAddr() + fullExeSize * 4 * (repeatTimes + 1);
    uint64_t maskLocalIntTail = maskTensor.GetPhyAddr() + fullExeSize * repeatTimes;
    uint64_t pseLocalInt = pseTensor.GetPhyAddr();
    uint64_t pseLocalIntTail = pseTensor.GetPhyAddr() + fullExeSize * sizeof(T1) * repeatTimes;
    MulsSelVFPseType1<T1, T, srcN, hasAtten, hasPse, isDeter, IsSparseIndicesExist>(
        srcLocalInt, dstLocalInt, softmaxL1LocalInt, sumLocalInt, maxLocalInt, maskLocalInt, srcLocalIntTail,
        dstLocalIntTail, softmaxL1LocalIntTail, dstLocalIntTailZero, maskLocalIntTail, pseLocalInt, pseLocalIntTail,
        tailSize, pseStride, repeatTimes, scale, minValue, pseTailStride, srcM, lseLocalInt, pScalar);
}
#else
template <typename T1, typename T, uint32_t srcN, bool hasAtten = 0, bool hasPse = 0, bool isDeter = 0,
          bool IsSparseIndicesExist = 0>
__aicore__ inline void MulsSelSimpleSoftMax(const LocalTensor<float> &dstTensor, const LocalTensor<float> &inLseTensor,
                                            const LocalTensor<float> &inMaxTensor, const LocalTensor<float> &srcTensor,
                                            const LocalTensor<float> &softmaxL1Tensor, const LocalTensor<T1> &pseTensor,
                                            const LocalTensor<uint8_t> &maskTensor, const T scale, const T minValue,
                                            uint32_t srcM, uint32_t realN = srcN, float pScalar = 1.0,
                                            uint32_t pseType = 1, uint32_t pse_layout_type = 1, float posShift = -1.0,
                                            float slopes = -1.0)
{}
#endif
} // namespace AscendC

#endif // MY_SIMPLE_SOFTMAX_INTERFACE_H
