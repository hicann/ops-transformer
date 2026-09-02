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
 * \file vf_muls_sel_simple_softmax_aligned256.h
 */
#ifndef VF_MULS_SEL_SIMPLE_SOFTMAX_ALIGNED_256_H
#define VF_MULS_SEL_SIMPLE_SOFTMAX_ALIGNED_256_H
#include "kernel_tensor.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
using namespace MicroAPI;
/* **************************************************************************************************

MulsSelSimpleSoftMaxAligned256 *

************************************************************************************************* /

@INGROUP MulsSelSimpleSoftMaxAligned256
brief compute exp(sel(x * scale, mask) - max) / expSum
param [out] dst output LocalTensor
param [in] inSumTensor input expsum of last axis
param [in] inMaxTensor input max value of last axis
param [in] src input LocalTensor
param [in] tiling softmaxtiling
*/
template <typename T1, typename T, uint16_t srcN, bool hasAtten = 0, bool hasPse = 0, bool isDeter = 0>
__aicore__ inline void MulsSelSimpleSoftMaxAligned256(
    const LocalTensor<float> &dstTensor, const LocalTensor<float> &inSumTensor, const LocalTensor<float> &inMaxTensor,
    const LocalTensor<float> &srcTensor, const LocalTensor<T1> &pseTensor, const LocalTensor<uint8_t> &maskTensor,
    const T scale, const T minValue, uint32_t srcM, uint32_t realN = srcN, uint32_t pseType = 1,
    uint32_t pse_layout_type = 1, float posShift = -1.0, float slopes = -1.0)
{
    constexpr uint16_t fullExeSize = 64;
    uint16_t repeatTimes = (realN + fullExeSize - 1) / fullExeSize - 1;
    uint32_t tailSize = realN % fullExeSize == 0 ? fullExeSize : realN % fullExeSize;
    uint16_t pseStride = pse_layout_type == 1 ? 0 : srcN;
    uint16_t pseTailStride = pse_layout_type == 1 ? 0 : srcN;

    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t sumLocalInt = inSumTensor.GetPhyAddr();
    uint64_t maxLocalInt = inMaxTensor.GetPhyAddr();
    uint64_t maskLocalInt = maskTensor.GetPhyAddr();
    uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t dstLocalIntTail = dstTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t dstLocalIntTailZero = dstTensor.GetPhyAddr() + fullExeSize * 4 * (repeatTimes + 1);
    uint64_t maskLocalIntTail = maskTensor.GetPhyAddr() + fullExeSize * repeatTimes;
    uint64_t pseLocalInt = pseTensor.GetPhyAddr();
    uint64_t pseLocalIntTail = pseTensor.GetPhyAddr() + fullExeSize * sizeof(T1) * repeatTimes;
    if (pseType == 1) {
        __VEC_SCOPE__
        {
            RegTensor<float> vregSrc;
            RegTensor<float> vregMax;
            RegTensor<float> vregSum;
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
            // tail
            RegTensor<float> vregSrcTail;
            RegTensor<float> vregSubTail;
            RegTensor<float> vregExpTail;
            RegTensor<float> vregDivTail;

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

            // 64 * 128 max/sum: 64 * 8
            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(vregMax, ((__ubuf__ float *&)maxLocalInt + m * 8));
                DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(vregSum, ((__ubuf__ float *&)sumLocalInt + m * 8));
                // 1 -> 64 【64，128】 【64，2，64】 64
                // #pragma unroll
                for (uint16_t n = 0; n < repeatTimes; n++) {
                    DataCopy(vregSrc, ((__ubuf__ float *&)srcLocalInt + (m * srcN + n * fullExeSize)));
                    if constexpr (IsSameType<T1, half>::value && hasPse == 1) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseHalf,
                                                                              ((__ubuf__ T1 *&)pseLocalInt), pseStride);
                        Interleave(vregPseHalfEven, vregPseHalfOdd, vregPseHalf, vregPseHalf);
                        Cast<float, T1, castTraitB162B32Even>(vregPse, vregPseHalfEven, pregFullExeB16);
                        Add(vregSrc, vregSrc, vregPse, pregFullExe);
                    } else if constexpr (IsSameType<T1, bfloat16_t>::value && hasPse == 1) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseBf, ((__ubuf__ T1 *&)pseLocalInt),
                                                                              pseStride);
                        Interleave(vregPseBfEven, vregPseBfOdd, vregPseBf, vregPseBf);
                        Cast<float, T1, castTraitB162B32Even>(vregPse, vregPseBfEven, pregFullExeB16);
                        Add(vregSrc, vregSrc, vregPse, pregFullExe);
                    } else if constexpr (IsSameType<T1, float>::value && hasPse == 1) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPse, ((__ubuf__ T1 *&)pseLocalInt),
                                                                              pseStride);
                        Add(vregSrc, vregSrc, vregPse, pregFullExe);
                    }
                    Muls(vregMuls, vregSrc, scale, pregFullExe);
                    if constexpr (hasAtten == 1) {
                        DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                            pregCompare, (__ubuf__ uint32_t *&)maskLocalInt, srcN);
                        Select(vregSel, vregMin, vregMuls, pregCompare);
                        FusedExpSub(vregExp, vregSel, vregMax, pregFullExe);
                    } else {
                        FusedExpSub(vregExp, vregMuls, vregMax, pregFullExe);
                    }
                    Div(vregDiv, vregExp, vregSum, pregFullExe);
                    DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                        ((__ubuf__ float *&)dstLocalInt + (m * srcN + n * fullExeSize)), vregDiv, pregFullExe);
                }
                // tailLoop
                DataCopy(vregSrcTail, ((__ubuf__ float *&)srcLocalIntTail + m * srcN));
                if constexpr (IsSameType<T1, half>::value && hasPse == 1) {
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        vregPseHalfTail, ((__ubuf__ T1 *&)pseLocalIntTail), pseTailStride);
                    Interleave(vregPseHalfTailEven, vregPseHalfTailOdd, vregPseHalfTail, vregPseHalfTail);
                    Cast<float, T1, castTraitB162B32Even>(vregPseTail, vregPseHalfTailEven, pregFullExeB16);
                    Add(vregSrcTail, vregSrcTail, vregPseTail, pregTailExe);
                } else if constexpr (IsSameType<T1, bfloat16_t>::value && hasPse == 1) {
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        vregPseBfTail, ((__ubuf__ T1 *&)pseLocalIntTail), pseTailStride);
                    Interleave(vregPseBfTailEven, vregPseBfTailOdd, vregPseBfTail, vregPseBfTail);
                    Cast<float, T1, castTraitB162B32Even>(vregPseTail, vregPseBfTailEven, pregFullExeB16);
                    Add(vregSrcTail, vregSrcTail, vregPseTail, pregTailExe);
                } else if constexpr (IsSameType<T1, float>::value && hasPse == 1) {
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        vregPseTail, ((__ubuf__ T1 *&)pseLocalIntTail), pseTailStride);
                    Add(vregSrcTail, vregSrcTail, vregPseTail, pregTailExe);
                }
                Muls(vregMulsTail, vregSrcTail, scale, pregTailExe);
                if constexpr (hasAtten == 1) {
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        pregCompareTail, (__ubuf__ uint32_t *&)maskLocalIntTail, srcN);
                    Select(vregSelTail, vregMin, vregMulsTail, pregCompareTail);
                    FusedExpSub(vregExpTail, vregSelTail, vregMax, pregFullExe);
                } else {
                    FusedExpSub(vregExpTail, vregMulsTail, vregMax, pregTailExe);
                }
                Div(vregDivTail, vregExpTail, vregSum, pregTailExe);
                DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(((__ubuf__ float *&)dstLocalIntTail + m * srcN),
                                                                    vregDivTail, pregFullExe);
            }
        }
    }
}
#else
template <typename T1, typename T, uint16_t srcN, bool hasAtten = 0, bool hasPse = 0, bool isDeter = 0>
__aicore__ inline void MulsSelSimpleSoftMaxAligned256(
    const LocalTensor<float> &dstTensor, const LocalTensor<float> &inSumTensor, const LocalTensor<float> &inMaxTensor,
    const LocalTensor<float> &srcTensor, const LocalTensor<T1> &pseTensor, const LocalTensor<uint8_t> &maskTensor,
    const T scale, const T minValue, uint32_t srcM, uint32_t realN = srcN, uint32_t pseType = 1,
    uint32_t pse_layout_type = 1, float posShift = -1.0, float slopes = -1.0)
{}
#endif
} // namespace AscendC

#endif // VF_MULS_SEL_SIMPLE_SOFTMAX_ALIGNED_256_H
