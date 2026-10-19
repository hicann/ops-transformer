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
 * \file vf_muls_sel_simple_softmax_david.h
 */
#ifndef VF_MULS_SEL_SIMPLE_SOFTMAX_DAVID_H
#define VF_MULS_SEL_SIMPLE_SOFTMAX_DAVID_H
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
template <typename T1, typename T, uint32_t srcN, const uint32_t VECTOR_BASEM = 64, bool hasAtten = 0, bool hasPse = 0,
          bool isDeter = 0>
__aicore__ inline void ComputeSimpleSoftMaxAndSubMul(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& dstDsTensor, const LocalTensor<float>& inSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<float>& srcTensor, const LocalTensor<float>& srcDsTensor,
    const LocalTensor<T1>& pseTensor, const LocalTensor<uint8_t>& maskTensor, const T scale, float deqScaleDyV,
    const T minValue, uint32_t srcM, uint32_t realN = srcN, uint32_t pseType = 1, uint32_t pse_layout_type = 1,
    float posShift = -1.0, float slopes = -1.0)
{
    const uint32_t fullExeSize = 64;
    uint16_t repeatTimes = srcN / fullExeSize - 1;
    uint32_t tailSize = realN % fullExeSize == 0 ? fullExeSize : realN % fullExeSize;
    uint32_t pseStride = pse_layout_type == 1 ? 0 : srcN;
    uint32_t pseTailStride = pse_layout_type == 1 ? 0 : 128;
    uint32_t sfmgOffset = VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE;

    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t srcDsLocalInt = srcDsTensor.GetPhyAddr();
    uint64_t dstDsLocalInt = dstDsTensor.GetPhyAddr();
    uint64_t sumLocalInt = inSumTensor.GetPhyAddr();
    uint64_t maxLocalInt = inMaxTensor.GetPhyAddr();
    uint64_t gradLocalInt = inMaxTensor.GetPhyAddr() + sfmgOffset;
    uint64_t maskLocalInt = maskTensor.GetPhyAddr();
    uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t dstLocalIntTail = dstTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t srcDsLocalIntTail = srcDsTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t dstDsLocalIntTail = dstDsTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t dstLocalIntTailZero = dstTensor.GetPhyAddr() + fullExeSize * 4 * (repeatTimes + 1);
    uint64_t dstDsLocalIntTailZero = dstDsTensor.GetPhyAddr() + fullExeSize * 4 * (repeatTimes + 1);
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
            // submul
            RegTensor<float> vregGrad;
            RegTensor<float> vregDp;
            RegTensor<float> vregMul;
            RegTensor<float> vregDpTail;
            RegTensor<float> vregMulTail;
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
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                    vregMax, ((__ubuf__ float*&)maxLocalInt), 8);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                    vregSum, ((__ubuf__ float*&)sumLocalInt), 8);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                    vregGrad, ((__ubuf__ float*&)gradLocalInt), 1);
                // 1 -> 64 【64，128】 【64，2，64】 64
                for (uint16_t n = 0; n < repeatTimes; n++) {
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float*&)srcLocalInt),
                                                                             srcN);
                    if constexpr (IsSameType<T1, half>::value && hasPse == 1) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseHalf, ((__ubuf__ T1*&)pseLocalInt),
                                                                              pseStride);
                        Interleave(vregPseHalfEven, vregPseHalfOdd, vregPseHalf, vregPseHalf);
                        Cast<float, T1, castTraitB162B32Even>(vregPse, vregPseHalfEven, pregFullExeB16);
                        Add(vregSrc, vregSrc, vregPse, pregFullExe);
                    } else if constexpr (IsSameType<T1, bfloat16_t>::value && hasPse == 1) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseBf, ((__ubuf__ T1*&)pseLocalInt),
                                                                              pseStride);
                        Interleave(vregPseBfEven, vregPseBfOdd, vregPseBf, vregPseBf);
                        Cast<float, T1, castTraitB162B32Even>(vregPse, vregPseBfEven, pregFullExeB16);
                        Add(vregSrc, vregSrc, vregPse, pregFullExe);
                    } else if constexpr (IsSameType<T1, float>::value && hasPse == 1) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPse, ((__ubuf__ T1*&)pseLocalInt),
                                                                              pseStride);
                        Add(vregSrc, vregSrc, vregPse, pregFullExe);
                    }
                    Muls(vregMuls, vregSrc, scale, pregFullExe);
                    if constexpr (hasAtten == 1) {
                        DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                            pregCompare, (__ubuf__ uint32_t*&)maskLocalInt, srcN);
                        Select(vregSel, vregMin, vregMuls, pregCompare);
                        FusedExpSub(vregExp, vregSel, vregMax, pregFullExe);
                    } else {
                        FusedExpSub(vregExp, vregMuls, vregMax, pregFullExe);
                    }
                    Div(vregDiv, vregExp, vregSum, pregFullExe);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                        ((__ubuf__ float*&)dstLocalInt), vregDiv, srcN, pregFullExe);

                    // submul
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregDp, ((__ubuf__ float*&)srcDsLocalInt),
                                                                             srcN);
                    if (std::is_same_v<T1, float8_e4m3_t> || std::is_same_v<T1, float8_e5m2_t>) {
                        Muls(vregDp, vregDp, deqScaleDyV, pregFullExe);
                    }
                    Sub(vregSub, vregDp, vregGrad, pregFullExe);
                    Mul(vregMul, vregSub, vregDiv, pregFullExe);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstDsLocalInt), vregMul,
                                                                             srcN, pregFullExe);
                }
                // tailLoop
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail,
                                                                         ((__ubuf__ float*&)srcLocalIntTail), 128);
                if constexpr (IsSameType<T1, half>::value && hasPse == 1) {
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        vregPseHalfTail, ((__ubuf__ T1*&)pseLocalIntTail), pseTailStride);
                    Interleave(vregPseHalfTailEven, vregPseHalfTailOdd, vregPseHalfTail, vregPseHalfTail);
                    Cast<float, T1, castTraitB162B32Even>(vregPseTail, vregPseHalfTailEven, pregFullExeB16);
                    Add(vregSrcTail, vregSrcTail, vregPseTail, pregTailExe);
                } else if constexpr (IsSameType<T1, bfloat16_t>::value && hasPse == 1) {
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        vregPseBfTail, ((__ubuf__ T1*&)pseLocalIntTail), pseTailStride);
                    Interleave(vregPseBfTailEven, vregPseBfTailOdd, vregPseBfTail, vregPseBfTail);
                    Cast<float, T1, castTraitB162B32Even>(vregPseTail, vregPseBfTailEven, pregFullExeB16);
                    Add(vregSrcTail, vregSrcTail, vregPseTail, pregTailExe);
                } else if constexpr (IsSameType<T1, float>::value && hasPse == 1) {
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseTail, ((__ubuf__ T1*&)pseLocalIntTail),
                                                                          pseTailStride);
                    Add(vregSrcTail, vregSrcTail, vregPseTail, pregTailExe);
                }
                Muls(vregMulsTail, vregSrcTail, scale, pregTailExe);
                if constexpr (hasAtten == 1) {
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        pregCompareTail, (__ubuf__ uint32_t*&)maskLocalIntTail, 128);
                    Select(vregSelTail, vregMin, vregMulsTail, pregCompareTail);
                    FusedExpSub(vregExpTail, vregSelTail, vregMax, pregTailExe);
                } else {
                    FusedExpSub(vregExpTail, vregMulsTail, vregMax, pregTailExe);
                }
                Div(vregDivTail, vregExpTail, vregSum, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                    ((__ubuf__ float*&)dstLocalIntTail), vregDivTail, 128, pregFullExe);
                if constexpr (isDeter == 1 && srcN == 64) { // 确定性计算需要将64~128的数据补零， 否则会有脏数据inf
                    Duplicate(vregDiv, 0);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                        (__ubuf__ float*&)dstLocalIntTailZero, vregDiv, 128, pregFullExe);
                }

                // submul
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregDpTail,
                                                                         ((__ubuf__ float*&)srcDsLocalIntTail), 128);
                if (std::is_same_v<T1, float8_e4m3_t> || std::is_same_v<T1, float8_e5m2_t>) {
                    Muls(vregDpTail, vregDpTail, deqScaleDyV, pregTailExe);
                }
                Sub(vregSubTail, vregDpTail, vregGrad, pregTailExe);
                Mul(vregMulTail, vregSubTail, vregDivTail, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstDsLocalIntTail),
                                                                         vregMulTail, 128, pregFullExe);
                if constexpr (isDeter == 1 && srcN == 64) { // 确定性计算需要将64~128的数据补零，否则会有脏数据inf
                    Duplicate(vregMul, 0);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                        (__ubuf__ float*&)dstDsLocalIntTailZero, vregMul, 128, pregFullExe);
                }
            }
        }
    } else if (pseType == 0) {
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
            // submul
            RegTensor<float> vregGrad;
            RegTensor<float> vregDp;
            RegTensor<float> vregMul;
            RegTensor<float> vregDpTail;
            RegTensor<float> vregMulTail;
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
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                    vregMax, ((__ubuf__ float*&)maxLocalInt), 8);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                    vregSum, ((__ubuf__ float*&)sumLocalInt), 8);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                    vregGrad, ((__ubuf__ float*&)gradLocalInt), 1);
                // 1 -> 64 【64，128】 【64，2，64】 64
                for (uint16_t n = 0; n < repeatTimes; n++) {
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float*&)srcLocalInt),
                                                                             srcN);
                    Muls(vregMuls, vregSrc, scale, pregFullExe);
                    if constexpr (IsSameType<T1, half>::value && hasPse == 1) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseHalf, ((__ubuf__ T1*&)pseLocalInt),
                                                                              pseStride);
                        Interleave(vregPseHalfEven, vregPseHalfOdd, vregPseHalf, vregPseHalf);
                        Cast<float, T1, castTraitB162B32Even>(vregPse, vregPseHalfEven, pregFullExeB16);
                        Add(vregMuls, vregMuls, vregPse, pregFullExe);
                    } else if constexpr (IsSameType<T1, bfloat16_t>::value && hasPse == 1) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseBf, ((__ubuf__ T1*&)pseLocalInt),
                                                                              pseStride);
                        Interleave(vregPseBfEven, vregPseBfOdd, vregPseBf, vregPseBf);
                        Cast<float, T1, castTraitB162B32Even>(vregPse, vregPseBfEven, pregFullExeB16);
                        Add(vregMuls, vregMuls, vregPse, pregFullExe);
                    } else if constexpr (IsSameType<T1, float>::value && hasPse == 1) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPse, ((__ubuf__ T1*&)pseLocalInt),
                                                                              pseStride);
                        Add(vregMuls, vregMuls, vregPse, pregFullExe);
                    }
                    if constexpr (hasAtten == 1) {
                        DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                            pregCompare, (__ubuf__ uint32_t*&)maskLocalInt, srcN);
                        Select(vregSel, vregMin, vregMuls, pregCompare);
                        FusedExpSub(vregExp, vregSel, vregMax, pregFullExe);
                    } else {
                        FusedExpSub(vregExp, vregMuls, vregMax, pregFullExe);
                    }
                    Div(vregDiv, vregExp, vregSum, pregFullExe);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                        ((__ubuf__ float*&)dstLocalInt), vregDiv, srcN, pregFullExe);
                    // submul
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregDp, ((__ubuf__ float*&)srcDsLocalInt),
                                                                             srcN);
                    if (std::is_same_v<T1, float8_e4m3_t> || std::is_same_v<T1, float8_e5m2_t>) {
                        Muls(vregDp, vregDp, deqScaleDyV, pregFullExe);
                    }
                    Sub(vregSub, vregDp, vregGrad, pregFullExe);
                    Mul(vregMul, vregSub, vregDiv, pregFullExe);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstDsLocalInt), vregMul,
                                                                             srcN, pregFullExe);
                }
                // tailLoop
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail,
                                                                         ((__ubuf__ float*&)srcLocalIntTail), 128);
                Muls(vregMulsTail, vregSrcTail, scale, pregTailExe);
                if constexpr (IsSameType<T1, half>::value && hasPse == 1) {
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        vregPseHalfTail, ((__ubuf__ T1*&)pseLocalIntTail), pseTailStride);
                    Interleave(vregPseHalfTailEven, vregPseHalfTailOdd, vregPseHalfTail, vregPseHalfTail);
                    Cast<float, T1, castTraitB162B32Even>(vregPseTail, vregPseHalfTailEven, pregFullExeB16);
                    Add(vregMulsTail, vregMulsTail, vregPseTail, pregTailExe);
                } else if constexpr (IsSameType<T1, bfloat16_t>::value && hasPse == 1) {
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        vregPseBfTail, ((__ubuf__ T1*&)pseLocalIntTail), pseTailStride);
                    Interleave(vregPseBfTailEven, vregPseBfTailOdd, vregPseBfTail, vregPseBfTail);
                    Cast<float, T1, castTraitB162B32Even>(vregPseTail, vregPseBfTailEven, pregFullExeB16);
                    Add(vregMulsTail, vregMulsTail, vregPseTail, pregTailExe);
                } else if constexpr (IsSameType<T1, float>::value && hasPse == 1) {
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseTail, ((__ubuf__ T1*&)pseLocalIntTail),
                                                                          pseTailStride);
                    Add(vregMulsTail, vregMulsTail, vregPseTail, pregTailExe);
                }
                if constexpr (hasAtten == 1) {
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        pregCompareTail, (__ubuf__ uint32_t*&)maskLocalIntTail, 128);
                    Select(vregSelTail, vregMin, vregMulsTail, pregCompareTail);
                    FusedExpSub(vregExpTail, vregSelTail, vregMax, pregTailExe);
                } else {
                    FusedExpSub(vregExpTail, vregMulsTail, vregMax, pregTailExe);
                }
                Div(vregDivTail, vregExpTail, vregSum, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                    ((__ubuf__ float*&)dstLocalIntTail), vregDivTail, 128, pregFullExe);
                if constexpr (isDeter == 1 && srcN == 64) { // 确定性计算需要将64~128的数据补零， 否则会有脏数据inf
                    Duplicate(vregDiv, 0);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                        (__ubuf__ float*&)dstLocalIntTailZero, vregDiv, 128, pregFullExe);
                }
                // submul
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregDpTail,
                                                                         ((__ubuf__ float*&)srcDsLocalIntTail), 128);
                if (std::is_same_v<T1, float8_e4m3_t> || std::is_same_v<T1, float8_e5m2_t>) {
                    Muls(vregDpTail, vregDpTail, deqScaleDyV, pregTailExe);
                }
                Sub(vregSubTail, vregDpTail, vregGrad, pregTailExe);
                Mul(vregMulTail, vregSubTail, vregDivTail, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstDsLocalIntTail),
                                                                         vregMulTail, 128, pregFullExe);
                if constexpr (isDeter == 1 && srcN == 64) { // 确定性计算需要将64~128的数据补零，否则会有脏数据inf
                    Duplicate(vregMul, 0);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                        (__ubuf__ float*&)dstDsLocalIntTailZero, vregMul, 128, pregFullExe);
                }
            }
        }
    } else if (pseType == 2) {
        uint32_t tailStartIdx = repeatTimes == 0 ? 0 : fullExeSize;
        __VEC_SCOPE__
        {
            RegTensor<float> vregSrc;
            RegTensor<float> vregMax;
            RegTensor<float> vregSum;
            RegTensor<float> vregPse;
            RegTensor<float> vregPseIdx;
            RegTensor<float> vregSub;
            RegTensor<float> vregExp;
            RegTensor<float> vregDiv;
            RegTensor<float> vregMuls;
            RegTensor<float> vregMin;
            // submul
            RegTensor<float> vregGrad;
            RegTensor<float> vregDp;
            RegTensor<float> vregMul;
            RegTensor<float> vregDpTail;
            RegTensor<float> vregMulTail;
            // tail
            RegTensor<float> vregSrcTail;
            RegTensor<float> vregSubTail;
            RegTensor<float> vregExpTail;
            RegTensor<float> vregDivTail;
            RegTensor<float> vregSel;
            RegTensor<float> vregPseTail;
            RegTensor<float> vregPseIdxTail;
            RegTensor<float> vregMulsTail;
            RegTensor<float> vregSelTail;
            MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregTailExe = UpdateMask<float>(tailSize);
            MaskReg pregFullExeB16 = CreateMask<half, MaskPattern::ALL>();
            MaskReg pregCompare;
            MaskReg pregCompareTail;

            Duplicate(vregMin, minValue);
            Arange(vregPseIdx, posShift);
            Arange(vregPseIdxTail, posShift + tailStartIdx);
            // 64 * 128 max/sum: 64 * 8
            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                    vregMax, ((__ubuf__ float*&)maxLocalInt), 8);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                    vregSum, ((__ubuf__ float*&)sumLocalInt), 8);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                    vregGrad, ((__ubuf__ float*&)gradLocalInt), 1);
                // 1 -> 64 【64，128】 【64，2，64】 64
                for (uint16_t n = 0; n < repeatTimes; n++) {
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float*&)srcLocalInt),
                                                                             srcN);
                    // pse inner generate start
                    Abs(vregPse, vregPseIdx, pregFullExe);
                    Adds(vregPseIdx, vregPseIdx, -1.0f, pregFullExe);
                    Muls(vregPse, vregPse, slopes, pregFullExe);
                    // pse inner generate end
                    Muls(vregMuls, vregSrc, scale, pregFullExe);
                    Add(vregMuls, vregMuls, vregPse, pregFullExe);
                    if constexpr (hasAtten == 1) {
                        DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                            pregCompare, (__ubuf__ uint32_t*&)maskLocalInt, srcN);
                        Select(vregSel, vregMin, vregMuls, pregCompare);
                        FusedExpSub(vregExp, vregSel, vregMax, pregFullExe);
                    } else {
                        FusedExpSub(vregExp, vregMuls, vregMax, pregFullExe);
                    }
                    Div(vregDiv, vregExp, vregSum, pregFullExe);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                        ((__ubuf__ float*&)dstLocalInt), vregDiv, srcN, pregFullExe);

                    // submul
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregDp, ((__ubuf__ float*&)srcDsLocalInt),
                                                                             srcN);
                    if (std::is_same_v<T1, float8_e4m3_t> || std::is_same_v<T1, float8_e5m2_t>) {
                        Muls(vregDp, vregDp, deqScaleDyV, pregFullExe);
                    }
                    Sub(vregSub, vregDp, vregGrad, pregFullExe);
                    Mul(vregMul, vregSub, vregDiv, pregFullExe);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstDsLocalInt), vregMul,
                                                                             srcN, pregFullExe);
                }
                // tailLoop
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail,
                                                                         ((__ubuf__ float*&)srcLocalIntTail), 128);
                // pse inner generate start
                Abs(vregPseTail, vregPseIdxTail, pregFullExe);
                Adds(vregPseIdxTail, vregPseIdxTail, -1.0f, pregFullExe);
                Muls(vregPseTail, vregPseTail, slopes, pregFullExe);
                // pse inner generate end
                Muls(vregMulsTail, vregSrcTail, scale, pregTailExe);
                Add(vregMulsTail, vregMulsTail, vregPseTail, pregTailExe);
                if constexpr (hasAtten == 1) {
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        pregCompareTail, (__ubuf__ uint32_t*&)maskLocalIntTail, 128);
                    Select(vregSelTail, vregMin, vregMulsTail, pregCompareTail);
                    FusedExpSub(vregExpTail, vregSelTail, vregMax, pregTailExe);
                } else {
                    FusedExpSub(vregExpTail, vregMulsTail, vregMax, pregTailExe);
                }
                Div(vregDivTail, vregExpTail, vregSum, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                    ((__ubuf__ float*&)dstLocalIntTail), vregDivTail, 128, pregFullExe);
                if constexpr (isDeter == 1 && srcN == 64) { // 确定性计算需要将64~128的数据补零， 否则会有脏数据inf
                    Duplicate(vregDiv, 0);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                        (__ubuf__ float*&)dstLocalIntTailZero, vregDiv, 128, pregFullExe);
                }

                // submul
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregDpTail,
                                                                         ((__ubuf__ float*&)srcDsLocalIntTail), 128);
                if (std::is_same_v<T1, float8_e4m3_t> || std::is_same_v<T1, float8_e5m2_t>) {
                    Muls(vregDpTail, vregDpTail, deqScaleDyV, pregTailExe);
                }
                Sub(vregSubTail, vregDpTail, vregGrad, pregTailExe);
                Mul(vregMulTail, vregSubTail, vregDivTail, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstDsLocalIntTail),
                                                                         vregMulTail, 128, pregFullExe);
                if constexpr (isDeter == 1 && srcN == 64) { // 确定性计算需要将64~128的数据补零，否则会有脏数据inf
                    Duplicate(vregMul, 0);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                        (__ubuf__ float*&)dstDsLocalIntTailZero, vregMul, 128, pregFullExe);
                }
            }
        }
    } else if (pseType == 3) {
        uint32_t tailStartIdx = repeatTimes == 0 ? 0 : fullExeSize;
        __VEC_SCOPE__
        {
            RegTensor<float> vregSrc;
            RegTensor<float> vregMax;
            RegTensor<float> vregSum;
            RegTensor<float> vregPse;
            RegTensor<float> vregPseIdx;
            RegTensor<float> vregSub;
            RegTensor<float> vregExp;
            RegTensor<float> vregDiv;
            RegTensor<float> vregMuls;
            RegTensor<float> vregMin;
            // submul
            RegTensor<float> vregGrad;
            RegTensor<float> vregDp;
            RegTensor<float> vregMul;
            RegTensor<float> vregDpTail;
            RegTensor<float> vregMulTail;
            // tail
            RegTensor<float> vregSrcTail;
            RegTensor<float> vregSubTail;
            RegTensor<float> vregExpTail;
            RegTensor<float> vregDivTail;

            RegTensor<float> vregSel;
            RegTensor<float> vregPseTail;
            RegTensor<float> vregPseIdxTail;
            RegTensor<float> vregMulsTail;
            RegTensor<float> vregSelTail;
            MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregTailExe = UpdateMask<float>(tailSize);
            MaskReg pregFullExeB16 = CreateMask<half, MaskPattern::ALL>();
            MaskReg pregCompare;
            MaskReg pregCompareTail;

            Duplicate(vregMin, minValue);
            Arange(vregPseIdx, posShift);
            Arange(vregPseIdxTail, posShift + tailStartIdx);

            // 64 * 128 max/sum: 64 * 8
            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                    vregMax, ((__ubuf__ float*&)maxLocalInt), 8);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                    vregSum, ((__ubuf__ float*&)sumLocalInt), 8);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                    vregGrad, ((__ubuf__ float*&)gradLocalInt), 1);
                // 1 -> 64 【64，128】 【64，2，64】 64
                for (uint16_t n = 0; n < repeatTimes; n++) {
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float*&)srcLocalInt),
                                                                             srcN);
                    // pse inner generate start
                    Abs(vregPse, vregPseIdx, pregFullExe);
                    Adds(vregPseIdx, vregPseIdx, -1.0f, pregFullExe);
                    Sqrt(vregPse, vregPse, pregFullExe);
                    Muls(vregPse, vregPse, slopes, pregFullExe);
                    // pse inner generate end
                    Muls(vregMuls, vregSrc, scale, pregFullExe);
                    Add(vregMuls, vregMuls, vregPse, pregFullExe);
                    if constexpr (hasAtten == 1) {
                        DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                            pregCompare, (__ubuf__ uint32_t*&)maskLocalInt, srcN);
                        Select(vregSel, vregMin, vregMuls, pregCompare);
                        FusedExpSub(vregExp, vregSel, vregMax, pregFullExe);
                    } else {
                        FusedExpSub(vregExp, vregMuls, vregMax, pregFullExe);
                    }
                    Div(vregDiv, vregExp, vregSum, pregFullExe);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                        ((__ubuf__ float*&)dstLocalInt), vregDiv, srcN, pregFullExe);

                    // submul
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregDp, ((__ubuf__ float*&)srcDsLocalInt),
                                                                             srcN);
                    if (std::is_same_v<T1, float8_e4m3_t> || std::is_same_v<T1, float8_e5m2_t>) {
                        Muls(vregDp, vregDp, deqScaleDyV, pregFullExe);
                    }
                    Sub(vregSub, vregDp, vregGrad, pregFullExe);
                    Mul(vregMul, vregSub, vregDiv, pregFullExe);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstDsLocalInt), vregMul,
                                                                             srcN, pregFullExe);
                }
                // tailLoop
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail,
                                                                         ((__ubuf__ float*&)srcLocalIntTail), 128);
                // pse inner generate start
                Abs(vregPseTail, vregPseIdxTail, pregFullExe);
                Adds(vregPseIdxTail, vregPseIdxTail, -1.0f, pregFullExe);
                Sqrt(vregPseTail, vregPseTail, pregFullExe);
                Muls(vregPseTail, vregPseTail, slopes, pregFullExe);
                // pse inner generate end
                Muls(vregMulsTail, vregSrcTail, scale, pregTailExe);
                Add(vregMulsTail, vregMulsTail, vregPseTail, pregTailExe);
                if constexpr (hasAtten == 1) {
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        pregCompareTail, (__ubuf__ uint32_t*&)maskLocalIntTail, 128);
                    Select(vregSelTail, vregMin, vregMulsTail, pregCompareTail);
                    FusedExpSub(vregExpTail, vregSelTail, vregMax, pregTailExe);
                } else {
                    FusedExpSub(vregExpTail, vregMulsTail, vregMax, pregTailExe);
                }
                Div(vregDivTail, vregExpTail, vregSum, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                    ((__ubuf__ float*&)dstLocalIntTail), vregDivTail, 128, pregFullExe);
                if constexpr (isDeter == 1 && srcN == 64) { // 确定性计算需要将64~128的数据补零， 否则会有脏数据inf
                    Duplicate(vregDiv, 0);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                        (__ubuf__ float*&)dstLocalIntTailZero, vregDiv, 128, pregFullExe);
                }

                // submul
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregDpTail,
                                                                         ((__ubuf__ float*&)srcDsLocalIntTail), 128);
                if (std::is_same_v<T1, float8_e4m3_t> || std::is_same_v<T1, float8_e5m2_t>) {
                    Muls(vregDpTail, vregDpTail, deqScaleDyV, pregTailExe);
                }
                Sub(vregSubTail, vregDpTail, vregGrad, pregTailExe);
                Mul(vregMulTail, vregSubTail, vregDivTail, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstDsLocalIntTail),
                                                                         vregMulTail, 128, pregFullExe);
                if constexpr (isDeter == 1 && srcN == 64) { // 确定性计算需要将64~128的数据补零，否则会有脏数据inf
                    Duplicate(vregMul, 0);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                        (__ubuf__ float*&)dstDsLocalIntTailZero, vregMul, 128, pregFullExe);
                }
            }
        }
    }
}

/*
@INGROUP MulsSelSimpleSoftMax for DN
brief compute exp(sel(x * scale, mask) - max) / expSum
param [out] dst output LocalTensor
param [in] inSumTensor input expsum of last axis
param [in] inMaxTensor input max value of last axis
param [in] src input LocalTensor
param [in] tiling softmaxtiling
*/
template <typename T1, typename T, uint32_t srcN, const uint32_t CUBE_BASEN = 64, bool hasAtten = 0, bool hasPse = 0,
          bool isDeter = 0>
__aicore__ inline void ComputeSimpleSoftMaxAndSubMulDn(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& dstDsTensor, const LocalTensor<float>& inSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<float>& inGradTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<float>& srcDsTensor, const LocalTensor<T1>& pseTensor, const LocalTensor<uint8_t>& maskTensor,
    const T scale, float deqScaleDyV, const T minValue, uint32_t srcM, uint32_t realN = srcN, uint32_t pseType = 1,
    uint32_t pse_layout_type = 1, float posShift = -1.0, float slopes = -1.0)
{
    const uint32_t fullExeSize = 64;
    uint16_t repeatTimes = srcN / fullExeSize - 1;
    uint32_t tailSize = realN % fullExeSize == 0 ? fullExeSize : realN % fullExeSize;
    uint32_t pseStride = pse_layout_type == 1 ? 0 : srcN;
    uint32_t pseTailStride = pse_layout_type == 1 ? 0 : 128;
    // uint32_t sfmgOffset = CUBE_BASEN * 4;

    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t srcDsLocalInt = srcDsTensor.GetPhyAddr();
    uint64_t dstDsLocalInt = dstDsTensor.GetPhyAddr();
    uint64_t sumLocalInt = inSumTensor.GetPhyAddr();
    uint64_t maxLocalInt = inMaxTensor.GetPhyAddr();
    uint64_t gradLocalInt = inGradTensor.GetPhyAddr();
    uint64_t maskLocalInt = maskTensor.GetPhyAddr();

    uint64_t sumLocalIntTail = inSumTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t maxLocalIntTail = inMaxTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    // uint64_t gradLocalIntTail = inMaxTensor.GetPhyAddr() + sfmgOffset + fullExeSize * 4 * repeatTimes;
    uint64_t gradLocalIntTail = inGradTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t dstLocalIntTail = dstTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t srcDsLocalIntTail = srcDsTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
    uint64_t dstDsLocalIntTail = dstDsTensor.GetPhyAddr() + fullExeSize * 4 * repeatTimes;
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
            RegTensor<float> vregSrc1;
            RegTensor<float> vregMax1;
            RegTensor<float> vregSum1;
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
            // submul
            RegTensor<float> vregGrad;
            RegTensor<float> vregGrad1;
            RegTensor<float> vregDp;
            RegTensor<float> vregMul;
            RegTensor<float> vregDpTail;
            RegTensor<float> vregMulTail;
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
            if (repeatTimes >= 1) {
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregMax, ((__ubuf__ float*&)maxLocalInt),
                                                                         srcN);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSum, ((__ubuf__ float*&)sumLocalInt),
                                                                         srcN);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ float*&)gradLocalInt),
                                                                         srcN);
            }

            DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregMax1, ((__ubuf__ float*&)maxLocalIntTail),
                                                                     srcN);
            DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSum1, ((__ubuf__ float*&)sumLocalIntTail),
                                                                     srcN);
            DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ float*&)gradLocalIntTail),
                                                                     srcN);

            // 64 * 128 max/sum: 64 * 8
            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                // DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                //     vregMax, ((__ubuf__ float *&)maxLocalInt), 8);
                // DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                //     vregSum, ((__ubuf__ float *&)sumLocalInt), 8);
                // DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                //     vregGrad, ((__ubuf__ float *&)gradLocalInt), 1);
                // 1 -> 64 【64，128】 【64，2，64】 64
                for (uint16_t n = 0; n < repeatTimes; n++) {
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float*&)srcLocalInt),
                                                                             srcN);
                    if constexpr (IsSameType<T1, half>::value && hasPse == 1) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseHalf, ((__ubuf__ T1*&)pseLocalInt),
                                                                              pseStride);
                        Interleave(vregPseHalfEven, vregPseHalfOdd, vregPseHalf, vregPseHalf);
                        Cast<float, T1, castTraitB162B32Even>(vregPse, vregPseHalfEven, pregFullExeB16);
                        Add(vregSrc, vregSrc, vregPse, pregFullExe);
                    } else if constexpr (IsSameType<T1, bfloat16_t>::value && hasPse == 1) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseBf, ((__ubuf__ T1*&)pseLocalInt),
                                                                              pseStride);
                        Interleave(vregPseBfEven, vregPseBfOdd, vregPseBf, vregPseBf);
                        Cast<float, T1, castTraitB162B32Even>(vregPse, vregPseBfEven, pregFullExeB16);
                        Add(vregSrc, vregSrc, vregPse, pregFullExe);
                    } else if constexpr (IsSameType<T1, float>::value && hasPse == 1) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPse, ((__ubuf__ T1*&)pseLocalInt),
                                                                              pseStride);
                        Add(vregSrc, vregSrc, vregPse, pregFullExe);
                    }
                    Muls(vregMuls, vregSrc, scale, pregFullExe);
                    if constexpr (hasAtten == 1) {
                        DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                            pregCompare, (__ubuf__ uint32_t*&)maskLocalInt, srcN);
                        // Select(vregSel, vregMin, vregMuls, pregCompare);
                        Select(vregSel, vregMuls, vregMin, pregCompare);
                        FusedExpSub(vregExp, vregSel, vregMax, pregFullExe);
                    } else {
                        FusedExpSub(vregExp, vregMuls, vregMax, pregFullExe);
                    }
                    Div(vregDiv, vregExp, vregSum, pregFullExe);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                        ((__ubuf__ float*&)dstLocalInt), vregDiv, srcN, pregFullExe);

                    // submul
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregDp, ((__ubuf__ float*&)srcDsLocalInt),
                                                                             srcN);
                    if (std::is_same_v<T1, float8_e4m3_t> || std::is_same_v<T1, float8_e5m2_t>) {
                        Muls(vregDp, vregDp, deqScaleDyV, pregFullExe);
                    }
                    Sub(vregSub, vregDp, vregGrad, pregFullExe);
                    Mul(vregMul, vregSub, vregDiv, pregFullExe);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstDsLocalInt), vregMul,
                                                                             srcN, pregFullExe);
                }
                // tailLoop
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail,
                                                                         ((__ubuf__ float*&)srcLocalIntTail), 128);
                if constexpr (IsSameType<T1, half>::value && hasPse == 1) {
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        vregPseHalfTail, ((__ubuf__ T1*&)pseLocalIntTail), pseTailStride);
                    Interleave(vregPseHalfTailEven, vregPseHalfTailOdd, vregPseHalfTail, vregPseHalfTail);
                    Cast<float, T1, castTraitB162B32Even>(vregPseTail, vregPseHalfTailEven, pregFullExeB16);
                    Add(vregSrcTail, vregSrcTail, vregPseTail, pregTailExe);
                } else if constexpr (IsSameType<T1, bfloat16_t>::value && hasPse == 1) {
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        vregPseBfTail, ((__ubuf__ T1*&)pseLocalIntTail), pseTailStride);
                    Interleave(vregPseBfTailEven, vregPseBfTailOdd, vregPseBfTail, vregPseBfTail);
                    Cast<float, T1, castTraitB162B32Even>(vregPseTail, vregPseBfTailEven, pregFullExeB16);
                    Add(vregSrcTail, vregSrcTail, vregPseTail, pregTailExe);
                } else if constexpr (IsSameType<T1, float>::value && hasPse == 1) {
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseTail, ((__ubuf__ T1*&)pseLocalIntTail),
                                                                          pseTailStride);
                    Add(vregSrcTail, vregSrcTail, vregPseTail, pregTailExe);
                }
                Muls(vregMulsTail, vregSrcTail, scale, pregTailExe);
                if constexpr (hasAtten == 1) {
                    DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                        pregCompareTail, (__ubuf__ uint32_t*&)maskLocalIntTail, 128);
                    // Select(vregSelTail, vregMin, vregMulsTail, pregCompareTail);
                    Select(vregSelTail, vregMulsTail, vregMin, pregCompareTail);
                    FusedExpSub(vregExpTail, vregSelTail, vregMax1, pregTailExe);
                } else {
                    FusedExpSub(vregExpTail, vregMulsTail, vregMax1, pregTailExe);
                }
                Div(vregDivTail, vregExpTail, vregSum1, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                    ((__ubuf__ float*&)dstLocalIntTail), vregDivTail, 128, pregFullExe);
                if constexpr (isDeter == 1 && srcN == 64) { // 确定性计算需要将64~128的数据补零， 否则会有脏数据inf
                    Duplicate(vregDiv, 0);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                        (__ubuf__ float*&)dstLocalIntTailZero, vregDiv, 128, pregFullExe);
                }

                // submul
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregDpTail,
                                                                         ((__ubuf__ float*&)srcDsLocalIntTail), 128);
                if (std::is_same_v<T1, float8_e4m3_t> || std::is_same_v<T1, float8_e5m2_t>) {
                    Muls(vregDpTail, vregDpTail, deqScaleDyV, pregTailExe);
                }
                Sub(vregSubTail, vregDpTail, vregGrad1, pregTailExe);
                Mul(vregMulTail, vregSubTail, vregDivTail, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstDsLocalIntTail),
                                                                         vregMulTail, 128, pregFullExe);
            }
        }
    }
}

#else
template <typename T1, typename T, uint32_t srcN, const uint32_t VECTOR_BASEM = 64, bool hasAtten = 0, bool hasPse = 0,
          bool isDeter = 0>
__aicore__ inline void ComputeSimpleSoftMaxAndSubMul(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& dstDsTensor, const LocalTensor<float>& inSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<float>& srcTensor, const LocalTensor<float>& srcDsTensor,
    const LocalTensor<T1>& pseTensor, const LocalTensor<uint8_t>& maskTensor, const T scale, float deqScaleDyV,
    const T minValue, uint32_t srcM, uint32_t realN = srcN, uint32_t pseType = 1, uint32_t pse_layout_type = 1,
    float posShift = -1.0, float slopes = -1.0)
{}
#endif
} // namespace AscendC

#endif // MY_SIMPLE_SOFTMAX_INTERFACE_DAVID_H
