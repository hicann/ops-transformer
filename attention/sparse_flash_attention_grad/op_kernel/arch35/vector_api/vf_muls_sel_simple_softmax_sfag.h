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
 * \file vf_muls_sel_simple_softmax_sfag.h
 */
#ifndef VF_MULS_SEL_SIMPLE_SOFTMAX_SFAG_H
#define VF_MULS_SEL_SIMPLE_SOFTMAX_SFAG_H
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
template <typename T1, typename T, uint32_t srcN, bool hasAtten, bool hasPse, bool isDeter>
__simd_vf__ inline void MulsSelVFPseType1(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t sumLocalInt,
                                          uint64_t maxLocalInt, uint64_t maskLocalInt, uint64_t srcLocalIntTail,
                                          uint64_t dstLocalIntTail, uint64_t dstLocalIntTailZero,
                                          uint64_t maskLocalIntTail, uint64_t pseLocalInt, uint64_t pseLocalIntTail,
                                          uint32_t tailSize, uint32_t pseStride, uint16_t repeatTimes, const T scale,
                                          const T minValue, uint32_t pseTailStride, uint32_t srcM)
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
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregMax, ((__ubuf__ float *&)maxLocalInt), 8);
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregSum, ((__ubuf__ float *&)sumLocalInt), 8);
        // 1 -> 64 【64，128】 【64，2，64】 64
        for (uint16_t n = 0; n < repeatTimes; n++) {
            LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float *&)srcLocalInt), srcN);
            if constexpr (IsSameType<T1, half>::value && hasPse == 1) {
                LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseHalf, ((__ubuf__ T1 *&)pseLocalInt),
                                                                       pseStride);
                Interleave(vregPseHalfEven, vregPseHalfOdd, vregPseHalf, vregPseHalf);
                Cast<float, T1, castTraitB162B32Even>(vregPse, vregPseHalfEven, pregFullExeB16);
                Add(vregSrc, vregSrc, vregPse, pregFullExe);
            } else if constexpr (IsSameType<T1, bfloat16_t>::value && hasPse == 1) {
                LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseBf, ((__ubuf__ T1 *&)pseLocalInt),
                                                                       pseStride);
                Interleave(vregPseBfEven, vregPseBfOdd, vregPseBf, vregPseBf);
                Cast<float, T1, castTraitB162B32Even>(vregPse, vregPseBfEven, pregFullExeB16);
                Add(vregSrc, vregSrc, vregPse, pregFullExe);
            } else if constexpr (IsSameType<T1, float>::value && hasPse == 1) {
                LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPse, ((__ubuf__ T1 *&)pseLocalInt),
                                                                       pseStride);
                Add(vregSrc, vregSrc, vregPse, pregFullExe);
            }
            Muls(vregMuls, vregSrc, scale, pregFullExe);
            if constexpr (hasAtten == 1) {
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    pregCompare, (__ubuf__ uint32_t *&)maskLocalInt, srcN);
                Select(vregSel, vregMin, vregMuls, pregCompare);
                Sub(vregSub, vregSel, vregMax, pregFullExe);
            } else {
                Sub(vregSub, vregMuls, vregMax, pregFullExe);
            }
            Exp(vregExp, vregSub, pregFullExe);
            Div(vregDiv, vregExp, vregSum, pregFullExe);
            StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                ((__ubuf__ float *&)dstLocalInt), vregDiv, srcN, pregFullExe);
        }
        // tailLoop
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ float *&)srcLocalIntTail),
                                                                  128);
        if constexpr (IsSameType<T1, half>::value && hasPse == 1) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseHalfTail, ((__ubuf__ T1 *&)pseLocalIntTail),
                                                                   pseTailStride);
            Interleave(vregPseHalfTailEven, vregPseHalfTailOdd, vregPseHalfTail, vregPseHalfTail);
            Cast<float, T1, castTraitB162B32Even>(vregPseTail, vregPseHalfTailEven, pregFullExeB16);
            Add(vregSrcTail, vregSrcTail, vregPseTail, pregTailExe);
        } else if constexpr (IsSameType<T1, bfloat16_t>::value && hasPse == 1) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseBfTail, ((__ubuf__ T1 *&)pseLocalIntTail),
                                                                   pseTailStride);
            Interleave(vregPseBfTailEven, vregPseBfTailOdd, vregPseBfTail, vregPseBfTail);
            Cast<float, T1, castTraitB162B32Even>(vregPseTail, vregPseBfTailEven, pregFullExeB16);
            Add(vregSrcTail, vregSrcTail, vregPseTail, pregTailExe);
        } else if constexpr (IsSameType<T1, float>::value && hasPse == 1) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseTail, ((__ubuf__ T1 *&)pseLocalIntTail),
                                                                   pseTailStride);
            Add(vregSrcTail, vregSrcTail, vregPseTail, pregTailExe);
        }
        Muls(vregMulsTail, vregSrcTail, scale, pregTailExe);
        if constexpr (hasAtten == 1) {
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                pregCompareTail, (__ubuf__ uint32_t *&)maskLocalIntTail, 128);
            Select(vregSelTail, vregMin, vregMulsTail, pregCompareTail);
            Sub(vregSubTail, vregSelTail, vregMax, pregTailExe);
        } else {
            Sub(vregSubTail, vregMulsTail, vregMax, pregTailExe);
        }
        Exp(vregExpTail, vregSubTail, pregTailExe);
        Div(vregDivTail, vregExpTail, vregSum, pregTailExe);
        StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
            ((__ubuf__ float *&)dstLocalIntTail), vregDivTail, 128, pregFullExe);
        if constexpr (isDeter == 1 && srcN == 64) { // 确定性计算需要将64~128的数据补零， 否则会有脏数据inf
            Duplicate(vregDiv, 0);
            StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ float *&)dstLocalIntTailZero, vregDiv, 128, pregFullExe);
        }
    }
}

template <typename T1, typename T, uint32_t srcN, bool hasAtten, bool hasPse, bool isDeter>
__simd_vf__ inline void MulsSelVFPseType0(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t sumLocalInt,
                                          uint64_t maxLocalInt, uint64_t maskLocalInt, uint64_t srcLocalIntTail,
                                          uint64_t dstLocalIntTail, uint64_t dstLocalIntTailZero,
                                          uint64_t maskLocalIntTail, uint64_t pseLocalInt, uint64_t pseLocalIntTail,
                                          uint32_t tailSize, uint32_t pseStride, uint16_t repeatTimes, const T scale,
                                          const T minValue, uint32_t pseTailStride, uint32_t srcM)
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
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregMax, ((__ubuf__ float *&)maxLocalInt), 8);
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregSum, ((__ubuf__ float *&)sumLocalInt), 8);
        // 1 -> 64 【64，128】 【64，2，64】 64
        for (uint16_t n = 0; n < repeatTimes; n++) {
            LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float *&)srcLocalInt), srcN);
            Muls(vregMuls, vregSrc, scale, pregFullExe);
            if constexpr (IsSameType<T1, half>::value && hasPse == 1) {
                LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseHalf, ((__ubuf__ T1 *&)pseLocalInt),
                                                                       pseStride);
                Interleave(vregPseHalfEven, vregPseHalfOdd, vregPseHalf, vregPseHalf);
                Cast<float, T1, castTraitB162B32Even>(vregPse, vregPseHalfEven, pregFullExeB16);
                Add(vregMuls, vregMuls, vregPse, pregFullExe);
            } else if constexpr (IsSameType<T1, bfloat16_t>::value && hasPse == 1) {
                LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseBf, ((__ubuf__ T1 *&)pseLocalInt),
                                                                       pseStride);
                Interleave(vregPseBfEven, vregPseBfOdd, vregPseBf, vregPseBf);
                Cast<float, T1, castTraitB162B32Even>(vregPse, vregPseBfEven, pregFullExeB16);
                Add(vregMuls, vregMuls, vregPse, pregFullExe);
            } else if constexpr (IsSameType<T1, float>::value && hasPse == 1) {
                LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPse, ((__ubuf__ T1 *&)pseLocalInt),
                                                                       pseStride);
                Add(vregMuls, vregMuls, vregPse, pregFullExe);
            }
            if constexpr (hasAtten == 1) {
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    pregCompare, (__ubuf__ uint32_t *&)maskLocalInt, srcN);
                Select(vregSel, vregMin, vregMuls, pregCompare);
                Sub(vregSub, vregSel, vregMax, pregFullExe);
            } else {
                Sub(vregSub, vregMuls, vregMax, pregFullExe);
            }
            Exp(vregExp, vregSub, pregFullExe);
            Div(vregDiv, vregExp, vregSum, pregFullExe);
            StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                ((__ubuf__ float *&)dstLocalInt), vregDiv, srcN, pregFullExe);
        }
        // tailLoop
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ float *&)srcLocalIntTail),
                                                                  128);
        Muls(vregMulsTail, vregSrcTail, scale, pregTailExe);
        if constexpr (IsSameType<T1, half>::value && hasPse == 1) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseHalfTail, ((__ubuf__ T1 *&)pseLocalIntTail),
                                                                   pseTailStride);
            Interleave(vregPseHalfTailEven, vregPseHalfTailOdd, vregPseHalfTail, vregPseHalfTail);
            Cast<float, T1, castTraitB162B32Even>(vregPseTail, vregPseHalfTailEven, pregFullExeB16);
            Add(vregMulsTail, vregMulsTail, vregPseTail, pregTailExe);
        } else if constexpr (IsSameType<T1, bfloat16_t>::value && hasPse == 1) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseBfTail, ((__ubuf__ T1 *&)pseLocalIntTail),
                                                                   pseTailStride);
            Interleave(vregPseBfTailEven, vregPseBfTailOdd, vregPseBfTail, vregPseBfTail);
            Cast<float, T1, castTraitB162B32Even>(vregPseTail, vregPseBfTailEven, pregFullExeB16);
            Add(vregMulsTail, vregMulsTail, vregPseTail, pregTailExe);
        } else if constexpr (IsSameType<T1, float>::value && hasPse == 1) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregPseTail, ((__ubuf__ T1 *&)pseLocalIntTail),
                                                                   pseTailStride);
            Add(vregMulsTail, vregMulsTail, vregPseTail, pregTailExe);
        }
        if constexpr (hasAtten == 1) {
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                pregCompareTail, (__ubuf__ uint32_t *&)maskLocalIntTail, 128);
            Select(vregSelTail, vregMin, vregMulsTail, pregCompareTail);
            Sub(vregSubTail, vregSelTail, vregMax, pregTailExe);
        } else {
            Sub(vregSubTail, vregMulsTail, vregMax, pregTailExe);
        }
        Exp(vregExpTail, vregSubTail, pregTailExe);
        Div(vregDivTail, vregExpTail, vregSum, pregTailExe);
        StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
            ((__ubuf__ float *&)dstLocalIntTail), vregDivTail, 128, pregFullExe);
        if constexpr (isDeter == 1 && srcN == 64) { // 确定性计算需要将64~128的数据补零， 否则会有脏数据inf
            Duplicate(vregDiv, 0);
            StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ float *&)dstLocalIntTailZero, vregDiv, 128, pregFullExe);
        }
    }
}

template <typename T, uint32_t srcN, bool hasAtten, bool hasPse, bool isDeter>
__simd_vf__ inline void MulsSelVFPseType2(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t sumLocalInt,
                                          uint64_t maxLocalInt, uint64_t maskLocalInt, uint64_t srcLocalIntTail,
                                          uint64_t dstLocalIntTail, uint64_t dstLocalIntTailZero,
                                          uint64_t maskLocalIntTail, uint64_t pseLocalInt, uint64_t pseLocalIntTail,
                                          uint32_t tailSize, uint32_t pseStride, uint16_t repeatTimes,
                                          uint32_t tailStartIdx, const T scale, const T minValue, float posShift,
                                          float slopes, uint32_t srcM)
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
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregMax, ((__ubuf__ float *&)maxLocalInt), 8);
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregSum, ((__ubuf__ float *&)sumLocalInt), 8);
        // 1 -> 64 【64，128】 【64，2，64】 64
        for (uint16_t n = 0; n < repeatTimes; n++) {
            LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float *&)srcLocalInt), srcN);
            // pse inner generate start
            Abs(vregPse, vregPseIdx, pregFullExe);
            Adds(vregPseIdx, vregPseIdx, -1.0f, pregFullExe);
            Muls(vregPse, vregPse, slopes, pregFullExe);
            // pse inner generate end
            Muls(vregMuls, vregSrc, scale, pregFullExe);
            Add(vregMuls, vregMuls, vregPse, pregFullExe);
            if constexpr (hasAtten == 1) {
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    pregCompare, (__ubuf__ uint32_t *&)maskLocalInt, srcN);
                Select(vregSel, vregMin, vregMuls, pregCompare);
                Sub(vregSub, vregSel, vregMax, pregFullExe);
            } else {
                Sub(vregSub, vregMuls, vregMax, pregFullExe);
            }
            Exp(vregExp, vregSub, pregFullExe);
            Div(vregDiv, vregExp, vregSum, pregFullExe);
            StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                ((__ubuf__ float *&)dstLocalInt), vregDiv, srcN, pregFullExe);
        }
        // tailLoop
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ float *&)srcLocalIntTail),
                                                                  128);
        // pse inner generate start
        Abs(vregPseTail, vregPseIdxTail, pregFullExe);
        Adds(vregPseIdxTail, vregPseIdxTail, -1.0f, pregFullExe);
        Muls(vregPseTail, vregPseTail, slopes, pregFullExe);
        // pse inner generate end
        Muls(vregMulsTail, vregSrcTail, scale, pregTailExe);
        Add(vregMulsTail, vregMulsTail, vregPseTail, pregTailExe);
        if constexpr (hasAtten == 1) {
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                pregCompareTail, (__ubuf__ uint32_t *&)maskLocalIntTail, 128);
            Select(vregSelTail, vregMin, vregMulsTail, pregCompareTail);
            Sub(vregSubTail, vregSelTail, vregMax, pregTailExe);
        } else {
            Sub(vregSubTail, vregMulsTail, vregMax, pregTailExe);
        }
        Exp(vregExpTail, vregSubTail, pregTailExe);
        Div(vregDivTail, vregExpTail, vregSum, pregTailExe);
        StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
            ((__ubuf__ float *&)dstLocalIntTail), vregDivTail, 128, pregFullExe);
        if constexpr (isDeter == 1 && srcN == 64) { // 确定性计算需要将64~128的数据补零， 否则会有脏数据inf
            Duplicate(vregDiv, 0);
            StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ float *&)dstLocalIntTailZero, vregDiv, 128, pregFullExe);
        }
    }
}

template <typename T, uint32_t srcN, bool hasAtten, bool hasPse, bool isDeter>
__simd_vf__ inline void MulsSelVFPseType3(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t sumLocalInt,
                                          uint64_t maxLocalInt, uint64_t maskLocalInt, uint64_t srcLocalIntTail,
                                          uint64_t dstLocalIntTail, uint64_t dstLocalIntTailZero,
                                          uint64_t maskLocalIntTail, uint64_t pseLocalInt, uint64_t pseLocalIntTail,
                                          uint32_t tailSize, uint32_t pseStride, uint16_t repeatTimes,
                                          uint32_t tailStartIdx, const T scale, const T minValue, float posShift,
                                          float slopes, uint32_t srcM)
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
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregMax, ((__ubuf__ float *&)maxLocalInt), 8);
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregSum, ((__ubuf__ float *&)sumLocalInt), 8);
        // 1 -> 64 【64，128】 【64，2，64】 64
        for (uint16_t n = 0; n < repeatTimes; n++) {
            LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float *&)srcLocalInt), srcN);
            // pse inner generate start
            Abs(vregPse, vregPseIdx, pregFullExe);
            Adds(vregPseIdx, vregPseIdx, -1.0f, pregFullExe);
            Sqrt(vregPse, vregPse, pregFullExe);
            Muls(vregPse, vregPse, slopes, pregFullExe);
            // pse inner generate end
            Muls(vregMuls, vregSrc, scale, pregFullExe);
            Add(vregMuls, vregMuls, vregPse, pregFullExe);
            if constexpr (hasAtten == 1) {
                LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                    pregCompare, (__ubuf__ uint32_t *&)maskLocalInt, srcN);
                Select(vregSel, vregMin, vregMuls, pregCompare);
                Sub(vregSub, vregSel, vregMax, pregFullExe);
            } else {
                Sub(vregSub, vregMuls, vregMax, pregFullExe);
            }
            Exp(vregExp, vregSub, pregFullExe);
            Div(vregDiv, vregExp, vregSum, pregFullExe);
            StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                ((__ubuf__ float *&)dstLocalInt), vregDiv, srcN, pregFullExe);
        }
        // tailLoop
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ float *&)srcLocalIntTail),
                                                                  128);
        // pse inner generate start
        Abs(vregPseTail, vregPseIdxTail, pregFullExe);
        Adds(vregPseIdxTail, vregPseIdxTail, -1.0f, pregFullExe);
        Sqrt(vregPseTail, vregPseTail, pregFullExe);
        Muls(vregPseTail, vregPseTail, slopes, pregFullExe);
        // pse inner generate end
        Muls(vregMulsTail, vregSrcTail, scale, pregTailExe);
        Add(vregMulsTail, vregMulsTail, vregPseTail, pregTailExe);
        if constexpr (hasAtten == 1) {
            LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_DS>(
                pregCompareTail, (__ubuf__ uint32_t *&)maskLocalIntTail, 128);
            Select(vregSelTail, vregMin, vregMulsTail, pregCompareTail);
            Sub(vregSubTail, vregSelTail, vregMax, pregTailExe);
        } else {
            Sub(vregSubTail, vregMulsTail, vregMax, pregTailExe);
        }
        Exp(vregExpTail, vregSubTail, pregTailExe);
        Div(vregDivTail, vregExpTail, vregSum, pregTailExe);
        StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
            ((__ubuf__ float *&)dstLocalIntTail), vregDivTail, 128, pregFullExe);
        if constexpr (isDeter == 1 && srcN == 64) { // 确定性计算需要将64~128的数据补零， 否则会有脏数据inf
            Duplicate(vregDiv, 0);
            StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ float *&)dstLocalIntTailZero, vregDiv, 128, pregFullExe);
        }
    }
}

template <typename T1, typename T, uint32_t srcN, bool hasAtten = 0, bool hasPse = 0, bool isDeter = 0>
__aicore__ inline void MulsSelSimpleSoftMax(const LocalTensor<float> &dstTensor, const LocalTensor<float> &inSumTensor,
                                            const LocalTensor<float> &inMaxTensor, const LocalTensor<float> &srcTensor,
                                            const LocalTensor<T1> &pseTensor, const LocalTensor<uint8_t> &maskTensor,
                                            const T scale, const T minValue, uint32_t srcM, uint32_t realN = srcN,
                                            uint32_t pseType = 1, uint32_t pse_layout_type = 1, float posShift = -1.0,
                                            float slopes = -1.0)
{
    const uint32_t fullExeSize = 64;
    uint16_t repeatTimes = srcN / fullExeSize - 1;
    uint32_t tailSize = realN % fullExeSize == 0 ? fullExeSize : realN % fullExeSize;
    uint32_t pseStride = pse_layout_type == 1 ? 0 : srcN;
    uint32_t pseTailStride = pse_layout_type == 1 ? 0 : 128;

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
        MulsSelVFPseType1<T1, T, srcN, hasAtten, hasPse, isDeter>(
            srcLocalInt, dstLocalInt, sumLocalInt, maxLocalInt, maskLocalInt, srcLocalIntTail, dstLocalIntTail,
            dstLocalIntTailZero, maskLocalIntTail, pseLocalInt, pseLocalIntTail, tailSize, pseStride, repeatTimes,
            scale, minValue, pseTailStride, srcM);
    } else if (pseType == 0) {
        MulsSelVFPseType0<T1, T, srcN, hasAtten, hasPse, isDeter>(
            srcLocalInt, dstLocalInt, sumLocalInt, maxLocalInt, maskLocalInt, srcLocalIntTail, dstLocalIntTail,
            dstLocalIntTailZero, maskLocalIntTail, pseLocalInt, pseLocalIntTail, tailSize, pseStride, repeatTimes,
            scale, minValue, pseTailStride, srcM);
    } else if (pseType == 2) {
        uint32_t tailStartIdx = repeatTimes == 0 ? 0 : fullExeSize;
        MulsSelVFPseType2<T, srcN, hasAtten, hasPse, isDeter>(
            srcLocalInt, dstLocalInt, sumLocalInt, maxLocalInt, maskLocalInt, srcLocalIntTail, dstLocalIntTail,
            dstLocalIntTailZero, maskLocalIntTail, pseLocalInt, pseLocalIntTail, tailSize, pseStride, repeatTimes,
            tailStartIdx, scale, minValue, posShift, slopes, srcM);
    } else if (pseType == 3) {
        uint32_t tailStartIdx = repeatTimes == 0 ? 0 : fullExeSize;
        MulsSelVFPseType3<T, srcN, hasAtten, hasPse, isDeter>(
            srcLocalInt, dstLocalInt, sumLocalInt, maxLocalInt, maskLocalInt, srcLocalIntTail, dstLocalIntTail,
            dstLocalIntTailZero, maskLocalIntTail, pseLocalInt, pseLocalIntTail, tailSize, pseStride, repeatTimes,
            tailStartIdx, scale, minValue, posShift, slopes, srcM);
    }
}
#else
template <typename T1, typename T, uint32_t srcN, bool hasAtten = 0, bool hasPse = 0, bool isDeter = 0>
__aicore__ inline void MulsSelSimpleSoftMax(const LocalTensor<float> &dstTensor, const LocalTensor<float> &inSumTensor,
                                            const LocalTensor<float> &inMaxTensor, const LocalTensor<float> &srcTensor,
                                            const LocalTensor<T1> &pseTensor, const LocalTensor<uint8_t> &maskTensor,
                                            const T scale, const T minValue, uint32_t srcM, uint32_t realN = srcN,
                                            uint32_t pseType = 1, uint32_t pse_layout_type = 1, float posShift = -1.0,
                                            float slopes = -1.0)
{}
#endif
} // namespace AscendC

#endif // MY_SIMPLE_SOFTMAX_INTERFACE_H
