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
 * \file vf_flash_decode_arch38.h
 * \brief
 */
#ifndef MY_FLASH_DECODE_ARCH38_H
#define MY_FLASH_DECODE_ARCH38_H

#include "kernel_tensor.h"

constexpr float FLT_ZERO = 0;
constexpr float FLT_MAX_NEW = 3.402823466e+38F;

namespace FaVectorApi {

// 处理循环splitKVIndex=0的场景，vregDst需要置0
template <typename T>
__aicore__ inline void ReduceFinalRes_VF_0(LocalTensor<T> &dstLocal, LocalTensor<T> &lseLocal,
                                           LocalTensor<T> &accumOutLocal, uint32_t dealRowCount,
                                           uint64_t headDimAlignFp32, uint32_t splitKVIndex)
{
    uint64_t dstUb = dstLocal.GetPhyAddr();
    uint64_t lseUb = lseLocal.GetPhyAddr();
    uint64_t accumOutUb = accumOutLocal.GetPhyAddr();
    uint16_t k = 0;
    uint16_t z = 0;
    uint32_t dealNum1Reg = 256 / sizeof(float);
    uint32_t repStride = headDimAlignFp32 / 8;
    const uint16_t floatRepSize = 64;
    const uint16_t dLoops = headDimAlignFp32 / floatRepSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vregDst;
        Reg::RegTensor<T> vregLse;
        Reg::RegTensor<T> vregAccumOut;
        uint32_t n = dealNum1Reg;
        Reg::MaskReg pregTailN = Reg::UpdateMask<T>(n);

        for (k = 0; k < static_cast<uint16_t>(dealRowCount); k++) { // repeat g

            Reg::DataCopy<T, Reg::LoadDist::DIST_BLK>(
                vregLse, (__ubuf__ float *&)lseUb + splitKVIndex * dealRowCount * 8 + k * 8);
            for (z = 0; z < dLoops; z++) {
                // splitKVIndex=0的场景，vregDst不需要load，直接置0
                Reg::Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vregDst, FLT_ZERO, pregTailN);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(
                    vregAccumOut, (__ubuf__ float *&)accumOutUb + k * repStride * 8 + z * floatRepSize);
                Reg::Mul<T, Reg::MaskMergeMode::ZEROING>(vregAccumOut, vregLse, vregAccumOut, pregTailN);
                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregDst, vregDst, vregAccumOut, pregTailN);
                Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ float *&)dstUb + k * repStride * 8 + z * floatRepSize, vregDst, pregTailN);
            }
        }
    }
}

// 处理循环splitKVIndex>0的场景，reg_dst需要先从dstUb中load之前的结果，再进行add
template <typename T>
__aicore__ inline void ReduceFinalRes_VF_rest(LocalTensor<T> &dstLocal, LocalTensor<T> &lseLocal,
                                              LocalTensor<T> &accumOutLocal, uint32_t dealRowCount,
                                              uint64_t headDimAlignFp32, uint32_t splitKVIndex)
{
    uint64_t dstUb = dstLocal.GetPhyAddr();
    uint64_t lseUb = lseLocal.GetPhyAddr();
    uint64_t accumOutUb = accumOutLocal.GetPhyAddr();
    uint16_t k = 0;
    uint16_t z = 0;
    uint32_t dealNum1Reg = 256 / sizeof(float);
    uint32_t repStride = headDimAlignFp32 / 8;
    const uint16_t floatRepSize = 64;
    const uint16_t dLoops = headDimAlignFp32 / floatRepSize;

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vregDst;
        Reg::RegTensor<T> vregLse;
        Reg::RegTensor<T> vregAccumOut;
        uint32_t n = dealNum1Reg;
        Reg::MaskReg pregTailN = Reg::UpdateMask<T>(n);
        uint32_t stride = (0x1 << 16) | 0x8;

        for (k = 0; k < static_cast<uint16_t>(dealRowCount); k++) { // repeat g
            Reg::DataCopy<T, Reg::LoadDist::DIST_BLK>(
                vregLse, (__ubuf__ float *&)lseUb + splitKVIndex * dealRowCount * 8 + k * 8);
            for (z = 0; z < dLoops; z++) {
                // splitKVIndex>0的场景，reg_dst需要先从dstUb中load之前的结果，再进行add
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(
                    vregDst, (__ubuf__ float *&)dstUb + k * repStride * 8 + z * floatRepSize);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(
                    vregAccumOut, (__ubuf__ float *&)accumOutUb + k * repStride * 8 + z * floatRepSize);
                Reg::Mul<T, Reg::MaskMergeMode::ZEROING>(vregAccumOut, vregLse, vregAccumOut, pregTailN);
                Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregDst, vregDst, vregAccumOut, pregTailN);
                Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>(
                    (__ubuf__ float *&)dstUb + k * repStride * 8 + z * floatRepSize, vregDst, pregTailN);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ReduceFinalRes_VF(LocalTensor<T> &dstLocal, LocalTensor<T> &lseLocal,
                                         LocalTensor<T> &accumOutLocal, uint32_t dealRowCount,
                                         uint64_t headDimAlignFp32, uint32_t splitKVIndex)
{
    if (splitKVIndex == 0) {
        ReduceFinalRes_VF_0(dstLocal, lseLocal, accumOutLocal, dealRowCount, headDimAlignFp32, splitKVIndex);
    } else {
        ReduceFinalRes_VF_rest(dstLocal, lseLocal, accumOutLocal, dealRowCount, headDimAlignFp32, splitKVIndex);
    }
}

template <typename T, uint32_t headDimAlignFp32 = 0>
__aicore__ inline void ReduceFinalRes_const_VF(LocalTensor<T> &dstLocal, LocalTensor<T> &lseLocal,
                                               LocalTensor<T> &accumOutLocal, uint32_t dealRowCount,
                                               uint32_t splitKVIndex)
{
    if (splitKVIndex == 0) {
        ReduceFinalRes_VF_0(dstLocal, lseLocal, accumOutLocal, dealRowCount, headDimAlignFp32, splitKVIndex);
    } else {
        ReduceFinalRes_VF_rest(dstLocal, lseLocal, accumOutLocal, dealRowCount, headDimAlignFp32, splitKVIndex);
    }
}
// 处理g<=8的场景
template <typename T>
__aicore__ inline void ComputeScaleValue_VF_8(const LocalTensor<T> &lseMaxUb, const LocalTensor<T> &lseSumUb,
                                              const LocalTensor<T> &lseOutputUb, uint32_t dealRowCount,
                                              uint32_t actualCombineLoopSize, bool softmaxLseFlag)
{
    uint32_t dealCount = dealRowCount * 8;
    uint16_t i = 0;

    uint64_t lseMax = lseMaxUb.GetPhyAddr();
    uint64_t lseMaxTmp = lseMax;
    uint64_t lseSum = lseSumUb.GetPhyAddr();
    uint64_t lseSumTmp = lseSum;
    __ubuf__ T *lseUb = (__ubuf__ T *)lseOutputUb.GetPhyAddr();

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vregLseMax;
        Reg::RegTensor<T> vregLseMaxTmp;
        Reg::RegTensor<T> vregLseSum;
        Reg::RegTensor<T> vregLseSumTmp;
        Reg::RegTensor<T> vregRes;
        uint32_t n = dealCount;
        Reg::MaskReg pregTailN = Reg::UpdateMask<T>(n);
        uint16_t blockStride = 0x1;
        uint16_t repeatStride = dealRowCount;

        Reg::Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vregLseMax, -FLT_MAX_NEW, pregTailN);
        Reg::Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vregLseSum, FLT_ZERO, pregTailN);

        for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregLseMaxTmp, (__ubuf__ float *&)lseMaxTmp + i * dealCount);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(vregLseMax, vregLseMax, vregLseMaxTmp, pregTailN);
        }

        for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregLseMaxTmp, (__ubuf__ float *&)lseMaxTmp + i * dealCount);
            Reg::Sub<T, Reg::MaskMergeMode::ZEROING>(vregLseMaxTmp, vregLseMaxTmp, vregLseMax, pregTailN);
            Reg::Exp<T, Reg::MaskMergeMode::ZEROING>(vregLseMaxTmp, vregLseMaxTmp, pregTailN);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregLseSumTmp, (__ubuf__ float *&)lseSumTmp + i * dealCount);
            Reg::Mul<T, Reg::MaskMergeMode::ZEROING>(vregLseSumTmp, vregLseSumTmp, vregLseMaxTmp, pregTailN);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregLseSum, vregLseSum, vregLseSumTmp, pregTailN);
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ float *&)lseSumTmp + i * dealCount, vregLseSumTmp,
                                                            pregTailN);
        }

        if (softmaxLseFlag) {
            Reg::RegTensor<float> vregMinValue;
            Reg::RegTensor<float> vregInfValue;
            Reg::MaskReg pregCompare;
            constexpr float infValue = 3e+99; // 3e+99 for float inf
            constexpr uint32_t tmpMin = 0xFF167699;
            float minValue = *((float *)&tmpMin);
            Reg::Duplicate<float, float>(vregMinValue, minValue);
            Reg::Duplicate<float, float>(vregInfValue, infValue);

            Reg::Log<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregLseSum, pregTailN);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregRes, vregLseMax, pregTailN);
            // 如果 softmaxMax 等于负无穷，则将 lse 结果置为 inf
            Reg::Compare<float, CMPMODE::EQ>(pregCompare, vregLseMax, vregMinValue, pregTailN);
            Reg::Select<T>(vregRes, vregInfValue, vregRes, pregCompare);
            Reg::DataCopy<T, StoreDist::DIST_NORM_B32>(lseUb, vregRes, pregTailN);
        }

        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregLseSumTmp, (__ubuf__ float *&)lseSumTmp + i * dealCount);
            Reg::Div<T, Reg::MaskMergeMode::ZEROING>(vregLseSumTmp, vregLseSumTmp, vregLseSum, pregTailN);
            Reg::DataCopy<T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ float *&)lseSum, vregLseSumTmp, blockStride, repeatStride, pregTailN);
        }
    }
}

// 处理8<g<=16的场景
template <typename T>
__aicore__ inline void ComputeScaleValue_VF_16(const LocalTensor<T> &lseMaxUb, const LocalTensor<T> &lseSumUb,
                                               const LocalTensor<T> &lseOutputUb, uint32_t dealRowCount,
                                               uint32_t actualCombineLoopSize, bool softmaxLseFlag)
{
    uint32_t dealCountSum = dealRowCount * 8;
    uint32_t dealCount = 8 * 8;
    uint32_t dealCount2 = dealCountSum - dealCount;
    uint16_t i = 0;

    __ubuf__ T *lseMax = (__ubuf__ T *)lseMaxUb.GetPhyAddr();
    __ubuf__ T *lseMax2 = lseMax + 64;
    __ubuf__ T *lseMaxSrc = lseMax;
    __ubuf__ T *lseSum = (__ubuf__ T *)lseSumUb.GetPhyAddr();
    __ubuf__ T *lseSum2 = lseSum + 64;
    __ubuf__ T *lseSumSrc = lseSum;
    __ubuf__ T *lseUb = (__ubuf__ T *)lseOutputUb.GetPhyAddr();
    __ubuf__ T *lseUb2 = lseUb + 64;

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vregLseMax;
        Reg::RegTensor<T> vregLseMaxTmp;
        Reg::RegTensor<T> vregLseMax2;
        Reg::RegTensor<T> vregLseMaxTmp2;
        Reg::RegTensor<T> vregLseSum;
        Reg::RegTensor<T> vregLseSumTmp;
        Reg::RegTensor<T> vregLseSum2;
        Reg::RegTensor<T> vregLseSumTmp2;
        Reg::RegTensor<T> vregRes;
        Reg::RegTensor<T> vregRes2;
        uint32_t n = dealCount;
        uint32_t n2 = dealCount2;
        Reg::MaskReg pregTailN = Reg::UpdateMask<T>(n);
        Reg::MaskReg pregTailN2 = Reg::UpdateMask<T>(n2);
        uint16_t blockStride = 0x1;
        uint16_t repeatStride = dealRowCount;

        Reg::Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vregLseMax, -FLT_MAX_NEW, pregTailN);
        Reg::Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vregLseMax2, -FLT_MAX_NEW, pregTailN);
        Reg::Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vregLseSum, FLT_ZERO, pregTailN);
        Reg::Duplicate<T, Reg::MaskMergeMode::ZEROING, float>(vregLseSum2, FLT_ZERO, pregTailN);

        for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregLseMaxTmp, lseMaxSrc + i * dealCountSum);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregLseMaxTmp2, lseMaxSrc + i * dealCountSum + dealCount);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(vregLseMax, vregLseMax, vregLseMaxTmp, pregTailN);
            Reg::Max<T, Reg::MaskMergeMode::ZEROING>(vregLseMax2, vregLseMax2, vregLseMaxTmp2, pregTailN2);
        }

        for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregLseMaxTmp, lseMaxSrc + i * dealCountSum);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregLseMaxTmp2, lseMaxSrc + i * dealCountSum + dealCount);
            Reg::Sub<T, Reg::MaskMergeMode::ZEROING>(vregLseMaxTmp, vregLseMaxTmp, vregLseMax, pregTailN);
            Reg::Sub<T, Reg::MaskMergeMode::ZEROING>(vregLseMaxTmp2, vregLseMaxTmp2, vregLseMax2, pregTailN2);
            Reg::Exp<T, Reg::MaskMergeMode::ZEROING>(vregLseMaxTmp, vregLseMaxTmp, pregTailN);
            Reg::Exp<T, Reg::MaskMergeMode::ZEROING>(vregLseMaxTmp2, vregLseMaxTmp2, pregTailN2);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregLseSumTmp, lseSumSrc + i * dealCountSum);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregLseSumTmp2, lseSumSrc + i * dealCountSum + dealCount);
            Reg::Mul<T, Reg::MaskMergeMode::ZEROING>(vregLseSumTmp, vregLseSumTmp, vregLseMaxTmp, pregTailN);
            Reg::Mul<T, Reg::MaskMergeMode::ZEROING>(vregLseSumTmp2, vregLseSumTmp2, vregLseMaxTmp2, pregTailN2);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregLseSum, vregLseSum, vregLseSumTmp, pregTailN);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregLseSum2, vregLseSum2, vregLseSumTmp2, pregTailN2);
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM>(lseSumSrc + i * dealCountSum, vregLseSumTmp, pregTailN);
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM>(lseSumSrc + i * dealCountSum + dealCount, vregLseSumTmp2,
                                                        pregTailN2);
        }

        if (softmaxLseFlag) {
            Reg::RegTensor<float> vregMinValue;
            Reg::RegTensor<float> vregInfValue;
            Reg::MaskReg pregCompare;
            Reg::MaskReg pregCompare2;
            constexpr float infValue = 3e+99; // 3e+99 for float inf
            constexpr uint32_t tmpMin = 0xFF167699;
            float minValue = *((float *)&tmpMin);
            Reg::Duplicate<float, float>(vregMinValue, minValue);
            Reg::Duplicate<float, float>(vregInfValue, infValue);

            Reg::Log<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregLseSum, pregTailN);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregRes, vregLseMax, pregTailN);
            Reg::Log<T, Reg::MaskMergeMode::ZEROING>(vregRes2, vregLseSum2, pregTailN2);
            Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregRes2, vregRes2, vregLseMax2, pregTailN2);
            // 如果 softmaxMax 等于负无穷，则将 lse 结果置为 inf
            Reg::Compare<float, CMPMODE::EQ>(pregCompare, vregLseMax, vregMinValue, pregTailN);
            Reg::Compare<float, CMPMODE::EQ>(pregCompare2, vregLseMax2, vregMinValue, pregTailN2);
            Reg::Select<T>(vregRes, vregInfValue, vregRes, pregCompare);
            Reg::Select<T>(vregRes2, vregInfValue, vregRes2, pregCompare2);
            Reg::DataCopy<T, StoreDist::DIST_NORM_B32>(lseUb, vregRes, pregTailN);
            Reg::DataCopy<T, StoreDist::DIST_NORM_B32>(lseUb2, vregRes2, pregTailN2);
        }

        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        for (i = 0; i < static_cast<uint16_t>(actualCombineLoopSize); ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregLseSumTmp, lseSumSrc + i * dealCountSum);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregLseSumTmp2, lseSumSrc + i * dealCountSum + dealCount);
            Reg::Div<T, Reg::MaskMergeMode::ZEROING>(vregLseSumTmp, vregLseSumTmp, vregLseSum, pregTailN);
            Reg::Div<T, Reg::MaskMergeMode::ZEROING>(vregLseSumTmp2, vregLseSumTmp2, vregLseSum2, pregTailN2);
            Reg::DataCopy<T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                lseSum, vregLseSumTmp, blockStride, repeatStride, pregTailN);
            Reg::DataCopy<T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                lseSum2, vregLseSumTmp2, blockStride, repeatStride, pregTailN2);
        }
    }
}

template <typename T>
__aicore__ inline void ComputeScaleValue_VF(const LocalTensor<T> &lseMaxUb, const LocalTensor<T> &lseSumUb,
                                            const LocalTensor<T> &lseOutputUb, uint32_t dealRowCount,
                                            uint32_t actualCombineLoopSize, bool softmaxLseFlag)
{
    if (dealRowCount <= 8) {
        ComputeScaleValue_VF_8(lseMaxUb, lseSumUb, lseOutputUb, dealRowCount, actualCombineLoopSize, softmaxLseFlag);
    } else if (dealRowCount <= 16) {
        ComputeScaleValue_VF_16(lseMaxUb, lseSumUb, lseOutputUb, dealRowCount, actualCombineLoopSize, softmaxLseFlag);
    }
}

// 处理g<=8的场景
template <typename T>
__aicore__ inline void ComputeLogSumExp_VF_8(const LocalTensor<T> &dstTensor, const LocalTensor<T> &softmaxSumTensor,
                                             const LocalTensor<T> &softmaxMaxTensor, uint32_t dealCount)
{
    uint64_t srcSumLocalInt = softmaxSumTensor.GetPhyAddr();
    uint64_t srcMaxLocalInt = softmaxMaxTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vregSum;
        Reg::RegTensor<T> vregMax;
        Reg::RegTensor<T> vregRes;
        Reg::MaskReg pregTailN = Reg::UpdateMask<T>(dealCount);
        Reg::RegTensor<float> vregMinValue;
        Reg::RegTensor<float> vregInfValue;
        Reg::MaskReg pregCompare;
        constexpr float infValue = 3e+99; // 3e+99 for float inf
        constexpr uint32_t tmpMin = 0xFF167699;
        float minValue = *((float *)&tmpMin);
        Reg::Duplicate<float, float>(vregMinValue, minValue);
        Reg::Duplicate<float, float>(vregInfValue, infValue);

        // 1.load to reg
        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregSum, (__ubuf__ float *&)srcSumLocalInt);
        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregMax, (__ubuf__ float *&)srcMaxLocalInt);

        // 2.LogSumExp
        Reg::Log<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregSum, pregTailN);
        Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregRes, vregMax, pregTailN);

        // 如果 softmaxMax 等于负无穷，则将 lse 结果置为 inf
        Reg::Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, pregTailN);
        Reg::Select<T>(vregRes, vregInfValue, vregRes, pregCompare);

        // 3.copy to ub
        Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ float *&)dstLocalInt, vregRes, pregTailN);
    }
}

// 处理8<g<=16的场景
template <typename T>
__aicore__ inline void ComputeLogSumExp_VF_16(const LocalTensor<T> &dstTensor, const LocalTensor<T> &softmaxSumTensor,
                                              const LocalTensor<T> &softmaxMaxTensor, uint32_t dealCount)
{
    __ubuf__ T *srcSumUb = (__ubuf__ T *)softmaxSumTensor.GetPhyAddr();
    __ubuf__ T *srcSumUb2 = srcSumUb + 64; // 一个寄存器最多处理64个数
    __ubuf__ T *srcMaxUb = (__ubuf__ T *)softmaxMaxTensor.GetPhyAddr();
    __ubuf__ T *srcMaxUb2 = srcMaxUb + 64;
    __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ T *dstUb2 = dstUb + 64;
    uint32_t dealCount1 = 8 * 8;
    uint32_t dealCount2 = dealCount - dealCount1;

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vregSum;
        Reg::RegTensor<T> vregSum2;
        Reg::RegTensor<T> vregMax;
        Reg::RegTensor<T> vregMax2;
        Reg::RegTensor<T> vregRes;
        Reg::RegTensor<T> vregRes2;
        Reg::MaskReg pregTailN = Reg::UpdateMask<T>(dealCount1);
        Reg::MaskReg pregTailN2 = Reg::UpdateMask<T>(dealCount2);
        Reg::RegTensor<float> vregMinValue;
        Reg::RegTensor<float> vregInfValue;
        Reg::MaskReg pregCompare;
        Reg::MaskReg pregCompare2;
        constexpr float infValue = 3e+99; // 3e+99 for float inf
        constexpr uint32_t tmpMin = 0xFF167699;
        float minValue = *((float *)&tmpMin);
        Reg::Duplicate<float, float>(vregMinValue, minValue);
        Reg::Duplicate<float, float>(vregInfValue, infValue);

        // 1.load to reg
        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregSum, srcSumUb);
        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregSum2, srcSumUb2);
        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregMax, srcMaxUb);
        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregMax2, srcMaxUb2);

        // 2.LogSumExp
        Reg::Log<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregSum, pregTailN);
        Reg::Log<T, Reg::MaskMergeMode::ZEROING>(vregRes2, vregSum2, pregTailN2);
        Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregRes, vregRes, vregMax, pregTailN);
        Reg::Add<T, Reg::MaskMergeMode::ZEROING>(vregRes2, vregRes2, vregMax2, pregTailN2);

        // 如果 softmaxMax 等于负无穷，则将 lse 结果置为 inf
        Reg::Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, pregTailN);
        Reg::Compare<float, CMPMODE::EQ>(pregCompare2, vregMax2, vregMinValue, pregTailN2);
        Reg::Select<T>(vregRes, vregInfValue, vregRes, pregCompare);
        Reg::Select<T>(vregRes2, vregInfValue, vregRes2, pregCompare2);

        // 3.copy to ub
        Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>(dstUb, vregRes, pregTailN);
        Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B32>(dstUb2, vregRes2, pregTailN2);
    }
}

template <typename T>
__aicore__ inline void ComputeLogSumExp_VF(const LocalTensor<T> &dstTensor, const LocalTensor<T> &softmaxSumTensor,
                                           const LocalTensor<T> &softmaxMaxTensor, uint32_t dealRowCount)
{
    if (dealRowCount <= 8) {
        ComputeLogSumExp_VF_8(dstTensor, softmaxSumTensor, softmaxMaxTensor, dealRowCount * 8); // 8:FP32 in one block
    } else if (dealRowCount <= 16) {
        ComputeLogSumExp_VF_16(dstTensor, softmaxSumTensor, softmaxMaxTensor, dealRowCount * 8); // 8:FP32 in one block
    }
}

} // namespace FaVectorApi

#endif // MY_FLASH_DECODE_ARCH38_H
