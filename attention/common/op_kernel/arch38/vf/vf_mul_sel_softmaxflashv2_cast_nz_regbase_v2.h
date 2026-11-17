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
 * \file vf_mul_sel_softmaxflashv2_cast_nz_regbase_v2.h
 * \brief
 */
#ifndef MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_REGBASE_V2_INTERFACE_H
#define MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_REGBASE_V2_INTERFACE_H

#include "kernel_tensor.h"
#include "../pse_arch38.h"
#include "vf_store_exp.h"

using namespace regbaseutil;

namespace FaVectorApi {
constexpr uint32_t floatRepSize = 64;
constexpr uint32_t halfRepSize = 128;
constexpr uint32_t blockBytesU8 = 32;

template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0, bool hasAtten = false>
__aicore__ inline void SoftmaxFlashV510NoUpdateImpl128(
    const LocalTensor<T2>& dstTensor, const LocalTensor<float>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<float>& expMaxTensor, const LocalTensor<T>& inSrcTensor, const LocalTensor<float>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor, const LocalTensor<T>& inPseTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m, const uint32_t originN, const T scale,
    const T minValue, const uint16_t blockStride, float quantScaleP)
{
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2* expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();
    __ubuf__ T* maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T* maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T* srcUb = (__ubuf__ T*)inSrcTensor.GetPhyAddr();
    __ubuf__ uint8_t* maskUb = nullptr;
    if constexpr (hasAtten) {
        maskUb = (__ubuf__ uint8_t*)inMaskTensor.GetPhyAddr();
    }
    uint32_t maskLen = originN;
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vregInputX;
        Reg::RegTensor<T> vregInputMax;
        Reg::RegTensor<T> vregMaxBrc;
        Reg::RegTensor<T> vregMin;
        Reg::RegTensor<float> vregExpSum;
        Reg::RegTensor<float> vregExpEven;
        Reg::RegTensor<float> vregExpOdd;

        Reg::UnalignReg uregMax;
        Reg::UnalignReg uregExpSum;

        Reg::MaskReg pregAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAllB16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAllB8 = Reg::CreateMask<int8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregS8 = Reg::CreateMask<int8_t, Reg::MaskPattern::VL128>();
        Reg::MaskReg pregMask;
        Reg::RegTensor<uint16_t> vregAttenMask;

        Reg::RegTensor<half> vregExpRes;
        Reg::RegTensor<half> vregMulsRes;
        Reg::RegTensor<int8_t> vregRes;

        Reg::Duplicate(vregMin, minValue);
        Reg::MaskReg pregOriTailN = Reg::UpdateMask<T>(maskLen);

        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInputX, srcUb + i * sInner);
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregInputX, vregInputX, scale, pregAllB16);
            Reg::Select(vregInputX, vregInputX, vregMin, pregOriTailN);
            if constexpr (hasAtten) {
                Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK_B8>((Reg::RegTensor<uint8_t>&)vregAttenMask,
                                                                      maskUb + i * sInner);
                Reg::CompareScalar<uint16_t, CMPMODE::NE>(pregMask, vregAttenMask, static_cast<uint16_t>(0),
                                                          pregAllB16);
                Reg::Select(vregInputX, vregMin, vregInputX, pregMask);
            }
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B16>(srcUb + i * sInner, vregInputX, pregOriTailN);
            Reg::ReduceMax<T, Reg::MaskMergeMode::ZEROING>(vregInputMax, vregInputX, pregAllB16);
            Reg::DataCopyUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(maxUb, vregInputMax, uregMax, 1);
        }
        Reg::DataCopyUnAlignPost<T, Reg::PostLiteral::POST_MODE_UPDATE>(maxUb, uregMax, 0);

        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_BRC_B16>(vregMaxBrc, maxUbStart + i);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInputX, srcUb + i * sInner);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregExpRes, vregInputX, vregMaxBrc,
                                                                                     pregOriTailN);

            static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            if constexpr (IsSameType<T2, int8_t>::value) {
                Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vregMulsRes, vregExpRes, (half)quantScaleP,
                                                                   pregAllB16);
                Reg::Cast<int8_t, half, castTrait>(vregRes, vregMulsRes, pregAllB16);
                Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint8_t>&)vregRes,
                                                                       (Reg::RegTensor<uint16_t>&)vregRes);
                Reg::DataCopy<int8_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ int8_t*&)expUb, vregRes, blockStride, repeatStride, pregS8);
            } else {
                StoreExp(expUb, vregExpRes, blockStride, repeatStride, pregAllB16, quantScaleP);
            }

            Reg::Cast<float, half, castTrait0>(vregExpEven, vregExpRes, pregAllB16);
            Reg::Cast<float, half, castTrait1>(vregExpOdd, vregExpRes, pregAllB16);
            Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregExpSum, vregExpEven, vregExpOdd, pregAllB32);
            Reg::ReduceSum<float, float, Reg::MaskMergeMode::ZEROING>(vregExpSum, vregExpSum, pregAllB32);
            Reg::DataCopyUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(expSumUb, vregExpSum, uregExpSum, 1);
        }
        Reg::DataCopyUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(expSumUb, uregExpSum, 0);
    }
}

template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0, bool hasAtten = false>
__aicore__ inline void SoftmaxFlashV510NoUpdateImpl256(
    const LocalTensor<T2>& dstTensor, const LocalTensor<float>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<float>& expMaxTensor, const LocalTensor<T>& inSrcTensor, const LocalTensor<float>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor, const LocalTensor<T>& inPseTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m, const uint32_t originN, const T scale,
    const T minValue, const uint16_t blockStride, float quantScaleP)
{
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2* expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2* expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + blockStride * 128;
    __ubuf__ float* expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();
    __ubuf__ T* maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T* maxUbStart = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T* srcUb = (__ubuf__ T*)inSrcTensor.GetPhyAddr();
    __ubuf__ uint8_t* maskUb = nullptr;
    if constexpr (hasAtten) {
        maskUb = (__ubuf__ uint8_t*)inMaskTensor.GetPhyAddr();
    }
    uint32_t maskLen1 = originN;
    uint32_t maskLen2 = originN > halfRepSize ? originN - halfRepSize : 0;

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vregInputX1;
        Reg::RegTensor<T> vregInputX2;
        Reg::RegTensor<T> vregInputMaxTmp;
        Reg::RegTensor<T> vregInputMax;
        Reg::RegTensor<T> vregMaxBrc;
        Reg::RegTensor<T> vregMin;
        Reg::RegTensor<float> vregExpSum;
        Reg::RegTensor<float> vregExpEven;
        Reg::RegTensor<float> vregExpOdd;

        Reg::UnalignReg uregMax;
        Reg::UnalignReg uregExpSum;

        Reg::MaskReg pregAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAllB16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAllB8 = Reg::CreateMask<int8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregS8 = Reg::CreateMask<int8_t, Reg::MaskPattern::VL128>();
        Reg::MaskReg pregMask1;
        Reg::MaskReg pregMask2;
        Reg::RegTensor<uint16_t> vregAttenMask1;
        Reg::RegTensor<uint16_t> vregAttenMask2;

        Reg::RegTensor<T> vregExpRes;
        Reg::RegTensor<T> vregExpRes1;
        Reg::RegTensor<T> vregExpRes2;
        Reg::RegTensor<T> vregMulsRes1;
        Reg::RegTensor<T> vregMulsRes2;
        Reg::RegTensor<int8_t> vregRes1;
        Reg::RegTensor<int8_t> vregRes2;
        Reg::MaskReg pregOriTailN1 = Reg::UpdateMask<T>(maskLen1);
        Reg::MaskReg pregOriTailN2 = Reg::UpdateMask<T>(maskLen2);

        Reg::Duplicate(vregMin, minValue);

        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInputX1, srcUb + i * sInner);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInputX2, srcUb + i * sInner + halfRepSize);
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregInputX1, vregInputX1, scale, pregAllB16);
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregInputX2, vregInputX2, scale, pregAllB16);
            Reg::Select(vregInputX1, vregInputX1, vregMin, pregOriTailN1);
            Reg::Select(vregInputX2, vregInputX2, vregMin, pregOriTailN2);
            if constexpr (hasAtten) {
                Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK_B8>((Reg::RegTensor<uint8_t>&)vregAttenMask1,
                                                                      maskUb + i * sInner);
                Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK_B8>((Reg::RegTensor<uint8_t>&)vregAttenMask2,
                                                                      maskUb + i * sInner + halfRepSize);

                Reg::CompareScalar<uint16_t, CMPMODE::NE>(pregMask1, vregAttenMask1, static_cast<uint16_t>(0),
                                                          pregAllB16);
                Reg::CompareScalar<uint16_t, CMPMODE::NE>(pregMask2, vregAttenMask2, static_cast<uint16_t>(0),
                                                          pregAllB16);

                Reg::Select(vregInputX1, vregMin, vregInputX1, pregMask1);
                Reg::Select(vregInputX2, vregMin, vregInputX2, pregMask2);
            }
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B16>(srcUb + i * sInner, vregInputX1, pregOriTailN1);
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B16>(srcUb + i * sInner + halfRepSize, vregInputX2,
                                                            pregOriTailN2);
            Reg::Max(vregInputMaxTmp, vregInputX1, vregInputX2, pregAllB16);
            Reg::ReduceMax<T, Reg::MaskMergeMode::ZEROING>(vregInputMax, vregInputMaxTmp, pregAllB16);
            Reg::DataCopyUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(maxUb, vregInputMax, uregMax, 1);
        }
        Reg::DataCopyUnAlignPost<T, Reg::PostLiteral::POST_MODE_UPDATE>(maxUb, uregMax, 0);

        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_BRC_B16>(vregMaxBrc, maxUbStart + i);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInputX1, srcUb + i * sInner);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInputX2, srcUb + i * sInner + halfRepSize);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregExpRes1, vregInputX1,
                                                                                     vregMaxBrc, pregOriTailN1);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregExpRes2, vregInputX2,
                                                                                     vregMaxBrc, pregOriTailN2);
            static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

            if constexpr (IsSameType<T2, int8_t>::value) {
                Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vregMulsRes1, vregExpRes1, (half)quantScaleP,
                                                                   pregAllB16);
                Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vregMulsRes2, vregExpRes2, (half)quantScaleP,
                                                                   pregAllB16);

                Reg::Cast<int8_t, half, castTrait>(vregRes1, vregMulsRes1, pregAllB16);
                Reg::Cast<int8_t, half, castTrait>(vregRes2, vregMulsRes2, pregAllB16);

                Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint8_t>&)vregRes1,
                                                                       (Reg::RegTensor<uint16_t>&)vregRes1);
                Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint8_t>&)vregRes2,
                                                                       (Reg::RegTensor<uint16_t>&)vregRes2);

                Reg::DataCopy<int8_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ int8_t*&)expUb1, vregRes1, blockStride, repeatStride, pregS8);
                Reg::DataCopy<int8_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ int8_t*&)expUb2, vregRes2, blockStride, repeatStride, pregS8);
            } else {
                StoreExp(expUb1, vregExpRes1, blockStride, repeatStride, pregAllB16, quantScaleP);
                StoreExp(expUb2, vregExpRes2, blockStride, repeatStride, pregAllB16, quantScaleP);
            }

            Reg::Add<half, Reg::MaskMergeMode::ZEROING>(vregExpRes, vregExpRes1, vregExpRes2, pregAllB16);
            Reg::Cast<float, half, castTrait0>(vregExpEven, vregExpRes, pregAllB16);
            Reg::Cast<float, half, castTrait1>(vregExpOdd, vregExpRes, pregAllB16);
            Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregExpSum, vregExpEven, vregExpOdd, pregAllB32);
            Reg::ReduceSum<float, float, Reg::MaskMergeMode::ZEROING>(vregExpSum, vregExpSum, pregAllB32);
            Reg::DataCopyUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(expSumUb, vregExpSum, uregExpSum, 1);
        }
        Reg::DataCopyUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(expSumUb, uregExpSum, 0);
    }
}

template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0, bool hasAtten = false>
__aicore__ inline void SoftmaxFlashV510NoUpdate8(
    const LocalTensor<T2>& dstTensor, const LocalTensor<float>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<float>& expMaxTensor, const LocalTensor<T>& inSrcTensor, const LocalTensor<float>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor, const LocalTensor<T>& inPseTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m, const uint32_t originN, const T scale,
    const T minValue, const uint16_t blockStride, float quantScaleP)
{
    // mode 1: originN = 128
    if constexpr (mode == 1) {
        SoftmaxFlashV510NoUpdateImpl128<T, T2, mode, sOuter, sInner, hasAtten>(
            dstTensor, expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor,
            inPseTensor, sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    } else {
        SoftmaxFlashV510NoUpdateImpl256<T, T2, mode, sOuter, sInner, hasAtten>(
            dstTensor, expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor,
            inPseTensor, sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    }
}

// originN = 128, Update
template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0, bool hasAtten = false>
__aicore__ inline void SoftmaxFlashV510UpdateImpl128(
    const LocalTensor<T2>& dstTensor, const LocalTensor<float>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<float>& expMaxTensor, const LocalTensor<T>& inSrcTensor, const LocalTensor<float>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor, const LocalTensor<T>& inPseTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m, const uint32_t originN, const T scale,
    const T minValue, const uint16_t blockStride, float quantScaleP)
{
    constexpr uint32_t reduceN = 1;
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2* expUb = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ float* expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();
    __ubuf__ T* maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T* srcUb = (__ubuf__ T*)inSrcTensor.GetPhyAddr();
    __ubuf__ float* expMaxUb = (__ubuf__ float*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* inExpSumUb = (__ubuf__ float*)inExpSumTensor.GetPhyAddr();
    __ubuf__ T* inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ float* tmpExpSumUb = (__ubuf__ float*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ float* tmpExpSumUbStart = (__ubuf__ float*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T* tmpMaxUb = (__ubuf__ T*)((__ubuf__ float*)sharedTmpBuffer.GetPhyAddr() + 64);
    __ubuf__ T* tmpMaxUbStart = (__ubuf__ T*)((__ubuf__ float*)sharedTmpBuffer.GetPhyAddr() + 64);
    __ubuf__ uint8_t* maskUb = nullptr;
    if constexpr (hasAtten) {
        maskUb = (__ubuf__ uint8_t*)inMaskTensor.GetPhyAddr();
    }
    uint32_t maskLen = originN;

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vregInputX;
        Reg::RegTensor<T> vregInputMax;
        Reg::RegTensor<T> vregMin;
        Reg::RegTensor<float> vregExpSum;
        Reg::RegTensor<float> vregExpSumBrcEven;
        Reg::RegTensor<float> vregExpSumBrcOdd;
        Reg::RegTensor<T> vregInMax;
        Reg::RegTensor<T> vregMax;
        Reg::RegTensor<T> vregExpMax;
        Reg::RegTensor<float> vregExpMaxEven;
        Reg::RegTensor<float> vregExpMaxOdd;
        Reg::RegTensor<float> vregInExpSumEven;
        Reg::RegTensor<float> vregInExpSumOdd;
        Reg::RegTensor<float> vregExpSumUpdateEven;
        Reg::RegTensor<float> vregExpSumUpdateOdd;
        Reg::RegTensor<T> vregExpRes;
        Reg::RegTensor<float> vregExpEven;
        Reg::RegTensor<float> vregExpOdd;

        Reg::UnalignReg uregMax;
        Reg::UnalignReg uregExpSum;

        Reg::MaskReg pregAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAllB16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAllB8 = Reg::CreateMask<int8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregS8 = Reg::CreateMask<int8_t, Reg::MaskPattern::VL128>();
        Reg::MaskReg pregMask;
        Reg::RegTensor<uint16_t> vregAttenMask;

        Reg::RegTensor<half> vregMulsRes;
        Reg::RegTensor<int8_t> vregRes;

        static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                     Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        Reg::MaskReg pregOriTailN = Reg::UpdateMask<T>(maskLen);
        Reg::Duplicate(vregMin, minValue);
        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInputX, srcUb + i * sInner);
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregInputX, vregInputX, scale, pregAllB16);
            Reg::Select(vregInputX, vregInputX, vregMin, pregOriTailN);
            if constexpr (hasAtten) {
                Reg::LoadAlign<uint8_t, Reg::LoadDist::DIST_UNPACK_B8>((Reg::RegTensor<uint8_t>&)vregAttenMask,
                                                                       maskUb + i * sInner);
                Reg::CompareScalar<uint16_t, CMPMODE::NE>(pregMask, vregAttenMask, static_cast<uint16_t>(0),
                                                          pregAllB16);
                Reg::Select(vregInputX, vregMin, vregInputX, pregMask);
            }
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B16>(srcUb + i * sInner, vregInputX, pregOriTailN);
            Reg::ReduceMax<T, Reg::MaskMergeMode::ZEROING>(vregInputMax, vregInputX, pregAllB16);
            Reg::DataCopyUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, vregInputMax, uregMax, 1);
        }
        Reg::DataCopyUnAlignPost<T, Reg::PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, uregMax, 0);
        mem_bar(VST_VLD);

        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInMax, inMaxUb);
        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInputMax, tmpMaxUbStart);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(vregMax, vregInputMax, vregInMax, pregAllB16);
        Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregExpMax, vregInMax, vregMax,
                                                                                 pregAllB16);
        Reg::Cast<float, half, castTrait0>(vregExpMaxEven, vregExpMax, pregAllB16);
        Reg::Cast<float, half, castTrait1>(vregExpMaxOdd, vregExpMax, pregAllB16);

        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(expMaxUb, vregExpMaxEven, vregExpMaxOdd, pregAllB32);
        Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B16>(maxUb, vregMax, pregAllB16);

        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_BRC_B16>(vregMax, maxUb + i);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInputX, srcUb + i * sInner);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregExpRes, vregInputX, vregMax,
                                                                                     pregOriTailN);

            static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
            static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::SAT,
                                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

            if constexpr (IsSameType<T2, int8_t>::value) {
                Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vregMulsRes, vregExpRes, (half)quantScaleP,
                                                                   pregAllB16);

                Reg::Cast<int8_t, half, castTrait0>(vregRes, vregMulsRes, pregAllB16);
                Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint8_t>&)vregRes,
                                                                       (Reg::RegTensor<uint16_t>&)vregRes);

                Reg::DataCopy<int8_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ int8_t*&)expUb), vregRes, blockStride, repeatStride, pregS8);
            } else {
                StoreExp(expUb, vregExpRes, blockStride, repeatStride, pregAllB16, quantScaleP);
            }

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            Reg::Cast<float, half, castTrait0>(vregExpEven, vregExpRes, pregAllB16);
            Reg::Cast<float, half, castTrait1>(vregExpOdd, vregExpRes, pregAllB16);
            Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregExpSum, vregExpEven, vregExpOdd, pregAllB32);
            Reg::ReduceSum<float, float, Reg::MaskMergeMode::ZEROING>(vregExpSum, vregExpSum, pregAllB32);
            Reg::DataCopyUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, vregExpSum, uregExpSum, 1);
        }
        Reg::DataCopyUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, uregExpSum, 0);
        mem_bar(VST_VLD);

        // x_sum = sum(exp_max * in_sum + x_sum)
        Reg::DataCopy<float, Reg::LoadDist::DIST_DINTLV_B32>(vregInExpSumEven, vregInExpSumOdd, inExpSumUb);
        Reg::DataCopy<float, Reg::LoadDist::DIST_DINTLV_B32>(vregExpSumBrcEven, vregExpSumBrcOdd, tmpExpSumUbStart);
        Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vregExpSumUpdateEven, vregExpMaxEven, vregInExpSumEven,
                                                     pregAllB32);
        Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vregExpSumUpdateOdd, vregExpMaxOdd, vregInExpSumOdd, pregAllB32);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregExpSumUpdateEven, vregExpSumUpdateEven, vregExpSumBrcEven,
                                                     pregAllB32);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregExpSumUpdateOdd, vregExpSumUpdateOdd, vregExpSumBrcOdd,
                                                     pregAllB32);
        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(expSumUb, vregExpSumUpdateEven, vregExpSumUpdateOdd,
                                                             pregAllB32);
    }
}

template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0, bool hasAtten = false>
__aicore__ inline void SoftmaxFlashV510UpdateImpl256(
    const LocalTensor<T2>& dstTensor, const LocalTensor<float>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<float>& expMaxTensor, const LocalTensor<T>& inSrcTensor, const LocalTensor<float>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor, const LocalTensor<T>& inPseTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m, const uint32_t originN, const T scale,
    const T minValue, const uint16_t blockStride, float quantScaleP)
{
    constexpr uint32_t reduceN = 1;
    const uint16_t rows = static_cast<uint16_t>(m);
    constexpr uint16_t repeatStride = 1;

    __ubuf__ T2* expUb1 = (__ubuf__ T2*)dstTensor.GetPhyAddr();
    __ubuf__ T2* expUb2 = (__ubuf__ T2*)dstTensor.GetPhyAddr() + blockStride * 128;
    __ubuf__ float* expSumUb = (__ubuf__ float*)expSumTensor.GetPhyAddr();
    __ubuf__ T* maxUb = (__ubuf__ T*)maxTensor.GetPhyAddr();
    __ubuf__ T* srcUb = (__ubuf__ T*)inSrcTensor.GetPhyAddr();
    __ubuf__ float* expMaxUb = (__ubuf__ float*)expMaxTensor.GetPhyAddr();
    __ubuf__ float* inExpSumUb = (__ubuf__ float*)inExpSumTensor.GetPhyAddr();
    __ubuf__ T* inMaxUb = (__ubuf__ T*)inMaxTensor.GetPhyAddr();
    __ubuf__ float* tmpExpSumUb = (__ubuf__ float*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ float* tmpExpSumUbStart = (__ubuf__ float*)sharedTmpBuffer.GetPhyAddr();
    __ubuf__ T* tmpMaxUb = (__ubuf__ T*)((__ubuf__ float*)sharedTmpBuffer.GetPhyAddr() + 64);
    __ubuf__ T* tmpMaxUbStart = (__ubuf__ T*)((__ubuf__ float*)sharedTmpBuffer.GetPhyAddr() + 64);
    __ubuf__ uint8_t* maskUb = nullptr;
    if constexpr (hasAtten) {
        maskUb = (__ubuf__ uint8_t*)inMaskTensor.GetPhyAddr();
    }
    uint32_t maskLen1 = originN;
    uint32_t maskLen2 = originN > halfRepSize ? originN - halfRepSize : 0;

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vregInputX1;
        Reg::RegTensor<T> vregInputX2;
        Reg::RegTensor<T> vregInputMax;
        Reg::RegTensor<T> vregInputMaxTmp;
        Reg::RegTensor<T> vregMin;
        Reg::RegTensor<float> vregExpSum;
        Reg::RegTensor<float> vregExpSumBrcEven;
        Reg::RegTensor<float> vregExpSumBrcOdd;
        Reg::RegTensor<T> vregInMax;
        Reg::RegTensor<T> vregMax;
        Reg::RegTensor<T> vregExpMax;
        Reg::RegTensor<float> vregExpMaxEven;
        Reg::RegTensor<float> vregExpMaxOdd;
        Reg::RegTensor<float> vregInExpSumEven;
        Reg::RegTensor<float> vregInExpSumOdd;
        Reg::RegTensor<float> vregExpSumUpdateEven;
        Reg::RegTensor<float> vregExpSumUpdateOdd;
        Reg::RegTensor<T> vregExpRes;
        Reg::RegTensor<T> vregExpRes1;
        Reg::RegTensor<T> vregExpRes2;
        Reg::RegTensor<float> vregExpEven;
        Reg::RegTensor<float> vregExpOdd;

        Reg::UnalignReg uregMax;
        Reg::UnalignReg uregExpSum;

        Reg::MaskReg pregAllB32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAllB16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregAllB8 = Reg::CreateMask<int8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg pregS8 = Reg::CreateMask<int8_t, Reg::MaskPattern::VL128>();
        Reg::MaskReg pregMask1;
        Reg::MaskReg pregMask2;
        Reg::RegTensor<uint16_t> vregAttenMask1;
        Reg::RegTensor<uint16_t> vregAttenMask2;

        Reg::RegTensor<T> vregMulsRes1;
        Reg::RegTensor<T> vregMulsRes2;
        Reg::RegTensor<int8_t> vregRes1;
        Reg::RegTensor<int8_t> vregRes2;

        static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                     Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        Reg::MaskReg pregOriTailN1 = Reg::UpdateMask<T>(maskLen1);
        Reg::MaskReg pregOriTailN2 = Reg::UpdateMask<T>(maskLen2);

        Reg::Duplicate(vregMin, minValue);

        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInputX1, srcUb + i * sInner);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInputX2, srcUb + i * sInner + halfRepSize);
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregInputX1, vregInputX1, scale, pregAllB16);
            Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregInputX2, vregInputX2, scale, pregAllB16);
            Reg::Select(vregInputX1, vregInputX1, vregMin, pregOriTailN1);
            Reg::Select(vregInputX2, vregInputX2, vregMin, pregOriTailN2);
            if constexpr (hasAtten) {
                Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK_B8>((Reg::RegTensor<uint8_t>&)vregAttenMask1,
                                                                      maskUb + i * sInner);
                Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK_B8>((Reg::RegTensor<uint8_t>&)vregAttenMask2,
                                                                      maskUb + i * sInner + halfRepSize);

                Reg::CompareScalar<uint16_t, CMPMODE::NE>(pregMask1, vregAttenMask1, static_cast<uint16_t>(0),
                                                          pregAllB16);
                Reg::CompareScalar<uint16_t, CMPMODE::NE>(pregMask2, vregAttenMask2, static_cast<uint16_t>(0),
                                                          pregAllB16);

                Reg::Select(vregInputX1, vregMin, vregInputX1, pregMask1);
                Reg::Select(vregInputX2, vregMin, vregInputX2, pregMask2);
            }
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B16>(srcUb + i * sInner, vregInputX1, pregOriTailN1);
            Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B16>(srcUb + i * sInner + halfRepSize, vregInputX2,
                                                            pregOriTailN2);
            Reg::Max(vregInputMaxTmp, vregInputX1, vregInputX2, pregAllB16);
            Reg::ReduceMax<T, Reg::MaskMergeMode::ZEROING>(vregInputMax, vregInputMaxTmp, pregAllB16);
            Reg::DataCopyUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, vregInputMax, uregMax, 1);
        }
        Reg::DataCopyUnAlignPost<T, Reg::PostLiteral::POST_MODE_UPDATE>(tmpMaxUb, uregMax, 0);
        mem_bar(VST_VLD);

        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInMax, inMaxUb);
        Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInputMax, tmpMaxUbStart);
        Reg::Max<T, Reg::MaskMergeMode::ZEROING>(vregMax, vregInputMax, vregInMax, pregAllB16);

        Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregExpMax, vregInMax, vregMax,
                                                                                 pregAllB16);
        Reg::Cast<float, half, castTrait0>(vregExpMaxEven, vregExpMax, pregAllB16);
        Reg::Cast<float, half, castTrait1>(vregExpMaxOdd, vregExpMax, pregAllB16);

        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(expMaxUb, vregExpMaxEven, vregExpMaxOdd, pregAllB32);
        Reg::DataCopy<T, Reg::StoreDist::DIST_NORM_B16>(maxUb, vregMax, pregAllB16);

        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < rows; ++i) {
            Reg::DataCopy<T, Reg::LoadDist::DIST_BRC_B16>(vregMax, maxUb + i);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInputX1, srcUb + i * sInner);
            Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vregInputX2, srcUb + i * sInner + halfRepSize);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregExpRes1, vregInputX1, vregMax,
                                                                                     pregOriTailN1);
            Reg::FusedExpSub<T, T, Reg::RegLayout::ONE, Reg::MaskMergeMode::ZEROING>(vregExpRes2, vregInputX2, vregMax,
                                                                                     pregOriTailN2);

            if constexpr (IsSameType<T2, int8_t>::value) {
                Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vregMulsRes1, vregExpRes1, (half)quantScaleP,
                                                                   pregAllB16);
                Reg::Muls<half, half, Reg::MaskMergeMode::ZEROING>(vregMulsRes2, vregExpRes2, (half)quantScaleP,
                                                                   pregAllB16);

                Reg::Cast<int8_t, half, castTrait>(vregRes1, vregMulsRes1, pregAllB16);
                Reg::Cast<int8_t, half, castTrait>(vregRes2, vregMulsRes2, pregAllB16);

                Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint8_t>&)vregRes1,
                                                                       (Reg::RegTensor<uint16_t>&)vregRes1);
                Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint8_t>&)vregRes2,
                                                                       (Reg::RegTensor<uint16_t>&)vregRes2);

                Reg::DataCopy<int8_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ int8_t*&)expUb1, vregRes1, blockStride, repeatStride, pregS8);
                Reg::DataCopy<int8_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ int8_t*&)expUb2, vregRes2, blockStride, repeatStride, pregS8);
            } else {
                StoreExp(expUb1, vregExpRes1, blockStride, repeatStride, pregAllB16, quantScaleP);
                StoreExp(expUb2, vregExpRes2, blockStride, repeatStride, pregAllB16, quantScaleP);
            }

            Reg::Add<half, Reg::MaskMergeMode::ZEROING>(vregExpRes, vregExpRes1, vregExpRes2, pregAllB16);
            Reg::Cast<float, half, castTrait0>(vregExpEven, vregExpRes, pregAllB16);
            Reg::Cast<float, half, castTrait1>(vregExpOdd, vregExpRes, pregAllB16);
            Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregExpSum, vregExpEven, vregExpOdd, pregAllB32);
            Reg::ReduceSum<float, float, Reg::MaskMergeMode::ZEROING>(vregExpSum, vregExpSum, pregAllB32);
            Reg::DataCopyUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, vregExpSum, uregExpSum, 1);
        }
        Reg::DataCopyUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(tmpExpSumUb, uregExpSum, 0);
        mem_bar(VST_VLD);

        Reg::DataCopy<float, Reg::LoadDist::DIST_DINTLV_B32>(vregInExpSumEven, vregInExpSumOdd, inExpSumUb);
        Reg::DataCopy<float, Reg::LoadDist::DIST_DINTLV_B32>(vregExpSumBrcEven, vregExpSumBrcOdd, tmpExpSumUbStart);
        Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vregExpSumUpdateEven, vregExpMaxEven, vregInExpSumEven,
                                                     pregAllB32);
        Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(vregExpSumUpdateOdd, vregExpMaxOdd, vregInExpSumOdd, pregAllB32);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregExpSumUpdateEven, vregExpSumUpdateEven, vregExpSumBrcEven,
                                                     pregAllB32);
        Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vregExpSumUpdateOdd, vregExpSumUpdateOdd, vregExpSumBrcOdd,
                                                     pregAllB32);
        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(expSumUb, vregExpSumUpdateEven, vregExpSumUpdateOdd,
                                                             pregAllB32);
    }
}

template <typename T, typename T2, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0, bool hasAtten = false>
__aicore__ inline void SoftmaxFlashV510Update8(
    const LocalTensor<T2>& dstTensor, const LocalTensor<float>& expSumTensor, const LocalTensor<T>& maxTensor,
    const LocalTensor<float>& expMaxTensor, const LocalTensor<T>& inSrcTensor, const LocalTensor<float>& inExpSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor, const LocalTensor<T>& inPseTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m, const uint32_t originN, const T scale,
    const T minValue, const uint16_t blockStride, float quantScaleP)
{
    // mode 1: originN = 128
    if constexpr (mode == 1) {
        SoftmaxFlashV510UpdateImpl128<T, T2, mode, sOuter, sInner, hasAtten>(
            dstTensor, expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor,
            inPseTensor, sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    } else {
        SoftmaxFlashV510UpdateImpl256<T, T2, mode, sOuter, sInner, hasAtten>(
            dstTensor, expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor,
            inPseTensor, sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    }
}

/*
 * @ingroup SoftmaxFlashV510
 * @brief compute max = reducemax, exp(x-max)/sum(exp(x-max))
 * @param [out] dstTensor, output LocalTensor
 * @param [out] expSumTensor, out sum(exp(x-max)) of last axis
 * @param [out] maxTensor, out max value of last axis
 * @param [out] expMaxTensor, output expmax LocalTensor
 * @param [in] inSrcTensor, input LocalTensor
 * @param [in] inExpSumTensor, in sum(exp(x-max)) of last softmax result
 * @param [in] inMaxTensor, in max value of last softmax result
 * @param [in] inMaskTensor, atten mask LocalTensor, each line padding to 32, padding value is 1
 * @param [in] inPseTensor, reserved
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] m, input rows
 * @param [in] originN, input origin columns, support range for sInner is: 0 < sInner <= 128
 * @param [in] scale, scale value
 * @param [in] minValue, minimum value
 * @param [in] quantScaleP, quant scale for softmax output
 * @param [in] isUpdate, enable flash mode
 * @param [in] mode
 *  mode 0: 64 < originN <= 128, and originN is not 8 aligned
 *  mode 1: 64 < originN <= 128, and originN is 8 aligned
 *  mode 2: 0 < originN <= 64
 * @param [in] hasAtten, indicates whether there is atten_mask
 */
template <typename T, typename T2, bool isUpdate = false, uint8_t mode = 0, uint32_t sOuter = 0, uint32_t sInner = 0,
          bool hasAtten = false>
__aicore__ inline void SoftmaxFlashV510Vf(const LocalTensor<T2>& dstTensor, const LocalTensor<float>& expSumTensor,
                                          const LocalTensor<T>& maxTensor, const LocalTensor<float>& expMaxTensor,
                                          const LocalTensor<T>& inSrcTensor, const LocalTensor<float>& inExpSumTensor,
                                          const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& inMaskTensor,
                                          const LocalTensor<T>& inPseTensor,
                                          const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t m,
                                          const uint32_t originN, const T scale, const T minValue, float quantScaleP)
{
    // NZ fractal stride, in units of 16-element blocks (the S1-direction row stride); 1 cube maps to
    // 1 vector, vec1 computes the whole S1 base block, and the |1 avoids bank conflicts.
    uint16_t blockStride = sOuter | 0x1;

    if constexpr (!isUpdate) {
        SoftmaxFlashV510NoUpdate8<T, T2, mode, sOuter, sInner, hasAtten>(
            dstTensor, expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor,
            inPseTensor, sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    } else {
        SoftmaxFlashV510Update8<T, T2, mode, sOuter, sInner, hasAtten>(
            dstTensor, expSumTensor, maxTensor, expMaxTensor, inSrcTensor, inExpSumTensor, inMaxTensor, inMaskTensor,
            inPseTensor, sharedTmpBuffer, m, originN, scale, minValue, blockStride, quantScaleP);
    }
}
} // namespace FaVectorApi

#endif // MY_MUL_SEL_SOFTMAX_FLASH_V2_CAST_NZ_REGBASE_V2_INTERFACE_H
