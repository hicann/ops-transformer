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
 * \file mhc_pre_m_k_split.h
 * \brief Two-stage M-K split implementation for MHC Pre
 */

#ifndef MHC_PRE_M_K_SPLIT_H
#define MHC_PRE_M_K_SPLIT_H

#include "mhc_pre_cube_compute.h"
#include "mhc_pre_vector_compute.h"

namespace MhcPre {

// Vector register specialization limits.
constexpr uint16_t MHC_PRE_MK_RMS_REG_ROW_LIMIT = 16U;

// Ordered K-partial policy used to match the reference accumulation order.
constexpr uint32_t MHC_PRE_MK_SEQUENTIAL_PARTIAL_K = 1024U;
constexpr uint32_t MHC_PRE_MK_SEQUENTIAL_PARTIAL_THRESHOLD = 2048U;

// Flat vector reduction thresholds for large workloads.
constexpr uint32_t MHC_PRE_MK_FLAT_REDUCE_MIN_M = 2048U;
constexpr uint32_t MHC_PRE_MK_FLAT_REDUCE_MIN_GROUP_K = 20U;
__aicore__ inline void MhcPreMkVFTransND2NZ(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                            uint16_t rowCount, uint16_t columnCount)
{
    __local_mem__ float *dst = (__local_mem__ float *)dstLocal.GetPhyAddr();
    __local_mem__ float *src = (__local_mem__ float *)srcLocal.GetPhyAddr();
    uint16_t rowAlign = BasicApiAlign(rowCount, MHC_PRE_BASIC_API_C0_SIZE);
    uint16_t dataBlockElements = MHC_PRE_BASIC_API_BLOCK_SIZE / sizeof(float);
    uint16_t columnAlign = BasicApiRoundUp<float>(columnCount);
    uint16_t mainRowBlocks = rowCount / MHC_PRE_BASIC_API_C0_SIZE;
    uint16_t tailRows = rowCount % MHC_PRE_BASIC_API_C0_SIZE;
    uint16_t columnLoops = columnAlign / dataBlockElements;
    uint32_t srcBlockStride = columnAlign / dataBlockElements;

    if (tailRows == 0) {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> data;
            AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float>();
            for (uint16_t rowBlock = 0; rowBlock < mainRowBlocks; ++rowBlock) {
                for (uint16_t columnBlock = 0; columnBlock < columnLoops; ++columnBlock) {
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        data,
                        src + rowBlock * MHC_PRE_BASIC_API_C0_SIZE * columnAlign + columnBlock * dataBlockElements,
                        srcBlockStride, mask);
                    AscendC::MicroAPI::DataCopy(dst + rowBlock * MHC_PRE_BASIC_API_C0_SIZE * dataBlockElements +
                                                    columnBlock * rowAlign * dataBlockElements,
                                                data, mask);
                }
            }
        }
        return;
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> data;
        AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::CreateMask<float>();
        uint32_t tailElements = tailRows * MHC_PRE_BASIC_API_C0_SIZE;
        AscendC::MicroAPI::MaskReg tailMask = AscendC::MicroAPI::UpdateMask<float>(tailElements);
        for (uint16_t rowBlock = 0; rowBlock < mainRowBlocks; ++rowBlock) {
            for (uint16_t columnBlock = 0; columnBlock < columnLoops; ++columnBlock) {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    data, src + rowBlock * MHC_PRE_BASIC_API_C0_SIZE * columnAlign + columnBlock * dataBlockElements,
                    srcBlockStride, fullMask);
                AscendC::MicroAPI::DataCopy(dst + rowBlock * MHC_PRE_BASIC_API_C0_SIZE * dataBlockElements +
                                                columnBlock * rowAlign * dataBlockElements,
                                            data, fullMask);
            }
        }
        src += mainRowBlocks * MHC_PRE_BASIC_API_C0_SIZE * columnAlign;
        dst += mainRowBlocks * MHC_PRE_BASIC_API_C0_SIZE * dataBlockElements;
        for (uint16_t columnBlock = 0; columnBlock < columnLoops; ++columnBlock) {
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                data, src + columnBlock * dataBlockElements, srcBlockStride, tailMask);
            AscendC::MicroAPI::DataCopy(dst + columnBlock * rowAlign * dataBlockElements, data, tailMask);
        }
    }
}

template <typename T, bool INIT_RMS>
__aicore__ inline void MhcPreMkVFProcessXInGammaLoops(const LocalTensor<float> &xOutLocal,
                                                      const LocalTensor<float> &rmsLocal,
                                                      const LocalTensor<T> &xInLocal,
                                                      const LocalTensor<float> &gammaLocal, uint16_t rowCount,
                                                      uint16_t columnCount, uint16_t loopStart, uint16_t processLoops)
{
    __local_mem__ float *xOut = (__local_mem__ float *)xOutLocal.GetPhyAddr();
    __local_mem__ float *rms = (__local_mem__ float *)rmsLocal.GetPhyAddr();
    __local_mem__ T *xIn = (__local_mem__ T *)xInLocal.GetPhyAddr();
    __local_mem__ float *gamma = (__local_mem__ float *)gammaLocal.GetPhyAddr();
    uint32_t srcStride = BasicApiRoundUp<T>(columnCount);
    uint32_t dstStride = BasicApiRoundUp<float>(columnCount);
    uint16_t loopEnd = loopStart + processLoops;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> xInReg;
        AscendC::MicroAPI::RegTensor<float> gammaReg;
        AscendC::MicroAPI::RegTensor<float> xReg;
        AscendC::MicroAPI::RegTensor<float> xGammaReg;
        AscendC::MicroAPI::RegTensor<float> squareReg;
        AscendC::MicroAPI::RegTensor<float> partialReg;
        AscendC::MicroAPI::RegTensor<float> sumReg;
        uint32_t remaining = columnCount - static_cast<uint32_t>(loopStart) * MHC_PRE_BASIC_API_VL_FP32;
        for (uint16_t loop = loopStart; loop < loopEnd; ++loop) {
            AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(remaining);
            uint32_t vectorOffset = static_cast<uint32_t>(loop) * MHC_PRE_BASIC_API_VL_FP32;
            AscendC::MicroAPI::LoadAlign(gammaReg, gamma + vectorOffset);
            for (uint16_t row = 0; row < rowCount; ++row) {
                uint32_t srcOffset = static_cast<uint32_t>(row) * srcStride + vectorOffset;
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xIn + srcOffset);
                AscendC::MicroAPI::Cast<float, T, MHC_PRE_BASIC_API_CAST_B16_TO_B32>(xReg, xInReg, mask);
                AscendC::MicroAPI::Mul(xGammaReg, gammaReg, xReg, mask);
                uint32_t dstOffset = static_cast<uint32_t>(row) * dstStride + vectorOffset;
                AscendC::MicroAPI::StoreAlign(xOut + dstOffset, xGammaReg, mask);
                AscendC::MicroAPI::Mul(squareReg, xReg, xReg, mask);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(partialReg, squareReg, mask);
                if constexpr (INIT_RMS) {
                    AscendC::MicroAPI::Duplicate(sumReg, 0.0f);
                } else {
                    AscendC::MicroAPI::Load(sumReg, rms + row);
                }
                AscendC::MicroAPI::Add(sumReg, sumReg, partialReg, mask);
                AscendC::MicroAPI::Store(rms + row, sumReg, 1U);
            }
        }
    }
}
template <typename T, bool IS_FIRST_K>
__aicore__ inline void MhcPreMkVFProcessXInGammaSmallRows(const LocalTensor<float> &xOutLocal,
                                                          const LocalTensor<float> &rmsLocal,
                                                          const LocalTensor<T> &xInLocal,
                                                          const LocalTensor<float> &gammaLocal, uint16_t rowCount,
                                                          uint16_t columnCount)
{
    __local_mem__ float *xOut = (__local_mem__ float *)xOutLocal.GetPhyAddr();
    __local_mem__ float *rms = (__local_mem__ float *)rmsLocal.GetPhyAddr();
    __local_mem__ T *xIn = (__local_mem__ T *)xInLocal.GetPhyAddr();
    __local_mem__ float *gamma = (__local_mem__ float *)gammaLocal.GetPhyAddr();
    uint32_t srcStride = BasicApiRoundUp<T>(columnCount);
    uint32_t dstStride = BasicApiRoundUp<float>(columnCount);
    uint16_t loopCount = BasicApiCeilDiv(columnCount, MHC_PRE_BASIC_API_VL_FP32);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> xInReg;
        AscendC::MicroAPI::RegTensor<float> gammaReg;
        AscendC::MicroAPI::RegTensor<float> xReg;
        AscendC::MicroAPI::RegTensor<float> xGammaReg;
        AscendC::MicroAPI::RegTensor<float> squareReg;
        AscendC::MicroAPI::RegTensor<float> partialReg;
        AscendC::MicroAPI::RegTensor<float> sumReg;
        for (uint16_t row = 0; row < rowCount; ++row) {
            if constexpr (IS_FIRST_K) {
                AscendC::MicroAPI::Duplicate(sumReg, 0.0f);
            } else {
                AscendC::MicroAPI::Load(sumReg, rms + row);
            }
            uint32_t remaining = columnCount;
            for (uint16_t loop = 0; loop < loopCount; ++loop) {
                AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(remaining);
                uint32_t vectorOffset = static_cast<uint32_t>(loop) * MHC_PRE_BASIC_API_VL_FP32;
                AscendC::MicroAPI::LoadAlign(gammaReg, gamma + vectorOffset);
                uint32_t srcOffset = static_cast<uint32_t>(row) * srcStride + vectorOffset;
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xIn + srcOffset);
                AscendC::MicroAPI::Cast<float, T, MHC_PRE_BASIC_API_CAST_B16_TO_B32>(xReg, xInReg, mask);
                AscendC::MicroAPI::Mul(xGammaReg, gammaReg, xReg, mask);
                uint32_t dstOffset = static_cast<uint32_t>(row) * dstStride + vectorOffset;
                AscendC::MicroAPI::StoreAlign(xOut + dstOffset, xGammaReg, mask);
                AscendC::MicroAPI::Mul(squareReg, xReg, xReg, mask);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(partialReg, squareReg, mask);
                AscendC::MicroAPI::Add(sumReg, sumReg, partialReg, mask);
            }
            AscendC::MicroAPI::Store(rms + row, sumReg, 1U);
        }
    }
}

template <typename T, bool IS_FIRST_K>
__aicore__ inline void MhcPreMkVFProcessXInGammaFull128(const LocalTensor<float> &xOutLocal,
                                                        const LocalTensor<float> &rmsLocal,
                                                        const LocalTensor<T> &xInLocal,
                                                        const LocalTensor<float> &gammaLocal, uint16_t rowCount)
{
    __local_mem__ float *xOut = (__local_mem__ float *)xOutLocal.GetPhyAddr();
    __local_mem__ float *rms = (__local_mem__ float *)rmsLocal.GetPhyAddr();
    __local_mem__ T *xIn = (__local_mem__ T *)xInLocal.GetPhyAddr();
    __local_mem__ float *gamma = (__local_mem__ float *)gammaLocal.GetPhyAddr();
    constexpr uint32_t columnCount = 128U;
    constexpr uint32_t vectorOffset = MHC_PRE_BASIC_API_VL_FP32;
    constexpr uint32_t srcStride = columnCount;
    constexpr uint32_t dstStride = columnCount;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> xInReg;
        AscendC::MicroAPI::RegTensor<float> gamma0Reg;
        AscendC::MicroAPI::RegTensor<float> gamma1Reg;
        AscendC::MicroAPI::RegTensor<float> xReg;
        AscendC::MicroAPI::RegTensor<float> xGammaReg;
        AscendC::MicroAPI::RegTensor<float> squareReg;
        AscendC::MicroAPI::RegTensor<float> squareSumReg;
        AscendC::MicroAPI::RegTensor<float> partialReg;
        AscendC::MicroAPI::RegTensor<float> sumReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float>();

        AscendC::MicroAPI::LoadAlign(gamma0Reg, gamma);
        AscendC::MicroAPI::LoadAlign(gamma1Reg, gamma + vectorOffset);
        for (uint16_t row = 0; row < rowCount; ++row) {
            uint32_t rowSrcOffset = static_cast<uint32_t>(row) * srcStride;
            uint32_t rowDstOffset = static_cast<uint32_t>(row) * dstStride;
            AscendC::MicroAPI::Duplicate(squareSumReg, 0.0f);

            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xIn + rowSrcOffset);
            AscendC::MicroAPI::Cast<float, T, MHC_PRE_BASIC_API_CAST_B16_TO_B32>(xReg, xInReg, mask);
            AscendC::MicroAPI::Mul(xGammaReg, gamma0Reg, xReg, mask);
            AscendC::MicroAPI::StoreAlign(xOut + rowDstOffset, xGammaReg, mask);
            AscendC::MicroAPI::Mul(squareReg, xReg, xReg, mask);
            AscendC::MicroAPI::Add(squareSumReg, squareSumReg, squareReg, mask);

            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                xInReg, xIn + rowSrcOffset + vectorOffset);
            AscendC::MicroAPI::Cast<float, T, MHC_PRE_BASIC_API_CAST_B16_TO_B32>(xReg, xInReg, mask);
            AscendC::MicroAPI::Mul(xGammaReg, gamma1Reg, xReg, mask);
            AscendC::MicroAPI::StoreAlign(xOut + rowDstOffset + vectorOffset, xGammaReg, mask);
            AscendC::MicroAPI::Mul(squareReg, xReg, xReg, mask);
            AscendC::MicroAPI::Add(squareSumReg, squareSumReg, squareReg, mask);

            AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(partialReg, squareSumReg, mask);
            if constexpr (IS_FIRST_K) {
                AscendC::MicroAPI::Store(rms + row, partialReg, 1U);
            } else {
                AscendC::MicroAPI::Load(sumReg, rms + row);
                AscendC::MicroAPI::Add(sumReg, sumReg, partialReg, mask);
                AscendC::MicroAPI::Store(rms + row, sumReg, 1U);
            }
        }
    }
}

template <typename T, bool IS_FIRST_K>
__aicore__ inline void MhcPreMkVFProcessXInGamma(const LocalTensor<float> &xOutLocal,
                                                 const LocalTensor<float> &rmsLocal, const LocalTensor<T> &xInLocal,
                                                 const LocalTensor<float> &gammaLocal, uint16_t rowCount,
                                                 uint16_t columnCount)
{
    // Hot-shape specializations keep RMS sums in registers and avoid repeated Reduce/load/store work.
    if (rowCount <= MHC_PRE_MK_RMS_REG_ROW_LIMIT && columnCount != 128U) {
        MhcPreMkVFProcessXInGammaSmallRows<T, IS_FIRST_K>(xOutLocal, rmsLocal, xInLocal, gammaLocal, rowCount,
                                                          columnCount);
        return;
    }
    if (columnCount == 128U) {
        MhcPreMkVFProcessXInGammaFull128<T, IS_FIRST_K>(xOutLocal, rmsLocal, xInLocal, gammaLocal, rowCount);
        return;
    }
    uint16_t loopCount = BasicApiCeilDiv(columnCount, MHC_PRE_BASIC_API_VL_FP32);
    if constexpr (IS_FIRST_K) {
        MhcPreMkVFProcessXInGammaLoops<T, true>(xOutLocal, rmsLocal, xInLocal, gammaLocal, rowCount, columnCount, 0U,
                                                1U);
        if (loopCount > 1U) {
            MhcPreMkVFProcessXInGammaLoops<T, false>(xOutLocal, rmsLocal, xInLocal, gammaLocal, rowCount, columnCount,
                                                     1U, loopCount - 1U);
        }
    } else {
        MhcPreMkVFProcessXInGammaLoops<T, false>(xOutLocal, rmsLocal, xInLocal, gammaLocal, rowCount, columnCount, 0U,
                                                 loopCount);
    }
}
__aicore__ inline void MhcPreMkVFReducePartials(const LocalTensor<float> &mmOutLocal,
                                                const LocalTensor<float> &invRmsOutLocal,
                                                const LocalTensor<float> &mmPartialLocal,

                                                const LocalTensor<float> &rmsPartialLocal, uint16_t groupK,
                                                uint16_t rowCount, uint16_t fusionSize, uint16_t mmSegmentOffset,
                                                uint16_t mmSegmentSize, uint16_t rmsStride, float scaleMean,
                                                float normEps)
{
    __local_mem__ float *mmOut = (__local_mem__ float *)mmOutLocal.GetPhyAddr();
    __local_mem__ float *invRmsOut = (__local_mem__ float *)invRmsOutLocal.GetPhyAddr();
    __local_mem__ float *mmPartial = (__local_mem__ float *)mmPartialLocal.GetPhyAddr();
    __local_mem__ float *rmsPartial = (__local_mem__ float *)rmsPartialLocal.GetPhyAddr();
    uint16_t fourLoopCount = groupK / 4U;
    uint16_t tailLoopCount = groupK % 4U;
    uint32_t mmGroupStride = BasicApiRoundUp<float>(static_cast<uint32_t>(rowCount) * fusionSize);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> rms0;
        AscendC::MicroAPI::RegTensor<float> rms1;
        AscendC::MicroAPI::RegTensor<float> rms2;
        AscendC::MicroAPI::RegTensor<float> rms3;
        AscendC::MicroAPI::RegTensor<float> mm0;
        AscendC::MicroAPI::RegTensor<float> mm1;
        AscendC::MicroAPI::RegTensor<float> mm2;
        AscendC::MicroAPI::RegTensor<float> mm3;
        AscendC::MicroAPI::RegTensor<float> sumRms0;
        AscendC::MicroAPI::RegTensor<float> sumRms1;
        AscendC::MicroAPI::RegTensor<float> sumRms2;
        AscendC::MicroAPI::RegTensor<float> sumRms3;
        AscendC::MicroAPI::RegTensor<float> sumMm0;
        AscendC::MicroAPI::RegTensor<float> sumMm1;
        AscendC::MicroAPI::RegTensor<float> sumMm2;
        AscendC::MicroAPI::RegTensor<float> sumMm3;
        AscendC::MicroAPI::RegTensor<float> one;
        uint32_t rmsMaskSize = 1U;
        uint32_t mmMaskSize = mmSegmentSize;
        AscendC::MicroAPI::MaskReg rmsMask = AscendC::MicroAPI::UpdateMask<float>(rmsMaskSize);
        AscendC::MicroAPI::MaskReg mmMask = AscendC::MicroAPI::UpdateMask<float>(mmMaskSize);
        AscendC::MicroAPI::Duplicate(one, 1.0f, rmsMask);

        for (uint16_t row = 0; row < rowCount; ++row) {
            AscendC::MicroAPI::Duplicate(sumRms0, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumRms1, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumRms2, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumRms3, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumMm0, 0.0f, mmMask);
            AscendC::MicroAPI::Duplicate(sumMm1, 0.0f, mmMask);
            AscendC::MicroAPI::Duplicate(sumMm2, 0.0f, mmMask);
            AscendC::MicroAPI::Duplicate(sumMm3, 0.0f, mmMask);

            for (uint16_t loop = 0; loop < fourLoopCount; ++loop) {
                uint16_t kBase = loop * 4U;
                MhcPreBasicApiLoadBroadcast(rms0, rmsPartial, rmsMask, static_cast<uint32_t>(kBase) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms1, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 1U) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms2, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 2U) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms3, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 3U) * rmsStride + row);
                AscendC::MicroAPI::Add(sumRms0, sumRms0, rms0, rmsMask);
                AscendC::MicroAPI::Add(sumRms1, sumRms1, rms1, rmsMask);
                AscendC::MicroAPI::Add(sumRms2, sumRms2, rms2, rmsMask);
                AscendC::MicroAPI::Add(sumRms3, sumRms3, rms3, rmsMask);

                AscendC::MicroAPI::Load<float>(
                    mm0, mmPartial + static_cast<uint32_t>(kBase) * mmGroupStride + row * fusionSize + mmSegmentOffset);
                AscendC::MicroAPI::Load<float>(mm1, mmPartial + static_cast<uint32_t>(kBase + 1U) * mmGroupStride +
                                                        row * fusionSize + mmSegmentOffset);
                AscendC::MicroAPI::Load<float>(mm2, mmPartial + static_cast<uint32_t>(kBase + 2U) * mmGroupStride +
                                                        row * fusionSize + mmSegmentOffset);
                AscendC::MicroAPI::Load<float>(mm3, mmPartial + static_cast<uint32_t>(kBase + 3U) * mmGroupStride +
                                                        row * fusionSize + mmSegmentOffset);
                AscendC::MicroAPI::Add(sumMm0, sumMm0, mm0, mmMask);
                AscendC::MicroAPI::Add(sumMm1, sumMm1, mm1, mmMask);
                AscendC::MicroAPI::Add(sumMm2, sumMm2, mm2, mmMask);
                AscendC::MicroAPI::Add(sumMm3, sumMm3, mm3, mmMask);
            }

            for (uint16_t tail = 0; tail < tailLoopCount; ++tail) {
                uint16_t kIndex = fourLoopCount * 4U + tail;
                MhcPreBasicApiLoadBroadcast(rms0, rmsPartial, rmsMask, static_cast<uint32_t>(kIndex) * rmsStride + row);
                AscendC::MicroAPI::Add(sumRms0, sumRms0, rms0, rmsMask);
                AscendC::MicroAPI::Load<float>(mm0, mmPartial + static_cast<uint32_t>(kIndex) * mmGroupStride +
                                                        row * fusionSize + mmSegmentOffset);
                AscendC::MicroAPI::Add(sumMm0, sumMm0, mm0, mmMask);
            }

            AscendC::MicroAPI::Add(sumRms0, sumRms0, sumRms3, rmsMask);
            AscendC::MicroAPI::Add(sumRms1, sumRms1, sumRms2, rmsMask);
            AscendC::MicroAPI::Add(sumRms0, sumRms0, sumRms1, rmsMask);
            AscendC::MicroAPI::Muls(sumRms0, sumRms0, scaleMean, rmsMask);
            AscendC::MicroAPI::Adds(sumRms0, sumRms0, normEps, rmsMask);
            AscendC::MicroAPI::Sqrt(sumRms0, sumRms0, rmsMask);
            AscendC::MicroAPI::Div(sumRms0, one, sumRms0, rmsMask);

            AscendC::MicroAPI::Add(sumMm0, sumMm0, sumMm3, mmMask);
            AscendC::MicroAPI::Add(sumMm1, sumMm1, sumMm2, mmMask);
            AscendC::MicroAPI::Add(sumMm0, sumMm0, sumMm1, mmMask);
            AscendC::MicroAPI::Store<float>(invRmsOut + row, sumRms0, 1U);
            AscendC::MicroAPI::Store<float>(mmOut + row * fusionSize + mmSegmentOffset, sumMm0, mmSegmentSize);
        }
    }
}

__aicore__ inline void MhcPreMkVFReducePartialsSequentialGeneral(
    const LocalTensor<float> &mmOutLocal, const LocalTensor<float> &invRmsOutLocal,
    const LocalTensor<float> &mmPartialLocal, const LocalTensor<float> &rmsPartialLocal, uint16_t mmGroupK,
    uint16_t rmsGroupK, uint16_t rowCount, uint16_t fusionSize, uint16_t mmSegmentOffset, uint16_t mmSegmentSize,
    uint16_t rmsStride, float scaleMean, float normEps)
{
    __local_mem__ float *mmOut = (__local_mem__ float *)mmOutLocal.GetPhyAddr();
    __local_mem__ float *invRmsOut = (__local_mem__ float *)invRmsOutLocal.GetPhyAddr();
    __local_mem__ float *mmPartial = (__local_mem__ float *)mmPartialLocal.GetPhyAddr();
    __local_mem__ float *rmsPartial = (__local_mem__ float *)rmsPartialLocal.GetPhyAddr();
    uint16_t fourLoopCount = rmsGroupK / 4U;
    uint16_t tailLoopCount = rmsGroupK % 4U;
    uint32_t mmGroupStride = BasicApiRoundUp<float>(static_cast<uint32_t>(rowCount) * fusionSize);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> rms0;
        AscendC::MicroAPI::RegTensor<float> rms1;
        AscendC::MicroAPI::RegTensor<float> rms2;
        AscendC::MicroAPI::RegTensor<float> rms3;
        AscendC::MicroAPI::RegTensor<float> sumRms0;
        AscendC::MicroAPI::RegTensor<float> sumRms1;
        AscendC::MicroAPI::RegTensor<float> sumRms2;
        AscendC::MicroAPI::RegTensor<float> sumRms3;
        AscendC::MicroAPI::RegTensor<float> mm;
        AscendC::MicroAPI::RegTensor<float> sumMm;
        AscendC::MicroAPI::RegTensor<float> one;
        uint32_t rmsMaskSize = 1U;
        uint32_t mmMaskSize = mmSegmentSize;
        AscendC::MicroAPI::MaskReg rmsMask = AscendC::MicroAPI::UpdateMask<float>(rmsMaskSize);
        AscendC::MicroAPI::MaskReg mmMask = AscendC::MicroAPI::UpdateMask<float>(mmMaskSize);
        AscendC::MicroAPI::Duplicate(one, 1.0f, rmsMask);

        for (uint16_t row = 0; row < rowCount; ++row) {
            AscendC::MicroAPI::Duplicate(sumRms0, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumRms1, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumRms2, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumRms3, 0.0f, rmsMask);
            for (uint16_t loop = 0; loop < fourLoopCount; ++loop) {
                uint16_t kBase = loop * 4U;
                MhcPreBasicApiLoadBroadcast(rms0, rmsPartial, rmsMask, static_cast<uint32_t>(kBase) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms1, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 1U) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms2, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 2U) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms3, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 3U) * rmsStride + row);
                AscendC::MicroAPI::Add(sumRms0, sumRms0, rms0, rmsMask);
                AscendC::MicroAPI::Add(sumRms1, sumRms1, rms1, rmsMask);
                AscendC::MicroAPI::Add(sumRms2, sumRms2, rms2, rmsMask);
                AscendC::MicroAPI::Add(sumRms3, sumRms3, rms3, rmsMask);
            }
            for (uint16_t tail = 0; tail < tailLoopCount; ++tail) {
                uint16_t kIndex = fourLoopCount * 4U + tail;
                MhcPreBasicApiLoadBroadcast(rms0, rmsPartial, rmsMask, static_cast<uint32_t>(kIndex) * rmsStride + row);
                AscendC::MicroAPI::Add(sumRms0, sumRms0, rms0, rmsMask);
            }
            AscendC::MicroAPI::Add(sumRms0, sumRms0, sumRms3, rmsMask);
            AscendC::MicroAPI::Add(sumRms1, sumRms1, sumRms2, rmsMask);
            AscendC::MicroAPI::Add(sumRms0, sumRms0, sumRms1, rmsMask);
            AscendC::MicroAPI::Muls(sumRms0, sumRms0, scaleMean, rmsMask);
            AscendC::MicroAPI::Adds(sumRms0, sumRms0, normEps, rmsMask);
            AscendC::MicroAPI::Sqrt(sumRms0, sumRms0, rmsMask);
            AscendC::MicroAPI::Div(sumRms0, one, sumRms0, rmsMask);

            AscendC::MicroAPI::Duplicate(sumMm, 0.0f, mmMask);
            for (uint16_t kIndex = 0; kIndex < mmGroupK; ++kIndex) {
                AscendC::MicroAPI::Load<float>(
                    mm, mmPartial + static_cast<uint32_t>(kIndex) * mmGroupStride + row * fusionSize + mmSegmentOffset);
                AscendC::MicroAPI::Add(sumMm, sumMm, mm, mmMask);
            }
            AscendC::MicroAPI::Store<float>(invRmsOut + row, sumRms0, 1U);
            AscendC::MicroAPI::Store<float>(mmOut + row * fusionSize + mmSegmentOffset, sumMm, mmSegmentSize);
        }
    }
}

__aicore__ inline void MhcPreMkVFReducePartialsSequentialFlat(
    const LocalTensor<float> &mmOutLocal, const LocalTensor<float> &invRmsOutLocal,
    const LocalTensor<float> &mmPartialLocal, const LocalTensor<float> &rmsPartialLocal, uint16_t mmGroupK,
    uint16_t rmsGroupK, uint16_t rowCount, uint16_t fusionSize, uint16_t rmsStride, float scaleMean, float normEps)
{
    __local_mem__ float *mmOut = (__local_mem__ float *)mmOutLocal.GetPhyAddr();
    __local_mem__ float *invRmsOut = (__local_mem__ float *)invRmsOutLocal.GetPhyAddr();
    __local_mem__ float *mmPartial = (__local_mem__ float *)mmPartialLocal.GetPhyAddr();
    __local_mem__ float *rmsPartial = (__local_mem__ float *)rmsPartialLocal.GetPhyAddr();
    uint16_t fourLoopCount = rmsGroupK / 4U;
    uint16_t tailLoopCount = rmsGroupK % 4U;
    uint32_t mmElements = static_cast<uint32_t>(rowCount) * fusionSize;
    uint32_t mmGroupStride = BasicApiRoundUp<float>(mmElements);
    uint16_t mmLoopCount = BasicApiCeilDiv(mmElements, MHC_PRE_BASIC_API_VL_FP32);

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> rms0;
        AscendC::MicroAPI::RegTensor<float> rms1;
        AscendC::MicroAPI::RegTensor<float> rms2;
        AscendC::MicroAPI::RegTensor<float> rms3;
        AscendC::MicroAPI::RegTensor<float> sumRms0;
        AscendC::MicroAPI::RegTensor<float> sumRms1;
        AscendC::MicroAPI::RegTensor<float> sumRms2;
        AscendC::MicroAPI::RegTensor<float> sumRms3;
        AscendC::MicroAPI::RegTensor<float> mm;
        AscendC::MicroAPI::RegTensor<float> sumMm;
        AscendC::MicroAPI::RegTensor<float> one;
        uint32_t rmsMaskSize = 1U;
        AscendC::MicroAPI::MaskReg rmsMask = AscendC::MicroAPI::UpdateMask<float>(rmsMaskSize);
        AscendC::MicroAPI::Duplicate(one, 1.0f, rmsMask);

        for (uint16_t row = 0; row < rowCount; ++row) {
            AscendC::MicroAPI::Duplicate(sumRms0, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumRms1, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumRms2, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumRms3, 0.0f, rmsMask);
            for (uint16_t loop = 0; loop < fourLoopCount; ++loop) {
                uint16_t kBase = loop * 4U;
                MhcPreBasicApiLoadBroadcast(rms0, rmsPartial, rmsMask, static_cast<uint32_t>(kBase) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms1, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 1U) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms2, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 2U) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms3, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 3U) * rmsStride + row);
                AscendC::MicroAPI::Add(sumRms0, sumRms0, rms0, rmsMask);
                AscendC::MicroAPI::Add(sumRms1, sumRms1, rms1, rmsMask);
                AscendC::MicroAPI::Add(sumRms2, sumRms2, rms2, rmsMask);
                AscendC::MicroAPI::Add(sumRms3, sumRms3, rms3, rmsMask);
            }
            for (uint16_t tail = 0; tail < tailLoopCount; ++tail) {
                uint16_t kIndex = fourLoopCount * 4U + tail;
                MhcPreBasicApiLoadBroadcast(rms0, rmsPartial, rmsMask, static_cast<uint32_t>(kIndex) * rmsStride + row);
                AscendC::MicroAPI::Add(sumRms0, sumRms0, rms0, rmsMask);
            }
            AscendC::MicroAPI::Add(sumRms0, sumRms0, sumRms3, rmsMask);
            AscendC::MicroAPI::Add(sumRms1, sumRms1, sumRms2, rmsMask);
            AscendC::MicroAPI::Add(sumRms0, sumRms0, sumRms1, rmsMask);
            AscendC::MicroAPI::Muls(sumRms0, sumRms0, scaleMean, rmsMask);
            AscendC::MicroAPI::Adds(sumRms0, sumRms0, normEps, rmsMask);
            AscendC::MicroAPI::Sqrt(sumRms0, sumRms0, rmsMask);
            AscendC::MicroAPI::Div(sumRms0, one, sumRms0, rmsMask);
            AscendC::MicroAPI::Store<float>(invRmsOut + row, sumRms0, 1U);
        }

        uint32_t remaining = mmElements;
        for (uint16_t loop = 0; loop < mmLoopCount; ++loop) {
            uint32_t currentCount = AscendC::Std::min(remaining, MHC_PRE_BASIC_API_VL_FP32);
            AscendC::MicroAPI::MaskReg mmMask = AscendC::MicroAPI::UpdateMask<float>(remaining);
            uint32_t vectorOffset = static_cast<uint32_t>(loop) * MHC_PRE_BASIC_API_VL_FP32;
            AscendC::MicroAPI::Duplicate(sumMm, 0.0f, mmMask);
            for (uint16_t kIndex = 0; kIndex < mmGroupK; ++kIndex) {
                AscendC::MicroAPI::Load<float>(
                    mm, mmPartial + static_cast<uint32_t>(kIndex) * mmGroupStride + vectorOffset);
                AscendC::MicroAPI::Add(sumMm, sumMm, mm, mmMask);
            }
            AscendC::MicroAPI::Store<float>(mmOut + vectorOffset, sumMm, currentCount);
        }
    }
}

template <class T, class P, int8_t RESI_MODE>
class MhcPreMKPart1 {
public:
    __aicore__ inline MhcPreMKPart1() = default;

    __aicore__ inline void Init(InitParams initParams)
    {
        vector_.BindGlobalTensors(initParams);
        vector_.InitFromTilingData(initParams.tilingData);
        vector_.InitMNConfig();
        vector_.pipe_ = initParams.tPipeIn;
        partialMmGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ P *>(initParams.workspace + vector_.tiling_->mkWorkspaceMmOffset));
        partialRmsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ P *>(initParams.workspace + vector_.tiling_->mkWorkspaceRmsOffset));
        if (UseGmStage()) {
            xStageGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ P *>(initParams.workspace + vector_.tiling_->mkWorkspaceFinalOffset));
        }

        vector_.pipe_->InitBuffer(l1Buffer_, MHC_PRE_BASIC_API_L1_ALLOC_SIZE);
        aL1_ = l1Buffer_.Get<P>();
        bL1_ = aL1_[MHC_PRE_BASIC_API_L1_BUF_NUM * MHC_PRE_BASIC_API_L1_BUF_OFFSET];

        if ASCEND_IS_AIC {
            mmService_.Init(vector_.implMode_);
            // Seed two availability tokens for each subblock because X staging
            // is ping-ponged. Every later token is returned after AIC consumes X.
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG);
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG +
                                                                      MHC_PRE_SUBBLOCK_FLAG_OFFSET);
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG);
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG +
                                                                      MHC_PRE_SUBBLOCK_FLAG_OFFSET);
        }
        if ASCEND_IS_AIV {
            uint32_t rowCapacity = (vector_.tiling_->mL1Size + 1U) / 2U;
            uint32_t kCapacity = BasicApiRoundUp<T>(vector_.tiling_->kUbSize);
            vector_.pipe_->InitBuffer(vector_.xInQueue_, 2, rowCapacity * kCapacity * sizeof(T));
            // GM staging is ping-ponged to overlap AIV MTE3, AIC MTE2 ND2NZ, and Cube execution.
            // Direct-L1 staging keeps one FP32 source because software ND2NZ is serialized on Vector.
            vector_.pipe_->InitBuffer(vector_.outQueue_, UseGmStage() ? 2 : 1,
                                      rowCapacity * BasicApiRoundUp<P>(vector_.tiling_->kUbSize) * sizeof(P));
            vector_.pipe_->InitBuffer(vector_.invRmsOutQueue_, 1, BasicApiRoundUp<P>(rowCapacity) * sizeof(P));
            if (vector_.hasGamma_) {
                vector_.pipe_->InitBuffer(vector_.gammaInQueue_, 2,
                                          BasicApiRoundUp<P>(vector_.tiling_->kUbSize) * sizeof(P));
            }
            if (!UseGmStage()) {
                vector_.pipe_->InitBuffer(nd2NzBuf_, BasicApiAlign(rowCapacity, MHC_PRE_BASIC_API_C0_SIZE) *
                                                         BasicApiRoundUp<P>(vector_.tiling_->kUbSize) * sizeof(P));
                nd2NzLocal_ = nd2NzBuf_.Get<P>();
            }
            vector_.invRmsUb_ = vector_.invRmsOutQueue_.template AllocTensor<P>();
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t aicIndex = GetBlockIdx();
        if ASCEND_IS_AIV {
            aicIndex /= 2U;
        }
        uint32_t usedCoreNum = vector_.tiling_->cubeBlockDimM * vector_.tiling_->cubeBlockDimK;
        if (aicIndex < usedCoreNum) {
            uint32_t mIndex = aicIndex / vector_.tiling_->cubeBlockDimK;
            uint32_t kIndex = aicIndex % vector_.tiling_->cubeBlockDimK;
            uint32_t mStart = mIndex * vector_.tiling_->mL1Size;
            uint32_t mReal =
                AscendC::Std::min(static_cast<uint64_t>(vector_.tiling_->mL1Size), vector_.totalLength_ - mStart);
            uint32_t kStart = kIndex * vector_.tiling_->multCoreSplitKSize;
            uint32_t kEnd = AscendC::Std::min(static_cast<uint64_t>(kStart + vector_.tiling_->multCoreSplitKSize),
                                              static_cast<uint64_t>(vector_.matrixInfo_.nD));
            if ASCEND_IS_AIC {
                ProcessAic(mStart, mReal, kIndex, kStart, kEnd);
            } else {
                ProcessAiv(mStart, mReal, kIndex, kStart, kEnd);
            }
        }
        if ASCEND_IS_AIC {
            mmService_.End(vector_.implMode_);
        }
        SyncAll<false>();
    }

private:
    __aicore__ inline bool UseSequentialKPartials() const
    {
        return vector_.tiling_->multCoreSplitKSize >= MHC_PRE_MK_SEQUENTIAL_PARTIAL_THRESHOLD;
    }

    __aicore__ inline bool UseGmStage() const
    {
        return vector_.tiling_->mkUseGmStage != 0U;
    }

    __aicore__ inline void ProcessAic(uint32_t mStart, uint32_t mReal, uint32_t kIndex, uint32_t kStart, uint32_t kEnd)
    {
        bool useSequentialPartials = UseSequentialKPartials();
        uint64_t partialK =
            useSequentialPartials ? MHC_PRE_MK_SEQUENTIAL_PARTIAL_K : static_cast<uint64_t>(kEnd - kStart);
        uint64_t mAlign = BasicApiAlign(mReal, AscendC::BLOCK_CUBE);
        uint64_t nAlign = BasicApiAlign(vector_.matrixInfo_.fusionSize, AscendC::BLOCK_CUBE);
        // Derive baseK from the active M/N footprint so the low-level path fills L0 across N/D shapes.
        uint64_t baseK = (256U / AscendC::Std::max(mAlign, nAlign)) * 32U;
        uint64_t stageSlotElements = static_cast<uint64_t>(vector_.tiling_->mL1Size) * vector_.tiling_->kL1Size;
        uint64_t stageCoreOffset = 2U * GetBlockIdx() * stageSlotElements;
        uint64_t partialBaseIndex = useSequentialPartials ? 0U : kIndex;
        uint64_t partialBaseOffset =
            (partialBaseIndex * vector_.totalLength_ + mStart) * vector_.matrixInfo_.fusionSize;
        uint64_t partialGroupStride = vector_.totalLength_ * vector_.matrixInfo_.fusionSize;
        mmService_.ProcessKRange(mReal, vector_.matrixInfo_.fusionSize, vector_.matrixInfo_.nD, mReal, baseK, kStart,
                                 kEnd, vector_.tiling_->kL1Size, useSequentialPartials, partialK, UseGmStage(),
                                 stageSlotElements, stageCoreOffset, vector_.phiGm_, xStageGm_,
                                 partialMmGm_[partialBaseOffset], partialGroupStride, aL1_, bL1_);
    }
    __aicore__ inline void ProcessAiv(uint32_t mStart, uint32_t mReal, uint32_t kIndex, uint32_t kStart, uint32_t kEnd)
    {
        uint32_t firstRows = (mReal + 1U) / 2U;
        uint32_t rowStart = GetSubBlockIdx() == 0 ? 0U : firstRows;
        uint32_t rowCount = GetSubBlockIdx() == 0 ? firstRows : mReal - firstRows;
        uint8_t aL1BufferId = 0;
        vector_.globalOffsetM_ = mStart;
        if (rowCount == 0U) {
            for (uint32_t kOffset = kStart; kOffset < kEnd; kOffset += vector_.tiling_->kL1Size) {
                CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_CONSUMED_FLAG);
                CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_READY_FLAG);
            }
            vector_.invRmsOutQueue_.FreeTensor(vector_.invRmsUb_);
            return;
        }
        if (!UseGmStage()) {
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        uint32_t firstK =
            AscendC::Std::min(static_cast<uint64_t>(vector_.tiling_->kL1Size), static_cast<uint64_t>(kEnd - kStart));
        vector_.xLocal_ = vector_.xInQueue_.template AllocTensor<T>();
        vector_.DataCopyX(rowCount, firstK, rowStart, kStart);
        if (vector_.hasGamma_) {
            vector_.gammaUb_ = vector_.gammaInQueue_.template AllocTensor<P>();
            vector_.DataCopyGamma(firstK, kStart);
        }

        for (uint32_t kOffset = kStart; kOffset < kEnd; kOffset += vector_.tiling_->kL1Size) {
            uint32_t currentK = AscendC::Std::min(static_cast<uint64_t>(vector_.tiling_->kL1Size),
                                                  static_cast<uint64_t>(kEnd - kOffset));
            CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_CONSUMED_FLAG);

            LocalTensor<T> currentXLocal = vector_.xInQueue_.template DeQue<T>();
            LocalTensor<P> xFloatLocal = vector_.outQueue_.template AllocTensor<P>();
            bool initializeRms = kOffset == kStart;
            LocalTensor<P> currentGammaLocal;
            if (vector_.hasGamma_) {
                currentGammaLocal = vector_.gammaInQueue_.template DeQue<P>();
            }

            // Prefetch the next X/gamma tile before processing the current tile.
            uint32_t nextKOffset = kOffset + vector_.tiling_->kL1Size;
            if (nextKOffset < kEnd) {
                uint32_t nextK = AscendC::Std::min(static_cast<uint64_t>(vector_.tiling_->kL1Size),
                                                   static_cast<uint64_t>(kEnd - nextKOffset));
                vector_.xLocal_ = vector_.xInQueue_.template AllocTensor<T>();
                vector_.DataCopyX(rowCount, nextK, rowStart, nextKOffset);
                if (vector_.hasGamma_) {
                    vector_.gammaUb_ = vector_.gammaInQueue_.template AllocTensor<P>();
                    vector_.DataCopyGamma(nextK, nextKOffset);
                }
            }

            if (vector_.hasGamma_) {
                if (initializeRms) {
                    MhcPreMkVFProcessXInGamma<T, true>(xFloatLocal, vector_.invRmsUb_, currentXLocal, currentGammaLocal,
                                                       rowCount, currentK);
                } else {
                    MhcPreMkVFProcessXInGamma<T, false>(xFloatLocal, vector_.invRmsUb_, currentXLocal,
                                                        currentGammaLocal, rowCount, currentK);
                }
                vector_.gammaInQueue_.FreeTensor(currentGammaLocal);
            } else if (initializeRms) {
                vector_.template VFDoV0ProcessXIn<false, true>(
                    (__ubuf__ P *)xFloatLocal.GetPhyAddr(), (__ubuf__ P *)vector_.invRmsUb_.GetPhyAddr(),
                    (__ubuf__ T *)currentXLocal.GetPhyAddr(), nullptr, rowCount, currentK);
            } else {
                vector_.template VFDoV0ProcessXIn<false, false>(
                    (__ubuf__ P *)xFloatLocal.GetPhyAddr(), (__ubuf__ P *)vector_.invRmsUb_.GetPhyAddr(),
                    (__ubuf__ T *)currentXLocal.GetPhyAddr(), nullptr, rowCount, currentK);
            }
            vector_.xInQueue_.FreeTensor(currentXLocal);

            if (UseGmStage()) {
                // Alternate two per-AIC GM slots to keep the AIV-to-GM-to-L1-to-Cube pipeline in flight.
                vector_.outQueue_.template EnQue<P>(xFloatLocal);
                xFloatLocal = vector_.outQueue_.template DeQue<P>();
                DataCopyExtParams xCopyParams;
                xCopyParams.blockCount = static_cast<uint16_t>(rowCount);
                xCopyParams.blockLen = currentK * sizeof(P);
                xCopyParams.srcStride = (BasicApiRoundUp<P>(currentK) - currentK) * sizeof(P);
                xCopyParams.dstStride = (vector_.tiling_->kL1Size - currentK) * sizeof(P);
                uint64_t stageSlotElements = static_cast<uint64_t>(vector_.tiling_->mL1Size) * vector_.tiling_->kL1Size;
                uint64_t stageOffset = (2U * (GetBlockIdx() / 2U) + aL1BufferId) * stageSlotElements +
                                       static_cast<uint64_t>(rowStart) * vector_.tiling_->kL1Size;
                DataCopyPad(xStageGm_[stageOffset], xFloatLocal, xCopyParams);
                vector_.outQueue_.FreeTensor(xFloatLocal);
            } else {
                constexpr event_t eventId = EVENT_ID0;
                WaitFlag<HardEvent::MTE3_V>(eventId);
                LocalTensor<P> currentNd2Nz = nd2NzLocal_;
                MhcPreMkVFTransND2NZ(currentNd2Nz, xFloatLocal, rowCount, currentK);
                vector_.outQueue_.FreeTensor(xFloatLocal);
                SetFlag<HardEvent::V_MTE3>(eventId);
                WaitFlag<HardEvent::V_MTE3>(eventId);
                DataCopyParams copyParams;
                copyParams.blockCount = BasicApiCeilDiv(currentK, MHC_PRE_BASIC_API_C0_SIZE);
                copyParams.blockLen = rowCount * MHC_PRE_BASIC_API_C0_SIZE * sizeof(P) / MHC_PRE_BASIC_API_BLOCK_SIZE;
                copyParams.srcStride = BasicApiAlign(rowCount, MHC_PRE_BASIC_API_C0_SIZE) - rowCount;
                copyParams.dstStride = BasicApiAlign(mReal, AscendC::BLOCK_CUBE) - rowCount;
                uint32_t dstRowOffset =
                    GetSubBlockIdx() == 0 ? 0U : firstRows * (MHC_PRE_BASIC_API_BLOCK_SIZE / sizeof(P));
                BasicApiCopyToL1(currentNd2Nz, aL1_[aL1BufferId * MHC_PRE_BASIC_API_L1_BUF_OFFSET + dstRowOffset],
                                 copyParams);
                SetFlag<HardEvent::MTE3_V>(eventId);
            }
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_READY_FLAG);
            aL1BufferId ^= 1U;
        }
        if (!UseGmStage()) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = rowCount * sizeof(P);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        uint64_t rmsOffset = static_cast<uint64_t>(kIndex) * vector_.totalLength_ + mStart + rowStart;
        DataCopyPad(partialRmsGm_[rmsOffset], vector_.invRmsUb_, copyParams);
        vector_.invRmsOutQueue_.FreeTensor(vector_.invRmsUb_);
    }

private:
    MhcPreVectorCompute<T, P, RESI_MODE> vector_;
    GlobalTensor<P> partialMmGm_;
    GlobalTensor<P> partialRmsGm_;
    GlobalTensor<P> xStageGm_;
    TBuf<TPosition::A1> l1Buffer_;
    TBuf<TPosition::VECCALC> nd2NzBuf_;
    LocalTensor<P> aL1_;
    LocalTensor<P> bL1_;
    LocalTensor<P> nd2NzLocal_;
    MhcPreCubeCompute mmService_;
};

template <class T, class P, int8_t RESI_MODE>
class MhcPreMKPart2 {
public:
    __aicore__ inline MhcPreMKPart2() = default;

    __aicore__ inline void Init(InitParams initParams)
    {
        vector_.BindGlobalTensors(initParams);
        vector_.InitFromTilingData(initParams.tilingData);
        vector_.InitMNConfig();
        vector_.pipe_ = initParams.tPipeIn;
        partialMmGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ P *>(initParams.workspace + vector_.tiling_->mkWorkspaceMmOffset));
        partialRmsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ P *>(initParams.workspace + vector_.tiling_->mkWorkspaceRmsOffset));
        if (vector_.outFlag_) {
            vector_.hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_mix));
        }
        vector_.InitUbBuffers(false, kPart2MaxRows);

        if ASCEND_IS_AIV {
            uint32_t aivIndex = GetBlockIdx();
            selected_ = aivIndex < vector_.tiling_->stage2UsedAivNum;
            if (selected_) {
                uint64_t absoluteMStart = static_cast<uint64_t>(aivIndex) * vector_.tiling_->stage2RowsPerCore;
                uint64_t mIndex = absoluteMStart / vector_.tiling_->mL1Size;
                uint64_t localMStart = absoluteMStart - mIndex * vector_.tiling_->mL1Size;
                vector_.coreIdx_ = mIndex;
                vector_.globalOffsetM_ = static_cast<uint64_t>(mIndex) * vector_.tiling_->mL1Size;
                vector_.curSingleT_ = AscendC::Std::min(static_cast<uint64_t>(vector_.tiling_->stage2RowsPerCore),
                                                        vector_.totalLength_ - absoluteMStart);
                vector_.mnConfig_.curSingleCoreM = vector_.curSingleT_;
                vector_.vectorOffset_.singleCoreM = vector_.curSingleT_;
                vector_.vectorOffset_.offsetMStart = localMStart;
                vector_.vectorOffset_.offsetMEnd = localMStart + vector_.curSingleT_;
                vector_.AIVPreLoad();
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        if (!selected_ || vector_.vectorOffset_.singleCoreM == 0) {
            return;
        }
        ReduceAndPostProcess();
        vector_.invRmsOutQueue_.FreeTensor(vector_.invRmsUb_);
        vector_.biasInQue_.FreeTensor(vector_.biasInUb_);
    }

private:
    // Part2 UB reduction capacity and row alignment.
    static constexpr uint32_t kPart2MaxRows = 32U;
    static constexpr uint32_t kPart2RowAlign = 8U;
    static constexpr uint32_t kPart2PartialBufferElements = 80U * 1024U / sizeof(P);

    __aicore__ inline bool UseSequentialKPartials() const
    {
        return vector_.tiling_->multCoreSplitKSize >= MHC_PRE_MK_SEQUENTIAL_PARTIAL_THRESHOLD;
    }

    __aicore__ inline uint32_t GetReduceRows(uint32_t remainingRows, uint32_t mmGroupK, uint32_t rmsGroupK,
                                             uint32_t fusionSize) const
    {
        uint32_t rows = AscendC::Std::min(remainingRows, kPart2MaxRows);
        if (rows > kPart2RowAlign) {
            rows = rows / kPart2RowAlign * kPart2RowAlign;
        }
        while (rows > 0U) {
            uint32_t mmElements = mmGroupK * BasicApiRoundUp<P>(rows * fusionSize);
            uint32_t rmsElements = rmsGroupK * BasicApiRoundUp<P>(rows);
            if (mmElements + rmsElements <= kPart2PartialBufferElements) {
                return rows;
            }
            rows -= rows > kPart2RowAlign ? kPart2RowAlign : 1U;
        }
        return 1U;
    }
    __aicore__ inline void ReduceAndPostProcess()
    {
        uint64_t localOffset = 0;
        bool useSequentialKPartials = UseSequentialKPartials();
        uint32_t groupK = useSequentialKPartials ?
                              BasicApiCeilDiv(vector_.matrixInfo_.nD, MHC_PRE_MK_SEQUENTIAL_PARTIAL_K) :
                              vector_.tiling_->cubeBlockDimK;
        uint32_t rmsGroupK = vector_.tiling_->cubeBlockDimK;
        uint32_t fusionSize = vector_.matrixInfo_.fusionSize;
        bool useSequentialMmReduce = useSequentialKPartials || vector_.implMode_ == MHC_PRE_IMPL_MODE_HF32;
        bool useFlatSequentialReduce =
            useSequentialKPartials &&
            (vector_.totalLength_ >= MHC_PRE_MK_FLAT_REDUCE_MIN_M || groupK >= MHC_PRE_MK_FLAT_REDUCE_MIN_GROUP_K);
        // Flatten large ordered partial groups to remove nested per-row VF setup without changing sum order.
        for (uint32_t offsetT = vector_.vectorOffset_.offsetMStart; offsetT < vector_.vectorOffset_.offsetMEnd;) {
            uint32_t lenT = GetReduceRows(vector_.vectorOffset_.offsetMEnd - offsetT, groupK, rmsGroupK, fusionSize);
            uint32_t rmsStride = BasicApiRoundUp<P>(lenT);
            uint32_t mmGroupStride = BasicApiRoundUp<P>(lenT * fusionSize);
            uint32_t mmPartialElements = groupK * mmGroupStride;
            LocalTensor<P> partialLocal = vector_.xInQueue_.template AllocTensor<P>();
            LocalTensor<P> rmsPartialLocal = partialLocal[mmPartialElements];

            DataCopyExtParams mmCopyParams;
            mmCopyParams.blockCount = groupK;
            mmCopyParams.blockLen = lenT * fusionSize * sizeof(P);
            mmCopyParams.srcStride = (vector_.totalLength_ - lenT) * fusionSize * sizeof(P);
            mmCopyParams.dstStride = 0;
            DataCopyPadExtParams<P> mmPadParams{true, 0, 0, 0};
            uint64_t mmSourceOffset = (vector_.globalOffsetM_ + offsetT) * fusionSize;
            DataCopyPad(partialLocal, partialMmGm_[mmSourceOffset], mmCopyParams, mmPadParams);

            DataCopyExtParams rmsCopyParams;
            rmsCopyParams.blockCount = useSequentialKPartials ? vector_.tiling_->cubeBlockDimK : groupK;
            rmsCopyParams.blockLen = lenT * sizeof(P);
            rmsCopyParams.srcStride = (vector_.totalLength_ - lenT) * sizeof(P);
            rmsCopyParams.dstStride = 0;
            DataCopyPadExtParams<P> rmsPadParams{true, 0, static_cast<uint8_t>(rmsStride - lenT), 0};
            uint64_t rmsSourceOffset = vector_.globalOffsetM_ + offsetT;
            DataCopyPad(rmsPartialLocal, partialRmsGm_[rmsSourceOffset], rmsCopyParams, rmsPadParams);

            vector_.xInQueue_.template EnQue<P>(partialLocal);
            partialLocal = vector_.xInQueue_.template DeQue<P>();
            rmsPartialLocal = partialLocal[mmPartialElements];
            LocalTensor<P> accumulator = vector_.outQueue_.template AllocTensor<P>();
            uint32_t invRmsOffset = offsetT - vector_.vectorOffset_.offsetMStart;
            uint16_t firstSegmentSize = static_cast<uint16_t>(AscendC::Std::min(fusionSize, MHC_PRE_BASIC_API_VL_FP32));
            if (useFlatSequentialReduce) {
                MhcPreMkVFReducePartialsSequentialFlat(
                    accumulator, vector_.invRmsUb_[invRmsOffset], partialLocal, rmsPartialLocal,
                    static_cast<uint16_t>(groupK), static_cast<uint16_t>(vector_.tiling_->cubeBlockDimK),
                    static_cast<uint16_t>(lenT), static_cast<uint16_t>(fusionSize), static_cast<uint16_t>(rmsStride),
                    vector_.scaleMean_, vector_.matrixInfo_.normEps);
            } else if (useSequentialMmReduce) {
                MhcPreMkVFReducePartialsSequentialGeneral(
                    accumulator, vector_.invRmsUb_[invRmsOffset], partialLocal, rmsPartialLocal,
                    static_cast<uint16_t>(groupK), static_cast<uint16_t>(vector_.tiling_->cubeBlockDimK),
                    static_cast<uint16_t>(lenT), static_cast<uint16_t>(fusionSize), 0U, firstSegmentSize,
                    static_cast<uint16_t>(rmsStride), vector_.scaleMean_, vector_.matrixInfo_.normEps);
            } else {
                MhcPreMkVFReducePartials(accumulator, vector_.invRmsUb_[invRmsOffset], partialLocal, rmsPartialLocal,
                                         static_cast<uint16_t>(groupK), static_cast<uint16_t>(lenT),
                                         static_cast<uint16_t>(fusionSize), 0U, firstSegmentSize,
                                         static_cast<uint16_t>(rmsStride), vector_.scaleMean_,
                                         vector_.matrixInfo_.normEps);
            }
            if (fusionSize > firstSegmentSize) {
                if (useSequentialMmReduce) {
                    MhcPreMkVFReducePartialsSequentialGeneral(
                        accumulator, vector_.invRmsUb_[invRmsOffset], partialLocal, rmsPartialLocal,
                        static_cast<uint16_t>(groupK), static_cast<uint16_t>(vector_.tiling_->cubeBlockDimK),
                        static_cast<uint16_t>(lenT), static_cast<uint16_t>(fusionSize), firstSegmentSize,
                        static_cast<uint16_t>(fusionSize - firstSegmentSize), static_cast<uint16_t>(rmsStride),
                        vector_.scaleMean_, vector_.matrixInfo_.normEps);
                } else {
                    MhcPreMkVFReducePartials(
                        accumulator, vector_.invRmsUb_[invRmsOffset], partialLocal, rmsPartialLocal,
                        static_cast<uint16_t>(groupK), static_cast<uint16_t>(lenT), static_cast<uint16_t>(fusionSize),
                        firstSegmentSize, static_cast<uint16_t>(fusionSize - firstSegmentSize),
                        static_cast<uint16_t>(rmsStride), vector_.scaleMean_, vector_.matrixInfo_.normEps);
                }
            }
            vector_.xInQueue_.FreeTensor(partialLocal);
            vector_.outQueue_.template EnQue<P>(accumulator);
            accumulator = vector_.outQueue_.template DeQue<P>();
            if (vector_.outFlag_) {
                DataCopyExtParams outputParams;
                outputParams.blockCount = 1;
                outputParams.blockLen = lenT * fusionSize * sizeof(P);
                outputParams.srcStride = 0;
                outputParams.dstStride = 0;
                uint64_t outputOffset = (vector_.globalOffsetM_ + offsetT) * fusionSize;
                DataCopyPad(vector_.hMixGm_[outputOffset], accumulator, outputParams);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
            vector_.template AIV1PostProcessTile<true>(accumulator, offsetT, lenT, static_cast<uint32_t>(localOffset),
                                                       fusionSize);
            vector_.outQueue_.FreeTensor(accumulator);
            offsetT += lenT;
            localOffset += lenT;
        }
        vector_.DataCopyOutInvRmsUb(vector_.vectorOffset_.singleCoreM, vector_.vectorOffset_.offsetMStart);
    }

private:
    MhcPreVectorCompute<T, P, RESI_MODE> vector_;
    GlobalTensor<P> partialMmGm_;
    GlobalTensor<P> partialRmsGm_;
    bool selected_ = false;
};

} // namespace MhcPre

#endif // MHC_PRE_M_K_SPLIT_H
