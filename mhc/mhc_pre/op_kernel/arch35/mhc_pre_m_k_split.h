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

#if defined(MHC_PRE_ENABLE_ASCEND950)
#define MHC_PRE_EXPERIMENT_WAVE_STAGES 1U
#endif

#include "mhc_pre_cube_compute.h"
#if defined(MHC_PRE_ENABLE_ASCEND950)
#include "mhc_pre_tiling_key.h"
#endif
#include "mhc_pre_vector_compute.h"

namespace MhcPre {

// Vector register specialization limits.
#if !defined(MHC_PRE_ENABLE_ASCEND950)
constexpr uint16_t MHC_PRE_MK_RMS_REG_ROW_LIMIT = 16U;
#endif

// Ordered K-partial policy used to match the reference accumulation order.
constexpr uint32_t MHC_PRE_MK_SEQUENTIAL_PARTIAL_K = 1024U;
constexpr uint32_t MHC_PRE_MK_SEQUENTIAL_PARTIAL_THRESHOLD = 2048U;

// Flat vector reduction thresholds for large workloads.
constexpr uint32_t MHC_PRE_MK_FLAT_REDUCE_MIN_M = 2048U;
constexpr uint32_t MHC_PRE_MK_FLAT_REDUCE_MIN_GROUP_K = 20U;
#if defined(MHC_PRE_ENABLE_ASCEND950)
// Keep in sync with the host work-count and workspace policy.
constexpr uint32_t MHC_PRE_MK_PRECISE_PROJECTION_MAX_M = 8192U;
constexpr uint32_t MHC_PRE_MK_PRECISE_PROJECTION_K = 512U;

__aicore__ inline uint32_t MhcPreMkWeightReusePartialK(uint32_t rows, uint32_t coreK, bool fp32, bool hasResidual)
{
    if (hasResidual && rows <= MHC_PRE_MK_FINE_REUSE_MAX_M) {
        return MhcPreMin(coreK, fp32 ? MHC_PRE_MK_FINE_PROJECTION_K : MHC_PRE_MK_PRECISE_PROJECTION_K);
    }
    if (fp32 && rows <= MHC_PRE_MK_PRECISE_PROJECTION_MAX_M) {
        return MhcPreMin(coreK, MHC_PRE_MK_PRECISE_PROJECTION_K);
    }
    return coreK >= MHC_PRE_MK_SEQUENTIAL_PARTIAL_THRESHOLD ? MHC_PRE_MK_SEQUENTIAL_PARTIAL_K : coreK;
}

__aicore__ inline uint32_t MhcPreMkFineOrdinaryProjectionK(uint32_t rows, uint32_t k, uint32_t splitK, uint32_t stage,
                                                           uint32_t mode)
{
    const uint32_t partialK =
        mode == MHC_PRE_IMPL_MODE_FP32 ? MHC_PRE_MK_FINE_FP32_ORDINARY_K : MHC_PRE_MK_FINE_PROJECTION_K;
    return stage != MHC_PRE_MK_SHARED_X_STAGE && rows <= MHC_PRE_MK_FINE_ORDINARY_MAX_M &&
                   k <= MHC_PRE_MK_FINE_ORDINARY_MAX_GROUPS * partialK && splitK > partialK ?
               partialK :
               0U;
}

__simd_vf__ inline void MhcPreMkVFTransND2NZAlignedImpl(__local_mem__ float* dst, __local_mem__ float* src,
                                                        uint16_t rowAlign, uint16_t dataBlockElements,
                                                        uint16_t columnAlign, uint16_t mainRowBlocks,
                                                        uint16_t columnLoops, uint32_t srcBlockStride)
{
    AscendC::Reg::RegTensor<float> data;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
    for (uint16_t rowBlock = 0; rowBlock < mainRowBlocks; ++rowBlock) {
        for (uint16_t columnBlock = 0; columnBlock < columnLoops; ++columnBlock) {
            AscendC::Reg::DataCopy<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
                data, src + rowBlock * MHC_PRE_BASIC_API_C0_SIZE * columnAlign + columnBlock * dataBlockElements,
                srcBlockStride, mask);
            AscendC::Reg::DataCopy(dst + rowBlock * MHC_PRE_BASIC_API_C0_SIZE * dataBlockElements +
                                       columnBlock * rowAlign * dataBlockElements,
                                   data, mask);
        }
    }
}

__simd_vf__ inline void MhcPreMkVFTransND2NZTailImpl(__local_mem__ float* dst, __local_mem__ float* src,
                                                     uint16_t rowAlign, uint16_t dataBlockElements,
                                                     uint16_t columnAlign, uint16_t mainRowBlocks, uint16_t tailRows,
                                                     uint16_t columnLoops, uint32_t srcBlockStride)
#else
__aicore__ inline void MhcPreMkVFTransND2NZ(const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal,
                                            uint16_t rowCount, uint16_t columnCount)
{
    __local_mem__ float* dst = (__local_mem__ float*)dstLocal.GetPhyAddr();
    __local_mem__ float* src = (__local_mem__ float*)srcLocal.GetPhyAddr();
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
            AscendC::Reg::RegTensor<float> data;
            AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
            for (uint16_t rowBlock = 0; rowBlock < mainRowBlocks; ++rowBlock) {
                for (uint16_t columnBlock = 0; columnBlock < columnLoops; ++columnBlock) {
                    AscendC::Reg::DataCopy<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
                        data,
                        src + rowBlock * MHC_PRE_BASIC_API_C0_SIZE * columnAlign + columnBlock * dataBlockElements,
                        srcBlockStride, mask);
                    AscendC::Reg::DataCopy(dst + rowBlock * MHC_PRE_BASIC_API_C0_SIZE * dataBlockElements +
                                               columnBlock * rowAlign * dataBlockElements,
                                           data, mask);
                }
            }
        }
        return;
    }

    __VEC_SCOPE__
#endif
{
    AscendC::Reg::RegTensor<float> data;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float>();
    uint32_t tailElements = tailRows * MHC_PRE_BASIC_API_C0_SIZE;
    AscendC::Reg::MaskReg tailMask = AscendC::Reg::UpdateMask<float>(tailElements);
    for (uint16_t rowBlock = 0; rowBlock < mainRowBlocks; ++rowBlock) {
        for (uint16_t columnBlock = 0; columnBlock < columnLoops; ++columnBlock) {
            AscendC::Reg::DataCopy<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
                data, src + rowBlock * MHC_PRE_BASIC_API_C0_SIZE * columnAlign + columnBlock * dataBlockElements,
                srcBlockStride, fullMask);
            AscendC::Reg::DataCopy(dst + rowBlock * MHC_PRE_BASIC_API_C0_SIZE * dataBlockElements +
                                       columnBlock * rowAlign * dataBlockElements,
                                   data, fullMask);
        }
    }
    src += mainRowBlocks * MHC_PRE_BASIC_API_C0_SIZE * columnAlign;
    dst += mainRowBlocks * MHC_PRE_BASIC_API_C0_SIZE * dataBlockElements;
    for (uint16_t columnBlock = 0; columnBlock < columnLoops; ++columnBlock) {
        AscendC::Reg::DataCopy<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
            data, src + columnBlock * dataBlockElements, srcBlockStride, tailMask);
        AscendC::Reg::DataCopy(dst + columnBlock * rowAlign * dataBlockElements, data, tailMask);
    }
}

#if defined(MHC_PRE_ENABLE_ASCEND950)
__aicore__ inline void MhcPreMkVFTransND2NZ(const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal,
                                            uint16_t rowCount, uint16_t columnCount)
{
    __local_mem__ float* dst = (__local_mem__ float*)dstLocal.GetPhyAddr();
    __local_mem__ float* src = (__local_mem__ float*)srcLocal.GetPhyAddr();
    uint16_t rowAlign = BasicApiAlign(rowCount, MHC_PRE_BASIC_API_C0_SIZE);
    uint16_t dataBlockElements = MHC_PRE_BASIC_API_BLOCK_SIZE / sizeof(float);
    uint16_t columnAlign = BasicApiRoundUp<float>(columnCount);
    uint16_t mainRowBlocks = rowCount / MHC_PRE_BASIC_API_C0_SIZE;
    uint16_t tailRows = rowCount % MHC_PRE_BASIC_API_C0_SIZE;
    uint16_t columnLoops = columnAlign / dataBlockElements;
    uint32_t srcBlockStride = columnAlign / dataBlockElements;

    if (tailRows == 0) {
        asc_vf_call<MhcPreMkVFTransND2NZAlignedImpl>(dst, src, rowAlign, dataBlockElements, columnAlign, mainRowBlocks,
                                                     columnLoops, srcBlockStride);
        return;
    }

    asc_vf_call<MhcPreMkVFTransND2NZTailImpl>(dst, src, rowAlign, dataBlockElements, columnAlign, mainRowBlocks,
                                              tailRows, columnLoops, srcBlockStride);
}

// The extra row in each NZ column block avoids UB bank conflicts; DMA skips it.
template <typename T, bool HAS_GAMMA>
__simd_vf__ inline void MhcPreMkVFPackOnlyNzImpl(uint16_t rows, uint16_t columns, __ubuf__ float* nz, __ubuf__ T* x,
                                                 uint16_t nzRows, uint16_t passes, __ubuf__ float* gamma)
{
    Reg::MaskReg full = Reg::CreateMask<float>();
    for (uint16_t row = 0U; row < rows; ++row) {
        __ubuf__ float* dst = nz + row * 8U;
        for (uint16_t pass = 0U; pass < passes; ++pass) {
            Reg::RegTensor<float> lo, hi;
            MhcPreMkLoadX128(lo, hi, x + row * columns + pass * 128U, full);
            if constexpr (HAS_GAMMA) {
                Reg::RegTensor<float> scale;
                Reg::LoadAlign(scale, gamma + pass * 128U);
                Reg::Mul(lo, lo, scale, full);
                Reg::LoadAlign(scale, gamma + pass * 128U + 64U);
                Reg::Mul(hi, hi, scale, full);
            }
            Reg::StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                dst, lo, nzRows, 8U * nzRows, full);
            Reg::StoreAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                dst, hi, nzRows, 8U * nzRows, full);
        }
    }
}

template <typename T>
__aicore__ inline void MhcPreMkVFPackOnlyNz(const LocalTensor<float>& nzLocal, const LocalTensor<T>& xLocal,
                                            uint16_t rows, uint16_t columns, __ubuf__ float* gamma)
{
    auto nz = (__ubuf__ float*)nzLocal.GetPhyAddr();
    auto x = (__ubuf__ T*)xLocal.GetPhyAddr();
    uint16_t nzRows = BasicApiAlign(rows, MHC_PRE_BASIC_API_C0_SIZE) + 1U;
    uint16_t passes = columns / 128U;
    if (gamma != nullptr) {
        asc_vf_call<MhcPreMkVFPackOnlyNzImpl<T, true>>(rows, columns, nz, x, nzRows, passes, gamma);
    } else {
        asc_vf_call<MhcPreMkVFPackOnlyNzImpl<T, false>>(rows, columns, nz, x, nzRows, passes, nullptr);
    }
}

__simd_vf__ inline void MhcPreMkVFReduceMmTree(uint16_t rowCount, uint16_t fusionSize, uint16_t segment, uint16_t width,
                                               __local_mem__ float* mmOut, __local_mem__ float* mmPartial,
                                               uint16_t fourLoopCount, uint16_t tailLoopCount, uint32_t groupStride)
{
    AscendC::Reg::RegTensor<float> mm0;
    AscendC::Reg::RegTensor<float> mm1;
    AscendC::Reg::RegTensor<float> mm2;
    AscendC::Reg::RegTensor<float> mm3;
    AscendC::Reg::RegTensor<float> sumMm0;
    AscendC::Reg::RegTensor<float> sumMm1;
    AscendC::Reg::RegTensor<float> sumMm2;
    AscendC::Reg::RegTensor<float> sumMm3;
    uint32_t mmMaskSize = width;
    AscendC::Reg::MaskReg mmMask = AscendC::Reg::UpdateMask<float>(mmMaskSize);

    for (uint16_t row = 0; row < rowCount; ++row) {
        AscendC::Reg::Duplicate(sumMm0, 0.0f, mmMask);
        AscendC::Reg::Duplicate(sumMm1, 0.0f, mmMask);
        AscendC::Reg::Duplicate(sumMm2, 0.0f, mmMask);
        AscendC::Reg::Duplicate(sumMm3, 0.0f, mmMask);

        for (uint16_t loop = 0; loop < fourLoopCount; ++loop) {
            uint16_t kBase = loop * 4U;

            AscendC::Reg::Load<float>(
                mm0, mmPartial + static_cast<uint32_t>(kBase) * groupStride + row * fusionSize + segment);
            AscendC::Reg::Load<float>(
                mm1, mmPartial + static_cast<uint32_t>(kBase + 1U) * groupStride + row * fusionSize + segment);
            AscendC::Reg::Load<float>(
                mm2, mmPartial + static_cast<uint32_t>(kBase + 2U) * groupStride + row * fusionSize + segment);
            AscendC::Reg::Load<float>(
                mm3, mmPartial + static_cast<uint32_t>(kBase + 3U) * groupStride + row * fusionSize + segment);
            AscendC::Reg::Add(sumMm0, sumMm0, mm0, mmMask);
            AscendC::Reg::Add(sumMm1, sumMm1, mm1, mmMask);
            AscendC::Reg::Add(sumMm2, sumMm2, mm2, mmMask);
            AscendC::Reg::Add(sumMm3, sumMm3, mm3, mmMask);
        }

        for (uint16_t tail = 0; tail < tailLoopCount; ++tail) {
            uint16_t kIndex = fourLoopCount * 4U + tail;
            AscendC::Reg::Load<float>(
                mm0, mmPartial + static_cast<uint32_t>(kIndex) * groupStride + row * fusionSize + segment);
            AscendC::Reg::Add(sumMm0, sumMm0, mm0, mmMask);
        }

        AscendC::Reg::Add(sumMm0, sumMm0, sumMm3, mmMask);
        AscendC::Reg::Add(sumMm1, sumMm1, sumMm2, mmMask);
        AscendC::Reg::Add(sumMm0, sumMm0, sumMm1, mmMask);
        AscendC::Reg::Store<float>(mmOut + row * fusionSize + segment, sumMm0, width);
    }
}

// Accumulate signed Cube partials with their rounding residual. TwoSum keeps the
// correction when terms have different signs and the running sum cancels.
__simd_callee__ inline void MhcPreMkAddProjectionPartial(Reg::RegTensor<float>& sum, Reg::RegTensor<float>& low,
                                                         Reg::RegTensor<float>& term, Reg::MaskReg& mask)
{
    Reg::RegTensor<float> next, roundedTerm, residual;
    Reg::Add(next, sum, term, mask);
    Reg::Sub(roundedTerm, next, sum, mask);
    Reg::Sub(residual, next, roundedTerm, mask);
    Reg::Sub(residual, sum, residual, mask);
    Reg::Sub(term, term, roundedTerm, mask);
    Reg::Add(residual, residual, term, mask);
    Reg::Add(low, low, residual, mask);
    sum = next;
}

__simd_callee__ inline void MhcPreMkFinishProjection(Reg::RegTensor<float>& sum, Reg::RegTensor<float>& low,
                                                     Reg::MaskReg& mask)
{
    Reg::MaskReg upper, finite;
    Reg::CompareScalar<float, CMPMODE::LE>(upper, sum, 3.402823466e38f, mask);
    Reg::CompareScalar<float, CMPMODE::GE>(finite, sum, -3.402823466e38f, upper);
    Reg::RegTensor<float> corrected;
    Reg::Add(corrected, sum, low, finite);
    Reg::Select(sum, corrected, sum, finite);
}

template <bool COMPENSATE>
__simd_vf__ inline void MhcPreMkVFReduceMmRows(uint16_t groupK, uint16_t rowCount, uint16_t fusionSize,
                                               uint16_t segment, uint16_t width, __local_mem__ float* mmOut,
                                               __local_mem__ float* mmPartial, uint32_t groupStride)
{
    AscendC::Reg::RegTensor<float> mm, sumMm;
    uint32_t maskSize = width;
    AscendC::Reg::MaskReg mmMask = AscendC::Reg::UpdateMask<float>(maskSize);
    for (uint16_t row = 0; row < rowCount; ++row) {
        AscendC::Reg::RegTensor<float> error;
        AscendC::Reg::Duplicate(sumMm, 0.0f, mmMask);
        if constexpr (COMPENSATE) {
            AscendC::Reg::Duplicate(error, 0.0f, mmMask);
        }
        for (uint16_t kIndex = 0; kIndex < groupK; ++kIndex) {
            AscendC::Reg::Load<float>(
                mm, mmPartial + static_cast<uint32_t>(kIndex) * groupStride + row * fusionSize + segment);
            if constexpr (COMPENSATE) {
                MhcPreMkAddProjectionPartial(sumMm, error, mm, mmMask);
            } else {
                AscendC::Reg::Add(sumMm, sumMm, mm, mmMask);
            }
        }
        if constexpr (COMPENSATE) {
            MhcPreMkFinishProjection(sumMm, error, mmMask);
        }
        AscendC::Reg::Store<float>(mmOut + row * fusionSize + segment, sumMm, width);
    }
}

template <bool COMPENSATE>
__simd_vf__ inline void MhcPreMkVFReduceMmFlat(uint16_t mmGroupK, __local_mem__ float* mmOut,
                                               __local_mem__ float* mmPartial, uint32_t mmElements,
                                               uint32_t mmGroupStride, uint16_t mmLoopCount)
{
    AscendC::Reg::RegTensor<float> mm;
    AscendC::Reg::RegTensor<float> sumMm;

    uint32_t remaining = mmElements;
    for (uint16_t loop = 0; loop < mmLoopCount; ++loop) {
        uint32_t currentCount = AscendC::Std::min(remaining, MHC_PRE_BASIC_API_VL_FP32);
        AscendC::Reg::MaskReg mmMask = AscendC::Reg::UpdateMask<float>(remaining);
        uint32_t vectorOffset = static_cast<uint32_t>(loop) * MHC_PRE_BASIC_API_VL_FP32;
        AscendC::Reg::RegTensor<float> error;
        AscendC::Reg::Duplicate(sumMm, 0.0f, mmMask);
        if constexpr (COMPENSATE) {
            AscendC::Reg::Duplicate(error, 0.0f, mmMask);
        }
        for (uint16_t kIndex = 0; kIndex < mmGroupK; ++kIndex) {
            AscendC::Reg::Load<float>(mm, mmPartial + static_cast<uint32_t>(kIndex) * mmGroupStride + vectorOffset);
            if constexpr (COMPENSATE) {
                MhcPreMkAddProjectionPartial(sumMm, error, mm, mmMask);
            } else {
                AscendC::Reg::Add(sumMm, sumMm, mm, mmMask);
            }
        }
        if constexpr (COMPENSATE) {
            MhcPreMkFinishProjection(sumMm, error, mmMask);
        }
        AscendC::Reg::Store<float>(mmOut + vectorOffset, sumMm, currentCount);
    }
}

__aicore__ inline void MhcPreMkReduceMm(const LocalTensor<float>& out, const LocalTensor<float>& partial,
                                        uint16_t groupK, uint16_t rows, uint16_t fusion, bool sequential, bool flat,
                                        bool compensate)
{
    auto dst = (__local_mem__ float*)out.GetPhyAddr();
    auto src = (__local_mem__ float*)partial.GetPhyAddr();
    uint32_t elements = static_cast<uint32_t>(rows) * fusion;
    uint32_t stride = BasicApiRoundUp<float>(elements);
    if (flat) {
        uint16_t loops = BasicApiCeilDiv(elements, MHC_PRE_BASIC_API_VL_FP32);
        if (compensate) {
            asc_vf_call<MhcPreMkVFReduceMmFlat<true>>(groupK, dst, src, elements, stride, loops);
        } else {
            asc_vf_call<MhcPreMkVFReduceMmFlat<false>>(groupK, dst, src, elements, stride, loops);
        }
        return;
    }
    for (uint16_t segment = 0; segment < fusion; segment += MHC_PRE_BASIC_API_VL_FP32) {
        uint16_t width = AscendC::Std::min<uint32_t>(fusion - segment, MHC_PRE_BASIC_API_VL_FP32);
        if (sequential) {
            if (compensate) {
                asc_vf_call<MhcPreMkVFReduceMmRows<true>>(groupK, rows, fusion, segment, width, dst, src, stride);
            } else {
                asc_vf_call<MhcPreMkVFReduceMmRows<false>>(groupK, rows, fusion, segment, width, dst, src, stride);
            }
        } else {
            asc_vf_call<MhcPreMkVFReduceMmTree>(rows, fusion, segment, width, dst, src, groupK / 4U, groupK % 4U,
                                                stride);
        }
    }
}

// Each projection producer writes the square sum of an aligned K256 tile.
__aicore__ inline void MhcPreMkCopyRmsPartials(const LocalTensor<float>& dst, const GlobalTensor<float>& src,
                                               uint16_t groups, uint16_t rows, uint32_t totalRows)
{
    DataCopyExtParams copy{groups, rows * static_cast<uint32_t>(sizeof(float)),
                           (totalRows - rows) * static_cast<uint32_t>(sizeof(float)), 0U, 0U};
    DataCopyPadExtParams<float> pad{true, 0U, static_cast<uint8_t>(BasicApiRoundUp<float>(rows) - rows), 0U};
    DataCopyPad(dst, src, copy, pad);
}

__simd_vf__ inline void MhcPreMkVFReduceRms(__ubuf__ float* out, __ubuf__ float* partials, uint16_t groups,
                                            uint32_t rows, uint32_t stride, float scaleMean, float normEps)
{
    Reg::RegTensor<float> sum, error, partial;
    Reg::RegTensor<uint32_t> indices;
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(indices), 0);
    uint32_t count = rows;
    Reg::MaskReg mask = Reg::UpdateMask<float>(count);
    Reg::Duplicate(sum, 0.0f, mask);
    Reg::Duplicate(error, 0.0f, mask);
    for (uint16_t k = 0; k < groups; ++k) {
        Reg::Gather(partial, partials + k * stride, indices, mask);
        MhcPreCompensatedAdd(sum, error, partial, mask);
    }
    MhcPreNormalizeRms(sum, mask, scaleMean, normEps);
    Reg::Store(out, sum, rows);
}

__aicore__ inline void MhcPreMkReduceRms(const LocalTensor<float>& out, const LocalTensor<float>& partial,
                                         uint16_t groups, uint16_t rows, float scaleMean, float normEps)
{
    asc_vf_call<MhcPreMkVFReduceRms>((__ubuf__ float*)out.GetPhyAddr(), (__ubuf__ float*)partial.GetPhyAddr(), groups,
                                     rows, BasicApiRoundUp<float>(rows), scaleMean, normEps);
}

__simd_vf__ inline void MhcPreMkReduceOrdinaryRms(__ubuf__ float* out, __ubuf__ float* partials, uint16_t kGroups,
                                                  uint32_t rows, uint32_t stride, float kSize, float scaleMean,
                                                  float normEps)
{
    Reg::RegTensor<uint32_t> indices;
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(indices), 0);
    uint32_t count = rows;
    Reg::MaskReg mask = Reg::UpdateMask<float>(count);
    Reg::RegTensor<float> sum, sumResidual;
    Reg::Duplicate(sum, 0.0f, mask);
    Reg::Duplicate(sumResidual, 0.0f, mask);
    for (uint16_t k = 0; k < kGroups; ++k) {
        Reg::RegTensor<float> partial, correction, next, roundedPartial, residual;
        Reg::Gather(partial, partials + k * stride, indices, mask);
        Reg::Gather(correction, partials + (kGroups + k) * stride, indices, mask);
        // TwoSum recovers the rounding discarded when the next K shard is added.
        Reg::Add(next, sum, partial, mask);
        Reg::Sub(roundedPartial, next, sum, mask);
        Reg::Sub(residual, next, roundedPartial, mask);
        Reg::Sub(residual, sum, residual, mask);
        Reg::Sub(partial, partial, roundedPartial, mask);
        Reg::Add(residual, residual, partial, mask);
        Reg::Sub(residual, residual, correction, mask);
        Reg::Add(sumResidual, sumResidual, residual, mask);
        sum = next;
    }

    Reg::RegTensor<float> scale, bias, k, negativeBias, epsilonK;
    Reg::Duplicate(scale, scaleMean, mask);
    Reg::Duplicate(bias, normEps, mask);
    Reg::Duplicate(k, kSize, mask);
    Reg::Duplicate(epsilonK, normEps * kSize, mask);
    Reg::MulAddDst(bias, sum, scale, mask);
    // Recover the error in bias = sum / K + eps with the exact integer K.
    Reg::Muls(negativeBias, bias, -1.0f, mask);
    Reg::MulAddDst(sum, negativeBias, k, mask);
    Reg::Add(sum, sum, epsilonK, mask);
    Reg::Add(sum, sum, sumResidual, mask);
    Reg::Mul(sumResidual, sum, scale, mask);
    MhcPreInvSqrt<true>(sum, bias, sumResidual, mask);
    Reg::Store(out, sum, rows);
}

template <class T, class P, int8_t RESI_MODE, bool WEIGHT_REUSE>
#else
}

template <typename T, bool INIT_RMS>
__aicore__ inline void MhcPreMkVFProcessXInGammaLoops(const LocalTensor<float>& xOutLocal,
                                                      const LocalTensor<float>& rmsLocal,
                                                      const LocalTensor<T>& xInLocal,
                                                      const LocalTensor<float>& gammaLocal, uint16_t rowCount,
                                                      uint16_t columnCount, uint16_t loopStart, uint16_t processLoops)
{
    __local_mem__ float* xOut = (__local_mem__ float*)xOutLocal.GetPhyAddr();
    __local_mem__ float* rms = (__local_mem__ float*)rmsLocal.GetPhyAddr();
    __local_mem__ T* xIn = (__local_mem__ T*)xInLocal.GetPhyAddr();
    __local_mem__ float* gamma = (__local_mem__ float*)gammaLocal.GetPhyAddr();
    uint32_t srcStride = BasicApiRoundUp<T>(columnCount);
    uint32_t dstStride = BasicApiRoundUp<float>(columnCount);
    uint16_t loopEnd = loopStart + processLoops;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> xInReg;
        AscendC::Reg::RegTensor<float> gammaReg;
        AscendC::Reg::RegTensor<float> xReg;
        AscendC::Reg::RegTensor<float> xGammaReg;
        AscendC::Reg::RegTensor<float> squareReg;
        AscendC::Reg::RegTensor<float> partialReg;
        AscendC::Reg::RegTensor<float> sumReg;
        uint32_t remaining = columnCount - static_cast<uint32_t>(loopStart) * MHC_PRE_BASIC_API_VL_FP32;
        for (uint16_t loop = loopStart; loop < loopEnd; ++loop) {
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
            uint32_t vectorOffset = static_cast<uint32_t>(loop) * MHC_PRE_BASIC_API_VL_FP32;
            AscendC::Reg::LoadAlign(gammaReg, gamma + vectorOffset);
            for (uint16_t row = 0; row < rowCount; ++row) {
                uint32_t srcOffset = static_cast<uint32_t>(row) * srcStride + vectorOffset;
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xInReg, xIn + srcOffset);
                AscendC::Reg::Cast<float, T, MHC_PRE_BASIC_API_CAST_B16_TO_B32>(xReg, xInReg, mask);
                AscendC::Reg::Mul(xGammaReg, gammaReg, xReg, mask);
                uint32_t dstOffset = static_cast<uint32_t>(row) * dstStride + vectorOffset;
                AscendC::Reg::StoreAlign(xOut + dstOffset, xGammaReg, mask);
                AscendC::Reg::Mul(squareReg, xReg, xReg, mask);
                AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(partialReg, squareReg, mask);
                if constexpr (INIT_RMS) {
                    AscendC::Reg::Duplicate(sumReg, 0.0f);
                } else {
                    AscendC::Reg::Load(sumReg, rms + row);
                }
                AscendC::Reg::Add(sumReg, sumReg, partialReg, mask);
                AscendC::Reg::Store(rms + row, sumReg, 1U);
            }
        }
    }
}
template <typename T, bool IS_FIRST_K>
__aicore__ inline void MhcPreMkVFProcessXInGammaSmallRows(const LocalTensor<float>& xOutLocal,
                                                          const LocalTensor<float>& rmsLocal,
                                                          const LocalTensor<T>& xInLocal,
                                                          const LocalTensor<float>& gammaLocal, uint16_t rowCount,
                                                          uint16_t columnCount)
{
    __local_mem__ float* xOut = (__local_mem__ float*)xOutLocal.GetPhyAddr();
    __local_mem__ float* rms = (__local_mem__ float*)rmsLocal.GetPhyAddr();
    __local_mem__ T* xIn = (__local_mem__ T*)xInLocal.GetPhyAddr();
    __local_mem__ float* gamma = (__local_mem__ float*)gammaLocal.GetPhyAddr();
    uint32_t srcStride = BasicApiRoundUp<T>(columnCount);
    uint32_t dstStride = BasicApiRoundUp<float>(columnCount);
    uint16_t loopCount = BasicApiCeilDiv(columnCount, MHC_PRE_BASIC_API_VL_FP32);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> xInReg;
        AscendC::Reg::RegTensor<float> gammaReg;
        AscendC::Reg::RegTensor<float> xReg;
        AscendC::Reg::RegTensor<float> xGammaReg;
        AscendC::Reg::RegTensor<float> squareReg;
        AscendC::Reg::RegTensor<float> partialReg;
        AscendC::Reg::RegTensor<float> sumReg;
        for (uint16_t row = 0; row < rowCount; ++row) {
            if constexpr (IS_FIRST_K) {
                AscendC::Reg::Duplicate(sumReg, 0.0f);
            } else {
                AscendC::Reg::Load(sumReg, rms + row);
            }
            uint32_t remaining = columnCount;
            for (uint16_t loop = 0; loop < loopCount; ++loop) {
                AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
                uint32_t vectorOffset = static_cast<uint32_t>(loop) * MHC_PRE_BASIC_API_VL_FP32;
                AscendC::Reg::LoadAlign(gammaReg, gamma + vectorOffset);
                uint32_t srcOffset = static_cast<uint32_t>(row) * srcStride + vectorOffset;
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xInReg, xIn + srcOffset);
                AscendC::Reg::Cast<float, T, MHC_PRE_BASIC_API_CAST_B16_TO_B32>(xReg, xInReg, mask);
                AscendC::Reg::Mul(xGammaReg, gammaReg, xReg, mask);
                uint32_t dstOffset = static_cast<uint32_t>(row) * dstStride + vectorOffset;
                AscendC::Reg::StoreAlign(xOut + dstOffset, xGammaReg, mask);
                AscendC::Reg::Mul(squareReg, xReg, xReg, mask);
                AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(partialReg, squareReg, mask);
                AscendC::Reg::Add(sumReg, sumReg, partialReg, mask);
            }
            AscendC::Reg::Store(rms + row, sumReg, 1U);
        }
    }
}

template <typename T, bool IS_FIRST_K>
__aicore__ inline void MhcPreMkVFProcessXInGammaFull128(const LocalTensor<float>& xOutLocal,
                                                        const LocalTensor<float>& rmsLocal,
                                                        const LocalTensor<T>& xInLocal,
                                                        const LocalTensor<float>& gammaLocal, uint16_t rowCount)
{
    __local_mem__ float* xOut = (__local_mem__ float*)xOutLocal.GetPhyAddr();
    __local_mem__ float* rms = (__local_mem__ float*)rmsLocal.GetPhyAddr();
    __local_mem__ T* xIn = (__local_mem__ T*)xInLocal.GetPhyAddr();
    __local_mem__ float* gamma = (__local_mem__ float*)gammaLocal.GetPhyAddr();
    constexpr uint32_t columnCount = 128U;
    constexpr uint32_t vectorOffset = MHC_PRE_BASIC_API_VL_FP32;
    constexpr uint32_t srcStride = columnCount;
    constexpr uint32_t dstStride = columnCount;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> xInReg;
        AscendC::Reg::RegTensor<float> gamma0Reg;
        AscendC::Reg::RegTensor<float> gamma1Reg;
        AscendC::Reg::RegTensor<float> xReg;
        AscendC::Reg::RegTensor<float> xGammaReg;
        AscendC::Reg::RegTensor<float> squareReg;
        AscendC::Reg::RegTensor<float> squareSumReg;
        AscendC::Reg::RegTensor<float> partialReg;
        AscendC::Reg::RegTensor<float> sumReg;
        AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();

        AscendC::Reg::LoadAlign(gamma0Reg, gamma);
        AscendC::Reg::LoadAlign(gamma1Reg, gamma + vectorOffset);
        for (uint16_t row = 0; row < rowCount; ++row) {
            uint32_t rowSrcOffset = static_cast<uint32_t>(row) * srcStride;
            uint32_t rowDstOffset = static_cast<uint32_t>(row) * dstStride;
            AscendC::Reg::Duplicate(squareSumReg, 0.0f);

            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xInReg, xIn + rowSrcOffset);
            AscendC::Reg::Cast<float, T, MHC_PRE_BASIC_API_CAST_B16_TO_B32>(xReg, xInReg, mask);
            AscendC::Reg::Mul(xGammaReg, gamma0Reg, xReg, mask);
            AscendC::Reg::StoreAlign(xOut + rowDstOffset, xGammaReg, mask);
            AscendC::Reg::Mul(squareReg, xReg, xReg, mask);
            AscendC::Reg::Add(squareSumReg, squareSumReg, squareReg, mask);

            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xInReg,
                                                                                xIn + rowSrcOffset + vectorOffset);
            AscendC::Reg::Cast<float, T, MHC_PRE_BASIC_API_CAST_B16_TO_B32>(xReg, xInReg, mask);
            AscendC::Reg::Mul(xGammaReg, gamma1Reg, xReg, mask);
            AscendC::Reg::StoreAlign(xOut + rowDstOffset + vectorOffset, xGammaReg, mask);
            AscendC::Reg::Mul(squareReg, xReg, xReg, mask);
            AscendC::Reg::Add(squareSumReg, squareSumReg, squareReg, mask);

            AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(partialReg, squareSumReg, mask);
            if constexpr (IS_FIRST_K) {
                AscendC::Reg::Store(rms + row, partialReg, 1U);
            } else {
                AscendC::Reg::Load(sumReg, rms + row);
                AscendC::Reg::Add(sumReg, sumReg, partialReg, mask);
                AscendC::Reg::Store(rms + row, sumReg, 1U);
            }
        }
    }
}

template <typename T, bool IS_FIRST_K>
__aicore__ inline void MhcPreMkVFProcessXInGamma(const LocalTensor<float>& xOutLocal,
                                                 const LocalTensor<float>& rmsLocal, const LocalTensor<T>& xInLocal,
                                                 const LocalTensor<float>& gammaLocal, uint16_t rowCount,
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
__aicore__ inline void MhcPreMkVFReducePartials(const LocalTensor<float>& mmOutLocal,
                                                const LocalTensor<float>& invRmsOutLocal,
                                                const LocalTensor<float>& mmPartialLocal,

                                                const LocalTensor<float>& rmsPartialLocal, uint16_t groupK,
                                                uint16_t rowCount, uint16_t fusionSize, uint16_t mmSegmentOffset,
                                                uint16_t mmSegmentSize, uint16_t rmsStride, float scaleMean,
                                                float normEps)
{
    __local_mem__ float* mmOut = (__local_mem__ float*)mmOutLocal.GetPhyAddr();
    __local_mem__ float* invRmsOut = (__local_mem__ float*)invRmsOutLocal.GetPhyAddr();
    __local_mem__ float* mmPartial = (__local_mem__ float*)mmPartialLocal.GetPhyAddr();
    __local_mem__ float* rmsPartial = (__local_mem__ float*)rmsPartialLocal.GetPhyAddr();
    uint16_t fourLoopCount = groupK / 4U;
    uint16_t tailLoopCount = groupK % 4U;
    uint32_t mmGroupStride = BasicApiRoundUp<float>(static_cast<uint32_t>(rowCount) * fusionSize);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> rms0;
        AscendC::Reg::RegTensor<float> rms1;
        AscendC::Reg::RegTensor<float> rms2;
        AscendC::Reg::RegTensor<float> rms3;
        AscendC::Reg::RegTensor<float> mm0;
        AscendC::Reg::RegTensor<float> mm1;
        AscendC::Reg::RegTensor<float> mm2;
        AscendC::Reg::RegTensor<float> mm3;
        AscendC::Reg::RegTensor<float> sumRms0;
        AscendC::Reg::RegTensor<float> sumRms1;
        AscendC::Reg::RegTensor<float> sumRms2;
        AscendC::Reg::RegTensor<float> sumRms3;
        AscendC::Reg::RegTensor<float> sumMm0;
        AscendC::Reg::RegTensor<float> sumMm1;
        AscendC::Reg::RegTensor<float> sumMm2;
        AscendC::Reg::RegTensor<float> sumMm3;
        AscendC::Reg::RegTensor<float> one;
        uint32_t rmsMaskSize = 1U;
        uint32_t mmMaskSize = mmSegmentSize;
        AscendC::Reg::MaskReg rmsMask = AscendC::Reg::UpdateMask<float>(rmsMaskSize);
        AscendC::Reg::MaskReg mmMask = AscendC::Reg::UpdateMask<float>(mmMaskSize);
        AscendC::Reg::Duplicate(one, 1.0f, rmsMask);

        for (uint16_t row = 0; row < rowCount; ++row) {
            AscendC::Reg::Duplicate(sumRms0, 0.0f, rmsMask);
            AscendC::Reg::Duplicate(sumRms1, 0.0f, rmsMask);
            AscendC::Reg::Duplicate(sumRms2, 0.0f, rmsMask);
            AscendC::Reg::Duplicate(sumRms3, 0.0f, rmsMask);
            AscendC::Reg::Duplicate(sumMm0, 0.0f, mmMask);
            AscendC::Reg::Duplicate(sumMm1, 0.0f, mmMask);
            AscendC::Reg::Duplicate(sumMm2, 0.0f, mmMask);
            AscendC::Reg::Duplicate(sumMm3, 0.0f, mmMask);

            for (uint16_t loop = 0; loop < fourLoopCount; ++loop) {
                uint16_t kBase = loop * 4U;
                MhcPreBasicApiLoadBroadcast(rms0, rmsPartial, rmsMask, static_cast<uint32_t>(kBase) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms1, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 1U) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms2, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 2U) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms3, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 3U) * rmsStride + row);
                AscendC::Reg::Add(sumRms0, sumRms0, rms0, rmsMask);
                AscendC::Reg::Add(sumRms1, sumRms1, rms1, rmsMask);
                AscendC::Reg::Add(sumRms2, sumRms2, rms2, rmsMask);
                AscendC::Reg::Add(sumRms3, sumRms3, rms3, rmsMask);

                AscendC::Reg::Load<float>(
                    mm0, mmPartial + static_cast<uint32_t>(kBase) * mmGroupStride + row * fusionSize + mmSegmentOffset);
                AscendC::Reg::Load<float>(mm1, mmPartial + static_cast<uint32_t>(kBase + 1U) * mmGroupStride +
                                                   row * fusionSize + mmSegmentOffset);
                AscendC::Reg::Load<float>(mm2, mmPartial + static_cast<uint32_t>(kBase + 2U) * mmGroupStride +
                                                   row * fusionSize + mmSegmentOffset);
                AscendC::Reg::Load<float>(mm3, mmPartial + static_cast<uint32_t>(kBase + 3U) * mmGroupStride +
                                                   row * fusionSize + mmSegmentOffset);
                AscendC::Reg::Add(sumMm0, sumMm0, mm0, mmMask);
                AscendC::Reg::Add(sumMm1, sumMm1, mm1, mmMask);
                AscendC::Reg::Add(sumMm2, sumMm2, mm2, mmMask);
                AscendC::Reg::Add(sumMm3, sumMm3, mm3, mmMask);
            }

            for (uint16_t tail = 0; tail < tailLoopCount; ++tail) {
                uint16_t kIndex = fourLoopCount * 4U + tail;
                MhcPreBasicApiLoadBroadcast(rms0, rmsPartial, rmsMask, static_cast<uint32_t>(kIndex) * rmsStride + row);
                AscendC::Reg::Add(sumRms0, sumRms0, rms0, rmsMask);
                AscendC::Reg::Load<float>(mm0, mmPartial + static_cast<uint32_t>(kIndex) * mmGroupStride +
                                                   row * fusionSize + mmSegmentOffset);
                AscendC::Reg::Add(sumMm0, sumMm0, mm0, mmMask);
            }

            AscendC::Reg::Add(sumRms0, sumRms0, sumRms3, rmsMask);
            AscendC::Reg::Add(sumRms1, sumRms1, sumRms2, rmsMask);
            AscendC::Reg::Add(sumRms0, sumRms0, sumRms1, rmsMask);
            AscendC::Reg::Muls(sumRms0, sumRms0, scaleMean, rmsMask);
            AscendC::Reg::Adds(sumRms0, sumRms0, normEps, rmsMask);
            AscendC::Reg::Sqrt(sumRms0, sumRms0, rmsMask);
            AscendC::Reg::Div(sumRms0, one, sumRms0, rmsMask);

            AscendC::Reg::Add(sumMm0, sumMm0, sumMm3, mmMask);
            AscendC::Reg::Add(sumMm1, sumMm1, sumMm2, mmMask);
            AscendC::Reg::Add(sumMm0, sumMm0, sumMm1, mmMask);
            AscendC::Reg::Store<float>(invRmsOut + row, sumRms0, 1U);
            AscendC::Reg::Store<float>(mmOut + row * fusionSize + mmSegmentOffset, sumMm0, mmSegmentSize);
        }
    }
}

__aicore__ inline void MhcPreMkVFReducePartialsSequentialGeneral(
    const LocalTensor<float>& mmOutLocal, const LocalTensor<float>& invRmsOutLocal,
    const LocalTensor<float>& mmPartialLocal, const LocalTensor<float>& rmsPartialLocal, uint16_t mmGroupK,
    uint16_t rmsGroupK, uint16_t rowCount, uint16_t fusionSize, uint16_t mmSegmentOffset, uint16_t mmSegmentSize,
    uint16_t rmsStride, float scaleMean, float normEps)
{
    __local_mem__ float* mmOut = (__local_mem__ float*)mmOutLocal.GetPhyAddr();
    __local_mem__ float* invRmsOut = (__local_mem__ float*)invRmsOutLocal.GetPhyAddr();
    __local_mem__ float* mmPartial = (__local_mem__ float*)mmPartialLocal.GetPhyAddr();
    __local_mem__ float* rmsPartial = (__local_mem__ float*)rmsPartialLocal.GetPhyAddr();
    uint16_t fourLoopCount = rmsGroupK / 4U;
    uint16_t tailLoopCount = rmsGroupK % 4U;
    uint32_t mmGroupStride = BasicApiRoundUp<float>(static_cast<uint32_t>(rowCount) * fusionSize);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> rms0;
        AscendC::Reg::RegTensor<float> rms1;
        AscendC::Reg::RegTensor<float> rms2;
        AscendC::Reg::RegTensor<float> rms3;
        AscendC::Reg::RegTensor<float> sumRms0;
        AscendC::Reg::RegTensor<float> sumRms1;
        AscendC::Reg::RegTensor<float> sumRms2;
        AscendC::Reg::RegTensor<float> sumRms3;
        AscendC::Reg::RegTensor<float> mm;
        AscendC::Reg::RegTensor<float> sumMm;
        AscendC::Reg::RegTensor<float> one;
        uint32_t rmsMaskSize = 1U;
        uint32_t mmMaskSize = mmSegmentSize;
        AscendC::Reg::MaskReg rmsMask = AscendC::Reg::UpdateMask<float>(rmsMaskSize);
        AscendC::Reg::MaskReg mmMask = AscendC::Reg::UpdateMask<float>(mmMaskSize);
        AscendC::Reg::Duplicate(one, 1.0f, rmsMask);

        for (uint16_t row = 0; row < rowCount; ++row) {
            AscendC::Reg::Duplicate(sumRms0, 0.0f, rmsMask);
            AscendC::Reg::Duplicate(sumRms1, 0.0f, rmsMask);
            AscendC::Reg::Duplicate(sumRms2, 0.0f, rmsMask);
            AscendC::Reg::Duplicate(sumRms3, 0.0f, rmsMask);
            for (uint16_t loop = 0; loop < fourLoopCount; ++loop) {
                uint16_t kBase = loop * 4U;
                MhcPreBasicApiLoadBroadcast(rms0, rmsPartial, rmsMask, static_cast<uint32_t>(kBase) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms1, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 1U) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms2, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 2U) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms3, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 3U) * rmsStride + row);
                AscendC::Reg::Add(sumRms0, sumRms0, rms0, rmsMask);
                AscendC::Reg::Add(sumRms1, sumRms1, rms1, rmsMask);
                AscendC::Reg::Add(sumRms2, sumRms2, rms2, rmsMask);
                AscendC::Reg::Add(sumRms3, sumRms3, rms3, rmsMask);
            }
            for (uint16_t tail = 0; tail < tailLoopCount; ++tail) {
                uint16_t kIndex = fourLoopCount * 4U + tail;
                MhcPreBasicApiLoadBroadcast(rms0, rmsPartial, rmsMask, static_cast<uint32_t>(kIndex) * rmsStride + row);
                AscendC::Reg::Add(sumRms0, sumRms0, rms0, rmsMask);
            }
            AscendC::Reg::Add(sumRms0, sumRms0, sumRms3, rmsMask);
            AscendC::Reg::Add(sumRms1, sumRms1, sumRms2, rmsMask);
            AscendC::Reg::Add(sumRms0, sumRms0, sumRms1, rmsMask);
            AscendC::Reg::Muls(sumRms0, sumRms0, scaleMean, rmsMask);
            AscendC::Reg::Adds(sumRms0, sumRms0, normEps, rmsMask);
            AscendC::Reg::Sqrt(sumRms0, sumRms0, rmsMask);
            AscendC::Reg::Div(sumRms0, one, sumRms0, rmsMask);

            AscendC::Reg::Duplicate(sumMm, 0.0f, mmMask);
            for (uint16_t kIndex = 0; kIndex < mmGroupK; ++kIndex) {
                AscendC::Reg::Load<float>(
                    mm, mmPartial + static_cast<uint32_t>(kIndex) * mmGroupStride + row * fusionSize + mmSegmentOffset);
                AscendC::Reg::Add(sumMm, sumMm, mm, mmMask);
            }
            AscendC::Reg::Store<float>(invRmsOut + row, sumRms0, 1U);
            AscendC::Reg::Store<float>(mmOut + row * fusionSize + mmSegmentOffset, sumMm, mmSegmentSize);
        }
    }
}

__aicore__ inline void MhcPreMkVFReducePartialsSequentialFlat(
    const LocalTensor<float>& mmOutLocal, const LocalTensor<float>& invRmsOutLocal,
    const LocalTensor<float>& mmPartialLocal, const LocalTensor<float>& rmsPartialLocal, uint16_t mmGroupK,
    uint16_t rmsGroupK, uint16_t rowCount, uint16_t fusionSize, uint16_t rmsStride, float scaleMean, float normEps)
{
    __local_mem__ float* mmOut = (__local_mem__ float*)mmOutLocal.GetPhyAddr();
    __local_mem__ float* invRmsOut = (__local_mem__ float*)invRmsOutLocal.GetPhyAddr();
    __local_mem__ float* mmPartial = (__local_mem__ float*)mmPartialLocal.GetPhyAddr();
    __local_mem__ float* rmsPartial = (__local_mem__ float*)rmsPartialLocal.GetPhyAddr();
    uint16_t fourLoopCount = rmsGroupK / 4U;
    uint16_t tailLoopCount = rmsGroupK % 4U;
    uint32_t mmElements = static_cast<uint32_t>(rowCount) * fusionSize;
    uint32_t mmGroupStride = BasicApiRoundUp<float>(mmElements);
    uint16_t mmLoopCount = BasicApiCeilDiv(mmElements, MHC_PRE_BASIC_API_VL_FP32);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> rms0;
        AscendC::Reg::RegTensor<float> rms1;
        AscendC::Reg::RegTensor<float> rms2;
        AscendC::Reg::RegTensor<float> rms3;
        AscendC::Reg::RegTensor<float> sumRms0;
        AscendC::Reg::RegTensor<float> sumRms1;
        AscendC::Reg::RegTensor<float> sumRms2;
        AscendC::Reg::RegTensor<float> sumRms3;
        AscendC::Reg::RegTensor<float> mm;
        AscendC::Reg::RegTensor<float> sumMm;
        AscendC::Reg::RegTensor<float> one;
        uint32_t rmsMaskSize = 1U;
        AscendC::Reg::MaskReg rmsMask = AscendC::Reg::UpdateMask<float>(rmsMaskSize);
        AscendC::Reg::Duplicate(one, 1.0f, rmsMask);

        for (uint16_t row = 0; row < rowCount; ++row) {
            AscendC::Reg::Duplicate(sumRms0, 0.0f, rmsMask);
            AscendC::Reg::Duplicate(sumRms1, 0.0f, rmsMask);
            AscendC::Reg::Duplicate(sumRms2, 0.0f, rmsMask);
            AscendC::Reg::Duplicate(sumRms3, 0.0f, rmsMask);
            for (uint16_t loop = 0; loop < fourLoopCount; ++loop) {
                uint16_t kBase = loop * 4U;
                MhcPreBasicApiLoadBroadcast(rms0, rmsPartial, rmsMask, static_cast<uint32_t>(kBase) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms1, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 1U) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms2, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 2U) * rmsStride + row);
                MhcPreBasicApiLoadBroadcast(rms3, rmsPartial, rmsMask,
                                            static_cast<uint32_t>(kBase + 3U) * rmsStride + row);
                AscendC::Reg::Add(sumRms0, sumRms0, rms0, rmsMask);
                AscendC::Reg::Add(sumRms1, sumRms1, rms1, rmsMask);
                AscendC::Reg::Add(sumRms2, sumRms2, rms2, rmsMask);
                AscendC::Reg::Add(sumRms3, sumRms3, rms3, rmsMask);
            }
            for (uint16_t tail = 0; tail < tailLoopCount; ++tail) {
                uint16_t kIndex = fourLoopCount * 4U + tail;
                MhcPreBasicApiLoadBroadcast(rms0, rmsPartial, rmsMask, static_cast<uint32_t>(kIndex) * rmsStride + row);
                AscendC::Reg::Add(sumRms0, sumRms0, rms0, rmsMask);
            }
            AscendC::Reg::Add(sumRms0, sumRms0, sumRms3, rmsMask);
            AscendC::Reg::Add(sumRms1, sumRms1, sumRms2, rmsMask);
            AscendC::Reg::Add(sumRms0, sumRms0, sumRms1, rmsMask);
            AscendC::Reg::Muls(sumRms0, sumRms0, scaleMean, rmsMask);
            AscendC::Reg::Adds(sumRms0, sumRms0, normEps, rmsMask);
            AscendC::Reg::Sqrt(sumRms0, sumRms0, rmsMask);
            AscendC::Reg::Div(sumRms0, one, sumRms0, rmsMask);
            AscendC::Reg::Store<float>(invRmsOut + row, sumRms0, 1U);
        }

        uint32_t remaining = mmElements;
        for (uint16_t loop = 0; loop < mmLoopCount; ++loop) {
            uint32_t currentCount = AscendC::Std::min(remaining, MHC_PRE_BASIC_API_VL_FP32);
            AscendC::Reg::MaskReg mmMask = AscendC::Reg::UpdateMask<float>(remaining);
            uint32_t vectorOffset = static_cast<uint32_t>(loop) * MHC_PRE_BASIC_API_VL_FP32;
            AscendC::Reg::Duplicate(sumMm, 0.0f, mmMask);
            for (uint16_t kIndex = 0; kIndex < mmGroupK; ++kIndex) {
                AscendC::Reg::Load<float>(mm, mmPartial + static_cast<uint32_t>(kIndex) * mmGroupStride + vectorOffset);
                AscendC::Reg::Add(sumMm, sumMm, mm, mmMask);
            }
            AscendC::Reg::Store<float>(mmOut + vectorOffset, sumMm, currentCount);
        }
    }
}

template <class T, class P, int8_t RESI_MODE>
#endif
class MhcPreMKPart1 {
public:
    __aicore__ inline MhcPreMKPart1() = default;

#if defined(MHC_PRE_ENABLE_ASCEND950)
    __aicore__ inline uint32_t WaveChunkD() const
    {
        return vector_.tiling_->v1ChunkDSize;
    }
    __aicore__ inline uint32_t WaveDChunks() const
    {
        return BasicApiCeilDiv(vector_.D_, WaveChunkD());
    }
    __aicore__ inline uint32_t WaveChunkLength(uint32_t chunk) const
    {
        return AscendC::Std::min(WaveChunkD(), static_cast<uint32_t>(vector_.D_) - chunk * WaveChunkD());
    }
    __aicore__ inline uint32_t WaveStride() const
    {
        return vector_.tiling_->cubeBlockDimM * 128U;
    }
    __aicore__ inline uint32_t WaveRows() const
    {
        return 64U / vector_.tiling_->cubeBlockDimK;
    }
    __aicore__ inline uint32_t WaveBatchRows() const
    {
        return AscendC::Std::min(WaveRows(), 4U);
    }
    __aicore__ inline uint32_t WavePartialK() const
    {
        return MhcPreMkWeightReusePartialK(vector_.totalLength_, vector_.tiling_->multCoreSplitKSize, true, true);
    }
    __aicore__ inline uint32_t WaveMmGroups() const
    {
        return vector_.tiling_->cubeBlockDimK * BasicApiCeilDiv(vector_.tiling_->multCoreSplitKSize, WavePartialK());
    }
    __aicore__ inline bool IsWaveCase() const
    {
        return WEIGHT_REUSE && vector_.tiling_->mkUseGmStage == MHC_PRE_MK_WEIGHT_REUSE_WAVE_STAGE;
    }

    static constexpr uint32_t kRmsRowsPerTile = 64U;

    __aicore__ inline uint32_t RmsGroups() const
    {
        return vector_.matrixInfo_.nD / MHC_PRE_RMS_K_BLOCK;
    }

    __aicore__ inline uint32_t WaveMmPartialElements() const
    {
        return WaveMmGroups() * WaveBatchRows() * 24U;
    }

    __aicore__ inline void StoreRmsPartial(const LocalTensor<P>& rms, uint32_t rows, uint64_t row, uint32_t k)
    {
        DataCopyExtParams copy{1U, rows * static_cast<uint32_t>(sizeof(P)), 0U, 0U, 0U};
        uint64_t group = k / MHC_PRE_RMS_K_BLOCK;
        DataCopyPad(partialRmsGm_[group * vector_.totalLength_ + row], rms, copy);
    }

    __aicore__ inline void InitWavePost(InitParams params)
    {
        post_.BindGlobalTensors(params);
        post_.InitFromTilingData(params.tilingData);
        post_.InitMNConfig();
        post_.pipe_ = params.tPipeIn;
        post_.curSingleT_ = WaveBatchRows();
        post_.hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(params.h_mix));
        post_.xGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_PERSISTENT);
        post_.pipe_->InitBuffer(post_.xInQueue_, 2U, 4U * WaveChunkD() * static_cast<uint32_t>(sizeof(T)));
        post_.pipe_->InitBuffer(post_.outQueue_, 2U, WaveChunkD() * static_cast<uint32_t>(sizeof(T)));
        post_.pipe_->InitBuffer(post_.invRmsOutQueue_, 1U, 32U);
        post_.pipe_->InitBuffer(post_.biasInQue_, 1U, 96U);
        post_.pipe_->InitBuffer(post_.alphaBuf_, 96U);
        post_.alphaInUb_ = post_.alphaBuf_.template Get<P>();
        post_.pipe_->InitBuffer(post_.tmpBuff_, 128U);
        post_.hPreBuff_ = post_.tmpBuff_.template Get<P>();
        post_.hPostBuff_ = post_.hPreBuff_[16U];
        uint32_t rmsElements = RmsGroups() * BasicApiRoundUp<P>(WaveBatchRows());
        post_.pipe_->InitBuffer(wavePartialBuf_,
                                (WaveMmPartialElements() + rmsElements + MHC_PRE_BASIC_API_VL_FP32) * sizeof(P));
        post_.pipe_->InitBuffer(waveMmBuf_, 384U);
        // Alpha/bias are not required before the first MMAD.
        wavePartialFreeEvent_ = static_cast<event_t>(vector_.pipe_->template AllocEventID<HardEvent::V_MTE2>());
        SetFlag<HardEvent::V_MTE2>(wavePartialFreeEvent_);
        waveNzEvent_ = static_cast<event_t>(vector_.pipe_->template AllocEventID<HardEvent::MTE3_V>());
        waveDrainEvent_ = static_cast<event_t>(vector_.pipe_->template AllocEventID<HardEvent::MTE3_V>());
        tailNzFreeEvent_ = static_cast<event_t>(vector_.pipe_->template AllocEventID<HardEvent::V_MTE2>());
        tailXReadyEvent_ = static_cast<event_t>(vector_.pipe_->template AllocEventID<HardEvent::MTE2_V>());
    }

    __aicore__ inline uint32_t WaveRow(uint32_t wave, uint32_t ordinal, uint32_t mIndex, uint32_t kIndex)
    {
        return wave * WaveStride() + mIndex * 128U + (2U * kIndex + GetSubBlockIdx()) * WaveRows() + ordinal;
    }

    __aicore__ inline void PrepareWaveBatch(uint32_t row, bool rmsPrepared = false)
    {
        // Only a batch boundary can overwrite the small output coefficient buffers.
        SetFlag<HardEvent::MTE3_V>(waveDrainEvent_);
        WaitFlag<HardEvent::MTE3_V>(waveDrainEvent_);
        post_.globalOffsetM_ = row;
        post_.vectorOffset_.offsetMStart = 0U;
        post_.vectorOffset_.singleCoreM = WaveBatchRows();
        auto partial = wavePartialBuf_.template Get<P>();
        auto rms = partial[WaveMmPartialElements()];
        auto mm = waveMmBuf_.template Get<P>();
        constexpr uint32_t projectionBytes = 24U * sizeof(P);
        constexpr uint32_t paddedProjectionN = MHC_PRE_MK_WAVE_PROJECTION_N;
        // Read only the 24 logical columns into the existing compact, group-major UB buffer.
        DataCopyExtParams mcopy{
            static_cast<uint16_t>(WaveMmGroups()), projectionBytes,
            static_cast<uint32_t>(vector_.totalLength_) * paddedProjectionN * static_cast<uint32_t>(sizeof(P)) -
                projectionBytes,
            (WaveBatchRows() - 1U) * projectionBytes / 32U, 0U};
        DataCopyPadExtParams<P> mpad{true, 0U, 0U, 0U};
        WaitFlag<HardEvent::V_MTE2>(wavePartialFreeEvent_);
        if (!rmsPrepared) {
            MhcPreMkCopyRmsPartials(rms, partialRmsGm_[row], RmsGroups(), WaveBatchRows(), vector_.totalLength_);
        }
        for (uint32_t localRow = 0U; localRow < WaveBatchRows(); ++localRow) {
            DataCopyPad(partial[localRow * 24U], partialMmGm_[(row + localRow) * paddedProjectionN], mcopy, mpad);
        }
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(post_.pipe_->FetchEventID(HardEvent::MTE2_V)));
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(post_.pipe_->FetchEventID(HardEvent::MTE2_V)));
        if (!rmsPrepared) {
            MhcPreMkReduceRms(post_.invRmsUb_, rms, RmsGroups(), WaveBatchRows(), post_.scaleMean_,
                              post_.matrixInfo_.normEps);
        }
        const bool compensate = vector_.totalLength_ <= MHC_PRE_MK_PRECISE_PROJECTION_MAX_M;
        MhcPreMkReduceMm(mm, partial, WaveMmGroups(), WaveBatchRows(), 24U, true, true, compensate);
        SetFlag<HardEvent::V_MTE2>(wavePartialFreeEvent_);
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(post_.pipe_->FetchEventID(HardEvent::V_MTE3)));
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(post_.pipe_->FetchEventID(HardEvent::V_MTE3)));
        DataCopyExtParams mmout{1U, WaveBatchRows() * 24U * static_cast<uint32_t>(sizeof(P)), 0U, 0U, 0U};
        DataCopyExtParams rout{1U, WaveBatchRows() * static_cast<uint32_t>(sizeof(P)), 0U, 0U, 0U};
        if (post_.outFlag_) {
            DataCopyPad(post_.hMixGm_[row * 24U], mm, mmout);
            DataCopyPad(post_.invRmsGm_[row], post_.invRmsUb_, rout);
        }
        LocalTensor<P> res;
        if constexpr (RESI_MODE == MHC_PRE_HAS_RESI) {
            res = post_.outQueue_.template AllocTensor<P>();
        }
        MhcPreVFPostSegments<P, 4U, RESI_MODE == MHC_PRE_HAS_RESI>(post_.hPreBuff_, post_.hPostBuff_, res, mm,
                                                                   post_.invRmsUb_, post_.alphaInUb_, post_.biasInUb_,
                                                                   WaveBatchRows(), 0U, 24U);
        PipeBarrier<PIPE_V>();
        if constexpr (RESI_MODE == MHC_PRE_HAS_RESI) {
            post_.outQueue_.EnQue(res);
            res = post_.outQueue_.template DeQue<P>();
            DataCopyExtParams copy{1U, WaveBatchRows() * 16U * static_cast<uint32_t>(sizeof(P)), 0U, 0U, 0U};
            DataCopyPad(post_.hResGm_[row * 16U], res, copy);
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(post_.pipe_->FetchEventID(HardEvent::MTE3_V)));
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(post_.pipe_->FetchEventID(HardEvent::MTE3_V)));
            post_.outQueue_.FreeTensor(res);
        }
        post_.AIV1ProcessHPost(0U, WaveBatchRows());
        post_.AIV1ProcessHPre(0U, WaveBatchRows());
    }

    __aicore__ inline void WaveLoadX(uint32_t row, uint32_t chunk = 0U)
    {
        uint32_t lenD = WaveChunkLength(chunk);
        auto x = post_.xInQueue_.template AllocTensor<T>();
        DataCopyExtParams copy{4U, lenD * static_cast<uint32_t>(sizeof(T)),
                               (static_cast<uint32_t>(vector_.D_) - lenD) * static_cast<uint32_t>(sizeof(T)), 0U, 0U};
        DataCopyPadExtParams<T> pad{false, 0U, 0U, 0U};
        DataCopyPad(x, post_.xGm_[row * 4U * vector_.D_ + chunk * WaveChunkD()], copy, pad);
        post_.xInQueue_.EnQue(x);
    }

    __aicore__ inline void WavePostTick(uint32_t wave, uint32_t ordinal, uint32_t mIndex, uint32_t kIndex,
                                        bool xPrefetched = false)
    {
        uint32_t row = WaveRow(wave, ordinal, mIndex, kIndex);
        if (ordinal % WaveBatchRows() == 0U) {
            PrepareWaveBatch(row);
        }
        if (ordinal == 0U && !xPrefetched) {
            WaveLoadX(row);
        }
        for (uint32_t chunk = 0U; chunk < WaveDChunks(); ++chunk) {
            auto x = post_.xInQueue_.template DeQue<T>();
            if (chunk + 1U < WaveDChunks()) {
                WaveLoadX(row, chunk + 1U);
            } else if (ordinal + 1U < WaveRows()) {
                WaveLoadX(WaveRow(wave, ordinal + 1U, mIndex, kIndex));
            }
            uint32_t lenD = WaveChunkLength(chunk);
            auto out = post_.outQueue_.template AllocTensor<T>();
            post_.template VFDoV1ProcessHInRows<4U>(
                (__ubuf__ T*)x.GetPhyAddr(), (__ubuf__ T*)out.GetPhyAddr(),
                (__ubuf__ P*)post_.hPreBuff_[(ordinal % WaveBatchRows()) * 4U].GetPhyAddr(), 1U, lenD, lenD);
            post_.xInQueue_.FreeTensor(x);
            post_.outQueue_.EnQue(out);
            out = post_.outQueue_.template DeQue<T>();
            DataCopyExtParams copy{1U, lenD * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(post_.hinGm_[row * vector_.D_ + chunk * WaveChunkD()], out, copy);
            post_.outQueue_.FreeTensor(out);
        }
    }

    __aicore__ inline void PrefetchTailX2(uint32_t row)
    {
        // NZ no longer feeds L1 after the final input tile. Bridge MTE3->V->MTE2
        // before overwriting that storage with an independent GM->UB transfer.
        WaitFlag<HardEvent::MTE3_V>(waveNzEvent_);
        SetFlag<HardEvent::V_MTE2>(tailNzFreeEvent_);
        WaitFlag<HardEvent::V_MTE2>(tailNzFreeEvent_);
        auto x = nd2NzLocal_.template ReinterpretCast<T>();
        DataCopyExtParams copy{1U, 4U * static_cast<uint32_t>(vector_.D_) * static_cast<uint32_t>(sizeof(T)), 0U, 0U,
                               0U};
        DataCopyPadExtParams<T> pad{false, 0U, 0U, 0U};
        DataCopyPad(x, post_.xGm_[row * 4U * static_cast<uint32_t>(vector_.D_)], copy, pad);
        SetFlag<HardEvent::MTE2_V>(tailXReadyEvent_);
        auto first = vector_.xInQueue_.template AllocTensor<T>();
        DataCopyExtParams firstCopy{
            1U, tailSplitHeads_ * static_cast<uint32_t>(vector_.D_) * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(first, post_.xGm_[(row + 1U) * 4U * static_cast<uint32_t>(vector_.D_)], firstCopy, pad);
        vector_.xInQueue_.EnQue(first);
        auto rest = vector_.xInQueue_.template AllocTensor<T>();
        DataCopyExtParams restCopy{
            1U, (4U - tailSplitHeads_) * static_cast<uint32_t>(vector_.D_) * static_cast<uint32_t>(sizeof(T)), 0U, 0U,
            0U};
        DataCopyPad(rest,
                    post_.xGm_[(row + 1U) * 4U * static_cast<uint32_t>(vector_.D_) +
                               tailSplitHeads_ * static_cast<uint32_t>(vector_.D_)],
                    restCopy, pad);
        vector_.xInQueue_.EnQue(rest);
        SetFlag<HardEvent::MTE3_V>(waveNzEvent_); // Preserve the existing final drain token.
    }

    __aicore__ inline void PrepareTailRms(uint32_t row)
    {
        SetFlag<HardEvent::MTE3_V>(waveDrainEvent_);
        WaitFlag<HardEvent::MTE3_V>(waveDrainEvent_);
        WaitFlag<HardEvent::V_MTE2>(wavePartialFreeEvent_);
        // Wait for every paired AIV's RMS stores before reading the final wave.
        CrossCoreSetFlag<0, PIPE_MTE3>(11U);
        CrossCoreWaitFlag<0, PIPE_MTE2>(11U);
        auto rms = wavePartialBuf_.template Get<P>()[WaveMmPartialElements()];
        MhcPreMkCopyRmsPartials(rms, partialRmsGm_[row], RmsGroups(), WaveBatchRows(), vector_.totalLength_);
        auto ready = static_cast<event_t>(post_.pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ready);
        WaitFlag<HardEvent::MTE2_V>(ready);
        MhcPreMkReduceRms(post_.invRmsUb_, rms, RmsGroups(), WaveBatchRows(), post_.scaleMean_,
                          post_.matrixInfo_.normEps);
        SetFlag<HardEvent::V_MTE2>(wavePartialFreeEvent_);
    }

    __aicore__ inline void FinishTailRows(uint32_t wave, uint32_t mIndex, uint32_t kIndex)
    {
        PrepareWaveBatch(WaveRow(wave, 0U, mIndex, kIndex), true);
        // Dequeue both prefetched first chunks, preserving queue order for later D chunks.
        auto first = post_.xInQueue_.template DeQue<T>();
        auto second = post_.xInQueue_.template DeQue<T>();
        for (uint32_t ordinal = 0U; ordinal < WaveBatchRows(); ++ordinal) {
            uint32_t row = WaveRow(wave, ordinal, mIndex, kIndex);
            LocalTensor<T> x;
            LocalTensor<T> rest;
            if (ordinal == 2U) {
                WaitFlag<HardEvent::MTE2_V>(tailXReadyEvent_);
                x = nd2NzLocal_.template ReinterpretCast<T>();
            } else if (ordinal == 3U) {
                x = vector_.xInQueue_.template DeQue<T>();
                rest = vector_.xInQueue_.template DeQue<T>();
            } else {
                x = ordinal == 0U ? first : second;
            }
            for (uint32_t chunk = 0U; chunk < WaveDChunks(); ++chunk) {
                if (ordinal < 2U && chunk > 0U) {
                    WaveLoadX(row, chunk);
                    x = post_.xInQueue_.template DeQue<T>();
                }
                uint32_t lenD = WaveChunkLength(chunk);
                uint32_t offsetD = chunk * WaveChunkD();
                auto out = post_.outQueue_.template AllocTensor<T>();
                if (ordinal == 3U) {
                    if (tailSplitHeads_ == 3U) {
                        post_.template VFDoTailHInSplit<3U>(
                            (__ubuf__ T*)x.GetPhyAddr(), (__ubuf__ T*)rest.GetPhyAddr(), (__ubuf__ T*)out.GetPhyAddr(),
                            (__ubuf__ P*)post_.hPreBuff_[ordinal * 4U].GetPhyAddr(), offsetD, lenD);
                    } else {
                        post_.template VFDoTailHInSplit<2U>(
                            (__ubuf__ T*)x.GetPhyAddr(), (__ubuf__ T*)rest.GetPhyAddr(), (__ubuf__ T*)out.GetPhyAddr(),
                            (__ubuf__ P*)post_.hPreBuff_[ordinal * 4U].GetPhyAddr(), offsetD, lenD);
                    }
                } else {
                    uint32_t localStride = ordinal == 2U ? static_cast<uint32_t>(vector_.D_) : lenD;
                    auto source = ordinal == 2U ? x[offsetD] : x;
                    post_.template VFDoV1ProcessHInRows<4U>(
                        (__ubuf__ T*)source.GetPhyAddr(), (__ubuf__ T*)out.GetPhyAddr(),
                        (__ubuf__ P*)post_.hPreBuff_[ordinal * 4U].GetPhyAddr(), 1U, lenD, localStride);
                    if (ordinal < 2U) {
                        post_.xInQueue_.FreeTensor(x);
                    }
                }
                post_.outQueue_.EnQue(out);
                out = post_.outQueue_.template DeQue<T>();
                DataCopyExtParams copy{1U, lenD * static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
                DataCopyPad(post_.hinGm_[row * vector_.D_ + offsetD], out, copy);
                post_.outQueue_.FreeTensor(out);
            }
            if (ordinal == 3U) {
                vector_.xInQueue_.FreeTensor(x);
                vector_.xInQueue_.FreeTensor(rest);
            }
        }
    }

    __aicore__ inline void WaitWave(uint32_t wave)
    {
        // Same visibility chain as MegaMhc: paired FIX -> all AIV MTE3 stores -> MTE2 reads.
        CrossCoreWaitFlag<2, PIPE_MTE3>(4U + (wave & 1U));
        CrossCoreSetFlag<0, PIPE_MTE3>(10U);
        CrossCoreWaitFlag<0, PIPE_MTE2>(10U);
    }

    // The schedulers pick the slots and publication order; the AIV tile producer is shared.
    enum class XNzInput {
        Input,
        InputWithRms,
        InputWithRmsPartials,
        Float
    };

    __aicore__ inline void PrefetchX(uint32_t mStart, uint32_t rowStart, uint32_t rows, uint32_t k, uint32_t columns)
    {
        vector_.globalOffsetM_ = mStart;
        vector_.xLocal_ = vector_.xInQueue_.template AllocTensor<T>();
        vector_.DataCopyX(rows, columns, rowStart, k);
    }

    static __aicore__ inline void WaitForXSlot()
    {
        CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_CONSUMED_FLAG);
    }

    static __aicore__ inline void PublishXReady()
    {
        CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_READY_FLAG);
    }

    template <XNzInput INPUT, bool WAVE = false>
    __aicore__ inline void StageXToL1(LocalTensor<T> x, LocalTensor<P> fp32, const LocalTensor<P>& rms,
                                      __ubuf__ P* gamma, uint32_t rows, uint32_t m, uint32_t k, uint32_t rowStart,
                                      uint32_t nzSlot, uint32_t l1Slot, event_t event, __ubuf__ P* correction = nullptr,
                                      bool firstKTile = true)
    {
        if constexpr (!WAVE) {
            WaitFlag<HardEvent::MTE3_V>(event);
        }
        auto nz = nd2NzLocal_[nzSlot * nzSlotElements_];
        if constexpr (INPUT == XNzInput::Float) {
            MhcPreMkVFTransND2NZ(nz, fp32, rows, k);
            vector_.outQueue_.FreeTensor(fp32);
        } else {
            if constexpr (INPUT == XNzInput::InputWithRms) {
                MhcPreMkVFPackNz<true>(nz, rms, x, rows, k, gamma);
            } else if constexpr (INPUT == XNzInput::InputWithRmsPartials) {
                if (firstKTile) {
                    MhcPreMkVFPackNz<true, true>(nz, rms, x, rows, k, gamma, correction);
                } else {
                    MhcPreMkVFPackNz<false, true>(nz, rms, x, rows, k, gamma, correction);
                }
            } else {
                MhcPreMkVFPackOnlyNz(nz, x, rows, k, gamma);
            }
            vector_.xInQueue_.FreeTensor(x);
        }
        // The wave path reserves its V_MTE3 event separately from the NZ reuse event.
        event_t copyEvent = WAVE ? static_cast<event_t>(vector_.pipe_->FetchEventID(HardEvent::V_MTE3)) : event;
        SetFlag<HardEvent::V_MTE3>(copyEvent);
        WaitFlag<HardEvent::V_MTE3>(copyEvent);
        constexpr uint32_t pad = INPUT == XNzInput::Float ? 0U : 1U;
        constexpr uint32_t rowStep = MHC_PRE_BASIC_API_BLOCK_SIZE / sizeof(P);
        DataCopyParams copy;
        copy.blockCount = BasicApiCeilDiv(k, MHC_PRE_BASIC_API_C0_SIZE);
        copy.blockLen = rows * MHC_PRE_BASIC_API_C0_SIZE / rowStep;
        copy.srcStride = BasicApiAlign(rows, MHC_PRE_BASIC_API_C0_SIZE) + pad - rows;
        copy.dstStride = BasicApiAlign(m, AscendC::BLOCK_CUBE) - rows;
        BasicApiCopyToL1(nz, aL1_[l1Slot * MHC_PRE_BASIC_API_L1_BUF_OFFSET + rowStart * rowStep], copy);
    }

    __aicore__ inline void MmadStagedX(uint32_t m, uint32_t n, uint32_t cSlot, bool firstPartial, bool lastPartial,
                                       uint32_t sequence)
    {
        CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_READY_FLAG);
        CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_READY_FLAG + MHC_PRE_SUBBLOCK_FLAG_OFFSET);
        mmService_.MmadWithReusableB(m, n, cSlot, firstPartial, lastPartial,
                                     aL1_[(sequence % 3U) * MHC_PRE_BASIC_API_L1_BUF_OFFSET]);
        CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG);
        CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG +
                                                                  MHC_PRE_SUBBLOCK_FLAG_OFFSET);
    }

    static __aicore__ inline void WaitForMmad()
    {
        SetFlag<HardEvent::M_FIX>(EVENT_ID2);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID2);
    }

    __aicore__ inline void ProcessWave()
    {
        constexpr uint32_t waveStages = MHC_PRE_EXPERIMENT_WAVE_STAGES;
        uint32_t waves = static_cast<uint32_t>(vector_.totalLength_) / WaveStride();
        uint32_t kTiles = vector_.tiling_->multCoreSplitKSize / 256U;
        uint32_t partialTiles = WavePartialK() / 256U;
        uint32_t aicIndex = GetBlockIdx();
        if ASCEND_IS_AIV {
            aicIndex /= 2U;
        }
        uint32_t mIndex = aicIndex % vector_.tiling_->cubeBlockDimM;
        uint32_t kIndex = aicIndex / vector_.tiling_->cubeBlockDimM;
        uint32_t kStart = kIndex * vector_.tiling_->multCoreSplitKSize;
        uint32_t sequence = 0U;
        if ASCEND_IS_AIC {
            mmService_.InitReusableBPadding(bL1_);
            SetFlag<HardEvent::FIX_M>(EVENT_ID2);
            SetFlag<HardEvent::FIX_M>(EVENT_ID3);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID5);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID6);
            for (uint32_t wave = 0U; wave < waves; ++wave) {
                event_t waveFixEvent = static_cast<event_t>(EVENT_ID2 + (wave & 1U));
                for (uint32_t kt = 0U; kt < kTiles; ++kt) {
                    bool firstPartial = kt % partialTiles == 0U;
                    bool lastPartial = (kt + 1U) % partialTiles == 0U || kt + 1U == kTiles;
                    if (firstPartial) {
                        WaitFlag<HardEvent::FIX_M>(waveFixEvent);
                    }
                    mmService_.LoadReusableB(vector_.matrixInfo_.nD, 24U, vector_.phiGm_[kStart + kt * 256U], bL1_);
                    for (uint32_t stage = 0U; stage < waveStages; ++stage) {
                        uint32_t cStage = (wave & 1U) * waveStages + stage;
                        uint32_t mStart = (wave * waveStages + stage) * WaveStride() + mIndex * 128U;
                        MmadStagedX(128U, MHC_PRE_MK_WAVE_PROJECTION_N, cStage, firstPartial, lastPartial, sequence);
                        if (lastPartial) {
                            WaitForMmad();
                            mmService_.CopyOutMBlock(
                                128U, MHC_PRE_MK_WAVE_PROJECTION_N, cStage,
                                partialMmGm_[((kIndex * BasicApiCeilDiv(kTiles, partialTiles) + kt / partialTiles) *
                                                  static_cast<uint32_t>(vector_.totalLength_) +
                                              mStart) *
                                             MHC_PRE_MK_WAVE_PROJECTION_N]);
                        }
                        ++sequence;
                    }
                    mmService_.ReleaseReusableB();
                    if (lastPartial) {
                        SetFlag<HardEvent::FIX_M>(waveFixEvent);
                    }
                }
                CrossCoreSetFlag<2, PIPE_FIX>(4U + (wave & 1U));
            }
            WaitFlag<HardEvent::FIX_M>(EVENT_ID2);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID3);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID6);
            mmService_.End(vector_.implMode_);
        }
        if ASCEND_IS_AIV {
            SetFlag<HardEvent::MTE3_V>(waveNzEvent_);
            for (uint32_t wave = 0U; wave < waves; ++wave) {
                uint32_t postDone = 0U;
                for (uint32_t tile = 0U; tile < kTiles * waveStages; ++tile) {
                    uint32_t stage = tile % waveStages;
                    uint32_t mStart = (wave * waveStages + stage) * WaveStride() + mIndex * 128U;
                    if (tile == 0U && wave != 0U) {
                        PrefetchX(mStart, GetSubBlockIdx() * 64U, 64U, kStart, MHC_PRE_RMS_K_BLOCK);
                    }
                    WaitForXSlot();
                    auto x = vector_.xInQueue_.template DeQue<T>();
                    if (tile + 1U < kTiles * waveStages) {
                        uint32_t next = tile + 1U;
                        const uint32_t nextM = (wave * waveStages + next % waveStages) * WaveStride() + mIndex * 128U;
                        PrefetchX(nextM, GetSubBlockIdx() * 64U, 64U,
                                  kStart + (next / waveStages) * MHC_PRE_RMS_K_BLOCK, MHC_PRE_RMS_K_BLOCK);
                    }
                    WaitFlag<HardEvent::MTE3_V>(waveNzEvent_);
                    auto rms = rmsTileBuffer_.template Get<P>();
                    const uint32_t kOffset = kStart + (tile / waveStages) * MHC_PRE_RMS_K_BLOCK;
                    LocalTensor<P> gamma;
                    __ubuf__ P* gammaAddr = nullptr;
                    if (vector_.hasGamma_) {
                        gamma = vector_.LoadGammaTile(kOffset);
                        gammaAddr = (__ubuf__ P*)gamma.GetPhyAddr();
                    }
                    StageXToL1<XNzInput::InputWithRms, true>(x, {}, rms, gammaAddr, 64U, 128U, MHC_PRE_RMS_K_BLOCK,
                                                             GetSubBlockIdx() * 64U, 0U, sequence % 3U, waveNzEvent_);
                    PublishXReady();
                    StoreRmsPartial(rms, 64U, mStart + GetSubBlockIdx() * 64U, kOffset);
                    if (vector_.hasGamma_) {
                        vector_.gammaInQueue_.FreeTensor(gamma);
                    }
                    SetFlag<HardEvent::MTE3_V>(waveNzEvent_);
                    ++sequence;
                    if (wave == 0U && tile == 1U) {
                        post_.AIVPreLoad();
                    }
                    if (wave != 0U && tile == 0U) {
                        WaitWave(wave - 1U);
                    }
                    if (wave != 0U && tile >= 2U && postDone < WaveRows()) {
                        WavePostTick(wave - 1U, postDone++, mIndex, kIndex);
                        if (wave + 1U == waves && postDone == WaveRows() - (WaveDChunks() == 1U ? 1U : 0U)) {
                            WaveLoadX(WaveRow(wave, 0U, mIndex, kIndex));
                        }
                    }
                }
                if (wave + 1U == waves) {
                    if (WaveBatchRows() == 4U) {
                        PrefetchTailX2(WaveRow(wave, 2U, mIndex, kIndex));
                    }
                }
                if (wave != 0U) {
                    while (postDone < WaveRows()) {
                        WavePostTick(wave - 1U, postDone++, mIndex, kIndex);
                        if (wave + 1U == waves && postDone == WaveRows() - (WaveDChunks() == 1U ? 1U : 0U)) {
                            WaveLoadX(WaveRow(wave, 0U, mIndex, kIndex));
                        }
                    }
                }
                if (wave + 1U == waves) {
                    if (waves == 1U) {
                        WaveLoadX(WaveRow(wave, 0U, mIndex, kIndex));
                    }
                    WaveLoadX(WaveRow(wave, 1U, mIndex, kIndex));
                    PrepareTailRms(WaveRow(wave, 0U, mIndex, kIndex));
                }
            }
            WaitFlag<HardEvent::MTE3_V>(waveNzEvent_);
            WaitWave(waves - 1U);
            FinishTailRows(waves - 1U, mIndex, kIndex);
            if (WaveRows() > WaveBatchRows()) {
                WaveLoadX(WaveRow(waves - 1U, WaveBatchRows(), mIndex, kIndex));
            }
            for (uint32_t ordinal = WaveBatchRows(); ordinal < WaveRows(); ++ordinal) {
                WavePostTick(waves - 1U, ordinal, mIndex, kIndex);
            }
            SetFlag<HardEvent::MTE3_V>(waveDrainEvent_);
            WaitFlag<HardEvent::MTE3_V>(waveDrainEvent_);
            WaitFlag<HardEvent::V_MTE2>(wavePartialFreeEvent_);
            post_.invRmsOutQueue_.FreeTensor(post_.invRmsUb_);
            post_.biasInQue_.FreeTensor(post_.biasInUb_);
        }
        SyncAll<false>();
    }
#endif

    __aicore__ inline void Init(InitParams initParams)
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        vector_.BindGlobalTensors(initParams);
        vector_.InitFromTilingData(initParams.tilingData);
        vector_.InitMNConfig();
        // Retain X in L2 across both stages for FP32 and HF32 weight reuse.
        if (UseWeightReuseStage()) {
            vector_.xGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_PERSISTENT);
        }
        vector_.pipe_ = initParams.tPipeIn;
        partialMmGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ P*>(initParams.workspace + vector_.tiling_->mkWorkspaceMmOffset));
        if constexpr (WEIGHT_REUSE) {
            partialRmsGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ P*>(initParams.workspace + vector_.tiling_->mkWorkspaceFinalOffset));
        } else if (vector_.tiling_->mkWorkspaceRmsOffset != 0U) {
            partialRmsGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ P*>(initParams.workspace + vector_.tiling_->mkWorkspaceRmsOffset));
        }
        if (UseGmStage() || UseSharedXStage()) {
            auto stage = reinterpret_cast<__gm__ P*>(initParams.workspace + vector_.tiling_->mkWorkspaceFinalOffset);
            xStageGm_.SetGlobalBuffer(stage);
            if (UseSharedXStage()) {
                uint32_t xElements = BasicApiRoundUp<P>(vector_.totalLength_ * vector_.matrixInfo_.nD);
                partialRmsGm_.SetGlobalBuffer(stage + xElements);
            }
        }

        vector_.pipe_->InitBuffer(l1Buffer_, MHC_PRE_BASIC_API_L1_ALLOC_SIZE);
        aL1_ = l1Buffer_.Get<P>();
        bL1_ = aL1_[(UseWeightReuseStage() ? 3U : MHC_PRE_BASIC_API_L1_BUF_NUM) * MHC_PRE_BASIC_API_L1_BUF_OFFSET];

        if ASCEND_IS_AIC {
            mmService_.Init(vector_.implMode_);
            if (UseSharedXStage()) {
                return;
            }
            // Seed two availability tokens for each subblock because X staging
            // is ping-ponged. Every later token is returned after AIC consumes X.
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG);
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG +
                                                                      MHC_PRE_SUBBLOCK_FLAG_OFFSET);
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG);
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG +
                                                                      MHC_PRE_SUBBLOCK_FLAG_OFFSET);
            if (UseWeightReuseStage()) {
                CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG);
                CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG +
                                                                          MHC_PRE_SUBBLOCK_FLAG_OFFSET);
            }
        }
        if ASCEND_IS_AIV {
            if (UseSharedXStage()) {
                InitSharedXBuffers();
                return;
            }
            uint32_t rowCapacity = (vector_.tiling_->mL1Size + 1U) / 2U;
            uint32_t kCapacity = BasicApiRoundUp<T>(vector_.tiling_->kUbSize);
            vector_.pipe_->InitBuffer(vector_.xInQueue_, IsWaveCase() ? 2 : (UseWeightReuseStage() ? 3 : 2),
                                      rowCapacity * kCapacity * sizeof(T));
            // GM staging is ping-ponged to overlap AIV MTE3, AIC MTE2 ND2NZ, and Cube execution.
            // Fused staging skips the intermediate FP32 ND buffer and ping-pongs padded NZ tiles.
            if (!UseFusedNzStage()) {
                vector_.pipe_->InitBuffer(vector_.outQueue_, UseGmStage() ? 2 : 1,
                                          rowCapacity * BasicApiRoundUp<P>(vector_.tiling_->kUbSize) * sizeof(P));
            }
            if (vector_.hasGamma_) {
                vector_.pipe_->InitBuffer(vector_.gammaInQueue_, 2,
                                          BasicApiRoundUp<P>(vector_.tiling_->kUbSize) * sizeof(P));
            }
            if (!UseGmStage()) {
                uint32_t nzRows = BasicApiAlign(rowCapacity, MHC_PRE_BASIC_API_C0_SIZE);
                nzSlotElements_ =
                    (nzRows + (UseFusedNzStage() ? 1U : 0U)) * BasicApiRoundUp<P>(vector_.tiling_->kUbSize);
                vector_.pipe_->InitBuffer(
                    nd2NzBuf_, nzSlotElements_ * sizeof(P) * (IsWaveCase() ? 1U : (UseFusedNzStage() ? 2U : 1U)));
                nd2NzLocal_ = nd2NzBuf_.Get<P>();
            }
            if constexpr (WEIGHT_REUSE) {
                uint32_t rmsSlots = IsWaveCase() ? 1U : 2U;
                vector_.pipe_->InitBuffer(rmsTileBuffer_, rmsSlots * kRmsRowsPerTile * sizeof(P));
            } else if (vector_.tiling_->mkWorkspaceRmsOffset != 0U) {
                vector_.pipe_->InitBuffer(rmsTileBuffer_, 2U * BasicApiRoundUp<P>(rowCapacity) * sizeof(P));
            }
            if (IsWaveCase()) {
                uint32_t aicIndex = GetBlockIdx() / 2U;
                tailSplitHeads_ = AscendC::Std::min(
                    3U, 32768U / (static_cast<uint32_t>(vector_.D_) * static_cast<uint32_t>(sizeof(T))));
                PrefetchX((aicIndex % vector_.tiling_->cubeBlockDimM) * 128U, GetSubBlockIdx() * 64U, 64U,
                          (aicIndex / vector_.tiling_->cubeBlockDimM) * vector_.tiling_->multCoreSplitKSize,
                          MHC_PRE_RMS_K_BLOCK);
                InitWavePost(initParams);
            }
        }
    }
#else
    {
        vector_.BindGlobalTensors(initParams);
        vector_.InitFromTilingData(initParams.tilingData);
        vector_.InitMNConfig();
        vector_.pipe_ = initParams.tPipeIn;
        partialMmGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ P*>(initParams.workspace + vector_.tiling_->mkWorkspaceMmOffset));
        partialRmsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ P*>(initParams.workspace + vector_.tiling_->mkWorkspaceRmsOffset));
        if (UseGmStage()) {
            xStageGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ P*>(initParams.workspace + vector_.tiling_->mkWorkspaceFinalOffset));
        }

        vector_.pipe_->InitBuffer(l1Buffer_, MHC_PRE_BASIC_API_L1_ALLOC_SIZE);
        aL1_ = l1Buffer_.Get<P>();
        bL1_ = aL1_[MHC_PRE_BASIC_API_L1_BUF_NUM * MHC_PRE_BASIC_API_L1_BUF_OFFSET];

        if ASCEND_IS_AIC {
            mmService_.Init(vector_.implMode_);
            // Seed two availability tokens for each subblock because X staging
            // is ping-ponged. Every later token is returned after AIC consumes X.
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG); // Ping
            if constexpr (MHC_PRE_AIV_PER_AIC == TWO_AIV_PER_AIC) {
                CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
                    MHC_PRE_X_CONSUMED_FLAG + MHC_PRE_SUBBLOCK_FLAG_OFFSET); // Ping token for the second AIV
            }
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG); // Pong
            if constexpr (MHC_PRE_AIV_PER_AIC == TWO_AIV_PER_AIC) {
                CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(
                    MHC_PRE_X_CONSUMED_FLAG + MHC_PRE_SUBBLOCK_FLAG_OFFSET); // Pong token for the second AIV
            }
        }
        if ASCEND_IS_AIV {
            uint32_t rowCapacity = vector_.tiling_->mL1Size;
            if constexpr (MHC_PRE_AIV_PER_AIC == TWO_AIV_PER_AIC) {
                rowCapacity = (rowCapacity + vector_.kHalfSplitDivisor - 1U) / vector_.kHalfSplitDivisor;
            }
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
#endif

    __aicore__ inline void Process()
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        uint32_t aicIndex = GetBlockIdx();
        if ASCEND_IS_AIV {
            aicIndex /= 2U;
            if (UseSharedXStage()) {
                ProcessAivSharedX(aicIndex);
            }
        }
        uint32_t usedCoreNum = vector_.tiling_->cubeBlockDimM * vector_.tiling_->cubeBlockDimK;
        if (aicIndex < usedCoreNum) {
            if (UseWeightReuseStage()) {
                uint32_t mIndex = aicIndex % vector_.tiling_->cubeBlockDimM;
                uint32_t kIndex = aicIndex / vector_.tiling_->cubeBlockDimM;
                uint32_t kStart = kIndex * vector_.tiling_->multCoreSplitKSize;
                uint32_t kEnd = kStart + vector_.tiling_->multCoreSplitKSize;
                if ASCEND_IS_AIC {
                    ProcessAicWithWeightReuse(mIndex, kIndex, kStart, kEnd);
                } else {
                    ProcessAivWithWeightReuse(mIndex, kStart, kEnd);
                }
            } else {
                uint32_t mIndex = aicIndex / vector_.tiling_->cubeBlockDimK;
                uint32_t kIndex = aicIndex % vector_.tiling_->cubeBlockDimK;
                uint32_t mStart = mIndex * vector_.tiling_->mL1Size;
                uint32_t mReal =
                    AscendC::Std::min(static_cast<uint64_t>(vector_.tiling_->mL1Size), vector_.totalLength_ - mStart);
                uint32_t kStart = kIndex * vector_.tiling_->multCoreSplitKSize;
                uint32_t kEnd = AscendC::Std::min(static_cast<uint64_t>(kStart + vector_.tiling_->multCoreSplitKSize),
                                                  static_cast<uint64_t>(vector_.matrixInfo_.nD));
                if ASCEND_IS_AIC {
                    if (UseSharedXStage()) {
                        ProcessAic<true>(mStart, mReal, kIndex, kStart, kEnd);
                    } else {
                        ProcessAic<false>(mStart, mReal, kIndex, kStart, kEnd);
                    }
                } else if (!UseSharedXStage()) {
                    ProcessAiv(mStart, mReal, kIndex, kStart, kEnd);
                }
            }
        }
        if ASCEND_IS_AIC {
            mmService_.End(vector_.implMode_);
        }
        SyncAll<false>();
    }
#else
    {
        uint32_t aicIndex = GetBlockIdx();
        if ASCEND_IS_AIV {
            aicIndex /= MHC_PRE_AIV_PER_AIC;
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
            mmService_.End();
        }
        SyncAll<false>();
    }
#endif

private:
    __aicore__ inline bool UseSequentialKPartials() const
    {
#if defined(MHC_PRE_ENABLE_ASCEND950)
        if constexpr (WEIGHT_REUSE) {
            const uint32_t splitK = vector_.tiling_->multCoreSplitKSize;
            return MhcPreMkWeightReusePartialK(vector_.totalLength_, splitK,
                                               vector_.implMode_ == MHC_PRE_IMPL_MODE_FP32,
                                               RESI_MODE == MHC_PRE_HAS_RESI) < splitK;
        }
        if (MhcPreMkFineOrdinaryProjectionK(vector_.totalLength_, vector_.matrixInfo_.nD,
                                            vector_.tiling_->multCoreSplitKSize, vector_.tiling_->mkUseGmStage,
                                            vector_.implMode_) != 0U) {
            return true;
        }
#endif
        return vector_.tiling_->multCoreSplitKSize >= MHC_PRE_MK_SEQUENTIAL_PARTIAL_THRESHOLD;
    }

#if defined(MHC_PRE_ENABLE_ASCEND950)
    __aicore__ inline bool UseFusedNzStage() const
    {
        if constexpr (WEIGHT_REUSE) {
            return true;
        }
        return !UseGmStage() && vector_.matrixInfo_.nD % 128U == 0U && vector_.tiling_->kL1Size % 128U == 0U &&
               vector_.tiling_->multCoreSplitKSize % 128U == 0U;
    }

    __aicore__ inline bool UseWeightReuseStage() const
    {
        return WEIGHT_REUSE;
    }
#endif

    __aicore__ inline bool UseGmStage() const
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        return !WEIGHT_REUSE && vector_.tiling_->mkUseGmStage == 1U;
    }

    static constexpr uint32_t kSharedXTileRows = 8U;

    __aicore__ inline bool UseSharedXStage() const
    {
        return !WEIGHT_REUSE && vector_.tiling_->mkUseGmStage == MHC_PRE_MK_SHARED_X_STAGE;
    }

    __aicore__ inline uint32_t SharedXRowsPerPair() const
    {
        return BasicApiCeilDiv(vector_.totalLength_, vector_.coreNum_);
    }

    __aicore__ inline void InitSharedXBuffers()
    {
        vector_.pipe_->InitBuffer(vector_.xInQueue_, 1U, kSharedXTileRows * MHC_PRE_RMS_K_BLOCK * sizeof(T));
        vector_.pipe_->InitBuffer(vector_.outQueue_, 1U, kSharedXTileRows * MHC_PRE_RMS_K_BLOCK * sizeof(P));
        if (vector_.hasGamma_) {
            vector_.pipe_->InitBuffer(vector_.gammaInQueue_, 1U, MHC_PRE_RMS_K_BLOCK * sizeof(P));
        }
        uint32_t rows = (SharedXRowsPerPair() + 1U) / 2U;
        // The VF loads one register from a row's scalar sum before masking unused lanes.
        uint32_t rmsElements = BasicApiAlign(rows, MHC_PRE_BASIC_API_VL_FP32) + MHC_PRE_BASIC_API_VL_FP32;
        vector_.pipe_->InitBuffer(rmsTileBuffer_, rmsElements * sizeof(P));
    }

    template <bool FIRST_K>
    __aicore__ inline void StageSharedXTile(const LocalTensor<P>& rms, __ubuf__ P* correction, uint32_t row,
                                            uint32_t rows, uint32_t k, uint32_t columns)
    {
        vector_.xLocal_ = vector_.xInQueue_.template AllocTensor<T>();
        vector_.DataCopyX(rows, columns, row, k);
        auto x = vector_.xInQueue_.template DeQue<T>();
        auto fp32 = vector_.outQueue_.template AllocTensor<P>();
        auto xPtr = (__ubuf__ T*)x.GetPhyAddr();
        auto dst = (__ubuf__ P*)fp32.GetPhyAddr();
        auto sums = (__ubuf__ P*)rms.GetPhyAddr();
        if (vector_.hasGamma_) {
            vector_.gammaUb_ = vector_.gammaInQueue_.template AllocTensor<P>();
            vector_.DataCopyGamma(columns, k);
            auto gamma = vector_.gammaInQueue_.template DeQue<P>();
            vector_.template VFDoV0ProcessXIn<true, FIRST_K, false, true, true>(
                dst, sums, xPtr, (__ubuf__ P*)gamma.GetPhyAddr(), rows, columns, 0U, correction);
            vector_.gammaInQueue_.FreeTensor(gamma);
        } else {
            vector_.template VFDoV0ProcessXIn<false, FIRST_K, false, true, true>(dst, sums, xPtr, nullptr, rows,
                                                                                 columns, 0U, correction);
        }
        vector_.xInQueue_.FreeTensor(x);

        vector_.outQueue_.template EnQue<P>(fp32);
        fp32 = vector_.outQueue_.template DeQue<P>();
        uint32_t rowBytes = columns * sizeof(P);
        uint32_t kSize = vector_.matrixInfo_.nD;
        DataCopyExtParams copy{static_cast<uint16_t>(rows), rowBytes, 0U,
                               (kSize - columns) * static_cast<uint32_t>(sizeof(P)), 0U};
        DataCopyPad(xStageGm_[static_cast<uint64_t>(row) * kSize + k], fp32, copy);
        vector_.outQueue_.FreeTensor(fp32);
    }

    __aicore__ inline void ProcessAivSharedX(uint32_t aicIndex)
    {
        uint32_t rowsPerPair = SharedXRowsPerPair();
        uint32_t pairStart = aicIndex * rowsPerPair;
        uint32_t totalRows = vector_.totalLength_;
        uint32_t pairRows = pairStart < totalRows ? AscendC::Std::min(rowsPerPair, totalRows - pairStart) : 0U;
        uint32_t firstRows = (pairRows + 1U) / 2U;
        uint32_t rowStart = pairStart + (GetSubBlockIdx() == 0U ? 0U : firstRows);
        uint32_t rows = GetSubBlockIdx() == 0U ? firstRows : pairRows - firstRows;
        uint32_t kSize = vector_.matrixInfo_.nD;
        vector_.globalOffsetM_ = 0U;
        auto rms = rmsTileBuffer_.template Get<P>();
        // Reuse the padded tail for one row tile's correction across K slices.
        uint32_t capacity = (rowsPerPair + 1U) / 2U;
        auto correction = (__ubuf__ P*)rms[BasicApiAlign(capacity, MHC_PRE_BASIC_API_VL_FP32)].GetPhyAddr();
        for (uint32_t localRow = 0U; localRow < rows; localRow += kSharedXTileRows) {
            uint32_t tileRows = AscendC::Std::min(kSharedXTileRows, rows - localRow);
            auto tileRms = rms[localRow];
            uint32_t firstK = AscendC::Std::min(MHC_PRE_RMS_K_BLOCK, kSize);
            StageSharedXTile<true>(tileRms, correction, rowStart + localRow, tileRows, 0U, firstK);
            for (uint32_t k = firstK; k < kSize; k += MHC_PRE_RMS_K_BLOCK) {
                uint32_t columns = AscendC::Std::min(MHC_PRE_RMS_K_BLOCK, kSize - k);
                StageSharedXTile<false>(tileRms, correction, rowStart + localRow, tileRows, k, columns);
            }
        }

        // All AIVs, including idle halves, publish the shared rows before any Cube reads X.
        constexpr uint16_t kSharedXVisible = 10U;
        CrossCoreSetFlag<0, PIPE_MTE3>(kSharedXVisible);
        CrossCoreWaitFlag<0, PIPE_MTE3>(kSharedXVisible);
        if (aicIndex < vector_.tiling_->cubeBlockDimK) {
            CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_READY_FLAG);
        }

        if (rows != 0U) {
            PipeBarrier<PIPE_V>();
            vector_.VFDoV0ProcessInvRms((__ubuf__ P*)rms.GetPhyAddr(), rows, vector_.scaleMean_,
                                        vector_.matrixInfo_.normEps);
            auto ready = static_cast<event_t>(vector_.pipe_->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(ready);
            WaitFlag<HardEvent::V_MTE3>(ready);
            DataCopyExtParams copy{1U, rows * static_cast<uint32_t>(sizeof(P)), 0U, 0U, 0U};
            DataCopyPad(partialRmsGm_[rowStart], rms, copy);
        }
    }

    __aicore__ inline void ProcessAicWithWeightReuse(uint32_t mIndex, uint32_t kIndex, uint32_t kStart, uint32_t kEnd)
    {
        uint32_t stride = vector_.tiling_->cubeBlockDimM * 128U;
        // Match the projection-partial size used to allocate workspace.
        uint32_t partialK =
            MhcPreMkWeightReusePartialK(vector_.totalLength_, vector_.tiling_->multCoreSplitKSize,
                                        vector_.implMode_ == MHC_PRE_IMPL_MODE_FP32, RESI_MODE == MHC_PRE_HAS_RESI);
        uint32_t sequence = 0;
        SetFlag<HardEvent::FIX_M>(EVENT_ID2);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID5);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID6);
        for (uint32_t chunkBase = mIndex * 128U; chunkBase < vector_.totalLength_; chunkBase += 16U * stride) {
            uint32_t stages =
                AscendC::Std::min(uint64_t(16), BasicApiCeilDiv(vector_.totalLength_ - chunkBase, stride));
            for (uint32_t kOffset = kStart; kOffset < kEnd; kOffset += 256U) {
                bool firstPartialK = (kOffset - kStart) % partialK == 0U;
                bool lastPartialK = (kOffset - kStart + 256U) % partialK == 0U || kOffset + 256U == kEnd;
                if (firstPartialK) {
                    WaitFlag<HardEvent::FIX_M>(EVENT_ID2);
                }
                // FP32 and HF32 stream completed M blocks into FIX on the final local K tile.
                // Previous partial groups keep their original bulk-write schedule.
                bool immediateFinalFix = lastPartialK && kOffset + 256U >= kEnd;
                uint32_t partialIndex = 0U;
                if (lastPartialK) {
                    uint32_t perSplit = BasicApiCeilDiv(vector_.tiling_->multCoreSplitKSize, partialK);
                    partialIndex = kIndex * perSplit + (kOffset - kStart) / partialK;
                }
                mmService_.LoadReusableB(vector_.matrixInfo_.nD, vector_.matrixInfo_.fusionSize,
                                         vector_.phiGm_[kOffset], bL1_);
                for (uint32_t stage = 0; stage < stages; ++stage) {
                    uint32_t mStart = chunkBase + stage * stride;
                    uint32_t mReal = AscendC::Std::min(uint64_t(128), vector_.totalLength_ - mStart);
                    MmadStagedX(mReal, vector_.matrixInfo_.fusionSize, stage, firstPartialK, lastPartialK, sequence);
                    if (immediateFinalFix) {
                        // This stage's L0C is final. The next stage uses a disjoint L0C region,
                        // so FIX may drain this result while M computes subsequent M blocks.
                        WaitForMmad();
                        uint64_t offset =
                            (uint64_t(partialIndex) * vector_.totalLength_ + mStart) * vector_.matrixInfo_.fusionSize;
                        mmService_.CopyOutMBlock(mReal, vector_.matrixInfo_.fusionSize, stage, partialMmGm_[offset]);
                    }
                    ++sequence;
                }
                mmService_.ReleaseReusableB();
                if (lastPartialK && !immediateFinalFix) {
                    WaitForMmad();
                    for (uint32_t stage = 0; stage < stages; ++stage) {
                        uint32_t mStart = chunkBase + stage * stride;
                        uint32_t mReal = AscendC::Std::min(uint64_t(128), vector_.totalLength_ - mStart);
                        uint64_t offset =
                            (uint64_t(partialIndex) * vector_.totalLength_ + mStart) * vector_.matrixInfo_.fusionSize;
                        mmService_.CopyOutMBlock(mReal, vector_.matrixInfo_.fusionSize, stage, partialMmGm_[offset]);
                    }
                }
                if (lastPartialK) {
                    // Protect all L0C stages before a new partial/chunk or the final global barrier.
                    SetFlag<HardEvent::FIX_M>(EVENT_ID2);
                }
            }
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID2);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID6);
    }

    __aicore__ inline void ProcessAivWithWeightReuse(uint32_t mIndex, uint32_t kStart, uint32_t kEnd)
    {
        uint32_t stride = vector_.tiling_->cubeBlockDimM * 128U;
        uint32_t sequence = 0;
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        for (uint32_t chunkBase = mIndex * 128U; chunkBase < vector_.totalLength_; chunkBase += 16U * stride) {
            uint32_t stages =
                AscendC::Std::min(uint64_t(16), BasicApiCeilDiv(vector_.totalLength_ - chunkBase, stride));
            // Flatten the K-outer/M-inner stream so the next X tile is prefetched before VF.
            uint32_t streamTiles = ((kEnd - kStart) / 256U) * stages;
            LocalTensor<P> gamma;
            __ubuf__ P* gammaAddr = nullptr;
            for (uint32_t tile = 0; tile < streamTiles; ++tile) {
                uint32_t stage = tile % stages;
                uint32_t kOffset = kStart + (tile / stages) * 256U;
                uint32_t mStart = chunkBase + stage * stride;
                uint32_t mReal = AscendC::Std::min(uint64_t(128), vector_.totalLength_ - mStart);
                uint32_t firstRows = (mReal + 1U) / 2U;
                uint32_t rowStart = GetSubBlockIdx() == 0 ? 0U : firstRows;
                uint32_t rowCount = GetSubBlockIdx() == 0 ? firstRows : mReal - firstRows;
                if (tile == 0 && rowCount != 0U) {
                    PrefetchX(mStart, rowStart, rowCount, kOffset, MHC_PRE_RMS_K_BLOCK);
                }
                WaitForXSlot();
                LocalTensor<T> currentXLocal;
                if (rowCount != 0U) {
                    currentXLocal = vector_.xInQueue_.template DeQue<T>();
                }
                if (tile + 1U < streamTiles) {
                    uint32_t nextStage = (tile + 1U) % stages;
                    uint32_t nextM = chunkBase + nextStage * stride;
                    uint32_t nextReal = AscendC::Std::min(uint64_t(128), vector_.totalLength_ - nextM);
                    uint32_t nextFirst = (nextReal + 1U) / 2U;
                    uint32_t nextRows = GetSubBlockIdx() == 0 ? nextFirst : nextReal - nextFirst;
                    if (nextRows != 0U) {
                        PrefetchX(nextM, GetSubBlockIdx() == 0 ? 0U : nextFirst, nextRows,
                                  kStart + ((tile + 1U) / stages) * MHC_PRE_RMS_K_BLOCK, MHC_PRE_RMS_K_BLOCK);
                    }
                }
                if (vector_.hasGamma_ && stage == 0U) {
                    gamma = vector_.LoadGammaTile(kOffset);
                    gammaAddr = (__ubuf__ P*)gamma.GetPhyAddr();
                }
                if (rowCount != 0U) {
                    event_t nzEvent = sequence % 2U == 0 ? EVENT_ID0 : EVENT_ID1;
                    uint32_t nzSlot = sequence % 2U;
                    auto rms = rmsTileBuffer_.template Get<P>()[nzSlot * kRmsRowsPerTile];
                    StageXToL1<XNzInput::InputWithRms>(currentXLocal, {}, rms, gammaAddr, rowCount, mReal,
                                                       MHC_PRE_RMS_K_BLOCK, rowStart, nzSlot, sequence % 3U, nzEvent);
                    StoreRmsPartial(rms, rowCount, static_cast<uint64_t>(mStart) + rowStart, kOffset);
                    SetFlag<HardEvent::MTE3_V>(nzEvent);
                }
                if (vector_.hasGamma_ && stage + 1U == stages) {
                    vector_.gammaInQueue_.FreeTensor(gamma);
                }
                PublishXReady();
                ++sequence;
            }
        }
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }

    template <bool SHARED_X>
#else
    {
        return vector_.tiling_->mkUseGmStage != 0U;
    }
#endif

    __aicore__ inline void ProcessAic(uint32_t mStart, uint32_t mReal, uint32_t kIndex, uint32_t kStart, uint32_t kEnd)
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        bool useSequentialPartials = UseSequentialKPartials();
        const uint32_t fine = MhcPreMkFineOrdinaryProjectionK(vector_.totalLength_, vector_.matrixInfo_.nD,
                                                              vector_.tiling_->multCoreSplitKSize,
                                                              vector_.tiling_->mkUseGmStage, vector_.implMode_);
        uint64_t partialK = fine != 0U ? fine :
                                         (useSequentialPartials ? MHC_PRE_MK_SEQUENTIAL_PARTIAL_K :
                                                                  static_cast<uint64_t>(kEnd - kStart));
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
        mmService_.template ProcessKRange<SHARED_X>(
            mReal, vector_.matrixInfo_.fusionSize, vector_.matrixInfo_.nD, mReal, baseK, kStart, kEnd,
            vector_.tiling_->kL1Size, useSequentialPartials, partialK, UseGmStage(), stageSlotElements, stageCoreOffset,
            vector_.phiGm_, xStageGm_, partialMmGm_[partialBaseOffset], partialGroupStride, aL1_, bL1_);
    }
    template <bool HAS_GAMMA>
    __aicore__ inline void StageOrdinaryMkInput(__ubuf__ P* dst, __ubuf__ T* x, __ubuf__ P* gamma, __ubuf__ P* rms,
                                                __ubuf__ P* correction, uint32_t rows, uint32_t columns,
                                                bool firstKTile)
    {
        if (rms == nullptr) {
            vector_.template VFDoV0ProcessXIn<HAS_GAMMA, true, false, false>(dst, nullptr, x, gamma, rows, columns);
        } else if (firstKTile) {
            vector_.template VFDoV0ProcessXIn<HAS_GAMMA, true, false, true, true>(dst, rms, x, gamma, rows, columns, 0U,
                                                                                  correction);
        } else {
            vector_.template VFDoV0ProcessXIn<HAS_GAMMA, false, false, true, true>(dst, rms, x, gamma, rows, columns,
                                                                                   0U, correction);
        }
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
                WaitForXSlot();
                PublishXReady();
            }
            return;
        }
        LocalTensor<P> rmsLocal;
        __ubuf__ P* rms = nullptr;
        __ubuf__ P* correction = nullptr;
        const uint32_t alignedRows = BasicApiRoundUp<P>((vector_.tiling_->mL1Size + 1U) / 2U);
        if (vector_.tiling_->mkWorkspaceRmsOffset != 0U) {
            rmsLocal = rmsTileBuffer_.template Get<P>();
            rms = (__ubuf__ P*)rmsLocal.GetPhyAddr();
            correction = rms + alignedRows;
        }
        if (!UseGmStage()) {
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            if (UseFusedNzStage()) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
            }
        }
        uint32_t firstK =
            AscendC::Std::min(static_cast<uint64_t>(vector_.tiling_->kL1Size), static_cast<uint64_t>(kEnd - kStart));
        PrefetchX(mStart, rowStart, rowCount, kStart, firstK);
        if (vector_.hasGamma_) {
            vector_.gammaUb_ = vector_.gammaInQueue_.template AllocTensor<P>();
            vector_.DataCopyGamma(firstK, kStart);
        }

        for (uint32_t kOffset = kStart; kOffset < kEnd; kOffset += vector_.tiling_->kL1Size) {
            uint32_t currentK = AscendC::Std::min(static_cast<uint64_t>(vector_.tiling_->kL1Size),
                                                  static_cast<uint64_t>(kEnd - kOffset));
            WaitForXSlot();

            LocalTensor<T> currentXLocal = vector_.xInQueue_.template DeQue<T>();
            LocalTensor<P> xFloatLocal;
            if (!UseFusedNzStage()) {
                xFloatLocal = vector_.outQueue_.template AllocTensor<P>();
            }
            LocalTensor<P> currentGammaLocal;
            if (vector_.hasGamma_) {
                currentGammaLocal = vector_.gammaInQueue_.template DeQue<P>();
            }

            // Prefetch the next X/gamma tile before processing the current tile.
            uint32_t nextKOffset = kOffset + vector_.tiling_->kL1Size;
            if (nextKOffset < kEnd) {
                uint32_t nextK = AscendC::Std::min(static_cast<uint64_t>(vector_.tiling_->kL1Size),
                                                   static_cast<uint64_t>(kEnd - nextKOffset));
                PrefetchX(mStart, rowStart, rowCount, nextKOffset, nextK);
                if (vector_.hasGamma_) {
                    vector_.gammaUb_ = vector_.gammaInQueue_.template AllocTensor<P>();
                    vector_.DataCopyGamma(nextK, nextKOffset);
                }
            }

            if (UseFusedNzStage()) {
                event_t nzEvent = aL1BufferId == 0 ? EVENT_ID0 : EVENT_ID1;
                __ubuf__ P* gamma = vector_.hasGamma_ ? (__ubuf__ P*)currentGammaLocal.GetPhyAddr() : nullptr;
                if (rms == nullptr) {
                    StageXToL1<XNzInput::Input>(currentXLocal, {}, {}, gamma, rowCount, mReal, currentK, rowStart,
                                                aL1BufferId, aL1BufferId, nzEvent);
                } else {
                    StageXToL1<XNzInput::InputWithRmsPartials>(currentXLocal, {}, rmsLocal, gamma, rowCount, mReal,
                                                               currentK, rowStart, aL1BufferId, aL1BufferId, nzEvent,
                                                               correction, kOffset == kStart);
                }
                if (vector_.hasGamma_) {
                    vector_.gammaInQueue_.FreeTensor(currentGammaLocal);
                }
                SetFlag<HardEvent::MTE3_V>(nzEvent);
            } else {
                if (vector_.hasGamma_) {
                    StageOrdinaryMkInput<true>((__ubuf__ P*)xFloatLocal.GetPhyAddr(),
                                               (__ubuf__ T*)currentXLocal.GetPhyAddr(),
                                               (__ubuf__ P*)currentGammaLocal.GetPhyAddr(), rms, correction, rowCount,
                                               currentK, kOffset == kStart);
                    vector_.gammaInQueue_.FreeTensor(currentGammaLocal);
                } else {
                    StageOrdinaryMkInput<false>((__ubuf__ P*)xFloatLocal.GetPhyAddr(),
                                                (__ubuf__ T*)currentXLocal.GetPhyAddr(), nullptr, rms, correction,
                                                rowCount, currentK, kOffset == kStart);
                }
                vector_.xInQueue_.FreeTensor(currentXLocal);

                if (UseGmStage()) {
                    // Alternate two per-AIC GM slots to keep the AIV-to-GM-to-L1-to-Cube pipeline in flight.
                    vector_.outQueue_.template EnQue<P>(xFloatLocal);
                    xFloatLocal = vector_.outQueue_.template DeQue<P>();
                    DataCopyExtParams xCopyParams;
                    xCopyParams.blockCount = static_cast<uint16_t>(rowCount);
                    xCopyParams.blockLen = currentK * sizeof(P);
                    // UB rows already use the 32-byte stride implied by DataCopyPad.
                    xCopyParams.srcStride = 0U;
                    xCopyParams.dstStride = (vector_.tiling_->kL1Size - currentK) * sizeof(P);
                    uint64_t stageSlotElements =
                        static_cast<uint64_t>(vector_.tiling_->mL1Size) * vector_.tiling_->kL1Size;
                    uint64_t stageOffset = (2U * (GetBlockIdx() / 2U) + aL1BufferId) * stageSlotElements +
                                           static_cast<uint64_t>(rowStart) * vector_.tiling_->kL1Size;
                    DataCopyPad(xStageGm_[stageOffset], xFloatLocal, xCopyParams);
                    vector_.outQueue_.FreeTensor(xFloatLocal);
                } else {
                    constexpr event_t eventId = EVENT_ID0;
                    StageXToL1<XNzInput::Float>({}, xFloatLocal, {}, nullptr, rowCount, mReal, currentK, rowStart, 0U,
                                                aL1BufferId, eventId);
                    SetFlag<HardEvent::MTE3_V>(eventId);
                }
            }
            PublishXReady();
            aL1BufferId ^= 1U;
        }
        if (!UseGmStage()) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            if (UseFusedNzStage()) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            }
        }
        if (rms != nullptr) {
            const auto ready = static_cast<event_t>(vector_.pipe_->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(ready);
            WaitFlag<HardEvent::V_MTE3>(ready);
            const uint64_t offset = static_cast<uint64_t>(kIndex) * vector_.totalLength_ + mStart + rowStart;
            const uint64_t correctionOffset =
                static_cast<uint64_t>(vector_.tiling_->cubeBlockDimK) * vector_.totalLength_;
            DataCopyExtParams copyParams{1U, rowCount * static_cast<uint32_t>(sizeof(P)), 0U, 0U, 0U};
            DataCopyPad(partialRmsGm_[offset], rmsLocal, copyParams);
            DataCopyPad(partialRmsGm_[correctionOffset + offset], rmsLocal[alignedRows], copyParams);
            // Complete both writes before the Part1 barrier and UB reuse.
            const auto written = static_cast<event_t>(vector_.pipe_->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(written);
            WaitFlag<HardEvent::MTE3_V>(written);
        }
    }
#else
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
        uint32_t rowStart;
        uint32_t rowCount;
        if constexpr (MHC_PRE_AIV_PER_AIC == 1) {
            rowStart = 0U;
            rowCount = mReal;
        } else {
            uint32_t firstRows = (mReal + vector_.kHalfSplitDivisor - 1U) / vector_.kHalfSplitDivisor;
            rowStart = GetSubBlockIdx() == 0 ? 0U : firstRows;
            rowCount = GetSubBlockIdx() == 0 ? firstRows : mReal - firstRows;
        }
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
                    (__ubuf__ P*)xFloatLocal.GetPhyAddr(), (__ubuf__ P*)vector_.invRmsUb_.GetPhyAddr(),
                    (__ubuf__ T*)currentXLocal.GetPhyAddr(), nullptr, rowCount, currentK);
            } else {
                vector_.template VFDoV0ProcessXIn<false, false>(
                    (__ubuf__ P*)xFloatLocal.GetPhyAddr(), (__ubuf__ P*)vector_.invRmsUb_.GetPhyAddr(),
                    (__ubuf__ T*)currentXLocal.GetPhyAddr(), nullptr, rowCount, currentK);
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
                uint32_t mixCoreIndex = GetBlockIdx();
                mixCoreIndex /= MHC_PRE_AIV_PER_AIC;
                uint64_t stageOffset = (vector_.kDoubleBufferCount * mixCoreIndex + aL1BufferId) * stageSlotElements +
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
                uint32_t dstRowOffset = rowStart * (MHC_PRE_BASIC_API_BLOCK_SIZE / sizeof(P));
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
#endif

private:
#if defined(MHC_PRE_ENABLE_ASCEND950)
    event_t waveNzEvent_;
    event_t waveDrainEvent_;
    event_t wavePartialFreeEvent_;
    event_t tailNzFreeEvent_;
    event_t tailXReadyEvent_;
    MhcPreVectorCompute<T, P, RESI_MODE> post_;
    TBuf<TPosition::VECCALC> rmsTileBuffer_;
    TBuf<TPosition::VECCALC> wavePartialBuf_;
    TBuf<TPosition::VECCALC> waveMmBuf_;
#endif
    MhcPreVectorCompute<T, P, RESI_MODE> vector_;
    GlobalTensor<P> partialMmGm_;
    GlobalTensor<P> partialRmsGm_;
    GlobalTensor<P> xStageGm_;
    TBuf<TPosition::A1> l1Buffer_;
    TBuf<TPosition::VECCALC> nd2NzBuf_;
    LocalTensor<P> aL1_;
    LocalTensor<P> bL1_;
    LocalTensor<P> nd2NzLocal_;
#if defined(MHC_PRE_ENABLE_ASCEND950)
    uint32_t nzSlotElements_ = 0;
    uint32_t tailSplitHeads_ = 3U;
#endif
    MhcPreCubeCompute mmService_;
};

#if defined(MHC_PRE_ENABLE_ASCEND950)
template <class T, class P, int8_t RESI_MODE, bool WEIGHT_REUSE>
#else
template <class T, class P, int8_t RESI_MODE>
#endif
class MhcPreMKPart2 {
public:
    __aicore__ inline MhcPreMKPart2() = default;

    __aicore__ inline void Init(InitParams initParams)
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        vector_.BindGlobalTensors(initParams);
        vector_.InitFromTilingData(initParams.tilingData);
        if (WEIGHT_REUSE) {
            vector_.xGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_PERSISTENT);
        }
        vector_.InitMNConfig();
        vector_.pipe_ = initParams.tPipeIn;
        partialMmGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ P*>(initParams.workspace + vector_.tiling_->mkWorkspaceMmOffset));
        if constexpr (WEIGHT_REUSE) {
            partialRmsGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ P*>(initParams.workspace + vector_.tiling_->mkWorkspaceFinalOffset));
        } else if (vector_.tiling_->mkUseGmStage == MHC_PRE_MK_SHARED_X_STAGE) {
            uint32_t xElements = BasicApiRoundUp<P>(vector_.totalLength_ * vector_.matrixInfo_.nD);
            auto stage = reinterpret_cast<__gm__ P*>(initParams.workspace + vector_.tiling_->mkWorkspaceFinalOffset);
            partialRmsGm_.SetGlobalBuffer(stage + xElements);
        } else if (vector_.tiling_->mkWorkspaceRmsOffset != 0U) {
            partialRmsGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ P*>(initParams.workspace + vector_.tiling_->mkWorkspaceRmsOffset));
        }
        if (vector_.outFlag_) {
            vector_.hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.h_mix));
        }
        vector_.InitPart2UbBuffers();

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
#else
    {
        vector_.BindGlobalTensors(initParams);
        vector_.InitFromTilingData(initParams.tilingData);
        vector_.InitMNConfig();
        vector_.pipe_ = initParams.tPipeIn;
        partialMmGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ P*>(initParams.workspace + vector_.tiling_->mkWorkspaceMmOffset));
        partialRmsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ P*>(initParams.workspace + vector_.tiling_->mkWorkspaceRmsOffset));
        if (vector_.outFlag_) {
            vector_.hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P*>(initParams.h_mix));
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
#endif

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
#if !defined(MHC_PRE_ENABLE_ASCEND950)
    static constexpr uint32_t kPart2MaxRows = 32U;
#endif
    static constexpr uint32_t kPart2RowAlign = 8U;
#if defined(MHC_PRE_ENABLE_ASCEND950)
    static constexpr uint32_t kPart2PartialBufferElements = MHC_PRE_MK_PART2_INPUT_BYTES / sizeof(P);
#else
    static constexpr uint32_t kPart2PartialBufferElements = 80U * 1024U / sizeof(P);
#endif

    __aicore__ inline bool UseSequentialKPartials() const
    {
#if defined(MHC_PRE_ENABLE_ASCEND950)
        if constexpr (WEIGHT_REUSE) {
            const uint32_t splitK = vector_.tiling_->multCoreSplitKSize;
            return MhcPreMkWeightReusePartialK(vector_.totalLength_, splitK,
                                               vector_.implMode_ == MHC_PRE_IMPL_MODE_FP32,
                                               RESI_MODE == MHC_PRE_HAS_RESI) < splitK;
        }
        if (MhcPreMkFineOrdinaryProjectionK(vector_.totalLength_, vector_.matrixInfo_.nD,
                                            vector_.tiling_->multCoreSplitKSize, vector_.tiling_->mkUseGmStage,
                                            vector_.implMode_) != 0U) {
            return true;
        }
#endif
        return vector_.tiling_->multCoreSplitKSize >= MHC_PRE_MK_SEQUENTIAL_PARTIAL_THRESHOLD;
    }

#if defined(MHC_PRE_ENABLE_ASCEND950)
    __aicore__ inline uint32_t GetReduceRows(uint32_t remainingRows, uint32_t mmGroupK, uint32_t fusionSize,
                                             uint32_t rmsGroups) const
    {
        uint32_t rows = AscendC::Std::min(remainingRows, MHC_PRE_MK_PART2_ROWS);
        if (rows > kPart2RowAlign) {
            rows = rows / kPart2RowAlign * kPart2RowAlign;
        }
        while (rows > 0U) {
            uint32_t mmElements = mmGroupK * BasicApiRoundUp<P>(rows * fusionSize);
            uint32_t rmsElements = rmsGroups * BasicApiRoundUp<P>(rows);
            if (mmElements + rmsElements + MHC_PRE_BASIC_API_VL_FP32 <= kPart2PartialBufferElements) {
                return rows;
            }
            rows -= rows > kPart2RowAlign ? kPart2RowAlign : 1U;
        }
        return 1U;
    }
#else
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
#endif
    __aicore__ inline void ReduceAndPostProcess()
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        bool useSequentialKPartials = UseSequentialKPartials();
        const uint32_t fine = WEIGHT_REUSE ?
                                  0U :
                                  MhcPreMkFineOrdinaryProjectionK(vector_.totalLength_, vector_.matrixInfo_.nD,
                                                                  vector_.tiling_->multCoreSplitKSize,
                                                                  vector_.tiling_->mkUseGmStage, vector_.implMode_);
        const uint32_t ordinaryPartialK = fine != 0U ? fine : MHC_PRE_MK_SEQUENTIAL_PARTIAL_K;
        uint32_t groupK = useSequentialKPartials ? BasicApiCeilDiv(vector_.matrixInfo_.nD, ordinaryPartialK) :
                                                   vector_.tiling_->cubeBlockDimK;
        if (WEIGHT_REUSE && useSequentialKPartials) {
            const uint32_t splitK = vector_.tiling_->multCoreSplitKSize;
            const uint32_t partialK =
                MhcPreMkWeightReusePartialK(vector_.totalLength_, splitK, vector_.implMode_ == MHC_PRE_IMPL_MODE_FP32,
                                            RESI_MODE == MHC_PRE_HAS_RESI);
            groupK = vector_.tiling_->cubeBlockDimK * BasicApiCeilDiv(splitK, partialK);
        }
        uint32_t fusionSize = vector_.matrixInfo_.fusionSize;
        uint32_t rmsGroups = WEIGHT_REUSE ? vector_.matrixInfo_.nD / MHC_PRE_RMS_K_BLOCK : 0U;
        if (!WEIGHT_REUSE && vector_.tiling_->mkWorkspaceRmsOffset != 0U) {
            rmsGroups = 2U * vector_.tiling_->cubeBlockDimK;
        }
        bool useSharedX = vector_.tiling_->mkUseGmStage == MHC_PRE_MK_SHARED_X_STAGE;
        bool fp32Reuse = WEIGHT_REUSE && vector_.implMode_ == MHC_PRE_IMPL_MODE_FP32;
        bool useSequentialMmReduce = useSequentialKPartials || vector_.implMode_ == MHC_PRE_IMPL_MODE_HF32 || fp32Reuse;
        bool useFlatSequentialReduce =
            (useSequentialKPartials || fp32Reuse || vector_.implMode_ == MHC_PRE_IMPL_MODE_HF32) &&
            (vector_.totalLength_ >= MHC_PRE_MK_FLAT_REDUCE_MIN_M || groupK >= MHC_PRE_MK_FLAT_REDUCE_MIN_GROUP_K);
        const bool compensate =
            groupK > 1U &&
            (WEIGHT_REUSE ? ((RESI_MODE == MHC_PRE_HAS_RESI && vector_.totalLength_ <= MHC_PRE_MK_FINE_REUSE_MAX_M) ||
                             (fp32Reuse && vector_.totalLength_ <= MHC_PRE_MK_PRECISE_PROJECTION_MAX_M)) :
                            (useSharedX || vector_.totalLength_ <= MHC_PRE_MK_FINE_ORDINARY_MAX_M));
        // Flatten large ordered partial groups to remove nested per-row VF setup without changing sum order.
        for (uint32_t offsetT = vector_.vectorOffset_.offsetMStart; offsetT < vector_.vectorOffset_.offsetMEnd;) {
            uint32_t lenT = GetReduceRows(vector_.vectorOffset_.offsetMEnd - offsetT, groupK, fusionSize, rmsGroups);
            uint64_t row = vector_.globalOffsetM_ + offsetT;
            uint32_t rmsOffset = groupK * BasicApiRoundUp<P>(lenT * fusionSize);
            LocalTensor<P> partialLocal = vector_.xInQueue_.template AllocTensor<P>();
            if (WEIGHT_REUSE || rmsGroups != 0U) {
                MhcPreMkCopyRmsPartials(partialLocal[rmsOffset], partialRmsGm_[row], rmsGroups, lenT,
                                        vector_.totalLength_);
            } else if (useSharedX) {
                MhcPreMkCopyRmsPartials(vector_.invRmsUb_, partialRmsGm_[row], 1U, lenT, vector_.totalLength_);
            } else {
                vector_.ComputeFullRowRms(vector_.invRmsUb_, partialLocal, kPart2PartialBufferElements * sizeof(P), row,
                                          lenT);
            }

            DataCopyExtParams mmCopyParams;
            mmCopyParams.blockCount = groupK;
            mmCopyParams.blockLen = lenT * fusionSize * sizeof(P);
            mmCopyParams.srcStride = (vector_.totalLength_ - lenT) * fusionSize * sizeof(P);
            mmCopyParams.dstStride = 0;
            DataCopyPadExtParams<P> mmPadParams{true, 0, 0, 0};
            uint64_t mmSourceOffset = (vector_.globalOffsetM_ + offsetT) * fusionSize;
            DataCopyPad(partialLocal, partialMmGm_[mmSourceOffset], mmCopyParams, mmPadParams);

            vector_.xInQueue_.template EnQue<P>(partialLocal);
            partialLocal = vector_.xInQueue_.template DeQue<P>();
            if constexpr (WEIGHT_REUSE) {
                MhcPreMkReduceRms(vector_.invRmsUb_, partialLocal[rmsOffset], rmsGroups, lenT, vector_.scaleMean_,
                                  vector_.matrixInfo_.normEps);
            } else if (rmsGroups != 0U) {
                asc_vf_call<MhcPreMkReduceOrdinaryRms>((__ubuf__ float*)vector_.invRmsUb_.GetPhyAddr(),
                                                       (__ubuf__ float*)partialLocal[rmsOffset].GetPhyAddr(),
                                                       vector_.tiling_->cubeBlockDimK, lenT, BasicApiRoundUp<P>(lenT),
                                                       static_cast<float>(vector_.matrixInfo_.nD), vector_.scaleMean_,
                                                       vector_.matrixInfo_.normEps);
            }
            LocalTensor<P> accumulator = vector_.outQueue_.template AllocTensor<P>();
            MhcPreMkReduceMm(accumulator, partialLocal, groupK, lenT, fusionSize, useSequentialMmReduce,
                             useFlatSequentialReduce || compensate, compensate);
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
            vector_.template AIV1PostProcessTile<true>(accumulator, offsetT, lenT, 0U, fusionSize);
            vector_.outQueue_.FreeTensor(accumulator);
            vector_.DataCopyOutInvRmsUb(lenT, offsetT);
            // The next batch reuses this UB slot.
            auto rmsCopied = static_cast<event_t>(vector_.pipe_->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(rmsCopied);
            WaitFlag<HardEvent::MTE3_V>(rmsCopied);
            if (useSharedX) {
                auto rmsCopyIn = static_cast<event_t>(vector_.pipe_->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(rmsCopyIn);
                WaitFlag<HardEvent::MTE3_MTE2>(rmsCopyIn);
            }
            offsetT += lenT;
        }
    }
#else
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
#endif

private:
    MhcPreVectorCompute<T, P, RESI_MODE> vector_;
    GlobalTensor<P> partialMmGm_;
    GlobalTensor<P> partialRmsGm_;
    bool selected_ = false;
};

} // namespace MhcPre

#endif // MHC_PRE_M_K_SPLIT_H
