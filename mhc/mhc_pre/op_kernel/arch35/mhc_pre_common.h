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
 * \file mhc_pre_common.h
 * \brief Shared kernel types, constants, and hardware helpers for MHC Pre
 */

#ifndef MHC_PRE_COMMON_H
#define MHC_PRE_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

#if !defined(MHC_PRE_ENABLE_ASCEND950)
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
#define MHC_PRE_AIV_PER_AIC 1U
#else
#define MHC_PRE_AIV_PER_AIC 2U
#endif
#endif

namespace MhcPre {

using namespace AscendC;

constexpr Reg::CastTrait MHC_PRE_CAST_B16_TO_FP32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                     Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

#if !defined(MHC_PRE_ENABLE_ASCEND950)
constexpr uint32_t TWO_AIV_PER_AIC = 2U;
#endif

struct InitParams {
    GM_ADDR x;
    GM_ADDR phi;
    GM_ADDR alpha;
    GM_ADDR bias;
    GM_ADDR gamma;
    GM_ADDR hin;
    GM_ADDR h_post;
    GM_ADDR h_res;
    GM_ADDR inv_rms;
    GM_ADDR h_mix;
    GM_ADDR h_pre;
    GM_ADDR workspace;
    TPipe* tPipeIn;
    const MhcPreTilingData* tilingData;
};

struct MatrixInfo {
    uint32_t totalLength = 0;
    uint32_t nD = 0;
    uint32_t fusionSize = 0;
    float normEps = 0.0f;
    float hcEps = 0.0f;
};

struct VectorOffsetParams {
    uint64_t globalOffsetM = 0;
    uint64_t singleCoreM = 0;
    uint64_t offsetMStart = 0;
    uint64_t offsetMEnd = 0;
};

struct MNConfig {
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t k = 0;
    uint64_t singleCoreM = 0;
    uint64_t singleCoreN = 0;
    uint64_t singleCoreK = 0;
    uint64_t curSingleCoreM = 0;
    uint64_t curSingleCoreN = 0;
    uint64_t curSingleCoreK = 0;
};

// Matmul implementation modes shared by vector and Cube paths.
constexpr uint32_t MHC_PRE_IMPL_MODE_FP32 = 0U;
constexpr uint32_t MHC_PRE_IMPL_MODE_HF32 = 1U;

// Basic API hardware granularity and alignment.
constexpr uint32_t MHC_PRE_BASIC_API_BLOCK_SIZE = 32U;
constexpr uint32_t MHC_PRE_BASIC_API_C0_SIZE = 8U;
constexpr uint32_t MHC_PRE_BASIC_API_VL_FP32 = 64U;

// Shared L1 double-buffer layout for A and B operands.
constexpr uint64_t MHC_PRE_BASIC_API_L1_ALLOC_SIZE = 512U * 1024U;
constexpr uint64_t MHC_PRE_BASIC_API_L1_BUF_NUM = 2U;
constexpr uint64_t MHC_PRE_BASIC_API_L1_BUF_OFFSET = 128U * 256U;

// Basic API vector cast traits.
constexpr AscendC::Reg::CastTrait MHC_PRE_BASIC_API_CAST_B16_TO_B32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

// UB alignment and cross-core synchronization protocol.
constexpr uint32_t MHC_PRE_UB_ALIGN_SIZE = 32U;
constexpr uint8_t MHC_PRE_CROSS_CORE_SYNC_MODE = 4U;
constexpr uint16_t MHC_PRE_SUBBLOCK_FLAG_OFFSET = 16U;
constexpr uint16_t MHC_PRE_X_READY_FLAG = 8U;
constexpr uint16_t MHC_PRE_X_CONSUMED_FLAG = 9U;
constexpr uint16_t MHC_PRE_MM_READY_FLAG = 10U;
#if defined(MHC_PRE_ENABLE_ASCEND950)
constexpr uint16_t MHC_PRE_BS_PARTIAL_READY_FLAG = 11U;
constexpr uint16_t MHC_PRE_BS_PARTIAL_CONSUMED_FLAG = 12U;
#endif

// Integer shape helpers.
__aicore__ inline uint32_t BasicApiCeilDiv(uint32_t value, uint32_t div)
{
    return (value + div - 1) / div;
}

__aicore__ inline uint32_t BasicApiAlign(uint32_t value, uint32_t align)
{
    return BasicApiCeilDiv(value, align) * align;
}

template <typename T>
__aicore__ inline T MhcPreMin(T a, T b)
{
    return a > b ? b : a;
}

__aicore__ inline uint64_t MhcPreCeilDiv(uint64_t value, uint64_t div)
{
    return div == 0U ? value : (value + div - 1U) / div;
}

__aicore__ inline uint64_t MhcPreAlign(uint64_t value, uint64_t align)
{
    return MhcPreCeilDiv(value, align) * align;
}

__aicore__ inline constexpr uint32_t MhcPreGetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

template <typename T>
__aicore__ inline uint32_t BasicApiRoundUp(uint32_t value)
{
    return BasicApiAlign(value, MHC_PRE_BASIC_API_BLOCK_SIZE / sizeof(T));
}

#if !defined(MHC_PRE_ENABLE_ASCEND950)
// Register and tensor data-movement helpers.
__aicore__ inline void MhcPreBasicApiLoadBroadcast(AscendC::Reg::RegTensor<float>& dst, __local_mem__ float* src,
                                                   AscendC::Reg::MaskReg mask, uint32_t offset)
{
    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(dst, src + offset);
}
#endif

template <typename T>
__aicore__ inline void BasicApiCopyToL1(const LocalTensor<T>& srcLocal, const LocalTensor<T>& dstLocal,
                                        const DataCopyParams& copyParams)
{
    DataCopy(dstLocal, srcLocal, copyParams);
#if defined(MHC_PRE_ENABLE_ASCEND950)
}

template <bool FIRST>
__simd_vf__ inline void MhcPreAccumulateBsPartialImpl(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count,
                                                      uint16_t loops)
{
    uint32_t remaining = count;
    for (uint16_t i = 0; i < loops; ++i) {
        Reg::MaskReg mask = Reg::UpdateMask<float>(remaining);
        Reg::RegTensor<float> partial;
        Reg::LoadAlign(partial, src + i * 64U);
        if constexpr (!FIRST) {
            Reg::RegTensor<float> previous;
            Reg::LoadAlign(previous, dst + i * 64U);
            Reg::Add(partial, previous, partial, mask);
        }
        Reg::StoreAlign(dst + i * 64U, partial, mask);
    }
}

template <bool FIRST>
__aicore__ inline void MhcPreAccumulateBsPartial(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                 uint32_t count)
{
    asc_vf_call<MhcPreAccumulateBsPartialImpl<FIRST>>(
        (__ubuf__ float*)dst.GetPhyAddr(), (__ubuf__ float*)src.GetPhyAddr(), count, BasicApiCeilDiv(count, 64U));
}

template <typename T>
__simd_vf__ inline void MhcPreVFCompactRowsImpl(uint16_t rowCount, uint32_t srcRowStride, uint32_t rowWidth,
                                                __ubuf__ T* dst, __ubuf__ T* src)
{
    Reg::RegTensor<T> data;
    for (uint16_t row = 0; row < rowCount; ++row) {
        Reg::Load<T>(data, src + static_cast<uint32_t>(row) * srcRowStride);
        Reg::Store<T>(dst + static_cast<uint32_t>(row) * rowWidth, data, rowWidth);
    }
#endif
}

template <typename T>
__aicore__ inline void MhcPreVFCompactRows(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                           uint16_t rowCount, uint32_t srcRowStride, uint32_t rowWidth)
{
    __ubuf__ T* dst = (__ubuf__ T*)dstLocal.GetPhyAddr();
    __ubuf__ T* src = (__ubuf__ T*)srcLocal.GetPhyAddr();
#if defined(MHC_PRE_ENABLE_ASCEND950)
    asc_vf_call<MhcPreVFCompactRowsImpl<T>>(rowCount, srcRowStride, rowWidth, dst, src);
}

constexpr uint32_t MHC_PRE_RMS_K_BLOCK = 256U;

template <bool CORRECT_BIAS>
__simd_callee__ inline void MhcPreInvSqrt(Reg::RegTensor<float>& squares, Reg::RegTensor<float>& bias,
                                          Reg::RegTensor<float>& biasResidual, Reg::MaskReg mask)
{
    Reg::RegTensor<float> one;
    Reg::RegTensor<float> root;
    Reg::Sqrt(root, bias, mask);
    Reg::Duplicate(one, 1.0f, mask);
    Reg::Div(squares, one, root, mask);

    // Compensate the separate square-root and reciprocal roundings for normal inputs.
    // Keep the original handling of zero, subnormals and non-finite inputs.
    Reg::MaskReg positiveNormal;
    Reg::MaskReg refine;
    Reg::CompareScalar<float, CMPMODE::GE>(positiveNormal, bias, 1.17549435e-38f, mask);
    Reg::CompareScalar<float, CMPMODE::LE>(refine, bias, 3.402823466e38f, positiveNormal);
    Reg::RegTensor<float> negativeProduct;
    Reg::RegTensor<float> error;
    Reg::RegTensor<float> refined;
    Reg::Mul(negativeProduct, bias, squares, refine);
    Reg::Muls(negativeProduct, negativeProduct, -1.0f, refine);
    Reg::Muls(error, negativeProduct, 1.0f, refine);
    Reg::MulAddDst(error, bias, squares, refine);
    Reg::Muls(error, error, -1.0f, refine);
    Reg::MulAddDst(one, negativeProduct, squares, refine);
    Reg::MulAddDst(one, error, squares, refine);
    if constexpr (CORRECT_BIAS) {
        Reg::Mul(negativeProduct, biasResidual, squares, refine);
        Reg::Muls(negativeProduct, negativeProduct, -1.0f, refine);
        Reg::MulAddDst(one, negativeProduct, squares, refine);
    }
    Reg::Muls(negativeProduct, squares, 0.5f, refine);
    Reg::Muls(refined, squares, 1.0f, refine);
    Reg::MulAddDst(refined, negativeProduct, one, refine);
    Reg::Select(squares, refined, squares, refine);
}

__simd_callee__ inline void MhcPreNormalizeRms(Reg::RegTensor<float>& squares, Reg::MaskReg mask, float scaleMean,
                                               float normEps)
{
    Reg::RegTensor<float> scale;
    Reg::RegTensor<float> bias;
    Reg::Duplicate(scale, scaleMean, mask);
    Reg::Duplicate(bias, normEps, mask);
    Reg::MulAddDst(bias, squares, scale, mask);
    MhcPreInvSqrt<false>(squares, bias, scale, mask);
}

__simd_callee__ inline void MhcPreAccumulateRmsLanes(Reg::RegTensor<float>& squares, Reg::RegTensor<float>& value,
                                                     Reg::MaskReg valid, Reg::MaskReg full)
{
    Reg::RegTensor<float> zero;
    Reg::Duplicate(zero, 0.0f);
    Reg::Select(value, value, zero, valid);
    Reg::MulAddDst(squares, value, value, full);
}

// Keep the low rounding error between K tiles without changing their traversal order.
template <bool NONNEGATIVE = true>
__simd_callee__ inline void MhcPreCompensatedAdd(Reg::RegTensor<float>& sum, Reg::RegTensor<float>& error,
                                                 Reg::RegTensor<float>& partial, Reg::MaskReg& mask)
{
    Reg::RegTensor<float> adjusted, next;
    Reg::Sub(adjusted, partial, error, mask);
    Reg::Add(next, sum, adjusted, mask);
    Reg::Sub(error, next, sum, mask);
    Reg::Sub(error, error, adjusted, mask);
    // A non-finite sum has no rounding residual to carry to the next K tile.
    Reg::MaskReg finite;
    Reg::CompareScalar<float, CMPMODE::LE>(finite, next, 3.402823466e38f, mask);
    Reg::RegTensor<float> zero;
    Reg::Duplicate(zero, 0.0f, mask);
    if constexpr (NONNEGATIVE) {
        Reg::Select(error, error, zero, finite);
    } else {
        Reg::MaskReg signedFinite;
        Reg::CompareScalar<float, CMPMODE::GE>(signedFinite, next, -3.402823466e38f, finite);
        Reg::Select(error, error, zero, signedFinite);
    }
    sum = next;
}

template <typename T, bool FIRST>
__simd_vf__ inline void MhcPreVFAccumulateRms(__ubuf__ float* rms, __ubuf__ float* correction, __ubuf__ T* x,
                                              uint16_t rows, uint16_t columns, uint16_t stride)
{
    Reg::MaskReg full = Reg::CreateMask<float>();
    for (uint16_t row = 0; row < rows; ++row) {
        Reg::RegTensor<float> sum, error;
        if constexpr (FIRST) {
            Reg::Duplicate(sum, 0.0f);
            Reg::Duplicate(error, 0.0f);
        } else {
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(sum, rms + row);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(error, correction + row);
        }

        for (uint16_t k = 0; k < columns; k += MHC_PRE_RMS_K_BLOCK) {
            uint32_t remaining = AscendC::Std::min(static_cast<uint32_t>(columns - k), MHC_PRE_RMS_K_BLOCK);
            Reg::RegTensor<float> squareLanes;
            Reg::Duplicate(squareLanes, 0.0f);
            for (uint16_t offset = k; remaining > 0U; offset += MHC_PRE_BASIC_API_VL_FP32) {
                Reg::MaskReg mask = Reg::UpdateMask<float>(remaining);
                Reg::RegTensor<T> xB16;
                Reg::RegTensor<float> value;
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(xB16, x + row * stride + offset);
                Reg::Cast<float, T, MHC_PRE_BASIC_API_CAST_B16_TO_B32>(value, xB16, mask);
                MhcPreAccumulateRmsLanes(squareLanes, value, mask, full);
            }
            Reg::RegTensor<float> blockSum;
            Reg::Reduce<Reg::ReduceType::SUM>(blockSum, squareLanes, full);
            MhcPreCompensatedAdd(sum, error, blockSum, full);
        }
        Reg::Store(rms + row, sum, 1U);
        Reg::Store(correction + row, error, 1U);
    }
}

// Keep the BF16 bit widening fast path and use a native FP16 conversion for half.
template <typename T>
__simd_callee__ inline void MhcPreMkLoadX128(Reg::RegTensor<float>& lo, Reg::RegTensor<float>& hi, __ubuf__ T* src,
                                             Reg::MaskReg& full)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<uint16_t> zero, packed, loBits, hiBits;
        Reg::Duplicate(zero, static_cast<uint16_t>(0));
        Reg::LoadAlign(packed, (__ubuf__ uint16_t*)src);
        Reg::Interleave<uint16_t>(loBits, hiBits, zero, packed);
        lo = reinterpret_cast<Reg::RegTensor<float>&>(loBits);
        hi = reinterpret_cast<Reg::RegTensor<float>&>(hiBits);
    } else {
        Reg::RegTensor<T> x;
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x, src);
        Reg::Cast<float, T, MHC_PRE_CAST_B16_TO_FP32>(lo, x, full);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(x, src + 64U);
        Reg::Cast<float, T, MHC_PRE_CAST_B16_TO_FP32>(hi, x, full);
    }
}

template <typename T, bool INIT_RMS, bool HAS_GAMMA, bool COMPENSATE_RMS>
__simd_vf__ inline void MhcPreMkVFPackNzImpl(uint16_t rows, uint16_t columns, __ubuf__ float* nz, __ubuf__ float* rms,
                                             __ubuf__ T* x, uint16_t nzRows, uint16_t passes, uint16_t groups,
                                             __ubuf__ float* gamma, __ubuf__ float* correction)
{
    Reg::MaskReg full = Reg::CreateMask<float>();
    Reg::RegTensor<float> indices;
    Reg::RegTensor<float> one;
    Reg::Arange(indices, 0.0f);
    Reg::Duplicate(one, 1.0f);
    for (uint16_t group = 0; group < groups; ++group) {
        uint16_t rowCount = AscendC::Std::min(static_cast<uint32_t>(rows - group * 64U), 64U);
        Reg::RegTensor<float> currentRow;
        Reg::RegTensor<float> packedSums;
        Reg::Duplicate(currentRow, 0.0f);
        Reg::Duplicate(packedSums, 0.0f);
        for (uint16_t row = 0; row < rowCount; ++row) {
            Reg::RegTensor<float> acc;
            Reg::RegTensor<float> sum;
            Reg::RegTensor<float> broadcast;
            Reg::Duplicate(acc, 0.0f);
            __ubuf__ float* dst = nz + (group * 64U + row) * 8U;
            for (uint16_t pass = 0; pass < passes; ++pass) {
                Reg::RegTensor<float> lo, hi;
                MhcPreMkLoadX128(lo, hi, x + (group * 64U + row) * columns + pass * 128U, full);
                Reg::MulAddDst(acc, lo, lo, full);
                Reg::MulAddDst(acc, hi, hi, full);
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
            Reg::Reduce<Reg::ReduceType::SUM>(sum, acc, full);
            Reg::Duplicate(broadcast, sum, full);
            Reg::MaskReg rowMask;
            Reg::Compare<float, AscendC::CMPMODE::EQ>(rowMask, indices, currentRow, full);
            Reg::Select(packedSums, broadcast, packedSums, rowMask);
            Reg::Add(currentRow, currentRow, one, full);
        }
        uint32_t count = rowCount;
        Reg::MaskReg validRows = Reg::UpdateMask<float>(count);
        auto* rmsGroup = rms + group * 64U;
        if constexpr (COMPENSATE_RMS) {
            auto* correctionGroup = correction + group * 64U;
            Reg::RegTensor<float> error;
            if constexpr (INIT_RMS) {
                Reg::Duplicate(error, 0.0f, validRows);
            } else {
                Reg::RegTensor<float> previous;
                Reg::RegTensor<uint32_t> rowIndices;
                Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(rowIndices), 0);
                Reg::Gather(previous, rmsGroup, rowIndices, validRows);
                Reg::Gather(error, correctionGroup, rowIndices, validRows);
                MhcPreCompensatedAdd(previous, error, packedSums, validRows);
                packedSums = previous;
            }
            Reg::StoreAlign(correctionGroup, error, validRows);
        } else if constexpr (!INIT_RMS) {
            Reg::RegTensor<float> previous;
            if (rowCount == MHC_PRE_BASIC_API_VL_FP32) {
                Reg::LoadAlign(previous, rmsGroup);
            } else {
                // The final RMS group may have fewer than one vector of storage.
                Reg::RegTensor<uint32_t> rowIndices;
                Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(rowIndices), 0);
                Reg::Gather(previous, rmsGroup, rowIndices, validRows);
            }
            Reg::Add(packedSums, packedSums, previous, validRows);
        }
        Reg::StoreAlign(rmsGroup, packedSums, validRows);
    }
}

template <bool INIT_RMS, bool COMPENSATE_RMS = false, typename T>
__aicore__ inline void MhcPreMkVFPackNz(const LocalTensor<float>& nzLocal, const LocalTensor<float>& rmsLocal,
                                        const LocalTensor<T>& xLocal, uint16_t rows, uint16_t columns,
                                        __ubuf__ float* gamma = nullptr, __ubuf__ float* correction = nullptr)
{
    __ubuf__ float* nz = (__ubuf__ float*)nzLocal.GetPhyAddr();
    __ubuf__ float* rms = (__ubuf__ float*)rmsLocal.GetPhyAddr();
    __ubuf__ T* x = (__ubuf__ T*)xLocal.GetPhyAddr();
    uint16_t nzRows = BasicApiAlign(rows, MHC_PRE_BASIC_API_C0_SIZE) + 1U;
    uint16_t passes = columns / 128U;
    uint16_t groups = BasicApiCeilDiv(rows, 64U);
    if (gamma != nullptr) {
        asc_vf_call<MhcPreMkVFPackNzImpl<T, INIT_RMS, true, COMPENSATE_RMS>>(rows, columns, nz, rms, x, nzRows, passes,
                                                                             groups, gamma, correction);
    } else {
        asc_vf_call<MhcPreMkVFPackNzImpl<T, INIT_RMS, false, COMPENSATE_RMS>>(rows, columns, nz, rms, x, nzRows, passes,
                                                                              groups, nullptr, correction);
    }
#else
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> data;
        for (uint16_t row = 0; row < rowCount; ++row) {
            Reg::Load<T>(data, src + static_cast<uint32_t>(row) * srcRowStride);
            Reg::Store<T>(dst + static_cast<uint32_t>(row) * rowWidth, data, rowWidth);
        }
    }
#endif
}

} // namespace MhcPre

#endif // MHC_PRE_COMMON_H
