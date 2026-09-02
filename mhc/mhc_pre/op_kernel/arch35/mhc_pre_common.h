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

namespace MhcPre {

using namespace AscendC;

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
    TPipe *tPipeIn;
    const MhcPreTilingData *tilingData;
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
constexpr AscendC::MicroAPI::CastTrait MHC_PRE_BASIC_API_CAST_B16_TO_B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

// UB alignment and cross-core synchronization protocol.
constexpr uint32_t MHC_PRE_UB_ALIGN_SIZE = 32U;
constexpr uint8_t MHC_PRE_CROSS_CORE_SYNC_MODE = 4U;
constexpr uint16_t MHC_PRE_SUBBLOCK_FLAG_OFFSET = 16U;
constexpr uint16_t MHC_PRE_X_READY_FLAG = 8U;
constexpr uint16_t MHC_PRE_X_CONSUMED_FLAG = 9U;
constexpr uint16_t MHC_PRE_MM_READY_FLAG = 10U;

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

// Register and tensor data-movement helpers.
__aicore__ inline void MhcPreBasicApiLoadBroadcast(AscendC::MicroAPI::RegTensor<float> &dst, __local_mem__ float *src,
                                                   AscendC::MicroAPI::MaskReg mask, uint32_t offset)
{
    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(dst, src + offset);
}

template <typename T>
__aicore__ inline void BasicApiCopyToL1(const LocalTensor<T> &srcLocal, const LocalTensor<T> &dstLocal,
                                        const DataCopyParams &copyParams)
{
    DataCopy(dstLocal, srcLocal, copyParams);
}

template <typename T>
__aicore__ inline void MhcPreVFCompactRows(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
                                           uint16_t rowCount, uint32_t srcRowStride, uint32_t rowWidth)
{
    __ubuf__ T *dst = (__ubuf__ T *)dstLocal.GetPhyAddr();
    __ubuf__ T *src = (__ubuf__ T *)srcLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> data;
        for (uint16_t row = 0; row < rowCount; ++row) {
            MicroAPI::Load<T>(data, src + static_cast<uint32_t>(row) * srcRowStride);
            MicroAPI::Store<T>(dst + static_cast<uint32_t>(row) * rowWidth, data, rowWidth);
        }
    }
}

} // namespace MhcPre

#endif // MHC_PRE_COMMON_H
