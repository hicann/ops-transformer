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
 * \file block_attn_res_prepare_vector_tiling.cpp
 * \brief Vector tiling template for BlockAttnResPrepare.
 */

#include "block_attn_res_prepare_vector_tiling.h"

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include "log/log.h"

namespace {

const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510)};
constexpr int32_t TILING_PRIORITY = 1000;
} // namespace

namespace optiling {
namespace {

constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t UB_RESERVED_BYTES = 8UL * 1024UL;
constexpr uint32_t FP32_REG_ELEMS = 64U;
constexpr uint32_t VECTOR_UB_ROW_ALIGN_ELEMS = FP32_REG_ELEMS;
constexpr uint32_t STAT_VECTOR_COUNT = 2U;
constexpr uint32_t STAT_BUFFER_NUM = 2U;
constexpr uint32_t SCALAR_BLOCK_ELEMS = 8U;
constexpr uint32_t STAT_UB_ELEMS = STAT_VECTOR_COUNT * (FP32_REG_ELEMS + SCALAR_BLOCK_ELEMS);
constexpr uint64_t DOUBLE_BUFFER_NUM = 2UL;
constexpr uint64_t VECTOR_BASE_D_LARGE = 1024UL;
constexpr uint64_t VECTOR_BASE_D_MEDIUM = 512UL;
constexpr uint64_t VECTOR_BASE_D_SMALL = 256UL;
constexpr size_t VECTOR_BASE_D_CANDIDATE_NUM = 4;

} // namespace

bool BlockAttnResPrepareVectorTiling::IsCapable()
{
    // Vector is the fallback template. Mix is registered with a higher priority and is considered first.
    return true;
}

bool BlockAttnResPrepareVectorTiling::SelectVectorBaseD(bool hasMultipleWorkRounds)
{
    const uint64_t usableUbBytes = ubSize_ - UB_RESERVED_BYTES;
    const uint64_t dAlign = AlignUp<uint64_t>(totalD_, VECTOR_UB_ROW_ALIGN_ELEMS);

    const std::array<uint64_t, VECTOR_BASE_D_CANDIDATE_NUM> candidates = {
        totalD_, std::min<uint64_t>(totalD_, VECTOR_BASE_D_LARGE), std::min<uint64_t>(totalD_, VECTOR_BASE_D_MEDIUM),
        std::min<uint64_t>(totalD_, VECTOR_BASE_D_SMALL)};
    uint64_t previous = 0;
    for (const uint64_t candidate : candidates) {
        if (candidate == 0 || candidate == previous) {
            continue;
        }
        previous = candidate;
        // LoadAlign reads one complete FP32 vector register. Keep every Q/V/O
        // row on a 64-element pitch so a tail load cannot cross into the next
        // logical UB row; invalid lanes are excluded by the VF mask.
        const uint64_t baseDAlign = AlignUp<uint64_t>(candidate, VECTOR_UB_ROW_ALIGN_ELEMS);
        const uint64_t dTileNum = CeilDiv(totalD_, candidate);
        const uint64_t qBufferNum = (dTileNum > 1 || hasMultipleWorkRounds) ? DOUBLE_BUFFER_NUM : 1UL;
        const uint64_t oBufferNum = (dTileNum > 1 || hasMultipleWorkRounds) ? DOUBLE_BUFFER_NUM : 1UL;
        OP_CHECK_IF(dTileNum > std::numeric_limits<uint64_t>::max() / totalN_,
                    OP_LOGE(context_->GetNodeName(),
                            "Vector D tile count times N overflows uint64: dTileNum=%lu, N=%lu, baseD=%lu", dTileNum,
                            totalN_, candidate),
                    return false);
        const uint64_t vLoopNum = dTileNum * totalN_;
        const uint64_t vBufferNum = (vLoopNum > 1 || hasMultipleWorkRounds) ? DOUBLE_BUFFER_NUM : 1UL;
        const uint64_t fixedElems = (qBufferNum + vBufferNum + oBufferNum) * baseDAlign +
                                    static_cast<uint64_t>(STAT_BUFFER_NUM) * STAT_UB_ELEMS;
        const uint64_t fixedBytes = fixedElems * FP32_BYTES;
        if (fixedBytes > usableUbBytes) {
            continue;
        }

        baseD_ = static_cast<uint32_t>(candidate);
        qBufferNum_ = static_cast<uint32_t>(qBufferNum);
        vBufferNum_ = static_cast<uint32_t>(vBufferNum);
        oBufferNum_ = static_cast<uint32_t>(oBufferNum);

        const uint64_t cacheRowBytes = dAlign * FP32_BYTES;
        const uint64_t cacheCapacity = (usableUbBytes - fixedBytes) / cacheRowBytes;
        vCacheRows_ = static_cast<uint32_t>(std::min<uint64_t>(totalN_, cacheCapacity));
        return true;
    }
    return false;
}

ge::graphStatus BlockAttnResPrepareVectorTiling::DoOpTiling()
{
    const uint64_t totalWorkUnits = totalT_ * totalS_;
    WorkDistribution distribution;
    if (CalculateWorkDistribution(totalWorkUnits, aivCoreNum_, "Vector", distribution) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    usedCoreNum_ = distribution.usedCoreNum;
    const bool hasMultipleWorkRounds = distribution.tailBlockFactor > 1U;

    OP_CHECK_IF(!SelectVectorBaseD(hasMultipleWorkRounds),
                OP_LOGE(context_->GetNodeName(),
                        "no Vector baseD satisfies UB capacity: T=%lu, N=%lu, S=%lu, D=%lu, usableUbBytes=%lu", totalT_,
                        totalN_, totalS_, totalD_, ubSize_ - UB_RESERVED_BYTES),
                return ge::GRAPH_FAILED);

    tilingData_.totalT = static_cast<uint32_t>(totalT_);
    tilingData_.totalN = static_cast<uint8_t>(totalN_);
    tilingData_.totalS = static_cast<uint32_t>(totalS_);
    tilingData_.totalWorkUnits = static_cast<uint32_t>(totalWorkUnits);
    tilingData_.totalD = static_cast<uint32_t>(totalD_);
    tilingData_.usedCoreNum = static_cast<uint16_t>(usedCoreNum_);
    tilingData_.bigCoreNum = static_cast<uint16_t>(distribution.bigCoreNum);
    tilingData_.blockFactor = distribution.blockFactor;
    tilingData_.tailBlockFactor = distribution.tailBlockFactor;
    tilingData_.baseD = baseD_;
    tilingData_.vCacheRows = static_cast<uint8_t>(vCacheRows_);
    tilingData_.qBufferNum = static_cast<uint8_t>(qBufferNum_);
    tilingData_.vBufferNum = static_cast<uint8_t>(vBufferNum_);
    tilingData_.oBufferNum = static_cast<uint8_t>(oBufferNum_);
    tilingData_.statUbElems = STAT_UB_ELEMS;
    tilingData_.eps = eps_;
    return ge::GRAPH_SUCCESS;
}

uint64_t BlockAttnResPrepareVectorTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(BLOCK_ATTN_RES_PREPARE_TPL_ONLY_VECTOR);
}

BlockAttnResPrepareBaseTiling::TilingDataView BlockAttnResPrepareVectorTiling::GetTilingDataView() const
{
    return {&tilingData_, sizeof(tilingData_), "Vector"};
}

void BlockAttnResPrepareVectorTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(),
            "BlockAttnResPrepare Vector tiling: T=%lu N=%lu S=%lu D=%lu cores=%u baseD=%u "
            "q/v/o buffers=%u/%u/%u vCacheRows=%u",
            totalT_, totalN_, totalS_, totalD_, usedCoreNum_, baseD_, qBufferNum_, vBufferNum_, oBufferNum_,
            vCacheRows_);
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(BlockAttnResPrepare, BlockAttnResPrepareVectorTiling, supportedNpuArch,
                                   TILING_PRIORITY);

} // namespace optiling
