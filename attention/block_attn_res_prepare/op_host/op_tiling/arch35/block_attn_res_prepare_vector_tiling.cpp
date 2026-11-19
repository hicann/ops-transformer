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
#include <limits>
#include <vector>

#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "../../../op_kernel/arch35/block_attn_res_prepare_apt_tiling_key.h"

namespace {

const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510),
                                               static_cast<int32_t>(NpuArch::DAV_9201)};
constexpr int32_t TILING_PRIORITY = 1000;
} // namespace

namespace optiling {
namespace {

constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t BF16_BYTES = sizeof(uint16_t);
constexpr uint64_t UB_BLOCK_BYTES = 32UL;
constexpr uint32_t FP32_REG_ELEMS = 64U;
constexpr uint32_t STAT_VECTOR_COUNT = 2U;
constexpr uint32_t ROW_STAT_BUFFER_NUM = 2U;
constexpr uint32_t SCALAR_BLOCK_ELEMS = 8U;
constexpr uint32_t STAT_UB_ELEMS = STAT_VECTOR_COUNT * (FP32_REG_ELEMS + SCALAR_BLOCK_ELEMS);
constexpr uint64_t DOUBLE_BUFFER_NUM = 2UL;
// Covers three S rows, or two tokens with two S rows each, in the kernel's grouped path.
constexpr uint32_t GROUP_STAT_BUFFER_NUM = 4U;
constexpr uint64_t MIN_S_PACK_SIZE = 4UL;
constexpr uint64_t THREE_S_QO_ROWS = 6UL;
constexpr uint64_t MAX_THREE_S_VALID_N = 5UL;
constexpr uint64_t QUERY_CACHE_S_ROWS = 3UL;
constexpr uint64_t QUERY_CACHE_OUTPUT_BUFFER_NUM = 3UL;

} // namespace

bool BlockAttnResPrepareVectorTiling::IsCapable()
{
    // Vector is the fallback template. Mix is registered with a higher priority and is considered first.
    return true;
}

bool BlockAttnResPrepareVectorTiling::CanCacheVPerToken() const
{
    if (blockResDtype_ != ge::DT_BF16 || totalD_ % FP32_REG_ELEMS != 0U) {
        return false;
    }
    // Reserve full-D V storage for all N rows because validBlocks is known only at runtime.
    // The kernel retains all validN V rows per token in UB across each work unit's S span.
    // In grouped mode, two Q rows and two O rows hold one two-row group. Two complete
    // ping-pong groups need four Q rows and four O rows, which this budget does not reserve.
    const uint64_t vBytes = totalN_ * totalD_ * BF16_BYTES;
    const uint64_t qBytes = DOUBLE_BUFFER_NUM * totalD_ * FP32_BYTES;
    const uint64_t oBytes = DOUBLE_BUFFER_NUM * totalD_ * FP32_BYTES;
    const uint64_t statBytes = GROUP_STAT_BUFFER_NUM * STAT_UB_ELEMS * FP32_BYTES;
    // Extra output rows use spare V-cache capacity. Three-S grouping has its own
    // UB limit (threeSMaxN), including three Q/O rows and statistics. Square sums stay in the first stat row.
    return vBytes + qBytes + oBytes + statBytes <= ubSize_;
}

bool BlockAttnResPrepareVectorTiling::SelectVCacheCapacity(bool hasMultipleWorkRounds)
{
    const uint64_t vElemBytes = blockResDtype_ == ge::DT_BF16 ? BF16_BYTES : FP32_BYTES;
    // Full-D rows fit for every supported D <= 8192 on DAV_3510. Keep the
    // runtime capacity check before subtracting the baseline buffer reservation.
    // A shared element pitch keeps both BLOCK_RES_DTYPE V and FP32 Q/O rows 32-byte aligned.
    const uint64_t dAlign = Ops::Base::CeilAlign<uint64_t>(totalD_, UB_BLOCK_BYTES / vElemBytes);
    const uint64_t qBufferNum = hasMultipleWorkRounds ? DOUBLE_BUFFER_NUM : 1UL;
    const uint64_t oBufferNum = qBufferNum;
    const uint64_t reservedUbBytes = (qBufferNum + oBufferNum) * dAlign * FP32_BYTES +
                                     DOUBLE_BUFFER_NUM * dAlign * vElemBytes +
                                     ROW_STAT_BUFFER_NUM * STAT_UB_ELEMS * FP32_BYTES;
    if (reservedUbBytes > ubSize_) {
        return false;
    }

    const uint64_t vRowBytes = dAlign * vElemBytes;
    const uint64_t availableVRows = (ubSize_ - reservedUbBytes) / vRowBytes;
    vCacheRows_ = static_cast<uint8_t>(std::min<uint64_t>(totalN_, availableVRows));
    return true;
}

ge::graphStatus BlockAttnResPrepareVectorTiling::DoOpTiling()
{
    const bool canCacheVPerToken = CanCacheVPerToken();
    // Start with one S column per task for each token.
    sBlockSize_ = 1U;
    if (canCacheVPerToken) {
        // Estimate the S span per task, then balance it across the S blocks.
        uint64_t sBlockSize = std::min(totalS_, Ops::Base::CeilDiv(totalT_ * totalS_, aivCoreNum_));
        const uint64_t sBlockCount = Ops::Base::CeilDiv(totalS_, sBlockSize);
        sBlockSize = Ops::Base::CeilDiv(totalS_, sBlockCount);
        if (sBlockSize >= MIN_S_PACK_SIZE) {
            sBlockSize_ = static_cast<uint32_t>(sBlockSize);
        }
    }
    const bool packS = sBlockSize_ > 1U;
    const uint64_t sBlockCount = Ops::Base::CeilDiv(totalS_, static_cast<uint64_t>(sBlockSize_));
    const uint64_t totalWorkUnits = totalT_ * sBlockCount;
    OP_CHECK_IF(totalWorkUnits > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context_->GetNodeName(), "Vector totalWorkUnits=%lu exceeds uint32 maximum=%u", totalWorkUnits,
                        std::numeric_limits<uint32_t>::max()),
                return ge::GRAPH_FAILED);
    usedCoreNum_ = static_cast<uint32_t>(std::min(aivCoreNum_, totalWorkUnits));
    OP_CHECK_IF(usedCoreNum_ == 0U,
                OP_LOGE(context_->GetNodeName(), "Vector usedCoreNum is zero: totalWorkUnits=%lu, availableCoreNum=%lu",
                        totalWorkUnits, aivCoreNum_),
                return ge::GRAPH_FAILED);
    const bool hasMultipleWorkRounds = totalWorkUnits > usedCoreNum_;

    if (packS) {
        // Extra output rows reuse spare V-cache storage. threeSMaxN separately
        // checks the full layout with three Q rows and three output rows.
        vCacheRows_ = static_cast<uint8_t>(totalN_);
    } else {
        // Single-S tasks use full D, double-buffered V and Q/O buffers sized for the work rounds.
        OP_CHECK_IF(!SelectVCacheCapacity(hasMultipleWorkRounds),
                    OP_LOGE(context_->GetNodeName(),
                            "full-D Vector buffers exceed UB capacity: T=%lu, N=%lu, S=%lu, D=%lu, usableUbBytes=%lu",
                            totalT_, totalN_, totalS_, totalD_, ubSize_),
                    return ge::GRAPH_FAILED);
    }

    // Single-S tasks select query-local or token-level V caching at runtime.
    useSingleSCache_ = canCacheVPerToken && !packS;
    SetTilingData(sBlockCount, totalWorkUnits);
    return ge::GRAPH_SUCCESS;
}

void BlockAttnResPrepareVectorTiling::SetTilingData(uint64_t sBlockCount, uint64_t totalWorkUnits)
{
    const bool packS = sBlockSize_ > 1U;
    const bool hasMultipleWorkRounds = totalWorkUnits > usedCoreNum_;

    tilingData_.totalT = static_cast<uint32_t>(totalT_);
    tilingData_.totalN = static_cast<uint8_t>(totalN_);
    tilingData_.totalS = static_cast<uint32_t>(totalS_);
    tilingData_.totalD = static_cast<uint32_t>(totalD_);
    tilingData_.usedCoreNum = static_cast<uint16_t>(usedCoreNum_);
    tilingData_.sBlockSize = sBlockSize_;
    tilingData_.vCacheRows = vCacheRows_;
    tilingData_.eps = eps_;
    tilingData_.sBlockCount = static_cast<uint32_t>(sBlockCount);
    tilingData_.workPerCore = static_cast<uint32_t>(totalWorkUnits / usedCoreNum_);
    tilingData_.remainder = static_cast<uint16_t>(totalWorkUnits % usedCoreNum_);
    tilingData_.reciprocalD = 1.0F / static_cast<float>(totalD_);
    const uint64_t vElemBytes = blockResDtype_ == ge::DT_BF16 ? BF16_BYTES : FP32_BYTES;
    tilingData_.dAlign = static_cast<uint32_t>(Ops::Base::CeilAlign<uint64_t>(totalD_, UB_BLOCK_BYTES / vElemBytes));
    tilingData_.qBufferNum = static_cast<uint8_t>(packS || hasMultipleWorkRounds ? DOUBLE_BUFFER_NUM : 1U);
    tilingData_.threeSMaxN = 0U;
    // Use the remaining usable UB instead of requiring four unused V-cache rows.
    const uint64_t threeSReservedBytes =
        THREE_S_QO_ROWS * totalD_ * FP32_BYTES + GROUP_STAT_BUFFER_NUM * STAT_UB_ELEMS * FP32_BYTES;
    if (packS && threeSReservedBytes < ubSize_) {
        tilingData_.threeSMaxN = static_cast<uint8_t>(std::min<uint64_t>(
            {totalN_, MAX_THREE_S_VALID_N, (ubSize_ - threeSReservedBytes) / (totalD_ * BF16_BYTES)}));
    }
    // Cache three Q rows while a core processes consecutive tokens. The kernel
    // selects this direction only when validN == 1. Each core needs at least
    // two tokens so Q is reused and the streamed V buffers can overlap.
    tilingData_.qCacheCoresPerGroup = 0U;
    const uint64_t sGroups = totalS_ / QUERY_CACHE_S_ROWS;
    const uint64_t cachedQOBytes = (QUERY_CACHE_S_ROWS + QUERY_CACHE_OUTPUT_BUFFER_NUM) * totalD_ * FP32_BYTES;
    const uint64_t streamedVBytes = DOUBLE_BUFFER_NUM * totalD_ * BF16_BYTES;
    const uint64_t cachedQStatBytes =
        QUERY_CACHE_OUTPUT_BUFFER_NUM * QUERY_CACHE_S_ROWS * STAT_VECTOR_COUNT * UB_BLOCK_BYTES;
    if (blockResDtype_ == ge::DT_BF16 && totalD_ % FP32_REG_ELEMS == 0U && totalS_ % QUERY_CACHE_S_ROWS == 0U &&
        sGroups <= usedCoreNum_ && usedCoreNum_ % sGroups == 0U &&
        totalT_ >= DOUBLE_BUFFER_NUM * (usedCoreNum_ / sGroups) &&
        cachedQOBytes + streamedVBytes + cachedQStatBytes <= ubSize_ &&
        totalT_ <= std::numeric_limits<uint32_t>::max() / FP32_BYTES) {
        tilingData_.qCacheCoresPerGroup = static_cast<uint16_t>(usedCoreNum_ / sGroups);
    }
}

uint64_t BlockAttnResPrepareVectorTiling::GetTilingKey() const
{
    using namespace BlockAttnResPrepareTilingKey;
    if (useSingleSCache_) {
        return GET_TPL_TILING_KEY(BLOCK_ATTN_RES_PREPARE_SINGLE_S_CACHE_TPL);
    }
    return GET_TPL_TILING_KEY(BLOCK_ATTN_RES_PREPARE_ONLY_VECTOR_TPL);
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(BlockAttnResPrepare, BlockAttnResPrepareVectorTiling, supportedNpuArch,
                                   TILING_PRIORITY);

} // namespace optiling
