/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "block_attn_res_update_full_d_tiling.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/soc_spec.h"
#include "securec.h"
#include "util/math_util.h"
#include "../../../op_kernel/arch35/block_attn_res_update_tiling_key.h"

namespace optiling {
namespace BlockAttnResUpdate {
namespace {

constexpr const char* OP_NAME = "BlockAttnResUpdate";
constexpr uint32_t FP32_ALIGN_ELEMENTS = 8U;  // 8 * 4 B = 32 B
constexpr uint32_t BF16_ALIGN_ELEMENTS = 16U; // 16 * 2 B = 32 B
constexpr uint64_t UB_ALIGN_BYTES = 32UL;
constexpr uint64_t VECTOR_REG_BYTES = 256UL;
constexpr uint32_t FP32_BYTES = 4U;
constexpr uint32_t BF16_BYTES = 2U;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2U;
// Each ping-pong buffer stores partial and numerator as two equal FP32 matrices.
constexpr uint32_t FP32_MATRIX_NUM_PER_BUFFER = 2U;
// The stats tensor contains logitMax, expSum, and score planes.
constexpr uint32_t STATS_PLANE_NUM = 3U;
// A 2-D data-copy command supports at most 4095 blocks; each T row is one block.
constexpr uint32_t MAX_DATA_COPY_BLOCK_COUNT = 4095U;
// Keep the generic full-D strategy as a fallback. A preferred split-D strategy should use a numerically smaller value.
constexpr int32_t FULL_D_TILING_PRIORITY = 1000;
const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510),
                                               static_cast<int32_t>(NpuArch::DAV_9201)};

} // namespace

bool BlockAttnResUpdateFullDTiling::IsCapable()
{
    return true;
}

ge::graphStatus BlockAttnResUpdateFullDTiling::DoOpTiling()
{
    tilingInfo_ = BlockAttnResUpdateFullDTilingInfo{};
    const uint64_t tPerCore = tSize_ / aivNum_ + (tSize_ % aivNum_ != 0U);
    OP_CHECK_IF(tPerCore > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(opName_, "T per core exceeds the uint32_t range, T=%lu, configured cores=%u, tPerCore=%lu.",
                        tSize_, aivNum_, tPerCore),
                return ge::GRAPH_FAILED);
    const uint64_t usedCoreNum = tSize_ / tPerCore + (tSize_ % tPerCore != 0U);
    OP_CHECK_IF(usedCoreNum > std::numeric_limits<uint16_t>::max(),
                OP_LOGE(opName_, "Used core count exceeds the uint16_t tiling range, cores=%lu, maximum=%u.",
                        usedCoreNum, static_cast<uint32_t>(std::numeric_limits<uint16_t>::max())),
                return ge::GRAPH_FAILED);
    const uint64_t tRemainder = tSize_ % tPerCore;
    const uint64_t lastTPerCore = tRemainder == 0U ? tPerCore : tRemainder;

    tilingInfo_.tPerCore = static_cast<uint32_t>(tPerCore);
    tilingInfo_.lastTPerCore = static_cast<uint32_t>(lastTPerCore);
    tilingInfo_.usedCoreNum = static_cast<uint32_t>(usedCoreNum);
    tilingInfo_.dAlignFp32 = Ops::Base::CeilAlign(dSize_, FP32_ALIGN_ELEMENTS);
    tilingInfo_.dAlignBf16 = Ops::Base::CeilAlign(dSize_, BF16_ALIGN_ELEMENTS);

    if (!TryUbTiling(tPerCore)) {
        return ge::GRAPH_FAILED;
    }
    // Preserve the UB-fit tile count, then distribute T evenly across those tiles to avoid a tiny final tile.
    // balancedTileT never exceeds reuseTileT, so it retains the UB-fit guarantee.
    const uint64_t reuseTileT = tilingInfo_.tileT;
    const uint64_t tileNumPerCore = (tPerCore + reuseTileT - 1U) / reuseTileT;
    const uint64_t balancedTileT = (tPerCore + tileNumPerCore - 1U) / tileNumPerCore;
    tilingInfo_.tileT = static_cast<uint32_t>(balancedTileT);
    tilingInfo_.statsTStride = Ops::Base::CeilAlign(tilingInfo_.tileT, FP32_ALIGN_ELEMENTS);
    OP_LOGD(opName_, "Select reuse-output kernel with tileNumPerCore=%lu, reuseTileT=%lu, balancedTileT=%u.",
            tileNumPerCore, reuseTileT, tilingInfo_.tileT);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t BlockAttnResUpdateFullDTiling::GetOutputNormWeightUbBytes() const
{
    if (!applyOutputNorm_) {
        return 0U;
    }
    const uint64_t weightBytes =
        static_cast<uint64_t>(dSize_) * static_cast<uint64_t>(ge::GetSizeByDataType(outputNormWeightDtype_));
    return Ops::Base::CeilAlign(weightBytes, UB_ALIGN_BYTES);
}

uint64_t BlockAttnResUpdateFullDTiling::CalcUbBytes(uint32_t tileT) const
{
    const uint32_t statsTStride = Ops::Base::CeilAlign(tileT, FP32_ALIGN_ELEMENTS);
    const uint64_t queryBytes = static_cast<uint64_t>(tilingInfo_.dAlignFp32) * FP32_BYTES;
    const uint64_t normWeightBytes = GetOutputNormWeightUbBytes();
    const uint64_t partialBytes = static_cast<uint64_t>(tileT) * tilingInfo_.dAlignFp32 * FP32_BYTES;
    const uint64_t deltaHBytes = static_cast<uint64_t>(tileT) * tilingInfo_.dAlignBf16 * BF16_BYTES;
    const uint64_t numeratorBytes = partialBytes;
    const uint64_t statsBytes = static_cast<uint64_t>(STATS_PLANE_NUM) * statsTStride * FP32_BYTES;
    const uint64_t bufferBytes = partialBytes + deltaHBytes + numeratorBytes + statsBytes;
    // Only the final stats allocation needs padding; earlier masked tail reads stay within the following buffer.
    const uint64_t tailPaddingBytes = Ops::Base::CeilAlign(statsBytes, VECTOR_REG_BYTES) - statsBytes;
    return queryBytes + normWeightBytes + static_cast<uint64_t>(DOUBLE_BUFFER_NUM) * bufferBytes + tailPaddingBytes;
}

bool BlockAttnResUpdateFullDTiling::TryUbTiling(uint64_t maxTPerCore)
{
    // Keep at least two tiles on the busiest core so both ping-pong buffers participate when possible.
    const uint64_t maxTileT = (maxTPerCore + static_cast<uint64_t>(DOUBLE_BUFFER_NUM) - 1U) / DOUBLE_BUFFER_NUM;
    const uint64_t maxCandidateT = std::min<uint64_t>(maxTileT, MAX_DATA_COPY_BLOCK_COUNT);
    const uint64_t bytesPerT = static_cast<uint64_t>(tilingInfo_.dAlignFp32) * FP32_BYTES * FP32_MATRIX_NUM_PER_BUFFER +
                               static_cast<uint64_t>(tilingInfo_.dAlignBf16) * BF16_BYTES +
                               static_cast<uint64_t>(STATS_PLANE_NUM) * FP32_BYTES;
    const uint64_t queryBytes = static_cast<uint64_t>(tilingInfo_.dAlignFp32) * FP32_BYTES;
    const uint64_t normWeightBytes = GetOutputNormWeightUbBytes();
    const uint64_t invariantBytes = queryBytes + normWeightBytes;
    if (ubSize_ <= invariantBytes || bytesPerT == 0) {
        return false;
    }

    // Estimate an initial candidate from the per-row cost, then verify the exact aligned footprint while shrinking.
    for (uint64_t candidateT = std::min<uint64_t>(
             maxCandidateT, (ubSize_ - invariantBytes) / (static_cast<uint64_t>(DOUBLE_BUFFER_NUM) * bytesPerT));
         candidateT > 0; --candidateT) {
        const uint64_t ubBytes = CalcUbBytes(static_cast<uint32_t>(candidateT));
        if (ubBytes <= ubSize_) {
            tilingInfo_.tileT = static_cast<uint32_t>(candidateT);
            return true;
        }
    }
    return false;
}

uint64_t BlockAttnResUpdateFullDTiling::GetTilingKey() const
{
    uint32_t normWeightMode = BARU_NORM_WEIGHT_BF16;
    if (applyOutputNorm_) {
        normWeightMode = outputNormWeightDtype_ == ge::DT_BF16 ? BARU_NORM_WEIGHT_BF16 : BARU_NORM_WEIGHT_FP32;
    }
    // SINGLE_TILE omits ping-pong reuse events when every row assigned to a core fits in one tile.
    return GET_TPL_TILING_KEY(tilingInfo_.tPerCore <= tilingInfo_.tileT, applyOutputNorm_, normWeightMode);
}

void BlockAttnResUpdateFullDTiling::SetTilingData()
{
    tilingData_ = BlockAttnResUpdateTilingData{};
    tilingData_.dSize = dSize_;
    tilingData_.tPerCore = tilingInfo_.tPerCore;
    tilingData_.lastTPerCore = tilingInfo_.lastTPerCore;
    tilingData_.eps = eps_;
    tilingData_.invD = 1.0F / static_cast<float>(dSize_);
    tilingData_.outputNormEps = outputNormEps_;
    tilingData_.tileT = static_cast<uint16_t>(tilingInfo_.tileT);
    tilingData_.statsTStride = static_cast<uint16_t>(tilingInfo_.statsTStride);
    tilingData_.usedCoreNum = static_cast<uint16_t>(tilingInfo_.usedCoreNum);
}

ge::graphStatus BlockAttnResUpdateFullDTiling::PostTiling()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(OP_NAME, "Context is nullptr."), return ge::GRAPH_FAILED);
    auto* rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr, OP_LOGE(opName_, "Raw tiling data is nullptr."), return ge::GRAPH_FAILED);
    void* tilingDataBuffer = rawTilingData->GetData();
    OP_CHECK_IF(tilingDataBuffer == nullptr, OP_LOGE(opName_, "Raw tiling data buffer is nullptr."),
                return ge::GRAPH_FAILED);
    constexpr size_t tilingDataSize = sizeof(BlockAttnResUpdateTilingData);
    OP_CHECK_IF(rawTilingData->GetCapacity() < tilingDataSize,
                OP_LOGE(opName_, "Context tiling data capacity %zu is less than actual tiling data size %zu.",
                        rawTilingData->GetCapacity(), tilingDataSize),
                return ge::GRAPH_FAILED);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(opName_, "Workspace size buffer is nullptr."), return ge::GRAPH_FAILED);

    const errno_t ret = memcpy_s(tilingDataBuffer, rawTilingData->GetCapacity(), &tilingData_, tilingDataSize);
    OP_CHECK_IF(ret != EOK,
                OP_LOGE(opName_, "Failed to copy tiling data: bytes=%zu, capacity=%zu, ret=%d.", tilingDataSize,
                        rawTilingData->GetCapacity(), ret),
                return ge::GRAPH_FAILED);

    context_->SetBlockDim(tilingInfo_.usedCoreNum);
    rawTilingData->SetDataSize(tilingDataSize);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(BlockAttnResUpdate, BlockAttnResUpdateFullDTiling, supportedNpuArch,
                                   FULL_D_TILING_PRIORITY);

} // namespace BlockAttnResUpdate
} // namespace optiling
