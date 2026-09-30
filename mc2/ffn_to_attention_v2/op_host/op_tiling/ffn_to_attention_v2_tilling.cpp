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
 * \file ffn_to_attention_v2_tilling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <cstddef>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "../../op_kernel/ffn_to_attention_v2_tiling.h"
#include "../../../ffn_to_attention/op_kernel/ffn_to_attention_tiling.h"
#include "../../../ffn_to_attention/op_host/op_tiling/ffn_to_attention_tiling_base.h"
#include "../../op_kernel/ffn_to_attention_v2_tilling_key.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace MC2Tiling {
// Base tiling writes the V1 prefix; V2 overwrites its two trailing slots after
// the base call and appends window address-table fields. Full V1/V2 sizes need not match.
#define CHECK_FFN_TILING_PREFIX(field) \
    static_assert(offsetof(FFNToAttentionV2Info, field) == offsetof(FFNToAttentionInfo, field), \
                  "FFNToAttention V1/V2 shared field offset mismatch: " #field)
CHECK_FFN_TILING_PREFIX(H);
CHECK_FFN_TILING_PREFIX(A);
CHECK_FFN_TILING_PREFIX(microBatchNum);
CHECK_FFN_TILING_PREFIX(BS);
CHECK_FFN_TILING_PREFIX(expertNumPerToken);
CHECK_FFN_TILING_PREFIX(HS);
CHECK_FFN_TILING_PREFIX(aivNum);
CHECK_FFN_TILING_PREFIX(worldSize);
CHECK_FFN_TILING_PREFIX(isInputRankTable);
CHECK_FFN_TILING_PREFIX(windowType);
#undef CHECK_FFN_TILING_PREFIX
static_assert(offsetof(FFNToAttentionV2Info, maxTokenNum) == offsetof(FFNToAttentionInfo, totalUbSize),
              "FFNToAttention base UB slot offset mismatch");
static_assert(offsetof(FFNToAttentionV2Info, cclBufferSize) == offsetof(FFNToAttentionInfo, totalWinSize),
              "FFNToAttention base window slot offset mismatch");
static_assert(offsetof(FFNToAttentionV2Info, addressTableWinOffset) >= sizeof(FFNToAttentionInfo),
              "FFNToAttention V2 extension overlaps base tiling");
static_assert(offsetof(FFNToAttentionV2TilingData, mc2InitTiling) ==
                      offsetof(FFNToAttentionTilingData, mc2InitTiling) &&
                  offsetof(FFNToAttentionV2TilingData, mc2CcTiling1) ==
                      offsetof(FFNToAttentionTilingData, mc2CcTiling1) &&
                  offsetof(FFNToAttentionV2TilingData, ffnToAttentionV2Info) ==
                      offsetof(FFNToAttentionTilingData, ffnToAttentionInfo),
              "FFNToAttention V1/V2 base tiling offsets mismatch");
static_assert(sizeof(FFNToAttentionV2TilingData) >= sizeof(FFNToAttentionTilingData),
              "FFNToAttention V2 tiling cannot contain base tiling");

constexpr size_t URMA_FLAG_SLOT_SIZE = 32U;
constexpr size_t ADDRESS_ENTRY_SIZE = 2U * sizeof(uint64_t);
constexpr size_t ADDRESS_TABLE_ALIGN = 512U;
constexpr uint32_t BATCH_MODE_SCHEDULE = 1U;
// Extra workspace headroom required by the URMA communication runtime.
constexpr size_t RESERVED_WORKSPACE_SIZE = 1024 * 1024 * 64LL;
// Mirrors of the kernel UB buffer constants in op_kernel/ffn_to_attention_urma.h; keep both sides in sync.
constexpr uint64_t TOKEN_DTYPE_BYTES = 2U;
constexpr uint64_t URMA_BUFFER_NUM = 2U;
constexpr uint64_t URMA_UB_ALIGN = 32U;
constexpr uint64_t URMA_HCOMM_INIT_SIZE = 512U;
constexpr uint64_t URMA_BATCH_BUFFER_SIZE = 256U * 64U;
constexpr uint64_t METADATA_CHUNK_BUFFER_SIZE = 256U * 4U * sizeof(int32_t);
constexpr uint64_t ADDRESS_TABLE_BUFFER_SIZE = 40U * 1024U;
constexpr uint64_t RANK_INFO_SEGMENT_NUM = 3U;

ge::graphStatus FFNToAttentionV2TilingFunc(gert::TilingContext* context)
{
    FFNToAttentionTilingConfig config;
    config.contextIndex = 0U;
    config.xIndex = 1U;
    config.sessionIdsIndex = 2U;
    config.microBatchIdsIndex = 3U;
    config.tokenIdsIndex = 4U;
    config.expertOffsetsIndex = 5U;
    config.actualTokenNumIndex = 6U;
    config.attnRankTableIndex = 7U;
    config.attrGroupIndex = 0U;
    config.attrWorldSizeIndex = 1U;
    config.attrTokenInfoTableShapeIndex = 2U;
    config.attrTokenDataShapeIndex = 3U;
    config.attrCclBufferSizeIndex = 4U;
    config.isMc2Context = true;
    config.allowMultiMicroBatch = true;

    auto ret = FFNToAttentionTilingFuncBase(context, config);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    OP_TILING_CHECK(context->SetScheduleMode(BATCH_MODE_SCHEDULE) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "failed to enable batch schedule mode"), return ge::GRAPH_FAILED);

    // Payloads are read directly from x. Flag sources and address tables occupy the local window tail.
    FFNToAttentionV2TilingData* tilingData = context->GetTilingData<FFNToAttentionV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tilingData->ffnToAttentionV2Info.H > tilingData->ffnToAttentionV2Info.HS,
                    OP_LOGE(context->GetNodeName(), "token_data_shape HS must be greater than or equal to x H"),
                    return ge::GRAPH_FAILED);
    const gert::StorageShape* xShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "xShape"),
                    return ge::GRAPH_FAILED);
    const int64_t maxTokenNumDim = xShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(maxTokenNumDim < 0 || maxTokenNumDim > std::numeric_limits<int32_t>::max(),
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "x",
                                              (std::string("dim0=") + std::to_string(maxTokenNumDim)).c_str(),
                                              "a non-negative int32 token count"),
                    return ge::GRAPH_FAILED);
    tilingData->ffnToAttentionV2Info.maxTokenNum = static_cast<uint64_t>(maxTokenNumDim);

    size_t* workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "workSpaces"),
                    return ge::GRAPH_FAILED);
    const size_t aivNum = static_cast<size_t>(tilingData->ffnToAttentionV2Info.aivNum);
    const size_t maxTokenNum = static_cast<size_t>(tilingData->ffnToAttentionV2Info.maxTokenNum);
    const size_t worldSize = static_cast<size_t>(tilingData->ffnToAttentionV2Info.worldSize);
    OP_TILING_CHECK(aivNum == 0U,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "aivNum", std::to_string(aivNum).c_str(),
                                              "greater than zero"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(worldSize == 0U,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "worldSize", std::to_string(worldSize).c_str(),
                                              "greater than zero"),
                    return ge::GRAPH_FAILED);
    const size_t flagWindowSize = aivNum * URMA_FLAG_SLOT_SIZE;
    const auto& info = tilingData->ffnToAttentionV2Info;
    // Mirror the explicit InitBuffer allocations per AIV. The HCOMM buffers are allocated on
    // every AIV for the multi-channel relay, so the sum is each AIV's UB footprint.
    const uint64_t alignedPayloadBytes =
        (static_cast<uint64_t>(info.H) * TOKEN_DTYPE_BYTES + URMA_UB_ALIGN - 1U) / URMA_UB_ALIGN * URMA_UB_ALIGN;
    const uint64_t ubCountRowBytes =
        (static_cast<uint64_t>(worldSize) * sizeof(int32_t) + URMA_UB_ALIGN - 1U) / URMA_UB_ALIGN * URMA_UB_ALIGN;
    const uint64_t rankTableUbBytes =
        info.isInputRankTable ?
            (static_cast<uint64_t>(info.A) * sizeof(int32_t) + URMA_UB_ALIGN - 1U) / URMA_UB_ALIGN * URMA_UB_ALIGN :
            0U;
    const uint64_t requiredUbBytes =
        URMA_BUFFER_NUM * alignedPayloadBytes + URMA_UB_ALIGN + METADATA_CHUNK_BUFFER_SIZE + ADDRESS_TABLE_BUFFER_SIZE +
        RANK_INFO_SEGMENT_NUM * ubCountRowBytes + URMA_HCOMM_INIT_SIZE + URMA_BATCH_BUFFER_SIZE + rankTableUbBytes;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t availableUbBytes = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbBytes);
    OP_TILING_CHECK(requiredUbBytes > availableUbBytes,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "requiredUbBytes",
                        (std::to_string(requiredUbBytes) + " > " + std::to_string(availableUbBytes)).c_str(),
                        "URMA explicit buffers must fit in available UB"),
                    return ge::GRAPH_FAILED);
    // Shape limits validated by the base tiling bound these products (MB in [1, uint32 max], BS<=512, K<=256,
    // HS<=8320).
    const uint64_t tokenSlotNum = static_cast<uint64_t>(info.microBatchNum) * info.BS * info.expertNumPerToken;
    const uint64_t rankTableStride =
        (static_cast<uint64_t>(maxTokenNum) * ADDRESS_ENTRY_SIZE + ADDRESS_TABLE_ALIGN - 1U) / ADDRESS_TABLE_ALIGN *
        ADDRESS_TABLE_ALIGN;
    const uint64_t countRowStride = (static_cast<uint64_t>(worldSize) * sizeof(int32_t) + ADDRESS_TABLE_ALIGN - 1U) /
                                    ADDRESS_TABLE_ALIGN * ADDRESS_TABLE_ALIGN;
    const uint64_t tableBytes = rankTableStride * worldSize + (static_cast<uint64_t>(aivNum) + 1U) * countRowStride;

    const uint64_t infoTableLastDimNum = 2U + static_cast<uint64_t>(info.BS) * info.expertNumPerToken;
    const auto recvRegionBytes = [&](uint64_t workerNum) {
        const uint64_t recvInfoRegionSize =
            (workerNum * info.microBatchNum * infoTableLastDimNum * sizeof(int32_t) + ADDRESS_TABLE_ALIGN - 1U) /
            ADDRESS_TABLE_ALIGN * ADDRESS_TABLE_ALIGN;
        const uint64_t recvDataRegionSize =
            (workerNum * tokenSlotNum * info.HS * TOKEN_DTYPE_BYTES + ADDRESS_TABLE_ALIGN - 1U) / ADDRESS_TABLE_ALIGN *
            ADDRESS_TABLE_ALIGN;
        return recvInfoRegionSize + recvDataRegionSize;
    };
    const auto requiredWindowSizeOf = [&](uint64_t workerNum) {
        const uint64_t tableOffset = (recvRegionBytes(workerNum) + flagWindowSize + ADDRESS_TABLE_ALIGN - 1U) /
                                     ADDRESS_TABLE_ALIGN * ADDRESS_TABLE_ALIGN;
        return tableOffset + tableBytes;
    };
    // Bound the N-scaled products (the base tiling's single-session bound does not include the worker factor).
    const uint64_t windowSizeSafeMax = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 4U;
    const uint64_t maxWorkerNum = worldSize - 1U; // worldSize >= 2 is validated by the base tiling
    OP_TILING_CHECK(
        maxWorkerNum > windowSizeSafeMax / tokenSlotNum ||
            static_cast<uint64_t>(info.HS) * TOKEN_DTYPE_BYTES > windowSizeSafeMax / tokenSlotNum / maxWorkerNum,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "world_size", std::to_string(worldSize).c_str(),
            (std::string("N-scaled token data region overflows the window size range, tokenSlotNum=") +
             std::to_string(tokenSlotNum) + ", HS=" + std::to_string(info.HS))
                .c_str()),
        return ge::GRAPH_FAILED);

    const auto cclBufferSize = context->GetAttrs()->GetAttrPointer<int64_t>(config.attrCclBufferSizeIndex);
    OP_TILING_CHECK(cclBufferSize == nullptr || *cclBufferSize <= 0,
                    OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "ccl_buffer_size"), return ge::GRAPH_FAILED);
    uint64_t attentionWorkerNum = 0U;
    if (info.isInputRankTable) {
        attentionWorkerNum = static_cast<uint64_t>(info.A);
        OP_TILING_CHECK(attentionWorkerNum == 0U || attentionWorkerNum >= worldSize,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context->GetNodeName(), "attn_rank_table", std::to_string(info.A).c_str(),
                            (std::string("dim0 in [1, worldSize=") + std::to_string(worldSize) +
                             ") matching attention_to_ffn_v2's attentionWorkerNum")
                                .c_str()),
                        return ge::GRAPH_FAILED);
    } else {
        for (uint64_t workerNum = 1U; workerNum <= maxWorkerNum; ++workerNum) {
            if (requiredWindowSizeOf(workerNum) > static_cast<uint64_t>(*cclBufferSize)) {
                break;
            }
            attentionWorkerNum = workerNum;
        }
    }
    OP_TILING_CHECK(attentionWorkerNum == 0U,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "ccl_buffer_size", std::to_string(*cclBufferSize).c_str(),
                        (std::string("should be >= ") + std::to_string(requiredWindowSizeOf(1U)) +
                         " bytes (one attention worker session + flag sources and address tables); use "
                         "get_ffn_to_attention_ccl_buffer_size to compute it")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    // The session count only drives the layout below; the kernel derives its flag-source base from
    // addressTableWinOffset, so no extra tiling field is needed.
    const uint64_t flagWindowOffset = recvRegionBytes(attentionWorkerNum);
    const uint64_t tableOffset =
        (flagWindowOffset + flagWindowSize + ADDRESS_TABLE_ALIGN - 1U) / ADDRESS_TABLE_ALIGN * ADDRESS_TABLE_ALIGN;
    const uint64_t requiredWindowSize = tableOffset + tableBytes;
    // Rank-table mode enforces the exact session count here; in default mode the search above already
    // guarantees requiredWindowSize <= ccl_buffer_size.
    OP_TILING_CHECK(static_cast<uint64_t>(*cclBufferSize) < requiredWindowSize,
                    OP_LOGE(context->GetNodeName(),
                            "ccl_buffer_size must include the attention_to_ffn receiving area, flag sources and "
                            "address tables (%llu bytes)",
                            static_cast<unsigned long long>(requiredWindowSize)),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(workSpaces[0] > std::numeric_limits<size_t>::max() - RESERVED_WORKSPACE_SIZE,
                    OP_LOGE(context->GetNodeName(), "workspace size overflow"), return ge::GRAPH_FAILED);
    workSpaces[0] += RESERVED_WORKSPACE_SIZE;
    tilingData->ffnToAttentionV2Info.cclBufferSize = static_cast<uint64_t>(*cclBufferSize);
    tilingData->ffnToAttentionV2Info.addressTableWinOffset = tableOffset;
    tilingData->ffnToAttentionV2Info.addressTableBytes = tableBytes;

    // FFNToAttentionV2 on A5 uses the URMA communication implementation.
    bool rankTableMode = tilingData->ffnToAttentionV2Info.isInputRankTable;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(rankTableMode, TILINGKEY_TPL_A5);
    context->SetTilingKey(tilingKey);
    OP_LOGD(context->GetNodeName(), "FFNToAttentionV2 cur case tilingKey is %lu", tilingKey);

    return ge::GRAPH_SUCCESS;
}

struct FFNToAttentionV2CompileInfo {};
ge::graphStatus TilingParseForFFNToAttentionV2(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FFNToAttentionV2)
    .Tiling(FFNToAttentionV2TilingFunc)
    .TilingParse<FFNToAttentionV2CompileInfo>(TilingParseForFFNToAttentionV2);
} // namespace MC2Tiling
