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
 * \file attention_to_ffn_v2_tiling.cpp
 * \brief
 */

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "attention_to_ffn_v2_tiling.h"
#include "../../op_kernel/attention_to_ffn_v2_tiling.h"
#include "../../op_kernel/attention_to_ffn_v2_tiling_key.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace MC2Tiling {
namespace {
constexpr uint32_t ATTR_GROUP_INDEX = 0U;
constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 1U;
constexpr uint32_t ATTR_FFN_TOKEN_INFO_SHAPE_INDEX = 2U;
constexpr uint32_t ATTR_FFN_TOKEN_DATA_SHAPE_INDEX = 3U;
constexpr uint32_t ATTR_ATTN_TOKEN_INFO_SHAPE_INDEX = 4U;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 5U;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 6U;
constexpr uint32_t ATTR_SYNC_FLAG_INDEX = 7U;
constexpr uint32_t ATTR_FFN_START_RANK_ID_INDEX = 8U;
constexpr uint32_t ATTR_CCL_BUFFER_SIZE_INDEX = 9U;

constexpr size_t URMA_FLAG_SLOT_SIZE = 32U;
constexpr size_t URMA_WIN_REGION_ALIGN = 512U;
constexpr size_t UB_ALIGN_SIZE = 32U;
constexpr size_t SCALE_PARAM_PAD_SIZE = 128U;
constexpr size_t MX_PARAM_PAD_SIZE = 256U;
constexpr size_t WORKSPACE_ELEMENT_OFFSET = 128U;
constexpr uint32_t BATCH_MODE_SCHEDULE = 1U;
constexpr size_t TOKEN_BYTES = 2U; // x dtype is FP16/BF16 (validated by the base tiling)
// Mirrors of the kernel UB buffer constants in op_kernel/attention_to_ffn_urma.h; keep both sides in sync.
constexpr size_t URMA_BUFFER_NUM = 2U;                 // kernel: BUFFER_NUM
constexpr size_t URMA_HCOMM_INIT_SIZE = 512U;          // kernel: URMA_HCOMM_INIT_SIZE
constexpr size_t URMA_BATCH_BUFFER_SIZE = 16U * 1024U; // kernel: ATTN_FFN_HCOMM_BATCH_UB_BYTES
constexpr size_t BUCKET_BUFFER_SIZE = 32U * 1024U;     // kernel: BUCKET_BUFFER_SIZE
constexpr size_t RANK_INFO_SEGMENT_NUM = 4U;           // kernel: RANK_INFO_SEGMENT_NUM
constexpr size_t MX_PERGROUP_BLOCK_SIZE = 128U;        // kernel: PERGROUP_BLOCK_SIZE

inline size_t CeilAlignSize(size_t val, size_t align)
{
    return (val + align - 1U) / align * align;
}

AttentionToFFNTilingConfig MakeTilingConfig(bool allowMxQuantMode)
{
    AttentionToFFNTilingConfig config;
    config.contextIndex = 0U;
    config.xIndex = 1U;
    config.sessionIdIndex = 2U;
    config.microBatchIdIndex = 3U;
    config.layerIdIndex = 4U;
    config.expertIdsIndex = 5U;
    config.expertRankTableIndex = 6U;
    config.scalesIndex = 7U;
    config.activeMaskIndex = 8U;
    config.attrGroupIndex = ATTR_GROUP_INDEX;
    config.attrWorldSizeIndex = ATTR_WORLD_SIZE_INDEX;
    config.attrFfnTokenInfoTableShapeIndex = ATTR_FFN_TOKEN_INFO_SHAPE_INDEX;
    config.attrFfnTokenDataShapeIndex = ATTR_FFN_TOKEN_DATA_SHAPE_INDEX;
    config.attrAttnTokenInfoTableShapeIndex = ATTR_ATTN_TOKEN_INFO_SHAPE_INDEX;
    config.attrMoeExpertNumIndex = ATTR_MOE_EXPERT_NUM_INDEX;
    config.attrQuantModeIndex = ATTR_QUANT_MODE_INDEX;
    config.attrSyncFlagIndex = ATTR_SYNC_FLAG_INDEX;
    config.attrFfnStartRankIdIndex = ATTR_FFN_START_RANK_ID_INDEX;
    config.attrCclBufferSizeIndex = ATTR_CCL_BUFFER_SIZE_INDEX;
    config.isMc2Context = true;
    config.allowMxQuantMode = allowMxQuantMode;
    config.allowMultiLayer = true;
    config.maxWorldSize = 1024;
    return config;
}

// User-facing quant_mode values that select the MX/MX_CLIP algorithm.
constexpr uint32_t USER_QUANT_MODE_MX_E5M2 = 3U;
constexpr uint32_t USER_QUANT_MODE_MX_E4M3 = 4U;
constexpr uint32_t USER_QUANT_MODE_MX_E2M1 = 5U;
constexpr uint32_t USER_QUANT_MODE_MX_CLIP_E5M2 = 6U;
constexpr uint32_t USER_QUANT_MODE_MX_CLIP_E4M3 = 7U;

bool IsMxQuantMode(uint32_t quantMode)
{
    return quantMode == USER_QUANT_MODE_MX_E5M2 || quantMode == USER_QUANT_MODE_MX_E4M3 ||
           quantMode == USER_QUANT_MODE_MX_E2M1 || quantMode == USER_QUANT_MODE_MX_CLIP_E5M2 ||
           quantMode == USER_QUANT_MODE_MX_CLIP_E4M3;
}

ge::graphStatus CheckQuantMode(const char* nodeName, uint32_t quantMode, bool isScales)
{
    static const std::set<uint32_t> validModes = {ATTN_FFN_TILINGKEY_NO_QUANT, ATTN_FFN_TILINGKEY_PERTOKEN_INT8,
                                                  USER_QUANT_MODE_MX_E5M2,     USER_QUANT_MODE_MX_E4M3,
                                                  USER_QUANT_MODE_MX_E2M1,     USER_QUANT_MODE_MX_CLIP_E5M2,
                                                  USER_QUANT_MODE_MX_CLIP_E4M3};
    OP_TILING_CHECK(
        validModes.find(quantMode) == validModes.end(),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "quant_mode", std::to_string(quantMode).c_str(), "0, 2, 3, 4, 5, 6 or 7"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(IsMxQuantMode(quantMode) && isScales,
                    OP_LOGE(nodeName, "scales must be absent in MX/MX_CLIP modes"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Map user-facing quant_mode to internal (tilingKeyQuantMode, outDtype) pair.
void ResolveTilingKeyFields(uint32_t quantMode, uint32_t& tilingKeyQuantMode, uint32_t& outDtype)
{
    switch (quantMode) {
        case USER_QUANT_MODE_MX_E5M2:
            tilingKeyQuantMode = ATTN_FFN_TILINGKEY_MX;
            outDtype = ATTN_FFN_TILINGKEY_OUT_E5M2;
            break;
        case USER_QUANT_MODE_MX_E4M3:
            tilingKeyQuantMode = ATTN_FFN_TILINGKEY_MX;
            outDtype = ATTN_FFN_TILINGKEY_OUT_E4M3;
            break;
        case USER_QUANT_MODE_MX_E2M1:
            tilingKeyQuantMode = ATTN_FFN_TILINGKEY_MX;
            outDtype = ATTN_FFN_TILINGKEY_OUT_E2M1;
            break;
        case USER_QUANT_MODE_MX_CLIP_E5M2:
            tilingKeyQuantMode = ATTN_FFN_TILINGKEY_MX_CLIP;
            outDtype = ATTN_FFN_TILINGKEY_OUT_E5M2;
            break;
        case USER_QUANT_MODE_MX_CLIP_E4M3:
            tilingKeyQuantMode = ATTN_FFN_TILINGKEY_MX_CLIP;
            outDtype = ATTN_FFN_TILINGKEY_OUT_E4M3;
            break;
        default:
            tilingKeyQuantMode = quantMode;
            outDtype = ATTN_FFN_TILINGKEY_OUT_INT8;
            break;
    }
}

ge::graphStatus SetV2TilingKey(gert::TilingContext* context, uint32_t quantMode, uint32_t outDtype, bool isScales,
                               bool isSync, bool isActiveMask)
{
    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(quantMode, outDtype, isScales, isSync, isActiveMask, TILINGKEY_TPL_A5);
    context->SetTilingKey(tilingKey);
    OP_LOGD(context->GetNodeName(), "AttentionToFfnV2 cur case tilingKey is %lu", tilingKey);
    return ge::GRAPH_SUCCESS;
}

// Per-token bytes written into a win slot (data + quant scale pad); must match the kernel's hCommuSize_.
static size_t GetUrmaCommuBytes(uint32_t tilingKeyQuantMode, uint32_t outDtype, size_t axisH)
{
    if (tilingKeyQuantMode == ATTN_FFN_TILINGKEY_NO_QUANT) {
        return axisH * sizeof(uint16_t);
    }
    if (tilingKeyQuantMode == ATTN_FFN_TILINGKEY_PERTOKEN_INT8) {
        return axisH * sizeof(int8_t) + SCALE_PARAM_PAD_SIZE;
    }
    const size_t outDataBytes = (outDtype == ATTN_FFN_TILINGKEY_OUT_E2M1) ? (axisH + 1U) / 2U : axisH * sizeof(int8_t);
    const size_t hOutSizeAlign = (outDataBytes + MX_PARAM_PAD_SIZE - 1U) / MX_PARAM_PAD_SIZE * MX_PARAM_PAD_SIZE;
    const size_t mxScaleNum = ((axisH + 31U) / 32U + 1U) / 2U * 2U; // E8M0 scales, 1 byte each
    return hOutSizeAlign + mxScaleNum;
}

// Token data is staged in the rank's own window slot (capacity HS * elemSize), so both the staging
// write and the destination WQE require HS * elemSize >= commuBytes.
ge::graphStatus CheckUrmaWinSlotCapacity(gert::TilingContext* context, const AttentionToFfnV2TilingData* tilingData,
                                         uint32_t tilingKeyQuantMode, uint32_t outDtype)
{
    const size_t axisH = static_cast<size_t>(tilingData->attentionToFfnV2Info.H);
    const size_t axisHS = static_cast<size_t>(tilingData->attentionToFfnV2Info.HS);
    const size_t elemSize = (tilingKeyQuantMode == ATTN_FFN_TILINGKEY_NO_QUANT) ? sizeof(uint16_t) : sizeof(int8_t);
    const size_t commuBytes = GetUrmaCommuBytes(tilingKeyQuantMode, outDtype, axisH);
    OP_TILING_CHECK(axisHS * elemSize < commuBytes,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "ffn_token_data_shape[4]", std::to_string(axisHS).c_str(),
                        (std::string("HS * elemSize (") + std::to_string(axisHS * elemSize) +
                         " bytes) must be >= per-token payload " + std::to_string(commuBytes) +
                         " bytes (data + quant scale pad), otherwise the win slot overflows")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// All URMA staging sources live in the rank's own window (persistent across op calls): token data
// is staged in a compact contiguous area, physically separated from the session-slot receiving
// area to avoid HBM bank conflicts. Token/layer flags use a dedicated area after the staging
// region. Nothing is staged in the per-call workspace because fire-and-forget WQEs may outlive the
// op. Only layout validations remain here.
ge::graphStatus CheckUrmaWinLayout(gert::TilingContext* context, AttentionToFfnV2TilingData* tilingData,
                                   uint32_t tilingKeyQuantMode, uint32_t outDtype)
{
    OP_TILING_CHECK(
        tilingData->attentionToFfnV2Info.aivNum == 0U,
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "aivNum",
                                  std::to_string(tilingData->attentionToFfnV2Info.aivNum).c_str(), "greater than zero"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckUrmaWinSlotCapacity(context, tilingData, tilingKeyQuantMode, outDtype) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "URMA win slot capacity check failed"), return ge::GRAPH_FAILED);

    const auto& info = tilingData->attentionToFfnV2Info;
    const size_t expertNumPerToken =
        static_cast<size_t>(info.K) + (static_cast<size_t>(info.expertNum) - info.moeExpertNum);
    const size_t elemSize = (tilingKeyQuantMode == ATTN_FFN_TILINGKEY_NO_QUANT) ? sizeof(uint16_t) : sizeof(int8_t);
    const size_t infoRegionSize = CeilAlignSize(
        static_cast<size_t>(info.attentionWorkerNum) * info.microBatchNum * info.infoTableLastDimNum * sizeof(int32_t),
        URMA_WIN_REGION_ALIGN);
    const size_t dataRegionSize = CeilAlignSize(static_cast<size_t>(info.attentionWorkerNum) * info.microBatchNum *
                                                    info.BS * expertNumPerToken * info.HS * elemSize,
                                                URMA_WIN_REGION_ALIGN);

    const size_t f2aReturnDataBytes =
        static_cast<size_t>(info.microBatchNum) * info.BS * expertNumPerToken * info.HS * sizeof(uint16_t);
    OP_TILING_CHECK(
        dataRegionSize < f2aReturnDataBytes,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "ffn_token_data_shape", std::to_string(info.attentionWorkerNum).c_str(),
            (std::string("attentionWorkerNum=") + std::to_string(info.attentionWorkerNum) +
             " with elemSize=" + std::to_string(elemSize) + " leaves a data region of " +
             std::to_string(dataRegionSize) + " bytes that cannot host the ffn_to_attention_v2 return data of " +
             std::to_string(f2aReturnDataBytes) + " bytes")
                .c_str()),
        return ge::GRAPH_FAILED);
    const size_t commuBytes = GetUrmaCommuBytes(tilingKeyQuantMode, outDtype, static_cast<size_t>(info.H));
    const size_t stagingStride = CeilAlignSize(commuBytes, UB_ALIGN_SIZE);
    const size_t maxTotalSendNum = static_cast<size_t>(info.X) * info.BS * expertNumPerToken;
    const size_t stagingRegionSize = CeilAlignSize(maxTotalSendNum * stagingStride, URMA_WIN_REGION_ALIGN);
    const size_t flagRegionSize =
        (static_cast<size_t>(info.X) * info.BS * expertNumPerToken + 1U) * URMA_FLAG_SLOT_SIZE;
    const size_t requiredWinBytes = infoRegionSize + dataRegionSize + stagingRegionSize + flagRegionSize;

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "ccl_buffer_size"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<uint64_t>(*cclBufferSizePtr) < static_cast<uint64_t>(requiredWinBytes),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "ccl_buffer_size", std::to_string(*cclBufferSizePtr).c_str(),
                        (std::string("should >= ") + std::to_string(requiredWinBytes) +
                         " bytes (token info + token data + compact staging + URMA flag staging area); use "
                         "get_buffer_for_attention_to_ffn to compute it")
                            .c_str()),
                    return ge::GRAPH_FAILED);

    tilingData->attentionToFfnV2Info.urmaWorkspaceOffset = 0U;
    return ge::GRAPH_SUCCESS;
}

// Mirror the per-AIV explicit InitBuffer footprint of the URMA kernel (op_kernel/attention_to_ffn_urma.h):
// Init() + QuantInit() + HcommInit() + the isSync syncStatusWorkspaceBuf staged at runtime in SetFlagToFFN().
// TPipe keeps every explicit buffer alive for the whole kernel lifetime, so the per-AIV UB footprint is the
// sum of all allocations below. Buffer constants mirror the kernel side (see the mirrors block above);
// keep both sides in sync when the kernel allocations change.
ge::graphStatus CheckUrmaUbCapacity(gert::TilingContext* context, const AttentionToFfnV2TilingData* tilingData,
                                    uint32_t tilingKeyQuantMode, uint32_t outDtype)
{
    const auto& info = tilingData->attentionToFfnV2Info;
    const size_t aivNum = static_cast<size_t>(info.aivNum);
    OP_TILING_CHECK(aivNum == 0U,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "aivNum", std::to_string(aivNum).c_str(),
                                              "greater than zero"),
                    return ge::GRAPH_FAILED);

    const size_t ubAlign = UB_ALIGN_SIZE;
    const size_t axisH = static_cast<size_t>(info.H);
    const size_t expertNumPerToken =
        static_cast<size_t>(info.K) + (static_cast<size_t>(info.expertNum) - info.moeExpertNum);
    const size_t maxTotalSendNum = static_cast<size_t>(info.X) * info.BS * expertNumPerToken;
    const size_t maxTokensPerAiv = (maxTotalSendNum + aivNum - 1U) / aivNum;

    // Init(): expertIds + status + per-AIV flag staging + rank scratch + bucket + rankInfo.
    const size_t expertIdsBytes = CeilAlignSize(maxTotalSendNum * sizeof(int32_t), ubAlign);
    const size_t statusBytes = ubAlign;
    const size_t flagValueBytes = CeilAlignSize(maxTokensPerAiv * URMA_FLAG_SLOT_SIZE, ubAlign);
    const size_t flagOffsetBytes = CeilAlignSize(maxTokensPerAiv * sizeof(uint32_t), ubAlign);
    const size_t tokenRankStashBytes = CeilAlignSize(maxTokensPerAiv * sizeof(int32_t), ubAlign);
    const size_t rankCountRowBytes = CeilAlignSize(static_cast<size_t>(info.worldSize) * sizeof(int32_t), ubAlign);
    const size_t rankInfoBytes = RANK_INFO_SEGMENT_NUM * rankCountRowBytes;

    // QuantInit()/Init() queues; castTempBuf_/sumOutBuf_ alias the fp32 buffers in quant mode.
    size_t queueBytes = 0U;
    if (tilingKeyQuantMode == ATTN_FFN_TILINGKEY_NO_QUANT) {
        queueBytes = URMA_BUFFER_NUM * CeilAlignSize(axisH * TOKEN_BYTES, ubAlign);
    } else {
        size_t xInBytes = 0U;
        if (tilingKeyQuantMode == ATTN_FFN_TILINGKEY_PERTOKEN_INT8) {
            xInBytes = CeilAlignSize(axisH * TOKEN_BYTES, ubAlign);
        } else {
            // MX/MX_CLIP stage the input per 128-element groups; the group count is a ceiling
            // division (kernel: Ceil(axisH, PERGROUP_BLOCK_SIZE)), NOT an align-up of axisH.
            xInBytes = CeilAlignSize(
                (axisH + MX_PERGROUP_BLOCK_SIZE - 1U) / MX_PERGROUP_BLOCK_SIZE * MX_PERGROUP_BLOCK_SIZE * TOKEN_BYTES,
                ubAlign);
        }
        const size_t xOutBytes = CeilAlignSize(GetUrmaCommuBytes(tilingKeyQuantMode, outDtype, axisH), ubAlign);
        queueBytes = URMA_BUFFER_NUM * (xInBytes + xOutBytes) +
                     2U * CeilAlignSize(axisH * sizeof(float), ubAlign); // receiveDataCastFloat + smoothScales
    }

    // isSync: ffnStatusBuf_ on every AIV plus the SetFlagToFFN syncStatusWorkspaceBuf (aivNum rows).
    size_t syncBytes = 0U;
    if (info.syncFlag == 1U) {
        const size_t ffnNum = static_cast<size_t>(info.worldSize) - info.attentionWorkerNum;
        const size_t ffnNumAlign = CeilAlignSize(ffnNum * sizeof(int32_t), ubAlign);
        syncBytes = (aivNum + 1U) * ffnNumAlign;
    }

    // isActiveMask: activeMaskBuf_ (BS bools); castTempBuf_/sumOutBuf_ only exist in non-quant mode.
    size_t activeMaskBytes = 0U;
    if (info.isActiveMask) {
        activeMaskBytes = CeilAlignSize(static_cast<size_t>(info.BS), ubAlign);
        if (tilingKeyQuantMode == ATTN_FFN_TILINGKEY_NO_QUANT) {
            activeMaskBytes += 2U * CeilAlignSize(static_cast<size_t>(info.BS) * sizeof(uint16_t), ubAlign);
        }
    }

    const uint64_t requiredUbBytes = static_cast<uint64_t>(
        expertIdsBytes + statusBytes + flagValueBytes + flagOffsetBytes + tokenRankStashBytes + BUCKET_BUFFER_SIZE +
        rankInfoBytes + queueBytes + syncBytes + activeMaskBytes + URMA_HCOMM_INIT_SIZE + URMA_BATCH_BUFFER_SIZE);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t availableUbBytes = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbBytes);
    OP_TILING_CHECK(requiredUbBytes > availableUbBytes,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "requiredUbBytes",
                        (std::to_string(requiredUbBytes) + " > " + std::to_string(availableUbBytes)).c_str(),
                        "URMA explicit buffers must fit in available UB"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Extend workspace for the relay rank tables. Phase 1 (all AIVs) counts its remote records per
// rank and publishes one count row per AIV; the scatter pass buckets token offsets into compact
// per-rank tables; the relay reads only the served ranks' tables, replacing the previous
// full-list rescan per (dstRank, channel) slot.
static void ExtendWorkspaceForRankTables(gert::TilingContext* context, AttentionToFfnV2TilingData* tilingData)
{
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    if (workSpaces == nullptr) {
        return;
    }
    const auto& info = tilingData->attentionToFfnV2Info;
    const size_t expertNumPerToken =
        static_cast<size_t>(info.K) + (static_cast<size_t>(info.expertNum) - info.moeExpertNum);
    const size_t maxTotalSendNum = static_cast<size_t>(info.X) * info.BS * expertNumPerToken;
    const size_t ffnNum = static_cast<size_t>(info.worldSize) - info.attentionWorkerNum;
    const size_t ffnNumAlignSize = CeilAlignSize(ffnNum * sizeof(int32_t), WORKSPACE_ELEMENT_OFFSET);
    const size_t syncStatusRegionBytes = ffnNumAlignSize * info.aivNum;
    const size_t regionsOffset = CeilAlignSize(syncStatusRegionBytes, WORKSPACE_ELEMENT_OFFSET);
    const size_t rankCountRowBytes =
        CeilAlignSize(static_cast<size_t>(info.worldSize) * sizeof(int32_t), UB_ALIGN_SIZE);
    const size_t countRowStride = CeilAlignSize(rankCountRowBytes, WORKSPACE_ELEMENT_OFFSET);
    const size_t countMatrixBytes =
        CeilAlignSize(static_cast<size_t>(info.aivNum) * countRowStride, WORKSPACE_ELEMENT_OFFSET);
    const size_t bucketTableBytes = CeilAlignSize(maxTotalSendNum * sizeof(int32_t), WORKSPACE_ELEMENT_OFFSET);
    workSpaces[0] += regionsOffset + countMatrixBytes + bucketTableBytes;
}

ge::graphStatus TilingNewQuantMode(gert::TilingContext* context, uint32_t quantMode)
{
    const char* nodeName = context->GetNodeName();

    AttentionToFFNTilingConfig config = MakeTilingConfig(true);

    auto ret = AttentionToFFNTilingFuncBase(context, config);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    AttentionToFfnV2TilingData* tilingData = context->GetTilingData<AttentionToFfnV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);
    auto& info = tilingData->attentionToFfnV2Info;
    info.windowType = 0U;

    auto attrs = context->GetAttrs();
    auto ffnDataShapeAttr = attrs->GetListInt(ATTR_FFN_TOKEN_DATA_SHAPE_INDEX);
    auto ffnInfoShapeAttr = attrs->GetListInt(ATTR_FFN_TOKEN_INFO_SHAPE_INDEX);
    const int64_t* ffnDataShape = ffnDataShapeAttr->GetData();
    const int64_t* ffnInfoShape = ffnInfoShapeAttr->GetData();
    const int64_t sharedExpertNum = static_cast<int64_t>(info.expertNum) - info.moeExpertNum;
    const int64_t kAndShared = static_cast<int64_t>(info.K) + sharedExpertNum;
    OP_TILING_CHECK(static_cast<int64_t>(ffnDataShape[3]) != kAndShared,
                    OP_LOGE(nodeName, "ffn_token_data_shape[3] must be equal to K + sharedExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<int64_t>(ffnInfoShape[2]) != 2 + static_cast<int64_t>(info.BS) * kAndShared,
                    OP_LOGE(nodeName, "ffn_token_info_table_shape[2] is inconsistent with inputs"),
                    return ge::GRAPH_FAILED);

    uint32_t tilingKeyQuantMode = quantMode;
    uint32_t outDtype = ATTN_FFN_TILINGKEY_OUT_INT8;
    ResolveTilingKeyFields(quantMode, tilingKeyQuantMode, outDtype);

    OP_TILING_CHECK(context->SetScheduleMode(BATCH_MODE_SCHEDULE) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "failed to enable batch schedule mode"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckUrmaWinLayout(context, tilingData, tilingKeyQuantMode, outDtype) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "failed to check URMA win layout"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckUrmaUbCapacity(context, tilingData, tilingKeyQuantMode, outDtype) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "failed to check URMA UB capacity"), return ge::GRAPH_FAILED);
    ExtendWorkspaceForRankTables(context, tilingData);
    return SetV2TilingKey(context, tilingKeyQuantMode, outDtype, info.isScales, info.syncFlag == 1U, info.isActiveMask);
}
} // namespace

ge::graphStatus AttentionToFfnV2TilingFunc(gert::TilingContext* context)
{
    // Deployment marker: grep plog for this tag to verify the rebuilt package is actually
    // loaded (v10: per-rank counting in Phase 1 + compact bucket tables; the relay reads only
    // the served ranks' tables instead of rescanning the full token list per slot).
    OP_LOGI(context->GetNodeName(), "[AT2FFN_V2_URMA_FIX] v10 (rankBucketTables) active");
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE(context->GetNodeName(), "quant_mode is null"),
                    return ge::GRAPH_FAILED);
    uint32_t quantMode = static_cast<uint32_t>(*quantModePtr);

    const gert::StorageShape* scalesShape = context->GetOptionalInputShape(7U); // scales输入索引
    bool isScales = scalesShape != nullptr && scalesShape->GetStorageShape().GetDimNum() != 0U;
    OP_TILING_CHECK(CheckQuantMode(context->GetNodeName(), quantMode, isScales) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "quant mode validation failed"), return ge::GRAPH_FAILED);
    // Legacy path: quantMode 0 (FP16/BF16) and 2 (PERTOKEN+INT8) use v1 tiling
    bool useNewTiling = IsMxQuantMode(quantMode);
    if (useNewTiling) {
        return TilingNewQuantMode(context, quantMode);
    }
    AttentionToFFNTilingConfig config = MakeTilingConfig(false);

    auto ret = AttentionToFFNTilingFuncBase(context, config);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // Override V1's A3 key with the V2 A5 key.
    AttentionToFfnV2TilingData* tilingData = context->GetTilingData<AttentionToFfnV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "tilingData"),
                    return ge::GRAPH_FAILED);
    auto& info = tilingData->attentionToFfnV2Info;
    OP_TILING_CHECK(context->SetScheduleMode(BATCH_MODE_SCHEDULE) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "failed to enable batch schedule mode"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckUrmaWinLayout(context, tilingData, info.quantMode, ATTN_FFN_TILINGKEY_OUT_INT8) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "failed to check URMA win layout"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckUrmaUbCapacity(context, tilingData, info.quantMode, ATTN_FFN_TILINGKEY_OUT_INT8) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "failed to check URMA UB capacity"), return ge::GRAPH_FAILED);
    ExtendWorkspaceForRankTables(context, tilingData);
    return SetV2TilingKey(context, info.quantMode, ATTN_FFN_TILINGKEY_OUT_INT8, info.isScales, info.syncFlag == 1U,
                          info.isActiveMask);
}

struct AttentionToFfnV2CompileInfo {};
ge::graphStatus TilingParseForAttentionToFfnV2(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AttentionToFfnV2)
    .Tiling(AttentionToFfnV2TilingFunc)
    .TilingParse<AttentionToFfnV2CompileInfo>(TilingParseForAttentionToFfnV2);
} // namespace MC2Tiling
