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
constexpr size_t URMA_WORKSPACE_ALIGN = 32U;
constexpr size_t SCALE_PARAM_PAD_SIZE = 128U;
constexpr size_t MX_PARAM_PAD_SIZE = 256U;
constexpr uint32_t BATCH_MODE_SCHEDULE = 1U;
constexpr size_t RESERVED_WORKSPACE_SIZE = 1024 * 1024 * 64LL;

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

ge::graphStatus CheckQuantMode(const char *nodeName, uint32_t quantMode, bool isScales)
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
void ResolveTilingKeyFields(uint32_t quantMode, uint32_t &tilingKeyQuantMode, uint32_t &outDtype)
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

ge::graphStatus SetV2TilingKey(gert::TilingContext *context, uint32_t quantMode, uint32_t outDtype, bool isScales,
                               bool isSync, bool isActiveMask)
{
    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(quantMode, outDtype, isScales, isSync, isActiveMask, TILINGKEY_TPL_A5);
    context->SetTilingKey(tilingKey);
    OP_LOGD(context->GetNodeName(), "AttentionToFfnV2 cur case tilingKey is %lu", tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetUrmaWorkspace(gert::TilingContext *context, AttentionToFfnV2TilingData *tilingData,
                                 uint32_t quantMode, uint32_t outDtype)
{
    const size_t aivNum = static_cast<size_t>(tilingData->attentionToFfnV2Info.aivNum);
    OP_TILING_CHECK(aivNum == 0U, OP_LOGE(context->GetNodeName(), "aivNum must be greater than zero"),
                    return ge::GRAPH_FAILED);
    const size_t axisH = static_cast<size_t>(tilingData->attentionToFfnV2Info.H);
    const size_t xTypeSize = sizeof(uint16_t);

    size_t commuBytes;
    if (quantMode == ATTN_FFN_TILINGKEY_NO_QUANT) {
        commuBytes = axisH * xTypeSize;
    } else if (quantMode == ATTN_FFN_TILINGKEY_PERTOKEN_INT8) {
        commuBytes = axisH * sizeof(int8_t) + SCALE_PARAM_PAD_SIZE;
    } else {
        size_t outDataBytes;
        if (outDtype == ATTN_FFN_TILINGKEY_OUT_E2M1) {
            outDataBytes = (axisH + 1U) / 2U;
        } else {
            outDataBytes = axisH * sizeof(int8_t);
        }
        size_t hOutSizeAlign = (outDataBytes + 255U) / MX_PARAM_PAD_SIZE * MX_PARAM_PAD_SIZE;
        size_t mxScaleNum = ((axisH + 31U) / 32U + 1U) / 2U * 2U;
        commuBytes = hOutSizeAlign + mxScaleNum;
    }

    const size_t dataWorkspaceStride =
        (commuBytes + URMA_WORKSPACE_ALIGN - 1U) / URMA_WORKSPACE_ALIGN * URMA_WORKSPACE_ALIGN;
    OP_TILING_CHECK(dataWorkspaceStride > std::numeric_limits<size_t>::max() / aivNum,
                    OP_LOGE(context->GetNodeName(), "URMA data workspace size overflow"), return ge::GRAPH_FAILED);
    const size_t dataWorkspaceSize = aivNum * dataWorkspaceStride;
    const size_t flagWorkspaceSize = aivNum * URMA_FLAG_SLOT_SIZE;
    OP_TILING_CHECK(dataWorkspaceSize > std::numeric_limits<size_t>::max() - flagWorkspaceSize,
                    OP_LOGE(context->GetNodeName(), "URMA workspace size overflow"), return ge::GRAPH_FAILED);
    const size_t urmaWorkspaceSize = dataWorkspaceSize + flagWorkspaceSize + RESERVED_WORKSPACE_SIZE;

    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "workSpaces"),
                    return ge::GRAPH_FAILED);
    const size_t urmaWorkspaceOffset =
        (workSpaces[0] + URMA_WORKSPACE_ALIGN - 1U) / URMA_WORKSPACE_ALIGN * URMA_WORKSPACE_ALIGN;
    OP_TILING_CHECK(urmaWorkspaceOffset > std::numeric_limits<size_t>::max() - urmaWorkspaceSize,
                    OP_LOGE(context->GetNodeName(), "workspace size overflow"), return ge::GRAPH_FAILED);
    tilingData->attentionToFfnV2Info.urmaWorkspaceOffset = static_cast<uint64_t>(urmaWorkspaceOffset);
    workSpaces[0] = urmaWorkspaceOffset + urmaWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingNewQuantMode(gert::TilingContext *context, uint32_t quantMode)
{
    const char *nodeName = context->GetNodeName();

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
    config.allowMxQuantMode = true;

    auto ret = AttentionToFFNTilingFuncBase(context, config);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    AttentionToFfnV2TilingData *tilingData = context->GetTilingData<AttentionToFfnV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);
    auto &info = tilingData->attentionToFfnV2Info;
    info.windowType = 0U;

    auto attrs = context->GetAttrs();
    auto ffnDataShapeAttr = attrs->GetListInt(ATTR_FFN_TOKEN_DATA_SHAPE_INDEX);
    auto ffnInfoShapeAttr = attrs->GetListInt(ATTR_FFN_TOKEN_INFO_SHAPE_INDEX);
    const int64_t *ffnDataShape = ffnDataShapeAttr->GetData();
    const int64_t *ffnInfoShape = ffnInfoShapeAttr->GetData();
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
    OP_TILING_CHECK(SetUrmaWorkspace(context, tilingData, tilingKeyQuantMode, outDtype) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "failed to set URMA workspace"), return ge::GRAPH_FAILED);
    return SetV2TilingKey(context, tilingKeyQuantMode, outDtype, info.isScales, info.syncFlag == 1U, info.isActiveMask);
}
} // namespace

ge::graphStatus AttentionToFfnV2TilingFunc(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE(context->GetNodeName(), "quant_mode is null"),
                    return ge::GRAPH_FAILED);
    uint32_t quantMode = static_cast<uint32_t>(*quantModePtr);

    const gert::StorageShape *scalesShape = context->GetOptionalInputShape(7U); // scales输入索引
    bool isScales = scalesShape != nullptr && scalesShape->GetStorageShape().GetDimNum() != 0U;
    OP_TILING_CHECK(CheckQuantMode(context->GetNodeName(), quantMode, isScales) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "quant mode validation failed"), return ge::GRAPH_FAILED);
    // Legacy path: quantMode 0 (FP16/BF16) and 2 (PERTOKEN+INT8) use v1 tiling
    bool useNewTiling = IsMxQuantMode(quantMode);
    if (useNewTiling) {
        return TilingNewQuantMode(context, quantMode);
    }
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
    config.attrGroupIndex = 0U;
    config.attrWorldSizeIndex = 1U;
    config.attrFfnTokenInfoTableShapeIndex = 2U;
    config.attrFfnTokenDataShapeIndex = 3U;
    config.attrAttnTokenInfoTableShapeIndex = 4U;
    config.attrMoeExpertNumIndex = 5U;
    config.attrQuantModeIndex = 6U;
    config.attrSyncFlagIndex = ATTR_SYNC_FLAG_INDEX;
    config.attrFfnStartRankIdIndex = ATTR_FFN_START_RANK_ID_INDEX;
    config.attrCclBufferSizeIndex = ATTR_CCL_BUFFER_SIZE_INDEX;
    config.isMc2Context = true;

    auto ret = AttentionToFFNTilingFuncBase(context, config);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // Override V1's A3 key with the V2 A5 key.
    AttentionToFfnV2TilingData *tilingData = context->GetTilingData<AttentionToFfnV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "tilingData"),
                    return ge::GRAPH_FAILED);
    auto &info = tilingData->attentionToFfnV2Info;
    OP_TILING_CHECK(context->SetScheduleMode(BATCH_MODE_SCHEDULE) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "failed to enable batch schedule mode"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        SetUrmaWorkspace(context, tilingData, info.quantMode, ATTN_FFN_TILINGKEY_OUT_INT8) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "failed to set URMA workspace"), return ge::GRAPH_FAILED);
    return SetV2TilingKey(context, info.quantMode, ATTN_FFN_TILINGKEY_OUT_INT8, info.isScales, info.syncFlag == 1U,
                          info.isActiveMask);
}

struct AttentionToFfnV2CompileInfo {};
ge::graphStatus TilingParseForAttentionToFfnV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AttentionToFfnV2)
    .Tiling(AttentionToFfnV2TilingFunc)
    .TilingParse<AttentionToFfnV2CompileInfo>(TilingParseForAttentionToFfnV2);
} // namespace MC2Tiling
