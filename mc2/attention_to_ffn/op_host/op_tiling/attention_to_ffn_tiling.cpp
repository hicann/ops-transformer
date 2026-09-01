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
 * \file attention_to_ffn_tiling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "attention_to_ffn_tiling_base.h"
#include "../../op_kernel/attention_to_ffn_tiling.h"
#include "../../op_kernel/attention_to_ffn_tiling_key.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace MC2Tiling {
constexpr char ATTN_TO_FFN_WIN_TYPE_ENV[] = "ASCEND_ATTN_TO_FFN_WIN_TYPE";

constexpr uint32_t INDEX_TWO = 2;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 128;
constexpr uint64_t MB_SIZE = 1024 * 1024;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
const char *ATTN_FFN_INNER_DEBUG = "AttentionToFFN Tiling Debug";

constexpr uint32_t INFO_HS_INDEX = 4;
constexpr uint32_t INFO_TABLE_LAST_DIM_NUM_INDEX = 2;
constexpr uint32_t NO_QUANT_MODE_WIN_SIZE = 2;

constexpr uint32_t ONE_DIM = 1;
constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t THREE_DIMS = 3;
constexpr uint32_t DIM2 = 2;
constexpr uint32_t DIM3 = 3;
constexpr uint32_t DIM5 = 5;

constexpr uint32_t SYNC_MODE = 1;
constexpr uint32_t UNQUANT_MODE = 0;
constexpr uint32_t STATIC_QUANT_MODE = 1;
constexpr uint32_t DYNAMIC_QUANT_MODE = 2;
constexpr uint32_t NO_SCALES = 0;
constexpr uint32_t STATIC_SCALES = 1;
constexpr uint32_t DYNAMIC_SCALES = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
constexpr int64_t SHARED_EXPERT_MAX_NUM = 4;
constexpr int64_t BS_MAX = 512;
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 8192;
constexpr int64_t K_MAX = 16;
constexpr int64_t MAX_WORLD_SIZE = 768L; // 384 * 2
constexpr int64_t MIN_WORLD_SIZE = 2;

constexpr uint32_t USER_QUANT_MODE_MX_E5M2 = 3U;
constexpr uint32_t USER_QUANT_MODE_MX_E4M3 = 4U;
constexpr uint32_t USER_QUANT_MODE_MX_E2M1 = 5U;
constexpr uint32_t USER_QUANT_MODE_MX_CLIP_E5M2 = 6U;
constexpr uint32_t USER_QUANT_MODE_MX_CLIP_E4M3 = 7U;

static bool IsMxQuantMode(uint32_t quantMode)
{
    return quantMode == USER_QUANT_MODE_MX_E5M2 || quantMode == USER_QUANT_MODE_MX_E4M3 ||
           quantMode == USER_QUANT_MODE_MX_E2M1 || quantMode == USER_QUANT_MODE_MX_CLIP_E5M2 ||
           quantMode == USER_QUANT_MODE_MX_CLIP_E4M3;
}

static void PrintTilingDataInfo(const char *nodeName, AttentionToFFNTilingData &tilingData)
{
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "X is %u.", tilingData.attentionToFFNInfo.X);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "BS is %u.", tilingData.attentionToFFNInfo.BS);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "H is %u.", tilingData.attentionToFFNInfo.H);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "K is %u.", tilingData.attentionToFFNInfo.K);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "L is %u.", tilingData.attentionToFFNInfo.L);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "expertNum is %u.", tilingData.attentionToFFNInfo.expertNum);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "moeExpertNum is %u.", tilingData.attentionToFFNInfo.moeExpertNum);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "HS is %u.", tilingData.attentionToFFNInfo.HS);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "expRankTableM is %u.", tilingData.attentionToFFNInfo.expRankTableM);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "microBatchNum is %u.", tilingData.attentionToFFNInfo.microBatchNum);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "attentionWorkerNum is %u.", tilingData.attentionToFFNInfo.attentionWorkerNum);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "infoTableLastDimNum is %u", tilingData.attentionToFFNInfo.infoTableLastDimNum);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "aivNum is %u.", tilingData.attentionToFFNInfo.aivNum);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "worldSize is %u.", tilingData.attentionToFFNInfo.worldSize);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "syncFlag is %u.", tilingData.attentionToFFNInfo.syncFlag);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "quantMode is %u.", tilingData.attentionToFFNInfo.quantMode);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "isScales is %u.", tilingData.attentionToFFNInfo.isScales);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "windowType is %u.", tilingData.attentionToFFNInfo.windowType);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "totalUbSize is %lu.", tilingData.attentionToFFNInfo.totalUbSize);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "totalWinSize is %lu.", tilingData.attentionToFFNInfo.totalWinSize);
}

static bool CheckAndSetShapeAttrs(gert::TilingContext *context, const char *nodeName,
                                  AttentionToFFNTilingData &tilingData, const AttentionToFFNTilingConfig &config)
{
    auto attrs = context->GetAttrs();
    auto worldSizePtr = attrs->GetAttrPointer<int>(config.attrWorldSizeIndex);
    auto ffnTokenInfoTableDimNum = attrs->GetListInt(config.attrFfnTokenInfoTableShapeIndex)->GetSize();
    auto ffnTokenDataDimNum = attrs->GetListInt(config.attrFfnTokenDataShapeIndex)->GetSize();
    auto attnTokenDataDimNum = attrs->GetListInt(config.attrAttnTokenInfoTableShapeIndex)->GetSize();
    auto ffnTokenInfoTableShape = attrs->GetListInt(config.attrFfnTokenInfoTableShapeIndex)->GetData();
    auto ffnTokenDataShape = attrs->GetListInt(config.attrFfnTokenDataShapeIndex)->GetData();
    auto attnTokenDataShape = attrs->GetListInt(config.attrAttnTokenInfoTableShapeIndex)->GetData();
    const gert::StorageShape *xShape = context->GetInputShape(config.xIndex);
    int32_t bs = xShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(ffnTokenInfoTableDimNum != DIM3,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "ffn_token_info_table_shape",
                                                 (std::to_string(ffnTokenInfoTableDimNum) + "D").c_str(), "3D"),
                    return false);
    OP_TILING_CHECK(ffnTokenDataDimNum != DIM5,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "ffn_token_data_shape",
                                                 (std::to_string(ffnTokenDataDimNum) + "D").c_str(), "5D"),
                    return false);
    OP_TILING_CHECK(attnTokenDataDimNum != DIM3,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "attn_token_info_table_shape",
                                                 (std::to_string(attnTokenDataDimNum) + "D").c_str(), "3D"),
                    return false);

    OP_TILING_CHECK(ffnTokenInfoTableShape[0] != ffnTokenDataShape[0],
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "ffn_token_info_table_shape",
                        (std::string("[") + std::to_string(ffnTokenInfoTableShape[0]) + "]").c_str(),
                        "Dim0 of ffn_token_info_table_shape must be equal to dim0 of ffn_token_data_shape"),
                    return false);
    OP_TILING_CHECK(ffnTokenInfoTableShape[1] != ffnTokenDataShape[1],
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "ffn_token_info_table_shape",
                        (std::string("[") + std::to_string(ffnTokenInfoTableShape[1]) + "]").c_str(),
                        "Dim1 of ffn_token_info_table_shape must be equal to dim1 of ffn_token_data_shape"),
                    return false);
    OP_TILING_CHECK(ffnTokenDataShape[DIM2] != bs,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "ffn_token_data_shape",
                        (std::string("[") + std::to_string(ffnTokenDataShape[DIM2]) + "]").c_str(),
                        "Dim2 of ffn_token_data_shape must be equal to bs"),
                    return false);
    OP_TILING_CHECK(attnTokenDataShape[0] != ffnTokenInfoTableShape[1],
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "attn_token_info_table_shape",
                        (std::string("[") + std::to_string(attnTokenDataShape[0]) + "]").c_str(),
                        "Dim0 of attn_token_info_table_shape must be equal to dim1 of ffn_token_info_table_shape"),
                    return false);
    OP_TILING_CHECK(
        attnTokenDataShape[1] != bs,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "attn_token_info_table_shape",
                                              (std::string("[") + std::to_string(attnTokenDataShape[1]) + "]").c_str(),
                                              "Dim1 of attn_token_info_table_shape must be equal to dim1 of x"),
        return false);
    OP_TILING_CHECK(attnTokenDataShape[DIM2] != ffnTokenDataShape[DIM3],
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "attn_token_info_table_shape",
                        (std::string("[") + std::to_string(attnTokenDataShape[DIM2]) + "]").c_str(),
                        "Dim2 of attn_token_info_table_shape must be equal to dim3 of ffn_token_data_shape"),
                    return false);

    // 校验 attentionWorkerNum 的值
    uint32_t attentionWorkerNum = ffnTokenDataShape[0];
    OP_TILING_CHECK(
        attentionWorkerNum <= 0 || attentionWorkerNum >= *worldSizePtr,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "ffn_token_data_shape", std::to_string(attentionWorkerNum).c_str(),
                                  (std::string("(0, worldSize=") + std::to_string(*worldSizePtr) + ")").c_str()),
        return false);

    tilingData.attentionToFFNInfo.attentionWorkerNum = attentionWorkerNum;
    tilingData.attentionToFFNInfo.microBatchNum = ffnTokenDataShape[1];
    tilingData.attentionToFFNInfo.HS = ffnTokenDataShape[INFO_HS_INDEX];
    tilingData.attentionToFFNInfo.infoTableLastDimNum = ffnTokenInfoTableShape[INFO_TABLE_LAST_DIM_NUM_INDEX];
    return true;
}

static bool CheckAndSetAttrs(gert::TilingContext *context, const char *nodeName, AttentionToFFNTilingData &tilingData,
                             std::string &group, const AttentionToFFNTilingConfig &config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return false);

    auto groupPtr = attrs->GetAttrPointer<char>(config.attrGroupIndex);
    auto worldSizePtr = attrs->GetAttrPointer<int>(config.attrWorldSizeIndex);
    auto quantModePtr = attrs->GetAttrPointer<int>(config.attrQuantModeIndex);
    auto syncFlagPtr = attrs->GetAttrPointer<int>(config.attrSyncFlagIndex);
    auto ffnStartRankIdPtr = attrs->GetAttrPointer<int>(config.attrFfnStartRankIdIndex);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int>(config.attrMoeExpertNumIndex);

    OP_TILING_CHECK(groupPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "group"), return false);
    OP_TILING_CHECK(worldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "worldSize"), return false);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "moeExpertNum"), return false);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "quantMode"), return false);
    OP_TILING_CHECK(syncFlagPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "syncFlag"), return false);
    OP_TILING_CHECK(ffnStartRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "ffnStartRankId"), return false);

    uint32_t moeExpertNum = *moeExpertNumPtr;
    uint32_t worldSize = *worldSizePtr;

    OP_TILING_CHECK(
        (moeExpertNum <= 0) || (moeExpertNum > MOE_EXPERT_MAX_NUM),
        OP_LOGE_WITH_INVALID_ATTR(ATTN_FFN_INNER_DEBUG, "moe_expert_num", std::to_string(moeExpertNum).c_str(),
                                  (std::string("(0, ") + std::to_string(MOE_EXPERT_MAX_NUM) + "]").c_str()),
        return false);
    if (config.allowMxQuantMode) {
        OP_TILING_CHECK(*quantModePtr != UNQUANT_MODE && *quantModePtr != DYNAMIC_QUANT_MODE &&
                            !IsMxQuantMode(static_cast<uint32_t>(*quantModePtr)),
                        OP_LOGE_WITH_INVALID_ATTR(ATTN_FFN_INNER_DEBUG, "quant_mode",
                                                  std::to_string(*quantModePtr).c_str(), "0, 2, 3, 4, 5, 6 or 7"),
                        return false);
    } else {
        OP_TILING_CHECK(
            (*quantModePtr < static_cast<int64_t>(NO_SCALES)) || (*quantModePtr > static_cast<int64_t>(DYNAMIC_SCALES)),
            OP_LOGE_WITH_INVALID_ATTR(ATTN_FFN_INNER_DEBUG, "quant_mode", std::to_string(*quantModePtr).c_str(),
                                      (std::string("[0, ") + std::to_string(DYNAMIC_SCALES) + "]").c_str()),
            return false);
    }
    OP_TILING_CHECK((*syncFlagPtr != 0) && (*syncFlagPtr != 1),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "sync_flag", std::to_string(*syncFlagPtr).c_str(), "0 or 1"),
                    return false);
    OP_TILING_CHECK(
        (*ffnStartRankIdPtr >= *worldSizePtr) || (*ffnStartRankIdPtr < 0),
        OP_LOGE_WITH_INVALID_ATTR(ATTN_FFN_INNER_DEBUG, "ffn_start_rank_id", std::to_string(*ffnStartRankIdPtr).c_str(),
                                  (std::string("[0, ") + std::to_string(*worldSizePtr) + ")").c_str()),
        return false);
    OP_TILING_CHECK(
        (worldSize < MIN_WORLD_SIZE) || (worldSize > MAX_WORLD_SIZE),
        OP_LOGE_WITH_INVALID_ATTR(
            nodeName, "world_size", std::to_string(worldSize).c_str(),
            (std::string("[") + std::to_string(MIN_WORLD_SIZE) + ", " + std::to_string(MAX_WORLD_SIZE) + "]").c_str()),
        return false);

    OP_TILING_CHECK(!CheckAndSetShapeAttrs(context, nodeName, tilingData, config),
                    OP_LOGE(nodeName, "Check and set shape list attrs failed!"), return false);

    tilingData.attentionToFFNInfo.moeExpertNum = static_cast<uint32_t>(moeExpertNum);
    tilingData.attentionToFFNInfo.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.attentionToFFNInfo.syncFlag = static_cast<uint32_t>(*syncFlagPtr);
    tilingData.attentionToFFNInfo.ffnStartRankId = static_cast<uint32_t>(*ffnStartRankIdPtr);
    tilingData.attentionToFFNInfo.worldSize = static_cast<uint32_t>(worldSize);
    OP_LOGD(nodeName, "group = %s", groupPtr);
    group = string(groupPtr);

    return true;
}

static bool CheckInputForScales(const gert::StorageShape *scalesShape, const int32_t expertRankTableDim0,
                                const int32_t expertRankTableDim1, const int32_t xDim2)
{
    // 校验数据各维度值
    int32_t scalesDim0 = scalesShape->GetStorageShape().GetDim(0);
    int32_t scalesDim1 = scalesShape->GetStorageShape().GetDim(1);
    int32_t scalesDim2 = scalesShape->GetStorageShape().GetDim(2);

    OP_TILING_CHECK(scalesDim0 != expertRankTableDim0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ATTN_FFN_INNER_DEBUG, "scales",
                                                          (std::string("[") + std::to_string(scalesDim0) + "]").c_str(),
                                                          "Dim0 of scales must be equal to dim0 of expertRankTable"),
                    return false);
    OP_TILING_CHECK(scalesDim1 != expertRankTableDim1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ATTN_FFN_INNER_DEBUG, "scales",
                                                          (std::string("[") + std::to_string(scalesDim1) + "]").c_str(),
                                                          "Dim1 of scales must be equal to dim1 of expertRankTable"),
                    return false);
    OP_TILING_CHECK(scalesDim2 != xDim2,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ATTN_FFN_INNER_DEBUG, "scales",
                                                          (std::string("[") + std::to_string(scalesDim2) + "]").c_str(),
                                                          "Dim2 of scales must be equal to dim2 of x"),
                    return false);

    return true;
}

static bool CheckInputForActiveMask(const gert::StorageShape *activeMaskShape, const int32_t xDim0, const int32_t xDim1)
{
    int32_t activeMaskDim0 = activeMaskShape->GetStorageShape().GetDim(0);
    int32_t activeMaskDim1 = activeMaskShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(
        activeMaskDim0 != xDim0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ATTN_FFN_INNER_DEBUG, "active_mask",
                                              (std::string("[") + std::to_string(activeMaskDim0) + "]").c_str(),
                                              "Dim0 of active_mask must be equal to dim0 of x"),
        return false);
    OP_TILING_CHECK(
        activeMaskDim1 != xDim1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ATTN_FFN_INNER_DEBUG, "active_mask",
                                              (std::string("[") + std::to_string(activeMaskDim1) + "]").c_str(),
                                              "Dim1 of active_mask must be equal to dim1 of x"),
        return false);

    return true;
}

static void InitTilingDataByXDim(AttentionToFFNTilingData &tilingData, const int32_t xDim0, const int32_t xDim1,
                                 const int32_t xDim2)
{
    tilingData.attentionToFFNInfo.X = xDim0;
    tilingData.attentionToFFNInfo.BS = xDim1;
    tilingData.attentionToFFNInfo.H = xDim2;
}

static void InitTilingDataByExpert(AttentionToFFNTilingData &tilingData, const int32_t expertIdsDim2,
                                   const int32_t expertRankTableDim0, const int32_t expertRankTableDim1,
                                   const int32_t expertRankTableDim2)
{
    tilingData.attentionToFFNInfo.K = expertIdsDim2;
    tilingData.attentionToFFNInfo.L = expertRankTableDim0;
    tilingData.attentionToFFNInfo.expertNum = expertRankTableDim1; // 所有专家总数，包含1个共享+所有moe专家
    tilingData.attentionToFFNInfo.expRankTableM = expertRankTableDim2;
    tilingData.attentionToFFNInfo.sharedExpertNum = expertRankTableDim1 - tilingData.attentionToFFNInfo.moeExpertNum;
}

static bool CheckTensorDataType(gert::TilingContext *context, const char *nodeName, const bool isScales,
                                const bool isActiveMask, const AttentionToFFNTilingConfig &config)
{
    OP_TILING_CHECK(context->GetInputDesc(config.xIndex) == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"),
                    return false);
    auto xDesc = context->GetInputDesc(config.xIndex);
    OP_TILING_CHECK(
        (xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(), "FLOAT16, BF16"),
        return false);

    OP_TILING_CHECK(context->GetInputDesc(config.sessionIdIndex) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "sessionIdDesc"), return false);
    auto sessionIdDesc = context->GetInputDesc(config.sessionIdIndex);
    OP_TILING_CHECK(sessionIdDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "session_id",
                                              Ops::Base::ToString(sessionIdDesc->GetDataType()).c_str(), "INT32"),
                    return false);

    OP_TILING_CHECK(context->GetInputDesc(config.microBatchIdIndex) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "microBatchIdDesc"), return false);
    auto microBatchIdDesc = context->GetInputDesc(config.microBatchIdIndex);
    OP_TILING_CHECK(microBatchIdDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "micro_batch_id",
                                              Ops::Base::ToString(microBatchIdDesc->GetDataType()).c_str(), "INT32"),
                    return false);

    OP_TILING_CHECK(context->GetInputDesc(config.layerIdIndex) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "layerIdDesc"), return false);
    auto layerIdDesc = context->GetInputDesc(config.layerIdIndex);
    OP_TILING_CHECK(layerIdDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "layer_id",
                                              Ops::Base::ToString(layerIdDesc->GetDataType()).c_str(), "INT32"),
                    return false);

    OP_TILING_CHECK(context->GetInputDesc(config.expertIdsIndex) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdDesc"), return false);
    auto expertIdDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(expertIdDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expert_ids",
                                              Ops::Base::ToString(expertIdDesc->GetDataType()).c_str(), "INT32"),
                    return false);

    OP_TILING_CHECK(context->GetInputDesc(config.expertRankTableIndex) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertRankTableDesc"), return false);
    auto expertRankTableDesc = context->GetInputDesc(config.expertRankTableIndex);
    OP_TILING_CHECK(expertRankTableDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expert_rank_table",
                                              Ops::Base::ToString(expertRankTableDesc->GetDataType()).c_str(), "INT32"),
                    return false);

    if (isScales) {
        OP_TILING_CHECK(context->GetOptionalInputDesc(config.scalesIndex) == nullptr,
                        OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesDesc"), return false);
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        OP_TILING_CHECK(scalesDesc->GetDataType() != ge::DT_FLOAT,
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales",
                                                  Ops::Base::ToString(scalesDesc->GetDataType()).c_str(), "FLOAT"),
                        return false);
    }

    if (isActiveMask) {
        OP_TILING_CHECK(context->GetOptionalInputDesc(config.activeMaskIndex) == nullptr,
                        OP_LOGE_WITH_INVALID_INPUT(nodeName, "activeMaskDesc"), return false);
        auto activeMaskDesc = context->GetOptionalInputDesc(config.activeMaskIndex);
        OP_TILING_CHECK(activeMaskDesc->GetDataType() != ge::DT_BOOL,
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "active_mask",
                                                  Ops::Base::ToString(activeMaskDesc->GetDataType()).c_str(), "BOOL"),
                        return false);
    }

    return true;
}

static bool CheckTensorFormat(gert::TilingContext *context, const char *nodeName, const bool isScales,
                              const bool isActiveMask, const AttentionToFFNTilingConfig &config)
{
    OP_TILING_CHECK(context->GetInputDesc(config.xIndex) == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"),
                    return false);
    auto xDesc = context->GetInputDesc(config.xIndex);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "x",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat()))).c_str(),
            "ND"),
        return false);

    OP_TILING_CHECK(context->GetInputDesc(config.sessionIdIndex) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "sessionIdDesc"), return false);
    auto sessionIdDesc = context->GetInputDesc(config.sessionIdIndex);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(sessionIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "session_id",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(sessionIdDesc->GetStorageFormat())))
                .c_str(),
            "ND"),
        return false);

    OP_TILING_CHECK(context->GetInputDesc(config.microBatchIdIndex) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "microBatchIdDesc"), return false);
    auto microBatchIdDesc = context->GetInputDesc(config.microBatchIdIndex);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(microBatchIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "micro_batch_id",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(microBatchIdDesc->GetStorageFormat())))
                .c_str(),
            "ND"),
        return false);

    OP_TILING_CHECK(context->GetInputDesc(config.layerIdIndex) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "layerIdDesc"), return false);
    auto layerIdDesc = context->GetInputDesc(config.layerIdIndex);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(layerIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "layer_id",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(layerIdDesc->GetStorageFormat()))).c_str(),
            "ND"),
        return false);

    OP_TILING_CHECK(context->GetInputDesc(config.expertIdsIndex) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdDesc"), return false);
    auto expertIdDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "expert_ids",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdDesc->GetStorageFormat())))
                .c_str(),
            "ND"),
        return false);

    OP_TILING_CHECK(context->GetInputDesc(config.expertRankTableIndex) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertRankTableDesc"), return false);
    auto expertRankTableDesc = context->GetInputDesc(config.expertRankTableIndex);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertRankTableDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "expert_rank_table",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(expertRankTableDesc->GetStorageFormat())))
                .c_str(),
            "ND"),
        return false);

    if (isScales) {
        OP_TILING_CHECK(context->GetOptionalInputDesc(config.scalesIndex) == nullptr,
                        OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesDesc"), return false);
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(scalesDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
            OP_LOGE_FOR_INVALID_FORMAT(
                nodeName, "scales",
                Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(scalesDesc->GetStorageFormat())))
                    .c_str(),
                "ND"),
            return false);
    }

    if (isActiveMask) {
        OP_TILING_CHECK(context->GetOptionalInputDesc(config.activeMaskIndex) == nullptr,
                        OP_LOGE_WITH_INVALID_INPUT(nodeName, "activeMaskDesc"), return false);
        auto activeMaskDesc = context->GetOptionalInputDesc(config.activeMaskIndex);
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(activeMaskDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
            OP_LOGE_FOR_INVALID_FORMAT(
                nodeName, "active_mask",
                Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(activeMaskDesc->GetStorageFormat())))
                    .c_str(),
                "ND"),
            return false);
    }

    return true;
}

static bool CheckTensorDim(gert::TilingContext *context, const char *nodeName, const bool isScales,
                           const bool isActiveMask, const AttentionToFFNTilingConfig &config)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return false);
    OP_TILING_CHECK(
        xStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "x", (std::to_string(xStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
        return false);

    const gert::StorageShape *sessionIdStorageShape = context->GetInputShape(config.sessionIdIndex);
    OP_TILING_CHECK(sessionIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "session_id"), return false);
    OP_TILING_CHECK(sessionIdStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "session_id",
                        (std::to_string(sessionIdStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return false);

    const gert::StorageShape *microBatchIdStorageShape = context->GetInputShape(config.microBatchIdIndex);
    OP_TILING_CHECK(microBatchIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "micro_batch_id"),
                    return false);
    OP_TILING_CHECK(microBatchIdStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "micro_batch_id",
                        (std::to_string(microBatchIdStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return false);

    const gert::StorageShape *layerIdStorageShape = context->GetInputShape(config.layerIdIndex);
    OP_TILING_CHECK(layerIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "layer_id"), return false);
    OP_TILING_CHECK(layerIdStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "layer_id",
                        (std::to_string(layerIdStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return false);

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expert_ids"), return false);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "expert_ids",
                        (std::to_string(expertIdStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
                    return false);

    const gert::StorageShape *expertRankTableStorageShape = context->GetInputShape(config.expertRankTableIndex);
    OP_TILING_CHECK(expertRankTableStorageShape == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(ATTN_FFN_INNER_DEBUG, "expert_rank_table"), return false);
    OP_TILING_CHECK(
        expertRankTableStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            ATTN_FFN_INNER_DEBUG, "expert_rank_table",
            (std::to_string(expertRankTableStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
        return false);

    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scales"), return false);
        OP_TILING_CHECK(scalesStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(
                            nodeName, "scales",
                            (std::to_string(scalesStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
                        return false);
    }

    if (isActiveMask) {
        const gert::StorageShape *activeMaskStorageShape = context->GetOptionalInputShape(config.activeMaskIndex);
        OP_TILING_CHECK(activeMaskStorageShape == nullptr,
                        OP_LOGE_WITH_INVALID_INPUT(ATTN_FFN_INNER_DEBUG, "active_mask"), return false);
        OP_TILING_CHECK(
            activeMaskStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
            OP_LOGE_FOR_INVALID_SHAPEDIM(
                ATTN_FFN_INNER_DEBUG, "active_mask",
                (std::to_string(activeMaskStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
            return false);
    }

    return true;
}

static bool CheckTensorShapeAndSetTinglingData(gert::TilingContext *context, const char *nodeName,
                                               AttentionToFFNTilingData &tilingData, const bool isScales,
                                               const bool isActiveMask, const AttentionToFFNTilingConfig &config)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const gert::StorageShape *sessionIdStorageShape = context->GetInputShape(config.sessionIdIndex);
    const gert::StorageShape *microBatchIdStorageShape = context->GetInputShape(config.microBatchIdIndex);
    const gert::StorageShape *layerIdStorageShape = context->GetInputShape(config.layerIdIndex);
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(config.expertIdsIndex);
    const gert::StorageShape *expertRankTableStorageShape = context->GetInputShape(config.expertRankTableIndex);
    const int32_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int32_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    const int32_t xDim2 = xStorageShape->GetStorageShape().GetDim(DIM2);
    const int32_t sessionIdDim0 = sessionIdStorageShape->GetStorageShape().GetDim(0);
    const int32_t microBatchIdDim0 = microBatchIdStorageShape->GetStorageShape().GetDim(0);
    const int32_t layerIdDim0 = layerIdStorageShape->GetStorageShape().GetDim(0);
    const int32_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    const int32_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);
    const int32_t expertIdsDim2 = expertIdsStorageShape->GetStorageShape().GetDim(DIM2);
    const int32_t expertRankTableDim0 = expertRankTableStorageShape->GetStorageShape().GetDim(0);
    const int32_t expertRankTableDim1 = expertRankTableStorageShape->GetStorageShape().GetDim(1);
    const int32_t expertRankTableDim2 = expertRankTableStorageShape->GetStorageShape().GetDim(DIM2);
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.attentionToFFNInfo.moeExpertNum);

    OP_TILING_CHECK(
        xDim0 != 1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            ATTN_FFN_INNER_DEBUG, "x", (std::string("[") + std::to_string(xDim0) + "]").c_str(), "Dim0 of x must be 1"),
        return false);
    OP_TILING_CHECK((xDim1 <= 0) || (xDim1 > BS_MAX),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "x", std::to_string(xDim1).c_str(),
                                              (std::string("(0, ") + std::to_string(BS_MAX) + "]").c_str()),
                    return false);
    OP_TILING_CHECK((xDim2 < H_MIN) || (xDim2 > H_MAX),
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "x", std::to_string(xDim2).c_str(),
                        (std::string("[") + std::to_string(H_MIN) + ", " + std::to_string(H_MAX) + "]").c_str()),
                    return false);
    OP_TILING_CHECK(
        (xDim0 != sessionIdDim0) || (xDim0 != microBatchIdDim0) || (xDim0 != layerIdDim0) || (xDim0 != expertIdsDim0),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            ATTN_FFN_INNER_DEBUG, "session_id, micro_batch_id, layer_id and expert_ids",
            (std::string("[") + std::to_string(sessionIdDim0) + ", " + std::to_string(microBatchIdDim0) + ", " +
             std::to_string(layerIdDim0) + ", " + std::to_string(expertIdsDim0) + "]")
                .c_str(),
            "Dim0 of session_id, micro_batch_id, layer_id and expert_ids must be 1"),
        return false);
    OP_TILING_CHECK(expertIdsDim1 != xDim1,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, "expert_ids and x",
                        (std::string("[") + std::to_string(expertIdsDim1) + ", " + std::to_string(xDim1) + "]").c_str(),
                        "Dim1 of expert_ids must be equal to dim1 of x"),
                    return false);
    OP_TILING_CHECK((expertIdsDim2 <= 0) || (expertIdsDim2 > K_MAX) || (expertIdsDim2 > moeExpertNum),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "expert_ids", std::to_string(expertIdsDim2).c_str(),
                                              (std::string("(0, min(") + std::to_string(K_MAX) +
                                               ", moe_expert_num=" + std::to_string(moeExpertNum) + ")]")
                                                  .c_str()),
                    return false);
    OP_TILING_CHECK(
        expertRankTableDim0 != 1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "expert_rank_table",
                                              (std::string("[") + std::to_string(expertRankTableDim0) + "]").c_str(),
                                              "Dim0 of expert_rank_table must be 1"),
        return false);
    OP_TILING_CHECK(
        (expertRankTableDim1 < moeExpertNum) || (expertRankTableDim1 > moeExpertNum + SHARED_EXPERT_MAX_NUM),
        OP_LOGE_FOR_INVALID_VALUE(ATTN_FFN_INNER_DEBUG, "expert_rank_table",
                                  std::to_string(expertRankTableDim1).c_str(),
                                  (std::string("[") + std::to_string(moeExpertNum) + ", " +
                                   std::to_string(moeExpertNum + SHARED_EXPERT_MAX_NUM) + "]")
                                      .c_str()),
        return false);

    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
        OP_TILING_CHECK(!CheckInputForScales(scalesStorageShape, expertRankTableDim0, expertRankTableDim1, xDim2),
                        OP_LOGE(nodeName, "Check input for scales shape failed!"), return false);
    }

    if (isActiveMask) {
        const gert::StorageShape *activeMaskStorageShape = context->GetOptionalInputShape(config.activeMaskIndex);
        OP_TILING_CHECK(!CheckInputForActiveMask(activeMaskStorageShape, xDim0, xDim1),
                        OP_LOGE(nodeName, "Check input for activeMask shape failed!"), return false);
    }
    InitTilingDataByXDim(tilingData, xDim0, xDim1, xDim2);
    InitTilingDataByExpert(tilingData, expertIdsDim2, expertRankTableDim0, expertRankTableDim1, expertRankTableDim2);

    return true;
}

static bool CheckInputAndSetTilingData(gert::TilingContext *context, const char *nodeName,
                                       AttentionToFFNTilingData &tilingData, const AttentionToFFNTilingConfig &config)
{
    const gert::StorageShape *scalesShape = context->GetOptionalInputShape(config.scalesIndex);
    const gert::StorageShape *activeMaskShape = context->GetOptionalInputShape(config.activeMaskIndex);
    bool isScales = (scalesShape != nullptr && scalesShape->GetStorageShape().GetDimNum() != 0U);
    bool isActiveMask = (activeMaskShape != nullptr && activeMaskShape->GetStorageShape().GetDimNum() != 0U);

    // 校验quantMode和scale是否匹配
    uint32_t quantMode = tilingData.attentionToFFNInfo.quantMode;
    OP_TILING_CHECK(
        quantMode == STATIC_SCALES,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ATTN_FFN_INNER_DEBUG, "quant_mode", std::to_string(quantMode).c_str(),
                                              "The value of quant_mode does not support static quant"),
        return false);
    if (config.allowMxQuantMode && IsMxQuantMode(quantMode)) {
        OP_TILING_CHECK(isScales, OP_LOGE(nodeName, "scales must be absent in MX/MX_CLIP modes"), return false);
    } else {
        OP_TILING_CHECK((isScales && (quantMode == NO_SCALES)) || ((!isScales) && (quantMode == STATIC_SCALES)),
                        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                            ATTN_FFN_INNER_DEBUG, "scales and quant_mode",
                            (std::string("isScales=") + std::to_string(static_cast<int32_t>(isScales)) +
                             ", quantMode=" + std::to_string(quantMode))
                                .c_str(),
                            "The value of scales must match quant_mode"),
                        return false);
    }

    // 校验输入数据dim、format、dataType
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isScales, isActiveMask, config),
                    OP_LOGE(nodeName, "input tensor dim is invalid."), return false);
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, isScales, isActiveMask, config),
                    OP_LOGE(nodeName, "input tensor dataType is invalid."), return false);
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName, isScales, isActiveMask, config),
                    OP_LOGE(nodeName, "input tensor format is invalid."), return false);

    // 校验输入数据维度值
    OP_TILING_CHECK(!CheckTensorShapeAndSetTinglingData(context, nodeName, tilingData, isScales, isActiveMask, config),
                    OP_LOGE(nodeName, "input tensor format is invalid."), return false);

    tilingData.attentionToFFNInfo.isScales = isScales;
    tilingData.attentionToFFNInfo.isActiveMask = isActiveMask;

    return true;
}

static uint64_t CalTilingKey(const uint32_t syncFlag, const uint32_t quantMode, const bool activeMaskFlag)
{
    // TilingKey
    bool isSync = false;
    bool isQuant = false;
    bool isActiveMask = false;

    if (syncFlag == SYNC_MODE) {
        isSync = true;
    }
    if (quantMode == DYNAMIC_QUANT_MODE) {
        isQuant = true;
    }
    if (activeMaskFlag) {
        isActiveMask = true;
    }

    const uint64_t tilingKey = GET_TPL_TILING_KEY(isQuant, isSync, isActiveMask, TILINGKEY_TPL_A3);
    return tilingKey;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName,
                                    AttentionToFFNTilingData *tilingData)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workSpaces"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t ffnNum = tilingData->attentionToFFNInfo.worldSize - tilingData->attentionToFFNInfo.attentionWorkerNum;
    uint32_t ffnNumAlignSize =
        (ffnNum * sizeof(int32_t) + WORKSPACE_ELEMENT_OFFSET - 1) / WORKSPACE_ELEMENT_OFFSET * WORKSPACE_ELEMENT_OFFSET;
    workSpaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + static_cast<size_t>(ffnNumAlignSize * aivNum);
    return ge::GRAPH_SUCCESS;
}

static void CalWinSize(AttentionToFFNTilingData &tilingData, const uint32_t quantMode, uint64_t &neededSize,
                       uint64_t &viableWindowSize)
{
    uint64_t BS = static_cast<uint64_t>(tilingData.attentionToFFNInfo.BS);
    uint64_t K = static_cast<uint64_t>(tilingData.attentionToFFNInfo.K);
    uint64_t HS = static_cast<uint64_t>(tilingData.attentionToFFNInfo.HS);
    uint64_t microBatchNum = static_cast<uint64_t>(tilingData.attentionToFFNInfo.microBatchNum);
    uint64_t attentionWorkerNum = static_cast<uint64_t>(tilingData.attentionToFFNInfo.attentionWorkerNum);
    uint64_t infoTableLastDimNum = static_cast<uint64_t>(tilingData.attentionToFFNInfo.infoTableLastDimNum);
    uint64_t sharedExpertNum = static_cast<uint64_t>(tilingData.attentionToFFNInfo.sharedExpertNum);

    uint64_t tokenInfoNeedWinSize = attentionWorkerNum * microBatchNum * infoTableLastDimNum * sizeof(int32_t);
    uint64_t tokenDataNeedWinSize = 0;
    if (quantMode != 0) {
        tokenDataNeedWinSize = attentionWorkerNum * microBatchNum * BS * (K + sharedExpertNum) * HS; // 量化模式
    } else {
        tokenDataNeedWinSize =
            attentionWorkerNum * microBatchNum * BS * (K + sharedExpertNum) * HS * NO_QUANT_MODE_WIN_SIZE; // 非量化模式
    }
    uint64_t actualNeedWinSize = tokenInfoNeedWinSize + tokenDataNeedWinSize;
    uint64_t maxWindowSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
    neededSize = actualNeedWinSize / MB_SIZE + 1;
    viableWindowSize = maxWindowSize / MB_SIZE;
    tilingData.attentionToFFNInfo.totalWinSize = maxWindowSize;

    OP_LOGD(ATTN_FFN_INNER_DEBUG,
            "BS is %lu K is %lu HS is %lu microBatchNum is %lu attentionWorkerNum is %lu infoTableLastDimNum is %lu.",
            BS, K, HS, microBatchNum, attentionWorkerNum, infoTableLastDimNum);
    OP_LOGD(ATTN_FFN_INNER_DEBUG,
            "actualNeedWinSize is %lu tokenInfoNeedWinSize is %lu tokenDataNeedWinSize is %lu neededSize is %lu.",
            actualNeedWinSize, tokenInfoNeedWinSize, tokenDataNeedWinSize, neededSize);
    return;
}

static ge::graphStatus SetHcommCfg(AttentionToFFNTilingData &tilingData, const std::string group)
{
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "attentionToFFN group = %s", group.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType1, algConfigAllToAllStr);

    mc2CcTilingConfig.GetTiling(tilingData.mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData.mc2CcTiling1);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckMc2Context(gert::TilingContext *context, const char *nodeName,
                                       const AttentionToFFNTilingConfig &config)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "contextShape"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "context",
                        (std::to_string(contextStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return ge::GRAPH_FAILED);

    auto contextDesc = context->GetInputDesc(config.contextIndex);
    OP_TILING_CHECK(contextDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "contextDesc"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "context",
                                              Ops::Base::ToString(contextDesc->GetDataType()).c_str(), "INT32"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "context",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat()))).c_str(),
            "non-FRACTAL_NZ"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionToFFNTilingFuncBase(gert::TilingContext *context, const AttentionToFFNTilingConfig &config)
{
    AttentionToFFNTilingData *tilingData = context->GetTilingData<AttentionToFFNTilingData>();
    std::string group = "";
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);

    if (config.isMc2Context) {
        OP_TILING_CHECK(CheckMc2Context(context, nodeName, config) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "Tiling check context failed."), return ge::GRAPH_FAILED);
    }

    // Function that get check and set Attrs
    OP_TILING_CHECK(!CheckAndSetAttrs(context, nodeName, *tilingData, group, config),
                    OP_LOGE(nodeName, "Check and set attributes failed!"), return ge::GRAPH_FAILED);

    // Function that check input
    OP_TILING_CHECK(!CheckInputAndSetTilingData(context, nodeName, *tilingData, config),
                    OP_LOGE(nodeName, "Check Inputs and Outputs failed!"), return ge::GRAPH_FAILED);

    // ffn_start_rank_id + ffn_num must not exceed world_size, otherwise the
    // kernel will access epHcclBuffer_[rankId] / hcommHandle_[rankId] beyond
    // the valid [0, worldSize) range populated during context init.
    uint32_t ffnNum = tilingData->attentionToFFNInfo.worldSize - tilingData->attentionToFFNInfo.attentionWorkerNum;
    OP_TILING_CHECK(
        tilingData->attentionToFFNInfo.ffnStartRankId + ffnNum > tilingData->attentionToFFNInfo.worldSize,
        OP_LOGE_WITH_INVALID_ATTR(
            nodeName, "ffn_start_rank_id", std::to_string(tilingData->attentionToFFNInfo.ffnStartRankId).c_str(),
            (std::string("[0, ") + std::to_string(tilingData->attentionToFFNInfo.attentionWorkerNum) +
             "] (ffn_start_rank_id + ffn_num must not exceed world_size)")
                .c_str()),
        return ge::GRAPH_FAILED);

    // Check window Size
    uint64_t neededSize = 0;
    uint64_t viableWindowSize = 0;
    uint32_t quantMode = tilingData->attentionToFFNInfo.quantMode;
    CalWinSize(*tilingData, quantMode, neededSize, viableWindowSize);
    OP_TILING_CHECK(
        neededSize > viableWindowSize,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "neededSize", (std::to_string(neededSize) + " > " + std::to_string(viableWindowSize)).c_str(),
            "The value of neededSize must not be greater than viable window size"),
        return ge::GRAPH_FAILED);

    // Validate ccl_buffer_size when provided (V2 path)
    if (config.attrCclBufferSizeIndex != UINT32_MAX) {
        auto attrsPtr = context->GetAttrs();
        OP_TILING_CHECK(attrsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);
        auto cclBufferSizePtr = attrsPtr->GetAttrPointer<int64_t>(config.attrCclBufferSizeIndex);
        OP_TILING_CHECK(cclBufferSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "ccl_buffer_size"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(*cclBufferSizePtr <= 0,
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "ccl_buffer_size",
                                                  std::to_string(*cclBufferSizePtr).c_str(), "greater than 0"),
                        return ge::GRAPH_FAILED);
        // neededSize is in MB (CalWinSize divides by MB_SIZE and adds 1).
        // ccl_buffer_size is in bytes. Minimum required bytes = neededSize * MB_SIZE.
        uint64_t leastCclBufferSizeBytes = neededSize * MB_SIZE;
        OP_TILING_CHECK(static_cast<uint64_t>(*cclBufferSizePtr) < leastCclBufferSizeBytes,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            nodeName, "ccl_buffer_size", std::to_string(*cclBufferSizePtr).c_str(),
                            (std::string("should >= ") + std::to_string(leastCclBufferSizeBytes) +
                             " bytes (token_info + token_data)")
                                .c_str()),
                        return ge::GRAPH_FAILED);
        OP_LOGD(nodeName, "ccl_buffer_size is %ld bytes, leastCclBufferSize is %lu bytes", *cclBufferSizePtr,
                leastCclBufferSizeBytes);
    }

    // Set WorkSpace
    OP_TILING_CHECK(SetWorkSpace(context, nodeName, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    // Set HcommCfg
    OP_TILING_CHECK(SetHcommCfg(*tilingData, group) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "setHcommCfg failed."),
                    return ge::GRAPH_FAILED);

    // Set TilingKey
    uint32_t syncFlag = tilingData->attentionToFFNInfo.syncFlag;
    bool activeMaskFlag = tilingData->attentionToFFNInfo.isActiveMask;
    uint64_t tilingKey = CalTilingKey(syncFlag, quantMode, activeMaskFlag); // 同步/异步、量化/非量化、mask/非mask
    OP_LOGD(nodeName, "cur case tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    // Set numBlocks
    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData->attentionToFFNInfo.totalUbSize = ubSize;
    tilingData->attentionToFFNInfo.aivNum = aivNum;
    OP_LOGD(nodeName, "numBlocks=%u, aivNum=%u, ubSize=%lu", numBlocks, aivNum, ubSize);

    PrintTilingDataInfo(nodeName, *tilingData);
    OP_LOGD("AttentionToFFN", "tiling process finished successfully!!!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AttentionToFFNTilingFunc(gert::TilingContext *context)
{
    const AttentionToFFNTilingConfig config{};
    return AttentionToFFNTilingFuncBase(context, config);
}

struct AttentionToFFNCompileInfo {};
ge::graphStatus TilingParseForAttentionToFFN(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AttentionToFFN)
    .Tiling(AttentionToFFNTilingFunc)
    .TilingParse<AttentionToFFNCompileInfo>(TilingParseForAttentionToFFN);
} // namespace MC2Tiling
