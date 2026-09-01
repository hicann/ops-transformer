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
 * \file ffn_to_attention_tilling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstdio>
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
#include "ffn_to_attention_tiling_base.h"
#include "../../op_kernel/ffn_to_attention_tiling.h"
#include "../../op_kernel/ffn_to_attention_tilling_key.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace MC2Tiling {
constexpr char FFN_TO_ATTN_WIN_TYPE_ENV[] = "ASCEND_FFN_TO_ATTN_WIN_TYPE";

constexpr uint32_t ONE_DIM = 1;
constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t INDEX_ZERO = 0;
constexpr uint32_t INDEX_ONE = 1;
constexpr uint32_t INDEX_TWO = 2;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint64_t MB_SIZE = 1024 * 1024;
constexpr uint64_t TOKEN_INFO_ALIGN_SIZE = 512;
constexpr uint64_t CCL_BUFFER_ALIGN_SIZE = 2 * MB_SIZE;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr int64_t BS_UPPER_BOUND = 512;
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 8192;
constexpr int64_t SCALE_SIZE = 128;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_WORLD_SIZE = 768L; // 384 * 2
constexpr int64_t MIN_WORLD_SIZE = 2;

constexpr uint32_t TOKEN_DATA_SHAPE_MICRO_BATCH_NUM_INDEX = 0;
constexpr uint32_t TOKEN_DATA_SHAPE_BS_INDEX = 1;
constexpr uint32_t TOKEN_DATA_SHAPE_HS_INDEX = 3;
constexpr uint32_t TOKEN_INFO_TABLE_SHAPE_EXPERT_NUM_INDEX = 2;

constexpr uint32_t TOKEN_INFO_TABLE_DIM_NUM = 3;
constexpr uint32_t TOKEN_DATA_DIM_NUM = 4;

constexpr uint32_t TOKEN_DATA_SIZE = 2; // float/bfloat16, 占2字节
constexpr uint32_t TOKEN_INFO_SIZE = 4; // int32, 占4字节

static void PrintTilingDataInfo(const char *nodeName, FFNToAttentionTilingData &tilingData)
{
    OP_LOGD(nodeName, "H is %u.", tilingData.ffnToAttentionInfo.H);
    OP_LOGD(nodeName, "A is %u.", tilingData.ffnToAttentionInfo.A);
    OP_LOGD(nodeName, "microBatchNum is %u.", tilingData.ffnToAttentionInfo.microBatchNum);
    OP_LOGD(nodeName, "BS is %u.", tilingData.ffnToAttentionInfo.BS);
    OP_LOGD(nodeName, "expertNumPerToken is %u.", tilingData.ffnToAttentionInfo.expertNumPerToken);
    OP_LOGD(nodeName, "HS is %u.", tilingData.ffnToAttentionInfo.HS);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.ffnToAttentionInfo.aivNum);
    OP_LOGD(nodeName, "worldSize is %u.", tilingData.ffnToAttentionInfo.worldSize);
    OP_LOGD(nodeName, "isInputRankTable is %u.", tilingData.ffnToAttentionInfo.isInputRankTable);
    OP_LOGD(nodeName, "windowType is %u.", tilingData.ffnToAttentionInfo.windowType);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.ffnToAttentionInfo.totalUbSize);
    OP_LOGD(nodeName, "totalWinSize is %lu.", tilingData.ffnToAttentionInfo.totalWinSize);
}

static bool CheckAndSetAttrs(gert::TilingContext *context, const char *nodeName, FFNToAttentionTilingData &tilingData,
                             std::string &group, const FFNToAttentionTilingConfig &config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return false);

    auto groupPtr = attrs->GetAttrPointer<char>(config.attrGroupIndex);
    auto worldSizePtr = attrs->GetAttrPointer<int>(config.attrWorldSizeIndex);
    auto tokenInfoTableDimNum = attrs->GetListInt(config.attrTokenInfoTableShapeIndex)->GetSize();
    auto tokenInfoTableShape = attrs->GetListInt(config.attrTokenInfoTableShapeIndex)->GetData();
    auto tokenDataDimNum = attrs->GetListInt(config.attrTokenDataShapeIndex)->GetSize();
    auto tokenDataShape = attrs->GetListInt(config.attrTokenDataShapeIndex)->GetData();

    // 当前仅对必选属性进行校空
    OP_TILING_CHECK(groupPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "group"), return false);
    OP_TILING_CHECK((strnlen(groupPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(groupPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE_WITH_INVALID_ATTR(
                        nodeName, "group",
                        (std::string("length=") + std::to_string(strnlen(groupPtr, MAX_GROUP_NAME_LENGTH))).c_str(),
                        "valid group name length"),
                    return false);
    OP_TILING_CHECK(worldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "worldSize"), return false);
    OP_TILING_CHECK(
        tokenInfoTableDimNum != TOKEN_INFO_TABLE_DIM_NUM,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "tokenInfoTableShape", std::to_string(tokenInfoTableDimNum).c_str(),
                                  std::to_string(TOKEN_INFO_TABLE_DIM_NUM).c_str()),
        return false);
    OP_TILING_CHECK(tokenDataDimNum != TOKEN_DATA_DIM_NUM,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "tokenDataShape", std::to_string(tokenDataDimNum).c_str(),
                                              std::to_string(TOKEN_DATA_DIM_NUM).c_str()),
                    return false);

    OP_TILING_CHECK(
        tokenInfoTableShape[INDEX_ZERO] != tokenDataShape[INDEX_ZERO],
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "tokenDataShape",
                                  (std::string("dim0=") + std::to_string(tokenDataShape[INDEX_ZERO])).c_str(),
                                  (std::string("dim0=") + std::to_string(tokenInfoTableShape[INDEX_ZERO])).c_str()),
        return false);
    OP_TILING_CHECK(
        tokenInfoTableShape[INDEX_ONE] != tokenDataShape[INDEX_ONE],
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "tokenDataShape",
                                  (std::string("dim1=") + std::to_string(tokenDataShape[INDEX_ONE])).c_str(),
                                  (std::string("dim1=") + std::to_string(tokenInfoTableShape[INDEX_ONE])).c_str()),
        return false);
    OP_TILING_CHECK(
        tokenInfoTableShape[INDEX_TWO] != tokenDataShape[INDEX_TWO],
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "tokenDataShape",
                                  (std::string("dim2=") + std::to_string(tokenDataShape[INDEX_TWO])).c_str(),
                                  (std::string("dim2=") + std::to_string(tokenInfoTableShape[INDEX_TWO])).c_str()),
        return false);
    // 判断是否满足其他限制
    int64_t worldSize = *worldSizePtr;
    OP_TILING_CHECK(
        (worldSize < MIN_WORLD_SIZE) || (worldSize > MAX_WORLD_SIZE),
        OP_LOGE_WITH_INVALID_ATTR(
            nodeName, "worldSize", std::to_string(worldSize).c_str(),
            (std::string("[") + std::to_string(MIN_WORLD_SIZE) + ", " + std::to_string(MAX_WORLD_SIZE) + "]").c_str()),
        return false);
    OP_TILING_CHECK(tokenInfoTableShape[INDEX_ZERO] != 1,
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "tokenInfoTableShape",
                        (std::string("dim0=") + std::to_string(tokenInfoTableShape[INDEX_ZERO])).c_str(), "dim0=1"),
                    return false);
    OP_TILING_CHECK((tokenDataShape[TOKEN_DATA_SHAPE_BS_INDEX] > BS_UPPER_BOUND) ||
                        (tokenDataShape[TOKEN_DATA_SHAPE_BS_INDEX] <= 0),
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "tokenDataShape",
                        (std::string("BS=") + std::to_string(tokenDataShape[TOKEN_DATA_SHAPE_BS_INDEX])).c_str(),
                        (std::string("[1, ") + std::to_string(BS_UPPER_BOUND) + "]").c_str()),
                    return false);
    OP_TILING_CHECK(
        (tokenDataShape[TOKEN_DATA_SHAPE_HS_INDEX] < H_MIN) ||
            (tokenDataShape[TOKEN_DATA_SHAPE_HS_INDEX] > H_MAX + SCALE_SIZE),
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "tokenDataShape",
            (std::string("HS=") + std::to_string(tokenDataShape[TOKEN_DATA_SHAPE_HS_INDEX])).c_str(),
            (std::string("[") + std::to_string(H_MIN) + ", " + std::to_string(H_MAX + SCALE_SIZE) + "]").c_str()),
        return false);

    tilingData.ffnToAttentionInfo.worldSize = *worldSizePtr;
    tilingData.ffnToAttentionInfo.microBatchNum = tokenDataShape[TOKEN_DATA_SHAPE_MICRO_BATCH_NUM_INDEX];
    tilingData.ffnToAttentionInfo.BS = tokenDataShape[TOKEN_DATA_SHAPE_BS_INDEX];
    tilingData.ffnToAttentionInfo.HS = tokenDataShape[TOKEN_DATA_SHAPE_HS_INDEX];
    tilingData.ffnToAttentionInfo.expertNumPerToken = tokenInfoTableShape[TOKEN_INFO_TABLE_SHAPE_EXPERT_NUM_INDEX];

    OP_LOGD(nodeName, "group = %s", groupPtr);
    group = string(groupPtr);

    return true;
}

static bool CheckInputDim0Dim1(gert::TilingContext *context, const char *nodeName, FFNToAttentionTilingData &tilingData,
                               const FFNToAttentionTilingConfig &config)
{
    const gert::StorageShape *xShape = context->GetInputShape(config.xIndex);
    const gert::StorageShape *sessionIdsShape = context->GetInputShape(config.sessionIdsIndex);
    const gert::StorageShape *microBatchIdsShape = context->GetInputShape(config.microBatchIdsIndex);
    const gert::StorageShape *tokenIdsShape = context->GetInputShape(config.tokenIdsIndex);
    const gert::StorageShape *expertOffsetsShape = context->GetInputShape(config.expertOffsetsIndex);
    const gert::StorageShape *actualTokenNumShape = context->GetInputShape(config.actualTokenNumIndex);
    const gert::StorageShape *attnRankTableShape = context->GetOptionalInputShape(config.attnRankTableIndex);
    // 校验输入x的维度Y和H
    const uint64_t xDim0 = xShape->GetStorageShape().GetDim(INDEX_ZERO);
    const uint64_t xDim1 = xShape->GetStorageShape().GetDim(INDEX_ONE);
    const int64_t MAX_INT64 = std::numeric_limits<int64_t>::max();
    OP_TILING_CHECK((xDim0 > static_cast<uint64_t>(MAX_INT64)),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "x", (std::string("dim0=") + std::to_string(xDim0)).c_str(),
                                              (std::string("dim0 <= ") + std::to_string(MAX_INT64)).c_str()),
                    return false);
    OP_TILING_CHECK((xDim1 < H_MIN) || (xDim1 > H_MAX),
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "x", (std::string("dim1=") + std::to_string(xDim1)).c_str(),
                        (std::string("[") + std::to_string(H_MIN) + ", " + std::to_string(H_MAX) + "]").c_str()),
                    return false);

    // 校验输入sessionIds的维度Y
    const int64_t sessionIdsDim0 = sessionIdsShape->GetStorageShape().GetDim(INDEX_ZERO);
    OP_TILING_CHECK(xDim0 != static_cast<uint64_t>(sessionIdsDim0),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "sessionIds",
                                              (std::string("dim0=") + std::to_string(sessionIdsDim0)).c_str(),
                                              (std::string("dim0=") + std::to_string(xDim0)).c_str()),
                    return false);
    // 校验输入microBatchIds的维度Y
    const int64_t microBatchIdsDim0 = microBatchIdsShape->GetStorageShape().GetDim(INDEX_ZERO);
    OP_TILING_CHECK(xDim0 != static_cast<uint64_t>(microBatchIdsDim0),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "microBatchIds",
                                              (std::string("dim0=") + std::to_string(microBatchIdsDim0)).c_str(),
                                              (std::string("dim0=") + std::to_string(xDim0)).c_str()),
                    return false);
    // 校验输入tokenIds的维度Y
    const int64_t tokenIdsDim0 = tokenIdsShape->GetStorageShape().GetDim(INDEX_ZERO);
    OP_TILING_CHECK(
        xDim0 != static_cast<uint64_t>(tokenIdsDim0),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "tokenIds", (std::string("dim0=") + std::to_string(tokenIdsDim0)).c_str(),
                                  (std::string("dim0=") + std::to_string(xDim0)).c_str()),
        return false);
    // 校验输入expertOffsets的维度Y
    const int64_t expertOffsetsDim0 = expertOffsetsShape->GetStorageShape().GetDim(INDEX_ZERO);
    OP_TILING_CHECK(xDim0 != static_cast<uint64_t>(expertOffsetsDim0),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "expertOffsets",
                                              (std::string("dim0=") + std::to_string(expertOffsetsDim0)).c_str(),
                                              (std::string("dim0=") + std::to_string(xDim0)).c_str()),
                    return false);
    // 校验输入actualTokenNum的维度1
    const uint64_t actualTokenNumDim0 = actualTokenNumShape->GetStorageShape().GetDim(INDEX_ZERO);
    OP_TILING_CHECK(
        actualTokenNumDim0 != 1,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "actualTokenNum",
                                  (std::string("dim0=") + std::to_string(actualTokenNumDim0)).c_str(), "dim0=1"),
        return false);
    tilingData.ffnToAttentionInfo.H = xDim1;
    return true;
}

static bool CheckInputDim(gert::TilingContext *context, const char *nodeName, FFNToAttentionTilingData &tilingData,
                          const FFNToAttentionTilingConfig &config)
{
    auto attrs = context->GetAttrs();
    auto worldSizePtr = attrs->GetAttrPointer<int>(config.attrWorldSizeIndex);
    int64_t worldSize = *worldSizePtr;
    const gert::StorageShape *xShape = context->GetInputShape(config.xIndex);
    const gert::StorageShape *sessionIdsShape = context->GetInputShape(config.sessionIdsIndex);
    const gert::StorageShape *microBatchIdsShape = context->GetInputShape(config.microBatchIdsIndex);
    const gert::StorageShape *tokenIdsShape = context->GetInputShape(config.tokenIdsIndex);
    const gert::StorageShape *expertOffsetsShape = context->GetInputShape(config.expertOffsetsIndex);
    const gert::StorageShape *actualTokenNumShape = context->GetInputShape(config.actualTokenNumIndex);
    const gert::StorageShape *attnRankTableShape = context->GetOptionalInputShape(config.attnRankTableIndex);
    bool isInputRankTable = (attnRankTableShape != nullptr);

    OP_TILING_CHECK(xShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return false);
    OP_TILING_CHECK(sessionIdsShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sessionIds"), return false);
    OP_TILING_CHECK(microBatchIdsShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "microBatchIds"), return false);
    OP_TILING_CHECK(tokenIdsShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tokenIds"), return false);
    OP_TILING_CHECK(expertOffsetsShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertOffsets"), return false);
    OP_TILING_CHECK(actualTokenNumShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "actualTokenNum"),
                    return false);
    OP_TILING_CHECK(xShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "x", (std::to_string(xShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
                    return false);
    OP_TILING_CHECK(sessionIdsShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "sessionIds",
                        (std::to_string(sessionIdsShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return false);
    OP_TILING_CHECK(microBatchIdsShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "microBatchIds",
                        (std::to_string(microBatchIdsShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return false);
    OP_TILING_CHECK(
        tokenIdsShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "tokenIds", (std::to_string(tokenIdsShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
        return false);
    OP_TILING_CHECK(expertOffsetsShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "expertOffsets",
                        (std::to_string(expertOffsetsShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return false);
    OP_TILING_CHECK(actualTokenNumShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "actualTokenNum",
                        (std::to_string(actualTokenNumShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return false);

    OP_TILING_CHECK(!CheckInputDim0Dim1(context, nodeName, tilingData, config),
                    OP_LOGE(nodeName, "Check Inputsdim0ordim1 failed!"), return false);

    if (isInputRankTable) {
        OP_TILING_CHECK(attnRankTableShape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(
                            nodeName, "attnRankTable",
                            (std::to_string(attnRankTableShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                        return false);
        const int32_t attnRankTableDim0 = attnRankTableShape->GetStorageShape().GetDim(INDEX_ZERO);
        tilingData.ffnToAttentionInfo.A = attnRankTableDim0;

        OP_TILING_CHECK(attnRankTableDim0 > worldSize - 1,
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "attnRankTable",
                                                  (std::string("dim0=") + std::to_string(attnRankTableDim0)).c_str(),
                                                  (std::string("dim0 <= ") + std::to_string(worldSize - 1)).c_str()),
                        return false);
    }
    tilingData.ffnToAttentionInfo.isInputRankTable = isInputRankTable;
    return true;
}

static bool CheckInputDataType(gert::TilingContext *context, const char *nodeName,
                               const FFNToAttentionTilingConfig &config)
{
    const gert::StorageShape *attnRankTableShape = context->GetOptionalInputShape(config.attnRankTableIndex);
    bool isInputRankTable = (attnRankTableShape != nullptr);

    auto xDesc = context->GetInputDesc(config.xIndex);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    OP_TILING_CHECK(
        (xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(), "BF16, FLOAT16"),
        return false);

    auto sessionIdDesc = context->GetInputDesc(config.sessionIdsIndex);
    OP_TILING_CHECK(sessionIdDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sessionIdDesc"), return false);
    OP_TILING_CHECK(sessionIdDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "sessionId",
                                              Ops::Base::ToString(sessionIdDesc->GetDataType()).c_str(), "INT32"),
                    return false);

    auto microBatchIdDesc = context->GetInputDesc(config.microBatchIdsIndex);
    OP_TILING_CHECK(microBatchIdDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "microBatchIdDesc"),
                    return false);
    OP_TILING_CHECK(microBatchIdDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "microBatchId",
                                              Ops::Base::ToString(microBatchIdDesc->GetDataType()).c_str(), "INT32"),
                    return false);

    auto tokenIdDesc = context->GetInputDesc(config.tokenIdsIndex);
    OP_TILING_CHECK(tokenIdDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tokenIdDesc"), return false);
    OP_TILING_CHECK(tokenIdDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "tokenId",
                                              Ops::Base::ToString(tokenIdDesc->GetDataType()).c_str(), "INT32"),
                    return false);

    auto expertOffsetDesc = context->GetInputDesc(config.expertOffsetsIndex);
    OP_TILING_CHECK(expertOffsetDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertOffsetDesc"),
                    return false);
    OP_TILING_CHECK(expertOffsetDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expertOffset",
                                              Ops::Base::ToString(expertOffsetDesc->GetDataType()).c_str(), "INT32"),
                    return false);

    auto atcualTokenNumDesc = context->GetInputDesc(config.actualTokenNumIndex);
    OP_TILING_CHECK(atcualTokenNumDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "atcualTokenNumDesc"),
                    return false);
    OP_TILING_CHECK(atcualTokenNumDesc->GetDataType() != ge::DT_INT64,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "atcualTokenNum",
                                              Ops::Base::ToString(atcualTokenNumDesc->GetDataType()).c_str(), "INT64"),
                    return false);

    if (isInputRankTable) {
        auto attnRankTableDesc = context->GetInputDesc(config.attnRankTableIndex);
        OP_TILING_CHECK(attnRankTableDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attnRankTableDesc"),
                        return false);
        OP_TILING_CHECK(
            attnRankTableDesc->GetDataType() != ge::DT_INT32,
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "attnRankTable",
                                      Ops::Base::ToString(attnRankTableDesc->GetDataType()).c_str(), "INT32"),
            return false);
    }

    return true;
}

static bool CheckInputFormat(gert::TilingContext *context, const char *nodeName,
                             const FFNToAttentionTilingConfig &config)
{
    const gert::StorageShape *attnRankTableShape = context->GetOptionalInputShape(config.attnRankTableIndex);
    bool isInputRankTable = (attnRankTableShape != nullptr);

    auto xDesc = context->GetInputDesc(config.xIndex);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "x",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat()))).c_str(),
            "non-FRACTAL_NZ"),
        return false);

    auto sessionIdDesc = context->GetInputDesc(config.sessionIdsIndex);
    OP_TILING_CHECK(sessionIdDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sessionIdDesc"), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(sessionIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "sessionId",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(sessionIdDesc->GetStorageFormat())))
                .c_str(),
            "non-FRACTAL_NZ"),
        return false);

    auto microBatchIdDesc = context->GetInputDesc(config.microBatchIdsIndex);
    OP_TILING_CHECK(microBatchIdDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "microBatchIdDesc"),
                    return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(microBatchIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "microBatchId",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(microBatchIdDesc->GetStorageFormat())))
                .c_str(),
            "non-FRACTAL_NZ"),
        return false);

    auto tokenIdDesc = context->GetInputDesc(config.tokenIdsIndex);
    OP_TILING_CHECK(tokenIdDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tokenIdDesc"), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(tokenIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "tokenId",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(tokenIdDesc->GetStorageFormat()))).c_str(),
            "non-FRACTAL_NZ"),
        return false);

    auto expertOffsetDesc = context->GetInputDesc(config.expertOffsetsIndex);
    OP_TILING_CHECK(expertOffsetDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertOffsetDesc"),
                    return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertOffsetDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "expertOffset",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(expertOffsetDesc->GetStorageFormat())))
                .c_str(),
            "non-FRACTAL_NZ"),
        return false);

    auto atcualTokenNumDesc = context->GetInputDesc(config.actualTokenNumIndex);
    OP_TILING_CHECK(atcualTokenNumDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "atcualTokenNumDesc"),
                    return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(atcualTokenNumDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "atcualTokenNum",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(atcualTokenNumDesc->GetStorageFormat())))
                .c_str(),
            "non-FRACTAL_NZ"),
        return false);

    if (isInputRankTable) {
        auto attnRankTableDesc = context->GetInputDesc(config.attnRankTableIndex);
        OP_TILING_CHECK(attnRankTableDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attnRankTableDesc"),
                        return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(attnRankTableDesc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "attnRankTable",
                                                   Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(
                                                                           attnRankTableDesc->GetStorageFormat())))
                                                       .c_str(),
                                                   "non-FRACTAL_NZ"),
                        return false);
    }
    return true;
}

static bool CheckInputAndSetTilingData(gert::TilingContext *context, const char *nodeName,
                                       FFNToAttentionTilingData &tilingData, const FFNToAttentionTilingConfig &config)
{
    // 校验输入数据dim、format、dataType
    OP_TILING_CHECK(!CheckInputDim(context, nodeName, tilingData, config),
                    OP_LOGE(nodeName, "Check Inputs dim failed!"), return false);

    OP_TILING_CHECK(!CheckInputDataType(context, nodeName, config),
                    OP_LOGE(nodeName, "Check Inputs dataType is invalid."), return false);
    OP_TILING_CHECK(!CheckInputFormat(context, nodeName, config), OP_LOGE(nodeName, "Check Inputs  format is invalid."),
                    return false);
    return true;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workSpaces"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    workSpaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

static uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

static void CalWinSize(FFNToAttentionTilingData &tilingData, uint64_t &neededSize, uint64_t &viableWindowSize)
{
    uint64_t microBatchNum = static_cast<uint64_t>(tilingData.ffnToAttentionInfo.microBatchNum);
    uint64_t BS = static_cast<uint64_t>(tilingData.ffnToAttentionInfo.BS);
    uint64_t expertNumPerToken = static_cast<uint64_t>(tilingData.ffnToAttentionInfo.expertNumPerToken);
    uint64_t HS = static_cast<uint64_t>(tilingData.ffnToAttentionInfo.HS);

    uint64_t tokenDataNeedWinSize = microBatchNum * BS * expertNumPerToken * HS * TOKEN_DATA_SIZE;
    uint64_t tokenInfoTableNeedWinSize = microBatchNum * BS * expertNumPerToken * TOKEN_INFO_SIZE;
    uint64_t maxWindowSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
    uint64_t tokenInfoTableAlignedSize = AlignUp(tokenInfoTableNeedWinSize, TOKEN_INFO_ALIGN_SIZE);
    neededSize = tokenInfoTableAlignedSize + tokenDataNeedWinSize;
    viableWindowSize = maxWindowSize;
    tilingData.ffnToAttentionInfo.totalWinSize = maxWindowSize;

    return;
}

static ge::graphStatus SetHcommCfg(gert::TilingContext *context, FFNToAttentionTilingData &tilingData,
                                   const std::string group)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "ffnToAttention group = %s", group.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tilingData.mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData.mc2CcTiling1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckMc2Context(gert::TilingContext *context, const char *nodeName,
                                       const FFNToAttentionTilingConfig &config)
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

ge::graphStatus FFNToAttentionTilingFuncBase(gert::TilingContext *context, const FFNToAttentionTilingConfig &config)
{
    FFNToAttentionTilingData *tilingData = context->GetTilingData<FFNToAttentionTilingData>();
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

    // Function that check input dim 、format、datatype
    OP_TILING_CHECK(!CheckInputAndSetTilingData(context, nodeName, *tilingData, config),
                    OP_LOGE(nodeName, "Check Inputs and Outputs failed!"), return ge::GRAPH_FAILED);

    // Check window Size
    uint64_t neededSize = 0;
    uint64_t viableWindowSize = 0;
    CalWinSize(*tilingData, neededSize, viableWindowSize);
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
        uint64_t leastCclBufferSizeBytes = AlignUp(neededSize, CCL_BUFFER_ALIGN_SIZE);
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
        uint64_t leastCclBufferSizeBytes = AlignUp(neededSize, CCL_BUFFER_ALIGN_SIZE);
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
    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    // Set HcommCfg
    OP_TILING_CHECK(SetHcommCfg(context, *tilingData, group) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "setHcommCfg failed."), return ge::GRAPH_FAILED);

    // Set TilingKey
    bool rankTableMode = tilingData->ffnToAttentionInfo.isInputRankTable;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(rankTableMode, TILINGKEY_TPL_A3);
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
    tilingData->ffnToAttentionInfo.totalUbSize = ubSize;
    tilingData->ffnToAttentionInfo.aivNum = aivNum;
    OP_LOGD(nodeName, "numBlocks=%u, aivNum=%u, ubSize=%lu", numBlocks, aivNum, ubSize);

    PrintTilingDataInfo(nodeName, *tilingData);
    OP_LOGD("FFNToAttention", "tiling process finished successfully!!!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFNToAttentionTilingFunc(gert::TilingContext *context)
{
    const FFNToAttentionTilingConfig config{};
    return FFNToAttentionTilingFuncBase(context, config);
}

struct FFNToAttentionCompileInfo {};
ge::graphStatus TilingParseForFFNToAttention(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FFNToAttention)
    .Tiling(FFNToAttentionTilingFunc)
    .TilingParse<FFNToAttentionCompileInfo>(TilingParseForFFNToAttention);
} // namespace MC2Tiling
