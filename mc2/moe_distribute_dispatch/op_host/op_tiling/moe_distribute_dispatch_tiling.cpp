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
 * \file moe_distribute_dispatch_tiling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "moe_distribute_dispatch_tiling_base.h"

using namespace Ops::Transformer::OpTiling;
using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;
namespace {
constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 5;
constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 6;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 8;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 10;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 11;
constexpr uint32_t ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX = 12;

constexpr uint32_t EXPERT_SCALES_INDEX = 4U;
constexpr size_t INPUT_DIM_TWO = 2UL;
constexpr uint32_t DYN_SCALE_DIMS = 1;
constexpr uint32_t EXPAND_IDX_DIMS = 1;
constexpr uint64_t INIT_TILINGKEY = 1000;
constexpr uint32_t ARR_LENGTH = 128;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;

constexpr uint32_t UNQUANT_MODE = 0;
constexpr uint32_t STATIC_QUANT_MODE = 1;
constexpr uint32_t DYNAMIC_QUANT_MODE = 2;
const size_t MAX_GROUP_NAME_LENGTH = 128UL;
const int64_t MAX_EP_WORLD_SIZE = 288;
const int64_t BS_UPPER_BOUND = 512;

constexpr uint32_t NO_SCALES = 0;
constexpr uint32_t STATIC_SCALES = 1;
constexpr uint32_t DYNAMIC_SCALES = 2;
constexpr uint32_t NUM_10 = 10;
constexpr uint32_t NUM_100 = 100;
constexpr uint32_t VERSION_2 = 2;
constexpr uint32_t HCOMMCNT_2 = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 512;
constexpr int64_t K_MAX = 8;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024; // Bytes
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;

constexpr uint64_t TILING_KEY_BASE_A2 = 2000000000;
constexpr uint64_t TILING_KEY_LAYERED_COMM_A2 = 100000000;
} // namespace

namespace optiling {

static void PrintTilingDataInfo(const char *nodeName, MoeDistributeDispatchTilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeDispatchInfo.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeDispatchInfo.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeDispatchInfo.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeDispatchInfo.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeDistributeDispatchInfo.expertShardType);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeDispatchInfo.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeDispatchInfo.moeExpertNum);
    OP_LOGD(nodeName, "quantMode is %u.", tilingData.moeDistributeDispatchInfo.quantMode);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeDispatchInfo.globalBs);
    OP_LOGD(nodeName, "isQuant is %d.", tilingData.moeDistributeDispatchInfo.isQuant);
    OP_LOGD(nodeName, "bs is %u.", tilingData.moeDistributeDispatchInfo.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.moeDistributeDispatchInfo.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.moeDistributeDispatchInfo.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.moeDistributeDispatchInfo.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.moeDistributeDispatchInfo.totalUbSize);
    OP_LOGD(nodeName, "totalWinSizeEP is %lu.", tilingData.moeDistributeDispatchInfo.totalWinSizeEp);
}

static ge::graphStatus CheckAttrPointersNotNull(gert::TilingContext *context, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX));

    OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "groupEpPtr"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epWorldSizePtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpWorldSizePtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epRankIdPtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpRankIdPtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertShardPtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sharedExpertRankNumPtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "moeExpertNumPtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "quantModePtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sharedExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertTokenNumsType"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCommAttrValuesValid(gert::TilingContext *context, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);

    OP_TILING_CHECK((*epWorldSizePtr <= 0) || (*epWorldSizePtr > MAX_EP_WORLD_SIZE),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "epWorldSize", std::to_string(*epWorldSizePtr).c_str(),
                                              (std::string("(0, ") + std::to_string(MAX_EP_WORLD_SIZE) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= *epWorldSizePtr),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "epRankId", std::to_string(*epRankIdPtr).c_str(),
                                              (std::string("[0, ") + std::to_string(*epWorldSizePtr) + ")").c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tp_world_size"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *tpWorldSizePtr >= 2,
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "tp_world_size", std::to_string(*tpWorldSizePtr).c_str(), "less than 2"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus CheckAttrValuesValid(gert::TilingContext *context, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX));

    OP_TILING_CHECK(
        CheckCommAttrValuesValid(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "ep/tp attrs", "invalid", "EP/TP attr values must be valid"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *expertShardPtr != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "expertShardType", std::to_string(*expertShardPtr).c_str(), "0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*sharedExpertRankNumPtr < 0) || (*sharedExpertRankNumPtr >= *epWorldSizePtr),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertRankNum", std::to_string(*sharedExpertRankNumPtr).c_str(),
                                  (std::string("[0, ") + std::to_string(*epWorldSizePtr) + ")").c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*moeExpertNumPtr <= 0) || (*moeExpertNumPtr > MOE_EXPERT_MAX_NUM),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "moeExpertNum", std::to_string(*moeExpertNumPtr).c_str(),
                                              (std::string("(0, ") + std::to_string(MOE_EXPERT_MAX_NUM) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*quantModePtr < static_cast<int64_t>(NO_SCALES)) || (*quantModePtr > static_cast<int64_t>(DYNAMIC_SCALES)),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "quantMode", std::to_string(*quantModePtr).c_str(),
                                  (std::string("[0, ") + std::to_string(DYNAMIC_SCALES) + "]").c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *sharedExpertNumPtr != 1,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertNum", std::to_string(*sharedExpertNumPtr).c_str(), "1"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*expertTokenNumsTypePtr != 0) && (*expertTokenNumsTypePtr != 1),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "expertTokenNumsType",
                                              std::to_string(*expertTokenNumsTypePtr).c_str(), "0 or 1"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
                                               MoeDistributeDispatchTilingData &tilingData, std::string &groupEp)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX));

    // 校验指针非空
    OP_TILING_CHECK(CheckAttrPointersNotNull(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "attrs", "pointer null or empty",
                                                          "All attr pointers must be non-null and valid"),
                    return ge::GRAPH_FAILED);

    // 判断是否满足uint32_t及其他限制
    OP_TILING_CHECK(
        CheckAttrValuesValid(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "attrs", "invalid value", "All attr values must be valid"),
        return ge::GRAPH_FAILED);

    groupEp = std::string(groupEpPtr);
    tilingData.moeDistributeDispatchInfo.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
    tilingData.moeDistributeDispatchInfo.tpWorldSize = 1U;
    tilingData.moeDistributeDispatchInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDistributeDispatchInfo.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.moeDistributeDispatchInfo.expertShardType = static_cast<uint32_t>(*expertShardPtr);
    tilingData.moeDistributeDispatchInfo.sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    tilingData.moeDistributeDispatchInfo.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    tilingData.moeDistributeDispatchInfo.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.moeDistributeDispatchInfo.expertTokenNumsType = static_cast<uint32_t>(*expertTokenNumsTypePtr);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertAttrs(gert::TilingContext *context, const char *nodeName,
                                        MoeDistributeDispatchTilingData &tilingData, uint32_t &localMoeExpertNum)
{
    uint32_t epWorldSize = tilingData.moeDistributeDispatchInfo.epWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeDispatchInfo.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchInfo.sharedExpertRankNum;

    // 校验ep能否均分共享专家
    if ((sharedExpertRankNum != 0) && (epWorldSize % sharedExpertRankNum != 0)) {
        OP_LOGE_WITH_INVALID_ATTR(
            nodeName, "epWorldSize", std::to_string(epWorldSize).c_str(),
            (std::string("divisible by sharedExpertRankNum (") + std::to_string(sharedExpertRankNum) + ")").c_str());
        return ge::GRAPH_FAILED;
    }

    // 校验moe专家数量能否均分给多机
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "moe_expert_num", std::to_string(moeExpertNum).c_str(),
                                              (std::string("divisible by ep_world_size - shared_expert_rank_num (") +
                                               std::to_string(epWorldSize - sharedExpertRankNum) + ")")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localMoeExpertNum <= 0,
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "moe_expert_num", std::to_string(moeExpertNum).c_str(),
                                              "produce a positive local expert count"),
                    return ge::GRAPH_FAILED);

    // 校验k > moeExpertNum
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    uint32_t K = static_cast<uint32_t>(expertIdsDim1);
    if (K > moeExpertNum) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "K", std::to_string(K).c_str(),
                                  (std::string("<= moeExpertNum (") + std::to_string(moeExpertNum) + ")").c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckBatchAttrs(gert::TilingContext *context, const char *nodeName,
                                       MoeDistributeDispatchTilingData &tilingData)
{
    uint32_t epWorldSize = tilingData.moeDistributeDispatchInfo.epWorldSize;

    // 校验输入x的dim 0并设bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK((xDim0 > BS_UPPER_BOUND) || (xDim0 <= 0),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "BS", std::to_string(xDim0).c_str(),
                                              (std::string("[1, ") + std::to_string(BS_UPPER_BOUND) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchInfo.bs = static_cast<uint32_t>(xDim0);

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "globalBsPtr"),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "MoeDistributeDispatch *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n", *globalBsPtr, xDim0,
            epWorldSize);
    OP_TILING_CHECK((*globalBsPtr != 0) && ((*globalBsPtr < xDim0 * static_cast<int64_t>(epWorldSize)) ||
                                            ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "globalBS", std::to_string(*globalBsPtr).c_str(),
                                              "0 or maxBs * epWorldSize"),
                    return ge::GRAPH_FAILED);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeDispatchInfo.globalBs = static_cast<uint32_t>(xDim0) * epWorldSize;
    } else {
        tilingData.moeDistributeDispatchInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchTilingBase::CheckAttrs(gert::TilingContext *context, const char *nodeName,
                                                            MoeDistributeDispatchTilingData &tilingData,
                                                            uint32_t &localMoeExpertNum)
{
    OP_TILING_CHECK(
        CheckExpertAttrs(context, nodeName, tilingData, localMoeExpertNum) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "expert", "invalid", "Expert params validation failed"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckEpWorldSizeAttrs(nodeName, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "epWorldSize", "invalid",
                                                          "EpWorldSize params validation failed"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        CheckBatchAttrs(context, nodeName, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "batch", "invalid", "Batch params validation failed"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputTensorShape(gert::TilingContext *context, const char *nodeName,
                                             MoeDistributeDispatchTilingData &tilingData, const bool isScales)
{
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchInfo.sharedExpertRankNum;

    // 校验输入x的维度1并设h, bs已校验过
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK((xDim1 != 7168), OP_LOGE_FOR_INVALID_VALUE(nodeName, "H", std::to_string(xDim1).c_str(), "7168"),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchInfo.h = static_cast<uint32_t>(xDim1);

    // 校验expert_id的维度并设k
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchInfo.moeExpertNum);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const int64_t expertIdsDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    if (xDim0 != expertIdsDim0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "x/expertIds",
            (std::string("x dim0=") + std::to_string(xDim0) + ", expertIds dim0=" + std::to_string(expertIdsDim0))
                .c_str(),
            "Dim0 of x must be equal to dim0 of expertIds");
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "K", std::to_string(expertIdsDim1).c_str(),
                                              (std::string("(0, ") + std::to_string(K_MAX) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchInfo.k = static_cast<uint32_t>(expertIdsDim1);

    const gert::StorageShape *expertScalesStorageShape = context->GetOptionalInputShape(EXPERT_SCALES_INDEX);
    if (expertScalesStorageShape != nullptr) {
        const auto &expertScalesShape = expertScalesStorageShape->GetStorageShape();
        OP_TILING_CHECK(
            expertScalesShape.GetDimNum() != INPUT_DIM_TWO,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                nodeName, "expert_scales", Ops::Base::ToString(expertScalesShape).c_str(), "expert_scales must be 2D"),
            return ge::GRAPH_FAILED);
        const int64_t expertScalesDim0 = expertScalesShape.GetDim(0);
        const int64_t expertScalesDim1 = expertScalesShape.GetDim(1);
        if ((expertScalesDim0 != expertIdsDim0) || (expertScalesDim1 != expertIdsDim1)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                nodeName, "expert_scales/expert_ids",
                (std::string("expert_scales shape=[") + std::to_string(expertScalesDim0) + ", " +
                 std::to_string(expertScalesDim1) + "], expert_ids shape=[" + std::to_string(expertIdsDim0) + ", " +
                 std::to_string(expertIdsDim1) + "]")
                    .c_str(),
                "expert_scales shape must be equal to expert_ids shape");
            return ge::GRAPH_FAILED;
        }
    }

    // 校验scales的维度
    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
        const int64_t scalesDim0 = scalesStorageShape->GetStorageShape().GetDim(0);
        const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
        if (sharedExpertRankNum == 0U) {
            if (scalesDim0 != moeExpertNum) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    nodeName, "scales", (std::string("dim0=") + std::to_string(scalesDim0)).c_str(),
                    "scales dim0 should be equal to moeExpertNum when sharedExpertRankNum == 0");
                return ge::GRAPH_FAILED;
            }
        } else {
            if (scalesDim0 != (moeExpertNum + 1)) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    nodeName, "scales", (std::string("dim0=") + std::to_string(scalesDim0)).c_str(),
                    "scales dim0 should be equal to moeExpertNum + 1 when sharedExpertRankNum != 0");
                return ge::GRAPH_FAILED;
            }
        }
        if (xDim1 != scalesDim1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                nodeName, "x/scales",
                (std::string("x dim1=") + std::to_string(xDim1) + ", scales dim1=" + std::to_string(scalesDim1))
                    .c_str(),
                "Dim1 of x must be equal to dim1 of scales");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertScalesDtypeAndFormat(gert::TilingContext *context, const char *nodeName)
{
    auto expertScalesDesc = context->GetOptionalInputDesc(EXPERT_SCALES_INDEX);
    if (expertScalesDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (expertScalesDesc->GetDataType() != ge::DT_FLOAT) {
        std::string dtypeStr = Ops::Base::ToString(expertScalesDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expert_scales", dtypeStr.c_str(), "FLOAT");
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertScalesDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_WITH_INVALID_INPUT_FORMAT(
            nodeName, "expert_scales",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(expertScalesDesc->GetStorageFormat())))
                .c_str(),
            "not FRACTAL_NZ"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCommTensorShape(gert::TilingContext *context, const char *nodeName,
                                            MoeDistributeDispatchTilingData &tilingData, const bool isSharedExpert,
                                            const int64_t localMoeExpertNum)
{
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchInfo.epWorldSize);
    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    const int64_t epRecvCountDim0 = epRecvCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epRecvCount = (isSharedExpert) ? epWorldSize : epWorldSize * localMoeExpertNum;
    OP_TILING_CHECK(
        epRecvCountDim0 < epRecvCount,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "ep_recv_count", Ops::Base::ToString(epRecvCountStorageShape->GetStorageShape()).c_str(),
            (std::string("dim 0 must be at least ") + std::to_string(epRecvCount)).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputTensorShape(gert::TilingContext *context, const char *nodeName,
                                              MoeDistributeDispatchTilingData &tilingData, const uint32_t quantMode,
                                              const bool isSharedExpert, const int64_t localMoeExpertNum,
                                              const uint32_t A)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);

    // 校验expandX的维度
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    const int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < static_cast<int64_t>(A),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        nodeName, "expand_x", Ops::Base::ToString(expandXStorageShape->GetStorageShape()).c_str(),
                        (std::string("dim 0 must be at least ") + std::to_string(A)).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        xDim1 != expandXDim1,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "x and expand_x",
                                               (Ops::Base::ToString(xStorageShape->GetStorageShape()) + " and " +
                                                Ops::Base::ToString(expandXStorageShape->GetStorageShape()))
                                                   .c_str(),
                                               "their dim 1 values must be equal"),
        return ge::GRAPH_FAILED);
    // 校验dynamicScales的维度
    if (quantMode != NO_SCALES) {
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        const int64_t dynamicScalesDim0 = dynamicScalesStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(
            dynamicScalesDim0 < static_cast<int64_t>(A),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                nodeName, "dynamic_scales", Ops::Base::ToString(dynamicScalesStorageShape->GetStorageShape()).c_str(),
                (std::string("dim 0 must be at least ") + std::to_string(A)).c_str()),
            return ge::GRAPH_FAILED);
    }
    // 校验expandIdx的维度
    const gert::StorageShape *expandIdxStorageShape = context->GetOutputShape(OUTPUT_EXPAND_IDX_INDEX);
    const int64_t expandIdxDim0 = expandIdxStorageShape->GetStorageShape().GetDim(0);
    if (expandIdxDim0 != expertIdsDim1 * xDim0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "expandIdx",
                                              (std::string("dim0=") + std::to_string(expandIdxDim0)).c_str(),
                                              "expandIdx dim0 should be bs * k");
        return ge::GRAPH_FAILED;
    }
    // 校验expertTokenNums的维度
    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);
    if (isSharedExpert) {
        if (expertTokenNumsDim0 != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "expertTokenNums",
                                                  (std::string("dim0=") + std::to_string(expertTokenNumsDim0)).c_str(),
                                                  "shared expertTokenNums dim0 should be 1");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (expertTokenNumsDim0 != localMoeExpertNum) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "expertTokenNums",
                                                  (std::string("dim0=") + std::to_string(expertTokenNumsDim0)).c_str(),
                                                  "moe expertTokenNums dim0 should be equal to localMoeExpertNum");
            return ge::GRAPH_FAILED;
        }
    }
    // 校验通信参数的维度
    OP_TILING_CHECK(
        CheckCommTensorShape(context, nodeName, tilingData, isSharedExpert, localMoeExpertNum) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "comm token shape", "invalid",
                                              "epRecvCount/tpRecvCount shape validation failed"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(gert::TilingContext *context, const char *nodeName,
                                        MoeDistributeDispatchTilingData &tilingData, const uint32_t quantMode,
                                        const bool isScales, const bool isSharedExpert, const int64_t localMoeExpertNum)
{
    // 输入Tensor校验
    OP_TILING_CHECK(CheckInputTensorShape(context, nodeName, tilingData, isScales) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "input tensor shape", "invalid",
                                                          "Input tensor shape validation failed"),
                    return ge::GRAPH_FAILED);

    uint32_t A = 0;
    uint32_t globalBs = tilingData.moeDistributeDispatchInfo.globalBs;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchInfo.sharedExpertRankNum;
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);

    if (isSharedExpert) { // 本卡为共享专家
        A = globalBs / sharedExpertRankNum;
    } else { // 本卡为moe专家
        A = globalBs * std::min(localMoeExpertNum, expertIdsDim1);
    }
    tilingData.moeDistributeDispatchInfo.a = A;

    // 输出Tensor校验
    OP_TILING_CHECK(CheckOutputTensorShape(context, nodeName, tilingData, quantMode, isSharedExpert, localMoeExpertNum,
                                           A) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "output tensor shape", "invalid",
                                                          "Output tensor shape validation failed"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static uint64_t CalTilingKey(const bool isScales, const uint32_t quantMode, bool isA5)
{
    uint32_t tilingKeyQuantMode = TILINGKEY_NO_QUANT;
    bool scaleMode = false; // A2 & A3
    uint32_t fullMesh = TILINGKEY_NO_FULLMESH;
    uint32_t layeredMode = TILINGKEY_TPL_MTE; // A2
    uint32_t archTag = isA5 ? TILINGKEY_TPL_A5 : TILINGKEY_TPL_A3;

    if (quantMode == STATIC_QUANT_MODE) {
        tilingKeyQuantMode = TILINGKEY_STATIC_QUANT;
    } else if (quantMode == DYNAMIC_QUANT_MODE) {
        tilingKeyQuantMode = TILINGKEY_PERTOKEN_QUANT;
    }
    if (isScales) {
        scaleMode = true;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(tilingKeyQuantMode, scaleMode, fullMesh, layeredMode, archTag);
    return tilingKey;
}

static ge::graphStatus SetHcommCfg(const gert::TilingContext *context, MoeDistributeDispatchTilingData *tiling,
                                   const std::string groupEp)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeDispatch groupEp = %s", groupEp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE); // 通过不拉起AICPU，提高算子退出性能
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling) != 0,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1) != 0,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling1 failed"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workSpaces"), return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWinSize(const gert::TilingContext *context, MoeDistributeDispatchTilingData *tilingData,
                                    const char *nodeName, uint32_t localMoeExpertNum)
{
    uint64_t hcclBufferSizeEp = 0;
    uint64_t maxWindowSizeEp = 0;
    OP_TILING_CHECK(mc2tiling::GetEpWinSize(context, nodeName, hcclBufferSizeEp, maxWindowSizeEp, ATTR_GROUP_EP_INDEX,
                                            false) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Get EP WinSize failed"), return ge::GRAPH_FAILED);
    uint64_t h = static_cast<uint64_t>(tilingData->moeDistributeDispatchInfo.h);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData->moeDistributeDispatchInfo.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData->moeDistributeDispatchInfo.globalBs) / epWorldSize;
    uint64_t actualSize = epWorldSize * maxBs * h * 2UL * 2UL * static_cast<uint64_t>(localMoeExpertNum);
    if (actualSize > maxWindowSizeEp) {
        OP_LOGE_WITHOUT_REPORT(
            nodeName,
            "HCCL_BUFFSIZE is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu, localMoeExpertNum = %u,"
            "ep_worldsize * maxBs * h * 2 * 2 * localMoeExpertNum = %luMB, HCCL_BUFFSIZE=%luMB.",
            maxBs, h, epWorldSize, localMoeExpertNum, actualSize / MB_SIZE + 1UL, hcclBufferSizeEp / MB_SIZE);
        return ge::GRAPH_FAILED;
    }
    tilingData->moeDistributeDispatchInfo.totalWinSizeEp = maxWindowSizeEp;
    OP_LOGD(nodeName, "EpwindowSize = %lu", maxWindowSizeEp);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchTilingBase::MoeDistributeDispatchA3A5TilingCheckAttr(gert::TilingContext *context,
                                                                                          uint32_t &quantMode,
                                                                                          bool &isScales)
{
    const char *nodeName = context->GetNodeName();
    MoeDistributeDispatchTilingData *tilingData = context->GetTilingData<MoeDistributeDispatchTilingData>();
    std::string groupEp = "";
    uint32_t localMoeExpertNum = 1;
    // 获取入参属性
    OP_TILING_CHECK(
        GetAttrAndSetTilingData(context, nodeName, *tilingData, groupEp) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "attr", "invalid", "Get attr and set tiling data failed"),
        return ge::GRAPH_FAILED);
    // 获取scales
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    isScales = (scalesStorageShape != nullptr);
    tilingData->moeDistributeDispatchInfo.isQuant = isScales;
    quantMode = tilingData->moeDistributeDispatchInfo.quantMode;
    // 检查quantMode和scales是否匹配
    OP_TILING_CHECK(quantMode == STATIC_SCALES, OP_LOGE_FOR_INVALID_VALUE(nodeName, "quantMode", "static", "dynamic"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((isScales && (quantMode == NO_SCALES)) || ((!isScales) && (quantMode == STATIC_SCALES)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "quantMode/scales",
                        (std::to_string(static_cast<int32_t>(isScales)) + "/" + std::to_string(quantMode)).c_str(),
                        "The values of quantMode and scales must match."),
                    return ge::GRAPH_FAILED);
    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(MoeDistributeDispatchTilingHelper::TilingCheckMoeDistributeDispatch(context, nodeName, isScales,
                                                                                        quantMode) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "param", "invalid", "Tiling check param failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckExpertScalesDtypeAndFormat(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "expert_scales", "invalid",
                                                          "expert_scales dtype or format check failed"),
                    return ge::GRAPH_FAILED);
    // 检查属性的取值是否合法
    OP_TILING_CHECK(CheckAttrs(context, nodeName, *tilingData, localMoeExpertNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "attr", "invalid", "Attr value check failed"),
                    return ge::GRAPH_FAILED);
    bool isSharedExpert = true;
    uint32_t epRankId = tilingData->moeDistributeDispatchInfo.epRankId;
    uint32_t sharedExpertRankNum = tilingData->moeDistributeDispatchInfo.sharedExpertRankNum;
    if (epRankId >= sharedExpertRankNum) { // 本卡为moe专家
        isSharedExpert = false;
    }
    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(
        CheckTensorShape(context, nodeName, *tilingData, quantMode, isScales, isSharedExpert,
                         static_cast<int64_t>(localMoeExpertNum)) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "tensor shape", "invalid", "Tensor shape check failed"),
        return ge::GRAPH_FAILED);
    // 校验win区大小
    OP_TILING_CHECK(
        CheckWinSize(context, tilingData, nodeName, localMoeExpertNum) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "window size", "invalid", "Window size check failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "workspace", "invalid", "Set workspace failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetHcommCfg(context, tilingData, groupEp) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "hcomm cfg", "invalid", "SetHcommCfg failed"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchTilingBase::MoeDistributeDispatchA3A5TilingFuncImpl(gert::TilingContext *context)
{
    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    auto ret = context->SetScheduleMode(batch_mode);
    MC2_CHECK_LOG_RET(context->GetNodeName(), ret);

    const char *nodeName = context->GetNodeName();
    MoeDistributeDispatchTilingData *tilingData = context->GetTilingData<MoeDistributeDispatchTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "Enter MoeDistributeDispatch tiling check func.");

    uint32_t quantMode = NO_SCALES;
    bool isScales = false;

    OP_TILING_CHECK(MoeDistributeDispatchA3A5TilingCheckAttr(context, quantMode, isScales) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "attr", "invalid",
                                                          "MoeDistributeDispatchA3A5TilingCheckAttr failed"),
                    return ge::GRAPH_FAILED);

    uint64_t tilingKey = CalTilingKey(isScales, quantMode, mc2tiling::GetNpuArch(context) == NpuArch::DAV_3510);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData->moeDistributeDispatchInfo.totalUbSize = ubSize;
    tilingData->moeDistributeDispatchInfo.aivNum = aivNum;
    OP_LOGD(nodeName, "numBlocks=%u, aivNum=%u, ubSize=%lu", numBlocks, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchTilingBase::DoOpTiling() { return MoeDistributeDispatchTilingFunc(context_); }

uint64_t MoeDistributeDispatchTilingBase::GetTilingKey() const
{
    // TilingKey calculation is done in DoOptiling
    const uint64_t tilingKey = context_->GetTilingKey();
    const char *nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeDispatchTiling get tiling key %lu", tilingKey);
    return tilingKey;
}

bool MoeDistributeDispatchTilingBase::IsCapable() { return true; }

struct MoeDistributeDispatchCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeDispatch(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchTiling(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

IMPL_OP_OPTILING(MoeDistributeDispatch)
    .Tiling(MoeDistributeDispatchTiling)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatch);
} // namespace optiling
