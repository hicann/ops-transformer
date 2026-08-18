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
 * \file moe_distribute_combine_tiling.cc
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
#include <type_traits>

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"

#include "op_host/op_tiling/moe_tiling_base.h"
#include "moe_distribute_combine_tiling_base.h"
#include "mc2_hcom_topo_info.h"

using namespace Ops::Transformer::OpTiling;
using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace {
constexpr uint32_t ARR_LENGTH = 128U;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;     // numeric representation of AlltoAll
constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7U; // numeric representation of AlltoAll
constexpr int64_t K_MAX = 8;
constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_GROUP_TP_INDEX = 4;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 5;
constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 6;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 8;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 10;
constexpr uint32_t ATTR_COMM_QUANT_MODE_INDEX = 12;

constexpr uint32_t INT8_COMM_QUANT = 2U;
constexpr uint32_t EXPAND_IDX_DIMS = 1U;
constexpr uint64_t INIT_TILINGKEY_TP_2 = 1100UL;
constexpr uint64_t INIT_TILINGKEY_TP_1 = 1000UL;
constexpr uint32_t TILINGKEY_INT8_COMM_QUANT = 20U;

const int64_t BS_UPPER_BOUND = 512;
const size_t MAX_GROUP_NAME_LENGTH = 128UL;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024; // Bytes
constexpr uint32_t VERSION_2 = 2;
constexpr uint32_t HCOMMCNT_2 = 2;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
const int64_t MAX_EP_WORLD_SIZE = 288;
const int64_t MAX_TP_WORLD_SIZE = 1;
constexpr uint64_t TP_WORLD_SIZE_TWO = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 512;

} // namespace

namespace optiling {

void MoeDistributeCombineTilingBase::PrintTilingDataInfo(const char *nodeName,
                                                         MoeDistributeCombineTilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeCombineInfo.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeCombineInfo.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeCombineInfo.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeCombineInfo.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeDistributeCombineInfo.expertShardType);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeCombineInfo.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeCombineInfo.moeExpertNum);
    OP_LOGD(nodeName, "moeExpertPerRankNum is %u.", tilingData.moeDistributeCombineInfo.moeExpertPerRankNum);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeCombineInfo.globalBs);
    OP_LOGD(nodeName, "bs is %d.", tilingData.moeDistributeCombineInfo.bs);
    OP_LOGD(nodeName, "k is %d.", tilingData.moeDistributeCombineInfo.k);
    OP_LOGD(nodeName, "h is %d.", tilingData.moeDistributeCombineInfo.h);
    OP_LOGD(nodeName, "aivNum is %d.", tilingData.moeDistributeCombineInfo.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %ld.", tilingData.moeDistributeCombineInfo.totalUbSize);
    OP_LOGD(nodeName, "totalWinSizeEP is %lu.", tilingData.moeDistributeCombineInfo.totalWinSizeEp);
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
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COMM_QUANT_MODE_INDEX));

    OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "groupEp"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epWorldSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpWorldSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epRankId"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpRankId"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertShardType"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sharedExpertRankNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "moeExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sharedExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commQuantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "commQuantMode"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCommAttrValuesValid(gert::TilingContext *context, const char *nodeName,
                                                std::string &groupTp)
{
    auto attrs = context->GetAttrs();
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COMM_QUANT_MODE_INDEX));

    OP_TILING_CHECK((*epWorldSizePtr <= 0) || (*epWorldSizePtr > MAX_EP_WORLD_SIZE),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "epWorldSize", std::to_string(*epWorldSizePtr).c_str(),
                                              (std::string("(0, ") + std::to_string(MAX_EP_WORLD_SIZE) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > MAX_TP_WORLD_SIZE),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "tpWorldSize", std::to_string(*tpWorldSizePtr).c_str(),
                                              (std::string("[0, ") + std::to_string(MAX_TP_WORLD_SIZE) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *tpWorldSizePtr >= 2,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "tpWorldSize", std::to_string(*tpWorldSizePtr).c_str(),
                                              "0 or 1 (TP domain communication is not supported)"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= *epWorldSizePtr),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "epRankId", std::to_string(*epRankIdPtr).c_str(),
                                              (std::string("[0, ") + std::to_string(*epWorldSizePtr) + ")").c_str()),
                    return ge::GRAPH_FAILED);
    if (*tpRankIdPtr != 0) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "tpRankId", std::to_string(*tpRankIdPtr).c_str(), "0");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrValuesValid(gert::TilingContext *context, const char *nodeName, std::string &groupTp)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COMM_QUANT_MODE_INDEX));

    OP_TILING_CHECK(CheckCommAttrValuesValid(context, nodeName, groupTp) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "ep/tp attrs", "invalid",
                                                          "The value of ep/tp attrs must be valid."),
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
    OP_TILING_CHECK(
        *sharedExpertNumPtr != 1,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertNum", std::to_string(*sharedExpertNumPtr).c_str(), "1"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*moeExpertNumPtr <= 0) || (*moeExpertNumPtr > MOE_EXPERT_MAX_NUM),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "moeExpertNum", std::to_string(*moeExpertNumPtr).c_str(),
                                              (std::string("(0, ") + std::to_string(MOE_EXPERT_MAX_NUM) + "]").c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*commQuantModePtr != 0) && (*commQuantModePtr != 2),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "commQuantMode", std::to_string(*commQuantModePtr).c_str(), "0 or 2"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTilingBase::GetAttrAndSetTilingData(gert::TilingContext *context,
                                                                        MoeDistributeCombineTilingData &tilingData,
                                                                        const char *nodeName, std::string &groupEp,
                                                                        std::string &groupTp, uint32_t &commQuantMode)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COMM_QUANT_MODE_INDEX));

    // 校验指针非空
    OP_TILING_CHECK(CheckAttrPointersNotNull(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "attrs", "pointer null or empty",
                                                          "All attr pointers must be non-null and valid"),
                    return ge::GRAPH_FAILED);

    // 判断是否满足uint32_t及其他限制
    OP_TILING_CHECK(
        CheckAttrValuesValid(context, nodeName, groupTp) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "attrs", "invalid value", "All attr values must be valid"),
        return ge::GRAPH_FAILED);

    commQuantMode = static_cast<uint32_t>(*commQuantModePtr);
    groupEp = string(groupEpPtr);
    tilingData.moeDistributeCombineInfo.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
    tilingData.moeDistributeCombineInfo.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDistributeCombineInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDistributeCombineInfo.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.moeDistributeCombineInfo.expertShardType = static_cast<uint32_t>(*expertShardPtr);
    tilingData.moeDistributeCombineInfo.sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    tilingData.moeDistributeCombineInfo.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);

    return ge::GRAPH_SUCCESS;
}

static bool CheckExpertAttrs(gert::TilingContext *context, MoeDistributeCombineTilingData &tilingData,
                             const char *nodeName, uint32_t &localMoeExpertNum)
{
    uint32_t epWorldSize = tilingData.moeDistributeCombineInfo.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDistributeCombineInfo.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeCombineInfo.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineInfo.sharedExpertRankNum;

    // 校验ep能均分共享
    if ((sharedExpertRankNum != 0) && (epWorldSize % sharedExpertRankNum != 0)) {
        OP_LOGE_WITH_INVALID_ATTR(
            nodeName, "epWorldSize", std::to_string(epWorldSize).c_str(),
            (std::string("divisible by sharedExpertRankNum (") + std::to_string(sharedExpertRankNum) + ")").c_str());
        return false;
    }

    // 校验moe专家数量能否均分给多机
    if (moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0) {
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "moeExpertNum",
                                  (std::string("moeExpertNum=") + std::to_string(moeExpertNum) +
                                   ", epWorldSize=" + std::to_string(epWorldSize) +
                                   ", sharedExpertRankNum=" + std::to_string(sharedExpertRankNum))
                                      .c_str(),
                                  "moeExpertNum % (epWorldSize - sharedExpertRankNum) == 0");
        return false;
    }
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    if (localMoeExpertNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "localMoeExpertNum", std::to_string(localMoeExpertNum).c_str(), "positive");
        return false;
    }
    if ((localMoeExpertNum > 1) && (tpWorldSize > 1)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "localMoeExpertNum", std::to_string(localMoeExpertNum).c_str(),
                                              "Cannot support multiple moe experts in a rank when tpWorldSize > 1");
        return false;
    }
    tilingData.moeDistributeCombineInfo.moeExpertPerRankNum = localMoeExpertNum;

    // 校验k > moeExpertNum
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    uint32_t K = static_cast<uint32_t>(expertIdsDim1);
    if (K > moeExpertNum) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "K", std::to_string(K).c_str(),
                                  (std::string("<= moeExpertNum (") + std::to_string(moeExpertNum) + ")").c_str());
        return false;
    }

    return true;
}

bool MoeDistributeCombineTilingBase::CheckEpWorldSizeAttrs(gert::TilingContext *context,
                                                           MoeDistributeCombineTilingData &tilingData,
                                                           const char *nodeName)
{
    uint32_t epWorldSize = tilingData.moeDistributeCombineInfo.epWorldSize;

    return CheckEpWorldSize(nodeName, epWorldSize);
}

static bool CheckBatchAttrs(gert::TilingContext *context, MoeDistributeCombineTilingData &tilingData,
                            const char *nodeName)
{
    uint32_t epWorldSize = tilingData.moeDistributeCombineInfo.epWorldSize;

    // 校验输入expertIds的维度0并设bs
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    if (expertIdsStorageShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIds");
        return false;
    }
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    if ((expertIdsDim0 <= 0) || (expertIdsDim0 > BS_UPPER_BOUND)) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "BS", std::to_string(expertIdsDim0).c_str(),
                                  (std::string("[1, ") + std::to_string(BS_UPPER_BOUND) + "]").c_str());
        return false;
    }
    tilingData.moeDistributeCombineInfo.bs = static_cast<uint32_t>(expertIdsDim0);

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return false);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "globalBs"), return false);
    OP_LOGD(nodeName, "MoeDistributeCombine *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n", *globalBsPtr,
            expertIdsDim0, epWorldSize);

    if ((*globalBsPtr != 0) && ((*globalBsPtr < static_cast<int64_t>(epWorldSize) * expertIdsDim0) ||
                                ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0))) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "globalBS", std::to_string(*globalBsPtr).c_str(),
                                  "0 or maxBs * epWorldSize");
        return false;
    }
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeCombineInfo.globalBs = static_cast<uint32_t>(expertIdsDim0) * epWorldSize;
    } else {
        tilingData.moeDistributeCombineInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);
    }

    return true;
}

static bool CheckBasicInputTensorShape(gert::TilingContext *context, MoeDistributeCombineTilingData &tilingData,
                                       const char *nodeName, bool isShared, uint32_t localExpertNum)
{
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);

    uint32_t A = 0;
    uint32_t globalBs = tilingData.moeDistributeCombineInfo.globalBs;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineInfo.sharedExpertRankNum;
    if (isShared) { // 本卡为共享专家
        A = globalBs / sharedExpertRankNum;
    } else { // 本卡为moe专家
        A = globalBs * std::min(static_cast<int64_t>(localExpertNum), expertIdsDim1);
    }
    tilingData.moeDistributeCombineInfo.a = A;

    // 校验expandX的维度并设h
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineInfo.tpWorldSize);
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(EXPAND_X_INDEX);
    int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    if (expandXDim0 < tpWorldSize * static_cast<int64_t>(A)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "expandX",
                                              (std::string("dim0=") + std::to_string(expandXDim0)).c_str(),
                                              "expandX dim0 should be >= A * tpWorldSize");
        return false;
    }
    OP_TILING_CHECK((expandXDim1 != 7168),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "H", std::to_string(expandXDim1).c_str(), "7168"),
                    return false);
    tilingData.moeDistributeCombineInfo.h = static_cast<uint32_t>(expandXDim1);

    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "K", std::to_string(expertIdsDim1).c_str(),
                                              (std::string("(0, ") + std::to_string(K_MAX) + "]").c_str()),
                    return false);
    tilingData.moeDistributeCombineInfo.k = static_cast<uint32_t>(expertIdsDim1);

    // 校验expandIdx的维度
    const gert::StorageShape *expandIdxStorageShape = context->GetInputShape(EXPAND_IDX_INDEX);
    int64_t expandIdxDim0 = expandIdxStorageShape->GetStorageShape().GetDim(0);
    if (expandIdxDim0 != expertIdsDim0 * expertIdsDim1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "expandIdx",
                                              (std::string("dim0=") + std::to_string(expandIdxDim0)).c_str(),
                                              "expandIdx dim0 should be bs * k");
        return false;
    }

    return true;
}

static bool CheckCommInputTensorShape(gert::TilingContext *context, MoeDistributeCombineTilingData &tilingData,
                                      const char *nodeName, bool isShared)
{
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);

    // 校验epSendCount的维度
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineInfo.epWorldSize);
    int64_t moeExpertPerRankNum = static_cast<int64_t>(tilingData.moeDistributeCombineInfo.moeExpertPerRankNum);
    const gert::StorageShape *epSendCountStorageShape = context->GetInputShape(EP_SEND_COUNTS_INDEX);
    const int64_t epSendCountDim0 = epSendCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epSendCount = (isShared) ? epWorldSize : epWorldSize * moeExpertPerRankNum;
    OP_TILING_CHECK(
        epSendCountDim0 < epSendCount,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "ep_send_counts", Ops::Base::ToString(epSendCountStorageShape->GetStorageShape()).c_str(),
            (std::string("dim 0 must be at least ") + std::to_string(epSendCount)).c_str()),
        return false);

    // 校验expertScales的维度
    const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(EXPERT_SCALES_INDEX);
    int64_t expertScalesDim0 = expertScalesStorageShape->GetStorageShape().GetDim(0);
    int64_t expertScalesDim1 = expertScalesStorageShape->GetStorageShape().GetDim(1);
    if (expertScalesDim0 != expertIdsDim0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "expertScales",
                                              (std::string("dim0=") + std::to_string(expertScalesDim0)).c_str(),
                                              "expertScales dim0 should be equal to bs");
        return false;
    }
    if (expertScalesDim1 != expertIdsDim1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "expertScales",
                                              (std::string("dim1=") + std::to_string(expertScalesDim1)).c_str(),
                                              "expertScales dim1 should be equal to k");
        return false;
    }

    return true;
}

bool MoeDistributeCombineTilingBase::CheckTensorShape(gert::TilingContext *context,
                                                      MoeDistributeCombineTilingData &tilingData, const char *nodeName,
                                                      bool isShared, uint32_t localExpertNum)
{
    // 校验输入expertIds的维度1并设k, bs已校验过
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(EXPAND_X_INDEX);
    int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(!CheckBasicInputTensorShape(context, tilingData, nodeName, isShared, localExpertNum),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "input", "invalid shape",
                                                          "Basic input tensor shapes validation failed"),
                    return false);

    OP_TILING_CHECK(!CheckCommInputTensorShape(context, tilingData, nodeName, isShared),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "comm input", "invalid shape",
                                                          "Comm input tensor shapes validation failed"),
                    return false);

    // 校验x的维度
    const gert::StorageShape *xStorageShape = context->GetOutputShape(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return false);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    if (xDim0 != expertIdsDim0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "x", (std::string("dim0=") + std::to_string(xDim0)).c_str(),
                                              "x dim0 should be equal to bs");
        return false;
    }
    if (xDim1 != expandXDim1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "x", (std::string("dim1=") + std::to_string(xDim1)).c_str(),
                                              "x dim1 should be equal to h");
        return false;
    }

    return true;
}

ge::graphStatus MoeDistributeCombineTilingBase::SetWorkspace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE_WITHOUT_REPORT(nodeName, "get workspace failed"),
                    return ge::GRAPH_FAILED);
    workspace[0] = SYSTEM_NEED_WORKSPACE;
    OP_LOGD(nodeName, "workspace[0] size is %ld", workspace[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTilingBase::SetHCommCfg(const gert::TilingContext *context,
                                                            MoeDistributeCombineTilingData *tiling,
                                                            const std::string groupEp, const std::string groupTp,
                                                            const uint32_t tpWorldSize)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeCombine groupEp = %s", groupEp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    uint32_t opType2 = OP_TYPE_REDUCE_SCATTER;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigReduceScatterStr = "ReduceScatter=level0:ring";

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

ge::graphStatus MoeDistributeCombineTilingBase::CheckWinSize(const gert::TilingContext *context,
                                                             MoeDistributeCombineTilingData *tilingData,
                                                             const char *nodeName, uint32_t localMoeExpertNum)
{
    auto attrs = context->GetAttrs();
    uint64_t hcclBufferSizeEp = 0;
    uint64_t maxWindowSizeEp = 0;
    OP_TILING_CHECK(mc2tiling::GetEpWinSize(context, nodeName, hcclBufferSizeEp, maxWindowSizeEp, ATTR_GROUP_EP_INDEX,
                                            false) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Get EP WinSize failed"), return ge::GRAPH_FAILED);
    uint64_t h = static_cast<uint64_t>(tilingData->moeDistributeCombineInfo.h);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData->moeDistributeCombineInfo.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData->moeDistributeCombineInfo.globalBs) / epWorldSize;
    uint64_t actualSize = epWorldSize * maxBs * h * 2UL * 2UL * static_cast<uint64_t>(localMoeExpertNum);
    if (actualSize > maxWindowSizeEp) {
        OP_LOGE_WITHOUT_REPORT(
            nodeName,
            "HCCL_BUFFSIZE is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu, localMoeExpertNum = %u,"
            "ep_worldsize * maxBs * h * 2 * 2 * localMoeExpertNum = %luMB, HCCL_BUFFSIZE=%luMB.",
            maxBs, h, epWorldSize, localMoeExpertNum, actualSize / MB_SIZE + 1UL, hcclBufferSizeEp / MB_SIZE);
        return ge::GRAPH_FAILED;
    }
    tilingData->moeDistributeCombineInfo.totalWinSizeEp = maxWindowSizeEp;
    OP_LOGD(nodeName, "EpwindowSize = %lu", maxWindowSizeEp);

    return ge::GRAPH_SUCCESS;
}

bool MoeDistributeCombineTilingBase::CheckAttrs(gert::TilingContext *context,
                                                MoeDistributeCombineTilingData &tilingData, const char *nodeName,
                                                uint32_t &localMoeExpertNum)
{
    OP_TILING_CHECK(
        !CheckExpertAttrs(context, tilingData, nodeName, localMoeExpertNum),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "expert", "invalid", "Expert params validation failed"),
        return false);

    OP_TILING_CHECK(!CheckEpWorldSizeAttrs(context, tilingData, nodeName),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "epWorldSize", "invalid",
                                                          "EpWorldSize params validation failed"),
                    return false);

    OP_TILING_CHECK(
        !CheckBatchAttrs(context, tilingData, nodeName),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "batch", "invalid", "Batch params validation failed"),
        return false);

    return true;
}

ge::graphStatus MoeDistributeCombineTilingBase::MoeDistributeCombineTilingFunc(gert::TilingContext *context)
{
    // 不支持 expandX数据类型为int32 type
    auto expandXDesc = context->GetInputDesc(EXPAND_X_INDEX);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandxDesc"),
                    return ge::GRAPH_FAILED);
    // 检查expandX数据类型为DT_INT32
    OP_TILING_CHECK(
        (expandXDesc->GetDataType() == ge::DT_INT32),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expandX", Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                                  "bf16 or float16"),
        return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    ge::graphStatus ret = MoeDistributeCombineTilingFuncImpl(socVersion, context);
    return ret;
}

struct MoeDistributeCombineCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeCombine(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTilingBase::DoOpTiling() { return MoeDistributeCombineTilingFunc(context_); }

ge::graphStatus MoeDistributeCombineTiling(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

uint64_t MoeDistributeCombineTilingBase::GetTilingKey() const
{
    // TilingKey calculation is done in DoOptiling
    const uint64_t tilingKey = context_->GetTilingKey();
    const char *nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeCombineTilingBase get tiling key %lu", tilingKey);
    return tilingKey;
}

bool MoeDistributeCombineTilingBase::IsCapable() { return true; }

IMPL_OP_OPTILING(MoeDistributeCombine)
    .Tiling(MoeDistributeCombineTiling)
    .TilingParse<MoeDistributeCombineCompileInfo>(TilingParseForMoeDistributeCombine);

} // namespace optiling
