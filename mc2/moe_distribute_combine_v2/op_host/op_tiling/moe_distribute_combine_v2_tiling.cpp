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
 * \file moe_distribute_combine_v2_tiling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../../common/op_kernel/mc2_moe_context.h"
#include "../../op_kernel/moe_distribute_combine_v2_tiling.h"
#include "mc2_hcom_topo_info.h"
#include "../../../moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_check_win_size.h"
#include "mc2_exception_dump.h"
#include "moe_distribute_combine_v2_tiling_base.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace {
constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
constexpr uint32_t OP_VERSION_2 = 2;
constexpr uint32_t EXPAND_X_INDEX = 0;
constexpr uint32_t EXPERT_IDS_INDEX = 1;
constexpr uint32_t ASSIST_INFO_INDEX = 2;
constexpr uint32_t EP_SEND_COUNTS_INDEX = 3;
constexpr uint32_t EXPERT_SCALES_INDEX = 4;
constexpr uint32_t EXPAND_SCALES_INDEX = 10;

constexpr uint32_t INT8_COMM_QUANT = 2U;
constexpr uint64_t INIT_TILINGKEY = 10000;
constexpr uint64_t TILINGKEY_TP_WORLD_SIZE = 100;
constexpr uint32_t TILINGKEY_INT8_COMM_QUANT = 20U;

constexpr uint32_t THREE_DIMS = 3U;
constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t ASSIST_INFO_DIMS = 1U;
constexpr uint32_t ARR_LENGTH = 128U;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;     // numeric representation of AlltoAll
constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7U; // numeric representation of AlltoAll
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t ALIGNED_LEN = 256U;
constexpr uint32_t DTYPE_SIZE_HALF = 2;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_FLAG_COUNT = 8U;
constexpr uint8_t BUFFER_SINGLE = 1;
constexpr uint8_t BUFFER_NUM = 2;

constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr int64_t MAX_EP_WORLD_SIZE_A3 = 768L; // 384 * 2
constexpr int64_t MAX_EP_WORLD_SIZE_A5 = 1024L;
constexpr int64_t MAX_EP_WORLD_SIZE_LAYERED = 256;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t EP_RESTRICT_8 = 8;
constexpr int64_t MAX_TP_WORLD_SIZE = 1;
constexpr int64_t BS_UPPER_BOUND = 512;
constexpr int64_t BS_UPPER_BOUND_LAYERED = 256;
constexpr uint32_t H_BASIC_BLOCK_LAYERED = 32;
constexpr int64_t RESIDUAL_X_DIM2_SIZE = 1;
constexpr uint32_t NUM_PER_REP_FP32 = 64U;
constexpr uint32_t RANK_NUM_PER_NODE = 16U;

constexpr size_t SYSTEM_NEED_WORKSPACE = 16UL * 1024UL * 1024UL;
constexpr size_t MASK_CALC_NEED_WORKSPACE = 10UL * 1024UL;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024; // Bytes
constexpr uint32_t VERSION_2 = 2;
constexpr uint32_t HCOMMCNT_2 = 2;
constexpr uint32_t RANK_LIST_NUM = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
constexpr int64_t MOE_EXPERT_MAX_NUM_LAYERED = 512;
constexpr int64_t LOCAL_EXPERT_MAX_SIZE = 2048;
constexpr int64_t K_MAX = 16;
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 8192;
constexpr int64_t H_MAX_LAYERED = 7168;
constexpr uint64_t TRIPLE = 3;
constexpr uint64_t ASSIST_NUM_PER_A = 128UL;
constexpr int64_t ELASTIC_METAINFO_OFFSET = 4;
constexpr uint32_t HAS_ADD_RMS_NORM = 1;

enum class CommQuantMode : int32_t {
    NON_QUANT = 0,
    INT12_QUANT = 1,
    INT8_QUANT = 2,
    MXFP8_E5M2_QUANT = 3,
    MXFP8_E4M3_QUANT = 4
};
using CommQuantModeType = std::underlying_type_t<CommQuantMode>;
} // namespace

namespace optiling {
static bool IsPresentTensorShape(const gert::StorageShape *storageShape)
{
    if (storageShape == nullptr) {
        return false;
    }
    const auto &shape = storageShape->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) == 0) {
            return false;
        }
    }
    return true;
}

// a3专有
static void PrintTilingDataInfo(const char *nodeName, MoeDistributeCombineV2TilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeCombineV2Info.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeCombineV2Info.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeCombineV2Info.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeCombineV2Info.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeDistributeCombineV2Info.expertShardType);
    OP_LOGD(nodeName, "sharedExpertNum is %u.", tilingData.moeDistributeCombineV2Info.sharedExpertNum);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeCombineV2Info.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeCombineV2Info.moeExpertNum);
    OP_LOGD(nodeName, "moeExpertPerRankNum is %u.", tilingData.moeDistributeCombineV2Info.moeExpertPerRankNum);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeCombineV2Info.globalBs);
    OP_LOGD(nodeName, "bs is %u.", tilingData.moeDistributeCombineV2Info.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.moeDistributeCombineV2Info.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.moeDistributeCombineV2Info.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.moeDistributeCombineV2Info.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.moeDistributeCombineV2Info.totalUbSize);
    OP_LOGD(nodeName, "totalWinSizeEP is %lu.", tilingData.moeDistributeCombineV2Info.totalWinSizeEp);
    OP_LOGD(nodeName, "hasElastic is %d.", tilingData.moeDistributeCombineV2Info.hasElasticInfo);
    OP_LOGD(nodeName, "isPerformance is %d.", tilingData.moeDistributeCombineV2Info.isPerformance);
    OP_LOGD(nodeName, "hasExpertScales is %d.", tilingData.moeDistributeCombineV2Info.hasExpertScales);
}

static ge::graphStatus GetExpertsAttrAndSetTilingData(const gert::TilingContext *context,
                                                      MoeDistributeCombineV2TilingData &tilingData,
                                                      const char *nodeName, const CombineV2Config &config,
                                                      const bool isLayered)
{
    auto attrs = context->GetAttrs();
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(config.attrMoeExpertNumIndex);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>((config.attrExpertSharedTypeIndex));
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "moeExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "zeroExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "copyExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "constExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertShardType"),
                    return ge::GRAPH_FAILED);
    int64_t moeExpertNum = *moeExpertNumPtr;
    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;
    OP_TILING_CHECK((moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum) > INT32_MAX,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "totalExpertNum", "too large", "should not exceed MAX_INT32"),
                    return ge::GRAPH_FAILED);
    int64_t moeExpertMaxNum = isLayered ? MOE_EXPERT_MAX_NUM_LAYERED : MOE_EXPERT_MAX_NUM;
    OP_TILING_CHECK(
        (moeExpertNum <= 0) || (moeExpertNum > moeExpertMaxNum),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "moeExpertNum", std::to_string(moeExpertNum).c_str(), "in supported range"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (zeroExpertNum < 0),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "zeroExpertNum", std::to_string(zeroExpertNum).c_str(), "should be >= 0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (copyExpertNum < 0),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "copyExpertNum", std::to_string(copyExpertNum).c_str(), "should be >= 0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (constExpertNum < 0),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "constExpertNum", std::to_string(constExpertNum).c_str(), "should be >= 0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *expertShardPtr != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "expertShardType", std::to_string(*expertShardPtr).c_str(), "should be 0"),
        return ge::GRAPH_FAILED);
    tilingData.moeDistributeCombineV2Info.moeExpertNum = static_cast<uint32_t>(moeExpertNum);
    tilingData.moeDistributeCombineV2Info.zeroExpertNum = static_cast<uint32_t>(zeroExpertNum);
    tilingData.moeDistributeCombineV2Info.copyExpertNum = static_cast<uint32_t>(copyExpertNum);
    tilingData.moeDistributeCombineV2Info.constExpertNum = static_cast<uint32_t>(constExpertNum);
    tilingData.moeDistributeCombineV2Info.expertShardType = static_cast<uint32_t>(*expertShardPtr);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetSharedAttrAndSetTilingData(const gert::TilingContext *context,
                                                     MoeDistributeCombineV2TilingData &tilingData, const char *nodeName,
                                                     const CombineV2Config &config, const bool isLayered)
{
    auto attrs = context->GetAttrs();
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>((config.attrSharedExpertNumIndex)));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>((config.attrSharedExpertRankNumIndex));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sharedExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sharedExpertRankNum"),
                    return ge::GRAPH_FAILED);
    int64_t sharedExpertRankNum = *sharedExpertRankNumPtr;
    int64_t epWorldSize = *epWorldSizePtr;
    if (isLayered) {
        OP_TILING_CHECK(
            (sharedExpertRankNum != 0),
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertNum", std::to_string(*sharedExpertNumPtr).c_str(),
                                      "should be 0 in hierarchy mode"),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((*sharedExpertNumPtr < 0) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertNum",
                                                  std::to_string(*sharedExpertNumPtr).c_str(), "in supported range"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK((sharedExpertRankNum < 0) || (sharedExpertRankNum >= epWorldSize),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertRankNum",
                                                  std::to_string(sharedExpertRankNum).c_str(), "in supported range"),
                        return ge::GRAPH_FAILED);
    }
    tilingData.moeDistributeCombineV2Info.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData.moeDistributeCombineV2Info.sharedExpertRankNum = static_cast<uint32_t>(sharedExpertRankNum);
    if (tilingData.moeDistributeCombineV2Info.sharedExpertRankNum == 0U) {
        if (tilingData.moeDistributeCombineV2Info.sharedExpertNum == 1U) {
            tilingData.moeDistributeCombineV2Info.sharedExpertNum = 0U;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTpAndEpAttrAndSetTilingData(const gert::TilingContext *context,
                                                      MoeDistributeCombineV2TilingData &tilingData,
                                                      const char *nodeName, const CombineV2Config &config,
                                                      const bool isLayered, const uint32_t commQuantMode)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrTpWorldSizeIndex));
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>((config.attrEpRankIdIndex));
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>((config.attrTpRankIdIndex));
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epWorldSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpWorldSize"),
                    return ge::GRAPH_FAILED);
    int64_t epWorldSize = *epWorldSizePtr;
    std::string socVersion = mc2tiling::GetSocVersion(context);
    int64_t maxEpworldsize = 0;
    if (socVersion == "Ascend950") {
        maxEpworldsize = isLayered ? MAX_EP_WORLD_SIZE_LAYERED : MAX_EP_WORLD_SIZE_A5;
    } else {
        maxEpworldsize = isLayered ? MAX_EP_WORLD_SIZE_LAYERED : MAX_EP_WORLD_SIZE_A3;
    }
    OP_TILING_CHECK(
        (epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > maxEpworldsize),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "epWorldSize", std::to_string(epWorldSize).c_str(), "in supported range"),
        return ge::GRAPH_FAILED);
    // 校验epWorldSize是否是16整数倍
    OP_TILING_CHECK(
        (isLayered && (epWorldSize % RANK_NUM_PER_NODE != 0)),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "epWorldSize", std::to_string(epWorldSize).c_str(), "should be 16-aligned"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "epRankId", std::to_string(*epRankIdPtr).c_str(), "in supported range"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > MAX_TP_WORLD_SIZE),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "tpWorldSize", std::to_string(*tpWorldSizePtr).c_str(),
                                              "in supported range"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        *tpWorldSizePtr >= 2,
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "tp_world_size", std::to_string(*tpWorldSizePtr).c_str(), "less than 2"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epRankId"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpRankId"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*tpRankIdPtr != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "tpRankId", std::to_string(*tpRankIdPtr).c_str(),
                                              "should be 0 in NoTp mode"),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeCombineV2Info.epWorldSize = static_cast<uint32_t>(epWorldSize);
    tilingData.moeDistributeCombineV2Info.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDistributeCombineV2Info.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDistributeCombineV2Info.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrAndSetTilingData(const gert::TilingContext *context,
                                               MoeDistributeCombineV2TilingData &tilingData, const char *nodeName,
                                               std::string &groupEp, uint32_t &commQuantMode,
                                               const CombineV2Config &config, bool &isLayered)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    if (!config.isMc2Context) {
        auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrGroupEpIndex));
        OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                            (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                        OP_LOGE_WITH_INVALID_INPUT(nodeName, "groupEp"), return ge::GRAPH_FAILED);
        groupEp = string(groupEpPtr);
    }
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>((config.attrCommQuantModeIndex)));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>((config.attrCommAlgIndex)));

    OP_TILING_CHECK(commQuantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "commQuantMode"),
                    return ge::GRAPH_FAILED);
    isLayered = strcmp(commAlgPtr, "hierarchy") == 0;
    if (config.isMc2Context) {
        OP_TILING_CHECK(
            isLayered, OP_LOGE_FOR_INVALID_VALUE(nodeName, "commAlg", commAlgPtr, "does not support comm with context"),
            return ge::GRAPH_FAILED);
    }
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>((config.attrEpWorldSizeIndex)));
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epWorldSize"),
                    return ge::GRAPH_FAILED);
    // 参考A2实现: 单超场景(16卡)无跨超通信收益, hierarchy自动退化为fullmesh
    if (isLayered && (*epWorldSizePtr == RANK_NUM_PER_NODE)) {
        isLayered = false;
        OP_LOGI(nodeName, "epWorldSize is %ld in single supernode, degrade hierarchy to fullmesh.", *epWorldSizePtr);
    }

    if (mc2tiling::GetNpuArch(context) != NpuArch::DAV_3510) {
        OP_TILING_CHECK((*commQuantModePtr != 0) && (*commQuantModePtr != INT8_COMM_QUANT),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "commQuantMode", std::to_string(*commQuantModePtr).c_str(),
                                                  "should be 0 or 2"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((*commQuantModePtr != 0) &&
                            (*commQuantModePtr != static_cast<CommQuantModeType>(CommQuantMode::INT8_QUANT)) &&
                            (*commQuantModePtr != static_cast<CommQuantModeType>(CommQuantMode::MXFP8_E5M2_QUANT)) &&
                            (*commQuantModePtr != static_cast<CommQuantModeType>(CommQuantMode::MXFP8_E4M3_QUANT)),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "commQuantMode", std::to_string(*commQuantModePtr).c_str(),
                                                  "should be 0, 2, 3, or 4"),
                        return ge::GRAPH_FAILED);
    }

    commQuantMode = static_cast<uint32_t>(*commQuantModePtr);
    OP_TILING_CHECK(
        GetExpertsAttrAndSetTilingData(context, tilingData, nodeName, config, isLayered) == ge::GRAPH_FAILED,
        OP_LOGE_WITHOUT_REPORT(nodeName, "Getting experts attr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetSharedAttrAndSetTilingData(context, tilingData, nodeName, config, isLayered) == ge::GRAPH_FAILED,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Getting shared expert attr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetTpAndEpAttrAndSetTilingData(context, tilingData, nodeName, config, isLayered, commQuantMode) ==
                        ge::GRAPH_FAILED,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Getting tp and ep attr failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckARNAttrAndSetTilingData(const gert::TilingContext *context,
                                                    MoeDistributeCombineV2TilingData &tilingData, const char *nodeName,
                                                    const CombineV2Config &config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);
    auto epsilonPtr = attrs->GetAttrPointer<float>(config.attrNormEpsIndex);
    OP_TILING_CHECK(epsilonPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epsilonPtr"), return ge::GRAPH_FAILED);
    tilingData.moeDistributeCombineV2Info.epsilon = *epsilonPtr;
    auto expandXDesc = context->GetInputDesc(config.expandXIndex);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandxDesc"), return false);

    auto residualXDesc = context->GetInputDesc(config.residualXIndex);
    OP_TILING_CHECK(residualXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "residualXDesc"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((residualXDesc->GetDataType() != ge::DT_BF16),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "residualX",
                                                          Ops::Base::ToString(residualXDesc->GetDataType()).c_str(),
                                                          "The dtype of residualX must be bfloat16."),
                    return ge::GRAPH_FAILED);
    auto gammaDesc = context->GetInputDesc(config.gammaIndex);
    OP_TILING_CHECK(gammaDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "gammaDesc"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (gammaDesc->GetDataType() != ge::DT_BF16),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "gamma", Ops::Base::ToString(gammaDesc->GetDataType()).c_str(),
                                              "The dtype of gamma must be bfloat16."),
        return ge::GRAPH_FAILED);
    auto yDesc = context->GetOutputDesc(config.outputYIndex);
    OP_TILING_CHECK(yDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "yDesc"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (yDesc->GetDataType() != expandXDesc->GetDataType()),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "yOut", Ops::Base::ToString(yDesc->GetDataType()).c_str(),
                                              "The dtype of yOut must be the same as that of expandX."),
        return ge::GRAPH_FAILED);
    auto rstdDesc = context->GetOutputDesc(config.outputRstdIndex);
    OP_TILING_CHECK(
        rstdDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "rstdOut", Ops::Base::ToString(rstdDesc->GetDataType()).c_str(),
                                              "The dtype of rstdOut must be float."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool CheckInputTensorDimARN(const gert::TilingContext *context, const char *nodeName,
                                   const CombineV2Config &config)
{
    const gert::StorageShape *residualXStorageShape = context->GetInputShape(config.residualXIndex);
    OP_TILING_CHECK(residualXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "residualX"), return false);
    OP_TILING_CHECK(
        residualXStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "residualX", std::to_string(residualXStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of residualX must be 3D."),
        return false);

    const gert::StorageShape *gammaStorageShape = context->GetInputShape(config.gammaIndex);
    OP_TILING_CHECK(gammaStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "gamma"), return false);
    OP_TILING_CHECK(gammaStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "gamma", std::to_string(gammaStorageShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of gamma must be 1D."),
                    return false);

    return true;
}

static bool CheckInputTensorDim(const gert::TilingContext *context, const char *nodeName, const CombineV2Config &config,
                                const bool hasExpertScales)
{
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(config.expandXIndex);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandX"), return false);
    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "expandX", std::to_string(expandXStorageShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of expandX must be 2D."),
                    return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(config.expertIdsIndex);
    OP_TILING_CHECK(expertIdsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIds"), return false);
    OP_TILING_CHECK(
        expertIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "expertIds", std::to_string(expertIdsStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of expertIds must be 2D."),
        return false);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "expertIds dim0 = %ld", expertIdsDim0);
    OP_LOGD(nodeName, "expertIds dim1 = %ld", expertIdsDim1);

    const gert::StorageShape *assistInfoStorageShape = context->GetInputShape(config.assistInfoIndex);
    OP_TILING_CHECK(assistInfoStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "assistInfoForCombine"),
                    return false);
    OP_TILING_CHECK(assistInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "assistInfoForCombine",
                        std::to_string(assistInfoStorageShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of assistInfoForCombine must be 1D."),
                    return false);
    OP_LOGD(nodeName, "assistInfoForCombine dim0 = %ld", assistInfoStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *epSendCountsStorageShape = context->GetInputShape(config.epSendCountIndex);
    OP_TILING_CHECK(epSendCountsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epSendCounts"),
                    return false);
    OP_TILING_CHECK(
        epSendCountsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "epSendCounts", std::to_string(epSendCountsStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of epSendCounts must be 1D."),
        return false);
    OP_LOGD(nodeName, "epSendCounts dim0 = %ld", epSendCountsStorageShape->GetStorageShape().GetDim(0));

    if (hasExpertScales) {
        const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(config.expertScalesIndex);
        OP_TILING_CHECK(expertScalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertScales"),
                        return false);
        OP_TILING_CHECK(expertScalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "expertScales",
                            std::to_string(expertScalesStorageShape->GetStorageShape().GetDimNum()).c_str(),
                            "The shape dim of expertScales must be 2D."),
                        return false);
        OP_LOGD(nodeName, "expertScales dim0 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(0));
        OP_LOGD(nodeName, "expertScales dim1 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(1));
    }

    return true;
}

static bool CheckOptionalScalesTensorDim(const gert::TilingContext *context, const char *nodeName,
                                         const CombineV2Config &config, const bool isLayered)
{
    if (isLayered) {
        const gert::StorageShape *expandScaleStorageShape = context->GetOptionalInputShape(EXPAND_SCALES_INDEX);
        const int64_t expandScaleDim = expandScaleStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(expandScaleStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandScales"),
                        return false);
        OP_TILING_CHECK(
            expandScaleDim != ONE_DIM,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "expandScales", std::to_string(expandScaleDim).c_str(),
                                                     "The shape dim of expandScales must be 1D."),
            return false);
        OP_LOGD(nodeName, "expandScales dim0 = %ld", expandScaleStorageShape->GetStorageShape().GetDim(0));
    }

    const gert::StorageShape *constExpertAlpha1StorageShape =
        context->GetOptionalInputShape(config.constExpertAlpha1Index);
    if (constExpertAlpha1StorageShape != nullptr) {
        OP_TILING_CHECK(constExpertAlpha1StorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "const_expert_alpha_1",
                            std::to_string(constExpertAlpha1StorageShape->GetStorageShape().GetDimNum()).c_str(),
                            "The shape dim of const_expert_alpha_1 must be 2D."),
                        return false);
    }

    const gert::StorageShape *constExpertAlpha2StorageShape =
        context->GetOptionalInputShape(config.constExpertAlpha2Index);
    if (constExpertAlpha2StorageShape != nullptr) {
        OP_TILING_CHECK(constExpertAlpha2StorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "const_expert_alpha_2",
                            std::to_string(constExpertAlpha2StorageShape->GetStorageShape().GetDimNum()).c_str(),
                            "The shape dim of const_expert_alpha_2 must be 2D."),
                        return false);
    }

    const gert::StorageShape *constExpertVStorageShape = context->GetOptionalInputShape(config.constExpertVIndex);
    if (constExpertVStorageShape != nullptr) {
        OP_TILING_CHECK(constExpertVStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "const_expert_v",
                            std::to_string(constExpertVStorageShape->GetStorageShape().GetDimNum()).c_str(),
                            "The shape dim of const_expert_v must be 2D."),
                        return false);
    }

    const gert::StorageShape *activationScaleStorageShape = context->GetOptionalInputShape(config.activationScaleIndex);
    OP_TILING_CHECK(activationScaleStorageShape != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationScale", "present", "should be None"), return false);

    const gert::StorageShape *weightScaleStorageShape = context->GetOptionalInputShape(config.weightScaleIndex);
    OP_TILING_CHECK(weightScaleStorageShape != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "weightScale", "present", "should be None"), return false);

    const gert::StorageShape *sharedExpertX = context->GetOptionalInputShape(config.sharedExpertXIndex);
    if (sharedExpertX != nullptr) {
        auto attrs = context->GetAttrs();
        auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>((config.attrSharedExpertRankNumIndex));
        OP_TILING_CHECK(*sharedExpertRankNumPtr != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "sharedExpertX", "present",
                                                              "only support None when sharedExpertRankNum is non-zero"),
                        return false);
        OP_TILING_CHECK(
            ((sharedExpertX->GetStorageShape().GetDimNum() != TWO_DIMS) &&
             (sharedExpertX->GetStorageShape().GetDimNum() != THREE_DIMS)),
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                nodeName, "sharedExpertX", std::to_string(sharedExpertX->GetStorageShape().GetDimNum()).c_str(),
                "The shape dim of sharedExpertX must be within the range {2D, 3D}."),
            return false);
    }

    return true;
}

static bool CheckOptionalInputTensorDim(const gert::TilingContext *context, const char *nodeName,
                                        const bool isActiveMask, const bool hasElasticInfo, const bool isPerformance,
                                        const CombineV2Config &config, const bool isLayered)
{
    const gert::StorageShape *oriXStorageShape = context->GetOptionalInputShape(config.oriXIndex);
    if (oriXStorageShape != nullptr) {
        OP_TILING_CHECK(oriXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "ori_x", std::to_string(oriXStorageShape->GetStorageShape().GetDimNum()).c_str(),
                            "The shape dim of ori_x must be 2D."),
                        return false);
    }

    OP_TILING_CHECK(!CheckOptionalScalesTensorDim(context, nodeName, config, isLayered),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "optional scales", "",
                                                          "Param shape of optional scales input tensor is invalid"),
                    return false);

    if (isActiveMask) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
        OP_TILING_CHECK(xActiveMaskStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xActiveMask"),
                        return false);
        const int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(((xActiveMaskDimNums != ONE_DIM) && (xActiveMaskDimNums != TWO_DIMS)),
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "xActiveMask", std::to_string(xActiveMaskDimNums).c_str(),
                            "The shape dim of xActiveMask must be within the range {1D, 2D}."),
                        return false);
    }
    if (hasElasticInfo) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "elasticInfo"),
                        return false);
        OP_TILING_CHECK(
            elasticInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                nodeName, "elasticInfo", std::to_string(elasticInfoStorageShape->GetStorageShape().GetDimNum()).c_str(),
                "The shape dim of elasticInfo must be 1D."),
            return false);
        OP_LOGD(nodeName, "elasticInfo dim0 = %ld", elasticInfoStorageShape->GetStorageShape().GetDim(0));
    }

    if (isPerformance) {
        const gert::StorageShape *performanceInfoStorageShape =
            context->GetOptionalInputShape(config.performanceInfoIndex);
        OP_TILING_CHECK(performanceInfoStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "performanceInfo"),
                        return false);
        OP_TILING_CHECK(performanceInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "performanceInfo",
                            std::to_string(performanceInfoStorageShape->GetStorageShape().GetDimNum()).c_str(),
                            "The shape dim of performanceInfo must be 1D."),
                        return false);
        OP_LOGD(nodeName, "performanceInfo dim0 = %ld", performanceInfoStorageShape->GetStorageShape().GetDim(0));
    }

    const gert::StorageShape *groupListStorageShape = context->GetOptionalInputShape(config.groupListIndex);
    OP_TILING_CHECK(groupListStorageShape != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "groupList", "present", "should be None"), return false);

    return true;
}

static bool CheckOutputTensorDimARN(const gert::TilingContext *context, const char *nodeName,
                                    const CombineV2Config &config)
{
    const gert::StorageShape *yStorageShape = context->GetOutputShape(config.outputYIndex);
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "yOut"), return false);
    OP_TILING_CHECK(yStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "yOut", std::to_string(yStorageShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of yOut must be 3D."),
                    return false);
    OP_LOGD(nodeName, "yOut dim0 = %ld", yStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "yOut dim1 = %ld", yStorageShape->GetStorageShape().GetDim(1));
    OP_LOGD(nodeName, "yOut dim2 = %ld", yStorageShape->GetStorageShape().GetDim(TWO_DIMS));

    const gert::StorageShape *rstdStorageShape = context->GetOutputShape(config.outputRstdIndex);
    OP_TILING_CHECK(rstdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "rstdOut"), return false);
    OP_TILING_CHECK(rstdStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "rstdOut", std::to_string(rstdStorageShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of rstdOut must be 3D."),
                    return false);
    OP_LOGD(nodeName, "rstdOut dim0 = %ld", rstdStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "rstdOut dim1 = %ld", rstdStorageShape->GetStorageShape().GetDim(1));
    OP_LOGD(nodeName, "rstdOut dim2 = %ld", rstdStorageShape->GetStorageShape().GetDim(TWO_DIMS));

    const gert::StorageShape *xStorageShape = context->GetOutputShape(config.outputXIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xOut"), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "xOut", std::to_string(xStorageShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of xOut must be 3D."),
                    return false);
    OP_LOGD(nodeName, "xOut dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "xOut dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));
    OP_LOGD(nodeName, "xOut dim2 = %ld", xStorageShape->GetStorageShape().GetDim(TWO_DIMS));

    return true;
}

static bool CheckOutputTensorDim(const gert::TilingContext *context, const char *nodeName,
                                 const CombineV2Config &config, int64_t expertIdsDim0, int64_t expandXDim1)
{
    const gert::StorageShape *xStorageShape = context->GetOutputShape(config.outputXIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "x", std::to_string(xStorageShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of x must be 2D."),
                    return false);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "x", std::to_string(xDim0).c_str(),
                                                             "The shape dim of x must be equal to bs."),
                    return false);
    OP_TILING_CHECK(xDim1 != expandXDim1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "x", std::to_string(xDim1).c_str(),
                                                             "The shape dim of x must be equal to h."),
                    return false);

    return true;
}

static bool CheckTensorDim(gert::TilingContext *context, const char *nodeName, const bool isActiveMask,
                           const bool hasExpertScales, const bool hasElasticInfo, const bool isPerformance,
                           const CombineV2Config &config, const bool isLayered)
{
    OP_TILING_CHECK(
        !CheckInputTensorDim(context, nodeName, config, hasExpertScales),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "input tensor", "", "param shape of input tensor is invalid"),
        return false);

    OP_TILING_CHECK(
        !CheckOptionalInputTensorDim(context, nodeName, isActiveMask, hasElasticInfo, isPerformance, config, isLayered),
        OP_LOGE_WITHOUT_REPORT(nodeName, "param shape of optional input tensor is invalid"), return false);

    return true;
}

static bool CheckExpertTensorDataType(const gert::TilingContext *context, const char *nodeName,
                                      const CombineV2Config &config, const bool hasExpertScales)
{
    auto expandXDesc = context->GetInputDesc(config.expandXIndex);
    auto oriXDesc = context->GetOptionalInputDesc(config.oriXIndex);
    if (oriXDesc != nullptr) {
        OP_TILING_CHECK((oriXDesc->GetDataType() != expandXDesc->GetDataType()),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            nodeName, "ori_x", Ops::Base::ToString(oriXDesc->GetDataType()).c_str(),
                            "The dtype of ori_x must be the same as that of expandX."),
                        return false);
    }

    auto constExpertAlpha1Desc = context->GetOptionalInputDesc(config.constExpertAlpha1Index);
    if (constExpertAlpha1Desc != nullptr) {
        OP_TILING_CHECK(
            (constExpertAlpha1Desc->GetDataType() != expandXDesc->GetDataType()),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                nodeName, "const_expert_alpha_1", Ops::Base::ToString(constExpertAlpha1Desc->GetDataType()).c_str(),
                "The dtype of const_expert_alpha_1 must be the same as that of expandX."),
            return false);
    }

    auto constExpertAlpha2Desc = context->GetOptionalInputDesc(config.constExpertAlpha2Index);
    if (constExpertAlpha2Desc != nullptr) {
        OP_TILING_CHECK(
            (constExpertAlpha2Desc->GetDataType() != expandXDesc->GetDataType()),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                nodeName, "const_expert_alpha_2", Ops::Base::ToString(constExpertAlpha2Desc->GetDataType()).c_str(),
                "The dtype of const_expert_alpha_2 must be the same as that of expandX."),
            return false);
    }

    auto constExpertVDesc = context->GetOptionalInputDesc(config.constExpertVIndex);
    if (constExpertVDesc != nullptr) {
        OP_TILING_CHECK((constExpertVDesc->GetDataType() != expandXDesc->GetDataType()),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            nodeName, "const_expert_v", Ops::Base::ToString(constExpertVDesc->GetDataType()).c_str(),
                            "The dtype of const_expert_v must be the same as that of expandX."),
                        return false);
    }

    auto expertIdsDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdsDesc"), return false);
    OP_TILING_CHECK((expertIdsDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "expertIds",
                                                          Ops::Base::ToString(expertIdsDesc->GetDataType()).c_str(),
                                                          "The dtype of expertIds must be int32."),
                    return false);
    auto sharedExpertXDesc = context->GetOptionalInputDesc(config.sharedExpertXIndex);
    if (sharedExpertXDesc != nullptr) {
        OP_TILING_CHECK(sharedExpertXDesc->GetDataType() != expandXDesc->GetDataType(),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            nodeName, "sharedExpertX", Ops::Base::ToString(sharedExpertXDesc->GetDataType()).c_str(),
                            "The dtype of sharedExpertX must be the same as that of expandX."),
                        return false);
    }
    if (hasExpertScales) {
        auto expertScalesDesc = context->GetInputDesc(config.expertScalesIndex);
        OP_TILING_CHECK(expertScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertScalesDesc"),
                        return false);
        OP_TILING_CHECK((expertScalesDesc->GetDataType() != ge::DT_FLOAT),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            nodeName, "expertScales", Ops::Base::ToString(expertScalesDesc->GetDataType()).c_str(),
                            "The dtype of expertScales must be float."),
                        return false);
    }
    return true;
}

// 校验数据类型
static bool CheckTensorDataType(const gert::TilingContext *context, const char *nodeName, const bool isActiveMask,
                                const bool hasElasticInfo, const bool isPerformance, const CombineV2Config &config)
{
    auto expandXDesc = context->GetInputDesc(config.expandXIndex);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandxDesc"), return false);
    OP_TILING_CHECK((expandXDesc->GetDataType() != ge::DT_BF16) && (expandXDesc->GetDataType() != ge::DT_FLOAT16),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "expandX", Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                        "The dtype of expandX must be within the range {bfloat16, float16}."),
                    return false);
    auto assistInfoDesc = context->GetInputDesc(config.assistInfoIndex);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "assistInfoDesc"), return false);
    OP_TILING_CHECK((assistInfoDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "assistInfoForCombine",
                                                          Ops::Base::ToString(assistInfoDesc->GetDataType()).c_str(),
                                                          "The dtype of assistInfoForCombine must be int32."),
                    return false);
    auto epSendCountsDesc = context->GetInputDesc(config.epSendCountIndex);
    OP_TILING_CHECK(epSendCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epSendCountsDesc"),
                    return false);
    OP_TILING_CHECK((epSendCountsDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "epSendCounts",
                                                          Ops::Base::ToString(epSendCountsDesc->GetDataType()).c_str(),
                                                          "The dtype of epSendCounts must be int32."),
                    return false);

    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xActiveMaskDesc"),
                        return false);
        OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_BOOL,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            nodeName, "xActiveMask", Ops::Base::ToString(xActiveMaskDesc->GetDataType()).c_str(),
                            "The dtype of xActiveMask must be bool."),
                        return false);
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "elasticInfoDesc"),
                        return false);
        OP_TILING_CHECK(elasticInfoDesc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            nodeName, "elasticInfo", Ops::Base::ToString(elasticInfoDesc->GetDataType()).c_str(),
                            "The dtype of elasticInfo must be int32."),
                        return false);
    }
    if (isPerformance) {
        auto performanceInfoDesc = context->GetOptionalInputDesc(config.performanceInfoIndex);
        OP_TILING_CHECK(
            performanceInfoDesc->GetDataType() != ge::DT_INT64,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "performanceInfo",
                                                  Ops::Base::ToString(performanceInfoDesc->GetDataType()).c_str(),
                                                  "The dtype of performanceInfo must be int64."),
            return false);
    }
    auto xDesc = context->GetOutputDesc(config.outputXIndex);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    OP_TILING_CHECK(
        (xDesc->GetDataType() != expandXDesc->GetDataType()),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(),
                                              "The dtype of x must be the same as that of expandX."),
        return false);
    return true;
}

static bool CheckZeroComputeExpertTensorFormat(const gert::TilingContext *context, const char *nodeName,
                                               const CombineV2Config &config)
{
    auto oriXDesc = context->GetOptionalInputDesc(config.oriXIndex);
    if (oriXDesc != nullptr) {
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(oriXDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "ori_x", "not FRACTAL_NZ", "FRACTAL_NZ"), return false);
    }

    auto constExpertAlpha1Desc = context->GetOptionalInputDesc(config.constExpertAlpha1Index);
    if (constExpertAlpha1Desc != nullptr) {
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(constExpertAlpha1Desc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "const_expert_alpha_1", "not FRACTAL_NZ", "FRACTAL_NZ"),
                        return false);
    }

    auto constExpertAlpha2Desc = context->GetOptionalInputDesc(config.constExpertAlpha2Index);
    if (constExpertAlpha2Desc != nullptr) {
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(constExpertAlpha2Desc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "const_expert_alpha_2", "not FRACTAL_NZ", "FRACTAL_NZ"),
                        return false);
    }

    auto constExpertVDesc = context->GetOptionalInputDesc(config.constExpertVIndex);
    if (constExpertVDesc != nullptr) {
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(constExpertVDesc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "const_expert_v", "not FRACTAL_NZ", "FRACTAL_NZ"),
                        return false);
    }
    return true;
}

static bool CheckInputTensorFormat(const gert::TilingContext *context, const char *nodeName,
                                   const CombineV2Config &config, const bool hasExpertScales)
{
    auto expandXDesc = context->GetInputDesc(config.expandXIndex);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandxDesc"), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expandX", "not FRACTAL_NZ", "FRACTAL_NZ"), return false);

    auto expertIdsDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdsDesc"), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expertIds", "not FRACTAL_NZ", "FRACTAL_NZ"), return false);

    auto assistInfoDesc = context->GetInputDesc(config.assistInfoIndex);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "assistInfoDesc"), return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(assistInfoDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "assistInfoForCombine", "not FRACTAL_NZ", "FRACTAL_NZ"), return false);

    auto epSendCountsDesc = context->GetInputDesc(config.epSendCountIndex);
    OP_TILING_CHECK(epSendCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epSendCountsDesc"),
                    return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(epSendCountsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "epSendCounts", "not FRACTAL_NZ", "FRACTAL_NZ"), return false);

    if (hasExpertScales) {
        auto expertScalesDesc = context->GetInputDesc(config.expertScalesIndex);
        OP_TILING_CHECK(expertScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertScalesDesc"),
                        return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertScalesDesc->GetStorageFormat())) ==
                            ge::FORMAT_FRACTAL_NZ,
                        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expertScales", "not FRACTAL_NZ", "FRACTAL_NZ"),
                        return false);
    }

    return true;
}

static bool CheckTensorFormat(const gert::TilingContext *context, const char *nodeName, const bool isActiveMask,
                              const bool hasExpertScales, const bool hasElasticInfo, const bool isPerformance,
                              const CombineV2Config &config)
{
    OP_TILING_CHECK(!CheckZeroComputeExpertTensorFormat(context, nodeName, config),
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "zero_compute_expert", "invalid", "FRACTAL_NZ"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckInputTensorFormat(context, nodeName, config, hasExpertScales),
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "input tensor", "invalid", "FRACTAL_NZ"),
                    return ge::GRAPH_FAILED);
    // 除特殊专家外的可选输入Format check
    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xActiveMaskDesc"),
                        return false);
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "xActiveMask", "not FRACTAL_NZ", "FRACTAL_NZ"), return false);
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "elasticInfoDesc"),
                        return false);
        auto elasticInfoDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(elasticInfoDesc->GetStorageFormat()));
        if (elasticInfoDescFormat == ge::FORMAT_FRACTAL_NZ) {
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "elasticInfo", Ops::Base::ToString(elasticInfoDescFormat).c_str(),
                                       "ND");
            return false;
        }
    }
    if (isPerformance) {
        auto performanceInfoDesc = context->GetOptionalInputDesc(config.performanceInfoIndex);
        auto performanceInfoDescFormat =
            static_cast<ge::Format>(ge::GetPrimaryFormat(performanceInfoDesc->GetStorageFormat()));
        if (performanceInfoDescFormat == ge::FORMAT_FRACTAL_NZ) {
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "performanceInfo",
                                       Ops::Base::ToString(performanceInfoDescFormat).c_str(), "ND");
            return false;
        }
    }
    auto sharedExpertXDesc = context->GetOptionalInputDesc(config.sharedExpertXIndex);
    OP_TILING_CHECK((sharedExpertXDesc != nullptr) &&
                        (static_cast<ge::Format>(ge::GetPrimaryFormat(sharedExpertXDesc->GetStorageFormat())) ==
                         ge::FORMAT_FRACTAL_NZ),
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "sharedExpertX", "not FRACTAL_NZ", "FRACTAL_NZ"),
                    return false);
    // 输出Format check
    auto xDesc = context->GetOutputDesc(config.outputXIndex);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x", "not FRACTAL_NZ", "FRACTAL_NZ"), return false);

    return true;
}

static bool CheckTensorShapeARN(const gert::TilingContext *context, const char *nodeName, const CombineV2Config &config,
                                int64_t expertIdsDim0, int64_t expandXDim1)
{
    // 校验residualX的维度
    const gert::StorageShape *residualXShape = context->GetOptionalInputShape(config.residualXIndex);
    if (residualXShape != nullptr) {
        int64_t residualXDimBs = residualXShape->GetStorageShape().GetDim(0);
        int64_t residualXDim1 = residualXShape->GetStorageShape().GetDim(1);
        int64_t residualXDimH = residualXShape->GetStorageShape().GetDim(TWO_DIMS);
        OP_TILING_CHECK(
            residualXDimBs != expertIdsDim0 || residualXDim1 != RESIDUAL_X_DIM2_SIZE || residualXDimH != expandXDim1,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                nodeName, "residualX",
                ("[" + std::to_string(residualXDimBs) + ", " + std::to_string(residualXDim1) + ", " +
                 std::to_string(residualXDimH) + "]")
                    .c_str(),
                ("Shape mismatch: residualX expected dim0=" + std::to_string(expertIdsDim0) +
                 ", dim1=" + std::to_string(RESIDUAL_X_DIM2_SIZE) + ", dim2=" + std::to_string(expandXDim1))
                    .c_str()),
            return false);
    }
    // 校验gamma的维度
    const gert::StorageShape *gammaShape = context->GetOptionalInputShape(config.gammaIndex);
    if (gammaShape != nullptr) {
        int64_t gammaDimH = gammaShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(
            gammaDimH != expandXDim1,
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "gamma", std::to_string(gammaDimH).c_str(),
                                      ("gamma's dim0 should equal to h=" + std::to_string(expandXDim1)).c_str()),
            return false);
    }
    // 校验y的维度
    const gert::StorageShape *yStorageShape = context->GetOutputShape(config.outputYIndex);
    int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);
    int64_t yDim2 = yStorageShape->GetStorageShape().GetDim(TWO_DIMS);
    OP_TILING_CHECK(
        yDim0 != expertIdsDim0 || yDim1 != 1 || yDim2 != expandXDim1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "y",
            ("[" + std::to_string(yDim0) + ", " + std::to_string(yDim1) + ", " + std::to_string(yDim2) + "]").c_str(),
            ("Shape mismatch: y expected dim0=" + std::to_string(expertIdsDim0) +
             ", dim1=1, dim2=" + std::to_string(expandXDim1))
                .c_str()),
        return false);
    // 校验rstd的维度
    const gert::StorageShape *rstdStorageShape = context->GetOutputShape(config.outputRstdIndex);
    int64_t rstdDim0 = rstdStorageShape->GetStorageShape().GetDim(0);
    int64_t rstdDim1 = rstdStorageShape->GetStorageShape().GetDim(1);
    int64_t rstdDim2 = rstdStorageShape->GetStorageShape().GetDim(TWO_DIMS);
    OP_TILING_CHECK(
        rstdDim0 != expertIdsDim0 || rstdDim1 != 1 || rstdDim2 != 1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "rstd",
            ("[" + std::to_string(rstdDim0) + ", " + std::to_string(rstdDim1) + ", " + std::to_string(rstdDim2) + "]")
                .c_str(),
            ("Shape mismatch: rstd expected dim0=" + std::to_string(expertIdsDim0) + ", dim1=1, dim2=1").c_str()),
        return false);
    // 校验x的维度
    const gert::StorageShape *xStorageShape = context->GetOutputShape(config.outputXIndex);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    int64_t xDim2 = xStorageShape->GetStorageShape().GetDim(TWO_DIMS);
    OP_TILING_CHECK(
        xDim0 != expertIdsDim0 || xDim1 != 1 || xDim2 != expandXDim1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "xOut",
            ("[" + std::to_string(xDim0) + ", " + std::to_string(xDim1) + ", " + std::to_string(xDim2) + "]").c_str(),
            ("Shape mismatch: xOut expected dim0=" + std::to_string(expertIdsDim0) +
             ", dim1=1, dim2=" + std::to_string(expandXDim1))
                .c_str()),
        return false);
    return true;
}

static bool CheckZeroComputeExpertsTensorShape(const gert::TilingContext *context,
                                               MoeDistributeCombineV2TilingData &tilingData, const char *nodeName,
                                               const CombineV2Config &config, const uint32_t expertIdsDim0)
{
    const gert::StorageShape *oriXShape = context->GetOptionalInputShape(config.oriXIndex);
    int64_t expandXDim1 = (context->GetInputShape(config.expandXIndex))->GetStorageShape().GetDim(1);
    if (oriXShape != nullptr) {
        int64_t oriXDim0 = oriXShape->GetStorageShape().GetDim(0);
        int64_t oriXDim1 = oriXShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(oriXDim0 != expertIdsDim0,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "ori_x", std::to_string(oriXDim0).c_str(),
                                                                 "The shape dim of ori_x must be equal to bs."),
                        return false);
        OP_TILING_CHECK(oriXDim1 != expandXDim1,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "ori_x", std::to_string(oriXDim1).c_str(),
                                                                 "The shape dim of ori_x must be equal to h."),
                        return false);
    }

    const gert::StorageShape *constExpertAlpha1Shape = context->GetOptionalInputShape(config.constExpertAlpha1Index);
    if (constExpertAlpha1Shape != nullptr) {
        int64_t constExpertAlpha1Dim0 = constExpertAlpha1Shape->GetStorageShape().GetDim(0);
        int64_t constExpertAlpha1Dim1 = constExpertAlpha1Shape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(
            constExpertAlpha1Dim0 != static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.constExpertNum),
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                nodeName, "const_expert_alpha_1", std::to_string(constExpertAlpha1Dim0).c_str(),
                "The shape dim of const_expert_alpha_1 must be equal to constExpertNum."),
            return false);
        OP_TILING_CHECK(constExpertAlpha1Dim1 != expandXDim1,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "const_expert_alpha_1", std::to_string(constExpertAlpha1Dim1).c_str(),
                            "The shape dim of const_expert_alpha_1 must be equal to h."),
                        return false);
    }

    const gert::StorageShape *constExpertAlpha2Shape = context->GetOptionalInputShape(config.constExpertAlpha2Index);
    if (constExpertAlpha2Shape != nullptr) {
        int64_t constExpertAlpha2Dim0 = constExpertAlpha2Shape->GetStorageShape().GetDim(0);
        int64_t constExpertAlpha2Dim1 = constExpertAlpha2Shape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(
            constExpertAlpha2Dim0 != static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.constExpertNum),
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                nodeName, "const_expert_alpha_2", std::to_string(constExpertAlpha2Dim0).c_str(),
                "The shape dim of const_expert_alpha_2 must be equal to constExpertNum."),
            return false);
        OP_TILING_CHECK(constExpertAlpha2Dim1 != expandXDim1,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "const_expert_alpha_2", std::to_string(constExpertAlpha2Dim1).c_str(),
                            "The shape dim of const_expert_alpha_2 must be equal to h."),
                        return false);
    }

    const gert::StorageShape *constExpertVShape = context->GetOptionalInputShape(config.constExpertVIndex);
    if (constExpertVShape != nullptr) {
        int64_t constExpertVDim0 = constExpertVShape->GetStorageShape().GetDim(0);
        int64_t constExpertVDim1 = constExpertVShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(constExpertVDim0 != static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.constExpertNum),
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "const_expert_v", std::to_string(constExpertVDim0).c_str(),
                            "The shape dim of const_expert_v must be equal to constExpertNum."),
                        return false);
        OP_TILING_CHECK(constExpertVDim1 != expandXDim1,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "const_expert_v",
                                                                 std::to_string(constExpertVDim1).c_str(),
                                                                 "The shape dim of const_expert_v must be equal to h."),
                        return false);
    }
    return true;
}

static bool CheckExpertsTensorShape(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
                                    const char *nodeName, const CombineV2Config &config, const uint32_t expertIdsDim0,
                                    const uint32_t expertIdsDim1, const bool hasExpertScales)
{
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.moeExpertNum);
    int64_t zeroExpertNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.zeroExpertNum);
    int64_t copyExpertNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.copyExpertNum);
    int64_t constExpertNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.constExpertNum);
    OP_TILING_CHECK(
        (expertIdsDim1 <= 0) ||
            (expertIdsDim1 > K_MAX || (expertIdsDim1 > moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum)),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "expertIds(k)", std::to_string(expertIdsDim1).c_str(),
                                  ("should be in (0, min(K_MAX=" + std::to_string(K_MAX) + ", totalExpertNum=" +
                                   std::to_string(moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum) + ")]")
                                      .c_str()),
        return false);
    tilingData.moeDistributeCombineV2Info.k = static_cast<uint32_t>(expertIdsDim1);

    // 校验expertScales的维度
    if (hasExpertScales) {
        const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(config.expertScalesIndex);
        OP_TILING_CHECK(expertScalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertScales"),
                        return false);
        int64_t expertScalesDim0 = expertScalesStorageShape->GetStorageShape().GetDim(0);
        int64_t expertScalesDim1 = expertScalesStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(
            expertScalesDim0 != expertIdsDim0,
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "expertScales(dim0)", std::to_string(expertScalesDim0).c_str(),
                                      ("should equal to bs=" + std::to_string(expertIdsDim0)).c_str()),
            return false);
        OP_TILING_CHECK(
            expertScalesDim1 != expertIdsDim1,
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "expertScales(dim1)", std::to_string(expertScalesDim1).c_str(),
                                      ("should equal to k=" + std::to_string(expertIdsDim1)).c_str()),
            return false);
    }

    // 校验sharedExpertX的维度
    const gert::StorageShape *sharedExpertXShape = context->GetOptionalInputShape(config.sharedExpertXIndex);
    tilingData.moeDistributeCombineV2Info.hasSharedExpertX = (sharedExpertXShape != nullptr);
    if (sharedExpertXShape != nullptr) {
        int64_t sharedExpertXDim0 = sharedExpertXShape->GetStorageShape().GetDim(0);
        int64_t sharedExpertXDim1 = sharedExpertXShape->GetStorageShape().GetDim(1);
        int64_t expandXDim1 = (context->GetInputShape(config.expandXIndex))->GetStorageShape().GetDim(1);
        if (sharedExpertXShape->GetStorageShape().GetDimNum() == TWO_DIMS) {
            OP_TILING_CHECK(
                sharedExpertXDim0 != expertIdsDim0,
                OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertX(dim0)", std::to_string(sharedExpertXDim0).c_str(),
                                          ("should equal to bs=" + std::to_string(expertIdsDim0)).c_str()),
                return false);
            OP_TILING_CHECK(
                sharedExpertXDim1 != expandXDim1,
                OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertX(dim1)", std::to_string(sharedExpertXDim1).c_str(),
                                          ("should equal to h=" + std::to_string(expandXDim1)).c_str()),
                return false);
        } else {
            int64_t sharedExpertXDim2 = sharedExpertXShape->GetStorageShape().GetDim(TWO_DIMS);
            OP_TILING_CHECK(sharedExpertXDim0 * sharedExpertXDim1 != expertIdsDim0,
                            OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertX(dim0*dim1)",
                                                      std::to_string(sharedExpertXDim0 * sharedExpertXDim1).c_str(),
                                                      ("should equal to bs=" + std::to_string(expertIdsDim0)).c_str()),
                            return false);
            OP_TILING_CHECK(
                sharedExpertXDim2 != expandXDim1,
                OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertX(dim2)", std::to_string(sharedExpertXDim2).c_str(),
                                          ("should equal to h=" + std::to_string(expandXDim1)).c_str()),
                return false);
        }
    }

    return true;
}

static bool CheckFromDispatchTensorShape(const gert::TilingContext *context,
                                         MoeDistributeCombineV2TilingData &tilingData, const char *nodeName,
                                         const CombineV2Config &config, const uint32_t A, const bool isLayered)
{
    // 校验expandX的维度并设h
    const int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.epWorldSize);
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(config.expandXIndex);
    int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < static_cast<int64_t>(A),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "expandX(dim0)", std::to_string(expandXDim0).c_str(),
                                              ("should be >= A, A=" + std::to_string(A)).c_str()),
                    return false);
    int64_t hMax = isLayered ? H_MAX_LAYERED : H_MAX;
    int64_t hMin = isLayered ? 0 : H_MIN;
    OP_TILING_CHECK(
        (isLayered && (expandXDim1 % H_BASIC_BLOCK_LAYERED != 0)),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "expandX(H)", std::to_string(expandXDim1).c_str(),
                                  ("should be " + std::to_string(H_BASIC_BLOCK_LAYERED) + "-aligned").c_str()),
        return false);
    OP_TILING_CHECK((expandXDim1 < hMin) || (expandXDim1 > hMax),
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "expandX(H)", std::to_string(expandXDim1).c_str(),
                        ("should be in [" + std::to_string(hMin) + ", " + std::to_string(hMax) + "]").c_str()),
                    return false); // 32对齐
    tilingData.moeDistributeCombineV2Info.h = static_cast<uint32_t>(expandXDim1);
    if (expandXDim1 != 0) {
        tilingData.moeDistributeCombineV2Info.armAvgFactor = 1.0f / expandXDim1;
    }

    // 校验assistInfo的维度
    const gert::StorageShape *assistInfoStorageShape = context->GetInputShape(config.assistInfoIndex);
    int64_t assistInfoDim0 = assistInfoStorageShape->GetStorageShape().GetDim(0);
    int64_t minAssistInfoDim0 = static_cast<int64_t>(A * ASSIST_NUM_PER_A);
    OP_TILING_CHECK(
        assistInfoDim0 < minAssistInfoDim0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "assistInfoForCombine(dim0)", std::to_string(assistInfoDim0).c_str(),
                                  ("should be >= minAssistInfoDim0=" + std::to_string(minAssistInfoDim0)).c_str()),
        return false);

    // 校验expandScales的维度
    if (isLayered) {
        const gert::StorageShape *expandScaleStorageShape = context->GetOptionalInputShape(EXPAND_SCALES_INDEX);
        int64_t expandScalesDim0 = expandScaleStorageShape->GetStorageShape().GetDim(0);
        int64_t minexpandScalesDim0 = static_cast<int64_t>(A);
        OP_TILING_CHECK(expandScalesDim0 < minexpandScalesDim0,
                        OP_LOGE_FOR_INVALID_VALUE(
                            nodeName, "expandScales(dim0)", std::to_string(expandScalesDim0).c_str(),
                            ("should be >= minexpandScalesDim0=" + std::to_string(minexpandScalesDim0)).c_str()),
                        return false);
    }

    return true;
}

static uint32_t CalA(MoeDistributeCombineV2TilingData &tilingData, const bool isShared,
                     const uint32_t localMoeExpertNum, const bool hasElasticInfo, const int64_t expertIdsDim1)
{
    uint32_t A = 0U;
    uint32_t rankNumPerSharedExpert = 0U;
    uint32_t maxSharedGroupNum = 0U;
    uint32_t globalBs = tilingData.moeDistributeCombineV2Info.globalBs;
    uint32_t sharedExpertNum = tilingData.moeDistributeCombineV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineV2Info.sharedExpertRankNum;
    uint32_t epWorldSizeU32 = tilingData.moeDistributeCombineV2Info.epWorldSize;
    uint32_t maxBs = globalBs / epWorldSizeU32;
    if ((sharedExpertNum != 0U) && (sharedExpertRankNum != 0U)) { // 除零保护
        rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
        maxSharedGroupNum = (epWorldSizeU32 + rankNumPerSharedExpert - 1U) / rankNumPerSharedExpert;
    }
    if (isShared) { // 本卡为共享专家
        A = maxBs * maxSharedGroupNum;
    } else { // 本卡为moe专家
        A = globalBs * std::min(static_cast<int64_t>(localMoeExpertNum), expertIdsDim1);
    }
    if (hasElasticInfo) {
        A = std::max(static_cast<int64_t>(maxBs * maxSharedGroupNum),
                     globalBs * std::min(static_cast<int64_t>(localMoeExpertNum), expertIdsDim1));
    }
    tilingData.moeDistributeCombineV2Info.a = A;
    return A;
}

static bool CheckSendCountTensorShape(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
                                      const char *nodeName, const bool isShared, const uint32_t localMoeExpertNum,
                                      const bool hasElasticInfo, const bool expertIdsDim1,
                                      const CombineV2Config &config, const bool isLayered)
{
    // 校验epSendCount的维度
    const int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.epWorldSize);
    uint32_t globalBs = tilingData.moeDistributeCombineV2Info.globalBs;
    int64_t moeExpertPerRankNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.moeExpertPerRankNum);
    const gert::StorageShape *epSendCountStorageShape = context->GetInputShape(config.epSendCountIndex);
    const int64_t epSendCountDim0 = epSendCountStorageShape->GetStorageShape().GetDim(0);
    int64_t localEpSendCountSize = (isShared) ? epWorldSize : epWorldSize * moeExpertPerRankNum;

    if (hasElasticInfo) {
        localEpSendCountSize = std::max(epWorldSize, epWorldSize * moeExpertPerRankNum);
    }
    if (isLayered) {
        // 如果是分层方案，则需要校验全新的shape，额外的globalBs * 2 * k * epWorldSize / 8
        // 用来存储token的cnt信息与offset信息，为了兼容A2&A5 这里取/8。
        localEpSendCountSize =
            epWorldSize * localMoeExpertNum + globalBs * 2 * expertIdsDim1 * epWorldSize / RANK_NUM_PER_NODE_A2;
    }
    OP_TILING_CHECK(
        epSendCountDim0 < localEpSendCountSize,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "ep_send_counts", Ops::Base::ToString(epSendCountStorageShape->GetStorageShape()).c_str(),
            (std::string("dim 0 must be at least ") + std::to_string(localEpSendCountSize)).c_str()),
        return false);
    return true;
}

static bool CheckTensorShape(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
                             const char *nodeName, bool isShared, bool isActiveMask, uint32_t localMoeExpertNum,
                             const bool hasElasticInfo, const bool isPerformance, const bool hasExpertScales,
                             const CombineV2Config &config, bool isLayered)
{
    // 校验输入expertIds的维度1并设k, bs已校验过
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(config.expertIdsIndex);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);

    const int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.epWorldSize);
    if (hasElasticInfo) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
        const int64_t elasticInfoDim0 = elasticInfoStorageShape->GetStorageShape().GetDim(0);

        OP_TILING_CHECK(
            elasticInfoDim0 != (ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * epWorldSize),
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "elasticInfo(dim0)", std::to_string(elasticInfoDim0).c_str(),
                                      ("should equal to 4 + 2 * epWorldSize=" +
                                       std::to_string(ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * epWorldSize))
                                          .c_str()),
            return false);
    }
    uint32_t A = CalA(tilingData, isShared, localMoeExpertNum, hasElasticInfo, expertIdsDim1);

    if (isPerformance) {
        const gert::StorageShape *performanceInfoStorageShape =
            context->GetOptionalInputShape(config.performanceInfoIndex);
        const int64_t performanceInfoDim0 = performanceInfoStorageShape->GetStorageShape().GetDim(0);

        OP_TILING_CHECK(
            performanceInfoDim0 != epWorldSize,
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "performanceInfo(dim0)", std::to_string(performanceInfoDim0).c_str(),
                                      ("should equal to epWorldSize=" + std::to_string(epWorldSize)).c_str()),
            return false);
    }

    // 校验activeMask的维度
    if (isActiveMask) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
        int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(
            xActiveMaskDim0 != expertIdsDim0,
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "xActiveMask(dim0)", std::to_string(xActiveMaskDim0).c_str(),
                                      ("should equal to expertIds dim0=" + std::to_string(expertIdsDim0)).c_str()),
            return false);
        OP_TILING_CHECK(
            ((xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) &&
             (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != expertIdsDim1)),
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "xActiveMask(dim1)",
                                      std::to_string(xActiveMaskStorageShape->GetStorageShape().GetDim(1)).c_str(),
                                      ("should equal to expertIds dim1=" + std::to_string(expertIdsDim1)).c_str()),
            return false);
    }

    OP_TILING_CHECK(!CheckZeroComputeExpertsTensorShape(context, tilingData, nodeName, config, expertIdsDim0),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Zero_compute_experts' param dim check failed."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        !CheckExpertsTensorShape(context, tilingData, nodeName, config, expertIdsDim0, expertIdsDim1, hasExpertScales),
        OP_LOGE_WITHOUT_REPORT(nodeName, "Experts' param dim check failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!CheckFromDispatchTensorShape(context, tilingData, nodeName, config, A, isLayered),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "From dispatch's param dim check failed."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!CheckSendCountTensorShape(context, tilingData, nodeName, isShared, localMoeExpertNum,
                                               hasElasticInfo, expertIdsDim1, config, isLayered),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "SendCount of dispatch's param dim check failed."),
                    return ge::GRAPH_FAILED);
    return true;
}

static ge::graphStatus CheckMc2Context(gert::TilingContext *context, const char *nodeName,
                                       const CombineV2Config &config)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "contextShape"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "contextShape", std::to_string(contextStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of contextShape must be 1D."),
        return ge::GRAPH_FAILED);
    int64_t contextDim0 = contextStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "context dim0 = %ld", contextDim0);

    auto contextDesc = context->GetInputDesc(config.contextIndex);
    OP_TILING_CHECK(contextDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "contextDesc"),
                    return ge::GRAPH_FAILED);
    if (contextDesc->GetDataType() != ge::DT_INT32) {
        std::string dtypeStr = Ops::Base::ToString(contextDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "context", dtypeStr.c_str(), "INT32");
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "context",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat()))).c_str(),
            "FRACTAL_NZ"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool CheckSharedAttrs(const char *nodeName, const MoeDistributeCombineV2TilingData &tilingData)
{
    uint32_t sharedExpertNum = tilingData.moeDistributeCombineV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineV2Info.sharedExpertRankNum;

    // 校验共享专家卡数和共享专家数是否只有一个为0
    OP_TILING_CHECK(
        (sharedExpertNum == 0U) && (sharedExpertRankNum > 0U),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertRankNum", std::to_string(sharedExpertRankNum).c_str(),
                                  "should be 0 when sharedExpertNum is 0"),
        return false);
    OP_TILING_CHECK((sharedExpertNum > 0U) && (sharedExpertRankNum == 0U),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertNum", std::to_string(sharedExpertNum).c_str(),
                                              "should be 0 when sharedExpertRankNum is 0"),
                    return false);

    if ((sharedExpertNum > 0U) && (sharedExpertRankNum > 0U)) {
        // 校验共享专家卡数能否整除共享专家数
        OP_TILING_CHECK(
            ((sharedExpertRankNum % sharedExpertNum) != 0U),
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertRankNum", std::to_string(sharedExpertRankNum).c_str(),
                                      "should be divisible by sharedExpertNum"),
            return false);
    }

    return true;
}

static bool CheckCommAlgAttrs(const char *nodeName, const MoeDistributeCombineV2TilingData &tilingData, bool isLayered)
{
    bool hasElasticInfo = tilingData.moeDistributeCombineV2Info.hasElasticInfo;
    uint32_t zeroExpertNum = tilingData.moeDistributeCombineV2Info.zeroExpertNum;
    uint32_t copyExpertNum = tilingData.moeDistributeCombineV2Info.copyExpertNum;
    uint32_t constExpertNum = tilingData.moeDistributeCombineV2Info.constExpertNum;
    int32_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;
    bool isExpertMask = tilingData.moeDistributeCombineV2Info.isExpertMask;
    bool isPerformance = tilingData.moeDistributeCombineV2Info.isPerformance;

    // 校验动态缩容和分层不能同时启用
    OP_TILING_CHECK(
        (isLayered && hasElasticInfo),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "elasticInfo", "present", "not supported when comm_alg = hierarchy"),
        return false);
    // 校验特殊专家和分层不能同时启用
    OP_TILING_CHECK(
        (isLayered && (zeroComputeExpertNum > 0)),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "zeroComputeExpert", "present", "not supported when comm_alg = hierarchy"),
        return false);
    // 校验二维Mask和分层不能同时启用
    OP_TILING_CHECK((isLayered && isExpertMask),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "xActiveMask", "2D", "not supported when comm_alg = hierarchy"),
                    return false);
    // 校验isPerformance和分层不能同时启用
    OP_TILING_CHECK(
        (isLayered && isPerformance),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "isPerformance", "true", "not supported when comm_alg = hierarchy"),
        return false);

    return true;
}

static bool CheckZeroComputeExpert(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
                                   const char *nodeName, const CombineV2Config &config)
{
    uint32_t copyExpertNum = tilingData.moeDistributeCombineV2Info.copyExpertNum;
    uint32_t constExpertNum = tilingData.moeDistributeCombineV2Info.constExpertNum;

    const gert::StorageShape *oriXStorageShape = context->GetOptionalInputShape(config.oriXIndex);
    const gert::StorageShape *constExpertAlpha1StorageShape =
        context->GetOptionalInputShape(config.constExpertAlpha1Index);
    const gert::StorageShape *constExpertAlpha2StorageShape =
        context->GetOptionalInputShape(config.constExpertAlpha2Index);
    const gert::StorageShape *constExpertVStorageShape = context->GetOptionalInputShape(config.constExpertVIndex);

    OP_TILING_CHECK(copyExpertNum > 0 && oriXStorageShape == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "oriX", "missing", "should exist when copyExpertNum > 0"),
                    return false);
    OP_TILING_CHECK(
        constExpertNum > 0 && (oriXStorageShape == nullptr || constExpertAlpha1StorageShape == nullptr ||
                               constExpertAlpha2StorageShape == nullptr || constExpertVStorageShape == nullptr),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "const_expert_inputs", "missing",
                                              "oriX, alpha1, alpha2, V must exist when constExpertNum > 0"),
        return false);
    return true;
}

static bool CheckAttrs(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
                       const char *nodeName, uint32_t &localMoeExpertNum, bool isActiveMask,
                       const CombineV2Config &config, bool isLayered)
{
    uint32_t epWorldSize = tilingData.moeDistributeCombineV2Info.epWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeCombineV2Info.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineV2Info.sharedExpertRankNum;

    OP_TILING_CHECK(!CheckSharedAttrs(nodeName, tilingData),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Check shared expert related attributes failed."), return false);
    OP_TILING_CHECK(!CheckCommAlgAttrs(nodeName, tilingData, isLayered),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Check comm_alg related attributes failed."), return false);
    // 校验moe专家数量能否均分给多机
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "moeExpertNum", std::to_string(moeExpertNum).c_str(),
                                              "should be divisible by (epWorldSize - sharedExpertRankNum)"),
                    return false);
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK((localMoeExpertNum <= 0) || (localMoeExpertNum * epWorldSize > LOCAL_EXPERT_MAX_SIZE),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "localMoeExpertNum", std::to_string(localMoeExpertNum).c_str(),
                                              "must be > 0 and localMoeExpertNum * epWorldSize <= 2048"),
                    return false);
    tilingData.moeDistributeCombineV2Info.moeExpertPerRankNum = localMoeExpertNum;

    // 校验输入expertIds的维度0并设bs
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(config.expertIdsIndex);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t bsUpperBound = isLayered ? BS_UPPER_BOUND_LAYERED : BS_UPPER_BOUND;
    OP_TILING_CHECK((expertIdsDim0 <= 0) || (expertIdsDim0 > bsUpperBound),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "expertIds(BS)", std::to_string(expertIdsDim0).c_str(),
                                              "in range [1, bsUpperBound]"),
                    return false);
    tilingData.moeDistributeCombineV2Info.bs = static_cast<uint32_t>(expertIdsDim0);

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return false);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>((config.attrGlobalBsIndex));
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "globalBs"), return false);
    OP_LOGD(nodeName, "MoeDistributeCombineV2 *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n", *globalBsPtr,
            expertIdsDim0, epWorldSize);

    OP_TILING_CHECK((*globalBsPtr != 0) && ((*globalBsPtr < static_cast<int64_t>(epWorldSize) * expertIdsDim0) ||
                                            ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "globalBS", std::to_string(*globalBsPtr).c_str(),
                                              "should be 0 or configured maxBs per rank * epWorldSize"),
                    return false);
    OP_TILING_CHECK(((*globalBsPtr > (expertIdsDim0 * static_cast<int64_t>(epWorldSize))) && isActiveMask),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "globalBS", std::to_string(*globalBsPtr).c_str(),
                                              "not > bs * epWorldSize when isActiveMask"),
                    return false);

    tilingData.moeDistributeCombineV2Info.globalBs = static_cast<uint32_t>(*globalBsPtr);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeCombineV2Info.globalBs = static_cast<uint32_t>(expertIdsDim0) * epWorldSize;
    }
    OP_TILING_CHECK(!CheckZeroComputeExpert(context, tilingData, nodeName, config),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Zero compute expert check failed."), return ge::GRAPH_FAILED);
    return true;
}

static ge::graphStatus TilingCheckMoeDistributeCombine(gert::TilingContext *context, const char *nodeName,
                                                       const bool isActiveMask, const bool hasElasticInfo,
                                                       const bool isPerformance, const bool hasExpertScales,
                                                       const CombineV2Config &config, const bool isLayered)
{
    // 检查参数shape信息
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isActiveMask, hasExpertScales, hasElasticInfo, isPerformance,
                                    config, isLayered),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "param shape is invalid"), return ge::GRAPH_FAILED);
    // 检查参数dataType信息
    OP_TILING_CHECK(!CheckExpertTensorDataType(context, nodeName, config, hasExpertScales),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Experts param dataType is invalid"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, isActiveMask, hasElasticInfo, isPerformance, config),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "param dataType is invalid"), return ge::GRAPH_FAILED);
    // 检查参数format信息
    OP_TILING_CHECK(
        !CheckTensorFormat(context, nodeName, isActiveMask, hasExpertScales, hasElasticInfo, isPerformance, config),
        OP_LOGE_WITHOUT_REPORT(nodeName, "param Format is invalid"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext *context, const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE_WITHOUT_REPORT(nodeName, "get workspace failed"),
                    return ge::GRAPH_FAILED);
    workspace[0] = SYSTEM_NEED_WORKSPACE + aivNum * MASK_CALC_NEED_WORKSPACE;
    OP_LOGD(nodeName, "workspace[0] size is %ld", workspace[0]);
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeDistributeCombineV2TilingFuncBase::CalTilingKey(uint32_t commQuantMode, bool isLayered)
{
    uint32_t quantMode = TILINGKEY_NO_QUANT;
    uint32_t hierarchy = TILINGKEY_TPL_MTE;
    if (commQuantMode >= INT8_COMM_QUANT) {
        quantMode = commQuantMode;
    }
    if (isLayered) {
        hierarchy = TILINGKEY_TPL_HIERARCHY;
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(quantMode, hierarchy, TILINGKEY_TPL_A3);
    return tilingKey;
}

static ge::graphStatus SetHCommCfg(const gert::TilingContext *context, MoeDistributeCombineV2TilingData *tiling,
                                   const std::string groupEp, bool isLayered)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeCombineV2 groupEp = %s", groupEp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr =
        isLayered ? "AlltoAll=level1:hierarchy" : "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE); // 通过不拉起AICPU，提高算子退出性能

    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling) != 0,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1) != 0,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "mc2CcTilingConfig mc2tiling1 GetTiling mc2CcTiling1 failed"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static uint32_t GetPartialBufferSizeByFlag(const uint32_t axisBS, const uint32_t axisK, const bool isInputTokenMaskFlag,
                                           const bool isInputExpertMaskFlag)
{
    uint32_t partialBufferSize = 0;

    if (isInputTokenMaskFlag) {
        uint32_t axisBsAlignSize = (axisBS * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
        partialBufferSize += axisBsAlignSize + axisBsAlignSize * DTYPE_SIZE_HALF * BUFFER_NUM;
    }
    if (isInputExpertMaskFlag) {
        partialBufferSize += (axisBS * DTYPE_SIZE_HALF + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN +
                             (axisBS * axisK * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    }
    return partialBufferSize;
}

static uint32_t GetA3PartialBufferSizeByFlag(const uint32_t axisBS, const uint32_t axisK,
                                             const bool isInputTokenMaskFlag, const bool isInputExpertMaskFlag,
                                             const bool enableSpecialExpert)
{
    uint32_t partialBufferSize = 0U;
    if (isInputTokenMaskFlag) {
        uint32_t axisBsAlignSize = (axisBS * sizeof(bool) + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN;
        partialBufferSize += axisBsAlignSize + axisBsAlignSize * DTYPE_SIZE_HALF * BUFFER_NUM;
    }
    if (isInputExpertMaskFlag) {
        partialBufferSize += (axisBS * DTYPE_SIZE_HALF + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN +
                             (axisBS * sizeof(int32_t) + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN +
                             (axisBS * axisK * sizeof(bool) + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN;
    }
    if (enableSpecialExpert && !isInputExpertMaskFlag) {
        partialBufferSize += (axisBS * DTYPE_SIZE_HALF + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN;
    }
    return partialBufferSize;
}

static uint32_t GetFloatAlignSize(uint32_t hFloatSize, uint32_t alignSize, bool isA5FusedQuant)
{
    // A5 + INT8/MXFP8 场景按512B对齐，其余场景使用调用方指定的对齐大小
    if (isA5FusedQuant) {
        alignSize = ALIGNED_LEN * 2U;
    }
    return (hFloatSize + alignSize - 1U) / alignSize * alignSize;
}

static uint64_t GetComputeBufferSize(const gert::TilingContext *context,
                                     const MoeDistributeCombineV2TilingData *tilingData, const CombineV2Config &config,
                                     uint32_t bufferNum)
{
    uint32_t axisH = tilingData->moeDistributeCombineV2Info.h;
    uint32_t axisBS = tilingData->moeDistributeCombineV2Info.bs;
    uint32_t axisK = tilingData->moeDistributeCombineV2Info.k;
    bool isInputExpertMaskFlag = tilingData->moeDistributeCombineV2Info.isExpertMask;
    bool isInputTokenMaskFlag = tilingData->moeDistributeCombineV2Info.isTokenMask;
    auto attrs = context->GetAttrs();
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCommQuantModeIndex));
    uint32_t commQuantMode = commQuantModePtr == nullptr ? 0U : static_cast<uint32_t>(*commQuantModePtr);
    bool isInt8Quant = commQuantMode == INT8_COMM_QUANT;
    bool isMxFp8Quant = (commQuantMode == static_cast<CommQuantModeType>(CommQuantMode::MXFP8_E5M2_QUANT)) ||
                        (commQuantMode == static_cast<CommQuantModeType>(CommQuantMode::MXFP8_E4M3_QUANT));
    bool isA5FusedQuant = mc2tiling::GetNpuArch(context) == NpuArch::DAV_3510 && (isInt8Quant || isMxFp8Quant);
    auto expandXDesc = context->GetInputDesc(config.expandXIndex);
    uint32_t expandXTypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(expandXDesc->GetDataType()));
    uint32_t hExpandXTypeSize = axisH * expandXTypeSize;
    uint32_t hFloatSize = axisH * static_cast<uint32_t>(sizeof(float));
    uint32_t maxSizeTokenBuf = (hExpandXTypeSize + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t activeMaskAlignSize = axisBS * ((axisK * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
    uint32_t hExpandXAlign32Size = (hExpandXTypeSize + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t hFloatAlign32Size = GetFloatAlignSize(hFloatSize, UB_ALIGN, isA5FusedQuant);
    uint32_t hFloatAlign256Size = GetFloatAlignSize(hFloatSize, ALIGNED_LEN, isA5FusedQuant);
    uint32_t bsKFloatAlign = (axisBS * axisK * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t mulBufSize = hFloatAlign256Size > bsKFloatAlign ? hFloatAlign256Size : bsKFloatAlign;
    uint32_t maxSizeRowTmpFloatBuf = hFloatAlign32Size;

    if (isInputExpertMaskFlag) {
        uint32_t activeMaskAlignHalfSize = activeMaskAlignSize * DTYPE_SIZE_HALF;
        maxSizeTokenBuf = (activeMaskAlignSize > hExpandXAlign32Size ? activeMaskAlignSize : hExpandXAlign32Size);
        maxSizeRowTmpFloatBuf =
            (activeMaskAlignHalfSize > hFloatAlign32Size ? activeMaskAlignHalfSize : hFloatAlign32Size);
    }

    uint64_t totalBufferSize = static_cast<uint64_t>(maxSizeTokenBuf) + maxSizeRowTmpFloatBuf + mulBufSize +
                               hFloatAlign32Size * bufferNum + UB_ALIGN;
    if (config.hasAddRmsNorm) {
        totalBufferSize += hExpandXAlign32Size + NUM_PER_REP_FP32 * sizeof(float) * BUFFER_NUM;
    }
    totalBufferSize += GetPartialBufferSizeByFlag(axisBS, axisK, isInputTokenMaskFlag, isInputExpertMaskFlag);
    return totalBufferSize;
}

static uint32_t GetQuantCommDataBytes(uint32_t axisH, uint32_t expandXTypeSize, uint32_t scaleBaseCnt,
                                      uint32_t commQuantMode, uint32_t &scaleNumAlignSize)
{
    if (commQuantMode == INT8_COMM_QUANT) {
        uint32_t scaleNum = scaleBaseCnt / static_cast<uint32_t>(UB_ALIGN / sizeof(float));
        scaleNumAlignSize = (scaleNum * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
        uint32_t hAlign32Size = (axisH + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
        return hAlign32Size + scaleNum * expandXTypeSize;
    }
    uint32_t scaleNum = (axisH + UB_ALIGN - 1) / UB_ALIGN;
    scaleNum = (scaleNum + 1U) / 2U * 2U;
    scaleNumAlignSize = (scaleNum + 127U) / 128U * 128U * expandXTypeSize * BUFFER_NUM;
    return (axisH + ALIGNED_LEN - 1) / ALIGNED_LEN * ALIGNED_LEN + scaleNum * expandXTypeSize;
}

static uint32_t GetA3QuantBufferSize(const gert::TilingContext *context, const CombineV2Config &config,
                                     uint32_t hExpandXAlign32Size)
{
    auto expandXDesc = context->GetInputDesc(config.expandXIndex);
    uint32_t expandXTypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(expandXDesc->GetDataType()));
    auto commQuantModePtr = context->GetAttrs()->GetAttrPointer<int64_t>((config.attrCommQuantModeIndex));
    if (*commQuantModePtr == INT8_COMM_QUANT) {
        uint32_t scaleBaseCnt = hExpandXAlign32Size / expandXTypeSize;
        uint32_t scaleNum = scaleBaseCnt / static_cast<uint32_t>(UB_ALIGN / sizeof(float));
        return (scaleNum * sizeof(float) + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN;
    }
    bool isMxFp8Quant = (*commQuantModePtr == static_cast<CommQuantModeType>(CommQuantMode::MXFP8_E5M2_QUANT)) ||
                        (*commQuantModePtr == static_cast<CommQuantModeType>(CommQuantMode::MXFP8_E4M3_QUANT));
    if (isMxFp8Quant) {
        uint32_t scaleNum = (hExpandXAlign32Size / expandXTypeSize + UB_ALIGN - 1U) / UB_ALIGN;
        return (scaleNum * expandXTypeSize + ALIGNED_LEN - 1U) / ALIGNED_LEN * ALIGNED_LEN * BUFFER_NUM;
    }
    return 0U;
}

static uint64_t GetA3ComputeBufferSize(const gert::TilingContext *context,
                                       const MoeDistributeCombineV2TilingData *tilingData,
                                       const CombineV2Config &config)
{
    uint32_t axisH = tilingData->moeDistributeCombineV2Info.h;
    uint32_t axisBS = tilingData->moeDistributeCombineV2Info.bs;
    uint32_t axisK = tilingData->moeDistributeCombineV2Info.k;
    bool isInputExpertMaskFlag = tilingData->moeDistributeCombineV2Info.isExpertMask;
    bool isInputTokenMaskFlag = tilingData->moeDistributeCombineV2Info.isTokenMask;
    uint32_t specialExpertNum = tilingData->moeDistributeCombineV2Info.zeroExpertNum;
    specialExpertNum += tilingData->moeDistributeCombineV2Info.copyExpertNum;
    specialExpertNum += tilingData->moeDistributeCombineV2Info.constExpertNum;
    bool enableSpecialExpert = specialExpertNum > 0U;
    auto expandXDesc = context->GetInputDesc(config.expandXIndex);
    uint32_t expandXTypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(expandXDesc->GetDataType()));
    uint32_t hExpandXTypeSize = axisH * expandXTypeSize;
    uint32_t hExpandXAlign32Size = (hExpandXTypeSize + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN;
    uint32_t activeMaskAlignSize = axisBS * ((axisK * sizeof(bool) + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN);
    uint32_t hFloatSize = axisH * static_cast<uint32_t>(sizeof(float));
    uint32_t hFloatAlign32Size = (hFloatSize + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN;
    uint32_t hFloatAlign256Size = (hFloatSize + ALIGNED_LEN - 1U) / ALIGNED_LEN * ALIGNED_LEN;
    uint32_t bsKFloatAlign = (axisBS * axisK * sizeof(float) + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN;
    uint32_t mulBufSize = hFloatAlign256Size > bsKFloatAlign ? hFloatAlign256Size : bsKFloatAlign;
    uint32_t maxSizeTokenBuf = hExpandXAlign32Size;
    uint32_t maxSizeRowTmpFloatBuf = hFloatAlign32Size;
    if (isInputExpertMaskFlag || enableSpecialExpert) {
        uint32_t activeMaskAlignHalfSize = activeMaskAlignSize * DTYPE_SIZE_HALF;
        maxSizeTokenBuf = activeMaskAlignSize > hExpandXAlign32Size ? activeMaskAlignSize : hExpandXAlign32Size;
        maxSizeRowTmpFloatBuf =
            activeMaskAlignHalfSize > hFloatAlign32Size ? activeMaskAlignHalfSize : hFloatAlign32Size;
    }
    uint32_t flagRcvCount = axisK + tilingData->moeDistributeCombineV2Info.sharedExpertNum;
    uint64_t totalBufferSize = maxSizeTokenBuf + maxSizeRowTmpFloatBuf + mulBufSize + hFloatAlign32Size +
                               hExpandXAlign32Size * BUFFER_NUM + flagRcvCount * STATE_OFFSET * BUFFER_NUM + UB_ALIGN;
    if (config.hasAddRmsNorm) {
        totalBufferSize += hExpandXAlign32Size + NUM_PER_REP_FP32 * sizeof(float) * BUFFER_NUM;
    }
    totalBufferSize += GetA3QuantBufferSize(context, config, hExpandXAlign32Size);
    totalBufferSize +=
        GetA3PartialBufferSizeByFlag(axisBS, axisK, isInputTokenMaskFlag, isInputExpertMaskFlag, enableSpecialExpert);
    return totalBufferSize;
}

static uint64_t GetDispatchBufferSize(const MoeDistributeCombineV2TilingData *tilingData, uint32_t moeQueueBytes,
                                      uint32_t commDataBytes, uint32_t packedDataBytes, uint64_t quantBufferSize)
{
    uint32_t blockCntPerToken = packedDataBytes / SPLIT_BLOCK_DATA_SIZE;
    // calBeginBuf、fixedEndBuf、最小indexCountsBuf和calEndBuf各占一个UB block
    uint64_t dispatchBufferSize = static_cast<uint64_t>(UB_ALIGN) * 4UL;
    if (quantBufferSize > 0U) {
        uint32_t quantResultBufSize = (commDataBytes + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN;
        quantResultBufSize = std::max(quantResultBufSize, packedDataBytes);
        dispatchBufferSize += quantBufferSize + quantResultBufSize + blockCntPerToken * WIN_ADDR_ALIGN;
    } else {
        dispatchBufferSize += static_cast<uint64_t>(moeQueueBytes) + blockCntPerToken * WIN_ADDR_ALIGN;
    }
    if (tilingData->moeDistributeCombineV2Info.hasElasticInfo) {
        uint32_t elasticInfoSize = (static_cast<uint32_t>(ELASTIC_METAINFO_OFFSET) +
                                    RANK_LIST_NUM * tilingData->moeDistributeCombineV2Info.epWorldSize) *
                                   sizeof(int32_t);
        dispatchBufferSize += (elasticInfoSize + UB_ALIGN - 1U) / UB_ALIGN * UB_ALIGN;
    }
    return dispatchBufferSize;
}

static uint64_t GetQuantScratchSize(uint32_t axisH, uint32_t expandXTypeSize, bool isMxFp8Quant,
                                    uint32_t hFloatAlign32Size, uint32_t hFloatAlign256Size)
{
    uint32_t hExpandXAlignSize = isMxFp8Quant ? (axisH + 127U) / 128U * 128U * expandXTypeSize :
                                                (axisH * expandXTypeSize + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    return static_cast<uint64_t>(hFloatAlign256Size) * 2UL + hFloatAlign256Size / (UB_ALIGN / sizeof(float)) +
           hFloatAlign32Size + hExpandXAlignSize;
}

static uint64_t GetCommBufferSize(const gert::TilingContext *context,
                                  const MoeDistributeCombineV2TilingData *tilingData, const CombineV2Config &config,
                                  uint32_t &moeQueueBytes, uint64_t &dispatchBufferSize)
{
    uint32_t axisH = tilingData->moeDistributeCombineV2Info.h;
    uint32_t flagRcvCount =
        tilingData->moeDistributeCombineV2Info.k + tilingData->moeDistributeCombineV2Info.sharedExpertNum;
    auto attrs = context->GetAttrs();
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCommQuantModeIndex));
    uint32_t commQuantMode = commQuantModePtr == nullptr ? 0U : static_cast<uint32_t>(*commQuantModePtr);
    bool isInt8Quant = commQuantMode == INT8_COMM_QUANT;
    bool isMxFp8Quant = (commQuantMode == static_cast<CommQuantModeType>(CommQuantMode::MXFP8_E5M2_QUANT)) ||
                        (commQuantMode == static_cast<CommQuantModeType>(CommQuantMode::MXFP8_E4M3_QUANT));
    bool isA5 = mc2tiling::GetNpuArch(context) == NpuArch::DAV_3510;
    auto expandXDesc = context->GetInputDesc(config.expandXIndex);
    uint32_t expandXTypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(expandXDesc->GetDataType()));
    uint32_t hExpandXTypeSize = axisH * expandXTypeSize;
    uint32_t hExpandXAlign32Size = (hExpandXTypeSize + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t hFloatSize = axisH * static_cast<uint32_t>(sizeof(float));
    uint32_t hFloatAlign32Size = GetFloatAlignSize(hFloatSize, UB_ALIGN, isA5 && (isInt8Quant || isMxFp8Quant));
    uint32_t hFloatAlign256Size = GetFloatAlignSize(hFloatSize, ALIGNED_LEN, isA5 && (isInt8Quant || isMxFp8Quant));
    uint32_t scaleNumAlignSize = 0U;
    uint32_t commDataBytes = hExpandXTypeSize;
    uint64_t quantBufferSize = 0U;
    if (isInt8Quant || isMxFp8Quant) {
        uint32_t scaleBaseCnt =
            isInt8Quant ? (isA5 ? hFloatAlign256Size / sizeof(float) : hExpandXAlign32Size / expandXTypeSize) : 0U;
        commDataBytes = GetQuantCommDataBytes(axisH, expandXTypeSize, scaleBaseCnt, commQuantMode, scaleNumAlignSize);
        quantBufferSize =
            GetQuantScratchSize(axisH, expandXTypeSize, isMxFp8Quant, hFloatAlign32Size, hFloatAlign256Size);
    }
    uint32_t blockCntPerToken = (commDataBytes + SPLIT_BLOCK_DATA_SIZE - 1) / SPLIT_BLOCK_DATA_SIZE;
    uint32_t packedDataBytes = blockCntPerToken * SPLIT_BLOCK_DATA_SIZE;
    uint32_t rawPackedDataBytes =
        (hExpandXTypeSize + SPLIT_BLOCK_DATA_SIZE - 1) / SPLIT_BLOCK_DATA_SIZE * SPLIT_BLOCK_DATA_SIZE;
    moeQueueBytes = packedDataBytes > rawPackedDataBytes ? packedDataBytes : rawPackedDataBytes;
    moeQueueBytes = moeQueueBytes > hExpandXAlign32Size ? moeQueueBytes : hExpandXAlign32Size;

    // flag检查和比较复用计算scratch，通信侧只保留清零buffer
    uint32_t clearFlagCount = flagRcvCount * blockCntPerToken * SPLIT_BLOCK_FLAG_COUNT;
    uint32_t clearFlagBufSize = (clearFlagCount * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint64_t commBufferSize = static_cast<uint64_t>(scaleNumAlignSize) + clearFlagBufSize;
    dispatchBufferSize =
        isA5 ? GetDispatchBufferSize(tilingData, moeQueueBytes, commDataBytes, packedDataBytes, quantBufferSize) : 0U;
    return commBufferSize;
}

static uint64_t GetReceiveBufferSize(const MoeDistributeCombineV2TilingData *tilingData)
{
    uint32_t axisBS = tilingData->moeDistributeCombineV2Info.bs;
    uint32_t axisK = tilingData->moeDistributeCombineV2Info.k;
    uint32_t flagRcvCount = axisK + tilingData->moeDistributeCombineV2Info.sharedExpertNum;
    uint32_t maxBs =
        tilingData->moeDistributeCombineV2Info.globalBs / tilingData->moeDistributeCombineV2Info.epWorldSize;
    uint32_t recvAivNum = std::min(maxBs, std::min(32U, tilingData->moeDistributeCombineV2Info.aivNum / 2U));
    recvAivNum = recvAivNum == 0U ? 1U : recvAivNum;
    uint32_t tokenPerAivNum = (axisBS + recvAivNum - 1U) / recvAivNum;
    uint64_t totalBufferSize = 0U;
    if (tilingData->moeDistributeCombineV2Info.hasExpertScales) {
        totalBufferSize += (tokenPerAivNum * axisK * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    }
    if (tilingData->moeDistributeCombineV2Info.isPerformance) {
        totalBufferSize +=
            (tilingData->moeDistributeCombineV2Info.epWorldSize * sizeof(int64_t) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
        totalBufferSize += (tokenPerAivNum * flagRcvCount * sizeof(int32_t) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    }
    return totalBufferSize;
}

static void SetA3BufferNum(const uint64_t ubSize, const gert::TilingContext *context,
                           MoeDistributeCombineV2TilingData *tilingData, const CombineV2Config &config)
{
    uint64_t totalBufferSize = GetA3ComputeBufferSize(context, tilingData, config);
    tilingData->moeDistributeCombineV2Info.bufferNum = totalBufferSize > ubSize ? BUFFER_SINGLE : BUFFER_NUM;
}

static bool UbUsedCal(const uint64_t ubSize, const gert::TilingContext *context,
                      MoeDistributeCombineV2TilingData *tilingData, const CombineV2Config &config)
{
    if (mc2tiling::GetNpuArch(context) != NpuArch::DAV_3510) {
        SetA3BufferNum(ubSize, context, tilingData, config);
        return true;
    }
    uint32_t moeQueueBytes = 0U;
    uint64_t dispatchBufferSize = 0U;
    uint64_t commBufferSize = GetCommBufferSize(context, tilingData, config, moeQueueBytes, dispatchBufferSize);
    if (dispatchBufferSize > ubSize) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "MoeDistributeCombineV2 dispatch-buffer UB usage exceeds available UB size");
        return false;
    }
    // 单buf路径必须先通过UB检查，不能依赖运行时分配器兜底
    uint64_t singleBufferSize = GetComputeBufferSize(context, tilingData, config, BUFFER_SINGLE) + commBufferSize +
                                GetReceiveBufferSize(tilingData) + moeQueueBytes;
    if (singleBufferSize > ubSize) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "MoeDistributeCombineV2 single-buffer UB usage exceeds available UB size");
        return false;
    }
    uint64_t totalBufferSize = GetComputeBufferSize(context, tilingData, config, BUFFER_NUM) + commBufferSize +
                               GetReceiveBufferSize(tilingData);
    tilingData->moeDistributeCombineV2Info.bufferNum =
        totalBufferSize + moeQueueBytes * BUFFER_NUM > ubSize ? BUFFER_SINGLE : BUFFER_NUM;
    return true;
}

static ge::graphStatus CheckAndCalWinSize(const gert::TilingContext *context,
                                          MoeDistributeCombineV2TilingData &tilingData, const char *nodeName,
                                          const bool isSetFullMeshV2, uint32_t localMoeExpertNum, bool isLayered,
                                          const CombineV2Config &config)
{
    CheckWinSizeData winSizeData;
    winSizeData.localMoeExpertNum = localMoeExpertNum;
    winSizeData.sharedExpertNum = tilingData.moeDistributeCombineV2Info.sharedExpertNum;
    winSizeData.a = static_cast<uint64_t>(tilingData.moeDistributeCombineV2Info.a);
    winSizeData.h = static_cast<uint64_t>(tilingData.moeDistributeCombineV2Info.h);
    winSizeData.k = static_cast<uint64_t>(tilingData.moeDistributeCombineV2Info.k);
    winSizeData.bs = static_cast<uint64_t>(tilingData.moeDistributeCombineV2Info.bs);
    winSizeData.epWorldSize = static_cast<uint64_t>(tilingData.moeDistributeCombineV2Info.epWorldSize);
    winSizeData.globalBs = static_cast<uint64_t>(tilingData.moeDistributeCombineV2Info.globalBs);
    winSizeData.moeExpertNum = static_cast<uint64_t>(tilingData.moeDistributeCombineV2Info.moeExpertNum);
    winSizeData.tpWorldSize = static_cast<uint64_t>(tilingData.moeDistributeCombineV2Info.tpWorldSize);
    winSizeData.isSetFullMeshV2 = false;
    winSizeData.isLayered = isLayered;
    winSizeData.isMc2Context = config.isMc2Context;
    OP_TILING_CHECK(CheckWinSize(context, nodeName, winSizeData) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Get WinSize failed."), return ge::GRAPH_FAILED);
    tilingData.moeDistributeCombineV2Info.totalWinSizeEp = winSizeData.totalWinSizeEp;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCombineOrARN(gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
                                         const char *nodeName, const CombineV2Config &config)
{
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(config.expertIdsIndex);
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(config.expandXIndex);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    if (config.hasAddRmsNorm) {
        // 校验combineARN新增的输入与输出
        OP_TILING_CHECK(CheckARNAttrAndSetTilingData(context, tilingData, nodeName, config) != ge::GRAPH_SUCCESS,
                        OP_LOGE_WITHOUT_REPORT(nodeName, "attr check or set tilingdata failed."),
                        return ge::GRAPH_FAILED);
        // 校验combineARN的输入与输出shape
        OP_TILING_CHECK(!CheckTensorShapeARN(context, nodeName, config, expertIdsDim0, expandXDim1),
                        OP_LOGE_WITHOUT_REPORT(nodeName, "CheckTensorShapeARN failed"), return ge::GRAPH_FAILED);
        // 校验combineARN新增的输入维数
        OP_TILING_CHECK(!CheckInputTensorDimARN(context, nodeName, config),
                        OP_LOGE_WITHOUT_REPORT(nodeName, "CheckInputTensorDimARN failed"), return ge::GRAPH_FAILED);
        // 校验combineARN的输出维数
        OP_TILING_CHECK(!CheckOutputTensorDimARN(context, nodeName, config),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "output tensor", "",
                                                              "param shape of output tensor is invalid"),
                        return ge::GRAPH_FAILED);
    } else {
        // 校验combine输出的维数与shape
        OP_TILING_CHECK(!CheckOutputTensorDim(context, nodeName, config, expertIdsDim0, expandXDim1),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "output tensor", "",
                                                              "param shape of output tensor is invalid"),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAndGetOptionalInput(gert::TilingContext *context, MoeDistributeCombineV2TilingData *tilingData,
                                         const CombineV2Config &config, bool &isActiveMask, bool &hasElasticInfo,
                                         bool &isPerformance, bool &hasExpertScales)
{
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
    isActiveMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeCombineV2Info.isTokenMask =
        ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    tilingData->moeDistributeCombineV2Info.isExpertMask =
        ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));

    // 获取elasticInfo
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
    hasElasticInfo = (elasticInfoStorageShape != nullptr);
    tilingData->moeDistributeCombineV2Info.hasElasticInfo = hasElasticInfo;
    // 获取performanceInfo
    const gert::StorageShape *performanceInfoStorageShape = context->GetOptionalInputShape(config.performanceInfoIndex);
    isPerformance = (performanceInfoStorageShape != nullptr);
    tilingData->moeDistributeCombineV2Info.isPerformance = isPerformance;

    const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(config.expertScalesIndex);
    hasExpertScales = IsPresentTensorShape(expertScalesStorageShape);
    tilingData->moeDistributeCombineV2Info.hasExpertScales = hasExpertScales;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckInputParam(gert::TilingContext *context, const char *nodeName, const bool isActiveMask,
                                const bool hasElasticInfo, const bool isPerformance, const bool hasExpertScales,
                                uint32_t &localMoeExpertNum, const bool isLayered, bool &isShared,
                                MoeDistributeCombineV2TilingData *tilingData, const CombineV2Config &config)
{
    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(TilingCheckMoeDistributeCombine(context, nodeName, isActiveMask, hasElasticInfo, isPerformance,
                                                    hasExpertScales, config, isLayered) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Tiling check params failed"), return ge::GRAPH_FAILED);

    // 检查属性的取值是否合法
    OP_TILING_CHECK(!CheckAttrs(context, *tilingData, nodeName, localMoeExpertNum, isActiveMask, config, isLayered),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "attr check failed."), return ge::GRAPH_FAILED);

    uint32_t sharedExpertRankNum = tilingData->moeDistributeCombineV2Info.sharedExpertRankNum;
    uint32_t epRankId = tilingData->moeDistributeCombineV2Info.epRankId;
    isShared = (epRankId < sharedExpertRankNum);

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(!CheckTensorShape(context, *tilingData, nodeName, isShared, isActiveMask, localMoeExpertNum,
                                      hasElasticInfo, isPerformance, hasExpertScales, config, isLayered),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "param dim check failed."), return ge::GRAPH_FAILED);

    // 校验combine或combineARN有差异的参数
    OP_TILING_CHECK(CheckCombineOrARN(context, *tilingData, nodeName, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "CheckCombineOrARN failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAndSetPlatformInfo(gert::TilingContext *context, MoeDistributeCombineV2TilingData *tilingData,
                                        const bool isLayered, const char *nodeName, const CombineV2Config &config)
{
    int32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t epWorldSize = tilingData->moeDistributeCombineV2Info.epWorldSize;
    uint32_t serverNum = epWorldSize / RANK_NUM_PER_NODE;
    OP_TILING_CHECK((isLayered && ((aivNum < RANK_NUM_PER_NODE) || (aivNum <= 2 * serverNum))),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "aivNum", std::to_string(aivNum).c_str(),
                                              "should be > 16 and > 2 * serverNum"),
                    return ge::GRAPH_FAILED);
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData->moeDistributeCombineV2Info.aivNum = aivNum;
    tilingData->moeDistributeCombineV2Info.totalUbSize = ubSize;
    OP_TILING_CHECK(!UbUsedCal(ubSize, context, tilingData, config),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "MoeDistributeCombineV2 UB usage exceeds available UB size"),
                    return ge::GRAPH_FAILED);
    context->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    OP_LOGD(nodeName, "numBlocks = %u, aivNum = %lu, ubsize = %lu", numBlocks, aivNum, ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineV2TilingFuncBase::MoeDistributeCombineA3TilingFuncImpl(
    gert::TilingContext *context, const CombineV2Config &config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter MoeDistributeCombineV2 Tiling func");
    MoeDistributeCombineV2TilingData *tilingData = context->GetTilingData<MoeDistributeCombineV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);
    std::string groupEp = "";
    bool isShared = true;
    bool isLayered = false;
    bool isActiveMask = false;
    bool hasElasticInfo = false;
    bool isPerformance = false;
    bool hasExpertScales = false;
    uint32_t localMoeExpertNum = 1;
    uint32_t commQuantMode = 0U;

    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, *tilingData, nodeName, groupEp, commQuantMode, config,
                                            isLayered) == ge::GRAPH_FAILED,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);
    // 检查并填充可选输入
    OP_TILING_CHECK(CheckAndGetOptionalInput(context, tilingData, config, isActiveMask, hasElasticInfo, isPerformance,
                                             hasExpertScales) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Check and get optional input param failed."),
                    return ge::GRAPH_FAILED);

    // 检查context输入
    if (config.isMc2Context) {
        OP_TILING_CHECK(CheckMc2Context(context, nodeName, config) != ge::GRAPH_SUCCESS,
                        OP_LOGE_WITHOUT_REPORT(nodeName, "Tiling check context failed."), return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(CheckInputParam(context, nodeName, isActiveMask, hasElasticInfo, isPerformance, hasExpertScales,
                                    localMoeExpertNum, isLayered, isShared, tilingData, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Tiling check input param failed."), return ge::GRAPH_FAILED);

    // 校验win区大小
    OP_TILING_CHECK(CheckAndCalWinSize(context, *tilingData, nodeName, false, localMoeExpertNum, isLayered, config) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Tiling check window size failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkspace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Tiling set workspace Failed"),
                    return ge::GRAPH_FAILED);

    if (!config.isMc2Context) {
        OP_TILING_CHECK(SetHCommCfg(context, tilingData, groupEp, isLayered) != ge::GRAPH_SUCCESS,
                        OP_LOGE_WITHOUT_REPORT(nodeName, "SetHCommCfg failed."), return ge::GRAPH_FAILED);
    }
    tilingData->moeDistributeCombineV2Info.isMc2Context = config.isMc2Context;
    uint64_t tilingKey = CalTilingKey(commQuantMode, isLayered);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    OP_TILING_CHECK(CheckAndSetPlatformInfo(context, tilingData, isLayered, nodeName, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Tiling set platformInfo Failed"),
                    return ge::GRAPH_FAILED);

    PrintTilingDataInfo(nodeName, *tilingData);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineV2TilingFuncBase::MoeDistributeCombineV2TilingFuncNew(gert::TilingContext *context,
                                                                                          const CombineV2Config &config)
{
    // 不支持 expandX数据类型为int32 type
    auto expandXDesc = context->GetInputDesc(config.expandXIndex);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandxDesc"),
                    return ge::GRAPH_FAILED);
    // 检查expandX数据类型为DT_INT32
    OP_TILING_CHECK(
        (expandXDesc->GetDataType() == ge::DT_INT32),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expandX", Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                                  "BF16 or FLOAT16"),
        return ge::GRAPH_FAILED);
    ge::graphStatus ret;

    ret = MoeDistributeCombineTilingFuncImpl(context, config);

    return ret;
}

static void MoeDistributeCombineV2ConfigIndexSet(CombineV2Config &config)
{
    config.tpSendCountsIndex = 5;       // 根据combineV2算子原型标志位设置tpSendCounts索引为5
    config.xActiveMaskIndex = 6;        // 根据combineV2算子原型标志位设置xActiveMask索引为6
    config.activationScaleIndex = 7;    // 根据combineV2算子原型标志位设置activationScale索引为7
    config.weightScaleIndex = 8;        // 根据combineV2算子原型标志位设置weightScale索引为8
    config.groupListIndex = 9;          // 根据combineV2算子原型标志位设置groupList索引为9
    config.sharedExpertXIndex = 11;     // 根据combineV2算子原型标志位设置sharedExpertX索引为11
    config.elasticInfoIndex = 12;       // 根据combineV2算子原型标志位设置elasticInfo索引为12
    config.oriXIndex = 13;              // 根据combineV2算子原型标志位设置oriX索引为13
    config.constExpertAlpha1Index = 14; // 根据combineV2算子原型标志位设置constExpertAlpha1索引为14
    config.constExpertAlpha2Index = 15; // 根据combineV2算子原型标志位设置constExpertAlpha2索引为15
    config.constExpertVIndex = 16;      // 根据combineV2算子原型标志位设置constExpertV索引为16
    config.performanceInfoIndex = 17;   // 根据combineV2算子原型标志位设置performanceInfo索引为17
    config.outputXIndex = 0;            // 根据combineV2算子原型标志位设置outputX索引为0
    config.attrGroupEpIndex = 0;        // 根据combineV2算子原型标志位初始化属性groupEp索引为0
    config.attrEpWorldSizeIndex = 1;    // 根据combineV2算子原型标志位初始化属性epWorldSize索引为1
    config.attrEpRankIdIndex = 2;       // 根据combineV2算子原型标志位初始化属性epRankId索引为2
    config.attrMoeExpertNumIndex = 3;   // 根据combineV2算子原型标志位初始化属性moeExpertNum索引为3
    config.attrGroupTpIndex = 4;        // 根据combineV2算子原型标志位初始化属性attrGroupTpIndex索引为4
    config.attrTpWorldSizeIndex = 5; // 根据combineV2算子原型标志位初始化属性attrTpWorldSizeIndex索引为5
    config.attrTpRankIdIndex = 6;    // 根据combineV2算子原型标志位初始化属性attrTpRankIdIndex索引为6
    config.attrExpertSharedTypeIndex = 7; // 根据combineV2算子原型标志位初始化属性attrExpertSharedTypeIndex索引为7
    config.attrSharedExpertNumIndex = 8; // 根据combineV2算子原型标志位初始化属性attrSharedExpertNumIndex索引为8
    config.attrSharedExpertRankNumIndex = 9; // 根据combineV2算子原型标志位初始化属性attrSharedExpertRankNumIndex索引为9
    config.attrGlobalBsIndex = 10; // 根据combineV2算子原型标志位初始化属性attrGlobalBsIndex索引为10
    config.attrOutDTypeIndex = 11; // 根据combineV2算子原型标志位初始化属性attrOutDTypeIndex索引为11
    config.attrCommQuantModeIndex = 12; // 根据combineV2算子原型标志位初始化属性attrCommQuantModeIndex索引为12
    config.attrGroupListTypeIndex = 13; // 根据combineV2算子原型标志位初始化属性attrGroupListTypeIndex索引为13
    config.attrCommAlgIndex = 14; // 根据combineV2算子原型标志位初始化属性attrCommAlgIndex索引为14
    config.attrZeroExpertNumIndex = 15;  // 根据combineV2算子原型标志位设置属性attrZeroExpertNum索引为15
    config.attrCopyExpertNumIndex = 16;  // 根据combineV2算子原型标志位设置属性attrCopyExpertNum索引为16
    config.attrConstExpertNumIndex = 17; // 根据combineV2算子原型标志位设置属性attrConstExpertNum索引为17
    config.hasAddRmsNorm = false;

    return;
}

static void MoeDistributeCombineARNConfigIndexSet(CombineV2Config &config)
{
    config.residualXIndex = 5;          // 根据combineARN算子原型标志位设置residualX索引为5
    config.gammaIndex = 6;              // 根据combineARN算子原型标志位设置gamma索引为6
    config.tpSendCountsIndex = 7;       // 根据combineARN算子原型标志位设置tpSendCounts索引为7
    config.xActiveMaskIndex = 8;        // 根据combineARN算子原型标志位设置xActiveMask索引为8
    config.activationScaleIndex = 9;    // 根据combineARN算子原型标志位设置activationScale索引为9
    config.weightScaleIndex = 10;       // 根据combineARN算子原型标志位设置weightScale索引为10
    config.groupListIndex = 11;         // 根据combineARN算子原型标志位设置groupList索引为11
    config.sharedExpertXIndex = 13;     // 根据combineARN算子原型标志位设置sharedExpertX索引为13
    config.elasticInfoIndex = 14;       // 根据combineARN算子原型标志位设置elasticInfo索引为14
    config.oriXIndex = 15;              // 根据combineARN算子原型标志位设置oriX索引为15
    config.constExpertAlpha1Index = 16; // 根据combineARN算子原型标志位设置constExpertAlpha1索引为16
    config.constExpertAlpha2Index = 17; // 根据combineARN算子原型标志位设置constExpertAlpha2索引为17
    config.constExpertVIndex = 18;      // 根据combineARN算子原型标志位设置constExpertV索引为18
    config.performanceInfoIndex = 19;   // combineARN算子原型没有传入performanceInfoIndex设置虚拟索引为19
    config.outputYIndex = 0;            // 根据combineARN算子原型标志位设置outputY索引为0
    config.outputRstdIndex = 1;         // 根据combineARN算子原型标志位设置outputRstd索引为1
    config.outputXIndex = 2;            // 根据combineARN算子原型标志位设置outputX索引为2
    config.attrGroupEpIndex = 0;        // 根据combineARN算子原型标志位初始化groupEp属性索引为0
    config.attrEpWorldSizeIndex = 1;    // 根据combineARN算子原型标志位初始化epWorldSize属性索引为1
    config.attrEpRankIdIndex = 2;       // 根据combineARN算子原型标志位初始化epRankId属性索引为2
    config.attrMoeExpertNumIndex = 3;   // 根据combineARN算子原型标志位初始化moeExpertNum属性索引为3
    config.attrGroupTpIndex = 4;     // 根据combineARN算子原型标志位初始化attrGroupTpIndex属性索引为4
    config.attrTpWorldSizeIndex = 5; // 根据combineARN算子原型标志位初始化attrTpWorldSizeIndex属性索引为5
    config.attrTpRankIdIndex = 6;    // 根据combineARN算子原型标志位初始化attrTpRankIdIndex属性索引为6
    config.attrExpertSharedTypeIndex = 7; // 根据combineARN算子原型标志位初始化attrExpertSharedTypeIndex属性索引为7
    config.attrSharedExpertNumIndex = 8; // 根据combineARN算子原型标志位初始化attrSharedExpertNumIndex属性索引为8
    config.attrSharedExpertRankNumIndex =
        9; // 根据combineARN算子原型标志位初始化attrSharedExpertRankNumIndex属性索引为9
    config.attrGlobalBsIndex = 10; // 根据combineARN算子原型标志位初始化attrGlobalBsIndex属性索引为10
    config.attrOutDTypeIndex = 11; // 根据combineARN算子原型标志位初始化attrOutDTypeIndex属性索引为11
    config.attrCommQuantModeIndex = 12; // 根据combineARN算子原型标志位初始化attrCommQuantModeIndex属性索引为12
    config.attrGroupListTypeIndex = 13; // 根据combineARN算子原型标志位初始化attrGroupListTypeIndex属性索引为13
    config.attrCommAlgIndex = 14; // 根据combineARN算子原型标志位初始化attrCommAlgIndex属性索引为14
    config.attrNormEpsIndex = 15; // 根据combineARN算子原型标志位设置attrNormEps属性索引为15
    config.attrZeroExpertNumIndex = 16; // 根据combineARN算子原型标志位设置attrZeroExpertNum属性索引为16
    config.attrCopyExpertNumIndex = 17; // 根据combineARN算子原型标志位设置attrCopyExpertNum属性索引为17
    config.attrConstExpertNumIndex = 18; // 根据combineARN算子原型标志位设置attrConstExpertNum属性索引为18
    config.hasAddRmsNorm = true;

    return;
}
ge::graphStatus MoeDistributeCombineV2TilingFuncBase::MoeDistributeCombineV2TilingFunc(gert::TilingContext *context)
{
    auto rstdOutDesc = context->GetOutputDesc(HAS_ADD_RMS_NORM);
    CombineV2Config config;
    ge::graphStatus ret;
    if (rstdOutDesc == nullptr) {
        MoeDistributeCombineV2ConfigIndexSet(config);
    } else {
        MoeDistributeCombineARNConfigIndexSet(config);
    }
    ret = MoeDistributeCombineV2TilingFuncNew(context, config);

    return ret;
}

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
// Register exception func
inline void MoeDistributeCombineV2ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char *socName = aclrtGetSocName();
    if ((std::strstr(socName, "Ascend950") == nullptr) && (std::strstr(socName, "Ascend910B") == nullptr)) {
        return;
    }
    Mc2Exception::Mc2ExceptionImpl(args, userdata, "MoeDistributeCombineV2");
}

__attribute__((constructor)) void RegisterMoeDistributeCombineV2ExceptionFunc()
{
    int32_t runtimeVersionNum = 0;
    int32_t metadefVersionNum = 0;

    if (aclsysGetVersionNum("runtime", &runtimeVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MoeDistributeCombineV2", "Get runtime version failed when register exception func.");
        return;
    }
    if (aclsysGetVersionNum("metadef", &metadefVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MoeDistributeCombineV2", "Get metadef version failed when register exception func.");
        return;
    }

    if (runtimeVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION || metadefVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION) {
        OP_LOGW("MoeDistributeCombineV2",
                "The runtime(%d) or metadata(%d) version is lower than the version(%d) supporting exception func.",
                runtimeVersionNum, metadefVersionNum, EXCEPTION_DUMP_SUPPORT_VERSION);
        return;
    }

    IMPL_OP(MoeDistributeCombineV2).ExceptionDumpParseFunc(MoeDistributeCombineV2ExceptionImplWrapper);
}
#endif

} // namespace optiling
