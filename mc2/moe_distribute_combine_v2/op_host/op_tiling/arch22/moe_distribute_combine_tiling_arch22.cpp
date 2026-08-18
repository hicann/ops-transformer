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
 * \file moe_distribute_combine_tiling_arch22.cpp
 * \brief
 */
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../../../common/op_kernel/mc2_moe_context.h"
#include "../../../op_kernel/moe_distribute_combine_tiling.h"
#include "../../../op_kernel/moe_distribute_combine_v2_tiling.h"
#include "mc2_hcom_topo_info.h"
#include "../../../../moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_check_win_size.h"
#include "mc2_exception_dump.h"
#include "moe_distribute_combine_tiling_arch22.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace {
constexpr size_t SYSTEM_NEED_WORKSPACE = 16UL * 1024UL * 1024UL;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 5;
constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 6;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 8;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 10;
constexpr uint32_t ATTR_OUT_DTYPE_INDEX = 11;
constexpr uint32_t ATTR_COMM_QUANT_MODE_INDEX = 12;
constexpr uint32_t ATTR_GROUP_LIST_TYPE_INDEX = 13;
constexpr uint32_t ATTR_COMM_ALG_INDEX = 14;

// A2
constexpr int32_t MAX_EP_WORLD_SIZE_A2 = 384;
constexpr int32_t MAX_EP_WORLD_SIZE_A2_LAYERED = 64;
constexpr int32_t MAX_MOE_EXPERT_NUMS_A2_FULLMESH = 1024;
constexpr int32_t MAX_MOE_EXPERT_NUMS_A2_HIERARCHY = 512;
constexpr uint32_t MAX_HIDDEN_SIZE_A2 = 10240;
constexpr uint32_t MAX_BATCH_SIZE_A2 = 256;
constexpr uint32_t LAYERED_MAX_BATCH_SIZE_A2 = 512;
constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
constexpr uint32_t BLOCK_SIZE_A2 = 32;
constexpr uint32_t MAX_K_VALUE_A2 = 16;
constexpr uint64_t TILING_KEY_BASE_A2 = 2000UL;
constexpr uint64_t TILING_KEY_LAYERED_COMM_A2 = 3000UL;
constexpr uint64_t TILING_KEY_INT8_COMM_QUANT_A2 = 100UL;
const char *K_OP_NAME = "MoeDistributeCombineV2";
const char *K_INNER_DEBUG = "MoeDistributeCombineV2 Tiling Debug";

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
// 为了兼容老版本，在未配置commAlg参数时，读取环境变量；
// commAlg参数当前支持"fullmesh"和"hierarchy"两种，其余报错。
static ge::graphStatus MoeDistributeCombineCheckCommAlg(const gert::TilingContext *context, bool &isLayered,
                                                        const CombineV2Config &config)
{
    isLayered = false;
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "attrs"), return ge::GRAPH_FAILED);
    auto commAlg = attrs->GetAttrPointer<char>(static_cast<int>((config.attrCommAlgIndex)));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    if ((epWorldSizePtr != nullptr) && (*epWorldSizePtr <= RANK_NUM_PER_NODE_A2)) {
        isLayered = false;
        OP_LOGD(K_INNER_DEBUG, "epWorldSize <= 8, use default fullmesh algorithm.");
        return ge::GRAPH_SUCCESS;
    }
    if (commAlg == nullptr || strlen(commAlg) == 0 || strcmp(commAlg, "0") == 0) {
        OP_LOGW(K_INNER_DEBUG, "Attr commAlg is invalid, please configure fullmesh or hierarchy.");

        const char *hcclIntraPcieEnable = getenv("HCCL_INTRA_PCIE_ENABLE");
        const char *hcclIntraRoceEnable = getenv("HCCL_INTRA_ROCE_ENABLE");
        if (hcclIntraPcieEnable != nullptr && hcclIntraRoceEnable != nullptr && strcmp(hcclIntraPcieEnable, "1") == 0 &&
            strcmp(hcclIntraRoceEnable, "0") == 0) {
            OP_LOGD(K_INNER_DEBUG,
                    "ENV HCCL_INTRA_PCIE_ENABLE = 1 and HCCL_INTRA_ROCE_ENABLE = 0, use hierarchy algorithm.");
            isLayered = true;
        } else {
            OP_LOGD(K_INNER_DEBUG,
                    "ENV HCCL_INTRA_PCIE_ENABLE != 1 or HCCL_INTRA_ROCE_ENABLE != 0, use default fullmesh algorithm.");
        }
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGI(K_INNER_DEBUG, "commAlg is %s", commAlg);

    if (strcmp(commAlg, "fullmesh") == 0) {
        return ge::GRAPH_SUCCESS;
    } else if (strcmp(commAlg, "hierarchy") == 0) {
        isLayered = true;
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "commAlg", commAlg != nullptr ? commAlg : "null", "fullmesh or hierarchy");
        return GRAPH_FAILED;
    }
}

static uint64_t MoeDistributeCombineA2CalcTilingKey(const bool isLayered, const int32_t commQuantMode)
{
    uint32_t quantMode = TILINGKEY_NO_QUANT;
    uint32_t layeredMode = TILINGKEY_TPL_MTE; // A2

    if (isLayered) {
        layeredMode = TILINGKEY_TPL_AICPU;
        if (commQuantMode == static_cast<CommQuantModeType>(CommQuantMode::INT8_QUANT)) {
            quantMode = TILINGKEY_INT8_QUANT;
        }
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(quantMode, layeredMode, TILINGKEY_TPL_A2);
    OP_LOGD(K_INNER_DEBUG, "tilingKey=%lu", tilingKey);
    return tilingKey;
}

static std::string MoeDistributeCombineA2GetAlgConfig(int32_t epWorldSize, bool isLayered)
{
    if (epWorldSize <= RANK_NUM_PER_NODE_A2) {
        return "BatchWrite=level0:fullmesh";
    }
    return isLayered ? "BatchWrite=level1:hierarchy" : "BatchWrite=level1:fullmesh";
}

static ge::graphStatus MoeDistributeCombineA2CheckWinSize(const gert::TilingContext *context, const char *nodeName,
                                                          MoeDistributeCombineA2Info &info, bool isLayered)
{
    // 为避免兼容性问题，校验失败时不直接返回错误，而是输出警告日志
    auto groupEp = context->GetAttrs()->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    uint64_t hcclBuffSize = 0ULL;
    auto ret = mc2tiling::GetCclBufferSize(groupEp, &hcclBuffSize, nodeName);
    OP_LOGD(nodeName, "HCCL_BUFFSIZE = %lu Bytes (%lu MB).", hcclBuffSize, ops::CeilDiv(hcclBuffSize, MB_SIZE));
    // 当处于在线编译、SuperKernel等特定场景时，可能无法获取到HCCL_BUFFSIZE，此时跳过校验
    OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS, OP_LOGW(nodeName, "Can't get HCCL_BUFFSIZE and skip validation."),
                    return ge::GRAPH_SUCCESS);
    uint32_t epWorldSize = info.epWorldSize;
    uint32_t localMoeExpertNum = info.moeExpertNum / epWorldSize;
    uint64_t maxBs = static_cast<uint64_t>(info.globalBs) / epWorldSize;
    uint64_t minHcclBuffSize = 0ULL;
    constexpr uint64_t sizeofDtypeX = 2ULL; // token数据类型为float16/bfloat16，每个元素字节数为2
    constexpr uint64_t BUFFER_NUM = 2UL;
    constexpr const char *HCCL_BUFFSIZE_HINT =
        "Please increase the HCCL_BUFFSIZE environment variable or provide an HcclCommConfig with a larger "
        "hcclBufferSize when creating the communication domain.";
    if (isLayered) {
        constexpr uint64_t BUFFER_ALIGN = 512UL;
        constexpr uint64_t flagBuffSize = 8 * MB_SIZE; // 固定8M空间作为存放同步Flag的区域
        // 每个token发往k个专家时额外需带上专家索引、topk权重、量化系数、到达标志位共4个信息，这些信息对齐到32字节
        const uint64_t extraTokenInfoSize = 4 * ((info.k + 7) / 8 * 8) * sizeof(uint32_t);
        const uint64_t perTokenSize = info.h * sizeofDtypeX + extraTokenInfoSize;
        uint64_t maxRecvTokenSize = (maxBs * perTokenSize + BUFFER_ALIGN - 1) / BUFFER_ALIGN * BUFFER_ALIGN;
        minHcclBuffSize =
            maxRecvTokenSize * (info.moeExpertNum + epWorldSize / RANK_NUM_PER_NODE_A2 * BUFFER_NUM) + flagBuffSize;
        if (minHcclBuffSize > hcclBuffSize) {
            OP_LOGW(nodeName,
                    "HCCL_BUFFSIZE is too small, min required HCCL_BUFFSIZE "
                    "((moeExpertNum + epWorldSize / 4) * Align512(maxBs "
                    "* (h * 2 + 16 * Align8(k))) / 1MB + 8MB) = %luMB, actual HCCL_BUFFSIZE = %luMB, "
                    "moeExpertNum = %u, maxBs = %lu, h = %u, k = %u. AlignY(x) = (x + Y - 1) / Y * Y. %s",
                    ops::CeilDiv(minHcclBuffSize, MB_SIZE), ops::CeilDiv(hcclBuffSize, MB_SIZE), info.moeExpertNum,
                    maxBs, info.h, info.k, HCCL_BUFFSIZE_HINT);
        }
    } else {
        constexpr uint64_t extraBuffSize = 2 * MB_SIZE; // 固定2M额外空间作为存储非数据信息的区域
        const uint64_t perTokenSize = info.h * sizeofDtypeX;
        const uint64_t maxRecvTokenNum = maxBs * epWorldSize * std::min(localMoeExpertNum, info.k);
        minHcclBuffSize = BUFFER_NUM * (maxRecvTokenNum * perTokenSize + extraBuffSize);
        if (minHcclBuffSize > hcclBuffSize) {
            OP_LOGW(nodeName,
                    "HCCL_BUFFSIZE is too small, min required HCCL_BUFFSIZE (%lu * (maxBs * epWorldSize * "
                    "min(localMoeExpertNum, k) * h * 2 / 1MB + 2MB)) = %luMB, actual HCCL_BUFFSIZE = %luMB, maxBs = "
                    "%lu, epWorldSize = %u, localMoeExpertNum = %u, k = %u, h = %u. %s",
                    BUFFER_NUM, ops::CeilDiv(minHcclBuffSize, MB_SIZE), ops::CeilDiv(hcclBuffSize, MB_SIZE), maxBs,
                    epWorldSize, localMoeExpertNum, info.k, info.h, HCCL_BUFFSIZE_HINT);
        }
    }
    return ge::GRAPH_SUCCESS;
}

// a2专有
static void PrintA2TilingDataInfo(MoeDistributeCombineA2Info &info)
{
    OP_LOGD(K_INNER_DEBUG, "epWorldSize is %u.", info.epWorldSize);
    OP_LOGD(K_INNER_DEBUG, "epRankId is %u.", info.epRankId);
    OP_LOGD(K_INNER_DEBUG, "expertSharedType is %u.", info.expertSharedType);
    OP_LOGD(K_INNER_DEBUG, "sharedExpertRankNum is %u.", info.sharedExpertRankNum);
    OP_LOGD(K_INNER_DEBUG, "moeExpertNum is %u.", info.moeExpertNum);
    OP_LOGD(K_INNER_DEBUG, "globalBs is %u.", info.globalBs);
}

static ge::graphStatus MoeDistributeCombineA2CheckAttrAndSetTiling(const gert::TilingContext *context,
                                                                   MoeDistributeCombineA2Info &info,
                                                                   int32_t &commQuantMode, const bool isLayered,
                                                                   const CombineV2Config &config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "attrs"), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto expertSharedTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_QUANT_MODE_INDEX);

    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));

    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "zeroExpertNum"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "copyExpertNum"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr || *constExpertNumPtr != 0,
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "constExpertNum",
                                              constExpertNumPtr ? std::to_string(*constExpertNumPtr).c_str() : "null",
                                              "Must be 0."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "groupEp", groupEpPtr != nullptr ? groupEpPtr : "null",
                                              "should be a valid non-empty group name"),
                    return ge::GRAPH_FAILED);
    int32_t maxEpWorldSizeA2 = MAX_EP_WORLD_SIZE_A2;
    if (isLayered) {
        maxEpWorldSizeA2 = MAX_EP_WORLD_SIZE_A2_LAYERED;
    }
    OP_TILING_CHECK(epWorldSizePtr == nullptr || *epWorldSizePtr <= 0 || *epWorldSizePtr > maxEpWorldSizeA2 ||
                        ((*epWorldSizePtr > RANK_NUM_PER_NODE_A2) && (*epWorldSizePtr % RANK_NUM_PER_NODE_A2 != 0)),
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "epWorldSize",
                                              epWorldSizePtr ? std::to_string(*epWorldSizePtr).c_str() : "null",
                                              "in valid range or 8-aligned"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(
        epRankIdPtr == nullptr || *epRankIdPtr < 0 || *epRankIdPtr >= *epWorldSizePtr,
        OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "epRankId", epRankIdPtr ? std::to_string(*epRankIdPtr).c_str() : "null",
                                  "in range [0, epWorldSize)"),
        return GRAPH_FAILED);
    int32_t maxMoeExpertNums = isLayered ? MAX_MOE_EXPERT_NUMS_A2_HIERARCHY : MAX_MOE_EXPERT_NUMS_A2_FULLMESH;
    OP_TILING_CHECK(moeExpertNumPtr == nullptr || *moeExpertNumPtr <= 0 || *moeExpertNumPtr > maxMoeExpertNums ||
                        *moeExpertNumPtr % *epWorldSizePtr != 0,
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "moeExpertNum",
                                              moeExpertNumPtr ? std::to_string(*moeExpertNumPtr).c_str() : "null",
                                              "in valid range and divisible by epWorldSize"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "tpWorldSize"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(
        *tpWorldSizePtr >= 2,
        OP_LOGE_WITH_INVALID_ATTR(K_OP_NAME, "tp_world_size", std::to_string(*tpWorldSizePtr).c_str(), "less than 2"),
        return GRAPH_FAILED);
    OP_TILING_CHECK(expertSharedTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "expertSharedType"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "sharedExpertRankNum"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "globalBs"), return GRAPH_FAILED);
    OP_TILING_CHECK(commQuantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "commQuantMode"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(!isLayered && *commQuantModePtr != static_cast<CommQuantModeType>(CommQuantMode::NON_QUANT),
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "commQuantMode", std::to_string(*commQuantModePtr).c_str(),
                                              "should be 0 (NON_QUANT) when not layered"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(isLayered && *commQuantModePtr != static_cast<CommQuantModeType>(CommQuantMode::NON_QUANT) &&
                        *commQuantModePtr != static_cast<CommQuantModeType>(CommQuantMode::INT8_QUANT),
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "commQuantMode", std::to_string(*commQuantModePtr).c_str(),
                                              "should be 0 (NON_QUANT) or 2 (INT8_QUANT) when layered"),
                    return GRAPH_FAILED);

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "xShape"), return false);
    int32_t globalBs = *epWorldSizePtr * expertIdStorageShape->GetStorageShape().GetDim(0);

    // 判断是否满足uint32_t及其他限制
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = 0LL;
    OP_TILING_CHECK(
        (moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum) > INT32_MAX,
        OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "totalExpertNum",
                                  std::to_string(moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum).c_str(),
                                  "should not exceed MAX_INT32"),
        return ge::GRAPH_FAILED);
    info.epWorldSize = *epWorldSizePtr;
    info.epRankId = *epRankIdPtr;
    info.expertSharedType = static_cast<uint32_t>(0);
    info.sharedExpertRankNum = static_cast<uint32_t>(0);
    info.moeExpertNum = *moeExpertNumPtr;

    info.zeroExpertNum = static_cast<uint32_t>(zeroExpertNum);
    info.copyExpertNum = static_cast<uint32_t>(copyExpertNum);
    info.constExpertNum = static_cast<uint32_t>(constExpertNum);

    if (*globalBsPtr == 0) {
        info.globalBs = static_cast<uint32_t>(globalBs);
    } else {
        info.globalBs = *globalBsPtr;
    }
    commQuantMode = *commQuantModePtr;
    PrintA2TilingDataInfo(info);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineA2CheckShapeAndSetTiling(const gert::TilingContext *context,
                                                                    MoeDistributeCombineA2Info &info, bool isLayered,
                                                                    const CombineV2Config &config)
{
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(config.expandXIndex);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
    const gert::StorageShape *performanceInfoStorageShape = context->GetOptionalInputShape(config.performanceInfoIndex);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "expandXShape"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "expertIdShape"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(elasticInfoStorageShape != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "elasticInfo", "present", "not supported in current version"),
                    return GRAPH_FAILED);

    // copy expert and const expert
    const gert::StorageShape *oriXStorageShape = context->GetOptionalInputShape(config.oriXIndex);
    const gert::StorageShape *constExpertAlpha1StorageShape =
        context->GetOptionalInputShape(config.constExpertAlpha1Index);
    const gert::StorageShape *constExpertAlpha2StorageShape =
        context->GetOptionalInputShape(config.constExpertAlpha2Index);
    const gert::StorageShape *constExpertVStorageShape = context->GetOptionalInputShape(config.constExpertVIndex);

    OP_TILING_CHECK(
        expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            K_OP_NAME, "expandX", std::to_string(expandXStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of expandX must be 2D."),
        return GRAPH_FAILED);
    uint32_t h = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        h == 0 || h > MAX_HIDDEN_SIZE_A2 || h % BLOCK_SIZE_A2 != 0,
        OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "h", std::to_string(h).c_str(), "in valid range [1, 10240], 32-aligned"),
        return GRAPH_FAILED);
    OP_TILING_CHECK(
        expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            K_OP_NAME, "expertId", std::to_string(expertIdStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of expertId must be 2D."),
        return GRAPH_FAILED);
    uint32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);
    uint32_t maxBatchSizeA2 = isLayered ? LAYERED_MAX_BATCH_SIZE_A2 : MAX_BATCH_SIZE_A2;
    OP_TILING_CHECK(bs == 0 || bs > maxBatchSizeA2,
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "bs", std::to_string(bs).c_str(), "in valid range"),
                    return GRAPH_FAILED);

    uint32_t k = expertIdStorageShape->GetStorageShape().GetDim(1);
    auto attrs = context->GetAttrs();
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));
    // 判断是否满足uint32_t及其他限制
    int32_t moeExpertNum = *moeExpertNumPtr;
    int32_t zeroExpertNum = static_cast<int32_t>(*zeroExpertNumPtr);
    int32_t copyExpertNum = static_cast<int32_t>(*copyExpertNumPtr);
    int32_t constExpertNum = 0;
    OP_TILING_CHECK(k == 0 || k > MAX_K_VALUE_A2 || k > moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum,
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "k", std::to_string(k).c_str(), "in valid range"),
                    return GRAPH_FAILED);

    bool isActiveMask = (xActiveMaskStorageShape != nullptr);
    if (isActiveMask) {
        const int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(((xActiveMaskDimNums != ONE_DIM) && (xActiveMaskDimNums != TWO_DIMS)),
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            K_OP_NAME, "xActiveMask", std::to_string(xActiveMaskDimNums).c_str(),
                            "The shape dim of xActiveMask must be within the range {1D, 2D}."),
                        return GRAPH_FAILED);

        int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(xActiveMaskDim0 != static_cast<int64_t>(bs),
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            K_OP_NAME, "xActiveMask", std::to_string(xActiveMaskDim0).c_str(),
                            "The dim0 of xActiveMask should equal to bs=" + std::to_string(bs)),
                        return GRAPH_FAILED);

        OP_TILING_CHECK(
            ((xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) &&
             (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != static_cast<int64_t>(k))),
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                K_OP_NAME, "xActiveMask", std::to_string(xActiveMaskStorageShape->GetStorageShape().GetDim(1)).c_str(),
                "The dim1 of xActiveMask should equal to k=" + std::to_string(k)),
            return GRAPH_FAILED);
    }

    // copy expert and const expert
    OP_TILING_CHECK(
        copyExpertNum > 0 && oriXStorageShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_OP_NAME, "oriX", "missing", "oriX must exist when copyExpertNum > 0"),
        return GRAPH_FAILED);
    OP_TILING_CHECK(
        constExpertNum > 0 && (oriXStorageShape == nullptr || constExpertAlpha1StorageShape == nullptr ||
                               constExpertAlpha2StorageShape == nullptr || constExpertVStorageShape == nullptr),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_OP_NAME, "const_expert_inputs", "missing",
                                              "oriX, alpha1, alpha2, V must exist when constExpertNum > 0"),
        return GRAPH_FAILED);

    OP_TILING_CHECK(
        constExpertAlpha1StorageShape != nullptr || constExpertAlpha2StorageShape != nullptr ||
            constExpertVStorageShape != nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_OP_NAME, "const_expert_alpha/v", "present",
                                              "current version does not support const_expert_alpha_1/alpha_2/v"),
        return GRAPH_FAILED);

    if (oriXStorageShape != nullptr) {
        // 必须是2维
        OP_TILING_CHECK(oriXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            K_OP_NAME, "ori_x", std::to_string(oriXStorageShape->GetStorageShape().GetDimNum()).c_str(),
                            "The shape dim of ori_x must be 2D."),
                        return GRAPH_FAILED);

        // shape为(bs, h)
        int64_t oriXDim0 = oriXStorageShape->GetStorageShape().GetDim(0);
        int64_t oriXDim1 = oriXStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(oriXDim0 != static_cast<int64_t>(bs),
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            K_OP_NAME, "ori_x", std::to_string(oriXDim0).c_str(),
                            ("The dim0 of ori_x should equal to bs=" + std::to_string(bs)).c_str()),
                        return GRAPH_FAILED);
        OP_TILING_CHECK(oriXDim1 != static_cast<int64_t>(h),
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            K_OP_NAME, "ori_x", std::to_string(oriXDim1).c_str(),
                            ("The dim1 of ori_x should equal to h=" + std::to_string(h)).c_str()),
                        return GRAPH_FAILED);
    }

    if (constExpertAlpha1StorageShape != nullptr) {
        // 必须是1维
        OP_TILING_CHECK(constExpertAlpha1StorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            K_OP_NAME, "const_expert_alpha_1",
                            std::to_string(constExpertAlpha1StorageShape->GetStorageShape().GetDimNum()).c_str(),
                            "The shape dim of const_expert_alpha_1 must be 1D."),
                        return GRAPH_FAILED);

        // shape为(constExpertNum)
        int64_t constExpertAlpha1Dim0 = constExpertAlpha1StorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(constExpertAlpha1Dim0 != *constExpertNumPtr,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            K_OP_NAME, "const_expert_alpha_1", std::to_string(constExpertAlpha1Dim0).c_str(),
                            ("The dim0 of const_expert_alpha_1 should equal to const_expert_num=" +
                             std::to_string(*constExpertNumPtr))
                                .c_str()),
                        return GRAPH_FAILED);
    }

    if (constExpertAlpha2StorageShape != nullptr) {
        // 必须是1维
        OP_TILING_CHECK(constExpertAlpha2StorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            K_OP_NAME, "const_expert_alpha_2",
                            std::to_string(constExpertAlpha2StorageShape->GetStorageShape().GetDimNum()).c_str(),
                            "The shape dim of const_expert_alpha_2 must be 1D."),
                        return GRAPH_FAILED);

        // shape为(constExpertNum)
        int64_t constExpertAlpha2Dim0 = constExpertAlpha2StorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(constExpertAlpha2Dim0 != *constExpertNumPtr,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            K_OP_NAME, "const_expert_alpha_2", std::to_string(constExpertAlpha2Dim0).c_str(),
                            ("The dim0 of const_expert_alpha_2 should equal to const_expert_num=" +
                             std::to_string(*constExpertNumPtr))
                                .c_str()),
                        return GRAPH_FAILED);
    }

    if (constExpertVStorageShape != nullptr) {
        // 必须是2维
        OP_TILING_CHECK(constExpertVStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            K_OP_NAME, "const_expert_v",
                            std::to_string(constExpertVStorageShape->GetStorageShape().GetDimNum()).c_str(),
                            "The shape dim of const_expert_v must be 2D."),
                        return GRAPH_FAILED);
        // 必须是2维(constExpertNum, H)
        int64_t constExpertVDim0 = constExpertVStorageShape->GetStorageShape().GetDim(0);
        int64_t constExpertVDim1 = constExpertVStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(
            constExpertVDim0 != *constExpertNumPtr,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                K_OP_NAME, "const_expert_v", std::to_string(constExpertVDim0).c_str(),
                ("The dim0 of const_expert_v should equal to const_expert_num=" + std::to_string(*constExpertNumPtr))
                    .c_str()),
            return GRAPH_FAILED);
        OP_TILING_CHECK(constExpertVDim1 != static_cast<int64_t>(h),
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            K_OP_NAME, "const_expert_v", std::to_string(constExpertVDim1).c_str(),
                            ("The dim1 of const_expert_v should equal to h=" + std::to_string(h)).c_str()),
                        return GRAPH_FAILED);
    }

    OP_TILING_CHECK(
        performanceInfoStorageShape != nullptr && performanceInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            K_OP_NAME, "performanceInfo",
            performanceInfoStorageShape != nullptr ?
                std::to_string(performanceInfoStorageShape->GetStorageShape().GetDimNum()).c_str() :
                "null",
            "When performanceInfo is not null, it needs to be one-dimensional."),
        return GRAPH_FAILED);
    attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "attrs"), return ge::GRAPH_FAILED);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(
        performanceInfoStorageShape != nullptr &&
            performanceInfoStorageShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(*epWorldSizePtr),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            K_OP_NAME, "performanceInfo",
            performanceInfoStorageShape != nullptr ?
                std::to_string(performanceInfoStorageShape->GetStorageShape().GetDim(0)).c_str() :
                "null",
            ("The size of performanceInfo should equal to epWorldSize=" + std::to_string(*epWorldSizePtr)).c_str()),
        return GRAPH_FAILED);

    info.isTokenMask = ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    info.isExpertMask = ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));
    info.bs = bs;
    info.k = k;
    info.h = h;

    OP_LOGD(K_INNER_DEBUG, "batchSize is %u", bs);
    OP_LOGD(K_INNER_DEBUG, "k is %u", k);
    OP_LOGD(K_INNER_DEBUG, "hiddenSize is %u", h);
    OP_LOGD(K_INNER_DEBUG, "isTokenMask is %d", static_cast<int32_t>(info.isTokenMask));
    OP_LOGD(K_INNER_DEBUG, "isExpertMask is %d", static_cast<int32_t>(info.isExpertMask));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineA2GetPlatformInfoAndSetTiling(const gert::TilingContext *context,
                                                                         MoeDistributeCombineA2Info &info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OP_LOGD(K_INNER_DEBUG, "aivNum=%d", info.aivNum);
    OP_LOGD(K_INNER_DEBUG, "ubSize=%lu", info.totalUbSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineA2TilingFuncImpl(gert::TilingContext *context, const CombineV2Config &config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MoeDistributeCombineA2 tiling func.");
    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    auto ret = context->SetScheduleMode(batch_mode);
    MC2_CHECK_LOG_RET(nodeName, ret);

    // tilingData
    MoeDistributeCombineA2TilingData *tilingData = context->GetTilingData<MoeDistributeCombineA2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);
    MoeDistributeCombineA2Info &info = tilingData->moeDistributeCombineInfo;

    bool isLayered = false;
    OP_TILING_CHECK(MoeDistributeCombineCheckCommAlg(context, isLayered, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "MoeDistributeCombineA2 CheckCommAlg Failed"),
                    return ge::GRAPH_FAILED);
    int32_t commQuantMode = 0;
    OP_TILING_CHECK(
        MoeDistributeCombineA2CheckShapeAndSetTiling(context, info, isLayered, config) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "MoeDistributeCombineA2 CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        MoeDistributeCombineA2CheckAttrAndSetTiling(context, info, commQuantMode, isLayered, config) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "MoeDistributeCombineA2 CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        MoeDistributeCombineA2GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "MoeDistributeCombineA2 GetPlatformInfoAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeCombineA2CheckWinSize(context, nodeName, info, isLayered) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "MoeDistributeCombineA2 CheckWinSize Failed"),
                    return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    uint32_t aicpuBlockDim = info.epWorldSize > RANK_NUM_PER_NODE_A2 ? mc2tiling::AICPU_NUM_BLOCKS_A2 : 1;
    context->SetAicpuBlockDim(aicpuBlockDim);

    uint64_t tilingKey = MoeDistributeCombineA2CalcTilingKey(isLayered, commQuantMode);
    context->SetTilingKey(tilingKey);
    // 2. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workSpaces"), return ge::GRAPH_FAILED);
    // SYSTEM_NEED_WORKSPACE + userWorkspaceSize
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + static_cast<size_t>(info.moeExpertNum * sizeof(uint32_t) * 2U);

    // 3. communication
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
    Mc2Exception::MC2GroupNameManager::GetInstance().SetGroupName(group);
#endif
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    std::string algConfig = MoeDistributeCombineA2GetAlgConfig(*epWorldSizePtr, isLayered);
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, static_cast<uint32_t>(18), algConfig); // opType=18
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling failed"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineV2TilingFuncA2A3::MoeDistributeCombineTilingFuncImpl(gert::TilingContext *context,
                                                                                         const CombineV2Config &config)
{
    std::string socVersion = mc2tiling::GetSocVersion(context);
    ge::graphStatus ret;

    if (socVersion == "Ascend910B") {
        ret = MoeDistributeCombineA2TilingFuncImpl(context, config);
    } else {
        ret = MoeDistributeCombineA3TilingFuncImpl(context, config);
    }

    return ret;
}

struct MoeDistributeCombineCompileInfo {};
static ge::graphStatus TilingParseForMoeDistributeCombineV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineV2TilingFuncImplA2A3(gert::TilingContext *context)
{
    MoeDistributeCombineV2TilingFuncA2A3 impl;
    return impl.MoeDistributeCombineV2TilingFunc(context);
}

IMPL_OP_OPTILING(MoeDistributeCombineV2)
    .Tiling(MoeDistributeCombineV2TilingFuncImplA2A3)
    .TilingParse<MoeDistributeCombineCompileInfo>(TilingParseForMoeDistributeCombineV2);

} // namespace optiling
