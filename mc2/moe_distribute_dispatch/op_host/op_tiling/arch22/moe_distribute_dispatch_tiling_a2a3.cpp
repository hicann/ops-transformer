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
 * \file moe_distribute_dispatch_tiling_a2a3.cpp
 * \brief
 */

#include "moe_distribute_dispatch_tiling_a2a3.h"

using namespace Ops::Transformer::OpTiling;
using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace {
constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
constexpr uint32_t BLOCK_SIZE_A2 = 32;
constexpr uint32_t MAX_K_VALUE_A2 = 16;
constexpr uint32_t LAYERED_SUPPORT_K = 8;
constexpr uint32_t LAYERED_SUPPORT_K_MAX = 16;
constexpr int32_t MAX_HIDDEN_SIZE_A2 = 7168;
constexpr int32_t MAX_EP_WORLD_SIZE_A2 = 256;
constexpr int32_t MAX_MOE_EXPERT_NUMS_A2 = 512;
constexpr int32_t UNLAYERED_EXP_NUM_PER_RANK_A2 = 24;
constexpr uint32_t MAX_BATCH_SIZE_A2 = 256;

constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 10;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 11;
constexpr uint32_t ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX = 12;
const char *K_INNER_DEBUG = "MoeDistributeDispatch";

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 5;
constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 6;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 8;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;

constexpr uint32_t UNQUANT_MODE = 0;
constexpr uint32_t STATIC_QUANT_MODE = 1;
constexpr uint32_t DYNAMIC_QUANT_MODE = 2;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr uint32_t USER_WORKSPACE_A2 = 1 * 1024 * 1024; // moeExpertNum_ * sizeof(uint32_t) + epWorldSize_ * 2 * 32
} // namespace
namespace optiling {

static void MoeDistributeDispatchA2SetTiling(gert::TilingContext *context, MoeDistributeDispatchA2Info &info,
                                             const bool isLayered, const int32_t bs)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX);

    info.epWorldSize = *epWorldSizePtr;
    info.tpWorldSize = static_cast<uint32_t>(0);
    info.epRankId = *epRankIdPtr;
    info.tpRankId = static_cast<uint32_t>(0);
    info.expertSharedType = static_cast<uint32_t>(0);
    info.sharedExpertRankNum = static_cast<uint32_t>(0);
    info.moeExpertNum = *moeExpertNumPtr;
    info.quantMode = *quantModePtr;
    info.maxMoeExpertNum = MAX_MOE_EXPERT_NUMS_A2;

    if (*globalBsPtr == 0) {
        info.globalBs = *epWorldSizePtr * bs;
    } else {
        info.globalBs = *globalBsPtr;
    }
    info.expertTokenNumsType = *expertTokenNumsTypePtr;

    OP_LOGD(K_INNER_DEBUG, "quantMode=%d", info.quantMode);
    OP_LOGD(K_INNER_DEBUG, "globalBs=%d", info.globalBs);
    OP_LOGD(K_INNER_DEBUG, "expertTokenNumsType=%d", info.expertTokenNumsType);
    OP_LOGD(K_INNER_DEBUG, "expertSharedType=%d", info.expertSharedType);
    OP_LOGD(K_INNER_DEBUG, "sharedExpertRankNum=%d", info.sharedExpertRankNum);
    OP_LOGD(K_INNER_DEBUG, "moeExpertNum=%d", info.moeExpertNum);
    OP_LOGD(K_INNER_DEBUG, "epWorldSize=%d", info.epWorldSize);
    OP_LOGD(K_INNER_DEBUG, "tpWorldSize=%d", info.tpWorldSize);
    OP_LOGD(K_INNER_DEBUG, "epRankId=%d", info.epRankId);
    OP_LOGD(K_INNER_DEBUG, "tpRankId=%d", info.tpRankId);
    OP_LOGD(K_INNER_DEBUG, "maxMoeExpertNum=%d", info.maxMoeExpertNum);
}

static ge::graphStatus MoeDistributeDispatchA2CheckAttrAndSetTiling(gert::TilingContext *context,
                                                                    MoeDistributeDispatchA2Info &info,
                                                                    const bool isLayered)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "attrs"), return ge::GRAPH_FAILED);
    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto expertSharedTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "expertIdShape"),
                    return GRAPH_FAILED);
    int32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);

    OP_TILING_CHECK(groupEpPtr == nullptr || strlen(groupEpPtr) == 0,
                    OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "groupEp", "invalid", "valid groupEp"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr || *epWorldSizePtr <= 0 || *epWorldSizePtr > MAX_EP_WORLD_SIZE_A2 ||
                        *epWorldSizePtr % RANK_NUM_PER_NODE_A2 != 0,
                    OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "epWorldSize", "invalid", "valid epWorldSize"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr || *epRankIdPtr < 0 || *epRankIdPtr >= *epWorldSizePtr,
                    OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "epRankId", "invalid", "valid epRankId"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr || *moeExpertNumPtr % *epWorldSizePtr != 0 || *moeExpertNumPtr <= 0 ||
                        *moeExpertNumPtr > MAX_MOE_EXPERT_NUMS_A2,
                    OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "moeExpertNum", "invalid", "valid moeExpertNum"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(!isLayered && *moeExpertNumPtr / *epWorldSizePtr > UNLAYERED_EXP_NUM_PER_RANK_A2,
                    OP_LOGE_WITH_INVALID_ATTR(
                        K_INNER_DEBUG, "moe_expert_num", std::to_string(*moeExpertNumPtr).c_str(),
                        (std::string("at most ") + std::to_string(UNLAYERED_EXP_NUM_PER_RANK_A2 * *epWorldSizePtr) +
                         " in unlayered mode")
                            .c_str()),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "tpWorldSize"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "tpRankId"), return GRAPH_FAILED);
    OP_TILING_CHECK(*tpWorldSizePtr >= 2,
                    OP_LOGE_WITH_INVALID_ATTR(K_INNER_DEBUG, "tp_world_size", std::to_string(*tpWorldSizePtr).c_str(),
                                              "less than 2"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(expertSharedTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "expertSharedType"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "sharedExpertRankNum"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr || (*quantModePtr != UNQUANT_MODE && *quantModePtr != DYNAMIC_QUANT_MODE),
                    OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "quantMode", "invalid", "valid quantMode"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "globalBs"), return GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr || *expertTokenNumsTypePtr < 0 || *expertTokenNumsTypePtr > 1,
                    OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "expertTokenNumsType", "invalid", "0 or 1"),
                    return GRAPH_FAILED);
    MoeDistributeDispatchA2SetTiling(context, info, isLayered, bs);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA2CheckShapeAndSetTiling(gert::TilingContext *context,
                                                                     MoeDistributeDispatchA2Info &info)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "MoeDistributeDispatchA2 MoeDistributeDispatchA2CheckShapeAndSetTiling.");
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);

    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "xShape"), return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "expertIdShape"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(K_INNER_DEBUG, "x", "invalid", "2D"), return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(K_INNER_DEBUG, "expertId", "invalid", "2D"), return GRAPH_FAILED);
    OP_LOGD(nodeName, "X dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "X dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdStorageShape->GetStorageShape().GetDim(1));

    uint32_t h = xStorageShape->GetStorageShape().GetDim(1);
    uint32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);
    uint32_t k = expertIdStorageShape->GetStorageShape().GetDim(1);
    bool isScales = (scalesStorageShape != nullptr);
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "attrs"), return ge::GRAPH_FAILED);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    OP_TILING_CHECK(h % BLOCK_SIZE_A2 != 0 || h == 0 || h > MAX_HIDDEN_SIZE_A2,
                    OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "hiddensize", "invalid", "valid hiddensize"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(bs == 0 || bs > MAX_BATCH_SIZE_A2,
                    OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "batchsize", "invalid", "valid batchsize"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(k == 0 || k > MAX_K_VALUE_A2, OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "k", "invalid", "valid k"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(*quantModePtr == UNQUANT_MODE && isScales,
                    OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "scales", "not null", "null when quantMode is unQuant"),
                    return GRAPH_FAILED);

    info.bs = bs;
    info.k = k;
    info.h = h;

    OP_LOGD(K_INNER_DEBUG, "batchSize is %u", info.bs);
    OP_LOGD(K_INNER_DEBUG, "k is %u", info.k);
    OP_LOGD(K_INNER_DEBUG, "hiddenSize is %u", info.h);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA2GetPlatformInfoAndSetTiling(gert::TilingContext *context,
                                                                          MoeDistributeDispatchA2Info &info)
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

static bool MoeDistributeDispatchA2IsLayered()
{
    const char *hcclIntraPcieEnable = getenv("HCCL_INTRA_PCIE_ENABLE");
    const char *hcclIntraRoceEnable = getenv("HCCL_INTRA_ROCE_ENABLE");

    if (hcclIntraPcieEnable == nullptr || hcclIntraRoceEnable == nullptr) {
        OP_LOGD(K_INNER_DEBUG, "ENV HCCL_INTRA_PCIE_ENABLE or HCCL_INTRA_ROCE_ENABLE don't set");
        return false;
    } else if (strcmp(hcclIntraPcieEnable, "1") == 0 && strcmp(hcclIntraRoceEnable, "0") == 0) {
        OP_LOGD(K_INNER_DEBUG, "ENV HCCL_INTRA_PCIE_ENABLE = 1 and HCCL_INTRA_ROCE_ENABLE = 0, use layered solution.");
        return true;
    }
    OP_LOGD(K_INNER_DEBUG, "ENV HCCL_INTRA_PCIE_ENABLE != 1 or HCCL_INTRA_ROCE_ENABLE != 0, use default solution.");
    return false;
}

static uint64_t MoeDistributeDispatchA2CalcTilingKey(gert::TilingContext *context, const bool isLayered)
{
    uint32_t tilingKeyQuantMode = TILINGKEY_NO_QUANT;
    bool scaleMode = false; // A2 & A3
    uint32_t fullMesh = TILINGKEY_NO_FULLMESH;
    uint32_t layeredMode = TILINGKEY_TPL_MTE; // A2

    if (isLayered) {
        layeredMode = TILINGKEY_TPL_AICPU;
    }
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    scaleMode = (scalesStorageShape != nullptr);
    uint64_t tilingKey = GET_TPL_TILING_KEY(tilingKeyQuantMode, scaleMode, fullMesh, layeredMode, TILINGKEY_TPL_A2);
    OP_LOGD(K_INNER_DEBUG, "tilingKey=%lu", tilingKey);
    return tilingKey;
}

static ge::graphStatus MoeDistributeDispatchA2CheckWinSize(const gert::TilingContext *context, const char *nodeName,
                                                           MoeDistributeDispatchA2Info &info, bool isLayered)
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
        constexpr uint64_t flagBuffSize = 6 * MB_SIZE; // 固定6M空间作为存放同步Flag的区域
        // 每个token发往k个专家时额外需带上专家索引、topk权重、量化系数、到达标志位共4个信息，这些信息对齐到32字节
        const uint64_t extraTokenInfoSize = 4 * ((info.k + 7) / 8 * 8) * sizeof(uint32_t);
        const uint64_t perTokenSize = info.h * sizeofDtypeX + extraTokenInfoSize;
        const uint64_t maxRecvTokenNum = maxBs * (info.moeExpertNum + epWorldSize / RANK_NUM_PER_NODE_A2 * BUFFER_NUM);
        minHcclBuffSize = maxRecvTokenNum * perTokenSize + flagBuffSize;
        if (minHcclBuffSize > hcclBuffSize) {
            OP_LOGW(nodeName,
                    "HCCL_BUFFSIZE is too small, min required HCCL_BUFFSIZE ((moeExpertNum + epWorldSize / 4) * maxBs "
                    "* (h * 2 + 16 * ((k + 7) / 8 * 8)) / 1MB + 6MB) = %luMB, actual HCCL_BUFFSIZE = %luMB, "
                    "moeExpertNum = %u, maxBs = %lu, h = %u, k = %u. %s",
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

static ge::graphStatus MoeDistributeDispatchA2TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MoeDistributeDispatchA2 tiling func.");

    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    auto ret = context->SetScheduleMode(batch_mode);
    MC2_CHECK_LOG_RET(nodeName, ret);

    // 1. tilingData
    MoeDistributeDispatchA2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchA2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITHOUT_REPORT(nodeName, "tilingData is nullptr."),
                    return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "MoeDistributeDispatchA2 get tilingData.");
    MoeDistributeDispatchA2Info &info = tilingData->moeDistributeDispatchInfo;
    OP_LOGI(nodeName, "MoeDistributeDispatchA2 get tilingData info.");

    bool isLayered = MoeDistributeDispatchA2IsLayered();
    OP_TILING_CHECK(
        MoeDistributeDispatchA2CheckShapeAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "MoeDistributeDispatchA2 CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        MoeDistributeDispatchA2CheckAttrAndSetTiling(context, info, isLayered) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "MoeDistributeDispatchA2 CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        MoeDistributeDispatchA2GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "MoeDistributeDispatchA2 GetPlatformInfoAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeDispatchA2CheckWinSize(context, nodeName, info, isLayered) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "MoeDistributeDispatchA2 CheckWinSize Failed"),
                    return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    context->SetAicpuBlockDim(mc2tiling::AICPU_NUM_BLOCKS_A2);

    uint64_t tilingKey = MoeDistributeDispatchA2CalcTilingKey(context, isLayered);
    context->SetTilingKey(tilingKey);
    // 2. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITHOUT_REPORT(nodeName, "workSpaces is nullptr."),
                    return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + USER_WORKSPACE_A2;

    // 3. communication
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    uint32_t opType = 18; // batch write=18,
    std::string algConfig = "BatchWrite=level0:fullmesh";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "Leave MoeDistributeDispatchA2 tiling func.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchTilingA2A3::MoeDistributeDispatchTilingFunc(gert::TilingContext *context)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    ge::graphStatus ret;
    if (socVersion == "Ascend910B") {
        ret = MoeDistributeDispatchA2TilingFuncImpl(context);
    } else {
        ret = MoeDistributeDispatchA3A5TilingFuncImpl(context);
    }
    return ret;
}

ge::graphStatus MoeDistributeDispatchTilingA2A3::CheckEpWorldSizeAttrs(const char *nodeName,
                                                                       MoeDistributeDispatchTilingData &tilingData)
{
    uint32_t epWorldSize = tilingData.moeDistributeDispatchInfo.epWorldSize;

    // 检验epWorldSize是否是8的倍数
    OP_TILING_CHECK(
        epWorldSize % 8 != 0,
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "ep_world_size", std::to_string(epWorldSize).c_str(), "divisible by 8"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK((256 % epWorldSize != 0) && (epWorldSize % 144 != 0),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "ep_world_size", std::to_string(epWorldSize).c_str(),
                                              "one of [8, 16, 32, 64, 128, 144, 256, 288]"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// A5 采用 MTE 方式通信，复用 A3 tiling
REGISTER_OPS_TILING_TEMPLATE(MoeDistributeDispatch, MoeDistributeDispatchTilingA2A3, 0);

} // namespace optiling
