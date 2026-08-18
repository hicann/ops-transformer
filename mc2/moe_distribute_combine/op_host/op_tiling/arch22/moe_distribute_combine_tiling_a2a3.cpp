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
 * \file moe_distribute_combine_tiling_a2a3.cpp
 * \brief
 */

#include "moe_distribute_combine_tiling_a2a3.h"

using namespace Ops::Transformer::OpTiling;
using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace {
const size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr uint64_t TILING_KEY_BASE_A2 = 2000UL;
constexpr uint64_t TILING_KEY_LAYERED_COMM_A2 = 3000UL;
constexpr uint64_t TILING_KEY_INT8_COMM_QUANT_A2 = 100UL;

constexpr int32_t MAX_EP_WORLD_SIZE_A2 = 256;
constexpr int32_t MAX_MOE_EXPERT_NUMS_A2 = 512;
constexpr int32_t MAX_HIDDEN_SIZE_A2 = 7168;
constexpr uint32_t MAX_BATCH_SIZE_A2 = 256;
constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
constexpr uint32_t BLOCK_SIZE_A2 = 32;
constexpr uint32_t MAX_K_VALUE_A2 = 16;
constexpr uint32_t INT8_COMM_QUANT = 2U;

const char *K_OP_NAME = "MoeDistributeCombine";
const char *K_INNER_DEBUG = "MoeDistributeCombine Tiling Debug";

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 8;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 10;
constexpr uint32_t ATTR_COMM_QUANT_MODE_INDEX = 12;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;

constexpr uint64_t MB_SIZE = 1024UL * 1024UL;

enum class CommQuantMode : int32_t {
    NON_QUANT = 0,
    INT12_QUANT = 1,
    INT8_QUANT = 2
};

using CommQuantModeType = std::underlying_type_t<CommQuantMode>;
} // namespace
namespace optiling {
static void PrintA2TilingDataInfo(MoeDistributeCombineA2Info &info)
{
    OP_LOGD(K_INNER_DEBUG, "epWorldSize is %u.", info.epWorldSize);
    OP_LOGD(K_INNER_DEBUG, "epRankId is %u.", info.epRankId);
    OP_LOGD(K_INNER_DEBUG, "expertSharedType is %u.", info.expertSharedType);
    OP_LOGD(K_INNER_DEBUG, "sharedExpertRankNum is %u.", info.sharedExpertRankNum);
    OP_LOGD(K_INNER_DEBUG, "moeExpertNum is %u.", info.moeExpertNum);
    OP_LOGD(K_INNER_DEBUG, "globalBs is %u.", info.globalBs);
}

static bool MoeDistributeCombineA2IsLayered()
{
    const char *hcclIntraPcieEnable = getenv("HCCL_INTRA_PCIE_ENABLE");
    const char *hcclIntraRoceEnable = getenv("HCCL_INTRA_ROCE_ENABLE");

    if (hcclIntraPcieEnable == nullptr || hcclIntraRoceEnable == nullptr) {
        OP_LOGD(K_INNER_DEBUG, "ENV HCCL_INTRA_PCIE_ENABLE or HCCL_INTRA_ROCE_ENABLE don't set");
        return false;
    }
    if (strcmp(hcclIntraPcieEnable, "1") == 0 && strcmp(hcclIntraRoceEnable, "0") == 0) {
        OP_LOGD(K_INNER_DEBUG, "ENV HCCL_INTRA_PCIE_ENABLE = 1 and HCCL_INTRA_ROCE_ENABLE = 0, use layered solution.");
        return true;
    }
    OP_LOGD(K_INNER_DEBUG, "ENV HCCL_INTRA_PCIE_ENABLE != 1 or HCCL_INTRA_ROCE_ENABLE != 0, use default solution.");
    return false;
}

static uint64_t MoeDistributeCombineA2CalcTilingKey(gert::TilingContext *context, const bool isLayered,
                                                    const int32_t commQuantMode)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MoeDistributeCombineA2 calc tiling func.");

    uint32_t quantMode = TILINGKEY_NO_QUANT;  // A2 & A3
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

static ge::graphStatus MoeDistributeCombineA2CheckShapeAndSetTiling(gert::TilingContext *context,
                                                                    MoeDistributeCombineA2Info &info)
{
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(EXPAND_X_INDEX);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "expandXShape"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "expertIdShape"),
                    return GRAPH_FAILED);

    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "expandXshape", "invalid", "valid expandXshape"),
                    return GRAPH_FAILED);
    uint32_t h = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(h == 0 || h > MAX_HIDDEN_SIZE_A2 || h % BLOCK_SIZE_A2 != 0,
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "hiddensize", "invalid", "valid hiddensize"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "expertIdshape", "invalid", "valid expertIdshape"),
                    return GRAPH_FAILED);
    uint32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(bs == 0 || bs > MAX_BATCH_SIZE_A2,
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "batchsize", "invalid", "valid batchsize"),
                    return GRAPH_FAILED);
    uint32_t k = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(k == 0 || k > MAX_K_VALUE_A2, OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "k", "invalid", "valid k"),
                    return GRAPH_FAILED);

    info.bs = bs;
    info.k = k;
    info.h = h;

    OP_LOGD(K_INNER_DEBUG, "batchSize is %u", bs);
    OP_LOGD(K_INNER_DEBUG, "k is %u", k);
    OP_LOGD(K_INNER_DEBUG, "hiddenSize is %u", h);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineA2CheckAttrAndSetTiling(gert::TilingContext *context,
                                                                   MoeDistributeCombineA2Info &info,
                                                                   int32_t &commQuantMode, const bool isLayered)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "attrs"), return ge::GRAPH_FAILED);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto expertSharedTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_QUANT_MODE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr || *epWorldSizePtr <= 0 || *epWorldSizePtr > MAX_EP_WORLD_SIZE_A2 ||
                        *epWorldSizePtr % RANK_NUM_PER_NODE_A2 != 0,
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "epWorldSize", "invalid", "valid epWorldSize"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr || *epRankIdPtr < 0 || *epRankIdPtr >= *epWorldSizePtr,
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "epRankId", "invalid", "valid epRankId"), return GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr || *moeExpertNumPtr <= 0 || *moeExpertNumPtr > MAX_MOE_EXPERT_NUMS_A2 ||
                        *moeExpertNumPtr % *epWorldSizePtr != 0,
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "moeExpertNum", "invalid", "valid moeExpertNum"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(expertSharedTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "expertSharedType"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "sharedExpertRankNum"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "globalBs"), return GRAPH_FAILED);
    OP_TILING_CHECK(commQuantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "commQuantMode"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(!isLayered && *commQuantModePtr != static_cast<CommQuantModeType>(CommQuantMode::NON_QUANT),
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "commQuantMode", "invalid", "valid commQuantMode"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(isLayered && *commQuantModePtr != static_cast<CommQuantModeType>(CommQuantMode::NON_QUANT) &&
                        *commQuantModePtr != static_cast<CommQuantModeType>(CommQuantMode::INT8_QUANT),
                    OP_LOGE_FOR_INVALID_VALUE(K_OP_NAME, "commQuantMode", "invalid", "valid commQuantMode"),
                    return GRAPH_FAILED);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_OP_NAME, "xShape"),
                    return GRAPH_FAILED);
    int32_t globalBs = *epWorldSizePtr * expertIdStorageShape->GetStorageShape().GetDim(0);
    info.epWorldSize = *epWorldSizePtr;
    info.epRankId = *epRankIdPtr;
    info.expertSharedType = static_cast<uint32_t>(0);
    info.sharedExpertRankNum = static_cast<uint32_t>(0);
    info.moeExpertNum = *moeExpertNumPtr;
    if (*globalBsPtr == 0) {
        info.globalBs = static_cast<uint32_t>(globalBs);
    } else {
        info.globalBs = *globalBsPtr;
    }
    commQuantMode = *commQuantModePtr;
    PrintA2TilingDataInfo(info);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineA2GetPlatformInfoAndSetTiling(gert::TilingContext *context,
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

static ge::graphStatus MoeDistributeCombineA2TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MoeDistributeCombineA2 tiling func.");
    uint32_t batch_mode = 1U;
    auto ret = context->SetScheduleMode(batch_mode);
    MC2_CHECK_LOG_RET(nodeName, ret);
    MoeDistributeCombineA2TilingData *tilingData = context->GetTilingData<MoeDistributeCombineA2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITHOUT_REPORT(nodeName, "tilingData is nullptr."),
                    return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "MoeDistributeCombineA2 get tilingData.");
    MoeDistributeCombineA2Info &info = tilingData->moeDistributeCombineInfo;
    bool isLayered = MoeDistributeCombineA2IsLayered();
    int32_t commQuantMode = 0;
    OP_TILING_CHECK(
        MoeDistributeCombineA2CheckShapeAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "MoeDistributeCombineA2 CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        MoeDistributeCombineA2CheckAttrAndSetTiling(context, info, commQuantMode, isLayered) != ge::GRAPH_SUCCESS,
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
    context->SetAicpuBlockDim(mc2tiling::AICPU_NUM_BLOCKS_A2);
    uint64_t tilingKey = MoeDistributeCombineA2CalcTilingKey(context, isLayered, commQuantMode);
    context->SetTilingKey(tilingKey);
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITHOUT_REPORT(nodeName, "workSpaces is nullptr."),
                    return ge::GRAPH_FAILED);
    uint32_t userWorkspaceSize = static_cast<uint32_t>(info.moeExpertNum) * sizeof(uint32_t) * 2;
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + userWorkspaceSize;
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    uint32_t opType = 18; // batch write=18
    std::string algConfig = "BatchWrite=level0:fullmesh";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "Leave MoeDistributeCombineA2 tiling func.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTilingA2A3::MoeDistributeCombineA3TilingCheckAttr(gert::TilingContext *context,
                                                                                      uint32_t &commQuantMode)
{
    const char *nodeName = context->GetNodeName();
    MoeDistributeCombineTilingData *tilingData = context->GetTilingData<MoeDistributeCombineTilingData>();
    std::string groupEp = "";
    std::string groupTp = "";
    uint32_t tpWorldSize = 1;
    bool isShared = true;
    uint32_t localMoeExpertNum = 1;
    commQuantMode = 0U;

    // 获取入参属性
    OP_TILING_CHECK(
        GetAttrAndSetTilingData(context, *tilingData, nodeName, groupEp, groupTp, commQuantMode) == ge::GRAPH_FAILED,
        OP_LOGE_WITHOUT_REPORT(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);

    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(
        MoeDistributeCombineTilingHelper::TilingCheckMoeDistributeCombine(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(nodeName, "Tiling check params failed"), return ge::GRAPH_FAILED);

    // 检查属性的取值是否合法
    OP_TILING_CHECK(!CheckAttrs(context, *tilingData, nodeName, localMoeExpertNum),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "attr check failed."), return ge::GRAPH_FAILED);

    uint32_t sharedExpertRankNum = tilingData->moeDistributeCombineInfo.sharedExpertRankNum;
    uint32_t epRankId = tilingData->moeDistributeCombineInfo.epRankId;
    if (epRankId >= sharedExpertRankNum) { // 本卡为moe专家
        isShared = false;
    }

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(!CheckTensorShape(context, *tilingData, nodeName, isShared, localMoeExpertNum),
                    OP_LOGE_WITHOUT_REPORT(nodeName, "param dim check failed."), return ge::GRAPH_FAILED);

    // 校验win区大小
    OP_TILING_CHECK(CheckWinSize(context, tilingData, nodeName, localMoeExpertNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Tiling check window size failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkspace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Tiling set workspace Failed"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetHCommCfg(context, tilingData, groupEp, groupTp, tpWorldSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "SetHCommCfg failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTilingA2A3::MoeDistributeCombineA3TilingFuncImpl(gert::TilingContext *context)
{
    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    auto ret = context->SetScheduleMode(batch_mode);
    MC2_CHECK_LOG_RET(context->GetNodeName(), ret);

    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter MoeDistributeCombine Tiling func");
    MoeDistributeCombineTilingData *tilingData = context->GetTilingData<MoeDistributeCombineTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITHOUT_REPORT(nodeName, "tilingData is nullptr."),
                    return ge::GRAPH_FAILED);
    uint32_t commQuantMode;

    // 统一调用所有校验逻辑
    OP_TILING_CHECK(MoeDistributeCombineA3TilingCheckAttr(context, commQuantMode) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "CombineA3Tiling attr check failed."), return ge::GRAPH_FAILED);

    uint32_t quantMode = TILINGKEY_NO_QUANT;
    uint32_t layeredMode = TILINGKEY_TPL_MTE; // A2
    if (commQuantMode == INT8_COMM_QUANT) {
        quantMode = TILINGKEY_INT8_QUANT;
    }
    uint32_t archTag = (mc2tiling::GetNpuArch(context) == NpuArch::DAV_3510) ? TILINGKEY_TPL_A5 : TILINGKEY_TPL_A3;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(quantMode, layeredMode, archTag);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    uint32_t numBlocks = 1U;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData->moeDistributeCombineInfo.aivNum = aivNum;
    tilingData->moeDistributeCombineInfo.totalUbSize = ubSize;
    OP_LOGD(nodeName, "numBlocks = %u, aivNum = %lu, ubsize = %lu", numBlocks, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);

    return ge::GRAPH_SUCCESS;
}

bool MoeDistributeCombineTilingA2A3::CheckEpWorldSize(const char *nodeName, uint32_t epWorldSize)
{
    // 检验epWorldSize是否是8的倍数
    OP_TILING_CHECK(
        epWorldSize % 8 != 0,
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "ep_world_size", std::to_string(epWorldSize).c_str(), "divisible by 8"),
        return false);

    OP_TILING_CHECK((256 % epWorldSize != 0) && (epWorldSize % 144 != 0),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "ep_world_size", std::to_string(epWorldSize).c_str(),
                                              "one of [8, 16, 32, 64, 128, 144, 256, 288]"),
                    return false);

    return true;
}

ge::graphStatus MoeDistributeCombineTilingA2A3::MoeDistributeCombineTilingFuncImpl(std::string &socVersion,
                                                                                   gert::TilingContext *context)
{
    ge::graphStatus ret;
    if (socVersion == "Ascend910B") {
        ret = MoeDistributeCombineA2TilingFuncImpl(context);
    } else {
        ret = MoeDistributeCombineA3TilingFuncImpl(context);
    }

    return ret;
}

// A5 采用 MTE 方式通信，复用 A3 tiling
REGISTER_OPS_TILING_TEMPLATE(MoeDistributeCombine, MoeDistributeCombineTilingA2A3, 0);

} // namespace optiling
