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
 * \file moe_distribute_dispatch_tiling_arch22.cpp
 * \brief
 */

#include "moe_distribute_dispatch_v2_tiling_arch22.h"

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <sys/types.h>
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../../../op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "mc2_exception_dump.h"
#include "mc2_log_compat.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace {
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_COMM_ALG_INDEX = 13;
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
constexpr uint32_t ATTR_ZERO_EXPERT_NUM_INDEX = 14;
constexpr uint32_t ATTR_COPY_EXPERT_NUM_INDEX = 15;
constexpr uint32_t ATTR_CONST_EXPERT_NUM_INDEX = 16;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr uint32_t EXPERT_SCALES_INDEX = 4U;
constexpr uint32_t ELASTIC_INFO_INDEX = 5U;
constexpr uint32_t PERFORMANCE_INFO_INDEX = 6U;
constexpr uint32_t OUTPUT_EXPAND_SCALES_INDEX = 6U;
constexpr size_t SYSTEM_NEED_WORKSPACE = 16UL * 1024UL * 1024UL;

// A2定义
const char *K_INNER_DEBUG = "MoeDistributeDispatchV2";
constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
constexpr uint32_t BLOCK_SIZE_A2 = 32;
constexpr uint32_t MAX_K_VALUE_A2 = 16;
constexpr uint32_t MAX_HIDDEN_SIZE_A2 = 10240;
constexpr uint32_t MAX_HIDDEN_SIZE_A2_INT32 = 5120;
constexpr int32_t MAX_EP_WORLD_SIZE_A2 = 384;
constexpr int32_t MAX_EP_WORLD_SIZE_A2_LAYERED = 64;
constexpr int32_t MAX_MOE_EXPERT_NUMS_A2_FULLMESH = 1024;
constexpr int32_t MAX_MOE_EXPERT_NUMS_A2_HIERARCHY = 512;
constexpr uint32_t MAX_BATCH_SIZE_A2 = 256;
constexpr uint32_t LAYERED_MAX_BATCH_SIZE_A2 = 512;
constexpr uint64_t SCALE_PAYLOAD_SIZE_A2 = 32ULL;
constexpr size_t USER_WORKSPACE_A2 = 1UL * 1024UL * 1024UL; // moeExpertNum_ * sizeof(uint32_t) + epWorldSize_ * 2 * 32
constexpr uint64_t TILING_KEY_BASE_A2 = 2000000000;
constexpr uint64_t TILING_KEY_LAYERED_COMM_A2 = 100000000;
constexpr uint64_t INIT_TILINGKEY_A2 = 1000;
} // namespace

namespace optiling {

// 为了兼容老版本，在未配置commAlg参数时，读取环境变量；
// commAlg参数当前支持"fullmesh"和"hierarchy"两种，其余使用默认fullmesh不分层方案。
static ge::graphStatus MoeDistributeDispatchA2CheckCommAlg(const gert::TilingContext *context, bool &isLayered)
{
    isLayered = false;
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "attrs"), return ge::GRAPH_FAILED);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    if ((epWorldSizePtr != nullptr) && (*epWorldSizePtr <= RANK_NUM_PER_NODE_A2)) {
        isLayered = false;
        OP_LOGD(K_INNER_DEBUG, "epWorldSize <= 8, use default fullmesh algorithm.");
        return ge::GRAPH_SUCCESS;
    }
    auto commAlg = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_ALG_INDEX));
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
        OP_LOGE_WITH_INVALID_ATTR(K_INNER_DEBUG, "comm_alg", commAlg, "\"fullmesh\" or \"hierarchy\"");
        return GRAPH_FAILED;
    }
}

static uint64_t MoeDistributeDispatchA2CalcTilingKey(const gert::TilingContext *context, const bool isLayered)
{
    bool scaleMode = false;
    uint32_t commMode = TILINGKEY_TPL_MTE;

    if (isLayered) {
        commMode = TILINGKEY_TPL_AICPU;
    }
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    bool isScales = (scalesStorageShape != nullptr);
    if (isScales) {
        scaleMode = true;
    }
    uint64_t tilingKey =
        GET_TPL_TILING_KEY(TILINGKEY_NO_QUANT, scaleMode, TILINGKEY_NO_FULLMESH, commMode, TILINGKEY_TPL_A2);
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
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return ge::GRAPH_FAILED);
    uint64_t sizeofDtypeX = static_cast<uint64_t>(ge::GetSizeByDataType(xDesc->GetDataType()));
    uint64_t scalePayloadSize = xDesc->GetDataType() == ge::DT_INT32 ? SCALE_PAYLOAD_SIZE_A2 : 0ULL;
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
                    "* (h * dtypeBytes + 16 * Align8(k))) / 1MB + 8MB) = %luMB, actual HCCL_BUFFSIZE = %luMB, "
                    "moeExpertNum = %u, maxBs = %lu, h = %u, k = %u, dtypeBytes = %lu. "
                    "AlignY(x) = (x + Y - 1) / Y * Y. %s",
                    ops::CeilDiv(minHcclBuffSize, MB_SIZE), ops::CeilDiv(hcclBuffSize, MB_SIZE), info.moeExpertNum,
                    maxBs, info.h, info.k, sizeofDtypeX, HCCL_BUFFSIZE_HINT);
        }
    } else {
        constexpr uint64_t extraBuffSize = 2 * MB_SIZE; // 固定2M额外空间作为存储非数据信息的区域
        const uint64_t perTokenSize = info.h * sizeofDtypeX + scalePayloadSize;
        const uint64_t maxRecvTokenNum = maxBs * epWorldSize * std::min(localMoeExpertNum, info.k);
        minHcclBuffSize = BUFFER_NUM * (maxRecvTokenNum * perTokenSize + extraBuffSize);
        if (minHcclBuffSize > hcclBuffSize) {
            OP_LOGW(nodeName,
                    "HCCL_BUFFSIZE is too small, min required HCCL_BUFFSIZE (%lu * (maxBs * epWorldSize * "
                    "min(localMoeExpertNum, k) * (h * dtypeBytes + scalePayloadBytes) / 1MB + 2MB)) = %luMB, "
                    "actual HCCL_BUFFSIZE = %luMB, maxBs = %lu, epWorldSize = %u, localMoeExpertNum = %u, "
                    "k = %u, h = %u, dtypeBytes = %lu, scalePayloadBytes = %lu. %s",
                    BUFFER_NUM, ops::CeilDiv(minHcclBuffSize, MB_SIZE), ops::CeilDiv(hcclBuffSize, MB_SIZE), maxBs,
                    epWorldSize, localMoeExpertNum, info.k, info.h, sizeofDtypeX, scalePayloadSize, HCCL_BUFFSIZE_HINT);
        }
    }
    return ge::GRAPH_SUCCESS;
}

// a2函数
static ge::graphStatus MoeDistributeDispatchA2CheckAttrAndSetTiling(const gert::TilingContext *context,
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
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_ZERO_EXPERT_NUM_INDEX));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COPY_EXPERT_NUM_INDEX));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_CONST_EXPERT_NUM_INDEX));

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "expertIdShape"),
                    return GRAPH_FAILED);
    int32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);

    if (groupEpPtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "groupEpPtr");
        return ge::GRAPH_FAILED;
    }
    size_t groupEpLen = strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH);
    if (groupEpLen == 0 || groupEpLen == MAX_GROUP_NAME_LENGTH) {
        OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "groupEp", std::to_string(groupEpLen).c_str(),
                                  "non-empty and less than MAX_GROUP_NAME_LENGTH");
        return ge::GRAPH_FAILED;
    }
    int32_t maxEpWorldSizeA2 = MAX_EP_WORLD_SIZE_A2;
    if (isLayered) {
        maxEpWorldSizeA2 = MAX_EP_WORLD_SIZE_A2_LAYERED;
    }
    if (epWorldSizePtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "epWorldSizePtr");
        return GRAPH_FAILED;
    }
    if (*epWorldSizePtr <= 0 || *epWorldSizePtr > maxEpWorldSizeA2 ||
        ((*epWorldSizePtr > RANK_NUM_PER_NODE_A2) && (*epWorldSizePtr % RANK_NUM_PER_NODE_A2 != 0))) {
        OP_LOGE_FOR_INVALID_VALUE(
            K_INNER_DEBUG, "epWorldSize", std::to_string(*epWorldSizePtr).c_str(),
            (std::string("[1,") + std::to_string(maxEpWorldSizeA2) + "], and must be a multiple of " +
             std::to_string(RANK_NUM_PER_NODE_A2) + " when > " + std::to_string(RANK_NUM_PER_NODE_A2))
                .c_str());
        return GRAPH_FAILED;
    }
    if (epRankIdPtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "epRankIdPtr");
        return GRAPH_FAILED;
    }
    if (*epRankIdPtr < 0 || *epRankIdPtr >= *epWorldSizePtr) {
        std::string reason = "epRankId should be in [0, epWorldSize)";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_INNER_DEBUG, "epRankId", std::to_string(*epRankIdPtr).c_str(),
                                              reason.c_str());
        return GRAPH_FAILED;
    }
    int32_t maxMoeExpertNums = isLayered ? MAX_MOE_EXPERT_NUMS_A2_HIERARCHY : MAX_MOE_EXPERT_NUMS_A2_FULLMESH;
    if (moeExpertNumPtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "moeExpertNumPtr");
        return GRAPH_FAILED;
    }
    if (*moeExpertNumPtr <= 0 || *moeExpertNumPtr > maxMoeExpertNums || *moeExpertNumPtr % *epWorldSizePtr != 0) {
        OP_LOGE_FOR_INVALID_VALUE(
            K_INNER_DEBUG, "moeExpertNum", std::to_string(*moeExpertNumPtr).c_str(),
            (std::string("(0, ") + std::to_string(maxMoeExpertNums) + "], and divisible by epWorldSize").c_str());
        return GRAPH_FAILED;
    }
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "tpWorldSize"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(*tpWorldSizePtr >= 2,
                    OP_LOGE_WITH_INVALID_ATTR(K_INNER_DEBUG, "tp_world_size", std::to_string(*tpWorldSizePtr).c_str(),
                                              "less than 2"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "tpRankId"), return GRAPH_FAILED);
    OP_TILING_CHECK(expertSharedTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "expertSharedType"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "sharedExpertRankNum"),
                    return GRAPH_FAILED);
    if (quantModePtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "quantModePtr");
        return GRAPH_FAILED;
    }
    if (*quantModePtr != static_cast<uint64_t>(QuantModeA5::NON_QUANT) &&
        *quantModePtr != static_cast<uint64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "quantMode", std::to_string(*quantModePtr).c_str(),
                                  (std::to_string(static_cast<uint64_t>(QuantModeA5::NON_QUANT)) + " or " +
                                   std::to_string(static_cast<uint64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)))
                                      .c_str());
        return GRAPH_FAILED;
    }
    // int32场景下，不进行量化操作
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "xDesc"), return GRAPH_FAILED);
    if ((xDesc->GetDataType() == ge::DT_INT32) && (*quantModePtr != static_cast<int64_t>(QuantModeA5::NON_QUANT))) {
        OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "quantMode", std::to_string(*quantModePtr).c_str(),
                                  "0 when x dtype is INT32");
        return GRAPH_FAILED;
    }
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "globalBs"), return GRAPH_FAILED);
    if (expertTokenNumsTypePtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "expertTokenNumsTypePtr");
        return GRAPH_FAILED;
    }
    if (*expertTokenNumsTypePtr < 0 || *expertTokenNumsTypePtr > 1) {
        OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "expertTokenNumsType", std::to_string(*expertTokenNumsTypePtr).c_str(),
                                  "0 or 1");
        return GRAPH_FAILED;
    }
    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "zeroExpertNumPtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "copyExpertNumPtr"),
                    return ge::GRAPH_FAILED);
    if (constExpertNumPtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "constExpertNumPtr");
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(
        *constExpertNumPtr != 0,
        OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "constExpertNum", std::to_string(*constExpertNumPtr).c_str(), "0"),
        return ge::GRAPH_FAILED);

    // 判断是否满足uint32_t及其他限制
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = 0LL;
    int64_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;

    OP_LOGD(K_INNER_DEBUG, "zeroExpertNum=%ld,copyExpertNum= %ld, constExpertNum=%ld", zeroExpertNum, copyExpertNum,
            constExpertNum);
    if (zeroComputeExpertNum + moeExpertNum > INT32_MAX) {
        std::string valueStr =
            "zeroExpertNum=" + std::to_string(zeroExpertNum) + ", copyExpertNum=" + std::to_string(copyExpertNum) +
            ", constExpertNum=" + std::to_string(constExpertNum) + ", moeExpertNum=" + std::to_string(moeExpertNum);
        std::string reason = "The sum of zeroExpertNum, copyExpertNum, constExpertNum and moeExpertNum "
                             "should not exceed INT32_MAX";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(K_INNER_DEBUG,
                                              "zeroExpertNum, copyExpertNum, constExpertNum and moeExpertNum",
                                              valueStr.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    info.epWorldSize = *epWorldSizePtr;
    info.tpWorldSize = static_cast<uint32_t>(0);
    info.epRankId = *epRankIdPtr;
    info.tpRankId = static_cast<uint32_t>(0);
    info.expertSharedType = static_cast<uint32_t>(0);
    info.sharedExpertRankNum = static_cast<uint32_t>(0);
    info.moeExpertNum = *moeExpertNumPtr;
    info.quantMode = *quantModePtr;
    info.maxMoeExpertNum = maxMoeExpertNums;

    if (*globalBsPtr == 0) {
        info.globalBs = *epWorldSizePtr * bs;
    } else {
        info.globalBs = *globalBsPtr;
    }
    info.expertTokenNumsType = *expertTokenNumsTypePtr;
    info.zeroComputeExpertNum = static_cast<int32_t>(zeroComputeExpertNum);

    OP_LOGD(K_INNER_DEBUG, "quantMode=%d", info.quantMode);
    OP_LOGD(K_INNER_DEBUG, "globalBs=%d", info.globalBs);
    OP_LOGD(K_INNER_DEBUG, "expertTokenNumsType=%d", info.expertTokenNumsType);
    OP_LOGD(K_INNER_DEBUG, "expertSharedType=%d", info.expertSharedType);
    OP_LOGD(K_INNER_DEBUG, "sharedExpertRankNum=%d", info.sharedExpertRankNum);
    OP_LOGD(K_INNER_DEBUG, "moeExpertNum=%d", info.moeExpertNum);
    OP_LOGD(K_INNER_DEBUG, "epWorldSize=%d", info.epWorldSize);
    OP_LOGD(K_INNER_DEBUG, "epRankId=%d", info.epRankId);
    OP_LOGD(K_INNER_DEBUG, "zeroComputeExpertNum=%d", info.zeroComputeExpertNum);
    OP_LOGD(K_INNER_DEBUG, "maxMoeExpertNum=%d", info.maxMoeExpertNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA2CheckShapeAndSetTiling(const gert::TilingContext *context,
                                                                     MoeDistributeDispatchA2Info &info, bool isLayered)
{
    const char *nodeName = context->GetNodeName();
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    const gert::StorageShape *expertScalesStorageShape = context->GetOptionalInputShape(EXPERT_SCALES_INDEX);
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(ELASTIC_INFO_INDEX);
    const gert::StorageShape *performanceInfoStorageShape = context->GetOptionalInputShape(PERFORMANCE_INFO_INDEX);
    const gert::StorageShape *expandScalesStorageShape = context->GetOutputShape(OUTPUT_EXPAND_SCALES_INDEX);

    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "xShape"), return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "expertIdShape"),
                    return GRAPH_FAILED);
    OP_TILING_CHECK(isLayered && expertScalesStorageShape == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "expertScales"), return GRAPH_FAILED);
    OP_TILING_CHECK(isLayered && expandScalesStorageShape == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "expandScales"), return GRAPH_FAILED);
    if (elasticInfoStorageShape != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "elastic_info", "provided",
                                  "not supported on current architecture, must be nullptr");
        return GRAPH_FAILED;
    }
    if (xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS) {
        std::string dimStr = std::to_string(xStorageShape->GetStorageShape().GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(K_INNER_DEBUG, "x", dimStr.c_str(), "2D");
        return GRAPH_FAILED;
    }
    if (expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS) {
        std::string dimStr = std::to_string(expertIdStorageShape->GetStorageShape().GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(K_INNER_DEBUG, "expert_ids", dimStr.c_str(), "2D");
        return GRAPH_FAILED;
    }
    OP_LOGD(nodeName, "X dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "X dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdStorageShape->GetStorageShape().GetDim(1));

    uint32_t h = xStorageShape->GetStorageShape().GetDim(1);
    uint32_t xBs = xStorageShape->GetStorageShape().GetDim(0);
    uint32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);
    uint32_t k = expertIdStorageShape->GetStorageShape().GetDim(1);
    bool isScales = (scalesStorageShape != nullptr);

    // int32场景下
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return ge::GRAPH_FAILED);
    if (xDesc->GetDataType() == ge::DT_INT32) {
        if (!isScales) { // 必须传入scales
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "scales", "nullptr", "a valid tensor when x dtype is INT32");
            return ge::GRAPH_FAILED;
        }
        if (isLayered) { // 不支持分层方案
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "commAlg", "hierarchy", "fullmesh when x dtype is INT32");
            return ge::GRAPH_FAILED;
        }
        const gert::Shape &scalesShape = scalesStorageShape->GetStorageShape();
        // 数量上必须和token一致
        if ((scalesShape.GetDimNum() != ONE_DIM) || (scalesShape.GetDim(0) != static_cast<int64_t>(xBs))) {
            std::string shapeStr = Ops::Base::ToString(scalesShape);
            std::string expectedShape = "[" + std::to_string(xBs) + "]";
            std::string reason = "scales shape should be " + expectedShape + " when x dtype is INT32";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "scales", shapeStr.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(K_INNER_DEBUG, "attrs"), return ge::GRAPH_FAILED);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    uint32_t maxHiddenSizeA2 = xDesc->GetDataType() == ge::DT_INT32 ? MAX_HIDDEN_SIZE_A2_INT32 : MAX_HIDDEN_SIZE_A2;
    if (h % BLOCK_SIZE_A2 != 0 || h == 0 || h > maxHiddenSizeA2) {
        OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "h (hidden size)", std::to_string(h).c_str(),
                                  (std::string("positive, divisible by ") + std::to_string(BLOCK_SIZE_A2) +
                                   ", and at most " + std::to_string(maxHiddenSizeA2))
                                      .c_str());
        return GRAPH_FAILED;
    }
    uint32_t maxBatchSizeA2 = isLayered ? LAYERED_MAX_BATCH_SIZE_A2 : MAX_BATCH_SIZE_A2;
    if (bs == 0 || bs > maxBatchSizeA2) {
        OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "bs (batch size)", std::to_string(bs).c_str(),
                                  (std::string("[1, ") + std::to_string(maxBatchSizeA2) + "]").c_str());
        return GRAPH_FAILED;
    }

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_ZERO_EXPERT_NUM_INDEX));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COPY_EXPERT_NUM_INDEX));
    // 判断是否满足uint32_t及其他限制
    int32_t moeExpertNum = *moeExpertNumPtr;
    int32_t zeroExpertNum = static_cast<int32_t>(*zeroExpertNumPtr);
    int32_t copyExpertNum = static_cast<int32_t>(*copyExpertNumPtr);
    int32_t constExpertNum = 0;
    int32_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;
    if (k == 0 || k > MAX_K_VALUE_A2 || k > moeExpertNum + zeroComputeExpertNum) {
        OP_LOGE_FOR_INVALID_VALUE(
            K_INNER_DEBUG, "k (topK)", std::to_string(k).c_str(),
            (std::string("[1, ") + std::to_string(MAX_K_VALUE_A2) + "], and <= moeExpertNum + zeroComputeExpertNum")
                .c_str());
        return GRAPH_FAILED;
    }
    if (*quantModePtr == static_cast<uint64_t>(QuantModeA5::NON_QUANT) && isScales &&
        xDesc->GetDataType() != ge::DT_INT32) { // int32场景下，scales是必要的
        OP_LOGE_FOR_INVALID_VALUE(K_INNER_DEBUG, "scales", "provided", "must be nullptr when quantMode is NON_QUANT");
        return GRAPH_FAILED;
    }

    bool isActiveMask = (xActiveMaskStorageShape != nullptr);
    if (isActiveMask) {
        const int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(((xActiveMaskDimNums != ONE_DIM) && (xActiveMaskDimNums != TWO_DIMS)),
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "xActiveMask", std::to_string(xActiveMaskDimNums).c_str(),
                            "The shape dim of xActiveMask must be within the range {1D, 2D}."),
                        return GRAPH_FAILED);

        int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        if (xActiveMaskDim0 != static_cast<int64_t>(bs)) {
            std::string shapeStr = "[" + std::to_string(xActiveMaskDim0) + ", ...]";
            std::string reason = "xActiveMask's dim0 should equal expertIds's dim0 (" + std::to_string(bs) + ")";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "xActiveMask", shapeStr.c_str(), reason.c_str());
            return GRAPH_FAILED;
        }

        if ((xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) &&
            (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != static_cast<int64_t>(k))) {
            int64_t xActiveMaskDim1 = xActiveMaskStorageShape->GetStorageShape().GetDim(1);
            std::string shapeStr = "[..., " + std::to_string(xActiveMaskDim1) + "]";
            std::string reason =
                "xActiveMask's dim1 (when 2D) should equal expertIds's dim1 (" + std::to_string(k) + ")";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "xActiveMask", shapeStr.c_str(), reason.c_str());
            return GRAPH_FAILED;
        }
    }

    if (performanceInfoStorageShape != nullptr &&
        performanceInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM) {
        std::string dimStr = std::to_string(performanceInfoStorageShape->GetStorageShape().GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(K_INNER_DEBUG, "performanceInfo", dimStr.c_str(), "1D");
        return GRAPH_FAILED;
    }
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    if (performanceInfoStorageShape != nullptr &&
        performanceInfoStorageShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(*epWorldSizePtr)) {
        std::string shapeStr = "[" + std::to_string(performanceInfoStorageShape->GetStorageShape().GetDim(0)) + "]";
        std::string reason =
            "performanceInfo's dim0 should equal epWorldSize (" + std::to_string(*epWorldSizePtr) + ")";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(K_INNER_DEBUG, "performanceInfo", shapeStr.c_str(), reason.c_str());
        return GRAPH_FAILED;
    }

    info.isTokenMask = ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    info.isExpertMask = ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));
    info.bs = bs;
    info.k = k;
    info.h = h;

    OP_LOGD(K_INNER_DEBUG, "isTokenMask is %d", static_cast<int32_t>(info.isTokenMask));
    OP_LOGD(K_INNER_DEBUG, "isExpertMask is %d", static_cast<int32_t>(info.isExpertMask));
    OP_LOGD(K_INNER_DEBUG, "batchSize is %u", info.bs);
    OP_LOGD(K_INNER_DEBUG, "k is %u", info.k);
    OP_LOGD(K_INNER_DEBUG, "hiddenSize is %u", info.h);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA2GetPlatformInfoAndSetTiling(const gert::TilingContext *context,
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

uint64_t MoeDistributeDispatchV2TilingFuncA2A3::CalTilingKey(const bool isScales, const uint32_t quantMode,
                                                             const bool isSetFullMeshV2, bool isLayered)
{
    uint32_t templateDispatch = TILINGKEY_NO_FULLMESH;
    uint32_t tilingKeyQuantMode = quantMode;
    bool scaleMode = false;
    uint64_t tilingKey;
    uint32_t commMode = TILINGKEY_TPL_MTE;
    if (isScales) {
        scaleMode = true;
    }
    if (isSetFullMeshV2) {
        templateDispatch = TILINGKEY_ENABLE_FULLMESH;
    }
    if (isLayered) {
        templateDispatch = TILINGKEY_ENABLE_HIERARCHY;
    }
    tilingKey = GET_TPL_TILING_KEY(tilingKeyQuantMode, scaleMode, templateDispatch, commMode, TILINGKEY_TPL_A3);
    return tilingKey;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA2A3::MoeDistributeDispatchTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MoeDistributeDispatchA2 tiling func.");

    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    auto ret = context->SetScheduleMode(batch_mode);
    MC2_CHECK_LOG_RET(nodeName, ret);

    // 1. tilingData
    MoeDistributeDispatchA2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchA2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);
    MoeDistributeDispatchA2Info &info = tilingData->moeDistributeDispatchInfo;

    bool isLayered = false;
    OP_TILING_CHECK(MoeDistributeDispatchA2CheckCommAlg(context, isLayered) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "MoeDistributeDispatchA2 CheckCommAlg Failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        MoeDistributeDispatchA2CheckShapeAndSetTiling(context, info, isLayered) != ge::GRAPH_SUCCESS,
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
    uint32_t aicpuBlockDim = info.epWorldSize > RANK_NUM_PER_NODE_A2 ? mc2tiling::AICPU_NUM_BLOCKS_A2 : 1;
    context->SetAicpuBlockDim(aicpuBlockDim);

    uint64_t tilingKey = MoeDistributeDispatchA2CalcTilingKey(context, isLayered);
    context->SetTilingKey(tilingKey);
    // 2. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workSpaces"), return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + USER_WORKSPACE_A2;

    // 3. communication
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
    Mc2Exception::MC2GroupNameManager::GetInstance().SetGroupName(group);
#endif
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    std::string algConfig = MoeDistributeCombineA2GetAlgConfig(*epWorldSizePtr, isLayered);
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, static_cast<uint32_t>(18), algConfig); // opType=18 BatchWrite
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "mc2CcTilingConfig mc2InitTiling GetTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "mc2CcTilingConfig mc2CcTiling GetTiling failed"),
                    return ge::GRAPH_FAILED);

    OP_LOGI(nodeName, "Leave MoeDistributeDispatchA2 tiling func.");
    return ge::GRAPH_SUCCESS;
}

bool MoeDistributeDispatchV2TilingFuncA2A3::CheckTensorDataType(const gert::TilingContext *context,
                                                                const char *nodeName, const bool isScales,
                                                                const uint32_t quantMode, const bool isActiveMask,
                                                                const bool hasElasticInfo, const bool isPerformance,
                                                                DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    if ((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16)) {
        std::string dtypeStr = Ops::Base::ToString(xDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", dtypeStr.c_str(), "BF16 or FLOAT16");
        return false;
    }

    if (quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        if (dynamicScalesDesc == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesDesc");
            return false;
        }
        if (dynamicScalesDesc->GetDataType() != ge::DT_FLOAT) {
            std::string dtypeStr = Ops::Base::ToString(dynamicScalesDesc->GetDataType());
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "dynamicScales", dtypeStr.c_str(), "FLOAT");
            return false;
        }
    }
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        if (scalesDesc->GetDataType() != ge::DT_FLOAT) {
            std::string dtypeStr = Ops::Base::ToString(scalesDesc->GetDataType());
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales", dtypeStr.c_str(), "FLOAT");
            return false;
        }
    }
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    if (quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        if (expandXDesc->GetDataType() != ge::DT_INT8) {
            std::string dtypeStr = Ops::Base::ToString(expandXDesc->GetDataType());
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expandX", dtypeStr.c_str(), "INT8");
            return false;
        }
    } else {
        if (expandXDesc->GetDataType() != xDesc->GetDataType()) {
            std::string dtypeStr = Ops::Base::ToString(expandXDesc->GetDataType());
            std::string xDtypeStr = Ops::Base::ToString(xDesc->GetDataType());
            std::string reason = "expandX should have the same dtype as x";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "expandX", dtypeStr.c_str(), reason.c_str());
            return false;
        }
    }

    if (!CheckCommomOtherInputTensorDataType(context, nodeName, isActiveMask, hasElasticInfo, isPerformance, config)) {
        std::string reason = "CheckCommomOtherInputTensorDataType failed";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "other inputs", "", reason.c_str());
        return false;
    }
    if (!CheckCommomOutputTensorDataType(context, nodeName)) {
        std::string reason = "CheckCommomOutputTensorDataType failed";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "outputs", "", reason.c_str());
        return false;
    }

    return true;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA2A3::MoeDistributeDispatchV2TilingFunc(gert::TilingContext *context)
{
    std::string socVersion = mc2tiling::GetSocVersion(context);
    NpuArch npuArch = mc2tiling::GetNpuArch(context);
    ge::graphStatus ret;
    if (socVersion == "Ascend910B") {
        ret = MoeDistributeDispatchTilingFuncImpl(context);
    } else {
        DispatchV2Config config;
        config.xIndex = 0U;                   // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
        config.expertIdsIndex = 1U;           // 1: 根据dispatchV2算子原型标志位初始化expertIds索引
        config.scalesIndex = 2U;              // 2: 根据dispatchV2算子原型标志位初始化scales索引
        config.xActiveMaskIndex = 3U;         // 3: 根据dispatchV2算子原型标志位初始化xActiveMask索引
        config.expertScalesIndex = 4U;        // 4: 根据dispatchV2算子原型标志位初始化expertScales索引
        config.elasticInfoIndex = 5U;         // 5: 根据dispatchV2算子原型标志位初始化elasticInfo索引
        config.performanceInfoIndex = 6U;     // 6: 根据dispatchV2算子原型标志位初始化performanceInfo索引
        config.attrGroupEpIndex = 0;          // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
        config.attrEpWorldSizeIndex = 1;      // 1: 根据dispatchV2算子原型标志位初始化epWorldSize索引
        config.attrEpRankIdIndex = 2;         // 2: 根据dispatchV2算子原型标志位初始化epRankId索引
        config.attrMoeExpertNumIndex = 3;     // 3: 根据dispatchV2算子原型标志位初始化moeExpertNum索引
        config.attrGroupTpIndex = 4;          // 4: 根据dispatchV2算子原型标志位初始化groupTp索引
        config.attrTpWorldSizeIndex = 5;      // 5: 根据dispatchV2算子原型标志位初始化tpWorldSize索引
        config.attrTpRankIdIndex = 6;         // 6: 根据dispatchV2算子原型标志位初始化tpRankId索引
        config.attrExpertSharedTypeIndex = 7; // 7: 根据dispatchV2算子原型标志位初始化expertSharedType索引
        config.attrSharedExpertNumIndex = 8;  // 8: 根据dispatchV2算子原型标志位初始化sharedExpertNum索引
        config.attrSharedExpertRankNumIndex = 9; // 9: 根据dispatchV2算子原型标志位初始化sharedExpertRankNum索引
        config.attrQuantModeIndex = 10;          // 10: 根据dispatchV2算子原型标志位初始化quantMode索引
        config.attrGlobalBsIndex = 11;           // 11: 根据dispatchV2算子原型标志位初始化globalBs索引
        config.attrExpertTokenNumsTypeIndex = 12; // 12: 根据dispatchV2算子原型标志位初始化expertTokenNumType索引
        config.attrCommAlgIndex = 13;             // 13: 根据dispatchV2算子原型标志位初始化commAlg索引
        config.attrZeroExpertNumIndex = 14; // 14: 根据dispatchV2算子原型标志位初始化zeroExpertNumIndex索引
        config.attrCopyExpertNumIndex = 15; // 15: 根据dispatchV2算子原型标志位初始化copyExpertNumIndex索引
        config.attrConstExpertNumIndex = 16; // 16: 根据dispatchV2算子原型标志位初始化constExpertNumIndex索引
        config.isMc2Context = false;
        ret = MoeDistributeDispatchA3TilingFuncImplPublic(context, config);
    }
    return ret;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA2A3::CheckQuantModeMatchScales(gert::TilingContext *context,
                                                                                 const char *nodeName, bool isScales,
                                                                                 uint32_t quantMode,
                                                                                 DispatchV2Config &config)
{
    OP_TILING_CHECK(quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT),
                    OP_LOGE_WITH_INVALID_ATTR(nodeName, "quant_mode", std::to_string(quantMode).c_str(),
                                              "a supported non-static quantization mode"),
                    return ge::GRAPH_FAILED);
    if ((isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT))) ||
        ((!isScales) && (quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)))) {
        std::string valueStr =
            "isScales=" + std::to_string(static_cast<int32_t>(isScales)) + ", quantMode=" + std::to_string(quantMode);
        std::string reason = "Scales must not be provided when quantMode is NON_QUANT, "
                             "and must be provided when quantMode is STATIC_QUANT";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales and quantMode", valueStr.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA2A3::CheckCommAlgPtr(const char *commAlgPtr, const char *nodeName)
{
    if ((strlen(commAlgPtr) != 0) && (strcmp(commAlgPtr, "fullmesh_v1") != 0) &&
        (strcmp(commAlgPtr, "fullmesh_v2") != 0) && (strcmp(commAlgPtr, "hierarchy") != 0)) {
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "comm_alg", commAlgPtr,
                                  "\"hierarchy\", \"fullmesh_v1\" or \"fullmesh_v2\"");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA2A3::CheckQuantModePtr(const int64_t *quantModePtr,
                                                                         const char *nodeName)
{
    OP_TILING_CHECK(
        (*quantModePtr < static_cast<int64_t>(QuantModeA5::NON_QUANT)) ||
            (*quantModePtr > static_cast<int64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "quantMode", std::to_string(*quantModePtr).c_str(),
                                  std::string("[") + std::to_string(static_cast<int64_t>(QuantModeA5::NON_QUANT)) +
                                      "," + std::to_string(static_cast<int64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) +
                                      std::string("]").c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

struct MoeDistributeDispatchCompileInfo {};
static ge::graphStatus TilingParseForMoeDistributeDispatchV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchV2TilingFuncImpl(gert::TilingContext *context)
{
    MoeDistributeDispatchV2TilingFuncA2A3 impl;
    return impl.MoeDistributeDispatchV2TilingFunc(context);
}

IMPL_OP_OPTILING(MoeDistributeDispatchV2)
    .Tiling(MoeDistributeDispatchV2TilingFuncImpl)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatchV2);
} // namespace optiling
