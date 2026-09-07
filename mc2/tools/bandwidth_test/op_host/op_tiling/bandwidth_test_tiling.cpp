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
 * \file bandwidth_test_tiling.cpp
 * \brief
 */

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <string>
#include <vector>

#include "../../op_kernel/bandwidth_test_tiling.h"
#include "../../op_kernel/bandwidth_test_tiling_key.h"
#include "graph/utils/type_utils.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_templates_registry.h"

using namespace BandwidthTestTiling;
using namespace AscendC;
using namespace ge;

namespace optiling {

// ============ 输入输出索引常量 ============
constexpr uint32_t INPUT_X_INDEX = 1U;            // 输入x的索引
constexpr uint32_t INPUT_DST_RANK_ID_INDEX = 2U;  // 输入dst_rank_id的索引
constexpr uint32_t OUTPUT_Y_INDEX = 0U;           // 输出y的索引
constexpr uint32_t OUTPUT_RECEIVE_CNT_INDEX = 1U; // 输出receive_cnt的索引

// ============ 属性索引常量 ============
constexpr uint32_t ATTR_GROUP_INDEX = 0U;      // 通信组名称
constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 1U; // 世界大小
constexpr uint32_t ATTR_MAX_BS_INDEX = 2U;     // 最大batch size
constexpr uint32_t ATTR_MODE_INDEX = 3U;       // 执行模式
constexpr uint32_t ATTR_AIV_NUM_INDEX = 6U;    // AIV核数量

// ============ 其他常量 ============
constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U; // 系统预留workspace 16MB
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U; // AlltoAll操作类型
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512UL;

// ============ 参数范围约束 ============
const size_t MAX_GROUP_NAME_LENGTH = 128UL;
const int64_t MIN_WORLD_SIZE = 1;

// ============ Tiling Key ============
constexpr uint64_t TILING_KEY_BASE_A5 = 2ULL; // A5架构base key

// 打印TilingData调试信息
static void PrintTilingDataInfo(const char *nodeName, BandwidthTestTilingData &tilingData)
{
    OP_LOGD(nodeName, "data_size is %d.", tilingData.dataSize);
    OP_LOGD(nodeName, "world_size is %d.", tilingData.worldSize);
    OP_LOGD(nodeName, "max_bs is %d.", tilingData.maxBs);
    OP_LOGD(nodeName, "mode is %d.", tilingData.mode);
    OP_LOGD(nodeName, "aiv_num is %d.", tilingData.aivNum);
    OP_LOGD(nodeName, "total_ub_size is %lu.", tilingData.totalUbSize);
    OP_LOGD(nodeName, "total_win_size is %lu.", tilingData.totalWinSize);
}

// 校验属性指针非空
static ge::graphStatus CheckAttrPointersNotNull(gert::TilingContext *context, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    auto worldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_WORLD_SIZE_INDEX);
    auto maxBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_BS_INDEX);
    auto modePtr = attrs->GetAttrPointer<int64_t>(ATTR_MODE_INDEX);

    OP_TILING_CHECK(groupPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "group"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(worldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "worldSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(maxBsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "maxBs"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(modePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "mode"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 校验属性值范围合法性
static ge::graphStatus CheckAttrValuesValid(gert::TilingContext *context, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    auto worldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_WORLD_SIZE_INDEX);
    auto maxBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_BS_INDEX);
    auto modePtr = attrs->GetAttrPointer<int64_t>(ATTR_MODE_INDEX);

    OP_TILING_CHECK((strnlen(groupPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(groupPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE(nodeName, "group name is invalid, length should be in (0, %zu), but got length=%zu.",
                            MAX_GROUP_NAME_LENGTH, strnlen(groupPtr, MAX_GROUP_NAME_LENGTH)),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        (*worldSizePtr < MIN_WORLD_SIZE),
        OP_LOGE(nodeName, "worldSize is invalid, should be positive, but got worldSize=%ld.", *worldSizePtr),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*maxBsPtr <= 0),
                    OP_LOGE(nodeName, "maxBs is invalid, should be positive, but got maxBs=%ld.", *maxBsPtr),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*modePtr < 0) || (*modePtr > 1),
                    OP_LOGE(nodeName, "mode is invalid, should be 0 or 1, but got mode=%ld.", *modePtr),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 获取属性并填充TilingData
static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
                                               BandwidthTestTilingData &tilingData, std::string &group)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrPointersNotNull(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "attr pointers", "null", "not null"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrValuesValid(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check attr values valid failed."), return ge::GRAPH_FAILED);

    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    auto worldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_WORLD_SIZE_INDEX);
    auto maxBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_BS_INDEX);
    auto modePtr = attrs->GetAttrPointer<int64_t>(ATTR_MODE_INDEX);

    group = std::string(groupPtr);
    tilingData.worldSize = static_cast<int32_t>(*worldSizePtr);
    tilingData.maxBs = static_cast<int32_t>(*maxBsPtr);
    tilingData.mode = static_cast<int32_t>(*modePtr);

    OP_LOGD(nodeName, "group = %s", groupPtr);
    OP_LOGD(nodeName, "worldSize = %ld", *worldSizePtr);
    OP_LOGD(nodeName, "maxBs = %ld", *maxBsPtr);
    OP_LOGD(nodeName, "mode = %ld", *modePtr);

    return ge::GRAPH_SUCCESS;
}

// 校验输入tensor shape
// x: [bs, data_size], dst_rank_id: [bs]
static ge::graphStatus CheckInputTensorShape(gert::TilingContext *context, const char *nodeName,
                                             BandwidthTestTilingData &tilingData)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(INPUT_X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "x", (std::to_string(xStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);

    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(xDim0 <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "x dim0 (bs)", std::to_string(xDim0).c_str(), "positive"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        xDim1 <= 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "x dim1 (data_size)", std::to_string(xDim1).c_str(), "positive"),
        return ge::GRAPH_FAILED);

    tilingData.dataSize = static_cast<int32_t>(xDim1);

    const gert::StorageShape *dstRankIdStorageShape = context->GetInputShape(INPUT_DST_RANK_ID_INDEX);
    OP_TILING_CHECK(dstRankIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dst_rank_id"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dstRankIdStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "dst_rank_id",
                        (std::to_string(dstRankIdStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return ge::GRAPH_FAILED);

    const int64_t dstRankIdDim0 = dstRankIdStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(
        dstRankIdDim0 != xDim0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "dst_rank_id", std::to_string(dstRankIdDim0).c_str(),
                                              "The dim0 of dst_rank_id must be the same as that of x"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 校验输出tensor shape
// y: [maxBs * worldSize, dataSize], receive_cnt: [worldSize]
static ge::graphStatus CheckOutputTensorShape(gert::TilingContext *context, const char *nodeName,
                                              BandwidthTestTilingData &tilingData)
{
    int64_t worldSize = static_cast<int64_t>(tilingData.worldSize);
    int64_t maxBs = static_cast<int64_t>(tilingData.maxBs);
    int64_t dataSize = static_cast<int64_t>(tilingData.dataSize);

    const gert::StorageShape *yStorageShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "y"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        yStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "y", (std::to_string(yStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);

    const int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    const int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);
    const int64_t expectedYDim0 = maxBs * worldSize;

    OP_TILING_CHECK(yDim0 < expectedYDim0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "y dim0", std::to_string(yDim0).c_str(),
                                              (std::string(">= ") + std::to_string(expectedYDim0)).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yDim1 != dataSize,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "y dim1", std::to_string(yDim1).c_str(),
                                              ("=" + std::to_string(dataSize)).c_str()),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *receiveCntStorageShape = context->GetOutputShape(OUTPUT_RECEIVE_CNT_INDEX);
    OP_TILING_CHECK(receiveCntStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "receive_cnt"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(receiveCntStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "receive_cnt",
                        (std::to_string(receiveCntStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return ge::GRAPH_FAILED);

    const int64_t receiveCntDim0 = receiveCntStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(
        receiveCntDim0 != worldSize,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "receive_cnt", std::to_string(receiveCntDim0).c_str(),
                                              "The dim0 of receive_cnt must be the same as worldSize"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(gert::TilingContext *context, const char *nodeName,
                                        BandwidthTestTilingData &tilingData)
{
    OP_TILING_CHECK(CheckInputTensorShape(context, nodeName, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check input tensor shape failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckOutputTensorShape(context, nodeName, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check output tensor shape failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 根据NPU架构计算TilingKey
static uint64_t CalTilingKey()
{
    uint64_t archTag = TILING_KEY_BASE_A5;
    uint64_t tilingKey = GET_TPL_TILING_KEY(archTag);
    return tilingKey;
}

// 设置Workspace大小
static ge::graphStatus SetWorkSpace(gert::TilingContext *context, BandwidthTestTilingData *tilingData,
                                    const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + static_cast<size_t>(WORKSPACE_ELEMENT_OFFSET * aivNum * aivNum);

    return ge::GRAPH_SUCCESS;
}

// 检查HCCL窗口大小是否满足通信需求
// 需要的空间: world_size * max_bs * data_size * 2 * 2 (双缓冲 + 预留)
static ge::graphStatus CheckWinSize(const gert::TilingContext *context, BandwidthTestTilingData *tilingData,
                                    const char *nodeName)
{
    uint64_t hcclBufferSize = 0;
    uint64_t maxWindowSize = 0;

    OP_TILING_CHECK(mc2tiling::GetEpWinSize(context, nodeName, hcclBufferSize, maxWindowSize, ATTR_GROUP_INDEX,
                                            false) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get WinSize failed"), return ge::GRAPH_FAILED);

    uint64_t dataSize = static_cast<uint64_t>(tilingData->dataSize);
    uint64_t worldSize = static_cast<uint64_t>(tilingData->worldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData->maxBs);
    uint64_t actualSize = worldSize * maxBs * dataSize * 2UL * 2UL;

    if (actualSize > maxWindowSize) {
        OP_LOGE(nodeName,
                "HCCL_BUFFSIZE is too SMALL, maxBs=%lu, dataSize=%lu, worldSize=%lu, "
                "actualSize=%luMB, HCCL_BUFFSIZE=%luMB.",
                maxBs, dataSize, worldSize, actualSize / MB_SIZE + 1UL, hcclBufferSize / MB_SIZE);
        return ge::GRAPH_FAILED;
    }

    tilingData->totalWinSize = maxWindowSize;
    OP_LOGD(nodeName, "WindowSize = %lu", maxWindowSize);

    return ge::GRAPH_SUCCESS;
}

// Tiling校验主函数
static ge::graphStatus BandwidthTestTilingCheckAttr(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    BandwidthTestTilingData *tilingData = context->GetTilingData<BandwidthTestTilingData>();
    std::string group = "";

    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, group) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorShape(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check tensor shape failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWinSize(context, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check winsize failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Tiling主入口函数
// 执行流程: 参数校验 -> 窗口检查 -> 设置通信配置 -> 填充TilingData
static ge::graphStatus BandwidthTestTilingFuncImpl(gert::TilingContext *context)
{
    uint32_t batchMode = 1U;
    context->SetScheduleMode(batchMode);

    const char *nodeName = context->GetNodeName();
    BandwidthTestTilingData *tilingData = context->GetTilingData<BandwidthTestTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "Enter BandwidthTest tiling check func.");

    OP_TILING_CHECK(BandwidthTestTilingCheckAttr(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "BandwidthTest tiling check failed."), return ge::GRAPH_FAILED);

    uint64_t tilingKey = CalTilingKey();
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t platformAivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    auto attrs = context->GetAttrs();
    auto aivNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_AIV_NUM_INDEX);
    uint32_t aivNum = platformAivNum;
    if (*aivNumPtr > 0 && *aivNumPtr <= platformAivNum) {
        aivNum = static_cast<uint32_t>(*aivNumPtr);
    }

    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);

    context->SetBlockDim(numBlocks);
    tilingData->totalUbSize = ubSize;
    tilingData->aivNum = aivNum;

    OP_LOGD(nodeName, "numBlocks = %u, aivNum = %u, ubSize = %lu", numBlocks, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);

    return ge::GRAPH_SUCCESS;
}

struct BandwidthTestCompileInfo {};

static ge::graphStatus TilingParseForBandwidthTest(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BandwidthTest)
    .Tiling(BandwidthTestTilingFuncImpl)
    .TilingParse<BandwidthTestCompileInfo>(TilingParseForBandwidthTest);

} // namespace optiling
