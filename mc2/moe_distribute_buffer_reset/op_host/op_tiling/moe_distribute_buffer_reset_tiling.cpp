/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_buffer_reset_tiling.cpp
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
#include "mc2_log.h"
#include "mc2_hcom_topo_info.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/mc2_tiling_utils.h"
#include "../../op_kernel/moe_distribute_buffer_reset_tiling.h"

using namespace AscendC;
using namespace ge;
namespace optiling {
constexpr uint64_t INIT_TILINGKEY = 10000UL;

constexpr uint32_t INPUT_ELASTIC_INFO_INDEX = 0;
constexpr uint32_t ONE_DIM = 1;
constexpr uint32_t RANK_NUM_PER_SEVER = 16U;

constexpr uint32_t ATTR_GROUP_INDEX = 0;
constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_NEED_SYNC_INDEX = 2;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr uint64_t MB_SIZE = 1024UL * 1024;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
const char* A_INNER_DEBUG_BUFFER_RESET = "MoeDistributeBufferReset Tiling Debug";

const int MIN_WORLD_SIZE = 16;
const int MAX_WORLD_SIZE = 128;
const int NEED_SYNC = 1;
const int NO_NEED_SYNC = 0;

static void PrintTilingDataInfo(MoeDistributeBufferResetTilingData &tilingData)
{
    OP_LOGD(A_INNER_DEBUG_BUFFER_RESET, "worldSize is %u.", tilingData.moeDistributeBufferReset.worldSize);
    OP_LOGD(A_INNER_DEBUG_BUFFER_RESET, "needSync is %u.", tilingData.moeDistributeBufferReset.needSync);
    OP_LOGD(A_INNER_DEBUG_BUFFER_RESET, "aivNum is %u.", tilingData.moeDistributeBufferReset.aivNum);
    OP_LOGD(A_INNER_DEBUG_BUFFER_RESET, "totalUbSize is %lu.", tilingData.moeDistributeBufferReset.totalUbSize);
}

static ge::graphStatus CheckAndSetAttrs(const gert::TilingContext* context, MoeDistributeBufferResetTilingData &tilingData, 
    std::string &group)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
        OP_LOGE(A_INNER_DEBUG_BUFFER_RESET, "GetAttrs returned nullptr!"), return ge::GRAPH_FAILED);

    auto groupPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    auto worldSizePtr = attrs->GetAttrPointer<int>(ATTR_WORLD_SIZE_INDEX);
    auto needSyncPtr = attrs->GetAttrPointer<int>(ATTR_NEED_SYNC_INDEX);

    // 当前仅对必选属性进行校空
    OP_TILING_CHECK(groupPtr == nullptr,
        OP_LOGE(A_INNER_DEBUG_BUFFER_RESET, "groupPtr is null!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(worldSizePtr == nullptr,
        OP_LOGE(A_INNER_DEBUG_BUFFER_RESET, "worldSizePtr is null!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(needSyncPtr == nullptr,
        OP_LOGE(A_INNER_DEBUG_BUFFER_RESET, "needSyncPtr is null!"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*worldSizePtr < MIN_WORLD_SIZE) || (*worldSizePtr > MAX_WORLD_SIZE),
        OP_LOGE(A_INNER_DEBUG_BUFFER_RESET, "WorldSize is invalid, only support [%d, %d], but got worldSize=%d.",
            MIN_WORLD_SIZE, MAX_WORLD_SIZE, *worldSizePtr), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*worldSizePtr % RANK_NUM_PER_SEVER != 0),
        OP_LOGE(A_INNER_DEBUG_BUFFER_RESET, "WorldSize only support WorldSize divisible by 16, but got worldSize=%d.",
        *worldSizePtr), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*needSyncPtr != NO_NEED_SYNC) && (*needSyncPtr != NEED_SYNC),
        OP_LOGE(A_INNER_DEBUG_BUFFER_RESET, "needSync is invalid, only support %d or %d, but got needSync=%d.",
            NO_NEED_SYNC, NEED_SYNC, *needSyncPtr), return ge::GRAPH_FAILED);

    tilingData.moeDistributeBufferReset.worldSize = *worldSizePtr;
    tilingData.moeDistributeBufferReset.needSync = *needSyncPtr;

    OP_LOGD(A_INNER_DEBUG_BUFFER_RESET, "group = %s", groupPtr);
    group = string(groupPtr);

    const gert::StorageShape *elasticInfoStorageShape = context->GetInputShape(INPUT_ELASTIC_INFO_INDEX);
    OP_TILING_CHECK(elasticInfoStorageShape == nullptr,
        OP_LOGE(A_INNER_DEBUG_BUFFER_RESET, "input is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(elasticInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(A_INNER_DEBUG_BUFFER_RESET, "Input must be 1-dimension, but got %lu dim",
        elasticInfoStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(elasticInfoStorageShape->GetStorageShape().GetDim(0) != *worldSizePtr,
        OP_LOGE(A_INNER_DEBUG_BUFFER_RESET, "Input length must be ep worldsize:%d, but got %ld dim",
        *worldSizePtr, elasticInfoStorageShape->GetStorageShape().GetDim(0)), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(A_INNER_DEBUG_BUFFER_RESET, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

static void SetHcommCfg([[maybe_unused]] gert::TilingContext *context, MoeDistributeBufferResetTilingData *tiling, 
    const std::string group)
{
    OP_LOGD(A_INNER_DEBUG_BUFFER_RESET, "MoeDistributeBufferReset group = %s", group.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);
}

ge::graphStatus MoeDistributeBufferResetTilingFunc(gert::TilingContext* context)
{
    const char *nodeName = context->GetNodeName();
    MoeDistributeBufferResetTilingData *tilingData = context->GetTilingData<MoeDistributeBufferResetTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string group = "";

    // Function that get check and set Attrs
    OP_TILING_CHECK(CheckAndSetAttrs(context, *tilingData, group) != ge::GRAPH_SUCCESS,
                    OP_LOGE(A_INNER_DEBUG_BUFFER_RESET, "Check and set attributes failed!"),
                    return ge::GRAPH_FAILED);

    // Set WorkSpace
    OP_TILING_CHECK(SetWorkSpace(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG_BUFFER_RESET, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    // Set HcommCfg
    SetHcommCfg(context, tilingData, group);

    // Set TilingKey
    uint64_t tilingKey = INIT_TILINGKEY;
    OP_LOGD(A_INNER_DEBUG_BUFFER_RESET, "cur case tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    // Set blockDim
    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    tilingData->moeDistributeBufferReset.totalUbSize = ubSize;
    tilingData->moeDistributeBufferReset.aivNum = aivNum;
    OP_LOGD(A_INNER_DEBUG_BUFFER_RESET, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);

    PrintTilingDataInfo(*tilingData);
    OP_LOGD("MoeDistributeBufferReset", "tiling process finished successfully!!!");
    return ge::GRAPH_SUCCESS;
}

struct MoeDistributeBufferResetCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeBufferReset(gert::TilingParseContext *context) { 
    const gert::TilingParseContext* const_context = context;
    //避免未使用变量警告
    (void)const_context;
    (void)context;
	return ge::GRAPH_SUCCESS; 
}

IMPL_OP_OPTILING(MoeDistributeBufferReset)
    .Tiling(MoeDistributeBufferResetTilingFunc)
    .TilingParse<MoeDistributeBufferResetCompileInfo>(TilingParseForMoeDistributeBufferReset);
}  // end of namespace optiling