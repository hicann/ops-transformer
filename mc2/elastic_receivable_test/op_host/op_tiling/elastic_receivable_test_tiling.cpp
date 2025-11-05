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
 * \file elastic_receivable_test_tiling.cc
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
#include "tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../op_kernel/elastic_receivable_test_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
constexpr uint32_t DST_RANK_INDEX = 0U;

constexpr uint32_t ONE_DIM = 1;

constexpr uint64_t INIT_TILINGKEY = 10000UL;

constexpr uint32_t ATTR_GROUP_INDEX = 0;
constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_RANK_NUM_INDEX = 2;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t AIV_NUM_USED = 6;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;

const int MIN_WORLD_SIZE = 2;
const int MAX_WORLD_SIZE = 128;
const int MIN_RANK_SIZE = 1;
const int MAX_RANK_SIZE = 16;
const int DIE_PER_RANK = 16;

static void PrintTilingDataInfo(const char* nodeName, ElasticReceivableTestTilingData &tilingData)
{
    OP_LOGD(nodeName, "worldSize is %u.", tilingData.elasticReceivableTestInfo.worldSize);
    OP_LOGD(nodeName, "rankNum is %u.", tilingData.elasticReceivableTestInfo.rankNum);
    OP_LOGD(nodeName, "rankId is %u.", tilingData.elasticReceivableTestInfo.rankId);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.elasticReceivableTestInfo.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.elasticReceivableTestInfo.totalUbSize);
}

static bool CheckTensorDim(const gert::TilingContext *context, const char *nodeName)
{
    const gert::StorageShape *dstRankStorageShape = context->GetInputShape(DST_RANK_INDEX);
    OP_TILING_CHECK(dstRankStorageShape == nullptr, OP_LOGE(nodeName, "dstRankShape is null."), return false);
    OP_TILING_CHECK(dstRankStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "dstRankShape dim must be 1, but current dim num is %lu.",
        dstRankStorageShape->GetStorageShape().GetDimNum()), return false);
    int64_t dstRankDim0 = dstRankStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(dstRankDim0 != DIE_PER_RANK,
        OP_LOGE(nodeName, "dstRankDim0 is invalid. Should be %d, but got dstRankDim0=%ld.", DIE_PER_RANK,
            dstRankDim0), return false);
    
    return true;
}

static ge::graphStatus TilingCheckInputTensor(gert::TilingContext *context, const char *nodeName)
{
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName),
        OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);

    auto dstRankDesc = context->GetInputDesc(DST_RANK_INDEX);
    OP_TILING_CHECK(dstRankDesc == nullptr, OP_LOGE(nodeName, "dstRankDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dstRankDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "dstRank dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(dstRankDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(dstRankDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "dstRank format is invalid."), return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static bool CheckAndSetAttrs(const char* nodeName, const gert::TilingContext* context,
                             ElasticReceivableTestTilingData &tilingData, std::string &group)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "GetAttrs returned nullptr!"), return false);

    auto groupPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    auto worldSizePtr = attrs->GetAttrPointer<int>(ATTR_WORLD_SIZE_INDEX);
    auto rankNumPtr = attrs->GetAttrPointer<int>(ATTR_RANK_NUM_INDEX);

    // 当前仅对必选属性进行校空
    OP_TILING_CHECK(groupPtr == nullptr, OP_LOGE(nodeName, "groupPtr is null!"), return false);
    OP_TILING_CHECK(worldSizePtr == nullptr, OP_LOGE(nodeName, "worldSizePtr is null!"), return false);
    OP_TILING_CHECK(rankNumPtr == nullptr, OP_LOGE(nodeName, "rankNumPtr is null!"), return false);

    OP_TILING_CHECK((*worldSizePtr < MIN_WORLD_SIZE) || (*worldSizePtr > MAX_WORLD_SIZE),
        OP_LOGE(nodeName, "WorldSize is invalid, only support [%d, %d], but got worldSize=%d.", 
            MIN_WORLD_SIZE, MAX_WORLD_SIZE, *worldSizePtr), return false);

    OP_TILING_CHECK((*worldSizePtr % DIE_PER_RANK != 0),
        OP_LOGE(nodeName, "WorldSize is invalid, only support WorldSize be a multiple of 16, but got worldSize=%d.", 
        *worldSizePtr), return false);

    OP_TILING_CHECK((*rankNumPtr != DIE_PER_RANK),
        OP_LOGE(nodeName, "rankSize is invalid, only support %d, but got rankSize=%d.", 
            DIE_PER_RANK, *rankNumPtr), return false);

    tilingData.elasticReceivableTestInfo.worldSize = *worldSizePtr;

    OP_TILING_CHECK((strnlen(groupPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
        (strnlen(groupPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
        OP_LOGE(nodeName, "group's length is invalid."), return false);

    tilingData.elasticReceivableTestInfo.rankNum = *rankNumPtr;

    OP_LOGD(nodeName, "group = %s", groupPtr);
    group = string(groupPtr);

    return true;
}

static ge::graphStatus SetWorkSpace(const char* nodeName, gert::TilingContext *context)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

static void SetHcommCfg(const char* nodeName, [[maybe_unused]] gert::TilingContext *context,
                        ElasticReceivableTestTilingData *tiling, const std::string group)
{
    OP_LOGD(nodeName, "ElasticReceivableTest group = %s", group.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1);
}

ge::graphStatus ElasticReceivableTestTilingFunc(gert::TilingContext* context)
{
    const char *nodeName = context->GetNodeName();
    ElasticReceivableTestTilingData *tilingData = context->GetTilingData<ElasticReceivableTestTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string group = "";

    OP_TILING_CHECK(
        TilingCheckInputTensor(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);

    // Function that get check and set Attrs
    OP_TILING_CHECK(!CheckAndSetAttrs(nodeName, context, *tilingData, group),
                    OP_LOGE(nodeName, "Check and set attributes failed!"),
                    return ge::GRAPH_FAILED);

    // Set WorkSpace
    OP_TILING_CHECK(SetWorkSpace(nodeName, context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    // Set HcommCfg
    SetHcommCfg(nodeName, context, tilingData, group);

    // Set TilingKey
    uint64_t tilingKey = INIT_TILINGKEY;
    OP_LOGD(nodeName, "cur case tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    // Set blockDim
    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = AIV_NUM_USED;
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    tilingData->elasticReceivableTestInfo.totalUbSize = ubSize;
    tilingData->elasticReceivableTestInfo.aivNum = aivNum;
    OP_LOGD(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);

    PrintTilingDataInfo(nodeName, *tilingData);
    OP_LOGD("ElasticReceivableTest", "tiling process finished successfully!!!");
    return ge::GRAPH_SUCCESS;
}

struct ElasticReceivableTestCompileInfo {};
ge::graphStatus TilingParseForElasticReceivableTest(gert::TilingParseContext *context) { 
    const gert::TilingParseContext* const_context = context;
    //避免未使用变量警告
    (void)const_context;
    (void)context;
	return ge::GRAPH_SUCCESS; 
}

IMPL_OP_OPTILING(ElasticReceivableTest)
    .Tiling(ElasticReceivableTestTilingFunc)
    .TilingParse<ElasticReceivableTestCompileInfo>(TilingParseForElasticReceivableTest);
}  // end of namespace optiling