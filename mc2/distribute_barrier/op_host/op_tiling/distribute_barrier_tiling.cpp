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
 * \file distribute_barrier_tiling.cc
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
#include <queue>
#include <string>
#include <vector>

#include "../../op_kernel/distribute_barrier_tiling.h"
// #include "graph/utils/op_desc_utils.h"   // 依赖 ge
#include "graph/utils/type_utils.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "distribute_barrier_tiling_helper.h"

using namespace AscendC;
using namespace ge;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
const char *A_INNER_DEBUG_BARRIER = "DistributeBarrier Tiling Debug";
static void PrintTilingDataInfo(DistributeBarrierTilingData &tilingData)
{
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "worldSize is %u.", tilingData.distributeBarrierInfo.worldSize);
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "rankId is %u.", tilingData.distributeBarrierInfo.rankId);
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "aivNum is %u.", tilingData.distributeBarrierInfo.aivNum);
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "totalUbSize is %lu.", tilingData.distributeBarrierInfo.totalUbSize);
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "isInputTimeOut is %u.", tilingData.distributeBarrierInfo.isInputTimeOut);
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "isInputElasticInfo is %u.", tilingData.distributeBarrierInfo.isInputElasticInfo);
}
static bool CheckTimeOut(const gert::TilingContext *context, const DistributeBarrierConfig &config)
{
    const char *nodeName = context->GetNodeName();
    const gert::StorageShape *timeOutStorageShape = context->GetOptionalInputShape(config.timeOutIndex);
    OP_TILING_CHECK(timeOutStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "timeOut",
                        (std::to_string(timeOutStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return false);
    auto timeOutDesc = context->GetOptionalInputDesc(config.timeOutIndex);
    OP_TILING_CHECK(timeOutDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "timeOutDesc"), return false);
    OP_TILING_CHECK(timeOutDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "timeOut",
                                              Ops::Base::ToString(timeOutDesc->GetDataType()).c_str(), "INT32"),
                    return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(timeOutDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "timeOut",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(timeOutDesc->GetStorageFormat()))).c_str(),
            "non-FRACTAL_NZ"),
        return false);
    const int64_t timeOutDim0 = timeOutStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(timeOutDim0 != 1,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "timeOut",
                                              (std::string("dim0=") + std::to_string(timeOutDim0)).c_str(), "dim0=1"),
                    return false);
    return true;
}
static bool CheckElasticInfo(const gert::TilingContext *context, const uint32_t worldsize,
                             const DistributeBarrierConfig &config)
{
    const char *nodeName = context->GetNodeName();
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
    OP_TILING_CHECK(elasticInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "elasticInfo",
                        (std::to_string(elasticInfoStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return false);
    auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
    OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "elasticInfoDesc"), return false);
    OP_TILING_CHECK(elasticInfoDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "elasticInfo",
                                              Ops::Base::ToString(elasticInfoDesc->GetDataType()).c_str(), "INT32"),
                    return false);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(elasticInfoDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "elasticInfo",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(elasticInfoDesc->GetStorageFormat())))
                .c_str(),
            "non-FRACTAL_NZ"),
        return false);
    const int64_t elasticInfoDim0 = elasticInfoStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(
        elasticInfoDim0 != (ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * worldsize),
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "elasticInfo", (std::string("dim0=") + std::to_string(elasticInfoDim0)).c_str(),
            (std::string("dim0=") + std::to_string(ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * worldsize)).c_str()),
        return false);
    return true;
}
static bool CheckAndSetAttrs(const gert::TilingContext *context, DistributeBarrierTilingData &tilingData,
                             std::string &group, bool isInputTimeOut, bool isInputElasticInfo,
                             const DistributeBarrierConfig &config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OPS_LOG_E(A_INNER_DEBUG_BARRIER, "GetAttrs returned nullptr!"), return false);
    NpuArch npuArch = mc2tiling::GetNpuArch(context);
    int64_t MAX_WORLD_SIZE = 0;
    if (npuArch == Ops::Base::DAV_3510) {
        MAX_WORLD_SIZE = MAX_WORLD_SIZE_A5;
    } else {
        MAX_WORLD_SIZE = MAX_WORLD_SIZE_A3;
    }

    auto groupPtr = attrs->GetAttrPointer<char>(config.attrGroupIndex);
    auto worldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrWorldSizeIndex);

    // 当前仅对必选属性进行校空
    OP_TILING_CHECK(groupPtr == nullptr, OPS_LOG_E(A_INNER_DEBUG_BARRIER, "groupPtr is null!"), return false);
    OP_TILING_CHECK(worldSizePtr == nullptr, OPS_LOG_E(A_INNER_DEBUG_BARRIER, "worldSizePtr is null!"), return false);

    OP_TILING_CHECK((*worldSizePtr < MIN_WORLD_SIZE) || (*worldSizePtr > MAX_WORLD_SIZE),
                    OPS_LOG_E(A_INNER_DEBUG_BARRIER,
                              "WorldSize is invalid, only support [%ld, %ld], "
                              "but got worldSize=%ld.",
                              MIN_WORLD_SIZE, MAX_WORLD_SIZE, *worldSizePtr),
                    return false);

    uint32_t worldSize = static_cast<uint32_t>(*worldSizePtr);
    tilingData.distributeBarrierInfo.worldSize = worldSize;

    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "group = %s", groupPtr);
    group = string(groupPtr);
    if (isInputTimeOut) {
        OP_TILING_CHECK(CheckTimeOut(context, config) == false, OPS_LOG_E(A_INNER_DEBUG_BARRIER, "timeOut is invalid!"),
                        return false);
    }
    if (isInputElasticInfo) {
        OP_TILING_CHECK(CheckElasticInfo(context, worldSize, config) == false,
                        OPS_LOG_E(A_INNER_DEBUG_BARRIER, "elasticInfo is invalid!"), return false);
    }

    return true;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OPS_LOG_E(A_INNER_DEBUG_BARRIER, "workSpaces is nullptr."),
                    return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetHcommCfg([[maybe_unused]] const gert::TilingContext *context,
                                   DistributeBarrierTilingData *tiling, const std::string group)
{
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "distributeBarrier group = %s", group.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE); // 通过不拉起AICPU，提高算子退出性能
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling) != 0,
                    OP_LOGE(context->GetNodeName(), "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1) != 0,
                    OP_LOGE(context->GetNodeName(), "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling1 failed"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void DistributeBarrierConfigIndexSet(DistributeBarrierConfig &config)
{
    config.xRefIndex = INPUT_X_INDEX;     // 根据disributeBarrierExtend算子原型标志位初始化xRef索引
    config.timeOutIndex = TIME_OUT_INDEX; // 根据disributeBarrierExtend算子原型标志位初始化timeOut索引
    config.elasticInfoIndex = ELASTIC_INFO_INDEX; // 根据disributeBarrierExtend算子原型标志位初始化elasticInfo索引
    config.xRefOutIndex = OUTPUT_X_INDEX; // 根据disributeBarrierExtend算子原型标志位设置xRefOut索引
    config.attrGroupIndex = ATTR_GROUP_INDEX; // 根据disributeBarrierExtend算子原型标志位初始化attrGroupIndex索引
    config.attrWorldSizeIndex =
        ATTR_WORLD_SIZE_INDEX; // 根据disributeBarrierExtend算子原型标志位设置attrWorldSizeIndex索引
    config.isMc2Context = false;
    return;
}

static ge::graphStatus CheckMc2Context(const gert::TilingContext *context, const char *nodeName,
                                       const DistributeBarrierConfig &config)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "contextShape"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "context",
                        (std::to_string(contextStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return ge::GRAPH_FAILED);
    int64_t contextDim0 = contextStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "context dim0 = %ld", contextDim0);

    auto contextDesc = context->GetInputDesc(config.contextIndex);
    OP_TILING_CHECK(contextDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "contextDesc"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "context",
                                              Ops::Base::ToString(contextDesc->GetDataType()).c_str(), "INT32"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(
            nodeName, "context",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat()))).c_str(),
            "non-FRACTAL_NZ"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DistributeBarrierExtendTilingFuncNew(gert::TilingContext *context,
                                                     const DistributeBarrierConfig &config)
{
    const char *nodeName = context->GetNodeName();
    DistributeBarrierTilingData *tilingData = context->GetTilingData<DistributeBarrierTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OPS_LOG_E(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string group = "";
    bool isInputTimeOut = false;
    bool isInputElasticInfo = false;
    const gert::StorageShape *timeOutStorageShape = context->GetOptionalInputShape(config.timeOutIndex);
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
    isInputTimeOut = (timeOutStorageShape != nullptr);
    isInputElasticInfo = (elasticInfoStorageShape != nullptr);
    tilingData->distributeBarrierInfo.isInputTimeOut = isInputTimeOut;
    tilingData->distributeBarrierInfo.isInputElasticInfo = isInputElasticInfo;
    tilingData->distributeBarrierInfo.isMc2Context = config.isMc2Context;

    // 检查context输入
    if (config.isMc2Context) {
        OP_TILING_CHECK(CheckMc2Context(context, nodeName, config) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "Tiling check context failed."), return ge::GRAPH_FAILED);
    }

    // Function that get check and set Attrs
    OP_TILING_CHECK(!CheckAndSetAttrs(context, *tilingData, group, isInputTimeOut, isInputElasticInfo, config),
                    OPS_LOG_E(A_INNER_DEBUG_BARRIER, "Check and set attributes failed!"), return ge::GRAPH_FAILED);

    // Set WorkSpace
    OP_TILING_CHECK(SetWorkSpace(context) != ge::GRAPH_SUCCESS,
                    OPS_LOG_E(A_INNER_DEBUG_BARRIER, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    if (!config.isMc2Context) {
        // Set HcommCfg
        OP_TILING_CHECK(SetHcommCfg(context, tilingData, group) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "Tiling SetHcommCfg failed."), return ge::GRAPH_FAILED);
    }
    // Set numBlocks
    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    context->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    tilingData->distributeBarrierInfo.totalUbSize = ubSize;
    tilingData->distributeBarrierInfo.aivNum = aivNum;
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "numBlocks=%u, aivNum=%u, ubSize=%lu", numBlocks, aivNum, ubSize);

    PrintTilingDataInfo(*tilingData);
    OPS_LOG_D("DistributeBarrier", "tiling process finished successfully!!!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DistributeBarrierTilingFunc(gert::TilingContext *context)
{
    DistributeBarrierConfig config;
    DistributeBarrierConfigIndexSet(config);

    return DistributeBarrierExtendTilingFuncNew(context, config);
}

struct DistributeBarrierCompileInfo {};
ge::graphStatus TilingParseForDistributeBarrier(gert::TilingParseContext *context)
{
    const gert::TilingParseContext *const_context = context;
    // 避免未使用变量警告
    (void)const_context;
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DistributeBarrier)
    .Tiling(DistributeBarrierTilingFunc)
    .TilingParse<DistributeBarrierCompileInfo>(TilingParseForDistributeBarrier);
} // end of namespace optiling
