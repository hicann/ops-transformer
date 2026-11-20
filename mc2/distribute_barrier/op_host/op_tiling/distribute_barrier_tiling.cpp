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
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <string>
#include <vector>

#include "../../op_kernel/distribute_barrier_tiling.h"
// #include "graph/utils/op_desc_utils.h"   // 依赖 ge
#include "graph/utils/type_utils.h"
#include "mc2/common/utils/mc2_hccl_tiling_report.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "distribute_barrier_tiling_helper.h"
#include "../../../common/op_kernel/mc2_moe_context.h"
#include "fallback/fallback.h"

using namespace AscendC;
using namespace ge;
using namespace Ops::Transformer::OpTiling;

namespace optiling {
const char* A_INNER_DEBUG_BARRIER = "DistributeBarrier Tiling Debug";
constexpr uint32_t BARRIER_UB_ALIGN = 32U;
constexpr uint32_t BARRIER_STATE_OFFSET = 512U;
constexpr uint64_t BARRIER_WIN_STATE_OFFSET = 512UL * 1024UL;
constexpr uint64_t BARRIER_STATE_WIN_OFFSET = 900UL * 1024UL;

static void PrintTilingDataInfo(DistributeBarrierTilingData& tilingData)
{
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "worldSize is %u.", tilingData.distributeBarrierInfo.worldSize);
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "rankId is %u.", tilingData.distributeBarrierInfo.rankId);
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "aivNum is %u.", tilingData.distributeBarrierInfo.aivNum);
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "totalUbSize is %lu.", tilingData.distributeBarrierInfo.totalUbSize);
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "isAuto is %u.", tilingData.distributeBarrierInfo.isAuto);
    if (tilingData.distributeBarrierInfo.isAuto) {
        OPS_LOG_D(A_INNER_DEBUG_BARRIER, "contextSize is %lu.", tilingData.distributeBarrierInfo.totalWinSize);
    }
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "isInputTimeOut is %u.", tilingData.distributeBarrierInfo.isInputTimeOut);
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "isInputElasticInfo is %u.", tilingData.distributeBarrierInfo.isInputElasticInfo);
}
static bool CheckTimeOut(const gert::TilingContext* context, const DistributeBarrierConfig& config)
{
    const char* nodeName = context->GetNodeName();
    const gert::StorageShape* timeOutStorageShape = context->GetOptionalInputShape(config.timeOutIndex);
    OP_TILING_CHECK(timeOutStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "timeOut",
                        (std::to_string(timeOutStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return false);
    auto timeOutDesc = context->GetOptionalInputDesc(config.timeOutIndex);
    OP_TILING_CHECK(
        timeOutDesc == nullptr,
        Mc2Error::ReportTilingFailure(nodeName, "timeOutDesc descriptor is absent while its input shape is present"),
        return false);
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
static bool CheckElasticInfo(const gert::TilingContext* context, const uint32_t worldsize,
                             const DistributeBarrierConfig& config)
{
    const char* nodeName = context->GetNodeName();
    const gert::StorageShape* elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
    OP_TILING_CHECK(elasticInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "elasticInfo",
                        (std::to_string(elasticInfoStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return false);
    auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
    OP_TILING_CHECK(elasticInfoDesc == nullptr,
                    Mc2Error::ReportTilingFailure(
                        nodeName, "elasticInfoDesc descriptor is absent while its input shape is present"),
                    return false);
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
static int64_t GetMaxWorldSize(const gert::TilingContext* context, const DistributeBarrierConfig& config, bool& isAuto)
{
    const auto* shape = config.isMc2Context ? context->GetInputShape(config.contextIndex) : nullptr;
    isAuto = shape != nullptr && shape->GetStorageShape().GetDimNum() == ONE_DIM &&
             shape->GetStorageShape().GetDim(0) % 2 != 0;
    NpuArch npuArch = mc2tiling::GetNpuArch(context);
    if (npuArch == Ops::Base::DAV_3510 || npuArch == Ops::Base::DAV_9201 || npuArch == Ops::Base::DAV_9202) {
        return isAuto ? Mc2Barrier::MAX_RANKS : MAX_WORLD_SIZE_A5;
    }
    return MAX_WORLD_SIZE_A3;
}

static bool CheckAndSetAttrs(const gert::TilingContext* context, DistributeBarrierTilingData& tilingData,
                             std::string& group, bool isInputTimeOut, bool isInputElasticInfo,
                             const DistributeBarrierConfig& config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
                    Mc2Error::ReportTilingFailure(context->GetNodeName(), "RuntimeAttrs holder is absent"),
                    return false);
    auto groupPtr = attrs->GetAttrPointer<char>(config.attrGroupIndex);
    auto worldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrWorldSizeIndex);

    // 当前仅对必选属性进行校空
    OP_TILING_CHECK(
        groupPtr == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "group", "Required attribute is absent"),
        return false);
    OP_TILING_CHECK(
        worldSizePtr == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "world_size", "Required attribute is absent"),
        return false);

    const int64_t MAX_WORLD_SIZE = GetMaxWorldSize(context, config, tilingData.distributeBarrierInfo.isAuto);
    OP_TILING_CHECK(
        (*worldSizePtr < MIN_WORLD_SIZE) || (*worldSizePtr > MAX_WORLD_SIZE),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "world_size", std::to_string(*worldSizePtr).c_str(),
            ("Expected [" + std::to_string(MIN_WORLD_SIZE) + ", " + std::to_string(MAX_WORLD_SIZE) + "]").c_str()),
        return false);

    uint32_t worldSize = static_cast<uint32_t>(*worldSizePtr);
    tilingData.distributeBarrierInfo.worldSize = worldSize;

    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "group = %s", groupPtr);
    group = string(groupPtr);
    OP_TILING_CHECK(group.empty() || group.size() >= 128,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "group",
                        ("Length=" + std::to_string(group.size()) + "; expected [1, 127]").c_str()),
                    return false);
    if (isInputTimeOut) {
        OP_TILING_CHECK(CheckTimeOut(context, config) == false,
                        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "timeOut is invalid!"), return false);
    }
    if (isInputElasticInfo) {
        OP_TILING_CHECK(CheckElasticInfo(context, worldSize, config) == false,
                        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "elasticInfo is invalid!"), return false);
    }

    return true;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext* context)
{
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(
        workSpaces == nullptr,
        Mc2Error::ReportTilingFailure(context->GetNodeName(), "Workspace size holder is absent; required one slot"),
        return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetHcommCfg([[maybe_unused]] const gert::TilingContext* context,
                                   DistributeBarrierTilingData* tiling, const std::string group)
{
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "distributeBarrier group = %s", group.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE); // 通过不拉起AICPU，提高算子退出性能
    return Mc2Error::GetValidatedHcclTiling(mc2CcTilingConfig, tiling->mc2InitTiling, tiling->mc2CcTiling1,
                                            context->GetNodeName());
}

static void DistributeBarrierConfigIndexSet(DistributeBarrierConfig& config)
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

static ge::graphStatus CheckMc2ContextShape(const gert::TilingContext* context, const char* nodeName,
                                            const DistributeBarrierConfig& config,
                                            DistributeBarrierTilingData& tilingData)
{
    const gert::StorageShape* contextStorageShape = context->GetInputShape(config.contextIndex);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "context",
                        (std::to_string(contextStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return ge::GRAPH_FAILED);
    int64_t contextDim0 = contextStorageShape->GetStorageShape().GetDim(0);
    tilingData.distributeBarrierInfo.totalWinSize = contextDim0;
    OP_LOGD(nodeName, "context dim0 = %ld", contextDim0);
    const uint32_t world = tilingData.distributeBarrierInfo.worldSize;
    const bool isAutoContext = tilingData.distributeBarrierInfo.isAuto;
    OP_TILING_CHECK(isAutoContext ? !Mc2Barrier::IsValidContextSize(contextDim0, world) :
                                    contextDim0 != Mc2Aclnn::MC2_CONTEXT_ELEMS,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        nodeName, "context", std::to_string(contextDim0).c_str(),
                        "context length must match the legacy layout or an automatic topology"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckMc2Context(const gert::TilingContext* context, const char* nodeName,
                                       const DistributeBarrierConfig& config, DistributeBarrierTilingData& tilingData)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    OP_TILING_CHECK(
        contextDesc == nullptr,
        Mc2Error::ReportTilingFailure(nodeName, "contextDesc descriptor is absent while its input shape is present"),
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

    return CheckMc2ContextShape(context, nodeName, config, tilingData);
}

constexpr uint64_t CalcDistributeBarrierWinSize(uint32_t aivNum)
{
    return BARRIER_WIN_STATE_OFFSET + BARRIER_STATE_WIN_OFFSET +
           static_cast<uint64_t>(aivNum) * BARRIER_UB_ALIGN * sizeof(float) * 2UL;
}

static ge::graphStatus SetContextBufferSize(gert::TilingContext* context, DistributeBarrierTilingData* tilingData,
                                            uint32_t numBlocks)
{
    auto& info = tilingData->distributeBarrierInfo;
    if (info.isAuto) {
        const uint32_t required = Mc2Barrier::AutoUbBytes(info.worldSize, numBlocks - 1U, info.isInputElasticInfo);
        OP_TILING_CHECK(info.totalUbSize < required,
                        Mc2Error::ReportTilingFailure(context->GetNodeName(), "Insufficient UB for Barrier buffers; "
                                                                              "required bytes=" +
                                                                                  std::to_string(required)),
                        return ge::GRAPH_FAILED);
    } else if (info.isMc2Context) {
        OP_TILING_CHECK(
            static_cast<uint64_t>(BARRIER_STATE_OFFSET) * info.worldSize > BARRIER_STATE_WIN_OFFSET,
            Mc2Error::ReportTilingFailure(context->GetNodeName(), "State region exceeds the flag window offset"),
            return ge::GRAPH_FAILED);
        info.totalWinSize = CalcDistributeBarrierWinSize(info.aivNum);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetLaunchConfig(gert::TilingContext* context, DistributeBarrierTilingData* tilingData)
{
    const char* nodeName = context->GetNodeName();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_TILING_CHECK(aivNum < 2,
                    Mc2Error::ReportTilingFailure(
                        nodeName, "Available AIV core count=" + std::to_string(aivNum) +
                                      "; required at least 2 because one core receives while the remaining cores send"),
                    return ge::GRAPH_FAILED);
    const uint32_t numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    OP_TILING_CHECK(numBlocks < 2,
                    Mc2Error::ReportTilingFailure(nodeName, "Barrier requires at least two launch blocks"),
                    return ge::GRAPH_FAILED);
    context->SetBlockDim(numBlocks);
    OP_TILING_CHECK(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    Mc2Error::ReportTilingFailure(nodeName, "Schedule mode output slot is absent"),
                    return ge::GRAPH_FAILED);
    tilingData->distributeBarrierInfo.totalUbSize = ubSize;
    tilingData->distributeBarrierInfo.aivNum = aivNum;
    OPS_LOG_D(A_INNER_DEBUG_BARRIER, "numBlocks=%u, aivNum=%u, ubSize=%lu", numBlocks, aivNum, ubSize);
    return SetContextBufferSize(context, tilingData, numBlocks);
}

ge::graphStatus DistributeBarrierExtendTilingFuncNew(gert::TilingContext* context,
                                                     const DistributeBarrierConfig& config)
{
    OP_TILING_CHECK(context == nullptr, Mc2Error::ReportTilingFailure("DistributeBarrier", "Tiling context is null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context->GetPlatformInfo() == nullptr,
                    Mc2Error::ReportTilingFailure(context->GetNodeName(), "Platform resource metadata is absent"),
                    return ge::GRAPH_FAILED);
    const char* nodeName = context->GetNodeName();
    DistributeBarrierTilingData* tilingData = context->GetTilingData<DistributeBarrierTilingData>();
    OP_TILING_CHECK(
        tilingData == nullptr,
        Mc2Error::ReportTilingFailure(nodeName, "Typed tiling buffer is absent or too small; required " +
                                                    std::to_string(sizeof(DistributeBarrierTilingData)) + " bytes"),
        return ge::GRAPH_FAILED);
    std::string group = "";
    const bool isInputTimeOut = context->GetOptionalInputShape(config.timeOutIndex) != nullptr;
    const bool isInputElasticInfo = context->GetOptionalInputShape(config.elasticInfoIndex) != nullptr;
    tilingData->distributeBarrierInfo.isInputTimeOut = isInputTimeOut;
    tilingData->distributeBarrierInfo.isInputElasticInfo = isInputElasticInfo;
    tilingData->distributeBarrierInfo.isMc2Context = config.isMc2Context;

    OP_TILING_CHECK(!CheckAndSetAttrs(context, *tilingData, group, isInputTimeOut, isInputElasticInfo, config),
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Check and set attributes failed!"),
                    return ge::GRAPH_FAILED);

    if (config.isMc2Context) {
        OP_TILING_CHECK(CheckMc2Context(context, nodeName, config, *tilingData) != ge::GRAPH_SUCCESS,
                        OP_LOGE_WITHOUT_REPORT(nodeName, "Tiling check context failed."), return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(SetWorkSpace(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Tiling set workspace failed."),
                    return ge::GRAPH_FAILED);

    if (!config.isMc2Context) {
        OP_TILING_CHECK(SetHcommCfg(context, tilingData, group) != ge::GRAPH_SUCCESS,
                        OP_LOGE_WITHOUT_REPORT(nodeName, "Tiling SetHcommCfg failed."), return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(SetLaunchConfig(context, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "Tiling set launch config failed."), return ge::GRAPH_FAILED);

    PrintTilingDataInfo(*tilingData);
    OPS_LOG_D("DistributeBarrier", "tiling process finished successfully!!!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DistributeBarrierTilingFunc(gert::TilingContext* context)
{
    DistributeBarrierConfig config;
    DistributeBarrierConfigIndexSet(config);

    return DistributeBarrierExtendTilingFuncNew(context, config);
}

struct DistributeBarrierCompileInfo {};

static ge::graphStatus SelectBarrierFormat(ge::DataType dtype, const char* group, int64_t worldSize,
                                           ge::AscendString& result)
{
    constexpr size_t dtypePrefixLength = sizeof("DT_") - 1U;
    std::string type = ge::TypeUtils::DataTypeToSerialString(dtype);
    if (type == "DT_UNDEFINED" || type.size() <= dtypePrefixLength) {
        return ge::GRAPH_FAILED;
    }
    type = type.substr(dtypePrefixLength);
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return std::tolower(c); });
    using SupportsStatic = bool (*)(const char*, int64_t);
    static auto supportsStatic =
        reinterpret_cast<SupportsStatic>(fallback::GetOpApiFuncAddr("BarrierSupportsStaticGraph"));
    const bool native = group != nullptr && worldSize >= MIN_WORLD_SIZE && worldSize <= MAX_WORLD_SIZE_A5 &&
                        supportsStatic != nullptr && supportsStatic(group, worldSize);
    const std::string json = "{\"input0\":{\"name\":\"x_ref\",\"format\":\"ND\",\"dtype\":\"" + type +
                             "\"},\"input1\":{\"name\":\"time_out\",\"format\":\"ND\",\"dtype\":\"int32\"},"
                             "\"input2\":{\"name\":\"elastic_info\",\"format\":\"ND\",\"dtype\":\"int32\"},"
                             "\"output0\":{\"name\":\"x_ref\",\"format\":\"ND\",\"dtype\":\"" +
                             type + "\"},\"fallback\":{\"enable\":\"" + (native ? "false" : "true") +
                             "\",\"unknownshape_enable\":\"true\"}}";
    result = ge::AscendString(json.c_str());
    OP_LOGD("DistributeBarrier", "Graph selection: world=%ld, native=%d", worldSize, native);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DistributeBarrierSelectFormat(const ge::Operator& op, ge::AscendString& result)
{
    ge::AscendString group;
    int64_t worldSize = 0;
    (void)op.GetAttr("group", group);
    (void)op.GetAttr("world_size", worldSize);
    return SelectBarrierFormat(op.GetInputDescByName("x_ref").GetDataType(), group.GetString(), worldSize, result);
}

static const OpCheckFuncHelper g_barrierSelectFormat(FUNC_OP_SELECT_FORMAT, "DistributeBarrier",
                                                     DistributeBarrierSelectFormat);

ge::graphStatus TilingParseForDistributeBarrier(gert::TilingParseContext* context)
{
    const gert::TilingParseContext* const_context = context;
    // 避免未使用变量警告
    (void)const_context;
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DistributeBarrier)
    .Tiling(DistributeBarrierTilingFunc)
    .TilingParse<DistributeBarrierCompileInfo>(TilingParseForDistributeBarrier);
} // end of namespace optiling
