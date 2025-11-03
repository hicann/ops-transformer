/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
#include "tiling/mc2_tiling_utils.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
constexpr uint64_t INIT_TILINGKEY = 10000UL;

constexpr uint32_t INPUT_X_INDEX = 0U;
constexpr uint32_t TIME_OUT_INDEX = 1U;
constexpr uint32_t ELASTIC_INFO_INDEX = 2U;
constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t ELASTIC_METAINFO_OFFSET = 4U;
constexpr uint32_t RANK_LIST_NUM = 2U;

constexpr uint32_t ATTR_GROUP_INDEX = 0U;
constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 1U;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;
const char *A_INNER_DEBUG_BARRIER = "DistributeBarrier Tiling Debug";

const int MIN_WORLD_SIZE = 2;
const int MAX_WORLD_SIZE = 384;

static void PrintTilingDataInfo(DistributeBarrierTilingData &tilingData) {
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "worldSize is %u.",
            tilingData.distributeBarrierInfo.worldSize);
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "rankId is %u.",
            tilingData.distributeBarrierInfo.rankId);
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "aivNum is %u.",
            tilingData.distributeBarrierInfo.aivNum);
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "totalUbSize is %lu.",
            tilingData.distributeBarrierInfo.totalUbSize);
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "isInputTimeOut is %u.",
            tilingData.distributeBarrierInfo.isInputTimeOut);
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "isInputElasticInfo is %u.",
            tilingData.distributeBarrierInfo.isInputElasticInfo);
}
static bool CheckTimeOut(const gert::TilingContext *context) {
  const char *nodeName = context->GetNodeName();
  const gert::StorageShape *timeOutStorageShape = context->GetOptionalInputShape(TIME_OUT_INDEX);
  OP_TILING_CHECK(timeOutStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                  OP_LOGE(nodeName, "timeOut dim must be 1, but current dim num is %lu.",
                  timeOutStorageShape->GetStorageShape().GetDimNum()), return false);    
  auto timeOutDesc = context->GetOptionalInputDesc(TIME_OUT_INDEX);
  OP_TILING_CHECK(timeOutDesc == nullptr, OP_LOGE(nodeName, "timeOutDesc is null."), return false);
  OP_TILING_CHECK(timeOutDesc->GetDataType() != ge::DT_INT32, OP_LOGE(nodeName,
                  "timeOutDesc dataType is invalid, dataType should be int32, but is %s.",
                  Ops::Base::ToString(timeOutDesc->GetDataType()).c_str()), return false);    
  OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(timeOutDesc->GetStorageFormat())) ==
                  ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "timeOut format is invalid."), return false);
  const int64_t timeOutDim0 = timeOutStorageShape->GetStorageShape().GetDim(0);
  OP_TILING_CHECK(timeOutDim0 != 1,
                  OP_LOGE(nodeName, "timeOut's dim0 should be 1, current timeOut's dim0 is %ld.",
                  timeOutDim0), return false);
  return true;
}
static bool CheckElasticInfo(const gert::TilingContext *context, const uint32_t worldsize) {
  const char *nodeName = context->GetNodeName();
  const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(ELASTIC_INFO_INDEX);
  OP_TILING_CHECK(elasticInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                  OP_LOGE(nodeName, "elasticInfo dim must be 1, but current dim num is %lu.",
                  elasticInfoStorageShape->GetStorageShape().GetDimNum()), return false);    
  auto elasticInfoDesc = context->GetOptionalInputDesc(ELASTIC_INFO_INDEX);
  OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE(nodeName, "elasticInfoDesc is null."), return false);
  OP_TILING_CHECK(elasticInfoDesc->GetDataType() != ge::DT_INT32, OP_LOGE(nodeName,
                  "elasticInfoDesc dataType is invalid, dataType should be int32, but is %s.",
                  Ops::Base::ToString(elasticInfoDesc->GetDataType()).c_str()), return false);    
  OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(elasticInfoDesc->GetStorageFormat())) ==
                  ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "elasticInfo format is invalid."), return false);
  const int64_t elasticInfoDim0 = elasticInfoStorageShape->GetStorageShape().GetDim(0);
  OP_TILING_CHECK(elasticInfoDim0 != (ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * worldsize),
                  OP_LOGE(nodeName, "elasticInfo's dim0 not equal to 4 + 2 * epWorldSize, "
                  "elasticInfo's dim0 is %ld, epWorldSize is %u.",
                  elasticInfoDim0, worldsize), return false);       
  return true;                              
}                                                        
static bool CheckAndSetAttrs(const gert::TilingContext *context, DistributeBarrierTilingData &tilingData,
                             std::string &group, bool isInputTimeOut, bool isInputElasticInfo) {
  auto attrs = context->GetAttrs();
  OP_TILING_CHECK(attrs == nullptr, OPS_LOG_E(A_INNER_DEBUG_BARRIER, "GetAttrs returned nullptr!"), return false);

  auto groupPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
  auto worldSizePtr = attrs->GetAttrPointer<int>(ATTR_WORLD_SIZE_INDEX);

  // 当前仅对必选属性进行校空
  OP_TILING_CHECK(groupPtr == nullptr,
                  OPS_LOG_E(A_INNER_DEBUG_BARRIER, "groupPtr is null!"),
                  return false);
  OP_TILING_CHECK(worldSizePtr == nullptr,
                  OPS_LOG_E(A_INNER_DEBUG_BARRIER, "worldSizePtr is null!"),
                  return false);

  OP_TILING_CHECK((*worldSizePtr < MIN_WORLD_SIZE) || (*worldSizePtr > MAX_WORLD_SIZE),
                  OPS_LOG_E(A_INNER_DEBUG_BARRIER, "WorldSize is invalid, only support [%d, %d], but got worldSize=%d.",
                  MIN_WORLD_SIZE, MAX_WORLD_SIZE, *worldSizePtr), return false);

  tilingData.distributeBarrierInfo.worldSize = *worldSizePtr;

  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "group = %s", groupPtr);
  group = string(groupPtr);
  if (isInputTimeOut) {
    OP_TILING_CHECK(CheckTimeOut(context) == false,
                    OPS_LOG_E(A_INNER_DEBUG_BARRIER, "timeOut is invalid!"),
                    return false);
  }
  if (isInputElasticInfo) {     
    OP_TILING_CHECK(CheckElasticInfo(context, *worldSizePtr) == false,
                    OPS_LOG_E(A_INNER_DEBUG_BARRIER, "elasticInfo is invalid!"),
                    return false);
  }

  return true;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context) {
  size_t *workSpaces = context->GetWorkspaceSizes(1);
  OP_TILING_CHECK(workSpaces == nullptr,
                  OPS_LOG_E(A_INNER_DEBUG_BARRIER, "workSpaces is nullptr."),
                  return ge::GRAPH_FAILED);
  workSpaces[0] = SYSTEM_NEED_WORKSPACE;
  return ge::GRAPH_SUCCESS;
}

static void SetHcommCfg([[maybe_unused]] gert::TilingContext *context,
                        DistributeBarrierTilingData *tiling,
                        const std::string group) {
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "distributeBarrier group = %s",
            group.c_str());
  uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
  std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";

  AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType1,
                                               algConfigAllToAllStr);
  mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
  mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1);
}

ge::graphStatus DistributeBarrierTilingFunc(gert::TilingContext *context) {
  const char *nodeName = context->GetNodeName();
  DistributeBarrierTilingData *tilingData =
      context->GetTilingData<DistributeBarrierTilingData>();
  OP_TILING_CHECK(tilingData == nullptr,
                  OPS_LOG_E(nodeName, "tilingData is nullptr."),
                  return ge::GRAPH_FAILED);
  std::string group = "";
  bool isInputTimeOut = false;
  bool isInputElasticInfo = false;
  const gert::StorageShape *timeOutStorageShape = context->GetOptionalInputShape(TIME_OUT_INDEX);
  const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(ELASTIC_INFO_INDEX);
  isInputTimeOut = (timeOutStorageShape != nullptr);
  isInputElasticInfo = (elasticInfoStorageShape != nullptr);
  tilingData->distributeBarrierInfo.isInputTimeOut = isInputTimeOut;
  tilingData->distributeBarrierInfo.isInputElasticInfo = isInputElasticInfo;

  // Function that get check and set Attrs
  OP_TILING_CHECK(
      !CheckAndSetAttrs(context, *tilingData, group, isInputTimeOut, isInputElasticInfo),
      OPS_LOG_E(A_INNER_DEBUG_BARRIER, "Check and set attributes failed!"),
      return ge::GRAPH_FAILED);

  // Set WorkSpace
  OP_TILING_CHECK(
      SetWorkSpace(context) != ge::GRAPH_SUCCESS,
      OPS_LOG_E(A_INNER_DEBUG_BARRIER, "Tiling set workspace failed."),
      return ge::GRAPH_FAILED);

  // Set HcommCfg
  SetHcommCfg(context, tilingData, group);

  // Set TilingKey
  uint64_t tilingKey = INIT_TILINGKEY;
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "cur case tilingKey is %lu", tilingKey);
  context->SetTilingKey(tilingKey);

  // Set blockDim
  uint32_t blockDim = 1U;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
  uint64_t ubSize = 0UL;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
  context->SetBlockDim(blockDim);
  context->SetScheduleMode(1);  // 设置为batch mode模式，所有核同时启动
  tilingData->distributeBarrierInfo.totalUbSize = ubSize;
  tilingData->distributeBarrierInfo.aivNum = aivNum;
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "blockDim=%u, aivNum=%u, ubSize=%lu",
            blockDim, aivNum, ubSize);

  PrintTilingDataInfo(*tilingData);
  OPS_LOG_D("DistributeBarrier", "tiling process finished successfully!!!");
  return ge::GRAPH_SUCCESS;
}

struct DistributeBarrierCompileInfo {};
ge::graphStatus TilingParseForDistributeBarrier(
    gert::TilingParseContext *context) {
  const gert::TilingParseContext *const_context = context;
  //避免未使用变量警告
  (void)const_context;
  (void)context;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DistributeBarrier)
    .Tiling(DistributeBarrierTilingFunc)
    .TilingParse<DistributeBarrierCompileInfo>(TilingParseForDistributeBarrier);
}  // end of namespace optiling