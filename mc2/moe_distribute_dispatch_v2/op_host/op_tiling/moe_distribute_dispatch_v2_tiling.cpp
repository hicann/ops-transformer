/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_tiling.cpp
 * \brief
 */

#include <algorithm>
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

#include "moe_distribute_dispatch_tiling_v2.h"
#include "mc2_log_compat.h"
#include "mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_constant.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;
namespace {
constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
constexpr uint32_t X_INDEX = 0U;
constexpr uint32_t EXPERT_IDS_INDEX = 1U;
constexpr uint32_t SCALES_INDEX = 2U;
constexpr uint32_t X_ACTIVE_MASK_INDEX = 3U;
constexpr uint32_t EXPERT_SCALES_INDEX = 4U;
constexpr uint32_t OUTPUT_EXPAND_X_INDEX = 0U;
constexpr uint32_t OUTPUT_DYNAMIC_SCALES_INDEX = 1U;
constexpr uint32_t OUTPUT_ASSIST_INFO_INDEX = 2U;
constexpr uint32_t OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3U;
constexpr uint32_t OUTPUT_EP_RECV_COUNTS_INDEX = 4U;
constexpr uint32_t OUTPUT_TP_RECV_COUNTS_INDEX = 5U;
constexpr uint32_t OUTPUT_EXPAND_SCALES_INDEX = 6U;

constexpr uint32_t LAYERED_METAINFO_LIST_NUM = 2;
constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t HALF_NUM = 2; // cumsum最多只能用一半的核
constexpr uint32_t ONE_DIM = 1;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;

constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr int64_t MAX_EP_WORLD_SIZE_A3 = 768L; // 384 * 2
constexpr int64_t MAX_EP_WORLD_SIZE_A5 = 1024L;
constexpr int64_t MAX_EP_WORLD_SIZE_LAYERED = 256;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t BS_UPPER_BOUND = 512;
constexpr int64_t BS_UPPER_BOUND_A6 = 1024;
constexpr int64_t BS_UPPER_BOUND_LAYERED = 256;
constexpr int64_t FULLMESH_BS_UPPER_BOUND_A3 = 256;
constexpr int64_t FULLMESH_BS_UPPER_BOUND_A5 = BS_UPPER_BOUND;
constexpr uint32_t H_BASIC_BLOCK_LAYERED = 32;
constexpr uint32_t RANK_NUM_PER_NODE = 16U;

constexpr uint32_t SIZE_OF_HALF = 2;
constexpr uint32_t SIZE_ALIGN_256 = 256;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_SIZE = 512UL;
constexpr uint32_t ELASTIC_INFO_OFFSET = 4U;
constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
constexpr int64_t MOE_EXPERT_MAX_NUM_A5 = 2048;
constexpr int64_t MOE_EXPERT_MAX_NUM_LAYERED = 512;
constexpr int64_t LOCAL_EXPERT_MAX_SIZE = 2048;
constexpr int64_t LOCAL_EXPERT_MAX_SIZE_A5 = 4096;
constexpr int64_t K_MAX = 16;
constexpr int64_t FULLMESH_K_MAX_A5 = K_MAX;
constexpr int64_t FULLMESH_K_MAX_A3 = 12;
constexpr size_t SYSTEM_NEED_WORKSPACE = 16UL * 1024UL * 1024UL;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512;
// A5前缀和临时区位于recvCnt工作区之后，分为数据区和同步标记区
constexpr uint64_t A5_CUMSUM_WS_FLAG_OFFSET = 32UL * 1024UL; // 32*32*32=32KB
constexpr uint64_t A5_CUMSUM_WORKSPACE_MAX_CORE_NUM = 32U;
constexpr uint32_t A5_CUMSUM_MAX_CORE_NUM = 8U;
constexpr uint32_t SERIAL_MODE_BS_THRESHOLD = 32U;
constexpr uint32_t SERIAL_MODE_EP_WORLD_SIZE_THRESHOLD = 16U;
constexpr uint64_t UB_ALIGN_VAL = 32U;
constexpr uint32_t UB_ALIGN_INT32_NUM = UB_ALIGN_VAL / sizeof(int32_t);
constexpr uint32_t RANK_LIST_NUM = 2;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024; // Bytes
constexpr int64_t H_MIN = 1024;
constexpr int64_t H_MAX = 8192;
constexpr int64_t H_MAX_LAYERED = 7168;
constexpr uint64_t TRIPLE = 3;
constexpr uint64_t ASSIST_NUM_PER_A = 128;
constexpr uint64_t EVEN_ALIGN = 2;
constexpr int64_t ELASTIC_METAINFO_OFFSET = 4;
constexpr int64_t CEIL_ALIGN32 = 8;

constexpr uint64_t STATIC_SCALE_DIM_0 = 1;
constexpr uint64_t ONE_DIM_SCALE_COL_NUM = 1;
constexpr uint64_t MX_BLOCK_SIZE = 32U;
constexpr uint64_t PERGROUP_BLOCK_SIZE = 128U;
constexpr uint64_t HIF4_GROUP_SIZE = 64U;
} // namespace

namespace optiling {
static void PrintTilingDataInfo(const char* nodeName, MoeDistributeDispatchV2TilingData& tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeDispatchV2Info.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeDispatchV2Info.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeDispatchV2Info.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeDistributeDispatchV2Info.expertShardType);
    OP_LOGD(nodeName, "sharedExpertNum is %u.", tilingData.moeDistributeDispatchV2Info.sharedExpertNum);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    OP_LOGD(nodeName, "quantMode is %u.", tilingData.moeDistributeDispatchV2Info.quantMode);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeDispatchV2Info.globalBs);
    OP_LOGD(nodeName, "bs is %u.", tilingData.moeDistributeDispatchV2Info.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.moeDistributeDispatchV2Info.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.moeDistributeDispatchV2Info.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.moeDistributeDispatchV2Info.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.moeDistributeDispatchV2Info.totalUbSize);
    OP_LOGD(nodeName, "totalWinSizeEP is %lu.", tilingData.moeDistributeDispatchV2Info.totalWinSizeEp);
    OP_LOGD(nodeName, "hasElastic is %d.", tilingData.moeDistributeDispatchV2Info.hasElasticInfo);
    OP_LOGD(nodeName, "isPerformance is %d.", tilingData.moeDistributeDispatchV2Info.isPerformance);
    OP_LOGD(nodeName, "hasExpertScales is %d.", tilingData.moeDistributeDispatchV2Info.hasExpertScales);
    OP_LOGD(nodeName, "zeroComputeExpertNum is %d", tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum);
}

static bool CheckDynamicScalesDim(const gert::TilingContext* context, const char* nodeName, const uint32_t quantMode)
{
    const gert::StorageShape* dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_TILING_CHECK(dynamicScalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesShape"),
                    return false);
    if ((quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT))) {
        // quantMode 2: 1dim, the same in A2/A3/A5
        if (dynamicScalesStorageShape->GetStorageShape().GetDimNum() != DYNAMIC_SCALE_ONE_DIM_NUM) {
            std::string dimStr = std::to_string(dynamicScalesStorageShape->GetStorageShape().GetDimNum()) + "D";
            OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "dynamicScales", dimStr.c_str(),
                                         (std::to_string(DYNAMIC_SCALE_ONE_DIM_NUM) + "D").c_str());
            return false;
        }
        OP_LOGD(nodeName, "dynamicScales dim0 = %ld", dynamicScalesStorageShape->GetStorageShape().GetDim(0));
    } else {
        // MX/PERGROUP
        if (dynamicScalesStorageShape->GetStorageShape().GetDimNum() != DYNAMIC_SCALE_TWO_DIM_NUM) {
            std::string dimStr = std::to_string(dynamicScalesStorageShape->GetStorageShape().GetDimNum()) + "D";
            OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "dynamicScales", dimStr.c_str(),
                                         (std::to_string(DYNAMIC_SCALE_TWO_DIM_NUM) + "D").c_str());
            return false;
        }
        OP_LOGD(nodeName, "dynamicScales dim0=%ld, dim1=%ld", dynamicScalesStorageShape->GetStorageShape().GetDim(0),
                dynamicScalesStorageShape->GetStorageShape().GetDim(1));
    }
    return true;
}

static bool CheckScaleTensorDim(const gert::TilingContext* context, const char* nodeName, const bool isScales,
                                const uint32_t quantMode, DispatchV2Config& config)
{
    return MoeDistributeDispatchTilingHelper::CheckScaleTensorDim(context, nodeName, isScales, quantMode,
                                                                  config.scalesIndex);
}

// x, expertIds, scales维度校验
static bool CheckInputTensorDim(const gert::TilingContext* context, const char* nodeName, const bool isScales,
                                const uint32_t quantMode, const bool isLayered, DispatchV2Config& config)
{
    const gert::StorageShape* xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xShape"), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "xShape", std::to_string(xStorageShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of xShape must be 2D."),
                    return false);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "x dim0 = %ld", xDim0);
    OP_LOGD(nodeName, "x dim1 = %ld", xDim1);

    const gert::StorageShape* expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    if (isLayered) {
        const gert::StorageShape* expertScaleStorageShape = context->GetOptionalInputShape(config.expertScalesIndex);
        OP_TILING_CHECK(expertScaleStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertScaleShape"),
                        return false);
        auto expertscaleDimNum = expertScaleStorageShape->GetStorageShape().GetDimNum();
        if (expertscaleDimNum != TWO_DIMS) {
            std::string dimStr = std::to_string(expertscaleDimNum) + "D";
            OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expertScaleShape", dimStr.c_str(), "2D");
            return false;
        }
        const int64_t expertScalesDim0 = expertScaleStorageShape->GetStorageShape().GetDim(0);
        const int64_t expertScalesDim1 = expertScaleStorageShape->GetStorageShape().GetDim(1);
        OP_LOGD(nodeName, "expertScales dim0 = %ld", expertScalesDim0);
        OP_LOGD(nodeName, "expertScales dim1 = %ld", expertScalesDim1);
    }
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdShape"),
                    return false);
    OP_TILING_CHECK(
        expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "expertIdShape", std::to_string(expertIdStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of expertIdShape must be 2D."),
        return false);
    const int64_t expertIdDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdDim0);
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdDim1);
    OP_TILING_CHECK(!CheckScaleTensorDim(context, nodeName, isScales, quantMode, config),
                    OP_LOGE(nodeName, "isScale Input param shape is invalid."), return false);
    return true;
}

static bool CheckOneDimOutputTensor(const gert::TilingContext* context, const char* nodeName, const size_t index,
                                    const char* shapeName, const char* dimName)
{
    const gert::StorageShape* storageShape = context->GetOutputShape(index);
    OP_TILING_CHECK(storageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, shapeName), return false);
    OP_TILING_CHECK(storageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, shapeName, std::to_string(storageShape->GetStorageShape().GetDimNum()).c_str(),
                        (std::string("The shape dim of ") + shapeName + " must be 1D.").c_str()),
                    return false);
    OP_LOGD(nodeName, "%s dim0 = %ld", dimName, storageShape->GetStorageShape().GetDim(0));
    return true;
}

// expertX, assistInfo, expertTokenNums, epRecvCount, tpRecvCount维度校验
static bool CheckCommonOutputTensorDim(const gert::TilingContext* context, const char* nodeName,
                                       const uint32_t quantMode)
{
    const gert::StorageShape* expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandXShape"), return false);
    OP_TILING_CHECK(
        expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "expandXShape", std::to_string(expandXStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of expandXShape must be 2D."),
        return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    if ((quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
        (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))) {
        OP_TILING_CHECK(!CheckDynamicScalesDim(context, nodeName, quantMode),
                        OP_LOGE(nodeName, "CheckDynamicScalesDim failed."), return false);
    }

    return CheckOneDimOutputTensor(context, nodeName, OUTPUT_ASSIST_INFO_INDEX, "assistInfoShape",
                                   "assistInfoForCombine") &&
           CheckOneDimOutputTensor(context, nodeName, OUTPUT_EXPERT_TOKEN_NUMS_INDEX, "expertTokenNumsShape",
                                   "expertTokenNums") &&
           CheckOneDimOutputTensor(context, nodeName, OUTPUT_EP_RECV_COUNTS_INDEX, "epRecvCountShape", "epRecvCount") &&
           CheckOneDimOutputTensor(context, nodeName, OUTPUT_TP_RECV_COUNTS_INDEX, "tpRecvCountShape", "tpRecvCount");
}

static bool CheckActiveMaskTensorDim(const gert::TilingContext* context, const char* nodeName, DispatchV2Config& config)
{
    const gert::StorageShape* xStorageShape = context->GetInputShape(config.xIndex);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const gert::StorageShape* expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    const int64_t expertIdDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    const gert::StorageShape* xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
    OP_TILING_CHECK(xActiveMaskStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xActiveMask shape"),
                    return false);
    const int64_t xActiveMaskDimNum = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(
        ((xActiveMaskDimNum != ONE_DIM) && (xActiveMaskDimNum != TWO_DIMS)),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "xActiveMask", std::to_string(xActiveMaskDimNum).c_str(),
                                                 "The shape dim of xActiveMask must be within the range {1D, 2D}."),
        return false);
    OP_TILING_CHECK((xActiveMaskStorageShape->GetStorageShape().GetDim(0) != xDim0),
                    OP_LOGE(nodeName, "The input of xActiveMask dim0 = %ld is not equal to x dim0 = %ld.",
                            xActiveMaskStorageShape->GetStorageShape().GetDim(0), xDim0),
                    return false);
    OP_TILING_CHECK(
        ((xActiveMaskDimNum == TWO_DIMS) && (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != expertIdDim1)),
        OP_LOGE(nodeName, "The input of xActiveMask dim1 = %ld is not equal to expertId dim1 = %ld.",
                xActiveMaskStorageShape->GetStorageShape().GetDim(1), expertIdDim1),
        return false);
    return true;
}

static bool CheckOptionalInfoTensorDim(const gert::TilingContext* context, const char* nodeName,
                                       const bool hasElasticInfo, const bool isPerformance, DispatchV2Config& config)
{
    if (hasElasticInfo) {
        const gert::StorageShape* shape = context->GetOptionalInputShape(config.elasticInfoIndex);
        OP_TILING_CHECK(shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "elasticInfo"), return false);
        OP_TILING_CHECK(shape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "elasticInfo", std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
                            "The shape dim of elasticInfo must be 1D."),
                        return false);
        OP_LOGD(nodeName, "elasticInfo dim0 = %ld", shape->GetStorageShape().GetDim(0));
    }
    if (isPerformance) {
        const gert::StorageShape* shape = context->GetOptionalInputShape(config.performanceInfoIndex);
        OP_TILING_CHECK(shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "performanceInfo"), return false);
        OP_TILING_CHECK(shape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "performanceInfo", std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
                            "The shape dim of performanceInfo must be 1D."),
                        return false);
        OP_LOGD(nodeName, "performanceInfo dim0 = %ld", shape->GetStorageShape().GetDim(0));
    }
    return true;
}

static bool CheckTensorDim(const gert::TilingContext* context, const char* nodeName, const bool isScales,
                           const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
                           const bool isPerformance, const bool isLayered, DispatchV2Config& config)
{
    OP_TILING_CHECK(!CheckInputTensorDim(context, nodeName, isScales, quantMode, isLayered, config),
                    OP_LOGE(nodeName, "Input param shape is invalid."), return false);
    if (isActiveMask && !CheckActiveMaskTensorDim(context, nodeName, config)) {
        return false;
    }
    if (!CheckOptionalInfoTensorDim(context, nodeName, hasElasticInfo, isPerformance, config)) {
        return false;
    }

    OP_TILING_CHECK(!CheckCommonOutputTensorDim(context, nodeName, quantMode),
                    OP_LOGE(nodeName, "Output param shape is invalid."), return false);

    return true;
}

static ge::graphStatus CheckQuantModeAndScales(const gert::TilingContext* context, const char* nodeName, bool isScales,
                                               const uint32_t quantMode, DispatchV2Config& config)
{
    if (!isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))) {
        std::string reason = "scales must be provided when quantMode is STATIC_QUANT";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales", "not provided", reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (isScales && ((quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) ||
                     (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT_CLIP)))) {
        std::string reason = "scales must be nullptr when quantMode is MX_QUANT";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales", "provided", reason.c_str());
        return ge::GRAPH_FAILED;
    }
    auto xDesc = context->GetInputDesc(config.xIndex);
    if (xDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc");
        return ge::GRAPH_FAILED;
    }
    std::string xDtypeStr = Ops::Base::ToString(xDesc->GetDataType());
    if (!isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
        ((xDesc->GetDataType() == ge::DT_HIFLOAT8) || (xDesc->GetDataType() == ge::DT_FLOAT8_E5M2) ||
         (xDesc->GetDataType() == ge::DT_FLOAT8_E4M3FN) || (xDesc->GetDataType() == ge::DT_FLOAT4_E2M1) ||
         (xDesc->GetDataType() == ge::DT_FLOAT4_E1M2))) {
        std::string reason = "scales must be provided when quantMode is NON_QUANT and x dtype is " + xDtypeStr;
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales", "not provided", reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
        ((xDesc->GetDataType() == ge::DT_BF16) || (xDesc->GetDataType() == ge::DT_FLOAT16))) {
        std::string reason = "scales must be nullptr when quantMode is NON_QUANT and x dtype is " + xDtypeStr;
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales", "provided", reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool MoeDistributeDispatchV2TilingFuncBase::CheckCommomOutputTensorDataType(const gert::TilingContext* context,
                                                                            const char* nodeName)
{
    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "assistInfoDesc"), return false);
    if (assistInfoDesc->GetDataType() != ge::DT_INT32) {
        std::string dtypeStr = Ops::Base::ToString(assistInfoDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "assistInfoForCombine", dtypeStr.c_str(), "INT32");
        return false;
    }

    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertTokenNumsDesc"),
                    return false);
    if (expertTokenNumsDesc->GetDataType() != ge::DT_INT64) {
        std::string dtypeStr = Ops::Base::ToString(expertTokenNumsDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expertTokenNums", dtypeStr.c_str(), "INT64");
        return false;
    }

    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epRecvCountsDesc"),
                    return false);
    if (epRecvCountsDesc->GetDataType() != ge::DT_INT32) {
        std::string dtypeStr = Ops::Base::ToString(epRecvCountsDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "epRecvCounts", dtypeStr.c_str(), "INT32");
        return false;
    }

    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpRecvCountsDesc"),
                    return false);
    if (tpRecvCountsDesc->GetDataType() != ge::DT_INT32) {
        std::string dtypeStr = Ops::Base::ToString(tpRecvCountsDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "tpRecvCounts", dtypeStr.c_str(), "INT32");
        return false;
    }

    return true;
}

bool MoeDistributeDispatchV2TilingFuncBase::CheckCommomOtherInputTensorDataType(
    const gert::TilingContext* context, const char* nodeName, const bool isActiveMask, const bool hasElasticInfo,
    const bool isPerformance, DispatchV2Config& config)
{
    auto expertIdDesc = context->GetInputDesc(config.expertIdsIndex);
    if (expertIdDesc->GetDataType() != ge::DT_INT32) {
        std::string dtypeStr = Ops::Base::ToString(expertIdDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expertId", dtypeStr.c_str(), "INT32");
        return false;
    }

    if (isPerformance) {
        auto performanceInfoDesc = context->GetOptionalInputDesc(config.performanceInfoIndex);
        if (performanceInfoDesc->GetDataType() != ge::DT_INT64) {
            std::string dtypeStr = Ops::Base::ToString(performanceInfoDesc->GetDataType());
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "performanceInfoDesc", dtypeStr.c_str(), "INT64");
            return false;
        }
    }
    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        if (xActiveMaskDesc->GetDataType() != ge::DT_BOOL) {
            std::string dtypeStr = Ops::Base::ToString(xActiveMaskDesc->GetDataType());
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "xActiveMask", dtypeStr.c_str(), "BOOL");
            return false;
        }
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        if (elasticInfoDesc->GetDataType() != ge::DT_INT32) {
            std::string dtypeStr = Ops::Base::ToString(elasticInfoDesc->GetDataType());
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "elasticInfoDesc", dtypeStr.c_str(), "INT32");
            return false;
        }
    }

    return true;
}

template <typename T>
static bool CheckTensorDescFormat(const T* desc, const char* nodeName, const char* tensorName)
{
    auto format = static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat()));
    if (format == ge::FORMAT_FRACTAL_NZ) {
        std::string fmtStr = Ops::Base::ToString(format);
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, tensorName, fmtStr.c_str(), "ND");
        return false;
    }
    return true;
}

static bool CheckOptionalInputTensorFormat(const gert::TilingContext* context, const char* nodeName,
                                           const bool isActiveMask, const bool hasElasticInfo, const bool isPerformance,
                                           DispatchV2Config& config)
{
    if (isActiveMask &&
        !CheckTensorDescFormat(context->GetOptionalInputDesc(config.xActiveMaskIndex), nodeName, "xActiveMask")) {
        return false;
    }
    if (hasElasticInfo &&
        !CheckTensorDescFormat(context->GetOptionalInputDesc(config.elasticInfoIndex), nodeName, "elastic_info")) {
        return false;
    }
    if (isPerformance && !CheckTensorDescFormat(context->GetOptionalInputDesc(config.performanceInfoIndex), nodeName,
                                                "performance_info")) {
        return false;
    }
    return true;
}

static bool CheckOutputTensorFormat(const gert::TilingContext* context, const char* nodeName, const uint32_t quantMode)
{
    if (!CheckTensorDescFormat(context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX), nodeName, "expandX")) {
        return false;
    }
    if ((quantMode >= static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) &&
        !CheckTensorDescFormat(context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX), nodeName, "dynamicScales")) {
        return false;
    }
    return CheckTensorDescFormat(context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX), nodeName, "assistInfoForCombine") &&
           CheckTensorDescFormat(context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX), nodeName, "expertTokenNums") &&
           CheckTensorDescFormat(context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX), nodeName, "epRecvCounts") &&
           CheckTensorDescFormat(context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX), nodeName, "tpRecvCounts");
}

// 校验输入输出数据格式
static bool CheckTensorFormat(const gert::TilingContext* context, const char* nodeName, const bool isScales,
                              const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
                              const bool isPerformance, DispatchV2Config& config)
{
    if (!CheckTensorDescFormat(context->GetInputDesc(config.xIndex), nodeName, "x")) {
        return false;
    }
    if (!CheckTensorDescFormat(context->GetInputDesc(config.expertIdsIndex), nodeName, "expert_ids")) {
        return false;
    }
    if (isScales && !CheckTensorDescFormat(context->GetOptionalInputDesc(config.scalesIndex), nodeName, "scales")) {
        return false;
    }
    return CheckOptionalInputTensorFormat(context, nodeName, isActiveMask, hasElasticInfo, isPerformance, config) &&
           CheckOutputTensorFormat(context, nodeName, quantMode);
}

template <typename T>
static ge::graphStatus GetGroupEp(const T* attrs, const char* nodeName, std::string& groupEp, DispatchV2Config& config)
{
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);
    if (!config.isMc2Context) {
        auto groupEpPtr = attrs->template GetAttrPointer<char>(static_cast<int>(config.attrGroupEpIndex));
        OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                            (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                        OP_LOGE_WITH_INVALID_INPUT(nodeName, "groupEpPtr"), return ge::GRAPH_FAILED);
        groupEp = std::string(groupEpPtr);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCopyAndConstExpertAttrPointers(const int64_t* copyExpertNumPtr,
                                                           const int64_t* constExpertNumPtr, const char* nodeName)
{
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "copyExpertNumPtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "constExpertNumPtr"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext* context, const char* nodeName,
                                           std::string& groupEp, DispatchV2Config& config)
{
    auto attrs = context->GetAttrs();
    if (GetGroupEp(attrs, nodeName, groupEp, config) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrEpWorldSizeIndex);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrTpWorldSizeIndex);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrEpRankIdIndex);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrTpRankIdIndex);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(config.attrExpertSharedTypeIndex);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrSharedExpertNumIndex));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(config.attrSharedExpertRankNumIndex);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(config.attrMoeExpertNumIndex);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(config.attrQuantModeIndex);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrExpertTokenNumsTypeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));

    // 判空
    OP_TILING_CHECK(commAlgPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "commAlgPtr"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epWorldSizePtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpWorldSizePtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epRankIdPtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpRankIdPtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertShardPtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sharedExpertNumPtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "sharedExpertRankNumPtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "moeExpertNumPtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "quantModePtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertTokenNumsTypePtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "zeroExpertNumPtr"),
                    return ge::GRAPH_FAILED);
    if (CheckCopyAndConstExpertAttrPointers(copyExpertNumPtr, constExpertNumPtr, nodeName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupAttrParams(const gert::TilingContext* context, const char* nodeName,
                                            std::string& groupTp, bool& isLayered, DispatchV2Config& config)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrEpWorldSizeIndex);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrEpRankIdIndex);
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int64_t>(config.attrCommAlgIndex));
    int64_t epWorldSize = *epWorldSizePtr;
    NpuArch npuArch = mc2tiling::GetNpuArch(context);
    isLayered = !config.isRoce && (commAlgPtr != nullptr) && (strcmp(commAlgPtr, "hierarchy") == 0); // isLayered赋值
    std::string socVersion = mc2tiling::GetSocVersion(context);
    int64_t maxEpworldsize = 0;
    if (npuArch == Ops::Base::DAV_3510 || npuArch == Ops::Base::DAV_9201 || npuArch == Ops::Base::DAV_9202) {
        maxEpworldsize = isLayered ? MAX_EP_WORLD_SIZE_LAYERED : MAX_EP_WORLD_SIZE_A5;
    } else {
        maxEpworldsize = isLayered ? MAX_EP_WORLD_SIZE_LAYERED : MAX_EP_WORLD_SIZE_A3;
    }
    OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > maxEpworldsize),
                    OP_LOGE(nodeName, "epWorldSize should be in [%ld, %ld], but got %ld.", MIN_EP_WORLD_SIZE,
                            maxEpworldsize, epWorldSize),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((isLayered && (epWorldSize % RANK_NUM_PER_NODE != 0)), // 校验epWorldSize是否是16整数倍
                    OP_LOGE(nodeName, "epWorldSize should be %u Aligned, but got %ld.", RANK_NUM_PER_NODE, epWorldSize),
                    return ge::GRAPH_FAILED);
    // 参考A2实现: 单超场景(16卡)无跨超通信收益, hierarchy自动退化为fullmesh
    if (isLayered && (epWorldSize == RANK_NUM_PER_NODE)) {
        isLayered = false;
        OP_LOGI(nodeName, "epWorldSize is %ld in single supernode, degrade hierarchy to fullmesh.", epWorldSize);
    }
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
                    OP_LOGE(nodeName, "epRankId is invalid, only support [0, %ld), but got epRankId=%ld.", epWorldSize,
                            *epRankIdPtr),
                    return ge::GRAPH_FAILED);

    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrTpWorldSizeIndex);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "tpWorldSize is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*tpWorldSizePtr >= 2,
                    OP_LOGE(nodeName, "tpWorldSize >= 2 is not supported, got tpWorldSize=%ld.", *tpWorldSizePtr),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::CheckOtherAttrParams(const gert::TilingContext* context,
                                                                            const char* nodeName, bool& isSetFullMeshV2,
                                                                            DispatchV2Config& config)
{
    auto attrs = context->GetAttrs();
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(config.attrExpertSharedTypeIndex);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(config.attrQuantModeIndex);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrExpertTokenNumsTypeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    if (config.isMc2Context && !config.isRoce) {
        if (commAlgPtr != nullptr && strcmp(commAlgPtr, "hierarchy") == 0) {
            std::string reason = "hierarchy does not support mc2 context mode";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "comm_alg", commAlgPtr, reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (*expertShardPtr != 0) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "expertShardType", std::to_string(*expertShardPtr).c_str(), "0");
        return ge::GRAPH_FAILED;
    }

    if (CheckQuantModePtr(quantModePtr, nodeName) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK((*expertTokenNumsTypePtr != 0) && (*expertTokenNumsTypePtr != 1),
                    OP_LOGE(nodeName, "expertTokenNumsType should be 0 or 1, but got %s.",
                            std::to_string(*expertTokenNumsTypePtr).c_str()),
                    return ge::GRAPH_FAILED);
    // 这里只校验 A3
    if (CheckCommAlgPtr(commAlgPtr, nodeName) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    NpuArch npuArch = mc2tiling::GetNpuArch(context);
    if (npuArch == Ops::Base::DAV_2201 || npuArch == Ops::Base::DAV_3510) {
        isSetFullMeshV2 = commAlgPtr != nullptr && strcmp(commAlgPtr, "fullmesh_v2") == 0;
    } else {
        isSetFullMeshV2 = true;
    }
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 isSetFullMeshV2 = %d\n", isSetFullMeshV2);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::CheckCommAttrParams(const gert::TilingContext* context,
                                                                           const char* nodeName, std::string& groupTp,
                                                                           bool& isSetFullMeshV2, bool& isLayered,
                                                                           DispatchV2Config& config)
{
    // 校验 epWorldSize tpWorldSize epRankId tpRankId
    OP_TILING_CHECK(CheckGroupAttrParams(context, nodeName, groupTp, isLayered, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "CheckGroupAttrParams failed."), return ge::GRAPH_FAILED);
    // 校验 expertSharedType quantMode expertTokenNumsType commAlg
    OP_TILING_CHECK(CheckOtherAttrParams(context, nodeName, isSetFullMeshV2, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "CheckOtherAttrParams failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckZeroComputeExpertAttrs(const char* nodeName, int64_t zeroExpertNum, int64_t copyExpertNum,
                                                   int64_t constExpertNum, int64_t moeExpertNum)
{
    OP_TILING_CHECK((zeroExpertNum < 0),
                    OP_LOGE(nodeName, "zeroExpertNum less than 0, zeroExpertNum is %ld.", zeroExpertNum),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((copyExpertNum < 0),
                    OP_LOGE(nodeName, "copyExpertNum less than 0, copyExpertNum is %ld.", copyExpertNum),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((constExpertNum < 0),
                    OP_LOGE(nodeName, "constExpertNum less than 0, constExpertNum is %ld.", constExpertNum),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "zeroExpertNum=%ld,copyExpertNum= %ld, constExpertNum=%ld", zeroExpertNum, copyExpertNum,
            constExpertNum);
    OP_TILING_CHECK(
        zeroExpertNum + copyExpertNum + constExpertNum + moeExpertNum > INT32_MAX,
        OP_LOGE(nodeName,
                "zeroExpertNum[%ld] + copyExpertNum[%ld] + constExpertNum[%ld] + moeExpertNum[%ld] exceed INT32_MAX.",
                zeroExpertNum, copyExpertNum, constExpertNum, moeExpertNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertAttrParams(const gert::TilingContext* context, const char* nodeName, bool isLayered,
                                             DispatchV2Config& config)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrEpWorldSizeIndex);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrSharedExpertNumIndex));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(config.attrSharedExpertRankNumIndex);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(config.attrMoeExpertNumIndex);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));

    int64_t moeExpertNum = *moeExpertNumPtr;
    int64_t epWorldSize = *epWorldSizePtr;
    int64_t sharedExpertRankNum = *sharedExpertRankNumPtr;
    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;
    if (CheckZeroComputeExpertAttrs(nodeName, zeroExpertNum, copyExpertNum, constExpertNum, moeExpertNum) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (isLayered) {
        OP_TILING_CHECK((sharedExpertRankNum != 0),
                        OP_LOGE(nodeName,
                                "sharedExpertNum is invalid in hierarchy mode, only support 0, but got "
                                "sharedExpertNum=%ld, sharedExpertRankNum=%ld",
                                *sharedExpertNumPtr, sharedExpertRankNum),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((*sharedExpertNumPtr < 0) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM),
                        OP_LOGE(nodeName, "sharedExpertNum should be in [0, %ld], but got %ld.", MAX_SHARED_EXPERT_NUM,
                                *sharedExpertNumPtr),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            (sharedExpertRankNum < 0) || (sharedExpertRankNum >= epWorldSize),
            OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support [0, %ld), but got sharedExpertRankNum=%ld.",
                    epWorldSize, sharedExpertRankNum),
            return ge::GRAPH_FAILED);
    }
    bool isArch22 = mc2tiling::GetNpuArch(context) == Ops::Base::DAV_2201;
    int64_t moeExpertMaxNum =
        isLayered ? MOE_EXPERT_MAX_NUM_LAYERED : (isArch22 ? MOE_EXPERT_MAX_NUM : MOE_EXPERT_MAX_NUM_A5);
    OP_TILING_CHECK(
        (moeExpertNum <= 0) || (moeExpertNum > moeExpertMaxNum),
        OP_LOGE(nodeName, "moeExpertNum should be in (0, %ld], but got %ld.", moeExpertMaxNum, moeExpertNum),
        return ge::GRAPH_FAILED);

    if (!isArch22) {
        int64_t moeExpertRankNum = epWorldSize - sharedExpertRankNum;
        OP_TILING_CHECK(moeExpertRankNum <= 0,
                        OP_LOGE(nodeName, "moeExpertRankNum must be positive, but got %ld.", moeExpertRankNum),
                        return ge::GRAPH_FAILED);
        uint64_t localMoeExpertNum = static_cast<uint64_t>(moeExpertNum / moeExpertRankNum);
        OP_TILING_CHECK(localMoeExpertNum * epWorldSize * Mc2Kernel::STATE_OFFSET > Mc2Kernel::WIN_STATE_OFFSET,
                        OP_LOGE(nodeName,
                                "local status slots exceed one state region: localMoeExpertNum=%lu, epWorldSize=%ld, "
                                "STATE_OFFSET=%u, WIN_STATE_OFFSET=%lu.",
                                localMoeExpertNum, epWorldSize, Mc2Kernel::STATE_OFFSET, Mc2Kernel::WIN_STATE_OFFSET),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAttrParams(const gert::TilingContext* context, const char* nodeName, DispatchV2Config& config,
                                     MoeDistributeDispatchV2TilingData& tilingData)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrEpWorldSizeIndex);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrSharedExpertNumIndex));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(config.attrSharedExpertRankNumIndex);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(config.attrMoeExpertNumIndex);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrTpWorldSizeIndex);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrEpRankIdIndex);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrTpRankIdIndex);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(config.attrExpertSharedTypeIndex);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(config.attrQuantModeIndex);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrExpertTokenNumsTypeIndex));

    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;
    int64_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;
    tilingData.moeDistributeDispatchV2Info.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
    tilingData.moeDistributeDispatchV2Info.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDistributeDispatchV2Info.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDistributeDispatchV2Info.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.moeDistributeDispatchV2Info.expertShardType = static_cast<uint32_t>(*expertShardPtr);
    tilingData.moeDistributeDispatchV2Info.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.moeDistributeDispatchV2Info.expertTokenNumsType = static_cast<uint32_t>(*expertTokenNumsTypePtr);
    tilingData.moeDistributeDispatchV2Info.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    tilingData.moeDistributeDispatchV2Info.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum = static_cast<int32_t>(zeroComputeExpertNum);
    if ((tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum == 0U) &&
        (tilingData.moeDistributeDispatchV2Info.sharedExpertNum == 1U)) {
        tilingData.moeDistributeDispatchV2Info.sharedExpertNum = 0U;
    }
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 zeroComputeExpertNum = %d\n",
            tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::GetAttrAndSetTilingData(
    const gert::TilingContext* context, const char* nodeName, MoeDistributeDispatchV2TilingData& tilingData,
    std::string& groupEp, std::string& groupTp, bool& isSetFullMeshV2, bool& isLayered, DispatchV2Config& config)
{
    OP_TILING_CHECK(CheckAttrPtrNullptr(context, nodeName, groupEp, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "params check nullptr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckCommAttrParams(context, nodeName, groupTp, isSetFullMeshV2, isLayered, config) != ge::GRAPH_SUCCESS,
        OP_LOGE_LIBOPAPI_REPORT(nodeName, "CheckCommAttrParams failed; propagate lower-layer error"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckExpertAttrParams(context, nodeName, isLayered, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE_LIBOPAPI_REPORT(nodeName, "CheckExpertAttrParams failed; propagate lower-layer error"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetAttrParams(context, nodeName, config, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "set attr params failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool CheckSharedAttrs(const char* nodeName, const MoeDistributeDispatchV2TilingData& tilingData)
{
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    // 校验共享专家卡数和共享专家数是否只有一个为0
    OP_TILING_CHECK(
        (sharedExpertNum == 0U) && (sharedExpertRankNum > 0U),
        OP_LOGE(nodeName, "sharedExpertRankNum must be 0 when sharedExpertNum is 0, but got %u.", sharedExpertRankNum),
        return false);
    OP_TILING_CHECK(
        (sharedExpertNum > 0U) && (sharedExpertRankNum == 0U),
        OP_LOGE(nodeName, "sharedExpertNum must be 0 when sharedExpertRankNum is 0, but got %u.", sharedExpertNum),
        return false);

    if ((sharedExpertNum > 0U) && (sharedExpertRankNum > 0U)) {
        // 校验共享专家卡数能否整除共享专家数
        OP_TILING_CHECK(
            ((sharedExpertRankNum % sharedExpertNum) != 0U),
            OP_LOGE(nodeName,
                    "sharedExpertRankNum should be divisible by sharedExpertNum, but sharedExpertRankNum=%u, "
                    "sharedExpertNum=%u.",
                    sharedExpertRankNum, sharedExpertNum),
            return false);
    }

    return true;
}

static bool CheckCommAlgAttrs(const gert::TilingContext* context, const char* nodeName,
                              const MoeDistributeDispatchV2TilingData& tilingData, bool isSetFullMeshV2,
                              DispatchV2Config& config, bool isLayered)
{
    bool hasElasticInfo = tilingData.moeDistributeDispatchV2Info.hasElasticInfo;
    int32_t zeroComputeExpertNum = tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum;
    bool isExpertMask = tilingData.moeDistributeDispatchV2Info.isExpertMask;
    bool isPerformance = tilingData.moeDistributeDispatchV2Info.isPerformance;
    // 获取bs
    const gert::StorageShape* xStorageShape = context->GetInputShape(config.xIndex);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    uint32_t bs = static_cast<uint32_t>(xDim0);

    // 获取topk
    const gert::StorageShape* expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    uint32_t k = static_cast<uint32_t>(expertIdsDim1);

    int64_t fullMeshBsUpperBound = FULLMESH_BS_UPPER_BOUND_A3;
    if (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_3510) {
        fullMeshBsUpperBound = FULLMESH_BS_UPPER_BOUND_A5;
    } else if ((mc2tiling::GetNpuArch(context) == Ops::Base::DAV_9201) ||
               (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_9202)) {
        fullMeshBsUpperBound = BS_UPPER_BOUND_A6;
    }

    // 检查comm_alg和bs是否冲突
    OP_TILING_CHECK(isSetFullMeshV2 && (bs > fullMeshBsUpperBound),
                    OP_LOGE(nodeName, "When comm_alg is fullmesh_v2, bs should be between [1, %lld], but got %u.",
                            fullMeshBsUpperBound, bs),
                    return false);
    int64_t fullMeshKUpperBound =
        mc2tiling::GetNpuArch(context) == Ops::Base::DAV_2201 ? FULLMESH_K_MAX_A3 : FULLMESH_K_MAX_A5;
    // 检查comm_alg和topK是否冲突
    OP_TILING_CHECK(isSetFullMeshV2 && (k > fullMeshKUpperBound),
                    OP_LOGE(nodeName, "When comm_alg is fullmesh_v2, topK should be between [1, %lld], but got %u.",
                            fullMeshKUpperBound, k),
                    return false);
    // 校验动态缩容和分层不能同时启用
    OP_TILING_CHECK((isLayered && hasElasticInfo),
                    OP_LOGE(nodeName, "Cannot support elasticInfo when comm_alg is hierarchy"), return false);
    // 校验特殊专家和分层不能同时启用
    OP_TILING_CHECK((isLayered && (zeroComputeExpertNum > 0)),
                    OP_LOGE(nodeName, "Cannot support zeroComputeExpert when comm_alg is hierarchy"), return false);
    // 校验二维Mask和分层不能同时启用
    OP_TILING_CHECK((isLayered && isExpertMask),
                    OP_LOGE(nodeName, "Cannot support 2D xActiveMask when comm_alg is hierarchy"), return false);
    // 校验isPerformance和分层不能同时启用
    OP_TILING_CHECK((isLayered && isPerformance),
                    OP_LOGE(nodeName, "Cannot support isPerformance when comm_alg is hierarchy"), return false);

    return true;
}

static ge::graphStatus CheckGlobalBsAttr(const gert::TilingContext* context, const char* nodeName,
                                         MoeDistributeDispatchV2TilingData& tilingData, int64_t xDim0,
                                         uint32_t epWorldSize, DispatchV2Config& config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(config.attrGlobalBsIndex);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "globalBsPtr"),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n", *globalBsPtr, xDim0,
            epWorldSize);
    OP_TILING_CHECK((*globalBsPtr != 0) && ((*globalBsPtr < xDim0 * static_cast<int64_t>(epWorldSize)) ||
                                            ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)),
                    OP_LOGE(nodeName,
                            "globalBS is invalid, only support 0 or maxBs(maxBs is the configured bs capacity per "
                            "rank) * epWorldSize, but got globalBS=%ld, bs=%ld, epWorldSize=%u.",
                            *globalBsPtr, xDim0, epWorldSize),
                    return ge::GRAPH_FAILED);
    int64_t bsUpperBound = BS_UPPER_BOUND;
    if ((mc2tiling::GetNpuArch(context) == Ops::Base::DAV_9201) ||
        (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_9202)) {
        bsUpperBound = BS_UPPER_BOUND_A6;
    }
    OP_TILING_CHECK(mc2tiling::GetNpuArch(context) != Ops::Base::DAV_2201 &&
                        *globalBsPtr > static_cast<int64_t>(epWorldSize) * bsUpperBound,
                    OP_LOGE(nodeName,
                            "Configured maxBs per rank must not exceed %ld, but got "
                            "globalBs=%ld, epWorldSize=%u.",
                            bsUpperBound, *globalBsPtr, epWorldSize),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.globalBs =
        (*globalBsPtr == 0) ? static_cast<uint32_t>(xDim0) * epWorldSize : static_cast<uint32_t>(*globalBsPtr);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrs(const gert::TilingContext* context, const char* nodeName,
                                  MoeDistributeDispatchV2TilingData& tilingData, uint32_t& localMoeExpertNum,
                                  bool isActiveMask, bool isSetFullMeshV2, bool isLayered, DispatchV2Config& config)
{
    uint32_t epWorldSize = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeDispatchV2Info.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    OP_TILING_CHECK(!CheckSharedAttrs(nodeName, tilingData),
                    OP_LOGE(nodeName, "Check shared expert related attributes failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckCommAlgAttrs(context, nodeName, tilingData, isSetFullMeshV2, config, isLayered),
                    OP_LOGE(nodeName, "Check comm_alg related attributes failed."), return ge::GRAPH_FAILED);
    // 校验moe专家数量能否均分给多机
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
                    OP_LOGE(nodeName,
                            "moeExpertNum should be divisible by (epWorldSize - sharedExpertRankNum), "
                            "but moeExpertNum=%u, epWorldSize=%u, sharedExpertRankNum=%u.",
                            moeExpertNum, epWorldSize, sharedExpertRankNum),
                    return ge::GRAPH_FAILED);
    int64_t localExpertMaxSize =
        (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_2201) ? LOCAL_EXPERT_MAX_SIZE : LOCAL_EXPERT_MAX_SIZE_A5;
    OP_TILING_CHECK((localMoeExpertNum <= 0) || (localMoeExpertNum * epWorldSize > localExpertMaxSize),
                    OP_LOGE(nodeName,
                            "localMoeExpertNum is invalid, "
                            "localMoeExpertNum * epWorldSize must be less than or equal to %ld, and localMoeExpertNum "
                            "must be greater than 0, "
                            "but got localMoeExpertNum * epWorldSize = %u, localMoeExpertNum = %u",
                            localExpertMaxSize, localMoeExpertNum * epWorldSize, localMoeExpertNum),
                    return ge::GRAPH_FAILED);

    // 校验输入x的dim 0并设bs
    const gert::StorageShape* xStorageShape = context->GetInputShape(config.xIndex);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t bsUpperBound = isLayered ? BS_UPPER_BOUND_LAYERED : BS_UPPER_BOUND;
    if ((mc2tiling::GetNpuArch(context) == Ops::Base::DAV_9201) ||
        (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_9202)) {
        bsUpperBound = BS_UPPER_BOUND_A6;
    }
    OP_TILING_CHECK((xDim0 > bsUpperBound) || (xDim0 <= 0),
                    OP_LOGE(nodeName,
                            "xDim0(BS) is invalid. Should be between "
                            "[1, %ld], but got xDim0=%ld.",
                            bsUpperBound, xDim0),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.bs = static_cast<uint32_t>(xDim0);

    return CheckGlobalBsAttr(context, nodeName, tilingData, xDim0, epWorldSize, config);
}

static ge::graphStatus CheckTwoDimScalesShape(const gert::TilingContext* context, const char* nodeName,
                                              const MoeDistributeDispatchV2TilingData& tilingData,
                                              const int64_t scalesDim0, const int64_t scalesDim1,
                                              DispatchV2Config& config)
{
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    const gert::StorageShape* xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xShape"), return ge::GRAPH_FAILED);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    if (sharedExpertRankNum != 0U) {
        OP_TILING_CHECK(scalesDim0 != (moeExpertNum + sharedExpertNum),
                        OP_LOGE(nodeName,
                                "scales's dim0 not equal to moeExpertNum + sharedExpertNum, scales's dim0=%ld, "
                                "(moeExpertNum + sharedExpertNum)=%ld.",
                                scalesDim0, moeExpertNum + sharedExpertNum),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            scalesDim0 != moeExpertNum,
            OP_LOGE(nodeName, "scales's dim0 not equal to moeExpertNum, scales's dim0=%ld, moeExpertNum=%ld.",
                    scalesDim0, moeExpertNum),
            return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(xDim1 != scalesDim1,
                    OP_LOGE(nodeName,
                            "scales's dim1 not equal to xShape's dim1, "
                            "xShape's dim1=%ld, scales's dim1=%ld.",
                            xDim1, scalesDim1),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckLayeredExpertScalesShape(const gert::TilingContext* context, const char* nodeName,
                                                     int64_t expertIdsDim0, int64_t expertIdsDim1)
{
    const gert::StorageShape* expertScaleStorageShape = context->GetOptionalInputShape(EXPERT_SCALES_INDEX);
    const int64_t expertScalesDim0 = expertScaleStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertScalesDim1 = expertScaleStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK((expertScalesDim0 != expertIdsDim0) || (expertScalesDim1 != expertIdsDim1),
                    OP_LOGE(nodeName,
                            "expertScaleShape's dim not equal to expertIdShape's dim, "
                            "expertScaleShape's dim0 is %ld, expertIdShape's dim0 is %ld, "
                            "expertScaleShape's dim1 is %ld, expertIdShape's dim1 is %ld.",
                            expertScalesDim0, expertIdsDim0, expertScalesDim1, expertIdsDim1),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckFloat4LastAxis(const gert::TilingContext* context, const char* nodeName, int64_t xDim1)
{
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    if ((expandXDesc->GetDataType() == ge::DT_FLOAT4_E2M1) || (expandXDesc->GetDataType() == ge::DT_FLOAT4_E1M2) ||
        (expandXDesc->GetDataType() == Ops::Transformer::DT_HIFLOAT4)) {
        OP_TILING_CHECK(xDim1 % EVEN_ALIGN,
                        OP_LOGE(nodeName, "When expandx data type is FLOAT_E2M1/FLOAT_E1M2/HIFLOAT4, the last axis "
                                          "should be even, please check."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckXAndExpertIdsTensorShape(const gert::TilingContext* context, const char* nodeName,
                                                     MoeDistributeDispatchV2TilingData& tilingData,
                                                     const bool isLayered, const int64_t zeroExpertNum,
                                                     const int64_t copyExpertNum, const int64_t constExpertNum,
                                                     DispatchV2Config& config)
{
    // 校验输入x的维度1并设h, bs已校验过
    const gert::StorageShape* xStorageShape = context->GetInputShape(config.xIndex);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    int64_t hMax = isLayered ? H_MAX_LAYERED : H_MAX;
    int64_t hMin = isLayered ? 0 : H_MIN;
    OP_TILING_CHECK(
        (isLayered && (xDim1 % H_BASIC_BLOCK_LAYERED != 0)),
        OP_LOGE(nodeName, "xShape dims1(H) should be %u Aligned, but got %ld.", H_BASIC_BLOCK_LAYERED, xDim1),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((xDim1 < hMin) || (xDim1 > hMax),
                    OP_LOGE(nodeName, "xShape dims1(H) should be in [%ld, %ld], but got %ld.", hMin, hMax, xDim1),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.h = static_cast<uint32_t>(xDim1);

    if (CheckFloat4LastAxis(context, nodeName, xDim1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 校验expert_id的维度并设k
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    const gert::StorageShape* expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    const int64_t expertIdsDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0,
                    OP_LOGE(nodeName,
                            "xShape's dim0 not equal to expertIdShape's dim0, "
                            "xShape's dim0 is %ld, expertIdShape's dim0 is %ld.",
                            xDim0, expertIdsDim0),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX) ||
                        (expertIdsDim1 > moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum),
                    OP_LOGE(nodeName,
                            "expertIdShape's dim1(k) should be in (0, min(%ld, "
                            "moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum = %ld)], "
                            "but got expertIdShape's dim1=%ld.",
                            K_MAX, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum, expertIdsDim1),
                    return ge::GRAPH_FAILED);
    if (isLayered &&
        CheckLayeredExpertScalesShape(context, nodeName, expertIdsDim0, expertIdsDim1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    tilingData.moeDistributeDispatchV2Info.k = static_cast<uint32_t>(expertIdsDim1);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillOneDimScalesInfo(const gert::TilingContext* context, const char* nodeName,
                                            uint32_t quantMode, int64_t scalesDim0, uint64_t h, uint64_t& scalesCol,
                                            uint64_t& scalesCount)
{
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK((quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)) &&
                        (expandXDesc->GetDataType() == ge::DT_INT8) && (scalesDim0 != static_cast<int64_t>(h)) &&
                        (scalesDim0 != static_cast<int64_t>(STATIC_SCALE_DIM_0)),
                    OP_LOGE(nodeName, "The expected scalesDim0 is %lu or %lu in static quant, but got %ld", h,
                            STATIC_SCALE_DIM_0, scalesDim0),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)) &&
            (expandXDesc->GetDataType() == ge::DT_HIFLOAT8) && (scalesDim0 != STATIC_SCALE_DIM_0),
        OP_LOGE(nodeName, "The expected scalesDim0 is 1 when expandX datatype is hif8 in static quant, but got %ld",
                scalesDim0),
        return ge::GRAPH_FAILED);
    scalesCol = ONE_DIM_SCALE_COL_NUM;
    scalesCount = static_cast<uint64_t>(scalesDim0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillNonQuantScalesInfo(const char* nodeName, int64_t scalesDim0, int64_t scalesDim1, uint32_t bs,
                                              uint64_t h, uint64_t& scalesCol, uint64_t& scalesCount)
{
    OP_TILING_CHECK(scalesDim0 != bs,
                    OP_LOGE(nodeName, "The expected scalesDim0 is %u when scales is not null in non-quant, but got %ld",
                            bs, scalesDim0),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        scalesDim1 > h,
        OP_LOGE(
            nodeName,
            "The expected scalesDim1 is less than or equal to %lu when scales is not null in non-quant, but got %ld", h,
            scalesDim1),
        return ge::GRAPH_FAILED);
    scalesCol = static_cast<uint64_t>(scalesDim1);
    scalesCount = static_cast<uint64_t>(scalesDim0 * scalesDim1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillScalesInfo(const gert::TilingContext* context, const char* nodeName,
                                      MoeDistributeDispatchV2TilingData& tilingData, const bool isScales,
                                      const uint32_t quantMode, DispatchV2Config& config, uint64_t& scalesRow,
                                      uint64_t& scalesCol, uint32_t& scalesTypeSize, uint64_t& scalesCount)
{
    uint64_t h = tilingData.moeDistributeDispatchV2Info.h;
    uint32_t bs = tilingData.moeDistributeDispatchV2Info.bs;

    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        const gert::StorageShape* scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesShape"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesDesc"),
                        return ge::GRAPH_FAILED);
        size_t scalesDimNum = scalesStorageShape->GetStorageShape().GetDimNum();
        const int64_t scalesDim0 = scalesStorageShape->GetStorageShape().GetDim(0);
        scalesRow = static_cast<uint64_t>(scalesDim0);
        scalesTypeSize = ge::GetSizeByDataType(scalesDesc->GetDataType());
        if (scalesDimNum == ONE_DIM) {
            // A3不会进此分支
            if (FillOneDimScalesInfo(context, nodeName, quantMode, scalesDim0, h, scalesCol, scalesCount) !=
                ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        } else if (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
            const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
            if (FillNonQuantScalesInfo(nodeName, scalesDim0, scalesDim1, bs, h, scalesCol, scalesCount) !=
                ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        } else {
            const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
            OP_TILING_CHECK(CheckTwoDimScalesShape(context, nodeName, tilingData, scalesDim0, scalesDim1, config) !=
                                ge::GRAPH_SUCCESS,
                            OP_LOGE(nodeName, "CheckTwoDimScalesShape failed."), return ge::GRAPH_FAILED);
            scalesCol = static_cast<uint64_t>(scalesDim1);
            scalesCount = static_cast<uint64_t>(scalesDim0 * scalesDim1);
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputInfoTensorShape(const gert::TilingContext* context, const char* nodeName,
                                                 MoeDistributeDispatchV2TilingData& tilingData, uint32_t& A,
                                                 const int64_t localMoeExpertNum, const int64_t expertIdsDim1,
                                                 const bool hasElasticInfo, const bool isSharedExpert,
                                                 const bool isPerformance, DispatchV2Config& config)
{
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;
    uint32_t globalBs = tilingData.moeDistributeDispatchV2Info.globalBs;
    uint32_t epWorldSizeU32 = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    uint32_t rankNumPerSharedExpert = 0;
    uint32_t maxBs = globalBs / epWorldSizeU32;
    uint32_t maxSharedGroupNum = 0;
    if ((sharedExpertNum != 0U) && (sharedExpertRankNum != 0U)) { // 除零保护
        rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
        maxSharedGroupNum = (epWorldSizeU32 + rankNumPerSharedExpert - 1U) / rankNumPerSharedExpert;
    }
    if (isSharedExpert) { // 本卡为共享专家
        A = maxBs * maxSharedGroupNum;
    } else { // 本卡为moe专家
        A = globalBs * std::min(localMoeExpertNum, expertIdsDim1);
    }

    // 校验elasticInfo的维度，并更新一下最大输出的值
    if (hasElasticInfo) {
        const gert::StorageShape* elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
        const int64_t elasticInfoDim0 = elasticInfoStorageShape->GetStorageShape().GetDim(0);
        const int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);

        OP_TILING_CHECK(elasticInfoDim0 != (ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * epWorldSize),
                        OP_LOGE(nodeName,
                                "elasticInfo's dim0 not equal to 4 + 2 * epWorldSize, "
                                "elasticInfo's dim0 is %ld, epWorldSize is %ld.",
                                elasticInfoDim0, epWorldSize),
                        return ge::GRAPH_FAILED);
        A = std::max(static_cast<int64_t>(maxBs * maxSharedGroupNum),
                     globalBs * std::min(localMoeExpertNum, expertIdsDim1));
    }
    tilingData.moeDistributeDispatchV2Info.a = A;

    if (isPerformance) {
        const gert::StorageShape* performanceInfoStorageShape =
            context->GetOptionalInputShape(config.performanceInfoIndex);
        const int64_t performanceInfoDim0 = performanceInfoStorageShape->GetStorageShape().GetDim(0);
        const int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);

        OP_TILING_CHECK(performanceInfoDim0 != epWorldSize,
                        OP_LOGE(nodeName,
                                "performanceInfo's dim0 not equal to epWorldSize, "
                                "performanceInfo's dim0 is %ld, epWorldSize is %ld.",
                                performanceInfoDim0, epWorldSize),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpandXAndDynamicScalesTensorShape(const gert::TilingContext* context, const char* nodeName,
                                                               MoeDistributeDispatchV2TilingData& tilingData,
                                                               const uint32_t quantMode, const uint32_t A,
                                                               const int64_t xDim1)
{
    // 校验expandX的维度
    const gert::StorageShape* expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    const int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < static_cast<int64_t>(A),
                    OP_LOGE(nodeName,
                            "expandX's dim0 not greater than or equal to A, "
                            "expandX's dim0 is %ld, A is %u.",
                            expandXDim0, A),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xDim1 != expandXDim1,
                    OP_LOGE(nodeName,
                            "expandX's dim1 not equal to xShape's dim1, "
                            "xShape's dim1 is %ld, expandX's dim1 is %ld.",
                            xDim1, expandXDim1),
                    return ge::GRAPH_FAILED);

    uint64_t h = tilingData.moeDistributeDispatchV2Info.h;
    // 校验dynamicScales的维度
    if ((quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
        (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))) {
        // Dim0
        const gert::StorageShape* dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        const int64_t dynamicScalesDim0 = dynamicScalesStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(dynamicScalesDim0 < static_cast<int64_t>(A),
                        OP_LOGE(nodeName,
                                "dynamicScales's dim0 should be equal to or greater than A, "
                                "dynamicScales's dim0=%ld, A=%u.",
                                dynamicScalesDim0, A),
                        return ge::GRAPH_FAILED);
        // Dim1, only for pergroup and mx and hif4
        if (quantMode != static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
            const uint64_t dynamicScalesDim1 =
                static_cast<uint64_t>(dynamicScalesStorageShape->GetStorageShape().GetDim(1));
            OP_TILING_CHECK(
                ((quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) ||
                 (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT_CLIP))) &&
                    (dynamicScalesDim1 !=
                     ops::CeilAlign(static_cast<uint64_t>(ops::CeilDiv(h, MX_BLOCK_SIZE)), EVEN_ALIGN)),
                OP_LOGE(nodeName,
                        "dynamicScales's dim1 should be equal to %lu and even when quantMode=%u, but got %lu.",
                        ops::CeilAlign(static_cast<uint64_t>(ops::CeilDiv(h, MX_BLOCK_SIZE)), EVEN_ALIGN), quantMode,
                        dynamicScalesDim1),
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK(
                (dynamicScalesDim1 != ops::CeilDiv(h, PERGROUP_BLOCK_SIZE)) &&
                    (quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT)),
                OP_LOGE(nodeName, "dynamicScales's dim1 should be equal to %lu when quantMode=%u, but got %lu.",
                        ops::CeilDiv(h, PERGROUP_BLOCK_SIZE), quantMode, dynamicScalesDim1),
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK(
                (dynamicScalesDim1 != ops::CeilDiv(h, HIF4_GROUP_SIZE)) &&
                    (quantMode == static_cast<uint32_t>(QuantModeA5::HIF4_QUANT)),
                OP_LOGE(nodeName, "dynamicScales's dim1 should be equal to %lu when quantMode=%u, but got %lu.",
                        ops::CeilDiv(h, HIF4_GROUP_SIZE), quantMode, dynamicScalesDim1),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAssistInfoAndExpertTokenNumsTensorShape(
    const gert::TilingContext* context, const char* nodeName, MoeDistributeDispatchV2TilingData& tilingData,
    const uint32_t A, const int64_t localMoeExpertNum, const bool hasElasticInfo, const bool isSharedExpert,
    const DispatchV2Config& config)
{
    // 校验assistInfo的维度
    const gert::StorageShape* assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    const int64_t assistInfoDim0 = assistInfoStorageShape->GetStorageShape().GetDim(0);
    int64_t minAssistInfoDim0 = static_cast<int64_t>(A) * ASSIST_NUM_PER_A;
    if (config.isRoce) {
        const auto& info = tilingData.moeDistributeDispatchV2Info;
        minAssistInfoDim0 = GetRoceAssistInfoSize(A, info.epWorldSize, info.globalBs / info.epWorldSize);
    }
    OP_TILING_CHECK(assistInfoDim0 < minAssistInfoDim0,
                    OP_LOGE(nodeName,
                            "assistInfoDim0 < minAssistInfoDim0,"
                            " assistInfoDim0 is %ld, minAssistInfoDim0 is %ld.",
                            assistInfoDim0, minAssistInfoDim0),
                    return ge::GRAPH_FAILED);

    // 校验expertTokenNums的维度
    const gert::StorageShape* expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);

    if (hasElasticInfo) {
        OP_TILING_CHECK(expertTokenNumsDim0 != (localMoeExpertNum > 1 ? localMoeExpertNum : 1),
                        OP_LOGE(nodeName,
                                "elastic scaling expertTokenNums's Dim0 not equal to max(localMoeExpertNum,1), "
                                "expertTokenNumsDim0 is %ld, "
                                "localMoeExpertNum is %ld.",
                                expertTokenNumsDim0, localMoeExpertNum),
                        return ge::GRAPH_FAILED);
    } else if (isSharedExpert) {
        OP_TILING_CHECK(expertTokenNumsDim0 != 1,
                        OP_LOGE(nodeName, "shared expertTokenNums's dim0 %ld not equal to 1.", expertTokenNumsDim0),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            expertTokenNumsDim0 != localMoeExpertNum,
            OP_LOGE(nodeName,
                    "moe expertTokenNums's Dim0 not equal to localMoeExpertNum, expertTokenNumsDim0 is %ld, "
                    "localMoeExpertNum is %ld.",
                    expertTokenNumsDim0, localMoeExpertNum),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckRecvCountTensorShape(const gert::TilingContext* context, const char* nodeName,
                                                 MoeDistributeDispatchV2TilingData& tilingData,
                                                 const bool isSharedExpert, const int64_t localMoeExpertNum,
                                                 const bool hasElasticInfo, const bool isLayered,
                                                 const int64_t expertIdsDim1)
{
    // 校验epRecvCount和tpRecvCount的维度
    uint32_t globalBs = tilingData.moeDistributeDispatchV2Info.globalBs;
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    const gert::StorageShape* epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    const int64_t epRecvCountDim0 = epRecvCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epRecvCount = ((isSharedExpert) ? epWorldSize : epWorldSize * localMoeExpertNum);
    if (hasElasticInfo) {
        epRecvCount = std::max(epWorldSize, epWorldSize * localMoeExpertNum);
    }

    if (isLayered) {
        // 如果是分层方案，则需要校验全新的shape，额外的globalBs * 2 * k * epWorldSize / 8
        // 用来存储token的cnt信息与offset信息，为了兼容A3以外的芯片架构 这里取/8。
        epRecvCount = epWorldSize * localMoeExpertNum +
                      globalBs * LAYERED_METAINFO_LIST_NUM * expertIdsDim1 * epWorldSize / RANK_NUM_PER_NODE_A2;
        OP_TILING_CHECK(
            epRecvCountDim0 < epRecvCount,
            OP_LOGE(nodeName,
                    "dimension 0 of epRecvCount should be greater than or equal to epWorldSize * localMoeExpertNum + "
                    "globalBs * 2 * k * epWorldSize / 8, "
                    "but dimension 0 of epRecvCount is %ld, epWorldSize is %ld, localMoeExpertNum is %ld, k is %ld.",
                    epRecvCountDim0, epWorldSize, localMoeExpertNum, expertIdsDim1),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(epRecvCountDim0 < epRecvCount,
                        OP_LOGE(nodeName,
                                "dimension 0 of epRecvCount should be greater than or equal to epWorldSize * "
                                "localMoeExpertNum, "
                                "but dimension 0 of epRecvCount is %ld, epWorldSize is %ld, localMoeExpertNum is %ld.",
                                epRecvCountDim0, epWorldSize, localMoeExpertNum),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(const gert::TilingContext* context, const char* nodeName,
                                        MoeDistributeDispatchV2TilingData& tilingData, const uint32_t quantMode,
                                        const bool isScales, const bool isSharedExpert, const bool hasElasticInfo,
                                        const bool isPerformance, const int64_t localMoeExpertNum, bool isLayered,
                                        DispatchV2Config& config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));

    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;

    OP_TILING_CHECK(CheckXAndExpertIdsTensorShape(context, nodeName, tilingData, isLayered, zeroExpertNum,
                                                  copyExpertNum, constExpertNum, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check x or expertIds failed"), return ge::GRAPH_FAILED);
    int64_t xDim1 = tilingData.moeDistributeDispatchV2Info.h;
    int64_t expertIdsDim1 = tilingData.moeDistributeDispatchV2Info.k;

    // 校验scales的维度
    uint64_t scalesRow = 0;
    uint64_t scalesCol = 0;
    uint32_t scalesTypeSize = 0;
    uint64_t scalesCount = 0;

    OP_TILING_CHECK(FillScalesInfo(context, nodeName, tilingData, isScales, quantMode, config, scalesRow, scalesCol,
                                   scalesTypeSize, scalesCount) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Fill scales relevant info failed"), return ge::GRAPH_FAILED);

    tilingData.moeDistributeDispatchV2Info.scalesRow = scalesRow;
    tilingData.moeDistributeDispatchV2Info.scalesCol = scalesCol;
    tilingData.moeDistributeDispatchV2Info.scalesCount = scalesCount;
    tilingData.moeDistributeDispatchV2Info.scalesTypeSize = scalesTypeSize;

    uint32_t A = 0U;
    OP_TILING_CHECK(
        CheckInputInfoTensorShape(context, nodeName, tilingData, A, localMoeExpertNum, expertIdsDim1, hasElasticInfo,
                                  isSharedExpert, isPerformance, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check input elasticInfo or performanceInfo failed"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckExpandXAndDynamicScalesTensorShape(context, nodeName, tilingData, quantMode, A, xDim1) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check expandX or dynamicScales failed"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        CheckAssistInfoAndExpertTokenNumsTensorShape(context, nodeName, tilingData, A, localMoeExpertNum,
                                                     hasElasticInfo, isSharedExpert, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check assistInfo or expertTokenNums failed"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckRecvCountTensorShape(context, nodeName, tilingData, isSharedExpert, localMoeExpertNum,
                                              hasElasticInfo, isLayered, expertIdsDim1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check epRecvCount or tpRecvCount failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 校验输入输出数据格式
static bool CheckTensorPtrNullptr(const gert::TilingContext* context, const char* nodeName, const bool isScales,
                                  const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
                                  const bool isPerformance, DispatchV2Config& config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto expertIdDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xDesc"), return false);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertIdDesc"), return false);
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesDesc"), return false);
    }
    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "xActiveMaskDesc"),
                        return false);
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "elasticInfoDesc"),
                        return false);
    }
    if (isPerformance) {
        auto performanceInfoDesc = context->GetOptionalInputDesc(config.performanceInfoIndex);
        OP_TILING_CHECK(performanceInfoDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "performanceInfoDesc"),
                        return false);
    }

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandXDesc"), return false);
    if (quantMode >= static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamicScalesDesc"),
                        return false);
    }
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "assistInfoDesc"), return false);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertTokenNumsDesc"),
                    return false);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epRecvCountsDesc"),
                    return false);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpRecvCountsDesc"),
                    return false);

    return true;
}

static ge::graphStatus CheckMc2Context(gert::TilingContext* context, const char* nodeName, DispatchV2Config& config)
{
    const gert::StorageShape* contextStorageShape = context->GetInputShape(config.contextIndex);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "contextShape"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "contextShape", std::to_string(contextStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of contextShape must be 1D"),
        return ge::GRAPH_FAILED);
    int64_t contextDim0 = contextStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "context dim0 = %ld", contextDim0);

    auto contextDesc = context->GetInputDesc(config.contextIndex);
    OP_TILING_CHECK(contextDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "contextDesc"),
                    return ge::GRAPH_FAILED);
    if (contextDesc->GetDataType() != ge::DT_INT32) {
        std::string dtypeStr = Ops::Base::ToString(contextDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "context", dtypeStr.c_str(), "INT32");
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "context format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::TilingCheckMoeDistributeDispatch(
    gert::TilingContext* context, const char* nodeName, const bool isActiveMask, const bool isScales,
    const bool hasElasticInfo, const bool isPerformance, const uint32_t quantMode, const bool isLayered,
    DispatchV2Config& config)
{
    OP_TILING_CHECK(!CheckTensorPtrNullptr(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo,
                                           isPerformance, config),
                    OP_LOGE(nodeName, "params check nullptr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo, isPerformance,
                                    isLayered, config),
                    OP_LOGE_LIBOPAPI_REPORT(nodeName, "CheckTensorDim failed; propagate lower-layer error"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo,
                                         isPerformance, config),
                    OP_LOGE_LIBOPAPI_REPORT(nodeName, "CheckTensorDataType failed; propagate lower-layer error"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !CheckTensorFormat(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo, isPerformance, config),
        OP_LOGE_LIBOPAPI_REPORT(nodeName, "CheckTensorFormat failed; propagate lower-layer error"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetHcommCfg(const gert::TilingContext* context, MoeDistributeDispatchV2TilingData* tiling,
                                   const std::string groupEp, bool isLayered)
{
    const char* nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE("MoeDistributeDispatchV2", "Fail to get nodeName."),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 groupEp = %s", groupEp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr =
        isLayered ? "AlltoAll=level1:hierarchy" : "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE); // 通过不拉起AICPU，提高算子退出性能
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling) != 0,
                    OP_LOGE(nodeName, "mc2CcTilingConfig mc2InitTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1) != 0,
                    OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling1 GetTiling failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static uint64_t CalcA5DispatchTokenBytes(const MoeDistributeDispatchV2Info& info)
{
    uint64_t tokenBytes = ((info.h * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1U) / UB_ALIGN) * UB_ALIGN;
    tokenBytes += static_cast<uint64_t>(info.scalesCol) * info.scalesTypeSize;
    return ((tokenBytes + UB_ALIGN - 1U) / UB_ALIGN) * UB_ALIGN + Mc2Kernel::EXPAND_IDX_INFO * sizeof(int32_t) +
           (info.hasExpertScales ? sizeof(float) : 0U);
}

static ge::graphStatus CheckAndCalWinSize(const gert::TilingContext* context,
                                          MoeDistributeDispatchV2TilingData& tilingData, const char* nodeName,
                                          const bool isSetFullMeshV2, uint32_t localMoeExpertNum, bool isLayered,
                                          DispatchV2Config& config)
{
    // win区的数据结构
    CheckWinSizeData winSizeData;
    winSizeData.localMoeExpertNum = localMoeExpertNum;
    winSizeData.sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    winSizeData.a = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.a);
    winSizeData.h = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.h);
    winSizeData.k = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.k);
    winSizeData.bs = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.bs);
    winSizeData.epWorldSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    winSizeData.globalBs = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.globalBs);
    winSizeData.moeExpertNum = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    winSizeData.isSetFullMeshV2 = isSetFullMeshV2;
    winSizeData.isLayered = isLayered;
    winSizeData.isMc2Context = config.isMc2Context;
    const auto& info = tilingData.moeDistributeDispatchV2Info;
    uint64_t dispatchTokenBytes = 0UL;
    if (mc2tiling::GetNpuArch(context) != Ops::Base::DAV_2201 &&
        info.quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT) && info.scalesCol != 0U) {
        dispatchTokenBytes = CalcA5DispatchTokenBytes(info);
    }
    const bool useRuntimeWinSize =
        !isLayered && !config.isMc2Context && mc2tiling::GetSocVersion(context) == "Ascend910_93";
    const auto ret = useRuntimeWinSize ? CalcMinWinSizeA3(context, nodeName, winSizeData) :
                                         CheckWinSize(context, nodeName, winSizeData, dispatchTokenBytes);
    OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "Get WinSize failed."), return ge::GRAPH_FAILED);
    // 总的win区大小
    tilingData.moeDistributeDispatchV2Info.totalWinSizeEp = winSizeData.totalWinSizeEp;
    return ge::GRAPH_SUCCESS;
}

static uint32_t Ceil(uint32_t dividend, uint32_t divisor, const char* nodeName)
{
    if (divisor != 0) {
        return ((dividend + divisor - 1) / divisor) * divisor;
    }
    OP_LOGE(nodeName, "ceil divisor should not be 0");
    return 0;
}

static bool IsFullMeshSerialMode(const MoeDistributeDispatchV2Info& info)
{
    uint32_t maxBs = info.globalBs / info.epWorldSize;
    return maxBs <= SERIAL_MODE_BS_THRESHOLD && info.epWorldSize < SERIAL_MODE_EP_WORLD_SIZE_THRESHOLD;
}

static uint32_t SendToMoeExpertUsedBuffer(const gert::TilingContext* context,
                                          const MoeDistributeDispatchV2TilingData& tilingData,
                                          uint32_t expertIdsBufSize, uint32_t kSize, const char* nodeName)
{
    uint32_t UbForMoe = 0;
    uint32_t moeExpertNum = tilingData.moeDistributeDispatchV2Info.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t totalExpertNum = sharedExpertRankNum + moeExpertNum;
    uint32_t epWorldSize = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    uint32_t epRankId = tilingData.moeDistributeDispatchV2Info.epRankId;
    uint32_t moeExpertRankNum = epWorldSize - sharedExpertRankNum;
    uint32_t moeExpertNumPerRank = moeExpertNum / moeExpertRankNum;
    uint32_t rscvStatusNum = epRankId < sharedExpertRankNum ? epWorldSize : epWorldSize * moeExpertNumPerRank;
    uint32_t aivUsedCumSum = totalExpertNum / 16; // 单核处理16个专家cnt发送
    aivUsedCumSum = (aivUsedCumSum == 0) ? 1 : aivUsedCumSum;
    aivUsedCumSum = (aivUsedCumSum >= (aivNum / HALF_NUM)) ? (aivNum / HALF_NUM) : aivUsedCumSum;
    aivUsedCumSum = (aivUsedCumSum >= A5_CUMSUM_MAX_CORE_NUM) ? A5_CUMSUM_MAX_CORE_NUM : aivUsedCumSum;
    aivUsedCumSum = (aivUsedCumSum >= rscvStatusNum) ? rscvStatusNum : aivUsedCumSum;
    uint32_t aivUsedAllToAll = aivNum - aivUsedCumSum;
    uint32_t sharedUsedAivNum = 0;
    if (sharedExpertRankNum != 0U) {
        sharedUsedAivNum = (aivUsedAllToAll * sharedExpertNum) / (kSize + sharedExpertNum);
        if (sharedUsedAivNum == 0) {
            sharedUsedAivNum = 1;
        }
    }
    uint32_t moeUsedAivNum = aivUsedAllToAll - sharedUsedAivNum;
    uint32_t maskSizePerExpert = Ceil((expertIdsBufSize / sizeof(int32_t)) / 8, UB_ALIGN, nodeName); // 8 is 1byte->8bit
    uint32_t expertMaskBufSize = maskSizePerExpert * Ceil(moeExpertNum, moeUsedAivNum, nodeName) / moeUsedAivNum;
    if (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_2201) {
        UbForMoe = UbForMoe + expertMaskBufSize;                                        // expertMaskBuf
        UbForMoe = UbForMoe + Ceil(moeExpertNum * sizeof(int32_t), UB_ALIGN, nodeName); // tokenNumToExpertBuf
    }
    if (IsFullMeshSerialMode(tilingData.moeDistributeDispatchV2Info) &&
        mc2tiling::GetNpuArch(context) != Ops::Base::DAV_2201) {
        uint32_t countNumPerCore = Ceil(totalExpertNum, aivNum, nodeName) / aivNum;
        if (mc2tiling::GetNpuArch(context) != Ops::Base::DAV_3510) {
            uint32_t rankNumPerCore = Ceil(epWorldSize, aivNum, nodeName) / aivNum;
            uint32_t expertNumPerRank = std::max(moeExpertNumPerRank, 1U);
            countNumPerCore = std::max(countNumPerCore, rankNumPerCore * expertNumPerRank);
        }
        uint32_t statusCntAlign = Ceil(countNumPerCore, UB_ALIGN_INT32_NUM, nodeName);
        UbForMoe = UbForMoe + statusCntAlign * UB_ALIGN_VAL;
        if (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_3510) {
            uint32_t histoGroupNum =
                (statusCntAlign + Mc2Kernel::EXPERT_PER_HISTO_GROUP - 1U) / Mc2Kernel::EXPERT_PER_HISTO_GROUP;
            UbForMoe += histoGroupNum * Mc2Kernel::EXPERT_HISTO_BIN_NUM * sizeof(uint16_t);
        }
    }
    return UbForMoe;
}

static uint32_t AllToAllBasicUsedBuffer(const gert::TilingContext* context,
                                        const MoeDistributeDispatchV2TilingData& tilingData, uint32_t quantMode,
                                        uint32_t& expertIdsBufSize, uint32_t hSize, uint32_t expertIdsCnt,
                                        uint32_t& hOutSizeAlign, const char* nodeName, const DispatchV2Config& config)
{
    uint32_t UbForBasic = 0;
    uint32_t epWorldSize = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    if (tilingData.moeDistributeDispatchV2Info.hasElasticInfo) {
        uint32_t elasticInfoSize = (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSize) * sizeof(int32_t);
        uint32_t elasticInfoSizeAlign = Ceil(elasticInfoSize, UB_ALIGN, nodeName);
        UbForBasic = UbForBasic + elasticInfoSizeAlign; // elasticInfoBuf_
    }

    bool isArch22 = mc2tiling::GetNpuArch(context) == Ops::Base::DAV_2201;
    bool isArch35 = mc2tiling::GetNpuArch(context) == Ops::Base::DAV_3510;
    bool isMxQuant = !isArch22 && (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT) ||
                                   quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT_CLIP));
    bool isPerGroupQuant = !isArch22 && quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT);
    bool isHif4Quant = !(isArch22 || isArch35) && quantMode == static_cast<uint32_t>(QuantModeA5::HIF4_QUANT);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    const auto inputType = xDesc->GetDataType();
    const bool isB4 = (inputType == ge::DT_FLOAT4_E2M1) || (inputType == ge::DT_FLOAT4_E1M2) ||
                      (inputType == Ops::Transformer::DT_HIFLOAT4);
    uint32_t hAlignSize = 0;
    if (isB4) {
        // 4bit 输入按 2 元素/字节打包
        const uint32_t hInBytes = static_cast<uint32_t>((hSize + EVEN_ALIGN - 1) / EVEN_ALIGN);
        hAlignSize = Ceil(hInBytes, UB_ALIGN, nodeName);
    } else {
        const uint32_t inputTypeSize = ge::GetSizeByDataType(inputType);
        hAlignSize = (isMxQuant || isPerGroupQuant) ? Ceil(hSize, PERGROUP_BLOCK_SIZE, nodeName) * inputTypeSize :
                                                      Ceil(hSize * inputTypeSize, UB_ALIGN, nodeName);
    }
    // xDesc与expandXDesc在之前的函数中已校验非空，此处不做校验
    if (xDesc->GetDataType() == expandXDesc->GetDataType()) {
        hAlignSize += UB_ALIGN;
    }
    UbForBasic = UbForBasic + hAlignSize * BUFFER_NUM; // inQueue

    const gert::StorageShape* scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
    bool isScales = (scalesStorageShape != nullptr);
    const auto outputType = expandXDesc->GetDataType();
    const bool isB4Out = (outputType == ge::DT_FLOAT4_E2M1) || (outputType == ge::DT_FLOAT4_E1M2) ||
                         (outputType == Ops::Transformer::DT_HIFLOAT4);
    uint32_t hOutSize = 0;
    if (isB4Out) {
        // 4bit 输出按 2 元素/字节打包
        hOutSize = static_cast<uint32_t>((hSize + EVEN_ALIGN - 1) / EVEN_ALIGN);
    } else {
        const uint32_t outputTypeSize = ge::GetSizeByDataType(outputType);
        hOutSize = hSize * outputTypeSize;
    }
    hOutSizeAlign = Ceil(hOutSize, UB_ALIGN, nodeName);
    if ((quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) && isScales) {
        uint32_t scaleInBytes =
            tilingData.moeDistributeDispatchV2Info.scalesCol * tilingData.moeDistributeDispatchV2Info.scalesTypeSize;
        hOutSizeAlign += scaleInBytes;
    } else if (quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        hOutSizeAlign += sizeof(float);
    } else if (isMxQuant) {
        bool isFp4 = outputType == ge::DT_FLOAT4_E2M1 || outputType == ge::DT_FLOAT4_E1M2;
        uint32_t outputBytes = isFp4 ? (hSize + 1U) / 2U : hSize;
        uint32_t scaleCount = Ceil(hSize, MX_BLOCK_SIZE, nodeName) / MX_BLOCK_SIZE;
        hOutSizeAlign = Ceil(outputBytes, SIZE_ALIGN_256, nodeName) + Ceil(scaleCount, EVEN_ALIGN, nodeName);
    } else if (isPerGroupQuant) {
        uint32_t paddedH = Ceil(hSize, PERGROUP_BLOCK_SIZE, nodeName);
        hOutSizeAlign = paddedH + paddedH / PERGROUP_BLOCK_SIZE * sizeof(float);
    } else if (isHif4Quant) {
        uint32_t scaleCount = Ceil(hSize, HIF4_GROUP_SIZE, nodeName) / HIF4_GROUP_SIZE;
        hOutSizeAlign +=
            scaleCount *
            sizeof(uint32_t); // hifloat4_scale为b32，此处使用uint32_t确保不含hifloat4_scale的低版本包可以正常编译
    }
    uint32_t hScaleSizeAlign = Ceil(hOutSizeAlign, UB_ALIGN, nodeName);
    uint32_t tokenQuantAlign = hScaleSizeAlign / sizeof(int32_t);
    hOutSizeAlign = tokenQuantAlign * sizeof(int32_t) + UB_ALIGN;
    uint32_t blockCntPerToken = Ceil(hOutSizeAlign, SPLIT_BLOCK_DATA_SIZE, nodeName) / SPLIT_BLOCK_DATA_SIZE;
    uint32_t hCommuSize = blockCntPerToken * SPLIT_BLOCK_SIZE;
    UbForBasic = UbForBasic + hCommuSize * BUFFER_NUM; // outBuf

    expertIdsBufSize = Ceil(expertIdsCnt * sizeof(int32_t), SIZE_ALIGN_256, nodeName);
    UbForBasic = UbForBasic + expertIdsBufSize; // expertIdsBuf_
    return UbForBasic;
}

static uint32_t A3FullMeshUsedBuffer(const gert::TilingContext* context,
                                     const MoeDistributeDispatchV2TilingData& tilingData, uint32_t expertIdsBufSize,
                                     uint32_t hOutSizeAlign, const DispatchV2Config& config, const char* nodeName)
{
    const auto& info = tilingData.moeDistributeDispatchV2Info;
    // calBeginBuf_, dataStateBuf and calEndBuf_ remain live until token sending ends.
    uint32_t usedSize = UB_ALIGN * 3U;
    if (info.hasElasticInfo) {
        usedSize +=
            Ceil((ELASTIC_INFO_OFFSET + RANK_LIST_NUM * info.epWorldSize) * sizeof(int32_t), UB_ALIGN, nodeName);
    }
    // SendCumSum retains both counting scratch buffers, the mask workspace and routing IDs.
    usedSize += BUFFER_NUM * info.maxSizeForUbBuffer + BUFFER_NUM * expertIdsBufSize;
    if (info.hasExpertScales) {
        usedSize += expertIdsBufSize;
    }
    uint32_t countBufferSize =
        Ceil(info.moeExpertNum + info.sharedExpertRankNum, UB_ALIGN / sizeof(int32_t), nodeName) * UB_ALIGN;
    usedSize += countBufferSize;
    if (info.isExpertMask || info.zeroComputeExpertNum != 0) {
        usedSize += Ceil(info.bs * sizeof(int32_t), UB_ALIGN, nodeName);
        uint32_t maskSize = info.bs * Ceil(info.k * sizeof(bool), UB_ALIGN, nodeName) * SIZE_OF_HALF;
        usedSize += std::max(Ceil(info.bs * info.k * sizeof(int32_t), UB_ALIGN, nodeName), maskSize);
    }
    uint32_t packedDataSize = Ceil(hOutSizeAlign, SPLIT_BLOCK_DATA_SIZE, nodeName);
    bool isSameType = context->GetInputDesc(config.xIndex)->GetDataType() ==
                      context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX)->GetDataType();
    uint32_t inputSize = Ceil(info.h * SIZE_OF_HALF, UB_ALIGN, nodeName);
    if (info.quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        inputSize += info.scalesCol * info.scalesTypeSize;
        if (isSameType) {
            inputSize = packedDataSize;
        }
    }
    usedSize += BUFFER_NUM * Ceil(inputSize, UB_ALIGN, nodeName);
    if (info.quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT) || !isSameType) {
        usedSize += packedDataSize; // quantization result, including the complete final 480B block
    }
    uint32_t sendSlotSize = packedDataSize / SPLIT_BLOCK_DATA_SIZE * SPLIT_BLOCK_SIZE;
    // A completed count buffer can provide the first send slot.
    if (countBufferSize < sendSlotSize) {
        usedSize += sendSlotSize;
    } else if (info.hasElasticInfo) {
        // After shrink, a count buffer smaller than one slot needs a separate slot.
        // Their combined capacity is below two slots; larger counts can be reused.
        usedSize += std::max(countBufferSize, BUFFER_NUM * sendSlotSize) - countBufferSize;
    }
    return usedSize;
}

static uint32_t SetMaxSizeForUbBuffer(MoeDistributeDispatchV2TilingData& tilingData, uint32_t hSize, uint32_t bsSize,
                                      uint32_t kSize, uint32_t expertIdsCnt, bool isScales, const char* nodeName)
{
    uint32_t quantMode = tilingData.moeDistributeDispatchV2Info.quantMode;
    uint32_t hFp32Size = Ceil(hSize * sizeof(float), UB_ALIGN, nodeName);
    if ((quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT)) && isScales) {
        hFp32Size = Ceil(hSize, PERGROUP_BLOCK_SIZE, nodeName) * sizeof(float);
    }
    uint32_t bsKAlign256 = Ceil(expertIdsCnt * SIZE_OF_HALF, SIZE_ALIGN_256, nodeName);
    uint32_t expertIdsSize = Ceil(expertIdsCnt * sizeof(int32_t), UB_ALIGN, nodeName);
    uint32_t xActivateMaskSize = bsSize * Ceil(kSize * sizeof(bool), UB_ALIGN, nodeName) * SIZE_OF_HALF;
    uint32_t maxSize = hFp32Size > expertIdsSize ? hFp32Size : expertIdsSize;
    maxSize = maxSize > xActivateMaskSize ? maxSize : xActivateMaskSize;
    maxSize = maxSize > bsKAlign256 ? maxSize : bsKAlign256;
    tilingData.moeDistributeDispatchV2Info.maxSizeForUbBuffer = maxSize;
    return maxSize;
}

static ge::graphStatus CheckUsedUbSize(const gert::TilingContext* context,
                                       const MoeDistributeDispatchV2TilingData& tilingData, const char* nodeName,
                                       uint32_t ubSize, uint32_t bsSize, uint32_t hSize, uint32_t kSize,
                                       const std::string& ubDetail)
{
    uint64_t ubRealSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubRealSize);
    OP_TILING_CHECK(ubSize > ubRealSize,
                    OP_LOGE(nodeName,
                            "Current scenario is exceeds the size limit, BS = %u, H = %u, topK = %u, moeExpertNum = "
                            "%u, aivNum available = %u, used UBsize = %u, real UBsize = %lu, UB breakdown: %s",
                            bsSize, hSize, kSize, tilingData.moeDistributeDispatchV2Info.moeExpertNum,
                            ascendcPlatform.GetCoreNumAiv(), ubSize, ubRealSize, ubDetail.c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcA5CumsumUbBudget(uint64_t ubRealSize, const char* nodeName, uint32_t& ubBudget)
{
    OP_TILING_CHECK(ubRealSize <= Mc2Kernel::A5_CUMSUM_UB_RESERVE,
                    OP_LOGE(nodeName, "UB size %luB must be greater than reserved size %uB.", ubRealSize,
                            Mc2Kernel::A5_CUMSUM_UB_RESERVE),
                    return ge::GRAPH_FAILED);
    ubBudget = static_cast<uint32_t>(ubRealSize - Mc2Kernel::A5_CUMSUM_UB_RESERVE);
    return ge::GRAPH_SUCCESS;
}

static uint32_t CalcA5TokenInAlignSize(const gert::TilingContext* context, const MoeDistributeDispatchV2Info& info,
                                       const DispatchV2Config& config, const char* nodeName)
{
    const uint32_t quantMode = info.quantMode;
    const bool isMxQuant = quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT) ||
                           quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT_CLIP);
    const bool isPerGroupQuant = quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT);
    auto inputType = context->GetInputDesc(config.xIndex)->GetDataType();
    const bool isB4 = (inputType == ge::DT_FLOAT4_E2M1) || (inputType == ge::DT_FLOAT4_E1M2) ||
                      (inputType == Ops::Transformer::DT_HIFLOAT4);
    uint32_t hAlignSize = 0;
    if (isB4) {
        // 同 AllToAllBasicUsedBuffer：4bit 输入按打包字节数记账，与 quant.h 的 hAlignSize_ 对齐
        const uint32_t hInBytes = static_cast<uint32_t>((info.h + EVEN_ALIGN - 1) / EVEN_ALIGN);
        hAlignSize = Ceil(hInBytes, UB_ALIGN, nodeName);
    } else {
        const uint32_t inputTypeSize = ge::GetSizeByDataType(inputType);
        hAlignSize = (isMxQuant || isPerGroupQuant) ? Ceil(info.h, PERGROUP_BLOCK_SIZE, nodeName) * inputTypeSize :
                                                      Ceil(info.h * inputTypeSize, UB_ALIGN, nodeName);
    }

    if (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        hAlignSize += info.scalesCol * info.scalesTypeSize;
    }
    return hAlignSize;
}

static uint32_t CalcA5TokenOutAlignSize(const gert::TilingContext* context, const MoeDistributeDispatchV2Info& info,
                                        const char* nodeName)
{
    const uint32_t quantMode = info.quantMode;
    const bool isMxQuant = quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT) ||
                           quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT_CLIP);
    const bool isPerGroupQuant = quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT);
    auto outputType = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX)->GetDataType();
    bool isOutputFp4 = outputType == ge::DT_FLOAT4_E2M1 || outputType == ge::DT_FLOAT4_E1M2 ||
                       outputType == Ops::Transformer::DT_HIFLOAT4;
    uint32_t outputTypeSize = isOutputFp4 ? 1U : ge::GetSizeByDataType(outputType);
    uint32_t scaleInBytes = info.scalesCol * info.scalesTypeSize;
    uint32_t outputBytes = isOutputFp4 ? (info.h + 1U) / 2U : info.h * outputTypeSize;
    uint32_t hOutSize = Ceil(outputBytes, UB_ALIGN, nodeName);
    if (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        hOutSize += scaleInBytes;
    } else if (quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        hOutSize += sizeof(float);
    } else if (isMxQuant) {
        uint32_t scaleCount = Ceil(info.h, MX_BLOCK_SIZE, nodeName) / MX_BLOCK_SIZE;
        hOutSize = Ceil(outputBytes, SIZE_ALIGN_256, nodeName) + Ceil(scaleCount, EVEN_ALIGN, nodeName);
    } else if (isPerGroupQuant) {
        uint32_t paddedH = Ceil(info.h, PERGROUP_BLOCK_SIZE, nodeName);
        hOutSize = paddedH * outputTypeSize + paddedH / PERGROUP_BLOCK_SIZE * sizeof(float);
    }
    return Ceil(
        Ceil(hOutSize, UB_ALIGN, nodeName) + TRIPLE * sizeof(int32_t) + (info.hasExpertScales ? sizeof(float) : 0U),
        UB_ALIGN, nodeName);
}

static bool NeedA5CastBuf(const gert::TilingContext* context, const MoeDistributeDispatchV2Info& info,
                          const DispatchV2Config& config)
{
    auto inputType = context->GetInputDesc(config.xIndex)->GetDataType();
    auto outputType = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX)->GetDataType();
    return info.quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT) || inputType != outputType;
}

static ge::graphStatus CheckSerialReceiveUbSizeA5(const gert::TilingContext* context, const char* nodeName,
                                                  const MoeDistributeDispatchV2TilingData& tilingData,
                                                  const DispatchV2Config& config)
{
    const auto& info = tilingData.moeDistributeDispatchV2Info;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubRealSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubRealSize);
    uint32_t cumsumUbBudget = 0;
    OP_TILING_CHECK(CalcA5CumsumUbBudget(ubRealSize, nodeName, cumsumUbBudget) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Calc cumsum UB budget failed."), return ge::GRAPH_FAILED);
    uint32_t moeExpertNumPerRank = info.moeExpertNum / (info.epWorldSize - info.sharedExpertRankNum);
    uint32_t recvStatusNum = info.epWorldSize * moeExpertNumPerRank;
    uint32_t recvStatusPerCore = Ceil(recvStatusNum, aivNum, nodeName) / aivNum;

    uint64_t fixedUbSize = 2U * UB_ALIGN;
    fixedUbSize += Ceil(recvStatusPerCore * UB_ALIGN, SIZE_ALIGN_256, nodeName);
    uint32_t gatherSize = std::max(Ceil(recvStatusPerCore * sizeof(float), UB_ALIGN, nodeName),
                                   Ceil(info.epWorldSize * moeExpertNumPerRank * sizeof(float), UB_ALIGN, nodeName));
    fixedUbSize += std::max(gatherSize, Ceil(recvStatusNum * Mc2Kernel::STATE_OFFSET, UB_ALIGN, nodeName));
    fixedUbSize += 2U * aivNum * UB_ALIGN;
    uint32_t expertPrefixNum = Ceil(recvStatusPerCore, info.epWorldSize, nodeName) / info.epWorldSize;
    fixedUbSize += std::max(Ceil(aivNum * sizeof(float), UB_ALIGN, nodeName),
                            Ceil(expertPrefixNum * sizeof(int32_t), UB_ALIGN, nodeName));
    fixedUbSize += 3U * UB_ALIGN;
    if (info.isPerformance) {
        fixedUbSize += Ceil(info.epWorldSize * sizeof(int64_t), UB_ALIGN, nodeName);
        fixedUbSize += Ceil(recvStatusPerCore * sizeof(int32_t), UB_ALIGN, nodeName);
    }
    fixedUbSize += Ceil(moeExpertNumPerRank * sizeof(int64_t), UB_ALIGN, nodeName);
    fixedUbSize += 4U * Ceil(recvStatusPerCore * sizeof(int32_t), UB_ALIGN, nodeName);

    uint32_t tokenAlign = CalcA5TokenOutAlignSize(context, info, nodeName);
    uint32_t blockCnt = Ceil(tokenAlign, SPLIT_BLOCK_DATA_SIZE, nodeName) / SPLIT_BLOCK_DATA_SIZE;
    uint64_t minCopyUbSize = BUFFER_NUM * static_cast<uint64_t>(blockCnt) * SPLIT_BLOCK_DATA_SIZE;
    minCopyUbSize += Ceil(blockCnt * sizeof(float), SIZE_ALIGN_256, nodeName) + blockCnt * UB_ALIGN;
    uint64_t requiredUbSize = fixedUbSize + minCopyUbSize + 2U * UB_ALIGN;
    OP_TILING_CHECK(
        requiredUbSize > cumsumUbBudget,
        OP_LOGE(nodeName,
                "Serial receive UB exceeds budget: fixed=%luB, minimum copy=%luB, reserved=%luB, budget=%uB, "
                "epRankId=%u, receive states per core=%u.",
                fixedUbSize, minCopyUbSize, static_cast<uint64_t>(2U * UB_ALIGN), cumsumUbBudget, info.epRankId,
                recvStatusPerCore),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static uint32_t CalcCumSumCoreNum(uint32_t totalExpertNum, uint32_t upperBound, uint32_t aivNum)
{
    uint32_t coreNum = totalExpertNum / 16U;
    coreNum = (coreNum == 0U) ? 1U : coreNum;
    coreNum = (coreNum >= (aivNum / HALF_NUM)) ? (aivNum / HALF_NUM) : coreNum;
    coreNum = (coreNum >= A5_CUMSUM_MAX_CORE_NUM) ? A5_CUMSUM_MAX_CORE_NUM : coreNum;
    coreNum = (coreNum >= upperBound) ? upperBound : coreNum;
    return (coreNum == 0U) ? 1U : coreNum;
}

static uint32_t CalcCumSumStatusCntAlign(const MoeDistributeDispatchV2Info& info, uint32_t aivNum, const char* nodeName)
{
    uint32_t totalExpertNum = info.sharedExpertRankNum + info.moeExpertNum;
    uint32_t coreNum = CalcCumSumCoreNum(totalExpertNum, totalExpertNum, aivNum);
    uint32_t expertNumPerCore = (totalExpertNum + coreNum - 1U) / coreNum;
    uint32_t statusCntAlign = Ceil(expertNumPerCore, UB_ALIGN / sizeof(int32_t), nodeName);
    if (info.sharedExpertRankNum != 0U) {
        uint32_t coreNumShare = CalcCumSumCoreNum(totalExpertNum, info.epWorldSize, aivNum);
        uint32_t expertNumPerCoreShare = (totalExpertNum + coreNumShare - 1U) / coreNumShare;
        uint32_t statusCntAlignShare = Ceil(expertNumPerCoreShare, UB_ALIGN / sizeof(int32_t), nodeName);
        statusCntAlign = std::max(statusCntAlign, statusCntAlignShare);
    }
    return statusCntAlign;
}

static uint32_t CalcCumSumRecStatusNumPerCore(const MoeDistributeDispatchV2Info& info, uint32_t aivNum,
                                              const char* nodeName)
{
    uint32_t moeExpertNumPerRank = info.moeExpertNum / (info.epWorldSize - info.sharedExpertRankNum);
    uint32_t rscvStatusNum = info.epWorldSize * moeExpertNumPerRank;
    uint32_t totalExpertNum = info.sharedExpertRankNum + info.moeExpertNum;
    uint32_t coreNum = CalcCumSumCoreNum(totalExpertNum, rscvStatusNum, aivNum);
    return (rscvStatusNum + coreNum - 1U) / coreNum;
}

static uint32_t CalcCumSumUsedUbSize(const gert::TilingContext* context, const MoeDistributeDispatchV2Info& info,
                                     uint32_t recStatusNumPerCore, uint32_t statusCntAlign, uint32_t expertIdsBufSize,
                                     uint32_t maxSizeForUbBuffer, uint32_t aivNum, const char* nodeName)
{
    uint32_t moeExpertNumPerRank = info.moeExpertNum / (info.epWorldSize - info.sharedExpertRankNum);
    uint32_t cumSumUbSize = 2U * UB_ALIGN;
    if (info.hasElasticInfo) {
        cumSumUbSize +=
            Ceil((ELASTIC_INFO_OFFSET + RANK_LIST_NUM * info.epWorldSize) * sizeof(int32_t), UB_ALIGN, nodeName);
    }
    cumSumUbSize += 2U * maxSizeForUbBuffer;
    cumSumUbSize += 2U * expertIdsBufSize;
    cumSumUbSize += statusCntAlign * UB_ALIGN;
    if (info.isExpertMask || (info.zeroComputeExpertNum != 0)) {
        uint32_t expertIdsSize = Ceil(info.bs * info.k * sizeof(int32_t), UB_ALIGN, nodeName);
        uint32_t xActivateMaskSize = info.bs * Ceil(info.k * sizeof(bool), UB_ALIGN, nodeName) * SIZE_OF_HALF;
        cumSumUbSize += Ceil(info.bs * sizeof(int32_t), UB_ALIGN, nodeName);
        cumSumUbSize += std::max(expertIdsSize, xActivateMaskSize);
    }
    cumSumUbSize += Ceil(recStatusNumPerCore * UB_ALIGN, SIZE_ALIGN_256, nodeName);
    uint32_t recStatusSpace = Ceil(recStatusNumPerCore * sizeof(float), UB_ALIGN, nodeName);
    uint32_t recvWinSpace = Ceil(info.epWorldSize * moeExpertNumPerRank * sizeof(float), UB_ALIGN, nodeName);
    cumSumUbSize += std::max(recStatusSpace, recvWinSpace);
    uint32_t histoExpertNum =
        mc2tiling::GetNpuArch(context) == Ops::Base::DAV_3510 ? statusCntAlign : info.moeExpertNum;
    uint32_t histoGroupNum =
        (histoExpertNum + Mc2Kernel::EXPERT_PER_HISTO_GROUP - 1U) / Mc2Kernel::EXPERT_PER_HISTO_GROUP;
    histoGroupNum = (histoGroupNum == 0U) ? 1U : histoGroupNum;
    cumSumUbSize += histoGroupNum * Mc2Kernel::EXPERT_HISTO_BIN_NUM * sizeof(uint16_t);
    cumSumUbSize += 2U * aivNum * UB_ALIGN;
    cumSumUbSize += Ceil(aivNum * sizeof(float), UB_ALIGN, nodeName);
    cumSumUbSize += 3U * UB_ALIGN;
    if (info.isPerformance) {
        cumSumUbSize += Ceil(info.epWorldSize * sizeof(int64_t), UB_ALIGN, nodeName);
        cumSumUbSize += Ceil(recStatusNumPerCore * sizeof(int32_t), UB_ALIGN, nodeName);
        if (info.hasElasticInfo) {
            cumSumUbSize +=
                Ceil((ELASTIC_INFO_OFFSET + RANK_LIST_NUM * info.epWorldSize) * sizeof(int32_t), UB_ALIGN, nodeName);
        }
    }
    cumSumUbSize += Ceil(moeExpertNumPerRank * sizeof(int64_t), UB_ALIGN, nodeName);
    return cumSumUbSize;
}

static ge::graphStatus CheckCumSumUbSizeA5(const gert::TilingContext* context, const char* nodeName,
                                           const MoeDistributeDispatchV2TilingData& tilingData,
                                           uint32_t expertIdsBufSize, uint32_t maxSizeForUbBuffer,
                                           const DispatchV2Config& config)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubRealSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubRealSize);
    uint32_t cumsumUbBudget = 0;
    OP_TILING_CHECK(CalcA5CumsumUbBudget(ubRealSize, nodeName, cumsumUbBudget) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Calc cumsum UB budget failed."), return ge::GRAPH_FAILED);
    const auto& info = tilingData.moeDistributeDispatchV2Info;
    if (IsFullMeshSerialMode(info)) {
        OP_TILING_CHECK(CheckSerialReceiveUbSizeA5(context, nodeName, tilingData, config) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "Check serial receive UB size failed."), return ge::GRAPH_FAILED);
        if (!info.hasElasticInfo) {
            return ge::GRAPH_SUCCESS;
        }
    }
    uint32_t recStatusNumPerCore = CalcCumSumRecStatusNumPerCore(info, aivNum, nodeName);
    uint32_t statusCntAlign = CalcCumSumStatusCntAlign(info, aivNum, nodeName);
    uint32_t cumSumUbSize = CalcCumSumUsedUbSize(context, info, recStatusNumPerCore, statusCntAlign, expertIdsBufSize,
                                                 maxSizeForUbBuffer, aivNum, nodeName);
    OP_TILING_CHECK(
        cumSumUbSize > cumsumUbBudget,
        OP_LOGE(nodeName,
                "Cumsum-core UB exceeds budget: used=%uB, budget=%uB, moeExpertNum=%u, sharedExpertRankNum=%u, "
                "epWorldSize=%u, BS=%u, topK=%u, aivNum=%u.",
                cumSumUbSize, cumsumUbBudget, info.moeExpertNum, info.sharedExpertRankNum, info.epWorldSize, info.bs,
                info.k, aivNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static uint64_t CalcMteSendUbSize(const gert::TilingContext* context, const MoeDistributeDispatchV2Info& info,
                                  const DispatchV2Config& config, uint32_t aivNum, uint64_t ubSize,
                                  const char* nodeName)
{
    uint32_t expertIdsCnt = info.bs * info.k;
    uint32_t expertIdsSize = Ceil(expertIdsCnt * sizeof(int32_t), UB_ALIGN, nodeName);
    uint32_t expertIdsBufSize = std::max(expertIdsCnt * static_cast<uint32_t>(sizeof(int32_t)),
                                         Ceil(info.bs * SIZE_OF_HALF, SIZE_ALIGN_256, nodeName));
    uint32_t maskSize = info.bs * Ceil(info.k * sizeof(bool), UB_ALIGN, nodeName) * SIZE_OF_HALF;
    uint32_t hFp32Size = info.h * sizeof(float);
    if (info.quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT) &&
        context->GetOptionalInputShape(config.scalesIndex) != nullptr) {
        hFp32Size = Ceil(info.h, PERGROUP_BLOCK_SIZE, nodeName) * sizeof(float);
    }
    uint32_t maxSize = std::max(std::max(hFp32Size, expertIdsSize),
                                std::max(maskSize, Ceil(expertIdsCnt * SIZE_OF_HALF, SIZE_ALIGN_256, nodeName)));
    uint32_t statusCount = Ceil(info.moeExpertNum + info.sharedExpertRankNum, aivNum, nodeName) / aivNum;
    uint32_t statusSize = Ceil(statusCount, UB_ALIGN / sizeof(int32_t), nodeName) * UB_ALIGN;
    uint64_t sendUb =
        UB_ALIGN + statusSize + Ceil(expertIdsBufSize, UB_ALIGN, nodeName) + Ceil(maxSize, UB_ALIGN, nodeName);
    if (info.hasElasticInfo) {
        sendUb += Ceil((ELASTIC_INFO_OFFSET + RANK_LIST_NUM * info.epWorldSize) * sizeof(int32_t), UB_ALIGN, nodeName);
    }
    if (info.isExpertMask || info.zeroComputeExpertNum != 0) {
        sendUb += Ceil(info.bs * sizeof(int32_t), UB_ALIGN, nodeName) + std::max(expertIdsSize, maskSize);
    }
    if (info.hasExpertScales) {
        sendUb += Ceil(expertIdsBufSize, UB_ALIGN, nodeName);
    }
    sendUb += BUFFER_NUM * Ceil(maxSize, UB_ALIGN, nodeName);
    bool needCastBuf = NeedA5CastBuf(context, info, config);
    if (needCastBuf && info.quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        sendUb += UB_ALIGN;
    }
    uint32_t hAlignSize = CalcA5TokenInAlignSize(context, info, config, nodeName);
    uint32_t hOutAlignUbSize = CalcA5TokenOutAlignSize(context, info, nodeName);
    uint32_t alignedQueueSize = needCastBuf ? Ceil(hAlignSize, UB_ALIGN, nodeName) + hOutAlignUbSize : hOutAlignUbSize;
    uint32_t queueDepth = sendUb + BUFFER_NUM * alignedQueueSize > ubSize ? 1U : BUFFER_NUM;
    return sendUb + queueDepth * alignedQueueSize;
}

static uint64_t CalcMteWaitUbSize(const gert::TilingContext* context, const MoeDistributeDispatchV2Info& info,
                                  uint32_t aivNum, const char* nodeName)
{
    uint32_t moeExpertNumPerRank = info.moeExpertNum / (info.epWorldSize - info.sharedExpertRankNum);
    uint32_t recvWinBlockNum = info.epWorldSize * moeExpertNumPerRank;
    uint32_t recStatusNumPerCore = Ceil(recvWinBlockNum, aivNum, nodeName) / aivNum;
    uint32_t hOutAlignUbSize = CalcA5TokenOutAlignSize(context, info, nodeName);
    uint64_t waitUb = std::max<uint32_t>(recStatusNumPerCore * UB_ALIGN, SIZE_ALIGN_256);
    waitUb += 2U * Ceil(recvWinBlockNum * sizeof(float), UB_ALIGN, nodeName);
    waitUb += 2U * aivNum * UB_ALIGN + 3U * UB_ALIGN;
    waitUb += BUFFER_NUM * hOutAlignUbSize;
    waitUb += Ceil(moeExpertNumPerRank * sizeof(int64_t), UB_ALIGN, nodeName);
    waitUb += recvWinBlockNum * UB_ALIGN;
    waitUb += UB_ALIGN + BUFFER_NUM * hOutAlignUbSize;
    if (info.hasElasticInfo) {
        waitUb += Ceil((ELASTIC_INFO_OFFSET + RANK_LIST_NUM * info.epWorldSize) * sizeof(int32_t), UB_ALIGN, nodeName);
    }
    if (info.isPerformance) {
        waitUb += Ceil(info.epWorldSize * sizeof(int64_t), UB_ALIGN, nodeName);
        waitUb += Ceil(recStatusNumPerCore * sizeof(int32_t), UB_ALIGN, nodeName);
    }
    return waitUb;
}

static ge::graphStatus CheckMteUbSizeA5(const gert::TilingContext* context, const char* nodeName,
                                        const MoeDistributeDispatchV2TilingData& tilingData,
                                        const DispatchV2Config& config)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_TILING_CHECK(aivNum == 0U, OP_LOGE(nodeName, "A5 AIV core count must be positive."), return ge::GRAPH_FAILED);
    const auto& info = tilingData.moeDistributeDispatchV2Info;
    uint64_t sendUb = CalcMteSendUbSize(context, info, config, aivNum, ubSize, nodeName);
    uint64_t waitUb = CalcMteWaitUbSize(context, info, aivNum, nodeName);
    OP_TILING_CHECK(sendUb > ubSize || waitUb > ubSize,
                    OP_LOGE(nodeName, "A5 fullmesh_v1 UB exceeds limit: send=%luB, wait=%luB, available=%luB.", sendUb,
                            waitUb, ubSize),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckUBSize(const gert::TilingContext* context, MoeDistributeDispatchV2TilingData& tilingData,
                                   const char* nodeName, const DispatchV2Config& config)
{
    uint32_t ubSize = UB_ALIGN; // dataStateBuf
    std::string ubDetail = "dataStateBuf:" + std::to_string(UB_ALIGN);
    const bool isArch22 = (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_2201);

    uint32_t quantMode = tilingData.moeDistributeDispatchV2Info.quantMode;
    uint32_t hSize = tilingData.moeDistributeDispatchV2Info.h;
    uint32_t bsSize = tilingData.moeDistributeDispatchV2Info.bs;
    uint32_t kSize = tilingData.moeDistributeDispatchV2Info.k;
    uint32_t expertIdsCnt = bsSize * kSize;
    uint32_t expertIdsBufSize = 0;
    uint32_t hOutSizeAlign = 0;
    uint32_t basicUsedBuffer = AllToAllBasicUsedBuffer(context, tilingData, quantMode, expertIdsBufSize, hSize,
                                                       expertIdsCnt, hOutSizeAlign, nodeName, config);
    ubSize = ubSize + basicUsedBuffer; // elasticInfoBuf_ + inQueue + outBuf + expertIdsBuf_
    ubDetail += "; basic(elastic+inQueue+outBuf+expertIds):" + std::to_string(basicUsedBuffer);

    const gert::StorageShape* xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
    const gert::StorageShape* scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
    bool isActiveMask = (xActiveMaskStorageShape != nullptr);
    bool isScales = (scalesStorageShape != nullptr);
    bool needMaskCalFlag = (isActiveMask || tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum != 0);
    bool hasExpertMaskOrZeroCompute = (tilingData.moeDistributeDispatchV2Info.isExpertMask ||
                                       (tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum != 0));
    bool reuseValidExpertIndexForTemp = !isArch22 && hasExpertMaskOrZeroCompute;

    uint32_t expertIdsSize = Ceil(expertIdsCnt * sizeof(int32_t), UB_ALIGN, nodeName);
    uint32_t xActivateMaskSize = bsSize * Ceil(kSize * sizeof(bool), UB_ALIGN, nodeName) * SIZE_OF_HALF;
    uint32_t maxSizeForUbBuffer =
        SetMaxSizeForUbBuffer(tilingData, hSize, bsSize, kSize, expertIdsCnt, isScales, nodeName);

    uint32_t calTokenIdxBuffSize = Ceil(expertIdsCnt * sizeof(int32_t), UB_ALIGN, nodeName);
    if (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_3510 &&
        IsFullMeshSerialMode(tilingData.moeDistributeDispatchV2Info)) {
        calTokenIdxBuffSize = std::max(calTokenIdxBuffSize, SIZE_ALIGN_256);
    }
    uint32_t calTempBufSize = calTokenIdxBuffSize; // A3 维持原公式
    if (!isArch22) {
        // 规约只需保存每64个float的中间结果（与 kernel 侧 reduceBytes 公式对齐）
        calTempBufSize = expertIdsCnt <= 256U ?
                             UB_ALIGN :
                             Ceil(Ceil(expertIdsCnt, 64U, nodeName) / 64U * sizeof(float), SIZE_ALIGN_256, nodeName);
    }
    uint32_t gatherMaskBufSize = needMaskCalFlag ? expertIdsBufSize : 0;
    uint32_t maskAndWorkBufSize =
        isArch22 ? (gatherMaskBufSize + calTokenIdxBuffSize) : (needMaskCalFlag ? expertIdsBufSize : calTempBufSize);
    ubSize = ubSize + maskAndWorkBufSize; // gatherMaskTBuf_ / calTempBuf
    ubDetail += "; gatherMask+calTemp:" + std::to_string(maskAndWorkBufSize);

    uint32_t expertScalesBufSize =
        (tilingData.moeDistributeDispatchV2Info.hasExpertScales && isArch22) ? expertIdsBufSize : 0;
    ubSize = ubSize + expertScalesBufSize; // expertScalesBuf_
    ubDetail += "; expertScalesBuf:" + std::to_string(expertScalesBufSize);

    uint32_t quantOrMaskBufSize = 0;
    uint32_t hOutAlignUbSize = 0;
    const auto* xDesc = context->GetInputDesc(config.xIndex);
    const auto* expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    bool isDirectCast = (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) && (xDesc != nullptr) &&
                        (expandXDesc != nullptr) && (xDesc->GetDataType() != expandXDesc->GetDataType());
    if (quantMode > static_cast<uint32_t>(QuantModeA5::NON_QUANT) || isDirectCast) {
        hOutAlignUbSize = Ceil(hOutSizeAlign, isArch22 ? UB_ALIGN : SPLIT_BLOCK_DATA_SIZE, nodeName);
        uint32_t tempBufSize = reuseValidExpertIndexForTemp ? 0 : hOutAlignUbSize;
        ubSize = ubSize + tempBufSize;
        ubSize = ubSize + BUFFER_NUM * maxSizeForUbBuffer; // receiveDataCastFloatBuf smoothScalesBuf
        quantOrMaskBufSize = tempBufSize + BUFFER_NUM * maxSizeForUbBuffer;
    } else if (needMaskCalFlag) {
        ubSize = ubSize + BUFFER_NUM * maxSizeForUbBuffer;
        quantOrMaskBufSize = BUFFER_NUM * maxSizeForUbBuffer;
    } else {
        ubSize = ubSize + BUFFER_NUM * calTokenIdxBuffSize; // dstExpBuf_ / subExpBuf_
        quantOrMaskBufSize = BUFFER_NUM * calTokenIdxBuffSize;
    }
    ubDetail += "; quantOrMaskBufs:" + std::to_string(quantOrMaskBufSize);

    uint32_t validIdxBufSize = 0;
    if (hasExpertMaskOrZeroCompute) {
        uint32_t axisBSAlign = Ceil(bsSize * sizeof(int32_t), UB_ALIGN, nodeName);
        ubSize = ubSize + axisBSAlign; // validBsIndexTBuf_
        uint32_t validBufferSize = expertIdsSize > xActivateMaskSize ? expertIdsSize : xActivateMaskSize;
        if (reuseValidExpertIndexForTemp && (hOutAlignUbSize > validBufferSize)) {
            validBufferSize = hOutAlignUbSize;
        }
        ubSize = ubSize + validBufferSize; // validExpertIndexBuf_
        validIdxBufSize = axisBSAlign + validBufferSize;
    }
    ubDetail += "; validBsIndex+validExpertIndex:" + std::to_string(validIdxBufSize);

    uint32_t sendToMoeUsedBuffer = SendToMoeExpertUsedBuffer(context, tilingData, expertIdsBufSize, kSize, nodeName);
    ubSize = ubSize + sendToMoeUsedBuffer;
    ubDetail += "; sendToMoeExpertBufs:" + std::to_string(sendToMoeUsedBuffer);

    if (!isArch22) {
        ubSize += 2U * UB_ALIGN; // 首尾计量buffer（calBeginBuf_/calEndBuf_）
        uint32_t groupIndexSize = 0;
        if (config.isMc2Context) {
            // V3分组开关在device context中，预留组内索引空间
            groupIndexSize = Ceil(expertIdsCnt * sizeof(int16_t), UB_ALIGN, nodeName);
            ubSize += groupIndexSize;
        }
        ubDetail += "; metering+groupIndex:" + std::to_string(2U * UB_ALIGN + groupIndexSize);
    }
    if (mc2tiling::GetSocVersion(context) == "Ascend910_93") {
        ubSize = A3FullMeshUsedBuffer(context, tilingData, expertIdsBufSize, hOutSizeAlign, config, nodeName);
    }
    ubDetail += "; total:" + std::to_string(ubSize);
    if (mc2tiling::GetNpuArch(context) != Ops::Base::DAV_2201) {
        OP_TILING_CHECK(CheckCumSumUbSizeA5(context, nodeName, tilingData, expertIdsBufSize, maxSizeForUbBuffer,
                                            config) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "Check cumsum core UB size failed."), return ge::GRAPH_FAILED);
    }
    return CheckUsedUbSize(context, tilingData, nodeName, ubSize, bsSize, hSize, kSize, ubDetail);
}

static ge::graphStatus SetWorkSpace(gert::TilingContext* context, const char* nodeName)
{
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workSpaces"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + static_cast<size_t>(WORKSPACE_ELEMENT_OFFSET * aivNum * aivNum);
    NpuArch npuArch = mc2tiling::GetNpuArch(context);
    if (npuArch == Ops::Base::DAV_3510 || npuArch == Ops::Base::DAV_9201 || npuArch == Ops::Base::DAV_9202) {
        // A5为前缀和同步标记预留空间
        workSpaces[0] +=
            A5_CUMSUM_WS_FLAG_OFFSET + static_cast<uint64_t>(aivNum) * A5_CUMSUM_WORKSPACE_MAX_CORE_NUM * UB_ALIGN_VAL;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAndGetOptionalInput(gert::TilingContext* context, const char* nodeName,
                                         MoeDistributeDispatchV2TilingData* tilingData, DispatchV2Config& config,
                                         bool& isScales, bool& isActiveMask, bool& hasElasticInfo, bool& isPerformance)
{
    // 获取scales
    const gert::StorageShape* scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
    isScales = (scalesStorageShape != nullptr);

    // 获取xActiveMask
    const gert::StorageShape* xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
    isActiveMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.isTokenMask =
        ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    tilingData->moeDistributeDispatchV2Info.isExpertMask =
        ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));

    // 获取elasticInfo
    const gert::StorageShape* elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
    hasElasticInfo = (elasticInfoStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.hasElasticInfo = hasElasticInfo;

    // 获取performanceInfo
    const gert::StorageShape* performanceInfoStorageShape = context->GetOptionalInputShape(config.performanceInfoIndex);
    isPerformance = (performanceInfoStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.isPerformance = isPerformance;

    const gert::StorageShape* expertScalesStorageShape = context->GetOptionalInputShape(config.expertScalesIndex);
    tilingData->moeDistributeDispatchV2Info.hasExpertScales = (expertScalesStorageShape != nullptr);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertScales(const gert::TilingContext* context, const char* nodeName, const uint32_t bs,
                                         const uint32_t k, DispatchV2Config& config)
{
    const gert::StorageShape* expertScalesStorageShape = context->GetOptionalInputShape(config.expertScalesIndex);
    OP_TILING_CHECK(expertScalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertScales"),
                    return ge::GRAPH_FAILED);
    auto expertScalesDesc = context->GetOptionalInputDesc(config.expertScalesIndex);
    OP_TILING_CHECK(expertScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertScalesDesc"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        expertScalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "expertScales", std::to_string(expertScalesStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of expertScales must be 2D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertScalesDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "expertScales",
                                                          Ops::Base::ToString(expertScalesDesc->GetDataType()).c_str(),
                                                          "The dtype of expertScales must be float."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertScalesDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expertScales", "not FRACTAL_NZ", "FRACTAL_NZ"), return ge::GRAPH_FAILED);

    const int64_t expertScalesDim0 = expertScalesStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertScalesDim1 = expertScalesStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expertScalesDim0 != static_cast<int64_t>(bs),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "expertScales(dim0)", std::to_string(expertScalesDim0).c_str(),
                                              ("should equal to bs=" + std::to_string(bs)).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertScalesDim1 != static_cast<int64_t>(k),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "expertScales(dim1)", std::to_string(expertScalesDim1).c_str(),
                                              ("should equal to k=" + std::to_string(k)).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpandScales(const gert::TilingContext* context, const char* nodeName, const uint32_t a)
{
    const gert::StorageShape* expandScalesStorageShape = context->GetOutputShape(OUTPUT_EXPAND_SCALES_INDEX);
    OP_TILING_CHECK(expandScalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandScales"),
                    return ge::GRAPH_FAILED);
    auto expandScalesDesc = context->GetOutputDesc(OUTPUT_EXPAND_SCALES_INDEX);
    OP_TILING_CHECK(expandScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandScalesDesc"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        expandScalesStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "expandScales", std::to_string(expandScalesStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of expandScales must be 1D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expandScalesDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "expandScales",
                                                          Ops::Base::ToString(expandScalesDesc->GetDataType()).c_str(),
                                                          "The dtype of expandScales must be float."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expandScalesDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expandScales", "not FRACTAL_NZ", "FRACTAL_NZ"), return ge::GRAPH_FAILED);

    const int64_t expandScalesDim0 = expandScalesStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(expandScalesDim0 < static_cast<int64_t>(a),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "expandScales(dim0)", std::to_string(expandScalesDim0).c_str(),
                                              ("should be >= a=" + std::to_string(a)).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckInputParam(gert::TilingContext* context, const char* nodeName, const bool isActiveMask,
                                const bool isScales, const bool hasElasticInfo, const bool isPerformance,
                                const uint32_t quantMode, uint32_t& localMoeExpertNum, const bool isLayered,
                                const bool isSetFullMeshV2, MoeDistributeDispatchV2TilingData* tilingData,
                                DispatchV2Config& config)
{
    // 检查属性的取值是否合法
    OP_TILING_CHECK(CheckAttrs(context, nodeName, *tilingData, localMoeExpertNum, isActiveMask, isSetFullMeshV2,
                               isLayered, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check attr failed."), return ge::GRAPH_FAILED);

    uint32_t epRankId = tilingData->moeDistributeDispatchV2Info.epRankId;
    uint32_t sharedExpertRankNum = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    bool isSharedExpert = (epRankId < sharedExpertRankNum);

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(CheckTensorShape(context, nodeName, *tilingData, quantMode, isScales, isSharedExpert,
                                     hasElasticInfo, isPerformance, static_cast<int64_t>(localMoeExpertNum), isLayered,
                                     config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check tensor shape failed."), return ge::GRAPH_FAILED);

    if (tilingData->moeDistributeDispatchV2Info.hasExpertScales) {
        const uint32_t aVal = tilingData->moeDistributeDispatchV2Info.a;
        const uint32_t bsVal = tilingData->moeDistributeDispatchV2Info.bs;
        const uint32_t kVal = tilingData->moeDistributeDispatchV2Info.k;
        OP_TILING_CHECK(CheckExpertScales(context, nodeName, bsVal, kVal, config) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "CheckExpertScales failed."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(CheckExpandScales(context, nodeName, aVal) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "CheckExpandScales failed."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// rsw_buffer挡位决策: 与combine侧同为A5 MTE、非跨超、window足够时启用; 否则保持0(旧协议)
// 必须与combine侧决策完全一致, 否则dispatch区起始偏移会与combine实际接收区错位
static void SetRswBufferSize(gert::TilingContext* context, MoeDistributeDispatchV2TilingData& tilingData,
                             uint32_t localMoeExpertNum, bool isLayered)
{
    tilingData.moeDistributeDispatchV2Info.rswBufferSize = 0UL;
    NpuArch npuArch = mc2tiling::GetNpuArch(context);
    if (npuArch != Ops::Base::DAV_9201 && npuArch != Ops::Base::DAV_9202) {
        return;
    }
    if (isLayered) {
        return;
    }
    const uint64_t h = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.h);
    const uint64_t axisBS = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.bs);
    const uint64_t epWorldSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    const uint64_t flagRcvCount = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.k) +
                                  static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.sharedExpertNum);
    if ((epWorldSize == 0UL) || (flagRcvCount == 0UL)) {
        return;
    }
    const uint64_t rswTier = GetRswBufferSize(h);
    if (rswTier == 0UL) {
        return;
    }
    uint64_t maxBs = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.globalBs) / epWorldSize;
    maxBs = maxBs > axisBS ? maxBs : axisBS;
    const uint64_t tokenActualLen =
        ((h * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN + SCALE_EXPAND_IDX_BUFFER;
    const uint64_t tokenNeedSizeDispatch = ((tokenActualLen + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    const uint64_t dispatchNeed =
        maxBs * tokenNeedSizeDispatch * epWorldSize * static_cast<uint64_t>(localMoeExpertNum);
    const uint64_t rswCombineNeed = maxBs * flagRcvCount * rswTier;
    if ((dispatchNeed + rswCombineNeed) * DOUBLE_DATA_BUFFER > tilingData.moeDistributeDispatchV2Info.totalWinSizeEp) {
        return;
    }
    tilingData.moeDistributeDispatchV2Info.rswBufferSize = rswTier;
}

ge::graphStatus CheckSizeAndSetRelevantCfg(gert::TilingContext* context, MoeDistributeDispatchV2TilingData* tilingData,
                                           const char* nodeName, const uint32_t localMoeExpertNum,
                                           const std::string groupEp, const bool isLayered, const bool isSetFullMeshV2,
                                           DispatchV2Config& config)
{
    // 校验UB大小
    if (isSetFullMeshV2) {
        OP_TILING_CHECK(CheckUBSize(context, *tilingData, nodeName, config) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "Tiling check UB size failed."), return ge::GRAPH_FAILED);
    } else if (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_3510 && !isLayered) {
        OP_TILING_CHECK(CheckMteUbSizeA5(context, nodeName, *tilingData, config) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "fullmesh_v1 UB size check failed."), return ge::GRAPH_FAILED);
    }

    // 校验win区大小
    OP_TILING_CHECK(!config.isRoce && CheckAndCalWinSize(context, *tilingData, nodeName, isSetFullMeshV2,
                                                         localMoeExpertNum, isLayered, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check window size failed."), return ge::GRAPH_FAILED);

    // rsw_buffer挡位决策(须在totalWinSizeEp确定后), 与combine侧一致, 决定dispatch区起始偏移
    SetRswBufferSize(context, *tilingData, localMoeExpertNum, isLayered);
    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    if (!config.isMc2Context) {
        OP_TILING_CHECK(SetHcommCfg(context, tilingData, groupEp, isLayered) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "Tiling set hcomm cfg failed."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAndSetPlatformInfo(gert::TilingContext* context, MoeDistributeDispatchV2TilingData* tilingData,
                                        const bool isLayered, const char* nodeName)
{
    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t epWorldSize = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    uint32_t serverNum = epWorldSize / RANK_NUM_PER_NODE;
    OP_TILING_CHECK(
        (isLayered && ((aivNum < RANK_NUM_PER_NODE) || (aivNum <= 2 * serverNum))),
        OP_LOGE(nodeName, "Layered should use aiv num greater than 16 and 2 * serverNum, now is %u", aivNum),
        return ge::GRAPH_FAILED);
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    context->SetScheduleMode(1); // 设置为batch mode模式, 所有核同时启动
    tilingData->moeDistributeDispatchV2Info.totalUbSize = ubSize;
    tilingData->moeDistributeDispatchV2Info.aivNum = aivNum;
    OP_LOGD(nodeName, "numBlocks = %u, aivNum = %u, ubSize = %lu", numBlocks, aivNum, ubSize);
    return ge::GRAPH_SUCCESS;
}

static void SetCumsumTmpMinSize(MoeDistributeDispatchV2TilingData& tilingData, uint32_t localMoeExpertNum,
                                const char* nodeName)
{
    if (tilingData.moeDistributeDispatchV2Info.expertTokenNumsType != 0) {
        tilingData.moeDistributeDispatchV2Info.cumsumTmpMinSize = 0;
        return;
    }
    uint32_t localExpertNumAlign = (localMoeExpertNum + 7U) / 8U * 8U;
    std::vector<int64_t> cumsumShapeDims = {1, static_cast<int64_t>(localExpertNumAlign)};
    auto cumsumSrcShape = ge::Shape(cumsumShapeDims);
    uint32_t cumsumMaxTmp = 0;
    uint32_t cumsumMinTmp = 0;
    GetCumSumMaxMinTmpSize(cumsumSrcShape, sizeof(float), true, false, cumsumMaxTmp, cumsumMinTmp);
    tilingData.moeDistributeDispatchV2Info.cumsumTmpMinSize = cumsumMinTmp;
    OP_LOGI(nodeName, "MTE cumsumTmpMinSize=%u. Kernel falls back to scalar prefix sum if UB is insufficient.",
            cumsumMinTmp);
}

static uint64_t GetRoceFlagBufferBytes(uint64_t flagCount)
{
    const uint64_t compareBytes = AlignRoceBytes(flagCount * sizeof(uint64_t), 256U);
    const uint64_t gatherBytes = AlignRoceBytes(flagCount * sizeof(uint32_t));
    const uint64_t maskBytes = AlignRoceBytes(compareBytes / sizeof(uint64_t), 256U) / 8U;
    return compareBytes + gatherBytes + maskBytes;
}

static uint64_t GetRoceExpertsPerServer(const MoeDistributeDispatchV2Info& info, uint64_t serverNum)
{
    const uint64_t ranksPerServer = info.epWorldSize / serverNum;
    const uint64_t firstRank = info.epRankId / ranksPerServer * ranksPerServer;
    const uint64_t sharedRanks =
        firstRank < info.sharedExpertRankNum ? std::min(ranksPerServer, info.sharedExpertRankNum - firstRank) : 0U;
    const uint64_t expertsPerRank = info.moeExpertNum / (info.epWorldSize - info.sharedExpertRankNum);
    return sharedRanks + (ranksPerServer - sharedRanks) * expertsPerRank;
}

struct RoceDispatchUbLayout {
    uint64_t inputBytes;
    uint64_t serverPacketBytes;
    uint64_t expertPacketBytes;
    uint64_t scratchBytes;
    uint64_t quantScratchBytes;
    uint64_t flagBytes;
};

static RoceDispatchUbLayout GetRoceDispatchUbLayout(const gert::TilingContext* context,
                                                    const MoeDistributeDispatchV2Info& info,
                                                    const DispatchV2Config& config, bool isScales)
{
    const auto inputType = context->GetInputDesc(config.xIndex)->GetDataType();
    const auto outputType = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX)->GetDataType();
    const bool isInputFp4 = inputType == ge::DT_FLOAT4_E2M1 || inputType == ge::DT_FLOAT4_E1M2;
    const auto isFp4 = outputType == ge::DT_FLOAT4_E2M1 || outputType == ge::DT_FLOAT4_E1M2;
    // FP4 reports a bit-width encoding; QuantInit reserves H * sizeof(uint8_t) for its input queue.
    const uint64_t inputSize = isInputFp4 ? sizeof(uint8_t) : std::max<int64_t>(ge::GetSizeByDataType(inputType), 1);
    const uint64_t outputSize = std::max<int64_t>(ge::GetSizeByDataType(outputType), 1);
    const uint64_t hidden = info.h;
    const uint64_t mode = info.quantMode;
    RoceDispatchUbLayout layout{};
    layout.inputBytes = AlignRoceBytes(hidden * inputSize);
    uint64_t outputBytes = AlignRoceBytes(isFp4 ? (hidden + 1U) / 2U : hidden * outputSize);
    if (mode == 0U && isScales) {
        layout.inputBytes += info.scalesCol * info.scalesTypeSize;
        outputBytes += info.scalesCol * info.scalesTypeSize;
    } else if (mode == 2U) {
        outputBytes += sizeof(float);
    } else if (mode >= 3U) {
        layout.inputBytes = AlignRoceBytes(hidden, 128U) * inputSize;
        outputBytes = mode == 3U ? AlignRoceBytes(hidden, 128U) * outputSize + (hidden + 127U) / 128U * sizeof(float) :
                                   (isFp4 ? AlignRoceBytes((hidden + 1U) / 2U, 256U) :
                                            AlignRoceBytes(hidden, 256U) * outputSize) +
                                       AlignRoceBytes((hidden + 31U) / 32U, 2U);
    }
    outputBytes = AlignRoceBytes(outputBytes);
    layout.expertPacketBytes =
        AlignRoceBytes(outputBytes + 2U * UB_ALIGN, SPLIT_BLOCK_DATA_SIZE) / SPLIT_BLOCK_DATA_SIZE * SPLIT_BLOCK_SIZE;
    const uint64_t serverDataBytes = mode > 0U && isScales ? layout.inputBytes : outputBytes;
    const uint64_t idsBytes = AlignRoceBytes(static_cast<uint64_t>(info.k) * sizeof(int32_t));
    const uint64_t countsBytes = AlignRoceBytes(static_cast<uint64_t>(info.k) * sizeof(int16_t));
    layout.serverPacketBytes =
        AlignRoceBytes(serverDataBytes + UB_ALIGN + 2U * idsBytes + countsBytes, SPLIT_BLOCK_DATA_SIZE) /
        SPLIT_BLOCK_DATA_SIZE * SPLIT_BLOCK_SIZE;
    const uint64_t idsCount = static_cast<uint64_t>(info.bs) * info.k;
    const uint64_t floatBytes = (mode == 3U && isScales ? AlignRoceBytes(hidden, 128U) : hidden) * sizeof(float);
    layout.scratchBytes = AlignRoceBytes(
        std::max({floatBytes, AlignRoceBytes(idsCount * sizeof(int32_t)),
                  info.bs * AlignRoceBytes(info.k) * sizeof(uint16_t),
                  AlignRoceBytes(idsCount * sizeof(uint16_t), 256U) / 2U,
                  AlignRoceBytes(static_cast<uint64_t>(info.moeExpertNum) * sizeof(int32_t), SPLIT_BLOCK_DATA_SIZE),
                  info.bs * countsBytes, GetRoceFlagBufferBytes(info.aivNum - 2U)}),
        256U);
    layout.quantScratchBytes = mode > 0U || inputType != outputType ? 2U * layout.scratchBytes : 0U;
    const uint64_t expertCountBytes = static_cast<uint64_t>(info.moeExpertNum) /
                                      (info.epWorldSize - info.sharedExpertRankNum) *
                                      (info.epWorldSize / config.roceServerNum) * sizeof(int32_t);
    const uint64_t maxBlocks =
        std::max({layout.serverPacketBytes / SPLIT_BLOCK_SIZE, layout.expertPacketBytes / SPLIT_BLOCK_SIZE,
                  (expertCountBytes + SPLIT_BLOCK_DATA_SIZE - 1U) / SPLIT_BLOCK_DATA_SIZE});
    layout.flagBytes = AlignRoceBytes(maxBlocks * UB_ALIGN, 256U);
    return layout;
}

static uint64_t GetRocePreprocessUbBytes(const MoeDistributeDispatchV2Info& info, const DispatchV2Config& config,
                                         const RoceDispatchUbLayout& layout)
{
    constexpr uint64_t hcommBytes = 512U;
    // HcclCommSend reserves five WQEBBs plus one WQEBB for alignment.
    constexpr uint64_t batchBytes = (5U + 1U) * 64U;
    const uint64_t idsCount = static_cast<uint64_t>(info.bs) * info.k;
    const uint64_t maskBytes = info.isExpertMask ? AlignRoceBytes(idsCount) : 0U;
    const uint64_t baseBytes = 3U * layout.scratchBytes + maskBytes + UB_ALIGN +
                               AlignRoceBytes(static_cast<uint64_t>(config.roceServerNum) * sizeof(uint32_t));
    const uint64_t expertBlocks =
        (static_cast<uint64_t>(info.moeExpertNum) / (info.epWorldSize - info.sharedExpertRankNum) *
             (info.epWorldSize / config.roceServerNum) * sizeof(int32_t) +
         SPLIT_BLOCK_DATA_SIZE - 1U) /
        SPLIT_BLOCK_DATA_SIZE;
    const uint64_t countSendBytes =
        AlignRoceBytes(expertBlocks * UB_ALIGN, 256U) + hcommBytes + expertBlocks * SPLIT_BLOCK_SIZE;
    const uint64_t countReceiveBytes =
        std::max(expertBlocks * SPLIT_BLOCK_DATA_SIZE, GetRoceFlagBufferBytes(expertBlocks));
    const uint64_t statusBytes =
        AlignRoceBytes(GetRoceExpertsPerServer(info, config.roceServerNum) * config.roceServerNum, 8U) * UB_ALIGN;
    const uint64_t cumsumBytes = countSendBytes + countReceiveBytes + statusBytes;
    const uint64_t tokensPerRank = info.globalBs / info.epWorldSize;
    const uint64_t mapBytes = AlignRoceBytes(AlignRoceBytes(tokensPerRank * sizeof(uint32_t)) / sizeof(uint32_t));
    const uint64_t indexBytes = AlignRoceBytes(idsCount * sizeof(uint16_t));
    // Each sending core owns one server; its routing buffers survive until the pool is reset.
    const uint64_t routingBytes = indexBytes + 256U + mapBytes + AlignRoceBytes(idsCount * sizeof(int32_t), 256U);
    const uint64_t sendBytes = routingBytes + SPLIT_BLOCK_SIZE + hcommBytes + batchBytes +
                               GetRoceFlagBufferBytes(layout.serverPacketBytes / SPLIT_BLOCK_SIZE);
    // Cumsum and token sending run on different cores, but both retain the initial pool buffers.
    return baseBytes + std::max(cumsumBytes, sendBytes);
}

static uint64_t GetRoceCumsumUbBytes(const MoeDistributeDispatchV2Info& info)
{
    const uint64_t expertsPerRank = info.moeExpertNum / (info.epWorldSize - info.sharedExpertRankNum);
    const uint64_t receiveSlots =
        info.epRankId < info.sharedExpertRankNum ? info.epWorldSize : info.epWorldSize * expertsPerRank;
    const uint64_t slotsPerCore = (receiveSlots + 1U) / 2U;
    const uint64_t waitBytes =
        std::max(AlignRoceBytes(slotsPerCore * UB_ALIGN, 256U), GetRoceFlagBufferBytes(slotsPerCore));
    const uint64_t countsBytes =
        AlignRoceBytes(std::max(slotsPerCore, info.epWorldSize * expertsPerRank) * sizeof(float));
    const uint64_t coreSumBytes = AlignRoceBytes(static_cast<uint64_t>(info.aivNum) * sizeof(float));
    const uint64_t localSumBytes = std::max(info.aivNum * UB_ALIGN, GetRoceFlagBufferBytes(2U));
    const uint64_t statusBytes = receiveSlots * UB_ALIGN;
    const uint64_t tokenCountsBytes = AlignRoceBytes(expertsPerRank * sizeof(int64_t));
    const uint64_t workBytes = AlignRoceBytes(static_cast<uint64_t>(info.epWorldSize) * sizeof(float));
    return waitBytes + countsBytes + info.aivNum * UB_ALIGN + localSumBytes + coreSumBytes + 3U * UB_ALIGN +
           statusBytes + tokenCountsBytes + workBytes;
}

static bool CheckRoceDispatchUb(const gert::TilingContext* context, const MoeDistributeDispatchV2Info& info,
                                const DispatchV2Config& config, bool isScales)
{
    constexpr uint64_t poolBytes = 192U * 1024U;
    constexpr uint64_t kernelUbBytes = 248U * 1024U;
    const auto layout = GetRoceDispatchUbLayout(context, info, config, isScales);
    const auto window = GetRoceWindowLayout(info, config.roceServerNum);
    OP_TILING_CHECK(
        layout.expertPacketBytes > window.dispatchPacketBytes || layout.serverPacketBytes > window.serverPacketBytes,
        OP_LOGE(context->GetNodeName(), "ROCE token and scales exceed the shared window record capacity"),
        return false);
    const uint64_t tokensPerRank = info.globalBs / info.epWorldSize;
    const uint64_t idsBytes = AlignRoceBytes(static_cast<uint64_t>(info.bs) * info.k * sizeof(int32_t));
    const uint64_t offsetsBytes = info.bs * AlignRoceBytes(static_cast<uint64_t>(info.k) * sizeof(int16_t));
    const uint64_t mapBytes = AlignRoceBytes(tokensPerRank * sizeof(uint32_t));
    const uint64_t bitmapBytes = AlignRoceBytes(tokensPerRank * sizeof(uint16_t));
    const uint64_t residentBytes =
        idsBytes + AlignRoceBytes(mapBytes / sizeof(uint32_t)) + offsetsBytes + layout.flagBytes;
    const uint64_t preprocessBytes = GetRocePreprocessUbBytes(info, config, layout);
    const uint64_t scatterQueueBytes = layout.inputBytes + 2U * layout.serverPacketBytes;
    const uint64_t scatterBuffers = 2U * scatterQueueBytes + 2U * layout.scratchBytes > poolBytes ? 1U : 2U;
    const uint64_t scatterBytes = scatterBuffers * scatterQueueBytes + layout.quantScratchBytes;
    const uint64_t serverFlagBytes = GetRoceFlagBufferBytes(layout.serverPacketBytes / SPLIT_BLOCK_SIZE);
    const uint64_t receiveFlagBytes = std::max(serverFlagBytes, layout.serverPacketBytes / SPLIT_BLOCK_SIZE * UB_ALIGN);
    const uint64_t gatherFixedBytes = layout.flagBytes + 512U * sizeof(uint32_t) + receiveFlagBytes + mapBytes +
                                      bitmapBytes + GetRoceFlagBufferBytes(1U);
    const uint64_t gatherQueueBytes = layout.serverPacketBytes + std::max(layout.inputBytes, layout.expertPacketBytes) +
                                      2U * layout.expertPacketBytes;
    const uint64_t gatherBuffers =
        2U * gatherQueueBytes + 2U * layout.scratchBytes + gatherFixedBytes > kernelUbBytes ? 1U : 2U;
    const uint64_t perTokenScaleBytes = info.quantMode == 2U ? UB_ALIGN : 0U;
    const uint64_t gatherBytes =
        gatherBuffers * gatherQueueBytes + layout.quantScratchBytes + gatherFixedBytes + perTokenScaleBytes;
    OP_TILING_CHECK(std::max(preprocessBytes, scatterBytes) > poolBytes ||
                        poolBytes + residentBytes + perTokenScaleBytes > info.totalUbSize ||
                        std::max(gatherBytes, GetRoceCumsumUbBytes(info)) > std::min(info.totalUbSize, kernelUbBytes),
                    OP_LOGE(context->GetNodeName(),
                            "ROCE UB insufficient: pool=%lu, resident=%lu, preprocess=%lu, scatter=%lu, gather=%lu",
                            poolBytes, residentBytes, preprocessBytes, scatterBytes, gatherBytes),
                    return false);
    return true;
}

static ge::graphStatus CheckRoceDispatchInputs(const gert::TilingContext* context,
                                               const MoeDistributeDispatchV2Info& info, const DispatchV2Config& config)
{
    auto nodeName = context->GetNodeName();
    OP_TILING_CHECK(
        info.aivNum < config.roceServerNum + 4U,
        OP_LOGE(nodeName, "ROCE requires two cumsum cores, one sender per server, and Scatter/Gather cores"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(info.globalBs / info.epWorldSize > BS_UPPER_BOUND,
                    OP_LOGE(nodeName, "ROCE global_bs / ep_world_size must not exceed 512"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(info.sharedExpertRankNum != 0U && info.sharedExpertRankNum != info.sharedExpertNum,
                    OP_LOGE(nodeName, "ROCE requires one dedicated rank per shared expert"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(info.hasElasticInfo || info.isPerformance,
                    OP_LOGE(nodeName, "ROCE does not support elastic_info or performance_info"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!info.hasExpertScales, OP_LOGE(nodeName, "ROCE Dispatch requires expert_scales for route weights"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetRoceDispatchWindow(gert::TilingContext* context, MoeDistributeDispatchV2Info& info,
                                             const DispatchV2Config& config)
{
    auto cclBytes = context->GetAttrs()->GetAttrPointer<int64_t>(config.attrCclBufferSizeIndex);
    const auto layout = GetRoceWindowLayout(info, config.roceServerNum);
    const uint64_t reservedBytes = MB_SIZE + static_cast<uint64_t>(info.epWorldSize) * EP_RANK_OFFSET_STEP;
    const uint64_t intraBytes = reservedBytes + 2U * (layout.combineIntraBytes + layout.dispatchIntraBytes);
    OP_TILING_CHECK(
        cclBytes == nullptr || *cclBytes < 0 || static_cast<uint64_t>(*cclBytes) < intraBytes ||
            static_cast<uint64_t>(*cclBytes) / 2U < layout.dispatchInterBytes,
        OP_LOGE(context->GetNodeName(), "ROCE CCL window requires intra=%lu B, inter Dispatch=%lu B per bank",
                intraBytes, layout.dispatchInterBytes),
        return ge::GRAPH_FAILED);
    info.totalWinSizeEp = static_cast<uint64_t>(*cclBytes) - reservedBytes;
    OP_TILING_CHECK(info.totalWinSizeEp % 1024U != 0U,
                    OP_LOGE(context->GetNodeName(), "ROCE requires 512-byte-aligned window banks"),
                    return ge::GRAPH_FAILED);
    info.roceDispatchOffset = layout.combineIntraBytes;
    return ge::GRAPH_SUCCESS;
}

static void SetRoceDispatchWorkspace(gert::TilingContext* context, MoeDistributeDispatchV2Info& info,
                                     const DispatchV2Config& config)
{
    const uint64_t expertsPerServer = GetRoceExpertsPerServer(info, config.roceServerNum);
    const uint64_t expertsPerRank = info.moeExpertNum / (info.epWorldSize - info.sharedExpertRankNum);
    const uint64_t countRows =
        info.epRankId < info.sharedExpertRankNum ? info.epWorldSize : info.epWorldSize * expertsPerRank;
    const uint64_t statusBytes = info.epWorldSize * expertsPerServer * UB_ALIGN;
    const uint64_t receiveCountsBytes = AlignRoceBytes(info.aivNum * countRows * sizeof(int32_t));
    const uint64_t expertOffsetsBytes =
        (info.aivNum - 2U) * AlignRoceBytes(static_cast<uint64_t>(info.bs) * info.k * sizeof(uint16_t));
    const uint64_t readyFlagsBytes = AlignRoceBytes((info.aivNum - 2U) * sizeof(uint64_t));
    info.roceExpertOffset = statusBytes + receiveCountsBytes;
    info.roceTokenMapOffset = info.roceExpertOffset + expertOffsetsBytes + readyFlagsBytes;
    const uint64_t tokenMapBytes =
        AlignRoceBytes(static_cast<uint64_t>(info.globalBs / info.epWorldSize) * sizeof(int32_t));
    context->GetWorkspaceSizes(1)[0] = SYSTEM_NEED_WORKSPACE + info.roceTokenMapOffset + tokenMapBytes;
}

static ge::graphStatus CheckRoceDispatchTiling(gert::TilingContext* context, MoeDistributeDispatchV2Info& info,
                                               const DispatchV2Config& config, bool isScales)
{
    OP_TILING_CHECK(CheckRoceDispatchInputs(context, info, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "ROCE input check failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckRoceDispatchUb(context, info, config, isScales),
                    OP_LOGE(context->GetNodeName(), "ROCE UB check failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetRoceDispatchWindow(context, info, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "ROCE window check failed"), return ge::GRAPH_FAILED);
    SetRoceDispatchWorkspace(context, info, config);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FinalizeDispatchTiling(gert::TilingContext* context,
                                              MoeDistributeDispatchV2TilingData& tilingData, uint64_t tilingKey,
                                              bool isLayered, const char* nodeName, bool isMc2Context,
                                              uint32_t localMoeExpertNum)
{
    tilingData.moeDistributeDispatchV2Info.isMc2Context = isMc2Context;
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    OP_TILING_CHECK(CheckAndSetPlatformInfo(context, &tilingData, isLayered, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set platformInfo Failed"), return ge::GRAPH_FAILED);
    // CumSum 最小 tmp 大小
    SetCumsumTmpMinSize(tilingData, localMoeExpertNum, nodeName);
    PrintTilingDataInfo(nodeName, tilingData);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::MoeDistributeDispatchA3TilingFuncImplPublic(
    gert::TilingContext* context, DispatchV2Config& config)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE("MoeDistributeDispatchV2", "Fail to get tiling context."),
                    return ge::GRAPH_FAILED);
    const char* nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE("MoeDistributeDispatchV2", "Fail to get nodeName."),
                    return ge::GRAPH_FAILED);
    MoeDistributeDispatchV2TilingData* tilingData = context->GetTilingData<MoeDistributeDispatchV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);
    std::string groupEp = "";
    std::string groupTp = "";
    uint32_t quantMode = static_cast<uint32_t>(QuantModeA5::NON_QUANT);
    bool isScales = false;
    bool isActiveMask = false;
    bool hasElasticInfo = false;
    bool isPerformance = false;
    bool isSetFullMeshV2 = false;
    bool isLayered = false;
    uint32_t localMoeExpertNum = 1;
    OP_LOGI(nodeName, "Enter MoeDistributeDispatchV2 tiling check func.");

    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, groupEp, groupTp, isSetFullMeshV2,
                                            isLayered, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(nodeName, "GetAttrAndSetTilingData failed."), return ge::GRAPH_FAILED);

    // 检查并填充可选输入
    OP_TILING_CHECK(CheckAndGetOptionalInput(context, nodeName, tilingData, config, isScales, isActiveMask,
                                             hasElasticInfo, isPerformance) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check and get optional input param failed."), return ge::GRAPH_FAILED);

    quantMode = tilingData->moeDistributeDispatchV2Info.quantMode;
    // 检查quantMode和scales是否匹配
    OP_TILING_CHECK(CheckQuantModeMatchScales(context, nodeName, isScales, quantMode, config) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Check quant mode match scales failed."), return ge::GRAPH_FAILED);

    // 检查context输入
    if (config.isMc2Context) {
        OP_TILING_CHECK(CheckMc2Context(context, nodeName, config) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "Tiling check context failed."), return ge::GRAPH_FAILED);
    }

    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(TilingCheckMoeDistributeDispatch(context, nodeName, isActiveMask, isScales, hasElasticInfo,
                                                     isPerformance, quantMode, isLayered, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE_LIBOPAPI_REPORT(nodeName, "Tiling check param failed; propagate lower-layer error"),
                    return ge::GRAPH_FAILED);

    // 检查Input参数
    OP_TILING_CHECK(
        CheckInputParam(context, nodeName, isActiveMask, isScales, hasElasticInfo, isPerformance, quantMode,
                        localMoeExpertNum, isLayered, isSetFullMeshV2, tilingData, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check input param failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckSizeAndSetRelevantCfg(context, tilingData, nodeName, localMoeExpertNum, groupEp, isLayered,
                                               isSetFullMeshV2, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check UB/window size or set workspace/hcomm cfg failed."),
                    return ge::GRAPH_FAILED);

    uint64_t tilingKey = config.isRoce ? GET_TPL_TILING_KEY(quantMode, isScales, TILINGKEY_ENABLE_HIERARCHY,
                                                            TILINGKEY_TPL_ROCE, TILINGKEY_TPL_A5) :
                                         CalTilingKey(isScales, quantMode, isSetFullMeshV2, isLayered);
    OP_TILING_CHECK(FinalizeDispatchTiling(context, *tilingData, tilingKey, isLayered, nodeName, config.isMc2Context,
                                           localMoeExpertNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Finalize dispatch tiling failed"), return ge::GRAPH_FAILED);
    if (config.isRoce) {
        return CheckRoceDispatchTiling(context, tilingData->moeDistributeDispatchV2Info, config, isScales);
    }
    return ge::GRAPH_SUCCESS;
}

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
// Register exception func
inline void MoeDistributeDispatchV2ExceptionImplWrapper(aclrtExceptionInfo* args, void* userdata)
{
    const char* socName = aclrtGetSocName();
    if ((std::strstr(socName, "Ascend960") == nullptr) && (std::strstr(socName, "Ascend950") == nullptr) &&
        (std::strstr(socName, "Ascend910B") == nullptr)) {
        return;
    }
    Mc2Exception::Mc2ExceptionImpl(args, userdata, "MoeDistributeDispatchV2");
}

__attribute__((constructor)) void RegisterMoeDistributeDispatchV2ExceptionFunc()
{
    int32_t runtimeVersionNum = 0;
    int32_t metadefVersionNum = 0;

    if (aclsysGetVersionNum("runtime", &runtimeVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MoeDistributeDispatchV2", "Get runtime version failed when registering exception func.");
        return;
    }
    if (aclsysGetVersionNum("metadef", &metadefVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MoeDistributeDispatchV2", "Get metadef version failed when registering exception func.");
        return;
    }

    if (runtimeVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION || metadefVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION) {
        OP_LOGW("MoeDistributeDispatchV2",
                "The runtime(%d) or metadata(%d) version is lower than the version(%d) supporting exception func.",
                runtimeVersionNum, metadefVersionNum, EXCEPTION_DUMP_SUPPORT_VERSION);
        return;
    }

    IMPL_OP(MoeDistributeDispatchV2).ExceptionDumpParseFunc(MoeDistributeDispatchV2ExceptionImplWrapper);
}
#endif

} // namespace optiling
