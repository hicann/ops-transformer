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
constexpr int64_t BS_UPPER_BOUND_LAYERED = 256;
constexpr int64_t FULLMESH_BS_UPPER_BOUND_A3 = 256;
constexpr int64_t FULLMESH_BS_UPPER_BOUND_A5 = BS_UPPER_BOUND;
constexpr uint32_t H_BASIC_BLOCK_LAYERED = 32;
constexpr uint32_t RANK_NUM_PER_NODE = 16U;

constexpr uint32_t SIZE_OF_UNQUANT = 2;
constexpr uint32_t SIZE_OF_HALF = 2;
constexpr uint32_t SIZE_ALIGN_256 = 256;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_SIZE = 512UL;
constexpr uint32_t ELASTIC_INFO_OFFSET = 4U;
constexpr uint32_t BUFFER_NUM = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
constexpr int64_t MOE_EXPERT_MAX_NUM_LAYERED = 512;
constexpr int64_t LOCAL_EXPERT_MAX_SIZE = 2048;
constexpr int64_t K_MAX = 16;
constexpr int64_t FULLMESH_K_MAX = 12;
constexpr size_t SYSTEM_NEED_WORKSPACE = 16UL * 1024UL * 1024UL;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512;
// A5前缀和临时区位于recvCnt工作区之后，分为数据区和同步标记区
constexpr uint64_t A5_CUMSUM_WS_FLAG_OFFSET = 32UL * 1024UL; // 32*32*32=32KB
constexpr uint64_t A5_CUMSUM_WORKSPACE_MAX_CORE_NUM = 32U;
constexpr uint32_t A5_CUMSUM_MAX_CORE_NUM = 8U;
constexpr uint64_t UB_ALIGN_VAL = 32U;
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
} // namespace

// Supported x datatype in nonquant mode, the same as expandX
static const std::set<ge::DataType> NON_QUANT_DTYPE = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8,
                                                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, MoeDistributeDispatchV2TilingData &tilingData)
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

static bool CheckDynamicScalesDim(const gert::TilingContext *context, const char *nodeName, const uint32_t quantMode)
{
    const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
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

static bool CheckScaleTensorDim(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                                const uint32_t quantMode, DispatchV2Config &config)
{
    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scalesShape"),
                        return false);
        if (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)) {
            // the cond is compatible with A2/A3 because static quant is only supported on A5
            if (scalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS) {
                std::string dimStr = std::to_string(scalesStorageShape->GetStorageShape().GetDimNum()) + "D";
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "scales", dimStr.c_str(), "2D");
                return false;
            }
            OP_LOGD(nodeName, "scales dim0 = %ld", scalesStorageShape->GetStorageShape().GetDim(0));
            OP_LOGD(nodeName, "scales dim1 = %ld", scalesStorageShape->GetStorageShape().GetDim(1));
        } else {
            size_t scalesDimNum = scalesStorageShape->GetStorageShape().GetDimNum();
            if ((scalesDimNum != ONE_DIM) && (scalesDimNum != TWO_DIMS)) {
                std::string dimStr = std::to_string(scalesDimNum) + "D";
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "scales", dimStr.c_str(), "1D or 2D");
                return false;
            }
            // additional check for hif8 quant
            auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
            if (expandXDesc == nullptr) {
                OP_LOGE_WITH_INVALID_INPUT(nodeName, "expandXDesc");
                return false;
            }
            if ((expandXDesc->GetDataType() == ge::DT_HIFLOAT8) && (scalesDimNum != ONE_DIM)) {
                std::string dimStr = std::to_string(scalesDimNum) + "D";
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "scales", dimStr.c_str(), "1D");
                return false;
            }
            OP_LOGD(nodeName, "scales dim0 = %ld", scalesStorageShape->GetStorageShape().GetDim(0));
            if (scalesStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) {
                OP_LOGD(nodeName, "scales dim1 = %ld", scalesStorageShape->GetStorageShape().GetDim(1));
            }
        }
    }
    return true;
}

// x, expertIds, scales维度校验
static bool CheckInputTensorDim(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                                const uint32_t quantMode, const bool isLayered, DispatchV2Config &config)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
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

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    if (isLayered) {
        const gert::StorageShape *expertScaleStorageShape = context->GetOptionalInputShape(config.expertScalesIndex);
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

// expertX, assistInfo, expertTokenNums, epRecvCount, tpRecvCount维度校验
static bool CheckCommonOutputTensorDim(const gert::TilingContext *context, const char *nodeName,
                                       const uint32_t quantMode)
{
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
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

    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "assistInfoShape"),
                    return false);
    OP_TILING_CHECK(
        assistInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "assistInfoShape", std::to_string(assistInfoStorageShape->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of assistInfoShape must be 1D."),
        return false);
    OP_LOGD(nodeName, "assistInfoForCombine dim0 = %ld", assistInfoStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsStorageShape == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "expertTokenNumsShape"), return false);
    OP_TILING_CHECK(expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "expertTokenNumsShape",
                        std::to_string(expertTokenNumsStorageShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of expertTokenNumsShape must be 1D."),
                    return false);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epRecvCountShape"),
                    return false);
    OP_TILING_CHECK(epRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "epRecvCountShape",
                        std::to_string(epRecvCountStorageShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of epRecvCountShape must be 1D."),
                    return false);
    OP_LOGD(nodeName, "epRecvCount dim0 = %ld", epRecvCountStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tpRecvCountShape"),
                    return false);
    OP_TILING_CHECK(tpRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "tpRecvCountShape",
                        std::to_string(tpRecvCountStorageShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of tpRecvCountShape must be 1D."),
                    return false);
    OP_LOGD(nodeName, "tpRecvCount dim0 = %ld", tpRecvCountStorageShape->GetStorageShape().GetDim(0));

    return true;
}

static bool CheckTensorDim(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                           const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
                           const bool isPerformance, const bool isLayered, DispatchV2Config &config)
{
    OP_TILING_CHECK(!CheckInputTensorDim(context, nodeName, isScales, quantMode, isLayered, config),
                    OP_LOGE(nodeName, "Input param shape is invalid."), return false);
    if (isActiveMask) {
        const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
        int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
        const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
        const int64_t expertIdDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
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
    }
    if (hasElasticInfo) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "elasticInfo"),
                        return false);
        OP_TILING_CHECK(
            elasticInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                nodeName, "elasticInfo", std::to_string(elasticInfoStorageShape->GetStorageShape().GetDimNum()).c_str(),
                "The shape dim of elasticInfo must be 1D."),
            return false);
        OP_LOGD(nodeName, "elasticInfo dim0 = %ld", elasticInfoStorageShape->GetStorageShape().GetDim(0));
    }
    if (isPerformance) {
        const gert::StorageShape *performanceInfoStorageShape =
            context->GetOptionalInputShape(config.performanceInfoIndex);
        OP_TILING_CHECK(performanceInfoStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "performanceInfo"),
                        return false);
        OP_TILING_CHECK(performanceInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            nodeName, "performanceInfo",
                            std::to_string(performanceInfoStorageShape->GetStorageShape().GetDimNum()).c_str(),
                            "The shape dim of performanceInfo must be 1D."),
                        return false);
        OP_LOGD(nodeName, "performanceInfo dim0 = %ld", performanceInfoStorageShape->GetStorageShape().GetDim(0));
    }

    OP_TILING_CHECK(!CheckCommonOutputTensorDim(context, nodeName, quantMode),
                    OP_LOGE(nodeName, "Output param shape is invalid."), return false);

    return true;
}

static ge::graphStatus CheckQuantModeAndScales(const gert::TilingContext *context, const char *nodeName, bool isScales,
                                               const uint32_t quantMode, DispatchV2Config &config)
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

bool MoeDistributeDispatchV2TilingFuncBase::CheckCommomOutputTensorDataType(const gert::TilingContext *context,
                                                                            const char *nodeName)
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
    const gert::TilingContext *context, const char *nodeName, const bool isActiveMask, const bool hasElasticInfo,
    const bool isPerformance, DispatchV2Config &config)
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

// 校验输入输出数据格式
static bool CheckTensorFormat(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                              const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
                              const bool isPerformance, DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex); // nullptr前面已check过
    auto xDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat()));
    if (xDescFormat == ge::FORMAT_FRACTAL_NZ) {
        std::string fmtStr = Ops::Base::ToString(xDescFormat);
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "x", fmtStr.c_str(), "ND");
        return false;
    }
    auto expertIdDesc = context->GetInputDesc(config.expertIdsIndex);
    auto expertIdDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdDesc->GetStorageFormat()));
    if (expertIdDescFormat == ge::FORMAT_FRACTAL_NZ) {
        std::string fmtStr = Ops::Base::ToString(expertIdDescFormat);
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expert_ids", fmtStr.c_str(), "ND");
        return false;
    }
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        auto scalesDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(scalesDesc->GetStorageFormat()));
        if (scalesDescFormat == ge::FORMAT_FRACTAL_NZ) {
            std::string fmtStr = Ops::Base::ToString(scalesDescFormat);
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "scales", fmtStr.c_str(), "ND");
            return false;
        }
    }
    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        auto xActiveMaskDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat()));
        if (xActiveMaskDescFormat == ge::FORMAT_FRACTAL_NZ) {
            std::string fmtStr = Ops::Base::ToString(xActiveMaskDescFormat);
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "xActiveMask", fmtStr.c_str(), "ND");
            return false;
        }
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        auto elasticInfoDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(elasticInfoDesc->GetStorageFormat()));
        if (elasticInfoDescFormat == ge::FORMAT_FRACTAL_NZ) {
            std::string fmtStr = Ops::Base::ToString(elasticInfoDescFormat);
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "elastic_info", fmtStr.c_str(), "ND");
            return false;
        }
    }
    if (isPerformance) {
        auto performanceInfoDesc = context->GetOptionalInputDesc(config.performanceInfoIndex);
        auto performanceInfoDescFormat =
            static_cast<ge::Format>(ge::GetPrimaryFormat(performanceInfoDesc->GetStorageFormat()));
        if (performanceInfoDescFormat == ge::FORMAT_FRACTAL_NZ) {
            std::string fmtStr = Ops::Base::ToString(performanceInfoDescFormat);
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "performance_info", fmtStr.c_str(), "ND");
            return false;
        }
    }
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    auto expandXDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat()));
    if (expandXDescFormat == ge::FORMAT_FRACTAL_NZ) {
        std::string fmtStr = Ops::Base::ToString(expandXDescFormat);
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expandX", fmtStr.c_str(), "ND");
        return false;
    }
    if (quantMode >= static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        auto dynamicScalesDescFormat =
            static_cast<ge::Format>(ge::GetPrimaryFormat(dynamicScalesDesc->GetStorageFormat()));
        if (dynamicScalesDescFormat == ge::FORMAT_FRACTAL_NZ) {
            std::string fmtStr = Ops::Base::ToString(dynamicScalesDescFormat);
            OP_LOGE_FOR_INVALID_FORMAT(nodeName, "dynamicScales", fmtStr.c_str(), "ND");
            return false;
        }
    }
    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    auto assistInfoDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(assistInfoDesc->GetStorageFormat()));
    if (assistInfoDescFormat == ge::FORMAT_FRACTAL_NZ) {
        std::string fmtStr = Ops::Base::ToString(assistInfoDescFormat);
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "assistInfoForCombine", fmtStr.c_str(), "ND");
        return false;
    }
    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    auto expertTokenNumsDescFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertTokenNumsDesc->GetStorageFormat()));
    if (expertTokenNumsDescFormat == ge::FORMAT_FRACTAL_NZ) {
        std::string fmtStr = Ops::Base::ToString(expertTokenNumsDescFormat);
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "expertTokenNums", fmtStr.c_str(), "ND");
        return false;
    }
    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    auto epRecvCountsDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(epRecvCountsDesc->GetStorageFormat()));
    if (epRecvCountsDescFormat == ge::FORMAT_FRACTAL_NZ) {
        std::string fmtStr = Ops::Base::ToString(epRecvCountsDescFormat);
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "epRecvCounts", fmtStr.c_str(), "ND");
        return false;
    }
    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    auto tpRecvCountsDescFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(tpRecvCountsDesc->GetStorageFormat()));
    if (tpRecvCountsDescFormat == ge::FORMAT_FRACTAL_NZ) {
        std::string fmtStr = Ops::Base::ToString(tpRecvCountsDescFormat);
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "tpRecvCounts", fmtStr.c_str(), "ND");
        return false;
    }
    return true;
}

static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext *context, const char *nodeName,
                                           std::string &groupEp, DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    if (!config.isMc2Context) {
        auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrGroupEpIndex));
        OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                            (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                        OP_LOGE_WITH_INVALID_INPUT(nodeName, "groupEpPtr"), return ge::GRAPH_FAILED);
        groupEp = std::string(groupEpPtr);
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
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "copyExpertNumPtr"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "constExpertNumPtr"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupAttrParams(const gert::TilingContext *context, const char *nodeName,
                                            std::string &groupTp, bool &isLayered, DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrEpWorldSizeIndex);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrEpRankIdIndex);
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int64_t>(config.attrCommAlgIndex));
    int64_t epWorldSize = *epWorldSizePtr;
    isLayered = strcmp(commAlgPtr, "hierarchy") == 0; // isLayered赋值
    std::string socVersion = mc2tiling::GetSocVersion(context);
    int64_t maxEpworldsize = 0;
    if (socVersion == "Ascend950") {
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

ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::CheckOtherAttrParams(const gert::TilingContext *context,
                                                                            const char *nodeName, bool &isSetFullMeshV2,
                                                                            DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(config.attrExpertSharedTypeIndex);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(config.attrQuantModeIndex);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrExpertTokenNumsTypeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    if (config.isMc2Context) {
        if (strcmp(commAlgPtr, "hierarchy") == 0) {
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
    // A5 已作校验，这里只校验 A3
    if (CheckCommAlgPtr(commAlgPtr, nodeName) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    isSetFullMeshV2 = ((strcmp(commAlgPtr, "fullmesh_v2") == 0) ? true : false);
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 isSetFullMeshV2 = %d\n", isSetFullMeshV2);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::CheckCommAttrParams(const gert::TilingContext *context,
                                                                           const char *nodeName, std::string &groupTp,
                                                                           bool &isSetFullMeshV2, bool &isLayered,
                                                                           DispatchV2Config &config)
{
    // 校验 epWorldSize tpWorldSize epRankId tpRankId
    OP_TILING_CHECK(CheckGroupAttrParams(context, nodeName, groupTp, isLayered, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckGroupAttrParams is failed."), return ge::GRAPH_FAILED);
    // 校验 expertSharedType quantMode expertTokenNumsType commAlg
    OP_TILING_CHECK(CheckOtherAttrParams(context, nodeName, isSetFullMeshV2, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CheckOtherAttrParams is failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertAttrParams(const gert::TilingContext *context, const char *nodeName, bool isLayered,
                                             DispatchV2Config &config)
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
    int64_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;
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
        zeroComputeExpertNum + moeExpertNum > INT32_MAX,
        OP_LOGE(nodeName,
                "zeroExpertNum[%ld] + copyExpertNum[%ld] + constExpertNum[%ld] + moeExpertNum[%ld] exceed INT32_MAX.",
                zeroExpertNum, copyExpertNum, constExpertNum, moeExpertNum),
        return ge::GRAPH_FAILED);
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
    int64_t moeExpertMaxNum = isLayered ? MOE_EXPERT_MAX_NUM_LAYERED : MOE_EXPERT_MAX_NUM;
    OP_TILING_CHECK(
        (moeExpertNum <= 0) || (moeExpertNum > moeExpertMaxNum),
        OP_LOGE(nodeName, "moeExpertNum should be in (0, %ld], but got %ld.", moeExpertMaxNum, moeExpertNum),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAttrParams(const gert::TilingContext *context, const char *nodeName, DispatchV2Config &config,
                                     MoeDistributeDispatchV2TilingData &tilingData)
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
    const gert::TilingContext *context, const char *nodeName, MoeDistributeDispatchV2TilingData &tilingData,
    std::string &groupEp, std::string &groupTp, bool &isSetFullMeshV2, bool &isLayered, DispatchV2Config &config)
{
    OP_TILING_CHECK(CheckAttrPtrNullptr(context, nodeName, groupEp, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "params check nullptr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckCommAttrParams(context, nodeName, groupTp, isSetFullMeshV2, isLayered, config) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "params shape"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckExpertAttrParams(context, nodeName, isLayered, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "params dataType"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetAttrParams(context, nodeName, config, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "set attr params failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool CheckSharedAttrs(const char *nodeName, const MoeDistributeDispatchV2TilingData &tilingData)
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

static bool CheckCommAlgAttrs(const gert::TilingContext *context, const char *nodeName,
                              const MoeDistributeDispatchV2TilingData &tilingData, bool isSetFullMeshV2,
                              DispatchV2Config &config, bool isLayered)
{
    bool hasElasticInfo = tilingData.moeDistributeDispatchV2Info.hasElasticInfo;
    int32_t zeroComputeExpertNum = tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum;
    bool isExpertMask = tilingData.moeDistributeDispatchV2Info.isExpertMask;
    bool isPerformance = tilingData.moeDistributeDispatchV2Info.isPerformance;
    // 获取bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    uint32_t bs = static_cast<uint32_t>(xDim0);

    // 获取topk
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    uint32_t k = static_cast<uint32_t>(expertIdsDim1);
    int64_t fullMeshBsUpperBound =
        mc2tiling::GetSocVersion(context) == "Ascend950" ? FULLMESH_BS_UPPER_BOUND_A5 : FULLMESH_BS_UPPER_BOUND_A3;

    // 检查comm_alg和bs是否冲突
    OP_TILING_CHECK(isSetFullMeshV2 && (bs > fullMeshBsUpperBound),
                    OP_LOGE(nodeName, "When comm_alg is fullmesh_v2, bs should be between [1, %lld], but got %u.",
                            fullMeshBsUpperBound, bs),
                    return false);
    // 检查comm_alg和topK是否冲突
    OP_TILING_CHECK(isSetFullMeshV2 && (k > FULLMESH_K_MAX),
                    OP_LOGE(nodeName, "When comm_alg is fullmesh_v2, topK should be between [1, %lld], but got %u.",
                            FULLMESH_K_MAX, k),
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

static ge::graphStatus CheckAttrs(const gert::TilingContext *context, const char *nodeName,
                                  MoeDistributeDispatchV2TilingData &tilingData, uint32_t &localMoeExpertNum,
                                  bool isActiveMask, bool isSetFullMeshV2, bool isLayered, DispatchV2Config &config)
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
    OP_TILING_CHECK((localMoeExpertNum <= 0) || (localMoeExpertNum * epWorldSize > LOCAL_EXPERT_MAX_SIZE),
                    OP_LOGE(nodeName,
                            "localMoeExpertNum is invalid, "
                            "localMoeExpertNum * epWorldSize must be less than or equal to 2048, and localMoeExpertNum "
                            "must be greater than 0, "
                            "but got localMoeExpertNum * epWorldSize = %u, localMoeExpertNum = %u",
                            localMoeExpertNum * epWorldSize, localMoeExpertNum),
                    return ge::GRAPH_FAILED);

    // 校验输入x的dim 0并设bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t bsUpperBound = isLayered ? BS_UPPER_BOUND_LAYERED : BS_UPPER_BOUND;
    OP_TILING_CHECK((xDim0 > bsUpperBound) || (xDim0 <= 0),
                    OP_LOGE(nodeName,
                            "xDim0(BS) is invalid. Should be between "
                            "[1, %ld], but got xDim0=%ld.",
                            bsUpperBound, xDim0),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.bs = static_cast<uint32_t>(xDim0);

    // 校验globalBS
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
                            "globalBS is invalid, only "
                            "support 0 or maxBs(maxBs is the configured bs capacity per rank) * epWorldSize, but got "
                            "globalBS=%ld, "
                            "bs=%ld, epWorldSize=%u.",
                            *globalBsPtr, xDim0, epWorldSize),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((*globalBsPtr > (xDim0 * static_cast<int64_t>(epWorldSize))) && isActiveMask),
                    OP_LOGE(nodeName,
                            "Different bs on different rank cannot work when isActiveMask=true, globalBS=%ld, "
                            "bs=%ld, epWorldSize=%u.",
                            *globalBsPtr, xDim0, epWorldSize),
                    return ge::GRAPH_FAILED);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeDispatchV2Info.globalBs = static_cast<uint32_t>(xDim0) * epWorldSize;
    } else {
        tilingData.moeDistributeDispatchV2Info.globalBs = static_cast<uint32_t>(*globalBsPtr);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTwoDimScalesShape(const gert::TilingContext *context, const char *nodeName,
                                              const MoeDistributeDispatchV2TilingData &tilingData,
                                              const int64_t scalesDim0, const int64_t scalesDim1,
                                              DispatchV2Config &config)
{
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
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

static ge::graphStatus CheckXAndExpertIdsTensorShape(const gert::TilingContext *context, const char *nodeName,
                                                     MoeDistributeDispatchV2TilingData &tilingData,
                                                     const bool isLayered, const int64_t zeroExpertNum,
                                                     const int64_t copyExpertNum, const int64_t constExpertNum,
                                                     DispatchV2Config &config)
{
    // 校验输入x的维度1并设h, bs已校验过
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
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

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    if ((expandXDesc->GetDataType() == ge::DT_FLOAT4_E2M1) || (expandXDesc->GetDataType() == ge::DT_FLOAT4_E1M2)) {
        OP_TILING_CHECK(
            xDim1 % EVEN_ALIGN,
            OP_LOGE(nodeName,
                    "When expandx data type is FLOAT_E2M1/FLOAT_E1M2, the last axis should be even, please check."),
            return ge::GRAPH_FAILED);
    }

    // 校验expert_id的维度并设k
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
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
    if (isLayered) {
        const gert::StorageShape *expertScaleStorageShape = context->GetOptionalInputShape(EXPERT_SCALES_INDEX);
        const int64_t expertScalesDim0 = expertScaleStorageShape->GetStorageShape().GetDim(0);
        const int64_t expertScalesDim1 = expertScaleStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK((expertScalesDim0 != expertIdsDim0) || (expertScalesDim1 != expertIdsDim1),
                        OP_LOGE(nodeName,
                                "expertScaleShape's dim not equal to expertIdShape's dim, "
                                "expertScaleShape's dim0 is %ld, expertIdShape's dim0 is %ld, "
                                "expertScaleShape's dim1 is %ld, expertIdShape's dim1 is %ld.",
                                expertScalesDim0, expertIdsDim0, expertScalesDim1, expertIdsDim1),
                        return ge::GRAPH_FAILED);
    }
    tilingData.moeDistributeDispatchV2Info.k = static_cast<uint32_t>(expertIdsDim1);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillScalesInfo(const gert::TilingContext *context, const char *nodeName,
                                      MoeDistributeDispatchV2TilingData &tilingData, const bool isScales,
                                      const uint32_t quantMode, DispatchV2Config &config, uint64_t &scalesRow,
                                      uint64_t &scalesCol, uint32_t &scalesTypeSize, uint64_t &scalesCount)
{
    uint64_t h = tilingData.moeDistributeDispatchV2Info.h;
    uint32_t bs = tilingData.moeDistributeDispatchV2Info.bs;

    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
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
            auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
            OP_TILING_CHECK((quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)) &&
                                (expandXDesc->GetDataType() == ge::DT_INT8) &&
                                (scalesDim0 != static_cast<int64_t>(h)) &&
                                (scalesDim0 != static_cast<int64_t>(STATIC_SCALE_DIM_0)),
                            OP_LOGE(nodeName, "The expected scalesDim0 is %lu or %lu in static quant, but got %ld", h,
                                    STATIC_SCALE_DIM_0, scalesDim0),
                            return ge::GRAPH_FAILED);
            OP_TILING_CHECK(
                (quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)) &&
                    (expandXDesc->GetDataType() == ge::DT_HIFLOAT8) && (scalesDim0 != STATIC_SCALE_DIM_0),
                OP_LOGE(nodeName,
                        "The expected scalesDim0 is 1 when expandX datatype is hif8 in static quant, but got %ld",
                        scalesDim0),
                return ge::GRAPH_FAILED);
            scalesCol = ONE_DIM_SCALE_COL_NUM;
            scalesCount = static_cast<uint64_t>(scalesDim0);
        } else if (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
            const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
            OP_TILING_CHECK(
                scalesDim0 != bs,
                OP_LOGE(nodeName, "The expected scalesDim0 is %u when scales is not null in non-quant, but got %ld", bs,
                        scalesDim0),
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK(
                scalesDim1 > h,
                OP_LOGE(nodeName,
                        "The expected scalesDim1 is less than or equal to %lu when scales is not null in non-quant, "
                        "but got %ld",
                        h, scalesDim1),
                return ge::GRAPH_FAILED);
            scalesCol = static_cast<uint64_t>(scalesDim1);
            scalesCount = static_cast<uint64_t>(scalesDim0 * scalesDim1);
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

static ge::graphStatus CheckInputInfoTensorShape(const gert::TilingContext *context, const char *nodeName,
                                                 MoeDistributeDispatchV2TilingData &tilingData, uint32_t &A,
                                                 const int64_t localMoeExpertNum, const int64_t expertIdsDim1,
                                                 const bool hasElasticInfo, const bool isSharedExpert,
                                                 const bool isPerformance, DispatchV2Config &config)
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
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
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
        const gert::StorageShape *performanceInfoStorageShape =
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

static ge::graphStatus CheckExpandXAndDynamicScalesTensorShape(const gert::TilingContext *context, const char *nodeName,
                                                               MoeDistributeDispatchV2TilingData &tilingData,
                                                               const uint32_t quantMode, const uint32_t A,
                                                               const int64_t xDim1)
{
    // 校验expandX的维度
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
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
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        const int64_t dynamicScalesDim0 = dynamicScalesStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(dynamicScalesDim0 < static_cast<int64_t>(A),
                        OP_LOGE(nodeName,
                                "dynamicScales's dim0 should be equal to or greater than A, "
                                "dynamicScales's dim0=%ld, A=%u.",
                                dynamicScalesDim0, A),
                        return ge::GRAPH_FAILED);
        // Dim1, only for pergroup and mx
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
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAssistInfoAndExpertTokenNumsTensorShape(
    const gert::TilingContext *context, const char *nodeName, MoeDistributeDispatchV2TilingData &tilingData,
    const uint32_t A, const int64_t localMoeExpertNum, const bool hasElasticInfo, const bool isSharedExpert)
{
    // 校验assistInfo的维度
    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    const int64_t assistInfoDim0 = assistInfoStorageShape->GetStorageShape().GetDim(0);
    int64_t minAssistInfoDim0 = static_cast<int64_t>(A * ASSIST_NUM_PER_A);
    OP_TILING_CHECK(assistInfoDim0 < minAssistInfoDim0,
                    OP_LOGE(nodeName,
                            "assistInfoDim0 < minAssistInfoDim0,"
                            " assistInfoDim0 is %ld, minAssistInfoDim0 is %ld.",
                            assistInfoDim0, minAssistInfoDim0),
                    return ge::GRAPH_FAILED);

    // 校验expertTokenNums的维度
    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
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

static ge::graphStatus CheckRecvCountTensorShape(const gert::TilingContext *context, const char *nodeName,
                                                 MoeDistributeDispatchV2TilingData &tilingData,
                                                 const bool isSharedExpert, const int64_t localMoeExpertNum,
                                                 const bool hasElasticInfo, const bool isLayered,
                                                 const int64_t expertIdsDim1)
{
    // 校验epRecvCount和tpRecvCount的维度
    uint32_t globalBs = tilingData.moeDistributeDispatchV2Info.globalBs;
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    const int64_t epRecvCountDim0 = epRecvCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epRecvCount = ((isSharedExpert) ? epWorldSize : epWorldSize * localMoeExpertNum);
    if (hasElasticInfo) {
        epRecvCount = std::max(epWorldSize, epWorldSize * localMoeExpertNum);
    }

    if (isLayered) {
        // 如果是分层方案，则需要校验全新的shape，额外的globalBs * 2 * k * epWorldSize / 8
        // 用来存储token的cnt信息与offset信息，为了兼容A2&A5 这里取/8。
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

static ge::graphStatus CheckTensorShape(const gert::TilingContext *context, const char *nodeName,
                                        MoeDistributeDispatchV2TilingData &tilingData, const uint32_t quantMode,
                                        const bool isScales, const bool isSharedExpert, const bool hasElasticInfo,
                                        const bool isPerformance, const int64_t localMoeExpertNum, bool isLayered,
                                        DispatchV2Config &config)
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

    OP_TILING_CHECK(CheckAssistInfoAndExpertTokenNumsTensorShape(context, nodeName, tilingData, A, localMoeExpertNum,
                                                                 hasElasticInfo, isSharedExpert) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check assistInfo or expertTokenNums failed"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckRecvCountTensorShape(context, nodeName, tilingData, isSharedExpert, localMoeExpertNum,
                                              hasElasticInfo, isLayered, expertIdsDim1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check epRecvCount or tpRecvCount failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 校验输入输出数据格式
static bool CheckTensorPtrNullptr(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                                  const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
                                  const bool isPerformance, DispatchV2Config &config)
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

static ge::graphStatus CheckMc2Context(gert::TilingContext *context, const char *nodeName, DispatchV2Config &config)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
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
    gert::TilingContext *context, const char *nodeName, const bool isActiveMask, const bool isScales,
    const bool hasElasticInfo, const bool isPerformance, const uint32_t quantMode, const bool isLayered,
    DispatchV2Config &config)
{
    OP_TILING_CHECK(!CheckTensorPtrNullptr(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo,
                                           isPerformance, config),
                    OP_LOGE(nodeName, "params check nullptr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo, isPerformance,
                                    isLayered, config),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "params shape"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo,
                                         isPerformance, config),
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "params dataType"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !CheckTensorFormat(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo, isPerformance, config),
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "params format"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetHcommCfg(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData *tiling,
                                   const std::string groupEp, bool isLayered)
{
    const char *nodeName = context->GetNodeName();
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

static ge::graphStatus CheckAndCalWinSize(const gert::TilingContext *context,
                                          MoeDistributeDispatchV2TilingData &tilingData, const char *nodeName,
                                          const bool isSetFullMeshV2, uint32_t localMoeExpertNum, bool isLayered,
                                          DispatchV2Config &config)
{
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
    OP_TILING_CHECK(CheckWinSize(context, nodeName, winSizeData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get WinSize failed."), return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.totalWinSizeEp = winSizeData.totalWinSizeEp;
    return ge::GRAPH_SUCCESS;
}

static uint32_t Ceil(uint32_t dividend, uint32_t divisor, const char *nodeName)
{
    if (divisor != 0) {
        return ((dividend + divisor - 1) / divisor) * divisor;
    }
    OP_LOGE(nodeName, "ceil divisor should not be 0");
    return 0;
}

static uint32_t SendToMoeExpertUsedBuffer(const gert::TilingContext *context,
                                          const MoeDistributeDispatchV2TilingData &tilingData,
                                          uint32_t expertIdsBufSize, uint32_t kSize, const char *nodeName)
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
    UbForMoe = UbForMoe + expertMaskBufSize;                                        // expertMaskBuf
    UbForMoe = UbForMoe + Ceil(moeExpertNum * sizeof(int32_t), UB_ALIGN, nodeName); // tokenNumToExpertBuf
    return UbForMoe;
}

static uint32_t AllToAllBasicUsedBuffer(const gert::TilingContext *context,
                                        const MoeDistributeDispatchV2TilingData &tilingData, uint32_t quantMode,
                                        uint32_t &expertIdsBufSize, uint32_t hSize, uint32_t expertIdsCnt,
                                        uint32_t &hOutSizeAlign, const char *nodeName)
{
    uint32_t UbForBasic = 0;
    uint32_t epWorldSize = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    if (tilingData.moeDistributeDispatchV2Info.hasElasticInfo) {
        uint32_t elasticInfoSize = (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSize) * sizeof(int32_t);
        uint32_t elasticInfoSizeAlign = Ceil(elasticInfoSize, UB_ALIGN, nodeName);
        UbForBasic = UbForBasic + elasticInfoSizeAlign; // elasticInfoBuf_
    }

    uint32_t sizeofX = 2U;
    uint32_t hAlignSize = Ceil(hSize * sizeofX, UB_ALIGN, nodeName);
    UbForBasic = UbForBasic + hAlignSize * BUFFER_NUM; // inQueue

    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    bool isScales = (scalesStorageShape != nullptr);
    uint32_t hOutSize = (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) ? hSize * SIZE_OF_UNQUANT : hSize;
    hOutSizeAlign = Ceil(hOutSize, UB_ALIGN, nodeName);
    if ((quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) && isScales) {
        uint32_t scaleInBytes =
            tilingData.moeDistributeDispatchV2Info.scalesCol * tilingData.moeDistributeDispatchV2Info.scalesTypeSize;
        hOutSizeAlign += scaleInBytes;
    } else if (quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        hOutSizeAlign += sizeof(float);
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

static ge::graphStatus CheckUBSize(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData &tilingData,
                                   const char *nodeName)
{
    uint32_t ubSize = UB_ALIGN; // dataStateBuf

    uint32_t quantMode = tilingData.moeDistributeDispatchV2Info.quantMode;
    uint32_t hSize = tilingData.moeDistributeDispatchV2Info.h;
    uint32_t bsSize = tilingData.moeDistributeDispatchV2Info.bs;
    uint32_t kSize = tilingData.moeDistributeDispatchV2Info.k;
    uint32_t expertIdsCnt = bsSize * kSize;
    uint32_t expertIdsBufSize = 0;
    uint32_t hOutSizeAlign = 0;
    ubSize = ubSize + AllToAllBasicUsedBuffer(context, tilingData, quantMode, expertIdsBufSize, hSize, expertIdsCnt,
                                              hOutSizeAlign, nodeName);

    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    bool isActiveMask = (xActiveMaskStorageShape != nullptr);
    bool isScales = (scalesStorageShape != nullptr);
    bool needMaskCalFlag = (isActiveMask || tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum != 0);
    ubSize = ubSize + (needMaskCalFlag ? expertIdsBufSize : 0);

    uint32_t hFp32Size = Ceil(hSize * sizeof(float), UB_ALIGN, nodeName);
    if ((quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT)) && isScales) {
        hFp32Size = Ceil(hSize, PERGROUP_BLOCK_SIZE, nodeName) * sizeof(float);
    }
    uint32_t bsKAlign256 = Ceil(expertIdsCnt * SIZE_OF_HALF, SIZE_ALIGN_256, nodeName);
    uint32_t expertIdsSize = Ceil(expertIdsCnt * sizeof(int32_t), UB_ALIGN, nodeName);
    uint32_t xActivateMaskSize = bsSize * Ceil(kSize * sizeof(bool), UB_ALIGN, nodeName) * SIZE_OF_HALF;
    uint32_t maxSizeForUbBuffer = hFp32Size > expertIdsSize ? hFp32Size : expertIdsSize;
    maxSizeForUbBuffer = maxSizeForUbBuffer > xActivateMaskSize ? maxSizeForUbBuffer : xActivateMaskSize;
    maxSizeForUbBuffer = maxSizeForUbBuffer > bsKAlign256 ? maxSizeForUbBuffer : bsKAlign256;
    tilingData.moeDistributeDispatchV2Info.maxSizeForUbBuffer = maxSizeForUbBuffer;
    if (quantMode > static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        uint32_t hOutAlignUbSize = Ceil(hOutSizeAlign, UB_ALIGN, nodeName);
        ubSize = ubSize + hOutAlignUbSize;
        ubSize = ubSize + BUFFER_NUM * maxSizeForUbBuffer; // receiveDataCastFloatBuf smoothScalesBuf
    } else if (needMaskCalFlag) {
        ubSize = ubSize + BUFFER_NUM * maxSizeForUbBuffer;
    }

    if (tilingData.moeDistributeDispatchV2Info.isExpertMask ||
        (tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum != 0)) {
        uint32_t axisBSAlign = Ceil(bsSize * sizeof(int32_t), UB_ALIGN, nodeName);
        ubSize = ubSize + axisBSAlign; // validBsIndexTBuf_
        uint32_t validBufferSize = expertIdsSize > xActivateMaskSize ? expertIdsSize : xActivateMaskSize;
        ubSize = ubSize + validBufferSize; // validExpertIndexBuf_
    }

    ubSize = ubSize + SendToMoeExpertUsedBuffer(context, tilingData, expertIdsBufSize, kSize, nodeName);
    uint64_t ubRealSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubRealSize);
    OP_TILING_CHECK(ubSize > ubRealSize,
                    OP_LOGE(nodeName,
                            "Current scenario is exceeds the size limit, BS = %u, H = %u, topK = %u, moeExpertNum = "
                            "%u, aivNum available = %u, used UBsize = %u",
                            bsSize, hSize, kSize, tilingData.moeDistributeDispatchV2Info.moeExpertNum,
                            ascendcPlatform.GetCoreNumAiv(), ubSize),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workSpaces"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + static_cast<size_t>(WORKSPACE_ELEMENT_OFFSET * aivNum * aivNum);
    if (mc2tiling::GetSocVersion(context) == "Ascend950") {
        // A5为前缀和同步标记预留空间
        workSpaces[0] +=
            A5_CUMSUM_WS_FLAG_OFFSET + static_cast<uint64_t>(aivNum) * A5_CUMSUM_WORKSPACE_MAX_CORE_NUM * UB_ALIGN_VAL;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAndGetOptionalInput(gert::TilingContext *context, const char *nodeName,
                                         MoeDistributeDispatchV2TilingData *tilingData, DispatchV2Config &config,
                                         bool &isScales, bool &isActiveMask, bool &hasElasticInfo, bool &isPerformance)
{
    // 获取scales
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
    isScales = (scalesStorageShape != nullptr);

    // 获取xActiveMask
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
    isActiveMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.isTokenMask =
        ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    tilingData->moeDistributeDispatchV2Info.isExpertMask =
        ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));

    // 获取elasticInfo
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
    hasElasticInfo = (elasticInfoStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.hasElasticInfo = hasElasticInfo;

    // 获取performanceInfo
    const gert::StorageShape *performanceInfoStorageShape = context->GetOptionalInputShape(config.performanceInfoIndex);
    isPerformance = (performanceInfoStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.isPerformance = isPerformance;

    const gert::StorageShape *expertScalesStorageShape = context->GetOptionalInputShape(config.expertScalesIndex);
    tilingData->moeDistributeDispatchV2Info.hasExpertScales = (expertScalesStorageShape != nullptr);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertScales(const gert::TilingContext *context, const char *nodeName, const uint32_t bs,
                                         const uint32_t k, DispatchV2Config &config)
{
    const gert::StorageShape *expertScalesStorageShape = context->GetOptionalInputShape(config.expertScalesIndex);
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

static ge::graphStatus CheckExpandScales(const gert::TilingContext *context, const char *nodeName, const uint32_t a)
{
    const gert::StorageShape *expandScalesStorageShape = context->GetOutputShape(OUTPUT_EXPAND_SCALES_INDEX);
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

ge::graphStatus CheckInputParam(gert::TilingContext *context, const char *nodeName, const bool isActiveMask,
                                const bool isScales, const bool hasElasticInfo, const bool isPerformance,
                                const uint32_t quantMode, uint32_t &localMoeExpertNum, const bool isLayered,
                                const bool isSetFullMeshV2, MoeDistributeDispatchV2TilingData *tilingData,
                                DispatchV2Config &config)
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

ge::graphStatus CheckSizeAndSetRelevantCfg(gert::TilingContext *context, MoeDistributeDispatchV2TilingData *tilingData,
                                           const char *nodeName, const uint32_t localMoeExpertNum,
                                           const std::string groupEp, const bool isLayered, const bool isSetFullMeshV2,
                                           DispatchV2Config &config)
{
    // 校验UB大小
    if (isSetFullMeshV2) {
        OP_TILING_CHECK(CheckUBSize(context, *tilingData, nodeName) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "Tiling check UB size failed."), return ge::GRAPH_FAILED);
    }

    // 校验win区大小
    OP_TILING_CHECK(CheckAndCalWinSize(context, *tilingData, nodeName, isSetFullMeshV2, localMoeExpertNum, isLayered,
                                       config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check window size failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    if (!config.isMc2Context) {
        OP_TILING_CHECK(SetHcommCfg(context, tilingData, groupEp, isLayered) != ge::GRAPH_SUCCESS,
                        OP_LOGE(nodeName, "Tiling set hcomm cfg failed."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAndSetPlatformInfo(gert::TilingContext *context, MoeDistributeDispatchV2TilingData *tilingData,
                                        const bool isLayered, const char *nodeName)
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

ge::graphStatus MoeDistributeDispatchV2TilingFuncBase::MoeDistributeDispatchA3TilingFuncImplPublic(
    gert::TilingContext *context, DispatchV2Config &config)
{
    const char *nodeName = context->GetNodeName();
    MoeDistributeDispatchV2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchV2TilingData>();
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
                    OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

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
                    OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);

    // 检查Input参数
    OP_TILING_CHECK(
        CheckInputParam(context, nodeName, isActiveMask, isScales, hasElasticInfo, isPerformance, quantMode,
                        localMoeExpertNum, isLayered, isSetFullMeshV2, tilingData, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check input param failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckSizeAndSetRelevantCfg(context, tilingData, nodeName, localMoeExpertNum, groupEp, isLayered,
                                               isSetFullMeshV2, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check UB/window size or set workspace/hcomm cfg failed."),
                    return ge::GRAPH_FAILED);

    uint64_t tilingKey = CalTilingKey(isScales, quantMode, isSetFullMeshV2, isLayered);
    tilingData->moeDistributeDispatchV2Info.isMc2Context = config.isMc2Context;
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    OP_TILING_CHECK(CheckAndSetPlatformInfo(context, tilingData, isLayered, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set platformInfo Failed"), return ge::GRAPH_FAILED);

    // A5 CumSum 最小 tmp 大小
    if (tilingData->moeDistributeDispatchV2Info.expertTokenNumsType != 0) {
        tilingData->moeDistributeDispatchV2Info.cumsumTmpMinSize = 0;
    } else {
        uint32_t localExpertNumAlign = (localMoeExpertNum + 7U) / 8U * 8U;
        std::vector<int64_t> cumsumShapeDims = {1, static_cast<int64_t>(localExpertNumAlign)};
        auto cumsumSrcShape = ge::Shape(cumsumShapeDims);
        uint32_t cumsumMaxTmp = 0;
        uint32_t cumsumMinTmp = 0;
        GetCumSumMaxMinTmpSize(cumsumSrcShape, sizeof(float), true, false, cumsumMaxTmp, cumsumMinTmp);
        tilingData->moeDistributeDispatchV2Info.cumsumTmpMinSize = cumsumMinTmp;
        OP_LOGI(nodeName, "MTE cumsumTmpMinSize=%u. Kernel falls back to scalar prefix sum if UB is insufficient.",
                cumsumMinTmp);
    }

    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
// Register exception func
inline void MoeDistributeDispatchV2ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char *socName = aclrtGetSocName();
    if ((std::strstr(socName, "Ascend950") == nullptr) && (std::strstr(socName, "Ascend910B") == nullptr)) {
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
