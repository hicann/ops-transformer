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
 * \file block_attn_res_prepare_base_tiling.cpp
 * \brief Base tiling workflow for BlockAttnResPrepare.
 */

#include "block_attn_res_prepare_base_tiling.h"

#include <cmath>
#include <limits>
#include <sstream>
#include <string>

#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

constexpr const char* OP_NAME = "BlockAttnResPrepare";
constexpr size_t BLOCK_RES_INDEX = 0;
constexpr size_t VALID_BLOCKS_INDEX = 1;
constexpr size_t PSEUDO_QUERY_INDEX = 2;
constexpr size_t EPS_ATTR_INDEX = 0;
constexpr size_t BLOCK_RES_RANK = 3;
constexpr size_t PSEUDO_QUERY_RANK = 2;
constexpr size_t T_DIM_INDEX = 0;
constexpr size_t N_DIM_INDEX = 1;
constexpr size_t D_DIM_INDEX = 2;
constexpr size_t S_DIM_INDEX = 0;
constexpr size_t PSEUDO_QUERY_D_DIM_INDEX = 1;

constexpr uint64_t MAX_D_SIZE = 8192UL;
constexpr uint32_t MAX_BLOCK_NUM = 64U;

bool IsPositiveShape(const gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) <= 0) {
            return false;
        }
    }
    return true;
}

} // namespace

ge::graphStatus BlockAttnResPrepareBaseTiling::GetPlatformInfo()
{
    auto* platformInfo = context_->GetPlatformInfo();
    const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    aicCoreNum_ = platform.GetCoreNumAic();
    aivCoreNum_ = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0ASize_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0BSize_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize_);
    systemWorkspaceSize_ = platform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF(aivCoreNum_ == 0, OP_LOGE(context_->GetNodeName(), "AIV core number is zero"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        aicCoreNum_ > std::numeric_limits<uint16_t>::max() || aivCoreNum_ > std::numeric_limits<uint16_t>::max(),
        OP_LOGE(context_->GetNodeName(), "core count exceeds uint16 range: AIC=%lu, AIV=%lu", aicCoreNum_, aivCoreNum_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ == 0, OP_LOGE(context_->GetNodeName(), "UB size is zero"), return ge::GRAPH_FAILED);
    workspaceSize_ = systemWorkspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareBaseTiling::CheckInputShapes()
{
    const auto* blockResStorageShape = context_->GetInputShape(BLOCK_RES_INDEX);
    const auto* validBlocksStorageShape = context_->GetInputShape(VALID_BLOCKS_INDEX);
    const auto* pseudoQueryStorageShape = context_->GetInputShape(PSEUDO_QUERY_INDEX);
    OP_CHECK_IF(blockResStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_res", "nullptr",
                                                      "the input shape of block_res must not be nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(validBlocksStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "valid_blocks", "nullptr",
                                                      "the input shape of valid_blocks must not be nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(pseudoQueryStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "pseudo_query", "nullptr",
                                                      "the input shape of pseudo_query must not be nullptr"),
                return ge::GRAPH_FAILED);

    const gert::Shape& blockResShape = blockResStorageShape->GetOriginShape();
    const gert::Shape& validBlocksShape = validBlocksStorageShape->GetOriginShape();
    const gert::Shape& pseudoQueryShape = pseudoQueryStorageShape->GetOriginShape();
    OP_CHECK_IF(blockResShape.GetDimNum() != BLOCK_RES_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "block_res",
                                             (std::to_string(blockResShape.GetDimNum()) + "D").c_str(), "3D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(pseudoQueryShape.GetDimNum() != PSEUDO_QUERY_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "pseudo_query",
                                             (std::to_string(pseudoQueryShape.GetDimNum()) + "D").c_str(), "2D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(validBlocksShape.GetDimNum() != 1 || validBlocksShape.GetDim(0) != 1,
                OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "valid_blocks",
                                          Ops::Base::ToString(validBlocksShape).c_str(), "[1]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsPositiveShape(blockResShape) || !IsPositiveShape(pseudoQueryShape),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "block_res, pseudo_query",
                    (Ops::Base::ToString(blockResShape) + ", " + Ops::Base::ToString(pseudoQueryShape)).c_str(),
                    "all dimensions of block_res and pseudo_query must be positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        blockResShape.GetDim(D_DIM_INDEX) != pseudoQueryShape.GetDim(PSEUDO_QUERY_D_DIM_INDEX),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "block_res.shape[2], pseudo_query.shape[1]",
                                               (std::to_string(blockResShape.GetDim(D_DIM_INDEX)) + ", " +
                                                std::to_string(pseudoQueryShape.GetDim(PSEUDO_QUERY_D_DIM_INDEX)))
                                                   .c_str(),
                                               "block_res.shape[2] must equal pseudo_query.shape[1]"),
        return ge::GRAPH_FAILED);

    totalT_ = static_cast<uint64_t>(blockResShape.GetDim(T_DIM_INDEX));
    totalN_ = static_cast<uint64_t>(blockResShape.GetDim(N_DIM_INDEX));
    totalD_ = static_cast<uint64_t>(blockResShape.GetDim(D_DIM_INDEX));
    totalS_ = static_cast<uint64_t>(pseudoQueryShape.GetDim(S_DIM_INDEX));
    OP_CHECK_IF(totalT_ > std::numeric_limits<uint32_t>::max() || totalN_ > std::numeric_limits<uint32_t>::max() ||
                    totalS_ > std::numeric_limits<uint32_t>::max(),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "block_res.shape[0], block_res.shape[1], pseudo_query.shape[0]",
                    (std::to_string(totalT_) + ", " + std::to_string(totalN_) + ", " + std::to_string(totalS_)).c_str(),
                    "the dimensions T, N and S must be within the uint32 range"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalN_ > MAX_BLOCK_NUM,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_res.shape[1]",
                                                      std::to_string(totalN_).c_str(),
                                                      "block_res.shape[1] must be less than or equal to 64"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalD_ > MAX_D_SIZE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_res.shape[2]",
                                                      std::to_string(totalD_).c_str(),
                                                      "block_res.shape[2] must be less than or equal to 8192"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalT_ > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / totalS_,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "block_res.shape[0], pseudo_query.shape[0]",
                    (std::to_string(totalT_) + ", " + std::to_string(totalS_)).c_str(),
                    "the product of block_res.shape[0] and pseudo_query.shape[0] must be within the int64 range"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareBaseTiling::CheckInputDtypes()
{
    const auto* blockResDesc = context_->GetInputDesc(BLOCK_RES_INDEX);
    const auto* validBlocksDesc = context_->GetInputDesc(VALID_BLOCKS_INDEX);
    const auto* pseudoQueryDesc = context_->GetInputDesc(PSEUDO_QUERY_INDEX);
    OP_CHECK_IF(blockResDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_res", "nullptr",
                                                      "the input descriptor of block_res must not be nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(validBlocksDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "valid_blocks", "nullptr",
                                                      "the input descriptor of valid_blocks must not be nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(pseudoQueryDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "pseudo_query", "nullptr",
                                                      "the input descriptor of pseudo_query must not be nullptr"),
                return ge::GRAPH_FAILED);
    blockResDtype_ = blockResDesc->GetDataType();
    OP_CHECK_IF(blockResDtype_ != ge::DT_FLOAT && blockResDtype_ != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "block_res",
                                                      ge::TypeUtils::DataTypeToSerialString(blockResDtype_).c_str(),
                                                      "supported dtypes are FLOAT and BFLOAT16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(pseudoQueryDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "pseudo_query",
                                          ge::TypeUtils::DataTypeToSerialString(pseudoQueryDesc->GetDataType()).c_str(),
                                          "FLOAT"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(validBlocksDesc->GetDataType() != ge::DT_UINT64,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "valid_blocks",
                                          ge::TypeUtils::DataTypeToSerialString(validBlocksDesc->GetDataType()).c_str(),
                                          "UINT64"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareBaseTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(OP_NAME, "tiling context is null"), return ge::GRAPH_FAILED);
    if (CheckInputShapes() != ge::GRAPH_SUCCESS || CheckInputDtypes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const auto* attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    const float* eps = attrs->GetFloat(EPS_ATTR_INDEX);
    eps_ = eps == nullptr ? BLOCK_ATTN_RES_PREPARE_DEFAULT_EPS : *eps;
    if (!std::isfinite(eps_) || eps_ < std::numeric_limits<float>::min()) {
        std::ostringstream epsValue;
        epsValue << eps_;
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "eps", epsValue.str().c_str(),
                                              "eps must be finite and in the float32 normal range "
                                              "[1.1754943508222875e-38, 3.4028234663852886e+38]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareBaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareBaseTiling::GetWorkspaceSize()
{
    size_t* workspaceSizes = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaceSizes == nullptr, OP_LOGE(context_->GetNodeName(), "workspace size address is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(workspaceSize_ > std::numeric_limits<size_t>::max(),
                OP_LOGE(context_->GetNodeName(), "workspaceSize=%lu exceeds size_t maximum=%lu", workspaceSize_,
                        static_cast<uint64_t>(std::numeric_limits<size_t>::max())),
                return ge::GRAPH_FAILED);
    workspaceSizes[0] = static_cast<size_t>(workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareBaseTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);

    OP_CHECK_IF(tilingDataPtr_ == nullptr || tilingDataSize_ == 0U,
                OP_LOGE(context_->GetNodeName(), "tiling data is empty"), return ge::GRAPH_FAILED);
    auto* rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr, OP_LOGE(context_->GetNodeName(), "raw tiling data is null"),
                return ge::GRAPH_FAILED);
    auto* rawData = rawTilingData->GetData();
    OP_CHECK_IF(rawData == nullptr, OP_LOGE(context_->GetNodeName(), "raw tiling data buffer is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawTilingData->GetCapacity() < tilingDataSize_,
                OP_LOGE(context_->GetNodeName(), "raw tiling data capacity is insufficient: got %zu, required %zu",
                        rawTilingData->GetCapacity(), tilingDataSize_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(memcpy_s(rawData, rawTilingData->GetCapacity(), tilingDataPtr_, tilingDataSize_) != EOK,
                OP_LOGE(context_->GetNodeName(), "copy tiling data failed"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(tilingDataSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingBlockAttnResPrepare(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "tiling context is null"), return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareBlockAttnResPrepare(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "tiling parse context is null"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BlockAttnResPrepare)
    .Tiling(TilingBlockAttnResPrepare)
    .TilingParse<BlockAttnResPrepareCompileInfo>(TilingPrepareBlockAttnResPrepare);

} // namespace optiling
