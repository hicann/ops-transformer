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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include "log/log.h"
#include "securec.h"

namespace optiling {
namespace {

constexpr const char *OP_NAME = "BlockAttnResPrepare";
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

constexpr uint64_t UB_RESERVED_BYTES = 8UL * 1024UL;
constexpr uint32_t D_ALIGN_ELEMS = 64U;
constexpr uint64_t MAX_HEAD_DIM = 8192UL;
constexpr uint32_t MAX_BLOCK_NUM = 64U;
constexpr uint64_t MAX_ALIGNED_D = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - (D_ALIGN_ELEMS - 1U);

bool IsPositiveShape(const gert::Shape &shape)
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
    auto *platformInfo = context_->GetPlatformInfo();
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
    OP_CHECK_IF(ubSize_ <= UB_RESERVED_BYTES, OP_LOGE(context_->GetNodeName(), "UB size is too small: %lu", ubSize_),
                return ge::GRAPH_FAILED);
    workspaceSize_ = systemWorkspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareBaseTiling::CheckInputShapes()
{
    const auto *blockResStorageShape = context_->GetInputShape(BLOCK_RES_INDEX);
    const auto *validStorageShape = context_->GetInputShape(VALID_BLOCKS_INDEX);
    const auto *pseudoQueryStorageShape = context_->GetInputShape(PSEUDO_QUERY_INDEX);
    OP_CHECK_IF(blockResStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_res", "nullptr",
                                                      "the input shape of block_res must not be nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(validStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "valid_blocks", "nullptr",
                                                      "the input shape of valid_blocks must not be nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(pseudoQueryStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "pseudo_query", "nullptr",
                                                      "the input shape of pseudo_query must not be nullptr"),
                return ge::GRAPH_FAILED);

    const gert::Shape &blockResShape = blockResStorageShape->GetOriginShape();
    const gert::Shape &validShape = validStorageShape->GetOriginShape();
    const gert::Shape &pseudoQueryShape = pseudoQueryStorageShape->GetOriginShape();
    OP_CHECK_IF(blockResShape.GetDimNum() != BLOCK_RES_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "block_res",
                                             (std::to_string(blockResShape.GetDimNum()) + "D").c_str(), "3D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(pseudoQueryShape.GetDimNum() != PSEUDO_QUERY_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "pseudo_query",
                                             (std::to_string(pseudoQueryShape.GetDimNum()) + "D").c_str(), "2D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(validShape.GetDimNum() != 1 || validShape.GetDim(0) != 1,
                OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "valid_blocks",
                                          Ops::Base::ToString(validShape).c_str(), "1"),
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
    OP_CHECK_IF(totalD_ > MAX_HEAD_DIM,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_res.shape[2]",
                                                      std::to_string(totalD_).c_str(),
                                                      "block_res.shape[2] must be less than or equal to 8192"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalD_ > MAX_ALIGNED_D,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "block_res.shape[2]", std::to_string(totalD_).c_str(),
                    "block_res.shape[2] must support uint32-aligned tiling without overflow"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalT_ > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / totalS_,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "block_res.shape[0], pseudo_query.shape[0]",
                    (std::to_string(totalT_) + ", " + std::to_string(totalS_)).c_str(),
                    "the product of block_res.shape[0] and pseudo_query.shape[0] must be within the int64 range"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareBaseTiling::CheckInputDtypes() const
{
    const auto *blockResDesc = context_->GetInputDesc(BLOCK_RES_INDEX);
    const auto *validDesc = context_->GetInputDesc(VALID_BLOCKS_INDEX);
    const auto *pseudoQueryDesc = context_->GetInputDesc(PSEUDO_QUERY_INDEX);
    OP_CHECK_IF(blockResDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_res", "nullptr",
                                                      "the input descriptor of block_res must not be nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(validDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "valid_blocks", "nullptr",
                                                      "the input descriptor of valid_blocks must not be nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(pseudoQueryDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "pseudo_query", "nullptr",
                                                      "the input descriptor of pseudo_query must not be nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockResDesc->GetDataType() != ge::DT_FLOAT || pseudoQueryDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "block_res, pseudo_query",
                    (ge::TypeUtils::DataTypeToSerialString(blockResDesc->GetDataType()) + ", " +
                     ge::TypeUtils::DataTypeToSerialString(pseudoQueryDesc->GetDataType()))
                        .c_str(),
                    "the dtypes of block_res and pseudo_query must both be FLOAT"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        validDesc->GetDataType() != ge::DT_UINT64,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "valid_blocks",
                                  ge::TypeUtils::DataTypeToSerialString(validDesc->GetDataType()).c_str(), "UINT64"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareBaseTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(OP_NAME, "tiling context is null"), return ge::GRAPH_FAILED);
    if (CheckInputShapes() != ge::GRAPH_SUCCESS || CheckInputDtypes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const auto *attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    const float *eps = attrs->GetFloat(EPS_ATTR_INDEX);
    eps_ = eps == nullptr ? BLOCK_ATTN_RES_PREPARE_DEFAULT_EPS : *eps;
    OP_CHECK_IF(!std::isfinite(eps_) || eps_ <= 0.0F,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "eps", std::to_string(eps_).c_str(),
                                                      "eps must be finite and be greater than zero"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareBaseTiling::CalculateWorkDistribution(uint64_t totalWorkUnits, uint64_t maxCoreNum,
                                                                         const char *templateName,
                                                                         WorkDistribution &distribution) const
{
    OP_CHECK_IF(totalWorkUnits > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context_->GetNodeName(), "%s totalWorkUnits=%lu exceeds uint32 maximum=%u", templateName,
                        totalWorkUnits, std::numeric_limits<uint32_t>::max()),
                return ge::GRAPH_FAILED);

    const uint64_t usedCoreNum = std::min(maxCoreNum, totalWorkUnits);
    OP_CHECK_IF(usedCoreNum == 0U,
                OP_LOGE(context_->GetNodeName(), "%s usedCoreNum is zero: totalWorkUnits=%lu, availableCoreNum=%lu",
                        templateName, totalWorkUnits, maxCoreNum),
                return ge::GRAPH_FAILED);
    const uint64_t workUnitsPerCore = totalWorkUnits / usedCoreNum;
    OP_CHECK_IF(workUnitsPerCore > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context_->GetNodeName(), "%s workUnitsPerCore=%lu exceeds uint32 maximum=%u", templateName,
                        workUnitsPerCore, std::numeric_limits<uint32_t>::max()),
                return ge::GRAPH_FAILED);

    distribution.usedCoreNum = static_cast<uint32_t>(usedCoreNum);
    distribution.blockFactor = static_cast<uint32_t>(workUnitsPerCore);
    distribution.bigCoreNum = static_cast<uint32_t>(totalWorkUnits % usedCoreNum);
    distribution.tailBlockFactor = distribution.blockFactor + ((distribution.bigCoreNum > 0U) ? 1U : 0U);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareBaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareBaseTiling::GetWorkspaceSize()
{
    size_t *workspaceSizes = context_->GetWorkspaceSizes(1);
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
    context_->SetTilingKey(GetTilingKey());

    const TilingDataView tilingDataView = GetTilingDataView();
    const char *templateName = tilingDataView.templateName == nullptr ? "unknown" : tilingDataView.templateName;
    OP_CHECK_IF(tilingDataView.data == nullptr || tilingDataView.size == 0U,
                OP_LOGE(context_->GetNodeName(), "%s tiling data is empty", templateName), return ge::GRAPH_FAILED);
    auto *rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr, OP_LOGE(context_->GetNodeName(), "raw tiling data is null"),
                return ge::GRAPH_FAILED);
    auto *rawData = rawTilingData->GetData();
    OP_CHECK_IF(rawData == nullptr, OP_LOGE(context_->GetNodeName(), "raw tiling data buffer is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawTilingData->GetCapacity() < tilingDataView.size,
                OP_LOGE(context_->GetNodeName(), "raw %s tiling data capacity is insufficient", templateName),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(memcpy_s(rawData, rawTilingData->GetCapacity(), tilingDataView.data, tilingDataView.size) != EOK,
                OP_LOGE(context_->GetNodeName(), "copy %s tiling data failed", templateName), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(tilingDataView.size);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingBlockAttnResPrepare(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "tiling context is null"), return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareBlockAttnResPrepare(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "tiling parse context is null"), return ge::GRAPH_FAILED);
    auto *compileInfo = context->GetCompiledInfo<BlockAttnResPrepareCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context->GetNodeName(), "compiled info is null"),
                return ge::GRAPH_FAILED);
    auto *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "platform info is null"),
                return ge::GRAPH_FAILED);
    const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aicCoreNum = platform.GetCoreNumAic();
    compileInfo->aivCoreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo->l1Size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo->l0ASize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo->l0BSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo->l0CSize);
    compileInfo->systemWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF(compileInfo->aicCoreNum == 0 || compileInfo->aivCoreNum == 0 || compileInfo->ubSize == 0 ||
                    compileInfo->l1Size == 0 || compileInfo->l0ASize == 0 || compileInfo->l0BSize == 0 ||
                    compileInfo->l0CSize == 0,
                OP_LOGE(context->GetNodeName(), "failed to query Ascend 950 core memory information"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BlockAttnResPrepare)
    .Tiling(TilingBlockAttnResPrepare)
    .TilingParse<BlockAttnResPrepareCompileInfo>(TilingPrepareBlockAttnResPrepare);

} // namespace optiling
