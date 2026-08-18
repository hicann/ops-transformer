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
 * \file grouped_matmul_swiglu_quant_v2_basic_api_tiling.cpp
 * \brief
 */

#include "grouped_matmul_swiglu_quant_v2_basic_api_tiling.h"

#include <algorithm>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h"

namespace optiling {
namespace {
using namespace GroupedMatmulSwigluQuantV2Tiling;

constexpr size_t SHAPE_DIM_ONE = 1;
constexpr size_t SHAPE_DIM_TWO = 2;
constexpr size_t SHAPE_DIM_THREE = 3;
constexpr size_t SHAPE_DIM_FOUR = 4;
constexpr uint64_t TENSOR_API_FP8_N_ALIGN = 128UL;
constexpr uint64_t TENSOR_API_TRANS_B_M_PER_GROUP_LOWER_LIMIT = 128UL;
constexpr uint64_t TENSOR_API_NOT_TRANS_B_M_PER_GROUP_LOWER_LIMIT = 512UL;
constexpr uint64_t TENSOR_API_B_NOTRANS_M_LOWER_LIMIT = 512UL;
constexpr uint64_t TENSOR_API_LARGE_M_BASE_M = GmmConstant::BASIC_BLOCK_SIZE_256;
constexpr uint64_t TENSOR_API_LARGE_N_BASE_N = GmmConstant::BASIC_BLOCK_SIZE_128;
constexpr uint64_t MX_SCALE_K_ALIGN = 64UL;
constexpr int64_t MX_QUANT_MODE = 2L;
constexpr uint64_t AIC_AIV_CORE_RATIO = 2UL;

} // namespace

void GroupedMatmulSwigluQuantV2BasicApiTiling950::Reset()
{
    tilingData_ = {};
    basicTiling_ = GQmmBasicTiling();
    xScaleDtype_ = ge::DT_UNDEFINED;
    quantDtype_ = ge::DT_UNDEFINED;
    xScaleFormat_ = static_cast<ge::Format>(-1);
    weightScaleFormat_ = static_cast<ge::Format>(-1);
    yScaleFormat_ = static_cast<ge::Format>(-1);
    aivNum_ = 0U;
    platformMemoryReady_ = false;
    return;
}

ge::graphStatus GroupedMatmulSwigluQuantV2BasicApiTiling950::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        const auto *compileInfo = context_->GetCompileInfo<GMMSwigluV2CompileInfo>();
        OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "Compile info is nullptr."),
                    return ge::GRAPH_FAILED);
        aicoreParams_.aicNum = compileInfo->aicNum_;
        aivNum_ = compileInfo->aivNum_;
        aicoreParams_.ubSize = compileInfo->ubSize_;
        OP_LOGD(context_->GetNodeName(),
                "Platform memory information is unavailable; Tensor API tiling will fall back.");
        return ge::GRAPH_SUCCESS;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.aicNum = ascendcPlatform.GetCoreNumAic();
        aivNum_ = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, aicoreParams_.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, aicoreParams_.l0bSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
        platformMemoryReady_ = aicoreParams_.l1Size > 0 && aicoreParams_.l0cSize > 0;
    }
    OP_LOGD(context_->GetNodeName(),
            "Tensor API platform info: aicNum=%lu, aivNum=%u, ub=%lu, l1=%lu, l0a=%lu, l0b=%lu, l0c=%lu.",
            aicoreParams_.aicNum, aivNum_, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0aSize,
            aicoreParams_.l0bSize, aicoreParams_.l0cSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2BasicApiTiling950::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    inputParams_.opName = context_->GetNodeName();
    inputParams_.opType = GetOpType();
    OP_CHECK_IF(!AnalyzeDtype() || !AnalyzeAttrs() || !AnalyzeInputs(),
                OP_LOGE(inputParams_.opName, "Failed to analyze Tensor API tiling inputs."), return ge::GRAPH_FAILED);
    inputParams_.initFlag = true;
    return ge::GRAPH_SUCCESS;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling950::AnalyzeDtype()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    auto xScaleDesc = context_->GetInputDesc(X_SCALE_INDEX);
    auto weightDesc = context_->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    auto weightScaleDesc = context_->GetDynamicInputDesc(WEIGHT_SCALE_INDEX, 0);
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    auto yScaleDesc = context_->GetOutputDesc(Y_SCALE_INDEX);
    OP_CHECK_IF(xDesc == nullptr || xScaleDesc == nullptr || weightDesc == nullptr || weightScaleDesc == nullptr ||
                    yDesc == nullptr || yScaleDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "Tensor API input or output desc is nullptr."), return false);

    inputParams_.aDtype = xDesc->GetDataType();
    inputParams_.bDtype = weightDesc->GetDataType();
    xScaleDtype_ = xScaleDesc->GetDataType();
    inputParams_.scaleDtype = weightScaleDesc->GetDataType();
    inputParams_.outDataDtype = yDesc->GetDataType();
    inputParams_.outScaleDtype = yScaleDesc->GetDataType();
    inputParams_.cDtype = inputParams_.outDataDtype;
    inputParams_.aFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetFormat().GetStorageFormat()));
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(weightDesc->GetFormat().GetStorageFormat()));
    xScaleFormat_ = static_cast<ge::Format>(ge::GetPrimaryFormat(xScaleDesc->GetFormat().GetStorageFormat()));
    weightScaleFormat_ = static_cast<ge::Format>(ge::GetPrimaryFormat(weightScaleDesc->GetFormat().GetStorageFormat()));
    inputParams_.cFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetFormat().GetStorageFormat()));
    yScaleFormat_ = static_cast<ge::Format>(ge::GetPrimaryFormat(yScaleDesc->GetFormat().GetStorageFormat()));

    auto biasShape = context_->GetOptionalInputShape(BIAS_INDEX);
    inputParams_.hasBias = biasShape != nullptr && biasShape->GetStorageShape().GetShapeSize() != 0;
    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    OP_CHECK_IF(inputParams_.hasBias && biasDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "Bias shape is present but bias desc is nullptr."), return false);
    if (biasDesc != nullptr) {
        inputParams_.biasDtype = biasDesc->GetDataType();
    }
    return true;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling950::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "Attrs is nullptr."), return false);

    const int64_t *dequantMode = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_MODE);
    const int64_t *dequantDtype = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_DTYPE);
    const int64_t *quantMode = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE);
    const int64_t *quantDtype = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_DTYPE);
    OP_CHECK_IF(dequantMode == nullptr || dequantDtype == nullptr || quantMode == nullptr || quantDtype == nullptr,
                OP_LOGE(context_->GetNodeName(), "Required quantization attrs are nullptr."), return false);
    OP_CHECK_IF(*dequantMode != MX_QUANT_MODE || *quantMode != MX_QUANT_MODE,
                OP_LOGE(context_->GetNodeName(), "dequant_mode and quant_mode must both be 2 for MX quant."),
                return false);
    OP_CHECK_IF(static_cast<ge::DataType>(*dequantDtype) != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "dequant_dtype must be DT_FLOAT for MX quant."), return false);
    quantDtype_ = static_cast<ge::DataType>(*quantDtype);
    const bool *transposeWeight = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_WEIGHT);
    const int64_t *groupListType = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPLIST_TYPE);
    inputParams_.transA = false;
    inputParams_.transB = transposeWeight != nullptr ? *transposeWeight : false;
    inputParams_.groupListType = groupListType != nullptr ? static_cast<int8_t>(*groupListType) : 0;
    OP_CHECK_IF(inputParams_.groupListType != 0 && inputParams_.groupListType != 1,
                OP_LOGE(context_->GetNodeName(), "group_list_type must be 0 or 1."), return false);
    inputParams_.groupType = static_cast<int8_t>(GroupedMatmul::SPLIT_M);
    inputParams_.aQuantMode = QuantMode::MX_PERGROUP_MODE;
    inputParams_.bQuantMode = QuantMode::MX_PERGROUP_MODE;
    return true;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling950::AnalyzeInputs()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    auto weightStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    auto groupListStorageShape = context_->GetOptionalInputShape(GROUPLIST_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr || weightStorageShape == nullptr || groupListStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "x, weight or group_list shape is nullptr."), return false);

    const auto &xShape = xStorageShape->GetOriginShape();
    const auto &weightShape = weightStorageShape->GetOriginShape();
    const auto &groupListShape = groupListStorageShape->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() < SHAPE_DIM_TWO || weightShape.GetDimNum() < SHAPE_DIM_TWO,
                OP_LOGE(context_->GetNodeName(), "x and weight must have at least two dimensions."), return false);
    OP_CHECK_IF(groupListShape.GetDimNum() != SHAPE_DIM_ONE,
                OP_LOGE(context_->GetNodeName(), "group_list must be one-dimensional."), return false);

    const size_t xDimNum = xShape.GetDimNum();
    const size_t weightDimNum = weightShape.GetDimNum();
    const int64_t m = xShape.GetDim(xDimNum - 2);
    const int64_t k = xShape.GetDim(xDimNum - 1);
    const int64_t weightK =
        inputParams_.transB ? weightShape.GetDim(weightDimNum - 1) : weightShape.GetDim(weightDimNum - 2);
    const int64_t n = inputParams_.transB ? weightShape.GetDim(weightDimNum - 2) : weightShape.GetDim(weightDimNum - 1);
    const int64_t groupNum = groupListShape.GetDim(0);
    OP_CHECK_IF(
        m <= 0 || n <= 0 || k <= 0 || weightK != k || groupNum <= 0 ||
            groupNum > static_cast<int64_t>(GmmConstant::GMM_MAX_GROUP_LIST_SIZE),
        OP_LOGE(context_->GetNodeName(), "Invalid Tensor API shape: M=%ld, N=%ld, K=%ld, weightK=%ld, groupNum=%ld.", m,
                n, k, weightK, groupNum),
        return false);

    inputParams_.mSize = static_cast<uint64_t>(m);
    inputParams_.nSize = static_cast<uint64_t>(n);
    inputParams_.kSize = static_cast<uint64_t>(k);
    inputParams_.groupNum = static_cast<uint64_t>(groupNum);
    inputParams_.isSingleX = true;
    inputParams_.isSingleW = GetDynamicInputCount(WEIGHT_INDEX) == 1U;
    inputParams_.isSingleY = true;
    return true;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling950::IsFp8(ge::DataType dtype) const
{
    return dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling950::IsSupportedFormat(ge::Format format) const
{
    return format == ge::FORMAT_ND || format == ge::FORMAT_NCL || format == ge::FORMAT_NCHW;
}

size_t GroupedMatmulSwigluQuantV2BasicApiTiling950::GetDynamicInputCount(uint32_t inputIndex) const
{
    size_t count = 0U;
    while (count < GmmConstant::GMM_MAX_GROUP_LIST_SIZE &&
           context_->GetDynamicInputShape(inputIndex, count) != nullptr) {
        ++count;
    }
    return count;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling950::CheckTensorApiShapes() const
{
    auto xShapePtr = context_->GetInputShape(X_INDEX);
    auto weightShapePtr = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    if (xShapePtr == nullptr || weightShapePtr == nullptr || GetDynamicInputCount(WEIGHT_INDEX) != 1U ||
        GetDynamicInputCount(WEIGHT_SCALE_INDEX) != 1U) {
        return false;
    }
    const auto &xShape = xShapePtr->GetOriginShape();
    const auto &weightShape = weightShapePtr->GetOriginShape();
    if (xShape.GetDimNum() != SHAPE_DIM_TWO ||
        (weightShape.GetDimNum() != SHAPE_DIM_TWO && weightShape.GetDimNum() != SHAPE_DIM_THREE)) {
        return false;
    }
    return (weightShape.GetDimNum() == SHAPE_DIM_TWO && inputParams_.groupNum == 1U) ||
           (weightShape.GetDimNum() == SHAPE_DIM_THREE &&
            weightShape.GetDim(0) == static_cast<int64_t>(inputParams_.groupNum));
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling950::CheckTensorApiScaleShapes() const
{
    auto xScaleShapePtr = context_->GetInputShape(X_SCALE_INDEX);
    auto weightScaleShapePtr = context_->GetDynamicInputShape(WEIGHT_SCALE_INDEX, 0);
    if (xScaleShapePtr == nullptr || weightScaleShapePtr == nullptr) {
        return false;
    }
    const auto &xScaleShape = xScaleShapePtr->GetOriginShape();
    const auto &weightScaleShape = weightScaleShapePtr->GetOriginShape();
    if (xScaleShape.GetDimNum() != SHAPE_DIM_THREE || weightScaleShape.GetDimNum() != SHAPE_DIM_FOUR) {
        return false;
    }

    const int64_t kBlocks = static_cast<int64_t>(GroupedMatmul::CeilDiv(inputParams_.kSize, MX_SCALE_K_ALIGN));
    if (xScaleShape.GetDim(0) != static_cast<int64_t>(inputParams_.mSize) || xScaleShape.GetDim(1) != kBlocks ||
        xScaleShape.GetDim(2) != SHAPE_DIM_TWO) {
        return false;
    }
    const int64_t weightScaleN = inputParams_.transB ? weightScaleShape.GetDim(1) : weightScaleShape.GetDim(2);
    const int64_t weightScaleK = inputParams_.transB ? weightScaleShape.GetDim(2) : weightScaleShape.GetDim(1);
    return weightScaleShape.GetDim(0) == static_cast<int64_t>(inputParams_.groupNum) &&
           weightScaleN == static_cast<int64_t>(inputParams_.nSize) && weightScaleK == kBlocks &&
           weightScaleShape.GetDim(3) == SHAPE_DIM_TWO;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling950::IsCapable()
{
    if (!platformMemoryReady_) {
        return false;
    }
    // Keep Tensor API-specific dtype restrictions in capability checking so unsupported cases can fall back.
    const bool quantDtypeSupported = IsFp8(quantDtype_) && quantDtype_ == inputParams_.outDataDtype;
    const bool dtypeSupported = IsFp8(inputParams_.aDtype) && IsFp8(inputParams_.bDtype) &&
                                xScaleDtype_ == ge::DT_FLOAT8_E8M0 && inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 &&
                                IsFp8(inputParams_.outDataDtype) && inputParams_.outScaleDtype == ge::DT_FLOAT8_E8M0;
    const bool formatSupported = IsSupportedFormat(inputParams_.aFormat) && IsSupportedFormat(inputParams_.bFormat) &&
                                 IsSupportedFormat(xScaleFormat_) && IsSupportedFormat(weightScaleFormat_) &&
                                 IsSupportedFormat(inputParams_.cFormat) && IsSupportedFormat(yScaleFormat_);
    const bool coreSupported = aicoreParams_.aicNum > 0 && aivNum_ == AIC_AIV_CORE_RATIO * aicoreParams_.aicNum;
    const bool checkTensorApiShapes = CheckTensorApiShapes();
    const bool checkTensorApiScaleShapes = CheckTensorApiScaleShapes();
    const uint64_t averageMPerGroup = inputParams_.mSize / inputParams_.groupNum;
    const uint64_t mPerGroupLowerLimit = inputParams_.transB ? TENSOR_API_TRANS_B_M_PER_GROUP_LOWER_LIMIT :
                                                               TENSOR_API_NOT_TRANS_B_M_PER_GROUP_LOWER_LIMIT;
    const bool checkMSizeGroupNumRatio = averageMPerGroup > mPerGroupLowerLimit;
    const bool checkNSizeAlign = inputParams_.nSize % TENSOR_API_FP8_N_ALIGN == 0;
    const bool checkBNoTransMLimlit =
        inputParams_.transB ? true : inputParams_.mSize > TENSOR_API_B_NOTRANS_M_LOWER_LIMIT;
    const bool checkKSize = inputParams_.kSize > 64;
    const bool checkKSizeAlign = inputParams_.kSize % MX_SCALE_K_ALIGN == 0;
    const bool capable = dtypeSupported && quantDtypeSupported && formatSupported && checkMSizeGroupNumRatio &&
                         checkNSizeAlign && coreSupported && checkTensorApiShapes && checkTensorApiScaleShapes &&
                         checkBNoTransMLimlit && checkKSize && checkKSizeAlign;
    OP_LOGD(context_->GetNodeName(),
            "Tensor API capability: dtype=%d, quantDtypeSupported=%d, format=%d, groupNum=%lu, M=%lu, "
            "averageMPerGroup=%lu, mPerGroupLowerLimit=%lu, transB=%d, N=%lu, "
            "cores=%lu:%u, "
            "checkTensorApiShapes=%d, checkKSize=%d, checkKSizeAlign=%d, "
            "checkTensorApiScaleShapes=%d, checkMSizeGroupNumRatio=%d, checkBNoTransMLimlit=%d, capable=%d.",
            dtypeSupported, quantDtypeSupported, formatSupported, inputParams_.groupNum, inputParams_.mSize,
            averageMPerGroup, mPerGroupLowerLimit, inputParams_.transB, inputParams_.nSize, aicoreParams_.aicNum,
            aivNum_, checkTensorApiShapes, checkKSize, checkKSizeAlign, checkTensorApiScaleShapes,
            checkMSizeGroupNumRatio, checkBNoTransMLimlit, capable);
    return capable;
}

ge::graphStatus GroupedMatmulSwigluQuantV2BasicApiTiling950::DoOpTiling()
{
    auto &params = tilingData_.gmmQuantParams;
    params.groupNum = static_cast<uint32_t>(inputParams_.groupNum);
    params.activeType = static_cast<uint32_t>(inputParams_.actType);
    params.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    params.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    params.singleX = static_cast<uint8_t>(inputParams_.isSingleX);
    params.singleW = static_cast<uint8_t>(inputParams_.isSingleW);
    params.singleY = static_cast<uint8_t>(inputParams_.isSingleY);
    params.groupType = static_cast<int8_t>(inputParams_.groupType);
    params.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2BasicApiTiling950::DoLibApiTiling()
{
    GroupedQmmTiling::CalBasicBlock();
    OP_CHECK_IF(GroupedQmmBasicApiTiling::CalL1Tiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed."), return ge::GRAPH_FAILED);
    if (inputParams_.mSize >= GmmConstant::BASIC_BLOCK_SIZE_256) {
        basicTiling_.baseM = TENSOR_API_LARGE_M_BASE_M;
    }
    if (inputParams_.nSize > GmmConstant::BASIC_BLOCK_SIZE_256 &&
        inputParams_.kSize >= GmmConstant::BASIC_BLOCK_SIZE_128) {
        basicTiling_.baseN = TENSOR_API_LARGE_N_BASE_N;
    }

    auto &mmTiling = tilingData_.mmTilingData;
    mmTiling.m = static_cast<uint32_t>(inputParams_.mSize);
    mmTiling.n = static_cast<uint32_t>(inputParams_.nSize);
    mmTiling.k = static_cast<uint32_t>(inputParams_.kSize);
    mmTiling.baseM = static_cast<uint32_t>(basicTiling_.baseM);
    mmTiling.baseN = static_cast<uint32_t>(basicTiling_.baseN);
    mmTiling.baseK = static_cast<uint32_t>(basicTiling_.baseK);
    mmTiling.kAL1 = static_cast<uint32_t>(basicTiling_.stepKa * basicTiling_.baseK);
    mmTiling.kBL1 = static_cast<uint32_t>(basicTiling_.stepKb * basicTiling_.baseK);
    const uint64_t scaleKL1 = std::min(
        std::max(basicTiling_.scaleFactorA * basicTiling_.stepKa, basicTiling_.scaleFactorB * basicTiling_.stepKb) *
            basicTiling_.baseK,
        inputParams_.kSize);
    mmTiling.scaleKAL1 = static_cast<uint32_t>(scaleKL1);
    mmTiling.scaleKBL1 = static_cast<uint32_t>(scaleKL1);
    mmTiling.dbL0C = static_cast<uint8_t>(basicTiling_.dbL0c);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2BasicApiTiling950::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = GmmConstant::SYS_WORKSPACE_SIZES;
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulSwigluQuantV2BasicApiTiling950::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transB), static_cast<uint64_t>(inputParams_.transA),
                              static_cast<uint64_t>(GMM_SWIGLU_QUANT_TENSOR_LEVEL_KERNEL_TYPE));
}

ge::graphStatus GroupedMatmulSwigluQuantV2BasicApiTiling950::PostTiling()
{
    context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    return SaveTilingDataToContext(tilingData_);
}
} // namespace optiling
