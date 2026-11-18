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
 * \file grouped_weight_quant_matmul_tiling_tool.cpp
 * \brief
 */
#include "common/include/op_host/op_const_def_transformer.h"
#include "grouped_weight_quant_matmul_tiling_tool.h"

namespace GmmConstants = optiling::GMMWeightQuantConstants;

namespace optiling {
namespace GMMWeightQuantTilingTool {

uint64_t GetShapeWithDataType(uint64_t shapeSize, ge::DataType dtype)
{
    bool is4BitInput = (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4 ||
                        dtype == Ops::Transformer::DT_HIFLOAT4);
    if (is4BitInput) {
        return shapeSize + shapeSize;
    } else {
        return shapeSize / static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

bool AnalyzeInputs(const gert::TilingContext* context, GMMWeightQuantInputInfo& inputParams,
                   GMMWeightQuantInputNum& inputNum)
{
    GetNumOfInputs(context, inputNum);

    OP_CHECK_IF(!AnalyzeAttrs(context, inputParams, inputNum), OP_LOGE(inputParams.opName, "Failed to AnalyzeAttrs."),
                return false);
    OP_CHECK_IF(!AnalyzeInputParams(context, inputParams, inputNum),
                OP_LOGE(inputParams.opName, "Failed to AnalyzeInputParams."), return false);
    OP_CHECK_IF(!AnalyzeGroupList(context, inputParams), OP_LOGE(inputParams.opName, "Failed to AnalyzeGroupList."),
                return false);
    OP_CHECK_IF(!AnalyzeQuantInputParams(context, inputParams),
                OP_LOGE(inputParams.opName, "Failed to AnalyzeQuantInputParams."), return false);

    inputParams.weightQuantMode = AntiQuantMode::DEFAULT;
    if (IsMxA8W4(inputParams) || IsMxA8HiF4(inputParams) || IsA16MxFp4(inputParams) || IsA16MxHiF4(inputParams)) {
        inputParams.weightQuantMode = AntiQuantMode::MX_PERGROUP_MODE;
    } else if (inputNum.numWeight == 1 && inputNum.numAntiquantScale == 1) {
        const auto* scaleShape = context->GetDynamicInputShape(GmmConstants::ANTIQUANT_SCALE_INDEX, 0);
        const auto* weightShape = context->GetDynamicInputShape(GmmConstants::WEIGHT_INDEX, 0);
        if (scaleShape != nullptr && weightShape != nullptr &&
            IsPerChannelAntiquantScale(scaleShape->GetStorageShape(), weightShape->GetOriginShape(),
                                       inputParams.groupNum, inputParams.transWeight)) {
            inputParams.weightQuantMode = AntiQuantMode::PERCHANNEL_MODE;
        }
    }
    return true;
}

void GetNumOfInputs(const gert::TilingContext* context, GMMWeightQuantInputNum& inputNum)
{
    inputNum.numX = GetTensorListSize(context, GmmConstants::X_INDEX);
    inputNum.numWeight = GetTensorListSize(context, GmmConstants::WEIGHT_INDEX);
    inputNum.numBias = GetTensorListSize(context, GmmConstants::BIAS_INDEX);
    inputNum.numAntiquantScale = GetTensorListSize(context, GmmConstants::ANTIQUANT_SCALE_INDEX);
    inputNum.numAntiquantOffset = GetTensorListSize(context, GmmConstants::ANTIQUANT_OFFSET_INDEX);
}

uint16_t GetTensorListSize(const gert::TilingContext* context, size_t inputIdx)
{
    uint16_t count = 0;
    for (uint16_t i = 0; i < GmmConstants::MAX_TENSOR_CONT; i++) {
        auto shapePtr = context->GetDynamicInputShape(inputIdx, count);
        if (shapePtr == nullptr) {
            break;
        }
        ++count;
    }
    return count;
}

bool AnalyzeAttrs(const gert::TilingContext* context, GMMWeightQuantInputInfo& inputParams,
                  GMMWeightQuantInputNum& inputNum)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(inputParams.opName, "Attrs is nullptr."), return false);

    const bool* transposeXPtr = attrs->GetAttrPointer<bool>(GmmConstants::ATTR_INDEX_TRANS_X);
    const bool* transposeWeightPtr = attrs->GetAttrPointer<bool>(GmmConstants::ATTR_INDEX_TRANS_W);
    const int64_t* groupTypePtr = attrs->GetAttrPointer<int64_t>(GmmConstants::ATTR_INDEX_GROUPTYPE);
    const int64_t* splitItemPtr = attrs->GetAttrPointer<int64_t>(GmmConstants::ATTR_INDEX_SPLIT_ITEM);
    const int64_t* groupListTypePtr = attrs->GetAttrPointer<int64_t>(GmmConstants::ATTR_INDEX_GROUP_LIST_TYPE);

    inputParams.transX = transposeXPtr != nullptr ? *transposeXPtr : false;
    inputParams.transWeight = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    inputParams.groupType = groupTypePtr != nullptr ? *groupTypePtr : inputParams.groupType;
    inputParams.splitItem = splitItemPtr != nullptr ? *splitItemPtr : inputParams.splitItem;
    inputParams.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams.groupListType;
    OP_LOGD(inputParams.opName, "transX=%d, transWeight=%d, groupType=%ld, splitItem=%ld, groupListType=%ld",
            inputParams.transX, inputParams.transWeight, inputParams.groupType, inputParams.splitItem,
            inputParams.groupListType);

    inputParams.isSingleX = inputParams.groupType != GmmConstants::NO_SPLIT && inputNum.numX == 1;
    inputParams.isSingleW = inputParams.groupType != GmmConstants::NO_SPLIT && inputNum.numWeight == 1;
    // 2: when x is multi-tensor, y is single-tensor; 3: when x is single-tensor, y is single-tensor
    inputParams.isSingleY = (inputParams.splitItem == 2 || inputParams.splitItem == 3);
    return true;
}

bool AnalyzeInputParams(const gert::TilingContext* context, GMMWeightQuantInputInfo& inputParams,
                        GMMWeightQuantInputNum& inputNum)
{
    auto xDesc = context->GetDynamicInputDesc(GmmConstants::X_INDEX, 0);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(inputParams.opName, "xDesc is nullptr."), return false);
    inputParams.xDtype = xDesc->GetDataType();

    auto wDesc = context->GetDynamicInputDesc(GmmConstants::WEIGHT_INDEX, 0);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(inputParams.opName, "wDesc is nullptr."), return false);
    inputParams.weightDtype = wDesc->GetDataType();

    inputParams.weightFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    inputParams.isWeightNz = inputParams.weightFormat == ge::FORMAT_FRACTAL_NZ;

    inputParams.hasBias = inputNum.numBias > 0;
    if (inputNum.numBias == 1) {
        auto biasShape = context->GetDynamicInputShape(GmmConstants::BIAS_INDEX, 0);
        inputParams.hasBias = biasShape->GetStorageShape().GetShapeSize() != 0;
    }
    if (inputParams.hasBias) {
        auto biasDesc = context->GetDynamicInputDesc(GmmConstants::BIAS_INDEX, 0);
        OP_CHECK_IF(biasDesc == nullptr, OP_LOGE(inputParams.opName, "biasDesc is nullptr."), return false);
        inputParams.biasDtype = biasDesc->GetDataType();
    }

    auto antiquantScaleDesc = context->GetDynamicInputDesc(GmmConstants::ANTIQUANT_SCALE_INDEX, 0);
    OP_CHECK_IF(antiquantScaleDesc == nullptr, OP_LOGE(inputParams.opName, "antiquantScaleDesc is nullptr."),
                return false);
    inputParams.antiquantScaleDtype = antiquantScaleDesc->GetDataType();

    inputParams.hasAntiquantOffset = inputNum.numAntiquantOffset > 0;
    if (inputNum.numAntiquantOffset == 1) {
        auto antiQuantOffsetShape = context->GetDynamicInputShape(GmmConstants::ANTIQUANT_OFFSET_INDEX, 0);
        inputParams.hasAntiquantOffset = antiQuantOffsetShape->GetStorageShape().GetShapeSize() != 0;
    }
    if (inputParams.hasAntiquantOffset) {
        auto antiquantOffsetDesc = context->GetDynamicInputDesc(GmmConstants::ANTIQUANT_OFFSET_INDEX, 0);
        OP_CHECK_IF(antiquantOffsetDesc == nullptr, OP_LOGE(inputParams.opName, "antiquantOffsetDesc is nullptr."),
                    return false);
        inputParams.antiquantOffsetDtype = antiquantOffsetDesc->GetDataType();
    }

    auto yDesc = context->GetOutputDesc(GmmConstants::Y_INDEX);
    OP_CHECK_IF(yDesc == nullptr, OP_LOGE(inputParams.opName, "yDesc is nullptr."), return false);
    inputParams.yDtype = yDesc->GetDataType();

    OP_LOGD(inputParams.opName, "xDtype [%s], weightDtype [%s], antiquantScaleDtype [%s], yDtype [%s]",
            ge::TypeUtils::DataTypeToSerialString(inputParams.xDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputParams.weightDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputParams.antiquantScaleDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputParams.yDtype).c_str());

    return true;
}

bool AnalyzeGroupList(const gert::TilingContext* context, GMMWeightQuantInputInfo& inputParams)
{
    if (inputParams.groupType == GmmConstants::SPLIT_M) {
        auto groupListShapePtr = context->GetOptionalInputShape(GmmConstants::GROUPLIST_INDEX);
        OP_CHECK_IF(groupListShapePtr == nullptr, OP_LOGE(inputParams.opName, "groupListShapePtr is nullptr."),
                    return false);
        auto groupListShape = groupListShapePtr->GetStorageShape();
        OP_CHECK_IF(
            groupListShape.GetDimNum() != GmmConstants::DIM_NUM_1,
            OP_LOGE(inputParams.opName, "The dimension of groupList should be [%zu], the actual dimension is [%zu].",
                    GmmConstants::DIM_NUM_1, groupListShape.GetDimNum()),
            return false);
        inputParams.groupNum = groupListShape.GetDim(0);
    }
    return true;
}

bool AnalyzeQuantInputParams(const gert::TilingContext* context, GMMWeightQuantInputInfo& inputParams)
{
    if (IsMxA8W4(inputParams) || IsMxA8HiF4(inputParams)) {
        auto perTokenScaleDesc = context->GetOptionalInputDesc(GmmConstants::PER_TOKEN_SCALE_INDEX);
        OP_CHECK_IF(perTokenScaleDesc == nullptr, OP_LOGE(inputParams.opName, "perTokenScaleDesc is nullptr."),
                    return false);
        inputParams.perTokenScaleDtype = perTokenScaleDesc->GetDataType();
    }
    return true;
}

bool IsMxA8W4(const GMMWeightQuantInputInfo& inputParams)
{
    return inputParams.xDtype == ge::DT_FLOAT8_E4M3FN && inputParams.weightDtype == ge::DT_FLOAT4_E2M1 &&
           inputParams.antiquantScaleDtype == ge::DT_FLOAT8_E8M0;
}

bool IsMxA8HiF4(const GMMWeightQuantInputInfo& inputParams)
{
    return inputParams.xDtype == ge::DT_FLOAT8_E4M3FN && inputParams.weightDtype == Ops::Transformer::DT_HIFLOAT4 &&
           inputParams.antiquantScaleDtype == Ops::Transformer::DT_HIFLOAT4_SCALE;
}

bool IsA16W8(const GMMWeightQuantInputInfo& inputParams)
{
    return (inputParams.xDtype == ge::DT_FLOAT16 || inputParams.xDtype == ge::DT_BF16) &&
           (inputParams.weightDtype == ge::DT_INT8 || inputParams.weightDtype == ge::DT_FLOAT8_E4M3FN ||
            inputParams.weightDtype == ge::DT_FLOAT8_E5M2 || inputParams.weightDtype == ge::DT_HIFLOAT8);
}

bool IsA16MxFp4(const GMMWeightQuantInputInfo& inputParams)
{
    return (inputParams.xDtype == ge::DT_FLOAT16 || inputParams.xDtype == ge::DT_BF16) &&
           inputParams.weightDtype == ge::DT_FLOAT4_E2M1 && inputParams.antiquantScaleDtype == ge::DT_FLOAT8_E8M0;
}

bool IsA16MxHiF4(const GMMWeightQuantInputInfo& inputParams)
{
    return (inputParams.xDtype == ge::DT_FLOAT16 || inputParams.xDtype == ge::DT_BF16) &&
           inputParams.weightDtype == Ops::Transformer::DT_HIFLOAT4 &&
           inputParams.antiquantScaleDtype == Ops::Transformer::DT_HIFLOAT4_SCALE;
}

} // namespace GMMWeightQuantTilingTool

} // namespace optiling
