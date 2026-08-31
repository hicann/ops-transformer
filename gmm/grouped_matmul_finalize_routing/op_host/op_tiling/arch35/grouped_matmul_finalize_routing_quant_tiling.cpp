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
 * \file grouped_matmul_finalize_routing_quant_tiling.cpp
 * \brief
 */

#include "grouped_matmul_finalize_routing_quant_tiling.h"
#include <alog_pub.h>
#include <climits>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"
using namespace Ops::Transformer::OpTiling;
using namespace optiling::GroupedMatmulFinalizeRoutingArch35TilingConstant;
using namespace optiling::GmmConstant;
using namespace GMMFinalizeRoutingArch35Tiling;

namespace optiling {

void GroupedMatmulFinalizeRoutingQuantTiling::Reset()
{
    tilingData_ = GMMFinalizeRoutingTilingData();
    OP_CHECK_IF(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                         context_->GetRawTilingData()->GetCapacity()) != EOK,
                OP_LOGE(inputParams_.opName, "Fail to clear tiling data"), return);
    return;
}

ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    return GroupedQmmTiling::GetShapeAttrsInfo();
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckOptionalAttr()
{
    auto *attrs = context_->GetAttrs();
    const int64_t *shareInputOffsetPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SHARE_INPUT_OFFSET);
    const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
    const int64_t *outputBSPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_OUTPUT_BS);
    const int64_t *outputDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DTYPE);
    OP_CHECK_IF(outputBSPtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "output_bs", "nullptr",
                                                      "attr batch cannot be nullptr"),
                return false);
    int64_t shareInputOffset = shareInputOffsetPtr != nullptr ? *shareInputOffsetPtr : 0;
    int64_t groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : 1;
    int64_t outputBS = outputBSPtr != nullptr ? *outputBSPtr : 0;
    int64_t outputDtype = outputDtypePtr != nullptr ? *outputDtypePtr : 0;
    OP_CHECK_IF(
        shareInputOffset < 0 || outputBS < 0 || groupListType < 0 || outputDtype < 0,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            inputParams_.opType, "share_input_offset, group_list_type, output_bs, dtype",
            ShapeDimsToString(shareInputOffset, groupListType, outputBS, outputDtype),
            "attr shareInputOffset, groupListType, outputBS and outputDtype must be greater than or equal to 0"),
        return false);

    inputParams_.groupListType = static_cast<uint64_t>(groupListType);
    sharedInputOffset_ = static_cast<uint64_t>(shareInputOffset);
    outputBs_ = static_cast<uint64_t>(outputBS);
    OP_CHECK_IF(inputParams_.groupListType != 0 && inputParams_.groupListType != 1,
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "group_list_type",
                                          std::to_string(inputParams_.groupListType), "0 or 1"),
                return false);

    OP_CHECK_IF(sharedInputOffset_ > outputBs_,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    inputParams_.opType, "share_input_offset", std::to_string(sharedInputOffset_),
                    BuildErrorMsgStr("shareInputOffset must be less than or equal to batch(", outputBs_, ")")),
                return false);

    OP_CHECK_IF(outputDtype > OUT_DTYPE_BF16_INDEX,
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "dtype", std::to_string(outputDtype),
                                          "0(float32), 1(float16) or 2(bfloat16)"),
                return false);

    OP_CHECK_IF(inputParams_.transA, OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "transpose_x", "true", "false"),
                return false);
    return true;
}
bool GroupedMatmulFinalizeRoutingQuantTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "attrs", "nullptr", "attrs cannot be nullptr"),
        return false);
    OP_CHECK_IF(attrs->GetAttrNum() < ATTR_INDEX_TUNING_CONFIG + 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    inputParams_.opType, "attrs", std::to_string(attrs->GetAttrNum()),
                    BuildErrorMsgStr("the num of attrs must be greater than ", ATTR_INDEX_TUNING_CONFIG + 1)),
                return false);
    const float *shareInputWeightPtr = attrs->GetAttrPointer<float>(ATTR_INDEX_SHARE_INPUT_WEIGHT);
    const bool *transposeXPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_X);
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_W);
    OP_CHECK_IF(shareInputWeightPtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "residual_scale", "nullptr",
                                                      "attr residualScale cannot be nullptr"),
                return false);
    inputParams_.transA = transposeXPtr != nullptr ? *transposeXPtr : false;
    inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    inputParams_.groupType = SPLIT_M;
    sharedInputWeight_ = *shareInputWeightPtr;
    OP_CHECK_IF(!CheckOptionalAttr(), OP_LOGE(context_->GetNodeName(), "Check Optional Attrs Failed."), return false);
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::AnalyzeDtype()
{
    OP_CHECK_IF(!LoadInputDescsForRouting(), OP_LOGE(inputParams_.opName, "LoadInputDescsForRouting failed."),
                return false);
    auto outDesc = context_->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(
        outDesc == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "y", "nullptr", "input outDesc cannot be nullptr"),
        return false);
    inputParams_.cDtype = outDesc->GetDataType();
    OP_CHECK_IF(inputParams_.cDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opType, "y",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype), "DT_FLOAT"),
                return false);

    auto biasStorageShape = context_->GetDynamicInputShape(BIAS_INDEX, 0);
    inputParams_.hasBias = !(biasStorageShape == nullptr || biasStorageShape->GetStorageShape().GetShapeSize() == 0);
    auto biasDesc = context_->GetDynamicInputDesc(BIAS_INDEX, 0);
    OP_CHECK_IF(inputParams_.hasBias && biasDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "bias", "nullptr",
                                                      "bias from tensor is nonnull, but bias from desc is nullptr"),
                return false);
    inputParams_.biasDtype = inputParams_.hasBias ? biasDesc->GetDataType() : ge::DT_BF16;

    OP_CHECK_IF(!CheckDtype(), OP_LOGE(context_->GetNodeName(), "Required input check failed."), return false);
    return CheckOptionalInputsForRouting();
}

bool GroupedMatmulFinalizeRoutingQuantTiling::LoadInputDescsForRouting()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_IF(
        xDesc == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "x", "nullptr", "input xDesc cannot be nullptr"),
        return false);
    inputParams_.aDtype = xDesc->GetDataType();
    auto wDesc = context_->GetInputDesc(W_INDEX);
    OP_CHECK_IF(wDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "weight", "nullptr",
                                                      "input wDesc cannot be nullptr"),
                return false);
    inputParams_.bDtype = wDesc->GetDataType();
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    auto scaleDesc = context_->GetInputDesc(SCALE_INDEX);
    OP_CHECK_IF(scaleDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "scale", "nullptr",
                                                      "input scaleDesc cannot be nullptr"),
                return false);
    inputParams_.scaleDtype = scaleDesc != nullptr ? scaleDesc->GetDataType() : inputParams_.scaleDtype;
    if (inputParams_.scaleDtype == ge::DT_FLOAT) {
        scaleType_ = 1;
    } else if (inputParams_.scaleDtype == ge::DT_BF16) {
        scaleType_ = 2; // 2 represents bf16 dtype
    }
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX);
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckOptionalInputsForRouting()
{
    OP_CHECK_IF(!CheckOptional(GROUPLIST_INDEX, "GroupList", ge::DT_INT64),
                OP_LOGE(context_->GetNodeName(), "GroupList check failed."), return false);
    OP_CHECK_IF(!CheckOptional(SHARE_INPUT_INDEX, "SharedInput", ge::DT_BF16),
                OP_LOGE(context_->GetNodeName(), "SharedInput check failed."), return false);
    OP_CHECK_IF(!CheckOptional(LOGIT_INDEX, "LogitIndex", ge::DT_FLOAT),
                OP_LOGE(context_->GetNodeName(), "LogitIndex check failed."), return false);
    if (inputParams_.aDtype != ge::DT_INT8) {
        OP_CHECK_IF(!CheckOptional(ROW_INDEX_INDEX, "RowIndex", ge::DT_INT64),
                    OP_LOGE(context_->GetNodeName(), "RowIndex check failed."), return false);
    } else if (context_->GetOptionalInputDesc(ROW_INDEX_INDEX) != nullptr) {
        auto rowIndexDtype = context_->GetOptionalInputDesc(ROW_INDEX_INDEX)->GetDataType();
        if (rowIndexDtype == ge::DT_INT32) {
            rowIndexType_ = 1;
        }
        OP_CHECK_IF(!(rowIndexDtype == ge::DT_INT64 || rowIndexDtype == ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opType, "row_index", ge::TypeUtils::DataTypeToSerialString(rowIndexDtype),
                        "when inputs are DT_INT8, the dtype of row_index must be within the range DT_INT64/DT_INT32"),
                    return false);
    }
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckOptional(uint32_t index, const char *paramName,
                                                            ge::DataType targetDtype) const
{
    auto optionalDesc = context_->GetOptionalInputDesc(index);
    if (optionalDesc == nullptr) {
        return true;
    }
    auto realDtype = optionalDesc->GetDataType();
    OP_CHECK_IF(
        realDtype != targetDtype,
        OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opType, paramName, ge::TypeUtils::DataTypeToSerialString(realDtype),
                                  ge::TypeUtils::DataTypeToSerialString(targetDtype)),
        return false);
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::IsFp4Dtype(ge::DataType dtype) const
{
    return dtype == ge::DT_FLOAT4_E2M1;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::IsFp8Dtype(ge::DataType dtype) const
{
    return (dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2);
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckDtype() const
{
    OP_CHECK_IF(inputParams_.biasDtype != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opType, "bias",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype), "DT_BF16"),
                return false);

    if (IsMicroScaling()) {
        bool a8w8 = IsFp8Dtype(inputParams_.aDtype) && IsFp8Dtype(inputParams_.bDtype);
        bool a4w4 = IsFp4Dtype(inputParams_.aDtype) && IsFp4Dtype(inputParams_.bDtype);
        OP_CHECK_IF(!(a8w8 || a4w4),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        inputParams_.opType, "x, weight",
                        ListToString(ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype),
                                     ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype)),
                        "in mx quant mode, the dtypes of x and weight must be within the range "
                        "DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2/DT_FLOAT4_E2M1"),
                    return false);
    } else {
        OP_CHECK_IF(!(inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_INT8 ||
                      inputParams_.aDtype == ge::DT_HIFLOAT8),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        inputParams_.opType, "x, weight",
                        ListToString(ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype),
                                     ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype)),
                        "in K-C/T-C quant mode, the dtypes of x and weight must be within the range "
                        "DT_FLOAT8_E4M3FN/DT_INT8/DT_HIFLOAT8"),
                    return false);

        OP_CHECK_IF(!(inputParams_.scaleDtype == ge::DT_FLOAT || inputParams_.scaleDtype == ge::DT_BF16),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opType, "scale", ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype),
                        "in K-C/T-C quant mode, the dtype of scale must be within the range DT_FLOAT/DT_BF16"),
                    return false);

        if (context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX) != nullptr) {
            OP_CHECK_IF(inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams_.opType, "perTokenScale",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype),
                            "in K-C quant mode, the dtype of perTokenScale must be DT_FLOAT"),
                        return false);
        }
    }

    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckDim(const gert::Shape &xShape, const gert::Shape &wShape,
                                                       const gert::StorageShape *pertokenScaleStorageShape,
                                                       const gert::Shape &scaleShape, const gert::Shape &yShape) const
{
    auto xDimNum = xShape.GetDimNum();
    OP_CHECK_IF(
        xDimNum != DIM_NUM_X,
        OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "x", std::to_string(xDimNum), std::to_string(DIM_NUM_X)),
        return false);

    auto wDimNum = wShape.GetDimNum();
    OP_CHECK_IF(wDimNum != DIM_NUM_WEIGHT,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "weight", std::to_string(wDimNum),
                                             std::to_string(DIM_NUM_WEIGHT)),
                return false);

    auto yDimNum = yShape.GetDimNum();
    OP_CHECK_IF(
        yDimNum != DIM_NUM_Y,
        OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "y", std::to_string(yDimNum), std::to_string(DIM_NUM_Y)),
        return false);

    return CheckScaleAndPerTokenDims(pertokenScaleStorageShape, scaleShape);
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckScaleAndPerTokenDims(
    const gert::StorageShape *pertokenScaleStorageShape, const gert::Shape &scaleShape) const
{
    auto scaleDimNum = scaleShape.GetDimNum();
    if (IsMicroScaling()) {
        OP_CHECK_IF(scaleDimNum != DIM_NUM_MX_SCALE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "scale", std::to_string(scaleDimNum),
                                                 std::to_string(DIM_NUM_MX_SCALE)),
                    return false);
        OP_CHECK_IF(pertokenScaleStorageShape == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "perTokenScale", "nullptr",
                                                          "input pertokenScaleStorageShape cannot be nullptr"),
                    return false);
        const gert::Shape &pertokenScaleShape = pertokenScaleStorageShape->GetOriginShape();
        auto pertokenScaleDimNum = pertokenScaleShape.GetDimNum();
        OP_CHECK_IF(
            pertokenScaleDimNum != DIM_NUM_MX_PERTOKENSCALE,
            OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "perTokenScale", std::to_string(pertokenScaleDimNum),
                                         std::to_string(DIM_NUM_MX_PERTOKENSCALE)),
            return false);
    } else {
        OP_CHECK_IF(scaleDimNum != DIM_NUM_SCALE,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "scale", std::to_string(scaleDimNum),
                                                 std::to_string(DIM_NUM_SCALE)),
                    return false);
        if (pertokenScaleStorageShape != nullptr) {
            const gert::Shape &pertokenScaleShape = pertokenScaleStorageShape->GetOriginShape();
            auto pertokenScaleDimNum = pertokenScaleShape.GetDimNum();
            OP_CHECK_IF(
                pertokenScaleDimNum != DIM_NUM_PERTOKENSCALE,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "perTokenScale", std::to_string(pertokenScaleDimNum),
                                             std::to_string(DIM_NUM_PERTOKENSCALE)),
                return false);
        }
    }
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckFp4Shape(const gert::Shape &xShape, const gert::Shape &wShape) const
{
    bool a4w4 = IsFp4Dtype(inputParams_.aDtype) && IsFp4Dtype(inputParams_.bDtype);
    if (!a4w4) {
        return true;
    }
    OP_CHECK_IF(inputParams_.kSize % EVEN_FACTOR != 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(inputParams_.opType, "x, weight",
                                                       ShapesToString({ShapeToString(xShape), ShapeToString(wShape)}),
                                                       "when the dtype of x is FLOAT4, k size must be even number"),
                return false);
    // 2: mxfp4场景下不支持K轴为2
    OP_CHECK_IF(inputParams_.kSize == 2,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(inputParams_.opType, "x, weight",
                                                       ShapesToString({ShapeToString(xShape), ShapeToString(wShape)}),
                                                       "when the dtype of x is FLOAT4, k size cannot be 2"),
                return false);
    if (!inputParams_.transB) {
        OP_CHECK_IF(inputParams_.nSize % EVEN_FACTOR != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opType, "weight", ShapeToString(wShape),
                        "when the dtype of weight is FLOAT4 and weight is not transposed, n size must be even number"),
                    return false);
    }
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckOptionalInputsShape()
{
    auto sharedInputDesc = context_->GetOptionalInputDesc(SHARE_INPUT_INDEX);
    sharedInputLen_ = sharedInputDesc != nullptr ?
                          context_->GetOptionalInputShape(SHARE_INPUT_INDEX)->GetStorageShape()[0] :
                          sharedInputLen_;

    OP_CHECK_IF(sharedInputLen_ > outputBs_,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    inputParams_.opType, "shared_input", std::to_string(sharedInputLen_),
                    BuildErrorMsgStr("shared_input_len must be less than or equal to batch(", outputBs_, ")")),
                return false);

    OP_CHECK_IF(sharedInputOffset_ + sharedInputLen_ > outputBs_,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    inputParams_.opType, "share_input_offset, shared_input",
                    std::to_string(sharedInputOffset_ + sharedInputLen_),
                    BuildErrorMsgStr("shareInputOffset plus sharedInputLen must be less than or equal to batch(",
                                     outputBs_, ")")),
                return false);

    auto LogitDesc = context_->GetOptionalInputDesc(LOGIT_INDEX);
    OP_CHECK_IF(
        LogitDesc == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "logit", "nullptr", "logitDesc cannot be nullptr"),
        return false);

    auto rowIndexDesc = context_->GetOptionalInputDesc(ROW_INDEX_INDEX);
    OP_CHECK_IF(rowIndexDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "row_index", "nullptr",
                                                      "rowIndexDesc cannot be nullptr"),
                return false);
    rowIndex_ = context_->GetOptionalInputShape(ROW_INDEX_INDEX)->GetStorageShape()[0];

    OP_CHECK_IF(rowIndex_ > inputParams_.mSize,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    inputParams_.opType, "row_index", std::to_string(rowIndex_),
                    BuildErrorMsgStr("rowIndex must be less than or equal to M (", inputParams_.mSize, ")")),
                return false);

    OP_CHECK_IF(outputBs_ > inputParams_.mSize,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    inputParams_.opType, "output_bs", std::to_string(outputBs_),
                    BuildErrorMsgStr("outputBS must be less than or equal to M (", inputParams_.mSize, ")")),
                return false);
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckInputsShape(const gert::Shape &xShape,
                                                               const gert::StorageShape *wStorageShape,
                                                               const gert::StorageShape *pertokenScaleStorageShape,
                                                               const gert::Shape &scaleShape,
                                                               const gert::Shape &yShape) const
{
    const gert::Shape &wShape = wStorageShape->GetOriginShape();
    OP_CHECK_IF(!CheckDim(xShape, wShape, pertokenScaleStorageShape, scaleShape, yShape),
                OP_LOGE(context_->GetNodeName(), "CheckDim failed."), return false);
    if (IsMicroScaling()) {
        OP_CHECK_IF(!CheckFp4Shape(xShape, wShape), OP_LOGE(context_->GetNodeName(), "CheckFp4Shape failed."),
                    return false);
    }
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::AnalyzeInputs()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "x", "nullptr",
                                                      "input xStorageShape cannot be nullptr"),
                return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();

    auto wStorageShape = context_->GetInputShape(W_INDEX);
    OP_CHECK_IF(wStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "weight", "nullptr",
                                                      "input wStorageShape cannot be nullptr"),
                return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();

    auto scaleStorageShape = context_->GetInputShape(SCALE_INDEX);
    OP_CHECK_IF(scaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "scale", "nullptr",
                                                      "input scaleStorageShape cannot be nullptr"),
                return false);
    const gert::Shape &scaleShape = scaleStorageShape->GetOriginShape();

    auto pertokenScaleStorageShape = context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX);

    auto yStorageShape = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_IF(yStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "y", "nullptr",
                                                      "output yStorageShape cannot be nullptr"),
                return false);
    const gert::Shape &yShape = yStorageShape->GetOriginShape();

    if (!IsMicroScaling()) {
        OP_CHECK_IF(inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                        inputParams_.opType, "weight", ge::TypeUtils::FormatToSerialString(inputParams_.bFormat),
                        "in K-C/T-C quant mode, the format of weight must be FRACTAL_NZ"),
                    return false);
    }
    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(context_->GetNodeName(), "SetGroupNum failed."), return false);
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(context_->GetNodeName(), "SetMKN failed."), return false);
    OP_CHECK_IF(!CheckInputsShape(xShape, wStorageShape, pertokenScaleStorageShape, scaleShape, yShape),
                OP_LOGE(context_->GetNodeName(), "CheckInputsShape failed."), return false);
    OP_CHECK_IF(!CheckOptionalInputsShape(), OP_LOGE(context_->GetNodeName(), "CheckOptionalInputsShape failed."),
                return false);
    OP_CHECK_IF(!SetQuantModeForGMMFinalizeRouting(),
                OP_LOGE(context_->GetNodeName(), "SetQuantModeForGMMFinalizeRouting failed."), return false);
    OP_CHECK_IF(!CheckCoreNum(), OP_LOGE(inputParams_.opName, "CheckCoreNum failed."), return false);
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::CheckCoreNum() const
{
    auto compileInfo = context_->GetCompileInfo<GroupedMatmulFinalizeRoutingCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr,
                OP_LOGE(inputParams_.opName, "GroupedMatmulFinalizeRouting compileInfo is nullptr."), return false);
    auto aicNum = compileInfo->aicNum;
    auto aivNum = compileInfo->aivNum;
    OP_CHECK_IF(aicNum == 0, OP_LOGE(inputParams_.opName, "aicNum should be positive integer, actual is %u.", aicNum),
                return false);
    OP_CHECK_IF(
        aivNum != GmmConstant::CORE_RATIO * aicNum,
        OP_LOGE(inputParams_.opName, "aicNum:aivNum should be 1:2, actual aicNum: %u, aivNum: %u.", aicNum, aivNum),
        return false);
    return true;
}

bool GroupedMatmulFinalizeRoutingQuantTiling::SetQuantModeForGMMFinalizeRouting()
{
    if (IsMicroScaling()) {
        inputParams_.bQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        inputParams_.aQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        return true;
    }

    inputParams_.bQuantMode = optiling::QuantMode::PERCHANNEL_MODE;
    if (context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX) != nullptr) {
        inputParams_.aQuantMode = optiling::QuantMode::PERTOKEN_MODE;
    } else {
        inputParams_.aQuantMode = optiling::QuantMode::DEFAULT;
    }

    return true;
}

ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::DoOpTiling()
{
    tilingData_.gmmFinalizeRoutingDataParams.groupNum = static_cast<uint32_t>(inputParams_.groupNum);
    tilingData_.gmmFinalizeRoutingDataParams.batch = static_cast<uint32_t>(outputBs_);
    tilingData_.gmmFinalizeRoutingDataParams.sharedInputOffset = static_cast<uint32_t>(sharedInputOffset_);
    tilingData_.gmmFinalizeRoutingDataParams.sharedInputLen = static_cast<uint32_t>(sharedInputLen_);
    tilingData_.gmmFinalizeRoutingDataParams.residualScale = static_cast<float>(sharedInputWeight_);
    tilingData_.gmmFinalizeRoutingDataParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmFinalizeRoutingDataParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmFinalizeRoutingDataParams.biasDtype = static_cast<uint32_t>(inputParams_.biasDtype);
    tilingData_.gmmFinalizeRoutingDataParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmFinalizeRoutingDataParams.hasBias = static_cast<uint8_t>(inputParams_.hasBias ? 1 : 0);

    OP_CHECK_IF(DeterministicTilingProcess() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "DeterministicTilingProcess failed."), return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "%ld", LogQuantParams());
    return ge::GRAPH_SUCCESS;
}

int64_t GroupedMatmulFinalizeRoutingQuantTiling::LogQuantParams() const
{
    std::ostringstream oss;
    oss << "GMMQuantParams: groupNum = " << tilingData_.gmmFinalizeRoutingDataParams.groupNum
        << ", groupListType = " << static_cast<uint32_t>(tilingData_.gmmFinalizeRoutingDataParams.groupListType)
        << ", batch = " << tilingData_.gmmFinalizeRoutingDataParams.batch
        << ", sharedInputOffset = " << tilingData_.gmmFinalizeRoutingDataParams.sharedInputOffset
        << ", sharedInputLen = " << tilingData_.gmmFinalizeRoutingDataParams.sharedInputLen
        << ", residualScale = " << tilingData_.gmmFinalizeRoutingDataParams.residualScale
        << ", aQuantMode = " << tilingData_.gmmFinalizeRoutingDataParams.aQuantMode
        << ", bQuantMode = " << tilingData_.gmmFinalizeRoutingDataParams.bQuantMode
        << ", hasBias = " << static_cast<uint32_t>(tilingData_.gmmFinalizeRoutingDataParams.hasBias);
    OP_LOGD(context_->GetNodeName(), "%s", oss.str().c_str());
    return 0;
}

uint64_t GroupedMatmulFinalizeRoutingQuantTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB),
                              static_cast<uint64_t>(scaleType_), static_cast<uint64_t>(rowIndexType_));
}

ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::DoLibApiTiling()
{
    CalBasicBlock();
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"),
                return ge::GRAPH_FAILED);
    tilingData_.matmulTiling.M = inputParams_.mSize;
    tilingData_.matmulTiling.N = inputParams_.nSize;
    tilingData_.matmulTiling.Ka = inputParams_.kSize;
    tilingData_.matmulTiling.Kb = inputParams_.kSize;
    tilingData_.matmulTiling.usedCoreNum = aicoreParams_.aicNum;
    tilingData_.matmulTiling.baseM = basicTiling_.baseM;
    tilingData_.matmulTiling.baseN = basicTiling_.baseN;
    tilingData_.matmulTiling.baseK = basicTiling_.baseK;
    tilingData_.matmulTiling.singleCoreM = basicTiling_.singleCoreM;
    tilingData_.matmulTiling.singleCoreN = basicTiling_.singleCoreN;
    tilingData_.matmulTiling.singleCoreK = basicTiling_.singleCoreK;
    tilingData_.matmulTiling.depthA1 = basicTiling_.depthA1;
    tilingData_.matmulTiling.depthB1 = basicTiling_.depthB1;
    tilingData_.matmulTiling.stepM = basicTiling_.stepM;
    tilingData_.matmulTiling.stepN = basicTiling_.stepN;
    tilingData_.matmulTiling.stepKa = basicTiling_.stepKa;
    tilingData_.matmulTiling.stepKb = basicTiling_.stepKb;
    tilingData_.matmulTiling.isBias = inputParams_.hasBias ? 1 : 0;
    tilingData_.matmulTiling.iterateOrder = basicTiling_.iterateOrder;
    tilingData_.matmulTiling.dbL0A = 2; // db switch, 1: off, 2: on
    tilingData_.matmulTiling.dbL0B = 2; // db switch, 1: off, 2: on
    tilingData_.matmulTiling.dbL0C = basicTiling_.dbL0c;
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        if (basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX &&
            basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX) {
            tilingData_.matmulTiling.mxTypePara =
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) + basicTiling_.scaleFactorA;
        } else {
            tilingData_.matmulTiling.mxTypePara =
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) + SCALER_FACTOR_DEFAULT;
        }
    }
    OP_LOGD(context_->GetNodeName(), "%ld", LogMatmulParams());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::PostTiling()
{
    auto tilingDataSize = sizeof(GMMFinalizeRoutingTilingData);
    context_->SetBlockDim(aicoreParams_.aicNum);
    context_->SetScheduleMode(1);
    OP_CHECK_IF(tilingDataSize % sizeof(uint64_t) != 0,
                OP_LOGE(context_->GetNodeName(), "Tiling data size[%zu] is not aligned to 8", tilingDataSize),
                return ge::GRAPH_FAILED);
    error_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(inputParams_.opName, "Memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    return ge::GRAPH_SUCCESS;
}

int64_t GroupedMatmulFinalizeRoutingQuantTiling::LogMatmulParams() const
{
    std::ostringstream oss;
    oss << "GMM matmul tiling: M = " << tilingData_.matmulTiling.M << ", N = " << tilingData_.matmulTiling.N
        << ", Ka = " << tilingData_.matmulTiling.Ka << ", Kb = " << tilingData_.matmulTiling.Kb
        << ", usedCoreNum = " << tilingData_.matmulTiling.usedCoreNum << ", baseM = " << tilingData_.matmulTiling.baseM
        << ", baseN = " << tilingData_.matmulTiling.baseN << ", baseK = " << tilingData_.matmulTiling.baseK;
    OP_LOGD(context_->GetNodeName(), "%s", oss.str().c_str());
    return 0;
}

ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::DeterministicTilingProcess()
{
    if (context_->GetDeterministic() == 0 || inputParams_.aDtype != ge::DT_INT8 || inputParams_.bDtype != ge::DT_INT8) {
        tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag = 0;
        if (context_->GetDeterministic() != 0) {
            std::ostringstream oss;
            oss << "DeterministicTilingProcess: deterministic tiling is enabled, but only int8 "
                   "quantization is supported, current aDtype="
                << ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype)
                << ", bDtype=" << ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype)
                << ", set deterministicFlag to 0";
            OP_LOGD(context_->GetNodeName(), "%s", oss.str().c_str());
        }
        return ge::GRAPH_SUCCESS;
    }
    tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag = 1;
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t l2Size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
    deterWorkspaceSize_ = static_cast<uint32_t>((l2Size - SYS_WORKSPACE_SIZE) * DETER_WORKSPACE_RATIO);
    OP_CHECK_IF(
        static_cast<int64_t>(deterWorkspaceSize_) <
            static_cast<int64_t>(inputParams_.nSize) * static_cast<int64_t>(sizeof(float)),
        OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(),
                                  "deterministic workspace size(%u) is less than one row(N=%ld * sizeof(float)), "
                                  "cannot enable deterministic mode",
                                  deterWorkspaceSize_, static_cast<int64_t>(inputParams_.nSize)),
        return ge::GRAPH_FAILED);
    tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize = deterWorkspaceSize_;
    OP_LOGD(context_->GetNodeName(), "DeterministicTilingProcess: flag=%u, wsSize=%u",
            tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag,
            tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::GetWorkspaceSize()
{
    auto ret = GroupedQmmTiling::GetWorkspaceSize();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    if (tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag == 1) {
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
        workspaces[0] = SYS_WORKSPACE_SIZE + deterWorkspaceSize_;
        OP_LOGD(context_->GetNodeName(), "GetWorkspaceSize: deterministic workspace %u bytes, total=%zu bytes",
                deterWorkspaceSize_, workspaces[0]);
    }
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulFinalizeRouting, GroupedMatmulFinalizeRoutingQuantTiling, 1);
} // namespace optiling
