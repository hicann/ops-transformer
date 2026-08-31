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
 * \file grouped_matmul_swiglu_quant_v2_basic_tiling.cpp
 * \brief
 */

#include <alog_pub.h>
#include <climits>
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"
#include "register/op_impl_registry.h"
#include "grouped_matmul_swiglu_quant_v2_basic_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h"
using namespace Ops::Transformer::OpTiling;
using namespace GroupedMatmulSwigluQuantParamsV2;
using namespace optiling::GmmConstant;
namespace optiling {
namespace {
constexpr int64_t INVALID_MXFP4_WEIGHT_DIM = 1;
constexpr size_t MIN_X_ORIGIN_SHAPE_DIM = 2;
constexpr size_t MIN_WEIGHT_ORIGIN_SHAPE_DIM = 2;
constexpr size_t MX_MULTI_WEIGHT_SCALE_DIM = 3;
constexpr size_t PERTOKEN_WEIGHT_ORIGIN_DIM = 3;
constexpr size_t PERTOKEN_WEIGHT_NZ_STORAGE_DIM = 5;
constexpr int64_t NZ_INNER_SIZE = 16;
constexpr int64_t B8_NZ_C0_SIZE = 32;
constexpr uint64_t MXFP_BASEK_FACTOR = 64UL;
constexpr size_t WEIGHT_ORIGIN_LAST_DIM_OFFSET = 1;
constexpr size_t WEIGHT_ORIGIN_LAST_SECOND_DIM_OFFSET = 2;
constexpr size_t MXFP4_ND_N_ALIGN = 4; // MXFP4、ND场景下，N需要4对齐

size_t GetDynamicInputCount(gert::TilingContext *context, size_t inputIndex)
{
    size_t count = 0;
    while (context->GetDynamicInputShape(inputIndex, count) != nullptr) {
        ++count;
    }
    return count;
}
} // namespace

void GroupedMatmulSwigluQuantV2Tiling950::Reset()
{
    tilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
    isMxWeightNzMultiTensor_ = false;
    return;
}

ge::graphStatus GroupedMatmulSwigluQuantV2Tiling950::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    return GroupedQmmTiling::GetShapeAttrsInfo();
}

bool GroupedMatmulSwigluQuantV2Tiling950::AnalyzeAttrsPertoken()
{
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE); // 通路保证非负数
        inputParams_.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams_.groupListType;
        OP_CHECK_IF(!(inputParams_.groupListType == 0 || inputParams_.groupListType == 1),
                    OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "groupListType",
                                              std::to_string(inputParams_.groupListType), "0 or 1"),
                    return false);
    }
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "attrs", "nullptr", "attrs cannot be nullptr"),
        return false);
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
    inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    const int64_t *dequantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_MODE);
    OP_CHECK_IF(dequantModePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "dequant_mode", "nullptr",
                                                      "dequantModePtr cannot be nullptr"),
                return false);
    const int64_t *quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE);
    OP_CHECK_IF(quantModePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "quant_mode", "nullptr",
                                                      "quantModePtr cannot be nullptr"),
                return false);
    const int64_t *dequantDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_DTYPE);
    OP_CHECK_IF(dequantDtypePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "dequant_dtype", "nullptr",
                                                      "dequantDtypePtr cannot be nullptr"),
                return false);
    ge::DataType dequantDtype = static_cast<ge::DataType>(*dequantDtypePtr);
    const int64_t *quantDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_DTYPE);
    OP_CHECK_IF(quantDtypePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "quant_dtype", "nullptr",
                                                      "quantDtypePtr cannot be nullptr"),
                return false);
    ge::DataType quantDtype = static_cast<ge::DataType>(*quantDtypePtr);
    // gmm quant tiling need groupType to calculate L1 tiling
    inputParams_.groupType = SPLIT_M;
    return true;
}

bool GroupedMatmulSwigluQuantV2Tiling950::AnalyzeAttrs()
{
    if (inputParams_.aQuantMode == optiling::QuantMode::PERTOKEN_MODE) {
        return AnalyzeAttrsPertoken();
    }
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE); // 通路保证非负数
        inputParams_.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams_.groupListType;
        OP_CHECK_IF(!(inputParams_.groupListType == 0 || inputParams_.groupListType == 1),
                    OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "groupListType",
                                              std::to_string(inputParams_.groupListType), "0 or 1"),
                    return false);
    }
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "attrs", "nullptr", "attrs cannot be nullptr"),
        return false);
    return ValidateAttrsCommon();
}

bool GroupedMatmulSwigluQuantV2Tiling950::ValidateAttrsCommon()
{
    auto attrs = context_->GetAttrs();
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
    inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    const int64_t *dequantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_MODE);
    OP_CHECK_IF(dequantModePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "dequant_mode", "nullptr",
                                                      "dequantModePtr cannot be nullptr"),
                return false);
    OP_CHECK_IF(*dequantModePtr != MXQuantMode,
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "dequant_mode", std::to_string(*dequantModePtr), "2"),
                return false);
    const int64_t *quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE);
    OP_CHECK_IF(quantModePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "quant_mode", "nullptr",
                                                      "quantModePtr cannot be nullptr"),
                return false);
    OP_CHECK_IF(*quantModePtr != MXQuantMode,
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "quant_mode", std::to_string(*quantModePtr), "2"),
                return false);
    const int64_t *dequantDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_DTYPE);
    OP_CHECK_IF(dequantDtypePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "dequant_dtype", "nullptr",
                                                      "dequantDtypePtr cannot be nullptr"),
                return false);
    ge::DataType dequantDtype = static_cast<ge::DataType>(*dequantDtypePtr);
    OP_CHECK_IF(dequantDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "dequant_dtype",
                                          ge::TypeUtils::DataTypeToSerialString(dequantDtype), "DT_FLOAT"),
                return false);
    const int64_t *quantDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_DTYPE);
    OP_CHECK_IF(quantDtypePtr == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "quant_dtype", "nullptr",
                                                      "quantDtypePtr cannot be nullptr"),
                return false);
    // gmm quant tiling need groupType to calculate L1 tiling
    inputParams_.groupType = SPLIT_M;
    return true;
}

bool GroupedMatmulSwigluQuantV2Tiling950::LoadDescsAndDtypes()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "x", "nullptr", "xDesc cannot be nullptr"),
                return false);
    inputParams_.aDtype = xDesc->GetDataType();
    auto wDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_IF(
        wDesc == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "weight", "nullptr", "wDesc cannot be nullptr"),
        return false);
    inputParams_.bDtype = wDesc->GetDataType();
    auto scaleDesc = context_->GetDynamicInputDesc(SCALE_INDEX, 0);
    OP_CHECK_IF(
        scaleDesc == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "scale", "nullptr", "scaleDesc cannot be nullptr"),
        return false);
    inputParams_.scaleDtype = scaleDesc->GetDataType();
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    auto outDesc = context_->GetOutputDesc(Y_DATA_INDEX);
    OP_CHECK_IF(outDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "y", "nullptr", "outDesc cannot be nullptr"),
                return false);
    inputParams_.outDataDtype = outDesc->GetDataType();
    auto outScaleDesc = context_->GetOutputDesc(Y_SCALE_INDEX);
    OP_CHECK_IF(outScaleDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "out_scale", "nullptr",
                                                      "outScaleDesc cannot be nullptr"),
                return false);
    inputParams_.outScaleDtype = outScaleDesc->GetDataType();
    return true;
}

bool GroupedMatmulSwigluQuantV2Tiling950::CheckWeightNzDtype(const gert::Shape &xShape, const gert::Shape &wShape,
                                                             ge::Format weightFormat)
{
    OP_CHECK_IF(!((inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN && inputParams_.bDtype == ge::DT_FLOAT8_E4M3FN) ||
                  ((inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2) &&
                   (inputParams_.bDtype == ge::DT_FLOAT4_E2M1 || inputParams_.bDtype == ge::DT_FLOAT4_E1M2))),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    inputParams_.opType, "x, weight",
                    ListToString(ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype),
                                 ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype)),
                    "when the format of weight is FRACTAL_NZ, the dtypes of x and weight must be both "
                    "DT_FLOAT8_E4M3FN or FLOAT4"),
                return false);
    if (IsMxFp4WeightNz() && !CheckMxFp4WeightNzShape(xShape, wShape)) {
        return false;
    }
    return true;
}

bool GroupedMatmulSwigluQuantV2Tiling950::AnalyzeDtype()
{
    OP_CHECK_IF(!LoadDescsAndDtypes(), OP_LOGE(inputParams_.opName, "LoadDescsAndDtypes failed."), return false);
    auto x1ScaleStorageShape = context_->GetInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(x1ScaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "per_token_scale", "nullptr",
                                                      "xScaleStorageShape cannot be nullptr"),
                return false);
    const gert::Shape &xScaleShape = x1ScaleStorageShape->GetOriginShape();
    auto scaleStorageShape = context_->GetDynamicInputShape(SCALE_INDEX, 0);
    OP_CHECK_IF(scaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "scale", "nullptr",
                                                      "scaleStorageShape cannot be nullptr"),
                return false);
    const gert::Shape &wScaleShape = scaleStorageShape->GetStorageShape();
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_IF(
        xStorageShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "x", "nullptr", "xStorageShape cannot be nullptr"),
        return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
    auto wStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_IF(wStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "weight", "nullptr",
                                                      "wStorageShape cannot be nullptr"),
                return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();
    return ValidateDtypeAndQuantParams(xShape, wShape, wScaleShape, xScaleShape);
}

bool GroupedMatmulSwigluQuantV2Tiling950::ValidateDtypeAndQuantParams(const gert::Shape &xShape,
                                                                      const gert::Shape &wShape,
                                                                      const gert::Shape &wScaleShape,
                                                                      const gert::Shape &xScaleShape)
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "attrs", "nullptr", "attrs cannot be nullptr"),
        return false);
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
    inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);
    OP_CHECK_IF(!SetQuantModeForGMMSwigluQuant(wScaleShape, xScaleShape),
                OP_LOGE(inputParams_.opName, "SetQuantModeForGMMSwigluQuant failed."), return false);
    auto wDesc = context_->GetInputDesc(WEIGHT_INDEX);
    auto weightFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetFormat().GetStorageFormat()));
    inputParams_.bFormat = weightFormat;
    if (inputParams_.aQuantMode == optiling::QuantMode::PERTOKEN_MODE) {
        return CheckDtypePertoken();
    }
    if (weightFormat == ge::FORMAT_FRACTAL_NZ) {
        OP_CHECK_IF(!CheckWeightNzDtype(xShape, wShape, weightFormat),
                    OP_LOGE(inputParams_.opName, "CheckWeightNzDtype failed."), return false);
    }
    OP_CHECK_IF(!CheckWeightNdDtype(), OP_LOGE(context_->GetNodeName(), "CheckWeightNdDtype failed."), return false);
    const int64_t *quantDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_DTYPE);
    if (quantDtypePtr != nullptr) {
        ge::DataType quantDtype = static_cast<ge::DataType>(*quantDtypePtr);
        OP_CHECK_IF(!CheckQuantDtypeByFormat(quantDtype, weightFormat),
                    OP_LOGE(context_->GetNodeName(), "CheckQuantDtypeByFormat failed."), return false);
    }
    return CheckDtype();
}

bool GroupedMatmulSwigluQuantV2Tiling950::CheckQuantDtypeByFormat(ge::DataType quantDtype, ge::Format weightFormat)
{
    if (IsMxFp4WeightNz()) {
        // 仅 NZ+MXFP4 场景：支持 FLOAT4_E1M2
        OP_CHECK_IF(std::find(quantDtypeMxFp4NzSupportList.begin(), quantDtypeMxFp4NzSupportList.end(), quantDtype) ==
                        quantDtypeMxFp4NzSupportList.end(),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opType, "quant_dtype", ge::TypeUtils::DataTypeToSerialString(quantDtype),
                        "when x and weight are FLOAT4 and weight is NZ, quant_dtype must be in "
                        "{FLOAT8_E4M3, FLOAT4_E2M1, FLOAT4_E1M2}"),
                    return false);
    } else {
        // 其他所有场景（ND/FP8等）：不支持 FLOAT4_E1M2
        OP_CHECK_IF(std::find(quantDtypeSupportList.begin(), quantDtypeSupportList.end(), quantDtype) ==
                        quantDtypeSupportList.end(),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opType, "quant_dtype", ge::TypeUtils::DataTypeToSerialString(quantDtype),
                        "quant_dtype must be in {FLOAT8_E4M3, FLOAT8_E5M2, FLOAT4_E2M1}"),
                    return false);
    }
    return true;
}

bool GroupedMatmulSwigluQuantV2Tiling950::IsFp4(ge::DataType dtype) const
{
    return dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2;
}

bool GroupedMatmulSwigluQuantV2Tiling950::IsFp8(ge::DataType dtype) const
{
    return dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2;
}

bool GroupedMatmulSwigluQuantV2Tiling950::IsFp4Input() const
{
    return IsFp4(inputParams_.aDtype) && IsFp4(inputParams_.bDtype);
}

bool GroupedMatmulSwigluQuantV2Tiling950::IsMxFp4WeightNz() const
{
    return IsFp4Input() && inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ;
}

bool GroupedMatmulSwigluQuantV2Tiling950::IsMxFp8WeightNz() const
{
    return IsMXFp8Input() && inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ;
}

bool GroupedMatmulSwigluQuantV2Tiling950::IsMxWeightNzMultiTensor(const gert::Shape &wShape) const
{
    return wShape.GetDimNum() == MIN_WEIGHT_ORIGIN_SHAPE_DIM && inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ;
}

bool GroupedMatmulSwigluQuantV2Tiling950::CheckMxFp4WeightNzShape(const gert::Shape &xShape,
                                                                  const gert::Shape &wShape) const
{
    OP_CHECK_IF(xShape.GetDimNum() < MIN_X_ORIGIN_SHAPE_DIM || wShape.GetDimNum() < MIN_WEIGHT_ORIGIN_SHAPE_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    inputParams_.opType, "x, weight",
                    ListToString(std::to_string(xShape.GetDimNum()), std::to_string(wShape.GetDimNum())),
                    "when x and weight are FLOAT4 and weight is NZ, dim num of x must be at least 2 and dim num of "
                    "weight must be at least 3"),
                return false);

    int64_t weightLastSecondDim = wShape.GetDim(wShape.GetDimNum() - WEIGHT_ORIGIN_LAST_SECOND_DIM_OFFSET);
    int64_t weightLastDim = wShape.GetDim(wShape.GetDimNum() - WEIGHT_ORIGIN_LAST_DIM_OFFSET);
    OP_CHECK_IF(weightLastSecondDim == INVALID_MXFP4_WEIGHT_DIM || weightLastDim == INVALID_MXFP4_WEIGHT_DIM,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    inputParams_.opType, "weight", ShapeToString(wShape),
                    "when x and weight are FLOAT4 and weight is NZ, the last two dimensions of weight can not be 1"),
                return false);
    return true;
}

bool GroupedMatmulSwigluQuantV2Tiling950::CheckWeightNdDtype()
{
    OP_CHECK_IF(inputParams_.bFormat == ge::FORMAT_ND && inputParams_.bDtype == ge::DT_FLOAT4_E1M2,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    inputParams_.opType, "weight", ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype),
                    "when the format of weight is ND, the dtype of weight can not be FLOAT4_E1M2"),
                return false);
    return true;
}

bool GroupedMatmulSwigluQuantV2Tiling950::IsFp8Input()
{
    return IsFp8(inputParams_.aDtype) && IsFp8(inputParams_.bDtype);
}

bool GroupedMatmulSwigluQuantV2Tiling950::IsMXFp8Input() const
{
    return IsFp8(inputParams_.aDtype) && IsFp8(inputParams_.bDtype) && inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0;
}

bool GroupedMatmulSwigluQuantV2Tiling950::CheckDtype()
{
    // 校验x和weight数据类型一致性：不能一个是fp4，一个是fp8
    bool xIsFp4 = IsFp4(inputParams_.aDtype);
    bool xIsFp8 = IsFp8(inputParams_.aDtype);
    bool weightIsFp4 = IsFp4(inputParams_.bDtype);
    bool weightIsFp8 = IsFp8(inputParams_.bDtype);

    OP_CHECK_IF(
        (xIsFp4 && weightIsFp8) || (xIsFp8 && weightIsFp4),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(inputParams_.opType, "x, weight",
                                               ListToString(ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype),
                                                            ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype)),
                                               "the dtypes of x and weight must both be FLOAT8 or FLOAT4"),
        return false);
    OP_CHECK_IF(
        !(IsFp4Input() || IsFp8Input()),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(inputParams_.opType, "x, weight",
                                               ListToString(ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype),
                                                            ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype)),
                                               "the dtypes of x and weight must be within the range FLOAT8 or FLOAT4"),
        return false);
    OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0 || inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    inputParams_.opType, "scale, per_token_scale",
                    ListToString(ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype),
                                 ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype)),
                    "the dtypes of scale and per_token_scale must be DT_FLOAT8_E8M0"),
                return false);
    OP_CHECK_IF(
        !(IsFp4(inputParams_.outDataDtype) || IsFp8(inputParams_.outDataDtype)),
        OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opType, "y",
                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.outDataDtype), "FLOAT8 or FLOAT4"),
        return false);
    OP_CHECK_IF(
        inputParams_.outScaleDtype != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opType, "out_scale",
                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.outScaleDtype), "DT_FLOAT8_E8M0"),
        return false);

    OP_CHECK_IF(IsFp8Input() && !IsFp8(inputParams_.outDataDtype),
                OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opType, "y",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.outDataDtype), "FLOAT8"),
                return false);
    return true;
}

bool GroupedMatmulSwigluQuantV2Tiling950::SetQuantModeForGMMSwigluQuant(const gert::Shape &wScaleShape,
                                                                        const gert::Shape &xScaleShape)
{
    auto wScaleDims = wScaleShape.GetDimNum();
    if (IsMicroScaling()) {
        inputParams_.bQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        inputParams_.aQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        return true;
    }
    if (wScaleDims == PRECHANNEL_WEIGHT_SCALE_DIM &&
        static_cast<uint64_t>(wScaleShape.GetDim(wScaleDims - 1)) == inputParams_.nSize && inputParams_.nSize != 1UL) {
        inputParams_.bQuantMode = optiling::QuantMode::PERCHANNEL_MODE;
    }
    auto xScaleDims = xScaleShape.GetDimNum();
    if (xScaleDims == 1 && xScaleShape[0] == inputParams_.mSize) {
        inputParams_.aQuantMode = optiling::QuantMode::PERTOKEN_MODE;
    }
    if (inputParams_.bQuantMode == optiling::QuantMode::PERCHANNEL_MODE &&
        inputParams_.aQuantMode == optiling::QuantMode::PERTOKEN_MODE) {
        return true;
    }
    return false;
}

bool GroupedMatmulSwigluQuantV2Tiling950::CheckDims(const gert::Shape &xShape, const gert::Shape &wShape) const
{
    auto aInnerSize = inputParams_.transA ? inputParams_.mSize : inputParams_.kSize;
    auto bInnerSize = inputParams_.transB ? inputParams_.kSize : inputParams_.nSize;
    OP_CHECK_IF(IsFp4Input() && (aInnerSize % B4_DATACOPY_MIN_NUM != 0 || bInnerSize % B4_DATACOPY_MIN_NUM != 0),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    inputParams_.opType, "x, weight", ShapesToString({ShapeToString(xShape), ShapeToString(wShape)}),
                    "when inputs are FLOAT4, inner axis element number must be even"),
                return false);

    // MXFP4场景不支持K=2
    OP_CHECK_IF(IsFp4Input() && inputParams_.kSize == MXFP4_K_MIN_VALUE,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    inputParams_.opType, "x, weight", ShapesToString({ShapeToString(xShape), ShapeToString(wShape)}),
                    "when the dtypes of x and weight are FLOAT4, k value must be greater than 2"),
                return false);
    // MXFP4场景下，当输出类型为FP4时，N需要满足为大于等于4的偶数
    if (IsFp4Input() && IsFp4(inputParams_.outDataDtype)) {
        OP_CHECK_IF(inputParams_.nSize < MXFP4_N_MIN_VALUE || inputParams_.nSize % EVEN_FACTOR != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opType, "weight", ShapeToString(wShape),
                        "when inputs and output are FLOAT4, n value must be even and greater or equal to 4"),
                    return false);
    }
    // MXFP4、NZ场景下，N需满足128对齐。
    if (IsMxFp4WeightNz()) {
        OP_CHECK_IF(
            inputParams_.nSize % GmmConstant::BASIC_BLOCK_SIZE_128 != 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opType, "weight", ShapeToString(wShape),
                                                  "when using the weightNZ format with FP4 data type, n axis "
                                                  "element number of weight must be an integer multiple of 128"),
            return false);
    }
    // MXFP8、NZ场景下，N需满足64对齐。
    if (IsMxFp8WeightNz()) {
        OP_CHECK_IF(inputParams_.nSize % GmmConstant::WEIGHTNZ_64 != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opType, "weight", ShapeToString(wShape),
                                                          "when using the weightNZ format with FP8 data type, n axis "
                                                          "element number of weight must be an integer multiple of 64"),
                    return false);
    }
    // MXFP4、ND场景下，N需满足4对齐。
    if (IsFp4Input()) {
        OP_CHECK_IF(inputParams_.nSize % MXFP4_ND_N_ALIGN != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opType, "weight", ShapeToString(wShape),
                                                          "when using the ND format with FP4 data type, n axis element "
                                                          "number of weight must be an integer multiple of 4"),
                    return false);
    }
    // MXFP8、ND场景下，N需满足2对齐。
    if (IsMXFp8Input()) {
        OP_CHECK_IF(inputParams_.nSize % EVEN_FACTOR != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opType, "weight", ShapeToString(wShape),
                                                          "when using the ND format with FP8 data type, n axis element "
                                                          "number of weight must be an integer multiple of 2"),
                    return false);
    }

    return true;
}
bool GroupedMatmulSwigluQuantV2Tiling950::AnalyzeInputs()
{
    OP_CHECK_IF(!CheckCoreNum(), OP_LOGE(inputParams_.opName, "CheckCoreNum failed."), return false);
    if (inputParams_.aQuantMode == optiling::QuantMode::PERTOKEN_MODE) {
        return AnalyzeInputsPertoken();
    }
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_IF(
        xStorageShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "x", "nullptr", "xStorageShape cannot be nullptr"),
        return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
    auto wStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_IF(wStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "weight", "nullptr",
                                                      "wStorageShape cannot be nullptr"),
                return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();
    auto scaleStorageShape = context_->GetDynamicInputShape(SCALE_INDEX, 0);
    OP_CHECK_IF(scaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "scale", "nullptr",
                                                      "scaleStorageShape cannot be nullptr"),
                return false);
    const gert::Shape &wScaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = wScaleShape.GetDimNum();
    size_t weightCount = GetDynamicInputCount(context_, WEIGHT_INDEX);
    size_t scaleCount = GetDynamicInputCount(context_, SCALE_INDEX);
    bool isMultiWeightNz = IsMxWeightNzMultiTensor(wShape);
    isMxWeightNzMultiTensor_ = isMultiWeightNz;
    size_t expectedScaleDim = isMultiWeightNz ? MX_MULTI_WEIGHT_SCALE_DIM : MX_WEIGHT_SCALE_DIM;
    OP_CHECK_IF(scaleDimNum != expectedScaleDim,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "weight_scale", std::to_string(scaleDimNum),
                                             std::to_string(expectedScaleDim)),
                return false);
    auto x1ScaleStorageShape = context_->GetInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(x1ScaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "x_scale", "nullptr",
                                                      "xScaleStorageShape cannot be nullptr"),
                return false);
    const gert::Shape &xScaleShape = x1ScaleStorageShape->GetOriginShape();
    auto xScaleDimNum = xScaleShape.GetDimNum();
    OP_CHECK_IF(xScaleDimNum != MX_X_SCALE_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "x_scale", std::to_string(xScaleDimNum), "3"),
                return false);
    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."), return false);
    OP_CHECK_IF(isMultiWeightNz && weightCount != static_cast<size_t>(inputParams_.groupNum),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "weight", std::to_string(weightCount),
                                                      "weight tensor list size must equal groupList length"),
                return false);
    OP_CHECK_IF(isMultiWeightNz && scaleCount != weightCount,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "weight_scale", std::to_string(scaleCount),
                                                      "weightScale tensor list size must equal "
                                                      "weight tensor list size"),
                return false);
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);
    OP_CHECK_IF(!CheckDims(xShape, wShape), OP_LOGE(inputParams_.opName, "CheckDims failed."), return false);
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        if (isMultiWeightNz) {
            auto expectedKDimValue = GroupedMatmul::CeilDiv(inputParams_.kSize, MXFP_BASEK_FACTOR);
            uint64_t expectedWeightDim0 = inputParams_.transB ? inputParams_.nSize : inputParams_.kSize;
            uint64_t expectedWeightDim1 = inputParams_.transB ? inputParams_.kSize : inputParams_.nSize;
            uint64_t expectedScaleDim0 = inputParams_.transB ? inputParams_.nSize : expectedKDimValue;
            uint64_t expectedScaleDim1 = inputParams_.transB ? expectedKDimValue : inputParams_.nSize;
            for (size_t i = 0; i < weightCount; ++i) {
                auto curWStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, i);
                auto curScaleStorageShape = context_->GetDynamicInputShape(SCALE_INDEX, i);
                OP_CHECK_IF(curWStorageShape == nullptr || curScaleStorageShape == nullptr,
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "weight/weight_scale", "nullptr",
                                                                  "dynamic weight and weightScale cannot be nullptr"),
                            return false);
                const gert::Shape &curWShape = curWStorageShape->GetOriginShape();
                const gert::Shape &curScaleShape = curScaleStorageShape->GetOriginShape();
                OP_CHECK_IF(curWShape.GetDimNum() != MIN_WEIGHT_ORIGIN_SHAPE_DIM ||
                                static_cast<uint64_t>(curWShape.GetDim(0)) != expectedWeightDim0 ||
                                static_cast<uint64_t>(curWShape.GetDim(1)) != expectedWeightDim1,
                            OP_LOGE_FOR_INVALID_SHAPE(inputParams_.opType, "weight", ShapeToString(curWShape),
                                                      ShapeDimsToString(expectedWeightDim0, expectedWeightDim1)),
                            return false);
                OP_CHECK_IF(curScaleShape.GetDimNum() != MX_MULTI_WEIGHT_SCALE_DIM ||
                                static_cast<uint64_t>(curScaleShape.GetDim(0)) != expectedScaleDim0 ||
                                static_cast<uint64_t>(curScaleShape.GetDim(1)) != expectedScaleDim1 ||
                                static_cast<uint64_t>(curScaleShape.GetDim(2)) != MXFP_MULTI_BASE_SIZE,
                            OP_LOGE_FOR_INVALID_SHAPE(
                                inputParams_.opType, "weight_scale", ShapeToString(curScaleShape),
                                ShapeDimsToString(expectedScaleDim0, expectedScaleDim1, MXFP_MULTI_BASE_SIZE)),
                            return false);
            }
            OP_CHECK_IF(static_cast<uint64_t>(xScaleShape.GetDim(0)) != inputParams_.mSize ||
                            static_cast<uint64_t>(xScaleShape.GetDim(1)) != expectedKDimValue ||
                            static_cast<uint64_t>(xScaleShape.GetDim(2)) != MXFP_MULTI_BASE_SIZE,
                        OP_LOGE_FOR_INVALID_SHAPE(
                            inputParams_.opType, "x_scale", ShapeToString(xScaleShape),
                            ShapeDimsToString(inputParams_.mSize, expectedKDimValue, MXFP_MULTI_BASE_SIZE)),
                        return false);
        } else {
            OP_CHECK_IF(!CheckQuantParamsForMXTypeM(xScaleShape, wScaleShape),
                        OP_LOGE(inputParams_.opName, "CheckQuantParamsForMXTypeM failed."), return false);
        }
    }
    return true;
}

bool GroupedMatmulSwigluQuantV2Tiling950::CheckCoreNum() const
{
    auto compileInfo = context_->GetCompileInfo<GMMSwigluV2CompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(inputParams_.opName, "compileInfo is nullptr."), return false);
    auto aicNum = compileInfo->aicNum_;
    auto aivNum = compileInfo->aivNum_;
    OP_CHECK_IF(aicNum == 0, OP_LOGE(inputParams_.opName, "aicNum should be positive integer, actual is %u.", aicNum),
                return false);
    OP_CHECK_IF(
        aivNum != GmmConstant::CORE_RATIO * aicNum,
        OP_LOGE(inputParams_.opName, "aicNum:aivNum should be 1:2, actual aicNum: %u, aivNum: %u.", aicNum, aivNum),
        return false);
    return true;
}

ge::graphStatus GroupedMatmulSwigluQuantV2Tiling950::DoOpTiling()
{
    tilingData_.gmmSwigluQuantParams.set_groupNum(inputParams_.groupNum);
    tilingData_.gmmSwigluQuantParams.set_groupListType(static_cast<uint8_t>(inputParams_.groupListType));
    tilingData_.gmmSwigluQuantParams.set_isMxWeightNzMultiTensor(static_cast<uint8_t>(isMxWeightNzMultiTensor_));
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const int64_t *dequantDtypeTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_DTYPE);
        int64_t dequantDtype = dequantDtypeTypePtr != nullptr ? static_cast<int64_t>(*dequantDtypeTypePtr) : 0L;
        if (inputParams_.aQuantMode == optiling::QuantMode::PERTOKEN_MODE) {
            tilingData_.gmmSwigluQuantParams.set_dequantDtype(static_cast<uint8_t>(dequantDtype));
        }
        const int64_t *quantDtypeTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_DTYPE);
        int64_t quantDtype = quantDtypeTypePtr != nullptr ? static_cast<int64_t>(*quantDtypeTypePtr) : 0L;
        tilingData_.gmmSwigluQuantParams.set_quantDtype(static_cast<uint8_t>(quantDtype));
    }
    if (inputParams_.aQuantMode == optiling::QuantMode::PERTOKEN_MODE) {
        return DoOpTilingPertoken();
    }
    OP_LOGD(inputParams_.opName, "%ld", LogQuantParams());
    return ge::GRAPH_SUCCESS;
}

int64_t GroupedMatmulSwigluQuantV2Tiling950::LogQuantParams()
{
    auto &params = tilingData_.gmmSwigluQuantParams;
    std::ostringstream oss;
    oss << "GMMQuantParams: groupNum = " << params.get_groupNum()
        << ", groupListType = " << static_cast<uint32_t>(params.get_groupListType())
        << ", quant_dtype = " << static_cast<int32_t>(params.get_quantDtype())
        << ", isMxWeightNzMultiTensor = " << static_cast<int32_t>(params.get_isMxWeightNzMultiTensor());
    OP_LOGD(inputParams_.opName, "%s", oss.str().c_str());
    return 0;
}

ge::graphStatus GroupedMatmulSwigluQuantV2Tiling950::DoLibApiTiling()
{
    CalBasicBlock();
    auto baseM_modified = std::min(basicTiling_.baseM, static_cast<uint64_t>(128));
    basicTiling_.baseM = GroupedMatmul::CeilAlign(baseM_modified, GmmConstant::CUBE_BLOCK);
    ge::graphStatus l1TilingStatus = IsMxFp4WeightNz() ? CalWeightNzL1Tiling() : CalL1Tiling();
    OP_CHECK_IF(l1TilingStatus != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"),
                return ge::GRAPH_FAILED);
    tilingData_.mmTilingData.set_M(inputParams_.mSize);
    tilingData_.mmTilingData.set_N(inputParams_.nSize);
    tilingData_.mmTilingData.set_Ka(inputParams_.kSize);
    tilingData_.mmTilingData.set_Kb(inputParams_.kSize);
    tilingData_.mmTilingData.set_usedCoreNum(aicoreParams_.aicNum);
    tilingData_.mmTilingData.set_baseM(basicTiling_.baseM);
    tilingData_.mmTilingData.set_baseN(basicTiling_.baseN);
    tilingData_.mmTilingData.set_baseK(basicTiling_.baseK);
    tilingData_.mmTilingData.set_singleCoreM(basicTiling_.baseM);
    tilingData_.mmTilingData.set_singleCoreN(basicTiling_.singleCoreN);
    tilingData_.mmTilingData.set_singleCoreK(basicTiling_.singleCoreK);
    tilingData_.mmTilingData.set_depthA1(basicTiling_.depthA1);
    tilingData_.mmTilingData.set_depthB1(basicTiling_.depthB1);
    tilingData_.mmTilingData.set_stepM(basicTiling_.stepM);
    tilingData_.mmTilingData.set_stepN(basicTiling_.stepN);
    tilingData_.mmTilingData.set_stepKa(basicTiling_.stepKa);
    tilingData_.mmTilingData.set_stepKb(basicTiling_.stepKb);
    tilingData_.mmTilingData.set_isBias(inputParams_.hasBias ? 1 : 0);
    tilingData_.mmTilingData.set_iterateOrder(basicTiling_.iterateOrder);
    tilingData_.mmTilingData.set_dbL0A(2); // db switch, 1: off, 2: on
    tilingData_.mmTilingData.set_dbL0B(2); // db switch, 1: off, 2: on
    tilingData_.mmTilingData.set_dbL0C(basicTiling_.dbL0c);
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        if (basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX &&
            basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX) {
            tilingData_.mmTilingData.set_mxTypePara(
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) + basicTiling_.scaleFactorA);
        } else {
            tilingData_.mmTilingData.set_mxTypePara(
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) + SCALER_FACTOR_DEFAULT);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2Tiling950::CalWeightNzL1Tiling()
{
    InitCommonL1TilingFields();
    if (inputParams_.kSize == 0) {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t leftL1Size = 0;
    OP_CHECK_IF(CalcLeftL1Size(leftL1Size) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalcLeftL1Size failed"), return ge::GRAPH_FAILED);
    return CalWeightNzL1Depth(leftL1Size);
}

ge::graphStatus GroupedMatmulSwigluQuantV2Tiling950::CalWeightNzL1Depth(uint64_t leftL1Size)
{
    uint64_t baseASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
    uint64_t baseBSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);
    uint64_t baseScaleASize = 0;
    uint64_t baseScaleBSize = 0;
    CalcAlignedMxBaseScaleSize(baseScaleASize, baseScaleBSize);
    uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
    OP_CHECK_IF(leftL1Size < baseL1Size,
                OP_LOGE(context_->GetNodeName(), "L1 space overflow. Free L1Size : %lu, used space: %lu", leftL1Size,
                        baseL1Size),
                return ge::GRAPH_FAILED);

    uint64_t depthInit = GetDepthA1B1(leftL1Size, baseL1Size, 1UL);
    basicTiling_.depthA1 = GetWeightNzDepthWithHighBW(std::min(inputParams_.mSize, basicTiling_.baseM));
    basicTiling_.depthB1 = GetWeightNzDepthWithHighBW(std::min(inputParams_.nSize, basicTiling_.baseN));
    if (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize +
            std::max(basicTiling_.depthA1, basicTiling_.depthB1) * (baseScaleASize + baseScaleBSize) >
        leftL1Size) {
        basicTiling_.depthA1 = depthInit;
        basicTiling_.depthB1 = depthInit;
    }
    ModifyWeightNzDepthForUnalign(leftL1Size, baseASize, baseBSize, baseScaleASize + baseScaleBSize);
    CalStepKs();
    return CalWeightNzScaleFactors();
}

uint64_t GroupedMatmulSwigluQuantV2Tiling950::GetWeightNzDepthWithHighBW(uint64_t mnL1) const
{
    uint64_t baseKSize = GetSizeWithDataType(basicTiling_.baseK, inputParams_.aDtype);
    uint64_t depth = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(MTE2_MIN_LOAD_SIZE_V120, mnL1),
                                              static_cast<uint64_t>(GmmConstant::BASIC_BLOCK_SIZE_256)) /
                     baseKSize * DB_SIZE;
    uint64_t pow2Depth = POWER_OF_TWO;
    while (pow2Depth < depth) {
        pow2Depth *= POWER_OF_TWO;
    }
    return std::min(pow2Depth, GroupedMatmul::CeilDiv(inputParams_.kSize, basicTiling_.baseK) * DB_SIZE);
}

void GroupedMatmulSwigluQuantV2Tiling950::ModifyWeightNzDepthForUnalign(uint64_t leftL1Size, uint64_t baseASize,
                                                                        uint64_t baseBSize, uint64_t baseScaleABSize)
{
    if (inputParams_.kSize % GmmConstant::BASIC_BLOCK_SIZE_128 == 0) {
        return;
    }
    if (inputParams_.transA && (!inputParams_.transB || inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ)) {
        return;
    }
    if (!inputParams_.transA) {
        if (basicTiling_.depthA1 <= basicTiling_.depthB1) {
            uint64_t leftASize = leftL1Size - basicTiling_.depthB1 * baseBSize - basicTiling_.depthB1 * baseScaleABSize;
            while (basicTiling_.depthA1 * POWER_OF_TWO * baseASize <= leftASize) {
                basicTiling_.depthA1 *= POWER_OF_TWO;
            }
            if (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize +
                    std::max(basicTiling_.depthA1, basicTiling_.depthB1) * baseScaleABSize >
                leftL1Size) {
                basicTiling_.depthA1 = basicTiling_.depthB1;
            }
        } else if (inputParams_.transB && inputParams_.bFormat == ge::FORMAT_ND) {
            uint64_t leftBSize = leftL1Size - basicTiling_.depthA1 * baseASize - basicTiling_.depthA1 * baseScaleABSize;
            while (basicTiling_.depthB1 * POWER_OF_TWO * baseBSize <= leftBSize) {
                basicTiling_.depthB1 *= POWER_OF_TWO;
            }
            if (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize +
                    std::max(basicTiling_.depthA1, basicTiling_.depthB1) * baseScaleABSize >
                leftL1Size) {
                basicTiling_.depthB1 = basicTiling_.depthA1;
            }
        }
    } else {
        while ((basicTiling_.depthA1 * baseASize -
                std::max(basicTiling_.depthA1, basicTiling_.depthB1 * POWER_OF_TWO) * baseScaleABSize) < leftL1Size) {
            basicTiling_.depthB1 *= POWER_OF_TWO;
        }
    }
}

ge::graphStatus GroupedMatmulSwigluQuantV2Tiling950::CalWeightNzScaleFactors()
{
    uint64_t baseASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
    uint64_t baseBSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);
    uint64_t baseScaleASize =
        GetSizeWithDataType(GroupedMatmul::CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE) * basicTiling_.baseM,
                            inputParams_.perTokenScaleDtype);
    uint64_t baseScaleBSize = GetSizeWithDataType(
        GroupedMatmul::CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE) * basicTiling_.baseN, inputParams_.scaleDtype);
    OP_CHECK_IF(baseScaleASize == 0 || baseScaleBSize == 0,
                OP_LOGE(context_->GetNodeName(),
                        "When m(%lu)/n(%lu)/k(%lu)/groupNum(%lu) in mx quant mode, baseScaleASize(%lu) and "
                        "baseScaleBSize(%lu) should not be equal to 0.",
                        inputParams_.mSize, inputParams_.nSize, inputParams_.kSize, inputParams_.groupNum,
                        baseScaleASize, baseScaleBSize),
                return ge::GRAPH_FAILED);
    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t baseBiasSize = inputParams_.hasBias ? basicTiling_.baseN * biasDtypeSize : 0;
    uint64_t leftL1Size =
        aicoreParams_.l1Size - (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize + baseBiasSize);
    uint32_t scaleInit = static_cast<uint32_t>(
        leftL1Size / (std::max(basicTiling_.depthA1, basicTiling_.depthB1) * (baseScaleASize + baseScaleBSize)));
    OP_CHECK_IF(
        scaleInit == 0,
        OP_LOGE(context_->GetNodeName(),
                "When m(%lu)/n(%lu)/k(%lu)/groupNum(%lu) in mx quant mode, scaleFactor should not be equal to 0.",
                inputParams_.mSize, inputParams_.nSize, inputParams_.kSize, inputParams_.groupNum),
        return ge::GRAPH_FAILED);

    uint32_t scaleFactorAMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V120 / baseScaleASize), SCALER_FACTOR_MAX);
    uint32_t scaleFactorBMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V120 / baseScaleBSize), SCALER_FACTOR_MAX);
    OP_CHECK_IF(scaleFactorAMax == 0 || scaleFactorBMax == 0,
                OP_LOGE(context_->GetNodeName(),
                        "When m(%lu)/n(%lu)/k(%lu)/groupNum(%lu) in mx quant mode, scaleFactorAMax(%u) and "
                        "scaleFactorBMax(%u) should not be equal to 0.",
                        inputParams_.mSize, inputParams_.nSize, inputParams_.kSize, inputParams_.groupNum,
                        scaleFactorAMax, scaleFactorBMax),
                return ge::GRAPH_FAILED);
    uint32_t scaleFactorA =
        static_cast<uint32_t>(GroupedMatmul::CeilDiv(inputParams_.kSize, basicTiling_.stepKa * basicTiling_.baseK));
    uint32_t scaleFactorB =
        static_cast<uint32_t>(GroupedMatmul::CeilDiv(inputParams_.kSize, basicTiling_.stepKb * basicTiling_.baseK));
    basicTiling_.scaleFactorA = std::max(SCALER_FACTOR_MIN, scaleFactorA);
    basicTiling_.scaleFactorB = std::max(SCALER_FACTOR_MIN, scaleFactorB);
    basicTiling_.scaleFactorA = std::min(scaleFactorAMax, basicTiling_.scaleFactorA);
    basicTiling_.scaleFactorB = std::min(scaleFactorBMax, basicTiling_.scaleFactorB);

    if (basicTiling_.scaleFactorA > scaleInit && basicTiling_.scaleFactorB > scaleInit) {
        if (basicTiling_.depthA1 >= basicTiling_.depthB1) {
            basicTiling_.scaleFactorA = scaleInit;
            basicTiling_.scaleFactorB = scaleInit * basicTiling_.depthA1 / basicTiling_.depthB1;
        } else {
            basicTiling_.scaleFactorA = scaleInit * basicTiling_.depthB1 / basicTiling_.depthA1;
            basicTiling_.scaleFactorB = scaleInit;
        }
    }
    OP_CHECK_IF(basicTiling_.scaleFactorA < SCALER_FACTOR_MIN || basicTiling_.scaleFactorA > SCALER_FACTOR_MAX ||
                    basicTiling_.scaleFactorB < SCALER_FACTOR_MIN || basicTiling_.scaleFactorB > SCALER_FACTOR_MAX,
                OP_LOGE(context_->GetNodeName(),
                        "When m(%lu)/n(%lu)/k(%lu)/groupNum(%lu) in mx quant mode, scaleFactorA(%u) and "
                        "scaleFactorB(%u) should be in range [%u, %u].",
                        inputParams_.mSize, inputParams_.nSize, inputParams_.kSize, inputParams_.groupNum,
                        basicTiling_.scaleFactorA, basicTiling_.scaleFactorB, SCALER_FACTOR_MIN, SCALER_FACTOR_MAX),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2Tiling950::PostTiling()
{
    context_->SetBlockDim(aicoreParams_.aicNum);
    context_->SetScheduleMode(1);
    OP_CHECK_IF(
        tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        OP_LOGE(context_->GetNodeName(), "Tiling data size[%zu] is not aligned to 8", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulSwigluQuantV2Tiling950::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transB), static_cast<uint64_t>(inputParams_.transA),
                              static_cast<uint64_t>(GMM_SWIGLU_QUANT_ORIGINAL_KERNEL_TYPE));
}

bool GroupedMatmulSwigluQuantV2Tiling950::IsB8(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2 || dtype == ge::DT_INT8 ||
           dtype == ge::DT_HIFLOAT8;
}

bool GroupedMatmulSwigluQuantV2Tiling950::CheckDtypePertoken()
{
    OP_CHECK_IF(!(IsB8(inputParams_.aDtype) && IsB8(inputParams_.bDtype)),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    inputParams_.opType, "x, weight",
                    ListToString(ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype),
                                 ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype)),
                    "the dtypes of x and weight must be within the range FLOAT8, INT8 or HIFLOAT8"),
                return false);
    if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        const bool isInt8 = inputParams_.aDtype == ge::DT_INT8;
        OP_CHECK_IF(isInt8 && (inputParams_.bDtype != ge::DT_INT8 || inputParams_.outDataDtype != ge::DT_INT8),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        inputParams_.opType, "x, weight, y",
                        ListToString(ListToString(ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype),
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype)),
                                     ge::TypeUtils::DataTypeToSerialString(inputParams_.outDataDtype)),
                        "in per-token mode, when weight format is FRACTAL_NZ and x dtype is INT8, "
                        "weight and y dtypes must be INT8"),
                    return false);
        const bool isHifloat8 = inputParams_.aDtype == ge::DT_HIFLOAT8;
        OP_CHECK_IF(
            isHifloat8 && (inputParams_.bDtype != ge::DT_HIFLOAT8 || inputParams_.outDataDtype != ge::DT_HIFLOAT8),
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                inputParams_.opType, "x, weight, y",
                ListToString(ListToString(ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype),
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype)),
                             ge::TypeUtils::DataTypeToSerialString(inputParams_.outDataDtype)),
                "in per-token mode, when weight format is FRACTAL_NZ and x dtype is HIFLOAT8, "
                "weight and y dtypes must be HIFLOAT8"),
            return false);
        const bool isFp8 = inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2;
        OP_CHECK_IF(isFp8 && (inputParams_.bDtype != ge::DT_FLOAT8_E4M3FN ||
                              (inputParams_.outDataDtype != ge::DT_FLOAT8_E4M3FN &&
                               inputParams_.outDataDtype != ge::DT_FLOAT8_E5M2)),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        inputParams_.opType, "x, weight, y",
                        ListToString(ListToString(ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype),
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype)),
                                     ge::TypeUtils::DataTypeToSerialString(inputParams_.outDataDtype)),
                        "in per-token mode, when weight format is FRACTAL_NZ and x dtype is FLOAT8, "
                        "weight dtype must be FLOAT8_E4M3FN and y dtype must be FLOAT8_E4M3FN or FLOAT8_E5M2"),
                    return false);
    }
    OP_CHECK_IF(
        inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opType, "x_scale",
                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype), "DT_FLOAT"),
        return false);
    OP_CHECK_IF(!(inputParams_.scaleDtype == ge::DT_FLOAT || inputParams_.scaleDtype == ge::DT_BF16 ||
                  (inputParams_.scaleDtype == ge::DT_FLOAT16 && inputParams_.aDtype == ge::DT_INT8)),
                OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opType, "weight_scale",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype),
                                          "DT_FLOAT, DT_BF16 or DT_FLOAT16 when x dtype is DT_INT8"),
                return false);
    OP_CHECK_IF(!IsB8(inputParams_.outDataDtype),
                OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opType, "y",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams_.outDataDtype),
                                          "FLOAT8, INT8 or HIFLOAT8"),
                return false);
    OP_CHECK_IF(
        inputParams_.outScaleDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opType, "out_scale",
                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.outScaleDtype), "DT_FLOAT"),
        return false);
    return true;
}

bool GroupedMatmulSwigluQuantV2Tiling950::CheckPertokenWeightNzShape(const gert::Shape &wShape,
                                                                     const gert::Shape &wStorageShape) const
{
    OP_CHECK_IF(
        wShape.GetDimNum() != PERTOKEN_WEIGHT_ORIGIN_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "weight origin shape", std::to_string(wShape.GetDimNum()),
                                     std::to_string(PERTOKEN_WEIGHT_ORIGIN_DIM)),
        return false);
    OP_CHECK_IF(wStorageShape.GetDimNum() != PERTOKEN_WEIGHT_NZ_STORAGE_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "weight storage shape",
                                             std::to_string(wStorageShape.GetDimNum()),
                                             std::to_string(PERTOKEN_WEIGHT_NZ_STORAGE_DIM)),
                return false);

    const int64_t groupNum = wShape.GetDim(0);
    const int64_t n1 = inputParams_.transB ? GroupedMatmul::CeilDiv(inputParams_.kSize, B8_NZ_C0_SIZE) :
                                             GroupedMatmul::CeilDiv(inputParams_.nSize, B8_NZ_C0_SIZE);
    const int64_t k1 = inputParams_.transB ? GroupedMatmul::CeilDiv(inputParams_.nSize, NZ_INNER_SIZE) :
                                             GroupedMatmul::CeilDiv(inputParams_.kSize, NZ_INNER_SIZE);
    const gert::Shape expectedStorageShape{groupNum, n1, k1, NZ_INNER_SIZE, B8_NZ_C0_SIZE};
    OP_CHECK_IF(wStorageShape != expectedStorageShape,
                OP_LOGE_FOR_INVALID_SHAPE(inputParams_.opType, "weight storage shape", ShapeToString(wStorageShape),
                                          ShapeToString(expectedStorageShape)),
                return false);
    return true;
}

bool GroupedMatmulSwigluQuantV2Tiling950::AnalyzeInputsPertoken()
{
    auto wStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_IF(wStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "weight", "nullptr",
                                                      "wStorageShape cannot be nullptr"),
                return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();
    const gert::Shape &wNzStorageShape = wStorageShape->GetStorageShape();
    auto scaleStorageShape = context_->GetDynamicInputShape(SCALE_INDEX, 0);
    OP_CHECK_IF(scaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "weight_scale", "nullptr",
                                                      "scaleStorageShape cannot be nullptr"),
                return false);
    const gert::Shape &wScaleShape = scaleStorageShape->GetStorageShape();
    auto scaleDimNum = wScaleShape.GetDimNum();
    OP_CHECK_IF(scaleDimNum != PRECHANNEL_WEIGHT_SCALE_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "weight_scale", std::to_string(scaleDimNum), "2"),
                return false);
    auto x1ScaleStorageShape = context_->GetInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(x1ScaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "x_scale", "nullptr",
                                                      "xScaleStorageShape cannot be nullptr"),
                return false);
    const gert::Shape &xScaleShape = x1ScaleStorageShape->GetOriginShape();
    auto xScaleDimNum = xScaleShape.GetDimNum();
    OP_CHECK_IF(xScaleDimNum != PERTOKEN_X_SCALE_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "x_scale", std::to_string(xScaleDimNum), "1"),
                return false);
    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."), return false);
    OP_CHECK_IF(wShape.GetDimNum() != PERTOKEN_WEIGHT_ORIGIN_DIM ||
                    static_cast<uint64_t>(wShape.GetDim(0)) != inputParams_.groupNum,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opType, "weight", ShapeToString(wShape),
                                                      "weight first dimension must equal groupList length"),
                return false);
    OP_CHECK_IF(static_cast<uint64_t>(wScaleShape.GetDim(0)) != inputParams_.groupNum ||
                    static_cast<uint64_t>(wScaleShape.GetDim(1)) != inputParams_.nSize,
                OP_LOGE_FOR_INVALID_SHAPE(inputParams_.opType, "weight_scale", ShapeToString(wScaleShape),
                                          ShapeDimsToString(inputParams_.groupNum, inputParams_.nSize)),
                return false);
    OP_CHECK_IF(static_cast<uint64_t>(xScaleShape.GetDim(0)) != inputParams_.mSize,
                OP_LOGE_FOR_INVALID_SHAPE(inputParams_.opType, "x_scale", ShapeToString(xScaleShape),
                                          ShapeDimsToString(inputParams_.mSize)),
                return false);
    if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        OP_CHECK_IF(inputParams_.nSize == 0 || inputParams_.nSize % GmmConstant::WEIGHTNZ_64 != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opType, "weight", ShapeToString(wShape),
                        "in per-token B8 mode, N of FRACTAL_NZ weight must be positive and aligned to 64"),
                    return false);
        OP_CHECK_IF(!CheckPertokenWeightNzShape(wShape, wNzStorageShape),
                    OP_LOGE(inputParams_.opName, "CheckPertokenWeightNzShape failed."), return false);
    }
    OP_CHECK_IF(inputParams_.nSize % GmmConstant::EVEN_FACTOR != 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opType, "weight", ShapeToString(wShape),
                                                      "n axis element number of weight must be an even number"),
                return false);
    return true;
}

ge::graphStatus GroupedMatmulSwigluQuantV2Tiling950::DoOpTilingPertoken()
{
    uint32_t rowLen = inputParams_.nSize >> 1;
    uint32_t alignedRowLen = rowLen;
    if (rowLen != 0) {
        alignedRowLen = (rowLen + GmmConstant::SCALER_FACTOR_M_BIT - 1) / GmmConstant::SCALER_FACTOR_M_BIT *
                        GmmConstant::SCALER_FACTOR_M_BIT;
    }
    uint64_t maxUseUbSize = aicoreParams_.ubSize - GmmConstant::RESERVED_LENGTH;
    uint64_t calcDbSize = static_cast<uint64_t>(alignedRowLen) * GmmConstant::DB_REQUIRED_BYTES_SIZE;
    uint32_t ubAvail = static_cast<uint32_t>(maxUseUbSize / calcDbSize);
    tilingData_.gmmSwigluQuantParams.set_rowLen(rowLen);
    tilingData_.gmmSwigluQuantParams.set_ubAvail(ubAvail);
    OP_LOGD(inputParams_.opName, "%ld", LogPertokenQuantParams());
    return ge::GRAPH_SUCCESS;
}

int64_t GroupedMatmulSwigluQuantV2Tiling950::LogPertokenQuantParams()
{
    auto &params = tilingData_.gmmSwigluQuantParams;
    std::ostringstream oss;
    oss << "GMMQuantParams: groupNum = " << params.get_groupNum()
        << ", groupListType = " << static_cast<uint32_t>(params.get_groupListType())
        << ", quantDtype = " << static_cast<int32_t>(params.get_quantDtype())
        << ", dequantDtype = " << static_cast<uint32_t>(params.get_dequantDtype())
        << ", rowLen = " << params.get_rowLen() << ", ubAvail = " << params.get_ubAvail();
    OP_LOGD(inputParams_.opName, "%s", oss.str().c_str());
    return 0;
}

ge::graphStatus GroupedMatmulSwigluQuantV2Tiling950::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = GmmConstant::SYS_WORKSPACE_SIZES;
    if (inputParams_.aQuantMode == optiling::QuantMode::PERTOKEN_MODE) {
        optiling::GMMSwigluQuantParams &params = tilingData_.gmmSwigluQuantParams;
        uint32_t workSize = 1;
        if (params.get_dequantDtype() == 1 || params.get_dequantDtype() == GmmConstant::BF16_VALUE) {
            workSize = GmmConstant::BF16_WORKSIZE;
        } else {
            workSize = GmmConstant::FP32_WORKSIZE;
        }
        workspaces[0] += static_cast<size_t>(inputParams_.nSize >> 1) * inputParams_.mSize * workSize;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
