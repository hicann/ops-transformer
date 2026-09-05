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
 * \file grouped_matmul_swiglu_quant_v2_weight_quant_tiling.cpp
 * \brief GMMSQ MxA8W4 weight quant tiling implementation
 */
#include "grouped_matmul_swiglu_quant_v2_weight_quant_tiling.h"
#include "../grouped_matmul_swiglu_quant_v2_tiling.h"
#include "../../../op_kernel/arch35/weight_quant_basic_block/grouped_matmul_swiglu_quant_v2_weight_quant_tiling_key.h"
#include "op_host/tiling_templates_registry.h"
#include "common/include/err/ops_err.h"
#include "../../../../grouped_matmul/op_host/grouped_matmul_host_util.h"
using namespace Ops::Transformer::OpTiling;

namespace {
static constexpr const char *OP_NAME = "GroupedMatmulSwigluQuantV2";
} // namespace

namespace optiling {
using namespace GroupedMatmulSwigluQuantV2Tiling;

constexpr uint32_t MX_BLOCK_SIZE = 32;
constexpr uint32_t MX_SCALE_BLOCK_SIZE = 64;
constexpr uint32_t MX_INNER_DIM = 2;
constexpr uint32_t ALIGN_SIZE_K = 32;
constexpr uint32_t ALIGN_SIZE_N = 128;
constexpr uint32_t AIC_AIV_CORE_RATIO = 2;

constexpr uint64_t DIM_NUM_X = 2;
constexpr uint64_t DIM_NUM_XSCALE = 3;
constexpr uint64_t DIM_NUM_GROUPLIST = 1;
constexpr uint64_t DIM_NUM_WEIGHT_NZ = 5;
constexpr uint64_t DIM_NUM_WEIGHT_ND = 3;
constexpr uint64_t DIM_NUM_WSCALE = 4;

constexpr int64_t C0_SIZE = 32;
constexpr int64_t CUBE_BLOCK = 16;

constexpr int64_t MIN_M_SIZE = 1;
constexpr int64_t MIN_K_SIZE = 32;
constexpr int64_t MIN_N_SIZE = 128;
constexpr int64_t MAX_GROUP_NUM = 1024;

constexpr int64_t MX_QUANT_MODE = 2;

template <typename T>
constexpr T GetOrDefault(const T *ptr, T defaultValue)
{
    return ptr == nullptr ? defaultValue : *ptr;
}

bool CheckCoreNum(uint32_t aicNum, uint32_t aivNum)
{
    if (unlikely(aicNum <= 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "aicNum", std::to_string(aicNum),
                                              "The aicNum must be greater than 0");
        return false;
    }

    if (unlikely(aivNum <= 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "aivNum", std::to_string(aivNum),
                                              "The aivNum must be greater than 0");
        return false;
    }

    if (unlikely(aicNum * AIC_AIV_CORE_RATIO != aivNum)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "AIC_AIV_CORE_RATIO", std::to_string(aicNum) + ":" + std::to_string(aivNum),
            "The cube:vector core ratio must be 1:" + std::to_string(AIC_AIV_CORE_RATIO));
        return false;
    }
    return true;
}

bool GetContext(GMMSQWeightQuantInputParams &params, const gert::TilingContext &context)
{
    params.opName = context.GetNodeName();
    return true;
}

// 检查动态TensorList中指定下标的shape是否存在且非空。
bool CheckDynamicInputShapeNotEmpty(const gert::TilingContext &context, uint32_t inputIndex, uint32_t tensorIndex,
                                    const char *inputName)
{
    const std::string paramName = std::string(inputName) + "[" + std::to_string(tensorIndex) + "]";
    const auto *shapePtr = context.GetDynamicInputShape(inputIndex, tensorIndex);
    OP_CHECK_IF(
        shapePtr == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, paramName, "nullptr", "Dynamic input shape cannot be nullptr"),
        return false);

    const auto &storageShape = shapePtr->GetStorageShape();
    OP_CHECK_IF(
        storageShape.GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, paramName, "shapeSize=0", "Empty tensor is not supported"),
        return false);
    return true;
}

bool IsDynamicInputEmpty(const gert::TilingContext &context, uint32_t index)
{
    const auto *shapePtr = context.GetDynamicInputShape(index, 0);
    if (shapePtr == nullptr) {
        return true;
    }
    const auto *desc = context.GetDynamicInputDesc(index, 0);
    if (desc == nullptr) {
        return true;
    }
    const auto &storageShape = shapePtr->GetStorageShape();
    return storageShape.GetShapeSize() == 0 || storageShape.GetShapeSize() == 1;
}

uint32_t GetDynamicInputTensorListSize(const gert::TilingContext &context, uint32_t inputIndex)
{
    uint32_t count = 0;
    for (; count <= MAX_GROUP_NUM; ++count) {
        if (context.GetDynamicInputShape(inputIndex, count) == nullptr) {
            break;
        }
    }
    return count;
}

bool CheckDynamicInputDesc(const gert::TilingContext &context, uint32_t inputIndex, uint32_t tensorIndex,
                           const char *inputName, ge::DataType expectedDtype)
{
    const auto *desc = context.GetDynamicInputDesc(inputIndex, tensorIndex);
    OP_CHECK_IF(desc == nullptr, OP_LOGE(OP_NAME, "Dynamic input desc of %s[%u] is nullptr.", inputName, tensorIndex),
                return false);
    OP_CHECK_IF(desc->GetDataType() != expectedDtype,
                OP_LOGE(OP_NAME, "%s[%u] dtype must be %s, got %s.", inputName, tensorIndex,
                        ge::TypeUtils::DataTypeToSerialString(expectedDtype).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(desc->GetDataType()).c_str()),
                return false);
    return true;
}

bool CheckWeightFormat(const gert::TilingContext &context, uint32_t tensorIndex)
{
    const auto *desc = context.GetDynamicInputDesc(WEIGHT_INDEX, tensorIndex);
    OP_CHECK_IF(desc == nullptr, OP_LOGE(OP_NAME, "Dynamic input desc of weight[%u] is nullptr.", tensorIndex),
                return false);
    auto wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat()));
    if (unlikely(wFormat != ge::FORMAT_FRACTAL_NZ && wFormat != ge::FORMAT_FRACTAL_NZ_C0_32)) {
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "weight", ge::TypeUtils::FormatToAscendString(wFormat).GetString(),
                                   "FRACTAL_NZ or FRACTAL_NZ_C0_32");
        return false;
    }
    return true;
}

bool CheckWeightScaleFormat(const gert::TilingContext &context, uint32_t tensorIndex, const char *opName)
{
    const auto *desc = context.GetDynamicInputDesc(WEIGHT_SCALE_INDEX, tensorIndex);
    OP_CHECK_IF(desc == nullptr, OP_LOGE(opName, "Dynamic input desc of weightScale[%u] is nullptr.", tensorIndex),
                return false);
    auto wScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat()));
    if (unlikely(wScaleFormat != ge::FORMAT_ND && wScaleFormat != ge::FORMAT_NCHW && wScaleFormat != ge::FORMAT_NCL)) {
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "weightScale", ge::TypeUtils::FormatToSerialString(wScaleFormat),
                                   "ND, NCHW or NCL");
        return false;
    }
    return true;
}

bool CheckWeightAndScaleListSize(const gert::TilingContext &context, GMMSQWeightQuantInputParams &params)
{
    uint32_t weightListSize = GetDynamicInputTensorListSize(context, WEIGHT_INDEX);
    uint32_t weightScaleListSize = GetDynamicInputTensorListSize(context, WEIGHT_SCALE_INDEX);
    params.isSingleMultiSingle = weightListSize > 1;
    OP_CHECK_IF((weightListSize > 1 || weightScaleListSize > 1) && weightListSize != weightScaleListSize,
                OP_LOGE(OP_NAME, "weight and weightScale tensor list size should be equal, got %u and %u.",
                        weightListSize, weightScaleListSize),
                return false);
    params.groupNum = params.isSingleMultiSingle ? static_cast<int64_t>(weightListSize) : params.groupNum;
    return true;
}

bool CheckTilingDataCapacity(gert::TilingContext &context, uint64_t tilingDataSize)
{
    const auto *rawTiling = context.GetRawTilingData();
    OP_CHECK_IF(rawTiling == nullptr, OP_LOGE(OP_NAME, "GetRawTilingData can not be empty."), return false);
    OP_CHECK_IF(rawTiling->GetData() == nullptr, OP_LOGE(OP_NAME, "RawTilingData cannot be a null pointer."),
                return false);
    OP_CHECK_IF(rawTiling->GetCapacity() < tilingDataSize,
                OP_LOGE(OP_NAME, "context tiling data capacity %zu < actual tiling data size %zu.",
                        static_cast<size_t>(rawTiling->GetCapacity()), static_cast<size_t>(tilingDataSize)),
                return false);
    return true;
}

bool CheckContext(gert::TilingContext &context, uint64_t tilingDataSize)
{
    const auto *attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(OP_NAME, "Function context.GetAttrs() failed!"), return false);

    OP_CHECK_IF(context.GetInputDesc(X_INDEX) == nullptr, OP_LOGE(OP_NAME, "InputDesc x is nullptr."), return false);
    OP_CHECK_IF(context.GetInputDesc(X_SCALE_INDEX) == nullptr, OP_LOGE(OP_NAME, "InputDesc xScale is nullptr."),
                return false);
    OP_CHECK_IF(context.GetInputDesc(GROUPLIST_INDEX) == nullptr, OP_LOGE(OP_NAME, "InputDesc groupList is nullptr."),
                return false);
    OP_CHECK_IF(context.GetInputDesc(WEIGHT_INDEX) == nullptr, OP_LOGE(OP_NAME, "InputDesc weight is nullptr."),
                return false);
    OP_CHECK_IF(context.GetInputDesc(WEIGHT_SCALE_INDEX) == nullptr,
                OP_LOGE(OP_NAME, "InputDesc weightScale is nullptr."), return false);
    OP_CHECK_IF(context.GetOutputDesc(Y_INDEX) == nullptr, OP_LOGE(OP_NAME, "OutputDesc y is nullptr."), return false);
    OP_CHECK_IF(context.GetOutputDesc(Y_SCALE_INDEX) == nullptr, OP_LOGE(OP_NAME, "OutputDesc yScale is nullptr."),
                return false);

    if (unlikely(!IsDynamicInputEmpty(context, WEIGHT_ASSIST_MATRIX_INDEX))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "weightAssistMatrix", "not nullptr",
            "weightAssistMatrix is not supported, please pass a null pointer or an empty list");
        return false;
    }
    if (unlikely(context.GetOptionalInputDesc(BIAS_INDEX) != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "bias", "not nullptr",
                                              "bias is not supported, please pass a null pointer");
        return false;
    }
    if (unlikely(context.GetOptionalInputDesc(SMOOTH_SCALE_INDEX) != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "smoothScale", "not nullptr",
                                              "smoothScale is not supported, please pass a null pointer");
        return false;
    }

    return CheckTilingDataCapacity(context, tilingDataSize);
}

bool GetAttrs(GMMSQWeightQuantInputParams &params, const gert::TilingContext &context)
{
    params.wTrans = true;
    params.groupListType = 0;
    params.dequantMode = MX_QUANT_MODE;
    params.dequantDtype = static_cast<int64_t>(ge::DataType::DT_FLOAT);
    params.quantMode = MX_QUANT_MODE;
    params.quantDtype = static_cast<int64_t>(ge::DataType::DT_FLOAT8_E4M3FN);

    const auto *attrs = context.GetAttrs();
    if (attrs == nullptr) {
        return true;
    }

    params.wTrans = GetOrDefault(attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_WEIGHT), true);
    params.groupListType = GetOrDefault(attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPLIST_TYPE), 0L);
    params.dequantMode = GetOrDefault(attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_MODE), MX_QUANT_MODE);
    params.dequantDtype = GetOrDefault(attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_DTYPE),
                                       static_cast<int64_t>(ge::DataType::DT_FLOAT));
    params.quantMode = GetOrDefault(attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE), MX_QUANT_MODE);
    params.quantDtype = GetOrDefault(attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_DTYPE),
                                     static_cast<int64_t>(ge::DataType::DT_FLOAT8_E4M3FN));
    return true;
}

bool CheckAttrs([[maybe_unused]] const gert::TilingContext &context, const GMMSQWeightQuantInputParams &params)
{
    if (unlikely(params.dequantMode != MX_QUANT_MODE)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "dequantMode", std::to_string(params.dequantMode),
                                              "dequantMode must be " + std::to_string(MX_QUANT_MODE) + " (MX quant)");
        return false;
    }

    if (unlikely(params.dequantDtype != ge::DataType::DT_FLOAT)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "dequantDtype", std::to_string(params.dequantDtype),
                                              "dequantDtype must be " + std::to_string(ge::DataType::DT_FLOAT) + " (" +
                                                  ge::TypeUtils::DataTypeToSerialString(ge::DataType::DT_FLOAT) + ")");
        return false;
    }

    if (unlikely(params.quantMode != MX_QUANT_MODE)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "quantMode", std::to_string(params.quantMode),
                                              "quantMode must be " + std::to_string(MX_QUANT_MODE) + " (MX quant)");
        return false;
    }

    if (unlikely(params.quantDtype != ge::DataType::DT_FLOAT8_E4M3FN)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "quantDtype", std::to_string(params.quantDtype),
            "quantDtype must be " + std::to_string(ge::DataType::DT_FLOAT8_E4M3FN) + " (" +
                ge::TypeUtils::DataTypeToSerialString(ge::DataType::DT_FLOAT8_E4M3FN) + ")");
        return false;
    }

    if (unlikely(!params.wTrans)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "transposeWeight", "false",
                                              "The value of transposeWeight must be true");
        return false;
    }

    if (unlikely(params.groupListType != 0 && params.groupListType != 1)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "groupListType", std::to_string(params.groupListType),
                                              "Attr groupListType must be 0 or 1");
        return false;
    }

    return true;
}

bool CheckDtypes(const gert::TilingContext &context)
{
    const auto *xDesc = context.GetInputDesc(X_INDEX);
    const auto *wDesc = context.GetDynamicInputDesc(WEIGHT_INDEX, 0);
    const auto *xScaleDesc = context.GetInputDesc(X_SCALE_INDEX);
    const auto *wScaleDesc = context.GetDynamicInputDesc(WEIGHT_SCALE_INDEX, 0);
    const auto *groupListDesc = context.GetInputDesc(GROUPLIST_INDEX);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(OP_NAME, "Dynamic input desc of weight[0] is nullptr."), return false);
    OP_CHECK_IF(wScaleDesc == nullptr, OP_LOGE(OP_NAME, "Dynamic input desc of weightScale[0] is nullptr."),
                return false);

    if (unlikely(xDesc->GetDataType() != ge::DataType::DT_FLOAT8_E4M3FN)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "x", ge::TypeUtils::DataTypeToSerialString(xDesc->GetDataType()),
            "The dtype of x must be " + ge::TypeUtils::DataTypeToSerialString(ge::DataType::DT_FLOAT8_E4M3FN));
        return false;
    }

    if (unlikely(wDesc->GetDataType() != ge::DataType::DT_FLOAT4_E2M1)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "weight", ge::TypeUtils::DataTypeToSerialString(wDesc->GetDataType()),
            "The dtype of weight must be " + ge::TypeUtils::DataTypeToSerialString(ge::DataType::DT_FLOAT4_E2M1));
        return false;
    }

    if (unlikely(xScaleDesc->GetDataType() != ge::DataType::DT_FLOAT8_E8M0)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "xScale", ge::TypeUtils::DataTypeToSerialString(xScaleDesc->GetDataType()),
            "The dtype of xScale must be " + ge::TypeUtils::DataTypeToSerialString(ge::DataType::DT_FLOAT8_E8M0));
        return false;
    }

    if (unlikely(wScaleDesc->GetDataType() != ge::DataType::DT_FLOAT8_E8M0)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "weightScale", ge::TypeUtils::DataTypeToSerialString(wScaleDesc->GetDataType()),
            "The dtype of weightScale must be " + ge::TypeUtils::DataTypeToSerialString(ge::DataType::DT_FLOAT8_E8M0));
        return false;
    }

    if (unlikely(groupListDesc->GetDataType() != ge::DataType::DT_INT64)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "groupList", ge::TypeUtils::DataTypeToSerialString(groupListDesc->GetDataType()),
            "The dtype of groupList must be " + ge::TypeUtils::DataTypeToSerialString(ge::DataType::DT_INT64));
        return false;
    }

    return true;
}

bool GetInputs(GMMSQWeightQuantInputParams &params, const gert::TilingContext &context)
{
    OP_CHECK_IF(!CheckWeightFormat(context, 0), OP_LOGE(OP_NAME, "Check weight format failed."), return false);
    const auto *wDesc = context.GetDynamicInputDesc(WEIGHT_INDEX, 0);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(OP_NAME, "Dynamic input desc of weight[0] is nullptr."), return false);
    params.wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));

    const auto *xShapePtr = context.GetInputShape(X_INDEX);
    OP_CHECK_IF(!CheckDynamicInputShapeNotEmpty(context, WEIGHT_INDEX, 0, "weight"),
                OP_LOGE(OP_NAME, "Check dynamic input shape of weight failed."), return false);
    OP_CHECK_IF(!CheckDynamicInputShapeNotEmpty(context, WEIGHT_SCALE_INDEX, 0, "weightScale"),
                OP_LOGE(OP_NAME, "Check dynamic input shape of weightScale failed."), return false);
    const auto *wShapePtr = context.GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_IF(xShapePtr == nullptr || wShapePtr == nullptr, OP_LOGE(OP_NAME, "Required input shape is nullptr."),
                return false);

    const gert::Shape &xShape = xShapePtr->GetOriginShape();
    const gert::Shape &wShape = wShapePtr->GetOriginShape();
    auto xShapeLen = xShape.GetDimNum();
    auto wShapeLen = wShape.GetDimNum();
    OP_CHECK_IF(!CheckWeightAndScaleListSize(context, params),
                OP_LOGE(OP_NAME, "Check weight and weightScale tensor list size failed."), return false);
    OP_CHECK_IF(xShape.GetShapeSize() == 0, OP_LOGE(OP_NAME, "Not yet support empty tensor for x"), return false);
    if (unlikely(xShapeLen != DIM_NUM_X)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "x", std::to_string(xShapeLen),
                                                 "The shape dim of x must be " + std::to_string(DIM_NUM_X));
        return false;
    }
    uint64_t expectedWeightDim = params.isSingleMultiSingle ? DIM_NUM_WEIGHT_ND - 1 : DIM_NUM_WEIGHT_ND;
    if (unlikely(wShapeLen != expectedWeightDim)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "weight", std::to_string(wShapeLen),
            "The shape dim of weight must be " + std::to_string(expectedWeightDim));
        return false;
    }

    // - 1 为内轴
    auto xInner = xShape.GetDim(DIM_NUM_X - 1);
    auto wInner = wShape.GetDim(wShapeLen - 1);
    if (unlikely(xInner != wInner)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(OP_NAME, "x, weight",
                                               "x K=" + std::to_string(xInner) + ", weight K=" + std::to_string(wInner),
                                               "K of x must be equal to K of weight");
        return false;
    }
    params.mSize = xShape.GetDim(DIM_0);
    params.kSize = xInner;
    params.nSize = params.isSingleMultiSingle ? wShape.GetDim(DIM_0) : wShape.GetDim(DIM_1);
    params.groupNum = params.isSingleMultiSingle ? params.groupNum : wShape.GetDim(DIM_0);
    return true;
}

bool CheckInputFormats(const gert::TilingContext &context, const GMMSQWeightQuantInputParams &params)
{
    const auto *xDesc = context.GetInputDesc(X_INDEX);
    auto xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetFormat().GetStorageFormat()));
    if (unlikely(xFormat != ge::FORMAT_ND)) {
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "x", ge::TypeUtils::FormatToSerialString(xFormat), "ND");
        return false;
    }

    const auto *xScaleDesc = context.GetInputDesc(X_SCALE_INDEX);
    auto xScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xScaleDesc->GetFormat().GetStorageFormat()));
    if (unlikely(xScaleFormat != ge::FORMAT_ND && xScaleFormat != ge::FORMAT_NCL)) {
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "xScale", ge::TypeUtils::FormatToSerialString(xScaleFormat), "ND or NCL");
        return false;
    }

    const auto *groupListDesc = context.GetInputDesc(GROUPLIST_INDEX);
    auto groupListFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(groupListDesc->GetFormat().GetStorageFormat()));
    if (unlikely(groupListFormat != ge::FORMAT_ND)) {
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "groupList", ge::TypeUtils::FormatToSerialString(groupListFormat), "ND");
        return false;
    }

    OP_CHECK_IF(!CheckWeightScaleFormat(context, 0, params.opName.c_str()),
                OP_LOGE(OP_NAME, "Check weightScale format failed."), return false);

    return true;
}

std::string ToShapeString(const std::vector<int64_t> &shape)
{
    std::string shapeStr("[");
    const char *sep = "";
    for (auto x : shape) {
        shapeStr.append(sep);
        shapeStr.append(std::to_string(x));
        sep = ", ";
    }
    shapeStr.push_back(']');
    return shapeStr;
}

bool CheckInputShape(const char *variableName, const gert::Shape &shape, const std::vector<int64_t> &expectedShape)
{
    auto shapeLen = shape.GetDimNum();
    if (unlikely(shapeLen != expectedShape.size())) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, variableName, std::to_string(shapeLen),
            "The shape dim of " + std::string(variableName) + " must be " + std::to_string(expectedShape.size()));
        return false;
    }
    size_t i = 0;
    for (auto dim : expectedShape) {
        if (unlikely(dim != shape.GetDim(i++))) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, variableName, Ops::Base::ToString(shape),
                "Shape does not match the expected value " + ToShapeString(expectedShape));
            return false;
        }
    }
    return true;
}

bool IsInputShapeValid([[maybe_unused]] gert::TilingContext &context, const GMMSQWeightQuantInputParams &params)
{
    if (unlikely(params.mSize < MIN_M_SIZE)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x", "M=" + std::to_string(params.mSize),
                                              "M of x must be >= " + std::to_string(MIN_M_SIZE));
        return false;
    }
    if (unlikely(params.kSize < MIN_K_SIZE)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x", "K=" + std::to_string(params.kSize),
                                              "K of x must be >= " + std::to_string(MIN_K_SIZE));
        return false;
    }
    if (unlikely(params.nSize < MIN_N_SIZE)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "weight", "N=" + std::to_string(params.nSize),
                                              "N of weight must be >= " + std::to_string(MIN_N_SIZE));
        return false;
    }
    if (unlikely(params.kSize % ALIGN_SIZE_K != 0)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x", "K=" + std::to_string(params.kSize),
                                              "K of x must be aligned to " + std::to_string(ALIGN_SIZE_K));
        return false;
    }
    if (unlikely(params.nSize % ALIGN_SIZE_N != 0)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "weight", "N=" + std::to_string(params.nSize),
                                              "N of weight must be aligned to " + std::to_string(ALIGN_SIZE_N));
        return false;
    }
    if (unlikely(params.groupNum < 1 || params.groupNum > MAX_GROUP_NUM)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "weight", "groupNum=" + std::to_string(params.groupNum),
            "groupNum of weight must be in range [1, " + std::to_string(MAX_GROUP_NUM) + "]");
        return false;
    }
    return true;
}

bool CheckSingleMultiSingleInputs(gert::TilingContext &context, const GMMSQWeightQuantInputParams &params,
                                  int64_t scaleKSize)
{
    const std::vector<int64_t> expectedWeightShape = {GroupedMatmul::CeilDiv<int64_t>(params.kSize, C0_SIZE),
                                                      GroupedMatmul::CeilDiv<int64_t>(params.nSize, CUBE_BLOCK),
                                                      CUBE_BLOCK, C0_SIZE};
    const std::vector<int64_t> expectedWeightScaleShape = {params.nSize, scaleKSize, MX_INNER_DIM};
    for (int64_t i = 0; i < params.groupNum; ++i) {
        uint32_t tensorIndex = static_cast<uint32_t>(i);
        auto *curWeightShapePtr = context.GetDynamicInputShape(WEIGHT_INDEX, tensorIndex);
        auto *curWeightScaleShapePtr = context.GetDynamicInputShape(WEIGHT_SCALE_INDEX, tensorIndex);
        OP_CHECK_IF(curWeightShapePtr == nullptr || curWeightScaleShapePtr == nullptr,
                    OP_LOGE(params.opName, "weight or weightScale shape[%ld] should not be null", i), return false);
        OP_CHECK_IF(!CheckDynamicInputShapeNotEmpty(context, WEIGHT_INDEX, tensorIndex, "weight"),
                    OP_LOGE(params.opName, "Check dynamic input shape of weight[%ld] failed.", i), return false);
        OP_CHECK_IF(!CheckDynamicInputShapeNotEmpty(context, WEIGHT_SCALE_INDEX, tensorIndex, "weightScale"),
                    OP_LOGE(params.opName, "Check dynamic input shape of weightScale[%ld] failed.", i), return false);
        OP_CHECK_IF(!CheckDynamicInputDesc(context, WEIGHT_INDEX, tensorIndex, "weight", ge::DT_FLOAT4_E2M1),
                    OP_LOGE(params.opName, "Check dynamic input desc of weight[%ld] failed.", i), return false);
        OP_CHECK_IF(!CheckDynamicInputDesc(context, WEIGHT_SCALE_INDEX, tensorIndex, "weightScale", ge::DT_FLOAT8_E8M0),
                    OP_LOGE(params.opName, "Check dynamic input desc of weightScale[%ld] failed.", i), return false);
        OP_CHECK_IF(!CheckWeightFormat(context, tensorIndex),
                    OP_LOGE(params.opName, "Check weight[%ld] format failed.", i), return false);
        OP_CHECK_IF(!CheckWeightScaleFormat(context, tensorIndex, params.opName.c_str()),
                    OP_LOGE(params.opName, "Check weightScale[%ld] format failed.", i), return false);
        if (!CheckInputShape("weight", curWeightShapePtr->GetStorageShape(), expectedWeightShape)) {
            return false;
        }
        if (!CheckInputShape("weightScale", curWeightScaleShapePtr->GetOriginShape(), expectedWeightScaleShape)) {
            return false;
        }
    }
    return true;
}

// 大函数整改：提取 xScale/groupList/weightScale shape 校验为独立函数
static bool CheckScaleAndGroupListShapes(gert::TilingContext &context, const GMMSQWeightQuantInputParams &params,
                                         const gert::Shape &xScaleShape, const gert::Shape &groupListShape,
                                         const gert::Shape &weightScaleShape)
{
    int64_t scaleKSize = GroupedMatmul::CeilDiv<int64_t>(params.kSize, MX_SCALE_BLOCK_SIZE);
    const std::vector<int64_t> expectedXScaleShape = {static_cast<int64_t>(params.mSize), scaleKSize, MX_INNER_DIM};
    if (!CheckInputShape("xScale", xScaleShape, expectedXScaleShape)) {
        return false;
    }
    if (!CheckInputShape("groupList", groupListShape, {params.groupNum})) {
        return false;
    }
    const std::vector<int64_t> expectedWeightScaleShape = {params.groupNum, params.nSize, scaleKSize, MX_INNER_DIM};
    if (!CheckInputShape("weightScale", weightScaleShape, expectedWeightScaleShape)) {
        return false;
    }
    return true;
}

bool CheckInputs(gert::TilingContext &context, const GMMSQWeightQuantInputParams &params)
{
    if (!IsInputShapeValid(context, params)) {
        return false;
    }

    if (!CheckInputFormats(context, params)) {
        return false;
    }

    auto *xScaleShapePtr = context.GetInputShape(X_SCALE_INDEX);
    auto *groupListShapePtr = context.GetInputShape(GROUPLIST_INDEX);
    OP_CHECK_IF(!CheckDynamicInputShapeNotEmpty(context, WEIGHT_INDEX, 0, "weight"),
                OP_LOGE(OP_NAME, "Check dynamic input shape of weight failed."), return false);
    OP_CHECK_IF(!CheckDynamicInputShapeNotEmpty(context, WEIGHT_SCALE_INDEX, 0, "weightScale"),
                OP_LOGE(OP_NAME, "Check dynamic input shape of weightScale failed."), return false);
    auto *weightShapePtr = context.GetDynamicInputShape(WEIGHT_INDEX, 0);
    auto *weightScaleShapePtr = context.GetDynamicInputShape(WEIGHT_SCALE_INDEX, 0);

    OP_CHECK_IF(xScaleShapePtr == nullptr, OP_LOGE(OP_NAME, "xScaleShape cannot be null"), return false);
    OP_CHECK_IF(groupListShapePtr == nullptr, OP_LOGE(OP_NAME, "groupListShape cannot be null"), return false);
    OP_CHECK_IF(weightShapePtr == nullptr || weightScaleShapePtr == nullptr,
                OP_LOGE(OP_NAME, "weightShape or weightScaleShape cannot be null"), return false);

    auto &xScaleShape = xScaleShapePtr->GetOriginShape();
    auto &groupListShape = groupListShapePtr->GetOriginShape();

    int64_t scaleKSize = GroupedMatmul::CeilDiv<int64_t>(params.kSize, MX_SCALE_BLOCK_SIZE);
    const std::vector<int64_t> expectedXScaleShape = {static_cast<int64_t>(params.mSize), scaleKSize, MX_INNER_DIM};
    if (!CheckInputShape("xScale", xScaleShape, expectedXScaleShape)) {
        return false;
    }
    if (!CheckInputShape("groupList", groupListShape, {params.groupNum})) {
        return false;
    }

    if (params.isSingleMultiSingle) {
        return CheckSingleMultiSingleInputs(context, params, scaleKSize);
    }

    if (params.wFormat == ge::FORMAT_FRACTAL_NZ || params.wFormat == ge::FORMAT_FRACTAL_NZ_C0_32) {
        const gert::Shape &wStorageShape = weightShapePtr->GetStorageShape();
        const std::vector<int64_t> expectedWeightShape = {
            static_cast<int64_t>(params.groupNum), GroupedMatmul::CeilDiv<int64_t>(params.kSize, C0_SIZE),
            GroupedMatmul::CeilDiv<int64_t>(params.nSize, CUBE_BLOCK), CUBE_BLOCK, C0_SIZE};
        if (!CheckInputShape("weight", wStorageShape, expectedWeightShape)) {
            return false;
        }
    }

    auto &weightScaleShape = weightScaleShapePtr->GetOriginShape();
    return CheckScaleAndGroupListShapes(context, params, xScaleShape, groupListShape, weightScaleShape);
}

void PrintInputParams(gert::TilingContext &context, const GMMSQWeightQuantInputParams &params)
{
    OP_LOGD(context.GetNodeName(),
            "Input params: mSize=%lld, kSize=%lld, nSize=%lld, groupNum=%lld, "
            "wTrans=%s, groupListType=%lld",
            static_cast<long long>(params.mSize), static_cast<long long>(params.kSize),
            static_cast<long long>(params.nSize), static_cast<long long>(params.groupNum),
            params.wTrans ? "true" : "false", static_cast<long long>(params.groupListType));
}

bool GroupedMatmulSwigluQuantV2WeightQuantTiling::IsCapable()
{
    return true;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::GetPlatformInfo()
{
    const auto *compileInfoPtr = context_->GetCompileInfo<GMMSwigluV2CompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(OP_NAME, "CompileInfo is null"), return ge::GRAPH_FAILED);
    compileInfoPtr_ = compileInfoPtr;
    OP_LOGI(context_, "Compile info: aicNum(%u) aivNum(%u) ubSize(%llu) baseM(%u) baseN(%u) supportL12BtBf16(%s).",
            compileInfoPtr_->aicNum_, compileInfoPtr_->aivNum_, compileInfoPtr_->ubSize_, compileInfoPtr_->baseM_,
            compileInfoPtr_->baseN_, compileInfoPtr_->supportL12BtBf16 ? "true" : "false");

    if (!CheckCoreNum(compileInfoPtr_->aicNum_, compileInfoPtr_->aivNum_)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::GetShapeAttrsInfo()
{
    if (!GetContext(inputParams_, *context_)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckContext(*context_, sizeof(GMMSQArch35Tiling::GMMSQWeightQuantTilingData))) {
        return ge::GRAPH_FAILED;
    }
    if (!GetAttrs(inputParams_, *context_)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckAttrs(*context_, inputParams_)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckDtypes(*context_)) {
        return ge::GRAPH_FAILED;
    }
    if (!GetInputs(inputParams_, *context_)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckInputs(*context_, inputParams_)) {
        return ge::GRAPH_FAILED;
    }

    PrintInputParams(*context_, inputParams_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Start setting tiling data.");

    tilingData_.groupListType = inputParams_.groupListType;
    tilingData_.coreNum = compileInfoPtr_->aicNum_;
    tilingData_.groupNum = inputParams_.groupNum;
    tilingData_.groupSize = MX_BLOCK_SIZE;
    tilingData_.kSize = inputParams_.kSize;
    tilingData_.nSize = inputParams_.nSize;

    OP_LOGD(context_->GetNodeName(),
            "Tiling data set: coreNum=%u, groupNum=%u, "
            "kSize=%llu, nSize=%llu, groupListType=%u.",
            tilingData_.coreNum, tilingData_.groupNum, static_cast<unsigned long long>(tilingData_.kSize),
            static_cast<unsigned long long>(tilingData_.nSize), static_cast<uint32_t>(tilingData_.groupListType));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulSwigluQuantV2WeightQuantTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.wTrans), 0UL,
                              static_cast<uint64_t>(inputParams_.isSingleMultiSingle));
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(OP_NAME, "workspaces is nullptr."), return ge::GRAPH_FAILED);
    workspaces[0] = SYS_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2WeightQuantTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "Start posting tiling data, blockDim=%u, tilingSize=%zu.", tilingData_.coreNum,
            sizeof(tilingData_));

    context_->SetBlockDim(tilingData_.coreNum);
    auto *rawTiling = context_->GetRawTilingData();
    OP_CHECK_IF(rawTiling == nullptr, OP_LOGE(OP_NAME, "RawTilingData is nullptr."), return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(rawTiling->GetData(), rawTiling->GetCapacity(), static_cast<void *>(&tilingData_),
                           sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(OP_NAME, "memcpy_s failed, ret = %d", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(tilingData_));

    OP_LOGD(context_->GetNodeName(), "Tiling data posted successfully.");
    return ge::GRAPH_SUCCESS;
}

void GroupedMatmulSwigluQuantV2WeightQuantTiling::Reset()
{
    tilingData_ = GMMSQArch35Tiling::GMMSQWeightQuantTilingData();
    auto *rawTiling = context_->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetData() == nullptr) {
        return;
    }
    OP_CHECK_IF(memset_s(rawTiling->GetData(), rawTiling->GetCapacity(), 0, rawTiling->GetCapacity()) != EOK,
                OP_LOGE(OP_NAME, "Fail to clear tiling data"), return);
}
} // namespace optiling
