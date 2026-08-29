/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_activation_quant_tiling.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <graph/utils/type_utils.h>
#include "gmm/common/op_host/log_format_util.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "util/math_util.h"
#include "../../../op_kernel/arch35/grouped_matmul_activation_quant_tiling_key.h"

using namespace Ops::Transformer::OpTiling;
using namespace optiling::GmmConstant;
using Ops::Transformer::Gmm::FormatString;

namespace optiling {
namespace {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t GROUP_LIST_INDEX = 1;
constexpr uint32_t WEIGHT_INDEX = 2;
constexpr uint32_t WEIGHT_SCALE_INDEX = 3;
constexpr uint32_t X_SCALE_INDEX = 5;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t Y_SCALE_INDEX = 1;
constexpr uint32_t ATTR_INDEX_ACTIVATION_TYPE = 0;
constexpr uint32_t ATTR_INDEX_TRANS_W = 1;
constexpr uint32_t ATTR_INDEX_GROUP_LIST_TYPE = 2;
constexpr uint32_t ATTR_INDEX_TUNING_CONFIG = 3;
constexpr uint32_t ATTR_INDEX_QUANT_MODE = 4;
constexpr uint32_t ATTR_INDEX_Y_DTYPE = 5;
constexpr uint32_t ATTR_INDEX_ROUND_MODE = 6;
constexpr uint32_t ATTR_INDEX_SCALE_ALG = 7;
constexpr uint32_t ATTR_INDEX_DST_TYPE_MAX = 8;
constexpr uint32_t WEIGHT_NZ_DIM_NUM = 5;
constexpr uint32_t X_SCALE_M_DIM_INDEX = 0;
constexpr uint32_t X_SCALE_K_BLOCK_DIM_INDEX = 1;
constexpr uint32_t X_SCALE_PAIR_DIM_INDEX = 2;
constexpr uint32_t WEIGHT_SCALE_GROUP_DIM_INDEX = 0;
constexpr uint32_t WEIGHT_SCALE_K_BLOCK_DIM_INDEX = 1;
constexpr uint32_t WEIGHT_SCALE_N_DIM_INDEX = 2;
constexpr uint32_t WEIGHT_SCALE_PAIR_DIM_INDEX = 3;
constexpr uint32_t WEIGHT_SCALE_TRANS_N_DIM_INDEX = 1;
constexpr uint32_t WEIGHT_SCALE_TRANS_K_BLOCK_DIM_INDEX = 2;
constexpr uint32_t WEIGHT_NZ_GROUP_DIM_INDEX = 0;
constexpr uint32_t WEIGHT_NZ_BLOCK1_DIM_INDEX = 1;
constexpr uint32_t WEIGHT_NZ_BLOCK2_DIM_INDEX = 2;
constexpr uint32_t WEIGHT_NZ_C0_DIM_INDEX = 3;
constexpr uint32_t WEIGHT_NZ_LAST_DIM_INDEX = 4;
constexpr uint32_t WEIGHT_NZ_LAST_DIM_FP8 = 32;
constexpr uint32_t WEIGHT_NZ_LAST_DIM_FP4 = 64;
constexpr uint32_t WEIGHT_NZ_C0_DIM = 16;
constexpr uint32_t N_SIZE_ALIGN = 64;
constexpr uint32_t MX_SCALE_GROUP_SIZE = 64;
constexpr uint32_t MXFP4_PACK_FACTOR = 2;
// One AIC is paired with two AIVs.  Each AIV processes ceil(baseM / 2) rows,
// while its epilogue buffers hold at most 128 * 256 elements.
constexpr uint64_t MX_EPILOGUE_MAX_ELEMENTS_PER_AIV = 128UL * 256UL;
constexpr uint64_t MX_EPILOGUE_AIV_COUNT = 2UL;
constexpr uint32_t FP8_E4M3FN_VALUE = 36;
constexpr uint32_t FP8_E5M2_VALUE = 35;
constexpr uint32_t FP4_E2M1_VALUE = 40;
constexpr uint32_t FP4_E1M2_VALUE = 41;
constexpr uint8_t ROUND_MODE_RINT = 4;
constexpr uint32_t SCALE_ALG_OCP = 0;
constexpr uint32_t SCALE_ALG_CUBLAS = 1;
constexpr uint32_t SCALE_ALG_DYNAMIC_DTYPE_RANGE = 2;
constexpr uint8_t ACTIVATION_TYPE_GELU_TANH = 2;
constexpr char ACTIVATION_TYPE_GELU_TANH_STR[] = "gelu_tanh";
constexpr char QUANT_MODE_MX_STR[] = "mx";
constexpr float DEFAULT_DST_TYPE_MAX = 0.0f;
constexpr float FP4_E2M1_DST_TYPE_MAX_MIN = 6.0f;
constexpr float FP4_E2M1_DST_TYPE_MAX_MAX = 12.0f;

std::string ShapeToDebugString(const gert::Shape &shape)
{
    std::string result = "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(shape.GetDim(i));
    }
    result += "]";
    return result;
}
} // namespace

void GroupedMatmulActivationQuantTiling950::Reset()
{
    tilingData_ = {};
    roundMode_ = DEFAULT_ROUND_MODE_RINT;
    scaleAlg_ = DEFAULT_SCALE_ALG_OCP;
    activationType_ = DEFAULT_ACTIVATION_TYPE_GELU_TANH;
    dstTypeMax_ = DEFAULT_DST_TYPE_MAX;
    yDtype_ = ge::DT_FLOAT8_E4M3FN;
}

ge::graphStatus GroupedMatmulActivationQuantTiling950::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    return GroupedQmmBasicApiTiling::GetShapeAttrsInfo();
}

bool GroupedMatmulActivationQuantTiling950::IsFp8(ge::DataType dtype) const
{
    return dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2;
}

bool GroupedMatmulActivationQuantTiling950::IsFp4(ge::DataType dtype) const
{
    return dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2;
}

bool GroupedMatmulActivationQuantTiling950::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is nullptr."), return false);
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
    inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    inputParams_.transA = false;
    const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
    inputParams_.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams_.groupListType;
    OP_CHECK_IF(inputParams_.groupListType != 0 && inputParams_.groupListType != 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "group_list_type",
                                                      std::to_string(inputParams_.groupListType),
                                                      "group_list_type must be 0 or 1"),
                return false);

    const char *quantModePtr = attrs->GetAttrPointer<char>(ATTR_INDEX_QUANT_MODE);
    const int64_t *yDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_Y_DTYPE);
    const char *roundModePtr = attrs->GetAttrPointer<char>(ATTR_INDEX_ROUND_MODE);
    const int64_t *scaleAlgPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SCALE_ALG);
    const float *dstTypeMaxPtr = attrs->GetAttrPointer<float>(ATTR_INDEX_DST_TYPE_MAX);
    const char *activationTypePtr = attrs->GetAttrPointer<char>(ATTR_INDEX_ACTIVATION_TYPE);
    OP_CHECK_IF(yDtypePtr == nullptr || roundModePtr == nullptr || scaleAlgPtr == nullptr || dstTypeMaxPtr == nullptr ||
                    activationTypePtr == nullptr,
                OP_LOGE(context_->GetNodeName(), "quant attrs must not be nullptr."), return false);
    auto yDtype = static_cast<ge::DataType>(*yDtypePtr);
    if (yDtype == ge::DT_UNDEFINED) {
        yDtype = inputParams_.aDtype;
    }
    std::string quantMode = quantModePtr == nullptr ? "" : std::string(quantModePtr);
    if (quantMode.empty()) {
        OP_CHECK_IF(
            !((IsFp8(inputParams_.aDtype) || IsFp4(inputParams_.aDtype)) &&
              inputParams_.perTokenScaleDtype == ge::DT_FLOAT8_E8M0),
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), "x and x_scale",
                FormatString("x=%s, x_scale=%s", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                             ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),
                "when quant_mode is empty, x must be FLOAT8 or FLOAT4 and x_scale must be FLOAT8_E8M0 "
                "to infer quant_mode as mx"),
            return false);
        quantMode = QUANT_MODE_MX_STR;
    }
    OP_CHECK_IF(quantMode != QUANT_MODE_MX_STR,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "quant_mode", quantMode,
                                                      "quant_mode must be mx"),
                return false);
    OP_CHECK_IF(!IsFp8(yDtype) && !IsFp4(yDtype),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "y_dtype", ge::TypeUtils::DataTypeToSerialString(yDtype),
                    "when the quantization mode is mx, y_dtype must be FLOAT8_E4M3FN, FLOAT8_E5M2, "
                    "FLOAT4_E2M1 or FLOAT4_E1M2"),
                return false);
    std::string roundMode(roundModePtr);
    OP_CHECK_IF(roundMode != "rint",
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "round_mode", roundMode,
                                                      "when the quantization mode is mx, round_mode must be rint"),
                return false);
    OP_CHECK_IF(
        *scaleAlgPtr != SCALE_ALG_OCP && *scaleAlgPtr != SCALE_ALG_CUBLAS &&
            *scaleAlgPtr != SCALE_ALG_DYNAMIC_DTYPE_RANGE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "scale_alg", std::to_string(*scaleAlgPtr),
                                              "when the quantization mode is mx, scale_alg must be 0, 1 or 2"),
        return false);
    OP_CHECK_IF(
        *scaleAlgPtr == SCALE_ALG_CUBLAS && !IsFp8(yDtype),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "scale_alg", std::to_string(*scaleAlgPtr),
                                              "scale_alg 1 only supports FLOAT8 output"),
        return false);
    OP_CHECK_IF(
        *scaleAlgPtr == SCALE_ALG_DYNAMIC_DTYPE_RANGE && yDtype != ge::DT_FLOAT4_E2M1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "scale_alg", std::to_string(*scaleAlgPtr),
                                              "scale_alg 2 only supports FLOAT4_E2M1 output"),
        return false);
    const bool isDefaultDstTypeMax = Ops::Base::IsFloatEqual(*dstTypeMaxPtr, DEFAULT_DST_TYPE_MAX);
    if (*scaleAlgPtr == SCALE_ALG_DYNAMIC_DTYPE_RANGE) {
        OP_CHECK_IF(!std::isfinite(*dstTypeMaxPtr) ||
                        (!isDefaultDstTypeMax &&
                         (*dstTypeMaxPtr < FP4_E2M1_DST_TYPE_MAX_MIN || *dstTypeMaxPtr > FP4_E2M1_DST_TYPE_MAX_MAX)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "dst_type_max", std::to_string(*dstTypeMaxPtr),
                        "when scale_alg is 2, dst_type_max must be 0.0 or within [6.0, 12.0]"),
                    return false);
    } else {
        OP_CHECK_IF(!isDefaultDstTypeMax,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dst_type_max",
                                                          std::to_string(*dstTypeMaxPtr),
                                                          "when scale_alg is not 2, dst_type_max must be 0.0"),
                    return false);
    }
    std::string activationType(activationTypePtr);
    OP_CHECK_IF(activationType != ACTIVATION_TYPE_GELU_TANH_STR,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "activation_type", activationType,
                                                      "activation_type must be gelu_tanh"),
                return false);
    roundMode_ = ROUND_MODE_RINT;
    scaleAlg_ = static_cast<uint8_t>(*scaleAlgPtr);
    activationType_ = ACTIVATION_TYPE_GELU_TANH;
    dstTypeMax_ = isDefaultDstTypeMax ? DEFAULT_DST_TYPE_MAX : *dstTypeMaxPtr;
    yDtype_ = yDtype;
    inputParams_.groupType = GroupedMatmul::SPLIT_M;
    return true;
}

bool GroupedMatmulActivationQuantTiling950::CheckDtype() const
{
    const bool isSupportedFp8 = IsFp8(inputParams_.aDtype) && inputParams_.bDtype == ge::DT_FLOAT8_E4M3FN;
    const bool isSupportedFp4 = IsFp4(inputParams_.aDtype) && IsFp4(inputParams_.bDtype);
    OP_CHECK_IF(!(isSupportedFp8 || isSupportedFp4),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "x, weight",
                    FormatString("{%s, %s}", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                 ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
                    "when the quantization mode is mx, x and weight must be a supported FP8 pair or both be FLOAT4"),
                return false);
    OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0 || inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "x_scale and weight_scale",
                    FormatString("x_scale=%s, weight_scale=%s",
                                 ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str(),
                                 ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
                    "when the quantization mode is mx, the dtype of x_scale and weight_scale must be FLOAT8_E8M0"),
                return false);
    OP_CHECK_IF(
        inputParams_.outScaleDtype != ge::DT_FLOAT8_E8M0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "y_scale", ge::TypeUtils::DataTypeToSerialString(inputParams_.outScaleDtype),
            "when the quantization mode is mx, the dtype of y_scale must be FLOAT8_E8M0"),
        return false);
    OP_CHECK_IF(IsFp4(inputParams_.outDataDtype) && !isSupportedFp4,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "x, weight, y",
                    FormatString("{%s, %s, %s}", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                 ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                                 ge::TypeUtils::DataTypeToSerialString(inputParams_.outDataDtype).c_str()),
                    "FLOAT4 output requires both x and weight to be FLOAT4"),
                return false);
    OP_CHECK_IF(
        inputParams_.outDataDtype != yDtype_,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "y and y_dtype",
            FormatString("y=%s, y_dtype=%s", ge::TypeUtils::DataTypeToSerialString(inputParams_.outDataDtype).c_str(),
                         ge::TypeUtils::DataTypeToSerialString(yDtype_).c_str()),
            "the y descriptor dtype must match the y_dtype attribute"),
        return false);
    return true;
}

bool GroupedMatmulActivationQuantTiling950::AnalyzeDtype()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    auto wDesc = context_->GetInputDesc(WEIGHT_INDEX);
    auto xScaleDesc = context_->GetInputDesc(X_SCALE_INDEX);
    auto wScaleDesc = context_->GetInputDesc(WEIGHT_SCALE_INDEX);
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    auto yScaleDesc = context_->GetOutputDesc(Y_SCALE_INDEX);
    OP_CHECK_IF(xDesc == nullptr || wDesc == nullptr || xScaleDesc == nullptr || wScaleDesc == nullptr ||
                    yDesc == nullptr || yScaleDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "input or output desc is nullptr."), return false);
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is nullptr."), return false);
    const int64_t *yDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_Y_DTYPE);
    OP_CHECK_IF(yDtypePtr == nullptr, OP_LOGE(context_->GetNodeName(), "y_dtype must not be nullptr."), return false);
    inputParams_.aDtype = xDesc->GetDataType();
    yDtype_ = static_cast<ge::DataType>(*yDtypePtr);
    inputParams_.bDtype = wDesc->GetDataType();
    inputParams_.scaleDtype = wScaleDesc->GetDataType();
    inputParams_.perTokenScaleDtype = xScaleDesc->GetDataType();
    inputParams_.outDataDtype = yDesc->GetDataType();
    inputParams_.outScaleDtype = yScaleDesc->GetDataType();
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    inputParams_.hasBias = false;
    return CheckDtype();
}

bool GroupedMatmulActivationQuantTiling950::CheckMxScaleShape(const gert::Shape &xScaleShape,
                                                              const gert::Shape &wScaleShape) const
{
    OP_CHECK_IF(
        xScaleShape.GetDimNum() != MXFP_PER_TOKEN_SCALE_DIM_NUM || wScaleShape.GetDimNum() != MXFP_TYPE_M_SCALE_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "x_scale and weight_scale",
            FormatString("x_scale=%zu, weight_scale=%zu", xScaleShape.GetDimNum(), wScaleShape.GetDimNum()),
            "x_scale dim num must be 3 and weight_scale dim num must be 4"),
        return false);
    auto expectedKBlocks = static_cast<int64_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(inputParams_.kSize), static_cast<int64_t>(MXFP_BASEK_FACTOR)));
    auto wScaleN = inputParams_.transB ? wScaleShape.GetDim(WEIGHT_SCALE_TRANS_N_DIM_INDEX) :
                                         wScaleShape.GetDim(WEIGHT_SCALE_N_DIM_INDEX);
    auto wScaleK = inputParams_.transB ? wScaleShape.GetDim(WEIGHT_SCALE_TRANS_K_BLOCK_DIM_INDEX) :
                                         wScaleShape.GetDim(WEIGHT_SCALE_K_BLOCK_DIM_INDEX);
    OP_CHECK_IF(xScaleShape.GetDim(X_SCALE_M_DIM_INDEX) != static_cast<int64_t>(inputParams_.mSize) ||
                    xScaleShape.GetDim(X_SCALE_K_BLOCK_DIM_INDEX) != expectedKBlocks ||
                    xScaleShape.GetDim(X_SCALE_PAIR_DIM_INDEX) != static_cast<int64_t>(MXFP_MULTI_BASE_SIZE),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "x_scale", ShapeToDebugString(xScaleShape),
                    FormatString("when the quantization mode is mx, the shape of x_scale must be (%lu, %ld, 2)",
                                 inputParams_.mSize, expectedKBlocks)),
                return false);
    OP_CHECK_IF(wScaleShape.GetDim(WEIGHT_SCALE_GROUP_DIM_INDEX) != static_cast<int64_t>(inputParams_.groupNum) ||
                    wScaleN != static_cast<int64_t>(inputParams_.nSize) || wScaleK != expectedKBlocks ||
                    wScaleShape.GetDim(WEIGHT_SCALE_PAIR_DIM_INDEX) != static_cast<int64_t>(MXFP_MULTI_BASE_SIZE),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "weight_scale", ShapeToDebugString(wScaleShape),
                    "weight_scale shape must be (E, ceil(K/64), N, 2) or (E, N, ceil(K/64), 2)"),
                return false);
    return true;
}

bool GroupedMatmulActivationQuantTiling950::CheckWeightNzShape(const gert::Shape &wStorageShape) const
{
    OP_CHECK_IF(inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ,
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                    context_->GetNodeName(), "weight", ge::TypeUtils::FormatToSerialString(inputParams_.bFormat),
                    "when the quantization mode is mx, the format of weight must be FRACTAL_NZ"),
                return false);
    OP_CHECK_IF(wStorageShape.GetDimNum() != WEIGHT_NZ_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->GetNodeName(), "weight storageShape", std::to_string(wStorageShape.GetDimNum()),
                    "when the quantization mode is mx, the dim num of weight storageShape must be 5"),
                return false);
    const int64_t expectedNzLastDim = IsFp4(inputParams_.bDtype) ? WEIGHT_NZ_LAST_DIM_FP4 : WEIGHT_NZ_LAST_DIM_FP8;
    OP_CHECK_IF(wStorageShape.GetDim(WEIGHT_NZ_C0_DIM_INDEX) != WEIGHT_NZ_C0_DIM ||
                    wStorageShape.GetDim(WEIGHT_NZ_LAST_DIM_INDEX) != expectedNzLastDim,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "weight storageShape", ShapeToDebugString(wStorageShape),
                    FormatString("when the quantization mode is mx, NZ weight storage tail dims must be 16 and %ld",
                                 expectedNzLastDim)
                        .c_str()),
                return false);
    auto kSize = static_cast<int64_t>(inputParams_.kSize);
    auto nSize = static_cast<int64_t>(inputParams_.nSize);
    auto nzLastDim = expectedNzLastDim;
    auto nzC0Dim = static_cast<int64_t>(WEIGHT_NZ_C0_DIM);
    auto expectedNBlock = inputParams_.transB ? static_cast<int64_t>(Ops::Base::CeilDiv(kSize, nzLastDim)) :
                                                static_cast<int64_t>(Ops::Base::CeilDiv(nSize, nzLastDim));
    auto expectedKBlock = inputParams_.transB ? static_cast<int64_t>(Ops::Base::CeilDiv(nSize, nzC0Dim)) :
                                                static_cast<int64_t>(Ops::Base::CeilDiv(kSize, nzC0Dim));
    OP_CHECK_IF(wStorageShape.GetDim(WEIGHT_NZ_GROUP_DIM_INDEX) != static_cast<int64_t>(inputParams_.groupNum) ||
                    wStorageShape.GetDim(WEIGHT_NZ_BLOCK1_DIM_INDEX) != expectedNBlock ||
                    wStorageShape.GetDim(WEIGHT_NZ_BLOCK2_DIM_INDEX) != expectedKBlock,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "weight storageShape", ShapeToDebugString(wStorageShape),
                    FormatString("weight storageShape first three dims must be [%lu, %ld, %ld]", inputParams_.groupNum,
                                 expectedNBlock, expectedKBlock)),
                return false);
    return true;
}

bool GroupedMatmulActivationQuantTiling950::CheckCoreNum() const
{
    auto compileInfo = context_->GetCompileInfo<GMMCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(inputParams_.opName, "compileInfo is nullptr."), return false);
    OP_CHECK_IF(compileInfo->aicNum == 0,
                OP_LOGE(inputParams_.opName, "aicNum should be positive integer, actual is %u.", compileInfo->aicNum),
                return false);
    OP_CHECK_IF(compileInfo->aivNum != GmmConstant::CORE_RATIO * compileInfo->aicNum,
                OP_LOGE(inputParams_.opName, "aicNum:aivNum should be 1:2, actual %u:%u.", compileInfo->aicNum,
                        compileInfo->aivNum),
                return false);
    return true;
}

bool GroupedMatmulActivationQuantTiling950::AnalyzeInputs()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    auto wStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    auto xScaleStorageShape = context_->GetInputShape(X_SCALE_INDEX);
    auto wScaleStorageShape = context_->GetDynamicInputShape(WEIGHT_SCALE_INDEX, 0);
    OP_CHECK_IF(xStorageShape == nullptr || wStorageShape == nullptr || xScaleStorageShape == nullptr ||
                    wScaleStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "input shape is nullptr."), return false);
    OP_CHECK_IF(!SetGroupNum(GROUP_LIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."), return false);
    const auto &xShape = xStorageShape->GetOriginShape();
    const auto &wShape = wStorageShape->GetOriginShape();
    const auto &weightStorageShape = wStorageShape->GetStorageShape();
    const auto &xScaleShape = xScaleStorageShape->GetOriginShape();
    const auto &wScaleShape = wScaleStorageShape->GetOriginShape();
    const auto &weightScaleStorageShape = wScaleStorageShape->GetStorageShape();
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);
    OP_CHECK_IF(IsFp4(inputParams_.aDtype) &&
                    (inputParams_.kSize % MXFP4_PACK_FACTOR != 0 || inputParams_.kSize == MXFP4_PACK_FACTOR),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "K", std::to_string(inputParams_.kSize),
                                                      "when x and weight are FLOAT4, K must be even and must not be 2"),
                return false);
    OP_CHECK_IF(inputParams_.nSize % N_SIZE_ALIGN != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    inputParams_.opName, "N", std::to_string(inputParams_.nSize),
                    FormatString("N must be a multiple of 64; xOrigin=%s, xStorage=%s, weightOrigin=%s, "
                                 "weightStorage=%s, xScaleOrigin=%s, weightScaleOrigin=%s, weightScaleStorage=%s",
                                 ShapeToDebugString(xShape).c_str(),
                                 ShapeToDebugString(xStorageShape->GetStorageShape()).c_str(),
                                 ShapeToDebugString(wShape).c_str(), ShapeToDebugString(weightStorageShape).c_str(),
                                 ShapeToDebugString(xScaleShape).c_str(), ShapeToDebugString(wScaleShape).c_str(),
                                 ShapeToDebugString(weightScaleStorageShape).c_str())),
                return false);
    inputParams_.aQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
    inputParams_.bQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
    inputParams_.isSingleX = true;
    inputParams_.isSingleW = true;
    inputParams_.isSingleY = true;
    OP_CHECK_IF(!CheckMxScaleShape(xScaleStorageShape->GetOriginShape(), wScaleStorageShape->GetOriginShape()),
                OP_LOGE(inputParams_.opName, "CheckMxScaleShape failed."), return false);
    OP_CHECK_IF(!CheckWeightNzShape(wStorageShape->GetStorageShape()),
                OP_LOGE(inputParams_.opName, "CheckWeightNzShape failed."), return false);
    OP_CHECK_IF(!CheckCoreNum(), OP_LOGE(inputParams_.opName, "CheckCoreNum failed."), return false);
    return true;
}

ge::graphStatus GroupedMatmulActivationQuantTiling950::DoOpTiling()
{
    tilingData_.gmmActivationQuantParams.groupNum = static_cast<uint32_t>(inputParams_.groupNum);
    tilingData_.gmmActivationQuantParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmActivationQuantParams.activationType = activationType_;
    if (yDtype_ == ge::DT_FLOAT8_E5M2) {
        tilingData_.gmmActivationQuantParams.quantDtype = FP8_E5M2_VALUE;
    } else if (yDtype_ == ge::DT_FLOAT4_E2M1) {
        tilingData_.gmmActivationQuantParams.quantDtype = FP4_E2M1_VALUE;
    } else if (yDtype_ == ge::DT_FLOAT4_E1M2) {
        tilingData_.gmmActivationQuantParams.quantDtype = FP4_E1M2_VALUE;
    } else {
        tilingData_.gmmActivationQuantParams.quantDtype = FP8_E4M3FN_VALUE;
    }
    tilingData_.gmmActivationQuantParams.roundMode = roundMode_;
    tilingData_.gmmActivationQuantParams.scaleAlg = scaleAlg_;
    tilingData_.gmmActivationQuantParams.dstTypeMax = dstTypeMax_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulActivationQuantTiling950::DoLibApiTiling()
{
    // Reuse GMM common tiling helpers and keep the output aligned with GroupedQmmBasicApiTiling::DoLibApiTiling.
    // The kernel consumes compact low-level BasicAPI fields instead of TCubeTiling, so scaleKAL1 and scaleKBL1
    // must be identical.
    GroupedQmmTiling::CalBasicBlock();
    const uint64_t alignedBaseM = GroupedMatmul::CeilAlign(basicTiling_.baseM, GmmConstant::CUBE_BLOCK);
    OP_CHECK_IF(basicTiling_.baseN == 0, OP_LOGE(context_->GetNodeName(), "baseN must be greater than zero."),
                return ge::GRAPH_FAILED);
    const uint64_t maxBaseMByUb =
        (MX_EPILOGUE_AIV_COUNT * MX_EPILOGUE_MAX_ELEMENTS_PER_AIV / basicTiling_.baseN / GmmConstant::CUBE_BLOCK) *
        GmmConstant::CUBE_BLOCK;
    OP_CHECK_IF(maxBaseMByUb == 0,
                OP_LOGE(context_->GetNodeName(), "baseN=%lu exceeds the MX epilogue UB capacity.", basicTiling_.baseN),
                return ge::GRAPH_FAILED);
    basicTiling_.baseM = std::min(alignedBaseM, maxBaseMByUb);
    OP_CHECK_IF(GroupedQmmBasicApiTiling::CalL1Tiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed."), return ge::GRAPH_FAILED);
    tilingData_.mmTilingData.m = inputParams_.mSize;
    tilingData_.mmTilingData.n = inputParams_.nSize;
    tilingData_.mmTilingData.k = inputParams_.kSize;
    tilingData_.mmTilingData.baseM = basicTiling_.baseM;
    tilingData_.mmTilingData.baseN = basicTiling_.baseN;
    tilingData_.mmTilingData.baseK = basicTiling_.baseK;
    tilingData_.mmTilingData.kAL1 = basicTiling_.stepKa * basicTiling_.baseK;
    tilingData_.mmTilingData.kBL1 = basicTiling_.stepKb * basicTiling_.baseK;
    const uint64_t scaleKRequested = static_cast<uint64_t>(std::max(basicTiling_.scaleFactorA * basicTiling_.stepKa,
                                                                    basicTiling_.scaleFactorB * basicTiling_.stepKb)) *
                                     basicTiling_.baseK;
    // MX scales address K in 64-element groups.  Round both the requested
    // span and the final K tail to the same unit so A/B scale layout updates
    // cannot stop in the middle of a scale group.
    const uint64_t scaleKL1 =
        std::min(GroupedMatmul::CeilAlign(scaleKRequested, static_cast<uint64_t>(MX_SCALE_GROUP_SIZE)),
                 GroupedMatmul::CeilAlign(inputParams_.kSize, static_cast<uint64_t>(MX_SCALE_GROUP_SIZE)));
    tilingData_.mmTilingData.scaleKAL1 = static_cast<uint32_t>(scaleKL1);
    tilingData_.mmTilingData.scaleKBL1 = static_cast<uint32_t>(scaleKL1);
    tilingData_.mmTilingData.isBias = 0;
    tilingData_.mmTilingData.dbL0C = basicTiling_.dbL0c;
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulActivationQuantTiling950::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transB), static_cast<uint64_t>(inputParams_.transA));
}

ge::graphStatus GroupedMatmulActivationQuantTiling950::GetWorkspaceSize()
{
    OP_CHECK_IF(
        context_ == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("GroupedMatmulActivationQuant", "context", "does not support nullptr"),
        return ge::GRAPH_FAILED);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORKSPACE_SIZES;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulActivationQuantTiling950::PostTiling()
{
    return SaveTilingDataToContext(tilingData_);
}

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulActivationQuant, GroupedMatmulActivationQuantTiling950, 0);

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForGroupedMatmulActivationQuant(gert::TilingParseContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<GMMCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GroupedMatmulActivationQuantTilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(
        context == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("GroupedMatmulActivationQuant", "context", "does not support nullptr"),
        return ge::GRAPH_FAILED);
    std::vector<int32_t> registerList = {0};
    return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
}

IMPL_OP_OPTILING(GroupedMatmulActivationQuant)
    .Tiling(GroupedMatmulActivationQuantTilingFunc)
    .TilingParse<GMMCompileInfo>(TilingPrepareForGroupedMatmulActivationQuant);
} // namespace optiling
