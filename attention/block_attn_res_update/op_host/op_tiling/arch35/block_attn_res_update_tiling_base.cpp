/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "block_attn_res_update_tiling_base.h"

#include <array>
#include <cmath>
#include <graph/utils/type_utils.h>
#include <limits>
#include <string>
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace block_attn_res_update {
namespace {

constexpr const char *OP_NAME = "BlockAttnResUpdate";
constexpr uint32_t INPUT_PARTIAL_BLOCK = 0U;
constexpr uint32_t INPUT_DELTA = 1U;
constexpr uint32_t INPUT_PSEUDO_QUERY = 2U;
constexpr uint32_t INPUT_NUMERATOR = 3U;
constexpr uint32_t INPUT_LOGIT_MAX = 4U;
constexpr uint32_t INPUT_EXP_SUM = 5U;
constexpr uint32_t OUTPUT_PARTIAL_BLOCK = 0U;
constexpr uint32_t OUTPUT_H = 1U;
constexpr uint32_t ATTR_EPS = 0U;
constexpr float DEFAULT_EPS = 1e-6F;
constexpr int64_t MAX_D_SIZE = 8192;

constexpr uint32_t MATRIX_DIM_NUM = 2U;
constexpr uint32_t VECTOR_DIM_NUM = 1U;
constexpr size_t INPUT_COUNT = static_cast<size_t>(INPUT_EXP_SUM) + 1U;

constexpr std::array<const char *, INPUT_COUNT> INPUT_NAMES = {
    "partial_block", "delta", "pseudo_query", "numerator", "logit_max", "exp_sum",
};
constexpr std::array<ge::DataType, INPUT_COUNT> EXPECTED_DTYPES = {
    ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
};
constexpr const char *REFERENCE_OUTPUT_PARTIAL_BLOCK_NAME = "reference output partial_block";

struct TensorContextInfo {
    std::array<const gert::StorageShape *, INPUT_COUNT> inputShapes{};
    const gert::StorageShape *partialBlockOutputStorageShape = nullptr;
    const gert::StorageShape *hStorageShape = nullptr;
};

bool SameShape(const gert::Shape &lhs, const gert::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

std::string ShapeToString(const gert::Shape &shape)
{
    std::string shapeStr = "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0U) {
            shapeStr += ", ";
        }
        shapeStr += std::to_string(shape.GetDim(i));
    }
    shapeStr += "]";
    return shapeStr;
}

bool CheckShapeDimNum(const char *opName, const char *tensorName, const gert::Shape &shape, uint32_t expectedDimNum)
{
    const size_t actualDimNum = shape.GetDimNum();
    if (actualDimNum == expectedDimNum) {
        return true;
    }

    const std::string actualDimNumStr = std::to_string(actualDimNum) + "D";
    const std::string reason =
        "The shape dim of " + std::string(tensorName) + " must be " + std::to_string(expectedDimNum) + "D";
    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, tensorName, actualDimNumStr.c_str(), reason.c_str());
    return false;
}

bool CheckShapeMatches(const char *opName, const char *tensorName, const gert::Shape &shape, const char *referenceName,
                       const gert::Shape &referenceShape)
{
    if (SameShape(shape, referenceShape)) {
        return true;
    }

    const std::string shapeStr = ShapeToString(shape);
    const std::string reason = "The shape of " + std::string(tensorName) + " must be the same as " + referenceName +
                               ", whose shape is " + ShapeToString(referenceShape);
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, tensorName, shapeStr.c_str(), reason.c_str());
    return false;
}

bool CheckVectorLength(const char *opName, const char *tensorName, const gert::Shape &shape, int64_t expectedLength,
                       const char *axisName)
{
    if (shape.GetDim(0) == expectedLength) {
        return true;
    }

    const std::string shapeStr = ShapeToString(shape);
    const std::string reason = "The shape of " + std::string(tensorName) + " must be [" +
                               std::to_string(expectedLength) + "] to match the " + axisName +
                               " dimension of partial_block";
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, tensorName, shapeStr.c_str(), reason.c_str());
    return false;
}

bool CheckTensorDtype(const char *opName, const char *tensorName, ge::DataType actualDtype, ge::DataType expectedDtype)
{
    if (actualDtype == expectedDtype) {
        return true;
    }

    const std::string actualDtypeStr = ge::TypeUtils::DataTypeToSerialString(actualDtype);
    const std::string expectedDtypeStr = ge::TypeUtils::DataTypeToSerialString(expectedDtype);
    const std::string reason = "The dtype of " + std::string(tensorName) + " must be " + expectedDtypeStr;
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, tensorName, actualDtypeStr.c_str(), reason.c_str());
    return false;
}

bool CheckTensorNdFormats(const char *opName, const char *tensorName, const gert::CompileTimeTensorDesc &tensorDesc)
{
    const ge::Format originFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(tensorDesc.GetOriginFormat()));
    const ge::Format storageFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(tensorDesc.GetStorageFormat()));
    if (originFormat == ge::FORMAT_ND && storageFormat == ge::FORMAT_ND) {
        return true;
    }

    const std::string actualFormat = "origin=" + ge::TypeUtils::FormatToSerialString(originFormat) +
                                     ", storage=" + ge::TypeUtils::FormatToSerialString(storageFormat);
    const std::string reason = "The origin and storage formats of " + std::string(tensorName) + " must both be ND";
    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(opName, tensorName, actualFormat.c_str(), reason.c_str());
    return false;
}

bool CheckStorageShapeMatchesOrigin(const char *opName, const char *tensorName, const gert::StorageShape &storageShape)
{
    const gert::Shape &originShape = storageShape.GetOriginShape();
    const gert::Shape &physicalShape = storageShape.GetStorageShape();
    if (SameShape(physicalShape, originShape)) {
        return true;
    }

    const std::string physicalShapeStr = ShapeToString(physicalShape);
    const std::string reason = "The storage shape of " + std::string(tensorName) +
                               " must be the same as its origin shape " + ShapeToString(originShape);
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, tensorName, physicalShapeStr.c_str(), reason.c_str());
    return false;
}

bool CheckInputMetadata(gert::TilingContext *context, const char *opName, TensorContextInfo &tensorInfo)
{
    for (uint32_t i = INPUT_PARTIAL_BLOCK; i <= INPUT_EXP_SUM; ++i) {
        tensorInfo.inputShapes[i] = context->GetInputShape(i);
        OP_CHECK_IF(tensorInfo.inputShapes[i] == nullptr, OP_LOGE(opName, "Input %s shape is nullptr.", INPUT_NAMES[i]),
                    return false);

        const auto *inputDesc = context->GetInputDesc(i);
        OP_CHECK_IF(inputDesc == nullptr, OP_LOGE(opName, "Input %s desc is nullptr.", INPUT_NAMES[i]), return false);
        if (!CheckTensorDtype(opName, INPUT_NAMES[i], inputDesc->GetDataType(), EXPECTED_DTYPES[i])) {
            return false;
        }
        if (!CheckTensorNdFormats(opName, INPUT_NAMES[i], *inputDesc)) {
            return false;
        }
    }
    return true;
}

bool CheckOutputMetadata(gert::TilingContext *context, const char *opName, TensorContextInfo &tensorInfo)
{
    tensorInfo.partialBlockOutputStorageShape = context->GetOutputShape(OUTPUT_PARTIAL_BLOCK);
    OP_CHECK_IF(tensorInfo.partialBlockOutputStorageShape == nullptr,
                OP_LOGE(opName, "Reference output partial_block shape is nullptr."), return false);
    const auto *partialBlockOutputDesc = context->GetOutputDesc(OUTPUT_PARTIAL_BLOCK);
    OP_CHECK_IF(partialBlockOutputDesc == nullptr, OP_LOGE(opName, "Reference output partial_block desc is nullptr."),
                return false);
    if (!CheckTensorDtype(opName, REFERENCE_OUTPUT_PARTIAL_BLOCK_NAME, partialBlockOutputDesc->GetDataType(),
                          ge::DT_FLOAT)) {
        return false;
    }

    tensorInfo.hStorageShape = context->GetOutputShape(OUTPUT_H);
    OP_CHECK_IF(tensorInfo.hStorageShape == nullptr, OP_LOGE(opName, "Output h shape is nullptr."), return false);
    const auto *hDesc = context->GetOutputDesc(OUTPUT_H);
    OP_CHECK_IF(hDesc == nullptr, OP_LOGE(opName, "Output h desc is nullptr."), return false);
    if (!CheckTensorDtype(opName, "h", hDesc->GetDataType(), ge::DT_BF16)) {
        return false;
    }
    return true;
}

bool CheckStorageShapes(const char *opName, const TensorContextInfo &tensorInfo)
{
    for (size_t i = 0U; i < INPUT_COUNT; ++i) {
        if (!CheckStorageShapeMatchesOrigin(opName, INPUT_NAMES[i], *tensorInfo.inputShapes[i])) {
            return false;
        }
    }
    return CheckStorageShapeMatchesOrigin(opName, REFERENCE_OUTPUT_PARTIAL_BLOCK_NAME,
                                          *tensorInfo.partialBlockOutputStorageShape) &&
           CheckStorageShapeMatchesOrigin(opName, "h", *tensorInfo.hStorageShape);
}

bool CheckDimNums(const char *opName, const TensorContextInfo &tensorInfo)
{
    return CheckShapeDimNum(opName, "partial_block", tensorInfo.inputShapes[INPUT_PARTIAL_BLOCK]->GetOriginShape(),
                            MATRIX_DIM_NUM) &&
           CheckShapeDimNum(opName, "delta", tensorInfo.inputShapes[INPUT_DELTA]->GetOriginShape(), MATRIX_DIM_NUM) &&
           CheckShapeDimNum(opName, "pseudo_query", tensorInfo.inputShapes[INPUT_PSEUDO_QUERY]->GetOriginShape(),
                            VECTOR_DIM_NUM) &&
           CheckShapeDimNum(opName, "numerator", tensorInfo.inputShapes[INPUT_NUMERATOR]->GetOriginShape(),
                            MATRIX_DIM_NUM) &&
           CheckShapeDimNum(opName, "logit_max", tensorInfo.inputShapes[INPUT_LOGIT_MAX]->GetOriginShape(),
                            VECTOR_DIM_NUM) &&
           CheckShapeDimNum(opName, "exp_sum", tensorInfo.inputShapes[INPUT_EXP_SUM]->GetOriginShape(),
                            VECTOR_DIM_NUM) &&
           CheckShapeDimNum(opName, REFERENCE_OUTPUT_PARTIAL_BLOCK_NAME,
                            tensorInfo.partialBlockOutputStorageShape->GetOriginShape(), MATRIX_DIM_NUM) &&
           CheckShapeDimNum(opName, "h", tensorInfo.hStorageShape->GetOriginShape(), MATRIX_DIM_NUM);
}

bool CheckRelatedShapes(const char *opName, const TensorContextInfo &tensorInfo)
{
    const gert::Shape &partialShape = tensorInfo.inputShapes[INPUT_PARTIAL_BLOCK]->GetOriginShape();
    return CheckShapeMatches(opName, "delta", tensorInfo.inputShapes[INPUT_DELTA]->GetOriginShape(), "partial_block",
                             partialShape) &&
           CheckShapeMatches(opName, "numerator", tensorInfo.inputShapes[INPUT_NUMERATOR]->GetOriginShape(),
                             "partial_block", partialShape) &&
           CheckShapeMatches(opName, REFERENCE_OUTPUT_PARTIAL_BLOCK_NAME,
                             tensorInfo.partialBlockOutputStorageShape->GetOriginShape(), "partial_block",
                             partialShape) &&
           CheckShapeMatches(opName, "h", tensorInfo.hStorageShape->GetOriginShape(), "partial_block", partialShape);
}

bool CheckDimSizes(const char *opName, const TensorContextInfo &tensorInfo, uint64_t &tSize, uint32_t &dSize)
{
    const gert::Shape &partialShape = tensorInfo.inputShapes[INPUT_PARTIAL_BLOCK]->GetOriginShape();
    const int64_t tDim = partialShape.GetDim(0);
    const int64_t dDim = partialShape.GetDim(1);
    OP_CHECK_IF(tDim <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "partial_block dimension T", std::to_string(tDim).c_str(),
                                                      "The T dimension of partial_block must be greater than 0"),
                return false);
    OP_CHECK_IF(dDim <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "partial_block dimension D", std::to_string(dDim).c_str(),
                                                      "The D dimension of partial_block must be greater than 0"),
                return false);
    if (dDim > MAX_D_SIZE) {
        const std::string dDimStr = std::to_string(dDim);
        const std::string reason =
            "The D dimension of partial_block must be in the range [1, " + std::to_string(MAX_D_SIZE) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "partial_block dimension D", dDimStr.c_str(), reason.c_str());
        return false;
    }
    OP_CHECK_IF(tDim > std::numeric_limits<int64_t>::max() / dDim,
                OP_LOGE(opName, "T * D exceeds the signed 64-bit kernel GM offset range, T=%ld, D=%ld.", tDim, dDim),
                return false);

    const gert::Shape &logitMaxShape = tensorInfo.inputShapes[INPUT_LOGIT_MAX]->GetOriginShape();
    const gert::Shape &expSumShape = tensorInfo.inputShapes[INPUT_EXP_SUM]->GetOriginShape();
    const gert::Shape &pseudoQueryShape = tensorInfo.inputShapes[INPUT_PSEUDO_QUERY]->GetOriginShape();
    if (!CheckVectorLength(opName, "logit_max", logitMaxShape, tDim, "T") ||
        !CheckVectorLength(opName, "exp_sum", expSumShape, tDim, "T") ||
        !CheckVectorLength(opName, "pseudo_query", pseudoQueryShape, dDim, "D")) {
        return false;
    }

    tSize = static_cast<uint64_t>(tDim);
    dSize = static_cast<uint32_t>(dDim);
    return true;
}

bool CheckRawTilingData(gert::TilingContext *context, const char *opName)
{
    auto *rawTilingData = context->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr, OP_LOGE(opName, "Raw tiling data is nullptr."), return false);
    OP_CHECK_IF(rawTilingData->GetData() == nullptr, OP_LOGE(opName, "Raw tiling data buffer is nullptr."),
                return false);
    return true;
}

} // namespace

ge::graphStatus BlockAttnResUpdateTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResUpdateTilingBase::GetWorkspaceSize()
{
    // Publish a zero-size workspace entry through the common tiling flow; the kernel does not access workspace.
    workspaceSize_ = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResUpdateTilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(OP_NAME, "Context is nullptr."), return ge::GRAPH_FAILED);
    opName_ = context_->GetNodeName();
    OP_CHECK_IF(opName_ == nullptr, OP_LOGE(OP_NAME, "Node name is nullptr."), return ge::GRAPH_FAILED);
    auto ret = CheckContext();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    const auto *attrs = context_->GetAttrs();
    const float *eps = attrs == nullptr ? nullptr : attrs->GetAttrPointer<float>(ATTR_EPS);
    const float epsValue = eps == nullptr ? DEFAULT_EPS : *eps;
    if (!std::isfinite(epsValue) || epsValue <= 0.0F) {
        const std::string epsValueStr = std::to_string(epsValue);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "eps", epsValueStr.c_str(),
                                              "The value of eps must be finite and greater than 0");
        return ge::GRAPH_FAILED;
    }
    eps_ = epsValue;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResUpdateTilingBase::CheckContext()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(OP_NAME, "Context is nullptr."), return ge::GRAPH_FAILED);

    TensorContextInfo tensorInfo;
    if (!CheckInputMetadata(context_, opName_, tensorInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckOutputMetadata(context_, opName_, tensorInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckStorageShapes(opName_, tensorInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckDimNums(opName_, tensorInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckRelatedShapes(opName_, tensorInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckDimSizes(opName_, tensorInfo, tSize_, dSize_)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckRawTilingData(context_, opName_)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResUpdateTilingBase::GetPlatformInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(OP_NAME, "Context is nullptr."), return ge::GRAPH_FAILED);
    auto *platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName_, "PlatformInfo is nullptr."), return ge::GRAPH_FAILED);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    aivNum_ = platform.GetCoreNumAiv();
    OP_CHECK_IF(aivNum_ == 0U || ubSize_ == 0U,
                OP_LOGE(opName_, "Invalid platform data: aivNum=%u, ubSize=%lu.", aivNum_, ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace block_attn_res_update
} // namespace optiling
