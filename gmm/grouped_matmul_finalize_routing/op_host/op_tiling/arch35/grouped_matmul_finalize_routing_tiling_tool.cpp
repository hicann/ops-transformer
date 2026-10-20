/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_finalize_routing_tiling_tool.h"
#include "common/include/op_host/op_const_def_transformer.h"
#include "log/log.h"
#include "common/include/err/ops_err.h"
#include "../../../../grouped_matmul/op_host/grouped_matmul_host_util.h"

namespace {
constexpr const char* OP_NAME = "GroupedMatmulFinalizeRouting";
} // namespace

namespace optiling {
namespace GMMFRTilingTools {

using namespace GMMFRTilingConstants;
using GmmConstant::GMM_MAX_GROUP_LIST_SIZE;

bool IsMxFp8Fp4(const GQmmInputInfo& inputParams)
{
    return inputParams.aDtype == ge::DT_FLOAT8_E4M3FN && inputParams.bDtype == ge::DT_FLOAT4_E2M1 &&
           inputParams.scaleDtype == ge::DT_FLOAT8_E8M0;
}

bool IsMxFp8Hif4(const GQmmInputInfo& inputParams)
{
    return inputParams.aDtype == ge::DT_FLOAT8_E4M3FN && inputParams.bDtype == Ops::Transformer::DT_HIFLOAT4 &&
           inputParams.scaleDtype == Ops::Transformer::DT_HIFLOAT4_SCALE;
}

ge::graphStatus CheckDescNotNull(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context, "Attrs is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(attrs->GetAttrNum() < ATTR_INDEX_TUNING_CONFIG + 1U,
                OP_LOGE(context, "The num of attrs should not be less than %u, actual is %zu",
                        ATTR_INDEX_TUNING_CONFIG + 1U, attrs->GetAttrNum()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context->GetInputDesc(X_INDEX) == nullptr, OP_LOGE(context, "Input x desc is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetInputDesc(W_INDEX) == nullptr, OP_LOGE(context, "Input w desc is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetInputDesc(SCALE_INDEX) == nullptr, OP_LOGE(context, "Input scale desc is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX) == nullptr,
                OP_LOGE(context, "Input pertoken_scale desc is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetOptionalInputDesc(GROUPLIST_INDEX) == nullptr,
                OP_LOGE(context, "Input group_list desc is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetOptionalInputDesc(ROW_INDEX_INDEX) == nullptr,
                OP_LOGE(context, "Input row_index desc is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetOptionalInputDesc(LOGIT_INDEX) == nullptr, OP_LOGE(context, "Input logit desc is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetOutputDesc(Y_INDEX) == nullptr, OP_LOGE(context, "Output y desc is nullptr."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckShapeNotNull(gert::TilingContext* context)
{
    OP_CHECK_IF(context->GetInputShape(X_INDEX) == nullptr, OP_LOGE(context, "Input x shape is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetInputShape(W_INDEX) == nullptr, OP_LOGE(context, "Input w shape is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetInputShape(SCALE_INDEX) == nullptr, OP_LOGE(context, "Input scale shape is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetOptionalInputShape(PERTOKEN_SCALE_INDEX) == nullptr,
                OP_LOGE(context, "Input pertoken_scale shape is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetOptionalInputShape(GROUPLIST_INDEX) == nullptr,
                OP_LOGE(context, "Input group_list shape is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetOptionalInputShape(ROW_INDEX_INDEX) == nullptr,
                OP_LOGE(context, "Input row_index shape is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetOptionalInputShape(LOGIT_INDEX) == nullptr,
                OP_LOGE(context, "Input logit shape is nullptr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetOutputShape(Y_INDEX) == nullptr, OP_LOGE(context, "Output y shape is nullptr."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetDtype(gert::TilingContext* context, GQmmInputInfo& inputParams)
{
    inputParams.aDtype = context->GetInputDesc(X_INDEX)->GetDataType();

    auto wDesc = context->GetInputDesc(W_INDEX);
    inputParams.bDtype = wDesc->GetDataType();
    inputParams.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));

    inputParams.scaleDtype = context->GetInputDesc(SCALE_INDEX)->GetDataType();
    inputParams.perTokenScaleDtype = context->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX)->GetDataType();
    inputParams.cDtype = context->GetOutputDesc(Y_INDEX)->GetDataType();

    auto biasStorageShape = context->GetOptionalInputShape(BIAS_INDEX);
    inputParams.hasBias = !(biasStorageShape == nullptr || biasStorageShape->GetStorageShape().GetShapeSize() == 0);
    auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
    OP_CHECK_IF(inputParams.hasBias && biasDesc == nullptr,
                OP_LOGE(context, "Bias from tensor is not nullptr, but bias from desc is nullptr."),
                return ge::GRAPH_FAILED);
    inputParams.biasDtype = inputParams.hasBias ? biasDesc->GetDataType() : ge::DT_BF16;

    OP_LOGD(context,
            "GetDtype: aDtype(%s), bDtype(%s), bFormat(%s), scaleDtype(%s), perTokenScaleDtype(%s), "
            "cDtype(%s), hasBias(%d), biasDtype(%s)",
            ge::TypeUtils::DataTypeToSerialString(inputParams.aDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputParams.bDtype).c_str(),
            ge::TypeUtils::FormatToSerialString(inputParams.bFormat).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputParams.scaleDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputParams.perTokenScaleDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(inputParams.cDtype).c_str(), static_cast<int>(inputParams.hasBias),
            ge::TypeUtils::DataTypeToSerialString(inputParams.biasDtype).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetAttrs(gert::TilingContext* context, GQmmInputInfo& inputParams)
{
    auto attrs = context->GetAttrs();

    const bool* transposeXPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_X);
    const bool* transposeWPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_W);
    const int64_t* groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);

    inputParams.transA = transposeXPtr != nullptr ? *transposeXPtr : false;
    inputParams.transB = transposeWPtr != nullptr ? *transposeWPtr : false;
    inputParams.groupType = SPLIT_M;
    int64_t groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : 1;
    OP_CHECK_IF(groupListType != 0 && groupListType != 1,
                OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "group_list_type", std::to_string(groupListType), "0 or 1"),
                return ge::GRAPH_FAILED);
    inputParams.groupListType = static_cast<int8_t>(groupListType);
    OP_LOGD(context, "GetAttrs: transA(%d), transB(%d), groupType(%d), groupListType(%d)",
            static_cast<int>(inputParams.transA), static_cast<int>(inputParams.transB),
            static_cast<int>(inputParams.groupType), static_cast<int>(inputParams.groupListType));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetInputs(gert::TilingContext* context, GQmmInputInfo& inputParams)
{
    const gert::Shape& xShape = context->GetInputShape(X_INDEX)->GetOriginShape();
    const gert::Shape& wShape = context->GetInputShape(W_INDEX)->GetOriginShape();

    size_t xDimNum = xShape.GetDimNum();
    size_t wDimNum = wShape.GetDimNum();
    if (unlikely(xDimNum != DIM_NUM_2)) {
        std::string xDimNumStr = std::to_string(xDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "x", xDimNumStr.c_str(), "the shape dim of x must be 2");
        return ge::GRAPH_FAILED;
    }
    if (unlikely(wDimNum != DIM_NUM_3)) {
        std::string wDimNumStr = std::to_string(wDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "weight", wDimNumStr.c_str(),
                                                 "the shape dim of weight must be 3");
        return ge::GRAPH_FAILED;
    }
    inputParams.mSize = inputParams.transA ? xShape.GetDim(xDimNum - LAST_FIRST_DIM_INDEX) :
                                             xShape.GetDim(xDimNum - LAST_SECOND_DIM_INDEX);
    inputParams.kSize = inputParams.transA ? xShape.GetDim(xDimNum - LAST_SECOND_DIM_INDEX) :
                                             xShape.GetDim(xDimNum - LAST_FIRST_DIM_INDEX);
    inputParams.nSize = inputParams.transB ? wShape.GetDim(wDimNum - LAST_SECOND_DIM_INDEX) :
                                             wShape.GetDim(wDimNum - LAST_FIRST_DIM_INDEX);

    const gert::Shape& groupListShape = context->GetOptionalInputShape(GROUPLIST_INDEX)->GetStorageShape();
    if (unlikely(groupListShape.GetDimNum() != 1)) {
        std::string groupListDimNumStr = std::to_string(groupListShape.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "group_list", groupListDimNumStr.c_str(),
                                                 "the shape dim of group_list must be 1");
        return ge::GRAPH_FAILED;
    }
    inputParams.groupNum = static_cast<uint64_t>(groupListShape.GetDim(0));
    if (unlikely(inputParams.groupNum > GmmConstant::GMM_MAX_GROUP_LIST_SIZE)) {
        std::string groupNumStr = std::to_string(inputParams.groupNum);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "group_list", groupNumStr.c_str(),
                                              "the value of group_list must be <= 1024");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "GetInputs: mSize(%lu), kSize(%lu), nSize(%lu), groupNum(%lu)", inputParams.mSize,
            inputParams.kSize, inputParams.nSize, inputParams.groupNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetFrInfo(gert::TilingContext* context, const GQmmInputInfo& inputParams, GMMFRExtraInfo& frInfo)
{
    // scaleType：由 scaleDtype 派生（0=e8m0, 1=fp32, 2=bf16）
    frInfo.scaleType = 0;
    if (inputParams.scaleDtype == ge::DT_FLOAT) {
        frInfo.scaleType = 1;
    } else if (inputParams.scaleDtype == ge::DT_BF16) {
        frInfo.scaleType = 2; // 2 represents bf16 dtype
    }

    // rowIndexType：由 rowIndexDtype 派生（0=int64, 1=int32）
    frInfo.rowIndexType = (context->GetOptionalInputDesc(ROW_INDEX_INDEX)->GetDataType() == ge::DT_INT32) ? 1 : 0;

    auto attrs = context->GetAttrs();
    const int64_t* outputDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DTYPE);
    const int64_t* outputBSPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_OUTPUT_BS);
    frInfo.outputBs = outputBSPtr != nullptr ? *outputBSPtr : 0;
    frInfo.outputDtype = outputDtypePtr != nullptr ? *outputDtypePtr : 0;

    // sharedInput 相关参数
    auto sharedInputShape = context->GetOptionalInputShape(SHARE_INPUT_INDEX);
    frInfo.hasSharedInput = !(sharedInputShape == nullptr || sharedInputShape->GetStorageShape().GetShapeSize() == 0);
    frInfo.sharedInputLen = frInfo.hasSharedInput ? sharedInputShape->GetStorageShape()[0] : 0;
    if (frInfo.hasSharedInput) {
        auto sharedInputDesc = context->GetOptionalInputDesc(SHARE_INPUT_INDEX);
        OP_CHECK_IF(sharedInputDesc == nullptr,
                    OP_LOGE(context, "shared_input tensor is not nullptr, but shared_input desc is nullptr."),
                    return ge::GRAPH_FAILED);

        const float* shareInputWeightPtr = attrs->GetAttrPointer<float>(ATTR_INDEX_SHARE_INPUT_WEIGHT);
        OP_CHECK_IF(shareInputWeightPtr == nullptr,
                    OP_LOGE(context, "Attr shared_input_weight cannot be nullptr when shared_input is provided."),
                    return ge::GRAPH_FAILED);
        frInfo.sharedInputWeight = *shareInputWeightPtr;

        const int64_t* shareInputOffsetPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SHARE_INPUT_OFFSET);
        OP_CHECK_IF(shareInputOffsetPtr == nullptr,
                    OP_LOGE(context, "Attr shared_input_offset cannot be nullptr when shared_input is provided."),
                    return ge::GRAPH_FAILED);
        frInfo.sharedInputOffset = *shareInputOffsetPtr;
    } else {
        frInfo.sharedInputWeight = 1.0f;
        frInfo.sharedInputOffset = 0;
    }
    OP_LOGD(context,
            "GetFrInfo: scaleType(%d), rowIndexType(%d), outputBs(%ld), outputDtype(%ld), "
            "hasSharedInput(%d), sharedInputLen(%ld), sharedInputWeight(%f), sharedInputOffset(%ld)",
            frInfo.scaleType, frInfo.rowIndexType, frInfo.outputBs, frInfo.outputDtype,
            static_cast<int>(frInfo.hasSharedInput), frInfo.sharedInputLen, frInfo.sharedInputWeight,
            frInfo.sharedInputOffset);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetInputsInfo(gert::TilingContext* context, GQmmInputInfo& inputParams, GMMFRExtraInfo& frInfo)
{
    OP_CHECK_IF(CheckDescNotNull(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckDescNotNull failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeNotNull(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckShapeNotNull failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetDtype(context, inputParams) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetDtype failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAttrs(context, inputParams) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetAttrs failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputs(context, inputParams) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetInputs failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetFrInfo(context, inputParams, frInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetFrInfo failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool CheckInputTensorShape(const char* name, const gert::Shape& shape, const std::vector<int64_t>& expectedShape)
{
    auto shapeLen = shape.GetDimNum();
    if (unlikely(shapeLen != expectedShape.size())) {
        std::string dimVal = std::to_string(shapeLen);
        std::string dimReason =
            "the shape dim of " + std::string(name) + " must be " + std::to_string(expectedShape.size());
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, name, dimVal.c_str(), dimReason.c_str());
        return false;
    }
    size_t i = 0;
    for (auto dim : expectedShape) {
        if (unlikely(dim != shape.GetDim(i++))) {
            std::string shapeStr = Ops::Base::ToString(shape);
            std::string expectedStr;
            for (size_t j = 0; j < expectedShape.size(); ++j) {
                expectedStr += (j == 0 ? "" : ", ") + std::to_string(expectedShape[j]);
            }
            std::string reason = "the shape of " + std::string(name) + " must be [" + expectedStr + "]";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, name, shapeStr.c_str(), reason.c_str());
            return false;
        }
    }
    return true;
}

bool CheckCommonInputsDtype(const GMMFRExtraInfo& frInfo, gert::TilingContext* context)
{
    auto groupListDesc = context->GetOptionalInputDesc(GROUPLIST_INDEX);
    if (unlikely(groupListDesc->GetDataType() != ge::DT_INT64)) {
        std::string glDtypeStr = ge::TypeUtils::DataTypeToSerialString(groupListDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "group_list", glDtypeStr.c_str(),
                                              "the dtype of group_list must be INT64");
        return false;
    }
    if (frInfo.hasSharedInput) {
        auto sharedInputDesc = context->GetOptionalInputDesc(SHARE_INPUT_INDEX);
        if (unlikely(sharedInputDesc->GetDataType() != ge::DT_BF16)) {
            std::string siDtypeStr = ge::TypeUtils::DataTypeToSerialString(sharedInputDesc->GetDataType());
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "shared_input", siDtypeStr.c_str(),
                                                  "the dtype of shared_input must be BF16");
            return false;
        }
    }
    auto logitDesc = context->GetOptionalInputDesc(LOGIT_INDEX);
    if (unlikely(logitDesc->GetDataType() != ge::DT_FLOAT)) {
        std::string logitDtypeStr = ge::TypeUtils::DataTypeToSerialString(logitDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "logit", logitDtypeStr.c_str(),
                                              "the dtype of logit must be FLOAT");
        return false;
    }
    auto rowIndexDesc = context->GetOptionalInputDesc(ROW_INDEX_INDEX);
    if (unlikely(rowIndexDesc->GetDataType() != ge::DT_INT64)) {
        std::string riDtypeStr = ge::TypeUtils::DataTypeToSerialString(rowIndexDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "row_index", riDtypeStr.c_str(),
                                              "the dtype of row_index must be INT64");
        return false;
    }
    return true;
}

bool CheckFinalizeRoutingInputsShape(const GQmmInputInfo& inputParams, const GMMFRExtraInfo& frInfo,
                                     gert::TilingContext* context)
{
    const gert::Shape& rowIndexShape = context->GetOptionalInputShape(ROW_INDEX_INDEX)->GetStorageShape();
    if (!CheckInputTensorShape("row_index", rowIndexShape, {static_cast<int64_t>(inputParams.mSize)})) {
        return false;
    }

    const gert::Shape& logitShape = context->GetOptionalInputShape(LOGIT_INDEX)->GetStorageShape();
    if (!CheckInputTensorShape("logit", logitShape, {static_cast<int64_t>(inputParams.mSize)})) {
        return false;
    }

    if (unlikely(frInfo.outputBs < 0)) {
        std::string outputBsStr = std::to_string(frInfo.outputBs);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "output_bs", outputBsStr.c_str(),
                                              "the value of output_bs must be >= 0");
        return false;
    }
    if (unlikely(static_cast<uint64_t>(frInfo.outputBs) > inputParams.mSize)) {
        std::string outputBsStr = std::to_string(frInfo.outputBs);
        std::string reason = "the value of output_bs must be <= m(" + std::to_string(inputParams.mSize) + ")";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "output_bs", outputBsStr.c_str(), reason.c_str());
        return false;
    }
    return true;
}

bool CheckSharedInput(const GQmmInputInfo& inputParams, const GMMFRExtraInfo& frInfo, gert::TilingContext* context)
{
    if (!frInfo.hasSharedInput) {
        return true;
    }
    if (unlikely(frInfo.sharedInputOffset < 0)) {
        std::string sharedInputOffsetStr = std::to_string(frInfo.sharedInputOffset);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "shared_input_offset", sharedInputOffsetStr.c_str(),
                                              "the value of shared_input_offset must be >= 0");
        return false;
    }
    if (unlikely(frInfo.sharedInputOffset > frInfo.outputBs)) {
        std::string sharedInputOffsetAndOutputBsStr =
            std::to_string(frInfo.sharedInputOffset) + ", " + std::to_string(frInfo.outputBs);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(OP_NAME, "shared_input_offset, output_bs",
                                               sharedInputOffsetAndOutputBsStr.c_str(),
                                               "the value of shared_input_offset must be <= the value of output_bs");
        return false;
    }
    if (unlikely(frInfo.sharedInputLen > frInfo.outputBs)) {
        std::string sharedInputLenAndOutputBsStr =
            std::to_string(frInfo.sharedInputLen) + ", " + std::to_string(frInfo.outputBs);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(OP_NAME, "shared_input_len, output_bs",
                                               sharedInputLenAndOutputBsStr.c_str(),
                                               "the value of shared_input_len must be <= the value of output_bs");
        return false;
    }
    if (unlikely(frInfo.sharedInputOffset + frInfo.sharedInputLen > frInfo.outputBs)) {
        std::string sharedInputOffsetLenAndOutputBsStr = std::to_string(frInfo.sharedInputOffset) + ", " +
                                                         std::to_string(frInfo.sharedInputLen) + ", " +
                                                         std::to_string(frInfo.outputBs);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "shared_input_offset, shared_input_len, output_bs", sharedInputOffsetLenAndOutputBsStr.c_str(),
            "the value of shared_input_offset + shared_input_len must be <= the value of output_bs");
        return false;
    }

    const gert::Shape& sharedInputShape = context->GetOptionalInputShape(SHARE_INPUT_INDEX)->GetStorageShape();
    if (!CheckInputTensorShape("shared_input", sharedInputShape,
                               {frInfo.sharedInputLen, static_cast<int64_t>(inputParams.nSize)})) {
        return false;
    }
    return true;
}

} // namespace GMMFRTilingTools
} // namespace optiling
