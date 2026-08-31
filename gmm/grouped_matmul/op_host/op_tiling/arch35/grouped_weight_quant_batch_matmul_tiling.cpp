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
 * \file grouped_weight_quant_batch_matmul_tiling.cpp
 * \brief
 */
#include "grouped_weight_quant_batch_matmul_tiling.h"
#include <limits>
#include "op_host/tiling_templates_registry.h"
#include "../../../op_kernel/arch35/weight_quant_basic_block/weight_quant_tiling_key.h"

namespace {
static constexpr const char *OP_NAME = "GroupedMatmul";
constexpr size_t LAST_DIM = 1;             // 倒数第1维
constexpr size_t FOURTH_FROM_LAST_DIM = 4; // 倒数第4维
constexpr uint32_t ND_WEIGHT_DIM_NUM = 2;  // ND weight形如(K, N)
constexpr uint32_t NZ_WEIGHT_DIM_NUM = 4;  // NZ weight为4维，具体轴顺序由transB_决定
} // namespace

enum class GmmTrans {
    NoTrans = 0,
    BTrans = 1,
    ATrans = 2,
    ABTrans = 3
};
namespace optiling {

static const std::map<ge::DataType, std::unordered_set<ge::DataType>> BIAS_TYPE_SUPPORT_MAP = {
    {ge::DT_FLOAT16, {ge::DT_FLOAT16}}, {ge::DT_BF16, {ge::DT_BF16, ge::DT_FLOAT}}};

bool GroupedWeightQuantBatchMatmulTiling::GetDimFromEnd(const gert::Shape &shape, size_t posFromEnd,
                                                        uint64_t &out) const
{
    OP_CHECK_IF(posFromEnd == 0 || posFromEnd > shape.GetDimNum(),
                OP_LOGE(OP_NAME, "posFromEnd[%zu] invalid, dimNum[%zu].", posFromEnd, shape.GetDimNum()), return false);
    int64_t v = shape.GetDim(static_cast<int64_t>(shape.GetDimNum() - posFromEnd));
    OP_CHECK_IF(v < 0, OP_LOGE(OP_NAME, "dim value[%ld] is negative.", v), return false);
    out = static_cast<uint64_t>(v);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckCoreNum(const gert::TilingContext *context) const
{
    OP_CHECK_IF(
        coreNum_ <= 0 || aivNum_ <= 0,
        OP_LOGE(context->GetNodeName(), "Invalid aicNum[%u] or aivNum[%u], expect greater than 0", coreNum_, aivNum_),
        return false);

    OP_CHECK_IF(coreNum_ * AIC_AIV_CORE_RATIO != aivNum_,
                OP_LOGE(context->GetNodeName(),
                        "Invalid cube/vector core ratio. Expected cube:vector = 1:%u, "
                        "but got cube=%u, vector=%u",
                        AIC_AIV_CORE_RATIO, coreNum_, aivNum_),
                return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::SetTiling(gert::TilingContext *context)
{
    OP_CHECK_IF(!AnalyzeAttr(context), OP_LOGE(context->GetNodeName(), "Invalid attr param"), return false);
    OP_CHECK_IF(!CalcResplitTiling(context), OP_LOGE(context->GetNodeName(), "Unable to calculate resplit-tiling"),
                return false);
    auto ret = SetBaseTiling();
    if (!ret) {
        OP_LOGE(context->GetNodeName(), "Set Base Tiling Failed");
        return false;
    }
    SetMatMulTiling();
    SetTilingKey(context);
    OP_CHECK_IF(!SetCustomParam(context), OP_LOGE(context->GetNodeName(), "Unable to set custom param"), return false);
    PrintTilingResult(context);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::SetShapeList(const gert::TilingContext *context)
{
    if (isSingleX_ && isSingleWeight_ && isSingleY_) {
        OP_CHECK_IF(!SetShapeListSplitMSingleXSingleWeightSingleY(context),
                    OP_LOGE(context->GetNodeName(), "Unable to get shape list"), return false);
    } else if (!isSingleX_ && !isSingleWeight_ && !isSingleY_) {
        OP_CHECK_IF(!SetShapeListMultiXMultiWeightMultiY(context),
                    OP_LOGE(context->GetNodeName(), "Unable to get MMM shape list"), return false);
    } else if (isSingleMultiSingle_) {
        OP_CHECK_IF(
            !SetShapeListSplitMSingleXMultiWeightSingleY(context),
            OP_LOGE(context->GetNodeName(),
                    "Unable to get single-multi-single shape list when x-weight is float8_e4m3fn-float4_e2m1/fp32"),
            return false);
    } else {
        std::string incorrectValue = "groupType: " + std::to_string(static_cast<int32_t>(groupType_)) +
                                     ", singleX: " + (isSingleX_ ? "true" : "false") +
                                     ", singleW: " + (isSingleWeight_ ? "true" : "false") +
                                     ", singleY: " + (isSingleY_ ? "true" : "false");
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "groupType", incorrectValue,
                                              "Only support single-single-single or multi-multi-multi mode");
        return false;
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckTensorListSize() const
{
    OP_CHECK_IF(numX_ >= GroupedMatmul::MAX_TENSOR_CONT,
                OP_LOGE_FOR_INVALID_LISTSIZE(OP_NAME, "x", std::to_string(numX_),
                                             "In multi-multi-multi scenario, listsize of x must <= 128"),
                return false);
    if (groupType_ == static_cast<int64_t>(GroupType::NO_SPLIT)) {
        OP_CHECK_IF(numX_ != numWeight_,
                    OP_LOGE_FOR_INVALID_LISTSIZE(
                        OP_NAME, "x, weight", std::to_string(numX_) + ", " + std::to_string(numWeight_),
                        "When groupType is -1 (no split), the sizes of x and weight must be all the same"),
                    return false);
    } else if (isSingleMultiSingle_) {
        OP_CHECK_IF(numX_ != 1 || numWeight_ < 1,
                    OP_LOGE_FOR_INVALID_LISTSIZE(
                        OP_NAME, "x, weight", std::to_string(numX_) + ", " + std::to_string(numWeight_),
                        "When x-weight is float8_e4m3fn-float4_e2m1/fp32 in single-multi-single mode, the size of x "
                        "must be 1 and weight must be greater than or equal to 1"),
                    return false);
    } else {
        OP_CHECK_IF(numX_ != 1 || numWeight_ != 1,
                    OP_LOGE_FOR_INVALID_LISTSIZE(
                        OP_NAME, "x, weight", std::to_string(numX_) + ", " + std::to_string(numWeight_),
                        "When groupType is 0 (split M), the sizes of x and weight should all be 1"),
                    return false);
    }
    uint16_t expectedParamNum = numWeight_;
    OP_CHECK_IF(numAntiquantScale_ != expectedParamNum,
                OP_LOGE_FOR_INVALID_LISTSIZE(OP_NAME, "antiquantScaleOptional", std::to_string(numAntiquantScale_),
                                             "The size of antiquantScaleOptional does not match the expected value"),
                return false);
    if (hasAntiquantOffset_) {
        OP_CHECK_IF(
            isSingleMultiSingle_,
            OP_LOGE_FOR_INVALID_LISTSIZE(OP_NAME, "antiquantOffsetOptional", std::to_string(numAntiquantOffset_),
                                         "When x-weight is float8_e4m3fn-float4_e2m1/fp32 in "
                                         "single-multi-single mode, antiquantOffsetOptional must be empty"),
            return false);
        OP_CHECK_IF(
            numAntiquantOffset_ != expectedParamNum,
            OP_LOGE_FOR_INVALID_LISTSIZE(OP_NAME, "antiquantOffsetOptional", std::to_string(numAntiquantOffset_),
                                         "antiquantOffsetOptional size does not match the expected value"),
            return false);
    }
    if (hasBias_) {
        OP_CHECK_IF(numBias_ != expectedParamNum,
                    OP_LOGE_FOR_INVALID_LISTSIZE(OP_NAME, "biasOptional", std::to_string(numBias_),
                                                 "biasOptional size does not match the expected value"),
                    return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckTensorDtype(const gert::TilingContext *context, uint32_t attrIdx,
                                                           size_t idx, const ge::DataType &tensorDtype,
                                                           const std::string &tensorType) const
{
    if (!IsA16W4ND()) {
        return true;
    }
    auto tmpDesc = context->GetDynamicInputDesc(attrIdx, idx);
    auto tmpDType = tmpDesc->GetDataType();
    OP_CHECK_IF(
        tmpDType != tensorDtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, tensorType, ge::TypeUtils::DataTypeToSerialString(tmpDType),
            std::string("The dtype of each tensor in ") + tensorType + " tensor list must be consistent. " +
                tensorType + "[" + std::to_string(idx) + "]'s dtype is different from the first tensor's dtype."),
        return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckTensorDtypeSingleXMultiWeightSingleY(const gert::TilingContext *context,
                                                                                    uint32_t attrIdx, size_t idx,
                                                                                    const ge::DataType &tensorDtype,
                                                                                    const std::string &tensorType) const
{
    auto tmpDesc = context->GetDynamicInputDesc(attrIdx, idx);
    auto tmpDType = tmpDesc->GetDataType();
    OP_CHECK_IF(
        tmpDType != tensorDtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, tensorType, ge::TypeUtils::DataTypeToSerialString(tmpDType),
            std::string("The dtype of each tensor in ") + tensorType + " tensor list must be consistent. " +
                tensorType + "[" + std::to_string(idx) + "]'s dtype is different from the first tensor's dtype."),
        return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::IsNzFormat(const gert::TilingContext *context, uint32_t attrIdx,
                                                     size_t idx) const
{
    auto tmpDesc = context->GetDynamicInputDesc(attrIdx, idx);
    auto tmpFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(tmpDesc->GetStorageFormat()));
    if (tmpFormat == ge::FORMAT_FRACTAL_NZ_C0_16 || tmpFormat == ge::FORMAT_FRACTAL_NZ_C0_32 ||
        tmpFormat == ge::FORMAT_FRACTAL_NZ_C0_4) {
        tmpFormat = ge::FORMAT_FRACTAL_NZ;
    }
    return tmpFormat == ge::FORMAT_FRACTAL_NZ;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckXAndWeightFormat(const gert::TilingContext *context, size_t idx) const
{
    bool xNzFlag = IsNzFormat(context, X_IDX, idx);
    OP_CHECK_IF(xNzFlag, OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "x", "NZ", "ND"), return false);
    if (IsA16W4ND()) {
        bool weightNzFlag = IsNzFormat(context, WEIGHT_IDX, idx);
        OP_CHECK_IF(weightNzFlag, OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "weight", "NZ", "ND"), return false);
        return true;
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckNotNullPtr(const gert::TilingContext *context, uint32_t attrIdx,
                                                          size_t idx) const
{
    auto tmpIDesc = context->GetDynamicInputDesc(attrIdx, idx);
    auto tmpITensor = context->GetDynamicInputTensor(attrIdx, idx);
    auto tmpIShape = context->GetDynamicInputShape(attrIdx, idx);
    OP_CHECK_IF(tmpIDesc == nullptr, OP_LOGE(context->GetNodeName(), "Desc[%zu] is nullptr.", idx), return false);
    OP_CHECK_IF(tmpITensor == nullptr, OP_LOGE(context->GetNodeName(), "Tensor[%zu] is nullptr.", idx), return false);
    OP_CHECK_IF(tmpIShape == nullptr, OP_LOGE(context->GetNodeName(), "Shape[%zu] is nullptr.", idx), return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckNotNull(const gert::TilingContext *context, size_t idx) const
{
    OP_CHECK_IF(!CheckNotNullPtr(context, X_IDX, idx),
                OP_LOGE(context->GetNodeName(), "x's tensor or Desc is nullptr."), return false);
    OP_CHECK_IF(!CheckNotNullPtr(context, WEIGHT_IDX, idx),
                OP_LOGE(context->GetNodeName(), "weight's tensor or Desc is nullptr."), return false);
    OP_CHECK_IF(!CheckNotNullPtr(context, ANTIQUANT_SCALE_IDX, idx),
                OP_LOGE(context->GetNodeName(), "antiquantscale's tensor or Desc is nullptr."), return false);
    if (hasBias_) {
        OP_CHECK_IF(!CheckNotNullPtr(context, BIAS_IDX, idx),
                    OP_LOGE(context->GetNodeName(), "bias's tensor or Desc is nullptr."), return false);
    }
    if (hasAntiquantOffset_) {
        OP_CHECK_IF(!CheckNotNullPtr(context, ANTIQUANT_OFFSET_IDX, idx),
                    OP_LOGE(context->GetNodeName(), "antiquantoffset's tensor or Desc is nullptr."), return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckTensorDimEqualTarget(const gert::TilingContext *context,
                                                                    uint32_t attrIdx, size_t idx, uint32_t targetDim,
                                                                    const std::string &tensorType) const
{
    auto tmpShapePtr = context->GetDynamicInputShape(attrIdx, idx);
    OP_CHECK_IF(tmpShapePtr == nullptr,
                OP_LOGE(context->GetNodeName(), "%s[%zu] shape is nullptr.", tensorType.c_str(), idx), return false);
    auto tmpShape = tmpShapePtr->GetStorageShape();
    uint32_t tmpDimNum = static_cast<uint32_t>(tmpShape.GetDimNum());
    OP_CHECK_IF(
        tmpDimNum != targetDim,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, tensorType + "[" + std::to_string(idx) + "]", std::to_string(tmpDimNum),
            "The shape dim of " + tensorType + "[" + std::to_string(idx) + "] must be " + std::to_string(targetDim)),
        return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckTensorDimSingleXSingleWeightSingleY(const gert::TilingContext *context,
                                                                                   size_t idx) const
{
    OP_CHECK_IF(!CheckTensorDimEqualTarget(context, X_IDX, idx, 2, "x"),
                OP_LOGE(context->GetNodeName(), "Invalid x dimension. "), return false);

    uint32_t targetWeightDim = weightNzFlag_ ? 5 : 3; // NZ (g, n1, k1, k0, n0) 5维 ND (g, k, n) 3维
    OP_CHECK_IF(!CheckTensorDimEqualTarget(context, WEIGHT_IDX, idx, targetWeightDim, "weight"),
                OP_LOGE(context->GetNodeName(), "Invalid weight dimension. "), return false);

    if (!IsA16W4ND() && !IsMxA8W4()) {
        return true;
    }
    if (IsA16W4ND()) {
        // perchannel: antiquantScale 2D [E, N]; pergroup: 3D [E, G, N]
        auto scaleShapePtr = context->GetDynamicInputShape(ANTIQUANT_SCALE_IDX, idx);
        uint32_t scaleDimNum = static_cast<uint32_t>(scaleShapePtr->GetStorageShape().GetDimNum());
        OP_CHECK_IF(scaleDimNum != A16W4_SINGLE_PER_CHANNEL_DIM && scaleDimNum != A16W4_SINGLE_PER_GROUP_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        OP_NAME, "antiquantScale", std::to_string(scaleDimNum),
                        "When x-weight is bf16/fp16-int4/int32 and groupType is 0, the "
                        "dimension of antiquantScale must be 2 (perchannel) or 3 (pergroup)"),
                    return false);
    } else {
        OP_CHECK_IF(!CheckTensorDimEqualTarget(context, ANTIQUANT_SCALE_IDX, idx, 4, "antiquantScale"),
                    OP_LOGE(context->GetNodeName(), "When x-weight is float8_e8m0-float4_e2m1 and groupType is 0, the "
                                                    "dimension of antiquantScale does not match the expected value."),
                    return false);
    }
    if (hasBias_) {
        OP_CHECK_IF(!CheckTensorDimEqualTarget(context, BIAS_IDX, idx, 2, "bias"),
                    OP_LOGE(context->GetNodeName(), "When x-weight is bf16/fp16-int4/int32 and groupType is 0, "
                                                    "the dimension of bias does not match the expected value."),
                    return false);
    }
    if (hasAntiquantOffset_) {
        // perchannel: antiquantOffset 2D [E, N]; pergroup: 3D [E, G, N]
        auto offsetShapePtr = context->GetDynamicInputShape(ANTIQUANT_OFFSET_IDX, idx);
        uint32_t offsetDimNum = static_cast<uint32_t>(offsetShapePtr->GetStorageShape().GetDimNum());
        OP_CHECK_IF(offsetDimNum != A16W4_SINGLE_PER_CHANNEL_DIM && offsetDimNum != A16W4_SINGLE_PER_GROUP_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        OP_NAME, "antiquantOffset", std::to_string(offsetDimNum),
                        "When x-weight is bf16/fp16-int4/int32 and groupType is 0, the "
                        "dimension of antiquantOffset must be 2 (perchannel) or 3 (pergroup)"),
                    return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckTensorDimMultiXMultiWeightMultiY(const gert::TilingContext *context,
                                                                                size_t idx) const
{
    // 多多多场景x已在CheckEmptyTensor中校验
    OP_CHECK_IF(!CheckTensorDimEqualTarget(context, WEIGHT_IDX, idx, 2, "weight"),
                OP_LOGE(context->GetNodeName(), "Invalid weight dimension. "), return false);

    if (IsA16W4ND()) {
        // perchannel: antiquantScale 1D [N_i]; pergroup: 2D [G_i, N_i]
        auto scaleShapePtr = context->GetDynamicInputShape(ANTIQUANT_SCALE_IDX, idx);
        uint32_t scaleDimNum = static_cast<uint32_t>(scaleShapePtr->GetStorageShape().GetDimNum());
        OP_CHECK_IF(scaleDimNum != A16W4_MULTI_PER_CHANNEL_DIM && scaleDimNum != A16W4_MULTI_PER_GROUP_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        OP_NAME, "antiquantScale", std::to_string(scaleDimNum),
                        "When x-weight is bf16/fp16-int4/int32 and groupType is -1, the dimension of antiquantScale "
                        "must be 1 (perchannel) or 2 (pergroup)"),
                    return false);
    } else {
        OP_CHECK_IF(!CheckTensorDimEqualTarget(context, ANTIQUANT_SCALE_IDX, idx, 1, "antiquantScale"),
                    OP_LOGE(context->GetNodeName(), "Invalid antiquantScale dimension. "), return false);
    }

    if (hasBias_) {
        OP_CHECK_IF(!CheckTensorDimEqualTarget(context, BIAS_IDX, idx, 1, "bias"),
                    OP_LOGE(context->GetNodeName(), "Invalid bias dimension. "), return false);
    }
    if (hasAntiquantOffset_) {
        if (IsA16W4ND()) {
            // perchannel: antiquantOffset 1D [N_i]; pergroup: 2D [G_i, N_i]
            auto offsetShapePtr = context->GetDynamicInputShape(ANTIQUANT_OFFSET_IDX, idx);
            uint32_t offsetDimNum = static_cast<uint32_t>(offsetShapePtr->GetStorageShape().GetDimNum());
            OP_CHECK_IF(
                offsetDimNum != A16W4_MULTI_PER_CHANNEL_DIM && offsetDimNum != A16W4_MULTI_PER_GROUP_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    OP_NAME, "antiquantOffset", std::to_string(offsetDimNum),
                    "When x-weight is bf16/fp16-int4/int32 and groupType is -1, the dimension of antiquantOffset "
                    "must be 1 (perchannel) or 2 (pergroup)"),
                return false);
        } else {
            OP_CHECK_IF(!CheckTensorDimEqualTarget(context, ANTIQUANT_OFFSET_IDX, idx, 1, "antiquantOffset"),
                        OP_LOGE(context->GetNodeName(), "Invalid antiquantOffset dimension. "), return false);
        }
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckTensorDim(const gert::TilingContext *context, size_t idx) const
{
    if (groupType_ == static_cast<int64_t>(GroupType::SPLIT_M)) {
        OP_CHECK_IF(!CheckTensorDimSingleXSingleWeightSingleY(context, idx),
                    OP_LOGE(context->GetNodeName(), "CheckTensorDimSingleXSingleWeightSingleY failed"), return false);
    } else {
        OP_CHECK_IF(!CheckTensorDimMultiXMultiWeightMultiY(context, idx),
                    OP_LOGE(context->GetNodeName(), "CheckTensorDimMultiXMultiWeightMultiY failed"), return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckTensorShape(const gert::TilingContext *context, uint32_t attrIdx,
                                                           size_t idx, const std::string &tensorType) const
{
    if (!IsA16W4ND() && !IsMxA8W4()) {
        return true;
    }
    // 校验bias、antiquantScale和antiquantOffset的shape
    auto tensorShapePtr = context->GetDynamicInputShape(attrIdx, idx);
    auto wShapePtr = context->GetDynamicInputShape(WEIGHT_IDX, idx);
    auto tensorShape = tensorShapePtr->GetStorageShape();
    auto wShape = wShapePtr->GetStorageShape();
    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
    uint32_t tensorDimNum = static_cast<uint32_t>(tensorShape.GetDimNum());

    if (groupType_ == static_cast<int64_t>(GroupType::SPLIT_M)) {
        // 校验bias、antiquantScale和antiquantOffset的第一根轴
        uint64_t groupNum = static_cast<uint64_t>(wShape.GetDim(0));
        uint64_t batchSize = static_cast<uint64_t>(tensorShape.GetDim(0));
        OP_CHECK_IF(groupNum != batchSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        OP_NAME, tensorType, Ops::Base::ToString(tensorShape),
                        "Axis 0 of " + tensorType + " must be equal to groupList length " + std::to_string(groupNum)),
                    return false);
    }
    // 校验bias、antiquantScale和antiquantOffset的N轴
    int64_t weightNDimValue =
        IsA16W4ND() ? wShape.GetDim(wDimNum - (transB_ ? 2 : 1)) : wShape.GetDim(wDimNum - 3) * 16L; // MXA8W4 NZ
    if (IsA16W4ND() && weightDtype_ == ge::DT_INT32 && !transB_ &&
        groupType_ == static_cast<int64_t>(GroupType::SPLIT_M)) {
        weightNDimValue *= static_cast<int64_t>(FP4_PER_FP32);
    }
    int64_t tensorNDimValue = (IsMxA8W4() && attrIdx == ANTIQUANT_SCALE_IDX) ? tensorShape.GetDim(tensorDimNum - 3) :
                                                                               tensorShape.GetDim(tensorDimNum - 1);
    OP_CHECK_IF(
        weightNDimValue != tensorNDimValue,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(OP_NAME, tensorType + ", weight",
                                               Ops::Base::ToString(tensorShape) + ", " + Ops::Base::ToString(wShape),
                                               "NDim of " + tensorType + " must be equal to NDim of weight"),
        return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckTensorShapeSingleXMultiWeightSingleY(const gert::TilingContext *context,
                                                                                    uint32_t attrIdx, size_t idx,
                                                                                    const std::string &tensorType) const
{
    auto tensorShapePtr = context->GetDynamicInputShape(attrIdx, idx);
    auto wShapePtr = context->GetDynamicInputShape(WEIGHT_IDX, idx);
    auto tensorShape = tensorShapePtr->GetStorageShape();
    auto wShape = wShapePtr->GetStorageShape();
    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
    uint32_t tensorDimNum = static_cast<uint32_t>(tensorShape.GetDimNum());

    // MXA8W4转置NZ排布下，weight的N轴由N1和N0相乘得到
    int64_t weightNDimValue = wShape.GetDim(wDimNum - ANTEPENULTIMATE_DIM) * wShape.GetDim(wDimNum - PENULTIMATE_DIM);
    int64_t tensorNDimValue = attrIdx == ANTIQUANT_SCALE_IDX ? tensorShape.GetDim(tensorDimNum - ANTEPENULTIMATE_DIM) :
                                                               tensorShape.GetDim(tensorDimNum - LAST_DIM);
    OP_CHECK_IF(
        weightNDimValue != tensorNDimValue,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(OP_NAME, tensorType + ", weight",
                                               Ops::Base::ToString(tensorShape) + ", " + Ops::Base::ToString(wShape),
                                               "NDim of " + tensorType + " must be equal to NDim of weight"),
        return false);
    if (attrIdx == ANTIQUANT_SCALE_IDX) {
        auto xShapePtr = context->GetDynamicInputShape(X_IDX, 0);
        auto xShape = xShapePtr->GetStorageShape();
        uint32_t xDimNum = static_cast<uint32_t>(xShape.GetDimNum());
        uint64_t kSize = 0;
        if (transA_) {
            OP_CHECK_IF(!GetDimFromEnd(xShape, xDimNum, kSize),
                        OP_LOGE(context->GetNodeName(), "Get kSize from x dim0 failed."), return false);
        } else {
            OP_CHECK_IF(!GetDimFromEnd(xShape, 1, kSize),
                        OP_LOGE(context->GetNodeName(), "Get kSize from x last dim failed."), return false);
        }
        int64_t scaleGroupNum = tensorShape.GetDim(tensorDimNum - PENULTIMATE_DIM) * MX_GROUP_FACTOR;
        OP_CHECK_IF(scaleGroupNum <= 0 || kSize % static_cast<uint64_t>(scaleGroupNum) != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        OP_NAME, tensorType, Ops::Base::ToString(tensorShape),
                        "The groupNum[" + std::to_string(scaleGroupNum) + "] of " + tensorType +
                            " must be greater than 0 and divisible by kSize[" + std::to_string(kSize) + "]"),
                    return false);
        OP_CHECK_IF(kSize % MX_GROUP_SIZE != 0 || scaleGroupNum != static_cast<int64_t>(kSize / MX_GROUP_SIZE),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        OP_NAME, tensorType, Ops::Base::ToString(tensorShape),
                        "The groupSize of " + tensorType + " must be " + std::to_string(MX_GROUP_SIZE)),
                    return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckDimValue(const gert::TilingContext *context, size_t idx) const
{
    auto xDimNum = context->GetDynamicInputTensor(X_IDX, idx)->GetStorageShape().GetDimNum();
    // 校验batch轴和M轴大于等于0，bf16/fp16-int4/int32 ND场景下不考虑X转置的情况
    for (size_t dimIdx = 0; dimIdx < xDimNum - 1; dimIdx++) {
        int64_t xDimValue = context->GetDynamicInputTensor(X_IDX, idx)->GetStorageShape().GetDim(dimIdx);
        OP_CHECK_IF(
            xDimValue < 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x[" + std::to_string(idx) + "]", std::to_string(xDimValue),
                                                  "The dimension[" + std::to_string(dimIdx) + "] of the tensor[" +
                                                      std::to_string(idx) + "] of x must be >= 0"),
            return false);
    }
    int64_t xKDimValue = context->GetDynamicInputTensor(X_IDX, idx)->GetStorageShape().GetDim(xDimNum - 1);
    OP_CHECK_IF(xKDimValue < 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    OP_NAME, "x[" + std::to_string(idx) + "]", std::to_string(xKDimValue),
                    "The k dimension of the tensor[" + std::to_string(idx) + "] cannot be negative"),
                return false);
    if (!IsA16W4ND()) {
        return true;
    }
    auto wShape = context->GetDynamicInputShape(WEIGHT_IDX, idx)->GetStorageShape();
    auto wDimNum = wShape.GetDimNum();
    auto wNDimNum = transB_ ? (wDimNum - 2) : (wDimNum - 1);
    auto wKDimNum = transB_ ? (wDimNum - 1) : (wDimNum - 2);
    int64_t weightNDimValue = wShape.GetDim(wNDimNum);
    int64_t weightKDimValue = wShape.GetDim(wKDimNum);
    OP_CHECK_IF(weightNDimValue < 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    OP_NAME, "weight[" + std::to_string(idx) + "]", std::to_string(weightNDimValue),
                    "The n dimensions of the tensor[" + std::to_string(idx) + "] should not be negative"),
                return false);
    OP_CHECK_IF(xKDimValue != weightKDimValue,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    OP_NAME, "x[" + std::to_string(idx) + "], weight[" + std::to_string(idx) + "]",
                    std::to_string(xKDimValue) + ", " + std::to_string(weightKDimValue),
                    "The k dimension of the tensor[" + std::to_string(idx) + "] of x and weight must be equal"),
                return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckWeightInnerAxisEven(const gert::TilingContext *context, size_t idx) const
{
    if (weightDtype_ == ge::DT_INT4) {
        auto wShapePtr = context->GetDynamicInputShape(WEIGHT_IDX, idx);
        auto wShape = wShapePtr->GetOriginShape();
        auto wDimNum = wShape.GetDimNum() - 1;
        int64_t wInnerDimvalue = wShape.GetDim(wDimNum);
        OP_CHECK_IF((wInnerDimvalue % 2) != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "weight[" + std::to_string(idx) + "]",
                                                          std::to_string(wInnerDimvalue),
                                                          "The last dimension of weight tensor must be even"),
                    return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckEveryTensor(const gert::TilingContext *context) const
{
    for (size_t i = 0; i < numX_; i++) {
        OP_CHECK_IF(!CheckNotNull(context, i), OP_LOGE(context->GetNodeName(), "CheckTensorNotNull failed"),
                    return false);
        OP_CHECK_IF(!CheckTensorDim(context, i), OP_LOGE(context->GetNodeName(), "CheckTensorDim failed"),
                    return false);
        OP_CHECK_IF(!CheckDimValue(context, i), OP_LOGE(context->GetNodeName(), "CheckDimValue failed"), return false);
        OP_CHECK_IF(!CheckXAndWeightFormat(context, i),
                    OP_LOGE(context->GetNodeName(), "Check X And Weight Format failed"), return false);
        OP_CHECK_IF(!CheckTensorDtype(context, X_IDX, i, xDType_, "x"),
                    OP_LOGE(context->GetNodeName(), "Check Tensor x Dtype failed"), return false);
        OP_CHECK_IF(!CheckTensorDtype(context, WEIGHT_IDX, i, weightDtype_, "weight"),
                    OP_LOGE(context->GetNodeName(), "Check Tensor weight Dtype failed"), return false);
        OP_CHECK_IF(!CheckTensorDtype(context, ANTIQUANT_SCALE_IDX, i, antiquantScaleDtype_, "antiquantScale"),
                    OP_LOGE(context->GetNodeName(), "Check Tensor antiquantScale Dtype failed"), return false);
        OP_CHECK_IF(!CheckTensorShape(context, ANTIQUANT_SCALE_IDX, i, "antiquantScale"),
                    OP_LOGE(context->GetNodeName(), "Check antiquantScale tensor shape failed"), return false);
        if (hasAntiquantOffset_) {
            OP_CHECK_IF(!CheckTensorDtype(context, ANTIQUANT_OFFSET_IDX, i, antiquantOffsetDtype_, "antiquantOffset"),
                        OP_LOGE(context->GetNodeName(), "Check Tensor antiquantOffset Dtype failed"), return false);
            OP_CHECK_IF(!CheckTensorShape(context, ANTIQUANT_OFFSET_IDX, i, "antiquantOffset"),
                        OP_LOGE(context->GetNodeName(), "Check antiquantOffset tensor shape failed"), return false);
        }
        if (hasBias_) {
            OP_CHECK_IF(!CheckTensorDtype(context, BIAS_IDX, i, biasDtype_, "bias"),
                        OP_LOGE(context->GetNodeName(), "Check Tensor bias Dtype failed"), return false);
            OP_CHECK_IF(!CheckTensorShape(context, BIAS_IDX, i, "bias"),
                        OP_LOGE(context->GetNodeName(), "Check bias tensor shape failed"), return false);
        }
        OP_CHECK_IF(!CheckWeightInnerAxisEven(context, i),
                    OP_LOGE(context->GetNodeName(), "CheckWeightInnerAxisEven failed"), return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckWeightSingleXMultiWeightSingleY(const gert::TilingContext *context) const
{
    auto xShapePtr = context->GetDynamicInputShape(X_IDX, 0);
    OP_CHECK_IF(xShapePtr == nullptr, OP_LOGE(context->GetNodeName(), "xShapePtr is nullptr."), return false);
    auto xShape = xShapePtr->GetStorageShape();
    uint32_t xDimNum = static_cast<uint32_t>(xShape.GetDimNum());
    int64_t xKDimValue = transA_ ? xShape.GetDim(0) : xShape.GetDim(xDimNum - LAST_DIM);

    auto yShapePtr = context->GetOutputShape(0);
    OP_CHECK_IF(yShapePtr == nullptr, OP_LOGE(context->GetNodeName(), "yShapePtr is nullptr."), return false);
    const gert::Shape &yShape = yShapePtr->GetOriginShape();
    int64_t yNDimValue = yShape.GetDim(static_cast<uint32_t>(yShape.GetDimNum()) - LAST_DIM);

    uint32_t targetWeightDim = weightNzFlag_ ? NZ_WEIGHT_DIM_NUM : ND_WEIGHT_DIM_NUM;
    for (size_t i = 0; i < numWeight_; ++i) {
        OP_CHECK_IF(!CheckNotNullPtr(context, WEIGHT_IDX, i),
                    OP_LOGE(context->GetNodeName(), "weight's tensor or Desc is nullptr."), return false);
        OP_CHECK_IF(!CheckTensorDimEqualTarget(context, WEIGHT_IDX, i, targetWeightDim, "weight"),
                    OP_LOGE(context->GetNodeName(), "Invalid weight dimension."), return false);
        auto wShape = context->GetDynamicInputShape(WEIGHT_IDX, i)->GetStorageShape();
        uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
        // ND非转置weight形如(K, N)，转置weight形如(N, K)
        int64_t weightKDimValue =
            transB_ ? wShape.GetDim(wDimNum - LAST_DIM) : wShape.GetDim(wDimNum - PENULTIMATE_DIM);
        int64_t weightNDimValue =
            transB_ ? wShape.GetDim(wDimNum - PENULTIMATE_DIM) : wShape.GetDim(wDimNum - LAST_DIM);
        if (weightNzFlag_) {
            // 非转置NZ排布(N1, K1, K0, N0)：K=K1*K0，N=N1*N0
            // 转置NZ排布(K1, N1, N0, K0)：K=K1*K0，N=N1*N0
            weightKDimValue =
                transB_ ? wShape.GetDim(wDimNum - FOURTH_FROM_LAST_DIM) * wShape.GetDim(wDimNum - LAST_DIM) :
                          wShape.GetDim(wDimNum - ANTEPENULTIMATE_DIM) * wShape.GetDim(wDimNum - PENULTIMATE_DIM);
            weightNDimValue =
                transB_ ? wShape.GetDim(wDimNum - ANTEPENULTIMATE_DIM) * wShape.GetDim(wDimNum - PENULTIMATE_DIM) :
                          wShape.GetDim(wDimNum - FOURTH_FROM_LAST_DIM) * wShape.GetDim(wDimNum - LAST_DIM);
        }
        OP_CHECK_IF(xKDimValue != weightKDimValue,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        OP_NAME, "x[0], weight[" + std::to_string(i) + "]",
                        Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(wShape),
                        "KDim of weight[" + std::to_string(i) + "] must be equal to KDim of x[0]"),
                    return false);
        OP_CHECK_IF(yNDimValue != weightNDimValue,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        OP_NAME, "y, weight[" + std::to_string(i) + "]",
                        Ops::Base::ToString(yShape) + ", " + Ops::Base::ToString(wShape),
                        "NDim of weight[" + std::to_string(i) + "] must be equal to NDim of y"),
                    return false);
        OP_CHECK_IF(!CheckTensorDtypeSingleXMultiWeightSingleY(context, WEIGHT_IDX, i, weightDtype_, "weight"),
                    OP_LOGE(context->GetNodeName(), "Check Tensor weight Dtype failed"), return false);
        OP_CHECK_IF(!CheckWeightInnerAxisEven(context, i),
                    OP_LOGE(context->GetNodeName(), "CheckWeightInnerAxisEven failed"), return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckAntiquantScaleSingleXMultiWeightSingleY(
    const gert::TilingContext *context) const
{
    for (size_t i = 0; i < numAntiquantScale_; ++i) {
        OP_CHECK_IF(!CheckNotNullPtr(context, ANTIQUANT_SCALE_IDX, i),
                    OP_LOGE(context->GetNodeName(), "antiquantScale's tensor or Desc is nullptr."), return false);
        OP_CHECK_IF(!CheckTensorDimEqualTarget(context, ANTIQUANT_SCALE_IDX, i, 3, "antiquantScale"),
                    OP_LOGE(context->GetNodeName(), "Invalid antiquantScale dimension."), return false);
        OP_CHECK_IF(!CheckTensorDtypeSingleXMultiWeightSingleY(context, ANTIQUANT_SCALE_IDX, i, antiquantScaleDtype_,
                                                               "antiquantScale"),
                    OP_LOGE(context->GetNodeName(), "Check Tensor antiquantScale Dtype failed"), return false);
        OP_CHECK_IF(!CheckTensorShapeSingleXMultiWeightSingleY(context, ANTIQUANT_SCALE_IDX, i, "antiquantScale"),
                    OP_LOGE(context->GetNodeName(), "Check antiquantScale tensor shape failed"), return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckAntiquantOffsetSingleXMultiWeightSingleY(
    const gert::TilingContext *context) const
{
    OP_CHECK_IF(hasAntiquantOffset_,
                OP_LOGE(context->GetNodeName(), "When x-weight is float8_e4m3fn-float4_e2m1/fp32 in "
                                                "single-multi-single mode, antiquantOffsetOptional must be empty."),
                return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckBiasSingleXMultiWeightSingleY(const gert::TilingContext *context) const
{
    if (!hasBias_) {
        return true;
    }
    for (size_t i = 0; i < numBias_; ++i) {
        OP_CHECK_IF(!CheckNotNullPtr(context, BIAS_IDX, i),
                    OP_LOGE(context->GetNodeName(), "bias's tensor or Desc is nullptr."), return false);
        OP_CHECK_IF(!CheckTensorDimEqualTarget(context, BIAS_IDX, i, 1, "bias"),
                    OP_LOGE(context->GetNodeName(), "Invalid bias dimension."), return false);
        OP_CHECK_IF(!CheckTensorDtypeSingleXMultiWeightSingleY(context, BIAS_IDX, i, biasDtype_, "bias"),
                    OP_LOGE(context->GetNodeName(), "Check Tensor bias Dtype failed"), return false);
        OP_CHECK_IF(!CheckTensorShapeSingleXMultiWeightSingleY(context, BIAS_IDX, i, "bias"),
                    OP_LOGE(context->GetNodeName(), "Check bias tensor shape failed"), return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckEveryTensorSingleXMultiWeightSingleY(
    const gert::TilingContext *context) const
{
    OP_CHECK_IF(!CheckNotNullPtr(context, X_IDX, 0), OP_LOGE(context->GetNodeName(), "x's tensor or Desc is nullptr."),
                return false);
    OP_CHECK_IF(!CheckTensorDimEqualTarget(context, X_IDX, 0, 2, "x"),
                OP_LOGE(context->GetNodeName(), "Invalid x dimension."), return false);
    OP_CHECK_IF(!CheckDimValue(context, 0), OP_LOGE(context->GetNodeName(), "CheckDimValue failed"), return false);
    OP_CHECK_IF(!CheckXAndWeightFormat(context, 0), OP_LOGE(context->GetNodeName(), "Check X And Weight Format failed"),
                return false);
    OP_CHECK_IF(!CheckTensorDtypeSingleXMultiWeightSingleY(context, X_IDX, 0, xDType_, "x"),
                OP_LOGE(context->GetNodeName(), "Check Tensor x Dtype failed"), return false);
    OP_CHECK_IF(!CheckWeightSingleXMultiWeightSingleY(context),
                OP_LOGE(context->GetNodeName(), "CheckWeightSingleXMultiWeightSingleY failed"), return false);
    OP_CHECK_IF(!CheckAntiquantScaleSingleXMultiWeightSingleY(context),
                OP_LOGE(context->GetNodeName(), "CheckAntiquantScaleSingleXMultiWeightSingleY failed"), return false);
    OP_CHECK_IF(!CheckAntiquantOffsetSingleXMultiWeightSingleY(context),
                OP_LOGE(context->GetNodeName(), "CheckAntiquantOffsetSingleXMultiWeightSingleY failed"), return false);
    OP_CHECK_IF(!CheckBiasSingleXMultiWeightSingleY(context),
                OP_LOGE(context->GetNodeName(), "CheckBiasSingleXMultiWeightSingleY failed"), return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckGroupList(const gert::TilingContext *context) const
{
    if (groupType_ == static_cast<int64_t>(GroupType::SPLIT_M)) {
        OP_CHECK_IF(groupListType_ != 0 && groupListType_ != 1,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        OP_NAME, "groupListType", std::to_string(groupListType_),
                        "When the dtype of x-weight is bf16/fp16-int4/int32 and groupType is 0, "
                        "groupListType only supports values 0 or 1"),
                    return false);
        auto groupListPtr = context->GetOptionalInputShape(GroupedMatmul::GROUPLIST_INDEX);
        OP_CHECK_IF(groupListPtr == nullptr, OP_LOGE(context->GetNodeName(), "GroupList is nullptr."), return false);
        int32_t groupListSize = static_cast<int32_t>(groupListPtr->GetOriginShape().GetDim(0));

        int32_t groupNum = static_cast<int32_t>(numWeight_);
        if (!isSingleMultiSingle_) {
            auto wTensor = context->GetDynamicInputTensor(WEIGHT_IDX, 0);
            OP_CHECK_IF(wTensor == nullptr, OP_LOGE(context->GetNodeName(), "wTensor is nullptr."), return false);
            gert::Shape wShape = wTensor->GetStorageShape();
            groupNum = static_cast<int32_t>(wShape.GetDim(0));
        }

        OP_LOGD(context->GetNodeName(), "groupNum is %d, groupListSize is %d.", groupNum, groupListSize);
        OP_CHECK_IF(groupListSize != groupNum,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        OP_NAME, "groupList", std::to_string(groupListSize),
                        "When groupType is 0, the length of groupList must match groupNum, but actual is " +
                            std::to_string(groupListSize) + " and " + std::to_string(groupNum)),
                    return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::AnalyzeAttr(const gert::TilingContext *context)
{
    auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is nullptr."), return false);
    coreNum_ = compileInfoPtr->aicNum;
    aivNum_ = compileInfoPtr->aivNum;

    auto attr = context->GetAttrs();
    OP_CHECK_IF(attr == nullptr, OP_LOGE(context->GetNodeName(), "attr is nullptr."), return false);
    const bool *transposeWeightPtr = attr->GetAttrPointer<bool>(ATTR_TRANS_W_IDX);
    const bool *transposeXPtr = attr->GetAttrPointer<bool>(ATTR_TRANS_X_IDX);
    const int64_t *groupTypePtr = attr->GetAttrPointer<int64_t>(ATTR_GROUPTYPE_IDX);
    const int64_t *splitItemPtr = attr->GetAttrPointer<int64_t>(ATTR_SPLIT_ITEM_IDX);
    const int64_t *groupListTypePtr = attr->GetAttrPointer<int64_t>(ATTR_GROUP_LIST_TYPE_IDX);
    transA_ = transposeXPtr != nullptr ? *transposeXPtr : false;
    transB_ = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    groupType_ = groupTypePtr != nullptr ? *groupTypePtr : static_cast<int64_t>(GroupType::NO_SPLIT);
    splitItem_ = splitItemPtr != nullptr ? *splitItemPtr : 0; // 0: 默认split_item
    groupListType_ = groupListTypePtr != nullptr ? *groupListTypePtr : 0;
    isSingleX_ = (groupType_ != static_cast<int64_t>(GroupType::NO_SPLIT) &&
                  context->GetDynamicInputTensor(X_IDX, 1) == nullptr);

    // 2: when x is multi-tensor, y is single-tensor; 3: when x is single-tensor, y is single-tensor
    isSingleY_ = (splitItem_ == 2 || splitItem_ == 3);
    GetNumOfInputs(context);
    // 获取weight format和各参数类型
    OP_CHECK_IF(!AnalyzeInput(context), OP_LOGE(context->GetNodeName(), "Invalid Input param"), return false);

    uint32_t singleWeightDim = weightNzFlag_ ? 5 : 3; // 3:单tensor ND时weight的维度为3
    isSingleWeight_ = (groupType_ != static_cast<int64_t>(GroupType::NO_SPLIT) &&
                       context->GetDynamicInputTensor(WEIGHT_IDX, 1) == nullptr &&
                       context->GetDynamicInputTensor(WEIGHT_IDX, 0) != nullptr &&
                       context->GetDynamicInputTensor(WEIGHT_IDX, 0)->GetStorageShape().GetDimNum() == singleWeightDim);
    isSingleMultiSingle_ = IsMxA8W4() && isSingleX_ && !isSingleWeight_ && isSingleY_;

    // 参数校验
    OP_CHECK_IF(!CheckCoreNum(context), OP_LOGE(context->GetNodeName(), "Invalid core number ratio"), return false);
    OP_CHECK_IF(!CheckUnsupportDataFlow(),
                OP_LOGE(context->GetNodeName(), "Input data contains unsupported dtype or format."), return false);
    OP_CHECK_IF(!CheckTransposeStatus(), OP_LOGE(context->GetNodeName(), "CheckTransposeStatus failed."), return false);
    OP_CHECK_IF(!CheckGroupTypeAndSplitItem(), OP_LOGE(context->GetNodeName(), "CheckParam failed."), return false);
    OP_CHECK_IF(!CheckTensorListSize(), OP_LOGE(context->GetNodeName(), "CheckTensorListSize failed."), return false);
    OP_CHECK_IF(!CheckEmptyTensor(context), OP_LOGE(context->GetNodeName(), "CheckEmptyTensor failed."), return false);
    OP_CHECK_IF(!CheckAntiQuantDtype(), OP_LOGE(context->GetNodeName(), "CheckAntiQuantDtype failed."), return false);
    OP_CHECK_IF(!CheckBiasDtype(), OP_LOGE(context->GetNodeName(), "CheckBiasDtype failed."), return false);
    OP_CHECK_IF(!CheckGroupList(context), OP_LOGE(context->GetNodeName(), "CheckGroupList failed."), return false);
    if (isSingleMultiSingle_) {
        OP_CHECK_IF(!CheckEveryTensorSingleXMultiWeightSingleY(context),
                    OP_LOGE(context->GetNodeName(), "CheckEveryTensorSingleXMultiWeightSingleY failed."), return false);
    } else {
        OP_CHECK_IF(!CheckEveryTensor(context), OP_LOGE(context->GetNodeName(), "CheckEveryTensor failed."),
                    return false);
    }

    // 参数设置
    OP_CHECK_IF(!SetShapeList(context), OP_LOGE(context->GetNodeName(), "SetShapeList failed."), return false);
    OP_CHECK_IF(!SetAntiquantGroupSize(context), OP_LOGE(context->GetNodeName(), "Unable to get antiquant groupSize"),
                return false);
    OP_CHECK_IF(!CheckPerTokenScale(context), OP_LOGE(context->GetNodeName(), "CheckPerTokenScale failed."),
                return false);
    PrintInputParam(context);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::AnalyzeInput(const gert::TilingContext *context)
{
    auto xDesc = context->GetDynamicInputDesc(X_IDX, 0);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context->GetNodeName(), "xDesc is nullptr."), return false);
    xDType_ = xDesc->GetDataType();

    auto wDesc = context->GetDynamicInputDesc(WEIGHT_IDX, 0);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context->GetNodeName(), "wDesc is nullptr."), return false);
    weightDtype_ = wDesc->GetDataType();
    auto wFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    if (wFormat == ge::FORMAT_FRACTAL_NZ_C0_16 || wFormat == ge::FORMAT_FRACTAL_NZ_C0_32 ||
        wFormat == ge::FORMAT_FRACTAL_NZ_C0_4 || wFormat == ge::FORMAT_FRACTAL_NZ_C0_2) {
        wFormat = ge::FORMAT_FRACTAL_NZ;
    }
    weightNzFlag_ = wFormat == ge::FORMAT_FRACTAL_NZ;

    hasBias_ = numBias_ > 0;
    if (numBias_ == 1) {
        auto biasShape = context->GetDynamicInputShape(BIAS_IDX, 0);
        hasBias_ = biasShape->GetStorageShape().GetShapeSize() != 0;
    }
    if (hasBias_) {
        auto biasDesc = context->GetDynamicInputDesc(BIAS_IDX, 0);
        OP_CHECK_IF(biasDesc == nullptr, OP_LOGE(context->GetNodeName(), "biasDesc is nullptr."), return false);
        biasDtype_ = biasDesc->GetDataType();
    }

    auto antiquantScaleDesc = context->GetDynamicInputDesc(ANTIQUANT_SCALE_IDX, 0);
    OP_CHECK_IF(antiquantScaleDesc == nullptr, OP_LOGE(context->GetNodeName(), "antiquantScaleDesc is nullptr."),
                return false);
    antiquantScaleDtype_ = antiquantScaleDesc->GetDataType();

    hasAntiquantOffset_ = numAntiquantOffset_ > 0;
    if (numAntiquantOffset_ == 1) {
        auto antiQuantOffsetShape = context->GetDynamicInputShape(ANTIQUANT_OFFSET_IDX, 0);
        hasAntiquantOffset_ = antiQuantOffsetShape->GetStorageShape().GetShapeSize() != 0;
    }
    if (hasAntiquantOffset_) {
        auto antiquantOffsetDesc = context->GetDynamicInputDesc(ANTIQUANT_OFFSET_IDX, 0);
        OP_CHECK_IF(antiquantOffsetDesc == nullptr, OP_LOGE(context->GetNodeName(), "antiquantOffsetDesc is nullptr."),
                    return false);
        antiquantOffsetDtype_ = antiquantOffsetDesc->GetDataType();
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::EnableTailResplit() const
{
    // 不使能尾块重切分的场景
    // 1. 多多多
    // 2. 单单单 weight ND B非转置
    // 3. 单单单 bf16/fp16-int4/int32 ND pergroup B转置
    if (!isSingleX_ && !isSingleWeight_ && !isSingleY_) {
        return false;
    }

    if (!weightNzFlag_ && !transB_) {
        return false;
    }

    if (IsA16W4NDPergroup() && transB_) {
        return false;
    }

    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CalcResplitTiling(const gert::TilingContext *context)
{
    if (!EnableTailResplit()) {
        return true;
    }

    uint64_t c0Size = 0;
    OP_CHECK_IF(!GetC0Size(context, xDType_, c0Size), OP_LOGE(context->GetNodeName(), "Get C0 size failed"),
                return false);
    OP_CHECK_IF(
        (weightNzFlag_ && !transB_ && nSize_ % c0Size > 0),
        OP_LOGE(context->GetNodeName(),
                "Invalid C0 size[%lu], expect greater than 0 and divisible by N[%lu] when weight format is FRACTAL_NZ",
                c0Size, nSize_),
        return false);
    OP_CHECK_IF(coreNum_ <= 0, OP_LOGE(context->GetNodeName(), "Invalid core num[%u], expect greater than 0", coreNum_),
                return false);

    cubeNumBlocksN_ = static_cast<uint8_t>(coreNum_);
    if (nSize_ % (coreNum_ * static_cast<uint64_t>(BASIC_BLOCK_BASE_N)) == 0UL) {
        resplitParam_.mainBlockSize = BASIC_BLOCK_BASE_N;
        resplitParam_.mainBlockCount = nSize_ / (coreNum_ * static_cast<uint64_t>(BASIC_BLOCK_BASE_N));
    } else if (nSize_ > coreNum_ * static_cast<uint64_t>(BASIC_BLOCK_BASE_N_MIN)) {
        // 该场景下可以保证分满核且尾块在128~256之间
        CalcFullNumBlocksResplitTiling(c0Size);
    } else {
        // N <= 4096场景，优先保证单核尾块大于128，可能无法分满核
        CalcNoFullNumBlocksResplitTiling(c0Size);
    }
    OP_CHECK_IF(!CheckResplitTilingResult(context), OP_LOGE(context->GetNodeName(), "Invalid resplit tiling result"),
                return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::SetBaseTiling()
{
    tilingData_.gmmWeightQuantParam.groupNum = groupNum_;
    tilingData_.gmmWeightQuantParam.coreNum = coreNum_;
    tilingData_.gmmWeightQuantParam.kSize = kSize_;
    tilingData_.gmmWeightQuantParam.nSize = nSizeOri_;
    tilingData_.gmmWeightQuantParam.singleX = static_cast<uint8_t>(isSingleX_);
    tilingData_.gmmWeightQuantParam.singleWeight = static_cast<uint8_t>(isSingleWeight_);
    tilingData_.gmmWeightQuantParam.singleY = static_cast<uint8_t>(isSingleY_);
    tilingData_.gmmWeightQuantParam.groupType = static_cast<int8_t>(groupType_);
    tilingData_.gmmWeightQuantParam.groupListType = static_cast<uint8_t>(groupListType_);
    tilingData_.gmmWeightQuantParam.hasBias = static_cast<uint8_t>(hasBias_);
    tilingData_.gmmWeightQuantParam.cubeNumBlocksN = cubeNumBlocksN_;
    tilingData_.gmmWeightQuantParam.groupSize = groupSize_;
    tilingData_.gmmWeightQuantParam.mainBlockSize = resplitParam_.mainBlockSize;
    tilingData_.gmmWeightQuantParam.mainBlockCount = resplitParam_.mainBlockCount * coreNum_;
    tilingData_.gmmWeightQuantParam.firstTailBlockSize = resplitParam_.firstTailBlockSize;
    tilingData_.gmmWeightQuantParam.secondTailBlockSize = resplitParam_.secondTailBlockSize;
    tilingData_.gmmWeightQuantParam.firstTailBlockCount = resplitParam_.firstTailBlockCount;
    tilingData_.gmmWeightQuantParam.secondTailBlockCount = resplitParam_.secondTailBlockCount;
    errno_t retM = memcpy_s(tilingData_.gmmArray.mList, sizeof(tilingData_.gmmArray.mList), mList_, sizeof(mList_));
    if (retM != EOK) {
        return false;
    }
    errno_t retK = memcpy_s(tilingData_.gmmArray.kList, sizeof(tilingData_.gmmArray.kList), kList_, sizeof(kList_));
    if (retK != EOK) {
        return false;
    }
    errno_t retN = memcpy_s(tilingData_.gmmArray.nList, sizeof(tilingData_.gmmArray.nList), nList_, sizeof(nList_));
    if (retN != EOK) {
        return false;
    }
    return true;
}

void GroupedWeightQuantBatchMatmulTiling::SetMatMulTiling()
{
    tilingData_.mmTilingData.baseM = BASIC_BLOCK_BASE_M;
    tilingData_.mmTilingData.singleCoreM = BASIC_BLOCK_BASE_M;
    tilingData_.mmTilingData.isBias = static_cast<int32_t>(hasBias_);
    tilingData_.mmTilingData.M = mSize_;
    tilingData_.mmTilingData.N = nSize_;
    tilingData_.mmTilingData.Ka = kSize_;
    tilingData_.mmTilingData.Kb = kSize_;
    tilingData_.mmTilingData.singleCoreN = BASIC_BLOCK_BASE_N;
    tilingData_.mmTilingData.singleCoreK = kSize_;
    tilingData_.mmTilingData.dbL0A = BUFFER_NUM_2;
    tilingData_.mmTilingData.dbL0B = BUFFER_NUM_2;
    tilingData_.mmTilingData.dbL0C = 1;
    tilingData_.mmTilingData.shareL0CSize = BASIC_BLOCK_BASE_M * BASIC_BLOCK_BASE_N * sizeof(float);

    tilingData_.mmTilingData.baseN = BASIC_BLOCK_BASE_N;
    tilingData_.mmTilingData.baseK = BASIC_BLOCK_BASE_K;
    tilingData_.mmTilingData.stepKa = STEP_K_4;
    tilingData_.mmTilingData.stepKb = STEP_K_4;
    tilingData_.mmTilingData.depthA1 = DEPTH_8;
    tilingData_.mmTilingData.depthB1 = DEPTH_8;
    tilingData_.mmTilingData.stepM = 1;
    tilingData_.mmTilingData.stepN = 1;
    tilingData_.mmTilingData.usedCoreNum = coreNum_;

    if (xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT4) {
        // S8S4场景，MAD采用S8类型，baseK需要放大2倍
        tilingData_.mmTilingData.baseK = BASIC_BLOCK_BASE_K * S8S4_BASEK_MULTIPLIER;
        // A8W4场景在UB中处理bias，mm api默认无bias
        tilingData_.mmTilingData.isBias = 0;
        // groupsize=192时, ubMte2InnerSize=384, stepKb=ubMte2InnerSize/baseK=3
        if (groupSize_ == 192u) {
            tilingData_.mmTilingData.stepKa = STEP_K_3;
            tilingData_.mmTilingData.stepKb = STEP_K_3;
        }
        OP_LOGI("SetMatMulTiling", "stepKb = %u", tilingData_.mmTilingData.stepKb);
    } else if (xDType_ == ge::DT_FLOAT8_E4M3FN &&
               (weightDtype_ == ge::DT_FLOAT4_E2M1 || weightDtype_ == ge::DT_FLOAT4_E1M2) &&
               antiquantScaleDtype_ == ge::DT_FLOAT8_E8M0) {
        // MxA8W4场景配置mxTypePara
        tilingData_.mmTilingData.mxTypePara = (SCALE_FACTOR_DEFAULT << SCALE_FACTOR_N_BIT) +
                                              (SCALE_FACTOR_DEFAULT << SCALE_FACTOR_M_BIT) +
                                              (SCALE_FACTOR_MIN << SCALE_FACTOR_B_BIT) + SCALE_FACTOR_MIN;
    } else if (hasBias_) {
        tilingData_.mmTilingData.baseM = BASIC_BLOCK_BASE_M_WITH_BIAS;
        tilingData_.mmTilingData.singleCoreM = BASIC_BLOCK_BASE_M_WITH_BIAS;
    }
}

void GroupedWeightQuantBatchMatmulTiling::SetTilingKey(gert::TilingContext *context)
{
    constexpr uint8_t DECIMAL = 10U;
    // 平台类型占2位(平台大类， 平台小类)，平台大类在高位，需要乘10
    tilingKeyConfig_.socVersionType = static_cast<uint8_t>(SocVersionType::SUPPORT_L1_TO_BT_BF16) * DECIMAL;
    tilingKeyConfig_.quantizationScenario = static_cast<uint8_t>(QuantizationScenario::DEFAULT);
    // 算法类型占2位(算法大类，算法小类)，算法大类在高位，需要乘10
    if (EnableTailResplit()) {
        tilingKeyConfig_.algorithm = static_cast<uint8_t>(OptimizationAlgorithmCategory::VECTOR_ANTIQUANT) * DECIMAL +
                                     static_cast<uint8_t>(OptimizationAlgorithmSubCategory::N_FIRST_TAIL_RESPLIT);
    } else {
        tilingKeyConfig_.algorithm = static_cast<uint8_t>(OptimizationAlgorithmCategory::VECTOR_ANTIQUANT) * DECIMAL +
                                     static_cast<uint8_t>(OptimizationAlgorithmSubCategory::N_FIRST_BASIC_BLOCK);
    }

    tilingKeyConfig_.transposeSituation = (static_cast<uint8_t>(transA_) << 1) | static_cast<uint8_t>(transB_);

    if (antiquantScaleDtype_ == ge::DT_FLOAT8_E8M0) {
        tilingKeyConfig_.antiquantType = static_cast<uint8_t>(QuantType::MX);
    } else if (IsA16W4NDPergroup()) {
        tilingKeyConfig_.antiquantType = static_cast<uint8_t>(QuantType::PER_GROUP);
    } else {
        tilingKeyConfig_.antiquantType = static_cast<uint8_t>(QuantType::PER_CHANNEL);
    }

    tilingKeyConfig_.quantType = static_cast<uint8_t>(QuantType::NONE);
    tilingKeyConfig_.optionInputSituation = static_cast<uint8_t>(hasAntiquantOffset_) << 1;
    tilingKeyConfig_.weightFormat =
        weightNzFlag_ ? static_cast<uint8_t>(WeightFormat::FRACTAL_NZ) : static_cast<uint8_t>(WeightFormat::ND);
    tilingKeyConfig_.templateCustom = static_cast<uint8_t>(Mte2Configuration::MTE2_INNER_SIZE_256_BUF_NUM_4);
    if (IsA16W4NDPergroup()) {
        tilingKeyConfig_.templateCustom = static_cast<uint8_t>(Mte2Configuration::MTE2_INNER_SIZE_256_BUF_NUM_2);
    }
    if (xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT4) {
        tilingKeyConfig_.templateCustom = static_cast<uint8_t>(Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_DEFAULT);
        if (groupSize_ == 192u) {
            tilingKeyConfig_.templateCustom = static_cast<uint8_t>(Mte2Configuration::MTE2_INNER_SIZE_384_BUF_NUM_3);
        }
    } else if (xDType_ == ge::DT_FLOAT8_E4M3FN &&
               (weightDtype_ == ge::DT_FLOAT4_E2M1 || weightDtype_ == ge::DT_FLOAT4_E1M2) &&
               antiquantScaleDtype_ == ge::DT_FLOAT8_E8M0) {
        tilingKeyConfig_.templateCustom = static_cast<uint8_t>(Mte2Configuration::MTE2_INNER_SIZE_DYNAMIC_BUF_NUM_4);
        tilingKeyConfig_.isSingleMultiSingle = isSingleMultiSingle_;
    }
    tilingKeyConfig_.apiConstexpr = 0U;
    context->SetTilingKey(tilingKeyConfig_.GenTilingKey());
}

bool GroupedWeightQuantBatchMatmulTiling::SetCustomParam(gert::TilingContext *context)
{
    size_t *workspaces = context->GetWorkspaceSizes(1); // get second variable
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context->GetNodeName(), "workspaces is nullptr."),
                return false); // check workspaces is not null
    workspaces[0] = DEFAULT_WORKSPACE_SIZE;

    context->SetBlockDim(coreNum_);
    OP_CHECK_IF(context->GetRawTilingData() == nullptr, OP_LOGE(context->GetNodeName(), "RawTilingData is nullptr."),
                return false);
    errno_t ret = memcpy_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(),
                           static_cast<void *>(&tilingData_), sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(context->GetNodeName(), "memcpy_s failed, ret = %d", ret);
        return false;
    }
    context->GetRawTilingData()->SetDataSize(sizeof(tilingData_));
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::IsA16W4ND() const
{
    if (ge::GetSizeByDataType(xDType_) == B16_DATA_SIZE &&
        ((weightDtype_ == ge::DT_INT4) || (weightDtype_ == ge::DT_INT32)) && !weightNzFlag_) {
        return true;
    }
    return false;
}

bool GroupedWeightQuantBatchMatmulTiling::IsA16W4NDPergroup() const
{
    return IsA16W4ND() && groupSize_ > 0;
}

bool GroupedWeightQuantBatchMatmulTiling::IsMxA8W4() const
{
    if (xDType_ == ge::DT_FLOAT8_E4M3FN &&
        (weightDtype_ == ge::DT_FLOAT4_E2M1 || weightDtype_ == ge::DT_FLOAT4_E1M2 || weightDtype_ == ge::DT_FLOAT) &&
        antiquantScaleDtype_ == ge::DT_FLOAT8_E8M0) {
        return true;
    }
    return false;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckUnsupportDataFlow() const
{
    if (ge::GetSizeByDataType(xDType_) == B16_DATA_SIZE &&
        (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT32 || weightDtype_ == ge::DT_INT4)) {
        OP_CHECK_IF(weightNzFlag_,
                    OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "weight", "FRACTAL_NZ",
                                               "ND (In weight quant, when x-weight is bf16/fp16-int8/int4/int32)"),
                    return false);
    } else if (ge::GetSizeByDataType(xDType_) == B16_DATA_SIZE &&
               (weightDtype_ == ge::DT_FLOAT8_E4M3FN || weightDtype_ == ge::DT_FLOAT8_E5M2 ||
                weightDtype_ == ge::DT_HIFLOAT8)) {
        OP_CHECK_IF(!(!weightNzFlag_ && transB_),
                    OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "weight",
                                               std::string("format=") + (weightNzFlag_ ? "NZ" : "ND") +
                                                   ", transpose=" + (transB_ ? "true" : "false"),
                                               "transposed and ND-format "
                                               "(In weight quant, when x-weight is bf16/fp16-fp8/hif8)"),
                    return false);
    } else if (ge::GetSizeByDataType(xDType_) == B16_DATA_SIZE &&
               (weightDtype_ == ge::DT_FLOAT4_E2M1 || weightDtype_ == ge::DT_FLOAT4_E1M2 ||
                weightDtype_ == ge::DT_FLOAT) &&
               antiquantScaleDtype_ == ge::DT_FLOAT8_E8M0) {
        OP_CHECK_IF(!(weightNzFlag_ && !transB_),
                    OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "weight",
                                               std::string("format=") + (weightNzFlag_ ? "NZ" : "ND") +
                                                   ", transpose=" + (transB_ ? "true" : "false"),
                                               "untransposed and FRACTAL_NZ-format "
                                               "(In weight quant, when x-weight is bf16/fp16-fp4/fp16)"),
                    return false);
    } else {
        return CheckUnsupportedRemainingCases();
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckUnsupportedRemainingCases() const
{
    if (xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT4) {
        OP_CHECK_IF(!(weightNzFlag_ && !transB_),
                    OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "weight",
                                               std::string("format=") + (weightNzFlag_ ? "NZ" : "ND") +
                                                   ", transpose=" + (transB_ ? "true" : "false"),
                                               "untransposed and FRACTAL_NZ-format "
                                               "(In weight quant, when x-weight is int8-int4)"),
                    return false);
    } else if (xDType_ == ge::DT_FLOAT8_E4M3FN &&
               (weightDtype_ == ge::DT_FLOAT4_E2M1 || weightDtype_ == ge::DT_FLOAT4_E1M2 ||
                weightDtype_ == ge::DT_FLOAT) &&
               antiquantScaleDtype_ == ge::DT_FLOAT8_E8M0) {
        OP_CHECK_IF(!(weightNzFlag_ && transB_),
                    OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "weight",
                                               std::string("format=") + (weightNzFlag_ ? "NZ" : "ND") +
                                                   ", transpose=" + (transB_ ? "true" : "false"),
                                               "transposed and FRACTAL_NZ-format "
                                               "(In weight quant, when x-weight is "
                                               "float8_e4m3fn-float4_e2m1/float4_e1m2/fp32)"),
                    return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckAntiQuantDtype() const
{
    if (IsA16W4ND()) {
        OP_CHECK_IF(
            antiquantScaleDtype_ != xDType_,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(OP_NAME, "antiquantScale, x",
                                                   ge::TypeUtils::DataTypeToSerialString(antiquantScaleDtype_) + ", " +
                                                       ge::TypeUtils::DataTypeToSerialString(xDType_),
                                                   "The dtype of antiquantScale must be the same as the dtype of x"),
            return false);
        if (hasAntiquantOffset_) {
            OP_CHECK_IF(antiquantOffsetDtype_ != xDType_,
                        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                            OP_NAME, "antiquantOffset, x",
                            ge::TypeUtils::DataTypeToSerialString(antiquantOffsetDtype_) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(xDType_),
                            "The dtype of antiquantOffset must be the same as the dtype of x"),
                        return false);
        }
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckPerTokenScale(const gert::TilingContext *context) const
{
    if (IsMxA8W4()) {
        auto perTokenScaleShapePtr = context->GetDynamicInputShape(GroupedMatmul::PER_TOKEN_SCALE_INDEX, 0);
        OP_CHECK_IF(perTokenScaleShapePtr == nullptr,
                    OP_LOGE(context->GetNodeName(), "perTokenScaleShapePtr is nullptr."), return false);
        gert::Shape perTokenScaleShape = perTokenScaleShapePtr->GetStorageShape();
        int64_t perTokenScaleDimNum = perTokenScaleShape.GetDimNum();
        OP_CHECK_IF(
            perTokenScaleDimNum != 3,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "perTokenScale", std::to_string(perTokenScaleDimNum),
                                                     "The shape dim of perTokenScaleShape must be 3"),
            return false);
        OP_CHECK_IF(perTokenScaleShape.GetDim(0) != static_cast<int64_t>(mSize_) ||
                        perTokenScaleShape.GetDim(1) != static_cast<int64_t>(kSize_) / (MX_GROUP_SIZE * 2) ||
                        perTokenScaleShape.GetDim(2) != 2UL,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        OP_NAME, "perTokenScale", Ops::Base::ToString(perTokenScaleShape),
                        "The shape of perTokenScale must be [" + std::to_string(mSize_) + ", " +
                            std::to_string(static_cast<int64_t>(kSize_) / (MX_GROUP_SIZE * 2)) + ", 2]"),
                    return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckBiasDtype() const
{
    if (IsA16W4ND()) {
        if (hasBias_) {
            OP_CHECK_IF(
                BIAS_TYPE_SUPPORT_MAP.find(xDType_) == BIAS_TYPE_SUPPORT_MAP.end(),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "x", ge::TypeUtils::DataTypeToSerialString(xDType_),
                                                      "Cannot find bias dtype match with dtype of x"),
                return false);
            OP_CHECK_IF(BIAS_TYPE_SUPPORT_MAP.at(xDType_).find(biasDtype_) == BIAS_TYPE_SUPPORT_MAP.at(xDType_).end(),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            OP_NAME, "bias", ge::TypeUtils::DataTypeToSerialString(biasDtype_),
                            "The dtype of bias cannot be " + ge::TypeUtils::DataTypeToSerialString(biasDtype_) +
                                ", when the dtype of x is " + ge::TypeUtils::DataTypeToSerialString(xDType_)),
                        return false);
        }
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckGroupTypeAndSplitItem() const
{
    if (IsA16W4ND()) {
        OP_CHECK_IF((groupType_ != static_cast<int64_t>(GroupType::NO_SPLIT)) &&
                        (groupType_ != static_cast<int64_t>(GroupType::SPLIT_M)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        OP_NAME, "groupType", std::to_string(groupType_),
                        "When the dtype of x-weight is bf16/fp16-int4/int32, groupType must be -1 or 0"),
                    return false);
        if (groupType_ == static_cast<int64_t>(GroupType::NO_SPLIT)) {
            OP_CHECK_IF((splitItem_ != 0 && splitItem_ != 1),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "splitItem", std::to_string(splitItem_),
                                                              "When groupType is -1, splitItem must be 0 or 1"),
                        return false);
        } else {
            OP_CHECK_IF((splitItem_ != 2 && splitItem_ != 3),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "splitItem", std::to_string(splitItem_),
                                                              "When groupType is 0, splitItem must be 2 or 3"),
                        return false);
        }
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckTransposeStatus() const
{
    OP_CHECK_IF(transA_,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "transposeX", "true", "TransposedX is not supported"),
                return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckEmptyTensor(const gert::TilingContext *context) const
{
    if (isSingleMultiSingle_) {
        return CheckEmptyTensorSingleXMultiWeightSingleY(context);
    }
    return CheckEmptyTensorDefault(context);
}

bool GroupedWeightQuantBatchMatmulTiling::CheckEmptyTensorSingleXMultiWeightSingleY(
    const gert::TilingContext *context) const
{
    auto xShapePtr = context->GetDynamicInputShape(X_IDX, 0);
    OP_CHECK_IF(xShapePtr == nullptr, OP_LOGE(context->GetNodeName(), "xShapePtr is nullptr."), return false);
    auto xShape = xShapePtr->GetStorageShape();
    size_t xDimNum = xShape.GetDimNum();
    OP_CHECK_IF(xDimNum < MIN_X_DIM || xDimNum > MAX_X_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "x[0]", std::to_string(xDimNum),
                                                         "The shape dim of x[0] must be within the range 2-6"),
                return false);
    // -1含义为(K, M)场景M索引，-2含义为(M, K)场景M索引
    int64_t m = transA_ ? xShape.GetDim(xDimNum - 1) : xShape.GetDim(xDimNum - 2);
    // -2含义为(K, M)场景K索引，-1含义为(M, K)场景K索引
    int64_t k = transA_ ? xShape.GetDim(xDimNum - 2) : xShape.GetDim(xDimNum - 1);
    // -2含义：x的最后2维为M和K，对除M, K的batch轴进行累乘
    for (size_t dimIdx = 0; dimIdx < xDimNum - 2; dimIdx++) {
        m *= xShape.GetDim(dimIdx);
    }
    for (size_t i = 0; i < numWeight_; ++i) {
        auto wShapePtr = context->GetDynamicInputShape(WEIGHT_IDX, i);
        OP_CHECK_IF(wShapePtr == nullptr, OP_LOGE(context->GetNodeName(), "wShapePtr is nullptr."), return false);
        auto wShape = wShapePtr->GetStorageShape();
        size_t wDimNum = wShape.GetDimNum();
        OP_CHECK_IF(wDimNum < GroupedMatmul::MIN_ND_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "weight", std::to_string(wDimNum),
                                                             "The shape dim of weight must be >= 2"),
                    return false);
        if (weightNzFlag_) {
            OP_CHECK_IF(
                wDimNum != 4,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    OP_NAME, "weight[" + std::to_string(i) + "]", std::to_string(wDimNum),
                    "The shape dim of weight must be 4 when x-weight is float8_e4m3fn-float4_e2m1/fp32 in NZ mode"),
                return false);
        }
        // -2含义为(N, K)场景N索引，-1含义为(K, N)场景N索引
        int64_t n = transB_ ? wShape.GetDim(wDimNum - 2) : wShape.GetDim(wDimNum - 1);
        if (weightNzFlag_) {
            // 非转置NZ排布(N1, K1, K0, N0), 转置NZ排布(K1, N1, N0, K0)
            n = transB_ ? wShape.GetDim(wDimNum - 3) * wShape.GetDim(wDimNum - 2) :
                          wShape.GetDim(wDimNum - 4) * wShape.GetDim(wDimNum - 1);
        }
        // k不能单独为0
        OP_CHECK_IF(
            ((m != 0) && (n != 0)) && k == 0,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                OP_NAME, "x[0], weight[" + std::to_string(i) + "]",
                Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(wShape),
                "K of x[0] cannot be 0 when M of x[0] is not 0 and N of weight[" + std::to_string(i) + "] is not 0"),
            return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckEmptyTensorDefault(const gert::TilingContext *context) const
{
    size_t numGroups = std::min(numX_, numWeight_);
    for (size_t i = 0; i < numGroups; ++i) {
        auto xShapePtr = context->GetDynamicInputShape(X_IDX, i);
        auto xShape = xShapePtr->GetStorageShape();
        size_t xDimNum = xShape.GetDimNum();
        OP_CHECK_IF(xDimNum < MIN_X_DIM || xDimNum > MAX_X_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        OP_NAME, "x[" + std::to_string(i) + "]", std::to_string(xDimNum),
                        "The shape dim of x[" + std::to_string(i) + "] must be within the range 2-6"),
                    return false);
        // -1含义为(K, M)场景M索引，-2含义为(M, K)场景M索引
        int64_t m = transA_ ? xShape.GetDim(xDimNum - 1) : xShape.GetDim(xDimNum - 2);
        // -2含义为(K, M)场景K索引，-1含义为(M, K)场景K索引
        int64_t k = transA_ ? xShape.GetDim(xDimNum - 2) : xShape.GetDim(xDimNum - 1);
        // -2含义：x的最后2维为M和K，对除M, K的batch轴进行累乘
        for (size_t dimIdx = 0; dimIdx < xDimNum - 2; dimIdx++) {
            m *= xShape.GetDim(dimIdx);
        }
        auto wShapePtr = context->GetDynamicInputShape(WEIGHT_IDX, i);
        auto wShape = wShapePtr->GetOriginShape();
        size_t wDimNum = wShape.GetDimNum();
        OP_CHECK_IF(wDimNum < GroupedMatmul::MIN_ND_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "weight", std::to_string(wDimNum),
                                                             "The shape dim of weight must be >= 2"),
                    return false);
        // -2含义为(N, K)场景N索引，-1含义为(K, N)场景N索引
        int64_t n = transB_ ? wShape.GetDim(wDimNum - 2) : wShape.GetDim(wDimNum - 1);
        // k不能单独为0
        OP_CHECK_IF(((m != 0) && (n != 0)) && k == 0,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        OP_NAME, "x[" + std::to_string(i) + "], weight[" + std::to_string(i) + "]",
                        Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(wShape),
                        "K of x[" + std::to_string(i) + "] cannot be 0 when M of x[" + std::to_string(i) +
                            "] is not 0 and N of weight[" + std::to_string(i) + "] is not 0"),
                    return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::SetShapeListSplitMSingleXSingleWeightSingleY(
    const gert::TilingContext *context)
{
    auto xTensor = context->GetDynamicInputTensor(X_IDX, 0); // 0: get first tensor
    OP_CHECK_IF(xTensor == nullptr, OP_LOGE(context->GetNodeName(), "xTensor is nullptr."), return false);
    gert::Shape xShape = xTensor->GetStorageShape();

    auto wTensor = context->GetDynamicInputTensor(WEIGHT_IDX, 0); // 0: get first tensor
    OP_CHECK_IF(wTensor == nullptr, OP_LOGE(context->GetNodeName(), "wTensor is nullptr."), return false);
    gert::Shape wShape = wTensor->GetStorageShape();

    groupNum_ = static_cast<int32_t>(wShape.GetDim(0));
    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
    uint32_t xDimNum = static_cast<uint32_t>(xShape.GetDimNum());
    mSize_ = transA_ ? xShape.GetDim(1) : xShape.GetDim(0);
    kSize_ = transA_ ? xShape.GetDim(0) : xShape.GetDim(xDimNum - 1);
    // -1含义为(K, N)场景N索引，-2含义为(N, K)场景N索引
    nSize_ = transB_ ? wShape.GetDim(wDimNum - 2) : wShape.GetDim(wDimNum - 1);
    if (weightNzFlag_) {
        // 非转置NZ排布(N1, K1, K0, N0), 转置NZ排布(K1, N1, N0, K0)
        // -3含义：转置N1索引；-2含义：转置N0索引
        nSize_ = transB_ ? wShape.GetDim(wDimNum - 3) * wShape.GetDim(wDimNum - 2)
                           // -4含义：非转置N1索引；-1含义：非转置N0索引
                           :
                           wShape.GetDim(wDimNum - 4) * wShape.GetDim(wDimNum - 1);
        const gert::StorageShape *yShapePtr = context->GetOutputShape(0);
        OP_CHECK_IF(yShapePtr == nullptr, OP_LOGE(context->GetNodeName(), "yShapePtr is nullptr."), return false);
        const gert::Shape &yShape = yShapePtr->GetOriginShape();
        OP_CHECK_IF(!GetDimFromEnd(yShape, 1, nSizeOri_),
                    OP_LOGE(context->GetNodeName(), "Get nSizeOri_ from y last dim failed."), return false);
    }

    if (weightDtype_ == ge::DT_FLOAT || weightDtype_ == ge::DT_INT32) {
        weightDtype_ = weightDtype_ == ge::DT_FLOAT ? ge::DT_FLOAT4_E2M1 : ge::DT_INT4;
        if (!transB_) {
            // 一个float32/int32表示8个fp4/int4，设置为正确shape；kSize来自x，不需要考虑转置场景k轴扩大
            nSize_ = FP4_PER_FP32 * nSize_;
        }
    }
    if (!weightNzFlag_) {
        nSizeOri_ = nSize_;
    }
    kList_[0] = static_cast<int32_t>(kSize_);
    nList_[0] = static_cast<int32_t>(nSizeOri_);
    mList_[0] = -1;
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::SetShapeListSplitMSingleXMultiWeightSingleY(
    const gert::TilingContext *context)
{
    auto xTensor = context->GetDynamicInputTensor(X_IDX, 0); // 0: get first tensor
    OP_CHECK_IF(xTensor == nullptr, OP_LOGE(context->GetNodeName(), "xTensor is nullptr."), return false);
    gert::Shape xShape = xTensor->GetStorageShape();
    uint32_t xDimNum = static_cast<uint32_t>(xShape.GetDimNum());
    mSize_ = transA_ ? xShape.GetDim(1) : xShape.GetDim(0);
    if (transA_) {
        OP_CHECK_IF(!GetDimFromEnd(xShape, xDimNum, kSize_),
                    OP_LOGE(context->GetNodeName(), "Get kSize_ from x dim0 failed."), return false);
    } else {
        OP_CHECK_IF(!GetDimFromEnd(xShape, 1, kSize_),
                    OP_LOGE(context->GetNodeName(), "Get kSize_ from x last dim failed."), return false);
    }

    if (weightNzFlag_) {
        const gert::StorageShape *yShapePtr = context->GetOutputShape(0);
        OP_CHECK_IF(yShapePtr == nullptr, OP_LOGE(context->GetNodeName(), "yShapePtr is nullptr."), return false);
        const gert::Shape &yShape = yShapePtr->GetOriginShape();
        OP_CHECK_IF(!GetDimFromEnd(yShape, 1, nSizeOri_),
                    OP_LOGE(context->GetNodeName(), "Get nSizeOri_ from y last dim failed."), return false);
    }

    bool isFp4PackType = weightDtype_ == ge::DT_FLOAT || weightDtype_ == ge::DT_INT32;
    for (uint16_t i = 0; i < GroupedMatmul::MAX_TENSOR_CONT; i++) {
        auto wTensor = context->GetDynamicInputTensor(WEIGHT_IDX, i);
        if (wTensor == nullptr) {
            break;
        }
        gert::Shape wShape = wTensor->GetStorageShape();
        uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
        // -1含义为(K, N)场景N索引，-2含义为(N, K)场景N索引
        uint64_t nSize = 0;
        if (transB_) {
            OP_CHECK_IF(!GetDimFromEnd(wShape, 2, nSize),
                        OP_LOGE(context->GetNodeName(), "Get nSize from w dim[-2] failed."), return false);
        } else {
            OP_CHECK_IF(!GetDimFromEnd(wShape, 1, nSize),
                        OP_LOGE(context->GetNodeName(), "Get nSize from w last dim failed."), return false);
        }
        if (weightNzFlag_) {
            // 非转置NZ排布(N1, K1, K0, N0), 转置NZ排布(K1, N1, N0, K0)
            // -3含义：转置N1索引；-2含义：转置N0索引
            nSize = transB_ ? wShape.GetDim(wDimNum - 3) * wShape.GetDim(wDimNum - 2)
                              // -4含义：非转置N1索引；-1含义：非转置N0索引
                              :
                              wShape.GetDim(wDimNum - 4) * wShape.GetDim(wDimNum - 1);
        }
        if (isFp4PackType && !transB_) {
            // 一个float32/int32表示8个fp4/int4，设置为正确shape；kSize来自x，不需要考虑转置场景k轴扩大
            nSize = static_cast<uint64_t>(8) * nSize;
        }
        if (!weightNzFlag_) {
            nSizeOri_ = nSize;
        }
        groupNum_ += 1U;
        kList_[i] = static_cast<int32_t>(kSize_);
        nList_[i] = static_cast<int32_t>(nSizeOri_);
        mList_[i] = -1;
        nSize_ = std::max(nSize_, nSize);
    }
    if (isFp4PackType) {
        weightDtype_ = weightDtype_ == ge::DT_FLOAT ? ge::DT_FLOAT4_E2M1 : ge::DT_INT4;
    }
    return true;
}

uint16_t GroupedWeightQuantBatchMatmulTiling::GetTensorListSize(const gert::TilingContext *context,
                                                                uint32_t attrIdx) const
{
    uint16_t count = 0;
    for (int i = 0; i <= GroupedMatmul::MAX_TENSOR_CONT; i++) {
        auto shapePtr = context->GetDynamicInputShape(attrIdx, count);
        if (shapePtr == nullptr) {
            break;
        }
        ++count;
    }
    return count;
}

void GroupedWeightQuantBatchMatmulTiling::GetNumOfInputs(const gert::TilingContext *context)
{
    numX_ = GetTensorListSize(context, X_IDX);
    numWeight_ = GetTensorListSize(context, WEIGHT_IDX);
    numBias_ = GetTensorListSize(context, BIAS_IDX);
    numAntiquantScale_ = GetTensorListSize(context, ANTIQUANT_SCALE_IDX);
    numAntiquantOffset_ = GetTensorListSize(context, ANTIQUANT_OFFSET_IDX);
}

bool GroupedWeightQuantBatchMatmulTiling::SetShapeListMultiXMultiWeightMultiY(const gert::TilingContext *context)
{
    for (uint16_t i = 0; i < GroupedMatmul::MAX_TENSOR_CONT; i++) {
        auto xShapePtr = context->GetDynamicInputShape(X_IDX, i);
        auto wShapePtr = context->GetDynamicInputShape(WEIGHT_IDX, i);
        if (xShapePtr == nullptr || wShapePtr == nullptr) {
            break;
        }
        auto xShape = xShapePtr->GetStorageShape();
        auto wShape = wShapePtr->GetOriginShape();
        uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
        uint32_t xDimNum = static_cast<uint32_t>(xShape.GetDimNum());
        groupNum_ += 1U;
        // -1含义为(K, M)场景M索引，-2含义为(M, K)场景M索引
        int64_t m = transA_ ? xShape.GetDim(xDimNum - 1) : xShape.GetDim(xDimNum - 2);
        // -2含义：x的最后2维为M和K，对除M, K的batch轴进行累乘
        for (uint16_t xDim = 0; xDim < static_cast<uint16_t>(xDimNum) - 2; xDim++) {
            m *= xShape.GetDim(xDim);
        }
        int64_t k = transB_ ? wShape.GetDim(1) : wShape.GetDim(0);
        int64_t n = transB_ ? wShape.GetDim(0) : wShape.GetDim(1);
        mList_[i] = static_cast<int32_t>(m);
        kList_[i] = static_cast<int32_t>(k);
        nList_[i] = static_cast<int32_t>(n);
        mSize_ = std::max(mSize_, static_cast<uint64_t>(m));
        kSize_ = std::max(kSize_, static_cast<uint64_t>(k));
        nSize_ = std::max(nSize_, static_cast<uint64_t>(n));
    }
    nSizeOri_ = nSize_;
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckGroupSize() const
{
    // bf16/fp16-int4/int32 ND pergroup supports groupSize 32/64/128/256
    if (IsA16W4NDPergroup()) {
        OP_CHECK_IF(groupSize_ != 32u && groupSize_ != 64u && groupSize_ != 128u && groupSize_ != 256u,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        OP_NAME, "groupSize", std::to_string(groupSize_),
                        "When x-weight is bf16/fp16-int4/int32 and antiquant type is pergroup, "
                        "groupSize must be 32/64/128/256"),
                    return false);
        return true;
    }
    if (xDType_ != ge::DT_INT8 || weightDtype_ != ge::DT_INT4) {
        return true;
    }
    // 伪量化S8S4场景支持groupsize为128/192/256/512
    OP_CHECK_IF(groupSize_ != 128u && groupSize_ != 256u && groupSize_ != 512u && groupSize_ != 192u,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "groupSize", std::to_string(groupSize_),
                                                      "groupSize must be 128/192/256/512"),
                return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::SetAntiquantGroupSize(const gert::TilingContext *context)
{
    groupSize_ = 0;
    auto antiquantScale = context->GetDynamicInputTensor(ANTIQUANT_SCALE_IDX, 0);
    OP_CHECK_IF(antiquantScale == nullptr, OP_LOGE(context->GetNodeName(), "antiquantScale is nullptr."), return false);

    if (IsA16W4ND() && !isSingleX_) {
        OP_CHECK_IF(!DeriveGroupSizeMulti(context), OP_LOGE(context->GetNodeName(), "DeriveGroupSizeMulti failed."),
                    return false);
    } else {
        OP_CHECK_IF(!DeriveGroupSizeSingle(context), OP_LOGE(context->GetNodeName(), "DeriveGroupSizeSingle failed."),
                    return false);
    }

    OP_CHECK_IF(!CheckGroupSize(), OP_LOGE(context->GetNodeName(), "CheckGroupSize failed."), return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::DeriveGroupSizeSingle(const gert::TilingContext *context)
{
    auto antiquantScale = context->GetDynamicInputTensor(ANTIQUANT_SCALE_IDX, 0);
    auto antiquantScaleShape = antiquantScale->GetStorageShape();
    int64_t antiquantScaleDimNum = antiquantScaleShape.GetDimNum();

    int64_t groupNum = 0;
    if (isSingleMultiSingle_ && antiquantScaleDimNum == ANTIQUANT_SCALE_DIM_NUM - 1) {
        // antiquantScaleShape: (n,k/64,2), compared with (g,n,k/64,2) missing g axis
        groupNum = antiquantScaleShape.GetDim(antiquantScaleDimNum - PENULTIMATE_DIM) * MX_GROUP_FACTOR;
    } else if (antiquantScaleDimNum == A16W4_SINGLE_PER_GROUP_DIM) {
        if (IsA16W4ND()) {
            // bf16/fp16-int4/int32 pergroup: scale is [E, G, N], G is always at dim 1
            groupNum = antiquantScaleShape.GetDim(1);
        } else {
            // (g, K, N) format: K axis index depends on transB
            groupNum = transB_ ? antiquantScaleShape.GetDim(antiquantScaleDimNum - 1) :
                                 antiquantScaleShape.GetDim(antiquantScaleDimNum - 2);
        }
    } else if (antiquantScaleDimNum == ANTIQUANT_SCALE_DIM_NUM) {
        // antiquantScaleShape: (g,n,k/64,2) or (g,k/64,n,2)
        groupNum = transB_ ? antiquantScaleShape.GetDim(antiquantScaleDimNum - PENULTIMATE_DIM) * MX_GROUP_FACTOR :
                             antiquantScaleShape.GetDim(antiquantScaleDimNum - ANTEPENULTIMATE_DIM) * MX_GROUP_FACTOR;
    } else {
        OP_CHECK_IF(IsA16W4ND() && !CheckAntiquantOffsetMatchScale(context, 0, antiquantScaleShape),
                    OP_LOGE(context->GetNodeName(), "CheckAntiquantOffsetMatchScale failed for tensor[0]."),
                    return false);
        return true;
    }

    OP_CHECK_IF(
        groupNum <= 0 || kSize_ % groupNum > 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "antiquantScaleShape", Ops::Base::ToString(antiquantScaleShape),
                                              "The groupNum[" + std::to_string(groupNum) +
                                                  "] of antiquantScaleShape must be "
                                                  "greater than 0 and divisible by kSize[" +
                                                  std::to_string(kSize_) + "]"),
        return false);
    groupSize_ = groupNum > 0 ? kSize_ / static_cast<uint64_t>(groupNum) : 0;
    OP_CHECK_IF(IsA16W4ND() && !CheckAntiquantOffsetMatchScale(context, 0, antiquantScaleShape),
                OP_LOGE(context->GetNodeName(), "CheckAntiquantOffsetMatchScale failed for tensor[0]."), return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::DeriveGroupSizeMulti(const gert::TilingContext *context)
{
    gert::Shape scaleShape0;
    OP_CHECK_IF(!GetA16W4MultiScaleShape(context, 0, scaleShape0),
                OP_LOGE(context->GetNodeName(), "GetA16W4MultiScaleShape failed for tensor[0]."), return false);
    int64_t scaleDimNum0 = scaleShape0.GetDimNum();
    OP_CHECK_IF(scaleDimNum0 != A16W4_MULTI_PER_CHANNEL_DIM && scaleDimNum0 != A16W4_MULTI_PER_GROUP_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    OP_NAME, "antiquantScale[0]", std::to_string(scaleDimNum0),
                    "When x-weight is bf16/fp16-int4/int32 and groupType is -1, the dimension of antiquantScale[0] "
                    "must be 1 (perchannel) or 2 (pergroup)"),
                return false);

    if (scaleDimNum0 == A16W4_MULTI_PER_CHANNEL_DIM) {
        OP_CHECK_IF(!CheckMultiA16W4PerChannelShape(context),
                    OP_LOGE(context->GetNodeName(), "CheckMultiA16W4PerChannelShape failed."), return false);
        groupSize_ = 0;
        return true;
    }

    int64_t k0 = static_cast<int64_t>(kList_[0]);
    OP_CHECK_IF(!DeriveA16W4MultiPergroupSizeFromScale(scaleShape0, k0, 0, groupSize_),
                OP_LOGE(context->GetNodeName(), "DeriveA16W4MultiPergroupSizeFromScale failed for tensor[0]."),
                return false);

    for (uint16_t i = 0; i < groupNum_; i++) {
        OP_CHECK_IF(!CheckA16W4MultiPergroupScaleShape(context, i, scaleDimNum0, groupSize_),
                    OP_LOGE(context->GetNodeName(), "CheckA16W4MultiPergroupScaleShape failed for tensor[%u].", i),
                    return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::GetA16W4MultiScaleShape(const gert::TilingContext *context, uint16_t idx,
                                                                  gert::Shape &scaleShape) const
{
    auto scaleShapePtr = context->GetDynamicInputShape(ANTIQUANT_SCALE_IDX, idx);
    OP_CHECK_IF(scaleShapePtr == nullptr, OP_LOGE(context->GetNodeName(), "antiquantScale[%u] shape is nullptr.", idx),
                return false);
    scaleShape = scaleShapePtr->GetStorageShape();
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::DeriveA16W4MultiPergroupSizeFromScale(const gert::Shape &scaleShape,
                                                                                int64_t kSize, uint16_t idx,
                                                                                uint32_t &groupSize) const
{
    int64_t scaleGroupNum = scaleShape.GetDim(0);
    OP_CHECK_IF(scaleGroupNum <= 0 || kSize <= 0 || kSize % scaleGroupNum != 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    OP_NAME, "antiquantScale[" + std::to_string(idx) + "]", Ops::Base::ToString(scaleShape),
                    "The groupNum[" + std::to_string(scaleGroupNum) + "] of antiquantScale[" + std::to_string(idx) +
                        "] must be greater than 0 and kSize[" + std::to_string(kSize) + "] of tensor[" +
                        std::to_string(idx) + "] must be divisible by it"),
                return false);
    groupSize = static_cast<uint32_t>(kSize / scaleGroupNum);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckA16W4MultiPergroupScaleShape(const gert::TilingContext *context,
                                                                            uint16_t idx, int64_t expectedDimNum,
                                                                            uint32_t expectedGroupSize) const
{
    gert::Shape scaleShape;
    OP_CHECK_IF(!GetA16W4MultiScaleShape(context, idx, scaleShape),
                OP_LOGE(context->GetNodeName(), "GetA16W4MultiScaleShape failed for tensor[%u].", idx), return false);
    OP_CHECK_IF(scaleShape.GetDimNum() != expectedDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    OP_NAME, "antiquantScale[" + std::to_string(idx) + "]", std::to_string(scaleShape.GetDimNum()),
                    "The quantization mode of all antiquantScale tensors must be the same as antiquantScale[0]. "
                    "Expected dimension " +
                        std::to_string(expectedDimNum)),
                return false);

    uint32_t groupSize = 0;
    int64_t kSize = static_cast<int64_t>(kList_[idx]);
    OP_CHECK_IF(!DeriveA16W4MultiPergroupSizeFromScale(scaleShape, kSize, idx, groupSize),
                OP_LOGE(context->GetNodeName(), "DeriveA16W4MultiPergroupSizeFromScale failed for tensor[%u].", idx),
                return false);
    OP_CHECK_IF(groupSize != expectedGroupSize,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    OP_NAME, "antiquantScale[" + std::to_string(idx) + "]", std::to_string(groupSize),
                    "The groupSize[" + std::to_string(groupSize) + "] derived from antiquantScale[" +
                        std::to_string(idx) + "] must be equal to groupSize[" + std::to_string(expectedGroupSize) +
                        "] derived from antiquantScale[0]"),
                return false);

    int64_t scaleNDim = scaleShape.GetDim(1);
    int64_t nSize = static_cast<int64_t>(nList_[idx]);
    OP_CHECK_IF(
        scaleNDim != nSize,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "antiquantScale[" + std::to_string(idx) + "]", Ops::Base::ToString(scaleShape),
            "The N dimension[" + std::to_string(scaleNDim) + "] of antiquantScale[" + std::to_string(idx) +
                "] must be equal to nSize[" + std::to_string(nSize) + "] of weight[" + std::to_string(idx) + "]"),
        return false);
    OP_CHECK_IF(!CheckAntiquantOffsetMatchScale(context, idx, scaleShape),
                OP_LOGE(context->GetNodeName(), "CheckAntiquantOffsetMatchScale failed for tensor[%u].", idx),
                return false);
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckAntiquantOffsetMatchScale(const gert::TilingContext *context,
                                                                         uint16_t idx,
                                                                         const gert::Shape &scaleShape) const
{
    if (!hasAntiquantOffset_) {
        return true;
    }
    auto offsetShapePtr = context->GetDynamicInputShape(ANTIQUANT_OFFSET_IDX, idx);
    OP_CHECK_IF(offsetShapePtr == nullptr,
                OP_LOGE(context->GetNodeName(), "antiquantOffset[%u] shape is nullptr.", idx), return false);
    auto offsetShape = offsetShapePtr->GetStorageShape();
    OP_CHECK_IF(offsetShape.GetDimNum() != scaleShape.GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    OP_NAME, "antiquantOffset[" + std::to_string(idx) + "]", std::to_string(offsetShape.GetDimNum()),
                    "The dimension of antiquantOffset[" + std::to_string(idx) +
                        "] must be equal to the dimension of antiquantScale[" + std::to_string(idx) + "]"),
                return false);
    for (int64_t dimIdx = 0; dimIdx < offsetShape.GetDimNum(); ++dimIdx) {
        OP_CHECK_IF(
            offsetShape.GetDim(dimIdx) != scaleShape.GetDim(dimIdx),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                OP_NAME, "antiquantOffset[" + std::to_string(idx) + "], antiquantScale[" + std::to_string(idx) + "]",
                Ops::Base::ToString(offsetShape) + ", " + Ops::Base::ToString(scaleShape),
                "The shape of antiquantOffset[" + std::to_string(idx) +
                    "] must be equal to the shape of antiquantScale[" + std::to_string(idx) + "]"),
            return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckMultiA16W4PerChannelShape(const gert::TilingContext *context) const
{
    for (uint16_t i = 0; i < groupNum_; i++) {
        auto scaleShapePtr = context->GetDynamicInputShape(ANTIQUANT_SCALE_IDX, i);
        OP_CHECK_IF(scaleShapePtr == nullptr,
                    OP_LOGE(context->GetNodeName(), "antiquantScale[%u] shape is nullptr.", i), return false);
        auto scaleShape = scaleShapePtr->GetStorageShape();
        OP_CHECK_IF(scaleShape.GetDimNum() != A16W4_MULTI_PER_CHANNEL_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        OP_NAME, "antiquantScale[" + std::to_string(i) + "]", std::to_string(scaleShape.GetDimNum()),
                        "The quantization mode of all antiquantScale tensors must be the same as antiquantScale[0]. "
                        "Expected dimension 1 for perchannel"),
                    return false);
        int64_t nI = static_cast<int64_t>(nList_[i]);
        OP_CHECK_IF(
            scaleShape.GetDim(0) != nI,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "antiquantScale[" + std::to_string(i) + "]", Ops::Base::ToString(scaleShape),
                "The N dimension[" + std::to_string(scaleShape.GetDim(0)) + "] of antiquantScale[" + std::to_string(i) +
                    "] must be equal to nSize[" + std::to_string(nI) + "] of weight[" + std::to_string(i) + "]"),
            return false);
        OP_CHECK_IF(!CheckAntiquantOffsetMatchScale(context, i, scaleShape),
                    OP_LOGE(context->GetNodeName(), "CheckAntiquantOffsetMatchScale failed for tensor[%u].", i),
                    return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::GetC0Size(const gert::TilingContext *context, ge::DataType dtype,
                                                    uint64_t &c0Size) const
{
    if (dtype == ge::DT_INT4) {
        c0Size = AscendC::ONE_BLK_SIZE + AscendC::ONE_BLK_SIZE;
    } else {
        int64_t dtypeSize = ge::GetSizeByDataType(dtype);
        OP_CHECK_IF(dtypeSize <= 0,
                    OP_LOGE(context->GetNodeName(), "Invalid dtypeSize[%ld], expect greater than 0", dtypeSize),
                    return false);
        if (dtypeSize > 0) {
            c0Size = AscendC::ONE_BLK_SIZE / dtypeSize;
        }
    }
    return true;
}

void GroupedWeightQuantBatchMatmulTiling::CalcFullNumBlocksResplitTiling(uint64_t c0Size)
{
    resplitParam_.mainBlockSize = BASIC_BLOCK_BASE_N;
    resplitParam_.mainBlockCount = 0UL;
    if (nSize_ / (coreNum_ * static_cast<uint64_t>(BASIC_BLOCK_BASE_N)) > 1UL) {
        resplitParam_.mainBlockCount = nSize_ / (coreNum_ * static_cast<uint64_t>(BASIC_BLOCK_BASE_N)) - 1UL;
    }
    uint64_t tailSizeOri = nSize_ - resplitParam_.mainBlockCount * resplitParam_.mainBlockSize * coreNum_;
    uint64_t tailSize = tailSizeOri;
    if (weightNzFlag_ && c0Size != 0UL) {
        tailSize = tailSizeOri / c0Size;
    }
    if (tailSizeOri > coreNum_ * static_cast<uint64_t>(BASIC_BLOCK_BASE_N)) {
        // 2含义：当尾块大于核数*256，除以2倍核数以保证单核尾块大小小于256
        constexpr uint64_t resplitFactor = 2;
        resplitParam_.firstTailBlockSize = static_cast<uint16_t>(tailSize / (coreNum_ * resplitFactor));
        resplitParam_.secondTailBlockSize = static_cast<uint16_t>(resplitParam_.firstTailBlockSize + 1U);
        resplitParam_.secondTailBlockCount = static_cast<uint16_t>(tailSize % (coreNum_ * resplitFactor));
        resplitParam_.firstTailBlockCount =
            static_cast<uint16_t>(coreNum_ * resplitFactor - resplitParam_.secondTailBlockCount);
    } else {
        resplitParam_.firstTailBlockSize = static_cast<uint16_t>(tailSize / coreNum_);
        resplitParam_.secondTailBlockSize = static_cast<uint16_t>(resplitParam_.firstTailBlockSize + 1U);
        resplitParam_.secondTailBlockCount = static_cast<uint16_t>(tailSize % coreNum_);
        resplitParam_.firstTailBlockCount = static_cast<uint16_t>(coreNum_ - resplitParam_.secondTailBlockCount);
    }
    if (weightNzFlag_) {
        resplitParam_.firstTailBlockSize *= static_cast<uint16_t>(c0Size);
        resplitParam_.secondTailBlockSize *= static_cast<uint16_t>(c0Size);
    }
}

void GroupedWeightQuantBatchMatmulTiling::CalcNoFullNumBlocksResplitTiling(uint64_t c0Size)
{
    resplitParam_.mainBlockSize = BASIC_BLOCK_BASE_N;
    // 该场景无法维持主块分核，默认写0
    resplitParam_.mainBlockCount = 0UL;

    uint64_t mainBlkCount = GroupedMatmul::CeilDiv(nSize_, BASIC_BLOCK_BASE_N);
    if (groupNum_ * mainBlkCount >= coreNum_) {
        // 按照BASIC_BLOCK_BASE_N分核，但是可以在group方向凑多个基本块，则按照mainBlkCount做均匀分核
        resplitParam_.firstTailBlockSize =
            GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(nSize_, mainBlkCount), c0Size);
        resplitParam_.firstTailBlockCount = mainBlkCount;
        resplitParam_.secondTailBlockSize = 0;
        resplitParam_.secondTailBlockCount = 0;
        cubeNumBlocksN_ = static_cast<uint8_t>(mainBlkCount);
        return;
    }

    uint64_t taskNum = std::max(1UL, nSize_ / BASIC_BLOCK_BASE_N_MIN); // 实际任务数，必然小于核数
    cubeNumBlocksN_ = static_cast<uint8_t>(taskNum);
    if (weightNzFlag_ && c0Size != 0UL && taskNum != 0UL) {
        resplitParam_.firstTailBlockSize = static_cast<uint16_t>(nSize_ / c0Size / taskNum);
        resplitParam_.secondTailBlockSize = static_cast<uint16_t>(resplitParam_.firstTailBlockSize + 1U);
        resplitParam_.secondTailBlockCount = static_cast<uint16_t>(nSize_ / c0Size % taskNum);
        resplitParam_.firstTailBlockCount = static_cast<uint16_t>(taskNum - resplitParam_.secondTailBlockCount);
        resplitParam_.firstTailBlockSize *= static_cast<uint16_t>(c0Size);
        resplitParam_.secondTailBlockSize *= static_cast<uint16_t>(c0Size);
    } else if (taskNum != 0UL) {
        resplitParam_.firstTailBlockSize = static_cast<uint16_t>(nSize_ / taskNum);
        resplitParam_.secondTailBlockSize = static_cast<uint16_t>(resplitParam_.firstTailBlockSize + 1U);
        resplitParam_.secondTailBlockCount = static_cast<uint16_t>(nSize_ % taskNum);
        resplitParam_.firstTailBlockCount = static_cast<uint16_t>(taskNum - resplitParam_.secondTailBlockCount);
    }
}

bool GroupedWeightQuantBatchMatmulTiling::CheckResplitTilingResult(const gert::TilingContext *context) const
{
    OP_CHECK_IF(nSize_ >= static_cast<uint64_t>(BASIC_BLOCK_BASE_N_MIN) && resplitParam_.firstTailBlockCount > 0 &&
                    (resplitParam_.firstTailBlockSize < BASIC_BLOCK_BASE_N_MIN ||
                     resplitParam_.firstTailBlockSize > BASIC_BLOCK_BASE_N),
                OP_LOGE(context->GetNodeName(),
                        "Invalid resplit tiling result, expect [%u] <= firstTailBlockSize [%hu] <= [%u] ",
                        BASIC_BLOCK_BASE_N_MIN, resplitParam_.firstTailBlockSize, BASIC_BLOCK_BASE_N),
                return false);
    OP_CHECK_IF(nSize_ >= static_cast<uint64_t>(BASIC_BLOCK_BASE_N_MIN) && resplitParam_.secondTailBlockCount > 0 &&
                    (resplitParam_.secondTailBlockSize < BASIC_BLOCK_BASE_N_MIN ||
                     resplitParam_.secondTailBlockSize > BASIC_BLOCK_BASE_N),
                OP_LOGE(context->GetNodeName(),
                        "Invalid resplit tiling result, expect [%u] <= secondTailBlockSize [%hu] <= [%u] ",
                        BASIC_BLOCK_BASE_N_MIN, resplitParam_.secondTailBlockSize, BASIC_BLOCK_BASE_N),
                return false);
    return true;
}

void GroupedWeightQuantBatchMatmulTiling::PrintInputParam(const gert::TilingContext *context) const
{
    OP_LOGI(context->GetNodeName(),
            "Input params: coreNum: %u, gmm groupNum: %u, groupType: %d, groupListType: %lld, splitItem: %lld, "
            "antiquant-groupSize: %u, mSize: %llu, kSize: %llu, nSize: %llu, nSizeOri: %llu, transA: %s, transB: %s, "
            "isSingleX: %s, isSingleWeight: %s, isSingleY: %s, hasBias: %s, weightNzFlag: %s, hasAntiquantOffset: "
            "%s, xDtype: %s, weightDtype: %s",
            coreNum_, groupNum_, static_cast<int32_t>(groupType_), groupListType_, splitItem_, groupSize_, mSize_,
            kSize_, nSize_, nSizeOri_, transA_ ? "true" : "false", transB_ ? "true" : "false",
            isSingleX_ ? "true" : "false", isSingleWeight_ ? "true" : "false", isSingleY_ ? "true" : "false",
            hasBias_ ? "true" : "false", weightNzFlag_ ? "true" : "false", hasAntiquantOffset_ ? "true" : "false",
            ge::TypeUtils::DataTypeToSerialString(xDType_).c_str(),
            ge::TypeUtils::DataTypeToSerialString(weightDtype_).c_str());
}

void GroupedWeightQuantBatchMatmulTiling::PrintTilingResult(const gert::TilingContext *context)
{
    OP_LOGI(context->GetNodeName(),
            "Tiling result: groupNum: %u, coreNum: %u, kSize: %llu, nSize: %llu, singleX: %u, singleWeight: %u, "
            "singleY: %u, groupType: %d, groupListType: %u, hasBias: %u, groupSize: %u, mainBlockSize: %u, "
            "mainBlockCount: %llu, firstTailBlockSize: %u, secondTailBlockSize: %u, firstTailBlockCount: %u, "
            "secondTailBlockCount: %u",
            tilingData_.gmmWeightQuantParam.groupNum, tilingData_.gmmWeightQuantParam.coreNum,
            tilingData_.gmmWeightQuantParam.kSize, tilingData_.gmmWeightQuantParam.nSize,
            static_cast<uint32_t>(tilingData_.gmmWeightQuantParam.singleX),
            static_cast<uint32_t>(tilingData_.gmmWeightQuantParam.singleWeight),
            static_cast<uint32_t>(tilingData_.gmmWeightQuantParam.singleY),
            static_cast<int32_t>(tilingData_.gmmWeightQuantParam.groupType),
            static_cast<uint32_t>(tilingData_.gmmWeightQuantParam.groupListType),
            static_cast<uint32_t>(tilingData_.gmmWeightQuantParam.hasBias), tilingData_.gmmWeightQuantParam.groupSize,
            tilingData_.gmmWeightQuantParam.mainBlockSize, tilingData_.gmmWeightQuantParam.mainBlockCount,
            tilingData_.gmmWeightQuantParam.firstTailBlockSize, tilingData_.gmmWeightQuantParam.secondTailBlockSize,
            tilingData_.gmmWeightQuantParam.firstTailBlockCount, tilingData_.gmmWeightQuantParam.secondTailBlockCount);
}

uint64_t TilingKeyConfigure::GenTilingKey() const
{
    PrintTilingKeyLog();
    constexpr uint8_t DECIMAL = 10U;
    uint64_t transInfo = this->transposeSituation;
    bool atrans_ = (transInfo == static_cast<uint64_t>(GmmTrans::ATrans)) ||
                   (transInfo == static_cast<uint64_t>(GmmTrans::ABTrans));
    bool btrans_ = (transInfo == static_cast<uint64_t>(GmmTrans::BTrans)) ||
                   (transInfo == static_cast<uint64_t>(GmmTrans::ABTrans));
    return GET_TPL_TILING_KEY(
        static_cast<uint64_t>(this->weightFormat), static_cast<uint64_t>(this->optionInputSituation),
        static_cast<uint64_t>(this->quantType), static_cast<uint64_t>(this->antiquantType),
        static_cast<uint64_t>(btrans_), static_cast<uint64_t>(atrans_), static_cast<uint64_t>(this->templateCustom),
        static_cast<uint64_t>(this->isSingleMultiSingle), static_cast<uint64_t>(this->algorithm % DECIMAL),
        static_cast<uint64_t>(this->algorithm / DECIMAL));
}

using namespace Ops::Transformer::OpTiling;
using namespace GroupedMatmul;
using namespace optiling::GmmConstant;

GroupedS8S4BasicApiTiling::GroupedS8S4BasicApiTiling(gert::TilingContext *context)
    : GroupedQmmTiling(context)
{
    Reset();
}

void GroupedS8S4BasicApiTiling::Reset()
{
    hasOffset_ = false;
    specialWeightFormat_ = false;
    weightPackedInt32_ = false;
    expectedTokenNum_ = 0;
    userWorkspaceSize_ = 0;
    dequantMode_ = DequantMode::SYMMETRIC_PER_GROUP;
    tilingData_ = GroupedMatmulTilingData::GMMS8S4BasicApiTilingData();
}

bool GroupedS8S4BasicApiTiling::IsCapable()
{
    return inputParams_.aDtype == ge::DT_INT8 &&
           (inputParams_.bDtype == ge::DT_INT4 || inputParams_.bDtype == ge::DT_INT32) &&
           inputParams_.scaleDtype == ge::DT_UINT64;
}

ge::graphStatus GroupedS8S4BasicApiTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    Reset();
    return GroupedQmmTiling::GetShapeAttrsInfo();
}

bool GroupedS8S4BasicApiTiling::AnalyzeDtype()
{
    auto xDesc = context_->GetDynamicInputDesc(X_INDEX, 0);
    auto weightDesc = context_->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    auto outDesc = context_->GetOutputDesc(Y_INDEX);
    auto scaleDesc = context_->GetDynamicInputDesc(SCALE_INDEX, 0);
    auto perTokenScaleDesc = context_->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(xDesc == nullptr || weightDesc == nullptr || outDesc == nullptr || scaleDesc == nullptr ||
                    perTokenScaleDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "S8S4 BasicApi requires x, weight, scale, perTokenScale and out."),
                return false);

    inputParams_.aDtype = xDesc->GetDataType();
    inputParams_.bDtype = weightDesc->GetDataType();
    inputParams_.cDtype = outDesc->GetDataType();
    inputParams_.scaleDtype = scaleDesc->GetDataType();
    inputParams_.perTokenScaleDtype = perTokenScaleDesc->GetDataType();
    inputParams_.aFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat()));
    const auto weightFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(weightDesc->GetStorageFormat()));
    const bool weightIsNdCompatible =
        weightFormat == ge::FORMAT_ND || weightFormat == ge::FORMAT_NCL || weightFormat == ge::FORMAT_NCHW;
    OP_CHECK_IF(!weightIsNdCompatible && weightFormat != ge::FORMAT_FRACTAL_NZ,
                OP_LOGE(context_->GetNodeName(), "S8S4 BasicApi expects weight format ND, NCL, NCHW or FRACTAL_NZ."),
                return false);
    inputParams_.bFormat = weightFormat == ge::FORMAT_FRACTAL_NZ ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND;

    OP_CHECK_IF(inputParams_.aDtype != ge::DT_INT8 ||
                    (inputParams_.bDtype != ge::DT_INT4 && inputParams_.bDtype != ge::DT_INT32),
                OP_LOGE(context_->GetNodeName(), "S8S4 BasicApi expects x INT8 and weight INT4/packed INT32."),
                return false);
    weightPackedInt32_ = inputParams_.bDtype == ge::DT_INT32;
    OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_UINT64,
                OP_LOGE(context_->GetNodeName(), "S8S4 BasicApi expects scale UINT64."), return false);
    OP_CHECK_IF(inputParams_.perTokenScaleDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "S8S4 BasicApi expects perTokenScale FLOAT."), return false);
    OP_CHECK_IF(inputParams_.cDtype != ge::DT_BF16 && inputParams_.cDtype != ge::DT_FLOAT16,
                OP_LOGE(context_->GetNodeName(), "S8S4 BasicApi expects out BF16 or FLOAT16."), return false);
    OP_CHECK_IF(inputParams_.aFormat != ge::FORMAT_ND,
                OP_LOGE(context_->GetNodeName(), "S8S4 BasicApi expects x in ND format."), return false);
    auto biasShape = context_->GetDynamicInputShape(BIAS_INDEX, 0);
    inputParams_.hasBias = biasShape != nullptr && biasShape->GetStorageShape().GetShapeSize() != 0;
    auto biasDesc = context_->GetDynamicInputDesc(BIAS_INDEX, 0);
    OP_CHECK_IF(!inputParams_.hasBias || biasDesc == nullptr || biasDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "S8S4 BasicApi requires bias FLOAT with shape [E,N]."), return false);
    inputParams_.biasDtype = biasDesc->GetDataType();

    auto offsetShape = context_->GetDynamicInputShape(OFFSET_INDEX, 0);
    hasOffset_ = offsetShape != nullptr && offsetShape->GetStorageShape().GetShapeSize() != 0;
    if (hasOffset_) {
        auto offsetDesc = context_->GetDynamicInputDesc(OFFSET_INDEX, 0);
        OP_CHECK_IF(offsetDesc == nullptr || offsetDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE(context_->GetNodeName(), "Asymmetric S8S4 BasicApi expects offset FLOAT."), return false);
    }
    dequantMode_ = hasOffset_ ? DequantMode::ASYMMETRIC_PER_CHANNEL : DequantMode::SYMMETRIC_PER_GROUP;
    OP_CHECK_IF(hasOffset_ && inputParams_.cDtype == ge::DT_BF16,
                OP_LOGE(context_->GetNodeName(), "BF16 S8S4 BasicApi does not support a non-null offset."),
                return false);
    return true;
}

bool GroupedS8S4BasicApiTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is nullptr."), return false);
    const int64_t *splitItemPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SPLIT_ITEM);
    const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
    const bool *transposeXPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_X);
    const int64_t *groupTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPTYPE);
    const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
    const int64_t *actTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_ACT_TYPE);
    const auto tuningConfigPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_TUNING_CONFIG);

    inputParams_.splitItem = splitItemPtr != nullptr ? static_cast<int8_t>(*splitItemPtr) : inputParams_.splitItem;
    inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    inputParams_.transA = transposeXPtr != nullptr ? *transposeXPtr : false;
    inputParams_.groupType = groupTypePtr != nullptr ? static_cast<int8_t>(*groupTypePtr) : inputParams_.groupType;
    inputParams_.actType = actTypePtr != nullptr ? static_cast<int8_t>(*actTypePtr) : inputParams_.actType;
    inputParams_.groupListType =
        groupListTypePtr != nullptr ? static_cast<int8_t>(*groupListTypePtr) : GROUP_LIST_COUNT;
    OP_CHECK_IF(inputParams_.groupListType != GROUP_LIST_COUNT,
                OP_LOGE(context_->GetNodeName(), "groupListType must be 1(count)."), return false);
    OP_CHECK_IF(inputParams_.splitItem != X_SEPARATED && inputParams_.splitItem != NO_SEPARATED,
                OP_LOGE(context_->GetNodeName(), "S8S4 BasicApi expects splitItem to be 2 or 3."), return false);
    if (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > 0) {
        const auto tuningConfig = reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData());
        OP_CHECK_IF(tuningConfig[TUNING_CONFIG_EXPECTED_TOKEN_INDEX] < 0 ||
                        static_cast<uint64_t>(tuningConfig[TUNING_CONFIG_EXPECTED_TOKEN_INDEX]) > UINT32_MAX,
                    OP_LOGE(context_->GetNodeName(), "tuningConfig[0] must be in [0, UINT32_MAX]."), return false);
        expectedTokenNum_ = static_cast<uint32_t>(tuningConfig[TUNING_CONFIG_EXPECTED_TOKEN_INDEX]);
        if (tuningConfigPtr->GetSize() > TUNING_CONFIG_WEIGHT_FORMAT_INDEX) {
            OP_CHECK_IF(tuningConfig[TUNING_CONFIG_WEIGHT_FORMAT_INDEX] != 0 &&
                            tuningConfig[TUNING_CONFIG_WEIGHT_FORMAT_INDEX] != TUNING_CONFIG_SPECIAL_WEIGHT_FORMAT,
                        OP_LOGE(context_->GetNodeName(), "tuningConfig[1] must be 0 or 1."), return false);
            specialWeightFormat_ =
                tuningConfig[TUNING_CONFIG_WEIGHT_FORMAT_INDEX] == TUNING_CONFIG_SPECIAL_WEIGHT_FORMAT;
        }
    }
    inputParams_.isSingleX = context_->GetDynamicInputDesc(X_INDEX, 1) == nullptr;
    inputParams_.isSingleW = context_->GetDynamicInputDesc(WEIGHT_INDEX, 1) == nullptr;
    inputParams_.isSingleY = inputParams_.splitItem == X_SEPARATED || inputParams_.splitItem == NO_SEPARATED;

    OP_CHECK_IF(
        inputParams_.groupType != SPLIT_M || inputParams_.actType != 0 || inputParams_.transA || inputParams_.transB,
        OP_LOGE(context_->GetNodeName(), "S8S4 BasicApi expects groupType=0, actType=0 and no transpose."),
        return false);
    OP_CHECK_IF(specialWeightFormat_ && inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ,
                OP_LOGE(context_->GetNodeName(), "tuningConfig[1]=1 expects [E,N,K] weight converted to FRACTAL_NZ."),
                return false);
    OP_CHECK_IF(
        specialWeightFormat_ && !hasOffset_,
        OP_LOGE(context_->GetNodeName(), "tuningConfig[1]=1 is supported only by non-null offset per-channel mode."),
        return false);
    return true;
}

bool GroupedS8S4BasicApiTiling::CheckTensorLists() const
{
    OP_CHECK_IF(context_->GetDynamicInputDesc(X_INDEX, 1) != nullptr ||
                    context_->GetDynamicInputDesc(WEIGHT_INDEX, 1) != nullptr ||
                    context_->GetDynamicInputDesc(BIAS_INDEX, 1) != nullptr ||
                    context_->GetDynamicInputDesc(SCALE_INDEX, 1) != nullptr ||
                    context_->GetDynamicInputDesc(OFFSET_INDEX, 1) != nullptr,
                OP_LOGE(context_->GetNodeName(),
                        "S8S4 BasicApi only supports length-1 x/weight/bias/scale/offset TensorLists."),
                return false);
    const auto outputInfo = context_->GetIrOutputInstanceInfo(Y_INDEX);
    OP_CHECK_IF(outputInfo == nullptr || outputInfo->GetInstanceNum() != 1,
                OP_LOGE(context_->GetNodeName(), "S8S4 BasicApi only supports a length-1 out TensorList."),
                return false);
    return true;
}

bool GroupedS8S4BasicApiTiling::CheckCommonShapes(const gert::Shape &xShape, const gert::Shape &wShape) const
{
    OP_CHECK_IF(xShape.GetDimNum() != SPLIT_K_W_DIMS || wShape.GetDimNum() != SPLIT_M_W_DIMS,
                OP_LOGE(context_->GetNodeName(), "S8S4 BasicApi expects 2D x and 3D weight."), return false);
    OP_CHECK_IF(inputParams_.mSize == 0 || inputParams_.groupNum == 0,
                OP_LOGE(context_->GetNodeName(), "M and groupNum must be positive."), return false);
    OP_CHECK_IF(static_cast<uint64_t>(wShape.GetDim(0)) != inputParams_.groupNum,
                OP_LOGE(context_->GetNodeName(), "weight axis 0 must equal groupNum."), return false);
    OP_CHECK_IF(inputParams_.kSize == 0, OP_LOGE(context_->GetNodeName(), "K must be positive."), return false);
    if (dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP) {
        OP_CHECK_IF(inputParams_.kSize > MAX_K_PER_GROUP || inputParams_.kSize % QUANT_GROUP_SIZE != 0,
                    OP_LOGE(context_->GetNodeName(),
                            "Symmetric per-group K must not exceed 18432 and must be divisible by 256."),
                    return false);
    }
    OP_CHECK_IF(inputParams_.nSize == 0, OP_LOGE(context_->GetNodeName(), "N must be positive."), return false);
    return true;
}

bool GroupedS8S4BasicApiTiling::SetLogicalMKN(const gert::Shape &xShape, const gert::Shape &wShape)
{
    auto outStorageShape = context_->GetOutputShape(Y_INDEX);
    OP_CHECK_IF(
        outStorageShape == nullptr || xShape.GetDimNum() != SPLIT_K_W_DIMS || wShape.GetDimNum() != SPLIT_M_W_DIMS,
        OP_LOGE(context_->GetNodeName(), "Invalid x/weight/out shape."), return false);
    const gert::Shape &outShape = outStorageShape->GetOriginShape();
    OP_CHECK_IF(outShape.GetDimNum() != SPLIT_K_W_DIMS,
                OP_LOGE(context_->GetNodeName(), "S8S4 BasicApi expects out [M,N]."), return false);

    inputParams_.mSize = static_cast<uint64_t>(xShape.GetDim(0));
    inputParams_.kSize = static_cast<uint64_t>(xShape.GetDim(1));
    inputParams_.nSize = static_cast<uint64_t>(outShape.GetDim(1));
    OP_CHECK_IF(static_cast<uint64_t>(outShape.GetDim(0)) != inputParams_.mSize,
                OP_LOGE(context_->GetNodeName(), "out axis 0 must equal M."), return false);

    // OriginShape remains the logical INT4 shape even when storage dtype is packed INT32.
    const uint64_t logicalK = static_cast<uint64_t>(wShape.GetDim(specialWeightFormat_ ? 2 : 1));
    const uint64_t logicalN = static_cast<uint64_t>(wShape.GetDim(specialWeightFormat_ ? 1 : 2));
    OP_CHECK_IF(logicalK != inputParams_.kSize || logicalN != inputParams_.nSize,
                OP_LOGE(context_->GetNodeName(),
                        "weight logical shape mismatches x/out; expected K=%lu,N=%lu, actual K=%lu,N=%lu.",
                        inputParams_.kSize, inputParams_.nSize, logicalK, logicalN),
                return false);
    return true;
}

bool GroupedS8S4BasicApiTiling::CheckBiasShape() const
{
    auto biasShape = context_->GetDynamicInputShape(BIAS_INDEX, 0);
    OP_CHECK_IF(!inputParams_.hasBias || biasShape == nullptr, OP_LOGE(context_->GetNodeName(), "bias is required."),
                return false);
    const gert::Shape &shape = biasShape->GetOriginShape();
    OP_CHECK_IF(shape.GetDimNum() != SPLIT_K_W_DIMS ||
                    static_cast<uint64_t>(shape.GetDim(0)) != inputParams_.groupNum ||
                    static_cast<uint64_t>(shape.GetDim(1)) != inputParams_.nSize,
                OP_LOGE(context_->GetNodeName(), "bias shape must be [groupNum, N]."), return false);
    return true;
}

bool GroupedS8S4BasicApiTiling::CheckPerTokenScaleShape() const
{
    auto perTokenScaleShape = context_->GetOptionalInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(perTokenScaleShape == nullptr, OP_LOGE(context_->GetNodeName(), "perTokenScale shape is nullptr."),
                return false);
    const gert::Shape &shape = perTokenScaleShape->GetOriginShape();
    OP_CHECK_IF(shape.GetDimNum() != 1 || static_cast<uint64_t>(shape.GetDim(0)) != inputParams_.mSize,
                OP_LOGE(context_->GetNodeName(), "perTokenScale shape must be [M]."), return false);
    return true;
}

bool GroupedS8S4BasicApiTiling::CheckAntiquantInputsEmpty() const
{
    auto antiquantScaleShape = context_->GetDynamicInputShape(ANTIQUANT_SCALE_INDEX, 0);
    auto antiquantOffsetShape = context_->GetDynamicInputShape(ANTIQUANT_OFFSET_INDEX, 0);
    const bool hasAntiquantScale =
        antiquantScaleShape != nullptr && antiquantScaleShape->GetStorageShape().GetShapeSize() != 0;
    const bool hasAntiquantOffset =
        antiquantOffsetShape != nullptr && antiquantOffsetShape->GetStorageShape().GetShapeSize() != 0;
    OP_CHECK_IF(
        hasAntiquantScale || hasAntiquantOffset,
        OP_LOGE(context_->GetNodeName(), "S8S4 BasicApi requires antiquantScale and antiquantOffset to be empty."),
        return false);
    return true;
}

bool GroupedS8S4BasicApiTiling::CheckWeightStorageShape(const gert::StorageShape &weightShape)
{
    const gert::Shape &storageShape = weightShape.GetStorageShape();
    if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        // torch_npu constructs the normal A5 KN-NZ path as follows:
        //   INT8 NZ:          [E, ceil(N/32), ceil(K/16), 16, 32]
        //   packed INT32:     [E, ceil(N/32), ceil(K/16), 16, 4]
        // The special ENK-NZ path uses the same C0 and swaps the logical K/N
        // block axes: [E, ceil(K/32), ceil(N/16), 16, 32/4].
        // ACLNN reinterprets the INT32 carrier as INT4 and restores the last
        // storage dimension from 4 to 32 before Host tiling. Keep support for
        // a direct INT32 Host invocation as well, but reject all other physical
        // layouts instead of accepting a same-sized, differently blocked NZ.
        const uint64_t expectedC0 = weightPackedInt32_ ? 4UL : 32UL;
        const uint64_t expectedOuter0 = specialWeightFormat_ ? CeilDiv(inputParams_.kSize, NZ_SPECIAL_K_ALIGN) :
                                                               CeilDiv(inputParams_.nSize, NZ_BASE_N_ALIGN);
        const uint64_t expectedOuter1 = specialWeightFormat_ ? CeilDiv(inputParams_.nSize, NZ_SPECIAL_N_ALIGN) :
                                                               CeilDiv(inputParams_.kSize, NZ_BASE_K_ALIGN);
        const bool shapeMatches = storageShape.GetDimNum() == 5 &&
                                  static_cast<uint64_t>(storageShape.GetDim(0)) == inputParams_.groupNum &&
                                  static_cast<uint64_t>(storageShape.GetDim(1)) == expectedOuter0 &&
                                  static_cast<uint64_t>(storageShape.GetDim(2)) == expectedOuter1 &&
                                  static_cast<uint64_t>(storageShape.GetDim(3)) == NZ_BASE_K_ALIGN &&
                                  static_cast<uint64_t>(storageShape.GetDim(4)) == expectedC0;
        OP_CHECK_IF(!shapeMatches,
                    OP_LOGE(context_->GetNodeName(),
                            "A5 INT4 %s weight storage shape must be "
                            "[E,%s,%s,16,%lu].",
                            specialWeightFormat_ ? "special ENK-NZ" : "normal KN-NZ",
                            specialWeightFormat_ ? "ceil(K/32)" : "ceil(N/32)",
                            specialWeightFormat_ ? "ceil(N/16)" : "ceil(K/16)", expectedC0),
                    return false);
        return true;
    }

    uint64_t scalarSlotNum = inputParams_.groupNum * inputParams_.kSize * inputParams_.nSize;
    if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        const uint64_t alignedK = specialWeightFormat_ ? CeilAlign(inputParams_.kSize, NZ_SPECIAL_K_ALIGN) :
                                                         CeilAlign(inputParams_.kSize, NZ_BASE_K_ALIGN);
        const uint64_t alignedN = specialWeightFormat_ ? CeilAlign(inputParams_.nSize, NZ_SPECIAL_N_ALIGN) :
                                                         CeilAlign(inputParams_.nSize, NZ_BASE_N_ALIGN);
        scalarSlotNum = inputParams_.groupNum * alignedK * alignedN;
    }
    const uint64_t actualStorageElements = static_cast<uint64_t>(storageShape.GetShapeSize());
    // aclnnGroupedMatmulV5 normalizes a packed INT32 carrier to logical INT4 before Host tiling.
    // For ND weights the 8:1 physical storage shape is intentionally retained, so dtype alone is
    // insufficient to recover the carrier representation at this point.
    if (!weightPackedInt32_ && inputParams_.bDtype == ge::DT_INT4 && scalarSlotNum % PACKED_INT32_RATIO == 0 &&
        actualStorageElements == scalarSlotNum / PACKED_INT32_RATIO) {
        weightPackedInt32_ = true;
    }
    OP_CHECK_IF(weightPackedInt32_ && scalarSlotNum % PACKED_INT32_RATIO != 0,
                OP_LOGE(context_->GetNodeName(), "Packed INT32 weight requires the physical INT4 slot count "
                                                 "to be divisible by 8."),
                return false);
    const uint64_t expectedStorageElements = weightPackedInt32_ ? scalarSlotNum / PACKED_INT32_RATIO : scalarSlotNum;
    OP_CHECK_IF(actualStorageElements != expectedStorageElements,
                OP_LOGE(context_->GetNodeName(),
                        "Weight storage shape mismatch: expected %lu physical elements for logical INT4, got %lu.",
                        expectedStorageElements, actualStorageElements),
                return false);
    return true;
}

bool GroupedS8S4BasicApiTiling::CheckScaleAndOffsetShapes()
{
    auto scaleShape = context_->GetDynamicInputShape(SCALE_INDEX, 0);
    OP_CHECK_IF(scaleShape == nullptr, OP_LOGE(context_->GetNodeName(), "scale shape is nullptr."), return false);
    const gert::Shape &shape = scaleShape->GetOriginShape();
    const uint64_t expectedScaleGroups =
        dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP ? inputParams_.kSize / QUANT_GROUP_SIZE : 1UL;
    OP_CHECK_IF(shape.GetDimNum() != 3 || static_cast<uint64_t>(shape.GetDim(0)) != inputParams_.groupNum ||
                    static_cast<uint64_t>(shape.GetDim(1)) != expectedScaleGroups ||
                    static_cast<uint64_t>(shape.GetDim(2)) != inputParams_.nSize,
                OP_LOGE(context_->GetNodeName(), "scale shape must be [groupNum, quantGroupNum, N]."), return false);

    if (!hasOffset_) {
        return true;
    }
    auto offsetShape = context_->GetDynamicInputShape(OFFSET_INDEX, 0);
    OP_CHECK_IF(offsetShape == nullptr, OP_LOGE(context_->GetNodeName(), "offset shape is nullptr."), return false);
    const gert::Shape &offset = offsetShape->GetOriginShape();
    OP_CHECK_IF(offset.GetDimNum() != 3 || static_cast<uint64_t>(offset.GetDim(0)) != inputParams_.groupNum ||
                    offset.GetDim(1) != 1 || static_cast<uint64_t>(offset.GetDim(2)) != inputParams_.nSize,
                OP_LOGE(context_->GetNodeName(), "offset shape must be [groupNum, 1, N]."), return false);
    return true;
}

bool GroupedS8S4BasicApiTiling::AnalyzeInputs()
{
    OP_CHECK_IF(!CheckTensorLists(), OP_LOGE(context_->GetNodeName(), "CheckTensorLists failed."), return false);
    auto xStorageShape = context_->GetDynamicInputShape(X_INDEX, 0);
    auto weightStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_IF(xStorageShape == nullptr || weightStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "x or weight shape is nullptr."), return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();
    const gert::Shape &weightShape = weightStorageShape->GetOriginShape();

    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(context_->GetNodeName(), "SetGroupNum failed."), return false);
    OP_CHECK_IF(!SetLogicalMKN(xShape, weightShape), OP_LOGE(context_->GetNodeName(), "SetLogicalMKN failed."),
                return false);
    OP_CHECK_IF(expectedTokenNum_ > inputParams_.mSize,
                OP_LOGE(context_->GetNodeName(), "tuningConfig[0] must not exceed M."), return false);
    OP_CHECK_IF(!CheckCommonShapes(xShape, weightShape) || !CheckWeightStorageShape(*weightStorageShape) ||
                    !CheckBiasShape() || !CheckPerTokenScaleShape() || !CheckAntiquantInputsEmpty() ||
                    !CheckScaleAndOffsetShapes(),
                OP_LOGE(context_->GetNodeName(), "S8S4 BasicApi shape validation failed."), return false);

    inputParams_.aQuantMode = QuantMode::PERTOKEN_MODE;
    inputParams_.bQuantMode =
        dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP ? QuantMode::PERGROUP_MODE : QuantMode::PERCHANNEL_MODE;
    inputParams_.kernelType = 0UL;
    return true;
}

ge::graphStatus GroupedS8S4BasicApiTiling::DoOpTiling()
{
    tilingData_.gmmQuantParams.groupNum = inputParams_.groupNum;
    tilingData_.gmmQuantParams.activeType = inputParams_.actType;
    tilingData_.gmmQuantParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmQuantParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmQuantParams.singleX = static_cast<uint8_t>(inputParams_.isSingleX);
    tilingData_.gmmQuantParams.singleW = static_cast<uint8_t>(inputParams_.isSingleW);
    tilingData_.gmmQuantParams.singleY = static_cast<uint8_t>(inputParams_.isSingleY);
    tilingData_.gmmQuantParams.groupType = inputParams_.groupType;
    tilingData_.gmmQuantParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmQuantParams.hasBias = static_cast<uint8_t>(inputParams_.hasBias);
    tilingData_.s8s4Params.quantGroupSize =
        dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP ? QUANT_GROUP_SIZE : static_cast<uint32_t>(inputParams_.kSize);
    tilingData_.s8s4Params.quantGroupNum = dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP ?
                                               static_cast<uint32_t>(inputParams_.kSize / QUANT_GROUP_SIZE) :
                                               1U;
    tilingData_.s8s4Params.expectedTokenNum = expectedTokenNum_;
    tilingData_.s8s4Params.coreNum = static_cast<uint32_t>(aicoreParams_.aicNum);
    tilingData_.s8s4Params.dequantMode = static_cast<uint8_t>(dequantMode_);
    tilingData_.s8s4Params.hasOffset = static_cast<uint8_t>(hasOffset_);
    tilingData_.s8s4Params.specialWeightFormat = static_cast<uint8_t>(specialWeightFormat_);
    tilingData_.s8s4Params.weightPackedInt32 = static_cast<uint8_t>(weightPackedInt32_);
    // Both modes expand the complete W4 weight to W8 in user workspace before
    // entering the ops-tensor S8S4 kernel.
    tilingData_.s8s4Params.enableWeightPreprocess = 1U;
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedS8S4BasicApiTiling::GetTilingKey() const
{
    const uint64_t weightFormat = inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ ? WQGMM_FRACTAL_NZ : WQGMM_ND;
    const uint64_t cQuantType = dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP ? WQGMM_PER_GROUP : WQGMM_PER_CHANNEL;
    return GET_TPL_TILING_KEY(weightFormat, WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST, cQuantType, WQGMM_NONE,
                              WQGMM_NO_TRANS, WQGMM_NO_TRANS, WQGMM_S8S4_FIXED_TEMPLATE, WQGMM_NOT_SINGLE_MULTI_SINGLE,
                              WQGMM_VDEFAULT, WQGMM_MULTI_SCALE_DEQUANT);
}

ge::graphStatus GroupedS8S4BasicApiTiling::ValidateFixedTileResources(uint64_t kL1) const
{
    const uint64_t int8C0 = GetShapeWithDataType(GmmConstant::L1_ALIGN_SIZE, ge::DT_INT8);
    const uint64_t baseMAligned = CeilAlign(basicTiling_.baseM, GmmConstant::CUBE_BLOCK);
    const uint64_t baseNCubeAligned = CeilAlign(basicTiling_.baseN, GmmConstant::CUBE_BLOCK);
    const uint64_t baseNInt8Aligned = CeilAlign(basicTiling_.baseN, int8C0);
    const uint64_t kL1CubeAligned = CeilAlign(kL1, GmmConstant::CUBE_BLOCK);
    const uint64_t kL1Int8Aligned = CeilAlign(kL1, int8C0);
    const uint64_t scaleL1Bytes = CeilAlign(basicTiling_.baseN * sizeof(uint64_t), GmmConstant::L1_ALIGN_SIZE);
    const uint64_t l1StageBytes = baseMAligned * kL1Int8Aligned + kL1CubeAligned * baseNInt8Aligned + scaleL1Bytes;
    const uint64_t l1HalfBytes = aicoreParams_.l1Size / GmmConstant::DB_SIZE;

    const uint64_t baseKInt8Aligned = CeilAlign(basicTiling_.baseK, int8C0);
    const uint64_t l0aBytes = GmmConstant::DB_SIZE * baseMAligned * baseKInt8Aligned;
    const uint64_t l0bBytes = GmmConstant::DB_SIZE * baseKInt8Aligned * baseNCubeAligned;
    const uint64_t l0cBytes = baseMAligned * baseNCubeAligned * GmmConstant::DATA_SIZE_L0C;

    const uint64_t vectorBaseM = CeilDiv(basicTiling_.baseM, GmmConstant::CORE_RATIO);
    const uint64_t dataTileBytes =
        CeilAlign(vectorBaseM * CeilAlign(basicTiling_.baseN * sizeof(uint16_t), GmmConstant::UB_ALIGN_SIZE),
                  GmmConstant::UB_ALIGN_SIZE);
    const uint64_t rowSumBytes = CeilAlign(vectorBaseM * sizeof(float), GmmConstant::UB_ALIGN_SIZE);
    const uint64_t offsetBytes = CeilAlign(basicTiling_.baseN * sizeof(float), GmmConstant::UB_ALIGN_SIZE);
    const uint64_t ubBytes = GmmConstant::NUM_HALF * (dataTileBytes + rowSumBytes) + offsetBytes;

    OP_CHECK_IF(l1StageBytes > l1HalfBytes || l0aBytes > aicoreParams_.l0aSize || l0bBytes > aicoreParams_.l0bSize ||
                    l0cBytes > aicoreParams_.l0cSize || ubBytes > aicoreParams_.ubSize,
                OP_LOGE(context_->GetNodeName(),
                        "S8S4 fixed tile resource overflow: baseM=%lu, baseN=%lu, baseK=%lu, kL1=%lu, "
                        "L1 stage=%lu/%lu, L0A=%lu/%lu, L0B=%lu/%lu, L0C=%lu/%lu, UB=%lu/%lu.",
                        basicTiling_.baseM, basicTiling_.baseN, basicTiling_.baseK, kL1, l1StageBytes, l1HalfBytes,
                        l0aBytes, aicoreParams_.l0aSize, l0bBytes, aicoreParams_.l0bSize, l0cBytes,
                        aicoreParams_.l0cSize, ubBytes, aicoreParams_.ubSize),
                return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(),
            "S8S4 fixed tile resources: baseM=%lu, baseN=%lu, baseK=%lu, kL1=%lu, "
            "L1 stage=%lu/%lu, L0A=%lu/%lu, L0B=%lu/%lu, L0C=%lu/%lu, UB=%lu/%lu.",
            basicTiling_.baseM, basicTiling_.baseN, basicTiling_.baseK, kL1, l1StageBytes, l1HalfBytes, l0aBytes,
            aicoreParams_.l0aSize, l0bBytes, aicoreParams_.l0bSize, l0cBytes, aicoreParams_.l0cSize, ubBytes,
            aicoreParams_.ubSize);
    return ge::GRAPH_SUCCESS;
}

void GroupedS8S4BasicApiTiling::SetBasicBlock()
{
    const uint64_t preferredBaseM = dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP ? S8S4_BASE_M : S8S4_ASYM_BASE_M;
    basicTiling_.baseM = CeilAlign(std::min(inputParams_.mSize, preferredBaseM), static_cast<uint64_t>(CUBE_BLOCK));
    basicTiling_.baseN =
        CeilAlign(std::min(inputParams_.nSize, static_cast<uint64_t>(S8S4_BASE_N)), static_cast<uint64_t>(CUBE_BLOCK));
    basicTiling_.baseK = S8S4_BASE_K;
}

ge::graphStatus GroupedS8S4BasicApiTiling::DoLibApiTiling()
{
    SetBasicBlock();
    uint64_t kL1 = dequantMode_ == DequantMode::SYMMETRIC_PER_GROUP ? S8S4_PER_GROUP_K_L1 : S8S4_PER_CHANNEL_K_L1;
    const uint64_t alignedK = CeilAlign(inputParams_.kSize, static_cast<uint64_t>(QUANT_GROUP_SIZE));
    kL1 = std::min(kL1, alignedK);
    OP_CHECK_IF(ValidateFixedTileResources(kL1) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "S8S4 fixed tile resource validation failed."),
                return ge::GRAPH_FAILED);

    tilingData_.mmTilingData.m = static_cast<uint32_t>(inputParams_.mSize);
    tilingData_.mmTilingData.n = static_cast<uint32_t>(inputParams_.nSize);
    tilingData_.mmTilingData.k = static_cast<uint32_t>(inputParams_.kSize);
    tilingData_.mmTilingData.baseM = static_cast<uint32_t>(basicTiling_.baseM);
    tilingData_.mmTilingData.baseN = static_cast<uint32_t>(basicTiling_.baseN);
    tilingData_.mmTilingData.baseK = S8S4_BASE_K;
    tilingData_.mmTilingData.kAL1 = static_cast<uint32_t>(kL1);
    tilingData_.mmTilingData.kBL1 = static_cast<uint32_t>(kL1);
    tilingData_.mmTilingData.scaleKAL1 = static_cast<uint32_t>(kL1);
    tilingData_.mmTilingData.scaleKBL1 = static_cast<uint32_t>(kL1);
    // V5 bias is the offline correction for an unsigned W4 transform. This path
    // sign-extends W4 directly, so it must not be injected into Cube.
    tilingData_.mmTilingData.isBias = 0U;
    const uint64_t l0cTileBytes = basicTiling_.baseM * basicTiling_.baseN * sizeof(int32_t);
    tilingData_.mmTilingData.dbL0C = l0cTileBytes * 2UL <= aicoreParams_.l0cSize ? 2U : 1U;

    {
        constexpr uint64_t WORKSPACE_ALIGN = 512UL;
        constexpr uint64_t NZ_K_ALIGN = 16UL;
        constexpr uint64_t NZ_N_ALIGN = 32UL;
        constexpr uint64_t UINT64_MAX_VALUE = std::numeric_limits<uint64_t>::max();
        auto checkedMul = [](uint64_t lhs, uint64_t rhs, uint64_t &result) {
            if (rhs != 0UL && lhs > UINT64_MAX_VALUE / rhs) {
                return false;
            }
            result = lhs * rhs;
            return true;
        };
        auto checkedAdd = [](uint64_t lhs, uint64_t rhs, uint64_t &result) {
            if (lhs > UINT64_MAX_VALUE - rhs) {
                return false;
            }
            result = lhs + rhs;
            return true;
        };
        auto checkedAlign = [&](uint64_t value, uint64_t &result) {
            if (value > UINT64_MAX_VALUE - (WORKSPACE_ALIGN - 1UL)) {
                return false;
            }
            result = CeilAlign(value, WORKSPACE_ALIGN);
            return true;
        };
        uint64_t expandedK = inputParams_.kSize;
        uint64_t expandedN = inputParams_.nSize;
        if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) {
            OP_CHECK_IF(
                expandedK > UINT64_MAX_VALUE - (NZ_K_ALIGN - 1UL) || expandedN > UINT64_MAX_VALUE - (NZ_N_ALIGN - 1UL),
                OP_LOGE(context_->GetNodeName(), "S8S4 expanded NZ alignment overflow."), return ge::GRAPH_FAILED);
            expandedK = CeilAlign(expandedK, NZ_K_ALIGN);
            expandedN = CeilAlign(expandedN, NZ_N_ALIGN);
        }
        uint64_t expertStride = 0UL;
        uint64_t expandedBytes = 0UL;
        uint64_t expandedSize = 0UL;
        uint64_t tileElements = 0UL;
        uint64_t tileBytes = 0UL;
        uint64_t tileStride = 0UL;
        uint64_t tileSize = 0UL;
        uint64_t rowSumBytes = 0UL;
        uint64_t rowSumSize = 0UL;
        uint64_t tileOffset = 0UL;
        uint64_t rowSumOffset = 0UL;
        uint64_t totalSize = 0UL;
        OP_CHECK_IF(!checkedMul(expandedK, expandedN, expertStride) ||
                        !checkedMul(expertStride, inputParams_.groupNum, expandedBytes) ||
                        !checkedAlign(expandedBytes, expandedSize),
                    OP_LOGE(context_->GetNodeName(), "S8S4 expanded weight workspace overflow."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!checkedMul(basicTiling_.baseM, basicTiling_.baseN, tileElements) ||
                        !checkedMul(tileElements, sizeof(uint16_t), tileBytes) ||
                        !checkedAlign(tileBytes, tileStride) || !checkedMul(tileStride, aicoreParams_.aicNum, tileSize),
                    OP_LOGE(context_->GetNodeName(), "S8S4 per-core tile workspace overflow."),
                    return ge::GRAPH_FAILED);
        const uint64_t rowSumElements = dequantMode_ == DequantMode::ASYMMETRIC_PER_CHANNEL ? inputParams_.mSize : 0UL;
        OP_CHECK_IF(!checkedMul(rowSumElements, sizeof(float), rowSumBytes) || !checkedAlign(rowSumBytes, rowSumSize) ||
                        !checkedAdd(expandedSize, tileSize, rowSumOffset) ||
                        !checkedAdd(rowSumOffset, rowSumSize, totalSize),
                    OP_LOGE(context_->GetNodeName(), "S8S4 row-sum workspace overflow."), return ge::GRAPH_FAILED);
        tileOffset = expandedSize;
        userWorkspaceSize_ = totalSize;
        tilingData_.s8s4Params.expandedWeightOffsetBytes = 0UL;
        tilingData_.s8s4Params.expandedWeightSizeBytes = expandedSize;
        tilingData_.s8s4Params.expandedWeightStrideBytes = expertStride;
        tilingData_.s8s4Params.tileWorkspaceOffsetBytes = tileOffset;
        tilingData_.s8s4Params.tileWorkspaceSizeBytes = tileSize;
        tilingData_.s8s4Params.tileWorkspaceStrideBytes = tileStride;
        tilingData_.s8s4Params.rowSumOffsetBytes = rowSumOffset;
        tilingData_.s8s4Params.rowSumSizeBytes = rowSumSize;
        tilingData_.s8s4Params.userWorkspaceSizeBytes = totalSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedS8S4BasicApiTiling::PostTiling()
{
    // Weight preprocessing synchronizes all vector cores, so all participating cores must be resident.
    context_->SetScheduleMode(1);
    return SaveTilingDataToContext(tilingData_);
}

ge::graphStatus GroupedS8S4BasicApiTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "workspaces is nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(userWorkspaceSize_ > std::numeric_limits<uint64_t>::max() - SYS_WORKSPACE_SIZES,
                OP_LOGE(context_->GetNodeName(), "S8S4 total workspace size overflow."), return ge::GRAPH_FAILED);
    // GetUserWorkspace() skips the first 16 MiB reserved by the runtime. The
    // complete expanded W8 weight is placed after that system area.
    uint64_t workspaceSize = SYS_WORKSPACE_SIZES + userWorkspaceSize_;
    OP_CHECK_IF(workspaceSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
                OP_LOGE(context_->GetNodeName(), "S8S4 workspace does not fit size_t."), return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(workspaceSize);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmul, GroupedS8S4BasicApiTiling, 2);

} // namespace optiling
