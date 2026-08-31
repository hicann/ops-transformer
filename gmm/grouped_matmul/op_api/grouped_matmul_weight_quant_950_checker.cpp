/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_weight_quant_950_checker.h"
#include "log/log.h"

using namespace gmm;

namespace {
static const std::unordered_set<DataType> FP8_SUPPORT_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8};
static constexpr int64_t X_Y_SEPARATED = 0L; // x,y no split
static constexpr int64_t Y_SEPARATED = 1L;   // x split
static constexpr int64_t X_SEPARATED = 2L;   // y split
static constexpr int64_t NO_SEPARATED = 3L;  // x,y split
static constexpr int64_t MAX_GROUP_LIST_SIZE_ARRAY = 128L;
static constexpr int64_t MULTI_WEIGHT_DIM = 2UL;
static constexpr int64_t MULTI_WEIGHT_NZ_DIM = 4UL;
static constexpr size_t S8S4_MATRIX_DIM_NUM = 2UL;
static constexpr size_t S8S4_WEIGHT_DIM_NUM = 3UL;
static constexpr size_t S8S4_OFFSET_DIM_NUM = 3UL;
static constexpr size_t S8S4_WEIGHT_K_DIM = 1UL;
static constexpr size_t S8S4_WEIGHT_N_DIM = 2UL;
static constexpr size_t TUNING_CONFIG_WEIGHT_FORMAT_INDEX = 1UL;
static constexpr int64_t TUNING_CONFIG_SPECIAL_WEIGHT_FORMAT = 1L;
static constexpr int64_t GROUP_LIST_COUNT = 1L;
static constexpr int64_t S8S4_PER_GROUP_MAX_K = 18432L;
static constexpr int64_t S8S4_QUANT_GROUP_SIZE = 256L;
} // namespace

std::string AclnnGroupedMatmulWeightQuantDAV3510Checker::GetDataFlowString() const
{
    return "when the dtype of x is " + std::string(op::ToString(xDtype_).GetString()) + " and the dtype of weight is " +
           std::string(op::ToString(weightDtype_).GetString());
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::GetDimFromEnd(const aclTensor *tensor, size_t posFromEnd,
                                                                       size_t &out, const std::string &tensorName) const
{
    auto shape = tensor->GetViewShape();
    if (unlikely(posFromEnd == 0 || posFromEnd > shape.GetDimNum())) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetAclnnName(), tensorName, std::to_string(posFromEnd),
                                              "posFromEnd must be in [1, dimNum]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t v = shape.GetDim(static_cast<int64_t>(shape.GetDimNum() - posFromEnd));
    if (unlikely(v < 0)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetAclnnName(), tensorName, std::to_string(v),
                                              "dim value must be non-negative");
        return ACLNN_ERR_PARAM_INVALID;
    }
    out = static_cast<size_t>(v);
    return ACLNN_SUCCESS;
}

// 根据 apiVersion 返回对应的 aclnn 函数名字符串
const char *AclnnGroupedMatmulWeightQuantDAV3510Checker::GetAclnnName() const
{
    switch (gmmParams_.apiVersion) {
        case gmm::GMMApiVersion::V1:
            return "aclnnGroupedMatmulGetWorkspaceSize";
        case gmm::GMMApiVersion::V2:
            return "aclnnGroupedMatmulV2GetWorkspaceSize";
        case gmm::GMMApiVersion::V3:
            return "aclnnGroupedMatmulV3GetWorkspaceSize";
        case gmm::GMMApiVersion::V4:
            return "aclnnGroupedMatmulV4GetWorkspaceSize";
        case gmm::GMMApiVersion::V5:
            return "aclnnGroupedMatmulV5GetWorkspaceSize";
        case gmm::GMMApiVersion::WeightNz:
            return "aclnnGroupedMatmulWeightNzGetWorkspaceSize";
        default:
            return "aclnnGroupedMatmulGetWorkspaceSize";
    }
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsA16MxFp4NZ() const
{
    return (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) && weightDtype_ == ge::DT_FLOAT4_E2M1;
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsMxA8W4NZ() const
{
    return xDtype_ == ge::DT_FLOAT8_E4M3FN &&
           (weightDtype_ == ge::DT_FLOAT4_E2M1 || weightDtype_ == ge::DT_FLOAT4_E1M2);
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsA16W8ND() const
{
    return (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) && weightDtype_ == ge::DT_INT8;
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsA16F8ND() const
{
    return (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) &&
           FP8_SUPPORT_SET.find(weightDtype_) != FP8_SUPPORT_SET.end();
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsS8S4NZ() const
{
    return xDtype_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT4;
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsS8S4PseudoQuant() const
{
    if (!IsS8S4NZ()) {
        return false;
    }
    if (gmmParams_.apiVersion == GMMApiVersion::V5) {
        return true;
    }
    return gmmParams_.apiVersion == GMMApiVersion::WeightNz && gmmParams_.scaleOptional != nullptr &&
           gmmParams_.scaleOptional->Size() > 0 && (*gmmParams_.scaleOptional)[0] != nullptr &&
           (*gmmParams_.scaleOptional)[0]->GetDataType() == ge::DT_UINT64;
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsS8S4SpecialWeightFormat() const
{
    return IsS8S4PseudoQuant() && gmmParams_.tuningConfigOptional != nullptr &&
           gmmParams_.tuningConfigOptional->Size() > TUNING_CONFIG_WEIGHT_FORMAT_INDEX &&
           (*gmmParams_.tuningConfigOptional)[TUNING_CONFIG_WEIGHT_FORMAT_INDEX] == TUNING_CONFIG_SPECIAL_WEIGHT_FORMAT;
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsS8S4AsymmetricQuant() const
{
    if (!IsS8S4PseudoQuant() || gmmParams_.offsetOptional == nullptr || gmmParams_.offsetOptional->Size() != 1 ||
        (*gmmParams_.offsetOptional)[0] == nullptr) {
        return false;
    }
    const op::Shape &offsetShape = (*gmmParams_.offsetOptional)[0]->GetViewShape();
    return !(offsetShape.GetDimNum() == 1 && offsetShape.GetDim(0) == 0);
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckS8S4GroupListType() const
{
    CHECK_COND(gmmParams_.groupListType == GROUP_LIST_COUNT, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when S8S4, groupListType must be 1(count).", GetAclnnName());
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckS8S4ScaleShape() const
{
    const aclTensor *x = (*gmmParams_.x)[0];
    const aclTensor *weight = (*gmmParams_.weight)[0];
    const aclTensor *scale = (*gmmParams_.scaleOptional)[0];
    const aclTensor *perTokenScale = (*gmmParams_.perTokenScaleOptional)[0];
    const aclTensor *out = (*gmmParams_.y)[0];

    const op::Shape &xShape = x->GetViewShape();
    const op::Shape &weightShape = weight->GetViewShape();
    const op::Shape &outShape = out->GetViewShape();
    CHECK_COND(xShape.GetDimNum() == S8S4_MATRIX_DIM_NUM && weightShape.GetDimNum() == S8S4_WEIGHT_DIM_NUM &&
                   outShape.GetDimNum() == S8S4_MATRIX_DIM_NUM,
               ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when S8S4, x, weight and out shapes must be [M,K], [E,K,N] and [M,N].", GetAclnnName());

    const bool specialWeightFormat = IsS8S4SpecialWeightFormat();
    CHECK_COND(!specialWeightFormat || IsS8S4AsymmetricQuant(), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], the S8S4 special ENK-NZ template requires non-null offset per-channel mode.",
               GetAclnnName());
    const int64_t e = weightShape.GetDim(0);
    const int64_t k = weightShape.GetDim(specialWeightFormat ? S8S4_WEIGHT_N_DIM : S8S4_WEIGHT_K_DIM);
    const int64_t n = weightShape.GetDim(specialWeightFormat ? S8S4_WEIGHT_K_DIM : S8S4_WEIGHT_N_DIM);
    const int64_t m = xShape.GetDim(0);
    CHECK_COND(m > 0 && e > 0 && k > 0 && n > 0 && xShape.GetDim(S8S4_WEIGHT_K_DIM) == k && outShape.GetDim(0) == m &&
                   outShape.GetDim(S8S4_WEIGHT_K_DIM) == n,
               ACLNN_ERR_PARAM_INVALID, "In op [%s], when S8S4, M, E, K and N must be positive and compatible.",
               GetAclnnName());

    const op::Shape &scaleShape = scale->GetViewShape();
    const op::Shape &perTokenScaleShape = perTokenScale->GetViewShape();
    CHECK_COND(perTokenScaleShape.GetDimNum() == 1 && perTokenScaleShape.GetDim(0) == m, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when S8S4, perTokenScale shape must be [M].", GetAclnnName());
    CHECK_COND(scaleShape.GetDimNum() == S8S4_WEIGHT_DIM_NUM && scaleShape.GetDim(0) == e &&
                   scaleShape.GetDim(S8S4_WEIGHT_N_DIM) == n,
               ACLNN_ERR_PARAM_INVALID, "In op [%s], when S8S4, scale shape must be [E,Q,N].", GetAclnnName());

    if (IsS8S4AsymmetricQuant()) {
        const aclTensor *offset = (*gmmParams_.offsetOptional)[0];
        const op::Shape &offsetShape = offset->GetViewShape();
        CHECK_COND(offsetShape.GetDimNum() == S8S4_OFFSET_DIM_NUM && offsetShape.GetDim(0) == e &&
                       offsetShape.GetDim(S8S4_WEIGHT_K_DIM) == 1 && offsetShape.GetDim(S8S4_WEIGHT_N_DIM) == n &&
                       scaleShape.GetDim(S8S4_WEIGHT_K_DIM) == 1,
                   ACLNN_ERR_PARAM_INVALID,
                   "In op [%s], when asymmetric S8S4, offset and scale shapes must be [E,1,N].", GetAclnnName());
    } else {
        CHECK_COND(k <= S8S4_PER_GROUP_MAX_K, ACLNN_ERR_PARAM_INVALID,
                   "In op [%s], when per-group S8S4, K must not exceed 18432.", GetAclnnName());
    }
    return ACLNN_SUCCESS;
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsA16W4() const
{
    return (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) && weightDtype_ == ge::DT_INT4;
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsMultiTensorWeight() const
{
    return (*gmmParams_.weight)[0]->GetViewShape().GetDimNum() == MULTI_WEIGHT_DIM;
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsA16W4Pergroup(const size_t idx) const
{
    if (!IsA16W4()) {
        return false;
    }

    auto antiquantScaleShape = (*gmmParams_.antiquantScaleOptional)[idx]->GetViewShape();
    auto antiquantScaleDimNum = antiquantScaleShape.GetDimNum();
    // pergroup 单单单场景默认维度为3，多多多场景默认维度为2
    size_t pergroupDim =
        gmmParams_.groupType == SPLIT_M ? PERGROUP_SINGLE_ANTIQUANT_SCALE_DIM : PERGROUP_MULTI_ANTIQUANT_SCALE_DIM;
    return antiquantScaleDimNum == pergroupDim;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckTensorNotNull(size_t xIdx, size_t yIdx, size_t wIdx) const
{
    CHECK_RET(CheckTensorNotNullPtr(gmmParams_.x, xIdx, "x") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckTensorNotNullPtr(gmmParams_.weight, wIdx, "weight") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckTensorNotNullPtr(gmmParams_.y, yIdx, "y") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    if (gmmParams_.antiquantScaleOptional != nullptr) {
        CHECK_RET(CheckTensorNotNullPtr(gmmParams_.antiquantScaleOptional, wIdx, "antiquantScale") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_NULLPTR);
    }

    if (gmmParams_.antiquantOffsetOptional != nullptr) {
        CHECK_RET(CheckTensorNotNullPtr(gmmParams_.antiquantOffsetOptional, wIdx, "antiquantOffset") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_NULLPTR);
    }

    if (gmmParams_.biasOptional != nullptr) {
        CHECK_RET(CheckTensorNotNullPtr(gmmParams_.biasOptional, wIdx, "bias") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckTensorNotNullPtr(const aclTensorList *tensorList,
                                                                               size_t idx,
                                                                               const std::string &tensorType) const
{
    const aclTensor *tensor = (*tensorList)[idx];
    if (unlikely(tensor == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            GetAclnnName(), tensorType, "nullptr",
            tensorType + "[" + std::to_string(idx) + "] is null, which is not supported");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckTensorDtype(const aclTensorList *tensorList,
                                                                          const DataType &tensorDtype, size_t idx,
                                                                          const std::string &tensorType) const
{
    const aclTensor *tensor = (*tensorList)[idx];
    if (unlikely(tensor->GetDataType() != tensorDtype)) {
        std::string incorrectDtypes = std::string(op::ToString(tensor->GetDataType()).GetString());
        std::string reason = "The dtype of each tensor in " + tensorType + " tensor list must be consistent. " +
                             tensorType + "[" + std::to_string(idx) + "]'s dtype [" + incorrectDtypes +
                             "] is different from the expected dtype [" + op::ToString(tensorDtype).GetString() + "]";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(GetAclnnName(), tensorType, incorrectDtypes, reason);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckTensorShape(const aclTensorList *tensorList, size_t idx,
                                                                          const std::string &tensorType) const
{
    // 校验bias、antiquantScale和antiquantOffset的dim和shape
    auto tensorShape = (*tensorList)[idx]->GetViewShape();
    auto wShape = (*gmmParams_.weight)[idx]->GetViewShape();

    size_t tensorDimNum = tensorShape.GetDimNum();
    size_t expectedDimNum = gmmParams_.groupType == SPLIT_M ? 2 : 1; // 单单单场景默认维度为2，多多多场景默认维度为1

    if ((IsA16MxFp4NZ() || IsS8S4NZ()) && tensorType.find("antiquant") != std::string::npos) {
        expectedDimNum = 3; // Mx / PerGroup量化，仅支持antiquantSacle/antiquantOffset维度为3
    } else if (IsMxA8W4NZ()) {
        if (tensorType.find("antiquant") != std::string::npos) {
            expectedDimNum = IsMultiTensorWeight() ? MX_MULTI_ANTIQUANT_SCALE_DIM : MX_SINGLE_ANTIQUANT_SCALE_DIM;
        } else if (tensorType.find("token") != std::string::npos) {
            expectedDimNum = 3; // MxA8W4场景，perTokenScale维度为3
        } else if (tensorType.find("bias") != std::string::npos) {
            expectedDimNum = IsMultiTensorWeight() ? MX_MULTI_BIAS_DIM : MX_SINGLE_BIAS_DIM;
        }
    } else if (IsA16W4() && tensorType.find("antiquant") != std::string::npos) {
        size_t perchannelDim = gmmParams_.groupType == SPLIT_M ? 2 : 1; // 单单单场景默认维度为2，多多多场景默认维度为1
        size_t pergroupDim = perchannelDim + 1;
        if (tensorDimNum != perchannelDim && tensorDimNum != pergroupDim) {
            std::string reason = "When x_dtype-weight_dtype is fp16/bf16-int4, Dim must be [" +
                                 std::to_string(perchannelDim) + "] (perchannel) or [" + std::to_string(pergroupDim) +
                                 "] (pergroup), but now is [" + std::to_string(tensorDimNum) + "]";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(GetAclnnName(), tensorType, std::to_string(tensorDimNum), reason);
            return ACLNN_ERR_PARAM_INVALID;
        }
        expectedDimNum = tensorDimNum;
    }

    if (unlikely(tensorDimNum != expectedDimNum)) {
        std::string incorrectDim = std::to_string(tensorDimNum);
        std::string reason = "Dim must be [" + std::to_string(expectedDimNum) + "], but now is [" + incorrectDim + "]";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(GetAclnnName(), tensorType, incorrectDim, reason);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (gmmParams_.groupType == SPLIT_M && !IsMultiTensorWeight()) {
        uint64_t groupNum = wShape.GetDim(0);
        uint64_t batchSize = tensorShape.GetDim(0);
        if (unlikely(batchSize != groupNum)) {
            std::string incorrectValue = std::to_string(batchSize);
            std::string reason = "batch size[" + incorrectValue + "] should be equal with groupList length[" +
                                 std::to_string(groupNum) + "]";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnName(), tensorType, incorrectValue, reason);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // Check tensor’s Ndim must match weight’s Ndim.
    uint64_t weightNDimIdx = IsS8S4SpecialWeightFormat() ? S8S4_WEIGHT_K_DIM : wShape.GetDimNum() - 1;
    int64_t weightNDimValue = wShape.GetDim(weightNDimIdx);
    int64_t tensorNDimValue;
    if (IsMxA8W4NZ() && tensorType.find("antiquant") != std::string::npos) { // viewShape,所以是-2
        tensorNDimValue = tensorShape.GetDim(tensorDimNum - 2);
    } else {
        tensorNDimValue = tensorShape.GetDim(tensorDimNum - 1);
    }
    if (unlikely(tensorNDimValue != weightNDimValue)) {
        std::string incorrectValues = std::to_string(tensorNDimValue) + ", " + std::to_string(weightNDimValue);
        std::string reason = "NDim[" + std::to_string(tensorNDimValue) + "] of " + tensorType +
                             " must be equal to NDim[" + std::to_string(weightNDimValue) + "] of weight";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(GetAclnnName(), tensorType + ", weight", incorrectValues, reason);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckWeightInnerAxisEven(size_t idx) const
{
    if (weightDtype_ == ge::DT_INT4) {
        auto wShape = (*gmmParams_.weight)[idx]->GetViewShape();
        bool isTrans = IsTransposeLastTwoDims((*gmmParams_.weight)[idx]);
        // -2：weight的倒数第二跟轴;-1：表示weight的倒数第一跟轴
        uint64_t weightLastDimIdx = isTrans ? wShape.GetDimNum() - 2 : wShape.GetDimNum() - 1;
        // 2：对内轴的维度是否位奇数
        int64_t evenDivider = 2;
        if (unlikely(wShape.GetDim(weightLastDimIdx) % evenDivider != 0)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetAclnnName(), "weight",
                                                  std::to_string(wShape.GetDim(weightLastDimIdx)),
                                                  "The last axis of each weight tensor must be even");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckAntiQuantParams() const
{
    // 单单单场景antiquantScale为[(g, n)]或[(g, k/gs, n)]，g不为0所以一定不为nullptr
    // 多多多场景antiQuantScale可能为[(0)]，此时会被aclnn_grouped_matmul.cpp中的CheckOptionalTensorListEmpty置为nullptr
    auto w0Shape = (*gmmParams_.weight)[0]->GetViewShape();
    int64_t w0NDim = w0Shape.GetDim(w0Shape.GetDimNum() - 1);
    bool antiquantScaleNullFlag = gmmParams_.groupType == NO_SPLIT && gmmParams_.weight->Size() == 1 && w0NDim == 0;
    if (IsS8S4PseudoQuant()) {
        CHECK_COND(gmmParams_.antiquantScaleOptional == nullptr && gmmParams_.antiquantOffsetOptional == nullptr,
                   ACLNN_ERR_PARAM_INVALID, "In S8S4, antiquantScale and antiquantOffset must be null.");
    } else if (!antiquantScaleNullFlag) {
        CHECK_COND(gmmParams_.antiquantScaleOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "AntiquantScale must not be nullptr in antiquant, but now is nullptr.");
    }

    if (!IsS8S4PseudoQuant() && (IsA16F8ND() || IsA16MxFp4NZ() || IsMxA8W4NZ() || IsS8S4NZ())) {
        CHECK_COND(gmmParams_.antiquantOffsetOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "In weight quant case, antiquantOffsetOptional is not supported when weightDtype is fp8/fp4 or "
                   "xDtype-weightDtype is int8-int4.");
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckQuantParams() const
{
    if (!IsS8S4NZ()) {
        CHECK_COND(gmmParams_.scaleOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "In weight quant case, scale must be null when xDtype-weightDtype is not int8-int4.");
    } else {
        CHECK_COND(gmmParams_.scaleOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
                   "In weight quant case, scale must not be null when xDtype-weightDtype is int8-int4.");
    }

    if (IsS8S4PseudoQuant()) {
        CHECK_COND(gmmParams_.biasOptional != nullptr, ACLNN_ERR_PARAM_INVALID, "In S8S4, bias must not be null.");
    } else {
        CHECK_COND(gmmParams_.offsetOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "In WeightQuant case, offset must be null.");
    }

    if (!IsMxA8W4NZ() && !IsS8S4NZ()) {
        CHECK_COND(gmmParams_.perTokenScaleOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "In WeightQuant case, perTokenScale must be null when xDtype-weightDtype is not "
                   "float8_e4m3fn-float4_e2m1 or int8-int4.");
    } else {
        CHECK_COND(gmmParams_.perTokenScaleOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
                   "In WeightQuant case, perTokenScale must not be null when xDtype-weightDtype is "
                   "float8_e4m3fn-float4_e2m1 or int8-int4.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckDimNumAndFormat(size_t xIdx, size_t yIdx,
                                                                              size_t wIdx) const
{
    if (IsS8S4PseudoQuant()) {
        CHECK_COND(ge::GetPrimaryFormat((*gmmParams_.x)[xIdx]->GetStorageFormat()) == op::Format::FORMAT_ND &&
                       ge::GetPrimaryFormat((*gmmParams_.y)[yIdx]->GetStorageFormat()) == op::Format::FORMAT_ND,
                   ACLNN_ERR_PARAM_INVALID, "In S8S4, x and y must be in ND format.");
    }
    if (unlikely(op::IsPrivateFormat((*gmmParams_.x)[xIdx]->GetStorageFormat()))) {
        OP_LOGE_FOR_INVALID_FORMAT(GetAclnnName(), "x",
                                   op::ToString((*gmmParams_.x)[xIdx]->GetStorageFormat()).GetString(), "ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (unlikely(op::IsPrivateFormat((*gmmParams_.y)[yIdx]->GetStorageFormat()))) {
        OP_LOGE_FOR_INVALID_FORMAT(GetAclnnName(), "y",
                                   op::ToString((*gmmParams_.y)[yIdx]->GetStorageFormat()).GetString(), "ND");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (IsS8S4PseudoQuant()) {
        const auto weightFormat =
            static_cast<op::Format>(ge::GetPrimaryFormat((*gmmParams_.weight)[wIdx]->GetStorageFormat()));
        const bool weightIsNz = weightFormat == op::Format::FORMAT_FRACTAL_NZ;
        const bool weightIsNdCompatible = weightFormat == op::Format::FORMAT_ND ||
                                          weightFormat == op::Format::FORMAT_NCL ||
                                          weightFormat == op::Format::FORMAT_NCHW;
        CHECK_COND(weightIsNdCompatible || weightIsNz, ACLNN_ERR_PARAM_INVALID,
                   "In S8S4, weight format must be ND, NCL, NCHW or FRACTAL_NZ.");
        CHECK_COND(gmmParams_.apiVersion != GMMApiVersion::WeightNz || weightIsNz, ACLNN_ERR_PARAM_INVALID,
                   "In S8S4 through WeightNz, weight must be in FRACTAL_NZ format.");
        CHECK_COND(!IsS8S4SpecialWeightFormat() || weightIsNz, ACLNN_ERR_PARAM_INVALID,
                   "In S8S4, tuningConfig[1]=1 requires FRACTAL_NZ weight.");
    } else if (IsA16W8ND() || IsA16F8ND() || IsA16W4()) {
        if (unlikely(op::IsPrivateFormat((*gmmParams_.weight)[wIdx]->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                GetAclnnName(), "weight", op::ToString((*gmmParams_.weight)[wIdx]->GetStorageFormat()).GetString(),
                "The format of weight must be ND " + GetDataFlowString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (unlikely(!op::IsPrivateFormat((*gmmParams_.weight)[wIdx]->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                GetAclnnName(), "weight", op::ToString((*gmmParams_.weight)[wIdx]->GetStorageFormat()).GetString(),
                "The format of weight must be NZ " + GetDataFlowString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // check dimNum
    size_t xDimNum = (*gmmParams_.x)[xIdx]->GetViewShape().GetDimNum();
    size_t weightDimNum = (*gmmParams_.weight)[wIdx]->GetViewShape().GetDimNum();
    size_t yDimNum = (*gmmParams_.y)[yIdx]->GetViewShape().GetDimNum();

    if (gmmParams_.groupType == NO_SPLIT) {
        if (IsA16W4Pergroup(wIdx)) {
            if (unlikely(xDimNum != MIN_FM_DIM)) {
                std::string reason =
                    "The shape dim of x[" + std::to_string(xIdx) + "] must be 2" + ", when weight separated";
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(GetAclnnName(), "x", std::to_string(xDimNum), reason);
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (unlikely(!(xDimNum <= MAX_FM_DIM && xDimNum >= MIN_FM_DIM))) {
                std::string incorrectDim = "The shape dim of x[" + std::to_string(xIdx) + "] must be within the range ";
                std::string reason = incorrectDim + std::to_string(MIN_FM_DIM) + "-" + std::to_string(MAX_FM_DIM);
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(GetAclnnName(), "x", std::to_string(xDimNum), reason);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        if (unlikely(weightDimNum != MIN_FM_DIM)) {
            std::string reason =
                "The shape dim of weight[" + std::to_string(wIdx) + "] must be 2" + ", when weight separated";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(GetAclnnName(), "weight", std::to_string(weightDimNum), reason);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (unlikely(xDimNum != MIN_FM_DIM)) {
            std::string reason = "The shape dim of x[" + std::to_string(xIdx) + "] must be 2";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(GetAclnnName(), "x", std::to_string(xDimNum), reason);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (IsMxA8W4NZ()) {
            if (unlikely(weightDimNum != MULTI_WEIGHT_DIM && weightDimNum != SPLIT_M_SINGLE_WEIGHT_DIM)) {
                std::string reason = "In weight quant case fp8_e4m3-fp4_e2m1/fp4_e1m2, The shape dim of weight[" +
                                     std::to_string(wIdx) + "] must be 2 or 3";
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(GetAclnnName(), "weight", std::to_string(weightDimNum),
                                                         reason);
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (unlikely(weightDimNum != SPLIT_M_SINGLE_WEIGHT_DIM)) {
                std::string reason = "The shape dim of weight[" + std::to_string(wIdx) + "] must be 3";
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(GetAclnnName(), "weight", std::to_string(weightDimNum),
                                                         reason);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }

    if (unlikely(xDimNum != yDimNum)) {
        std::string incorrectValues = std::to_string(xDimNum) + ", " + std::to_string(yDimNum);
        std::string reason = "The shape dim of x[" + std::to_string(xIdx) + "] " +
                             "must be equal to the shape dim of y[" + std::to_string(yIdx) + "]";
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(GetAclnnName(),
                                               "x[" + std::to_string(xIdx) + "], y[" + std::to_string(yIdx) + "]",
                                               incorrectValues, reason);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckTransposeStatus() const
{
    if (unlikely(gmmParams_.transposeX)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnName(), "transposeX", "true",
                                              "x cannot be transposed, when weight quant");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (gmmParams_.groupType == NO_SPLIT) {
        if (unlikely(gmmParams_.apiVersion == gmm::GMMApiVersion::V1 && gmmParams_.transposeWeight)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                GetAclnnName(), "transposeWeight", "true",
                "weight cannot be transposed, for aclnnGroupedMatmul V1, when x, weight and y are all separated");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (IsA16F8ND() || IsMxA8W4NZ()) {
        if (unlikely(!gmmParams_.transposeWeight)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnName(), "transposeWeight", "false",
                                                  "In weight quant case fp16/bf16-int8, fp16/bf16-fp8/hif8 and "
                                                  "fp8_e4m3-fp4_e2m1/fp4_e1m2, weight must be transposed");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (IsA16MxFp4NZ() || IsS8S4NZ()) {
        if (unlikely(gmmParams_.transposeWeight)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                GetAclnnName(), "transposeWeight", "true",
                "In weight quant case fp16/bf16-fp4_e2m1 and int8-int4, weight must be not transposed");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckDimValue(size_t xIdx, size_t yIdx, size_t wIdx) const
{
    size_t xDimNum = (*gmmParams_.x)[xIdx]->GetViewShape().GetDimNum();
    for (size_t dimIdx = 0; dimIdx < xDimNum - 1; dimIdx++) {
        size_t xDimValue = (*gmmParams_.x)[xIdx]->GetViewShape().GetDim(dimIdx);
        size_t yDimValue = (*gmmParams_.y)[yIdx]->GetViewShape().GetDim(dimIdx);
        if (unlikely(xDimValue != yDimValue)) {
            std::string incorrectValues = std::to_string(xDimValue) + ", " + std::to_string(yDimValue);
            std::string reason = std::to_string(dimIdx) + " of x[" + std::to_string(xIdx) + "] must be equal to axis " +
                                 std::to_string(dimIdx) + " of y[" + std::to_string(yIdx) + "]";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(GetAclnnName(),
                                                   "x[" + std::to_string(xIdx) + "], y[" + std::to_string(yIdx) + "]",
                                                   incorrectValues, reason);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return CheckDimMatching(xIdx, yIdx, wIdx);
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckDimMatching(size_t xIdx, size_t yIdx, size_t wIdx) const
{
    size_t xKDim = 0;
    if (unlikely(GetDimFromEnd((*gmmParams_.x)[xIdx], 1, xKDim, "x[" + std::to_string(xIdx) + "]") != ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t weightKDim = 0;
    const size_t weightKPosFromEnd = IsS8S4SpecialWeightFormat() ? 1 : 2;
    if (unlikely(GetDimFromEnd((*gmmParams_.weight)[wIdx], weightKPosFromEnd, weightKDim,
                               "weight[" + std::to_string(wIdx) + "]") != ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t weightNDim = 0;
    const size_t weightNPosFromEnd = IsS8S4SpecialWeightFormat() ? 2 : 1;
    if (unlikely(GetDimFromEnd((*gmmParams_.weight)[wIdx], weightNPosFromEnd, weightNDim,
                               "weight[" + std::to_string(wIdx) + "]") != ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (unlikely(xKDim != weightKDim)) {
        std::string incorrectValues = "x[" + std::to_string(xIdx) + "] dim k value " + std::to_string(xKDim) +
                                      ", weight[" + std::to_string(wIdx) + "] dim k value " +
                                      std::to_string(weightKDim);
        std::string reason = "x[" + std::to_string(xIdx) + "] dim k value " + std::to_string(xKDim) +
                             " must be equal to weight[" + std::to_string(wIdx) + "] dim k value " +
                             std::to_string(weightKDim);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(GetAclnnName(),
                                               "x[" + std::to_string(xIdx) + "], weight[" + std::to_string(wIdx) + "]",
                                               incorrectValues, reason);
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t yNDim = 0;
    if (unlikely(GetDimFromEnd((*gmmParams_.y)[yIdx], 1, yNDim, "y[" + std::to_string(yIdx) + "]") != ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (unlikely(yNDim != weightNDim)) {
        std::string incorrectValues = "y[" + std::to_string(yIdx) + "] dim n value " + std::to_string(yNDim) +
                                      ", weight[" + std::to_string(wIdx) + "] dim n value " +
                                      std::to_string(weightNDim);
        std::string reason = "y[" + std::to_string(yIdx) + "] dim n value " + std::to_string(yNDim) +
                             " must be equal to weight[" + std::to_string(wIdx) + "] dim n value " +
                             std::to_string(weightNDim);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(GetAclnnName(),
                                               "y[" + std::to_string(yIdx) + "], weight[" + std::to_string(wIdx) + "]",
                                               incorrectValues, reason);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsS8S4PseudoQuant() && (IsA16MxFp4NZ() || IsMxA8W4NZ() || IsS8S4NZ())) {
        if (unlikely(!((weightNDim % N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT == 0) &&
                       (weightKDim % N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT == 0)))) {
            std::string incorrectValues = "n=" + std::to_string(weightNDim) + ", k=" + std::to_string(weightKDim);
            std::string reason = "The value of dim n, k should be an integer multiple of " +
                                 std::to_string(N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(GetAclnnName(), "weight n and k dims", incorrectValues, reason);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckV1GroupList(size_t idx) const
{
    // 多多多 V1接口校验groupList
    if (gmmParams_.groupType != NO_SPLIT || gmmParams_.apiVersion != GMMApiVersion::V1) {
        return ACLNN_SUCCESS;
    }

    size_t xDimNum = (*gmmParams_.x)[idx]->GetViewShape().GetDimNum();
    if (xDimNum > MIN_FM_DIM) {
        CHECK_COND(gmmParams_.groupListOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "groupListOptional should be nullptr when x, y both separated and dim num larger than 2.");
    }

    if (xDimNum == MIN_FM_DIM && gmmParams_.groupListOptional != nullptr) {
        int64_t xMDimValue = (*gmmParams_.x)[idx]->GetViewShape().GetDim(0);
        int64_t preGroupList = idx == 0 ? 0 : (*gmmParams_.groupListOptional)[idx - 1];
        int64_t mValueGroupList = (*gmmParams_.groupListOptional)[idx] - preGroupList;
        std::string errorMessage = idx == 0 ? "groupListOptional[0]" :
                                              "groupListOptional[" + std::to_string(idx) + "] - groupListOptional[" +
                                                  std::to_string(idx - 1) + "]";
        if (unlikely(xMDimValue != mValueGroupList)) {
            std::string incorrectValues = "x[" + std::to_string(idx) + "] dim 0 value " + std::to_string(xMDimValue) +
                                          ", " + errorMessage + " " + std::to_string(mValueGroupList);
            std::string reason = "x[" + std::to_string(idx) + "] dim 0 value " + std::to_string(xMDimValue) +
                                 " should be equal to " + errorMessage + " " + std::to_string(mValueGroupList);
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(GetAclnnName(), "x, groupListOptional", incorrectValues, reason);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckYDtype() const
{
    if (IsS8S4AsymmetricQuant()) {
        if (unlikely(yDtype_ != DataType::DT_FLOAT16)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(GetAclnnName(), "y", op::ToString(yDtype_).GetString(),
                                                  "The dtype of y must be float16 in asymmetric S8S4");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (IsMxA8W4NZ() || IsS8S4NZ()) {
        if (unlikely(yDtype_ != DataType::DT_BF16 && yDtype_ != DataType::DT_FLOAT16)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(GetAclnnName(), "y", op::ToString(yDtype_).GetString(),
                                                  "The dtype of y must be float16 or bfloat16, " + GetDataFlowString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (unlikely(yDtype_ != xDtype_)) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                GetAclnnName(), "y, x",
                std::string(op::ToString(yDtype_).GetString()) + ", " + op::ToString(xDtype_).GetString(),
                "The dtype of y must be equal to the dtype of x in weight quant case");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckBiasDtype()
{
    if (gmmParams_.biasOptional != nullptr) {
        CHECK_RET(CheckTensorNotNullPtr(gmmParams_.biasOptional, 0, "bias") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
        biasDtype_ = (*gmmParams_.biasOptional)[0]->GetDataType();
        if (xDtype_ == DataType::DT_BF16) {
            if (unlikely(!(biasDtype_ == DataType::DT_BF16 || biasDtype_ == DataType::DT_FLOAT))) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    GetAclnnName(), "bias", op::ToString(biasDtype_).GetString(),
                    "the dtype of bias must be bfloat16 or float32 when the dtype of x is bfloat16");
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else if (xDtype_ == DataType::DT_FLOAT16) {
            if (unlikely(biasDtype_ != DataType::DT_FLOAT16)) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    GetAclnnName(), "bias", op::ToString(biasDtype_).GetString(),
                    "the dtype of bias must be float16 when the dtype of x is float16");
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else if (IsMxA8W4NZ()) {
            if (unlikely(biasDtype_ != yDtype_)) {
                std::string incorrectDtypes =
                    std::string(op::ToString(biasDtype_).GetString()) + ", " + op::ToString(yDtype_).GetString();
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    GetAclnnName(), "bias, y", incorrectDtypes,
                    "The dtype of bias must be equal to the dtype of y, " + GetDataFlowString());
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else if (IsS8S4NZ()) {
            if (unlikely(biasDtype_ != DataType::DT_FLOAT)) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(GetAclnnName(), "bias", op::ToString(biasDtype_).GetString(),
                                                      "The dtype of bias must be float32, " + GetDataFlowString());
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckAntiQuantDtype(size_t idx) const
{
    if (gmmParams_.antiquantScaleOptional != nullptr) {
        if (IsA16W8ND() || IsA16F8ND() || IsA16W4()) {
            CHECK_RET(
                CheckTensorDtype(gmmParams_.antiquantScaleOptional, xDtype_, idx, "antiquantScale") == ACLNN_SUCCESS,
                ACLNN_ERR_PARAM_INVALID);
        } else if (IsA16MxFp4NZ() || IsMxA8W4NZ()) {
            CHECK_RET(CheckTensorDtype(gmmParams_.antiquantScaleOptional, ge::DT_FLOAT8_E8M0, idx, "antiquantScale") ==
                          ACLNN_SUCCESS,
                      ACLNN_ERR_PARAM_INVALID);
        } else {
            // S8S4的antiquantScale类型为FP16
            CHECK_RET(CheckTensorDtype(gmmParams_.antiquantScaleOptional, DataType::DT_FLOAT16, idx,
                                       "antiquantScale") == ACLNN_SUCCESS,
                      ACLNN_ERR_PARAM_INVALID);
        }
    }

    if (gmmParams_.antiquantOffsetOptional != nullptr) {
        // 当前有antiquantOffset的数据流，antiquantOffset的数据类型均和x一致
        CHECK_RET(
            CheckTensorDtype(gmmParams_.antiquantOffsetOptional, xDtype_, idx, "antiquantOffset") == ACLNN_SUCCESS,
            ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckAntiQuantShape(size_t idx) const
{
    if (gmmParams_.antiquantScaleOptional != nullptr) {
        CHECK_RET(CheckTensorShape(gmmParams_.antiquantScaleOptional, idx, "antiquantScale") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }

    if (gmmParams_.antiquantOffsetOptional != nullptr) {
        CHECK_RET(CheckTensorShape(gmmParams_.antiquantOffsetOptional, idx, "antiquantOffset") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckDimValueAllOne(const aclTensorList *tensorList,
                                                                             const size_t idx,
                                                                             const std::string &paramName) const
{
    auto viewShape = (*tensorList)[idx]->GetViewShape();
    auto viewShapeDimNum = viewShape.GetDimNum();
    bool dimValueAllOne = true;
    for (size_t i = 0; i < viewShapeDimNum; i++) {
        if (viewShape.GetDim(i) != 1) {
            dimValueAllOne = false;
            break;
        }
    }

    if (dimValueAllOne) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            GetAclnnName(), paramName, "true",
            "When xDtype-weightDtype is fp16/bf16-int4, the dim values of " + paramName + " must not all be 1");
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckAntiQuantTranspose(size_t idx) const
{
    if (IsA16W4()) {
        auto antiquantScaleShape = (*gmmParams_.antiquantScaleOptional)[idx]->GetViewShape();
        auto antiquantScaleDimNum = antiquantScaleShape.GetDimNum();
        size_t perchannelDim = gmmParams_.groupType == SPLIT_M ? 2 : 1; // 单单单场景默认维度为2，多多多场景默认维度为1
        if (antiquantScaleDimNum <= perchannelDim) {
            return ACLNN_SUCCESS;
        }

        CHECK_RET(CheckDimValueAllOne(gmmParams_.antiquantScaleOptional, idx, "antiquantScale") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);

        // pergroup不支持antiquant_scale和antiquant_offset转置
        bool antiquant_scale_trans = IsTransposeLastTwoDims((*gmmParams_.antiquantScaleOptional)[idx]);
        if (antiquant_scale_trans) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                GetAclnnName(), "antiquantScale", "true",
                "When xDtype-weightDtype is fp16/bf16-int4, the transpose of antiquantScale must be false");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (gmmParams_.antiquantOffsetOptional != nullptr) {
            CHECK_RET(CheckDimValueAllOne(gmmParams_.antiquantOffsetOptional, idx, "antiquantOffset") == ACLNN_SUCCESS,
                      ACLNN_ERR_PARAM_INVALID);
            bool antiquant_offset_trans = IsTransposeLastTwoDims((*gmmParams_.antiquantOffsetOptional)[idx]);
            if (antiquant_offset_trans) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    GetAclnnName(), "antiquantOffset", "true",
                    "When xDtype-weightDtype is fp16/bf16-int4, the transpose of antiquantOffset must be false");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckQuantDtype() const
{
    // check pertokenScaleDtype for MxA8W4
    if (IsMxA8W4NZ()) {
        auto pertokenScaleDtype = (*gmmParams_.perTokenScaleOptional)[0]->GetDataType();
        if (unlikely(pertokenScaleDtype != ge::DT_FLOAT8_E8M0)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                GetAclnnName(), "perTokenScale", op::ToString(pertokenScaleDtype).GetString(),
                "The dtype of pertokenScale must be float8_e8m0, " + GetDataFlowString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (IsS8S4NZ()) {
        auto pertokenScaleDtype = (*gmmParams_.perTokenScaleOptional)[0]->GetDataType();
        if (unlikely(pertokenScaleDtype != ge::DT_FLOAT)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(GetAclnnName(), "perTokenScale",
                                                  op::ToString(pertokenScaleDtype).GetString(),
                                                  "The dtype of pertokenScale must be float, " + GetDataFlowString());
            return ACLNN_ERR_PARAM_INVALID;
        }
        const DataType scaleDtype = (*gmmParams_.scaleOptional)[0]->GetDataType();
        if (IsS8S4PseudoQuant()) {
            if (unlikely(scaleDtype != ge::DT_UINT64)) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(GetAclnnName(), "scale", op::ToString(scaleDtype).GetString(),
                                                      "The dtype of scale must be uint64 in S8S4");
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (IsS8S4AsymmetricQuant() && (*gmmParams_.offsetOptional)[0]->GetDataType() != ge::DT_FLOAT) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    GetAclnnName(), "offset", op::ToString((*gmmParams_.offsetOptional)[0]->GetDataType()).GetString(),
                    "The dtype of offset must be float in asymmetric S8S4");
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else if (unlikely(scaleDtype != ge::DT_FLOAT)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(GetAclnnName(), "scale", op::ToString(scaleDtype).GetString(),
                                                  "The dtype of scale must be float, " + GetDataFlowString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckScaleAndPerTokenScaleShape() const
{
    if (IsMxA8W4NZ()) {
        // check pertokenscale shape for MxA8W4
        auto perTokenScaleShape = (*gmmParams_.perTokenScaleOptional)[0]->GetViewShape();
        auto perTokenScaleShapeDimNum = perTokenScaleShape.GetDimNum();
        // MxA8W4NZ仅支持perTokenScale维度为3
        size_t perTokenScaleSupportDimNum = 3;
        if (unlikely(perTokenScaleShapeDimNum != perTokenScaleSupportDimNum)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(GetAclnnName(), "perTokenScale",
                                                     std::to_string(perTokenScaleShapeDimNum),
                                                     "The shape dim of perTokenScale must be 3");
            return ACLNN_ERR_PARAM_INVALID;
        }
        auto xShape = (*gmmParams_.x)[0]->GetViewShape();
        auto perTokenScaleShapeMDim = perTokenScaleShape.GetDim(0);
        auto perTokenScaleShapeKDim = perTokenScaleShape.GetDim(1);
        auto xShapeKDim = xShape.GetDim(1);
        auto xShapeMDim = xShape.GetDim(0);
        if (unlikely(xShapeMDim != perTokenScaleShapeMDim)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetAclnnName(), "perTokenScale",
                                                  std::to_string(perTokenScaleShapeMDim),
                                                  "The first dim of perTokenScale must be equal to the first dim of x");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 64含义：pertokenscale的shape应为(m,k/64,2)
        if (unlikely(xShapeKDim != perTokenScaleShapeKDim * 64)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                GetAclnnName(), "perTokenScale", std::to_string(perTokenScaleShapeKDim),
                "The second dim of x must be 64 times the second dim of perTokenScale");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (IsS8S4NZ()) {
        if (IsS8S4PseudoQuant()) {
            return CheckS8S4ScaleShape();
        }
        return CheckS8S4NZScaleShape();
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckS8S4NZScaleShape() const
{
    auto perTokenScaleShape = (*gmmParams_.perTokenScaleOptional)[0]->GetViewShape();
    auto perTokenScaleShapeDimNum = perTokenScaleShape.GetDimNum();
    if (unlikely(perTokenScaleShapeDimNum != 1)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(GetAclnnName(), "perTokenScale",
                                                 std::to_string(perTokenScaleShapeDimNum),
                                                 "The shape dim of perTokenScale must be 1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto xShape = (*gmmParams_.x)[0]->GetViewShape();
    auto weightShape = (*gmmParams_.weight)[0]->GetViewShape();
    auto xShapeMDim = xShape.GetDim(0);
    auto perTokenScaleShapeMDim = perTokenScaleShape.GetDim(0);
    if (unlikely(xShapeMDim != perTokenScaleShapeMDim)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetAclnnName(), "perTokenScale", std::to_string(perTokenScaleShapeMDim),
                                              "The first axis of pertokenscale must be equal to the first dim of x");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto scaleShape = (*gmmParams_.scaleOptional)[0]->GetViewShape();
    if (unlikely(scaleShape.GetDimNum() != 2)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(GetAclnnName(), "scale", std::to_string(scaleShape.GetDimNum()),
                                                 "The shape dim of scale must be 2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (unlikely(scaleShape.GetDim(0) != weightShape.GetDim(0))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetAclnnName(), "scale", std::to_string(scaleShape.GetDim(0)),
                                              "The dim0 of scale must be equal to g");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (unlikely(scaleShape.GetDim(1) != weightShape.GetDim(2))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnName(), "scale", std::to_string(scaleShape.GetDim(1)),
                                              "The dim1 of scale must be equal to n");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckUnsupportedApi() const
{
    if (IsA16W8ND()) {
        if (gmmParams_.groupType == NO_SPLIT) {
            CHECK_COND(gmmParams_.apiVersion != GMMApiVersion::WeightNz, ACLNN_ERR_PARAM_INVALID,
                       "When xDtype-weightDtype is fp16/bf16-int8, only aclnnGroupedMatmul V1/V2/V3/V4/V5 support "
                       "multi-multi-multi scenario.");
        } else {
            CHECK_COND(gmmParams_.apiVersion != GMMApiVersion::WeightNz && gmmParams_.apiVersion != GMMApiVersion::V1,
                       ACLNN_ERR_PARAM_INVALID,
                       "When xDtype-weightDtype is fp16/bf16-int8, only aclnnGroupedMatmul V2/V3/V4/V5 support "
                       "single-single-single scenario.");
        }
    } else if (IsA16F8ND() || IsA16W4()) {
        CHECK_COND(gmmParams_.apiVersion == GMMApiVersion::V5 || gmmParams_.apiVersion == GMMApiVersion::V4,
                   ACLNN_ERR_PARAM_INVALID,
                   "Only AclnnGroupedMatmulV4/V5 support fp16/bf16-fp8/hif8/int4 for xDtype-weightDtype.");
    } else if (IsS8S4PseudoQuant()) {
        CHECK_COND(gmmParams_.apiVersion == GMMApiVersion::V5 || gmmParams_.apiVersion == GMMApiVersion::WeightNz,
                   ACLNN_ERR_PARAM_INVALID,
                   "Only aclnnGroupedMatmulV5 and aclnnGroupedMatmulWeightNz support this S8S4 "
                   "data flow.");
    } else if (IsA16MxFp4NZ() || IsMxA8W4NZ() || IsS8S4NZ()) {
        CHECK_COND(gmmParams_.apiVersion == GMMApiVersion::WeightNz, ACLNN_ERR_PARAM_INVALID,
                   "Only AclnnGroupedMatmulNz supports fp16/bf16-fp4_e2m1, "
                   "fp8_e4m3fn-fp4_e2m1/fp4_e1m2 and int8-int4 for xDtype-weightDtype.");
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Weight quant case with x dtype [%s] and weight dtype [%s] is not supported.",
                op::ToString(xDtype_).GetString(), op::ToString(weightDtype_).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckS8S4GroupSize(size_t idx) const
{
    if (!IsS8S4AsymmetricQuant()) {
        const op::Shape &weightShape = (*gmmParams_.weight)[idx]->GetViewShape();
        const op::Shape &scaleShape = (*gmmParams_.scaleOptional)[idx]->GetViewShape();
        const size_t weightKDim = IsS8S4SpecialWeightFormat() ? S8S4_WEIGHT_N_DIM : S8S4_WEIGHT_K_DIM;
        const int64_t kSize = weightShape.GetDim(weightKDim);
        const int64_t groupNum = scaleShape.GetDim(S8S4_WEIGHT_K_DIM);
        CHECK_COND(groupNum > 0 && kSize % groupNum == 0, ACLNN_ERR_PARAM_INVALID,
                   "In op [%s], when per-group S8S4, K must be divisible by scale group number.", GetAclnnName());
        const int64_t groupSize = kSize / groupNum;
        CHECK_COND(groupSize == S8S4_QUANT_GROUP_SIZE, ACLNN_ERR_PARAM_INVALID,
                   "In op [%s], when per-group S8S4, groupSize must be 256, but got %ld.", GetAclnnName(), groupSize);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckGroupSize(size_t idx) const
{
    if (IsS8S4PseudoQuant()) {
        return CheckS8S4GroupSize(idx);
    }
    if (!(IsA16MxFp4NZ() || IsMxA8W4NZ() || IsS8S4NZ() || IsA16W4Pergroup(idx))) {
        return ACLNN_SUCCESS;
    }

    auto antiquantScaleShape = (*gmmParams_.antiquantScaleOptional)[idx]->GetViewShape();
    auto weightShape = (*gmmParams_.weight)[idx]->GetViewShape();
    auto antiquantScaleDimNum = antiquantScaleShape.GetDimNum();
    int64_t groupSize = 0;

    // 2含义: (g,k,n)的k轴索引
    int64_t kSize = weightShape.GetDim(weightShape.GetDimNum() - 2);
    // 2含义: (g,k/groupSize,n)的k轴索引
    int64_t groupNum = IsMxA8W4NZ() ? antiquantScaleShape.GetDim(antiquantScaleDimNum - 3) * 2 :
                                      antiquantScaleShape.GetDim(antiquantScaleDimNum - 2);
    if (unlikely(groupNum <= 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnName(), "groupNum", std::to_string(groupNum),
                                              "The value of groupNum must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (unlikely(kSize % groupNum != 0)) {
        std::string incorrectValues = "kSize: " + std::to_string(kSize) + ", groupNum: " + std::to_string(groupNum);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(GetAclnnName(), "kSize, groupNum", incorrectValues,
                                               "kSize must be a multiple of groupNum");
        return ACLNN_ERR_PARAM_INVALID;
    }
    groupSize = kSize / groupNum;
    if (IsMxA8W4NZ()) {
        // 2：MxA8W4NZ的antiquantScaleViewShape: (g, k / groupSIze / 2, n, 2)
        groupSize = kSize / groupNum;
    }
    // 当前伪量化仅支持groupsize为32整数倍
    if (IsS8S4NZ()) {
        // 伪量化S8S4场景支持groupsize为128/192/256/512
        if (unlikely(!(groupSize == 128 || groupSize == 256 || groupSize == 512 || groupSize == 192))) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnName(), "groupSize", std::to_string(groupSize),
                                                  "The value of groupSize must be 128/192/256/512");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (IsA16W4Pergroup(idx)) {
        // 伪量化A16W4 pergroup场景支持groupsize为32/64/128/256
        if (unlikely(!(groupSize == 32 || groupSize == 64 || groupSize == 128 || groupSize == 256))) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                GetAclnnName(), "groupSize", std::to_string(groupSize),
                "When the quant mode is pergroup, The value of groupSize must be 32/64/128/256 on arch3510");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        // 当前伪量化非S8S4仅支持groupsize为32
        if (unlikely(groupSize != 32)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnName(), "groupSize", std::to_string(groupSize),
                                                  "The value of groupSize must be 32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckGroupTypeScenario() const
{
    std::string errorMessage;

    // groupType校验
    // V1接口没有groupType字段，是在aclnnGroupedMatmulGetWorkspaceSize赋值的，只会出现0/-1，不会出现2
    if (IsA16W8ND() || IsA16W4()) {
        if (unlikely(!(gmmParams_.groupType == NO_SPLIT || gmmParams_.groupType == SPLIT_M))) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                GetAclnnName(), "groupType", std::to_string(gmmParams_.groupType),
                "GMM only support groupType 0 (split M) or groupType -1 (no split), " + GetDataFlowString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        errorMessage =
            gmmParams_.apiVersion == gmm::GMMApiVersion::V1 ?
                "M-split scenario, but the actual scenario is no-split (multi-multi-multi)" :
                "groupType 0 (split M), but the actual groupType is [" + std::to_string(gmmParams_.groupType) + "]";
        if (unlikely(gmmParams_.groupType != SPLIT_M)) {
            std::string reason = std::string("Weight quant cases with x dtype [") + op::ToString(xDtype_).GetString() +
                                 "] and weight dtype [" + op::ToString(weightDtype_).GetString() + "] only support " +
                                 errorMessage;
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnName(), "groupType", std::to_string(gmmParams_.groupType),
                                                  reason);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // 多多多/单单单校验
    // groupType为-1仅对应多多多；groupType为0会出现单单单/单多单/单多多/多多单，仅支持单单单
    if (gmmParams_.groupType == NO_SPLIT) {
        if (unlikely(
                !(gmmParams_.x->Size() == gmmParams_.weight->Size() && gmmParams_.x->Size() == gmmParams_.y->Size()))) {
            std::string incorrectSizes = "x size: " + std::to_string(gmmParams_.x->Size()) +
                                         ", weight size: " + std::to_string(gmmParams_.weight->Size()) +
                                         ", y size: " + std::to_string(gmmParams_.y->Size());
            OP_LOGE_FOR_INVALID_LISTSIZE(
                GetAclnnName(), "x, weight, y", incorrectSizes,
                "In multi-multi-multi scenario, the sizes of x, weight and y should be all the same");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        errorMessage =
            gmmParams_.apiVersion == gmm::GMMApiVersion::V1 ? "When split axis is M" : "When groupType is 0 (split M)";
        if (IsMxA8W4NZ()) {
            if (unlikely(!(gmmParams_.x->Size() == 1 && gmmParams_.y->Size() == 1 && gmmParams_.weight->Size() >= 1))) {
                std::string incorrectSizes = "x size: " + std::to_string(gmmParams_.x->Size()) +
                                             ", weight size: " + std::to_string(gmmParams_.weight->Size()) +
                                             ", y size: " + std::to_string(gmmParams_.y->Size());
                std::string reason =
                    errorMessage + ", When the quant mode is A8W4 MX, x and y size must be 1, weight size must be >= 1";
                OP_LOGE_FOR_INVALID_LISTSIZE(GetAclnnName(), "x, weight, y", incorrectSizes, reason);
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (unlikely(!(gmmParams_.x->Size() == 1 && gmmParams_.weight->Size() == 1 && gmmParams_.y->Size() == 1))) {
                std::string incorrectSizes = "x size: " + std::to_string(gmmParams_.x->Size()) +
                                             ", weight size: " + std::to_string(gmmParams_.weight->Size()) +
                                             ", y size: " + std::to_string(gmmParams_.y->Size());
                std::string reason = errorMessage + ", the sizes of x, weight and y should all be 1";
                OP_LOGE_FOR_INVALID_LISTSIZE(GetAclnnName(), "x, weight, y", incorrectSizes, reason);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckGroupListAndSplitItem() const
{
    if (gmmParams_.groupType == NO_SPLIT) {
        if (gmmParams_.apiVersion == GMMApiVersion::V2) {
            CHECK_COND(gmmParams_.groupListOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                       "GroupListOptional should be nullptr when groupType is -1.");
        } else if (gmmParams_.apiVersion != GMMApiVersion::V1) {
            CHECK_COND(gmmParams_.groupTensorOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                       "GroupListOptional(groupTensorOptional) should be nullptr when groupType is -1.");
        }
        if (unlikely(!(gmmParams_.splitItem == X_Y_SEPARATED || gmmParams_.splitItem == Y_SEPARATED))) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnName(), "splitItem", std::to_string(gmmParams_.splitItem),
                                                  "When y is separated, splitItem must be 0 or 1");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (gmmParams_.apiVersion == gmm::GMMApiVersion::V1 || gmmParams_.apiVersion == gmm::GMMApiVersion::V2) {
            CHECK_COND(gmmParams_.groupListOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
                       "GroupListOptional should not be nullptr when split axis is M."); // V1 没有groupType参数
        } else {
            CHECK_COND(gmmParams_.groupTensorOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
                       "GroupListOptional should not be nullptr when groupType is 0.");
        }

        if (unlikely(!(gmmParams_.splitItem == X_SEPARATED || gmmParams_.splitItem == NO_SEPARATED))) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnName(), "splitItem", std::to_string(gmmParams_.splitItem),
                                                  "When y is not separated, splitItem must be 2 or 3");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (IsS8S4PseudoQuant()) {
        CHECK_RET(CheckS8S4GroupListType() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckOptionalListSize(const aclTensorList *optional,
                                                                               const aclTensorList *reference,
                                                                               const char *optionalName,
                                                                               const char *referenceName,
                                                                               const char *opName)
{
    if (optional == nullptr) {
        return ACLNN_SUCCESS;
    }
    if (unlikely(optional->Size() != reference->Size())) {
        std::string incorrectSizes = std::string(optionalName) + " size: " + std::to_string(optional->Size()) + ", " +
                                     std::string(referenceName) + " size: " + std::to_string(reference->Size());
        std::string combinedName = std::string(optionalName) + ", " + std::string(referenceName);
        std::string reason =
            std::string(optionalName) + " size must be equal to " + std::string(referenceName) + " size";
        OP_LOGE_FOR_INVALID_LISTSIZE(opName, combinedName.c_str(), incorrectSizes, reason.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckTensorListSize() const
{
    CHECK_RET(CheckOptionalListSize(gmmParams_.antiquantScaleOptional, gmmParams_.weight, "antiquantScaleOptional",
                                    "weight", GetAclnnName()) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalListSize(gmmParams_.antiquantOffsetOptional, gmmParams_.weight, "antiquantOffsetOptional",
                                    "weight", GetAclnnName()) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalListSize(gmmParams_.biasOptional, gmmParams_.weight, "biasOptional", "weight",
                                    GetAclnnName()) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalListSize(gmmParams_.perTokenScaleOptional, gmmParams_.x, "perTokenScaleOptional", "x",
                                    GetAclnnName()) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalListSize(gmmParams_.scaleOptional, gmmParams_.weight, "scaleOptional", "weight",
                                    GetAclnnName()) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalListSize(gmmParams_.offsetOptional, gmmParams_.weight, "offsetOptional", "weight",
                                    GetAclnnName()) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckGroupedMatmulWeightQuantDAV3510()
{
    xDtype_ = gmmParams_.xDtype;
    weightDtype_ = (*gmmParams_.weight)[0]->GetDataType();
    yDtype_ = (*gmmParams_.y)[0]->GetDataType();

    CHECK_COND(CheckGroupTypeScenario() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckGroupTypeScenario failed.");
    CHECK_COND(CheckUnsupportedApi() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckUnsupportedApi failed.");
    CHECK_COND(CheckGroupListAndSplitItem() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckGroupListAndSplitItem failed.");

    // CheckAntiQuantParams和CheckQuantParams校验各种量化参数在各数据流的支持情况，后续对量化参数的通用校验不再区分数据流
    CHECK_COND(CheckAntiQuantParams() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckAntiQuantParams failed.");
    CHECK_COND(CheckQuantParams() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckQuantParams failed!");
    CHECK_COND(CheckTensorListSize() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckTensorListSize failed.");
    CHECK_RET(CheckYDtype() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckQuantDtype() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckBiasDtype() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckTransposeStatus() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckScaleAndPerTokenScaleShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    size_t loopCount = gmmParams_.weight->Size();
    for (size_t wIdx = 0; wIdx < loopCount; wIdx++) {
        size_t xIdx = std::min(wIdx, gmmParams_.x->Size() - 1);
        size_t yIdx = std::min(wIdx, gmmParams_.y->Size() - 1);

        CHECK_RET(CheckTensorNotNull(xIdx, yIdx, wIdx) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

        CHECK_RET(CheckTensorDtype(gmmParams_.x, xDtype_, xIdx, "x") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckTensorDtype(gmmParams_.weight, weightDtype_, wIdx, "weight") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckTensorDtype(gmmParams_.y, yDtype_, yIdx, "y") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

        CHECK_RET(CheckDimNumAndFormat(xIdx, yIdx, wIdx) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        if (unlikely(IsTransposeLastTwoDims((*gmmParams_.weight)[wIdx]) != gmmParams_.transposeWeight)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetAclnnName(), "weight[" + std::to_string(wIdx) + "]",
                                                  "inconsistent",
                                                  "The transpose state must be the same for each tensor in weight");
            return ACLNN_ERR_PARAM_INVALID;
        }
        CHECK_RET(CheckDimValue(xIdx, yIdx, wIdx) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckWeightInnerAxisEven(wIdx) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

        CHECK_RET(CheckV1GroupList(xIdx) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

        if (gmmParams_.biasOptional != nullptr) {
            CHECK_RET(CheckTensorDtype(gmmParams_.biasOptional, biasDtype_, wIdx, "bias") == ACLNN_SUCCESS,
                      ACLNN_ERR_PARAM_INVALID);
            CHECK_RET(CheckTensorShape(gmmParams_.biasOptional, wIdx, "bias") == ACLNN_SUCCESS,
                      ACLNN_ERR_PARAM_INVALID);
        }

        CHECK_RET(CheckAntiQuantDtype(wIdx) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckAntiQuantShape(wIdx) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckAntiQuantTranspose(wIdx) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_COND(CheckGroupSize(wIdx) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckGroupSize failed");
    }
    return ACLNN_SUCCESS;
}
