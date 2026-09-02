/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_950_checker.h"
#include "log/log.h"

#include <sstream>

using namespace gmm;

namespace gmm {
template class AclnnGroupedMatmulDAV3510Checker<aclTensorList>;
template class AclnnGroupedMatmulDAV3510Checker<aclTensor>;
} // namespace gmm

namespace {
#define GMM_CHECK_REPORT(cond, reportExpr) \
    do { \
        if (unlikely(!(cond))) { \
            reportExpr; \
            return ACLNN_ERR_PARAM_INVALID; \
        } \
    } while (false)

const aclTensor *GetInputTensor(const aclTensorList *input, size_t index = 0)
{
    if (input == nullptr || index >= input->Size()) {
        return nullptr;
    }
    return (*input)[index];
}

const aclTensor *GetInputTensor(const aclTensor *input, size_t index = 0)
{
    (void)index;
    return input;
}

size_t GetInputTensorSize(const aclTensorList *input)
{
    return input == nullptr ? 0UL : input->Size();
}

size_t GetInputTensorSize(const aclTensor *input)
{
    return input == nullptr ? 0UL : 1UL;
}

std::string ShapeToStringWithoutBracket(const op::Shape *shape)
{
    if (shape == nullptr) {
        return "nullptr";
    }
    std::ostringstream oss;
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << shape->GetDim(i);
    }
    return oss.str();
}

std::string ViewShapeToString(const aclTensor *tensor)
{
    return tensor == nullptr ? "nullptr" : ShapeToStringWithoutBracket(&tensor->GetViewShape());
}

std::string StorageShapeToString(const aclTensor *tensor)
{
    return tensor == nullptr ? "nullptr" : ShapeToStringWithoutBracket(&tensor->GetStorageShape());
}
} // namespace

template <typename T>
bool AclnnGroupedMatmulDAV3510Checker<T>::IsMxfp4() const
{
    const auto *weightTensor = GetInputTensor(gmmParams_.weight);
    if (weightTensor == nullptr) {
        return false;
    }
    const auto weightDtype = weightTensor->GetDataType();
    return (gmmParams_.xDtype == DataType::DT_FLOAT4_E2M1 || gmmParams_.xDtype == DataType::DT_FLOAT4_E1M2) &&
           (weightDtype == DataType::DT_FLOAT4_E2M1 || weightDtype == DataType::DT_FLOAT4_E1M2);
}

template <typename T>
bool AclnnGroupedMatmulDAV3510Checker<T>::IsMultiTensorWeight() const
{
    const auto *weightTensor = GetInputTensor(gmmParams_.weight);
    return weightTensor != nullptr && weightTensor->GetViewShape().GetDimNum() == MIN_FM_DIM;
}

template <typename T>
bool AclnnGroupedMatmulDAV3510Checker<T>::IsWeightNzMultiTensorLayout() const
{
    const auto *weightTensor = GetInputTensor(gmmParams_.weight);
    return gmmParams_.groupType == SPLIT_M && GetInputTensorSize(gmmParams_.x) == 1 &&
           GetInputTensorSize(gmmParams_.y) == 1 && IsMultiTensorWeight() && weightTensor != nullptr &&
           weightTensor->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ;
}

template <typename T>
typename AclnnGroupedMatmulDAV3510Checker<T>::TensorIndexInfo AclnnGroupedMatmulDAV3510Checker<T>::GetTensorIndexInfo(
    size_t index) const
{
    const bool singleMultiSingle = IsWeightNzMultiTensorLayout();
    return {
        singleMultiSingle ? GetInputTensorSize(gmmParams_.weight) : GetInputTensorSize(gmmParams_.x),
        singleMultiSingle ? 0UL : index,
        index,
        singleMultiSingle ? 0UL : index,
        index,
        singleMultiSingle ? 0UL : index,
        singleMultiSingle ? 0UL : index,
    };
}

template <typename T>
const char *AclnnGroupedMatmulDAV3510Checker<T>::GetAclnnOpName() const
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

template <typename T>
void AclnnGroupedMatmulDAV3510Checker<T>::SetInputName(const std::string &xName, const std::string &weightName,
                                                       const std::string &perTokenScaleName,
                                                       const std::string &scaleName, const std::string &groupTensorName)
{
    this->xName_ = xName;
    this->weightName_ = weightName;
    this->perTokenScaleName_ = perTokenScaleName;
    this->scaleName_ = scaleName;
    this->groupTensorName_ = groupTensorName;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckInputTensorsNotNull() const
{
    const auto checkTensorContainer = [&](const auto *input, const std::string &name, bool required) -> aclnnStatus {
        if (input == nullptr) {
            if (!required) {
                return ACLNN_SUCCESS;
            }
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnOpName(), name.c_str(), "nullptr",
                                                  "required input does not support nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        const size_t tensorNum = GetInputTensorSize(input);
        if (tensorNum == 0UL) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                GetAclnnOpName(), name.c_str(), "empty",
                "required input must contain at least one tensor and does not support nullptr");
            return required ? ACLNN_ERR_PARAM_INVALID : ACLNN_ERR_PARAM_NULLPTR;
        }
        for (size_t i = 0UL; i < tensorNum; ++i) {
            if (GetInputTensor(input, i) == nullptr) {
                const std::string indexedName = name + "[" + std::to_string(i) + "]";
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnOpName(), indexedName.c_str(), "nullptr",
                                                      "input tensor does not support nullptr");
                return ACLNN_ERR_PARAM_NULLPTR;
            }
        }
        return ACLNN_SUCCESS;
    };

    aclnnStatus status = checkTensorContainer(gmmParams_.x, xName_, true);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = checkTensorContainer(gmmParams_.weight, weightName_, true);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = checkTensorContainer(gmmParams_.y, yName_, true);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (gmmParams_.groupTensorOptional == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnOpName(), groupTensorName_.c_str(), "nullptr",
                                              "required input does not support nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    status = checkTensorContainer(gmmParams_.scaleOptional, scaleName_, false);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = checkTensorContainer(gmmParams_.perTokenScaleOptional, perTokenScaleName_, false);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return checkTensorContainer(gmmParams_.biasOptional, biasName_, false);
}

template <typename T>
bool AclnnGroupedMatmulDAV3510Checker<T>::LastTwoDimValueIsOne(const aclTensor *tensor) const
{
    if (tensor == nullptr) {
        return false;
    }
    // 检查tensor维度是否小于2
    if (tensor->GetViewShape().GetDimNum() < 2) {
        return false;
    }
    auto dim1 = tensor->GetViewShape().GetDimNum() - 1;
    auto dim2 = tensor->GetViewShape().GetDimNum() - LAST_TWO_DIM_INDEX;
    if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
        return true;
    }
    return false;
}

template <typename T>
bool AclnnGroupedMatmulDAV3510Checker<T>::CheckTensorListSizeForEachInput() const
{
    const auto *yTensor = GetInputTensor(gmmParams_.y);
    if (yTensor == nullptr) {
        return false;
    }
    if (gmmParams_.scaleOptional == nullptr && yTensor->GetDataType() == DataType::DT_INT32) {
        return gmmParams_.perTokenScaleOptional == nullptr ||
               GetInputTensorSize(gmmParams_.perTokenScaleOptional) == GetInputTensorSize(gmmParams_.x);
    }
    size_t expectedScaleTensorNum =
        IsWeightNzMultiTensorLayout() ? GetInputTensorSize(gmmParams_.weight) : GetInputTensorSize(gmmParams_.x);
    if (GetInputTensorSize(gmmParams_.scaleOptional) != expectedScaleTensorNum) {
        return false;
    }
    if (gmmParams_.perTokenScaleOptional != nullptr &&
        GetInputTensorSize(gmmParams_.perTokenScaleOptional) != GetInputTensorSize(gmmParams_.x)) {
        return false;
    }
    return true;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGeneralQuantShape() const
{
    if (!CheckTensorListSizeForEachInput()) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    const auto tensorIndexInfo = GetTensorIndexInfo();
    for (size_t i = 0; i < tensorIndexInfo.loopSize; i++) {
        const auto tensorIndex = GetTensorIndexInfo(i);
        const auto *xTensor = GetInputTensor(gmmParams_.x, tensorIndex.x);
        const auto *weightTensor = GetInputTensor(gmmParams_.weight, tensorIndex.weight);
        const auto *yTensor = GetInputTensor(gmmParams_.y, tensorIndex.y);
        GMM_CHECK_REPORT(xTensor != nullptr && weightTensor != nullptr && yTensor != nullptr,
                         OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnOpName(), "x, weight or y", "nullptr",
                                                               "every tensor used by grouped matmul must be non-null"));
        auto weightNIndex = weightTensor->GetViewShape().GetDimNum() - 1;
        auto xMDim = xTensor->GetViewShape().GetDim(0);
        auto weightNDim = weightTensor->GetViewShape().GetDim(weightNIndex);
        if (xMDim == 0 || weightNDim == 0) {
            return ACLNN_SUCCESS;
        }
        if (gmmParams_.groupType == SPLIT_K) {
            auto yFirstDim = yTensor->GetViewShape().GetDim(0);
            GMM_CHECK_REPORT(
                yFirstDim == groupNum,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    GetAclnnOpName(), yName_.c_str(), ViewShapeToString(yTensor),
                    "when groupType is 2 (split K), axis 0 of y must be equal to axis 0 of groupTensor [" +
                        std::to_string(groupNum) + "], but the current value is " + std::to_string(yFirstDim)));
        } else {
            auto weightKIndex = weightTensor->GetViewShape().GetDimNum() - LAST_TWO_DIM_INDEX;
            auto xKDim = xTensor->GetViewShape().GetDim(1);
            GMM_CHECK_REPORT(xKDim > 0, OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                            GetAclnnOpName(), xName_.c_str(), ViewShapeToString(xTensor),
                                            "when groupType is 0 (split M) and when the M or N value is not 0 [M=" +
                                                std::to_string(xMDim) + ", N=" + std::to_string(weightNDim) +
                                                "], axis K of x must be a positive number, but the current value is " +
                                                std::to_string(xKDim)));
            auto weightKDim = weightTensor->GetViewShape().GetDim(weightKIndex);
            GMM_CHECK_REPORT(weightKDim > 0,
                             OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                 GetAclnnOpName(), weightName_.c_str(), ViewShapeToString(weightTensor),
                                 "when groupType is 0 (split M) and when the M or N value is not 0 [M=" +
                                     std::to_string(xMDim) + ", N=" + std::to_string(weightNDim) +
                                     "], axis K of weight must be a positive number, but the "
                                     "current value is " +
                                     std::to_string(weightKDim)));
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckQuantCasesFormat() const
{
    const auto tensorIndexInfo = GetTensorIndexInfo();
    for (size_t i = 0; i < tensorIndexInfo.loopSize; i++) {
        const auto tensorIndex = GetTensorIndexInfo(i);
        auto xFormat = GetInputTensor(gmmParams_.x, tensorIndex.x)->GetStorageFormat();
        GMM_CHECK_REPORT(
            !op::IsPrivateFormat(xFormat),
            OP_LOGE_FOR_INVALID_FORMAT(GetAclnnOpName(), xName_.c_str(), op::ToString(xFormat).GetString(), "ND"));
        auto weightFormat = GetInputTensor(gmmParams_.weight, tensorIndex.weight)->GetStorageFormat();
        GMM_CHECK_REPORT(!op::IsPrivateFormat(weightFormat) || weightFormat == op::Format::FORMAT_FRACTAL_NZ,
                         OP_LOGE_FOR_INVALID_FORMAT(GetAclnnOpName(), weightName_.c_str(),
                                                    op::ToString(weightFormat).GetString(), "ND or FRACTAL_NZ"));
        auto yFormat = GetInputTensor(gmmParams_.y, tensorIndex.y)->GetStorageFormat();
        GMM_CHECK_REPORT(
            !op::IsPrivateFormat(yFormat),
            OP_LOGE_FOR_INVALID_FORMAT(GetAclnnOpName(), yName_.c_str(), op::ToString(yFormat).GetString(), "ND"));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckWeightStorageShape(const aclTensor *weightTensor,
                                                                         int64_t kDimValue, int64_t nDimValue) const
{
    auto weightStorage = weightTensor->GetStorageShape();
    auto weightStorageShapeDim = weightStorage.GetDimNum();
    const bool validWeightNzStorageDim =
        weightStorageShapeDim == QUANT_WEIGHTNZ_STORAGE_DIM ||
        (IsWeightNzMultiTensorLayout() && weightStorageShapeDim == WEIGHTNZ_MULTI_TENSOR_STORAGE_DIM);
    GMM_CHECK_REPORT(validWeightNzStorageDim,
                     OP_LOGE_FOR_INVALID_SHAPEDIM(
                         GetAclnnOpName(), weightName_.c_str(), std::to_string(weightStorageShapeDim),
                         IsWeightNzMultiTensorLayout() ? std::to_string(WEIGHTNZ_MULTI_TENSOR_STORAGE_DIM) + " or " +
                                                             std::to_string(QUANT_WEIGHTNZ_STORAGE_DIM) :
                                                         std::to_string(QUANT_WEIGHTNZ_STORAGE_DIM)));

    auto weightStorageLastFourthDim = weightStorage.GetDim(weightStorageShapeDim - LAST_FOURTH_DIM_INDEX);
    auto weightStorageLastThirdDim = weightStorage.GetDim(weightStorageShapeDim - LAST_THIRD_DIM_INDEX);
    auto weightStorageLastSecondDim = weightStorage.GetDim(weightStorageShapeDim - LAST_SECOND_DIM_INDEX);
    auto weightStorageLastDim = weightStorage.GetDim(weightStorageShapeDim - LAST_FIRST_DIM_INDEX);

    const bool isInt4 = (gmmParams_.xDtype == DataType::DT_INT4);
    const bool isMxfp4 =
        (gmmParams_.xDtype == DataType::DT_FLOAT4_E2M1 || gmmParams_.xDtype == DataType::DT_FLOAT4_E1M2);
    // A5 aclnnConvertWeightToINT4Pack generates an A4W4 NZ storage tile whose logical
    // INT4 C0 is 32. Keep accepting the historical C0=64 layout as well; the selected
    // C0 is used below to validate the two outer NZ dimensions consistently.
    const bool validInt4C0 = weightStorageLastDim == CUBE_BLOCK_SIZE_32 || weightStorageLastDim == CUBE_BLOCK_SIZE_64;
    const int64_t cubeBlockSizeK = isInt4 ? weightStorageLastDim : (isMxfp4 ? CUBE_BLOCK_SIZE_64 : CUBE_BLOCK_SIZE_32);
    GMM_CHECK_REPORT(isInt4 ? validInt4C0 : weightStorageLastDim == cubeBlockSizeK,
                     OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                         GetAclnnOpName(), weightName_.c_str(), StorageShapeToString(weightTensor),
                         "when the format of weight is FRACTAL_NZ, storage shape last dim of weight must be equal to "
                         "32 for fp8 dtype, 32 or 64 for int4 dtype, or 64 for mxfp4 dtype, but the current value is " +
                             std::to_string(weightStorageLastDim)));
    GMM_CHECK_REPORT(
        weightStorageLastSecondDim == CUBE_BLOCK_SIZE_16,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            GetAclnnOpName(), weightName_.c_str(), StorageShapeToString(weightTensor),
            "when the format of weight is FRACTAL_NZ, storage shape last second dim of weight must be equal to 16, "
            "but the current value is " +
                std::to_string(weightStorageLastSecondDim)));
    if (gmmParams_.transposeWeight) {
        GMM_CHECK_REPORT(
            weightStorageLastFourthDim == (kDimValue + cubeBlockSizeK - 1) / cubeBlockSizeK,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                GetAclnnOpName(), weightName_.c_str(), StorageShapeToString(weightTensor),
                "when the format of weight is FRACTAL_NZ and transposition is true, storage shape second dim "
                "of weight must be equal to ceil(k/32) for fp8 dtype or ceil(k/64) for fp4 dtype [" +
                    std::to_string((kDimValue + cubeBlockSizeK - 1) / cubeBlockSizeK) + "], but the current value is " +
                    std::to_string(weightStorageLastFourthDim)));
        GMM_CHECK_REPORT(
            weightStorageLastThirdDim == (nDimValue + CUBE_BLOCK_SIZE_16 - 1) / CUBE_BLOCK_SIZE_16,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                GetAclnnOpName(), weightName_.c_str(), StorageShapeToString(weightTensor),
                "when the format of weight is FRACTAL_NZ and transposition is true, storage shape third dim "
                "of weight must be equal to ceil(n/16) [" +
                    std::to_string((nDimValue + CUBE_BLOCK_SIZE_16 - 1) / CUBE_BLOCK_SIZE_16) +
                    "], but the current value is " + std::to_string(weightStorageLastThirdDim)));
    } else {
        GMM_CHECK_REPORT(
            weightStorageLastFourthDim == (nDimValue + cubeBlockSizeK - 1) / cubeBlockSizeK,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                GetAclnnOpName(), weightName_.c_str(), StorageShapeToString(weightTensor),
                "when the format of weight is FRACTAL_NZ and transposition is false, storage shape second dim "
                "of weight must be equal to ceil(n/32) for fp8 dtype or ceil(n/64) for fp4 dtype [" +
                    std::to_string((nDimValue + cubeBlockSizeK - 1) / cubeBlockSizeK) + "], but the current value is " +
                    std::to_string(weightStorageLastFourthDim)));
        GMM_CHECK_REPORT(
            weightStorageLastThirdDim == (kDimValue + CUBE_BLOCK_SIZE_16 - 1) / CUBE_BLOCK_SIZE_16,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                GetAclnnOpName(), weightName_.c_str(), StorageShapeToString(weightTensor),
                "when the format of weight is FRACTAL_NZ and transposition is false, storage shape third dim "
                "of weight must be equal to ceil(k/16) [" +
                    std::to_string((kDimValue + CUBE_BLOCK_SIZE_16 - 1) / CUBE_BLOCK_SIZE_16) +
                    "], but the current value is " + std::to_string(weightStorageLastThirdDim)));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckWeightNzSpecialParams() const
{
    CHECK_RET(CheckWeightNzMultiTensorElements() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    GMM_CHECK_REPORT(
        gmmParams_.apiVersion == gmm::GMMApiVersion::WeightNz,
        OP_LOGE_FOR_INVALID_VALUE(GetAclnnOpName(), "apiVersion", GetAclnnOpName(), "aclnnGroupedMatmulWeightNz"));

    auto wDtype = GetInputTensor(gmmParams_.weight)->GetDataType();
    bool isInputFp8e4m3 = gmmParams_.xDtype == DataType::DT_FLOAT8_E4M3FN && wDtype == DataType::DT_FLOAT8_E4M3FN;
    bool isInputFp4 =
        (gmmParams_.xDtype == DataType::DT_FLOAT4_E2M1 || gmmParams_.xDtype == DataType::DT_FLOAT4_E1M2) &&
        (wDtype == DataType::DT_FLOAT4_E2M1 || wDtype == DataType::DT_FLOAT4_E1M2);
    bool isInputInt4 = gmmParams_.xDtype == DataType::DT_INT4 && wDtype == DataType::DT_INT4;
    std::string xWeightDtypes = std::string("x=") + op::ToString(gmmParams_.xDtype).GetString() +
                                ", weight=" + op::ToString(wDtype).GetString();
    GMM_CHECK_REPORT((gmmParams_.xDtype == DataType::DT_INT8 && wDtype == DataType::DT_INT8) || isInputFp8e4m3 ||
                         isInputFp4 || isInputInt4,
                     OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                         GetAclnnOpName(), "x and weight", xWeightDtypes.c_str(),
                         "when the format of weight is FRACTAL_NZ, the dtypes of x and weight must be within the range "
                         "INT8, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2, INT4"));
    if (isInputFp8e4m3 || isInputFp4) {
        GMM_CHECK_REPORT(
            gmmParams_.perTokenScaleOptional != nullptr,
            OP_LOGE_FOR_INVALID_VALUE(GetAclnnOpName(), perTokenScaleName_.c_str(), "nullptr", "not nullptr"));
        DataType scaleDtype = GetInputTensor(gmmParams_.scaleOptional)->GetDataType();
        DataType perTokenDtype = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetDataType();
        std::string scaleDtypes = std::string("scale=") + op::ToString(scaleDtype).GetString() +
                                  ", perTokenScale=" + op::ToString(perTokenDtype).GetString();
        GMM_CHECK_REPORT((scaleDtype == DataType::DT_FLOAT8_E8M0 && perTokenDtype == DataType::DT_FLOAT8_E8M0),
                         OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                             GetAclnnOpName(), "scale and perTokenScale", scaleDtypes.c_str(),
                             "when the format of weight is FRACTAL_NZ and when the dtypes of x and weight are "
                             "FLOAT8_E4M3FN/FLOAT8_E4M3FN or FLOAT4_E2M1|FLOAT4_E1M2 combinations, the dtypes of scale "
                             "and perTokenScale must be within the range FLOAT8_E8M0"));
    }

    auto yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
    GMM_CHECK_REPORT(yDtype != DataType::DT_INT8,
                     OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                         GetAclnnOpName(), yName_.c_str(), op::ToString(yDtype).GetString(),
                         "when the format of weight is FRACTAL_NZ, the dtype of y must be within the range "
                         "FLOAT16, BFLOAT16, FLOAT32, INT32"));

    GMM_CHECK_REPORT(
        !(IsWeightNzMultiTensorLayout() && gmmParams_.biasOptional != nullptr),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            GetAclnnOpName(), "biasOptional", "not nullptr",
            "bias is not supported when x and y each contain one tensor and weight contains multiple NZ tensors"));
    return CheckWeightNzTensorShapes();
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckWeightNzMultiTensorElements() const
{
    if (!IsWeightNzMultiTensorLayout()) {
        return ACLNN_SUCCESS;
    }
    const size_t weightTensorNum = GetInputTensorSize(gmmParams_.weight);
    for (size_t i = 0UL; i < weightTensorNum; ++i) {
        GMM_CHECK_REPORT(
            GetInputTensor(gmmParams_.weight, i) != nullptr,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                GetAclnnOpName(), weightName_.c_str(), ("weight[" + std::to_string(i) + "]=nullptr").c_str(),
                "in WeightNz single-multi-single mode, expected every weight tensor to be non-null"));
    }
    if (gmmParams_.scaleOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const size_t scaleTensorNum = GetInputTensorSize(gmmParams_.scaleOptional);
    for (size_t i = 0UL; i < scaleTensorNum; ++i) {
        GMM_CHECK_REPORT(GetInputTensor(gmmParams_.scaleOptional, i) != nullptr,
                         OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                             GetAclnnOpName(), scaleName_.c_str(), ("scale[" + std::to_string(i) + "]=nullptr").c_str(),
                             "in WeightNz single-multi-single mode, expected every scale tensor to be non-null"));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckWeightNzTensorShapes() const
{
    const size_t weightTensorNum = GetInputTensorSize(gmmParams_.weight);
    if (IsWeightNzMultiTensorLayout()) {
        const size_t groupNum = static_cast<size_t>(gmmParams_.groupTensorOptional->GetViewShape().GetDim(0));
        GMM_CHECK_REPORT(
            weightTensorNum == groupNum,
            OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
                GetAclnnOpName(), "weight and groupTensor",
                (std::string("weight=") + std::to_string(weightTensorNum) + ", groupTensor=" + std::to_string(groupNum))
                    .c_str(),
                "the number of weight tensors must equal the number of groups"));
    }
    const auto *firstWeightTensor = GetInputTensor(gmmParams_.weight);
    GMM_CHECK_REPORT(firstWeightTensor != nullptr,
                     OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnOpName(), weightName_.c_str(), "nullptr",
                                                           "each weight tensor must be non-null"));
    int64_t firstKDimValue = 0L;
    int64_t firstNDimValue = 0L;
    for (size_t i = 0; i < weightTensorNum; ++i) {
        const auto *weightTensor = GetInputTensor(gmmParams_.weight, i);
        CHECK_RET(CheckWeightNzTensorShape(weightTensor, firstWeightTensor, i, firstKDimValue, firstNDimValue) ==
                      ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckWeightNzTensorShape(const aclTensor *weightTensor,
                                                                          const aclTensor *firstWeightTensor,
                                                                          size_t index, int64_t &firstKDimValue,
                                                                          int64_t &firstNDimValue) const
{
    GMM_CHECK_REPORT(weightTensor != nullptr,
                     OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnOpName(), weightName_.c_str(), "nullptr",
                                                           "each weight tensor must be non-null"));
    const size_t dimNum = weightTensor->GetViewShape().GetDimNum();
    GMM_CHECK_REPORT(dimNum >= MIN_FM_DIM,
                     OP_LOGE_FOR_INVALID_SHAPEDIM(GetAclnnOpName(), weightName_.c_str(), std::to_string(dimNum),
                                                  std::to_string(MIN_FM_DIM)));
    const int64_t kDimValue = weightTensor->GetViewShape().GetDim(dimNum - LAST_SECOND_DIM_INDEX);
    const int64_t nDimValue = weightTensor->GetViewShape().GetDim(dimNum - LAST_FIRST_DIM_INDEX);
    GMM_CHECK_REPORT(
        kDimValue != 1L && nDimValue != 1L,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetAclnnOpName(), weightName_.c_str(), ViewShapeToString(weightTensor),
                                              "the logical K and N dimensions of an NZ weight must not be 1, but the "
                                              "current values are K=" +
                                                  std::to_string(kDimValue) + ", N=" + std::to_string(nDimValue)));
    if (index == 0UL) {
        firstKDimValue = kDimValue;
        firstNDimValue = nDimValue;
    } else if (IsWeightNzMultiTensorLayout()) {
        GMM_CHECK_REPORT(
            kDimValue == firstKDimValue && nDimValue == firstNDimValue,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                GetAclnnOpName(), "weight[0] and weight[i]",
                ViewShapeToString(firstWeightTensor) + ", " + ViewShapeToString(weightTensor),
                "in WeightNz single-multi-single mode, current weight[" + std::to_string(index) + "] logical K/N is [" +
                    std::to_string(kDimValue) + ", " + std::to_string(nDimValue) + "], expected [" +
                    std::to_string(firstKDimValue) + ", " + std::to_string(firstNDimValue) + "] from weight[0]"));
    }
    return CheckWeightStorageShape(weightTensor, kDimValue, nDimValue);
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulMxDtype() const
{
    if (gmmParams_.biasOptional != nullptr) {
        DataType biasDtype = GetInputTensor(gmmParams_.biasOptional)->GetDataType();
        GMM_CHECK_REPORT(biasDtype == DataType::DT_FLOAT,
                         OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), biasName_.c_str(),
                                                   op::ToString(biasDtype).GetString(), "FLOAT32"));
    }
    DataType perTokenDtype = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetDataType();
    GMM_CHECK_REPORT(perTokenDtype == DataType::DT_FLOAT8_E8M0,
                     OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), perTokenScaleName_.c_str(),
                                               op::ToString(perTokenDtype).GetString(), "FLOAT8_E8M0"));

    DataType yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
    GMM_CHECK_REPORT(yDtype == DataType::DT_FLOAT16 || yDtype == DataType::DT_BF16 || yDtype == DataType::DT_FLOAT,
                     OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), yName_.c_str(), op::ToString(yDtype).GetString(),
                                               "FLOAT16, BFLOAT16 or FLOAT32"));
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulPerGroupDim() const
{
    const auto tensorIndexInfo = GetTensorIndexInfo();
    for (size_t i = 0; i < tensorIndexInfo.loopSize; i++) {
        const auto tensorIndex = GetTensorIndexInfo(i);
        auto xDimNumber = GetInputTensor(gmmParams_.x, tensorIndex.x)->GetViewShape().GetDimNum();
        auto weightDimNumber = GetInputTensor(gmmParams_.weight, tensorIndex.weight)->GetViewShape().GetDimNum();
        auto scaleDimNumber = GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)->GetViewShape().GetDimNum();
        auto perTokenDimNumber =
            GetInputTensor(gmmParams_.perTokenScaleOptional, tensorIndex.perTokenScale)->GetViewShape().GetDimNum();
        GMM_CHECK_REPORT(xDimNumber == MIN_FM_DIM,
                         OP_LOGE_FOR_INVALID_SHAPEDIM(GetAclnnOpName(), xName_.c_str(), std::to_string(xDimNumber),
                                                      std::to_string(MIN_FM_DIM)));
        if (gmmParams_.groupType == SPLIT_M) {
            const size_t expectedWeightDim = IsWeightNzMultiTensorLayout() ? MIN_FM_DIM : SPLIT_M_SINGLE_WEIGHT_DIM;
            GMM_CHECK_REPORT(
                weightDimNumber == expectedWeightDim,
                OP_LOGE_FOR_INVALID_SHAPEDIM(GetAclnnOpName(), weightName_.c_str(), std::to_string(weightDimNumber),
                                             std::to_string(expectedWeightDim)));
        } else if (gmmParams_.groupType == SPLIT_K) {
            GMM_CHECK_REPORT(
                weightDimNumber == SPLIT_K_SINGLE_WEIGHT_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(GetAclnnOpName(), weightName_.c_str(), std::to_string(weightDimNumber),
                                             std::to_string(SPLIT_K_SINGLE_WEIGHT_DIM)));
        }
        if (gmmParams_.groupType == SPLIT_M) {
            DataType scaleDtype = GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)->GetDataType();
            if (scaleDtype == DataType::DT_FLOAT8_E8M0) {
                const size_t expectedScaleDim =
                    IsWeightNzMultiTensorLayout() ? MX_SPLIT_K_SCALE_DIM : MX_SPLIT_M_SCALE_DIM;
                GMM_CHECK_REPORT(
                    scaleDimNumber == expectedScaleDim,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(GetAclnnOpName(), scaleName_.c_str(), std::to_string(scaleDimNumber),
                                                 std::to_string(expectedScaleDim)));
                GMM_CHECK_REPORT(perTokenDimNumber == MX_SPLIT_M_PER_TOKEN_SCALE_DIM,
                                 OP_LOGE_FOR_INVALID_SHAPEDIM(GetAclnnOpName(), perTokenScaleName_.c_str(),
                                                              std::to_string(perTokenDimNumber),
                                                              std::to_string(MX_SPLIT_M_PER_TOKEN_SCALE_DIM)));
            } else {
                GMM_CHECK_REPORT(scaleDimNumber == weightDimNumber,
                                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                                     GetAclnnOpName(), scaleName_.c_str(), std::to_string(scaleDimNumber),
                                     "when split m and in G-B quant mode, the shape dim of scale must be equal to the "
                                     "shape dim of weight [" +
                                         std::to_string(weightDimNumber) + "]"));
                GMM_CHECK_REPORT(
                    perTokenDimNumber == xDimNumber,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        GetAclnnOpName(), perTokenScaleName_.c_str(), std::to_string(perTokenDimNumber),
                        "when split m and in G-B quant mode, the shape dim of perTokenScale must be equal to "
                        "the shape dim of x [" +
                            std::to_string(xDimNumber) + "]"));
            }
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckMxBiasInputShape(const TensorDimInfo &dimInfo, size_t index) const
{
    const auto tensorIndex = GetTensorIndexInfo(index);
    auto weightNIndex = GetInputTensor(gmmParams_.weight, tensorIndex.weight)->GetViewShape().GetDimNum() - 1;
    size_t biasDimNum = dimInfo.biasDimNum;
    int64_t groupNum = dimInfo.groupNum;
    if (biasDimNum != 0) {
        GMM_CHECK_REPORT(biasDimNum == MX_BIAS_DIM,
                         OP_LOGE_FOR_INVALID_SHAPEDIM(GetAclnnOpName(), biasName_.c_str(), std::to_string(biasDimNum),
                                                      std::to_string(MX_BIAS_DIM)));
    }
    auto weightNDimValue = GetInputTensor(gmmParams_.weight, tensorIndex.weight)->GetViewShape().GetDim(weightNIndex);
    if (gmmParams_.biasOptional != nullptr) {
        const auto *biasTensor = GetInputTensor(gmmParams_.biasOptional, tensorIndex.bias);
        auto biasGDimValue = GetInputTensor(gmmParams_.biasOptional, tensorIndex.bias)->GetViewShape().GetDim(0);
        auto biasNDimValue = GetInputTensor(gmmParams_.biasOptional, tensorIndex.bias)->GetViewShape().GetDim(1);
        GMM_CHECK_REPORT(biasGDimValue == groupNum,
                         OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                             GetAclnnOpName(), biasName_.c_str(), ViewShapeToString(biasTensor),
                             "axis group of bias must be equal to axis 0 of groupTensor [" + std::to_string(groupNum) +
                                 "], but the current value is " + std::to_string(biasGDimValue)));
        GMM_CHECK_REPORT(biasNDimValue == weightNDimValue,
                         OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                             GetAclnnOpName(), biasName_.c_str(), ViewShapeToString(biasTensor),
                             "axis N of bias must be equal to axis N of weight [" + std::to_string(weightNDimValue) +
                                 "], but the current value is " + std::to_string(biasNDimValue)));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckMxTypeMCaseInputShape(const TensorDimInfo &dimInfo,
                                                                            size_t index) const
{
    const auto tensorIndex = GetTensorIndexInfo(index);
    auto weightNIndex = GetInputTensor(gmmParams_.weight, tensorIndex.weight)->GetViewShape().GetDimNum() - 1;
    size_t scaleDimNum = dimInfo.scaleDimNum;
    size_t pertokenScaleDimNum = dimInfo.pertokenScaleDimNum;
    int64_t groupNum = dimInfo.groupNum;
    auto xMDimValue = GetInputTensor(gmmParams_.x, tensorIndex.x)->GetViewShape().GetDim(0);
    auto xKDimValue = GetInputTensor(gmmParams_.x, tensorIndex.x)->GetViewShape().GetDim(1);
    auto pertokenMDimValue =
        GetInputTensor(gmmParams_.perTokenScaleOptional, tensorIndex.perTokenScale)->GetViewShape().GetDim(0);
    auto pertokenScaleKDimValue =
        GetInputTensor(gmmParams_.perTokenScaleOptional, tensorIndex.perTokenScale)->GetViewShape().GetDim(1);
    auto pertokenScaleLastDimValue = GetInputTensor(gmmParams_.perTokenScaleOptional, tensorIndex.perTokenScale)
                                         ->GetViewShape()
                                         .GetDim(pertokenScaleDimNum - 1);
    auto weightNDimValue = GetInputTensor(gmmParams_.weight, tensorIndex.weight)->GetViewShape().GetDim(weightNIndex);
    auto inferedScaleKDimValue = (xKDimValue + MXFP_DIVISOR_SIZE - 1) / MXFP_DIVISOR_SIZE;
    auto scaleLastDimValue =
        GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)->GetViewShape().GetDim(scaleDimNum - 1);
    CHECK_RET(CheckMxBiasInputShape(dimInfo, index) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    GMM_CHECK_REPORT(
        xMDimValue == pertokenMDimValue,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            GetAclnnOpName(), xName_.c_str(), ViewShapeToString(GetInputTensor(gmmParams_.x, tensorIndex.x)),
            "axis M of x must be equal to axis M of perTokenScale [" + std::to_string(pertokenMDimValue) +
                "], but the current value is " + std::to_string(xMDimValue)));
    if (IsWeightNzMultiTensorLayout()) {
        auto scaleFirstDimValue = GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)->GetViewShape().GetDim(0);
        auto scaleSecondDimValue =
            GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)->GetViewShape().GetDim(1);
        const bool isNkLayout = scaleFirstDimValue == weightNDimValue && scaleSecondDimValue == inferedScaleKDimValue;
        const bool isKnLayout = scaleFirstDimValue == inferedScaleKDimValue && scaleSecondDimValue == weightNDimValue;
        const bool isBroadcastLayout = scaleFirstDimValue == 1L || scaleSecondDimValue == 1L;
        GMM_CHECK_REPORT(
            isNkLayout || isKnLayout || isBroadcastLayout,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                GetAclnnOpName(), scaleName_.c_str(),
                ViewShapeToString(GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)),
                "the first two axes of scale must match axis N of weight and ceil(k/" +
                    std::to_string(MXFP_DIVISOR_SIZE) + ") [" + std::to_string(weightNDimValue) + ", " +
                    std::to_string(inferedScaleKDimValue) + "], but the current values are [" +
                    std::to_string(scaleFirstDimValue) + ", " + std::to_string(scaleSecondDimValue) + "]"));
    } else {
        auto scaleNDimValue =
            GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)->GetViewShape().GetDim(weightNIndex);
        auto scaleKDimValue = GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)->GetViewShape().GetDim(1);
        GMM_CHECK_REPORT(weightNDimValue == scaleNDimValue,
                         OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                             GetAclnnOpName(), weightName_.c_str(),
                             ViewShapeToString(GetInputTensor(gmmParams_.weight, tensorIndex.weight)),
                             "axis N of weight must be equal to axis N of scale [" + std::to_string(scaleNDimValue) +
                                 "], but the current value is " + std::to_string(weightNDimValue)));
        auto scaleGDimValue = GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)->GetViewShape().GetDim(0);
        GMM_CHECK_REPORT(scaleGDimValue == groupNum,
                         OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                             GetAclnnOpName(), scaleName_.c_str(),
                             ViewShapeToString(GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)),
                             "axis group of scale must be equal to axis 0 of groupTensor [" + std::to_string(groupNum) +
                                 "], but the current value is " + std::to_string(scaleGDimValue)));
        GMM_CHECK_REPORT(scaleKDimValue == inferedScaleKDimValue,
                         OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                             GetAclnnOpName(), scaleName_.c_str(),
                             ViewShapeToString(GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)),
                             "axis K of scale must be equal to ceil(k/" + std::to_string(MXFP_DIVISOR_SIZE) + ") [" +
                                 std::to_string(inferedScaleKDimValue) + "], but the current value is " +
                                 std::to_string(scaleKDimValue)));
    }
    GMM_CHECK_REPORT(pertokenScaleKDimValue == inferedScaleKDimValue,
                     OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                         GetAclnnOpName(), perTokenScaleName_.c_str(),
                         ViewShapeToString(GetInputTensor(gmmParams_.perTokenScaleOptional, tensorIndex.perTokenScale)),
                         "axis K of perTokenScale must be equal to ceil(k/" + std::to_string(MXFP_DIVISOR_SIZE) +
                             ") [" + std::to_string(inferedScaleKDimValue) + "], but the current value is " +
                             std::to_string(pertokenScaleKDimValue)));
    GMM_CHECK_REPORT(pertokenScaleLastDimValue == MXFP_MULTI_BASE_SIZE,
                     OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                         GetAclnnOpName(), perTokenScaleName_.c_str(),
                         ViewShapeToString(GetInputTensor(gmmParams_.perTokenScaleOptional, tensorIndex.perTokenScale)),
                         "current perTokenScale last axis is " + std::to_string(pertokenScaleLastDimValue) +
                             ", expected " + std::to_string(MXFP_MULTI_BASE_SIZE)));
    GMM_CHECK_REPORT(scaleLastDimValue == MXFP_MULTI_BASE_SIZE,
                     OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                         GetAclnnOpName(), scaleName_.c_str(),
                         ViewShapeToString(GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)),
                         "current scale last axis is " + std::to_string(scaleLastDimValue) + ", expected " +
                             std::to_string(MXFP_MULTI_BASE_SIZE)));
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckMxFp8TypeKCaseInputShape(const TensorDimInfo &dimInfo,
                                                                               size_t index) const
{
    const auto tensorIndex = GetTensorIndexInfo(index);
    size_t xDimNum = dimInfo.xDimNum;
    size_t weightDimNum = dimInfo.weightDimNum;
    size_t scaleDimNum = dimInfo.scaleDimNum;
    size_t pertokenScaleDimNum = dimInfo.pertokenScaleDimNum;
    int64_t groupNum = dimInfo.groupNum;
    // split k, x is (m,k), weight is (k,n), scale is (k//64+g, n, 2), pertoken is (m, k//64+g, 2)
    GMM_CHECK_REPORT(xDimNum == MX_SPLIT_K_SINGLE_X_DIM,
                     OP_LOGE_FOR_INVALID_SHAPEDIM(GetAclnnOpName(), xName_.c_str(), std::to_string(xDimNum),
                                                  std::to_string(MX_SPLIT_K_SINGLE_X_DIM)));
    GMM_CHECK_REPORT(weightDimNum == MX_SPLIT_K_SINGLE_WEIGHT_DIM,
                     OP_LOGE_FOR_INVALID_SHAPEDIM(GetAclnnOpName(), weightName_.c_str(), std::to_string(weightDimNum),
                                                  std::to_string(MX_SPLIT_K_SINGLE_WEIGHT_DIM)));
    GMM_CHECK_REPORT(scaleDimNum == MX_SPLIT_K_SCALE_DIM,
                     OP_LOGE_FOR_INVALID_SHAPEDIM(GetAclnnOpName(), scaleName_.c_str(), std::to_string(scaleDimNum),
                                                  std::to_string(MX_SPLIT_K_SCALE_DIM)));
    GMM_CHECK_REPORT(
        pertokenScaleDimNum == MX_SPLIT_K_PER_TOKEN_SCALE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(GetAclnnOpName(), perTokenScaleName_.c_str(), std::to_string(pertokenScaleDimNum),
                                     std::to_string(MX_SPLIT_K_PER_TOKEN_SCALE_DIM)));
    auto xMDimValue = GetInputTensor(gmmParams_.x, tensorIndex.x)->GetViewShape().GetDim(0);
    auto xKDimValue = GetInputTensor(gmmParams_.x, tensorIndex.x)->GetViewShape().GetDim(1);
    auto weightNDimValue = GetInputTensor(gmmParams_.weight, tensorIndex.weight)->GetViewShape().GetDim(1);
    auto pertokenMDimValue =
        GetInputTensor(gmmParams_.perTokenScaleOptional, tensorIndex.perTokenScale)->GetViewShape().GetDim(0);
    auto pertokenKDimValue =
        GetInputTensor(gmmParams_.perTokenScaleOptional, tensorIndex.perTokenScale)->GetViewShape().GetDim(1);
    auto pertokenLastDimValue = GetInputTensor(gmmParams_.perTokenScaleOptional, tensorIndex.perTokenScale)
                                    ->GetViewShape()
                                    .GetDim(MX_SPLIT_K_PER_TOKEN_SCALE_DIM - 1);
    auto scaleKDimValue = GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)->GetViewShape().GetDim(0);
    auto scaleNDimValue = GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)->GetViewShape().GetDim(1);
    auto scaleLastDimValue =
        GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)->GetViewShape().GetDim(MX_SPLIT_K_SCALE_DIM - 1);
    GMM_CHECK_REPORT(
        xMDimValue == pertokenMDimValue,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            GetAclnnOpName(), xName_.c_str(), ViewShapeToString(GetInputTensor(gmmParams_.x, tensorIndex.x)),
            "axis M of x must be equal to axis M of perTokenScale [" + std::to_string(pertokenMDimValue) +
                "], but the current value is " + std::to_string(xMDimValue)));
    GMM_CHECK_REPORT(weightNDimValue == scaleNDimValue,
                     OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                         GetAclnnOpName(), weightName_.c_str(),
                         ViewShapeToString(GetInputTensor(gmmParams_.weight, tensorIndex.weight)),
                         "axis N of weight must be equal to axis N of scale [" + std::to_string(scaleNDimValue) +
                             "], but the current value is " + std::to_string(weightNDimValue)));
    GMM_CHECK_REPORT(pertokenKDimValue == (xKDimValue / MXFP_DIVISOR_SIZE + groupNum),
                     OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                         GetAclnnOpName(), perTokenScaleName_.c_str(),
                         ViewShapeToString(GetInputTensor(gmmParams_.perTokenScaleOptional, tensorIndex.perTokenScale)),
                         "axis K of perTokenScale must be equal to axis K of x divided by 64, plus groupSize [" +
                             std::to_string(xKDimValue / MXFP_DIVISOR_SIZE + groupNum) +
                             "], but the current value is " + std::to_string(pertokenKDimValue)));
    GMM_CHECK_REPORT(scaleKDimValue == (xKDimValue / MXFP_DIVISOR_SIZE + groupNum),
                     OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                         GetAclnnOpName(), scaleName_.c_str(),
                         ViewShapeToString(GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)),
                         "axis K of scale must be equal to axis K of x divided by 64, plus groupSize [" +
                             std::to_string(xKDimValue / MXFP_DIVISOR_SIZE + groupNum) +
                             "], but the current value is " + std::to_string(scaleKDimValue)));
    GMM_CHECK_REPORT(scaleLastDimValue == MXFP_MULTI_BASE_SIZE,
                     OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                         GetAclnnOpName(), scaleName_.c_str(),
                         ViewShapeToString(GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)),
                         "when split K in MX quant mode, current scale last axis is " +
                             std::to_string(scaleLastDimValue) + ", expected " + std::to_string(MXFP_MULTI_BASE_SIZE)));
    GMM_CHECK_REPORT(
        pertokenLastDimValue == MXFP_MULTI_BASE_SIZE,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            GetAclnnOpName(), perTokenScaleName_.c_str(),
            ViewShapeToString(GetInputTensor(gmmParams_.perTokenScaleOptional, tensorIndex.perTokenScale)),
            "when split K in MX quant mode, current perTokenScale last axis is " +
                std::to_string(pertokenLastDimValue) + ", expected " + std::to_string(MXFP_MULTI_BASE_SIZE)));
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulMxShape() const
{
    const auto tensorIndexInfo = GetTensorIndexInfo();
    for (size_t i = 0; i < tensorIndexInfo.loopSize; i++) {
        const auto tensorIndex = GetTensorIndexInfo(i);
        auto xDimNum = GetInputTensor(gmmParams_.x, tensorIndex.x)->GetViewShape().GetDimNum();
        auto weightDimNum = GetInputTensor(gmmParams_.weight, tensorIndex.weight)->GetViewShape().GetDimNum();
        auto scaleDimNum = GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale)->GetViewShape().GetDimNum();
        auto pertokenScaleDimNum =
            GetInputTensor(gmmParams_.perTokenScaleOptional, tensorIndex.perTokenScale)->GetViewShape().GetDimNum();
        auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
        size_t biasDimNum = 0;
        if (gmmParams_.biasOptional != nullptr) {
            biasDimNum = GetInputTensor(gmmParams_.biasOptional, tensorIndex.bias)->GetViewShape().GetDimNum();
        }
        const TensorDimInfo dimInfo = {xDimNum, weightDimNum, scaleDimNum, pertokenScaleDimNum, groupNum, biasDimNum};
        if (gmmParams_.groupType == SPLIT_M) {
            CHECK_RET(CheckMxTypeMCaseInputShape(dimInfo, i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        } else {
            CHECK_RET(CheckMxFp8TypeKCaseInputShape(dimInfo, i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
bool AclnnGroupedMatmulDAV3510Checker<T>::IsSpecialMXCase(const T *tensorList) const
{
    // 已校验mx场景scale的shape大于或等于3维度，不存在越界取值问题
    // mx特殊场景 (m,k,2) -> shape(1,1,2), stride(2,2,1); (k,m,2) -> shape(1,1,2), stride(2,2,1),
    // 无法通过stride识别转置
    for (size_t i = 0; i < GetInputTensorSize(tensorList); i++) {
        auto tensorDimNum = GetInputTensor(tensorList, i)->GetViewShape().GetDimNum();
        auto secondLastDimValue =
            GetInputTensor(tensorList, i)->GetViewShape().GetDim(tensorDimNum - LAST_SECOND_DIM_INDEX);
        auto thirdLastDimValue =
            GetInputTensor(tensorList, i)->GetViewShape().GetDim(tensorDimNum - LAST_THIRD_DIM_INDEX);
        if (secondLastDimValue == 1 && thirdLastDimValue == 1) {
            return true;
        }
    }
    return false;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulMxScaleTranspose() const
{
    const auto tensorIndexInfo = GetTensorIndexInfo();
    for (size_t i = 0; i < tensorIndexInfo.loopSize; ++i) {
        const auto tensorIndex = GetTensorIndexInfo(i);
        const auto *scaleTensor = GetInputTensor(gmmParams_.scaleOptional, tensorIndex.scale);
        const auto *weightTensor = GetInputTensor(gmmParams_.weight, tensorIndex.weight);
        GMM_CHECK_REPORT(scaleTensor != nullptr && weightTensor != nullptr,
                         OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnOpName(), "scale or weight", "nullptr",
                                                               "each scale and weight tensor must be non-null"));
        bool transposeScale = IsTransposeForMxShape(scaleTensor);
        if (!IsSpecialMXCase(gmmParams_.scaleOptional)) {
            GMM_CHECK_REPORT(transposeScale == gmmParams_.transposeWeight,
                             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                                 GetAclnnOpName(), "scale and weight",
                                 scaleName_ + "=" + ViewShapeToString(scaleTensor) + ", " + weightName_ + "=" +
                                     ViewShapeToString(weightTensor),
                                 "in mx case, the values of scale and weight transposition must be the same"));
        }
    }
    const auto tensorIndex = GetTensorIndexInfo();
    const auto *perTokenScaleTensor = GetInputTensor(gmmParams_.perTokenScaleOptional, tensorIndex.perTokenScale);
    const auto *xTensor = GetInputTensor(gmmParams_.x, tensorIndex.x);
    GMM_CHECK_REPORT(perTokenScaleTensor != nullptr && xTensor != nullptr,
                     OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnOpName(), "perTokenScale or x", "nullptr",
                                                           "each perTokenScale and x tensor must be non-null"));
    bool transposePerTokenScale = IsTransposeForMxShape(perTokenScaleTensor);
    if (!IsSpecialMXCase(gmmParams_.perTokenScaleOptional)) {
        GMM_CHECK_REPORT(transposePerTokenScale == gmmParams_.transposeX,
                         OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                             GetAclnnOpName(), "perTokenScale and x",
                             perTokenScaleName_ + "=" + ViewShapeToString(perTokenScaleTensor) + ", " + xName_ + "=" +
                                 ViewShapeToString(xTensor),
                             "in mx case, the values of perTokenScale and x transposition must be the same"));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulMxfp8() const
{
    GMM_CHECK_REPORT(gmmParams_.biasOptional == nullptr,
                     OP_LOGE_FOR_INVALID_VALUE(GetAclnnOpName(), biasName_.c_str(), "nonnull", "nullptr"));
    GMM_CHECK_REPORT(gmmParams_.perTokenScaleOptional != nullptr,
                     OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnOpName(), perTokenScaleName_.c_str(), "nullptr",
                                                           "in mx case, the value of perTokenScale cannot be nullptr"));
    CHECK_RET(CheckGroupedMatmulMxDtype() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (gmmParams_.groupType == SPLIT_M) {
        GMM_CHECK_REPORT(
            !gmmParams_.transposeX,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                GetAclnnOpName(), xName_.c_str(), ViewShapeToString(GetInputTensor(gmmParams_.x)),
                "when groupType is 0 (split m) and in mx case, the value of x transposition must be false"));
    } else if (gmmParams_.groupType == SPLIT_K) {
        GMM_CHECK_REPORT(
            !gmmParams_.transposeWeight && gmmParams_.transposeX,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                GetAclnnOpName(), "x and weight",
                xName_ + "=" + ViewShapeToString(GetInputTensor(gmmParams_.x)) + ", " + weightName_ + "=" +
                    ViewShapeToString(GetInputTensor(gmmParams_.weight)),
                "when groupType is 2 (split K) and in mx case, the value of x and weight transposition must "
                "be true/false"));
    }
    CHECK_RET(CheckGroupedMatmulMxScaleTranspose() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGroupedMatmulPerGroupDim() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGroupedMatmulMxShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulMxfp4() const
{
    GMM_CHECK_REPORT(gmmParams_.perTokenScaleOptional != nullptr,
                     OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnOpName(), perTokenScaleName_.c_str(), "nullptr",
                                                           "in mx case, the value of perTokenScale cannot be nullptr"));
    CHECK_RET(CheckGroupedMatmulMxDtype() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    GMM_CHECK_REPORT(!gmmParams_.transposeX,
                     OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                         GetAclnnOpName(), xName_.c_str(), ViewShapeToString(GetInputTensor(gmmParams_.x)),
                         "when groupType is 0 (split m) and in mxfp4, the value of x transposition must be false"));
    CHECK_RET(CheckGroupedMatmulMxScaleTranspose() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGroupedMatmulPerGroupDim() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGroupedMatmulMxShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGroupedMatmulFp4MxDimValue() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulFp4MxDimValue() const
{
    const auto tensorIndexInfo = GetTensorIndexInfo();
    for (size_t i = 0; i < tensorIndexInfo.loopSize; i++) {
        const auto tensorIndex = GetTensorIndexInfo(i);
        const auto *xTensor = GetInputTensor(gmmParams_.x, tensorIndex.x);
        const auto *weightTensor = GetInputTensor(gmmParams_.weight, tensorIndex.weight);
        GMM_CHECK_REPORT(xTensor != nullptr && weightTensor != nullptr,
                         OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnOpName(), "x or weight", "nullptr",
                                                               "each x and weight tensor must be non-null"));
        auto weightNIndex = weightTensor->GetViewShape().GetDimNum() - 1;
        auto xKDimValue = xTensor->GetViewShape().GetDim(1);
        auto weightNDimValue = weightTensor->GetViewShape().GetDim(weightNIndex);
        auto weightKIndex = weightTensor->GetViewShape().GetDimNum() - LAST_TWO_DIM_INDEX;
        auto weightKDimValue = weightTensor->GetViewShape().GetDim(weightKIndex);
        // 2：检查N是否为偶数
        auto weightNDimModValue = weightNDimValue % 2;
        // 2：检查K是否为偶数
        auto xKDimModValue = xKDimValue % 2;
        if (!gmmParams_.transposeWeight) {
            GMM_CHECK_REPORT(weightNDimModValue == 0,
                             OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                 GetAclnnOpName(), weightName_.c_str(), ViewShapeToString(weightTensor),
                                 "when the weight is not transposed, axis N of weight must be an even number, but the "
                                 "current value is " +
                                     std::to_string(weightNDimValue)));
        }
        GMM_CHECK_REPORT(xKDimModValue == 0,
                         OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                             GetAclnnOpName(), xName_.c_str(), ViewShapeToString(xTensor),
                             "when the dtypes of x and weight inputs are fp4, axis K of x must be an even number, but "
                             "the current value is " +
                                 std::to_string(xKDimValue)));
        // 2: mxfp4场景下不支持k轴为2
        GMM_CHECK_REPORT(
            xKDimValue != 2 && weightKDimValue != 2,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                GetAclnnOpName(), "x and weight",
                xName_ + "=" + ViewShapeToString(xTensor) + ", " + weightName_ + "=" + ViewShapeToString(weightTensor),
                "when the dtypes of x and weight inputs are fp4, all axes K of x and weight must not be equal to 2, "
                "but "
                "the current values are x=" +
                    std::to_string(xKDimValue) + ", weight=" + std::to_string(weightKDimValue)));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckNonMxQuantTransposeStatus() const
{
    if (gmmParams_.groupType == SPLIT_M) {
        GMM_CHECK_REPORT(!gmmParams_.transposeX,
                         OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                             GetAclnnOpName(), xName_.c_str(), ViewShapeToString(GetInputTensor(gmmParams_.x)),
                             "in non-pergroup quantification mode and when groupType is 0 (split M), the value of x "
                             "transposition must be false"));
    } else if (gmmParams_.groupType == SPLIT_K) {
        GMM_CHECK_REPORT(
            gmmParams_.transposeX && !gmmParams_.transposeWeight,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                GetAclnnOpName(), "x and weight",
                xName_ + "=" + ViewShapeToString(GetInputTensor(gmmParams_.x)) + ", " + weightName_ + "=" +
                    ViewShapeToString(GetInputTensor(gmmParams_.weight)),
                "in non-pergroup quantification mode and when groupType is 2 (split K), the value of x and "
                "weight transposition must be true/false"));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckNonPerGroupQuantDim() const
{
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); ++i) {
        auto scaleDimNumber = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDimNum();
        // 2: (E, N)
        GMM_CHECK_REPORT(scaleDimNumber == 1 || scaleDimNumber == 2,
                         OP_LOGE_FOR_INVALID_SHAPEDIM(GetAclnnOpName(), scaleName_.c_str(),
                                                      std::to_string(scaleDimNumber), "1 or 2"));
        if (GetInputTensor(gmmParams_.y)->GetDataType() == DataType::DT_INT8) {
            // 2: (E, N)
            GMM_CHECK_REPORT(scaleDimNumber == 2, OP_LOGE_FOR_INVALID_SHAPEDIM(GetAclnnOpName(), scaleName_.c_str(),
                                                                               std::to_string(scaleDimNumber), "2"));
        }

        if (gmmParams_.perTokenScaleOptional != nullptr) {
            auto perTokenDimNumber = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDimNum();
            GMM_CHECK_REPORT(perTokenDimNumber == 1 || perTokenDimNumber == 2,
                             OP_LOGE_FOR_INVALID_SHAPEDIM(GetAclnnOpName(), perTokenScaleName_.c_str(),
                                                          std::to_string(perTokenDimNumber), "1 or 2"));
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckNonPerGroupQuantPertokenShape() const
{
    auto perTokenDimNumber = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetViewShape().GetDimNum();
    auto perTokenFirstDim = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetViewShape().GetDim(0);
    auto xMDim = GetInputTensor(gmmParams_.x)->GetViewShape().GetDim(0);
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    if (gmmParams_.groupType == SPLIT_M) {
        if (perTokenDimNumber == 1) {
            GMM_CHECK_REPORT(perTokenFirstDim == xMDim || perTokenFirstDim == groupNum,
                             OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                 GetAclnnOpName(), perTokenScaleName_.c_str(),
                                 ViewShapeToString(GetInputTensor(gmmParams_.perTokenScaleOptional)),
                                 "in non-pergroup quantification mode and when groupType is 0 (split M) and "
                                 "perTokenScale has 1 dim, axis 0 of perTokenScale must be equal to axis M of x [" +
                                     std::to_string(xMDim) + "] or axis 0 of groupTensor [" + std::to_string(groupNum) +
                                     "], but the current value is " + std::to_string(perTokenFirstDim)));
        } else {
            GMM_CHECK_REPORT(
                perTokenFirstDim == groupNum,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    GetAclnnOpName(), perTokenScaleName_.c_str(),
                    ViewShapeToString(GetInputTensor(gmmParams_.perTokenScaleOptional)),
                    "in non-pergroup quantification mode and when groupType is 0 (split M) and "
                    "perTokenScale has 2 dims, axis 0 of perTokenScale must be equal to axis 0 of "
                    "groupTensor [" +
                        std::to_string(groupNum) + "], but the current value is " + std::to_string(perTokenFirstDim)));
            auto perTokenSecondDim = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetViewShape().GetDim(1);
            GMM_CHECK_REPORT(perTokenSecondDim == 1,
                             OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                 GetAclnnOpName(), perTokenScaleName_.c_str(),
                                 ViewShapeToString(GetInputTensor(gmmParams_.perTokenScaleOptional)),
                                 "in non-pergroup quantification mode and when groupType is 0 (split M) and "
                                 "perTokenScale has 2 dims, axis 1 of perTokenScale must be equal to 1, but the "
                                 "current value is " +
                                     std::to_string(perTokenSecondDim)));
        }
    } else if (gmmParams_.groupType == SPLIT_K) {
        GMM_CHECK_REPORT(
            perTokenFirstDim == groupNum,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                GetAclnnOpName(), perTokenScaleName_.c_str(),
                ViewShapeToString(GetInputTensor(gmmParams_.perTokenScaleOptional)),
                "in non-pergroup quantification mode and when groupType is 2 (split K), axis 0 of "
                "perTokenScale must be equal to axis 0 of groupTensor [" +
                    std::to_string(groupNum) + "], but the current value is " + std::to_string(perTokenFirstDim)));
        if (perTokenDimNumber > 1) {
            auto perTokenSecondDim = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetViewShape().GetDim(1);
            GMM_CHECK_REPORT(perTokenSecondDim == xMDim || perTokenSecondDim == 1,
                             OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                 GetAclnnOpName(), perTokenScaleName_.c_str(),
                                 ViewShapeToString(GetInputTensor(gmmParams_.perTokenScaleOptional)),
                                 "in non-pergroup quantification mode and when groupType is 2 (split K) and "
                                 "perTokenScale has 2 dims, axis 1 of perTokenScale must be equal to axis M of x [" +
                                     std::to_string(xMDim) + "] or 1, but the current value is " +
                                     std::to_string(perTokenSecondDim)));
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckNonPerGroupQuantShape() const
{
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); ++i) {
        auto scaleDimNumber = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDimNum();
        auto scaleNDim = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(scaleDimNumber - 1);
        auto weightNIndex = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum() - 1;
        auto weightNDim = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightNIndex);
        auto scaleFirstDim = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(0);
        GMM_CHECK_REPORT(
            scaleFirstDim == groupNum,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                GetAclnnOpName(), scaleName_.c_str(), ViewShapeToString(GetInputTensor(gmmParams_.scaleOptional, i)),
                "in non-pergroup quantification mode, axis 0 of scale must be equal to axis 0 of groupTensor [" +
                    std::to_string(groupNum) + "], but the current value is " + std::to_string(scaleFirstDim)));
        if (scaleDimNumber > 1) {
            GMM_CHECK_REPORT(
                scaleNDim == 1 || scaleNDim == weightNDim,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    GetAclnnOpName(), scaleName_.c_str(),
                    ViewShapeToString(GetInputTensor(gmmParams_.scaleOptional, i)),
                    "in non-pergroup quantification mode, axis N of scale must be 1 or equal to axis N of weight [" +
                        std::to_string(weightNDim) + "], but the current value is " + std::to_string(scaleNDim)));
            DataType yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
            if (yDtype == DataType::DT_INT8) {
                GMM_CHECK_REPORT(
                    scaleNDim == weightNDim,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        GetAclnnOpName(), scaleName_.c_str(),
                        ViewShapeToString(GetInputTensor(gmmParams_.scaleOptional, i)),
                        "when the output dtype is int8, axis N of scale must be equal to axis N of weight [" +
                            std::to_string(weightNDim) + "], but the current value is " + std::to_string(scaleNDim)));
            }
        }

        if (gmmParams_.perTokenScaleOptional != nullptr) {
            CHECK_RET(CheckNonPerGroupQuantPertokenShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckInt8QuantDtype() const
{
    static const std::vector<DataType> legalOutputDtypes = {DataType::DT_INT8, DataType::DT_INT32, DataType::DT_BF16,
                                                            DataType::DT_FLOAT16};
    DataType yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
    GMM_CHECK_REPORT(std::find(legalOutputDtypes.begin(), legalOutputDtypes.end(), yDtype) != legalOutputDtypes.end(),
                     OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), yName_.c_str(), op::ToString(yDtype).GetString(),
                                               "INT8, INT32, FLOAT16 or BFLOAT16"));
    if (gmmParams_.biasOptional != nullptr) {
        DataType biasDtype = (*gmmParams_.biasOptional)[0]->GetDataType();
        if (yDtype == DataType::DT_BF16) {
            GMM_CHECK_REPORT(
                biasDtype == DataType::DT_INT32 || biasDtype == DataType::DT_BF16 || biasDtype == DataType::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), biasName_.c_str(), op::ToString(biasDtype).GetString(),
                                          "INT32, BFLOAT16 or FLOAT32"));
        } else if (yDtype == DataType::DT_FLOAT16) {
            GMM_CHECK_REPORT(
                biasDtype == DataType::DT_INT32 || biasDtype == DataType::DT_FLOAT16 || biasDtype == DataType::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), biasName_.c_str(), op::ToString(biasDtype).GetString(),
                                          "INT32, FLOAT16 or FLOAT32"));
        } else if (yDtype == DataType::DT_INT8 || yDtype == DataType::DT_INT32) {
            GMM_CHECK_REPORT(biasDtype == DataType::DT_INT32,
                             OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), biasName_.c_str(),
                                                       op::ToString(biasDtype).GetString(), "INT32"));
        }
    }
    if (yDtype == DataType::DT_INT32 && gmmParams_.scaleOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    DataType scaleDtype = GetInputTensor(gmmParams_.scaleOptional)->GetDataType();
    if (yDtype == DataType::DT_BF16) {
        GMM_CHECK_REPORT(
            scaleDtype == DataType::DT_BF16 || scaleDtype == DataType::DT_UINT64 || scaleDtype == DataType::DT_INT64 ||
                scaleDtype == DataType::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), scaleName_.c_str(), op::ToString(scaleDtype).GetString(),
                                      "BFLOAT16, UINT64, INT64 or FLOAT32"));
    } else if (yDtype == DataType::DT_FLOAT16) {
        GMM_CHECK_REPORT(
            scaleDtype == DataType::DT_UINT64 || scaleDtype == DataType::DT_INT64 || scaleDtype == DataType::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), scaleName_.c_str(), op::ToString(scaleDtype).GetString(),
                                      "UINT64, INT64 or FLOAT32"));
    } else if (yDtype == DataType::DT_INT8) {
        GMM_CHECK_REPORT(scaleDtype == DataType::DT_UINT64 || scaleDtype == DataType::DT_INT64,
                         OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), scaleName_.c_str(),
                                                   op::ToString(scaleDtype).GetString(), "UINT64 or INT64"));
    } else if (yDtype == DataType::DT_INT32) {
        GMM_CHECK_REPORT(scaleDtype == DataType::DT_UINT64 || scaleDtype == DataType::DT_INT64,
                         OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), scaleName_.c_str(),
                                                   op::ToString(scaleDtype).GetString(), "UINT64 or INT64"));
    }
    if (gmmParams_.perTokenScaleOptional != nullptr) {
        GMM_CHECK_REPORT(
            yDtype != DataType::DT_INT8,
            OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), yName_.c_str(), op::ToString(yDtype).GetString(), "not INT8"));
        DataType perTokenScaleDtype = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetDataType();
        GMM_CHECK_REPORT(perTokenScaleDtype == DataType::DT_FLOAT,
                         OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), perTokenScaleName_.c_str(),
                                                   op::ToString(perTokenScaleDtype).GetString(), "FLOAT32"));
        if (yDtype == DataType::DT_BF16) {
            GMM_CHECK_REPORT(scaleDtype == DataType::DT_BF16 || scaleDtype == DataType::DT_FLOAT,
                             OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), scaleName_.c_str(),
                                                       op::ToString(scaleDtype).GetString(), "BFLOAT16 or FLOAT32"));
        } else if (yDtype == DataType::DT_FLOAT16) {
            GMM_CHECK_REPORT(scaleDtype == DataType::DT_FLOAT,
                             OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), scaleName_.c_str(),
                                                       op::ToString(scaleDtype).GetString(), "FLOAT32"));
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckInt8QuantParams() const
{
    GMM_CHECK_REPORT(
        gmmParams_.groupType == SPLIT_M,
        OP_LOGE_FOR_INVALID_VALUE(GetAclnnOpName(), "groupType", std::to_string(gmmParams_.groupType), "0"));
    CHECK_RET(CheckInt8QuantDtype() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNonMxQuantTransposeStatus() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (GetInputTensor(gmmParams_.y)->GetDataType() == DataType::DT_INT32) {
        return ACLNN_SUCCESS;
    }
    CHECK_RET(CheckNonPerGroupQuantDim() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNonPerGroupQuantShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (gmmParams_.perTokenScaleOptional != nullptr) {
        auto perTokenDimNum = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetViewShape().GetDimNum();
        GMM_CHECK_REPORT(perTokenDimNum == 1, OP_LOGE_FOR_INVALID_SHAPEDIM(GetAclnnOpName(), perTokenScaleName_.c_str(),
                                                                           std::to_string(perTokenDimNum), "1"));
        auto perTokenMDim = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetViewShape().GetDim(0);
        auto xMDim = GetInputTensor(gmmParams_.x)->GetViewShape().GetDim(0);
        GMM_CHECK_REPORT(
            perTokenMDim == xMDim,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                GetAclnnOpName(), perTokenScaleName_.c_str(),
                ViewShapeToString(GetInputTensor(gmmParams_.perTokenScaleOptional)),
                "in int8 quant case, axis M of perTokenScale must be equal to axis M of x [" + std::to_string(xMDim) +
                    "], but the current value is " + std::to_string(perTokenMDim)));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
bool AclnnGroupedMatmulDAV3510Checker<T>::IsSpecialperTileScene(int64_t groupNum, int64_t weightNDim,
                                                                int64_t weightKDim, int64_t xMDim,
                                                                int64_t perTokenMDim) const
{
    return groupNum > 1L && gmmParams_.groupType == SPLIT_K && weightKDim < PERTILE_GROUP_SIZE &&
           weightNDim <= PERTILE_GROUP_SIZE && xMDim > 1L && xMDim == perTokenMDim;
}

template <typename T>
bool AclnnGroupedMatmulDAV3510Checker<T>::IsPerTileQuantMode() const
{
    if (CheckInputTensorsNotNull() != ACLNN_SUCCESS) {
        return false;
    }
    bool isPerTileQuantMode = false;
    if (gmmParams_.perTokenScaleOptional == nullptr) {
        return isPerTileQuantMode;
    }
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.weight); ++i) {
        auto scaleDimNumber = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDimNum();
        auto weightDimNumber = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum();
        // 2 is the minimum dimension for scale in perTile case
        if (scaleDimNumber < 2 || weightDimNumber != scaleDimNumber) {
            return false;
        }
        auto xDimNumber = GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDimNum();
        auto perTokenDimNumber = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDimNum();
        // 2 is the minimum dimension for perTokenScaleOptional in perTile case
        if (perTokenDimNumber < 2 || xDimNumber != perTokenDimNumber) {
            return false;
        }
        bool transposePerTokenScale = IsTransposeLastTwoDims(GetInputTensor(gmmParams_.perTokenScaleOptional, i));
        auto scaleNDim = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(scaleDimNumber - 1);
        auto scaleKDim =
            GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(scaleDimNumber - LAST_TWO_DIM_INDEX);
        auto weightNDim = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightDimNumber - 1);
        auto weightKDim =
            GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightDimNumber - LAST_TWO_DIM_INDEX);
        auto xMDim = GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDim(0);
        auto perTokenMDim = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDim(0);
        // m轴分组通过校验weight与scale是否维度一致可区分开其他场景，
        // 这里仅通过判断k分组时x与perToken的转置是否一致区分其他场景
        if (!LastTwoDimValueIsOne(GetInputTensor(gmmParams_.perTokenScaleOptional, i)) &&
            transposePerTokenScale != gmmParams_.transposeX &&
            !IsSpecialperTileScene(groupNum, weightNDim, weightKDim, xMDim, perTokenMDim)) {
            return false;
        }
        isPerTileQuantMode = false;
        bool isKdimValid =
            ((gmmParams_.groupType == SPLIT_K && scaleKDim == (weightKDim / PERTILE_GROUP_SIZE) + groupNum) ||
             (gmmParams_.groupType == SPLIT_M &&
              scaleKDim == (weightKDim + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE));
        if (scaleNDim == (weightNDim + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE && isKdimValid) {
            isPerTileQuantMode = true;
        }
    }
    return isPerTileQuantMode;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulPerTileShape() const
{
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); i++) {
        auto xMIndex = 0;
        auto weightNIndex = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum() - 1;
        auto xKIndex = 1;
        auto weightKIndex = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum() - LAST_TWO_DIM_INDEX;
        auto weightKDim = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightKIndex);
        auto xMDim = GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDim(xMIndex);
        auto weightNDim = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightNIndex);
        auto scaleKDim = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(weightKIndex);
        auto perTokenKDim = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDim(xKIndex);
        auto scaleNDim = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(weightNIndex);
        auto perTokenMDim = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDim(xMIndex);
        GMM_CHECK_REPORT(
            xMDim == perTokenMDim,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                GetAclnnOpName(), "x and perTokenScale",
                xName_ + "=" + ViewShapeToString(GetInputTensor(gmmParams_.x, i)) + ", " + perTokenScaleName_ + "=" +
                    ViewShapeToString(GetInputTensor(gmmParams_.perTokenScaleOptional, i)),
                "when quantification mode is G-B quantification, the M value in x and perTokenScale should "
                "be consistent, but the current values are x=" +
                    std::to_string(xMDim) + ", perTokenScale=" + std::to_string(perTokenMDim)));
        GMM_CHECK_REPORT(
            scaleNDim == (weightNDim + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                GetAclnnOpName(), scaleName_.c_str(), ViewShapeToString(GetInputTensor(gmmParams_.scaleOptional, i)),
                "when quantification mode is G-B quantification, axis N of scale must be equal to axis N of "
                "weight divided by 128 [" +
                    std::to_string((weightNDim + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE) +
                    "], but the current value is " + std::to_string(scaleNDim)));
        if (gmmParams_.groupType == SPLIT_M) {
            int64_t expectScaleKValue = (weightKDim + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE;
            GMM_CHECK_REPORT(
                perTokenKDim == scaleKDim && scaleKDim == expectScaleKValue,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    GetAclnnOpName(), perTokenScaleName_.c_str(),
                    ViewShapeToString(GetInputTensor(gmmParams_.perTokenScaleOptional, i)),
                    "when quantification mode is G-B quantification and groupType is 0 (split M), axis K of "
                    "perTokenScale must be equal to axis K of scale [" +
                        std::to_string(scaleKDim) +
                        "], and its value must be equal to axis K of weight divided by 128, rounded up to the "
                        "next integer [" +
                        std::to_string(expectScaleKValue) + "], but the current values are perTokenScale=" +
                        std::to_string(perTokenKDim) + ", scale=" + std::to_string(scaleKDim)));
        } else {
            int64_t expectScaleKValue =
                (weightKDim / PERTILE_GROUP_SIZE) + gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
            GMM_CHECK_REPORT(
                perTokenKDim == scaleKDim && scaleKDim == expectScaleKValue,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    GetAclnnOpName(), perTokenScaleName_.c_str(),
                    ViewShapeToString(GetInputTensor(gmmParams_.perTokenScaleOptional, i)),
                    "when quantification mode is G-B quantification and groupType is 2 (split K), axis K of "
                    "perTokenScale must be equal to axis K of scale [" +
                        std::to_string(scaleKDim) +
                        "], and its value must be equal to axis K of weight divided by 128, plus groupSize [" +
                        std::to_string(expectScaleKValue) + "], but the current values are perTokenScale=" +
                        std::to_string(perTokenKDim) + ", scale=" + std::to_string(scaleKDim)));
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulPerTile() const
{
    CHECK_RET(CheckGroupedMatmulPerGroupDim() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    bool transposeScale = IsTransposeLastTwoDims(GetInputTensor(gmmParams_.scaleOptional));
    bool transposePerTokenScale = IsTransposeLastTwoDims(GetInputTensor(gmmParams_.perTokenScaleOptional));
    GMM_CHECK_REPORT(
        transposeScale == gmmParams_.transposeWeight || LastTwoDimValueIsOne(GetInputTensor(gmmParams_.scaleOptional)),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            GetAclnnOpName(), "scale and weight",
            scaleName_ + "=" + ViewShapeToString(GetInputTensor(gmmParams_.scaleOptional)) + ", " + weightName_ + "=" +
                ViewShapeToString(GetInputTensor(gmmParams_.weight)),
            "when quantification mode is G-B quantification, the values of scale and weight transposition "
            "must be the same"));
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.weight); ++i) {
        auto weightDimNumber = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum();
        auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
        auto weightKDim =
            GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightDimNumber - LAST_TWO_DIM_INDEX);
        auto weightNDim = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightDimNumber - 1);
        auto perTokenMDim = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDim(0);
        auto xMDim = GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDim(0);
        GMM_CHECK_REPORT(
            transposePerTokenScale == gmmParams_.transposeX ||
                LastTwoDimValueIsOne(GetInputTensor(gmmParams_.perTokenScaleOptional)) ||
                IsSpecialperTileScene(groupNum, weightNDim, weightKDim, xMDim, perTokenMDim),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                GetAclnnOpName(), "perTokenScale and x",
                perTokenScaleName_ + "=" + ViewShapeToString(GetInputTensor(gmmParams_.perTokenScaleOptional, i)) +
                    ", " + xName_ + "=" + ViewShapeToString(GetInputTensor(gmmParams_.x, i)),
                "when quantification mode is G-B quantification, the values of perTokenScale and x "
                "transposition must be the same"));
    }
    CHECK_RET(CheckGroupedMatmulPerTileShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckFp8Hif8QuantParams() const
{
    GMM_CHECK_REPORT(gmmParams_.biasOptional == nullptr,
                     OP_LOGE_FOR_INVALID_VALUE(GetAclnnOpName(), biasName_.c_str(), "nonnull", "nullptr"));
    CHECK_RET(CheckNonMxQuantTransposeStatus() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    DataType scaleDtype = GetInputTensor(gmmParams_.scaleOptional)->GetDataType();
    if (scaleDtype == DataType::DT_UINT64 || scaleDtype == DataType::DT_INT64) {
        GMM_CHECK_REPORT(
            gmmParams_.groupType == SPLIT_M,
            OP_LOGE_FOR_INVALID_VALUE(GetAclnnOpName(), "groupType", std::to_string(gmmParams_.groupType), "0"));
    } else {
        GMM_CHECK_REPORT(
            gmmParams_.groupType != SPLIT_N,
            OP_LOGE_FOR_INVALID_VALUE(GetAclnnOpName(), "groupType", std::to_string(gmmParams_.groupType), "0 or 2"));
    }

    if (gmmParams_.perTokenScaleOptional != nullptr) {
        DataType perTokenScaleDtype = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetDataType();
        GMM_CHECK_REPORT(perTokenScaleDtype == DataType::DT_FLOAT,
                         OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), perTokenScaleName_.c_str(),
                                                   op::ToString(perTokenScaleDtype).GetString(), "FLOAT32"));
        GMM_CHECK_REPORT(scaleDtype == DataType::DT_FLOAT,
                         OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), scaleName_.c_str(),
                                                   op::ToString(scaleDtype).GetString(), "FLOAT32"));
    }
    DataType yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
    GMM_CHECK_REPORT(yDtype == DataType::DT_FLOAT || yDtype == DataType::DT_BF16 || yDtype == DataType::DT_FLOAT16,
                     OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), yName_.c_str(), op::ToString(yDtype).GetString(),
                                               "FLOAT32, FLOAT16 or BFLOAT16"));
    if (IsPerTileQuantMode()) {
        CHECK_RET(CheckGroupedMatmulPerTile() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }
    CHECK_RET(CheckNonPerGroupQuantDim() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNonPerGroupQuantShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckFp8Params(const DataType &scaleDtype) const
{
    if (scaleDtype == DataType::DT_FLOAT8_E8M0) {
        return CheckGroupedMatmulMxfp8();
    } else if (scaleDtype == DataType::DT_FLOAT || scaleDtype == DataType::DT_UINT64 ||
               scaleDtype == DataType::DT_INT64) {
        return CheckFp8Hif8QuantParams();
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            GetAclnnOpName(), scaleName_.c_str(), op::ToString(scaleDtype).GetString(),
            "with float8 case, the dtype of scale must be within the range {FLOAT8_E8M0, FLOAT32, UINT64, INT64}");
        return ACLNN_ERR_PARAM_INVALID;
    }
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckFp4Params(const DataType &scaleDtype) const
{
    GMM_CHECK_REPORT(
        gmmParams_.groupType == SPLIT_M,
        OP_LOGE_FOR_INVALID_VALUE(GetAclnnOpName(), "groupType", std::to_string(gmmParams_.groupType), "0"));
    auto weightStorageFormat = GetInputTensor(gmmParams_.weight)->GetStorageFormat();
    bool isE1M2 = (gmmParams_.xDtype == DataType::DT_FLOAT4_E1M2 ||
                   GetInputTensor(gmmParams_.weight)->GetDataType() == DataType::DT_FLOAT4_E1M2);
    if (isE1M2 && weightStorageFormat == op::Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT(GetAclnnOpName(), weightName_.c_str(), "ND", "FRACTAL_NZ");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (scaleDtype == DataType::DT_FLOAT8_E8M0) {
        return CheckGroupedMatmulMxfp4();
    }
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
        GetAclnnOpName(), scaleName_.c_str(), op::ToString(scaleDtype).GetString(),
        "with float4 case, the dtype of scale must be within the range {FLOAT8_E8M0}");
    return ACLNN_ERR_PARAM_INVALID;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckInputAndOutputDtypeForV3Version() const
{
    DataType xDtype = gmmParams_.xDtype;
    DataType weightDtype = GetInputTensor(gmmParams_.weight)->GetDataType();
    DataType scaleDtype = GetInputTensor(gmmParams_.scaleOptional)->GetDataType();
    DataType yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
    std::string xWeightDtypes =
        std::string("x=") + op::ToString(xDtype).GetString() + ", weight=" + op::ToString(weightDtype).GetString();
    GMM_CHECK_REPORT(xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT8,
                     OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                         GetAclnnOpName(), "x and weight", xWeightDtypes.c_str(),
                         "in quant case, the dtypes of x and weight must be within the range INT8"));
    GMM_CHECK_REPORT(scaleDtype == DataType::DT_UINT64 || scaleDtype == DataType::DT_INT64,
                     OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), scaleName_.c_str(),
                                               op::ToString(scaleDtype).GetString(), "UINT64 or INT64"));
    GMM_CHECK_REPORT(yDtype == DataType::DT_INT8, OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), yName_.c_str(),
                                                                            op::ToString(yDtype).GetString(), "INT8"));
    if (gmmParams_.biasOptional != nullptr) {
        DataType biasDtype = (*gmmParams_.biasOptional)[0]->GetDataType();
        GMM_CHECK_REPORT(biasDtype == DataType::DT_INT32,
                         OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), biasName_.c_str(),
                                                   op::ToString(biasDtype).GetString(), "INT32"));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckInputShapeForV3Version() const
{
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    auto scaleDimNumber = GetInputTensor(gmmParams_.scaleOptional)->GetViewShape().GetDimNum();
    auto scaleNDim = GetInputTensor(gmmParams_.scaleOptional)->GetViewShape().GetDim(scaleDimNumber - 1);
    auto weightNIndex = GetInputTensor(gmmParams_.weight)->GetViewShape().GetDimNum() - 1;
    auto weightNDim = GetInputTensor(gmmParams_.weight)->GetViewShape().GetDim(weightNIndex);
    auto scaleFirstDim = GetInputTensor(gmmParams_.scaleOptional)->GetViewShape().GetDim(0);
    GMM_CHECK_REPORT(
        scaleFirstDim == groupNum,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            GetAclnnOpName(), scaleName_.c_str(), ViewShapeToString(GetInputTensor(gmmParams_.scaleOptional)),
            "in quant case, axis 0 of scale must be equal to axis 0 of groupTensor [" + std::to_string(groupNum) +
                "], but the current value is " + std::to_string(scaleFirstDim)));
    GMM_CHECK_REPORT(
        scaleNDim == weightNDim,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            GetAclnnOpName(), scaleName_.c_str(), ViewShapeToString(GetInputTensor(gmmParams_.scaleOptional)),
            "in quant case, axis N of scale must be equal to axis N of weight [" + std::to_string(weightNDim) +
                "], but the current value is " + std::to_string(scaleNDim)));
    if (gmmParams_.biasOptional != nullptr) {
        auto biasGDimValue = GetInputTensor(gmmParams_.biasOptional)->GetViewShape().GetDim(0);
        auto biasNDimValue = GetInputTensor(gmmParams_.biasOptional)->GetViewShape().GetDim(1);
        GMM_CHECK_REPORT(
            biasGDimValue == groupNum,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                GetAclnnOpName(), biasName_.c_str(), ViewShapeToString(GetInputTensor(gmmParams_.biasOptional)),
                "axis group of bias must be equal to axis 0 of groupTensor [" + std::to_string(groupNum) +
                    "], but the current value is " + std::to_string(biasGDimValue)));
        GMM_CHECK_REPORT(
            biasNDimValue == weightNDim,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                GetAclnnOpName(), biasName_.c_str(), ViewShapeToString(GetInputTensor(gmmParams_.biasOptional)),
                "axis N of bias must be equal to axis N of weight [" + std::to_string(weightNDim) +
                    "], but the current value is " + std::to_string(biasNDimValue)));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckInputParamsForV3Version() const
{
    CHECK_RET(CheckInputAndOutputDtypeForV3Version() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckInputShapeForV3Version() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulDAV3510() const
{
    const aclnnStatus inputStatus = CheckInputTensorsNotNull();
    if (inputStatus != ACLNN_SUCCESS) {
        return inputStatus;
    }
    DataType xDtype = gmmParams_.xDtype;
    DataType weightDtype = GetInputTensor(gmmParams_.weight)->GetDataType();
    DataType yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
    if (yDtype != DataType::DT_INT32) {
        GMM_CHECK_REPORT(
            gmmParams_.scaleOptional != nullptr,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                GetAclnnOpName(), scaleName_.c_str(), "nullptr",
                "in quant case and when the output dtype is not int32, the value of scale cannot be nullptr"));
    }
    GMM_CHECK_REPORT(gmmParams_.groupTensorOptional != nullptr,
                     OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnOpName(), groupTensorName_.c_str(), "nullptr",
                                                           "in quant case, the value of groupList cannot be nullptr"));
    GMM_CHECK_REPORT(gmmParams_.offsetOptional == nullptr,
                     OP_LOGE_FOR_INVALID_VALUE(GetAclnnOpName(), "offset", "nonnull", "nullptr"));
    GMM_CHECK_REPORT(
        gmmParams_.groupType != SPLIT_N,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnOpName(), "groupType", std::to_string(gmmParams_.groupType),
                                              "the value of groupType must be in 0 or 2"));
    const size_t xTensorNum = GetInputTensorSize(gmmParams_.x);
    const size_t weightTensorNum = GetInputTensorSize(gmmParams_.weight);
    const size_t yTensorNum = GetInputTensorSize(gmmParams_.y);
    std::string tensorNums = std::string("x=") + std::to_string(xTensorNum) +
                             ", weight=" + std::to_string(weightTensorNum) + ", y=" + std::to_string(yTensorNum);
    GMM_CHECK_REPORT(
        (xTensorNum == 1 && weightTensorNum == 1 && yTensorNum == 1) || IsWeightNzMultiTensorLayout(),
        OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
            GetAclnnOpName(), "x, weight and y", tensorNums.c_str(),
            "in quant case, only single-single-single or weightNz single-multi-single tensor layout is supported"));
    CHECK_RET(CheckQuantCasesFormat() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (GetInputTensor(gmmParams_.weight)->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        CHECK_RET(CheckWeightNzSpecialParams() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    CHECK_RET(CheckGeneralQuantShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (gmmParams_.apiVersion == gmm::GMMApiVersion::V3) {
        CHECK_RET(CheckInputParamsForV3Version() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    DataType scaleDtype = DataType::DT_UINT64;
    if (gmmParams_.scaleOptional != nullptr) {
        scaleDtype = GetInputTensor(gmmParams_.scaleOptional)->GetDataType();
    }
    if (xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT8) {
        return CheckInt8QuantParams();
    } else if (xDtype == DataType::DT_HIFLOAT8 && weightDtype == DataType::DT_HIFLOAT8) {
        GMM_CHECK_REPORT(
            scaleDtype == DataType::DT_UINT64 || scaleDtype == DataType::DT_FLOAT || scaleDtype == DataType::DT_INT64,
            OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), scaleName_.c_str(), op::ToString(scaleDtype).GetString(),
                                      "UINT64, INT64 or FLOAT32"));
        return CheckFp8Hif8QuantParams();
    } else if ((xDtype == DataType::DT_FLOAT8_E4M3FN || xDtype == DataType::DT_FLOAT8_E5M2) &&
               (weightDtype == DataType::DT_FLOAT8_E4M3FN || weightDtype == DataType::DT_FLOAT8_E5M2)) {
        return CheckFp8Params(scaleDtype);
    } else if ((xDtype == DataType::DT_FLOAT4_E2M1 || xDtype == DataType::DT_FLOAT4_E1M2) &&
               (weightDtype == DataType::DT_FLOAT4_E2M1 || weightDtype == DataType::DT_FLOAT4_E1M2)) {
        return CheckFp4Params(scaleDtype);
    } else if (xDtype == DataType::DT_INT4 && weightDtype == DataType::DT_INT4) {
        GMM_CHECK_REPORT(scaleDtype == DataType::DT_UINT64,
                         OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), scaleName_.c_str(),
                                                   op::ToString(scaleDtype).GetString(), "UINT64"));
        GMM_CHECK_REPORT(
            yDtype == DataType::DT_FLOAT16 || yDtype == DataType::DT_BF16,
            OP_LOGE_FOR_INVALID_DTYPE(GetAclnnOpName(), "y", op::ToString(yDtype).GetString(), "FLOAT16 or BF16"));
        GMM_CHECK_REPORT(gmmParams_.biasOptional == nullptr,
                         OP_LOGE_FOR_INVALID_VALUE(GetAclnnOpName(), "bias", "nonnull", "nullptr"));
        GMM_CHECK_REPORT(
            gmmParams_.groupType == SPLIT_M,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GetAclnnOpName(), "groupType", std::to_string(gmmParams_.groupType),
                                                  "A4W4 only supports SPLIT_M(0)"));
        GMM_CHECK_REPORT(!gmmParams_.transposeX,
                         OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                             GetAclnnOpName(), xName_.c_str(), ViewShapeToString(GetInputTensor(gmmParams_.x)),
                             "in A4W4 quant case, the transposition of x is not supported"));
        if (gmmParams_.transposeWeight) {
            GMM_CHECK_REPORT(
                GetInputTensor(gmmParams_.weight)->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    GetAclnnOpName(), weightName_.c_str(), ViewShapeToString(GetInputTensor(gmmParams_.weight)),
                    "in A4W4 quant case, the transposition of weight is only supported in NZ format, "
                    "ND format does not support transposition"));
        }
        return ACLNN_SUCCESS;
    } else {
        std::string quantDtypes =
            std::string("x=") + op::ToString(xDtype).GetString() + ", weight=" + op::ToString(weightDtype).GetString();
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(GetAclnnOpName(), "x and weight", quantDtypes.c_str(),
                                               "the dtypes of x and weight must be within supported quant dtype pairs");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
