/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_grouped_matmul_inplace_add_950_checker.h"
#include "log/log.h"

#include <sstream>

using namespace QGmmInPlaceAdd;

namespace QGmmInPlaceAdd {
template class AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<aclTensor>;
}

namespace {
constexpr const char *QGMM_INPLACE_ADD_ACLNN_OP_NAME = "aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize";
constexpr size_t DIM_NUM_2D = 2U;

#define QGMM_INPLACE_ADD_CHECK_REPORT(cond, retExpr, reportExpr) \
    do { \
        if (!(cond)) { \
            reportExpr; \
            retExpr; \
        } \
    } while (0)

const aclTensor *GetInputTensor(const aclTensor *input, size_t index = 0)
{
    (void)index;
    return input;
}

size_t GetInputTensorSize(const aclTensor *input)
{
    (void)input;
    return 1;
}

std::string ShapeToStringWithoutBracket(const op::Shape &shape)
{
    std::ostringstream oss;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << shape.GetDim(i);
    }
    return oss.str();
}

std::string ViewShapeToString(const aclTensor *tensor)
{
    return ShapeToStringWithoutBracket(tensor->GetViewShape());
}
} // namespace

template <typename T>
void AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::SetInputName(const std::string &xName,
                                                                      const std::string &weightName,
                                                                      const std::string &perTokenScaleName,
                                                                      const std::string &scaleName,
                                                                      const std::string &groupTensorName)
{
    this->xName_ = xName;
    this->weightName_ = weightName;
    this->perTokenScaleName_ = perTokenScaleName;
    this->scaleName_ = scaleName;
    this->groupTensorName_ = groupTensorName;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckTensorListSizeForEachInput() const
{
    QGMM_INPLACE_ADD_CHECK_REPORT(
        GetInputTensorSize(gmmParams_.scaleOptional) == GetInputTensorSize(gmmParams_.x),
        return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
            QGMM_INPLACE_ADD_ACLNN_OP_NAME, scaleName_ + " and " + xName_,
            scaleName_ + "=" + std::to_string(GetInputTensorSize(gmmParams_.scaleOptional)) + ", " + xName_ + "=" +
                std::to_string(GetInputTensorSize(gmmParams_.x)),
            "in quant case, the tensor nums in " + scaleName_ + " and " + xName_ + " must be the same"));
    if (gmmParams_.perTokenScaleOptional != nullptr) {
        QGMM_INPLACE_ADD_CHECK_REPORT(
            GetInputTensorSize(gmmParams_.perTokenScaleOptional) == GetInputTensorSize(gmmParams_.x),
            return ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
                QGMM_INPLACE_ADD_ACLNN_OP_NAME, perTokenScaleName_ + " and " + xName_,
                perTokenScaleName_ + "=" + std::to_string(GetInputTensorSize(gmmParams_.perTokenScaleOptional)) + ", " +
                    xName_ + "=" + std::to_string(GetInputTensorSize(gmmParams_.x)),
                "in quant case, the tensor nums in " + perTokenScaleName_ + " and " + xName_ + " must be the same"));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckGeneralQuantShape() const
{
    CHECK_RET(CheckTensorListSizeForEachInput() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); i++) {
        const auto *xTensor = GetInputTensor(gmmParams_.x, i);
        const auto *weightTensor = GetInputTensor(gmmParams_.weight, i);
        const auto *yTensor = GetInputTensor(gmmParams_.y, i);
        auto weightNIndex = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum() - 1;
        auto yNIndex = GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDimNum() - 1;
        QGMM_INPLACE_ADD_CHECK_REPORT(
            yTensor->GetViewShape().GetDim(0) == groupNum, return ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                QGMM_INPLACE_ADD_ACLNN_OP_NAME, yName_, ViewShapeToString(yTensor),
                "when the value of groupType is 2 (split K), first dim of y must be equal to axis 0 of groupList [" +
                    std::to_string(groupNum) + "]"));
        // y shape dim num must 3
        QGMM_INPLACE_ADD_CHECK_REPORT(
            GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDimNum() == 3, return ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_SHAPEDIM(QGMM_INPLACE_ADD_ACLNN_OP_NAME, yName_,
                                         std::to_string(GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDimNum()),
                                         "3"));
        QGMM_INPLACE_ADD_CHECK_REPORT(
            yTensor->GetViewShape().GetDim(1) == xTensor->GetViewShape().GetDim(0), return ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                QGMM_INPLACE_ADD_ACLNN_OP_NAME, yName_ + " and " + xName_,
                yName_ + "=" + ViewShapeToString(yTensor) + ", " + xName_ + "=" + ViewShapeToString(xTensor),
                "axis M of y must be equal to axis M of x"));
        QGMM_INPLACE_ADD_CHECK_REPORT(
            yTensor->GetViewShape().GetDim(yNIndex) == weightTensor->GetViewShape().GetDim(weightNIndex),
            return ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                QGMM_INPLACE_ADD_ACLNN_OP_NAME, yName_ + " and " + weightName_,
                yName_ + "=" + ViewShapeToString(yTensor) + ", " + weightName_ + "=" + ViewShapeToString(weightTensor),
                "axis N of y must be equal to axis N of weight"));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckQuantCasesFormat() const
{
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); i++) {
        QGMM_INPLACE_ADD_CHECK_REPORT(
            !op::IsPrivateFormat(GetInputTensor(gmmParams_.x, i)->GetStorageFormat()), return ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_FORMAT(QGMM_INPLACE_ADD_ACLNN_OP_NAME, xName_,
                                       op::ToString(GetInputTensor(gmmParams_.x, i)->GetStorageFormat()).GetString(),
                                       "ND"));
        QGMM_INPLACE_ADD_CHECK_REPORT(
            !op::IsPrivateFormat(GetInputTensor(gmmParams_.weight, i)->GetStorageFormat()),
            return ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_FORMAT(
                QGMM_INPLACE_ADD_ACLNN_OP_NAME, weightName_,
                op::ToString(GetInputTensor(gmmParams_.weight, i)->GetStorageFormat()).GetString(), "ND"));
        QGMM_INPLACE_ADD_CHECK_REPORT(
            !op::IsPrivateFormat(GetInputTensor(gmmParams_.y, i)->GetStorageFormat()), return ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_FORMAT(QGMM_INPLACE_ADD_ACLNN_OP_NAME, yName_,
                                       op::ToString(GetInputTensor(gmmParams_.y, i)->GetStorageFormat()).GetString(),
                                       "ND"));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckHif8QuantParamsShape() const
{
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.weight); ++i) {
        auto scaleDimNumber = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDimNum();
        auto perTokenDimNumber = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDimNum();
        const auto *perTokenScaleTensor = GetInputTensor(gmmParams_.perTokenScaleOptional, i);
        const auto *scaleTensor = GetInputTensor(gmmParams_.scaleOptional, i);

        // 校验 scale1 (perTokenScale)
        if (!(perTokenDimNumber == 1 || perTokenDimNumber == DIM_NUM_2D)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(QGMM_INPLACE_ADD_ACLNN_OP_NAME, perTokenScaleName_,
                                         std::to_string(perTokenDimNumber), "1 or 2");
            return ACLNN_ERR_PARAM_INVALID;
        }
        auto perTokenFirstDim = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDim(0);
        QGMM_INPLACE_ADD_CHECK_REPORT(
            perTokenFirstDim == groupNum, return ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                QGMM_INPLACE_ADD_ACLNN_OP_NAME, perTokenScaleName_, ViewShapeToString(perTokenScaleTensor),
                "in T-T/T-C mode, first dim of perTokenScale must be equal to groupnum [" + std::to_string(groupNum) +
                    "]"));
        if (perTokenDimNumber == DIM_NUM_2D) {
            auto perTokenLastDim = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDim(1);
            QGMM_INPLACE_ADD_CHECK_REPORT(
                perTokenLastDim == 1, return ACLNN_ERR_PARAM_INVALID,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, perTokenScaleName_,
                                                      ViewShapeToString(perTokenScaleTensor),
                                                      "in T-T/T-C mode, last dim of perTokenScale must be equal to 1"));
        }

        if (!(scaleDimNumber == 1 || scaleDimNumber == DIM_NUM_2D)) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(QGMM_INPLACE_ADD_ACLNN_OP_NAME, scaleName_, std::to_string(scaleDimNumber),
                                         "1 or 2");
            return ACLNN_ERR_PARAM_INVALID;
        }
        auto scaleFirstDim = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(0);
        QGMM_INPLACE_ADD_CHECK_REPORT(
            scaleFirstDim == groupNum, return ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                QGMM_INPLACE_ADD_ACLNN_OP_NAME, scaleName_, ViewShapeToString(scaleTensor),
                "in T-T mode, first dim of scale must be equal to groupnum [" + std::to_string(groupNum) + "]"));
        if (scaleDimNumber == DIM_NUM_2D) {
            auto n = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(1);
            auto scaleLastDim = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(1);
            QGMM_INPLACE_ADD_CHECK_REPORT(
                scaleLastDim == 1 || scaleLastDim == n, return ACLNN_ERR_PARAM_INVALID,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    QGMM_INPLACE_ADD_ACLNN_OP_NAME, scaleName_, ViewShapeToString(scaleTensor),
                    "in T-T/T-C mode, last dim of scale must be equal to 1 or n [" + std::to_string(n) + "]"));
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckHif8QuantParams() const
{
    DataType scaleDtype = GetInputTensor(gmmParams_.scaleOptional)->GetDataType();
    QGMM_INPLACE_ADD_CHECK_REPORT(scaleDtype == DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID,
                                  OP_LOGE_FOR_INVALID_DTYPE(QGMM_INPLACE_ADD_ACLNN_OP_NAME, scaleName_,
                                                            op::ToString(scaleDtype).GetString(), "FLOAT32"));
    QGMM_INPLACE_ADD_CHECK_REPORT(
        gmmParams_.perTokenScaleOptional != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, perTokenScaleName_, "nullptr",
                                              "in hifloat8 case, the value of perTokenScale cannot be nullptr"));
    DataType perTokenScaleDtype = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetDataType();
    QGMM_INPLACE_ADD_CHECK_REPORT(perTokenScaleDtype == DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID,
                                  OP_LOGE_FOR_INVALID_DTYPE(QGMM_INPLACE_ADD_ACLNN_OP_NAME, perTokenScaleName_,
                                                            op::ToString(perTokenScaleDtype).GetString(), "FLOAT32"));
    QGMM_INPLACE_ADD_CHECK_REPORT(
        gmmParams_.biasOptional == nullptr, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, biasName_, "nonnull",
                                              "in hifloat8 case, the value of bias must be nullptr"));
    QGMM_INPLACE_ADD_CHECK_REPORT(
        gmmParams_.y != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, yName_, "nullptr",
                                              "in hifloat8 case, the value of y cannot be nullptr"));
    DataType yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
    QGMM_INPLACE_ADD_CHECK_REPORT(
        yDtype == DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_DTYPE(QGMM_INPLACE_ADD_ACLNN_OP_NAME, yName_, op::ToString(yDtype).GetString(), "FLOAT32"));
    CHECK_RET(CheckHif8QuantParamsShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<T>::CheckQuantGroupedMatmulInplaceAddDAV3510() const
{
    DataType xDtype = gmmParams_.xDtype;
    QGMM_INPLACE_ADD_CHECK_REPORT(
        gmmParams_.weight != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, weightName_, "nullptr",
                                              "in quant case, the value of weight cannot be nullptr"));
    DataType weightDtype = GetInputTensor(gmmParams_.weight)->GetDataType();
    QGMM_INPLACE_ADD_CHECK_REPORT(
        gmmParams_.scaleOptional != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, scaleName_, "nullptr",
                                              "in quant case, the value of scale cannot be nullptr"));
    QGMM_INPLACE_ADD_CHECK_REPORT(
        gmmParams_.groupTensorOptional != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, groupTensorName_, "nullptr",
                                              "in quant case, the value of groupList cannot be nullptr"));
    QGMM_INPLACE_ADD_CHECK_REPORT(
        gmmParams_.offsetOptional == nullptr, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "offset", "nonnull",
                                              "in quant case, the value of offset must be nullptr"));
    QGMM_INPLACE_ADD_CHECK_REPORT(GetInputTensorSize(gmmParams_.x) == 1 && GetInputTensorSize(gmmParams_.weight) == 1 &&
                                      GetInputTensorSize(gmmParams_.y) == 1,
                                  return ACLNN_ERR_PARAM_INVALID,
                                  OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
                                      QGMM_INPLACE_ADD_ACLNN_OP_NAME, xName_ + ", " + weightName_ + " and " + yName_,
                                      xName_ + "=" + std::to_string(GetInputTensorSize(gmmParams_.x)) + ", " +
                                          weightName_ + "=" + std::to_string(GetInputTensorSize(gmmParams_.weight)) +
                                          ", " + yName_ + "=" + std::to_string(GetInputTensorSize(gmmParams_.y)),
                                      "in quant case, the tensor nums in " + xName_ + ", " + weightName_ + " and " +
                                          yName_ + " must be the same"));
    int64_t groupListLen = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    QGMM_INPLACE_ADD_CHECK_REPORT(
        groupListLen <= 1024, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_LISTSIZE(QGMM_INPLACE_ADD_ACLNN_OP_NAME, groupTensorName_, std::to_string(groupListLen),
                                     "less than or equal to 1024"));
    CHECK_RET(CheckQuantCasesFormat() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGeneralQuantShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (xDtype == DataType::DT_HIFLOAT8 && weightDtype == DataType::DT_HIFLOAT8) {
        return CheckHif8QuantParams();
    } else {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, xName_ + " and " + weightName_,
                                               xName_ + "=" + op::ToString(xDtype).GetString() + ", " + weightName_ +
                                                   "=" + op::ToString(weightDtype).GetString(),
                                               "the dtypes of x and weight must be within the range {both HIFLOAT8}");
        return ACLNN_ERR_PARAM_INVALID;
    }
}

#undef QGMM_INPLACE_ADD_CHECK_REPORT
