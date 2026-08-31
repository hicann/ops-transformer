/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_grouped_matmul_activation_quant_weight_nz.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "gmm/common/op_host/log_format_util.h"
#include "grouped_matmul_activation_quant.h"
#include "grouped_matmul_activation_quant_common.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "util/math_util.h"
#include "log/log.h"

using namespace op;
using Ops::Transformer::Gmm::FormatString;

namespace {
constexpr char OP_NAME[] = "aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize";
constexpr char GELU_TANH[] = "gelu_tanh";
constexpr char QUANT_MODE_MX[] = "mx";
constexpr char ROUND_MODE_RINT[] = "rint";
constexpr int64_t MAX_GROUP_NUM = 1024L;
constexpr int64_t FLOAT8_E5M2_VALUE = static_cast<int64_t>(DataType::DT_FLOAT8_E5M2);
constexpr int64_t FLOAT8_E4M3FN_VALUE = static_cast<int64_t>(DataType::DT_FLOAT8_E4M3FN);
constexpr int64_t FLOAT4_E2M1_VALUE = static_cast<int64_t>(DataType::DT_FLOAT4_E2M1);
constexpr int64_t FLOAT4_E1M2_VALUE = static_cast<int64_t>(DataType::DT_FLOAT4_E1M2);
constexpr int64_t SCALE_ALG_OCP = 0L;
constexpr int64_t SCALE_ALG_CUBLAS = 1L;
constexpr int64_t SCALE_ALG_DYNAMIC_DTYPE_RANGE = 2L;
constexpr float DEFAULT_DST_TYPE_MAX = 0.0F;
constexpr float FP4_E2M1_DST_TYPE_MAX_MIN = 6.0F;
constexpr float FP4_E2M1_DST_TYPE_MAX_MAX = 12.0F;
constexpr float FLOAT_EPS = 1e-6F;

#define GMMAQ_CHECK_WITH_LOG(cond, retCode, logExpr) \
    do { \
        if (!(cond)) { \
            logExpr; \
            return retCode; \
        } \
    } while (0)

bool IsFp8(DataType dtype)
{
    return dtype == DataType::DT_FLOAT8_E4M3FN || dtype == DataType::DT_FLOAT8_E5M2;
}

bool IsFp4(DataType dtype)
{
    return dtype == DataType::DT_FLOAT4_E2M1 || dtype == DataType::DT_FLOAT4_E1M2;
}

bool IsSupportedMxInputPair(DataType xDtype, DataType weightDtype)
{
    return (IsFp8(xDtype) && weightDtype == DataType::DT_FLOAT8_E4M3FN) || (IsFp4(xDtype) && IsFp4(weightDtype));
}

bool IsMxOutputDtypeValue(int64_t dtype)
{
    return dtype == FLOAT8_E4M3FN_VALUE || dtype == FLOAT8_E5M2_VALUE || dtype == FLOAT4_E2M1_VALUE ||
           dtype == FLOAT4_E1M2_VALUE;
}

bool IsMxScaleDtype(DataType dtype)
{
    return dtype == DataType::DT_FLOAT8_E8M0;
}

bool IsEmptyQuantMode(const char *quantMode)
{
    return quantMode == nullptr || quantMode[0] == '\0';
}

aclnnStatus CheckMxQuantDtype(const aclTensor *x, const aclTensor *weight, const aclTensor *xScaleOptional,
                              const aclTensor *weightScale, const aclTensor *y, const aclTensor *yScale,
                              int64_t effectiveYDtype)
{
    GMMAQ_CHECK_WITH_LOG(
        IsSupportedMxInputPair(x->GetDataType(), weight->GetDataType()), ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            OP_NAME, "x, weight",
            FormatString("{%s, %s}", op::ToString(x->GetDataType()).GetString(),
                         op::ToString(weight->GetDataType()).GetString())
                .c_str(),
            "when the quantization mode is mx, x and weight must be a supported FP8 pair or both be FLOAT4"));
    GMMAQ_CHECK_WITH_LOG(IsMxScaleDtype(xScaleOptional->GetDataType()), ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                             OP_NAME, "xScaleOptional", op::ToString(xScaleOptional->GetDataType()).GetString(),
                             "when the quantization mode is mx, the dtype of xScaleOptional must be FLOAT8_E8M0"));
    GMMAQ_CHECK_WITH_LOG(IsMxScaleDtype(weightScale->GetDataType()), ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                             OP_NAME, "weightScale", op::ToString(weightScale->GetDataType()).GetString(),
                             "when the quantization mode is mx, the dtype of weightScale must be FLOAT8_E8M0"));
    const bool isFp4Input = IsFp4(x->GetDataType()) && IsFp4(weight->GetDataType());
    GMMAQ_CHECK_WITH_LOG(
        IsFp8(y->GetDataType()) || (isFp4Input && IsFp4(y->GetDataType())), ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "y", op::ToString(y->GetDataType()).GetString(),
                                              "when the quantization mode is mx, y supports FLOAT8_E4M3FN/FLOAT8_E5M2; "
                                              "FLOAT4_E2M1/FLOAT4_E1M2 output additionally requires FLOAT4 inputs"));
    GMMAQ_CHECK_WITH_LOG(IsMxScaleDtype(yScale->GetDataType()), ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                             OP_NAME, "yScale", op::ToString(yScale->GetDataType()).GetString(),
                             "when the quantization mode is mx, the dtype of yScale must be FLOAT8_E8M0"));
    return ACLNN_SUCCESS;
}

bool IsNzFormat(const aclTensor *tensor)
{
    return tensor != nullptr && ge::GetPrimaryFormat(tensor->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ;
}

bool IsEmptyTensorList(const aclTensorList *tensorList)
{
    if (tensorList == nullptr || tensorList->Size() == 0) {
        return true;
    }
    if (tensorList->Size() != gmaq::SINGLE_TENSOR_SIZE || (*tensorList)[gmaq::FIRST_TENSOR_INDEX] == nullptr) {
        return false;
    }
    const auto &shape = (*tensorList)[gmaq::FIRST_TENSOR_INDEX]->GetViewShape();
    return shape.GetDimNum() == 1 && shape.GetDim(gmaq::DIM_0) == 0;
}

void NormalizeEmptyTensorList(const aclTensorList *&tensorList)
{
    if (IsEmptyTensorList(tensorList)) {
        tensorList = nullptr;
    }
}

aclnnStatus CheckTensorList(const aclTensorList *tensorList, const char *name)
{
    GMMAQ_CHECK_WITH_LOG(tensorList != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                         OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, name, "does not support nullptr"));
    GMMAQ_CHECK_WITH_LOG(
        tensorList->Size() == gmaq::SINGLE_TENSOR_SIZE, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_TENSORNUM(OP_NAME, name, tensorList->Size(), std::to_string(gmaq::SINGLE_TENSOR_SIZE)));
    const std::string tensorName = std::string(name) + "[0]";
    GMMAQ_CHECK_WITH_LOG(
        (*tensorList)[gmaq::FIRST_TENSOR_INDEX] != nullptr, ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, tensorName.c_str(), "does not support nullptr"));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckTensorDimNum(const aclTensor *tensor, const char *name, size_t expectedDimNum)
{
    auto actualDimNum = tensor->GetViewShape().GetDimNum();
    GMMAQ_CHECK_WITH_LOG(actualDimNum == expectedDimNum, ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                             OP_NAME, name, std::to_string(actualDimNum),
                             FormatString("the shape dim of %s must be %zu", name, expectedDimNum).c_str()));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckTensorShapeEqual(const aclTensor *tensor, const char *name, const op::Shape &expectedShape)
{
    GMMAQ_CHECK_WITH_LOG(
        tensor->GetViewShape() == expectedShape, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, name, op::ToString(tensor->GetViewShape()).GetString(),
            FormatString("the shape of %s must be %s", name, op::ToString(expectedShape).GetString()).c_str()));
    return ACLNN_SUCCESS;
}

aclnnStatus CreateEmptyTensorList(const aclDataType dataType, const aclTensorList *&tensorList, aclOpExecutor *executor)
{
    if (tensorList != nullptr) {
        return ACLNN_SUCCESS;
    }
    std::vector<aclTensor *> emptyTensors;
    aclTensor *emptyTensor = executor->AllocTensor({0}, static_cast<DataType>(dataType));
    CHECK_RET(emptyTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    emptyTensors.emplace_back(emptyTensor);
    tensorList = executor->AllocTensorList(emptyTensors.data(), emptyTensors.size());
    CHECK_RET(tensorList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus WrapWeightNzTensorList(const aclTensorList *&weight, aclOpExecutor *executor)
{
    CHECK_RET(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    std::vector<const aclTensor *> tensors;
    tensors.reserve(weight->Size());
    for (size_t i = 0; i < weight->Size(); ++i) {
        const aclTensor *tensor = (*weight)[i];
        CHECK_RET(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        op::Shape viewShape = tensor->GetViewShape();
        op::Shape storageShape = tensor->GetStorageShape();
        auto wrappedTensor = executor->CreateView(tensor, viewShape, tensor->GetViewOffset());
        CHECK_RET(wrappedTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        wrappedTensor->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
        wrappedTensor->SetOriginalFormat(tensor->GetViewFormat());
        wrappedTensor->SetViewShape(tensor->GetViewShape());
        wrappedTensor->SetStorageShape(storageShape);
        tensors.emplace_back(wrappedTensor);
    }
    weight = executor->AllocTensorList(tensors.data(), tensors.size());
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

bool TryMultiplyInt64(int64_t lhs, int64_t rhs, int64_t &result)
{
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool IsTransposeLastTwoDims(const aclTensor *tensor)
{
    auto shape = tensor->GetViewShape();
    if (shape.GetDimNum() < gmaq::X_DIM_NUM) {
        return false;
    }
    int64_t lastDim = static_cast<int64_t>(shape.GetDimNum()) - 1;
    int64_t penultimateDim = static_cast<int64_t>(shape.GetDimNum()) - 2;
    auto strides = tensor->GetViewStrides();
    if (strides[penultimateDim] != 1 || strides[lastDim] != shape.GetDim(penultimateDim)) {
        return false;
    }
    int64_t expectedStride = 0;
    if (!TryMultiplyInt64(shape.GetDim(lastDim), shape.GetDim(penultimateDim), expectedStride)) {
        return false;
    }
    for (int64_t batchDim = static_cast<int64_t>(shape.GetDimNum()) - 3; batchDim >= 0; --batchDim) {
        if (strides[batchDim] != expectedStride) {
            return false;
        }
        if (!TryMultiplyInt64(expectedStride, shape.GetDim(batchDim), expectedStride)) {
            return false;
        }
    }
    return true;
}

bool IsTransposeForMxShape(const aclTensor *tensor)
{
    auto shape = tensor->GetViewShape();
    if (shape.GetDimNum() < gmaq::WEIGHT_SCALE_DIM_NUM) {
        return false;
    }
    auto strides = tensor->GetViewStrides();
    int64_t lastDim = static_cast<int64_t>(shape.GetDimNum()) - 1;
    int64_t secondLastDim = static_cast<int64_t>(shape.GetDimNum()) - 2;
    int64_t thirdLastDim = static_cast<int64_t>(shape.GetDimNum()) - 3;
    int64_t expectedSecondLastStride = 0;
    if (!TryMultiplyInt64(shape.GetDim(thirdLastDim), gmaq::MX_SCALE_PAIR, expectedSecondLastStride)) {
        return false;
    }
    return strides[lastDim] == 1 && strides[thirdLastDim] == gmaq::MX_SCALE_PAIR &&
           strides[secondLastDim] == expectedSecondLastStride;
}

bool IsSpecialMxXTransposeCase(const aclTensor *x, const aclTensor *xScaleOptional)
{
    auto xShape = x->GetViewShape();
    auto xScaleShape = xScaleOptional->GetViewShape();
    bool specialX =
        xShape.GetDimNum() >= gmaq::X_DIM_NUM && xShape.GetDim(gmaq::DIM_0) == 1 && xShape.GetDim(gmaq::DIM_1) == 1;
    bool specialXScale = xScaleShape.GetDimNum() >= gmaq::SCALE_DIM_NUM && xScaleShape.GetDim(gmaq::DIM_0) == 1 &&
                         xScaleShape.GetDim(gmaq::DIM_1) == 1;
    return specialX || specialXScale;
}

aclnnStatus CreateContiguousTensorListForMXTypeMScale(const aclTensorList *tensorList,
                                                      const aclTensorList *&newTensorList, aclOpExecutor *executor)
{
    std::vector<aclTensor *> tensors;
    tensors.reserve(tensorList->Size());
    for (size_t idx = 0; idx < tensorList->Size(); ++idx) {
        const aclTensor *inputTensor = (*tensorList)[idx];
        CHECK_RET(inputTensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        op::Shape viewShape = inputTensor->GetViewShape();
        op::Shape shape;
        shape.SetScalar();
        GMMAQ_CHECK_WITH_LOG(
            viewShape.GetDimNum() >= gmaq::WEIGHT_SCALE_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "weightScale", std::to_string(viewShape.GetDimNum()),
                                                     "weightScale dim num must be greater than or equal to 4"));
        shape.AppendDim(viewShape.GetDim(gmaq::DIM_0));
        shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - gmaq::DIM_2));
        shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - gmaq::DIM_3));
        shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - gmaq::DIM_1));
        aclTensor *tensor = executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset());
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensor->SetStorageFormat(inputTensor->GetStorageFormat());
        tensors.emplace_back(tensor);
    }
    newTensorList = executor->AllocTensorList(tensors.data(), tensors.size());
    CHECK_RET(newTensorList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CreateContiguousTensorList(const aclTensorList *tensorList, const aclTensorList *&newTensorList,
                                       aclOpExecutor *executor)
{
    std::vector<aclTensor *> tensors;
    tensors.reserve(tensorList->Size());
    for (size_t idx = 0; idx < tensorList->Size(); ++idx) {
        const aclTensor *inputTensor = (*tensorList)[idx];
        CHECK_RET(inputTensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        op::Shape viewShape = inputTensor->GetViewShape();
        uint32_t viewShapeDimsNum = viewShape.GetDimNum();
        op::Shape shape;
        shape.SetScalar();
        GMMAQ_CHECK_WITH_LOG(
            viewShapeDimsNum >= gmaq::X_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "weight", std::to_string(viewShapeDimsNum),
                                                     "weight dim num must be greater than or equal to 2"));
        for (uint32_t i = 0; i < viewShapeDimsNum - gmaq::X_DIM_NUM; ++i) {
            shape.AppendDim(viewShape.GetDim(i));
        }
        shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 1));
        shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 2));
        aclTensor *tensor = executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset());
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensor->SetStorageFormat(inputTensor->GetStorageFormat());
        tensor->SetStorageShape(inputTensor->GetStorageShape());
        tensors.emplace_back(tensor);
    }
    newTensorList = executor->AllocTensorList(tensors.data(), tensors.size());
    CHECK_RET(newTensorList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMXTranspose(const aclTensor *x, const aclTensor *xScaleOptional, const aclTensorList *weight,
                             const aclTensorList *weightScale, bool &transposeWeight)
{
    const aclTensor *w = (*weight)[gmaq::FIRST_TENSOR_INDEX];
    const aclTensor *wScale = (*weightScale)[gmaq::FIRST_TENSOR_INDEX];
    bool transposeWeightScale = IsTransposeForMxShape(wScale);
    bool transposeWeightByStride = IsTransposeLastTwoDims(w);
    bool transposeX = IsTransposeLastTwoDims(x);
    bool transposeXScale = IsTransposeForMxShape(xScaleOptional);
    GMMAQ_CHECK_WITH_LOG(transposeWeightScale == transposeWeightByStride, ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                             OP_NAME, "weightScale", transposeWeightScale ? "transposed" : "non-transposed",
                             "the transposition of weightScale must be equal to the transposition of weight"));
    if (transposeWeightScale && transposeWeightByStride) {
        transposeWeight = true;
    }
    if (IsSpecialMxXTransposeCase(x, xScaleOptional)) {
        return ACLNN_SUCCESS;
    }
    GMMAQ_CHECK_WITH_LOG(
        !transposeX && !transposeXScale, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(OP_NAME, "x, xScaleOptional",
                                               FormatString("{%s, %s}", transposeX ? "transposed" : "non-transposed",
                                                            transposeXScale ? "transposed" : "non-transposed")
                                                   .c_str(),
                                               "x and xScaleOptional only support non-transposed input"));
    return ACLNN_SUCCESS;
}

aclnnStatus NormalizeMXTranspose(const aclTensorList *&weight, const aclTensorList *&weightScale, bool transposeWeight,
                                 aclOpExecutor *executor)
{
    if (!transposeWeight) {
        return ACLNN_SUCCESS;
    }
    CHECK_RET(CreateContiguousTensorListForMXTypeMScale(weightScale, weightScale, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CreateContiguousTensorList(weight, weight, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMxQuantShape(const aclTensor *x, const aclTensor *groupList, const aclTensor *xScaleOptional,
                              const aclTensor *weight, const aclTensor *weightScale, bool transposeWeight,
                              const aclTensor *y, const aclTensor *yScale)
{
    int64_t m = x->GetViewShape().GetDim(gmaq::DIM_0);
    int64_t k = x->GetViewShape().GetDim(gmaq::DIM_1);
    int64_t e = groupList->GetViewShape().GetDim(gmaq::DIM_0);
    int64_t n = transposeWeight ? weightScale->GetViewShape().GetDim(gmaq::DIM_1) :
                                  weightScale->GetViewShape().GetDim(gmaq::DIM_2);
    const bool isMxfp4 = IsFp4(x->GetDataType()) && IsFp4(weight->GetDataType());
    bool emptyOutput = y->IsEmpty() || groupList->IsEmpty() || yScale->IsEmpty();
    GMMAQ_CHECK_WITH_LOG((e >= 1 && e <= MAX_GROUP_NUM) || (e == 0 && emptyOutput), ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "groupList dim 0", std::to_string(e),
                                                               "groupList dim 0 must be in range [1, 1024], "
                                                               "or be 0 in empty tensor scenario"));
    GMMAQ_CHECK_WITH_LOG(m == 0 || n == 0 || k > 0, ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "K", std::to_string(k),
                                                               "K must be greater than 0 unless M or N is 0"));
    GMMAQ_CHECK_WITH_LOG(
        n % gmaq::MX_GROUP_SIZE == 0, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "N", std::to_string(n), "N must be divisible by 64"));
    GMMAQ_CHECK_WITH_LOG(
        !isMxfp4 || (k % gmaq::MXFP4_PACK_FACTOR == 0 && k != gmaq::MXFP4_PACK_FACTOR), ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "K", std::to_string(k),
                                              "when x and weight are FLOAT4, K must be even and must not be 2"));

    const auto &weightShape = weight->GetStorageShape();
    const auto &weightViewShape = weight->GetViewShape();
    const auto &weightScaleShape = weightScale->GetViewShape();
    GMMAQ_CHECK_WITH_LOG(
        weightShape.GetDim(gmaq::DIM_0) == e, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "weight dim 0", std::to_string(weightShape.GetDim(gmaq::DIM_0)),
                                              "weight dim 0 must be equal to groupList dim 0"));
    GMMAQ_CHECK_WITH_LOG(weightScaleShape.GetDim(gmaq::DIM_0) == e, ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "weightScale dim 0",
                                                               std::to_string(weightScaleShape.GetDim(gmaq::DIM_0)),
                                                               "weightScale dim 0 must be equal to groupList dim 0"));

    op::Shape xScaleExpectShape = {m, Ops::Base::CeilDiv(k, gmaq::MX_GROUP_SIZE), gmaq::MX_SCALE_PAIR};
    op::Shape yExpectShape = {m, n};
    op::Shape yScaleExpectShape = {m, Ops::Base::CeilDiv(n, gmaq::MX_GROUP_SIZE), gmaq::MX_SCALE_PAIR};
    CHECK_RET(CheckTensorShapeEqual(xScaleOptional, "xScaleOptional", xScaleExpectShape) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorShapeEqual(y, "y", yExpectShape) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorShapeEqual(yScale, "yScale", yScaleExpectShape) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    op::Shape weightExpectShape;
    op::Shape weightScaleExpectShape;
    const int64_t nzLastDim = isMxfp4 ? gmaq::NZ_LAST_DIM_FP4 : gmaq::NZ_LAST_DIM_FP8;
    if (transposeWeight) {
        weightExpectShape = {e, Ops::Base::CeilDiv(k, nzLastDim), Ops::Base::CeilDiv(n, gmaq::NZ_C0_DIM),
                             gmaq::NZ_C0_DIM, nzLastDim};
        weightScaleExpectShape = {e, n, Ops::Base::CeilDiv(k, gmaq::MX_GROUP_SIZE), gmaq::MX_SCALE_PAIR};
    } else {
        weightExpectShape = {e, Ops::Base::CeilDiv(n, nzLastDim), Ops::Base::CeilDiv(k, gmaq::NZ_C0_DIM),
                             gmaq::NZ_C0_DIM, nzLastDim};
        weightScaleExpectShape = {e, Ops::Base::CeilDiv(k, gmaq::MX_GROUP_SIZE), n, gmaq::MX_SCALE_PAIR};
    }
    op::Shape weightViewExpectShape = transposeWeight ? op::Shape({e, n, k}) : op::Shape({e, k, n});
    CHECK_RET(CheckTensorShapeEqual(weight, "weight", weightViewExpectShape) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    GMMAQ_CHECK_WITH_LOG(
        weightShape == weightExpectShape, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "weight storageShape", op::ToString(weightShape).GetString(),
            FormatString("the storageShape of weight must be %s", op::ToString(weightExpectShape).GetString())
                .c_str()));
    CHECK_RET(CheckTensorShapeEqual(weightScale, "weightScale", weightScaleExpectShape) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckRequiredParams(const aclTensor *x, const aclTensor *groupList, const aclTensorList *weight,
                                const aclTensorList *weightScale, const char *activationType, const aclTensor *y,
                                const aclTensor *yScale)
{
    GMMAQ_CHECK_WITH_LOG(x != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                         OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "x", "does not support nullptr"));
    GMMAQ_CHECK_WITH_LOG(groupList != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                         OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "groupList", "does not support nullptr"));
    aclnnStatus checkTensorListRet = CheckTensorList(weight, "weight");
    CHECK_RET(checkTensorListRet == ACLNN_SUCCESS, checkTensorListRet);
    checkTensorListRet = CheckTensorList(weightScale, "weightScale");
    CHECK_RET(checkTensorListRet == ACLNN_SUCCESS, checkTensorListRet);
    GMMAQ_CHECK_WITH_LOG(y != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                         OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "y", "does not support nullptr"));
    GMMAQ_CHECK_WITH_LOG(yScale != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                         OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "yScale", "does not support nullptr"));
    GMMAQ_CHECK_WITH_LOG(
        activationType != nullptr, ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "activationType", "does not support nullptr"));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckActivationType(const char *activationType)
{
    GMMAQ_CHECK_WITH_LOG(std::strcmp(activationType, GELU_TANH) == 0, ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "activationType", activationType,
                                                               "activationType must be gelu_tanh"));
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveAndCheckQuantMode(const aclTensor *x, const aclTensor *xScaleOptional, const char *quantMode,
                                     const char *&effectiveQuantMode)
{
    if (!IsEmptyQuantMode(quantMode)) {
        GMMAQ_CHECK_WITH_LOG(
            std::strcmp(quantMode, QUANT_MODE_MX) == 0, ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "quantMode", quantMode, "quantMode must be mx"));
        effectiveQuantMode = quantMode;
        return ACLNN_SUCCESS;
    }

    if (xScaleOptional != nullptr && (IsFp8(x->GetDataType()) || IsFp4(x->GetDataType())) &&
        IsMxScaleDtype(xScaleOptional->GetDataType())) {
        effectiveQuantMode = QUANT_MODE_MX;
        return ACLNN_SUCCESS;
    }

    const std::string actualDtypes =
        FormatString("{%s, %s}", op::ToString(x->GetDataType()).GetString(),
                     xScaleOptional == nullptr ? "nullptr" : op::ToString(xScaleOptional->GetDataType()).GetString());
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
        OP_NAME, "x, xScaleOptional", actualDtypes.c_str(),
        "the current dtype combination can not match any quantization mode supported by this op");
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus ResolveAndCheckRoundMode(const char *roundMode, const char *&effectiveRoundMode)
{
    if (roundMode == nullptr || roundMode[0] == '\0') {
        effectiveRoundMode = ROUND_MODE_RINT;
        return ACLNN_SUCCESS;
    }
    GMMAQ_CHECK_WITH_LOG(
        std::strcmp(roundMode, ROUND_MODE_RINT) == 0, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "roundMode", roundMode,
                                              "when the quantization mode is mx, roundMode must be rint"));
    effectiveRoundMode = roundMode;
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMxQuantAttrs(int64_t groupListType, int64_t effectiveYDtype, const char *roundMode,
                              const char *&effectiveRoundMode, int64_t scaleAlg, float dstTypeMax,
                              const aclTensorList *bias)
{
    CHECK_RET(ResolveAndCheckRoundMode(roundMode, effectiveRoundMode) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    GMMAQ_CHECK_WITH_LOG(IsMxOutputDtypeValue(effectiveYDtype), ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                             OP_NAME, "y", op::ToString(static_cast<DataType>(effectiveYDtype)).GetString(),
                             "the dtype of y must be FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1 or FLOAT4_E1M2"));
    GMMAQ_CHECK_WITH_LOG(groupListType == 0 || groupListType == 1, ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "groupListType", std::to_string(groupListType),
                                                               "groupListType must be 0 or 1"));
    GMMAQ_CHECK_WITH_LOG(
        scaleAlg == SCALE_ALG_OCP || scaleAlg == SCALE_ALG_CUBLAS || scaleAlg == SCALE_ALG_DYNAMIC_DTYPE_RANGE,
        ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "scaleAlg", std::to_string(scaleAlg),
                                              "when the quantization mode is mx, scaleAlg must be 0, 1 or 2"));
    GMMAQ_CHECK_WITH_LOG(
        scaleAlg != SCALE_ALG_CUBLAS || effectiveYDtype == FLOAT8_E4M3FN_VALUE || effectiveYDtype == FLOAT8_E5M2_VALUE,
        ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "scaleAlg", std::to_string(scaleAlg),
                                              "scaleAlg 1 only supports FLOAT8_E4M3FN or FLOAT8_E5M2 output"));
    GMMAQ_CHECK_WITH_LOG(scaleAlg != SCALE_ALG_DYNAMIC_DTYPE_RANGE || effectiveYDtype == FLOAT4_E2M1_VALUE,
                         ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "scaleAlg", std::to_string(scaleAlg),
                                                               "scaleAlg 2 only supports FLOAT4_E2M1 output"));
    const bool isDefaultDstTypeMax = std::fabs(dstTypeMax - DEFAULT_DST_TYPE_MAX) <= FLOAT_EPS;
    if (scaleAlg == SCALE_ALG_DYNAMIC_DTYPE_RANGE) {
        const bool isValidDstTypeMax =
            std::isfinite(dstTypeMax) && (isDefaultDstTypeMax || (dstTypeMax >= FP4_E2M1_DST_TYPE_MAX_MIN &&
                                                                  dstTypeMax <= FP4_E2M1_DST_TYPE_MAX_MAX));
        GMMAQ_CHECK_WITH_LOG(
            isValidDstTypeMax, ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "dstTypeMax", std::to_string(dstTypeMax),
                                                  "when scaleAlg is 2, dstTypeMax must be 0.0 or within [6.0, 12.0]"));
    } else {
        GMMAQ_CHECK_WITH_LOG(isDefaultDstTypeMax, ACLNN_ERR_PARAM_INVALID,
                             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "dstTypeMax", std::to_string(dstTypeMax),
                                                                   "when scaleAlg is not 2, dstTypeMax must be 0.0"));
    }
    GMMAQ_CHECK_WITH_LOG(IsEmptyTensorList(bias), ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
                             OP_NAME, "bias", FormatString("bias=%zu", bias == nullptr ? 0UL : bias->Size()),
                             "when the quantization mode is mx, bias must be nullptr or an empty tensorList"));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMxQuantDim(const aclTensor *x, const aclTensor *groupList, const aclTensor *weight,
                            const aclTensor *weightScale, const aclTensor *xScaleOptional, const aclTensor *y,
                            const aclTensor *yScale)
{
    CHECK_RET(CheckTensorDimNum(x, "x", gmaq::X_DIM_NUM) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimNum(xScaleOptional, "xScaleOptional", gmaq::SCALE_DIM_NUM) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimNum(weight, "weight", gmaq::WEIGHT_LOGICAL_DIM_NUM) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimNum(weightScale, "weightScale", gmaq::WEIGHT_SCALE_DIM_NUM) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimNum(groupList, "groupList", gmaq::GROUP_LIST_DIM_NUM) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimNum(y, "y", gmaq::OUT_DIM_NUM) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimNum(yScale, "yScale", gmaq::OUT_SCALE_DIM_NUM) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    GMMAQ_CHECK_WITH_LOG(weight->GetStorageShape().GetDimNum() == gmaq::WEIGHT_NZ_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "weight storageShape",
                                                                  std::to_string(weight->GetStorageShape().GetDimNum()),
                                                                  "the dim num of weight storageShape must be 5"));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMxQuantFormat(const aclTensor *x, const aclTensor *groupList, const aclTensor *weight,
                               const aclTensor *weightScale, const aclTensor *xScaleOptional, const aclTensor *y,
                               const aclTensor *yScale)
{
    GMMAQ_CHECK_WITH_LOG(
        !op::IsPrivateFormat(x->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(OP_NAME, "x", op::ToString(x->GetStorageFormat()).GetString(),
                                               "when the quantization mode is mx, the format of x must be ND"));
    GMMAQ_CHECK_WITH_LOG(!op::IsPrivateFormat(xScaleOptional->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                             OP_NAME, "xScaleOptional", op::ToString(xScaleOptional->GetStorageFormat()).GetString(),
                             "when the quantization mode is mx, the format of xScaleOptional must be ND"));
    GMMAQ_CHECK_WITH_LOG(!op::IsPrivateFormat(groupList->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                             OP_NAME, "groupList", op::ToString(groupList->GetStorageFormat()).GetString(),
                             "when the quantization mode is mx, the format of groupList must be ND"));
    GMMAQ_CHECK_WITH_LOG(!op::IsPrivateFormat(weightScale->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                             OP_NAME, "weightScale", op::ToString(weightScale->GetStorageFormat()).GetString(),
                             "when the quantization mode is mx, the format of weightScale must be ND"));
    GMMAQ_CHECK_WITH_LOG(
        !op::IsPrivateFormat(y->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(OP_NAME, "y", op::ToString(y->GetStorageFormat()).GetString(),
                                               "when the quantization mode is mx, the format of y must be ND"));
    GMMAQ_CHECK_WITH_LOG(
        !op::IsPrivateFormat(yScale->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(OP_NAME, "yScale", op::ToString(yScale->GetStorageFormat()).GetString(),
                                               "when the quantization mode is mx, the format of yScale must be ND"));
    GMMAQ_CHECK_WITH_LOG(IsNzFormat(weight), ACLNN_ERR_PARAM_INVALID,
                         OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                             OP_NAME, "weight", op::ToString(weight->GetStorageFormat()).GetString(),
                             "when the quantization mode is mx, the format of weight must be FRACTAL_NZ"));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMxQuantParams(const aclTensor *x, const aclTensor *groupList, const aclTensorList *&weight,
                               const aclTensorList *&weightScale, const aclTensorList *bias,
                               const aclTensor *xScaleOptional, bool &transposeWeight, int64_t groupListType,
                               int64_t effectiveYDtype, const char *roundMode, const char *&effectiveRoundMode,
                               int64_t scaleAlg, float dstTypeMax, const aclTensor *y, const aclTensor *yScale,
                               aclOpExecutor *executor)
{
    GMMAQ_CHECK_WITH_LOG(
        xScaleOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "xScaleOptional", "does not support nullptr"));
    const aclTensor *w = (*weight)[gmaq::FIRST_TENSOR_INDEX];
    const aclTensor *wScale = (*weightScale)[gmaq::FIRST_TENSOR_INDEX];

    aclnnStatus checkRet =
        CheckMxQuantAttrs(groupListType, effectiveYDtype, roundMode, effectiveRoundMode, scaleAlg, dstTypeMax, bias);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    checkRet = CheckMxQuantDim(x, groupList, w, wScale, xScaleOptional, y, yScale);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    checkRet = CheckMxQuantDtype(x, w, xScaleOptional, wScale, y, yScale, effectiveYDtype);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    GMMAQ_CHECK_WITH_LOG(
        groupList->GetDataType() == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "groupList", op::ToString(groupList->GetDataType()).GetString(),
                                              "the dtype of groupList must be INT64"));
    checkRet = CheckMxQuantFormat(x, groupList, w, wScale, xScaleOptional, y, yScale);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    checkRet = CheckMXTranspose(x, xScaleOptional, weight, weightScale, transposeWeight);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    checkRet = NormalizeMXTranspose(weight, weightScale, transposeWeight, executor);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    w = (*weight)[gmaq::FIRST_TENSOR_INDEX];
    wScale = (*weightScale)[gmaq::FIRST_TENSOR_INDEX];
    checkRet = CheckMxQuantShape(x, groupList, xScaleOptional, w, wScale, transposeWeight, y, yScale);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor *x, const aclTensor *groupList, const aclTensorList *&weight,
                        const aclTensorList *&weightScale, const aclTensorList *bias, const aclTensor *xScaleOptional,
                        const char *activationType, bool &transposeWeight, int64_t groupListType, const char *quantMode,
                        const char *roundMode, int64_t scaleAlg, float dstTypeMax, const aclTensor *y,
                        const aclTensor *yScale, aclOpExecutor *executor, const char *&effectiveQuantMode,
                        const char *&effectiveRoundMode, int64_t &effectiveYDtype)
{
    aclnnStatus checkRet = CheckActivationType(activationType);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    checkRet = ResolveAndCheckQuantMode(x, xScaleOptional, quantMode, effectiveQuantMode);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    effectiveYDtype = static_cast<int64_t>(y->GetDataType());
    if (std::strcmp(effectiveQuantMode, QUANT_MODE_MX) == 0) {
        checkRet = CheckMxQuantParams(x, groupList, weight, weightScale, bias, xScaleOptional, transposeWeight,
                                      groupListType, effectiveYDtype, roundMode, effectiveRoundMode, scaleAlg,
                                      dstTypeMax, y, yScale, executor);
        CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "quantMode", effectiveQuantMode, "quantMode must be mx");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus PrepareTensorListContiguous(const aclTensorList *&tensorList, aclOpExecutor *executor)
{
    std::vector<const aclTensor *> tensorVec;
    tensorVec.reserve(tensorList->Size());
    for (size_t i = 0; i < tensorList->Size(); ++i) {
        const aclTensor *contiguousTensor = l0op::Contiguous((*tensorList)[i], executor);
        CHECK_RET(contiguousTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensorVec.push_back(contiguousTensor);
    }
    tensorList = executor->AllocTensorList(tensorVec.data(), tensorVec.size());
    CHECK_RET(tensorList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
    const aclTensor *x, const aclTensor *groupList, const aclTensorList *weight, const aclTensorList *weightScale,
    const aclTensorList *bias, const aclTensor *xScaleOptional, const char *activationType, int64_t groupListType,
    const aclIntArray *tuningConfig, const char *quantMode, const char *roundMode, int64_t scaleAlg, float dstTypeMax,
    aclTensor *y, aclTensor *yScale, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    aclnnStatus checkRet = CheckRequiredParams(x, groupList, weight, weightScale, activationType, y, yScale);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    bool transposeWeight = false;

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    const char *effectiveQuantMode = nullptr;
    const char *effectiveRoundMode = nullptr;
    int64_t effectiveYDtype = 0L;
    checkRet = CheckParams(x, groupList, weight, weightScale, bias, xScaleOptional, activationType, transposeWeight,
                           groupListType, quantMode, roundMode, scaleAlg, dstTypeMax, y, yScale, uniqueExecutor.get(),
                           effectiveQuantMode, effectiveRoundMode, effectiveYDtype);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    // Required external inputs are checked before DFX traverses them.
    L2_DFX_PHASE_1(aclnnGroupedMatmulActivationQuantWeightNz,
                   DFX_IN(x, groupList, weight, weightScale, bias, xScaleOptional, activationType, groupListType,
                          tuningConfig, quantMode, roundMode, scaleAlg, dstTypeMax),
                   DFX_OUT(y, yScale));

    if (y->IsEmpty() || groupList->IsEmpty() || yScale->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    x = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(x != nullptr, ACLNN_ERR_INNER_NULLPTR);
    xScaleOptional = l0op::Contiguous(xScaleOptional, uniqueExecutor.get());
    CHECK_RET(xScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    groupList = l0op::Contiguous(groupList, uniqueExecutor.get());
    CHECK_RET(groupList != nullptr, ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(PrepareTensorListContiguous(weightScale, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    NormalizeEmptyTensorList(bias);
    CHECK_RET(CreateEmptyTensorList(aclDataType::ACL_FLOAT, bias, uniqueExecutor.get()) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(WrapWeightNzTensorList(weight, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::GroupedMatmulActivationQuant(x, groupList, weight, weightScale, bias, xScaleOptional,
                                                     activationType, transposeWeight, groupListType, tuningConfig,
                                                     effectiveQuantMode, effectiveYDtype, effectiveRoundMode, scaleAlg,
                                                     dstTypeMax, uniqueExecutor.get());
    CHECK_RET(result != std::tuple(nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);
    auto yResult = l0op::ViewCopy(std::get<0>(result), y, uniqueExecutor.get());
    CHECK_RET(yResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto yScaleResult = l0op::ViewCopy(std::get<1>(result), yScale, uniqueExecutor.get());
    CHECK_RET(yScaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulActivationQuantWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                      aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulActivationQuantWeightNz);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "An error occurred when aclnnGroupedMatmulActivationQuantWeightNz launched on aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
