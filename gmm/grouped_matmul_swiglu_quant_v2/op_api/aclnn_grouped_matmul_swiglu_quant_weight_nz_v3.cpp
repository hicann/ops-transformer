/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <vector>
#include "../../common/op_api/gmm_tensor_storage_check.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "grouped_matmul_swiglu_quant_v2.h"
#include "grouped_matmul_swiglu_quant_api_check_utils.h"
#include "mc2_log_compat.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "util/math_util.h"

using namespace op;

namespace {
constexpr int64_t DIMENSION_STEP = 1;
constexpr int64_t CONTIGUOUS_ELEMENT_STRIDE = 1;
constexpr size_t X_K_DIM = 1;
constexpr size_t BATCHED_MATRIX_ROW_DIM = 1;
constexpr size_t BATCHED_MATRIX_COLUMN_DIM = 2;
constexpr size_t SCALE_PAIR_DIM = 3;
constexpr size_t EMPTY_TENSOR_RANK = 1;
constexpr int64_t MIN_GROUP_NUM = 1;
constexpr int64_t SPLIT_SWIGLU_MODE = 2;
constexpr int64_t GROUP_LIST_COUNTS = 1;
constexpr size_t OUTPUT_SCALE_RESULT_INDEX = 1;

constexpr char API_NAME[] = "aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize";
constexpr char EXECUTE_API_NAME[] = "aclnnGroupedMatmulSwigluQuantWeightNzV3";

#define GMM_SWIGLU_V3_CHECK_WITH_LOG(cond, retCode, logExpr) \
    do { \
        if (!(cond)) { \
            logExpr; \
            return retCode; \
        } \
    } while (0)

bool IsMxfp8(DataType dtype)
{
    return dtype == DataType::DT_FLOAT8_E4M3FN;
}

bool IsNzFormat(const aclTensor* tensor)
{
    return tensor != nullptr && ge::GetPrimaryFormat(tensor->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ;
}

bool IsTransposedWeight(const aclTensor* tensor)
{
    if (tensor == nullptr || tensor->GetViewShape().GetDimNum() != gmm_swiglu_quant_v3::WEIGHT_VIEW_DIM_NUM ||
        tensor->GetStorageShape().GetDimNum() != gmm_swiglu_quant_v3::WEIGHT_STORAGE_DIM_NUM) {
        return false;
    }
    const auto& viewShape = tensor->GetViewShape();
    const auto strides = tensor->GetViewStrides();
    const int64_t lastDim = static_cast<int64_t>(viewShape.GetDimNum()) - DIMENSION_STEP;
    const int64_t penultimateDim = lastDim - DIMENSION_STEP;

    // Match GMMAQ and GMM WeightNZ: the transposed view is identified by
    // stride (K * N, 1, K), rather than by dimensions whose values can be
    // equal or whose order is normalized by the Torch/NPU descriptor.
    if (strides[penultimateDim] != CONTIGUOUS_ELEMENT_STRIDE || strides[lastDim] != viewShape.GetDim(penultimateDim)) {
        return false;
    }
    int64_t expectedStride = viewShape.GetDim(lastDim) * viewShape.GetDim(penultimateDim);
    for (int64_t batchDim = penultimateDim - DIMENSION_STEP; batchDim >= 0; --batchDim) {
        if (strides[batchDim] != expectedStride) {
            return false;
        }
        expectedStride *= viewShape.GetDim(batchDim);
    }
    return true;
}

bool IsTransposedWeightScale(const aclTensor* tensor)
{
    if (tensor == nullptr || tensor->GetViewShape().GetDimNum() != gmm_swiglu_quant_v3::WEIGHT_SCALE_DIM_NUM) {
        return false;
    }
    const auto& shape = tensor->GetViewShape();
    const auto strides = tensor->GetViewStrides();
    return strides[SCALE_PAIR_DIM] == CONTIGUOUS_ELEMENT_STRIDE &&
           strides[BATCHED_MATRIX_ROW_DIM] == gmm_swiglu_quant_v3::MX_SCALE_PAIR &&
           strides[BATCHED_MATRIX_COLUMN_DIM] ==
               shape.GetDim(BATCHED_MATRIX_ROW_DIM) * gmm_swiglu_quant_v3::MX_SCALE_PAIR;
}

aclnnStatus CheckTransposeConsistency(const aclTensorList* weight, const aclTensorList* weightScale,
                                      bool& transposeWeight)
{
    transposeWeight = IsTransposedWeight((*weight)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX]);
    const bool transposeWeightScale = IsTransposedWeightScale((*weightScale)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX]);
    GMM_SWIGLU_V3_CHECK_WITH_LOG(transposeWeight == transposeWeightScale, ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                     API_NAME, "weightScale", transposeWeightScale ? "transposed" : "non-transposed",
                                     "the transposition of weightScale must be equal to the transposition of weight"));
    return ACLNN_SUCCESS;
}

aclnnStatus NormalizeTransposedTensorList(const aclTensorList* input, const aclTensorList*& output, bool isWeightScale,
                                          aclOpExecutor* executor)
{
    std::vector<aclTensor*> tensors;
    tensors.reserve(input->Size());
    for (size_t index = 0; index < input->Size(); ++index) {
        const aclTensor* tensor = (*input)[index];
        const auto& viewShape = tensor->GetViewShape();
        op::Shape normalizedShape;
        normalizedShape.SetScalar();
        normalizedShape.AppendDim(viewShape.GetDim(0));
        if (isWeightScale) {
            normalizedShape.AppendDim(viewShape.GetDim(BATCHED_MATRIX_COLUMN_DIM));
            normalizedShape.AppendDim(viewShape.GetDim(BATCHED_MATRIX_ROW_DIM));
            normalizedShape.AppendDim(viewShape.GetDim(SCALE_PAIR_DIM));
        } else {
            normalizedShape.AppendDim(viewShape.GetDim(BATCHED_MATRIX_COLUMN_DIM));
            normalizedShape.AppendDim(viewShape.GetDim(BATCHED_MATRIX_ROW_DIM));
        }
        aclTensor* normalized = executor->CreateView(tensor, normalizedShape, tensor->GetViewOffset());
        CHECK_RET(normalized != nullptr, ACLNN_ERR_INNER_NULLPTR);
        normalized->SetViewFormat(tensor->GetViewFormat());
        normalized->SetOriginalFormat(tensor->GetOriginalFormat());
        normalized->SetStorageFormat(tensor->GetStorageFormat());
        normalized->SetStorageShape(tensor->GetStorageShape());
        tensors.emplace_back(normalized);
    }
    output = executor->AllocTensorList(tensors.data(), tensors.size());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus NormalizeMXTranspose(const aclTensorList* weight, const aclTensorList* weightScale, bool transposeWeight,
                                 const aclTensorList*& normalizedWeight, const aclTensorList*& normalizedWeightScale,
                                 aclOpExecutor* executor)
{
    normalizedWeight = weight;
    normalizedWeightScale = weightScale;
    if (!transposeWeight) {
        return ACLNN_SUCCESS;
    }
    CHECK_RET(NormalizeTransposedTensorList(weight, normalizedWeight, false, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(NormalizeTransposedTensorList(weightScale, normalizedWeightScale, true, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

bool IsEmptyTensorList(const aclTensorList* tensorList)
{
    if (tensorList == nullptr || tensorList->Size() == 0) {
        return true;
    }
    if (tensorList->Size() != gmm_swiglu_quant_v3::SINGLE_TENSOR_SIZE) {
        return false;
    }
    const aclTensor* tensor = (*tensorList)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX];
    if (tensor == nullptr) {
        return true;
    }
    const auto& shape = tensor->GetViewShape();
    return shape.GetDimNum() == EMPTY_TENSOR_RANK && shape.GetDim(0) == 0;
}

aclnnStatus CheckOptionalInputs(const aclTensorList* weightAssistMatrix, const aclTensor* bias,
                                const aclTensor* smoothScale, const aclIntArray* tuningConfigOptional)
{
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        IsEmptyTensorList(weightAssistMatrix), ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "weightAssistMatrix",
                                                 "only nullptr or an empty tensorList is supported"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        bias == nullptr, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "bias", "only nullptr is supported"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        smoothScale == nullptr, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "smoothScale", "only nullptr is supported"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(tuningConfigOptional == nullptr || tuningConfigOptional->Size() == 0,
                                 ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                                     API_NAME, "tuningConfigOptional", "only nullptr or an empty array is supported"));
    return ACLNN_SUCCESS;
}

bool IsRepresentableFloat(double value)
{
    return std::isfinite(value) && std::fabs(value) <= static_cast<double>(std::numeric_limits<float>::max());
}

aclnnStatus CheckSwigluAttrs(int64_t swigluMode, double clampLimit, double gluAlpha, double gluBias,
                             const char* roundMode, int64_t scaleAlg, double dstTypeMax)
{
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        swigluMode == SPLIT_SWIGLU_MODE, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "swigluMode", std::to_string(swigluMode),
                                              "currently only 2 is supported"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        IsRepresentableFloat(clampLimit), ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "clampLimit", std::to_string(clampLimit),
                                              "must be finite and representable by float32"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        clampLimit > 0.0, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "clampLimit", std::to_string(clampLimit),
                                              "must be greater than 0 for swigluMode 2"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(IsRepresentableFloat(gluAlpha), ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "gluAlpha", std::to_string(gluAlpha),
                                                                       "must be finite and representable by float32"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(IsRepresentableFloat(gluBias), ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "gluBias", std::to_string(gluBias),
                                                                       "must be finite and representable by float32"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        roundMode == nullptr || roundMode[0] == '\0' ||
            std::strcmp(roundMode, gmm_swiglu_quant_v3::DEFAULT_ROUND_MODE) == 0,
        ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "roundMode", roundMode == nullptr ? "nullptr" : roundMode,
                                              "MXFP8 currently only supports rint"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        scaleAlg == gmm_swiglu_quant_v3::SCALE_ALG_OCP || scaleAlg == gmm_swiglu_quant_v3::SCALE_ALG_CUBLAS,
        ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            API_NAME, "scaleAlg", std::to_string(scaleAlg),
            "MXFP8 only supports scaleAlg 0 or 1; scaleAlg 2 is only supported for FLOAT4_E2M1"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        std::isfinite(dstTypeMax) && dstTypeMax == gmm_swiglu_quant_v3::DEFAULT_DST_TYPE_MAX, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "dstTypeMax", std::to_string(dstTypeMax),
                                              "MXFP8 only supports dstTypeMax 0.0"));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGmmAttrs(int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t groupListType)
{
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        dequantMode == gmm_swiglu_quant_v3::QUANT_MODE_MX && quantMode == gmm_swiglu_quant_v3::QUANT_MODE_MX &&
            dequantDtype == gmm_swiglu_quant_v3::DEQUANT_DTYPE_FLOAT,
        ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(API_NAME, "dequantMode, dequantDtype, quantMode",
                                               ("{" + std::to_string(dequantMode) + ", " +
                                                std::to_string(dequantDtype) + ", " + std::to_string(quantMode) + "}"),
                                               "MXFP8 requires dequantMode=2, dequantDtype=DT_FLOAT and quantMode=2"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(groupListType == 0 || groupListType == GROUP_LIST_COUNTS, ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                     API_NAME, "groupListType", std::to_string(groupListType), "must be 0 or 1"));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckTensorDimNum(const aclTensor* tensor, const char* name, size_t expectedDimNum)
{
    const size_t actualDimNum = tensor->GetViewShape().GetDimNum();
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        actualDimNum == expectedDimNum, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            API_NAME, name, std::to_string(actualDimNum),
            (std::string("the dim num of ") + name + " must be " + std::to_string(expectedDimNum)).c_str()));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDimensionNumbers(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale,
                                  const aclTensor* xScale, const aclTensor* groupList, const aclTensor* output,
                                  const aclTensor* outputScale)
{
    GMM_SWIGLU_V3_CHECK_WITH_LOG(weight->Size() == gmm_swiglu_quant_v3::SINGLE_TENSOR_SIZE &&
                                     weightScale->Size() == gmm_swiglu_quant_v3::SINGLE_TENSOR_SIZE,
                                 ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                                     API_NAME, "weight, weightScale",
                                     (std::to_string(weight->Size()) + ", " + std::to_string(weightScale->Size())),
                                     "only one weight tensor and one weightScale tensor are supported"));
    CHECK_RET(CheckTensorDimNum(x, "x", gmm_swiglu_quant_v3::X_DIM_NUM) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimNum((*weight)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX], "weight",
                                gmm_swiglu_quant_v3::WEIGHT_VIEW_DIM_NUM) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimNum((*weightScale)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX], "weightScale",
                                gmm_swiglu_quant_v3::WEIGHT_SCALE_DIM_NUM) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimNum(xScale, "xScale", gmm_swiglu_quant_v3::X_SCALE_DIM_NUM) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimNum(groupList, "groupList", gmm_swiglu_quant_v3::GROUP_LIST_DIM_NUM) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimNum(output, "output", gmm_swiglu_quant_v3::OUTPUT_DIM_NUM) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimNum(outputScale, "outputScale", gmm_swiglu_quant_v3::OUTPUT_SCALE_DIM_NUM) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        (*weight)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX]->GetStorageShape().GetDimNum() ==
            gmm_swiglu_quant_v3::WEIGHT_STORAGE_DIM_NUM,
        ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(API_NAME, "weight storageShape",
                                                 std::to_string((*weight)[0]->GetStorageShape().GetDimNum()),
                                                 "the dim num of weight storageShape must be 5"));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtype(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale,
                       const aclTensor* xScale, const aclTensor* groupList, const aclTensor* output,
                       const aclTensor* outputScale)
{
    const aclTensor* weightTensor = (*weight)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX];
    const aclTensor* weightScaleTensor = (*weightScale)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX];
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        IsMxfp8(x->GetDataType()), ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(API_NAME, "x", op::ToString(x->GetDataType()).GetString(),
                                              "x must be FLOAT8_E4M3FN"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        IsMxfp8(weightTensor->GetDataType()), ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(API_NAME, "weight", op::ToString(weightTensor->GetDataType()).GetString(),
                                              "weight must be FLOAT8_E4M3FN"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        xScale->GetDataType() == DataType::DT_FLOAT8_E8M0, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(API_NAME, "xScale", op::ToString(xScale->GetDataType()).GetString(),
                                              "xScale must be FLOAT8_E8M0"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        weightScaleTensor->GetDataType() == DataType::DT_FLOAT8_E8M0, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(API_NAME, "weightScale",
                                              op::ToString(weightScaleTensor->GetDataType()).GetString(),
                                              "weightScale must be FLOAT8_E8M0"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        groupList->GetDataType() == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(API_NAME, "groupList", op::ToString(groupList->GetDataType()).GetString(),
                                              "groupList must be INT64"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        output->GetDataType() == DataType::DT_FLOAT8_E4M3FN, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(API_NAME, "output", op::ToString(output->GetDataType()).GetString(),
                                              "WeightNZ MXFP8 output must be FLOAT8_E4M3FN"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(outputScale->GetDataType() == DataType::DT_FLOAT8_E8M0, ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                                     API_NAME, "outputScale", op::ToString(outputScale->GetDataType()).GetString(),
                                     "outputScale must be FLOAT8_E8M0"));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckFormat(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale,
                        const aclTensor* xScale, const aclTensor* groupList, const aclTensor* output,
                        const aclTensor* outputScale)
{
    const struct {
        const aclTensor* tensor;
        const char* name;
    } ndTensors[] = {{x, "x"},           {(*weightScale)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX], "weightScale"},
                     {xScale, "xScale"}, {groupList, "groupList"},
                     {output, "output"}, {outputScale, "outputScale"}};
    for (const auto& item : ndTensors) {
        GMM_SWIGLU_V3_CHECK_WITH_LOG(!op::IsPrivateFormat(item.tensor->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
                                     OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                                         API_NAME, item.name, op::ToString(item.tensor->GetStorageFormat()).GetString(),
                                         "the format must be ND"));
    }
    const aclTensor* weightTensor = (*weight)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX];
    GMM_SWIGLU_V3_CHECK_WITH_LOG(IsNzFormat(weightTensor), ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                                     API_NAME, "weight", op::ToString(weightTensor->GetStorageFormat()).GetString(),
                                     "the format must be FRACTAL_NZ"));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckShape(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale,
                       const aclTensor* xScale, const aclTensor* groupList, const aclTensor* output,
                       const aclTensor* outputScale, bool transposeWeight)
{
    const aclTensor* weightTensor = (*weight)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX];
    const aclTensor* weightScaleTensor = (*weightScale)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX];
    const int64_t m = x->GetViewShape().GetDim(0);
    const int64_t k = x->GetViewShape().GetDim(X_K_DIM);
    const int64_t e = groupList->GetViewShape().GetDim(0);
    const int64_t weightE = weightTensor->GetViewShape().GetDim(0);
    const int64_t n = transposeWeight ? weightTensor->GetViewShape().GetDim(BATCHED_MATRIX_ROW_DIM) :
                                        weightTensor->GetViewShape().GetDim(BATCHED_MATRIX_COLUMN_DIM);
    const int64_t weightK = transposeWeight ? weightTensor->GetViewShape().GetDim(BATCHED_MATRIX_COLUMN_DIM) :
                                              weightTensor->GetViewShape().GetDim(BATCHED_MATRIX_ROW_DIM);

    GMM_SWIGLU_V3_CHECK_WITH_LOG(e >= MIN_GROUP_NUM && e <= gmm_swiglu_quant_v3::MAX_GROUP_NUM, ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "groupList dim 0", std::to_string(e),
                                                                       "must be in range [1, 1024]"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(m > 0 && k > 0, ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                                     API_NAME, "M, K", ("{" + std::to_string(m) + ", " + std::to_string(k) + "}"),
                                     "M and K must both be greater than 0"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        n > 0 && n % gmm_swiglu_quant_v3::MXFP8_N_ALIGN == 0 && n % gmm_swiglu_quant_v3::SWIGLU_SPLIT_FACTOR == 0,
        ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "N", std::to_string(n),
                                              "N must be greater than 0 and divisible by 64"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        weightE == e, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            API_NAME, "groupList length", std::to_string(e),
            (std::string("must be equal to the first dimension of the single weight tensor, got ") +
             std::to_string(weightE))
                .c_str()));

    const op::Shape xExpected = {m, k};
    const op::Shape xScaleExpected = {m, Ops::Base::CeilDiv(k, gmm_swiglu_quant_v3::MX_GROUP_SIZE),
                                      gmm_swiglu_quant_v3::MX_SCALE_PAIR};
    const op::Shape weightExpected = transposeWeight ? op::Shape({e, n, k}) : op::Shape({e, k, n});
    const op::Shape weightScaleExpected =
        transposeWeight ? op::Shape({e, n, Ops::Base::CeilDiv(k, gmm_swiglu_quant_v3::MX_GROUP_SIZE),
                                     gmm_swiglu_quant_v3::MX_SCALE_PAIR}) :
                          op::Shape({e, Ops::Base::CeilDiv(k, gmm_swiglu_quant_v3::MX_GROUP_SIZE), n,
                                     gmm_swiglu_quant_v3::MX_SCALE_PAIR});
    const int64_t nAfterSwiGlu = n / gmm_swiglu_quant_v3::SWIGLU_SPLIT_FACTOR;
    const op::Shape outputExpected = {m, nAfterSwiGlu};
    const op::Shape outputScaleExpected = {m, Ops::Base::CeilDiv(nAfterSwiGlu, gmm_swiglu_quant_v3::MX_GROUP_SIZE),
                                           gmm_swiglu_quant_v3::MX_SCALE_PAIR};
    const op::Shape weightStorageExpected =
        transposeWeight ? op::Shape({e, Ops::Base::CeilDiv(k, gmm_swiglu_quant_v3::NZ_C0_SIZE),
                                     Ops::Base::CeilDiv(n, gmm_swiglu_quant_v3::NZ_INNER_SIZE),
                                     gmm_swiglu_quant_v3::NZ_INNER_SIZE, gmm_swiglu_quant_v3::NZ_C0_SIZE}) :
                          op::Shape({e, Ops::Base::CeilDiv(n, gmm_swiglu_quant_v3::NZ_C0_SIZE),
                                     Ops::Base::CeilDiv(k, gmm_swiglu_quant_v3::NZ_INNER_SIZE),
                                     gmm_swiglu_quant_v3::NZ_INNER_SIZE, gmm_swiglu_quant_v3::NZ_C0_SIZE});

    const struct {
        const aclTensor* tensor;
        const char* name;
        const op::Shape* expected;
    } shapes[] = {{x, "x", &xExpected},
                  {xScale, "xScale", &xScaleExpected},
                  {weightTensor, "weight", &weightExpected},
                  {weightScaleTensor, "weightScale", &weightScaleExpected},
                  {output, "output", &outputExpected},
                  {outputScale, "outputScale", &outputScaleExpected}};
    for (const auto& item : shapes) {
        GMM_SWIGLU_V3_CHECK_WITH_LOG(
            item.tensor->GetViewShape() == *item.expected, ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                API_NAME, item.name, op::ToString(item.tensor->GetViewShape()).GetString(),
                (std::string("the shape must be ") + op::ToString(*item.expected).GetString()).c_str()));
    }
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        weightTensor->GetStorageShape() == weightStorageExpected, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            API_NAME, "weight storageShape", op::ToString(weightTensor->GetStorageShape()).GetString(),
            (std::string("the storage shape must be ") + op::ToString(weightStorageExpected).GetString()).c_str()));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(weightK == k, ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                     API_NAME, "weight", std::to_string(weightK),
                                     (std::string("the logical K dimension must be ") + std::to_string(k)).c_str()));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckHostGroupListValues(const aclTensor* x, const aclTensor* groupList, int64_t groupListType)
{
    if (!gert::TensorPlacementUtils::IsOnHost(groupList->GetPlacement())) {
        return ACLNN_SUCCESS;
    }
    const auto* data = static_cast<const int64_t*>(groupList->GetData());
    if (data == nullptr) {
        return ACLNN_SUCCESS;
    }
    const int64_t m = x->GetViewShape().GetDim(0);
    const int64_t groupNum = groupList->GetViewShape().GetDim(0);
    const int64_t groupStride = groupList->GetViewStrides()[0];
    int64_t previous = 0;
    int64_t sum = 0;
    for (int64_t index = 0; index < groupNum; ++index) {
        const int64_t value = data[index * groupStride];
        GMM_SWIGLU_V3_CHECK_WITH_LOG(value >= 0, ACLNN_ERR_PARAM_INVALID,
                                     OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "groupList", std::to_string(value),
                                                                           "values must be non-negative"));
        if (groupListType == 0) {
            GMM_SWIGLU_V3_CHECK_WITH_LOG(value >= previous && value <= m, ACLNN_ERR_PARAM_INVALID,
                                         OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                             API_NAME, "groupList", std::to_string(value),
                                             "groupListType 0 requires a non-decreasing sequence not greater than M"));
            previous = value;
        } else {
            GMM_SWIGLU_V3_CHECK_WITH_LOG(
                value <= m - sum, ACLNN_ERR_PARAM_INVALID,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "groupList", std::to_string(value),
                                                      "groupListType 1 requires the sum not greater than M"));
            sum += value;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale,
                        const aclTensorList* weightAssistMatrix, const aclTensor* bias, const aclTensor* xScale,
                        const aclTensor* smoothScale, const aclTensor* groupList, int64_t dequantMode,
                        int64_t dequantDtype, int64_t quantMode, int64_t groupListType,
                        const aclIntArray* tuningConfigOptional, int64_t swigluMode, double clampLimit, double gluAlpha,
                        double gluBias, const char* roundMode, int64_t scaleAlg, double dstTypeMax,
                        const aclTensor* output, const aclTensor* outputScale, bool transposeWeight)
{
    CHECK_RET(CheckOptionalInputs(weightAssistMatrix, bias, smoothScale, tuningConfigOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        CheckSwigluAttrs(swigluMode, clampLimit, gluAlpha, gluBias, roundMode, scaleAlg, dstTypeMax) == ACLNN_SUCCESS,
        ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGmmAttrs(dequantMode, dequantDtype, quantMode, groupListType) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDimensionNumbers(x, weight, weightScale, xScale, groupList, output, outputScale) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(x, weight, weightScale, xScale, groupList, output, outputScale) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    GMM_SWIGLU_V3_CHECK_WITH_LOG(op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510,
                                 ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Only Ascend 950 is supported by %s.", API_NAME));
    CHECK_RET(CheckFormat(x, weight, weightScale, xScale, groupList, output, outputScale) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        CheckShape(x, weight, weightScale, xScale, groupList, output, outputScale, transposeWeight) == ACLNN_SUCCESS,
        ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckHostGroupListValues(x, groupList, groupListType) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus CreateEmptyWeightAssistTensorList(const aclTensorList*& weightAssistMatrix, aclOpExecutor* executor)
{
    if (!IsEmptyTensorList(weightAssistMatrix)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    std::vector<aclTensor*> tensors;
    tensors.emplace_back(executor->AllocTensor({0}, DataType::DT_FLOAT));
    CHECK_RET(tensors[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    weightAssistMatrix = executor->AllocTensorList(tensors.data(), tensors.size());
    CHECK_RET(weightAssistMatrix != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus PrepareContiguousTensorList(const aclTensorList* tensorList, const aclTensorList*& newTensorList,
                                        aclOpExecutor* executor)
{
    std::vector<const aclTensor*> tensors;
    tensors.reserve(tensorList->Size());
    for (size_t index = 0; index < tensorList->Size(); ++index) {
        const aclTensor* tensor = (*tensorList)[index];
        const aclTensor* contiguousTensor = l0op::Contiguous(tensor, executor);
        CHECK_RET(contiguousTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // Contiguous may materialize an ND scale as a default-format view.
        // Restore the ND descriptor while retaining the contiguous buffer.
        aclTensor* ndTensor =
            executor->CreateView(contiguousTensor, contiguousTensor->GetViewShape(), contiguousTensor->GetViewOffset());
        CHECK_RET(ndTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        ndTensor->SetViewFormat(op::Format::FORMAT_ND);
        ndTensor->SetOriginalFormat(op::Format::FORMAT_ND);
        ndTensor->SetStorageFormat(op::Format::FORMAT_ND);
        ndTensor->SetStorageShape(contiguousTensor->GetStorageShape());
        tensors.emplace_back(ndTensor);
    }
    newTensorList = executor->AllocTensorList(tensors.data(), tensors.size());
    CHECK_RET(newTensorList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus WrapWeightNzTensorList(const aclTensorList* weight, const aclTensorList*& wrappedWeight,
                                   aclOpExecutor* executor)
{
    std::vector<const aclTensor*> tensors;
    tensors.reserve(weight->Size());
    for (size_t index = 0; index < weight->Size(); ++index) {
        const aclTensor* inputTensor = (*weight)[index];
        const op::Shape storageShape = inputTensor->GetStorageShape();
        aclTensor* wrappedTensor =
            executor->CreateView(inputTensor, inputTensor->GetViewShape(), inputTensor->GetViewOffset());
        CHECK_RET(wrappedTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        wrappedTensor->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
        wrappedTensor->SetOriginalFormat(inputTensor->GetViewFormat());
        wrappedTensor->SetViewShape(inputTensor->GetViewShape());
        wrappedTensor->SetStorageShape(storageShape);
        tensors.emplace_back(wrappedTensor);
    }
    wrappedWeight = executor->AllocTensorList(tensors.data(), tensors.size());
    CHECK_RET(wrappedWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale,
    const aclTensorList* weightAssistMatrix, const aclTensor* bias, const aclTensor* xScale,
    const aclTensor* smoothScale, const aclTensor* groupList, int64_t dequantMode, int64_t dequantDtype,
    int64_t quantMode, int64_t groupListType, const aclIntArray* tuningConfigOptional, int64_t swigluMode,
    double clampLimit, double gluAlpha, double gluBias, const char* roundMode, int64_t scaleAlg, double dstTypeMax,
    aclTensor* output, aclTensor* outputScale, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    aclnnStatus status =
        gmm_swiglu_quant_api_check::CheckRequiredInputs(API_NAME, x, weight, weightScale, xScale, groupList, output,
                                                        outputScale, gmm_swiglu_quant_v3::SINGLE_TENSOR_SIZE);
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    CHECK_RET(gmm::CheckTensorStorageBounds(x, "GroupedMatmulSwigluQuant", "x"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorListStorageBounds(weight, "GroupedMatmulSwigluQuant", "weight"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorListStorageBounds(weightScale, "GroupedMatmulSwigluQuant", "weightScale"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorListStorageBounds(weightAssistMatrix, "GroupedMatmulSwigluQuant", "weightAssistMatrix"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(bias, "GroupedMatmulSwigluQuant", "bias"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(xScale, "GroupedMatmulSwigluQuant", "xScale"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(smoothScale, "GroupedMatmulSwigluQuant", "smoothScale"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(groupList, "GroupedMatmulSwigluQuant", "groupList"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(output, "GroupedMatmulSwigluQuant", "output"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(outputScale, "GroupedMatmulSwigluQuant", "outputScale"),
              ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    bool transposeWeight = false;
    CHECK_RET(CheckTransposeConsistency(weight, weightScale, transposeWeight) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    const aclTensorList* normalizedWeight = nullptr;
    const aclTensorList* normalizedWeightScale = nullptr;
    CHECK_RET(NormalizeMXTranspose(weight, weightScale, transposeWeight, normalizedWeight, normalizedWeightScale,
                                   uniqueExecutor.get()) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckParams(x, normalizedWeight, normalizedWeightScale, weightAssistMatrix, bias, xScale, smoothScale,
                          groupList, dequantMode, dequantDtype, quantMode, groupListType, tuningConfigOptional,
                          swigluMode, clampLimit, gluAlpha, gluBias, roundMode, scaleAlg, dstTypeMax, output,
                          outputScale, transposeWeight) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    L2_DFX_PHASE_1(aclnnGroupedMatmulSwigluQuantWeightNzV3,
                   DFX_IN(x, weight, weightScale, weightAssistMatrix, bias, xScale, smoothScale, groupList, dequantMode,
                          dequantDtype, quantMode, groupListType, tuningConfigOptional, swigluMode, clampLimit,
                          gluAlpha, gluBias, roundMode, scaleAlg, dstTypeMax),
                   DFX_OUT(output, outputScale));

    const aclTensor* xForOp = l0op::Contiguous(x, uniqueExecutor.get());
    const aclTensor* xScaleForOp = l0op::Contiguous(xScale, uniqueExecutor.get());
    const aclTensor* groupListForOp = l0op::Contiguous(groupList, uniqueExecutor.get());
    CHECK_RET(xForOp != nullptr && xScaleForOp != nullptr && groupListForOp != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensorList* weightScaleForOp = nullptr;
    CHECK_RET(
        PrepareContiguousTensorList(normalizedWeightScale, weightScaleForOp, uniqueExecutor.get()) == ACLNN_SUCCESS,
        ACLNN_ERR_INNER_NULLPTR);
    const aclTensorList* weightForOp = nullptr;
    CHECK_RET(WrapWeightNzTensorList(normalizedWeight, weightForOp, uniqueExecutor.get()) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    const aclTensorList* weightAssistForOp = weightAssistMatrix;
    CHECK_RET(CreateEmptyWeightAssistTensorList(weightAssistForOp, uniqueExecutor.get()) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    const char* effectiveRoundMode =
        (roundMode == nullptr || roundMode[0] == '\0') ? gmm_swiglu_quant_v3::DEFAULT_ROUND_MODE : roundMode;
    auto result = l0op::GroupedMatmulSwigluQuantV2(
        xForOp, weightForOp, weightScaleForOp, xScaleForOp, weightAssistForOp, bias, smoothScale, groupListForOp,
        dequantMode, dequantDtype, quantMode, static_cast<int64_t>(output->GetDataType()), transposeWeight,
        groupListType, nullptr, swigluMode, static_cast<float>(clampLimit), static_cast<float>(gluAlpha),
        static_cast<float>(gluBias), effectiveRoundMode, scaleAlg, static_cast<float>(dstTypeMax),
        uniqueExecutor.get());
    CHECK_RET(result != std::tuple(nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* outputResult = l0op::ViewCopy(std::get<0>(result), output, uniqueExecutor.get());
    const aclTensor* outputScaleResult =
        l0op::ViewCopy(std::get<OUTPUT_SCALE_RESULT_INDEX>(result), outputScale, uniqueExecutor.get());
    CHECK_RET(outputResult != nullptr && outputScaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                    aclrtStream stream)
{
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(EXECUTE_API_NAME, "executor", "does not support nullptr"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        workspaceSize == 0 || workspace != nullptr, ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(EXECUTE_API_NAME, "workspace",
                                                 "must not be nullptr when workspaceSize is greater than 0"));
    L2_DFX_PHASE_2(aclnnGroupedMatmulSwigluQuantWeightNzV3);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in GroupedMatmulSwigluQuantWeightNzV3 launch aicore");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
