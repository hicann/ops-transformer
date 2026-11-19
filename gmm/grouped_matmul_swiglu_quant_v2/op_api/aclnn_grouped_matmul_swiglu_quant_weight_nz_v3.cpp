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
#include "grouped_matmul_activation_prepare.h"
#include "grouped_matmul_activation.h"

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
constexpr int64_t NO_CLAMP_SWIGLU_MODE = 0;
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

bool IsMxfp4A4W4(DataType xDtype, DataType weightDtype)
{
    return xDtype == DataType::DT_FLOAT4_E2M1 && weightDtype == DataType::DT_FLOAT4_E2M1;
}

bool IsMxA8W4(DataType xDtype, DataType weightDtype)
{
    return IsMxfp8(xDtype) && weightDtype == DataType::DT_FLOAT4_E2M1;
}

bool IsA4W4Bf16Output(const aclTensor* x, const aclTensorList* weight, const aclTensor* output)
{
    return x != nullptr && weight != nullptr && weight->Size() == gmm_swiglu_quant_v3::SINGLE_TENSOR_SIZE &&
           (*weight)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX] != nullptr && output != nullptr &&
           IsMxfp4A4W4(x->GetDataType(), (*weight)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX]->GetDataType()) &&
           output->GetDataType() == DataType::DT_BF16;
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
    // Empty Torch views may retain a nonzero batch stride although K*N is
    // zero. No batch is accessed; the matrix strides still identify transpose.
    if (tensor->IsEmpty()) {
        return true;
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
        normalizedShape.AppendDim(viewShape.GetDim(BATCHED_MATRIX_COLUMN_DIM));
        normalizedShape.AppendDim(viewShape.GetDim(BATCHED_MATRIX_ROW_DIM));
        if (isWeightScale) {
            normalizedShape.AppendDim(viewShape.GetDim(SCALE_PAIR_DIM));
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
                                const aclIntArray* tuningConfigOptional)
{
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        IsEmptyTensorList(weightAssistMatrix), ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "weightAssistMatrix",
                                                 "weightAssistMatrix must be nullptr or an empty tensorList"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(bias == nullptr, ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "bias", "bias must be nullptr"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        tuningConfigOptional == nullptr || tuningConfigOptional->Size() == 0, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "tuningConfigOptional",
                                                 "tuningConfigOptional must be nullptr or an empty array"));
    return ACLNN_SUCCESS;
}

bool IsRepresentableFloat(double value)
{
    return std::isfinite(value) && std::fabs(value) <= static_cast<double>(std::numeric_limits<float>::max());
}

aclnnStatus CheckSwigluAttrs(int64_t swigluMode, double clampLimit, double gluAlpha, double gluBias,
                             const char* roundMode, int64_t scaleAlg, double dstTypeMax, bool allowNoClamp)
{
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        swigluMode == SPLIT_SWIGLU_MODE || (allowNoClamp && swigluMode == NO_CLAMP_SWIGLU_MODE),
        ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "swigluMode", std::to_string(swigluMode),
                                              "swigluMode must be 2, or 0 for FLOAT4_E2M1 A4W4"));
    if (swigluMode == SPLIT_SWIGLU_MODE) {
        GMM_SWIGLU_V3_CHECK_WITH_LOG(
            IsRepresentableFloat(clampLimit), ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "clampLimit", std::to_string(clampLimit),
                                                  "clampLimit must be finite and representable by float32"));
        GMM_SWIGLU_V3_CHECK_WITH_LOG(
            static_cast<float>(clampLimit) > 0.0F, ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                API_NAME, "clampLimit", std::to_string(clampLimit),
                "clampLimit must remain greater than 0 after conversion to float32 for swigluMode 2"));
        GMM_SWIGLU_V3_CHECK_WITH_LOG(
            IsRepresentableFloat(gluAlpha), ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "gluAlpha", std::to_string(gluAlpha),
                                                  "gluAlpha must be finite and representable by float32"));
        GMM_SWIGLU_V3_CHECK_WITH_LOG(
            IsRepresentableFloat(gluBias), ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "gluBias", std::to_string(gluBias),
                                                  "gluBias must be finite and representable by float32"));
    }
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        roundMode == nullptr || roundMode[0] == '\0' ||
            std::strcmp(roundMode, gmm_swiglu_quant_v3::DEFAULT_ROUND_MODE) == 0,
        ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "roundMode", roundMode == nullptr ? "nullptr" : roundMode,
                                              "roundMode must be nullptr, an empty string, or rint for MXFP8"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        scaleAlg == gmm_swiglu_quant_v3::SCALE_ALG_OCP || scaleAlg == gmm_swiglu_quant_v3::SCALE_ALG_CUBLAS,
        ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            API_NAME, "scaleAlg", std::to_string(scaleAlg),
            "scaleAlg must be 0 or 1 for MXFP8; scaleAlg 2 is only supported for FLOAT4_E2M1"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        std::isfinite(dstTypeMax) && dstTypeMax == gmm_swiglu_quant_v3::DEFAULT_DST_TYPE_MAX, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "dstTypeMax", std::to_string(dstTypeMax),
                                              "dstTypeMax must be 0.0 for MXFP8"));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGmmAttrs(int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t groupListType)
{
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        dequantMode == gmm_swiglu_quant_v3::QUANT_MODE_MX, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "dequantMode", std::to_string(dequantMode),
                                              "dequantMode must be 2 (MX quantization)"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        dequantDtype == gmm_swiglu_quant_v3::DEQUANT_DTYPE_FLOAT, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "dequantDtype", std::to_string(dequantDtype),
                                              "dequantDtype must be 0 (ACL_FLOAT)"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(quantMode == gmm_swiglu_quant_v3::QUANT_MODE_MX, ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "quantMode", std::to_string(quantMode),
                                                                       "quantMode must be 2 (MX quantization)"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        groupListType == 0 || groupListType == GROUP_LIST_COUNTS, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "groupListType", std::to_string(groupListType),
                                              "groupListType must be 0 or 1"));
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
                                     "weight and weightScale must each contain exactly one tensor"));
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
    if (outputScale != nullptr) {
        CHECK_RET(
            CheckTensorDimNum(outputScale, "outputScale", gmm_swiglu_quant_v3::OUTPUT_SCALE_DIM_NUM) == ACLNN_SUCCESS,
            ACLNN_ERR_PARAM_INVALID);
    }
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
    const bool isMxfp8 = IsMxfp8(x->GetDataType()) && IsMxfp8(weightTensor->GetDataType());
    const bool isMxfp4A4W4 = IsMxfp4A4W4(x->GetDataType(), weightTensor->GetDataType());
    const bool isMxA8W4 = IsMxA8W4(x->GetDataType(), weightTensor->GetDataType());
    GMM_SWIGLU_V3_CHECK_WITH_LOG(isMxfp8 || isMxfp4A4W4 || isMxA8W4, ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "x, weight",
                                                                          "must be FLOAT8_E4M3FN/FLOAT8_E4M3FN, "
                                                                          "FLOAT8_E4M3FN/FLOAT4_E2M1 or "
                                                                          "FLOAT4_E2M1/FLOAT4_E2M1"));
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
    if (isMxfp8 || isMxA8W4) {
        GMM_SWIGLU_V3_CHECK_WITH_LOG(
            output->GetDataType() == DataType::DT_FLOAT8_E4M3FN && outputScale != nullptr &&
                outputScale->GetDataType() == DataType::DT_FLOAT8_E8M0,
            ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "output, outputScale",
                                                     "output must be FLOAT8_E4M3FN and outputScale must be a "
                                                     "non-null FLOAT8_E8M0 tensor for MXFP8"));
    } else {
        GMM_SWIGLU_V3_CHECK_WITH_LOG(
            output->GetDataType() == DataType::DT_BF16 && outputScale == nullptr, ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "output, outputScale",
                                                     "output must be BF16 and outputScale must be nullptr for "
                                                     "FLOAT4_E2M1 A4W4"));
    }
    return ACLNN_SUCCESS;
}

bool IsDenseNdCompatibleFormat(op::Format format)
{
    // The Torch converter labels dense 3D/4D scale tensors NCL/NCHW.
    return format == op::Format::FORMAT_ND || format == op::Format::FORMAT_NCL || format == op::Format::FORMAT_NCHW;
}

aclnnStatus CheckFormat(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale,
                        const aclTensor* xScale, const aclTensor* groupList, const aclTensor* output,
                        const aclTensor* outputScale)
{
    const bool isMxA8W4 = IsMxA8W4(x->GetDataType(), (*weight)[0]->GetDataType());
    const struct {
        const aclTensor* tensor;
        const char* name;
    } ndTensors[] = {{x, "x"},           {(*weightScale)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX], "weightScale"},
                     {xScale, "xScale"}, {groupList, "groupList"},
                     {output, "output"}, {outputScale, "outputScale"}};
    for (const auto& item : ndTensors) {
        if (item.tensor == nullptr) {
            continue;
        }
        GMM_SWIGLU_V3_CHECK_WITH_LOG(isMxA8W4 ? IsDenseNdCompatibleFormat(item.tensor->GetStorageFormat()) :
                                                !op::IsPrivateFormat(item.tensor->GetStorageFormat()),
                                     ACLNN_ERR_PARAM_INVALID,
                                     OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                                         API_NAME, item.name, op::ToString(item.tensor->GetStorageFormat()).GetString(),
                                         isMxA8W4 ? "the format must be ND, NCL or NCHW" :
                                                    (std::string(item.name) + " must use ND format").c_str()));
    }
    const aclTensor* weightTensor = (*weight)[gmm_swiglu_quant_v3::FIRST_TENSOR_INDEX];
    GMM_SWIGLU_V3_CHECK_WITH_LOG(IsNzFormat(weightTensor), ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                                     API_NAME, "weight", op::ToString(weightTensor->GetStorageFormat()).GetString(),
                                     "weight must use FRACTAL_NZ format"));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckLogicalDimensions(int64_t m, int64_t k, int64_t e, int64_t weightE, int64_t n, bool isFp4Weight)
{
    GMM_SWIGLU_V3_CHECK_WITH_LOG(e >= MIN_GROUP_NUM && e <= gmm_swiglu_quant_v3::MAX_GROUP_NUM, ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "groupList dim 0", std::to_string(e),
                                                                       "groupList dim 0 must be in range [1, 1024]"));
    const bool validM = isFp4Weight ? m > 0 : m >= 0;
    GMM_SWIGLU_V3_CHECK_WITH_LOG(validM && k > 0, ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                                     API_NAME, "M, K", ("{" + std::to_string(m) + ", " + std::to_string(k) + "}"),
                                     isFp4Weight ? "M and K must both be greater than 0" :
                                                   "M must be non-negative and K must be greater than 0"));
    const int64_t nAlignment = isFp4Weight ? gmm_swiglu_quant_v3::MXFP4_N_ALIGN : gmm_swiglu_quant_v3::MXFP8_N_ALIGN;
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        n > 0 && n % nAlignment == 0 && n % gmm_swiglu_quant_v3::SWIGLU_SPLIT_FACTOR == 0, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "N", std::to_string(n),
                                              isFp4Weight ? "N must be greater than 0 and divisible by 128" :
                                                            "N must be greater than 0 and divisible by 64"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        !isFp4Weight || k % gmm_swiglu_quant_v3::MXFP4_K_ALIGN == 0, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "K", std::to_string(k),
                                              "K must be divisible by 128 for FLOAT4_E2M1 V3"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        weightE == e, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            API_NAME, "groupList length", std::to_string(e),
            (std::string("groupList length must be equal to the first dimension of the single weight tensor, got ") +
             std::to_string(weightE))
                .c_str()));

    return ACLNN_SUCCESS;
}

aclnnStatus CheckViewShape(const aclTensor* tensor, const char* name, const op::Shape& expected)
{
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        tensor->GetViewShape() == expected, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            API_NAME, name, op::ToString(tensor->GetViewShape()).GetString(),
            (std::string("the shape of ") + name + " must be " + op::ToString(expected).GetString()).c_str()));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckWeightStorageShape(const aclTensor* weightTensor, int64_t e, int64_t n, int64_t k, int64_t weightK,
                                    bool isFp4Weight, bool transposeWeight)
{
    const int64_t nzC0Size = isFp4Weight ? gmm_swiglu_quant_v3::MXFP4_NZ_C0_SIZE : gmm_swiglu_quant_v3::NZ_C0_SIZE;
    const op::Shape weightStorageExpected =
        transposeWeight ?
            op::Shape({e, Ops::Base::CeilDiv(k, nzC0Size), Ops::Base::CeilDiv(n, gmm_swiglu_quant_v3::NZ_INNER_SIZE),
                       gmm_swiglu_quant_v3::NZ_INNER_SIZE, nzC0Size}) :
            op::Shape({e, Ops::Base::CeilDiv(n, nzC0Size), Ops::Base::CeilDiv(k, gmm_swiglu_quant_v3::NZ_INNER_SIZE),
                       gmm_swiglu_quant_v3::NZ_INNER_SIZE, nzC0Size});

    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        weightTensor->GetStorageShape() == weightStorageExpected, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            API_NAME, "weight storageShape", op::ToString(weightTensor->GetStorageShape()).GetString(),
            (std::string("the storage shape of weight must be ") + op::ToString(weightStorageExpected).GetString())
                .c_str()));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        weightK == k, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            API_NAME, "weight", std::to_string(weightK),
            (std::string("the logical K dimension of weight must be ") + std::to_string(k)).c_str()));
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
    const bool isFp4Weight = weightTensor->GetDataType() == DataType::DT_FLOAT4_E2M1;
    CHECK_RET(CheckLogicalDimensions(m, k, e, weightE, n, isFp4Weight) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

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
    CHECK_RET(CheckViewShape(x, "x", xExpected) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckViewShape(xScale, "xScale", xScaleExpected) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckViewShape(weightTensor, "weight", weightExpected) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckViewShape(weightScaleTensor, "weightScale", weightScaleExpected) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckViewShape(output, "output", outputExpected) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (outputScale != nullptr) {
        CHECK_RET(CheckViewShape(outputScale, "outputScale", outputScaleExpected) == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }
    return CheckWeightStorageShape(weightTensor, e, n, k, weightK, isFp4Weight, transposeWeight);
}

aclnnStatus CheckA8W4Inputs(const aclTensor* x, const aclTensorList* weight, const aclTensor* smoothScale,
                            const aclTensor* groupList, bool transposeWeight)
{
    const bool isMxA8W4 = IsMxA8W4(x->GetDataType(), (*weight)[0]->GetDataType());
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        !isMxA8W4 || transposeWeight, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "weight", "MX A8W4 only supports transposed WeightNZ"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        isMxA8W4 || smoothScale == nullptr, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "smoothScale", "only MX A8W4 supports smoothScale"));
    // These inputs are dereferenced as GM by the A8W4 kernel. Contiguous is not an H2D guarantee.
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        !isMxA8W4 || groupList->GetPlacement() == op::TensorPlacement::kOnDeviceHbm, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "groupList", "MX A8W4 requires device HBM memory"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        !isMxA8W4 || smoothScale == nullptr || smoothScale->GetPlacement() == op::TensorPlacement::kOnDeviceHbm,
        ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "smoothScale", "MX A8W4 requires device HBM memory"));
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
    CHECK_RET(CheckOptionalInputs(weightAssistMatrix, bias, tuningConfigOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckGmmAttrs(dequantMode, dequantDtype, quantMode, groupListType) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(x, weight, weightScale, xScale, groupList, output, outputScale) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckA8W4Inputs(x, weight, smoothScale, groupList, transposeWeight) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSwigluAttrs(swigluMode, clampLimit, gluAlpha, gluBias, roundMode, scaleAlg, dstTypeMax,
                               IsA4W4Bf16Output(x, weight, output)) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    GMM_SWIGLU_V3_CHECK_WITH_LOG(op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510,
                                 ACLNN_ERR_PARAM_INVALID,
                                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Only Ascend 950 is supported by %s.", API_NAME));
    CHECK_RET(CheckFormat(x, weight, weightScale, xScale, groupList, output, outputScale) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        CheckShape(x, weight, weightScale, xScale, groupList, output, outputScale, transposeWeight) == ACLNN_SUCCESS,
        ACLNN_ERR_PARAM_INVALID);
    if (smoothScale != nullptr) {
        const op::Shape smoothExpected = {groupList->GetViewShape().GetDim(0), output->GetViewShape().GetDim(1)};
        GMM_SWIGLU_V3_CHECK_WITH_LOG(
            smoothScale->GetDataType() == DataType::DT_FLOAT &&
                smoothScale->GetStorageFormat() == op::Format::FORMAT_ND &&
                smoothScale->GetViewShape() == smoothExpected,
            ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "smoothScale", "must be FLOAT ND with shape (E, N/2)"));
    }
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        !IsA4W4Bf16Output(x, weight, output) || !gert::TensorPlacementUtils::IsOnHost(groupList->GetPlacement()),
        ACLNN_ERR_PARAM_INVALID, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "A4W4 groupList must be backed by device memory"));
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

const aclTensor* PrepareContiguousNdTensor(const aclTensor* tensor, aclOpExecutor* executor)
{
    const aclTensor* contiguousTensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(contiguousTensor != nullptr, nullptr);
    // Contiguous may materialize an ND scale as a default-format view.
    // Restore the ND descriptor while retaining the contiguous buffer.
    aclTensor* ndTensor =
        executor->CreateView(contiguousTensor, contiguousTensor->GetViewShape(), contiguousTensor->GetViewOffset());
    CHECK_RET(ndTensor != nullptr, nullptr);
    ndTensor->SetViewFormat(op::Format::FORMAT_ND);
    ndTensor->SetOriginalFormat(op::Format::FORMAT_ND);
    ndTensor->SetStorageFormat(op::Format::FORMAT_ND);
    ndTensor->SetStorageShape(contiguousTensor->GetStorageShape());
    return ndTensor;
}

aclnnStatus PrepareContiguousTensorList(const aclTensorList* tensorList, const aclTensorList*& newTensorList,
                                        aclOpExecutor* executor)
{
    std::vector<const aclTensor*> tensors;
    tensors.reserve(tensorList->Size());
    for (size_t index = 0; index < tensorList->Size(); ++index) {
        const aclTensor* ndTensor = PrepareContiguousNdTensor((*tensorList)[index], executor);
        CHECK_RET(ndTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
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
aclnnStatus CheckWorkspaceStorage(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale,
                                  const aclTensorList* weightAssistMatrix, const aclTensor* bias,
                                  const aclTensor* xScale, const aclTensor* smoothScale, const aclTensor* groupList,
                                  const aclTensor* output, const aclTensor* outputScale)
{
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
    if (outputScale != nullptr) {
        CHECK_RET(gmm::CheckTensorStorageBounds(outputScale, "GroupedMatmulSwigluQuant", "outputScale"),
                  ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus PrepareQuantInputs(const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList,
                               const aclTensor* smoothScale, const aclTensorList* normalizedWeight,
                               const aclTensorList* normalizedWeightScale, aclOpExecutor* executor,
                               const aclTensor*& xForOp, const aclTensor*& xScaleForOp,
                               const aclTensor*& groupListForOp, const aclTensor*& smoothScaleForOp,
                               const aclTensorList*& weightScaleForOp, const aclTensorList*& weightForOp)
{
    const bool isMxA8W4 = IsMxA8W4(x->GetDataType(), (*normalizedWeight)[0]->GetDataType());
    xForOp = isMxA8W4 ? PrepareContiguousNdTensor(x, executor) : l0op::Contiguous(x, executor);
    xScaleForOp = isMxA8W4 ? PrepareContiguousNdTensor(xScale, executor) : l0op::Contiguous(xScale, executor);
    groupListForOp = isMxA8W4 ? PrepareContiguousNdTensor(groupList, executor) : l0op::Contiguous(groupList, executor);
    CHECK_RET(xForOp != nullptr && xScaleForOp != nullptr && groupListForOp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    smoothScaleForOp = nullptr;
    if (smoothScale != nullptr) {
        smoothScaleForOp = PrepareContiguousNdTensor(smoothScale, executor);
        CHECK_RET(smoothScaleForOp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    weightScaleForOp = nullptr;
    CHECK_RET(PrepareContiguousTensorList(normalizedWeightScale, weightScaleForOp, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    weightForOp = nullptr;
    CHECK_RET(WrapWeightNzTensorList(normalizedWeight, weightForOp, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CopyQuantOutputs(const std::tuple<aclTensor*, aclTensor*>& result, aclTensor* output,
                             aclTensor* outputScale, aclOpExecutor* executor)
{
    const aclTensor* outputResult = l0op::ViewCopy(std::get<0>(result), output, executor);
    CHECK_RET(outputResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (outputScale != nullptr) {
        const aclTensor* outputScaleResult =
            l0op::ViewCopy(std::get<OUTPUT_SCALE_RESULT_INDEX>(result), outputScale, executor);
        CHECK_RET(outputScaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus ExecuteQuant(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightAssistMatrix,
                         const aclTensor* bias, int64_t dequantMode, int64_t dequantDtype, int64_t quantMode,
                         int64_t groupListType, int64_t swigluMode, double clampLimit, double gluAlpha, double gluBias,
                         const char* roundMode, int64_t scaleAlg, double dstTypeMax, aclTensor* output,
                         aclTensor* outputScale, bool transposeWeight, aclOpExecutor* executor, const aclTensor* xForOp,
                         const aclTensor* xScaleForOp, const aclTensor* groupListForOp,
                         const aclTensor* smoothScaleForOp, const aclTensorList* weightScaleForOp,
                         const aclTensorList* weightForOp)
{
    std::tuple<aclTensor*, aclTensor*> result;
    if (IsA4W4Bf16Output(x, weight, output)) {
        const aclTensor* activationOut = l0op::GroupedMatmulActivation(
            xForOp, xScaleForOp, groupListForOp, weightForOp, weightScaleForOp, nullptr, dequantMode, dequantDtype,
            "swiglu", groupListType, nullptr, swigluMode, static_cast<float>(clampLimit), static_cast<float>(gluAlpha),
            static_cast<float>(gluBias), transposeWeight, executor);
        result = std::make_tuple(const_cast<aclTensor*>(activationOut), nullptr);
    } else {
        const aclTensorList* weightAssistForOp = weightAssistMatrix;
        CHECK_RET(CreateEmptyWeightAssistTensorList(weightAssistForOp, executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        const char* effectiveRoundMode =
            (roundMode == nullptr || roundMode[0] == '\0') ? gmm_swiglu_quant_v3::DEFAULT_ROUND_MODE : roundMode;
        result = l0op::GroupedMatmulSwigluQuantV2(
            xForOp, weightForOp, weightScaleForOp, xScaleForOp, weightAssistForOp, bias, smoothScaleForOp,
            groupListForOp, dequantMode, dequantDtype, quantMode, static_cast<int64_t>(output->GetDataType()),
            transposeWeight, groupListType, nullptr, swigluMode, static_cast<float>(clampLimit),
            static_cast<float>(gluAlpha), static_cast<float>(gluBias), effectiveRoundMode, scaleAlg,
            static_cast<float>(dstTypeMax), executor);
    }
    CHECK_RET(result != std::tuple(nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);

    return CopyQuantOutputs(result, output, outputScale, executor);
}

aclnnStatus RunQuant(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightAssistMatrix,
                     const aclTensor* bias, const aclTensor* xScale, const aclTensor* smoothScale,
                     const aclTensor* groupList, int64_t dequantMode, int64_t dequantDtype, int64_t quantMode,
                     int64_t groupListType, int64_t swigluMode, double clampLimit, double gluAlpha, double gluBias,
                     const char* roundMode, int64_t scaleAlg, double dstTypeMax, aclTensor* output,
                     aclTensor* outputScale, bool transposeWeight, const aclTensorList* normalizedWeight,
                     const aclTensorList* normalizedWeightScale, aclOpExecutor* executor)
{
    const aclTensor *xForOp = nullptr, *xScaleForOp = nullptr, *groupListForOp = nullptr, *smoothScaleForOp = nullptr;
    const aclTensorList *weightScaleForOp = nullptr, *weightForOp = nullptr;
    CHECK_RET(PrepareQuantInputs(x, xScale, groupList, smoothScale, normalizedWeight, normalizedWeightScale, executor,
                                 xForOp, xScaleForOp, groupListForOp, smoothScaleForOp, weightScaleForOp,
                                 weightForOp) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    return ExecuteQuant(x, weight, weightAssistMatrix, bias, dequantMode, dequantDtype, quantMode, groupListType,
                        swigluMode, clampLimit, gluAlpha, gluBias, roundMode, scaleAlg, dstTypeMax, output, outputScale,
                        transposeWeight, executor, xForOp, xScaleForOp, groupListForOp, smoothScaleForOp,
                        weightScaleForOp, weightForOp);
}

aclnnStatus ValidateQuantInputs(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale,
                                const aclTensorList* weightAssistMatrix, const aclTensor* bias, const aclTensor* xScale,
                                const aclTensor* smoothScale, const aclTensor* groupList, int64_t dequantMode,
                                int64_t dequantDtype, int64_t quantMode, int64_t groupListType,
                                const aclIntArray* tuningConfigOptional, int64_t swigluMode, double clampLimit,
                                double gluAlpha, double gluBias, const char* roundMode, int64_t scaleAlg,
                                double dstTypeMax, aclTensor* output, aclTensor* outputScale, aclOpExecutor* executor,
                                bool& transposeWeight, const aclTensorList*& normalizedWeight,
                                const aclTensorList*& normalizedWeightScale)
{
    // Reject invalid ranks before transpose normalization or any internal operator.
    CHECK_RET(CheckDimensionNumbers(x, weight, weightScale, xScale, groupList, output, outputScale) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTransposeConsistency(weight, weightScale, transposeWeight) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(NormalizeMXTranspose(weight, weightScale, transposeWeight, normalizedWeight, normalizedWeightScale,
                                   executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckParams(x, normalizedWeight, normalizedWeightScale, weightAssistMatrix, bias, xScale, smoothScale,
                          groupList, dequantMode, dequantDtype, quantMode, groupListType, tuningConfigOptional,
                          swigluMode, clampLimit, gluAlpha, gluBias, roundMode, scaleAlg, dstTypeMax, output,
                          outputScale, transposeWeight) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

} // namespace

namespace gmm_activation {
aclnnStatus PrepareInputs(const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList,
                          const aclTensorList* weight, const aclTensorList* weightScale, int64_t groupListType,
                          const aclIntArray* tuningConfig, int64_t swigluMode, double clampLimit, double gluAlpha,
                          double gluBias, aclTensor* y, aclOpExecutor* executor, PreparedInputs& inputs)
{
    // Use the same storage, shape, scale and layout validation as WeightNzV3.
    CHECK_RET(gmm::CheckTensorStorageBounds(x, "GroupedMatmulActivation", "x"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(xScale, "GroupedMatmulActivation", "xScale"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(groupList, "GroupedMatmulActivation", "groupList"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorListStorageBounds(weight, "GroupedMatmulActivation", "weight"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorListStorageBounds(weightScale, "GroupedMatmulActivation", "weightScale"),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(gmm::CheckTensorStorageBounds(y, "GroupedMatmulActivation", "y"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTransposeConsistency(weight, weightScale, inputs.transposeWeight) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    const aclTensorList* normalizedWeight = nullptr;
    const aclTensorList* normalizedScale = nullptr;
    CHECK_RET(NormalizeMXTranspose(weight, weightScale, inputs.transposeWeight, normalizedWeight, normalizedScale,
                                   executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckParams(x, normalizedWeight, normalizedScale, nullptr, nullptr, xScale, nullptr, groupList,
                          gmm_swiglu_quant_v3::QUANT_MODE_MX, gmm_swiglu_quant_v3::DEQUANT_DTYPE_FLOAT,
                          gmm_swiglu_quant_v3::QUANT_MODE_MX, groupListType, tuningConfig, swigluMode, clampLimit,
                          gluAlpha, gluBias, "rint", 0, 0.0, y, nullptr, inputs.transposeWeight) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    inputs.x = l0op::Contiguous(x, executor);
    inputs.xScale = l0op::Contiguous(xScale, executor);
    inputs.groupList = l0op::Contiguous(groupList, executor);
    CHECK_RET(inputs.x != nullptr && inputs.xScale != nullptr && inputs.groupList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(PrepareContiguousTensorList(normalizedScale, inputs.weightScale, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(WrapWeightNzTensorList(normalizedWeight, inputs.weight, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
} // namespace gmm_activation

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
    aclnnStatus status = gmm_swiglu_quant_api_check::CheckRequiredInputs(
        API_NAME, x, weight, weightScale, xScale, groupList, output, outputScale,
        gmm_swiglu_quant_v3::SINGLE_TENSOR_SIZE, !IsA4W4Bf16Output(x, weight, output));
    CHECK_RET(status == ACLNN_SUCCESS, status);
    CHECK_RET(CheckWorkspaceStorage(x, weight, weightScale, weightAssistMatrix, bias, xScale, smoothScale, groupList,
                                    output, outputScale) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    // Initialize DFX before the executor captures the API name and profiling context.
    L2_DFX_PHASE_1(aclnnGroupedMatmulSwigluQuantWeightNzV3,
                   DFX_IN(x, weight, weightScale, weightAssistMatrix, bias, xScale, smoothScale, groupList, dequantMode,
                          dequantDtype, quantMode, groupListType, tuningConfigOptional, swigluMode, clampLimit,
                          gluAlpha, gluBias, roundMode, scaleAlg, dstTypeMax),
                   DFX_OUT(output, outputScale));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    bool transposeWeight = false;
    const aclTensorList *normalizedWeight = nullptr, *normalizedWeightScale = nullptr;
    status = ValidateQuantInputs(x, weight, weightScale, weightAssistMatrix, bias, xScale, smoothScale, groupList,
                                 dequantMode, dequantDtype, quantMode, groupListType, tuningConfigOptional, swigluMode,
                                 clampLimit, gluAlpha, gluBias, roundMode, scaleAlg, dstTypeMax, output, outputScale,
                                 uniqueExecutor.get(), transposeWeight, normalizedWeight, normalizedWeightScale);
    CHECK_RET(status == ACLNN_SUCCESS, status);

    // Empty outputs still require complete pointer, attribute and metadata validation.
    if (output->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    CHECK_RET(RunQuant(x, weight, weightAssistMatrix, bias, xScale, smoothScale, groupList, dequantMode, dequantDtype,
                       quantMode, groupListType, swigluMode, clampLimit, gluAlpha, gluBias, roundMode, scaleAlg,
                       dstTypeMax, output, outputScale, transposeWeight, normalizedWeight, normalizedWeightScale,
                       uniqueExecutor.get()) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                    aclrtStream stream)
{
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(EXECUTE_API_NAME, "executor", "executor must not be nullptr"));
    GMM_SWIGLU_V3_CHECK_WITH_LOG(
        workspaceSize == 0 || workspace != nullptr, ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(EXECUTE_API_NAME, "workspace",
                                                 "workspace must not be nullptr when workspaceSize is greater than 0"));
    L2_DFX_PHASE_2(aclnnGroupedMatmulSwigluQuantWeightNzV3);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in GroupedMatmulSwigluQuantWeightNzV3 launch aicore");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
