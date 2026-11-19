/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <initializer_list>
#include <limits>
#include <string>

#include "aclnn_kernels/contiguous.h"
#include "block_attn_res_update.h"
#include "block_attn_res_update_common.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "log/log.h"

using namespace op;

namespace BlockAttnResUpdateAclnn {
namespace {
constexpr size_t MATRIX_DIM_NUM = 2UL;
constexpr size_t VECTOR_DIM_NUM = 1UL;
constexpr size_t TOKEN_DIM_INDEX = 0UL;
constexpr size_t HIDDEN_DIM_INDEX = 1UL;
constexpr int64_t MAX_D_SIZE = 8192L;
constexpr const char* PARTIAL_BLOCK_REF_NAME = "partialBlockRef";
constexpr const char* DELTA_NAME = "delta";
constexpr const char* PSEUDO_QUERY_NAME = "pseudoQuery";
constexpr const char* NUMERATOR_NAME = "numerator";
constexpr const char* LOGIT_MAX_NAME = "logitMax";
constexpr const char* EXP_SUM_NAME = "expSum";
constexpr const char* OUTPUT_NORM_WEIGHT_NAME = "outputNormWeightOptional";
constexpr const char* H_NAME = "h";

static const std::initializer_list<op::DataType> FLOAT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> BF16_DTYPE_SUPPORT_LIST = {DataType::DT_BF16};

static bool CheckNotNull(const char* apiName, const BlockAttnResUpdateParams& params)
{
    OP_CHECK(params.partialBlockRef != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(apiName, PARTIAL_BLOCK_REF_NAME, "The tensor must not be null"),
             return false);
    OP_CHECK(params.delta != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(apiName, DELTA_NAME, "The tensor must not be null"),
             return false);
    OP_CHECK(params.pseudoQuery != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(apiName, PSEUDO_QUERY_NAME, "The tensor must not be null"),
             return false);
    OP_CHECK(params.numerator != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(apiName, NUMERATOR_NAME, "The tensor must not be null"),
             return false);
    OP_CHECK(params.logitMax != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(apiName, LOGIT_MAX_NAME, "The tensor must not be null"),
             return false);
    OP_CHECK(params.expSum != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(apiName, EXP_SUM_NAME, "The tensor must not be null"),
             return false);
    OP_CHECK(params.h != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(apiName, H_NAME, "The tensor must not be null"), return false);
    return true;
}

static aclnnStatus CheckAiCoreSupport(const char* apiName)
{
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch != NpuArch::DAV_3510 && npuArch != NpuArch::DAV_9201) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                "%s only supports DAV_3510 (Ascend 950/350) and DAV_9201 (Ascend 960DT), but got NPU architecture %d.",
                apiName, static_cast<int32_t>(npuArch));
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
}

static bool CheckDtypeValid(const char* apiName, const BlockAttnResUpdateParams& params)
{
    OP_CHECK(CheckType(params.partialBlockRef->GetDataType(), FLOAT_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(apiName, PARTIAL_BLOCK_REF_NAME,
                                       op::ToString(params.partialBlockRef->GetDataType()).GetString(),
                                       op::ToString(FLOAT_DTYPE_SUPPORT_LIST).GetString()),
             return false);
    OP_CHECK(CheckType(params.delta->GetDataType(), BF16_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(apiName, DELTA_NAME, op::ToString(params.delta->GetDataType()).GetString(),
                                       op::ToString(BF16_DTYPE_SUPPORT_LIST).GetString()),
             return false);
    OP_CHECK(CheckType(params.pseudoQuery->GetDataType(), FLOAT_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(apiName, PSEUDO_QUERY_NAME,
                                       op::ToString(params.pseudoQuery->GetDataType()).GetString(),
                                       op::ToString(FLOAT_DTYPE_SUPPORT_LIST).GetString()),
             return false);
    OP_CHECK(
        CheckType(params.numerator->GetDataType(), FLOAT_DTYPE_SUPPORT_LIST),
        OP_LOGE_FOR_INVALID_DTYPE(apiName, NUMERATOR_NAME, op::ToString(params.numerator->GetDataType()).GetString(),
                                  op::ToString(FLOAT_DTYPE_SUPPORT_LIST).GetString()),
        return false);
    OP_CHECK(
        CheckType(params.logitMax->GetDataType(), FLOAT_DTYPE_SUPPORT_LIST),
        OP_LOGE_FOR_INVALID_DTYPE(apiName, LOGIT_MAX_NAME, op::ToString(params.logitMax->GetDataType()).GetString(),
                                  op::ToString(FLOAT_DTYPE_SUPPORT_LIST).GetString()),
        return false);
    OP_CHECK(CheckType(params.expSum->GetDataType(), FLOAT_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(apiName, EXP_SUM_NAME, op::ToString(params.expSum->GetDataType()).GetString(),
                                       op::ToString(FLOAT_DTYPE_SUPPORT_LIST).GetString()),
             return false);
    OP_CHECK(CheckType(params.h->GetDataType(), BF16_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(apiName, H_NAME, op::ToString(params.h->GetDataType()).GetString(),
                                       op::ToString(BF16_DTYPE_SUPPORT_LIST).GetString()),
             return false);
    return true;
}

static bool CheckTensorFormat(const char* apiName, const aclTensor* tensor, const char* tensorName)
{
    const auto originalFormat = tensor->GetOriginalFormat();
    const auto storageFormat = tensor->GetStorageFormat();
    const std::string actualFormats = std::string("original=") + op::ToString(originalFormat).GetString() +
                                      ", storage=" + op::ToString(storageFormat).GetString();
    OP_CHECK(originalFormat == Format::FORMAT_ND && storageFormat == Format::FORMAT_ND,
             OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(apiName, tensorName, actualFormats.c_str(),
                                                     "original and storage formats must both be ND"),
             return false);
    return true;
}

static bool CheckFormat(const char* apiName, const BlockAttnResUpdateParams& params)
{
    if (!CheckTensorFormat(apiName, params.partialBlockRef, PARTIAL_BLOCK_REF_NAME)) {
        return false;
    }
    if (!CheckTensorFormat(apiName, params.delta, DELTA_NAME)) {
        return false;
    }
    if (!CheckTensorFormat(apiName, params.pseudoQuery, PSEUDO_QUERY_NAME)) {
        return false;
    }
    if (!CheckTensorFormat(apiName, params.numerator, NUMERATOR_NAME)) {
        return false;
    }
    if (!CheckTensorFormat(apiName, params.logitMax, LOGIT_MAX_NAME)) {
        return false;
    }
    if (!CheckTensorFormat(apiName, params.expSum, EXP_SUM_NAME)) {
        return false;
    }
    return CheckTensorFormat(apiName, params.h, H_NAME);
}

static bool CheckTensorDimNum(const char* apiName, const aclTensor* tensor, const char* tensorName,
                              size_t expectedDimNum)
{
    const size_t actualDimNum = tensor->GetViewShape().GetDimNum();
    const std::string actualDimNumString = std::to_string(actualDimNum) + "D";
    const std::string reason = "tensor must be " + std::to_string(expectedDimNum) + "D";
    OP_CHECK(actualDimNum == expectedDimNum,
             OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(apiName, tensorName, actualDimNumString.c_str(), reason.c_str()),
             return false);
    return true;
}

static bool CheckTensorShape(const char* apiName, const aclTensor* tensor, const char* tensorName,
                             const aclTensor* reference, const char* referenceName)
{
    const auto& shape = tensor->GetViewShape();
    const auto& referenceShape = reference->GetViewShape();
    const std::string actualShape = op::ToString(shape).GetString();
    const std::string reason =
        std::string("shape must match ") + referenceName + " " + op::ToString(referenceShape).GetString();
    OP_CHECK(shape == referenceShape,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(apiName, tensorName, actualShape.c_str(), reason.c_str()),
             return false);
    return true;
}

static bool CheckTensorDimNums(const char* apiName, const BlockAttnResUpdateParams& params)
{
    if (!CheckTensorDimNum(apiName, params.partialBlockRef, PARTIAL_BLOCK_REF_NAME, MATRIX_DIM_NUM)) {
        return false;
    }
    if (!CheckTensorDimNum(apiName, params.delta, DELTA_NAME, MATRIX_DIM_NUM)) {
        return false;
    }
    if (!CheckTensorDimNum(apiName, params.pseudoQuery, PSEUDO_QUERY_NAME, VECTOR_DIM_NUM)) {
        return false;
    }
    if (!CheckTensorDimNum(apiName, params.numerator, NUMERATOR_NAME, MATRIX_DIM_NUM)) {
        return false;
    }
    if (!CheckTensorDimNum(apiName, params.logitMax, LOGIT_MAX_NAME, VECTOR_DIM_NUM)) {
        return false;
    }
    if (!CheckTensorDimNum(apiName, params.expSum, EXP_SUM_NAME, VECTOR_DIM_NUM)) {
        return false;
    }
    if (!CheckTensorDimNum(apiName, params.h, H_NAME, MATRIX_DIM_NUM)) {
        return false;
    }
    return true;
}

static bool CheckMatrixShapes(const char* apiName, const BlockAttnResUpdateParams& params)
{
    if (!CheckTensorShape(apiName, params.delta, DELTA_NAME, params.partialBlockRef, PARTIAL_BLOCK_REF_NAME)) {
        return false;
    }
    if (!CheckTensorShape(apiName, params.numerator, NUMERATOR_NAME, params.partialBlockRef, PARTIAL_BLOCK_REF_NAME)) {
        return false;
    }
    if (!CheckTensorShape(apiName, params.h, H_NAME, params.partialBlockRef, PARTIAL_BLOCK_REF_NAME)) {
        return false;
    }
    return true;
}

static bool CheckVectorShapes(const char* apiName, const BlockAttnResUpdateParams& params, int64_t tSize, int64_t dSize)
{
    const auto& logitMaxShape = params.logitMax->GetViewShape();
    const std::string logitMaxShapeString = op::ToString(logitMaxShape).GetString();
    OP_CHECK(
        logitMaxShape.GetDim(TOKEN_DIM_INDEX) == tSize,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(apiName, LOGIT_MAX_NAME, logitMaxShapeString.c_str(),
                                               "shape must be [T], where T is the first dimension of partialBlockRef"),
        return false);
    const auto& expSumShape = params.expSum->GetViewShape();
    const std::string expSumShapeString = op::ToString(expSumShape).GetString();
    OP_CHECK(
        expSumShape.GetDim(TOKEN_DIM_INDEX) == tSize,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(apiName, EXP_SUM_NAME, expSumShapeString.c_str(),
                                               "shape must be [T], where T is the first dimension of partialBlockRef"),
        return false);
    const auto& pseudoQueryShape = params.pseudoQuery->GetViewShape();
    const std::string pseudoQueryShapeString = op::ToString(pseudoQueryShape).GetString();
    OP_CHECK(
        pseudoQueryShape.GetDim(TOKEN_DIM_INDEX) == dSize,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(apiName, PSEUDO_QUERY_NAME, pseudoQueryShapeString.c_str(),
                                               "shape must be [D], where D is the second dimension of partialBlockRef"),
        return false);
    return true;
}

static bool CheckDimensionRangesAndSize(const char* apiName, int64_t tSize, int64_t dSize)
{
    OP_CHECK(tSize >= 0,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, PARTIAL_BLOCK_REF_NAME, std::to_string(tSize),
                                                   "the T dimension must be greater than or equal to 0"),
             return false);
    OP_CHECK(dSize >= 0,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, PARTIAL_BLOCK_REF_NAME, std::to_string(dSize),
                                                   "the D dimension must be greater than or equal to 0"),
             return false);
    const std::string maxDReason = "the D dimension must be less than or equal to " + std::to_string(MAX_D_SIZE);
    OP_CHECK(dSize <= MAX_D_SIZE,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, PARTIAL_BLOCK_REF_NAME, std::to_string(dSize),
                                                   maxDReason.c_str()),
             return false);

    // An empty [T, D] workload has no elements to update. All dimension and shape relations above still apply;
    // only InferShape, tiling and kernel launch are skipped by the L0 operator.
    if (tSize == 0 || dSize == 0) {
        return true;
    }

    const std::string tdValues = "T=" + std::to_string(tSize) + ", D=" + std::to_string(dSize);
    OP_CHECK(tSize <= std::numeric_limits<int64_t>::max() / dSize,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(apiName, PARTIAL_BLOCK_REF_NAME, tdValues.c_str(),
                                                    "the product T * D must be less than or equal to INT64_MAX"),
             return false);
    return true;
}

static bool CheckShape(const char* apiName, const BlockAttnResUpdateParams& params)
{
    if (!CheckTensorDimNums(apiName, params) || !CheckMatrixShapes(apiName, params)) {
        return false;
    }

    const auto& partialBlockRefShape = params.partialBlockRef->GetViewShape();
    const int64_t tSize = partialBlockRefShape.GetDim(TOKEN_DIM_INDEX);
    const int64_t dSize = partialBlockRefShape.GetDim(HIDDEN_DIM_INDEX);
    if (!CheckVectorShapes(apiName, params, tSize, dSize)) {
        return false;
    }
    return CheckDimensionRangesAndSize(apiName, tSize, dSize);
}

static bool CheckTensorContiguous(const char* apiName, const aclTensor* tensor, const char* tensorName)
{
    const std::string reason = std::string("tensor must be contiguous, but got shape ") +
                               op::ToString(tensor->GetViewShape()).GetString() + " and strides " +
                               op::ToString(tensor->GetViewStrides()).GetString();
    OP_CHECK(IsContiguous(tensor), OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(apiName, tensorName, reason.c_str()),
             return false);
    return true;
}

static bool CheckContiguous(const char* apiName, const BlockAttnResUpdateParams& params)
{
    if (!CheckTensorContiguous(apiName, params.partialBlockRef, PARTIAL_BLOCK_REF_NAME)) {
        return false;
    }
    if (!CheckTensorContiguous(apiName, params.delta, DELTA_NAME)) {
        return false;
    }
    if (!CheckTensorContiguous(apiName, params.pseudoQuery, PSEUDO_QUERY_NAME)) {
        return false;
    }
    if (!CheckTensorContiguous(apiName, params.numerator, NUMERATOR_NAME)) {
        return false;
    }
    if (!CheckTensorContiguous(apiName, params.logitMax, LOGIT_MAX_NAME)) {
        return false;
    }
    if (!CheckTensorContiguous(apiName, params.expSum, EXP_SUM_NAME)) {
        return false;
    }
    return CheckTensorContiguous(apiName, params.h, H_NAME);
}

static bool CheckEps(const char* apiName, double eps)
{
    OP_CHECK(std::isfinite(eps) && eps > 0.0,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "eps", std::to_string(eps),
                                                   "The value of eps must be finite and greater than 0"),
             return false);
    return true;
}

static bool CheckOutputNorm(const char* apiName, const BlockAttnResUpdateParams& params)
{
    if (params.outputNormWeightOptional == nullptr) {
        return true;
    }
    const auto* weight = params.outputNormWeightOptional;
    static const std::initializer_list<op::DataType> WEIGHT_DTYPES = {DataType::DT_BF16, DataType::DT_FLOAT};
    OP_CHECK(
        CheckType(weight->GetDataType(), WEIGHT_DTYPES),
        OP_LOGE_FOR_INVALID_DTYPE(apiName, OUTPUT_NORM_WEIGHT_NAME, op::ToString(weight->GetDataType()).GetString(),
                                  op::ToString(WEIGHT_DTYPES).GetString()),
        return false);
    if (!CheckTensorFormat(apiName, weight, OUTPUT_NORM_WEIGHT_NAME) ||
        !CheckTensorDimNum(apiName, weight, OUTPUT_NORM_WEIGHT_NAME, VECTOR_DIM_NUM) ||
        !CheckTensorShape(apiName, weight, OUTPUT_NORM_WEIGHT_NAME, params.pseudoQuery, PSEUDO_QUERY_NAME) ||
        !CheckTensorContiguous(apiName, weight, OUTPUT_NORM_WEIGHT_NAME)) {
        return false;
    }
    OP_CHECK(params.outputNormEps >= std::numeric_limits<float>::min() &&
                 params.outputNormEps <= std::numeric_limits<float>::max(),
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                 apiName, "outputNormEps", std::to_string(params.outputNormEps),
                 "The value of outputNormEps must be in [FLT_MIN, FLT_MAX] before conversion to FLOAT"),
             return false);
    return true;
}

static aclnnStatus CheckParams(const char* apiName, const BlockAttnResUpdateParams& params)
{
    if (!CheckNotNull(apiName, params)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    const aclnnStatus aiCoreSupportStatus = CheckAiCoreSupport(apiName);
    if (aiCoreSupportStatus != ACLNN_SUCCESS) {
        return aiCoreSupportStatus;
    }
    if (!CheckDtypeValid(apiName, params)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckFormat(apiName, params)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(apiName, params)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckContiguous(apiName, params)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckEps(apiName, params.eps)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckOutputNorm(apiName, params)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static bool PreProcess(BlockAttnResUpdateParams& params, aclOpExecutor* executor)
{
    // CreateView normalizes storage/original shapes to the contiguous logical view and preserves its offset.
    // Keep these descriptors in the executor so caller metadata is unchanged, including when inputs alias.
    const auto createView = [executor](auto& tensor) {
        tensor = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetViewOffset());
        return tensor != nullptr;
    };
    return createView(params.partialBlockRef) && createView(params.delta) && createView(params.pseudoQuery) &&
           createView(params.numerator) && createView(params.logitMax) && createView(params.expSum) &&
           createView(params.h) &&
           (params.outputNormWeightOptional == nullptr || createView(params.outputNormWeightOptional));
}
} // namespace

aclnnStatus GetWorkspaceSize(const char* apiName, BlockAttnResUpdateParams params, uint64_t* workspaceSize,
                             aclOpExecutor** executor)
{
    OP_CHECK(workspaceSize != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(apiName, "workspaceSize", "The pointer must not be null"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(executor != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(apiName, "executor", "The pointer must not be null"),
             return ACLNN_ERR_PARAM_NULLPTR);
    const auto ret = CheckParams(apiName, params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(PreProcess(params, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);

    const double outputNormEps = params.outputNormWeightOptional == nullptr ? 1e-6 : params.outputNormEps;
    auto opResult = l0op::BlockAttnResUpdate(params.partialBlockRef, params.delta, params.pseudoQuery, params.numerator,
                                             params.logitMax, params.expSum, params.outputNormWeightOptional,
                                             params.eps, outputNormEps, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyH = l0op::ViewCopy(opResult, params.h, uniqueExecutor.get());
    CHECK_RET(viewCopyH != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
} // namespace BlockAttnResUpdateAclnn
