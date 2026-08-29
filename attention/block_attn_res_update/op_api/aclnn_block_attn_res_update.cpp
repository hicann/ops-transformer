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
#include <limits>
#include <string>

#include "aclnn_kernels/contiguous.h"
#include "aclnn_block_attn_res_update.h"
#include "block_attn_res_update.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "log/log.h"

using namespace op;

namespace {
constexpr size_t MATRIX_DIM_NUM = 2UL;
constexpr size_t VECTOR_DIM_NUM = 1UL;
constexpr size_t TOKEN_DIM_INDEX = 0UL;
constexpr size_t HIDDEN_DIM_INDEX = 1UL;
constexpr int64_t MAX_D_SIZE = 8192L;
constexpr const char *ACLNN_API_NAME = "aclnnBlockAttnResUpdateGetWorkspaceSize";
constexpr const char *PARTIAL_BLOCK_REF_NAME = "partialBlockRef";
constexpr const char *DELTA_NAME = "delta";
constexpr const char *PSEUDO_QUERY_NAME = "pseudoQuery";
constexpr const char *NUMERATOR_NAME = "numerator";
constexpr const char *LOGIT_MAX_NAME = "logitMax";
constexpr const char *EXP_SUM_NAME = "expSum";
constexpr const char *H_NAME = "h";

struct BlockAttnResUpdateParams {
    aclTensor *partialBlockRef{nullptr};
    const aclTensor *delta{nullptr};
    const aclTensor *pseudoQuery{nullptr};
    const aclTensor *numerator{nullptr};
    const aclTensor *logitMax{nullptr};
    const aclTensor *expSum{nullptr};
    float eps{0.0F};
    aclTensor *h{nullptr};
};

static const std::initializer_list<op::DataType> FLOAT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> BF16_DTYPE_SUPPORT_LIST = {DataType::DT_BF16};

static bool CheckNotNull(const BlockAttnResUpdateParams &params)
{
    OP_CHECK(params.partialBlockRef != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, PARTIAL_BLOCK_REF_NAME, "tensor can not be null"),
             return false);
    OP_CHECK(params.delta != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, DELTA_NAME, "tensor can not be null"),
             return false);
    OP_CHECK(params.pseudoQuery != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, PSEUDO_QUERY_NAME, "tensor can not be null"),
             return false);
    OP_CHECK(params.numerator != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, NUMERATOR_NAME, "tensor can not be null"),
             return false);
    OP_CHECK(params.logitMax != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, LOGIT_MAX_NAME, "tensor can not be null"),
             return false);
    OP_CHECK(params.expSum != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, EXP_SUM_NAME, "tensor can not be null"),
             return false);
    OP_CHECK(params.h != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, H_NAME, "tensor can not be null"), return false);
    return true;
}

static aclnnStatus CheckAiCoreSupport()
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "BlockAttnResUpdate only supports Ascend 950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
}

static bool CheckDtypeValid(const BlockAttnResUpdateParams &params)
{
    OP_CHECK(CheckType(params.partialBlockRef->GetDataType(), FLOAT_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(ACLNN_API_NAME, PARTIAL_BLOCK_REF_NAME,
                                       op::ToString(params.partialBlockRef->GetDataType()).GetString(),
                                       op::ToString(FLOAT_DTYPE_SUPPORT_LIST).GetString()),
             return false);
    OP_CHECK(
        CheckType(params.delta->GetDataType(), BF16_DTYPE_SUPPORT_LIST),
        OP_LOGE_FOR_INVALID_DTYPE(ACLNN_API_NAME, DELTA_NAME, op::ToString(params.delta->GetDataType()).GetString(),
                                  op::ToString(BF16_DTYPE_SUPPORT_LIST).GetString()),
        return false);
    OP_CHECK(CheckType(params.pseudoQuery->GetDataType(), FLOAT_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(ACLNN_API_NAME, PSEUDO_QUERY_NAME,
                                       op::ToString(params.pseudoQuery->GetDataType()).GetString(),
                                       op::ToString(FLOAT_DTYPE_SUPPORT_LIST).GetString()),
             return false);
    OP_CHECK(CheckType(params.numerator->GetDataType(), FLOAT_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(ACLNN_API_NAME, NUMERATOR_NAME,
                                       op::ToString(params.numerator->GetDataType()).GetString(),
                                       op::ToString(FLOAT_DTYPE_SUPPORT_LIST).GetString()),
             return false);
    OP_CHECK(CheckType(params.logitMax->GetDataType(), FLOAT_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(ACLNN_API_NAME, LOGIT_MAX_NAME,
                                       op::ToString(params.logitMax->GetDataType()).GetString(),
                                       op::ToString(FLOAT_DTYPE_SUPPORT_LIST).GetString()),
             return false);
    OP_CHECK(
        CheckType(params.expSum->GetDataType(), FLOAT_DTYPE_SUPPORT_LIST),
        OP_LOGE_FOR_INVALID_DTYPE(ACLNN_API_NAME, EXP_SUM_NAME, op::ToString(params.expSum->GetDataType()).GetString(),
                                  op::ToString(FLOAT_DTYPE_SUPPORT_LIST).GetString()),
        return false);
    OP_CHECK(CheckType(params.h->GetDataType(), BF16_DTYPE_SUPPORT_LIST),
             OP_LOGE_FOR_INVALID_DTYPE(ACLNN_API_NAME, H_NAME, op::ToString(params.h->GetDataType()).GetString(),
                                       op::ToString(BF16_DTYPE_SUPPORT_LIST).GetString()),
             return false);
    return true;
}

static bool CheckTensorFormat(const aclTensor *tensor, const char *tensorName)
{
    const auto originalFormat = tensor->GetOriginalFormat();
    const auto storageFormat = tensor->GetStorageFormat();
    const std::string actualFormats = std::string("original=") + op::ToString(originalFormat).GetString() +
                                      ", storage=" + op::ToString(storageFormat).GetString();
    OP_CHECK(originalFormat == Format::FORMAT_ND && storageFormat == Format::FORMAT_ND,
             OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(ACLNN_API_NAME, tensorName, actualFormats.c_str(),
                                                     "original and storage formats must both be ND"),
             return false);
    return true;
}

static bool CheckFormat(const BlockAttnResUpdateParams &params)
{
    if (!CheckTensorFormat(params.partialBlockRef, PARTIAL_BLOCK_REF_NAME)) {
        return false;
    }
    if (!CheckTensorFormat(params.delta, DELTA_NAME)) {
        return false;
    }
    if (!CheckTensorFormat(params.pseudoQuery, PSEUDO_QUERY_NAME)) {
        return false;
    }
    if (!CheckTensorFormat(params.numerator, NUMERATOR_NAME)) {
        return false;
    }
    if (!CheckTensorFormat(params.logitMax, LOGIT_MAX_NAME)) {
        return false;
    }
    if (!CheckTensorFormat(params.expSum, EXP_SUM_NAME)) {
        return false;
    }
    return CheckTensorFormat(params.h, H_NAME);
}

static bool CheckTensorDimNum(const aclTensor *tensor, const char *tensorName, size_t expectedDimNum)
{
    const size_t actualDimNum = tensor->GetViewShape().GetDimNum();
    const std::string actualDimNumString = std::to_string(actualDimNum) + "D";
    const std::string reason = "tensor must be " + std::to_string(expectedDimNum) + "D";
    OP_CHECK(actualDimNum == expectedDimNum,
             OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(ACLNN_API_NAME, tensorName, actualDimNumString.c_str(),
                                                       reason.c_str()),
             return false);
    return true;
}

static bool CheckTensorShape(const aclTensor *tensor, const char *tensorName, const aclTensor *reference,
                             const char *referenceName)
{
    const auto &shape = tensor->GetViewShape();
    const auto &referenceShape = reference->GetViewShape();
    const std::string actualShape = op::ToString(shape).GetString();
    const std::string reason =
        std::string("shape must match ") + referenceName + " " + op::ToString(referenceShape).GetString();
    OP_CHECK(shape == referenceShape,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_API_NAME, tensorName, actualShape.c_str(), reason.c_str()),
             return false);
    return true;
}

static bool CheckTensorDimNums(const BlockAttnResUpdateParams &params)
{
    if (!CheckTensorDimNum(params.partialBlockRef, PARTIAL_BLOCK_REF_NAME, MATRIX_DIM_NUM)) {
        return false;
    }
    if (!CheckTensorDimNum(params.delta, DELTA_NAME, MATRIX_DIM_NUM)) {
        return false;
    }
    if (!CheckTensorDimNum(params.pseudoQuery, PSEUDO_QUERY_NAME, VECTOR_DIM_NUM)) {
        return false;
    }
    if (!CheckTensorDimNum(params.numerator, NUMERATOR_NAME, MATRIX_DIM_NUM)) {
        return false;
    }
    if (!CheckTensorDimNum(params.logitMax, LOGIT_MAX_NAME, VECTOR_DIM_NUM)) {
        return false;
    }
    if (!CheckTensorDimNum(params.expSum, EXP_SUM_NAME, VECTOR_DIM_NUM)) {
        return false;
    }
    if (!CheckTensorDimNum(params.h, H_NAME, MATRIX_DIM_NUM)) {
        return false;
    }
    return true;
}

static bool CheckMatrixShapes(const BlockAttnResUpdateParams &params)
{
    if (!CheckTensorShape(params.delta, DELTA_NAME, params.partialBlockRef, PARTIAL_BLOCK_REF_NAME)) {
        return false;
    }
    if (!CheckTensorShape(params.numerator, NUMERATOR_NAME, params.partialBlockRef, PARTIAL_BLOCK_REF_NAME)) {
        return false;
    }
    if (!CheckTensorShape(params.h, H_NAME, params.partialBlockRef, PARTIAL_BLOCK_REF_NAME)) {
        return false;
    }
    return true;
}

static bool CheckVectorShapes(const BlockAttnResUpdateParams &params, int64_t tSize, int64_t dSize)
{
    const auto &logitMaxShape = params.logitMax->GetViewShape();
    const std::string logitMaxShapeString = op::ToString(logitMaxShape).GetString();
    OP_CHECK(
        logitMaxShape.GetDim(TOKEN_DIM_INDEX) == tSize,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_API_NAME, LOGIT_MAX_NAME, logitMaxShapeString.c_str(),
                                               "shape must be [T], where T is the first dimension of partialBlockRef"),
        return false);
    const auto &expSumShape = params.expSum->GetViewShape();
    const std::string expSumShapeString = op::ToString(expSumShape).GetString();
    OP_CHECK(
        expSumShape.GetDim(TOKEN_DIM_INDEX) == tSize,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_API_NAME, EXP_SUM_NAME, expSumShapeString.c_str(),
                                               "shape must be [T], where T is the first dimension of partialBlockRef"),
        return false);
    const auto &pseudoQueryShape = params.pseudoQuery->GetViewShape();
    const std::string pseudoQueryShapeString = op::ToString(pseudoQueryShape).GetString();
    OP_CHECK(
        pseudoQueryShape.GetDim(TOKEN_DIM_INDEX) == dSize,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_API_NAME, PSEUDO_QUERY_NAME, pseudoQueryShapeString.c_str(),
                                               "shape must be [D], where D is the second dimension of partialBlockRef"),
        return false);
    return true;
}

static bool CheckDimensionRangesAndSize(int64_t tSize, int64_t dSize)
{
    OP_CHECK(tSize >= 0,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_API_NAME, PARTIAL_BLOCK_REF_NAME, std::to_string(tSize),
                                                   "the T dimension must be greater than or equal to 0"),
             return false);
    OP_CHECK(dSize >= 0,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_API_NAME, PARTIAL_BLOCK_REF_NAME, std::to_string(dSize),
                                                   "the D dimension must be greater than or equal to 0"),
             return false);
    const std::string maxDReason = "the D dimension must be less than or equal to " + std::to_string(MAX_D_SIZE);
    OP_CHECK(dSize <= MAX_D_SIZE,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_API_NAME, PARTIAL_BLOCK_REF_NAME, std::to_string(dSize),
                                                   maxDReason.c_str()),
             return false);

    // An empty [T, D] workload has no elements to update. All dimension and shape relations above still apply;
    // only InferShape, tiling and kernel launch are skipped by the L0 operator.
    if (tSize == 0 || dSize == 0) {
        return true;
    }

    const std::string tdValues = std::to_string(tSize) + ", " + std::to_string(dSize);
    OP_CHECK(tSize <= std::numeric_limits<int64_t>::max() / dSize,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(ACLNN_API_NAME, PARTIAL_BLOCK_REF_NAME, tdValues.c_str(),
                                                    "the product T * D must be less than or equal to INT64_MAX"),
             return false);
    return true;
}

static bool CheckShape(const BlockAttnResUpdateParams &params)
{
    if (!CheckTensorDimNums(params) || !CheckMatrixShapes(params)) {
        return false;
    }

    const auto &partialBlockRefShape = params.partialBlockRef->GetViewShape();
    const int64_t tSize = partialBlockRefShape.GetDim(TOKEN_DIM_INDEX);
    const int64_t dSize = partialBlockRefShape.GetDim(HIDDEN_DIM_INDEX);
    if (!CheckVectorShapes(params, tSize, dSize)) {
        return false;
    }
    return CheckDimensionRangesAndSize(tSize, dSize);
}

static bool CheckTensorContiguous(const aclTensor *tensor, const char *tensorName)
{
    const std::string reason = std::string("tensor must be contiguous, but got shape ") +
                               op::ToString(tensor->GetViewShape()).GetString() + " and strides " +
                               op::ToString(tensor->GetViewStrides()).GetString();
    OP_CHECK(IsContiguous(tensor), OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, tensorName, reason.c_str()),
             return false);
    return true;
}

static bool CheckContiguous(const BlockAttnResUpdateParams &params)
{
    if (!CheckTensorContiguous(params.partialBlockRef, PARTIAL_BLOCK_REF_NAME)) {
        return false;
    }
    if (!CheckTensorContiguous(params.delta, DELTA_NAME)) {
        return false;
    }
    if (!CheckTensorContiguous(params.pseudoQuery, PSEUDO_QUERY_NAME)) {
        return false;
    }
    if (!CheckTensorContiguous(params.numerator, NUMERATOR_NAME)) {
        return false;
    }
    if (!CheckTensorContiguous(params.logitMax, LOGIT_MAX_NAME)) {
        return false;
    }
    if (!CheckTensorContiguous(params.expSum, EXP_SUM_NAME)) {
        return false;
    }
    return CheckTensorContiguous(params.h, H_NAME);
}

static bool CheckEps(float eps)
{
    OP_CHECK(std::isfinite(eps) && eps > 0.0F,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_API_NAME, "eps", std::to_string(eps),
                                                   "eps must be finite and greater than 0"),
             return false);
    return true;
}

static aclnnStatus CheckParams(const BlockAttnResUpdateParams &params)
{
    if (!CheckNotNull(params)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    const aclnnStatus aiCoreSupportStatus = CheckAiCoreSupport();
    if (aiCoreSupportStatus != ACLNN_SUCCESS) {
        return aiCoreSupportStatus;
    }
    if (!CheckDtypeValid(params)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckFormat(params)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(params)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckContiguous(params)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckEps(params.eps)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(BlockAttnResUpdateParams &params)
{
    // Torch tensors may use a flattened storage shape. The operator requires storage and original shapes to match
    // the logical view shape, so normalize all input and output descriptors before invoking the L0 operator.
    const aclTensor *tensors[] = {
        params.partialBlockRef, params.delta, params.pseudoQuery, params.numerator, params.logitMax,
        params.expSum,          params.h};
    for (const aclTensor *tensor : tensors) {
        tensor->SetStorageShape(tensor->GetViewShape());
        tensor->SetOriginalShape(tensor->GetViewShape());
    }
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" aclnnStatus aclnnBlockAttnResUpdateGetWorkspaceSize(aclTensor *partialBlockRef, const aclTensor *delta,
                                                               const aclTensor *pseudoQuery, const aclTensor *numerator,
                                                               const aclTensor *logitMax, const aclTensor *expSum,
                                                               float eps, aclTensor *h, uint64_t *workspaceSize,
                                                               aclOpExecutor **executor)
{
    OP_CHECK(workspaceSize != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, "workspaceSize", "pointer can not be null"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(executor != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ACLNN_API_NAME, "executor", "pointer can not be null"),
             return ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_1(aclnnBlockAttnResUpdate,
                   DFX_IN(partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum, eps),
                   DFX_OUT(partialBlockRef, h));

    BlockAttnResUpdateParams params{partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum, eps, h};
    auto ret = CheckParams(params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    ret = PreProcess(params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto opResult = l0op::BlockAttnResUpdate(partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum, eps,
                                             uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyH = l0op::ViewCopy(opResult, h, uniqueExecutor.get());
    CHECK_RET(viewCopyH != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnBlockAttnResUpdate(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBlockAttnResUpdate);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
