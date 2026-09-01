/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_block_attn_res_prepare.cpp
 * \brief ACLNN implementation for BlockAttnResPrepare.
 */

#include "aclnn_block_attn_res_prepare.h"
#include "block_attn_res_prepare.h"

#include <cmath>
#include <string>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "op_common/log/log.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

constexpr int64_t BLOCK_RES_DIM_NUM = 3;
constexpr int64_t VALID_BLOCKS_DIM_NUM = 1;
constexpr int64_t PSEUDO_QUERY_DIM_NUM = 2;
constexpr int64_t STATS_DIM_NUM = 2;
constexpr int64_t NUMERATOR_DIM_NUM = 3;
constexpr int64_t MIN_HEAD_DIM = 1;
constexpr int64_t MAX_HEAD_DIM = 8192;
constexpr int64_t MIN_BLOCK_NUM = 1;
constexpr int64_t MAX_BLOCK_NUM = 64;
constexpr size_t T_DIM_INDEX = 0;
constexpr size_t N_DIM_INDEX = 1;
constexpr size_t D_DIM_INDEX = 2;
constexpr size_t S_DIM_INDEX = 0;
constexpr size_t PSEUDO_QUERY_D_DIM_INDEX = 1;
constexpr size_t NUMERATOR_T_DIM_INDEX = 1;
constexpr size_t STATS_T_DIM_INDEX = 1;
constexpr size_t VALID_BLOCKS_VALUE_DIM_INDEX = 0;
constexpr size_t NUMERATOR_OUTPUT_INDEX = 0;
constexpr size_t LOGIT_MAX_OUTPUT_INDEX = 1;
constexpr size_t EXP_SUM_OUTPUT_INDEX = 2;
constexpr const char *API_NAME = "aclnnBlockAttnResPrepareGetWorkspaceSize";

aclnnStatus CheckRequiredParameter(const void *parameter, const char *parameterName)
{
    if (parameter == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, parameterName, "nullptr",
                                              "required parameter must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckNotNull(const aclTensor *blockRes, const aclTensor *validBlocks, const aclTensor *pseudoQuery,
                         const aclTensor *numerator, const aclTensor *logitMax, const aclTensor *expSum,
                         uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (CheckRequiredParameter(blockRes, "blockRes") != ACLNN_SUCCESS ||
        CheckRequiredParameter(validBlocks, "validBlocks") != ACLNN_SUCCESS ||
        CheckRequiredParameter(pseudoQuery, "pseudoQuery") != ACLNN_SUCCESS ||
        CheckRequiredParameter(numerator, "numerator") != ACLNN_SUCCESS ||
        CheckRequiredParameter(logitMax, "logitMax") != ACLNN_SUCCESS ||
        CheckRequiredParameter(expSum, "expSum") != ACLNN_SUCCESS ||
        CheckRequiredParameter(workspaceSize, "workspaceSize") != ACLNN_SUCCESS ||
        CheckRequiredParameter(executor, "executor") != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckTensorDtype(const aclTensor *tensor, const char *tensorName, DataType expectedDtype)
{
    const DataType actualDtype = tensor->GetDataType();
    if (actualDtype != expectedDtype) {
        OP_LOGE_FOR_INVALID_DTYPE("aclnnBlockAttnResPrepareGetWorkspaceSize", tensorName,
                                  op::ToString(actualDtype).GetString(), op::ToString(expectedDtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtype(const aclTensor *blockRes, const aclTensor *validBlocks, const aclTensor *pseudoQuery,
                       const aclTensor *numerator, const aclTensor *logitMax, const aclTensor *expSum)
{
    if (CheckTensorDtype(blockRes, "blockRes", DataType::DT_FLOAT) != ACLNN_SUCCESS ||
        CheckTensorDtype(validBlocks, "validBlocks", DataType::DT_UINT64) != ACLNN_SUCCESS ||
        CheckTensorDtype(pseudoQuery, "pseudoQuery", DataType::DT_FLOAT) != ACLNN_SUCCESS ||
        CheckTensorDtype(numerator, "numerator", DataType::DT_FLOAT) != ACLNN_SUCCESS ||
        CheckTensorDtype(logitMax, "logitMax", DataType::DT_FLOAT) != ACLNN_SUCCESS ||
        CheckTensorDtype(expSum, "expSum", DataType::DT_FLOAT) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckTensorContiguous(const aclTensor *tensor, const char *tensorName)
{
    if (!IsContiguous(tensor)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be contiguous, but got shape %s and strides %s.", tensorName,
                op::ToString(tensor->GetViewShape()).GetString(), op::ToString(tensor->GetViewStrides()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckContiguous(const aclTensor *blockRes, const aclTensor *validBlocks, const aclTensor *pseudoQuery,
                            const aclTensor *numerator, const aclTensor *logitMax, const aclTensor *expSum)
{
    if (CheckTensorContiguous(blockRes, "blockRes") != ACLNN_SUCCESS ||
        CheckTensorContiguous(validBlocks, "validBlocks") != ACLNN_SUCCESS ||
        CheckTensorContiguous(pseudoQuery, "pseudoQuery") != ACLNN_SUCCESS ||
        CheckTensorContiguous(numerator, "numerator") != ACLNN_SUCCESS ||
        CheckTensorContiguous(logitMax, "logitMax") != ACLNN_SUCCESS ||
        CheckTensorContiguous(expSum, "expSum") != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckTensorDimension(const aclTensor *tensor, const char *tensorName, size_t expectedDimension)
{
    const auto &shape = tensor->GetViewShape();
    const size_t actualDimension = shape.GetDimNum();
    if (actualDimension != expectedDimension) {
        OP_LOGE_FOR_INVALID_SHAPEDIM("aclnnBlockAttnResPrepareGetWorkspaceSize", tensorName,
                                     (std::to_string(actualDimension) + "D").c_str(),
                                     (std::to_string(expectedDimension) + "D").c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckInputDimensions(const aclTensor *blockRes, const aclTensor *validBlocks, const aclTensor *pseudoQuery)
{
    if (CheckTensorDimension(blockRes, "blockRes", BLOCK_RES_DIM_NUM) != ACLNN_SUCCESS ||
        CheckTensorDimension(validBlocks, "validBlocks", VALID_BLOCKS_DIM_NUM) != ACLNN_SUCCESS ||
        CheckTensorDimension(pseudoQuery, "pseudoQuery", PSEUDO_QUERY_DIM_NUM) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckOutputDimensions(const aclTensor *numerator, const aclTensor *logitMax, const aclTensor *expSum)
{
    if (CheckTensorDimension(numerator, "numerator", NUMERATOR_DIM_NUM) != ACLNN_SUCCESS ||
        CheckTensorDimension(logitMax, "logitMax", STATS_DIM_NUM) != ACLNN_SUCCESS ||
        CheckTensorDimension(expSum, "expSum", STATS_DIM_NUM) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckInputShapes(const aclTensor *blockRes, const aclTensor *validBlocks, const aclTensor *pseudoQuery)
{
    const auto &blockResShape = blockRes->GetViewShape();
    const auto &validBlocksShape = validBlocks->GetViewShape();
    const auto &pseudoQueryShape = pseudoQuery->GetViewShape();
    const int64_t totalT = blockResShape.GetDim(T_DIM_INDEX);
    const int64_t totalN = blockResShape.GetDim(N_DIM_INDEX);
    const int64_t totalD = blockResShape.GetDim(D_DIM_INDEX);
    const int64_t totalS = pseudoQueryShape.GetDim(S_DIM_INDEX);

    OP_CHECK(totalT >= 0,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnBlockAttnResPrepareGetWorkspaceSize", "blockRes.shape[0]",
                                                   std::to_string(totalT).c_str(),
                                                   "blockRes.shape[0] must be greater than or equal to 0"),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        totalN >= MIN_BLOCK_NUM && totalN <= MAX_BLOCK_NUM,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnBlockAttnResPrepareGetWorkspaceSize", "blockRes.shape[1]",
                                              std::to_string(totalN).c_str(), "blockRes.shape[1] must be in [1, 64]"),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        totalD >= MIN_HEAD_DIM && totalD <= MAX_HEAD_DIM,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnBlockAttnResPrepareGetWorkspaceSize", "blockRes.shape[2]",
                                              std::to_string(totalD).c_str(), "blockRes.shape[2] must be in [1, 8192]"),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(totalS >= 0,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnBlockAttnResPrepareGetWorkspaceSize", "pseudoQuery.shape[0]",
                                                   std::to_string(totalS).c_str(),
                                                   "pseudoQuery.shape[0] must be greater than or equal to 0"),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        validBlocksShape.GetDim(VALID_BLOCKS_VALUE_DIM_INDEX) == 1,
        OP_LOGE_FOR_INVALID_VALUE("aclnnBlockAttnResPrepareGetWorkspaceSize", "validBlocks.shape[0]",
                                  std::to_string(validBlocksShape.GetDim(VALID_BLOCKS_VALUE_DIM_INDEX)).c_str(), "1"),
        return ACLNN_ERR_PARAM_INVALID);
    const int64_t pseudoQueryD = pseudoQueryShape.GetDim(PSEUDO_QUERY_D_DIM_INDEX);
    OP_CHECK(pseudoQueryD == totalD,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                 "aclnnBlockAttnResPrepareGetWorkspaceSize", "pseudoQuery.shape[1], blockRes.shape[2]",
                 (std::to_string(pseudoQueryD) + ", " + std::to_string(totalD)).c_str(),
                 "pseudoQuery.shape[1] must equal blockRes.shape[2]"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckOutputShapes(const aclTensor *blockRes, const aclTensor *pseudoQuery, const aclTensor *numerator,
                              const aclTensor *logitMax, const aclTensor *expSum)
{
    const auto &blockResShape = blockRes->GetViewShape();
    const auto &pseudoQueryShape = pseudoQuery->GetViewShape();
    const auto &numeratorShape = numerator->GetViewShape();
    const auto &logitMaxShape = logitMax->GetViewShape();
    const auto &expSumShape = expSum->GetViewShape();
    const int64_t totalT = blockResShape.GetDim(T_DIM_INDEX);
    const int64_t totalD = blockResShape.GetDim(D_DIM_INDEX);
    const int64_t totalS = pseudoQueryShape.GetDim(S_DIM_INDEX);
    if (numeratorShape.GetDim(S_DIM_INDEX) != totalS || numeratorShape.GetDim(NUMERATOR_T_DIM_INDEX) != totalT ||
        numeratorShape.GetDim(D_DIM_INDEX) != totalD) {
        const std::string incorrectShape = std::to_string(numeratorShape.GetDim(S_DIM_INDEX)) + ", " +
                                           std::to_string(numeratorShape.GetDim(NUMERATOR_T_DIM_INDEX)) + ", " +
                                           std::to_string(numeratorShape.GetDim(D_DIM_INDEX));
        const std::string correctShape =
            std::to_string(totalS) + ", " + std::to_string(totalT) + ", " + std::to_string(totalD);
        OP_LOGE_FOR_INVALID_SHAPE("aclnnBlockAttnResPrepareGetWorkspaceSize", "numerator", incorrectShape.c_str(),
                                  correctShape.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (logitMaxShape.GetDim(S_DIM_INDEX) != totalS || logitMaxShape.GetDim(STATS_T_DIM_INDEX) != totalT) {
        const std::string incorrectShape = std::to_string(logitMaxShape.GetDim(S_DIM_INDEX)) + ", " +
                                           std::to_string(logitMaxShape.GetDim(STATS_T_DIM_INDEX));
        const std::string correctShape = std::to_string(totalS) + ", " + std::to_string(totalT);
        OP_LOGE_FOR_INVALID_SHAPE("aclnnBlockAttnResPrepareGetWorkspaceSize", "logitMax", incorrectShape.c_str(),
                                  correctShape.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (expSumShape.GetDim(S_DIM_INDEX) != totalS || expSumShape.GetDim(STATS_T_DIM_INDEX) != totalT) {
        const std::string incorrectShape = std::to_string(expSumShape.GetDim(S_DIM_INDEX)) + ", " +
                                           std::to_string(expSumShape.GetDim(STATS_T_DIM_INDEX));
        const std::string correctShape = std::to_string(totalS) + ", " + std::to_string(totalT);
        OP_LOGE_FOR_INVALID_SHAPE("aclnnBlockAttnResPrepareGetWorkspaceSize", "expSum", incorrectShape.c_str(),
                                  correctShape.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckShape(const aclTensor *blockRes, const aclTensor *validBlocks, const aclTensor *pseudoQuery,
                       const aclTensor *numerator, const aclTensor *logitMax, const aclTensor *expSum)
{
    if (CheckInputDimensions(blockRes, validBlocks, pseudoQuery) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (CheckOutputDimensions(numerator, logitMax, expSum) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (CheckInputShapes(blockRes, validBlocks, pseudoQuery) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return CheckOutputShapes(blockRes, pseudoQuery, numerator, logitMax, expSum);
}

aclnnStatus CheckParams(const aclTensor *blockRes, const aclTensor *validBlocks, const aclTensor *pseudoQuery,
                        const aclTensor *numerator, const aclTensor *logitMax, const aclTensor *expSum, float eps,
                        uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckNotNull(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum, workspaceSize, executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    if (!std::isfinite(eps) || eps <= 0.0F) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnBlockAttnResPrepareGetWorkspaceSize", "eps",
                                              std::to_string(eps).c_str(), "eps must be finite and greater than zero");
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(CheckDtype(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckContiguous(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    return CheckShape(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum);
}

} // namespace

extern "C" {
aclnnStatus aclnnBlockAttnResPrepareGetWorkspaceSize(const aclTensor *blockRes, const aclTensor *validBlocks,
                                                     const aclTensor *pseudoQuery, aclTensor *numerator,
                                                     aclTensor *logitMax, aclTensor *expSum, float eps,
                                                     uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnBlockAttnResPrepare, DFX_IN(blockRes, validBlocks, pseudoQuery, eps),
                   DFX_OUT(numerator, logitMax, expSum));

    CHECK_RET(CheckNotNull(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum, workspaceSize, executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "BlockAttnResPrepare only supports Ascend 950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorImpl = uniqueExecutor.get();

    const aclnnStatus checkRet =
        CheckParams(blockRes, validBlocks, pseudoQuery, numerator, logitMax, expSum, eps, workspaceSize, executor);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    blockRes = l0op::Contiguous(blockRes, executorImpl);
    validBlocks = l0op::Contiguous(validBlocks, executorImpl);
    pseudoQuery = l0op::Contiguous(pseudoQuery, executorImpl);
    CHECK_RET(blockRes != nullptr && validBlocks != nullptr && pseudoQuery != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const auto outputs = l0op::BlockAttnResPrepare(blockRes, validBlocks, pseudoQuery, eps, executorImpl);
    CHECK_RET(outputs[NUMERATOR_OUTPUT_INDEX] != nullptr && outputs[LOGIT_MAX_OUTPUT_INDEX] != nullptr &&
                  outputs[EXP_SUM_OUTPUT_INDEX] != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[NUMERATOR_OUTPUT_INDEX], numerator, executorImpl) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[LOGIT_MAX_OUTPUT_INDEX], logitMax, executorImpl) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[EXP_SUM_OUTPUT_INDEX], expSum, executorImpl) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBlockAttnResPrepare(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBlockAttnResPrepare);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // extern "C"
