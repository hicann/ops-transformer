/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_lightning_indexer_grad.cpp
 * \brief
 */

#include "aclnn_lightning_indexer_grad.h"
#include "lightning_indexer_grad.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

struct LightningIndexerGradParams {
    const aclTensor* query = nullptr;
    const aclTensor* key = nullptr;
    const aclTensor* dy = nullptr;
    const aclTensor* sparseIndices = nullptr;
    const aclTensor* weights = nullptr;
    const aclTensor* actualSeqLengthsQueryOptional = nullptr;
    const aclTensor* actualSeqLengthsKeyOptional = nullptr;
    int64_t headNum;
    char* layout;
    int64_t sparseMode;
    int64_t preTokens;
    int64_t nextTokens;
    bool deterministic;
    const aclTensor* dqOut = nullptr;
    const aclTensor* dkOut = nullptr;
    const aclTensor* dweightsOut = nullptr;
};

static aclnnStatus CheckParams(const LightningIndexerGradParams& params)
{
    CHECK_COND(params.query != nullptr, ACLNN_ERR_PARAM_NULLPTR, "query must not be nullptr.");
    CHECK_COND(params.key != nullptr, ACLNN_ERR_PARAM_NULLPTR, "key must not be nullptr.");
    CHECK_COND(params.dy != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dy must not be nullptr.");
    CHECK_COND(params.sparseIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR, "sparseIndices must not be nullptr.");
    CHECK_COND(params.weights != nullptr, ACLNN_ERR_PARAM_NULLPTR, "weights must not be nullptr.");

    CHECK_COND(params.dqOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dqOut must not be nullptr.");
    CHECK_COND(params.dkOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dkOut must not be nullptr.");
    CHECK_COND(params.dweightsOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dweightsOut must not be nullptr.");
    return ACLNN_SUCCESS;
}

aclnnStatus ContiguousAndLightningIndexerGrad(const LightningIndexerGradParams& params, aclOpExecutor* executor,
                                              const aclTensor* dqOut, const aclTensor* dkOut,
                                              const aclTensor* dweightsOut)
{
    auto queryContiguous = l0op::Contiguous(params.query, executor);
    CHECK_RET(queryContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyContiguous = l0op::Contiguous(params.key, executor);
    CHECK_RET(keyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dyContiguous = l0op::Contiguous(params.dy, executor);
    CHECK_RET(dyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto sparseIndicesContiguous = l0op::Contiguous(params.sparseIndices, executor);
    CHECK_RET(sparseIndicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto weightsContiguous = l0op::Contiguous(params.weights, executor);
    CHECK_RET(weightsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* actualSeqLengthsQueryOptionalContiguous = nullptr;
    if (params.actualSeqLengthsQueryOptional != nullptr) {
        actualSeqLengthsQueryOptionalContiguous = l0op::Contiguous(params.actualSeqLengthsQueryOptional, executor);
        CHECK_RET(actualSeqLengthsQueryOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* actualSeqLengthsKeyOptionalContiguous = nullptr;
    if (params.actualSeqLengthsKeyOptional != nullptr) {
        actualSeqLengthsKeyOptionalContiguous = l0op::Contiguous(params.actualSeqLengthsKeyOptional, executor);
        CHECK_RET(actualSeqLengthsKeyOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    string inputLayoutStr = op::ToString(params.layout).GetString();
    // call l0 interface
    auto result = l0op::LightningIndexerGrad(
        queryContiguous, keyContiguous, dyContiguous, sparseIndicesContiguous, weightsContiguous,
        actualSeqLengthsQueryOptionalContiguous, actualSeqLengthsKeyOptionalContiguous, params.headNum,
        inputLayoutStr.c_str(), params.sparseMode, params.preTokens, params.nextTokens, params.deterministic, executor);

    // convert output tensor to contiguous tensor
    CHECK_RET(result[0] != nullptr && result[1] != nullptr && result[2] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(result[0], dqOut, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    viewCopyResult = l0op::ViewCopy(result[1], dkOut, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    viewCopyResult = l0op::ViewCopy(result[2], dweightsOut, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static bool CheckDataType(const aclTensor* query, const aclTensor* key, const aclTensor* dy, const aclTensor* weights)
{
    const DataType qDtype = query->GetDataType();
    const DataType kDtype = key->GetDataType();
    const DataType dyDtype = dy->GetDataType();
    const DataType weightsDtype = weights->GetDataType();

    static const std::unordered_map<DataType, std::vector<DataType>> validKvType = {
        {DataType::DT_FLOAT16, {DataType::DT_FLOAT16}},
        {DataType::DT_BF16, {DataType::DT_BF16}},
    };

    auto iter = validKvType.find(qDtype);
    if (iter == validKvType.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported query datatype %d.", static_cast<int>(qDtype));
        return false;
    }

    if (std::find(iter->second.begin(), iter->second.end(), kDtype) == iter->second.end() ||
        std::find(iter->second.begin(), iter->second.end(), dyDtype) == iter->second.end() ||
        std::find(iter->second.begin(), iter->second.end(), weightsDtype) == iter->second.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Key/Dy/Weights datatype mismatch with query.");
        return false;
    }

    return true;
}

static aclnnStatus ValidateParams(const aclTensor* query, const aclTensor* key, const aclTensor* dy,
                                  const aclTensor* sparseIndices, const aclTensor* weights,
                                  const aclTensor* actualSeqQLenOptional, const aclTensor* actualSeqKvLenOptional,
                                  int64_t headNum, char* inputLayout, int64_t sparseMode, int64_t preTokens,
                                  int64_t nextTokens, bool deterministic)
{
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(dy != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(sparseIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(weights != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    if (!CheckDataType(query, key, dy, weights)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLightningIndexerGradGetWorkspaceSize(const aclTensor* query, const aclTensor* key, const aclTensor* dy,
                                                      const aclTensor* sparseIndices, const aclTensor* weights,
                                                      const aclTensor* actualSeqQLenOptional,
                                                      const aclTensor* actualSeqKvLenOptional, int64_t headNum,
                                                      char* inputLayout, int64_t sparseMode, int64_t preTokens,
                                                      int64_t nextTokens, bool deterministic, const aclTensor* dqOut,
                                                      const aclTensor* dkOut, const aclTensor* dweightsOut,
                                                      uint64_t* workspaceSize, aclOpExecutor** executor)
{
    aclnnStatus validateRet =
        ValidateParams(query, key, dy, sparseIndices, weights, actualSeqQLenOptional, actualSeqKvLenOptional, headNum,
                       inputLayout, sparseMode, preTokens, nextTokens, deterministic);
    if (validateRet != ACLNN_SUCCESS) {
        return validateRet;
    }

    L2_DFX_PHASE_1(aclnnLightningIndexerGrad,
                   DFX_IN(query, key, dy, sparseIndices, weights, actualSeqQLenOptional, actualSeqKvLenOptional,
                          headNum, inputLayout, sparseMode, preTokens, nextTokens, deterministic),
                   DFX_OUT(dqOut, dkOut, dweightsOut));
    LightningIndexerGradParams params{
        query,   key,         dy,         sparseIndices, weights,    actualSeqQLenOptional, actualSeqKvLenOptional,
        headNum, inputLayout, sparseMode, preTokens,     nextTokens, deterministic,         dqOut,
        dkOut,   dweightsOut};
    // check params
    aclnnStatus ret = CheckParams(params);
    CHECK_COND(ret == ACLNN_SUCCESS, ret, "aclnnLightningIndexerGradGetWorkspaceSize checkParams failed.");
    // create OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if (query->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // call l0 interface
    ret = ContiguousAndLightningIndexerGrad(params, uniqueExecutor.get(), dqOut, dkOut, dweightsOut);
    CHECK_COND(ret == ACLNN_SUCCESS, ret, "ContiguousAndLightningIndexerGrad failed.");
    // get workspace size
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLightningIndexerGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLightningIndexerGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
