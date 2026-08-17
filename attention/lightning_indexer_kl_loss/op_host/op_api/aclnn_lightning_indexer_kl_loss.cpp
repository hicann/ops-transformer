/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_lightning_indexer_kl_loss.h"
#include "lightning_indexer_kl_loss.h"
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

struct LightningIndexerKLLossParams {
    const aclTensor *targetScore = nullptr;
    const aclTensor *indexProbs = nullptr;
    double eps;
    const char *weightType = nullptr;
    const aclTensor *loss = nullptr;
};

static bool CheckShape(const aclTensor *targetScore)
{
    const auto &viewShape = targetScore->GetViewShape();
    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    if (dimNum != 2 && dimNum != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "targetScore dimNum %ld is not 2 or 3.", dimNum);
        return false;
    }
    int64_t lastDim = viewShape.GetDim(static_cast<uint64_t>(dimNum - 1));
    if (lastDim <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "targetScore last dim %ld must be positive.", lastDim);
        return false;
    }
    if (lastDim > 8192) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "targetScore last dim %ld exceeds 8192.", lastDim);
        return false;
    }
    // 3D 输入时，第一维 B（batch）必须属于 1~512
    if (dimNum == 3) {
        int64_t batchDim = viewShape.GetDim(0);
        if (batchDim < 1 || batchDim > 512) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "targetScore batch dim %ld must be in [1, 512].", batchDim);
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(const LightningIndexerKLLossParams &params)
{
    CHECK_COND(params.targetScore != nullptr, ACLNN_ERR_PARAM_NULLPTR, "targetScore must not be nullptr.");
    CHECK_COND(params.indexProbs != nullptr, ACLNN_ERR_PARAM_NULLPTR, "indexProbs must not be nullptr.");
    CHECK_COND(params.weightType != nullptr, ACLNN_ERR_PARAM_NULLPTR, "weightType must not be nullptr.");
    CHECK_COND(params.loss != nullptr, ACLNN_ERR_PARAM_NULLPTR, "loss must not be nullptr.");

    if (!CheckShape(params.targetScore)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    // dtype 一致性校验
    if (params.targetScore->GetDataType() != params.indexProbs->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "targetScore dtype %d != indexProbs dtype %d.",
                static_cast<int32_t>(params.targetScore->GetDataType()),
                static_cast<int32_t>(params.indexProbs->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    // shape 一致性校验（逐维比较）
    const auto &targetShape = params.targetScore->GetViewShape();
    const auto &indexShape = params.indexProbs->GetViewShape();
    if (targetShape.GetDimNum() != indexShape.GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "targetScore dimNum %u != indexProbs dimNum %u.", targetShape.GetDimNum(),
                indexShape.GetDimNum());
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (uint64_t i = 0; i < targetShape.GetDimNum(); ++i) {
        if (targetShape.GetDim(i) != indexShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "targetScore shape dim[%llu] %ld != indexProbs dim[%llu] %ld.",
                    static_cast<unsigned long long>(i), targetShape.GetDim(i), static_cast<unsigned long long>(i),
                    indexShape.GetDim(i));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    std::string weightTypeStr(params.weightType);
    if (weightTypeStr != "logits" && weightTypeStr != "probs") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weightType must be 'logits' or 'probs', but got '%s'.", params.weightType);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ContiguousAndLightningIndexerKLLoss(const LightningIndexerKLLossParams &params,
                                                       aclOpExecutor *executor, const aclTensor *loss)
{
    auto targetScoreContiguous = l0op::Contiguous(params.targetScore, executor);
    CHECK_RET(targetScoreContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto indexProbsContiguous = l0op::Contiguous(params.indexProbs, executor);
    CHECK_RET(indexProbsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    string weightTypeStr = op::ToString(params.weightType).GetString();
    // call l0 interface
    auto result = l0op::LightningIndexerKLLoss(targetScoreContiguous, indexProbsContiguous, params.eps,
                                               weightTypeStr.c_str(), executor);

    // convert output tensor to contiguous tensor
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(result, loss, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLightningIndexerKLLossGetWorkspaceSize(const aclTensor *targetScore, const aclTensor *indexProbs,
                                                        double eps, const char *weightType, const aclTensor *loss,
                                                        uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnLightningIndexerKLLoss, DFX_IN(targetScore, indexProbs, eps, weightType), DFX_OUT(loss));
    LightningIndexerKLLossParams params{targetScore, indexProbs, eps, weightType, loss};
    // check params
    aclnnStatus ret = CheckParams(params);
    CHECK_COND(ret == ACLNN_SUCCESS, ret, "aclnnLightningIndexerKLLossGetWorkspaceSize checkParams failed.");

    // create OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if (targetScore->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    std::string weightTypeStr = op::ToString(params.weightType).GetString();
    // call l0 interface
    ret = ContiguousAndLightningIndexerKLLoss(params, uniqueExecutor.get(), loss);
    CHECK_COND(ret == ACLNN_SUCCESS, ret, "ContiguousAndLightningIndexerKLLoss failed.");

    // get workspace size
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLightningIndexerKLLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLightningIndexerKLLoss);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
