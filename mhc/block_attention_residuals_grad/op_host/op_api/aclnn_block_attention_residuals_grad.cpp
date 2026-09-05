/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_block_attention_residuals_grad.h"
#include "block_attention_residuals_grad.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "aclnn/aclnn_base.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"

#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnBlockAttentionResidualsGradGetWorkspaceSize(
    const aclTensor *partialBlock, const aclTensor *blockRes, const aclTensor *projWeight, const aclTensor *normWeight,
    const aclTensor *gradHiddenStates, const aclTensor *invNorm, const aclTensor *probs, int64_t validBlockNum,
    const aclTensor *gradPartialBlock, const aclTensor *gradBlockRes, const aclTensor *gradProjWeight,
    const aclTensor *gradNormWeight, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnBlockAttentionResidualsGrad,
        DFX_IN(partialBlock, blockRes, projWeight, normWeight, gradHiddenStates, invNorm, probs, validBlockNum),
        DFX_OUT(gradPartialBlock, gradBlockRes, gradProjWeight, gradNormWeight));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 与正向算子对齐：输入统一转 Contiguous 后再下发
    auto partialBlock_ = l0op::Contiguous(partialBlock, uniqueExecutor.get());
    CHECK_RET(partialBlock_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto blockRes_ = l0op::Contiguous(blockRes, uniqueExecutor.get());
    CHECK_RET(blockRes_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto projWeight_ = l0op::Contiguous(projWeight, uniqueExecutor.get());
    CHECK_RET(projWeight_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto normWeight_ = l0op::Contiguous(normWeight, uniqueExecutor.get());
    CHECK_RET(normWeight_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradHiddenState_ = l0op::Contiguous(gradHiddenStates, uniqueExecutor.get());
    CHECK_RET(gradHiddenState_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto invNorm_ = l0op::Contiguous(invNorm, uniqueExecutor.get());
    CHECK_RET(invNorm_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto probs_ = l0op::Contiguous(probs, uniqueExecutor.get());
    CHECK_RET(probs_ != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto output = l0op::BlockAttentionResidualsGrad(
        partialBlock_, blockRes_, projWeight_, normWeight_, gradHiddenState_, invNorm_, probs_, validBlockNum,
        const_cast<aclTensor *>(gradPartialBlock), const_cast<aclTensor *>(gradBlockRes),
        const_cast<aclTensor *>(gradProjWeight), const_cast<aclTensor *>(gradNormWeight), uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBlockAttentionResidualsGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream)
{
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != 0) {
        OP_LOGE(ACLNN_ERR_INNER, "BlockAttentionResidualsGrad launch failed, ret = %d.", ret);
        return ret;
    }
    return ret;
}

#ifdef __cplusplus
}
#endif
