/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "block_attention_residuals_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BlockAttentionResidualsGrad);

const aclTensor *BlockAttentionResidualsGrad(const aclTensor *partialBlock, const aclTensor *blockRes,
                                             const aclTensor *projWeight, const aclTensor *normWeight,
                                             const aclTensor *gradHiddenStates, const aclTensor *invNorm,
                                             const aclTensor *probs, int64_t validBlockNum, aclTensor *gradPartialBlock,
                                             aclTensor *gradBlockRes, aclTensor *gradProjWeight,
                                             aclTensor *gradNormWeight, aclOpExecutor *executor)
{
    L0_DFX(BlockAttentionResidualsGrad, partialBlock, blockRes, projWeight, normWeight, gradHiddenStates, invNorm,
           probs, validBlockNum, gradPartialBlock, gradBlockRes, gradProjWeight, gradNormWeight);

    auto ret =
        INFER_SHAPE(BlockAttentionResidualsGrad,
                    OP_INPUT(partialBlock, blockRes, projWeight, normWeight, gradHiddenStates, invNorm, probs),
                    OP_OUTPUT(gradPartialBlock, gradBlockRes, gradProjWeight, gradNormWeight), OP_ATTR(validBlockNum));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID, "BlockAttentionResidualsGrad INFER_SHAPE failed.");
        return nullptr;
    }

    auto launcherRet = ADD_TO_LAUNCHER_LIST_AICORE(
        BlockAttentionResidualsGrad,
        OP_INPUT(partialBlock, blockRes, projWeight, normWeight, gradHiddenStates, invNorm, probs),
        OP_OUTPUT(gradPartialBlock, gradBlockRes, gradProjWeight, gradNormWeight), OP_ATTR(validBlockNum));
    if (launcherRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID,
                "BlockAttentionResidualsGrad ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }

    return gradPartialBlock;
}

} // namespace l0op
