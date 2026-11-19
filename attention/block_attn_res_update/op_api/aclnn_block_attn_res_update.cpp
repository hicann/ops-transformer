/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_block_attn_res_update.h"
#include "block_attn_res_update_common.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

extern "C" aclnnStatus aclnnBlockAttnResUpdateGetWorkspaceSize(aclTensor* partialBlockRef, const aclTensor* delta,
                                                               const aclTensor* pseudoQuery, const aclTensor* numerator,
                                                               const aclTensor* logitMax, const aclTensor* expSum,
                                                               double eps, aclTensor* h, uint64_t* workspaceSize,
                                                               aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBlockAttnResUpdate,
                   DFX_IN(partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum, eps),
                   DFX_OUT(partialBlockRef, h));
    return BlockAttnResUpdateAclnn::GetWorkspaceSize(
        __func__, {partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum, eps, h}, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnBlockAttnResUpdate(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBlockAttnResUpdate);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
