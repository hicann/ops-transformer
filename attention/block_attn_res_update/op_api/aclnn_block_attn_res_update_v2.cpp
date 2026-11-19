/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_block_attn_res_update_v2.h"
#include "block_attn_res_update_common.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

extern "C" aclnnStatus aclnnBlockAttnResUpdateV2GetWorkspaceSize(
    aclTensor* partialBlockRef, const aclTensor* delta, const aclTensor* pseudoQuery, const aclTensor* numerator,
    const aclTensor* logitMax, const aclTensor* expSum, const aclTensor* outputNormWeightOptional, double eps,
    double outputNormEps, aclTensor* h, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBlockAttnResUpdateV2,
                   DFX_IN(partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum, outputNormWeightOptional,
                          eps, outputNormEps),
                   DFX_OUT(partialBlockRef, h));
    return BlockAttnResUpdateAclnn::GetWorkspaceSize(__func__,
                                                     {partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum,
                                                      eps, h, outputNormWeightOptional, outputNormEps},
                                                     workspaceSize, executor);
}

extern "C" aclnnStatus aclnnBlockAttnResUpdateV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBlockAttnResUpdateV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
