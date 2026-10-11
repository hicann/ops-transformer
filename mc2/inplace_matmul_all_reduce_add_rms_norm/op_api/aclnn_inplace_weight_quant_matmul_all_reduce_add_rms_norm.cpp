/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_inplace_weight_quant_matmul_all_reduce_add_rms_norm.h"
#include "matmul_all_reduce_add_rms_norm/op_api/aclnn_weight_quant_matmul_all_reduce_add_rms_norm.h"
#include "securec.h"

#include "acl/acl.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* antiquantScale,
    const aclTensor* antiquantOffset, const aclTensor* residual, const aclTensor* gamma, double epsilon,
    const char* group, const char* reduceOp, int64_t commTurn, int64_t streamMode, int64_t antiquantGroupSize,
    const aclTensor* normOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(
        x1, x2, bias, antiquantScale, antiquantOffset, residual, gamma, epsilon, group, reduceOp, commTurn, streamMode,
        antiquantGroupSize, residual, normOut, workspaceSize, executor);
}

aclnnStatus aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm(void* workspace, uint64_t workspaceSize,
                                                             aclOpExecutor* executor, const aclrtStream stream)
{
    return aclnnWeightQuantMatmulAllReduceAddRmsNorm(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
