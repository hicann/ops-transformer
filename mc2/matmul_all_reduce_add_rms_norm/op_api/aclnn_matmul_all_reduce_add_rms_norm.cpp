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
 * \file aclnn_matmul_all_reduce_add_rms_norm.cpp
 * \brief
 */
#include "aclnn_matmul_all_reduce_add_rms_norm.h"
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
#include "mc2_log_compat.h"
#include "aclnn_kernels/contiguous.h"
#include "matmul_all_reduce_arn_util.h"
#include "aclnnInner_matmul_all_reduce_add_rms_norm.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                           const aclTensor* bias, const aclTensor* residual,
                                                           const aclTensor* gamma, double epsilon, const char* group,
                                                           const char* reduceOp, int64_t commTurn, int64_t streamMode,
                                                           const aclTensor* y, const aclTensor* normOut,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // inplace接口复用非inplace接口 告警内提示两个接口都将被废弃
    DEPRECATED_MM_ARN_API_WARN_ONCE("aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize",
                                    "aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize", "December 2026",
                                    "aclnnMatmulAllReduceGetWorkspaceSize", "aclnnAddRmsNormGetWorkspaceSize");
    // 处理空tensor
    CHECK_RET(ArnCheckNotNull(x1, x2, residual, gamma), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(ArnCheckShape(x1, x2, residual), ACLNN_ERR_PARAM_INVALID);
    if (x1->IsEmpty() || x2->IsEmpty() || residual->IsEmpty() || gamma->IsEmpty()) {
        // 处理k = 0 场景，报错
        OP_LOGD("MatmulAllReduceAddRmsNorm, OriginShape: %s; ViewShape %s.",
                op::ToString(x1->GetOriginalShape()).GetString(), op::ToString(x1->GetViewShape()).GetString());
        int64_t kValue = x1->GetViewShape().GetDim(x1->GetViewShape().GetDimNum() - 1);
        OP_LOGD("MatmulAllReduceAddRmsNorm, kValue: %ld.", kValue);
        if (kValue == 0) {
            OP_LOGE_FOR_INVALID_VALUE("aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize", "x1 k-dim", "0", "non-zero");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 根据实际支持情况补充
        OP_LOGD("MatmulAllReduceAddRmsNorm, dealing with empty tensor.");
        // 固定写法，创建OpExecutor
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // check streamMode = NUM_ACL_STOP_ON_FAILURE
    CHECK_RET(streamMode == NUM_ACL_STOP_ON_FAILURE, ACLNN_ERR_PARAM_INVALID);
    int64_t antiquantGroupSize = 0;
    aclnnStatus ret = InnerMatmulAllReduceAddRmsNormGetWorkspaceSize(
        x1, x2, bias, nullptr, nullptr, nullptr, residual, gamma, epsilon, group, reduceOp, commTurn,
        antiquantGroupSize, y, normOut, workspaceSize, executor);
    OP_LOGD("MatmulAllReduceAddRmsNorm, end ret %d", ret);
    return ret;
}

aclnnStatus aclnnMatmulAllReduceAddRmsNorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           const aclrtStream stream)
{
    DEPRECATED_MM_ARN_API_WARN_ONCE("aclnnMatmulAllReduceAddRmsNorm", "aclnnInplaceMatmulAllReduceAddRmsNorm",
                                    "December 2026", "aclnnMatmulAllReduce", "aclnnAddRmsNorm");
    if (workspace == nullptr || workspaceSize == 0UL) {
        OP_LOGD("Skip the api for empty tensor, workspace addr %p, size %lu.", workspace, workspaceSize);
        return ACLNN_SUCCESS;
    }

    aclnnStatus ret = aclnnInnerMatmulAllReduceAddRmsNorm(workspace, workspaceSize, executor, stream);
    OP_LOGD("MatmulAllReduceAddRmsNorm, aclnnMatmulAllReduceAddRmsNorm ret %d", ret);
    if (ret != 0) {
        OP_LOGE_LIBOPAPI_REPORT("aclnnMatmulAllReduceAddRmsNorm", "This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
