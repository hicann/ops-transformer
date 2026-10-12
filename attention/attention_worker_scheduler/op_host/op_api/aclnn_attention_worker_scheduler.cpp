/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_attention_worker_scheduler.h"
#include "util/tensor_util.h"
#include "attention_worker_scheduler.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus CheckParams(aclTensor* scheduleContextRef)
{
    OP_CHECK(
        scheduleContextRef->GetViewShape().GetDimNum() == 1,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The input shape must be one-dimensional with the shape (1024), but got %s.",
                op::ToString(scheduleContextRef->GetViewShape()).GetString()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize(aclTensor* scheduleContextRef, uint64_t* workspaceSize,
                                                                 aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceAttentionWorkerScheduler, DFX_IN(scheduleContextRef), DFX_OUT());
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    OP_CHECK_NULL(scheduleContextRef, return ACLNN_ERR_PARAM_NULLPTR);

    // 固定写法，参数检查
    auto ret = CheckParams(scheduleContextRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (scheduleContextRef->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnInplaceAttentionWorkerScheduler do not support empty tensor!");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto output = l0op::AttentionWorkerScheduler(scheduleContextRef, scheduleContextRef, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceAttentionWorkerScheduler(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceAttentionWorkerScheduler);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
