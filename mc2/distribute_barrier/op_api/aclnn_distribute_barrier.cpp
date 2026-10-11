/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>

#include "aclnn_distribute_barrier.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "distribute_barrier_base.h"
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnDistributeBarrierGetWorkspaceSize(aclTensor* xRef, const char* group, int64_t worldSize,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnn_barrier WorkspaceSize start");
    return aclnnDistributeBarrierGetWorkspaceSizeBase(xRef, nullptr, nullptr, group, worldSize, workspaceSize,
                                                      executor);
}

aclnnStatus aclnnDistributeBarrier(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    OP_LOGD("aclnn_barrier start");
    return aclnnDistributeBarrierBase(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
