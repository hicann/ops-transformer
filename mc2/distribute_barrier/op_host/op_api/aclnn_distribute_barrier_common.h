/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNN_DISTRIBUTE_BARRIER_COMMON
#define ACLNN_DISTRIBUTE_BARRIER_COMMON

#include <algorithm>

#include "aclnn_kernels/common/op_error_check.h"
#include "op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif
enum NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0U,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerDistributeBarrierGetWorkspaceSize(
    const aclTensor* xRef, const aclTensor* timeOut,
    const aclTensor* elasticInfo, const char* group,
    int64_t worldSize, uint64_t* workspaceSize,
    aclOpExecutor** executor);
extern aclnnStatus aclnnInnerDistributeBarrier(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

static inline aclnnStatus aclnnDistributeBarrierCommon(void* workspace, uint64_t workspaceSize,
                                                        aclOpExecutor* executor, aclrtStream stream) {
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerDistributeBarrier(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif

#endif  // ACLNN_DISTRIBUTE_BARRIER_COMMON