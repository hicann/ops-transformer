/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file distribute_barrier_base.h
 * \brief
 */
#ifndef DISTRIBUTE_BARRIER_BASE_H_
#define DISTRIBUTE_BARRIER_BASE_H_

#include <string>

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif
enum NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

// check nullptr
__attribute__((visibility("default"))) bool BarrierCheckNullStatus(const aclTensor* xRef, const char* group);
// 入参校验
__attribute__((visibility("default"))) aclnnStatus BarrierCheckParams(const aclTensor* xRef, const char* group);

__attribute__((visibility("default"))) aclnnStatus aclnnDistributeBarrierGetWorkspaceSizeBase(
    const aclTensor* xRef, const aclTensor* timeOut, const aclTensor* elasticInfo, const char* group, int64_t worldSize,
    uint64_t* workspaceSize, aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnDistributeBarrierBase(void* workspace, uint64_t workspaceSize,
                                                                              aclOpExecutor* executor,
                                                                              aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif // DISTRIBUTE_BARRIER_BASE_H_
