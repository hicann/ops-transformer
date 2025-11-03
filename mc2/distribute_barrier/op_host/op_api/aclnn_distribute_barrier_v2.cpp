/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_distribute_barrier_common.h"
#include "aclnn_distribute_barrier_v2.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// check nullptr
static bool CheckNullStatus(const aclTensor* xRef, const char* group) {
  // 检查必选入参出参为非空
  OP_LOGD("aclnn_distribute_barrier_v2 CheckNotNull start");
  OP_CHECK_NULL(xRef, return false);
  if (group == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name is Empty.");
    return false;
  }
  OP_LOGD("aclnn_distribute_barrier_v2 CheckNotNull end");
  return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor* xRef, const char* group) {
  OP_LOGD("aclnn_distribute_barrier_v2 checkParams start");
  CHECK_RET(CheckNullStatus(xRef, group), ACLNN_ERR_PARAM_NULLPTR);
  auto groupStrnLen = strnlen(group, HCCL_GROUP_NAME_MAX);
  if ((groupStrnLen >= HCCL_GROUP_NAME_MAX) || (groupStrnLen == 0)) {
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
            "Required group name length in range (0, HCCL_GROUP_NAME_MAX), but "
            "it's %zu.",
            strnlen(group, HCCL_GROUP_NAME_MAX));
    return false;
  }
  OP_LOGD("aclnn_distribute_barrier_v2 checkParams success");
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnDistributeBarrierV2GetWorkspaceSize(const aclTensor* xRef, const aclTensor* timeOut,
                                                     const aclTensor* elasticInfo, const char* group,
                                                     int64_t worldSize, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor)

{
  auto retParam = CheckParams(xRef, group);
  CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
  return aclnnInnerDistributeBarrierGetWorkspaceSize(xRef, timeOut, elasticInfo,
                                                     group, worldSize, workspaceSize, executor);
}

aclnnStatus aclnnDistributeBarrierV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream) {
  return aclnnDistributeBarrierCommon(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif