/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_ffn_worker_batching.h"
#include "aclnnInner_ffn_worker_batching.h"

extern "C" {
ACLNN_API aclnnStatus aclnnFfnWorkerBatchingGetWorkspaceSize(
    const aclTensor* scheduleContext, int64_t expertNum, const aclIntArray* maxOutShape, int64_t tokenDtype,
    int64_t needSchedule, int64_t layerNum, const aclTensor* y, const aclTensor* groupList, const aclTensor* sessionIds,
    const aclTensor* microBatchIds, const aclTensor* tokenIds, const aclTensor* expertOffsets,
    const aclTensor* dynamicScale, const aclTensor* actualTokenNum, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 公共V1符号必须保留旧参数布局。不能将syncFlag插入该签名，否则旧二进制的
    // y指针会被解释为属性，后续输出参数全部错位。仅在转发到内部接口时补默认值。
    constexpr bool SYNC_FLAG_WAIT_ALL = false;
    return aclnnInnerFfnWorkerBatchingGetWorkspaceSize(
        scheduleContext, expertNum, maxOutShape, tokenDtype, needSchedule, layerNum, SYNC_FLAG_WAIT_ALL, y, groupList,
        sessionIds, microBatchIds, tokenIds, expertOffsets, dynamicScale, actualTokenNum, workspaceSize, executor);
}

ACLNN_API aclnnStatus aclnnFfnWorkerBatching(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             aclrtStream stream)
{
    return aclnnInnerFfnWorkerBatching(workspace, workspaceSize, executor, stream);
}
}
