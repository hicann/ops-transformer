/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_ffn_worker_batching_v2.h"
#include "aclnnInner_ffn_worker_batching.h"

extern "C" {
ACLNN_API aclnnStatus aclnnFfnWorkerBatchingV2GetWorkspaceSize(
    const aclTensor* scheduleContext, int64_t expertNum, const aclIntArray* maxOutShape, int64_t tokenDtype,
    int64_t needSchedule, int64_t layerNum, bool syncFlag, const aclTensor* y, const aclTensor* groupList,
    const aclTensor* sessionIds, const aclTensor* microBatchIds, const aclTensor* tokenIds,
    const aclTensor* expertOffsets, const aclTensor* dynamicScale, const aclTensor* actualTokenNum,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 只有V2公共接口接收syncFlag。内部接口随OpDef生成，集中处理参数校验、
    // workspace和执行器，不复制scheduleContext，保留内核对调度状态的原地更新。
    return aclnnInnerFfnWorkerBatchingGetWorkspaceSize(
        scheduleContext, expertNum, maxOutShape, tokenDtype, needSchedule, layerNum, syncFlag, y, groupList, sessionIds,
        microBatchIds, tokenIds, expertOffsets, dynamicScale, actualTokenNum, workspaceSize, executor);
}

ACLNN_API aclnnStatus aclnnFfnWorkerBatchingV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream)
{
    return aclnnInnerFfnWorkerBatching(workspace, workspaceSize, executor, stream);
}
}
