/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_FFN_WORKER_BATCHING_V2_H_
#define OP_API_INC_FFN_WORKER_BATCHING_V2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 获取FfnWorkerBatching的workspace大小及执行器。
 * syncFlag为bool类型：false等待全部session，true消费ready快照；仅Ascend 950接收模式生效。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnFfnWorkerBatchingV2GetWorkspaceSize(
    const aclTensor* scheduleContext, int64_t expertNum, const aclIntArray* maxOutShape, int64_t tokenDtype,
    int64_t needSchedule, int64_t layerNum, bool syncFlag, const aclTensor* y, const aclTensor* groupList,
    const aclTensor* sessionIds, const aclTensor* microBatchIds, const aclTensor* tokenIds,
    const aclTensor* expertOffsets, const aclTensor* dynamicScale, const aclTensor* actualTokenNum,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/** @brief 执行第一段接口构造的任务。 */
ACLNN_API aclnnStatus aclnnFfnWorkerBatchingV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
