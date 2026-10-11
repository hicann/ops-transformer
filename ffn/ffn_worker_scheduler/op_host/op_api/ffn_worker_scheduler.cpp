/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ffn_worker_scheduler.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(FfnWorkerScheduler);

const aclTensor* FfnWorkerScheduler(const aclTensor* scheduleContext, int32_t syncGroupSize, int32_t executeMode,
                                    aclTensor* scheduleContextOut, aclOpExecutor* executor)
{
    L0_DFX(FfnWorkerScheduler, scheduleContext, syncGroupSize, executeMode, scheduleContextOut);
    static internal::AicpuTaskSpace space("FfnWorkerScheduler");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(FfnWorkerScheduler, OP_ATTR_NAMES({"sync_group_size", "execute_mode"}),
                                          OP_INPUT(scheduleContext), OP_OUTPUT(scheduleContextOut),
                                          OP_ATTR(syncGroupSize, executeMode));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FfnWorkerScheduler ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
    return scheduleContextOut;
}

} // namespace l0op
