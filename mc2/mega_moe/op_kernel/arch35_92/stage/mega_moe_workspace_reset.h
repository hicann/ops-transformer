/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_WORKSPACE_RESET_H
#define MEGA_MOE_WORKSPACE_RESET_H

#include "../common/mega_moe_sync.h"
#include "../common/mega_moe_utils.h"

namespace MegaMoeImpl {

using namespace AscendC;

#if defined(MEGA_MOE_IS_KERNEL_COMPILE)
// 仅由 AIV 调用，清理布局给出的连续 flag workspace。
// 清零地址和字节范围由布局给出；在调用 int32 清零函数时转换为元素数。
__aicore__ inline void ResetSyncStatus(const AivJobContext& job, const Params& params, int32_t resetBatchElementCount,
                                       __ubuf__ int32_t* resetTensor)
{
    const auto& workspace = params.workspace.layout;
    const auto& buffer = params.workspace.base;
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs || resetBatchElementCount <= 0) {
        return;
    }
    PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID2>();
    ResetWorkspaceRegion<1>(job, workspace.reset.flags.get_ptr(buffer),
                            static_cast<int32_t>(workspace.reset.flags.sizeBytes / SIZE_INT_32), resetBatchElementCount,
                            resetTensor);
}
#endif

} // namespace MegaMoeImpl

#endif // MEGA_MOE_WORKSPACE_RESET_H
