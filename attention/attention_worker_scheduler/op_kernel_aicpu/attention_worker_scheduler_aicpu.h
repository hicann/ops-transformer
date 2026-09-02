/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_KERNEL_AICPU_ATTENTION_WORKER_SCHEDULER_H
#define OP_KERNEL_AICPU_ATTENTION_WORKER_SCHEDULER_H

#include "cpu_kernel.h"
#include "cpu_types.h"
#include "attention_ffn_schedule.h"

namespace aicpu {
class AttentionWorkerSchedulerKernel : public CpuKernel {
public:
    AttentionWorkerSchedulerKernel() = default;
    ~AttentionWorkerSchedulerKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    void WaitForFfnCompletion(AttentionDataDesc *data_desc_ptr, uint64_t flag_num, uint64_t last_ready_cnt);
    uint32_t InitAndCheck(CpuKernelContext &ctx);
    uint32_t InitAndCheckScheduleContext();
    uint32_t InitAndCheckScheduleContextCommon();
    uint32_t InitAndCheckScheduleContextAttention();

    ScheduleContext *schedule_context_ = nullptr;
    uint32_t micro_batch_num_ = 0;
    uint32_t micro_batch_size_ = 0;
    uint32_t selected_expert_num_ = 0;
    uint64_t per_data_desc_size_ = 0;
    int8_t *token_info_buf_ = nullptr;
    uint64_t token_info_buf_size_ = 0;
    uint32_t micro_batch_id_ = 0;
};
} // namespace aicpu
#endif // OP_KERNEL_AICPU_ATTENTION_WORKER_SCHEDULER_H
