/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_KERNEL_AICPU_FFN_WORKER_SCHEDULER_H
#define OP_KERNEL_AICPU_FFN_WORKER_SCHEDULER_H

#include "attention_ffn_schedule.h"
#include "cpu_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class FfnWorkerSchedulerKernel : public CpuKernel {
public:
    FfnWorkerSchedulerKernel() = default;
    ~FfnWorkerSchedulerKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    uint32_t DoCompute(CpuKernelContext &ctx);
    uint32_t DoComputeAllSync(CpuKernelContext &ctx);
    void ToOutput(CpuKernelContext &ctx, uint32_t micro_batch_id, uint32_t start_session_id,
                  uint32_t end_session_id) const;
    uint32_t InitAndCheckScheduleContextCommon();
    uint32_t InitAndCheckScheduleContextFfnInput();
    uint32_t InitAndCheckScheduleContextFfn();
    uint32_t InitAndCheckScheduleContext();
    uint32_t InitAndCheck(CpuKernelContext &ctx);
    bool CheckSessionsReady(uint32_t micro_batch_id, uint32_t start_session_id, uint32_t end_session_id) const;
    void CopyAndResetSessions(uint32_t micro_batch_id, uint32_t start_session_id, uint32_t end_session_id,
                              uint32_t out_idx) const;
    bool CheckHasReadyGroup() const;
    void CheckHandleSessions(CpuKernelContext &ctx, uint32_t start_session_id, uint32_t end_session_id) const;
    int32_t execute_mode_ = kExecuteModeOnce;
    ScheduleContext *schedule_context_ = nullptr;
    uint32_t session_num_ = 0;
    uint32_t micro_batch_num_ = 0;
    uint32_t micro_batch_size_ = 0;
    uint32_t selected_expert_num_ = 0;
    uint32_t sync_group_size_ = 0;
    uint32_t group_num_ = 1;
    uint64_t per_ffn_data_desc_size_ = 0;
    int8_t *token_info_buf_ = nullptr;
    uint64_t token_info_buf_size_ = 0;
    int32_t *layer_ids_buf_ = nullptr;
    uint64_t layer_ids_buf_size_ = 0;
    int32_t *session_ids_buf_ = nullptr;
    uint64_t session_ids_buf_size_ = 0;
    int32_t *micro_batch_ids_buf_ = nullptr;
    uint64_t micro_batch_ids_buf_size_ = 0;
    int32_t *expert_ids_buf_ = nullptr;
    uint64_t expert_ids_buf_size_ = 0;
};
} // namespace aicpu
#endif // OP_KERNEL_AICPU_FFN_WORKER_SCHEDULER_H
