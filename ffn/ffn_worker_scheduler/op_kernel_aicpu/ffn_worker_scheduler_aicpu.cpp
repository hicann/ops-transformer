/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ffn_worker_scheduler_aicpu.h"

#include <sched.h>
#include "securec.h"
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace aicpu {
namespace {
constexpr const char* kFfnWorkerScheduler = "FfnWorkerScheduler";
} // namespace

uint32_t FfnWorkerSchedulerKernel::InitAndCheckScheduleContextCommon()
{
    // check schedule_mode.
    KERNEL_CHECK_FALSE((schedule_context_->common.schedule_mode == kScheduleModeFfn),
                       aicpu::KERNEL_STATUS_PARAM_INVALID, "schedule mode[%u] must be 0",
                       schedule_context_->common.schedule_mode);
    KERNEL_CHECK_FALSE((schedule_context_->common.session_num > 0), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "session num[%u] must be > 0", schedule_context_->common.session_num);
    KERNEL_CHECK_FALSE((schedule_context_->common.micro_batch_num > 0), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "micro batch num[%u] must be > 0", schedule_context_->common.micro_batch_num);
    KERNEL_CHECK_FALSE((schedule_context_->common.micro_batch_size > 0), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "micro batch size[%u] must be > 0", schedule_context_->common.micro_batch_size);
    KERNEL_CHECK_FALSE((schedule_context_->common.selected_expert_num > 0), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "selected expert num[%u] must be > 0", schedule_context_->common.selected_expert_num);
    session_num_ = schedule_context_->common.session_num;
    micro_batch_num_ = schedule_context_->common.micro_batch_num;
    micro_batch_size_ = schedule_context_->common.micro_batch_size;
    selected_expert_num_ = schedule_context_->common.selected_expert_num;
    KERNEL_LOG_INFO("InitAndCheck schedule context common end. session num=%u, "
                    "micro batch num=%u, micro batch size=%u, selected expert num=%u",
                    session_num_, micro_batch_num_, micro_batch_size_, selected_expert_num_);
    return aicpu::KERNEL_STATUS_OK;
}

uint32_t FfnWorkerSchedulerKernel::InitAndCheckScheduleContextFfnInput()
{
    // check buf ptr
    KERNEL_CHECK_FALSE((schedule_context_->ffn.token_info_buf != 0), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "token info buf[%lu] can not be nullptr", schedule_context_->ffn.token_info_buf);
    KERNEL_CHECK_FALSE((schedule_context_->ffn.token_data_buf != 0), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "token data buf[%lu] can not be nullptr", schedule_context_->ffn.token_data_buf);

    KERNEL_CHECK_FALSE((schedule_context_->ffn.layer_ids_buf != 0), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "layer ids buf[%u] can not be nullptr", schedule_context_->ffn.layer_ids_buf);

    KERNEL_CHECK_FALSE((schedule_context_->ffn.session_ids_buf != 0), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "session ids buf[%u] can not be nullptr", schedule_context_->ffn.session_ids_buf);

    KERNEL_CHECK_FALSE((schedule_context_->ffn.micro_batch_ids_buf != 0), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "micro batch ids buf[%u] can not be nullptr", schedule_context_->ffn.micro_batch_ids_buf);

    KERNEL_CHECK_FALSE((schedule_context_->ffn.expert_ids_buf != 0), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "expert ids buf[%u] can not be nullptr", schedule_context_->ffn.expert_ids_buf);

    KERNEL_CHECK_FALSE(CheckUint64MulOverflow(sizeof(int32_t) * micro_batch_size_, selected_expert_num_),
                       KERNEL_STATUS_INNER_ERROR, "uint64 mul over flow");
    // check buf size
    per_ffn_data_desc_size_ = sizeof(FfnDataDesc) + sizeof(int32_t) * micro_batch_size_ * selected_expert_num_;

    KERNEL_CHECK_FALSE(CheckUint64MulOverflow(session_num_ * micro_batch_num_, per_ffn_data_desc_size_),
                       KERNEL_STATUS_INNER_ERROR, "uint64 mul over flow");

    uint64_t expect_token_info_buf_size =
        static_cast<uint64_t>(session_num_) * micro_batch_num_ * per_ffn_data_desc_size_;

    KERNEL_CHECK_FALSE((schedule_context_->ffn.token_info_buf_size >= expect_token_info_buf_size),
                       aicpu::KERNEL_STATUS_PARAM_INVALID, "token info buf size[%lu] must be at least %lu",
                       schedule_context_->ffn.token_info_buf_size, expect_token_info_buf_size);

    KERNEL_LOG_INFO("Init and check input's schedule context ffn end. token_info_buf_size_=%lu", token_info_buf_size_);
    return aicpu::KERNEL_STATUS_OK;
}

uint32_t FfnWorkerSchedulerKernel::InitAndCheckScheduleContextFfn()
{
    uint64_t expect_buf_size = static_cast<uint64_t>(session_num_) * sizeof(int32_t);
    KERNEL_CHECK_FALSE((schedule_context_->ffn.layer_ids_buf_size >= expect_buf_size),
                       aicpu::KERNEL_STATUS_PARAM_INVALID, "layer ids buf size[%lu] must be at least %lu",
                       schedule_context_->ffn.layer_ids_buf_size, expect_buf_size);
    KERNEL_CHECK_FALSE((schedule_context_->ffn.session_ids_buf_size >= expect_buf_size),
                       aicpu::KERNEL_STATUS_PARAM_INVALID, "session ids buf size[%lu] must be at least %lu",
                       schedule_context_->ffn.session_ids_buf_size, expect_buf_size);
    KERNEL_CHECK_FALSE((schedule_context_->ffn.micro_batch_ids_buf_size >= expect_buf_size),
                       aicpu::KERNEL_STATUS_PARAM_INVALID, "micro batch ids buf size[%lu] must be at least %lu",
                       schedule_context_->ffn.micro_batch_ids_buf_size, expect_buf_size);
    KERNEL_CHECK_FALSE(CheckUint64MulOverflow(session_num_ * micro_batch_num_, selected_expert_num_ * sizeof(int32_t)),
                       KERNEL_STATUS_INNER_ERROR, "uint64 mul over flow");
    uint64_t expect_expert_id_buf_size =
        static_cast<uint64_t>(session_num_) * micro_batch_size_ * selected_expert_num_ * sizeof(int32_t);
    KERNEL_CHECK_FALSE((schedule_context_->ffn.expert_ids_buf_size >= expect_expert_id_buf_size),
                       aicpu::KERNEL_STATUS_PARAM_INVALID, "expert ids buf size[%lu] must be at least %lu",
                       schedule_context_->ffn.expert_ids_buf_size, expect_expert_id_buf_size);

    token_info_buf_ = reinterpret_cast<int8_t*>(static_cast<uintptr_t>(schedule_context_->ffn.token_info_buf));
    token_info_buf_size_ = schedule_context_->ffn.token_info_buf_size;
    layer_ids_buf_ = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(schedule_context_->ffn.layer_ids_buf));
    layer_ids_buf_size_ = schedule_context_->ffn.layer_ids_buf_size;
    session_ids_buf_ = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(schedule_context_->ffn.session_ids_buf));
    session_ids_buf_size_ = schedule_context_->ffn.session_ids_buf_size;
    micro_batch_ids_buf_ =
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(schedule_context_->ffn.micro_batch_ids_buf));
    micro_batch_ids_buf_size_ = schedule_context_->ffn.micro_batch_ids_buf_size;
    expert_ids_buf_ = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(schedule_context_->ffn.expert_ids_buf));
    expert_ids_buf_size_ = schedule_context_->ffn.expert_ids_buf_size;

    KERNEL_CHECK_FALSE((schedule_context_->ffn.polling_index < micro_batch_num_), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "polling index[%lu] must be less than micro_batch_num %zu", schedule_context_->ffn.polling_index,
                       micro_batch_num_);

    KERNEL_LOG_INFO("Init and check schedule context ffn end."
                    "layer ids buf size=%lu, session ids buf size=%lu, "
                    "micro batch ids buf size=%lu, expert ids buf size=%lu",
                    layer_ids_buf_size_, session_ids_buf_size_, micro_batch_ids_buf_size_, expert_ids_buf_size_);
    return aicpu::KERNEL_STATUS_OK;
}

uint32_t FfnWorkerSchedulerKernel::InitAndCheckScheduleContext()
{
    KERNEL_CHECK_NULLPTR(schedule_context_, KERNEL_STATUS_PARAM_INVALID, "schedule context is nullptr");
    KERNEL_CHECK_ERROR(InitAndCheckScheduleContextCommon());
    KERNEL_CHECK_ERROR(InitAndCheckScheduleContextFfnInput());
    KERNEL_CHECK_ERROR(InitAndCheckScheduleContextFfn());
    return KERNEL_STATUS_OK;
}

uint32_t FfnWorkerSchedulerKernel::InitAndCheck(CpuKernelContext& ctx)
{
    Tensor* schedule_context_tensor = ctx.Input(0);

    KERNEL_CHECK_NULLPTR(schedule_context_tensor, KERNEL_STATUS_PARAM_INVALID,
                         "[%s] input[0] schedule context tensor is nullptr", ctx.GetOpType().c_str());
    KERNEL_CHECK_FALSE(schedule_context_tensor->GetTensorShape()->GetDims() == 1, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] input[0] schedule context tensor rank must be 1, but now is [%d]", ctx.GetOpType().c_str(),
                       schedule_context_tensor->GetTensorShape()->GetDims());
    KERNEL_CHECK_FALSE(schedule_context_tensor->GetDataSize() >= sizeof(ScheduleContext), KERNEL_STATUS_PARAM_INVALID,
                       "schedule context tensor data size[%zu] must >= [%zu]", schedule_context_tensor->GetDataSize(),
                       sizeof(ScheduleContext));
    schedule_context_ = static_cast<ScheduleContext*>(schedule_context_tensor->GetData());
    uint32_t ret = InitAndCheckScheduleContext();
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    auto sync_group_size_attr = ctx.GetAttr("sync_group_size");
    sync_group_size_ = session_num_;
    if (sync_group_size_attr != nullptr) {
        int64_t sync_group_size_attr_value = sync_group_size_attr->GetInt();

        KERNEL_CHECK_FALSE(
            (sync_group_size_attr_value > 0) && (sync_group_size_attr_value <= static_cast<int64_t>(session_num_)),
            KERNEL_STATUS_PARAM_INVALID, "sync group size[%ld] out of range [1, %u]", sync_group_size_attr_value,
            session_num_);

        sync_group_size_ = static_cast<uint32_t>(sync_group_size_attr_value);
        // the session num must be evently divisible by the synchronization group size
        KERNEL_CHECK_FALSE((session_num_ % sync_group_size_ == 0), KERNEL_STATUS_PARAM_INVALID,
                           "session num[%u] divided by sync group size[%u] must be "
                           "divisible exactly",
                           session_num_, sync_group_size_);
    }

    group_num_ = session_num_ / sync_group_size_;

    auto execute_mode_attr = ctx.GetAttr("execute_mode");
    if (execute_mode_attr != nullptr) {
        int64_t execute_mode = execute_mode_attr->GetInt();
        KERNEL_CHECK_FALSE((execute_mode == kExecuteModeOnce), KERNEL_STATUS_PARAM_INVALID,
                           "execute mode[%ld] must be once[%d]", execute_mode, kExecuteModeOnce);
        execute_mode_ = static_cast<int32_t>(execute_mode);
    }

    return KERNEL_STATUS_OK;
}

bool FfnWorkerSchedulerKernel::CheckSessionsReady(uint32_t micro_batch_id, uint32_t start_session_id,
                                                  uint32_t end_session_id) const
{
    const uint64_t data_desc_size_per_session = per_ffn_data_desc_size_ * micro_batch_num_;
    // start from micro_batch of session 0
    const int8_t* check_start_ffn_desc = token_info_buf_ + micro_batch_id * per_ffn_data_desc_size_;
    for (uint32_t i = start_session_id; i <= end_session_id; ++i) {
        auto flag_ptr =
            reinterpret_cast<const volatile int32_t*>(check_start_ffn_desc + i * data_desc_size_per_session);
        if (*flag_ptr != kValidFlag) {
            KERNEL_LOG_EVENT("Find invalid flag. session num=%u, micro batch num=%u, "
                             "micro batch size=%u, selected expert num=%u, "
                             "micro batch id=%u, session id=%u, flag=%d",
                             session_num_, micro_batch_num_, micro_batch_size_, selected_expert_num_, micro_batch_id, i,
                             *flag_ptr);
            return false;
        }
    }
    return true;
}

void FfnWorkerSchedulerKernel::CopyAndResetSessions(uint32_t micro_batch_id, uint32_t start_session_id,
                                                    uint32_t end_session_id, uint32_t out_idx) const
{
    const uint64_t data_desc_size_per_session = per_ffn_data_desc_size_ * micro_batch_num_;
    // start from micro_batch of session 0
    int8_t* start_ffn_desc = token_info_buf_ + micro_batch_id * per_ffn_data_desc_size_;
    const uint32_t expert_info_num_per_session = micro_batch_size_ * selected_expert_num_;
    const size_t expert_id_buf_len = sizeof(int32_t) * expert_info_num_per_session;
    for (uint32_t i = start_session_id; i <= end_session_id; ++i) {
        auto moe_data_desc = PtrToPtr<int8_t, FfnDataDesc>(start_ffn_desc + i * data_desc_size_per_session);
        layer_ids_buf_[out_idx] = moe_data_desc->layer_id;
        session_ids_buf_[out_idx] = static_cast<int32_t>(i);
        micro_batch_ids_buf_[out_idx] = static_cast<int32_t>(micro_batch_id);
        auto* dst_expert_ids_buf = &(expert_ids_buf_[out_idx * expert_info_num_per_session]);
        auto* expert_id_buf = const_cast<int32_t*>(moe_data_desc->expert_ids);
        auto cpy_ret = memcpy_s(dst_expert_ids_buf, expert_id_buf_len, expert_id_buf, expert_id_buf_len);
        if (cpy_ret != EOK) {
            KERNEL_LOG_ERROR("memcpy_s expert_ids failed, cpy_ret=%d, expert id buf len=%zu", cpy_ret,
                             expert_id_buf_len);
            return;
        }
        // reset value to invalid
        auto set_ret = memset_s(expert_id_buf, expert_id_buf_len, 0x7F, expert_id_buf_len);
        if (set_ret != EOK) {
            KERNEL_LOG_ERROR("memset_s expert_ids failed, set_ret=%d, expert id buf len=%zu", set_ret,
                             expert_id_buf_len);
            return;
        }
        moe_data_desc->flag = kInvalidFlag;
        ++out_idx;
        KERNEL_LOG_INFO("session[%u-%u] micro batch id[%u] ready.", start_session_id, end_session_id, micro_batch_id);
    }
}

bool FfnWorkerSchedulerKernel::CheckHasReadyGroup() const
{
    for (uint32_t group_id = 0; group_id < group_num_; ++group_id) {
        uint32_t start_session_id = group_id * sync_group_size_;
        uint32_t end_session_id = start_session_id + sync_group_size_ - 1U;
        for (uint32_t micro_batch_id = 0; micro_batch_id < micro_batch_num_; ++micro_batch_id) {
            if (CheckSessionsReady(micro_batch_id, start_session_id, end_session_id)) {
                return true;
            }
        }
    }
    return false;
}

void FfnWorkerSchedulerKernel::ToOutput(CpuKernelContext& ctx, uint32_t micro_batch_id, uint32_t start_session_id,
                                        uint32_t end_session_id) const
{
    uint32_t start_out_idx = schedule_context_->ffn.out_num;
    auto sharder = [micro_batch_id, start_session_id, start_out_idx, this](uint32_t start, uint32_t end) {
        CopyAndResetSessions(micro_batch_id, start_session_id + start, start_session_id + end - 1U,
                             start_out_idx + start);
    };
    (void)CpuKernelUtils::ParallelFor(ctx, end_session_id - start_session_id + 1, 1, sharder);

    KERNEL_LOG_INFO("session[%u-%u] micro batch id[%u] ready.", start_session_id, end_session_id, micro_batch_id);
    schedule_context_->ffn.out_num += end_session_id - start_session_id + 1U;
}

void FfnWorkerSchedulerKernel::CheckHandleSessions(CpuKernelContext& ctx, uint32_t start_session_id,
                                                   uint32_t end_session_id) const
{
    for (uint32_t micro_batch_id = 0; micro_batch_id < micro_batch_num_; ++micro_batch_id) {
        if (CheckSessionsReady(micro_batch_id, start_session_id, end_session_id)) {
            ToOutput(ctx, micro_batch_id, start_session_id, end_session_id);
            // handle one batch for each group
            break;
        }
    }
}

uint32_t FfnWorkerSchedulerKernel::DoCompute(CpuKernelContext& ctx)
{
    // wait one group ready
    while (schedule_context_->control.run_flag != 0) {
        if (CheckHasReadyGroup()) {
            break;
        }
        (void)sched_yield();
    }

    if (schedule_context_->control.run_flag == 0) {
        KERNEL_LOG_INFO("compute %s exit because run flag=0.", kFfnWorkerScheduler);
        return KERNEL_STATUS_OK;
    }

    for (uint32_t group_id = 0; group_id < group_num_; ++group_id) {
        uint32_t start_session_id = group_id * sync_group_size_;
        uint32_t end_session_id = start_session_id + sync_group_size_ - 1U;
        CheckHandleSessions(ctx, start_session_id, end_session_id);
    }

    return KERNEL_STATUS_OK;
}

uint32_t FfnWorkerSchedulerKernel::DoComputeAllSync(CpuKernelContext& ctx)
{
    while (schedule_context_->control.run_flag != 0) {
        // all sync check by polling index.
        if (CheckSessionsReady(schedule_context_->ffn.polling_index, 0U, session_num_ - 1U)) {
            ToOutput(ctx, schedule_context_->ffn.polling_index, 0U, session_num_ - 1U);
            schedule_context_->ffn.polling_index = (schedule_context_->ffn.polling_index + 1UL) % micro_batch_num_;
            // handle one batch for each time
            return KERNEL_STATUS_OK;
        }
    }
    KERNEL_LOG_INFO("compute %s exit because run flag=0.", kFfnWorkerScheduler);
    return KERNEL_STATUS_OK;
}

uint32_t FfnWorkerSchedulerKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_CHECK_ERROR(InitAndCheck(ctx));
    KERNEL_LOG_INFO("compute %s begin, group num is %u.", kFfnWorkerScheduler, group_num_);
    schedule_context_->ffn.out_num = 0U;

    uint32_t ret = KERNEL_STATUS_OK;
    if (group_num_ == 1U) {
        ret = DoComputeAllSync(ctx);
    } else {
        ret = DoCompute(ctx);
    }
    if (ret != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("compute %s failed, ret=%u", kFfnWorkerScheduler, ret);
        return ret;
    }
    KERNEL_LOG_INFO("compute %s end. out num=%u", kFfnWorkerScheduler, schedule_context_->ffn.out_num);
    return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kFfnWorkerScheduler, FfnWorkerSchedulerKernel);
} // namespace aicpu
