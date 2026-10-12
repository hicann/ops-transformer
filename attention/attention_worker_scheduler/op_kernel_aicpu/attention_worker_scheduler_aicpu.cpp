/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "attention_worker_scheduler_aicpu.h"

#include <sched.h>
#include <sstream>
#include "securec.h"
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace aicpu {
namespace {
const char* const kAttentionWorkerScheduler = "AttentionWorkerScheduler";
constexpr int32_t kNopTime = 50;
} // namespace

uint32_t AttentionWorkerSchedulerKernel::InitAndCheckScheduleContextCommon()
{
    KERNEL_CHECK_FALSE((schedule_context_->common.schedule_mode == 1), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "The scheduler mode must be 'attention' with a value of "
                       "1, but the current value is %d",
                       schedule_context_->common.schedule_mode);
    KERNEL_CHECK_FALSE((schedule_context_->common.micro_batch_num > 0), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "micro_batch_num[%u] must be > 0", schedule_context_->common.micro_batch_num);
    KERNEL_CHECK_FALSE((schedule_context_->common.micro_batch_size > 0), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "micro_batch_size[%u] must be > 0", schedule_context_->common.micro_batch_size);
    KERNEL_CHECK_FALSE((schedule_context_->common.selected_expert_num > 0), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "selectedExpertNum[%u] must be > 0", schedule_context_->common.selected_expert_num);
    micro_batch_num_ = schedule_context_->common.micro_batch_num;
    micro_batch_size_ = schedule_context_->common.micro_batch_size;
    selected_expert_num_ = schedule_context_->common.selected_expert_num;
    KERNEL_LOG_INFO("InitAndCheck schedule_context common end. micro_batch_num=%u, "
                    "micro_batch_size=%u, selected_expert_num=%u",
                    micro_batch_num_, micro_batch_size_, selected_expert_num_);
    return aicpu::KERNEL_STATUS_OK;
}

uint32_t AttentionWorkerSchedulerKernel::InitAndCheckScheduleContextAttention()
{
    // check buf ptr
    KERNEL_CHECK_FALSE((schedule_context_->attention.token_info_buf != 0), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "token_info_buf[%lu] can not be 0", schedule_context_->attention.token_info_buf);
    KERNEL_CHECK_FALSE((schedule_context_->attention.token_data_buf != 0), aicpu::KERNEL_STATUS_PARAM_INVALID,
                       "token_data_buf[%lu] can not be 0", schedule_context_->attention.token_data_buf);

    KERNEL_CHECK_FALSE(CheckUint64MulOverflow(micro_batch_size_, selected_expert_num_), KERNEL_STATUS_INNER_ERROR,
                       "uint64 overflow occurred.");
    uint64_t micro_batch_expert_product = static_cast<uint64_t>(micro_batch_size_ * selected_expert_num_);
    KERNEL_CHECK_FALSE(CheckUint64MulOverflow(sizeof(int32_t), micro_batch_expert_product), KERNEL_STATUS_INNER_ERROR,
                       "uint64 overflow occurred.");
    // check buf size
    per_data_desc_size_ = sizeof(AttentionDataDesc) + sizeof(int32_t) * micro_batch_expert_product;

    KERNEL_CHECK_FALSE(CheckUint64MulOverflow(static_cast<uint64_t>(micro_batch_num_), per_data_desc_size_),
                       KERNEL_STATUS_INNER_ERROR, "uint64 overflow occurred.");
    uint64_t expect_token_info_buf_size = static_cast<uint64_t>(micro_batch_num_) * per_data_desc_size_;

    KERNEL_CHECK_FALSE((schedule_context_->attention.token_info_buf_size >= expect_token_info_buf_size),
                       aicpu::KERNEL_STATUS_PARAM_INVALID, "token_info_buf_size[%lu] must be at least %zu",
                       schedule_context_->attention.token_info_buf_size, expect_token_info_buf_size);

    token_info_buf_ = reinterpret_cast<int8_t*>(static_cast<uintptr_t>(schedule_context_->attention.token_info_buf));
    token_info_buf_size_ = schedule_context_->attention.token_info_buf_size;

    KERNEL_CHECK_FALSE((schedule_context_->attention.micro_batch_id < micro_batch_num_),
                       aicpu::KERNEL_STATUS_PARAM_INVALID, "micro_batch_id[%u] is out of range [0,%u)",
                       schedule_context_->attention.micro_batch_id, micro_batch_num_);
    micro_batch_id_ = (schedule_context_->attention.micro_batch_id + 1U) % micro_batch_num_;
    KERNEL_LOG_INFO("InitAndCheck schedule_context attention end. token_info_buf_size_=%lu, "
                    "micro_batch_id=%u",
                    token_info_buf_size_, micro_batch_id_);
    return aicpu::KERNEL_STATUS_OK;
}

uint32_t AttentionWorkerSchedulerKernel::InitAndCheckScheduleContext()
{
    KERNEL_CHECK_NULLPTR(schedule_context_, KERNEL_STATUS_PARAM_INVALID, "schedule_context nullptr");
    KERNEL_CHECK_ERROR(InitAndCheckScheduleContextCommon());
    KERNEL_CHECK_ERROR(InitAndCheckScheduleContextAttention());
    return KERNEL_STATUS_OK;
}

uint32_t AttentionWorkerSchedulerKernel::InitAndCheck(CpuKernelContext& ctx)
{
    Tensor* schedule_context_tensor = ctx.Input(0);

    KERNEL_CHECK_NULLPTR(schedule_context_tensor, KERNEL_STATUS_PARAM_INVALID,
                         "[%s] input[0] schedule context tensor nullptr", ctx.GetOpType().c_str());
    KERNEL_CHECK_FALSE(schedule_context_tensor->GetTensorShape()->GetDims() == 1, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] input[0] schedule context tensor rank must be 1, but now is [%d]", ctx.GetOpType().c_str(),
                       schedule_context_tensor->GetTensorShape()->GetDims());
    KERNEL_CHECK_FALSE(schedule_context_tensor->GetDataSize() >= sizeof(ScheduleContext), KERNEL_STATUS_PARAM_INVALID,
                       "schedule_context_tensor data size[%zu] must >= [%zu]", schedule_context_tensor->GetDataSize(),
                       sizeof(ScheduleContext));
    schedule_context_ = static_cast<ScheduleContext*>(schedule_context_tensor->GetData());
    uint32_t ret = InitAndCheckScheduleContext();
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    return KERNEL_STATUS_OK;
}

std::string FlagsToString(volatile int32_t* flags, uint32_t micro_batch_size, uint32_t selected_expert_num)
{
    std::stringstream ss;
    ss << "[";
    for (uint32_t i = 0; i < micro_batch_size; ++i) {
        if (i > 0U) {
            ss << ", ";
        }
        ss << "[";

        for (uint32_t j = 0; j < selected_expert_num; ++j) {
            if (j > 0U) {
                ss << ", ";
            }
            ss << flags[i * selected_expert_num + j];
        }
        ss << "]";
    }
    ss << "]";
    return ss.str();
}

void AttentionWorkerSchedulerKernel::WaitForFfnCompletion(AttentionDataDesc* data_desc_ptr, uint64_t flag_num,
                                                          uint64_t last_ready_cnt)
{
    while (schedule_context_->control.run_flag != 0) {
#ifdef __aarch64__
        __asm__ __volatile__("dsb ld" : : : "memory");
#endif
        uint64_t ready_cnt = 0;
        for (uint64_t i = 0; i < flag_num; ++i) {
            if (data_desc_ptr->flag[i] == kValidFlag) {
                ++ready_cnt;
            }
        }
        if (ready_cnt == flag_num) {
            KERNEL_LOG_INFO("The number of flags is %zu.", flag_num);
            break;
        }

        if (last_ready_cnt != ready_cnt) {
            last_ready_cnt = ready_cnt;
            KERNEL_LOG_EVENT("micro_batch_id[%u] ready count=%zu, need flag_num=%zu. "
                             "micro_batch_size_=%u, selected_expert_num_=%u, flags=%s.",
                             micro_batch_id_, ready_cnt, flag_num, micro_batch_size_, selected_expert_num_,
                             FlagsToString(data_desc_ptr->flag, micro_batch_size_, selected_expert_num_).c_str());
        }

        (void)sched_yield();
#ifdef __aarch64__
        for (uint64_t i = 0; i < kNopTime; ++i) {
            __asm__ __volatile__("nop");
        }
#endif
    }
}

uint32_t AttentionWorkerSchedulerKernel::Compute(CpuKernelContext& ctx)
{
    uint32_t ret = InitAndCheck(ctx);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    KERNEL_LOG_INFO("Compute %s begin, micro_batch_id=%u, micro_batch_num=%u, "
                    "micro_batch_size=%u, selected_expert_num=%u.",
                    kAttentionWorkerScheduler, micro_batch_id_, micro_batch_num_, micro_batch_size_,
                    selected_expert_num_);

    auto data_desc_ptr = PtrToPtr<int8_t, AttentionDataDesc>(token_info_buf_ + per_data_desc_size_ * micro_batch_id_);

    uint64_t flag_num = static_cast<uint64_t>(micro_batch_size_) * selected_expert_num_;

    uint64_t last_ready_cnt = 0;
    WaitForFfnCompletion(data_desc_ptr, flag_num, last_ready_cnt);
    if (schedule_context_->control.run_flag == 0) {
        KERNEL_LOG_INFO("Compute %s exit because run_flag=0, ready_cnt=%zu, flag_num=%zu.", kAttentionWorkerScheduler,
                        last_ready_cnt, flag_num);
        return KERNEL_STATUS_OK;
    }

    KERNEL_CHECK_FALSE(CheckUint64MulOverflow(flag_num, sizeof(int32_t)), KERNEL_STATUS_INNER_ERROR,
                       "uint64 overflow occurred.");
    uint64_t flag_len = flag_num * sizeof(int32_t);
    auto set_ret = memset_s(const_cast<int32_t*>(data_desc_ptr->flag), flag_len, 0, flag_len);
    if (set_ret != EOK) {
        KERNEL_LOG_ERROR("memset_s flag failed, set_ret=%d, flag_len=%zu", set_ret, flag_len);
        return KERNEL_STATUS_INNER_ERROR;
    }

    schedule_context_->attention.micro_batch_id = micro_batch_id_;
    KERNEL_LOG_INFO("Compute %s end, micro_batch_id=%u.", kAttentionWorkerScheduler, micro_batch_id_);
    return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kAttentionWorkerScheduler, AttentionWorkerSchedulerKernel);
} // namespace aicpu
