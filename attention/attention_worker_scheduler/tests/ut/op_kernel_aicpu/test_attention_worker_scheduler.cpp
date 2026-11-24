/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "attention_ffn_schedule.h"
#undef private
#undef protected
#include <string>
#include <cstring>
#include <iostream>
using namespace std;
using namespace aicpu;
class TEST_ATTENTION_WORKER_SCHEDULER_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas) \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "AttentionWorkerScheduler", "AttentionWorkerScheduler") \
        .Input({"schedule_context", data_types[0], shapes[0], datas[0]}) \
        .Output({"schedule_context", data_types[0], shapes[0], datas[0]});

TEST_F(TEST_ATTENTION_WORKER_SCHEDULER_UT, INPUT_ALL_FLAG_SUCC)
{
    ScheduleContext input_context = {};

    // Initialize  CommonArea
    input_context.common.session_num = 1;
    input_context.common.micro_batch_num = 2;
    input_context.common.micro_batch_size = 6;
    input_context.common.selected_expert_num = 2;
    input_context.common.expert_num = 16;
    input_context.common.attn_to_ffn_token_size = 512;
    input_context.common.ffn_to_attn_token_size = 512;
    input_context.common.schedule_mode = 1; // attention

    // Initialize ControlArea
    input_context.control.run_flag = 1; // running

    // Initialize AttentionArea
    input_context.attention.micro_batch_id = 1;
    size_t per_data_desc_size = sizeof(AttentionDataDesc) + sizeof(int32_t) * input_context.common.micro_batch_size *
                                                                input_context.common.selected_expert_num;

    size_t expect_token_info_buf_size = static_cast<size_t>(input_context.common.micro_batch_num) * per_data_desc_size;
    input_context.attention.token_info_buf = reinterpret_cast<uint64_t>(new uint64_t[expect_token_info_buf_size]);
    input_context.attention.token_info_buf_size = expect_token_info_buf_size;
    int target_micro_batch_id = 0;
    auto data_desc_ptr =
        reinterpret_cast<AttentionDataDesc*>(reinterpret_cast<uint8_t*>(input_context.attention.token_info_buf) +
                                             per_data_desc_size * target_micro_batch_id);

    size_t flag_num =
        static_cast<size_t>(input_context.common.micro_batch_size) * input_context.common.selected_expert_num;

    // Set all flags to 1.
    for (size_t i = 0; i < flag_num; ++i) {
        data_desc_ptr->flag[i] = 1;
    }

    int64_t* token_data = new int64_t[100];
    input_context.attention.token_data_buf = reinterpret_cast<uint64_t>(token_data);

    ScheduleContext output_context = {};

    vector<DataType> data_types = {DT_UINT8, DT_UINT8};
    vector<vector<int64_t>> shapes = {{sizeof(ScheduleContext)}, {sizeof(ScheduleContext)}};

    vector<void*> datas = {(void*)&input_context, (void*)&output_context};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] reinterpret_cast<uint64_t*>(input_context.attention.token_info_buf);
    delete[] token_data;
    EXPECT_EQ(output_context.attention.micro_batch_id, 0);
}

TEST_F(TEST_ATTENTION_WORKER_SCHEDULER_UT, INPUT_SCHEDULE_MODE_INVALID)
{
    ScheduleContext input_context = {};

    // Initialize  CommonArea
    input_context.common.session_num = 1;
    input_context.common.micro_batch_num = 4;
    input_context.common.micro_batch_size = 32;
    input_context.common.selected_expert_num = 8;
    input_context.common.expert_num = 16;
    input_context.common.attn_to_ffn_token_size = 512;
    input_context.common.ffn_to_attn_token_size = 512;
    input_context.common.schedule_mode = 2; // invalid

    // Initialize ControlArea
    input_context.control.run_flag = 1; // running

    // Initialize AttentionArea
    input_context.attention.micro_batch_id = 0;

    ScheduleContext output_context = {};

    vector<DataType> data_types = {DT_UINT8, DT_UINT8};
    vector<vector<int64_t>> shapes = {{sizeof(ScheduleContext)}, {sizeof(ScheduleContext)}};

    vector<void*> datas = {(void*)&input_context, (void*)&output_context};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ATTENTION_WORKER_SCHEDULER_UT, INPUT_RUN_FLAG_PROCESS_TERMINATE)
{
    ScheduleContext input_context = {};

    // Initialize  CommonArea
    input_context.common.session_num = 1;
    input_context.common.micro_batch_num = 2;
    input_context.common.micro_batch_size = 6;
    input_context.common.selected_expert_num = 2;
    input_context.common.expert_num = 16;
    input_context.common.attn_to_ffn_token_size = 512;
    input_context.common.ffn_to_attn_token_size = 512;
    input_context.common.schedule_mode = 1; // attention

    // Initialize ControlArea
    input_context.control.run_flag = 0; // Process termination

    // Initialize AttentionArea
    input_context.attention.micro_batch_id = 0;
    size_t per_data_desc_size = sizeof(AttentionDataDesc) + sizeof(int32_t) * input_context.common.micro_batch_size *
                                                                input_context.common.selected_expert_num;

    size_t expect_token_info_buf_size = static_cast<size_t>(input_context.common.micro_batch_num) * per_data_desc_size;
    input_context.attention.token_info_buf = reinterpret_cast<uint64_t>(new uint64_t[expect_token_info_buf_size]);
    input_context.attention.token_info_buf_size = expect_token_info_buf_size;
    int64_t* token_data = new int64_t[100];
    input_context.attention.token_data_buf = reinterpret_cast<uint64_t>(token_data);
    ScheduleContext output_context = {};

    vector<DataType> data_types = {DT_UINT8, DT_UINT8};
    vector<vector<int64_t>> shapes = {{sizeof(ScheduleContext)}, {sizeof(ScheduleContext)}};

    vector<void*> datas = {(void*)&input_context, (void*)&output_context};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] reinterpret_cast<uint64_t*>(input_context.attention.token_info_buf);
    delete[] token_data;
    EXPECT_EQ(output_context.attention.micro_batch_id, 0);
}

TEST_F(TEST_ATTENTION_WORKER_SCHEDULER_UT, INPUT_DIM_INVALID)
{
    ScheduleContext input_context = {};

    // Initialize  CommonArea
    input_context.common.session_num = 1;
    input_context.common.micro_batch_num = 4;
    input_context.common.micro_batch_size = 32;
    input_context.common.selected_expert_num = 8;
    input_context.common.expert_num = 16;
    input_context.common.attn_to_ffn_token_size = 512;
    input_context.common.ffn_to_attn_token_size = 512;
    input_context.common.schedule_mode = 2; // invalid

    // Initialize ControlArea
    input_context.control.run_flag = 1; // running

    // Initialize AttentionArea
    input_context.attention.micro_batch_id = 0;

    ScheduleContext output_context = {};

    vector<DataType> data_types = {DT_UINT8, DT_UINT8};
    vector<vector<int64_t>> shapes = {{sizeof(ScheduleContext), 2}, {sizeof(ScheduleContext)}};

    vector<void*> datas = {(void*)&input_context, (void*)&output_context};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

namespace {
void RunAttentionWorkerSchedulerWithBufSize(uint32_t micro_batch_num, uint32_t micro_batch_size,
                                            uint32_t selected_expert_num, uint64_t token_info_buf_size,
                                            uint32_t expect_ret)
{
    ScheduleContext input_context = {};

    input_context.common.session_num = 1;
    input_context.common.micro_batch_num = micro_batch_num;
    input_context.common.micro_batch_size = micro_batch_size;
    input_context.common.selected_expert_num = selected_expert_num;
    input_context.common.expert_num = 16;
    input_context.common.attn_to_ffn_token_size = 512;
    input_context.common.ffn_to_attn_token_size = 512;
    input_context.common.schedule_mode = 1; // attention

    input_context.control.run_flag = 0; // early exit, only the buffer size check is exercised

    input_context.attention.micro_batch_id = 0;
    input_context.attention.token_info_buf = reinterpret_cast<uint64_t>(new uint8_t[64]);
    input_context.attention.token_info_buf_size = token_info_buf_size;
    input_context.attention.token_data_buf = reinterpret_cast<uint64_t>(new uint8_t[64]);

    ScheduleContext output_context = {};

    vector<DataType> data_types = {DT_UINT8, DT_UINT8};
    vector<vector<int64_t>> shapes = {{sizeof(ScheduleContext)}, {sizeof(ScheduleContext)}};
    vector<void*> datas = {(void*)&input_context, (void*)&output_context};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, expect_ret);

    delete[] reinterpret_cast<uint8_t*>(input_context.attention.token_info_buf);
    delete[] reinterpret_cast<uint8_t*>(input_context.attention.token_data_buf);
}
} // namespace

// micro_batch_size * selected_expert_num == 2^32, which truncates to 0 in 32-bit multiplication.
// The declared 64 bytes must be rejected; the real requirement is 2 * (0 + 4 * 2^32) = 32 GiB.
TEST_F(TEST_ATTENTION_WORKER_SCHEDULER_UT, INPUT_TOKEN_INFO_BUF_SIZE_2POW32_TRUNCATION_REJECTED)
{
    RunAttentionWorkerSchedulerWithBufSize(2, 65536, 65536, 64, KERNEL_STATUS_PARAM_INVALID);
}

// micro_batch_size * selected_expert_num > 2^32 and the truncated value is non-zero.
// Truncation alone still under-reports the real 34360262656 bytes, so 524288 bytes must be rejected.
TEST_F(TEST_ATTENTION_WORKER_SCHEDULER_UT, INPUT_TOKEN_INFO_BUF_SIZE_TRUNCATION_NONZERO_REJECTED)
{
    RunAttentionWorkerSchedulerWithBufSize(2, 65537, 65536, 524288, KERNEL_STATUS_PARAM_INVALID);
}

// Boundary: the exact required size is 2 * (0 + 4 * 4 * 2) = 64 bytes, one byte less must be rejected.
TEST_F(TEST_ATTENTION_WORKER_SCHEDULER_UT, INPUT_TOKEN_INFO_BUF_SIZE_ONE_BYTE_SHORT_REJECTED)
{
    RunAttentionWorkerSchedulerWithBufSize(2, 4, 2, 63, KERNEL_STATUS_PARAM_INVALID);
}

// 4 * micro_batch_size * selected_expert_num exceeds UINT64_MAX and must report an inner error.
TEST_F(TEST_ATTENTION_WORKER_SCHEDULER_UT, INPUT_TOKEN_INFO_BUF_SIZE_FLAG_BYTES_OVERFLOW)
{
    RunAttentionWorkerSchedulerWithBufSize(1, 0xFFFFFFFFU, 0xFFFFFFFFU, 64, KERNEL_STATUS_INNER_ERROR);
}

// micro_batch_num * (sizeof(AttentionDataDesc) + 4 * micro_batch_size * selected_expert_num) overflows UINT64.
TEST_F(TEST_ATTENTION_WORKER_SCHEDULER_UT, INPUT_TOKEN_INFO_BUF_SIZE_TOTAL_OVERFLOW)
{
    RunAttentionWorkerSchedulerWithBufSize(2, 0x80000000U, 0x40000000U, 0xFFFFFFFFFFFFFFFFULL,
                                           KERNEL_STATUS_INNER_ERROR);
}
