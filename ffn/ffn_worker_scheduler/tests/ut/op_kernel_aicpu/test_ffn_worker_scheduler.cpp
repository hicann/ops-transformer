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
namespace {
constexpr uint32_t kSuccess = 0;
constexpr uint32_t kFailure = 1;
constexpr uint64_t kBufAlignSize = 512;

inline uint64_t AlignUp(uint64_t num, uint64_t align)
{
    return ((num + align - 1) / align) * align;
}

template <typename T>
class IntegerChecker {
public:
    template <typename T1>
    static bool Compat(const T1 v)
    {
        static_assert(((sizeof(T) <= sizeof(uint64_t)) && (sizeof(T1) <= sizeof(uint64_t))),
                      "IntegerChecker can only check integers less than 64 bits");
        if (v >= static_cast<T1>(0)) {
            return static_cast<uint64_t>(v) <= static_cast<uint64_t>(std::numeric_limits<T>::max());
        }
        return static_cast<int64_t>(v) >= static_cast<int64_t>(std::numeric_limits<T>::min());
    }
};

template <typename TLhs, typename TRhs, typename TRet>
bool MulOverflow(TLhs lhs, TRhs rhs, TRet& ret)
{
#if __GNUC__ >= 5
    return __builtin_mul_overflow(lhs, rhs, &ret);
#else
    if ((!IntegerChecker<TRet>::Compat(lhs)) || (!IntegerChecker<TRet>::Compat(rhs))) {
        return true;
    }
    if ((lhs == 0) || (rhs == 0)) {
        ret = 0;
        return false;
    }
    TRet reminder = std::numeric_limits<TRet>::max() / static_cast<TRet>(rhs);
    const TRet lhs_ret_type = static_cast<TRet>(lhs);
    if (lhs_ret_type < 0) {
        if (reminder > 0) {
            reminder *= static_cast<TRet>(-1);
        }
        if (lhs_ret_type < reminder) {
            return true;
        }
    } else {
        if (reminder < 0) {
            reminder *= static_cast<TRet>(-1);
        }
        if (lhs_ret_type > reminder) {
            return true;
        }
    }
    ret = static_cast<TRet>(lhs) * static_cast<TRet>(rhs);
    return false;
#endif
}

template <typename TLhs, typename TRhs, typename TRet>
bool AddOverflow(TLhs lhs, TRhs rhs, TRet& ret)
{
#if __GNUC__ >= 5
    return __builtin_add_overflow(lhs, rhs, &ret);
#else
    if ((!IntegerChecker<TRet>::Compat(lhs)) || (!IntegerChecker<TRet>::Compat(rhs))) {
        return true;
    }
    if (rhs >= 0) {
        if (static_cast<TRet>(lhs) > std::numeric_limits<TRet>::max() - static_cast<TRet>(rhs)) {
            return true;
        }
    } else {
        if (static_cast<TRet>(lhs) < std::numeric_limits<TRet>::min() - static_cast<TRet>(rhs)) {
            return true;
        }
    }
    ret = static_cast<TRet>(lhs) + static_cast<TRet>(rhs);
    return false;
#endif
}

uint64_t CalcFfnTokenInfoSize(aicpu::ScheduleContext& schedule_context)
{
    uint64_t token_info_size = sizeof(int32_t) * static_cast<uint64_t>(schedule_context.common.selected_expert_num);
    if (MulOverflow(token_info_size, static_cast<uint64_t>(schedule_context.common.micro_batch_size),
                    token_info_size)) {
        std::cout << "check mul with micro_batch_size over flow failed." << std::endl;
        return 0UL;
    }
    uint64_t flag_and_layer_id_size = sizeof(int32_t) * 2;
    if (AddOverflow(token_info_size, flag_and_layer_id_size, token_info_size)) {
        std::cout << "check add flag and layer id over flow failed." << std::endl;
        return 0UL;
    }

    if (MulOverflow(token_info_size, static_cast<uint64_t>(schedule_context.common.micro_batch_num), token_info_size)) {
        std::cout << "check mul with micro_batch_num over flow failed." << std::endl;
        return 0UL;
    }
    if (MulOverflow(token_info_size, static_cast<uint64_t>(schedule_context.common.session_num), token_info_size)) {
        std::cout << "check mul with session_num over flow failed." << std::endl;
        return 0UL;
    }
    return token_info_size;
}

uint32_t InitFfnTokenInfoBuf(aicpu::ScheduleContext& schedule_context)
{
    std::unique_ptr<uint8_t[]> tmp_buf(new (std::nothrow) uint8_t[schedule_context.ffn.token_info_buf_size]);
    if (tmp_buf == nullptr) {
        std::cout << "alloc token info host tmp buf failed, buf_size= " << schedule_context.ffn.token_info_buf_size
                  << std::endl;
        return kFailure;
    }
    auto tmp_buf_int = reinterpret_cast<int32_t*>(tmp_buf.get());
    auto token_buf_int = tmp_buf_int;
    for (uint32_t session_id = 0; session_id < schedule_context.common.session_num; ++session_id) {
        for (uint32_t micro_batch_id = 0; micro_batch_id < schedule_context.common.micro_batch_num; ++micro_batch_id) {
            // flag
            *tmp_buf_int++ = 1;
            // layer_id
            *tmp_buf_int++ = 55;
            for (uint32_t idx = 0;
                 idx < schedule_context.common.micro_batch_size * schedule_context.common.selected_expert_num; ++idx) {
                // expert_id
                *tmp_buf_int++ = idx;
            }
        }
    }
    schedule_context.ffn.token_info_buf =
        reinterpret_cast<uint64_t>(malloc(sizeof(uint8_t) * schedule_context.ffn.token_info_buf_size));
    auto token_info_buf = reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.token_info_buf));
    auto ret = memcpy_s(token_info_buf, schedule_context.ffn.token_info_buf_size, token_buf_int,
                        schedule_context.ffn.token_info_buf_size);
    if (ret != EOK) {
        std::cout << "memory copy token info buf failed, size_= " << schedule_context.ffn.token_info_buf_size
                  << " token_info_buf ptr is ." << token_buf_int << std::endl;
        return kFailure;
    }
    return kSuccess;
}

uint32_t InitFfn(aicpu::ScheduleContext& schedule_context)
{
    uint64_t token_info_size = CalcFfnTokenInfoSize(schedule_context);
    if (token_info_size == 0U) {
        return kFailure;
    }

    uint64_t token_info_aligned_size = AlignUp(token_info_size, kBufAlignSize);
    if (token_info_aligned_size < token_info_size) {
        std::cout << "token_info_size=" << token_info_size << " overflow after align with " << kBufAlignSize << "."
                  << std::endl;
        return kFailure;
    }

    schedule_context.ffn.token_info_buf_size = token_info_size;
    schedule_context.ffn.token_data_buf_size = 1024 - token_info_aligned_size;
    schedule_context.ffn.token_data_buf =
        reinterpret_cast<uint64_t>(malloc(sizeof(uint8_t) * schedule_context.ffn.token_data_buf_size));
    auto ret = InitFfnTokenInfoBuf(schedule_context);
    if (ret != kSuccess) {
        return ret;
    }

    // calc output size
    schedule_context.ffn.layer_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num;
    schedule_context.ffn.session_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num;
    schedule_context.ffn.micro_batch_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num;
    schedule_context.ffn.expert_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num *
                                               schedule_context.common.micro_batch_size *
                                               schedule_context.common.selected_expert_num;
    schedule_context.ffn.layer_ids_buf =
        reinterpret_cast<uint64_t>(malloc(sizeof(uint8_t) * schedule_context.ffn.layer_ids_buf_size));
    schedule_context.ffn.session_ids_buf =
        reinterpret_cast<uint64_t>(malloc(sizeof(uint8_t) * schedule_context.ffn.session_ids_buf_size));
    schedule_context.ffn.micro_batch_ids_buf =
        reinterpret_cast<uint64_t>(malloc(sizeof(uint8_t) * schedule_context.ffn.micro_batch_ids_buf_size));
    schedule_context.ffn.expert_ids_buf =
        reinterpret_cast<uint64_t>(malloc(sizeof(uint8_t) * schedule_context.ffn.expert_ids_buf_size));
    std::cout << "Init ffn success, token_info_buf= " << schedule_context.ffn.token_info_buf << "token_info_buf_size=, "
              << schedule_context.ffn.token_info_buf_size << "token_data_buf= " << schedule_context.ffn.token_data_buf
              << "token_data_buf_size= ." << schedule_context.ffn.token_data_buf_size << std::endl;
    return kSuccess;
}

uint32_t UninitFfn(aicpu::ScheduleContext& schedule_context)
{
    if (schedule_context.ffn.token_info_buf != 0) {
        free(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.token_info_buf)));
        schedule_context.ffn.token_info_buf = 0;
    }
    if (schedule_context.ffn.token_data_buf != 0) {
        free(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.token_data_buf)));
        schedule_context.ffn.token_data_buf = 0;
    }
    if (schedule_context.ffn.layer_ids_buf != 0) {
        free(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.layer_ids_buf)));
        schedule_context.ffn.layer_ids_buf = 0;
    }
    if (schedule_context.ffn.session_ids_buf != 0) {
        free(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.session_ids_buf)));
        schedule_context.ffn.session_ids_buf = 0;
    }
    if (schedule_context.ffn.micro_batch_ids_buf != 0) {
        free(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.micro_batch_ids_buf)));
        schedule_context.ffn.micro_batch_ids_buf = 0;
    }
    if (schedule_context.ffn.expert_ids_buf != 0) {
        free(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.expert_ids_buf)));
        schedule_context.ffn.expert_ids_buf = 0;
    }
    return kSuccess;
}
} // namespace
class TEST_FfnWorkerScheduler_UTest : public testing::Test {};

#define CREATE_NODE_DEFINE() \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "FfnWorkerScheduler", "FfnWorkerScheduler") \
        .Input({"schedule_context", data_types[0], shapes[0], datas[0]}) \
        .Input({"stop_scan", data_types[1], shapes[1], datas[1]}) \
        .Output({"schedule_context", data_types[2], shapes[2], datas[2]}) \
        .Attr("sync_group_size", sync_group_size) \
        .Attr("execute_mode", execute_mode);

TEST_F(TEST_FfnWorkerScheduler_UTest, INPUT_COMPUTE_ALL_SYNC_SUCCESS)
{
    std::cout << "FfnWorkerScheduler start" << std::endl;
    vector<DataType> data_types = {DT_INT8, DT_INT64, DT_INT8};
    vector<vector<int64_t>> shapes = {{1024}, {}, {1024}};
    aicpu::ScheduleContext schedule_context = {};
    schedule_context.common.session_num = 2;
    schedule_context.common.micro_batch_num = 2;
    schedule_context.common.micro_batch_size = 2;
    schedule_context.common.selected_expert_num = 5;
    schedule_context.common.schedule_mode = 0;
    schedule_context.control.run_flag = 1;
    schedule_context.ffn.polling_index = 1;
    int32_t execute_mode = 0;
    int32_t sync_group_size = 2; // test
    InitFfn(schedule_context);
    int8_t* input1 = reinterpret_cast<int8_t*>(&schedule_context);
    int64_t input2[1] = {1};
    int8_t output[1024] = {0, 0, 0, 0, 0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODE_DEFINE()
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    aicpu::ScheduleContext* out_schedule_context = reinterpret_cast<aicpu::ScheduleContext*>(input1);
    int32_t* layer_ids_buf =
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(out_schedule_context->ffn.layer_ids_buf));
    for (int i = 0; i < out_schedule_context->ffn.layer_ids_buf_size / sizeof(int32_t); i++) {
        std::cout << "out_schedule_context layer_ids index = ," << i << "layer_ids = " << layer_ids_buf[i] << std::endl;
    }
    int32_t* session_ids_buf =
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(out_schedule_context->ffn.session_ids_buf));
    for (int i = 0; i < out_schedule_context->ffn.session_ids_buf_size / sizeof(int32_t); i++) {
        std::cout << "out_schedule_context session_ids index = ," << i << "session_ids = " << session_ids_buf[i]
                  << std::endl;
    }
    int32_t* micro_batch_ids_buf =
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(out_schedule_context->ffn.micro_batch_ids_buf));
    for (int i = 0; i < out_schedule_context->ffn.micro_batch_ids_buf_size / sizeof(int32_t); i++) {
        std::cout << "out_schedule_context micro_batch_ids_buf index = ," << i
                  << "micro_batch_ids_buf = " << micro_batch_ids_buf[i] << std::endl;
    }
    int32_t* expert_ids_buf =
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(out_schedule_context->ffn.expert_ids_buf));
    for (int i = 0; i < out_schedule_context->ffn.expert_ids_buf_size / sizeof(int32_t); i++) {
        std::cout << "out_schedule_context expert_ids_buf index = ," << i << "expert_ids_buf = " << expert_ids_buf[i]
                  << std::endl;
    }
    UninitFfn(schedule_context);
    std::cout << "FfnWorkerScheduler end" << std::endl;
}

TEST_F(TEST_FfnWorkerScheduler_UTest, INPUT_COMPUTE_SUCCESS)
{
    std::cout << "FfnWorkerScheduler start" << std::endl;
    vector<DataType> data_types = {DT_INT8, DT_INT64, DT_INT8};
    vector<vector<int64_t>> shapes = {{1024}, {}, {1024}};
    aicpu::ScheduleContext schedule_context = {};
    schedule_context.common.session_num = 2;
    schedule_context.common.micro_batch_num = 2;
    schedule_context.common.micro_batch_size = 2;
    schedule_context.common.selected_expert_num = 5;
    schedule_context.control.run_flag = 1;
    schedule_context.common.schedule_mode = 0;
    int32_t execute_mode = 0;
    int32_t sync_group_size = 1; // test
    InitFfn(schedule_context);
    int8_t* input1 = reinterpret_cast<int8_t*>(&schedule_context);
    int64_t input2[1] = {1};
    int8_t output[1024] = {0, 0, 0, 0, 0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODE_DEFINE()
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    aicpu::ScheduleContext* out_schedule_context = reinterpret_cast<aicpu::ScheduleContext*>(input1);
    int32_t* layer_ids_buf =
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(out_schedule_context->ffn.layer_ids_buf));
    for (int i = 0; i < out_schedule_context->ffn.layer_ids_buf_size / sizeof(int32_t); i++) {
        std::cout << "out_schedule_context layer_ids index = ," << i << "layer_ids = " << layer_ids_buf[i] << std::endl;
    }
    int32_t* session_ids_buf =
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(out_schedule_context->ffn.session_ids_buf));
    for (int i = 0; i < out_schedule_context->ffn.session_ids_buf_size / sizeof(int32_t); i++) {
        std::cout << "out_schedule_context session_ids index = ," << i << "session_ids = " << session_ids_buf[i]
                  << std::endl;
    }
    int32_t* micro_batch_ids_buf =
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(out_schedule_context->ffn.micro_batch_ids_buf));
    for (int i = 0; i < out_schedule_context->ffn.micro_batch_ids_buf_size / sizeof(int32_t); i++) {
        std::cout << "out_schedule_context micro_batch_ids_buf index = ," << i
                  << "micro_batch_ids_buf = " << micro_batch_ids_buf[i] << std::endl;
    }
    int32_t* expert_ids_buf =
        reinterpret_cast<int32_t*>(static_cast<uintptr_t>(out_schedule_context->ffn.expert_ids_buf));
    for (int i = 0; i < out_schedule_context->ffn.expert_ids_buf_size / sizeof(int32_t); i++) {
        std::cout << "out_schedule_context expert_ids_buf index = ," << i << "expert_ids_buf = " << expert_ids_buf[i]
                  << std::endl;
    }
    UninitFfn(schedule_context);
    std::cout << "FfnWorkerScheduler end" << std::endl;
}

TEST_F(TEST_FfnWorkerScheduler_UTest, INPUT_SCHEDULE_MODE_INVALID)
{
    vector<DataType> data_types = {DT_INT8, DT_INT64, DT_INT8};
    vector<vector<int64_t>> shapes = {{1024}, {}, {1024}};
    aicpu::ScheduleContext schedule_context = {};
    schedule_context.common.session_num = 2;
    schedule_context.common.micro_batch_num = 2;
    schedule_context.common.micro_batch_size = 2;
    schedule_context.common.selected_expert_num = 5;
    schedule_context.common.schedule_mode = 1; // test
    schedule_context.control.run_flag = 1;
    schedule_context.ffn.polling_index = 1;
    int32_t execute_mode = 0;
    int32_t sync_group_size = 2;
    InitFfn(schedule_context);
    int8_t* input1 = reinterpret_cast<int8_t*>(&schedule_context);
    int64_t input2[1] = {1};
    int8_t output[1024] = {0, 0, 0, 0, 0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODE_DEFINE()
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
    UninitFfn(schedule_context);
}

TEST_F(TEST_FfnWorkerScheduler_UTest, INPUT_EXCUTE_MODE_INVALID)
{
    vector<DataType> data_types = {DT_INT8, DT_INT64, DT_INT8};
    vector<vector<int64_t>> shapes = {{1024}, {}, {1024}};
    aicpu::ScheduleContext schedule_context = {};
    schedule_context.common.session_num = 2;
    schedule_context.common.micro_batch_num = 2;
    schedule_context.common.micro_batch_size = 2;
    schedule_context.common.selected_expert_num = 5;
    schedule_context.common.schedule_mode = 1;
    schedule_context.control.run_flag = 1;
    schedule_context.ffn.polling_index = 1;
    int32_t execute_mode = 2; // test
    int32_t sync_group_size = 2;
    InitFfn(schedule_context);
    int8_t* input1 = reinterpret_cast<int8_t*>(&schedule_context);
    int64_t input2[1] = {1};
    int8_t output[1024] = {0, 0, 0, 0, 0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODE_DEFINE()
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
    UninitFfn(schedule_context);
}

TEST_F(TEST_FfnWorkerScheduler_UTest, INPUT_DIM_INVALID)
{
    vector<DataType> data_types = {DT_INT8, DT_INT64, DT_INT8};
    vector<vector<int64_t>> shapes = {{1024, 2}, {}, {1024}};
    aicpu::ScheduleContext schedule_context = {};
    schedule_context.common.session_num = 2;
    schedule_context.common.micro_batch_num = 2;
    schedule_context.common.micro_batch_size = 2;
    schedule_context.common.selected_expert_num = 5;
    schedule_context.common.schedule_mode = 0; // test
    schedule_context.control.run_flag = 1;
    schedule_context.ffn.polling_index = 1;
    int32_t execute_mode = 0;
    int32_t sync_group_size = 2;
    InitFfn(schedule_context);
    int8_t* input1 = reinterpret_cast<int8_t*>(&schedule_context);
    int64_t input2[1] = {1};
    int8_t output[1024] = {0, 0, 0, 0, 0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODE_DEFINE()
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
    UninitFfn(schedule_context);
}

TEST_F(TEST_FfnWorkerScheduler_UTest, INPUT_CONTROL_RUN_TERMINAL)
{
    vector<DataType> data_types = {DT_INT8, DT_INT64, DT_INT8};
    vector<vector<int64_t>> shapes = {{1024}, {}, {1024}};
    aicpu::ScheduleContext schedule_context = {};
    schedule_context.common.session_num = 2;
    schedule_context.common.micro_batch_num = 2;
    schedule_context.common.micro_batch_size = 2;
    schedule_context.common.selected_expert_num = 5;
    schedule_context.common.schedule_mode = 0;
    schedule_context.control.run_flag = 0; // test
    schedule_context.ffn.polling_index = 1;
    int32_t execute_mode = 0;
    int32_t sync_group_size = 2;
    InitFfn(schedule_context);
    int8_t* input1 = reinterpret_cast<int8_t*>(&schedule_context);
    int64_t input2[1] = {1};
    int8_t output[1024] = {0, 0, 0, 0, 0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODE_DEFINE()
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    UninitFfn(schedule_context);
}
