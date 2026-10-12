/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_aclnn_inplace_attention_worker_scheduler.cpp
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "acl/acl.h"
#include "aclnnop/aclnn_attention_worker_scheduler.h"
#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            Finalize(deviceId, stream); \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

int CreateAclTensor(const void* hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                    aclTensor** tensor)
{
    auto size = GetShapeSize(shape);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的stride
    std::vector<int64_t> stride(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        stride[i] = shape[i + 1] * stride[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

#pragma pack(push, 1)
struct AttentionDataDesc {
    int32_t flag[0];
};

struct ScheduleContext {
    struct CommonArea {
        uint32_t session_num; // Number of attention nodes
        uint32_t micro_batch_num;
        uint32_t micro_batch_size;
        uint32_t selected_expert_num;
        uint32_t expert_num; // Number of experts per layer, including routing experts and shared experts.
        uint32_t
            attn_to_ffn_token_size; // Each token in the Ffn window data area has a space size aligned to 512 bytes.
        uint32_t ffn_to_attn_token_size; // Each token in the Attention window data area has a space size aligned to 512
                                         // bytes.
        int32_t schedule_mode;           // 0: Ffn only 1: Attention only
        int8_t reserve0[96];
    };
    struct ControlArea {
        int32_t run_flag; // 0 : exited  1 : running
        int8_t reserve2[124];
    };
    struct FfnArea {
        // ffn area
        uint64_t token_info_buf; // Points to device memory.
        uint64_t token_info_buf_size;
        uint64_t token_data_buf; // Points to device memory.
        uint64_t token_data_buf_size;
        uint64_t polling_index; // For synchronous computation only:records the micro-batch ids to be processed
                                // internally by the ffn worker scheduler.
        int8_t reserve3[88];

        // ffn out area
        uint64_t layer_ids_buf; // Points to a device memory region that stores the organized layer ids, with an array
                                // size of session_num.
        uint64_t layer_ids_buf_size; // Total size: session_num * sizeof(int32_t)
        uint64_t session_ids_buf; //  Points to a device memory region that stores the organized session ids. The array
                                  //  size is session_num.
        uint64_t session_ids_buf_size; // Total size: session_num * sizeof(int32_t)
        uint64_t micro_batch_ids_buf; // Points to a device memory region that stores the organized micro batch ids. The
                                      // array size is session_num.
        uint64_t micro_batch_ids_buf_size; // Total size: session_num * sizeof(int32_t)
        uint64_t expert_ids_buf; //  Points to a device memory region that stores the organized expert ids. The tensor
                                 //  dimensions are [session_num, batch_size, selected_expert_num].
        uint64_t
            expert_ids_buf_size; // Total memory size: session_num * batch_size * selected_expert_num * sizeof(int32_t)
        uint32_t out_num;        // Indicates the number of sessions that have been processed.
        int8_t reserve4[60];
    };

    struct AttentionArea {
        // attention area
        uint64_t token_info_buf; // Points to device memory.
        uint64_t token_info_buf_size;
        uint64_t token_data_buf; // Points to device memory.
        uint64_t token_data_buf_size;
        uint32_t micro_batch_id; // Records the latest ready micro batch id.
        int8_t reserve5[92];
    };

    // common area
    CommonArea common;
    ControlArea control;
    AttentionArea attention;
    FfnArea ffn;
    // reserve area
    int8_t reserve6[384]; // Padding to 1024 bytes.
};

static_assert(sizeof(ScheduleContext) == 1024, "ScheduleContext size must be 1024 bytes");

#pragma pack(pop)

int aclnnAttentionWorkerSchedulerTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    ScheduleContext hostScheduleContext = {};

    // Initialize  CommonArea
    hostScheduleContext.common.session_num = 1;
    hostScheduleContext.common.micro_batch_num = 2;
    hostScheduleContext.common.micro_batch_size = 48;
    hostScheduleContext.common.selected_expert_num = 9;
    hostScheduleContext.common.expert_num = 16;
    hostScheduleContext.common.attn_to_ffn_token_size = 512;
    hostScheduleContext.common.ffn_to_attn_token_size = 512;
    hostScheduleContext.common.schedule_mode = 1; // attention

    // Initialize ControlArea
    hostScheduleContext.control.run_flag = 1; // running

    // Initialize AttentionArea
    hostScheduleContext.attention.micro_batch_id = 1;
    size_t per_data_desc_size = sizeof(AttentionDataDesc) + sizeof(int32_t) *
                                                                hostScheduleContext.common.micro_batch_size *
                                                                hostScheduleContext.common.selected_expert_num;
    size_t expect_token_info_buf_size =
        static_cast<size_t>(hostScheduleContext.common.micro_batch_num) * per_data_desc_size;
    void* tokenBufDeviceAddr = nullptr;
    ret = aclrtMalloc(&tokenBufDeviceAddr, expect_token_info_buf_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    hostScheduleContext.attention.token_info_buf = reinterpret_cast<uint64_t>(tokenBufDeviceAddr);
    hostScheduleContext.attention.token_info_buf_size = expect_token_info_buf_size;
    int target_micro_batch_id = 0;
    auto data_desc_ptr =
        reinterpret_cast<AttentionDataDesc*>(reinterpret_cast<uint8_t*>(hostScheduleContext.attention.token_info_buf) +
                                             per_data_desc_size * target_micro_batch_id);
    size_t flag_num = static_cast<size_t>(hostScheduleContext.common.micro_batch_size) *
                      hostScheduleContext.common.selected_expert_num;

    // Set all flags to 1.
    std::vector<int32_t> host_flags(flag_num, 1);
    ret = aclrtMemcpy(data_desc_ptr->flag, flag_num * sizeof(int32_t), host_flags.data(),
                      host_flags.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy flags from host to device failed. ERROR: %d\n", ret); return ret);

    uint64_t token_data_buf_size = 100;
    void* tokenDataDeviceAddr = nullptr;
    ret = aclrtMalloc(&tokenDataDeviceAddr, token_data_buf_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    hostScheduleContext.attention.token_data_buf = reinterpret_cast<uint64_t>(tokenDataDeviceAddr);

    // 创建scheduleContext aclTensor
    std::vector<int64_t> scheduleContextShape = {1024};
    void* scheduleContextDeviceAddr = nullptr;
    aclTensor* scheduleContextRef = nullptr;
    ret = CreateAclTensor(&hostScheduleContext, scheduleContextShape, &scheduleContextDeviceAddr, aclDataType::ACL_INT8,
                          &scheduleContextRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnInplaceAttentionWorkerScheduler第一段接口
    ret = aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize(scheduleContextRef, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }

    // 调用aclnnInplaceAttentionWorkerScheduler第二段接口
    ret = aclnnInplaceAttentionWorkerScheduler(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAttentionWorkerScheduler failed. ERROR: %d\n", ret);
              return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(scheduleContextShape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), scheduleContextDeviceAddr,
                      size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    // 打印输出结果
    ScheduleContext* out_schedule_context = reinterpret_cast<ScheduleContext*>(resultData.data());
    LOG_PRINT("micro_batch_id = %u.\n", out_schedule_context->attention.micro_batch_id);

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(scheduleContextRef);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(scheduleContextDeviceAddr);
    aclrtFree(tokenBufDeviceAddr);
    aclrtFree(tokenDataDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnAttentionWorkerSchedulerTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionWorkerSchedulerTest failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return 0;
}
