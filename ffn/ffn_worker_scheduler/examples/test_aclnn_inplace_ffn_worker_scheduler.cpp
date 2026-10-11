/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <limits>
#include "acl/acl.h"
#include "aclnnop/aclnn_ffn_worker_scheduler.h"

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

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
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

constexpr uint32_t kSuccess = 0;
constexpr uint32_t kFailure = 1;
constexpr uint64_t kBufAlignSize = 512;

inline uint64_t AlignUp(uint64_t num, uint64_t align)
{
    return ((num + align - 1) / align) * align;
}

#pragma pack(push, 1)
struct FfnDataDesc {
    volatile int32_t flag;
    volatile int32_t layer_id;
    volatile int32_t expert_ids[0];
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
        uint64_t polling_index; // For synchronous computation only: records the micro-batch ids to be processed
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

uint64_t CalcFfnTokenInfoSize(ScheduleContext& schedule_context)
{
    // token_info_size = (sizeof(FfnDataDesc) + selected_expert_num * micro_batch_size) * micro_batch_num * session_num
    uint64_t flag_and_layer_id_size = sizeof(FfnDataDesc);
    uint64_t token_info_size = (sizeof(int32_t) * static_cast<uint64_t>(schedule_context.common.selected_expert_num) *
                                    schedule_context.common.micro_batch_size +
                                flag_and_layer_id_size) *
                               static_cast<uint64_t>(schedule_context.common.micro_batch_num) *
                               static_cast<uint64_t>(schedule_context.common.session_num);
    return token_info_size;
}

uint32_t InitFfnTokenInfoBuf(ScheduleContext& schedule_context)
{
    std::unique_ptr<uint8_t[]> tmp_buf(new (std::nothrow) uint8_t[schedule_context.ffn.token_info_buf_size]);
    if (tmp_buf == nullptr) {
        LOG_PRINT("alloc token info host tmp buf failed, buf_size=   %lu\n", schedule_context.ffn.token_info_buf_size);
        return kFailure;
    }
    auto buf_int = reinterpret_cast<int32_t*>(tmp_buf.get());
    auto token_buf_int = buf_int;
    for (uint32_t session_id = 0; session_id < schedule_context.common.session_num; ++session_id) {
        for (uint32_t micro_batch_id = 0; micro_batch_id < schedule_context.common.micro_batch_num; ++micro_batch_id) {
            // flag
            *buf_int++ = 1;
            // layer_id
            *buf_int++ = 55;
            for (uint32_t idx = 0;
                 idx < schedule_context.common.micro_batch_size * schedule_context.common.selected_expert_num; ++idx) {
                // expert_id
                *buf_int++ = idx;
            }
        }
    }
    void* token_info_buf = nullptr;
    size_t token_info_buf_size = sizeof(uint8_t) * schedule_context.ffn.token_info_buf_size;
    if (token_info_buf_size > 0) {
        auto ret = aclrtMalloc(&token_info_buf, token_info_buf_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate token info buf failed. ERROR: %d\n", ret); return ret;);
    }
    schedule_context.ffn.token_info_buf = reinterpret_cast<uint64_t>(token_info_buf);
    auto token_info_buf_tmp = reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.token_info_buf));
    auto ret = aclrtMemcpy(token_info_buf_tmp, token_info_buf_size, token_buf_int, token_info_buf_size,
                           ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("memory copy token info buf failed, size_=  %lu\n", token_info_buf_size);
              return ret;);
    return ACL_SUCCESS;
}

uint32_t InitFfn(ScheduleContext& schedule_context)
{
    uint64_t token_info_size = CalcFfnTokenInfoSize(schedule_context);
    if (token_info_size == 0U) {
        return ACL_ERROR_INVALID_PARAM;
    }

    uint64_t token_info_aligned_size = AlignUp(token_info_size, kBufAlignSize);
    if (token_info_aligned_size < token_info_size) {
        LOG_PRINT("token_info_size[%lu]  overflow after align with %lu\n", token_info_size, kBufAlignSize);
        return ACL_ERROR_INVALID_PARAM;
    }

    schedule_context.ffn.token_info_buf_size = token_info_size;

    // 申请1024大小作为token data buf
    schedule_context.ffn.token_data_buf_size = 1024;
    void* token_data_buf = nullptr;
    size_t token_data_buf_size = sizeof(uint8_t) * schedule_context.ffn.token_data_buf_size;
    auto ret = aclrtMalloc(&token_data_buf, token_data_buf_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate token data buf failed. ERROR: %d\n", ret); return ret;);
    schedule_context.ffn.token_data_buf = reinterpret_cast<uint64_t>(token_data_buf);

    ret = InitFfnTokenInfoBuf(schedule_context);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    // calc output size
    schedule_context.ffn.layer_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num;
    schedule_context.ffn.session_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num;
    schedule_context.ffn.micro_batch_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num;
    schedule_context.ffn.expert_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num *
                                               schedule_context.common.micro_batch_size *
                                               schedule_context.common.selected_expert_num;

    void* layer_ids_buf = nullptr;
    size_t layer_ids_buf_size = sizeof(uint8_t) * schedule_context.ffn.layer_ids_buf_size;
    ret = aclrtMalloc(&layer_ids_buf, layer_ids_buf_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate layer ids buf failed. ERROR: %d\n", ret); return ret;);
    schedule_context.ffn.layer_ids_buf = reinterpret_cast<uint64_t>(layer_ids_buf);

    void* session_ids_buf = nullptr;
    size_t session_ids_buf_size = sizeof(uint8_t) * schedule_context.ffn.session_ids_buf_size;
    ret = aclrtMalloc(&session_ids_buf, session_ids_buf_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate session ids buf failed. ERROR: %d\n", ret); return ret;);
    schedule_context.ffn.session_ids_buf = reinterpret_cast<uint64_t>(session_ids_buf);

    void* micro_batch_ids_buf = nullptr;
    size_t micro_batch_ids_buf_size = sizeof(uint8_t) * schedule_context.ffn.micro_batch_ids_buf_size;
    ret = aclrtMalloc(&micro_batch_ids_buf, micro_batch_ids_buf_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate micro batch ids buf failed. ERROR: %d\n", ret); return ret;);
    schedule_context.ffn.micro_batch_ids_buf = reinterpret_cast<uint64_t>(micro_batch_ids_buf);

    void* expert_ids_buf = nullptr;
    size_t expert_ids_buf_size = sizeof(uint8_t) * schedule_context.ffn.expert_ids_buf_size;
    ret = aclrtMalloc(&expert_ids_buf, expert_ids_buf_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate expert ids buf failed. ERROR: %d\n", ret); return ret;);
    schedule_context.ffn.expert_ids_buf = reinterpret_cast<uint64_t>(expert_ids_buf);

    LOG_PRINT("Init ffn success, token_info_buf_size=%lu,token_data_buf_size= %lu.\n",
              schedule_context.ffn.token_info_buf_size, schedule_context.ffn.token_data_buf_size);
    return ACL_SUCCESS;
}

uint32_t UninitFfn(ScheduleContext& schedule_context)
{
    if (schedule_context.ffn.token_info_buf != 0) {
        aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.token_info_buf)));
        schedule_context.ffn.token_info_buf = 0;
    }
    if (schedule_context.ffn.token_data_buf != 0) {
        aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.token_data_buf)));
        schedule_context.ffn.token_data_buf = 0;
    }
    if (schedule_context.ffn.layer_ids_buf != 0) {
        aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.layer_ids_buf)));
        schedule_context.ffn.layer_ids_buf = 0;
    }
    if (schedule_context.ffn.session_ids_buf != 0) {
        aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.session_ids_buf)));
        schedule_context.ffn.session_ids_buf = 0;
    }
    if (schedule_context.ffn.micro_batch_ids_buf != 0) {
        aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.micro_batch_ids_buf)));
        schedule_context.ffn.micro_batch_ids_buf = 0;
    }
    if (schedule_context.ffn.expert_ids_buf != 0) {
        aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.expert_ids_buf)));
        schedule_context.ffn.expert_ids_buf = 0;
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化, 参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {1024};
    void* selfDeviceAddr = nullptr;
    aclTensor* scheduleContextRef = nullptr;
    std::vector<int8_t> selfHostData(1024);
    ScheduleContext schedule_context = {};
    schedule_context.common.session_num = 2;
    schedule_context.common.micro_batch_num = 2;
    schedule_context.common.micro_batch_size = 2;
    schedule_context.common.selected_expert_num = 5;
    schedule_context.control.run_flag = 1;
    schedule_context.common.schedule_mode = 0;
    schedule_context.ffn.polling_index = 1;
    InitFfn(schedule_context);
    ret = aclrtMemcpy(selfHostData.data(), sizeof(ScheduleContext), &schedule_context, sizeof(ScheduleContext),
                      ACL_MEMCPY_HOST_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("copy schedule context to host failed. ERROR: %d\n", ret);
        UninitFfn(schedule_context);
        return ret;
    }

    int32_t syncGroupSize = 1;
    int32_t executeMode = 0;
    // 创建scheduleContext aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT8, &scheduleContextRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnInplaceFfnWorkerScheduler第一段接口
    ret = aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize(scheduleContextRef, syncGroupSize, executeMode, &workspaceSize,
                                                         &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnInplaceFfnWorkerScheduler第二段接口
    ret = aclnnInplaceFfnWorkerScheduler(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceFfnWorkerScheduler failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(selfShape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int8_t), selfDeviceAddr, size * sizeof(int8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 打印输出结果
    ScheduleContext* out_schedule_context = reinterpret_cast<ScheduleContext*>(resultData.data());
    LOG_PRINT("layer_ids_buf_size = %lu.\n", out_schedule_context->ffn.layer_ids_buf_size);
    LOG_PRINT("session_ids_buf_size = %lu.\n", out_schedule_context->ffn.session_ids_buf_size);
    LOG_PRINT("micro_batch_ids_buf_size = %lu.\n", out_schedule_context->ffn.micro_batch_ids_buf_size);
    LOG_PRINT("expert_ids_buf_size = %lu.\n", out_schedule_context->ffn.expert_ids_buf_size);

    // 打印 layer_ids 信息
    std::vector<int32_t> layer_ids_buf(out_schedule_context->ffn.layer_ids_buf_size / sizeof(int32_t), 0);
    ret = aclrtMemcpy(layer_ids_buf.data(), out_schedule_context->ffn.layer_ids_buf_size,
                      reinterpret_cast<void*>(static_cast<uintptr_t>(out_schedule_context->ffn.layer_ids_buf)),
                      out_schedule_context->ffn.layer_ids_buf_size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy layer ids buf from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int i = 0; i < out_schedule_context->ffn.layer_ids_buf_size / sizeof(int32_t); i++) {
        LOG_PRINT("layer_ids[%d] is: %d\n", i, layer_ids_buf[i]);
    }

    // 打印 session_ids 信息
    std::vector<int32_t> session_ids_buf(out_schedule_context->ffn.session_ids_buf_size / sizeof(int32_t), 0);
    ret = aclrtMemcpy(session_ids_buf.data(), out_schedule_context->ffn.session_ids_buf_size,
                      reinterpret_cast<void*>(static_cast<uintptr_t>(out_schedule_context->ffn.session_ids_buf)),
                      out_schedule_context->ffn.session_ids_buf_size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy session ids buf from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int i = 0; i < out_schedule_context->ffn.session_ids_buf_size / sizeof(int32_t); i++) {
        LOG_PRINT("session_ids[%d] is: %d\n", i, session_ids_buf[i]);
    }

    // 打印 micro_batch_ids 信息
    std::vector<int32_t> micro_batch_ids_buf(out_schedule_context->ffn.micro_batch_ids_buf_size / sizeof(int32_t), 0);
    ret = aclrtMemcpy(micro_batch_ids_buf.data(), out_schedule_context->ffn.micro_batch_ids_buf_size,
                      reinterpret_cast<void*>(static_cast<uintptr_t>(out_schedule_context->ffn.micro_batch_ids_buf)),
                      out_schedule_context->ffn.micro_batch_ids_buf_size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy micro batch ids buf from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int i = 0; i < out_schedule_context->ffn.micro_batch_ids_buf_size / sizeof(int32_t); i++) {
        LOG_PRINT("micro_batch_ids[%d] is: %d\n", i, micro_batch_ids_buf[i]);
    }

    // 打印 expert_ids 信息
    std::vector<int32_t> expert_ids_buf(out_schedule_context->ffn.expert_ids_buf_size / sizeof(int32_t), 0);
    ret = aclrtMemcpy(expert_ids_buf.data(), out_schedule_context->ffn.expert_ids_buf_size,
                      reinterpret_cast<void*>(static_cast<uintptr_t>(out_schedule_context->ffn.expert_ids_buf)),
                      out_schedule_context->ffn.expert_ids_buf_size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy expert ids buf from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int i = 0; i < out_schedule_context->ffn.expert_ids_buf_size / sizeof(int32_t); i++) {
        LOG_PRINT("expert_ids[%d] is: %d\n", i, expert_ids_buf[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(scheduleContextRef);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    UninitFfn(schedule_context);
    aclrtFree(selfDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
