/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_attention_worker_combine.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
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
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> stride(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        stride[i] = shape[i + 1] * stride[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int CreateAclTensorNoData(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                          aclTensor** tensor)
{
    uint64_t elemSize = sizeof(int8_t);
    if (dataType == ACL_INT32) {
        elemSize = sizeof(int32_t);
    }
    if (dataType == ACL_FLOAT16) {
        elemSize = sizeof(int16_t);
    }
    if (dataType == ACL_BF16) {
        elemSize = sizeof(int16_t);
    }
    auto size = GetShapeSize(shape) * elemSize;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> stride(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        stride[i] = shape[i + 1] * stride[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
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
        uint32_t selected_expert_num; // topK + 1
        uint32_t expert_num;          // experts per layer
        uint32_t attn_to_ffn_token_size;
        uint32_t ffn_to_attn_token_size;
        int32_t schedule_mode; // 0: Ffn only 1: Attention only
        int8_t reserve0[96];
    };
    struct ControlArea {
        int32_t run_flag; // 0 : exited  1 : running
        int8_t reserve2[124];
    };
    struct AttentionArea {
        uint64_t token_info_buf; // Points to device memory.
        uint64_t token_info_buf_size;
        uint64_t token_data_buf; // Points to device memory.
        uint64_t token_data_buf_size;
        uint32_t micro_batch_id; // Records the latest ready micro batch id.
        int8_t reserve5[92];
    };
    struct FfnArea {
        uint64_t token_info_buf;
        uint64_t token_info_buf_size;
        uint64_t token_data_buf;
        uint64_t token_data_buf_size;
        uint64_t polling_index;
        int8_t reserve3[88];
        uint64_t layer_ids_buf;
        uint64_t layer_ids_buf_size;
        uint64_t session_ids_buf;
        uint64_t session_ids_buf_size;
        uint64_t micro_batch_ids_buf;
        uint64_t micro_batch_ids_buf_size;
        uint64_t expert_ids_buf;
        uint64_t expert_ids_buf_size;
        uint32_t out_num;
        int8_t reserve4[60];
    };
    CommonArea common;
    ControlArea control;
    AttentionArea attention;
    FfnArea ffn;
    int8_t reserve6[384]; // Padding to 1024 bytes.
};
static_assert(sizeof(ScheduleContext) == 1024, "ScheduleContext size must be 1024 bytes");
#pragma pack(pop)

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t BS = 48; // batch size
    int64_t K = 8;   // expert num per token
    int64_t hiddenSize = 20480;
    int64_t tokenDtype = 1; // BF16
    int64_t needSchedule = 0;

    // 初始化ScheduleContext
    ScheduleContext scheduleContext = {};
    scheduleContext.common.session_num = 1;
    scheduleContext.common.micro_batch_num = 1;
    scheduleContext.common.micro_batch_size = BS;
    scheduleContext.common.selected_expert_num = K;
    scheduleContext.common.expert_num = 16;
    scheduleContext.common.attn_to_ffn_token_size = 512;
    scheduleContext.common.ffn_to_attn_token_size = 512;
    scheduleContext.common.schedule_mode = 1; // Attention only
    scheduleContext.control.run_flag = 1;     // running
    scheduleContext.attention.micro_batch_id = 0;

    // 初始化Attention token_info_buf（flag置1表示数据就绪）
    size_t perDataDescSize = sizeof(AttentionDataDesc) + sizeof(int32_t) * BS * K;
    size_t tokenInfoBufSize = static_cast<size_t>(scheduleContext.common.micro_batch_num) * perDataDescSize;
    void* tokenInfoBuf = nullptr;
    ret = aclrtMalloc(&tokenInfoBuf, tokenInfoBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc token info buf failed. ERROR: %d\n", ret); return ret);
    scheduleContext.attention.token_info_buf = reinterpret_cast<uint64_t>(tokenInfoBuf);
    scheduleContext.attention.token_info_buf_size = tokenInfoBufSize;
    std::vector<int32_t> hostFlags(static_cast<size_t>(BS) * K, 1);
    ret = aclrtMemcpy(tokenInfoBuf, tokenInfoBufSize, hostFlags.data(), static_cast<size_t>(BS) * K * sizeof(int32_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy token info buf failed. ERROR: %d\n", ret); return ret);

    // 初始化Attention token_data_buf
    uint64_t tokenDataSize = static_cast<uint64_t>(BS) * K * hiddenSize * sizeof(int16_t);
    void* tokenDataBuf = nullptr;
    ret = aclrtMalloc(&tokenDataBuf, tokenDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc token data buf failed. ERROR: %d\n", ret); return ret);
    scheduleContext.attention.token_data_buf = reinterpret_cast<uint64_t>(tokenDataBuf);
    scheduleContext.attention.token_data_buf_size = tokenDataSize;
    std::vector<int16_t> hostTokenData(static_cast<size_t>(BS) * K * hiddenSize, 1);
    ret = aclrtMemcpy(tokenDataBuf, tokenDataSize, hostTokenData.data(), tokenDataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy token data buf failed. ERROR: %d\n", ret); return ret);

    // 创建scheduleContext aclTensor
    std::vector<int64_t> scheduleContextShape = {1024};
    void* scheduleContextDeviceAddr = nullptr;
    aclTensor* scheduleContextRef = nullptr;
    std::vector<int8_t> hostCtxData(1024, 0);
    std::memcpy(hostCtxData.data(), &scheduleContext, sizeof(ScheduleContext));
    ret = CreateAclTensor(hostCtxData, scheduleContextShape, &scheduleContextDeviceAddr, aclDataType::ACL_INT8,
                          &scheduleContextRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建expert_scales aclTensor, shape (BS, K)
    std::vector<int64_t> expertScalesShape = {BS, K};
    std::vector<float> hostExpertScales(static_cast<size_t>(BS) * K, 0.125f);
    void* expertScalesDeviceAddr = nullptr;
    aclTensor* expertScalesRef = nullptr;
    ret = CreateAclTensor(hostExpertScales, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT,
                          &expertScalesRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建layer_id aclTensor, shape (1)
    std::vector<int64_t> layerIdShape = {1};
    std::vector<int32_t> hostLayerId = {0};
    void* layerIdDeviceAddr = nullptr;
    aclTensor* layerIdRef = nullptr;
    ret = CreateAclTensor(hostLayerId, layerIdShape, &layerIdDeviceAddr, aclDataType::ACL_INT32, &layerIdRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出y aclTensor, shape (BS, hiddenSize)
    std::vector<int64_t> yShape = {BS, hiddenSize};
    void* yDeviceAddr = nullptr;
    aclTensor* yRef = nullptr;
    ret = CreateAclTensorNoData(yShape, &yDeviceAddr, aclDataType::ACL_BF16, &yRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出next_layer_id aclTensor, shape (1)
    std::vector<int64_t> nextLayerIdShape = {1};
    void* nextLayerIdDeviceAddr = nullptr;
    aclTensor* nextLayerIdRef = nullptr;
    ret = CreateAclTensorNoData(nextLayerIdShape, &nextLayerIdDeviceAddr, aclDataType::ACL_INT32, &nextLayerIdRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAttentionWorkerCombineGetWorkspaceSize(scheduleContextRef, expertScalesRef, layerIdRef, hiddenSize,
                                                      tokenDtype, needSchedule, yRef, nextLayerIdRef, &workspaceSize,
                                                      &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionWorkerCombineGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnAttentionWorkerCombine(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionWorkerCombine failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出结果，将device侧内存上的结果拷贝至host侧
    int32_t nextLayerId = 0;
    ret = aclrtMemcpy(&nextLayerId, sizeof(int32_t), nextLayerIdDeviceAddr, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy next_layer_id failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("next_layer_id = %d.\n", nextLayerId);

    // 6. 释放aclTensor
    aclDestroyTensor(scheduleContextRef);
    aclDestroyTensor(expertScalesRef);
    aclDestroyTensor(layerIdRef);
    aclDestroyTensor(yRef);
    aclDestroyTensor(nextLayerIdRef);

    // 7. 释放device资源
    aclrtFree(scheduleContextDeviceAddr);
    aclrtFree(expertScalesDeviceAddr);
    aclrtFree(layerIdDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(nextLayerIdDeviceAddr);
    aclrtFree(tokenInfoBuf);
    aclrtFree(tokenDataBuf);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    Finalize(deviceId, stream);
    return 0;
}
