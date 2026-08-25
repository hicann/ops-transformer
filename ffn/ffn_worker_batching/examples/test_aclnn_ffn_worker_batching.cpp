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
#include <memory>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_ffn_worker_batching.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
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

int CreateAclTensorNoData(const std::vector<int64_t> &shape, void **deviceAddr, aclDataType dataType,
                          aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(int8_t);
    if (dataType == ACL_INT32) {
        size = GetShapeSize(shape) * sizeof(int32_t);
    }
    if (dataType == ACL_INT64) {
        size = GetShapeSize(shape) * sizeof(int64_t);
    }
    if (dataType == ACL_FLOAT) {
        size = GetShapeSize(shape) * sizeof(float);
    }
    if (dataType == ACL_FLOAT16) {
        size = GetShapeSize(shape) * sizeof(int16_t);
    }
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
    struct FfnArea {
        uint64_t token_info_buf; // Points to device memory.
        uint64_t token_info_buf_size;
        uint64_t token_data_buf; // Points to device memory.
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
    struct AttentionArea {
        uint64_t token_info_buf;
        uint64_t token_info_buf_size;
        uint64_t token_data_buf;
        uint64_t token_data_buf_size;
        uint32_t micro_batch_id;
        int8_t reserve5[92];
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
    ScheduleContext scheduleContext = {};
    scheduleContext.common.session_num = 1;
    scheduleContext.common.micro_batch_num = 1;
    scheduleContext.common.micro_batch_size = 8;
    scheduleContext.common.selected_expert_num = 9; // topK + 1
    scheduleContext.common.expert_num = 8;
    scheduleContext.common.attn_to_ffn_token_size = 512;
    scheduleContext.common.ffn_to_attn_token_size = 512;
    scheduleContext.common.schedule_mode = 0; // Ffn only
    scheduleContext.control.run_flag = 1;     // running
    scheduleContext.ffn.polling_index = 0;

    // 属性参数
    int64_t expertNum = 8;                                  // 每层专家数 × layer_num
    int64_t A = scheduleContext.common.session_num;         // Attention worker数量
    int64_t BS = scheduleContext.common.micro_batch_size;   // micro batch size
    int64_t K = scheduleContext.common.selected_expert_num; // topK + 1
    int64_t H = 4096;                                       // hidden size
    int64_t Y = A * BS * K;
    std::vector<int64_t> maxOutShapeValue = {A, BS, K, H};
    int64_t tokenDtype = 0;   // FP16
    int64_t needSchedule = 0; // 仅做batching
    int64_t layerNum = 1;

    // 初始化Ffn token_info_buf
    uint64_t flagAndLayerIdSize = sizeof(FfnDataDesc);
    uint64_t tokenInfoSize = (sizeof(int32_t) * static_cast<uint64_t>(K) * BS + flagAndLayerIdSize) *
                             static_cast<uint64_t>(scheduleContext.common.micro_batch_num) *
                             static_cast<uint64_t>(scheduleContext.common.session_num);
    scheduleContext.ffn.token_info_buf_size = tokenInfoSize;
    void *tokenInfoBuf = nullptr;
    ret = aclrtMalloc(&tokenInfoBuf, tokenInfoSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc token info buf failed. ERROR: %d\n", ret); return ret);
    scheduleContext.ffn.token_info_buf = reinterpret_cast<uint64_t>(tokenInfoBuf);

    // 填充token_info_buf：flag=1，layer_id，expert_ids
    std::vector<int32_t> hostTokenInfo(tokenInfoSize / sizeof(int32_t), 0);
    int32_t *pInt = hostTokenInfo.data();
    for (uint32_t s = 0; s < scheduleContext.common.session_num; ++s) {
        for (uint32_t m = 0; m < scheduleContext.common.micro_batch_num; ++m) {
            *pInt++ = 1; // flag
            *pInt++ = 0; // layer_id
            for (uint32_t i = 0; i < BS * K; ++i) {
                *pInt++ = static_cast<int32_t>(i % scheduleContext.common.expert_num); // expert_id
            }
        }
    }
    ret = aclrtMemcpy(tokenInfoBuf, tokenInfoSize, hostTokenInfo.data(), tokenInfoSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy token info buf failed. ERROR: %d\n", ret); return ret);

    // 初始化Ffn token_data_buf
    uint64_t tokenDataSize = static_cast<uint64_t>(Y) * H * sizeof(int16_t);
    scheduleContext.ffn.token_data_buf_size = tokenDataSize;
    void *tokenDataBuf = nullptr;
    ret = aclrtMalloc(&tokenDataBuf, tokenDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc token data buf failed. ERROR: %d\n", ret); return ret);
    scheduleContext.ffn.token_data_buf = reinterpret_cast<uint64_t>(tokenDataBuf);
    std::vector<int16_t> hostTokenData(Y * H, 1);
    ret = aclrtMemcpy(tokenDataBuf, tokenDataSize, hostTokenData.data(), tokenDataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy token data buf failed. ERROR: %d\n", ret); return ret);

    // 创建scheduleContext aclTensor
    std::vector<int64_t> scheduleContextShape = {1024};
    void *scheduleContextDeviceAddr = nullptr;
    aclTensor *scheduleContextRef = nullptr;
    // ==== FfnArea 中的扁平 id 缓冲 ====
    // 这四个字段是二级指针:context 里存的是 device 地址,由 kernel 自行解引用。框架看不到内层缓冲,
    // 不填也能通过 tiling 校验,只在 kernel 访存时暴露为 errcode:(95) MTE 访问 DDR 越界。
    // NORM 路径(needSchedule=0)两代实现都会读它们,必须提供。
    std::vector<int32_t> hostExpertIds(Y);
    for (int64_t i = 0; i < Y; ++i) {
        hostExpertIds[i] = static_cast<int32_t>(i % expertNum);
    }
    void *expertIdsBuf = nullptr;
    ret = aclrtMalloc(&expertIdsBuf, hostExpertIds.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc expert ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(expertIdsBuf, hostExpertIds.size() * sizeof(int32_t), hostExpertIds.data(),
                      hostExpertIds.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy expert ids buf failed. ERROR: %d\n", ret); return ret);
    scheduleContext.ffn.expert_ids_buf = reinterpret_cast<uint64_t>(expertIdsBuf);
    scheduleContext.ffn.expert_ids_buf_size = hostExpertIds.size() * sizeof(int32_t);

    std::vector<int32_t> hostSessionIds(A, 0);
    std::vector<int32_t> hostMicroBatchIds(A, 0);
    std::vector<int32_t> hostLayerIds(A, 0);
    const uint64_t idsBufSize = static_cast<uint64_t>(A) * sizeof(int32_t);
    void *sessionIdsBuf = nullptr;
    void *microBatchIdsBuf = nullptr;
    void *layerIdsBuf = nullptr;
    ret = aclrtMalloc(&sessionIdsBuf, idsBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc session ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&microBatchIdsBuf, idsBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc micro batch ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&layerIdsBuf, idsBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("malloc layer ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(sessionIdsBuf, idsBufSize, hostSessionIds.data(), idsBufSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy session ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(microBatchIdsBuf, idsBufSize, hostMicroBatchIds.data(), idsBufSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy micro batch ids buf failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(layerIdsBuf, idsBufSize, hostLayerIds.data(), idsBufSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("cpy layer ids buf failed. ERROR: %d\n", ret); return ret);
    scheduleContext.ffn.session_ids_buf = reinterpret_cast<uint64_t>(sessionIdsBuf);
    scheduleContext.ffn.session_ids_buf_size = idsBufSize;
    scheduleContext.ffn.micro_batch_ids_buf = reinterpret_cast<uint64_t>(microBatchIdsBuf);
    scheduleContext.ffn.micro_batch_ids_buf_size = idsBufSize;
    scheduleContext.ffn.layer_ids_buf = reinterpret_cast<uint64_t>(layerIdsBuf);
    scheduleContext.ffn.layer_ids_buf_size = idsBufSize;

    std::vector<int8_t> hostCtxData(1024, 0);
    std::memcpy(hostCtxData.data(), &scheduleContext, sizeof(ScheduleContext));
    ret = CreateAclTensor(hostCtxData, scheduleContextShape, &scheduleContextDeviceAddr, aclDataType::ACL_INT8,
                          &scheduleContextRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出 aclTensor
    std::vector<int64_t> yShape = {Y, H};
    void *yDeviceAddr = nullptr;
    aclTensor *yRef = nullptr;
    ret = CreateAclTensorNoData(yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &yRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> groupListShape = {expertNum, 2};
    void *groupListDeviceAddr = nullptr;
    aclTensor *groupListRef = nullptr;
    ret = CreateAclTensorNoData(groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupListRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> oneDimShape = {Y};
    void *sessionIdsAddr = nullptr;
    aclTensor *sessionIdsRef = nullptr;
    ret = CreateAclTensorNoData(oneDimShape, &sessionIdsAddr, aclDataType::ACL_INT32, &sessionIdsRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *microBatchIdsAddr = nullptr;
    aclTensor *microBatchIdsRef = nullptr;
    ret = CreateAclTensorNoData(oneDimShape, &microBatchIdsAddr, aclDataType::ACL_INT32, &microBatchIdsRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *tokenIdsAddr = nullptr;
    aclTensor *tokenIdsRef = nullptr;
    ret = CreateAclTensorNoData(oneDimShape, &tokenIdsAddr, aclDataType::ACL_INT32, &tokenIdsRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *expertOffsetsAddr = nullptr;
    aclTensor *expertOffsetsRef = nullptr;
    ret = CreateAclTensorNoData(oneDimShape, &expertOffsetsAddr, aclDataType::ACL_INT32, &expertOffsetsRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void *dynamicScaleAddr = nullptr;
    aclTensor *dynamicScaleRef = nullptr;
    ret = CreateAclTensorNoData(oneDimShape, &dynamicScaleAddr, aclDataType::ACL_FLOAT, &dynamicScaleRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> actualTokenNumShape = {1};
    void *actualTokenNumAddr = nullptr;
    aclTensor *actualTokenNumRef = nullptr;
    ret = CreateAclTensorNoData(actualTokenNumShape, &actualTokenNumAddr, aclDataType::ACL_INT64, &actualTokenNumRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建maxOutShape aclIntArray
    aclIntArray *maxOutShapeArray = aclCreateIntArray(maxOutShapeValue.data(), maxOutShapeValue.size());

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnFfnWorkerBatchingGetWorkspaceSize(scheduleContextRef, expertNum, maxOutShapeArray, tokenDtype,
                                                 needSchedule, layerNum, yRef, groupListRef, sessionIdsRef,
                                                 microBatchIdsRef, tokenIdsRef, expertOffsetsRef, dynamicScaleRef,
                                                 actualTokenNumRef, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFfnWorkerBatchingGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnFfnWorkerBatching(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFfnWorkerBatching failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出结果，将device侧内存上的结果拷贝至host侧
    int64_t actualTokenNum = 0;
    ret = aclrtMemcpy(&actualTokenNum, sizeof(int64_t), actualTokenNumAddr, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy actual_token_num failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("actual_token_num = %ld.\n", actualTokenNum);

    std::vector<int64_t> groupListHost(expertNum * 2, 0);
    ret = aclrtMemcpy(groupListHost.data(), expertNum * 2 * sizeof(int64_t), groupListDeviceAddr,
                      expertNum * 2 * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy group_list failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expertNum; i++) {
        LOG_PRINT("group_list[%ld] = [expert_id=%ld, token_num=%ld]\n", i, groupListHost[i * 2],
                  groupListHost[i * 2 + 1]);
    }

    // 6. 释放aclTensor与aclIntArray
    aclDestroyTensor(scheduleContextRef);
    aclDestroyTensor(yRef);
    aclDestroyTensor(groupListRef);
    aclDestroyTensor(sessionIdsRef);
    aclDestroyTensor(microBatchIdsRef);
    aclDestroyTensor(tokenIdsRef);
    aclDestroyTensor(expertOffsetsRef);
    aclDestroyTensor(dynamicScaleRef);
    aclDestroyTensor(actualTokenNumRef);
    aclDestroyIntArray(maxOutShapeArray);

    // 7. 释放device资源
    aclrtFree(scheduleContextDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(groupListDeviceAddr);
    aclrtFree(sessionIdsAddr);
    aclrtFree(microBatchIdsAddr);
    aclrtFree(tokenIdsAddr);
    aclrtFree(expertOffsetsAddr);
    aclrtFree(dynamicScaleAddr);
    aclrtFree(actualTokenNumAddr);
    aclrtFree(tokenInfoBuf);
    aclrtFree(tokenDataBuf);
    aclrtFree(expertIdsBuf);
    aclrtFree(sessionIdsBuf);
    aclrtFree(microBatchIdsBuf);
    aclrtFree(layerIdsBuf);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    Finalize(deviceId, stream);
    return 0;
}
