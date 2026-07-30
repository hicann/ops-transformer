/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_stem_oam_prep_varlen_q.cpp
 * \brief aclnn test demo for StemOamPrepVarlenQ operator
 */

#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_stem_oam_prep_varlen_q.h"

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

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0; // 根据实际设备ID填写
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 参数设置
    int64_t total_tokens = 256;
    int64_t H_q = 32;
    int64_t D = 128;
    int64_t batch = 2;
    int64_t stemBlockSize = 128;
    int64_t stemStride = 16;
    int64_t max_Qb = 1; // ceil(max(qSeqLens)/stemBlockSize)
    int64_t kflat_dim = stemStride * D;

    // qSeqLens 和 cuSeqLensQ
    int64_t qSeqLens_data[] = {128, 128};
    int64_t cuSeqLensQ_data[] = {0, 128, 256};

    // 创建输入输出形状
    std::vector<int64_t> qShape = {total_tokens, H_q, D};
    std::vector<int64_t> qScaleShape = {total_tokens, H_q};
    std::vector<int64_t> qFlatShape = {batch, H_q, max_Qb, kflat_dim};

    // 设备地址
    void *qDeviceAddr = nullptr;
    void *qScaleDeviceAddr = nullptr;
    void *qFlatDeviceAddr = nullptr;

    // aclTensor
    aclTensor *q = nullptr;
    aclTensor *qScale = nullptr;
    aclTensor *qFlat = nullptr;
    aclIntArray *qSeqLens = nullptr;
    aclIntArray *cuSeqLensQ = nullptr;

    // 创建 q tensor (FP8_E4M3FN)
    std::vector<uint8_t> hostQ(GetShapeSize(qShape), 1);
    ret = CreateAclTensor(hostQ, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 qScale tensor (FLOAT)
    std::vector<float> hostQScale(GetShapeSize(qScaleShape), 1.0f);
    ret = CreateAclTensor(hostQScale, qScaleShape, &qScaleDeviceAddr, aclDataType::ACL_FLOAT, &qScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 qFlat output tensor (BFLOAT16)
    std::vector<uint16_t> hostQFlat(GetShapeSize(qFlatShape), 0);
    ret = CreateAclTensor(hostQFlat, qFlatShape, &qFlatDeviceAddr, aclDataType::ACL_BF16, &qFlat);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 qSeqLens 和 cuSeqLensQ
    qSeqLens = aclCreateIntArray(qSeqLens_data, batch);
    CHECK_RET(qSeqLens != nullptr, LOG_PRINT("aclCreateIntArray qSeqLens failed\n"); return -1);

    cuSeqLensQ = aclCreateIntArray(cuSeqLensQ_data, batch + 1);
    CHECK_RET(cuSeqLensQ != nullptr, LOG_PRINT("aclCreateIntArray cuSeqLensQ failed\n"); return -1);

    // 1. 获取 workspace 大小
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    ret = aclnnStemOamPrepVarlenQGetWorkspaceSize(q, qSeqLens, cuSeqLensQ, qScale, stemBlockSize, stemStride, qFlat,
                                                  &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStemOamPrepVarlenQGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 2. 申请 workspace
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 3. 执行计算
    ret = aclnnStemOamPrepVarlenQ(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStemOamPrepVarlenQ failed. ERROR: %d\n", ret); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出结果
    auto qFlatSize = GetShapeSize(qFlatShape);
    std::vector<uint16_t> qFlatData(qFlatSize, 0);
    ret = aclrtMemcpy(qFlatData.data(), qFlatData.size() * sizeof(qFlatData[0]), qFlatDeviceAddr,
                      qFlatSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 打印前10个结果 (BF16 -> float 转换打印)
    for (int64_t i = 0; i < 10 && i < qFlatSize; i++) {
        uint16_t bf16Val = qFlatData[i];
        uint32_t floatBits = static_cast<uint32_t>(bf16Val) << 16;
        float floatVal;
        std::memcpy(&floatVal, &floatBits, sizeof(float));
        LOG_PRINT("qFlat[%ld] is: %f\n", i, floatVal);
    }

    // 6. 释放资源
    aclDestroyTensor(q);
    aclDestroyTensor(qScale);
    aclDestroyTensor(qFlat);
    aclDestroyIntArray(qSeqLens);
    aclDestroyIntArray(cuSeqLensQ);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtFree(qDeviceAddr);
    aclrtFree(qScaleDeviceAddr);
    aclrtFree(qFlatDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
