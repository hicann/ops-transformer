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
 * \file test_aclnn_compressor_grad.cpp
 * \brief CompressorGrad 算子 aclnn 调用示例（A5 / ascend950）
 *        场景：C4A（D=512, coff=2, cmp_ratio=4, BSH layout, BF16）
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_compressor_grad.h"
#include "opdev/bfloat16.h"

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

namespace {

using op::bfloat16;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    if (size > 0) {
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    } else {
        *deviceAddr = nullptr;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void PrintBf16Result(const std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<bfloat16> resultData(size, bfloat16(0.0f));
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size && i < 10; i++) { // 10: max print
        LOG_PRINT("result[%ld] is: %f\n", i, static_cast<float>(resultData[i]));
    }
}

void PrintF32Result(const std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float_t> resultData(size, 0.0f);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size && i < 10; i++) { // 10: max print
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

} // namespace

int main()
{
    // 1. device/stream 初始化
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 场景参数: C4A (D=512, coff=2, cmp_ratio=4, BSH layout, BF16)
    int64_t B = 1;
    int64_t S = 4;
    int64_t hiddenSize = 4096;
    int64_t headDim = 512;
    int64_t coff = 2;
    int64_t cmpRatio = 4;

    int64_t coffD = coff * headDim;
    int64_t Sr = (S + cmpRatio - 1) / cmpRatio;

    // 2. 构造输入与输出 shape
    std::vector<int64_t> xShape = {B, S, hiddenSize};
    std::vector<int64_t> wkvShape = {coffD, hiddenSize};
    std::vector<int64_t> wgateShape = {coffD, hiddenSize};
    std::vector<int64_t> dCmpKvShape = {B, Sr, headDim};
    std::vector<int64_t> softmaxScoreShape = {B, Sr, coff * cmpRatio, headDim};
    std::vector<int64_t> kvShape = {B, Sr, coff * cmpRatio, headDim};
    std::vector<int64_t> startPosShape = {B};
    std::vector<int64_t> dXShape = {B, S, hiddenSize};
    std::vector<int64_t> dWkvShape = {coffD, hiddenSize};
    std::vector<int64_t> dWgateShape = {coffD, hiddenSize};
    std::vector<int64_t> dApeShape = {cmpRatio, coffD};

    // 3. 构造 host 数据
    int64_t xSize = GetShapeSize(xShape);
    int64_t wkvSize = GetShapeSize(wkvShape);
    int64_t wgateSize = GetShapeSize(wgateShape);
    int64_t dCmpKvSize = GetShapeSize(dCmpKvShape);
    int64_t softmaxScoreSize = GetShapeSize(softmaxScoreShape);
    int64_t kvSize = GetShapeSize(kvShape);
    int64_t dXSize = GetShapeSize(dXShape);
    int64_t dWkvSize = GetShapeSize(dWkvShape);
    int64_t dWgateSize = GetShapeSize(dWgateShape);
    int64_t dApeSize = GetShapeSize(dApeShape);

    std::vector<bfloat16> xHostData(xSize, bfloat16(0.1f));
    std::vector<bfloat16> wkvHostData(wkvSize, bfloat16(0.1f));
    std::vector<bfloat16> wgateHostData(wgateSize, bfloat16(0.1f));
    std::vector<bfloat16> dCmpKvHostData(dCmpKvSize, bfloat16(0.1f));
    std::vector<float_t> softmaxScoreHostData(softmaxScoreSize, 0.1f);
    std::vector<float_t> kvHostData(kvSize, 0.1f);
    std::vector<int32_t> startPosHostData(B, 0);
    std::vector<bfloat16> dXHostData(dXSize, bfloat16(0.0f));
    std::vector<bfloat16> dWkvHostData(dWkvSize, bfloat16(0.0f));
    std::vector<bfloat16> dWgateHostData(dWgateSize, bfloat16(0.0f));
    std::vector<float_t> dApeHostData(dApeSize, 0.0f);

    // 4. 创建 aclTensor
    void *xDeviceAddr = nullptr;
    void *wkvDeviceAddr = nullptr;
    void *wgateDeviceAddr = nullptr;
    void *dCmpKvDeviceAddr = nullptr;
    void *softmaxScoreDeviceAddr = nullptr;
    void *kvDeviceAddr = nullptr;
    void *startPosDeviceAddr = nullptr;
    void *dXDeviceAddr = nullptr;
    void *dWkvDeviceAddr = nullptr;
    void *dWgateDeviceAddr = nullptr;
    void *dApeDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *wkv = nullptr;
    aclTensor *wgate = nullptr;
    aclTensor *dCmpKv = nullptr;
    aclTensor *softmaxScore = nullptr;
    aclTensor *kv = nullptr;
    aclTensor *startPos = nullptr;
    aclTensor *dX = nullptr;
    aclTensor *dWkv = nullptr;
    aclTensor *dWgate = nullptr;
    aclTensor *dApe = nullptr;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wkvHostData, wkvShape, &wkvDeviceAddr, aclDataType::ACL_BF16, &wkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wgateHostData, wgateShape, &wgateDeviceAddr, aclDataType::ACL_BF16, &wgate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dCmpKvHostData, dCmpKvShape, &dCmpKvDeviceAddr, aclDataType::ACL_BF16, &dCmpKv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxScoreHostData, softmaxScoreShape, &softmaxScoreDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxScore);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kvHostData, kvShape, &kvDeviceAddr, aclDataType::ACL_FLOAT, &kv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(startPosHostData, startPosShape, &startPosDeviceAddr, aclDataType::ACL_INT32, &startPos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dXHostData, dXShape, &dXDeviceAddr, aclDataType::ACL_BF16, &dX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dWkvHostData, dWkvShape, &dWkvDeviceAddr, aclDataType::ACL_BF16, &dWkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dWgateHostData, dWgateShape, &dWgateDeviceAddr, aclDataType::ACL_BF16, &dWgate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dApeHostData, dApeShape, &dApeDeviceAddr, aclDataType::ACL_FLOAT, &dApe);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 5. 调用 aclnnCompressorGradGetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    ret = aclnnCompressorGradGetWorkspaceSize(x, wkv, wgate, dCmpKv, softmaxScore, kv, nullptr, nullptr, startPos,
                                              cmpRatio, coff, dX, dWkv, dWgate, dApe, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCompressorGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 6. 申请 workspace
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 7. 调用 aclnnCompressorGrad
    ret = aclnnCompressorGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCompressorGrad failed. ERROR: %d\n", ret); return ret);

    // 8. 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 9. 获取输出
    LOG_PRINT("CompressorGrad execution succeeded.\n");
    PrintBf16Result(dXShape, &dXDeviceAddr);
    PrintBf16Result(dWkvShape, &dWkvDeviceAddr);
    PrintBf16Result(dWgateShape, &dWgateDeviceAddr);
    PrintF32Result(dApeShape, &dApeDeviceAddr);

    // 10. 释放资源
    aclDestroyTensor(x);
    aclDestroyTensor(wkv);
    aclDestroyTensor(wgate);
    aclDestroyTensor(dCmpKv);
    aclDestroyTensor(softmaxScore);
    aclDestroyTensor(kv);
    aclDestroyTensor(startPos);
    aclDestroyTensor(dX);
    aclDestroyTensor(dWkv);
    aclDestroyTensor(dWgate);
    aclDestroyTensor(dApe);

    aclrtFree(xDeviceAddr);
    aclrtFree(wkvDeviceAddr);
    aclrtFree(wgateDeviceAddr);
    aclrtFree(dCmpKvDeviceAddr);
    aclrtFree(softmaxScoreDeviceAddr);
    aclrtFree(kvDeviceAddr);
    aclrtFree(startPosDeviceAddr);
    aclrtFree(dXDeviceAddr);
    aclrtFree(dWkvDeviceAddr);
    aclrtFree(dWgateDeviceAddr);
    aclrtFree(dApeDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
