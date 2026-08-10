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
 * \file test_aclnn_quant_compressor.cpp
 * \brief QuantCompressor 算子 aclnn 调用示例（A5 / ascend950）
 *        场景：C4A（D=512, coff=2, cmp_ratio=4, cache_mode=1, BSH layout, HIFLOAT8）
 *        QuantCompressor 仅支持 ascend950。
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_compressor.h"
#include "opdev/bfloat16.h"
#include "opdev/hifloat8.h"

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
using op::HiFloat8;

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
        LOG_PRINT("cmp_kv[%ld] is: %f\n", i, static_cast<float>(resultData[i]));
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

    // 场景参数: C4A (D=512, coff=2, cmp_ratio=4, cache_mode=1, BSH layout)
    int64_t B = 1;
    int64_t S = 4;
    int64_t hiddenSize = 4096;
    int64_t headDim = 512;
    int64_t coff = 2;
    int64_t cmpRatio = 4;
    int64_t cacheMode = 1; // 1: 连续buffer (LINEAR_BUFFER)
    int64_t quantMode = 1; // 1: A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL
    int64_t blockSize = 128;

    int64_t Smax = S;
    int64_t maxBlockNumPerBatch = (Smax + blockSize - 1) / blockSize;
    int64_t blockNum = B * maxBlockNumPerBatch;
    int64_t coffD = coff * headDim;
    int64_t stateCacheStrideDim0 = blockSize * 2 * coffD; // state_cache 0轴 stride = dim1 * dim2

    // 2. 构造输入与输出 shape
    std::vector<int64_t> xShape = {B, S, hiddenSize};
    std::vector<int64_t> wkvShape = {coffD, hiddenSize};
    std::vector<int64_t> wgateShape = {coffD, hiddenSize};
    std::vector<int64_t> stateCacheShape = {blockNum, blockSize, 2 * coffD};
    std::vector<int64_t> apeShape = {cmpRatio, coffD};
    std::vector<int64_t> xDescaleShape = {1};
    std::vector<int64_t> wkvDescaleShape = {coffD};
    std::vector<int64_t> wgateDescaleShape = {coffD};
    std::vector<int64_t> stateBlockTableShape = {B, maxBlockNumPerBatch};
    std::vector<int64_t> startPosShape = {B};
    int64_t Sr = (S + cmpRatio - 1) / cmpRatio;
    std::vector<int64_t> cmpKvShape = {B, Sr, headDim};

    // 3. 构造 host 数据
    int64_t xSize = GetShapeSize(xShape);
    int64_t wkvSize = GetShapeSize(wkvShape);
    int64_t wgateSize = GetShapeSize(wgateShape);
    int64_t stateCacheSize = GetShapeSize(stateCacheShape);
    int64_t apeSize = GetShapeSize(apeShape);
    int64_t cmpKvSize = GetShapeSize(cmpKvShape);

    std::vector<HiFloat8> xHostData(xSize, HiFloat8(0.1f));
    std::vector<HiFloat8> wkvHostData(wkvSize, HiFloat8(0.1f));
    std::vector<HiFloat8> wgateHostData(wgateSize, HiFloat8(0.1f));
    std::vector<float_t> stateCacheHostData(stateCacheSize, 0.1f);
    std::vector<float_t> apeHostData(apeSize, 0.1f);
    std::vector<float_t> xDescaleHostData = {0.5f};
    std::vector<float_t> wkvDescaleHostData(coffD, 0.1f);
    std::vector<float_t> wgateDescaleHostData(coffD, 0.1f);
    std::vector<int32_t> stateBlockTableHostData;
    for (int64_t i = 0; i < B * maxBlockNumPerBatch; i++) {
        stateBlockTableHostData.push_back(static_cast<int32_t>(i + 1));
    }
    std::vector<int32_t> startPosHostData(B, 0);
    std::vector<bfloat16> cmpKvHostData(cmpKvSize, bfloat16(0.0f));

    // 4. 创建 aclTensor
    void *xDeviceAddr = nullptr;
    void *wkvDeviceAddr = nullptr;
    void *wgateDeviceAddr = nullptr;
    void *stateCacheDeviceAddr = nullptr;
    void *apeDeviceAddr = nullptr;
    void *xDescaleDeviceAddr = nullptr;
    void *wkvDescaleDeviceAddr = nullptr;
    void *wgateDescaleDeviceAddr = nullptr;
    void *stateBlockTableDeviceAddr = nullptr;
    void *startPosDeviceAddr = nullptr;
    void *cmpKvDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *wkv = nullptr;
    aclTensor *wgate = nullptr;
    aclTensor *stateCacheRef = nullptr;
    aclTensor *ape = nullptr;
    aclTensor *xDescale = nullptr;
    aclTensor *wkvDescale = nullptr;
    aclTensor *wgateDescale = nullptr;
    aclTensor *stateBlockTable = nullptr;
    aclTensor *startPos = nullptr;
    aclTensor *cmpKvOut = nullptr;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_HIFLOAT8, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wkvHostData, wkvShape, &wkvDeviceAddr, aclDataType::ACL_HIFLOAT8, &wkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wgateHostData, wgateShape, &wgateDeviceAddr, aclDataType::ACL_HIFLOAT8, &wgate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateCacheHostData, stateCacheShape, &stateCacheDeviceAddr, aclDataType::ACL_FLOAT,
                          &stateCacheRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(apeHostData, apeShape, &apeDeviceAddr, aclDataType::ACL_FLOAT, &ape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xDescaleHostData, xDescaleShape, &xDescaleDeviceAddr, aclDataType::ACL_FLOAT, &xDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wkvDescaleHostData, wkvDescaleShape, &wkvDescaleDeviceAddr, aclDataType::ACL_FLOAT,
                          &wkvDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wgateDescaleHostData, wgateDescaleShape, &wgateDescaleDeviceAddr, aclDataType::ACL_FLOAT,
                          &wgateDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateBlockTableHostData, stateBlockTableShape, &stateBlockTableDeviceAddr,
                          aclDataType::ACL_INT32, &stateBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(startPosHostData, startPosShape, &startPosDeviceAddr, aclDataType::ACL_INT32, &startPos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpKvHostData, cmpKvShape, &cmpKvDeviceAddr, aclDataType::ACL_BF16, &cmpKvOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 5. 调用 aclnnQuantCompressorGetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    ret = aclnnQuantCompressorGetWorkspaceSize(x, wkv, wgate, stateCacheRef, ape, xDescale, wkvDescale, wgateDescale,
                                               stateBlockTable, nullptr, nullptr, startPos, quantMode, cmpRatio, coff,
                                               cacheMode, stateCacheStrideDim0, cmpKvOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantCompressorGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 6. 申请 workspace
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 7. 调用 aclnnQuantCompressor
    ret = aclnnQuantCompressor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantCompressor failed. ERROR: %d\n", ret); return ret);

    // 8. 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 9. 获取输出
    LOG_PRINT("QuantCompressor execution succeeded.\n");
    PrintBf16Result(cmpKvShape, &cmpKvDeviceAddr);

    // 10. 释放资源
    aclDestroyTensor(x);
    aclDestroyTensor(wkv);
    aclDestroyTensor(wgate);
    aclDestroyTensor(stateCacheRef);
    aclDestroyTensor(ape);
    aclDestroyTensor(xDescale);
    aclDestroyTensor(wkvDescale);
    aclDestroyTensor(wgateDescale);
    aclDestroyTensor(stateBlockTable);
    aclDestroyTensor(startPos);
    aclDestroyTensor(cmpKvOut);

    aclrtFree(xDeviceAddr);
    aclrtFree(wkvDeviceAddr);
    aclrtFree(wgateDeviceAddr);
    aclrtFree(stateCacheDeviceAddr);
    aclrtFree(apeDeviceAddr);
    aclrtFree(xDescaleDeviceAddr);
    aclrtFree(wkvDescaleDeviceAddr);
    aclrtFree(wgateDescaleDeviceAddr);
    aclrtFree(stateBlockTableDeviceAddr);
    aclrtFree(startPosDeviceAddr);
    aclrtFree(cmpKvDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
