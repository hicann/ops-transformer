/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_mhc_post.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_post_backward.h"
#include "securec.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

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
    // Fixed writing method, AscendCL initialization.
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to request device side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    // Call aclrtMemcpy to copy host side data to device side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    // Calculate the strides of continuous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor interface to create aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

} // namespace

int main()
{
    // 1. (Fixed writing method)  device/stream initialization. Refer to AscendCL's list of external interfaces.
    // Fill in the deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    LOG_PRINT("ACL environment initialized successfully!\n");

    // 2. To construct input and output, it is necessary to customize the construction according to the API interface.
    // Example: BSND format (B, S, n, D)
    std::vector<int64_t> gradYShape = {1, 1024, 4, 5120}; // BSND
    std::vector<int64_t> xShape = {1, 1024, 4, 5120};     // BSND
    std::vector<int64_t> hResShape = {1, 1024, 4, 4};     // BSND
    std::vector<int64_t> hOutShape = {1, 1024, 5120};     // BSD
    std::vector<int64_t> hPostShape = {1, 1024, 4};       // BSn
    std::vector<int64_t> gradXShape = {1, 1024, 4, 5120}; // BSND
    std::vector<int64_t> gradHresShape = {1, 1024, 4, 4}; // BSND
    std::vector<int64_t> gradHoutShape = {1, 1024, 5120}; // BSD
    std::vector<int64_t> gradHpostShape = {1, 1024, 4};   // BSn

    void* gradYDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* hResDeviceAddr = nullptr;
    void* hOutDeviceAddr = nullptr;
    void* hPostDeviceAddr = nullptr;
    void* gradXDeviceAddr = nullptr;
    void* gradHresDeviceAddr = nullptr;
    void* gradHoutDeviceAddr = nullptr;
    void* gradHpostDeviceAddr = nullptr;

    aclTensor* gradYTensor = nullptr;
    aclTensor* xTensor = nullptr;
    aclTensor* hResTensor = nullptr;
    aclTensor* hOutTensor = nullptr;
    aclTensor* hPostTensor = nullptr;
    aclTensor* gradXTensor = nullptr;
    aclTensor* gradHresTensor = nullptr;
    aclTensor* gradHoutTensor = nullptr;
    aclTensor* gradHpostTensor = nullptr;

    int64_t gradYShapeSize = GetShapeSize(gradYShape);
    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t hResShapeSize = GetShapeSize(hResShape);
    int64_t hOutShapeSize = GetShapeSize(hOutShape);
    int64_t hPostShapeSize = GetShapeSize(hPostShape);
    int64_t gradXShapeSize = GetShapeSize(gradXShape);
    int64_t gradHresShapeSize = GetShapeSize(gradHresShape);
    int64_t gradHoutShapeSize = GetShapeSize(gradHoutShape);
    int64_t gradHpostShapeSize = GetShapeSize(gradHpostShape);
    std::vector<aclFloat16> gradYHostData(gradYShapeSize, aclFloatToFloat16(1.0f));
    std::vector<aclFloat16> xHostData(xShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> hResHostData(hResShapeSize, 1.0f);
    std::vector<aclFloat16> hOutHostData(hOutShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> hPostHostData(hPostShapeSize, 1.0f);
    std::vector<aclFloat16> gradXHostData(gradXShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> gradHresHostData(gradHresShapeSize, 1.0f);
    std::vector<aclFloat16> gradHoutHostData(gradHoutShapeSize, aclFloatToFloat16(1.0f));
    std::vector<float> gradHpostHostData(gradHpostShapeSize, 1.0f);

    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT16, &gradYTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &xTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create hRes aclTensor.
    ret = CreateAclTensor(hResHostData, hResShape, &hResDeviceAddr, aclDataType::ACL_FLOAT, &hResTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create hOut aclTensor.
    ret = CreateAclTensor(hOutHostData, hOutShape, &hOutDeviceAddr, aclDataType::ACL_FLOAT16, &hOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create hPost aclTensor.
    ret = CreateAclTensor(hPostHostData, hPostShape, &hPostDeviceAddr, aclDataType::ACL_FLOAT, &hPostTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // Create gradX aclTensor.
    ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT16, &gradXTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create gradHres aclTensor.
    ret =
        CreateAclTensor(gradHresHostData, gradHresShape, &gradHresDeviceAddr, aclDataType::ACL_FLOAT, &gradHresTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create gradHout aclTensor.
    ret = CreateAclTensor(gradHoutHostData, gradHoutShape, &gradHoutDeviceAddr, aclDataType::ACL_FLOAT16,
                          &gradHoutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create gradHpost aclTensor.
    ret = CreateAclTensor(gradHpostHostData, gradHpostShape, &gradHpostDeviceAddr, aclDataType::ACL_FLOAT,
                          &gradHpostTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // 3. Call CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first interface.
    ret = aclnnMhcPostBackwardGetWorkspaceSize(gradYTensor, xTensor, hResTensor, hOutTensor, hPostTensor, gradXTensor,
                                               gradHresTensor, gradHoutTensor, gradHpostTensor, &workspaceSize,
                                               &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPostBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    // Apply for device memory based on the workspaceSize calculated from the first interface paragraph.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }
    // Call the second interface.
    ret = aclnnMhcPostBackward(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPostBackward failed. ERROR: %d\n", ret);
        return ret;
    }

    // 4. (Fixed writing method) Synchronize and wait for task execution to end.
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }
    // 5. Retrieve the output value, copy the gradX from the device side memory to the host side.
    auto gradXsize = GetShapeSize(gradXShape);
    std::vector<aclFloat16> gradXData(gradXsize, aclFloatToFloat16(0.0f));
    ret = aclrtMemcpy(gradXData.data(), gradXData.size() * sizeof(gradXData[0]), gradXDeviceAddr,
                      gradXsize * sizeof(gradXData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy gradX from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    auto gradHressize = GetShapeSize(gradHresShape);
    std::vector<float> gradHresData(gradHressize, 0.0f);
    ret = aclrtMemcpy(gradHresData.data(), gradHresData.size() * sizeof(gradHresData[0]), gradHresDeviceAddr,
                      gradHressize * sizeof(gradHresData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy gradHres from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    auto gradHoutsize = GetShapeSize(gradHoutShape);
    std::vector<aclFloat16> gradHoutData(gradHoutsize, aclFloatToFloat16(0.0f));
    ret = aclrtMemcpy(gradHoutData.data(), gradHoutData.size() * sizeof(gradHoutData[0]), gradHoutDeviceAddr,
                      gradHoutsize * sizeof(gradHoutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy gradHout from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    auto gradHpostsize = GetShapeSize(gradHpostShape);
    std::vector<float> gradHpostData(gradHpostsize, 0.0f);
    ret = aclrtMemcpy(gradHpostData.data(), gradHpostData.size() * sizeof(gradHpostData[0]), gradHpostDeviceAddr,
                      gradHpostsize * sizeof(gradHpostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy gradHpost from device to host failed. ERROR: %d\n", ret);
        return ret;
    }

    // 6. Release resources.
    aclDestroyTensor(gradYTensor);
    aclDestroyTensor(xTensor);
    aclDestroyTensor(hResTensor);
    aclDestroyTensor(hOutTensor);
    aclDestroyTensor(hPostTensor);
    aclDestroyTensor(gradXTensor);
    aclDestroyTensor(gradHresTensor);
    aclDestroyTensor(gradHoutTensor);
    aclDestroyTensor(gradHpostTensor);
    aclrtFree(gradYDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(hResDeviceAddr);
    aclrtFree(hOutDeviceAddr);
    aclrtFree(hPostDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    aclrtFree(gradHresDeviceAddr);
    aclrtFree(gradHoutDeviceAddr);
    aclrtFree(gradHpostDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
