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
 * \file test_aclnn_inplace_partial_rotary_mul_grad_bab.cpp
 * \brief BAB template test for InplacePartialRotaryMulGrad (multi-B/N, partial slice, interleave mode).
 *
 * BAB template constraints:
 *   - Layout: BSND (dy shape [B, S, N, D])
 *   - cosb_==1 (cos/sin broadcast on B dimension)
 *   - rotary_mode=1 (interleave) only
 *
 * This test covers:
 *   - B=2, S=2, N=2, D=8 (multi-batch, multi-head)
 *   - cos/sin shape [1, 2, 1, 4] (broadcast on B and N)
 *   - partialSliceData = {2, 6} (rotary applied only to D indices [2, 6))
 *   - Elements outside the slice pass through unchanged
 *
 * Interleave gradient formula (within slice [start, end)):
 *   dx[start+2k]   = cos[k*2]   * dy[start+2k]   + sin[k*2+1] * dy[start+2k+1]
 *   dx[start+2k+1] = cos[k*2+1] * dy[start+2k+1] - sin[k*2]   * dy[start+2k]
 */

#include "acl/acl.h"
#include "aclnnop/aclnn_inplace_partial_rotary_mul_grad.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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

// Reference: interleave gradient computation with partial slice
void InterleaveGradRef(const std::vector<float>& dy, const std::vector<float>& cos, const std::vector<float>& sin,
                       std::vector<float>& dx, int64_t b, int64_t s, int64_t n, int64_t d, int64_t sliceStart,
                       int64_t sliceEnd)
{
    int64_t cosD = sliceEnd - sliceStart;
    for (int64_t bi = 0; bi < b; bi++) {
        for (int64_t si = 0; si < s; si++) {
            for (int64_t ni = 0; ni < n; ni++) {
                int64_t groupOffset = ((bi * s + si) * n + ni) * d;
                // Copy dy to dx first (pass-through for elements outside the slice)
                for (int64_t di = 0; di < d; di++) {
                    dx[groupOffset + di] = dy[groupOffset + di];
                }
                // Apply interleave rotary grad on the slice
                int64_t cosSOffset = si * cosD;
                for (int64_t k = 0; k < cosD / 2; k++) {
                    int64_t idx0 = groupOffset + sliceStart + 2 * k;
                    int64_t idx1 = groupOffset + sliceStart + 2 * k + 1;
                    int64_t cIdx0 = cosSOffset + 2 * k;
                    int64_t cIdx1 = cosSOffset + 2 * k + 1;
                    dx[idx0] = cos[cIdx0] * dy[idx0] + sin[cIdx1] * dy[idx1];
                    dx[idx1] = cos[cIdx1] * dy[idx1] - sin[cIdx0] * dy[idx0];
                }
            }
        }
    }
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // BSND layout: dy=[B, S, N, D], cos=[1, S, 1, D_cos], cosb_=1
    int64_t B = 2;
    int64_t S = 2;
    int64_t N = 2;
    int64_t D = 8;

    std::vector<int64_t> dyShape = {B, S, N, D};
    std::vector<int64_t> cosShape = {1, S, 1, 4}; // cosb_=1: B dimension broadcast; D_cos=4
    std::vector<int64_t> sinShape = {1, S, 1, 4};
    int64_t rotaryMode = 1; // interleave
    std::vector<int64_t> partialSliceData = {2, 6};

    // Test data
    // dy: 8 groups (B*S*N = 2*2*2 = 8) of 8 elements each
    // Each group: [1, 2, 3, 4, 5, 6, 7, 8]
    // cos: S=2 groups, each [0.5, 0.6, 0.7, 0.8]
    // sin: S=2 groups, each [0.1, 0.2, 0.3, 0.4]
    //
    // partialSliceData = {2, 6} => rotary applied to D indices [2, 6), i.e. positions 2,3,4,5
    // Elements at D indices 0,1,6,7 pass through unchanged.
    //
    // Interleave grad within slice [2, 6):
    //   Pair (2,3): cos[0,1]=[0.5,0.6], sin[0,1]=[0.1,0.2]
    //     dx[2] = 0.5*3 + 0.2*4 = 1.5 + 0.8 = 2.3
    //     dx[3] = 0.6*4 - 0.1*3 = 2.4 - 0.3 = 2.1
    //   Pair (4,5): cos[2,3]=[0.7,0.8], sin[2,3]=[0.3,0.4]
    //     dx[4] = 0.7*5 + 0.4*6 = 3.5 + 2.4 = 5.9
    //     dx[5] = 0.8*6 - 0.3*5 = 4.8 - 1.5 = 3.3
    //
    // Per-group expected dx: [1, 2, 2.3, 2.1, 5.9, 3.3, 7, 8]

    std::vector<float> dyHostData;
    std::vector<float> expectedDx;
    for (int i = 0; i < B * S * N; i++) {
        dyHostData.insert(dyHostData.end(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
        expectedDx.insert(expectedDx.end(), {1.0f, 2.0f, 2.3f, 2.1f, 5.9f, 3.3f, 7.0f, 8.0f});
    }

    // cos/sin: same values for both S positions (broadcast on B and N)
    std::vector<float> cosHostData = {0.5f, 0.6f, 0.7f, 0.8f, 0.5f, 0.6f, 0.7f, 0.8f};
    std::vector<float> sinHostData = {0.1f, 0.2f, 0.3f, 0.4f, 0.1f, 0.2f, 0.3f, 0.4f};

    std::vector<float> referenceDx(expectedDx.size(), 0.0f);
    InterleaveGradRef(dyHostData, cosHostData, sinHostData, referenceDx, B, S, N, D, partialSliceData[0],
                      partialSliceData[1]);

    void* dyRefDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    aclTensor* dyRef = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;

    ret = CreateAclTensor(dyHostData, dyShape, &dyRefDeviceAddr, aclDataType::ACL_FLOAT, &dyRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray* partialSlice = aclCreateIntArray(partialSliceData.data(), partialSliceData.size());

    // First stage: get workspace size
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnInplacePartialRotaryMulGradGetWorkspaceSize(dyRef, cos, sin, rotaryMode, partialSlice, &workspaceSize,
                                                           &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnInplacePartialRotaryMulGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Second stage: execute
    ret = aclnnInplacePartialRotaryMulGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePartialRotaryMulGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Read back result (inplace: dyRef was overwritten with dx)
    auto size = GetShapeSize(dyShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dyRefDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // Verify results
    const float epsilon = 1e-6f;
    bool allPassed = true;
    for (int64_t i = 0; i < size; i++) {
        bool ok = std::fabs(resultData[i] - referenceDx[i]) < epsilon;
        if (!ok) {
            LOG_PRINT("MISMATCH at [%ld]: got %.6f, expected %.6f (reference %.6f)\n", i, resultData[i], expectedDx[i],
                      referenceDx[i]);
            allPassed = false;
        }
    }

    if (allPassed) {
        LOG_PRINT("BAB test PASSED: all %ld elements match reference.\n", size);
        LOG_PRINT("Result: ");
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("%.6f ", resultData[i]);
        }
        LOG_PRINT("\n");
    } else {
        LOG_PRINT("BAB test FAILED.\n");
    }

    // Cleanup
    aclDestroyIntArray(partialSlice);
    aclDestroyTensor(dyRef);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);

    aclrtFree(dyRefDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return allPassed ? 0 : 1;
}
