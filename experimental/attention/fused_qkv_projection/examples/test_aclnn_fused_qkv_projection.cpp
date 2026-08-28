/** * Copyright (c) 2026 Huawei Technologies Co., Ltd. * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of * CANN Open Software License Agreement Version 2.0 (the "License"). *
 * Please refer to the License for details. You may not use this file except in compliance with the License. * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. * See LICENSE in the root of
 * the software repository for the full text of the License. */
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_qkv_projection.h"

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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n %s", ret, aclGetRecentErrMsg()); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n %s", ret, aclGetRecentErrMsg());
              return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n %s", ret, aclGetRecentErrMsg());
              return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n%s", ret, aclGetRecentErrMsg());
              return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n%s", ret, aclGetRecentErrMsg());
              return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename DeviceT, aclDataType aclDType>
int RunTest(int64_t batchSize, int64_t seqLen, int64_t hiddenSize, int64_t qDim, int64_t kDim, int64_t vDim,
            bool useBias, int testIdx, const char *typeStr, aclrtStream stream)
{
    int64_t fusedDim = qDim + kDim + vDim;
    int64_t m = batchSize * seqLen;
    int ret;
    std::vector<int64_t> hsShape = {batchSize, seqLen, hiddenSize};
    std::vector<int64_t> wtShape = {hiddenSize, fusedDim};
    std::vector<int64_t> biasShape = {fusedDim};
    std::vector<int64_t> qShape = {batchSize, seqLen, qDim};
    std::vector<int64_t> kShape = {batchSize, seqLen, kDim};
    std::vector<int64_t> vShape = {batchSize, seqLen, vDim};

    // CPU 参考保持高精度 float 计算
    std::vector<float> hsDataFloat(GetShapeSize(hsShape));
    std::vector<float> wtDataFloat(GetShapeSize(wtShape));
    std::vector<float> biasDataFloat(GetShapeSize(biasShape));

    for (int64_t i = 0; i < (int64_t)hsDataFloat.size(); i++)
        hsDataFloat[i] = std::sin(static_cast<float>(i) * 0.73f) * 1.5f;
    for (int64_t i = 0; i < (int64_t)wtDataFloat.size(); i++)
        wtDataFloat[i] = std::cos(static_cast<float>(i) * 1.19f);
    for (int64_t i = 0; i < (int64_t)biasDataFloat.size(); i++)
        biasDataFloat[i] = std::sin(static_cast<float>(i) * 2.37f) * 0.5f;

    // 转换为指定设备数据类型
    std::vector<DeviceT> hsData(hsDataFloat.size());
    std::vector<DeviceT> wtData(wtDataFloat.size());
    std::vector<DeviceT> biasData(biasDataFloat.size());
    for (size_t i = 0; i < hsData.size(); i++)
        hsData[i] = static_cast<DeviceT>(hsDataFloat[i]);
    for (size_t i = 0; i < wtData.size(); i++)
        wtData[i] = static_cast<DeviceT>(wtDataFloat[i]);
    for (size_t i = 0; i < biasData.size(); i++)
        biasData[i] = static_cast<DeviceT>(biasDataFloat[i]);

    // CPU 计算期望输出
    std::vector<float> expQ(GetShapeSize(qShape));
    std::vector<float> expK(GetShapeSize(kShape));
    std::vector<float> expV(GetShapeSize(vShape));

    for (int64_t r = 0; r < m; r++) {
        for (int64_t fd = 0; fd < fusedDim; fd++) {
            float sum = 0.0f;
            for (int64_t hh = 0; hh < hiddenSize; hh++)
                sum += hsDataFloat[r * hiddenSize + hh] * wtDataFloat[hh * fusedDim + fd];
            if (useBias)
                sum += biasDataFloat[fd];

            if (fd < qDim)
                expQ[r * qDim + fd] = sum;
            else if (fd < qDim + kDim)
                expK[r * kDim + (fd - qDim)] = sum;
            else
                expV[r * vDim + (fd - qDim - kDim)] = sum;
        }
    }

    int printN = 5;
    LOG_PRINT("[Test %d %s] Input X[0..%d]: ", testIdx, typeStr, std::min(printN, (int)hsData.size()) - 1);
    for (int i = 0; i < printN && i < (int)hsData.size(); i++)
        LOG_PRINT("%.4f ", (float)hsData[i]);
    LOG_PRINT("\n");

    aclTensor *hs = nullptr;
    void *hsDev = nullptr;
    ret = CreateAclTensor(hsData, hsShape, &hsDev, aclDType, &hs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *wt = nullptr;
    void *wtDev = nullptr;
    ret = CreateAclTensor(wtData, wtShape, &wtDev, aclDType, &wt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *bias = nullptr;
    void *biasDev = nullptr;
    if (useBias) {
        ret = CreateAclTensor(biasData, biasShape, &biasDev, aclDType, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    aclTensor *q = nullptr, *k = nullptr, *v = nullptr;
    void *qDev = nullptr, *kDev = nullptr, *vDev = nullptr;
    int64_t qs = GetShapeSize(qShape), ks = GetShapeSize(kShape), vs = GetShapeSize(vShape);
    int64_t maxOutSize = qs > ks ? (qs > vs ? qs : vs) : (ks > vs ? ks : vs);
    std::vector<DeviceT> zeros(maxOutSize, static_cast<DeviceT>(0));
    ret = CreateAclTensor(zeros, qShape, &qDev, aclDType, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(zeros, kShape, &kDev, aclDType, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(zeros, vShape, &vDev, aclDType, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    ret = aclnnFusedQkvProjectionGetWorkspaceSize(hs, wt, bias, qDim, kDim, vDim, q, k, v, &wsSize, &exec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[Test %d %s] GetWorkspaceSize failed. %d\n", testIdx, typeStr, ret);
              return ret);

    void *ws = nullptr;
    if (wsSize > 0) {
        aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnFusedQkvProjection(ws, wsSize, exec, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[Test %d %s] Execute failed. %d\n", testIdx, typeStr, ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[Test %d %s] Sync failed. %d\n%s", testIdx, typeStr, ret, aclGetRecentErrMsg());
              return ret);

    std::vector<DeviceT> qD(GetShapeSize(qShape)), kD(GetShapeSize(kShape)), vD(GetShapeSize(vShape));
    aclrtMemcpy(qD.data(), qD.size() * sizeof(DeviceT), qDev, qD.size() * sizeof(DeviceT), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(kD.data(), kD.size() * sizeof(DeviceT), kDev, kD.size() * sizeof(DeviceT), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(vD.data(), vD.size() * sizeof(DeviceT), vDev, vD.size() * sizeof(DeviceT), ACL_MEMCPY_DEVICE_TO_HOST);

    float eps = std::is_same<DeviceT, float>::value ? 1e-4f : 5e-2f;

    for (int64_t i = 0; i < (int64_t)qD.size(); i++) {
        if (std::fabs((float)qD[i] - expQ[i]) > eps) {
            LOG_PRINT("[Test %d %s] Q[%ld] mismatch: exp %.6f got %.6f (diff=%.6f)\n", testIdx, typeStr, i, expQ[i],
                      (float)qD[i], std::fabs((float)qD[i] - expQ[i]));
            return -1;
        }
    }
    for (int64_t i = 0; i < (int64_t)kD.size(); i++) {
        if (std::fabs((float)kD[i] - expK[i]) > eps) {
            LOG_PRINT("[Test %d %s] K[%ld] mismatch: exp %.6f got %.6f (diff=%.6f)\n", testIdx, typeStr, i, expK[i],
                      (float)kD[i], std::fabs((float)kD[i] - expK[i]));
            return -1;
        }
    }
    for (int64_t i = 0; i < (int64_t)vD.size(); i++) {
        if (std::fabs((float)vD[i] - expV[i]) > eps) {
            LOG_PRINT("[Test %d %s] V[%ld] mismatch: exp %.6f got %.6f (diff=%.6f)\n", testIdx, typeStr, i, expV[i],
                      (float)vD[i], std::fabs((float)vD[i] - expV[i]));
            return -1;
        }
    }

    LOG_PRINT("[Test %d %s] PASSED (batch=%ld seq=%ld h=%ld q=%ld k=%ld v=%ld bias=%d)\n", testIdx, typeStr, batchSize,
              seqLen, hiddenSize, qDim, kDim, vDim, useBias);

    aclDestroyTensor(hs);
    aclrtFree(hsDev);
    aclDestroyTensor(wt);
    aclrtFree(wtDev);
    if (bias) {
        aclDestroyTensor(bias);
        aclrtFree(biasDev);
    }
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclrtFree(qDev);
    aclrtFree(kDev);
    aclrtFree(vDev);
    if (wsSize > 0)
        aclrtFree(ws);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. %d\n", ret); return ret);

    struct Case {
        int64_t b, s, h, q, k, v;
        bool bias;
        const char *desc;
    };
    Case tests[] = {
        {2, 8, 16, 16, 8, 8, true, "M=16 最小Cube"},          {1, 32, 16, 16, 8, 8, true, "M=32"},
        {1, 16, 64, 32, 16, 16, false, "M=16 无偏置"},        {1, 64, 64, 32, 16, 16, true, "M=64"},
        {4, 32, 64, 64, 32, 32, true, "M=128 q≠k≠v"},         {8, 16, 32, 16, 8, 8, false, "M=128 大batch无偏置"},
        {1, 256, 64, 32, 16, 16, true, "M=256 长序列"},       {1, 128, 64, 32, 16, 16, false, "M=128 无偏置"},
        {2, 64, 32, 16, 8, 8, true, "M=128 中等batch"},       {1, 16, 8, 8, 8, 8, true, "M=16 h=8 极小hidden"},
        {1, 1024, 64, 32, 16, 16, true, "M=1024 超长序列"},   {2, 8, 256, 128, 64, 64, false, "M=16 h=256 大hidden"},
        {4, 4, 128, 64, 32, 32, true, "M=16 h=128 大hidden"}, {1, 17, 32, 16, 8, 8, true, "M=17 非16倍数"},
        {1, 31, 32, 16, 8, 8, false, "M=31 非16倍数"},        {2, 8, 48, 16, 16, 16, true, "q=k=v=16"},
        {1, 32, 96, 32, 32, 32, false, "q=k=v=32"},           {2, 16, 24, 8, 8, 8, true, "q=k=v=8"},
        {3, 7, 32, 17, 7, 8, false, "q=17 非8B对齐"},         {2, 8, 32, 9, 9, 14, true, "q=9 k=9 非对齐"},
        {1, 16, 32, 10, 10, 12, false, "q=10 非对齐"},        {2, 10, 32, 20, 6, 6, true, "k=6 非对齐"},
    };

    int numTests = sizeof(tests) / sizeof(tests[0]);
    int passed = 0, failed = 0;

    LOG_PRINT("\n========== dtype=FLOAT32 ==========\n\n");
    for (int i = 0; i < numTests; i++) {
        LOG_PRINT("[F32 %d/%d] %s (b=%ld s=%ld h=%ld q=%ld k=%ld v=%ld bias=%d)\n", i + 1, numTests, tests[i].desc,
                  tests[i].b, tests[i].s, tests[i].h, tests[i].q, tests[i].k, tests[i].v, tests[i].bias);
        ret = RunTest<float, ACL_FLOAT>(tests[i].b, tests[i].s, tests[i].h, tests[i].q, tests[i].k, tests[i].v,
                                        tests[i].bias, i + 1, "F32", stream);
        if (ret) {
            LOG_PRINT("=== F32 Test %d FAILED ===\n", i + 1);
            failed++;
        } else {
            passed++;
        }
    }

    LOG_PRINT("\n========== dtype=FLOAT16 ==========\n\n");
    for (int i = 0; i < numTests; i++) {
        LOG_PRINT("[F16 %d/%d] %s (b=%ld s=%ld h=%ld q=%ld k=%ld v=%ld bias=%d)\n", i + 1, numTests, tests[i].desc,
                  tests[i].b, tests[i].s, tests[i].h, tests[i].q, tests[i].k, tests[i].v, tests[i].bias);
        ret = RunTest<__fp16, ACL_FLOAT16>(tests[i].b, tests[i].s, tests[i].h, tests[i].q, tests[i].k, tests[i].v,
                                           tests[i].bias, i + 1, "F16", stream);
        if (ret) {
            LOG_PRINT("=== F16 Test %d FAILED ===\n", i + 1);
            failed++;
        } else {
            passed++;
        }
    }

    LOG_PRINT("\n=== %d/%d tests PASSED, %d FAILED ===\n", passed, numTests * 2, failed);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
