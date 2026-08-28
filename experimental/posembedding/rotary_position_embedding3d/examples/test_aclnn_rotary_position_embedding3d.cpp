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
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include "acl/acl.h"
#include "aclnnop/aclnn_rotary_position_embedding3d.h"

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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n%s", ret, aclGetRecentErrMsg()); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n%s", ret, aclGetRecentErrMsg());
              return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n%s", ret, aclGetRecentErrMsg());
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

// ---- Wan2.2-style 3D RoPE reference ----
// Three-axis frequency bands (2:1:1 split), position -> (t,i,j) decoding

static void ComputeBandDims3D(int64_t headDim, int64_t &tBand, int64_t &hBand, int64_t &wBand)
{
    int64_t bandUnit = headDim / 4;
    if (bandUnit % 2 != 0)
        bandUnit -= 1;
    tBand = bandUnit * 2;
    hBand = bandUnit;
    wBand = headDim - tBand - hBand;
    if (wBand % 2 != 0) {
        wBand -= 1;
        tBand += 1;
        if (tBand % 2 != 0) {
            tBand -= 1;
            hBand += 1;
        }
    }
    if (hBand % 2 != 0) {
        hBand -= 1;
        tBand += 1;
    }
    if (tBand % 2 != 0) {
        tBand -= 1;
        hBand += 1;
    }
}

static void FactorVideoDims(int64_t seqLen, int64_t &T, int64_t &H, int64_t &W)
{
    T = 1;
    H = 1;
    W = seqLen;
    if (seqLen <= 1)
        return;
    int64_t maxDim = std::min((int64_t)128, (int64_t)std::sqrt((double)seqLen));
    for (int64_t w = maxDim; w >= 1; w--) {
        if (seqLen % w == 0) {
            int64_t rest = seqLen / w;
            for (int64_t h = std::min((int64_t)128, (int64_t)std::sqrt((double)rest)); h >= 1; h--) {
                if (rest % h == 0) {
                    T = rest / h;
                    H = h;
                    W = w;
                    return;
                }
            }
        }
    }
}

// CPU reference: true 3D spatial RoPE
// Computes theta tables from freqBase internally, decodes l->(t,i,j)
static void RoPE3DCore(int64_t batch, int64_t seqLen, int64_t headDim, int64_t vidT, int64_t vidH, int64_t vidW,
                       const float *x, float *y, float freqBase)
{
    int64_t tBand, hBand, wBand;
    ComputeBandDims3D(headDim, tBand, hBand, wBand);
    int64_t halfD = headDim / 2;
    int64_t halfTBand = tBand / 2;
    int64_t halfHBand = hBand / 2;
    int64_t halfWBand = wBand / 2;

    float rT = std::pow(freqBase, -2.0 / (double)tBand);
    float rH = std::pow(freqBase, -2.0 / (double)hBand);
    float rW = std::pow(freqBase, -2.0 / (double)wBand);

    std::vector<float> thetaT(halfTBand), thetaH(halfHBand), thetaW(halfWBand);
    if (halfTBand > 0) {
        thetaT[0] = 1.0f;
        for (int64_t i = 1; i < halfTBand; i++)
            thetaT[i] = thetaT[i - 1] * rT;
    }
    if (halfHBand > 0) {
        thetaH[0] = 1.0f;
        for (int64_t i = 1; i < halfHBand; i++)
            thetaH[i] = thetaH[i - 1] * rH;
    }
    if (halfWBand > 0) {
        thetaW[0] = 1.0f;
        for (int64_t i = 1; i < halfWBand; i++)
            thetaW[i] = thetaW[i - 1] * rW;
    }

    int64_t hw = vidH * vidW;
    if (hw <= 0)
        hw = 1;
    int64_t wv = (vidW <= 0) ? 1 : vidW;

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t l = 0; l < seqLen; l++) {
            int64_t base = (b * seqLen + l) * headDim;

            // Decode 1D position -> 3D video coordinates
            int64_t tCoord = l / hw;
            int64_t rem = l % hw;
            int64_t iCoord = rem / wv;
            int64_t jCoord = rem % wv;

            // T-band rotation (temporal axis)
            for (int64_t k = 0; k < halfTBand; k++) {
                float angle = static_cast<float>(tCoord) * thetaT[k];
                float c = std::cos(angle);
                float s = std::sin(angle);
                float xl = x[base + k];
                float xr = x[base + halfD + k];
                y[base + k] = xl * c - xr * s;
                y[base + halfD + k] = xl * s + xr * c;
            }

            // H-band rotation (height axis)
            for (int64_t k = 0; k < halfHBand; k++) {
                float angle = static_cast<float>(iCoord) * thetaH[k];
                float c = std::cos(angle);
                float s = std::sin(angle);
                int64_t offset = halfTBand + k;
                float xl = x[base + offset];
                float xr = x[base + halfD + offset];
                y[base + offset] = xl * c - xr * s;
                y[base + halfD + offset] = xl * s + xr * c;
            }

            // W-band rotation (width axis)
            for (int64_t k = 0; k < halfWBand; k++) {
                float angle = static_cast<float>(jCoord) * thetaW[k];
                float c = std::cos(angle);
                float s = std::sin(angle);
                int64_t offset = halfTBand + halfHBand + k;
                float xl = x[base + offset];
                float xr = x[base + halfD + offset];
                y[base + offset] = xl * c - xr * s;
                y[base + halfD + offset] = xl * s + xr * c;
            }
        }
    }
}

// 自动分解网格的参考实现（1D 退化或未显式指定时使用）
static void RefRoPE3D(int64_t batch, int64_t seqLen, int64_t headDim, const float *x, float *y, float freqBase)
{
    int64_t vidT, vidH, vidW;
    FactorVideoDims(seqLen, vidT, vidH, vidW);
    RoPE3DCore(batch, seqLen, headDim, vidT, vidH, vidW, x, y, freqBase);
}

// 显式指定真实视频网格(T/H/W)的参考实现（Wan2.2 风格）
static void RefRoPE3DExplicit(int64_t batch, int64_t seqLen, int64_t headDim, int64_t vidT, int64_t vidH, int64_t vidW,
                              const float *x, float *y, float freqBase)
{
    RoPE3DCore(batch, seqLen, headDim, vidT, vidH, vidW, x, y, freqBase);
}

// ========== 动态加载 aclnn API ==========
int RunTest(int64_t batch, int64_t seqLen, int64_t headDim, int testIdx, aclrtStream stream, bool benchmark)
{
    int64_t total = batch * seqLen * headDim;
    std::vector<int64_t> shape = {batch, seqLen, headDim};
    int ret;

    std::vector<float> x1Data(total);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (int64_t i = 0; i < total; i++) {
        x1Data[i] = dist(rng);
    }

    // Build pre-computed cos/sin for true 3D RoPE (Wan2.2 style)
    int64_t tBand, hBand, wBand;
    ComputeBandDims3D(headDim, tBand, hBand, wBand);
    int64_t halfD = headDim / 2;
    int64_t halfTBand = tBand / 2;
    int64_t halfHBand = hBand / 2;
    int64_t halfWBand = wBand / 2;
    float freqBase = 10000.0f;
    float rT = std::pow(freqBase, -2.0 / (double)tBand);
    float rH = std::pow(freqBase, -2.0 / (double)hBand);
    float rW = std::pow(freqBase, -2.0 / (double)wBand);

    std::vector<float> thetaT(halfTBand), thetaH(halfHBand), thetaW(halfWBand);
    if (halfTBand > 0) {
        thetaT[0] = 1.0f;
        for (int64_t i = 1; i < halfTBand; i++)
            thetaT[i] = thetaT[i - 1] * rT;
    }
    if (halfHBand > 0) {
        thetaH[0] = 1.0f;
        for (int64_t i = 1; i < halfHBand; i++)
            thetaH[i] = thetaH[i - 1] * rH;
    }
    if (halfWBand > 0) {
        thetaW[0] = 1.0f;
        for (int64_t i = 1; i < halfWBand; i++)
            thetaW[i] = thetaW[i - 1] * rW;
    }

    int64_t vidT, vidH, vidW;
    FactorVideoDims(seqLen, vidT, vidH, vidW);
    int64_t hw = vidH * vidW;
    if (hw <= 0)
        hw = 1;
    int64_t wv = (vidW <= 0) ? 1 : vidW;

    std::vector<float> x2Data(total, 0.0f);
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t l = 0; l < seqLen; l++) {
            int64_t base = (b * seqLen + l) * headDim;
            int64_t tCoord = l / hw;
            int64_t rem = l % hw;
            int64_t iCoord = rem / wv;
            int64_t jCoord = rem % wv;
            // T-band cos/sin
            for (int64_t k = 0; k < halfTBand; k++) {
                float a = static_cast<float>(tCoord) * thetaT[k];
                x2Data[base + k] = std::cos(a);
                x2Data[base + halfD + k] = std::sin(a);
            }
            // H-band cos/sin
            for (int64_t k = 0; k < halfHBand; k++) {
                float a = static_cast<float>(iCoord) * thetaH[k];
                x2Data[base + halfTBand + k] = std::cos(a);
                x2Data[base + halfD + halfTBand + k] = std::sin(a);
            }
            // W-band cos/sin
            for (int64_t k = 0; k < halfWBand; k++) {
                float a = static_cast<float>(jCoord) * thetaW[k];
                x2Data[base + halfTBand + halfHBand + k] = std::cos(a);
                x2Data[base + halfD + halfTBand + halfHBand + k] = std::sin(a);
            }
        }
    }

    std::vector<float> yRef(total, 0);
    RefRoPE3D(batch, seqLen, headDim, x1Data.data(), yRef.data(), 10000.0f);

    int printN = (int)std::min((int64_t)headDim, total);
    int64_t printPos = std::min(seqLen - 1, (int64_t)4);
    for (int64_t pos = 0; pos <= printPos; pos++) {
        int64_t pb = pos * headDim;
        LOG_PRINT("[Test %d] X1(pos=%ld)[0..%ld]:\n  ", testIdx, pos, std::min(headDim, (int64_t)16) - 1);
        for (int i = 0; i < std::min(headDim, (int64_t)16); i++) {
            LOG_PRINT("%.4f ", x1Data[pb + i]);
            if ((i + 1) % 16 == 0)
                LOG_PRINT("\n  ");
        }
        LOG_PRINT("\n");
    }
    LOG_PRINT("[Test %d] X2(pos=0..%ld)[0..%ld]:\n  ", testIdx, printPos, std::min(headDim, (int64_t)8) - 1);
    for (int64_t pos = 0; pos <= printPos; pos++) {
        int64_t pb = pos * headDim;
        LOG_PRINT("pos%ld: ", pos);
        for (int i = 0; i < std::min(headDim, (int64_t)8); i++) {
            LOG_PRINT("%.4f ", x2Data[pb + i]);
        }
        LOG_PRINT("\n  ");
    }
    LOG_PRINT("\n");

    aclTensor *x1 = nullptr;
    void *x1Dev = nullptr;
    ret = CreateAclTensor(x1Data, shape, &x1Dev, ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *x2 = nullptr;
    void *x2Dev = nullptr;
    ret = CreateAclTensor(x2Data, shape, &x2Dev, ACL_FLOAT, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *out = nullptr;
    void *outDev = nullptr;
    std::vector<float> outInit(total, 0);
    ret = CreateAclTensor(outInit, shape, &outDev, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    ret = aclnnRotaryPositionEmbedding3dGetWorkspaceSize(x1, x2, 0, 0, 0, out, &wsSize, &exec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[Test %d] GetWorkspaceSize failed. %d\n", testIdx, ret); return ret);

    void *ws = nullptr;
    if (wsSize > 0) {
        aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 正确性验证（1次）
    ret = aclnnRotaryPositionEmbedding3d(ws, wsSize, exec, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[Test %d] Execute failed. %d\n", testIdx, ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[Test %d] Sync failed. %d\n%s", testIdx, ret, aclGetRecentErrMsg());
              return ret);

    std::vector<float> outData(total, 0);
    aclrtMemcpy(outData.data(), total * sizeof(float), outDev, total * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    const float eps = 1e-4f;
    bool allOk = true;
    for (int64_t i = 0; i < total; i++) {
        if (std::fabs(outData[i] - yRef[i]) > eps) {
            if (allOk) {
                LOG_PRINT("  y[0..%d] (pos0):\n  ", printN - 1);
                for (int j = 0; j < printN; j++) {
                    LOG_PRINT("%.4f ", outData[j]);
                    if ((j + 1) % 16 == 0 && j + 1 < printN)
                        LOG_PRINT("\n  ");
                }
                LOG_PRINT("\n  yRef[0..%d] (pos0):\n  ", printN - 1);
                for (int j = 0; j < printN; j++) {
                    LOG_PRINT("%.4f ", yRef[j]);
                    if ((j + 1) % 16 == 0 && j + 1 < printN)
                        LOG_PRINT("\n  ");
                }
                LOG_PRINT("\n");
                allOk = false;
            }
            LOG_PRINT("[Test %d] y[%ld] mismatch: exp %.6f got %.6f (diff %.6f)\n", testIdx, i, yRef[i], outData[i],
                      std::fabs(outData[i] - yRef[i]));
        }
    }

    // 性能测试（仅 benchmark 模式跑）
    double avgMs = 0.0;
    if (benchmark && allOk) {
        const int kWarmup = 5;
        const int kIter = 20;
        // 每次调用需要重建 executor（one-shot）
        for (int i = 0; i < kWarmup; i++) {
            exec = nullptr;
            ret = aclnnRotaryPositionEmbedding3dGetWorkspaceSize(x1, x2, 0, 0, 0, out, &wsSize, &exec);
            if (ret != ACL_SUCCESS)
                break;
            ret = aclnnRotaryPositionEmbedding3d(ws, wsSize, exec, stream);
            aclrtSynchronizeStream(stream);
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < kIter; i++) {
            exec = nullptr;
            aclnnRotaryPositionEmbedding3dGetWorkspaceSize(x1, x2, 0, 0, 0, out, &wsSize, &exec);
            aclnnRotaryPositionEmbedding3d(ws, wsSize, exec, stream);
            aclrtSynchronizeStream(stream);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        avgMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / kIter;
    }

    if (!allOk) {
        aclDestroyTensor(x1);
        aclrtFree(x1Dev);
        aclDestroyTensor(x2);
        aclrtFree(x2Dev);
        aclDestroyTensor(out);
        aclrtFree(outDev);
        if (wsSize > 0)
            aclrtFree(ws);
        return -1;
    }

    LOG_PRINT("[Test %d] PASSED (batch=%ld seq=%ld D=%ld)", testIdx, batch, seqLen, headDim);
    if (avgMs > 0) {
        LOG_PRINT("  %.3f ms\n", avgMs);
    } else {
        LOG_PRINT("\n");
    }
    // Print y for first few positions (first 8 dims each)
    int64_t showPos = std::min(seqLen - 1, (int64_t)4);
    for (int64_t pos = 0; pos <= showPos; pos++) {
        int64_t pb = pos * headDim;
        LOG_PRINT("  y(pos=%ld)[0..7]: ", pos);
        for (int i = 0; i < std::min(headDim, (int64_t)8); i++) {
            LOG_PRINT("%.4f ", outData[pb + i]);
        }
        LOG_PRINT("\n");
    }

    aclDestroyTensor(x1);
    aclrtFree(x1Dev);
    aclDestroyTensor(x2);
    aclrtFree(x2Dev);
    aclDestroyTensor(out);
    aclrtFree(outDev);
    if (wsSize > 0)
        aclrtFree(ws);
    return 0;
}

// ===== Explicit 3D test with known T/H/W (bypasses FactorVideoDims) =====
// ===== 3D test using FactorVideoDims (same as kernel tiling) =====
int RunTest3D(int64_t hintT, int64_t hintH, int64_t hintW, int64_t headDim, int testIdx, aclrtStream stream)
{
    int64_t seqLen = hintT * hintH * hintW;
    int64_t total = seqLen * headDim;
    std::vector<int64_t> shape = {1, seqLen, headDim};
    int ret;

    // 显式使用调用方指定的真实视频网格(T/H/W)，不做自动分解（Wan2.2 风格）
    int64_t actualT = hintT, actualH = hintH, actualW = hintW;
    int64_t hw = actualH * actualW, wv = actualW;

    // Random input data
    std::vector<float> x1Data(total);
    std::mt19937 rng(42 + testIdx);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (int64_t i = 0; i < total; i++)
        x1Data[i] = dist(rng);

    // Band dims
    int64_t tBand, hBand, wBand;
    ComputeBandDims3D(headDim, tBand, hBand, wBand);
    int64_t halfD = headDim / 2;
    int64_t halfTBand = tBand / 2, halfHBand = hBand / 2, halfWBand = wBand / 2;
    float freqBase = 10000.0f;
    float rT = std::pow(freqBase, -2.0 / (double)tBand);
    float rH = std::pow(freqBase, -2.0 / (double)hBand);
    float rW = std::pow(freqBase, -2.0 / (double)wBand);

    std::vector<float> thetaT(halfTBand), thetaH(halfHBand), thetaW(halfWBand);
    if (halfTBand > 0) {
        thetaT[0] = 1.0f;
        for (int64_t i = 1; i < halfTBand; i++)
            thetaT[i] = thetaT[i - 1] * rT;
    }
    if (halfHBand > 0) {
        thetaH[0] = 1.0f;
        for (int64_t i = 1; i < halfHBand; i++)
            thetaH[i] = thetaH[i - 1] * rH;
    }
    if (halfWBand > 0) {
        thetaW[0] = 1.0f;
        for (int64_t i = 1; i < halfWBand; i++)
            thetaW[i] = thetaW[i - 1] * rW;
    }

    // Build cos/sin using FactorVideoDims-derived (t,i,j) — matches kernel
    std::vector<float> x2Data(total, 0.0f);
    for (int64_t l = 0; l < seqLen; l++) {
        int64_t base = l * headDim;
        int64_t tCoord = l / hw, rem = l % hw, iCoord = rem / wv, jCoord = rem % wv;
        for (int64_t k = 0; k < halfTBand; k++) {
            float a = (float)tCoord * thetaT[k];
            x2Data[base + k] = cosf(a);
            x2Data[base + halfD + k] = sinf(a);
        }
        for (int64_t k = 0; k < halfHBand; k++) {
            float a = (float)iCoord * thetaH[k];
            x2Data[base + halfTBand + k] = cosf(a);
            x2Data[base + halfD + halfTBand + k] = sinf(a);
        }
        for (int64_t k = 0; k < halfWBand; k++) {
            float a = (float)jCoord * thetaW[k];
            x2Data[base + halfTBand + halfHBand + k] = cosf(a);
            x2Data[base + halfD + halfTBand + halfHBand + k] = sinf(a);
        }
    }

    // Reference：使用显式网格(T/H/W)生成 golden
    std::vector<float> yRef(total, 0);
    RefRoPE3DExplicit(1, seqLen, headDim, actualT, actualH, actualW, x1Data.data(), yRef.data(), freqBase);

    // NPU kernel execution
    aclTensor *x1 = nullptr, *x2 = nullptr, *out = nullptr;
    void *x1Dev = nullptr, *x2Dev = nullptr, *outDev = nullptr;
    ret = CreateAclTensor(x1Data, shape, &x1Dev, ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2Data, shape, &x2Dev, ACL_FLOAT, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<float> outInit(total, 0);
    ret = CreateAclTensor(outInit, shape, &outDev, ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    ret = aclnnRotaryPositionEmbedding3dGetWorkspaceSize(x1, x2, hintT, hintH, hintW, out, &wsSize, &exec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[3DTest %d] GetWorkspaceSize failed. %d\n", testIdx, ret); return ret);
    void *ws = nullptr;
    if (wsSize > 0) {
        aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnRotaryPositionEmbedding3d(ws, wsSize, exec, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclrtSynchronizeStream(stream);

    std::vector<float> outData(total, 0);
    aclrtMemcpy(outData.data(), total * sizeof(float), outDev, total * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    // Full validation
    const float eps = 1e-4f;
    bool allOk = true;
    for (int64_t i = 0; i < total; i++) {
        if (fabsf(outData[i] - yRef[i]) > eps) {
            if (allOk) {
                LOG_PRINT("[3DTest %d] FAIL: first mismatch at idx %ld: exp %.6f got %.6f\n", testIdx, i, yRef[i],
                          outData[i]);
                allOk = false;
            }
        }
    }

    // Diagnostics: print positions with actual (t,i,j)
    LOG_PRINT("[3DTest %d] hint=(T=%ld,H=%ld,W=%ld) actual=(T=%ld,H=%ld,W=%ld) D=%ld:\n  ", testIdx, hintT, hintH,
              hintW, actualT, actualH, actualW, headDim);
    int64_t showPos[] = {0, 1, wv, wv + 1, hw, hw + 1, 2 * hw};
    for (int si = 0; si < 7 && showPos[si] < seqLen; si++) {
        int64_t l = showPos[si];
        int64_t t = l / hw, rem = l % hw, i = rem / wv, j = rem % wv;
        LOG_PRINT("l=%ld(t=%ld,i=%ld,j=%ld):", l, t, i, j);
        for (int d = 0; d < std::min(headDim, (int64_t)4); d++)
            LOG_PRINT("%+.2f ", outData[l * headDim + d]);
        LOG_PRINT("|");
    }
    LOG_PRINT("\n");

    // Axis independence: verify X2 (cos/sin) band separation
    // Same (i,j), diff t -> T-band cos/sin differs, H/W-band cos/sin same
    if (halfTBand > 0 && 2 * hw < seqLen) {
        int64_t b1 = hw * headDim, b2 = (2 * hw) * headDim;
        bool tBandDiff = false;
        for (int64_t k = 0; k < halfTBand; k++)
            if (fabsf(x2Data[b1 + k] - x2Data[b2 + k]) > 1e-6f) {
                tBandDiff = true;
                break;
            }
        bool hwSame = true;
        for (int64_t k = halfTBand; k < halfD; k++)
            if (fabsf(x2Data[b1 + k] - x2Data[b2 + k]) > 1e-6f) {
                hwSame = false;
                break;
            }
        LOG_PRINT("[3DTest %d] Axis: (i=0,j=0) fixed, t=1->2: T-band(cs)%s, H/W-band(cs)%s\n", testIdx,
                  tBandDiff ? " DIFFERS OK" : " SAME BUG", hwSame ? " SAME OK" : " DIFFERS BUG");
        if (!tBandDiff || !hwSame)
            allOk = false;
    }
    // Same (i,j), diff t -> T-band cos/sin differs, H/W-band cos/sin same
    if (halfTBand > 0 && 2 * hw < seqLen) {
        int64_t b1 = hw * headDim, b2 = (2 * hw) * headDim;
        bool tBandDiff = false;
        for (int64_t k = 0; k < halfTBand; k++)
            if (fabsf(x2Data[b1 + k] - x2Data[b2 + k]) > 1e-6f) {
                tBandDiff = true;
                break;
            }
        bool hwSame = true;
        for (int64_t k = halfTBand; k < halfD; k++)
            if (fabsf(x2Data[b1 + k] - x2Data[b2 + k]) > 1e-6f) {
                hwSame = false;
                break;
            }
        LOG_PRINT("[3DTest %d] Axis: (i=0,j=0) fixed, t=1->2: T-band(cs)%s, H/W-band(cs)%s\n", testIdx,
                  tBandDiff ? " DIFFERS OK" : " SAME BUG", hwSame ? " SAME OK" : " DIFFERS BUG");
        if (!tBandDiff || !hwSame)
            allOk = false;
    }

    LOG_PRINT("[3DTest %d] %s\n", testIdx, allOk ? "PASSED" : "FAILED");
    aclDestroyTensor(x1);
    aclrtFree(x1Dev);
    aclDestroyTensor(x2);
    aclrtFree(x2Dev);
    aclDestroyTensor(out);
    aclrtFree(outDev);
    if (wsSize > 0)
        aclrtFree(ws);
    return allOk ? 0 : -1;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. %d\n", ret); return ret);

    struct Case {
        int64_t b, s, d;
        const char *desc;
    };
    Case tests[] = {
        // 小规模
        {1, 16, 16, "M=16  D=16"},
        {2, 16, 32, "M=32  D=32"},
        {4, 8, 64, "M=32  D=64"},
        // 中规模
        {1, 64, 128, "M=64  D=128"},
        {4, 32, 128, "M=128 D=128"},
        {8, 16, 256, "M=128 D=256"},
        // 大规模
        {2, 128, 512, "M=256 D=512"},
        {4, 128, 256, "M=512 D=256"},
        {8, 128, 128, "M=1024 D=128"},
    };
    int numTests = sizeof(tests) / sizeof(tests[0]);
    int passed = 0, failed = 0;
    LOG_PRINT("%-25s %-12s %-8s\n", "Case", "Time(ms)", "Result");
    LOG_PRINT("------------------------------------------\n");
    for (int i = 0; i < numTests; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ret = RunTest(tests[i].b, tests[i].s, tests[i].d, i + 1, stream, numTests <= 10);
        auto t1 = std::chrono::high_resolution_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ret) {
            LOG_PRINT("=== Test %d FAILED ===\n", i + 1);
            failed++;
            LOG_PRINT("%-25s %-12.3f %-8s\n", tests[i].desc, totalMs, "FAIL");
        } else {
            passed++;
        }
    }
    LOG_PRINT("------------------------------------------\n");
    LOG_PRINT("=== %d/%d tests PASSED, %d FAILED ===\n", passed, numTests, failed);

    // ===== 3D Axis Independence Tests =====
    LOG_PRINT("\n--- 3D Axis Independence Tests ---\n");
    struct Case3D {
        int64_t T, H, W, D;
        const char *desc;
    };
    Case3D axisTests[] = {
        {2, 4, 4, 128, "(T=2,H=4,W=4,D=128)"},
        {3, 2, 4, 128, "(T=3,H=2,W=4,D=128)"},
        {4, 2, 2, 256, "(T=4,H=2,W=2,D=256)"},
    };
    int num3d = sizeof(axisTests) / sizeof(axisTests[0]);
    int passed3d = 0, failed3d = 0;
    for (int i = 0; i < num3d; i++) {
        ret = RunTest3D(axisTests[i].T, axisTests[i].H, axisTests[i].W, axisTests[i].D, i + 1, stream);
        if (ret)
            failed3d++;
        else
            passed3d++;
    }
    LOG_PRINT("--- 3D Tests: %d/%d PASSED, %d FAILED ---\n", passed3d, num3d, failed3d);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return (failed + failed3d) ? -1 : 0;
}
