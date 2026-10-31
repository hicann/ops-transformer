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
 * \file test_aclnn_sliding_tile_attention.cpp
 * \brief aclnn two-phase call example for SlidingTileAttention.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_sliding_tile_attention.h"

#define CHECK_RET(cond, expr) \
    do { \
        if (!(cond)) { \
            expr; \
        } \
    } while (0)

static void PrintAclError(const char* message, int ret)
{
    std::cerr << message << ": " << ret << std::endl;
    const char* recent = aclGetRecentErrMsg();
    if (recent != nullptr) {
        std::cerr << recent << std::endl;
    }
}

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

static int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}

template <typename T>
static int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                           aclDataType dataType, aclTensor** tensor)
{
    const auto size = static_cast<size_t>(GetShapeSize(shape) * static_cast<int64_t>(sizeof(T)));
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, return -1);
    return 0;
}

// One aclnn two-phase call: the three inputs are copied to the device, the window is broadcast to every head, and the
// output is read back into outHost. Every device resource of the call is released before returning.
static int RunStaCase(aclrtStream stream, const std::vector<int64_t>& qkvShape, const char* seqShape,
                      const std::vector<int64_t>& windowHost, int64_t textLength, bool hasText,
                      const std::vector<aclFloat16>& qHost, const std::vector<aclFloat16>& kHost,
                      const std::vector<aclFloat16>& vHost, std::vector<aclFloat16>& outHost)
{
    void* qDev = nullptr;
    void* kDev = nullptr;
    void* vDev = nullptr;
    void* outDev = nullptr;
    aclTensor* q = nullptr;
    aclTensor* k = nullptr;
    aclTensor* v = nullptr;
    aclTensor* out = nullptr;
    void* workspace = nullptr;
    aclIntArray* window = nullptr;
    int result = -1;

    do {
        auto ret = CreateAclTensor(qHost, qkvShape, &qDev, ACL_FLOAT16, &q);
        CHECK_RET(ret == ACL_SUCCESS, PrintAclError("Create query tensor failed", ret); break);
        ret = CreateAclTensor(kHost, qkvShape, &kDev, ACL_FLOAT16, &k);
        CHECK_RET(ret == ACL_SUCCESS, PrintAclError("Create key tensor failed", ret); break);
        ret = CreateAclTensor(vHost, qkvShape, &vDev, ACL_FLOAT16, &v);
        CHECK_RET(ret == ACL_SUCCESS, PrintAclError("Create value tensor failed", ret); break);
        ret = CreateAclTensor(outHost, qkvShape, &outDev, ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, PrintAclError("Create output tensor failed", ret); break);

        window = aclCreateIntArray(windowHost.data(), windowHost.size());
        CHECK_RET(window != nullptr, std::cerr << "Create window array failed" << std::endl; break);
        const aclIntArray* windows[] = {window};

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnSlidingTileAttentionGetWorkspaceSize(q, k, v, out, windows, 1, textLength, hasText, seqShape,
                                                        &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, PrintAclError("GetWorkspaceSize failed", ret); break);
        std::cout << "workspaceSize: " << workspaceSize << std::endl;

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, PrintAclError("workspace malloc failed", ret); break);
        }
        ret = aclnnSlidingTileAttention(workspace, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, PrintAclError("SlidingTileAttention failed", ret); break);
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, PrintAclError("sync failed", ret); break);

        const size_t outputBytes = outHost.size() * sizeof(aclFloat16);
        ret = aclrtMemcpy(outHost.data(), outputBytes, outDev, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, PrintAclError("copy output failed", ret); break);
        result = 0;
    } while (false);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyIntArray(window);
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(out);
    aclrtFree(qDev);
    aclrtFree(kDev);
    aclrtFree(vDev);
    aclrtFree(outDev);
    return result;
}

// Zero queries and keys make the attention weights uniform over the keys the window selects, so every output row is
// the mean of the values of its own window. With a value that only depends on the head dim, that mean is the value
// itself whatever the window selects: this checks the call sequence, the output shape and the normalization.
static int UniformAttentionCase(aclrtStream stream)
{
    const int64_t batch = 1;
    const int64_t heads = 1;
    const int64_t headDim = 16;
    const int64_t canvasT = 18;
    const int64_t canvasH = 48;
    const int64_t canvasW = 80;
    const int64_t seqLen = canvasT * canvasH * canvasW;
    const std::vector<int64_t> qkvShape = {batch, heads, seqLen, headDim};
    const int64_t qkvSize = GetShapeSize(qkvShape);

    std::vector<aclFloat16> qHost(static_cast<size_t>(qkvSize), aclFloatToFloat16(0.0F));
    std::vector<aclFloat16> kHost(static_cast<size_t>(qkvSize), aclFloatToFloat16(0.0F));
    std::vector<aclFloat16> vHost(static_cast<size_t>(qkvSize));
    std::vector<aclFloat16> outHost(static_cast<size_t>(qkvSize), aclFloatToFloat16(0.0F));
    for (int64_t index = 0; index < qkvSize; ++index) {
        const int64_t dim = index % headDim;
        const float value = 0.25F + static_cast<float>(dim) * 0.001F;
        vHost[static_cast<size_t>(index)] = aclFloatToFloat16(value);
    }

    const std::vector<int64_t> windowHost = {1, 1, 1};
    auto ret = RunStaCase(stream, qkvShape, "18x48x80", windowHost, 0, false, qHost, kHost, vHost, outHost);
    CHECK_RET(ret == 0, return ret);

    std::cout << "output[0:8]:";
    for (size_t index = 0; index < 8U; ++index) {
        std::cout << ' ' << aclFloat16ToFloat(outHost[index]);
    }
    std::cout << std::endl;
    float maxAbsError = 0.0F;
    for (int64_t index = 0; index < qkvSize; ++index) {
        const float actual = aclFloat16ToFloat(outHost[static_cast<size_t>(index)]);
        CHECK_RET(std::isfinite(actual), std::cerr << "non-finite output" << std::endl; return -1);
        const float expected = aclFloat16ToFloat(vHost[static_cast<size_t>(index)]);
        maxAbsError = std::max(maxAbsError, std::fabs(actual - expected));
    }
    std::cout << "uniform-attention max_abs_error: " << maxAbsError << std::endl;
    CHECK_RET(maxAbsError <= 5e-3F, std::cerr << "uniform-attention accuracy check failed" << std::endl; return -1);
    return 0;
}

// The window has to decide which keys a row sees, so the values have to differ between tiles. The canvas 6x8x16 is two
// tiles of 384 tokens; with a window of one tile each row may only see its own tile, so the rows of the first tile
// must come out as the value of that tile and the rows of the second tile as the value of the second one. Selecting
// the wrong tile, or falling back to attending everything, gives the mean of the two values instead.
static int PerTileWindowCase(aclrtStream stream)
{
    const int64_t batch = 1;
    const int64_t heads = 1;
    const int64_t headDim = 16;
    const int64_t tileTokens = 384;
    const int64_t tiles = 2;
    const int64_t seqLen = tiles * tileTokens;
    const std::vector<int64_t> qkvShape = {batch, heads, seqLen, headDim};
    const int64_t qkvSize = GetShapeSize(qkvShape);
    const float tileValue[] = {0.5F, 2.0F};

    std::vector<aclFloat16> qHost(static_cast<size_t>(qkvSize), aclFloatToFloat16(0.0F));
    std::vector<aclFloat16> kHost(static_cast<size_t>(qkvSize), aclFloatToFloat16(0.0F));
    std::vector<aclFloat16> vHost(static_cast<size_t>(qkvSize));
    std::vector<aclFloat16> outHost(static_cast<size_t>(qkvSize), aclFloatToFloat16(0.0F));
    for (int64_t index = 0; index < qkvSize; ++index) {
        const int64_t tile = index / (tileTokens * headDim);
        vHost[static_cast<size_t>(index)] = aclFloatToFloat16(tileValue[tile]);
    }

    const std::vector<int64_t> windowHost = {1, 1, 1};
    auto ret = RunStaCase(stream, qkvShape, "6x8x16", windowHost, 0, false, qHost, kHost, vHost, outHost);
    CHECK_RET(ret == 0, return ret);

    float maxAbsError = 0.0F;
    for (int64_t index = 0; index < qkvSize; ++index) {
        const int64_t tile = index / (tileTokens * headDim);
        const float actual = aclFloat16ToFloat(outHost[static_cast<size_t>(index)]);
        CHECK_RET(std::isfinite(actual), std::cerr << "non-finite output" << std::endl; return -1);
        maxAbsError = std::max(maxAbsError, std::fabs(actual - tileValue[tile]));
    }
    std::cout << "per-tile-window out[tile0]: " << aclFloat16ToFloat(outHost[0])
              << " out[tile1]: " << aclFloat16ToFloat(outHost[static_cast<size_t>(tileTokens * headDim)])
              << " expected: " << tileValue[0] << " " << tileValue[1] << std::endl;
    std::cout << "per-tile-window max_abs_error: " << maxAbsError << std::endl;
    CHECK_RET(maxAbsError <= 5e-3F, std::cerr << "per-tile-window accuracy check failed" << std::endl; return -1);
    return 0;
}

// Queries just above the smallest normal FLOAT16 with keys and values near the largest one: the scores of the two
// halves of the keys differ only in the lowest bits of the queries, so an implementation that rescales the queries in
// FLOAT16 before Q*K loses that difference and returns a uniform attention. The two key halves cancel in the value,
// so the correct output is 32752*tanh(score) and the lossy one is zero.
static int SubnormalScaleCase(aclrtStream stream)
{
    const int64_t batch = 1;
    const int64_t heads = 1;
    const int64_t headDim = 16;
    const int64_t seqLen = 384;
    const std::vector<int64_t> qkvShape = {batch, heads, seqLen, headDim};
    const int64_t qkvSize = GetShapeSize(qkvShape);
    const float queryHigh = 1025.0F * std::ldexp(1.0F, -24); // one step above the smallest normal FLOAT16
    const float queryLow = -1024.0F * std::ldexp(1.0F, -24); // the smallest normal FLOAT16, negative
    const float keyValue = 65504.0F;
    const float valueValue = 32752.0F;

    std::vector<aclFloat16> qHost(static_cast<size_t>(qkvSize));
    std::vector<aclFloat16> kHost(static_cast<size_t>(qkvSize));
    std::vector<aclFloat16> vHost(static_cast<size_t>(qkvSize));
    std::vector<aclFloat16> outHost(static_cast<size_t>(qkvSize), aclFloatToFloat16(0.0F));
    for (int64_t row = 0; row < seqLen; ++row) {
        for (int64_t dim = 0; dim < headDim; ++dim) {
            const size_t index = static_cast<size_t>(row * headDim + dim);
            qHost[index] = aclFloatToFloat16(dim < headDim / 2 ? queryHigh : queryLow);
            kHost[index] = aclFloatToFloat16(row < seqLen / 2 ? keyValue : -keyValue);
            vHost[index] = aclFloatToFloat16(row < seqLen / 2 ? valueValue : -valueValue);
        }
    }
    // Every query row attends all 384 keys, so the score of a key is sum(q) * key * 1/sqrt(headDim).
    const double score = 8.0 * std::ldexp(1.0, -24) * static_cast<double>(keyValue) * 0.25;
    const double expected = static_cast<double>(valueValue) * std::tanh(score);

    const std::vector<int64_t> windowHost = {1, 1, 1};
    auto ret = RunStaCase(stream, qkvShape, "6x8x8", windowHost, 0, false, qHost, kHost, vHost, outHost);
    CHECK_RET(ret == 0, return ret);

    double maxRelError = 0.0;
    for (int64_t index = 0; index < qkvSize; ++index) {
        const double actual = aclFloat16ToFloat(outHost[static_cast<size_t>(index)]);
        CHECK_RET(std::isfinite(actual), std::cerr << "non-finite output" << std::endl; return -1);
        maxRelError = std::max(maxRelError, std::fabs(actual - expected) / expected);
    }
    std::cout << "subnormal-scale out[0]: " << aclFloat16ToFloat(outHost[0]) << " expected: " << expected
              << " max_rel_error: " << maxRelError << std::endl;
    // The probabilities reach the cube as FLOAT16, which costs about one percent on a score difference this small.
    CHECK_RET(maxRelError <= 0.05, std::cerr << "subnormal-scale accuracy check failed" << std::endl; return -1);
    return 0;
}

int main()
{
    // The DevEnv container exposes physical NPU 5 as logical device 0.
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cerr << "Init failed: " << ret << std::endl; return ret);

    ret = UniformAttentionCase(stream);
    CHECK_RET(ret == 0, return ret);
    ret = PerTileWindowCase(stream);
    CHECK_RET(ret == 0, return ret);
    ret = SubnormalScaleCase(stream);
    CHECK_RET(ret == 0, return ret);
    std::cout << "aclnnSlidingTileAttention example finished successfully." << std::endl;

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
