/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_scaled_cosine_attention_score.h"

#define CHECK_ACL(expr) \
    do { \
        const auto status = (expr); \
        if (status != ACL_SUCCESS) { \
            std::printf("%s failed: %d\n", #expr, static_cast<int>(status)); \
            return status; \
        } \
    } while (0)

namespace {
int64_t Numel(const std::vector<int64_t> &shape)
{
    int64_t count = 1;
    for (int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

int CreateFloatTensor(const std::vector<float> &data, const std::vector<int64_t> &shape, void **device,
                      aclTensor **tensor)
{
    const size_t bytes = data.size() * sizeof(float);
    CHECK_ACL(aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(*device, bytes, data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *device);
    return *tensor == nullptr ? -1 : ACL_SUCCESS;
}

std::vector<float> Golden(const std::vector<float> &query, const std::vector<float> &key,
                          const std::vector<float> &scale, int64_t batch, int64_t heads, int64_t seqLen,
                          int64_t headDim, float clampMax, float eps)
{
    std::vector<float> out(batch * heads * seqLen * seqLen, 0.0F);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < heads; ++h) {
            const float logitScale = std::exp(std::min(scale[h], clampMax));
            for (int64_t i = 0; i < seqLen; ++i) {
                const int64_t qBase = ((b * heads + h) * seqLen + i) * headDim;
                float qNorm2 = 0.0F;
                for (int64_t d = 0; d < headDim; ++d)
                    qNorm2 += query[qBase + d] * query[qBase + d];
                for (int64_t j = 0; j < seqLen; ++j) {
                    const int64_t kBase = ((b * heads + h) * seqLen + j) * headDim;
                    float kNorm2 = 0.0F;
                    float dot = 0.0F;
                    for (int64_t d = 0; d < headDim; ++d) {
                        kNorm2 += key[kBase + d] * key[kBase + d];
                        dot += query[qBase + d] * key[kBase + d];
                    }
                    const int64_t outOffset = ((b * heads + h) * seqLen + i) * seqLen + j;
                    out[outOffset] = dot * logitScale / (std::sqrt(qNorm2 + eps) * std::sqrt(kNorm2 + eps));
                }
            }
        }
    }
    return out;
}
} // namespace

int main()
{
    constexpr int64_t B = 1;
    constexpr int64_t H = 2;
    constexpr int64_t N = 5;
    constexpr int64_t D = 8;
    constexpr double CLAMP_MAX = 4.6052;
    constexpr double EPS = 1.0e-12;
    const std::vector<int64_t> inputShape{B, H, N, D};
    const std::vector<int64_t> scaleShape{H, 1, 1};
    const std::vector<int64_t> outputShape{B, H, N, N};
    std::vector<float> query(Numel(inputShape));
    std::vector<float> key(query.size());
    std::vector<float> scale{0.0F, 0.5F};
    for (size_t i = 0; i < query.size(); ++i) {
        query[i] = static_cast<float>(static_cast<int32_t>(i % 13U) - 6) * 0.0625F;
        key[i] = static_cast<float>(static_cast<int32_t>((i * 3U) % 17U) - 8) * 0.03125F;
    }
    const auto expected = Golden(query, key, scale, B, H, N, D, CLAMP_MAX, EPS);
    std::vector<float> output(expected.size(), 0.0F);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void *queryDev = nullptr;
    void *keyDev = nullptr;
    void *scaleDev = nullptr;
    void *outDev = nullptr;
    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *scaleTensor = nullptr;
    aclTensor *outTensor = nullptr;
    CHECK_ACL(CreateFloatTensor(query, inputShape, &queryDev, &queryTensor));
    CHECK_ACL(CreateFloatTensor(key, inputShape, &keyDev, &keyTensor));
    CHECK_ACL(CreateFloatTensor(scale, scaleShape, &scaleDev, &scaleTensor));
    CHECK_ACL(CreateFloatTensor(output, outputShape, &outDev, &outTensor));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACL(aclnnScaledCosineAttentionScoreGetWorkspaceSize(queryTensor, keyTensor, scaleTensor, CLAMP_MAX, EPS,
                                                              outTensor, &workspaceSize, &executor));
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACL(aclnnScaledCosineAttentionScore(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(output.data(), output.size() * sizeof(float), outDev, output.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));

    float maxError = 0.0F;
    for (size_t i = 0; i < output.size(); ++i)
        maxError = std::max(maxError, std::fabs(output[i] - expected[i]));
    std::printf("max_abs_error=%g, %s\n", maxError, maxError < 1.0e-4F ? "PASSED" : "FAILED");

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyTensor(queryTensor);
    aclDestroyTensor(keyTensor);
    aclDestroyTensor(scaleTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(queryDev);
    aclrtFree(keyDev);
    aclrtFree(scaleDev);
    aclrtFree(outDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return maxError < 1.0e-4F ? 0 : 1;
}
