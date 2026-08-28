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
#include <cstring>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "scaled_cosine_attention_score_tiling_def.h"

extern "C" __global__ __aicore__ void scaled_cosine_attention_score(GM_ADDR query, GM_ADDR key, GM_ADDR scale,
                                                                    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);

namespace {
class GmBuffer {
public:
    explicit GmBuffer(size_t bytes)
        : ptr_(reinterpret_cast<uint8_t *>(AscendC::GmAlloc(bytes)))
    {
        if (ptr_ != nullptr)
            std::memset(ptr_, 0, bytes);
    }
    ~GmBuffer()
    {
        if (ptr_ != nullptr)
            AscendC::GmFree(ptr_);
    }
    uint8_t *Get() const
    {
        return ptr_;
    }

private:
    uint8_t *ptr_ = nullptr;
};

template <typename T>
float ToFloat(T value)
{
    if constexpr (std::is_same_v<T, bfloat16_t>)
        return AscendC::ToFloat(value);
    return static_cast<float>(value);
}

template <typename T>
T FromFloat(float value)
{
    if constexpr (std::is_same_v<T, bfloat16_t>)
        return AscendC::ToBfloat16(value);
    return static_cast<T>(value);
}

template <typename T>
uint32_t AlignHeadDim(uint32_t dim)
{
    return ((dim * sizeof(T) + 31U) / 32U * 32U) / sizeof(T);
}

template <typename T>
std::vector<float> Golden(const std::vector<T> &query, const std::vector<T> &key, const std::vector<float> &scale,
                          uint32_t batch, uint32_t heads, uint32_t seqLen, uint32_t dim, float clampMax, float eps)
{
    std::vector<float> out(static_cast<size_t>(batch) * heads * seqLen * seqLen);
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t h = 0; h < heads; ++h) {
            const float factor = std::exp(std::min(scale[h], clampMax));
            for (uint32_t i = 0; i < seqLen; ++i) {
                const size_t qBase = ((static_cast<size_t>(b) * heads + h) * seqLen + i) * dim;
                float qNorm2 = 0.0F;
                for (uint32_t d = 0; d < dim; ++d)
                    qNorm2 += ToFloat(query[qBase + d]) * ToFloat(query[qBase + d]);
                for (uint32_t j = 0; j < seqLen; ++j) {
                    const size_t kBase = ((static_cast<size_t>(b) * heads + h) * seqLen + j) * dim;
                    float kNorm2 = 0.0F;
                    float dot = 0.0F;
                    for (uint32_t d = 0; d < dim; ++d) {
                        const float qValue = ToFloat(query[qBase + d]);
                        const float kValue = ToFloat(key[kBase + d]);
                        dot += qValue * kValue;
                        kNorm2 += kValue * kValue;
                    }
                    const size_t offset = ((static_cast<size_t>(b) * heads + h) * seqLen + i) * seqLen + j;
                    out[offset] = dot * factor / (std::sqrt(qNorm2 + eps) * std::sqrt(kNorm2 + eps));
                }
            }
        }
    }
    return out;
}

template <typename T>
float Tolerance()
{
    if constexpr (std::is_same_v<T, float>)
        return 2.0e-5F;
    if constexpr (std::is_same_v<T, half>)
        return 2.0e-3F;
    return 8.0e-3F;
}

template <typename T>
void RunCase(uint64_t tilingKey, uint32_t batch, uint32_t heads, uint32_t seqLen, uint32_t dim, uint32_t coreNum,
             float clampMax = 4.6052F, float eps = 1.0e-12F)
{
    const size_t inputCount = static_cast<size_t>(batch) * heads * seqLen * dim;
    std::vector<T> query(inputCount);
    std::vector<T> key(inputCount);
    std::vector<float> scale(heads);
    for (size_t i = 0; i < inputCount; ++i) {
        query[i] = FromFloat<T>(static_cast<float>(static_cast<int32_t>(i % 13U) - 6) * 0.0625F);
        key[i] = FromFloat<T>(static_cast<float>(static_cast<int32_t>((i * 3U) % 17U) - 8) * 0.03125F);
    }
    for (uint32_t h = 0; h < heads; ++h)
        scale[h] = h == 0 ? 0.0F : 8.0F;
    const auto expected = Golden(query, key, scale, batch, heads, seqLen, dim, clampMax, eps);
    const uint32_t usedCoreNum = std::min<uint32_t>(coreNum, batch * heads * seqLen);

    GmBuffer queryGm(query.size() * sizeof(T));
    GmBuffer keyGm(key.size() * sizeof(T));
    GmBuffer scaleGm(scale.size() * sizeof(float));
    GmBuffer outputGm(expected.size() * sizeof(T));
    GmBuffer workspaceGm(32);
    GmBuffer tilingGm(sizeof(optiling::ScaledCosineAttentionScoreTilingData));
    ASSERT_NE(queryGm.Get(), nullptr);
    ASSERT_NE(keyGm.Get(), nullptr);
    ASSERT_NE(scaleGm.Get(), nullptr);
    ASSERT_NE(outputGm.Get(), nullptr);
    ASSERT_NE(tilingGm.Get(), nullptr);
    std::memcpy(queryGm.Get(), query.data(), query.size() * sizeof(T));
    std::memcpy(keyGm.Get(), key.data(), key.size() * sizeof(T));
    std::memcpy(scaleGm.Get(), scale.data(), scale.size() * sizeof(float));

    auto *tiling = reinterpret_cast<optiling::ScaledCosineAttentionScoreTilingData *>(tilingGm.Get());
    std::memset(tiling, 0, sizeof(*tiling));
    tiling->batch = batch;
    tiling->heads = heads;
    tiling->seqLen = seqLen;
    tiling->headDim = dim;
    tiling->alignedHeadDim = AlignHeadDim<T>(dim);
    tiling->keyTileRows = std::min<uint32_t>(3, seqLen);
    tiling->usedCoreNum = usedCoreNum;
    tiling->totalQueryRows = static_cast<uint64_t>(batch) * heads * seqLen;
    tiling->clampMax = clampMax;
    tiling->eps = eps;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(scaled_cosine_attention_score, usedCoreNum, queryGm.Get(), keyGm.Get(), scaleGm.Get(), outputGm.Get(),
                workspaceGm.Get(), tilingGm.Get());

    const T *actual = reinterpret_cast<const T *>(outputGm.Get());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(ToFloat(actual[i]), ToFloat(FromFloat<T>(expected[i])), Tolerance<T>())
            << "mismatch at output index " << i;
    }
}
} // namespace

TEST(ScaledCosineAttentionScoreKernel, Fp32TailAndMulticore)
{
    RunCase<float>(optiling::SCAS_TILING_KEY_FP32, 1, 2, 5, 7, 4);
}

TEST(ScaledCosineAttentionScoreKernel, Fp16UnalignedHeadDim)
{
    RunCase<half>(optiling::SCAS_TILING_KEY_FP16, 1, 2, 5, 7, 3);
}

TEST(ScaledCosineAttentionScoreKernel, Bf16)
{
    RunCase<bfloat16_t>(optiling::SCAS_TILING_KEY_BF16, 1, 1, 4, 8, 2);
}
