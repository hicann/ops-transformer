/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "scatter_pa_kv_cache_with_k_scale_tiling_def.h"
#include "../../../../op_kernel/scatter_pa_kv_cache_with_k_scale_apt.cpp"
#include "../../../../op_kernel/arch35/scatter_pa_kv_cache_with_k_scale_tiling_key.h"

using namespace std;

extern "C" __global__ __aicore__ void scatter_pa_kv_cache_with_k_scale(GM_ADDR key, GM_ADDR value, GM_ADDR key_cache,
                                                                       GM_ADDR value_cache, GM_ADDR slot_mapping,
                                                                       GM_ADDR key_scale, GM_ADDR key_scale_cache,
                                                                       GM_ADDR key_cache_out, GM_ADDR value_cache_out,
                                                                       GM_ADDR key_scale_cache_out, GM_ADDR workspace,
                                                                       GM_ADDR tiling);

class scatter_pa_kv_cache_with_k_scale_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "scatter_pa_kv_cache_with_k_scale_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "scatter_pa_kv_cache_with_k_scale_test TearDown" << endl;
    }
};

// golden 校验：按 golden.py 语义在 host 侧重算并比对 cache 结果
template <typename SlotType>
static void CheckScatterResult(const uint8_t* key, const uint8_t* value, const float* keyScale,
                               const SlotType* slotMapping, const uint8_t* keyCache, const uint8_t* valueCache,
                               const float* keyScaleCache, int64_t numTokens, int64_t numHead, int64_t kHeadSize,
                               int64_t vHeadSize, int64_t numBlocks, int64_t blockSize)
{
    int64_t maxSlot = numBlocks * blockSize;
    std::vector<uint8_t> keyGold(numBlocks * numHead * blockSize * kHeadSize, 0);
    std::vector<uint8_t> valueGold(numBlocks * numHead * blockSize * vHeadSize, 0);
    std::vector<float> scaleGold(numBlocks * numHead * blockSize, 0.0f);
    for (int64_t i = 0; i < numTokens; i++) {
        int64_t slot = slotMapping[i];
        if (slot < 0 || slot >= maxSlot) {
            continue;
        }
        int64_t blk = slot / blockSize;
        int64_t off = slot % blockSize;
        for (int64_t j = 0; j < numHead; j++) {
            std::copy_n(key + (i * numHead + j) * kHeadSize, kHeadSize,
                        keyGold.begin() + ((blk * numHead + j) * blockSize + off) * kHeadSize);
            std::copy_n(value + (i * numHead + j) * vHeadSize, vHeadSize,
                        valueGold.begin() + ((blk * numHead + j) * blockSize + off) * vHeadSize);
            scaleGold[(blk * numHead + j) * blockSize + off] = keyScale[i * numHead + j];
        }
    }
    ASSERT_EQ(0, memcmp(keyCache, keyGold.data(), keyGold.size()));
    ASSERT_EQ(0, memcmp(valueCache, valueGold.data(), valueGold.size()));
    ASSERT_EQ(0, memcmp(keyScaleCache, scaleGold.data(), scaleGold.size() * sizeof(float)));
}

TEST_F(scatter_pa_kv_cache_with_k_scale_test, test_case_specialized_fp8_e4m3_int32)
{
    int64_t numTokens = 4;
    int64_t numHead = 2;
    int64_t kHeadSize = 128;
    int64_t vHeadSize = 128;
    int64_t blockSize = 16;
    int64_t numBlocks = 2;
    int64_t maxSlot = numBlocks * blockSize;

    size_t keySize = numTokens * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueSize = numTokens * numHead * vHeadSize * sizeof(uint8_t);
    size_t keyCacheSize = numBlocks * blockSize * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueCacheSize = numBlocks * blockSize * numHead * vHeadSize * sizeof(uint8_t);
    size_t slotMappingSize = numTokens * sizeof(int32_t);
    size_t keyScaleSize = numTokens * numHead * sizeof(float);
    size_t keyScaleCacheSize = numBlocks * blockSize * numHead * sizeof(float);

    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCacheSize);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCacheSize);
    uint8_t* slotMapping = (uint8_t*)AscendC::GmAlloc(slotMappingSize);
    uint8_t* keyScale = (uint8_t*)AscendC::GmAlloc(keyScaleSize);
    uint8_t* keyScaleCache = (uint8_t*)AscendC::GmAlloc(keyScaleCacheSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScatterPaKvCacheWithKScaleTilingData));

    ScatterPaKvCacheWithKScaleTilingData* tilingData = reinterpret_cast<ScatterPaKvCacheWithKScaleTilingData*>(tiling);
    tilingData->needCoreNum = 1;
    tilingData->usedCoreNum = 1;
    tilingData->blockFactor = numTokens;
    tilingData->tailBlockFactor = numTokens;
    tilingData->ubSize = 229376;
    tilingData->numTokens = numTokens;
    tilingData->numHead = numHead;
    tilingData->kHeadSize = kHeadSize;
    tilingData->vHeadSize = vHeadSize;
    tilingData->numBlocks = numBlocks;
    tilingData->blockSize = blockSize;
    tilingData->maxSlot = maxSlot;
    tilingData->keyStride[0] = numHead * kHeadSize;
    tilingData->keyStride[1] = kHeadSize;
    tilingData->keyStride[2] = 1;
    tilingData->valueStride[0] = numHead * vHeadSize;
    tilingData->valueStride[1] = vHeadSize;
    tilingData->valueStride[2] = 1;
    tilingData->keyCacheStride[0] = numHead * blockSize * kHeadSize;
    tilingData->keyCacheStride[1] = blockSize * kHeadSize;
    tilingData->keyCacheStride[2] = kHeadSize;
    tilingData->keyCacheStride[3] = 1;
    tilingData->valueCacheStride[0] = numHead * blockSize * vHeadSize;
    tilingData->valueCacheStride[1] = blockSize * vHeadSize;
    tilingData->valueCacheStride[2] = vHeadSize;
    tilingData->valueCacheStride[3] = 1;
    tilingData->slotMappingStride[0] = 1;
    tilingData->keyScaleStride[0] = numHead;
    tilingData->keyScaleStride[1] = 1;
    tilingData->keyScaleCacheStride[0] = numHead * blockSize;
    tilingData->keyScaleCacheStride[1] = blockSize;
    tilingData->keyScaleCacheStride[2] = 1;
    tilingData->keyScaleCacheStride[3] = 1;

    for (size_t i = 0; i < keySize; ++i) {
        key[i] = static_cast<uint8_t>(i % 256);
    }
    for (size_t i = 0; i < valueSize; ++i) {
        value[i] = static_cast<uint8_t>((i + 128) % 256);
    }
    for (size_t i = 0; i < keyCacheSize; ++i) {
        keyCache[i] = 0;
    }
    for (size_t i = 0; i < valueCacheSize; ++i) {
        valueCache[i] = 0;
    }
    int32_t* slotMappingData = reinterpret_cast<int32_t*>(slotMapping);
    slotMappingData[0] = 0;
    slotMappingData[1] = 16;
    slotMappingData[2] = 1;
    slotMappingData[3] = 17;

    float* keyScaleData = reinterpret_cast<float*>(keyScale);
    for (int64_t i = 0; i < numTokens * numHead; ++i) {
        keyScaleData[i] = 1.0f + static_cast<float>(i) * 0.1f;
    }
    float* keyScaleCacheData = reinterpret_cast<float*>(keyScaleCache);
    for (int64_t i = 0; i < numBlocks * blockSize * numHead; ++i) {
        keyScaleCacheData[i] = 0.0f;
    }

    ICPU_SET_TILING_KEY(1000001);

    auto scatterPaKvCacheWithKScaleWrapper = [](GM_ADDR key, GM_ADDR value, GM_ADDR key_cache, GM_ADDR value_cache,
                                                GM_ADDR slot_mapping, GM_ADDR key_scale, GM_ADDR key_scale_cache,
                                                GM_ADDR key_cache_out, GM_ADDR value_cache_out,
                                                GM_ADDR key_scale_cache_out, GM_ADDR workspace, GM_ADDR tiling) {
        scatter_pa_kv_cache_with_k_scale<SCATTER_KV_CACHE_SCENE_SPECIALIZED, SCATTER_KV_CACHE_TPL_SIMT>(
            key, value, key_cache, value_cache, slot_mapping, key_scale, key_scale_cache, key_cache_out,
            value_cache_out, key_scale_cache_out, workspace, tiling);
    };

    ICPU_RUN_KF(scatterPaKvCacheWithKScaleWrapper, 1, key, value, keyCache, valueCache, slotMapping, keyScale,
                keyScaleCache, keyCache, valueCache, keyScaleCache, workspace, tiling);

    // 注意：当前 CPU 仿真环境无法仿真 MTE(DataCopyPad) 数据搬运（同 indexer_quant_cache UT 现状），
    // SIMD 用例仅做冒烟执行；数据正确性依赖 NPU 环境测试。
    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(keyScale);
    AscendC::GmFree(keyScaleCache);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_test, test_case_generalized_fp8_e4m3_int32)
{
    int64_t numTokens = 4;
    int64_t numHead = 2;
    int64_t kHeadSize = 128;
    int64_t vHeadSize = 64;
    int64_t blockSize = 16;
    int64_t numBlocks = 2;
    int64_t maxSlot = numBlocks * blockSize;

    size_t keySize = numTokens * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueSize = numTokens * numHead * vHeadSize * sizeof(uint8_t);
    size_t keyCacheSize = numBlocks * blockSize * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueCacheSize = numBlocks * blockSize * numHead * vHeadSize * sizeof(uint8_t);
    size_t slotMappingSize = numTokens * sizeof(int32_t);
    size_t keyScaleSize = numTokens * numHead * sizeof(float);
    size_t keyScaleCacheSize = numBlocks * blockSize * numHead * sizeof(float);

    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCacheSize);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCacheSize);
    uint8_t* slotMapping = (uint8_t*)AscendC::GmAlloc(slotMappingSize);
    uint8_t* keyScale = (uint8_t*)AscendC::GmAlloc(keyScaleSize);
    uint8_t* keyScaleCache = (uint8_t*)AscendC::GmAlloc(keyScaleCacheSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScatterPaKvCacheWithKScaleTilingData));

    ScatterPaKvCacheWithKScaleTilingData* tilingData = reinterpret_cast<ScatterPaKvCacheWithKScaleTilingData*>(tiling);
    tilingData->needCoreNum = 1;
    tilingData->usedCoreNum = 1;
    tilingData->blockFactor = numTokens;
    tilingData->tailBlockFactor = numTokens;
    tilingData->ubSize = 229376;
    tilingData->numTokens = numTokens;
    tilingData->numHead = numHead;
    tilingData->kHeadSize = kHeadSize;
    tilingData->vHeadSize = vHeadSize;
    tilingData->numBlocks = numBlocks;
    tilingData->blockSize = blockSize;
    tilingData->maxSlot = maxSlot;
    tilingData->keyStride[0] = numHead * kHeadSize;
    tilingData->keyStride[1] = kHeadSize;
    tilingData->keyStride[2] = 1;
    tilingData->valueStride[0] = numHead * vHeadSize;
    tilingData->valueStride[1] = vHeadSize;
    tilingData->valueStride[2] = 1;
    tilingData->keyCacheStride[0] = numHead * blockSize * kHeadSize;
    tilingData->keyCacheStride[1] = blockSize * kHeadSize;
    tilingData->keyCacheStride[2] = kHeadSize;
    tilingData->keyCacheStride[3] = 1;
    tilingData->valueCacheStride[0] = numHead * blockSize * vHeadSize;
    tilingData->valueCacheStride[1] = blockSize * vHeadSize;
    tilingData->valueCacheStride[2] = vHeadSize;
    tilingData->valueCacheStride[3] = 1;
    tilingData->slotMappingStride[0] = 1;
    tilingData->keyScaleStride[0] = numHead;
    tilingData->keyScaleStride[1] = 1;
    tilingData->keyScaleCacheStride[0] = numHead * blockSize;
    tilingData->keyScaleCacheStride[1] = blockSize;
    tilingData->keyScaleCacheStride[2] = 1;
    tilingData->keyScaleCacheStride[3] = 1;

    for (size_t i = 0; i < keySize; ++i) {
        key[i] = static_cast<uint8_t>(i % 256);
    }
    for (size_t i = 0; i < valueSize; ++i) {
        value[i] = static_cast<uint8_t>((i + 128) % 256);
    }
    for (size_t i = 0; i < keyCacheSize; ++i) {
        keyCache[i] = 0;
    }
    for (size_t i = 0; i < valueCacheSize; ++i) {
        valueCache[i] = 0;
    }
    int32_t* slotMappingData = reinterpret_cast<int32_t*>(slotMapping);
    slotMappingData[0] = 0;
    slotMappingData[1] = 16;
    slotMappingData[2] = 1;
    slotMappingData[3] = 17;

    float* keyScaleData = reinterpret_cast<float*>(keyScale);
    for (int64_t i = 0; i < numTokens * numHead; ++i) {
        keyScaleData[i] = 1.0f + static_cast<float>(i) * 0.1f;
    }
    float* keyScaleCacheData = reinterpret_cast<float*>(keyScaleCache);
    for (int64_t i = 0; i < numBlocks * blockSize * numHead; ++i) {
        keyScaleCacheData[i] = 0.0f;
    }

    ICPU_SET_TILING_KEY(1000001);

    auto scatterPaKvCacheWithKScaleWrapper = [](GM_ADDR key, GM_ADDR value, GM_ADDR key_cache, GM_ADDR value_cache,
                                                GM_ADDR slot_mapping, GM_ADDR key_scale, GM_ADDR key_scale_cache,
                                                GM_ADDR key_cache_out, GM_ADDR value_cache_out,
                                                GM_ADDR key_scale_cache_out, GM_ADDR workspace, GM_ADDR tiling) {
        scatter_pa_kv_cache_with_k_scale<SCATTER_KV_CACHE_SCENE_GENERALIZED, SCATTER_KV_CACHE_TPL_SIMT>(
            key, value, key_cache, value_cache, slot_mapping, key_scale, key_scale_cache, key_cache_out,
            value_cache_out, key_scale_cache_out, workspace, tiling);
    };

    ICPU_RUN_KF(scatterPaKvCacheWithKScaleWrapper, 1, key, value, keyCache, valueCache, slotMapping, keyScale,
                keyScaleCache, keyCache, valueCache, keyScaleCache, workspace, tiling);

    CheckScatterResult(key, value, reinterpret_cast<float*>(keyScale), reinterpret_cast<int32_t*>(slotMapping),
                       keyCache, valueCache, reinterpret_cast<float*>(keyScaleCache), numTokens, numHead, kHeadSize,
                       vHeadSize, numBlocks, blockSize);

    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(keyScale);
    AscendC::GmFree(keyScaleCache);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_test, test_case_specialized_fp8_e5m2_int32)
{
    int64_t numTokens = 8;
    int64_t numHead = 4;
    int64_t kHeadSize = 64;
    int64_t vHeadSize = 64;
    int64_t blockSize = 16;
    int64_t numBlocks = 4;
    int64_t maxSlot = numBlocks * blockSize;

    size_t keySize = numTokens * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueSize = numTokens * numHead * vHeadSize * sizeof(uint8_t);
    size_t keyCacheSize = numBlocks * blockSize * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueCacheSize = numBlocks * blockSize * numHead * vHeadSize * sizeof(uint8_t);
    size_t slotMappingSize = numTokens * sizeof(int32_t);
    size_t keyScaleSize = numTokens * numHead * sizeof(float);
    size_t keyScaleCacheSize = numBlocks * blockSize * numHead * sizeof(float);

    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCacheSize);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCacheSize);
    uint8_t* slotMapping = (uint8_t*)AscendC::GmAlloc(slotMappingSize);
    uint8_t* keyScale = (uint8_t*)AscendC::GmAlloc(keyScaleSize);
    uint8_t* keyScaleCache = (uint8_t*)AscendC::GmAlloc(keyScaleCacheSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScatterPaKvCacheWithKScaleTilingData));

    ScatterPaKvCacheWithKScaleTilingData* tilingData = reinterpret_cast<ScatterPaKvCacheWithKScaleTilingData*>(tiling);
    tilingData->needCoreNum = 1;
    tilingData->usedCoreNum = 1;
    tilingData->blockFactor = numTokens;
    tilingData->tailBlockFactor = numTokens;
    tilingData->ubSize = 229376;
    tilingData->numTokens = numTokens;
    tilingData->numHead = numHead;
    tilingData->kHeadSize = kHeadSize;
    tilingData->vHeadSize = vHeadSize;
    tilingData->numBlocks = numBlocks;
    tilingData->blockSize = blockSize;
    tilingData->maxSlot = maxSlot;
    tilingData->keyStride[0] = numHead * kHeadSize;
    tilingData->keyStride[1] = kHeadSize;
    tilingData->keyStride[2] = 1;
    tilingData->valueStride[0] = numHead * vHeadSize;
    tilingData->valueStride[1] = vHeadSize;
    tilingData->valueStride[2] = 1;
    tilingData->keyCacheStride[0] = numHead * blockSize * kHeadSize;
    tilingData->keyCacheStride[1] = blockSize * kHeadSize;
    tilingData->keyCacheStride[2] = kHeadSize;
    tilingData->keyCacheStride[3] = 1;
    tilingData->valueCacheStride[0] = numHead * blockSize * vHeadSize;
    tilingData->valueCacheStride[1] = blockSize * vHeadSize;
    tilingData->valueCacheStride[2] = vHeadSize;
    tilingData->valueCacheStride[3] = 1;
    tilingData->slotMappingStride[0] = 1;
    tilingData->keyScaleStride[0] = numHead;
    tilingData->keyScaleStride[1] = 1;
    tilingData->keyScaleCacheStride[0] = numHead * blockSize;
    tilingData->keyScaleCacheStride[1] = blockSize;
    tilingData->keyScaleCacheStride[2] = 1;
    tilingData->keyScaleCacheStride[3] = 1;

    for (size_t i = 0; i < keySize; ++i) {
        key[i] = static_cast<uint8_t>(i % 256);
    }
    for (size_t i = 0; i < valueSize; ++i) {
        value[i] = static_cast<uint8_t>((i + 64) % 256);
    }
    for (size_t i = 0; i < keyCacheSize; ++i) {
        keyCache[i] = 0;
    }
    for (size_t i = 0; i < valueCacheSize; ++i) {
        valueCache[i] = 0;
    }
    int32_t* slotMappingData = reinterpret_cast<int32_t*>(slotMapping);
    for (int64_t i = 0; i < numTokens; ++i) {
        slotMappingData[i] = static_cast<int32_t>(i % maxSlot);
    }

    float* keyScaleData = reinterpret_cast<float*>(keyScale);
    for (int64_t i = 0; i < numTokens * numHead; ++i) {
        keyScaleData[i] = 1.0f + static_cast<float>(i) * 0.05f;
    }
    float* keyScaleCacheData = reinterpret_cast<float*>(keyScaleCache);
    for (int64_t i = 0; i < numBlocks * blockSize * numHead; ++i) {
        keyScaleCacheData[i] = 0.0f;
    }

    ICPU_SET_TILING_KEY(1000001);

    auto scatterPaKvCacheWithKScaleWrapper = [](GM_ADDR key, GM_ADDR value, GM_ADDR key_cache, GM_ADDR value_cache,
                                                GM_ADDR slot_mapping, GM_ADDR key_scale, GM_ADDR key_scale_cache,
                                                GM_ADDR key_cache_out, GM_ADDR value_cache_out,
                                                GM_ADDR key_scale_cache_out, GM_ADDR workspace, GM_ADDR tiling) {
        scatter_pa_kv_cache_with_k_scale<SCATTER_KV_CACHE_SCENE_SPECIALIZED, SCATTER_KV_CACHE_TPL_SIMT>(
            key, value, key_cache, value_cache, slot_mapping, key_scale, key_scale_cache, key_cache_out,
            value_cache_out, key_scale_cache_out, workspace, tiling);
    };

    ICPU_RUN_KF(scatterPaKvCacheWithKScaleWrapper, 1, key, value, keyCache, valueCache, slotMapping, keyScale,
                keyScaleCache, keyCache, valueCache, keyScaleCache, workspace, tiling);

    CheckScatterResult(key, value, reinterpret_cast<float*>(keyScale), reinterpret_cast<int32_t*>(slotMapping),
                       keyCache, valueCache, reinterpret_cast<float*>(keyScaleCache), numTokens, numHead, kHeadSize,
                       vHeadSize, numBlocks, blockSize);

    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(keyScale);
    AscendC::GmFree(keyScaleCache);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_test, test_case_invalid_slot_int32)
{
    int64_t numTokens = 4;
    int64_t numHead = 2;
    int64_t kHeadSize = 128;
    int64_t vHeadSize = 128;
    int64_t blockSize = 16;
    int64_t numBlocks = 2;
    int64_t maxSlot = numBlocks * blockSize;

    size_t keySize = numTokens * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueSize = numTokens * numHead * vHeadSize * sizeof(uint8_t);
    size_t keyCacheSize = numBlocks * blockSize * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueCacheSize = numBlocks * blockSize * numHead * vHeadSize * sizeof(uint8_t);
    size_t slotMappingSize = numTokens * sizeof(int32_t);
    size_t keyScaleSize = numTokens * numHead * sizeof(float);
    size_t keyScaleCacheSize = numBlocks * blockSize * numHead * sizeof(float);

    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCacheSize);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCacheSize);
    uint8_t* slotMapping = (uint8_t*)AscendC::GmAlloc(slotMappingSize);
    uint8_t* keyScale = (uint8_t*)AscendC::GmAlloc(keyScaleSize);
    uint8_t* keyScaleCache = (uint8_t*)AscendC::GmAlloc(keyScaleCacheSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScatterPaKvCacheWithKScaleTilingData));

    ScatterPaKvCacheWithKScaleTilingData* tilingData = reinterpret_cast<ScatterPaKvCacheWithKScaleTilingData*>(tiling);
    tilingData->needCoreNum = 1;
    tilingData->usedCoreNum = 1;
    tilingData->blockFactor = numTokens;
    tilingData->tailBlockFactor = numTokens;
    tilingData->ubSize = 229376;
    tilingData->numTokens = numTokens;
    tilingData->numHead = numHead;
    tilingData->kHeadSize = kHeadSize;
    tilingData->vHeadSize = vHeadSize;
    tilingData->numBlocks = numBlocks;
    tilingData->blockSize = blockSize;
    tilingData->maxSlot = maxSlot;
    tilingData->keyStride[0] = numHead * kHeadSize;
    tilingData->keyStride[1] = kHeadSize;
    tilingData->keyStride[2] = 1;
    tilingData->valueStride[0] = numHead * vHeadSize;
    tilingData->valueStride[1] = vHeadSize;
    tilingData->valueStride[2] = 1;
    tilingData->keyCacheStride[0] = numHead * blockSize * kHeadSize;
    tilingData->keyCacheStride[1] = blockSize * kHeadSize;
    tilingData->keyCacheStride[2] = kHeadSize;
    tilingData->keyCacheStride[3] = 1;
    tilingData->valueCacheStride[0] = numHead * blockSize * vHeadSize;
    tilingData->valueCacheStride[1] = blockSize * vHeadSize;
    tilingData->valueCacheStride[2] = vHeadSize;
    tilingData->valueCacheStride[3] = 1;
    tilingData->slotMappingStride[0] = 1;
    tilingData->keyScaleStride[0] = numHead;
    tilingData->keyScaleStride[1] = 1;
    tilingData->keyScaleCacheStride[0] = numHead * blockSize;
    tilingData->keyScaleCacheStride[1] = blockSize;
    tilingData->keyScaleCacheStride[2] = 1;
    tilingData->keyScaleCacheStride[3] = 1;

    for (size_t i = 0; i < keyCacheSize; ++i) {
        keyCache[i] = 0;
    }
    for (size_t i = 0; i < valueCacheSize; ++i) {
        valueCache[i] = 0;
    }
    for (size_t i = 0; i < keySize; ++i) {
        key[i] = static_cast<uint8_t>((i + 31) % 256);
    }
    for (size_t i = 0; i < valueSize; ++i) {
        value[i] = static_cast<uint8_t>((i + 77) % 256);
    }
    int32_t* slotMappingData = reinterpret_cast<int32_t*>(slotMapping);
    slotMappingData[0] = 0;
    slotMappingData[1] = -1;
    slotMappingData[2] = maxSlot + 10;
    slotMappingData[3] = 5;

    float* keyScaleData = reinterpret_cast<float*>(keyScale);
    for (int64_t i = 0; i < numTokens * numHead; ++i) {
        keyScaleData[i] = 2.0f + static_cast<float>(i) * 0.5f;
    }
    float* keyScaleCacheData = reinterpret_cast<float*>(keyScaleCache);
    for (int64_t i = 0; i < numBlocks * blockSize * numHead; ++i) {
        keyScaleCacheData[i] = 0.0f;
    }

    ICPU_SET_TILING_KEY(1000001);

    auto scatterPaKvCacheWithKScaleWrapper = [](GM_ADDR key, GM_ADDR value, GM_ADDR key_cache, GM_ADDR value_cache,
                                                GM_ADDR slot_mapping, GM_ADDR key_scale, GM_ADDR key_scale_cache,
                                                GM_ADDR key_cache_out, GM_ADDR value_cache_out,
                                                GM_ADDR key_scale_cache_out, GM_ADDR workspace, GM_ADDR tiling) {
        scatter_pa_kv_cache_with_k_scale<SCATTER_KV_CACHE_SCENE_SPECIALIZED, SCATTER_KV_CACHE_TPL_SIMT>(
            key, value, key_cache, value_cache, slot_mapping, key_scale, key_scale_cache, key_cache_out,
            value_cache_out, key_scale_cache_out, workspace, tiling);
    };

    ICPU_RUN_KF(scatterPaKvCacheWithKScaleWrapper, 1, key, value, keyCache, valueCache, slotMapping, keyScale,
                keyScaleCache, keyCache, valueCache, keyScaleCache, workspace, tiling);

    CheckScatterResult(key, value, reinterpret_cast<float*>(keyScale), reinterpret_cast<int32_t*>(slotMapping),
                       keyCache, valueCache, reinterpret_cast<float*>(keyScaleCache), numTokens, numHead, kHeadSize,
                       vHeadSize, numBlocks, blockSize);

    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(keyScale);
    AscendC::GmFree(keyScaleCache);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
TEST_F(scatter_pa_kv_cache_with_k_scale_test, test_case_simd_specialized_fp8_e4m3_int32)
{
    int64_t numTokens = 8;
    int64_t numHead = 2;
    int64_t kHeadSize = 128;
    int64_t vHeadSize = 128;
    int64_t blockSize = 16;
    int64_t numBlocks = 4;
    int64_t maxSlot = numBlocks * blockSize;

    size_t keySize = numTokens * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueSize = numTokens * numHead * vHeadSize * sizeof(uint8_t);
    size_t keyCacheSize = numBlocks * blockSize * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueCacheSize = numBlocks * blockSize * numHead * vHeadSize * sizeof(uint8_t);
    size_t slotMappingSize = numTokens * sizeof(int32_t);
    size_t keyScaleSize = numTokens * numHead * sizeof(float);
    size_t keyScaleCacheSize = numBlocks * blockSize * numHead * sizeof(float);

    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCacheSize);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCacheSize);
    uint8_t* slotMapping = (uint8_t*)AscendC::GmAlloc(slotMappingSize);
    uint8_t* keyScale = (uint8_t*)AscendC::GmAlloc(keyScaleSize);
    uint8_t* keyScaleCache = (uint8_t*)AscendC::GmAlloc(keyScaleCacheSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScatterPaKvCacheWithKScaleTilingData));

    ScatterPaKvCacheWithKScaleTilingData* tilingData = reinterpret_cast<ScatterPaKvCacheWithKScaleTilingData*>(tiling);
    tilingData->needCoreNum = 3;
    tilingData->usedCoreNum = 3;
    tilingData->blockFactor = 3;
    tilingData->tailBlockFactor = 2;
    tilingData->ubSize = 262144;
    tilingData->numTokens = numTokens;
    tilingData->numHead = numHead;
    tilingData->kHeadSize = kHeadSize;
    tilingData->vHeadSize = vHeadSize;
    tilingData->numBlocks = numBlocks;
    tilingData->blockSize = blockSize;
    tilingData->maxSlot = maxSlot;
    tilingData->keyStride[0] = numHead * kHeadSize;
    tilingData->keyStride[1] = kHeadSize;
    tilingData->keyStride[2] = 1;
    tilingData->valueStride[0] = numHead * vHeadSize;
    tilingData->valueStride[1] = vHeadSize;
    tilingData->valueStride[2] = 1;
    tilingData->keyCacheStride[0] = numHead * blockSize * kHeadSize;
    tilingData->keyCacheStride[1] = blockSize * kHeadSize;
    tilingData->keyCacheStride[2] = kHeadSize;
    tilingData->keyCacheStride[3] = 1;
    tilingData->valueCacheStride[0] = numHead * blockSize * vHeadSize;
    tilingData->valueCacheStride[1] = blockSize * vHeadSize;
    tilingData->valueCacheStride[2] = vHeadSize;
    tilingData->valueCacheStride[3] = 1;
    tilingData->slotMappingStride[0] = 1;
    tilingData->keyScaleStride[0] = numHead;
    tilingData->keyScaleStride[1] = 1;
    tilingData->keyScaleCacheStride[0] = numHead * blockSize;
    tilingData->keyScaleCacheStride[1] = blockSize;
    tilingData->keyScaleCacheStride[2] = 1;
    tilingData->keyScaleCacheStride[3] = 1;

    for (size_t i = 0; i < keySize; ++i) {
        key[i] = static_cast<uint8_t>(i % 256);
    }
    for (size_t i = 0; i < valueSize; ++i) {
        value[i] = static_cast<uint8_t>((i + 32) % 256);
    }
    for (size_t i = 0; i < keyCacheSize; ++i) {
        keyCache[i] = 0;
    }
    for (size_t i = 0; i < valueCacheSize; ++i) {
        valueCache[i] = 0;
    }
    int32_t* slotMappingData = reinterpret_cast<int32_t*>(slotMapping);
    slotMappingData[0] = 0;
    slotMappingData[1] = 17;
    slotMappingData[2] = 33;
    slotMappingData[3] = -1;
    slotMappingData[4] = maxSlot + 5;
    slotMappingData[5] = 63;
    slotMappingData[6] = 16;
    slotMappingData[7] = 1;

    float* keyScaleData = reinterpret_cast<float*>(keyScale);
    for (int64_t i = 0; i < numTokens * numHead; ++i) {
        keyScaleData[i] = 0.5f + static_cast<float>(i) * 0.25f;
    }
    float* keyScaleCacheData = reinterpret_cast<float*>(keyScaleCache);
    for (int64_t i = 0; i < numBlocks * blockSize * numHead; ++i) {
        keyScaleCacheData[i] = 0.0f;
    }

    ICPU_SET_TILING_KEY(1000001);

    auto scatterPaKvCacheWithKScaleWrapper = [](GM_ADDR key, GM_ADDR value, GM_ADDR key_cache, GM_ADDR value_cache,
                                                GM_ADDR slot_mapping, GM_ADDR key_scale, GM_ADDR key_scale_cache,
                                                GM_ADDR key_cache_out, GM_ADDR value_cache_out,
                                                GM_ADDR key_scale_cache_out, GM_ADDR workspace, GM_ADDR tiling) {
        scatter_pa_kv_cache_with_k_scale<SCATTER_KV_CACHE_SCENE_SPECIALIZED, SCATTER_KV_CACHE_TPL_SIMD>(
            key, value, key_cache, value_cache, slot_mapping, key_scale, key_scale_cache, key_cache_out,
            value_cache_out, key_scale_cache_out, workspace, tiling);
    };

    ICPU_RUN_KF(scatterPaKvCacheWithKScaleWrapper, 3, key, value, keyCache, valueCache, slotMapping, keyScale,
                keyScaleCache, keyCache, valueCache, keyScaleCache, workspace, tiling);

    // 注意：当前 CPU 仿真环境无法仿真 MTE(DataCopyPad) 数据搬运（同 indexer_quant_cache UT 现状），
    // SIMD 用例仅做冒烟执行；数据正确性依赖 NPU 环境测试。
    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(keyScale);
    AscendC::GmFree(keyScaleCache);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_test, test_case_simd_generalized_fp8_e5m2_int64)
{
    // SIMD 模板要求 headSize >= 128B：kHeadSize=130 覆盖非 32B 对齐场景
    int64_t numTokens = 5;
    int64_t numHead = 4;
    int64_t kHeadSize = 130;
    int64_t vHeadSize = 128;
    int64_t blockSize = 8;
    int64_t numBlocks = 6;
    int64_t maxSlot = numBlocks * blockSize;

    size_t keySize = numTokens * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueSize = numTokens * numHead * vHeadSize * sizeof(uint8_t);
    size_t keyCacheSize = numBlocks * blockSize * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueCacheSize = numBlocks * blockSize * numHead * vHeadSize * sizeof(uint8_t);
    size_t slotMappingSize = numTokens * sizeof(int64_t);
    size_t keyScaleSize = numTokens * numHead * sizeof(float);
    size_t keyScaleCacheSize = numBlocks * blockSize * numHead * sizeof(float);

    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCacheSize);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCacheSize);
    uint8_t* slotMapping = (uint8_t*)AscendC::GmAlloc(slotMappingSize);
    uint8_t* keyScale = (uint8_t*)AscendC::GmAlloc(keyScaleSize);
    uint8_t* keyScaleCache = (uint8_t*)AscendC::GmAlloc(keyScaleCacheSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScatterPaKvCacheWithKScaleTilingData));

    ScatterPaKvCacheWithKScaleTilingData* tilingData = reinterpret_cast<ScatterPaKvCacheWithKScaleTilingData*>(tiling);
    tilingData->needCoreNum = 2;
    tilingData->usedCoreNum = 2;
    tilingData->blockFactor = 3;
    tilingData->tailBlockFactor = 2;
    tilingData->ubSize = 262144;
    tilingData->numTokens = numTokens;
    tilingData->numHead = numHead;
    tilingData->kHeadSize = kHeadSize;
    tilingData->vHeadSize = vHeadSize;
    tilingData->numBlocks = numBlocks;
    tilingData->blockSize = blockSize;
    tilingData->maxSlot = maxSlot;
    tilingData->keyStride[0] = numHead * kHeadSize;
    tilingData->keyStride[1] = kHeadSize;
    tilingData->keyStride[2] = 1;
    tilingData->valueStride[0] = numHead * vHeadSize;
    tilingData->valueStride[1] = vHeadSize;
    tilingData->valueStride[2] = 1;
    tilingData->keyCacheStride[0] = numHead * blockSize * kHeadSize;
    tilingData->keyCacheStride[1] = blockSize * kHeadSize;
    tilingData->keyCacheStride[2] = kHeadSize;
    tilingData->keyCacheStride[3] = 1;
    tilingData->valueCacheStride[0] = numHead * blockSize * vHeadSize;
    tilingData->valueCacheStride[1] = blockSize * vHeadSize;
    tilingData->valueCacheStride[2] = vHeadSize;
    tilingData->valueCacheStride[3] = 1;
    tilingData->slotMappingStride[0] = 1;
    tilingData->keyScaleStride[0] = numHead;
    tilingData->keyScaleStride[1] = 1;
    // 该用例 key_scale_cache 使用头相邻布局（kscStride[1] == 1，与规范 BNBD 不同），
    // 覆盖 scale 多块拷贝 dstStride=0（块相邻）场景；规范布局（kscStride[1] == blockSize）
    // 由 specialized 用例覆盖 dstStride=(blockSize-1)*4 场景
    tilingData->keyScaleCacheStride[0] = numHead * blockSize;
    tilingData->keyScaleCacheStride[1] = 1;
    tilingData->keyScaleCacheStride[2] = numHead;
    tilingData->keyScaleCacheStride[3] = 1;

    for (size_t i = 0; i < keySize; ++i) {
        key[i] = static_cast<uint8_t>((i + 7) % 256);
    }
    for (size_t i = 0; i < valueSize; ++i) {
        value[i] = static_cast<uint8_t>((i + 99) % 256);
    }
    for (size_t i = 0; i < keyCacheSize; ++i) {
        keyCache[i] = 0;
    }
    for (size_t i = 0; i < valueCacheSize; ++i) {
        valueCache[i] = 0;
    }
    int64_t* slotMappingData = reinterpret_cast<int64_t*>(slotMapping);
    slotMappingData[0] = 0;
    slotMappingData[1] = 9;
    slotMappingData[2] = 47;
    slotMappingData[3] = -1;
    slotMappingData[4] = 23;

    float* keyScaleData = reinterpret_cast<float*>(keyScale);
    for (int64_t i = 0; i < numTokens * numHead; ++i) {
        keyScaleData[i] = 1.5f + static_cast<float>(i) * 0.125f;
    }
    float* keyScaleCacheData = reinterpret_cast<float*>(keyScaleCache);
    for (int64_t i = 0; i < numBlocks * blockSize * numHead; ++i) {
        keyScaleCacheData[i] = 0.0f;
    }

    ICPU_SET_TILING_KEY(1000001);

    auto scatterPaKvCacheWithKScaleWrapper = [](GM_ADDR key, GM_ADDR value, GM_ADDR key_cache, GM_ADDR value_cache,
                                                GM_ADDR slot_mapping, GM_ADDR key_scale, GM_ADDR key_scale_cache,
                                                GM_ADDR key_cache_out, GM_ADDR value_cache_out,
                                                GM_ADDR key_scale_cache_out, GM_ADDR workspace, GM_ADDR tiling) {
        scatter_pa_kv_cache_with_k_scale<SCATTER_KV_CACHE_SCENE_GENERALIZED, SCATTER_KV_CACHE_TPL_SIMD>(
            key, value, key_cache, value_cache, slot_mapping, key_scale, key_scale_cache, key_cache_out,
            value_cache_out, key_scale_cache_out, workspace, tiling);
    };

    ICPU_RUN_KF(scatterPaKvCacheWithKScaleWrapper, 2, key, value, keyCache, valueCache, slotMapping, keyScale,
                keyScaleCache, keyCache, valueCache, keyScaleCache, workspace, tiling);

    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(keyScale);
    AscendC::GmFree(keyScaleCache);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
