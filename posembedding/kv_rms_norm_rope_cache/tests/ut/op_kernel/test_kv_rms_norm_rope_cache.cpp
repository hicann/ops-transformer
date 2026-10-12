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
 * \file test_kv_rms_norm_rope_cache.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "kv_rms_norm_rope_cache_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

#ifndef DTYPE_KV
#define DTYPE_KV half
#endif
#ifndef DTYPE_K_CACHE
#define DTYPE_K_CACHE int8_t
#endif
#ifndef DTYPE_CKV_CACHE
#define DTYPE_CKV_CACHE int8_t
#endif

using namespace std;

extern "C" __global__ __aicore__ void kv_rms_norm_rope_cache(
    GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR k_cache, GM_ADDR ckv_cache,
    GM_ADDR k_rope_scale, GM_ADDR c_kv_scale, GM_ADDR k_rope_offset, GM_ADDR c_kv_offset, GM_ADDR v,
    GM_ADDR k_cache_out, GM_ADDR c_kv_offset_out, GM_ADDR k_rope, GM_ADDR c_kv, GM_ADDR workspace, GM_ADDR tiling);

class kv_rms_norm_rope_cache_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "kv_rms_norm_rope_cache_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "kv_rms_norm_rope_cache_test TearDown\n" << endl;
    }
};

TEST_F(kv_rms_norm_rope_cache_test, test_case_5011)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 64 * 1 * 1 * 576 * sizeof(half);
    size_t gammaSize = 512 * sizeof(half);
    size_t indexSize = 64 * sizeof(uint64_t);
    size_t cosSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t sinSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t kCacheSize = 576 * 128 * 1 * 64 * sizeof(half);
    size_t ckvCacheSize = 576 * 128 * 1 * 512 * sizeof(half);
    size_t kRopeScaleSize = 64 * sizeof(float);
    size_t ckvScaleSize = 512 * sizeof(float);
    size_t kRopeSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t ckvSize = 64 * 1 * 1 * 512 * sizeof(half);
    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 32;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* kRopeScale = (uint8_t*)AscendC::GmAlloc(kRopeScaleSize);
    uint8_t* ckvScale = (uint8_t*)AscendC::GmAlloc(ckvScaleSize);
    uint8_t* kRope = (uint8_t*)AscendC::GmAlloc(kRopeSize);
    uint8_t* ckv = (uint8_t*)AscendC::GmAlloc(ckvSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 32;
    tilingDatafromBin->rowsPerBlock = 0;
    tilingDatafromBin->cacheLength = 1;
    tilingDatafromBin->batchSize = 64;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 1;
    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->blockFactor = 2;
    tilingDatafromBin->ubFactor = 16;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 512.0f;
    tilingDatafromBin->isOutputKv = true;
    tilingDatafromBin->isKQuant = 1;
    tilingDatafromBin->isVQuant = 1;
    tilingDatafromBin->methodMode = 0;

    ICPU_SET_TILING_KEY(5011);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
                nullptr, nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(kRopeScale);
    AscendC::GmFree(ckvScale);
    AscendC::GmFree(kRope);
    AscendC::GmFree(ckv);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(kv_rms_norm_rope_cache_test, test_case_5010)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 64 * 1 * 1 * 576 * sizeof(half);
    size_t gammaSize = 512 * sizeof(half);
    size_t indexSize = 64 * sizeof(uint64_t);
    size_t cosSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t sinSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t kCacheSize = 576 * 128 * 1 * 64 * sizeof(half);
    size_t ckvCacheSize = 576 * 128 * 1 * 512 * sizeof(half);
    size_t kRopeScaleSize = 64 * sizeof(float);
    size_t ckvScaleSize = 512 * sizeof(float);
    size_t kRopeSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t ckvSize = 64 * 1 * 1 * 512 * sizeof(half);
    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 32;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* kRopeScale = (uint8_t*)AscendC::GmAlloc(kRopeScaleSize);
    uint8_t* ckvScale = (uint8_t*)AscendC::GmAlloc(ckvScaleSize);
    uint8_t* kRope = (uint8_t*)AscendC::GmAlloc(kRopeSize);
    uint8_t* ckv = (uint8_t*)AscendC::GmAlloc(ckvSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 32;
    tilingDatafromBin->rowsPerBlock = 0;
    tilingDatafromBin->cacheLength = 1;
    tilingDatafromBin->batchSize = 64;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 1;
    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->blockFactor = 2;
    tilingDatafromBin->ubFactor = 16;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 512.0f;
    tilingDatafromBin->isOutputKv = false;
    tilingDatafromBin->isKQuant = 1;
    tilingDatafromBin->isVQuant = 1;
    tilingDatafromBin->methodMode = 0;

    ICPU_SET_TILING_KEY(5010);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
                nullptr, nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(kRopeScale);
    AscendC::GmFree(ckvScale);
    AscendC::GmFree(kRope);
    AscendC::GmFree(ckv);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(kv_rms_norm_rope_cache_test, test_case_5001)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 64 * 1 * 1 * 576 * sizeof(half);
    size_t gammaSize = 512 * sizeof(half);
    size_t indexSize = 64 * sizeof(uint64_t);
    size_t cosSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t sinSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t kCacheSize = 576 * 128 * 1 * 64 * sizeof(half);
    size_t ckvCacheSize = 576 * 128 * 1 * 512 * sizeof(half);
    size_t kRopeSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t ckvSize = 64 * 1 * 1 * 512 * sizeof(half);
    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 32;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* kRope = (uint8_t*)AscendC::GmAlloc(kRopeSize);
    uint8_t* ckv = (uint8_t*)AscendC::GmAlloc(ckvSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 32;
    tilingDatafromBin->rowsPerBlock = 0;
    tilingDatafromBin->cacheLength = 1;
    tilingDatafromBin->batchSize = 64;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 1;
    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->blockFactor = 2;
    tilingDatafromBin->ubFactor = 16;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 512.0f;
    tilingDatafromBin->isOutputKv = true;
    tilingDatafromBin->isKQuant = 0;
    tilingDatafromBin->isVQuant = 0;
    tilingDatafromBin->methodMode = 0;

    ICPU_SET_TILING_KEY(5001);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(kRope);
    AscendC::GmFree(ckv);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(kv_rms_norm_rope_cache_test, test_case_5000)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 38 * 1 * 3809 * 576 * sizeof(half);
    size_t gammaSize = 512 * sizeof(half);
    size_t indexSize = 228 * sizeof(uint64_t);
    size_t cosSize = 38 * 1 * 3809 * 64 * sizeof(half);
    size_t sinSize = 38 * 1 * 3809 * 64 * sizeof(half);
    size_t kCacheSize = 230 * 745 * 1 * 64 * sizeof(half);
    size_t ckvCacheSize = 230 * 745 * 1 * 512 * sizeof(half);
    size_t kRopeSize = 38 * 1 * 3809 * 64 * sizeof(half);
    size_t ckvSize = 38 * 1 * 3809 * 512 * sizeof(half);
    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 48;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* kRope = (uint8_t*)AscendC::GmAlloc(kRopeSize);
    uint8_t* ckv = (uint8_t*)AscendC::GmAlloc(ckvSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 48;
    tilingDatafromBin->rowsPerBlock = 0;
    tilingDatafromBin->cacheLength = 1;
    tilingDatafromBin->batchSize = 38;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 3809;
    tilingDatafromBin->blockSize = 745;
    tilingDatafromBin->blockFactor = 3016;
    tilingDatafromBin->ubFactor = 16;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 512.0f;
    tilingDatafromBin->isOutputKv = false;
    tilingDatafromBin->isKQuant = 0;
    tilingDatafromBin->isVQuant = 0;
    tilingDatafromBin->methodMode = 0;

    ICPU_SET_TILING_KEY(5000);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(kRope);
    AscendC::GmFree(ckv);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(kv_rms_norm_rope_cache_test, test_case_4011)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 64 * 1 * 1 * 576 * sizeof(half);
    size_t gammaSize = 512 * sizeof(half);
    size_t indexSize = 64 * sizeof(uint64_t);
    size_t cosSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t sinSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t kCacheSize = 576 * 128 * 1 * 64 * sizeof(half);
    size_t ckvCacheSize = 576 * 128 * 1 * 512 * sizeof(half);
    size_t kRopeScaleSize = 64 * sizeof(float);
    size_t ckvScaleSize = 512 * sizeof(float);
    size_t kRopeSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t ckvSize = 64 * 1 * 1 * 512 * sizeof(half);
    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 32;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* kRopeScale = (uint8_t*)AscendC::GmAlloc(kRopeScaleSize);
    uint8_t* ckvScale = (uint8_t*)AscendC::GmAlloc(ckvScaleSize);
    uint8_t* kRope = (uint8_t*)AscendC::GmAlloc(kRopeSize);
    uint8_t* ckv = (uint8_t*)AscendC::GmAlloc(ckvSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 32;
    tilingDatafromBin->rowsPerBlock = 0;
    tilingDatafromBin->cacheLength = 1;
    tilingDatafromBin->batchSize = 64;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 1;
    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->blockFactor = 2;
    tilingDatafromBin->ubFactor = 16;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 512.0f;
    tilingDatafromBin->isOutputKv = false;
    tilingDatafromBin->isKQuant = 1;
    tilingDatafromBin->isVQuant = 1;
    tilingDatafromBin->methodMode = 0;

    ICPU_SET_TILING_KEY(4011);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
                nullptr, nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(kRopeScale);
    AscendC::GmFree(ckvScale);
    AscendC::GmFree(kRope);
    AscendC::GmFree(ckv);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(kv_rms_norm_rope_cache_test, test_case_4010)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 64 * 1 * 1 * 576 * sizeof(half);
    size_t gammaSize = 512 * sizeof(half);
    size_t indexSize = 64 * sizeof(uint64_t);
    size_t cosSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t sinSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t kCacheSize = 576 * 128 * 1 * 64 * sizeof(half);
    size_t ckvCacheSize = 576 * 128 * 1 * 512 * sizeof(half);
    size_t kRopeScaleSize = 64 * sizeof(float);
    size_t ckvScaleSize = 512 * sizeof(float);
    size_t kRopeSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t ckvSize = 64 * 1 * 1 * 512 * sizeof(half);
    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 32;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* kRopeScale = (uint8_t*)AscendC::GmAlloc(kRopeScaleSize);
    uint8_t* ckvScale = (uint8_t*)AscendC::GmAlloc(ckvScaleSize);
    uint8_t* kRope = (uint8_t*)AscendC::GmAlloc(kRopeSize);
    uint8_t* ckv = (uint8_t*)AscendC::GmAlloc(ckvSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 32;
    tilingDatafromBin->rowsPerBlock = 0;
    tilingDatafromBin->cacheLength = 1;
    tilingDatafromBin->batchSize = 64;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 1;
    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->blockFactor = 2;
    tilingDatafromBin->ubFactor = 16;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 512.0f;
    tilingDatafromBin->isOutputKv = false;
    tilingDatafromBin->isKQuant = 1;
    tilingDatafromBin->isVQuant = 1;
    tilingDatafromBin->methodMode = 0;

    ICPU_SET_TILING_KEY(4010);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
                nullptr, nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(kRopeScale);
    AscendC::GmFree(ckvScale);
    AscendC::GmFree(kRope);
    AscendC::GmFree(ckv);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(kv_rms_norm_rope_cache_test, test_case_4001)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 64 * 1 * 1 * 576 * sizeof(half);
    size_t gammaSize = 512 * sizeof(half);
    size_t indexSize = 64 * sizeof(uint64_t);
    size_t cosSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t sinSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t kCacheSize = 576 * 128 * 1 * 64 * sizeof(half);
    size_t ckvCacheSize = 576 * 128 * 1 * 512 * sizeof(half);
    size_t kRopeSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t ckvSize = 64 * 1 * 1 * 512 * sizeof(half);
    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 32;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* kRope = (uint8_t*)AscendC::GmAlloc(kRopeSize);
    uint8_t* ckv = (uint8_t*)AscendC::GmAlloc(ckvSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 32;
    tilingDatafromBin->rowsPerBlock = 0;
    tilingDatafromBin->cacheLength = 1;
    tilingDatafromBin->batchSize = 64;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 1;
    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->blockFactor = 2;
    tilingDatafromBin->ubFactor = 16;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 512.0f;
    tilingDatafromBin->isOutputKv = false;
    tilingDatafromBin->isKQuant = 1;
    tilingDatafromBin->isVQuant = 1;
    tilingDatafromBin->methodMode = 0;

    ICPU_SET_TILING_KEY(4001);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(kRope);
    AscendC::GmFree(ckv);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(kv_rms_norm_rope_cache_test, test_case_4000)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 38 * 1 * 3809 * 576 * sizeof(half);
    size_t gammaSize = 512 * sizeof(half);
    size_t indexSize = 228 * sizeof(uint64_t);
    size_t cosSize = 38 * 1 * 3809 * 64 * sizeof(half);
    size_t sinSize = 38 * 1 * 3809 * 64 * sizeof(half);
    size_t kCacheSize = 230 * 745 * 1 * 64 * sizeof(half);
    size_t ckvCacheSize = 230 * 745 * 1 * 512 * sizeof(half);
    size_t kRopeSize = 38 * 1 * 3809 * 64 * sizeof(half);
    size_t ckvSize = 38 * 1 * 3809 * 512 * sizeof(half);
    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 48;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* kRope = (uint8_t*)AscendC::GmAlloc(kRopeSize);
    uint8_t* ckv = (uint8_t*)AscendC::GmAlloc(ckvSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 32;
    tilingDatafromBin->rowsPerBlock = 0;
    tilingDatafromBin->cacheLength = 1;
    tilingDatafromBin->batchSize = 64;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 1;
    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->blockFactor = 2;
    tilingDatafromBin->ubFactor = 16;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 512.0f;
    tilingDatafromBin->isOutputKv = false;
    tilingDatafromBin->isKQuant = 1;
    tilingDatafromBin->isVQuant = 1;
    tilingDatafromBin->methodMode = 0;

    ICPU_SET_TILING_KEY(4000);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(kRope);
    AscendC::GmFree(ckv);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(kv_rms_norm_rope_cache_test, test_case_3001)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 64 * 1 * 7 * 576 * sizeof(half);
    size_t gammaSize = 512 * sizeof(half);
    size_t indexSize = 64 * 7 * sizeof(uint64_t);
    size_t cosSize = 64 * 1 * 7 * 64 * sizeof(half);
    size_t sinSize = 64 * 1 * 7 * 64 * sizeof(half);
    size_t kCacheSize = 576 * 128 * 1 * 64 * sizeof(half);
    size_t ckvCacheSize = 576 * 128 * 1 * 512 * sizeof(half);
    size_t kRopeSize = 64 * 1 * 7 * 64 * sizeof(half);
    size_t ckvSize = 64 * 1 * 7 * 512 * sizeof(half);
    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 45;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* kRope = (uint8_t*)AscendC::GmAlloc(kRopeSize);
    uint8_t* ckv = (uint8_t*)AscendC::GmAlloc(ckvSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 45;
    tilingDatafromBin->rowsPerBlock = 0;
    tilingDatafromBin->cacheLength = 1;
    tilingDatafromBin->batchSize = 64;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 7;
    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->blockFactor = 10;
    tilingDatafromBin->ubFactor = 16;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 192.0f;
    tilingDatafromBin->isOutputKv = true;
    tilingDatafromBin->isKQuant = 0;
    tilingDatafromBin->isVQuant = 0;
    tilingDatafromBin->methodMode = 0;

    ICPU_SET_TILING_KEY(3001);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(kRope);
    AscendC::GmFree(ckv);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(kv_rms_norm_rope_cache_test, test_case_3000)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 64 * 1 * 7 * 576 * sizeof(half);
    size_t gammaSize = 512 * sizeof(half);
    size_t indexSize = 64 * 7 * sizeof(uint64_t);
    size_t cosSize = 64 * 1 * 7 * 64 * sizeof(half);
    size_t sinSize = 64 * 1 * 7 * 64 * sizeof(half);
    size_t kCacheSize = 576 * 128 * 1 * 64 * sizeof(half);
    size_t ckvCacheSize = 576 * 128 * 1 * 512 * sizeof(half);
    size_t kRopeSize = 64 * 1 * 7 * 64 * sizeof(half);
    size_t ckvSize = 64 * 1 * 7 * 512 * sizeof(half);
    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 45;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* kRope = (uint8_t*)AscendC::GmAlloc(kRopeSize);
    uint8_t* ckv = (uint8_t*)AscendC::GmAlloc(ckvSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 45;
    tilingDatafromBin->rowsPerBlock = 0;
    tilingDatafromBin->cacheLength = 1;
    tilingDatafromBin->batchSize = 64;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 7;
    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->blockFactor = 10;
    tilingDatafromBin->ubFactor = 16;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 512.0f;
    tilingDatafromBin->isOutputKv = true;
    tilingDatafromBin->isKQuant = 0;
    tilingDatafromBin->isVQuant = 0;
    tilingDatafromBin->methodMode = 0;

    ICPU_SET_TILING_KEY(3000);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(kRope);
    AscendC::GmFree(ckv);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(kv_rms_norm_rope_cache_test, test_case_1000_2000A)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 27 * 1 * 1 * 576 * sizeof(half);
    size_t gammaSize = 512 * sizeof(half);
    size_t indexSize = 27 * 1 * sizeof(uint64_t);
    size_t cosSize = 27 * 1 * 1 * 64 * sizeof(half);
    size_t sinSize = 27 * 1 * 1 * 64 * sizeof(half);
    size_t kCacheSize = 27 * 1 * 4 * 64 * sizeof(half);
    size_t ckvCacheSize = 27 * 1 * 4 * 512 * sizeof(half);

    size_t kRopeSize = 27 * 1 * 1 * 64 * sizeof(half);
    size_t ckvSize = 27 * 1 * 1 * 512 * sizeof(half);
    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 27;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* kRope = (uint8_t*)AscendC::GmAlloc(kRopeSize);
    uint8_t* ckv = (uint8_t*)AscendC::GmAlloc(ckvSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 27;
    tilingDatafromBin->rowsPerBlock = 1;
    tilingDatafromBin->cacheLength = 4;
    tilingDatafromBin->batchSize = 27;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 1;
    tilingDatafromBin->blockSize = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->ubFactor = 0;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 512.0f;
    tilingDatafromBin->isOutputKv = false;
    tilingDatafromBin->isKQuant = 0;
    tilingDatafromBin->isVQuant = 0;
    tilingDatafromBin->methodMode = 0;

    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(1001);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(2000);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(2001);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(kRope);
    AscendC::GmFree(ckv);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(kv_rms_norm_rope_cache_test, test_case_2000_1000B)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 17 * 4 * 153 * 192 * sizeof(half);
    size_t gammaSize = 192 * sizeof(half);
    size_t indexSize = 2601 * sizeof(uint64_t);
    size_t cosSize = 17 * 4 * 1 * 64 * sizeof(half);
    size_t sinSize = 17 * 4 * 1 * 64 * sizeof(half);
    size_t kCacheSize = 113 * 53 * 4 * 192 * sizeof(half);
    size_t ckvCacheSize = 113 * 53 * 4 * 128 * sizeof(half);
    size_t vSize = 17 * 4 * 153 * 128 * sizeof(half);

    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 17;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(vSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 17;
    tilingDatafromBin->rowsPerBlock = 1;
    tilingDatafromBin->cacheLength = 4;
    tilingDatafromBin->batchSize = 17;
    tilingDatafromBin->numHead = 4;
    tilingDatafromBin->seqLength = 153;
    tilingDatafromBin->blockSize = 53;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->ubFactor = 0;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 512.0f;
    tilingDatafromBin->isOutputKv = false;
    tilingDatafromBin->isKQuant = 0;
    tilingDatafromBin->isVQuant = 0;
    tilingDatafromBin->methodMode = 1;

    ICPU_SET_TILING_KEY(2001);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(2000);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(1011);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(1010);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(1001);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(v);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(kv_rms_norm_rope_cache_test, test_case_5000_4000B)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 5 * 8 * 301 * 192 * sizeof(half);
    size_t gammaSize = 192 * sizeof(half);
    size_t indexSize = 5 * 301 * sizeof(uint64_t);
    size_t cosSize = 5 * 8 * 301 * 64 * sizeof(half);
    size_t sinSize = 5 * 8 * 301 * 64 * sizeof(half);
    size_t kCacheSize = 5 * 8 * 316 * 192 * sizeof(half);
    size_t ckvCacheSize = 5 * 8 * 316 * 128 * sizeof(half);
    size_t vSize = 5 * 8 * 301 * 128 * sizeof(half);

    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 17;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(vSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 40;
    tilingDatafromBin->rowsPerBlock = 0;
    tilingDatafromBin->cacheLength = 316;
    tilingDatafromBin->batchSize = 5;
    tilingDatafromBin->numHead = 8;
    tilingDatafromBin->seqLength = 301;
    tilingDatafromBin->blockSize = 8;
    tilingDatafromBin->blockFactor = 301;
    tilingDatafromBin->ubFactor = 16;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 192.0f;
    tilingDatafromBin->isOutputKv = false;
    tilingDatafromBin->isKQuant = 0;
    tilingDatafromBin->isVQuant = 0;
    tilingDatafromBin->methodMode = 1;

    ICPU_SET_TILING_KEY(5011);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(5010);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(5001);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(5000);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(4011);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(4010);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(4001);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(3010);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(v);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(kv_rms_norm_rope_cache_test, test_case_4000B)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 67 * 1 * 284 * 192 * sizeof(half);
    size_t gammaSize = 192 * sizeof(half);
    size_t indexSize = 201 * sizeof(uint64_t);
    size_t cosSize = 67 * 1 * 284 * 64 * sizeof(half);
    size_t sinSize = 67 * 1 * 284 * 64 * sizeof(half);
    size_t kCacheSize = 263 * 106 * 1 * 192 * sizeof(half);
    size_t ckvCacheSize = 263 * 106 * 1 * 128 * sizeof(half);
    size_t vSize = 67 * 1 * 284 * 128 * sizeof(half);

    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 48;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(vSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 48;
    tilingDatafromBin->rowsPerBlock = 0;
    tilingDatafromBin->cacheLength = 1;
    tilingDatafromBin->batchSize = 67;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 284;
    tilingDatafromBin->blockSize = 106;
    tilingDatafromBin->blockFactor = 397;
    tilingDatafromBin->ubFactor = 16;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 192.0f;
    tilingDatafromBin->isOutputKv = false;
    tilingDatafromBin->isKQuant = 0;
    tilingDatafromBin->isVQuant = 0;
    tilingDatafromBin->methodMode = 1;

    ICPU_SET_TILING_KEY(4000);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(v);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(kv_rms_norm_rope_cache_test, test_case_3000_3001B)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 16 * 1 * 364 * 192 * sizeof(half);
    size_t gammaSize = 192 * sizeof(half);
    size_t indexSize = 16 * 364 * sizeof(uint64_t);
    size_t cosSize = 16 * 1 * 364 * 64 * sizeof(half);
    size_t sinSize = 16 * 1 * 364 * 64 * sizeof(half);
    size_t kCacheSize = 16 * 1 * 379 * 192 * sizeof(half);
    size_t ckvCacheSize = 16 * 1 * 379 * 128 * sizeof(half);
    size_t vSize = 16 * 1 * 364 * 128 * sizeof(half);

    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 48;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(vSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 48;
    tilingDatafromBin->rowsPerBlock = 0;
    tilingDatafromBin->cacheLength = 379;
    tilingDatafromBin->batchSize = 16;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 364;
    tilingDatafromBin->blockSize = 1;
    tilingDatafromBin->blockFactor = 122;
    tilingDatafromBin->ubFactor = 16;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 192.0f;
    tilingDatafromBin->isOutputKv = true;
    tilingDatafromBin->isKQuant = 0;
    tilingDatafromBin->isVQuant = 0;
    tilingDatafromBin->methodMode = 1;

    ICPU_SET_TILING_KEY(3001);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(3000);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
                nullptr, nullptr, v, kCache, ckvCache, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(v);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(kv_rms_norm_rope_cache_test, test_case_1000B)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 144 * 1 * 1 * 192 * sizeof(half);
    size_t gammaSize = 192 * sizeof(half);
    size_t indexSize = 144 * 1 * sizeof(uint64_t);
    size_t cosSize = 144 * 1 * 1 * 64 * sizeof(half);
    size_t sinSize = 144 * 1 * 1 * 64 * sizeof(half);
    size_t kCacheSize = 144 * 1 * 1 * 192 * sizeof(half);
    size_t ckvCacheSize = 144 * 1 * 1 * 128 * sizeof(half);
    size_t kRopeScaleSize = 192 * sizeof(float);
    size_t ckvScaleSize = 128 * sizeof(float);
    size_t vSize = 144 * 1 * 1 * 128 * sizeof(half);

    size_t kRopeSize = 144 * 1 * 1 * 192 * sizeof(half);
    size_t ckvSize = 144 * 1 * 1 * 128 * sizeof(half);
    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 36;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* kRopeScale = (uint8_t*)AscendC::GmAlloc(kRopeScaleSize);
    uint8_t* ckvScale = (uint8_t*)AscendC::GmAlloc(ckvScaleSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(vSize);
    uint8_t* kRope = (uint8_t*)AscendC::GmAlloc(kRopeSize);
    uint8_t* ckv = (uint8_t*)AscendC::GmAlloc(ckvSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 36;
    tilingDatafromBin->rowsPerBlock = 4;
    tilingDatafromBin->cacheLength = 1;
    tilingDatafromBin->batchSize = 144;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 1;
    tilingDatafromBin->blockSize = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->ubFactor = 0;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 512.0f;
    tilingDatafromBin->isOutputKv = true;
    tilingDatafromBin->isKQuant = 1;
    tilingDatafromBin->isVQuant = 1;
    tilingDatafromBin->methodMode = 1;

    ICPU_SET_TILING_KEY(1010);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
                nullptr, nullptr, v, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(1011);
    ICPU_RUN_KF(kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
                nullptr, nullptr, v, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(kRopeScale);
    AscendC::GmFree(ckvScale);
    AscendC::GmFree(v);
    AscendC::GmFree(kRope);
    AscendC::GmFree(ckv);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

#undef DTYPE_KV
#undef DTYPE_K_CACHE
#undef DTYPE_CKV_CACHE
