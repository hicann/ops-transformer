/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "dequant_rope_quant_kvcache_tiling.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void dequant_rope_quant_kvcache(
    GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR indices, GM_ADDR scale_k,
    GM_ADDR scale_v, GM_ADDR offset_k, GM_ADDR offset_v, GM_ADDR weight_scale, GM_ADDR activation_scale, GM_ADDR bias,
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR k_cache_ref, GM_ADDR v_cache_ref, GM_ADDR workspace, GM_ADDR tiling);
class dequant_rope_quant_kvcache_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "dequant_rope_quant_kvcache_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "dequant_rope_quant_kvcache_test TearDown\n" << endl;
    }
};

TEST_F(dequant_rope_quant_kvcache_test, test_dequant_rope_quant_kvcache_0)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t batch = 320;
    size_t seqlen = 1;
    size_t cacheSeqlen = 1024;
    size_t hiddensize = 128;
    size_t qHeadNum = 8;
    size_t kvHeadNum = 1;
    size_t xFileSize = batch * seqlen * (qHeadNum + kvHeadNum + kvHeadNum) * hiddensize * sizeof(half);

    size_t cosFileSize = batch * seqlen * hiddensize * sizeof(half);
    size_t cacheFileSize = batch * cacheSeqlen * (kvHeadNum)*hiddensize * sizeof(int8_t);
    size_t indicesFileSize = batch * sizeof(int32_t);
    size_t scaleFileSize = hiddensize * sizeof(float);

    size_t qFileSize = batch * seqlen * (qHeadNum)*hiddensize * sizeof(half);
    size_t kFileSize = batch * seqlen * (kvHeadNum)*hiddensize * sizeof(half);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xFileSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosFileSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(cosFileSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(cacheFileSize);
    uint8_t* vCache = (uint8_t*)AscendC::GmAlloc(cacheFileSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesFileSize);
    uint8_t* scaleK = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* scaleV = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* offsetK = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* offsetV = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* weightScale = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* activationScale = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(scaleFileSize);

    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qFileSize);
    uint8_t* k = (uint8_t*)AscendC::GmAlloc(kFileSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(kFileSize);

    uint64_t tilingKey = 0;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(DequantRopeQuantKvcacheTilingData);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    DequantRopeQuantKvcacheTilingData* tilingData = reinterpret_cast<DequantRopeQuantKvcacheTilingData*>(tiling);
    tilingData->qHeadNum = 8;
    tilingData->kvHeadNum = 1;
    tilingData->hiddenSize = 128;
    tilingData->hiddenSizeFp32Align = 128;
    tilingData->hiddenSizeFp16Align = 256;
    tilingData->hiddenSizeInt8Align = 512;
    tilingData->OnceUBMaxS = 4;
    tilingData->cacheSeqlen = 1024;
    tilingData->seqlen = 320;
    tilingData->qHiddenSize = 1024;
    tilingData->kHiddenSize = 128;
    tilingData->vHiddenSize = 128;
    tilingData->realCoreNum = 64;
    tilingData->frontCoreNum = 64;
    tilingData->blockFactor = 5;
    tilingData->tailCoreBlockFactor = 5;
    tilingData->hasQuantOffset = 0;
    tilingData->ifKVout = 0;
    tilingData->isPA = 0;
    tilingData->hasBias = 0;
    tilingData->hasAS = 0;

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        dequant_rope_quant_kvcache, blockDim, x, cos, sin, kCache, vCache, indices, scaleK, scaleV, offsetK, offsetV,
        weightScale, activationScale, bias, q, k, v, kCache, vCache, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)cos);
    AscendC::GmFree((void*)sin);
    AscendC::GmFree((void*)kCache);
    AscendC::GmFree((void*)vCache);
    AscendC::GmFree((void*)indices);
    AscendC::GmFree((void*)scaleK);
    AscendC::GmFree((void*)scaleV);
    AscendC::GmFree((void*)offsetK);
    AscendC::GmFree((void*)offsetV);
    AscendC::GmFree((void*)weightScale);
    AscendC::GmFree((void*)activationScale);
    AscendC::GmFree((void*)bias);
    AscendC::GmFree((void*)q);
    AscendC::GmFree((void*)k);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(dequant_rope_quant_kvcache_test, test_dequant_rope_quant_kvcache_1)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t batch = 320;
    size_t seqlen = 1;
    size_t cacheSeqlen = 1024;
    size_t hiddensize = 128;
    size_t qHeadNum = 8;
    size_t kvHeadNum = 1;
    size_t xFileSize = batch * seqlen * (qHeadNum + kvHeadNum + kvHeadNum) * hiddensize * sizeof(half);

    size_t cosFileSize = batch * seqlen * hiddensize * sizeof(half);
    size_t cacheFileSize = batch * cacheSeqlen * (kvHeadNum)*hiddensize * sizeof(int8_t);
    size_t indicesFileSize = batch * sizeof(int32_t);
    size_t scaleFileSize = hiddensize * sizeof(float);

    size_t qFileSize = batch * seqlen * (qHeadNum)*hiddensize * sizeof(half);
    size_t kFileSize = batch * seqlen * (kvHeadNum)*hiddensize * sizeof(half);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xFileSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosFileSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(cosFileSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(cacheFileSize);
    uint8_t* vCache = (uint8_t*)AscendC::GmAlloc(cacheFileSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesFileSize);
    uint8_t* scaleK = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* scaleV = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* offsetK = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* offsetV = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* weightScale = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* activationScale = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(scaleFileSize);

    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qFileSize);
    uint8_t* k = (uint8_t*)AscendC::GmAlloc(kFileSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(kFileSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(DequantRopeQuantKvcacheTilingData);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    DequantRopeQuantKvcacheTilingData* tilingData = reinterpret_cast<DequantRopeQuantKvcacheTilingData*>(tiling);
    tilingData->qHeadNum = 8;
    tilingData->kvHeadNum = 1;
    tilingData->hiddenSize = 128;
    tilingData->hiddenSizeFp32Align = 128;
    tilingData->hiddenSizeFp16Align = 256;
    tilingData->hiddenSizeInt8Align = 512;
    tilingData->OnceUBMaxS = 4;
    tilingData->cacheSeqlen = 1024;
    tilingData->seqlen = 320;
    tilingData->qHiddenSize = 1024;
    tilingData->kHiddenSize = 128;
    tilingData->vHiddenSize = 128;
    tilingData->realCoreNum = 64;
    tilingData->frontCoreNum = 64;
    tilingData->blockFactor = 5;
    tilingData->tailCoreBlockFactor = 5;
    tilingData->hasQuantOffset = 0;
    tilingData->ifKVout = 0;
    tilingData->isPA = 0;
    tilingData->hasBias = 0;
    tilingData->hasAS = 0;

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        dequant_rope_quant_kvcache, blockDim, x, cos, sin, kCache, vCache, indices, scaleK, scaleV, offsetK, offsetV,
        weightScale, activationScale, bias, q, k, v, kCache, vCache, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)cos);
    AscendC::GmFree((void*)sin);
    AscendC::GmFree((void*)kCache);
    AscendC::GmFree((void*)vCache);
    AscendC::GmFree((void*)indices);
    AscendC::GmFree((void*)scaleK);
    AscendC::GmFree((void*)scaleV);
    AscendC::GmFree((void*)offsetK);
    AscendC::GmFree((void*)offsetV);
    AscendC::GmFree((void*)weightScale);
    AscendC::GmFree((void*)activationScale);
    AscendC::GmFree((void*)bias);
    AscendC::GmFree((void*)q);
    AscendC::GmFree((void*)k);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(dequant_rope_quant_kvcache_test, test_dequant_rope_quant_kvcache_2)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t batch = 320;
    size_t seqlen = 1;
    size_t cacheSeqlen = 1024;
    size_t hiddensize = 128;
    size_t qHeadNum = 8;
    size_t kvHeadNum = 1;
    size_t xFileSize = batch * seqlen * (qHeadNum + kvHeadNum + kvHeadNum) * hiddensize * sizeof(half);

    size_t cosFileSize = batch * seqlen * hiddensize * sizeof(half);
    size_t cacheFileSize = batch * cacheSeqlen * (kvHeadNum)*hiddensize * sizeof(int8_t);
    size_t indicesFileSize = batch * sizeof(int32_t);
    size_t scaleFileSize = hiddensize * sizeof(float);

    size_t qFileSize = batch * seqlen * (qHeadNum)*hiddensize * sizeof(half);
    size_t kFileSize = batch * seqlen * (kvHeadNum)*hiddensize * sizeof(half);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xFileSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosFileSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(cosFileSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(cacheFileSize);
    uint8_t* vCache = (uint8_t*)AscendC::GmAlloc(cacheFileSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesFileSize);
    uint8_t* scaleK = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* scaleV = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* offsetK = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* offsetV = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* weightScale = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* activationScale = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(scaleFileSize);

    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qFileSize);
    uint8_t* k = (uint8_t*)AscendC::GmAlloc(kFileSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(kFileSize);

    uint64_t tilingKey = 2;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(DequantRopeQuantKvcacheTilingData);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    DequantRopeQuantKvcacheTilingData* tilingData = reinterpret_cast<DequantRopeQuantKvcacheTilingData*>(tiling);
    tilingData->qHeadNum = 8;
    tilingData->kvHeadNum = 1;
    tilingData->hiddenSize = 128;
    tilingData->hiddenSizeFp32Align = 128;
    tilingData->hiddenSizeFp16Align = 256;
    tilingData->hiddenSizeInt8Align = 512;
    tilingData->OnceUBMaxS = 4;
    tilingData->cacheSeqlen = 1024;
    tilingData->seqlen = 320;
    tilingData->qHiddenSize = 1024;
    tilingData->kHiddenSize = 128;
    tilingData->vHiddenSize = 128;
    tilingData->realCoreNum = 64;
    tilingData->frontCoreNum = 64;
    tilingData->blockFactor = 5;
    tilingData->tailCoreBlockFactor = 5;
    tilingData->hasQuantOffset = 0;
    tilingData->ifKVout = 0;
    tilingData->isPA = 0;
    tilingData->hasBias = 0;
    tilingData->hasAS = 0;

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        dequant_rope_quant_kvcache, blockDim, x, cos, sin, kCache, vCache, indices, scaleK, scaleV, offsetK, offsetV,
        weightScale, activationScale, bias, q, k, v, kCache, vCache, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)cos);
    AscendC::GmFree((void*)sin);
    AscendC::GmFree((void*)kCache);
    AscendC::GmFree((void*)vCache);
    AscendC::GmFree((void*)indices);
    AscendC::GmFree((void*)scaleK);
    AscendC::GmFree((void*)scaleV);
    AscendC::GmFree((void*)offsetK);
    AscendC::GmFree((void*)offsetV);
    AscendC::GmFree((void*)weightScale);
    AscendC::GmFree((void*)activationScale);
    AscendC::GmFree((void*)bias);
    AscendC::GmFree((void*)q);
    AscendC::GmFree((void*)k);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(dequant_rope_quant_kvcache_test, test_dequant_rope_quant_kvcache_3)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t batch = 320;
    size_t seqlen = 1;
    size_t cacheSeqlen = 1024;
    size_t hiddensize = 128;
    size_t qHeadNum = 8;
    size_t kvHeadNum = 1;
    size_t xFileSize = batch * seqlen * (qHeadNum + kvHeadNum + kvHeadNum) * hiddensize * sizeof(half);

    size_t cosFileSize = batch * seqlen * hiddensize * sizeof(half);
    size_t cacheFileSize = batch * cacheSeqlen * (kvHeadNum)*hiddensize * sizeof(int8_t);
    size_t indicesFileSize = batch * sizeof(int32_t);
    size_t scaleFileSize = hiddensize * sizeof(float);

    size_t qFileSize = batch * seqlen * (qHeadNum)*hiddensize * sizeof(half);
    size_t kFileSize = batch * seqlen * (kvHeadNum)*hiddensize * sizeof(half);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xFileSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosFileSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(cosFileSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(cacheFileSize);
    uint8_t* vCache = (uint8_t*)AscendC::GmAlloc(cacheFileSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesFileSize);
    uint8_t* scaleK = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* scaleV = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* offsetK = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* offsetV = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* weightScale = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* activationScale = (uint8_t*)AscendC::GmAlloc(scaleFileSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(scaleFileSize);

    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qFileSize);
    uint8_t* k = (uint8_t*)AscendC::GmAlloc(kFileSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(kFileSize);

    uint64_t tilingKey = 3;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(DequantRopeQuantKvcacheTilingData);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    DequantRopeQuantKvcacheTilingData* tilingData = reinterpret_cast<DequantRopeQuantKvcacheTilingData*>(tiling);
    tilingData->qHeadNum = 8;
    tilingData->kvHeadNum = 1;
    tilingData->hiddenSize = 128;
    tilingData->hiddenSizeFp32Align = 128;
    tilingData->hiddenSizeFp16Align = 256;
    tilingData->hiddenSizeInt8Align = 512;
    tilingData->OnceUBMaxS = 4;
    tilingData->cacheSeqlen = 1024;
    tilingData->seqlen = 320;
    tilingData->qHiddenSize = 1024;
    tilingData->kHiddenSize = 128;
    tilingData->vHiddenSize = 128;
    tilingData->realCoreNum = 64;
    tilingData->frontCoreNum = 64;
    tilingData->blockFactor = 5;
    tilingData->tailCoreBlockFactor = 5;
    tilingData->hasQuantOffset = 0;
    tilingData->ifKVout = 0;
    tilingData->isPA = 0;
    tilingData->hasBias = 0;
    tilingData->hasAS = 0;

    char* path_ = get_current_dir_name();
    string path(path_);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        dequant_rope_quant_kvcache, blockDim, x, cos, sin, kCache, vCache, indices, scaleK, scaleV, offsetK, offsetV,
        weightScale, activationScale, bias, q, k, v, kCache, vCache, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)cos);
    AscendC::GmFree((void*)sin);
    AscendC::GmFree((void*)kCache);
    AscendC::GmFree((void*)vCache);
    AscendC::GmFree((void*)indices);
    AscendC::GmFree((void*)scaleK);
    AscendC::GmFree((void*)scaleV);
    AscendC::GmFree((void*)offsetK);
    AscendC::GmFree((void*)offsetV);
    AscendC::GmFree((void*)weightScale);
    AscendC::GmFree((void*)activationScale);
    AscendC::GmFree((void*)bias);
    AscendC::GmFree((void*)q);
    AscendC::GmFree((void*)k);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}
