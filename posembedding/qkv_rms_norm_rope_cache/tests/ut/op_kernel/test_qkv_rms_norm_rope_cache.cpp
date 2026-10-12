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
 * \file test_qkv_rms_norm_rope_cache.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include <cstdint>
#include "test_qkv_rms_norm_rope_cache_tiling.h"

#ifndef DTYPE_QKV_fp16
#define DTYPE_QKV_fp16 half
#endif
#ifndef DTYPE_QKV_bf16
#define DTYPE_QKV_bf16 bfloat16
#endif
#ifndef DTYPE_K_CACHE
#define DTYPE_K_CACHE int8_t
#endif
#ifndef DTYPE_V_CACHE
#define DTYPE_V_CACHE int8_t
#endif

using namespace std;

int batch_size;
int seq_len;
int Nqkv;
int Nq;
int Nk;
int Nv;
int dim;
int blockNum;
int blockSize;

extern "C" __global__ __aicore__ void qkv_rms_norm_rope_cache(
    GM_ADDR qkv, GM_ADDR gamma_q, GM_ADDR gamma_k, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR q_out,
    GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR k_scale, GM_ADDR v_scale, GM_ADDR k_offset, GM_ADDR v_offset,
    GM_ADDR qOut_out, GM_ADDR kCache_out, GM_ADDR vCache_out, GM_ADDR q_out_before_quant, GM_ADDR k_before_quant,
    GM_ADDR v_before_quant, GM_ADDR workspace, GM_ADDR tiling);

class qkv_rms_norm_rope_cache_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "qkv_rms_norm_rope_cache_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "qkv_rms_norm_rope_cache_test TearDown\n" << endl;
    }
};

TEST_F(qkv_rms_norm_rope_cache_test, test_QkvRmsNormRopeCache_fp16_pa_nz_bath_quant_small)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    batch_size = 4;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    blockNum = 72;
    blockSize = 128;

    size_t qkvSize = batch_size * seq_len * Nqkv * dim * sizeof(DTYPE_QKV_fp16);
    size_t gammaQSize = dim * sizeof(DTYPE_QKV_fp16);
    size_t gammaKSize = dim * sizeof(DTYPE_QKV_fp16);
    size_t cosSize = batch_size * seq_len * dim * sizeof(DTYPE_QKV_fp16);
    size_t sinSize = batch_size * seq_len * dim * sizeof(DTYPE_QKV_fp16);
    size_t indexSize = batch_size * seq_len * sizeof(uint64_t);
    size_t qOutSize = batch_size * seq_len * Nq * dim * sizeof(DTYPE_QKV_fp16);
    size_t kCacheSize = blockNum * (Nk * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t vCacheSize = blockNum * (Nv * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t kScaleSize = Nk * dim * sizeof(float);
    size_t vScaleSize = Nv * dim * sizeof(float);

    size_t qOutOutSize = batch_size * seq_len * Nq * dim * sizeof(DTYPE_QKV_fp16);
    size_t kCacheOutSize = blockNum * (Nk * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t vCacheOutSize = blockNum * (Nv * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t qOutProtoOutSize = batch_size * seq_len * Nq * dim * sizeof(DTYPE_QKV_fp16);
    size_t kCacheProtoOutSize = batch_size * seq_len * Nk * dim * sizeof(DTYPE_QKV_fp16);
    size_t vCacheProtoOutSize = batch_size * seq_len * Nv * dim * sizeof(DTYPE_QKV_fp16);

    size_t tiling_data_size = sizeof(QkvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 47;

    uint8_t* qkv = (uint8_t*)AscendC::GmAlloc(qkvSize);
    uint8_t* gammaQ = (uint8_t*)AscendC::GmAlloc(gammaQSize);
    uint8_t* gammaK = (uint8_t*)AscendC::GmAlloc(gammaKSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* qOut = (uint8_t*)AscendC::GmAlloc(qOutSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* vCache = (uint8_t*)AscendC::GmAlloc(vCacheSize);
    uint8_t* kScale = (uint8_t*)AscendC::GmAlloc(kScaleSize);
    uint8_t* vScale = (uint8_t*)AscendC::GmAlloc(vScaleSize);
    uint8_t* qOutOut = (uint8_t*)AscendC::GmAlloc(qOutOutSize);
    uint8_t* kCacheOut = (uint8_t*)AscendC::GmAlloc(kCacheOutSize);
    uint8_t* vCacheOut = (uint8_t*)AscendC::GmAlloc(vCacheOutSize);
    uint8_t* qOutProtoOut = (uint8_t*)AscendC::GmAlloc(qOutProtoOutSize);
    uint8_t* kCacheProtoOut = (uint8_t*)AscendC::GmAlloc(kCacheProtoOutSize);
    uint8_t* vCacheProtoOut = (uint8_t*)AscendC::GmAlloc(vCacheProtoOutSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QkvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<QkvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->batchSize = 4;
    tilingDatafromBin->seqLength = 2;
    tilingDatafromBin->numHead = 18;
    tilingDatafromBin->qkvDim = 128;
    tilingDatafromBin->ropeRange = 128;
    tilingDatafromBin->numHeadQ = 16;
    tilingDatafromBin->numHeadK = 1;
    tilingDatafromBin->numHeadV = 1;
    tilingDatafromBin->blockNum = 72;
    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->epsilon = 1e-6;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->blockFactorQ = 3;
    tilingDatafromBin->blockFactorK = 4;
    tilingDatafromBin->blockFactorV = 4;
    tilingDatafromBin->blockDim = 47;
    tilingDatafromBin->blockDimQ = 43;
    tilingDatafromBin->blockDimK = 2;
    tilingDatafromBin->blockDimV = 2;
    tilingDatafromBin->ubFactor = 42;
    tilingDatafromBin->ubFactorQ = 42;
    tilingDatafromBin->ubFactorK = 42;
    tilingDatafromBin->ubFactorV = 63;
    tilingDatafromBin->reciprocal = 1.0f / 128.0f;
    tilingDatafromBin->isOutputQkv = false;
    tilingDatafromBin->isKQuant = 1;
    tilingDatafromBin->isVQuant = 1;

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(qkv_rms_norm_rope_cache, blockDim, qkv, gammaQ, gammaK, cos, sin, index, qOut, kCache, vCache, kScale,
                vScale, nullptr, nullptr, qOutOut, kCacheOut, vCacheOut, qOutProtoOut, kCacheProtoOut, vCacheProtoOut,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(qkv);
    AscendC::GmFree(gammaQ);
    AscendC::GmFree(gammaK);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(qOut);
    AscendC::GmFree(kCache);
    AscendC::GmFree(vCache);
    AscendC::GmFree(kScale);
    AscendC::GmFree(vScale);
    AscendC::GmFree(qOutOut);
    AscendC::GmFree(kCacheOut);
    AscendC::GmFree(vCacheOut);
    AscendC::GmFree(qOutProtoOut);
    AscendC::GmFree(kCacheProtoOut);
    AscendC::GmFree(vCacheProtoOut);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(qkv_rms_norm_rope_cache_test, test_QkvRmsNormRopeCache_fp16_pa_nz_k_quant_small)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    batch_size = 4;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    blockNum = 72;
    blockSize = 128;

    size_t qkvSize = batch_size * seq_len * Nqkv * dim * sizeof(DTYPE_QKV_fp16);
    size_t gammaQSize = dim * sizeof(DTYPE_QKV_fp16);
    size_t gammaKSize = dim * sizeof(DTYPE_QKV_fp16);
    size_t cosSize = batch_size * seq_len * dim * sizeof(DTYPE_QKV_fp16);
    size_t sinSize = batch_size * seq_len * dim * sizeof(DTYPE_QKV_fp16);
    size_t indexSize = batch_size * seq_len * sizeof(uint64_t);
    size_t qOutSize = batch_size * seq_len * Nq * dim * sizeof(DTYPE_QKV_fp16);
    size_t kCacheSize = blockNum * (Nk * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t vCacheSize = blockNum * (Nv * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t kScaleSize = Nk * dim * sizeof(float);
    size_t vScaleSize = Nv * dim * sizeof(float);

    size_t qOutOutSize = batch_size * seq_len * Nq * dim * sizeof(DTYPE_QKV_fp16);
    size_t kCacheOutSize = blockNum * (Nk * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t vCacheOutSize = blockNum * (Nv * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t qOutProtoOutSize = batch_size * seq_len * Nq * dim * sizeof(DTYPE_QKV_fp16);
    size_t kCacheProtoOutSize = batch_size * seq_len * Nk * dim * sizeof(DTYPE_QKV_fp16);
    size_t vCacheProtoOutSize = batch_size * seq_len * Nv * dim * sizeof(DTYPE_QKV_fp16);

    size_t tiling_data_size = sizeof(QkvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 47;

    uint8_t* qkv = (uint8_t*)AscendC::GmAlloc(qkvSize);
    uint8_t* gammaQ = (uint8_t*)AscendC::GmAlloc(gammaQSize);
    uint8_t* gammaK = (uint8_t*)AscendC::GmAlloc(gammaKSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* qOut = (uint8_t*)AscendC::GmAlloc(qOutSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* vCache = (uint8_t*)AscendC::GmAlloc(vCacheSize);
    uint8_t* kScale = (uint8_t*)AscendC::GmAlloc(kScaleSize);
    uint8_t* vScale = (uint8_t*)AscendC::GmAlloc(vScaleSize);
    uint8_t* qOutOut = (uint8_t*)AscendC::GmAlloc(qOutOutSize);
    uint8_t* kCacheOut = (uint8_t*)AscendC::GmAlloc(kCacheOutSize);
    uint8_t* vCacheOut = (uint8_t*)AscendC::GmAlloc(vCacheOutSize);
    uint8_t* qOutProtoOut = (uint8_t*)AscendC::GmAlloc(qOutProtoOutSize);
    uint8_t* kCacheProtoOut = (uint8_t*)AscendC::GmAlloc(kCacheProtoOutSize);
    uint8_t* vCacheProtoOut = (uint8_t*)AscendC::GmAlloc(vCacheProtoOutSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QkvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<QkvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->batchSize = 4;
    tilingDatafromBin->seqLength = 2;
    tilingDatafromBin->numHead = 18;
    tilingDatafromBin->qkvDim = 128;
    tilingDatafromBin->ropeRange = 128;
    tilingDatafromBin->numHeadQ = 16;
    tilingDatafromBin->numHeadK = 1;
    tilingDatafromBin->numHeadV = 1;
    tilingDatafromBin->blockNum = 72;
    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->epsilon = 1e-6;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->blockFactorQ = 3;
    tilingDatafromBin->blockFactorK = 4;
    tilingDatafromBin->blockFactorV = 4;
    tilingDatafromBin->blockDim = 47;
    tilingDatafromBin->blockDimQ = 43;
    tilingDatafromBin->blockDimK = 2;
    tilingDatafromBin->blockDimV = 2;
    tilingDatafromBin->ubFactor = 42;
    tilingDatafromBin->ubFactorQ = 42;
    tilingDatafromBin->ubFactorK = 42;
    tilingDatafromBin->ubFactorV = 63;
    tilingDatafromBin->reciprocal = 1.0f / 128.0f;
    tilingDatafromBin->isOutputQkv = false;
    tilingDatafromBin->isKQuant = 1;
    tilingDatafromBin->isVQuant = 0;

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(qkv_rms_norm_rope_cache, blockDim, qkv, gammaQ, gammaK, cos, sin, index, qOut, kCache, vCache, kScale,
                nullptr, nullptr, nullptr, qOutOut, kCacheOut, vCacheOut, qOutProtoOut, kCacheProtoOut, vCacheProtoOut,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(qkv);
    AscendC::GmFree(gammaQ);
    AscendC::GmFree(gammaK);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(qOut);
    AscendC::GmFree(kCache);
    AscendC::GmFree(vCache);
    AscendC::GmFree(kScale);
    AscendC::GmFree(vScale);
    AscendC::GmFree(qOutOut);
    AscendC::GmFree(kCacheOut);
    AscendC::GmFree(vCacheOut);
    AscendC::GmFree(qOutProtoOut);
    AscendC::GmFree(kCacheProtoOut);
    AscendC::GmFree(vCacheProtoOut);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(qkv_rms_norm_rope_cache_test, test_QkvRmsNormRopeCache_fp16_pa_nz_v_quant_small)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    batch_size = 4;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    blockNum = 72;
    blockSize = 128;

    size_t qkvSize = batch_size * seq_len * Nqkv * dim * sizeof(DTYPE_QKV_fp16);
    size_t gammaQSize = dim * sizeof(DTYPE_QKV_fp16);
    size_t gammaKSize = dim * sizeof(DTYPE_QKV_fp16);
    size_t cosSize = batch_size * seq_len * dim * sizeof(DTYPE_QKV_fp16);
    size_t sinSize = batch_size * seq_len * dim * sizeof(DTYPE_QKV_fp16);
    size_t indexSize = batch_size * seq_len * sizeof(uint64_t);
    size_t qOutSize = batch_size * seq_len * Nq * dim * sizeof(DTYPE_QKV_fp16);
    size_t kCacheSize = blockNum * (Nk * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t vCacheSize = blockNum * (Nv * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t kScaleSize = Nk * dim * sizeof(float);
    size_t vScaleSize = Nv * dim * sizeof(float);

    size_t qOutOutSize = batch_size * seq_len * Nq * dim * sizeof(DTYPE_QKV_fp16);
    size_t kCacheOutSize = blockNum * (Nk * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t vCacheOutSize = blockNum * (Nv * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t qOutProtoOutSize = batch_size * seq_len * Nq * dim * sizeof(DTYPE_QKV_fp16);
    size_t kCacheProtoOutSize = batch_size * seq_len * Nk * dim * sizeof(DTYPE_QKV_fp16);
    size_t vCacheProtoOutSize = batch_size * seq_len * Nv * dim * sizeof(DTYPE_QKV_fp16);

    size_t tiling_data_size = sizeof(QkvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 47;

    uint8_t* qkv = (uint8_t*)AscendC::GmAlloc(qkvSize);
    uint8_t* gammaQ = (uint8_t*)AscendC::GmAlloc(gammaQSize);
    uint8_t* gammaK = (uint8_t*)AscendC::GmAlloc(gammaKSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* qOut = (uint8_t*)AscendC::GmAlloc(qOutSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* vCache = (uint8_t*)AscendC::GmAlloc(vCacheSize);
    uint8_t* kScale = (uint8_t*)AscendC::GmAlloc(kScaleSize);
    uint8_t* vScale = (uint8_t*)AscendC::GmAlloc(vScaleSize);
    uint8_t* qOutOut = (uint8_t*)AscendC::GmAlloc(qOutOutSize);
    uint8_t* kCacheOut = (uint8_t*)AscendC::GmAlloc(kCacheOutSize);
    uint8_t* vCacheOut = (uint8_t*)AscendC::GmAlloc(vCacheOutSize);
    uint8_t* qOutProtoOut = (uint8_t*)AscendC::GmAlloc(qOutProtoOutSize);
    uint8_t* kCacheProtoOut = (uint8_t*)AscendC::GmAlloc(kCacheProtoOutSize);
    uint8_t* vCacheProtoOut = (uint8_t*)AscendC::GmAlloc(vCacheProtoOutSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QkvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<QkvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->batchSize = 4;
    tilingDatafromBin->seqLength = 2;
    tilingDatafromBin->numHead = 18;
    tilingDatafromBin->qkvDim = 128;
    tilingDatafromBin->ropeRange = 128;
    tilingDatafromBin->numHeadQ = 16;
    tilingDatafromBin->numHeadK = 1;
    tilingDatafromBin->numHeadV = 1;
    tilingDatafromBin->blockNum = 72;
    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->epsilon = 1e-6;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->blockFactorQ = 3;
    tilingDatafromBin->blockFactorK = 4;
    tilingDatafromBin->blockFactorV = 4;
    tilingDatafromBin->blockDim = 47;
    tilingDatafromBin->blockDimQ = 43;
    tilingDatafromBin->blockDimK = 2;
    tilingDatafromBin->blockDimV = 2;
    tilingDatafromBin->ubFactor = 42;
    tilingDatafromBin->ubFactorQ = 42;
    tilingDatafromBin->ubFactorK = 42;
    tilingDatafromBin->ubFactorV = 63;
    tilingDatafromBin->reciprocal = 1.0f / 128.0f;
    tilingDatafromBin->isOutputQkv = false;
    tilingDatafromBin->isKQuant = 0;
    tilingDatafromBin->isVQuant = 1;

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(qkv_rms_norm_rope_cache, blockDim, qkv, gammaQ, gammaK, cos, sin, index, qOut, kCache, vCache, nullptr,
                vScale, nullptr, nullptr, qOutOut, kCacheOut, vCacheOut, qOutProtoOut, kCacheProtoOut, vCacheProtoOut,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(qkv);
    AscendC::GmFree(gammaQ);
    AscendC::GmFree(gammaK);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(qOut);
    AscendC::GmFree(kCache);
    AscendC::GmFree(vCache);
    AscendC::GmFree(kScale);
    AscendC::GmFree(vScale);
    AscendC::GmFree(qOutOut);
    AscendC::GmFree(kCacheOut);
    AscendC::GmFree(vCacheOut);
    AscendC::GmFree(qOutProtoOut);
    AscendC::GmFree(kCacheProtoOut);
    AscendC::GmFree(vCacheProtoOut);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(qkv_rms_norm_rope_cache_test, test_QkvRmsNormRopeCache_fp16_pa_nz_no_quant_small)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    batch_size = 4;
    seq_len = 2;
    Nqkv = 18;
    Nq = 16;
    Nk = 1;
    Nv = 1;
    dim = 128;
    blockNum = 72;
    blockSize = 128;

    size_t qkvSize = batch_size * seq_len * Nqkv * dim * sizeof(DTYPE_QKV_fp16);
    size_t gammaQSize = dim * sizeof(DTYPE_QKV_fp16);
    size_t gammaKSize = dim * sizeof(DTYPE_QKV_fp16);
    size_t cosSize = batch_size * seq_len * dim * sizeof(DTYPE_QKV_fp16);
    size_t sinSize = batch_size * seq_len * dim * sizeof(DTYPE_QKV_fp16);
    size_t indexSize = batch_size * seq_len * sizeof(uint64_t);
    size_t qOutSize = batch_size * seq_len * Nq * dim * sizeof(DTYPE_QKV_fp16);
    size_t kCacheSize = blockNum * (Nk * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t vCacheSize = blockNum * (Nv * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t kScaleSize = Nk * dim * sizeof(float);
    size_t vScaleSize = Nv * dim * sizeof(float);

    size_t qOutOutSize = batch_size * seq_len * Nq * dim * sizeof(DTYPE_QKV_fp16);
    size_t kCacheOutSize = blockNum * (Nk * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t vCacheOutSize = blockNum * (Nv * dim / 32) * blockSize * 32 * sizeof(int8_t);
    size_t qOutProtoOutSize = batch_size * seq_len * Nq * dim * sizeof(DTYPE_QKV_fp16);
    size_t kCacheProtoOutSize = batch_size * seq_len * Nk * dim * sizeof(DTYPE_QKV_fp16);
    size_t vCacheProtoOutSize = batch_size * seq_len * Nv * dim * sizeof(DTYPE_QKV_fp16);

    size_t tiling_data_size = sizeof(QkvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 47;

    uint8_t* qkv = (uint8_t*)AscendC::GmAlloc(qkvSize);
    uint8_t* gammaQ = (uint8_t*)AscendC::GmAlloc(gammaQSize);
    uint8_t* gammaK = (uint8_t*)AscendC::GmAlloc(gammaKSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* qOut = (uint8_t*)AscendC::GmAlloc(qOutSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* vCache = (uint8_t*)AscendC::GmAlloc(vCacheSize);
    uint8_t* kScale = (uint8_t*)AscendC::GmAlloc(kScaleSize);
    uint8_t* vScale = (uint8_t*)AscendC::GmAlloc(vScaleSize);
    uint8_t* qOutOut = (uint8_t*)AscendC::GmAlloc(qOutOutSize);
    uint8_t* kCacheOut = (uint8_t*)AscendC::GmAlloc(kCacheOutSize);
    uint8_t* vCacheOut = (uint8_t*)AscendC::GmAlloc(vCacheOutSize);
    uint8_t* qOutProtoOut = (uint8_t*)AscendC::GmAlloc(qOutProtoOutSize);
    uint8_t* kCacheProtoOut = (uint8_t*)AscendC::GmAlloc(kCacheProtoOutSize);
    uint8_t* vCacheProtoOut = (uint8_t*)AscendC::GmAlloc(vCacheProtoOutSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QkvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<QkvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->batchSize = 4;
    tilingDatafromBin->seqLength = 2;
    tilingDatafromBin->numHead = 18;
    tilingDatafromBin->qkvDim = 128;
    tilingDatafromBin->ropeRange = 128;
    tilingDatafromBin->numHeadQ = 16;
    tilingDatafromBin->numHeadK = 1;
    tilingDatafromBin->numHeadV = 1;
    tilingDatafromBin->blockNum = 72;
    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->epsilon = 1e-6;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->blockFactorQ = 3;
    tilingDatafromBin->blockFactorK = 4;
    tilingDatafromBin->blockFactorV = 4;
    tilingDatafromBin->blockDim = 47;
    tilingDatafromBin->blockDimQ = 43;
    tilingDatafromBin->blockDimK = 2;
    tilingDatafromBin->blockDimV = 2;
    tilingDatafromBin->ubFactor = 42;
    tilingDatafromBin->ubFactorQ = 42;
    tilingDatafromBin->ubFactorK = 42;
    tilingDatafromBin->ubFactorV = 63;
    tilingDatafromBin->reciprocal = 1.0f / 128.0f;
    tilingDatafromBin->isOutputQkv = false;
    tilingDatafromBin->isKQuant = 0;
    tilingDatafromBin->isVQuant = 0;

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(qkv_rms_norm_rope_cache, blockDim, qkv, gammaQ, gammaK, cos, sin, index, qOut, kCache, vCache, nullptr,
                nullptr, nullptr, nullptr, qOutOut, kCacheOut, vCacheOut, qOutProtoOut, kCacheProtoOut, vCacheProtoOut,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(qkv);
    AscendC::GmFree(gammaQ);
    AscendC::GmFree(gammaK);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(qOut);
    AscendC::GmFree(kCache);
    AscendC::GmFree(vCache);
    AscendC::GmFree(kScale);
    AscendC::GmFree(vScale);
    AscendC::GmFree(qOutOut);
    AscendC::GmFree(kCacheOut);
    AscendC::GmFree(vCacheOut);
    AscendC::GmFree(qOutProtoOut);
    AscendC::GmFree(kCacheProtoOut);
    AscendC::GmFree(vCacheProtoOut);
    AscendC::GmFree(tiling);
    free(path_);
}
