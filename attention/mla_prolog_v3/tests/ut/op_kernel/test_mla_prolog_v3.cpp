/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "mla_prolog_tiling_data.h"
#include "data_utils.h"
#include "../../../op_kernel/mla_prolog_v3.cpp"
#include "mla_prolog_tiling.cpp"

using namespace std;


#define PARAM_LIST_DEF  __gm__ uint8_t *tokenX,\
    __gm__ uint8_t *weightDq,\
    __gm__ uint8_t *weightUqQr,\
    __gm__ uint8_t *weightUk,\
    __gm__ uint8_t *weightDkvKr,\
    __gm__ uint8_t *rmsnormGammaCq,\
    __gm__ uint8_t *rmsnormGammaCkv,\
    __gm__ uint8_t *ropeSin,\
    __gm__ uint8_t *ropeCos,\
    __gm__ uint8_t *kvCache,\
    __gm__ uint8_t *krCache,\
    __gm__ uint8_t *cacheIndex,\
    __gm__ uint8_t *dequantScaleX,\
    __gm__ uint8_t *dequantScaleWDq,\
    __gm__ uint8_t *dequantScaleWUqQr,\
    __gm__ uint8_t *dequantScaleWDkvKr,\
    __gm__ uint8_t *quantScaleCkv,\
    __gm__ uint8_t *quantScaleCkr,\
    __gm__ uint8_t *smoothScalesCq,\
    __gm__ uint8_t *actualSeqLen,\
    __gm__ uint8_t *kNopeClipAlpha,\
    __gm__ uint8_t *queryOut,\
    __gm__ uint8_t *queryRopeOut,\
    __gm__ uint8_t *kvCacheOut,\
    __gm__ uint8_t *krCacheOut,\
    __gm__ uint8_t *dequantScaleQNopeOut,\
    __gm__ uint8_t *queryNormOut,\
    __gm__ uint8_t *dequantScaleQNormOut,\
    __gm__ uint8_t *workspace,\
    __gm__ uint8_t *tiling

#define PARAM_LIST tokenX,\
    weightDq,\
    weightUqQr,\
    weightUk,\
    weightDkvKr,\
    rmsnormGammaCq,\
    rmsnormGammaCkv,\
    ropeSin,\
    ropeCos,\
    kvCache,\
    krCache,\
    cacheIndex,\
    dequantScaleX,\
    dequantScaleWDq,\
    dequantScaleWUqQr,\
    dequantScaleWDkvKr,\
    quantScaleCkv,\
    quantScaleCkr,\
    smoothScalesCq,\
    actualSeqLen,\
    kNopeClipAlpha,\
    queryOut,\
    queryRopeOut,\
    kvCacheOut,\
    krCacheOut,\
    dequantScaleQNopeOut,\
    queryNormOut,\
    dequantScaleQNormOut,\
    workspace,\
    tiling



class MlaPrologV3Kernel : public testing::Test
{
protected:
    static void FreeAllGmMemory(
        uint8_t* tokenX,
        uint8_t* weightDq,
        uint8_t* weightUqQr,
        uint8_t* weightUk,
        uint8_t* weightDkvKr,
        uint8_t* rmsnormGammaCq,
        uint8_t* rmsnormGammaCkv,
        uint8_t* ropeSin,
        uint8_t* ropeCos,
        uint8_t* kvCache,
        uint8_t* krCache,
        uint8_t* cacheIndex,
        uint8_t* dequantScaleX,
        uint8_t* dequantScaleWDq,
        uint8_t* dequantScaleWUqQr,
        uint8_t* dequantScaleWDkvKr,
        uint8_t* quantScaleCkv,
        uint8_t* quantScaleCkr,
        uint8_t* smoothScalesCq,
        uint8_t* actualSeqLen,
        uint8_t* kNopeClipAlpha,
        uint8_t* queryOut,
        uint8_t* queryRopeOut,
        uint8_t* dequantScaleQNopeOut,
        uint8_t* queryNormOut,
        uint8_t* dequantScaleQNormOut,
        uint8_t* workspace,
        uint8_t* tiling ) {
        // 逐个释放非空指针，避免空指针访问
        if (tokenX != nullptr) AscendC::GmFree(tokenX);
        if (weightDq != nullptr) AscendC::GmFree(weightDq);
        if (weightUqQr != nullptr) AscendC::GmFree(weightUqQr);
        if (weightUk != nullptr) AscendC::GmFree(weightUk);
        if (weightDkvKr != nullptr) AscendC::GmFree(weightDkvKr);
        if (rmsnormGammaCq != nullptr) AscendC::GmFree(rmsnormGammaCq);
        if (rmsnormGammaCkv != nullptr) AscendC::GmFree(rmsnormGammaCkv);
        if (ropeSin != nullptr) AscendC::GmFree(ropeSin);
        if (ropeCos != nullptr) AscendC::GmFree(ropeCos);
        if (kvCache != nullptr) AscendC::GmFree(kvCache);
        if (krCache != nullptr) AscendC::GmFree(krCache);
        if (cacheIndex != nullptr) AscendC::GmFree(cacheIndex);
        if (dequantScaleX != nullptr) AscendC::GmFree(dequantScaleX);
        if (dequantScaleWDq != nullptr) AscendC::GmFree(dequantScaleWDq);
        if (dequantScaleWUqQr != nullptr) AscendC::GmFree(dequantScaleWUqQr);
        if (dequantScaleWDkvKr != nullptr) AscendC::GmFree(dequantScaleWDkvKr);
        if (quantScaleCkv != nullptr) AscendC::GmFree(quantScaleCkv);
        if (quantScaleCkr != nullptr) AscendC::GmFree(quantScaleCkr);
        if (smoothScalesCq != nullptr) AscendC::GmFree(smoothScalesCq);
        if (actualSeqLen != nullptr) AscendC::GmFree(actualSeqLen);
        if (kNopeClipAlpha != nullptr) AscendC::GmFree(kNopeClipAlpha);
        if (queryOut != nullptr) AscendC::GmFree(queryOut);
        if (queryRopeOut != nullptr) AscendC::GmFree(queryRopeOut);
        if (dequantScaleQNopeOut != nullptr) AscendC::GmFree(dequantScaleQNopeOut);
        if (queryNormOut != nullptr) AscendC::GmFree(queryNormOut);
        if (workspace != nullptr) AscendC::GmFree(workspace);
        if (tiling != nullptr) AscendC::GmFree(tiling);
    }
    static void SetUpTestCase()
    {
        std::cout << "MlaPrologV3Kernel SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "MlaPrologV3Kernel TearDown\n" << std::endl;
    }
};


// 非量化场景
TEST_F(MlaPrologV3Kernel, test_case_v3_noQuant)
{
    uint32_t B = 1;
    uint32_t N = 4; 
    uint32_t N2 = 1;
    uint32_t D = 128;
    uint32_t Dr = 64;
    uint32_t S1 = 1;
    uint32_t S2 = 1;
    uint32_t He = 7168;
    uint32_t Hckv = 512;
    uint32_t T = 0;
    uint32_t Block_Number = 1;
    uint32_t Block_Size = 16;
    uint32_t Nkv = 1;
    uint32_t Dtile = 512;
    uint32_t blockDim = 20;
    uint32_t Hcq= 1536;
    
    // 输入变量
    uint8_t* tokenX = (uint8_t*)AscendC::GmAlloc(B * S1 * He * sizeof(half));
    uint8_t* weightDq = (uint8_t*)AscendC::GmAlloc(He * Hcq * sizeof(half));
    uint8_t* weightUqQr = (uint8_t*)AscendC::GmAlloc(Hcq * N*(D + Dr) * sizeof(half));
    uint8_t* weightUk = (uint8_t*)AscendC::GmAlloc(N * D * Hckv *sizeof(half));
    uint8_t* weightDkvKr = (uint8_t*)AscendC::GmAlloc(He * (Hckv + Dr) * sizeof(half));
    uint8_t* rmsnormGammaCq = (uint8_t*)AscendC::GmAlloc(Hcq * sizeof(half));
    uint8_t* rmsnormGammaCkv = (uint8_t*)AscendC::GmAlloc(Hckv * sizeof(half));
    uint8_t* ropeSin = (uint8_t*)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t* ropeCos = (uint8_t*)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t* cacheIndex = (uint8_t*)AscendC::GmAlloc(B * S1 * sizeof(int64_t));
    uint8_t* kvCache = (uint8_t*)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Hckv * sizeof(half));
    uint8_t* krCache = (uint8_t*)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Dr * sizeof(half));
    uint8_t* dequantScaleX = nullptr;
    uint8_t* dequantScaleWDq = nullptr;
    uint8_t* dequantScaleWUqQr = nullptr;
    uint8_t* dequantScaleWDkvKr = nullptr;
    uint8_t* quantScaleCkv = nullptr;
    uint8_t* quantScaleCkr = nullptr;
    uint8_t* smoothScalesCq = nullptr;
    uint8_t* actualSeqLen = nullptr;
    uint8_t* kNopeClipAlpha = nullptr;
    uint8_t* dequantScaleQNopeOut = nullptr;
    // 输出类变量
    uint8_t* queryOut = (uint8_t*)AscendC::GmAlloc(B * S1 * N * Hckv * sizeof(half));
    uint8_t* queryRopeOut = (uint8_t*)AscendC::GmAlloc(B * S1 * N * Dr * sizeof(half));
    uint8_t* queryNormOut = nullptr;
    uint8_t* dequantScaleQNormOut = nullptr;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16777216);
    size_t tilingDataSize = sizeof(optiling::MlaPrologBaseParams);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    // ===================tiligndata赋值====================
    optiling::MlaPrologBaseParams* baseParams_ = reinterpret_cast<optiling::MlaPrologBaseParams*>(tiling);
    baseParams_->batchSize = 1;
    baseParams_->stepBatchSize = 1;
    baseParams_->stepNumHeadDequant = 4;
    baseParams_->mSubSize = 0;
    baseParams_->mSubCoreNum = 0;
    baseParams_->tokenSize = 1;
    baseParams_->seq1Size = 1;
    baseParams_->seq2Size = 1;
    baseParams_->headSizeX = 7168;
    baseParams_->headSizeCq = 1536;
    baseParams_->headSizeCkv = 512;
    baseParams_->dtileSize = 512;
    baseParams_->headSizeQc = 512;
    baseParams_->headSizeQr = 256;
    baseParams_->headSizeKr = 64;
    baseParams_->numHeadSize = 4;
    baseParams_->numHeadKvSize = 1;
    baseParams_->dimHeadSizeQc = 128;
    baseParams_->dimHeadRope = 64;
    baseParams_->blockNum = 1;
    baseParams_->blockSize = 16;
    baseParams_->mm1BlockNum = 24;
    baseParams_->mm2BlockNum = 9;
    baseParams_->mm3BlockNum = 24;
    baseParams_->mm4BlockNum = 4;
    baseParams_->mm1SingleCoreN = 64;
    baseParams_->mm2SingleCoreN = 64;
    baseParams_->mm3SingleCoreN = 32;
    baseParams_->mm4SingleCoreBatch = 1;
    baseParams_->vectorBlockNum = 1;
    baseParams_->reciprocalCq = 0.000651;
    baseParams_->epsilonCq = 0.000010;
    baseParams_->reciprocalCkv = 0.001953;
    baseParams_->epsilonCkv = 0.000010;
    baseParams_->queryNormFlag = 0U;
    baseParams_->kvQuantMode = 0U;
    baseParams_->ckvkrRepoMode = 0U;
    baseParams_->quantScaleRepoMode = 0U;
    baseParams_->tileSize = 0U;
    baseParams_->qcQrScale = 1U;
    baseParams_->kcScale = 1U;
    baseParams_->isQcQrScaleEnable = 0U;
    baseParams_->isKcScaleEnable = 0U;
    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF){return mla_prolog_v3< 1, // CACHE_MODE
                                                1, // SCENARIO
                                                0, // QUANT_MODE
                                                0, // ENABLE_DEQUANT_OPTIONAL
                                                0, // ENABLE_GROUP_COMPUTE_OPTIONAL
                                                0, // EMPTY_TENSOR_MODE
                                                0, // ACTUAL_SEQ_LEN_MODE
                                                0, // SPLIT_M_MODE
                                                7 // CV_MODE
                                                >(PARAM_LIST);};

    ICPU_RUN_KF(func, blockDim, tokenX, weightDq, weightUqQr,
        weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv,
        ropeSin, ropeCos, kvCache, krCache, cacheIndex, dequantScaleX, dequantScaleWDq, 
        dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq,
        actualSeqLen, kNopeClipAlpha, queryOut, queryRopeOut, kvCache, krCache,
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling);

    MlaPrologV3Kernel::FreeAllGmMemory(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr,
        rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache,
        krCache, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
        dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen,
        kNopeClipAlpha, queryOut, queryRopeOut,
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling);
}

// 半量化kv非量化场景
TEST_F(MlaPrologV3Kernel, test_case_v3_semiQuantKVNoQuant)
{
    uint32_t B = 1;
    uint32_t N = 4; 
    uint32_t N2 = 1;
    uint32_t D = 128;
    uint32_t Dr = 64;
    uint32_t S1 = 1;
    uint32_t S2 = 1;
    uint32_t He = 7168;
    uint32_t Hckv = 512;
    uint32_t T = 0;
    uint32_t Block_Number = 1;
    uint32_t Block_Size = 16;
    uint32_t Nkv = 1;
    uint32_t Dtile = 512;
    uint32_t blockDim = 20;
    uint32_t Hcq= 1536;
    
    // 输入变量
    uint8_t* tokenX = (uint8_t*)AscendC::GmAlloc(B * S1 * He * sizeof(half));
    uint8_t* weightDq = (uint8_t*)AscendC::GmAlloc(He * Hcq * sizeof(half));
    uint8_t* weightUqQr = (uint8_t*)AscendC::GmAlloc(Hcq * N*(D + Dr) * sizeof(int8_t));
    uint8_t* weightUk = (uint8_t*)AscendC::GmAlloc(N * D * Hckv *sizeof(half));
    uint8_t* weightDkvKr = (uint8_t*)AscendC::GmAlloc(He * (Hckv + Dr) * sizeof(half));
    uint8_t* rmsnormGammaCq = (uint8_t*)AscendC::GmAlloc(Hcq * sizeof(half));
    uint8_t* rmsnormGammaCkv = (uint8_t*)AscendC::GmAlloc(Hckv * sizeof(half));
    uint8_t* ropeSin = (uint8_t*)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t* ropeCos = (uint8_t*)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t* cacheIndex = (uint8_t*)AscendC::GmAlloc(B * S1 * sizeof(int64_t));
    uint8_t* kvCache = (uint8_t*)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Hckv * sizeof(half));
    uint8_t* krCache = (uint8_t*)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Dr * sizeof(half));
    uint8_t* dequantScaleX = nullptr;
    uint8_t* dequantScaleWDq = nullptr;
    uint8_t* dequantScaleWUqQr = (uint8_t*)AscendC::GmAlloc(N*(D+Dr)*sizeof(float));
    uint8_t* dequantScaleWDkvKr = nullptr;
    uint8_t* quantScaleCkv = nullptr;
    uint8_t* quantScaleCkr = nullptr;
    uint8_t* smoothScalesCq = nullptr;
    uint8_t* actualSeqLen = nullptr;
    uint8_t* kNopeClipAlpha = nullptr;
    uint8_t* dequantScaleQNopeOut = nullptr;
    // 输出类变量
    uint8_t* queryOut = (uint8_t*)AscendC::GmAlloc(B * S1 * N * Hckv * sizeof(half));
    uint8_t* queryRopeOut = (uint8_t*)AscendC::GmAlloc(B * S1 * N * Dr * sizeof(half));
    uint8_t* queryNormOut = nullptr;
    uint8_t* dequantScaleQNormOut = nullptr;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16788608);
    size_t tilingDataSize = sizeof(optiling::MlaPrologBaseParams);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    // ===================tiligndata赋值====================
    optiling::MlaPrologBaseParams* baseParams_ = reinterpret_cast<optiling::MlaPrologBaseParams*>(tiling);
    baseParams_->batchSize = 1;
    baseParams_->stepBatchSize = 1;
    baseParams_->stepNumHeadDequant = 4;
    baseParams_->mSubSize = 0;
    baseParams_->mSubCoreNum = 0;
    baseParams_->tokenSize = 1;
    baseParams_->seq1Size = 1;
    baseParams_->seq2Size = 1;
    baseParams_->headSizeX = 7168;
    baseParams_->headSizeCq = 1536;
    baseParams_->headSizeCkv = 512;
    baseParams_->dtileSize = 512;
    baseParams_->headSizeQc = 512;
    baseParams_->headSizeQr = 256;
    baseParams_->headSizeKr = 64;
    baseParams_->numHeadSize = 4;
    baseParams_->numHeadKvSize = 1;
    baseParams_->dimHeadSizeQc = 128;
    baseParams_->dimHeadRope = 64;
    baseParams_->blockNum = 1;
    baseParams_->blockSize = 16;
    baseParams_->mm1BlockNum = 24;
    baseParams_->mm2BlockNum = 9;
    baseParams_->mm3BlockNum = 24;
    baseParams_->mm4BlockNum = 4;
    baseParams_->mm1SingleCoreN = 64;
    baseParams_->mm2SingleCoreN = 64;
    baseParams_->mm3SingleCoreN = 32;
    baseParams_->mm4SingleCoreBatch = 1;
    baseParams_->vectorBlockNum = 1;
    baseParams_->reciprocalCq = 0.000651;
    baseParams_->epsilonCq = 0.000010;
    baseParams_->reciprocalCkv = 0.001953;
    baseParams_->epsilonCkv = 0.000010;
    baseParams_->queryNormFlag = 0U;
    baseParams_->kvQuantMode = 0U;
    baseParams_->ckvkrRepoMode = 0U;
    baseParams_->quantScaleRepoMode = 0U;
    baseParams_->tileSize = 0U;
    baseParams_->qcQrScale = 1U;
    baseParams_->kcScale = 1U;
    baseParams_->isQcQrScaleEnable = 0U;
    baseParams_->isKcScaleEnable = 0U;
    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF){return mla_prolog_v3< 1, // CACHE_MODE
                                                2, // SCENARIO
                                                1, // QUANT_MODE
                                                0, // ENABLE_DEQUANT_OPTIONAL
                                                0, // ENABLE_GROUP_COMPUTE_OPTIONAL
                                                0, // EMPTY_TENSOR_MODE
                                                0, // ACTUAL_SEQ_LEN_MODE
                                                0, // SPLIT_M_MODE
                                                7 // CV_MODE
                                                >(PARAM_LIST);};

    ICPU_RUN_KF(func, blockDim, tokenX, weightDq, weightUqQr,
        weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv,
        ropeSin, ropeCos, kvCache, krCache, cacheIndex, dequantScaleX, dequantScaleWDq, 
        dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq,
        actualSeqLen, kNopeClipAlpha, queryOut, queryRopeOut, kvCache, krCache,
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling);

    MlaPrologV3Kernel::FreeAllGmMemory(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr,
        rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache,
        krCache, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
        dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen,
        kNopeClipAlpha, queryOut, queryRopeOut,
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling);
}

// 半量化kv量化场景
TEST_F(MlaPrologV3Kernel, test_case_v3_semiQuantKVQuant)
{
    uint32_t B = 1;
    uint32_t N = 4; 
    uint32_t N2 = 1;
    uint32_t D = 128;
    uint32_t Dr = 64;
    uint32_t S1 = 1;
    uint32_t S2 = 1;
    uint32_t He = 7168;
    uint32_t Hckv = 512;
    uint32_t T = 0;
    uint32_t Block_Number = 1;
    uint32_t Block_Size = 16;
    uint32_t Nkv = 1;
    uint32_t Dtile = 512;
    uint32_t blockDim = 20;
    uint32_t Hcq= 1536;
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF){return mla_prolog_v3< 1, // CACHE_MODE
                                                3, // SCENARIO
                                                2, // QUANT_MODE
                                                0, // ENABLE_DEQUANT_OPTIONAL
                                                0, // ENABLE_GROUP_COMPUTE_OPTIONAL
                                                0, // EMPTY_TENSOR_MODE
                                                0, // ACTUAL_SEQ_LEN_MODE
                                                0, // SPLIT_M_MODE
                                                7 // CV_MODE
                                                >(PARAM_LIST);};
    // 输入变量
    uint8_t* tokenX = (uint8_t*)AscendC::GmAlloc(B * S1 * He * sizeof(half));
    uint8_t* weightDq = (uint8_t*)AscendC::GmAlloc(He * Hcq * sizeof(half));
    uint8_t* weightUqQr = (uint8_t*)AscendC::GmAlloc(Hcq * N*(D + Dr) * sizeof(int8_t));
    uint8_t* weightUk = (uint8_t*)AscendC::GmAlloc(N * D * Hckv *sizeof(half));
    uint8_t* weightDkvKr = (uint8_t*)AscendC::GmAlloc(He * (Hckv + Dr) * sizeof(half));
    uint8_t* rmsnormGammaCq = (uint8_t*)AscendC::GmAlloc(Hcq * sizeof(half));
    uint8_t* rmsnormGammaCkv = (uint8_t*)AscendC::GmAlloc(Hckv * sizeof(half));
    uint8_t* ropeSin = (uint8_t*)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t* ropeCos = (uint8_t*)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t* cacheIndex = (uint8_t*)AscendC::GmAlloc(B * S1 * sizeof(int64_t));
    uint8_t* kvCache = (uint8_t*)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Hckv * sizeof(int8_t));
    uint8_t* krCache = (uint8_t*)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Dr * sizeof(int8_t));
    uint8_t* dequantScaleX = nullptr;
    uint8_t* dequantScaleWDq = nullptr;
    uint8_t* dequantScaleWUqQr = (uint8_t*)AscendC::GmAlloc(N*(D+Dr)*sizeof(float));
    uint8_t* dequantScaleWDkvKr = nullptr;
    uint8_t* quantScaleCkv = (uint8_t*)AscendC::GmAlloc(Hckv * sizeof(float));
    uint8_t* quantScaleCkr = (uint8_t*)AscendC::GmAlloc(Dr * sizeof(float));
    uint8_t* smoothScalesCq = nullptr;
    uint8_t* actualSeqLen = nullptr;
    uint8_t* kNopeClipAlpha = nullptr;
    uint8_t* dequantScaleQNopeOut = nullptr;
    // 输出类变量
    uint8_t* queryOut = (uint8_t*)AscendC::GmAlloc(B * S1 * N * Hckv * sizeof(half));
    uint8_t* queryRopeOut = (uint8_t*)AscendC::GmAlloc(B * S1 * N * Dr * sizeof(half));
    uint8_t* queryNormOut = nullptr;
    uint8_t* dequantScaleQNormOut = nullptr;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16788608);
    size_t tilingDataSize = sizeof(optiling::MlaPrologBaseParams);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    // ===================tiligndata赋值====================
    optiling::MlaPrologBaseParams* baseParams_ = reinterpret_cast<optiling::MlaPrologBaseParams*>(tiling);
    baseParams_->batchSize = 1;
    baseParams_->stepBatchSize = 1;
    baseParams_->stepNumHeadDequant = 4;
    baseParams_->mSubSize = 0;
    baseParams_->mSubCoreNum = 0;
    baseParams_->tokenSize = 1;
    baseParams_->seq1Size = 1;
    baseParams_->seq2Size = 1;
    baseParams_->headSizeX = 7168;
    baseParams_->headSizeCq = 1536;
    baseParams_->headSizeCkv = 512;
    baseParams_->dtileSize = 512;
    baseParams_->headSizeQc = 512;
    baseParams_->headSizeQr = 256;
    baseParams_->headSizeKr = 64;
    baseParams_->numHeadSize = 4;
    baseParams_->numHeadKvSize = 1;
    baseParams_->dimHeadSizeQc = 128;
    baseParams_->dimHeadRope = 64;
    baseParams_->blockNum = 1;
    baseParams_->blockSize = 16;
    baseParams_->mm1BlockNum = 24;
    baseParams_->mm2BlockNum = 9;
    baseParams_->mm3BlockNum = 24;
    baseParams_->mm4BlockNum = 4;
    baseParams_->mm1SingleCoreN = 64;
    baseParams_->mm2SingleCoreN = 64;
    baseParams_->mm3SingleCoreN = 32;
    baseParams_->mm4SingleCoreBatch = 1;
    baseParams_->vectorBlockNum = 1;
    baseParams_->reciprocalCq = 0.000651;
    baseParams_->epsilonCq = 0.000010;
    baseParams_->reciprocalCkv = 0.001953;
    baseParams_->epsilonCkv = 0.000010;
    baseParams_->queryNormFlag = 0U;
    baseParams_->kvQuantMode = 0U;
    baseParams_->ckvkrRepoMode = 0U;
    baseParams_->quantScaleRepoMode = 0U;
    baseParams_->tileSize = 0U;
    baseParams_->qcQrScale = 1U;
    baseParams_->kcScale = 1U;
    baseParams_->isQcQrScaleEnable = 0U;
    baseParams_->isKcScaleEnable = 0U;

    ICPU_RUN_KF(func, blockDim, tokenX, weightDq, weightUqQr,
        weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv,
        ropeSin, ropeCos, kvCache, krCache, cacheIndex, dequantScaleX, dequantScaleWDq, 
        dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq,
        actualSeqLen, kNopeClipAlpha, queryOut, queryRopeOut, kvCache, krCache,
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling);

    MlaPrologV3Kernel::FreeAllGmMemory(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr,
        rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache,
        krCache, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
        dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen,
        kNopeClipAlpha, queryOut, queryRopeOut,
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling);
}

// 半量化kv量化pertile场景
TEST_F(MlaPrologV3Kernel, test_case_v3_semiQuantKVQuantPerfill)
{
   uint32_t B = 1;
    uint32_t N = 4; 
    uint32_t N2 = 1;
    uint32_t D = 128;
    uint32_t Dr = 64;
    uint32_t S1 = 1;
    uint32_t S2 = 1;
    uint32_t He = 7168;
    uint32_t Hckv = 512;
    uint32_t T = 0;
    uint32_t Block_Number = 1;
    uint32_t Block_Size = 16;
    uint32_t Nkv = 1;
    uint32_t Dtile = 656;
    uint32_t blockDim = 20;
    uint32_t Hcq= 1536;
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF){return mla_prolog_v3< 1, // CACHE_MODE
                                                4, // SCENARIO
                                                2, // QUANT_MODE
                                                0, // ENABLE_DEQUANT_OPTIONAL
                                                0, // ENABLE_GROUP_COMPUTE_OPTIONAL
                                                0, // EMPTY_TENSOR_MODE
                                                0, // ACTUAL_SEQ_LEN_MODE
                                                0, // SPLIT_M_MODE
                                                7 // CV_MODE
                                                >(PARAM_LIST);};
    // 输入变量
    uint8_t* tokenX = (uint8_t*)AscendC::GmAlloc(B * S1 * He * sizeof(half));
    uint8_t* weightDq = (uint8_t*)AscendC::GmAlloc(He * Hcq * sizeof(half));
    uint8_t* weightUqQr = (uint8_t*)AscendC::GmAlloc(Hcq * N*(D + Dr) * sizeof(int8_t));
    uint8_t* weightUk = (uint8_t*)AscendC::GmAlloc(N * D * Hckv *sizeof(half));
    uint8_t* weightDkvKr = (uint8_t*)AscendC::GmAlloc(He * (Hckv + Dr) * sizeof(half));
    uint8_t* rmsnormGammaCq = (uint8_t*)AscendC::GmAlloc(Hcq * sizeof(half));
    uint8_t* rmsnormGammaCkv = (uint8_t*)AscendC::GmAlloc(Hckv * sizeof(half));
    uint8_t* ropeSin = (uint8_t*)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t* ropeCos = (uint8_t*)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t* cacheIndex = (uint8_t*)AscendC::GmAlloc(B * S1 * sizeof(int64_t));
    uint8_t* kvCache = (uint8_t*)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Hckv * sizeof(int8_t));
    uint8_t* krCache = (uint8_t*)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Dr * sizeof(int8_t));
    uint8_t* dequantScaleX = nullptr;
    uint8_t* dequantScaleWDq = nullptr;
    uint8_t* dequantScaleWUqQr = (uint8_t*)AscendC::GmAlloc(N*(D+Dr)*sizeof(float));
    uint8_t* dequantScaleWDkvKr = nullptr;
    uint8_t* quantScaleCkv = (uint8_t*)AscendC::GmAlloc(Hckv * sizeof(float));
    uint8_t* quantScaleCkr = (uint8_t*)AscendC::GmAlloc(Dr * sizeof(float));
    uint8_t* smoothScalesCq = nullptr;
    uint8_t* actualSeqLen = nullptr;
    uint8_t* kNopeClipAlpha = nullptr;
    uint8_t* dequantScaleQNopeOut = nullptr;
    // 输出类变量
    uint8_t* queryOut = (uint8_t*)AscendC::GmAlloc(B * S1 * N * Hckv * sizeof(half));
    uint8_t* queryRopeOut = (uint8_t*)AscendC::GmAlloc(B * S1 * N * Dr * sizeof(half));
    uint8_t* queryNormOut = nullptr;
    uint8_t* dequantScaleQNormOut = nullptr;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16788608);
    size_t tilingDataSize = sizeof(optiling::MlaPrologBaseParams);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    // ===================tiligndata赋值====================
    optiling::MlaPrologBaseParams* baseParams_ = reinterpret_cast<optiling::MlaPrologBaseParams*>(tiling);
    baseParams_->batchSize = 1;
    baseParams_->stepBatchSize = 1;
    baseParams_->stepNumHeadDequant = 4;
    baseParams_->mSubSize = 0;
    baseParams_->mSubCoreNum = 0;
    baseParams_->tokenSize = 1;
    baseParams_->seq1Size = 1;
    baseParams_->seq2Size = 1;
    baseParams_->headSizeX = 7168;
    baseParams_->headSizeCq = 1536;
    baseParams_->headSizeCkv = 512;
    baseParams_->dtileSize = 512;
    baseParams_->headSizeQc = 512;
    baseParams_->headSizeQr = 256;
    baseParams_->headSizeKr = 64;
    baseParams_->numHeadSize = 4;
    baseParams_->numHeadKvSize = 1;
    baseParams_->dimHeadSizeQc = 128;
    baseParams_->dimHeadRope = 64;
    baseParams_->blockNum = 1;
    baseParams_->blockSize = 16;
    baseParams_->mm1BlockNum = 24;
    baseParams_->mm2BlockNum = 9;
    baseParams_->mm3BlockNum = 24;
    baseParams_->mm4BlockNum = 4;
    baseParams_->mm1SingleCoreN = 64;
    baseParams_->mm2SingleCoreN = 64;
    baseParams_->mm3SingleCoreN = 32;
    baseParams_->mm4SingleCoreBatch = 1;
    baseParams_->vectorBlockNum = 1;
    baseParams_->reciprocalCq = 0.000651;
    baseParams_->epsilonCq = 0.000010;
    baseParams_->reciprocalCkv = 0.001953;
    baseParams_->epsilonCkv = 0.000010;
    baseParams_->queryNormFlag = 0U;
    baseParams_->kvQuantMode = 0U;
    baseParams_->ckvkrRepoMode = 0U;
    baseParams_->quantScaleRepoMode = 0U;
    baseParams_->tileSize = 0U;
    baseParams_->qcQrScale = 1U;
    baseParams_->kcScale = 1U;
    baseParams_->isQcQrScaleEnable = 0U;
    baseParams_->isKcScaleEnable = 0U;

    ICPU_RUN_KF(func, blockDim, tokenX, weightDq, weightUqQr,
        weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv,
        ropeSin, ropeCos, kvCache, krCache, cacheIndex, dequantScaleX, dequantScaleWDq, 
        dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq,
        actualSeqLen, kNopeClipAlpha, queryOut, queryRopeOut, kvCache, krCache,
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling);

    MlaPrologV3Kernel::FreeAllGmMemory(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr,
        rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache,
        krCache, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
        dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen,
        kNopeClipAlpha, queryOut, queryRopeOut,
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling);
}

// 全量化kv非量化场景
TEST_F(MlaPrologV3Kernel, test_case_v3_QuantKVNoQuant)
{
   
    uint32_t B = 1;
    uint32_t N = 4; 
    uint32_t N2 = 1;
    uint32_t D = 128;
    uint32_t Dr = 64;
    uint32_t S1 = 1;
    uint32_t S2 = 1;
    uint32_t He = 7168;
    uint32_t Hckv = 512;
    uint32_t T = 0;
    uint32_t Block_Number = 1;
    uint32_t Block_Size = 16;
    uint32_t Nkv = 1;
    uint32_t Dtile = 512;
    uint32_t blockDim = 20;
    uint32_t Hcq= 1536;
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF){return mla_prolog_v3< 1, // CACHE_MODE
                                                5, // SCENARIO
                                                2, // QUANT_MODE
                                                0, // ENABLE_DEQUANT_OPTIONAL
                                                0, // ENABLE_GROUP_COMPUTE_OPTIONAL
                                                0, // EMPTY_TENSOR_MODE
                                                0, // ACTUAL_SEQ_LEN_MODE
                                                0, // SPLIT_M_MODE
                                                7 // CV_MODE
                                                >(PARAM_LIST);};
    // 输入变量
    uint8_t* tokenX = (uint8_t*)AscendC::GmAlloc(B * S1 * He * sizeof(half));
    uint8_t* weightDq = (uint8_t*)AscendC::GmAlloc(He * Hcq * sizeof(half));
    uint8_t* weightUqQr = (uint8_t*)AscendC::GmAlloc(Hcq * N*(D + Dr) * sizeof(int8_t));
    uint8_t* weightUk = (uint8_t*)AscendC::GmAlloc(N * D * Hckv *sizeof(half));
    uint8_t* weightDkvKr = (uint8_t*)AscendC::GmAlloc(He * (Hckv + Dr) * sizeof(half));
    uint8_t* rmsnormGammaCq = (uint8_t*)AscendC::GmAlloc(Hcq * sizeof(half));
    uint8_t* rmsnormGammaCkv = (uint8_t*)AscendC::GmAlloc(Hckv * sizeof(half));
    uint8_t* ropeSin = (uint8_t*)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t* ropeCos = (uint8_t*)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t* cacheIndex = (uint8_t*)AscendC::GmAlloc(B * S1 * sizeof(int64_t));
    uint8_t* kvCache = (uint8_t*)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Hckv * sizeof(int8_t));
    uint8_t* krCache = (uint8_t*)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Dr * sizeof(int8_t));
    uint8_t* dequantScaleX = nullptr;
    uint8_t* dequantScaleWDq = nullptr;
    uint8_t* dequantScaleWUqQr = (uint8_t*)AscendC::GmAlloc(N*(D+Dr)*sizeof(float));
    uint8_t* dequantScaleWDkvKr = nullptr;
    uint8_t* quantScaleCkv = (uint8_t*)AscendC::GmAlloc(Hckv * sizeof(float));
    uint8_t* quantScaleCkr = (uint8_t*)AscendC::GmAlloc(Dr * sizeof(float));
    uint8_t* smoothScalesCq = nullptr;
    uint8_t* actualSeqLen = nullptr;
    uint8_t* kNopeClipAlpha = nullptr;
    uint8_t* dequantScaleQNopeOut = nullptr;
    // 输出类变量
    uint8_t* queryOut = (uint8_t*)AscendC::GmAlloc(B * S1 * N * Hckv * sizeof(half));
    uint8_t* queryRopeOut = (uint8_t*)AscendC::GmAlloc(B * S1 * N * Dr * sizeof(half));
    uint8_t* queryNormOut = nullptr;
    uint8_t* dequantScaleQNormOut = nullptr;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16788608);
    size_t tilingDataSize = sizeof(optiling::MlaPrologBaseParams);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    // ===================tiligndata赋值====================
    optiling::MlaPrologBaseParams* baseParams_ = reinterpret_cast<optiling::MlaPrologBaseParams*>(tiling);
    baseParams_->batchSize = 1;
    baseParams_->stepBatchSize = 1;
    baseParams_->stepNumHeadDequant = 4;
    baseParams_->mSubSize = 0;
    baseParams_->mSubCoreNum = 0;
    baseParams_->tokenSize = 1;
    baseParams_->seq1Size = 1;
    baseParams_->seq2Size = 1;
    baseParams_->headSizeX = 7168;
    baseParams_->headSizeCq = 1536;
    baseParams_->headSizeCkv = 512;
    baseParams_->dtileSize = 512;
    baseParams_->headSizeQc = 512;
    baseParams_->headSizeQr = 256;
    baseParams_->headSizeKr = 64;
    baseParams_->numHeadSize = 4;
    baseParams_->numHeadKvSize = 1;
    baseParams_->dimHeadSizeQc = 128;
    baseParams_->dimHeadRope = 64;
    baseParams_->blockNum = 1;
    baseParams_->blockSize = 16;
    baseParams_->mm1BlockNum = 24;
    baseParams_->mm2BlockNum = 9;
    baseParams_->mm3BlockNum = 24;
    baseParams_->mm4BlockNum = 4;
    baseParams_->mm1SingleCoreN = 64;
    baseParams_->mm2SingleCoreN = 64;
    baseParams_->mm3SingleCoreN = 32;
    baseParams_->mm4SingleCoreBatch = 1;
    baseParams_->vectorBlockNum = 1;
    baseParams_->reciprocalCq = 0.000651;
    baseParams_->epsilonCq = 0.000010;
    baseParams_->reciprocalCkv = 0.001953;
    baseParams_->epsilonCkv = 0.000010;
    baseParams_->queryNormFlag = 0U;
    baseParams_->kvQuantMode = 0U;
    baseParams_->ckvkrRepoMode = 0U;
    baseParams_->quantScaleRepoMode = 0U;
    baseParams_->tileSize = 0U;
    baseParams_->qcQrScale = 1U;
    baseParams_->kcScale = 1U;
    baseParams_->isQcQrScaleEnable = 0U;
    baseParams_->isKcScaleEnable = 0U;

    ICPU_RUN_KF(func, blockDim, tokenX, weightDq, weightUqQr,
        weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv,
        ropeSin, ropeCos, kvCache, krCache, cacheIndex, dequantScaleX, dequantScaleWDq, 
        dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq,
        actualSeqLen, kNopeClipAlpha, queryOut, queryRopeOut, kvCache, krCache,
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling);

    MlaPrologV3Kernel::FreeAllGmMemory(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr,
        rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache,
        krCache, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
        dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen,
        kNopeClipAlpha, queryOut, queryRopeOut,
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling);
}

// 全量化kv量化场景
TEST_F(MlaPrologV3Kernel, test_case_v3_QuantKVQuant)
{
    uint32_t B = 1;
    uint32_t N = 4; 
    uint32_t N2 = 1;
    uint32_t D = 128;
    uint32_t Dr = 64;
    uint32_t S1 = 1;
    uint32_t S2 = 1;
    uint32_t He = 7168;
    uint32_t Hckv = 512;
    uint32_t T = 0;
    uint32_t Block_Number = 1;
    uint32_t Block_Size = 16;
    uint32_t Nkv = 1;
    uint32_t Dtile = 512;
    uint32_t blockDim = 20;
    uint32_t Hcq= 1536;
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF){return mla_prolog_v3< 1, // CACHE_MODE
                                                5, // SCENARIO
                                                2, // QUANT_MODE
                                                0, // ENABLE_DEQUANT_OPTIONAL
                                                0, // ENABLE_GROUP_COMPUTE_OPTIONAL
                                                0, // EMPTY_TENSOR_MODE
                                                0, // ACTUAL_SEQ_LEN_MODE
                                                0, // SPLIT_M_MODE
                                                7 // CV_MODE
                                                >(PARAM_LIST);};
    // 输入变量
    uint8_t* tokenX = (uint8_t*)AscendC::GmAlloc(B * S1 * He * sizeof(half));
    uint8_t* weightDq = (uint8_t*)AscendC::GmAlloc(He * Hcq * sizeof(half));
    uint8_t* weightUqQr = (uint8_t*)AscendC::GmAlloc(Hcq * N*(D + Dr) * sizeof(int8_t));
    uint8_t* weightUk = (uint8_t*)AscendC::GmAlloc(N * D * Hckv *sizeof(half));
    uint8_t* weightDkvKr = (uint8_t*)AscendC::GmAlloc(He * (Hckv + Dr) * sizeof(half));
    uint8_t* rmsnormGammaCq = (uint8_t*)AscendC::GmAlloc(Hcq * sizeof(half));
    uint8_t* rmsnormGammaCkv = (uint8_t*)AscendC::GmAlloc(Hckv * sizeof(half));
    uint8_t* ropeSin = (uint8_t*)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t* ropeCos = (uint8_t*)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t* cacheIndex = (uint8_t*)AscendC::GmAlloc(B * S1 * sizeof(int64_t));
    uint8_t* kvCache = (uint8_t*)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Hckv * sizeof(int8_t));
    uint8_t* krCache = (uint8_t*)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Dr * sizeof(half));
    uint8_t* dequantScaleX = nullptr;
    uint8_t* dequantScaleWDq = nullptr;
    uint8_t* dequantScaleWUqQr = (uint8_t*)AscendC::GmAlloc(N*(D+Dr)*sizeof(float));
    uint8_t* dequantScaleWDkvKr = nullptr;
    uint8_t* quantScaleCkv = (uint8_t*)AscendC::GmAlloc(Hckv * sizeof(float));
    uint8_t* quantScaleCkr = (uint8_t*)AscendC::GmAlloc(Dr * sizeof(float));
    uint8_t* smoothScalesCq = nullptr;
    uint8_t* actualSeqLen = nullptr;
    uint8_t* kNopeClipAlpha = nullptr;
    uint8_t* dequantScaleQNopeOut = nullptr;
    // 输出类变量
    uint8_t* queryOut = (uint8_t*)AscendC::GmAlloc(B * S1 * N * Hckv * sizeof(half));
    uint8_t* queryRopeOut = (uint8_t*)AscendC::GmAlloc(B * S1 * N * Dr * sizeof(half));
    uint8_t* queryNormOut = nullptr;
    uint8_t* dequantScaleQNormOut = nullptr;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16788608);
    size_t tilingDataSize = sizeof(optiling::MlaPrologBaseParams);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    // ===================tiligndata赋值====================
    optiling::MlaPrologBaseParams* baseParams_ = reinterpret_cast<optiling::MlaPrologBaseParams*>(tiling);
    baseParams_->batchSize = 1;
    baseParams_->stepBatchSize = 1;
    baseParams_->stepNumHeadDequant = 4;
    baseParams_->mSubSize = 0;
    baseParams_->mSubCoreNum = 0;
    baseParams_->tokenSize = 1;
    baseParams_->seq1Size = 1;
    baseParams_->seq2Size = 1;
    baseParams_->headSizeX = 7168;
    baseParams_->headSizeCq = 1536;
    baseParams_->headSizeCkv = 512;
    baseParams_->dtileSize = 512;
    baseParams_->headSizeQc = 512;
    baseParams_->headSizeQr = 256;
    baseParams_->headSizeKr = 64;
    baseParams_->numHeadSize = 4;
    baseParams_->numHeadKvSize = 1;
    baseParams_->dimHeadSizeQc = 128;
    baseParams_->dimHeadRope = 64;
    baseParams_->blockNum = 1;
    baseParams_->blockSize = 16;
    baseParams_->mm1BlockNum = 24;
    baseParams_->mm2BlockNum = 9;
    baseParams_->mm3BlockNum = 24;
    baseParams_->mm4BlockNum = 4;
    baseParams_->mm1SingleCoreN = 64;
    baseParams_->mm2SingleCoreN = 64;
    baseParams_->mm3SingleCoreN = 32;
    baseParams_->mm4SingleCoreBatch = 1;
    baseParams_->vectorBlockNum = 1;
    baseParams_->reciprocalCq = 0.000651;
    baseParams_->epsilonCq = 0.000010;
    baseParams_->reciprocalCkv = 0.001953;
    baseParams_->epsilonCkv = 0.000010;
    baseParams_->queryNormFlag = 0U;
    baseParams_->kvQuantMode = 0U;
    baseParams_->ckvkrRepoMode = 0U;
    baseParams_->quantScaleRepoMode = 0U;
    baseParams_->tileSize = 0U;
    baseParams_->qcQrScale = 1U;
    baseParams_->kcScale = 1U;
    baseParams_->isQcQrScaleEnable = 0U;
    baseParams_->isKcScaleEnable = 0U;

    ICPU_RUN_KF(func, blockDim, tokenX, weightDq, weightUqQr,
        weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv,
        ropeSin, ropeCos, kvCache, krCache, cacheIndex, dequantScaleX, dequantScaleWDq, 
        dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq,
        actualSeqLen, kNopeClipAlpha, queryOut, queryRopeOut, kvCache, krCache,
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling);

    MlaPrologV3Kernel::FreeAllGmMemory(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr,
        rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache,
        krCache, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
        dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen,
        kNopeClipAlpha, queryOut, queryRopeOut,
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling);
}

// 全量化kv量化pertile场景
TEST_F(MlaPrologV3Kernel, test_case_v3_QuantKVQuantPerfill)
{
    uint32_t B = 1;
    uint32_t N = 4; 
    uint32_t N2 = 1;
    uint32_t D = 128;
    uint32_t Dr = 64;
    uint32_t S1 = 1;
    uint32_t S2 = 1;
    uint32_t He = 7168;
    uint32_t Hckv = 512;
    uint32_t T = 0;
    uint32_t Block_Number = 1;
    uint32_t Block_Size = 16;
    uint32_t Nkv = 1;
    uint32_t Dtile = 512;
    uint32_t blockDim = 20;
    uint32_t Hcq= 1536;
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF){return mla_prolog_v3< 1, // CACHE_MODE
                                                6, // SCENARIO
                                                2, // QUANT_MODE
                                                0, // ENABLE_DEQUANT_OPTIONAL
                                                0, // ENABLE_GROUP_COMPUTE_OPTIONAL
                                                0, // EMPTY_TENSOR_MODE
                                                0, // ACTUAL_SEQ_LEN_MODE
                                                0, // SPLIT_M_MODE
                                                7 // CV_MODE
                                                >(PARAM_LIST);};
    // 输入变量
    uint8_t* tokenX = (uint8_t*)AscendC::GmAlloc(B * S1 * He * sizeof(half));
    uint8_t* weightDq = (uint8_t*)AscendC::GmAlloc(He * Hcq * sizeof(half));
    uint8_t* weightUqQr = (uint8_t*)AscendC::GmAlloc(Hcq * N*(D + Dr) * sizeof(int8_t));
    uint8_t* weightUk = (uint8_t*)AscendC::GmAlloc(N * D * Hckv *sizeof(half));
    uint8_t* weightDkvKr = (uint8_t*)AscendC::GmAlloc(He * (Hckv + Dr) * sizeof(half));
    uint8_t* rmsnormGammaCq = (uint8_t*)AscendC::GmAlloc(Hcq * sizeof(half));
    uint8_t* rmsnormGammaCkv = (uint8_t*)AscendC::GmAlloc(Hckv * sizeof(half));
    uint8_t* ropeSin = (uint8_t*)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t* ropeCos = (uint8_t*)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t* cacheIndex = (uint8_t*)AscendC::GmAlloc(B * S1 * sizeof(int64_t));
    uint8_t* kvCache = (uint8_t*)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Hckv * sizeof(int8_t));
    uint8_t* krCache = (uint8_t*)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Dr * sizeof(half));
    uint8_t* dequantScaleX = nullptr;
    uint8_t* dequantScaleWDq = nullptr;
    uint8_t* dequantScaleWUqQr = (uint8_t*)AscendC::GmAlloc(N*(D+Dr)*sizeof(float));
    uint8_t* dequantScaleWDkvKr = nullptr;
    uint8_t* quantScaleCkv = (uint8_t*)AscendC::GmAlloc(Hckv * sizeof(float));
    uint8_t* quantScaleCkr = (uint8_t*)AscendC::GmAlloc(Dr * sizeof(float));
    uint8_t* smoothScalesCq = nullptr;
    uint8_t* actualSeqLen = nullptr;
    uint8_t* kNopeClipAlpha = nullptr;
    uint8_t* dequantScaleQNopeOut = nullptr;
    // 输出类变量
    uint8_t* queryOut = (uint8_t*)AscendC::GmAlloc(B * S1 * N * Hckv * sizeof(half));
    uint8_t* queryRopeOut = (uint8_t*)AscendC::GmAlloc(B * S1 * N * Dr * sizeof(half));
    uint8_t* queryNormOut = nullptr;
    uint8_t* dequantScaleQNormOut = nullptr;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16788608);
    size_t tilingDataSize = sizeof(optiling::MlaPrologBaseParams);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    // ===================tiligndata赋值====================
    optiling::MlaPrologBaseParams* baseParams_ = reinterpret_cast<optiling::MlaPrologBaseParams*>(tiling);
    baseParams_->batchSize = 1;
    baseParams_->stepBatchSize = 1;
    baseParams_->stepNumHeadDequant = 4;
    baseParams_->mSubSize = 0;
    baseParams_->mSubCoreNum = 0;
    baseParams_->tokenSize = 1;
    baseParams_->seq1Size = 1;
    baseParams_->seq2Size = 1;
    baseParams_->headSizeX = 7168;
    baseParams_->headSizeCq = 1536;
    baseParams_->headSizeCkv = 512;
    baseParams_->dtileSize = 512;
    baseParams_->headSizeQc = 512;
    baseParams_->headSizeQr = 256;
    baseParams_->headSizeKr = 64;
    baseParams_->numHeadSize = 4;
    baseParams_->numHeadKvSize = 1;
    baseParams_->dimHeadSizeQc = 128;
    baseParams_->dimHeadRope = 64;
    baseParams_->blockNum = 1;
    baseParams_->blockSize = 16;
    baseParams_->mm1BlockNum = 24;
    baseParams_->mm2BlockNum = 9;
    baseParams_->mm3BlockNum = 24;
    baseParams_->mm4BlockNum = 4;
    baseParams_->mm1SingleCoreN = 64;
    baseParams_->mm2SingleCoreN = 64;
    baseParams_->mm3SingleCoreN = 32;
    baseParams_->mm4SingleCoreBatch = 1;
    baseParams_->vectorBlockNum = 1;
    baseParams_->reciprocalCq = 0.000651;
    baseParams_->epsilonCq = 0.000010;
    baseParams_->reciprocalCkv = 0.001953;
    baseParams_->epsilonCkv = 0.000010;
    baseParams_->queryNormFlag = 0U;
    baseParams_->kvQuantMode = 0U;
    baseParams_->ckvkrRepoMode = 0U;
    baseParams_->quantScaleRepoMode = 0U;
    baseParams_->tileSize = 0U;
    baseParams_->qcQrScale = 1U;
    baseParams_->kcScale = 1U;
    baseParams_->isQcQrScaleEnable = 0U;
    baseParams_->isKcScaleEnable = 0U;

    ICPU_RUN_KF(func, blockDim, tokenX, weightDq, weightUqQr,
        weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv,
        ropeSin, ropeCos, kvCache, krCache, cacheIndex, dequantScaleX, dequantScaleWDq, 
        dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq,
        actualSeqLen, kNopeClipAlpha, queryOut, queryRopeOut, kvCache, krCache,
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling);

    MlaPrologV3Kernel::FreeAllGmMemory(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr,
        rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache,
        krCache, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
        dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen,
        kNopeClipAlpha, queryOut, queryRopeOut,
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling);
}