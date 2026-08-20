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
#include "../../../op_kernel/mla_prolog_v2.cpp"
#include "mla_prolog_tiling.cpp"

using namespace std;

#define PARAM_LIST_DEF \
    __gm__ uint8_t *tokenX, __gm__ uint8_t *weightDq, __gm__ uint8_t *weightUqQr, __gm__ uint8_t *weightUk, \
        __gm__ uint8_t *weightDkvKr, __gm__ uint8_t *rmsnormGammaCq, __gm__ uint8_t *rmsnormGammaCkv, \
        __gm__ uint8_t *ropeSin, __gm__ uint8_t *ropeCos, __gm__ uint8_t *cacheIndex, __gm__ uint8_t *kvCache, \
        __gm__ uint8_t *krCache, __gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq, \
        __gm__ uint8_t *dequantScaleWUqQr, __gm__ uint8_t *dequantScaleWDkvKr, __gm__ uint8_t *quantScaleCkv, \
        __gm__ uint8_t *quantScaleCkr, __gm__ uint8_t *smoothScalesCq, __gm__ uint8_t *queryOut, \
        __gm__ uint8_t *queryRopeOut, __gm__ uint8_t *kvCacheOut, __gm__ uint8_t *krCacheOut, \
        __gm__ uint8_t *dequantScaleQNopeOut, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling

#define PARAM_LIST \
    tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, \
        cacheIndex, kvCache, krCache, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, \
        quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut, kvCacheOut, krCacheOut, \
        dequantScaleQNopeOut, workspace, tiling

class MlaPrologV2Kernel : public testing::Test {
protected:
    static void FreeAllGmMemory(uint8_t *tokenX, uint8_t *weightDq, uint8_t *weightUqQr, uint8_t *weightUk,
                                uint8_t *weightDkvKr, uint8_t *rmsnormGammaCq, uint8_t *rmsnormGammaCkv,
                                uint8_t *ropeSin, uint8_t *ropeCos, uint8_t *cacheIndex, uint8_t *kvCache,
                                uint8_t *krCache, uint8_t *dequantScaleX, uint8_t *dequantScaleWDq,
                                uint8_t *dequantScaleWUqQr, uint8_t *dequantScaleWDkvKr, uint8_t *quantScaleCkv,
                                uint8_t *quantScaleCkr, uint8_t *smoothScalesCq, uint8_t *queryOut,
                                uint8_t *queryRopeOut, uint8_t *dequantScaleQNopeOut, uint8_t *workspace,
                                uint8_t *tiling)
    {
        // 逐个释放非空指针，避免空指针访问
        if (tokenX != nullptr)
            AscendC::GmFree(tokenX);
        if (weightDq != nullptr)
            AscendC::GmFree(weightDq);
        if (weightUqQr != nullptr)
            AscendC::GmFree(weightUqQr);
        if (weightUk != nullptr)
            AscendC::GmFree(weightUk);
        if (weightDkvKr != nullptr)
            AscendC::GmFree(weightDkvKr);
        if (rmsnormGammaCq != nullptr)
            AscendC::GmFree(rmsnormGammaCq);
        if (rmsnormGammaCkv != nullptr)
            AscendC::GmFree(rmsnormGammaCkv);
        if (ropeSin != nullptr)
            AscendC::GmFree(ropeSin);
        if (ropeCos != nullptr)
            AscendC::GmFree(ropeCos);
        if (cacheIndex != nullptr)
            AscendC::GmFree(cacheIndex);
        if (kvCache != nullptr)
            AscendC::GmFree(kvCache);
        if (krCache != nullptr)
            AscendC::GmFree(krCache);
        if (dequantScaleX != nullptr)
            AscendC::GmFree(dequantScaleX);
        if (dequantScaleWDq != nullptr)
            AscendC::GmFree(dequantScaleWDq);
        if (dequantScaleWUqQr != nullptr)
            AscendC::GmFree(dequantScaleWUqQr);
        if (dequantScaleWDkvKr != nullptr)
            AscendC::GmFree(dequantScaleWDkvKr);
        if (quantScaleCkv != nullptr)
            AscendC::GmFree(quantScaleCkv);
        if (quantScaleCkr != nullptr)
            AscendC::GmFree(quantScaleCkr);
        if (smoothScalesCq != nullptr)
            AscendC::GmFree(smoothScalesCq);
        if (queryOut != nullptr)
            AscendC::GmFree(queryOut);
        if (queryRopeOut != nullptr)
            AscendC::GmFree(queryRopeOut);
        if (workspace != nullptr)
            AscendC::GmFree(workspace);
        if (tiling != nullptr)
            AscendC::GmFree(tiling);
    }
    static void SetUpTestCase() { std::cout << "MlaPrologV2Kernel SetUp\n" << std::endl; }
    static void TearDownTestCase() { std::cout << "MlaPrologV2Kernel TearDown\n" << std::endl; }
};

// 半量化kv非量化场景
TEST_F(MlaPrologV2Kernel, test_case_v2_semiQuantKVNoQuant)
{
    uint32_t B = 1;
    uint32_t N = 8;
    uint32_t N2 = 1;
    uint32_t D = 128;
    uint32_t Dr = 63;
    uint32_t S1 = 1;
    uint32_t S2 = 0;
    uint32_t He = 7168;
    uint32_t Hckv = 512;
    uint32_t T = 0;
    uint32_t Block_Number = 3;
    uint32_t Block_Size = 64;
    uint32_t Nkv = 1;
    uint32_t Dtile = 656;
    uint32_t blockDim = 12;
    uint32_t Hcq = 1536;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF) {
        return mla_prolog_v2<1, // CACHE_MODE
                             1, // SCENARIO
                             1, // QUANT_MODE
                             0, // ENABLE_DEQUANT_OPTIONAL
                             1, // ENABLE_GROUP_COMPUTE_OPTIONAL
                             0, // EMPTY_TENSOR_MODE
                             0, // ACTUAL_SEQ_LEN_MODE
                             0, // SPLIT_M_MODE
                             1, // ENABLE_ROPE
                             7  // CV_MODE
                             >(PARAM_LIST);
    };
    // 输入变量
    uint8_t *tokenX = (uint8_t *)AscendC::GmAlloc(B * S1 * He * sizeof(half));
    uint8_t *weightDq = (uint8_t *)AscendC::GmAlloc(He * Hcq * sizeof(half));
    uint8_t *weightUqQr = (uint8_t *)AscendC::GmAlloc(Hcq * N * (D + Dr) * sizeof(int8_t));
    uint8_t *weightUk = (uint8_t *)AscendC::GmAlloc(N * D * Hckv * sizeof(half));
    uint8_t *weightDkvKr = (uint8_t *)AscendC::GmAlloc(He * (Hckv + Dr) * sizeof(half));
    uint8_t *rmsnormGammaCq = (uint8_t *)AscendC::GmAlloc(Hcq * sizeof(half));
    uint8_t *rmsnormGammaCkv = (uint8_t *)AscendC::GmAlloc(Hckv * sizeof(half));
    uint8_t *ropeSin = (uint8_t *)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t *ropeCos = (uint8_t *)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t *cacheIndex = (uint8_t *)AscendC::GmAlloc(B * S1 * sizeof(half));
    uint8_t *kvCache = (uint8_t *)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Dtile * sizeof(int8_t));
    uint8_t *krCache = (uint8_t *)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Dr * sizeof(half));
    uint8_t *dequantScaleX = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *dequantScaleWDq = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *dequantScaleWUqQr = (uint8_t *)AscendC::GmAlloc(1 * N * (D + Dr) * sizeof(float));
    uint8_t *dequantScaleWDkvKr = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *quantScaleCkv = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *quantScaleCkr = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *smoothScalesCq = (uint8_t *)AscendC::GmAlloc(1 * Hcq * sizeof(float));
    // 输出类变量
    uint8_t *queryOut = (uint8_t *)AscendC::GmAlloc(B * S1 * N * Hckv * sizeof(half));
    uint8_t *queryRopeOut = (uint8_t *)AscendC::GmAlloc(B * S1 * N * Dr * sizeof(half));
    uint8_t *dequantScaleQNopeOut = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(2048 * 2048 * 2048);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(sizeof(optiling::MlaPrologBaseParams));

    // ===================tiligndata赋值====================
    optiling::MlaPrologBaseParams *baseParams_ = reinterpret_cast<optiling::MlaPrologBaseParams *>(tiling);
    uint32_t aicNum_ = 12;
    uint32_t aivNum_ = 12;
    baseParams_->batchSize = B;
    baseParams_->seq1Size = S1;
    baseParams_->headSizeKr = Dr;
    baseParams_->numHeadSize = N;
    baseParams_->numHeadKvSize = Nkv;
    baseParams_->dimHeadSizeQc = D;
    baseParams_->dimHeadRope = Dr;
    baseParams_->headSizeX = He;
    baseParams_->headSizeCq = Hcq;
    baseParams_->headSizeCkv = Hckv;
    baseParams_->dtileSize = Dtile;
    baseParams_->blockNum = Block_Number;
    baseParams_->blockSize = Block_Size;
    baseParams_->tokenSize = baseParams_->batchSize * baseParams_->seq1Size;
    baseParams_->stepBatchSize = std::min(64U, baseParams_->tokenSize);
    baseParams_->stepNumHeadDequant = std::min(64U, N);
    baseParams_->mSubCoreNum = baseParams_->tokenSize - (baseParams_->mSubSize - 1U) * aicNum_;
    baseParams_->mSubSize = (baseParams_->tokenSize + aicNum_ - 1U) / aicNum_;
    baseParams_->seq2Size = baseParams_->numHeadKvSize;
    baseParams_->headSizeQc = baseParams_->dimHeadSizeQc * baseParams_->numHeadSize;
    baseParams_->headSizeQr = baseParams_->headSizeKr * baseParams_->numHeadSize;
    baseParams_->mm1SingleCoreN = optiling::CalcSingleCoreNV2(baseParams_->headSizeCq, aicNum_, Block_Size / 1);
    baseParams_->mm2SingleCoreN = 64U;
    baseParams_->mm3SingleCoreN = optiling::CalcSingleCoreNV2(
        baseParams_->numHeadSize * (baseParams_->headSizeKr + baseParams_->dimHeadSizeQc), aicNum_, D + Dr);
    baseParams_->mm4SingleCoreBatch = CeilDiv(baseParams_->numHeadSize, aicNum_);
    baseParams_->mm1BlockNum = CeilDiv(baseParams_->headSizeCq, baseParams_->mm1SingleCoreN);
    baseParams_->mm2BlockNum = (baseParams_->headSizeCkv + baseParams_->dimHeadRope) / baseParams_->mm2SingleCoreN;
    baseParams_->mm3BlockNum = CeilDiv(
        baseParams_->numHeadSize * (baseParams_->headSizeKr + baseParams_->dimHeadSizeQc), baseParams_->mm3SingleCoreN);
    baseParams_->mm4BlockNum = CeilDiv(baseParams_->numHeadSize, baseParams_->mm4SingleCoreBatch);
    baseParams_->vectorBlockNum = std::min(std::min(128U, baseParams_->tokenSize), aivNum_);
    baseParams_->reciprocalCq = 1.0f / baseParams_->headSizeCq;
    baseParams_->epsilonCq = 0.00461f;
    baseParams_->reciprocalCkv = 1.0f / baseParams_->headSizeCkv;
    baseParams_->epsilonCkv = 0.001166f;
    // v3
    baseParams_->queryNormFlag = 0U;
    baseParams_->kvQuantMode = static_cast<uint32_t>(0);
    baseParams_->ckvkrRepoMode = static_cast<uint32_t>(0);
    baseParams_->quantScaleRepoMode = static_cast<uint32_t>(0);
    baseParams_->tileSize = baseParams_->batchSize * baseParams_->stepBatchSize;
    baseParams_->qcQrScale = 1U;
    baseParams_->kcScale = 1U;
    baseParams_->isQcQrScaleEnable =
        static_cast<uint16_t>(std::abs(baseParams_->qcQrScale - 1.0f) >= std::numeric_limits<float>::epsilon());
    baseParams_->isKcScaleEnable =
        static_cast<uint16_t>(std::abs(baseParams_->kcScale - 1.0f) >= std::numeric_limits<float>::epsilon());

    ICPU_RUN_KF(func, blockDim, tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv,
                ropeSin, ropeCos, cacheIndex, kvCache, krCache, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut, nullptr,
                nullptr, dequantScaleQNopeOut, workspace, tiling);

    MlaPrologV2Kernel::FreeAllGmMemory(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
        cacheIndex, kvCache, krCache, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr,
        quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut, dequantScaleQNopeOut, workspace, tiling);
}

// 半量化kv量化场景
TEST_F(MlaPrologV2Kernel, test_case_v2_semiQuantKVQuant)
{
    uint32_t B = 1;
    uint32_t N = 8;
    uint32_t N2 = 1;
    uint32_t D = 128;
    uint32_t Dr = 63;
    uint32_t S1 = 1;
    uint32_t S2 = 0;
    uint32_t He = 7168;
    uint32_t Hckv = 512;
    uint32_t T = 0;
    uint32_t Block_Number = 3;
    uint32_t Block_Size = 64;
    uint32_t Nkv = 1;
    uint32_t Dtile = 656;
    uint32_t blockDim = 12;
    uint32_t Hcq = 1536;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF) {
        return mla_prolog_v2<1, // CACHE_MODE
                             2, // SCENARIO
                             2, // QUANT_MODE
                             0, // ENABLE_DEQUANT_OPTIONAL
                             0, // ENABLE_GROUP_COMPUTE_OPTIONAL
                             0, // EMPTY_TENSOR_MODE
                             0, // ACTUAL_SEQ_LEN_MODE
                             0, // SPLIT_M_MODE
                             1, // ENABLE_ROPE
                             7  // CV_MODE
                             >(PARAM_LIST);
    };
    // 输入变量
    uint8_t *tokenX = (uint8_t *)AscendC::GmAlloc(B * S1 * He * sizeof(half));
    uint8_t *weightDq = (uint8_t *)AscendC::GmAlloc(He * Hcq * sizeof(half));
    uint8_t *weightUqQr = (uint8_t *)AscendC::GmAlloc(Hcq * N * (D + Dr) * sizeof(int8_t));
    uint8_t *weightUk = (uint8_t *)AscendC::GmAlloc(N * D * Hckv * sizeof(half));
    uint8_t *weightDkvKr = (uint8_t *)AscendC::GmAlloc(He * (Hckv + Dr) * sizeof(half));
    uint8_t *rmsnormGammaCq = (uint8_t *)AscendC::GmAlloc(Hcq * sizeof(half));
    uint8_t *rmsnormGammaCkv = (uint8_t *)AscendC::GmAlloc(Hckv * sizeof(half));
    uint8_t *ropeSin = (uint8_t *)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t *ropeCos = (uint8_t *)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t *cacheIndex = (uint8_t *)AscendC::GmAlloc(B * S1 * sizeof(half));
    uint8_t *kvCache = (uint8_t *)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Dtile * sizeof(int8_t));
    uint8_t *krCache = (uint8_t *)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Dr * sizeof(half));
    uint8_t *dequantScaleX = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *dequantScaleWDq = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *dequantScaleWUqQr = (uint8_t *)AscendC::GmAlloc(1 * N * (D + Dr) * sizeof(float));
    uint8_t *dequantScaleWDkvKr = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *quantScaleCkv = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *quantScaleCkr = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *smoothScalesCq = (uint8_t *)AscendC::GmAlloc(1 * Hcq * sizeof(float));
    // 输出类变量
    uint8_t *queryOut = (uint8_t *)AscendC::GmAlloc(B * S1 * N * Hckv * sizeof(half));
    uint8_t *queryRopeOut = (uint8_t *)AscendC::GmAlloc(B * S1 * N * Dr * sizeof(half));
    uint8_t *dequantScaleQNopeOut = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(2048 * 2048 * 2048);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(sizeof(optiling::MlaPrologBaseParams));

    // ===================tiligndata赋值====================
    optiling::MlaPrologBaseParams *baseParams_ = reinterpret_cast<optiling::MlaPrologBaseParams *>(tiling);
    uint32_t aicNum_ = 12;
    uint32_t aivNum_ = 12;
    baseParams_->batchSize = B;
    baseParams_->seq1Size = S1;
    baseParams_->headSizeKr = Dr;
    baseParams_->numHeadSize = N;
    baseParams_->numHeadKvSize = Nkv;
    baseParams_->dimHeadSizeQc = D;
    baseParams_->dimHeadRope = Dr;
    baseParams_->headSizeX = He;
    baseParams_->headSizeCq = Hcq;
    baseParams_->headSizeCkv = Hckv;
    baseParams_->dtileSize = Dtile;
    baseParams_->blockNum = Block_Number;
    baseParams_->blockSize = Block_Size;
    baseParams_->tokenSize = baseParams_->batchSize * baseParams_->seq1Size;
    baseParams_->stepBatchSize = std::min(64U, baseParams_->tokenSize);
    baseParams_->stepNumHeadDequant = std::min(64U, N);
    baseParams_->mSubCoreNum = baseParams_->tokenSize - (baseParams_->mSubSize - 1U) * aicNum_;
    baseParams_->mSubSize = (baseParams_->tokenSize + aicNum_ - 1U) / aicNum_;
    baseParams_->seq2Size = baseParams_->numHeadKvSize;
    baseParams_->headSizeQc = baseParams_->dimHeadSizeQc * baseParams_->numHeadSize;
    baseParams_->headSizeQr = baseParams_->headSizeKr * baseParams_->numHeadSize;
    baseParams_->mm1SingleCoreN = optiling::CalcSingleCoreNV2(baseParams_->headSizeCq, aicNum_, Block_Size / 1);
    baseParams_->mm2SingleCoreN = 64U;
    baseParams_->mm3SingleCoreN = optiling::CalcSingleCoreNV2(
        baseParams_->numHeadSize * (baseParams_->headSizeKr + baseParams_->dimHeadSizeQc), aicNum_, D + Dr);
    baseParams_->mm4SingleCoreBatch = CeilDiv(baseParams_->numHeadSize, aicNum_);
    baseParams_->mm1BlockNum = CeilDiv(baseParams_->headSizeCq, baseParams_->mm1SingleCoreN);
    baseParams_->mm2BlockNum = (baseParams_->headSizeCkv + baseParams_->dimHeadRope) / baseParams_->mm2SingleCoreN;
    baseParams_->mm3BlockNum = CeilDiv(
        baseParams_->numHeadSize * (baseParams_->headSizeKr + baseParams_->dimHeadSizeQc), baseParams_->mm3SingleCoreN);
    baseParams_->mm4BlockNum = CeilDiv(baseParams_->numHeadSize, baseParams_->mm4SingleCoreBatch);
    baseParams_->vectorBlockNum = std::min(std::min(128U, baseParams_->tokenSize), aivNum_);
    baseParams_->reciprocalCq = 1.0f / baseParams_->headSizeCq;
    baseParams_->epsilonCq = 0.00461f;
    baseParams_->reciprocalCkv = 1.0f / baseParams_->headSizeCkv;
    baseParams_->epsilonCkv = 0.001166f;
    // v3
    baseParams_->queryNormFlag = 0U;
    baseParams_->kvQuantMode = static_cast<uint32_t>(2);
    baseParams_->ckvkrRepoMode = static_cast<uint32_t>(0);
    baseParams_->quantScaleRepoMode = static_cast<uint32_t>(0);
    baseParams_->tileSize = baseParams_->batchSize * baseParams_->stepBatchSize;
    baseParams_->qcQrScale = 1U;
    baseParams_->kcScale = 1U;
    baseParams_->isQcQrScaleEnable =
        static_cast<uint16_t>(std::abs(baseParams_->qcQrScale - 1.0f) >= std::numeric_limits<float>::epsilon());
    baseParams_->isKcScaleEnable =
        static_cast<uint16_t>(std::abs(baseParams_->kcScale - 1.0f) >= std::numeric_limits<float>::epsilon());

    ICPU_RUN_KF(func, blockDim, tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv,
                ropeSin, ropeCos, cacheIndex, kvCache, krCache, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut, nullptr,
                nullptr, dequantScaleQNopeOut, workspace, tiling);

    MlaPrologV2Kernel::FreeAllGmMemory(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
        cacheIndex, kvCache, krCache, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr,
        quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut, dequantScaleQNopeOut, workspace, tiling);
}

// 全量化kv非量化场景
TEST_F(MlaPrologV2Kernel, test_case_v2_QuantKVNoQuant)
{
    uint32_t B = 1;
    uint32_t N = 8;
    uint32_t N2 = 1;
    uint32_t D = 128;
    uint32_t Dr = 63;
    uint32_t S1 = 1;
    uint32_t S2 = 0;
    uint32_t He = 7168;
    uint32_t Hckv = 512;
    uint32_t T = 0;
    uint32_t Block_Number = 3;
    uint32_t Block_Size = 64;
    uint32_t Nkv = 1;
    uint32_t Dtile = 656;
    uint32_t blockDim = 12;
    uint32_t Hcq = 1536;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF) {
        return mla_prolog_v2<1, // CACHE_MODE
                             2, // SCENARIO
                             3, // QUANT_MODE
                             0, // ENABLE_DEQUANT_OPTIONAL
                             1, // ENABLE_GROUP_COMPUTE_OPTIONAL
                             0, // EMPTY_TENSOR_MODE
                             0, // ACTUAL_SEQ_LEN_MODE
                             0, // SPLIT_M_MODE
                             1, // ENABLE_ROPE
                             7  // CV_MODE
                             >(PARAM_LIST);
    };
    // 输入变量
    uint8_t *tokenX = (uint8_t *)AscendC::GmAlloc(B * S1 * He * sizeof(half));
    uint8_t *weightDq = (uint8_t *)AscendC::GmAlloc(He * Hcq * sizeof(half));
    uint8_t *weightUqQr = (uint8_t *)AscendC::GmAlloc(Hcq * N * (D + Dr) * sizeof(int8_t));
    uint8_t *weightUk = (uint8_t *)AscendC::GmAlloc(N * D * Hckv * sizeof(half));
    uint8_t *weightDkvKr = (uint8_t *)AscendC::GmAlloc(He * (Hckv + Dr) * sizeof(half));
    uint8_t *rmsnormGammaCq = (uint8_t *)AscendC::GmAlloc(Hcq * sizeof(half));
    uint8_t *rmsnormGammaCkv = (uint8_t *)AscendC::GmAlloc(Hckv * sizeof(half));
    uint8_t *ropeSin = (uint8_t *)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t *ropeCos = (uint8_t *)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t *cacheIndex = (uint8_t *)AscendC::GmAlloc(B * S1 * sizeof(half));
    uint8_t *kvCache = (uint8_t *)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Dtile * sizeof(int8_t));
    uint8_t *krCache = (uint8_t *)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Dr * sizeof(half));
    uint8_t *dequantScaleX = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *dequantScaleWDq = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *dequantScaleWUqQr = (uint8_t *)AscendC::GmAlloc(1 * N * (D + Dr) * sizeof(float));
    uint8_t *dequantScaleWDkvKr = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *quantScaleCkv = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *quantScaleCkr = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *smoothScalesCq = (uint8_t *)AscendC::GmAlloc(1 * Hcq * sizeof(float));
    // 输出类变量
    uint8_t *queryOut = (uint8_t *)AscendC::GmAlloc(B * S1 * N * Hckv * sizeof(half));
    uint8_t *queryRopeOut = (uint8_t *)AscendC::GmAlloc(B * S1 * N * Dr * sizeof(half));
    uint8_t *dequantScaleQNopeOut = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(2048 * 2048 * 2048);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(sizeof(optiling::MlaPrologBaseParams));

    // ===================tiligndata赋值====================
    optiling::MlaPrologBaseParams *baseParams_ = reinterpret_cast<optiling::MlaPrologBaseParams *>(tiling);
    uint32_t aicNum_ = 12;
    uint32_t aivNum_ = 12;
    baseParams_->batchSize = B;
    baseParams_->seq1Size = S1;
    baseParams_->headSizeKr = Dr;
    baseParams_->numHeadSize = N;
    baseParams_->numHeadKvSize = Nkv;
    baseParams_->dimHeadSizeQc = D;
    baseParams_->dimHeadRope = Dr;
    baseParams_->headSizeX = He;
    baseParams_->headSizeCq = Hcq;
    baseParams_->headSizeCkv = Hckv;
    baseParams_->dtileSize = Dtile;
    baseParams_->blockNum = Block_Number;
    baseParams_->blockSize = Block_Size;
    baseParams_->tokenSize = baseParams_->batchSize * baseParams_->seq1Size;
    baseParams_->stepBatchSize = std::min(64U, baseParams_->tokenSize);
    baseParams_->stepNumHeadDequant = std::min(64U, N);
    baseParams_->mSubSize = (baseParams_->tokenSize + aicNum_ - 1U) / aicNum_;
    baseParams_->mSubCoreNum = baseParams_->tokenSize - (baseParams_->mSubSize - 1U) * aicNum_;
    baseParams_->seq2Size = baseParams_->numHeadKvSize;
    baseParams_->headSizeQc = baseParams_->dimHeadSizeQc * baseParams_->numHeadSize;
    baseParams_->headSizeQr = baseParams_->headSizeKr * baseParams_->numHeadSize;
    baseParams_->mm1SingleCoreN = optiling::CalcSingleCoreNV2(baseParams_->headSizeCq, aicNum_, Block_Size / 1);
    baseParams_->mm2SingleCoreN = 64U;
    baseParams_->mm3SingleCoreN = optiling::CalcSingleCoreNV2(
        baseParams_->numHeadSize * (baseParams_->headSizeKr + baseParams_->dimHeadSizeQc), aicNum_, D + Dr);
    baseParams_->mm4SingleCoreBatch = CeilDiv(baseParams_->numHeadSize, aicNum_);
    baseParams_->mm1BlockNum = CeilDiv(baseParams_->headSizeCq, baseParams_->mm1SingleCoreN);
    baseParams_->mm2BlockNum = (baseParams_->headSizeCkv + baseParams_->dimHeadRope) / baseParams_->mm2SingleCoreN;
    baseParams_->mm3BlockNum = CeilDiv(
        baseParams_->numHeadSize * (baseParams_->headSizeKr + baseParams_->dimHeadSizeQc), baseParams_->mm3SingleCoreN);
    baseParams_->mm4BlockNum = CeilDiv(baseParams_->numHeadSize, baseParams_->mm4SingleCoreBatch);
    baseParams_->vectorBlockNum = std::min(std::min(128U, baseParams_->tokenSize), aivNum_);
    baseParams_->reciprocalCq = 1.0f / baseParams_->headSizeCq;
    baseParams_->epsilonCq = 0.00461f;
    baseParams_->reciprocalCkv = 1.0f / baseParams_->headSizeCkv;
    baseParams_->epsilonCkv = 0.001166f;
    // v3
    baseParams_->queryNormFlag = 0U;
    baseParams_->kvQuantMode = static_cast<uint32_t>(0);
    baseParams_->ckvkrRepoMode = static_cast<uint32_t>(0);
    baseParams_->quantScaleRepoMode = static_cast<uint32_t>(0);
    baseParams_->tileSize = baseParams_->batchSize * baseParams_->stepBatchSize;
    baseParams_->qcQrScale = 1.0f;
    baseParams_->kcScale = 1.0f;
    baseParams_->isQcQrScaleEnable =
        static_cast<uint16_t>(std::abs(baseParams_->qcQrScale - 1.0f) >= std::numeric_limits<float>::epsilon());
    baseParams_->isKcScaleEnable =
        static_cast<uint16_t>(std::abs(baseParams_->kcScale - 1.0f) >= std::numeric_limits<float>::epsilon());

    ICPU_RUN_KF(func, blockDim, tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv,
                ropeSin, ropeCos, cacheIndex, kvCache, krCache, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut, nullptr,
                nullptr, dequantScaleQNopeOut, workspace, tiling);

    MlaPrologV2Kernel::FreeAllGmMemory(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
        cacheIndex, kvCache, krCache, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr,
        quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut, dequantScaleQNopeOut, workspace, tiling);
}

// 全量化kv量化场景
TEST_F(MlaPrologV2Kernel, test_case_v2_QuantKVQuant)
{
    uint32_t B = 1;
    uint32_t N = 8;
    uint32_t N2 = 1;
    uint32_t D = 128;
    uint32_t Dr = 63;
    uint32_t S1 = 1;
    uint32_t S2 = 0;
    uint32_t He = 7168;
    uint32_t Hckv = 512;
    uint32_t T = 0;
    uint32_t Block_Number = 3;
    uint32_t Block_Size = 64;
    uint32_t Nkv = 1;
    uint32_t Dtile = 656;
    uint32_t blockDim = 12;
    uint32_t Hcq = 1536;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF) {
        return mla_prolog_v2<1, // CACHE_MODE
                             2, // SCENARIO
                             4, // QUANT_MODE
                             0, // ENABLE_DEQUANT_OPTIONAL
                             1, // ENABLE_GROUP_COMPUTE_OPTIONAL
                             0, // EMPTY_TENSOR_MODE
                             0, // ACTUAL_SEQ_LEN_MODE
                             0, // SPLIT_M_MODE
                             1, // ENABLE_ROPE
                             7  // CV_MODE
                             >(PARAM_LIST);
    };
    // 输入变量
    uint8_t *tokenX = (uint8_t *)AscendC::GmAlloc(B * S1 * He * sizeof(half));
    uint8_t *weightDq = (uint8_t *)AscendC::GmAlloc(He * Hcq * sizeof(half));
    uint8_t *weightUqQr = (uint8_t *)AscendC::GmAlloc(Hcq * N * (D + Dr) * sizeof(int8_t));
    uint8_t *weightUk = (uint8_t *)AscendC::GmAlloc(N * D * Hckv * sizeof(half));
    uint8_t *weightDkvKr = (uint8_t *)AscendC::GmAlloc(He * (Hckv + Dr) * sizeof(half));
    uint8_t *rmsnormGammaCq = (uint8_t *)AscendC::GmAlloc(Hcq * sizeof(half));
    uint8_t *rmsnormGammaCkv = (uint8_t *)AscendC::GmAlloc(Hckv * sizeof(half));
    uint8_t *ropeSin = (uint8_t *)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t *ropeCos = (uint8_t *)AscendC::GmAlloc(B * S1 * Dr * sizeof(half));
    uint8_t *cacheIndex = (uint8_t *)AscendC::GmAlloc(B * S1 * sizeof(half));
    uint8_t *kvCache = (uint8_t *)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Dtile * sizeof(int8_t));
    uint8_t *krCache = (uint8_t *)AscendC::GmAlloc(Block_Number * Block_Size * Nkv * Dr * sizeof(half));
    uint8_t *dequantScaleX = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *dequantScaleWDq = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *dequantScaleWUqQr = (uint8_t *)AscendC::GmAlloc(1 * N * (D + Dr) * sizeof(float));
    uint8_t *dequantScaleWDkvKr = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *quantScaleCkv = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *quantScaleCkr = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *smoothScalesCq = (uint8_t *)AscendC::GmAlloc(1 * Hcq * sizeof(float));
    // 输出类变量
    uint8_t *queryOut = (uint8_t *)AscendC::GmAlloc(B * S1 * N * Hckv * sizeof(half));
    uint8_t *queryRopeOut = (uint8_t *)AscendC::GmAlloc(B * S1 * N * Dr * sizeof(half));
    uint8_t *dequantScaleQNopeOut = (uint8_t *)AscendC::GmAlloc(1);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(2048 * 2048 * 2048);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(sizeof(optiling::MlaPrologBaseParams));

    // ===================tiligndata赋值====================
    optiling::MlaPrologBaseParams *baseParams_ = reinterpret_cast<optiling::MlaPrologBaseParams *>(tiling);
    uint32_t aicNum_ = 12;
    uint32_t aivNum_ = 12;
    baseParams_->batchSize = B;
    baseParams_->seq1Size = S1;
    baseParams_->headSizeKr = Dr;
    baseParams_->numHeadSize = N;
    baseParams_->numHeadKvSize = Nkv;
    baseParams_->dimHeadSizeQc = D;
    baseParams_->dimHeadRope = Dr;
    baseParams_->headSizeX = He;
    baseParams_->headSizeCq = Hcq;
    baseParams_->headSizeCkv = Hckv;
    baseParams_->dtileSize = Dtile;
    baseParams_->blockNum = Block_Number;
    baseParams_->blockSize = Block_Size;
    baseParams_->tokenSize = baseParams_->batchSize * baseParams_->seq1Size;
    baseParams_->stepBatchSize = std::min(64U, baseParams_->tokenSize);
    baseParams_->stepNumHeadDequant = std::min(64U, N);
    baseParams_->mSubCoreNum = baseParams_->tokenSize - (baseParams_->mSubSize - 1U) * aicNum_;
    baseParams_->mSubSize = (baseParams_->tokenSize + aicNum_ - 1U) / aicNum_;
    baseParams_->seq2Size = baseParams_->numHeadKvSize;
    baseParams_->headSizeQc = baseParams_->dimHeadSizeQc * baseParams_->numHeadSize;
    baseParams_->headSizeQr = baseParams_->headSizeKr * baseParams_->numHeadSize;
    baseParams_->mm1SingleCoreN = optiling::CalcSingleCoreNV2(baseParams_->headSizeCq, aicNum_, Block_Size / 1);
    baseParams_->mm2SingleCoreN = 64U;
    baseParams_->mm3SingleCoreN = optiling::CalcSingleCoreNV2(
        baseParams_->numHeadSize * (baseParams_->headSizeKr + baseParams_->dimHeadSizeQc), aicNum_, D + Dr);
    baseParams_->mm4SingleCoreBatch = CeilDiv(baseParams_->numHeadSize, aicNum_);
    baseParams_->mm1BlockNum = CeilDiv(baseParams_->headSizeCq, baseParams_->mm1SingleCoreN);
    baseParams_->mm2BlockNum = (baseParams_->headSizeCkv + baseParams_->dimHeadRope) / baseParams_->mm2SingleCoreN;
    baseParams_->mm3BlockNum = CeilDiv(
        baseParams_->numHeadSize * (baseParams_->headSizeKr + baseParams_->dimHeadSizeQc), baseParams_->mm3SingleCoreN);
    baseParams_->mm4BlockNum = CeilDiv(baseParams_->numHeadSize, baseParams_->mm4SingleCoreBatch);
    baseParams_->vectorBlockNum = std::min(std::min(128U, baseParams_->tokenSize), aivNum_);
    baseParams_->reciprocalCq = 1.0f / baseParams_->headSizeCq;
    baseParams_->epsilonCq = 0.00461f;
    baseParams_->reciprocalCkv = 1.0f / baseParams_->headSizeCkv;
    baseParams_->epsilonCkv = 0.001166f;
    // v3
    baseParams_->queryNormFlag = 0U;
    baseParams_->kvQuantMode = static_cast<uint32_t>(1);
    baseParams_->ckvkrRepoMode = static_cast<uint32_t>(0);
    baseParams_->quantScaleRepoMode = static_cast<uint32_t>(0);
    baseParams_->tileSize = baseParams_->batchSize * baseParams_->stepBatchSize;
    baseParams_->qcQrScale = 1U;
    baseParams_->kcScale = 1U;
    baseParams_->isQcQrScaleEnable =
        static_cast<uint16_t>(std::abs(baseParams_->qcQrScale - 1.0f) >= std::numeric_limits<float>::epsilon());
    baseParams_->isKcScaleEnable =
        static_cast<uint16_t>(std::abs(baseParams_->kcScale - 1.0f) >= std::numeric_limits<float>::epsilon());

    ICPU_RUN_KF(func, blockDim, tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv,
                ropeSin, ropeCos, cacheIndex, kvCache, krCache, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut, nullptr,
                nullptr, dequantScaleQNopeOut, workspace, tiling);

    MlaPrologV2Kernel::FreeAllGmMemory(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
        cacheIndex, kvCache, krCache, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr,
        quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut, dequantScaleQNopeOut, workspace, tiling);
}
