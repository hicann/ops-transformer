/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;


extern "C" __global__ __aicore__ void
mla_preprocess(GM_ADDR hiddenStateGm, GM_ADDR gamma1Gm, GM_ADDR beta1Gm, GM_ADDR quantScale1Gm, GM_ADDR quantOffset1Gm,
               GM_ADDR wdqkvGm, GM_ADDR descale1Gm, GM_ADDR bias1Gm, GM_ADDR gamma2Gm, GM_ADDR beta2Gm,
               GM_ADDR quantScale2Gm, GM_ADDR quantOffset2Gm, GM_ADDR wuqGm, GM_ADDR descale2Gm, GM_ADDR bias2Gm,
               GM_ADDR gamma3Gm, GM_ADDR cos1Gm, GM_ADDR sin1Gm, GM_ADDR wukGm, GM_ADDR keycacheGm,
               GM_ADDR keycacheRopeGm, GM_ADDR slotMappingGm, GM_ADDR gmCtkvScale, GM_ADDR gmQnopeScale, GM_ADDR qGm,
               GM_ADDR keycacheOutGm, GM_ADDR qGm2, GM_ADDR keycacheOutGm2, GM_ADDR workspace, GM_ADDR tiling);

class mla_preprocess_test : public testing::TestWithParam<bool> {
protected:
    static void SetUpTestCase()
    {
        cout << "mla_preprocess_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "mla_preprocess_test TearDown\n" << endl;
    }
};

TEST_P(mla_preprocess_test, test_case_0)
{
    const bool enableRope = GetParam();
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    size_t tokenNum = 8;
    size_t hiddenNum = 7168;
    size_t headNum = 32;
    size_t bloackNum = 192;
    size_t bloackSize = 128;

    size_t tiling_data_size = sizeof(MlaTilingData);

    size_t hiddenStateSize = tokenNum * hiddenNum * sizeof(half);
    size_t gamma1Size = hiddenNum * sizeof(half);
    size_t beta1Size = hiddenNum * sizeof(half);
    size_t quantScale1Size = 1 * sizeof(half);
    size_t quantOffset1Size = 1 * sizeof(int8_t);
    size_t wdqkvSize = 2112 * hiddenNum * sizeof(int8_t);
    size_t descale1Size = 2112 * sizeof(int64_t);
    size_t bias1Size = 2112 * sizeof(int32_t);
    size_t gamma2Size = 1536 * sizeof(half);
    size_t beta2Size = 1536 * sizeof(half);
    size_t quantScale2Size = 1 * sizeof(half);
    size_t quantOffset2Size = 1 * sizeof(int8_t);
    size_t wuqSize = headNum * 192 * 1536 * sizeof(int8_t);
    size_t descale2Size = headNum * 192 * sizeof(int64_t);
    size_t bias2Size = 1 * headNum * 192 * sizeof(int32_t);
    size_t gamma3Size = 512 * sizeof(half);
    size_t cos1Size = tokenNum * 64 * sizeof(half);
    size_t sin1Size = tokenNum * 64 * sizeof(half);
    size_t wukSize = headNum * 128 * 512 * sizeof(half);
    size_t keycacheSize = bloackNum * bloackSize * 1 * 512 * sizeof(half);
    size_t keycacheRopeSize = bloackNum * bloackSize * 1 * 64 * sizeof(half);
    size_t slotMappingSize = tokenNum * sizeof(half);
    size_t gmCtkvScaleSize = 1 * sizeof(half);
    size_t gmQnopeScaleSize = headNum * sizeof(half);
    size_t qGmSize = tokenNum * headNum * 1 * 512 * sizeof(half);
    size_t keycacheOutGmSize = bloackNum * bloackSize * 1 * 512 * sizeof(half);
    size_t qGm2Size = tokenNum * headNum * 1 * 64 * sizeof(half);
    size_t keycacheOutGm2Size = bloackNum * bloackSize * 1 * 64 * sizeof(half);

    uint8_t *hiddenStateGm = (uint8_t *)AscendC::GmAlloc(hiddenStateSize);
    uint8_t *gamma1Gm = (uint8_t *)AscendC::GmAlloc(gamma1Size);
    uint8_t *beta1Gm = (uint8_t *)AscendC::GmAlloc(beta1Size);
    uint8_t *quantScale1Gm = (uint8_t *)AscendC::GmAlloc(quantScale1Size);
    uint8_t *quantOffset1Gm = (uint8_t *)AscendC::GmAlloc(quantOffset1Size);
    uint8_t *wdqkvGm = (uint8_t *)AscendC::GmAlloc(wdqkvSize);
    uint8_t *descale1Gm = (uint8_t *)AscendC::GmAlloc(descale1Size);
    uint8_t *bias1Gm = (uint8_t *)AscendC::GmAlloc(bias1Size);
    uint8_t *gamma2Gm = (uint8_t *)AscendC::GmAlloc(gamma2Size);
    uint8_t *beta2Gm = (uint8_t *)AscendC::GmAlloc(beta2Size);
    uint8_t *quantScale2Gm = (uint8_t *)AscendC::GmAlloc(quantScale2Size);
    uint8_t *quantOffset2Gm = (uint8_t *)AscendC::GmAlloc(quantOffset2Size);
    uint8_t *wuqGm = (uint8_t *)AscendC::GmAlloc(wuqSize);
    uint8_t *descale2Gm = (uint8_t *)AscendC::GmAlloc(descale2Size);
    uint8_t *bias2Gm = (uint8_t *)AscendC::GmAlloc(bias2Size);
    uint8_t *gamma3Gm = (uint8_t *)AscendC::GmAlloc(gamma3Size);
    uint8_t *cos1Gm = (uint8_t *)AscendC::GmAlloc(cos1Size);
    uint8_t *sin1Gm = (uint8_t *)AscendC::GmAlloc(sin1Size);
    uint8_t *wukGm = (uint8_t *)AscendC::GmAlloc(wukSize);
    uint8_t *keycacheGm = (uint8_t *)AscendC::GmAlloc(keycacheSize);
    uint8_t *keycacheRopeGm = (uint8_t *)AscendC::GmAlloc(keycacheRopeSize);
    uint8_t *slotMappingGm = (uint8_t *)AscendC::GmAlloc(slotMappingSize);
    uint8_t *gmCtkvScale = (uint8_t *)AscendC::GmAlloc(gmCtkvScaleSize);
    uint8_t *gmQnopeScale = (uint8_t *)AscendC::GmAlloc(gmQnopeScaleSize);
    uint8_t *qGm = (uint8_t *)AscendC::GmAlloc(qGmSize);
    uint8_t *keycacheOutGm = (uint8_t *)AscendC::GmAlloc(keycacheOutGmSize);
    uint8_t *qGm2 = (uint8_t *)AscendC::GmAlloc(qGm2Size);
    uint8_t *keycacheOutGm2 = (uint8_t *)AscendC::GmAlloc(keycacheOutGm2Size);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(17956864);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 24;

    char *path_ = get_current_dir_name();
    string path(path_);

    MlaTilingData *tilingData = reinterpret_cast<MlaTilingData *>(tiling);
    tilingData->numCore = 24;
    tilingData->n = 8;
    tilingData->perTaskNum = 0;
    tilingData->resTaskNum = 0;
    tilingData->mm1.numBatch = 1;
    tilingData->mm1.m = 8;
    tilingData->mm1.k = 7168;
    tilingData->mm1.n = 2112;
    tilingData->mm1.m0 = 16;
    tilingData->mm1.k0 = 2048;
    tilingData->mm1.n0 = 96;
    tilingData->mm1.mLoop = 1;
    tilingData->mm1.kLoop = 4;
    tilingData->mm1.nLoop = 22;
    tilingData->mm1.coreLoop = 22;
    tilingData->mm1.swizzleCount = 2;
    tilingData->mm1.swizzleDirect = 1;
    tilingData->mm1.blockDim = 22;
    tilingData->mm1.enShuffleK = 0;
    tilingData->mm1.enLoadAllAmat = 0;
    tilingData->mm1.b0matPingPongBufferLen = 262144;

    tilingData->mm2.numBatch = 1;
    tilingData->mm2.m = 8;
    tilingData->mm2.k = 1536;
    tilingData->mm2.n = 6144;
    tilingData->mm2.m0 = 16;
    tilingData->mm2.k0 = 512;
    tilingData->mm2.n0 = 256;
    tilingData->mm2.mLoop = 1;
    tilingData->mm2.kLoop = 3;
    tilingData->mm2.nLoop = 24;
    tilingData->mm2.coreLoop = 24;
    tilingData->mm2.swizzleCount = 1;
    tilingData->mm2.swizzleDirect = 1;
    tilingData->mm2.blockDim = 24;
    tilingData->mm2.enShuffleK = 0;
    tilingData->mm2.enLoadAllAmat = 0;
    tilingData->mm2.b0matPingPongBufferLen = 262144;

    tilingData->mm3.numBatch = 32;
    tilingData->mm3.m = 8;
    tilingData->mm3.k = 128;
    tilingData->mm3.n = 512;
    tilingData->mm3.m0 = 16;
    tilingData->mm3.k0 = 240;
    tilingData->mm3.n0 = 512;
    tilingData->mm3.mLoop = 1;
    tilingData->mm3.kLoop = 1;
    tilingData->mm3.nLoop = 1;
    tilingData->mm3.coreLoop = 32;
    tilingData->mm3.swizzleCount = 1;
    tilingData->mm3.swizzleDirect = 0;
    tilingData->mm3.blockDim = 24;
    tilingData->mm3.enShuffleK = 0;
    tilingData->mm3.enLoadAllAmat = 0;
    tilingData->mm3.b0matPingPongBufferLen = 262144;

    tilingData->rmsNumCore1 = 48;
    tilingData->rmsNumCol1 = 7168;
    tilingData->rmsNumRow1 = 8;
    tilingData->rmsQuantMin1 = -128;
    tilingData->hiddtenState = 7168;
    tilingData->rmsNumCore2 = 48;
    tilingData->rmsNumCol2 = 2112;
    tilingData->rmsNumRow2 = 8;
    tilingData->rmsQuantMin2 = -128;
    tilingData->hiddenSizeQ = 2048;
    tilingData->headNumQ = 32;
    tilingData->headDim = 64;
    tilingData->concatSize = 512;
    tilingData->rotaryCoeff = 2;
    tilingData->ntokens = 6;
    tilingData->realCore = 43;
    tilingData->nlCoreRun = 6;
    tilingData->lCoreRun = 6;
    tilingData->maxNPerLoopForUb = 73;
    tilingData->preCoreLoopTime = 1;
    tilingData->preCoreLoopNLast = 6;
    tilingData->lastCoreLoopTime = 1;
    tilingData->lastCoreLoopNLast = 4;
    tilingData->esqFrontCore = 8;
    tilingData->esqTailCore = 40;
    tilingData->esqFrontCoreBatch = 1;
    tilingData->esqTailCoreBatch = 0;
    tilingData->esqHeadNum = 32;
    tilingData->esqColNum = 512;
    tilingData->esqUbHeadLoop = 0;
    tilingData->esqHeadPerLoop = 64;
    tilingData->esqHeadTail = 32;
    tilingData->esqColLoop = 4;
    tilingData->esqColTail = 0;
    tilingData->maxWorkspaceSize = 393216;
    tilingData->epsilon = 1e-5f;
    tilingData->doRmsNorm = true;
    tilingData->qDownOutFlag = false;
    tilingData->enableRope = enableRope;

    ICPU_SET_TILING_KEY(56);
    ICPU_RUN_KF(mla_preprocess, blockDim, hiddenStateGm, gamma1Gm, beta1Gm, quantScale1Gm, quantOffset1Gm, wdqkvGm,
                descale1Gm, bias1Gm, gamma2Gm, beta2Gm, quantScale2Gm, quantOffset2Gm, wuqGm, descale2Gm, bias2Gm,
                gamma3Gm, enableRope ? cos1Gm : nullptr, enableRope ? sin1Gm : nullptr, wukGm, keycacheGm,
                keycacheRopeGm, slotMappingGm, gmCtkvScale, gmQnopeScale, qGm, keycacheOutGm, qGm2, keycacheOutGm2,
                workspace, (uint8_t *)(tilingData));

    AscendC::GmFree(hiddenStateGm);
    AscendC::GmFree(gamma1Gm);
    AscendC::GmFree(beta1Gm);
    AscendC::GmFree(quantScale1Gm);
    AscendC::GmFree(quantOffset1Gm);
    AscendC::GmFree(wdqkvGm);
    AscendC::GmFree(descale1Gm);
    AscendC::GmFree(bias1Gm);
    AscendC::GmFree(gamma2Gm);
    AscendC::GmFree(beta2Gm);
    AscendC::GmFree(quantScale2Gm);
    AscendC::GmFree(quantOffset2Gm);
    AscendC::GmFree(wuqGm);

    AscendC::GmFree(descale2Gm);
    AscendC::GmFree(bias2Gm);
    AscendC::GmFree(gamma3Gm);
    AscendC::GmFree(cos1Gm);
    AscendC::GmFree(sin1Gm);
    AscendC::GmFree(wukGm);
    AscendC::GmFree(keycacheGm);
    AscendC::GmFree(keycacheRopeGm);
    AscendC::GmFree(slotMappingGm);
    AscendC::GmFree(gmCtkvScale);
    AscendC::GmFree(gmQnopeScale);
    AscendC::GmFree(qGm);
    AscendC::GmFree(keycacheOutGm);
    AscendC::GmFree(qGm2);
    AscendC::GmFree(keycacheOutGm2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

INSTANTIATE_TEST_SUITE_P(RopeModes, mla_preprocess_test, testing::Bool());
