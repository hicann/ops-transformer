/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "test_norm_rope_concat_grad.h"
#include "data_utils.h"
#include <filesystem>

#include <cstdint>

#define DTYPE_ROPE_SIN DT_FLOAT
#define DTYPE_QUERY DT_FLOAT

using namespace std;

// CPU模式下kernel最多支持32个输入输出tensor，超过部分统一设置uint32_t固定常数且避开使用
extern "C" __global__ __aicore__ void norm_rope_concat_grad(
    GM_ADDR gm_grad_query_output, GM_ADDR gm_grad_key_output, GM_ADDR gm_grad_value_output, GM_ADDR gm_query,
    GM_ADDR gm_key, GM_ADDR gm_encoder_query, GM_ADDR gm_encoder_key, GM_ADDR gm_norm_query_weight,
    GM_ADDR gm_norm_query_mean, GM_ADDR gm_norm_query_rstd, GM_ADDR gm_norm_key_weight, GM_ADDR gm_norm_key_mean,
    GM_ADDR gm_norm_key_rstd, GM_ADDR gm_norm_added_query_weight, GM_ADDR gm_norm_added_query_mean,
    GM_ADDR gm_norm_added_query_rstd, GM_ADDR gm_norm_added_key_weight, GM_ADDR gm_norm_added_key_mean,
    GM_ADDR gm_norm_added_key_rstd, GM_ADDR gm_rope_sin, GM_ADDR gm_rope_cos, GM_ADDR gm_grad_query,
    GM_ADDR gm_grad_key, GM_ADDR gm_grad_value, GM_ADDR gm_grad_encoder_query, GM_ADDR gm_grad_encoder_key,
    GM_ADDR gm_grad_encoder_value, GM_ADDR gm_grad_norm_query_weight, GM_ADDR gm_grad_norm_query_bias,
    GM_ADDR gm_grad_norm_key_weight, GM_ADDR gm_grad_norm_key_bias, uint32_t gm_grad_norm_added_query_weight,
    uint32_t gm_grad_norm_added_query_bias, uint32_t gm_grad_norm_added_key_weight,
    uint32_t gm_grad_norm_added_key_bias, GM_ADDR workspaceGM, GM_ADDR tiling);

class norm_rope_concat_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "norm_rope_concat_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "norm_rope_concat_grad_test TearDown\n" << endl;
    }
};
void run_norm_rope_concat_grad_case(uint64_t batchSize, uint64_t querySeq, uint64_t keySeq, uint64_t encoderQuerySeq,
                                    uint64_t encoderKeySeq, uint64_t ropeSeq, uint64_t headDim, uint64_t headNum,
                                    uint32_t usedCore, std::string& originDtype, std::string& caseName,
                                    uint64_t tiling_key)
{
    system("cp -r ../../../../../posembedding/norm_rope_concat_grad/tests/ut/op_kernel/norm_rope_concat_grad_data "
           "./");
    system("chmod -R 755 ./norm_rope_concat_grad_data/");
    system("cd ./norm_rope_concat_grad_data/ && rm -rf ./*bin");
    system("ls -lh ./norm_rope_concat_grad_data/");

    std::string gen_data_cmd = "cd ./norm_rope_concat_grad_data/ && python3 gen_data.py " + std::to_string(batchSize) +
                               std::string(" ") + std::to_string(headNum) + std::string(" ") + std::to_string(headDim) +
                               std::string(" ") + std::to_string(querySeq + encoderQuerySeq) + std::string(" ") +
                               std::to_string(keySeq + encoderKeySeq) + std::string(" ") + std::to_string(querySeq) +
                               std::string(" ") + std::to_string(keySeq) + std::string(" ") + std::to_string(ropeSeq) +
                               std::string(" ") + originDtype + std::string(" ") + std::to_string(tiling_key);
    system(gen_data_cmd.c_str());

    std::string gen_tiling_cmd = "cd ./norm_rope_concat_grad_data/ && python3 gen_tiling.py " + caseName;
    system(gen_tiling_cmd.c_str());
    system("ls -lh ./norm_rope_concat_grad_data/");

    uint64_t valueSeq = keySeq;
    uint64_t encoderValueSeq = encoderKeySeq;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t gradQueryOutputSize = (batchSize * headNum * (querySeq + encoderQuerySeq) * headDim) * sizeof(float);
    size_t gradKeyOutputSize = (batchSize * headNum * (keySeq + encoderKeySeq) * headDim) * sizeof(float);
    size_t gradValueOutputSize = (batchSize * headNum * (valueSeq + encoderValueSeq) * headDim) * sizeof(float);
    size_t querySize = (batchSize * querySeq * headNum * headDim) * sizeof(float);
    size_t keySize = (batchSize * keySeq * headNum * headDim) * sizeof(float);
    size_t encoderQuerySize = (batchSize * encoderQuerySeq * headNum * headDim) * sizeof(float);
    size_t encoderKeySize = (batchSize * encoderKeySeq * headNum * headDim) * sizeof(float);
    size_t normQueryWeightSize = (headDim) * sizeof(float);
    size_t normQueryMeanSize = (batchSize * querySeq * headNum * 1) * sizeof(float);
    size_t normQueryRstdSize = (batchSize * querySeq * headNum * 1) * sizeof(float);
    size_t normKeyWeightSize = (headDim) * sizeof(float);
    size_t normKeyMeanSize = (batchSize * keySeq * headNum * 1) * sizeof(float);
    size_t normKeyRstdSize = (batchSize * keySeq * headNum * 1) * sizeof(float);
    size_t normAddedQueryWeightSize = (headDim) * sizeof(float);
    size_t normAddedQueryMeanSize = (batchSize * encoderQuerySeq * headNum * 1) * sizeof(float);
    size_t normAddedQueryRstdSize = (batchSize * encoderQuerySeq * headNum * 1) * sizeof(float);
    size_t normAddedKeyWeightSize = (headDim) * sizeof(float);
    size_t normAddedKeyMeanSize = (batchSize * encoderKeySeq * headNum * 1) * sizeof(float);
    size_t normAddedKeyRstdSize = (batchSize * encoderKeySeq * headNum * 1) * sizeof(float);
    size_t ropeSinSize = (ropeSeq * headDim) * sizeof(float);
    size_t ropeCosSize = (ropeSeq * headDim) * sizeof(float);
    size_t gradQuerySize = (batchSize * querySeq * headNum * headDim) * sizeof(float);
    size_t gradKeySize = (batchSize * keySeq * headNum * headDim) * sizeof(float);
    size_t gradValueSize = (batchSize * valueSeq * headNum * headDim) * sizeof(float);
    size_t gradEncoderQuerySize = (batchSize * encoderQuerySeq * headNum * headDim) * sizeof(float);
    size_t gradEncoderKeySize = (batchSize * encoderKeySeq * headNum * headDim) * sizeof(float);
    size_t gradEncoderValueSize = (batchSize * encoderValueSeq * headNum * headDim) * sizeof(float);
    size_t gradNormQueryWeightSize = (headDim) * sizeof(float);
    size_t gradNormQueryBiasSize = (headDim) * sizeof(float);
    size_t gradNormKeyWeightSize = (headDim) * sizeof(float);
    size_t gradNormKeyBiasSize = (headDim) * sizeof(float);
    size_t gradNormAddedQueryWeightSize = (headDim) * sizeof(float);
    size_t gradNormAddedQueryBiasSize = (headDim) * sizeof(float);
    size_t gradNormAddedKeyWeightSize = (headDim) * sizeof(float);
    size_t gradNormAddedKeyBiasSize = (headDim) * sizeof(float);

    uint8_t* gradQueryOutput = (uint8_t*)AscendC::GmAlloc(gradQueryOutputSize);
    uint8_t* gradKeyOutput = (uint8_t*)AscendC::GmAlloc(gradKeyOutputSize);
    uint8_t* gradValueOutput = (uint8_t*)AscendC::GmAlloc(gradValueOutputSize);
    uint8_t* query = (uint8_t*)AscendC::GmAlloc(querySize);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* encoderQuery = (uint8_t*)AscendC::GmAlloc(encoderQuerySize);
    uint8_t* encoderKey = (uint8_t*)AscendC::GmAlloc(encoderKeySize);
    uint8_t* normQueryWeight = (uint8_t*)AscendC::GmAlloc(normQueryWeightSize);
    uint8_t* normQueryMean = (uint8_t*)AscendC::GmAlloc(normQueryMeanSize);
    uint8_t* normQueryRstd = (uint8_t*)AscendC::GmAlloc(normQueryRstdSize);
    uint8_t* normKeyWeight = (uint8_t*)AscendC::GmAlloc(normKeyWeightSize);
    uint8_t* normKeyMean = (uint8_t*)AscendC::GmAlloc(normKeyMeanSize);
    uint8_t* normKeyRstd = (uint8_t*)AscendC::GmAlloc(normKeyRstdSize);
    uint8_t* normAddedQueryWeight = (uint8_t*)AscendC::GmAlloc(normAddedQueryWeightSize);
    uint8_t* normAddedQueryMean = (uint8_t*)AscendC::GmAlloc(normAddedQueryMeanSize);
    uint8_t* normAddedQueryRstd = (uint8_t*)AscendC::GmAlloc(normAddedQueryRstdSize);
    uint8_t* normAddedKeyWeight = (uint8_t*)AscendC::GmAlloc(normAddedKeyWeightSize);
    uint8_t* normAddedKeyMean = (uint8_t*)AscendC::GmAlloc(normAddedKeyMeanSize);
    uint8_t* normAddedKeyRstd = (uint8_t*)AscendC::GmAlloc(normAddedKeyRstdSize);
    uint8_t* ropeSin = (uint8_t*)AscendC::GmAlloc(ropeSinSize);
    uint8_t* ropeCos = (uint8_t*)AscendC::GmAlloc(ropeCosSize);
    uint8_t* gradQuery = (uint8_t*)AscendC::GmAlloc(gradQuerySize);
    uint8_t* gradKey = (uint8_t*)AscendC::GmAlloc(gradKeySize);
    uint8_t* gradValue = (uint8_t*)AscendC::GmAlloc(gradValueSize);
    uint8_t* gradEncoderQuery = (uint8_t*)AscendC::GmAlloc(gradEncoderQuerySize);
    uint8_t* gradEncoderKey = (uint8_t*)AscendC::GmAlloc(gradEncoderKeySize);
    uint8_t* gradEncoderValue = (uint8_t*)AscendC::GmAlloc(gradEncoderValueSize);
    uint8_t* gradNormQueryWeight = (uint8_t*)AscendC::GmAlloc(gradNormQueryWeightSize);
    uint8_t* gradNormQueryBias = (uint8_t*)AscendC::GmAlloc(gradNormQueryBiasSize);
    uint8_t* gradNormKeyWeight = (uint8_t*)AscendC::GmAlloc(gradNormKeyWeightSize);
    uint8_t* gradNormKeyBias = (uint8_t*)AscendC::GmAlloc(gradNormKeyBiasSize);
    uint8_t* gradNormAddedQueryWeight = (uint8_t*)AscendC::GmAlloc(gradNormAddedQueryWeightSize);
    uint8_t* gradNormAddedQueryBias = (uint8_t*)AscendC::GmAlloc(gradNormAddedQueryBiasSize);
    uint8_t* gradNormAddedKeyWeight = (uint8_t*)AscendC::GmAlloc(gradNormAddedKeyWeightSize);
    uint8_t* gradNormAddedKeyBias = (uint8_t*)AscendC::GmAlloc(gradNormAddedKeyBiasSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(37814272 * 2);
    size_t tilingSize = sizeof(NormRopeConcatGradTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    NormRopeConcatGradTilingData* tilingDatafromBin = reinterpret_cast<NormRopeConcatGradTilingData*>(tiling);
    string curPath = "./";

    ReadFile(curPath + "norm_rope_concat_grad_data/gradQueryOutput.bin", gradQueryOutputSize, gradQueryOutput,
             gradQueryOutputSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/gradKeyOutput.bin", gradKeyOutputSize, gradKeyOutput,
             gradKeyOutputSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/gradValueOutput.bin", gradValueOutputSize, gradValueOutput,
             gradValueOutputSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/query.bin", querySize, query, querySize);
    ReadFile(curPath + "norm_rope_concat_grad_data/key.bin", keySize, key, keySize);
    ReadFile(curPath + "norm_rope_concat_grad_data/encoderQuery.bin", encoderQuerySize, encoderQuery, encoderQuerySize);
    ReadFile(curPath + "norm_rope_concat_grad_data/encoderKey.bin", encoderKeySize, encoderKey, encoderKeySize);
    ReadFile(curPath + "norm_rope_concat_grad_data/normQueryWeight.bin", normQueryWeightSize, normQueryWeight,
             normQueryWeightSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/normQueryMean.bin", normQueryMeanSize, normQueryMean,
             normQueryMeanSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/normQueryRstd.bin", normQueryRstdSize, normQueryRstd,
             normQueryRstdSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/normKeyWeight.bin", normKeyWeightSize, normKeyWeight,
             normKeyWeightSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/normKeyMean.bin", normKeyMeanSize, normKeyMean, normKeyMeanSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/normKeyRstd.bin", normKeyRstdSize, normKeyRstd, normKeyRstdSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/normAddedQueryWeight.bin", normAddedQueryWeightSize,
             normAddedQueryWeight, normAddedQueryWeightSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/normAddedQueryMean.bin", normAddedQueryMeanSize, normAddedQueryMean,
             normAddedQueryMeanSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/normAddedQueryRstd.bin", normAddedQueryRstdSize, normAddedQueryRstd,
             normAddedQueryRstdSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/normAddedKeyWeight.bin", normAddedKeyWeightSize, normAddedKeyWeight,
             normAddedKeyWeightSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/normAddedKeyMean.bin", normAddedKeyMeanSize, normAddedKeyMean,
             normAddedKeyMeanSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/normAddedKeyRstd.bin", normAddedKeyRstdSize, normAddedKeyRstd,
             normAddedKeyRstdSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/ropeSin.bin", ropeSinSize, ropeSin, ropeSinSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/ropeCos.bin", ropeCosSize, ropeCos, ropeCosSize);
    ReadFile(curPath + "norm_rope_concat_grad_data/tiling.bin", tilingSize, tilingDatafromBin, tilingSize);
    ICPU_SET_TILING_KEY(tiling_key);

    uint32_t blockDim = usedCore;
    ICPU_RUN_KF(norm_rope_concat_grad, blockDim, gradQueryOutput, gradKeyOutput, gradValueOutput, query, key,
                encoderQuery, encoderKey, normQueryWeight, normQueryMean, normQueryRstd, normKeyWeight, normKeyMean,
                normKeyRstd, normAddedQueryWeight, normAddedQueryMean, normAddedQueryRstd, normAddedKeyWeight,
                normAddedKeyMean, normAddedKeyRstd, ropeSin, ropeCos, gradQuery, gradKey, gradValue, gradEncoderQuery,
                gradEncoderKey, gradEncoderValue, gradNormQueryWeight, gradNormQueryBias, gradNormKeyWeight,
                gradNormKeyBias, 0, 0, 0, 0, workSpace, (uint8_t*)(tilingDatafromBin));

    WriteFile(curPath + "/norm_rope_concat_grad_data/gradQuery.bin", gradQuery, gradQuerySize);
    WriteFile(curPath + "/norm_rope_concat_grad_data/gradKey.bin", gradKey, gradKeySize);
    WriteFile(curPath + "/norm_rope_concat_grad_data/gradValue.bin", gradValue, gradValueSize);
    WriteFile(curPath + "/norm_rope_concat_grad_data/gradEncoderQuery.bin", gradEncoderQuery, gradEncoderQuerySize);
    WriteFile(curPath + "/norm_rope_concat_grad_data/gradEncoderKey.bin", gradEncoderKey, gradEncoderKeySize);
    WriteFile(curPath + "/norm_rope_concat_grad_data/gradEncoderValue.bin", gradEncoderValue, gradEncoderValueSize);
    WriteFile(curPath + "/norm_rope_concat_grad_data/gradNormQueryWeight.bin", gradNormQueryWeight,
              gradNormQueryWeightSize);
    WriteFile(curPath + "/norm_rope_concat_grad_data/gradNormQueryBias.bin", gradNormQueryBias, gradNormQueryBiasSize);
    WriteFile(curPath + "/norm_rope_concat_grad_data/gradNormKeyWeight.bin", gradNormKeyWeight, gradNormKeyWeightSize);
    WriteFile(curPath + "/norm_rope_concat_grad_data/gradNormKeyBias.bin", gradNormKeyBias, gradNormKeyBiasSize);
    system("ls -lh ./norm_rope_concat_grad_data/");

    std::string compare_cmd = "cd ./norm_rope_concat_grad_data/ && python3 compare_data.py " +
                              std::to_string(batchSize) + std::string(" ") + std::to_string(headNum) +
                              std::string(" ") + std::to_string(headDim) + std::string(" ") + std::to_string(querySeq) +
                              std::string(" ") + std::to_string(keySeq) + std::string(" ") +
                              std::to_string(encoderQuerySeq) + std::string(" ") + std::to_string(encoderKeySeq) +
                              std::string(" ") + originDtype + std::string(" ") + std::to_string(tiling_key);

    EXPECT_EQ(system(compare_cmd.c_str()), 0);

    AscendC::GmFree((void*)gradQueryOutput);
    AscendC::GmFree((void*)gradKeyOutput);
    AscendC::GmFree((void*)gradValueOutput);
    AscendC::GmFree((void*)query);
    AscendC::GmFree((void*)key);
    AscendC::GmFree((void*)encoderQuery);
    AscendC::GmFree((void*)encoderKey);
    AscendC::GmFree((void*)normQueryWeight);
    AscendC::GmFree((void*)normQueryMean);
    AscendC::GmFree((void*)normQueryRstd);
    AscendC::GmFree((void*)normKeyWeight);
    AscendC::GmFree((void*)normKeyMean);
    AscendC::GmFree((void*)normKeyRstd);
    AscendC::GmFree((void*)normAddedQueryWeight);
    AscendC::GmFree((void*)normAddedQueryMean);
    AscendC::GmFree((void*)normAddedQueryRstd);
    AscendC::GmFree((void*)normAddedKeyWeight);
    AscendC::GmFree((void*)normAddedKeyMean);
    AscendC::GmFree((void*)normAddedKeyRstd);
    AscendC::GmFree((void*)ropeSin);
    AscendC::GmFree((void*)ropeCos);
    AscendC::GmFree((void*)gradQuery);
    AscendC::GmFree((void*)gradKey);
    AscendC::GmFree((void*)gradValue);
    AscendC::GmFree((void*)gradEncoderQuery);
    AscendC::GmFree((void*)gradEncoderKey);
    AscendC::GmFree((void*)gradEncoderValue);
    AscendC::GmFree((void*)gradNormQueryWeight);
    AscendC::GmFree((void*)gradNormQueryBias);
    AscendC::GmFree((void*)gradNormKeyWeight);
    AscendC::GmFree((void*)gradNormKeyBias);
    AscendC::GmFree((void*)tiling);
}

TEST_F(norm_rope_concat_grad_test, test_case_fp32_001)
{
    uint64_t batchSize = 2;
    uint64_t querySeq = 4;
    uint64_t keySeq = 8;
    uint64_t encoderQuerySeq = 16;
    uint64_t encoderKeySeq = 32;
    uint64_t ropeSeq = 8;
    uint64_t headNum = 32;
    uint64_t headDim = 64;
    uint32_t usedCore = 32;
    std::string originDtype = "float32";
    std::string caseName = "case001";
    uint64_t tiling_key = 0;

    run_norm_rope_concat_grad_case(batchSize, querySeq, keySeq, encoderQuerySeq, encoderKeySeq, ropeSeq, headDim,
                                   headNum, usedCore, originDtype, caseName, tiling_key);
}
