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
 * \file test_norm_rope_concat.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_norm_rope_concat.h"
#include "data_utils.h"
#include "tiling_case_executor.h"

using namespace std;

extern "C" __global__ __aicore__ void norm_rope_concat(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR encoder_query, GM_ADDR encoder_key, GM_ADDR encoder_value,
    GM_ADDR norm_query_weight, GM_ADDR norm_query_bias, GM_ADDR norm_key_weight, GM_ADDR norm_key_bias,
    GM_ADDR norm_added_query_weight, GM_ADDR norm_added_query_bias, GM_ADDR norm_added_key_weight,
    GM_ADDR norm_added_key_bias, GM_ADDR rope_sin, GM_ADDR rope_cos, GM_ADDR query_output, GM_ADDR key_output,
    GM_ADDR value_output, GM_ADDR norm_query_mean, GM_ADDR norm_query_rstd, GM_ADDR norm_key_mean,
    GM_ADDR norm_key_rstd, GM_ADDR norm_added_query_mean, GM_ADDR norm_added_query_rstd, GM_ADDR norm_added_key_mean,
    GM_ADDR norm_added_key_rstd, GM_ADDR workspace, GM_ADDR tiling);

class norm_rope_concat_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "norm_rope_concat_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "norm_rope_concat_test TearDown\n" << endl;
    }
};

TEST_F(norm_rope_concat_test, test_case_0001)
{
    size_t querySize = 1 * 8 * 8 * 64 * sizeof(half);
    size_t keySize = 1 * 8 * 8 * 64 * sizeof(half);
    size_t valueSize = 1 * 8 * 8 * 64 * sizeof(half);
    size_t encoderQuerySize = 1 * 8 * 8 * 64 * sizeof(half);
    size_t encoderKeySize = 1 * 8 * 8 * 64 * sizeof(half);
    size_t encoderValueSize = 1 * 8 * 8 * 64 * sizeof(half);
    size_t normQueryWeightSize = 64 * sizeof(half);
    size_t normQueryBiasSize = 64 * sizeof(half);
    size_t normKeyWeightSize = 64 * sizeof(half);
    size_t normKeyBiasSize = 64 * sizeof(half);
    size_t normAddedQueryWeightSize = 64 * sizeof(half);
    size_t normAddedQueryBiasSize = 64 * sizeof(half);
    size_t normAddedKeyWeightSize = 64 * sizeof(half);
    size_t normAddedKeyBiasSize = 64 * sizeof(half);
    size_t ropeSinSize = 16 * 64 * sizeof(half);
    size_t ropeCosSize = 16 * 64 * sizeof(half);

    size_t queryOutSize = 1 * 8 * 16 * 64 * sizeof(half);
    size_t keyOutSize = 1 * 8 * 16 * 64 * sizeof(half);
    size_t valueOutSize = 1 * 8 * 16 * 64 * sizeof(half);
    size_t normQueryMeanSize = 8 * 8 * sizeof(half);
    size_t normQueryRstdSize = 8 * 8 * sizeof(half);
    size_t normKeyMeanSize = 8 * 8 * sizeof(half);
    size_t normKeyRstdSize = 8 * 8 * sizeof(half);
    size_t normAddedQueryMeanSize = 8 * 8 * sizeof(half);
    size_t normAddedQueryRstdSize = 8 * 8 * sizeof(half);
    size_t normAddedKeyMeanSize = 8 * 8 * sizeof(half);
    size_t normAddedKeyRstdSize = 8 * 8 * sizeof(half);

    size_t workspaceSize = 0;
    size_t tiling_data_size = sizeof(NormRopeConcatTilingData);
    uint32_t blockDim = 8;

    uint8_t* query = (uint8_t*)AscendC::GmAlloc(querySize);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* encoderQuery = (uint8_t*)AscendC::GmAlloc(encoderQuerySize);
    uint8_t* encoderKey = (uint8_t*)AscendC::GmAlloc(encoderKeySize);
    uint8_t* encoderValue = (uint8_t*)AscendC::GmAlloc(encoderValueSize);
    uint8_t* normQueryWeight = (uint8_t*)AscendC::GmAlloc(normQueryWeightSize);
    uint8_t* normQueryBias = (uint8_t*)AscendC::GmAlloc(normQueryBiasSize);
    uint8_t* normKeyWeight = (uint8_t*)AscendC::GmAlloc(normKeyWeightSize);
    uint8_t* normKeyBias = (uint8_t*)AscendC::GmAlloc(normKeyBiasSize);
    uint8_t* normAddedQueryWeight = (uint8_t*)AscendC::GmAlloc(normAddedQueryWeightSize);
    uint8_t* normAddedQueryBias = (uint8_t*)AscendC::GmAlloc(normAddedQueryBiasSize);
    uint8_t* normAddedKeyWeight = (uint8_t*)AscendC::GmAlloc(normAddedKeyWeightSize);
    uint8_t* normAddedKeyBias = (uint8_t*)AscendC::GmAlloc(normAddedKeyBiasSize);
    uint8_t* ropeSin = (uint8_t*)AscendC::GmAlloc(ropeSinSize);
    uint8_t* ropeCos = (uint8_t*)AscendC::GmAlloc(ropeCosSize);

    uint8_t* queryOut = (uint8_t*)AscendC::GmAlloc(queryOutSize);
    uint8_t* keyOut = (uint8_t*)AscendC::GmAlloc(keyOutSize);
    uint8_t* valueOut = (uint8_t*)AscendC::GmAlloc(valueOutSize);
    uint8_t* normQueryMean = (uint8_t*)AscendC::GmAlloc(normQueryMeanSize);
    uint8_t* normQueryRstd = (uint8_t*)AscendC::GmAlloc(normQueryRstdSize);
    uint8_t* normKeyMean = (uint8_t*)AscendC::GmAlloc(normKeyMeanSize);
    uint8_t* normKeyRstd = (uint8_t*)AscendC::GmAlloc(normKeyRstdSize);
    uint8_t* normAddedQueryMean = (uint8_t*)AscendC::GmAlloc(normAddedQueryMeanSize);
    uint8_t* normAddedQueryRstd = (uint8_t*)AscendC::GmAlloc(normAddedQueryRstdSize);
    uint8_t* normAddedKeyMean = (uint8_t*)AscendC::GmAlloc(normAddedKeyMeanSize);
    uint8_t* normAddedKeyRstd = (uint8_t*)AscendC::GmAlloc(normAddedKeyRstdSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    NormRopeConcatTilingData* tilingDatafromBin = reinterpret_cast<NormRopeConcatTilingData*>(tiling);

    ICPU_SET_TILING_KEY(0);
    // ICPU_RUN_KF(interleave_rope, blockDim, x, cos, sin, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(query);
    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(encoderQuery);
    AscendC::GmFree(encoderKey);
    AscendC::GmFree(encoderValue);
    AscendC::GmFree(normQueryWeight);
    AscendC::GmFree(normQueryBias);
    AscendC::GmFree(normKeyWeight);
    AscendC::GmFree(normKeyBias);
    AscendC::GmFree(normAddedQueryWeight);
    AscendC::GmFree(normAddedQueryBias);
    AscendC::GmFree(normAddedKeyWeight);
    AscendC::GmFree(normAddedKeyBias);
    AscendC::GmFree(ropeSin);
    AscendC::GmFree(ropeCos);
    AscendC::GmFree(queryOut);
    AscendC::GmFree(keyOut);
    AscendC::GmFree(valueOut);
    AscendC::GmFree(normQueryMean);
    AscendC::GmFree(normQueryRstd);
    AscendC::GmFree(normKeyMean);
    AscendC::GmFree(normKeyRstd);
    AscendC::GmFree(normAddedQueryMean);
    AscendC::GmFree(normAddedQueryRstd);
    AscendC::GmFree(normAddedKeyMean);
    AscendC::GmFree(normAddedKeyRstd);
    AscendC::GmFree(tiling);
    free(path_);
}
