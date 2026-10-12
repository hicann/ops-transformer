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
 * \file test_gather_pa_kv_cache.cpp
 * \brief
 */
#include "data_utils.h"
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_gather_pa_kv_cache_tiling_def.h"

using namespace std;

extern "C" __global__ __aicore__ void gather_pa_kv_cache(GM_ADDR keyCacheInGm, GM_ADDR valueCacheInGm,
                                                         GM_ADDR blockTablesInGm, GM_ADDR seqLensInGm, GM_ADDR keyInGm,
                                                         GM_ADDR valueInGm, GM_ADDR seqOffsetInGm, GM_ADDR keyOutGm,
                                                         GM_ADDR valueOutGm, GM_ADDR workspace, GM_ADDR tiling);

class gather_pa_kv_cache_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "gather_pa_kv_cache_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "gather_pa_kv_cache_test TearDown\n" << endl;
    }
};

TEST_F(gather_pa_kv_cache_test, test_tiling_key_577)
{
    uint32_t keyCache_shape = 30 * 101 * 128 * 32;
    uint32_t valueCache_shape = 30 * 127 * 128 * 32;
    uint32_t blockTables_shape = 36 * 8;
    uint32_t seqLens_shape = 36;
    uint32_t key_shape = 18933 * 3232;
    uint32_t value_shape = 18933 * 4064;
    uint32_t seqOffset_shape = 1;

    size_t keyCache_size = keyCache_shape * sizeof(int8_t);
    size_t valueCache_size = valueCache_shape * sizeof(int8_t);
    size_t blockTables_size = blockTables_shape * sizeof(int32_t);
    size_t seqLens_size = seqLens_shape * sizeof(int32_t);
    size_t key_size = key_shape * sizeof(int8_t);
    size_t value_size = value_shape * sizeof(int8_t);
    size_t seqOffset_size = seqOffset_shape * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GatherPaKvCacheTilingDataTest);

    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCache_size);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCache_size);
    uint8_t* blockTables = (uint8_t*)AscendC::GmAlloc(blockTables_size);
    uint8_t* seqLens = (uint8_t*)AscendC::GmAlloc(seqLens_size);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t* seqOffset = (uint8_t*)AscendC::GmAlloc(seqOffset_size);
    uint8_t* keyOut = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* valueOut = (uint8_t*)AscendC::GmAlloc(value_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    GatherPaKvCacheTilingDataTest* tilingDatafromBin = reinterpret_cast<GatherPaKvCacheTilingDataTest*>(tiling);

    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->numTokens = 36;
    tilingDatafromBin->numblkTabCol = 8;
    tilingDatafromBin->tokenSizeK = 3232;
    tilingDatafromBin->tokenSizeV = 4064;
    tilingDatafromBin->typeByte = 1;
    tilingDatafromBin->hasSeqStarts = false;
    tilingDatafromBin->isSeqLensCumsum = false;

    ICPU_SET_TILING_KEY(577);
    ICPU_RUN_KF(gather_pa_kv_cache, blockDim, keyCache, valueCache, blockTables, seqLens, key, value, seqOffset, keyOut,
                valueOut, workspace, (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(blockTables);
    AscendC::GmFree(seqLens);
    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(seqOffset);
    AscendC::GmFree(keyOut);
    AscendC::GmFree(valueOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(gather_pa_kv_cache_test, test_tiling_key_619)
{
    uint32_t keyCache_shape = 128 * 128 * 16 * 144;
    uint32_t valueCache_shape = 128 * 128 * 16 * 128;
    uint32_t blockTables_shape = 16 * 12;
    uint32_t seqLens_shape = 16;
    uint32_t key_shape = 8931 * 16 * 144;
    uint32_t value_shape = 8931 * 16 * 128;
    uint32_t seqOffset_shape = 16;

    size_t keyCache_size = keyCache_shape * sizeof(half);
    size_t valueCache_size = valueCache_shape * sizeof(half);
    size_t blockTables_size = blockTables_shape * sizeof(int32_t);
    size_t seqLens_size = seqLens_shape * sizeof(int32_t);
    size_t key_size = key_shape * sizeof(half);
    size_t value_size = value_shape * sizeof(half);
    size_t seqOffset_size = seqOffset_shape * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GatherPaKvCacheTilingDataTest);

    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCache_size);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCache_size);
    uint8_t* blockTables = (uint8_t*)AscendC::GmAlloc(blockTables_size);
    uint8_t* seqLens = (uint8_t*)AscendC::GmAlloc(seqLens_size);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t* seqOffset = (uint8_t*)AscendC::GmAlloc(seqOffset_size);
    uint8_t* keyOut = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* valueOut = (uint8_t*)AscendC::GmAlloc(value_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    GatherPaKvCacheTilingDataTest* tilingDatafromBin = reinterpret_cast<GatherPaKvCacheTilingDataTest*>(tiling);

    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->numTokens = 16;
    tilingDatafromBin->numblkTabCol = 12;
    tilingDatafromBin->tokenSizeK = 16 * 144;
    tilingDatafromBin->tokenSizeV = 16 * 128;
    tilingDatafromBin->typeByte = 2;
    tilingDatafromBin->hasSeqStarts = false;
    tilingDatafromBin->isSeqLensCumsum = false;

    ICPU_SET_TILING_KEY(619);
    ICPU_RUN_KF(gather_pa_kv_cache, blockDim, keyCache, valueCache, blockTables, seqLens, key, value, seqOffset, keyOut,
                valueOut, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(blockTables);
    AscendC::GmFree(seqLens);
    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(seqOffset);
    AscendC::GmFree(keyOut);
    AscendC::GmFree(valueOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(gather_pa_kv_cache_test, test_tiling_key_618)
{
    uint32_t keyCache_shape = 128 * 128 * 16 * 144;
    uint32_t valueCache_shape = 128 * 128 * 16 * 128;
    uint32_t blockTables_shape = 16 * 12;
    uint32_t seqLens_shape = 16;
    uint32_t key_shape = 9547 * 16 * 144;
    uint32_t value_shape = 9547 * 16 * 128;
    uint32_t seqOffset_shape = 16;

    size_t keyCache_size = keyCache_shape * sizeof(int8_t);
    size_t valueCache_size = valueCache_shape * sizeof(int8_t);
    size_t blockTables_size = blockTables_shape * sizeof(int32_t);
    size_t seqLens_size = seqLens_shape * sizeof(int32_t);
    size_t key_size = key_shape * sizeof(int8_t);
    size_t value_size = value_shape * sizeof(int8_t);
    size_t seqOffset_size = seqOffset_shape * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GatherPaKvCacheTilingDataTest);

    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCache_size);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCache_size);
    uint8_t* blockTables = (uint8_t*)AscendC::GmAlloc(blockTables_size);
    uint8_t* seqLens = (uint8_t*)AscendC::GmAlloc(seqLens_size);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t* seqOffset = (uint8_t*)AscendC::GmAlloc(seqOffset_size);
    uint8_t* keyOut = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* valueOut = (uint8_t*)AscendC::GmAlloc(value_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    GatherPaKvCacheTilingDataTest* tilingDatafromBin = reinterpret_cast<GatherPaKvCacheTilingDataTest*>(tiling);

    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->numTokens = 16;
    tilingDatafromBin->numblkTabCol = 12;
    tilingDatafromBin->tokenSizeK = 2304;
    tilingDatafromBin->tokenSizeV = 2048;
    tilingDatafromBin->typeByte = 1;
    tilingDatafromBin->hasSeqStarts = false;
    tilingDatafromBin->isSeqLensCumsum = false;

    ICPU_SET_TILING_KEY(618);
    ICPU_RUN_KF(gather_pa_kv_cache, blockDim, keyCache, valueCache, blockTables, seqLens, key, value, seqOffset, keyOut,
                valueOut, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(blockTables);
    AscendC::GmFree(seqLens);
    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(seqOffset);
    AscendC::GmFree(keyOut);
    AscendC::GmFree(valueOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
