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
 * \file test_layer_norm_v4.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "scatter_pa_kv_cache_tiling_def.h"
#include "data_utils.h"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void scatter_pa_kv_cache(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping,
                                                          GM_ADDR value, GM_ADDR value_cache_in, GM_ADDR compress_lens,
                                                          GM_ADDR compress_seq_offset, GM_ADDR seq_lens,
                                                          GM_ADDR key_cache_out, GM_ADDR value_cache_out,
                                                          GM_ADDR workspace, GM_ADDR tiling);

class scatter_pa_kv_cache_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "scatter_pa_kv_cache_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "scatter_pa_kv_cache_test TearDown\n" << endl;
    }
};

// ScatterPaKvCache
TEST_F(scatter_pa_kv_cache_test, test_case_nz_full_load_001)
{
    int64_t batch = 256;
    int64_t seqLen = 1;
    int64_t numHeads = 1;
    int64_t kHeadSize = 512;
    int64_t vHeadSize = 64;
    int64_t blockSize = 16;
    int64_t numBlocks = 20;
    int64_t numTokens = batch * seqLen;

    size_t keySize = numTokens * numHeads * kHeadSize * sizeof(half);
    size_t valueSize = numTokens * numHeads * vHeadSize * sizeof(half);
    size_t keyCacheSize = numBlocks * blockSize * numHeads * kHeadSize * sizeof(half);
    size_t valueCacheSize = numBlocks * blockSize * numHeads * vHeadSize * sizeof(half);
    size_t slotMappingSize = numTokens * sizeof(int32_t);
    size_t tilingDataSize = sizeof(ScatterPaKvCacheTilingData);

    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCacheSize);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCacheSize);
    uint8_t* slotMapping = (uint8_t*)AscendC::GmAlloc(slotMappingSize);

    uint8_t* compress_lens = nullptr;
    uint8_t* compress_seq_offset = nullptr;
    uint8_t* seq_lens = nullptr;

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 24;

    system("cp -r "
           "../../../../index/scatter_pa_kv_cache/tests/ut/op_kernel/"
           "scatter_pa_kv_cache_data ./ && chmod -R 755 ./scatter_pa_kv_cache_data/");
    system("cd ./scatter_pa_kv_cache_data/ && rm -rf ./*bin");
    system("cd ./scatter_pa_kv_cache_data/ && python3 gen_data.py test_scatter_pa_kv_cache_full_load1");
    system("cd ./scatter_pa_kv_cache_data/ && python3 gen_tiling.py test_scatter_pa_kv_cache_full_load1");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_key.bin", keySize, key, keySize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_value.bin", valueSize, value, valueSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_slot_mapping.bin", slotMappingSize, slotMapping, slotMappingSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_key_cache.bin", keyCacheSize, keyCache, keyCacheSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_value_cache.bin", valueCacheSize, valueCache, valueCacheSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(1141);
    ICPU_RUN_KF(scatter_pa_kv_cache, blockDim, key, keyCache, slotMapping, value, valueCache, compress_lens,
                compress_seq_offset, seq_lens, keyCache, valueCache, workspace, tiling);

    WriteFile(path + "/scatter_pa_kv_cache_data/output_key_cache.bin", keyCache, keyCacheSize);
    WriteFile(path + "/scatter_pa_kv_cache_data/output_value_cache.bin", valueCache, valueCacheSize);

    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(scatter_pa_kv_cache_test, test_case_nz_full_load_002)
{
    int64_t batch = 256;
    int64_t seqLen = 1;
    int64_t numHeads = 1;
    int64_t kHeadSize = 512;
    int64_t vHeadSize = 64;
    int64_t blockSize = 16;
    int64_t numBlocks = 20;
    int64_t numTokens = batch * seqLen;

    size_t keySize = numTokens * numHeads * kHeadSize * sizeof(half);
    size_t valueSize = numTokens * numHeads * vHeadSize * sizeof(half);
    size_t keyCacheSize = numBlocks * blockSize * numHeads * kHeadSize * sizeof(half);
    size_t valueCacheSize = numBlocks * blockSize * numHeads * vHeadSize * sizeof(half);
    size_t slotMappingSize = numTokens * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ScatterPaKvCacheTilingData);

    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCacheSize);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCacheSize);
    uint8_t* slotMapping = (uint8_t*)AscendC::GmAlloc(slotMappingSize);

    uint8_t* compress_lens = nullptr;
    uint8_t* compress_seq_offset = nullptr;
    uint8_t* seq_lens = nullptr;

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 24;

    system("cp -r "
           "../../../../index/scatter_pa_kv_cache/tests/ut/op_kernel/"
           "scatter_pa_kv_cache_data ./ && chmod -R 755 ./scatter_pa_kv_cache_data/");
    system("cd ./scatter_pa_kv_cache_data/ && rm -rf ./*bin");
    system("cd ./scatter_pa_kv_cache_data/ && python3 gen_data.py test_scatter_pa_kv_cache_full_load2");
    system("cd ./scatter_pa_kv_cache_data/ && python3 gen_tiling.py test_scatter_pa_kv_cache_full_load2");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_key.bin", keySize, key, keySize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_value.bin", valueSize, value, valueSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_slot_mapping.bin", slotMappingSize, slotMapping, slotMappingSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_key_cache.bin", keyCacheSize, keyCache, keyCacheSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_value_cache.bin", valueCacheSize, valueCache, valueCacheSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(1181);
    ICPU_RUN_KF(scatter_pa_kv_cache, blockDim, key, keyCache, slotMapping, value, valueCache, compress_lens,
                compress_seq_offset, seq_lens, keyCache, valueCache, workspace, tiling);

    WriteFile(path + "/scatter_pa_kv_cache_data/output_key_cache.bin", keyCache, keyCacheSize);
    WriteFile(path + "/scatter_pa_kv_cache_data/output_value_cache.bin", valueCache, valueCacheSize);

    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(scatter_pa_kv_cache_test, test_case_nz_not_full_load_001)
{
    // 仅用于申请内存，与scatter_pa_kv_cache_data中保持一致
    int64_t batch = 256;
    int64_t seqLen = 1;
    int64_t numHeads = 1;
    int64_t kHeadSize = 512;
    int64_t vHeadSize = 64;
    int64_t blockSize = 16;
    int64_t numBlocks = 20;
    int64_t numTokens = batch * seqLen;

    size_t keySize = numTokens * numHeads * kHeadSize * sizeof(half);
    size_t valueSize = numTokens * numHeads * vHeadSize * sizeof(half);
    size_t keyCacheSize = numBlocks * blockSize * numHeads * kHeadSize * sizeof(half);
    size_t valueCacheSize = numBlocks * blockSize * numHeads * vHeadSize * sizeof(half);
    size_t slotMappingSize = numTokens * sizeof(int32_t);
    size_t tilingDataSize = sizeof(ScatterPaKvCacheTilingData);

    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCacheSize);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCacheSize);
    uint8_t* slotMapping = (uint8_t*)AscendC::GmAlloc(slotMappingSize);

    uint8_t* compress_lens = nullptr;
    uint8_t* compress_seq_offset = nullptr;
    uint8_t* seq_lens = nullptr;

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 24;

    system("cp -r "
           "../../../../index/scatter_pa_kv_cache/tests/ut/op_kernel/"
           "scatter_pa_kv_cache_data ./ && chmod -R 755 ./scatter_pa_kv_cache_data/");
    system("cd ./scatter_pa_kv_cache_data/ && rm -rf ./*bin");
    system("cd ./scatter_pa_kv_cache_data/ && python3 gen_data.py test_case_nz_not_full_load_001");
    system("cd ./scatter_pa_kv_cache_data/ && python3 gen_tiling.py test_case_nz_not_full_load_001");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_key.bin", keySize, key, keySize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_value.bin", valueSize, value, valueSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_slot_mapping.bin", slotMappingSize, slotMapping, slotMappingSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_key_cache.bin", keyCacheSize, keyCache, keyCacheSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_value_cache.bin", valueCacheSize, valueCache, valueCacheSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(1140);
    ICPU_RUN_KF(scatter_pa_kv_cache, blockDim, key, keyCache, slotMapping, value, valueCache, compress_lens,
                compress_seq_offset, seq_lens, keyCache, valueCache, workspace, tiling);

    WriteFile(path + "/scatter_pa_kv_cache_data/output_key_cache.bin", keyCache, keyCacheSize);
    WriteFile(path + "/scatter_pa_kv_cache_data/output_value_cache.bin", valueCache, valueCacheSize);

    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(scatter_pa_kv_cache_test, test_case_nz_not_full_load_002)
{
    int64_t batch = 256;
    int64_t seqLen = 1;
    int64_t numHeads = 1;
    int64_t kHeadSize = 512;
    int64_t vHeadSize = 64;
    int64_t blockSize = 16;
    int64_t numBlocks = 20;
    int64_t numTokens = batch * seqLen;

    size_t keySize = numTokens * numHeads * kHeadSize * sizeof(half);
    size_t valueSize = numTokens * numHeads * vHeadSize * sizeof(half);
    size_t keyCacheSize = numBlocks * blockSize * numHeads * kHeadSize * sizeof(half);
    size_t valueCacheSize = numBlocks * blockSize * numHeads * vHeadSize * sizeof(half);
    size_t slotMappingSize = numTokens * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ScatterPaKvCacheTilingData);

    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCacheSize);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCacheSize);
    uint8_t* slotMapping = (uint8_t*)AscendC::GmAlloc(slotMappingSize);

    uint8_t* compress_lens = nullptr;
    uint8_t* compress_seq_offset = nullptr;
    uint8_t* seq_lens = nullptr;

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 24;

    system("cp -r "
           "../../../../index/scatter_pa_kv_cache/tests/ut/op_kernel/"
           "scatter_pa_kv_cache_data ./ && chmod -R 755 ./scatter_pa_kv_cache_data/");
    system("cd ./scatter_pa_kv_cache_data/ && rm -rf ./*bin");
    system("cd ./scatter_pa_kv_cache_data/ && python3 gen_data.py test_case_nz_not_full_load_002");
    system("cd ./scatter_pa_kv_cache_data/ && python3 gen_tiling.py test_case_nz_not_full_load_002");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_key.bin", keySize, key, keySize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_value.bin", valueSize, value, valueSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_slot_mapping.bin", slotMappingSize, slotMapping, slotMappingSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_key_cache.bin", keyCacheSize, keyCache, keyCacheSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/input_value_cache.bin", valueCacheSize, valueCache, valueCacheSize);
    ReadFile(path + "/scatter_pa_kv_cache_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(1180);
    ICPU_RUN_KF(scatter_pa_kv_cache, blockDim, key, keyCache, slotMapping, value, valueCache, compress_lens,
                compress_seq_offset, seq_lens, keyCache, valueCache, workspace, tiling);

    WriteFile(path + "/scatter_pa_kv_cache_data/output_key_cache.bin", keyCache, keyCacheSize);
    WriteFile(path + "/scatter_pa_kv_cache_data/output_value_cache.bin", valueCache, valueCacheSize);

    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}
