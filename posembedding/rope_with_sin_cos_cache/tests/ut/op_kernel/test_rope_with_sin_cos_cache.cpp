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
#include "test_rope_with_sin_cos_cache.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void rope_with_sin_cos_cache(
    GM_ADDR positions, GM_ADDR queryIn, GM_ADDR keyIn, GM_ADDR cosSinCache, GM_ADDR queryOut, GM_ADDR keyOut,
    GM_ADDR workspace, GM_ADDR tiling);

class rope_with_sin_cos_cache_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "rope_with_sin_cos_cache_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "rope_with_sin_cos_cache_test TearDown\n" << std::endl;
    }
};

TEST_F(rope_with_sin_cos_cache_test, test_case_bf16)
{
    system(
        "cp -rf "
        "../../../../../posembedding/rope_with_sin_cos_cache/tests/ut/op_kernel/rope_with_sin_cos_cache_data ./");
    system("chmod -R 755 ./rope_with_sin_cos_cache_data/");
    system("cd ./rope_with_sin_cos_cache_data/ && python3 gen_data.py '48' '2' '4' '128' '128' 'bf16' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint64_t numTokens = 48;
    uint64_t numQHeads = 2;
    uint64_t numKHeads = 4;
    uint64_t headSize = 128;
    uint64_t rotaryDim = 128;

    size_t positionSize = numTokens * sizeof(int64_t);
    size_t querySize = numTokens * numQHeads * headSize * sizeof(half);
    size_t keySize = numTokens * numKHeads * headSize * sizeof(half);
    size_t cosSinCacheSize = numTokens * rotaryDim * sizeof(half);
    size_t queryOutSize = numTokens * numQHeads * headSize * sizeof(half);
    size_t keyOutSize = numTokens * numKHeads * headSize * sizeof(half);

    size_t tilingSize = sizeof(RopeWithSinCosCacheTilingData);

    uint8_t* position = (uint8_t*)AscendC::GmAlloc(positionSize);
    uint8_t* query = (uint8_t*)AscendC::GmAlloc(querySize);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* cosSinCache = (uint8_t*)AscendC::GmAlloc(cosSinCacheSize);
    uint8_t* queryOut = (uint8_t*)AscendC::GmAlloc(queryOutSize);
    uint8_t* keyOut = (uint8_t*)AscendC::GmAlloc(keyOutSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    RopeWithSinCosCacheTilingData* tilingData = reinterpret_cast<RopeWithSinCosCacheTilingData*>(tiling);
    tilingData->core_num_use = 48;
    tilingData->num_tokens = 48;
    tilingData->num_q_heads = 2;
    tilingData->num_kv_heads = 4;
    tilingData->head_size = 128;
    tilingData->rotary_dim = 128;
    tilingData->mrope_section0 = 0;
    tilingData->mrope_section1 = 0;
    tilingData->mrope_section2 = 0;
    tilingData->q_leading_dimension = 256;
    tilingData->k_leading_dimension = 512;
    tilingData->isNeoxStyle = 1;
    tilingData->front_core = 48;
    tilingData->tail_core = 0;
    tilingData->num_tokens_each_front_core = 1;
    tilingData->num_tokens_each_tail_core = 1;
    tilingData->num_tokens_front_core_each_loop = 1;
    tilingData->num_tokens_tail_core_each_loop = 1;
    tilingData->loop_time_each_front_core = 1;
    tilingData->loop_time_each_tail_core = 1;
    tilingData->num_tokens_front_core_last_loop = 0;
    tilingData->num_tokens_tail_core_last_loop = 0;

    ReadFile("./rope_with_sin_cos_cache_data/input_position.bin", positionSize, position, positionSize);
    ReadFile("./rope_with_sin_cos_cache_data/input_query.bin", querySize, query, querySize);
    ReadFile("./rope_with_sin_cos_cache_data/input_key.bin", keySize, key, keySize);
    ReadFile("./rope_with_sin_cos_cache_data/input_cosSinCache.bin", cosSinCacheSize, cosSinCache, cosSinCacheSize);
    ReadFile("./rope_with_sin_cos_cache_data/output_queryOut.bin", queryOutSize, queryOut, queryOutSize);
    ReadFile("./rope_with_sin_cos_cache_data/output_keyOut.bin", keyOutSize, keyOut, keyOutSize);

    ICPU_SET_TILING_KEY(20);
    ICPU_RUN_KF(
        rope_with_sin_cos_cache, blockDim, position, query, key, cosSinCache, queryOut, keyOut, workspace, tiling);

    AscendC::GmFree(position);
    AscendC::GmFree(query);
    AscendC::GmFree(key);
    AscendC::GmFree(cosSinCache);
    AscendC::GmFree(queryOut);
    AscendC::GmFree(keyOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rope_with_sin_cos_cache_test, test_case_fp32)
{
    system(
        "cp -rf "
        "../../../../../posembedding/rope_with_sin_cos_cache/tests/ut/op_kernel/rope_with_sin_cos_cache_data ./");
    system("chmod -R 755 ./rope_with_sin_cos_cache_data/");
    system("cd ./rope_with_sin_cos_cache_data/ && python3 gen_data.py '48' '2' '4' '128' '128' 'fp32' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint64_t numTokens = 48;
    uint64_t numQHeads = 2;
    uint64_t numKHeads = 4;
    uint64_t headSize = 128;
    uint64_t rotaryDim = 128;

    size_t positionSize = numTokens * sizeof(int64_t);
    size_t querySize = numTokens * numQHeads * headSize * sizeof(float);
    size_t keySize = numTokens * numKHeads * headSize * sizeof(float);
    size_t cosSinCacheSize = numTokens * rotaryDim * sizeof(float);
    size_t queryOutSize = numTokens * numQHeads * headSize * sizeof(float);
    size_t keyOutSize = numTokens * numKHeads * headSize * sizeof(float);

    size_t tilingSize = sizeof(RopeWithSinCosCacheTilingData);

    uint8_t* position = (uint8_t*)AscendC::GmAlloc(positionSize);
    uint8_t* query = (uint8_t*)AscendC::GmAlloc(querySize);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* cosSinCache = (uint8_t*)AscendC::GmAlloc(cosSinCacheSize);
    uint8_t* queryOut = (uint8_t*)AscendC::GmAlloc(queryOutSize);
    uint8_t* keyOut = (uint8_t*)AscendC::GmAlloc(keyOutSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    RopeWithSinCosCacheTilingData* tilingData = reinterpret_cast<RopeWithSinCosCacheTilingData*>(tiling);
    tilingData->core_num_use = 48;
    tilingData->num_tokens = 48;
    tilingData->num_q_heads = 2;
    tilingData->num_kv_heads = 4;
    tilingData->head_size = 128;
    tilingData->rotary_dim = 128;
    tilingData->mrope_section0 = 0;
    tilingData->mrope_section1 = 0;
    tilingData->mrope_section2 = 0;
    tilingData->q_leading_dimension = 256;
    tilingData->k_leading_dimension = 512;
    tilingData->isNeoxStyle = 1;
    tilingData->front_core = 48;
    tilingData->tail_core = 0;
    tilingData->num_tokens_each_front_core = 1;
    tilingData->num_tokens_each_tail_core = 1;
    tilingData->num_tokens_front_core_each_loop = 1;
    tilingData->num_tokens_tail_core_each_loop = 1;
    tilingData->loop_time_each_front_core = 1;
    tilingData->loop_time_each_tail_core = 1;
    tilingData->num_tokens_front_core_last_loop = 0;
    tilingData->num_tokens_tail_core_last_loop = 0;

    ReadFile("./rope_with_sin_cos_cache_data/input_position.bin", positionSize, position, positionSize);
    ReadFile("./rope_with_sin_cos_cache_data/input_query.bin", querySize, query, querySize);
    ReadFile("./rope_with_sin_cos_cache_data/input_key.bin", keySize, key, keySize);
    ReadFile("./rope_with_sin_cos_cache_data/input_cosSinCache.bin", cosSinCacheSize, cosSinCache, cosSinCacheSize);
    ReadFile("./rope_with_sin_cos_cache_data/output_queryOut.bin", queryOutSize, queryOut, queryOutSize);
    ReadFile("./rope_with_sin_cos_cache_data/output_keyOut.bin", keyOutSize, keyOut, keyOutSize);

    ICPU_SET_TILING_KEY(22);
    ICPU_RUN_KF(
        rope_with_sin_cos_cache, blockDim, position, query, key, cosSinCache, queryOut, keyOut, workspace, tiling);

    AscendC::GmFree(position);
    AscendC::GmFree(query);
    AscendC::GmFree(key);
    AscendC::GmFree(cosSinCache);
    AscendC::GmFree(queryOut);
    AscendC::GmFree(keyOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rope_with_sin_cos_cache_test, test_case_fp16)
{
    system(
        "cp -rf "
        "../../../../../posembedding/rope_with_sin_cos_cache/tests/ut/op_kernel/rope_with_sin_cos_cache_data ./");
    system("chmod -R 755 ./rope_with_sin_cos_cache_data/");
    system("cd ./rope_with_sin_cos_cache_data/ && python3 gen_data.py '48' '2' '4' '128' '128' 'fp16' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint64_t numTokens = 48;
    uint64_t numQHeads = 2;
    uint64_t numKHeads = 4;
    uint64_t headSize = 128;
    uint64_t rotaryDim = 128;

    size_t positionSize = numTokens * sizeof(int64_t);
    size_t querySize = numTokens * numQHeads * headSize * sizeof(half);
    size_t keySize = numTokens * numKHeads * headSize * sizeof(half);
    size_t cosSinCacheSize = numTokens * rotaryDim * sizeof(half);
    size_t queryOutSize = numTokens * numQHeads * headSize * sizeof(half);
    size_t keyOutSize = numTokens * numKHeads * headSize * sizeof(half);

    size_t tilingSize = sizeof(RopeWithSinCosCacheTilingData);

    uint8_t* position = (uint8_t*)AscendC::GmAlloc(positionSize);
    uint8_t* query = (uint8_t*)AscendC::GmAlloc(querySize);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* cosSinCache = (uint8_t*)AscendC::GmAlloc(cosSinCacheSize);
    uint8_t* queryOut = (uint8_t*)AscendC::GmAlloc(queryOutSize);
    uint8_t* keyOut = (uint8_t*)AscendC::GmAlloc(keyOutSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    RopeWithSinCosCacheTilingData* tilingData = reinterpret_cast<RopeWithSinCosCacheTilingData*>(tiling);
    tilingData->core_num_use = 48;
    tilingData->num_tokens = 48;
    tilingData->num_q_heads = 2;
    tilingData->num_kv_heads = 4;
    tilingData->head_size = 128;
    tilingData->rotary_dim = 128;
    tilingData->mrope_section0 = 0;
    tilingData->mrope_section1 = 0;
    tilingData->mrope_section2 = 0;
    tilingData->q_leading_dimension = 256;
    tilingData->k_leading_dimension = 512;
    tilingData->isNeoxStyle = 1;
    tilingData->front_core = 48;
    tilingData->tail_core = 0;
    tilingData->num_tokens_each_front_core = 1;
    tilingData->num_tokens_each_tail_core = 1;
    tilingData->num_tokens_front_core_each_loop = 1;
    tilingData->num_tokens_tail_core_each_loop = 1;
    tilingData->loop_time_each_front_core = 1;
    tilingData->loop_time_each_tail_core = 1;
    tilingData->num_tokens_front_core_last_loop = 0;
    tilingData->num_tokens_tail_core_last_loop = 0;

    ReadFile("./rope_with_sin_cos_cache_data/input_position.bin", positionSize, position, positionSize);
    ReadFile("./rope_with_sin_cos_cache_data/input_query.bin", querySize, query, querySize);
    ReadFile("./rope_with_sin_cos_cache_data/input_key.bin", keySize, key, keySize);
    ReadFile("./rope_with_sin_cos_cache_data/input_cosSinCache.bin", cosSinCacheSize, cosSinCache, cosSinCacheSize);
    ReadFile("./rope_with_sin_cos_cache_data/output_queryOut.bin", queryOutSize, queryOut, queryOutSize);
    ReadFile("./rope_with_sin_cos_cache_data/output_keyOut.bin", keyOutSize, keyOut, keyOutSize);

    ICPU_SET_TILING_KEY(21);
    ICPU_RUN_KF(
        rope_with_sin_cos_cache, blockDim, position, query, key, cosSinCache, queryOut, keyOut, workspace, tiling);

    AscendC::GmFree(position);
    AscendC::GmFree(query);
    AscendC::GmFree(key);
    AscendC::GmFree(cosSinCache);
    AscendC::GmFree(queryOut);
    AscendC::GmFree(keyOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rope_with_sin_cos_cache_test, test_case_fp32_2)
{
    system(
        "cp -rf "
        "../../../../../posembedding/rope_with_sin_cos_cache/tests/ut/op_kernel/rope_with_sin_cos_cache_data ./");
    system("chmod -R 755 ./rope_with_sin_cos_cache_data/");
    system("cd ./rope_with_sin_cos_cache_data/ && python3 gen_data.py '48' '2' '4' '128' '128' 'fp32' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint64_t numTokens = 48;
    uint64_t numQHeads = 2;
    uint64_t numKHeads = 4;
    uint64_t headSize = 128;
    uint64_t rotaryDim = 128;

    size_t positionSize = numTokens * sizeof(int64_t);
    size_t querySize = numTokens * numQHeads * headSize * sizeof(float);
    size_t keySize = numTokens * numKHeads * headSize * sizeof(float);
    size_t cosSinCacheSize = numTokens * rotaryDim * sizeof(float);
    size_t queryOutSize = numTokens * numQHeads * headSize * sizeof(float);
    size_t keyOutSize = numTokens * numKHeads * headSize * sizeof(float);

    size_t tilingSize = sizeof(RopeWithSinCosCacheTilingData);

    uint8_t* position = (uint8_t*)AscendC::GmAlloc(positionSize);
    uint8_t* query = (uint8_t*)AscendC::GmAlloc(querySize);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* cosSinCache = (uint8_t*)AscendC::GmAlloc(cosSinCacheSize);
    uint8_t* queryOut = (uint8_t*)AscendC::GmAlloc(queryOutSize);
    uint8_t* keyOut = (uint8_t*)AscendC::GmAlloc(keyOutSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    RopeWithSinCosCacheTilingData* tilingData = reinterpret_cast<RopeWithSinCosCacheTilingData*>(tiling);
    tilingData->core_num_use = 48;
    tilingData->num_tokens = 48;
    tilingData->num_q_heads = 2;
    tilingData->num_kv_heads = 4;
    tilingData->head_size = 128;
    tilingData->rotary_dim = 128;
    tilingData->mrope_section0 = 0;
    tilingData->mrope_section1 = 0;
    tilingData->mrope_section2 = 0;
    tilingData->q_leading_dimension = 256;
    tilingData->k_leading_dimension = 512;
    tilingData->isNeoxStyle = 1;
    tilingData->front_core = 48;
    tilingData->tail_core = 0;
    tilingData->num_tokens_each_front_core = 1;
    tilingData->num_tokens_each_tail_core = 1;
    tilingData->num_tokens_front_core_each_loop = 1;
    tilingData->num_tokens_tail_core_each_loop = 1;
    tilingData->loop_time_each_front_core = 1;
    tilingData->loop_time_each_tail_core = 1;
    tilingData->num_tokens_front_core_last_loop = 0;
    tilingData->num_tokens_tail_core_last_loop = 0;

    ReadFile("./rope_with_sin_cos_cache_data/input_position.bin", positionSize, position, positionSize);
    ReadFile("./rope_with_sin_cos_cache_data/input_query.bin", querySize, query, querySize);
    ReadFile("./rope_with_sin_cos_cache_data/input_key.bin", keySize, key, keySize);
    ReadFile("./rope_with_sin_cos_cache_data/input_cosSinCache.bin", cosSinCacheSize, cosSinCache, cosSinCacheSize);
    ReadFile("./rope_with_sin_cos_cache_data/output_queryOut.bin", queryOutSize, queryOut, queryOutSize);
    ReadFile("./rope_with_sin_cos_cache_data/output_keyOut.bin", keyOutSize, keyOut, keyOutSize);

    ICPU_SET_TILING_KEY(22);
    ICPU_RUN_KF(
        rope_with_sin_cos_cache, blockDim, position, query, key, cosSinCache, queryOut, keyOut, workspace, tiling);

    AscendC::GmFree(position);
    AscendC::GmFree(query);
    AscendC::GmFree(key);
    AscendC::GmFree(cosSinCache);
    AscendC::GmFree(queryOut);
    AscendC::GmFree(keyOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
