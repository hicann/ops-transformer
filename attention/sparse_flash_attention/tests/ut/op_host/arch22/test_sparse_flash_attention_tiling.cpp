/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "register/tilingdata_base.h"
using namespace std;

class SparseFlashAttentionTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseFlashAttentionTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseFlashAttentionTiling TearDown" << std::endl;
    }
};

// Shape of tensor softmaxMax is nullptr
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_0)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {1, 1, 1, 1};
    int64_t actual_seq_kvlist[] = {65536, 65536, 65536, 65536};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query            input0
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input2
            {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},          // sparse_indices   input3
            {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table      input4
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query_rope       input5
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}  // key_rope         input6
        },
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // softmax_max
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Shape of queryRope is nullptr
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_1)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {1, 2};
    int64_t actual_seq_kvlist[] = {1, 2};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{2, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},    // query            input0
            {{{2, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},    // key              input1
            {{{2, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},    // value            input2
            {{{2, 1, 206}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},        // sparse_indices   input3
            {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},              // block_table      input4
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_query
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                             // query_rope       input5
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}                              // key_rope         input6
        },
        {
            {{{2, 128, 512}, {2, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{1, 2, 128}, {1, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},    // softmax_max
            {{{1, 2, 128}, {1, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}     // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Shape of tensor softmaxMax is nullptr
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_2)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {1, 1, 1, 1};
    int64_t actual_seq_kvlist[] = {65536, 65536, 65536, 65536};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query            input0
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input2
            {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},          // sparse_indices   input3
            {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table      input4
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query_rope       input5
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}  // key_rope         input6
        },
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // softmax_max
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// The dimensions of query and query_rope should be equal, but query has dimension 3 while query_rope has dimension 4.
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_3)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {18432, 36864, 55296, 73728, 92160, 110592, 129024, 147456, 165888, 184320};
    int64_t actual_seq_kvlist[] = {37748, 40836, 49869, 64087, 19343, 52909, 4619, 22910, 7926, 65488};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND},       // query            input0
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input2
            {{{184320, 1, 16}, {184320, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},        // sparse_indices   input3
            {{{10, 128}, {10, 128}}, ge::DT_INT32, ge::FORMAT_ND},                    // block_table      input4
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope       input5
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}    // key_rope         input6
        },
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                           // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Shape of tensor softmaxMax is nullptr
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_4)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {1, 1, 1, 1};
    int64_t actual_seq_kvlist[] = {65536, 65536, 65536, 65536};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query            input0
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input2
            {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},          // sparse_indices   input3
            {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table      input4
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query_rope       input5
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}  // key_rope         input6
        },
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // softmax_max
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Shape of tensor softmaxMax is nullptr
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_5)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {1, 1, 1, 1};
    int64_t actual_seq_kvlist[] = {65536, 65536, 65536, 65536};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query            input0
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input2
            {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},          // sparse_indices   input3
            {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table      input4
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{4096, 64, 1, 64}, {4096, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope       input5
            {{{4096, 64, 1, 64}, {4096, 64, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}    // key_rope         input6
        },
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // softmax_max
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// query_rope layout is BSND, shape is [1, 128, 1, 64], expected shape is [1, 128, 128, 64].
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_6)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query            input0
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // key              input1
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // value            input2
            {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},  // sparse_indices   input3
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                        // block_table      input4
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},       // query_rope       input5
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}        // key_rope         input6
        },
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 0 < blockSize_ < 1024 success
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_7)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query            input0
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // key              input1
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // value            input2
            {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},  // sparse_indices   input3
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                        // block_table      input4
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope       input5
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}        // key_rope         input6
        },
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 50306; // 17538 | HAS_ROPE(bit15)
    std::string expectTilingData = "549755813889 128 128 4407522836751581185 4294967424 1 3 "
                                   "9223372036854775807 9223372036854775807 2 1 2048 0 36028797018963968 "
                                   "262144 197525504 64 281474976727040 549755814016 ";
    std::vector<size_t> expectWorkspaces = {214302720};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// the layout_kv is PA_BSND, blockTable must be provided.
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_8)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {18432, 36864, 55296, 73728, 92160, 110592, 129024, 147456, 165888, 184320};
    int64_t actual_seq_kvlist[] = {37748, 40836, 49869, 64087, 19343, 52909, 4619, 22910, 7926, 65488};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND},       // query            input0
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input2
            {{{184320, 1, 16}, {184320, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},        // sparse_indices   input3
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                  // block_table      input4
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope       input5
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}    // key_rope         input6
        },
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                           // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// the dim num of block_table is 3, it should be 2.
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_9)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {18432, 36864, 55296, 73728, 92160, 110592, 129024, 147456, 165888, 184320};
    int64_t actual_seq_kvlist[] = {37748, 40836, 49869, 64087, 19343, 52909, 4619, 22910, 7926, 65488};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND},       // query            input0
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input2
            {{{184320, 1, 16}, {184320, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},        // sparse_indices   input3
            {{{1, 2, 1}, {1, 2, 1}}, ge::DT_INT32, ge::FORMAT_ND},                    // block_table      input4
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope       input5
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}    // key_rope         input6
        },
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                           // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// block_table's second dimension(-1) should be greater than 0
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_10)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {18432, 36864, 55296, 73728, 92160, 110592, 129024, 147456, 165888, 184320};
    int64_t actual_seq_kvlist[] = {37748, 40836, 49869, 64087, 19343, 52909, 4619, 22910, 7926, 65488};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND},       // query            input0
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{10, 65488, 1, 512}, {10, 65488, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input2
            {{{184320, 1, 16}, {184320, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},        // sparse_indices   input3
            {{{1, -1}, {1, -1}}, ge::DT_INT32, ge::FORMAT_ND},                        // block_table      input4
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope       input5
            {{{753, 512, 1, 512}, {753, 512, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}    // key_rope         input6
        },
        {
            {{{184320, 4, 512}, {184320, 4, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                          // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                           // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Shape of tensor softmaxMax is nullptr
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_11)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {1, 1, 1, 1};
    int64_t actual_seq_kvlist[] = {65536, 65536, 65536, 65536};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // query            input0
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // key              input1
            {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // value            input2
            {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},          // sparse_indices   input3
            {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},                  // block_table      input4
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query_rope       input5
            {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}  // key_rope         input6
        },
        {
            {{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // softmax_max
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}                             // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// when the layout_kv is BSND, block_table should be null
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_12)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // query            input0
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // key              input1
            {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},     // value            input2
            {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},  // sparse_indices   input3
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},                        // block_table      input4
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},      // actual_seq_lengths_query
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},     // actual_seq_lengths_kv
            {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},   // query_rope       input5
            {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}        // key_rope         input6
        },
        {
            {{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                               // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// when return_softmax_lse is true, key layout do not support PA_BSND.
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_13)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {1, 1, 1, 1};
    int64_t actual_seq_kvlist[] = {65536, 65536, 65536, 65536};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {{{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4, 65536, 1, 512}, {4, 65536, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4, 1, 1, 16}, {4, 1, 1, 16}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4, 1024}, {4, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4096, 64, 1, 512}, {4096, 64, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{4, 1, 128, 512}, {4, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4, 1, 128, 128}, {4, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{4, 1, 128, 128}, {4, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Tensor query only supports dtype DT_FLOAT16, DT_BFLOAT16, but got DT_FLOAT
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_14)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// The dimensions of query and query_rope should be equal, but query has dimension 2 while query_rope has dimension 4.
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_15)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {{{{128, 512}, {128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// oftmax_max dimension is 3, expected dimension is 4.
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_16)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 8}, {1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 128, 8}, {1, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// query_rope layout is TND, shape is [2, 1, 64], expected shape is [2, 128, 64].
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_17)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {1, 2};
    int64_t actual_seq_kvlist[] = {1, 2};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {{{{2, 128, 512}, {2, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 1, 512}, {2, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 1, 512}, {2, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 1, 206}, {2, 1, 206}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{2, 1, 64}, {2, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 1, 64}, {2, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{2, 128, 512}, {2, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 8, 1}, {2, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 8, 1}, {2, 8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// split kv try
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_18)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {1};
    int64_t actual_seq_kvlist[] = {8192};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {{{{1, 1, 128, 512}, {1, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 8192, 1, 512}, {1, 8192, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 8192, 1, 512}, {1, 8192, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 1, 1, 8192}, {1, 1, 1, 8192}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{1, 1, 128, 64}, {1, 1, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 8192, 1, 64}, {1, 8192, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1, 1, 128, 512}, {1, 1, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    int64_t expectTilingKey = 50177; // FLASH_DECODE | 17408 | HAS_ROPE(bit15)
    // BF16 residual compensation allows 16 partitions for one query with 8192 tokens.
    std::string expectTilingData = "35184372088833 1 0 4407522836751581184 4294967424 1 3 "
                                   "9223372036854775807 9223372036854775807 2 1 8192 0 36028797018963984 "
                                   "262144 252051456 16 281474976776192 2199023255680 ";
    std::vector<size_t> expectWorkspaces = {252051456};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData);
}

// TND query and TND key layout, success
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_20)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {128};
    int64_t actual_seq_kvlist[] = {128};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // query TND
            {{{128, 1, 512}, {128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},       // key TND
            {{{128, 1, 512}, {128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},       // value TND
            {{{128, 1, 2048}, {128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},    // sparse_indices
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                            // block_table
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_query
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_kv
            {{{128, 128, 64}, {128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},     // query_rope
            {{{128, 1, 64}, {128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}          // key_rope
        },
        {
            {{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND}, // attention_out
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                        // softmax_max
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}                         // softmax_sum
        },
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// BSND query and BSND key layout with return_softmax_lse, success
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_21)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 1, 128, 128}, {1, 1, 128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// FP16 dtype with BSND query and PA_BSND key layout, success
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_22)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {113};
    int64_t actual_seq_kvlist[] = {113};
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},
         {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// layout_kv is invalid
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_23)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("INVALID")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// layout_query is invalid
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_24)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("INVALID")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// layout_query is BSND but layout_kv is TND, mismatch
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_25)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 1, 512}, {128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 1, 512}, {128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 1, 2048}, {128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128, 128, 64}, {128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 1, 64}, {128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{128, 128, 512}, {128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// layout_kv is PA_BSND but key dim num is 3
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_26)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128, 1, 512}, {128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// return_softmax_lse is true while layout_kv is PA_BSND
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_27)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 128, 1}, {1, 128, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 128, 128, 1}, {1, 128, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend910B", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// soc is not supported by the op
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_910b_tiling_28)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SparseFlashAttention",
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 512}, {1, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 2048}, {1, 128, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1, 128, 128, 64}, {1, 128, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 128, 1, 64}, {1, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1, 128, 128, 512}, {1, 128, 128, 512}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend310P", 64, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Tiling data classes are registered for the op
TEST_F(SparseFlashAttentionTiling, SparseFlashAttention_tiling_data_class_registered)
{
    auto& factory = optiling::CTilingDataClassFactory::GetInstance();
    EXPECT_NE(factory.CreateTilingDataInstance("SparseFlashAttention"), nullptr);
    EXPECT_NE(factory.CreateTilingDataInstance("SparseFlashAttentionBaseParamsMlaOp"), nullptr);
    EXPECT_NE(factory.CreateTilingDataInstance("SparseFlashAttentionSingleCoreParamsMlaOp"), nullptr);
    EXPECT_NE(factory.CreateTilingDataInstance("SparseFlashAttentionSingleCoreTensorSizeMlaOp"), nullptr);
    EXPECT_NE(factory.CreateTilingDataInstance("SparseFlashAttentionSplitKVParamsMlaOp"), nullptr);
    EXPECT_NE(factory.CreateTilingDataInstance("SparseFlashAttentionInnerSplitParamsOp"), nullptr);
}

TEST_F(SparseFlashAttentionTiling, SplitKvUsesArch22InnerBlocks)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    for (int64_t heads : {1, 2, 4, 8, 16, 32, 64, 128}) {
        for (bool returnLse : {false, true}) {
            for (const auto& [batchCount, queryCount, topK, expectedBlocks, expectedFd] :
                 {std::tuple<int64_t, int64_t, int64_t, uint32_t, bool>{1, 1, 2048, 4, true},
                  {1, 4, 2048, 16, true},
                  {2, 4, 2048, 16, true},
                  {1, 1, 1024, 2, true},
                  {1, 1, 1536, 3, true},
                  {1, 1, 2049, 5, true},
                  {1, 1, 3072, 6, true},
                  {1, 1, 4096, 8, true},
                  {1, 1, 8192, 16, true},
                  {1, 1, 24 * 512, 24, true}, // 24 tiles on 24 cores: one tile per partition.
                  {1, 1, 25 * 512, 13, true}, // 25 tiles need two rounds; 13 partitions suffice.
                  {1, 1, 32 * 512, 16, true}, // 32 tiles need two rounds; 16 partitions suffice.
                  {1, 4, 16384, 24, true},
                  {1, 4, 512, 24, false},
                  {2, 4, 1024, 8, true},
                  {4, 4, 1024, 8, true},
                  {8, 8, 1024, 16, true},
                  {2, 4, 1536, 12, true},
                  {2, 6, 2048, 24, true},
                  {2, 16, 2048, 24, false}}) {
                const int64_t kvLength = topK > 4096 ? (topK + 127) / 128 * 128 : 4096;
                const int64_t pageCount = kvLength / 128;
                int32_t actualSeqQ[8] = {};
                int32_t actualSeqKV[8] = {};
                for (int64_t batch = 0; batch < batchCount; ++batch) {
                    actualSeqQ[batch] = static_cast<int32_t>((batch + 1) * queryCount / batchCount);
                    actualSeqKV[batch] = static_cast<int32_t>(kvLength);
                }
                gert::TilingContextPara para(
                    "SparseFlashAttention",
                    {{{{queryCount, heads, 512}, {queryCount, heads, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                     {{{pageCount, 128, 1, 512}, {pageCount, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                     {{{pageCount, 128, 1, 512}, {pageCount, 128, 1, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                     {{{queryCount, 1, topK}, {queryCount, 1, topK}}, ge::DT_INT32, ge::FORMAT_ND},
                     {{{batchCount, pageCount}, {batchCount, pageCount}}, ge::DT_INT32, ge::FORMAT_ND},
                     {{{batchCount}, {batchCount}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqQ},
                     {{{batchCount}, {batchCount}}, ge::DT_INT32, ge::FORMAT_ND, true, actualSeqKV},
                     {{{queryCount, heads, 64}, {queryCount, heads, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                     {{{pageCount, 128, 1, 64}, {pageCount, 128, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND}},
                    {{{{queryCount, heads, 512}, {queryCount, heads, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                     {{{1, returnLse ? queryCount : 0, heads}, {1, returnLse ? queryCount : 0, heads}},
                      ge::DT_FLOAT,
                      ge::FORMAT_ND},
                     {{{1, returnLse ? queryCount : 0, heads}, {1, returnLse ? queryCount : 0, heads}},
                      ge::DT_FLOAT,
                      ge::FORMAT_ND}},
                    {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0416666666666667)},
                     {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                     {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                     {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
                     {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                     {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                     {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                     {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                     {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(returnLse)}},
                    &compileInfo, "Ascend910B", 24, 262144, 16384);
                for (auto dtype : {ge::DT_BF16, ge::DT_FLOAT16}) {
                    for (bool hasRope : {false, true}) {
                        auto variant = para;
                        for (uint32_t index : {0, 1, 2, 7, 8}) {
                            variant.inputTensorDesc_[index].dtype_ = dtype;
                        }
                        variant.outputTensorDesc_[0].dtype_ = dtype;
                        if (!hasRope) {
                            variant.inputTensorDesc_[7].shape_ = gert::StorageShape{};
                            variant.inputTensorDesc_[8].shape_ = gert::StorageShape{};
                        }
                        TilingInfo info;
                        if (returnLse) {
                            EXPECT_FALSE(ExecuteTiling(variant, info))
                                << "A2 PA_BSND must retain the original LSE rejection";
                            continue;
                        }
                        ASSERT_TRUE(ExecuteTiling(variant, info))
                            << "G=" << heads << ", B=" << batchCount << ", Q=" << queryCount << ", TopK=" << topK;
                        EXPECT_EQ(info.blockNum, expectedBlocks);
                        EXPECT_EQ(info.tilingKey & 1, expectedFd);
                        if (heads == 1 && batchCount == 1 && queryCount == 1 && topK == 2048 && !returnLse &&
                            dtype == ge::DT_BF16 && !hasRope) {
                            auto oddGroup = variant;
                            oddGroup.inputTensorDesc_[0].shape_ = gert::StorageShape{{1, 3, 512}, {1, 3, 512}};
                            oddGroup.outputTensorDesc_[0].shape_ = gert::StorageShape{{1, 3, 512}, {1, 3, 512}};
                            TilingInfo rejected;
                            EXPECT_FALSE(ExecuteTiling(oddGroup, rejected));
                            auto unsupportedType = variant;
                            unsupportedType.inputTensorDesc_[0].dtype_ = ge::DT_FLOAT;
                            unsupportedType.outputTensorDesc_[0].dtype_ = ge::DT_FLOAT;
                            EXPECT_FALSE(ExecuteTiling(unsupportedType, rejected));
                            for (const auto& [blockSize, sparseMode, indexCount, expectedCores, splitEnabled] :
                                 {std::tuple<int64_t, int64_t, int64_t, uint32_t, bool>{1, 3, 2048, 4, true},
                                  {1, 3, 256, 24, false},
                                  {1, 3, 512, 24, false},
                                  {1, 3, 513, 2, true},
                                  {2, 3, 256, 24, false},
                                  {2, 3, 257, 2, true},
                                  {4, 3, 256, 2, true},
                                  {4, 3, 257, 3, true},
                                  {2, 0, 2048, 8, true},
                                  {2, 3, 2048, 8, true},
                                  {4, 0, 512, 4, true},
                                  {4, 3, 512, 4, true},
                                  {4, 0, 2048, 24, false},
                                  {4, 0, 128, 24, false},
                                  {4, 0, 129, 2, true},
                                  {4, 3, 192, 2, true},
                                  {4, 0, 255, 2, true},
                                  {8, 3, 64, 24, false},
                                  {8, 3, 65, 2, true},
                                  {8, 3, 512, 8, true},
                                  {16, 0, 128, 4, true},
                                  {32, 3, 128, 8, true},
                                  {64, 0, 16, 2, true},
                                  {128, 3, 16, 4, true}}) {
                                auto scopeVariant = variant;
                                scopeVariant.attrs_[1].attr_ =
                                    Ops::Transformer::AnyValue::CreateFrom<int64_t>(blockSize);
                                scopeVariant.attrs_[4].attr_ =
                                    Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseMode);
                                scopeVariant.inputTensorDesc_[3].shape_ =
                                    gert::StorageShape{{1, 1, indexCount}, {1, 1, indexCount}};
                                TilingInfo scopeInfo;
                                ASSERT_TRUE(ExecuteTiling(scopeVariant, scopeInfo));
                                EXPECT_EQ(scopeInfo.tilingKey & 1, splitEnabled);
                                EXPECT_EQ(scopeInfo.blockNum, expectedCores);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_F(SparseFlashAttentionTiling, SplitKvNativeLayoutsUseTotalQueryCount)
{
    struct SparseFlashAttentionCompileInfo {
    } compileInfo;
    constexpr int64_t batchCount = 2;
    constexpr int64_t heads = 8;
    constexpr int64_t kvLength = 4096;
    for (const auto& [queryLayout, kvLayout] : {std::pair<std::string, std::string>{"TND", "PA_BSND"},
                                                {"BSND", "PA_BSND"},
                                                {"TND", "TND"},
                                                {"BSND", "BSND"}}) {
        for (int64_t queriesPerBatch : {1, 2, 8, 8192}) {
            const int64_t totalQueries = batchCount * queriesPerBatch;
            const bool bsnd = queryLayout == "BSND";
            const bool page = kvLayout == "PA_BSND";
            int32_t actualQ[] = {static_cast<int32_t>(queriesPerBatch),
                                 static_cast<int32_t>(bsnd ? queriesPerBatch : totalQueries)};
            int32_t actualKV[] = {kvLength, kvLayout == "TND" ? batchCount * kvLength : kvLength};
            const gert::StorageShape qShape =
                bsnd ? gert::StorageShape{{batchCount, queriesPerBatch, heads, 512},
                                          {batchCount, queriesPerBatch, heads, 512}} :
                       gert::StorageShape{{totalQueries, heads, 512}, {totalQueries, heads, 512}};
            const gert::StorageShape indicesShape =
                bsnd ?
                    gert::StorageShape{{batchCount, queriesPerBatch, 1, 2048}, {batchCount, queriesPerBatch, 1, 2048}} :
                    gert::StorageShape{{totalQueries, 1, 2048}, {totalQueries, 1, 2048}};
            const gert::StorageShape kvShape =
                page ? gert::StorageShape{{batchCount * kvLength / 128, 128, 1, 512},
                                          {batchCount * kvLength / 128, 128, 1, 512}} :
                       (bsnd ? gert::StorageShape{{batchCount, kvLength, 1, 512}, {batchCount, kvLength, 1, 512}} :
                               gert::StorageShape{{batchCount * kvLength, 1, 512}, {batchCount * kvLength, 1, 512}});
            for (bool returnLse : {false, true}) {
                const gert::StorageShape lseShape =
                    !returnLse ? gert::StorageShape{{0}, {0}} :
                                 (bsnd ? gert::StorageShape{{batchCount, 1, queriesPerBatch, heads},
                                                            {batchCount, 1, queriesPerBatch, heads}} :
                                         gert::StorageShape{{1, totalQueries, heads}, {1, totalQueries, heads}});
                gert::TilingContextPara para(
                    "SparseFlashAttention",
                    {{qShape, ge::DT_BF16, ge::FORMAT_ND},
                     {kvShape, ge::DT_BF16, ge::FORMAT_ND},
                     {kvShape, ge::DT_BF16, ge::FORMAT_ND},
                     {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                     {page ? gert::StorageShape{{batchCount, kvLength / 128}, {batchCount, kvLength / 128}} :
                             gert::StorageShape{},
                      ge::DT_INT32, ge::FORMAT_ND},
                     {{{batchCount}, {batchCount}}, ge::DT_INT32, ge::FORMAT_ND, true, actualQ},
                     {{{batchCount}, {batchCount}}, ge::DT_INT32, ge::FORMAT_ND, true, actualKV},
                     {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                     {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
                    {{qShape, ge::DT_BF16, ge::FORMAT_ND},
                     {lseShape, ge::DT_FLOAT, ge::FORMAT_ND},
                     {lseShape, ge::DT_FLOAT, ge::FORMAT_ND}},
                    {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.04419417382415922)},
                     {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                     {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>(queryLayout)},
                     {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(kvLayout)},
                     {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                     {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                     {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                     {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                     {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(returnLse)}},
                    &compileInfo, "Ascend910B", 24, 262144, 16384);
                TilingInfo info;
                if (page && returnLse) {
                    EXPECT_FALSE(ExecuteTiling(para, info));
                    continue;
                }
                ASSERT_TRUE(ExecuteTiling(para, info)) << queryLayout << "/" << kvLayout;
                EXPECT_EQ(info.tilingKey & 1, totalQueries <= 12);
                EXPECT_EQ(info.blockNum, totalQueries <= 4 ? totalQueries * 4 : 24);
                if (!page && !returnLse && queriesPerBatch == 1) {
                    auto emptyKv = para;
                    const gert::StorageShape emptyKvShape =
                        bsnd ? gert::StorageShape{{batchCount, 0, 1, 512}, {batchCount, 0, 1, 512}} :
                               gert::StorageShape{{0, 1, 512}, {0, 1, 512}};
                    emptyKv.inputTensorDesc_[1].shape_ = emptyKvShape;
                    emptyKv.inputTensorDesc_[2].shape_ = emptyKvShape;
                    int32_t emptyLengths[] = {0, 0};
                    emptyKv.inputTensorDesc_[6].constValue_ = emptyLengths;
                    TilingInfo emptyInfo;
                    ASSERT_TRUE(ExecuteTiling(emptyKv, emptyInfo));
                    EXPECT_EQ(emptyInfo.tilingKey & 1, 0);
                }
            }
        }
    }
}
