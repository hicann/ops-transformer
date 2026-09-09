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

// DAV_3510 (Ascend950) tiling cases for LightningIndexer
class LightningIndexerTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LightningIndexerTilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LightningIndexerTilingArch35 TearDown" << std::endl;
    }
};

// PA_BSND key with valid params, success on Ascend950
TEST_F(LightningIndexerTilingArch35, LightningIndexer_950_tiling_0)
{
    struct LightningIndexerCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {39, 39};
    int64_t actual_seq_kvlist[] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexer",
        {
            {{{2, 39, 64, 128}, {2, 39, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // query        input0
            {{{2, 16, 1, 128}, {2, 16, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // key          input1
            {{{2, 39, 64}, {2, 39, 64}}, ge::DT_BF16, ge::FORMAT_ND},           // weights      input2
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},  // actual_seq_lengths_query
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist}, // actual_seq_lengths_key
            {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND}                     // block_table  input3
        },
        {
            {{{2, 39, 1, 2048}, {2, 39, 1, 2048}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_BF16, ge::FORMAT_ND}                             // sparse_values
        },
        {{"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2048)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"return_values", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// sparse_count out of range is rejected on Ascend950
TEST_F(LightningIndexerTilingArch35, LightningIndexer_950_tiling_1)
{
    struct LightningIndexerCompileInfo {
    } compileInfo;
    int64_t actual_seq_qlist[] = {509};
    int64_t actual_seq_kvlist[] = {1111};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexer",
        {
            {{{1, 1856, 8, 128}, {1, 1856, 8, 128}}, ge::DT_BF16, ge::FORMAT_ND},     // query        input0
            {{{1, 131072, 1, 128}, {1, 131072, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // key          input1
            {{{1, 1856, 8}, {1, 1856, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},              // weights      input2
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},        // actual_seq_lengths_query
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},       // actual_seq_lengths_key
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}                                 // block_table  input3
        },
        {
            {{{1, 1856, 1, 1024}, {1, 1856, 1, 1024}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{1, 1856, 1, 1024}, {1, 1856, 1, 1024}}, ge::DT_BF16, ge::FORMAT_ND}   // sparse_values
        },
        {{"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1024)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"return_values", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// key shape[2] is numhead, only support 1
TEST_F(LightningIndexerTilingArch35, LightningIndexer_950_tiling_2)
{
    struct LightningIndexerCompileInfo {
    } compileInfo;
    int64_t *actual_seq_qlist = nullptr;
    int64_t actual_seq_kvlist[] = {16484, 16484, 16484, 16484};
    gert::TilingContextPara tilingContextPara(
        "LightningIndexer",
        {
            {{{4, 16484, 64, 128}, {4, 16484, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query        input0
            {{{4, 1, 16484, 128}, {4, 1, 16484, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // key          input1
            {{{4, 16484, 64}, {4, 16484, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},             // weights      input2
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_qlist},           // actual_seq_lengths_query
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, actual_seq_kvlist},          // actual_seq_lengths_key
            {{{4, 129}, {4, 129}}, ge::DT_INT32, ge::FORMAT_ND}                          // block_table  input3
        },
        {
            {{{4, 16484, 1, 3072}, {4, 16484, 1, 3072}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND}                                // sparse_values
        },
        {{"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
         {"sparse_count", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3072)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
         {"return_values", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend950", 56, 262144, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Tiling data classes are registered for the op
TEST_F(LightningIndexerTilingArch35, LightningIndexer_tiling_data_class_registered)
{
    auto &factory = optiling::CTilingDataClassFactory::GetInstance();
    EXPECT_NE(factory.CreateTilingDataInstance("LightningIndexer"), nullptr);
}
