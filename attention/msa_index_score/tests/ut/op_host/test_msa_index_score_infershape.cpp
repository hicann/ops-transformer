/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MsaIndexScoreProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MsaIndexScoreProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MsaIndexScoreProto TearDown" << std::endl;
    }
};

// BBND PageAttention：block_table 末维 2，score 末维 RoundUp(2, 16)=16
TEST_F(MsaIndexScoreProto, MsaIndexScore_infershape_bbnd)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MsaIndexScore",
        {
            {{{32, 8, 128}, {32, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},         // query
            {{{2, 128, 1, 128}, {2, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key BBND
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},                       // block_table
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},                             // scale
            {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},                              // atten_mask
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},                             // actual_seq_qlen
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                             // actual_seq_klen
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}                              // start_loc
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} // score
        },
        {{"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BBND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 32, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// BNBD：block_table 末维 20，score 末维 RoundUp(20, 16)=32
TEST_F(MsaIndexScoreProto, MsaIndexScore_infershape_bnbd)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MsaIndexScore",
        {{{{48, 4, 128}, {48, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{4, 1, 128, 128}, {4, 1, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 20}, {2, 20}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNBD")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 48, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TND：actual_seq_klen 前缀和差 256，逻辑 block 数 ceil(256/128)=2，对齐到 16
TEST_F(MsaIndexScoreProto, MsaIndexScore_infershape_tnd_prefix)
{
    int32_t klenList[] = {0, 256};
    gert::InfershapeContextPara infershapeContextPara(
        "MsaIndexScore",
        {{{{32, 8, 128}, {32, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2048, 2048}, {2048, 2048}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, klenList},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 32, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TND 多 batch，取各请求 kv 的最大 block 数
TEST_F(MsaIndexScoreProto, MsaIndexScore_infershape_tnd_multi_batch)
{
    int32_t klenList[] = {0, 128, 512};
    gert::InfershapeContextPara infershapeContextPara(
        "MsaIndexScore",
        {{{{64, 2, 128}, {64, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{512, 1, 128}, {512, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, klenList},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    // max kv = 384, blocks = 3, RoundUp(3, 16) = 16
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 64, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 整 batch kv_len=0：按 1 个 dummy block 再 16 对齐
TEST_F(MsaIndexScoreProto, MsaIndexScore_infershape_tnd_empty_kv)
{
    int32_t klenList[] = {0, 0};
    gert::InfershapeContextPara infershapeContextPara(
        "MsaIndexScore",
        {{{{8, 1, 128}, {8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{0, 1, 128}, {0, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, klenList},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 8, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MsaIndexScoreProto, MsaIndexScore_infershape_query_rank_invalid)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MsaIndexScore",
        {{{{1, 32, 8, 128}, {1, 32, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 128, 1, 128}, {2, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BBND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MsaIndexScoreProto, MsaIndexScore_infershape_layout_invalid)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MsaIndexScore",
        {{{{32, 8, 128}, {32, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 128, 1, 128}, {2, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSH")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// 非 V2：query TND，key BSND，klen 为 [B+1] 前缀和，逻辑块长 128。kv=200 → 2 blocks → 对齐 16。
TEST_F(MsaIndexScoreProto, MsaIndexScore_infershape_bsnd_pool128)
{
    int32_t klenList[] = {0, 200};
    gert::InfershapeContextPara infershapeContextPara(
        "MsaIndexScore",
        {{{{32, 8, 128}, {32, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 256, 1, 128}, {1, 256, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, klenList},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 32, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// BNSD 多 batch：klen 为前缀和，取各请求可见长度的最大 block 数。
TEST_F(MsaIndexScoreProto, MsaIndexScore_infershape_bnsd_pool128)
{
    int32_t klenList[] = {0, 100, 400};
    gert::InfershapeContextPara infershapeContextPara(
        "MsaIndexScore",
        {{{{48, 4, 128}, {48, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2, 1, 320, 128}, {2, 1, 320, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, klenList},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    // max kv = 300, blocks = 3, RoundUp(3, 16) = 16
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 48, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MsaIndexScoreProto, MsaIndexScore_infershape_block_table_rank_invalid)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MsaIndexScore",
        {{{{32, 8, 128}, {32, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2, 128, 1, 128}, {2, 128, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BBND")},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(MsaIndexScoreProto, MsaIndexScore_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MsaIndexScore")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType queryDtype = ge::DT_BF16;
        ge::DataType keyDtype = ge::DT_BF16;
        ge::DataType i32Dtype = ge::DT_INT32;
        ge::DataType f32Dtype = ge::DT_FLOAT;
        ge::DataType maskDtype = ge::DT_INT8;
        ge::DataType outputDtype = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(8, 1)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes(
                    {&queryDtype, &keyDtype, &i32Dtype, &f32Dtype, &maskDtype, &i32Dtype, &i32Dtype, &i32Dtype})
                .NodeAttrs({{"layout_key", Ops::Transformer::AnyValue::CreateFrom<std::string>("BBND")},
                            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                            {"init_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                            {"local_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetOutputDataType(0), outputDtype);
    }
}
