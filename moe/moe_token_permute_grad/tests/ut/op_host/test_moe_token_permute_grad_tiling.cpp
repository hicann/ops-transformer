/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/moe_token_permute_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

struct MoeTokenPermuteGradCompileInfo {};
class MoeTokenPermuteGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteGradTiling TearDown" << std::endl;
    }
};

TEST_F(MoeTokenPermuteGradTiling, test_tiling_bf16)
{
    MoeTokenPermuteGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteGrad",
                                              {
                                                  {{{49152, 5120}, {49152, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6144, 5120}, {6144, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                  {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "5120 8 49152 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteGradTiling, test_tiling_fp16)
{
    MoeTokenPermuteGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteGrad",
                                              {
                                                  {{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                  {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "5120 8 49152 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteGradTiling, test_tiling_fp32)
{
    MoeTokenPermuteGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteGrad",
                                              {
                                                  {{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                  {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "5120 8 49152 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteGradTiling, test_tiling_small_tokens)
{
    MoeTokenPermuteGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteGrad",
                                              {
                                                  {{{32, 5120}, {32, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{32}, {32}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 5120}, {4, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                  {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "5120 8 32 5120 1 0 1 0 1 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteGradTiling, test_tiling_split_hidden_bf16)
{
    MoeTokenPermuteGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteGrad",
                                              {
                                                  {{{49152, 8192}, {49152, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6144, 8192}, {6144, 8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                  {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "8192 8 49152 8192 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteGradTiling, test_tiling_split_hidden_fp32)
{
    MoeTokenPermuteGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteGrad",
                                              {
                                                  {{{49152, 8192}, {49152, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6144, 8192}, {6144, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                  {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "8192 8 49152 8192 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteGradTiling, test_tiling_small_ub_large_topk)
{
    MoeTokenPermuteGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteGrad",
                                              {
                                                  {{{49152, 5120}, {49152, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6144, 5120}, {6144, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                                  {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              },
                                              &compileInfo, "Ascend910B", 64, 65536);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "5120 64 49152 4096 1 1024 12 0 12 1 0 2 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteGradTiling, test_tiling_small_ub_token_split)
{
    MoeTokenPermuteGradCompileInfo compileInfo = {};
    int64_t totalLength = 196608;
    int64_t topk = 512;
    int64_t tokensNum = totalLength / topk;
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteGrad",
        {
            {{{totalLength, 5120}, {totalLength, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{totalLength}, {totalLength}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{tokensNum, 5120}, {tokensNum, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topk)},
            {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend910B", 64, 32768);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "5120 512 196608 1536 3 512 6 0 3 2 0 3 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteGradTiling, test_tiling_split_hidden_fp32_large)
{
    MoeTokenPermuteGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteGrad",
                                              {
                                                  {{{49152, 32768}, {49152, 32768}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6144, 32768}, {6144, 32768}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                  {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "32768 8 49152 20992 1 11776 96 0 96 1 0 2 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteGradTiling, test_tiling_with_probs_bf16)
{
    MoeTokenPermuteGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteGrad",
                                              {
                                                  {{{49152, 5120}, {49152, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6144, 8}, {6144, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6144, 5120}, {6144, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                  {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "5120 8 49152 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteGradTiling, test_tiling_with_probs_fp32_mix)
{
    MoeTokenPermuteGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermuteGrad",
                                              {
                                                  {{{49152, 5120}, {49152, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{6144, 5120}, {6144, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                  {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 25;
    string expectTilingData = "5120 8 49152 5120 1 0 96 0 96 1 0 4 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteGradTiling, MoeTokenPermuteGrad_tiling_data_field_access)
{
    optiling::MoeTokenPermuteGradTilingData tilingData;
    tilingData.set_hidden_size(5120);
    tilingData.set_top_k(8);
    tilingData.set_num_out_tokens(49152);
    tilingData.set_buffer_num(4);
    EXPECT_EQ(tilingData.get_hidden_size(), 5120);
    EXPECT_EQ(tilingData.get_top_k(), 8);
    EXPECT_EQ(tilingData.get_num_out_tokens(), 49152);
    EXPECT_EQ(tilingData.get_buffer_num(), 4);
}

TEST_F(MoeTokenPermuteGradTiling, test_tiling_parse)
{
    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.Build();
    auto parseContext = holder.GetContext();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeTokenPermuteGrad");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext*>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
