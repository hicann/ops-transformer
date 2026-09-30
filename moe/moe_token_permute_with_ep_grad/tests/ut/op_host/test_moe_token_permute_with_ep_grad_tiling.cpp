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
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/moe_token_permute_with_ep_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_infos_def.h"

using namespace std;

class MoeTokenPermuteWithEpGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteWithEpGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteWithEpGradTiling TearDown" << std::endl;
    }
};

struct MoeTokenPermuteWithEpGradCompileInfo {};

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_bf16)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    std::vector<int64_t> range({1, 49152});
    gert::StorageShape permuted_token_output_d_shape = {{range[1] - range[0], 5120}, {range[1] - range[0], 5120}};
    gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
    gert::StorageShape permuted_probs_output_d_shape = {{range[1] - range[0]}, {range[1] - range[0]}};
    gert::StorageShape input_token_grad_shape = {{6144, 5120}, {6144, 5120}};
    gert::StorageShape input_probs_grad_shape = {{6144, 8}, {6144, 8}};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{permuted_token_output_d_shape, ge::DT_BF16, ge::FORMAT_ND},
         {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
         {permuted_probs_output_d_shape, ge::DT_BF16, ge::FORMAT_ND}},
        {{input_token_grad_shape, ge::DT_BF16, ge::FORMAT_ND}, {input_probs_grad_shape, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_fp16)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    std::vector<int64_t> range({1, 49152});
    gert::StorageShape permuted_token_output_d_shape = {{range[1] - range[0], 5120}, {range[1] - range[0], 5120}};
    gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
    gert::StorageShape permuted_probs_output_d_shape = {{range[1] - range[0]}, {range[1] - range[0]}};
    gert::StorageShape input_token_grad_shape = {{6144, 5120}, {6144, 5120}};
    gert::StorageShape input_probs_grad_shape = {{6144, 8}, {6144, 8}};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{permuted_token_output_d_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
         {permuted_probs_output_d_shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{input_token_grad_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {input_probs_grad_shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_fp32)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    std::vector<int64_t> range({1, 49152});
    gert::StorageShape permuted_token_output_d_shape = {{range[1] - range[0], 5120}, {range[1] - range[0], 5120}};
    gert::StorageShape sorted_indices_shape = {{49152}, {49152}};
    gert::StorageShape permuted_probs_output_d_shape = {{range[1] - range[0]}, {range[1] - range[0]}};
    gert::StorageShape input_token_grad_shape = {{6144, 5120}, {6144, 5120}};
    gert::StorageShape input_probs_grad_shape = {{6144, 8}, {6144, 8}};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{permuted_token_output_d_shape, ge::DT_FLOAT, ge::FORMAT_ND},
         {sorted_indices_shape, ge::DT_INT32, ge::FORMAT_ND},
         {permuted_probs_output_d_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{input_token_grad_shape, ge::DT_FLOAT, ge::FORMAT_ND}, {input_probs_grad_shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_no_probs_bf16)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 49152});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{{{49152, 5120}, {49152, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{6144, 5120}, {6144, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{6144, 8}, {6144, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_small_shape)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 16});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{{{16, 128}, {16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{16, 128}, {16, 128}}, ge::DT_BF16, ge::FORMAT_ND}, {{{16, 8}, {16, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_multi_core)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    int64_t tokenNum = 16384;
    int64_t topk = 8;
    std::vector<int64_t> range({0, tokenNum});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{{{tokenNum, 5120}, {tokenNum, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{tokenNum, 5120}, {tokenNum, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topk)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_long_hidden)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    int64_t tokenNum = 256;
    int64_t topk = 8;
    int64_t hidden = 32768;
    std::vector<int64_t> range({0, tokenNum});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{{{tokenNum, hidden}, {tokenNum, hidden}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{tokenNum, hidden}, {tokenNum, hidden}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topk)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_topk_invalid)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 16});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{{{16, 128}, {16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{16, 128}, {16, 128}}, ge::DT_BF16, ge::FORMAT_ND}, {{{16, 8}, {16, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_tokens_not_2d)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 16});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{{{16}, {16}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{16, 128}, {16, 128}}, ge::DT_BF16, ge::FORMAT_ND}, {{{16, 8}, {16, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_probs_not_1d)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 16});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{{{16, 128}, {16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{16, 8}, {16, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{16, 128}, {16, 128}}, ge::DT_BF16, ge::FORMAT_ND}, {{{16, 8}, {16, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_range_size_invalid)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 16, 1});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{{{16, 128}, {16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{16, 128}, {16, 128}}, ge::DT_BF16, ge::FORMAT_ND}, {{{16, 8}, {16, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_int64_indices)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    std::vector<int64_t> range({0, 16});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{{{16, 128}, {16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{16, 128}, {16, 128}}, ge::DT_BF16, ge::FORMAT_ND}, {{{16, 8}, {16, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_tight_ub_token_split)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    int64_t tokenNum = 4096;
    int64_t topk = 8;
    int64_t hidden = 8192;
    std::vector<int64_t> range({0, tokenNum});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{{{tokenNum, hidden}, {tokenNum, hidden}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{tokenNum, hidden}, {tokenNum, hidden}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topk)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 8192);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_parse)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    const char* compileJson = R"({"hardware_info": {"UB_SIZE": 262144, "CORE_NUM": 64, "socVersion": "Ascend910B"}})";
    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.OpType(ge::AscendString("MoeTokenPermuteWithEpGrad"))
                      .OpName(ge::AscendString("MoeTokenPermuteWithEpGrad"))
                      .IOInstanceNum({1, 1, 1}, {1, 1})
                      .InputTensorDesc(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensorDesc(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .OutputTensorDesc(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompiledJson(compileJson)
                      .CompiledInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .Build();
    auto* parseContext = holder.GetContext();
    ASSERT_NE(parseContext, nullptr);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeTokenPermuteWithEpGrad");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext*>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_extreme_ub_token_split)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    int64_t tokenNum = 8192;
    int64_t topk = 8;
    int64_t hidden = 65536;
    std::vector<int64_t> range({0, tokenNum});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{{{tokenNum, hidden}, {tokenNum, hidden}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{tokenNum, hidden}, {tokenNum, hidden}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topk)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_minimal_ub_multi_core)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    int64_t tokenNum = 16384;
    int64_t topk = 8;
    int64_t hidden = 32768;
    std::vector<int64_t> range({0, tokenNum});
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{{{tokenNum, hidden}, {tokenNum, hidden}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{tokenNum, hidden}, {tokenNum, hidden}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topk)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(range)},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo, "Ascend910B", 64, 2048);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_no_probs_fp16)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 49152})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}

TEST_F(MoeTokenPermuteWithEpGradTiling, test_tiling_no_probs_fp32)
{
    MoeTokenPermuteWithEpGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenPermuteWithEpGrad",
        {{{{49152, 5120}, {49152, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{49152}, {49152}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{6144, 5120}, {6144, 5120}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{6144, 8}, {6144, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"num_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"range", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0, 49152})},
         {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
}
