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
#include <vector>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

class StemIndexerTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StemIndexerTilingArch35 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StemIndexerTilingArch35 TearDown" << std::endl;
    }
};

namespace {
using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

constexpr int64_t EXPECTED_WORKSPACE_ASCEND950 = 29360128;
constexpr int64_t TOPK_SCORE_PRECISION_UINT16 = 2;
constexpr int64_t INVALID_TOPK_SCORE_PRECISION = 3;

struct StemIndexerCompileInfo {};

std::vector<TensorDesc> MakeValidInputs(int64_t batch = 2, int64_t qHeads = 32, int64_t kvHeads = 4, int64_t maxQb = 8,
                                        int64_t maxKb = 16, int64_t flattenDim = 2048)
{
    return {
        {{{batch, qHeads, maxQb, flattenDim}, {batch, qHeads, maxQb, flattenDim}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{batch, kvHeads, maxKb, flattenDim}, {batch, kvHeads, maxKb, flattenDim}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{batch, kvHeads, maxKb}, {batch, kvHeads, maxKb}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{batch}, {batch}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{batch}, {batch}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{batch}, {batch}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{2048}, {2048}}, ge::DT_INT32, ge::FORMAT_ND},
    };
}

std::vector<TensorDesc> MakeValidOutputs(int64_t batch = 2, int64_t qHeads = 32, int64_t maxQb = 8, int64_t maxKb = 16)
{
    return {
        {{{batch, qHeads, maxQb, maxKb}, {batch, qHeads, maxQb, maxKb}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{batch, qHeads, maxQb}, {batch, qHeads, maxQb}}, ge::DT_INT32, ge::FORMAT_ND},
    };
}

std::vector<OpAttr> MakeValidAttrs(bool causal = true, int64_t topkScorePrecision = 1)
{
    return {
        {"causal", Ops::Transformer::AnyValue::CreateFrom<bool>(causal)},
        {"stem_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
        {"stem_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        {"alpha", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"initial_blocks", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        {"window_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        {"k_block_num_rate_medium", Ops::Transformer::AnyValue::CreateFrom<float>(0.2f)},
        {"k_block_num_bias_medium", Ops::Transformer::AnyValue::CreateFrom<int64_t>(30)},
        {"k_block_num_rate_large", Ops::Transformer::AnyValue::CreateFrom<float>(0.1f)},
        {"k_block_num_bias_large", Ops::Transformer::AnyValue::CreateFrom<int64_t>(30)},
        {"topk_score_precision", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topkScorePrecision)},
    };
}

gert::TilingContextPara BuildTilingPara(const std::vector<TensorDesc> &inputs, const std::vector<TensorDesc> &outputs,
                                        const std::vector<OpAttr> &attrs)
{
    static StemIndexerCompileInfo compileInfo;
    return gert::TilingContextPara("StemIndexer", inputs, outputs, attrs, &compileInfo, "Ascend950", 64, 262144, 16384);
}

void ExpectTilingResult(const gert::TilingContextPara &tilingContextPara, bool expectSuccess)
{
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(ok, expectSuccess);
    if (expectSuccess) {
        ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
        EXPECT_EQ(tilingInfo.workspaceSizes[0], EXPECTED_WORKSPACE_ASCEND950);
        EXPECT_GT(tilingInfo.blockNum, 0U);
        EXPECT_GT(tilingInfo.tilingDataSize, 0U);
    }
}
} // namespace

TEST_F(StemIndexerTilingArch35, StemIndexer_950_tiling_basic)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeValidAttrs()), true);
}

TEST_F(StemIndexerTilingArch35, StemIndexer_950_tiling_causal_false_q64_kv8)
{
    ExpectTilingResult(
        BuildTilingPara(MakeValidInputs(1, 64, 8, 4, 32), MakeValidOutputs(1, 64, 4, 32), MakeValidAttrs(false)), true);
}

TEST_F(StemIndexerTilingArch35, StemIndexer_950_tiling_topk_score_uint16)
{
    ExpectTilingResult(
        BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeValidAttrs(true, TOPK_SCORE_PRECISION_UINT16)),
        true);
}

TEST_F(StemIndexerTilingArch35, StemIndexer_950_tiling_topk_score_precision_invalid)
{
    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeValidAttrs(true, 0)), false);
    ExpectTilingResult(
        BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), MakeValidAttrs(true, INVALID_TOPK_SCORE_PRECISION)),
        false);
}

TEST_F(StemIndexerTilingArch35, StemIndexer_950_tiling_qflat_dtype_invalid)
{
    auto inputs = MakeValidInputs();
    inputs[0] = TensorDesc({{2, 32, 8, 2048}, {2, 32, 8, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(StemIndexerTilingArch35, StemIndexer_950_tiling_q_head_invalid)
{
    auto inputs = MakeValidInputs(2, 16, 4, 8, 16);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(StemIndexerTilingArch35, StemIndexer_950_tiling_flatten_dim_invalid)
{
    auto inputs = MakeValidInputs(2, 32, 4, 8, 16, 1024);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(StemIndexerTilingArch35, StemIndexer_950_tiling_vbias_shape_invalid)
{
    auto inputs = MakeValidInputs();
    inputs[2] = TensorDesc({{2, 2, 16}, {2, 2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(inputs, MakeValidOutputs(), MakeValidAttrs()), false);
}

TEST_F(StemIndexerTilingArch35, StemIndexer_950_tiling_sparse_indices_shape_invalid)
{
    auto outputs = MakeValidOutputs();
    outputs[0] = TensorDesc({{2, 32, 8, 15}, {2, 32, 8, 15}}, ge::DT_INT32, ge::FORMAT_ND);

    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), outputs, MakeValidAttrs()), false);
}

TEST_F(StemIndexerTilingArch35, StemIndexer_950_tiling_stem_stride_invalid)
{
    auto attrs = MakeValidAttrs();
    attrs[2] = OpAttr("stem_stride", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8));

    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), attrs), false);
}

TEST_F(StemIndexerTilingArch35, StemIndexer_950_tiling_alpha_invalid)
{
    auto attrs = MakeValidAttrs();
    attrs[3] = OpAttr("alpha", Ops::Transformer::AnyValue::CreateFrom<float>(0.0f));

    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), attrs), false);
}

TEST_F(StemIndexerTilingArch35, StemIndexer_950_tiling_attr_missing)
{
    auto attrs = MakeValidAttrs();
    attrs.pop_back();

    ExpectTilingResult(BuildTilingPara(MakeValidInputs(), MakeValidOutputs(), attrs), false);
}
