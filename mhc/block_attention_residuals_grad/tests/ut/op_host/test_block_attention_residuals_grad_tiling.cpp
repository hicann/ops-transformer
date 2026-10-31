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
#include "../../../op_host/block_attention_residuals_grad_tiling.h"
#include "tiling_case_executor.h"

namespace {
constexpr int64_t TOKENS = 64;
constexpr int64_t BLOCKS = 4;
constexpr int64_t HIDDEN = 2048;
constexpr uint64_t UB_SIZE = 262144;

gert::TilingContextPara MakeContext(BlockAttentionResidualsGradCompileInfo& compileInfo, int64_t hidden = HIDDEN,
                                    uint64_t cores = 8, uint64_t ub = UB_SIZE)
{
    return gert::TilingContextPara(
        "BlockAttentionResidualsGrad",
        {
            {{{TOKENS, hidden}, {TOKENS, hidden}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{TOKENS, BLOCKS, hidden}, {TOKENS, BLOCKS, hidden}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, hidden}, {1, hidden}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hidden}, {hidden}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{TOKENS, hidden}, {TOKENS, hidden}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{TOKENS, BLOCKS + 1}, {TOKENS, BLOCKS + 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{TOKENS, BLOCKS + 1}, {TOKENS, BLOCKS + 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{TOKENS, hidden}, {TOKENS, hidden}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{TOKENS, BLOCKS, hidden}, {TOKENS, BLOCKS, hidden}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, hidden}, {1, hidden}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{hidden}, {hidden}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"valid_block_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)}}, &compileInfo, "Ascend910B", cores,
        ub);
}

TEST(BlockAttentionResidualsGradPlatform, EmptyParseCacheUsesRuntimeResources)
{
    BlockAttentionResidualsGradCompileInfo compileInfo{};
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakeContext(compileInfo), info));
    EXPECT_GT(info.blockNum, 0U);
    EXPECT_EQ(info.tilingKey, 0);
}

TEST(BlockAttentionResidualsGradPlatform, StaleParseCacheDoesNotOverrideRuntimeResources)
{
    BlockAttentionResidualsGradCompileInfo compileInfo{4096, 96};
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakeContext(compileInfo), info));
    EXPECT_GT(info.blockNum, 0U);
    EXPECT_NE(info.blockNum, compileInfo.coreNum);
}

TEST(BlockAttentionResidualsGradPlatform, WorkspaceUsesLaunchedCoreCount)
{
    BlockAttentionResidualsGradCompileInfo compileInfo{};
    TilingInfo eightCores;
    TilingInfo twelveCores;
    ASSERT_TRUE(ExecuteTiling(MakeContext(compileInfo, HIDDEN, 8), eightCores));
    ASSERT_TRUE(ExecuteTiling(MakeContext(compileInfo, HIDDEN, 12), twelveCores));
    ASSERT_EQ(eightCores.workspaceSizes.size(), 1U);
    ASSERT_EQ(twelveCores.workspaceSizes.size(), 1U);
    ASSERT_GT(twelveCores.blockNum, eightCores.blockNum);
    EXPECT_EQ(twelveCores.workspaceSizes[0] - eightCores.workspaceSizes[0],
              static_cast<int64_t>(twelveCores.blockNum - eightCores.blockNum) * HIDDEN * sizeof(float));
}

TEST(BlockAttentionResidualsGradPlatform, SplitHUsesRuntimeResourcesWithEmptyParseCache)
{
    BlockAttentionResidualsGradCompileInfo compileInfo{};
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(MakeContext(compileInfo, 65536), info));
    EXPECT_GT(info.blockNum, 0U);
    EXPECT_EQ(info.tilingKey, 1);
}

TEST(BlockAttentionResidualsGradPlatform, ZeroRuntimeCoreCountIsRejected)
{
    BlockAttentionResidualsGradCompileInfo compileInfo{UB_SIZE, 8};
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(MakeContext(compileInfo, HIDDEN, 0), info));
}

TEST(BlockAttentionResidualsGradPlatform, InsufficientRuntimeUbIsRejected)
{
    BlockAttentionResidualsGradCompileInfo compileInfo{UB_SIZE, 8};
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(MakeContext(compileInfo, HIDDEN, 8, 1024), info));
}
} // namespace
