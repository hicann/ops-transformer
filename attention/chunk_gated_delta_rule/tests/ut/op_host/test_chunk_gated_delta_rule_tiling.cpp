/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_chunk_gated_delta_rule_tiling.cpp
 * \brief Chunk Gated Delta Rule Tiling 单元测试
 *
 * 测试类别:
 *   - 基础正确性测试（有/无 gamma）
 *   - 不同 shape 组合测试（batch、heads、dims）
 *   - 边界测试（较大 shape、维度约束）
 *   - GQA 模式测试（Grouped Query Attention）
 *   - scale 参数变化测试
 *   - 异常输入测试（nv 不是 nk 的整数倍）
 *   - dtype 异常测试（输入/输出/gamma dtype 错误）
 *   - shape/rank 异常测试（rank 错误、维度不匹配、输出 shape 错误）
 *   - format 异常测试（不支持的 tensor format）
 *
 * 约束条件:
 *   - nv 必须是 nk 的整数倍（nv % nk == 0）
 *   - nk, nv 最大为 64
 *   - dk, dv 最大为 128
 *   - bs（batch size）最大为 8
 */

#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>

#include "../../../op_host/chunk_gated_delta_rule_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace optiling;

// ============================================================================
// 约束常量
// ============================================================================
constexpr uint32_t MAX_BATCH_SIZE = 8;
constexpr uint32_t MAX_HEAD_NUM = 64;
constexpr uint32_t MAX_DIM = 128;

class ChunkGatedDeltaRuleTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // std::cout << "ChunkGatedDeltaRuleTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        // std::cout << "ChunkGatedDeltaRuleTilingTest TearDown" << std::endl;
    }
};

// ============================================================================
// 1. 基础正确性测试 - 有/无 gamma
// ============================================================================
TEST_F(ChunkGatedDeltaRuleTilingTest, BasicWithGamma)
{
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};

    uint32_t bs = 2;
    uint32_t seqLen = 100;
    uint32_t t = bs * seqLen;
    uint32_t nk = 4;
    uint32_t nv = 8; // nv = 2 * nk, 符合约束
    uint32_t dk = 128;
    uint32_t dv = 128;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};
    gert::StorageShape gShape = {{t, nv}, {t, nv}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {
                                                  {queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                              },
                                              &compileinfo);

    int64_t expectTilingKey = 0UL;
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, BasicWithoutGamma)
{
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};

    uint32_t bs = 2;
    uint32_t seqLen = 100;
    uint32_t t = bs * seqLen;
    uint32_t nk = 4;
    uint32_t nv = 8;
    uint32_t dk = 128;
    uint32_t dv = 128;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {
                                                  {queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                              },
                                              &compileinfo);

    int64_t expectTilingKey = 0UL;
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, BasicFp32StateWithoutGamma)
{
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};

    uint32_t bs = 2;
    uint32_t seqLen = 100;
    uint32_t t = bs * seqLen;
    uint32_t nk = 4;
    uint32_t nv = 8;
    uint32_t dk = 128;
    uint32_t dv = 128;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {
                                                  {queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {stateShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                              },
                                              &compileinfo, "Ascend950");

    int64_t expectTilingKey = 1UL;
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(ChunkGatedDeltaRuleTilingTest, BasicFp32StateWithGamma)
{
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};

    uint32_t bs = 2;
    uint32_t seqLen = 100;
    uint32_t t = bs * seqLen;
    uint32_t nk = 4;
    uint32_t nv = 8;
    uint32_t dk = 128;
    uint32_t dv = 128;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};
    gert::StorageShape gShape = {{t, nv}, {t, nv}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {
                                                  {queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {stateShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                              },
                                              &compileinfo, "Ascend950");

    int64_t expectTilingKey = 11UL;
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

// ============================================================================
// 2. 不同 batch size 测试
// ============================================================================
TEST_F(ChunkGatedDeltaRuleTilingTest, SingleBatch)
{
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 1;
    uint32_t seqLen = 64;
    uint32_t t = bs * seqLen;
    uint32_t nk = 4, nv = 4, dk = 64, dv = 64;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(ChunkGatedDeltaRuleTilingTest, MaxBatch8)
{
    // 最大batch = 8
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 8; // 最大值
    uint32_t seqLen = 32;
    uint32_t t = bs * seqLen;
    uint32_t nk = 4, nv = 4, dk = 64, dv = 64;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// ============================================================================
// 3. GQA 模式测试（nv 是 nk 的整数倍）
// ============================================================================
TEST_F(ChunkGatedDeltaRuleTilingTest, GQA_2x)
{
    // GQA: nv = 2 * nk
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 4, nv = 8, dk = 64, dv = 64; // nv = 2 * nk

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(ChunkGatedDeltaRuleTilingTest, GQA_4x)
{
    // GQA: nv = 4 * nk
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 2, nv = 8, dk = 64, dv = 64; // nv = 4 * nk

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(ChunkGatedDeltaRuleTilingTest, MaxHeads48)
{
    // 最大head数 nk=nv=48
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 48, nv = 48, dk = 32, dv = 32; // nk, nv = 48 (最大值)

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// ============================================================================
// 4. 不同维度测试
// ============================================================================
TEST_F(ChunkGatedDeltaRuleTilingTest, Dim128)
{
    // 最大维度 dk=dv=128
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 4, nv = 4, dk = 128, dv = 128; // dk, dv = 128 (最大值)

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(ChunkGatedDeltaRuleTilingTest, DifferentDKDV)
{
    // dk != dv
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 4, nv = 4, dk = 64, dv = 128;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// ============================================================================
// 5. scale 参数变化测试
// ============================================================================
TEST_F(ChunkGatedDeltaRuleTilingTest, ScaleInvSqrt)
{
    // scale = 1/sqrt(dk) = 1/sqrt(64) = 0.125
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 4, nv = 4, dk = 64, dv = 64;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// ============================================================================
// 6. 异常输入测试 - nv 不是 nk 的整数倍（期望失败）
// ============================================================================
TEST_F(ChunkGatedDeltaRuleTilingTest, Invalid_NvNotMultipleOfNk_3_4)
{
    // nv=3, nk=4: 3 % 4 != 0, 期望失败
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 4, nv = 3, dk = 64, dv = 64; // nv 不是 nk 的倍数

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(ChunkGatedDeltaRuleTilingTest, Invalid_NvNotMultipleOfNk_5_4)
{
    // nv=5, nk=4: 5 % 4 != 0, 期望失败
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 4, nv = 5, dk = 64, dv = 64; // nv 不是 nk 的倍数

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(ChunkGatedDeltaRuleTilingTest, Invalid_NvNotMultipleOfNk_7_8)
{
    // nv=7, nk=8: 7 % 8 != 0, 期望失败
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 8, nv = 7, dk = 64, dv = 64; // nv 不是 nk 的倍数

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}
// ============================================================================
// 7. Dtype 异常场景（期望失败）
// ============================================================================
// 输出 out 的 dtype 非 BF16，期望在 AnalyzeDtype 中被拦截
TEST_F(ChunkGatedDeltaRuleTilingTest, Invalid_OutDtype_Fp16)
{
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 4, nv = 8, dk = 64, dv = 64;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// 输出 final_state 的 dtype 非 BF16，期望在 AnalyzeDtype 中被拦截
TEST_F(ChunkGatedDeltaRuleTilingTest, Invalid_FinalStateDtype_Fp16)
{
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 4, nv = 8, dk = 64, dv = 64;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// actual_seq_lengths 的 dtype 非 INT32，期望在 AnalyzeDtype 中被拦截
TEST_F(ChunkGatedDeltaRuleTilingTest, Invalid_ActualSeqLengthsDtype_Int64)
{
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 4, nv = 8, dk = 64, dv = 64;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT64, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// 可选输入 gamma 的 dtype 非 FLOAT，期望在 AnalyzeDtype 中被拦截
TEST_F(ChunkGatedDeltaRuleTilingTest, Invalid_GammaDtype_Bf16)
{
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 4, nv = 8, dk = 64, dv = 64;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};
    gert::StorageShape gShape = {{t, nv}, {t, nv}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND},
                                               {gShape, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}
// ============================================================================
// 8. Shape / Rank 异常场景（期望失败）
// ============================================================================
// query 的 rank 错误（2 维而非 3 维），期望在 AnalyzeShapes 中被拦截
TEST_F(ChunkGatedDeltaRuleTilingTest, Invalid_QueryRank2)
{
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 4, nv = 8, dk = 64, dv = 64;

    gert::StorageShape queryShape = {{t, dk}, {t, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// query 和 key 的 Nk 维不一致，期望在输入 shape 一致性校验中被拦截
TEST_F(ChunkGatedDeltaRuleTilingTest, Invalid_QueryKeyNkMismatch)
{
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 4, nv = 8, dk = 64, dv = 64;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk + 1, dk}, {t, nk + 1, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// final_state 的第一维错误地写成 T 而不是 B，期望在输出 shape 校验中被拦截
TEST_F(ChunkGatedDeltaRuleTilingTest, Invalid_FinalStateShape_TInsteadOfB)
{
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 4, nv = 8, dk = 64, dv = 64;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{t, nv, dv, dk}, {t, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// ============================================================================
// 9. Format 异常场景（期望失败）
// ============================================================================
// state 的 format 为当前不支持的 FRACTAL_NZ，期望在 AnalyzeFormat 阶段校验失败
TEST_F(ChunkGatedDeltaRuleTilingTest, Invalid_StateFormat_NZ)
{
    optiling::ChunkGatedDeltaRuleCompileInfo compileinfo = {48, 196608};
    uint32_t bs = 2, seqLen = 64, t = bs * seqLen;
    uint32_t nk = 4, nv = 8, dk = 64, dv = 64;

    gert::StorageShape queryShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape keyShape = {{t, nk, dk}, {t, nk, dk}};
    gert::StorageShape valueShape = {{t, nv, dv}, {t, nv, dv}};
    gert::StorageShape betaShape = {{t, nv}, {t, nv}};
    gert::StorageShape stateShape = {{bs, nv, dv, dk}, {bs, nv, dv, dk}};
    gert::StorageShape seqLengthsShape = {{bs}, {bs}};

    gert::TilingContextPara tilingContextPara("ChunkGatedDeltaRule",
                                              {{queryShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {keyShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {valueShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {betaShape, ge::DT_BF16, ge::FORMAT_ND},
                                               {stateShape, ge::DT_BF16, ge::FORMAT_FRACTAL_NZ},
                                               {seqLengthsShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{t, nv, dv}, {t, nv, dv}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{bs, nv, dv, dk}, {bs, nv, dv, dk}}, ge::DT_BF16, ge::FORMAT_ND}},
                                              {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)}},
                                              &compileinfo);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}
