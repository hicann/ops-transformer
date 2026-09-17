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
 * \file test_fused_gdn_gating_tiling.cpp
 * \brief Tiling unit tests for FusedGdnGating.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "../../../op_host/fused_gdn_gating_tiling.h"
#include "../../../op_kernel/fused_gdn_gating_tiling_data.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace optiling;

class FusedGdnGatingTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedGdnGatingTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedGdnGatingTilingTest TearDown" << std::endl;
    }
};

TEST_F(FusedGdnGatingTilingTest, Bf16FloatTilingKey)
{
    optiling::FusedGdnGatingCompileInfo compileinfo = {48, 196608};

    int batch = 4;
    int numHeads = 8;

    gert::StorageShape aLogShape = {{numHeads}, {numHeads}};
    gert::StorageShape aShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape bShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape dtBiasShape = {{numHeads}, {numHeads}};
    gert::StorageShape gShape = {{1, batch, numHeads}, {1, batch, numHeads}};
    gert::StorageShape betaOutputShape = {{1, batch, numHeads}, {1, batch, numHeads}};

    gert::TilingContextPara tilingContextPara("FusedGdnGating",
                                              {
                                                  {aLogShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {aShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {bShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {dtBiasShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {betaOutputShape, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"threshold", Ops::Transformer::AnyValue::CreateFrom<float>(20.0f)},
                                              },
                                              &compileinfo);

    int64_t expectTilingKey = 1UL;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    auto tilingData = reinterpret_cast<FusedGdnGating::FusedGdnGatingTilingData *>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->numHeads, static_cast<uint32_t>(numHeads));
    EXPECT_EQ(tilingData->numBatches, static_cast<uint32_t>(batch));
    EXPECT_FLOAT_EQ(tilingData->beta, 1.0f);
    EXPECT_FLOAT_EQ(tilingData->threshold, 20.0f);
}

TEST_F(FusedGdnGatingTilingTest, Fp16FloatTilingKey)
{
    optiling::FusedGdnGatingCompileInfo compileinfo = {48, 196608};

    int batch = 4;
    int numHeads = 8;

    gert::StorageShape aLogShape = {{numHeads}, {numHeads}};
    gert::StorageShape aShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape bShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape dtBiasShape = {{numHeads}, {numHeads}};
    gert::StorageShape gShape = {{1, batch, numHeads}, {1, batch, numHeads}};
    gert::StorageShape betaOutputShape = {{1, batch, numHeads}, {1, batch, numHeads}};

    gert::TilingContextPara tilingContextPara("FusedGdnGating",
                                              {
                                                  {aLogShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {aShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {bShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {dtBiasShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {betaOutputShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"threshold", Ops::Transformer::AnyValue::CreateFrom<float>(20.0f)},
                                              },
                                              &compileinfo);

    int64_t expectTilingKey = 2UL;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    auto tilingData = reinterpret_cast<FusedGdnGating::FusedGdnGatingTilingData *>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->numHeads, static_cast<uint32_t>(numHeads));
}

TEST_F(FusedGdnGatingTilingTest, Bf16Bf16TilingKey)
{
    optiling::FusedGdnGatingCompileInfo compileinfo = {48, 196608};

    int batch = 4;
    int numHeads = 8;

    gert::StorageShape aLogShape = {{numHeads}, {numHeads}};
    gert::StorageShape aShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape bShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape dtBiasShape = {{numHeads}, {numHeads}};
    gert::StorageShape gShape = {{1, batch, numHeads}, {1, batch, numHeads}};
    gert::StorageShape betaOutputShape = {{1, batch, numHeads}, {1, batch, numHeads}};

    gert::TilingContextPara tilingContextPara("FusedGdnGating",
                                              {
                                                  {aLogShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {aShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {bShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {dtBiasShape, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {betaOutputShape, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"threshold", Ops::Transformer::AnyValue::CreateFrom<float>(20.0f)},
                                              },
                                              &compileinfo);

    int64_t expectTilingKey = 3UL;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(FusedGdnGatingTilingTest, Fp16Fp16TilingKey)
{
    optiling::FusedGdnGatingCompileInfo compileinfo = {48, 196608};

    int batch = 4;
    int numHeads = 8;

    gert::StorageShape aLogShape = {{numHeads}, {numHeads}};
    gert::StorageShape aShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape bShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape dtBiasShape = {{numHeads}, {numHeads}};
    gert::StorageShape gShape = {{1, batch, numHeads}, {1, batch, numHeads}};
    gert::StorageShape betaOutputShape = {{1, batch, numHeads}, {1, batch, numHeads}};

    gert::TilingContextPara tilingContextPara("FusedGdnGating",
                                              {
                                                  {aLogShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {aShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {bShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {dtBiasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {betaOutputShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"threshold", Ops::Transformer::AnyValue::CreateFrom<float>(20.0f)},
                                              },
                                              &compileinfo);

    int64_t expectTilingKey = 6UL;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
}

TEST_F(FusedGdnGatingTilingTest, InvalidDtypeMismatch)
{
    optiling::FusedGdnGatingCompileInfo compileinfo = {48, 196608};

    int batch = 4;
    int numHeads = 8;

    gert::StorageShape aLogShape = {{numHeads}, {numHeads}};
    gert::StorageShape aShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape bShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape dtBiasShape = {{numHeads}, {numHeads}};
    gert::StorageShape gShape = {{1, batch, numHeads}, {1, batch, numHeads}};
    gert::StorageShape betaOutputShape = {{1, batch, numHeads}, {1, batch, numHeads}};

    gert::TilingContextPara tilingContextPara("FusedGdnGating",
                                              {
                                                  {aLogShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {aShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {bShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {dtBiasShape, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {betaOutputShape, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"threshold", Ops::Transformer::AnyValue::CreateFrom<float>(20.0f)},
                                              },
                                              &compileinfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedGdnGatingTilingTest, InvalidDim)
{
    optiling::FusedGdnGatingCompileInfo compileinfo = {48, 196608};

    int numHeads = 8;

    gert::StorageShape aLogShape = {{numHeads}, {numHeads}};
    gert::StorageShape aShape = {{numHeads}, {numHeads}};
    gert::StorageShape bShape = {{numHeads}, {numHeads}};
    gert::StorageShape dtBiasShape = {{numHeads}, {numHeads}};
    gert::StorageShape gShape = {{1, numHeads}, {1, numHeads}};
    gert::StorageShape betaOutputShape = {{1, numHeads}, {1, numHeads}};

    gert::TilingContextPara tilingContextPara("FusedGdnGating",
                                              {
                                                  {aLogShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {aShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {bShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {dtBiasShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {betaOutputShape, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"threshold", Ops::Transformer::AnyValue::CreateFrom<float>(20.0f)},
                                              },
                                              &compileinfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedGdnGatingTilingTest, Ascend310PTilingKey)
{
    optiling::FusedGdnGatingCompileInfo compileinfo = {8, 196608};

    int batch = 4;
    int numHeads = 8;

    gert::StorageShape aLogShape = {{numHeads}, {numHeads}};
    gert::StorageShape aShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape bShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape dtBiasShape = {{numHeads}, {numHeads}};
    gert::StorageShape gShape = {{1, batch, numHeads}, {1, batch, numHeads}};
    gert::StorageShape betaOutputShape = {{1, batch, numHeads}, {1, batch, numHeads}};

    gert::TilingContextPara tilingContextPara("FusedGdnGating",
                                              {
                                                  {aLogShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {aShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {bShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {dtBiasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {betaOutputShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"threshold", Ops::Transformer::AnyValue::CreateFrom<float>(20.0f)},
                                              },
                                              &compileinfo, "Ascend310P", 8, 196608);

    int64_t expectTilingKey = 200000UL;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    auto tilingData = reinterpret_cast<FusedGdnGating::FusedGdnGatingTilingData *>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->numHeads, static_cast<uint32_t>(numHeads));
    EXPECT_EQ(tilingData->beta, 1.0f);
    EXPECT_EQ(tilingData->threshold, 20.0f);
    EXPECT_GT(tilingData->usedCoreNum, 0u);
    EXPECT_GT(tilingData->alignedLength, 0u);
    EXPECT_GT(tilingData->tailLength, 0u);
    EXPECT_GT(tilingData->tileRows, 0u);
    EXPECT_FLOAT_EQ(tilingData->inv_beta, 1.0f);
}

TEST_F(FusedGdnGatingTilingTest, Ascend310PLargeBatch)
{
    optiling::FusedGdnGatingCompileInfo compileinfo = {8, 196608};

    int batch = 128;
    int numHeads = 64;

    gert::StorageShape aLogShape = {{numHeads}, {numHeads}};
    gert::StorageShape aShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape bShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape dtBiasShape = {{numHeads}, {numHeads}};
    gert::StorageShape gShape = {{1, batch, numHeads}, {1, batch, numHeads}};
    gert::StorageShape betaOutputShape = {{1, batch, numHeads}, {1, batch, numHeads}};

    gert::TilingContextPara tilingContextPara("FusedGdnGating",
                                              {
                                                  {aLogShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {aShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {bShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {dtBiasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {betaOutputShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                                  {"threshold", Ops::Transformer::AnyValue::CreateFrom<float>(20.0f)},
                                              },
                                              &compileinfo, "Ascend310P", 8, 196608);

    int64_t expectTilingKey = 200000UL;

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    auto tilingData = reinterpret_cast<FusedGdnGating::FusedGdnGatingTilingData *>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->numHeads, static_cast<uint32_t>(numHeads));
    EXPECT_GT(tilingData->usedCoreNum, 0u);
    EXPECT_GT(tilingData->tileRows, 0u);
}

TEST_F(FusedGdnGatingTilingTest, Ascend310PNonDefaultAttrs)
{
    optiling::FusedGdnGatingCompileInfo compileinfo = {8, 196608};

    int batch = 4;
    int numHeads = 8;

    gert::StorageShape aLogShape = {{numHeads}, {numHeads}};
    gert::StorageShape aShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape bShape = {{batch, numHeads}, {batch, numHeads}};
    gert::StorageShape dtBiasShape = {{numHeads}, {numHeads}};
    gert::StorageShape gShape = {{1, batch, numHeads}, {1, batch, numHeads}};
    gert::StorageShape betaOutputShape = {{1, batch, numHeads}, {1, batch, numHeads}};

    gert::TilingContextPara tilingContextPara("FusedGdnGating",
                                              {
                                                  {aLogShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {aShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {bShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {dtBiasShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {gShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {betaOutputShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"beta", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)},
                                                  {"threshold", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                                              },
                                              &compileinfo, "Ascend310P", 8, 196608);

    TilingInfo tilingInfo;
    ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingInfo.tilingKey, 200000UL);
    auto tilingData = reinterpret_cast<FusedGdnGating::FusedGdnGatingTilingData *>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    // 属性必须原样写入 tiling data（此前全链路仅测过默认值 1.0/20.0）
    EXPECT_FLOAT_EQ(tilingData->beta, 0.5f);
    EXPECT_FLOAT_EQ(tilingData->threshold, 1.0f);
    EXPECT_FLOAT_EQ(tilingData->inv_beta, 2.0f);
    EXPECT_EQ(tilingData->numHeads, static_cast<uint32_t>(numHeads));
}
