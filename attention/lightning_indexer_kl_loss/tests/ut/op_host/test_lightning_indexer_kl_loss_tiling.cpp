/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "../../../../op_kernel/lightning_indexer_kl_loss_tiling_data.h"
#include "../../../../op_kernel/lightning_indexer_kl_loss_tiling_key.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
using OpAttr = gert::TilingContextPara::OpAttr;

constexpr const char *kOpName = "LightningIndexerKLLoss";
constexpr uint32_t kCoreNum = 48U;
constexpr uint64_t kUbSize = 196608ULL;
constexpr size_t kTilingDataSize = 8192;

static const uint64_t kTilingKeyFp16 = GET_TPL_TILING_KEY(LightningIndexerKLLoss, 0, 0);
static const uint64_t kTilingKeyFp32 = GET_TPL_TILING_KEY(LightningIndexerKLLoss, 0, 1);
static const uint64_t kTilingKeyBf16 = GET_TPL_TILING_KEY(LightningIndexerKLLoss, 0, 2);
static const uint64_t kTilingKeyDetFp16 = GET_TPL_TILING_KEY(LightningIndexerKLLoss, 1, 0);
static const uint64_t kTilingKeyDetFp32 = GET_TPL_TILING_KEY(LightningIndexerKLLoss, 1, 1);
static const uint64_t kTilingKeyDetBf16 = GET_TPL_TILING_KEY(LightningIndexerKLLoss, 1, 2);

optiling::LightningIndexerKLLossCompileInfo MakeCompileInfo() { return {}; }

std::vector<OpAttr> MakeAttrs(float eps = 1e-9f, bool deterministic = false)
{
    return {{"eps", Ops::Transformer::AnyValue::CreateFrom<float>(eps)},
            {"deterministic", Ops::Transformer::AnyValue::CreateFrom<bool>(deterministic)}};
}

gert::TilingContextPara MakePara(optiling::LightningIndexerKLLossCompileInfo &compileInfo,
                                 ge::DataType dataType = ge::DT_FLOAT, std::initializer_list<int64_t> shape = {8, 22},
                                 float eps = 1e-9f, bool deterministic = false)
{
    return gert::TilingContextPara(
        kOpName, {{{shape, shape}, dataType, ge::FORMAT_ND}, {{shape, shape}, dataType, ge::FORMAT_ND}},
        {{{{1}, {1}}, dataType, ge::FORMAT_ND}}, MakeAttrs(eps, deterministic), std::vector<uint32_t>{1, 1},
        std::vector<uint32_t>{1}, &compileInfo, "Ascend910B", kCoreNum, kUbSize, kTilingDataSize);
}
} // namespace

class LightningIndexerKLLossTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LightningIndexerKLLossTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "LightningIndexerKLLossTiling TearDown" << std::endl; }

    void VerifyTilingData(const LightningIndexerKLLossTilingData &tiling, int64_t expectedTotalLength,
                          int64_t expectedK, int64_t expectedKAligned)
    {
        EXPECT_EQ(tiling.totalLength, expectedTotalLength);
        EXPECT_EQ(tiling.K, expectedK);
        EXPECT_EQ(tiling.KAligned, expectedKAligned);
        EXPECT_GT(tiling.tileLength, 0);
        EXPECT_GT(tiling.coreNum, 0);
    }
};

TEST_F(LightningIndexerKLLossTiling, tiling_2d_fp32_success)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara(compileInfo, ge::DT_FLOAT, {8, 22});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyFp32);
}

TEST_F(LightningIndexerKLLossTiling, tiling_2d_fp16_success)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara(compileInfo, ge::DT_FLOAT16, {8, 22});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyFp16);
}

TEST_F(LightningIndexerKLLossTiling, tiling_2d_bf16_success)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara(compileInfo, ge::DT_BF16, {8, 22});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyBf16);
}

TEST_F(LightningIndexerKLLossTiling, tiling_3d_fp32_success)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara(compileInfo, ge::DT_FLOAT, {4, 10, 333});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyFp32);
}

TEST_F(LightningIndexerKLLossTiling, tiling_3d_deterministic_fp32_success)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara(compileInfo, ge::DT_FLOAT, {4, 10, 333}, 1e-9f, true);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyDetFp32);
}

TEST_F(LightningIndexerKLLossTiling, tiling_3d_deterministic_bf16_success)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara(compileInfo, ge::DT_BF16, {4, 10, 333}, 1e-9f, true);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyDetBf16);
}

TEST_F(LightningIndexerKLLossTiling, tiling_3d_deterministic_fp16_success)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara(compileInfo, ge::DT_FLOAT16, {4, 10, 333}, 1e-9f, true);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyDetFp16);
}

TEST_F(LightningIndexerKLLossTiling, tiling_custom_eps_success)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara(compileInfo, ge::DT_FLOAT, {8000, 65}, 1e-6f);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, kTilingKeyFp32);
}

// --------------- CheckInputDim 失败测试 ---------------

TEST_F(LightningIndexerKLLossTiling, tiling_shape_mismatch_failed)
{
    std::vector<int64_t> shapeA = {4, 10, 333};
    std::vector<int64_t> shapeB = {4, 10, 111};
    auto compileInfo = MakeCompileInfo();

    auto para = gert::TilingContextPara(
        kOpName, {{{shapeA, shapeA}, ge::DT_FLOAT, ge::FORMAT_ND}, {{shapeB, shapeB}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeAttrs(), std::vector<uint32_t>{1, 1}, std::vector<uint32_t>{1},
        &compileInfo, "Ascend910B", kCoreNum, kUbSize, kTilingDataSize);

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(LightningIndexerKLLossTiling, tiling_1d_shape_failed)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara(compileInfo, ge::DT_FLOAT, {256});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(LightningIndexerKLLossTiling, tiling_4d_shape_failed)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara(compileInfo, ge::DT_FLOAT, {2, 4, 10, 128});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(LightningIndexerKLLossTiling, tiling_last_dim_ge_8192_failed)
{
    auto compileInfo = MakeCompileInfo();
    auto para = MakePara(compileInfo, ge::DT_FLOAT, {4, 10, 8192});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
