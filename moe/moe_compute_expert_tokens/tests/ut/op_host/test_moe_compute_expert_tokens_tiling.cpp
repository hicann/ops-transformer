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
#include "../../../op_host/moe_compute_expert_tokens_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

namespace {
void TouchAllMoeComputeExpertTokensTilingDataFields(optiling::MoeComputeExpertTokensTilingData& tilingData)
{
    tilingData.set_totalCoreNum(64);
    tilingData.set_usedCoreNumBefore(50);
    tilingData.set_usedCoreNumBefore3(33);
    tilingData.set_usedCoreNumAfter(49);
    tilingData.set_ubSize(65536);
    tilingData.set_workLocalNeedSize(24);
    tilingData.set_sortedExpertNum(100);
    tilingData.set_normalCoreHandleNumAfter(2);
    tilingData.set_normalCoreLoopNumAfter(1);
    tilingData.set_normalCoreHandleNumPerLoopAfter(2);
    tilingData.set_normalCoreHandleNumTailLoopAfter(2);
    tilingData.set_tailCoreHandleNumAfter(1);
    tilingData.set_tailCoreLoopNumAfter(1);
    tilingData.set_tailCoreHandleNumPerLoopAfter(1);
    tilingData.set_tailCoreHandleNumTailLoopAfter(1);
    tilingData.set_numOfExpert(98);
    tilingData.set_normalCoreHandleNumBefore(2);
    tilingData.set_normalCoreLoopNumBefore(1);
    tilingData.set_normalCoreHandleNumPerLoopBefore(2);
    tilingData.set_normalCoreHandleNumTailLoopBefore(2);
    tilingData.set_tailCoreHandleNumBefore(1);
    tilingData.set_tailCoreLoopNumBefore(1);
    tilingData.set_tailCoreHandleNumPerLoopBefore(1);
    tilingData.set_tailCoreHandleNumTailLoopBefore(1);
    tilingData.set_handleNumPerCoreBefore(320);
    tilingData.set_handleNumTailCoreBefore(100);
    tilingData.set_loopCountBefore(1);
    tilingData.set_loopCountTailBefore(1);
    tilingData.set_handleNumPerLoopBefore(320);
    tilingData.set_handleNumTailCorePerLoopBefore(100);
    tilingData.set_handleExpertNumLoopCount(1);
    tilingData.set_handleExpertNumMainCorePerLoop(98);
    tilingData.set_handleExpertNumTailCorePerLoop(98);
    tilingData.set_loopCountTailCoreMainLoop(1);
    tilingData.set_handleNumTailCoreMainLoop(10);
    tilingData.set_loopCountTailCoreTailLoop(0);
    tilingData.set_handleNumTailCoreTailLoop(0);
    tilingData.set_userWorkspaceSize(16802304);
    tilingData.set_tilingKey(optiling::COM_SCENE_FLAG_1);

    EXPECT_EQ(tilingData.get_totalCoreNum(), 64);
    EXPECT_EQ(tilingData.get_usedCoreNumBefore(), 50);
    EXPECT_EQ(tilingData.get_usedCoreNumBefore3(), 33);
    EXPECT_EQ(tilingData.get_usedCoreNumAfter(), 49);
    EXPECT_EQ(tilingData.get_tilingKey(), optiling::COM_SCENE_FLAG_1);
}
} // namespace

class MoeComputeExpertTokensTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeComputeExpertTokensTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeComputeExpertTokensTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

TEST_F(MoeComputeExpertTokensTiling, MoeComputeExpertTokens_tiling_data_all_fields)
{
    optiling::MoeComputeExpertTokensTilingData tilingData;
    TouchAllMoeComputeExpertTokensTilingDataFields(tilingData);

    optiling::MoeComputeExpertTokensCompileInfo defaultCompileInfo;
    EXPECT_EQ(defaultCompileInfo.totalCoreNum, 0);
    EXPECT_EQ(defaultCompileInfo.ubSizePlatForm, 0U);

    optiling::MoeComputeExpertTokensCompileInfo compileInfo = {40, 65536};
    EXPECT_EQ(compileInfo.totalCoreNum, 40);
    EXPECT_EQ(compileInfo.ubSizePlatForm, 65536U);

    optiling::MoeComputeExpertTokensCompileInfo copiedCompileInfo = compileInfo;
    EXPECT_EQ(copiedCompileInfo.totalCoreNum, 40);
}

TEST_F(MoeComputeExpertTokensTiling, MoeComputeExpertTokens_tiling_parse_success)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeComputeExpertTokens");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);
    EXPECT_EQ(opImpl->tiling_parse(nullptr), ge::GRAPH_SUCCESS);
}

TEST_F(MoeComputeExpertTokensTiling, MoeComputeExpertTokens_tiling_int32_1)
{
    optiling::MoeComputeExpertTokensCompileInfo compileInfo = {40, 65536}; // 根据tiling头文件中的compileInfo填入对应值
    gert::TilingContextPara tilingContextPara("MoeComputeExpertTokens", // op_name
                                              {
                                                  // input info
                                                  {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{98}, {98}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {{{"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(98)}}},
                                              &compileInfo);
    int64_t expectTilingKey = 1001; // tilngkey
    string expectTilingData = "64 50 1 49 261888 24 100 2 1 2 2 2 1 2 2 98 2 1 2 2 2 1 2 2 320 100 1 1 320 100 1 98 98 "
                              "1 100 0 0 16802304 1001 "; // tilingData
    std::vector<size_t> expectWorkspaces = {16802304};    // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeComputeExpertTokensTiling, MoeComputeExpertTokens_tiling_int32_2)
{
    optiling::MoeComputeExpertTokensCompileInfo compileInfo = {40, 65536}; // 根据tiling头文件中的compileInfo填入对应值
    gert::TilingContextPara tilingContextPara("MoeComputeExpertTokens", // op_name
                                              {
                                                  // input info
                                                  {{{20000}, {20000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {{{"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(98)}}},
                                              &compileInfo);
    int64_t expectTilingKey = 1002; // tilngkey
    string expectTilingData = "64 64 63 49 261888 32 20000 2 1 2 2 2 1 2 2 98 313 1 313 313 281 1 281 281 320 160 1 1 "
                              "320 160 1 98 98 1 160 0 0 16802304 1002 "; // tilingData
    std::vector<size_t> expectWorkspaces = {16802304};                    // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeComputeExpertTokensTiling, MoeComputeExpertTokens_tiling_int32_3)
{
    optiling::MoeComputeExpertTokensCompileInfo compileInfo = {40, 65536}; // 根据tiling头文件中的compileInfo填入对应值
    gert::TilingContextPara tilingContextPara("MoeComputeExpertTokens", // op_name
                                              {
                                                  // input info
                                                  {{{65536}, {65536}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // output info
                                              {
                                                  {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              // attr
                                              {{{"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(98)}}},
                                              &compileInfo);
    int64_t expectTilingKey = 1002; // tilngkey
    string expectTilingData = "64 64 52 49 261888 48 65536 2 1 2 2 2 1 2 2 98 1024 4 320 64 1024 4 320 64 1280 256 4 1 "
                              "320 256 1 98 98 1 256 0 0 16802304 1002 "; // tilingData
    std::vector<size_t> expectWorkspaces = {16802304};                    // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// COM_SCENE_FLAG_3: sortedExpertNum > 12000 && numOfExpert > 256, large tail core path
TEST_F(MoeComputeExpertTokensTiling, MoeComputeExpertTokens_tiling_int32_4)
{
    optiling::MoeComputeExpertTokensCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeComputeExpertTokens",
                                              {
                                                  {{{21000}, {21000}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{300}, {300}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {{{"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(300)}}},
                                              &compileInfo);
    int64_t expectTilingKey = 1003;
    string expectTilingData = "64 64 33 60 261888 32 21000 5 1 5 5 5 1 5 5 300 329 2 320 9 273 1 273 273 640 520 2 1 "
                              "320 520 2 256 44 1 320 1 200 16854016 1003 ";
    std::vector<size_t> expectWorkspaces = {16854016};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// isNetScene: sortedExpertNum == 65536 && numOfExpert == 8
TEST_F(MoeComputeExpertTokensTiling, MoeComputeExpertTokens_tiling_int32_5)
{
    optiling::MoeComputeExpertTokensCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeComputeExpertTokens",
                                              {
                                                  {{{65536}, {65536}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {{{"num_experts", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)}}},
                                              &compileInfo);
    int64_t expectTilingKey = 1001;
    string expectTilingData = "64 64 52 8 261888 48 65536 1 1 1 1 1 1 1 1 8 1024 4 320 64 1024 4 320 64 1280 256 4 1 "
                              "320 256 1 8 8 1 256 0 0 16779264 1001 ";
    std::vector<size_t> expectWorkspaces = {16779264};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
