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
#include <limits>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "../../../op_host/moe_gating_top_k_softmax_v2_tiling.h"

using namespace std;

namespace {
void TouchAllMoeGatingTopKSoftmaxV2TilingDataFields()
{
    optiling::MoeGatingTopKSoftmaxV2TilingData baseData;
    baseData.set_row(4);
    baseData.set_col(16);
    EXPECT_EQ(baseData.get_row(), 4U);
    EXPECT_EQ(baseData.get_col(), 16U);

    optiling::MoeGatingTopKSoftmaxV2PerfRegbaseTilingData regbaseData;
    regbaseData.set_row(4);
    regbaseData.set_col(16);
    regbaseData.set_colBytesAlign(8);
    regbaseData.set_colAlign(16);
    regbaseData.set_k(8);
    regbaseData.set_kAlign(8);
    regbaseData.set_blockNum(1);
    regbaseData.set_blockFormer(4);
    regbaseData.set_blockTail(4);
    regbaseData.set_ubLoopOfFormerBlock(1);
    regbaseData.set_ubLoopOfTailBlock(1);
    regbaseData.set_ubFormer(4);
    regbaseData.set_ubTailOfFormerBlock(4);
    regbaseData.set_ubTailOfTailBlock(4);
    regbaseData.set_topKValuesMask(0);
    regbaseData.set_topKIndicesMask(0);
    regbaseData.set_bufferElemSize(64);
    regbaseData.set_softmaxFlag(1);
    EXPECT_EQ(regbaseData.get_k(), 8U);

    optiling::MoeGatingTopKSoftmaxV2PerfTilingData perfData;
    perfData.set_row(256);
    perfData.set_col(16);
    perfData.set_colBytesAlign(8);
    perfData.set_colAlign(16);
    perfData.set_k(8);
    perfData.set_kAlign(8);
    perfData.set_blockNum(16);
    perfData.set_blockFormer(16);
    perfData.set_blockTail(16);
    perfData.set_ubLoopOfFormerBlock(1);
    perfData.set_ubLoopOfTailBlock(1);
    perfData.set_ubFormer(16);
    perfData.set_ubTailOfFormerBlock(16);
    perfData.set_ubTailOfTailBlock(16);
    perfData.set_topKValuesMask(0);
    perfData.set_topKIndicesMask(0);
    perfData.set_bufferElemSize(256);
    perfData.set_softmaxFlag(1);
    EXPECT_EQ(perfData.get_col(), 16U);
}
} // namespace

class MoeGatingTopKSoftmaxV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeGatingTopKSoftmaxV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeGatingTopKSoftmaxV2Tiling TearDown" << std::endl;
    }
};

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_001)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                  {{{1, 256, 16}, {1, 256, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 256, 16}, {1, 256, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 256, 16}, {1, 256, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                              },
                                              &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 101031;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_002)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                  {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 256, 16}, {1, 256, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                                  {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 101111;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_003)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                  {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 24, 16}, {1, 24, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 24, 16}, {1, 24, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                              },
                                              &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 102011;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 47);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_004)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                  {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 256, 8}, {1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 256, 8}, {1, 256, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                              },
                                              &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 103012;
    string expectTilingData =
        "68719476992 137438953488 34359738376 30064771109 4294967300 30064771073 17179869191 187647121184085 224 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_005)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                  {{{1, 256, 16}, {1, 256, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 256, 8}, {1, 256, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 256, 8}, {1, 256, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                              },
                                              &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 103032;
    string expectTilingData =
        "68719476992 137438953488 34359738376 30064771109 4294967300 30064771073 17179869191 187647121184085 224 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_006)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                  {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0000;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_007)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                  {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 24}, {1, 24}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0000;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_008)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                  {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 24}, {1, 24}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0000;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_009)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                  {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1, 5000}, {1, 1, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0000;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_010)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                  {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 24, 16}, {1, 24, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 24, 16}, {1, 24, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                                  {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 102111;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 47);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_011)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKSoftmaxV2",
        {
            {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 24, 16}, {1, 24, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 24, 16}, {1, 24, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 102011;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 47);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_012)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKSoftmaxV2",
        {
            {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 256, 8}, {1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 256, 8}, {1, 256, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 103012;
    string expectTilingData = "68719476992 137438953488 34359738376 30064771109 4294967300 30064771073 17179869191 "
                              "187647121184085 4294967520 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_data_all_fields)
{
    TouchAllMoeGatingTopKSoftmaxV2TilingDataFields();
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_parse_stub)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeGatingTopKSoftmaxV2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);
    EXPECT_EQ(opImpl->tiling_parse(nullptr), ge::GRAPH_SUCCESS);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_regbase_ascend950_fp16)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKSoftmaxV2",
        {
            {{{1, 4, 16}, {1, 4, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 8}, {1, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 8}, {1, 4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 4, 16}, {1, 4, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend950", 48ULL, 196608ULL);
    uint64_t expectTilingKey = 103020;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_regbase_ascend950_fp16_renorm)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKSoftmaxV2",
        {
            {{{1, 4, 16}, {1, 4, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 8}, {1, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 4, 8}, {1, 4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 4, 16}, {1, 4, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend950", 48ULL, 196608ULL);
    uint64_t expectTilingKey = 103120;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_regbase_ascend950_bf16)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKSoftmaxV2",
        {
            {{{4, 16}, {4, 16}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{4, 8}, {4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 16}, {4, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 48ULL, 196608ULL);
    uint64_t expectTilingKey = 103030;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_regbase_ascend950_col_small)
{
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara(
        "MoeGatingTopKSoftmaxV2",
        {
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo, "Ascend950", 48ULL, 196608ULL);
    uint64_t expectTilingKey = 103010;
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}
