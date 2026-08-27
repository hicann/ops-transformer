/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_fused_topk_tiling.cpp
 * \brief
 */
#include <iostream>
#include <map>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"

#include "../../../op_host/moe_fused_topk_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_infos_def.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;

namespace {
void SetupParsePlatformInfo(fe::PlatFormInfos *platformInfo)
{
    platformInfo->Init();
    map<string, string> socInfos = {
        {"ai_core_cnt", "64"},
        {"l2_size", "33554432"},
        {"core_type_list", "AICore"},
    };
    map<string, string> aicoreSpec = {
        {"ub_size", "65536"},    {"l0_a_size", "65536"}, {"l0_b_size", "65536"},
        {"l0_c_size", "131072"}, {"l1_size", "524288"},  {"bt_size", "0"},
    };
    map<string, string> intrinsics = {
        {"Intrinsic_data_move_l12ub", "float16"},
        {"Intrinsic_data_move_l0c2ub", "float16"},
    };
    map<string, string> versions = {
        {"NpuArch", "2201"},
        {"Short_SoC_version", "Ascend910B"},
    };
    platformInfo->SetPlatformRes("SoCInfo", socInfos);
    platformInfo->SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo->SetCoreNumByCoreType("AICore");
    platformInfo->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    platformInfo->SetPlatformRes("version", versions);
}

static vector<gert::TilingContextPara::OpAttr> DefaultTopkAttrs(int64_t groupNum = 8, bool enableExpertMapping = false)
{
    return {
        {"group_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupNum)},
        {"group_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        {"top_n", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        {"activate_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"is_norm", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"enable_expert_mapping", Ops::Transformer::AnyValue::CreateFrom<bool>(enableExpertMapping)},
    };
}
} // namespace
class MoeFusedTopkTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFusedTopkTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFusedTopkTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData *tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t *>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

TEST_F(MoeFusedTopkTiling, test_tiling_bf16)
{
    optiling::MoeFusedTopkCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 192 * 1024;
    gert::TilingContextPara tilingContextPara(
        "MoeFusedTopk",
        {
            {{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{128, 8}, {128, 8}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 8}, {128, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"group_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"group_topk", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"top_n", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"top_k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"activate_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"is_norm", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
         {"enable_expert_mapping", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "549755814016 34359738496 8589934596 8 4575657221408423937 171798691856 12884907452 8 "
                              "4398046511104 1024 512 1099511627936 4294967360 34359738376 274877906960 34359738372 "
                              "17179869186 8589934598 4294967360 34359738496 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024 + 40 * 512};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFusedTopkTiling, test_tiling_float_small_batch)
{
    optiling::MoeFusedTopkCompileInfo compileInfo = {40, 192 * 1024};
    gert::TilingContextPara tilingContextPara("MoeFusedTopk",
                                              {
                                                  {{{8, 128}, {8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              DefaultTopkAttrs(), &compileInfo);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024 + 8 * 512};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", expectWorkspaces);
}

TEST_F(MoeFusedTopkTiling, test_tiling_expert_mapping)
{
    optiling::MoeFusedTopkCompileInfo compileInfo = {40, 192 * 1024};
    gert::TilingContextPara tilingContextPara("MoeFusedTopk",
                                              {
                                                  {{{16, 128}, {16, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{128, 64}, {128, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 8}, {16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              DefaultTopkAttrs(8, true), &compileInfo);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024 + 16 * 512};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1, "", expectWorkspaces);
}

TEST_F(MoeFusedTopkTiling, test_tiling_group_num_zero)
{
    optiling::MoeFusedTopkCompileInfo compileInfo = {40, 192 * 1024};
    gert::TilingContextPara tilingContextPara("MoeFusedTopk",
                                              {
                                                  {{{32, 128}, {32, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              DefaultTopkAttrs(0, false), &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", {});
}

TEST_F(MoeFusedTopkTiling, test_tiling_ub_overflow)
{
    optiling::MoeFusedTopkCompileInfo compileInfo = {40, 16 * 1024};
    gert::TilingContextPara tilingContextPara("MoeFusedTopk",
                                              {
                                                  {{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128}, {128}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{128, 8}, {128, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 8}, {128, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              DefaultTopkAttrs(), &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(MoeFusedTopkTiling, test_tiling_float16_large_batch)
{
    optiling::MoeFusedTopkCompileInfo compileInfo = {8, 192 * 1024};
    gert::TilingContextPara tilingContextPara("MoeFusedTopk",
                                              {
                                                  {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{128}, {128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{64, 8}, {64, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{64, 8}, {64, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              DefaultTopkAttrs(), &compileInfo);
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024 + 8 * 512};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", expectWorkspaces);
}

TEST_F(MoeFusedTopkTiling, test_tiling_ascend950_fp16_non_16_aligned_expert_num)
{
    // 136 exercises the FP16 staging allocation: it is 32-byte aligned as FP32
    // but not as FP16.  The Host UB budget must match Kernel::InitBuffer exactly.
    optiling::MoeFusedTopkCompileInfo compileInfo = {40, 256 * 1024};
    gert::TilingContextPara tilingContextPara("MoeFusedTopk",
                                              {
                                                  {{{41, 136}, {41, 136}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{136}, {136}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{41, 8}, {41, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{41, 8}, {41, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              DefaultTopkAttrs(), &compileInfo, "Ascend950");
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024 + 40 * 136 * sizeof(float)};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", expectWorkspaces);
}

TEST_F(MoeFusedTopkTiling, MoeFusedTopk_tiling_data_field_accessors)
{
    optiling::MoeFusedTopkTilingData tilingData;
    tilingData.set_firstDimSize(16);
    tilingData.set_secondDimSize(128);
    tilingData.set_addNumDimSize(128);
    tilingData.set_groupNum(8);
    tilingData.set_groupTopk(4);
    tilingData.set_topN(2);
    tilingData.set_topK(8);
    tilingData.set_activateType(0);
    tilingData.set_isNorm(1);
    tilingData.set_scale(1.0f);
    tilingData.set_groupEles(16);
    tilingData.set_blockNum(16);
    tilingData.set_ubFactorElement(512);
    tilingData.set_batchPerCore(1);
    tilingData.set_tailBatch(0);
    tilingData.set_expertNum(128);
    tilingData.set_tableDim(64);
    tilingData.set_topkMaxValue(4096);
    tilingData.set_topkMinValue(2048);
    tilingData.set_workspacePerCore(512);

    EXPECT_EQ(tilingData.get_firstDimSize(), 16U);
    EXPECT_EQ(tilingData.get_secondDimSize(), 128U);
    EXPECT_EQ(tilingData.get_addNumDimSize(), 128U);
    EXPECT_EQ(tilingData.get_groupNum(), 8U);
    EXPECT_EQ(tilingData.get_groupTopk(), 4U);
    EXPECT_EQ(tilingData.get_topN(), 2U);
    EXPECT_EQ(tilingData.get_topK(), 8U);
    EXPECT_EQ(tilingData.get_activateType(), 0U);
    EXPECT_EQ(tilingData.get_isNorm(), 1U);
    EXPECT_FLOAT_EQ(tilingData.get_scale(), 1.0f);
    EXPECT_EQ(tilingData.get_groupEles(), 16U);
    EXPECT_EQ(tilingData.get_blockNum(), 16U);
    EXPECT_EQ(tilingData.get_ubFactorElement(), 512U);
    EXPECT_EQ(tilingData.get_batchPerCore(), 1U);
    EXPECT_EQ(tilingData.get_tailBatch(), 0U);
    EXPECT_EQ(tilingData.get_expertNum(), 128U);
    EXPECT_EQ(tilingData.get_tableDim(), 64U);
    EXPECT_EQ(tilingData.get_topkMaxValue(), 4096U);
    EXPECT_EQ(tilingData.get_topkMinValue(), 2048U);
    EXPECT_EQ(tilingData.get_workspacePerCore(), 512U);

    const size_t dataSize = tilingData.GetDataSize();
    ASSERT_GT(dataSize, 0U);
    std::vector<uint8_t> buf(dataSize);
    tilingData.SaveToBuffer(buf.data(), buf.size());

    optiling::MoeFusedTopkCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 192 * 1024;
    EXPECT_EQ(compileInfo.coreNum, 40U);
    EXPECT_EQ(compileInfo.ubSize, static_cast<uint64_t>(192 * 1024));
}

TEST_F(MoeFusedTopkTiling, TilingParse_Success)
{
    optiling::MoeFusedTopkCompileInfo compileInfo = {0, 0};
    fe::PlatFormInfos platformInfo;
    SetupParsePlatformInfo(&platformInfo);

    const char *compileInfoString =
        R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, )"
        R"("Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, )"
        R"("Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 65536, "L2_SIZE": 33554432, )"
        R"("L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, )"
        R"("CORE_NUM": 64, "socVersion": "Ascend910B"}})";

    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.OpType("MoeFusedTopk")
                      .OpName("MoeFusedTopk")
                      .IONum(4, 2)
                      .CompiledJson(compileInfoString)
                      .CompiledInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                      .Build();
    auto parseContext = holder.GetContext();
    ASSERT_NE(parseContext, nullptr);

    SetupParsePlatformInfo(parseContext->GetPlatformInfo());

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MoeFusedTopk");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext *>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_GT(compileInfo.coreNum, 0);
    EXPECT_GT(compileInfo.ubSize, 0);
}
