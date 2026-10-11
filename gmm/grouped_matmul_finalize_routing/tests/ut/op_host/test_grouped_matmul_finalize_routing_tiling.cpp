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
 * \file test_grouped_matmul_finalize_routing_tiling.cpp
 * \brief CSV-driven unit tests for grouped_matmul_finalize_routing tiling.
 */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../../../op_host/grouped_matmul_finalize_routing_base_tiling.h"
#include "../../../op_host/grouped_matmul_finalize_routing_tiling.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "op_tiling_parse_context_builder.h"
#include "platform/platform_infos_def.h"

#include "gmm_csv_ge_parse_utils.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;
using namespace ge;
using namespace optiling;

namespace {
using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr size_t kCsvColumnCount = 52;

uint64_t ParseU64(const string& value)
{
    const string trimmed = Trim(value);
    return trimmed.empty() ? 0UL : static_cast<uint64_t>(stoull(trimmed));
}

float ParseFloat(const string& value)
{
    const string trimmed = Trim(value);
    return trimmed.empty() ? 0.0F : stof(trimmed);
}

vector<int64_t> ParseDims(const string& value)
{
    return ops::ut::ParseDims(value);
}

ge::DataType ParseDtype(const string& dtype)
{
    return ops::ut::ParseGeDtype(dtype);
}

ge::Format ParseFormat(const string& format)
{
    return ops::ut::ParseGeFormat(format);
}

ge::graphStatus ParseGraphStatus(const string& status)
{
    return Trim(status) == "SUCCESS" ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

platform_ascendc::SocVersion ParseSocVersion(const string& socVersion)
{
    const string trimmed = Trim(socVersion);
    if (trimmed == "Ascend910B") {
        return platform_ascendc::SocVersion::ASCEND910B;
    }
    if (trimmed == "Ascend950") {
        return platform_ascendc::SocVersion::ASCEND950;
    }
    return static_cast<platform_ascendc::SocVersion>(0);
}

NpuArch ParseNpuArch(const string& npuArch)
{
    return Trim(npuArch) == "DAV_3510" ? NpuArch::DAV_3510 : static_cast<NpuArch>(0);
}

struct GroupedMatmulFinalizeRoutingTilingCase {
    void Run() const
    {
        if (Trim(caseType) == "tiling_parse") {
            RunTilingParse();
            return;
        }
        if (Trim(caseType) == "tiling_data") {
            RunTilingData();
            return;
        }
        optiling::GroupedMatmulFinalizeRoutingCompileInfo compileInfo;
        compileInfo.aicNum = ParseU64(aicNum);
        compileInfo.aivNum = ParseU64(aivNum);
        compileInfo.ubSize = ParseU64(ubSize);
        compileInfo.l1Size = ParseU64(l1Size);
        compileInfo.l2Size = ParseU64(l2Size);
        compileInfo.l0CSize = ParseU64(l0CSize);
        compileInfo.l0ASize = ParseU64(l0ASize);
        compileInfo.l0BSize = ParseU64(l0BSize);
        compileInfo.btSize = ParseU64(btSize);
        compileInfo.cubeFreq = ParseFloat(cubeFreq);
        compileInfo.socVersion = ParseSocVersion(socVersion);
        compileInfo.supportL0c2out = ParseBool(supportL0c2out);
        compileInfo.supportL12BtBf16 = ParseBool(supportL12BtBf16);
        compileInfo.npuArch = ParseNpuArch(npuArch);

        vector<int64_t> tuningConfigVec = ops::ut::ParseDims(tuningConfig, {});
        if (tuningConfigVec.empty()) {
            tuningConfigVec = {0};
        }

        vector<gert::TilingContextPara::TensorDescription> inputDescs = {
            {ops::ut::MakeGertStorageShape(ParseDims(xShape)), ParseDtype(xDtype), ge::FORMAT_ND},
            {ParseFormat(wFormat) == ge::FORMAT_FRACTAL_NZ && ParseDims(wShape).size() == 3 ?
                 ops::ut::MakeGertStorageShape(ParseDims(wShape), {ParseDims(wShape)[0], ParseDims(wShape)[1] / 32,
                                                                   ParseDims(wShape)[2] / 16, 32, 16}) :
                 ops::ut::MakeGertStorageShape(ParseDims(wShape)),
             ParseDtype(wDtype), ParseFormat(wFormat)},
            {ops::ut::MakeGertStorageShape(ParseDims(scaleShape)), ParseDtype(scaleDtype), ge::FORMAT_ND},
            {ops::ut::MakeGertStorageShape(ParseDims(biasShape)), ParseDtype(biasDtype), ge::FORMAT_ND},
            {ops::ut::MakeGertStorageShape(ParseDims(pertokenScaleShape)), ParseDtype(pertokenScaleDtype),
             ge::FORMAT_ND},
            {ops::ut::MakeGertStorageShape(ParseDims(groupListShape)), ParseDtype(groupListDtype), ge::FORMAT_ND},
            {ops::ut::MakeGertStorageShape(ParseDims(sharedInputShape)), ParseDtype(sharedInputDtype), ge::FORMAT_ND},
            {ops::ut::MakeGertStorageShape(ParseDims(logitShape)), ParseDtype(logitDtype), ge::FORMAT_ND},
            {ops::ut::MakeGertStorageShape(ParseDims(rowindexShape)), ParseDtype(rowindexDtype), ge::FORMAT_ND},
        };
        if (Trim(offsetShape) != "NONE") {
            inputDescs.emplace_back(ops::ut::MakeGertStorageShape(ParseDims(offsetShape)), ParseDtype(offsetDtype),
                                    ge::FORMAT_ND);
        }

        string socVersionStr =
            ParseSocVersion(socVersion) == platform_ascendc::SocVersion::ASCEND950 ? "Ascend950" : "Ascend910B";
        gert::TilingContextPara tilingContextPara(
            "GroupedMatmulFinalizeRouting", inputDescs,
            {
                {ops::ut::MakeGertStorageShape(ParseDims(yShape)), ParseDtype(yDtype), ge::FORMAT_ND},
            },
            {
                {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(stoll(dtypeAttr)))},
                {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(ParseFloat(sharedInputWeight))},
                {"shared_input_offset",
                 Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(stoll(sharedInputOffset)))},
                {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(ParseBool(transposeX))},
                {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(ParseBool(transposeW))},
                {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(stoll(outputBs)))},
                {"group_list_type",
                 Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(stoll(groupListType)))},
                {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tuningConfigVec)},
            },
            &compileInfo, socVersionStr);
        tilingContextPara.deterministicInfo_ =
            deterministicInfo.empty() ? 0 : static_cast<int32_t>(stoll(deterministicInfo));

        const ge::graphStatus expectStatus = ParseGraphStatus(expectResult);
        if (expectStatus == ge::GRAPH_SUCCESS) {
            TilingInfo tilingInfo;
            bool tilingResult = ExecuteTiling(tilingContextPara, tilingInfo);
            ASSERT_TRUE(tilingResult) << "prefix=" << prefix << ", case=" << caseName;
            EXPECT_EQ(static_cast<uint64_t>(tilingInfo.tilingKey), ParseU64(expectTilingKey))
                << "prefix=" << prefix << ", case=" << caseName;
            return;
        }

        ExecuteTestCase(tilingContextPara, expectStatus);
    }

    void RunTilingParse() const
    {
        optiling::GroupedMatmulFinalizeRoutingCompileInfo compileInfo = {};
        fe::PlatFormInfos platformInfo;
        platformInfo.Init();
        std::map<std::string, std::string> socInfos = {
            {"ai_core_cnt", "24"}, {"l2_size", "33554432"}, {"core_type_list", "AICore"}};
        std::map<std::string, std::string> aicoreSpec = {{"ub_size", "196608"},
                                                         {"l1_size", "524288"},
                                                         {"l0_a_size", "65536"},
                                                         {"l0_b_size", "65536"},
                                                         {"l0_c_size", "131072"}};
        std::map<std::string, std::string> intrinsics = {{"Intrinsic_data_move_l12ub", "float16"},
                                                         {"Intrinsic_data_move_l0c2ub", "float16"},
                                                         {"Intrinsic_fix_pipe_l0c2out", "float16"},
                                                         {"Intrinsic_data_move_out2l1_nd2nz", "float16"},
                                                         {"Intrinsic_data_move_l12bt", "bf16"}};
        platformInfo.SetPlatformRes("SoCInfo", socInfos);
        platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
        platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

        const char* compileJsonStr = "{}";
        gert::OpTilingParseContextBuilder builder;
        builder.OpType("GroupedMatmulFinalizeRouting")
            .OpName("GroupedMatmulFinalizeRouting")
            .IONum(2, 1)
            .CompiledJson(compileJsonStr)
            .CompiledInfo(reinterpret_cast<void*>(&compileInfo))
            .PlatformInfo(reinterpret_cast<void*>(&platformInfo));
        auto holder = builder.Build();
        auto parseContext = holder.GetContext();

        auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
        ASSERT_NE(spaceRegistry, nullptr);
        auto opImpl = spaceRegistry->GetOpImpl("GroupedMatmulFinalizeRouting");
        ASSERT_NE(opImpl, nullptr);
        ASSERT_NE(opImpl->tiling_parse, nullptr);

        auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext*>(parseContext));
        EXPECT_EQ(ret, ParseGraphStatus(expectResult));
    }

    void RunTilingData() const
    {
        optiling::GroupMatmulFRTilingData tilingData;
        tilingData.set_coreNum(24);
        EXPECT_EQ(tilingData.get_coreNum(), 24U);
        tilingData.set_groupNum(16);
        EXPECT_EQ(tilingData.get_groupNum(), 16U);
        tilingData.set_totalInGroup(1024);
        EXPECT_EQ(tilingData.get_totalInGroup(), 1024U);
        tilingData.set_batch(64);
        EXPECT_EQ(tilingData.get_batch(), 64U);
        tilingData.set_k(2048);
        EXPECT_EQ(tilingData.get_k(), 2048U);
        tilingData.set_n(7168);
        EXPECT_EQ(tilingData.get_n(), 7168U);
        tilingData.set_vBaseM(16);
        EXPECT_EQ(tilingData.get_vBaseM(), 16U);
        tilingData.set_ubCalSize(4096);
        EXPECT_EQ(tilingData.get_ubCalSize(), 4096U);
        tilingData.set_ubRestBytes(126976);
        EXPECT_EQ(tilingData.get_ubRestBytes(), 126976U);
        tilingData.set_parallNum(4);
        EXPECT_EQ(tilingData.get_parallNum(), 4U);
        tilingData.set_sharedInputOffset(0);
        EXPECT_EQ(tilingData.get_sharedInputOffset(), 0U);
        tilingData.set_sharedInputLen(1024);
        EXPECT_EQ(tilingData.get_sharedInputLen(), 1024U);
        tilingData.set_residualScale(1.0f);
        EXPECT_FLOAT_EQ(tilingData.get_residualScale(), 1.0f);
        tilingData.set_quantGroupNum(112);
        EXPECT_EQ(tilingData.get_quantGroupNum(), 112U);
        tilingData.set_withOffset(1);
        EXPECT_EQ(tilingData.get_withOffset(), 1U);
        tilingData.set_hasPertokenScale(1);
        EXPECT_EQ(tilingData.get_hasPertokenScale(), 1U);
        tilingData.set_hasBias(1);
        EXPECT_EQ(tilingData.get_hasBias(), 1U);
        tilingData.set_deterministicFlag(1);
        EXPECT_EQ(tilingData.get_deterministicFlag(), 1U);
        tilingData.set_deterWorkspaceSize(96 * 1024 * 1024);
        EXPECT_EQ(tilingData.get_deterWorkspaceSize(), 96U * 1024U * 1024U);
    }

    string caseName;
    bool enable = true;
    string prefix;
    string aicNum;
    string aivNum;
    string ubSize;
    string l1Size;
    string l2Size;
    string l0CSize;
    string l0ASize;
    string l0BSize;
    string btSize;
    string cubeFreq;
    string socVersion;
    string supportL0c2out;
    string supportL12BtBf16;
    string npuArch;
    string xShape;
    string wShape;
    string scaleShape;
    string biasShape;
    string pertokenScaleShape;
    string groupListShape;
    string sharedInputShape;
    string logitShape;
    string rowindexShape;
    string yShape;
    string xDtype;
    string wDtype;
    string scaleDtype;
    string biasDtype;
    string pertokenScaleDtype;
    string groupListDtype;
    string sharedInputDtype;
    string logitDtype;
    string rowindexDtype;
    string yDtype;
    string wFormat;
    string dtypeAttr;
    string sharedInputWeight;
    string sharedInputOffset;
    string transposeX;
    string transposeW;
    string outputBs;
    string groupListType;
    string tuningConfig;
    string offsetShape;
    string offsetDtype;
    string deterministicInfo;
    string expectResult;
    string expectTilingKey;
    string caseType;
};

vector<GroupedMatmulFinalizeRoutingTilingCase> LoadCases(const string& csvPath)
{
    vector<GroupedMatmulFinalizeRoutingTilingCase> cases;
    ifstream csvData(csvPath, ios::in);
    if (!csvData.is_open()) {
        ADD_FAILURE() << "cannot open case file " << csvPath;
        return cases;
    }

    string line;
    while (getline(csvData, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        vector<string> items;
        SplitStr2Vec(line, ",", items);
        if (items.empty() || items[0] == "caseName") {
            continue;
        }
        if (items.size() != kCsvColumnCount) {
            ADD_FAILURE() << "Bad csv row column count in " << csvPath << ": " << line;
            continue;
        }

        GroupedMatmulFinalizeRoutingTilingCase testCase;
        size_t idx = 0;
        testCase.caseName = Trim(items[idx++]);
        testCase.enable = ParseBool(items[idx++]);
        if (!testCase.enable) {
            continue;
        }
        testCase.prefix = Trim(items[idx++]);
        testCase.aicNum = Trim(items[idx++]);
        testCase.aivNum = Trim(items[idx++]);
        testCase.ubSize = Trim(items[idx++]);
        testCase.l1Size = Trim(items[idx++]);
        testCase.l2Size = Trim(items[idx++]);
        testCase.l0CSize = Trim(items[idx++]);
        testCase.l0ASize = Trim(items[idx++]);
        testCase.l0BSize = Trim(items[idx++]);
        testCase.btSize = Trim(items[idx++]);
        testCase.cubeFreq = Trim(items[idx++]);
        testCase.socVersion = Trim(items[idx++]);
        testCase.supportL0c2out = Trim(items[idx++]);
        testCase.supportL12BtBf16 = Trim(items[idx++]);
        testCase.npuArch = Trim(items[idx++]);
        testCase.xShape = Trim(items[idx++]);
        testCase.wShape = Trim(items[idx++]);
        testCase.scaleShape = Trim(items[idx++]);
        testCase.biasShape = Trim(items[idx++]);
        testCase.pertokenScaleShape = Trim(items[idx++]);
        testCase.groupListShape = Trim(items[idx++]);
        testCase.sharedInputShape = Trim(items[idx++]);
        testCase.logitShape = Trim(items[idx++]);
        testCase.rowindexShape = Trim(items[idx++]);
        testCase.yShape = Trim(items[idx++]);
        testCase.xDtype = Trim(items[idx++]);
        testCase.wDtype = Trim(items[idx++]);
        testCase.scaleDtype = Trim(items[idx++]);
        testCase.biasDtype = Trim(items[idx++]);
        testCase.pertokenScaleDtype = Trim(items[idx++]);
        testCase.groupListDtype = Trim(items[idx++]);
        testCase.sharedInputDtype = Trim(items[idx++]);
        testCase.logitDtype = Trim(items[idx++]);
        testCase.rowindexDtype = Trim(items[idx++]);
        testCase.yDtype = Trim(items[idx++]);
        testCase.wFormat = Trim(items[idx++]);
        testCase.dtypeAttr = Trim(items[idx++]);
        testCase.sharedInputWeight = Trim(items[idx++]);
        testCase.sharedInputOffset = Trim(items[idx++]);
        testCase.transposeX = Trim(items[idx++]);
        testCase.transposeW = Trim(items[idx++]);
        testCase.outputBs = Trim(items[idx++]);
        testCase.groupListType = Trim(items[idx++]);
        testCase.tuningConfig = Trim(items[idx++]);
        testCase.offsetShape = Trim(items[idx++]);
        testCase.offsetDtype = Trim(items[idx++]);
        testCase.deterministicInfo = Trim(items[idx++]);
        testCase.expectResult = Trim(items[idx++]);
        testCase.expectTilingKey = Trim(items[idx++]);
        testCase.caseType = Trim(items[idx++]);
        cases.emplace_back(testCase);
    }

    EXPECT_FALSE(cases.empty()) << "No cases loaded from " << csvPath;
    return cases;
}

const vector<GroupedMatmulFinalizeRoutingTilingCase>& GetCases()
{
    static const auto cases =
        LoadCases(ops::ut::ResolveCsvPath("test_grouped_matmul_finalize_routing_tiling.csv",
                                          "gmm/grouped_matmul_finalize_routing/tests/ut/op_host", __FILE__));
    return cases;
}

string MakeParamName(const testing::TestParamInfo<GroupedMatmulFinalizeRoutingTilingCase>& info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

} // namespace

namespace GroupedMatmulFinalizeRoutingTilingUT {

class TestGroupedMatmulFinalizeRoutingTiling : public testing::TestWithParam<GroupedMatmulFinalizeRoutingTilingCase> {};

TEST_P(TestGroupedMatmulFinalizeRoutingTiling, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMMFR_TILING_CSV, TestGroupedMatmulFinalizeRoutingTiling, testing::ValuesIn(GetCases()),
                         MakeParamName);

class TestTilingParseForGroupedMatmulFinalizeRouting : public testing::Test {
protected:
};

TEST_F(TestTilingParseForGroupedMatmulFinalizeRouting, NullContext)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("GroupedMatmulFinalizeRouting");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);
    auto tilingParseFunc = opImpl->tiling_parse;
    auto result = tilingParseFunc(nullptr);
    EXPECT_NE(result, ge::GRAPH_SUCCESS);
}

} // namespace GroupedMatmulFinalizeRoutingTilingUT
