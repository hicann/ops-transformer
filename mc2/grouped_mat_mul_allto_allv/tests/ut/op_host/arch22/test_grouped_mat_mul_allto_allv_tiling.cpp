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
#include "mc2/common/tests/ut/internal_fallback_assertions.h"
#include "../../../../op_host/op_tiling/grouped_mat_mul_allto_allv_tiling.h"
#include "mc2_tiling_case_executor.h"
#include "mc2/common/tests/ut/formal_error_assertions.h"
#include "mc2_hcom_topo_info.h"

namespace GroupedMatMulAlltoAllvUT {

struct TestParam {
    string testName{};
    std::vector<std::pair<string, string>> tilingParamsStrPair{};
    std::vector<std::pair<string, std::vector<int64_t>>> tilingParamsVecPair{};
    std::vector<std::pair<size_t, ge::DataType>> tilingDTypesPair{};
    ge::graphStatus status;
    bool omitCommMode{false};
    std::string socVersion{"Ascend910_93"};
    std::string commMode{"ai_cpu"};
    // 960dt+ccu 组网 mock 形态: default(单层Mesh+CLOS混联,白名单拦截面)/fullmesh(纯fullmesh 1D主路径)/
    // clos(纯CLOS)/multilevel(跨机多层)/queryfail(组网查询失败)/ranksize(超rank上限>16)
    std::string topoMock{"default"};
    uint64_t expectTilingKey{9UL};
};

struct TilingParams {
    uint64_t BSK{4096};
    uint64_t BS{2048};
    uint64_t K{2};
    uint64_t H1{7168};
    uint64_t H2{7168};
    uint64_t A{4096};
    uint64_t N1{4096};
    uint64_t N2{64};
    uint64_t epWorldSize{8};
    uint64_t e{4};
    uint64_t aivCoreNum{40};
    uint64_t aicCoreNum{20};
    uint64_t gmmWeightDim1{7168};
    uint64_t yDim1{4096};
    uint64_t mmWeightDim0{7168};
    bool transGmmWeight{false};
    bool transMmWeight{false};
    std::string group{"group"};
    std::vector<int64_t> sendCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    std::vector<int64_t> recvCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
};

std::vector<int64_t> sendCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
std::vector<int64_t> recvCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};

std::unordered_map<string, std::function<void(TilingParams& tilingParams, const string& valueStr)>>
    g_tilingParamsStrHandlers = {
        {"BSK", [](TilingParams& tilingParams, const string& valueStr) { tilingParams.BSK = std::stoi(valueStr); }},
        {"BS", [](TilingParams& tilingParams, const string& valueStr) { tilingParams.BS = std::stoi(valueStr); }},
        {"K", [](TilingParams& tilingParams, const string& valueStr) { tilingParams.K = std::stoi(valueStr); }},
        {"H1", [](TilingParams& tilingParams, const string& valueStr) { tilingParams.H1 = std::stoi(valueStr); }},
        {"H2", [](TilingParams& tilingParams, const string& valueStr) { tilingParams.H2 = std::stoi(valueStr); }},
        {"A", [](TilingParams& tilingParams, const string& valueStr) { tilingParams.A = std::stoi(valueStr); }},
        {"N1", [](TilingParams& tilingParams, const string& valueStr) { tilingParams.N1 = std::stoi(valueStr); }},
        {"N2", [](TilingParams& tilingParams, const string& valueStr) { tilingParams.N2 = std::stoi(valueStr); }},
        {"epWorldSize",
         [](TilingParams& tilingParams, const string& valueStr) { tilingParams.epWorldSize = std::stoi(valueStr); }},
        {"e", [](TilingParams& tilingParams, const string& valueStr) { tilingParams.e = std::stoi(valueStr); }},
        {"gmmWeightDim1",
         [](TilingParams& tilingParams, const string& valueStr) { tilingParams.gmmWeightDim1 = std::stoi(valueStr); }},
        {"yDim1", [](TilingParams& tilingParams, const string& valueStr) { tilingParams.yDim1 = std::stoi(valueStr); }},
        {"mmWeightDim0",
         [](TilingParams& tilingParams, const string& valueStr) { tilingParams.mmWeightDim0 = std::stoi(valueStr); }},
        {"transGmmWeight",
         [](TilingParams& tilingParams, const string& valueStr) { tilingParams.transGmmWeight = valueStr == "true"; }},
        {"transMmWeight",
         [](TilingParams& tilingParams, const string& valueStr) { tilingParams.transMmWeight = valueStr == "true"; }}};

std::unordered_map<string, std::function<void(TilingParams& tilingParams, const std::vector<int64_t> valueVec)>>
    g_tilingParamsVecHandlers = {
        {"sendCounts",
         [](TilingParams& tilingParams, const std::vector<int64_t> valueVec) { tilingParams.sendCounts = valueVec; }},
        {"recvCounts",
         [](TilingParams& tilingParams, const std::vector<int64_t> valueVec) { tilingParams.recvCounts = valueVec; }}};

class GroupedMatMulAlltoAllvArch22TilingTest : public testing::TestWithParam<TestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatMulAlltoAllvArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatMulAlltoAllvArch22TilingTest TearDown" << std::endl;
    }

    void TearDown() override
    {
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    }
};

// 960dt(a6)+CCU 组网白名单 mock 路由: 正向用例显式 mock 单层纯 fullmesh 1D 主路径
// (六键显式设置, 不隐式依赖 mocker 默认值; mocker 默认为单层 Mesh+CLOS 混联, 是 960dt CCU
// 拦截面, 正向用例不 mock 会被默认拦截); 四类拦截面负向按 topoMock 逐类构造
static void ApplyCcuTopoMock(Mc2Hcom::MockValues& mockValues, const std::string& socVersion,
                             const std::string& commMode, const std::string& topoMock)
{
    if (socVersion != "Ascend960DT" || commMode != "ccu") {
        return;
    }
    auto addFullmesh1DMainPath = [&mockValues](uint64_t userRankSize, uint64_t levelNums = 1U,
                                               uint64_t topoInfoRet = static_cast<uint64_t>(HCCL_SUCCESS)) {
        mockValues.emplace_back("l0InstTypeSet", 1ULL << COMM_TOPO_1DMESH);  // 实例类型位图恰为 1DMESH 单位
        mockValues.emplace_back("l0CoverTypeSet", 1ULL << COMM_TOPO_1DMESH); // 覆盖类型位图恰为 1DMESH 单位
        mockValues.emplace_back("level0Topo", static_cast<uint64_t>(Mc2Hcom::Mc2Level0Topo::MESH_1D)); // 日志辅助字段
        mockValues.emplace_back("topoLevelNums", levelNums);    // 单层组网(默认)
        mockValues.emplace_back("getTopoInfoRet", topoInfoRet); // 组网查询结果(默认成功)
        mockValues.emplace_back("userRankSize", userRankSize);  // 防御面正常态(<=16)
    };
    if (topoMock == "fullmesh") {
        addFullmesh1DMainPath(8U);
    } else if (topoMock == "clos") { // 拦截面①: 纯CLOS
        mockValues.emplace_back("l0InstTypeSet", 1ULL << COMM_TOPO_CLOS);
        mockValues.emplace_back("l0CoverTypeSet", 1ULL << COMM_TOPO_CLOS);
        mockValues.emplace_back("level0Topo", static_cast<uint64_t>(Mc2Hcom::Mc2Level0Topo::CLOS));
    } else if (topoMock == "multilevel") { // 拦截面②: 跨机多层
        addFullmesh1DMainPath(8U, 2U);
    } else if (topoMock == "queryfail") { // 拦截面③: 组网查询失败
        addFullmesh1DMainPath(8U, 1U, static_cast<uint64_t>(HCCL_E_PARA));
    } else if (topoMock == "ranksize") { // 防御面: 超rank上限(>16)
        addFullmesh1DMainPath(32U);
    }
    // "default": 保持默认混联形态, 即 960dt CCU 拦截面
}

TEST_P(GroupedMatMulAlltoAllvArch22TilingTest, ShapeSize)
{
    auto testParam = GetParam();
    Mc2ErrorCodeUT::ExcelErrorReportScope errors;
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = testParam.socVersion;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    auto tilingParams = TilingParams{};
    for (auto& kv : testParam.tilingParamsStrPair) {
        if (g_tilingParamsStrHandlers.count(kv.first) != 0) {
            g_tilingParamsStrHandlers[kv.first](tilingParams, kv.second);
        }
    }
    for (auto& kv : testParam.tilingParamsVecPair) {
        if (g_tilingParamsVecHandlers.count(kv.first) != 0) {
            g_tilingParamsVecHandlers[kv.first](tilingParams, kv.second);
        }
    }

    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{tilingParams.A, tilingParams.H1}, {tilingParams.A, tilingParams.H1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tilingParams.e, tilingParams.gmmWeightDim1, tilingParams.N1},
              {tilingParams.e, tilingParams.gmmWeightDim1, tilingParams.N1}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tilingParams.BS, tilingParams.H2}, {tilingParams.BS, tilingParams.H2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tilingParams.mmWeightDim0, tilingParams.N2}, {tilingParams.mmWeightDim0, tilingParams.N2}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},
        },
        {
            {{{tilingParams.BSK, tilingParams.yDim1}, {tilingParams.BSK, tilingParams.yDim1}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},
            {{{tilingParams.BS, tilingParams.N2}, {tilingParams.BS, tilingParams.N2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.epWorldSize)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tilingParams.sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tilingParams.recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>(testParam.commMode)},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    if (testParam.omitCommMode) {
        tilingContextPara.attrs_.pop_back();
    }
    // attrs_ is already RuntimeAttrs storage: removing its last entry simulates
    // a missing materialized default slot, not legal omission at the public IR.
    (void)ErrorManager::GetInstance().GetRawErrorMessages();
    Mc2Hcom::MockValues hcomTopologyMockValues{
        {"rankNum", testParam.status == ge::GRAPH_FAILED ? 8U : tilingParams.epWorldSize}};
    ApplyCcuTopoMock(hcomTopologyMockValues, socVersion, testParam.commMode, testParam.topoMock);
    if (testParam.status == ge::GRAPH_FAILED) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
    } else {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, testParam.expectTilingKey);
    }
    if (testParam.testName == "Test_missing_comm_mode_retains_internal_fallback_a3") {
        Mc2ErrorCodeUT::ExpectInternalFallbackReport(
            "GetAttrPointer(comm_mode) returned null; expected default storage");
    } else if (testParam.testName == "Test_default_comm_mode_materialized_a3" ||
               testParam.testName == "Test_a3_aicpu_topk8") {
        EXPECT_TRUE(ErrorManager::GetInstance().GetRawErrorMessages().empty());
    } else if (testParam.testName == "Test_a3_aicpu_topk9") {
        Mc2ErrorCodeUT::ExpectExcelFormalError("EZ0024", "K", "9", nullptr, nullptr,
                                               {{"correct_value", "[2, 8]"}, {"op_name", "GroupedMatMulAlltoAllv"}});
    }
    (void)ErrorManager::GetInstance().GetRawErrorMessages();
    // 960dt+ccu 白名单正向: 断言生效算法名被强制为 sole[mesh](经 Mc2CcTilingConfig 写入
    // tilingData 的 algConfig, 覆盖 legacy 默认名 AlltoAll=level0:fullmesh;level1:pairwise)
    if (socVersion == "Ascend960DT" && testParam.commMode == "ccu" && testParam.status == ge::GRAPH_SUCCESS) {
        // Mc2ExecuteTestCase 内部已 Reset mocker, 断言用的二次执行前需重新注入白名单组网 mock
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
        TilingInfo tilingInfo;
        const bool tilingOk = ExecuteTiling(tilingContextPara, tilingInfo);
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
        ASSERT_TRUE(tilingOk);
        const std::string rawTiling(reinterpret_cast<const char*>(tilingInfo.tilingData.get()),
                                    tilingInfo.tilingDataSize);
        EXPECT_NE(rawTiling.find("sole[mesh]"), std::string::npos)
            << "960dt+ccu effective algConfig should be forced to sole[mesh]";
    }
}

static TestParam g_testParams[] = {
    {"Test_sample", {}, {}, {}, ge::GRAPH_SUCCESS},
    {"Test_a2_aicpu_rejected", {}, {}, {}, ge::GRAPH_FAILED, false, "Ascend910B"},
    {"Test_a2_missing_comm_mode_rejected", {}, {}, {}, ge::GRAPH_FAILED, true, "Ascend910B"},
    {"Test_a2_default_rejected", {}, {}, {}, ge::GRAPH_FAILED, false, "Ascend910B", "default"},
    {"Test_a2_ccu_rejected", {}, {}, {}, ge::GRAPH_FAILED, false, "Ascend910B", "ccu"},
    {"Test_a5_aicpu_unchanged", {}, {}, {}, ge::GRAPH_SUCCESS, false, "Ascend950"},
    {"Test_missing_comm_mode_retains_internal_fallback_a3", {}, {}, {}, ge::GRAPH_FAILED, true, "Ascend910_93"},
    // The explicit ai_cpu slot matches the OpDef default; this fixture does not materialize IR defaults.
    {"Test_default_comm_mode_materialized_a3", {}, {}, {}, ge::GRAPH_SUCCESS, false, "Ascend910_93", "ai_cpu"},
    // A5 validates commMode in SetHcclTiling, before the final key conversion.
    {"Test_a5_missing_comm_mode_rejected", {}, {}, {}, ge::GRAPH_FAILED, true, "Ascend950"},
    {"Test_a3_aicpu_topk8", {{"BS", "512"}}, {}, {}, ge::GRAPH_SUCCESS},
    {"Test_a3_aicpu_topk9",
     {{"BS", "512"}, {"BSK", "4608"}},
     {{"recvCounts", std::vector<int64_t>(32, 144)}},
     {},
     ge::GRAPH_FAILED},
    {"Test_a5_aicpu_topk8", {{"BS", "512"}}, {}, {}, ge::GRAPH_SUCCESS, false, "Ascend950"},
    {"Test_a5_aicpu_topk9",
     {{"BS", "512"}, {"BSK", "4608"}},
     {{"recvCounts", std::vector<int64_t>(32, 144)}},
     {},
     ge::GRAPH_FAILED,
     false,
     "Ascend950"},
    // A5 CCU: 白名单对非960dt零干预, 默认算法配置, key=1(有共享专家/无转置/CCU)
    {"Test_a5_ccu", {}, {}, {}, ge::GRAPH_SUCCESS, false, "Ascend950", "ccu", "default", 1UL},
    // A6(960dt) ai_cpu: 与A5行为一致, key=9
    {"Test_a6_aicpu_unchanged", {}, {}, {}, ge::GRAPH_SUCCESS, false, "Ascend960DT", "ai_cpu", "default", 9UL},
    // A6 ccu 白名单正向: 单层纯fullmesh 1D mock放行并强制sole[mesh], key=1
    {"Test_a6_ccu_fullmesh", {}, {}, {}, ge::GRAPH_SUCCESS, false, "Ascend960DT", "ccu", "fullmesh", 1UL},
    // A6 ccu 拦截面: 默认混联/纯CLOS/跨机多层/组网查询失败/超rank上限(>16)
    {"Test_a6_ccu_meshclos_blocked", {}, {}, {}, ge::GRAPH_FAILED, false, "Ascend960DT", "ccu", "default"},
    {"Test_a6_ccu_clos_blocked", {}, {}, {}, ge::GRAPH_FAILED, false, "Ascend960DT", "ccu", "clos"},
    {"Test_a6_ccu_multilevel_blocked", {}, {}, {}, ge::GRAPH_FAILED, false, "Ascend960DT", "ccu", "multilevel"},
    {"Test_a6_ccu_queryfail_blocked", {}, {}, {}, ge::GRAPH_FAILED, false, "Ascend960DT", "ccu", "queryfail"},
    {"Test_a6_ccu_ranksize_blocked", {}, {}, {}, ge::GRAPH_FAILED, false, "Ascend960DT", "ccu", "ranksize"},
    {"Test_BSK_1", {{"BSK", "52428800"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_BS_1", {{"BS", "52428800"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H1", {{"H1", "65536"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H2", {{"H2", "12889"}, {"mmWeightDim0", "12889"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_N1", {{"N1", "65536"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_N2", {{"N2", "65536"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_ep_world_size", {{"epWorldSize", "4"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_send_counts_size", {{"epWorldSize", "16"}, {"e", "32"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H_1", {{"H1", "7168"}, {"gmmWeightDim1", "7169"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H_3", {{"H2", "7168"}, {"mmWeightDim0", "7169"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_send_counts_0",
     {{"A", "16386"}},
     {{"sendCounts",
       std::vector<int64_t>{
           3201, 3201, 3200, 3200, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
           128,  128,  128,  128,  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
       }}},
     {},
     ge::GRAPH_SUCCESS},
    {"Test_recv_counts_0",
     {{"BSK", "16386"}, {"BS", "8193"}},
     {{"recvCounts",
       std::vector<int64_t>{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,
                            128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 3200}}},
     {},
     ge::GRAPH_SUCCESS},
    {"Test_recv_counts_1",
     {{"BSK", "16386"}},
     {{"recvCounts",
       std::vector<int64_t>{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,  128, 128,
                            128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 1600, 1600}}},
     {},
     ge::GRAPH_FAILED},
};

INSTANTIATE_TEST_SUITE_P(GroupedMatMulAlltoAllv, GroupedMatMulAlltoAllvArch22TilingTest,
                         testing::ValuesIn(g_testParams),
                         [](const testing::TestParamInfo<GroupedMatMulAlltoAllvArch22TilingTest::ParamType>& info) {
                             return info.param.testName;
                         });

TEST_F(GroupedMatMulAlltoAllvArch22TilingTest, Dim1)
{
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{2, 4096, 7168}, {2, 4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(GroupedMatMulAlltoAllvArch22TilingTest, Dim2)
{
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 4096}, {7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(GroupedMatMulAlltoAllvArch22TilingTest, Dim3)
{
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{2, 4096, 7168}, {2, 4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(GroupedMatMulAlltoAllvArch22TilingTest, Dim4)
{
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2048, 7168}, {2, 2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 6464}, {7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(GroupedMatMulAlltoAllvArch22TilingTest, Dim5)
{
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 7168, 6464}, {2, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(GroupedMatMulAlltoAllvArch22TilingTest, Dim6)
{
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 6464}, {7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2048, 64}, {2, 2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(GroupedMatMulAlltoAllvArch22TilingTest, TransMmWeightInvalid)
{
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    gert::TilingContextPara tilingContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"hcom", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcom")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingData);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}
} // namespace GroupedMatMulAlltoAllvUT
