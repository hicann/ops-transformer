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
#include "mc2_tiling_case_executor.h"
#include "mc2_hcom_topo_info.h"
#include "../allto_allv_quant_grouped_mat_mul_host_ut_param.h"
#include "../../../../op_host/op_tiling/arch35/allto_allv_quant_grouped_mat_mul_tiling_common.h"

using namespace std;

namespace AlltoAllvQuantGroupedMatMulUT {

class AlltoAllvQuantGroupedMatMulTilingTest
    : public ::testing::TestWithParam<AlltoAllvQuantGroupedMatMulTilingUTParam> {
protected:
    static void SetUpTestCase()
    {
        cout << "AlltoAllvQuantGroupedMatMulTilingTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "AlltoAllvQuantGroupedMatMulTilingTest TearDown" << endl;
    }

    void TearDown() override
    {
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    }
};

TEST(AlltoAllvQuantGroupedMatMulTilingStandaloneTest, mxApaceNetworkCaseUsesFourExpertsPerBatch)
{
    constexpr uint64_t expertNum = 32U;
    constexpr uint64_t rankSize = 2U;
    constexpr uint64_t bsk = 100352U;
    constexpr uint64_t n = 1024U;
    constexpr uint32_t expectedBatchExpertNum = 4U;
    constexpr uint32_t expectedLoopCount = 8U;

    const uint32_t batchExpertNum = optiling::AlltoAllvQuantGmmTilingCommon::CalcExpertNum(expertNum, rankSize, bsk, n);
    EXPECT_EQ(batchExpertNum, expectedBatchExpertNum);
    EXPECT_EQ((expertNum + batchExpertNum - 1U) / batchExpertNum, expectedLoopCount);
}

// 960dt(a6)+CCU 组网白名单 mock 路由(对齐PR#15055用例形态): 正向用例显式 mock 单层纯 fullmesh 1D
// 主路径(六键显式设置, 不隐式依赖 mocker 默认值; mocker 默认为单层 Mesh+CLOS 混联, 是 960dt CCU
// 拦截面, 正向用例不 mock 会被默认拦截); 四类拦截面负向按用例名逐类构造
static Mc2Hcom::MockValues BuildHcomTopologyMockValues(const AlltoAllvQuantGroupedMatMulTilingUTParam& param)
{
    Mc2Hcom::MockValues mockValues{{"rankNum", 8}};
    if (param.soc != "9201") {
        return mockValues;
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
    const auto& caseName = param.case_name;
    if (param.comm_mode == "ccu") {
        if (caseName.find("_clos_blocked") != std::string::npos) { // 拦截面①: 纯CLOS
            mockValues.emplace_back("l0InstTypeSet", 1ULL << COMM_TOPO_CLOS);
            mockValues.emplace_back("l0CoverTypeSet", 1ULL << COMM_TOPO_CLOS);
            mockValues.emplace_back("level0Topo", static_cast<uint64_t>(Mc2Hcom::Mc2Level0Topo::CLOS));
        } else if (caseName.find("_multilevel_blocked") != std::string::npos) { // 拦截面②: 跨机多层
            addFullmesh1DMainPath(8U, 2U);
        } else if (caseName.find("_queryfail_blocked") != std::string::npos) { // 拦截面③: 组网查询失败
            addFullmesh1DMainPath(8U, 1U, static_cast<uint64_t>(HCCL_E_PARA));
        } else if (caseName.find("_ranksize_blocked") != std::string::npos) { // 防御面: 超rank上限(>16)
            addFullmesh1DMainPath(32U);
        } else if (caseName.find("fullmesh_mock") != std::string::npos || param.expectedStatus == ge::GRAPH_SUCCESS) {
            addFullmesh1DMainPath(8U); // 白名单正向主路径; meshclos 拦截面用默认混联零设置
        }
    } else if (param.comm_mode == "ai_cpu" && caseName.find("fullmesh_mock") != std::string::npos) {
        // 默认回归: AICPU 在白名单组网下行为一致(白名单函数步骤②引擎前置过滤零干预)
        addFullmesh1DMainPath(8U);
    }
    return mockValues;
}

TEST_P(AlltoAllvQuantGroupedMatMulTilingTest, param)
{
    auto param = GetParam();

    struct AlltoAllvQuantGroupedMatMulCompileInfo {};
    AlltoAllvQuantGroupedMatMulCompileInfo compileInfo;
    uint64_t coreNum = 36;
    uint64_t ubSize = 256 * 1024;
    size_t tilingDataSize = sizeof(QuantAlltoAllvGroupedMatmulTilingData);

    gert::TilingContextPara tilingContextPara(
        "AlltoAllvQuantGroupedMatMul",
        {
            param.gmm_x,
            param.gmm_weight,
            param.gmm_x_scale,
            param.gmm_weight_scale,
            param.send_counts_tensor,
            param.recv_counts_tensor,
            param.mm_x,
            param.mm_x_weight,
            param.mm_x_scale,
            param.mm_x_weight_scale,
        },
        {param.gmm_y, param.mm_y, param.permute_out},
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.ep_world_size)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(param.sendCounts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(param.recvCounts)},
            {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmm_x_quant_mode)},
            {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmm_weight_quant_mode)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.trans_gmm_weight_flag)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.trans_mm_weight_flag)},
            {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(param.permute_out_flag)},
            {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mm_x_quant_mode)},
            {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mm_weight_quant_mode)},
            {"groupSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_size)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"mm_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.comm_mode)},
        },
        &compileInfo, param.soc, coreNum, ubSize, tilingDataSize);

    Mc2Hcom::MockValues hcomTopologyMockValues = BuildHcomTopologyMockValues(param);

    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectedStatus, param.expectTilingKey);
    // 960dt+ccu 白名单正向: 断言生效算法名被强制为 sole[mesh](经 Mc2CcTilingConfig 写入
    // tilingData 的 algConfig, 覆盖 legacy 默认名 AlltoAll=level0:fullmesh;level1:pairwise)
    if (param.soc == "9201" && param.comm_mode == "ccu" && param.expectedStatus == ge::GRAPH_SUCCESS) {
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

INSTANTIATE_TEST_SUITE_P(
    AlltoAllvQuantGroupedMatMul, AlltoAllvQuantGroupedMatMulTilingTest,
    testing::ValuesIn(GetCasesFromCsv<AlltoAllvQuantGroupedMatMulTilingUTParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AlltoAllvQuantGroupedMatMulTilingUTParam>);

} // namespace AlltoAllvQuantGroupedMatMulUT
