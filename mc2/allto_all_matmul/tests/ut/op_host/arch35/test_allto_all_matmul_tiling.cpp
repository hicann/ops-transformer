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
#include <cstdlib>
#include "../allto_all_matmul_host_ut_param.h"
#include "tiling_case_executor.h"
#include "mc2_hcom_topology_mocker.h"
#include "mc2_hcom_topo_info.h"

static std::string GetCsvPath(const char* file)
{
    const char* envPath = std::getenv("CSV_CASE_PATH");
    if (envPath != nullptr && strlen(envPath) > 0) {
        return std::string(envPath);
    }
    return ReplaceFileExtension2Csv(file);
}

namespace AlltoAllMatmulUT {

static const std::string OP_NAME = "AlltoAllMatmul";

struct AlltoAllMatmulCompileInfo {
} compileInfo;

// 960dt(a6)+CCU 白名单正向 mock 模板: 单层纯 fullmesh 1D 主路径(六键显式设置, 不隐式依赖 mocker 默认值;
// mocker 默认为单层 Mesh+CLOS 混联, 是 960dt CCU 拦截面, 正向用例不 mock 会被默认拦截)
static void SetFullmesh1DMainPathMock(uint64_t userRankSize)
{
    auto& mocker = Mc2Hcom::MC2HcomTopologyMocker::GetInstance();
    mocker.SetValue("l0InstTypeSet", 1ULL << COMM_TOPO_1DMESH);  // 实例类型位图恰为 1DMESH 单位
    mocker.SetValue("l0CoverTypeSet", 1ULL << COMM_TOPO_1DMESH); // 覆盖类型位图恰为 1DMESH 单位
    mocker.SetValue("level0Topo", static_cast<uint64_t>(Mc2Hcom::Mc2Level0Topo::MESH_1D)); // 日志辅助字段
    mocker.SetValue("topoLevelNums", 1U);                                                  // 单层组网
    mocker.SetValue("getTopoInfoRet", static_cast<uint64_t>(HCCL_SUCCESS));                // 组网查询成功
    mocker.SetValue("userRankSize", userRankSize);                                         // 防御面正常态(<=16)
}

class AlltoAllMatmulArch35TilingTest : public testing::TestWithParam<AlltoAllMatmulTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllMatmulArch35TilingTest SetUp." << std::endl;
    }

    void TearDown() override
    {
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllMatmulArch35TilingTest TearDown." << std::endl;
    }
};

TEST_P(AlltoAllMatmulArch35TilingTest, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {param.x1, param.x2, param.bias, param.x1Scale, param.x2Scale, param.commScale, param.x1Offset,
         param.x2Offset});

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_({param.y, param.all2allOut});

    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.worldSize)},
         {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.all2allAxes))},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.yDtypeAttr))},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantMode)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x2QuantMode)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantMode)},
         {"x1_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantDtype)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.commQuantDtype)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeX1)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeX2)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSize)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commMode)},
         {"alltoallout_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(param.alltoalloutFlag)}});

    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.soc);
    if (param.worldSize > 0 && param.worldSize <= 64 &&
        ((param.x1QuantMode == 6 && param.x2QuantMode == 6) ||
         (param.x1QuantMode == 8 && param.x2QuantMode == 8))) { // MX与HiF4量化均需mock通信域rank数
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValue("rankNum", static_cast<uint64_t>(param.worldSize));
    }
    // 960dt+CCU 组网白名单 mock 路由: 正向用例显式 mock 单层纯 fullmesh 1D 主路径; 四类拦截面负向
    // 按用例名逐类构造(meshclos 为 mocker 默认形态零设置)
    if (param.soc == "9201" && param.commMode == "ccu" && param.expectResult == ge::GRAPH_SUCCESS) {
        SetFullmesh1DMainPathMock(8U);
    } else if (param.soc == "9201" && param.commMode == "ccu" &&
               param.case_name.find("_clos_blocked") != std::string::npos) { // 拦截面：纯CLOS
        auto& mocker = Mc2Hcom::MC2HcomTopologyMocker::GetInstance();
        mocker.SetValue("l0InstTypeSet", 1ULL << COMM_TOPO_CLOS);
        mocker.SetValue("l0CoverTypeSet", 1ULL << COMM_TOPO_CLOS);
        mocker.SetValue("level0Topo", static_cast<uint64_t>(Mc2Hcom::Mc2Level0Topo::CLOS));
    } else if (param.soc == "9201" && param.commMode == "ccu" &&
               param.case_name.find("_multilevel_blocked") != std::string::npos) { // 拦截面：跨机多层
        SetFullmesh1DMainPathMock(8U);
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValue("topoLevelNums", 2U);
    } else if (param.soc == "9201" && param.commMode == "ccu" &&
               param.case_name.find("_queryfail_blocked") != std::string::npos) { // 拦截面：组网查询失败
        SetFullmesh1DMainPathMock(8U);
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValue("getTopoInfoRet", static_cast<uint64_t>(HCCL_E_PARA));
    } else if (param.soc == "9201" && param.commMode == "ccu" &&
               param.case_name.find("_ranksize_blocked") != std::string::npos) { // 防御面：超rank上限（>16）
        SetFullmesh1DMainPathMock(32U);
    } else if (param.soc == "9201" && param.commMode == "ai_cpu" &&
               param.case_name.find("fullmesh_mock") != std::string::npos) {
        // 默认回归: AICPU 在白名单组网下行为一致（白名单函数引擎前置过滤零干预）
        SetFullmesh1DMainPathMock(8U);
    }
    ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, param.expectTilingData,
                    param.expectWorkspaces, param.mc2TilingDataReservedLen);
    // 960dt+ccu 白名单正向: 断言生效算法名被强制为 sole[mesh](经 Mc2CcTilingConfigBuilder 写入
    // tilingData 的 mc2CcTiling.algConfig, 覆盖算法表兜底 legacy 默认名)
    if (param.soc == "9201" && param.commMode == "ccu" && param.expectResult == ge::GRAPH_SUCCESS) {
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
        const std::string rawTiling(reinterpret_cast<const char*>(tilingInfo.tilingData.get()),
                                    tilingInfo.tilingDataSize);
        EXPECT_NE(rawTiling.find("sole[mesh]"), std::string::npos)
            << "960dt+ccu effective algoName (mc2CcTiling.algConfig) should be forced to sole[mesh]";
    }
}

INSTANTIATE_TEST_SUITE_P(AlltoAllMatmulTilingUT, AlltoAllMatmulArch35TilingTest,
                         testing::ValuesIn(GetCasesFromCsv<AlltoAllMatmulTilingUtParam>(GetCsvPath(__FILE__))),
                         PrintCaseInfoString<AlltoAllMatmulTilingUtParam>);

} // namespace AlltoAllMatmulUT
