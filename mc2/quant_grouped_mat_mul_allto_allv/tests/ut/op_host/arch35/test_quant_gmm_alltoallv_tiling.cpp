/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <iostream>
#include "../quant_gmm_alltoallv_host_ut_param.h"
#include "mc2_tiling_case_executor.h"
#include "mc2_hcom_topo_info.h"
#include "../../../../op_host/op_tiling/arch35/tt_quant_grouped_mat_mul_allto_allv_tiling.h"
#include "../../../../op_host/op_tiling/arch35/mx_quant_grouped_mat_mul_allto_allv_tiling.h"

static std::string GetCsvPath(const char* file)
{
    const char* envPath = std::getenv("CSV_CASE_PATH");
    if (envPath != nullptr && strlen(envPath) > 0) {
        return std::string(envPath);
    }
    return ReplaceFileExtension2Csv(file);
}

namespace QuantGmmAlltoAllvUT {

static const std::string OP_NAME = "QuantGroupedMatMulAlltoAllv";

struct GroupedMatMulAlltoAllvCompileInfo {
} compileInfo;

class QuantGmmAlltoAllvTilingTest : public testing::TestWithParam<QuantGmmAlltoAllvTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantGmmAlltoAllvTilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantGmmAlltoAllvTilingTest TearDown." << std::endl;
    }

    void TearDown() override
    {
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    }
};

// 960dt(a6)+CCU 组网白名单 mock 路由: 正向用例显式 mock 单层纯 fullmesh 1D 主路径
// (六键显式设置, 不隐式依赖 mocker 默认值; mocker 默认为单层 Mesh+CLOS 混联, 是 960dt CCU
// 拦截面, 正向用例不 mock 会被默认拦截); 拦截面及边界用例按用例名逐类构造
static Mc2Hcom::MockValues BuildHcomTopologyMockValues(const QuantGmmAlltoAllvTilingUtParam& param)
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
    if (param.commMode == "ccu") {
        if (caseName.find("_clos_blocked") != std::string::npos) { // 拦截面①: 纯CLOS
            mockValues.emplace_back("l0InstTypeSet", 1ULL << COMM_TOPO_CLOS);
            mockValues.emplace_back("l0CoverTypeSet", 1ULL << COMM_TOPO_CLOS);
            mockValues.emplace_back("level0Topo", static_cast<uint64_t>(Mc2Hcom::Mc2Level0Topo::CLOS));
        } else if (caseName.find("_multilevel_blocked") != std::string::npos) { // 拦截面②: 跨机多层
            addFullmesh1DMainPath(8U, 2U);
        } else if (caseName.find("_queryfail_blocked") != std::string::npos) { // 拦截面③: 组网查询失败
            addFullmesh1DMainPath(8U, 1U, static_cast<uint64_t>(HCCL_E_PARA));
        } else if (caseName.find("_rank16_pass") != std::string::npos) { // 放行边界: rankSize恰为16
            addFullmesh1DMainPath(16U);
        } else if (caseName.find("_rank17_blocked") != std::string::npos) { // 拦截边界: rankSize超1(17>16)
            addFullmesh1DMainPath(17U);
        } else if (caseName.find("_ranksize_blocked") != std::string::npos) { // 拦截面④: 超rank上限(>16)
            addFullmesh1DMainPath(32U);
        } else if (caseName.find("fullmesh_mock") != std::string::npos || param.expectResult == ge::GRAPH_SUCCESS) {
            addFullmesh1DMainPath(8U); // 白名单正向主路径; meshclos 拦截面用默认混联零设置
        }
    } else if (param.commMode == "ai_cpu" && caseName.find("fullmesh_mock") != std::string::npos) {
        // 默认回归: AICPU 在白名单组网下行为一致(白名单函数引擎前置过滤零干预)
        addFullmesh1DMainPath(8U);
    }
    return mockValues;
}

TEST_P(QuantGmmAlltoAllvTilingTest, test_grouped_quant_mat_mul_allto_allv_tiling)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

#if !QUANT_GMM_A2AV_UT_A6_SUPPORTED
    // devkit < 9.3 无 Ascend960DT(9201) tiling 支持, A6 用例在该环境必然失败, 跳过
    if (param.soc == "9201") {
        GTEST_SKIP() << "A6(9201) cases require devkit >= 9.3 (Ascend960DT), skip";
    }
#endif

    if (param.gmmXShape.size() < 2 || param.gmmWeightShape.size() < 3 || param.gmmYShape.size() < 2) {
        FAIL() << "Invalid shape dimensions in test case: " << param.case_name;
        return;
    }

    uint64_t coreNum = 36;
    uint64_t ubSize = 256 * 1024;
    uint64_t gmmXScaleDimMx = 3;
    uint64_t gmmWeightScaleDimMx = 4;
    uint64_t mmXScaleDimMx = 3;
    uint64_t mmWeightScaleDimMx = 3;
    size_t tilingDataSize = sizeof(QuantGmmA2avTilingData);

    gert::StorageShape mmXStorageShape;
    if (param.mmXShape.size() >= 2 && param.mmXShape[0] > 0) {
        mmXStorageShape = {{param.mmXShape[0], param.mmXShape[1]}, {param.mmXShape[0], param.mmXShape[1]}};
    } else {
        mmXStorageShape = {};
    }

    gert::StorageShape mmWeightStorageShape;
    if (param.mmWeightShape.size() >= 2 && param.mmWeightShape[0] > 0) {
        mmWeightStorageShape = {{param.mmWeightShape[0], param.mmWeightShape[1]},
                                {param.mmWeightShape[0], param.mmWeightShape[1]}};
    } else {
        mmWeightStorageShape = {};
    }

    gert::StorageShape mmYStorageShape;
    if (param.mmYShape.size() >= 2 && param.mmYShape[0] > 0) {
        mmYStorageShape = {{param.mmYShape[0], param.mmYShape[1]}, {param.mmYShape[0], param.mmYShape[1]}};
    } else {
        mmYStorageShape = {};
    }

    gert::StorageShape gmmXScaleStorageShape;
    if (param.gmmXScaleShape.size() > 0 && param.gmmXScaleShape[0] > 0) {
        if (param.gmmXScaleShape.size() == gmmXScaleDimMx) {
            gmmXScaleStorageShape = {{param.gmmXScaleShape[0], param.gmmXScaleShape[1], param.gmmXScaleShape[2]},
                                     {param.gmmXScaleShape[0], param.gmmXScaleShape[1], param.gmmXScaleShape[2]}};
        } else {
            std::cerr << "[WARN] gmmXScaleShape unexpected dim: " << param.gmmXScaleShape.size()
                      << " in case: " << param.case_name << std::endl;
            gmmXScaleStorageShape = {{param.gmmXScaleShape[0]}, {param.gmmXScaleShape[0]}};
        }
    } else {
        gmmXScaleStorageShape = {};
    }

    gert::StorageShape gmmWeightScaleStorageShape;
    if (param.gmmWeightScaleShape.size() > 0 && param.gmmWeightScaleShape[0] > 0) {
        if (param.gmmWeightScaleShape.size() == gmmWeightScaleDimMx) {
            gmmWeightScaleStorageShape = {{param.gmmWeightScaleShape[0], param.gmmWeightScaleShape[1],
                                           param.gmmWeightScaleShape[2], param.gmmWeightScaleShape[3]},
                                          {param.gmmWeightScaleShape[0], param.gmmWeightScaleShape[1],
                                           param.gmmWeightScaleShape[2], param.gmmWeightScaleShape[3]}};
        } else {
            std::cerr << "[WARN] gmmWeightScaleShape unexpected dim: " << param.gmmWeightScaleShape.size()
                      << " in case: " << param.case_name << std::endl;
            gmmWeightScaleStorageShape = {{param.gmmWeightScaleShape[0]}, {param.gmmWeightScaleShape[0]}};
        }
    } else {
        gmmWeightScaleStorageShape = {};
    }

    gert::StorageShape mmXScaleStorageShape;
    if (param.mmXScaleShape.size() > 0 && param.mmXScaleShape[0] > 0) {
        if (param.mmXScaleShape.size() == mmXScaleDimMx) {
            mmXScaleStorageShape = {{param.mmXScaleShape[0], param.mmXScaleShape[1], param.mmXScaleShape[2]},
                                    {param.mmXScaleShape[0], param.mmXScaleShape[1], param.mmXScaleShape[2]}};
        } else {
            std::cerr << "[WARN] mmXScaleShape unexpected dim: " << param.mmXScaleShape.size()
                      << " in case: " << param.case_name << std::endl;
            mmXScaleStorageShape = {{param.mmXScaleShape[0]}, {param.mmXScaleShape[0]}};
        }
    } else {
        mmXScaleStorageShape = {};
    }

    gert::StorageShape mmWeightScaleStorageShape;
    if (param.mmWeightScaleShape.size() > 0 && param.mmWeightScaleShape[0] > 0) {
        if (param.mmWeightScaleShape.size() == mmWeightScaleDimMx) {
            mmWeightScaleStorageShape = {
                {param.mmWeightScaleShape[0], param.mmWeightScaleShape[1], param.mmWeightScaleShape[2]},
                {param.mmWeightScaleShape[0], param.mmWeightScaleShape[1], param.mmWeightScaleShape[2]}};
        } else {
            std::cerr << "[WARN] mmWeightScaleShape unexpected dim: " << param.mmWeightScaleShape.size()
                      << " in case: " << param.case_name << std::endl;
            mmWeightScaleStorageShape = {{param.mmWeightScaleShape[0]}, {param.mmWeightScaleShape[0]}};
        }
    } else {
        mmWeightScaleStorageShape = {};
    }

    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        {{{{param.gmmXShape[0], param.gmmXShape[1]}, {param.gmmXShape[0], param.gmmXShape[1]}},
          param.gmmXDataType,
          param.gmmXFormat},
         {{{param.gmmWeightShape[0], param.gmmWeightShape[1], param.gmmWeightShape[2]},
           {param.gmmWeightShape[0], param.gmmWeightShape[1], param.gmmWeightShape[2]}},
          param.gmmWeightDataType,
          param.gmmWeightFormat},
         {gmmXScaleStorageShape, param.gmmXScaleDataType, param.gmmXScaleFormat},
         {gmmWeightScaleStorageShape, param.gmmWeightScaleDataType, param.gmmWeightScaleFormat},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {{}, ge::DT_INT64, ge::FORMAT_ND},
         {mmXStorageShape, param.mmXDataType, param.mmXFormat},
         {mmWeightStorageShape, param.mmWeightDataType, param.mmWeightFormat},
         {mmXScaleStorageShape, param.mmXScaleDataType, param.mmXScaleFormat},
         {mmWeightScaleStorageShape, param.mmWeightScaleDataType, param.mmWeightScaleFormat},
         {{}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{param.gmmYShape[0], param.gmmYShape[1]}, {param.gmmYShape[0], param.gmmYShape[1]}},
          param.gmmYDataType,
          param.gmmYFormat},
         {mmYStorageShape, param.mmYDataType, param.mmYFormat}},
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.epWorldSize)},
            {"send_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.sendCounts)},
            {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.recvCounts)},
            {"gmm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmmXQuantMode)},
            {"gmm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gmmWeightQuantMode)},
            {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transGmmWeight)},
            {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transMmWeight)},
            {"mm_x_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mmXQuantMode)},
            {"mm_weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mmWeightQuantMode)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSize)},
            {"gmm_y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"mm_y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commMode)},
        },
        &compileInfo, param.soc, coreNum, ubSize, tilingDataSize);

    Mc2Hcom::MockValues hcomTopologyMockValues = BuildHcomTopologyMockValues(param);

    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey);
    // 960dt+ccu 白名单正向: 断言生效算法名被强制为 sole[mesh](经 Mc2CcTilingConfig 写入
    // tilingData 的 algConfig, 覆盖 legacy 默认名 AlltoAll=level0:fullmesh;level1:pairwise)
    if (param.soc == "9201" && param.commMode == "ccu" && param.expectResult == ge::GRAPH_SUCCESS) {
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

INSTANTIATE_TEST_SUITE_P(QuantGmmAlltoAllvTilingUT, QuantGmmAlltoAllvTilingTest,
                         testing::ValuesIn(GetCasesFromCsv<QuantGmmAlltoAllvTilingUtParam>(GetCsvPath(__FILE__))),
                         PrintCaseInfoString<QuantGmmAlltoAllvTilingUtParam>);

} // namespace QuantGmmAlltoAllvUT
