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
#include "../../../../op_host/op_tiling/grouped_mat_mul_allto_allv_tiling.h"
#include "mc2_tiling_case_executor.h"

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

std::unordered_map<string, std::function<void(TilingParams &tilingParams, const string &valueStr)>>
    g_tilingParamsStrHandlers = {
        {"BSK", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.BSK = std::stoi(valueStr); }},
        {"BS", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.BS = std::stoi(valueStr); }},
        {"K", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.K = std::stoi(valueStr); }},
        {"H1", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.H1 = std::stoi(valueStr); }},
        {"H2", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.H2 = std::stoi(valueStr); }},
        {"A", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.A = std::stoi(valueStr); }},
        {"N1", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.N1 = std::stoi(valueStr); }},
        {"N2", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.N2 = std::stoi(valueStr); }},
        {"epWorldSize",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.epWorldSize = std::stoi(valueStr); }},
        {"e", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.e = std::stoi(valueStr); }},
        {"gmmWeightDim1",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.gmmWeightDim1 = std::stoi(valueStr); }},
        {"yDim1", [](TilingParams &tilingParams, const string &valueStr) { tilingParams.yDim1 = std::stoi(valueStr); }},
        {"mmWeightDim0",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.mmWeightDim0 = std::stoi(valueStr); }},
        {"transGmmWeight",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.transGmmWeight = valueStr == "true"; }},
        {"transMmWeight",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.transMmWeight = valueStr == "true"; }}};

std::unordered_map<string, std::function<void(TilingParams &tilingParams, const std::vector<int64_t> valueVec)>>
    g_tilingParamsVecHandlers = {
        {"sendCounts",
         [](TilingParams &tilingParams, const std::vector<int64_t> valueVec) { tilingParams.sendCounts = valueVec; }},
        {"recvCounts",
         [](TilingParams &tilingParams, const std::vector<int64_t> valueVec) { tilingParams.recvCounts = valueVec; }}};

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
};

TEST_P(GroupedMatMulAlltoAllvArch22TilingTest, ShapeSize)
{
    auto testParam = GetParam();
    struct GroupedMatMulAlltoAllvCompileInfo {};
    GroupedMatMulAlltoAllvCompileInfo compileInfo;
    std::string socVersion = testParam.socVersion;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingData = 8192;
    auto tilingParams = TilingParams{};
    for (auto &kv : testParam.tilingParamsStrPair) {
        if (g_tilingParamsStrHandlers.count(kv.first) != 0) {
            g_tilingParamsStrHandlers[kv.first](tilingParams, kv.second);
        }
    }
    for (auto &kv : testParam.tilingParamsVecPair) {
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
    if (testParam.status == ge::GRAPH_FAILED) {
        Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
    } else {
        Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", tilingParams.epWorldSize}};
        // Match master: conversion failure is returned as the key, not the outer status.
        uint64_t expectTilingKey = testParam.omitCommMode ? static_cast<uint64_t>(ge::GRAPH_FAILED) : 9UL;
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
    }
}

static TestParam g_testParams[] = {
    {"Test_sample", {}, {}, {}, ge::GRAPH_SUCCESS},
    {"Test_910b_aicpu_rejected", {}, {}, {}, ge::GRAPH_FAILED, false, "Ascend910B"},
    {"Test_910b_missing_comm_mode_rejected", {}, {}, {}, ge::GRAPH_FAILED, true, "Ascend910B"},
    {"Test_910b_default_rejected", {}, {}, {}, ge::GRAPH_FAILED, false, "Ascend910B", "default"},
    {"Test_910b_ccu_rejected", {}, {}, {}, ge::GRAPH_FAILED, false, "Ascend910B", "ccu"},
    {"Test_950_aicpu_unchanged", {}, {}, {}, ge::GRAPH_SUCCESS, false, "Ascend950"},
    {"Test_missing_comm_mode_matches_master_a3", {}, {}, {}, ge::GRAPH_SUCCESS, true, "Ascend910_93"},
    // 950 validates commMode in SetHcclTiling, before the final key conversion.
    {"Test_950_missing_comm_mode_rejected", {}, {}, {}, ge::GRAPH_FAILED, true, "Ascend950"},
    {"Test_a3_aicpu_topk8", {{"BS", "512"}}, {}, {}, ge::GRAPH_SUCCESS},
    {"Test_a3_aicpu_topk9",
     {{"BS", "512"}, {"BSK", "4608"}},
     {{"recvCounts", std::vector<int64_t>(32, 144)}},
     {},
     ge::GRAPH_FAILED},
    {"Test_950_aicpu_topk8", {{"BS", "512"}}, {}, {}, ge::GRAPH_SUCCESS, false, "Ascend950"},
    {"Test_950_aicpu_topk9",
     {{"BS", "512"}, {"BSK", "4608"}},
     {{"recvCounts", std::vector<int64_t>(32, 144)}},
     {},
     ge::GRAPH_FAILED,
     false,
     "Ascend950"},
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
                         [](const testing::TestParamInfo<GroupedMatMulAlltoAllvArch22TilingTest::ParamType> &info) {
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
