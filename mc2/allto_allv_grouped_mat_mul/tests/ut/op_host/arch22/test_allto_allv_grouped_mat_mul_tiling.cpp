/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../op_host/op_tiling/allto_allv_grouped_mat_mul_tiling.h"
#include "../../../../op_host/op_tiling/arch22/allto_allv_grouped_mat_mul_aiv_plan.h"
#include "../../../../op_kernel/allto_allv_grouped_mat_mul_aiv_comm.h"
#include "../../../../op_kernel/allto_allv_grouped_mat_mul_aiv_mode.h"
#include "../../../../op_kernel/allto_allv_grouped_mat_mul_catlass.h"

#include <iostream>
#include <gtest/gtest.h>

#include "mc2_tiling_case_executor.h"

using namespace std;

namespace AlltoAllvGroupedMatMulUT {
struct TestParam {
    string testName{};
    std::vector<std::pair<string, string>> tilingParamsStrPair{};
    std::vector<std::pair<string, std::vector<int64_t>>> tilingParamsVecPair{};
    std::vector<std::pair<size_t, ge::DataType>> tilingDTypesPair{};
    ge::graphStatus status;
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
    uint64_t commOut;
    uint64_t aivCoreNum{40};
    uint64_t aicCoreNum{20};
    uint64_t totalUbSize{196608};
    uint64_t gmmWeightDim1{7168};
    uint64_t gmmYDim1{4096};
    uint64_t mmWeightDim0{7168};
    bool transGmmWeight{false};
    bool transMmWeight{false};
    bool permuteOutFlag{false};
    bool isNeedMM{true};
    std::string group{"group"};
    std::vector<int64_t> sendCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    std::vector<int64_t> recvCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
};

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
        {"gmmYDim1",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.gmmYDim1 = std::stoi(valueStr); }},
        {"mmWeightDim0",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.mmWeightDim0 = std::stoi(valueStr); }},
        {"transGmmWeight",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.transGmmWeight = valueStr == "true"; }},
        {"transMmWeight",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.transMmWeight = valueStr == "true"; }},
        {"permuteOutFlag",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.permuteOutFlag = valueStr == "true"; }},
        {"isNeedMM",
         [](TilingParams &tilingParams, const string &valueStr) { tilingParams.isNeedMM = valueStr == "true"; }}};

std::unordered_map<string, std::function<void(TilingParams &tilingParams, const std::vector<int64_t> valueVec)>>
    g_tilingParamsVecHandlers = {
        {"sendCounts",
         [](TilingParams &tilingParams, const std::vector<int64_t> valueVec) { tilingParams.sendCounts = valueVec; }},
        {"recvCounts",
         [](TilingParams &tilingParams, const std::vector<int64_t> valueVec) { tilingParams.recvCounts = valueVec; }}};

bool has_any_target_key(const std::vector<std::pair<std::string, std::string>> &params,
                        const std::vector<std::string> &targets)
{
    return std::any_of(params.begin(), params.end(), [&targets](const auto &p) {
        return std::find(targets.begin(), targets.end(), p.first) != targets.end();
    });
}

// 提取：初始化 tilingParams
void InitializeTilingParams(const TestParam &testParam, TilingParams &tilingParams)
{
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
}

std::unique_ptr<gert::TilingContextPara::TensorDescription> CreateTensorShape(gert::StorageShape shape,
                                                                              ge::DataType dtype, ge::Format format)
{
    return std::unique_ptr<gert::TilingContextPara::TensorDescription>(
        new gert::TilingContextPara::TensorDescription(shape, // 这里用大括号构造
                                                       dtype, format));
}

std::vector<gert::TilingContextPara::TensorDescription> CreateInputTensors(
    const TilingParams &tilingParams, const std::unique_ptr<gert::TilingContextPara::TensorDescription> &mmXShape,
    const std::unique_ptr<gert::TilingContextPara::TensorDescription> &mmWeightShape)
{
    return {
        {{{tilingParams.BSK, tilingParams.H1}, {tilingParams.BSK, tilingParams.H1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{tilingParams.e, tilingParams.gmmWeightDim1, tilingParams.N1},
          {tilingParams.e, tilingParams.gmmWeightDim1, tilingParams.N1}},
         ge::DT_FLOAT16,
         ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        {{}, ge::DT_INT32, ge::FORMAT_ND},
        *mmXShape,
        *mmWeightShape,
    };
}

std::vector<gert::TilingContextPara::TensorDescription> CreateOutputTensors(
    const TilingParams &tilingParams, const std::unique_ptr<gert::TilingContextPara::TensorDescription> &mmYShape)
{
    auto mmYDesc = (mmYShape->shape_.GetStorageShape().GetDimNum() == 0) ?
                       gert::TilingContextPara::TensorDescription{
                           {{tilingParams.BS, tilingParams.N2}, {tilingParams.BS, tilingParams.N2}},
                           ge::DT_FLOAT16,
                           ge::FORMAT_ND} :
                       *mmYShape;
    return {
        {{{tilingParams.A, tilingParams.gmmYDim1}, {tilingParams.A, tilingParams.gmmYDim1}},
         ge::DT_FLOAT16,
         ge::FORMAT_ND},
        mmYDesc,
        {{{tilingParams.A, tilingParams.H1}, {tilingParams.A, tilingParams.H1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    };
}

std::vector<std::pair<std::string, Ops::Transformer::AnyValue>> CreateAttrs(const TestParam &testParam,
                                                                            const TilingParams &tilingParams)
{
    return {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tilingParams.group)},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.epWorldSize)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tilingParams.sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tilingParams.recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParams.permuteOutFlag)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}};
}

class AlltoAllvGroupedMatMulArch22TilingTest : public testing::TestWithParam<TestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllvGroupedMatMulArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllvGroupedMatMulArch22TilingTest TearDown" << std::endl;
    }

public:
    std::vector<int64_t> sendCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    std::vector<int64_t> recvCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
};

TEST_P(AlltoAllvGroupedMatMulArch22TilingTest, ShapeSize)
{
    auto testParam = GetParam();

    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;

    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    TilingParams tilingParams;
    InitializeTilingParams(testParam, tilingParams);

    std::vector<std::string> targets = {"BS", "H2", "mmWeightDim0", "N2"};

    auto mmXShape = CreateTensorShape({{tilingParams.BS, tilingParams.H2}, {tilingParams.BS, tilingParams.H2}},
                                      ge::DT_FLOAT16, ge::FORMAT_ND);
    auto mmWeightShape =
        CreateTensorShape({{tilingParams.mmWeightDim0, tilingParams.N2}, {tilingParams.mmWeightDim0, tilingParams.N2}},
                          ge::DT_FLOAT16, ge::FORMAT_ND);
    auto mmYShape = CreateTensorShape({{tilingParams.BS, tilingParams.N2}, {tilingParams.BS, tilingParams.N2}},
                                      ge::DT_FLOAT16, ge::FORMAT_ND);

    if (!(has_any_target_key(testParam.tilingParamsStrPair, targets) || tilingParams.isNeedMM == false)) {
        mmXShape->shape_ = {};
        mmWeightShape->shape_ = {};
    }

    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul", CreateInputTensors(tilingParams, mmXShape, mmWeightShape),
        CreateOutputTensors(tilingParams, mmYShape),
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tilingParams.group)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.epWorldSize)},
         {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tilingParams.sendCounts)},
         {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tilingParams.recvCounts)},
         {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParams.permuteOutFlag)},
         {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}},
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    if (testParam.status == ge::GRAPH_FAILED) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
    } else {
        uint64_t expectTilingKey = 4UL;
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
    }
}

static TestParam testParams[] = {
    {"Test_gmmWeight_size", {{"e", "64"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_ep_world_size", {{"epWorldSize", "4"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_e", {{"e", "64"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_e_multi_ep",
     {{"epWorldSize", "16"}, {"e", "32"}, {"permuteOutFlag", "true"}},
     {{"sendCounts", std::vector<int64_t>(512, 128)}, {"recvCounts", std::vector<int64_t>(512, 128)}},
     {},
     ge::GRAPH_FAILED},
    {"Test_send_counts_size",
     {{"epWorldSize", "16"}, {"e", "32"}, {"permuteOutFlag", "true"}},
     {},
     {},
     ge::GRAPH_FAILED},
    {"Test_BSK_1", {{"BSK", "52428800"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_BS_1", {{"BS", "52428800"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H1", {{"H1", "65536"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H2", {{"H2", "12289"}, {"mmWeightDim0", "12289"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_N1", {{"N1", "65536"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_N2", {{"N2", "65536"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H_1", {{"H1", "7168"}, {"gmmWeightDim1", "7169"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H_3", {{"H2", "7168"}, {"mmWeightDim0", "7169"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H_4", {{"H1", "65536"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_send_counts_0",
     {{"BSK", "16386"}, {"BS", "2048"}, {"permuteOutFlag", "true"}},
     {{"sendCounts",
       std::vector<int64_t>{
           3201, 3201, 3200, 3200, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
           128,  128,  128,  128,  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
       }}},
     {},
     ge::GRAPH_FAILED},
    {"Test_recv_counts_0",
     {{"A", "16386"}, {"BS", "8193"}, {"permuteOutFlag", "true"}},
     {{"recvCounts",
       std::vector<int64_t>{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,
                            128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 3200}}},
     {},
     ge::GRAPH_FAILED},
    {"Test_recv_counts_1",
     {{"A", "16386"}, {"BS", "8193"}, {"permuteOutFlag", "true"}},
     {{"recvCounts",
       std::vector<int64_t>{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,  128, 128,
                            128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 1600, 1600}}},
     {},
     ge::GRAPH_FAILED},
    {"Test_no_MM", {{"permuteOutFlag", "true"}, {"isNeedMM", "false"}}, {}, {}, ge::GRAPH_SUCCESS}};

INSTANTIATE_TEST_SUITE_P(AlltoAllvGroupedMatMul, AlltoAllvGroupedMatMulArch22TilingTest, testing::ValuesIn(testParams),
                         [](const testing::TestParamInfo<AlltoAllvGroupedMatMulArch22TilingTest::ParamType> &info) {
                             return info.param.testName;
                         });

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, H4)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},

        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7169}, {4096, 7169}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, A1)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},

        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4097, 7168}, {4097, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, BS1)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 64}, {7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2047, 64}, {2047, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, Dim1)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{2, 4096, 7168}, {2, 4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},

        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, Dim2)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 4096}, {7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},

        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, Dim3)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 64}, {7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4097}, {4096, 4097}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, Dim5)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 2048, 7168}, {2, 2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 64}, {7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2047, 64}, {2047, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, Dim6)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 7168, 64}, {2, 7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2047, 64}, {2047, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, Dim7)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 7168, 64}, {2, 7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2047, 64}, {2047, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, Dim10)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},

        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// 无 MM 时 BSK 触顶（permuteOutFlag=false，避免 K 校验抢先失败）
TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, CheckShapeSizeBskMaxNoMm)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{52428800, 7168}, {52428800, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, PermuteOutGmmXH1Mismatch)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 64}, {7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7169}, {4096, 7169}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, MmYAndMmWeightN2Mismatch)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 64}, {7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 63}, {2048, 63}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, CheckDTypeMmMismatch)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 64}, {7168, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// gmmX/gmmWeight 非法 dtype（无 MM，触达 CheckDType gmmX 分支）
TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, CheckDTypeGmmXInvalidNoMm)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// mmX/mmWeight 同为非法 dtype（非 FP16/BF16），触达 CheckDType mmX 分支
TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, CheckDTypeMmXInvalidDtype)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{7168, 64}, {7168, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// gmmX 与 mmX dtype 不一致（均为合法 FP16/BF16）
TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, CheckDTypeGmmXMmXDtypeMismatch)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{7168, 64}, {7168, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// 无 MM、permuteOutFlag=false，BSK=0 触达 CheckShapeSize BSK 下界
TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, CheckShapeSizeBskZeroNoMm)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{0, 7168}, {0, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// 带 MM 的完整 shape 路径，用于触达 CheckShapeSize / CheckShapeRelation 深分支
TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, CheckShapeSizeBskMaxWithMm)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{52428800, 7168}, {52428800, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 64}, {7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, CheckShapeSizeH1MaxWithMm)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 65536}, {4096, 65536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 65536, 4096}, {4, 65536, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 65536}, {2048, 65536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{65536, 64}, {65536, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 65536}, {4096, 65536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, PermuteOutFlagFalseWithMmOutput)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 64}, {7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, PermuteOutGmmYAMismatch)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 64}, {7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4097, 7168}, {4097, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, CheckShapeRelationKOutOfRange)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 64}, {7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, CheckDTypeGmmMismatch)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 64}, {7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, CheckAttrsNegativeSendCount)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    std::vector<int64_t> sendCountsNeg = sendCounts;
    sendCountsNeg[0] = -1;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 64}, {7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCountsNeg)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// CheckAttrsShapeSize：sendCounts / recvCounts 长度不一致
TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, SendRecvCountsSizeMismatch)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    std::vector<int64_t> recvCountsShort(recvCounts.begin(), recvCounts.begin() + 31);
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7168, 64}, {7168, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCountsShort)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, TransMmWeight1)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 7168, 4096}, {4, 7168, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 7169}, {64, 7169}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2048, 64}, {2048, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4096, 7168}, {4096, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}
} // namespace AlltoAllvGroupedMatMulUT
namespace AlltoAllvGroupedMatMulUT {
TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, AivBf16TilingContract)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    const std::vector<int64_t> counts(8, 2);
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{16, 256}, {16, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 256, 128}, {2, 256, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{16, 128}, {16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(counts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(counts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aiv")},
        },
        &compileInfo, "Ascend910_93", 20, 196608, 16U * 1024U);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
    TilingInfo tilingInfo;
    const bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();

    ASSERT_TRUE(success);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(AlltoAllvGmmAivTilingData));
    const auto *tiling = reinterpret_cast<const AlltoAllvGmmAivTilingData *>(tilingInfo.tilingData.get());
    EXPECT_EQ(tiling->gmmInfo.rankSize, 4U);
    EXPECT_EQ(tiling->gmmInfo.expertPerRank, 2U);
    EXPECT_EQ(tiling->gmmInfo.K, 256U);
    EXPECT_EQ(tiling->gmmInfo.N, 128U);
    EXPECT_EQ(tiling->gmmCocTiling.ubMoveNum % 16, 0);
    EXPECT_GT(tilingInfo.blockNum, 0U);
    EXPECT_EQ(tiling->is910C, 1U);
    EXPECT_EQ(tilingInfo.tilingKey, 8);
}
} // namespace AlltoAllvGroupedMatMulUT

namespace AlltoAllvGroupedMatMulUT {
namespace {
constexpr uint64_t AIV_TILING_DATA_SIZE = 16U * 1024U;

bool ExecuteAivHostCase(ge::DataType dtype, int64_t rankSize, const std::vector<int64_t> &sendCounts,
                        const std::vector<int64_t> &recvCounts, bool transpose, bool withSharedMm, bool withPermute,
                        TilingInfo &tilingInfo, const std::string &socVersion = "Ascend910_93", uint64_t coreNum = 20,
                        int64_t routedTokens = 16, int64_t hiddenSize = 256, int64_t outputSize = 128,
                        int64_t localExpertNum = 2)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    const int64_t weightDim1 = transpose ? outputSize : hiddenSize;
    const int64_t weightDim2 = transpose ? hiddenSize : outputSize;

    auto mmX = CreateTensorShape({{4, hiddenSize}, {4, hiddenSize}}, dtype, ge::FORMAT_ND);
    auto mmWeight = CreateTensorShape(transpose ? gert::StorageShape{{64, hiddenSize}, {64, hiddenSize}} :
                                                  gert::StorageShape{{hiddenSize, 64}, {hiddenSize, 64}},
                                      dtype, ge::FORMAT_ND);
    auto mmY = CreateTensorShape({{4, 64}, {4, 64}}, dtype, ge::FORMAT_ND);
    auto permuteOut = CreateTensorShape({{routedTokens, hiddenSize}, {routedTokens, hiddenSize}}, dtype, ge::FORMAT_ND);
    if (!withSharedMm) {
        mmX->shape_ = {};
        mmWeight->shape_ = {};
        mmY->shape_ = {};
    }
    if (!withPermute) {
        permuteOut->shape_ = {};
    }

    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{routedTokens, hiddenSize}, {routedTokens, hiddenSize}}, dtype, ge::FORMAT_ND},
            {{{localExpertNum, weightDim1, weightDim2}, {localExpertNum, weightDim1, weightDim2}},
             dtype,
             ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            *mmX,
            *mmWeight,
        },
        {
            {{{routedTokens, outputSize}, {routedTokens, outputSize}}, dtype, ge::FORMAT_ND},
            *mmY,
            *permuteOut,
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(rankSize)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(transpose)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(transpose)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(withPermute)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aiv")},
        },
        &compileInfo, socVersion, coreNum, 196608, AIV_TILING_DATA_SIZE);

    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues({{"rankNum", rankSize}});
    const bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    return success;
}

std::vector<int64_t> MakeAivCounts(int64_t rankSize)
{
    std::vector<int64_t> counts(static_cast<size_t>(rankSize) * 2U, 0);
    for (size_t token = 0; token < 16U; ++token) {
        ++counts[token % counts.size()];
    }
    return counts;
}
} // namespace

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, A3AivRankContracts)
{
    for (const int64_t rankSize : {2, 4, 8, 16, 32, 64, 128}) {
        const std::vector<int64_t> counts = MakeAivCounts(rankSize);
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteAivHostCase(ge::DT_BF16, rankSize, counts, counts, false, false, false, tilingInfo,
                                       "Ascend910_93", 20));
        const auto *tiling = reinterpret_cast<const AlltoAllvGmmAivTilingData *>(tilingInfo.tilingData.get());
        EXPECT_EQ(tiling->gmmInfo.rankSize, static_cast<uint32_t>(rankSize));
        EXPECT_EQ(tiling->countNum, static_cast<uint32_t>(rankSize * 2));
        EXPECT_EQ(tilingInfo.tilingKey, 8);
    }
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, A2AivSupportsEp2Ep4Ep8)
{
    for (const int64_t rankSize : {2, 4, 8}) {
        const std::vector<int64_t> counts = MakeAivCounts(rankSize);
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteAivHostCase(ge::DT_BF16, rankSize, counts, counts, false, false, false, tilingInfo,
                                       "Ascend910B", 24));
    }
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, A2AivRejectsEp16AndAbove)
{
    for (const int64_t rankSize : {16, 32, 64, 128}) {
        const std::vector<int64_t> counts = MakeAivCounts(rankSize);
        TilingInfo tilingInfo;
        EXPECT_FALSE(ExecuteAivHostCase(ge::DT_BF16, rankSize, counts, counts, false, false, false, tilingInfo,
                                        "Ascend910B", 24));
    }
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, AivMarksA2AndA3CommunicationContexts)
{
    const std::vector<int64_t> counts(8, 2);

    TilingInfo a2TilingInfo;
    ASSERT_TRUE(
        ExecuteAivHostCase(ge::DT_BF16, 4, counts, counts, false, false, false, a2TilingInfo, "Ascend910B", 24));
    ASSERT_GE(a2TilingInfo.tilingDataSize, sizeof(AlltoAllvGmmAivTilingData));
    const auto *a2Tiling = reinterpret_cast<const AlltoAllvGmmAivTilingData *>(a2TilingInfo.tilingData.get());
    EXPECT_EQ(a2Tiling->is910C, 0U);

    TilingInfo a3TilingInfo;
    ASSERT_TRUE(
        ExecuteAivHostCase(ge::DT_BF16, 4, counts, counts, false, false, false, a3TilingInfo, "Ascend910_93", 20));
    ASSERT_GE(a3TilingInfo.tilingDataSize, sizeof(AlltoAllvGmmAivTilingData));
    const auto *a3Tiling = reinterpret_cast<const AlltoAllvGmmAivTilingData *>(a3TilingInfo.tilingData.get());
    EXPECT_EQ(a3Tiling->is910C, 1U);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, AivWritesAutomaticOverlapDecisionToTiling)
{
    const std::vector<int64_t> belowThresholdCounts = {8, 8, 8, 8, 8, 8, 8, 7};

    TilingInfo offTilingInfo;
    ASSERT_TRUE(ExecuteAivHostCase(ge::DT_BF16, 4, belowThresholdCounts, belowThresholdCounts, false, false, false,
                                   offTilingInfo, "Ascend910_93", 20, 63));
    const auto *offTiling = reinterpret_cast<const AlltoAllvGmmAivTilingData *>(offTilingInfo.tilingData.get());
    EXPECT_EQ(offTiling->expertOverlapMode, A2AVGMM_EXPERT_OVERLAP_DISABLED);

    const std::vector<int64_t> thresholdCounts(8, 8);
    TilingInfo onTilingInfo;
    ASSERT_TRUE(ExecuteAivHostCase(ge::DT_BF16, 4, thresholdCounts, thresholdCounts, false, false, false, onTilingInfo,
                                   "Ascend910_93", 20, 64));
    const auto *onTiling = reinterpret_cast<const AlltoAllvGmmAivTilingData *>(onTilingInfo.tilingData.get());
    EXPECT_EQ(onTiling->expertOverlapMode, A2AVGMM_EXPERT_OVERLAP_ENABLED);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, AivTransposeSharedMmAndPermuteContract)
{
    const std::vector<int64_t> counts(8, 2);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteAivHostCase(ge::DT_BF16, 4, counts, counts, true, true, true, tilingInfo));
    const auto *tiling = reinterpret_cast<const AlltoAllvGmmAivTilingData *>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingInfo.tilingKey, 11);
    EXPECT_EQ(tiling->gmmInfo.isTransposeB, 1U);
    EXPECT_EQ(tiling->gmmInfo.hasSharedExpert, 1U);
    EXPECT_EQ(tiling->gmmInfo.hasPermuteOut, 1U);
    EXPECT_EQ(tiling->mmInfo.M, 4U);
    EXPECT_EQ(tiling->mmInfo.K, 256U);
    EXPECT_EQ(tiling->mmInfo.N, 64U);
    EXPECT_TRUE(AlltoAllvGroupedMatMulCatlass::IsSupportedTile(tiling->gmmCocTiling.m0, tiling->gmmCocTiling.k0,
                                                               tiling->gmmCocTiling.n0));
    EXPECT_TRUE(AlltoAllvGroupedMatMulCatlass::IsSupportedTile(tiling->mmCocTiling.m0, tiling->mmCocTiling.k0,
                                                               tiling->mmCocTiling.n0));
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, AivSupportsFp16)
{
    const std::vector<int64_t> counts(8, 2);
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteAivHostCase(ge::DT_FLOAT16, 4, counts, counts, false, false, false, tilingInfo));
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, AivRejectsCountLengthMismatch)
{
    const std::vector<int64_t> invalidCounts(7, 2);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteAivHostCase(ge::DT_BF16, 4, invalidCounts, invalidCounts, false, false, false, tilingInfo));
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, AivRejectsNegativeCount)
{
    std::vector<int64_t> sendCounts(8, 2);
    const std::vector<int64_t> recvCounts(8, 2);
    sendCounts[0] = -1;
    sendCounts[1] = 5;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteAivHostCase(ge::DT_BF16, 4, sendCounts, recvCounts, false, false, false, tilingInfo));
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, AivRejectsCountSumMismatch)
{
    std::vector<int64_t> sendCounts(8, 2);
    const std::vector<int64_t> recvCounts(8, 2);
    sendCounts[0] = 1;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteAivHostCase(ge::DT_BF16, 4, sendCounts, recvCounts, false, false, false, tilingInfo));
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, AivRejectsDimensionConstraints)
{
    const std::vector<int64_t> counts(8, 2);
    TilingInfo tilingInfo;

    EXPECT_FALSE(ExecuteAivHostCase(ge::DT_BF16, 4, counts, counts, false, false, false, tilingInfo, "Ascend910_93", 20,
                                    5000001));
    EXPECT_FALSE(ExecuteAivHostCase(ge::DT_BF16, 4, counts, counts, false, false, false, tilingInfo, "Ascend910_93", 20,
                                    16, 65536, 128));
    EXPECT_FALSE(ExecuteAivHostCase(ge::DT_BF16, 4, counts, counts, false, false, false, tilingInfo, "Ascend910_93", 20,
                                    16, 256, 65536));
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, AivRejectsTooManyLocalExperts)
{
    std::vector<int64_t> counts(4U * 513U, 0);
    for (uint32_t index = 0U; index < 16U; ++index) {
        counts[index] = 1;
    }
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteAivHostCase(ge::DT_BF16, 4, counts, counts, false, false, false, tilingInfo, "Ascend910_93", 20,
                                    16, 256, 128, 513));
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, AivRejectsCountEntryAboveTokenCapacity)
{
    std::vector<int64_t> sendCounts(8, 0);
    const std::vector<int64_t> recvCounts(8, 2);
    sendCounts[0] = 17;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteAivHostCase(ge::DT_BF16, 4, sendCounts, recvCounts, false, false, false, tilingInfo));
}
} // namespace AlltoAllvGroupedMatMulUT

namespace AlltoAllvGroupedMatMulUT {
namespace AivPlan = AlltoAllvGroupedMatMulAivPlan;

TEST(AlltoAllvGroupedMatMulAivPlanTest, BuildsRankMajorSendAndExpertMajorRecvPrefixes)
{
    const int64_t send[] = {2, 1, 3, 0};
    const int64_t recv[] = {2, 1, 1, 2};
    int32_t sendPrefix[A2AVGMM_MAX_COUNT_NUM] = {};
    int32_t recvPrefix[A2AVGMM_MAX_COUNT_NUM] = {};

    ASSERT_TRUE(AivPlan::BuildInclusivePrefixes(send, recv, 2U, 2U, 6U, 6U, sendPrefix, recvPrefix));
    EXPECT_EQ(std::vector<int32_t>(sendPrefix, sendPrefix + 4), (std::vector<int32_t>{2, 3, 6, 6}));
    EXPECT_EQ(std::vector<int32_t>(recvPrefix, recvPrefix + 4), (std::vector<int32_t>{2, 3, 4, 6}));
}

TEST(AlltoAllvGroupedMatMulAivPlanTest, RejectsInvalidPrefixInputs)
{
    const int64_t valid[] = {2, 1, 1, 2};
    const int64_t negative[] = {-1, 1, 1, 2};
    const int64_t wrongSum[] = {2, 1, 1, 1};
    int32_t sendPrefix[A2AVGMM_MAX_COUNT_NUM] = {};
    int32_t recvPrefix[A2AVGMM_MAX_COUNT_NUM] = {};

    EXPECT_FALSE(AivPlan::BuildInclusivePrefixes(negative, valid, 2U, 2U, 6U, 6U, sendPrefix, recvPrefix));
    EXPECT_FALSE(AivPlan::BuildInclusivePrefixes(wrongSum, valid, 2U, 2U, 6U, 6U, sendPrefix, recvPrefix));
    EXPECT_FALSE(AivPlan::BuildInclusivePrefixes(nullptr, valid, 2U, 2U, 6U, 6U, sendPrefix, recvPrefix));
    EXPECT_FALSE(AivPlan::BuildInclusivePrefixes(valid, nullptr, 2U, 2U, 6U, 6U, sendPrefix, recvPrefix));
    EXPECT_FALSE(AivPlan::BuildInclusivePrefixes(valid, valid, 0U, 2U, 6U, 6U, sendPrefix, recvPrefix));
    EXPECT_FALSE(AivPlan::BuildInclusivePrefixes(valid, valid, 2U, 0U, 6U, 6U, sendPrefix, recvPrefix));
    EXPECT_FALSE(AivPlan::BuildInclusivePrefixes(valid, valid, 2U, 2U, 0U, 6U, sendPrefix, recvPrefix));
    EXPECT_FALSE(AivPlan::BuildInclusivePrefixes(valid, valid, 2U, 2U, 6U, 0U, sendPrefix, recvPrefix));
}

TEST(AlltoAllvGroupedMatMulAivPlanTest, RejectsCountNumBeyondAbiLimit)
{
    std::vector<int64_t> counts1025(1025U, 0);
    std::vector<int32_t> prefix1025(1025U, 0);
    counts1025[0] = 1;
    EXPECT_FALSE(AivPlan::BuildInclusivePrefixes(counts1025.data(), counts1025.data(), 5U, 205U, 1U, 1U,
                                                 prefix1025.data(), prefix1025.data()));
}

TEST(AlltoAllvGroupedMatMulAivPlanTest, PrefixOnlyTilingAbiFitsExpandedBuffer)
{
    AlltoAllvGmmAivTilingData tiling = {};
    EXPECT_EQ(A2AVGMM_MAX_COUNT_NUM, 1024U);
    EXPECT_EQ(sizeof(tiling.sendPrefix), 1024U * sizeof(int32_t));
    EXPECT_EQ(sizeof(tiling.recvPrefix), 1024U * sizeof(int32_t));
    EXPECT_EQ(sizeof(tiling.sendPrefix) + sizeof(tiling.recvPrefix), 8192U);
    EXPECT_GT(sizeof(AlltoAllvGmmAivTilingData), 8192U);
    EXPECT_LE(sizeof(AlltoAllvGmmAivTilingData), AIV_TILING_DATA_SIZE);
}

TEST_F(AlltoAllvGroupedMatMulArch22TilingTest, AivPublishesInt32PrefixesFor1024GlobalExperts)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    const std::vector<int64_t> counts(1024U, 1);
    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        {
            {{{1024, 256}, {1024, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8, 256, 128}, {8, 256, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1024, 128}, {1024, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
            {{}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(counts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(counts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aiv")},
        },
        &compileInfo, "Ascend910_93", 20, 196608, AIV_TILING_DATA_SIZE);

    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues({{"rankNum", 128}});
    TilingInfo tilingInfo;
    const bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();

    ASSERT_TRUE(success);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(AlltoAllvGmmAivTilingData));
    const auto *tiling = reinterpret_cast<const AlltoAllvGmmAivTilingData *>(tilingInfo.tilingData.get());
    EXPECT_EQ(tiling->countNum, 1024U);
    EXPECT_EQ(tiling->gmmInfo.rankSize, 128U);
    EXPECT_EQ(tiling->gmmInfo.expertPerRank, 8U);
    EXPECT_EQ(tiling->sendPrefix[0], 1);
    EXPECT_EQ(tiling->sendPrefix[1023], 1024);
    EXPECT_EQ(tiling->recvPrefix[0], 1);
    EXPECT_EQ(tiling->recvPrefix[1023], 1024);
    EXPECT_EQ(tiling->countBytes, 4096U);
}
} // namespace AlltoAllvGroupedMatMulUT

namespace AlltoAllvGroupedMatMulUT {
namespace AivComm = AlltoAllvGroupedMatMulAiv;

struct CountingPrefix {
    const int32_t *values = nullptr;
    uint32_t *readCount = nullptr;

    bool operator==(std::nullptr_t) const
    {
        return values == nullptr;
    }

    int32_t operator[](uint32_t index) const
    {
        ++(*readCount);
        return values[index];
    }
};

TEST(AlltoAllvGroupedMatMulAivCommTest, BuildsExpertMajorSourceMinorMetadata)
{
    const int32_t recvPrefix[] = {2, 3, 4, 6};
    AivComm::ExpertMeta expertMeta[2] = {};
    AivComm::ExpertSourceMeta sourceMeta[4] = {};

    ASSERT_TRUE(AivComm::BuildExpertMetadata(recvPrefix, 2, 2, 6, expertMeta, sourceMeta));
    EXPECT_EQ(expertMeta[0].recvTokenBase, 0U);
    EXPECT_EQ(expertMeta[0].tokenCount, 3U);
    EXPECT_EQ(expertMeta[1].recvTokenBase, 3U);
    EXPECT_EQ(expertMeta[1].tokenCount, 3U);

    EXPECT_EQ(sourceMeta[0].dstTokenOffset, 0U);
    EXPECT_EQ(sourceMeta[0].tokenCount, 2U);
    EXPECT_EQ(sourceMeta[1].dstTokenOffset, 2U);
    EXPECT_EQ(sourceMeta[1].tokenCount, 1U);
    EXPECT_EQ(sourceMeta[2].dstTokenOffset, 3U);
    EXPECT_EQ(sourceMeta[2].tokenCount, 1U);
    EXPECT_EQ(sourceMeta[3].dstTokenOffset, 4U);
    EXPECT_EQ(sourceMeta[3].tokenCount, 2U);
}

TEST(AlltoAllvGroupedMatMulAivCommTest, SupportsZeroTokenExpert)
{
    const int32_t recvPrefix[] = {2, 3, 3, 3};
    AivComm::ExpertMeta expertMeta[2] = {};
    AivComm::ExpertSourceMeta sourceMeta[4] = {};

    ASSERT_TRUE(AivComm::BuildExpertMetadata(recvPrefix, 2, 2, 3, expertMeta, sourceMeta));
    EXPECT_EQ(expertMeta[0].recvTokenBase, 0U);
    EXPECT_EQ(expertMeta[0].tokenCount, 3U);
    EXPECT_EQ(expertMeta[1].recvTokenBase, 3U);
    EXPECT_EQ(expertMeta[1].tokenCount, 0U);
    EXPECT_EQ(sourceMeta[2].dstTokenOffset, 3U);
    EXPECT_EQ(sourceMeta[3].dstTokenOffset, 3U);
}

TEST(AlltoAllvGroupedMatMulAivCommTest, ComputesPeerSourceOffsets)
{
    const int32_t peer0SendPrefix[] = {2, 3, 6, 6};
    const int32_t peer1SendPrefix[] = {1, 3, 3, 7};
    uint64_t offset = 0;

    ASSERT_TRUE(AivComm::GetPeerSourceTokenOffset(peer0SendPrefix, 2, 2, 0, 0, offset));
    EXPECT_EQ(offset, 0U);
    ASSERT_TRUE(AivComm::GetPeerSourceTokenOffset(peer0SendPrefix, 2, 2, 0, 1, offset));
    EXPECT_EQ(offset, 2U);
    ASSERT_TRUE(AivComm::GetPeerSourceTokenOffset(peer0SendPrefix, 2, 2, 1, 0, offset));
    EXPECT_EQ(offset, 3U);
    ASSERT_TRUE(AivComm::GetPeerSourceTokenOffset(peer1SendPrefix, 2, 2, 1, 1, offset));
    EXPECT_EQ(offset, 3U);

    const int32_t invalidPeerPrefix[] = {-1, 2, 2, 6};
    EXPECT_FALSE(AivComm::GetPeerSourceTokenOffset(invalidPeerPrefix, 2, 2, 0, 0, offset));
}

TEST(AlltoAllvGroupedMatMulAivCommTest, ReadsInclusivePrefixRangeInConstantTime)
{
    std::vector<int32_t> prefix(A2AVGMM_MAX_COUNT_NUM);
    for (uint32_t index = 0U; index < A2AVGMM_MAX_COUNT_NUM; ++index) {
        prefix[index] = static_cast<int32_t>(index + 1U);
    }
    uint32_t readCount = 0U;
    const CountingPrefix countingPrefix{prefix.data(), &readCount};
    uint32_t begin = 0U;
    uint32_t end = 0U;

    ASSERT_TRUE(AivComm::PrefixRange(countingPrefix, A2AVGMM_MAX_COUNT_NUM, A2AVGMM_MAX_COUNT_NUM - 1U, begin, end));
    EXPECT_EQ(begin, A2AVGMM_MAX_COUNT_NUM - 1U);
    EXPECT_EQ(end, A2AVGMM_MAX_COUNT_NUM);
    EXPECT_LE(readCount, 2U);
}

TEST(AlltoAllvGroupedMatMulAivCommTest, ValidatesWindowAndReadyLayout)
{
    AivComm::A2avWindowLayout layout = {};
    AivComm::RuntimeControlLayout control = {};
    ASSERT_TRUE(AivComm::BuildRuntimeControlLayout(128U, 2U, control));
    ASSERT_TRUE(AivComm::BuildWindowLayout(6, 256, 4, control, 32U * 1024U * 1024U, layout));
    EXPECT_EQ(layout.inputBytes, 3072U);
    EXPECT_EQ(layout.controlOffset % AivComm::kWindowAlignment, 0U);
    EXPECT_GE(layout.countsOffset, layout.inputBytes);
    EXPECT_EQ(layout.readyOffset, control.firstExpertFlagOffset);
    EXPECT_FALSE(AivComm::BuildWindowLayout(6, 256, 4, control, layout.totalBytes - 1U, layout));
    EXPECT_FALSE(AivComm::BuildWindowLayout(UINT64_MAX, 256, 4, control, UINT64_MAX, layout));
    EXPECT_EQ(AivComm::ExpertReadyBytes(32), 1024U);
    EXPECT_EQ(AivComm::ExpertReadyOffset(4096, 3), 4192U);
}

TEST(AlltoAllvGroupedMatMulAivCommTest, KeepsPeerControlOffsetStableAcrossExpertCounts)
{
    constexpr uint64_t windowBytes = 200U * 1024U * 1024U;
    AivComm::RuntimeControlLayout controlE2 = {};
    AivComm::RuntimeControlLayout controlE16 = {};
    AivComm::A2avWindowLayout layoutE2 = {};
    AivComm::A2avWindowLayout layoutE16 = {};

    ASSERT_TRUE(AivComm::BuildRuntimeControlLayout(8U, 2U, controlE2));
    ASSERT_TRUE(AivComm::BuildRuntimeControlLayout(8U, 16U, controlE16));
    ASSERT_TRUE(AivComm::BuildWindowLayout(32U, 256U, 16U, controlE2, windowBytes, layoutE2));
    ASSERT_TRUE(AivComm::BuildWindowLayout(32U, 256U, 128U, controlE16, windowBytes, layoutE16));

    EXPECT_EQ(controlE2.releaseSlotsOffset, controlE16.releaseSlotsOffset);
    EXPECT_EQ(controlE2.peerControlBytes, 544U);
    EXPECT_EQ(controlE2.peerControlBytes, controlE16.peerControlBytes);
    EXPECT_NE(controlE2.totalBytes, controlE16.totalBytes);
    EXPECT_EQ(layoutE2.controlOffset, layoutE16.controlOffset);
}
} // namespace AlltoAllvGroupedMatMulUT

namespace AlltoAllvGroupedMatMulUT {
namespace AivCatlass = AlltoAllvGroupedMatMulCatlass;

TEST(AlltoAllvGroupedMatMulCatlassTest, BuildsZeroMExpertSpecWithoutWork)
{
    const AivComm::ExpertMeta expert{7U, 0U, 0U};
    AivCatlass::GemmLaunchSpec spec = {};
    ASSERT_TRUE(AivCatlass::BuildExpertGemmSpec<false>(3, expert, 272, 130, spec));
    EXPECT_FALSE(spec.hasWork);
    EXPECT_EQ(spec.offsetA, 7U * 272U);
    EXPECT_EQ(spec.offsetB, 3U * 272U * 130U);
    EXPECT_EQ(spec.offsetC, 7U * 130U);
}

TEST(AlltoAllvGroupedMatMulCatlassTest, BuildsTailExpertSpec)
{
    const AivComm::ExpertMeta expert{5U, 17U, 0U};
    AivCatlass::GemmLaunchSpec spec = {};
    ASSERT_TRUE(AivCatlass::BuildExpertGemmSpec<false>(2, expert, 272, 130, spec));
    EXPECT_TRUE(spec.hasWork);
    EXPECT_EQ(spec.m, 17U);
    EXPECT_EQ(spec.k, 272U);
    EXPECT_EQ(spec.n, 130U);
    EXPECT_EQ(spec.lda, 272U);
    EXPECT_EQ(spec.ldb, 130U);
    EXPECT_EQ(spec.ldc, 130U);
    EXPECT_FALSE(spec.transposeB);
}

TEST(AlltoAllvGroupedMatMulCatlassTest, BuildsTransposedBAndSharedSpecs)
{
    const AivComm::ExpertMeta expert{0U, 1U, 0U};
    AivCatlass::GemmLaunchSpec expertSpec = {};
    ASSERT_TRUE(AivCatlass::BuildExpertGemmSpec<true>(1, expert, 272, 130, expertSpec));
    EXPECT_TRUE(expertSpec.transposeB);
    EXPECT_EQ(expertSpec.ldb, 272U);

    AivCatlass::GemmLaunchSpec sharedSpec = {};
    ASSERT_TRUE(AivCatlass::BuildSharedGemmSpec<true>(true, 1, 272, 130, sharedSpec));
    EXPECT_TRUE(sharedSpec.hasWork);
    EXPECT_EQ(sharedSpec.offsetA, 0U);
    EXPECT_EQ(sharedSpec.offsetB, 0U);
    EXPECT_EQ(sharedSpec.offsetC, 0U);
    EXPECT_EQ(sharedSpec.ldb, 272U);
    EXPECT_FALSE(AivCatlass::BuildSharedGemmSpec<false>(false, 1, 272, 130, sharedSpec));
}

TEST(AlltoAllvGroupedMatMulCatlassTest, AcceptsOnlyCompiledTileSet)
{
    EXPECT_TRUE(AivCatlass::IsSupportedTile(128, 64, 128));
    EXPECT_TRUE(AivCatlass::IsSupportedTile(128, 32, 128));
    EXPECT_TRUE(AivCatlass::IsSupportedTile(64, 64, 128));
    EXPECT_TRUE(AivCatlass::IsSupportedTile(64, 32, 64));
    EXPECT_FALSE(AivCatlass::IsSupportedTile(64, 64, 64));
}

namespace AivMode = AlltoAllvGroupedMatMulAivMode;

TEST(AlltoAllvGroupedMatMulAivModeTest, BuildsExpertMetadataOneExpertAtATime)
{
    const int32_t recvPrefix[] = {1, 5, 7, 12, 15, 21};
    AivComm::ExpertMeta expert = {};
    ASSERT_TRUE(AivMode::BuildExpertMetaForIndex(recvPrefix, 2, 3, 1, 21, expert));
    EXPECT_EQ(expert.recvTokenBase, 5U);
    EXPECT_EQ(expert.tokenCount, 7U);

    uint64_t sourceOffset = 0U;
    ASSERT_TRUE(AivMode::GetDestinationSourceTokenOffset(recvPrefix, 2, 3, 1, 1, 21, sourceOffset));
    EXPECT_EQ(sourceOffset, 7U);
}

TEST(AlltoAllvGroupedMatMulAivModeTest, RejectsInvalidExpertMetadata)
{
    const int32_t invalidPrefix[] = {1, 3, 2, 5};
    AivComm::ExpertMeta expert = {};
    EXPECT_FALSE(AivMode::BuildExpertMetaForIndex(invalidPrefix, 2, 2, 1, 5, expert));
    EXPECT_FALSE(AivMode::BuildExpertMetaForIndex(invalidPrefix, 2, 2, 2, 5, expert));
}

TEST(AlltoAllvGroupedMatMulAivModeTest, MultipliesUint32WithoutDeviceRuntimeHelper)
{
    EXPECT_EQ(AivComm::MulU32ToU64(272U, 130U), 35360U);
    EXPECT_EQ(AivComm::MulU32ToU64(0xffffffffU, 0xffffffffU), 0xfffffffe00000001ULL);
}

TEST(AlltoAllvGroupedMatMulAivModeTest, PartitionsInitialWindowCopyAcrossAivTasks)
{
    AivMode::ElementRange range = {};
    ASSERT_TRUE(AivMode::PartitionElements(10, 0, 3, range));
    EXPECT_EQ(range.offset, 0U);
    EXPECT_EQ(range.count, 4U);
    ASSERT_TRUE(AivMode::PartitionElements(10, 1, 3, range));
    EXPECT_EQ(range.offset, 4U);
    EXPECT_EQ(range.count, 3U);
    ASSERT_TRUE(AivMode::PartitionElements(10, 2, 3, range));
    EXPECT_EQ(range.offset, 7U);
    EXPECT_EQ(range.count, 3U);
    EXPECT_FALSE(AivMode::PartitionElements(10, 3, 3, range));
}

TEST(AlltoAllvGroupedMatMulAivModeTest, AssignsOneSubblockZeroWorkerPerSourceRank)
{
    EXPECT_TRUE(AivMode::IsSourceWorker(0, 2, 0, 4, 0));
    EXPECT_TRUE(AivMode::IsSourceWorker(2, 2, 0, 4, 1));
    EXPECT_FALSE(AivMode::IsSourceWorker(2, 2, 1, 4, 1));
    EXPECT_FALSE(AivMode::IsSourceWorker(8, 2, 0, 4, 0));

    uint32_t workerNum = 0U;
    ASSERT_TRUE(AivMode::GetProducerWorkerCount(40U, 2U, workerNum));
    EXPECT_EQ(workerNum, 20U);
    // Twenty producer subblocks must cover all 128 ranks by striding 20.
    std::array<uint32_t, 128> visits = {};
    for (uint32_t worker = 0U; worker < workerNum; ++worker) {
        for (uint32_t rank = worker; rank < visits.size(); rank += workerNum) {
            ++visits[rank];
        }
    }
    for (const uint32_t visitCount : visits) {
        EXPECT_EQ(visitCount, 1U);
    }
    EXPECT_FALSE(AivMode::GetProducerWorkerCount(40U, 0U, workerNum));
    EXPECT_FALSE(AivMode::GetProducerWorkerCount(41U, 2U, workerNum));
}

} // namespace AlltoAllvGroupedMatMulUT
