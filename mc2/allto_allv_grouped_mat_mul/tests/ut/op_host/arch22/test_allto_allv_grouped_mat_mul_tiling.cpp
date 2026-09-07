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
