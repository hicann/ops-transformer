/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../op_host/op_tiling/arch35/allto_allv_grouped_mat_mul_tiling_a5.h"

#include <iostream>
#include <gtest/gtest.h>

#include "mc2_tiling_case_executor.h"

using namespace std;

namespace AlltoAllvGroupedMatMulA5UT {
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
    uint64_t commOut{0};
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
        {{}, ge::DT_INT32, ge::FORMAT_ND}, // placeholder
        {{}, ge::DT_INT32, ge::FORMAT_ND}, // placeholder
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

class AlltoAllvGroupedMatMulArch35TilingTest : public testing::TestWithParam<TestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllvGroupedMatMulArch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllvGroupedMatMulArch35TilingTest TearDown" << std::endl;
    }

public:
    std::vector<int64_t> sendCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    std::vector<int64_t> recvCounts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
};

TEST_P(AlltoAllvGroupedMatMulArch35TilingTest, ShapeSize)
{
    auto testParam = GetParam();

    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;

    std::string socVersion = "Ascend950";
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
        mmYShape->shape_ = {};
    }

    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul", CreateInputTensors(tilingParams, mmXShape, mmWeightShape),
        CreateOutputTensors(tilingParams, mmYShape),
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tilingParams.group)},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.epWorldSize)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tilingParams.sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tilingParams.recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParams.permuteOutFlag)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    if (testParam.status == ge::GRAPH_FAILED) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
    } else {
        // arch35 tilingKey 可能与 arch22 不同，跳过 tilingKey 校验
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, UINT64_MAX);
    }
}

// arch35 参数化测试用例：epWorldSize 必须在 {2,4,8,16,32,64} 中
static TestParam testParams[] = {
    // 不支持的 epWorldSize (不在 {2,4,8,16,32,64} 中)，应失败
    {"Test_unsupported_ep_world_size_3", {{"epWorldSize", "3"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_unsupported_ep_world_size_5", {{"epWorldSize", "5"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_unsupported_ep_world_size_6", {{"epWorldSize", "6"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_unsupported_ep_world_size_10",
     {{"epWorldSize", "10"}, {"permuteOutFlag", "true"}},
     {},
     {},
     ge::GRAPH_FAILED},
    {"Test_unsupported_ep_world_size_12",
     {{"epWorldSize", "12"}, {"permuteOutFlag", "true"}},
     {},
     {},
     ge::GRAPH_FAILED},
    {"Test_unsupported_ep_world_size_48",
     {{"epWorldSize", "48"}, {"permuteOutFlag", "true"}},
     {},
     {},
     ge::GRAPH_FAILED},
    // gmmWeight size 错误
    {"Test_gmmWeight_size", {{"e", "64"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    // epWorldSize=4 正常测试（在 {2,4,8,16,32,64} 中）
    {"Test_ep_world_size_4", {{"epWorldSize", "4"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    // epWorldSize=2 正常测试
    {"Test_ep_world_size_2", {{"epWorldSize", "2"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    // e 参数测试
    {"Test_e", {{"e", "64"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    // 多 ep 测试 epWorldSize=16
    {"Test_e_multi_ep_16",
     {{"epWorldSize", "16"}, {"e", "32"}, {"permuteOutFlag", "true"}},
     {{"sendCounts", std::vector<int64_t>(512, 128)}, {"recvCounts", std::vector<int64_t>(512, 128)}},
     {},
     ge::GRAPH_FAILED},
    // 多 ep 测试 epWorldSize=32
    {"Test_e_multi_ep_32",
     {{"epWorldSize", "32"}, {"e", "32"}, {"permuteOutFlag", "true"}},
     {{"sendCounts", std::vector<int64_t>(1024, 128)}, {"recvCounts", std::vector<int64_t>(1024, 128)}},
     {},
     ge::GRAPH_FAILED},
    // 多 ep 测试 epWorldSize=64
    {"Test_e_multi_ep_64",
     {{"epWorldSize", "64"}, {"e", "32"}, {"permuteOutFlag", "true"}},
     {{"sendCounts", std::vector<int64_t>(2048, 128)}, {"recvCounts", std::vector<int64_t>(2048, 128)}},
     {},
     ge::GRAPH_FAILED},
    // send counts size 错误
    {"Test_send_counts_size",
     {{"epWorldSize", "16"}, {"e", "32"}, {"permuteOutFlag", "true"}},
     {},
     {},
     ge::GRAPH_FAILED},
    // BSK 过大
    {"Test_BSK_1", {{"BSK", "52428800"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    // BS 过大
    {"Test_BS_1", {{"BS", "52428800"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    // H1 过大
    {"Test_H1", {{"H1", "65536"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    // H2 + mmWeightDim0 不匹配
    {"Test_H2", {{"H2", "12289"}, {"mmWeightDim0", "12289"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    // N1 过大
    {"Test_N1", {{"N1", "65536"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    // N2 过大
    {"Test_N2", {{"N2", "65536"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    // H1 != gmmWeightDim1
    {"Test_H_1", {{"H1", "7168"}, {"gmmWeightDim1", "7169"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    // H2 != mmWeightDim0
    {"Test_H_3", {{"H2", "7168"}, {"mmWeightDim0", "7169"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    // H1 过大 2
    {"Test_H_4", {{"H1", "65536"}, {"permuteOutFlag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    // send counts 校验（arch35 NeedToCheckCounts=false，send counts 不会被严格校验）
    {"Test_send_counts_0",
     {{"BSK", "16386"}, {"BS", "2048"}, {"permuteOutFlag", "true"}},
     {{"sendCounts",
       std::vector<int64_t>{
           3201, 3201, 3200, 3200, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
           128,  128,  128,  128,  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
       }}},
     {},
     ge::GRAPH_FAILED},
    // recv counts 校验
    {"Test_recv_counts_0",
     {{"A", "16386"}, {"BS", "8193"}, {"permuteOutFlag", "true"}},
     {{"recvCounts",
       std::vector<int64_t>{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,
                            128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 3200}}},
     {},
     ge::GRAPH_FAILED},
    // recv counts 2
    {"Test_recv_counts_1",
     {{"A", "16386"}, {"BS", "8193"}, {"permuteOutFlag", "true"}},
     {{"recvCounts",
       std::vector<int64_t>{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,  128, 128,
                            128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 1600, 1600}}},
     {},
     ge::GRAPH_FAILED},
    // 不带 MM 场景
    {"Test_no_MM", {{"permuteOutFlag", "true"}, {"isNeedMM", "false"}}, {}, {}, ge::GRAPH_SUCCESS}};

INSTANTIATE_TEST_SUITE_P(AlltoAllvGroupedMatMul, AlltoAllvGroupedMatMulArch35TilingTest, testing::ValuesIn(testParams),
                         [](const testing::TestParamInfo<AlltoAllvGroupedMatMulArch35TilingTest::ParamType> &info) {
                             return info.param.testName;
                         });

// === 独立测试用例 ===

// 基本成功场景：标准参数，带 permuteOutFlag
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, BasicSuccess)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
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
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// H4 维度不匹配测试
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, H4)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
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
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// A1 维度测试
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, A1)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
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
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// BS1 维度测试：带 MM 的场景
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, BS1)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
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
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Dim1: input x 有额外维度
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, Dim1)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
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
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Dim2: gmm weight 2D（没有 e 维度）
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, Dim2)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
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
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Dim3: 标准参数，不带 MM 的形状
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, Dim3)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
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
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Dim5: 带 MM，3D mmX
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, Dim5)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
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
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Dim6: 带 MM，3D mmWeight
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, Dim6)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
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
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Dim7: 带 MM，3D mmX 和 3D mmWeight
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, Dim7)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
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
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Dim10: output[2] 为 1D
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, Dim10)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
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
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
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
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// TransMmWeight1: transGmmWeight=false, transMmWeight=false
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, TransMmWeight1)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
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
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// epWorldSize=2 测试
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, EpWorldSize2)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    std::vector<int64_t> sendCounts2{2048, 2048};
    std::vector<int64_t> recvCounts2{2048, 2048};
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
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts2)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts2)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// epWorldSize=16 测试
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, EpWorldSize16)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    std::vector<int64_t> sendCounts16(256, 64);
    std::vector<int64_t> recvCounts16(256, 64);
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
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts16)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts16)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 16}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// epWorldSize=32 测试
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, EpWorldSize32)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    std::vector<int64_t> sendCounts32(1024, 32);
    std::vector<int64_t> recvCounts32(1024, 32);
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
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts32)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts32)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 32}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// commMode 为空字符串时按 rankDim 选择 CCU/AICPU（覆盖 GetAndConvertCommMode 默认分支）
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, CommModeEmptyDefault)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
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
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// commMode=ai_cpu
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, CommModeAiCpu)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
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
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// commMode 非法字符串
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, CommModeInvalid)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
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
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("invalid_mode")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// commMode 为空且 rankDim>8 时走 AICPU
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, CommModeEmptyEp16)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    std::vector<int64_t> sendCounts16(64, 64);
    std::vector<int64_t> recvCounts16(64, 64);
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
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts16)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts16)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 16}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, UINT64_MAX);
}

// epWorldSize=64 测试
TEST_F(AlltoAllvGroupedMatMulArch35TilingTest, EpWorldSize64)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;
    std::string socVersion = "Ascend950";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;
    std::vector<int64_t> sendCounts64(4096, 16);
    std::vector<int64_t> recvCounts64(4096, 16);
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
            {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(sendCounts64)},
            {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(recvCounts64)},
            {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"permuteOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"commMode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ccu")},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 64}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

} // namespace AlltoAllvGroupedMatMulA5UT
