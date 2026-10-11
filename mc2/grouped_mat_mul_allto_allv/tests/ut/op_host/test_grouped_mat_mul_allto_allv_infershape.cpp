/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace GroupedMatMulAlltoAllvUT {
struct TestParams {
    std::string testName{};
    std::vector<std::pair<std::string, std::string>> tilingParamsStrPair{};
    std::vector<std::pair<std::string, std::vector<int64_t>>> tilingParamsVecPair{};
    std::vector<std::pair<size_t, ge::DataType>> tilingInputDtypesPair{};
    std::vector<std::pair<size_t, ge::DataType>> tilingOutputDtypesPair{};
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
    uint64_t N2{4096};
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

struct TilingShapes {
    gert::StorageShape gmmXShape;
    gert::StorageShape gmmWeightShape;
    gert::StorageShape sendCountsShape;
    gert::StorageShape recvCountsShape;
    gert::StorageShape mmXShape;
    gert::StorageShape mmWeightShape;

    gert::StorageShape yShape;
    gert::StorageShape mmYShape;
};

struct TilingDTypes {
    std::vector<ge::DataType> inputDtypes{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT64,
                                          ge::DT_INT64,   ge::DT_FLOAT16, ge::DT_FLOAT16};
    std::vector<ge::DataType> outputDtypes{ge::DT_FLOAT16, ge::DT_FLOAT16};
};

class GroupedMatMulAlltoAllvInfershape : public testing::TestWithParam<TestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatMulAlltoAllvRuntimeProtoTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatMulAlltoAllvRuntimeProtoTest TearDown" << std::endl;
    }
};

std::unordered_map<std::string, std::function<void(TilingParams& tilingParams, const std::string& valueStr)>>
    g_infershapeParamsStrHandlers = {
        {"BSK",
         [](TilingParams& tilingParams, const std::string& valueStr) { tilingParams.BSK = std::stoi(valueStr); }},
        {"BS", [](TilingParams& tilingParams, const std::string& valueStr) { tilingParams.BS = std::stoi(valueStr); }},
        {"K", [](TilingParams& tilingParams, const std::string& valueStr) { tilingParams.K = std::stoi(valueStr); }},
        {"H1", [](TilingParams& tilingParams, const std::string& valueStr) { tilingParams.H1 = std::stoi(valueStr); }},
        {"H2", [](TilingParams& tilingParams, const std::string& valueStr) { tilingParams.H2 = std::stoi(valueStr); }},
        {"A", [](TilingParams& tilingParams, const std::string& valueStr) { tilingParams.A = std::stoi(valueStr); }},
        {"N1", [](TilingParams& tilingParams, const std::string& valueStr) { tilingParams.N1 = std::stoi(valueStr); }},
        {"N2", [](TilingParams& tilingParams, const std::string& valueStr) { tilingParams.N2 = std::stoi(valueStr); }},
        {"epWorldSize", [](TilingParams& tilingParams,
                           const std::string& valueStr) { tilingParams.epWorldSize = std::stoi(valueStr); }},
        {"e", [](TilingParams& tilingParams, const std::string& valueStr) { tilingParams.e = std::stoi(valueStr); }},
        {"gmmWeightDim1", [](TilingParams& tilingParams,
                             const std::string& valueStr) { tilingParams.gmmWeightDim1 = std::stoi(valueStr); }},
        {"yDim1",
         [](TilingParams& tilingParams, const std::string& valueStr) { tilingParams.yDim1 = std::stoi(valueStr); }},
        {"mmWeightDim0", [](TilingParams& tilingParams,
                            const std::string& valueStr) { tilingParams.mmWeightDim0 = std::stoi(valueStr); }},
        {"transGmmWeight", [](TilingParams& tilingParams,
                              const std::string& valueStr) { tilingParams.transGmmWeight = valueStr == "true"; }},
        {"transMmWeight", [](TilingParams& tilingParams, const std::string& valueStr) {
             tilingParams.transMmWeight = valueStr == "true";
         }}};

std::unordered_map<std::string, std::function<void(TilingParams& tilingParams, const std::vector<int64_t> valueVec)>>
    g_infershapeParamsVecHandlers = {
        {"sendCounts",
         [](TilingParams& tilingParams, const std::vector<int64_t> valueVec) { tilingParams.sendCounts = valueVec; }},
        {"recvCounts",
         [](TilingParams& tilingParams, const std::vector<int64_t> valueVec) { tilingParams.recvCounts = valueVec; }}};

TEST_P(GroupedMatMulAlltoAllvInfershape, InferdatatypeTest)
{
    auto testParam = GetParam();
    int64_t inputNum{6};
    int64_t outputNum{2};
    auto tilingParam = TilingParams{};
    auto tilingDtypes = TilingDTypes{};

    for (auto& kv : testParam.tilingInputDtypesPair) {
        if (kv.first >= 0 && kv.first < tilingDtypes.inputDtypes.size()) {
            tilingDtypes.inputDtypes[kv.first] = kv.second;
        }
    }
    for (auto& kv : testParam.tilingOutputDtypesPair) {
        if (kv.first >= 0 && kv.first < tilingDtypes.outputDtypes.size()) {
            tilingDtypes.outputDtypes[kv.first] = kv.second;
        }
    }

    std::vector<void*> inputDtypesPtrs(inputNum);
    for (int64_t i = 0; i < inputNum; i++) {
        inputDtypesPtrs[i] = &tilingDtypes.inputDtypes[i];
    }
    std::vector<void*> outputDtypesPtrs(outputNum);

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .NodeIoNum(inputNum, outputNum)
            .InputDataTypes(inputDtypesPtrs)
            .OutputDataTypes(outputDtypesPtrs)
            .NodeAttrs(
                {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tilingParam.group)},
                 {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParam.epWorldSize)},
                 {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParam.sendCounts)},
                 {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParam.recvCounts)},
                 {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParam.transGmmWeight)},
                 {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParam.transMmWeight)}})
            .Build();
    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto opImpl = spaceRegistry->GetOpImpl("GroupedMatMulAlltoAllv");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    /* do infershape */
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_P(GroupedMatMulAlltoAllvInfershape, InfershapeTest)
{
    auto testParam = GetParam();
    auto tilingParams = TilingParams{};
    for (auto& kv : testParam.tilingParamsStrPair) {
        if (g_infershapeParamsStrHandlers.count(kv.first) != 0) {
            g_infershapeParamsStrHandlers[kv.first](tilingParams, kv.second);
        }
    }
    for (auto& kv : testParam.tilingParamsVecPair) {
        if (g_infershapeParamsVecHandlers.count(kv.first) != 0) {
            g_infershapeParamsVecHandlers[kv.first](tilingParams, kv.second);
        }
    }

    gert::InfershapeContextPara infershapeContextPara(
        "GroupedMatMulAlltoAllv",
        {
            {{{tilingParams.A, tilingParams.H1}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tilingParams.e, tilingParams.gmmWeightDim1, tilingParams.N1}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tilingParams.BS, tilingParams.H2}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tilingParams.mmWeightDim0, tilingParams.N2}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tilingParams.group)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tilingParams.epWorldSize)},
         {"sendCounts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParams.sendCounts)},
         {"recvCounts", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(tilingParams.recvCounts)},
         {"transGmmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParams.transGmmWeight)},
         {"transMmWeight", Ops::Transformer::AnyValue::CreateFrom<bool>(tilingParams.transMmWeight)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{4096, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, testParam.status, expectOutputShape);
}

static TestParams g_testParamss[] = {
    {"Test_sample", {}, {}, {}, {}, ge::GRAPH_SUCCESS},
    {"Test_transGmmWeight_true",
     {{"transGmmWeight", "true"}, {"N1", "7168"}, {"gmmWeightDim1", "4096"}},
     {},
     {},
     {},
     ge::GRAPH_SUCCESS},
    {"Test_transMmWeight_true",
     {{"transMmWeight", "true"}, {"N2", "7168"}, {"mmWeightDim0", "4096"}},
     {},
     {},
     {},
     ge::GRAPH_SUCCESS},
    {"Test_both_trans_true",
     {{"transGmmWeight", "true"},
      {"transMmWeight", "true"},
      {"N1", "7168"},
      {"gmmWeightDim1", "4096"},
      {"N2", "7168"},
      {"mmWeightDim0", "4096"}},
     {},
     {},
     {},
     ge::GRAPH_SUCCESS},
    {"Test_gmm_dim_mismatch", {{"gmmWeightDim1", "4096"}}, {}, {}, {}, ge::GRAPH_FAILED},
    {"Test_mm_dim_mismatch", {{"mmWeightDim0", "4096"}}, {}, {}, {}, ge::GRAPH_FAILED},
};

INSTANTIATE_TEST_SUITE_P(GroupedMatMulAlltoAllv, GroupedMatMulAlltoAllvInfershape, testing::ValuesIn(g_testParamss),
                         [](const testing::TestParamInfo<GroupedMatMulAlltoAllvInfershape::ParamType>& info) {
                             return info.param.testName;
                         });
} // namespace GroupedMatMulAlltoAllvUT
