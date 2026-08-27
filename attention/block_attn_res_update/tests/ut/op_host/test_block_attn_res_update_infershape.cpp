/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_block_attn_res_update_infershape.cpp
 * \brief CSV-driven unit tests for BlockAttnResUpdate shape and dtype inference.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "base/registry/op_impl_space_registry_v2.h"
#include "gmm_csv_ge_parse_utils.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

namespace {

constexpr const char *OP_NAME = "BlockAttnResUpdate";
constexpr const char *SOC_VERSION = "Ascend950";
constexpr size_t CSV_COLUMN_COUNT = 15UL;

ge::DataType ParseDtype(const std::string &value)
{
    return ops::ut::ParseGeDtype(ops::ut::Trim(value));
}

gert::StorageShape ParseShape(const std::string &value)
{
    return ops::ut::MakeGertStorageShape(ops::ut::ParseDims(ops::ut::Trim(value)));
}

struct BlockAttnResUpdateInferCase {
    void RunShapeInference() const
    {
        std::vector<gert::InfershapeContextPara::TensorDescription> inputs;
        inputs.emplace_back(ParseShape(partialShape), partialDtype, ge::FORMAT_ND);
        inputs.emplace_back(ParseShape(partialShape), ge::DT_BF16, ge::FORMAT_ND);
        inputs.emplace_back(ParseShape("1"), ge::DT_FLOAT, ge::FORMAT_ND);
        inputs.emplace_back(ParseShape(partialShape), ge::DT_FLOAT, ge::FORMAT_ND);
        inputs.emplace_back(ParseShape("1"), ge::DT_FLOAT, ge::FORMAT_ND);
        inputs.emplace_back(ParseShape("1"), ge::DT_FLOAT, ge::FORMAT_ND);

        std::vector<gert::InfershapeContextPara::TensorDescription> outputs;
        if (partialRefPresent) {
            outputs.emplace_back(ParseShape(""), ge::DT_FLOAT, ge::FORMAT_ND);
        }
        if (hPresent) {
            outputs.emplace_back(ParseShape(""), ge::DT_BF16, ge::FORMAT_ND);
        }

        const std::vector<uint32_t> inputInstanceNum = {
            1U, 1U, 1U, 1U, 1U, 1U,
        };
        const std::vector<uint32_t> outputInstanceNum = {
            partialRefPresent ? 1U : 0U,
            hPresent ? 1U : 0U,
        };
        gert::InfershapeContextPara para(OP_NAME, inputs, outputs, inputInstanceNum, outputInstanceNum);

        std::vector<std::vector<int64_t>> expectedShapes;
        if (expectSuccess) {
            expectedShapes.emplace_back(ops::ut::ParseDims(expectPartialShape));
            expectedShapes.emplace_back(ops::ut::ParseDims(expectHShape));
        }
        ExecuteTestCase(para, expectSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED, expectedShapes);
    }

    void RunDtypeInference() const
    {
        auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
        ASSERT_NE(spaceRegistry, nullptr);
        auto opImpl = spaceRegistry->GetOpImpl(OP_NAME);
        ASSERT_NE(opImpl, nullptr);
        ASSERT_NE(opImpl->infer_datatype, nullptr);

        ge::DataType partialInputDtype = partialDtype;
        ge::DataType deltaDtype = ge::DT_BF16;
        ge::DataType floatDtype = ge::DT_FLOAT;
        ge::DataType outputPartialDtype = ge::DT_UNDEFINED;
        ge::DataType outputHDtype = ge::DT_UNDEFINED;
        auto contextHolder =
            gert::InferDataTypeContextFaker()
                .SetOpType(OP_NAME)
                .IrInputNum(6)
                .NodeIoNum(6, 2)
                .NodeInputTd(0, partialInputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, deltaDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, floatDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, floatDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, floatDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, floatDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&partialInputDtype, &deltaDtype, &floatDtype, &floatDtype, &floatDtype, &floatDtype})
                .OutputDataTypes({&outputPartialDtype, &outputHDtype})
                .Build();
        auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS) << "case=" << caseName;
        EXPECT_EQ(context->GetOutputDataType(0), expectPartialDtype) << "case=" << caseName;
        EXPECT_EQ(context->GetOutputDataType(1), expectHDtype) << "case=" << caseName;
    }

    void Run() const
    {
        if (runShape) {
            RunShapeInference();
        }
        if (runDtype) {
            RunDtypeInference();
        }
    }

    std::string caseName;
    std::string prefix;
    bool runShape = false;
    bool runDtype = false;
    bool partialRefPresent = true;
    bool hPresent = true;
    std::string partialShape;
    ge::DataType partialDtype = ge::DT_UNDEFINED;
    bool expectSuccess = false;
    std::string expectPartialShape;
    std::string expectHShape;
    ge::DataType expectPartialDtype = ge::DT_UNDEFINED;
    ge::DataType expectHDtype = ge::DT_UNDEFINED;
};

struct InferCsvLoadResult {
    std::vector<BlockAttnResUpdateInferCase> cases;
    std::vector<std::string> errors;
};

InferCsvLoadResult LoadInferCases()
{
    InferCsvLoadResult result;
    const std::string csvPath = ops::ut::ResolveCsvPath("test_block_attn_res_update_infershape.csv",
                                                        "attention/block_attn_res_update/tests/ut/op_host", __FILE__);
    std::ifstream csvData(csvPath, std::ios::in);
    if (!csvData.is_open()) {
        result.errors.emplace_back("Cannot open inference case file: " + csvPath);
        return result;
    }

    std::string line;
    size_t lineNo = 0UL;
    while (std::getline(csvData, line)) {
        ++lineNo;
        if (ops::ut::Trim(line).empty() || ops::ut::Trim(line).front() == '#') {
            continue;
        }

        std::vector<std::string> items;
        ops::ut::SplitStr2Vec(line, ",", items);
        if (items.empty() || ops::ut::Trim(items[0]) == "socVersion") {
            continue;
        }
        if (items.size() < CSV_COLUMN_COUNT) {
            result.errors.emplace_back("Invalid inference CSV column count at " + csvPath + ":" +
                                       std::to_string(lineNo));
            continue;
        }

        const std::string caseName = items.size() > 1UL ? ops::ut::Trim(items[1]) : "";
        try {
            size_t index = 0UL;
            const std::string socVersion = ops::ut::Trim(items[index++]);
            if (socVersion != SOC_VERSION) {
                continue;
            }

            BlockAttnResUpdateInferCase testCase;
            testCase.caseName = ops::ut::Trim(items[index++]);
            const bool enable = ops::ut::ParseBool(items[index++]);
            if (!enable) {
                continue;
            }
            testCase.prefix = ops::ut::Trim(items[index++]);
            testCase.runShape = ops::ut::ParseBool(items[index++]);
            testCase.runDtype = ops::ut::ParseBool(items[index++]);
            testCase.partialRefPresent = ops::ut::ParseBool(items[index++]);
            testCase.hPresent = ops::ut::ParseBool(items[index++]);
            testCase.partialShape = ops::ut::Trim(items[index++]);
            testCase.partialDtype = ParseDtype(items[index++]);
            testCase.expectSuccess = ops::ut::ParseBool(items[index++]);
            testCase.expectPartialShape = ops::ut::Trim(items[index++]);
            testCase.expectHShape = ops::ut::Trim(items[index++]);
            testCase.expectPartialDtype = ParseDtype(items[index++]);
            testCase.expectHDtype = ParseDtype(items[index++]);
            result.cases.emplace_back(std::move(testCase));
        } catch (const std::exception &error) {
            result.errors.emplace_back(ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error));
        }
    }
    if (result.cases.empty()) {
        result.errors.emplace_back("No enabled Ascend950 inference cases were loaded from: " + csvPath);
    }
    return result;
}

const InferCsvLoadResult &GetInferCases()
{
    static const InferCsvLoadResult result = LoadInferCases();
    return result;
}

std::string MakeParamName(const testing::TestParamInfo<BlockAttnResUpdateInferCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

class BlockAttnResUpdateInferTest : public testing::TestWithParam<BlockAttnResUpdateInferCase> {};

TEST(BlockAttnResUpdateInferCsv, LoadsEnabledCases)
{
    const auto &result = GetInferCases();
    for (const auto &error : result.errors) {
        ADD_FAILURE() << error;
    }
    EXPECT_FALSE(result.cases.empty());
}

TEST(BlockAttnResUpdateInferDirect, RejectsNullContexts)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl(OP_NAME);
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);
    EXPECT_EQ(opImpl->infer_shape(nullptr), ge::GRAPH_FAILED);
    EXPECT_EQ(opImpl->infer_datatype(nullptr), ge::GRAPH_FAILED);
}

TEST_P(BlockAttnResUpdateInferTest, CsvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(BLOCK_ATTN_RES_UPDATE_INFERENCE, BlockAttnResUpdateInferTest,
                         testing::ValuesIn(GetInferCases().cases), MakeParamName);

} // namespace
