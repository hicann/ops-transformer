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
 * \file test_block_attn_res_prepare_infershape.cpp
 * \brief CSV-driven shape and dtype inference unit tests for BlockAttnResPrepare.
 */

#include <gtest/gtest.h>

#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "base/registry/op_impl_space_registry_v2.h"
#include "gmm_csv_ge_parse_utils.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

namespace {

using TensorDescription = gert::InfershapeContextPara::TensorDescription;

constexpr size_t CSV_COLUMN_COUNT = 16U;
constexpr const char *CSV_FILE_NAME = "test_block_attn_res_prepare_infershape.csv";
constexpr const char *CSV_REPO_DIR = "attention/block_attn_res_prepare/tests/ut/op_host";

struct BlockAttnResPrepareInfershapeCase {
    void RunShapeCase() const
    {
        const auto makeTensorDescription = [](const std::string &shape, const std::string &dtype) {
            const auto dims = ops::ut::ParseDims(shape);
            return TensorDescription(ops::ut::MakeGertStorageShape(dims, dims), ops::ut::ParseGeDtype(dtype),
                                     ge::FORMAT_ND);
        };
        const std::vector<TensorDescription> inputs = {
            makeTensorDescription(blockResShape, blockResDtype),
            makeTensorDescription(validBlocksShape, validBlocksDtype),
            makeTensorDescription(pseudoQueryShape, pseudoQueryDtype),
        };
        const std::vector<TensorDescription> outputs = {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        };
        gert::InfershapeContextPara context("BlockAttnResPrepare", inputs, outputs, {});
        if (!expectSuccess) {
            ExecuteTestCase(context, ge::GRAPH_FAILED);
            return;
        }
        ExecuteTestCase(context, ge::GRAPH_SUCCESS,
                        {ops::ut::ParseDims(expectNumeratorShape), ops::ut::ParseDims(expectLogitMaxShape),
                         ops::ut::ParseDims(expectExpSumShape)});
    }

    void RunDtypeCase() const
    {
        auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
        ASSERT_NE(spaceRegistry, nullptr);
        const auto opImpl = spaceRegistry->GetOpImpl("BlockAttnResPrepare");
        ASSERT_NE(opImpl, nullptr);
        const auto inferDtype = opImpl->infer_datatype;
        ASSERT_NE(inferDtype, nullptr);

        ge::DataType blockResType = ops::ut::ParseGeDtype(blockResDtype);
        ge::DataType validBlocksType = ops::ut::ParseGeDtype(validBlocksDtype);
        ge::DataType pseudoQueryType = ops::ut::ParseGeDtype(pseudoQueryDtype);
        std::vector<std::string> outputDtypes;
        ops::ut::SplitStr2Vec(expectOutputDtypes, "|", outputDtypes);
        ASSERT_EQ(outputDtypes.size(), 3U) << "case=" << caseName;
        ge::DataType numeratorType = ops::ut::ParseGeDtype(outputDtypes[0]);
        ge::DataType logitMaxType = ops::ut::ParseGeDtype(outputDtypes[1]);
        ge::DataType expSumType = ops::ut::ParseGeDtype(outputDtypes[2]);
        auto contextHolder = gert::InferDataTypeContextFaker()
                                 .IrInputNum(3)
                                 .NodeIoNum(3, 3)
                                 .NodeInputTd(0, blockResType, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(1, validBlocksType, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(2, pseudoQueryType, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .InputDataTypes({&blockResType, &validBlocksType, &pseudoQueryType})
                                 .OutputDataTypes({&numeratorType, &logitMaxType, &expSumType})
                                 .Build();
        auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        const ge::graphStatus expectedStatus = expectSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
        ASSERT_EQ(inferDtype(context), expectedStatus) << "case=" << caseName;
        if (expectSuccess) {
            EXPECT_EQ(context->GetOutputDataType(0), numeratorType);
            EXPECT_EQ(context->GetOutputDataType(1), logitMaxType);
            EXPECT_EQ(context->GetOutputDataType(2), expSumType);
        }
    }

    void Run() const
    {
        if (caseType == "dtype") {
            RunDtypeCase();
        } else {
            RunShapeCase();
        }
    }

    std::string socVersion;
    std::string caseName;
    std::string prefix;
    std::string caseType;
    std::string blockResShape;
    std::string validBlocksShape;
    std::string pseudoQueryShape;
    std::string blockResDtype;
    std::string validBlocksDtype;
    std::string pseudoQueryDtype;
    bool expectSuccess = false;
    std::string expectNumeratorShape;
    std::string expectLogitMaxShape;
    std::string expectExpSumShape;
    std::string expectOutputDtypes;
};

std::vector<BlockAttnResPrepareInfershapeCase> LoadCases()
{
    std::vector<BlockAttnResPrepareInfershapeCase> cases;
    const std::string csvPath = ops::ut::ResolveCsvPath(CSV_FILE_NAME, CSV_REPO_DIR, __FILE__);
    std::ifstream csvData(csvPath, std::ios::in);
    if (!csvData.is_open()) {
        std::cout << "cannot open case file " << csvPath << std::endl;
        return cases;
    }

    std::string line;
    size_t lineNo = 0U;
    while (std::getline(csvData, line)) {
        ++lineNo;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::vector<std::string> items;
        ops::ut::SplitStr2Vec(line, ",", items);
        if (items.empty() || items[0] == "socVersion" || items.size() < CSV_COLUMN_COUNT) {
            continue;
        }

        const std::string caseName = ops::ut::Trim(items[1]);
        try {
            size_t index = 0U;
            BlockAttnResPrepareInfershapeCase testCase;
            testCase.socVersion = ops::ut::Trim(items[index++]);
            testCase.caseName = ops::ut::Trim(items[index++]);
            if (!ops::ut::ParseBool(items[index++]) || testCase.socVersion != "Ascend950") {
                continue;
            }
            testCase.prefix = ops::ut::Trim(items[index++]);
            testCase.caseType = ops::ut::Trim(items[index++]);
            testCase.blockResShape = ops::ut::Trim(items[index++]);
            testCase.validBlocksShape = ops::ut::Trim(items[index++]);
            testCase.pseudoQueryShape = ops::ut::Trim(items[index++]);
            testCase.blockResDtype = ops::ut::Trim(items[index++]);
            testCase.validBlocksDtype = ops::ut::Trim(items[index++]);
            testCase.pseudoQueryDtype = ops::ut::Trim(items[index++]);
            testCase.expectSuccess = ops::ut::ParseBool(items[index++]);
            testCase.expectNumeratorShape = ops::ut::Trim(items[index++]);
            testCase.expectLogitMaxShape = ops::ut::Trim(items[index++]);
            testCase.expectExpSumShape = ops::ut::Trim(items[index++]);
            testCase.expectOutputDtypes = ops::ut::Trim(items[index++]);
            cases.emplace_back(std::move(testCase));
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error);
        }
    }
    return cases;
}

std::string MakeParamName(const testing::TestParamInfo<BlockAttnResPrepareInfershapeCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

const std::vector<BlockAttnResPrepareInfershapeCase> &GetCases()
{
    static const std::vector<BlockAttnResPrepareInfershapeCase> cases = LoadCases();
    return cases;
}

class BlockAttnResPrepareInfershapeTest : public testing::TestWithParam<BlockAttnResPrepareInfershapeCase> {};

TEST_P(BlockAttnResPrepareInfershapeTest, CsvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(BLOCK_ATTN_RES_PREPARE_INFERSHAPE_CSV, BlockAttnResPrepareInfershapeTest,
                         testing::ValuesIn(GetCases()), MakeParamName);

} // namespace
