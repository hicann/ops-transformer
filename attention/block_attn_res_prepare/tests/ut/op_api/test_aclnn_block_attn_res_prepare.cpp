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
 * \file test_aclnn_block_attn_res_prepare.cpp
 * \brief CSV-driven ACLNN unit tests for BlockAttnResPrepare.
 */

#include <gtest/gtest.h>

#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../../op_host/op_api/aclnn_block_attn_res_prepare.h"
#include "gmm_csv_acl_parse_utils.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {

constexpr size_t CSV_COLUMN_COUNT = 25U;
constexpr const char *CSV_FILE_NAME = "test_aclnn_block_attn_res_prepare.csv";
constexpr const char *CSV_REPO_DIR = "attention/block_attn_res_prepare/tests/ut/op_api";

std::vector<uint64_t> ParseU64List(const std::string &value)
{
    const std::string trimmed = ops::ut::Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    std::vector<std::string> tokens;
    ops::ut::SplitStr2Vec(trimmed, "|", tokens);
    std::vector<uint64_t> result;
    result.reserve(tokens.size());
    for (const auto &token : tokens) {
        const std::string trimmedToken = ops::ut::Trim(token);
        if (trimmedToken.empty() || trimmedToken.front() == '-') {
            throw std::invalid_argument("validBlocksValue must contain uint64 values");
        }
        size_t parsedLength = 0U;
        const uint64_t parsedValue = std::stoull(trimmedToken, &parsedLength);
        if (parsedLength != trimmedToken.size()) {
            throw std::invalid_argument("validBlocksValue must contain uint64 values");
        }
        result.emplace_back(parsedValue);
    }
    return result;
}

struct BlockAttnResPrepareOpApiCase {
    void Run() const
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
        TensorDesc blockRes = ops::ut::BuildAclTensorDescFromSpec(blockResShape, blockResDtype, blockResFormat);
        TensorDesc validBlocks(ops::ut::ParseAclTensorViewDims(validBlocksShape),
                               ops::ut::ParseAclDtype(validBlocksDtype), ops::ut::ParseAclFormat(validBlocksFormat));
        const std::vector<uint64_t> validBlocksData = ParseU64List(validBlocksValue);
        if (!validBlocksData.empty()) {
            validBlocks.Value(validBlocksData);
        }
        TensorDesc pseudoQuery =
            ops::ut::BuildAclTensorDescFromSpec(pseudoQueryShape, pseudoQueryDtype, pseudoQueryFormat);
        TensorDesc numerator = ops::ut::BuildAclTensorDescFromSpec(numeratorShape, numeratorDtype, numeratorFormat);
        TensorDesc logitMax = ops::ut::BuildAclTensorDescFromSpec(logitMaxShape, logitMaxDtype, logitMaxFormat);
        TensorDesc expSum = ops::ut::BuildAclTensorDescFromSpec(expSumShape, expSumDtype, expSumFormat);

        auto ut = OP_API_UT(aclnnBlockAttnResPrepare, INPUT(blockRes, validBlocks, pseudoQuery),
                            OUTPUT(numerator, logitMax, expSum), eps);
        uint64_t workspaceSize = 0U;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), expectRet) << "case=" << caseName;
    }

    std::string socVersion;
    std::string caseName;
    std::string prefix;
    std::string blockResShape;
    std::string blockResDtype;
    std::string blockResFormat;
    std::string validBlocksShape;
    std::string validBlocksDtype;
    std::string validBlocksFormat;
    std::string validBlocksValue;
    std::string pseudoQueryShape;
    std::string pseudoQueryDtype;
    std::string pseudoQueryFormat;
    std::string numeratorShape;
    std::string numeratorDtype;
    std::string numeratorFormat;
    std::string logitMaxShape;
    std::string logitMaxDtype;
    std::string logitMaxFormat;
    std::string expSumShape;
    std::string expSumDtype;
    std::string expSumFormat;
    float eps = 1.0e-6F;
    aclnnStatus expectRet = ACLNN_SUCCESS;
};

std::vector<BlockAttnResPrepareOpApiCase> LoadCases()
{
    std::vector<BlockAttnResPrepareOpApiCase> cases;
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
            BlockAttnResPrepareOpApiCase testCase;
            testCase.socVersion = ops::ut::Trim(items[index++]);
            testCase.caseName = ops::ut::Trim(items[index++]);
            if (!ops::ut::ParseBool(items[index++]) || testCase.socVersion != "Ascend950") {
                continue;
            }
            testCase.prefix = ops::ut::Trim(items[index++]);
            testCase.blockResShape = ops::ut::Trim(items[index++]);
            testCase.blockResDtype = ops::ut::Trim(items[index++]);
            testCase.blockResFormat = ops::ut::Trim(items[index++]);
            testCase.validBlocksShape = ops::ut::Trim(items[index++]);
            testCase.validBlocksDtype = ops::ut::Trim(items[index++]);
            testCase.validBlocksFormat = ops::ut::Trim(items[index++]);
            testCase.validBlocksValue = ops::ut::Trim(items[index++]);
            testCase.pseudoQueryShape = ops::ut::Trim(items[index++]);
            testCase.pseudoQueryDtype = ops::ut::Trim(items[index++]);
            testCase.pseudoQueryFormat = ops::ut::Trim(items[index++]);
            testCase.numeratorShape = ops::ut::Trim(items[index++]);
            testCase.numeratorDtype = ops::ut::Trim(items[index++]);
            testCase.numeratorFormat = ops::ut::Trim(items[index++]);
            testCase.logitMaxShape = ops::ut::Trim(items[index++]);
            testCase.logitMaxDtype = ops::ut::Trim(items[index++]);
            testCase.logitMaxFormat = ops::ut::Trim(items[index++]);
            testCase.expSumShape = ops::ut::Trim(items[index++]);
            testCase.expSumDtype = ops::ut::Trim(items[index++]);
            testCase.expSumFormat = ops::ut::Trim(items[index++]);
            testCase.eps = std::stof(ops::ut::Trim(items[index++]));
            testCase.expectRet = ops::ut::ParseAclnnStatus(items[index++]);
            cases.emplace_back(std::move(testCase));
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error);
        }
    }
    return cases;
}

std::string MakeParamName(const testing::TestParamInfo<BlockAttnResPrepareOpApiCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

const std::vector<BlockAttnResPrepareOpApiCase> &GetCases()
{
    static const std::vector<BlockAttnResPrepareOpApiCase> cases = LoadCases();
    return cases;
}

class BlockAttnResPrepareOpApiTest : public testing::TestWithParam<BlockAttnResPrepareOpApiCase> {};

TEST_P(BlockAttnResPrepareOpApiTest, CsvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(BLOCK_ATTN_RES_PREPARE_OP_API_CSV, BlockAttnResPrepareOpApiTest, testing::ValuesIn(GetCases()),
                         MakeParamName);

} // namespace
