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
 * \file test_aclnn_block_attn_res_update.cpp
 * \brief CSV-driven opapi UT for aclnnBlockAttnResUpdate.
 */

#include <exception>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../op_api/aclnn_block_attn_res_update.h"
#include "../../../op_api/aclnn_block_attn_res_update_v2.h"
#include "block_attn_res_update_test_utils.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

namespace {

namespace TestUtils = BlockAttnResUpdate::Test;

constexpr size_t CSV_COLUMN_COUNT = 27UL;
constexpr size_t CASE_NAME_COLUMN_INDEX = 1UL;
constexpr const char* RUN_MODE_GET_WORKSPACE = "GET_WORKSPACE";
constexpr const char* RUN_MODE_GET_WORKSPACE_V2 = "GET_WORKSPACE_V2";
constexpr const char* RUN_MODE_NULL_WORKSPACE_SIZE = "NULL_WORKSPACE_SIZE";
constexpr const char* RUN_MODE_NULL_EXECUTOR_OUT = "NULL_EXECUTOR_OUT";
constexpr const char* RUN_MODE_NULL_PARTIAL_BLOCK_REF = "NULL_PARTIAL_BLOCK_REF";
constexpr const char* RUN_MODE_NULL_DELTA = "NULL_DELTA";
constexpr const char* RUN_MODE_NULL_PSEUDO_QUERY = "NULL_PSEUDO_QUERY";
constexpr const char* RUN_MODE_NULL_NUMERATOR = "NULL_NUMERATOR";
constexpr const char* RUN_MODE_NULL_LOGIT_MAX = "NULL_LOGIT_MAX";
constexpr const char* RUN_MODE_NULL_EXP_SUM = "NULL_EXP_SUM";
constexpr const char* RUN_MODE_NULL_H = "NULL_H";
constexpr const char* RUN_MODE_BLOCK_ATTN_RES_UPDATE_NULL_EXECUTOR = "BLOCK_ATTN_RES_UPDATE_NULL_EXECUTOR";

TensorDesc MakeTensorDesc(const std::string& spec, const std::string& dtype, const std::string& format)
{
    const auto parsed = TestUtils::ParseAclTensorSpec(spec);
    return TensorDesc(parsed.viewDims, TestUtils::ParseAclDtype(dtype), TestUtils::ParseAclFormat(format),
                      parsed.stride, parsed.offset, parsed.storageDims)
        .ValueRange(-10, 10);
}

struct BlockAttnResUpdateOpApiCase {
    aclnnStatus RunSpecialCase(const TensorDesc& partialBlockRefDesc, const TensorDesc& deltaDesc,
                               const TensorDesc& pseudoQueryDesc, const TensorDesc& numeratorDesc,
                               const TensorDesc& logitMaxDesc, const TensorDesc& expSumDesc,
                               const TensorDesc& hDesc) const
    {
        if (runMode == RUN_MODE_BLOCK_ATTN_RES_UPDATE_NULL_EXECUTOR) {
            return aclnnBlockAttnResUpdate(nullptr, 0U, nullptr, nullptr);
        }

        auto partialBlockRef = partialBlockRefDesc.ToAclType();
        auto delta = deltaDesc.ToAclType();
        auto pseudoQuery = pseudoQueryDesc.ToAclType();
        auto numerator = numeratorDesc.ToAclType();
        auto logitMax = logitMaxDesc.ToAclType();
        auto expSum = expSumDesc.ToAclType();
        auto h = hDesc.ToAclType();
        auto weight = MakeTensorDesc(outputNormWeightSpec == "NONE" ? "0" : outputNormWeightSpec, outputNormWeightDtype,
                                     outputNormWeightFormat)
                          .ToAclType();

        aclTensor* partialBlockRefArg = partialBlockRef.get();
        const aclTensor* deltaArg = delta.get();
        const aclTensor* pseudoQueryArg = pseudoQuery.get();
        const aclTensor* numeratorArg = numerator.get();
        const aclTensor* logitMaxArg = logitMax.get();
        const aclTensor* expSumArg = expSum.get();
        aclTensor* hArg = h.get();
        uint64_t workspaceSize = 0U;
        uint64_t* workspaceSizeArg = &workspaceSize;
        aclOpExecutor* executor = nullptr;
        aclOpExecutor** executorArg = &executor;

        if (runMode == RUN_MODE_NULL_WORKSPACE_SIZE) {
            workspaceSizeArg = nullptr;
        } else if (runMode == RUN_MODE_NULL_EXECUTOR_OUT) {
            executorArg = nullptr;
        } else if (runMode == RUN_MODE_NULL_PARTIAL_BLOCK_REF) {
            partialBlockRefArg = nullptr;
        } else if (runMode == RUN_MODE_NULL_DELTA) {
            deltaArg = nullptr;
        } else if (runMode == RUN_MODE_NULL_PSEUDO_QUERY) {
            pseudoQueryArg = nullptr;
        } else if (runMode == RUN_MODE_NULL_NUMERATOR) {
            numeratorArg = nullptr;
        } else if (runMode == RUN_MODE_NULL_LOGIT_MAX) {
            logitMaxArg = nullptr;
        } else if (runMode == RUN_MODE_NULL_EXP_SUM) {
            expSumArg = nullptr;
        } else if (runMode == RUN_MODE_NULL_H) {
            hArg = nullptr;
        } else if (runMode != RUN_MODE_GET_WORKSPACE_V2) {
            ADD_FAILURE() << "Unsupported runMode: " << runMode << ", case=" << caseName;
            return ACLNN_ERR_PARAM_INVALID;
        }

        aclnnStatus ret;
        if (runMode == RUN_MODE_GET_WORKSPACE_V2) {
            const auto storageShape = weight->GetStorageShape();
            const auto originalShape = weight->GetOriginalShape();
            const auto viewShape = weight->GetViewShape();
            const auto strides = weight->GetViewStrides();
            const auto viewOffset = weight->GetViewOffset();
            const auto storageOffset = weight->GetStorageOffset();
            ret = aclnnBlockAttnResUpdateV2GetWorkspaceSize(partialBlockRefArg, deltaArg, pseudoQueryArg, numeratorArg,
                                                            logitMaxArg, expSumArg,
                                                            outputNormWeightSpec == "NONE" ? nullptr : weight.get(),
                                                            eps, outputNormEps, hArg, workspaceSizeArg, executorArg);
            EXPECT_EQ(weight->GetStorageShape(), storageShape);
            EXPECT_EQ(weight->GetOriginalShape(), originalShape);
            EXPECT_EQ(weight->GetViewShape(), viewShape);
            EXPECT_EQ(weight->GetViewStrides(), strides);
            EXPECT_EQ(weight->GetViewOffset(), viewOffset);
            EXPECT_EQ(weight->GetStorageOffset(), storageOffset);
        } else {
            ret = aclnnBlockAttnResUpdateGetWorkspaceSize(partialBlockRefArg, deltaArg, pseudoQueryArg, numeratorArg,
                                                          logitMaxArg, expSumArg, eps, hArg, workspaceSizeArg,
                                                          executorArg);
        }
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
        return ret;
    }

    void Run() const
    {
        if (!enable) {
            GTEST_SKIP() << "Skip disabled case: " << caseName;
        }
        const op::ScopedPlatformNpuArch platformGuard(npuArch);

        const TensorDesc partialBlockRefDesc =
            MakeTensorDesc(partialBlockRefSpec, partialBlockRefDtype, partialBlockRefFormat);
        const TensorDesc deltaDesc = MakeTensorDesc(deltaSpec, deltaDtype, deltaFormat);
        const TensorDesc pseudoQueryDesc = MakeTensorDesc(pseudoQuerySpec, pseudoQueryDtype, pseudoQueryFormat);
        const TensorDesc numeratorDesc = MakeTensorDesc(numeratorSpec, numeratorDtype, numeratorFormat);
        const TensorDesc logitMaxDesc = MakeTensorDesc(logitMaxSpec, logitMaxDtype, logitMaxFormat);
        const TensorDesc expSumDesc = MakeTensorDesc(expSumSpec, expSumDtype, expSumFormat);
        const TensorDesc hDesc = MakeTensorDesc(hSpec, hDtype, hFormat);

        aclnnStatus ret = ACLNN_SUCCESS;
        if (runMode == RUN_MODE_GET_WORKSPACE) {
            auto ut = OP_API_UT(
                aclnnBlockAttnResUpdate,
                INPUT(partialBlockRefDesc, deltaDesc, pseudoQueryDesc, numeratorDesc, logitMaxDesc, expSumDesc, eps),
                OUTPUT(hDesc));
            uint64_t workspaceSize = 0U;
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        } else {
            ret = RunSpecialCase(partialBlockRefDesc, deltaDesc, pseudoQueryDesc, numeratorDesc, logitMaxDesc,
                                 expSumDesc, hDesc);
        }
        EXPECT_EQ(ret, TestUtils::ParseAclnnStatus(expectRet)) << "case=" << caseName;
    }

    std::string platformName;
    NpuArch npuArch = NpuArch::DAV_RESV;
    std::string caseName;
    bool enable = true;
    std::string expectRet;
    std::string partialBlockRefSpec;
    std::string partialBlockRefDtype;
    std::string partialBlockRefFormat;
    std::string deltaSpec;
    std::string deltaDtype;
    std::string deltaFormat;
    std::string pseudoQuerySpec;
    std::string pseudoQueryDtype;
    std::string pseudoQueryFormat;
    std::string numeratorSpec;
    std::string numeratorDtype;
    std::string numeratorFormat;
    std::string logitMaxSpec;
    std::string logitMaxDtype;
    std::string logitMaxFormat;
    std::string expSumSpec;
    std::string expSumDtype;
    std::string expSumFormat;
    double eps = 1.0e-6;
    std::string hSpec;
    std::string hDtype;
    std::string hFormat;
    std::string runMode = RUN_MODE_GET_WORKSPACE;
    std::string outputNormWeightSpec = "NONE";
    std::string outputNormWeightDtype = "FLOAT";
    std::string outputNormWeightFormat = "ND";
    double outputNormEps = 1e-6;
};

std::vector<BlockAttnResUpdateOpApiCase> LoadCases(const std::string& csvFilePath)
{
    std::ifstream csvData(csvFilePath);
    EXPECT_TRUE(csvData.is_open()) << "Failed to open CSV file: " << csvFilePath;
    std::vector<BlockAttnResUpdateOpApiCase> cases;
    std::string line;
    bool headerSkipped = false;
    size_t lineNo = 0U;
    while (std::getline(csvData, line)) {
        ++lineNo;
        if (TestUtils::Trim(line).empty()) {
            continue;
        }
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }

        std::vector<std::string> columns;
        TestUtils::SplitString(line, ",", columns);
        if (columns.size() < CSV_COLUMN_COUNT) {
            ADD_FAILURE() << "Expected at least " << CSV_COLUMN_COUNT << " CSV columns, but got " << columns.size()
                          << " at " << csvFilePath << ":" << lineNo;
            continue;
        }

        const std::string caseName =
            columns.size() > CASE_NAME_COLUMN_INDEX ? TestUtils::Trim(columns[CASE_NAME_COLUMN_INDEX]) : "";
        try {
            BlockAttnResUpdateOpApiCase testCase;
            size_t index = 0U;
            testCase.platformName = TestUtils::Trim(columns[index++]);
            testCase.caseName = TestUtils::Trim(columns[index++]);
            testCase.enable = TestUtils::ParseBool(columns[index++]);
            if (testCase.enable) {
                testCase.npuArch = TestUtils::ParseNpuArch(testCase.platformName);
            }
            testCase.expectRet = TestUtils::Trim(columns[index++]);
            testCase.partialBlockRefSpec = TestUtils::Trim(columns[index++]);
            testCase.partialBlockRefDtype = TestUtils::Trim(columns[index++]);
            testCase.partialBlockRefFormat = TestUtils::Trim(columns[index++]);
            testCase.deltaSpec = TestUtils::Trim(columns[index++]);
            testCase.deltaDtype = TestUtils::Trim(columns[index++]);
            testCase.deltaFormat = TestUtils::Trim(columns[index++]);
            testCase.pseudoQuerySpec = TestUtils::Trim(columns[index++]);
            testCase.pseudoQueryDtype = TestUtils::Trim(columns[index++]);
            testCase.pseudoQueryFormat = TestUtils::Trim(columns[index++]);
            testCase.numeratorSpec = TestUtils::Trim(columns[index++]);
            testCase.numeratorDtype = TestUtils::Trim(columns[index++]);
            testCase.numeratorFormat = TestUtils::Trim(columns[index++]);
            testCase.logitMaxSpec = TestUtils::Trim(columns[index++]);
            testCase.logitMaxDtype = TestUtils::Trim(columns[index++]);
            testCase.logitMaxFormat = TestUtils::Trim(columns[index++]);
            testCase.expSumSpec = TestUtils::Trim(columns[index++]);
            testCase.expSumDtype = TestUtils::Trim(columns[index++]);
            testCase.expSumFormat = TestUtils::Trim(columns[index++]);
            testCase.eps = std::stod(TestUtils::Trim(columns[index++]));
            testCase.hSpec = TestUtils::Trim(columns[index++]);
            testCase.hDtype = TestUtils::Trim(columns[index++]);
            testCase.hFormat = TestUtils::Trim(columns[index++]);
            testCase.runMode = TestUtils::Trim(columns[index++]);
            if (columns.size() >= CSV_COLUMN_COUNT + 4UL) {
                testCase.outputNormWeightSpec = TestUtils::Trim(columns[index++]);
                testCase.outputNormWeightDtype = TestUtils::Trim(columns[index++]);
                testCase.outputNormWeightFormat = TestUtils::Trim(columns[index++]);
                testCase.outputNormEps = std::stod(TestUtils::Trim(columns[index++]));
            }
            cases.emplace_back(testCase);
        } catch (const std::exception& error) {
            ADD_FAILURE() << TestUtils::BuildCsvParseErrorMessage(csvFilePath, lineNo, caseName, error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

std::string BuildCaseName(const testing::TestParamInfo<BlockAttnResUpdateOpApiCase>& info)
{
    return TestUtils::MakeSafeParamName(info.param.platformName + "_" + info.param.caseName);
}

class BlockAttnResUpdateOpApiCsvTest : public testing::TestWithParam<BlockAttnResUpdateOpApiCase> {};

TEST_P(BlockAttnResUpdateOpApiCsvTest, RunCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(BlockAttnResUpdateOpApiCsv, BlockAttnResUpdateOpApiCsvTest,
                         testing::ValuesIn(LoadCases(
                             TestUtils::ResolveCsvPath("test_aclnn_block_attn_res_update.csv",
                                                       "attention/block_attn_res_update/tests/ut/op_api", __FILE__))),
                         BuildCaseName);

} // namespace
