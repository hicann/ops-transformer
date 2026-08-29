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
#include "block_attn_res_update_test_utils.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

namespace {

namespace test_utils = block_attn_res_update::test;

constexpr size_t kCsvColumnCount = 27UL;
constexpr size_t kCaseNameColumnIndex = 1UL;
constexpr const char *kRunModeGetWorkspace = "GET_WORKSPACE";
constexpr const char *kRunModeNullWorkspaceSize = "NULL_WORKSPACE_SIZE";
constexpr const char *kRunModeNullExecutorOut = "NULL_EXECUTOR_OUT";
constexpr const char *kRunModeNullPartialBlockRef = "NULL_PARTIAL_BLOCK_REF";
constexpr const char *kRunModeNullDelta = "NULL_DELTA";
constexpr const char *kRunModeNullPseudoQuery = "NULL_PSEUDO_QUERY";
constexpr const char *kRunModeNullNumerator = "NULL_NUMERATOR";
constexpr const char *kRunModeNullLogitMax = "NULL_LOGIT_MAX";
constexpr const char *kRunModeNullExpSum = "NULL_EXP_SUM";
constexpr const char *kRunModeNullH = "NULL_H";
constexpr const char *kRunModeBlockAttnResUpdateNullExecutor = "BLOCK_ATTN_RES_UPDATE_NULL_EXECUTOR";

TensorDesc MakeTensorDesc(const string &spec, const string &dtype, const string &format)
{
    const auto parsed = test_utils::ParseAclTensorSpec(spec);
    return TensorDesc(parsed.viewDims, test_utils::ParseAclDtype(dtype), test_utils::ParseAclFormat(format),
                      parsed.stride, 0, parsed.storageDims)
        .ValueRange(-10, 10);
}

struct BlockAttnResUpdateOpApiCase {
    aclnnStatus RunSpecialCase(const TensorDesc &partialBlockRefDesc, const TensorDesc &deltaDesc,
                               const TensorDesc &pseudoQueryDesc, const TensorDesc &numeratorDesc,
                               const TensorDesc &logitMaxDesc, const TensorDesc &expSumDesc,
                               const TensorDesc &hDesc) const
    {
        if (runMode == kRunModeBlockAttnResUpdateNullExecutor) {
            return aclnnBlockAttnResUpdate(nullptr, 0U, nullptr, nullptr);
        }

        auto partialBlockRef = partialBlockRefDesc.ToAclType();
        auto delta = deltaDesc.ToAclType();
        auto pseudoQuery = pseudoQueryDesc.ToAclType();
        auto numerator = numeratorDesc.ToAclType();
        auto logitMax = logitMaxDesc.ToAclType();
        auto expSum = expSumDesc.ToAclType();
        auto h = hDesc.ToAclType();

        aclTensor *partialBlockRefArg = partialBlockRef.get();
        const aclTensor *deltaArg = delta.get();
        const aclTensor *pseudoQueryArg = pseudoQuery.get();
        const aclTensor *numeratorArg = numerator.get();
        const aclTensor *logitMaxArg = logitMax.get();
        const aclTensor *expSumArg = expSum.get();
        aclTensor *hArg = h.get();
        uint64_t workspaceSize = 0U;
        uint64_t *workspaceSizeArg = &workspaceSize;
        aclOpExecutor *executor = nullptr;
        aclOpExecutor **executorArg = &executor;

        if (runMode == kRunModeNullWorkspaceSize) {
            workspaceSizeArg = nullptr;
        } else if (runMode == kRunModeNullExecutorOut) {
            executorArg = nullptr;
        } else if (runMode == kRunModeNullPartialBlockRef) {
            partialBlockRefArg = nullptr;
        } else if (runMode == kRunModeNullDelta) {
            deltaArg = nullptr;
        } else if (runMode == kRunModeNullPseudoQuery) {
            pseudoQueryArg = nullptr;
        } else if (runMode == kRunModeNullNumerator) {
            numeratorArg = nullptr;
        } else if (runMode == kRunModeNullLogitMax) {
            logitMaxArg = nullptr;
        } else if (runMode == kRunModeNullExpSum) {
            expSumArg = nullptr;
        } else if (runMode == kRunModeNullH) {
            hArg = nullptr;
        } else {
            ADD_FAILURE() << "Unsupported runMode: " << runMode << ", case=" << caseName;
            return ACLNN_ERR_PARAM_INVALID;
        }

        const aclnnStatus ret =
            aclnnBlockAttnResUpdateGetWorkspaceSize(partialBlockRefArg, deltaArg, pseudoQueryArg, numeratorArg,
                                                    logitMaxArg, expSumArg, eps, hArg, workspaceSizeArg, executorArg);
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
        test_utils::SetPlatformSocVersion(socVersion);

        const TensorDesc partialBlockRefDesc =
            MakeTensorDesc(partialBlockRefSpec, partialBlockRefDtype, partialBlockRefFormat);
        const TensorDesc deltaDesc = MakeTensorDesc(deltaSpec, deltaDtype, deltaFormat);
        const TensorDesc pseudoQueryDesc = MakeTensorDesc(pseudoQuerySpec, pseudoQueryDtype, pseudoQueryFormat);
        const TensorDesc numeratorDesc = MakeTensorDesc(numeratorSpec, numeratorDtype, numeratorFormat);
        const TensorDesc logitMaxDesc = MakeTensorDesc(logitMaxSpec, logitMaxDtype, logitMaxFormat);
        const TensorDesc expSumDesc = MakeTensorDesc(expSumSpec, expSumDtype, expSumFormat);
        const TensorDesc hDesc = MakeTensorDesc(hSpec, hDtype, hFormat);

        aclnnStatus ret = ACLNN_SUCCESS;
        if (runMode == kRunModeGetWorkspace) {
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
        EXPECT_EQ(ret, test_utils::ParseAclnnStatus(expectRet)) << "case=" << caseName;
    }

    string socVersion;
    string caseName;
    bool enable = true;
    string expectRet;
    string partialBlockRefSpec;
    string partialBlockRefDtype;
    string partialBlockRefFormat;
    string deltaSpec;
    string deltaDtype;
    string deltaFormat;
    string pseudoQuerySpec;
    string pseudoQueryDtype;
    string pseudoQueryFormat;
    string numeratorSpec;
    string numeratorDtype;
    string numeratorFormat;
    string logitMaxSpec;
    string logitMaxDtype;
    string logitMaxFormat;
    string expSumSpec;
    string expSumDtype;
    string expSumFormat;
    float eps = 1.0e-6F;
    string hSpec;
    string hDtype;
    string hFormat;
    string runMode = kRunModeGetWorkspace;
};

vector<BlockAttnResUpdateOpApiCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<BlockAttnResUpdateOpApiCase> cases;
    string line;
    bool headerSkipped = false;
    size_t lineNo = 0U;
    while (getline(in, line)) {
        ++lineNo;
        if (test_utils::Trim(line).empty()) {
            continue;
        }
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }

        vector<string> cols;
        test_utils::SplitString(line, ",", cols);
        if (cols.size() < kCsvColumnCount) {
            ADD_FAILURE() << "Expected at least " << kCsvColumnCount << " CSV columns, but got " << cols.size()
                          << " at " << csvFilePath << ":" << lineNo;
            continue;
        }

        const string caseName = cols.size() > kCaseNameColumnIndex ? test_utils::Trim(cols[kCaseNameColumnIndex]) : "";
        try {
            BlockAttnResUpdateOpApiCase c;
            size_t i = 0U;
            c.socVersion = test_utils::Trim(cols[i++]);
            c.caseName = test_utils::Trim(cols[i++]);
            c.enable = test_utils::ParseBool(cols[i++]);
            c.expectRet = test_utils::Trim(cols[i++]);
            c.partialBlockRefSpec = test_utils::Trim(cols[i++]);
            c.partialBlockRefDtype = test_utils::Trim(cols[i++]);
            c.partialBlockRefFormat = test_utils::Trim(cols[i++]);
            c.deltaSpec = test_utils::Trim(cols[i++]);
            c.deltaDtype = test_utils::Trim(cols[i++]);
            c.deltaFormat = test_utils::Trim(cols[i++]);
            c.pseudoQuerySpec = test_utils::Trim(cols[i++]);
            c.pseudoQueryDtype = test_utils::Trim(cols[i++]);
            c.pseudoQueryFormat = test_utils::Trim(cols[i++]);
            c.numeratorSpec = test_utils::Trim(cols[i++]);
            c.numeratorDtype = test_utils::Trim(cols[i++]);
            c.numeratorFormat = test_utils::Trim(cols[i++]);
            c.logitMaxSpec = test_utils::Trim(cols[i++]);
            c.logitMaxDtype = test_utils::Trim(cols[i++]);
            c.logitMaxFormat = test_utils::Trim(cols[i++]);
            c.expSumSpec = test_utils::Trim(cols[i++]);
            c.expSumDtype = test_utils::Trim(cols[i++]);
            c.expSumFormat = test_utils::Trim(cols[i++]);
            c.eps = stof(test_utils::Trim(cols[i++]));
            c.hSpec = test_utils::Trim(cols[i++]);
            c.hDtype = test_utils::Trim(cols[i++]);
            c.hFormat = test_utils::Trim(cols[i++]);
            c.runMode = test_utils::Trim(cols[i++]);
            cases.emplace_back(c);
        } catch (const std::exception &error) {
            ADD_FAILURE() << test_utils::BuildCsvParseErrorMessage(csvFilePath, lineNo, caseName, error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<BlockAttnResUpdateOpApiCase> &info)
{
    return test_utils::MakeSafeParamName(info.param.socVersion + "_" + info.param.caseName);
}

class BlockAttnResUpdateOpApiCsvTest : public testing::TestWithParam<BlockAttnResUpdateOpApiCase> {};

TEST_P(BlockAttnResUpdateOpApiCsvTest, RunCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(BlockAttnResUpdateOpApiCsv, BlockAttnResUpdateOpApiCsvTest,
                         testing::ValuesIn(LoadCases(
                             test_utils::ResolveCsvPath("test_aclnn_block_attn_res_update.csv",
                                                        "attention/block_attn_res_update/tests/ut/op_api", __FILE__))),
                         BuildCaseName);

} // namespace
