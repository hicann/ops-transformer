/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_api/aclnn_quant_grouped_matmul_inplace_add.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "gmm_csv_acl_parse_utils.h"
#include "gmm_soc_version_utils.h"

using namespace std;

namespace {

using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr size_t kCsvColumnCount = 22;
constexpr const char *kRunModeGetWorkspace = "GET_WORKSPACE";
constexpr const char *kRunModeScale2Null = "SCALE2_NULL";
constexpr const char *kRunModeGroupListNull = "GROUP_LIST_NULL";
constexpr const char *kRunModeYNull = "Y_NULL";

vector<int64_t> ParseDims(const string &value)
{
    return ops::ut::ParseAclTensorViewDims(value);
}

struct QGmmInplaceAddOpApiCase {
    void Run() const
    {
        if (!enable) {
            GTEST_SKIP() << "Skip disabled case: " << caseName;
        }
        ops::ut::SetPlatformSocVersion(socVersion);

        TensorDesc x1Desc(ParseDims(x1Shape), ops::ut::ParseAclDtype(x1Dtype), ACL_FORMAT_ND, ParseDims(x1Stride));
        TensorDesc x2Desc(ParseDims(x2Shape), ops::ut::ParseAclDtype(x2Dtype), ACL_FORMAT_ND, ParseDims(x2Stride));
        TensorDesc scale2Desc(ParseDims(scale2Shape), ops::ut::ParseAclDtype(scale2Dtype), ACL_FORMAT_ND);
        TensorDesc yDesc(ParseDims(yShape), ops::ut::ParseAclDtype(yDtype), ACL_FORMAT_ND);
        TensorDesc scale1Desc(ParseDims(scale1Shape), ops::ut::ParseAclDtype(scale1Dtype), ACL_FORMAT_ND,
                              ParseDims(scale1Stride));
        TensorDesc groupListDesc(ParseDims(groupListShape), ACL_INT64, ACL_FORMAT_ND);
        const vector<int64_t> groupVals = ops::ut::ParseI64List(groupListValues);
        if (!groupVals.empty()) {
            groupListDesc.Value(groupVals);
        }

        uint64_t workspaceSize = 0;
        aclnnStatus ret = ACLNN_SUCCESS;
        if (runMode == kRunModeScale2Null) {
            auto ut = OP_API_UT(
                aclnnQuantGroupedMatmulInplaceAdd,
                INPUT(x1Desc, x2Desc, scale1Desc, nullptr, groupListDesc, yDesc, groupListType, groupSize), OUTPUT());
            aclOpExecutor *executor = nullptr;
            ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
        } else if (runMode == kRunModeGroupListNull) {
            auto ut = OP_API_UT(aclnnQuantGroupedMatmulInplaceAdd,
                                INPUT(x1Desc, x2Desc, scale1Desc, scale2Desc, nullptr, yDesc, groupListType, groupSize),
                                OUTPUT());
            aclOpExecutor *executor = nullptr;
            ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
        } else if (runMode == kRunModeYNull) {
            auto ut = OP_API_UT(
                aclnnQuantGroupedMatmulInplaceAdd,
                INPUT(x1Desc, x2Desc, scale1Desc, scale2Desc, groupListDesc, nullptr, groupListType, groupSize),
                OUTPUT());
            aclOpExecutor *executor = nullptr;
            ret = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
        } else {
            auto ut =
                OP_API_UT(aclnnQuantGroupedMatmulInplaceAdd,
                          INPUT(x1Desc, x2Desc, scale1Desc, scale2Desc, groupListDesc, yDesc, groupListType, groupSize),
                          OUTPUT());
            ret = ut.TestGetWorkspaceSize(&workspaceSize);
        }
        if (checkRet) {
            EXPECT_EQ(ret, ops::ut::ParseAclnnStatus(expectRet)) << "case=" << caseName;
        }
    }

    string socVersion;
    string caseName;
    bool enable = true;
    string x1Shape;
    string x1Dtype;
    string x1Stride;
    string x2Shape;
    string x2Dtype;
    string x2Stride;
    string scale2Shape;
    string scale2Dtype;
    string yShape;
    string yDtype;
    string scale1Shape;
    string scale1Dtype;
    string scale1Stride;
    string groupListShape;
    string groupListValues;
    int64_t groupListType = 0;
    int64_t groupSize = 0;
    string expectRet;
    bool checkRet = true;
    string runMode = kRunModeGetWorkspace;
};

vector<QGmmInplaceAddOpApiCase> LoadCases(const string &csvFilePath)
{
    ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;
    vector<QGmmInplaceAddOpApiCase> cases;
    string line;
    bool headerSkipped = false;
    size_t lineNo = 0U;
    while (getline(in, line)) {
        ++lineNo;
        if (line.empty()) {
            continue;
        }
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }
        vector<string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() < kCsvColumnCount) {
            continue;
        }

        const string caseName = cols.size() > 1U ? Trim(cols[1]) : "";
        try {
            QGmmInplaceAddOpApiCase c;
            size_t i = 0;
            c.socVersion = Trim(cols[i++]);
            c.caseName = Trim(cols[i++]);
            c.enable = ParseBool(Trim(cols[i++]));
            c.x1Shape = Trim(cols[i++]);
            c.x1Dtype = Trim(cols[i++]);
            c.x1Stride = Trim(cols[i++]);
            c.x2Shape = Trim(cols[i++]);
            c.x2Dtype = Trim(cols[i++]);
            c.x2Stride = Trim(cols[i++]);
            c.scale2Shape = Trim(cols[i++]);
            c.scale2Dtype = Trim(cols[i++]);
            c.yShape = Trim(cols[i++]);
            c.yDtype = Trim(cols[i++]);
            c.scale1Shape = Trim(cols[i++]);
            c.scale1Dtype = Trim(cols[i++]);
            c.scale1Stride = Trim(cols[i++]);
            c.groupListShape = Trim(cols[i++]);
            c.groupListValues = Trim(cols[i++]);
            c.groupListType = stoll(Trim(cols[i++]));
            c.groupSize = stoll(Trim(cols[i++]));
            c.expectRet = Trim(cols[i++]);
            c.checkRet = ParseBool(Trim(cols[i++]));
            if (i < cols.size()) {
                c.runMode = Trim(cols[i++]);
            }
            cases.emplace_back(c);
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvFilePath, lineNo, caseName, error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

string BuildCaseName(const testing::TestParamInfo<QGmmInplaceAddOpApiCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

class qgmm_inplace_add_opapi_csv_test : public testing::TestWithParam<QGmmInplaceAddOpApiCase> {};

TEST_P(qgmm_inplace_add_opapi_csv_test, run_case)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(qgmm_inplace_add_opapi_csv, qgmm_inplace_add_opapi_csv_test,
                         testing::ValuesIn(LoadCases(ops::ut::ResolveCsvPath(
                             "test_aclnn_quant_grouped_matmul_inplace_add.csv",
                             "gmm/quant_grouped_matmul_inplace_add/tests/ut/op_host/op_api", __FILE__))),
                         BuildCaseName);

TEST(qgmm_inplace_add_opapi_direct_test, phase2_null_executor_path)
{
    // Cover phase2 API path: L2_DFX_PHASE_2 + CommonOpExecutorRun check branch.
    aclnnStatus ret = aclnnQuantGroupedMatmulInplaceAdd(nullptr, 0, nullptr, nullptr);
    EXPECT_NE(ret, ACLNN_SUCCESS);
}

} // namespace
