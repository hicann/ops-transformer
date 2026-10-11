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

#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "infer_shape_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"

using namespace std;

namespace {

using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

constexpr size_t kCsvColumnCount = 20U;

struct QGmmInplaceAddInfershapeCase {
    string socVersion;
    string caseName;
    bool enable = true;
    string prefix;
    bool expectSuccess = false;
    string expectOutputShape;
    bool hasScale1 = true;
    string x1Shape;
    string x1Dtype;
    string x2Shape;
    string x2Dtype;
    string scale2Shape;
    string scale2Dtype;
    string groupListShape;
    string yRefShape;
    string yRefDtype;
    string scale1Shape;
    string scale1Dtype;
    int64_t groupListType = 0;
    int64_t groupSize = 0;

    gert::InfershapeContextPara BuildContext() const
    {
        if (hasScale1) {
            return gert::InfershapeContextPara(
                "QuantGroupedMatmulInplaceAdd",
                {
                    {ops::ut::MakeGertStorageShape(x1Shape), ops::ut::ParseGeDtype(x1Dtype), ge::FORMAT_ND},
                    {ops::ut::MakeGertStorageShape(x2Shape), ops::ut::ParseGeDtype(x2Dtype), ge::FORMAT_ND},
                    {ops::ut::MakeGertStorageShape(scale2Shape), ops::ut::ParseGeDtype(scale2Dtype), ge::FORMAT_ND},
                    {ops::ut::MakeGertStorageShape(groupListShape), ge::DT_INT64, ge::FORMAT_ND},
                    {ops::ut::MakeGertStorageShape(yRefShape), ops::ut::ParseGeDtype(yRefDtype), ge::FORMAT_ND},
                    {ops::ut::MakeGertStorageShape(scale1Shape), ops::ut::ParseGeDtype(scale1Dtype), ge::FORMAT_ND},
                },
                {{{{}, {}}, ops::ut::ParseGeDtype(yRefDtype), ge::FORMAT_ND}},
                {{"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
                 {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupSize)}});
        }

        return gert::InfershapeContextPara(
            "QuantGroupedMatmulInplaceAdd",
            {
                {ops::ut::MakeGertStorageShape(x1Shape), ops::ut::ParseGeDtype(x1Dtype), ge::FORMAT_ND},
                {ops::ut::MakeGertStorageShape(x2Shape), ops::ut::ParseGeDtype(x2Dtype), ge::FORMAT_ND},
                {ops::ut::MakeGertStorageShape(scale2Shape), ops::ut::ParseGeDtype(scale2Dtype), ge::FORMAT_ND},
                {ops::ut::MakeGertStorageShape(groupListShape), ge::DT_INT64, ge::FORMAT_ND},
                {ops::ut::MakeGertStorageShape(yRefShape), ops::ut::ParseGeDtype(yRefDtype), ge::FORMAT_ND},
            },
            {{{{}, {}}, ops::ut::ParseGeDtype(yRefDtype), ge::FORMAT_ND}},
            {{"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
             {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupSize)}});
    }

    void Run() const
    {
        if (!enable) {
            GTEST_SKIP() << "Skip disabled case: " << caseName;
        }
        auto para = BuildContext();
        ExecuteTestCase(para, expectSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED,
                        ops::ut::ParseDimsList(expectOutputShape));
    }
};

vector<QGmmInplaceAddInfershapeCase> LoadCases(const string& socVersion)
{
    vector<QGmmInplaceAddInfershapeCase> cases;
    string csvPath = ops::ut::ResolveCsvPath("test_quant_grouped_matmul_inplace_add_infershape.csv",
                                             "gmm/quant_grouped_matmul_inplace_add/tests/ut/op_host", __FILE__);
    ifstream csvData(csvPath, ios::in);
    if (!csvData.is_open()) {
        cout << "cannot open case file " << csvPath << ", maybe not exist" << endl;
        return cases;
    }

    string line;
    size_t lineNo = 0U;
    while (getline(csvData, line)) {
        ++lineNo;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        vector<string> items;
        SplitStr2Vec(line, ",", items);
        if (items.empty() || items[0] == "socVersion" || items.size() < kCsvColumnCount) {
            continue;
        }

        const string caseName = items.size() > 1U ? Trim(items[1]) : "";
        try {
            size_t idx = 0;
            QGmmInplaceAddInfershapeCase tc;
            tc.socVersion = Trim(items[idx++]);
            if (tc.socVersion != socVersion) {
                continue;
            }
            tc.caseName = Trim(items[idx++]);
            tc.enable = ParseBool(Trim(items[idx++]));
            if (!tc.enable) {
                continue;
            }
            tc.prefix = Trim(items[idx++]);
            tc.expectSuccess = ParseBool(Trim(items[idx++]));
            tc.expectOutputShape = Trim(items[idx++]);
            tc.hasScale1 = ParseBool(Trim(items[idx++]));
            tc.x1Shape = Trim(items[idx++]);
            tc.x1Dtype = Trim(items[idx++]);
            tc.x2Shape = Trim(items[idx++]);
            tc.x2Dtype = Trim(items[idx++]);
            tc.scale2Shape = Trim(items[idx++]);
            tc.scale2Dtype = Trim(items[idx++]);
            tc.groupListShape = Trim(items[idx++]);
            tc.yRefShape = Trim(items[idx++]);
            tc.yRefDtype = Trim(items[idx++]);
            tc.scale1Shape = Trim(items[idx++]);
            tc.scale1Dtype = Trim(items[idx++]);
            tc.groupListType = stoll(Trim(items[idx++]));
            tc.groupSize = stoll(Trim(items[idx++]));
            cases.push_back(tc);
        } catch (const std::exception& error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvPath;
    return cases;
}

string MakeParamName(const testing::TestParamInfo<QGmmInplaceAddInfershapeCase>& info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

} // namespace

namespace QuantGroupedMatmulInplaceAddInfershapeUT {

const vector<QGmmInplaceAddInfershapeCase>& GetAscend950Params()
{
    static const vector<QGmmInplaceAddInfershapeCase> params = LoadCases("Ascend950");
    return params;
}

class TestQuantGroupedMatmulInplaceAddInfershape : public testing::TestWithParam<QGmmInplaceAddInfershapeCase> {};

TEST_P(TestQuantGroupedMatmulInplaceAddInfershape, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(QUANT_GROUPED_MATMUL_INPLACE_ADD_INFERSHAPE_CSV, TestQuantGroupedMatmulInplaceAddInfershape,
                         testing::ValuesIn(GetAscend950Params()), MakeParamName);

} // namespace QuantGroupedMatmulInplaceAddInfershapeUT
