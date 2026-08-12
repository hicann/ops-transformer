/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_matmul_infershape.cpp
 * \brief CSV-driven unit tests for grouped_matmul infershape.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"

#include "base/registry/op_impl_space_registry_v2.h"
#define private public
#include "platform/platform_info.h"

using namespace std;

namespace {

using ops::ut::ParseBool;
using ops::ut::SplitStr2Vec;
using ops::ut::Trim;

ge::DataType ParseDtype(const string &dtype)
{
    return ops::ut::ParseGeDtype(dtype);
}

ge::Format ParseFormat(const string &format)
{
    return ops::ut::ParseGeFormat(format);
}

bool IsAbsentTensorListSpec(const string &value)
{
    const string lower = ops::ut::ToLower(Trim(value));
    return lower == "absent" || lower == "null";
}

vector<string> ParseTensorSpecs(const string &value)
{
    if (IsAbsentTensorListSpec(value)) {
        return {};
    }
    vector<string> specs;
    SplitStr2Vec(value, ";", specs);
    for (auto &spec : specs) {
        spec = Trim(spec);
    }
    return specs;
}

vector<int64_t> ParseInt64List(const string &value)
{
    return ops::ut::ParseI64List(value, ":");
}

uint32_t GetTensorInstanceNum(const string &shapeSpec)
{
    return static_cast<uint32_t>(ParseTensorSpecs(shapeSpec).size());
}

void AppendTensorDescs(vector<gert::InfershapeContextPara::TensorDescription> &descs, const string &shapeSpec,
                       ge::DataType dtype, ge::Format format, bool isConst = false, void *constValue = nullptr)
{
    for (const auto &spec : ParseTensorSpecs(shapeSpec)) {
        descs.emplace_back(ops::ut::MakeGertStorageShape(ops::ut::ParseDims(spec, {}, Trim(spec) == "NONE")),
                           dtype, format, isConst, constValue);
    }
}

void SetupPlatformForCase(const string &socVersion)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = socVersion;
    platformInfo.str_info.short_soc_version = socVersion;
    fe::PlatformInfoManager::Instance().platform_info_map_[socVersion] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}

struct GroupedMatmulInfershapeCase {
    bool ShouldRunInferShape() const
    {
        return caseType.find("dtype_only") == string::npos;
    }

    bool ShouldRunInferDtype() const
    {
        return caseType.find("dtype_only") != string::npos;
    }

    vector<gert::InfershapeContextPara::OpAttr> BuildShapeAttrs() const
    {
        return {
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(splitItem)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outputDtypeAttr)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeX)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupType)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(actType)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(GetTuningConfig())},
        };
    }

    vector<pair<string, Ops::Transformer::AnyValue>> BuildDtypeAttrs() const
    {
        return {
            {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(splitItem)},
            {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outputDtypeAttr)},
            {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
            {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeX)},
            {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupType)},
            {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)},
            {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(actType)},
            {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(GetTuningConfig())},
        };
    }

    vector<int64_t> GetTuningConfig() const
    {
        vector<int64_t> tuningConfig = ParseInt64List(tuningConfigData);
        return tuningConfig.empty() ? vector<int64_t>{0} : tuningConfig;
    }

    void RunInferShape() const
    {
        SetupPlatformForCase(socVersion);
        vector<int64_t> groupListValues = ParseInt64List(groupListData);
        void *groupListValuePtr = groupListValues.empty() ? nullptr : static_cast<void *>(groupListValues.data());
        vector<gert::InfershapeContextPara::TensorDescription> inputTensorDescs;
        AppendTensorDescs(inputTensorDescs, xShape, ParseDtype(xDtype), ParseFormat(xFormat));
        AppendTensorDescs(inputTensorDescs, weightShape, ParseDtype(weightDtype), ParseFormat(weightFormat));
        AppendTensorDescs(inputTensorDescs, biasShape, ParseDtype(biasDtype), ParseFormat(biasFormat));
        AppendTensorDescs(inputTensorDescs, scaleShape, ParseDtype(scaleDtype), ParseFormat(scaleFormat));
        AppendTensorDescs(inputTensorDescs, offsetShape, ParseDtype(offsetDtype), ParseFormat(offsetFormat));
        AppendTensorDescs(inputTensorDescs, antiquantScaleShape, ParseDtype(antiquantScaleDtype),
                          ParseFormat(antiquantScaleFormat));
        AppendTensorDescs(inputTensorDescs, antiquantOffsetShape, ParseDtype(antiquantOffsetDtype),
                          ParseFormat(antiquantOffsetFormat));
        AppendTensorDescs(inputTensorDescs, groupListShape, ParseDtype(groupListDtype), ParseFormat(groupListFormat),
                          groupListValuePtr != nullptr, groupListValuePtr);
        AppendTensorDescs(inputTensorDescs, perTokenScaleShape, ParseDtype(perTokenScaleDtype),
                          ParseFormat(perTokenScaleFormat));

        vector<string> outputSpecs = ParseTensorSpecs(expectOutputShape);
        if (outputSpecs.empty()) {
            outputSpecs.emplace_back("");
        }
        vector<gert::InfershapeContextPara::TensorDescription> outputTensorDescs;
        vector<vector<int64_t>> expectShape;
        for (const auto &outputSpec : outputSpecs) {
            outputTensorDescs.emplace_back(ops::ut::MakeGertStorageShape(ops::ut::ParseDims("")), ParseDtype(outDtype),
                                           ParseFormat(outFormat));
            expectShape.push_back(ops::ut::ParseDims(outputSpec, {}, Trim(outputSpec) == "NONE"));
        }

        vector<uint32_t> inputInstanceNum = {
            GetTensorInstanceNum(xShape),
            GetTensorInstanceNum(weightShape),
            GetTensorInstanceNum(biasShape),
            GetTensorInstanceNum(scaleShape),
            GetTensorInstanceNum(offsetShape),
            GetTensorInstanceNum(antiquantScaleShape),
            GetTensorInstanceNum(antiquantOffsetShape),
            GetTensorInstanceNum(groupListShape),
            GetTensorInstanceNum(perTokenScaleShape),
        };
        vector<uint32_t> outputInstanceNum = {static_cast<uint32_t>(outputSpecs.size())};
        auto usesDynamicInstanceNum =
            any_of(inputInstanceNum.begin(), inputInstanceNum.end(), [](uint32_t count) { return count != 1U; }) ||
            outputInstanceNum.front() != 1U;

        if (usesDynamicInstanceNum) {
            gert::InfershapeContextPara infershapeContextPara(
                "GroupedMatmul", inputTensorDescs, outputTensorDescs, BuildShapeAttrs(), inputInstanceNum,
                outputInstanceNum);
            ExecuteTestCase(infershapeContextPara, expectSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED, expectShape);
            return;
        }

        gert::InfershapeContextPara infershapeContextPara(
            "GroupedMatmul", inputTensorDescs, outputTensorDescs, BuildShapeAttrs());
        ExecuteTestCase(infershapeContextPara, expectSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED, expectShape);
    }

    void RunInferDtype() const
    {
        SetupPlatformForCase(socVersion);
        auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
        ASSERT_NE(spaceRegistry, nullptr);
        auto opImpl = spaceRegistry->GetOpImpl("GroupedMatmul");
        ASSERT_NE(opImpl, nullptr);
        ASSERT_NE(opImpl->infer_datatype, nullptr);

        ge::DataType xDataType = ParseDtype(xDtype);
        ge::DataType weightDataType = ParseDtype(weightDtype);
        ge::DataType biasDataType = ParseDtype(biasDtype);
        ge::DataType scaleDataType = ParseDtype(scaleDtype);
        ge::DataType offsetDataType = ParseDtype(offsetDtype);
        ge::DataType antiquantScaleDataType = ParseDtype(antiquantScaleDtype);
        ge::DataType antiquantOffsetDataType = ParseDtype(antiquantOffsetDtype);
        ge::DataType groupListDataType = ParseDtype(groupListDtype);
        ge::DataType perTokenScaleDataType = ParseDtype(perTokenScaleDtype);
        vector<void *> inputDataTypes = {
            &xDataType,
            &weightDataType,
            &biasDataType,
            &scaleDataType,
            &offsetDataType,
            &antiquantScaleDataType,
            &antiquantOffsetDataType,
            &groupListDataType,
            &perTokenScaleDataType,
        };

        auto contextHolder = gert::InferDataTypeContextFaker()
                                 .SetOpType("GroupedMatmul")
                                 .NodeIoNum(9, 1)
                                 .NodeOutputTd(0, ParseFormat(outFormat), ParseFormat(outFormat))
                                 .InputDataTypes(inputDataTypes)
                                 .NodeAttrs(BuildDtypeAttrs())
                                 .Build();
        auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        auto inferDtypeRet = opImpl->infer_datatype(context);
        ASSERT_EQ(inferDtypeRet, expectSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED);
        if (inferDtypeRet != ge::GRAPH_SUCCESS) {
            return;
        }
        EXPECT_EQ(context->GetOutputDataType(0), ParseDtype(outDtype));
    }

    void Run() const
    {
        if (ShouldRunInferShape()) {
            RunInferShape();
        }
        if (ShouldRunInferDtype()) {
            RunInferDtype();
        }
    }
    string socVersion;
    string caseName;
    bool enable = true;
    string prefix;
    string caseType;
    bool expectSuccess = false;
    string expectOutputShape;

    string xShape;
    string weightShape;
    string biasShape;
    string scaleShape;
    string offsetShape;
    string antiquantScaleShape;
    string antiquantOffsetShape;
    string groupListShape;
    string perTokenScaleShape;

    string xDtype;
    string weightDtype;
    string biasDtype;
    string scaleDtype;
    string offsetDtype;
    string antiquantScaleDtype;
    string antiquantOffsetDtype;
    string groupListDtype;
    string perTokenScaleDtype;
    string outDtype;

    string xFormat;
    string weightFormat;
    string biasFormat;
    string scaleFormat;
    string offsetFormat;
    string antiquantScaleFormat;
    string antiquantOffsetFormat;
    string groupListFormat;
    string perTokenScaleFormat;
    string outFormat;

    int64_t splitItem = 3;
    int64_t outputDtypeAttr = 0;
    bool transposeWeight = false;
    bool transposeX = false;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;
    string groupListData;
    string tuningConfigData;
};

vector<GroupedMatmulInfershapeCase> LoadCases(const string &socVersion)
{
    vector<GroupedMatmulInfershapeCase> cases;
    string csvPath = ops::ut::ResolveCsvPath("test_grouped_matmul_infershape.csv",
                                             "gmm/grouped_matmul/tests/ut/op_host", __FILE__);
    ifstream csvData(csvPath, ios::in);
    if (!csvData.is_open()) {
        cout << "cannot open case file " << csvPath << endl;
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
        if (items.empty() || items[0] == "socVersion" || items.size() < 43U) {
            continue;
        }

        const string caseName = items.size() > 1U ? Trim(items[1]) : "";
        try {
            size_t idx = 0;
            GroupedMatmulInfershapeCase tc;
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
            tc.caseType = Trim(items[idx++]);
            tc.expectSuccess = ParseBool(Trim(items[idx++]));
            tc.expectOutputShape = Trim(items[idx++]);

            tc.xShape = Trim(items[idx++]);
            tc.weightShape = Trim(items[idx++]);
            tc.biasShape = Trim(items[idx++]);
            tc.scaleShape = Trim(items[idx++]);
            tc.offsetShape = Trim(items[idx++]);
            tc.antiquantScaleShape = Trim(items[idx++]);
            tc.antiquantOffsetShape = Trim(items[idx++]);
            tc.groupListShape = Trim(items[idx++]);
            tc.perTokenScaleShape = Trim(items[idx++]);

            tc.xDtype = Trim(items[idx++]);
            tc.weightDtype = Trim(items[idx++]);
            tc.biasDtype = Trim(items[idx++]);
            tc.scaleDtype = Trim(items[idx++]);
            tc.offsetDtype = Trim(items[idx++]);
            tc.antiquantScaleDtype = Trim(items[idx++]);
            tc.antiquantOffsetDtype = Trim(items[idx++]);
            tc.groupListDtype = Trim(items[idx++]);
            tc.perTokenScaleDtype = Trim(items[idx++]);
            tc.outDtype = Trim(items[idx++]);

            tc.xFormat = Trim(items[idx++]);
            tc.weightFormat = Trim(items[idx++]);
            tc.biasFormat = Trim(items[idx++]);
            tc.scaleFormat = Trim(items[idx++]);
            tc.offsetFormat = Trim(items[idx++]);
            tc.antiquantScaleFormat = Trim(items[idx++]);
            tc.antiquantOffsetFormat = Trim(items[idx++]);
            tc.groupListFormat = Trim(items[idx++]);
            tc.perTokenScaleFormat = Trim(items[idx++]);
            tc.outFormat = Trim(items[idx++]);

            tc.splitItem = stoll(Trim(items[idx++]));
            tc.outputDtypeAttr = stoll(Trim(items[idx++]));
            tc.transposeWeight = ParseBool(Trim(items[idx++]));
            tc.transposeX = ParseBool(Trim(items[idx++]));
            tc.groupType = stoll(Trim(items[idx++]));
            tc.groupListType = stoll(Trim(items[idx++]));
            tc.actType = stoll(Trim(items[idx++]));
            if (idx < items.size()) {
                tc.groupListData = Trim(items[idx++]);
            }
            if (idx < items.size()) {
                tc.tuningConfigData = Trim(items[idx++]);
            }
            cases.push_back(tc);
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, caseName, error);
        }
    }
    return cases;
}

string MakeParamName(const testing::TestParamInfo<GroupedMatmulInfershapeCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

} // namespace

namespace GroupedMatmulInfershapeUT {

const vector<GroupedMatmulInfershapeCase> &GetGroupedMatmulInfershapeCsvCases()
{
    static const vector<GroupedMatmulInfershapeCase> cases = [] {
        vector<GroupedMatmulInfershapeCase> merged = LoadCases("Ascend950");
        vector<GroupedMatmulInfershapeCase> ascend910b = LoadCases("Ascend910B");
        merged.insert(merged.end(), ascend910b.begin(), ascend910b.end());
        return merged;
    }();
    return cases;
}

class TestGroupedMatmulInfershape : public testing::TestWithParam<GroupedMatmulInfershapeCase> {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_P(TestGroupedMatmulInfershape, csvDrivenCase)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GROUPED_MATMUL_INFERSHAPE_CSV, TestGroupedMatmulInfershape,
                         testing::ValuesIn(GetGroupedMatmulInfershapeCsvCases()), MakeParamName);

} // namespace GroupedMatmulInfershapeUT
