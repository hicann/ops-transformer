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
 * \file test_grouped_matmul_finalize_routing_infershape.cpp
 * \brief CSV-driven unit tests for grouped_matmul_finalize_routing infershape & inferdtype.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "base/registry/op_impl_space_registry_v2.h"
#include "csv_case_load_utils.h"
#include "gmm_csv_ge_parse_utils.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

using namespace std;

namespace {

using ops::ut::MakeGertStorageShape;
using ops::ut::ParseBool;
using ops::ut::ParseDims;
using ops::ut::ParseGeDtype;
using ops::ut::ParseGeFormat;
using ops::ut::ParseScalarToken;
using ops::ut::ResolveCsvPath;
using ops::ut::Trim;

static float ParseFloatDefault(const string &v, float d = 0.0F)
{
    string t = Trim(v);
    return t.empty() ? d : stof(t);
}

static const char *K_IN_NAMES[] = {"x",         "w",           "scale", "bias",    "pertokenScale",
                                   "groupList", "sharedInput", "logit", "rowIndex"};

struct GmmFrInfershapeCase {
    GmmFrInfershapeCase(const csv_map &m)
    {
        caseName = ReadMap(m, "caseName");
        enable = ParseBool(ReadMap(m, "enable"));
        prefix = ReadMap(m, "prefix");
        expectSuccess = ParseBool(ReadMap(m, "expectSuccess"));
        expectYShape = ReadMap(m, "expectYShape");
        for (size_t i = 0; i < 9; ++i) {
            inShapes.push_back(ReadMap(m, string(K_IN_NAMES[i]) + "Shape"));
            inDtypes.push_back(ReadMap(m, string(K_IN_NAMES[i]) + "Dtype"));
            inFormats.push_back(ReadMap(m, string(K_IN_NAMES[i]) + "Format"));
        }
        yDtype = ReadMap(m, "yDtype");
        yFormat = ReadMap(m, "yFormat");
        dtypeAttr = ParseScalarToken(ReadMap(m, "dtype"));
        siWeight = ParseFloatDefault(ReadMap(m, "sharedInputWeight"), 1.0F);
        siOffset = ParseScalarToken(ReadMap(m, "sharedInputOffset"));
        tpX = ParseBool(ReadMap(m, "transposeX"));
        tpW = ParseBool(ReadMap(m, "transposeW"));
        outBs = ParseScalarToken(ReadMap(m, "outputBs"));
        glType = ParseScalarToken(ReadMap(m, "groupListType"));
    }

    void Run() const
    {
        vector<gert::InfershapeContextPara::TensorDescription> ins;
        for (size_t i = 0; i < 9; ++i) {
            ins.emplace_back(MakeGertStorageShape(ParseDims(inShapes[i], {}, true)), ParseGeDtype(inDtypes[i]),
                             ParseGeFormat(inFormats[i]));
        }
        gert::InfershapeContextPara para(
            "GroupedMatmulFinalizeRouting", ins,
            {{MakeGertStorageShape(ParseDims("NONE", {}, true)), ParseGeDtype(yDtype), ParseGeFormat(yFormat)}},
            {{"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dtypeAttr)},
             {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(siWeight)},
             {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(siOffset)},
             {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(tpX)},
             {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(tpW)},
             {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outBs)},
             {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(glType)}});
        ExecuteTestCase(para, expectSuccess ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED,
                        {ParseDims(expectYShape, {}, true)});
    }

    string caseName;
    bool enable = true;
    string prefix;
    bool expectSuccess = false;
    string expectYShape;
    vector<string> inShapes, inDtypes, inFormats;
    string yDtype, yFormat;
    int64_t dtypeAttr = 0;
    float siWeight = 1.0F;
    int64_t siOffset = 0;
    bool tpX = false;
    bool tpW = false;
    int64_t outBs = 0;
    int64_t glType = 1;
};

vector<GmmFrInfershapeCase> LoadEnabled()
{
    string path = ResolveCsvPath("test_grouped_matmul_finalize_routing_infershape.csv",
                                 "gmm/grouped_matmul_finalize_routing/tests/ut/op_host", __FILE__);
    auto all = GetCasesFromCsv<GmmFrInfershapeCase>(path);
    vector<GmmFrInfershapeCase> out;
    for (auto &c : all) {
        if (c.enable) {
            out.push_back(std::move(c));
        }
    }
    return out;
}

ge::graphStatus RunInferDtype(const ge::DataType dtypes[9], bool tpW, int64_t dtypeAttr = 0,
                              ge::DataType *outDtype = nullptr)
{
    auto reg = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto impl = reg->GetOpImpl("GroupedMatmulFinalizeRouting");
    ge::DataType v[9];
    vector<void *> iv;
    for (size_t i = 0; i < 9; ++i) {
        v[i] = dtypes[i];
        iv.push_back(&v[i]);
    }
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("GroupedMatmulFinalizeRouting")
                             .NodeIoNum(9, 1)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes(iv)
                             .NodeAttrs({
                                 {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(dtypeAttr)},
                                 {"shared_input_weight", Ops::Transformer::AnyValue::CreateFrom<float>(1.0F)},
                                 {"shared_input_offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                 {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                 {"transpose_w", Ops::Transformer::AnyValue::CreateFrom<bool>(tpW)},
                                 {"output_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                 {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                             })
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    auto ret = impl->infer_datatype(context);
    if (outDtype != nullptr) {
        *outDtype = context->GetOutputDataType(0);
    }
    return ret;
}

string MakeParamName(const testing::TestParamInfo<GmmFrInfershapeCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.prefix);
}

} // namespace

namespace GroupedMatmulFinalizeRoutingInfershapeUT {

class TestInfershape : public testing::TestWithParam<GmmFrInfershapeCase> {};

TEST_P(TestInfershape, csvDriven)
{
    GetParam().Run();
}

INSTANTIATE_TEST_SUITE_P(GMMFR_INFERSHAPE, TestInfershape, testing::ValuesIn(LoadEnabled()), MakeParamName);

class TestInferdtype : public testing::Test {};

TEST_F(TestInferdtype, W8A8ValidDtypes)
{
    ge::DataType d[] = {ge::DT_INT8,  ge::DT_INT8,      ge::DT_FLOAT,     ge::DT_UNDEFINED, ge::DT_UNDEFINED,
                        ge::DT_INT64, ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_INT64};
    EXPECT_EQ(RunInferDtype(d, false), ge::GRAPH_SUCCESS);
}

TEST_F(TestInferdtype, W8A8WithBf16Scale)
{
    ge::DataType d[] = {ge::DT_INT8,  ge::DT_INT8,      ge::DT_BF16,      ge::DT_UNDEFINED, ge::DT_UNDEFINED,
                        ge::DT_INT64, ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_INT32};
    EXPECT_EQ(RunInferDtype(d, false), ge::GRAPH_SUCCESS);
}

TEST_F(TestInferdtype, W4A8ValidDtypes)
{
    ge::DataType d[] = {ge::DT_INT8,  ge::DT_INT4,      ge::DT_INT64, ge::DT_FLOAT, ge::DT_UNDEFINED,
                        ge::DT_INT64, ge::DT_UNDEFINED, ge::DT_FLOAT, ge::DT_INT64};
    EXPECT_EQ(RunInferDtype(d, false), ge::GRAPH_SUCCESS);
}

TEST_F(TestInferdtype, MXFP8ValidDtypes)
{
    ge::DataType d[] = {ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0,
                        ge::DT_UNDEFINED,   ge::DT_FLOAT8_E8M0, ge::DT_INT64,
                        ge::DT_UNDEFINED,   ge::DT_FLOAT,       ge::DT_INT64};
    EXPECT_EQ(RunInferDtype(d, true), ge::GRAPH_SUCCESS);
}

TEST_F(TestInferdtype, MXFP4ValidDtypes)
{
    ge::DataType d[] = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E8M0,
                        ge::DT_UNDEFINED,   ge::DT_FLOAT8_E8M0, ge::DT_INT64,
                        ge::DT_UNDEFINED,   ge::DT_FLOAT,       ge::DT_INT64};
    EXPECT_EQ(RunInferDtype(d, true), ge::GRAPH_SUCCESS);
}

TEST_F(TestInferdtype, MXFP8Bf16OutDtype)
{
    ge::DataType d[] = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0,
                        ge::DT_UNDEFINED,     ge::DT_FLOAT8_E8M0,   ge::DT_INT64,
                        ge::DT_UNDEFINED,     ge::DT_BF16,          ge::DT_INT32};
    ge::DataType outDtype = ge::DT_UNDEFINED;
    EXPECT_EQ(RunInferDtype(d, true, 2, &outDtype), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outDtype, ge::DT_BF16);
}

TEST_F(TestInferdtype, MXFP4Bf16OutDtype)
{
    ge::DataType d[] = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E8M0,
                        ge::DT_UNDEFINED,   ge::DT_FLOAT8_E8M0, ge::DT_INT64,
                        ge::DT_UNDEFINED,   ge::DT_BF16,        ge::DT_INT32};
    ge::DataType outDtype = ge::DT_UNDEFINED;
    EXPECT_EQ(RunInferDtype(d, true, 2, &outDtype), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outDtype, ge::DT_BF16);
}

TEST_F(TestInferdtype, W8A8WithSharedInput)
{
    ge::DataType d[] = {ge::DT_INT8,  ge::DT_INT8, ge::DT_FLOAT, ge::DT_UNDEFINED, ge::DT_UNDEFINED,
                        ge::DT_INT64, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT64};
    EXPECT_EQ(RunInferDtype(d, false), ge::GRAPH_SUCCESS);
}

TEST_F(TestInferdtype, W4A8WithSharedInput)
{
    ge::DataType d[] = {ge::DT_INT8,  ge::DT_INT4, ge::DT_INT64, ge::DT_FLOAT, ge::DT_UNDEFINED,
                        ge::DT_INT64, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT64};
    EXPECT_EQ(RunInferDtype(d, false), ge::GRAPH_SUCCESS);
}

TEST_F(TestInferdtype, InvalidDtypeCompletelyWrong)
{
    ge::DataType d[] = {ge::DT_FLOAT16, ge::DT_BF16,      ge::DT_FLOAT,     ge::DT_UNDEFINED, ge::DT_UNDEFINED,
                        ge::DT_INT64,   ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_INT64};
    EXPECT_EQ(RunInferDtype(d, false), ge::GRAPH_FAILED);
}

TEST_F(TestInferdtype, W8A8WrongScaleDtype)
{
    ge::DataType d[] = {ge::DT_INT8,  ge::DT_INT8,      ge::DT_INT32,     ge::DT_UNDEFINED, ge::DT_UNDEFINED,
                        ge::DT_INT64, ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_INT64};
    EXPECT_EQ(RunInferDtype(d, false), ge::GRAPH_FAILED);
}

TEST_F(TestInferdtype, W4A8WrongBiasDtype)
{
    ge::DataType d[] = {ge::DT_INT8,  ge::DT_INT4,      ge::DT_INT64, ge::DT_INT32, ge::DT_UNDEFINED,
                        ge::DT_INT64, ge::DT_UNDEFINED, ge::DT_FLOAT, ge::DT_INT64};
    EXPECT_EQ(RunInferDtype(d, false), ge::GRAPH_FAILED);
}

TEST_F(TestInferdtype, W8A8WrongGroupListDtype)
{
    ge::DataType d[] = {ge::DT_INT8,  ge::DT_INT8,      ge::DT_FLOAT,     ge::DT_UNDEFINED, ge::DT_UNDEFINED,
                        ge::DT_INT32, ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_INT64};
    EXPECT_EQ(RunInferDtype(d, false), ge::GRAPH_FAILED);
}

TEST_F(TestInferdtype, W8A8WrongRowIndexDtype)
{
    ge::DataType d[] = {ge::DT_INT8,  ge::DT_INT8,      ge::DT_FLOAT,     ge::DT_UNDEFINED, ge::DT_UNDEFINED,
                        ge::DT_INT64, ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_FLOAT};
    EXPECT_EQ(RunInferDtype(d, false), ge::GRAPH_FAILED);
}

TEST_F(TestInferdtype, MXWrongScaleDtype)
{
    ge::DataType d[] = {ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT, ge::DT_UNDEFINED, ge::DT_FLOAT8_E8M0,
                        ge::DT_INT64,       ge::DT_UNDEFINED,   ge::DT_FLOAT, ge::DT_INT64};
    EXPECT_EQ(RunInferDtype(d, true), ge::GRAPH_FAILED);
}

TEST_F(TestInferdtype, W4A8WithOffset)
{
    ge::DataType d[] = {ge::DT_INT8,  ge::DT_INT4,      ge::DT_INT64, ge::DT_FLOAT, ge::DT_UNDEFINED,
                        ge::DT_INT64, ge::DT_UNDEFINED, ge::DT_FLOAT, ge::DT_INT64};
    EXPECT_EQ(RunInferDtype(d, false), ge::GRAPH_SUCCESS);
}

} // namespace GroupedMatmulFinalizeRoutingInfershapeUT
