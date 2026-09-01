/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <string>

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"

namespace {
using OpAttr = gert::InfershapeContextPara::OpAttr;

std::vector<gert::InfershapeContextPara::TensorDescription> BuildInputs()
{
    return {
        {{{20, 17, 128}, {20, 17, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{256, 128}, {256, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{17}, {17}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{8, 2, 128, 128}, {8, 2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{8, 2, 128, 128}, {8, 2, 128, 128}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{{8, 2, 128, 1}, {8, 2, 128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{128, 128}, {128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
}

std::vector<gert::InfershapeContextPara::TensorDescription> BuildOutputs()
{
    return {
        {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND}, {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
}

std::vector<OpAttr> BuildAttrs(const std::vector<int64_t> &headNums, const std::string &layoutQkv = "NTD",
                               const std::string &layoutQOut = "NTD", float epsilon = 1e-6f)
{
    return {
        {"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(headNums)},
        {"layout_qkv", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutQkv)},
        {"layout_q_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(layoutQOut)},
        {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(epsilon)},
    };
}

void VerifyInferDtype(const std::vector<std::pair<std::string, Ops::Transformer::AnyValue>> &attrs,
                      ge::DataType kCacheDtype, ge::DataType expectedQOutDtype,
                      ge::DataType expectedQScaleDtype = ge::DT_FLOAT, ge::DataType kScaleCacheDtype = ge::DT_FLOAT)
{
    ge::DataType bf16 = ge::DT_BF16;
    ge::DataType fp32 = ge::DT_FLOAT;
    ge::DataType int32 = ge::DT_INT32;
    ge::DataType fp8 = ge::DT_FLOAT8_E4M3FN;
    ge::DataType qOut = ge::DT_UNDEFINED;
    ge::DataType qScale = ge::DT_UNDEFINED;
    ge::DataType kCacheOut = ge::DT_UNDEFINED;
    ge::DataType vCacheOut = ge::DT_UNDEFINED;
    ge::DataType kScaleCacheOut = ge::DT_UNDEFINED;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("QkvRmsNormRopeCacheWithKScale")
                             .IrInputNum(13)
                             .NodeIoNum(13, 5)
                             .NodeAttrs(attrs)
                             .InputDataTypes({&bf16, &fp32, &fp32, &fp32, &int32, &kCacheDtype, &fp8, &kScaleCacheDtype,
                                              &int32, &int32, &bf16, &fp32, &int32})
                             .OutputDataTypes({&qOut, &qScale, &kCacheOut, &vCacheOut, &kScaleCacheOut})
                             .Build();

    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry->GetOpImpl("QkvRmsNormRopeCacheWithKScale"), nullptr);
    auto inferDtype = registry->GetOpImpl("QkvRmsNormRopeCacheWithKScale")->infer_datatype;
    ASSERT_NE(inferDtype, nullptr);
    auto *context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferDtype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), expectedQOutDtype);
    EXPECT_EQ(context->GetOutputDataType(1), expectedQScaleDtype);
    EXPECT_EQ(context->GetOutputDataType(2), kCacheDtype);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT8_E4M3FN);
    EXPECT_EQ(context->GetOutputDataType(4), kScaleCacheDtype);
}

} // namespace

TEST(QkvRmsNormRopeCacheWithKScaleInferDtype, DefaultsQOutToFp8)
{
    const std::vector<int64_t> headNums = {16, 2, 2};
    VerifyInferDtype({{"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(headNums)},
                      {"layout_qkv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                      {"layout_q_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("NTD")},
                      {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)}},
                     ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferDtype, UsesQOutDtypeAttr)
{
    const std::vector<int64_t> headNums = {16, 2, 2};
    const std::vector<int64_t> mropeSection = {22, 12, 10};
    VerifyInferDtype(
        {{"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(headNums)},
         {"layout_qkv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_q_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
         {"mrope_section", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(mropeSection)},
         {"q_quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("NoQuant")},
         {"q_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_BF16))},
         {"k_quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PerTokenPerHead")}},
        ge::DT_INT8, ge::DT_BF16);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferDtype, PreservesSevenAttrQOutDtypeIndex)
{
    const std::vector<int64_t> headNums = {16, 2, 2};
    const std::vector<int64_t> mropeSection = {22, 12, 10};
    VerifyInferDtype(
        {{"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(headNums)},
         {"layout_qkv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_q_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
         {"mrope_section", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(mropeSection)},
         {"q_quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("NoQuant")},
         {"q_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_BF16))}},
        ge::DT_INT8, ge::DT_BF16);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferDtype, InfersMropeMxQScaleE8m0)
{
    const std::vector<int64_t> headNums = {16, 2, 2};
    const std::vector<int64_t> mropeSection = {22, 12, 10};
    VerifyInferDtype(
        {{"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(headNums)},
         {"layout_qkv", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"layout_q_out", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
         {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
         {"mrope_section", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(mropeSection)},
         {"q_quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Mx")},
         {"q_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN))},
         {"k_quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Mx")}},
        ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, InfersOutputAndCacheShapes)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
                                     BuildAttrs(headNums));

    std::vector<std::vector<int64_t>> expected = {
        {16, 17, 128}, {16, 17}, {8, 2, 128, 128}, {8, 2, 128, 128}, {8, 2, 128, 1},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, InfersTndOutputAndCacheShapes)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    auto inputs = BuildInputs();
    inputs[0] = {{{17, 20, 128}, {17, 20, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", inputs, BuildOutputs(),
                                     BuildAttrs(headNums, "TND", "TND"));

    std::vector<std::vector<int64_t>> expected = {
        {17, 16, 128}, {17, 16}, {8, 2, 128, 128}, {8, 2, 128, 128}, {8, 2, 128, 1},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, InfersMropeMxRank3QScale)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    auto ropeInputs = BuildInputs();
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs(ropeInputs.begin(), ropeInputs.begin() + 8);
    inputs[0] = {{{17, 20, 128}, {17, 20, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    inputs[7] = {{{8, 2, 128, 4}, {8, 2, 128, 4}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
    inputs.push_back({{{2, 128}, {2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND});
    inputs.push_back({{{17, 3}, {17, 3}}, ge::DT_INT32, ge::FORMAT_ND});
    auto attrs = BuildAttrs(headNums, "TND", "TND");
    const std::vector<int64_t> mropeSection = {22, 12, 10};
    attrs.push_back({"mrope_section", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(mropeSection)});
    attrs.push_back({"q_quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Mx")});
    attrs.push_back(
        {"q_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN))});
    attrs.push_back({"k_quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Mx")});
    const std::vector<uint32_t> inputInstanceNum = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1};
    const std::vector<uint32_t> outputInstanceNum = {1, 1, 1, 1, 1};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", inputs, BuildOutputs(), attrs, inputInstanceNum,
                                     outputInstanceNum);
    std::vector<std::vector<int64_t>> expected = {
        {17, 16, 128}, {17, 16, 4}, {8, 2, 128, 128}, {8, 2, 128, 128}, {8, 2, 128, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, InfersTndInputNtdOutputAndCacheShapes)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    auto inputs = BuildInputs();
    inputs[0] = {{{17, 20, 128}, {17, 20, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", inputs, BuildOutputs(),
                                     BuildAttrs(headNums, "TND", "NTD"));

    std::vector<std::vector<int64_t>> expected = {
        {16, 17, 128}, {16, 17}, {8, 2, 128, 128}, {8, 2, 128, 128}, {8, 2, 128, 1},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, RejectsInvalidLayoutQkv)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
                                     BuildAttrs(headNums, "tnd"));

    std::vector<std::vector<int64_t>> expected;
    ExecuteTestCase(para, ge::GRAPH_FAILED, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, InfersDefaultLayoutsWhenLayoutAttrsMissing)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    auto inputs = BuildInputs();
    inputs[0] = {{{17, 20, 128}, {17, 20, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    gert::InfershapeContextPara para(
        "QkvRmsNormRopeCacheWithKScale", inputs, BuildOutputs(),
        {{"head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(headNums)}});

    std::vector<std::vector<int64_t>> expected = {
        {16, 17, 128}, {16, 17}, {8, 2, 128, 128}, {8, 2, 128, 128}, {8, 2, 128, 1},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, InfersDefaultLayoutsWhenLayoutAttrsEmpty)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    auto inputs = BuildInputs();
    inputs[0] = {{{17, 20, 128}, {17, 20, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", inputs, BuildOutputs(),
                                     BuildAttrs(headNums, "", ""));

    std::vector<std::vector<int64_t>> expected = {
        {16, 17, 128}, {16, 17}, {8, 2, 128, 128}, {8, 2, 128, 128}, {8, 2, 128, 1},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, RejectsQkvHeadLessThanNq)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    auto inputs = BuildInputs();
    inputs[0] = {{{17, 15, 128}, {17, 15, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", inputs, BuildOutputs(),
                                     BuildAttrs(headNums, "TND", "TND"));

    std::vector<std::vector<int64_t>> expected;
    ExecuteTestCase(para, ge::GRAPH_FAILED, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, RejectsInvalidLayoutQOut)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
                                     BuildAttrs(headNums, "NTD", "ntd"));

    std::vector<std::vector<int64_t>> expected;
    ExecuteTestCase(para, ge::GRAPH_FAILED, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, RejectsNtdInputTndLayoutQOut)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
                                     BuildAttrs(headNums, "NTD", "TND"));

    std::vector<std::vector<int64_t>> expected;
    ExecuteTestCase(para, ge::GRAPH_FAILED, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, RejectsInvalidHeadNumsAttr)
{
    std::vector<int64_t> headNums = {};
    gert::InfershapeContextPara para("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
                                     BuildAttrs(headNums));

    std::vector<std::vector<int64_t>> expected;
    ExecuteTestCase(para, ge::GRAPH_FAILED, expected);

    headNums = {0, 2, 2};
    gert::InfershapeContextPara zeroHeadPara("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
                                             BuildAttrs(headNums));
    ExecuteTestCase(zeroHeadPara, ge::GRAPH_FAILED, expected);

    headNums = {-1, 2, 2};
    gert::InfershapeContextPara negativeHeadPara("QkvRmsNormRopeCacheWithKScale", BuildInputs(), BuildOutputs(),
                                                 BuildAttrs(headNums));
    ExecuteTestCase(negativeHeadPara, ge::GRAPH_FAILED, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, RejectsInvalidInputDimNum)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    std::vector<std::vector<int64_t>> expected;

    auto inputs = BuildInputs();
    inputs[0] = {{{20, 17}, {20, 17}}, ge::DT_BF16, ge::FORMAT_ND};
    gert::InfershapeContextPara qkvPara("QkvRmsNormRopeCacheWithKScale", inputs, BuildOutputs(), BuildAttrs(headNums));
    ExecuteTestCase(qkvPara, ge::GRAPH_FAILED, expected);
}

TEST(QkvRmsNormRopeCacheWithKScaleInferShape, RejectsNonPositiveInputDims)
{
    std::vector<int64_t> headNums = {16, 2, 2};
    std::vector<std::vector<int64_t>> expected;

    auto inputs = BuildInputs();
    inputs[0] = {{{20, -1, 128}, {20, -1, 128}}, ge::DT_BF16, ge::FORMAT_ND};
    gert::InfershapeContextPara qkvPara("QkvRmsNormRopeCacheWithKScale", inputs, BuildOutputs(), BuildAttrs(headNums));
    ExecuteTestCase(qkvPara, ge::GRAPH_FAILED, expected);

    inputs = BuildInputs();
    inputs[0] = {{{20, 17, 0}, {20, 17, 0}}, ge::DT_BF16, ge::FORMAT_ND};
    gert::InfershapeContextPara qkvHeadDimPara("QkvRmsNormRopeCacheWithKScale", inputs, BuildOutputs(),
                                               BuildAttrs(headNums));
    ExecuteTestCase(qkvHeadDimPara, ge::GRAPH_FAILED, expected);
}
