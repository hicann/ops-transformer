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
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class FlashAttentionScoreGradProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttentionScoreGradProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScoreGradProto TearDown" << std::endl;
    }
};

TEST_F(FlashAttentionScoreGradProto, flash_attention_score_grad_infershape_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FlashAttentionScoreGrad",
        // 输入Tensor
        {
            // q
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // k: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // v: S2=256, B=1, H2=128
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dy
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // pse_shift
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // drop_mask
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // padding_mask
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            // atten_mask: S1=256, S2=256
            {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
            // softmax_max
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_sum
            {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // softmax_in
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // attention_in
            {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // prefix
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_qlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // actual_seq_kvlen
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // q_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // kv_start_idx
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            // dScaleQ
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleK
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleV
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaledy
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dScaleo
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // queryRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // keyRope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // 输出Tensor
            // dq
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dv
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dpse
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dq_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            // dk_rope
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// 属性
         {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {}, {}, {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ---------------------------------------------------------------------------
// 覆盖补充：input_layout 分支（BSH/BSND/BNSD/TND）、大小写归一化、
// 非法 layout 失败分支、可选输入（pse/rope/sink）存在时的输出推导。
// ---------------------------------------------------------------------------

// gert::Shape 仅支持 initializer_list/指针构造，无法从 std::vector<int64_t> 直接构造，
// 按维度个数逐元素展开为 StorageShape（空向量对应空 shape，即"可选输入不存在"）。
static gert::StorageShape MakeFasgStorageShape(const std::vector<int64_t>& dims)
{
    switch (dims.size()) {
        case 1:
            return gert::StorageShape({dims[0]}, {dims[0]});
        case 2:
            return gert::StorageShape({dims[0], dims[1]}, {dims[0], dims[1]});
        case 3:
            return gert::StorageShape({dims[0], dims[1], dims[2]}, {dims[0], dims[1], dims[2]});
        case 4:
            return gert::StorageShape({dims[0], dims[1], dims[2], dims[3]}, {dims[0], dims[1], dims[2], dims[3]});
        default: // 0 维（可选输入为空）等场景返回默认空 shape
            return gert::StorageShape();
    }
}

// 基础输入构造：24 输入（不含 sink），全可选输入为空；sink 非空时追加第 25 个输入
// （sink）与第 7 个输出（dsink），此时 outputs 需提供 7 项（第 7 项为 dsink）。
static gert::InfershapeContextPara MakeFasgInfershapePara(
    const std::vector<int64_t>& q, const std::vector<int64_t>& k, const std::vector<int64_t>& v,
    const std::vector<int64_t>& dy, const std::vector<int64_t>& attenMask, const std::vector<int64_t>& pse,
    const std::vector<int64_t>& queryRope, const std::vector<int64_t>& keyRope,
    const std::vector<std::vector<int64_t>>& outputs, const std::string& layout, const std::vector<int64_t>& sink = {})
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensors = {
        {MakeFasgStorageShape(q), ge::DT_FLOAT, ge::FORMAT_ND},          // query
        {MakeFasgStorageShape(k), ge::DT_FLOAT, ge::FORMAT_ND},          // key
        {MakeFasgStorageShape(v), ge::DT_FLOAT, ge::FORMAT_ND},          // value
        {MakeFasgStorageShape(dy), ge::DT_FLOAT, ge::FORMAT_ND},         // dy
        {MakeFasgStorageShape(pse), ge::DT_FLOAT, ge::FORMAT_ND},        // pse_shift
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // drop_mask
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // padding_mask
        {MakeFasgStorageShape(attenMask), ge::DT_UINT8, ge::FORMAT_ND},  // atten_mask
        {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_max
        {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_sum
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // softmax_in
        {MakeFasgStorageShape(dy), ge::DT_FLOAT, ge::FORMAT_ND},         // attention_in
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                         // prefix
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                         // actual_seq_qlen
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                         // actual_seq_kvlen
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                         // q_start_idx
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                         // kv_start_idx
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dScaleQ
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dScaleK
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dScaleV
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dScaledy
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                         // dScaleo
        {MakeFasgStorageShape(queryRope), ge::DT_FLOAT, ge::FORMAT_ND},  // queryRope
        {MakeFasgStorageShape(keyRope), ge::DT_FLOAT, ge::FORMAT_ND},    // keyRope
    };
    std::vector<gert::InfershapeContextPara::TensorDescription> outputTensors = {
        {MakeFasgStorageShape(outputs[0]), ge::DT_FLOAT, ge::FORMAT_ND}, // dq
        {MakeFasgStorageShape(outputs[1]), ge::DT_FLOAT, ge::FORMAT_ND}, // dk
        {MakeFasgStorageShape(outputs[2]), ge::DT_FLOAT, ge::FORMAT_ND}, // dv
        {MakeFasgStorageShape(outputs[3]), ge::DT_FLOAT, ge::FORMAT_ND}, // dpse
        {MakeFasgStorageShape(outputs[4]), ge::DT_FLOAT, ge::FORMAT_ND}, // dq_rope
        {MakeFasgStorageShape(outputs[5]), ge::DT_FLOAT, ge::FORMAT_ND}, // dk_rope
    };
    if (!sink.empty()) {
        // sink：第 25 个输入；dsink：第 7 个输出，形状跟随 sink
        inputTensors.push_back({MakeFasgStorageShape(sink), ge::DT_FLOAT, ge::FORMAT_ND});
        outputTensors.push_back({MakeFasgStorageShape(outputs[6]), ge::DT_FLOAT, ge::FORMAT_ND});
    }
    return gert::InfershapeContextPara(
        "FlashAttentionScoreGrad", inputTensors, outputTensors,
        {{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
         {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
         {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
         {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
         {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}});
}

static const std::vector<std::vector<int64_t>> kDefaultOutputs = {
    {256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {}, {}, {}};

// BSH 布局：b=dim0, sQ=dim1，输出跟随输入 3D 形状。
TEST_F(FlashAttentionScoreGradProto, flash_attention_score_grad_infershape_bsh)
{
    auto para = MakeFasgInfershapePara({1, 256, 128}, {1, 256, 128}, {1, 256, 128}, {1, 256, 128}, {256, 256}, {}, {},
                                       {}, kDefaultOutputs, "BSH");
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 256, 128}, {1, 256, 128}, {1, 256, 128}, {}, {}, {}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// BSND 布局：b=dim0, sQ=dim1（4D 形状原样透传到输出）。
TEST_F(FlashAttentionScoreGradProto, flash_attention_score_grad_infershape_bsnd)
{
    auto para =
        MakeFasgInfershapePara({1, 256, 1, 128}, {1, 256, 1, 128}, {1, 256, 1, 128}, {1, 256, 1, 128}, {256, 256}, {},
                               {}, {}, {{1, 256, 1, 128}, {1, 256, 1, 128}, {1, 256, 1, 128}, {}, {}, {}}, "BSND");
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 256, 1, 128}, {1, 256, 1, 128}, {1, 256, 1, 128}, {}, {}, {}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// BNSD 布局：b=dim0, sQ=dim2。
TEST_F(FlashAttentionScoreGradProto, flash_attention_score_grad_infershape_bnsd)
{
    auto para =
        MakeFasgInfershapePara({1, 1, 256, 128}, {1, 1, 256, 128}, {1, 1, 256, 128}, {1, 1, 256, 128}, {256, 256}, {},
                               {}, {}, {{1, 1, 256, 128}, {1, 1, 256, 128}, {1, 1, 256, 128}, {}, {}, {}}, "BNSD");
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 1, 256, 128}, {1, 1, 256, 128}, {1, 1, 256, 128}, {}, {}, {}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TND 布局：t=dim0。
TEST_F(FlashAttentionScoreGradProto, flash_attention_score_grad_infershape_tnd)
{
    auto para = MakeFasgInfershapePara({256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {256, 256}, {}, {},
                                       {}, kDefaultOutputs, "TND");
    std::vector<std::vector<int64_t>> expectOutputShape = {{256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {}, {}, {}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 小写 layout：覆盖 toupper 归一化分支。
TEST_F(FlashAttentionScoreGradProto, flash_attention_score_grad_infershape_lowercase_layout)
{
    auto para = MakeFasgInfershapePara({1, 256, 128}, {1, 256, 128}, {1, 256, 128}, {1, 256, 128}, {256, 256}, {}, {},
                                       {}, kDefaultOutputs, "sbh");
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 256, 128}, {1, 256, 128}, {1, 256, 128}, {}, {}, {}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 非法 layout：GRAPH_FAILED 失败分支。
TEST_F(FlashAttentionScoreGradProto, flash_attention_score_grad_infershape_invalid_layout)
{
    auto para = MakeFasgInfershapePara({1, 256, 128}, {1, 256, 128}, {1, 256, 128}, {1, 256, 128}, {256, 256}, {}, {},
                                       {}, kDefaultOutputs, "XXX");
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// pse_shift 存在：dpse 输出跟随 pse 形状。
TEST_F(FlashAttentionScoreGradProto, flash_attention_score_grad_infershape_pse_present)
{
    // SBH 布局输入用规范形状 [S, B, H]；dq/dk/dv 透传输入形状，dpse 跟随 pse。
    auto para =
        MakeFasgInfershapePara({256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {256, 256}, {1, 1, 256, 128},
                               {}, {}, {{256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {1, 1, 256, 128}, {}, {}}, "SBH");
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {1, 1, 256, 128}, {}, {}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// queryRope/keyRope 存在：dq_rope/dk_rope 跟随各自 rope 形状。
TEST_F(FlashAttentionScoreGradProto, flash_attention_score_grad_infershape_rope_present)
{
    auto para = MakeFasgInfershapePara(
        {256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {256, 256}, {}, {1, 256, 64}, {1, 256, 64},
        {{256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {}, {1, 256, 64}, {1, 256, 64}}, "SBH");
    std::vector<std::vector<int64_t>> expectOutputShape = {{256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {},
                                                           {1, 256, 64},  {1, 256, 64}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// sink 存在（第 25 个输入 + 第 7 个输出）：dsink 跟随 sink 形状。
TEST_F(FlashAttentionScoreGradProto, flash_attention_score_grad_infershape_sink_present)
{
    auto para = MakeFasgInfershapePara({256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {256, 256}, {}, {},
                                       {}, {{256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {}, {}, {}, {4}}, "SBH", {4});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {256, 1, 128}, {256, 1, 128}, {256, 1, 128}, {}, {}, {}, {4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ---------------------------------------------------------------------------
// inferdatatype 覆盖补充（合并自 test_flash_attention_score_grad_inferdatatype_infershape.cpp）。
// ---------------------------------------------------------------------------
namespace FlashAttentionScoreGradUT {

// 参数：输入 dtype、out_dtype 属性值、期望结果与期望输出 dtype。
struct FasgInferDTypeParam {
    ge::DataType input_dtype;
    int64_t out_dtype;
    ge::graphStatus expect_result;
    ge::DataType expect_dtype;
    std::string case_name;
};

class FlashAttentionScoreGradInferDTypeTest : public testing::TestWithParam<FasgInferDTypeParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttentionScoreGrad InferDTypeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScoreGrad InferDTypeTest TearDown" << std::endl;
    }
};

// 构造任意 dtype 的 24 输入 / 7 输出 faker 并完成 Build（Faker 不可按值返回，直接返回 holder）。
static gert::KernelRunContextHolder MakeFasgDTypeContext(ge::DataType inputDtype, int64_t outDtype)
{
    std::vector<uint32_t> inputInstance(24, 1);
    std::vector<uint32_t> outputInstance(7, 1);
    ge::DataType expected =
        (inputDtype == ge::DT_FLOAT8_E5M2 || inputDtype == ge::DT_FLOAT8_E4M3FN || inputDtype == ge::DT_HIFLOAT8) ?
            (outDtype == 0 ? ge::DT_FLOAT16 : ge::DT_BF16) :
            inputDtype;
    std::vector<void*> inputDataTypes(24, &inputDtype);
    std::vector<void*> outputDataTypes = {&expected, &expected, &expected, &expected, &expected, &expected, &expected};
    gert::InferDataTypeContextFaker faker;
    faker.SetOpType("FlashAttentionScoreGrad")
        .IrInstanceNum(inputInstance, outputInstance)
        .InputDataTypes(inputDataTypes)
        .OutputDataTypes(outputDataTypes)
        .NodeAttrs({{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
                    {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
                    {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
                    {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
                    {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                    {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
                    {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                    {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                    {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                    {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                    {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                    {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
                    {"softmax_in_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("same_as_input")}});
    return faker.Build();
}

TEST_P(FlashAttentionScoreGradInferDTypeTest, param)
{
    auto param = GetParam();
    auto contextHolder = MakeFasgDTypeContext(param.input_dtype, param.out_dtype);
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("FlashAttentionScoreGrad")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), param.expect_result);
    if (param.expect_result == ge::GRAPH_SUCCESS) {
        auto* context = contextHolder.GetContext<gert::InferDataTypeContext>();
        // dq/dk/dv/dpse/dq_rope/dk_rope 跟随输入（FP8 时按 out_dtype 定档）
        for (int32_t i = 0; i < 6; i++) {
            EXPECT_EQ(context->GetOutputDataType(i), param.expect_dtype) << "output " << i;
        }
        // dsink 恒为 FP32
        EXPECT_EQ(context->GetOutputDataType(6), ge::DT_FLOAT);
    }
}

INSTANTIATE_TEST_SUITE_P(FlashAttentionScoreGrad, FlashAttentionScoreGradInferDTypeTest,
                         testing::ValuesIn(std::vector<FasgInferDTypeParam>{
                             // 普通 dtype：输出跟随输入
                             {ge::DT_FLOAT16, 0, ge::GRAPH_SUCCESS, ge::DT_FLOAT16, "fp16"},
                             {ge::DT_BF16, 0, ge::GRAPH_SUCCESS, ge::DT_BF16, "bf16"},
                             {ge::DT_FLOAT, 0, ge::GRAPH_SUCCESS, ge::DT_FLOAT, "fp32"},
                             // FP8 族：out_dtype=0 输出 FP16
                             {ge::DT_FLOAT8_E5M2, 0, ge::GRAPH_SUCCESS, ge::DT_FLOAT16, "e5m2_out_fp16"},
                             {ge::DT_FLOAT8_E4M3FN, 0, ge::GRAPH_SUCCESS, ge::DT_FLOAT16, "e4m3fn_out_fp16"},
                             {ge::DT_HIFLOAT8, 0, ge::GRAPH_SUCCESS, ge::DT_FLOAT16, "hifloat8_out_fp16"},
                             // FP8 族：out_dtype=1 输出 BF16
                             {ge::DT_FLOAT8_E5M2, 1, ge::GRAPH_SUCCESS, ge::DT_BF16, "e5m2_out_bf16"},
                             {ge::DT_HIFLOAT8, 1, ge::GRAPH_SUCCESS, ge::DT_BF16, "hifloat8_out_bf16"},
                             // FP8 族：out_dtype 非法值 -> 失败
                             {ge::DT_FLOAT8_E5M2, 2, ge::GRAPH_FAILED, ge::DT_FLOAT16, "e5m2_invalid_out_dtype"},
                         }),
                         [](const testing::TestParamInfo<FasgInferDTypeParam>& info) { return info.param.case_name; });

} // namespace FlashAttentionScoreGradUT
