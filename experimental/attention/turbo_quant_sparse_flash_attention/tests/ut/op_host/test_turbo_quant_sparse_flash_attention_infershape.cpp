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

#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
using AnyValue = Ops::Transformer::AnyValue;
using TensorDesc = gert::InfershapeContextPara::TensorDescription;
using OpAttr = gert::InfershapeContextPara::OpAttr;

// TND 布局下 actual_seq_lengths 为累加和：单 batch、query 1 个 token、KV 有效长度 512。
int32_t g_actualSeqQuery[] = {1};
int32_t g_actualSeqKv[] = {512};

// query [T=1, N1=8, D=576]；KV 为 PA_BSND，slot 386B = 256B nibble + 128B rope + 2B scale。
std::vector<TensorDesc> BaseInputs()
{
    return {{{{1, 8, 576}, {1, 8, 576}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 128, 1, 386}, {4, 128, 1, 386}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 128, 1, 386}, {4, 128, 1, 386}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 1, 256}, {1, 1, 256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // key_dequant_scale：COMBINE 模式下不消费
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // value_dequant_scale
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, g_actualSeqQuery},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, g_actualSeqKv}};
}

std::vector<TensorDesc> BaseOutputs()
{
    return {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}};
}

// 属性基线取 host 侧唯一合法值，负向用例在此基础上单点改写。
std::vector<OpAttr> BaseAttrs(const std::string &layoutQuery = "TND", bool returnSoftmaxLse = false)
{
    return {{"scale_value", AnyValue::CreateFrom<float>(0.044194173f)},
            {"key_quant_mode", AnyValue::CreateFrom<int64_t>(3)},
            {"value_quant_mode", AnyValue::CreateFrom<int64_t>(3)},
            {"sparse_block_size", AnyValue::CreateFrom<int64_t>(1)},
            {"layout_query", AnyValue::CreateFrom<std::string>(layoutQuery)},
            {"layout_kv", AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"sparse_mode", AnyValue::CreateFrom<int64_t>(3)},
            {"pre_tokens", AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"next_tokens", AnyValue::CreateFrom<int64_t>(INT64_MAX)},
            {"attention_mode", AnyValue::CreateFrom<int64_t>(2)},
            {"quant_scale_repo_mode", AnyValue::CreateFrom<int64_t>(1)},
            {"tile_size", AnyValue::CreateFrom<int64_t>(128)},
            {"rope_head_dim", AnyValue::CreateFrom<int64_t>(64)},
            {"return_softmax_lse", AnyValue::CreateFrom<bool>(returnSoftmaxLse)}};
}
} // namespace

class TurboQuantSparseFlashAttentionProto : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TurboQuantSparseFlashAttentionProto SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TurboQuantSparseFlashAttentionProto TearDown" << std::endl; }
};

// attention_out 最后一维 = query 最后一维 - rope_head_dim = 576 - 64 = 512。
TEST_F(TurboQuantSparseFlashAttentionProto, TurboQuantSparseFlashAttention_infershape_tnd)
{
    gert::InfershapeContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(), BaseAttrs());
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 8, 512}, {0}, {0}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// return_softmax_lse 为 true 时 LSE shape 为 [N2, T, N1 / N2]，首维是 KV 头数而非 T。
TEST_F(TurboQuantSparseFlashAttentionProto, TurboQuantSparseFlashAttention_infershape_lse)
{
    gert::InfershapeContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(),
                                     BaseAttrs("TND", true));
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 8, 512}, {1, 1, 8}, {1, 1, 8}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 不支持的 layout 曾被默认按 TND 处理，与 host tiling 的严格校验不一致，现应直接拒绝。
TEST_F(TurboQuantSparseFlashAttentionProto, TurboQuantSparseFlashAttention_infershape_layout_bsnd_rejected)
{
    gert::InfershapeContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(), BaseAttrs("BSND"));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionProto, TurboQuantSparseFlashAttention_infershape_layout_unknown_rejected)
{
    gert::InfershapeContextPara para("TurboQuantSparseFlashAttention", BaseInputs(), BaseOutputs(), BaseAttrs("BNSD"));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TurboQuantSparseFlashAttentionProto, TurboQuantSparseFlashAttention_inferdtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto dataTypeFunc = spaceRegistry->GetOpImpl("TurboQuantSparseFlashAttention")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType queryType = ge::DT_BF16;
    ge::DataType kvType = ge::DT_INT8;
    ge::DataType indexType = ge::DT_INT32;
    ge::DataType scaleType = ge::DT_FLOAT;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(9, 3)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&queryType, &kvType, &kvType, &indexType, &scaleType, &scaleType,
                                              &indexType, &indexType, &indexType})
                             .NodeAttrs({{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.044194173f)},
                                         {"key_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                         {"value_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                         {"sparse_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                         {"layout_query", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
                                         {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
                                         {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                         {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                                         {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                                         {"attention_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                         {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                         {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                                         {"rope_head_dim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                         {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
}

// 算子仅支持 BFLOAT16：非法 dtype 不应先得到一份看似合法的图推导结果。
TEST_F(TurboQuantSparseFlashAttentionProto, TurboQuantSparseFlashAttention_inferdtype_fp16_rejected)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto dataTypeFunc = spaceRegistry->GetOpImpl("TurboQuantSparseFlashAttention")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType queryType = ge::DT_FLOAT16;
    ge::DataType kvType = ge::DT_INT8;
    ge::DataType indexType = ge::DT_INT32;
    ge::DataType scaleType = ge::DT_FLOAT;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(9, 3)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&queryType, &kvType, &kvType, &indexType, &scaleType, &scaleType,
                                              &indexType, &indexType, &indexType})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_FAILED);
}
