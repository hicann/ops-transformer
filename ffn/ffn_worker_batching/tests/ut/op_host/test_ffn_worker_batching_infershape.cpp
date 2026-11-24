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
#include <limits>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class FfnWorkerBatchingInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FfnWorkerBatchingInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FfnWorkerBatchingInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(FfnWorkerBatchingInfershapeTest, ffn_worker_batching_infershape_test01)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};

    gert::InfershapeContextPara infershapeContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        {
            // output
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // group_list
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // session_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // micro_batch_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // token_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // expert_offsets
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dynamic_scale
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // actual_token_num
        },
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1152, 4096}, {8, 2}, {1152}, {1152},
                                                           {1152},       {1152}, {1152}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ---------------------------------------------------------------------------------------------------------------------
// 迭代一 A5(arch35/ascend950) infershape 规格覆盖
//
// InferShape 与代次无关（A2/A3/A5 共用 InferShape4FfnWorkerBatching，不改），故不区分 socVersion。
// 这些用例覆盖 A5 对外 shape 规格：Y = A*BS*K、H = max_out_shape[3]、group_list = {expert_num, 2}、
// 其余 5 个索引输出与 dynamic_scale = {Y}、actual_token_num = {1}。规格边界对齐 spec.yaml
// （A≤1024、K≤64、expert_num≤8192）。
// ---------------------------------------------------------------------------------------------------------------------

// A5 规格：最大 session 数（A=1024）+ 单 topK（K=1），H=2048 -> Y=1024*1*1=1024。
TEST_F(FfnWorkerBatchingInfershapeTest, ffn_worker_batching_infershape_arch35_max_session)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};

    gert::InfershapeContextPara infershapeContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        {
            // output
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // group_list
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // session_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // micro_batch_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // token_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // expert_offsets
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dynamic_scale
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // actual_token_num
        },
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1024, 1, 1, 2048})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});
    // Y=1024, H=2048; group_list={256,2}
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 2048}, {256, 2}, {1024}, {1024},
                                                           {1024},       {1024},   {1024}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// A5 规格：最大 topK（K=64），A=16、BS=8、H=4096 -> Y=16*8*64=8192。
TEST_F(FfnWorkerBatchingInfershapeTest, ffn_worker_batching_infershape_arch35_max_topk)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};

    gert::InfershapeContextPara infershapeContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        {
            // output
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},  // y (token_dtype=1)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}, // group_list
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // session_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // micro_batch_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // token_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // expert_offsets
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dynamic_scale
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}, // actual_token_num
        },
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 64, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});
    // Y=8192, H=4096; group_list={8,2}
    std::vector<std::vector<int64_t>> expectOutputShape = {{8192, 4096}, {8, 2}, {8192}, {8192},
                                                           {8192},       {8192}, {8192}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 六组类型映射均验证，确保新增类型不会改变原有输出签名。
TEST_F(FfnWorkerBatchingInfershapeTest, ffn_worker_batching_inferdtype_arch35)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("FfnWorkerBatching");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    const ge::DataType tokenTypes[] = {ge::DT_FLOAT16,     ge::DT_BF16,          ge::DT_INT8,
                                       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1};
    for (int64_t tokenDtype = 0; tokenDtype < 6; ++tokenDtype) {
        ge::DataType input = ge::DT_INT8;
        ge::DataType outputs[8] = {};
        auto holder = gert::InferDataTypeContextFaker()
                          .IrInputNum(1)
                          .NodeIoNum(1, 8)
                          .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                          .InputDataTypes({&input})
                          .OutputDataTypes({&outputs[0], &outputs[1], &outputs[2], &outputs[3], &outputs[4],
                                            &outputs[5], &outputs[6], &outputs[7]})
                          .NodeAttrs({{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                      {"max_out_shape",
                                       Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1, 8, 9, 64})},
                                      {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenDtype)}})
                          .Build();
        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), tokenTypes[tokenDtype]);
        EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT64);
        for (size_t i = 2; i < 6; ++i) {
            EXPECT_EQ(context->GetOutputDataType(i), ge::DT_INT32);
        }
        EXPECT_EQ(context->GetOutputDataType(6), tokenDtype >= 3 ? ge::DT_FLOAT8_E8M0 : ge::DT_FLOAT);
        EXPECT_EQ(context->GetOutputDataType(7), ge::DT_INT64);
    }
}

TEST_F(FfnWorkerBatchingInfershapeTest, ffn_worker_batching_mx_shape_3)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};

    gert::InfershapeContextPara infershapeContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        {
            // output
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // group_list
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // session_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // micro_batch_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // token_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // expert_offsets
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dynamic_scale
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // actual_token_num
        },
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 66})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1152, 66}, {8, 2}, {1152},    {1152},
                                                           {1152},     {1152}, {1152, 3}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FfnWorkerBatchingInfershapeTest, ffn_worker_batching_mx_shape_4)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};

    gert::InfershapeContextPara infershapeContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        {
            // output
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // group_list
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // session_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // micro_batch_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // token_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // expert_offsets
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dynamic_scale
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // actual_token_num
        },
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 66})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1152, 66}, {8, 2}, {1152},    {1152},
                                                           {1152},     {1152}, {1152, 3}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FfnWorkerBatchingInfershapeTest, ffn_worker_batching_mx_shape_5)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};

    gert::InfershapeContextPara infershapeContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        {
            // output
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // group_list
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // session_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // micro_batch_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // token_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // expert_offsets
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dynamic_scale
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // actual_token_num
        },
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 66})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1152, 66}, {8, 2}, {1152},    {1152},
                                                           {1152},     {1152}, {1152, 3}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FfnWorkerBatchingInfershapeTest, ffn_worker_batching_infershape_shape_product_overflow_fail)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::InfershapeContextPara infershapeContextPara(
        "FfnWorkerBatching", {{schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(
                               {1024, std::numeric_limits<int64_t>::max(), 1, 1})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// ---------------------------------------------------------------------------------------------------------------------
// 动态 shape 场景（skill rt2.0 加固：-1 未知维 / -2 未知秩）。
// InferShape 输出全由 attr 推导、不读输入 shape：未知维/未知秩输入不得误拦，
// 输出与静态基线 test01（max_out_shape={16,8,9,4096} -> Y=1152）逐项一致。
// ---------------------------------------------------------------------------------------------------------------------

// -1 未知维：desc shape 为 (-1,) 时正常推导。
TEST_F(FfnWorkerBatchingInfershapeTest, ffn_worker_batching_infershape_dyn_dim)
{
    gert::StorageShape schedule_context_shape = {{-1}, {-1}};

    gert::InfershapeContextPara infershapeContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        {
            // output
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // group_list
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // session_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // micro_batch_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // token_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // expert_offsets
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dynamic_scale
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // actual_token_num
        },
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1152, 4096}, {8, 2}, {1152}, {1152},
                                                           {1152},       {1152}, {1152}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// -2 未知秩：StorageShape 按 {{-2}, {-2}} 写法（origin 同 -2），不误拦、不塌缩输出 rank。
TEST_F(FfnWorkerBatchingInfershapeTest, ffn_worker_batching_infershape_dyn_rank)
{
    gert::StorageShape schedule_context_shape = {{-2}, {-2}};

    gert::InfershapeContextPara infershapeContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        {
            // output
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // group_list
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // session_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // micro_batch_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // token_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // expert_offsets
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dynamic_scale
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // actual_token_num
        },
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1152, 4096}, {8, 2}, {1152}, {1152},
                                                           {1152},       {1152}, {1152}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
