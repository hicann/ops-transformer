/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../../../op_graph/engram_gate_proto.h"
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "test_engram_gate_utils.h"

static void CheckShape(const std::vector<int64_t>& x, bool save, const std::vector<int64_t>& stat)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> outputs;
    outputs.emplace_back(MakeStorageShape({}), ge::DT_BF16, ge::FORMAT_ND);
    for (int i = 0; i < 4; ++i)
        outputs.emplace_back(MakeStorageShape({}), ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara p("EngramGate",
                                  {{MakeStorageShape(x), ge::DT_BF16, ge::FORMAT_ND},
                                   {MakeStorageShape({}), ge::DT_BF16, ge::FORMAT_ND},
                                   {MakeStorageShape({}), ge::DT_FLOAT, ge::FORMAT_ND}},
                                  outputs,
                                  {{"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6F)},
                                   {"clamp_value", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6F)},
                                   {"save_for_backward", Ops::Transformer::AnyValue::CreateFrom<bool>(save)}});
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, {x, stat, stat, stat, stat});
}
TEST(EngramGateProto, Shape0_True)
{
    CheckShape({17, 3, 65}, true, {17, 3});
}
TEST(EngramGateProto, Shape0_False)
{
    CheckShape({17, 3, 65}, false, {0});
}
TEST(EngramGateProto, Shape1_True)
{
    CheckShape({0, 3, 65}, true, {0, 3});
}
TEST(EngramGateProto, Shape1_False)
{
    CheckShape({0, 3, 65}, false, {0});
}
TEST(EngramGateProto, Shape2_True)
{
    CheckShape({-1, 3, -1}, true, {-1, 3});
}
TEST(EngramGateProto, Shape2_False)
{
    CheckShape({-1, 3, -1}, false, {0});
}
TEST(EngramGateProto, Shape3_True)
{
    CheckShape({-2}, true, {-1, -1});
}
TEST(EngramGateProto, Shape3_False)
{
    CheckShape({-2}, false, {0});
}
TEST(EngramGateProto, DtypeBF16)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("EngramGate");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    ge::DataType x = ge::DT_BF16, k = x, w = ge::DT_FLOAT;
    ge::DataType y = ge::DT_UNDEFINED, a = y, b = y, c = y, d = y;
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(3)
                      .NodeIoNum(3, 5)
                      .NodeInputTd(0, x, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, k, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, w, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&x, &k, &w})
                      .OutputDataTypes({&y, &a, &b, &c, &d})
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), x);
    for (size_t i = 1; i < 5; ++i)
        EXPECT_EQ(context->GetOutputDataType(i), ge::DT_FLOAT);
}
TEST(EngramGateProto, DtypeFLOAT16)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("EngramGate");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    ge::DataType x = ge::DT_FLOAT16, k = x, w = ge::DT_FLOAT;
    ge::DataType y = ge::DT_UNDEFINED, a = y, b = y, c = y, d = y;
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(3)
                      .NodeIoNum(3, 5)
                      .NodeInputTd(0, x, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, k, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, w, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&x, &k, &w})
                      .OutputDataTypes({&y, &a, &b, &c, &d})
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), x);
    for (size_t i = 1; i < 5; ++i)
        EXPECT_EQ(context->GetOutputDataType(i), ge::DT_FLOAT);
}
