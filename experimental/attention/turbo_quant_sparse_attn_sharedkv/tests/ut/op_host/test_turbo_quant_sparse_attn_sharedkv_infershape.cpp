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

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

namespace {
using AnyValue = Ops::Transformer::AnyValue;
using TensorDesc = gert::InfershapeContextPara::TensorDescription;
using OpAttr = gert::InfershapeContextPara::OpAttr;

std::vector<TensorDesc> Inputs(ge::DataType dtype = ge::DT_BF16)
{
    return {{{{3, 64, 512}, {3, 64, 512}}, dtype, ge::FORMAT_ND},
            {{{5, 128, 1, 512}, {5, 128, 1, 512}}, dtype, ge::FORMAT_ND},
            {{{2, 128, 1, 258}, {2, 128, 1, 258}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, 1, 512}, {3, 1, 512}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 5}, {1, 5}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}};
}

std::vector<TensorDesc> Outputs()
{
    return {{{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND}, {{{}, {}}, ge::DT_UNDEFINED, ge::FORMAT_ND}};
}

std::vector<OpAttr> Attrs(bool returnSoftmaxLse)
{
    return {{"softmax_scale", AnyValue::CreateFrom<float>(0.044194173f)},
            {"cmp_ratio", AnyValue::CreateFrom<int64_t>(4)},
            {"ori_mask_mode", AnyValue::CreateFrom<int64_t>(4)},
            {"cmp_mask_mode", AnyValue::CreateFrom<int64_t>(3)},
            {"ori_kv_stride", AnyValue::CreateFrom<int64_t>(65536)},
            {"cmp_kv_stride", AnyValue::CreateFrom<int64_t>(33024)},
            {"ori_win_left", AnyValue::CreateFrom<int64_t>(127)},
            {"ori_win_right", AnyValue::CreateFrom<int64_t>(0)},
            {"layout_q", AnyValue::CreateFrom<std::string>("TND")},
            {"layout_kv", AnyValue::CreateFrom<std::string>("PA_ND")},
            {"return_softmax_lse", AnyValue::CreateFrom<bool>(returnSoftmaxLse)},
            {"kv_quant_mode", AnyValue::CreateFrom<int64_t>(3)}};
}

} // namespace

TEST(TurboQuantSparseAttnSharedkvProto, InferShapeWithoutLse)
{
    gert::InfershapeContextPara para("TurboQuantSparseAttnSharedkv", Inputs(), Outputs(), Attrs(false));
    std::vector<std::vector<int64_t>> expected = {{3, 64, 512}, {0}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(TurboQuantSparseAttnSharedkvProto, InferShapeWithLse)
{
    gert::InfershapeContextPara para("TurboQuantSparseAttnSharedkv", Inputs(), Outputs(), Attrs(true));
    std::vector<std::vector<int64_t>> expected = {{3, 64, 512}, {3, 64, 1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expected);
}

TEST(TurboQuantSparseAttnSharedkvProto, InferDtypeFp16AndBf16)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto func = registry->GetOpImpl("TurboQuantSparseAttnSharedkv")->infer_datatype;
    ASSERT_NE(func, nullptr);
    for (ge::DataType queryType : {ge::DT_FLOAT16, ge::DT_BF16}) {
        auto faker = gert::InferDataTypeContextFaker();
        auto holder = faker.NodeIoNum(14, 2)
                          .NodeInputTd(0, queryType, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                          .SetOpType("TurboQuantSparseAttnSharedkv")
                          .Build();
        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(func(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), queryType);
        EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    }
}
