/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
class MoeFinalizeRoutingV2GradProto : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFinalizeRoutingV2GradProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFinalizeRoutingV2GradProto TearDown" << std::endl;
    }
};

TEST_F(MoeFinalizeRoutingV2GradProto, shape_infer)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2Grad",
    {
        {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{15, 8}, {5, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2GradProto, dtype_infer)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeFinalizeRoutingV2Grad"), nullptr);
    auto data_type_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MoeFinalizeRoutingV2Grad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref1 = ge::DT_FLOAT;
        ge::DataType input_ref2 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(6, 2)
                .IrInstanceNum({1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref1, &input_ref2, &input_ref1, &input_ref1, &input_ref2, &input_ref1})
                .OutputDataTypes({&output_ref, &output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
    }
}

TEST_F(MoeFinalizeRoutingV2GradProto, invalid_shape_infer_0)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2Grad",
    {
        {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
    },
    {
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{ 15, 8 }, { 5, 1 }};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2GradProto, shape_infer_option_param_shape_invalid)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2Grad",
    {
        {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {}
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{ 15, 8 }, { 5, 3 }};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2GradProto, shape_infer_set_drop_mode_on)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2Grad",
    {
        {{{5, 8}, {5, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{15}, {15}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{15, 8}, {15, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{5, 3}, {5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{5, 3}, {5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {"drop_pad_mode",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"expert_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"expert_capacity",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"active_num",Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
    }
    );
    std::vector<std::vector<int64_t>> expectOutputShape = {{ 1, 1, 8 }, {5, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

