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
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeFinalizeRoutingHost : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFinalizeRoutingProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFinalizeRoutingProto TearDown" << std::endl;
    }
};

TEST_F(MoeFinalizeRoutingHost, moe_finalize_routing_infershape_0)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRouting",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingHost, moe_finalize_routing_neg_one_shape)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRouting",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingHost, moe_finalize_routing_error_shape)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRouting",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5, 1}, {3, 5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingHost, moe_finalize_routing_infer_data_type)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeFinalizeRouting")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(7, 1)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref1, &input_ref1})
                .OutputDataTypes({&output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}
