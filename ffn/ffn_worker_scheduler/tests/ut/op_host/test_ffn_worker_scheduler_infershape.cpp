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
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class FfnWorkerSchedulerInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FfnWorkerSchedulerInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FfnWorkerSchedulerInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(FfnWorkerSchedulerInfershapeTest, FfnWorkerScheduler_infershape_test01)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::InfershapeContextPara infershapeContextPara("FfnWorkerScheduler",
                                                      {// input
                                                       {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
                                                      {// output
                                                       {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FfnWorkerSchedulerInfershapeTest, FfnWorkerScheduler_inferdtype_test01)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("FfnWorkerScheduler");
    ASSERT_NE(opImpl, nullptr);
    auto dataTypeFunc = opImpl->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);
    ge::DataType dtInt8 = ge::DT_INT8;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .IrInputNum(1)
                             .NodeIoNum(1, 1)
                             .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&dtInt8})
                             .OutputDataTypes({&dtInt8})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
}
