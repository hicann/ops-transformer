/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace all_gather_matmul_ut {

class AllGatherMatmulInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "AllGatherMatmulInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "AllGatherMatmulInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(AllGatherMatmulInferShapeTest, basic) {
    gert::StorageShape x1_shape = {{8192, 12288}, {}};
    gert::StorageShape x2_shape = {{12288, 3904}, {}};

    gert::InfershapeContextPara infershapeContextPara("AllGatherMatmul",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"groupstr", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<int64_t>(false)}
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = {{65536, 3904}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulInferShapeTest, empty_tensor_test) {
    gert::StorageShape x1_shape = {{8192, 0}, {}};
    gert::StorageShape x2_shape = {{0, 3904}, {}};

    gert::InfershapeContextPara infershapeContextPara("AllGatherMatmul",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"groupstr", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<int64_t>(false)}
        }
    );

    ExecuteTestCase(infershapeContextPara);
}

TEST_F(AllGatherMatmulInferShapeTest, is_gather_out_false) {
    gert::StorageShape x1_shape = {{8192, 12288}, {}};
    gert::StorageShape x2_shape = {{12288, 3904}, {}};

    gert::InfershapeContextPara infershapeContextPara("AllGatherMatmul",
        {
            {x1_shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2_shape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"groupstr", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<int64_t>(false)}
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = {{65536, 3904}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulInferShapeTest, infer_datatype) {
    ge::DataType x1_type = ge::DT_FLOAT16;
    ge::DataType x2_type = ge::DT_FLOAT16;
    ge::DataType bias_type = ge::DT_FLOAT16;
    ge::DataType output_type = ge::DT_UNDEFINED;
    ge::DataType gather_output_type = ge::DT_UNDEFINED;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(3, 2)
        .NodeAttrs({{"groupstr", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
                    {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                    {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                    {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                    {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                    {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                    {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<int64_t>(true)}})
        .NodeInputTd(0, x1_type, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, x2_type, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, bias_type, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputDataTypes({&x1_type, &x2_type, &bias_type})
        .OutputDataTypes({&output_type, &gather_output_type})
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("AllGatherMatmul")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_FLOAT16);
}

} // AllGatherMatmulUT