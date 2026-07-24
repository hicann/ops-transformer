/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {

// inferShape用例 ======================================================================================================
class MatmulReduceScatterV2InferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MatmulReduceScatterV2InferShapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "MatmulReduceScatterV2InferShapeTest TearDown" << std::endl; }
};

TEST_F(MatmulReduceScatterV2InferShapeTest, Basic)
{
    gert::StorageShape x1Shape = {{8192, 1536}, {}};
    gert::StorageShape x2Shape = {{1536, 12288}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulReduceScatterV2",
        {
            {x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 12288}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MatmulReduceScatterV2InferShapeTest, EmptyTensorTest)
{
    gert::StorageShape x1Shape = {{8192, 0}, {}};
    gert::StorageShape x2Shape = {{0, 12288}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulReduceScatterV2",
        {
            {x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues);
}

TEST_F(MatmulReduceScatterV2InferShapeTest, Pertensor)
{
    gert::StorageShape x1Shape = {{8192, 1536}, {}};
    gert::StorageShape x2Shape = {{1536, 12288}, {}};
    gert::StorageShape biasShape = {{12288}, {}};
    gert::StorageShape x1ScaleShape = {{1}, {}};
    gert::StorageShape x2ScaleShape = {{1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulReduceScatterV2",
        {
            {x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {biasShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {x1ScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {x2ScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 12288}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MatmulReduceScatterV2InferShapeTest, Perblock)
{
    gert::StorageShape x1Shape = {{8192, 1536}, {}};
    gert::StorageShape x2Shape = {{1536, 12288}, {}};
    gert::StorageShape x1ScaleShape = {{1}, {}};
    gert::StorageShape x2ScaleShape = {{1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulReduceScatterV2",
        {
            {x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {x1ScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {x2ScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 12288}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

// inferDtype用例 ======================================================================================================
class MatmulReduceScatterV2InferDTypeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MatmulReduceScatterV2InferDTypeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "MatmulReduceScatterV2InferDTypeTest TearDown" << std::endl; }
};

// TEST_F(MatmulReduceScatterV2InferDTypeTest, y_dtype_equal_x1_dtype_fp16)
// {
//     ge::DataType x1Type = ge::DT_FLOAT16;

//     auto contextHolder = gert::InferDataTypeContextFaker()
//         .NodeIoNum(1, 2)
//         .InputDataTypes({&x1Type})
//         .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
//         .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
//         .Build();

//     auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
//     auto inferDataTypeFunc = spaceRegistry->GetOpImpl("MatmulReduceScatterV2")->infer_datatype;
//     ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
//     EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), x1Type);
// }


// TEST_F(MatmulReduceScatterV2InferDTypeTest, y_dtype_equal_x1_dtype_bf16)
// {
//     ge::DataType x1Type = ge::DT_BF16;

//     auto contextHolder = gert::InferDataTypeContextFaker()
//         .NodeIoNum(1, 2)
//         .InputDataTypes({&x1Type})
//         .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
//         .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
//         .Build();

//     auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
//     auto inferDataTypeFunc = spaceRegistry->GetOpImpl("MatmulReduceScatterV2")->infer_datatype;
//     ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
//     EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), x1Type);
// }

TEST_F(MatmulReduceScatterV2InferDTypeTest, AttrYDtype)
{
    ge::DataType x1Type = ge::DT_FLOAT8_E4M3FN;
    ge::DataType attrYDtype = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(1, 2)
        .InputDataTypes({&x1Type})
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))}
        })
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("MatmulReduceScatterV2")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), attrYDtype);
}

// inferShape用例 - isAmaxOut=true
TEST_F(MatmulReduceScatterV2InferShapeTest, AmaxOutEnabled)
{
    gert::StorageShape x1Shape = {{8192, 1536}, {}};
    gert::StorageShape x2Shape = {{1536, 12288}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MatmulReduceScatterV2",
        {
            {x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int>(ge::DT_FLOAT))},
            {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 12288}, {1}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

// TilingParse回调测试
class MatmulReduceScatterV2TilingParseTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MatmulReduceScatterV2TilingParseTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "MatmulReduceScatterV2TilingParseTest TearDown" << std::endl; }
};

TEST_F(MatmulReduceScatterV2TilingParseTest, TilingParse_Success)
{
    gert::OpTilingParseContextBuilder builder;
    auto holder = builder.Build();
    auto parseContext = holder.GetContext();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MatmulReduceScatterV2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext*>(parseContext));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

} // namespace
