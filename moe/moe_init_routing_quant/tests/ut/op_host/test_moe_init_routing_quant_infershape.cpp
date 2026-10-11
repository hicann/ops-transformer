/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_moe_init_routing_quant_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
struct MoeInitRoutingQuantInfo {
    gert::StorageShape& xShape;
    gert::StorageShape& RowIdxShape;
    gert::StorageShape& expertIdxShape;
    std::vector<int64_t> expandedXShape;
    std::vector<int64_t> expandedRowIdxShape;
    std::vector<int64_t> expandedExpertShape;

    ge::DataType XDtype;
    ge::DataType RowIdxDtype;
    ge::DataType expertIdxDtype;
    ge::DataType expandedXDtype;
    ge::DataType expandedRowIdxDtype;
    ge::DataType expandedExpertDtype;
    int64_t activeNum = 0;
};

static std::vector<int64_t> ToVector(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCase(const MoeInitRoutingQuantInfo ioInfo, ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    /* make infershape context */
    gert::StorageShape expandedXStorageShape = {};
    gert::StorageShape expandedRowIdxStorageShape = {};
    gert::StorageShape expandedExpertStorageShape = {};

    std::vector<gert::StorageShape*> ouputShapes = {&expandedXStorageShape, &expandedRowIdxStorageShape,
                                                    &expandedExpertStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("MoeInitRoutingQuant")
                             .NodeIoNum(3, 3)
                             .NodeInputTd(0, ioInfo.XDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ioInfo.RowIdxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ioInfo.expertIdxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ioInfo.expandedXDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ioInfo.expandedRowIdxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ioInfo.expandedExpertDtype, ge::FORMAT_ND, ge::FORMAT_ND)

                             .InputTensors({(gert::Tensor*)&ioInfo.xShape})
                             .InputTensors({(gert::Tensor*)&ioInfo.RowIdxShape})
                             .InputTensors({(gert::Tensor*)&ioInfo.expertIdxShape})

                             .OutputShapes(ouputShapes)
                             .Attr("active_num", int64_t(ioInfo.activeNum))
                             .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("MoeInitRoutingQuant")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(0)), ioInfo.expandedXShape);
    EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(1)), ioInfo.expandedRowIdxShape);
    EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(2)), ioInfo.expandedExpertShape);
}
} // namespace

class MoeInitRoutingQuant : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingQuant SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingQuant TearDown" << std::endl;
    }
};

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_01)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape RowIdxShape = {{2, 4}, {2, 4}};
    gert::StorageShape expertIdxShape = {{2, 4}, {2, 4}};
    std::vector<int64_t> expandedXShap = {8, 3};
    std::vector<int64_t> expandedRowIdxShape = {8};
    std::vector<int64_t> expandedExpertShape = {8};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShap,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       8};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_02)
{
    gert::StorageShape xShape = {{-1, -1}, {-1, -1}};
    gert::StorageShape RowIdxShape = {{-1, -1}, {-1, -1}};
    gert::StorageShape expertIdxShape = {{-1, -1}, {-1, -1}};
    std::vector<int64_t> expandedXShap = {-1, -1};
    std::vector<int64_t> expandedRowIdxShape = {-1};
    std::vector<int64_t> expandedExpertShape = {-1};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShap,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       1};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_03)
{
    gert::StorageShape xShape = {{-1, 2}, {-1, 2}};
    gert::StorageShape RowIdxShape = {{-1, -1}, {-1, -1}};
    gert::StorageShape expertIdxShape = {{-1, -1}, {-1, -1}};
    std::vector<int64_t> expandedXShap = {-1, 2};
    std::vector<int64_t> expandedRowIdxShape = {-1};
    std::vector<int64_t> expandedExpertShape = {-1};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShap,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       1};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_04)
{
    gert::StorageShape xShape = {{2, -1}, {2, -1}};
    gert::StorageShape RowIdxShape = {{-1, -1}, {-1, -1}};
    gert::StorageShape expertIdxShape = {{-1, -1}, {-1, -1}};
    std::vector<int64_t> expandedXShap = {-1, -1};
    std::vector<int64_t> expandedRowIdxShape = {-1};
    std::vector<int64_t> expandedExpertShape = {-1};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShap,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       1};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_05)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape RowIdxShape = {{-1, -1}, {-1, -1}};
    gert::StorageShape expertIdxShape = {{-1, -1}, {-1, -1}};
    std::vector<int64_t> expandedXShap = {-1, 3};
    std::vector<int64_t> expandedRowIdxShape = {-1};
    std::vector<int64_t> expandedExpertShape = {-1};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShap,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       1};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_06)
{
    gert::StorageShape xShape = {{-2}, {-2}};
    gert::StorageShape RowIdxShape = {{-2}, {-2}};
    gert::StorageShape expertIdxShape = {{-1, -1}, {-1, -1}};
    std::vector<int64_t> expandedXShap = {-1, -1};
    std::vector<int64_t> expandedRowIdxShape = {-1};
    std::vector<int64_t> expandedExpertShape = {-1};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShap,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       1};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_07)
{
    gert::StorageShape xShape = {{-1, -1}, {-1, -1}};
    gert::StorageShape RowIdxShape = {{-2}, {-2}};
    gert::StorageShape expertIdxShape = {{-2}, {-2}};
    std::vector<int64_t> expandedXShap = {-1, -1};
    std::vector<int64_t> expandedRowIdxShape = {-1};
    std::vector<int64_t> expandedExpertShape = {-1};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShap,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       1};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_08)
{
    gert::StorageShape xShape = {{-1, 2}, {-1, 2}};
    gert::StorageShape RowIdxShape = {{-1, 3}, {-1, 3}};
    gert::StorageShape expertIdxShape = {{2, -1}, {2, -1}};
    std::vector<int64_t> expandedXShap = {3, 2};
    std::vector<int64_t> expandedRowIdxShape = {6};
    std::vector<int64_t> expandedExpertShape = {6};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShap,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       1};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_09)
{
    gert::StorageShape xShape = {{2, -1}, {2, -1}};
    gert::StorageShape RowIdxShape = {{2, 3}, {2, 3}};
    gert::StorageShape expertIdxShape = {{-1, -1}, {-1, -1}};
    std::vector<int64_t> expandedXShap = {3, -1};
    std::vector<int64_t> expandedRowIdxShape = {6};
    std::vector<int64_t> expandedExpertShape = {6};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShap,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       1};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_10)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape RowIdxShape = {{-1, -1}, {-1, -1}};
    gert::StorageShape expertIdxShape = {{-1, -1}, {-1, -1}};
    std::vector<int64_t> expandedXShap = {-1, 3};
    std::vector<int64_t> expandedRowIdxShape = {-1};
    std::vector<int64_t> expandedExpertShape = {-1};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShap,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       1};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_11)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape RowIdxShape = {{2, 4}, {2, 4}};
    gert::StorageShape expertIdxShape = {{2, 4}, {2, 4}};
    std::vector<int64_t> expandedXShap = {8, 3};
    std::vector<int64_t> expandedRowIdxShape = {8};
    std::vector<int64_t> expandedExpertShape = {8};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShap,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       2};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_12)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape RowIdxShape = {{2, 4}, {2, 4}};
    gert::StorageShape expertIdxShape = {{2, 4}, {2, 4}};
    std::vector<int64_t> expandedXShap = {4, 3};
    std::vector<int64_t> expandedRowIdxShape = {8};
    std::vector<int64_t> expandedExpertShape = {8};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShap,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       1};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_13)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape RowIdxShape = {{2, 4}, {2, 4}};
    gert::StorageShape expertIdxShape = {{2, 4}, {2, 4}};
    std::vector<int64_t> expandedXShap = {8, 3};
    std::vector<int64_t> expandedRowIdxShape = {8};
    std::vector<int64_t> expandedExpertShape = {8};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShap,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       3};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_14)
{
    gert::StorageShape xShape = {{-1}, {-1}};                 // 对应x_tensor_desc形状{-1}
    gert::StorageShape rowIdxShape = {{-1, -1}, {-1, -1}};    // 对应row_idx形状{-1, -1}
    gert::StorageShape expertIdxShape = {{-1, -1}, {-1, -1}}; // 对应expert_idx形状{-1, -1}
    std::vector<int64_t> expandedXShape = {};                 // 失败用例输出为空
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       rowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       5};  // active_num=5
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED); // 期望失败
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_15)
{
    gert::StorageShape xShape = {{-1, -1}, {-1, -1}};         // x形状{-1, -1}
    gert::StorageShape rowIdxShape = {{-1}, {-1}};            // row_idx形状{-1}
    gert::StorageShape expertIdxShape = {{-1, -1}, {-1, -1}}; // expert_idx形状{-1, -1}
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       rowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       5}; // active_num=5
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_16)
{
    gert::StorageShape xShape = {{-1, -1}, {-1, -1}};      // x形状{-1, -1}
    gert::StorageShape rowIdxShape = {{-1, -1}, {-1, -1}}; // row_idx形状{-1, -1}
    gert::StorageShape expertIdxShape = {{-1}, {-1}};      // expert_idx形状{-1}
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       rowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       5}; // active_num=5
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_17)
{
    gert::StorageShape xShape = {{-1, -1}, {-1, -1}};         // x形状{-1, -1}
    gert::StorageShape rowIdxShape = {{2}, {2}};              // row_idx形状{2}
    gert::StorageShape expertIdxShape = {{-1, -1}, {-1, -1}}; // expert_idx形状{-1, -1}
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       rowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       5}; // active_num=5
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_18)
{
    gert::StorageShape xShape = {{-1, -1}, {-1, -1}};         // x形状{-1, -1}
    gert::StorageShape rowIdxShape = {{2}, {2}};              // row_idx形状{2}
    gert::StorageShape expertIdxShape = {{-1, -1}, {-1, -1}}; // expert_idx形状{-1, -1}
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       rowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       5}; // active_num=5
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_19)
{
    gert::StorageShape xShape = {{-1, -1}, {-1, -1}};      // x形状{-1, -1}
    gert::StorageShape rowIdxShape = {{-1, -1}, {-1, -1}}; // row_idx形状{-1, -1}
    gert::StorageShape expertIdxShape = {{2}, {2}};        // expert_idx形状{2}
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       rowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       5}; // active_num=5
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_20)
{
    gert::StorageShape xShape = {{-1, -1}, {-1, -1}};         // x形状{-1, -1}
    gert::StorageShape rowIdxShape = {{-1, -1}, {-1, -1}};    // row_idx形状{-1, -1}
    gert::StorageShape expertIdxShape = {{-1, -1}, {-1, -1}}; // expert_idx形状{-1, -1}
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       rowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       -1}; // active_num=-1
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_21)
{
    gert::StorageShape xShape = {{2, -1}, {2, -1}};           // x形状{2, -1}
    gert::StorageShape rowIdxShape = {{3, -1}, {3, -1}};      // row_idx形状{3, -1}
    gert::StorageShape expertIdxShape = {{-1, -1}, {-1, -1}}; // expert_idx形状{-1, -1}
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       rowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       2}; // active_num=2
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_22)
{
    gert::StorageShape xShape = {{2, -1}, {2, -1}};         // x形状{2, -1}
    gert::StorageShape rowIdxShape = {{-1, -1}, {-1, -1}};  // row_idx形状{-1, -1}
    gert::StorageShape expertIdxShape = {{3, -1}, {3, -1}}; // expert_idx形状{3, -1}
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       rowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       2}; // active_num=2
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_23)
{
    gert::StorageShape xShape = {{-1, -1}, {-1, -1}};       // x形状{-1, -1}
    gert::StorageShape rowIdxShape = {{2, -1}, {2, -1}};    // row_idx形状{2, -1}
    gert::StorageShape expertIdxShape = {{3, -1}, {3, -1}}; // expert_idx形状{3, -1}
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       rowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       2}; // active_num=2
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_24)
{
    gert::StorageShape xShape = {{-1, -1}, {-1, -1}};       // x形状{-1, -1}
    gert::StorageShape rowIdxShape = {{-1, 2}, {-1, 2}};    // row_idx形状{-1, 2}
    gert::StorageShape expertIdxShape = {{-1, 3}, {-1, 3}}; // expert_idx形状{-1, 3}
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       rowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       2}; // active_num=2
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_25)
{
    gert::StorageShape xShape = {{2, 5}, {2, 5}};         // x形状{2, 5}
    gert::StorageShape rowIdxShape = {{2}, {2}};          // row_idx形状{2}
    gert::StorageShape expertIdxShape = {{2, 3}, {2, 3}}; // expert_idx形状{2, 3}
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       rowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       5}; // active_num=5
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_26)
{
    gert::StorageShape xShape = {{2, 5}, {2, 5}};      // x形状{2, 5}
    gert::StorageShape rowIdxShape = {{2, 3}, {2, 3}}; // row_idx形状{2, 3}
    gert::StorageShape expertIdxShape = {{2}, {2}};    // expert_idx形状{2}
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       rowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       5}; // active_num=5
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_invalid_x_dim)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape RowIdxShape = {{2, 4}, {2, 4}};
    gert::StorageShape expertIdxShape = {{2, 4}, {2, 4}};
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       8};
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_invalid_row_idx_dim)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape RowIdxShape = {{2, 4, 1}, {2, 4, 1}};
    gert::StorageShape expertIdxShape = {{2, 4}, {2, 4}};
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       8};
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_invalid_expert_idx_dim)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape RowIdxShape = {{2, 4}, {2, 4}};
    gert::StorageShape expertIdxShape = {{2, 4, 1}, {2, 4, 1}};
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       8};
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_invalid_x_neg_dim)
{
    gert::StorageShape xShape = {{-3, 3}, {-3, 3}};
    gert::StorageShape RowIdxShape = {{2, 4}, {2, 4}};
    gert::StorageShape expertIdxShape = {{2, 4}, {2, 4}};
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       8};
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_datatype_01)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto dataTypeFunc = spaceRegistry->GetOpImpl("MoeInitRoutingQuant")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType inputRef = ge::DT_FLOAT;
    ge::DataType outputRef = ge::DT_INT8;
    ge::DataType idxRef = ge::DT_INT32;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .IrInputNum(3)
                             .NodeIoNum(3, 3)
                             .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&inputRef, &idxRef, &idxRef})
                             .OutputDataTypes({&outputRef, &idxRef, &idxRef})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_INT32);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_invalid_row_idx_neg_dim)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape RowIdxShape = {{-3, 4}, {-3, 4}};
    gert::StorageShape expertIdxShape = {{2, 4}, {2, 4}};
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       8};
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingQuant, moe_init_routing_quant_infer_shape_invalid_expert_idx_neg_dim)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape RowIdxShape = {{2, 4}, {2, 4}};
    gert::StorageShape expertIdxShape = {{-3, 4}, {-3, 4}};
    std::vector<int64_t> expandedXShape = {};
    std::vector<int64_t> expandedRowIdxShape = {};
    std::vector<int64_t> expandedExpertShape = {};
    MoeInitRoutingQuantInfo ioInfoT = {xShape,
                                       RowIdxShape,
                                       expertIdxShape,
                                       expandedXShape,
                                       expandedRowIdxShape,
                                       expandedExpertShape,
                                       ge::DT_FLOAT16,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       ge::DT_INT8,
                                       ge::DT_INT32,
                                       ge::DT_INT32,
                                       8};
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}
