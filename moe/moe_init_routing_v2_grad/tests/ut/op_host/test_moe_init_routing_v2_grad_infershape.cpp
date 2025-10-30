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
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MoeInitRoutingV2GradInferShape : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeInitRoutingV2Grad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeInitRoutingV2Grad TearDown" << std::endl;
  }
};

struct MoeInitRoutingV2GradInfo {
    gert::StorageShape &gradExpandedXShape;
    gert::StorageShape &expandedRowIdxShape;
    std::vector<int64_t> expectOutShape;

    ge::DataType gradExpandedXDtype;
    ge::DataType expandedRowIdxDtype;
    ge::DataType yDtype;

    int64_t topK = 0;
    int64_t dropPadMode = 0;
    int64_t active_num = 0;
};

static std::vector<int64_t> ToVector(const gert::Shape &shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
      shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCase(const MoeInitRoutingV2GradInfo ioInfo, ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    /* make infershape context */
    gert::StorageShape yStorageShape = {};
    std::vector<gert::StorageShape *> ouputShapes = {&yStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("MoeInitRoutingV2Grad")
                             .NodeIoNum(2, 1)
                             .NodeInputTd(0, ioInfo.gradExpandedXDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ioInfo.expandedRowIdxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ioInfo.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)

                             .InputTensors({(gert::Tensor *)&ioInfo.gradExpandedXShape})
                             .InputTensors({(gert::Tensor *)&ioInfo.expandedRowIdxShape})
                             .Attr("top_k", int64_t(ioInfo.topK))
                             .Attr("drop_pad_mode", int64_t(ioInfo.dropPadMode))
                             .Attr("active_num", int64_t(ioInfo.active_num))
                             .OutputShapes(ouputShapes)
                             .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("MoeInitRoutingV2Grad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(0)), ioInfo.expectOutShape);
}

TEST_F(MoeInitRoutingV2GradInferShape, moe_init_routing_v2_grad_infershape_0)
{
    gert::StorageShape gradExpandedXShape = {{16, 16}, {16, 16}};
    gert::StorageShape expandedRowIdxShape = {{16}, {16}};
    std::vector<int64_t> expectOutShape = {2, 16}; // scale第0维，expandedX第一维
    MoeInitRoutingV2GradInfo ioInfoT = {gradExpandedXShape,
                                      expandedRowIdxShape,
                                      expectOutShape,
                                      ge::DT_FLOAT,
                                      ge::DT_INT32,
                                      ge::DT_FLOAT,
                                      6, 0, 0};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingV2GradInferShape, moe_init_routing_v2_grad_infer_shape_01)
{
    gert::StorageShape gradExpandedXShape = {{-2}, {-2}};
    gert::StorageShape expandedRowIdxShape = {{-2}, {-2}};
    std::vector<int64_t> expectOutShape = {-1, -1}; // scale第0维，expandedX第一维
    MoeInitRoutingV2GradInfo ioInfoT = {gradExpandedXShape,
                                      expandedRowIdxShape,
                                      expectOutShape,
                                      ge::DT_FLOAT16,
                                      ge::DT_INT32,
                                      ge::DT_FLOAT16,
                                      6, 0, 0};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingV2GradInferShape, moe_init_routing_v2_grad_infer_shape_02)
{
    gert::StorageShape gradExpandedXShape = {{-1, -1}, {-1, -1}};
    gert::StorageShape expandedRowIdxShape = {{-1}, {-1}};
    std::vector<int64_t> expectOutShape = {-1, -1}; // scale第0维，expandedX第一维
    MoeInitRoutingV2GradInfo ioInfoT = {gradExpandedXShape,
                                      expandedRowIdxShape,
                                      expectOutShape,
                                      ge::DT_FLOAT16,
                                      ge::DT_INT32,
                                      ge::DT_FLOAT16,
                                      6, 0, 0};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingV2GradInferShape, moe_init_routing_v2_grad_infer_shape_03)
{
    gert::StorageShape gradExpandedXShape = {{1024, 512}, {1024, 512}};
    gert::StorageShape expandedRowIdxShape = {{1024}, {1024}};
    std::vector<int64_t> expectOutShape = {16, 512}; // scale第0维，expandedX第一维
    MoeInitRoutingV2GradInfo ioInfoT = {gradExpandedXShape,
                                      expandedRowIdxShape,
                                      expectOutShape,
                                      ge::DT_FLOAT16,
                                      ge::DT_INT32,
                                      ge::DT_FLOAT16,
                                      64, 0, 0};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingV2GradInferShape, moe_init_routing_v2_grad_infer_shape_04)
{
    gert::StorageShape gradExpandedXShape = {{1024, 512}, {1024, 512}};
    gert::StorageShape expandedRowIdxShape = {{1024}, {1024}};
    std::vector<int64_t> expectOutShape = {}; // scale第0维，expandedX第一维
    MoeInitRoutingV2GradInfo ioInfoT = {gradExpandedXShape,
                                      expandedRowIdxShape,
                                      expectOutShape,
                                      ge::DT_FLOAT16,
                                      ge::DT_INT32,
                                      ge::DT_FLOAT16,
                                      0, 0, 0};
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradInferShape, moe_init_routing_v2_grad_infer_shape_05)
{
    gert::StorageShape gradExpandedXShape = {{1024, 512}, {1024, 512}};
    gert::StorageShape expandedRowIdxShape = {{1024}, {1024}};
    std::vector<int64_t> expectOutShape = {}; // scale第0维，expandedX第一维
    MoeInitRoutingV2GradInfo ioInfoT = {gradExpandedXShape,
                                      expandedRowIdxShape,
                                      expectOutShape,
                                      ge::DT_FLOAT16,
                                      ge::DT_INT32,
                                      ge::DT_FLOAT16,
                                      64, 2, 0};
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradInferShape, moe_init_routing_v2_grad_infer_shape_06)
{
    gert::StorageShape gradExpandedXShape = {{1024, 512}, {1024, 512}};
    gert::StorageShape expandedRowIdxShape = {{1024}, {1024}};
    std::vector<int64_t> expectOutShape = {}; // scale第0维，expandedX第一维
    MoeInitRoutingV2GradInfo ioInfoT = {gradExpandedXShape,
                                      expandedRowIdxShape,
                                      expectOutShape,
                                      ge::DT_FLOAT16,
                                      ge::DT_INT32,
                                      ge::DT_FLOAT16,
                                      64, 0, -1};
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradInferShape, moe_init_routing_v2_grad_infer_shape_07)
{
    gert::StorageShape gradExpandedXShape = {{1024, 512}, {1024, 512}};
    gert::StorageShape expandedRowIdxShape = {{1024, 512}, {1024, 512}};
    std::vector<int64_t> expectOutShape = {}; // scale第0维，expandedX第一维
    MoeInitRoutingV2GradInfo ioInfoT = {gradExpandedXShape,
                                      expandedRowIdxShape,
                                      expectOutShape,
                                      ge::DT_FLOAT16,
                                      ge::DT_INT32,
                                      ge::DT_FLOAT16,
                                      64, 0, -1};
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradInferShape, moe_init_routing_v2_grad_infer_shape_08)
{
    gert::StorageShape gradExpandedXShape = {{1024, 512}, {1024, 512}};
    gert::StorageShape expandedRowIdxShape = {{1024}, {1024}};
    std::vector<int64_t> expectOutShape = {}; // scale第0维，expandedX第一维
    MoeInitRoutingV2GradInfo ioInfoT = {gradExpandedXShape,
                                      expandedRowIdxShape,
                                      expectOutShape,
                                      ge::DT_FLOAT16,
                                      ge::DT_INT32,
                                      ge::DT_FLOAT16,
                                      64, 2, -1};
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradInferShape, moe_init_routing_v2_grad_infer_shape_09)
{
    gert::StorageShape gradExpandedXShape = {{-1, -1, -1}, {-1, -1, -1}};
    gert::StorageShape expandedRowIdxShape = {{-1}, {-1}};
    std::vector<int64_t> expectOutShape = {-1, -1}; // scale第0维，expandedX第一维
    MoeInitRoutingV2GradInfo ioInfoT = {gradExpandedXShape,
                                      expandedRowIdxShape,
                                      expectOutShape,
                                      ge::DT_FLOAT16,
                                      ge::DT_INT32,
                                      ge::DT_FLOAT16,
                                      6, 1, 0};
    ExeTestCase(ioInfoT, ge::GRAPH_SUCCESS);
}

TEST_F(MoeInitRoutingV2GradInferShape, moe_init_routing_v2_grad_infer_shape_10)
{
    gert::StorageShape gradExpandedXShape = {{-1, -1}, {-1, -1}};
    gert::StorageShape expandedRowIdxShape = {{-1}, {-1}};
    std::vector<int64_t> expectOutShape = {}; // scale第0维，expandedX第一维
    MoeInitRoutingV2GradInfo ioInfoT = {gradExpandedXShape,
                                      expandedRowIdxShape,
                                      expectOutShape,
                                      ge::DT_FLOAT16,
                                      ge::DT_INT32,
                                      ge::DT_FLOAT16,
                                      6, 1, 0};
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}

TEST_F(MoeInitRoutingV2GradInferShape, moe_init_routing_v2_grad_infer_shape_11)
{
    gert::StorageShape gradExpandedXShape = {{1024, 512}, {1024, 512}};
    gert::StorageShape expandedRowIdxShape = {{1024, 512}, {1024, 512}};
    std::vector<int64_t> expectOutShape = {}; // scale第0维，expandedX第一维
    MoeInitRoutingV2GradInfo ioInfoT = {gradExpandedXShape,
                                      expandedRowIdxShape,
                                      expectOutShape,
                                      ge::DT_FLOAT16,
                                      ge::DT_INT32,
                                      ge::DT_FLOAT16,
                                      64, 0, 0};
    ExeTestCase(ioInfoT, ge::GRAPH_FAILED);
}