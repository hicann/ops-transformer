/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_mat_mul_all_reduce_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace ge;

namespace {
class GroupedMatMulAllReduceInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatMulAllReduceInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatMulAllReduceInfershape TearDown" << std::endl;
    }
};

template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

TEST_F(GroupedMatMulAllReduceInfershape, Runtime0)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape xShape0 = {{1, 11}, {}};
    gert::StorageShape xShape1 = {{2, 12}, {}};
    gert::StorageShape xShape2 = {{3, 13}, {}};
    gert::StorageShape xShape3 = {{4, 14}, {}};
    gert::StorageShape weightShape0 = {{11, 71}, {}};
    gert::StorageShape weightShape1 = {{12, 72}, {}};
    gert::StorageShape weightShape2 = {{13, 73}, {}};
    gert::StorageShape weightShape3 = {{14, 74}, {}};
    gert::StorageShape yShape0 = {{}, {}};
    gert::StorageShape yShape1 = {{}, {}};
    gert::StorageShape yShape2 = {{}, {}};
    gert::StorageShape yShape3 = {{}, {}};
    std::vector<gert::Tensor*> inputShapeRef(8);
    inputShapeRef[0] = (gert::Tensor*)&xShape0;
    inputShapeRef[1] = (gert::Tensor*)&xShape1;
    inputShapeRef[2] = (gert::Tensor*)&xShape2;
    inputShapeRef[3] = (gert::Tensor*)&xShape3;
    inputShapeRef[4] = (gert::Tensor*)&weightShape0;
    inputShapeRef[5] = (gert::Tensor*)&weightShape1;
    inputShapeRef[6] = (gert::Tensor*)&weightShape2;
    inputShapeRef[7] = (gert::Tensor*)&weightShape3;
    std::vector<gert::StorageShape*> outputShapeRef(4);
    outputShapeRef[0] = &yShape0;
    outputShapeRef[1] = &yShape1;
    outputShapeRef[2] = &yShape2;
    outputShapeRef[3] = &yShape3;
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("GroupedMatMulAllReduce")
                             .NodeIoNum(8, 4)
                             .InputTensors(inputShapeRef)
                             .OutputShapes(outputShapeRef)
                             .Attr("splitItem", int64_t(0))
                             .Attr("group", AscendString("group"))
                             .Attr("reduceOp", AscendString("sum"))
                             .Attr("commTurn", int64_t(0))
                             .IrInstanceNum({4, 4, 0}, {4})
                             .Build();
    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto outputShape0 = context->GetOutputShape(0);
    EXPECT_EQ(Shape2String(*outputShape0), "[1, 71]");
    auto outputShape1 = context->GetOutputShape(1);
    EXPECT_EQ(Shape2String(*outputShape1), "[2, 72]");
    auto outputShape2 = context->GetOutputShape(2);
    EXPECT_EQ(Shape2String(*outputShape2), "[3, 73]");
    auto outputShape3 = context->GetOutputShape(3);
    EXPECT_EQ(Shape2String(*outputShape3), "[4, 74]");
}

TEST_F(GroupedMatMulAllReduceInfershape, Runtime0MultiDim)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape0 = {{1, 2, 3, 4, 11}, {}};
    gert::StorageShape xShape1 = {{2, 3, 4, 5, 12}, {}};
    gert::StorageShape xShape2 = {{3, 4, 5, 6, 13}, {}};
    gert::StorageShape xShape3 = {{4, 5, 6, 7, 14}, {}};
    gert::StorageShape weightShape0 = {{11, 71}, {}};
    gert::StorageShape weightShape1 = {{12, 72}, {}};
    gert::StorageShape weightShape2 = {{13, 73}, {}};
    gert::StorageShape weightShape3 = {{14, 74}, {}};
    gert::StorageShape yShape0 = {{}, {}};
    gert::StorageShape yShape1 = {{}, {}};
    gert::StorageShape yShape2 = {{}, {}};
    gert::StorageShape yShape3 = {{}, {}};

    std::vector<gert::Tensor*> inputShapeRef(8);
    inputShapeRef[0] = (gert::Tensor*)&xShape0;
    inputShapeRef[1] = (gert::Tensor*)&xShape1;
    inputShapeRef[2] = (gert::Tensor*)&xShape2;
    inputShapeRef[3] = (gert::Tensor*)&xShape3;
    inputShapeRef[4] = (gert::Tensor*)&weightShape0;
    inputShapeRef[5] = (gert::Tensor*)&weightShape1;
    inputShapeRef[6] = (gert::Tensor*)&weightShape2;
    inputShapeRef[7] = (gert::Tensor*)&weightShape3;

    std::vector<gert::StorageShape*> outputShapeRef(4);
    outputShapeRef[0] = &yShape0;
    outputShapeRef[1] = &yShape1;
    outputShapeRef[2] = &yShape2;
    outputShapeRef[3] = &yShape3;

    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("GroupedMatMulAllReduce")
                             .NodeIoNum(8, 4)
                             .IrInstanceNum({4, 4, 0}, {4})
                             .InputTensors(inputShapeRef)
                             .OutputShapes(outputShapeRef)
                             .Attr("splitItem", int64_t(0))
                             .Attr("group", AscendString("group"))
                             .Attr("reduceOp", AscendString("sum"))
                             .Attr("commTurn", int64_t(0))
                             .Build();

    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    auto outputShape0 = context->GetOutputShape(0);
    EXPECT_EQ(Shape2String(*outputShape0), "[1, 2, 3, 4, 71]");

    auto outputShape1 = context->GetOutputShape(1);
    EXPECT_EQ(Shape2String(*outputShape1), "[2, 3, 4, 5, 72]");

    auto outputShape2 = context->GetOutputShape(2);
    EXPECT_EQ(Shape2String(*outputShape2), "[3, 4, 5, 6, 73]");

    auto outputShape3 = context->GetOutputShape(3);
    EXPECT_EQ(Shape2String(*outputShape3), "[4, 5, 6, 7, 74]");
}

TEST_F(GroupedMatMulAllReduceInfershape, Runtime1)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape0 = {{10, 7}, {}};
    gert::StorageShape weightShape0 = {{7, 11}, {}};
    gert::StorageShape weightShape1 = {{7, 22}, {}};
    gert::StorageShape weightShape2 = {{7, 33}, {}};
    gert::StorageShape weightShape3 = {{7, 44}, {}};
    gert::StorageShape yShape0 = {{}, {}};
    gert::StorageShape yShape1 = {{}, {}};
    gert::StorageShape yShape2 = {{}, {}};
    gert::StorageShape yShape3 = {{}, {}};

    int64_t valueSize = 4;
    size_t size = static_cast<size_t>(valueSize) * sizeof(int64_t);
    int64_t* dataInt64 = new int64_t[4];
    dataInt64[0] = 1;
    dataInt64[1] = 3;
    dataInt64[2] = 6;
    dataInt64[3] = 10;
    uint8_t* data = reinterpret_cast<uint8_t*>(dataInt64);

    ge::DataType constDtype = ge::DT_INT64;
    uint8_t* groupListTensorHolder = new uint8_t[sizeof(gert::Tensor) + size];
    auto inputTensor = reinterpret_cast<gert::Tensor*>(groupListTensorHolder);
    std::memcpy(inputTensor + 1, data, size);
    gert::Tensor tensor({{valueSize}, {valueSize}},         // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
                        gert::kFollowing,                   // placement
                        constDtype,                         // dtype
                        nullptr);
    std::memcpy(inputTensor, &tensor, sizeof(gert::Tensor));

    std::vector<gert::Tensor*> inputShapeRef(6);
    inputShapeRef[0] = (gert::Tensor*)&xShape0;
    inputShapeRef[1] = (gert::Tensor*)&weightShape0;
    inputShapeRef[2] = (gert::Tensor*)&weightShape1;
    inputShapeRef[3] = (gert::Tensor*)&weightShape2;
    inputShapeRef[4] = (gert::Tensor*)&weightShape3;
    inputShapeRef[5] = (gert::Tensor*)groupListTensorHolder;

    std::vector<gert::StorageShape*> outputShapeRef(4);
    outputShapeRef[0] = &yShape0;
    outputShapeRef[1] = &yShape1;
    outputShapeRef[2] = &yShape2;
    outputShapeRef[3] = &yShape3;

    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("GroupedMatMulAllReduce")
                             .NodeIoNum(6, 4)
                             .IrInstanceNum({1, 4, 0}, {4})
                             .InputTensors(inputShapeRef)
                             .OutputShapes(outputShapeRef)
                             .Attr("splitItem", int64_t(1))
                             .Attr("group", AscendString("group"))
                             .Attr("reduceOp", AscendString("sum"))
                             .Attr("commTurn", int64_t(0))
                             .Build();

    auto context = contextHolder.GetContext();
    ASSERT_NE(context, (gert::InferShapeContext*)nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    auto outputShape0 = context->GetOutputShape(0);
    EXPECT_EQ(Shape2String(*outputShape0), "[1, 11]");

    auto outputShape1 = context->GetOutputShape(1);
    EXPECT_EQ(Shape2String(*outputShape1), "[2, 22]");

    auto outputShape2 = context->GetOutputShape(2);
    EXPECT_EQ(Shape2String(*outputShape2), "[3, 33]");

    auto outputShape3 = context->GetOutputShape(3);
    EXPECT_EQ(Shape2String(*outputShape3), "[4, 44]");

    delete[] dataInt64;
    delete[] groupListTensorHolder;
}

TEST_F(GroupedMatMulAllReduceInfershape, Runtime1NonIncreasingGroupList)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape0 = {{10, 7}, {}};
    gert::StorageShape weightShape0 = {{7, 11}, {}};
    gert::StorageShape weightShape1 = {{7, 22}, {}};
    gert::StorageShape weightShape2 = {{7, 33}, {}};
    gert::StorageShape weightShape3 = {{7, 44}, {}};
    gert::StorageShape yShape0 = {{}, {}};
    gert::StorageShape yShape1 = {{}, {}};
    gert::StorageShape yShape2 = {{}, {}};
    gert::StorageShape yShape3 = {{}, {}};

    int64_t valueSize = 4;
    size_t size = static_cast<size_t>(valueSize) * sizeof(int64_t);
    int64_t* dataInt64 = new int64_t[4];
    dataInt64[0] = 10;
    dataInt64[1] = 6;
    dataInt64[2] = 3;
    dataInt64[3] = 1;
    uint8_t* data = reinterpret_cast<uint8_t*>(dataInt64);

    ge::DataType constDtype = ge::DT_INT64;
    uint8_t* groupListTensorHolder = new uint8_t[sizeof(gert::Tensor) + size];
    auto inputTensor = reinterpret_cast<gert::Tensor*>(groupListTensorHolder);
    std::memcpy(inputTensor + 1, data, size);
    gert::Tensor tensor({{valueSize}, {valueSize}},         // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
                        gert::kFollowing,                   // placement
                        constDtype,                         // dtype
                        nullptr);
    std::memcpy(inputTensor, &tensor, sizeof(gert::Tensor));

    std::vector<gert::Tensor*> inputShapeRef(6);
    inputShapeRef[0] = (gert::Tensor*)&xShape0;
    inputShapeRef[1] = (gert::Tensor*)&weightShape0;
    inputShapeRef[2] = (gert::Tensor*)&weightShape1;
    inputShapeRef[3] = (gert::Tensor*)&weightShape2;
    inputShapeRef[4] = (gert::Tensor*)&weightShape3;
    inputShapeRef[5] = (gert::Tensor*)groupListTensorHolder;

    std::vector<gert::StorageShape*> outputShapeRef(4);
    outputShapeRef[0] = &yShape0;
    outputShapeRef[1] = &yShape1;
    outputShapeRef[2] = &yShape2;
    outputShapeRef[3] = &yShape3;

    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("GroupedMatMulAllReduce")
                             .NodeIoNum(6, 4)
                             .IrInstanceNum({1, 4, 0}, {4})
                             .InputTensors(inputShapeRef)
                             .OutputShapes(outputShapeRef)
                             .Attr("splitItem", int64_t(1))
                             .Attr("group", AscendString("group"))
                             .Attr("reduceOp", AscendString("sum"))
                             .Attr("commTurn", int64_t(0))
                             .Build();

    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    delete[] dataInt64;
    delete[] groupListTensorHolder;
}

TEST_F(GroupedMatMulAllReduceInfershape, Runtime2)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape0 = {{1, 11}, {}};
    gert::StorageShape xShape1 = {{2, 12}, {}};
    gert::StorageShape xShape2 = {{3, 13}, {}};
    gert::StorageShape xShape3 = {{4, 14}, {}};
    gert::StorageShape weightShape0 = {{11, 10}, {}};
    gert::StorageShape weightShape1 = {{12, 10}, {}};
    gert::StorageShape weightShape2 = {{13, 10}, {}};
    gert::StorageShape weightShape3 = {{14, 10}, {}};
    gert::StorageShape yShape0 = {{}, {}};

    std::vector<gert::Tensor*> inputShapeRef(8);
    inputShapeRef[0] = (gert::Tensor*)&xShape0;
    inputShapeRef[1] = (gert::Tensor*)&xShape1;
    inputShapeRef[2] = (gert::Tensor*)&xShape2;
    inputShapeRef[3] = (gert::Tensor*)&xShape3;
    inputShapeRef[4] = (gert::Tensor*)&weightShape0;
    inputShapeRef[5] = (gert::Tensor*)&weightShape1;
    inputShapeRef[6] = (gert::Tensor*)&weightShape2;
    inputShapeRef[7] = (gert::Tensor*)&weightShape3;

    std::vector<gert::StorageShape*> outputShapeRef(1);
    outputShapeRef[0] = &yShape0;

    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("GroupedMatMulAllReduce")
                             .NodeIoNum(8, 1)
                             .IrInstanceNum({4, 4, 0}, {1})
                             .InputTensors(inputShapeRef)
                             .OutputShapes(outputShapeRef)
                             .Attr("splitItem", int64_t(2))
                             .Attr("group", AscendString("group"))
                             .Attr("reduceOp", AscendString("sum"))
                             .Attr("commTurn", int64_t(0))
                             .Build();
    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    auto outputShape0 = context->GetOutputShape(0);
    EXPECT_EQ(Shape2String(*outputShape0), "[10, 10]");
}

TEST_F(GroupedMatMulAllReduceInfershape, Runtime3)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape0 = {{10, 7}, {}};
    gert::StorageShape weightShape0 = {{7, 10}, {}};
    gert::StorageShape weightShape1 = {{7, 10}, {}};
    gert::StorageShape weightShape2 = {{7, 10}, {}};
    gert::StorageShape weightShape3 = {{7, 10}, {}};
    gert::StorageShape yShape0 = {{}, {}};

    int64_t valueSize = 4;
    size_t size = static_cast<size_t>(valueSize) * sizeof(int64_t);
    int64_t* dataInt64 = new int64_t[4];
    dataInt64[0] = 1;
    dataInt64[1] = 3;
    dataInt64[2] = 6;
    dataInt64[3] = 10;
    uint8_t* data = reinterpret_cast<uint8_t*>(dataInt64);

    ge::DataType constDtype = ge::DT_INT64;
    uint8_t* groupListTensorHolder = new uint8_t[sizeof(gert::Tensor) + size];
    auto inputTensor = reinterpret_cast<gert::Tensor*>(groupListTensorHolder);
    std::memcpy(inputTensor + 1, data, size);
    gert::Tensor tensor({{valueSize}, {valueSize}},         // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
                        gert::kFollowing,                   // placement
                        constDtype,                         // dtype
                        nullptr);
    std::memcpy(inputTensor, &tensor, sizeof(gert::Tensor));

    std::vector<gert::Tensor*> inputShapeRef(6);
    inputShapeRef[0] = (gert::Tensor*)&xShape0;
    inputShapeRef[1] = (gert::Tensor*)&weightShape0;
    inputShapeRef[2] = (gert::Tensor*)&weightShape1;
    inputShapeRef[3] = (gert::Tensor*)&weightShape2;
    inputShapeRef[4] = (gert::Tensor*)&weightShape3;
    inputShapeRef[5] = (gert::Tensor*)groupListTensorHolder;

    std::vector<gert::StorageShape*> outputShapeRef(1);
    outputShapeRef[0] = &yShape0;

    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("GroupedMatMulAllReduce")
                             .NodeIoNum(6, 1)
                             .IrInstanceNum({1, 4, 0}, {1})
                             .InputTensors(inputShapeRef)
                             .OutputShapes(outputShapeRef)
                             .Attr("splitItem", int64_t(3))
                             .Attr("group", AscendString("group"))
                             .Attr("reduceOp", AscendString("sum"))
                             .Attr("commTurn", int64_t(0))
                             .Build();

    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    auto outputShape0 = context->GetOutputShape(0);
    EXPECT_EQ(Shape2String(*outputShape0), "[10, 10]");

    delete[] dataInt64;
    delete[] groupListTensorHolder;
}

TEST_F(GroupedMatMulAllReduceInfershape, InferShapeFor2dim)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{32, 64}, {4, 2, 16, 16}};
    gert::StorageShape weightShape = {{64, 128}, {4, 2, 16, 16}};
    gert::StorageShape biasShape = {{128}, {128}};
    gert::StorageShape groupListShape = {{}, {}};
    gert::StorageShape outputShape = {{}, {}};
    std::vector<gert::Tensor*> inputShapeRef(4);
    inputShapeRef[0] = (gert::Tensor*)&xShape;
    inputShapeRef[1] = (gert::Tensor*)&weightShape;
    inputShapeRef[2] = (gert::Tensor*)&biasShape;
    inputShapeRef[3] = (gert::Tensor*)&groupListShape;
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("GroupedMatMulAllReduce")
                             .NodeIoNum(4, 1)
                             .IrInstanceNum({1, 1, 1}, {1})
                             .InputTensors(inputShapeRef)
                             .OutputShapes({&outputShape})
                             .Attr("splitItem", int64_t(0))
                             .Attr("group", AscendString("group"))
                             .Attr("reduceOp", AscendString("sum"))
                             .Attr("commTurn", int64_t(0))
                             .Build();

    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto output = context->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[32, 128]");
}
TEST_F(GroupedMatMulAllReduceInfershape, InferShapeFor3dim)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{4, 8, 64}, {4, 2, 16, 16}};
    gert::StorageShape weightShape = {{64, 128}, {8, 4, 16, 16}};
    gert::StorageShape biasShape = {{128}, {128}};
    gert::StorageShape groupListShape = {{}, {}};
    gert::StorageShape outputShape = {{}, {}};
    std::vector<gert::Tensor*> inputShapeRef(4);
    inputShapeRef[0] = (gert::Tensor*)&xShape;
    inputShapeRef[1] = (gert::Tensor*)&weightShape;
    inputShapeRef[2] = (gert::Tensor*)&biasShape;
    inputShapeRef[3] = (gert::Tensor*)&groupListShape;

    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("GroupedMatMulAllReduce")
                             .NodeIoNum(4, 1)
                             .IrInstanceNum({1, 1, 1}, {1})
                             .InputTensors(inputShapeRef)
                             .OutputShapes({&outputShape})
                             .Attr("splitItem", int64_t(0))
                             .Attr("group", AscendString("group"))
                             .Attr("reduceOp", AscendString("sum"))
                             .Attr("commTurn", int64_t(0))
                             .Build();
    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto output = context->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[4, 8, 128]");
}

TEST_F(GroupedMatMulAllReduceInfershape, InferShapeForinvalidK)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{32, 8}, {4, 2, 16, 16}};
    gert::StorageShape weightShape = {{64, 128}, {4, 2, 16, 16}};
    gert::StorageShape biasShape = {{128}, {128}};
    gert::StorageShape groupListShape = {{}, {}};
    gert::StorageShape outputShape = {{}, {}};
    std::vector<gert::Tensor*> inputShapeRef(4);
    inputShapeRef[0] = (gert::Tensor*)&xShape;
    inputShapeRef[1] = (gert::Tensor*)&weightShape;
    inputShapeRef[2] = (gert::Tensor*)&biasShape;
    inputShapeRef[3] = (gert::Tensor*)&groupListShape;
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("GroupedMatMulAllReduce")
                             .NodeIoNum(4, 1)
                             .IrInstanceNum({1, 1, 1}, {1})
                             .InputTensors(inputShapeRef)
                             .OutputShapes({&outputShape})
                             .Attr("splitItem", int64_t(0))
                             .Attr("group", AscendString("group"))
                             .Attr("reduceOp", AscendString("sum"))
                             .Attr("commTurn", int64_t(0))
                             .Build();

    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

TEST_F(GroupedMatMulAllReduceInfershape, InferShapeForzeroK)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{32, 0}, {4, 2, 16, 16}};
    gert::StorageShape weightShape = {{0, 128}, {4, 2, 16, 16}};
    gert::StorageShape biasShape = {{128}, {128}};
    gert::StorageShape groupListShape = {{}, {}};
    gert::StorageShape outputShape = {{}, {}};
    std::vector<gert::Tensor*> inputShapeRef(4);
    inputShapeRef[0] = (gert::Tensor*)&xShape;
    inputShapeRef[1] = (gert::Tensor*)&weightShape;
    inputShapeRef[2] = (gert::Tensor*)&biasShape;
    inputShapeRef[3] = (gert::Tensor*)&groupListShape;
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("GroupedMatMulAllReduce")
                             .NodeIoNum(4, 1)
                             .IrInstanceNum({1, 1, 1}, {1})
                             .InputTensors(inputShapeRef)
                             .OutputShapes({&outputShape})
                             .Attr("splitItem", int64_t(0))
                             .Attr("group", AscendString("group"))
                             .Attr("reduceOp", AscendString("sum"))
                             .Attr("commTurn", int64_t(0))
                             .Build();
    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    auto output = context->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[32, 128]");
}

TEST_F(GroupedMatMulAllReduceInfershape, InferDtype)
{
    ge::DataType x1Dtype = ge::DT_FLOAT16;
    ge::DataType x2Dtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_UNDEFINED;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("GroupedMatMulAllReduce")
                             .NodeIoNum(2, 1)
                             .IrInstanceNum({1, 1}, {1})
                             .NodeInputTd(0, x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeAttrs({{"splitItem", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                         {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                         {"reduceOp", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
                                         {"commTurn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
                             .InputDataTypes({&x1Dtype, &x2Dtype})
                             .OutputDataTypes({&yDtype})
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_datatype;

    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_F(GroupedMatMulAllReduceInfershape, InferDtypeTestRuntime2)
{
    ge::DataType xDtype0 = ge::DT_FLOAT16;
    ge::DataType xDtype1 = ge::DT_FLOAT16;
    ge::DataType xDtype2 = ge::DT_FLOAT16;
    ge::DataType xDtype3 = ge::DT_FLOAT16;
    ge::DataType weightDtype0 = ge::DT_FLOAT16;
    ge::DataType weightDtype1 = ge::DT_FLOAT16;
    ge::DataType weightDtype2 = ge::DT_FLOAT16;
    ge::DataType weightDtype3 = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_UNDEFINED;
    std::vector<void*> inputDtypeRef(8);
    inputDtypeRef[0] = &xDtype0;
    inputDtypeRef[1] = &xDtype1;
    inputDtypeRef[2] = &xDtype2;
    inputDtypeRef[3] = &xDtype3;
    inputDtypeRef[4] = &weightDtype0;
    inputDtypeRef[5] = &weightDtype1;
    inputDtypeRef[6] = &weightDtype2;
    inputDtypeRef[7] = &weightDtype3;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("GroupedMatMulAllReduce")
                             .NodeIoNum(2, 1)
                             .IrInstanceNum({1, 1}, {1})
                             .NodeInputTd(0, xDtype0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, xDtype1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, xDtype2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, xDtype3, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(4, weightDtype0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(5, weightDtype1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(6, weightDtype2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(7, weightDtype3, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeAttrs({{"splitItem", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                         {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                         {"reduceOp", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
                                         {"commTurn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
                             .InputDataTypes(inputDtypeRef)
                             .OutputDataTypes({&yDtype})
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_datatype;

    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}
} // namespace
