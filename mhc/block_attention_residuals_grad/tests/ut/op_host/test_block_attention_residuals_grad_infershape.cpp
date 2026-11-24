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
#include <memory>
#include <vector>
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_shape_context_faker.h"

namespace {
std::unique_ptr<gert::Tensor> MakeInput(const std::vector<int64_t>& dimensions)
{
    gert::Shape shape;
    shape.SetDimNum(dimensions.size());
    for (size_t i = 0; i < dimensions.size(); ++i) {
        shape.SetDim(i, dimensions[i]);
    }
    gert::StorageShape storageShape;
    storageShape.MutableOriginShape() = shape;
    storageShape.MutableStorageShape() = shape;
    return std::make_unique<gert::Tensor>(storageShape,
                                          gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                          gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
}
} // namespace

TEST(BlockAttentionResidualsGradInferDataType, RejectNullContext)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto impl = registry->GetOpImpl("BlockAttentionResidualsGrad");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    EXPECT_EQ(impl->infer_datatype(nullptr), ge::GRAPH_FAILED);
}

TEST(BlockAttentionResidualsGradInferShape, RejectsEachMissingOutputShape)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto impl = registry->GetOpImpl("BlockAttentionResidualsGrad");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);

    const std::vector<std::vector<int64_t>> inputShapes = {{64, 2048}, {64, 4, 2048}, {1, 2048}, {2048},
                                                           {64, 2048}, {64, 5},       {64, 5}};
    for (size_t missing = 0; missing < 4; ++missing) {
        SCOPED_TRACE(missing);
        std::vector<std::unique_ptr<gert::Tensor>> ownedInputs;
        std::vector<gert::Tensor*> inputTensors;
        for (const auto& shape : inputShapes) {
            ownedInputs.push_back(MakeInput(shape));
            inputTensors.push_back(ownedInputs.back().get());
        }

        gert::InferShapeContextFaker faker;
        faker.SetOpType("BlockAttentionResidualsGrad").NodeIoNum(7, 4);
        for (int32_t i = 0; i < 4; ++i) {
            faker.NodeOutputTd(i, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        }
        faker.InputTensors(inputTensors).Attr("valid_block_num", int64_t{-1});
        auto holder = faker.Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        for (size_t i = 0; i < inputShapes.size(); ++i) {
            ASSERT_NE(context->GetInputShape(i), nullptr);
        }
        for (size_t i = 0; i < 4; ++i) {
            ASSERT_NE(context->GetOutputShape(i), nullptr);
        }
        ASSERT_EQ(impl->infer_shape(context), ge::GRAPH_SUCCESS);
        // KernelContext is a protected base; access the output slot only for this test's fault injection.
        auto* output = reinterpret_cast<gert::KernelContext*>(context)->GetOutput(missing);
        ASSERT_NE(output, nullptr);
        output->Set(nullptr, nullptr);
        for (size_t i = 0; i < 4; ++i) {
            ASSERT_EQ(context->GetOutputShape(i) == nullptr, i == missing);
        }
        EXPECT_EQ(impl->infer_shape(context), ge::GRAPH_FAILED);
    }
}

TEST(BlockAttentionResidualsGradInferShape, RejectsMissingAttributeMetadata)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto impl = registry->GetOpImpl("BlockAttentionResidualsGrad");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);

    const std::vector<std::vector<int64_t>> inputShapes = {{64, 2048}, {64, 4, 2048}, {1, 2048}, {2048},
                                                           {64, 2048}, {64, 5},       {64, 5}};
    for (const bool missingAttrs : {true, false}) {
        SCOPED_TRACE(missingAttrs ? "null attrs" : "missing valid_block_num metadata");
        std::vector<std::unique_ptr<gert::Tensor>> ownedInputs;
        std::vector<gert::Tensor*> inputTensors;
        for (const auto& shape : inputShapes) {
            ownedInputs.push_back(MakeInput(shape));
            inputTensors.push_back(ownedInputs.back().get());
        }

        gert::InferShapeContextFaker faker;
        faker.SetOpType("BlockAttentionResidualsGrad").NodeIoNum(7, 4);
        for (int32_t i = 0; i < 4; ++i) {
            faker.NodeOutputTd(i, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        }
        faker.InputTensors(inputTensors);
        if (missingAttrs) {
            faker.Attr("valid_block_num", int64_t{-1});
        }
        auto holder = faker.Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        for (size_t i = 0; i < inputShapes.size(); ++i) {
            ASSERT_NE(context->GetInputShape(i), nullptr);
        }
        for (size_t i = 0; i < 4; ++i) {
            ASSERT_NE(context->GetOutputShape(i), nullptr);
        }

        if (missingAttrs) {
            ASSERT_NE(context->GetAttrs(), nullptr);
            ASSERT_NE(context->GetAttrs()->GetInt(0), nullptr);
            ASSERT_EQ(impl->infer_shape(context), ge::GRAPH_SUCCESS);
            auto* kernelContext = reinterpret_cast<gert::KernelContext*>(context);
            auto* runContext = kernelContext->GetContext();
            ASSERT_NE(runContext, nullptr);
            auto* originalNodeInfo = runContext->compute_node_info;
            ASSERT_NE(originalNodeInfo, nullptr);
            runContext->compute_node_info = nullptr;
            const bool attrsAreNull = context->GetAttrs() == nullptr;
            const auto result = attrsAreNull ? impl->infer_shape(context) : ge::GRAPH_SUCCESS;
            runContext->compute_node_info = originalNodeInfo;
            EXPECT_TRUE(attrsAreNull);
            EXPECT_EQ(result, ge::GRAPH_FAILED);
        } else {
            ASSERT_NE(context->GetAttrs(), nullptr);
            ASSERT_EQ(context->GetAttrs()->GetInt(0), nullptr);
            EXPECT_EQ(impl->infer_shape(context), ge::GRAPH_FAILED);
        }
    }
}
