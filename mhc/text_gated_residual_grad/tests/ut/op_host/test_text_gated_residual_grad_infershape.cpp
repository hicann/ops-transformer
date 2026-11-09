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
#include <limits>
#include "../../../op_kernel/arch35/text_gated_residual_grad_pipeline_arch35.h"

#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_shape_context_faker.h"

class TextGatedResidualGradInferShape : public testing::Test {};

namespace {
constexpr int32_t INPUT_COUNT = 11;
constexpr int32_t OUTPUT_COUNT = 5;
constexpr size_t REQUIRED_OUTPUT_COUNT = 3;
constexpr size_t INPUT_GRAD_INJECT = 7;
constexpr size_t INPUT_X = 1;
constexpr size_t INPUT_GAMMA = 9;
constexpr size_t INPUT_INV_RMS = 4;
constexpr size_t INPUT_INJECT = 10;
constexpr size_t OUTPUT_DX = 0;
constexpr size_t OUTPUT_DW_DOWN = 1;
constexpr size_t OUTPUT_DW_GATE = 3;
constexpr size_t OUTPUT_DGAMMA = 4;
constexpr int64_t CHANNELS = 4;
constexpr int64_t SEQUENCE = 64;
constexpr int64_t FEATURES = 10240;
constexpr int64_t PROJECTION = 320;
constexpr int64_t HIDDEN = FEATURES / CHANNELS;
constexpr int64_t UNKNOWN_RANK = -2;
constexpr int64_t NON_DIVISOR_CHANNELS = 3;
constexpr int64_t INVALID_PROJECTION = -3;
constexpr uint32_t LARGE_FEATURES = 65536;
constexpr uint32_t LARGE_PROJECTION = 65536;
constexpr uint32_t LARGE_CHANNELS = 64;
constexpr uint32_t LARGE_HIDDEN = 1024;
constexpr uint32_t LARGE_AIV_COUNT = 64;
constexpr uint32_t LARGE_SPLIT_K = 2;
constexpr uint32_t LARGE_WORKSPACE_SLOTS = 3;
constexpr uint64_t LARGE_G56_ROWS = 65600;

using TensorDesc = gert::InfershapeContextPara::TensorDescription;

TensorDesc Tensor(const std::vector<int64_t>& dims, ge::DataType dtype)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
    gert::StorageShape storage;
    storage.MutableOriginShape() = shape;
    storage.MutableStorageShape() = shape;
    return {storage, dtype, ge::FORMAT_ND};
}

gert::InfershapeContextPara MakeCase(bool gate = true, bool gamma = true, int64_t b = 1, int64_t s = SEQUENCE,
                                     int64_t d = FEATURES, int64_t r = PROJECTION)
{
    constexpr int64_t c = CHANNELS;
    const int64_t h = d == -1 ? -1 : d / c;
    std::vector<TensorDesc> allInputs{
        Tensor({b, s, h}, ge::DT_BF16), Tensor({b, s, d}, ge::DT_BF16),  Tensor({r, d}, ge::DT_BF16),
        Tensor({d, r}, ge::DT_BF16),    Tensor({b, s, c}, ge::DT_FLOAT), Tensor({b, s, r}, ge::DT_BF16),
        Tensor({b, s, d}, ge::DT_BF16), Tensor({b, s, c}, ge::DT_BF16),  Tensor({c, d}, ge::DT_BF16),
        Tensor({d}, ge::DT_FLOAT),      Tensor({b, s, c}, ge::DT_BF16)};
    const uint32_t g = gate ? 1U : 0U;
    const uint32_t y = gamma ? 1U : 0U;
    std::vector<uint32_t> inputCounts{1, 1, 1, 1, 1, 1, 1, g, g, y, g};
    std::vector<TensorDesc> inputs;
    for (size_t i = 0; i < inputCounts.size(); ++i) {
        if (inputCounts[i]) {
            inputs.push_back(allInputs[i]);
        }
    }
    std::vector<TensorDesc> outputs{Tensor({}, ge::DT_BF16), Tensor({}, ge::DT_BF16), Tensor({}, ge::DT_BF16)};
    if (gate) {
        outputs.push_back(Tensor({}, ge::DT_BF16));
    }
    if (gamma) {
        outputs.push_back(Tensor({}, ge::DT_FLOAT));
    }
    return {"TextGatedResidualGrad",
            inputs,
            outputs,
            {{"channels", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c)}},
            inputCounts,
            {1, 1, 1, g, y}};
}

void CheckDtypes(std::vector<ge::DataType> inputs, ge::graphStatus expected)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto impl = registry->GetOpImpl("TextGatedResidualGrad");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    gert::InferDataTypeContextFaker faker;
    faker.SetOpType("TextGatedResidualGrad").IrInputNum(INPUT_COUNT).NodeIoNum(INPUT_COUNT, OUTPUT_COUNT);
    for (int32_t i = 0; i < INPUT_COUNT; ++i) {
        faker.NodeInputTd(i, inputs[i], ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int32_t i = 0; i < OUTPUT_COUNT; ++i) {
        faker.NodeOutputTd(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = faker.Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(impl->infer_datatype(context), expected);
    if (expected == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(context->GetOutputDataType(OUTPUT_DX), inputs[INPUT_X]);
        for (size_t i = OUTPUT_DW_DOWN; i < OUTPUT_DGAMMA; ++i) {
            EXPECT_EQ(context->GetOutputDataType(i), inputs[INPUT_X]);
        }
        EXPECT_EQ(context->GetOutputDataType(OUTPUT_DGAMMA), ge::DT_FLOAT);
    }
}
} // namespace

TEST_F(TextGatedResidualGradInferShape, AllOptionalCombinations)
{
    for (bool gate : {false, true}) {
        for (bool gamma : {false, true}) {
            auto p = MakeCase(gate, gamma);
            std::vector<std::vector<int64_t>> expected{
                {1, SEQUENCE, FEATURES}, {PROJECTION, FEATURES}, {FEATURES, PROJECTION}};
            if (gate) {
                expected.push_back({CHANNELS, FEATURES});
            }
            if (gamma) {
                expected.push_back({FEATURES});
            }
            ExecuteTestCase(p, ge::GRAPH_SUCCESS, expected);
        }
    }
}

TEST_F(TextGatedResidualGradInferShape, DynamicBatchAndSequence)
{
    auto p = MakeCase(true, true, -1, -1);
    ExecuteTestCase(
        p, ge::GRAPH_SUCCESS,
        {{-1, -1, FEATURES}, {PROJECTION, FEATURES}, {FEATURES, PROJECTION}, {CHANNELS, FEATURES}, {FEATURES}});
}

TEST_F(TextGatedResidualGradInferShape, DynamicHiddenAndProjectionDims)
{
    auto p = MakeCase(true, true, 1, SEQUENCE, -1, -1);
    ExecuteTestCase(p, ge::GRAPH_SUCCESS, {{1, SEQUENCE, -1}, {-1, -1}, {-1, -1}, {CHANNELS, -1}, {-1}});
}

TEST_F(TextGatedResidualGradInferShape, UnknownRank)
{
    for (bool gate : {false, true}) {
        for (bool gamma : {false, true}) {
            auto p = MakeCase(gate, gamma);
            p.inputTensorDesc_[INPUT_X] = Tensor({UNKNOWN_RANK}, ge::DT_BF16);
            std::vector<std::vector<int64_t>> expected(
                REQUIRED_OUTPUT_COUNT + static_cast<size_t>(gate) + static_cast<size_t>(gamma), {UNKNOWN_RANK});
            ExecuteTestCase(p, ge::GRAPH_SUCCESS, expected);
        }
    }
}

TEST_F(TextGatedResidualGradInferShape, EmptyTokenDimension)
{
    auto p = MakeCase(true, true, 1, 0);
    ExecuteTestCase(
        p, ge::GRAPH_SUCCESS,
        {{1, 0, FEATURES}, {PROJECTION, FEATURES}, {FEATURES, PROJECTION}, {CHANNELS, FEATURES}, {FEATURES}});
}

TEST_F(TextGatedResidualGradInferShape, Channels_FAILED)
{
    for (int64_t c : std::initializer_list<int64_t>{0, -1, NON_DIVISOR_CHANNELS}) {
        auto p = MakeCase();
        p.attrs_[0].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(c);
        ExecuteTestCase(p, ge::GRAPH_FAILED);
    }
}

TEST_F(TextGatedResidualGradInferShape, UnknownRankInvalidChannels_FAILED)
{
    for (int64_t c : {0, -1}) {
        auto p = MakeCase();
        p.inputTensorDesc_[INPUT_X] = Tensor({UNKNOWN_RANK}, ge::DT_BF16);
        p.attrs_[0].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(c);
        ExecuteTestCase(p, ge::GRAPH_FAILED);
    }
}

TEST_F(TextGatedResidualGradInferShape, MissingGateOutput_FAILED)
{
    for (bool unknownRank : {false, true}) {
        auto p = MakeCase();
        if (unknownRank) {
            p.inputTensorDesc_[INPUT_X] = Tensor({UNKNOWN_RANK}, ge::DT_BF16);
        }
        p.outputInstanceNum_[OUTPUT_DW_GATE] = 0;
        p.outputTensorDesc_.erase(p.outputTensorDesc_.begin() + OUTPUT_DW_GATE);
        ExecuteTestCase(p, ge::GRAPH_FAILED);
    }
}

TEST_F(TextGatedResidualGradInferShape, MissingGammaOutput_FAILED)
{
    for (bool unknownRank : {false, true}) {
        auto p = MakeCase();
        if (unknownRank) {
            p.inputTensorDesc_[INPUT_X] = Tensor({UNKNOWN_RANK}, ge::DT_BF16);
        }
        p.outputInstanceNum_[OUTPUT_DGAMMA] = 0;
        p.outputTensorDesc_.pop_back();
        ExecuteTestCase(p, ge::GRAPH_FAILED);
    }
}

TEST_F(TextGatedResidualGradInferShape, MissingGateInput_FAILED)
{
    for (bool unknownRank : {false, true}) {
        for (size_t index : {INPUT_GRAD_INJECT, INPUT_INJECT}) {
            auto p = MakeCase();
            if (unknownRank) {
                p.inputTensorDesc_[INPUT_X] = Tensor({UNKNOWN_RANK}, ge::DT_BF16);
            }
            p.inputInstanceNum_[index] = 0;
            p.inputTensorDesc_.erase(p.inputTensorDesc_.begin() + index);
            ExecuteTestCase(p, ge::GRAPH_FAILED);
        }
    }
}

TEST_F(TextGatedResidualGradInferShape, GammaShape_FAILED)
{
    for (auto dims : {std::vector<int64_t>{CHANNELS, HIDDEN}, std::vector<int64_t>{FEATURES - 1}}) {
        auto p = MakeCase();
        p.inputTensorDesc_[INPUT_GAMMA] = Tensor(dims, ge::DT_FLOAT);
        ExecuteTestCase(p, ge::GRAPH_FAILED);
    }
}

TEST_F(TextGatedResidualGradInferShape, InvalidProjectionDimension_FAILED)
{
    for (int64_t r : std::initializer_list<int64_t>{0, INVALID_PROJECTION}) {
        auto p = MakeCase(true, true, 1, SEQUENCE, FEATURES, r);
        ExecuteTestCase(p, ge::GRAPH_FAILED);
    }
}

TEST_F(TextGatedResidualGradInferShape, OutputDtypes)
{
    for (auto dtype : {ge::DT_BF16, ge::DT_FLOAT16}) {
        std::vector<ge::DataType> inputs(INPUT_COUNT, dtype);
        inputs[INPUT_GAMMA] = inputs[INPUT_INV_RMS] = ge::DT_FLOAT;
        CheckDtypes(inputs, ge::GRAPH_SUCCESS);
    }
}

TEST_F(TextGatedResidualGradInferShape, InputDtype_FAILED)
{
    for (size_t index = 0; index < INPUT_COUNT; ++index) {
        std::vector<ge::DataType> inputs(INPUT_COUNT, ge::DT_BF16);
        inputs[INPUT_GAMMA] = inputs[INPUT_INV_RMS] = ge::DT_FLOAT;
        inputs[index] = (index == INPUT_GAMMA || index == INPUT_INV_RMS) ? ge::DT_BF16 : ge::DT_FLOAT;
        CheckDtypes(inputs, ge::GRAPH_FAILED);
    }
}

namespace {
void CheckDtype(gert::InfershapeContextPara para, ge::graphStatus expected)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("TextGatedResidualGrad");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    gert::InferDataTypeContextFaker faker;
    faker.SetOpType("TextGatedResidualGrad")
        .IrInputNum(INPUT_COUNT)
        .IrInstanceNum(para.inputInstanceNum_, para.outputInstanceNum_);
    for (size_t i = 0; i < para.inputTensorDesc_.size(); ++i) {
        faker.NodeInputTd(i, para.inputTensorDesc_[i].dtype_, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (size_t i = 0; i < para.outputTensorDesc_.size(); ++i) {
        faker.NodeOutputTd(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = faker.Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(impl->infer_datatype(context), expected);
    if (expected == ge::GRAPH_SUCCESS) {
        for (size_t i = 0; i < para.outputTensorDesc_.size(); ++i) {
            const bool isDgamma =
                para.outputInstanceNum_[OUTPUT_DGAMMA] != 0U && i + 1U == para.outputTensorDesc_.size();
            EXPECT_EQ(context->GetOutputDataType(i), isDgamma ? ge::DT_FLOAT : para.inputTensorDesc_[0].dtype_);
        }
    }
}
} // namespace

TEST_F(TextGatedResidualGradInferShape, OptionalInputOutputCombinations)
{
    for (bool gate : {false, true}) {
        for (bool gamma : {false, true}) {
            auto para = MakeCase(gate, gamma);
            std::vector<std::vector<int64_t>> shapes = {
                {1, SEQUENCE, FEATURES}, {PROJECTION, FEATURES}, {FEATURES, PROJECTION}};
            if (gate) {
                shapes.push_back({CHANNELS, FEATURES});
            }
            if (gamma) {
                shapes.push_back({FEATURES});
            }
            ExecuteTestCase(para, ge::GRAPH_SUCCESS, shapes);
            for (auto type : {ge::DT_BF16, ge::DT_FLOAT16}) {
                for (auto& tensor : para.inputTensorDesc_) {
                    if (tensor.dtype_ != ge::DT_FLOAT) {
                        tensor.dtype_ = type;
                    }
                }
                CheckDtype(para, ge::GRAPH_SUCCESS);
            }
        }
    }
}

TEST_F(TextGatedResidualGradInferShape, ChannelsNonPositive_FAILED)
{
    for (int64_t channels : {0, -1}) {
        auto para = MakeCase();
        para.attrs_[0].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(channels);
        ExecuteTestCase(para, ge::GRAPH_FAILED);
    }
}

TEST_F(TextGatedResidualGradInferShape, HiddenNotDivisible_FAILED)
{
    auto para = MakeCase();
    para.inputTensorDesc_[INPUT_X].shape_ = {{1, SEQUENCE, FEATURES + 1}, {1, SEQUENCE, FEATURES + 1}};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(TextGatedResidualGradInferShape, MissingOptionalOutput_FAILED)
{
    for (size_t output : {OUTPUT_DW_GATE, OUTPUT_DGAMMA}) {
        auto para = MakeCase();
        para.outputInstanceNum_[output] = 0;
        para.outputTensorDesc_.erase(para.outputTensorDesc_.begin() + output);
        ExecuteTestCase(para, ge::GRAPH_FAILED);
    }
}

TEST_F(TextGatedResidualGradInferShape, MissingGateCompanion_FAILED)
{
    for (size_t input : {INPUT_GRAD_INJECT, INPUT_INJECT}) {
        auto para = MakeCase();
        para.inputInstanceNum_[input] = 0;
        para.inputTensorDesc_.erase(para.inputTensorDesc_.begin() + input);
        ExecuteTestCase(para, ge::GRAPH_FAILED);
    }
}

TEST_F(TextGatedResidualGradInferShape, InputDtypeMismatch_FAILED)
{
    for (size_t input = 0; input < INPUT_COUNT; ++input) {
        auto para = MakeCase();
        para.inputTensorDesc_[input].dtype_ =
            (input == INPUT_GAMMA || input == INPUT_INV_RMS) ? ge::DT_BF16 : ge::DT_FLOAT;
        CheckDtype(para, ge::GRAPH_FAILED);
    }
}

TEST(TextGatedResidualGradWorkspace, PartialSlotBeyondUint32)
{
    // Test layout arithmetic without allocating memory or requiring this shape to fit the vector UB.
    using namespace TextGatedResidualGrad;
    const auto layout =
        BuildHcGradWorkspaceLayout(LARGE_WORKSPACE_SLOTS, LARGE_SPLIT_K, LARGE_AIV_COUNT, LARGE_CHANNELS, LARGE_HIDDEN,
                                   LARGE_PROJECTION, LARGE_FEATURES, sizeof(uint16_t), true);
    const uint64_t expectedSlot = LARGE_G56_ROWS * LARGE_FEATURES;
    ASSERT_GT(expectedSlot, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(layout.dwSlotElems, expectedSlot);
    EXPECT_EQ(layout.dwGateRowElems, static_cast<uint64_t>(LARGE_PROJECTION) * LARGE_FEATURES);
    EXPECT_EQ(layout.dactPartial - layout.dwPartial, HcGradAlign(LARGE_WORKSPACE_SLOTS * expectedSlot * sizeof(float)));
    EXPECT_GT(layout.totalBytes, layout.dactPartial);
}
