/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <gtest/gtest.h>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
gert::InfershapeContextPara::TensorDescription Tensor(const std::vector<int64_t>& dims, ge::DataType type)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
    gert::StorageShape storage;
    storage.MutableOriginShape() = shape;
    storage.MutableStorageShape() = shape;
    return {storage, type, ge::FORMAT_ND};
}
} // namespace

TEST(MegaMhcInferShape, BothLayoutsAndActivationTypes)
{
    for (auto type : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (bool bs : {false, true}) {
            const std::vector<int64_t> x = bs ? std::vector<int64_t>{2, 17, 2560} : std::vector<int64_t>{34, 2560};
            const std::vector<int64_t> residual =
                bs ? std::vector<int64_t>{2, 17, 4, 2560} : std::vector<int64_t>{34, 4, 2560};
            const std::vector<int64_t> post = bs ? std::vector<int64_t>{2, 17, 4, 1} : std::vector<int64_t>{34, 4, 1};
            const std::vector<int64_t> comb = bs ? std::vector<int64_t>{2, 17, 4, 4} : std::vector<int64_t>{34, 4, 4};
            std::vector<gert::InfershapeContextPara::TensorDescription> inputs{Tensor(x, type),
                                                                               Tensor(residual, type),
                                                                               Tensor(post, ge::DT_FLOAT),
                                                                               Tensor({24, 10240}, ge::DT_FLOAT),
                                                                               Tensor({3}, ge::DT_FLOAT),
                                                                               Tensor({24}, ge::DT_FLOAT),
                                                                               Tensor({2560}, ge::DT_FLOAT),
                                                                               Tensor(comb, ge::DT_FLOAT),
                                                                               Tensor({}, ge::DT_FLOAT),
                                                                               Tensor({2560}, ge::DT_FLOAT),
                                                                               Tensor({}, ge::DT_FLOAT)};
            std::vector<gert::InfershapeContextPara::TensorDescription> outputs{Tensor({}, type),
                                                                                Tensor({}, ge::DT_FLOAT),
                                                                                Tensor({}, ge::DT_FLOAT),
                                                                                Tensor({}, ge::DT_FLOAT),
                                                                                Tensor({}, type),
                                                                                Tensor({}, ge::DT_FLOAT8_E4M3FN),
                                                                                Tensor({}, ge::DT_FLOAT8_E8M0)};
            for (bool external : {false, true}) {
                inputs[8] = Tensor(external ? post : std::vector<int64_t>{}, ge::DT_FLOAT);
                gert::InfershapeContextPara para("MegaMhc", inputs, outputs);
                ExecuteTestCase(para, ge::GRAPH_SUCCESS,
                                {residual, post, comb, external ? post : std::vector<int64_t>{0}, x, {0}, {0}});
            }
        }
    }
}

TEST(MegaMhcInferShape, RejectInvalidInputRank)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs{
        Tensor({2560}, ge::DT_FLOAT16),    Tensor({4, 2560}, ge::DT_FLOAT16), Tensor({4, 1}, ge::DT_FLOAT),
        Tensor({24, 10240}, ge::DT_FLOAT), Tensor({3}, ge::DT_FLOAT),         Tensor({24}, ge::DT_FLOAT),
        Tensor({2560}, ge::DT_FLOAT),      Tensor({4, 4}, ge::DT_FLOAT)};
    std::vector<gert::InfershapeContextPara::TensorDescription> outputs(7, Tensor({}, ge::DT_FLOAT));
    gert::InfershapeContextPara para("MegaMhc", inputs, outputs);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(MegaMhcInferShape, OutputDataTypes)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto impl = registry->GetOpImpl("MegaMhc");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    for (auto type : {ge::DT_FLOAT16, ge::DT_BF16}) {
        gert::InferDataTypeContextFaker faker;
        faker.SetOpType("MegaMhc").IrInputNum(11).NodeIoNum(11, 7).NodeInputTd(0, type, ge::FORMAT_ND, ge::FORMAT_ND);
        for (int32_t i = 0; i < 7; ++i) {
            faker.NodeOutputTd(i, ge::FORMAT_ND, ge::FORMAT_ND);
        }
        auto holder = faker.Build();
        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
        const std::array<ge::DataType, 7> expected{type, ge::DT_FLOAT,         ge::DT_FLOAT,      ge::DT_FLOAT,
                                                   type, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0};
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(context->GetOutputDataType(i), expected[i]);
        }
    }
}
