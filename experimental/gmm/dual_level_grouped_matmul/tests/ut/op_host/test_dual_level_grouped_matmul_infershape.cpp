// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include <vector>
#include "gtest/gtest.h"
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "dual_level_grouped_matmul_ut_param.h"

namespace DualLevelGmmUT {

class DualLevelGmmInferShapeTest : public testing::TestWithParam<DualLevelGmmInferShapeUtParam> {};

namespace {

void RunInferDtype(const DualLevelGmmInferShapeUtParam& param,
                   const std::vector<gert::InfershapeContextPara::TensorDescription>& inputs)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto* impl = registry->GetOpImpl("DualLevelGroupedMatmul");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    std::vector<ge::DataType> inputDtypes;
    inputDtypes.reserve(inputs.size());
    for (const auto& desc : inputs) {
        inputDtypes.emplace_back(desc.dtype_);
    }
    std::vector<void*> inputDtypePtrs;
    inputDtypePtrs.reserve(inputDtypes.size());
    for (auto& dtype : inputDtypes) {
        inputDtypePtrs.emplace_back(&dtype);
    }
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .SetOpType("DualLevelGroupedMatmul")
                      .NodeIoNum(inputs.size(), 1)
                      .InputDataTypes(inputDtypePtrs)
                      .OutputDataTypes({&outputDtype})
                      .NodeAttrs(param.BuildDtypeAttrs())
                      .Build();
    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(impl->infer_datatype(context), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(context->GetOutputDataType(0), param.y.dtype_);
    }
}

} // namespace

// CSV-driven batch infershape/infertype cases; add rows to test_dual_level_grouped_matmul_infershape.csv.
TEST_P(DualLevelGmmInferShapeTest, csvDrivenCase)
{
    const auto& param = GetParam();
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs{
        param.x,     param.xLevel0Scale, param.perTokenScale, param.weight, param.weightLevel0Scale,
        param.scale, param.groupList,    param.bias};
    if (param.RunInferDtypeOnly()) {
        RunInferDtype(param, inputs);
        return;
    }
    gert::InfershapeContextPara para("DualLevelGroupedMatmul", inputs, {param.y},
                                     param.BuildAttrs<gert::InfershapeContextPara>());
    ExecuteTestCase(para, param.expectResult, {param.expectYShape});
}

INSTANTIATE_TEST_SUITE_P(
    DUAL_LEVEL_GMM_INFERSHAPE_CSV, DualLevelGmmInferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<DualLevelGmmInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<DualLevelGmmInferShapeUtParam>);

} // namespace DualLevelGmmUT
