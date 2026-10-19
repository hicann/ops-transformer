/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"

#include "gmm/quant_grouped_matmul_inplace_add/op_graph/fusion_pass/quant_grouped_matmul_inplace_add_transpose_fusion_pass.h"
#include "gmm_csv_parse_utils.h"

#ifndef QGMM_INPLACE_ADD_TRANSPOSE_PASS_CSV_PATH
#define QGMM_INPLACE_ADD_TRANSPOSE_PASS_CSV_PATH ""
#endif

namespace {
using ge::DataType;
using ge::GNode;
using ge::Graph;
using ge::GraphPtr;
using ge::Shape;
using ge::TensorDesc;

constexpr char kTargetType[] = "QuantGroupedMatmulInplaceAdd";
constexpr char kTransposeDType[] = "TransposeD";
constexpr char kReshapeType[] = "Reshape";

struct FusionCase {
    std::string caseName;
    bool mxMode = false;
    bool transposeScale = false;
    bool reshapeX = false;
    bool validReshape = false;
    ge::Status expectedStatus = ge::GRAPH_NOT_CHANGED;
    int expectedTransposeCount = 0;
    int expectedReshapeCount = 0;
};

FusionCase ParseCase(const std::vector<std::string>& fields)
{
    constexpr size_t kColumnCount = 8;
    if (fields.size() != kColumnCount) {
        throw std::runtime_error("Expected 8 CSV columns, got " + std::to_string(fields.size()));
    }
    FusionCase param;
    param.caseName = ops::ut::Trim(fields[0]);
    if (param.caseName.empty()) {
        throw std::runtime_error("CSV case name must not be empty");
    }
    auto parseBool = [](const std::string& value) {
        const auto token = ops::ut::Trim(value);
        if (token != "0" && token != "1") {
            throw std::runtime_error("Expected boolean 0 or 1, got " + token);
        }
        return token == "1";
    };
    param.mxMode = parseBool(fields[1]);
    param.transposeScale = parseBool(fields[2]);
    param.reshapeX = parseBool(fields[3]);
    param.validReshape = parseBool(fields[4]);
    const auto status = ops::ut::Trim(fields[5]);
    if (status == "SUCCESS") {
        param.expectedStatus = ge::SUCCESS;
    } else if (status != "GRAPH_NOT_CHANGED") {
        throw std::runtime_error("Unsupported expected status: " + status);
    }
    param.expectedTransposeCount = std::stoi(fields[6]);
    param.expectedReshapeCount = std::stoi(fields[7]);
    return param;
}

std::vector<FusionCase> LoadCases()
{
    std::string csvPath = QGMM_INPLACE_ADD_TRANSPOSE_PASS_CSV_PATH;
    if (csvPath.empty()) {
        csvPath = ops::ut::ResolveCsvPath("test_quant_grouped_matmul_inplace_add_transpose_fusion_pass.csv",
                                          "gmm/quant_grouped_matmul_inplace_add/tests/ut/op_graph", __FILE__);
    }
    std::ifstream csv(csvPath);
    if (!csv.is_open()) {
        throw std::runtime_error("Failed to open CSV: " + csvPath);
    }
    std::vector<FusionCase> cases;
    std::string line;
    size_t lineNo = 0;
    while (std::getline(csv, line)) {
        ++lineNo;
        if (ops::ut::Trim(line).empty()) {
            continue;
        }
        std::vector<std::string> fields;
        ops::ut::SplitStr2Vec(line, ",", fields);
        if (lineNo == 1 && ops::ut::Trim(fields[0]) == "case_name") {
            continue;
        }
        try {
            cases.emplace_back(ParseCase(fields));
        } catch (const std::exception& error) {
            throw std::runtime_error(ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, fields[0], error));
        }
    }
    if (cases.empty()) {
        throw std::runtime_error("No test cases in CSV: " + csvPath);
    }
    return cases;
}

struct TensorWithDesc {
    ge::es::EsTensorHolder tensor;
    TensorDesc desc;
};

class TestPass : public ops::QuantGroupedMatmulInplaceAddTransposeFusionPass {
public:
    ge::Status RunForTest(GraphPtr& graph, ge::CustomPassContext& context)
    {
        return Run(graph, context);
    }
};

TensorDesc MakeDesc(const std::vector<int64_t>& shape, DataType dtype)
{
    TensorDesc desc(Shape(shape), ge::FORMAT_ND, dtype);
    desc.SetOriginShape(Shape(shape));
    desc.SetOriginFormat(ge::FORMAT_ND);
    return desc;
}

void Link(Graph* graph, TensorWithDesc& input, GNode& destination, int32_t inputIndex)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(),
                                     destination, inputIndex);
    destination.UpdateInputDesc(inputIndex, input.desc);
}

void AddUnary(Graph* graph, ge::es::EsGraphBuilder& builder, TensorWithDesc& input, const char* type, const char* name,
              const std::vector<int64_t>& outputShape, const std::vector<int64_t>& perm = {})
{
    auto outputDesc = MakeDesc(outputShape, input.desc.GetDataType());
    auto nodeBuilder = ge::es::CompliantNodeBuilder(graph);
    nodeBuilder.OpType(type)
        .Name(name)
        .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .InstanceOutputDataType("y", outputDesc.GetDataType())
        .InstanceOutputShape("y", outputShape)
        .InstanceOutputFormat("y", ge::FORMAT_ND);
    auto node = nodeBuilder.Build();
    if (!perm.empty()) {
        auto attr = perm;
        node.SetAttr("perm", attr);
    }
    Link(graph, input, node, 0);
    node.UpdateOutputDesc(0, outputDesc);
    input.tensor = ge::es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
    input.desc = outputDesc;
}

GraphPtr BuildGraph(bool mxMode, bool transposeScale, bool reshapeX, bool validReshape)
{
    ge::es::EsGraphBuilder builder("qgmmInplaceAddTransposePass");
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    int64_t inputIndex = 0;
    auto createInput = [&](const char* name, const std::vector<int64_t>& shape, DataType dtype) {
        auto desc = MakeDesc(shape, dtype);
        auto tensor = builder.CreateInput(inputIndex++, name, dtype, ge::FORMAT_ND, shape);
        tensor.GetProducer()->UpdateOutputDesc(0, desc);
        return TensorWithDesc{tensor, desc};
    };

    auto x1 = createInput("x1", {64, 32}, ge::DT_HIFLOAT8);
    auto x2 = createInput("x2", {2, 32, 48}, ge::DT_HIFLOAT8);
    auto scale2 = createInput("scale2", {2, 1}, ge::DT_FLOAT);
    auto groupList = createInput("groupList", {2}, ge::DT_INT64);
    auto y = createInput("y", {64, 48}, ge::DT_FLOAT);
    auto scale1 = createInput("scale1", {4, 64, 2}, mxMode ? ge::DT_FLOAT8_E8M0 : ge::DT_FLOAT);

    if (reshapeX) {
        AddUnary(graph, builder, x1, kReshapeType, "reshapeX1",
                 validReshape ? std::vector<int64_t>{32, 64} : std::vector<int64_t>{16, 128});
    } else {
        AddUnary(graph, builder, x1, kTransposeDType, "transposeX1", {32, 64}, {1, 0});
    }
    if (mxMode && transposeScale) {
        AddUnary(graph, builder, scale1, kTransposeDType, "transposeScale1", {64, 4, 2}, {1, 0, 2});
    }

    auto target = ge::es::CompliantNodeBuilder(graph)
                      .OpType(kTargetType)
                      .Name("quantGroupedMatmulInplaceAdd")
                      .IrDefInputs({
                          {"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                          {"x2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                          {"scale2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                          {"group_list", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                          {"y", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                          {"scale1", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                      })
                      .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                      .InstanceOutputDataType("y", ge::DT_FLOAT)
                      .InstanceOutputShape("y", {64, 48})
                      .InstanceOutputFormat("y", ge::FORMAT_ND)
                      .Build();
    Link(graph, x1, target, 0);
    Link(graph, x2, target, 1);
    Link(graph, scale2, target, 2);
    Link(graph, groupList, target, 3);
    Link(graph, y, target, 4);
    Link(graph, scale1, target, 5);
    target.UpdateOutputDesc(0, MakeDesc({64, 48}, ge::DT_FLOAT));
    auto output = ge::es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(target, 0));
    return builder.BuildAndReset({output});
}

int CountType(const GraphPtr& graph, const char* expected)
{
    int count = 0;
    for (auto node : graph->GetAllNodes()) {
        ge::AscendString type;
        if (node.GetType(type) == ge::GRAPH_SUCCESS && type == expected) {
            ++count;
        }
    }
    return count;
}

ge::Status RunPass(GraphPtr& graph)
{
    ge::CustomPassContext context;
    TestPass pass;
    return pass.RunForTest(graph, context);
}
} // namespace

class QuantGroupedMatmulInplaceAddTransposeFusionPassTest : public testing::TestWithParam<FusionCase> {};

TEST_P(QuantGroupedMatmulInplaceAddTransposeFusionPassTest, RunFusionPass)
{
    const auto& param = GetParam();
    auto graph = BuildGraph(param.mxMode, param.transposeScale, param.reshapeX, param.validReshape);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(RunPass(graph), param.expectedStatus);
    EXPECT_EQ(CountType(graph, kTransposeDType), param.expectedTransposeCount);
    EXPECT_EQ(CountType(graph, kReshapeType), param.expectedReshapeCount);
}

INSTANTIATE_TEST_SUITE_P(QuantGroupedMatmulInplaceAddTransposeFusionPass,
                         QuantGroupedMatmulInplaceAddTransposeFusionPassTest, testing::ValuesIn(LoadCases()),
                         [](const testing::TestParamInfo<FusionCase>& info) {
                             return ops::ut::MakeSafeParamName(info.param.caseName);
                         });
