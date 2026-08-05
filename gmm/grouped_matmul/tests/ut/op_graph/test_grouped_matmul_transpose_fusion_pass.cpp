/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cctype>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "acl/acl_rt.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "graph/graph.h"
// The platform test stub has no public setter for the per-SoC map. Limit private access to this include only.
#define private public
#include "platform/platform_info.h"
#undef private
#include "register/register_custom_pass.h"

#include "../../../op_graph/fusion_pass/grouped_matmul_transpose_fusion_pass.h"
#include "gmm_csv_parse_utils.h"

using namespace ge;
using namespace ge::fusion;

namespace {
constexpr int32_t kGraphFusionSupportVersion = 90100000;
constexpr int32_t kXIndex = 0;
constexpr int32_t kWeightIndex = 1;
constexpr char kGroupedMatmulType[] = "GroupedMatmul";
constexpr char kTransposeType[] = "Transpose";
constexpr char kTransposeDType[] = "TransposeD";
constexpr char kReshapeType[] = "Reshape";
constexpr char kBitcastType[] = "Bitcast";
constexpr char kShapeType[] = "Shape";
constexpr char kGatherType[] = "Gather";
constexpr char kPackType[] = "Pack";
constexpr char kTestCsvFile[] = "test_grouped_matmul_transpose_fusion_pass.csv";
constexpr size_t kCaseFieldNum = 26;
bool gVersionKeyValid = false;
#ifndef GROUPED_MATMUL_TRANSPOSE_FUSION_PASS_CSV_PATH
#define GROUPED_MATMUL_TRANSPOSE_FUSION_PASS_CSV_PATH ""
#endif

struct GroupedMatmulTransposeFusionPassParam {
    std::string caseName;
    std::string scenario;
    std::string transposeOpType;
    int32_t xCount;
    int32_t weightCount;
    std::vector<int64_t> xShape;
    DataType xDtype;
    std::vector<int64_t> weightShape;
    DataType weightDtype;
    std::vector<int64_t> outShape;
    DataType outDtype;
    bool transposeX;
    bool transposeWeight;
    std::vector<int64_t> xPerm;
    std::vector<int64_t> weightPerm;
    bool initialTransposeX;
    bool initialTransposeWeight;
    Status expectedStatus;
    int expectedTransposeCount;
    int expectedReshapeCount;
    int expectedBitcastCount;
    bool expectedTransposeX;
    bool expectedTransposeWeight;
    std::vector<int64_t> expectedXShape;
    std::vector<int64_t> expectedWeightShape;
    DataType expectedWeightDtype;
};

struct TensorWithDesc {
    ge::es::EsTensorHolder tensor;
    TensorDesc desc;
};

class TestGroupedMatmulTransposeFusionPass : public ops::GroupedMatmulTransFusionPass {
public:
    Status RunForTest(GraphPtr &graph, CustomPassContext &passContext)
    {
        return Run(graph, passContext);
    }
};

std::vector<std::string> SplitCsvLine(const std::string &line)
{
    std::vector<std::string> fields;
    ops::ut::SplitStr2Vec(line, ",", fields);
    for (auto &field : fields) {
        field = ops::ut::Trim(field);
    }
    return fields;
}

DataType ParseDataType(const std::string &value)
{
    if (value == "DT_FLOAT16") {
        return DT_FLOAT16;
    }
    if (value == "DT_BF16") {
        return DT_BF16;
    }
    if (value == "DT_INT8") {
        return DT_INT8;
    }
    if (value == "DT_FLOAT") {
        return DT_FLOAT;
    }
    if (value == "DT_HIFLOAT8") {
        return DT_HIFLOAT8;
    }
    if (value == "DT_FLOAT8_E8M0") {
        return DT_FLOAT8_E8M0;
    }
    if (value == "DT_FLOAT4_E2M1") {
        return DT_FLOAT4_E2M1;
    }
    if (value == "DT_FLOAT8_E4M3FN") {
        return DT_FLOAT8_E4M3FN;
    }
    throw std::runtime_error("Unsupported data type: " + value);
}

Status ParseStatus(const std::string &value)
{
    if (value == "SUCCESS") {
        return SUCCESS;
    }
    if (value == "GRAPH_NOT_CHANGED") {
        return GRAPH_NOT_CHANGED;
    }
    if (value == "GRAPH_FAILED") {
        return GRAPH_FAILED;
    }
    throw std::runtime_error("Unsupported status: " + value);
}

GroupedMatmulTransposeFusionPassParam ParseParam(const std::vector<std::string> &fields)
{
    if (fields.size() != kCaseFieldNum) {
        throw std::runtime_error("Invalid csv column size: " + std::to_string(fields.size()));
    }

    size_t index = 0;
    GroupedMatmulTransposeFusionPassParam param;
    param.caseName = fields[index++];
    param.scenario = fields[index++];
    param.transposeOpType = fields[index++];
    param.xCount = std::stoi(fields[index++]);
    param.weightCount = std::stoi(fields[index++]);
    param.xShape = ops::ut::ParseDims(fields[index++]);
    param.xDtype = ParseDataType(fields[index++]);
    param.weightShape = ops::ut::ParseDims(fields[index++]);
    param.weightDtype = ParseDataType(fields[index++]);
    param.outShape = ops::ut::ParseDims(fields[index++]);
    param.outDtype = ParseDataType(fields[index++]);
    param.transposeX = ops::ut::ParseBool(fields[index++]);
    param.transposeWeight = ops::ut::ParseBool(fields[index++]);
    param.xPerm = ops::ut::ParseDims(fields[index++]);
    param.weightPerm = ops::ut::ParseDims(fields[index++]);
    param.initialTransposeX = ops::ut::ParseBool(fields[index++]);
    param.initialTransposeWeight = ops::ut::ParseBool(fields[index++]);
    param.expectedStatus = ParseStatus(fields[index++]);
    param.expectedTransposeCount = std::stoi(fields[index++]);
    param.expectedReshapeCount = std::stoi(fields[index++]);
    param.expectedBitcastCount = std::stoi(fields[index++]);
    param.expectedTransposeX = ops::ut::ParseBool(fields[index++]);
    param.expectedTransposeWeight = ops::ut::ParseBool(fields[index++]);
    param.expectedXShape = ops::ut::ParseDims(fields[index++]);
    param.expectedWeightShape = ops::ut::ParseDims(fields[index++]);
    param.expectedWeightDtype = ParseDataType(fields[index++]);
    return param;
}

std::vector<GroupedMatmulTransposeFusionPassParam> GetParams()
{
    std::string csvPath = GROUPED_MATMUL_TRANSPOSE_FUSION_PASS_CSV_PATH;
    if (csvPath.empty()) {
        csvPath = ops::ut::ResolveCsvPath(kTestCsvFile, "gmm/grouped_matmul/tests/ut/op_graph", __FILE__);
    }
    std::ifstream csvData(csvPath, std::ios::in);
    if (!csvData.is_open()) {
        throw std::runtime_error("Open csv file failed: " + csvPath);
    }

    std::vector<GroupedMatmulTransposeFusionPassParam> params;
    std::string line;
    size_t lineNo = 0;
    while (std::getline(csvData, line)) {
        ++lineNo;
        if (ops::ut::Trim(line).empty()) {
            continue;
        }
        auto fields = SplitCsvLine(line);
        if (!fields.empty() && fields[0] == "case_name") {
            continue;
        }
        try {
            params.emplace_back(ParseParam(fields));
        } catch (const std::exception &error) {
            throw std::runtime_error(
                ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, fields.empty() ? "" : fields[0], error));
        }
    }
    return params;
}

void SetPlatformSupport()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optionalInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
    fe::PlatformInfoManager::Instance().platform_info_map_[optionalInfo.soc_version] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

void ClearPlatformSupport()
{
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
}

std::vector<std::pair<int64_t, int64_t>> MakeShapeRange(const std::vector<int64_t> &shape)
{
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    shapeRange.reserve(shape.size());
    for (const auto dim : shape) {
        shapeRange.emplace_back(dim, dim);
    }
    return shapeRange;
}

TensorDesc MakeTensorDesc(const std::vector<int64_t> &shape, DataType dtype)
{
    TensorDesc desc(Shape(shape), FORMAT_ND, dtype);
    desc.SetOriginFormat(FORMAT_ND);
    desc.SetOriginShape(Shape(shape));
    desc.SetShapeRange(MakeShapeRange(shape));
    return desc;
}

std::vector<int64_t> PermuteShape(const std::vector<int64_t> &shape, const std::vector<int64_t> &perm)
{
    if (shape.size() != perm.size()) {
        throw std::runtime_error("Shape and perm ranks are different.");
    }
    std::vector<int64_t> outputShape;
    outputShape.reserve(shape.size());
    for (const auto index : perm) {
        if (index < 0 || static_cast<size_t>(index) >= shape.size()) {
            throw std::runtime_error("Perm index is out of range.");
        }
        outputShape.emplace_back(shape[static_cast<size_t>(index)]);
    }
    return outputShape;
}

GNode BuildTransposeNode(Graph *graph, const char *opType, const char *name, const TensorDesc &outputDesc,
                         const std::vector<int64_t> &perm)
{
    auto builder = ge::es::CompliantNodeBuilder(graph);
    builder.OpType(opType).Name(name);
    if (std::string(opType) == kTransposeType) {
        builder.IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"perm", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}});
    } else {
        builder.IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}});
    }
    auto transposeNode = builder.IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .InstanceOutputDataType("y", outputDesc.GetDataType())
                             .InstanceOutputShape("y", outputDesc.GetShape().GetDims())
                             .InstanceOutputFormat("y", FORMAT_ND)
                             .Build();
    if (std::string(opType) == kTransposeDType) {
        auto transposePerm = perm;
        transposeNode.SetAttr("perm", transposePerm);
        return transposeNode;
    }

    TensorDesc permDesc(Shape({static_cast<int64_t>(perm.size())}), FORMAT_ND, DT_INT64);
    auto constNode = ge::es::CompliantNodeBuilder(graph)
                         .OpType("Const")
                         .Name((std::string(name) + "_perm").c_str())
                         .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
    constNode.UpdateOutputDesc(0, permDesc);
    Tensor permTensor(permDesc, reinterpret_cast<const uint8_t *>(perm.data()), perm.size() * sizeof(int64_t));
    constNode.SetAttr("value", permTensor);
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, constNode, 0, transposeNode, 1);
    transposeNode.UpdateInputDesc(1, permDesc);
    return transposeNode;
}

void LinkUnaryNode(Graph *graph, ge::es::EsGraphBuilder &graphBuilder, TensorWithDesc &input, GNode &node,
                   const TensorDesc &outputDesc)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(), node, 0);
    node.UpdateInputDesc(0, input.desc);
    node.UpdateOutputDesc(0, outputDesc);
    input.tensor = ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
    input.desc = outputDesc;
}

GNode BuildUnaryNode(Graph *graph, const char *opType, const char *name, const TensorDesc &outputDesc)
{
    return ge::es::CompliantNodeBuilder(graph)
        .OpType(opType)
        .Name(name)
        .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .InstanceOutputDataType("y", outputDesc.GetDataType())
        .InstanceOutputShape("y", outputDesc.GetShape().GetDims())
        .InstanceOutputFormat("y", FORMAT_ND)
        .Build();
}

void AddUnaryNode(Graph *graph, ge::es::EsGraphBuilder &graphBuilder, TensorWithDesc &input, const char *opType,
                  const char *name, const TensorDesc &outputDesc)
{
    auto node = BuildUnaryNode(graph, opType, name, outputDesc);
    LinkUnaryNode(graph, graphBuilder, input, node, outputDesc);
}

GNode BuildInt64ScalarConst(Graph *graph, const char *name, int64_t value)
{
    auto desc = MakeTensorDesc({}, DT_INT64);
    auto constNode = ge::es::CompliantNodeBuilder(graph)
                         .OpType("Const")
                         .Name(name)
                         .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
    constNode.UpdateOutputDesc(0, desc);
    Tensor tensor(desc, reinterpret_cast<const uint8_t *>(&value), sizeof(value));
    constNode.SetAttr("value", tensor);
    return constNode;
}

void AddDynamicSwapReshape(Graph *graph, ge::es::EsGraphBuilder &graphBuilder, TensorWithDesc &input,
                           bool validSwap)
{
    auto shapeDesc = MakeTensorDesc({2}, DT_INT64);
    auto shapeNode = ge::es::CompliantNodeBuilder(graph)
                         .OpType(kShapeType)
                         .Name("shapePerToken")
                         .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
    ge::es::AddEdgeAndUpdatePeerDesc(
        *graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(), shapeNode, 0);
    shapeNode.UpdateInputDesc(0, input.desc);
    shapeNode.UpdateOutputDesc(0, shapeDesc);

    auto gatherNode = ge::es::CompliantNodeBuilder(graph)
                          .OpType(kGatherType)
                          .Name("gatherPerTokenDim")
                          .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"indices", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .Build();
    auto gatherIndexNode = BuildInt64ScalarConst(graph, "gatherPerTokenIndex", validSwap ? 1 : 0);
    auto scalarDesc = MakeTensorDesc({}, DT_INT64);
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, shapeNode, 0, gatherNode, 0);
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, gatherIndexNode, 0, gatherNode, 1);
    gatherNode.UpdateInputDesc(0, shapeDesc);
    gatherNode.UpdateInputDesc(1, scalarDesc);
    gatherNode.UpdateOutputDesc(0, scalarDesc);

    auto packNode = ge::es::CompliantNodeBuilder(graph)
                        .OpType(kPackType)
                        .Name("packPerTokenShape")
                        .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""}})
                        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .InstanceDynamicInputNum("x", 2)
                        .Build();
    auto unitNode = BuildInt64ScalarConst(graph, "perTokenUnitDim", 1);
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, gatherNode, 0, packNode, 0);
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, unitNode, 0, packNode, 1);
    packNode.UpdateInputDesc(0, scalarDesc);
    packNode.UpdateInputDesc(1, scalarDesc);
    packNode.UpdateOutputDesc(0, shapeDesc);

    auto outputDesc = MakeTensorDesc({-1, -1}, input.desc.GetDataType());
    auto reshapeNode = ge::es::CompliantNodeBuilder(graph)
                           .OpType(kReshapeType)
                           .Name("reshapePerTokenDynamicSwap")
                           .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"shape", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .Build();
    ge::es::AddEdgeAndUpdatePeerDesc(
        *graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(), reshapeNode, 0);
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, packNode, 0, reshapeNode, 1);
    reshapeNode.UpdateInputDesc(0, input.desc);
    reshapeNode.UpdateInputDesc(1, shapeDesc);
    reshapeNode.UpdateOutputDesc(0, outputDesc);
    input.tensor = ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reshapeNode, 0));
    input.desc = outputDesc;
}

void AddTransposeIfNeeded(Graph *graph, ge::es::EsGraphBuilder &graphBuilder, TensorWithDesc &input, bool enabled,
                          const char *opType, const char *name, const std::vector<int64_t> &perm)
{
    if (!enabled) {
        return;
    }
    auto outputDesc = MakeTensorDesc(PermuteShape(input.desc.GetShape().GetDims(), perm), input.desc.GetDataType());
    auto transposeNode = BuildTransposeNode(graph, opType, name, outputDesc, perm);
    LinkUnaryNode(graph, graphBuilder, input, transposeNode, outputDesc);
}

void LinkGroupedMatmulInput(Graph *graph, GNode &groupedMatmulNode, const TensorWithDesc &input, int32_t inputIndex)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(),
                                     groupedMatmulNode, inputIndex);
    groupedMatmulNode.UpdateInputDesc(inputIndex, input.desc);
}

GraphPtr BuildGraph(const GroupedMatmulTransposeFusionPassParam &param)
{
    auto graphBuilder = ge::es::EsGraphBuilder(param.caseName.c_str());
    auto *graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto xDesc = MakeTensorDesc(param.xShape, param.xDtype);
    auto weightDesc = MakeTensorDesc(param.weightShape, param.weightDtype);
    auto outDesc = MakeTensorDesc(param.outShape, param.outDtype);

    std::vector<TensorWithDesc> xInputs;
    std::vector<TensorWithDesc> weightInputs;
    int64_t graphInputIndex = 0;
    for (int32_t i = 0; i < param.xCount; ++i) {
        const auto name = "dataX" + std::to_string(i);
        auto input = graphBuilder.CreateInput(graphInputIndex++, name.c_str(), param.xDtype, FORMAT_ND, param.xShape);
        input.GetProducer()->UpdateOutputDesc(0, xDesc);
        xInputs.push_back({input, xDesc});
    }
    for (int32_t i = 0; i < param.weightCount; ++i) {
        const auto name = "dataWeight" + std::to_string(i);
        auto input = graphBuilder.CreateInput(graphInputIndex++, name.c_str(), param.weightDtype, FORMAT_ND,
                                              param.weightShape);
        input.GetProducer()->UpdateOutputDesc(0, weightDesc);
        weightInputs.push_back({input, weightDesc});
    }
    AddTransposeIfNeeded(graph, graphBuilder, xInputs[0], param.transposeX, param.transposeOpType.c_str(),
                         "transposeX", param.xPerm);
    AddTransposeIfNeeded(graph, graphBuilder, weightInputs[0], param.transposeWeight, param.transposeOpType.c_str(),
                         "transposeWeight", param.weightPerm);

    std::vector<std::pair<std::string, TensorWithDesc>> optionalInputs;
    std::vector<ge::es::EsTensorHolder> extraOutputs;
    if (param.scenario == "reshape_pattern") {
        AddUnaryNode(graph, graphBuilder, weightInputs[0], kReshapeType, "reshapeWeight1", weightInputs[0].desc);
        AddTransposeIfNeeded(graph, graphBuilder, weightInputs[0], true, kTransposeDType, "transposeWeight", {1, 0});
        AddUnaryNode(graph, graphBuilder, weightInputs[0], kReshapeType, "reshapeWeight2", weightInputs[0].desc);
    } else if (param.scenario == "transpose_shared") {
        auto sharedOutput = weightInputs[0];
        AddUnaryNode(graph, graphBuilder, sharedOutput, "Identity", "transposeSideConsumer", sharedOutput.desc);
        extraOutputs.emplace_back(sharedOutput.tensor);
    } else if (param.scenario == "bitcast" || param.scenario == "bitcast_shared") {
        const auto bitcastDesc = MakeTensorDesc(weightInputs[0].desc.GetShape().GetDims(), DT_HIFLOAT8);
        AddUnaryNode(graph, graphBuilder, weightInputs[0], kBitcastType, "bitcastWeight", bitcastDesc);
        if (param.scenario == "bitcast_shared") {
            auto sharedOutput = weightInputs[0];
            AddUnaryNode(graph, graphBuilder, sharedOutput, "Identity", "bitcastSideConsumer", bitcastDesc);
            extraOutputs.emplace_back(sharedOutput.tensor);
        }
    } else if (param.scenario == "scale_reshape_valid" || param.scenario == "scale_reshape_invalid" ||
               param.scenario == "scale_reshape_unknown" || param.scenario == "scale_reshape_unknown_invalid" ||
               param.scenario == "scale_reshape_all_unknown" ||
               param.scenario == "mx_scale_reshape_unknown" ||
               param.scenario == "mx_scale_reshape_unknown_invalid" ||
               param.scenario == "mxfp4_scale_reshape_all_unknown" ||
               param.scenario == "mxfp4_scale_reshape_all_unknown_invalid") {
        std::vector<int64_t> inputShape = {2, 1, 64};
        std::vector<int64_t> outputShape = {2, 64, 1};
        DataType scaleDtype = DT_FLOAT;
        if (param.scenario == "scale_reshape_invalid") {
            outputShape = {1, 2, 64};
        } else if (param.scenario == "scale_reshape_unknown") {
            inputShape = {2, -1, 1};
            outputShape = {2, 1, -1};
        } else if (param.scenario == "scale_reshape_unknown_invalid") {
            inputShape = {2, -1, 1};
            outputShape = {-1, 1, 2};
        } else if (param.scenario == "scale_reshape_all_unknown") {
            inputShape = {-1, -1, -1};
            outputShape = {-1, -1, -1};
        } else if (param.scenario == "mx_scale_reshape_unknown") {
            inputShape = {2, -1, 1, 2};
            outputShape = {2, 1, -1, 2};
            scaleDtype = DT_FLOAT8_E8M0;
        } else if (param.scenario == "mx_scale_reshape_unknown_invalid") {
            inputShape = {2, -1, 1, 2};
            outputShape = {-1, 1, 2, 2};
            scaleDtype = DT_FLOAT8_E8M0;
        } else if (param.scenario == "mxfp4_scale_reshape_all_unknown") {
            inputShape = {-1, -1, 1, -1};
            outputShape = {-1, -1, -1, -1};
            scaleDtype = DT_INT8;
        } else if (param.scenario == "mxfp4_scale_reshape_all_unknown_invalid") {
            inputShape = {-1, -1, 2, -1};
            outputShape = {-1, -1, -1, -1};
            scaleDtype = DT_INT8;
        }
        auto scaleDesc = MakeTensorDesc(inputShape, scaleDtype);
        auto scaleTensor =
            graphBuilder.CreateInput(graphInputIndex++, "dataScale", scaleDtype, FORMAT_ND, inputShape);
        scaleTensor.GetProducer()->UpdateOutputDesc(0, scaleDesc);
        TensorWithDesc scaleInput{scaleTensor, scaleDesc};
        AddUnaryNode(graph, graphBuilder, scaleInput, kReshapeType, "reshapeScale",
                     MakeTensorDesc(outputShape, scaleDtype));
        optionalInputs.emplace_back("scale", scaleInput);
    } else if (param.scenario == "per_token_dynamic_swap" ||
               param.scenario == "per_token_dynamic_swap_invalid") {
        const std::vector<int64_t> sourceShape = {1, -1};
        auto perTokenDesc = MakeTensorDesc(sourceShape, DT_FLOAT);
        auto perTokenTensor = graphBuilder.CreateInput(
            graphInputIndex++, "dataPerTokenScale", DT_FLOAT, FORMAT_ND, sourceShape);
        perTokenTensor.GetProducer()->UpdateOutputDesc(0, perTokenDesc);
        TensorWithDesc perTokenInput{perTokenTensor, perTokenDesc};
        AddDynamicSwapReshape(
            graph, graphBuilder, perTokenInput, param.scenario == "per_token_dynamic_swap");
        optionalInputs.emplace_back("per_token_scale", perTokenInput);
    } else if (param.scenario == "mx_per_token") {
        const std::vector<int64_t> scaleShape = {2, 1, 64, 2};
        auto scaleDesc = MakeTensorDesc(scaleShape, DT_FLOAT8_E8M0);
        auto scaleTensor = graphBuilder.CreateInput(
            graphInputIndex++, "dataScale", DT_FLOAT8_E8M0, FORMAT_ND, scaleShape);
        scaleTensor.GetProducer()->UpdateOutputDesc(0, scaleDesc);
        optionalInputs.push_back({"scale", {scaleTensor, scaleDesc}});

        const std::vector<int64_t> sourceShape = {64, 2, 1};
        auto perTokenDesc = MakeTensorDesc(sourceShape, DT_FLOAT8_E8M0);
        auto perTokenTensor = graphBuilder.CreateInput(
            graphInputIndex++, "dataPerTokenScale", DT_FLOAT8_E8M0, FORMAT_ND, sourceShape);
        perTokenTensor.GetProducer()->UpdateOutputDesc(0, perTokenDesc);
        TensorWithDesc perTokenInput{perTokenTensor, perTokenDesc};
        AddUnaryNode(graph, graphBuilder, perTokenInput, kReshapeType, "reshapePerToken1", perTokenDesc);
        AddTransposeIfNeeded(
            graph, graphBuilder, perTokenInput, true, kTransposeDType, "transposePerToken", {1, 0, 2});
        AddUnaryNode(graph, graphBuilder, perTokenInput, kReshapeType, "reshapePerToken2", perTokenInput.desc);
        optionalInputs.emplace_back("per_token_scale", perTokenInput);
    }

    auto groupedMatmulNode = ge::es::CompliantNodeBuilder(graph)
                                 .OpType(kGroupedMatmulType)
                                 .Name("groupedMatmul")
                                 .IrDefInputs({
                                     {"x", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
                                     {"weight", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
                                     {"bias", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                     {"scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                     {"offset", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                     {"antiquant_scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                     {"antiquant_offset", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                     {"group_list", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                     {"per_token_scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                 })
                                 .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                 .InstanceDynamicInputNum("x", param.xCount)
                                 .InstanceDynamicInputNum("weight", param.weightCount)
                                 .InstanceOutputDataType("y", param.outDtype)
                                 .InstanceOutputShape("y", param.outShape)
                                 .InstanceOutputFormat("y", FORMAT_ND)
                                 .Build();
    int32_t physicalInputIndex = 0;
    for (const auto &input : xInputs) {
        LinkGroupedMatmulInput(graph, groupedMatmulNode, input, physicalInputIndex++);
    }
    for (const auto &input : weightInputs) {
        LinkGroupedMatmulInput(graph, groupedMatmulNode, input, physicalInputIndex++);
    }
    for (const auto &optionalInput : optionalInputs) {
        int32_t inputIndex = -1;
        if (groupedMatmulNode.GetInputIndexByName(optionalInput.first.c_str(), inputIndex) != GRAPH_SUCCESS) {
            throw std::runtime_error("Failed to get GroupedMatmul optional input index: " + optionalInput.first);
        }
        LinkGroupedMatmulInput(graph, groupedMatmulNode, optionalInput.second, inputIndex);
    }
    groupedMatmulNode.UpdateOutputDesc(0, outDesc);
    bool transposeX = param.initialTransposeX;
    bool transposeWeight = param.initialTransposeWeight;
    groupedMatmulNode.SetAttr("transpose_x", transposeX);
    groupedMatmulNode.SetAttr("transpose_weight", transposeWeight);

    auto output =
        ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(groupedMatmulNode, 0));
    std::vector<ge::es::EsTensorHolder> outputs = {output};
    outputs.insert(outputs.end(), extraOutputs.begin(), extraOutputs.end());
    return graphBuilder.BuildAndReset(outputs);
}

int CountNodes(const GraphPtr &graph, const char *type)
{
    int count = 0;
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == type) {
            ++count;
        }
    }
    return count;
}

GNode FindGroupedMatmulNode(const GraphPtr &graph)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == kGroupedMatmulType) {
            return node;
        }
    }
    throw std::runtime_error("GroupedMatmul node is not found.");
}

std::string TestName(const testing::TestParamInfo<GroupedMatmulTransposeFusionPassParam> &info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}
} // namespace

extern "C" aclError aclsysGetVersionNum(char *name, int32_t *version)
{
    gVersionKeyValid = name != nullptr && std::strcmp(name, "ge_compiler") == 0;
    if (version != nullptr) {
        *version = kGraphFusionSupportVersion;
    }
    return ACL_SUCCESS;
}

class GroupedMatmulTransposeFusionPassTest : public testing::TestWithParam<GroupedMatmulTransposeFusionPassParam> {};

TEST_P(GroupedMatmulTransposeFusionPassTest, RunFusionPass)
{
    const auto &param = GetParam();
    auto graph = BuildGraph(param);
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestGroupedMatmulTransposeFusionPass pass;
    SetPlatformSupport();
    gVersionKeyValid = false;
    const auto status = pass.RunForTest(graph, passContext);
    ClearPlatformSupport();

    EXPECT_TRUE(gVersionKeyValid);
    ASSERT_EQ(status, param.expectedStatus);
    EXPECT_EQ(CountNodes(graph, kTransposeDType) + CountNodes(graph, kTransposeType), param.expectedTransposeCount);
    EXPECT_EQ(CountNodes(graph, kReshapeType), param.expectedReshapeCount);
    EXPECT_EQ(CountNodes(graph, kBitcastType), param.expectedBitcastCount);

    auto groupedMatmulNode = FindGroupedMatmulNode(graph);
    bool transposeX = false;
    bool transposeWeight = false;
    ASSERT_EQ(groupedMatmulNode.GetAttr("transpose_x", transposeX), GRAPH_SUCCESS);
    ASSERT_EQ(groupedMatmulNode.GetAttr("transpose_weight", transposeWeight), GRAPH_SUCCESS);
    EXPECT_EQ(transposeX, param.expectedTransposeX);
    EXPECT_EQ(transposeWeight, param.expectedTransposeWeight);

    std::vector<int32_t> xIndexes;
    std::vector<int32_t> weightIndexes;
    ASSERT_EQ(groupedMatmulNode.GetDynamicInputIndexesByName("x", xIndexes), GRAPH_SUCCESS);
    ASSERT_EQ(groupedMatmulNode.GetDynamicInputIndexesByName("weight", weightIndexes), GRAPH_SUCCESS);
    ASSERT_FALSE(xIndexes.empty());
    ASSERT_FALSE(weightIndexes.empty());
    TensorDesc xDesc;
    TensorDesc weightDesc;
    ASSERT_EQ(groupedMatmulNode.GetInputDesc(xIndexes[0], xDesc), GRAPH_SUCCESS);
    ASSERT_EQ(groupedMatmulNode.GetInputDesc(weightIndexes[0], weightDesc), GRAPH_SUCCESS);
    EXPECT_EQ(xDesc.GetShape().GetDims(), param.expectedXShape);
    EXPECT_EQ(weightDesc.GetShape().GetDims(), param.expectedWeightShape);
    EXPECT_EQ(weightDesc.GetDataType(), param.expectedWeightDtype);
}

INSTANTIATE_TEST_CASE_P(GroupedMatmulTransposeFusionPass, GroupedMatmulTransposeFusionPassTest,
                        testing::ValuesIn(GetParams()), TestName);
