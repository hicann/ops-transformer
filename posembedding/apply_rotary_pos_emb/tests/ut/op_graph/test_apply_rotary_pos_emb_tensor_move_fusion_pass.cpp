/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "graph/graph.h"

#include "../../../op_graph/fusion_pass/apply_rotary_pos_emb_tensor_move_fusion_pass.h"

using namespace ge;

namespace {
constexpr char kArpeType[] = "ApplyRotaryPosEmb";
constexpr char kTensorMoveType[] = "TensorMove";
constexpr char kReluType[] = "Relu";
constexpr char kArpeName[] = "applyRotaryPosEmb";
constexpr char kTmQueryName[] = "tensorMoveQ";
constexpr char kTmKeyName[] = "tensorMoveK";
constexpr char kCtrlSrcName[] = "ctrlSrc";
constexpr char kCtrlDstName[] = "ctrlDst";
constexpr DataType kDtype = DT_FLOAT16;
const std::vector<int64_t> kQkShape = {2, 3, 4, 8};
const std::vector<int64_t> kCosSinShape = {2, 3, 1, 8};

class TestApplyRotaryPosEmbTensorMoveFusionPass : public ops::ApplyRotaryPosEmbTensorMoveFusionPass {
public:
    Status RunForTest(GraphPtr& graph, CustomPassContext& passContext)
    {
        return Run(graph, passContext);
    }
};

TensorDesc MakeDesc(const std::vector<int64_t>& shape)
{
    return TensorDesc(Shape(shape), FORMAT_ND, kDtype);
}

GNode BuildUnaryNode(Graph* graph, const char* opType, const char* name, const TensorDesc& desc)
{
    return ge::es::CompliantNodeBuilder(graph)
        .OpType(opType)
        .Name(name)
        .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .InstanceOutputDataType("y", desc.GetDataType())
        .InstanceOutputShape("y", desc.GetShape().GetDims())
        .InstanceOutputFormat("y", desc.GetFormat())
        .Build();
}

GNode BuildArpeNode(Graph* graph, const TensorDesc& qkDesc)
{
    return ge::es::CompliantNodeBuilder(graph)
        .OpType(kArpeType)
        .Name(kArpeName)
        .IrDefInputs({{"query", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                      {"key", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                      {"cos", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                      {"sin", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"query", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                       {"key", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .InstanceOutputDataType("query", qkDesc.GetDataType())
        .InstanceOutputShape("query", qkDesc.GetShape().GetDims())
        .InstanceOutputFormat("query", qkDesc.GetFormat())
        .InstanceOutputDataType("key", qkDesc.GetDataType())
        .InstanceOutputShape("key", qkDesc.GetShape().GetDims())
        .InstanceOutputFormat("key", qkDesc.GetFormat())
        .Build();
}

void LinkInput(Graph* graph, const ge::es::EsTensorHolder& src, GNode& dstNode, int32_t dstInputIdx,
               const TensorDesc& desc)
{
    EXPECT_EQ(
        ge::es::AddEdgeAndUpdatePeerDesc(*graph, *src.GetProducer(), src.GetProducerOutIndex(), dstNode, dstInputIdx),
        GRAPH_SUCCESS);
    EXPECT_EQ(dstNode.UpdateInputDesc(dstInputIdx, desc), GRAPH_SUCCESS);
}

ge::es::EsTensorHolder MakeHolder(ge::es::EsGraphBuilder& graphBuilder, GNode& node, int32_t outIdx)
{
    return ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, outIdx));
}

// Graph under test:
//   q -> [TensorMove] -> ApplyRotaryPosEmb.query     k -> [TensorMove] -> ApplyRotaryPosEmb.key
//   cos -> ApplyRotaryPosEmb.cos                     sin -> ApplyRotaryPosEmb.sin
// Optional sharing: shareTmOut adds a Relu consumer on the query TensorMove output,
// shareTmIn adds a Relu consumer on the query data (the TensorMove input source).
GraphPtr BuildArpeGraph(const char* caseName, bool tmOnQuery, bool tmOnKey, bool shareTmOut, bool shareTmIn)
{
    auto graphBuilder = ge::es::EsGraphBuilder(caseName);
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    const auto qkDesc = MakeDesc(kQkShape);
    const auto csDesc = MakeDesc(kCosSinShape);
    auto q = graphBuilder.CreateInput(0, "q", kDtype, FORMAT_ND, kQkShape);
    q.GetProducer()->UpdateOutputDesc(0, qkDesc);
    auto k = graphBuilder.CreateInput(1, "k", kDtype, FORMAT_ND, kQkShape);
    k.GetProducer()->UpdateOutputDesc(0, qkDesc);
    auto cos = graphBuilder.CreateInput(2, "cos", kDtype, FORMAT_ND, kCosSinShape);
    cos.GetProducer()->UpdateOutputDesc(0, csDesc);
    auto sin = graphBuilder.CreateInput(3, "sin", kDtype, FORMAT_ND, kCosSinShape);
    sin.GetProducer()->UpdateOutputDesc(0, csDesc);

    std::vector<ge::es::EsTensorHolder> outputs;
    if (shareTmIn) {
        auto relu = BuildUnaryNode(graph, kReluType, "reluShareTmIn", qkDesc);
        LinkInput(graph, q, relu, 0, qkDesc);
        outputs.emplace_back(MakeHolder(graphBuilder, relu, 0));
    }

    auto queryHolder = q;
    if (tmOnQuery) {
        auto tmNode = BuildUnaryNode(graph, kTensorMoveType, kTmQueryName, qkDesc);
        LinkInput(graph, q, tmNode, 0, qkDesc);
        if (shareTmOut) {
            auto relu = BuildUnaryNode(graph, kReluType, "reluShareTmOut", qkDesc);
            LinkInput(graph, MakeHolder(graphBuilder, tmNode, 0), relu, 0, qkDesc);
            outputs.emplace_back(MakeHolder(graphBuilder, relu, 0));
        }
        queryHolder = MakeHolder(graphBuilder, tmNode, 0);
    }

    auto keyHolder = k;
    if (tmOnKey) {
        auto tmNode = BuildUnaryNode(graph, kTensorMoveType, kTmKeyName, qkDesc);
        LinkInput(graph, k, tmNode, 0, qkDesc);
        keyHolder = MakeHolder(graphBuilder, tmNode, 0);
    }

    auto arpeNode = BuildArpeNode(graph, qkDesc);
    LinkInput(graph, queryHolder, arpeNode, 0, qkDesc);
    LinkInput(graph, keyHolder, arpeNode, 1, qkDesc);
    LinkInput(graph, cos, arpeNode, 2, csDesc);
    LinkInput(graph, sin, arpeNode, 3, csDesc);
    // CANN 9.2.0 GNode::SetAttr takes int64_t by non-const lvalue reference, so the value must be an lvalue.
    int64_t layoutAttr = 1;
    EXPECT_EQ(arpeNode.SetAttr("layout", layoutAttr), GRAPH_SUCCESS);

    outputs.emplace_back(MakeHolder(graphBuilder, arpeNode, 0));
    outputs.emplace_back(MakeHolder(graphBuilder, arpeNode, 1));
    return graphBuilder.BuildAndReset(outputs);
}

int CountNodes(const GraphPtr& graph, const char* type)
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

std::string GetProducerName(const GNode& node, int32_t inputIdx)
{
    const auto srcNodeAndPort = node.GetInDataNodesAndPortIndexs(inputIdx);
    if (srcNodeAndPort.first == nullptr) {
        return "";
    }
    AscendString name;
    if (srcNodeAndPort.first->GetName(name) != GRAPH_SUCCESS || name.GetString() == nullptr) {
        return "";
    }
    return name.GetString();
}

bool HasControlNode(const std::vector<GNodePtr>& ctrlNodes, const char* name)
{
    for (const auto& ctrlNode : ctrlNodes) {
        AscendString nodeName;
        if (ctrlNode != nullptr && ctrlNode->GetName(nodeName) == GRAPH_SUCCESS && nodeName == name) {
            return true;
        }
    }
    return false;
}
} // namespace

TEST(ApplyRotaryPosEmbTensorMoveFusionPassTest, NullGraphNotChanged)
{
    GraphPtr graph = nullptr;
    CustomPassContext passContext;
    TestApplyRotaryPosEmbTensorMoveFusionPass pass;
    EXPECT_EQ(pass.RunForTest(graph, passContext), GRAPH_NOT_CHANGED);
}

TEST(ApplyRotaryPosEmbTensorMoveFusionPassTest, NoTensorMoveNotChanged)
{
    auto graph = BuildArpeGraph("no_tensor_move", false, false, false, false);
    ASSERT_NE(graph, nullptr);
    CustomPassContext passContext;
    TestApplyRotaryPosEmbTensorMoveFusionPass pass;
    EXPECT_EQ(pass.RunForTest(graph, passContext), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, kTensorMoveType), 0);
}

TEST(ApplyRotaryPosEmbTensorMoveFusionPassTest, TensorMoveOnQueryAndKeyFused)
{
    auto graph = BuildArpeGraph("tm_on_query_and_key", true, true, false, false);
    ASSERT_NE(graph, nullptr);
    CustomPassContext passContext;
    TestApplyRotaryPosEmbTensorMoveFusionPass pass;
    EXPECT_EQ(pass.RunForTest(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodes(graph, kTensorMoveType), 0);
    const auto arpeNode = graph->FindNodeByName(AscendString(kArpeName));
    ASSERT_NE(arpeNode, nullptr);
    EXPECT_EQ(GetProducerName(*arpeNode, 0), "q");
    EXPECT_EQ(GetProducerName(*arpeNode, 1), "k");
}

TEST(ApplyRotaryPosEmbTensorMoveFusionPassTest, TensorMoveOnQueryOnlyFused)
{
    auto graph = BuildArpeGraph("tm_on_query_only", true, false, false, false);
    ASSERT_NE(graph, nullptr);
    CustomPassContext passContext;
    TestApplyRotaryPosEmbTensorMoveFusionPass pass;
    EXPECT_EQ(pass.RunForTest(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodes(graph, kTensorMoveType), 0);
    const auto arpeNode = graph->FindNodeByName(AscendString(kArpeName));
    ASSERT_NE(arpeNode, nullptr);
    EXPECT_EQ(GetProducerName(*arpeNode, 0), "q");
    EXPECT_EQ(GetProducerName(*arpeNode, 1), "k");
}

TEST(ApplyRotaryPosEmbTensorMoveFusionPassTest, TensorMoveOnKeyOnlyFused)
{
    auto graph = BuildArpeGraph("tm_on_key_only", false, true, false, false);
    ASSERT_NE(graph, nullptr);
    CustomPassContext passContext;
    TestApplyRotaryPosEmbTensorMoveFusionPass pass;
    EXPECT_EQ(pass.RunForTest(graph, passContext), SUCCESS);

    EXPECT_EQ(CountNodes(graph, kTensorMoveType), 0);
    const auto arpeNode = graph->FindNodeByName(AscendString(kArpeName));
    ASSERT_NE(arpeNode, nullptr);
    EXPECT_EQ(GetProducerName(*arpeNode, 0), "q");
    EXPECT_EQ(GetProducerName(*arpeNode, 1), "k");
}

TEST(ApplyRotaryPosEmbTensorMoveFusionPassTest, TensorMoveOutputMultiConsumerNotFused)
{
    auto graph = BuildArpeGraph("tm_out_multi_consumer", true, false, true, false);
    ASSERT_NE(graph, nullptr);
    CustomPassContext passContext;
    TestApplyRotaryPosEmbTensorMoveFusionPass pass;
    EXPECT_EQ(pass.RunForTest(graph, passContext), GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, kTensorMoveType), 1);
    const auto arpeNode = graph->FindNodeByName(AscendString(kArpeName));
    ASSERT_NE(arpeNode, nullptr);
    EXPECT_EQ(GetProducerName(*arpeNode, 0), kTmQueryName);
}

TEST(ApplyRotaryPosEmbTensorMoveFusionPassTest, TensorMoveInputMultiConsumerNotFused)
{
    auto graph = BuildArpeGraph("tm_in_multi_consumer", true, false, false, true);
    ASSERT_NE(graph, nullptr);
    CustomPassContext passContext;
    TestApplyRotaryPosEmbTensorMoveFusionPass pass;
    EXPECT_EQ(pass.RunForTest(graph, passContext), GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodes(graph, kTensorMoveType), 1);
    const auto arpeNode = graph->FindNodeByName(AscendString(kArpeName));
    ASSERT_NE(arpeNode, nullptr);
    EXPECT_EQ(GetProducerName(*arpeNode, 0), kTmQueryName);
}

TEST(ApplyRotaryPosEmbTensorMoveFusionPassTest, ControlEdgesTransferredToArpe)
{
    auto graphBuilder = ge::es::EsGraphBuilder("ctrl_edge_transfer");
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    const auto qkDesc = MakeDesc(kQkShape);
    auto ctrlSrc = BuildUnaryNode(graph, kReluType, kCtrlSrcName, qkDesc);
    auto ctrlDst = BuildUnaryNode(graph, kReluType, kCtrlDstName, qkDesc);
    auto q = graphBuilder.CreateInput(0, "q", kDtype, FORMAT_ND, kQkShape);
    q.GetProducer()->UpdateOutputDesc(0, qkDesc);
    auto k = graphBuilder.CreateInput(1, "k", kDtype, FORMAT_ND, kQkShape);
    k.GetProducer()->UpdateOutputDesc(0, qkDesc);
    auto cos = graphBuilder.CreateInput(2, "cos", kDtype, FORMAT_ND, kCosSinShape);
    cos.GetProducer()->UpdateOutputDesc(0, MakeDesc(kCosSinShape));
    auto sin = graphBuilder.CreateInput(3, "sin", kDtype, FORMAT_ND, kCosSinShape);
    sin.GetProducer()->UpdateOutputDesc(0, MakeDesc(kCosSinShape));

    auto tmNode = BuildUnaryNode(graph, kTensorMoveType, kTmQueryName, qkDesc);
    LinkInput(graph, q, tmNode, 0, qkDesc);
    auto arpeNode = BuildArpeNode(graph, qkDesc);
    LinkInput(graph, MakeHolder(graphBuilder, tmNode, 0), arpeNode, 0, qkDesc);
    LinkInput(graph, k, arpeNode, 1, qkDesc);
    LinkInput(graph, cos, arpeNode, 2, MakeDesc(kCosSinShape));
    LinkInput(graph, sin, arpeNode, 3, MakeDesc(kCosSinShape));

    // ctrlSrc --> TensorMove --> ctrlDst control edges; ctrlDst consumes nothing, keep it as a graph output.
    ASSERT_EQ(graph->AddControlEdge(ctrlSrc, tmNode), GRAPH_SUCCESS);
    ASSERT_EQ(graph->AddControlEdge(tmNode, ctrlDst), GRAPH_SUCCESS);
    std::vector<ge::es::EsTensorHolder> outputs = {
        MakeHolder(graphBuilder, arpeNode, 0), MakeHolder(graphBuilder, arpeNode, 1),
        MakeHolder(graphBuilder, ctrlSrc, 0), MakeHolder(graphBuilder, ctrlDst, 0)};
    GraphPtr graphPtr = graphBuilder.BuildAndReset(outputs);
    ASSERT_NE(graphPtr, nullptr);

    CustomPassContext passContext;
    TestApplyRotaryPosEmbTensorMoveFusionPass pass;
    EXPECT_EQ(pass.RunForTest(graphPtr, passContext), SUCCESS);

    EXPECT_EQ(CountNodes(graphPtr, kTensorMoveType), 0);
    const auto fusedArpeNode = graphPtr->FindNodeByName(AscendString(kArpeName));
    ASSERT_NE(fusedArpeNode, nullptr);
    EXPECT_EQ(GetProducerName(*fusedArpeNode, 0), "q");
    EXPECT_TRUE(HasControlNode(fusedArpeNode->GetInControlNodes(), kCtrlSrcName));
    EXPECT_TRUE(HasControlNode(fusedArpeNode->GetOutControlNodes(), kCtrlDstName));
}
