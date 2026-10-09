/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "graph/graph.h"
#define private public
#include "platform/platform_info.h"
#undef private
#include "register/register_custom_pass.h"

#include "../../../op_graph/fusion_pass/rotary_position_embedding_grad_fusion_pass.h"

using namespace ge;
using namespace ge::fusion;

namespace {
constexpr char kSrcOpType[] = "RotaryMulGrad";
constexpr char kDstOpType[] = "RotaryPositionEmbeddingGrad";
constexpr int64_t kExpectMode = 0L;

// rtGetSocSpec mock 控制接口（tests/ut/framework_normal/common/rt_soc_spec_mocker.cpp 提供）
extern "C" {
void SetRtSocSpecNpuArch(const char* arch);
void SetRtSocSpecFail(bool fail);
}

// socVersion -> NpuArch 映射，与 pass 侧 rtGetSocSpec("version", "NpuArch") 取到的数值字符串一致。
// 注意 Ascend960DT 复用 950 同族 regbase kernel 但上报独立 arch DAV_9201；其余未列举 SoC 走默认值。
void SetPlatform(const std::string& socVersion)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = socVersion;
    optionalInfo.soc_version = socVersion;
    fe::PlatformInfoManager::Instance().platform_info_map_[socVersion] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);

    SetRtSocSpecFail(false);
    if (socVersion == "Ascend950" || socVersion == "Ascend350") {
        SetRtSocSpecNpuArch("3510"); // DAV_3510
    } else if (socVersion == "Ascend960DT") {
        SetRtSocSpecNpuArch("9201"); // DAV_9201
    } else if (socVersion == "MC62CM12A") {
        SetRtSocSpecNpuArch("5102"); // DAV_5102
    } else {
        SetRtSocSpecNpuArch("2201"); // DAV_2201, non-regbase (e.g. Ascend910B)
    }
}

void SetInputDesc(const es::EsTensorHolder& input, DataType dtype, const Shape& shape)
{
    TensorDesc desc;
    EXPECT_EQ(input.GetProducer()->GetOutputDesc(0, desc), GRAPH_SUCCESS);
    desc.SetDataType(dtype);
    desc.SetShape(shape);
    desc.SetFormat(FORMAT_ND);
    EXPECT_EQ(input.GetProducer()->UpdateOutputDesc(0, desc), GRAPH_SUCCESS);
}

void ExpectInputNames(const GNode& node, const std::vector<std::string>& expectNames)
{
    EXPECT_EQ(node.GetInputsSize(), expectNames.size());
    for (size_t i = 0; i < expectNames.size(); ++i) {
        auto inNodeAndPort = node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        AscendString name;
        EXPECT_EQ(inNodeAndPort.first->GetName(name), GRAPH_SUCCESS);
        EXPECT_STREQ(name.GetString(), expectNames[i].c_str()) << "input index " << i;
    }
}

size_t CountNodes(const std::shared_ptr<Graph>& graph, const char* type)
{
    size_t count = 0;
    for (auto& node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == type) {
            ++count;
        }
    }
    return count;
}

bool FindNode(std::shared_ptr<Graph>& graph, const char* type, GNode& out)
{
    for (auto& node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == type) {
            out = node;
            return true;
        }
    }
    return false;
}

std::shared_ptr<Graph> BuildRotaryMulGradGraph(DataType dtype, bool needBackward)
{
    std::vector<int64_t> dimsX{2, 64, 8, 62};
    std::vector<int64_t> dimsR{1, 64, 1, 62};
    Shape shapeX(dimsX);
    Shape shapeR(dimsR);

    auto graphBuilder = es::EsGraphBuilder("rotary_mul_grad_ut");
    auto x = graphBuilder.CreateInput(0, "x", dtype, FORMAT_ND, shapeX.GetDims());
    auto r1 = graphBuilder.CreateInput(1, "r1", dtype, FORMAT_ND, shapeR.GetDims());
    auto r2 = graphBuilder.CreateInput(2, "r2", dtype, FORMAT_ND, shapeR.GetDims());
    auto dy = graphBuilder.CreateInput(3, "dy", dtype, FORMAT_ND, shapeX.GetDims());
    SetInputDesc(x, dtype, shapeX);
    SetInputDesc(r1, dtype, shapeR);
    SetInputDesc(r2, dtype, shapeR);
    SetInputDesc(dy, dtype, shapeX);

    Graph* graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode rotaryMulGradNode =
        es::CompliantNodeBuilder(graphPtr)
            .OpType(kSrcOpType)
            .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                          {"r1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                          {"r2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                          {"dy", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"dx", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                           {"dr1", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                           {"dr2", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .IrDefAttrs({{"need_backward", es::CompliantNodeBuilder::kEsAttrOptional, "Bool", es::CreateFrom(true)}})
            .Build();
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), 0, rotaryMulGradNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *r1.GetProducer(), 0, rotaryMulGradNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *r2.GetProducer(), 0, rotaryMulGradNode, 2);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *dy.GetProducer(), 0, rotaryMulGradNode, 3);
    if (!needBackward) {
        bool attrValue = false;
        EXPECT_EQ(rotaryMulGradNode.SetAttr("need_backward", attrValue), GRAPH_SUCCESS);
    }
    es::EsTensorHolder dx(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(rotaryMulGradNode, 0));
    es::EsTensorHolder dr1(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(rotaryMulGradNode, 1));
    es::EsTensorHolder dr2(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(rotaryMulGradNode, 2));
    return graphBuilder.BuildAndReset({dx, dr1, dr2});
}
} // namespace

class RotaryPositionEmbeddingGradFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        SetPlatform("Ascend950");
    }

    void SetUp() override
    {
        SetPlatform("Ascend950");
    }
};

TEST_F(RotaryPositionEmbeddingGradFusionPassTest, patternCreateSuccess)
{
    ops::RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(RotaryPositionEmbeddingGradFusionPassTest, rotaryMulGradFusionAscend910bNotChanged)
{
    SetPlatform("Ascend910B");
    std::shared_ptr<Graph> graph = BuildRotaryMulGradGraph(DT_FLOAT16, true);

    CustomPassContext passContext;
    ops::RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, kSrcOpType), 1);
    EXPECT_EQ(CountNodes(graph, kDstOpType), 0);
}

TEST_F(RotaryPositionEmbeddingGradFusionPassTest, rotaryMulGradNeedBackwardTrueFusionSuccess)
{
    SetPlatform("Ascend950");
    std::shared_ptr<Graph> graph = BuildRotaryMulGradGraph(DT_FLOAT, true);

    CustomPassContext passContext;
    ops::RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, kSrcOpType), 0);
    EXPECT_EQ(CountNodes(graph, kDstOpType), 1);

    GNode gradNode;
    ASSERT_TRUE(FindNode(graph, kDstOpType, gradNode));
    // The ES builder omits attrs equal to the IR default, so a missing mode attr means mode = 0.
    int64_t mode = 0;
    if (gradNode.GetAttr("mode", mode) != GRAPH_SUCCESS) {
        mode = 0;
    }
    EXPECT_EQ(mode, kExpectMode);

    // need_backward = true: optional x input must be wired; input remapping
    // dy/cos/sin/x come from source inputs dy/r1/r2/x.
    ExpectInputNames(gradNode, {"dy", "r1", "r2", "x"});
}

TEST_F(RotaryPositionEmbeddingGradFusionPassTest, rotaryMulGradNeedBackwardFalseFusionSuccess)
{
    SetPlatform("Ascend950");
    std::shared_ptr<Graph> graph = BuildRotaryMulGradGraph(DT_FLOAT, false);

    CustomPassContext passContext;
    ops::RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, kSrcOpType), 0);
    EXPECT_EQ(CountNodes(graph, kDstOpType), 1);

    GNode gradNode;
    ASSERT_TRUE(FindNode(graph, kDstOpType, gradNode));
    // The ES builder omits attrs equal to the IR default, so a missing mode attr means mode = 0.
    int64_t mode = 0;
    if (gradNode.GetAttr("mode", mode) != GRAPH_SUCCESS) {
        mode = 0;
    }
    EXPECT_EQ(mode, kExpectMode);

    // need_backward = false: optional x input must not be wired.
    ExpectInputNames(gradNode, {"dy", "r1", "r2"});
}

// Ascend350 is also NpuArch DAV_3510 (regbase): fusion must fire, same as the built-in pass.
TEST_F(RotaryPositionEmbeddingGradFusionPassTest, rotaryMulGradFusionAscend350Success)
{
    SetPlatform("Ascend350");
    std::shared_ptr<Graph> graph = BuildRotaryMulGradGraph(DT_FLOAT, true);

    CustomPassContext passContext;
    ops::RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, kSrcOpType), 0);
    EXPECT_EQ(CountNodes(graph, kDstOpType), 1);
}

// A future SoC sharing NpuArch DAV_3510 with Ascend950 (regbase) but not on the soc list:
// arch-based gating must still fire the fusion.
TEST_F(RotaryPositionEmbeddingGradFusionPassTest, rotaryMulGradFusionNpuArch3510SocNotOnListSuccess)
{
    SetPlatform("Ascend950X");   // hypothetical SoC, only fills the fe-side platform map
    SetRtSocSpecNpuArch("3510"); // DAV_3510
    std::shared_ptr<Graph> graph = BuildRotaryMulGradGraph(DT_FLOAT16, true);

    CustomPassContext passContext;
    ops::RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, kSrcOpType), 0);
    EXPECT_EQ(CountNodes(graph, kDstOpType), 1);
}

// Ascend960DT runs the same regbase kernels as Ascend950 but reports its own NpuArch DAV_9201:
// the arch-based gating must cover it.
TEST_F(RotaryPositionEmbeddingGradFusionPassTest, rotaryMulGradFusionAscend960dtSuccess)
{
    SetPlatform("Ascend960DT");
    std::shared_ptr<Graph> graph = BuildRotaryMulGradGraph(DT_FLOAT16, true);

    CustomPassContext passContext;
    ops::RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, kSrcOpType), 0);
    EXPECT_EQ(CountNodes(graph, kDstOpType), 1);
}

// rtGetSocSpec failure must disable the fusion instead of crashing.
TEST_F(RotaryPositionEmbeddingGradFusionPassTest, rotaryMulGradFusionNpuArchQueryFailNotChanged)
{
    SetRtSocSpecFail(true);
    std::shared_ptr<Graph> graph = BuildRotaryMulGradGraph(DT_FLOAT16, true);

    CustomPassContext passContext;
    ops::RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, kSrcOpType), 1);
    EXPECT_EQ(CountNodes(graph, kDstOpType), 0);
}
