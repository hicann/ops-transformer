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

#include "../../../op_graph/fusion_pass/interleave_rope_fusion_pass.h"

using namespace ge;
using namespace ge::fusion;

namespace {
constexpr char kSrcOpType[] = "InterleaveRope";
constexpr char kDstOpType[] = "RotaryPositionEmbedding";
constexpr int64_t kExpectMode = 3L;

// rtGetSocSpec mock 控制接口（tests/ut/framework_normal/common/rt_soc_spec_mocker.cpp 提供）
extern "C" {
void SetRtSocSpecNpuArch(const char* arch);
void SetRtSocSpecFail(bool fail);
}

// socVersion -> NpuArch 映射，与 pass 侧 rtGetSocSpec("version", "NpuArch") 取到的数值字符串一致。
// 注意 Ascend960DT 等同 arch 新 SoC 复用 "3510"，不在此表 —— 用 SetRtSocSpecNpuArch 直接指定。
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

std::shared_ptr<Graph> BuildInterleaveRopeGraph(DataType dtype)
{
    std::vector<int64_t> dimsX{1, 64, 2, 22};
    std::vector<int64_t> dimsR{1, 64, 1, 22};
    Shape shapeX(dimsX);
    Shape shapeR(dimsR);

    auto graphBuilder = es::EsGraphBuilder("interleave_rope_ut");
    auto x = graphBuilder.CreateInput(0, "x", dtype, FORMAT_ND, shapeX.GetDims());
    auto cos = graphBuilder.CreateInput(1, "cos", dtype, FORMAT_ND, shapeR.GetDims());
    auto sin = graphBuilder.CreateInput(2, "sin", dtype, FORMAT_ND, shapeR.GetDims());
    SetInputDesc(x, dtype, shapeX);
    SetInputDesc(cos, dtype, shapeR);
    SetInputDesc(sin, dtype, shapeR);

    Graph* graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();
    GNode interleaveRopeNode = es::CompliantNodeBuilder(graphPtr)
                                   .OpType(kSrcOpType)
                                   .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                                 {"cos", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                                 {"sin", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                   .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                   .Build();
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *x.GetProducer(), 0, interleaveRopeNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *cos.GetProducer(), 0, interleaveRopeNode, 1);
    es::AddEdgeAndUpdatePeerDesc(*graphPtr, *sin.GetProducer(), 0, interleaveRopeNode, 2);
    es::EsTensorHolder y(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(interleaveRopeNode, 0));
    return graphBuilder.BuildAndReset({y});
}
} // namespace

class InterleaveRopeFusionPassTest : public testing::Test {
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

TEST_F(InterleaveRopeFusionPassTest, patternCreateSuccess)
{
    ops::InterleaveRope2RotaryPositionEmbeddingFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(InterleaveRopeFusionPassTest, interleaveRopeFusionAscend910bNotChanged)
{
    SetPlatform("Ascend910B");
    std::shared_ptr<Graph> graph = BuildInterleaveRopeGraph(DT_FLOAT16);

    CustomPassContext passContext;
    ops::InterleaveRope2RotaryPositionEmbeddingFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, kSrcOpType), 1);
    EXPECT_EQ(CountNodes(graph, kDstOpType), 0);
}

TEST_F(InterleaveRopeFusionPassTest, interleaveRopeFusionAscend950Success)
{
    SetPlatform("Ascend950");
    std::shared_ptr<Graph> graph = BuildInterleaveRopeGraph(DT_FLOAT16);

    CustomPassContext passContext;
    ops::InterleaveRope2RotaryPositionEmbeddingFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, kSrcOpType), 0);
    EXPECT_EQ(CountNodes(graph, kDstOpType), 1);

    GNode rpeNode;
    ASSERT_TRUE(FindNode(graph, kDstOpType, rpeNode));
    // The ES builder omits attrs equal to the IR default, so a missing mode attr means mode = 0.
    int64_t mode = 0;
    if (rpeNode.GetAttr("mode", mode) != GRAPH_SUCCESS) {
        mode = 0;
    }
    EXPECT_EQ(mode, kExpectMode);

    // Input wiring: x/cos/sin trace back to the source graph inputs.
    ExpectInputNames(rpeNode, {"x", "cos", "sin"});
}

// Ascend350 is also NpuArch DAV_3510 (regbase): fusion must fire, same as the built-in pass.
TEST_F(InterleaveRopeFusionPassTest, interleaveRopeFusionAscend350Success)
{
    SetPlatform("Ascend350");
    std::shared_ptr<Graph> graph = BuildInterleaveRopeGraph(DT_FLOAT16);

    CustomPassContext passContext;
    ops::InterleaveRope2RotaryPositionEmbeddingFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, kSrcOpType), 0);
    EXPECT_EQ(CountNodes(graph, kDstOpType), 1);
}

// Ascend960DT shares NpuArch DAV_3510 with Ascend950 (regbase) but is not on the soc list:
// arch-based gating must still fire the fusion.
TEST_F(InterleaveRopeFusionPassTest, interleaveRopeFusionNpuArch3510SocNotOnListSuccess)
{
    SetPlatform("Ascend960DT");  // only fills the fe-side platform map
    SetRtSocSpecNpuArch("3510"); // DAV_3510
    std::shared_ptr<Graph> graph = BuildInterleaveRopeGraph(DT_FLOAT16);

    CustomPassContext passContext;
    ops::InterleaveRope2RotaryPositionEmbeddingFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodes(graph, kSrcOpType), 0);
    EXPECT_EQ(CountNodes(graph, kDstOpType), 1);
}

// rtGetSocSpec failure must disable the fusion instead of crashing.
TEST_F(InterleaveRopeFusionPassTest, interleaveRopeFusionNpuArchQueryFailNotChanged)
{
    SetRtSocSpecFail(true);
    std::shared_ptr<Graph> graph = BuildInterleaveRopeGraph(DT_FLOAT16);

    CustomPassContext passContext;
    ops::InterleaveRope2RotaryPositionEmbeddingFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodes(graph, kSrcOpType), 1);
    EXPECT_EQ(CountNodes(graph, kDstOpType), 0);
}
