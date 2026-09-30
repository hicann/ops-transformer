/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rotary_position_embedding_grad_fusion_pass.cpp
 * \brief RotaryMulGrad --> RotaryPositionEmbeddingGrad(mode = 0), migrated from canndev built-in rope_fusion_pass.
 */

#include "rotary_position_embedding_grad_fusion_pass.h"

#if CANN_VERSION_NUM >= ROTARY_POSITION_EMBEDDING_GRAD_GRAPH_FUSION_SUPPORT_VERSION
#include <string>
#include <vector>

#include "es_RotaryPositionEmbeddingGrad.h" // es autogen header
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/ge_utils.h"
#include "log/log.h"
#include "op_host/util/op_const_def.h"
#include "runtime/rt_external_base.h"

namespace ops {
namespace {
const std::string kPassName = "RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass";
const std::string kSrcOpType = "RotaryMulGrad";
const std::string kNeedBackwardAttr = "need_backward";
constexpr int64_t kCaptureTensorIdx = 0L;
constexpr int64_t kRotaryMulGradMode = 0L;
constexpr size_t kInputNum = 4;
constexpr size_t kXIdx = 0;
constexpr size_t kCosIdx = 1;
constexpr size_t kSinIdx = 2;
constexpr size_t kDyIdx = 3;

// Same gating as the built-in pass: fuse only on regbase platforms (NpuArch DAV_3510/DAV_5102).
// Gating on NpuArch instead of a short_soc_version string list: one arch covers multiple SoCs
// (DAV_3510: Ascend950/Ascend350/Ascend960DT..., DAV_5102: MC62CM12A), so SoCs sharing the regbase
// arch are covered without maintaining the soc list.
bool IsRegBasePlatform()
{
    char npuArchVal[16] = {0};
    if (rtGetSocSpec("version", "NpuArch", npuArchVal, sizeof(npuArchVal)) != RT_ERROR_NONE) {
        OP_LOGW(kPassName, "Get NpuArch from soc spec failed.");
        return false;
    }
    OP_LOGD(kPassName, "Platform NpuArch: %s.", npuArchVal);
    const std::string npuArch(npuArchVal);
    return npuArch == std::to_string(static_cast<uint32_t>(Ops::Base::DAV_3510)) ||
           npuArch == std::to_string(static_cast<uint32_t>(Ops::Base::DAV_5102));
}

void GetInputsInfo(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs, std::vector<ge::Shape>& inputShapes,
                   std::vector<ge::DataType>& inputDtypes, std::vector<ge::Format>& inputFormats)
{
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNodes = subgraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            OP_LOGD(kPassName, "GetInputsInfo: match nodes is empty.");
            continue;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tensorDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tensorDesc) != ge::GRAPH_SUCCESS) {
            OP_LOGD(kPassName, "GetInputsInfo: get input desc failed.");
            continue;
        }
        inputShapes.emplace_back(tensorDesc.GetShape());
        inputDtypes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

bool GetNeedBackward(const std::unique_ptr<ge::fusion::MatchResult>& matchResult, bool& needBackward)
{
    ge::fusion::NodeIo nodeIo;
    if (matchResult->GetCapturedTensor(kCaptureTensorIdx, nodeIo) != ge::SUCCESS) {
        OP_LOGE(kPassName, "Get captured RotaryMulGrad node failed.");
        return false;
    }
    // Attr need_backward is optional with default true in the RotaryMulGrad IR.
    needBackward = true;
    if (nodeIo.node.GetAttr(kNeedBackwardAttr.c_str(), needBackward) != ge::GRAPH_SUCCESS) {
        OP_LOGD(kPassName, "Attr need_backward is not set, use default true.");
        needBackward = true;
    }
    return true;
}
} // namespace

std::vector<ge::fusion::PatternUniqPtr> RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass::Patterns()
{
    OP_LOGD(kPassName, "Enter Patterns for RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass.");
    std::vector<ge::fusion::PatternUniqPtr> patterns;

    auto graphBuilder = ge::es::EsGraphBuilder(kPassName.c_str());
    auto inputs = graphBuilder.CreateInputs<kInputNum>();

    // RotaryMulGrad has no ES API, build the pattern node with CompliantNodeBuilder.
    ge::Graph* graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();
    ge::GNode rotaryMulGradNode = ge::es::CompliantNodeBuilder(graphPtr)
                                      .OpType(kSrcOpType.c_str())
                                      .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                                    {"r1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                                    {"r2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                                    {"dy", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                      .IrDefOutputs({{"dx", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                                     {"dr1", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                                     {"dr2", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                      .IrDefAttrs({{kNeedBackwardAttr, ge::es::CompliantNodeBuilder::kEsAttrOptional,
                                                    "Bool", ge::es::CreateFrom(true)}})
                                      .Build();
    for (size_t i = 0; i < kInputNum; ++i) {
        ge::GNode inputNode = *inputs[i].GetProducer();
        ge::es::AddEdgeAndUpdatePeerDesc(*graphPtr, inputNode, 0, rotaryMulGradNode, static_cast<int32_t>(i));
    }
    ge::es::EsTensorHolder dx(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(rotaryMulGradNode, 0));
    ge::es::EsTensorHolder dr1(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(rotaryMulGradNode, 1));
    ge::es::EsTensorHolder dr2(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(rotaryMulGradNode, 2));

    auto graph = graphBuilder.BuildAndReset({dx, dr1, dr2});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*dx.GetProducer(), kCaptureTensorIdx});
    patterns.emplace_back(std::move(pattern));
    return patterns;
}

bool RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass::MeetRequirements(
    const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OP_LOGD(kPassName, "Enter MeetRequirements for RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass.");
    (void)matchResult;
    if (!IsRegBasePlatform()) {
        OP_LOGD(kPassName, "RotaryMulGrad can not be fused to RotaryPositionEmbeddingGrad on current platform, skip.");
        return false;
    }
    return true;
}

ge::fusion::GraphUniqPtr RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OP_LOGD(kPassName, "Enter Replacement for RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass.");
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OP_LOGE(kPassName, "Get subgraph inputs failed in Replacement.");
        return nullptr;
    }
    if (subgraphInputs.size() != kInputNum) {
        OP_LOGE(kPassName, "Subgraph input num %zu is not expected %zu.", subgraphInputs.size(), kInputNum);
        return nullptr;
    }

    bool needBackward = true;
    if (!GetNeedBackward(matchResult, needBackward)) {
        return nullptr;
    }

    std::vector<ge::Shape> inputShapes;
    std::vector<ge::DataType> inputDtypes;
    std::vector<ge::Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);
    if (inputShapes.size() != kInputNum) {
        OP_LOGE(kPassName, "Get inputs info failed in Replacement.");
        return nullptr;
    }

    auto replaceGraphBuilder = ge::es::EsGraphBuilder("replacement");
    // Graph inputs must be created contiguously from index 0 in boundary order (x, r1, r2, dy).
    // RotaryPositionEmbeddingGrad: dy -> input 0, cos -> input 1, sin -> input 2, x(optional) -> input 3.
    // The x input node is always created, but only wired into the fused node when need_backward is true.
    auto rX = replaceGraphBuilder.CreateInput(kXIdx, "x", inputDtypes[kXIdx], inputFormats[kXIdx],
                                              inputShapes[kXIdx].GetDims());
    auto rCos = replaceGraphBuilder.CreateInput(kCosIdx, "cos", inputDtypes[kCosIdx], inputFormats[kCosIdx],
                                                inputShapes[kCosIdx].GetDims());
    auto rSin = replaceGraphBuilder.CreateInput(kSinIdx, "sin", inputDtypes[kSinIdx], inputFormats[kSinIdx],
                                                inputShapes[kSinIdx].GetDims());
    auto rDy = replaceGraphBuilder.CreateInput(kDyIdx, "dy", inputDtypes[kDyIdx], inputFormats[kDyIdx],
                                               inputShapes[kDyIdx].GetDims());

    ge::es::EsTensorHolder rXUsed = needBackward ? rX : ge::es::EsTensorHolder(nullptr);
    auto out = ge::es::RotaryPositionEmbeddingGrad(rDy, rCos, rSin, rXUsed, kRotaryMulGradMode);
    ge::fusion::GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({out.dx, out.dcos, out.dsin});
    if (replaceGraph == nullptr) {
        OP_LOGE(kPassName, "Build replacement graph failed.");
        return nullptr;
    }
    std::vector<ge::Shape> inferShapes = {inputShapes[kXIdx], inputShapes[kCosIdx], inputShapes[kSinIdx],
                                          inputShapes[kDyIdx]};
    if (ge::GeUtils::InferShape(*replaceGraph, inferShapes) != ge::SUCCESS) {
        OP_LOGE(kPassName, "InferShape for replacement graph failed.");
        return nullptr;
    }
    return replaceGraph;
}

REG_FUSION_PASS(RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);
} // namespace ops
#endif // CANN_VERSION_NUM >= ROTARY_POSITION_EMBEDDING_GRAD_GRAPH_FUSION_SUPPORT_VERSION
