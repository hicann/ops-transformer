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
 * \file interleave_rope_fusion_pass.cpp
 * \brief InterleaveRope --> RotaryPositionEmbedding(mode = 3), migrated from canndev built-in rope_fusion_pass.
 */

#include "interleave_rope_fusion_pass.h"

#if CANN_VERSION_NUM >= INTERLEAVE_ROPE_GRAPH_FUSION_SUPPORT_VERSION
#include <string>
#include <vector>

#include "es_InterleaveRope.h"          // es autogen header
#include "es_RotaryPositionEmbedding.h" // es autogen header
#include "ge/es_graph_builder.h"
#include "ge/ge_utils.h"
#include "log/log.h"
#include "op_host/util/op_const_def.h"
#include "runtime/rt_external_base.h"

namespace ops {
namespace {
const std::string kPassName = "InterleaveRope2RotaryPositionEmbeddingFusionPass";
constexpr int64_t kCaptureTensorIdx = 0L;
constexpr int64_t kInterleaveRopeMode = 3L;
constexpr size_t kInputNum = 3;
constexpr size_t kXIdx = 0;
constexpr size_t kCosIdx = 1;
constexpr size_t kSinIdx = 2;

// Same gating as the built-in pass: fuse only on regbase platforms (NpuArch DAV_3510/DAV_9201/DAV_5102).
// Gating on NpuArch instead of a short_soc_version string list: one arch covers multiple SoCs
// (DAV_3510: Ascend950/Ascend350, DAV_9201: Ascend960DT, DAV_5102: MC62CM12A), so SoCs sharing the regbase
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
           npuArch == std::to_string(static_cast<uint32_t>(Ops::Base::DAV_9201)) ||
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

ge::Status InferShapeReplaceGraph(const ge::fusion::GraphUniqPtr& replaceGraph,
                                  const std::vector<ge::fusion::SubgraphInput>& subgraphInputs)
{
    std::vector<ge::Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNodes = subgraphInput.GetAllInputs();
        if (matchNodes.empty()) {
            continue;
        }
        auto matchNode = matchNodes.at(0);
        ge::TensorDesc tensorDesc;
        if (matchNode.node.GetInputDesc(matchNode.index, tensorDesc) != ge::GRAPH_SUCCESS) {
            OP_LOGD(kPassName, "InferShapeReplaceGraph: get input desc failed.");
            continue;
        }
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return ge::GeUtils::InferShape(*replaceGraph, inputShapes);
}
} // namespace

std::vector<ge::fusion::PatternUniqPtr> InterleaveRope2RotaryPositionEmbeddingFusionPass::Patterns()
{
    OP_LOGD(kPassName, "Enter Patterns for InterleaveRope2RotaryPositionEmbeddingFusionPass.");
    std::vector<ge::fusion::PatternUniqPtr> patterns;

    auto graphBuilder = ge::es::EsGraphBuilder(kPassName.c_str());
    auto inputs = graphBuilder.CreateInputs<kInputNum>();
    auto y = ge::es::InterleaveRope(inputs[kXIdx], inputs[kCosIdx], inputs[kSinIdx]);

    auto graph = graphBuilder.BuildAndReset({y});
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*y.GetProducer(), kCaptureTensorIdx});
    patterns.emplace_back(std::move(pattern));
    return patterns;
}

bool InterleaveRope2RotaryPositionEmbeddingFusionPass::MeetRequirements(
    const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OP_LOGD(kPassName, "Enter MeetRequirements for InterleaveRope2RotaryPositionEmbeddingFusionPass.");
    (void)matchResult;
    if (!IsRegBasePlatform()) {
        OP_LOGD(kPassName, "InterleaveRope can not be fused to RotaryPositionEmbedding on current platform, skip.");
        return false;
    }
    return true;
}

ge::fusion::GraphUniqPtr InterleaveRope2RotaryPositionEmbeddingFusionPass::Replacement(
    const std::unique_ptr<ge::fusion::MatchResult>& matchResult)
{
    OP_LOGD(kPassName, "Enter Replacement for InterleaveRope2RotaryPositionEmbeddingFusionPass.");
    std::vector<ge::fusion::SubgraphInput> subgraphInputs;
    if (matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs) != ge::SUCCESS) {
        OP_LOGE(kPassName, "Get subgraph inputs failed in Replacement.");
        return nullptr;
    }
    if (subgraphInputs.size() != kInputNum) {
        OP_LOGE(kPassName, "Subgraph input num %zu is not expected %zu.", subgraphInputs.size(), kInputNum);
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
    auto rX = replaceGraphBuilder.CreateInput(kXIdx, "x", inputDtypes[kXIdx], inputFormats[kXIdx],
                                              inputShapes[kXIdx].GetDims());
    auto rCos = replaceGraphBuilder.CreateInput(kCosIdx, "cos", inputDtypes[kCosIdx], inputFormats[kCosIdx],
                                                inputShapes[kCosIdx].GetDims());
    auto rSin = replaceGraphBuilder.CreateInput(kSinIdx, "sin", inputDtypes[kSinIdx], inputFormats[kSinIdx],
                                                inputShapes[kSinIdx].GetDims());

    auto y = ge::es::RotaryPositionEmbedding(rX, rCos, rSin, nullptr, kInterleaveRopeMode);
    ge::fusion::GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({y});
    if (InferShapeReplaceGraph(replaceGraph, subgraphInputs) != ge::SUCCESS) {
        OP_LOGE(kPassName, "InferShape for replacement graph failed.");
        return nullptr;
    }
    return replaceGraph;
}

REG_FUSION_PASS(InterleaveRope2RotaryPositionEmbeddingFusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);
} // namespace ops
#endif // CANN_VERSION_NUM >= INTERLEAVE_ROPE_GRAPH_FUSION_SUPPORT_VERSION
