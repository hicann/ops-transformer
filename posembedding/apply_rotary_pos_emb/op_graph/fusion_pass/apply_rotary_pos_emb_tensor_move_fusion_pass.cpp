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
 * \file apply_rotary_pos_emb_tensor_move_fusion_pass.cpp
 * \brief Remove redundant TensorMove nodes on the query/key inputs of ApplyRotaryPosEmb,
 * migrated from canndev built-in apply_rotary_pos_emb_tensormove_pass.
 */

#include "apply_rotary_pos_emb_tensor_move_fusion_pass.h"

#if CANN_VERSION_NUM >= APPLY_ROTARY_POS_EMB_GRAPH_FUSION_SUPPORT_VERSION
#include <string>
#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h" // for the REG_FUSION_PASS macro
#include "log/log.h"

namespace ops {
namespace {
const std::string kPassName = "ApplyRotaryPosEmbTensorMoveFusionPass";
const char* kArpeOpType = "ApplyRotaryPosEmb";
const char* kTensorMoveOpType = "TensorMove";
constexpr int32_t kQueryIdx = 0;
constexpr int32_t kKeyIdx = 1;
constexpr int32_t kTensorMoveIoIdx = 0;
constexpr size_t kSingleConsumer = 1U;

/* Remove the TensorMove node if its input data is only used by the TensorMove node:
       q           k       cos  sin           q     k    cos  sin
       |           |        /   /              \    |    /   /
   TensorMove  TensorMove  /   /      ->       ApplyRotaryPosEmb
            \      |      /   /                        |
             \     |     /   /                      output
             ApplyRotaryPosEmb
                    |
                 output
*/

bool IsType(const ge::GNode& node, const char* type)
{
    ge::AscendString nodeType;
    return node.GetType(nodeType) == ge::GRAPH_SUCCESS && nodeType == type;
}

bool IsType(const ge::GNodePtr& node, const char* type)
{
    if (node == nullptr) {
        return false;
    }
    ge::AscendString nodeType;
    return node->GetType(nodeType) == ge::GRAPH_SUCCESS && nodeType == type;
}

bool IsSameNode(const ge::GNode& lhs, const ge::GNode& rhs)
{
    ge::AscendString lhsName;
    ge::AscendString rhsName;
    return lhs.GetName(lhsName) == ge::GRAPH_SUCCESS && rhs.GetName(rhsName) == ge::GRAPH_SUCCESS &&
           lhsName.GetString() != nullptr && rhsName.GetString() != nullptr &&
           std::string(lhsName.GetString()) == std::string(rhsName.GetString());
}

bool AddControlEdgeIfAbsent(const ge::GraphPtr& graph, ge::GNode& srcNode, ge::GNode& dstNode)
{
    if (IsSameNode(srcNode, dstNode)) {
        OP_LOGD(kPassName, "Skip control edge with the same source and destination.");
        return true;
    }
    for (const auto& outCtrlNode : srcNode.GetOutControlNodes()) {
        if (outCtrlNode != nullptr && IsSameNode(*outCtrlNode, dstNode)) {
            return true;
        }
    }
    if (graph->AddControlEdge(srcNode, dstNode) != ge::GRAPH_SUCCESS) {
        OP_LOGE(kPassName, "Add control edge failed when transferring TensorMove control edges.");
        return false;
    }
    return true;
}

// Transfer the TensorMove in/out control edges to the ApplyRotaryPosEmb node, same as
// FusionTurbo::TransferInCtrlEdges/TransferOutCtrlEdges in the built-in pass. The original
// control edges are unlinked together with the TensorMove node when it is removed.
bool TransferControlEdges(const ge::GraphPtr& graph, const ge::GNodePtr& tmNode, ge::GNode& arpeNode)
{
    for (const auto& inCtrlNode : tmNode->GetInControlNodes()) {
        if (inCtrlNode == nullptr) {
            continue;
        }
        if (!AddControlEdgeIfAbsent(graph, *inCtrlNode, arpeNode)) {
            OP_LOGE(kPassName, "Link ApplyRotaryPosEmb node input ctrl edges failed.");
            return false;
        }
    }
    for (const auto& outCtrlNode : tmNode->GetOutControlNodes()) {
        if (outCtrlNode == nullptr) {
            continue;
        }
        if (!AddControlEdgeIfAbsent(graph, arpeNode, *outCtrlNode)) {
            OP_LOGE(kPassName, "Link ApplyRotaryPosEmb node output ctrl edges failed.");
            return false;
        }
    }
    return true;
}

// Bypass the TensorMove node: connect its producer to the ApplyRotaryPosEmb input directly.
// TensorMove is an identity copy, so the ApplyRotaryPosEmb input desc is unchanged and
// does not need to be refreshed, same as the built-in pass.
ge::graphStatus RemoveTensorMoveNode(const ge::GraphPtr& graph, ge::GNode& arpeNode, int32_t inputIdx,
                                     const ge::GNodePtr& tmNode, const ge::GNodePtr& srcNode, int32_t srcOutPort)
{
    if (graph->RemoveEdge(*tmNode, kTensorMoveIoIdx, arpeNode, inputIdx) != ge::GRAPH_SUCCESS) {
        OP_LOGE(kPassName, "Remove TensorMove out data edge failed.");
        return ge::GRAPH_FAILED;
    }
    if (graph->AddDataEdge(*srcNode, srcOutPort, arpeNode, inputIdx) != ge::GRAPH_SUCCESS) {
        OP_LOGE(kPassName, "Add data edge to ApplyRotaryPosEmb input %d failed.", inputIdx);
        return ge::GRAPH_FAILED;
    }
    if (!TransferControlEdges(graph, tmNode, arpeNode)) {
        return ge::GRAPH_FAILED;
    }
    if (graph->RemoveNode(*tmNode) != ge::GRAPH_SUCCESS) {
        OP_LOGE(kPassName, "Remove TensorMove node failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::SUCCESS;
}

ge::graphStatus FusionProcess(const ge::GraphPtr& graph, ge::GNode& arpeNode, int32_t inputIdx)
{
    const auto inNodeAndPort = arpeNode.GetInDataNodesAndPortIndexs(inputIdx);
    const ge::GNodePtr& tmNode = inNodeAndPort.first;
    if (tmNode == nullptr) {
        OP_LOGD(kPassName, "Get peer node for ApplyRotaryPosEmb input %d failed.", inputIdx);
        return ge::GRAPH_NOT_CHANGED;
    }
    if (!IsType(tmNode, kTensorMoveOpType)) {
        OP_LOGD(kPassName, "Node before ApplyRotaryPosEmb(input %d) is not %s.", inputIdx, kTensorMoveOpType);
        return ge::GRAPH_NOT_CHANGED;
    }

    // The TensorMove output must only be used by the ApplyRotaryPosEmb node.
    if (tmNode->GetOutDataNodesAndPortIndexs(kTensorMoveIoIdx).size() != kSingleConsumer) {
        OP_LOGD(kPassName, "ApplyRotaryPosEmb(input %d) data used for %zu nodes.", inputIdx,
                tmNode->GetOutDataNodesAndPortIndexs(kTensorMoveIoIdx).size());
        return ge::GRAPH_NOT_CHANGED;
    }

    // The output of the TensorMove producer must only be used by the TensorMove node.
    const auto srcNodeAndPort = tmNode->GetInDataNodesAndPortIndexs(kTensorMoveIoIdx);
    const ge::GNodePtr& srcNode = srcNodeAndPort.first;
    const int32_t srcOutPort = srcNodeAndPort.second;
    if (srcNode == nullptr) {
        OP_LOGD(kPassName, "Get producer node of TensorMove before ApplyRotaryPosEmb(input %d) failed.", inputIdx);
        return ge::GRAPH_NOT_CHANGED;
    }
    if (srcNode->GetOutDataNodesAndPortIndexs(srcOutPort).size() != kSingleConsumer) {
        OP_LOGD(kPassName, "Input(%d) data for TensorMove used for %zu nodes.", inputIdx,
                srcNode->GetOutDataNodesAndPortIndexs(srcOutPort).size());
        return ge::GRAPH_NOT_CHANGED;
    }

    return RemoveTensorMoveNode(graph, arpeNode, inputIdx, tmNode, srcNode, srcOutPort);
}

ge::graphStatus FusionArpeNode(const ge::GraphPtr& graph, ge::GNode& arpeNode)
{
    const auto queryResult = FusionProcess(graph, arpeNode, kQueryIdx);
    if (queryResult == ge::GRAPH_FAILED) {
        OP_LOGE(kPassName, "ApplyRotaryPosEmbTensorMoveFusionPass failed due to query input fail.");
        return ge::GRAPH_FAILED;
    }
    const auto keyResult = FusionProcess(graph, arpeNode, kKeyIdx);
    if (keyResult == ge::GRAPH_FAILED) {
        OP_LOGE(kPassName, "ApplyRotaryPosEmbTensorMoveFusionPass failed due to key input fail.");
        return ge::GRAPH_FAILED;
    }
    if (queryResult == ge::GRAPH_NOT_CHANGED && keyResult == ge::GRAPH_NOT_CHANGED) {
        return ge::GRAPH_NOT_CHANGED;
    }
    return ge::SUCCESS;
}
} // namespace

ge::Status ApplyRotaryPosEmbTensorMoveFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    (void)passContext;
    OP_LOGD(kPassName, "Enter ApplyRotaryPosEmbTensorMoveFusionPass.");
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(kPassName, "Graph is null or invalid.");
        return ge::GRAPH_NOT_CHANGED;
    }

    std::vector<ge::GNode> arpeNodes;
    for (auto& node : graph->GetDirectNode()) {
        if (IsType(node, kArpeOpType)) {
            arpeNodes.emplace_back(node);
        }
    }
    if (arpeNodes.empty()) {
        return ge::GRAPH_NOT_CHANGED;
    }

    bool changed = false;
    for (auto& arpeNode : arpeNodes) {
        const auto status = FusionArpeNode(graph, arpeNode);
        if (status == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
        changed = changed || (status == ge::SUCCESS);
    }
    if (!changed) {
        OP_LOGD(kPassName, "ApplyRotaryPosEmbTensorMoveFusionPass is not match.");
        return ge::GRAPH_NOT_CHANGED;
    }
    OP_LOGI(kPassName, "ApplyRotaryPosEmbTensorMoveFusionPass success end.");
    return ge::SUCCESS;
}

REG_FUSION_PASS(ApplyRotaryPosEmbTensorMoveFusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);
} // namespace ops
#endif // CANN_VERSION_NUM >= APPLY_ROTARY_POS_EMB_GRAPH_FUSION_SUPPORT_VERSION
