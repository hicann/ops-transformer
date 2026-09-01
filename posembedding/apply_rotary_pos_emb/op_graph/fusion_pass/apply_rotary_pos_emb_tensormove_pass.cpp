/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_rotary_pos_emb_tensormove_pass.cc
 * \brief
 */
#include "apply_rotary_pos_emb_tensormove_pass.h"
#include <vector>
#include <string>

#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "graph_optimizer/fusion_common/fusion_turbo.h"
#include "pattern_fusion_util.h"
#include "platform/platform_info.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE_ARPE = "ApplyRotaryPosEmb";
static const std::string PATTERN_ARPE = "ApplyRotaryPosEmb";
static const char* FUSED_NODE_TENSORMOVE = "TensorMove";

/* Remove tensormove node if input data only used for tensormove node!
       q           k       cos  sin           q     k    cos  sin
       |           |        /   /              \    |    /   /
   tensormove  tensormove  /   /      ->       ApplyRotaryPosEmb
            \      |      /   /                        |
             \     |     /   /                      output
             ApplyRotaryPosEmb
                    |
                 output
*/

vector<FusionPattern*> ArpeTmFusionPass::DefinePatterns() {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define ApplyRotaryPosEmbTensorMoveFusionPass pattern begin");
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("ArpeTmFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_ARPE, {FUSED_NODE_ARPE}).SetOutput(PATTERN_ARPE);
  patterns.push_back(pattern);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "Define ApplyRotaryPosEmbTensorMoveFusionPass pattern end");
  return patterns;
}

Status ArpeTmFusionPass::RemoveTensorMoveNode(ge::NodePtr arpeNode,
                                              ge::NodePtr fusedNode, ge::ComputeGraph& graph) {
  // connect data anchor and bypass fusedNode
  auto preOutDataAnchor = fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor();
  FUSION_PASS_CHECK(fusedNode->GetOutDataAnchor(0) == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode->GetOutDataAnchor(0) is null."),
                    return FAILED);
  for (auto inDataAnchor : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."),
                      return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(preOutDataAnchor, inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Add out data edge failed."),
                      return FAILED);
  }
  // connect control anchor and bypass fusedNode
  std::vector<ge::NodePtr> oriFusedNodes = {fusedNode};
  FUSION_PASS_CHECK(FusionTurbo::TransferInCtrlEdges(oriFusedNodes, arpeNode) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "link ApplyRotaryPosEmb node input ctrl edges failed"),
                    return FAILED);
  FUSION_PASS_CHECK(FusionTurbo::TransferOutCtrlEdges(oriFusedNodes, arpeNode) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE, "link ApplyRotaryPosEmb node output ctrl edges failed"),
                    return FAILED);
  // remove fusedNode node
  FUSION_PASS_CHECK(graph.RemoveNode(fusedNode) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove node failed"), return FAILED);
  return SUCCESS;
}

Status ArpeTmFusionPass::FusionProcess(ge::NodePtr arpeNode, int idx, ge::ComputeGraph& graph) {
  FUSION_PASS_CHECK(arpeNode->GetInDataAnchor(idx) == nullptr,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "arpeNode->GetInDataAnchor(%d) is null.", idx),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(arpeNode->GetInDataAnchor(idx)->GetPeerOutAnchor() == nullptr,
                    OP_LOGD(FUSED_OP_TYPE.c_str(),"arpeNode->GetInDataAnchor(%d)->GetPeerOutAnchor() is null.", idx),
                    return NOT_CHANGED);

  ge::NodePtr tmNode = arpeNode->GetInDataAnchor(idx)->GetPeerOutAnchor()->GetOwnerNode();
  FUSION_PASS_CHECK(tmNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                                   "Get peer node for ApplyRotaryPosEmb input %d failed.", idx),
                    return FAILED);
  FUSION_PASS_CHECK(tmNode->GetType() != FUSED_NODE_TENSORMOVE,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Node before ApplyRotaryPosEmb(input %d) is not %s.",
                            idx, FUSED_NODE_TENSORMOVE),
                    return NOT_CHANGED);

  size_t tmOutPeerNodeNum = arpeNode->GetInDataAnchor(idx)->GetPeerOutAnchor()->GetPeerAnchorsSize();
  FUSION_PASS_CHECK(
      1 != tmOutPeerNodeNum,
      OP_LOGD(FUSED_OP_TYPE.c_str(), "ApplyRotaryPosEmb(input %d) data used for %zu nodes.", idx, tmOutPeerNodeNum),
      return NOT_CHANGED);

  /* The output from topNode must only used in one node */
  FUSION_PASS_CHECK(tmNode->GetInDataAnchor(0) == nullptr,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "tmNode->GetInDataAnchor(0) is null."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(tmNode->GetInDataAnchor(0)->GetPeerOutAnchor() == nullptr,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "tmNode->GetInDataAnchor(0)->GetPeerOutAnchor() is null."),
                    return NOT_CHANGED);
  size_t topOutAnchorNum = tmNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetPeerAnchorsSize();
  FUSION_PASS_CHECK(topOutAnchorNum != 1,
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "Input(%d) data for TensorMove used for %zu nodes.",
                            idx, topOutAnchorNum),
                    return NOT_CHANGED);

  return RemoveTensorMoveNode(arpeNode, tmNode, graph);
}

Status ArpeTmFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter ArpeTmFusionPass!");
  Status result_input0 = SUCCESS;
  Status result_input1 = SUCCESS;
  ge::NodePtr arpeNode = GetNodeFromMapping(PATTERN_ARPE, mapping);

  FUSION_PASS_CHECK(arpeNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ApplyRotaryPosEmb Node is null."),
                    return PARAM_INVALID);

  result_input0 = FusionProcess(arpeNode, 0, graph);
  FUSION_PASS_CHECK(
      result_input0 == FAILED,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ArpeTmFusionPass failed due to input0 fail."),
      return FAILED);
  result_input1 = FusionProcess(arpeNode, 1, graph);
  FUSION_PASS_CHECK(
      result_input1 == FAILED,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ArpeTmFusionPass failed due to input1 fail."),
      return FAILED);
  if ((result_input0 == NOT_CHANGED) && (result_input1 == NOT_CHANGED)) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "ArpeTmFusionPass does not match.");
    return NOT_CHANGED;
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "ArpeTmFusionPass success end!");
  return SUCCESS;
}

REG_PASS("ApplyRotaryPosEmbTensorMoveFusionPass", BUILT_IN_GRAPH_PASS, ArpeTmFusionPass, FORBIDDEN_CLOSE);
}  // namespace fe
