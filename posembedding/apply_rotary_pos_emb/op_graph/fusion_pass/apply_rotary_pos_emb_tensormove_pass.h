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
 * \file apply_rotary_pos_emb_tensormove_pass.h
 * \brief Permute fusion pass(TensorMove -> ApplyRotaryPosEmb/ApplyRotaryPosEmb)
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ArpeTmFUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ArpeTmFUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ArpeTmFusionPass : public PatternFusionBasePass {
protected:
    vector<FusionPattern *> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes) override;

private:
    Status FusionProcess(ge::NodePtr arpeNode, int idx, ge::ComputeGraph &graph);
    Status RemoveTensorMoveNode(ge::NodePtr arpeNode, ge::NodePtr fusedNode, ge::ComputeGraph &graph);
    const string FUSED_OP_TYPE = "ArpeTmFusionPass";
};
} // namespace fe

#endif // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_ArpeTmFUSION_PASS_H_
