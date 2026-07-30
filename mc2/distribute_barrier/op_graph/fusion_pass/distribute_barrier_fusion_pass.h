/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSFORMER_DISTRIBUTE_BARRIER_FUSION_PASS_H
#define TRANSFORMER_DISTRIBUTE_BARRIER_FUSION_PASS_H
#include "version/cann_version.h"
#define GRAPH_FUSION_SUPPORT_VERSION 90000000
#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {
class __attribute__((visibility("default"))) DistributeBarrierFusionPass : public ge::fusion::FusionBasePass {
public:
    ge::graphStatus Run(ge::GraphPtr &graph, ge::CustomPassContext &pass_context) override;

private:
    ge::graphStatus FusionNode(ge::Graph &graph, ge::GNode &barrierNode, ge::GNode &contextNode,
                               const ge::TensorDesc &contextDesc);
    ge::graphStatus CreateFusionNode(ge::Graph &graph, ge::GNode &barrierNode, const ge::TensorDesc &contextDesc,
                                     ge::GNode &fusionNode);
    ge::graphStatus AddAttr(ge::GNode &barrierNode, ge::GNode &fusionNode);
    ge::graphStatus AddEdge(ge::Graph &graph, ge::GNode &barrierNode, ge::GNode &fusionNode, ge::GNode &contextNode);
};
} // namespace ops
#endif // CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#endif // TRANSFORMER_DISTRIBUTE_BARRIER_FUSION_PASS_H
