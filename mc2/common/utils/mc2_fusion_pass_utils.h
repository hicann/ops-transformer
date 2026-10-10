/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_MC2_FUSION_PASS_UTILS_H
#define OPS_TRANSFORMER_MC2_FUSION_PASS_UTILS_H

#include <initializer_list>
#include <memory>
#include <string>

#include "ge/fusion/match_result.h"
#include "graph/graph.h"
#include "mc2_common_log.h"

namespace ops {
/**
 * 目标图是否包含任一锚点算子类型。
 * 用于 PatternFusionPass::Run 前置剪枝：无锚点时跳过 Patterns()/MatchNext。
 */
inline bool GraphHasAnyOpType(const ge::Graph& graph, std::initializer_list<const char*> opTypes)
{
    for (const auto& node : graph.GetDirectNode()) {
        ge::AscendString type;
        if (node.GetType(type) != ge::GRAPH_SUCCESS) {
            continue;
        }
        for (const char* opType : opTypes) {
            if (opType != nullptr && type == opType) {
                return true;
            }
        }
    }
    return false;
}

/** 从 MatchResult 读取 pattern graph 名称。 */
inline bool GetPatternNameStr(const std::unique_ptr<ge::fusion::MatchResult>& matchResult, std::string& patternNameStr,
                              const char* passName)
{
    ge::AscendString patternName = "";
    if (matchResult->GetPatternGraph().GetName(patternName) != ge::SUCCESS) {
        OPS_LOG_W(passName, "Get pattern graph name failed.");
        return false;
    }
    patternNameStr = patternName.GetString() != nullptr ? patternName.GetString() : "";
    return true;
}

/** 按 capture 下标取出匹配到的 MC2 锚点节点。 */
inline bool GetCapturedMc2Node(const std::unique_ptr<ge::fusion::MatchResult>& matchResult, ge::GNode& mc2Node,
                               int64_t captureIdx, const char* passName, const char* errMsg)
{
    ge::fusion::NodeIo mc2NodeIo;
    OP_LOGE_IF(matchResult->GetCapturedTensor(captureIdx, mc2NodeIo) != ge::SUCCESS, false, passName, "%s", errMsg);
    mc2Node = mc2NodeIo.node;
    return true;
}
} // namespace ops

#endif // OPS_TRANSFORMER_MC2_FUSION_PASS_UTILS_H
