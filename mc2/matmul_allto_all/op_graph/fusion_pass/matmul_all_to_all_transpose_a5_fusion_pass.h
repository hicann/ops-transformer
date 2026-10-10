/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSFORMER_MATMUL_ALL_TO_ALL_TRANSPOSE_A5_FUSION_PASS_H
#define TRANSFORMER_MATMUL_ALL_TO_ALL_TRANSPOSE_A5_FUSION_PASS_H
#include "version/ge-compiler_version.h"
#define GRAPH_FUSION_SUPPORT_VERSION 90000000
#if GE_COMPILER_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {
class __attribute__((visibility("default"))) MatmulAllToAllTransposeA5FusionPass
    : public ge::fusion::PatternFusionPass {
public:
    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& pass_context) override;

protected:
    std::vector<ge::fusion::PatternUniqPtr> Patterns() override;

    bool MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& matchResult) override;

    ge::fusion::GraphUniqPtr Replacement(const std::unique_ptr<ge::fusion::MatchResult>& matchResult) override;
};
} // namespace ops
#endif // GRAPH_FUSION_SUPPORT_VERSION
#endif // TRANSFORMER_MATMUL_ALL_TO_ALL_TRANSPOSE_A5_FUSION_PASS_H
