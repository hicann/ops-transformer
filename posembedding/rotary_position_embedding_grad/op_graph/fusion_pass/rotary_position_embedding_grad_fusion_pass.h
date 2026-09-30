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
 * \file rotary_position_embedding_grad_fusion_pass.h
 * \brief RotaryMulGrad --> RotaryPositionEmbeddingGrad(mode = 0), migrated from canndev built-in rope_fusion_pass.
 */

#ifndef TRANSFORMER_ROTARY_POSITION_EMBEDDING_GRAD_FUSION_PASS_H
#define TRANSFORMER_ROTARY_POSITION_EMBEDDING_GRAD_FUSION_PASS_H

#include "version/cann_version.h"

#define ROTARY_POSITION_EMBEDDING_GRAD_GRAPH_FUSION_SUPPORT_VERSION 90000000

#if CANN_VERSION_NUM >= ROTARY_POSITION_EMBEDDING_GRAD_GRAPH_FUSION_SUPPORT_VERSION
#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {

class __attribute__((visibility("default"))) RotaryMulGrad2RotaryPositionEmbeddingGradFusionPass
    : public ge::fusion::PatternFusionPass {
protected:
    std::vector<ge::fusion::PatternUniqPtr> Patterns() override;

    bool MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& matchResult) override;

    ge::fusion::GraphUniqPtr Replacement(const std::unique_ptr<ge::fusion::MatchResult>& matchResult) override;
};

} // namespace ops

#endif // CANN_VERSION_NUM >= ROTARY_POSITION_EMBEDDING_GRAD_GRAPH_FUSION_SUPPORT_VERSION
#endif // TRANSFORMER_ROTARY_POSITION_EMBEDDING_GRAD_FUSION_PASS_H
