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
 * \file moe_token_unpermute_with_routing_map.cpp
 * \brief
 */

#include "moe_token_unpermute_with_routing_map.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeTokenUnpermuteWithRoutingMap);

const std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*> MoeTokenUnpermuteWithRoutingMapAICore(
    const aclTensor* permutedTokens, const aclTensor* sortedIndices, const aclTensor* routingMapOptional,
    const aclTensor* probsOptional, bool paddedMode, const aclIntArray* restoreShapeOptional,
    aclTensor* unpermutedTokens, aclTensor* outIndex, aclTensor* permuteTokenId, aclTensor* permuteProbs,
    aclOpExecutor* executor)
{
    L0_DFX(MoeTokenUnpermuteWithRoutingMapAICore, permutedTokens, sortedIndices, routingMapOptional, probsOptional,
           paddedMode, restoreShapeOptional, unpermutedTokens, outIndex, permuteTokenId, permuteProbs);
    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将算子加入任务队列
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        MoeTokenUnpermuteWithRoutingMap, OP_INPUT(permutedTokens, sortedIndices, routingMapOptional, probsOptional),
        OP_OUTPUT(unpermutedTokens, outIndex, permuteTokenId, permuteProbs), OP_ATTR(paddedMode, restoreShapeOptional));
    (void)retAicore;
    return std::tie(unpermutedTokens, outIndex, permuteTokenId, permuteProbs);
}

const std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*> MoeTokenUnpermuteWithRoutingMap(
    const aclTensor* permutedTokens, const aclTensor* sortedIndices, const aclTensor* routingMapOptional,
    const aclTensor* probsOptional, bool paddedMode, const aclIntArray* restoreShapeOptional, aclOpExecutor* executor)
{
    // 根据推导出的输出shape申请输出tensor
    // 给非pad模式预留构造permuteTokenId
    op::Shape unpermutedTokensShape;
    aclTensor* permuteProbs = nullptr;
    op::Shape zeroShape;
    zeroShape.AppendDim(0);
    int32_t num_tokens = -1;
    int32_t restoreShapeSize = 2;
    int32_t hidden_size = permutedTokens->GetViewShape().GetDim(1);

    if (restoreShapeOptional != nullptr && restoreShapeOptional->Size() == restoreShapeSize) {
        num_tokens = (*restoreShapeOptional)[0];
    }
    if (probsOptional != nullptr) {
        num_tokens = probsOptional->GetViewShape().GetDim(0);
    }
    if (num_tokens == -1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Cannot get num_tokens which is needed to use, failed.");
        return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*>(nullptr, nullptr, nullptr, nullptr);
    }

    unpermutedTokensShape.AppendDim(num_tokens);
    unpermutedTokensShape.AppendDim(hidden_size);

    auto unpermutedTokens =
        executor->AllocTensor(unpermutedTokensShape, permutedTokens->GetDataType(), permutedTokens->GetViewFormat());
    auto permuteTokenId = executor->AllocTensor(sortedIndices->GetViewShape(), sortedIndices->GetDataType(),
                                                sortedIndices->GetViewFormat());
    auto outIndex = executor->AllocTensor(sortedIndices->GetViewShape(), sortedIndices->GetDataType(),
                                          sortedIndices->GetViewFormat());

    if (probsOptional != nullptr) {
        permuteProbs = executor->AllocTensor(sortedIndices->GetViewShape(), probsOptional->GetDataType(),
                                             sortedIndices->GetViewFormat());
    } else {
        permuteProbs = executor->AllocTensor(zeroShape, permutedTokens->GetDataType(), op::Format::FORMAT_ND);
    }

    return MoeTokenUnpermuteWithRoutingMapAICore(permutedTokens, sortedIndices, routingMapOptional, probsOptional,
                                                 paddedMode, restoreShapeOptional, unpermutedTokens, outIndex,
                                                 permuteTokenId, permuteProbs, executor);
}
} // namespace l0op
