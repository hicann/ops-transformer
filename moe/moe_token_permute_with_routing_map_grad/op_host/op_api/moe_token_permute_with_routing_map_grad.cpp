/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_token_permute_with_routing_map_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
static constexpr int64_t GRAD_Y_SHAPE_WITH_GROUP_IDX = 2;
static constexpr int64_t GRAD_Y_SHAPE_NO_GROUP_IDX = 3;

OP_TYPE_REGISTER(MoeTokenPermuteWithRoutingMapGrad);

const std::array<const aclTensor*, 2> MoeTokenPermuteWithRoutingMapGrad(
    const aclTensor* permutedTokenOutputGrad, const aclTensor* permutedProbsOutputGradOptional,
    const aclTensor* sortedIndices, const aclTensor* routingMapOptional, int64_t numExperts, int64_t tokensNum,
    bool dropAndPad, const aclTensor* tokensGradOut, const aclTensor* probsGradOut, aclOpExecutor* executor)
{
    L0_DFX(MoeTokenPermuteWithRoutingMapGrad, permutedTokenOutputGrad, permutedProbsOutputGradOptional, sortedIndices,
           routingMapOptional, numExperts, tokensNum, dropAndPad);
    if (dropAndPad) {
        op::Shape permuteTokensGradShape; // pad模式构造
        permuteTokensGradShape.AppendDim(permutedTokenOutputGrad->GetViewShape().GetDim(0));
        permuteTokensGradShape.AppendDim(0);
        ge::DataType tokenDtype;
        if (permutedProbsOutputGradOptional != nullptr) {
            tokenDtype = permutedProbsOutputGradOptional->GetDataType();
        } else {
            tokenDtype = permutedTokenOutputGrad->GetDataType();
        }
        auto permuteTokensGrad = executor->AllocTensor(permuteTokensGradShape, tokenDtype, op::Format::FORMAT_ND);
        if (permuteTokensGrad == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc permuteTokensGrad tensor failed.");
            return std::array<const aclTensor*, 2>{nullptr, nullptr};
        }
        auto tokensOutGrad = executor->AllocTensor(permuteTokensGradShape, tokenDtype, op::Format::FORMAT_ND);
        if (tokensOutGrad == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc tokensOutGrad tensor failed.");
            return std::array<const aclTensor*, 2>{nullptr, nullptr};
        }
        ADD_TO_LAUNCHER_LIST_AICORE(
            MoeTokenPermuteWithRoutingMapGrad,
            OP_INPUT(permuteTokensGrad, permutedProbsOutputGradOptional, sortedIndices, routingMapOptional),
            OP_OUTPUT(tokensOutGrad, probsGradOut), OP_ATTR(numExperts, tokensNum, dropAndPad));

        return std::array<const aclTensor*, 2>{tokensOutGrad, probsGradOut};
    } else {
        ADD_TO_LAUNCHER_LIST_AICORE(
            MoeTokenPermuteWithRoutingMapGrad,
            OP_INPUT(permutedTokenOutputGrad, permutedProbsOutputGradOptional, sortedIndices, routingMapOptional),
            OP_OUTPUT(tokensGradOut, probsGradOut), OP_ATTR(numExperts, tokensNum, dropAndPad));

        return std::array<const aclTensor*, 2>{tokensGradOut, probsGradOut};
    }
}
} // namespace l0op
