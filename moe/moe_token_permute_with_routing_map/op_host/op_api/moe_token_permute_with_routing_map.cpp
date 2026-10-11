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
 * \file moe_token_permute_with_routing_map.cpp
 * \brief
 */

#include "moe_token_permute_with_routing_map.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "external/aclnn_kernels/aclnn_platform.h"

using namespace op;

namespace l0op {

static constexpr int64_t GRAD_Y_SHAPE_WITH_GROUP_IDX = 2;
static constexpr int64_t GRAD_Y_SHAPE_NO_GROUP_IDX = 3;

OP_TYPE_REGISTER(MoeTokenPermuteWithRoutingMap);

const std::array<const aclTensor*, 3> MoeTokenPermuteWithRoutingMap(const aclTensor* tokens,
                                                                    const aclTensor* routingMap,
                                                                    const aclTensor* probsOptional,
                                                                    int64_t numOutTokens, bool dropAndPad,
                                                                    aclOpExecutor* executor)
{
    L0_DFX(MoeTokenPermuteWithRoutingMap, tokens, routingMap, probsOptional, numOutTokens, dropAndPad);

    op::Shape probsOutShape;
    const aclTensor* probsOut = nullptr;

    int64_t tokenNum = routingMap->GetViewShape().GetDim(1);
    int64_t expertNum = routingMap->GetViewShape().GetDim(0);
    int64_t alignNum = (dropAndPad == true) ? expertNum : tokenNum;
    alignNum = (alignNum == 0) ? 1 : alignNum;
    numOutTokens = numOutTokens / alignNum * alignNum;
    probsOutShape.AppendDim(numOutTokens);

    if (probsOptional != nullptr) {
        probsOut = executor->AllocTensor(probsOutShape, probsOptional->GetDataType(), op::Format::FORMAT_ND);
    } else {
        op::Shape zeroShape;
        zeroShape.AppendDim(0);
        probsOut = executor->AllocTensor(zeroShape, tokens->GetDataType(), op::Format::FORMAT_ND);
    }
    if (probsOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc probsOut tensor failed.");
        return {nullptr, nullptr, nullptr};
    }
    op::Shape permuteTokensOutShape;

    bool isRegbase = Ops::Transformer::AclnnUtil::IsRegbase();

    if (!dropAndPad || isRegbase) { // regbase 场景在算子内部进行gather，其余场景由gather算子计算
        permuteTokensOutShape.AppendDim(numOutTokens);
        permuteTokensOutShape.AppendDim(tokens->GetViewShape().GetDim(1));
    } else {
        permuteTokensOutShape.AppendDim(0);
        permuteTokensOutShape.AppendDim(0);
    }

    auto permuteTokensOut = executor->AllocTensor(permuteTokensOutShape, tokens->GetDataType(), op::Format::FORMAT_ND);
    if (permuteTokensOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc probsOut tensor failed.");
        return {nullptr, nullptr, nullptr};
    }
    auto sortedIndicesOut = executor->AllocTensor(probsOutShape, op::DataType::DT_INT32, op::Format::FORMAT_ND);
    if (sortedIndicesOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc sortedIndicesOut tensor failed.");
        return {nullptr, nullptr, nullptr};
    }

    // 根据输入shape申请输出tensor
    ADD_TO_LAUNCHER_LIST_AICORE(MoeTokenPermuteWithRoutingMap, OP_INPUT(tokens, routingMap, probsOptional),
                                OP_OUTPUT(permuteTokensOut, probsOut, sortedIndicesOut),
                                OP_ATTR(numOutTokens, dropAndPad));
    return {permuteTokensOut, probsOut, sortedIndicesOut};
}
} // namespace l0op
