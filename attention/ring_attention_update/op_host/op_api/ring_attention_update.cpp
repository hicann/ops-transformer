/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ring_attention_update.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(RingAttentionUpdate);

const std::array<const aclTensor*, 3> RingAttentionUpdate(const aclTensor* prevAttnOut, const aclTensor* prevSoftmaxMax,
                                                          const aclTensor* prevSoftmaxSum, const aclTensor* curAttnOut,
                                                          const aclTensor* curSoftmaxMax,
                                                          const aclTensor* curSoftmaxSum,
                                                          const aclTensor* actualSeqQLen, const char* inputLayout,
                                                          const char* inputSoftmaxLayout, aclOpExecutor* executor)
{
    L0_DFX(RingAttentionUpdate, prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum,
           actualSeqQLen);

    auto attentionOutOut = executor->AllocTensor(prevAttnOut->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxMaxOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxSumOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(
        RingAttentionUpdate,
        OP_INPUT(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, actualSeqQLen),
        OP_OUTPUT(attentionOutOut, softmaxMaxOut, softmaxSumOut), OP_ATTR(inputLayout, inputSoftmaxLayout));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "RingAttentionUpdate InferShape failed.");
        return {nullptr, nullptr, nullptr};
    }

    ADD_TO_LAUNCHER_LIST_AICORE(
        RingAttentionUpdate,
        OP_INPUT(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum, curAttnOut, curSoftmaxMax, curSoftmaxSum, actualSeqQLen),
        OP_OUTPUT(attentionOutOut, softmaxMaxOut, softmaxSumOut), OP_ATTR(inputLayout, inputSoftmaxLayout));
    return {attentionOutOut, softmaxMaxOut, softmaxSumOut};
}
} // namespace l0op
