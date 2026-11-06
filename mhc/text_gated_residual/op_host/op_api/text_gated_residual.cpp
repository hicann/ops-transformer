/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "text_gated_residual.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(TextGatedResidual);

const std::array<const aclTensor*, 5> TextGatedResidual(const aclTensor* x, const aclTensor* wDown,
                                                        const aclTensor* wUp, const aclTensor* wGateOptional,
                                                        const aclTensor* gammaOptional, int64_t channels, float eps,
                                                        bool needBackward, aclOpExecutor* executor)
{
    L0_DFX(TextGatedResidual, x, wDown, wUp, wGateOptional, gammaOptional, channels, eps, needBackward);
    const auto dtype = x->GetDataType();
    constexpr auto format = Format::FORMAT_ND;
    auto mixed = executor->AllocTensor(dtype, format, format);
    auto inject = executor->AllocTensor(dtype, format, format);
    auto invRms = executor->AllocTensor(DataType::DT_FLOAT, format, format);
    auto downPreAct = executor->AllocTensor(dtype, format, format);
    auto upPreAct = executor->AllocTensor(dtype, format, format);
    if (mixed == nullptr || inject == nullptr || invRms == nullptr || downPreAct == nullptr || upPreAct == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TextGatedResidual output allocation failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    auto ret =
        INFER_SHAPE(TextGatedResidual, OP_INPUT(x, wDown, wUp, wGateOptional, gammaOptional),
                    OP_OUTPUT(mixed, inject, invRms, downPreAct, upPreAct), OP_ATTR(channels, eps, needBackward));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TextGatedResidual InferShape failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    ret = ADD_TO_LAUNCHER_LIST_AICORE(TextGatedResidual, OP_INPUT(x, wDown, wUp, wGateOptional, gammaOptional),
                                      OP_OUTPUT(mixed, inject, invRms, downPreAct, upPreAct),
                                      OP_ATTR(channels, eps, needBackward));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE TextGatedResidual failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {mixed, inject, invRms, downPreAct, upPreAct};
}
} // namespace l0op
