/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "text_gated_residual_grad.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(TextGatedResidualGrad);

namespace {

const std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*> TextGatedResidualGradAICore(
    const aclTensor* gradMixed, const aclTensor* x, const aclTensor* wDown, const aclTensor* wUp,
    const aclTensor* invRms, const aclTensor* downPreAct, const aclTensor* up, const aclTensor* gradInject,
    const aclTensor* wGate, const aclTensor* gamma, const aclTensor* inject, int64_t channels, aclTensor* dx,
    aclTensor* dwDown, aclTensor* dwUp, aclTensor* dwGate, aclTensor* dgamma, aclOpExecutor* executor)
{
    L0_DFX(TextGatedResidualGradAICore, gradMixed, x, wDown, wUp, invRms, downPreAct, up, gradInject, wGate, gamma,
           inject, dx, dwDown, dwUp, dwGate, dgamma);

    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        TextGatedResidualGrad,
        OP_INPUT(gradMixed, x, wDown, wUp, invRms, downPreAct, up, gradInject, wGate, gamma, inject),
        OP_OUTPUT(dx, dwDown, dwUp, dwGate, dgamma), OP_ATTR(channels));
    if (retAicore != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TextGatedResidualGrad launch kernel failed.");
        return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(nullptr, nullptr, nullptr,
                                                                                      nullptr, nullptr);
    }
    return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(dx, dwDown, dwUp, dwGate, dgamma);
}

} // namespace

const std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*> TextGatedResidualGrad(
    const aclTensor* gradMixed, const aclTensor* x, const aclTensor* wDown, const aclTensor* wUp,
    const aclTensor* invRms, const aclTensor* downPreAct, const aclTensor* up, const aclTensor* gradInject,
    const aclTensor* wGate, const aclTensor* gamma, const aclTensor* inject, int64_t channels, aclOpExecutor* executor)
{
    L0_DFX(TextGatedResidualGrad, gradMixed, x, wDown, wUp, invRms, downPreAct, up, gradInject, wGate, gamma, inject);

    // Empty tensor placeholders for omitted optional inputs/outputs.
    op::Shape emptyShape({0});
    auto allocateEmptyTensor = [&](op::DataType dtype, op::Format format) {
        return executor->AllocTensor(emptyShape, dtype, format);
    };

    auto gammaInput = gamma != nullptr ? gamma : allocateEmptyTensor(op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto gradInjectInput =
        gradInject != nullptr ? gradInject : allocateEmptyTensor(x->GetDataType(), x->GetViewFormat());
    auto wGateInput = wGate != nullptr ? wGate : allocateEmptyTensor(x->GetDataType(), x->GetViewFormat());
    auto injectInput = inject != nullptr ? inject : allocateEmptyTensor(x->GetDataType(), x->GetViewFormat());

    // Allocate outputs.
    auto dx = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), x->GetViewFormat());
    auto dwDown = executor->AllocTensor(wDown->GetViewShape(), wDown->GetDataType(), wDown->GetViewFormat());
    auto dwUp = executor->AllocTensor(wUp->GetViewShape(), wUp->GetDataType(), wUp->GetViewFormat());
    aclTensor* dwGate = nullptr;
    if (wGate != nullptr) {
        dwGate = executor->AllocTensor(wGate->GetViewShape(), wGate->GetDataType(), wGate->GetViewFormat());
    } else {
        dwGate = allocateEmptyTensor(x->GetDataType(), op::Format::FORMAT_ND);
    }
    aclTensor* dgamma = nullptr;
    if (gamma != nullptr) {
        dgamma = executor->AllocTensor(gamma->GetViewShape(), op::DataType::DT_FLOAT, gamma->GetViewFormat());
    } else {
        dgamma = allocateEmptyTensor(op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    }

    return TextGatedResidualGradAICore(gradMixed, x, wDown, wUp, invRms, downPreAct, up, gradInjectInput, wGateInput,
                                       gammaInput, injectInput, channels, dx, dwDown, dwUp, dwGate, dgamma, executor);
}

} // namespace l0op
