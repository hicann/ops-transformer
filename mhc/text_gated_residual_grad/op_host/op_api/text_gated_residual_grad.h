/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file text_gated_residual_grad.h
 * \brief TextGatedResidualGrad L0 kernel interface.
 */

#ifndef OP_API_INC_LEVEL0_OP_TEXT_GATED_RESIDUAL_GRAD_H
#define OP_API_INC_LEVEL0_OP_TEXT_GATED_RESIDUAL_GRAD_H

#include <tuple>

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Allocate output tensors and enqueue TextGatedResidualGrad on the executor.
 * Inputs follow the OpDef order: required tensors first, followed by optional tensors.
 * The gate group (gradInject, wGate, inject) is provided or omitted together.
 * @return (dx, dwDown, dwUp, dwGate, dgamma), with empty tensors for omitted outputs.
 * Weight gradients use the corresponding weight dtype; dgamma uses FLOAT32.
 */
const std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*> TextGatedResidualGrad(
    const aclTensor* gradMixed, const aclTensor* x, const aclTensor* wDown, const aclTensor* wUp,
    const aclTensor* invRms, const aclTensor* downPreAct, const aclTensor* up, const aclTensor* gradInject,
    const aclTensor* wGate, const aclTensor* gamma, const aclTensor* inject, int64_t channels, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_TEXT_GATED_RESIDUAL_GRAD_H
