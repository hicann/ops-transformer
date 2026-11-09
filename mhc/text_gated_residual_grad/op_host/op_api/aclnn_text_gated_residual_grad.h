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
 * \file aclnn_text_gated_residual_grad.h
 * \brief TextGatedResidualGrad ACLNN workspace-query and execution interfaces.
 */

#ifndef OP_API_INC_ACLNN_TEXT_GATED_RESIDUAL_GRAD_H
#define OP_API_INC_ACLNN_TEXT_GATED_RESIDUAL_GRAD_H

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief First-stage aclnnTextGatedResidualGrad API: calculate workspace size.
 * Compute gradients for the RMS normalization, mixed output and injection branch of TextGatedResidual.
 * Inputs follow OpDef order; C = channels, D = C * H, and R is the projection size.
 * The activation and projection-weight tensors use one common FLOAT16 or BFLOAT16 dtype.
 * @param [in] gradMixed Gradient of mixed output, shape [B, S, H].
 * @param [in] x Forward input, shape [B, S, D].
 * @param [in] wDown Down-projection weight, shape [R, D].
 * @param [in] wUp Up-projection weight, shape [D, R].
 * @param [in] invRms Saved inverse RMS, shape [B, S, C], FLOAT32.
 * @param [in] downPreAct Saved down-projection result before division by C, shape [B, S, R].
 * @param [in] up Saved up-projection result, shape [B, S, D].
 * @param [in] gradInjectOptional Injection-output gradient, shape [B, S, C].
 * @param [in] wGateOptional Gate-projection weight, shape [C, D].
 * @param [in] gammaOptional RMS weight, shape [D], FLOAT32; normalization uses 1 + gamma.
 * @param [in] injectOptional Saved injection output, shape [B, S, C].
 * @param [in] channels Number of channels.
 * @param [out] dx Input gradient, with the shape and dtype of x.
 * @param [out] dwDown Down-weight gradient, with the shape and dtype of wDown.
 * @param [out] dwUp Up-weight gradient, with the shape and dtype of wUp.
 * @param [out] dwGateOptional Gate-weight gradient, with the shape and dtype of wGateOptional.
 * @param [out] dgammaOptional RMS-weight gradient, shape [D], FLOAT32.
 * @param [out] workspaceSize Required workspace size in bytes.
 * @param [out] executor Executor for the second-stage API.
 * gradInjectOptional, wGateOptional, injectOptional and dwGateOptional must be provided or omitted together.
 * gammaOptional and dgammaOptional must be provided or omitted together.
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnTextGatedResidualGradGetWorkspaceSize(
    const aclTensor* gradMixed, const aclTensor* x, const aclTensor* wDown, const aclTensor* wUp,
    const aclTensor* invRms, const aclTensor* downPreAct, const aclTensor* up, const aclTensor* gradInjectOptional,
    const aclTensor* wGateOptional, const aclTensor* gammaOptional, const aclTensor* injectOptional, int64_t channels,
    aclTensor* dx, aclTensor* dwDown, aclTensor* dwUp, aclTensor* dwGateOptional, aclTensor* dgammaOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief Second-stage aclnnTextGatedResidualGrad API: execute the operator.
 */
ACLNN_API aclnnStatus aclnnTextGatedResidualGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_TEXT_GATED_RESIDUAL_GRAD_H
