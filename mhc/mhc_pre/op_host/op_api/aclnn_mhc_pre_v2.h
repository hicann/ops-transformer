/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MHC_PRE_V2_H
#define OP_API_INC_MHC_PRE_V2_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first-stage interface of aclnnMhcPreV2. Calculates the required workspace size.
 * @domain aclnn_ops_infer
 * @param [in] x: Input tensor.
 * @param [in] phi: Input projection tensor.
 * @param [in] alpha: Input scaling tensor.
 * @param [in] bias: Input bias tensor.
 * @param [in] gammaOptional: Optional RMSNorm scale tensor.
 * @param [in] normEps: RMSNorm epsilon.
 * @param [in] hcEps: Hyper-connection epsilon.
 * @param [in] opImplMode: Matrix multiplication precision mode. 0 uses FP32 and 1 uses HF32.
 * @param [out] hIn: Output tensor.
 * @param [out] hPost: Output tensor.
 * @param [out] hRes: Output tensor.
 * @param [out] invRmsOptional: Optional output tensor.
 * @param [out] hMixOptional: Optional output tensor.
 * @param [out] hPreOptional: Optional output tensor.
 * @param [out] workspaceSize: Required device workspace size.
 * @param [out] executor: Operator executor.
 * @return aclnnStatus: Status code.
 */
ACLNN_API aclnnStatus aclnnMhcPreV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
    const aclTensor *gammaOptional, double normEps, double hcEps, int64_t opImplMode, aclTensor *hIn,
    aclTensor *hPost, aclTensor *hRes, aclTensor *invRmsOptional, aclTensor *hMixOptional, aclTensor *hPreOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief The second-stage interface of aclnnMhcPreV2. Executes the operator.
 * @domain aclnn_ops_infer
 * @param [in] workspace: Device workspace address.
 * @param [in] workspaceSize: Workspace size returned by aclnnMhcPreV2GetWorkspaceSize.
 * @param [in] executor: Operator executor returned by aclnnMhcPreV2GetWorkspaceSize.
 * @param [in] stream: ACL stream used to execute the operator.
 * @return aclnnStatus: Status code.
 */
ACLNN_API aclnnStatus aclnnMhcPreV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MHC_PRE_V2_H
