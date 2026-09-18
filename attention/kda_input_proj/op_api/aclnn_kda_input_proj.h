/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_KDA_INPUT_PROJ_H
#define ACLNN_KDA_INPUT_PROJ_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算KdaInputProj所需的workspace大小并创建算子执行器。
 * @domain aclnn_ops_infer
 *
 * 权重按 matmul 右操作数解释：view shape 为 [K, N]。若末两维是列主序 view
 *（stride[-2]==1 且 stride[-1]==shape[-2]，与 MatMulV3 相同），则按转置权重处理。
 *
 * @param x [IN] 隐藏层输入，shape为[T, K]，BF16，ND。
 * @param weightQkv [IN] qkv投影权重，FLOAT8_E4M3FN，ND，view [K, N_qkv]。
 * @param weightBeta [IN] beta投影权重，BF16，ND，view [K, N_beta]。
 * @param weightGate [IN] gate投影权重，BF16，ND，view [K, N_gate]。
 * @param weightG [IN] g投影权重，BF16，ND，view [K, N_g]。
 * @param weightQkvScale [IN] qkv MX量化缩放，FLOAT8_E8M0，ND，最后一维固定为2。
 * @param qkvOut [OUT] qkv输出，shape为[T, N_qkv]，BF16，ND。
 * @param betaOut [OUT] beta输出，shape为[T, N_beta]，FLOAT，ND。
 * @param gateOut [OUT] gate输出，shape为[T, N_gate]，BF16，ND。
 * @param gOut [OUT] g输出，shape为[T, N_g]，BF16，ND。
 * @param workspaceSize [OUT] Device侧workspace大小，单位为字节。
 * @param executor [OUT] 算子执行器。
 * @return aclnnStatus 成功时返回ACLNN_SUCCESS。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnKdaInputProjGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weightQkv, const aclTensor *weightBeta, const aclTensor *weightGate,
    const aclTensor *weightG, const aclTensor *weightQkvScale, const aclTensor *qkvOut, const aclTensor *betaOut,
    const aclTensor *gateOut, const aclTensor *gOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief 使用aclnnKdaInputProjGetWorkspaceSize创建的执行器异步执行KdaInputProj。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnKdaInputProj(void *workspace, uint64_t workspaceSize,
                                                                     aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_KDA_INPUT_PROJ_H
