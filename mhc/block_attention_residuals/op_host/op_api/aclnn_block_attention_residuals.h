/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_ACLNN_BLOCK_ATTENTION_RESIDUALS_H
#define OP_API_ACLNN_BLOCK_ATTENTION_RESIDUALS_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算BlockAttentionResiduals所需的工作空间大小。
 *
 * @param partialBlock [输入] 形状为[T, H]的ND张量，FLOAT16/BFLOAT16/FLOAT32。
 * @param blockRes [输入] 形状为[T, N, H]的ND张量，dtype须与partialBlock一致。
 * @param projWeight [输入] 形状为[H]或[1, H]的ND张量，dtype须与partialBlock一致。
 * @param normWeight [输入] 形状为[H]的ND张量，dtype须与partialBlock一致。
 * @param validBlockNum [输入] 有效block数量，-1表示使用blockRes的N；当前须等于N或为-1。
 * @param normEps [输入] RMS归一化数值稳定性参数，须为大于0的DOUBLE，默认1e-6。
 * @param needBackward [输入] 是否保存invNorm和probs供反向使用；为true时后两者不能为空。
 * @param hiddenStates [输出] 形状为[T, H]的ND张量，dtype须与partialBlock一致。
 * @param invNorm [输出] 形状为[T, N+1]的FLOAT32 ND张量；needBackward为false时可传入nullptr。
 * @param probs [输出] 形状为[T, N+1]的FLOAT32 ND张量；needBackward为false时可传入nullptr。
 * @param workspaceSize [输出] 设备侧所需的工作空间大小，单位为字节。
 * @param executor [输出] 第二段接口使用的执行器。
 * @return 成功时返回ACLNN_SUCCESS。
 */
ACLNN_API aclnnStatus aclnnBlockAttentionResidualsGetWorkspaceSize(
    const aclTensor *partialBlock, const aclTensor *blockRes, const aclTensor *projWeight, const aclTensor *normWeight,
    int64_t validBlockNum, double normEps, bool needBackward, aclTensor *hiddenStates, aclTensor *invNorm,
    aclTensor *probs, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief 使用第一段接口生成的执行器运行BlockAttentionResiduals。
 *
 * @param workspace [输入] Device侧workspace地址；workspaceSize为0时可传入nullptr。
 * @param workspaceSize [输入] 第一段接口返回的workspace大小，单位为字节。
 * @param executor [输入] 第一段接口返回的op执行器。
 * @param stream [输入] AscendCL Stream。
 * @return ACLNN_SUCCESS或对应错误码。
 */
ACLNN_API aclnnStatus aclnnBlockAttentionResiduals(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_BLOCK_ATTENTION_RESIDUALS_H
