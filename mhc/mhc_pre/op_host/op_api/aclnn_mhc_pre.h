/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MHC_PRE_H
#define OP_API_INC_MHC_PRE_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMhcPreGetWorkspaceSize 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：MhcPre正向算子
 * @param [in] x: 输入tensor，shape为[B,S,N,D]或[T,N,D]，数据类型支持：BF16/FP16。
 * @param [in] phi: 输入tensor，shape为[n!+2n, nD]或[n^2+2n, nD]或[2n, nD]（alpha维度决定），数据类型支持：FP32。
 * @param [in] alpha: 输入tensor，shape为[3]或[2]，数据类型支持：FP32。alpha=[3]时有hRes输出，alpha=[2]时无hRes输出。
 * @param [in] bias: 输入tensor，shape为[2N]或[n^2+2n]或[n!+2n]（alpha维度决定），数据类型支持：FP32。
 * @param [in] gammaOptional: 可选输入tensor，shape为[n, D]，数据类型支持：FP32。
 * @param [in] normEps: 归一化epsilon参数。
 * @param [in] hcEps: hyper connection epsilon参数。
 * @param [out] hIn: 输出tensor，shape为[B,S,D]或[T,D]，数据类型：BF16/FP16。
 * @param [out] hPost: 输出tensor，shape为[B,S,N]或[T,N]，数据类型：FP32。
 * @param [out] hRes:
 * 可选输出tensor，alpha=[3]时shape为[B,S,N!]或[B,S,N,N]或[T,N!]或[T,N,N]，alpha=[2]时为0，数据类型：FP32。
 * @param [out] invRmsOptional: 可选输出tensor，shape为[B,S]或[T]，数据类型：FP32。
 * @param [out] hMixOptional:
 * 可选输出tensor，shape为[B,S,N!+2N]或[B,S,N^2+2N]或[B,S,2N]（alpha维度决定），数据类型：FP32。
 * @param [out] hPreOptional: 可选输出tensor，shape为[B,S,N]或[T,N]，数据类型：FP32。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMhcPreGetWorkspaceSize(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                                  const aclTensor *bias, const aclTensor *gammaOptional, double normEps,
                                                  double hcEps, aclTensor *hIn, aclTensor *hPost, aclTensor *hRes,
                                                  aclTensor *invRmsOptional, aclTensor *hMixOptional,
                                                  aclTensor *hPreOptional, uint64_t *workspaceSize,
                                                  aclOpExecutor **executor);

/**
 * @brief aclnnMhcPre 的第二段接口，执行算子计算。
 * @domain aclnn_ops_infer
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口 aclnnMhcPreGetWorkspaceSize 获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMhcPre(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MHC_PRE_H
