/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_BLOCK_ATTN_RES_PREPARE_H_
#define ACLNN_BLOCK_ATTN_RES_PREPARE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算BlockAttnResPrepare所需的工作空间大小。
 *
 * @param blockRes [输入] 形状为[T, N, D]的FP32 ND张量。
 * @param validBlocks [输入] 形状为[1]的UINT64 ND张量。
 * @param pseudoQuery [输入] FP32 ND张量，shape为[S, D]。
 *
 * @param numerator [输出] 形状为[S, T, D]的FP32 ND张量。
 * @param logitMax [输出] 形状为[S, T]的FP32 ND张量。
 * @param expSum [输出] 形状为[S, T]的FP32 ND张量。
 * @param eps [输入] 用于保证数值稳定性的FP32正数。
 * @param workspaceSize [输出] 设备侧所需的工作空间大小，单位为字节。
 * @param executor [输出] 第二段接口使用的执行器。
 * @return 成功时返回ACLNN_SUCCESS。
 */
ACLNN_API aclnnStatus aclnnBlockAttnResPrepareGetWorkspaceSize(const aclTensor *blockRes, const aclTensor *validBlocks,
                                                               const aclTensor *pseudoQuery, aclTensor *numerator,
                                                               aclTensor *logitMax, aclTensor *expSum, float eps,
                                                               uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief 使用第一段接口生成的执行器运行BlockAttnResPrepare。
 *
 * @param workspace [输入] Device侧workspace地址；workspaceSize为0时可传入nullptr。
 * @param workspaceSize [输入] 第一段接口返回的workspace大小，单位为字节。
 * @param executor [输入] 第一段接口返回的op执行器。
 * @param stream [输入] AscendCL Stream。
 * @return ACLNN_SUCCESS或对应错误码。
 */
ACLNN_API aclnnStatus aclnnBlockAttnResPrepare(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_BLOCK_ATTN_RES_PREPARE_H_
