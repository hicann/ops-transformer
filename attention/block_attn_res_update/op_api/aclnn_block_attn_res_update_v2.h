/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_BLOCK_ATTN_RES_UPDATE_V2_H_
#define ACLNN_BLOCK_ATTN_RES_UPDATE_V2_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算执行 aclnnBlockAttnResUpdateV2 所需的 workspace 大小并创建执行器，支持可选输出 RMSNorm。
 * @domain aclnn_ops_infer
 * @param partialBlockRef [in/out] 必选连续 FLOAT [T, D]，原地写回未归一化的残差累加值。
 * @param delta [in] 必选连续 BFLOAT16 [T, D]。
 * @param pseudoQuery [in] 必选连续 FLOAT [D]。
 * @param numerator [in] 必选连续 FLOAT [T, D]。
 * @param logitMax [in] 必选连续 FLOAT [T]。
 * @param expSum [in] 必选连续 FLOAT [T]。
 * @param outputNormWeightOptional [in] 可选连续 ND gamma [D]，FLOAT 或 BFLOAT16；nullptr 禁用输出 RMSNorm。
 * @param eps [in] 残差 RMS score 的 epsilon，必须为有限正数，非空计算时转换为 FLOAT32 后仍须为有限正数。
 * @param outputNormEps [in] 输出 RMSNorm 的独立 epsilon。传入权重时须在 [FLT_MIN, FLT_MAX] 范围内，权重为空指针时忽略。
 * @param h [out] 必选连续 BFLOAT16 [T, D]；RMSNorm 使用舍入后的 BFLOAT16 注意力混合结果。
 * @param workspaceSize [out] 返回所需 workspace 大小。
 * @param executor [out] 返回执行器。
 * @return aclnnStatus 状态码。
 */
ACLNN_API aclnnStatus aclnnBlockAttnResUpdateV2GetWorkspaceSize(
    aclTensor* partialBlockRef, const aclTensor* delta, const aclTensor* pseudoQuery, const aclTensor* numerator,
    const aclTensor* logitMax, const aclTensor* expSum, const aclTensor* outputNormWeightOptional, double eps,
    double outputNormEps, aclTensor* h, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief 执行支持可选输出 RMSNorm 的 block_attn_res_update 算子计算。
 * @param workspace [in] workspace 内存地址，workspaceSize 为 0 时可传入 nullptr。
 * @param workspaceSize [in] workspace 大小。
 * @param executor [in] 执行器。
 * @param stream [in] ACL 流。
 * @return aclnnStatus 状态码。
 */
ACLNN_API aclnnStatus aclnnBlockAttnResUpdateV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_BLOCK_ATTN_RES_UPDATE_V2_H_
