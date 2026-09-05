/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_BLOCK_ATTENTION_RESIDUALS_GRAD_H
#define OP_API_INC_ACLNN_BLOCK_ATTENTION_RESIDUALS_GRAD_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBlockAttentionResidualsGrad的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] partialBlock: 表示公式中的partialBlock，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 *
 * @param [in] blockRes: 表示公式中的blockRes，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [in]
 * projWeight: 表示公式中的projWeight，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [in]
 * normWeight: 表示公式中的normWeight，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [in]
 * gradHiddenStates:
 * 表示上游梯度gradHiddenStates，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param
 * [in] invNorm: 表示逆RMS归一化系数invNorm，数据类型支持FLOAT32，数据格式支持ND。
 * @param [in] probs:
 * 表示softmax概率probs，数据类型支持FLOAT32，数据格式支持ND。
 * @param [in] validBlockNum:
 * 表示预留属性，当前内核未使用，默认值为0。
 * @param [out] gradPartialBlock:
 * 表示partialBlock的梯度，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [out] gradBlockRes:
 * 表示blockRes的梯度，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [out] gradProjWeight:
 * 表示projWeight的梯度，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [out] gradNormWeight:
 * 表示normWeight的梯度，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
 * @param [out] workspaceSize:
 * 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBlockAttentionResidualsGradGetWorkspaceSize(
    const aclTensor *partialBlock, const aclTensor *blockRes, const aclTensor *projWeight, const aclTensor *normWeight,
    const aclTensor *gradHiddenStates, const aclTensor *invNorm, const aclTensor *probs, int64_t validBlockNum,
    const aclTensor *gradPartialBlock, const aclTensor *gradBlockRes, const aclTensor *gradProjWeight,
    const aclTensor *gradNormWeight, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnBlockAttentionResidualsGrad的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnBlockAttentionResidualsGradGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBlockAttentionResidualsGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_BLOCK_ATTENTION_RESIDUALS_GRAD_H
