/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_BLOCK_SPARSE_ATTENTION_GRAD_H
#define ACLNN_BLOCK_SPARSE_ATTENTION_GRAD_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBlockSparseAttentionGrad的第一段接口,根据具体的计算流程,计算workspace大小
 * @domain aclnn_ops_infer
 *
 * @param [in] dout: attention out 梯度输入
 * @param [in] query: Query tensor输入
 * @param [in] key: Key tensor输入
 * @param [in] value: Value tensor输入
 * @param [in] attentionOut: attention out tensor输入
 * @param [in] softmaxLse: softmax tensor输入
 * @param [in] blockSparseMaskOptional: 稀疏块索引 [T, headNum, maxKvBlockNum]
 * @param [in] attenMaskOptional: Attention mask (只支持传入nullptr)
 * @param [in] blockShapeOptional: 稀疏块形状数组 [blockShapeX, blockShapeY]
 * @param [in] actualSeqLengthsOptional: 实际序列长度Q (在qInputLayout为“TND”时必选)
 * @param [in] actualSeqLengthsKvOptional: 实际序列长度KV (在kvInputLayout为“TND”时必选)
 * @param [in] qInputLayout: Query输入layout ("TND", "BNSD")
 * @param [in] kvInputLayout: KV输入layout ("TND", "BNSD")
 * @param [in] numKeyValueHeads: KV头数
 * @param [in] maskType: Mask类型
 * @param [in] scaleValue: 缩放因子 (double类型)
 * @param [in] preTokens: 滑动窗口大小
 * @param [in] nextTokens: 滑动窗口大小
 * @param [out] dq: Attention输出tensor
 * @param [out] dk: Attention输出tensor
 * @param [out] dv: Attention输出tensor
 * @param [out] workspaceSize: 返回计算所需workspace大小
 * @param [out] executor: 返回op执行器
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionGradGetWorkspaceSize(
    const aclTensor *dout, const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, const aclTensor *blockSparseMaskOptional,
    const aclTensor *attenMaskOptional, const aclIntArray *blockShapeOptional,
    const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional, char *qInputLayout,
    char *kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue, int64_t preTokens,
    int64_t nextTokens, aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnBlockSparseAttentionGrad的第二段接口,用于执行计算
 *
 * @param [in] workspace: 工作空间地址
 * @param [in] workspaceSize: 工作空间大小
 * @param [in] executor: op执行器
 * @param [in] stream: acl stream流
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnBlockSparseAttentionGrad(void *workspace,
                                                                                 uint64_t workspaceSize,
                                                                                 aclOpExecutor *executor,
                                                                                 const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_BLOCK_SPARSE_ATTENTION_GRAD_H
