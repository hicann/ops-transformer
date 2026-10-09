/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ATTENTION_TO_FFN_H_
#define OP_API_INC_ATTENTION_TO_FFN_H_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现attention_to_ffn数据发送
 * @brief aclnnAttentionToFFN的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x: 计算输入，Tensor，输入的token数据.
 * @param [in] sessionId: 计算输入，Tensor，会话ID，代表当前请求的唯一标识。
 * @param [in] microBatchId: 计算输入，Tensor, Batch切分成多少个小batch，对应的小BatchId
 * @param [in] layerId: 计算输入，Tensor, 模型层数对应的层Id
 * @param [in] expertIds: 计算输入，Tensor, 每个token的topK个专家索引。
 * @param [in] expertRankTable: 计算输入，Tensor, 专家rank表。
 * @param [in] scales: 计算输入，可选输入，Tensor, 专家smooth权重。
 * @param [in] activeMask: 计算输入，可选输入，Tensor, token的激活掩码。
 * @param [in] group: 计算输入，str, 通信group名称
 * @param [in] worldSize: 计算输入，int, world size
 * @param [in] ffnTokenInfoTableShape: 计算输入，list int, 发给FFN的token信息表的shape信息
 * @param [in] ffnTokenDataShape: 计算输入，list int, 发给FFN的实际token数据的shape信息
 * @param [in] attnTokenInfoTableShape: 计算输入，list int, Attention侧的token信息表的shape信息
 * @param [in] moeExpertNum: 计算输入，int, MOE专家数量
 * @param [in] quantMode: 计算输入，int, 量化模式
 * @param [in] syncFlag: 计算输入，int, 同步标志
 * @param [in] ffnStartRankId: 计算输入，int, ffn起始rankId
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码。
 *
 */
ACLNN_API aclnnStatus aclnnAttentionToFFNGetWorkspaceSize(
    const aclTensor* x, const aclTensor* sessionId, const aclTensor* microBatchId, const aclTensor* layerId,
    const aclTensor* expertIds, const aclTensor* expertRankTable, const aclTensor* scales, const aclTensor* activeMask,
    const char* group, int64_t worldSize, const aclIntArray* ffnTokenInfoTableShape,
    const aclIntArray* ffnTokenDataShape, const aclIntArray* attnTokenInfoTableShape, int64_t moeExpertNum,
    int64_t quantMode, int64_t syncFlag, int64_t ffnStartRankId, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAttentionToFFN的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAttentionToFFNGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnAttentionToFFN(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ATTENTION_TO_FFN_
