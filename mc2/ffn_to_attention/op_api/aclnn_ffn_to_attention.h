/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_FFN_TO_ATTENTION_H_
#define OP_API_INC_FFN_TO_ATTENTION_H_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * 算子功能：实现ffn_to_attention数据发送
 * @brief aclnnFFNToAttention的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x: 计算输入，Tensor，输入的token
 * @param [in] sessionIds: 计算输入，Tensor，表示每个token对应的attention
 * @param [in] microBatchIds: 计算输入，Tensor，表示各个token归属的micro_batch_ids
 * @param [in] tokenIds: 计算输入，Tensor，每个token对应的micro_batch中batch偏移
 * @param [in] expertOffsets: 计算输入，Tensor，表示topk中的第几个
 * @param [in] actualTokenNum: 计算输入，Tensor，所有专家有效token数之和，用于保证只发送有效数据
 * @param [in] attnRankTable: 计算输入，可选输入，Tensor，缺省场景attn_worker_id=rank_id，attn
 * worker必须从0卡开始连续部署
 * @param [in] group: 计算输入，str，通信域名称，专家并行的通信域。
 * @param [in] worldSize: 计算输入，int，通信域size
 * @param [in] tokenInfoTableShape: 计算输入，list int，token信息表的shape
 * @param [in] tokenDataShape: 计算输入，list int，token数据表的shape
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码。
 *
 */
ACLNN_API aclnnStatus aclnnFFNToAttentionGetWorkspaceSize(
    const aclTensor* x, const aclTensor* sessionIds, const aclTensor* microBatchIds, const aclTensor* tokenIds,
    const aclTensor* expertOffsets, const aclTensor* actualTokenNum, const aclTensor* attnRankTable, const char* group,
    int64_t worldSize, const aclIntArray* tokenInfoTableShape, const aclIntArray* tokenDataShape,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnFFNToAttention的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnFFNToAttentionGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnFFNToAttention(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ATTENTION_TO_MOE_
