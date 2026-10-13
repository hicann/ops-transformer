/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_RING_ATTENTION_UPDATE_H_
#define ACLNN_RING_ATTENTION_UPDATE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRingAttentionUpdate的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] prevAttnOut:
 * 表示公式中的prev_attn_out。第一次FlashAttention的输出。输入shape和inputLayoutOptional属性保持一致。当输入数据排布inputLayoutOptional为TND时，D限制为64的倍数。
 * @param [in] prevSoftmaxMax:
 * 表示公式中的prev_softmax_max，第一次FlashAttention的softmax的max结果，输入shape为(B,N,S,8)或(T,N,8)，最后一维8个数字相同，且需要为正数。此处B为batch
 * size，N为head number，S为sequence length，T为time。
 * @param [in] prevSoftmaxSum:
 * 表示公式中的prev_softmax_sum，第一次FlashAttention的softmax的sum结果，输入shape和prevSoftmaxMax保持一致，最后一维8个数字相同，且需要为正数。
 * @param [in] curAttnOut:
 * 表示公式中的cur_attn_out，第二次FlashAttention的输出，数据类型和输入shape和prevAttnOut保持一致。当输入数据排布inputLayoutOptional为TND时，D限制为64的倍数。
 * @param [in] curSoftmaxMax:
 * 表示公式中的cur_softmax_max，第二次FlashAttention的softmax的max结果，输入shape和prevSoftmaxMax保持一致，最后一维8个数字相同，且需要为正数。
 * @param [in] curSoftmaxSum:
 * 表示公式中的cur_softmax_sum，第二次FlashAttention的softmax的sum结果，输入shape和prevSoftmaxMax保持一致，最后一维8个数字相同，且需要为正数。
 * @param [in] actualSeqQlenOptional: 表示从0开始的sequence
 * length的累加，数据类型支持INT64。当数据排布inputLayoutOptional为TND时，需要传入该参数，这是一个从0开始递增至T的整数aclTensor。
 * @param [in] inputLayoutOptional: 表示attn_out相关输入的数据排布。当前支持“TND”和“SBH”。	STRING	-
 * @param [out] attnOutOut:
 * 表示公式中的attn_out，通过两次结果更新后的输出，数据类型支持FLOAT16、FLOAT、BFLOAT16，数据类型和输出shape和prevAttnOut保持一致。
 * @param [out] softmaxMaxOut:
 * 表示公式中的softmax_max，通过两次结果更新后的softmax的max，数据类型支持FLOAT，输出shape和prevSoftmaxMax保持一致。
 * @param [out] softmaxSumOut:
 * 表示公式中的softmax_sum，通过两次结果更新后的softmax的sum，数据类型支持FLOAT，输出shape和prevSoftmaxMax保持一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
aclnnStatus aclnnRingAttentionUpdateGetWorkspaceSize(const aclTensor* prevAttnOut, const aclTensor* prevSoftmaxMax,
                                                     const aclTensor* prevSoftmaxSum, const aclTensor* curAttnOut,
                                                     const aclTensor* curSoftmaxMax, const aclTensor* curSoftmaxSum,
                                                     const aclTensor* actualSeqQlenOptional, char* inputLayoutOptional,
                                                     const aclTensor* attnOutOut, const aclTensor* softmaxMaxOut,
                                                     const aclTensor* softmaxSumOut, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor);

/**
 * @brief aclnnRingAttentionUpdate的第二段接口，用于执行计算。
 *
 * 算子功能：对输入的 attn 和 softmax， 计算 RingAttentionUpdate，形成新的 Tensor，并返回。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnRingAttentionUpdateGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
aclnnStatus aclnnRingAttentionUpdate(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream);

/**
 * @brief aclnnRingAttentionUpdateV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] prevAttnOut:
 * 表示公式中的prev_attn_out。第一次FlashAttention的输出。输入shape和inputLayoutOptional属性保持一致。当输入数据排布inputLayoutOptional为TND时，D限制为64的倍数。
 * @param [in] prevSoftmaxMax:
 * 表示公式中的prev_softmax_max，第一次FlashAttention的softmax的max结果，输入shape为(B,N,S,8)或(T,N,8)，最后一维8个数字相同，且需要为正数。此处B为batch
 * size，N为head number，S为sequence length，T为time。
 * @param [in] prevSoftmaxSum:
 * 表示公式中的prev_softmax_sum，第一次FlashAttention的softmax的sum结果，输入shape和prevSoftmaxMax保持一致，最后一维8个数字相同，且需要为正数。
 * @param [in] curAttnOut:
 * 表示公式中的cur_attn_out，第二次FlashAttention的输出，数据类型和输入shape和prevAttnOut保持一致。当输入数据排布inputLayoutOptional为TND时，D限制为64的倍数。
 * @param [in] curSoftmaxMax:
 * 表示公式中的cur_softmax_max，第二次FlashAttention的softmax的max结果，输入shape和prevSoftmaxMax保持一致，最后一维8个数字相同，且需要为正数。
 * @param [in] curSoftmaxSum:
 * 表示公式中的cur_softmax_sum，第二次FlashAttention的softmax的sum结果，输入shape和prevSoftmaxMax保持一致，最后一维8个数字相同，且需要为正数。
 * @param [in] actualSeqQlenOptional: 表示从0开始的sequence
 * length的累加，数据类型支持INT64。当数据排布inputLayoutOptional为TND时，需要传入该参数，这是一个从0开始递增至T的整数aclTensor。
 * @param [in] inputLayoutOptional: 表示attn_out相关输入的数据排布。当前支持“TND”和“SBH”。	STRING	-
 * @param [in] inputSoftmaxLayoutOptional: 表示softmax相关输入的数据排布。当前支持“TND”和“SBH”。	STRING	-
 * @param [out] attnOutOut:
 * 表示公式中的attn_out，通过两次结果更新后的输出，数据类型支持FLOAT16、FLOAT、BFLOAT16，数据类型和输出shape和prevAttnOut保持一致。
 * @param [out] softmaxMaxOut:
 * 表示公式中的softmax_max，通过两次结果更新后的softmax的max，数据类型支持FLOAT，输出shape和prevSoftmaxMax保持一致。
 * @param [out] softmaxSumOut:
 * 表示公式中的softmax_sum，通过两次结果更新后的softmax的sum，数据类型支持FLOAT，输出shape和prevSoftmaxMax保持一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
aclnnStatus aclnnRingAttentionUpdateV2GetWorkspaceSize(const aclTensor* prevAttnOut, const aclTensor* prevSoftmaxMax,
                                                       const aclTensor* prevSoftmaxSum, const aclTensor* curAttnOut,
                                                       const aclTensor* curSoftmaxMax, const aclTensor* curSoftmaxSum,
                                                       const aclTensor* actualSeqQlenOptional,
                                                       char* inputLayoutOptional, char* inputSoftmaxLayoutOptional,
                                                       const aclTensor* attnOutOut, const aclTensor* softmaxMaxOut,
                                                       const aclTensor* softmaxSumOut, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief aclnnRingAttentionUpdateV2的第二段接口，用于执行计算。
 *
 * 算子功能：对输入的 attn 和 softmax， 计算 RingAttentionUpdate，形成新的 Tensor，并返回。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnRingAttentionUpdateV2GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
aclnnStatus aclnnRingAttentionUpdateV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
