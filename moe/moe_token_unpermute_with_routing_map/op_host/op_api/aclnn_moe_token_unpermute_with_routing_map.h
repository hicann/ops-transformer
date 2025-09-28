/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/
 
#ifndef OP_API_INC_MOE_TOKEN_UNPERMUTE_WITH_ROTING_MAP_H_
#define OP_API_INC_MOE_TOKEN_UNPERMUTE_WITH_ROTING_MAP_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMoeTokenUnpermuteWithRoutingMap的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * @param [in] permutedTokens: 计算输入，Tensor，数据类型float16，bfloat16,float。输入的token数据。
 * @param [in] sortedIndices: 计算输入，Tensor，数据类型int32，仅支持1维，数据格式支持ND。
 * @param [in] routingMapOptional: 计算可选输入，Tensor，数据类型int8,bool，数据格式支持ND。代表token到expert的映射关系。
 * @param [in] probsOptional: 计算可选输入，Tensor，数据类型float16，bfloat16,float，必须为2维，数据格式支持ND。输入的prob数据。
 * @param [in] paddedMode: 计算可选输入，bool。 表示是否开启dropAndPad模式。
 * @param [out] unpermutedTokens: 计算输出，Tensor，必选输出，数据类型支持float16, bfloat16,float，仅支持2维，数据格式支持ND。
 * @param [out] outIndex: 计算输出，Tensor，必选输出，数据类型int32，仅支持1维，数据格式支持ND。
 * @param [out] permuteTokenId: 计算输出，Tensor，必选输出，数据类型int32，仅支持1维，数据格式支持ND。
 * @param [out] permuteProbs: 计算输出，Tensor，可选输出，数据类型float16, bfloat16,float，仅支持1维，数据格式支持ND,根据routingMap处理后的prob。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 *
 */
ACLNN_API aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize(const aclTensor* permutedTokens,
                                                             const aclTensor* sortedIndices,
                                                             const aclTensor* routingMapOptional,
                                                             const aclTensor* probsOptional, 
                                                             bool paddedMode,
                                                             const aclIntArray* restoreShapeOptional, 
                                                             aclTensor* unpermutedTokens, aclTensor* outIndex, aclTensor* permuteTokenId, aclTensor* permuteProbs,
                                                             uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief aclnnMoeTokenUnpermuteWithRoutingMap的第二段接口，根据具体的计算流程，计算workspace大小。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */

ACLNN_API aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMap(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                           const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif