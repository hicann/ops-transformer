/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_ACLNN_RECURRENT_KDA_H
#define OP_API_ACLNN_RECURRENT_KDA_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief RecurrentKda 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @param [in] query: 数据类型支持：bfloat16。
 * @param [in] key: 数据类型支持：bfloat16。
 * @param [in] value: 数据类型支持：bfloat16。
 * @param [in] gate: 数据类型支持：float32、bfloat16、float16。
 * @param [in] beta: 数据类型支持：float32、bfloat16、float16。
 * @param [in, out] initialStateRef: 数据类型支持：bfloat16、float32。
 * @param [in] cuSeqlensOptional: 可选参数，数据类型支持：int32、int64。
 * @param [in] ssmStateIndicesOptional: 可选参数，数据类型支持：int32、int64。
 * @param [in] aLogOptional: 可选参数，数据类型支持：float32。
 * @param [in] dtBiasOptional: 可选参数，数据类型支持：float32。
 * @param [in] numAcceptedTokensOptional: 可选参数，数据类型支持：int32、int64。
 * @param [in] layout: query、key、value的数据排布格式，支持BSND、TND。
 * @param [in] scale: query的缩放系数。
 * @param [in] outputFinalState: 是否输出有效的最终状态。
 * @param [in] inplaceFinalState: 是否将最终状态原地写回initialStateRef。
 * @param [in] useQkL2normInKernel: 是否在kernel内对query、key进行L2归一化。
 * @param [in] useGateInKernel: 是否在kernel内将gate作为raw gate处理。
 * @param [in] useBetaSigmoidInKernel: 是否在kernel内对beta进行sigmoid计算。
 * @param [in] allowNegEigval: beta经过sigmoid计算后是否乘2。
 * @param [in] safeGate: 是否使用raw gate的safe分支。
 * @param [in] lowerBound: safe gate的下界。
 * @param [in] stateVFirst: 状态矩阵是否采用V维在前的排布格式。
 * @param [out] attnOut: 数据类型支持：bfloat16。
 * @param [out] finalState: 数据类型支持：bfloat16、float32。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRecurrentKdaGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *gate, const aclTensor *beta,
    aclTensor *initialStateRef, const aclTensor *cuSeqlensOptional, const aclTensor *ssmStateIndicesOptional,
    const aclTensor *aLogOptional, const aclTensor *dtBiasOptional, const aclTensor *numAcceptedTokensOptional,
    const char *layout, double scale, bool outputFinalState, bool inplaceFinalState, bool useQkL2normInKernel,
    bool useGateInKernel, bool useBetaSigmoidInKernel, bool allowNegEigval, bool safeGate, double lowerBound,
    bool stateVFirst, const aclTensor *attnOut, const aclTensor *finalState, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief RecurrentKda 的第二段接口，用于执行算子计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口
 * aclnnRecurrentKdaGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRecurrentKda(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_RECURRENT_KDA_H
