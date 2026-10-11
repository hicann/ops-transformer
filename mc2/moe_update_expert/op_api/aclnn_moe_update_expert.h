/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MOE_UPDATE_EXPERT_
#define OP_API_INC_MOE_UPDATE_EXPERT_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * 算子功能：实现MoeUpdateExpert功能。
 * @brief aclnnMoeUpdateExpert的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] expertIds: 计算输入，Tensor，数据类型int32，int64, 必须为2维，数据格式支持ND。
 * @param [in] eplbTable: 计算输入，Tensor，数据类型int32，必须为2维，数据格式支持ND。
 * @param [in] expertScalesOptional: 计算输入，Tensor，数据类型float16，bfloat16，float32，必须为2维，数据格式支持ND。
 * @param [in] pruningThresholdOptional: 计算输入，Tensor，数据类型float32，必须为1维或2维，数据格式支持ND。
 * @param [in] activeMaskOptional: 计算输入，Tensor，数据类型bool，必须为1维，数据格式支持ND。
 * @param [in] localRankId: 计算输入，int。ep通信域本卡Id。
 * @param [in] worldSize: 计算输入，int。ep通信域size。在昇腾910_93场景中取值支持8/16/32/64/128/144/256/288。
 * @param [in] balanceMode: 可选入参，int。均衡模式，当前默认为0，根据rank分发。
 * @param [out] balancedExpertIds: 计算输出，Tensor，必选输出，数据类型支持int32，int64，仅支持2维，数据格式支持ND。
 * @param [out] balancedActiveMask: 计算输出，Tensor，必选输出，数据类型支持bool，仅支持2维，数据格式支持ND。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 *
 */
ACLNN_API aclnnStatus aclnnMoeUpdateExpertGetWorkspaceSize(const aclTensor* expertIds, const aclTensor* eplbTable,
                                                           const aclTensor* expertScalesOptional,
                                                           const aclTensor* pruningThresholdOptional,
                                                           const aclTensor* activeMaskOptional, int64_t localRankId,
                                                           int64_t worldSize, int64_t balanceMode,
                                                           aclTensor* balancedExpertIds, aclTensor* balancedActiveMask,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMoeUpdateExpert的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnMoeUpdateExpertGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMoeUpdateExpert(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MOE_UPDATE_EXPERT_
