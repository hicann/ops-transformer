/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_DISTRIBUTE_BARRIER_V2_H_
#define OP_API_INC_DISTRIBUTE_BARRIER_V2_H_

#include <string>

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现全卡同步
 * @brief
 * aclnnDistributeBarrierV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] xRef: 计算输入，占位未使用，Tensor，支持bfloat16, float16,
 * float32, bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64。
 * @param [in] timeOut: 计算输入，超时时间，Tensor，int32。
 * @param [in] elasticInfo: 计算输入，动态缩容表，Tensor，int32。
 * @param [in] group: 计算输入，str。通信域名称，专家并行的通信域。
 * @param [in] worldSize: 计算输入，int64_t。通信域size。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码。
 *
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDistributeBarrierV2GetWorkspaceSize(
    const aclTensor* xRef, const aclTensor* timeOut,
    const aclTensor* elasticInfo, const char* group,
    int64_t worldSize, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnDistributeBarrierV2的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnDistributeBarrierGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDistributeBarrierV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_DISTRIBUTE_BARRIER_V2_H