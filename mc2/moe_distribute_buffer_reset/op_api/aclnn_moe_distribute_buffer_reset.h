/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_moe_distribute_buffer_reset.h
 * \brief
 */
#ifndef OP_API_INC_MOE_DISTRIBUTE_BUFFER_RESET_
#define OP_API_INC_MOE_DISTRIBUTE_BUFFER_RESET_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现 aclnnMoeDistributeBufferReset 数据区和状态区的清理
 * @brief aclnnMoeDistributeBufferReset的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] elasticInfo: 有效rank：int32_t。
 * @param [in] groupEp: 计算输入，str。ep通信域名称。
 * @param [in] epWorldSize: 计算输入，int。ep通信域size
 * @param [in] needSync: 计算输入，int。标识是否需要进行全卡同步。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMoeDistributeBufferResetGetWorkspaceSize(const aclTensor* elasticInfo, const char* groupEp,
    int64_t epWorldSize, int64_t needSync, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMoeDistributeBufferReset的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAbsGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMoeDistributeBufferReset(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MOE_DISTRIBUTE_BUFFER_RESET_