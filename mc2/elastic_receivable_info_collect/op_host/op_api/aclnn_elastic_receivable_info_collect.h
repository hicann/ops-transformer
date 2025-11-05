/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ELASTIC_RECEIVABLE_INFO_COLLECT_H_
#define OP_API_INC_ELASTIC_RECEIVABLE_INFO_COLLECT_H_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：获取test修改后的数据区结果，输出给主机端处理
 * @brief aclnnElasticReceivableInfoCollect的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] group: 计算输入，str。通信域名称，专家并行的通信域。
 * @param [in] worldSize: 计算输入，int64_t。通信域size。
 * @param [in] y: 计算输出，Tensor，必选输出，数据类型支持int32，仅支持2维，数据格式支持ND。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码。
 *
 */
ACLNN_API aclnnStatus aclnnElasticReceivableInfoCollectGetWorkspaceSize(const char* group, int64_t worldSize,
    aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnElasticReceivableInfoCollect的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnElasticReceivableInfoCollectGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnElasticReceivableInfoCollect
    (void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ELASTIC_RECEIVABLE_INFO_COLLECT_H_