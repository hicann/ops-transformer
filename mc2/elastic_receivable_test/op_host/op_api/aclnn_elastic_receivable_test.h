/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ELASTIC_RECEIVABLE_TEST_H_
#define OP_API_INC_ELASTIC_RECEIVABLE_TEST_H_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：对一个通信域内的所有卡发送数据并写状态位，以检测通信链路是否正常。
 * @brief aclnnElasticReceivableTest的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] dstRank: 计算输入，表示同一个通信域内指定server内的通信卡，Device侧的aclTensor，要求为一个1D的Tensor，shape为(,rankNum)，数据类型支持INT32，数据格式要求为ND，支持非连续的Tensor。
 * @param [in] group: 计算输入，str。ep通信域名称，专家并行的通信域。
 * @param [in] worldSize: 计算输入，int。通信域大小。
 * @param [in] rank_num: 计算输入，int。需要发送对端server内的卡数。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码。
 *
 */
ACLNN_API aclnnStatus aclnnElasticReceivableTestGetWorkspaceSize(aclTensor* dstRank, const char* group, int64_t worldSize, int64_t rank_num,
                                                             uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnElasticReceivableTest的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnElasticReceivableTestGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnElasticReceivableTest(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ELASTIC_RECEIVABLE_TEST_H_