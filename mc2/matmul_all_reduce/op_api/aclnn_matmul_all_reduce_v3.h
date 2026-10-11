/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MATMUL_ALL_REDUCE_V3_
#define OP_API_INC_MATMUL_ALL_REDUCE_V3_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMatmulAllReduceV3的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：实现mm+AllReduce融合计算
 * @param [in] x1: matmul左矩阵，数据类型支持：float16, bf16。
 * @param [in] x2: matmul右矩阵，数据类型支持：int8。
 * @param [in] bias: 偏置，数据类型支持：float16, bf16。
 * @param [in] x3: add计算参数，数据类型支持：float16, bf16。
 * @param [in] group: 标识列组的字符串。
 * @param [in] reduceOp: reduce操作类型，默认值：sum。
 * @param [in] commMode: 通信模式参数。
 * @param [in] commTurn: 通信数据切分数，即总数据量/单次通信量，默认值：0。
 * @param [in] streamMode: acl流模式的枚举，类型支持：0/1。
 * @param [out] output: 计算+通信的结果，数据类型：同输入。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMatmulAllReduceV3GetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                             const aclTensor* bias, const aclTensor* x3,
                                                             const char* group, const char* reduceOp,
                                                             const char* commMode, int64_t commTurn, int64_t streamMode,
                                                             const aclTensor* output, uint64_t* workspaceSize,
                                                             aclOpExecutor** executor);

/**
 * @brief aclnnMatmulAllReduceV3的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnMatmulAllReduceV3GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMatmulAllReduceV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MATMUL_ALL_REDUCE_V3_
