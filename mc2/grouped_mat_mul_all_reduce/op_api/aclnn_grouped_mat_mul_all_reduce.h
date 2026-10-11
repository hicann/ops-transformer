/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_grouped_mat_mul_all_reduce.h
 * \brief
 */
#ifndef OP_API_INC_GROUPED_MATMUL_ALL_REDUCE_H_
#define OP_API_INC_GROUPED_MATMUL_ALL_REDUCE_H_
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatMulAllReduce的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnnop_ops_infer
 * 算子功能：实现 gmm + AllReduce 融合计算
 * @param [in] x: 表示公式中的x，数据类型支持FLOAT16，BFLOAT16数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] weight:
 * 表示公式中的weight，数据类型支持FLOAT16，BFLOAT16数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] bias: 表示公式中的bias，数据类型支持FLOAT16，BFLOAT16数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [in] groupListOptional:
 * 可选参数，代表输入和输出M方向的matmul大小分布，数据类型支持INT64，支持的最大长度为128个。
 * @param [in] splitItemOptional:
 * 可选参数，代表输入和输出是否要做tensor切分，1代表输入需要切分，输出不需要切分；2代表输入不需要切分，
 * 输出需要切分；3代表输入和输出都需要切分。
 * @param [in] group: 可选参数，标识列组的字符串。
 * @param [in] reduceOp: 可选参数，reduce操作类型，默认值：sum。
 * @param [in] commTurn: 可选参数，通信数据切分数，即总数据量/单次通信量，默认值：0。
 * @param [in] streamMode: 可选参数，acl流模式的枚举，类型支持：0/1.
 * @param [out] out: 表示公式中的out，数据类型支持FLOAT16，BFLOAT16数据类型，数据格式支持ND，支持的最大长度为128个。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACL_DEPRECATED_MESSAGE("aclnnGroupedMatMulAllReduceGetWorkspaceSize will be deprecated")
ACLNN_API aclnnStatus aclnnGroupedMatMulAllReduceGetWorkspaceSize(
    const aclTensorList* x, const aclTensorList* weight, const aclTensorList* bias,
    const aclIntArray* groupListOptional, int64_t splitItem, const char* group, const char* reduceOp, int64_t commTurn,
    int64_t streamMode, const aclTensorList* y, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGroupedMatMulAllReduce的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnGtTensorGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACL_DEPRECATED_MESSAGE("aclnnGroupedMatMulAllReduce will be deprecated")
ACLNN_API aclnnStatus aclnnGroupedMatMulAllReduce(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif // OP_API_INC_GROUPED_MATMUL_ALL_REDUCE_H_
