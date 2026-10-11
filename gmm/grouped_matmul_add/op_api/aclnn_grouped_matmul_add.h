/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_ADD_H
#define OP_API_INC_GROUPED_MATMUL_ADD_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatmulAdd的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: 表示公式中的x，数据类型支持FLOAT16、BFLOAT16数据类型，数据格式支持ND。
 * @param [in] weight:
 * 表示公式中的weight，数据类型支持FLOAT16、BFLOAT16数据类型，数据格式支持ND。
 * @param [in] groupList: 代表输入和输出分组轴的matmul大小分布，数据类型支持INT64，数据格式支持ND。
 * @param [in/out] yRef: 表示公式中的y，数据类型支持FLOAT32数据类型，数据格式支持ND。
 * @param [in] transposeX:
 * bool类型，仅可取值true，表示x矩阵转置。
 * @param [in] transposeWeight:
 * bool类型，仅可取值false，表示weight矩阵不转置。
 * @param [in] groupType:
 * 整数型参数，代表需要切分的轴，仅可取值2，表示切分k轴，
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulAddGetWorkspaceSize(
    const aclTensor* x, const aclTensor* weight, const aclTensor* groupList, aclTensor* yRef, bool transposeX,
    bool transposeWeight, int64_t groupType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGroupedMatmulAdd的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulAddGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulAdd(void* workspace, uint64_t workspaceSize,
                                                                         aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
