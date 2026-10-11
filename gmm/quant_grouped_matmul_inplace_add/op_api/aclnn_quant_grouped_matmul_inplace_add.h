/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_QUANT_GROUPED_MATMUL_INPLACE_ADD_H
#define OP_API_INC_QUANT_GROUPED_MATMUL_INPLACE_ADD_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQuantGroupedMatmulInplaceAdd的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x1: 表示公式中的x1，数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8数据类型，数据格式支持ND。
 * @param [in] x2:
 * 表示公式中的x2，数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8数据类型，数据格式支持ND。
 * @param [in] scale1Optional：
 * 代表量化参数中的由输入x1量化引入的缩放因子，数据类型支持FLOAT32、FLOAT8_E8M0，数据格式支持ND。
 * @param [in] scale2: 表示量化参数，数据类型支持FLOAT32、FLOAT8_E8M0数据类型，数据格式支持ND。
 * @param [in] groupList: 代表输入和输出分组轴的matmul大小分布，数据类型支持INT64，数据格式支持ND。
 * @param [in/out] yRef: 表示公式中的y，数据类型支持FLOAT32数据类型，数据格式支持ND。
 * @param [in] groupListType:
 * 整数型参数，可取值0或1，0代表groupList中数值为分组轴大小的cumsum结果（累积和），
 * 1代表groupList中数值为分组轴上每组大小。
 * @param [in] groupSize:
 * 整数型参数，用于输入m、n、k方向上的量化分组大小。
 * groupSize输入由3个方向的groupSizeM，groupSizeN，groupSizeK三个值拼接组成，
 * 每个值占16位，共占用int64_t类型groupSize的低48位（groupSize中的高16位的数值无效），
 * 计算公式为：groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。
 * 暂时只支持0
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* scale1Optional, const aclTensor* scale2,
    const aclTensor* groupList, aclTensor* yRef, int64_t groupListType, int64_t groupSize, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnQuantGroupedMatmulInplaceAdd的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnQuantGroupedMatmulInplaceAdd(void* workspace,
                                                                                     uint64_t workspaceSize,
                                                                                     aclOpExecutor* executor,
                                                                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
