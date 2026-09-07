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
 * \file aclnn_matmul_all_to_all.h
 * \brief
 */
#ifndef OP_API_INC_MATMUL_ALL_TO_ALL_
#define OP_API_INC_MATMUL_ALL_TO_ALL_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现all2all + matmul融合计算
 * @brief aclnnMatmulAlltoAll的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x1: 左矩阵输入张量。
 * @param [in] x2: 右矩阵输入张量。
 * @param [in] biasOptional: 可选输入张量，偏置项。
 * @param [in] alltoAllAxesOptional: all2all做数据交换的方向。
 * @param [in] group: 通信域标识，用于标识不同的通信组。
 * @param [in] transposeX1: 是否对x1转置。
 * @param [in] transposeX2: 是否对x2转置。
 * @param [out] output: 矩阵乘法的输出结果。
 * @param [out] workspaceSize: 用于存储计算所需的workspace大小。
 * @param [out] executor: 执行器指针，用于后续的计算执行
 *
 * @return aclnnStatus: 执行状态，返回0表示成功，其他值表示错误。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMatmulAlltoAllGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional, const aclIntArray *alltoAllAxesOptional,
    const char *group, bool transposeX1, bool transposeX2, const aclTensor *output, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnMatmulAlltoAll的第二段接口，用于执行计算。
 * @param [in] workspace: 在Device侧申请的workspace内存地址。
 * @param [in] workspacesize: 在Device侧申请的workspace大小，由第一段接口aclnnMatmulAlltoAllGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的Stream。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMatmulAlltoAll(void *workspace, uint64_t workspaceSize,
                                                                       aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MATMUL_ALL_TO_ALL_
