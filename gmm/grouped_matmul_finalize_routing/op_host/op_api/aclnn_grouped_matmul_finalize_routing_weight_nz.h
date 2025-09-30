/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_NZ
#define OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_NZ

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatmulFinalizeRoutingWeightNz的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：实现GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作, 当前仅支持w为nz格式。
 * @param [in] x1: matmul左矩阵，数据类型支持：int8。
 * @param [in] x2: matmul右矩阵，数据类型支持：int8。
 * @param [in] scale: 量化参数中的缩放因子，数据类型支持：float32。
 * @param [in] bias: 偏置，数据类型支持：float32。
 * @param [in] pertokenScaleOptional: 反量化参数，数据类型支持：float32。
 * @param [in] groupList: 代表输入和输出分组轴方向的matmul大小分布，数据类型支持：int64。
 * @param [in] sharedInput: 
 * moe计算中共享专家的输出，需要与moe专家的输出进行combine操作，数据类型支持：bfloat16、float16。
 * @param [in] logit: 
 * moe专家对各个token的logit大小，矩阵乘的计算输出与该logit做乘法，然后索引进行combine，数据类型支持：float32。
 * @param [in] rowIndex: 
 * moe专家输出按照该rowIndex进行combine，其中的值即为combine做scatter add的索引，数据类型支持：int64、int32。
 * @param [in] sharedInputWeight: 
 * 共享专家与moe专家进行combine的系数，shareInput先于该参数乘，然后在和moe专家结果累加，数据类型支持：float32。
 * @param [in] sharedInputOffset: 共享专家输出的在总输出中的偏移，数据类型支持：int64。
 * @param [in] transposeX1: 左矩阵是否转置，默认值：false。
 * @param [in] transposeX2: 右矩阵是否转置，默认值：false。
 * @param [in] groupListType: GroupedMatmul分组类型，默认值：1，count模式，数据类型支持：int64。。
 * @param [out] out: 计算结果，数据类型：float32，float16, bfloat16。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNzGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *scale,
                                                     const aclTensor* bias, const aclTensor *pertokenScaleOptional, 
                                                     const aclTensor *groupList, const aclTensor *sharedInput, const aclTensor* logit,
                                                     const aclTensor *rowIndex, int64_t dtype, float sharedInputWeight,  
                                                     int64_t sharedInputOffset, bool transposeX1, bool transposeX2, int64_t groupListType, aclTensor *out,
                                                     uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnGroupedMatmulFinalizeRoutingWeightNz的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulFinalizeRoutingNZGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_MATMUL_NZ