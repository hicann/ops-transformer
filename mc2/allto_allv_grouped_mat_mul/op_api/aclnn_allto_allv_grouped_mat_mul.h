/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ALL_TO_ALLV_GROUPED_MATMUL_
#define OP_API_INC_ALL_TO_ALLV_GROUPED_MATMUL_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现alltoallv + grouped matmul 融合计算
 * @brief aclnnAlltoAllvGroupedMatMul的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] gmmX: 计算输入，Tensor，数据类型支持float16，bfloat16。该输入进行AllToAll通信，仅支持二维,
 * 数据格式支持ND，通信后结果作为GroupedMatMul计算的左矩阵
 * @param [in] gmmWeight: 计算输入，Tensor，数据类型支持float16, bfloat16，类型需与x保持一致，仅支持三维,
 * 数据格式支持ND，GroupedMatMul计算的右矩阵
 * @param [in] sendCountsTensorOptional: 可选入参，计算输入，Tensor，数据类型支持int32, int64，数据格式支持ND
 * @param [in] recvCountsTensorOptional: 可选入参，计算输入，Tensor，数据类型支持int32, int64，数据格式支持ND
 * @param [in] mmXOptional: 可选入参，计算输入，并行进行的共享专家matmul计算中的左矩阵。
 * @param [in] mmWeightOptional: 可选入参，计算输入，并行进行的共享专家matmul计算中的右矩阵。
 * @param [in] group: 计算输入，str。ep通信域名称，专家并行的通信域
 * @param [in] epWorldSize: 计算输入，int。ep通信域size
 * @param [in] sendCounts: 计算输入，list int。通信发送的数据量
 * @param [in] recvCounts: 计算输入，list int。通信接受的数据量
 * @param [in] transGmmWeight: 可选入参，计算输入。表明gmm的右矩阵是否需要转置，默认为false
 * @param [in] transMmWeight: 可选入参，计算输入。表明共享专家mm的右矩阵是否需要转置。
 * @param [in] permuteOutFlag: 可选入参，计算输入。表明是否输出permute结果。默认为false，当前仅支持true
 * @param [out] gmmY: 计算输出，Tensor，数据类型支持float16, bfloat16。最终计算结果，数据类型与输入gmmX保持一致
 * @param [out] mmYOptional: 可选输出，计算输出，Tensor，数据类型支持float16,
 * bfloat16，共享专家matmul的输出，仅当传入mmXOptional与mmWeightOptional才输出，数据类型与mmX保持一致。
 * @param [out] permuteOutOptional: 计算输出，Tensor，数据类型支持float16,
 * bfloat16。permute输出结果，数据类型与输入gmmX保持一致
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码。
 *
 * 因为集合通信及BatchMatMul计算所需，输入输出shape需满足以下数学关系：（其中ep=epWorldSize）
 * gmmX: (BSK, H);
 * gmmWeight: (e, H, N1);
 * groupList: (e);
 * mmXOptional: (BS, H);
 * mmWeightOptional: (H, N2);
 * gmmY: (A, N1);
 * mmYOptional: (BS, N2);
 * permuteOutOptional: (A, H);
 *
 * 数据关系说明：
 * e表示单卡上的专家数量;
 * A = recvCounts的累加和;
 */
ACLNN_API aclnnStatus aclnnAlltoAllvGroupedMatMulGetWorkspaceSize(
    const aclTensor* gmmX, const aclTensor* gmmWeight, const aclTensor* sendCountsTensorOptional,
    const aclTensor* recvCountsTensorOptional, const aclTensor* mmXOptional, const aclTensor* mmWeightOptional,
    const char* group, int64_t epWorldSize, const aclIntArray* sendCounts, const aclIntArray* recvCounts,
    bool transGmmWeight, bool transMmWeight, bool permuteOutFlag, aclTensor* gmmY, aclTensor* mmYOptional,
    aclTensor* permuteOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAlltoAllvGroupedMatMul的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnAlltoAllvGroupedMatMulGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnAlltoAllvGroupedMatMul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ALL_TO_ALLV_GROUPED_MATMUL_
