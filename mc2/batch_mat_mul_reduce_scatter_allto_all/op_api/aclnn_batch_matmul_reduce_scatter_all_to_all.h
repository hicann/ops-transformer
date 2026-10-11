/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_
#define OP_API_INC_BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现bmm + reduceScatter + alltoall 融合计算
 * @brief aclnnBatchMatMulReduceScatterAlltoAll的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x: 计算输入，Tensor，数据类型float16，bfloat16，必须为3维。BatchMatMul计算的左矩阵
 * @param [in] weight: 计算输入，Tensor，数据类型float16, bfloat16，必须为3维，类型与x保持一致。BatchMatMul计算的右矩阵
 * @param [in] biasOptional: 计算输入，Tensor，数据类型float16,
 * float32。x为float16时，bias需为float16；x为bfloat16时，bias需为float32。支持两维或三维。BatchMatMul计算的bias。(由于要进行ReduceScatter通信，因此需要在通信之后再Add)
 * @param [in] groupEp: 计算输入，str。ep通信域名称，专家并行的通信域
 * @param [in] groupTp: 计算输入，str。tp通信域名称，Tensor并行的通信域
 * @param [in] epWorldSize: 计算输入，int。ep通信域size，支持2/4/8/16/32
 * @param [in] tpWorldSize: 计算输入，int。tp通信域size，支持2/4/8/16/32
 * @param [in] yShardType: 计算输入，int，默认值为0。0表示输出在H维度按tp分片，1表示输出在C维度按tp分片。
 * @param [out] out: 计算输出，Tensor，数据类型float16, bfloat16，必须为3维。最终计算结果，类型与输入x保持一致
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 * 因为集合通信及BatchMatMul计算所需，输入输出shape需满足以下数学关系：（其中ep=epWorldSize，tp=tpWorldSize）
 * 按H轴进行ReduceScatter场景，即yShardType为0场景：
 * x: (E/ep, ep * C, M/tp)
 * weight：(E/ep, M/tp, H)
 * bias：(E/ep, 1, H/tp) 两维时为(E/ep, H/tp)
 * y：(E, C, H/tp)
 * 按C轴进行ReduceScatter场景，即yShardType为1场景：
 * x: (E/ep, ep * tp * C/tp, M/tp)
 * weight：(E/ep, M/tp, H)
 * bias：(E/ep, 1, H) 两维时为(E/ep, H)
 * y：(E, C/tp, H)
 * 数据关系说明：
 * 比如x.size(0)等于E/tp，y.size(0)等于E，则表示，y.size(0) = ep * x.size(0)，y.size(0)是ep的整数倍；其他关系类似
 * E的取值范围为[2, 512]，且E是ep的整数倍；
 * H的取值范围为：[1, 65535]，当yShardType为0时，H是tp的整数倍；
 * M/tp的取值范围为：[1, 65535]；
 * E/ep的取值范围为：[1, 32]；
 * ep、tp均仅支持2、4、8、16、32；
 * groupEp和groupTp名称不能相同；
 * C大于0，上限为算子device内存上限，当yShardType为1时，C是tp的整数倍；
 * 不支持跨超节点，只支持超节点内；
 */
ACLNN_API aclnnStatus aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(
    const aclTensor* x, const aclTensor* weight, const aclTensor* biasOptional, const char* groupEp,
    const char* groupTp, int64_t epWorldSize, int64_t tpWorldSize, int64_t yShardType, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnBatchMatMulReduceScatterAlltoAll的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnBatchMatMulReduceScatterAlltoAll(void* workspace, uint64_t workspaceSize,
                                                            aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_
