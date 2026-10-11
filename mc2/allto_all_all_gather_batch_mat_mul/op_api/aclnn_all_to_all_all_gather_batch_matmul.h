/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ALL_TO_ALL_ALL_GATHER_BATCH_MATMUL_
#define OP_API_INC_ALL_TO_ALL_ALL_GATHER_BATCH_MATMUL_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现alltoall + allGather + bmm 融合计算
 * @brief aclnnAlltoAllAllGatherBatchMatMul的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x:
 * 计算输入，Tensor，数据类型支持float16，bfloat16。该输入进行AllToAll、AllGather集合通信，必须为3维，数据格式支持ND，通信后结果作为BatchMatMul计算的左矩阵
 * @param [in] weight: 计算输入，Tensor，数据类型支持float16,
 * bfloat16，类型需与x保持一致，必须为3维，数据格式支持ND，BatchMatMul计算的右矩阵
 * @param [in] biasOptional: 计算输入，Tensor，数据类型支持float16,
 * float32。x为float16时，bias需为float16；x为bfloat16时，bias需为float32，必须为两维或三维，数据格式支持ND，BatchMatMul计算的bias
 * @param [in] groupEp: 计算输入，str。ep通信域名称，专家并行的通信域
 * @param [in] groupTp: 计算输入，str。tp通信域名称，Tensor并行的通信域
 * @param [in] epWorldSize: 计算输入，int。ep通信域size，支持2/4/8/16/32
 * @param [in] tpWorldSize: 计算输入，int。tp通信域size，支持2/4/8/16/32
 * @param [in] xShardType:
 * 计算输入，int，默认值为0，0表示在H维度按tp域进行allgather，1表示在C维度上按tp域进行allgather。
 * @param [in] actType: 计算输入，int，激活函数类型，默认值为0，表示无激活函数。支持传入0/1/2/3/4, 对应关系为[0: None,
 * 1: GELU, 2: Silu, 3: Relu, 4: FastGELU]等
 * @param [out] y1Out: 计算输出，Tensor，数据类型支持float16,
 * bfloat16，仅支持3维。最终计算结果，如果有激活函数则为激活函数的输出，否则为BatchMatMul的输出。数据类型与输入x保持一致
 * @param [out] y2OutOptional: 计算输出，Tensor，可选输出，数据类型支持float16,
 * bfloat16，仅支持3维。AllGather的输出，数据类型与输入x保持一致。反向可能需要仅allgather通信操作的结果，数据类型：同输入。
 * @param [out] y3OutOptional: 计算输出，Tensor，可选输出，数据类型支持float16,
 * bfloat16，仅支持3维。有激活函数时，BatchMatMul的输出，类型与输入x保持一致ctType非0时仅bmm计算的结果，数据类型：同输入。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码。
 *
 * 因为集合通信及BatchMatMul计算所需，输入输出shape需满足以下数学关系：（其中ep=epWorldSize，tp=tpWorldSize）
 * 按H轴进行AllGather场景，即xShardType为0场景：
 * x: (E, C, H/tp)；
 * weight：(E/ep, H, M/tp)；
 * bias：(E/ep, 1, M/tp)； 支持两维或三维，两维时shape为：(E/ep, M/tp)；
 * y1Out：(E/ep, ep * C, M/tp)；
 * y2OutOptional：(E/ep, ep * C, H)；
 * y3OutOptional：(E/ep, ep * C, M/tp)；
 * 按C轴进行AllGather场景，即xShardType为0场景：
 * x: (E, C/tp, H)；
 * weight：(E/ep, H, M/tp)；
 * bias：(E/ep, 1, M/tp)； 支持两维或三维，两维时shape为：(E/ep, M/tp)；
 * y1：(E/ep, ep * tp * C/tp, M/tp)；
 * y2：(E/ep, ep * tp * C/tp, H)；
 * y3：(E/ep, ep * tp * C/tp, M/tp)
 * 数据关系说明：
 * 比如x.size(0)等于E，weight.size(0)等于E/ep，则表示，x.size(0) = ep *
 * weight.size(0)，x.size(0)是ep的整数倍；其他关系类似 E的取值范围为[2, 512]，且E是ep的整数倍； H的取值范围为：[1,
 * 65535]，当xShardType为0时，H是tp的整数倍； M/tp的取值范围为：[1, 65535]； E/ep的取值范围为：[1, 32]；
 * ep、tp均仅支持2、4、8、16、32；
 * groupEp和groupTp名称不能相同；
 * C大于0，上限为算子device内存上限，当xShardType为1时，C是tp的整数倍；
 * 不支持跨超节点，只支持超节点内。
 */
ACLNN_API aclnnStatus aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize(
    const aclTensor* x, const aclTensor* weight, const aclTensor* biasOptional, const char* groupEp,
    const char* groupTp, int64_t epWorldSize, int64_t tpWorldSize, int64_t xShardType, int64_t actType,
    aclTensor* y1Out, aclTensor* y2OutOptional, aclTensor* y3OutOptional, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnAlltoAllAllGatherBatchMatMul的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnAlltoAllAllGatherBatchMatMul(void* workspace, uint64_t workspaceSize,
                                                        aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ALL_TO_ALL_ALL_GATHER_BATCH_MATMUL_
