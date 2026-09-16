/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_QUANT_GROUPED_MAT_MUL_ALL_TO_ALLV_H_
#define OP_API_INC_QUANT_GROUPED_MAT_MUL_ALL_TO_ALLV_H_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现quant grouped matmul + alltoallv 融合计算
 * @brief aclnnGroupedMatMulAlltoAllv的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] gmmX: 计算输入，Tensor。该输入进行AllToAll通信，仅支持二维,
 * 数据格式支持ND，通信后结果作为GroupedMatMul计算的左矩阵    // GroupedMatMul计算的左矩阵输入
 * @param [in] gmmWeight: 计算输入，Tensor，类型需与gmmX保持一致，仅支持三维,
 * 数据格式支持ND，GroupedMatMul计算的右矩阵
 * @param [in] gmmXScale: 左矩阵的量化参数，数据类型为FLOAT32。
 * @param [in] gmmWeightScale: 右矩阵的量化参数，数据类型为FLOAT32。
 * @param [in] sendCountsTensorOptional: 可选入参，计算输入，Tensor，数据类型支持int32,
 * int64，shape为(e*epWorldSize)，数据格式支持ND，当前版本不支持，传入nullptr。
 * @param [in] recvCountsTensorOptional: 可选入参，计算输入，Tensor，数据类型支持int32,
 * int64，shape为(e*epWorldSize)，数据格式支持ND，当前版本不支持，传入nullptr。
 * @param [in] mmXOptional:
 * 可选入参，计算输入，并行进行的共享专家matmul计算中的左矩阵，需与mmWeightOptional同时传入或同为nullptr，数据类型与gmmX保持一致，支持2维，shape为(BS,H2)。
 * @param [in] mmWeightOptional:
 * 可选入参，计算输入，并行进行的共享专家matmul计算中的右矩阵，需与mmXOptional同时传入或同为nullptr，数据类型与gmmX保持一致，支持2维，shape为(H2,N2)。
 * @param [in] mmXScaleOptional: 可选入参，共享专家matmul计算中左矩阵的量化参数，数据类型为FLOAT32。
 * @param [in] mmWeightScaleOptional: 可选入参，共享专家matmul计算中的右矩阵中的量化参数，数据类型为FLOAT32。
 * @param [in] commQuantScaleOptional: 可选输入，低比特通信的量化参数，暂不支持。
 * @param [in] gmmXQuantMode: 左矩阵的量化模式，支持以下模式：
 *         - 0：无量化
 *         - 1：PerTensor量化
 *         - 2：PerChannel量化
 *         - 3：PerToken量化
 *         - 4：PerGroup量化
 *         - 5：PerBlock量化
 *         - 6：Mx Quant量化
 *         - 7: Dyn PerToken量化
 *         当前仅支持配置为1，先完成pertensor-pertensor量化功能打通。
 * @param [in] gmmWeightQuantMode: 右矩阵的量化模式，同上，当前配置为1。
 * @param [in] mmXQuantMode: 共享专家matmul计算中的左矩阵的量化模式，同上，当前仅支持配置为1。
 * @param [in] mmWeightQuantMode: 共享专家matmul计算中的右矩阵的量化模式，同上，当前仅支持配置为1。
 * @param [in] commQuantMode: 预留，低比特通信的量化模式，当前仅支持0，表示不支持低比特通信。
 * @param [in] commQuantDtypeOptional: 可选输入，低比特通信量化后的数据类型，当前不支持。
 * @param [in] group: 计算输入，str。ep通信域名称，专家并行的通信域，字符串长度要求(0,128)。
 * @param [in] epWorldSize: 计算输入，int。ep通信域size，支持4、8、16、32、64。
 * @param [in] sendCounts:
 * 计算输入，表示发送给其他卡的token数，数据类型支持INT64，取值大小为e*epWorldSize，最大为256。输入类型需为list。
 * @param [in] recvCounts:
 * 计算输入，表示接收其他卡的token数，数据类型支持INT64，取值大小为e*epWorldSize，最大为256。输入类型需为list。
 * @param [in] transGmmWeight: 可选入参，计算输入。表明gmm的右矩阵是否需要转置，默认为false表示不转置。
 * @param [in] transMmWeight: 可选入参，计算输入。表明共享专家mm的右矩阵是否需要转置，默认为false表示不转置。
 * @param [out] y: 计算输出，最终计算结果，支持2维，shape为(BSK,N1)。
 * @param [out] mmYOptional:
 * 可选输出，共享专家计算输出，数据类型与mmXOptional保持一致，支持2维，shape为(BS,N2)，仅当传入mmXOptional与mmWeightOptional才输出。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码。
 *
 * 因为集合通信及BatchMatMul计算所需，输入输出shape需满足以下数学关系：（其中ep=epWorldSize）
 * gmmX: (A, H1);
 * gmmWeight: (e, H1, N1);
 * gmmXScale: pertensor场景(1,);
 * gmmWeightScale: pertensor场景(1,)
 * mmXOptional: (BS, H2);
 * mmWeightOptional: (H2, N2);
 * mmXScaleOptional: pertensor场景(1,);
 * mmWeightScaleOptional: pertensor场景(1,);
 * y: (BSK, N1);
 * mmYOptional: (BS, N2);
 *
 * 数据关系说明：
 * e表示单卡上的专家数量;
 * A = sendCounts的累加和;
 * K = topK;
 */
ACLNN_API aclnnStatus aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const aclTensor *commQuantScaleOptional, int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode,
    int64_t mmWeightQuantMode, int64_t commQuantMode, int64_t commQuantDtypeOptional, int64_t groupSize,
    const char *group, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    bool transGmmWeight, bool transMmWeight, const aclTensor *y, const aclTensor *mmYOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnQuantGroupedMatMulAlltoAllv的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnQuantGroupedMatMulAlltoAllv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_QUANT_GROUPED_MAT_MUL_ALL_TO_ALLV_H_
