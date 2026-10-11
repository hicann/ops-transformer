/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */
#ifndef OP_API_INC_QUANT_ALL_TO_ALLV_GROUPED_MATMUL_
#define OP_API_INC_QUANT_ALL_TO_ALLV_GROUPED_MATMUL_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现AlltoAllv + GroupedMatMul 融合计算
 * @brief 计算AlltoAllv + GroupedMatMul 融合计算所需的workspace大小。
 *
 * @param [in] gmmX: 计算输入，Tensor，数据类型支持hifloat8, float8_e4m3fn, float8_e5m2, float4_e2m1。
 * 该输入进行AllToAllv通信，仅支持二维, 数据格式支持ND，通信后结果作为GroupedMatMul计算的左矩阵
 * @param [in] gmmWeight: 计算输入，Tensor，数据类型支持hifloat8, float8_e4m3fn, float8_e5m2, float4_e2m1。
 * 仅支持三维, 数据格式支持ND，GroupedMatMul计算的右矩阵
 * @param [in] gmmXScale: 左矩阵的量化参数，数据类型支持float32, flaot8_e8m0。
 * @param [in] gmmWeightScale: 右矩阵的量化参数，数据类型为float32, flaot8_e8m0。
 * @param [in] sendCountsTensorOptional: 可选输入，计算输入，Tensor，数据类型支持int64，数据格式支持ND。
 * @param [in] recvCountsTensorOptional: 可选输入，计算输入，Tensor，数据类型支持int64，数据格式支持ND。
 * @param [in] mmXOptional: 可选输入，计算输入，Tensor，数据类型支持hifloat8, float8_e4m3fn, float8_e5m2,
 * float4_e2m1, 与gmmX的数据类型保持一致。仅支持二维, 数据格式支持ND，并行进行的共享专家matmul计算中的左矩阵。
 * @param [in] mmWeightOptional: 可选输入，计算输入，Tensor，数据类型支持hifloat8, float8_e4m3fn, float8_e5m2,
 * float4_e2m1, 与gmmWeight的数据类型保持一致。仅支持三维, 数据格式支持ND，并行进行的共享专家matmul计算中的右矩阵。
 * @param [in] mmXScaleOptional: 共享专家matmul计算中的左矩阵的量化参数，数据类型支持float32, flaot8_e8m0。
 * @param [in] mmWeightScaleOptional: 共享专家matmul计算中的右矩阵的量化参数，数据类型支持float32, flaot8_e8m0。
 * @param [in] gmmXQuantMode: 左矩阵的量化模式，支持以下模式：
 * - 0：无量化
 * - 1：PerTensor量化
 * - 2：PerChannel量化
 * - 3：PerToken量化
 * - 4：PerGroup量化
 * - 5：PerBlock量化
 * - 6：Mx Quant量化
 * 当前支持配置为1或6。
 * @param [in] gmmWeightQuantMode: 右矩阵的量化模式，同上，当前仅支持配置为1或6。
 * @param [in] mmXQuantMode: 共享专家matmul计算中的左矩阵的量化模式，同上，当前仅支持配置为1或6。
 * @param [in] mmWeightQuantMode: 共享专家matmul计算中的右矩阵的量化模式，同上，当前仅支持配置为1或6。
 * @param [in] group: 计算输入，str。ep通信域名称，专家并行的通信域。
 * @param [in] epWorldSize: 计算输入，int。ep通信域size。
 * @param [in] sendCounts: 计算输入，list int。通信发送的数据量。
 * @param [in] recvCounts: 计算输入，list int。通信接受的数据量。
 * @param [in] transGmmWeight: 可选输入，计算输入。表明gmm的右矩阵是否需要转置，默认为false。
 * @param [in] transMmWeight: 可选输入，计算输入。表明共享专家mm的右矩阵是否需要转置。
 * @param [in] groupSize: 可选输入，用于Matmul计算三个方向上的分量大小，预留参数，当前不支持。
 * @param [in] permuteOutFlag: 可选输入，计算输入。表明是否输出permute结果。默认为false，当前仅支持true。
 * @param [out] gmmY: 计算输出，Tensor，数据类型支持float16, bfloat16。最终计算结果，数据类型与输入gmmX保持一致。
 * @param [out] mmYOptional: 可选输出，计算输出，Tensor，数据类型支持float16，bfloat16，共享专家matmul的输出，
 * 仅当传入mmXOptional与mmWeightOptional才输出，数据类型与mmX保持一致。
 * @param [out] permuteOutOptional:
 * 计算输出，Tensor，数据类型支持float16，bfloat16。permute输出结果，数据类型与输入gmmX保持一致。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码。
 *
 * 因为集合通信及GroupedMatMul计算所需，输入输出shape需满足以下数学关系：（其中ep=epWorldSize）
 * gmmX: (BSK, H1);
 * gmmWeight: (e, H1, N1);
 * gmmXScale: pertensor场景(1,), Mx场景(BSK, CeilDiv(H1, 64), 2);
 * gmmWeightScale: pertensor场景(1,), Mx场景(e, CeilDiv(H1, 64), N1, 2);
 * mmXOptional: (BS, H2);
 * mmWeightOptional: (H2, N2);
 * mmXScaleOptional: pertensor场景(1,), Mx场景(BS, CeilDiv(H2, 64), 2);
 * mmWeightScaleOptional: pertensor场景(1,), Mx场景(CeilDiv(H2, 64), N2, 2);
 * gmmY: (A, N1);
 * mmYOptional: (BS, N2);
 * permuteOutOptional: (A, H1);
 *
 * 数据关系说明：
 * e表示单卡上的专家数量;
 * A = recvCounts的累加和;
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor* gmmX, const aclTensor* gmmWeight, const aclTensor* gmmXScale, const aclTensor* gmmWeightScale,
    const aclTensor* sendCountsTensorOptional, const aclTensor* recvCountsTensorOptional, const aclTensor* mmXOptional,
    const aclTensor* mmWeightOptional, const aclTensor* mmXScaleOptional, const aclTensor* mmWeightScaleOptional,
    int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
    const char* group, int64_t epWorldSize, const aclIntArray* sendCounts, const aclIntArray* recvCounts,
    bool transGmmWeight, bool transMmWeight, int64_t groupSize, bool permuteOutFlag, aclTensor* gmmY,
    aclTensor* mmYOptional, aclTensor* permuteOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor);

/* *
 * @brief aclnnAlltoAllvGroupedMatMul的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnAlltoAllvGroupedMatMulGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllvQuantGroupedMatMul(void* workspace,
                                                                                    uint64_t workspaceSize,
                                                                                    aclOpExecutor* executor,
                                                                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_QUANT_ALL_TO_ALLV_GROUPED_MATMUL_
