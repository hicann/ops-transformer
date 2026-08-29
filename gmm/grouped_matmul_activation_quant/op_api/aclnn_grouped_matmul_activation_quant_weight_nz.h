/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GROUPED_MATMUL_ACTIVATION_QUANT_WEIGHT_NZ_H_
#define OP_API_INC_GROUPED_MATMUL_ACTIVATION_QUANT_WEIGHT_NZ_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatmulActivationQuantWeightNz的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x:
 * 表示公式中的x，当前MX场景下数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1、
 * FLOAT4_E1M2，数据格式支持ND。
 * @param [in] groupList:
 * 表示分组信息，数据类型支持INT64，数据格式支持ND。groupListType为0时，表示每个group在M轴上尾部索引的累积和；
 * groupListType为1时，表示每个group在M轴上的大小。
 * @param [in] weight:
 * 表示公式中的weight，tensorList长度当前仅支持1。调用者必须传入FRACTAL_NZ格式的weight，
 * 当前MX场景下数据类型支持FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2，viewShape要求为3维，
 * storageShape要求为5维。
 * @param [in] weightScale:
 * 表示weight矩阵的量化因子，tensorList长度当前仅支持1。当前MX场景下数据类型支持FLOAT8_E8M0，
 * 数据格式支持ND。
 * @param [in] bias:
 * 表示偏置。当前MX场景下必须为空，支持nullptr、空tensorList或长度为1且元素shape为(0)的空tensorList。
 * @param [in] xScaleOptional:
 * 表示x矩阵的量化因子。当前MX场景下必选，不能传nullptr，数据类型支持FLOAT8_E8M0，数据格式支持ND。
 * @param [in] activationType:
 * 表示激活函数类型，当前仅支持"gelu_tanh"。
 * @param [in] groupListType:
 * 整数型参数，表示groupList的解释方式，当前支持0或1。
 * @param [in] tuningConfig:
 * 调优参数，预留参数，当前暂不使用，支持传入nullptr。
 * @param [in] quantMode:
 * 表示量化模式，当前仅支持"mx"；传入nullptr或空字符串时根据x和xScaleOptional的数据类型推导为"mx"。
 * @param [in] roundMode:
 * 表示量化舍入模式，当前仅支持"rint"；传入nullptr或空字符串时按"rint"处理。
 * @param [in] scaleAlg:
 * 表示量化因子计算算法，支持0、1和2。0表示OCP实现；1表示cuBLAS实现且仅支持FLOAT8输出；
 * 2表示FLOAT4动态范围实现且仅支持FLOAT4_E2M1输出。FLOAT4_E1M2输出仅支持0。
 * @param [in] dstTypeMax:
 * 表示maxType的取值，对应公式中的Amax(DType)。scaleAlg为2时支持0.0或[6.0, 12.0]；
 * 其他scaleAlg仅支持0.0。
 * @param [out] y:
 * 表示激活并量化后的输出矩阵，数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2；MXFP4输入时还支持
 * FLOAT4_E2M1、FLOAT4_E1M2，数据格式支持ND。
 * @param [out] yScale:
 * 表示输出量化因子，数据类型支持FLOAT8_E8M0，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
    const aclTensor *x, const aclTensor *groupList, const aclTensorList *weight, const aclTensorList *weightScale,
    const aclTensorList *bias, const aclTensor *xScaleOptional, const char *activationType, int64_t groupListType,
    const aclIntArray *tuningConfig, const char *quantMode, const char *roundMode, int64_t scaleAlg, float dstTypeMax,
    aclTensor *y, aclTensor *yScale, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnGroupedMatmulActivationQuantWeightNz的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize:
 * 在npu device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulActivationQuantWeightNz(void *workspace, uint64_t workspaceSize,
                                                                aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_GROUPED_MATMUL_ACTIVATION_QUANT_WEIGHT_NZ_H_
