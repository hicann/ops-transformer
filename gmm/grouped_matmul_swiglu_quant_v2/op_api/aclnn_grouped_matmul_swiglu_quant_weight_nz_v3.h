/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either express or implied,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_V3_H
#define OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_V3_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatmulSwigluQuantWeightNzV3 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 当前仅支持 Ascend 950 的单 Tensor MXFP8 weight FRACTAL_NZ 场景。
 *
 * @param [in] x: 激活矩阵，FLOAT8_E4M3FN、ND。
 * @param [in] weight: 权重 TensorList，FLOAT8_E4M3FN、FRACTAL_NZ。
 * @param [in] weightScale: 权重量化因子 TensorList，FLOAT8_E8M0、ND。
 * @param [in] weightAssistMatrix: 权重辅助矩阵，当前仅支持空值。
 * @param [in] bias: 偏置，当前不支持。
 * @param [in] xScale: 激活矩阵量化因子，FLOAT8_E8M0、ND。
 * @param [in] smoothScale: 平滑缩放因子，当前不支持。
 * @param [in] groupList: 分组索引，INT64、ND。
 * @param [in] dequantMode: 反量化模式，当前仅支持 2。
 * @param [in] dequantDtype: GroupedMatmul 中间结果的数据类型，当前仅支持 DT_FLOAT。
 * @param [in] quantMode: 输出量化模式，当前仅支持 2。
 * @param [in] groupListType: groupList 的解释方式；0 表示 cumsum，1 表示 count。
 * @param [in] tuningConfigOptional: 调优参数，当前仅支持空值。
 * @param [in] swigluMode: SwiGLU 模式，当前仅支持 2。
 * @param [in] clampLimit: SwiGLU clamp 上界。
 * @param [in] gluAlpha: SwiGLU 的 alpha 参数。
 * @param [in] gluBias: SwiGLU 的 bias 参数。
 * @param [in] roundMode: MX 量化舍入模式，当前仅支持 "rint"。
 * @param [in] scaleAlg: MX 量化 scale 算法，支持 0（OCP）和 1（cuBLAS）。
 * @param [in] dstTypeMax: MX 量化目标类型最大值，当前仅支持 0.0。
 * @param [out] output: 量化结果，FLOAT8_E4M3FN、ND。
 * @param [out] outputScale: 输出量化因子，FLOAT8_E8M0、ND。
 * @param [out] workspaceSize: 在 NPU device 侧申请的 workspace 大小。
 * @param [out] executor: 算子执行器，包含计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale,
    const aclTensorList* weightAssistMatrix, const aclTensor* bias, const aclTensor* xScale,
    const aclTensor* smoothScale, const aclTensor* groupList, int64_t dequantMode, int64_t dequantDtype,
    int64_t quantMode, int64_t groupListType, const aclIntArray* tuningConfigOptional, int64_t swigluMode,
    double clampLimit, double gluAlpha, double gluBias, const char* roundMode, int64_t scaleAlg, double dstTypeMax,
    aclTensor* output, aclTensor* outputScale, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGroupedMatmulSwigluQuantWeightNzV3 的第二段接口，用于执行计算。
 * @param [in] workspace: 在 NPU device 侧申请的 workspace 内存起址。
 * @param [in] workspaceSize: workspace 大小，由第一段接口获取。
 * @param [in] executor: 算子执行器，包含计算流程。
 * @param [in] stream: acl stream 流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3(void* workspace, uint64_t workspaceSize,
                                                              aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
