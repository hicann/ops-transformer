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
 * \file aclnn_allto_all_quant_matmul.h
 * \brief
 */
#ifndef OP_API_INC_ALL_TO_ALL_QUANT_MATMUL_
#define OP_API_INC_ALL_TO_ALL_QUANT_MATMUL_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现all2all + quant + matmul融合计算
 * @brief aclnnAlltoAllQuantMatMul的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] x1: 左矩阵输入张量，数据类型支持FLOAT16、BFLOAT16、INT4。
 * @param [in] x2: 右矩阵输入张量，数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2、INT8、INT4。
 * @param [in] biasOptional: 可选输入张量，偏置项，仅在传入非空时生效，数据类型为FLOAT16、BFLOAT16、FLOAT32。
 * @param [in] x1ScaleOptional: 可选输入张量，左矩阵的量化参数，数据类型为FLOAT32。
 * @param [in] x2Scale: 右矩阵的反量化参数，数据类型为FLOAT32。
 * @param [in] commScaleOptional: 可选输入张量，低比特通信的量化系数，暂不支持。
 * @param [in] x1OffsetOptional: 可选输入，左矩阵的量化偏置，暂不支持。
 * @param [in] x2OffsetOptional: 可选输入，右矩阵的量化偏置，暂不支持。
 * @param [in] group: 通信域标识，用于标识不同的通信组。
 * @param [in] alltoAllAxesOptional: all2all做数据交换的方向，支持配置空或[-2,-1]，传入空时默认按[-2,-1]处理。
 * @param [in] x1QuantMode: Matmul左矩阵的量化方式，支持以下模式：
 *        - 0：无量化
 *        - 1：PerTensor量化
 *        - 2：PerChannel量化
 *        - 3：PerToken量化
 *        - 4：PerGroup量化
 *        - 5：PerBlock量化
 *        - 6：Mx Quant量化
 *        - 7：PerToken动态量化
 *        当前仅支持配置为3或者7。
 * @param [in] x2QuantMode: Matmul右矩阵的量化方式，同上，当前仅支持配置为2。
 * @param [in] commQuantMode: 低比特通信的量化方式，预留参数，当前仅支持配置为0，表示不量化。
 * @param [in] commQuantDtype: 低比特通信的量化类型，预留参数，当前仅支持配置为-1，表示ACL_DT_UNDEFINED。
 * @param [in] x1QuantDtype:
 * Matmul左矩阵的量化类型，在A5中仅支持配置35（表示ACL_FLOAT8_E5M2）或36（表示ACL_FLOAT8_E4M3FN），在A2中仅支持配置2（表示ACL_INT8）。
 * @param [in] groupSize: Matmul计算三个方向上的量化分组大小。
 * @param [in] transposeX1: 左矩阵是否转置过，暂不支持配置为True。
 * @param [in] transposeX2: 右矩阵是否转置过。
 * @param [out] output: 矩阵乘法的输出结果。
 * @param [out] alltoAllOutOptional: all-to-all通信的输出结果。
 * @param [out] workspaceSize: 用于存储计算所需的workspace大小。
 * @param [out] executor: 执行器指针，用于后续的计算执行
 *
 * @return aclnnStatus: 执行状态，返回0表示成功，其他值表示错误。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllQuantMatmulGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional, const aclTensor *x1ScaleOptional,
    const aclTensor *x2Scale, const aclTensor *commScaleOptional, const aclTensor *x1OffsetOptional,
    const aclTensor *x2OffsetOptional, const char *group, const aclIntArray *alltoAllAxesOptional, int64_t x1QuantMode,
    int64_t x2QuantMode, int64_t commQuantMode, int64_t commQuantDtype, int64_t x1QuantDtype, int64_t groupSize,
    bool transposeX1, bool transposeX2, const aclTensor *output, const aclTensor *alltoAllOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnAlltoAllQuantMatMul的第二段接口，用于执行计算。
 * @param [in] workspace: 在Device侧申请的workspace内存地址。
 * @param [in] workspacesize: 在Device侧申请的workspace大小，由第一段接口aclnnAlltoAllQuantMatMulGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的Stream。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllQuantMatmul(void *workspace, uint64_t workspaceSize,
                                                                            aclOpExecutor *executor,
                                                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ALL_TO_ALL_QUANT_MATMUL_
