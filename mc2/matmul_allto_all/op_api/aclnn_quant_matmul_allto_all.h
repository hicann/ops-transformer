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
 * \file aclnn_quant_matmul_allto_all.h
 * \brief
 */
#ifndef OP_API_INC_QUANT_MATMUL_ALL_TO_ALL_
#define OP_API_INC_QUANT_MATMUL_ALL_TO_ALL_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现matmul + alltoall融合计算
 * @brief 计算全连接（All-to-All）矩阵乘法所需的workspace大小。
 *
 * 该接口用于计算分布式训练中通信和计算所需的workspace大小。支持多种数据类型和量化模式。
 *
 * @param [in] x1: 左矩阵输入张量，对应公式中的x1，数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2、INT8。
 * @param [in] x2: 右矩阵输入张量，对应公式中的x2，数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2、INT8。
 * @param [in] biasOptional: 可选输入张量，偏置项，仅在传入非空时生效，数据类型为FLOAT32、FLAOT16、BFLOAT16。
 * @param [in] x1Scale: 左矩阵的量化参数，对应公式中的x1Scale，数据类型为FLOAT32。
 * @param [in] x2Scale: 右矩阵的量化参数，对应公式中的x2Scale，数据类型为FLOAT32。
 * @param [in] commScaleOptional: 可选输入，低比特通信的量化系数，暂不支持。
 * @param [in] x1OffsetOptional: 可选输入，左矩阵的量化偏置，暂不支持。
 * @param [in] x2OffsetOptional: 可选输入，右矩阵的量化偏置，暂不支持。
 * @param [in] group: 通信域名，字符串长度要求（0，128）。
 * @param [in] alltoAllAxesOptional:
 * 可选输入，AlltoAll和Permute数据交换的方向，支持配置空或[-2,-1]，传入空时默认按[-2,-1]处理。
 * @param [in] x1QuantMode: 左矩阵的量化模式，支持以下模式：
 *        - 0：无量化
 *        - 1：PerTensor量化
 *        - 2：PerChannel量化
 *        - 3：PerToken量化
 *        - 4：PerGroup量化
 *        - 5：PerBlock量化
 *        - 6：Mx Quant量化
 *        当前仅支持配置为3，表示PerToken。
 * @param [in] x2QuantMode: 右矩阵的量化模式，同上，当前仅支持配置为2，表示PerChannel。
 * @param [in] commQuantMode: 低比特通信的量化模式，预留参数，当前仅支持配置为0，表示不量化。
 * @param [in] commQuantDtype: 低比特通信的量化类型，预留参数，当前仅支持配置为-1，表示ACL_DT_UNDEFINED。
 * @param [in] groupSize: 用于Matmul计算三个方向上的量化分组大小，预留参数，T-C量化模式下仅支持配置为0，取值不生效。
 * @param [in] transposeX1: 标识左矩阵是否转置过，配置为True时左矩阵Shape为（H1，BS），暂不支持配置为True。
 * @param [in] transposeX2: 标识右矩阵是否转置过，配置为True时右矩阵Shape为（H2，H1）。
 * @param [out] output: 矩阵乘法的输出结果。
 * @param [out] workspaceSize: 用于存储计算所需的workspace大小。
 * @param [out] executor: 执行器指针，用于后续的计算执行
 *
 * @return aclnnStatus: 执行状态，返回0表示成功，其他值表示错误。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnQuantMatmulAlltoAllGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *biasOptional, const aclTensor *x1Scale,
    const aclTensor *x2Scale, const aclTensor *commScaleOptional, const aclTensor *x1OffsetOptional,
    const aclTensor *x2OffsetOptional, const aclIntArray *alltoAllAxesOptional, const char *group, int64_t x1QuantMode,
    int64_t x2QuantMode, int64_t commQuantMode, int64_t commQuantDtype, int64_t groupSize, bool transposeX1,
    bool transposeX2, const aclTensor *output, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief 执行全连接（All-to-All）矩阵乘法计算。
 *
 * 该接口用于执行分布式训练中的通信和计算流程，需在调用前检查HCCL_BUFFSIZE环境变量配置是否合理。
 *
 * @param [in] workspace: 在Device侧申请的workspace内存地址。
 * @param [in] workspacesize: 在Device侧申请的workspace大小，由第一段接口aclnnQuantMatmulAlltoAllGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的Stream。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnQuantMatmulAlltoAll(void *workspace, uint64_t workspaceSize,
                                                                            aclOpExecutor *executor,
                                                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_QUANT_MATMUL_ALL_TO_ALL_
