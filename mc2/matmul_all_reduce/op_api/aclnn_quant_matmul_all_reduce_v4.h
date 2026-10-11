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
 * \file aclnn_quant_matmul_all_reduce_v4.h
 * \brief
 */
#ifndef OP_API_INC_QUANT_MATMUL_ALL_REDUCE_V4_
#define OP_API_INC_QUANT_MATMUL_ALL_REDUCE_V4_

#include <string>

#include "aclnn/aclnn_base.h"
#include "hccl/hccl_types.h"

#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQuantMatmulAllReduceV4的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：实现mm+AllReduce融合计算
 * @param [in] x1: matmul左矩阵，数据类型支持：float4_e2m1, float4_e1m2, int8, float8_e4m3fn, float8_e5m2, hifloat8。
 * @param [in] x2: matmul右矩阵，数据类型支持：float4_e2m1, float4_e1m2, int8, float8_e4m3fn, float8_e5m2, hifloat8。
 * @param [in] biasOptional: 偏置，数据类型支持：int32, float32。
 * @param [in] x3Optional: add操作参数，数据类型支持：float16, bfloat16, float32。
 * @param [in] x1ScaleOptional: x1去量化系数，数据类型支持：float32, float8_e8m0。
 * @param [in] x2Scale: x2去量化系数，数据类型支持：int64, uint64, bfloat16, float32。
 * @param [in] commQuantScale1Optional: 通信量化计算参数，数据类型支持：float16, bfloat16。
 * @param [in] commQuantScale2Optional: 通信量化计算参数，数据类型支持：float16, bfloat16。
 * @param [in] group: 标识列组的字符串。
 * @param [in] reduceOp: reduce操作类型，默认值：sum。
 * @param [in] commTurn: 通信数据切分数，即总数据量/单次通信量，默认值：0。
 * @param [in] streamMode: acl流模式的枚举，类型支持：1。
 * @param [in] groupSize: 量化系数在不同维度上量化尺度，默认值：0。
 * @param [in] commQuantMode: 静态量化和动态量化的标志位，默认值：0。动态量化目前只为fp8通信场景。
 * @param [out] output: 计算+通信的结果，数据类型：float16, bfloat16, float32。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnQuantMatmulAllReduceV4GetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional, const aclTensor* x3Optional,
    const aclTensor* x1ScaleOptional, const aclTensor* x2Scale, const aclTensor* commQuantScale1Optional,
    const aclTensor* commQuantScale2Optional, const char* group, const char* reduceOp, int64_t commTurn,
    int64_t streamMode, int64_t groupSize, int64_t commQuantMode, const aclTensor* output, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnQuantMatmulAllReduceV4的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnQuantMatmulAllReduceV4GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnQuantMatmulAllReduceV4(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                  const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_QUANT_MATMUL_ALL_REDUCE_V4_
