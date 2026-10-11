/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_QUANT_GROUPED_MATMUL_DEQUANT_WEIGHT_NZ_H_
#define OP_API_INC_QUANT_GROUPED_MATMUL_DEQUANT_WEIGHT_NZ_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQuantGroupedMatmulDequantWeightNZ的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: 表示公式中的x，数据类型支持FLOAT16数据类型，数据格式支持ND。
 * @param [in] weight:
 * 表示公式中的weight，数据类型支持INT8数据类型，数据格式支持NZ。该接口会忽略weight的数据格式，并强制视为FRACTAL_NZ格式。
 * @param [in] weightScale:
 * 表示weight的量化系数，数据类型支持FLOAT32、INT64数据类型，数据格式支持ND。
 * @param [in] groupList: 必选参数，代表输入和输出分组轴上的索引情况，数据类型支持INT64。
 * @param [in] biasOptional: 表示计算的偏移量，可选参数，当前仅支持传入空指针。
 * @param [in] xScaleOptional:
 * 表示x的量化系数，可选参数，数据类型支持FLOAT32、FLOAT16数据类型，数据格式支持ND。
 * @param [in] xOffsetOptional: 表示x的偏移量，可选参数，当前仅支持传入空指针。
 * @param [in] smoothScaleOptional:
 * 表示x的平滑系数，可选参数，数据类型支持FLOAT16数据类型，数据格式支持ND。
 * @param [in] xQuantMode: 指定输入x的量化模式，支持取值pertoken/pertensor。
 * @param [in] transposeWeight: 表示输入weight是否转置，当前只支持true。
 * @param [out] out: 表示公式中的out，数据类型支持FLOAT16数据类型，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize(
    const aclTensor* x, const aclTensor* weight, const aclTensor* weightScale, const aclTensor* groupList,
    const aclTensor* biasOptional, const aclTensor* xScaleOptional, const aclTensor* xOffsetOptional,
    const aclTensor* smoothScaleOptional, char* xQuantMode, bool transposeWeight, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnQuantGroupedMatmulDequantWeightNZ的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnQuantGroupedMatmulDequantWeightNZ(void* workspace,
                                                                                          uint64_t workspaceSize,
                                                                                          aclOpExecutor* executor,
                                                                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_QUANT_GROUPED_MATMUL_DEQUANT_WEIGHT_NZ_H_
