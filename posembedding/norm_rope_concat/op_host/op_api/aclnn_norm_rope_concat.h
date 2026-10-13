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
 * \file aclnn_norm_rope_concat.h
 * \brief
 */
#ifndef ACLNN_NORM_ROPE_CONCAT_H_
#define ACLNN_NORM_ROPE_CONCAT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnInterleaveRope的第一段接口，根据具体的计算流程，计算workspace大小。
 *
 * @param [in] query: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] key: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] value: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] encoderQuery: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] encoderKey: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] encoderValue: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] normQueryWeight: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] normQueryBias: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] normKeyWeight: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] normKeyBias: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] normAddedQueryWeight: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] normAddedQueryBias: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] normAddedKeyWeight: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] normAddedKeyBias: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] ropeSin: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] ropeCos: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,支持非连续的Tensor,数据格式支持ND。
 * @param [in] normType: 归一化类型, 取值范围[0, 1], 0表示LayerNorm, 1表示AFFINE LayerNorm。
 * @param [in] normAddedType: 带仿射变换参数的归一化类型, 取值范围[0, 1], 0表示LayerNorm, 1表示AFFINE LayerNorm。
 * @param [in] ropeType: 旋转位置编码类型, 取值范围[0, 1], 0表示Interleave, 1表示Half。
 * @param [in] concatOrder: 拼接顺序, 取值范围[0, 1], 0表示query在前, 1表示query在后。
 * @param [in] eps: 归一化时的epsilon值。
 * @param [in] isTraining: 是否为训练阶段。
 * @param [in] out: npu
 * device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,且数据类型与x一致,
 * shape与x相同, 数据格式支持ND, 且数据格式需要与x一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小.
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnNormRopeConcatGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* encoderQuery,
    const aclTensor* encoderKey, const aclTensor* encoderValue, const aclTensor* normQueryWeight,
    const aclTensor* normQueryBias, const aclTensor* normKeyWeight, const aclTensor* normKeyBias,
    const aclTensor* normAddedQueryWeight, const aclTensor* normAddedQueryBias, const aclTensor* normAddedKeyWeight,
    const aclTensor* normAddedKeyBias, const aclTensor* ropeSin, const aclTensor* ropeCos, int64_t normType,
    int64_t normAddedType, int64_t ropeType, int64_t concatOrder, double eps, bool isTraining,
    const aclTensor* queryOutput, const aclTensor* keyOutput, const aclTensor* valueOutput,
    const aclTensor* normQueryMean, const aclTensor* normQueryRstd, const aclTensor* normKeyMean,
    const aclTensor* normKeyRstd, const aclTensor* normAddedQueryMean, const aclTensor* normAddedQueryRstd,
    const aclTensor* normAddedKeyMean, const aclTensor* normAddedKeyRstd, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/* @brief aclnnInterleaveRope的第二段接口，用于执行计算.
 * @param [in] workspace: 在npu device侧申请的workspace内存起址
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnBitwiseNotGetWorkspaceSize获取.
 * @param [in] executor: op执行器，包含了算子计算流程.
 * @param [in] stream: acl stream流.
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnNormRopeConcat(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif
