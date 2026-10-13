/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_NSA_COMPRESS_ATTENTION_INFER_H_
#define OP_API_INC_LEVEL2_ACLNN_NSA_COMPRESS_ATTENTION_INFER_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnNsaCompressAttentionInfer的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] query : required
 * @param [in] key : required
 * @param [in] value : required
 * @param [in] attentionMaskOptional : optional
 * @param [in] blockTableOptional : optional
 * @param [in] actualQSeqLenOptional : optional
 * @param [in] actualCmpKvSeqLenOptional : optional
 * @param [in] actualSelKvSeqLenOptional : optional
 * @param [in] topKMaskOptional : optional
 * @param [in] numHeads : optional
 * @param [in] numKeyValueHeads : optional
 * @param [in] selectBlockSize : optional
 * @param [in] selectBlockCount : optional
 * @param [in] compressBlockSize : optional
 * @param [in] compressBlockStride : optional
 * @param [in] scaleValue : optional
 * @param [in] layoutOptional : optional
 * @param [in] pageBlockSize : optional
 * @param [in] sparseMode : optional
 * @param [out] output : required
 * @param [out] topKOutput : required
 * @param [out] workspaceSize : size of workspace(output).
 * @param [out] executor : executor context(output).
 * @return aclnnStatus: 返回状态码
 */

aclnnStatus aclnnNsaCompressAttentionInferGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* attentionMaskOptional,
    const aclTensor* blockTableOptional, const aclIntArray* actualQSeqLenOptional,
    const aclIntArray* actualCmpKvSeqLenOptional, const aclIntArray* actualSelKvSeqLenOptional,
    const aclTensor* topKMaskOptional, int64_t numHeads, int64_t numKeyValueHeads, int64_t selectBlockSize,
    int64_t selectBlockCount, int64_t compressBlockSize, int64_t compressBlockStride, double scaleValue,
    char* layoutOptional, int64_t pageBlockSize, int64_t sparseMode, const aclTensor* output,
    const aclTensor* topKOutput, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnNsaCompressAttentionInfer的第二段接口，用于执行计算。
 * @param [in] workspace : workspace memory addr(input).
 * @param [in] workspaceSize : size of workspace(input).
 * @param [in] executor : executor context(input).
 * @param [in] stream : acl stream.
 * @return aclnnStatus: 返回状态码
 */

aclnnStatus aclnnNsaCompressAttentionInfer(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
