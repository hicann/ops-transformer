/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_nsa_selected_attention_infer.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_NSA_SELECTED_ATTENTION_INFER_H_
#define OP_API_INC_LEVEL2_ACLNN_NSA_SELECTED_ATTENTION_INFER_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
* @brief aclnnNsaSelectedAttentionInfer的第一段接口，根据具体的计算流程，计算workspace大小。
* @domain aclnn_ops_infer
*/
aclnnStatus aclnnNsaSelectedAttentionInferGetWorkspaceSize(
    const aclTensor *query, 
    const aclTensor *key, 
    const aclTensor *value, 
    const aclTensor *topkIndices, 
    const aclTensor *attenMaskOptional,
    const aclTensor *blockTableOptional,
    const aclIntArray *actualQSeqLenOptional,
    const aclIntArray *actualKvSeqLenOptional,
    char *layoutOptional,
    int64_t numHeads,
    int64_t numKeyValueHeads,
    int64_t selectBlockSize,
    int64_t selectBlockCount,
    int64_t pageBlockSize,
    double scaleValue,
    int64_t sparseMode,
    aclTensor *output,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
* @brief aclnnNsaSelectedAttentionInfer的第二段接口，用于执行计算。
*/
aclnnStatus aclnnNsaSelectedAttentionInfer(
    void *workspace, 
    uint64_t workspaceSize, 
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_NSA_SELECTED_ATTENTION_INFER_H_
