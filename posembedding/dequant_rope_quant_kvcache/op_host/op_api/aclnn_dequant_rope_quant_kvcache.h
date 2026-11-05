/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_ACLNN_DEQUANT_ROPE_QUANT_KVCACHE_H_
#define OP_API_ACLNN_DEQUANT_ROPE_QUANT_KVCACHE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Calculate workspace for dequantization, RoPE and quantized KV cache update.
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnDequantRopeQuantKvcacheGetWorkspaceSize(
    const aclTensor* x, const aclTensor* cos, const aclTensor* sin, aclTensor* kCacheRef, aclTensor* vCacheRef,
    const aclTensor* indices, const aclTensor* scaleK, const aclTensor* scaleV, const aclTensor* offsetKOptional,
    const aclTensor* offsetVOptional, const aclTensor* weightScaleOptional, const aclTensor* activationScaleOptional,
    const aclTensor* biasOptional, const aclIntArray* sizeSplits, char* quantModeOptional, char* layoutOptional,
    bool kvOutput, char* cacheModeOptional, const aclTensor* qOut, const aclTensor* kOut, const aclTensor* vOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnDequantRopeQuantKvcache(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
