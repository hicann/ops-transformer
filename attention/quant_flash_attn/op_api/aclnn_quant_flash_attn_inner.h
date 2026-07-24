/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_QUANT_FLASH_ATTN_INNER_H_
#define ACLNN_QUANT_FLASH_ATTN_INNER_H_
#define ACLNN_API __attribute__((visibility("default")))

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerQuantFlashAttnGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *qDescale, const aclTensor *kDescale, const aclTensor *vDescale,
    const aclTensor *blockTableOptional,
    const aclTensor *pScaleOptional,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    const aclTensor *sinksOptional, const aclTensor *attnMaskOptional,
    const aclTensor *metadataOptional,
    int64_t quantMode,
    double softmaxScale, int64_t maskMode, int64_t winLeft, int64_t winRight,
    int64_t maxSeqlenQ, int64_t maxSeqlenKV,
    const char *layoutQ, const char *layoutQDescale, const char *layoutKv, const char *layoutOut,
    bool returnSoftmaxLse,
    const aclTensor *attnOut, const aclTensor *softmaxLse,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerQuantFlashAttn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            const aclrtStream stream);

void QuantFlashAttnProcessSoftmaxLse(bool returnSoftmaxLse, const aclTensor *softmaxLse,
                                     const aclTensor *&tempTensor, const aclTensor *&placeHolder);

void QuantFlashAttnProcessSinks(const aclTensor *&sinksOptional);

#ifdef __cplusplus
}
#endif

#endif
