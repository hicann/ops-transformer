/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file aclnn_flash_attn.cpp
 * \brief
 */

#include "aclnn_flash_attn.h"

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "aclnn_flash_attn_inner.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// 第一段接口：计算workspace大小
aclnnStatus aclnnFlashAttnGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
                                           const aclTensor *blockTableOptional, const aclTensor *cuSeqlensQOptional,
                                           const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
                                           const aclTensor *sequsedKvOptional, const aclTensor *sinksOptional,
                                           const aclTensor *attnMaskOptional, const aclTensor *metadataOptional,
                                           double softmaxScale, int64_t maskMode, int64_t winLeft, int64_t winRight,
                                           int64_t maxSeqlenQ, int64_t maxSeqlenKV, const char *layoutQ,
                                           const char *layoutKv, const char *layoutOut, int64_t returnSoftmaxLse,
                                           const aclTensor *attnOut, const aclTensor *softmaxLseOptional,
                                           uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("start aclnnFlashAttnGetWorkspaceSize");

    // sinks shape为{0}时置nullptr
    FlashAttnProcessSinks(sinksOptional);

    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;

    FlashAttnProcessSoftmaxLse(returnSoftmaxLse, softmaxLseOptional, tempTensor, placeHolder);

    aclnnStatus ret = aclnnInnerFlashAttnGetWorkspaceSize(
        q, k, v, blockTableOptional, cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional,
        sinksOptional, attnMaskOptional, metadataOptional, softmaxScale, maskMode, winLeft, winRight, maxSeqlenQ,
        maxSeqlenKV, layoutQ, layoutKv, layoutOut, returnSoftmaxLse, attnOut, placeHolder, workspaceSize, executor);

    // 销毁占位符
    if (returnSoftmaxLse == 0) {
        aclDestroyTensor(tempTensor);
    }

    return ret;
}

// 第二段接口：执行计算
aclnnStatus aclnnFlashAttn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    return aclnnInnerFlashAttn(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
