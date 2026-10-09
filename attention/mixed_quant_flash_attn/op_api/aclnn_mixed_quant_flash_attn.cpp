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
 * \file aclnn_mixed_quant_flash_attn.cpp
 * \brief
 */

#include "aclnn_mixed_quant_flash_attn.h"

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "aclnn_mixed_quant_flash_attn_inner.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// 第一段接口：计算workspace大小
aclnnStatus aclnnMixedQuantFlashAttnGetWorkspaceSize(
    const aclTensor* q, const aclTensor* k, const aclTensor* v, const aclTensor* kDescale, const aclTensor* vDescale,
    const aclTensor* blockTableOptional, const aclTensor* cuSeqlensQOptional, const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKvOptional, const aclTensor* sinksOptional, const aclTensor* attnMaskOptional,
    const aclTensor* metadataOptional, int64_t quantComputeMode, double softmaxScale, int64_t maskMode, int64_t winLeft,
    int64_t winRight, int64_t maxSeqlenQ, int64_t maxSeqlenKV, const char* layoutQ, const char* layoutKv,
    const char* layoutAttnOut, bool returnSoftmaxLse, const aclTensor* attnOut, const aclTensor* softmaxLseOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("start aclnnMixedQuantFlashAttnGetWorkspaceSize");

    // sinks shape为{0}时置nullptr
    MixedQuantFlashAttnProcessSinks(sinksOptional);

    const aclTensor* placeHolder = nullptr;
    const aclTensor* tempTensor = nullptr;

    MixedQuantFlashAttnProcessSoftmaxLse(returnSoftmaxLse, softmaxLseOptional, tempTensor, placeHolder);

    aclnnStatus ret = aclnnInnerMixedQuantFlashAttnGetWorkspaceSize(
        q, k, v, kDescale, vDescale, blockTableOptional, cuSeqlensQOptional, sequsedQOptional, sequsedKvOptional,
        sinksOptional, attnMaskOptional, metadataOptional, quantComputeMode, softmaxScale, maskMode, winLeft, winRight,
        maxSeqlenQ, maxSeqlenKV, layoutQ, layoutKv, layoutAttnOut, returnSoftmaxLse, attnOut, placeHolder,
        workspaceSize, executor);

    // 销毁占位符
    if (returnSoftmaxLse == false) {
        aclDestroyTensor(tempTensor);
    }

    return ret;
}

// 第二段接口：执行计算
aclnnStatus aclnnMixedQuantFlashAttn(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     const aclrtStream stream)
{
    return aclnnInnerMixedQuantFlashAttn(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
