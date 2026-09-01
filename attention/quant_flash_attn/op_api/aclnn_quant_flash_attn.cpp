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
 * \file aclnn_quant_flash_attn.cpp
 * \brief
 */

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_quant_flash_attn_inner.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// 新版本opbase存在TensorV2的新接口，用弱符号判断当前opbase是新版本还是旧版本，旧版本不支持传入非连续tensor
bool NnopbaseSupportTensorV2() __attribute__((weak));

static aclnnStatus CheckTensorContiguous(const aclTensor *k, const aclTensor *v, const aclTensor *kDescale,
                                         const aclTensor *vDescale)
{
    if ((k != nullptr && !IsContiguous(k)) || (v != nullptr && !IsContiguous(v))) {
        return ACLNN_ERR_INNER;
    }
    if ((kDescale != nullptr && !IsContiguous(kDescale)) || (vDescale != nullptr && !IsContiguous(vDescale))) {
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantFlashAttnGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *qDescale, const aclTensor *kDescale,
    const aclTensor *vDescale, const aclTensor *blockTableOptional, const aclTensor *pScaleOptional,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional, const aclTensor *sinksOptional, const aclTensor *attnMaskOptional,
    const aclTensor *metadataOptional, int64_t quantMode, double softmaxScale, int64_t maskMode, int64_t winLeft,
    int64_t winRight, int64_t maxSeqlenQ, int64_t maxSeqlenKV, const char *layoutQ, const char *layoutQDescale,
    const char *layoutKv, const char *layoutOut, bool returnSoftmaxLse, const aclTensor *attnOut,
    const aclTensor *softmaxLseOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("start aclnnQuantFlashAttnGetWorkspaceSize");

    QuantFlashAttnProcessSinks(sinksOptional);

    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;

    QuantFlashAttnProcessSoftmaxLse(returnSoftmaxLse, softmaxLseOptional, tempTensor, placeHolder);

    aclnnStatus ret = CheckTensorContiguous(k, v, kDescale, vDescale);
    if (ret != ACLNN_SUCCESS && NnopbaseSupportTensorV2 == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "When tensor is not contiguous, opbase package version check failed");
        return ret;
    }
    ret = aclnnInnerQuantFlashAttnGetWorkspaceSize(
        q, k, v, qDescale, kDescale, vDescale, blockTableOptional, pScaleOptional, cuSeqlensQOptional,
        cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional, sinksOptional, attnMaskOptional, metadataOptional,
        quantMode, softmaxScale, maskMode, winLeft, winRight, maxSeqlenQ, maxSeqlenKV, layoutQ, layoutQDescale,
        layoutKv, layoutOut, returnSoftmaxLse, attnOut, placeHolder, workspaceSize, executor);

    if (!returnSoftmaxLse) {
        aclDestroyTensor(tempTensor);
    }

    return ret;
}

aclnnStatus aclnnQuantFlashAttn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                const aclrtStream stream)
{
    return aclnnInnerQuantFlashAttn(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
