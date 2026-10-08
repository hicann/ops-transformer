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
 * \file aclnn_quant_flash_mla_with_kvcache.cpp
 * \brief QuantFlashMlaWithKvcache公共aclnn接口：参数检查后调用编译框架生成的inner函数
 */

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_quant_flash_mla_with_kvcache_inner.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// 新版本opbase存在TensorV2的新接口，用弱符号判断当前opbase是新版本还是旧版本，旧版本不支持传入非连续tensor
bool NnopbaseSupportTensorV2() __attribute__((weak));

static aclnnStatus CheckTensorContiguous(const aclTensor* kCache)
{
    if (kCache != nullptr && !IsContiguous(kCache)) {
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantFlashMlaWithKvcacheGetWorkspaceSize(
    const aclTensor* q, const aclTensor* kCache, const aclTensor* qDescale, const aclTensor* kDescale,
    const aclTensor* blockTable, const aclTensor* cacheSeqlens, const aclTensor* cuSeqlensQOptional,
    const aclTensor* sequsedQOptional, const aclTensor* attnMaskOptional, const aclTensor* metadataOptional,
    int64_t quantMode, double softmaxScale, int64_t maskMode, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t headDimV,
    const char* layoutQ, const char* layoutKv, const char* layoutOut, bool returnSoftmaxLse, const aclTensor* attnOut,
    const aclTensor* softmaxLseOptional, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("start aclnnQuantFlashMlaWithKvcacheGetWorkspaceSize");

    // 文档约束: metadata由quant_flash_mla_with_kvcache_metadata生成，当前不支持不传入
    CHECK_COND(metadataOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
               "metadata should be provided by quant_flash_mla_with_kvcache_metadata, but got null");
    // 文档约束: layout_q/layout_kv/layout_out当前不支持不传入
    CHECK_COND(layoutQ != nullptr, ACLNN_ERR_PARAM_INVALID, "layoutQ should be provided, but got null");
    CHECK_COND(layoutKv != nullptr, ACLNN_ERR_PARAM_INVALID, "layoutKv should be provided, but got null");
    CHECK_COND(layoutOut != nullptr, ACLNN_ERR_PARAM_INVALID, "layoutOut should be provided, but got null");

    const aclTensor* placeHolder = nullptr;
    const aclTensor* tempTensor = nullptr;

    QuantFlashMlaWithKvcacheProcessSoftmaxLse(returnSoftmaxLse, softmaxLseOptional, tempTensor, placeHolder);

    aclnnStatus ret = CheckTensorContiguous(kCache);
    if (ret != ACLNN_SUCCESS && NnopbaseSupportTensorV2 == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "When tensor is not contiguous, opbase package version check failed");
        return ret;
    }

    ret = aclnnInnerQuantFlashMlaWithKvcacheGetWorkspaceSize(
        q, kCache, qDescale, kDescale, blockTable, cacheSeqlens, cuSeqlensQOptional, sequsedQOptional, attnMaskOptional,
        metadataOptional, quantMode, softmaxScale, maskMode, maxSeqlenQ, maxSeqlenKv, headDimV,
        const_cast<char*>(layoutQ), const_cast<char*>(layoutKv), const_cast<char*>(layoutOut), returnSoftmaxLse,
        attnOut, placeHolder, workspaceSize, executor);

    if (!returnSoftmaxLse) {
        aclDestroyTensor(tempTensor);
    }

    return ret;
}

aclnnStatus aclnnQuantFlashMlaWithKvcache(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          const aclrtStream stream)
{
    return aclnnInnerQuantFlashMlaWithKvcache(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
