/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_flash_mla_with_kvcache.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(QuantFlashMlaWithKvcache);

const std::array<const aclTensor*, 2> QuantFlashMlaWithKvcache(
    const aclTensor* q, const aclTensor* kCache, const aclTensor* qDescale, const aclTensor* kDescale,
    const aclTensor* blockTable, const aclTensor* cacheSeqlens, const aclTensor* cuSeqlensQOptional,
    const aclTensor* sequsedQOptional, const aclTensor* attnMaskOptional, const aclTensor* metadataOptional,
    int64_t quantMode, double softmaxScale, int64_t maskMode, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t headDimV,
    const char* layoutQ, const char* layoutKv, const char* layoutOut, bool returnSoftmaxLse, aclOpExecutor* executor)
{
    L0_DFX(QuantFlashMlaWithKvcache, q, kCache, qDescale, kDescale, blockTable, cacheSeqlens, cuSeqlensQOptional,
           sequsedQOptional, attnMaskOptional, metadataOptional, quantMode, softmaxScale, maskMode, maxSeqlenQ,
           maxSeqlenKv, headDimV, layoutQ, layoutKv, layoutOut, returnSoftmaxLse);

    if (cuSeqlensQOptional == nullptr) {
        cuSeqlensQOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (sequsedQOptional == nullptr) {
        sequsedQOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (attnMaskOptional == nullptr) {
        attnMaskOptional = executor->AllocTensor(DataType::DT_INT8, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (metadataOptional == nullptr) {
        metadataOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto attentionOutAlloc = executor->AllocTensor(DataType::DT_BF16, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxLseAlloc = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(QuantFlashMlaWithKvcache,
                           OP_INPUT(q, kCache, qDescale, kDescale, blockTable, cacheSeqlens, cuSeqlensQOptional,
                                    sequsedQOptional, attnMaskOptional, metadataOptional),
                           OP_OUTPUT(attentionOutAlloc, softmaxLseAlloc),
                           OP_ATTR(quantMode, softmaxScale, maskMode, maxSeqlenQ, maxSeqlenKv, headDimV, layoutQ,
                                   layoutKv, layoutOut, returnSoftmaxLse));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantFlashMlaWithKvcache InferShape failed.");
        return {nullptr, nullptr};
    }

    ret =
        ADD_TO_LAUNCHER_LIST_AICORE(QuantFlashMlaWithKvcache,
                                    OP_INPUT(q, kCache, qDescale, kDescale, blockTable, cacheSeqlens,
                                             cuSeqlensQOptional, sequsedQOptional, attnMaskOptional, metadataOptional),
                                    OP_OUTPUT(attentionOutAlloc, softmaxLseAlloc),
                                    OP_ATTR(quantMode, softmaxScale, maskMode, maxSeqlenQ, maxSeqlenKv, headDimV,
                                            layoutQ, layoutKv, layoutOut, returnSoftmaxLse));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantFlashMlaWithKvcache launch kernel failed.");
        return {nullptr, nullptr};
    }

    return {attentionOutAlloc, softmaxLseAlloc};
}

} // namespace l0op
