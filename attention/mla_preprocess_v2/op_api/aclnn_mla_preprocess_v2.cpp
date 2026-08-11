/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_mla_preprocess_v2.h"

#include "../../mla_preprocess/op_api/empty_tensor_holder.h"
#include "opdev/op_log.h"

namespace {

extern "C" aclnnStatus aclnnInnerMlaPreprocessV2GetWorkspaceSize(
    const aclTensor *input, const aclTensor *gamma0, const aclTensor *beta0, const aclTensor *quantScale0,
    const aclTensor *quantOffset0, const aclTensor *wdqkv, const aclTensor *descale0, const aclTensor *bias0,
    const aclTensor *gamma1, const aclTensor *beta1, const aclTensor *quantScale1, const aclTensor *quantOffset1,
    const aclTensor *wuq, const aclTensor *descale1, const aclTensor *bias1, const aclTensor *gamma2,
    const aclTensor *cos, const aclTensor *sin, const aclTensor *wuk, const aclTensor *kvCache,
    const aclTensor *kvCacheRope, const aclTensor *slotMapping, const aclTensor *ctkvScale,
    const aclTensor *qNopeScale, int64_t wdqDim, int64_t qRopeDim, int64_t kRopeDim, double epsilon,
    int64_t qRotaryCoeff, int64_t kRotaryCoeff, bool transeposeWdq, bool transeposeWuq, bool transeposeWuk,
    int64_t cacheMode, int64_t quantMode, bool doRmsNorm, int64_t wdkvSplitCount, bool qDownOutFlag,
    const aclTensor *qOutOut, const aclTensor *kvCacheOutOut, const aclTensor *qRopeOutOut,
    const aclTensor *krCacheOutOut, const aclTensor *qDownOutOut, uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerMlaPreprocessV2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

}  // namespace

extern "C" aclnnStatus aclnnMlaPreprocessV2GetWorkspaceSize(
    const aclTensor *input, const aclTensor *gamma0, const aclTensor *beta0, const aclTensor *quantScale0,
    const aclTensor *quantOffset0, const aclTensor *wdqkv, const aclTensor *descale0, const aclTensor *bias0,
    const aclTensor *gamma1, const aclTensor *beta1, const aclTensor *quantScale1, const aclTensor *quantOffset1,
    const aclTensor *wuq, const aclTensor *descale1, const aclTensor *bias1, const aclTensor *gamma2,
    const aclTensor *cos, const aclTensor *sin, const aclTensor *wuk, const aclTensor *kvCache,
    const aclTensor *kvCacheRope, const aclTensor *slotMapping, const aclTensor *ctkvScale,
    const aclTensor *qNopeScale, int64_t wdqDim, int64_t qRopeDim, int64_t kRopeDim, double epsilon,
    int64_t qRotaryCoeff, int64_t kRotaryCoeff, bool transeposeWdq, bool transeposeWuq, bool transeposeWuk,
    int64_t cacheMode, int64_t quantMode, bool doRmsNorm, int64_t wdkvSplitCount, bool qDownOutFlag,
    const aclTensor *qOutOut, const aclTensor *kvCacheOutOut, const aclTensor *qRopeOutOut,
    const aclTensor *krCacheOutOut, const aclTensor *qDownOutOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    const bool cosIsNull = cos == nullptr;
    const bool sinIsNull = sin == nullptr;
    if (cosIsNull != sinIsNull) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "cos and sin must both be nullptr or both be non-null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    aclDataType ropeDataType = ACL_DT_UNDEFINED;
    if (cosIsNull) {
        const aclnnStatus status = aclGetDataType(input, &ropeDataType);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    MlaPreprocessApi::EmptyTensorHolder cosHolder(cos, ropeDataType);
    MlaPreprocessApi::EmptyTensorHolder sinHolder(sin, ropeDataType);
    if (cosIsNull && (!cosHolder.IsValid() || !sinHolder.IsValid())) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to create empty cos or sin tensor.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    return aclnnInnerMlaPreprocessV2GetWorkspaceSize(
        input, gamma0, beta0, quantScale0, quantOffset0, wdqkv, descale0, bias0, gamma1, beta1, quantScale1,
        quantOffset1, wuq, descale1, bias1, gamma2, cos, sin, wuk, kvCache, kvCacheRope, slotMapping, ctkvScale,
        qNopeScale, wdqDim, qRopeDim, kRopeDim, epsilon, qRotaryCoeff, kRotaryCoeff, transeposeWdq, transeposeWuq,
        transeposeWuk, cacheMode, quantMode, doRmsNorm, wdkvSplitCount, qDownOutFlag, qOutOut, kvCacheOutOut,
        qRopeOutOut, krCacheOutOut, qDownOutOut, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnMlaPreprocessV2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return aclnnInnerMlaPreprocessV2(workspace, workspaceSize, executor, stream);
}
