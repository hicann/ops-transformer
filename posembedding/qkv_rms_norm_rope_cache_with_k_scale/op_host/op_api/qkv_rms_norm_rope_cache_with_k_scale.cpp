/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "qkv_rms_norm_rope_cache_with_k_scale.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

namespace l0op {
OP_TYPE_REGISTER(QkvRmsNormRopeCacheWithKScale);

namespace {
constexpr const char *DEFAULT_QKV_LAYOUT = "TND";
constexpr const char *DEFAULT_Q_OUT_LAYOUT = "NTD";

const char *GetLayoutOrDefault(const char *layout, const char *defaultLayout)
{
    return layout == nullptr || layout[0] == '\0' ? defaultLayout : layout;
}

} // namespace

QkvRmsNormRopeCacheWithKScaleResult QkvRmsNormRopeCacheWithKScale(
    const aclTensor *qkv, const aclTensor *qGamma, const aclTensor *kGamma, const aclTensor *cosSin,
    const aclTensor *slotMapping, aclTensor *kCache, aclTensor *vCache, aclTensor *kScaleCache,
    const aclTensor *queryStartLoc, const aclTensor *seqLens, const aclTensor *rotationOptional,
    const aclTensor *vScaleOptional, const aclTensor *mropePositionOptional, const aclIntArray *headNums,
    const char *layoutQkv, const char *layoutQOut, float epsilon, const aclIntArray *mropeSectionOptional,
    const char *qQuantMode, const char *kQuantMode, int64_t qOutDtype, aclTensor *qOut, aclTensor *qScaleOptional,
    aclOpExecutor *executor)
{
    const char *layoutQkvAttr = GetLayoutOrDefault(layoutQkv, DEFAULT_QKV_LAYOUT);
    const char *layoutQOutAttr = GetLayoutOrDefault(layoutQOut, DEFAULT_Q_OUT_LAYOUT);
    // The ACLNN entry point resolves null/empty quant modes before calling this L0 wrapper.
    L0_DFX(QkvRmsNormRopeCacheWithKScale, qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache,
           queryStartLoc, seqLens, rotationOptional, vScaleOptional, mropePositionOptional, headNums, layoutQkvAttr,
           layoutQOutAttr, epsilon, mropeSectionOptional, qQuantMode, kQuantMode, qOutDtype);

    // A null optional output is omitted from the launcher argument list. Keep
    // the middle qScale ABI slot materialized internally; the public return
    // value remains null when the caller did not request qScale.
    aclTensor *qScaleForKernel = qScaleOptional;
    if (qScaleForKernel == nullptr) {
        qScaleForKernel = executor->AllocTensor(op::DataType::DT_FLOAT, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
        if (qScaleForKernel == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to allocate internal qScale tensor.");
            return {ACLNN_ERR_INNER_NULLPTR, nullptr, nullptr, nullptr, nullptr, nullptr};
        }
    }

    auto ret = INFER_SHAPE(QkvRmsNormRopeCacheWithKScale,
                           OP_INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache,
                                    queryStartLoc, seqLens, rotationOptional, vScaleOptional, mropePositionOptional),
                           OP_OUTPUT(qOut, qScaleForKernel, kCache, vCache, kScaleCache),
                           OP_ATTR(headNums, layoutQkvAttr, layoutQOutAttr, epsilon, mropeSectionOptional, qQuantMode,
                                   qOutDtype, kQuantMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed, internal status is %d.", ret);
        return {ACLNN_ERR_PARAM_INVALID, nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        QkvRmsNormRopeCacheWithKScale,
        OP_INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, queryStartLoc, seqLens,
                 rotationOptional, vScaleOptional, mropePositionOptional),
        OP_OUTPUT(qOut, qScaleForKernel, kCache, vCache, kScaleCache),
        OP_ATTR(headNums, layoutQkvAttr, layoutQOutAttr, epsilon, mropeSectionOptional, qQuantMode, qOutDtype,
                kQuantMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed, internal status is %d.", ret);
        return {ACLNN_ERR_PARAM_INVALID, nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    return {ACLNN_SUCCESS, qOut, qScaleOptional, kCache, vCache, kScaleCache};
}

} // namespace l0op
