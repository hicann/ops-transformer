/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace {
bool FakeArray(const aclIntArray *inArray, aclTensor *&outTensor)
{
    OP_LOGD("start fake tensor");
    if (inArray != nullptr) {
        OP_LOGD("input array is not nullptr");
        int64_t size = static_cast<int64_t>(inArray->Size());
        std::vector<int64_t> shape = {size};
        outTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT64, nullptr, 0, ACL_FORMAT_ND,
                                    shape.data(), shape.size(), nullptr);
        if (outTensor == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc tensor failed");
            return false;
        }
    }
    OP_LOGD("end fake tensor");
    return true;
}
} // namespace

namespace l0op {

OP_TYPE_REGISTER(FlashAttentionScore);

const std::array<const aclTensor *, 4> FlashAttentionScore(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclTensor *sinkOptional, const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional, const aclTensor *dScaleQOptional, const aclTensor *dScaleKOptional,
    const aclTensor *dScaleVOptional, const aclTensor *pScaleOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, double scaleValue, double keepProb, int64_t preTockens, int64_t nextTockens,
    int64_t headNum, const char *inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t pseType, int64_t seed,
    int64_t offset, int64_t outDtype, const char *softmaxOutLayout, aclOpExecutor *executor,
    bool isMaxWorkspace = false)
{
    L0_DFX(FlashAttentionScore, query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
           attenMaskOptional, sinkOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional,
           qStartIdxOptional, kvStartIdxOptional, dScaleQOptional, dScaleKOptional, dScaleVOptional, pScaleOptional,
           queryRopeOptional, keyRopeOptional, scaleValue, keepProb, preTockens, nextTockens, headNum, inputLayout,
           innerPrecise, sparseMode, pseType, seed, offset, outDtype, softmaxOutLayout);

    // Preserve absent inputs as nullptr through shape inference and launcher
    // registration, avoiding workspace-backed placeholders that may alias the
    // kernel workspace start address.

    const aclTensor *prefixOptionalTensor = nullptr;
    if (prefixOptional) {
        prefixOptionalTensor = executor->ConvertToTensor(prefixOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(prefixOptionalTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(prefixOptionalTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(prefixOptionalTensor)->SetOriginalFormat(Format::FORMAT_ND);
    }

    const aclTensor *actualSeqQLen = nullptr;
    if (actualSeqQLenOptional) {
        if (!isMaxWorkspace) {
            actualSeqQLen = executor->ConvertToTensor(actualSeqQLenOptional, DataType::DT_INT64);
            const_cast<aclTensor *>(actualSeqQLen)->SetStorageFormat(Format::FORMAT_ND);
            const_cast<aclTensor *>(actualSeqQLen)->SetViewFormat(Format::FORMAT_ND);
            const_cast<aclTensor *>(actualSeqQLen)->SetOriginalFormat(Format::FORMAT_ND);
        } else {
            FakeArray(actualSeqQLenOptional, const_cast<aclTensor *&>(actualSeqQLen));
        }
    }

    const aclTensor *actualSeqKvLen = nullptr;
    if (actualSeqKvLenOptional) {
        if (!isMaxWorkspace) {
            actualSeqKvLen = executor->ConvertToTensor(actualSeqKvLenOptional, DataType::DT_INT64);
            const_cast<aclTensor *>(actualSeqKvLen)->SetStorageFormat(Format::FORMAT_ND);
            const_cast<aclTensor *>(actualSeqKvLen)->SetViewFormat(Format::FORMAT_ND);
            const_cast<aclTensor *>(actualSeqKvLen)->SetOriginalFormat(Format::FORMAT_ND);
        } else {
            FakeArray(actualSeqKvLenOptional, const_cast<aclTensor *&>(actualSeqKvLen));
        }
    }

    const aclTensor *qStartIdxOptionalTensor = nullptr;
    if (qStartIdxOptional) {
        qStartIdxOptionalTensor = executor->ConvertToTensor(qStartIdxOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(qStartIdxOptionalTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(qStartIdxOptionalTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(qStartIdxOptionalTensor)->SetOriginalFormat(Format::FORMAT_ND);
    }

    const aclTensor *kvStartIdxOptionalTensor = nullptr;
    if (kvStartIdxOptional) {
        kvStartIdxOptionalTensor = executor->ConvertToTensor(kvStartIdxOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(kvStartIdxOptionalTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(kvStartIdxOptionalTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(kvStartIdxOptionalTensor)->SetOriginalFormat(Format::FORMAT_ND);
    }

    auto softmaxMaxOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxSumOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    DataType outputDtype = query->GetDataType();
    if (query->GetDataType() == DataType::DT_FLOAT8_E4M3FN || query->GetDataType() == DataType::DT_FLOAT8_E5M2 ||
        query->GetDataType() == DataType::DT_HIFLOAT8) {
        if (outDtype == 0) {
            outputDtype = DataType::DT_FLOAT16;
        } else {
            outputDtype = DataType::DT_BF16;
        }
    }
    auto softmaxOutOut = executor->AllocTensor(outputDtype, Format::FORMAT_ND, Format::FORMAT_ND);
    auto attentionOutOut = executor->AllocTensor(outputDtype, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(
        FlashAttentionScore,
        OP_INPUT(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
                 prefixOptionalTensor, actualSeqQLen, actualSeqKvLen, qStartIdxOptionalTensor, kvStartIdxOptionalTensor,
                 dScaleQOptional, dScaleKOptional, dScaleVOptional, queryRopeOptional, keyRopeOptional, sinkOptional,
                 pScaleOptional),
        OP_OUTPUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut),
        OP_ATTR(static_cast<float>(scaleValue), static_cast<float>(keepProb), preTockens, nextTockens, headNum,
                inputLayout, innerPrecise, sparseMode, pseType, seed, offset, outDtype, softmaxOutLayout));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FlashAttentionScore InferShape failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        FlashAttentionScore,
        OP_INPUT(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
                 prefixOptionalTensor, actualSeqQLen, actualSeqKvLen, qStartIdxOptionalTensor, kvStartIdxOptionalTensor,
                 dScaleQOptional, dScaleKOptional, dScaleVOptional, queryRopeOptional, keyRopeOptional, sinkOptional,
                 pScaleOptional),
        OP_OUTPUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut),
        OP_ATTR(static_cast<float>(scaleValue), static_cast<float>(keepProb), preTockens, nextTockens, headNum,
                inputLayout, innerPrecise, sparseMode, pseType, seed, offset, outDtype, softmaxOutLayout));
    if (actualSeqKvLenOptional && isMaxWorkspace) {
        aclDestroyTensor(actualSeqKvLen);
    }
    if (actualSeqQLenOptional && isMaxWorkspace) {
        aclDestroyTensor(actualSeqQLen);
    }
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FlashAttentionScore launch kernel failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    return {softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut};
}

} // namespace l0op
