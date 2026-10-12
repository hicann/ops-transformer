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

namespace l0op {

OP_TYPE_REGISTER(NsaCompressAttention);

const std::array<const aclTensor*, 4> NsaCompressAttention(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* attenMaskOptional,
    const aclIntArray* actualSeqQLenOptional, const aclIntArray* actualCmpSeqKvLenOptional,
    const aclIntArray* actualSelSeqKvLenOptional, const aclTensor* topkMaskOptional, double scaleValueOptional,
    int64_t headNum, const char* inputLayout, int64_t sparseModeOptional, int64_t compressBlockSize,
    int64_t compressStride, int64_t selectBlockSize, int64_t selectBlockCount, aclOpExecutor* executor)
{
    L0_DFX(NsaCompressAttention, query, key, value, attenMaskOptional, actualSeqQLenOptional, actualCmpSeqKvLenOptional,
           actualSelSeqKvLenOptional, topkMaskOptional, scaleValueOptional, headNum, inputLayout, sparseModeOptional,
           compressBlockSize, compressStride, selectBlockSize, selectBlockCount);

    if (attenMaskOptional == nullptr) {
        attenMaskOptional = executor->AllocTensor(DataType::DT_BOOL, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor* actualSeqQLen = nullptr;
    if (actualSeqQLenOptional) {
        actualSeqQLen = executor->ConvertToTensor(actualSeqQLenOptional, DataType::DT_INT64);
        const_cast<aclTensor*>(actualSeqQLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(actualSeqQLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(actualSeqQLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualSeqQLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor* actualCmpSeqKvLen = nullptr;
    if (actualCmpSeqKvLenOptional) {
        actualCmpSeqKvLen = executor->ConvertToTensor(actualCmpSeqKvLenOptional, DataType::DT_INT64);
        const_cast<aclTensor*>(actualCmpSeqKvLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(actualCmpSeqKvLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(actualCmpSeqKvLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualCmpSeqKvLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor* actualSelSeqKvLen = nullptr;
    if (actualSelSeqKvLenOptional) {
        actualSelSeqKvLen = executor->ConvertToTensor(actualSelSeqKvLenOptional, DataType::DT_INT64);
        const_cast<aclTensor*>(actualSelSeqKvLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(actualSelSeqKvLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(actualSelSeqKvLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualSelSeqKvLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    if (topkMaskOptional == nullptr) {
        topkMaskOptional = executor->AllocTensor(DataType::DT_BOOL, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto softmaxMaxOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxSumOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    auto attentionOutOut = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto topkIndicesOutOut = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(NsaCompressAttention,
                           OP_INPUT(query, key, value, attenMaskOptional, actualSeqQLen, actualCmpSeqKvLen,
                                    actualSelSeqKvLen, topkMaskOptional),
                           OP_OUTPUT(softmaxMaxOut, softmaxSumOut, attentionOutOut, topkIndicesOutOut),
                           OP_ATTR(static_cast<float>(scaleValueOptional), headNum, inputLayout, sparseModeOptional,
                                   compressBlockSize, compressStride, selectBlockSize, selectBlockCount));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "NsaCompressAttention InferShape failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }

    ADD_TO_LAUNCHER_LIST_AICORE(
        NsaCompressAttention,
        OP_INPUT(query, key, value, attenMaskOptional, actualSeqQLen, actualCmpSeqKvLen, actualSelSeqKvLen,
                 topkMaskOptional),
        OP_OUTPUT(softmaxMaxOut, softmaxSumOut, attentionOutOut, topkIndicesOutOut),
        OP_ATTR(static_cast<float>(scaleValueOptional), headNum, inputLayout, sparseModeOptional, compressBlockSize,
                compressStride, selectBlockSize, selectBlockCount));
    return {softmaxMaxOut, softmaxSumOut, attentionOutOut, topkIndicesOutOut};
}

} // namespace l0op
