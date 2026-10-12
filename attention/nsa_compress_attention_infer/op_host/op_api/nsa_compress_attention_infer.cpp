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

OP_TYPE_REGISTER(NsaCompressAttentionInfer);

const std::array<const aclTensor*, 2> NsaCompressAttentionInfer(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* attentionMaskOptional,
    const aclTensor* blockTableOptional, const aclIntArray* actualQSeqLenOptional,
    const aclIntArray* actualCmpKvSeqLenOptional, const aclIntArray* actualSelKvSeqLenOptional,
    const aclTensor* topKMaskOptional, int64_t numHeads, int64_t numKeyValueHeads, int64_t selectBlockSize,
    int64_t selectBlockCount, int64_t compressBlockSize, int64_t compressBlockStride, double scaleValue,
    const char* layoutOptional, int64_t pageBlockSize, int64_t sparseMode, aclOpExecutor* executor)
{
    L0_DFX(NsaCompressAttentionInfer, query, key, value, attentionMaskOptional, blockTableOptional,
           actualQSeqLenOptional, actualCmpKvSeqLenOptional, actualSelKvSeqLenOptional, topKMaskOptional, numHeads,
           numKeyValueHeads, selectBlockSize, selectBlockCount, compressBlockSize, compressBlockStride, scaleValue,
           layoutOptional, pageBlockSize, sparseMode);

    if (attentionMaskOptional == nullptr) {
        attentionMaskOptional = executor->AllocTensor(DataType::DT_BOOL, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor* actualSeqQLen = nullptr;
    if (actualQSeqLenOptional) {
        actualSeqQLen = executor->ConvertToTensor(actualQSeqLenOptional, DataType::DT_INT64);
        const_cast<aclTensor*>(actualSeqQLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(actualSeqQLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(actualSeqQLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualSeqQLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor* actualCmpSeqKvLen = nullptr;
    if (actualCmpKvSeqLenOptional) {
        actualCmpSeqKvLen = executor->ConvertToTensor(actualCmpKvSeqLenOptional, DataType::DT_INT64);
        const_cast<aclTensor*>(actualCmpSeqKvLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(actualCmpSeqKvLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(actualCmpSeqKvLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualCmpSeqKvLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor* actualSelKvLen = nullptr;
    if (actualSelKvSeqLenOptional) {
        actualSelKvLen = executor->ConvertToTensor(actualSelKvSeqLenOptional, DataType::DT_INT64);
        const_cast<aclTensor*>(actualSelKvLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(actualSelKvLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor*>(actualSelKvLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualSelKvLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto output = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto topKOutput = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(
        NsaCompressAttentionInfer,
        OP_INPUT(query, key, value, attentionMaskOptional, blockTableOptional, actualSeqQLen, actualCmpSeqKvLen,
                 actualSelKvLen, topKMaskOptional),
        OP_OUTPUT(output, topKOutput),
        OP_ATTR(numHeads, numKeyValueHeads, selectBlockSize, selectBlockCount, compressBlockSize, compressBlockStride,
                static_cast<float>(scaleValue), layoutOptional, pageBlockSize, sparseMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "NsaCompressAttentionInfer InferShape failed.");
        return {nullptr, nullptr};
    }

    ADD_TO_LAUNCHER_LIST_AICORE(
        NsaCompressAttentionInfer,
        OP_INPUT(query, key, value, attentionMaskOptional, blockTableOptional, actualSeqQLen, actualCmpSeqKvLen,
                 actualSelKvLen, topKMaskOptional),
        OP_OUTPUT(output, topKOutput),
        OP_ATTR(numHeads, numKeyValueHeads, selectBlockSize, selectBlockCount, compressBlockSize, compressBlockStride,
                static_cast<float>(scaleValue), layoutOptional, pageBlockSize, sparseMode));
    return {output, topKOutput};
}

} // namespace l0op
