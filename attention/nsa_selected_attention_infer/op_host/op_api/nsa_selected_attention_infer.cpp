/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nsa_selected_attention_infer.cpp
 * \brief
 */

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(NsaSelectedAttentionInfer);

const aclTensor* NsaSelectedAttentionInfer(
    const aclTensor *query, 
    const aclTensor *key, 
    const aclTensor *value, 
    const aclTensor *topkIndices, 
    const aclTensor *attenMaskOptional,
    const aclTensor *blockTableOptional,
    const aclIntArray *actualQSeqLenOptional,
    const aclIntArray *actualKvSeqLenOptional,
    char *layoutOptional,
    int64_t numHeads,    
    int64_t numKeyValueHeads,
    int64_t selectBlockSize,
    int64_t selectBlockCount,
    int64_t pageBlockSize,
    double scaleValue,
    int64_t sparseMode,
    aclOpExecutor *executor)
{
    L0_DFX(NsaSelectedAttentionInfer, query, key, value, topkIndices, attenMaskOptional, blockTableOptional,
            actualQSeqLenOptional, actualKvSeqLenOptional, layoutOptional, numHeads,
            numKeyValueHeads, selectBlockSize, selectBlockCount, pageBlockSize, scaleValue, sparseMode);

    if (attenMaskOptional == nullptr) {
        attenMaskOptional = executor->AllocTensor(DataType::DT_BOOL, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *actualSeqQLen = nullptr;
    if (actualQSeqLenOptional) {
        actualSeqQLen = executor->ConvertToTensor(actualQSeqLenOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualSeqQLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqQLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqQLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualSeqQLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    const aclTensor *actualSeqKvLen = nullptr;
    if (actualKvSeqLenOptional) {
        actualSeqKvLen = executor->ConvertToTensor(actualKvSeqLenOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualSeqKvLen)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqKvLen)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqKvLen)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualSeqKvLen = executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto attentionOut = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(NsaSelectedAttentionInfer,
                        OP_INPUT(query, key, value, topkIndices, attenMaskOptional, blockTableOptional,
                                    actualSeqQLen, actualSeqKvLen),
                        OP_OUTPUT(attentionOut),
                        OP_ATTR(layoutOptional, numHeads, numKeyValueHeads, selectBlockSize, selectBlockCount,
                                pageBlockSize, static_cast<float>(scaleValue), sparseMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "NsaSelectedAttentionInfer InferShape failed.");
        return nullptr;
    }

    ADD_TO_LAUNCHER_LIST_AICORE(NsaSelectedAttentionInfer,
                                OP_INPUT(query, key, value, topkIndices, attenMaskOptional, blockTableOptional,
                                        actualSeqQLen, actualSeqKvLen),
                                OP_OUTPUT(attentionOut),
                                OP_ATTR(layoutOptional, numHeads, numKeyValueHeads, selectBlockSize, selectBlockCount,
                                pageBlockSize, static_cast<float>(scaleValue), sparseMode));
    return attentionOut;
}

} // namespace l0op
 