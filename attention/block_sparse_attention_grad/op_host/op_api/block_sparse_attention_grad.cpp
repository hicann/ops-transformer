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
 * \file block_sparse_attention_grad.cpp
 * \brief
 */


#include "block_sparse_attention_grad.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BlockSparseAttentionGrad);

static const aclTensor *ConvertIntArrayToTensor(const aclIntArray *intArray,
                                                aclOpExecutor *executor,
                                                DataType dtype)
{
    if (intArray != nullptr) {
        const aclTensor *tensor = executor->ConvertToTensor(intArray, dtype);
        auto mutableTensor = const_cast<aclTensor *>(tensor);
        mutableTensor->SetStorageFormat(Format::FORMAT_ND);
        mutableTensor->SetViewFormat(Format::FORMAT_ND);
        mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
        return tensor;
    }
    return executor->AllocTensor(dtype, Format::FORMAT_ND, Format::FORMAT_ND);
}

const std::array<const aclTensor *, 3> BlockSparseAttentionGrad(
    const aclTensor *dout,
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLse,
    const aclTensor *blockSparseMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *blockShapeOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    char *qInputLayout,
    char *kvInputLayout,
    int64_t numKeyValueHeads,
    int64_t maskType,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    aclOpExecutor *executor)
{
    const char *safeKvInputLayout = (kvInputLayout != nullptr) ? kvInputLayout : qInputLayout;
    
    L0_DFX(BlockSparseAttentionGrad, dout, query, key, value, attentionOut, softmaxLse, blockSparseMaskOptional, 
           attenMaskOptional, blockShapeOptional,actualSeqLengthsOptional, actualSeqLengthsKvOptional, qInputLayout, safeKvInputLayout,
           numKeyValueHeads, maskType, scaleValue, preTokens, nextTokens);
    const aclTensor *attenMaskTensor = (attenMaskOptional != nullptr) ? attenMaskOptional :
                                       executor->AllocTensor(DataType::DT_BOOL, Format::FORMAT_ND, Format::FORMAT_ND);
    const aclTensor *blockShapeTensor = ConvertIntArrayToTensor(blockShapeOptional, executor, DataType::DT_INT64);
    const aclTensor *actualSeqTensor = ConvertIntArrayToTensor(actualSeqLengthsOptional, executor, DataType::DT_INT64);
    const aclTensor *actualSeqKvTensor = ConvertIntArrayToTensor(actualSeqLengthsKvOptional, executor, DataType::DT_INT64);

    auto dqTensor = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto dkTensor = executor->AllocTensor(key->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto dvTensor = executor->AllocTensor(value->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);

    // scaleValue is already float type, no need for cast
    auto ret = INFER_SHAPE(BlockSparseAttentionGrad,
                           OP_INPUT(dout, query, key, value, attentionOut, softmaxLse, blockSparseMaskOptional,
                                    attenMaskTensor,  blockShapeTensor,actualSeqTensor, actualSeqKvTensor),
                           OP_OUTPUT(dqTensor, dkTensor, dvTensor),
                           OP_ATTR(qInputLayout, safeKvInputLayout,
                                   static_cast<uint32_t>(numKeyValueHeads), static_cast<uint32_t>(maskType),
                                   static_cast<float>(scaleValue), static_cast<uint32_t>(preTokens),
                                   static_cast<uint32_t>(nextTokens)));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockSparseAttentionGrad infer shape failed, scaleValue: %f.", scaleValue);
        return {nullptr, nullptr};
    }

    ADD_TO_LAUNCHER_LIST_AICORE(BlockSparseAttentionGrad,
                                OP_INPUT(dout, query, key, value, attentionOut, softmaxLse, blockSparseMaskOptional,
                                         attenMaskTensor, blockShapeTensor,actualSeqTensor, actualSeqKvTensor),
                                OP_OUTPUT(dqTensor, dkTensor, dvTensor),
                                OP_ATTR(qInputLayout, safeKvInputLayout, static_cast<uint32_t>(numKeyValueHeads),
                                        static_cast<uint32_t>(maskType), static_cast<float>(scaleValue),
                                        static_cast<uint32_t>(preTokens), static_cast<uint32_t>(nextTokens)));

    return {dqTensor, dkTensor, dvTensor};
}
} // namespace l0op


