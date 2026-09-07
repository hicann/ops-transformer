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
 * \file generic_block_sparse_attention_grad.cpp
 * \brief L0 AICore launcher for GenericBlockSparseAttentionGrad.
 */

#include "generic_block_sparse_attention_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(GenericBlockSparseAttentionGrad);

const std::array<const aclTensor *, 3> GenericBlockSparseAttentionGrad(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *dout, const aclTensor *out,
    const aclTensor *lse, const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount, const aclTensor *metadata,
    const aclTensor *attenMaskOptional, const aclTensor *cuSeqLengthsQOptional, const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, const aclIntArray *blockShape,
    int64_t isPackedGQA, char *layoutQ, char *layoutKv, double scaleValue, int64_t maskType, int64_t softmaxPrecision,
    int64_t winLeft, int64_t winRight, aclOpExecutor *executor)
{
    const char *safeKvLayout = (layoutKv != nullptr) ? layoutKv : layoutQ;
    L0_DFX(GenericBlockSparseAttentionGrad, query, key, value, dout, out, lse, sparseBlockIdx, sparseBlockCount,
           metadata, attenMaskOptional, cuSeqLengthsQOptional, cuSeqLengthsKvOptional, blockShape, isPackedGQA, layoutQ,
           safeKvLayout, scaleValue, maskType, softmaxPrecision, winLeft, winRight);

    const aclTensor *attenMaskTensor =
        (attenMaskOptional != nullptr) ? attenMaskOptional :
                                         executor->AllocTensor(DataType::DT_BOOL, Format::FORMAT_ND, Format::FORMAT_ND);
    const aclTensor *cuSeqTensor = (cuSeqLengthsQOptional != nullptr) ?
                                       cuSeqLengthsQOptional :
                                       executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    const aclTensor *cuSeqKvTensor =
        (cuSeqLengthsKvOptional != nullptr) ?
            cuSeqLengthsKvOptional :
            executor->AllocTensor(DataType::DT_INT64, Format::FORMAT_ND, Format::FORMAT_ND);
    const aclTensor *sequsedQTensor =
        (sequsedQOptional != nullptr) ? sequsedQOptional :
                                        executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    const aclTensor *sequsedKvTensor =
        (sequsedKvOptional != nullptr) ?
            sequsedKvOptional :
            executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);

    const int64_t defaultBlockShape[] = {1, 128};
    const aclIntArray *blockShapeAttr =
        (blockShape != nullptr) ? blockShape : executor->AllocIntArray(defaultBlockShape, 2);

    auto dQuery = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto dKey = executor->AllocTensor(key->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto dValue = executor->AllocTensor(value->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(
        GenericBlockSparseAttentionGrad,
        OP_INPUT(query, key, value, dout, out, lse, sparseBlockIdx, sparseBlockCount, metadata, attenMaskTensor,
                 cuSeqTensor, cuSeqKvTensor, sequsedQTensor, sequsedKvTensor),
        OP_OUTPUT(dQuery, dKey, dValue),
        OP_ATTR(blockShapeAttr, static_cast<int64_t>(isPackedGQA), layoutQ, safeKvLayout,
                static_cast<float>(scaleValue), static_cast<int64_t>(maskType), static_cast<int64_t>(softmaxPrecision),
                static_cast<int64_t>(winLeft), static_cast<int64_t>(winRight)));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GenericBlockSparseAttentionGrad infer shape failed.");
        return {nullptr, nullptr, nullptr};
    }

    ADD_TO_LAUNCHER_LIST_AICORE(
        GenericBlockSparseAttentionGrad,
        OP_INPUT(query, key, value, dout, out, lse, sparseBlockIdx, sparseBlockCount, metadata, attenMaskTensor,
                 cuSeqTensor, cuSeqKvTensor, sequsedQTensor, sequsedKvTensor),
        OP_OUTPUT(dQuery, dKey, dValue),
        OP_ATTR(blockShapeAttr, static_cast<int64_t>(isPackedGQA), layoutQ, safeKvLayout,
                static_cast<float>(scaleValue), static_cast<int64_t>(maskType), static_cast<int64_t>(softmaxPrecision),
                static_cast<int64_t>(winLeft), static_cast<int64_t>(winRight)));

    return {dQuery, dKey, dValue};
}

} // namespace l0op
