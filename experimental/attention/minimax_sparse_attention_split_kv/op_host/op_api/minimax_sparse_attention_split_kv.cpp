/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "minimax_sparse_attention_split_kv.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(MinimaxSparseAttentionSplitKv);

std::tuple<const aclTensor *, const aclTensor *> MinimaxSparseAttentionSplitKv(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *blockTable,
    const aclTensor *k2qRowPtr, const aclTensor *k2qQIndices, const aclTensor *k2qSlotIndices,
    const aclTensor *actualSeqLengths, const aclTensor *actualSeqLengthsKv, int64_t numKeyValueHeads, double scaleValue,
    int64_t blockSize, int64_t topK, int64_t innerPrecise, bool softmaxLseFlag, const char *inputLayout,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, aclOpExecutor *executor)
{
    L0_DFX(MinimaxSparseAttentionSplitKv, query, key, value, blockTable, k2qRowPtr, k2qQIndices, k2qSlotIndices,
           actualSeqLengths, actualSeqLengthsKv, numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise,
           softmaxLseFlag, inputLayout);

    if (executor == nullptr || attentionOut == nullptr || softmaxLse == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "MinimaxSparseAttentionSplitKv: executor/attentionOut/softmaxLse is nullptr.");
        return {nullptr, nullptr};
    }

    auto attentionOutTensor = executor->AllocTensor(attentionOut->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxLseTensor = executor->AllocTensor(softmaxLse->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    if (attentionOutTensor == nullptr || softmaxLseTensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MinimaxSparseAttentionSplitKv: alloc output tensors failed.");
        return {nullptr, nullptr};
    }

    auto ret = INFER_SHAPE(
        MinimaxSparseAttentionSplitKv,
        OP_INPUT(query, key, value, blockTable, k2qRowPtr, k2qQIndices, k2qSlotIndices, actualSeqLengths,
                 actualSeqLengthsKv),
        OP_OUTPUT(attentionOutTensor, softmaxLseTensor),
        OP_ATTR(static_cast<int64_t>(numKeyValueHeads), static_cast<float>(scaleValue), static_cast<int64_t>(blockSize),
                static_cast<int64_t>(topK), static_cast<int64_t>(innerPrecise), softmaxLseFlag, inputLayout));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MinimaxSparseAttentionSplitKv infer shape failed.");
        return {nullptr, nullptr};
    }

    ADD_TO_LAUNCHER_LIST_AICORE(
        MinimaxSparseAttentionSplitKv,
        OP_INPUT(query, key, value, blockTable, k2qRowPtr, k2qQIndices, k2qSlotIndices, actualSeqLengths,
                 actualSeqLengthsKv),
        OP_OUTPUT(attentionOutTensor, softmaxLseTensor),
        OP_ATTR(static_cast<int64_t>(numKeyValueHeads), static_cast<float>(scaleValue), static_cast<int64_t>(blockSize),
                static_cast<int64_t>(topK), static_cast<int64_t>(innerPrecise), softmaxLseFlag, inputLayout));

    return {attentionOutTensor, softmaxLseTensor};
}

} // namespace l0op
