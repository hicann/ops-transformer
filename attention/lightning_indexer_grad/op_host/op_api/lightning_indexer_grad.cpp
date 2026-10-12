/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_grad.cpp
 * \brief
 */

#include "lightning_indexer_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/format_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

#define CHECK_NULL(aclnTensor) \
    do { \
        if ((aclnTensor) == nullptr) { \
            return {nullptr, nullptr, nullptr}; \
        } \
    } while (0)

namespace l0op {
OP_TYPE_REGISTER(LightningIndexerGrad);

const std::array<const aclTensor*, LIGHTNING_INDEXER_GRAD_OUTPUT_CNT> LightningIndexerGrad(
    const aclTensor* query, const aclTensor* key, const aclTensor* dy, const aclTensor* sparseIndices,
    const aclTensor* weights, const aclTensor* actualSeqQLenOptional, const aclTensor* actualSeqKvLenOptional,
    int64_t headNum, const char* inputLayout, int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    bool deterministic, aclOpExecutor* executor)
{
    L0_DFX(LightningIndexerGrad, query, key, dy, sparseIndices, weights, actualSeqQLenOptional, actualSeqKvLenOptional,
           headNum, inputLayout, sparseMode, preTokens, nextTokens, deterministic);
    DataType outputDtype = query->GetDataType();
    auto dqOut = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto dkOut = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto dweightsOut = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);

    auto ret =
        INFER_SHAPE(LightningIndexerGrad,
                    OP_INPUT(query, key, dy, sparseIndices, weights, actualSeqQLenOptional, actualSeqKvLenOptional),
                    OP_OUTPUT(dqOut, dkOut, dweightsOut),
                    OP_ATTR(headNum, inputLayout, sparseMode, preTokens, nextTokens, deterministic));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "LightningIndexerGrad InferShape failed.");
        return {nullptr, nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        LightningIndexerGrad,
        OP_INPUT(query, key, dy, sparseIndices, weights, actualSeqQLenOptional, actualSeqKvLenOptional),
        OP_OUTPUT(dqOut, dkOut, dweightsOut),
        OP_ATTR(headNum, inputLayout, sparseMode, preTokens, nextTokens, deterministic));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "LightningIndexerGrad launch kernel failed.");
        return {nullptr, nullptr, nullptr};
    }
    return {dqOut, dkOut, dweightsOut};
}

} // namespace l0op
