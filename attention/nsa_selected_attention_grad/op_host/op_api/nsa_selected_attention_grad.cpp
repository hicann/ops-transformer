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
 * \file nsa_selected_attention_grad.cpp
 * \brief
 */

#include "nsa_selected_attention_grad.h"
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
OP_TYPE_REGISTER(NsaSelectedAttentionGrad);

const std::array<const aclTensor*, NSA_GRAD_OUTPUT_CNT> NsaSelectedAttentionGrad(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* attentionOut,
    const aclTensor* attentionOutGrad, const aclTensor* softmaxMax, const aclTensor* softmaxSum,
    const aclTensor* topkIndices, const aclIntArray* actualSeqQLenOptional, const aclIntArray* actualSeqKVLlenOptional,
    const aclTensor* attenMskOptional, double scaleValue, int64_t selectedBlockCount, int64_t selectedBlockSize,
    int64_t headNum, const char* inputLayout, int64_t sparseMode, aclOpExecutor* executor)
{
    L0_DFX(NsaSelectedAttentionGrad, query, key, value, attentionOut, attentionOutGrad, softmaxMax, softmaxSum,
           topkIndices, actualSeqQLenOptional, actualSeqKVLlenOptional, attenMskOptional, scaleValue,
           selectedBlockCount, selectedBlockSize, headNum, inputLayout, sparseMode);
    DataType outputDtype = query->GetDataType();
    auto dqOut = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto dkOut = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto dvOut = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);

    const aclTensor* actualSeqQLen = nullptr;
    if (actualSeqQLenOptional) {
        actualSeqQLen = executor->ConvertToTensor(actualSeqQLenOptional, op::DataType::DT_INT64);
        CHECK_NULL(actualSeqQLen);
        const_cast<aclTensor*>(actualSeqQLen)->SetStorageFormat(op::Format::FORMAT_ND);
        const_cast<aclTensor*>(actualSeqQLen)->SetViewFormat(op::Format::FORMAT_ND);
        const_cast<aclTensor*>(actualSeqQLen)->SetOriginalFormat(op::Format::FORMAT_ND);
    }

    const aclTensor* actualSeqKvLen = nullptr;
    if (actualSeqKVLlenOptional) {
        actualSeqKvLen = executor->ConvertToTensor(actualSeqKVLlenOptional, op::DataType::DT_INT64);
        CHECK_NULL(actualSeqKvLen);
        const_cast<aclTensor*>(actualSeqKvLen)->SetStorageFormat(op::Format::FORMAT_ND);
        const_cast<aclTensor*>(actualSeqKvLen)->SetViewFormat(op::Format::FORMAT_ND);
        const_cast<aclTensor*>(actualSeqKvLen)->SetOriginalFormat(op::Format::FORMAT_ND);
    }

    auto ret = INFER_SHAPE(NsaSelectedAttentionGrad,
                           OP_INPUT(query, key, value, attentionOut, attentionOutGrad, softmaxMax, softmaxSum,
                                    topkIndices, actualSeqQLen, actualSeqKvLen, attenMskOptional),
                           OP_OUTPUT(dqOut, dkOut, dvOut),
                           OP_ATTR(static_cast<float>(scaleValue), selectedBlockCount, selectedBlockSize, headNum,
                                   inputLayout, sparseMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "NsaSelectedAttentionGrad InferShape failed.");
        return {nullptr, nullptr, nullptr};
    }

    ret =
        ADD_TO_LAUNCHER_LIST_AICORE(NsaSelectedAttentionGrad,
                                    OP_INPUT(query, key, value, attentionOut, attentionOutGrad, softmaxMax, softmaxSum,
                                             topkIndices, actualSeqQLen, actualSeqKvLen, attenMskOptional),
                                    OP_OUTPUT(dqOut, dkOut, dvOut),
                                    OP_ATTR(static_cast<float>(scaleValue), selectedBlockCount, selectedBlockSize,
                                            headNum, inputLayout, sparseMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "NsaSelectedAttentionGrad launch kernel failed.");
        return {nullptr, nullptr, nullptr};
    }
    return {dqOut, dkOut, dvOut};
}

} // namespace l0op
