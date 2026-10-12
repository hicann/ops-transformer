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
 * \file rain_fusion_attention.cpp
 * \brief
 */

#include "rain_fusion_attention.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(RainFusionAttention);

static const aclTensor* ConvertIntArrayToTensor(const aclIntArray* intArray, aclOpExecutor* executor, DataType dtype)
{
    if (intArray != nullptr) {
        const aclTensor* tensor = executor->ConvertToTensor(intArray, dtype);
        auto mutableTensor = const_cast<aclTensor*>(tensor);
        mutableTensor->SetStorageFormat(Format::FORMAT_ND);
        mutableTensor->SetViewFormat(Format::FORMAT_ND);
        mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
        return tensor;
    }
    return executor->AllocTensor(dtype, Format::FORMAT_ND, Format::FORMAT_ND);
}

const std::array<const aclTensor*, 2> RainFusionAttention(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* selectIdx,
    const aclTensor* selectNumIdx, const aclIntArray* blockShape, const aclTensor* attenMaskOptional,
    const aclIntArray* actualSeqLengthsOptional, const aclIntArray* actualSeqLengthsKvOptional,
    const aclTensor* blockTableOptional, const char* qInputLayout, const char* kvInputLayout, int64_t numKeyValueHeads,
    int64_t maskType, double scaleValue, int64_t innerPrecise, int64_t blockSize, aclOpExecutor* executor)
{
    const char* safeKvInputLayout = (kvInputLayout != nullptr) ? kvInputLayout : qInputLayout;

    L0_DFX(RainFusionAttention, query, key, value, selectIdx, selectNumIdx, blockShape, attenMaskOptional,
           actualSeqLengthsOptional, actualSeqLengthsKvOptional, blockTableOptional, qInputLayout, safeKvInputLayout,
           numKeyValueHeads, maskType, scaleValue, innerPrecise, blockSize);

    const aclTensor* blockShapeTensor = ConvertIntArrayToTensor(blockShape, executor, DataType::DT_INT64);
    const aclTensor* attenMaskTensor =
        (attenMaskOptional != nullptr) ? attenMaskOptional :
                                         executor->AllocTensor(DataType::DT_BOOL, Format::FORMAT_ND, Format::FORMAT_ND);
    const aclTensor* actualSeqTensor = ConvertIntArrayToTensor(actualSeqLengthsOptional, executor, DataType::DT_INT64);
    const aclTensor* actualSeqKvTensor =
        ConvertIntArrayToTensor(actualSeqLengthsKvOptional, executor, DataType::DT_INT64);

    auto attentionOutTensor = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxLseTensor = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);

    // scaleValue is already float type, no need for cast
    auto ret = INFER_SHAPE(RainFusionAttention,
                           OP_INPUT(query, key, value, selectIdx, selectNumIdx, blockShapeTensor, attenMaskTensor,
                                    actualSeqTensor, actualSeqKvTensor, blockTableOptional),
                           OP_OUTPUT(attentionOutTensor, softmaxLseTensor),
                           OP_ATTR(qInputLayout, safeKvInputLayout, static_cast<uint32_t>(numKeyValueHeads),
                                   static_cast<uint32_t>(maskType), static_cast<float>(scaleValue),
                                   static_cast<uint32_t>(innerPrecise), static_cast<uint32_t>(blockSize)));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "RainFusionAttention infer shape failed, scaleValue: %f.", scaleValue);
        return {nullptr, nullptr};
    }

    ADD_TO_LAUNCHER_LIST_AICORE(RainFusionAttention,
                                OP_INPUT(query, key, value, selectIdx, selectNumIdx, blockShapeTensor, attenMaskTensor,
                                         actualSeqTensor, actualSeqKvTensor, blockTableOptional),
                                OP_OUTPUT(attentionOutTensor, softmaxLseTensor),
                                OP_ATTR(qInputLayout, safeKvInputLayout, static_cast<uint32_t>(numKeyValueHeads),
                                        static_cast<uint32_t>(maskType), static_cast<float>(scaleValue),
                                        static_cast<uint32_t>(innerPrecise), static_cast<uint32_t>(blockSize)));

    return {attentionOutTensor, softmaxLseTensor};
}
} // namespace l0op
