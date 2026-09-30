/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_API_CHECK_UTILS_H
#define GROUPED_MATMUL_SWIGLU_QUANT_API_CHECK_UTILS_H

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "opdev/op_log.h"

namespace gmm_swiglu_quant_api_check {

inline aclnnStatus CheckRequiredPointer(const void* pointer, const char* apiName, const char* parameterName)
{
    if (pointer != nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(apiName, parameterName, "does not support nullptr");
    return ACLNN_ERR_PARAM_NULLPTR;
}

// expectedSize == 0 preserves the V2 contract (a non-empty list); V3 requires exactly one tensor.
inline aclnnStatus CheckRequiredTensorList(const aclTensorList* tensorList, const char* apiName,
                                           const char* parameterName, size_t expectedSize = 0)
{
    auto status = CheckRequiredPointer(tensorList, apiName, parameterName);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (expectedSize == 0 && tensorList->Size() == 0) {
        OP_LOGE_FOR_INVALID_TENSORNUM(apiName, parameterName, tensorList->Size(), "at least 1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t i = 0; i < tensorList->Size(); ++i) {
        if ((*tensorList)[i] == nullptr) {
            const std::string indexedName = std::string(parameterName) + "[" + std::to_string(i) + "]";
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(apiName, indexedName.c_str(), "does not support nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    if (expectedSize != 0 && tensorList->Size() != expectedSize) {
        OP_LOGE_FOR_INVALID_TENSORNUM(apiName, parameterName, tensorList->Size(), "exactly 1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

inline aclnnStatus CheckRequiredInputs(const char* apiName, const aclTensor* x, const aclTensorList* weight,
                                       const aclTensorList* weightScale, const aclTensor* xScale,
                                       const aclTensor* groupList, const aclTensor* output,
                                       const aclTensor* outputScale, size_t expectedWeightSize = 0)
{
    auto status = CheckRequiredPointer(x, apiName, "x");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckRequiredTensorList(weight, apiName, "weight", expectedWeightSize);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckRequiredTensorList(weightScale, apiName, "weightScale", expectedWeightSize);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    const struct {
        const void* pointer;
        const char* name;
    } requiredInputs[] = {
        {xScale, "xScale"}, {groupList, "groupList"}, {output, "output"}, {outputScale, "outputScale"}};
    for (const auto& input : requiredInputs) {
        status = CheckRequiredPointer(input.pointer, apiName, input.name);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    return ACLNN_SUCCESS;
}

} // namespace gmm_swiglu_quant_api_check

#endif // GROUPED_MATMUL_SWIGLU_QUANT_API_CHECK_UTILS_H
