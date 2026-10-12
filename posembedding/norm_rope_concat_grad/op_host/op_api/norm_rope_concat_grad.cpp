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
 * \file norm_rope_concat_grad.cpp
 * \brief
 */

#include "norm_rope_concat_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/format_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(NormRopeConcatGrad);

const std::array<const aclTensor*, 14> NormRopeConcatGrad(
    const aclTensor* gradQueryOutput, const aclTensor* gradKeyOutput, const aclTensor* gradValueOutput,
    const aclTensor* query, const aclTensor* key, const aclTensor* encoderQuery, const aclTensor* encoderKey,
    const aclTensor* normQueryWeight, const aclTensor* normQueryMean, const aclTensor* normQueryRstd,
    const aclTensor* normKeyWeight, const aclTensor* normKeyMean, const aclTensor* normKeyRstd,
    const aclTensor* normAddedQueryWeight, const aclTensor* normAddedQueryMean, const aclTensor* normAddedQueryRstd,
    const aclTensor* normAddedKeyWeight, const aclTensor* normAddedKeyMean, const aclTensor* normAddedKeyRstd,
    const aclTensor* ropeSin, const aclTensor* ropeCos, int64_t normType, int64_t normAddedType, int64_t ropeType,
    int64_t concatOrder, aclOpExecutor* executor)
{
    L0_DFX(NormRopeConcatGrad, gradQueryOutput, gradKeyOutput, gradValueOutput, query, key, encoderQuery, encoderKey,
           normQueryWeight, normQueryMean, normQueryRstd, normKeyWeight, normKeyMean, normKeyRstd, normAddedQueryWeight,
           normAddedQueryMean, normAddedQueryRstd, normAddedKeyWeight, normAddedKeyMean, normAddedKeyRstd, ropeSin,
           ropeCos, normType, normAddedType, ropeType, concatOrder);

    DataType outputDtype = gradQueryOutput->GetDataType();
    auto gradQuery = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto gradKey = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto gradValue = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto gradEncoderQuery = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto gradEncoderKey = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto gradEncoderValue = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto gradNormQueryWeight = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto gradNormQueryBias = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto gradNormKeyWeight = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto gradNormKeyBias = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto gradNormAddedQueryWeight = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto gradNormAddedQueryBias = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto gradNormAddedKeyWeight = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto gradNormAddedKeyBias = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);

    auto ret = INFER_SHAPE(
        NormRopeConcatGrad,
        OP_INPUT(gradQueryOutput, gradKeyOutput, gradValueOutput, query, key, encoderQuery, encoderKey, normQueryWeight,
                 normQueryMean, normQueryRstd, normKeyWeight, normKeyMean, normKeyRstd, normAddedQueryWeight,
                 normAddedQueryMean, normAddedQueryRstd, normAddedKeyWeight, normAddedKeyMean, normAddedKeyRstd,
                 ropeSin, ropeCos),
        OP_OUTPUT(gradQuery, gradKey, gradValue, gradEncoderQuery, gradEncoderKey, gradEncoderValue,
                  gradNormQueryWeight, gradNormQueryBias, gradNormKeyWeight, gradNormKeyBias, gradNormAddedQueryWeight,
                  gradNormAddedQueryBias, gradNormAddedKeyWeight, gradNormAddedKeyBias),
        OP_ATTR(normType, normAddedType, ropeType, concatOrder));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "NormRopeConcatGrad InferShape failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        NormRopeConcatGrad,
        OP_INPUT(gradQueryOutput, gradKeyOutput, gradValueOutput, query, key, encoderQuery, encoderKey, normQueryWeight,
                 normQueryMean, normQueryRstd, normKeyWeight, normKeyMean, normKeyRstd, normAddedQueryWeight,
                 normAddedQueryMean, normAddedQueryRstd, normAddedKeyWeight, normAddedKeyMean, normAddedKeyRstd,
                 ropeSin, ropeCos),
        OP_OUTPUT(gradQuery, gradKey, gradValue, gradEncoderQuery, gradEncoderKey, gradEncoderValue,
                  gradNormQueryWeight, gradNormQueryBias, gradNormKeyWeight, gradNormKeyBias, gradNormAddedQueryWeight,
                  gradNormAddedQueryBias, gradNormAddedKeyWeight, gradNormAddedKeyBias),
        OP_ATTR(normType, normAddedType, ropeType, concatOrder));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "NormRopeConcatGrad launch kernel failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {gradQuery,
            gradKey,
            gradValue,
            gradEncoderQuery,
            gradEncoderKey,
            gradEncoderValue,
            gradNormQueryWeight,
            gradNormQueryBias,
            gradNormKeyWeight,
            gradNormKeyBias,
            gradNormAddedQueryWeight,
            gradNormAddedQueryBias,
            gradNormAddedKeyWeight,
            gradNormAddedKeyBias};
}

} // namespace l0op
