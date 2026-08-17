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
 * \file aclnn_norm_rope_concat_grad.cpp
 * \brief
 */

#include "acl/acl.h"
#include "norm_rope_concat_grad.h"
#include "../norm_rope_concat_grad_base_host.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_norm_rope_concat_grad.h"

using namespace op;
using namespace NormRopeConcatGrad;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                       DataType::DT_BF16};

static inline bool CheckNotNull(const aclTensor *gradQueryOutput, const aclTensor *gradKeyOutput,
                                const aclTensor *gradValueOutput, const aclTensor *query, const aclTensor *key,
                                const aclTensor *gradQuery, const aclTensor *gradKey, const aclTensor *gradValue)
{
    OP_CHECK_NULL(gradQueryOutput, return false);
    OP_CHECK_NULL(gradKeyOutput, return false);
    OP_CHECK_NULL(gradValueOutput, return false);
    OP_CHECK_NULL(query, return false);
    OP_CHECK_NULL(key, return false);
    OP_CHECK_NULL(gradQuery, return false);
    OP_CHECK_NULL(gradKey, return false);
    OP_CHECK_NULL(gradValue, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *gradQueryOutput, const aclTensor *gradKeyOutput,
                            const aclTensor *gradValueOutput, const aclTensor *query, const aclTensor *key,
                            const aclTensor *encoderQuery, const aclTensor *encoderKey,
                            const aclTensor *normQueryWeight, const aclTensor *normKeyWeight,
                            const aclTensor *normAddedQueryWeight, const aclTensor *normAddedKeyWeight,
                            const aclTensor *ropeSin, const aclTensor *ropeCos, const aclTensor *gradQuery,
                            const aclTensor *gradKey, const aclTensor *gradValue, const aclTensor *gradEncoderQuery,
                            const aclTensor *gradEncoderKey, const aclTensor *gradEncoderValue,
                            const aclTensor *gradNormQueryWeight, const aclTensor *gradNormQueryBias,
                            const aclTensor *gradNormKeyWeight, const aclTensor *gradNormKeyBias,
                            const aclTensor *gradNormAddedQueryWeight, const aclTensor *gradNormAddedQueryBias,
                            const aclTensor *gradNormAddedKeyWeight, const aclTensor *gradNormAddedKeyBias)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradQueryOutput, DTYPE_SUPPORT_LIST, return false);

    DataType mainDtype = gradQueryOutput->GetDataType();
    OP_CHECK(gradKeyOutput->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input gradKeyOutput dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(gradValueOutput->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input gradValueOutput dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(query->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input query dtype should be same with gradQueryOutput."), return false);
    OP_CHECK(key->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input key dtype should be same with gradQueryOutput."), return false);
    OP_CHECK(encoderQuery == nullptr || encoderQuery->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input encoderQuery dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(encoderKey == nullptr || encoderKey->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input encoderKey dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(normQueryWeight == nullptr || normQueryWeight->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input normQueryWeight dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(normKeyWeight == nullptr || normKeyWeight->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input normKeyWeight dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(normAddedQueryWeight == nullptr || normAddedQueryWeight->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input normAddedQueryWeight dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(normAddedKeyWeight == nullptr || normAddedKeyWeight->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input normAddedKeyWeight dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(ropeSin == nullptr || ropeSin->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input ropeSin dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(ropeCos == nullptr || ropeCos->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input ropeCos dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(gradQuery == nullptr || gradQuery->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output gradQuery dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(gradKey == nullptr || gradKey->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output gradKey dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(gradValue == nullptr || gradValue->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output gradValue dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(gradEncoderQuery == nullptr || gradEncoderQuery->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output gradEncoderQuery dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(gradEncoderKey == nullptr || gradEncoderKey->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output gradEncoderKey dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(gradEncoderValue == nullptr || gradEncoderValue->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output gradEncoderValue dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(gradNormQueryWeight == nullptr || gradNormQueryWeight->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output gradNormQueryWeight dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(gradNormQueryBias == nullptr || gradNormQueryBias->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output gradNormQueryBias dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(gradNormKeyWeight == nullptr || gradNormKeyWeight->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output gradNormKeyWeight dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(gradNormKeyBias == nullptr || gradNormKeyBias->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output gradNormKeyBias dtype should be same with gradQueryOutput."),
             return false);
    OP_CHECK(
        gradNormAddedQueryWeight == nullptr || gradNormAddedQueryWeight->GetDataType() == mainDtype,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output gradNormAddedQueryWeight dtype should be same with gradQueryOutput."),
        return false);
    OP_CHECK(
        gradNormAddedQueryBias == nullptr || gradNormAddedQueryBias->GetDataType() == mainDtype,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output gradNormAddedQueryBias dtype should be same with gradQueryOutput."),
        return false);
    OP_CHECK(
        gradNormAddedKeyWeight == nullptr || gradNormAddedKeyWeight->GetDataType() == mainDtype,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output gradNormAddedKeyWeight dtype should be same with gradQueryOutput."),
        return false);
    OP_CHECK(gradNormAddedKeyBias == nullptr || gradNormAddedKeyBias->GetDataType() == mainDtype,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output gradNormAddedKeyBias dtype should be same with gradQueryOutput."),
             return false);
    return true;
}

static bool CheckNormDtypeValid(const aclTensor *normQueryMean, const aclTensor *normQueryRstd,
                                const aclTensor *normKeyMean, const aclTensor *normKeyRstd,
                                const aclTensor *normAddedQueryMean, const aclTensor *normAddedQueryRstd,
                                const aclTensor *normAddedKeyMean, const aclTensor *normAddedKeyRstd)
{
    OP_CHECK(normQueryMean == nullptr || normQueryMean->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input normQueryMean dtype should be DT_FLOAT."), return false);
    OP_CHECK(normQueryRstd == nullptr || normQueryRstd->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input normQueryRstd dtype should be DT_FLOAT."), return false);
    OP_CHECK(normKeyMean == nullptr || normKeyMean->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input normKeyMean dtype should be DT_FLOAT."), return false);
    OP_CHECK(normKeyRstd == nullptr || normKeyRstd->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input normKeyRstd dtype should be DT_FLOAT."), return false);
    OP_CHECK(normAddedQueryMean == nullptr || normAddedQueryMean->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input normAddedQueryMean dtype should be DT_FLOAT."), return false);
    OP_CHECK(normAddedQueryRstd == nullptr || normAddedQueryRstd->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input normAddedQueryRstd dtype should be DT_FLOAT."), return false);
    OP_CHECK(normAddedKeyMean == nullptr || normAddedKeyMean->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input normAddedKeyMean dtype should be DT_FLOAT."), return false);
    OP_CHECK(normAddedKeyRstd == nullptr || normAddedKeyRstd->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input normAddedKeyRstd dtype should be DT_FLOAT."), return false);
    return true;
}

inline static aclnnStatus CheckParam(const aclTensor *gradQueryOutput, const aclTensor *gradKeyOutput, const aclTensor *gradValueOutput,
                                     const aclTensor *query, const aclTensor *key, const aclTensor *encoderQuery, const aclTensor *encoderKey,
                                     const aclTensor *normQueryWeight, const aclTensor *normQueryMean, const aclTensor *normQueryRstd,
                                     const aclTensor *normKeyWeight, const aclTensor *normKeyMean, const aclTensor *normKeyRstd,
                                     const aclTensor *normAddedQueryWeight, const aclTensor *normAddedQueryMean,
                                     const aclTensor *normAddedQueryRstd, const aclTensor *normAddedKeyWeight, const aclTensor *normAddedKeyMean,
                                     const aclTensor *normAddedKeyRstd, const aclTensor *ropeSin, const aclTensor *ropeCos,
                                     const aclTensor *gradQuery, const aclTensor *gradKey, const aclTensor *gradValue,
                                     const aclTensor *gradEncoderQuery, const aclTensor *gradEncoderKey, const aclTensor *gradEncoderValue,
                                     const aclTensor *gradNormQueryWeight, const aclTensor *gradNormQueryBias, const aclTensor *gradNormKeyWeight,
                                     const aclTensor *gradNormKeyBias, const aclTensor *gradNormAddedQueryWeight,
                                     const aclTensor *gradNormAddedQueryBias, const aclTensor *gradNormAddedKeyWeight,
                                     const aclTensor *gradNormAddedKeyBias)
{
    // check nullptr
    CHECK_RET(CheckNotNull(gradQueryOutput, gradKeyOutput, gradValueOutput, query, key, gradQuery, gradKey, gradValue),
              ACLNN_ERR_PARAM_NULLPTR);

    // check dtype
    CHECK_RET(CheckDtypeValid(gradQueryOutput, gradKeyOutput, gradValueOutput, query, key, encoderQuery, encoderKey,
                              normQueryWeight, normKeyWeight, normAddedQueryWeight, normAddedKeyWeight, ropeSin,
                              ropeCos, gradQuery, gradKey, gradValue, gradEncoderQuery, gradEncoderKey,
                              gradEncoderValue, gradNormQueryWeight, gradNormQueryBias, gradNormKeyWeight,
                              gradNormKeyBias, gradNormAddedQueryWeight, gradNormAddedQueryBias, gradNormAddedKeyWeight,
                              gradNormAddedKeyBias),
              ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckNormDtypeValid(normQueryMean, normQueryRstd, normKeyMean, normKeyRstd, normAddedQueryMean,
                                  normAddedQueryRstd, normAddedKeyMean, normAddedKeyRstd),
              ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

bool checkViewCopyResult(const aclTensor *l0Tensor, const aclTensor *inTensor, aclOpExecutor *l0Executor)
{
    if (l0Tensor != nullptr) {
        auto viewCopyResult = l0op::ViewCopy(l0Tensor, inTensor, l0Executor);
        CHECK_RET(viewCopyResult != nullptr, false);
    }
    return true;
}

aclnnStatus aclnnNormRopeConcatBackwardGetWorkspaceSize(
    const aclTensor *gradQueryOutput, const aclTensor *gradKeyOutput, const aclTensor *gradValueOutput,
    const aclTensor *query, const aclTensor *key, const aclTensor *encoderQuery, const aclTensor *encoderKey,
    const aclTensor *normQueryWeight, const aclTensor *normQueryMean, const aclTensor *normQueryRstd,
    const aclTensor *normKeyWeight, const aclTensor *normKeyMean, const aclTensor *normKeyRstd,
    const aclTensor *normAddedQueryWeight, const aclTensor *normAddedQueryMean, const aclTensor *normAddedQueryRstd,
    const aclTensor *normAddedKeyWeight, const aclTensor *normAddedKeyMean, const aclTensor *normAddedKeyRstd,
    const aclTensor *ropeSin, const aclTensor *ropeCos, int64_t normType, int64_t normAddedType, int64_t ropeType,
    int64_t concatOrder, const aclTensor *gradQuery, const aclTensor *gradKey, const aclTensor *gradValue,
    const aclTensor *gradEncoderQuery, const aclTensor *gradEncoderKey, const aclTensor *gradEncoderValue,
    const aclTensor *gradNormQueryWeight, const aclTensor *gradNormQueryBias, const aclTensor *gradNormKeyWeight,
    const aclTensor *gradNormKeyBias, const aclTensor *gradNormAddedQueryWeight,
    const aclTensor *gradNormAddedQueryBias, const aclTensor *gradNormAddedKeyWeight,
    const aclTensor *gradNormAddedKeyBias, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnNormRopeConcatBackward,
        DFX_IN(gradQueryOutput, gradKeyOutput, gradValueOutput, query, key, encoderQuery, encoderKey, normQueryWeight,
               normQueryMean, normQueryRstd, normKeyWeight, normKeyMean, normKeyRstd, normAddedQueryWeight,
               normAddedQueryMean, normAddedQueryRstd, normAddedKeyWeight, normAddedKeyMean, normAddedKeyRstd, ropeSin,
               ropeCos, normType, normAddedType, ropeType, concatOrder),
        DFX_OUT(gradQuery, gradKey, gradValue, gradEncoderQuery, gradEncoderKey, gradEncoderValue, gradNormQueryWeight,
                gradNormQueryBias, gradNormKeyWeight, gradNormKeyBias, gradNormAddedQueryWeight, gradNormAddedQueryBias,
                gradNormAddedKeyWeight, gradNormAddedKeyBias));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // check input tensor
    auto ret =
        CheckParam(gradQueryOutput, gradKeyOutput, gradValueOutput, query, key, encoderQuery, encoderKey,
                   normQueryWeight, normQueryMean, normQueryRstd, normKeyWeight, normKeyMean, normKeyRstd,
                   normAddedQueryWeight, normAddedQueryMean, normAddedQueryRstd, normAddedKeyWeight, normAddedKeyMean,
                   normAddedKeyRstd, ropeSin, ropeCos, gradQuery, gradKey, gradValue, gradEncoderQuery, gradEncoderKey,
                   gradEncoderValue, gradNormQueryWeight, gradNormQueryBias, gradNormKeyWeight, gradNormKeyBias,
                   gradNormAddedQueryWeight, gradNormAddedQueryBias, gradNormAddedKeyWeight, gradNormAddedKeyBias);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    auto l0NormRopeConcatGradOuts = l0op::NormRopeConcatGrad(
        gradQueryOutput, gradKeyOutput, gradValueOutput, query, key, encoderQuery, encoderKey, normQueryWeight,
        normQueryMean, normQueryRstd, normKeyWeight, normKeyMean, normKeyRstd, normAddedQueryWeight, normAddedQueryMean,
        normAddedQueryRstd, normAddedKeyWeight, normAddedKeyMean, normAddedKeyRstd, ropeSin, ropeCos, normType,
        normAddedType, ropeType, concatOrder, l0Executor);

    // required tensor
    CHECK_RET(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_QUERY_INDEX)] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_KEY_INDEX)] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_VALUE_INDEX)] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(checkViewCopyResult(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_QUERY_INDEX)], gradQuery, l0Executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(checkViewCopyResult(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_KEY_INDEX)], gradKey, l0Executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(checkViewCopyResult(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_VALUE_INDEX)], gradValue, l0Executor), ACLNN_ERR_INNER_NULLPTR);
    // optional tensor
    CHECK_RET(checkViewCopyResult(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_ENCODER_QUERY_INDEX)], gradEncoderQuery, l0Executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(checkViewCopyResult(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_ENCODER_KEY_INDEX)], gradEncoderKey, l0Executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(checkViewCopyResult(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_ENCODER_VALUE_INDEX)], gradEncoderValue, l0Executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(checkViewCopyResult(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_NORM_QUERY_WEIGHT_INDEX)], gradNormQueryWeight, l0Executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(checkViewCopyResult(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_NORM_QUERY_BIAS_INDEX)], gradNormQueryBias, l0Executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(checkViewCopyResult(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_NORM_KEY_WEIGHT_INDEX)], gradNormKeyWeight, l0Executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(checkViewCopyResult(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_NORM_KEY_BIAS_INDEX)], gradNormKeyBias, l0Executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(checkViewCopyResult(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_NORM_ADDED_QUERY_WEIGHT_INDEX)], gradNormAddedQueryWeight, l0Executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(checkViewCopyResult(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_NORM_ADDED_QUERY_BIAS_INDEX)], gradNormAddedQueryBias, l0Executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(checkViewCopyResult(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_NORM_ADDED_KEY_WEIGHT_INDEX)], gradNormAddedKeyWeight, l0Executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(checkViewCopyResult(l0NormRopeConcatGradOuts[static_cast<size_t>(OutputIndexBackward::GRAD_NORM_ADDED_KEY_BIAS_INDEX)], gradNormAddedKeyBias, l0Executor), ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNormRopeConcatBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNormRopeConcatBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
