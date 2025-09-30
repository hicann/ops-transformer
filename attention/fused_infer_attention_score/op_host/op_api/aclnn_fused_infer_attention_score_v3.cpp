/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include "graph/types.h"
#include "aclnn_fused_infer_attention_score_v3.h"

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_fused_infer_attention_score_inner.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
/**
 * @brief for acl graph calculates the max workspace size based on the specific calculation process.
 * declaration here for testcase to use by extern the interface
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedInferAttentionScoreV3GetMaxWorkspaceSize(
    const aclTensor *query, const aclTensorList *tensorListKey, const aclTensorList *tensorListValue, const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional, const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional, const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional, const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional, const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional, const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional, const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional, const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *tensorKeySharedPrefixOptional, const aclTensor *tensorValueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, const aclTensor *keyRopeAntiquantScaleOptional,
    int64_t numHeads, double scaleValue, int64_t preTokens,
    int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise,
    int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode, int64_t valueAntiquantMode,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void *executor, const size_t irIndex);

struct FiaActualSeqInArrays {
    const aclIntArray *actualSeqLengthsOptional = nullptr;
    const aclIntArray *actualSeqLengthsKvOptional = nullptr;
    const aclIntArray *actualSharedPrefixLenOptional = nullptr;
};

struct FiaActualSeqOutTensors {
    aclTensor *actualSeqLengthsOptional = nullptr;
    aclTensor *actualSeqLengthsKvOptional = nullptr;
    aclTensor *actualSharedPrefixLenOptional = nullptr;
};

aclnnStatus FakeFiaActualSeqArrays(const FiaActualSeqInArrays &inArrays,  FiaActualSeqOutTensors &outTensors) {
    // nullptr不处理， nullptr是空指针，这样不会影响原来就不传入actual seq length为空的逻辑
    aclnnStatus ret = FakeArray(inArrays.actualSeqLengthsOptional, outTensors.actualSeqLengthsOptional);
    CHECK_RET_CODE(ret, "Try alloc fake actualSeqLengthsOptional failed");

    ret = FakeArray(inArrays.actualSeqLengthsKvOptional, outTensors.actualSeqLengthsKvOptional);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc fake actualSeqLengthsKvOptional failed");
        aclDestroyTensor(outTensors.actualSeqLengthsOptional); // 没有返回值无需校验
        return ret;
    }

    ret = FakeArray(inArrays.actualSharedPrefixLenOptional, outTensors.actualSharedPrefixLenOptional);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc fake actualSharedPrefixLenOptional failed");
        aclDestroyTensor(outTensors.actualSeqLengthsOptional); // 没有返回值无需校验
        aclDestroyTensor(outTensors.actualSeqLengthsKvOptional);
        return ret;
    }
    return ret;
}

aclnnStatus aclnnFusedInferAttentionScoreV3GetMaxWorkspaceSize(
    const aclTensor *query, const aclTensorList *tensorListKey, const aclTensorList *tensorListValue,
    const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional,
    const aclTensor *deqScale2Optional,
    const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional,
    const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional,
    const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional,
    const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional,
    const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *tensorKeySharedPrefixOptional,
    const aclTensor *tensorValueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    const aclTensor *keyRopeAntiquantScaleOptional,
    int64_t numHeads, double scaleValue, int64_t preTokens,
    int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads,
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize,
    int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode, int64_t valueAntiquantMode,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("start aclnnFusedInferAttentionScoreV3GetMaxWorkspaceSize");
    TensorPreProcess(tensorListKey, tensorListValue);
    PrefixTensorPreProcess(tensorKeySharedPrefixOptional, tensorValueSharedPrefixOptional);

    FiaActualSeqInArrays actualSeqArrays{actualSeqLengthsOptional, actualSeqLengthsKvOptional, actualSharedPrefixLenOptional};
    FiaActualSeqOutTensors fakeActualSeqTensors{};
    aclnnStatus ret = FakeFiaActualSeqArrays(actualSeqArrays, fakeActualSeqTensors);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;
    FusedInferAttentionScoreProcessSoftmaxLse(softmaxLseFlag, softmaxLse, tempTensor, placeHolder);

    ret = aclnnInnerFusedInferAttentionScoreTensorGetWorkspaceSize(
        query, tensorListKey, tensorListValue, pseShiftOptional, attenMaskOptional, fakeActualSeqTensors.actualSeqLengthsOptional,
        fakeActualSeqTensors.actualSeqLengthsKvOptional, deqScale1Optional, quantScale1Optional, deqScale2Optional, quantScale2Optional,
        quantOffset2Optional, antiquantScaleOptional, antiquantOffsetOptional, blockTableOptional,
        queryPaddingSizeOptional, kvPaddingSizeOptional, keyAntiquantScaleOptional, keyAntiquantOffsetOptional,
        valueAntiquantScaleOptional, valueAntiquantOffsetOptional, tensorKeySharedPrefixOptional,
        tensorValueSharedPrefixOptional, fakeActualSeqTensors.actualSharedPrefixLenOptional, queryRopeOptional,
        keyRopeOptional, keyRopeAntiquantScaleOptional, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens,
        inputLayout, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag,
        keyAntiquantMode, valueAntiquantMode, 0, 0, 0, attentionOut, placeHolder, workspaceSize, executor);
    if (softmaxLseFlag == false) {
        aclDestroyTensor(tempTensor);
    }
    aclDestroyTensor(fakeActualSeqTensors.actualSeqLengthsOptional); // 只会成功，无需校验
    aclDestroyTensor(fakeActualSeqTensors.actualSeqLengthsKvOptional);
    aclDestroyTensor(fakeActualSeqTensors.actualSharedPrefixLenOptional);
    return ret;
}

aclnnStatus aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
    const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional,
    const aclTensor *deqScale2Optional,
    const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional,
    const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional,
    const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional,
    const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional,
    const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional,
    const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    const aclTensor *keyRopeAntiquantScaleOptional,
    int64_t numHeads, double scaleValue, int64_t preTokens,
    int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads,
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize,
    int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode, int64_t valueAntiquantMode,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    const aclTensorList *tensorListKey = key;
    const aclTensorList *tensorListValue = value;
    TensorPreProcess(tensorListKey, tensorListValue);

    const aclTensor *tensorKeySharedPrefixOptional = keySharedPrefixOptional;
    const aclTensor *tensorValueSharedPrefixOptional = valueSharedPrefixOptional;
    PrefixTensorPreProcess(tensorKeySharedPrefixOptional, tensorValueSharedPrefixOptional);

    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;
    FusedInferAttentionScoreProcessSoftmaxLse(softmaxLseFlag, softmaxLse, tempTensor, placeHolder);

    aclnnStatus ret = aclnnInnerFusedInferAttentionScoreGetWorkspaceSize(
        query, tensorListKey, tensorListValue, pseShiftOptional, attenMaskOptional, actualSeqLengthsOptional,
        actualSeqLengthsKvOptional, deqScale1Optional, quantScale1Optional, deqScale2Optional, quantScale2Optional,
        quantOffset2Optional, antiquantScaleOptional, antiquantOffsetOptional, blockTableOptional,
        queryPaddingSizeOptional, kvPaddingSizeOptional, keyAntiquantScaleOptional, keyAntiquantOffsetOptional,
        valueAntiquantScaleOptional, valueAntiquantOffsetOptional, tensorKeySharedPrefixOptional,
        tensorValueSharedPrefixOptional, actualSharedPrefixLenOptional, queryRopeOptional,
        keyRopeOptional, keyRopeAntiquantScaleOptional, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens,
        inputLayout, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag,
        keyAntiquantMode, valueAntiquantMode, 0, 0, 0, attentionOut, placeHolder, workspaceSize, executor);
    if (softmaxLseFlag == false) {
        aclDestroyTensor(tempTensor);
    }
    if (ret == 0) {
        if (NnopbaseDisableOptionalInput != nullptr) {
            NnopbaseDisableOptionalInput(*executor, 27U); // 27 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 28U); // 28 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 29U); // 29 is input irIndex，占位符
            NnopbaseDisableOptionalInput(*executor, 30U); // 30 is input irIndex
        }
    }
    return ret;
}

aclnnStatus aclnnFusedInferAttentionScoreV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            const aclrtStream stream)
{
    return aclnnInnerFusedInferAttentionScore(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif