/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <algorithm>
#include <unordered_map>
#include "nsa_compress_attention_infer.h"
#include "aclnn_nsa_compress_attention_infer.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

__attribute__((visibility("default"))) aclnnStatus aclnnNsaCompressAttentionInferGetMaxWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* attentionMaskOptional,
    const aclTensor* blockTableOptional, const aclIntArray* actualQSeqLenOptional,
    const aclIntArray* actualCmpKvSeqLenOptional, const aclIntArray* actualSelKvSeqLenOptional,
    const aclTensor* topKMaskOptional, int64_t numHeads, int64_t numKeyValueHeads, int64_t selectBlockSize,
    int64_t selectBlockCount, int64_t compressBlockSize, int64_t compressBlockStride, double scaleValue,
    char* layoutOptional, int64_t pageBlockSize, int64_t sparseMode, const aclTensor* output,
    const aclTensor* topKOutput, uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus nsaCompressAttentionInferContiguous(const aclTensor*& query, const aclTensor*& key, const aclTensor*& value,
                                                const aclTensor* attentionMaskOptional,
                                                const aclTensor*& blockTableOptional,
                                                const aclTensor*& topKMaskOptional, aclOpExecutor* executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (attentionMaskOptional) {
        attentionMaskOptional = l0op::Contiguous(attentionMaskOptional, executor);
        CHECK_RET(attentionMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (blockTableOptional) {
        blockTableOptional = l0op::Contiguous(blockTableOptional, executor);
        CHECK_RET(blockTableOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (topKMaskOptional) {
        topKMaskOptional = l0op::Contiguous(topKMaskOptional, executor);
        CHECK_RET(topKMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static bool nsaCompressAttentionInferCheckDataType(const aclTensor* query, const aclTensor* key, const aclTensor* value,
                                                   const aclTensor* blockTableOptional, const aclTensor* output,
                                                   const aclTensor* topKOutput)
{
    auto qDtype = query->GetDataType();
    auto kDtype = key->GetDataType();
    auto vDtype = value->GetDataType();
    auto blockTableDtype = blockTableOptional->GetDataType();
    auto attnOutDtype = output->GetDataType();
    auto topKOutDtype = topKOutput->GetDataType();

    static const std::unordered_map<DataType, std::vector<DataType>> dTypeMappingQtoKv = {
        {DataType::DT_FLOAT16, {DataType::DT_FLOAT16}},
        {DataType::DT_BF16, {DataType::DT_BF16}},
    };
    if (dTypeMappingQtoKv.find(qDtype) == dTypeMappingQtoKv.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input Q data type is invalid, please check.");
        return false;
    } else {
        auto validKvDtypeList = dTypeMappingQtoKv.at(qDtype);
        if (std::find(validKvDtypeList.begin(), validKvDtypeList.end(), kDtype) == validKvDtypeList.end() ||
            std::find(validKvDtypeList.begin(), validKvDtypeList.end(), vDtype) == validKvDtypeList.end()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input K or V data type is invalid, please check.");
            return false;
        }
        if (attnOutDtype != qDtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output attnOut data type is invalid, please check.");
            return false;
        }
    }
    if (blockTableDtype != DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockTable data type must be int32, please check.");
        return false;
    }
    if (topKOutDtype != DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output topkOut data type must be int32, please check.");
        return false;
    }
    return true;
}

aclnnStatus nsaCompressAttentionInferValidateParam(const aclTensor* query, const aclTensor* key, const aclTensor* value,
                                                   const aclTensor* blockTableOptional, const aclTensor* output,
                                                   const aclTensor* topKOutput)
{
    Shape qShape = query->GetViewShape();
    auto batchSize = qShape[0];
    if (batchSize > 10000U || batchSize < 1U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batchSize %ld not in  [1, 10000], please check", batchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(nsaCompressAttentionInferCheckDataType(query, key, value, blockTableOptional, output, topKOutput),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus nsaCompressAttentionInferCheckTensorNull(const aclTensor* query, const aclTensor* key,
                                                     const aclTensor* value, const aclTensor* blockTableOptional,
                                                     const aclIntArray* actualCmpKvSeqLenOptional,
                                                     const char* layoutOptional, const aclTensor* output,
                                                     const aclTensor* topKOutput, const uint64_t* workspaceSize)
{
    // 必须的参数指针判空
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(blockTableOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(actualCmpKvSeqLenOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(layoutOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(output != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(topKOutput != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus FakeArray_cai(const aclIntArray* inArray, aclIntArray*& outArray)
{
    OP_LOGD("start fake array");
    if (inArray != nullptr) {
        OP_LOGD("input array is not nullptr");
        uint64_t size = inArray->Size();
        // tiling侧认为有tensor但没有data就是计算最大workspace
        std::vector<int64_t> fake_array(size, -1);
        outArray = aclCreateIntArray(fake_array.data(), size);
        if (outArray == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc array failed");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    OP_LOGD("end fake array");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaCompressAttentionInferGetMaxWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* attentionMaskOptional,
    const aclTensor* blockTableOptional, const aclIntArray* actualQSeqLenOptional,
    const aclIntArray* actualCmpKvSeqLenOptional, const aclIntArray* actualSelKvSeqLenOptional,
    const aclTensor* topKMaskOptional, int64_t numHeads, int64_t numKeyValueHeads, int64_t selectBlockSize,
    int64_t selectBlockCount, int64_t compressBlockSize, int64_t compressBlockStride, double scaleValue,
    char* layoutOptional, int64_t pageBlockSize, int64_t sparseMode, const aclTensor* output,
    const aclTensor* topKOutput, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    aclIntArray* fakeActualCmpKvSeqLenOptional{nullptr};
    // nullptr不处理， nullptr是空指针，这样不会影响原来就不传入actual seq length为空的逻辑
    aclnnStatus ret = FakeArray_cai(actualCmpKvSeqLenOptional, fakeActualCmpKvSeqLenOptional);
    CHECK_RET_CODE(ret, "Try alloc fake actualSeqLengthsKvOptional failed");
    ret = aclnnNsaCompressAttentionInferGetWorkspaceSize(
        query, key, value, attentionMaskOptional, blockTableOptional, actualQSeqLenOptional,
        fakeActualCmpKvSeqLenOptional, actualSelKvSeqLenOptional, topKMaskOptional, numHeads, numKeyValueHeads,
        selectBlockSize, selectBlockCount, compressBlockSize, compressBlockStride, scaleValue, layoutOptional,
        pageBlockSize, sparseMode, output, topKOutput, workspaceSize, executor);
    aclDestroyIntArray(fakeActualCmpKvSeqLenOptional);
    return ret;
}

aclnnStatus aclnnNsaCompressAttentionInferGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* attentionMaskOptional,
    const aclTensor* blockTableOptional, const aclIntArray* actualQSeqLenOptional,
    const aclIntArray* actualCmpKvSeqLenOptional, const aclIntArray* actualSelKvSeqLenOptional,
    const aclTensor* topKMaskOptional, int64_t numHeads, int64_t numKeyValueHeads, int64_t selectBlockSize,
    int64_t selectBlockCount, int64_t compressBlockSize, int64_t compressBlockStride, double scaleValue,
    char* layoutOptional, int64_t pageBlockSize, int64_t sparseMode, const aclTensor* output,
    const aclTensor* topKOutput, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CHECK_RET(
        nsaCompressAttentionInferCheckTensorNull(query, key, value, blockTableOptional, actualCmpKvSeqLenOptional,
                                                 layoutOptional, output, topKOutput, workspaceSize) == ACLNN_SUCCESS,
        ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(nsaCompressAttentionInferValidateParam(query, key, value, blockTableOptional, output, topKOutput) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    L2_DFX_PHASE_1(aclnnNsaCompressAttentionInfer,
                   DFX_IN(query, key, value, attentionMaskOptional, blockTableOptional, actualQSeqLenOptional,
                          actualCmpKvSeqLenOptional, actualSelKvSeqLenOptional, topKMaskOptional, numHeads,
                          numKeyValueHeads, selectBlockSize, selectBlockCount, compressBlockSize, compressBlockStride,
                          scaleValue, layoutOptional, pageBlockSize, sparseMode),
                   DFX_OUT(output, topKOutput));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if (output->IsEmpty() && topKOutput->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (strcmp(layoutOptional, "TND") != 0 && strcmp(layoutOptional, "BSND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Layout %s is not TND and BSND, invalid shape, please check", layoutOptional);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_INVALID;
    }

    aclOpExecutor* l0Executor = uniqueExecutor.get();

    CHECK_RET(nsaCompressAttentionInferContiguous(query, key, value, attentionMaskOptional, blockTableOptional,
                                                  topKMaskOptional, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    string inputLayoutStr = op::ToString(layoutOptional).GetString();
    auto l0NsaCompressAttentionInferOuts = l0op::NsaCompressAttentionInfer(
        query, key, value, attentionMaskOptional, blockTableOptional, actualQSeqLenOptional, actualCmpKvSeqLenOptional,
        actualSelKvSeqLenOptional, topKMaskOptional, numHeads, numKeyValueHeads, selectBlockSize, selectBlockCount,
        compressBlockSize, compressBlockStride, scaleValue, inputLayoutStr.c_str(), pageBlockSize, sparseMode,
        l0Executor);

    CHECK_RET(l0NsaCompressAttentionInferOuts[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0NsaCompressAttentionInferOuts[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto l0output = l0NsaCompressAttentionInferOuts[0];
    auto l0topKOutput = l0NsaCompressAttentionInferOuts[1];

    if (l0output == nullptr || l0topKOutput == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "l0output or l0topKOutput is null");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto viewCopyResult0 = l0op::ViewCopy(l0output, output, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0topKOutput, topKOutput, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaCompressAttentionInfer(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNsaCompressAttentionInfer);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
} // namespace

#ifdef __cplusplus
}
#endif
