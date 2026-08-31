/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_minimax_sparse_attention_split_kv.h"

#include "minimax_sparse_attention_split_kv.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/common_types.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include <acl/acl.h>
#include <cstring>
#include <tuple>
#include <vector>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static aclnnStatus MakeContiguous(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                                  const aclTensor *&blockTable, const aclTensor *&k2qRowPtr,
                                  const aclTensor *&k2qQIndices, const aclTensor *&k2qSlotIndices,
                                  const aclTensor *&actualSeqLengths, const aclTensor *&actualSeqLengthsKv,
                                  aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    if (blockTable != nullptr) {
        blockTable = l0op::Contiguous(blockTable, executor);
        CHECK_RET(blockTable != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    k2qRowPtr = l0op::Contiguous(k2qRowPtr, executor);
    CHECK_RET(k2qRowPtr != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    k2qQIndices = l0op::Contiguous(k2qQIndices, executor);
    CHECK_RET(k2qQIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    k2qSlotIndices = l0op::Contiguous(k2qSlotIndices, executor);
    CHECK_RET(k2qSlotIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    actualSeqLengths = l0op::Contiguous(actualSeqLengths, executor);
    CHECK_RET(actualSeqLengths != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    actualSeqLengthsKv = l0op::Contiguous(actualSeqLengthsKv, executor);
    CHECK_RET(actualSeqLengthsKv != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

} // namespace

__attribute__((visibility("default"))) aclnnStatus aclnnMinimaxSparseAttentionSplitKvGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *blockTable,
    const aclTensor *k2qRowPtr, const aclTensor *k2qQIndices, const aclTensor *k2qSlotIndices,
    const aclTensor *actualSeqLengths, const aclTensor *actualSeqLengthsKv, int64_t numKeyValueHeads, double scaleValue,
    int64_t blockSize, int64_t topK, int64_t innerPrecise, bool softmaxLseFlag, const char *inputLayout,
    aclTensor *attentionOut, aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(k2qRowPtr != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(k2qQIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(k2qSlotIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(actualSeqLengths != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(actualSeqLengthsKv != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (softmaxLseFlag) {
        CHECK_RET(softmaxLse != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (innerPrecise != 0 && innerPrecise != 1 && innerPrecise != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "innerPrecise must be 0, 1 or 4, got %ld.", innerPrecise);
        return ACLNN_ERR_PARAM_INVALID;
    }
    DataType qDtype = query->GetDataType();
    DataType kDtype = key->GetDataType();
    DataType vDtype = value->GetDataType();
    auto isQkvOk = [](DataType d) { return d == DataType::DT_BF16 || d == DataType::DT_FLOAT8_E4M3FN; };
    if (!isQkvOk(qDtype) || !isQkvOk(kDtype) || !isQkvOk(vDtype)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MinimaxSparseAttentionSplitKv Q/K/V only support BF16 or FLOAT8_E4M3FN.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qDtype != kDtype || qDtype != vDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MinimaxSparseAttentionSplitKv Q/K/V dtype must be consistent.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (attentionOut->GetDataType() != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MinimaxSparseAttentionSplitKv attentionOut must be BF16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qDtype == DataType::DT_FLOAT8_E4M3FN && innerPrecise != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "MinimaxSparseAttentionSplitKv fp8 path uses FP32 O_partial only "
                "(innerPrecise=4); got %ld. innerPrecise=0/1 are not implemented.",
                innerPrecise);
        return ACLNN_ERR_PARAM_INVALID;
    }
    const char *layout = (inputLayout == nullptr || inputLayout[0] == '\0') ? "TND" : inputLayout;
    if (strcmp(layout, "TND") != 0 && strcmp(layout, "BNSD") != 0 && strcmp(layout, "BSND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputLayout must be TND, BNSD or BSND, got %s.", layout);
        return ACLNN_ERR_PARAM_INVALID;
    }

    L2_DFX_PHASE_1(
        aclnnMinimaxSparseAttentionSplitKv,
        DFX_IN(query, key, value, blockTable, k2qRowPtr, k2qQIndices, k2qSlotIndices, actualSeqLengths,
               actualSeqLengthsKv, numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise, softmaxLseFlag, layout),
        DFX_OUT(attentionOut, softmaxLse));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();

    aclnnStatus ret = MakeContiguous(query, key, value, blockTable, k2qRowPtr, k2qQIndices, k2qSlotIndices,
                                     actualSeqLengths, actualSeqLengthsKv, executorImpl);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    const aclTensor *tempTensor = nullptr;
    const aclTensor *lsePlaceHolder = nullptr;
    int64_t dummyLseAddr = 0xff;
    if (softmaxLseFlag == false) {
        std::vector<int64_t> shape = {0};
        tempTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
                                     shape.data(), shape.size(), static_cast<void *>(&dummyLseAddr));
        lsePlaceHolder = tempTensor;
    } else {
        lsePlaceHolder = softmaxLse;
    }
    if (lsePlaceHolder == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "softmaxLse placeholder is nullptr.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto outputs = l0op::MinimaxSparseAttentionSplitKv(
        query, key, value, blockTable, k2qRowPtr, k2qQIndices, k2qSlotIndices, actualSeqLengths, actualSeqLengthsKv,
        numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise, softmaxLseFlag, layout, attentionOut,
        lsePlaceHolder, executorImpl);
    auto output = std::get<0>(outputs);
    auto lseOut = std::get<1>(outputs);
    if (softmaxLseFlag == false) {
        aclDestroyTensor(const_cast<aclTensor *>(tempTensor));
    }
    if (output == nullptr || lseOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MinimaxSparseAttentionSplitKv returned nullptr output.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(output, attentionOut, executorImpl);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (softmaxLseFlag) {
        auto lseViewCopy = l0op::ViewCopy(lseOut, softmaxLse, executorImpl);
        CHECK_RET(lseViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnMinimaxSparseAttentionSplitKv(void *workspace,
                                                                                      uint64_t workspaceSize,
                                                                                      aclOpExecutor *executor,
                                                                                      aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMinimaxSparseAttentionSplitKv);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
