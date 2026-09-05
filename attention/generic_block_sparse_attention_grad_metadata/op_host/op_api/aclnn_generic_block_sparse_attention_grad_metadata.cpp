/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_generic_block_sparse_attention_grad_metadata.cpp
 * \brief
 */

#include "aclnn_generic_block_sparse_attention_grad_metadata.h"
#include "generic_block_sparse_attention_grad_metadata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "../generic_block_sparse_attention_grad_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
    const aclTensor *rsvdBlockIdx, const aclTensor *rsvdBlockCount, const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t numQHeads, int64_t numKvHeads, int64_t headDim,
    const aclIntArray *blockShape, int64_t isPackedGQA, char *layoutQ, char *layoutKv, int64_t maskType,
    int64_t softmaxPrecision, int64_t winLeft, int64_t winRight, aclTensor *metadata, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    if (workspaceSize == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "workspaceSize is nullptr");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "executor is nullptr");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    L2_DFX_PHASE_1(aclnnGenericBlockSparseAttentionGradMetadata,
                   DFX_IN(rsvdBlockIdx, rsvdBlockCount, cuSeqLengthsQOptional, cuSeqLengthsKvOptional, sequsedQOptional,
                          sequsedKvOptional, maxQSeqlen, maxKvSeqlen, numQHeads, numKvHeads, headDim, blockShape,
                          isPackedGQA, layoutQ, layoutKv, maskType, softmaxPrecision, winLeft, winRight),
                   DFX_OUT(metadata));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    std::string socVersionStr = npuInfo.GetSocLongVersion();
    const char *socVersion = socVersionStr.c_str();

    int64_t blockShapeX = 1;
    int64_t blockShapeY = 128;
    if (blockShape != nullptr) {
        if (blockShape->Size() >= 1) {
            blockShapeX = (*blockShape)[0];
        }
        if (blockShape->Size() >= 2) {
            blockShapeY = (*blockShape)[1];
        }
    }

    auto ret = ParamsCheck(rsvdBlockIdx, rsvdBlockCount, cuSeqLengthsQOptional, cuSeqLengthsKvOptional,
                           sequsedQOptional, sequsedKvOptional, maxQSeqlen, maxKvSeqlen, numQHeads, numKvHeads, headDim,
                           blockShapeX, blockShapeY, isPackedGQA, layoutQ, layoutKv, maskType, softmaxPrecision,
                           winLeft, winRight, aicCoreNum, aivCoreNum, socVersion, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor *rsvdBlockIdxContiguous = l0op::Contiguous(rsvdBlockIdx, uniqueExecutor.get());
    CHECK_RET(rsvdBlockIdxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *rsvdBlockCountContiguous = l0op::Contiguous(rsvdBlockCount, uniqueExecutor.get());
    CHECK_RET(rsvdBlockCountContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *cuSeqLengthsQOptionalContiguous = nullptr;
    if (cuSeqLengthsQOptional != nullptr) {
        cuSeqLengthsQOptionalContiguous = l0op::Contiguous(cuSeqLengthsQOptional, uniqueExecutor.get());
        CHECK_RET(cuSeqLengthsQOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *cuSeqLengthsKvOptionalContiguous = nullptr;
    if (cuSeqLengthsKvOptional != nullptr) {
        cuSeqLengthsKvOptionalContiguous = l0op::Contiguous(cuSeqLengthsKvOptional, uniqueExecutor.get());
        CHECK_RET(cuSeqLengthsKvOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *sequsedQOptionalContiguous = nullptr;
    if (sequsedQOptional != nullptr) {
        sequsedQOptionalContiguous = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
        CHECK_RET(sequsedQOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *sequsedKvOptionalContiguous = nullptr;
    if (sequsedKvOptional != nullptr) {
        sequsedKvOptionalContiguous = l0op::Contiguous(sequsedKvOptional, uniqueExecutor.get());
        CHECK_RET(sequsedKvOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto output = l0op::GenericBlockSparseAttentionGradMetadata(
        rsvdBlockIdxContiguous, rsvdBlockCountContiguous, cuSeqLengthsQOptionalContiguous,
        cuSeqLengthsKvOptionalContiguous, sequsedQOptionalContiguous, sequsedKvOptionalContiguous, maxQSeqlen,
        maxKvSeqlen, numQHeads, numKvHeads, headDim, blockShapeX, blockShapeY, isPackedGQA, layoutQ, layoutKv, maskType,
        softmaxPrecision, winLeft, winRight, socVersion, aicCoreNum, aivCoreNum, metadata, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGradMetadata(void *workspace,
                                                                                                uint64_t workspaceSize,
                                                                                                aclOpExecutor *executor,
                                                                                                aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGenericBlockSparseAttentionGradMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
