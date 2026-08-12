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
 * \file aclnn_quant_block_sparse_attn_metadata.cpp
 * \brief
 */

#include "aclnn_quant_block_sparse_attn_metadata.h"
#include "l0_quant_block_sparse_attn_metadata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
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

#include "../quant_block_sparse_attn_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnQuantBlockSparseAttnMetadataGetWorkspaceSize(
    const aclTensor *sparseSeqLen, const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, int64_t batchSize, int64_t numHeadsQ,
    int64_t numHeadsKv, int64_t headDim, int64_t sparseBlockSizeQ, int64_t sparseBlockSizeK, int64_t quantMode,
    int64_t maskMode, const char *layoutQOptional, const char *layoutKvOptional,
    const char *layoutSparseIndicesOptional, const aclTensor *metadata, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnQuantBlockSparseAttnMetadata,
                   DFX_IN(sparseSeqLen, cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional,
                          batchSize, numHeadsQ, numHeadsKv, headDim, sparseBlockSizeQ, sparseBlockSizeK, quantMode,
                          maskMode, layoutQOptional, layoutKvOptional, layoutSparseIndicesOptional),
                   DFX_OUT(metadata));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();

    auto ret = ParamsCheck(sparseSeqLen, cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional,
                           batchSize, numHeadsQ, numHeadsKv, headDim, sparseBlockSizeQ, sparseBlockSizeK, quantMode,
                           maskMode, layoutQOptional, layoutKvOptional, layoutSparseIndicesOptional, aicCoreNum,
                           aivCoreNum, socVersion, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Contiguous conversion - INPUT_TENSOR_ROWS order
    const aclTensor *sparseSeqLenContiguous = l0op::Contiguous(sparseSeqLen, uniqueExecutor.get());
    if (sparseSeqLenContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "sparse_seq_len contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const aclTensor *cuSeqlensQOptionalContiguous = nullptr;
    if (cuSeqlensQOptional != nullptr) {
        cuSeqlensQOptionalContiguous = l0op::Contiguous(cuSeqlensQOptional, uniqueExecutor.get());
        if (cuSeqlensQOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_q contiguous is none");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *cuSeqlensKvOptionalContiguous = nullptr;
    if (cuSeqlensKvOptional != nullptr) {
        cuSeqlensKvOptionalContiguous = l0op::Contiguous(cuSeqlensKvOptional, uniqueExecutor.get());
        if (cuSeqlensKvOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_kv contiguous is none");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *sequsedQOptionalContiguous = nullptr;
    if (sequsedQOptional != nullptr) {
        sequsedQOptionalContiguous = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
        if (sequsedQOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "seqused_q contiguous is none");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *sequsedKvOptionalContiguous = nullptr;
    if (sequsedKvOptional != nullptr) {
        sequsedKvOptionalContiguous = l0op::Contiguous(sequsedKvOptional, uniqueExecutor.get());
        if (sequsedKvOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "seqused_kv contiguous is none");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    auto output = l0op::QuantBlockSparseAttnMetadata(
        sparseSeqLenContiguous, cuSeqlensQOptionalContiguous, cuSeqlensKvOptionalContiguous,
        sequsedQOptionalContiguous, sequsedKvOptionalContiguous, batchSize, numHeadsQ, numHeadsKv, headDim,
        sparseBlockSizeQ, sparseBlockSizeK, quantMode, maskMode, layoutQOptional, layoutKvOptional,
        layoutSparseIndicesOptional, socVersion, aicCoreNum, aivCoreNum, metadata, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnQuantBlockSparseAttnMetadata(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantBlockSparseAttnMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
