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
 * \file quant_block_sparse_attn_metadata_check.h
 * \brief
 */

#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include "../op_kernel_aicpu/quant_block_sparse_attn_metadata.h"

static constexpr int64_t QBSA_QUANT_MODE_FP8 = 1;
static constexpr int64_t QBSA_QUANT_MODE_MXFP8 = 2;
static constexpr int64_t QBSA_SPARSE_BLOCK_SIZE_64 = 64;
static constexpr int64_t QBSA_SPARSE_BLOCK_SIZE_128 = 128;

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus CheckSingleParamQbsa(int64_t batchSize, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
                                        int64_t sparseBlockSizeQ, int64_t sparseBlockSizeK, int64_t quantMode,
                                        int64_t maskMode, const char *layoutSparseIndices)
{
    if (batchSize < 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "batchSize must be >= 0, but got %ld", batchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numHeadsQ < 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "numHeadsQ must be >= 0, but got %ld", numHeadsQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numHeadsKv < 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "numHeadsKv must be >= 0, but got %ld", numHeadsKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (headDim < 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "headDim must be >= 0, but got %ld", headDim);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (maskMode != 0 && maskMode != 3) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "maskMode only supports 0, 3, but got %ld", maskMode);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (quantMode != QBSA_QUANT_MODE_FP8 && quantMode != QBSA_QUANT_MODE_MXFP8) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "quantMode only supports 1 or 2, but got %ld", quantMode);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (quantMode == QBSA_QUANT_MODE_FP8) {
        if (sparseBlockSizeQ != QBSA_SPARSE_BLOCK_SIZE_128) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sparseBlockSizeQ only supports 128, but got %ld", sparseBlockSizeQ);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (sparseBlockSizeK != QBSA_SPARSE_BLOCK_SIZE_128) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sparseBlockSizeK only supports 128, but got %ld", sparseBlockSizeK);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (sparseBlockSizeQ != QBSA_SPARSE_BLOCK_SIZE_64 && sparseBlockSizeQ != QBSA_SPARSE_BLOCK_SIZE_128) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                    "sparseBlockSizeQ only supports 64 or 128 when quantMode is 2, but got %ld", sparseBlockSizeQ);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (sparseBlockSizeK != QBSA_SPARSE_BLOCK_SIZE_64 && sparseBlockSizeK != QBSA_SPARSE_BLOCK_SIZE_128) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                    "sparseBlockSizeK only supports 64 or 128 when quantMode is 2, but got %ld", sparseBlockSizeK);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if ((layoutSparseIndices == nullptr) || (strcmp(layoutSparseIndices, "B_N_Qb_Kb") != 0)) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "layoutSparseIndices only supports B_N_Qb_Kb, but got %s",
                layoutSparseIndices == nullptr ? "nullptr" : layoutSparseIndices);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckExistenceQbsa(const aclTensor *sparseSeqLen, const aclTensor *cuSeqlensQOptional,
                                      const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
                                      const aclTensor *sequsedKvOptional, int64_t batchSize, int64_t numHeadsQ,
                                      uint32_t aicCoreNum, uint32_t aivCoreNum, const char *socVersion,
                                      const aclTensor *metadata)
{
    if (sparseSeqLen == nullptr) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sparseSeqLen is nullptr");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sparseSeqLen->GetViewShape().GetDimNum() != 2 && sparseSeqLen->GetViewShape().GetDimNum() != 3) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sparseSeqLen must be 2D or 3D tensor");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sparseSeqLen->GetDataType() != ACL_INT32) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sparseSeqLen dtype must be INT32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // metadata
    if (metadata == nullptr) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "metadata must not be null");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (metadata->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "metadata must be 1D tensor, but got %ld dims",
                metadata->GetViewShape().GetDimNum());
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t minMetaSize = optiling::QBSA_HEAD_METADATA_SIZE +
                           static_cast<uint64_t>(batchSize) * static_cast<uint64_t>(numHeadsQ) *
                               optiling::AIC_CORE_NUM * optiling::QBSA_METADATA_SIZE +
                           static_cast<uint64_t>(optiling::AIV_CORE_NUM) * optiling::FD_METADATA_SIZE;
    if (metadata->GetViewShape().GetDim(0) < static_cast<int64_t>(minMetaSize)) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "metadata shape too small, need at least %llu int32, but got %ld",
                static_cast<unsigned long long>(minMetaSize), metadata->GetViewShape().GetDim(0));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (metadata->GetDataType() != ACL_INT32) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "metadata dtype must be INT32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // platform
    if (aicCoreNum <= 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "aicCoreNum must be > 0, but got %u", aicCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aivCoreNum <= 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "aivCoreNum must be > 0, but got %u", aivCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckConsistencyQbsa(int64_t numHeadsQ, int64_t numHeadsKv)
{
    if (numHeadsKv != 0 && numHeadsQ % numHeadsKv != 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                "numHeadsQ must be divisible by numHeadsKv, but got numHeadsQ=%ld, numHeadsKv=%ld", numHeadsQ,
                numHeadsKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // TODO: validate Q/KV batch consistency
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsCheck(const aclTensor *sparseSeqLen, const aclTensor *cuSeqlensQOptional,
                               const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
                               const aclTensor *sequsedKvOptional, int64_t batchSize, int64_t numHeadsQ,
                               int64_t numHeadsKv, int64_t headDim, int64_t sparseBlockSizeQ, int64_t sparseBlockSizeK,
                               int64_t quantMode, int64_t maskMode, const char *layoutQOptional,
                               const char *layoutKvOptional,
                               const char *layoutSparseIndicesOptional, uint32_t aicCoreNum, uint32_t aivCoreNum,
                               const char *socVersion, const aclTensor *metadata)
{
    aclnnStatus ret =
        CheckSingleParamQbsa(batchSize, numHeadsQ, numHeadsKv, headDim, sparseBlockSizeQ, sparseBlockSizeK, quantMode,
                             maskMode, layoutSparseIndicesOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckExistenceQbsa(sparseSeqLen, cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional,
                             batchSize, numHeadsQ, aicCoreNum, aivCoreNum, socVersion, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckConsistencyQbsa(numHeadsQ, numHeadsKv);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
