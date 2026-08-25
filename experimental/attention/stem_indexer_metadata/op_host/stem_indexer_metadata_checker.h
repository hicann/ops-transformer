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
 * \file stem_indexer_metadata_check.h
 * \brief
 */

#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"

static constexpr int64_t STEM_BLOCK_SIZE_128 = 128;
static constexpr int64_t Q_HEADS_32 = 32;
static constexpr int64_t Q_HEADS_64 = 64;
static constexpr int64_t KV_HEADS_2 = 2;
static constexpr int64_t KV_HEADS_4 = 4;
static constexpr int64_t KV_HEADS_8 = 8;
static constexpr int64_t DIM_QKFLAT_2048 = 2048;
static constexpr int64_t WINDOW_SIZE_4 = 4;
static constexpr int64_t STEM_INDEXER_BATCH_SIZE_LIMIT = 65536;
static constexpr int64_t STEM_INDEXER_METADATA_HEADER_SIZE = 16;
static constexpr int64_t STEM_INDEXER_METADATA_CORE_STRIDE = 16;
static constexpr int64_t STEM_INDEXER_METADATA_AIC_CORE_NUM = 36;
static constexpr int64_t STEM_INDEXER_METADATA_AIV_CORE_NUM = 72;
static constexpr int64_t STEM_INDEXER_METADATA_ALIGN_SIZE = 4096;

static int64_t CalcMetadataCapacity(int64_t batchSize, int64_t kvHeads)
{
    int64_t maxSectionNum = batchSize * kvHeads;
    int64_t requiredSize = STEM_INDEXER_METADATA_HEADER_SIZE +
                           maxSectionNum * (STEM_INDEXER_METADATA_AIC_CORE_NUM + STEM_INDEXER_METADATA_AIV_CORE_NUM) *
                               STEM_INDEXER_METADATA_CORE_STRIDE;
    return (requiredSize + STEM_INDEXER_METADATA_ALIGN_SIZE - 1) / STEM_INDEXER_METADATA_ALIGN_SIZE *
           STEM_INDEXER_METADATA_ALIGN_SIZE;
}

static bool IsTensorExist(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetData() != nullptr);
}

static aclnnStatus ParamsCheck(const aclTensor *qSeqLens, const aclTensor *kvSeqLens, int64_t qHeads, int64_t kvHeads,
                               int64_t stemBlockSize, int64_t dimQkflat, int64_t windowSize, const aclTensor *metadata)
{
    if (!IsTensorExist(qSeqLens)) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "qSeqLens does not exists");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (!IsTensorExist(kvSeqLens)) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "kvSeqLens does not exists");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qSeqLens->GetViewShape().GetDimNum() != 1 || qSeqLens->GetViewShape().GetDim(0) <= 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "qSeqLens must be a non-empty 1D tensor");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (kvSeqLens->GetViewShape().GetDimNum() != 1 || kvSeqLens->GetViewShape().GetDim(0) <= 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "kvSeqLens must be a non-empty 1D tensor");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qSeqLens->GetDataType() != ACL_INT32) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "qSeqLens only supports int32");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (kvSeqLens->GetDataType() != ACL_INT32) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "kvSeqLens only supports int32");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qSeqLens->GetViewShape().GetDim(0) != kvSeqLens->GetViewShape().GetDim(0)) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "qSeqLens must have the same length as kvSeqLens, but got %ld and %ld",
                qSeqLens->GetViewShape().GetDim(0), kvSeqLens->GetViewShape().GetDim(0));
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qSeqLens->GetViewShape().GetDim(0) > STEM_INDEXER_BATCH_SIZE_LIMIT) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "batchSize must be less than or equal to %ld, but got %ld",
                STEM_INDEXER_BATCH_SIZE_LIMIT, qSeqLens->GetViewShape().GetDim(0));
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qHeads != Q_HEADS_32 && qHeads != Q_HEADS_64) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "qHeads supports %ld or %ld, but got %ld", Q_HEADS_32, Q_HEADS_64, qHeads);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (kvHeads != KV_HEADS_2 && kvHeads != KV_HEADS_4 && kvHeads != KV_HEADS_8) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "kvHeads supports %ld, %ld or %ld, but got %ld", KV_HEADS_2, KV_HEADS_4,
                KV_HEADS_8, kvHeads);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (stemBlockSize != STEM_BLOCK_SIZE_128) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "stemBlockSize supports %ld, but got %ld", STEM_BLOCK_SIZE_128, stemBlockSize);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (dimQkflat != DIM_QKFLAT_2048) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "dimQkflat supports %ld, but got %ld", DIM_QKFLAT_2048, dimQkflat);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (windowSize != WINDOW_SIZE_4) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "windowSize supports %ld, but got %ld", WINDOW_SIZE_4, windowSize);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (!IsTensorExist(metadata) || metadata->GetViewShape().GetDimNum() != 1 ||
        metadata->GetViewShape().GetDim(0) <= 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "metadata must be a non-empty 1D tensor");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (metadata->GetDataType() != ACL_INT32) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "metadata only supports int32");
        return ACLNN_ERR_PARAM_INVALID;
    }

    int64_t metadataCapacity = CalcMetadataCapacity(qSeqLens->GetViewShape().GetDim(0), kvHeads);
    if (metadata->GetViewShape().GetDim(0) < metadataCapacity) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                "metadata dim0 must be at least %ld for batch %ld and kvHeads %ld, but got %ld", metadataCapacity,
                qSeqLens->GetViewShape().GetDim(0), kvHeads, metadata->GetViewShape().GetDim(0));
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}
