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
 * \file generic_block_sparse_attention_grad_metadata_check.h
 * \brief Host-side parameter validation for GenericBlockSparseAttentionGradMetadata.
 */

#include <cstring>
#include <string>
#include "log/log.h"
#include "opdev/format_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include "../op_kernel/generic_block_sparse_attention_grad_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr const char *GSAG_ACLNN_OP_NAME = "aclnnGenericBlockSparseAttentionGradMetadata";

inline bool IsTensorExist(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0);
}

static constexpr int64_t GSAG_MAX_HEAD_NUM = 128;
static constexpr int64_t GSAG_SUPPORTED_HEAD_DIM = 128;
static constexpr int64_t GSAG_BLOCK_SHAPE_Y_MIN = 128;
static constexpr int64_t GSAG_BLOCK_SHAPE_Y_ALIGN = 64;
static constexpr int64_t GSAG_SUPPORTED_MASK_TYPE = 1;
// rsvd_block_idx: [B, N2, J, maxS1]; rsvd_block_count: [B, N2, J]
static constexpr size_t RSVD_BLOCK_IDX_DIM_NUM = 4;
static constexpr size_t RSVD_BLOCK_COUNT_DIM_NUM = 3;
static constexpr size_t DIM_B = 0;
static constexpr size_t DIM_N2 = 1;
static constexpr size_t DIM_J = 2;
static constexpr size_t DIM_MAX_S1 = 3;

aclnnStatus CheckSingleParam(int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t numQHeads, int64_t numKvHeads,
                             int64_t headDim, int64_t blockShapeX, int64_t blockShapeY, int64_t isPackedGQA,
                             const char *layoutQ, const char *layoutKv, int64_t maskType, int64_t winLeft,
                             int64_t winRight, uint32_t aicCoreNum, uint32_t aivCoreNum)
{
    if (maxQSeqlen < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GSAG_ACLNN_OP_NAME, "max_q_seqlen", std::to_string(maxQSeqlen),
                                              "The value of max_q_seqlen must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (maxKvSeqlen < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GSAG_ACLNN_OP_NAME, "max_kv_seqlen", std::to_string(maxKvSeqlen),
                                              "The value of max_kv_seqlen must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numQHeads <= 0 || numQHeads > GSAG_MAX_HEAD_NUM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GSAG_ACLNN_OP_NAME, "num_q_heads", std::to_string(numQHeads),
                                              "The value of num_q_heads must be in [1, 128]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numKvHeads <= 0 || numKvHeads > GSAG_MAX_HEAD_NUM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GSAG_ACLNN_OP_NAME, "num_kv_heads", std::to_string(numKvHeads),
                                              "The value of num_kv_heads must be in [1, 128]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numQHeads % numKvHeads != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(GSAG_ACLNN_OP_NAME, "num_q_heads, num_kv_heads",
                                               std::to_string(numQHeads) + ", " + std::to_string(numKvHeads),
                                               "num_q_heads must be divisible by num_kv_heads");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (maskType != GSAG_SUPPORTED_MASK_TYPE) {
        OP_LOGE_FOR_INVALID_VALUE(GSAG_ACLNN_OP_NAME, "mask_type", std::to_string(maskType), "1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (headDim != GSAG_SUPPORTED_HEAD_DIM) {
        OP_LOGE_FOR_INVALID_VALUE(GSAG_ACLNN_OP_NAME, "head_dim", std::to_string(headDim), "128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (blockShapeX != 1) {
        OP_LOGE_FOR_INVALID_VALUE(GSAG_ACLNN_OP_NAME, "block_shape[0]", std::to_string(blockShapeX), "1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (blockShapeY < GSAG_BLOCK_SHAPE_Y_MIN || blockShapeY % GSAG_BLOCK_SHAPE_Y_ALIGN != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GSAG_ACLNN_OP_NAME, "block_shape[1]", std::to_string(blockShapeY),
                                              "block_shape[1] must be >= 128 and aligned to 64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (isPackedGQA != 1) {
        OP_LOGE_FOR_INVALID_VALUE(GSAG_ACLNN_OP_NAME, "is_packed_gqa", std::to_string(isPackedGQA), "1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutQ == nullptr || layoutKv == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(GSAG_ACLNN_OP_NAME, "layout_q/layout_kv",
                                                 "layout_q and layout_kv cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutQ, "TND") != 0 && strcmp(layoutQ, "BNSD") != 0 && strcmp(layoutQ, "BSND") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(GSAG_ACLNN_OP_NAME, "layout_q", layoutQ, "TND, BNSD or BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutKv, "TND") != 0 && strcmp(layoutKv, "BNSD") != 0 && strcmp(layoutKv, "BSND") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(GSAG_ACLNN_OP_NAME, "layout_kv", layoutKv, "TND, BNSD or BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutQ, layoutKv) != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(GSAG_ACLNN_OP_NAME, "layout_q, layout_kv",
                                               std::string(layoutQ) + ", " + std::string(layoutKv),
                                               "layout_q must equal layout_kv");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (winLeft != -1 || winRight != -1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GSAG_ACLNN_OP_NAME, "window_size_left/right",
                                              std::to_string(winLeft) + ", " + std::to_string(winRight),
                                              "window_size_left and window_size_right must be -1 in current version");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aicCoreNum == 0 || aivCoreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GSAG_ACLNN_OP_NAME, "aic_core_num/aiv_core_num",
                                              std::to_string(aicCoreNum) + ", " + std::to_string(aivCoreNum),
                                              "core num must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistence(const aclTensor *rsvdBlockIdx, const aclTensor *rsvdBlockCount,
                           const aclTensor *cuSeqLengthsQOptional, const aclTensor *cuSeqLengthsKvOptional,
                           const char *layoutQ, const char *layoutKv, const aclTensor *metadata)
{
    if (!IsTensorExist(rsvdBlockIdx)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(GSAG_ACLNN_OP_NAME, "rsvd_block_idx",
                                                 "rsvd_block_idx cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsTensorExist(rsvdBlockCount)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(GSAG_ACLNN_OP_NAME, "rsvd_block_count",
                                                 "rsvd_block_count cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutQ, "TND") == 0) {
        if (!IsTensorExist(cuSeqLengthsQOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(GSAG_ACLNN_OP_NAME, "cu_seq_lengths",
                                                     "cu_seq_lengths is required when layout_q is TND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!IsTensorExist(cuSeqLengthsKvOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(GSAG_ACLNN_OP_NAME, "cu_seq_lengths_kv",
                                                     "cu_seq_lengths_kv is required when layout_kv is TND");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!IsTensorExist(metadata)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(GSAG_ACLNN_OP_NAME, "metadata", "metadata cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistency(const aclTensor *rsvdBlockIdx, const aclTensor *rsvdBlockCount, int64_t maxQSeqlen,
                             int64_t maxKvSeqlen, int64_t numQHeads, int64_t numKvHeads, int64_t blockShapeY,
                             int64_t isPackedGQA, const aclTensor *metadata)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (rsvdBlockIdx->GetViewShape().GetDimNum() != RSVD_BLOCK_IDX_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(GSAG_ACLNN_OP_NAME, "rsvd_block_idx",
                                     std::to_string(rsvdBlockIdx->GetViewShape().GetDimNum()), "4");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (rsvdBlockCount->GetViewShape().GetDimNum() != RSVD_BLOCK_COUNT_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(GSAG_ACLNN_OP_NAME, "rsvd_block_count",
                                     std::to_string(rsvdBlockCount->GetViewShape().GetDimNum()), "3");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const auto &idxShape = rsvdBlockIdx->GetViewShape();
    const auto &cntShape = rsvdBlockCount->GetViewShape();
    int64_t batchSize = idxShape.GetDim(DIM_B);
    int64_t n2 = idxShape.GetDim(DIM_N2);
    int64_t j = idxShape.GetDim(DIM_J);
    int64_t maxS1 = idxShape.GetDim(DIM_MAX_S1);

    if (cntShape.GetDim(DIM_B) != batchSize || cntShape.GetDim(DIM_N2) != n2 || cntShape.GetDim(DIM_J) != j) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            GSAG_ACLNN_OP_NAME, "rsvd_block_count",
            std::to_string(cntShape.GetDim(DIM_B)) + ", " + std::to_string(cntShape.GetDim(DIM_N2)) + ", " +
                std::to_string(cntShape.GetDim(DIM_J)),
            "rsvd_block_count shape must be [B, N2, J] and consistent with rsvd_block_idx");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (n2 != numKvHeads) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(GSAG_ACLNN_OP_NAME, "rsvd_block_idx N2", std::to_string(n2),
                                                  "N2 must equal num_kv_heads");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (maxS1 < maxQSeqlen) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(GSAG_ACLNN_OP_NAME, "rsvd_block_idx maxS1", std::to_string(maxS1),
                                                  "last dim of rsvd_block_idx must be >= max_q_seqlen");
        return ACLNN_ERR_PARAM_INVALID;
    }

    int64_t expectedJ = (maxKvSeqlen + blockShapeY - 1) / blockShapeY;
    if (j != expectedJ) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(GSAG_ACLNN_OP_NAME, "rsvd_block_idx J", std::to_string(j),
                                                  "J must equal ceilDiv(max_kv_seqlen, block_shape[1])");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const uint64_t maxPossibleTasks = optiling::CalcGsagMetadataMaxTasks(
        static_cast<uint64_t>(batchSize), static_cast<uint64_t>(numQHeads), static_cast<uint64_t>(j));
    if (maxPossibleTasks > optiling::GSAG_METADATA_ABSOLUTE_MAX_TASKS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GSAG_ACLNN_OP_NAME, "task_count_upper_bound",
                                              std::to_string(maxPossibleTasks),
                                              "task count upper bound B * num_q_heads * J exceeds absolute max " +
                                                  std::to_string(optiling::GSAG_METADATA_ABSOLUTE_MAX_TASKS));
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t requiredMetadataSize = static_cast<int64_t>(optiling::CalcGsagMetadataSize(
        static_cast<uint64_t>(batchSize), static_cast<uint64_t>(numQHeads), static_cast<uint64_t>(j)));

    aclGetDataType(rsvdBlockIdx, &dataType);
    if (dataType != aclDataType::ACL_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(GSAG_ACLNN_OP_NAME, "rsvd_block_idx", ToString(dataType).GetString(),
                                              "dtype of rsvd_block_idx must be int32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    aclGetDataType(rsvdBlockCount, &dataType);
    if (dataType != aclDataType::ACL_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(GSAG_ACLNN_OP_NAME, "rsvd_block_count", ToString(dataType).GetString(),
                                              "dtype of rsvd_block_count must be int32");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (metadata->GetViewShape().GetDimNum() != 1 || metadata->GetViewShape().GetDim(0) < requiredMetadataSize) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(GSAG_ACLNN_OP_NAME, "metadata",
                                                  std::to_string(metadata->GetViewShape().GetDim(0)),
                                                  "metadata length must be >= " + std::to_string(requiredMetadataSize) +
                                                      " (TASK_LIST_OFFSET + B * num_q_heads * J * TASK_ENTRY_SIZE)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    aclGetDataType(metadata, &dataType);
    if (dataType != aclDataType::ACL_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(GSAG_ACLNN_OP_NAME, "metadata", ToString(dataType).GetString(),
                                              "dtype of metadata must be int64");
        return ACLNN_ERR_PARAM_INVALID;
    }

    (void)isPackedGQA;
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsCheck(const aclTensor *rsvdBlockIdx, const aclTensor *rsvdBlockCount,
                               const aclTensor *cuSeqLengthsQOptional, const aclTensor *cuSeqLengthsKvOptional,
                               const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
                               int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t numQHeads, int64_t numKvHeads,
                               int64_t headDim, int64_t blockShapeX, int64_t blockShapeY, int64_t isPackedGQA,
                               const char *layoutQ, const char *layoutKv, int64_t maskType, int64_t softmaxPrecision,
                               int64_t winLeft, int64_t winRight, uint32_t aicCoreNum, uint32_t aivCoreNum,
                               const char *socVersion, const aclTensor *metadata)
{
    (void)sequsedQOptional;
    (void)sequsedKvOptional;
    (void)softmaxPrecision;
    (void)socVersion;

    if (CheckSingleParam(maxQSeqlen, maxKvSeqlen, numQHeads, numKvHeads, headDim, blockShapeX, blockShapeY, isPackedGQA,
                         layoutQ, layoutKv, maskType, winLeft, winRight, aicCoreNum, aivCoreNum) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (CheckExistence(rsvdBlockIdx, rsvdBlockCount, cuSeqLengthsQOptional, cuSeqLengthsKvOptional, layoutQ, layoutKv,
                       metadata) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (CheckConsistency(rsvdBlockIdx, rsvdBlockCount, maxQSeqlen, maxKvSeqlen, numQHeads, numKvHeads, blockShapeY,
                         isPackedGQA, metadata) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
}
#endif
