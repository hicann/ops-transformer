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
 * \file sparse_flash_mla_softmax_l1_norm_metadata_check.h
 * \brief
 */

#include "log/log.h"
#include "opdev/format_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr const char *SMLA_ACLNN_OP_NAME = "aclnnSparseFlashMlaSoftmaxL1NormMetadata";

inline constexpr int64_t SMLA_NO_MASK_MODE = 0;
inline constexpr int64_t SMLA_CAUSAL_MASK_MODE = 3;
inline constexpr int64_t SMLA_CMP_RATIO_LOWER_BOUND = 1;
inline constexpr int64_t SMLA_CMP_RATIO_UPPER_BOUND = 128;
inline constexpr int64_t SMLA_METADATA_SIZE = 64;
inline constexpr int64_t SMLA_NUM_HEADS_Q_LOWER_BOUND = 1;
inline constexpr int64_t SMLA_NUM_HEADS_Q_UPPER_BOUND = 128;
inline constexpr int64_t SMLA_NUM_HEADS_K = 1;
inline constexpr int64_t SMLA_HEAD_DIM = 512;

inline bool IsTensorExistSmlaL1Norm(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0);
}

int64_t GetDimNumSmlaL1Norm(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return -1;
    }
    return tensor->GetViewShape().GetDimNum();
}

aclDataType GetDataTypeSmlaL1Norm(const aclTensor *tensor)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (tensor == nullptr) {
        return dataType;
    }
    aclGetDataType(tensor, &dataType);
    return dataType;
}

int64_t GetQueryBatchSizeSmlaL1Norm(int64_t batchSize, const aclTensor *cuSeqLensQOptional,
                                    const aclTensor *seqUsedQOptional, const char *layoutQ)
{
    if (IsTensorExistSmlaL1Norm(seqUsedQOptional)) {
        return seqUsedQOptional->GetViewShape().GetDim(0);
    }
    if (strcmp(layoutQ, "TND") == 0) {
        if (IsTensorExistSmlaL1Norm(cuSeqLensQOptional)) {
            return cuSeqLensQOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    return batchSize;
}

int64_t GetKeyBatchSizeSmlaL1Norm(int64_t batchSize, const aclTensor *cuSeqLensKOptional,
                                  const aclTensor *seqUsedKOptional, const char *layoutK)
{
    if (IsTensorExistSmlaL1Norm(seqUsedKOptional)) {
        return seqUsedKOptional->GetViewShape().GetDim(0);
    }
    if (strcmp(layoutK, "TND") == 0) {
        if (IsTensorExistSmlaL1Norm(cuSeqLensKOptional)) {
            return cuSeqLensKOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    return batchSize;
}

aclnnStatus CheckSingleParamSmlaL1Norm(int64_t batchSize, int64_t maxSeqLenQ, int64_t maxSeqLenK, int64_t numHeadsQ,
                                       int64_t numHeadsK, int64_t headDim, int64_t topk, int64_t maskMode,
                                       int64_t cmpRatio, const char *layoutQ, const char *layoutK, int64_t aicCoreNum)
{
    // batch_size 非负校验
    if (batchSize < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "batch_size", std::to_string(batchSize),
                                              "The value of batch_size must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q 非负校验
    if (maxSeqLenQ < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqLenQ),
                                              "The value of max_seqlen_q must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_k 非负校验
    if (maxSeqLenK < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "max_seqlen_k", std::to_string(maxSeqLenK),
                                              "The value of max_seqlen_k must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_q 校验
    if (numHeadsQ < SMLA_NUM_HEADS_Q_LOWER_BOUND || numHeadsQ > SMLA_NUM_HEADS_Q_UPPER_BOUND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "num_heads_q", std::to_string(numHeadsQ),
                                              "The current value is not within the valid range. "
                                              "The valid range is [" +
                                                  std::to_string(SMLA_NUM_HEADS_Q_LOWER_BOUND) + ", " +
                                                  std::to_string(SMLA_NUM_HEADS_Q_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_k 校验
    if (numHeadsK != SMLA_NUM_HEADS_K) {
        OP_LOGE_FOR_INVALID_VALUE(SMLA_ACLNN_OP_NAME, "num_heads_k", std::to_string(numHeadsK),
                                  std::to_string(SMLA_NUM_HEADS_K));
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_q 必须能被 num_heads_k 整除
    if (numHeadsQ % numHeadsK != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(SMLA_ACLNN_OP_NAME, "num_heads_q, num_heads_k",
                                               std::to_string(numHeadsQ) + ", " + std::to_string(numHeadsK),
                                               "The value of num_heads_q must be divisible by num_heads_k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // head_dim 校验
    if (headDim != SMLA_HEAD_DIM) {
        OP_LOGE_FOR_INVALID_VALUE(SMLA_ACLNN_OP_NAME, "head_dim", std::to_string(headDim),
                                  std::to_string(SMLA_HEAD_DIM));
        return ACLNN_ERR_PARAM_INVALID;
    }
    // topk 非负校验
    if (topk < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "topk", std::to_string(topk),
                                              "The value of topk must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // mask_mode 校验
    if (maskMode != SMLA_NO_MASK_MODE && maskMode != SMLA_CAUSAL_MASK_MODE) {
        OP_LOGE_FOR_INVALID_VALUE(SMLA_ACLNN_OP_NAME, "mask_mode", std::to_string(maskMode), "0 or 3");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // cmp_ratio 校验
    if (cmpRatio < SMLA_CMP_RATIO_LOWER_BOUND || cmpRatio > SMLA_CMP_RATIO_UPPER_BOUND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "cmp_ratio", std::to_string(cmpRatio),
                                              "The current value is not within the valid range. "
                                              "The valid range is [" +
                                                  std::to_string(SMLA_CMP_RATIO_LOWER_BOUND) + ", " +
                                                  std::to_string(SMLA_CMP_RATIO_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q 校验
    if (layoutQ == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "layout_q", "layout_q cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutQ, "TND") != 0 && strcmp(layoutQ, "BSND") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(SMLA_ACLNN_OP_NAME, "layout_q", layoutQ, "TND or BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 layout_q 为 BSND 时，max_seqlen_q 必须大于 0
    if (strcmp(layoutQ, "BSND") == 0 && maxSeqLenQ <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqLenQ),
                                              "When layout_q is BSND, the value of max_seqlen_q must be "
                                              "greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_k 校验
    if (layoutK == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "layout_k", "layout_k cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutK, "TND") != 0 && strcmp(layoutK, "BSND") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(SMLA_ACLNN_OP_NAME, "layout_k", layoutK, "TND or BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // aic_core_num 校验
    if (aicCoreNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "aic_core_num", std::to_string(aicCoreNum),
                                              "The value of aic_core_num must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceSmlaL1Norm(const aclTensor *cuSeqLensQOptional, const aclTensor *cuSeqLensKOptional,
                                     const aclTensor *topkLengthOptional, const aclTensor *cmpResidualKOptional,
                                     int64_t topk, int64_t maskMode, int64_t cmpRatio, const char *layoutQ,
                                     const char *layoutK, const aclTensor *metadata)
{
    // layout_q 为 TND 时，cu_seq_lens_q 必传
    if (strcmp(layoutQ, "TND") == 0) {
        if (!IsTensorExistSmlaL1Norm(cuSeqLensQOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "cu_seq_lens_q",
                                                     "When layout_q is TND, cu_seq_lens_q cannot be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // layout_k 为 TND 时，cu_seq_lens_k 必传
    if (strcmp(layoutK, "TND") == 0) {
        if (!IsTensorExistSmlaL1Norm(cuSeqLensKOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "cu_seq_lens_k",
                                                     "When layout_k is TND, cu_seq_lens_k cannot be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // mask_mode 为 0 且 topk 大于 0 时，topk_length 必传
    if (maskMode == SMLA_NO_MASK_MODE && topk > 0) {
        if (!IsTensorExistSmlaL1Norm(topkLengthOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                SMLA_ACLNN_OP_NAME, "topk_length",
                "When mask_mode is 0 and sparse index exists (topk > 0), topk_length cannot be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // mask_mode 为 3 且 cmp_ratio 不为 1 时，cmp_residual_k 必传
    if (maskMode == SMLA_CAUSAL_MASK_MODE && cmpRatio != SMLA_CMP_RATIO_LOWER_BOUND) {
        if (!IsTensorExistSmlaL1Norm(cmpResidualKOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "cmp_residual_k",
                                                     "When mask_mode is 3 and cmp_ratio is not 1, "
                                                     "cmp_residual_k cannot be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // metadata 必传
    if (!IsTensorExistSmlaL1Norm(metadata)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "metadata", "metadata cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistencySmlaL1Norm(int64_t batchSize, const aclTensor *cuSeqLensQOptional,
                                       const aclTensor *cuSeqLensKOptional, const aclTensor *seqUsedQOptional,
                                       const aclTensor *seqUsedKOptional, const aclTensor *cmpResidualKOptional,
                                       const aclTensor *topkLengthOptional, const char *layoutQ, const char *layoutK,
                                       const aclTensor *metadata)
{
    int64_t dimNum = -1;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;

    // 校验 cu_seq_lens_q
    if (IsTensorExistSmlaL1Norm(cuSeqLensQOptional)) {
        dimNum = GetDimNumSmlaL1Norm(cuSeqLensQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "cu_seq_lens_q", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeSmlaL1Norm(cuSeqLensQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "cu_seq_lens_q", ToString(dataType).GetString(),
                                                  "The dtype of cu_seq_lens_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cu_seq_lens_k
    if (IsTensorExistSmlaL1Norm(cuSeqLensKOptional)) {
        dimNum = GetDimNumSmlaL1Norm(cuSeqLensKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "cu_seq_lens_k", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeSmlaL1Norm(cuSeqLensKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "cu_seq_lens_k", ToString(dataType).GetString(),
                                                  "The dtype of cu_seq_lens_k must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seq_used_q
    if (IsTensorExistSmlaL1Norm(seqUsedQOptional)) {
        dimNum = GetDimNumSmlaL1Norm(seqUsedQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "seq_used_q", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeSmlaL1Norm(seqUsedQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "seq_used_q", ToString(dataType).GetString(),
                                                  "The dtype of seq_used_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seq_used_k
    if (IsTensorExistSmlaL1Norm(seqUsedKOptional)) {
        dimNum = GetDimNumSmlaL1Norm(seqUsedKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "seq_used_k", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeSmlaL1Norm(seqUsedKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "seq_used_k", ToString(dataType).GetString(),
                                                  "The dtype of seq_used_k must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cmp_residual_k
    if (IsTensorExistSmlaL1Norm(cmpResidualKOptional)) {
        dimNum = GetDimNumSmlaL1Norm(cmpResidualKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "cmp_residual_k", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeSmlaL1Norm(cmpResidualKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "cmp_residual_k", ToString(dataType).GetString(),
                                                  "The dtype of cmp_residual_k must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 topk_length
    if (IsTensorExistSmlaL1Norm(topkLengthOptional)) {
        dimNum = GetDimNumSmlaL1Norm(topkLengthOptional);
        if (strcmp(layoutQ, "BSND") == 0) {
            if (dimNum != 3) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "topk_length", std::to_string(dimNum), "3");
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (dimNum != 2) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "topk_length", std::to_string(dimNum), "2");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        dataType = GetDataTypeSmlaL1Norm(topkLengthOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "topk_length", ToString(dataType).GetString(),
                                                  "The dtype of topk_length must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 metadata
    if (IsTensorExistSmlaL1Norm(metadata)) {
        dimNum = GetDimNumSmlaL1Norm(metadata);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "metadata", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeSmlaL1Norm(metadata);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "metadata", ToString(dataType).GetString(),
                                                  "The dtype of metadata must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 元素数
        if (metadata->GetViewShape().GetDim(0) != SMLA_METADATA_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(SMLA_ACLNN_OP_NAME, "metadata",
                                          std::to_string(metadata->GetViewShape().GetDim(0)),
                                          std::to_string(SMLA_METADATA_SIZE));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 有效 batch 大小校验
    int64_t queryBatchSize = GetQueryBatchSizeSmlaL1Norm(batchSize, cuSeqLensQOptional, seqUsedQOptional, layoutQ);
    int64_t keyBatchSize = GetKeyBatchSizeSmlaL1Norm(batchSize, cuSeqLensKOptional, seqUsedKOptional, layoutK);
    if (queryBatchSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "batch_size", std::to_string(queryBatchSize),
                                              "The valid batch size must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // query 与 key 侧有效 batch 一致性校验
    if (queryBatchSize != keyBatchSize) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(SMLA_ACLNN_OP_NAME, "query batch size, key batch size",
                                               std::to_string(queryBatchSize) + ", " + std::to_string(keyBatchSize),
                                               "The batch_size obtained from query must be the same as "
                                               "that obtained from key");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 cu_seq_lens_q 长度
    if (IsTensorExistSmlaL1Norm(cuSeqLensQOptional)) {
        int64_t cuSeqLensQLen = cuSeqLensQOptional->GetViewShape().GetDim(0);
        if (cuSeqLensQLen != queryBatchSize + 1) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(SMLA_ACLNN_OP_NAME, "cu_seq_lens_q", std::to_string(cuSeqLensQLen),
                                          std::to_string(queryBatchSize + 1));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cu_seq_lens_k 长度
    if (IsTensorExistSmlaL1Norm(cuSeqLensKOptional)) {
        int64_t cuSeqLensKLen = cuSeqLensKOptional->GetViewShape().GetDim(0);
        if (cuSeqLensKLen != keyBatchSize + 1) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(SMLA_ACLNN_OP_NAME, "cu_seq_lens_k", std::to_string(cuSeqLensKLen),
                                          std::to_string(keyBatchSize + 1));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seq_used_q 长度
    if (IsTensorExistSmlaL1Norm(seqUsedQOptional)) {
        if (seqUsedQOptional->GetViewShape().GetDim(0) != queryBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(SMLA_ACLNN_OP_NAME, "seq_used_q",
                                          std::to_string(seqUsedQOptional->GetViewShape().GetDim(0)),
                                          std::to_string(queryBatchSize));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seq_used_k 长度
    if (IsTensorExistSmlaL1Norm(seqUsedKOptional)) {
        if (seqUsedKOptional->GetViewShape().GetDim(0) != keyBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(SMLA_ACLNN_OP_NAME, "seq_used_k",
                                          std::to_string(seqUsedKOptional->GetViewShape().GetDim(0)),
                                          std::to_string(keyBatchSize));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cmp_residual_k 长度
    if (IsTensorExistSmlaL1Norm(cmpResidualKOptional)) {
        auto cmpResidualKBatch = cmpResidualKOptional->GetViewShape().GetDim(0);
        if (cmpResidualKBatch != queryBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(SMLA_ACLNN_OP_NAME, "cmp_residual_k", std::to_string(cmpResidualKBatch),
                                          std::to_string(queryBatchSize));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckSmlaL1Norm(const aclTensor *cuSeqLensQOptional, const aclTensor *cuSeqLensKOptional,
                                  const aclTensor *seqUsedQOptional, const aclTensor *seqUsedKOptional,
                                  const aclTensor *cmpResidualKOptional, const aclTensor *topkLengthOptional,
                                  int64_t batchSize, int64_t maxSeqLenQ, int64_t maxSeqLenK, int64_t numHeadsQ,
                                  int64_t numHeadsK, int64_t headDim, int64_t topk, int64_t maskMode, int64_t cmpRatio,
                                  char *layoutQ, char *layoutK, const aclTensor *metadata, int64_t aicCoreNum)
{
    auto ret = CheckSingleParamSmlaL1Norm(batchSize, maxSeqLenQ, maxSeqLenK, numHeadsQ, numHeadsK, headDim, topk,
                                          maskMode, cmpRatio, layoutQ, layoutK, aicCoreNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckExistenceSmlaL1Norm(cuSeqLensQOptional, cuSeqLensKOptional, topkLengthOptional, cmpResidualKOptional,
                                   topk, maskMode, cmpRatio, layoutQ, layoutK, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckConsistencySmlaL1Norm(batchSize, cuSeqLensQOptional, cuSeqLensKOptional, seqUsedQOptional,
                                     seqUsedKOptional, cmpResidualKOptional, topkLengthOptional, layoutQ, layoutK,
                                     metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
}
#endif
