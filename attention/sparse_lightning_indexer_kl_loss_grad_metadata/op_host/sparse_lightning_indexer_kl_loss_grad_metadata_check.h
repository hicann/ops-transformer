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
 * \file sparse_lightning_indexer_kl_loss_grad_metadata_check.h
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

static constexpr const char *SLI_ACLNN_OP_NAME = "aclnnSparseLightningIndexerKLLossGradMetadata";

inline constexpr int64_t SLI_NO_MASK_MODE = 0;
inline constexpr int64_t SLI_CAUSAL_MASK_MODE = 3;
inline constexpr int64_t SLI_CMP_RATIO_LOWER_BOUND = 1;
inline constexpr int64_t SLI_CMP_RATIO_UPPER_BOUND = 128;
inline constexpr int64_t SLI_NUM_HEADS_Q_LOWER_BOUND_A5 = 1;
inline constexpr int64_t SLI_NUM_HEADS_Q_UPPER_BOUND_A5 = 128;
inline constexpr int64_t SLI_TOPK_LOWER_BOUND_A5 = 1;
inline constexpr int64_t SLI_TOPK_UPPER_BOUND_A5 = 2048;
inline constexpr int64_t SLIKG_METADATA_SIZE = 64;

inline bool IsTensorExistSli(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0);
}

int64_t GetDimNumSli(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return -1;
    }
    return tensor->GetViewShape().GetDimNum();
}

aclDataType GetDataTypeSli(const aclTensor *tensor)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (tensor == nullptr) {
        return dataType;
    }
    aclGetDataType(tensor, &dataType);
    return dataType;
}

inline bool IsTensorSourceSli(const std::string &source)
{
    return source != "batch_size";
}

inline int64_t GetRawShapeSizeSli(const std::string &source, int64_t batchValue)
{
    if (source.find("cu_seqlens") != std::string::npos) {
        return batchValue + 1;
    }
    return batchValue;
}

inline std::string GetSourceDescSli(const std::string &source)
{
    if (source == "batch_size") {
        return "batch_size";
    }
    if (source.find("cu_seqlens") != std::string::npos) {
        return "the shape size of " + source + " minus 1";
    }
    return "the shape size of " + source;
}

int64_t GetQueryBatchSizeSli(int64_t batchSize, const aclTensor *cuSeqlensQOptional, const aclTensor *sequsedQOptional,
                             const char *layoutQOptional, std::string *source)
{
    if (IsTensorExistSli(sequsedQOptional)) {
        *source = "seqused_q";
        return sequsedQOptional->GetViewShape().GetDim(0);
    }
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (IsTensorExistSli(cuSeqlensQOptional)) {
            *source = "cu_seqlens_q";
            return cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    *source = "batch_size";
    return batchSize;
}

int64_t GetKeyBatchSizeSli(int64_t batchSize, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedKOptional,
                           const char *layoutKOptional, std::string *source)
{
    if (IsTensorExistSli(sequsedKOptional)) {
        *source = "seqused_k";
        return sequsedKOptional->GetViewShape().GetDim(0);
    }
    if (strcmp(layoutKOptional, "TND") == 0) {
        if (IsTensorExistSli(cuSeqlensKOptional)) {
            *source = "cu_seqlens_k";
            return cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    *source = "batch_size";
    return batchSize;
}

aclnnStatus CheckSingleParamSli(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK, int64_t numHeadsQ,
                                int64_t numHeadsK, int64_t headDim, int64_t topk, const char *layoutQOptional,
                                const char *layoutKOptional, int64_t maskMode, int64_t cmpRatio, uint32_t aicCoreNum,
                                uint32_t aivCoreNum, const std::string &socVersion)
{
    // num_heads_q 校验
    if (numHeadsQ < SLI_NUM_HEADS_Q_LOWER_BOUND_A5 || numHeadsQ > SLI_NUM_HEADS_Q_UPPER_BOUND_A5) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SLI_ACLNN_OP_NAME, "num_heads_q", std::to_string(numHeadsQ),
                                              "The current value is not within the valid range. "
                                              "The valid range is [" +
                                                  std::to_string(SLI_NUM_HEADS_Q_LOWER_BOUND_A5) + ", " +
                                                  std::to_string(SLI_NUM_HEADS_Q_UPPER_BOUND_A5) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_k 校验
    if (numHeadsK != 1) {
        OP_LOGE_FOR_INVALID_VALUE(SLI_ACLNN_OP_NAME, "num_heads_kv", std::to_string(numHeadsK), "1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // head_dim 校验
    if (headDim != 128) {
        OP_LOGE_FOR_INVALID_VALUE(SLI_ACLNN_OP_NAME, "head_dim", std::to_string(headDim), "128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // topk 校验
    if (topk < SLI_TOPK_LOWER_BOUND_A5 || topk > SLI_TOPK_UPPER_BOUND_A5) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SLI_ACLNN_OP_NAME, "topk", std::to_string(topk),
                                              "The current value is not within the valid range. "
                                              "The valid range is [" +
                                                  std::to_string(SLI_TOPK_LOWER_BOUND_A5) + ", " +
                                                  std::to_string(SLI_TOPK_UPPER_BOUND_A5) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // batch_size 非负校验
    if (batchSize < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SLI_ACLNN_OP_NAME, "batch_size", std::to_string(batchSize),
                                              "The value of batch_size must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q 校验
    if (maxSeqlenQ < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SLI_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqlenQ),
                                              "The value of max_seqlen_q must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_k 校验
    if (maxSeqlenK < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SLI_ACLNN_OP_NAME, "max_seqlen_k", std::to_string(maxSeqlenK),
                                              "The value of max_seqlen_k must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // mask_mode 校验
    if (maskMode != SLI_NO_MASK_MODE && maskMode != SLI_CAUSAL_MASK_MODE) {
        OP_LOGE_FOR_INVALID_VALUE(SLI_ACLNN_OP_NAME, "mask_mode", std::to_string(maskMode), "0 or 3");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // cmp_ratio 校验
    if (cmpRatio < SLI_CMP_RATIO_LOWER_BOUND || cmpRatio > SLI_CMP_RATIO_UPPER_BOUND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SLI_ACLNN_OP_NAME, "cmp_ratio", std::to_string(cmpRatio),
                                              "The current value is not within the valid range. "
                                              "The valid range is [" +
                                                  std::to_string(SLI_CMP_RATIO_LOWER_BOUND) + ", " +
                                                  std::to_string(SLI_CMP_RATIO_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q 校验
    if (layoutQOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SLI_ACLNN_OP_NAME, "layout_q", "layout_q cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutQOptional, "TND") != 0 && strcmp(layoutQOptional, "BSND") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(SLI_ACLNN_OP_NAME, "layout_q", layoutQOptional, "TND or BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_k 校验
    if (layoutKOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SLI_ACLNN_OP_NAME, "layout_k", "layout_k cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutKOptional, "TND") != 0 && strcmp(layoutKOptional, "BSND") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(SLI_ACLNN_OP_NAME, "layout_k", layoutKOptional, "TND or BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q必须与layout_k相同
    if (strcmp(layoutQOptional, layoutKOptional) != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(SLI_ACLNN_OP_NAME, "layout_q, layout_k",
                                               std::string(layoutQOptional) + ", " + std::string(layoutKOptional),
                                               "The value of layout_q must be equal to that of layout_k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 layout_q 为 BSND 时，max_seqlen_q 必须大于 0
    if (strcmp(layoutQOptional, "BSND") == 0 && maxSeqlenQ <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SLI_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqlenQ),
                                              "When layout_q is BSND, the value of max_seqlen_q "
                                              "must be equal to the size of the second axis of q");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 核心数校验
    if (aicCoreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SLI_ACLNN_OP_NAME, "aic_core_num", std::to_string(aicCoreNum),
                                              "The value of aic_core_num must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aivCoreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SLI_ACLNN_OP_NAME, "aiv_core_num", std::to_string(aivCoreNum),
                                              "The value of aiv_core_num must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceSli(int64_t maskMode, int64_t cmpRatio, const aclTensor *cuSeqlensQOptional,
                              const aclTensor *cuSeqlensKOptional, const aclTensor *cmpResidualKOptional,
                              const char *layoutQOptional, const char *layoutKOptional, const aclTensor *metadata)
{
    // cu_seqlens_q 存在性校验
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (!IsTensorExistSli(cuSeqlensQOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SLI_ACLNN_OP_NAME, "cu_seqlens_q",
                                                     "When layout_q is TND, cu_seqlens_q cannot be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // cu_seqlens_k 存在性校验
    if (strcmp(layoutKOptional, "TND") == 0) {
        if (!IsTensorExistSli(cuSeqlensKOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SLI_ACLNN_OP_NAME, "cu_seqlens_k",
                                                     "When layout_k is TND, cu_seqlens_k cannot be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // cmp_residual_k 存在性校验
    if (cmpRatio != SLI_CMP_RATIO_LOWER_BOUND && maskMode == SLI_CAUSAL_MASK_MODE) {
        if (!IsTensorExistSli(cmpResidualKOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SLI_ACLNN_OP_NAME, "cmp_residual_k",
                                                     "When cmp_ratio is not 1 and mask_mode is CAUSAL, "
                                                     "cmp_residual_k cannot be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // metadata 存在性校验
    if (!IsTensorExistSli(metadata)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SLI_ACLNN_OP_NAME, "metadata", "metadata cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistencySli(int64_t batchSize, const aclTensor *cuSeqlensQOptional,
                                const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
                                const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional,
                                const char *layoutQOptional, const char *layoutKOptional, const aclTensor *metadata)
{
    int64_t dimNum = -1;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;

    // 校验 cu_seqlens_q
    if (IsTensorExistSli(cuSeqlensQOptional)) {
        dimNum = GetDimNumSli(cuSeqlensQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SLI_ACLNN_OP_NAME, "cu_seqlens_q", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeSli(cuSeqlensQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SLI_ACLNN_OP_NAME, "cu_seqlens_q", ToString(dataType).GetString(),
                                                  "The dtype of cu_seqlens_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cu_seqlens_k
    if (IsTensorExistSli(cuSeqlensKOptional)) {
        dimNum = GetDimNumSli(cuSeqlensKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SLI_ACLNN_OP_NAME, "cu_seqlens_k", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeSli(cuSeqlensKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SLI_ACLNN_OP_NAME, "cu_seqlens_k", ToString(dataType).GetString(),
                                                  "The dtype of cu_seqlens_k must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_q
    if (IsTensorExistSli(sequsedQOptional)) {
        dimNum = GetDimNumSli(sequsedQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SLI_ACLNN_OP_NAME, "seqused_q", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeSli(sequsedQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SLI_ACLNN_OP_NAME, "seqused_q", ToString(dataType).GetString(),
                                                  "The dtype of seqused_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_k
    if (IsTensorExistSli(sequsedKOptional)) {
        dimNum = GetDimNumSli(sequsedKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SLI_ACLNN_OP_NAME, "seqused_k", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeSli(sequsedKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SLI_ACLNN_OP_NAME, "seqused_k", ToString(dataType).GetString(),
                                                  "The dtype of seqused_k must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cmp_residual_k
    if (IsTensorExistSli(cmpResidualKOptional)) {
        dimNum = GetDimNumSli(cmpResidualKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SLI_ACLNN_OP_NAME, "cmp_residual_k", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeSli(cmpResidualKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SLI_ACLNN_OP_NAME, "cmp_residual_k", ToString(dataType).GetString(),
                                                  "The dtype of cmp_residual_k must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 metadata
    if (IsTensorExistSli(metadata)) {
        dimNum = GetDimNumSli(metadata);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SLI_ACLNN_OP_NAME, "metadata", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeSli(metadata);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SLI_ACLNN_OP_NAME, "metadata", ToString(dataType).GetString(),
                                                  "The dtype of metadata must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 元素数
        if (metadata->GetViewShape().GetDim(0) != SLIKG_METADATA_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(SLI_ACLNN_OP_NAME, "metadata",
                                          std::to_string(metadata->GetViewShape().GetDim(0)),
                                          std::to_string(SLIKG_METADATA_SIZE));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验batch
    std::string querySource;
    int64_t queryBatchSize =
        GetQueryBatchSizeSli(batchSize, cuSeqlensQOptional, sequsedQOptional, layoutQOptional, &querySource);
    std::string keySource;
    int64_t keyBatchSize =
        GetKeyBatchSizeSli(batchSize, cuSeqlensKOptional, sequsedKOptional, layoutKOptional, &keySource);
    if (queryBatchSize != keyBatchSize) {
        if (IsTensorSourceSli(querySource) && IsTensorSourceSli(keySource)) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                SLI_ACLNN_OP_NAME, querySource + " and " + keySource,
                std::to_string(GetRawShapeSizeSli(querySource, queryBatchSize)) + " and " +
                    std::to_string(GetRawShapeSizeSli(keySource, keyBatchSize)),
                GetSourceDescSli(querySource) + " must be equal to " + GetSourceDescSli(keySource));
        } else if (IsTensorSourceSli(querySource)) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(SLI_ACLNN_OP_NAME, querySource,
                                                      std::to_string(GetRawShapeSizeSli(querySource, queryBatchSize)),
                                                      GetSourceDescSli(querySource) + " must be equal to batch_size");
        } else {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(SLI_ACLNN_OP_NAME, keySource,
                                                      std::to_string(GetRawShapeSizeSli(keySource, keyBatchSize)),
                                                      GetSourceDescSli(keySource) + " must be equal to batch_size");
        }
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验TND场景q维度一致性
    if (strcmp(layoutQOptional, "TND") == 0 && IsTensorExistSli(sequsedQOptional)) {
        int64_t cuSeqlensQBatchSize = cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        if (cuSeqlensQBatchSize != queryBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(SLI_ACLNN_OP_NAME, "cu_seqlens_q and seqused_q",
                                                       std::to_string(cuSeqlensQOptional->GetViewShape().GetDim(0)) +
                                                           " and " +
                                                           std::to_string(sequsedQOptional->GetViewShape().GetDim(0)),
                                                       "When layout_q is TND and seqused_q is passed, "
                                                       "the shape size of cu_seqlens_q minus 1 must be equal to "
                                                       "the shape size of seqused_q");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验TND场景k维度一致性
    if (strcmp(layoutKOptional, "TND") == 0 && IsTensorExistSli(sequsedKOptional)) {
        int64_t cuSeqlensKBatchSize = cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
        if (cuSeqlensKBatchSize != keyBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(SLI_ACLNN_OP_NAME, "cu_seqlens_k and seqused_k",
                                                       std::to_string(cuSeqlensKOptional->GetViewShape().GetDim(0)) +
                                                           " and " +
                                                           std::to_string(sequsedKOptional->GetViewShape().GetDim(0)),
                                                       "When layout_k is TND and seqused_k is passed, "
                                                       "the shape size of cu_seqlens_k minus 1 must be equal to "
                                                       "the shape size of seqused_k");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cmp_residual_k 元素数
    if (IsTensorExistSli(cmpResidualKOptional)) {
        auto cmpResidualKBatch = cmpResidualKOptional->GetViewShape().GetDim(0);
        if (cmpResidualKBatch != queryBatchSize) {
            if (IsTensorSourceSli(querySource)) {
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    SLI_ACLNN_OP_NAME, "cmp_residual_k and " + querySource,
                    std::to_string(cmpResidualKBatch) + " and " +
                        std::to_string(GetRawShapeSizeSli(querySource, queryBatchSize)),
                    "The shape size of cmp_residual_k must be equal to " + GetSourceDescSli(querySource));
            } else {
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(SLI_ACLNN_OP_NAME, "cmp_residual_k",
                                                          std::to_string(cmpResidualKBatch),
                                                          "The shape size of cmp_residual_k must be equal "
                                                          "to batch_size");
            }
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckSliA5(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional,
                             const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
                             const aclTensor *cmpResidualKOptional, int64_t batchSize, int64_t maxSeqlenQ,
                             int64_t maxSeqlenK, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk,
                             char *layoutQOptional, char *layoutKOptional, int64_t maskMode, int64_t cmpRatio,
                             const aclTensor *metadata, uint32_t aicCoreNum, uint32_t aivCoreNum,
                             const std::string &socVersion)
{
    auto ret =
        CheckSingleParamSli(batchSize, maxSeqlenQ, maxSeqlenK, numHeadsQ, numHeadsK, headDim, topk, layoutQOptional,
                            layoutKOptional, maskMode, cmpRatio, aicCoreNum, aivCoreNum, socVersion);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckExistenceSli(maskMode, cmpRatio, cuSeqlensQOptional, cuSeqlensKOptional, cmpResidualKOptional,
                            layoutQOptional, layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckConsistencySli(batchSize, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                              cmpResidualKOptional, layoutQOptional, layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckSli(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional,
                           const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
                           const aclTensor *cmpResidualKOptional, int64_t batchSize, int64_t maxSeqlenQ,
                           int64_t maxSeqlenK, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk,
                           char *layoutQOptional, char *layoutKOptional, int64_t maskMode, int64_t cmpRatio,
                           const aclTensor *metadata, uint32_t aicCoreNum, uint32_t aivCoreNum,
                           const std::string &socVersion)
{
    // A2/A3 校验
    const std::string ascend950 = "Ascend950";
    if (socVersion.find(ascend950) == std::string::npos) {
        CHECK_RET(metadata != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        return ACLNN_SUCCESS;
    }
    // A5 校验
    return ParamsCheckSliA5(cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                            cmpResidualKOptional, batchSize, maxSeqlenQ, maxSeqlenK, numHeadsQ, numHeadsK, headDim,
                            topk, layoutQOptional, layoutKOptional, maskMode, cmpRatio, metadata, aicCoreNum,
                            aivCoreNum, socVersion);
}

} // namespace

#ifdef __cplusplus
}
#endif
