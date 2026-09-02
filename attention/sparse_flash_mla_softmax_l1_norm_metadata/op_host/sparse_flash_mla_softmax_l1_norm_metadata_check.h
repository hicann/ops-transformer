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

#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

inline constexpr int64_t SMLA_NO_MASK_MODE = 0;
inline constexpr int64_t SMLA_CAUSAL_MASK_MODE = 3;
inline constexpr int64_t SMLA_CMP_RATIO_LOWER_BOUND = 1;
inline constexpr int64_t SMLA_CMP_RATIO_UPPER_BOUND = 128;
inline constexpr int64_t SMLA_METADATA_SIZE = 64;

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
    CHECK_COND(batchSize >= 0, ACLNN_ERR_PARAM_INVALID, "batch_size should be >= 0, but got %lld", batchSize);
    CHECK_COND(maxSeqLenQ >= 0, ACLNN_ERR_PARAM_INVALID, "max_seqlen_q should be >= 0, but got %lld", maxSeqLenQ);
    CHECK_COND(maxSeqLenK >= 0, ACLNN_ERR_PARAM_INVALID, "max_seqlen_k should be >= 0, but got %lld", maxSeqLenK);
    CHECK_COND(numHeadsQ > 0, ACLNN_ERR_PARAM_INVALID, "num_heads_q should be > 0, but got %lld", numHeadsQ);
    CHECK_COND(numHeadsK > 0, ACLNN_ERR_PARAM_INVALID, "num_heads_k should be > 0, but got %lld", numHeadsK);
    CHECK_COND(numHeadsQ % numHeadsK == 0, ACLNN_ERR_PARAM_INVALID,
               "num_heads_q must be divisible by num_heads_k, but got %lld and %lld", numHeadsQ, numHeadsK);
    CHECK_COND(headDim > 0, ACLNN_ERR_PARAM_INVALID, "head_dim should be > 0, but got %lld", headDim);
    CHECK_COND(topk >= 0, ACLNN_ERR_PARAM_INVALID, "topk should be >= 0, but got %lld", topk);
    CHECK_COND((maskMode == SMLA_NO_MASK_MODE) || (maskMode == SMLA_CAUSAL_MASK_MODE), ACLNN_ERR_PARAM_INVALID,
               "mask_mode should be %lld/%lld, but got %lld", SMLA_NO_MASK_MODE, SMLA_CAUSAL_MASK_MODE, maskMode);
    CHECK_COND((cmpRatio >= SMLA_CMP_RATIO_LOWER_BOUND) && (cmpRatio <= SMLA_CMP_RATIO_UPPER_BOUND),
               ACLNN_ERR_PARAM_INVALID, "cmp_ratio should be between [%lld, %lld], but got %lld",
               SMLA_CMP_RATIO_LOWER_BOUND, SMLA_CMP_RATIO_UPPER_BOUND, cmpRatio);
    CHECK_COND((layoutQ != nullptr), ACLNN_ERR_PARAM_INVALID, "layout_q is null!");
    CHECK_COND((strcmp(layoutQ, "TND") == 0) || (strcmp(layoutQ, "BSND") == 0), ACLNN_ERR_PARAM_INVALID,
               "layout_q must be TND or BSND, but got %s", layoutQ);
    CHECK_COND((layoutK != nullptr), ACLNN_ERR_PARAM_INVALID, "layout_k is null!");
    CHECK_COND((strcmp(layoutK, "TND") == 0) || (strcmp(layoutK, "BSND") == 0), ACLNN_ERR_PARAM_INVALID,
               "layout_k must be TND or BSND, but got %s", layoutK);
    CHECK_COND(aicCoreNum > 0, ACLNN_ERR_PARAM_INVALID, "AIC num should be larger than 0, but got %lld", aicCoreNum);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceSmlaL1Norm(const aclTensor *cuSeqLensQOptional, const aclTensor *cuSeqLensKOptional,
                                     const aclTensor *topkLengthOptional, const aclTensor *cmpResidualKOptional,
                                     int64_t topk, int64_t maskMode, int64_t cmpRatio, const char *layoutQ,
                                     const char *layoutK, const aclTensor *metadata)
{
    if (strcmp(layoutQ, "TND") == 0) {
        CHECK_COND(IsTensorExistSmlaL1Norm(cuSeqLensQOptional), ACLNN_ERR_PARAM_INVALID,
                   "For layout_q TND, cu_seq_lens_q must be provided!");
    }
    if (strcmp(layoutK, "TND") == 0) {
        CHECK_COND(IsTensorExistSmlaL1Norm(cuSeqLensKOptional), ACLNN_ERR_PARAM_INVALID,
                   "For layout_k TND, cu_seq_lens_k must be provided!");
    }
    if (maskMode == SMLA_NO_MASK_MODE && topk > 0) {
        CHECK_COND(IsTensorExistSmlaL1Norm(topkLengthOptional), ACLNN_ERR_PARAM_INVALID,
                   "When mask_mode is 0 and sparse index exists (topk > 0), topk_length must be provided!");
    }
    if (maskMode == SMLA_CAUSAL_MASK_MODE && cmpRatio != SMLA_CMP_RATIO_LOWER_BOUND) {
        CHECK_COND(IsTensorExistSmlaL1Norm(cmpResidualKOptional), ACLNN_ERR_PARAM_INVALID,
                   "When mask_mode is 3 and cmp_ratio != 1, cmp_residual_k must be provided!");
    }
    CHECK_COND(IsTensorExistSmlaL1Norm(metadata), ACLNN_ERR_PARAM_INVALID, "Output metadata is nullptr!");
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

    if (IsTensorExistSmlaL1Norm(cuSeqLensQOptional)) {
        dimNum = GetDimNumSmlaL1Norm(cuSeqLensQOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seq_lens_q must be 1, but got %lld",
                   dimNum);
        dataType = GetDataTypeSmlaL1Norm(cuSeqLensQOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
                   "The data type of cu_seq_lens_q must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    if (IsTensorExistSmlaL1Norm(cuSeqLensKOptional)) {
        dimNum = GetDimNumSmlaL1Norm(cuSeqLensKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seq_lens_k must be 1, but got %lld",
                   dimNum);
        dataType = GetDataTypeSmlaL1Norm(cuSeqLensKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
                   "The data type of cu_seq_lens_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    if (IsTensorExistSmlaL1Norm(seqUsedQOptional)) {
        dimNum = GetDimNumSmlaL1Norm(seqUsedQOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of seq_used_q must be 1, but got %lld", dimNum);
        dataType = GetDataTypeSmlaL1Norm(seqUsedQOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
                   "The data type of seq_used_q must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    if (IsTensorExistSmlaL1Norm(seqUsedKOptional)) {
        dimNum = GetDimNumSmlaL1Norm(seqUsedKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of seq_used_k must be 1, but got %lld", dimNum);
        dataType = GetDataTypeSmlaL1Norm(seqUsedKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
                   "The data type of seq_used_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    if (IsTensorExistSmlaL1Norm(cmpResidualKOptional)) {
        dimNum = GetDimNumSmlaL1Norm(cmpResidualKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of cmp_residual_k must be 1, but got %lld",
                   dimNum);
        dataType = GetDataTypeSmlaL1Norm(cmpResidualKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
                   "The data type of cmp_residual_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    if (IsTensorExistSmlaL1Norm(topkLengthOptional)) {
        dimNum = GetDimNumSmlaL1Norm(topkLengthOptional);
        if (strcmp(layoutQ, "BSND") == 0) {
            CHECK_COND(dimNum == 3, ACLNN_ERR_PARAM_INVALID,
                       "For layout_q BSND, the dim num of topk_length must be 3 (B,S1,N2), but got %lld", dimNum);
        } else {
            CHECK_COND(dimNum == 2, ACLNN_ERR_PARAM_INVALID,
                       "For layout_q TND, the dim num of topk_length must be 2 (T1,N2), but got %lld", dimNum);
        }
        dataType = GetDataTypeSmlaL1Norm(topkLengthOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
                   "The data type of topk_length must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    if (IsTensorExistSmlaL1Norm(metadata)) {
        dimNum = GetDimNumSmlaL1Norm(metadata);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of metadata must be 1, but got %lld", dimNum);
        dataType = GetDataTypeSmlaL1Norm(metadata);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
                   "The data type of metadata must be int32, but got %d", static_cast<int32_t>(dataType));
        if (metadata->GetViewShape().GetDim(0) != SMLA_METADATA_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The element num of metadata must be %lld, but got %lld",
                    SMLA_METADATA_SIZE, metadata->GetViewShape().GetDim(0));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    int64_t queryBatchSize = GetQueryBatchSizeSmlaL1Norm(batchSize, cuSeqLensQOptional, seqUsedQOptional, layoutQ);
    int64_t keyBatchSize = GetKeyBatchSizeSmlaL1Norm(batchSize, cuSeqLensKOptional, seqUsedKOptional, layoutK);
    CHECK_COND(queryBatchSize == keyBatchSize, ACLNN_ERR_PARAM_INVALID,
               "The batch_size obtained from query should be the same as that obtained from key, but got %lld and %lld",
               queryBatchSize, keyBatchSize);
    if (strcmp(layoutQ, "TND") == 0 && IsTensorExistSmlaL1Norm(seqUsedQOptional)) {
        int64_t cuSeqLensQBatchSize = cuSeqLensQOptional->GetViewShape().GetDim(0) - 1;
        CHECK_COND(
            cuSeqLensQBatchSize == queryBatchSize, ACLNN_ERR_PARAM_INVALID,
            "When layout_q is TND and seq_used_q is passed, the batch_size from cu_seq_lens_q should equal that from "
            "seq_used_q, but got %lld and %lld",
            cuSeqLensQBatchSize, queryBatchSize);
    }
    if (strcmp(layoutK, "TND") == 0 && IsTensorExistSmlaL1Norm(seqUsedKOptional)) {
        int64_t cuSeqLensKBatchSize = cuSeqLensKOptional->GetViewShape().GetDim(0) - 1;
        CHECK_COND(
            cuSeqLensKBatchSize == keyBatchSize, ACLNN_ERR_PARAM_INVALID,
            "When layout_k is TND and seq_used_k is passed, the batch_size from cu_seq_lens_k should equal that from "
            "seq_used_k, but got %lld and %lld",
            cuSeqLensKBatchSize, keyBatchSize);
    }
    if (IsTensorExistSmlaL1Norm(cmpResidualKOptional)) {
        auto cmpResidualKBatch = cmpResidualKOptional->GetViewShape().GetDim(0);
        CHECK_COND(cmpResidualKBatch == queryBatchSize, ACLNN_ERR_PARAM_INVALID,
                   "The batch_size of cmp_residual_k should match the valid batch size, but got %lld and %lld",
                   cmpResidualKBatch, queryBatchSize);
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
