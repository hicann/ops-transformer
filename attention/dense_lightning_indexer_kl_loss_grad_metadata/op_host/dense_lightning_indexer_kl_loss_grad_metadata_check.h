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
 * \file dense_lightning_indexer_kl_loss_grad_metadata_check.h
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

inline constexpr int64_t DLI_NO_MASK_MODE = 0;
inline constexpr int64_t DLI_CAUSAL_MASK_MODE = 3;
inline constexpr int64_t DLI_CMP_RATIO_LOWER_BOUND = 1;
inline constexpr int64_t DLI_CMP_RATIO_UPPER_BOUND = 128;
inline constexpr int64_t DLI_NUM_HEADS_Q_LOWER_BOUND_A5 = 1;
inline constexpr int64_t DLI_NUM_HEADS_Q_UPPER_BOUND_A5 = 128;
inline constexpr int64_t DLIKG_METADATA_SIZE = 64;

inline bool IsTensorExistDli(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0);
}

int64_t GetDimNumDli(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return -1;
    }
    return tensor->GetViewShape().GetDimNum();
}

aclDataType GetDataTypeDli(const aclTensor *tensor)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (tensor == nullptr) {
        return dataType;
    }
    aclGetDataType(tensor, &dataType);
    return dataType;
}

int64_t GetQueryBatchSizeDli(int64_t batchSize, const aclTensor *cuSeqlensQOptional, const aclTensor *sequsedQOptional,
                             const char *layoutQOptional)
{
    if (IsTensorExistDli(sequsedQOptional)) {
        return sequsedQOptional->GetViewShape().GetDim(0);
    }
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (IsTensorExistDli(cuSeqlensQOptional)) {
            return cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    return batchSize;
}

int64_t GetKeyBatchSizeDli(int64_t batchSize, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedKOptional,
                           const char *layoutKOptional)
{
    if (IsTensorExistDli(sequsedKOptional)) {
        return sequsedKOptional->GetViewShape().GetDim(0);
    }
    if (strcmp(layoutKOptional, "TND") == 0) {
        if (IsTensorExistDli(cuSeqlensKOptional)) {
            return cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    return batchSize;
}

aclnnStatus CheckSingleParamDli(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK, int64_t numHeadsQ,
                                int64_t numHeadsK, int64_t headDim, const char *layoutQOptional,
                                const char *layoutKOptional, int64_t maskMode, int64_t cmpRatio, uint32_t aicCoreNum,
                                uint32_t aivCoreNum, const std::string &socVersion)
{
    CHECK_COND(numHeadsQ >= DLI_NUM_HEADS_Q_LOWER_BOUND_A5 && numHeadsQ <= DLI_NUM_HEADS_Q_UPPER_BOUND_A5,
               ACLNN_ERR_PARAM_INVALID, "num_heads_q should be [%lld, %lld], but got %lld",
               DLI_NUM_HEADS_Q_LOWER_BOUND_A5, DLI_NUM_HEADS_Q_UPPER_BOUND_A5, numHeadsQ);
    CHECK_COND(numHeadsK == 1, ACLNN_ERR_PARAM_INVALID, "num_heads_kv should only be 1, but got %lld", numHeadsK);
    CHECK_COND(headDim == 128, ACLNN_ERR_PARAM_INVALID, "head_dim should only be 128, but got %lld", headDim);
    CHECK_COND(batchSize >= 0, ACLNN_ERR_PARAM_INVALID, "batch_size should be >= 0, but got %lld", batchSize);
    CHECK_COND(maxSeqlenQ >= 0, ACLNN_ERR_PARAM_INVALID, "max_seqlen_q should be >= 0, but got %lld", maxSeqlenQ);
    CHECK_COND(maxSeqlenK >= 0, ACLNN_ERR_PARAM_INVALID, "max_seqlen_k should be >= 0, but got %lld", maxSeqlenK);
    CHECK_COND((maskMode == DLI_NO_MASK_MODE) || (maskMode == DLI_CAUSAL_MASK_MODE), ACLNN_ERR_PARAM_INVALID,
               "mask_mode should be %lld/%lld, but got %lld", DLI_NO_MASK_MODE, DLI_CAUSAL_MASK_MODE, maskMode);
    CHECK_COND((cmpRatio >= DLI_CMP_RATIO_LOWER_BOUND) && (cmpRatio <= DLI_CMP_RATIO_UPPER_BOUND),
               ACLNN_ERR_PARAM_INVALID, "cmp_ratio should be between [%lld, %lld], but got %lld",
               DLI_CMP_RATIO_LOWER_BOUND, DLI_CMP_RATIO_UPPER_BOUND, cmpRatio);
    CHECK_COND((layoutQOptional != nullptr), ACLNN_ERR_PARAM_INVALID, "layout_q is null!");
    CHECK_COND((strcmp(layoutQOptional, "TND") == 0) || (strcmp(layoutQOptional, "BSND") == 0), ACLNN_ERR_PARAM_INVALID,
               "layout_q must be TND or BSND, but got %s", layoutQOptional);
    CHECK_COND((layoutKOptional != nullptr), ACLNN_ERR_PARAM_INVALID, "layout_k is null!");
    CHECK_COND((strcmp(layoutKOptional, "TND") == 0) || (strcmp(layoutKOptional, "BSND") == 0), ACLNN_ERR_PARAM_INVALID,
               "layout_k must be TND/BSND, but got %s", layoutKOptional);
    CHECK_COND(aicCoreNum > 0, ACLNN_ERR_PARAM_INVALID, "AIC num should be larger than 0, but got %u", aicCoreNum);
    CHECK_COND(aivCoreNum > 0, ACLNN_ERR_PARAM_INVALID, "AIV num should be larger than 0, but got %u", aivCoreNum);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceDli(int64_t maskMode, int64_t cmpRatio, const aclTensor *cuSeqlensQOptional,
                              const aclTensor *cuSeqlensKOptional, const aclTensor *cmpResidualKOptional,
                              const char *layoutQOptional, const char *layoutKOptional, const aclTensor *metadata)
{
    if (strcmp(layoutQOptional, "TND") == 0) {
        CHECK_COND(IsTensorExistDli(cuSeqlensQOptional), ACLNN_ERR_PARAM_INVALID,
                   "For layout_q TND, cu_seqlens_q must be provided!");
    }
    if (strcmp(layoutKOptional, "TND") == 0) {
        CHECK_COND(IsTensorExistDli(cuSeqlensKOptional), ACLNN_ERR_PARAM_INVALID,
                   "For layout_k TND, cu_seqlens_k must be provided!");
    }
    if (cmpRatio != DLI_CMP_RATIO_LOWER_BOUND && maskMode == DLI_CAUSAL_MASK_MODE) {
        CHECK_COND(IsTensorExistDli(cmpResidualKOptional), ACLNN_ERR_PARAM_INVALID,
                   "When cmp_ratio is not 1 and mask_mode is CAUSAL, cmp_residual_k must be provided!");
    }
    CHECK_COND(IsTensorExistDli(metadata), ACLNN_ERR_PARAM_INVALID, "Output metadata is nullptr!");
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistencyDli(int64_t batchSize, const aclTensor *cuSeqlensQOptional,
                                const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
                                const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional,
                                const char *layoutQOptional, const char *layoutKOptional, const aclTensor *metadata)
{
    int64_t dimNum = -1;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;

    if (IsTensorExistDli(cuSeqlensQOptional)) {
        dimNum = GetDimNumDli(cuSeqlensQOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_q must be 1, but got %lld", dimNum);
        dataType = GetDataTypeDli(cuSeqlensQOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
                   "The data type of cu_seqlens_q must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    if (IsTensorExistDli(cuSeqlensKOptional)) {
        dimNum = GetDimNumDli(cuSeqlensKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_k must be 1, but got %lld", dimNum);
        dataType = GetDataTypeDli(cuSeqlensKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
                   "The data type of cu_seqlens_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    if (IsTensorExistDli(sequsedQOptional)) {
        dimNum = GetDimNumDli(sequsedQOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_q must be 1, but got %lld", dimNum);
        dataType = GetDataTypeDli(sequsedQOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
                   "The data type of seqused_q must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    if (IsTensorExistDli(sequsedKOptional)) {
        dimNum = GetDimNumDli(sequsedKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_k must be 1, but got %lld", dimNum);
        dataType = GetDataTypeDli(sequsedKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
                   "The data type of seqused_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    if (IsTensorExistDli(cmpResidualKOptional)) {
        dimNum = GetDimNumDli(cmpResidualKOptional);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of cmp_residual_k must be 1, but got %lld",
                   dimNum);
        dataType = GetDataTypeDli(cmpResidualKOptional);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
                   "The data type of cmp_residual_k must be int32, but got %d", static_cast<int32_t>(dataType));
    }
    if (IsTensorExistDli(metadata)) {
        dimNum = GetDimNumDli(metadata);
        CHECK_COND(dimNum == 1, ACLNN_ERR_PARAM_INVALID, "The dim num of metadata must be 1, but got %lld", dimNum);
        dataType = GetDataTypeDli(metadata);
        CHECK_COND(dataType == aclDataType::ACL_INT32, ACLNN_ERR_PARAM_INVALID,
                   "The data type of metadata must be int32, but got %d", static_cast<int32_t>(dataType));
        if (metadata->GetViewShape().GetDim(0) != DLIKG_METADATA_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The element num of metadata must be %u, but got %lld",
                    DLIKG_METADATA_SIZE, metadata->GetViewShape().GetDim(0));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    int64_t queryBatchSize = GetQueryBatchSizeDli(batchSize, cuSeqlensQOptional, sequsedQOptional, layoutQOptional);
    int64_t keyBatchSize = GetKeyBatchSizeDli(batchSize, cuSeqlensKOptional, sequsedKOptional, layoutKOptional);
    CHECK_COND(queryBatchSize == keyBatchSize, ACLNN_ERR_PARAM_INVALID,
               "The batch_size obtained from query should be the same as that obtained from key, but got %lld and %lld",
               queryBatchSize, keyBatchSize);
    if (strcmp(layoutQOptional, "TND") == 0 && IsTensorExistDli(sequsedQOptional)) {
        int64_t cuSeqlensQBatchSize = cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        CHECK_COND(
            cuSeqlensQBatchSize == queryBatchSize, ACLNN_ERR_PARAM_INVALID,
            "When layout_q is TND and seqused_q is passed, The batch_size obtained from cu_seqlens_q should be the "
            "same as that obtained from seqused_q, but got %lld and %lld",
            cuSeqlensQBatchSize, queryBatchSize);
    }
    if (strcmp(layoutKOptional, "TND") == 0 && IsTensorExistDli(sequsedKOptional)) {
        int64_t cuSeqlensKBatchSize = cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
        CHECK_COND(
            cuSeqlensKBatchSize == keyBatchSize, ACLNN_ERR_PARAM_INVALID,
            "When layout_k is TND and seqused_k is passed, The batch_size obtained from cu_seqlens_k should be the "
            "same as that obtained from seqused_k, but got %lld and %lld",
            cuSeqlensKBatchSize, keyBatchSize);
    }
    if (IsTensorExistDli(cmpResidualKOptional)) {
        auto cmpResidualKBatch = cmpResidualKOptional->GetViewShape().GetDim(0);
        CHECK_COND(cmpResidualKBatch == queryBatchSize, ACLNN_ERR_PARAM_INVALID,
                   "The batch_size of cmp_residual_k should match the valid batch size, but got %lld and %lld",
                   cmpResidualKBatch, queryBatchSize);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckDliA5(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional,
                             const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
                             const aclTensor *cmpResidualKOptional, int64_t batchSize, int64_t maxSeqlenQ,
                             int64_t maxSeqlenK, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim,
                             char *layoutQOptional, char *layoutKOptional, int64_t maskMode, int64_t cmpRatio,
                             const aclTensor *metadata, uint32_t aicCoreNum, uint32_t aivCoreNum,
                             const std::string &socVersion)
{
    auto ret = CheckSingleParamDli(batchSize, maxSeqlenQ, maxSeqlenK, numHeadsQ, numHeadsK, headDim, layoutQOptional,
                                   layoutKOptional, maskMode, cmpRatio, aicCoreNum, aivCoreNum, socVersion);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckExistenceDli(maskMode, cmpRatio, cuSeqlensQOptional, cuSeqlensKOptional, cmpResidualKOptional,
                            layoutQOptional, layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckConsistencyDli(batchSize, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                              cmpResidualKOptional, layoutQOptional, layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckDli(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional,
                           const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
                           const aclTensor *cmpResidualKOptional, int64_t batchSize, int64_t maxSeqlenQ,
                           int64_t maxSeqlenK, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim,
                           char *layoutQOptional, char *layoutKOptional, int64_t maskMode, int64_t cmpRatio,
                           const aclTensor *metadata, uint32_t aicCoreNum, uint32_t aivCoreNum,
                           const std::string &socVersion)
{
    const std::string ascend950 = "Ascend950";
    if (socVersion.find(ascend950) == std::string::npos) {
        CHECK_RET(metadata != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        return ACLNN_SUCCESS;
    }
    return ParamsCheckDliA5(cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                            cmpResidualKOptional, batchSize, maxSeqlenQ, maxSeqlenK, numHeadsQ, numHeadsK, headDim,
                            layoutQOptional, layoutKOptional, maskMode, cmpRatio, metadata, aicCoreNum, aivCoreNum,
                            socVersion);
}

} // namespace

#ifdef __cplusplus
}
#endif
