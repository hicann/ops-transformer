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
 * \file flash_mla_metadata_check.h
 * \brief MLA metadata checker.
 * MLA uses M96/S2=112 on arch35 and M64/S2=128 on arch92.
 * GenMetadata emits the architecture tile from flash_mla/op_kernel/utils/flash_mla_config.h;
 * the main kernel checks the same dimensions in its Cube InitBuffers().
 */

#ifndef FLASH_MLA_METADATA_CHECK_H
#define FLASH_MLA_METADATA_CHECK_H

#include <unordered_set>
#include <string>
#include "../../flash_mla/op_kernel/flash_mla_shared.h"
#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

class FlashMlaMetadataCheck {
public:
    static inline aclnnStatus ParamsCheck(const aclTensor* cuSeqlensQOptional, const aclTensor* cacheSeqlensOptional,
                                          const aclTensor* sequsedQOptional, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
                                          int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDimQk, int64_t headDimV,
                                          int64_t maskMode, int64_t winLeft, int64_t winRight, const char* layoutQ,
                                          const aclTensor* metadata, uint32_t aicCoreNum, uint32_t aivCoreNum);

private:
    static constexpr int64_t NONE_VALUE = -1;

    static inline bool IsTensorExist(const aclTensor* tensor);
    static inline aclnnStatus CheckSeqLens(const char* name, bool isCu, int64_t batchSize, const aclTensor* seqLens);

    static inline aclnnStatus CheckBaseAttr(int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ,
                                            int64_t numHeadsKv, int64_t headDimQk, int64_t headDimV,
                                            const char* layoutQ);

    static inline aclnnStatus CheckMask(int64_t maskMode, int64_t winLeft, int64_t winRight);

    static inline aclnnStatus CheckExistency(int64_t maxSeqlenQ, const char* layoutQ,
                                             const aclTensor* cuSeqlensQOptional, const aclTensor* cacheSeqlensOptional,
                                             const aclTensor* sequsedQOptional, const aclTensor* metadata,
                                             int64_t& batchSize);

    static inline aclnnStatus CheckConsistency(int64_t batchSize, const aclTensor* cuSeqlensQOptional,
                                               const aclTensor* cacheSeqlensOptional,
                                               const aclTensor* sequsedQOptional);

    static inline aclnnStatus CheckMetadataCapacity(const aclTensor* metadata, int64_t batchSize, uint32_t aicCoreNum,
                                                    uint32_t aivCoreNum);
};

inline aclnnStatus FlashMlaMetadataCheck::ParamsCheck(
    const aclTensor* cuSeqlensQOptional, const aclTensor* cacheSeqlensOptional, const aclTensor* sequsedQOptional,
    int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDimQk, int64_t headDimV,
    int64_t maskMode, int64_t winLeft, int64_t winRight, const char* layoutQ, const aclTensor* metadata,
    uint32_t aicCoreNum, uint32_t aivCoreNum)
{
    CHECK_COND(aicCoreNum > 0, ACLNN_ERR_RUNTIME_ERROR, "aicCoreNum should be larger than 0, but got %u", aicCoreNum);
    CHECK_COND(aivCoreNum > 0, ACLNN_ERR_RUNTIME_ERROR, "aivCoreNum should be larger than 0, but got %u", aivCoreNum);

    auto ret = CheckBaseAttr(maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDimQk, headDimV, layoutQ);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckMask(maskMode, winLeft, winRight);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    int64_t batchSize = NONE_VALUE;
    ret = CheckExistency(maxSeqlenQ, layoutQ, cuSeqlensQOptional, cacheSeqlensOptional, sequsedQOptional, metadata,
                         batchSize);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckMetadataCapacity(metadata, batchSize, aicCoreNum, aivCoreNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckConsistency(batchSize, cuSeqlensQOptional, cacheSeqlensOptional, sequsedQOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}

inline aclnnStatus FlashMlaMetadataCheck::CheckBaseAttr(int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ,
                                                        int64_t numHeadsKv, int64_t headDimQk, int64_t headDimV,
                                                        const char* layoutQ)
{
    CHECK_COND(layoutQ != nullptr, ACLNN_ERR_PARAM_NULLPTR, "layoutQ must not be null");
    CHECK_COND((maxSeqlenQ == NONE_VALUE || maxSeqlenQ >= 0), ACLNN_ERR_RUNTIME_ERROR,
               "maxSeqlenQ must be %ld or greater than or equal to 0, but got %ld", NONE_VALUE, maxSeqlenQ);
    CHECK_COND((maxSeqlenKv == NONE_VALUE || maxSeqlenKv >= 0), ACLNN_ERR_RUNTIME_ERROR,
               "maxSeqlenKv must be %ld or greater than or equal to 0, but got %ld", NONE_VALUE, maxSeqlenKv);

    CHECK_COND(numHeadsQ >= 1 && numHeadsQ <= 128, ACLNN_ERR_RUNTIME_ERROR,
               "numHeadsQ must be in [1, 128], but got %ld", numHeadsQ);
    CHECK_COND(numHeadsKv == FlashMlaCommon::MLA_KV_HEADS, ACLNN_ERR_RUNTIME_ERROR,
               "MLA kv head num must be %ld, but got %ld", FlashMlaCommon::MLA_KV_HEADS, numHeadsKv);
    CHECK_COND(headDimQk == FlashMlaCommon::MLA_HEAD_DIM_QK, ACLNN_ERR_RUNTIME_ERROR,
               "MLA head_dim_qk (q/k_cache last dim, nope 512 + rope 64) must be %ld, but got %ld",
               FlashMlaCommon::MLA_HEAD_DIM_QK, headDimQk);
    CHECK_COND(headDimV == FlashMlaCommon::MLA_HEAD_DIM_V, ACLNN_ERR_RUNTIME_ERROR,
               "MLA head_dim_v (value/output width) must be %ld, but got %ld", FlashMlaCommon::MLA_HEAD_DIM_V,
               headDimV);

    static const std::unordered_set<std::string> layoutQSet = {"TND", "BNSD", "BSND"};
    CHECK_COND(layoutQSet.count(layoutQ) > 0, ACLNN_ERR_RUNTIME_ERROR,
               "layoutQ only supports TND, BNSD, BSND, but got %s", layoutQ);

    return ACLNN_SUCCESS;
}

inline aclnnStatus FlashMlaMetadataCheck::CheckMask(int64_t maskMode, int64_t winLeft, int64_t winRight)
{
    constexpr int64_t NO_MASK = 0;
    constexpr int64_t CAUSAL_MASK = 3;

    constexpr int64_t BAND_MASK = 4;
    static const std::unordered_set<int64_t> maskSet = {NO_MASK, CAUSAL_MASK, BAND_MASK};
    CHECK_COND(maskSet.count(maskMode) > 0, ACLNN_ERR_RUNTIME_ERROR, "maskMode only supports 0, 3, 4, but got %ld",
               maskMode);
    CHECK_COND(winLeft >= -1 && winLeft <= INT32_MAX && winRight >= -1 && winRight <= INT32_MAX,
               ACLNN_ERR_PARAM_INVALID, "win_left and win_right must be in [-1, INT32_MAX]");
    CHECK_COND(maskMode == BAND_MASK || (winLeft == -1 && winRight == -1), ACLNN_ERR_PARAM_INVALID,
               "win_left and win_right must be -1 for mask_mode 0 or 3");
    return ACLNN_SUCCESS;
}

inline aclnnStatus FlashMlaMetadataCheck::CheckExistency(int64_t maxSeqlenQ, const char* layoutQ,
                                                         const aclTensor* cuSeqlensQOptional,
                                                         const aclTensor* cacheSeqlensOptional,
                                                         const aclTensor* sequsedQOptional, const aclTensor* metadata,
                                                         int64_t& batchSize)
{
    CHECK_COND(metadata != nullptr, ACLNN_ERR_RUNTIME_ERROR, "metadata should be provided, but got null");
    CHECK_COND(metadata->GetDataType() == op::DataType::DT_INT32, ACLNN_ERR_RUNTIME_ERROR,
               "metadata dtype must be INT32, but got %s", op::ToString(metadata->GetDataType()).GetString());
    CHECK_COND(metadata->GetViewShape().GetDimNum() == 1, ACLNN_ERR_RUNTIME_ERROR,
               "metadata must be 1D tensor, but got %ld dims", metadata->GetViewShape().GetDimNum());
    CHECK_COND(metadata->GetViewShape().GetDim(0) > 0, ACLNN_ERR_RUNTIME_ERROR,
               "metadata shape (max_schedule_size,) must be positive, but got %ld", metadata->GetViewShape().GetDim(0));

    CHECK_COND(IsTensorExist(cuSeqlensQOptional) || IsTensorExist(sequsedQOptional) || maxSeqlenQ != NONE_VALUE,
               ACLNN_ERR_RUNTIME_ERROR,
               "maxSeqlenQ must not be -1 when cuSeqlensQOptional and sequsedQOptional are null or empty tensors");
    CHECK_COND(IsTensorExist(cacheSeqlensOptional), ACLNN_ERR_RUNTIME_ERROR,
               "cacheSeqlens should be provided, but got null or empty tensor");
    CHECK_COND(cacheSeqlensOptional->GetViewShape().GetDimNum() == 1, ACLNN_ERR_RUNTIME_ERROR,
               "cacheSeqlens must be 1D tensor, but got %ld dims", cacheSeqlensOptional->GetViewShape().GetDimNum());
    batchSize = cacheSeqlensOptional->GetViewShape().GetDim(0);
    constexpr int64_t MIN_BATCH = 0;
    constexpr int64_t MAX_BATCH = 65536;
    CHECK_COND((batchSize > MIN_BATCH && batchSize < MAX_BATCH), ACLNN_ERR_RUNTIME_ERROR,
               "batchSize (derived from cacheSeqlens) must be between (%ld, %ld), but got %ld", MIN_BATCH, MAX_BATCH,
               batchSize);

    if (strcmp(layoutQ, "TND") == 0) {
        CHECK_COND(IsTensorExist(cuSeqlensQOptional), ACLNN_ERR_RUNTIME_ERROR,
                   "When layoutQ is TND, cuSeqlensQOptional should be provided, but got null");
    } else {
        CHECK_COND(!IsTensorExist(cuSeqlensQOptional), ACLNN_ERR_RUNTIME_ERROR,
                   "When layoutQ is not TND, cuSeqlensQOptional should not be provided, but got non-null");
    }

    return ACLNN_SUCCESS;
}

inline aclnnStatus FlashMlaMetadataCheck::CheckConsistency(int64_t batchSize, const aclTensor* cuSeqlensQOptional,
                                                           const aclTensor* cacheSeqlensOptional,
                                                           const aclTensor* sequsedQOptional)
{
    CHECK_COND(CheckSeqLens("cuSeqlensQ", true, batchSize, cuSeqlensQOptional) == ACLNN_SUCCESS,
               ACLNN_ERR_RUNTIME_ERROR, "cuSeqlensQOptional is not valid!");
    CHECK_COND(CheckSeqLens("cacheSeqlens", false, batchSize, cacheSeqlensOptional) == ACLNN_SUCCESS,
               ACLNN_ERR_RUNTIME_ERROR, "cacheSeqlensOptional is not valid!");
    CHECK_COND(CheckSeqLens("sequsedQ", false, batchSize, sequsedQOptional) == ACLNN_SUCCESS, ACLNN_ERR_RUNTIME_ERROR,
               "sequsedQOptional is not valid!");
    return ACLNN_SUCCESS;
}

inline aclnnStatus FlashMlaMetadataCheck::CheckMetadataCapacity(const aclTensor* metadata, int64_t batchSize,
                                                                uint32_t aicCoreNum, uint32_t aivCoreNum)
{
    int64_t needWords = FlashMlaCommon::HEAD_METADATA_STRIDE +
                        batchSize * (static_cast<int64_t>(aicCoreNum) * FlashMlaCommon::FA_METADATA_STRIDE +
                                     static_cast<int64_t>(aivCoreNum) * FlashMlaCommon::FD_METADATA_STRIDE);
    int64_t actualWords = metadata->GetViewShape().GetDim(0);
    CHECK_COND(actualWords >= needWords, ACLNN_ERR_RUNTIME_ERROR,
               "metadata buffer too small: %ld words, required at least %ld words "
               "(capacity = ((aic %u + aiv %u) * batch %ld + 1) * 16, kv head num is 1 for MLA)",
               actualWords, needWords, aicCoreNum, aivCoreNum, batchSize);
    return ACLNN_SUCCESS;
}

inline bool FlashMlaMetadataCheck::IsTensorExist(const aclTensor* tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0) &&
           (tensor->GetData() != nullptr);
}

inline aclnnStatus FlashMlaMetadataCheck::CheckSeqLens(const char* name, bool isCu, int64_t batchSize,
                                                       const aclTensor* seqLens)
{
    if (seqLens == nullptr) {
        return ACLNN_SUCCESS;
    }
    // 空 tensor = 缺省 optional（get_valid_tensor 对 None 生成 (0,) int32）；必传性由 CheckExistency 保证
    if (seqLens->GetViewShape().GetDimNum() == 0 || seqLens->GetViewShape().GetDim(0) == 0) {
        return ACLNN_SUCCESS;
    }

    CHECK_COND(seqLens->GetDataType() == op::DataType::DT_INT32, ACLNN_ERR_RUNTIME_ERROR,
               "%s dtype must be INT32, but got %s", name, op::ToString(seqLens->GetDataType()).GetString());
    CHECK_COND(seqLens->GetViewShape().GetDimNum() == 1, ACLNN_ERR_RUNTIME_ERROR,
               "%s must be 1D tensor, but got %ld dims", name, seqLens->GetViewShape().GetDimNum());

    if (isCu) {
        CHECK_COND(seqLens->GetViewShape().GetDim(0) == batchSize + 1, ACLNN_ERR_RUNTIME_ERROR,
                   "%s shape must be (batchSize+1,), but got %ld", name, seqLens->GetViewShape().GetDim(0));
    } else {
        CHECK_COND(seqLens->GetViewShape().GetDim(0) == batchSize, ACLNN_ERR_RUNTIME_ERROR,
                   "%s shape must be (batchSize,), but got %ld", name, seqLens->GetViewShape().GetDim(0));
    }

    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif

#endif // FLASH_MLA_METADATA_CHECK_H
