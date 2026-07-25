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
 * \file flash_attn_metadata_check.h
 * \brief
 */

#include <unordered_set>
#include <string>
#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

class FlashAttnMetadataCheck {
public:
    static inline aclnnStatus ParamsCheck(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
                                          const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
                                          int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ,
                                          int64_t numHeadsKv, int64_t headDim, int64_t maskMode, int64_t winLeft,
                                          int64_t winRight, const char *layoutQ, const char *layoutKv,
                                          const char *layoutOut, const aclTensor *metadata);

private:
    static inline bool IsTensorExist(const aclTensor *tensor);

    static inline aclnnStatus CheckSeqLens(bool isCu, int64_t batchSize, const aclTensor *seqLens);

    static inline aclnnStatus CheckBaseAttr(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
                                            int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
                                            const char *layoutQ, const char *layoutKv, const char *layoutOut);

    static inline aclnnStatus CheckMask(int64_t maskMode, int64_t winLeft, int64_t winRight);

    static inline aclnnStatus CheckExistency(int64_t maxSeqlenQ, int64_t maxSeqlenKv,
                                             const char *layoutQ, const char *layoutKv,
                                             const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
                                             const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
                                             const aclTensor *metadata);

    static inline aclnnStatus CheckConsistency(int64_t batchSize, int64_t numHeadsQ, int64_t numHeadsKv,
                                               const aclTensor *cuSeqlensQOptional,
                                               const aclTensor *cuSeqlensKvOptional,
                                               const aclTensor *sequsedQOptional,
                                               const aclTensor *sequsedKvOptional);
};

inline aclnnStatus
FlashAttnMetadataCheck::ParamsCheck(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
                                    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
                                    int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ,
                                    int64_t numHeadsKv, int64_t headDim, int64_t maskMode, int64_t winLeft,
                                    int64_t winRight, const char *layoutQ, const char *layoutKv,
                                    const char *layoutOut, const aclTensor *metadata)
{
    auto ret = CheckBaseAttr(batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim,
                             layoutQ, layoutKv, layoutOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckMask(maskMode, winLeft, winRight);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckExistency(maxSeqlenQ, maxSeqlenKv, layoutQ, layoutKv, cuSeqlensQOptional, cuSeqlensKvOptional,
                         sequsedQOptional, sequsedKvOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckConsistency(batchSize, numHeadsQ, numHeadsKv, cuSeqlensQOptional, cuSeqlensKvOptional,
                           sequsedQOptional, sequsedKvOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}

inline bool FlashAttnMetadataCheck::IsTensorExist(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0) &&
        (tensor->GetData() != nullptr);
}

inline aclnnStatus
FlashAttnMetadataCheck::CheckBaseAttr(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
                                      int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
                                      const char *layoutQ, const char *layoutKv, const char *layoutOut)
{
    CHECK_COND((batchSize == -1 || batchSize > 0), ACLNN_ERR_RUNTIME_ERROR,
        "batchSize must be -1 or greater than 0, but got %ld", batchSize);
    CHECK_COND((maxSeqlenQ == -1 || maxSeqlenQ > 0), ACLNN_ERR_RUNTIME_ERROR,
        "maxSeqlenQ must be -1 or greater than 0, but got %ld", maxSeqlenQ);
    CHECK_COND((maxSeqlenKv == -1 || maxSeqlenKv > 0), ACLNN_ERR_RUNTIME_ERROR,
        "maxSeqlenKv must be -1 or greater than 0, but got %ld", maxSeqlenKv);

    CHECK_COND(numHeadsQ > 0, ACLNN_ERR_RUNTIME_ERROR, "numHeadsQ must be greater than 0, but got %ld", numHeadsQ);
    CHECK_COND(numHeadsKv > 0, ACLNN_ERR_RUNTIME_ERROR, "numHeadsKv must be greater than 0, but got %ld", maxSeqlenKv);

    constexpr int64_t HEAD_DIM_64 = 64;
    constexpr int64_t HEAD_DIM_128 = 128;
    constexpr int64_t HEAD_DIM_256 = 256;
    static const std::unordered_set<int64_t> headDimSet = { HEAD_DIM_64, HEAD_DIM_128, HEAD_DIM_256 };
    CHECK_COND(headDimSet.count(headDim) > 0, ACLNN_ERR_RUNTIME_ERROR,
        "headDim only supports %ld, %ld, %ld, but got %ld", HEAD_DIM_64, HEAD_DIM_128, HEAD_DIM_256, headDim);

    static const std::unordered_set<std::string> layoutQSet = { "BSND", "TND", "BNSD" };
    CHECK_COND(layoutQSet.count(layoutQ) > 0, ACLNN_ERR_RUNTIME_ERROR,
        "layoutQ only supports BSND, TND, BNSD, but got %s", layoutQ);

    static const std::unordered_set<std::string> layoutKvSet = { "BSND", "TND", "BNSD", "PA_BNBD", "PA_BBND", "PA_NZ" };
    CHECK_COND(layoutKvSet.count(layoutKv) > 0, ACLNN_ERR_RUNTIME_ERROR,
        "layoutKv only supports BSND, TND, BNSD, PA_BNBD, PA_BBND, PA_NZ, but got %s", layoutKv);

    static const std::unordered_set<std::string> layoutOutSet = { "BSND", "TND", "BNSD" };
    CHECK_COND(layoutOutSet.count(layoutOut) > 0, ACLNN_ERR_RUNTIME_ERROR,
        "layoutOut only supports BSND, TND, BNSD, but got %s", layoutOut);

    return ACLNN_SUCCESS;
}

inline aclnnStatus
FlashAttnMetadataCheck::CheckMask(int64_t maskMode, int64_t winLeft, int64_t winRight)
{
    constexpr int64_t NO_MASK = 0;
    constexpr int64_t CAUSAL_MASK = 3;
    constexpr int64_t WINDOW_MASK = 4;

    static const std::unordered_set<int64_t> maskSet = { NO_MASK, CAUSAL_MASK, WINDOW_MASK };
    CHECK_COND(maskSet.count(maskMode) > 0, ACLNN_ERR_RUNTIME_ERROR,
        "maskMode only supports %ld, %ld, %ld, but got %ld", NO_MASK, CAUSAL_MASK, WINDOW_MASK, maskMode);
    CHECK_COND(winLeft >= -1, ACLNN_ERR_RUNTIME_ERROR, "winLeft must be -1 or at least 0, but got %ld", winLeft);
    CHECK_COND(winRight >= -1, ACLNN_ERR_RUNTIME_ERROR, "winRight must be -1 or at least 0, but got %ld", winRight);

    return ACLNN_SUCCESS;
}

inline aclnnStatus
FlashAttnMetadataCheck::CheckExistency(int64_t maxSeqlenQ, int64_t maxSeqlenKv,
                                       const char *layoutQ, const char *layoutKv,
                                       const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
                                       const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
                                       const aclTensor *metadata)
{
    CHECK_COND(metadata != nullptr, ACLNN_ERR_RUNTIME_ERROR, "metadata should be provided, but got null");

    if (strcmp(layoutQ, "TND") == 0) {
        // layoutQ为TND时，必须传入cuSeqlensQOptional
        CHECK_COND(IsTensorExist(cuSeqlensQOptional), ACLNN_ERR_RUNTIME_ERROR,
            "When layoutQ is TND, cuSeqlensQOptional should be provided, but got null");
    } else {
        // layoutQ不为TND时，不可以传入cuSeqlensQOptional
        CHECK_COND(!IsTensorExist(cuSeqlensQOptional), ACLNN_ERR_RUNTIME_ERROR,
            "When layoutQ is not TND, cuSeqlensQOptional should not be provided, but got non-null");

        // maxSeqlenQ和sequsedQOptional必须有一个（-1表示不传）
        CHECK_COND(((maxSeqlenQ >= 0) || IsTensorExist(sequsedQOptional)), ACLNN_ERR_RUNTIME_ERROR,
            "When layoutQ is not TND, at least one of maxSeqlenQ or sequsedQOptional must be provided");
    }

    if (strcmp(layoutKv, "TND") == 0) {
        // layoutQ为TND时，必须传入cuSeqlensKvOptional
        CHECK_COND(IsTensorExist(cuSeqlensKvOptional), ACLNN_ERR_RUNTIME_ERROR,
            "When layoutKv is TND, cuSeqlensKvOptional should be provided, but got null");
    } else {
        // layoutKv不为TND时，不可以传入cuSeqlensKvOptional
        CHECK_COND(!IsTensorExist(cuSeqlensKvOptional), ACLNN_ERR_RUNTIME_ERROR,
            "When layoutKv is not TND, cuSeqlensKvOptional should not be provided, but got non-null");
        // maxSeqlenKv和sequsedKvOptional必须有一个（-1表示不传）
        CHECK_COND(((maxSeqlenKv >= 0) || IsTensorExist(sequsedKvOptional)), ACLNN_ERR_RUNTIME_ERROR,
            "When layoutKv is not TND, at least one of maxSeqlenKv or sequsedKvOptional must be provided");
    }
    return ACLNN_SUCCESS;
}

inline aclnnStatus
FlashAttnMetadataCheck::CheckConsistency(int64_t batchSize, int64_t numHeadsQ, int64_t numHeadsKv,
                                         const aclTensor *cuSeqlensQOptional,
                                         const aclTensor *cuSeqlensKvOptional,
                                         const aclTensor *sequsedQOptional,
                                         const aclTensor *sequsedKvOptional)
{
    if (batchSize <= 0) {
        return ACLNN_SUCCESS;
    }

    bool isCu = true;
    CHECK_COND(CheckSeqLens(isCu, batchSize, cuSeqlensQOptional) == ACLNN_SUCCESS, ACLNN_ERR_RUNTIME_ERROR,
        "cuSeqlensQOptional is not valid!");
    CHECK_COND(CheckSeqLens(isCu, batchSize, cuSeqlensKvOptional) == ACLNN_SUCCESS, ACLNN_ERR_RUNTIME_ERROR,
        "cuSeqlensKvOptional is not valid!");
    CHECK_COND(CheckSeqLens(!isCu, batchSize, sequsedQOptional) == ACLNN_SUCCESS, ACLNN_ERR_RUNTIME_ERROR,
        "sequsedQOptional is not valid!");
    CHECK_COND(CheckSeqLens(!isCu, batchSize, sequsedKvOptional) == ACLNN_SUCCESS, ACLNN_ERR_RUNTIME_ERROR,
        "sequsedKvOptional is not valid!");

    CHECK_COND((numHeadsQ % numHeadsKv == 0), ACLNN_ERR_RUNTIME_ERROR,
        "numHeadsQ must be divisible by numHeadsKv, but got numHeadsQ=%ld, numHeadsKv=%ld", numHeadsQ, numHeadsKv);

    return ACLNN_SUCCESS;
}

inline aclnnStatus
FlashAttnMetadataCheck::CheckSeqLens(bool isCu, int64_t batchSize, const aclTensor *seqLens)
{
    if (seqLens == nullptr) {
        return ACLNN_SUCCESS;
    }

    CHECK_COND(seqLens->GetViewShape().GetDimNum() == 1, ACLNN_ERR_RUNTIME_ERROR,
        "seqLens must be 1D tensor, but got %ld dims", seqLens->GetViewShape().GetDimNum());

    if (isCu) {
        CHECK_COND(seqLens->GetViewShape().GetDim(0) == batchSize + 1, ACLNN_ERR_RUNTIME_ERROR,
            "cuSeqLens shape must be (batchSize+1,), but got %ld", seqLens->GetViewShape().GetDim(0));
    } else {
        CHECK_COND(seqLens->GetViewShape().GetDim(0) == batchSize, ACLNN_ERR_RUNTIME_ERROR,
            "seqLens shape must be (batchSize,), but got %ld", seqLens->GetViewShape().GetDim(0));
    }

    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif