/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_mla_metadata_check.h
 * \brief QuantFlashMlaWithKvcacheMetadata aclnn层参数校验
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

class QuantFlashMlaMetadataCheck {
public:
    static inline aclnnStatus ParamsCheck(const aclTensor* cacheSeqlens, const aclTensor* cuSeqlensQOptional,
                                          const aclTensor* sequsedQOptional, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
                                          int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDimQk, int64_t headDimV,
                                          int64_t quantMode, int64_t maskMode, const char* layoutQ,
                                          const aclTensor* metadata);

private:
    static inline bool IsTensorExist(const aclTensor* tensor);

    static inline aclnnStatus CheckSeqLens(bool isCu, int64_t batchSize, const aclTensor* seqLens);

    // 校验基础属性: maxSeqlen / numHeads / headDim / quantMode / layout
    // 文档约束: num_heads_kv仅支持1(MLA KV_N=1); head_dim_qk仅支持576; head_dim_v仅支持512;
    // quantMode当前仅支持1(FP8_E4M3)/0(HIFLOAT8)
    static inline aclnnStatus CheckBaseAttr(int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ,
                                            int64_t numHeadsKv, int64_t headDimQk, int64_t headDimV, int64_t quantMode,
                                            const char* layoutQ);

    // 校验 mask 参数组: maskMode 仅支持 0(NO_MASK)/3(CAUSAL)
    static inline aclnnStatus CheckMask(int64_t maskMode);

    // 校验参数存在性: metadata/cache_seqlens必须传入; TND时必须传cu_seqlens_q且seqused_q与max_seqlen_q
    // 至少提供其一; 非TND时不可传cu_seqlens_q
    static inline aclnnStatus CheckExistency(int64_t maxSeqlenQ, const char* layoutQ, const aclTensor* cacheSeqlens,
                                             const aclTensor* cuSeqlensQOptional, const aclTensor* sequsedQOptional,
                                             const aclTensor* metadata);

    // 校验一致性: batchSize取自cache_seqlens shape; cache_seqlens/cu_seqlens_q/seqused_q仅支持int32
    static inline aclnnStatus CheckConsistency(const aclTensor* cacheSeqlens, const aclTensor* cuSeqlensQOptional,
                                               const aclTensor* sequsedQOptional);
};

inline aclnnStatus QuantFlashMlaMetadataCheck::ParamsCheck(
    const aclTensor* cacheSeqlens, const aclTensor* cuSeqlensQOptional, const aclTensor* sequsedQOptional,
    int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDimQk, int64_t headDimV,
    int64_t quantMode, int64_t maskMode, const char* layoutQ, const aclTensor* metadata)
{
    auto ret = CheckBaseAttr(maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDimQk, headDimV, quantMode, layoutQ);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckMask(maskMode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckExistency(maxSeqlenQ, layoutQ, cacheSeqlens, cuSeqlensQOptional, sequsedQOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckConsistency(cacheSeqlens, cuSeqlensQOptional, sequsedQOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}

inline bool QuantFlashMlaMetadataCheck::IsTensorExist(const aclTensor* tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0) &&
           (tensor->GetData() != nullptr);
}

inline aclnnStatus QuantFlashMlaMetadataCheck::CheckBaseAttr(int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ,
                                                             int64_t numHeadsKv, int64_t headDimQk, int64_t headDimV,
                                                             int64_t quantMode, const char* layoutQ)
{
    constexpr int64_t MLA_FP8_E4M3_FULLQUANT = 1;
    constexpr int64_t MLA_HIF8_FULLQUANT = 0;
    static const std::unordered_set<int64_t> quantModeSet = {MLA_FP8_E4M3_FULLQUANT, MLA_HIF8_FULLQUANT};
    CHECK_COND(quantModeSet.count(quantMode) > 0, ACLNN_ERR_RUNTIME_ERROR,
               "quantMode only supports 1 (MLA_FP8_E4M3_FULLQUANT: Q/K/V FP8_E4M3, Q per-token-head, K per-tensor) "
               "and 0 (MLA_HIF8_FULLQUANT) but got %ld",
               quantMode);

    // 文档约束(单参数校验): max_seqlen_q / max_seqlen_kv 值域为 -1(默认,表示未传) 或 >=0(有效值)
    // 场景相关的存在性约束由 CheckExistency(特性交叉校验)负责: TND时与seqused_q至少传1个
    CHECK_COND((maxSeqlenQ == -1 || maxSeqlenQ >= 0), ACLNN_ERR_RUNTIME_ERROR,
               "maxSeqlenQ must be -1 or greater than 0, but got %ld", maxSeqlenQ);
    CHECK_COND((maxSeqlenKv == -1 || maxSeqlenKv >= 0), ACLNN_ERR_RUNTIME_ERROR,
               "maxSeqlenKv must be -1 or greater than 0, but got %ld", maxSeqlenKv);

    CHECK_COND(numHeadsQ > 0, ACLNN_ERR_RUNTIME_ERROR, "numHeadsQ must be greater than 0, but got %ld", numHeadsQ);
    // 文档约束: MLA场景KV共享单一latent头, num_heads_kv仅支持1
    CHECK_COND(numHeadsKv == 1, ACLNN_ERR_RUNTIME_ERROR, "numHeadsKv only supports 1 in MLA scenario, but got %ld",
               numHeadsKv);

    // 文档约束: head_dim_qk仅支持576(512+64rope), head_dim_v仅支持512
    CHECK_COND(headDimQk == 576, ACLNN_ERR_RUNTIME_ERROR, "headDimQk only supports 576, but got %ld", headDimQk);
    CHECK_COND(headDimV == 512, ACLNN_ERR_RUNTIME_ERROR, "headDimV only supports 512, but got %ld", headDimV);

    static const std::unordered_set<std::string> layoutQSet = {"BSND", "TND", "BNSD"};
    CHECK_COND(layoutQSet.count(layoutQ) > 0, ACLNN_ERR_RUNTIME_ERROR,
               "layoutQ only supports BSND, TND, BNSD, but got %s", layoutQ);

    return ACLNN_SUCCESS;
}

inline aclnnStatus QuantFlashMlaMetadataCheck::CheckMask(int64_t maskMode)
{
    constexpr int64_t NO_MASK = 0;
    constexpr int64_t CAUSAL_MASK = 3;

    static const std::unordered_set<int64_t> maskSet = {NO_MASK, CAUSAL_MASK};
    CHECK_COND(maskSet.count(maskMode) > 0, ACLNN_ERR_RUNTIME_ERROR,
               "maskMode only supports %ld (NO_MASK) and %ld (CAUSAL), but got %ld", NO_MASK, CAUSAL_MASK, maskMode);
    return ACLNN_SUCCESS;
}

inline aclnnStatus QuantFlashMlaMetadataCheck::CheckExistency(int64_t maxSeqlenQ, const char* layoutQ,
                                                              const aclTensor* cacheSeqlens,
                                                              const aclTensor* cuSeqlensQOptional,
                                                              const aclTensor* sequsedQOptional,
                                                              const aclTensor* metadata)
{
    CHECK_COND(metadata != nullptr, ACLNN_ERR_RUNTIME_ERROR, "metadata should be provided, but got null");

    // 文档约束: cache_seqlens为必选输入
    CHECK_COND(IsTensorExist(cacheSeqlens), ACLNN_ERR_RUNTIME_ERROR,
               "cacheSeqlens must be provided in MLA scenario, but got null");

    if (strcmp(layoutQ, "TND") == 0) {
        // 文档约束: layout_q为TND时, cu_seqlens_q必须传入
        CHECK_COND(IsTensorExist(cuSeqlensQOptional), ACLNN_ERR_RUNTIME_ERROR,
                   "When layoutQ is TND, cuSeqlensQOptional should be provided, but got null");
        // 文档约束: layout_q为TND时, seqused_q与max_seqlen_q至少传入其中一个 (-1表示max_seqlen_q未传)
        CHECK_COND(((maxSeqlenQ >= 0) || IsTensorExist(sequsedQOptional)), ACLNN_ERR_RUNTIME_ERROR,
                   "When layoutQ is TND, at least one of maxSeqlenQ or sequsedQOptional must be provided");
    } else {
        // 文档约束: layout_q不为TND时, cu_seqlens_q不支持传入
        CHECK_COND(!IsTensorExist(cuSeqlensQOptional), ACLNN_ERR_RUNTIME_ERROR,
                   "When layoutQ is not TND, cuSeqlensQOptional should not be provided, but got non-null");
    }
    return ACLNN_SUCCESS;
}

inline aclnnStatus QuantFlashMlaMetadataCheck::CheckConsistency(const aclTensor* cacheSeqlens,
                                                                const aclTensor* cuSeqlensQOptional,
                                                                const aclTensor* sequsedQOptional)
{
    // 文档约束: cache_seqlens/cu_seqlens_q/seqused_q仅支持int32
    if (cacheSeqlens != nullptr && cacheSeqlens->GetViewShape().GetDimNum() > 0) {
        CHECK_COND(cacheSeqlens->GetDataType() == ge::DT_INT32, ACLNN_ERR_RUNTIME_ERROR,
                   "cacheSeqlens only supports int32, but got %s",
                   op::ToString(cacheSeqlens->GetDataType()).GetString());
    }
    if (cuSeqlensQOptional != nullptr && cuSeqlensQOptional->GetViewShape().GetDimNum() > 0) {
        CHECK_COND(cuSeqlensQOptional->GetDataType() == ge::DT_INT32, ACLNN_ERR_RUNTIME_ERROR,
                   "cuSeqlensQOptional only supports int32, but got %s",
                   op::ToString(cuSeqlensQOptional->GetDataType()).GetString());
    }
    if (sequsedQOptional != nullptr && sequsedQOptional->GetViewShape().GetDimNum() > 0) {
        CHECK_COND(sequsedQOptional->GetDataType() == ge::DT_INT32, ACLNN_ERR_RUNTIME_ERROR,
                   "sequsedQOptional only supports int32, but got %s",
                   op::ToString(sequsedQOptional->GetDataType()).GetString());
    }

    // cache_seqlens为必选输入, batchSize直接取其shape首个维度
    CHECK_COND(IsTensorExist(cacheSeqlens), ACLNN_ERR_RUNTIME_ERROR, "cacheSeqlens must be provided and non-empty");
    CHECK_COND(cacheSeqlens->GetViewShape().GetDimNum() == 1, ACLNN_ERR_RUNTIME_ERROR,
               "cacheSeqlens must be 1D tensor, but got %ld dims", cacheSeqlens->GetViewShape().GetDimNum());
    int64_t batchSize = cacheSeqlens->GetViewShape().GetDim(0);
    CHECK_COND(batchSize > 0, ACLNN_ERR_RUNTIME_ERROR,
               "cacheSeqlens shape must be (batchSize,) with batchSize > 0, but got %ld", batchSize);

    bool isCu = true;
    CHECK_COND(CheckSeqLens(isCu, batchSize, cuSeqlensQOptional) == ACLNN_SUCCESS, ACLNN_ERR_RUNTIME_ERROR,
               "cuSeqlensQOptional is not valid!");
    CHECK_COND(CheckSeqLens(!isCu, batchSize, sequsedQOptional) == ACLNN_SUCCESS, ACLNN_ERR_RUNTIME_ERROR,
               "sequsedQOptional is not valid!");

    return ACLNN_SUCCESS;
}

inline aclnnStatus QuantFlashMlaMetadataCheck::CheckSeqLens(bool isCu, int64_t batchSize, const aclTensor* seqLens)
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
