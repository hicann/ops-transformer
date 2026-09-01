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
 * \file quant_flash_attn_metadata_check.h
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

class QuantFlashAttnMetadataCheck {
public:
    static inline aclnnStatus ParamsCheck(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
                                          const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
                                          int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ,
                                          int64_t numHeadsKv, int64_t headDim, int64_t headDimV, int64_t quantMode,
                                          int64_t maskMode, int64_t winLeft, int64_t winRight, const char *layoutQ,
                                          const char *layoutQDescale, const char *layoutKv, const char *layoutOut,
                                          bool isGradEnabled, const aclTensor *metadata);

private:
    static inline bool IsTensorExist(const aclTensor *tensor);

    static inline aclnnStatus CheckSeqLens(bool isCu, int64_t batchSize, const aclTensor *seqLens);

    // 校验基础属性：batchSize / maxSeqlen / numHeads / headDim / quantMode / layout
    // 文档约束: headDim 仅支持 64/128; quantMode 当前仅支持 1（A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32）
    static inline aclnnStatus CheckBaseAttr(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
                                            int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t headDimV,
                                            int64_t quantMode, const char *layoutQ, const char *layoutQDescale,
                                            const char *layoutKv, const char *layoutOut, bool isGradEnabled);
    // 校验 mask 参数组: maskMode 支持 0/3/4, winLeft/winRight >= -1
    static inline aclnnStatus CheckMask(int64_t maskMode, int64_t winLeft, int64_t winRight);

    // 校验参数存在性: metadata 必须传入; TND 时必须传 cu_seqlens, 非 TND 时不可传 cu_seqlens 且
    // max_seqlen/seqused 至少提供一个
    static inline aclnnStatus CheckExistency(int64_t maxSeqlenQ, int64_t maxSeqlenKv, const char *layoutQ,
                                             const char *layoutKv, const aclTensor *cuSeqlensQOptional,
                                             const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
                                             const aclTensor *sequsedKvOptional, const aclTensor *metadata);

    // 校验一致性: seqLens 形状与 batchSize 匹配; numHeadsQ 必须能被 numHeadsKv 整除
    static inline aclnnStatus CheckConsistency(int64_t batchSize, int64_t numHeadsQ, int64_t numHeadsKv,
                                               const aclTensor *cuSeqlensQOptional,
                                               const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
                                               const aclTensor *sequsedKvOptional);
};

inline aclnnStatus QuantFlashAttnMetadataCheck::ParamsCheck(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ,
    int64_t numHeadsKv, int64_t headDim, int64_t headDimV, int64_t quantMode, int64_t maskMode, int64_t winLeft,
    int64_t winRight, const char *layoutQ, const char *layoutQDescale, const char *layoutKv, const char *layoutOut,
    bool isGradEnabled, const aclTensor *metadata)
{
    auto ret = CheckBaseAttr(batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, headDimV, quantMode,
                             layoutQ, layoutQDescale, layoutKv, layoutOut, isGradEnabled);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckMask(maskMode, winLeft, winRight);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckExistency(maxSeqlenQ, maxSeqlenKv, layoutQ, layoutKv, cuSeqlensQOptional, cuSeqlensKvOptional,
                         sequsedQOptional, sequsedKvOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckConsistency(batchSize, numHeadsQ, numHeadsKv, cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional,
                           sequsedKvOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}

inline bool QuantFlashAttnMetadataCheck::IsTensorExist(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0) &&
           (tensor->GetData() != nullptr);
}

inline aclnnStatus QuantFlashAttnMetadataCheck::CheckBaseAttr(int64_t batchSize, int64_t maxSeqlenQ,
                                                              int64_t maxSeqlenKv, int64_t numHeadsQ,
                                                              int64_t numHeadsKv, int64_t headDim, int64_t headDimV,
                                                              int64_t quantMode, const char *layoutQ,
                                                              const char *layoutQDescale, const char *layoutKv,
                                                              const char *layoutOut, bool isGradEnabled)
{
    CHECK_COND((batchSize == -1 || batchSize > 0), ACLNN_ERR_RUNTIME_ERROR,
               "batchSize must be -1 or greater than 0, but got %ld", batchSize);

    constexpr int64_t A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 = 1;
    constexpr int64_t A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32 = 0;
    static const std::unordered_set<int64_t> quantModeSet = {A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32,
                                                             A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32};
    CHECK_COND(quantModeSet.count(quantMode) > 0, ACLNN_ERR_RUNTIME_ERROR,
               "quantMode only supports 1 (A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) "
               "and 0(A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32) "
               "but got %ld",
               quantMode);

    // 文档约束(单参数校验): max_seqlen_q / max_seqlen_kv 值域为 -1(默认,表示未传) 或 >=0(有效值)
    // 场景相关的存在性约束由 CheckExistency(特性交叉校验)负责: 非TND时与seqused_q/kv至少传1个
    CHECK_COND((maxSeqlenQ == -1 || maxSeqlenQ >= 0), ACLNN_ERR_RUNTIME_ERROR,
               "maxSeqlenQ must be -1 or greater than 0, but got %ld", maxSeqlenQ);
    CHECK_COND((maxSeqlenKv == -1 || maxSeqlenKv >= 0), ACLNN_ERR_RUNTIME_ERROR,
               "maxSeqlenKv must be -1 or greater than 0, but got %ld", maxSeqlenKv);

    CHECK_COND(numHeadsQ > 0, ACLNN_ERR_RUNTIME_ERROR, "numHeadsQ must be greater than 0, but got %ld", numHeadsQ);
    CHECK_COND(numHeadsKv > 0, ACLNN_ERR_RUNTIME_ERROR, "numHeadsKv must be greater than 0, but got %ld", numHeadsKv);

    constexpr int64_t HEAD_DIM_64 = 64;
    constexpr int64_t HEAD_DIM_72 = 72;
    constexpr int64_t HEAD_DIM_128 = 128;
    constexpr int64_t HEAD_DIM_256 = 256;
    static const std::unordered_set<int64_t> headDimSet = {HEAD_DIM_64, HEAD_DIM_72, HEAD_DIM_128, HEAD_DIM_256};
    CHECK_COND(headDimSet.count(headDim) > 0, ACLNN_ERR_RUNTIME_ERROR,
               "headDim only supports %ld, %ld, %ld, %ld, but got %ld", HEAD_DIM_64, HEAD_DIM_72, HEAD_DIM_128,
               HEAD_DIM_256, headDim);

    // 校验 headDimV: 当前仅支持与 headDim 相等 (head_dim_v 接口预留, 暂不支持异值)
    CHECK_COND(headDimV == headDim, ACLNN_ERR_RUNTIME_ERROR,
               "headDimV must be equal to headDim currently, but got headDimV=%ld, headDim=%ld", headDimV, headDim);

    static const std::unordered_set<std::string> layoutQSet = {"BSND", "TND", "BNSD", "NTD"};
    CHECK_COND(layoutQSet.count(layoutQ) > 0, ACLNN_ERR_RUNTIME_ERROR,
               "layoutQ only supports BSND, TND, BNSD, NTD, but got %s", layoutQ);

    static const std::unordered_set<std::string> layoutQDescaleSet = {"BSND", "TND", "BNSD", "N2TGD", "NT"};
    CHECK_COND(layoutQDescaleSet.count(layoutQDescale) > 0, ACLNN_ERR_RUNTIME_ERROR,
               "layoutQDescale only supports BSND, TND, BNSD, N2TGD, NT, but got %s", layoutQDescale);

    static const std::unordered_set<std::string> layoutKvSet = {"BSND", "TND", "BNSD", "PA_BNBD", "PA_BBND", "PA_NZ"};
    CHECK_COND(layoutKvSet.count(layoutKv) > 0, ACLNN_ERR_RUNTIME_ERROR,
               "layoutKv only supports BSND, TND, BNSD, PA_BNBD, PA_BBND, PA_NZ, but got %s", layoutKv);

    static const std::unordered_set<std::string> layoutOutSet = {"BSND", "TND", "BNSD"};
    CHECK_COND(layoutOutSet.count(layoutOut) > 0, ACLNN_ERR_RUNTIME_ERROR,
               "layoutOut only supports BSND, TND, BNSD, but got %s", layoutOut);

    // 一致性校验: MxFP8 (quantMode=1) 场景下 layout_out 仅支持 TND
    CHECK_COND(!(quantMode == A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 && strcmp(layoutOut, "TND") != 0),
               ACLNN_ERR_RUNTIME_ERROR, "When quantMode is %ld (MxFP8), layoutOut only supports TND, but got %s",
               quantMode, layoutOut);

    // 一致性校验: MxFP8 (quantMode=1) 场景下 layout_q 仅支持 TND
    CHECK_COND(!(quantMode == A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 && strcmp(layoutQ, "TND") != 0),
               ACLNN_ERR_RUNTIME_ERROR, "When quantMode is %ld (MxFP8), layoutQ only supports TND, but got %s",
               quantMode, layoutQ);

    return ACLNN_SUCCESS;
}

inline aclnnStatus QuantFlashAttnMetadataCheck::CheckMask(int64_t maskMode, int64_t winLeft, int64_t winRight)
{
    constexpr int64_t NO_MASK = 0;
    constexpr int64_t CAUSAL_MASK = 3;
    constexpr int64_t SLIDING_WINDOW = 4;

    static const std::unordered_set<int64_t> maskSet = {NO_MASK, CAUSAL_MASK, SLIDING_WINDOW};
    CHECK_COND(maskSet.count(maskMode) > 0, ACLNN_ERR_RUNTIME_ERROR,
               "maskMode only supports %ld, %ld, %ld, but got %ld", NO_MASK, CAUSAL_MASK, SLIDING_WINDOW, maskMode);
    CHECK_COND(winLeft >= -1, ACLNN_ERR_RUNTIME_ERROR, "winLeft must be -1 or at least 0, but got %ld", winLeft);
    CHECK_COND(winRight >= -1, ACLNN_ERR_RUNTIME_ERROR, "winRight must be -1 or at least 0, but got %ld", winRight);

    // 非 maskMode = 4 (SLIDING_WINDOW) 场景下 winLeft 和 winRight 必须为 -1
    if (maskMode != SLIDING_WINDOW) {
        CHECK_COND(winLeft == -1 && winRight == -1, ACLNN_ERR_RUNTIME_ERROR,
                   "When maskMode is not 4 (SLIDING_WINDOW), winLeft and winRight must be -1, "
                   "but got winLeft=%ld, winRight=%ld",
                   winLeft, winRight);
    }

    return ACLNN_SUCCESS;
}

inline aclnnStatus QuantFlashAttnMetadataCheck::CheckExistency(
    int64_t maxSeqlenQ, int64_t maxSeqlenKv, const char *layoutQ, const char *layoutKv,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional, const aclTensor *metadata)
{
    CHECK_COND(metadata != nullptr, ACLNN_ERR_RUNTIME_ERROR, "metadata should be provided, but got null");

    if (strcmp(layoutQ, "TND") == 0 || strcmp(layoutQ, "NTD") == 0) {
        // 文档约束: layout_q为TND或NTD时, cu_seqlens_q必须传入, seqused_q与max_seqlen_q可选
        CHECK_COND(IsTensorExist(cuSeqlensQOptional), ACLNN_ERR_RUNTIME_ERROR,
                   "When layoutQ is TND or NTD, cuSeqlensQOptional should be provided, but got null");
    } else {
        // 文档约束: layout_q不为TND/NTD时, cu_seqlens_q不支持传入
        CHECK_COND(!IsTensorExist(cuSeqlensQOptional), ACLNN_ERR_RUNTIME_ERROR,
                   "When layoutQ is not TND or NTD, cuSeqlensQOptional should not be provided, but got non-null");
        // 文档约束: layout_q不为TND/NTD时, seqused_q与max_seqlen_q至少传入其中一个 (-1表示max_seqlen_q未传)
        CHECK_COND(((maxSeqlenQ >= 0) || IsTensorExist(sequsedQOptional)), ACLNN_ERR_RUNTIME_ERROR,
                   "When layoutQ is not TND or NTD, at least one of maxSeqlenQ or sequsedQOptional must be provided");
    }

    if (strcmp(layoutKv, "TND") == 0) {
        // 文档约束: layout_kv为TND时, cu_seqlens_kv必须传入, seqused_kv与max_seqlen_kv可选
        CHECK_COND(IsTensorExist(cuSeqlensKvOptional), ACLNN_ERR_RUNTIME_ERROR,
                   "When layoutKv is TND, cuSeqlensKvOptional should be provided, but got null");
    } else {
        // 文档约束: layout_kv不为TND时, cu_seqlens_kv不支持传入
        CHECK_COND(!IsTensorExist(cuSeqlensKvOptional), ACLNN_ERR_RUNTIME_ERROR,
                   "When layoutKv is not TND, cuSeqlensKvOptional should not be provided, but got non-null");

        // 判断是否为PA场景 (PA_BBND/PA_BNBD/PA_NZ)
        bool isPaLayout =
            (strcmp(layoutKv, "PA_BBND") == 0 || strcmp(layoutKv, "PA_BNBD") == 0 || strcmp(layoutKv, "PA_NZ") == 0);
        if (isPaLayout) {
            // 文档约束: layout_kv为PA场景时, seqused_kv必须传入
            CHECK_COND(IsTensorExist(sequsedKvOptional), ACLNN_ERR_RUNTIME_ERROR,
                       "When layoutKv is PA (PA_BBND/PA_BNBD/PA_NZ), sequsedKv must be provided, but got null");
        } else {
            // 文档约束: layout_kv不为TND且不为PA场景时, seqused_kv与max_seqlen_kv至少传入其中一个
            // (-1表示max_seqlen_kv未传)
            CHECK_COND(((maxSeqlenKv >= 0) || IsTensorExist(sequsedKvOptional)), ACLNN_ERR_RUNTIME_ERROR,
                       "When layoutKv is not TND and not PA, at least one of maxSeqlenKv or "
                       "sequsedKv must be provided");
        }
    }
    return ACLNN_SUCCESS;
}

inline aclnnStatus QuantFlashAttnMetadataCheck::CheckConsistency(
    int64_t batchSize, int64_t numHeadsQ, int64_t numHeadsKv, const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional)
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
               "numHeadsQ must be divisible by numHeadsKv, but got numHeadsQ=%ld, numHeadsKv=%ld", numHeadsQ,
               numHeadsKv);

    return ACLNN_SUCCESS;
}

inline aclnnStatus QuantFlashAttnMetadataCheck::CheckSeqLens(bool isCu, int64_t batchSize, const aclTensor *seqLens)
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
