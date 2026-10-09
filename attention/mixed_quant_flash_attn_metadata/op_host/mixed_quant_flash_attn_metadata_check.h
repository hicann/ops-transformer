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
 * \file mixed_quant_flash_attn_metadata_check.h
 * \brief
 */

#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus ParamsCheck(const aclTensor* cuSeqlensQOptional, const aclTensor* sequsedQOptional,
                               const aclTensor* sequsedKvOptional, int64_t batchSize, int64_t maxSeqlenQ,
                               int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
                               int64_t quantComputeMode, int64_t maskMode, int64_t winLeft, int64_t winRight,
                               const char* layoutQ, const char* layoutKv, const char* layoutAttnOut,
                               const aclTensor* metaData)
{
    // TODO: 魔鬼数字
    if (quantComputeMode != 1 && quantComputeMode != 2 && quantComputeMode != 3) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "quantComputeMode only supports 1, 2, 3, but got %ld", quantComputeMode);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (maskMode != 0 && maskMode != 3 && maskMode != 4) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "maskMode only supports 0, 3, 4, but got %ld", maskMode);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (winLeft < -1) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "winLeft must be >= -1, but got %ld", winLeft);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (winRight < -1) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "winRight must be >= -1, but got %ld", winRight);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (maxSeqlenQ < -1) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "maxSeqlenQ must be >= -1, but got %ld", maxSeqlenQ);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (maxSeqlenKv < -1) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "maxSeqlenKv must be >= -1, but got %ld", maxSeqlenKv);
        return ACLNN_ERR_PARAM_INVALID;
    }

    bool layoutQValid = (strcmp(layoutQ, "BSND") == 0 || strcmp(layoutQ, "TND") == 0 || strcmp(layoutQ, "BNSD") == 0);
    if (!layoutQValid) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "layoutQ only supports BSND, TND, BNSD, but got %s", layoutQ);
        return ACLNN_ERR_PARAM_INVALID;
    }

    bool layoutAttnOutValid =
        (strcmp(layoutAttnOut, "BSND") == 0 || strcmp(layoutAttnOut, "TND") == 0 || strcmp(layoutAttnOut, "BNSD") == 0);
    if (!layoutAttnOutValid) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "layoutAttnOut only supports BSND, TND, BNSD, but got %s", layoutAttnOut);
        return ACLNN_ERR_PARAM_INVALID;
    }

    bool layoutKvValid =
        (strcmp(layoutKv, "PA_BNBD") == 0 || strcmp(layoutKv, "PA_BBND") == 0 || strcmp(layoutKv, "PA_NZ") == 0);
    if (!layoutKvValid) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "layoutKv only supports PA_BNBD, PA_BBND, PA_NZ, but got %s", layoutKv);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (batchSize != 0) {
        if (cuSeqlensQOptional != nullptr) {
            if (cuSeqlensQOptional->GetViewShape().GetDimNum() != 1) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "cuSeqlensQOptional must be 1D tensor, but got %ld dims",
                        cuSeqlensQOptional->GetViewShape().GetDimNum());
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (cuSeqlensQOptional->GetViewShape().GetDim(0) != batchSize + 1) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "cuSeqlensQOptional shape must be (batchSize+1,), but got %ld",
                        cuSeqlensQOptional->GetViewShape().GetDim(0));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        if (sequsedQOptional != nullptr) {
            if (sequsedQOptional->GetViewShape().GetDimNum() != 1) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sequsedQOptional must be 1D tensor, but got %ld dims",
                        sequsedQOptional->GetViewShape().GetDimNum());
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (sequsedQOptional->GetViewShape().GetDim(0) != batchSize) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sequsedQOptional shape must be (batchSize,), but got %ld",
                        sequsedQOptional->GetViewShape().GetDim(0));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        if (sequsedKvOptional != nullptr) {
            if (sequsedKvOptional->GetViewShape().GetDimNum() != 1) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sequsedKvOptional must be 1D tensor, but got %ld dims",
                        sequsedKvOptional->GetViewShape().GetDimNum());
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (sequsedKvOptional->GetViewShape().GetDim(0) != batchSize) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sequsedKvOptional shape must be (batchSize,), but got %ld",
                        sequsedKvOptional->GetViewShape().GetDim(0));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }

    bool isQ_TND = (strcmp(layoutQ, "TND") == 0);
    if (isQ_TND) {
        // layoutQ为TND时，cuSeqlensQOptional、maxSeqlenQ、sequsedQOptional至少传一个
        bool hasQParam = (cuSeqlensQOptional != nullptr) || (maxSeqlenQ != -1) || (sequsedQOptional != nullptr);
        if (!hasQParam) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "When layoutQ is TND, at least one of cuSeqlensQOptional, "
                                             "maxSeqlenQ, sequsedQOptional must be provided");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        // layoutQ不为TND时，不应该传入cuSeqlensQOptional
        if (cuSeqlensQOptional != nullptr) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                    "When layoutQ is not TND, cuSeqlensQOptional should not be provided, but got non-null");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // maxSeqlenQ和sequsedQOptional必须有一个（-1表示不传）
        bool hasQParam = (maxSeqlenQ != -1) || (sequsedQOptional != nullptr);
        if (!hasQParam) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                    "When layoutQ is not TND, at least one of maxSeqlenQ or sequsedQOptional must be provided");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // maxSeqlenKv和sequsedKvOptional必须有一个（-1表示不传）
    bool hasKvParam = (maxSeqlenKv != -1) || (sequsedKvOptional != nullptr);
    if (!hasKvParam) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                "When layoutKv is not TND, at least one of maxSeqlenKv or sequsedKvOptional must be provided");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (numHeadsKv != 0) {
        if (numHeadsQ % numHeadsKv != 0) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                    "numHeadsQ must be divisible by numHeadsKv, but got numHeadsQ=%ld, numHeadsKv=%ld", numHeadsQ,
                    numHeadsKv);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
