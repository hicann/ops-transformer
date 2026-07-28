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
 * \file quant_sparse_flash_mla_metadata_check.h
 * \brief
 */

#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include "../../quant_sparse_flash_mla/op_kernel/quant_sparse_flash_mla_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

enum class SparseModeQsmla : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    SPARSE_BUTT,
};

inline constexpr int64_t QSMLA_QUANT_MODE_LOWER_BOUND = 1;
inline constexpr int64_t QSMLA_QUANT_MODE_UPPER_BOUND = 2;
inline constexpr int64_t QSMLA_CMP_RATIO_LOWER_BOUND = 1;
inline constexpr int64_t QSMLA_CMP_RATIO_UPPER_BOUND = 128;
inline constexpr int64_t QSMLA_NUM_HEADS_Q_LOWER_BOUND = 1;
inline constexpr int64_t QSMLA_NUM_HEADS_Q_UPPER_BOUND = 128;

inline bool IsTensorExistQsmla(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0);
}

int64_t GetDimNumQsmla(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return -1;
    }
    return tensor->GetViewShape().GetDimNum();
}

aclDataType GetDataTypeQsmla(const aclTensor *tensor)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (tensor == nullptr) {
        return dataType;
    }
    aclGetDataType(tensor, &dataType);
    return dataType;
}

aclnnStatus CheckSingleParamQsmla(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenOriKv, int64_t maxSeqlenCmpKv,
                                  int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t quantMode,
                                  int64_t oriTopk, int64_t cmpTopk, int64_t cmpRatio, int64_t oriMaskMode,
                                  int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight,
                                  const char *layoutQOptional, const char *layoutKvOptional, bool hasOriKv,
                                  bool hasCmpKv, uint32_t aicCoreNum, uint32_t aivCoreNum, const char *socVersion)
{
    // batch_size >= 0
    if (batchSize < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch_size should be >= 0, but got %lld", batchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q >= 0
    if (maxSeqlenQ < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_q should be >= 0, but got %lld", maxSeqlenQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_ori_kv >= 0
    if (maxSeqlenOriKv < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_ori_kv should be >= 0, but got %lld", maxSeqlenOriKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_cmp_kv >= 0
    if (maxSeqlenCmpKv < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_cmp_kv should be >= 0, but got %lld", maxSeqlenCmpKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_q [1, 128]
    if (numHeadsQ < QSMLA_NUM_HEADS_Q_LOWER_BOUND || numHeadsQ > QSMLA_NUM_HEADS_Q_UPPER_BOUND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_q should be [%lld, %lld], but got %lld",
                QSMLA_NUM_HEADS_Q_LOWER_BOUND, QSMLA_NUM_HEADS_Q_UPPER_BOUND, numHeadsQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_kv: 1
    if (numHeadsKv != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_kv should only be 1, but got %lld", numHeadsKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // head_dim: 512
    if (headDim != 512) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "head_dim should only be 512, but got %lld", headDim);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // quant_mode
    if (quantMode < QSMLA_QUANT_MODE_LOWER_BOUND || quantMode > QSMLA_QUANT_MODE_UPPER_BOUND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quant_mode should be [%lld, %lld], but got %lld",
                QSMLA_QUANT_MODE_LOWER_BOUND, QSMLA_QUANT_MODE_UPPER_BOUND, quantMode);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hasOriKv) {
        // ori_topk >= 0
        if (oriTopk < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_topk should be >= 0, but got %lld", oriTopk);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // ori_mask_mode: 0, 3, or 4
        if (oriMaskMode != static_cast<int64_t>(SparseModeQsmla::DEFAULT_MASK) &&
            oriMaskMode != static_cast<int64_t>(SparseModeQsmla::RIGHT_DOWN_CAUSAL) &&
            oriMaskMode != static_cast<int64_t>(SparseModeQsmla::BAND)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_mask_mode should be 0, 3 or 4, but got %lld",
                    oriMaskMode);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // ori_win_left >= -1
        if (oriWinLeft < -1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_win_left should be >= -1, but got %lld",
                    oriWinLeft);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // ori_win_right >= -1
        if (oriWinRight < -1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_win_right should be >= -1, but got %lld",
                    oriWinRight);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (hasCmpKv) {
        // cmp_topk >= 0
        if (cmpTopk < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true, cmp_topk should be >= 0, but got %lld", cmpTopk);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // cmp_mask_mode: 0 or 3
        if (cmpMaskMode != static_cast<int64_t>(SparseModeQsmla::DEFAULT_MASK) &&
            cmpMaskMode != static_cast<int64_t>(SparseModeQsmla::RIGHT_DOWN_CAUSAL)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true, cmp_mask_mode should be 0 or 3, but got %lld",
                    cmpMaskMode);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // cmp_ratio: 1~128
        if (cmpRatio < QSMLA_CMP_RATIO_LOWER_BOUND || cmpRatio > QSMLA_CMP_RATIO_UPPER_BOUND) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When has_cmp_kv is true, cmp_ratio should be in [%lld, %lld], "
                    "but got %lld",
                    QSMLA_CMP_RATIO_LOWER_BOUND, QSMLA_CMP_RATIO_UPPER_BOUND, cmpRatio);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (layoutQOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_q is null!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutKvOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_kv is null!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q: BSND or TND
    if ((strcmp(layoutQOptional, "TND") != 0) && (strcmp(layoutQOptional, "BSND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_q must be TND or BSND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_kv: BSND, TND, or PA_BBND
    if ((strcmp(layoutKvOptional, "BSND") != 0) && (strcmp(layoutKvOptional, "TND") != 0) &&
        (strcmp(layoutKvOptional, "PA_BBND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_kv must be TND, BSND or PA_BBND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 核数校验
    if (aicCoreNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AIC num should be larger than 0, but got %u", aicCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aicCoreNum > optiling::AIC_CORE_MAX_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The maximum supported AIC num is %u, but got %u", optiling::AIC_CORE_MAX_NUM,
                aicCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aivCoreNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AIV num should be larger than 0, but got %u", aivCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aivCoreNum > optiling::AIV_CORE_MAX_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The maximum supported AIV num is %u, but got %u", optiling::AIV_CORE_MAX_NUM,
                aivCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验切g模板核数
    if (numHeadsQ == 128) {
        if (aicCoreNum == 1 || aivCoreNum == 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When num_heads_q is 128, AIC num and AIV num should not be 1, "
                    "but got %u and %u",
                    aicCoreNum, aivCoreNum);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceQsmla(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
                                const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedOriKvOptional,
                                const aclTensor *sequsedCmpKvOptional, const aclTensor *cmpResidualKvOptional,
                                const aclTensor *oriTopkLengthOptional, const aclTensor *cmpTopkLengthOptional,
                                int64_t oriTopk, int64_t cmpTopk, int64_t cmpRatio, int64_t oriMaskMode,
                                int64_t cmpMaskMode, bool hasOriKv, bool hasCmpKv, const char *layoutQOptional,
                                const char *layoutKvOptional, const aclTensor *metadata)
{
    // cu_seqlens_q 存在性校验
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (!IsTensorExistQsmla(cuSeqlensQOptional)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When layout_q is TND, cu_seqlens_q must be provided!");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (hasOriKv) {
        // cu_seqlens_ori_kv 存在性校验
        if (strcmp(layoutKvOptional, "TND") == 0) {
            if (!IsTensorExistQsmla(cuSeqlensOriKvOptional)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true and layout_kv is TND, "
                                                 "cu_seqlens_ori_kv must be provided!");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // seqused_ori_kv 存在性校验
        if ((oriMaskMode != 0 || oriTopk == 0) && strcmp(layoutKvOptional, "PA_BBND") == 0) {
            if (!IsTensorExistQsmla(sequsedOriKvOptional)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_mask_mode != 0 or ori_topk == 0, and "
                                                 "layout_kv is PA_BBND, seqused_ori_kv must be provided!");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // ori_topk_length 存在性校验
        if (oriTopk != 0 && oriMaskMode == static_cast<int64_t>(SparseModeQsmla::DEFAULT_MASK)) {
            if (!IsTensorExistQsmla(oriTopkLengthOptional)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_ori_kv is true, ori_topk is not 0 and ori_mask_mode is 0, "
                                                 "ori_topk_length must be provided!");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    if (hasCmpKv) {
        // cu_seqlens_cmp_kv 存在性校验
        if (strcmp(layoutKvOptional, "TND") == 0) {
            if (!IsTensorExistQsmla(cuSeqlensCmpKvOptional)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true and layout_kv is TND, "
                                                 "cu_seqlens_cmp_kv must be provided!");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // seqused_cmp_kv 存在性校验
        if ((cmpMaskMode != 0 || cmpTopk == 0) && strcmp(layoutKvOptional, "PA_BBND") == 0) {
            if (!IsTensorExistQsmla(sequsedCmpKvOptional)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true, cmp_mask_mode != 0 or cmp_topk == 0, and "
                                                 "layout_kv is PA_BBND, seqused_cmp_kv must be provided!");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // cmp_residual_kv 存在性校验
        if (cmpRatio != 1 && cmpMaskMode == static_cast<int64_t>(SparseModeQsmla::RIGHT_DOWN_CAUSAL)) {
            if (!IsTensorExistQsmla(cmpResidualKvOptional)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true, cmp_ratio is not 1 and cmp_mask_mode is 3, "
                                                 "cmp_residual_kv must be provided!");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // cmp_topk_length 存在性校验
        if (cmpTopk != 0 && cmpMaskMode == static_cast<int64_t>(SparseModeQsmla::DEFAULT_MASK)) {
            if (!IsTensorExistQsmla(cmpTopkLengthOptional)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When has_cmp_kv is true, cmp_topk is not 0 and cmp_mask_mode is 0, "
                                                 "cmp_topk_length must be provided!");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    // metadata 存在性校验
    if (!IsTensorExistQsmla(metadata)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output metadata is nullptr!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

int64_t GetQueryBatchSizeQsmla(const aclTensor *sequsedQOptional, const aclTensor *cuSeqlensQOptional,
                               const char *layoutQOptional, int64_t batchSize)
{
    // 1. 如果 sequsedQOptional 传了，使用 sequsedQOptional 获取 BatchSize
    if (IsTensorExistQsmla(sequsedQOptional)) {
        return sequsedQOptional->GetViewShape().GetDim(0);
    }
    // 2. sequsedQOptional 没传，判断 Layout
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (IsTensorExistQsmla(
                cuSeqlensQOptional)) { // 前序校验已保证layout_q = TND时，cu_seqlens_q必须传入，此通路必达
            return cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    // 3. 使用 batchSize
    return batchSize;
}

int64_t GetOriKvBatchSizeQsmla(const aclTensor *sequsedOriKvOptional, const aclTensor *cuSeqlensOriKvOptional,
                               const char *layoutKvOptional, int64_t batchSize)
{
    // 1. 如果 sequsedOriKvOptional 传了，使用 sequsedOriKvOptional 获取 BatchSize
    if (IsTensorExistQsmla(sequsedOriKvOptional)) {
        return sequsedOriKvOptional->GetViewShape().GetDim(0);
    }
    // 2. sequsedOriKvOptional 没传，判断 Layout
    if (strcmp(layoutKvOptional, "TND") == 0) {
        if (IsTensorExistQsmla(cuSeqlensOriKvOptional)) { // 同上，此通路必达
            return cuSeqlensOriKvOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    // 3. 使用 batchSize
    return batchSize;
}

int64_t GetCmpKvBatchSizeQsmla(const aclTensor *sequsedCmpKvOptional, const aclTensor *cuSeqlensCmpKvOptional,
                               const char *layoutKvOptional, int64_t batchSize)
{
    // 1. 如果 sequsedCmpKvOptional 传了，使用 sequsedCmpKvOptional 获取 BatchSize
    if (IsTensorExistQsmla(sequsedCmpKvOptional)) {
        return sequsedCmpKvOptional->GetViewShape().GetDim(0);
    }
    // 2. sequsedCmpKvOptional 没传，判断 Layout
    if (strcmp(layoutKvOptional, "TND") == 0) {
        // 如果是 TND，尝试使用 cuSeqlensCmpKvOptional 获取 BatchSize
        if (IsTensorExistQsmla(cuSeqlensCmpKvOptional)) {
            return cuSeqlensCmpKvOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    // 3. 如果不是 TND，或者 cuSeqlensCmpKvOptional 为空，使用 batchSize
    return batchSize;
}

aclnnStatus CheckConsistencyQsmla(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
                                  const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional,
                                  const aclTensor *sequsedOriKvOptional, const aclTensor *sequsedCmpKvOptional,
                                  const aclTensor *cmpResidualKvOptional, const aclTensor *oriTopkLengthOptional,
                                  const aclTensor *cmpTopkLengthOptional, int64_t batchSize,
                                  const char *layoutQOptional, const char *layoutKvOptional, bool hasOriKv,
                                  bool hasCmpKv, const aclTensor *metadata)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    int64_t dimNum = -1;
    // 校验 cu_seqlens_q
    if (IsTensorExistQsmla(cuSeqlensQOptional)) {
        // 校验 cu_seqlens_q 维度
        dimNum = GetDimNumQsmla(cuSeqlensQOptional);
        if (dimNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_q must be 1, but got %lld", dimNum);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 cu_seqlens_q 数据类型
        dataType = GetDataTypeQsmla(cuSeqlensQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_q
    if (IsTensorExistQsmla(sequsedQOptional)) {
        // 校验 seqused_q 维度
        dimNum = GetDimNumQsmla(sequsedQOptional);
        if (dimNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_q must be 1, but got %lld", dimNum);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 seqused_q 数据类型
        dataType = GetDataTypeQsmla(sequsedQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of seqused_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // ori_kv部分
    if (hasOriKv) {
        // 校验 cu_seqlens_ori_kv
        if (IsTensorExistQsmla(cuSeqlensOriKvOptional)) {
            // 校验 cu_seqlens_ori_kv 维度
            dimNum = GetDimNumQsmla(cuSeqlensOriKvOptional);
            if (dimNum != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_ori_kv must be 1, but got %lld", dimNum);
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cu_seqlens_ori_kv 数据类型
            dataType = GetDataTypeQsmla(cuSeqlensOriKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_ori_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 seqused_ori_kv
        if (IsTensorExistQsmla(sequsedOriKvOptional)) {
            // 校验 seqused_ori_kv 维度
            dimNum = GetDimNumQsmla(sequsedOriKvOptional);
            if (dimNum != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_ori_kv must be 1, but got %lld", dimNum);
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 seqused_ori_kv 数据类型
            dataType = GetDataTypeQsmla(sequsedOriKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of seqused_ori_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 ori_topk_length
        if (IsTensorExistQsmla(oriTopkLengthOptional)) {
            // 校验 ori_topk_length 维度
            dimNum = GetDimNumQsmla(oriTopkLengthOptional);
            if (strcmp(layoutQOptional, "TND") == 0) {
                if (dimNum != 2) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                            "When layout_q is TND, the dim num of ori_topk_length must be 2, "
                            "but got %lld",
                            dimNum);
                    return ACLNN_ERR_PARAM_INVALID;
                }
            } else if (strcmp(layoutQOptional, "BSND") == 0) {
                if (dimNum != 3) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                            "When layout_q is BSND, the dim num of ori_topk_length must be 3, "
                            "but got %lld",
                            dimNum);
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
            // 校验 ori_topk_length 数据类型
            dataType = GetDataTypeQsmla(oriTopkLengthOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of ori_topk_length must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    // cmp_kv部分
    if (hasCmpKv) {
        // 校验 cu_seqlens_cmp_kv
        if (IsTensorExistQsmla(cuSeqlensCmpKvOptional)) {
            // 校验 cu_seqlens_cmp_kv 维度
            dimNum = GetDimNumQsmla(cuSeqlensCmpKvOptional);
            if (dimNum != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_cmp_kv must be 1, but got %lld", dimNum);
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cu_seqlens_cmp_kv 数据类型
            dataType = GetDataTypeQsmla(cuSeqlensCmpKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cu_seqlens_cmp_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 seqused_cmp_kv
        if (IsTensorExistQsmla(sequsedCmpKvOptional)) {
            // 校验 seqused_cmp_kv 维度
            dimNum = GetDimNumQsmla(sequsedCmpKvOptional);
            if (dimNum != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_cmp_kv must be 1, but got %lld", dimNum);
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 seqused_cmp_kv 数据类型
            dataType = GetDataTypeQsmla(sequsedCmpKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of seqused_cmp_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 cmp_residual_kv
        if (IsTensorExistQsmla(cmpResidualKvOptional)) {
            // 校验 cmp_residual_kv 维度
            dimNum = GetDimNumQsmla(cmpResidualKvOptional);
            if (dimNum != 1) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cmp_residual_kv must be 1, but got %lld", dimNum);
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cmp_residual_kv 数据类型
            dataType = GetDataTypeQsmla(cmpResidualKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cmp_residual_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 cmp_topk_length
        if (IsTensorExistQsmla(cmpTopkLengthOptional)) {
            // 校验 cmp_topk_length 维度
            dimNum = GetDimNumQsmla(cmpTopkLengthOptional);
            if (strcmp(layoutQOptional, "TND") == 0) {
                if (dimNum != 2) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                            "When layout_q is TND, the dim num of cmp_topk_length must be 2, "
                            "but got %lld",
                            dimNum);
                    return ACLNN_ERR_PARAM_INVALID;
                }
            } else if (strcmp(layoutQOptional, "BSND") == 0) {
                if (dimNum != 3) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                            "When layout_q is BSND, the dim num of cmp_topk_length must be 3, "
                            "but got %lld",
                            dimNum);
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
            // 校验 cmp_topk_length 数据类型
            dataType = GetDataTypeQsmla(cmpTopkLengthOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of cmp_topk_length must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    // 校验 metadata
    if (IsTensorExistQsmla(metadata)) {
        // 校验 metadata 维度
        dimNum = GetDimNumQsmla(metadata);
        if (dimNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of metadata must be 1, but got %lld", dimNum);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 元素数
        if (metadata->GetViewShape().GetDim(0) != optiling::QSMLA_METADATA_TOTAL_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The element num of metadata must be %u, but got %lld",
                    optiling::QSMLA_METADATA_TOTAL_SIZE, metadata->GetViewShape().GetDim(0));
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 数据类型
        dataType = GetDataTypeQsmla(metadata);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of metadata must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 q/kv 维度一致性
    int64_t queryBatchSize = GetQueryBatchSizeQsmla(sequsedQOptional, cuSeqlensQOptional, layoutQOptional, batchSize);
    // 校验TND场景q维度一致性
    if (strcmp(layoutQOptional, "TND") == 0 && IsTensorExistQsmla(sequsedQOptional)) {
        int64_t cuSeqlensQBatchSize = cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        if (cuSeqlensQBatchSize != queryBatchSize) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "when layout_q is TND and seqused_q is passed, the batch_size obtained "
                    "from cu_seqlens_q should be the same as that obtained from seqused_q, but got %lld and %lld",
                    cuSeqlensQBatchSize, queryBatchSize);
        }
    }
    if (hasOriKv) {
        int64_t oriKvBatchSize =
            GetOriKvBatchSizeQsmla(sequsedOriKvOptional, cuSeqlensOriKvOptional, layoutKvOptional, batchSize);
        // 校验q与ori_kv维度一致性
        if (queryBatchSize != oriKvBatchSize) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "when has_ori_kv is true, the batch_size obtained from q should be "
                    "the same as that obtained from ori_kv, but got %lld and %lld",
                    queryBatchSize, oriKvBatchSize);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验TND场景ori_kv维度一致性
        if (strcmp(layoutKvOptional, "TND") == 0 && IsTensorExistQsmla(sequsedOriKvOptional)) {
            int64_t cuSeqlensOriKvBatchSize = cuSeqlensOriKvOptional->GetViewShape().GetDim(0) - 1;
            if (cuSeqlensOriKvBatchSize != oriKvBatchSize) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "when has_ori_kv is true, layout_kv is TND and seqused_ori_kv is "
                    "passed, the batch_size obtained from cu_seqlens_ori_kv should be the same as that obtained from "
                    "seqused_ori_kv, but got %lld and %lld",
                    cuSeqlensOriKvBatchSize, oriKvBatchSize);
            }
        }
    }
    if (hasCmpKv) {
        int64_t cmpKvBatchSize =
            GetCmpKvBatchSizeQsmla(sequsedCmpKvOptional, cuSeqlensCmpKvOptional, layoutKvOptional, batchSize);
        // 校验q与cmp_kv维度一致性
        if (queryBatchSize != cmpKvBatchSize) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "when has_cmp_kv is true, the batch_size obtained from q should be "
                    "the same as that obtained from cmp_kv, but got %lld and %lld",
                    queryBatchSize, cmpKvBatchSize);
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验TND场景cmp_kv维度一致性
        if (strcmp(layoutKvOptional, "TND") == 0 && IsTensorExistQsmla(sequsedCmpKvOptional)) {
            int64_t cuSeqlensCmpKvBatchSize = cuSeqlensCmpKvOptional->GetViewShape().GetDim(0) - 1;
            if (cuSeqlensCmpKvBatchSize != cmpKvBatchSize) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "when has_cmp_kv is true, layout_kv is TND and seqused_cmp_kv is "
                    "passed, the batch_size obtained from cu_seqlens_cmp_kv should be the same as that obtained from "
                    "seqused_cmp_kv, but got %lld and %lld",
                    cuSeqlensCmpKvBatchSize, cmpKvBatchSize);
            }
        }
        // 校验 cmp_residual_kv 元素数
        if (IsTensorExistQsmla(cmpResidualKvOptional)) {
            if (cmpResidualKvOptional->GetViewShape().GetDim(0) != queryBatchSize) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "The elements num of cmp_residual_kv should match the valid batch "
                        "size, but got %lld and %lld",
                        cmpResidualKvOptional->GetViewShape().GetDim(0), queryBatchSize);
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsCheck(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
                               const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional,
                               const aclTensor *sequsedOriKvOptional, const aclTensor *sequsedCmpKvOptional,
                               const aclTensor *cmpResidualKvOptional, const aclTensor *oriTopkLengthOptional,
                               const aclTensor *cmpTopkLengthOptional, int64_t numHeadsQ, int64_t numHeadsKv,
                               int64_t headDim, int64_t quantMode, int64_t batchSize, int64_t maxSeqlenQ,
                               int64_t maxSeqlenOriKv, int64_t maxSeqlenCmpKv, int64_t oriTopk, int64_t cmpTopk,
                               int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft,
                               int64_t oriWinRight, const char *layoutQOptional, const char *layoutKvOptional,
                               bool hasOriKv, bool hasCmpKv, uint32_t aicCoreNum, uint32_t aivCoreNum,
                               const char *socVersion, const aclTensor *metaData)
{
    if (CheckSingleParamQsmla(batchSize, maxSeqlenQ, maxSeqlenOriKv, maxSeqlenCmpKv, numHeadsQ, numHeadsKv, headDim,
                              quantMode, oriTopk, cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
                              layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv, aicCoreNum, aivCoreNum,
                              socVersion) == ACLNN_SUCCESS &&
        CheckExistenceQsmla(cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedOriKvOptional,
                            sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional,
                            oriTopk, cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, hasOriKv, hasCmpKv, layoutQOptional,
                            layoutKvOptional, metaData) == ACLNN_SUCCESS &&
        CheckConsistencyQsmla(cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional,
                              sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional,
                              cmpTopkLengthOptional, batchSize, layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv,
                              metaData) == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    } else {
        return ACLNN_ERR_PARAM_INVALID;
    }
}
} // namespace

#ifdef __cplusplus
}
#endif
