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
 * \file sparse_attn_sharedkv_metadata_check.h.cpp
 * \brief
 */

#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "../../quant_lightning_indexer/op_kernel/quant_lightning_indexer_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus CheckSingleParamQli(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK, int64_t numHeadsQ,
                                int64_t numHeadsK, char *layoutQueryOptional, char *layoutKeyOptional,
                                int64_t sparseMode, int64_t preTokens, int64_t nextTokens, int64_t cmpRatio,
                                NpuArch npuArch)
{
    // batch_size 非负校验
    if (batchSize < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch_size should not be negative, but got %ld", batchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q 非负校验
    if (maxSeqlenQ < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_q should not be negative, but got %ld", maxSeqlenQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_k 非负校验
    if (maxSeqlenK < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_k should not be negative, but got %ld", maxSeqlenK);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_kv 校验
    if (numHeadsQ != 64) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_q should only be 64, but got %ld", numHeadsQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_k 校验
    if (numHeadsK != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_k should only be 1, but got %ld", numHeadsK);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_query 校验
    if ((strcmp(layoutQueryOptional, "TND") != 0) && (strcmp(layoutQueryOptional, "BSND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_query must be TND or BSND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_kv 校验
    if ((strcmp(layoutKeyOptional, "PA_BSND") != 0) && (strcmp(layoutKeyOptional, "TND") != 0) &&
        (strcmp(layoutKeyOptional, "BSND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_key must be TND, BSND or PA_BSND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout交叉校验
    if (strcmp(layoutQueryOptional, "TND") == 0 && strcmp(layoutKeyOptional, "BSND") == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_query TND, layout_key should be PA_BSND/TND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutQueryOptional, "BSND") == 0 && strcmp(layoutKeyOptional, "TND") == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_query BSND, layout_key should be PA_BSND/BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // sparse_mode 校验
    if (sparseMode != 0 && sparseMode != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sparse_mode should be 0, 3, but got %ld", sparseMode);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // pre_tokens 校验
    if (preTokens != INT64_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pre_tokens should only be 2^63-1, but got %ld", preTokens);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // next_tokens 校验
    if (nextTokens != INT64_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "next_tokens should only be 2^63-1, but got %ld", nextTokens);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // cmp_ratio 校验
    if (cmpRatio < 1 || cmpRatio > 128) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmp_ratio should be [1, 128], but got %d", cmpRatio);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (npuArch == NpuArch::DAV_3510) {
        if (cmpRatio != 1 && cmpRatio != 4 && cmpRatio != 128) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For Ascend950, cmp_ratio should be 1/4/128, but got %d", cmpRatio);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if ((cmpRatio & (cmpRatio - 1)) != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For Atlas A3, cmp_ratio should be 1/2/4/8/16/32/64/128, but got %d",
                    cmpRatio);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceQli(char *layoutQueryOptional, char *layoutKeyOptional,
                              const aclTensor *actualSeqLengthsQueryOptional,
                              const aclTensor *actualSeqLengthsKeyOptional, const aclTensor *metadata)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    // cu_seqlens_q 存在性校验
    if (strcmp(layoutQueryOptional, "TND") == 0) {
        auto ret = aclGetViewShape(actualSeqLengthsQueryOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_query TND, actual_seq_lengths_query must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // seqused_kv 存在性校验
    if (strcmp(layoutKeyOptional, "PA_BSND") == 0) {
        auto ret = aclGetViewShape(actualSeqLengthsKeyOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_key PA_BSND, actual_seq_lengths_key must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (strcmp(layoutKeyOptional, "TND") == 0) {
        auto ret = aclGetViewShape(actualSeqLengthsKeyOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_key TND, actual_seq_lengths_key must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // metadata 存在性校验
    auto ret = aclGetViewShape(metadata, &viewDims, &viewDimsNum);
    if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Output metadata is nullptr");
        delete[] viewDims;
        return ACLNN_ERR_PARAM_INVALID;
    }
    delete[] viewDims;
    return ACLNN_SUCCESS;
}

int64_t GetQueryBatchSizeQli(const aclTensor *actualSeqLengthsQueryOptional, int64_t batchSize)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(actualSeqLengthsQueryOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    delete[] viewDims;
    return batchSize;
}

int64_t GetKvBatchSizeQli(const aclTensor *actualSeqLengthsKeyOptional, int64_t batchSize)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(actualSeqLengthsKeyOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    delete[] viewDims;
    return batchSize;
}

aclnnStatus CheckConsistencyQli(char *layoutQueryOptional, char *layoutKeyOptional,
                                const aclTensor *actualSeqLengthsQueryOptional,
                                const aclTensor *actualSeqLengthsKeyOptional, int64_t batchSize,
                                const aclTensor *metadata)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    // 校验 actual_seq_lengths_query
    auto ret = aclGetViewShape(actualSeqLengthsQueryOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 校验 actual_seq_lengths_query 维度
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of actual_seq_lengths_query must be 1, but got %lu",
                    viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 actual_seq_lengths_query 数据类型
        ret = aclGetDataType(actualSeqLengthsQueryOptional, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of actual_seq_lengths_query must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 actual_seq_lengths_key
    ret = aclGetViewShape(actualSeqLengthsKeyOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 校验 actual_seq_lengths_key 维度
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of actual_seq_lengths_key must be 1, but got %lu",
                    viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 actual_seq_lengths_key 数据类型
        ret = aclGetDataType(actualSeqLengthsKeyOptional, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of actual_seq_lengths_key must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 metadata
    ret = aclGetViewShape(metadata, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        // 校验 metadata 维度
        if (viewDimsNum != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of metadata must be 1, but got %lu", viewDimsNum);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 元素数
        if (viewDims[0] != optiling::QLI_META_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The element num of metadata must be %u, but got %ld",
                    optiling::QLI_META_SIZE, viewDims[0]);
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 数据类型
        ret = aclGetDataType(metadata, &dataType);
        if (ret != ACLNN_ERR_PARAM_INVALID && dataType != aclDataType::ACL_INT32) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of metadata must be int32");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    int64_t queryBatchSize = GetQueryBatchSizeQli(actualSeqLengthsQueryOptional, batchSize);
    int64_t kvBatchSize = GetKvBatchSizeQli(actualSeqLengthsKeyOptional, batchSize);
    if ((strcmp(layoutQueryOptional, "BSND") == 0 ||
         (strcmp(layoutQueryOptional, "TND") == 0 && strcmp(layoutKeyOptional, "TND") == 0)) &&
        queryBatchSize != kvBatchSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "For the layout_query is BSND or both layout_query and layout_key are TND, "
                "the batch_size obtained from q Tensor should be the same as that obtained from kv tensor, "
                "but got %ld and %ld",
                queryBatchSize, kvBatchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (std::abs(queryBatchSize - kvBatchSize) > 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The difference between the dim of actual_seq_lengths_query and the dim of "
                "actual_seq_lengths_key should not be greater than 1, but got %ld and %ld",
                queryBatchSize, kvBatchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsCheck(const aclTensor *actualSeqLengthsQueryOptional,
                               const aclTensor *actualSeqLengthsKeyOptional, int64_t numHeadsQ, int64_t numHeadsK,
                               int64_t headDim, int64_t queryQuantMode, int64_t keyQuantMode, int64_t batchSize,
                               int64_t maxSeqlenQ, int64_t maxSeqlenK, char *layoutQueryOptional,
                               char *layoutKeyOptional, int64_t sparseCount, int64_t sparseMode, int64_t preTokens,
                               int64_t nextTokens, int64_t cmpRatio, NpuArch npuArch, const aclTensor *metadata)
{
    if (CheckSingleParamQli(batchSize, maxSeqlenQ, maxSeqlenK, numHeadsQ, numHeadsK, layoutQueryOptional,
                            layoutKeyOptional, sparseMode, preTokens, nextTokens, cmpRatio, npuArch) == ACLNN_SUCCESS &&
        CheckExistenceQli(layoutQueryOptional, layoutKeyOptional, actualSeqLengthsQueryOptional,
                          actualSeqLengthsKeyOptional, metadata) == ACLNN_SUCCESS &&
        CheckConsistencyQli(layoutQueryOptional, layoutKeyOptional, actualSeqLengthsQueryOptional,
                            actualSeqLengthsKeyOptional, batchSize, metadata) == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    } else {
        return ACLNN_ERR_PARAM_INVALID;
    }
}

#ifdef __cplusplus
}
#endif
