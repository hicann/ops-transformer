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
 * \file l0_sparse_flash_mla_softmax_l1_norm_metadata.cpp
 * \brief
 */

#include "l0_sparse_flash_mla_softmax_l1_norm_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SparseFlashMlaSoftmaxL1NormMetadata);

const aclTensor *SparseFlashMlaSoftmaxL1NormMetadata(
    const aclTensor *cuSeqLensQOptional, const aclTensor *cuSeqLensKOptional, const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKOptional, const aclTensor *cmpResidualKOptional, const aclTensor *topkLengthOptional,
    int64_t batchSize, int64_t maxSeqLenQ, int64_t maxSeqLenK, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim,
    int64_t topk, int64_t cmpRatio, int64_t maskMode, char *layoutQ, char *layoutK, const char *socVersion,
    int64_t aicCoreNum, const aclTensor *metadata, aclOpExecutor *executor)
{
    L0_DFX(SparseFlashMlaSoftmaxL1NormMetadata, cuSeqLensQOptional, cuSeqLensKOptional, seqUsedQOptional,
           seqUsedKOptional, cmpResidualKOptional, topkLengthOptional, batchSize, maxSeqLenQ, maxSeqLenK, numHeadsQ,
           numHeadsK, headDim, topk, cmpRatio, maskMode, layoutQ, layoutK, socVersion, aicCoreNum, metadata);

    static internal::AicpuTaskSpace space("SparseFlashMlaSoftmaxL1NormMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        SparseFlashMlaSoftmaxL1NormMetadata,
        OP_ATTR_NAMES({"batch_size", "max_seqlen_q", "max_seqlen_k", "num_heads_q", "num_heads_k", "head_dim", "topk",
                       "cmp_ratio", "mask_mode", "layout_q", "layout_k", "soc_version", "aic_core_num"}),
        OP_INPUT(cuSeqLensQOptional, cuSeqLensKOptional, seqUsedQOptional, seqUsedKOptional, cmpResidualKOptional,
                 topkLengthOptional),
        OP_OUTPUT(metadata),
        OP_ATTR(batchSize, maxSeqLenQ, maxSeqLenK, numHeadsQ, numHeadsK, headDim, topk, cmpRatio, maskMode, layoutQ,
                layoutK, socVersion, aicCoreNum));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SparseFlashMlaSoftmaxL1NormMetadata ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return metadata;
}
} // namespace l0op
