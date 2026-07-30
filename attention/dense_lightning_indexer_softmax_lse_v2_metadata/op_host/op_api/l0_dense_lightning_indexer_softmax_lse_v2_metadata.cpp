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
 * \\file l0_dense_lightning_indexer_softmax_lse_v2_metadata.cpp
 * \brief
 */

#include "l0_dense_lightning_indexer_softmax_lse_v2_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(DenseLightningIndexerSoftmaxLseV2Metadata);

const aclTensor *DenseLightningIndexerSoftmaxLseV2Metadata(
    const aclTensor *cuSeqLensQOptional, const aclTensor *cuSeqLensKOptional, const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKOptional, const aclTensor *cmpResidualKOptional, int64_t batchSize, int64_t maxSeqLenQ,
    int64_t maxSeqLenK, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, char *layoutQ, char *layoutK,
    int64_t maskMode, int64_t cmpRatio, const char *socVersion, int64_t aicCoreNum, const aclTensor *metadata,
    aclOpExecutor *executor)
{
    L0_DFX(DenseLightningIndexerSoftmaxLseV2Metadata, cuSeqLensQOptional, cuSeqLensKOptional, seqUsedQOptional,
           seqUsedKOptional, cmpResidualKOptional, batchSize, maxSeqLenQ, maxSeqLenK, numHeadsQ, numHeadsK, headDim,
           layoutQ, layoutK, maskMode, cmpRatio, socVersion, aicCoreNum, metadata);

    static internal::AicpuTaskSpace space("DenseLightningIndexerSoftmaxLseV2Metadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        DenseLightningIndexerSoftmaxLseV2Metadata,
        OP_ATTR_NAMES({"batch_size", "max_seqlen_q", "max_seqlen_k", "num_heads_q", "num_heads_k", "head_dim",
                       "layout_q", "layout_k", "mask_mode", "cmp_ratio", "soc_version", "aic_core_num"}),
        OP_INPUT(cuSeqLensQOptional, cuSeqLensKOptional, seqUsedQOptional, seqUsedKOptional, cmpResidualKOptional),
        OP_OUTPUT(metadata),
        OP_ATTR(batchSize, maxSeqLenQ, maxSeqLenK, numHeadsQ, numHeadsK, headDim, layoutQ, layoutK, maskMode, cmpRatio,
                socVersion, aicCoreNum));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "DenseLightningIndexerSoftmaxLseV2Metadata ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return metadata;
}
} // namespace l0op
