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
 * \file generic_block_sparse_attention_grad_metadata.cpp
 * \brief
 */

#include "generic_block_sparse_attention_grad_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(GenericBlockSparseAttentionGradMetadata);

const aclTensor *GenericBlockSparseAttentionGradMetadata(
    const aclTensor *rsvdBlockIdx, const aclTensor *rsvdBlockCount, const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t numQHeads, int64_t numKvHeads, int64_t headDim,
    int64_t blockShapeX, int64_t blockShapeY, int64_t isPackedGQA, const char *layoutQOptional,
    const char *layoutKvOptional, int64_t maskType, int64_t softmaxPrecision, int64_t winLeft, int64_t winRight,
    const char *socVersion, int64_t aicCoreNum, int64_t aivCoreNum, const aclTensor *metaData, aclOpExecutor *executor)
{
    L0_DFX(GenericBlockSparseAttentionGradMetadata, rsvdBlockIdx, rsvdBlockCount, cuSeqLengthsQOptional,
           cuSeqLengthsKvOptional, sequsedQOptional, sequsedKvOptional, maxQSeqlen, maxKvSeqlen, numQHeads, numKvHeads,
           headDim, blockShapeX, blockShapeY, isPackedGQA, layoutQOptional, layoutKvOptional, maskType,
           softmaxPrecision, winLeft, winRight, socVersion, aicCoreNum, aivCoreNum, metaData);

    static internal::AicpuTaskSpace space("GenericBlockSparseAttentionGradMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        GenericBlockSparseAttentionGradMetadata,
        OP_ATTR_NAMES({"max_q_seqlen", "max_kv_seqlen", "num_q_heads", "num_kv_heads", "head_dim", "block_shape_x",
                       "block_shape_y", "is_packed_gqa", "q_input_layout", "kv_input_layout", "mask_type",
                       "softmax_precision", "window_size_left", "window_size_right", "soc_version", "aic_core_num",
                       "aiv_core_num"}),
        OP_INPUT(rsvdBlockIdx, rsvdBlockCount, cuSeqLengthsQOptional, cuSeqLengthsKvOptional, sequsedQOptional,
                 sequsedKvOptional),
        OP_OUTPUT(metaData),
        OP_ATTR(maxQSeqlen, maxKvSeqlen, numQHeads, numKvHeads, headDim, blockShapeX, blockShapeY, isPackedGQA,
                layoutQOptional, layoutKvOptional, maskType, softmaxPrecision, winLeft, winRight, socVersion,
                aicCoreNum, aivCoreNum));
    OP_CHECK(
        ret == ACL_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GenericBlockSparseAttentionGradMetadata ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return metaData;
}

} // namespace l0op
