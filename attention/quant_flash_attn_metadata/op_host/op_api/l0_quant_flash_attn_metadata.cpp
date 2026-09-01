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
 * \file l0_quant_flash_attn_metadata.cpp
 * \brief
 */

#include "l0_quant_flash_attn_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(QuantFlashAttnMetadata);

const aclTensor *QuantFlashAttnMetadata(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
                                        const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
                                        int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ,
                                        int64_t numHeadsKv, int64_t headDim, int64_t headDimV, int64_t quantMode,
                                        int64_t maskMode, int64_t winLeft, int64_t winRight, const char *layoutQ,
                                        const char *layoutQDescale, const char *layoutKv, const char *layoutOut,
                                        bool isGradEnabled, const char *socVersion, int64_t aicCoreNum,
                                        int64_t aivCoreNum, const aclTensor *metaData, aclOpExecutor *executor)
{
    L0_DFX(QuantFlashAttnMetadata, cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional,
           batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, headDimV, quantMode, maskMode, winLeft,
           winRight, layoutQ, layoutQDescale, layoutKv, layoutOut, isGradEnabled, socVersion, aicCoreNum, aivCoreNum,
           metaData);

    static internal::AicpuTaskSpace space("QuantFlashAttnMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        QuantFlashAttnMetadata,
        OP_ATTR_NAMES({"batch_size", "max_seqlen_q", "max_seqlen_kv", "num_heads_q", "num_heads_kv", "head_dim",
                       "head_dim_v", "quant_compute_mode", "mask_mode", "win_left", "win_right", "layout_q",
                       "layout_q_descale", "layout_kv", "layout_out", "is_grad_enabled", "soc_version", "aic_core_num",
                       "aiv_core_num"}),
        OP_INPUT(cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional), OP_OUTPUT(metaData),
        OP_ATTR(batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, headDimV, quantMode, maskMode,
                winLeft, winRight, layoutQ, layoutQDescale, layoutKv, layoutOut, isGradEnabled, socVersion, aicCoreNum,
                aivCoreNum));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "QuantFlashAttnMetadata"
                                              " ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return metaData;
}

} // namespace l0op
