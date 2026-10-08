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
 * \file l0_quant_flash_mla_with_kvcache_metadata.cpp
 * \brief QuantFlashMlaWithKvcacheMetadata l0层实现: 发射AICPU任务执行负载均衡分核
 */

#include "l0_quant_flash_mla_with_kvcache_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(QuantFlashMlaWithKvcacheMetadata);

const aclTensor* QuantFlashMlaWithKvcacheMetadata(const aclTensor* cacheSeqlens, const aclTensor* cuSeqlensQOptional,
                                                  const aclTensor* sequsedQOptional, int64_t maxSeqlenQ,
                                                  int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv,
                                                  int64_t headDimQk, int64_t headDimV, int64_t quantMode,
                                                  int64_t maskMode, const char* layoutQ, const char* socVersion,
                                                  int64_t aicCoreNum, int64_t aivCoreNum, const aclTensor* metaData,
                                                  aclOpExecutor* executor)
{
    L0_DFX(QuantFlashMlaWithKvcacheMetadata, cacheSeqlens, cuSeqlensQOptional, sequsedQOptional, maxSeqlenQ,
           maxSeqlenKv, numHeadsQ, numHeadsKv, headDimQk, headDimV, quantMode, maskMode, layoutQ, socVersion,
           aicCoreNum, aivCoreNum, metaData);

    static internal::AicpuTaskSpace space("QuantFlashMlaWithKvcacheMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        QuantFlashMlaWithKvcacheMetadata,
        OP_ATTR_NAMES({"max_seqlen_q", "max_seqlen_kv", "num_heads_q", "num_heads_kv", "head_dim_qk", "head_dim_v",
                       "quant_mode", "mask_mode", "layout_q", "soc_version", "aic_core_num", "aiv_core_num"}),
        OP_INPUT(cacheSeqlens, cuSeqlensQOptional, sequsedQOptional), OP_OUTPUT(metaData),
        OP_ATTR(maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDimQk, headDimV, quantMode, maskMode, layoutQ,
                socVersion, aicCoreNum, aivCoreNum));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "QuantFlashMlaWithKvcacheMetadata"
                                              " ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return metaData;
}

} // namespace l0op
