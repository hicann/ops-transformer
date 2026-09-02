/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_QUANT_FLASH_ATTN_H_
#define OP_API_INC_LEVEL0_QUANT_FLASH_ATTN_H_

#include <array>
#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief QuantFlashAttn level-0 operator.
 *        Encapsulates the low-level scheduling of QuantFlashAttn, completing InferShape and Kernel Launch registration.
 *        This interface is internal and only for use by the aclnn layer.
 *
 * @param q                   query tensor (FP8_E4M3 for quant_mode=1)
 * @param k                   key tensor (FP8_E4M3 for quant_mode=1)
 * @param v                   value tensor (FP8_E4M3 for quant_mode=1)
 * @param qDescale            query descale tensor (E8M0 for quant_mode=1)
 * @param kDescale            key descale tensor (E8M0 for quant_mode=1)
 * @param vDescale            value descale tensor (E8M0 for quant_mode=1)
 * @param blockTableOptional  optional block index table for paged attention (INT32)
 * @param pScaleOptional      optional P scale tensor
 * @param cuSeqlensQOptional  query cumulative sequence lengths (optional, INT32)
 * @param cuSeqlensKvOptional kv cumulative sequence lengths (optional, INT32)
 * @param sequsedQOptional    query actual sequence lengths per batch (optional, INT32)
 * @param sequsedKvOptional   kv actual sequence lengths per batch (optional, INT32)
 * @param sinksOptional       learnable sink weights (optional, FLOAT32)
 * @param attnMaskOptional    attention mask (optional, INT8)
 * @param metadataOptional    pre-computed tiling metadata (optional, INT32)
 * @param quantMode           quantization mode (int64_t): 1=A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32
 * @param softmaxScale        softmax scaling factor (double)
 * @param maskMode            mask mode (int64_t)
 * @param winLeft             left window size (int64_t)
 * @param winRight            right window size (int64_t)
 * @param maxSeqlenQ          max query sequence length (int64_t)
 * @param maxSeqlenKV         max kv sequence length (int64_t)
 * @param layoutQ             query layout string
 * @param layoutQDescale      q descale layout string
 * @param layoutKv            kv layout string
 * @param layoutOut           output layout string
 * @param returnSoftmaxLse    whether to output softmax_lse (bool)
 * @param executor            op executor
 * @return std::array<const aclTensor*, 2> [attnOut, softmaxLse]
 *         Any element being nullptr indicates InferShape or Launch failure for that output.
 */
const std::array<const aclTensor *, 2> QuantFlashAttn(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *qDescale, const aclTensor *kDescale,
    const aclTensor *vDescale, const aclTensor *blockTableOptional, const aclTensor *pScaleOptional,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional, const aclTensor *sinksOptional, const aclTensor *attnMaskOptional,
    const aclTensor *metadataOptional, int64_t quantMode, double softmaxScale, int64_t maskMode, int64_t winLeft,
    int64_t winRight, int64_t maxSeqlenQ, int64_t maxSeqlenKV, const char *layoutQ, const char *layoutQDescale,
    const char *layoutKv, const char *layoutOut, bool returnSoftmaxLse, aclOpExecutor *executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_QUANT_FLASH_ATTN_H_
