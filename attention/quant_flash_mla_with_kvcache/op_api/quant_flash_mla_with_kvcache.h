/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_QUANT_FLASH_MLA_WITH_KVCACHE_H_
#define OP_API_INC_LEVEL0_QUANT_FLASH_MLA_WITH_KVCACHE_H_

#include <array>
#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief QuantFlashMlaWithKvcache level-0 operator.
 *        Encapsulates the low-level scheduling of QuantFlashMlaWithKvcache, completing InferShape and
 *        Kernel Launch registration. This interface is internal and only for use by the aclnn layer.
 *
 * @param q                   query tensor (FP8_E4M3/HIFLOAT8, layout_q: BSND/BNSD/TND)
 * @param kCache              kv cache tensor (FP8_E4M3/HIFLOAT8, PA_BBND/PA_BNBD/PA_NZ, 支持非连续)
 * @param qDescale            query descale tensor (FP32, per-token-head动态量化)
 * @param kDescale            key descale tensor (FP32, per-tensor静态量化, shape为(1,))
 * @param blockTable          paged attention块索引映射表 (INT32)
 * @param cacheSeqlens        每个batch的KV序列长度 (INT32)
 * @param cuSeqlensQOptional  query累积序列长度 (optional, INT32)
 * @param sequsedQOptional    每batch实际使用的query序列长度 (optional, INT32)
 * @param attnMaskOptional    注意力掩码 (optional, INT8)
 * @param metadataOptional    预计算的任务切分结果 (optional, INT32)
 * @param quantMode           量化模式 (int64_t): 0=HIF8场景, 1=FP8_E4M3场景
 * @param softmaxScale        softmax缩放系数 (double), 未传入时默认取1/sqrt(headdim)
 * @param maskMode            掩码模式 (int64_t): 0=NO_MASK, 3=CAUSAL
 * @param maxSeqlenQ          query最大序列长度 (int64_t)
 * @param maxSeqlenKv         kv最大序列长度 (int64_t)
 * @param headDimV            v的每个注意力头维度 (int64_t), 当前仅支持512
 * @param layoutQ             query布局字符串: BSND/BNSD/TND
 * @param layoutKv            kv cache布局字符串: PA_BBND/PA_BNBD/PA_NZ
 * @param layoutOut           输出布局字符串: BSND/BNSD/TND/NTD
 * @param returnSoftmaxLse    是否输出softmax_lse (bool)
 * @param executor            op executor
 * @return std::array<const aclTensor*, 2> [attnOut, softmaxLse]
 *         Any element being nullptr indicates InferShape or Launch failure for that output.
 */
const std::array<const aclTensor*, 2> QuantFlashMlaWithKvcache(
    const aclTensor* q, const aclTensor* kCache, const aclTensor* qDescale, const aclTensor* kDescale,
    const aclTensor* blockTable, const aclTensor* cacheSeqlens, const aclTensor* cuSeqlensQOptional,
    const aclTensor* sequsedQOptional, const aclTensor* attnMaskOptional, const aclTensor* metadataOptional,
    int64_t quantMode, double softmaxScale, int64_t maskMode, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t headDimV,
    const char* layoutQ, const char* layoutKv, const char* layoutOut, bool returnSoftmaxLse, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_QUANT_FLASH_MLA_WITH_KVCACHE_H_
