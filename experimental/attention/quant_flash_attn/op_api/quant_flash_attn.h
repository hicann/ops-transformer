/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
 * @brief QuantFlashAttn level-0 operator。
 *        封装QuantFlashAttn算子的底层调度，完成InferShape与Kernel Launch注册。
 *        该接口为内部接口，仅供aclnn层调用。
 *
 * @param q                   query tensor
 * @param k                   key tensor
 * @param v                   value tensor
 * @param qDescale            query descale tensor
 * @param kDescale            key descale tensor
 * @param vDescale            value descale tensor
 * @param blockTableOptional  分页KV缓存块映射表（可选，INT32）
 * @param pScaleOptional      P scale tensor（可选，FLOAT32）
 * @param cuSeqlensQOptional  query累积序列长度tensor（可选，INT32）
 * @param cuSeqlensKvOptional kv累积序列长度tensor（可选，INT32）
 * @param sequsedQOptional    query各batch实际序列长度tensor（可选，INT32）
 * @param sequsedKvOptional   kv各batch实际序列长度tensor（可选，INT32）
 * @param sinksOptional       可学习sink权重（可选，FLOAT32）
 * @param attnMaskOptional    attnMask参数（可选，INT8）
 * @param metadataOptional    预计算tiling元数据（可选，INT32）
 * @param quantMode           量化模式（int64_t）
 * @param softmaxScale        softmax缩放系数（double）
 * @param maskMode            掩码模式（int64_t）
 * @param winLeft             左窗口大小（int64_t）
 * @param winRight            右窗口大小（int64_t）
 * @param maxSeqlenQ          query最大序列长度（int64_t）
 * @param maxSeqlenKV         kv最大序列长度（int64_t）
 * @param layoutQ             query布局字符串
 * @param layoutQDescale      q descale 布局字符串（mxfp4 不使用，透传保持接口一致）
 * @param layoutKv            kv布局字符串
 * @param layoutOut           输出布局字符串
 * @param returnSoftmaxLse    是否输出softmax_lse（bool）
 * @param executor            op执行器
 * @return std::array<const aclTensor*, 2> [attnOut, softmaxLse]
 *         任意元素为nullptr表示对应输出的InferShape或Launch失败。
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
