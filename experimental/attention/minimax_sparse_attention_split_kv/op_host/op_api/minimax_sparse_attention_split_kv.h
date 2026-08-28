/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MINIMAX_SPARSE_ATTENTION_SPLIT_KV_L0OP_H_
#define MINIMAX_SPARSE_ATTENTION_SPLIT_KV_L0OP_H_

#include <tuple>
#include "opdev/op_executor.h"

namespace l0op {

std::tuple<const aclTensor *, const aclTensor *> MinimaxSparseAttentionSplitKv(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *blockTable,
    const aclTensor *k2qRowPtr, const aclTensor *k2qQIndices, const aclTensor *k2qSlotIndices,
    const aclTensor *actualSeqLengths, const aclTensor *actualSeqLengthsKv, int64_t numKeyValueHeads, double scaleValue,
    int64_t blockSize, int64_t topK, int64_t innerPrecise, bool softmaxLseFlag, const char *inputLayout,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, aclOpExecutor *executor);

} // namespace l0op

#endif // MINIMAX_SPARSE_ATTENTION_SPLIT_KV_L0OP_H_
