/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_SPARSE_ATTENTION_GRAD_H
#define BLOCK_SPARSE_ATTENTION_GRAD_H

#include <array>
#include "opdev/op_executor.h"

namespace l0op {

const std::array<const aclTensor *, 3> BlockSparseAttentionGrad(
    const aclTensor *dout, const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, const aclTensor *blockSparseMaskOptional,
    const aclTensor *attenMaskOptional, const aclIntArray *blockShapeOptional,
    const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional, char *qInputLayout,
    char *kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue, int64_t preTokens,
    int64_t nextTokens, aclOpExecutor *executor);

} // namespace l0op

#endif // BLOCK_SPARSE_ATTENTION_GRAD_H
