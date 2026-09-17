/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_COMPUTE_WY_OP_API_H
#define CHUNK_GATED_DELTA_RULE_COMPUTE_WY_OP_API_H

#include "opdev/op_executor.h"

namespace l0op {
const std::array<const aclTensor *, 5> ChunkGatedDeltaRuleComputeWy(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g, const aclTensor *beta,
    int64_t chunkSize, const aclTensor *qKernelOut, const aclTensor *kKernelOut, const aclTensor *wKernelOut,
    const aclTensor *uKernelOut, const aclTensor *gKernelOut, aclOpExecutor *executor);
}

#endif
