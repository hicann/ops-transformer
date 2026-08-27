/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_INPLACE_FUSED_CAUSAL_CONV1D_L0OP_H_
#define OP_API_INC_LEVEL0_INPLACE_FUSED_CAUSAL_CONV1D_L0OP_H_

#include <tuple>
#include "opdev/op_executor.h"

namespace l0op {
bool InplaceFusedCausalConv1d(aclTensor *x, const aclTensor *weight, aclTensor *convStates,
                              const aclTensor *queryStartLoc, const aclTensor *cacheIndices,
                              const aclTensor *initialStateMode, const aclTensor *bias,
                              const aclTensor *numAcceptedTokens, const aclTensor *numComputedTokens,
                              const aclTensor *blockIdxFirstScheduledToken, const aclTensor *blockIdxLastScheduledToken,
                              const aclTensor *initialStateIdx, int64_t activationMode, int64_t padSlotId,
                              int64_t runMode, int64_t maxQueryLen, int64_t residualConnection, int64_t blockSize,
                              int64_t convMode, int64_t maxDraftTokens, aclOpExecutor *executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_INPLACE_FUSED_CAUSAL_CONV1D_L0OP_H_
