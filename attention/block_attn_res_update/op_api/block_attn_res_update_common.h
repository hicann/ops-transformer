/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_BLOCK_ATTN_RES_UPDATE_COMMON_H_
#define OP_API_BLOCK_ATTN_RES_UPDATE_COMMON_H_

#include "aclnn/aclnn_base.h"

namespace BlockAttnResUpdateAclnn {

struct BlockAttnResUpdateParams {
    aclTensor* partialBlockRef{nullptr};
    const aclTensor* delta{nullptr};
    const aclTensor* pseudoQuery{nullptr};
    const aclTensor* numerator{nullptr};
    const aclTensor* logitMax{nullptr};
    const aclTensor* expSum{nullptr};
    double eps{1e-6};
    aclTensor* h{nullptr};
    const aclTensor* outputNormWeightOptional{nullptr};
    double outputNormEps{1e-6};
};

// Shared validation and executor construction for the original and V2 ACLNN interfaces.
aclnnStatus GetWorkspaceSize(const char* apiName, BlockAttnResUpdateParams params, uint64_t* workspaceSize,
                             aclOpExecutor** executor);

} // namespace BlockAttnResUpdateAclnn

#endif // OP_API_BLOCK_ATTN_RES_UPDATE_COMMON_H_
