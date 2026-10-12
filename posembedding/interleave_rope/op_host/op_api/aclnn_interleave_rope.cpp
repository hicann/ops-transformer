/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_interleave_rope.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "posembedding/rotary_position_embedding/op_host/op_api/aclnn_rotary_position_embedding.h"
#include "aclnnInner_interleave_rope.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

constexpr int64_t HALF_INTERLEAVE_MODE = 3;

aclnnStatus aclnnInterleaveRopeGetWorkspaceSize(const aclTensor* x, const aclTensor* cos, const aclTensor* sin,
                                                aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (Ops::Transformer::AclnnUtil::IsRegbase()) {
        return aclnnRotaryPositionEmbeddingGetWorkspaceSize(x, cos, sin, HALF_INTERLEAVE_MODE, out, workspaceSize,
                                                            executor);
    } else {
        return aclnnInnerInterleaveRopeGetWorkspaceSize(x, cos, sin, out, workspaceSize, executor);
    }
}

aclnnStatus aclnnInterleaveRope(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    if (Ops::Transformer::AclnnUtil::IsRegbase()) {
        return aclnnRotaryPositionEmbedding(workspace, workspaceSize, executor, stream);
    } else {
        return aclnnInnerInterleaveRope(workspace, workspaceSize, executor, stream);
    }
}

#ifdef __cplusplus
}
#endif
