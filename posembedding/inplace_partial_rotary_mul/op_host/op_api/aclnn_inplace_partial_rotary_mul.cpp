/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_inplace_partial_rotary_mul.h"
#include "aclnnInner_inplace_partial_rotary_mul.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnInplacePartialRotaryMulGetWorkspaceSize(const aclTensor* xRef, const aclTensor* cos,
                                                         const aclTensor* sin, int64_t rotary_mode,
                                                         const aclIntArray* partialSlice, uint64_t* workspaceSize,
                                                         aclOpExecutor** executor)
{
    aclnnStatus ret = aclnnInnerInplacePartialRotaryMulGetWorkspaceSize(
        const_cast<aclTensor*>(xRef), cos, sin, rotary_mode, partialSlice, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnInplacePartialRotaryMul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerInplacePartialRotaryMul(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
