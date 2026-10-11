/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_allto_all_matmul.h
 * \brief
 */
#ifndef OP_API_INC_ALL_TO_ALL_MATMUL_BASE
#define OP_API_INC_ALL_TO_ALL_MATMUL_BASE

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllMatmulBaseGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional, const aclIntArray* alltoAllAxesOptional,
    const char* group, const char* commMode, bool transposeX1, bool transposeX2, const aclTensor* output,
    const aclTensor* alltoAllOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllMatmulBase(void* workspace, uint64_t workspaceSize,
                                                                           aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ALL_TO_ALL_MATMUL_
