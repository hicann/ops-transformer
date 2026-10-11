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
 * \file matmul_allto_all_base.h
 * \brief
 */
#ifndef OP_API_INC_MATMUL_ALL_TO_ALL_BASE_
#define OP_API_INC_MATMUL_ALL_TO_ALL_BASE_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

ACLNN_API aclnnStatus aclnnMatmulAlltoAllBaseGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                              const aclTensor* biasOptional,
                                                              const aclIntArray* alltoAllAxesOptional,
                                                              const char* group, const char* commMode, bool transposeX1,
                                                              bool transposeX2, const aclTensor* output,
                                                              uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnMatmulAlltoAllBase(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              aclrtStream stream);

#endif // OP_API_INC_MATMUL_ALL_TO_ALL_BASE_
