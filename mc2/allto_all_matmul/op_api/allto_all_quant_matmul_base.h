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
 * \file aclnn_allto_all_quant_matmul.h
 * \brief
 */
#ifndef OP_API_INC_ALL_TO_ALL_QUANT_MATMUL_BASE_
#define OP_API_INC_ALL_TO_ALL_QUANT_MATMUL_BASE_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllQuantMatmulBaseGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional, const aclTensor* x1ScaleOptional,
    const aclTensor* x2Scale, const aclTensor* commScaleOptional, const aclTensor* x1OffsetOptional,
    const aclTensor* x2OffsetOptional, const char* group, const char* commMode, const aclIntArray* alltoAllAxesOptional,
    int64_t x1QuantMode, int64_t x2QuantMode, int64_t commQuantMode, int64_t commQuantDtype, int64_t x1QuantDtype,
    int64_t groupSize, bool transposeX1, bool transposeX2, const aclTensor* output,
    const aclTensor* alltoAllOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnAlltoAllQuantMatmulBase(void* workspace, uint64_t workspaceSize,
                                                                                aclOpExecutor* executor,
                                                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ALL_TO_ALL_QUANT_MATMUL_
