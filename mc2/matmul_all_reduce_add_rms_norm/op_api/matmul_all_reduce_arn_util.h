/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_SRC_LEVEL2_MATMUL_ALL_REDUCE_ARN_UTIL_H_
#define OP_API_SRC_LEVEL2_MATMUL_ALL_REDUCE_ARN_UTIL_H_

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "acl/acl.h"
#include "matmul_all_reduce/op_api/matmul_all_reduce_util.h"

#include <atomic>

#ifdef __cplusplus
extern "C" {
#endif

// MatmulAllReduceAddRmsNorm
bool ArnCheckNotNull(const aclTensor *x1, const aclTensor *x2, const aclTensor *residual, const aclTensor *gamma);
bool ArnCheckShape(const aclTensor *x1, const aclTensor *x2, const aclTensor *residual);

#define DEPRECATED_MM_ARN_API_WARN_ONCE(oldApiName1, oldApiName2, deprecatedDate, newApiName1, newApiName2) \
    do { \
        static std::atomic<bool> isFirstWarn = true; \
        bool expected = true; \
        if (isFirstWarn.compare_exchange_strong(expected, false, std::memory_order_relaxed, \
                                                std::memory_order_relaxed)) { \
            OP_LOGW("%s and %s is scheduled to be deprecated in a post-%s version update. " \
                    "Please migrate to %s and %s. " \
                    "We apologize for any inconvenience caused and appreciate " \
                    "your timely migration to the other interface.", \
                    (oldApiName1), (oldApiName2), (deprecatedDate), (newApiName1), (newApiName2)); \
        } \
    } while (0)

aclnnStatus InnerMatmulAllReduceAddRmsNormGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *antiquantScale,
    const aclTensor *antiquantOffset, const aclTensor *dequant, const aclTensor *residual, const aclTensor *gamma,
    double epsilon, const char *group, const char *reduceOp, int64_t commTurn, int64_t antiquantGroupSize,
    const aclTensor *y, const aclTensor *normOut, uint64_t *workspaceSize, aclOpExecutor **executor);

#ifdef __cplusplus
}
#endif

#endif // OP_API_SRC_LEVEL2_MATMUL_ALL_REDUCE_ARN_UTIL_H_
