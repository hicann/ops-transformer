/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_SCATTER_PA_KV_CACHE_H_
#define ACLNN_SCATTER_PA_KV_CACHE_H_
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnScatterPaKvCacheGetWorkspaceSize(const aclTensor* key, aclTensor* keyCacheRef,
                                                  const aclTensor* slotMapping, const aclTensor* value,
                                                  aclTensor* valueCacheRef, const aclTensor* compressLensOptional,
                                                  const aclTensor* compressSeqOffsetOptional,
                                                  const aclTensor* seqLensOptional, char* cacheModeOptional,
                                                  char* scatterModeOptional, const aclIntArray* stridesOptional,
                                                  const aclIntArray* offsetsOptional, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);

aclnnStatus aclnnScatterPaKvCache(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
