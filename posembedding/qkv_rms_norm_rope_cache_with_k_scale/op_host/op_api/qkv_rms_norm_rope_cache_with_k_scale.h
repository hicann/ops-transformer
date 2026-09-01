/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_OP_QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_OP_H
#define OP_API_INC_LEVEL0_OP_QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_OP_H

#include <tuple>
#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"

namespace l0op {
using QkvRmsNormRopeCacheWithKScaleResult = std::tuple<aclnnStatus, const aclTensor *, const aclTensor *,
                                                       const aclTensor *, const aclTensor *, const aclTensor *>;

QkvRmsNormRopeCacheWithKScaleResult QkvRmsNormRopeCacheWithKScale(
    const aclTensor *qkv, const aclTensor *qGamma, const aclTensor *kGamma, const aclTensor *cosSin,
    const aclTensor *slotMapping, aclTensor *kCache, aclTensor *vCache, aclTensor *kScaleCache,
    const aclTensor *queryStartLoc, const aclTensor *seqLens, const aclTensor *rotationOptional,
    const aclTensor *vScaleOptional, const aclTensor *mropePositionOptional, const aclIntArray *headNums,
    const char *layoutQkv, const char *layoutQOut, float epsilon, const aclIntArray *mropeSectionOptional,
    const char *qQuantMode, const char *kQuantMode, int64_t qOutDtype, aclTensor *qOut, aclTensor *qScaleOptional,
    aclOpExecutor *executor);
} // namespace l0op

#endif
