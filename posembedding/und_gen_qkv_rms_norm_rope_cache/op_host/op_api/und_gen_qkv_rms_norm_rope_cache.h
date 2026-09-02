/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_UND_GEN_QKV_RMS_NORM_ROPE_CACHE_H_
#define OP_API_INC_LEVEL0_UND_GEN_QKV_RMS_NORM_ROPE_CACHE_H_

#include "opdev/op_executor.h"

namespace l0op {
aclnnStatus UndGenQkvRmsNormRopeCache(const aclTensor *undQkv, const aclTensor *undWeightsQ,
                                      const aclTensor *undWeightsK, const aclTensor *cosSinCache, aclTensor *kCacheRef,
                                      aclTensor *vCacheRef, const aclTensor *slotMapping, const aclTensor *positions,
                                      const aclTensor *genQkv, const aclTensor *genWeightsQ,
                                      const aclTensor *genWeightsK, const aclTensor *catIndices, int64_t numHeadsQ,
                                      int64_t numHeadsK, int64_t numHeadsV, double normEps,
                                      const aclIntArray *mropeSection, aclTensor *qOut, aclOpExecutor *executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_UND_GEN_QKV_RMS_NORM_ROPE_CACHE_H_
