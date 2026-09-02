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
 * \file rope_with_sin_cos_cache.h
 * \brief
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_ROPE_WITH_SIN_COS_CACHE_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_ROPE_WITH_SIN_COS_CACHE_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const std::tuple<aclTensor *, aclTensor *> RopeWithSinCosCache(const aclTensor *positions, const aclTensor *queryIn,
                                                               const aclTensor *keyIn, const aclTensor *cosSinCache,
                                                               const aclIntArray *mropeSection, int64_t headSize,
                                                               bool isNeoxStyle, int64_t qStride, int64_t kStride,
                                                               int64_t numQHeads, int64_t numKHeads, int64_t cacheMode,
                                                               aclOpExecutor *executor);
}
#endif // OP_API_INC_LEVEL0_OP_ROPE_WITH_SIN_COS_CACHE_OP_H_
