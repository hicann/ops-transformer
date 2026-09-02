/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RAIN_FUSION_ATTENTION_H_
#define RAIN_FUSION_ATTENTION_H_

#include <array>
#include "opdev/op_executor.h"

namespace l0op {

const std::array<const aclTensor *, 2> RainFusionAttention(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *selectIdx,
    const aclTensor *selectNumIdx, const aclIntArray *blockShape, const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *blockTableOptional, const char *qInputLayout, const char *kvInputLayout, int64_t numKeyValueHeads,
    int64_t maskType, double scaleValue, int64_t innerPrecise, int64_t blockSize, aclOpExecutor *executor);

} // namespace l0op

#endif // RAIN_FUSION_ATTENTION_H_
