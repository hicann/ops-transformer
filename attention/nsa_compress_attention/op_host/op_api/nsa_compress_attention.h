/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_OP_NSA_COMPRESS_ATTENTION_OP_H_
#define OP_API_INC_LEVEL0_OP_NSA_COMPRESS_ATTENTION_OP_H_

#include "opdev/op_executor.h"

namespace l0op {

const std::array<const aclTensor*, 4> NsaCompressAttention(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* attenMaskOptional,
    const aclIntArray* actualSeqQLenOptional, const aclIntArray* actualCmpSeqKvLenOptional,
    const aclIntArray* actualSelSeqKvLenOptional, const aclTensor* topkMaskOptional, double scaleValueOptional,
    int64_t headNum, const char* inputLayout, int64_t sparseModeOptional, int64_t compressBlockSize,
    int64_t compressStride, int64_t selectBlockSize, int64_t selectBlockCount, aclOpExecutor* executor);
}

#endif // OP_API_INC_LEVEL0_OP_NSA_COMPRESS_ATTENTION_OP_H_
