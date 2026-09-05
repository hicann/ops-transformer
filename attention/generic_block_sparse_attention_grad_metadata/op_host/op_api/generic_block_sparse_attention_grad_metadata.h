/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef L0_GENERIC_SPARSE_ATTENTION_GRAD_METADATA_H
#define L0_GENERIC_SPARSE_ATTENTION_GRAD_METADATA_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *GenericBlockSparseAttentionGradMetadata(
    const aclTensor *rsvdBlockIdx, const aclTensor *rsvdBlockCount, const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t numQHeads, int64_t numKvHeads, int64_t headDim,
    int64_t blockShapeX, int64_t blockShapeY, int64_t isPackedGQA, const char *layoutQOptional,
    const char *layoutKvOptional, int64_t maskType, int64_t softmaxPrecision, int64_t winLeft, int64_t winRight,
    const char *socVersion, int64_t aicCoreNum, int64_t aivCoreNum, const aclTensor *metaData, aclOpExecutor *executor);
} // namespace l0op

#endif
