/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file l0_sparse_flash_mla_softmax_l1_norm_metadata.h
 * \brief
 */
#ifndef L0_SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_METADATA_H
#define L0_SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_METADATA_H

#include <cstdint>
#include <string>
#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *SparseFlashMlaSoftmaxL1NormMetadata(
    const aclTensor *cuSeqLensQOptional, const aclTensor *cuSeqLensKOptional, const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKOptional, const aclTensor *cmpResidualKOptional, const aclTensor *topkLengthOptional,
    int64_t batchSize, int64_t maxSeqLenQ, int64_t maxSeqLenK, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim,
    int64_t topk, int64_t cmpRatio, int64_t maskMode, char *layoutQ, char *layoutK, const char *socVersion,
    int64_t aicCoreNum, const aclTensor *metadata, aclOpExecutor *executor);
} // namespace l0op

#endif // L0_SPARSE_FLASH_MLA_SOFTMAX_L1_NORM_METADATA_H
