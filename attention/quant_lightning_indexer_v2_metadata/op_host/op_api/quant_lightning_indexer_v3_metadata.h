/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef L0_QUANT_LIGHTNING_INDEXER_V3_METADATA_H
#define L0_QUANT_LIGHTNING_INDEXER_V3_METADATA_H

#include <string>
#include "opdev/op_executor.h"

namespace l0op {
// candidate 配套 metadata：在 V2 参数基础上多传 candidateTopkBlocks / candidateBlockSize 两个属性。
// AICPU 内核按属性名取值，未知属性会被忽略，因此两者不参与分核计算，metadata 输出与 V2 一致。
const aclTensor* QuantLightningIndexerV3Metadata(
    const aclTensor* cuSeqlensQOptional, const aclTensor* cuSeqlensKOptional, const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKOptional, const aclTensor* cmpResidualKOptional,
    const aclTensor* causalSeqlensKvGlobalOptional, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk,
    int64_t quantMode, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK, const std::string& layoutQOptional,
    const std::string& layoutKOptional, int64_t maskMode, int64_t cmpRatio, int64_t candidateTopkBlocks,
    int64_t candidateBlockSize, int64_t aicCoreNum, int64_t aivCoreNum, const std::string& socVersion,
    const aclIntArray* dcpParams, const aclTensor* metadata, aclOpExecutor* executor);
} // namespace l0op

#endif
