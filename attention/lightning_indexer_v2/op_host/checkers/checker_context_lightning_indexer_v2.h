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
 * \file checker_context_lightning_indexer_v2.h
 * \brief Declares the shared validation context for Lightning Indexer V2 checkers.
 */

#ifndef CHECKER_CONTEXT_LIGHTNING_INDEXER_V2_H
#define CHECKER_CONTEXT_LIGHTNING_INDEXER_V2_H

#include <cstdint>
#include <string>
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
namespace lightning_indexer_v2_checker {

constexpr int32_t QUANT_MODE_FP8 = 1;
constexpr int32_t QUANT_MODE_INT8 = 2;
constexpr int32_t QUANT_MODE_MXFP8 = 3;
constexpr int32_t QUANT_MODE_HIFLOAT8 = 4;
constexpr int32_t QUANT_MODE_MXFP4 = 5;
constexpr uint32_t MX_SCALE_SHAPE_ALIGN = 64;
constexpr uint32_t MX_E8M0_SCALE_PACK_NUM = 2; // MX的E8M0 scale形状最后一维打包数为2
constexpr uint32_t MXFP4_PACK_NUM = 2;         // 每个uint8承载2个FP4 E2M1逻辑元素

enum class CheckerLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2,
    INVALID = 3
};

struct CheckerTensor {
    const gert::CompileTimeTensorDesc *desc = nullptr;
    const gert::StorageShape *storageShape = nullptr;
    const gert::Tensor *tensor = nullptr;
    const gert::Stride *stride = nullptr;

    bool IsPresent() const
    {
        return storageShape != nullptr || tensor != nullptr;
    }

    const gert::Shape *GetShape() const
    {
        if (storageShape != nullptr) {
            return &storageShape->GetStorageShape();
        }
        return tensor == nullptr ? nullptr : &tensor->GetStorageShape();
    }
};

struct LightningIndexerV2CheckerInfo {
    const char *opName = nullptr;
    bool quantized = false;

    CheckerTensor query;
    CheckerTensor key;
    CheckerTensor weights;
    CheckerTensor queryDescale;
    CheckerTensor keyDescale;
    CheckerTensor cuSeqlensQ;
    CheckerTensor cuSeqlensK;
    CheckerTensor sequsedQ;
    CheckerTensor sequsedK;
    CheckerTensor cmpResidualK;
    CheckerTensor blockTable;
    CheckerTensor outputIdxOffset;
    CheckerTensor metadata;
    CheckerTensor sparseIndices;
    CheckerTensor sparseValues;

    CheckerLayout layoutQ = CheckerLayout::INVALID;
    CheckerLayout layoutK = CheckerLayout::INVALID;
    std::string layoutQText;
    std::string layoutKText;
    int64_t topk = 0;
    int64_t quantMode = 0;
    int64_t maxSeqlenQ = -1;
    int64_t maskMode = 0;
    int64_t cmpRatio = 1;
    int64_t returnValue = 0;

    int64_t batch = 0;
    int64_t qSeq = 0;
    int64_t kSeq = 0;
    int64_t qHeads = 0;
    int64_t kHeads = 0;
    int64_t headDim = 0;
    int64_t blockSize = 0;
};

} // namespace lightning_indexer_v2_checker
} // namespace optiling
#endif // CHECKER_CONTEXT_LIGHTNING_INDEXER_V2_H
