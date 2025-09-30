/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v3_tiling_cache.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_V3_TILING_CACHE_H
#define QUANT_BATCH_MATMUL_V3_TILING_CACHE_H

#include "quant_batch_matmul_v3_basic_tiling.h"
#include "common/op_host/op_tiling/tiling_cache.h"

namespace optiling {
struct QuantBatchMatmulV3BitField {
    // 这里要保证是32bit
    uint16_t transA : 1;
    uint16_t transB : 1;
    uint16_t hasBias : 1;
    uint16_t aFormatNd : 1;
    uint16_t bFormatNd : 1;
    uint16_t cFormatNd : 1;
    uint16_t isPerTensor : 1;
    uint16_t isPertoken : 1;
    uint32_t reserved : 24;
};

class QuantBatchMatmulV3HashInput {
public:
    explicit QuantBatchMatmulV3HashInput(const QuantBatchMatmulInfo &params, const Ops::Transformer::OpTiling::AiCoreParams &aicoreParams);
    ~QuantBatchMatmulV3HashInput() = default;
    bool operator==(const QuantBatchMatmulV3HashInput &params) const
    {
        return memcmp(this, &params, sizeof(params)) == 0;
    }
    std::string ToString() const;

private:
    uint64_t mSize = 0;
    uint64_t mSizePerNpu = 0;
    uint64_t kSize = 0;
    uint64_t nSize = 0;
    uint64_t batchA1 = 0;
    uint64_t batchA2 = 0;
    uint64_t batchA3 = 0;
    uint64_t batchA4 = 0;
    uint64_t batchB1 = 0;
    uint64_t batchB2 = 0;
    uint64_t batchB3 = 0;
    uint64_t batchB4 = 0;
    uint64_t batchBias = 0;
    int32_t aDtype = 0;
    int32_t bDtype = 0;
    int32_t cDtype = 0;
    int32_t biasDtype = 0;
    int32_t scaleDtype = 0;
    int32_t outDtype = 0;
    int32_t aicNum = 0;
    int32_t reserved = 0;
    QuantBatchMatmulV3BitField bitField = {0, 0, 0, 0, 0, 0, 0, 0, 0};
};

class QuantBatchMatmulV3HashItem {
public:
    explicit QuantBatchMatmulV3HashItem(const QuantBatchMatmulInfo &params, const Ops::Transformer::OpTiling::AiCoreParams &aicoreParams)
        : hashKey_(params, aicoreParams)
    {
    }
    const QuantBatchMatmulV3HashInput &input() const { return hashKey_; }
    const BasicTiling &GetTiling() const { return tiling_; }
    void SetTiling(const BasicTiling &tiling) { tiling_ = tiling; }

private:
    QuantBatchMatmulV3HashInput hashKey_;
    BasicTiling tiling_;
};

using MMBasicTilingHash = Ops::Transformer::TilingCache<QuantBatchMatmulV3HashInput, QuantBatchMatmulV3HashItem>;
}  // namespace optiling
#endif  // QUANT_BATCH_MATMUL_V3_TILING_CACHE_H