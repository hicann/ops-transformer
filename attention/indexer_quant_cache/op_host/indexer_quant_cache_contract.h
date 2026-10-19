/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDEXER_QUANT_CACHE_CONTRACT_H
#define INDEXER_QUANT_CACHE_CONTRACT_H

#include <cstdint>
#include "graph/types.h"

namespace indexer_quant_cache {
constexpr int64_t MXFP8_QUANT_MODE = 0;
constexpr int64_t NORMAL_QUANT_MODE = 1;
constexpr int64_t HIFLOAT_QUANT_MODE = 2;
constexpr int64_t MXFP4_QUANT_MODE = 3;
constexpr int64_t HIFP4_QUANT_MODE = 4;

inline bool IsValidQuantTypes(int64_t mode, ge::DataType cache, ge::DataType scale, ge::DataType x, ge::DataType slots)
{
    if ((x != ge::DT_FLOAT16 && x != ge::DT_BF16) || slots != ge::DT_INT32) {
        return false;
    }
    const bool fp8 = cache == ge::DT_FLOAT8_E4M3FN || cache == ge::DT_FLOAT8_E5M2;
    switch (mode) {
        case MXFP8_QUANT_MODE:
            return fp8 && scale == ge::DT_FLOAT8_E8M0;
        case NORMAL_QUANT_MODE:
            return (fp8 || cache == ge::DT_UINT8) && scale == ge::DT_FLOAT;
        case HIFLOAT_QUANT_MODE:
            // FP8 and UINT8 are existing one-byte carriers for HiFloat8 encoded output.
            return (fp8 || cache == ge::DT_UINT8) && scale == ge::DT_FLOAT;
        case MXFP4_QUANT_MODE:
            return (cache == ge::DT_FLOAT4_E2M1 || cache == ge::DT_FLOAT4_E1M2) && scale == ge::DT_FLOAT8_E8M0;
        case HIFP4_QUANT_MODE:
            return cache == ge::DT_HIFLOAT4 && scale == ge::DT_HIFLOAT4_SCALE;
        default:
            return false;
    }
}
} // namespace indexer_quant_cache
#endif
