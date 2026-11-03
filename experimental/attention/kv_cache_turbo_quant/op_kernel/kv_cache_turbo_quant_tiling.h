/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KV_CACHE_TURBO_QUANT_TILING_H
#define KV_CACHE_TURBO_QUANT_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {
struct KvCacheTurboQuantTilingData {
    uint32_t vectors = 0;
    uint32_t blocks = 0;
    uint32_t mseBits = 0;
    uint32_t idxStride = 0;
    uint32_t qjlStride = 0;
    uint32_t headDim = 0;
    uint32_t qjlDim = 0;
    uint32_t mode = 0;
    uint32_t rotationDtype = 0;
    uint32_t qjlDtype = 0;
    uint32_t outputDtype = 0;
    float codebookScale = 1.0f;
    AscendC::tiling::TCubeTiling cubeTiling;
};
} // namespace optiling

#endif
