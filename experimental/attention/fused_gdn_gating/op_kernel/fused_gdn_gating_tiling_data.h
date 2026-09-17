/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_gdn_gating_tiling_data.h
 * \brief Tiling data shared between host-side tiling and device-side kernel.
 */

#ifndef FUSED_GDN_GATING_TILING_DATA_H
#define FUSED_GDN_GATING_TILING_DATA_H

#include <cstdint>

namespace FusedGdnGating {

struct FusedGdnGatingTilingData {
    uint32_t numHeads = 0;
    float beta = 1.0f;

    // --------------------------------------------------------
    // 2. 910/910B 架构专用参数
    // --------------------------------------------------------
    uint32_t numBatches = 0;
    uint32_t rowsPerIter = 0;
    uint32_t useBulkDma = 0;

    // --------------------------------------------------------
    // 3. 310P 架构专用参数
    // --------------------------------------------------------
    uint32_t usedCoreNum = 0;
    uint32_t alignedLength = 0;
    uint32_t tailLength = 0;
    uint32_t tileRows = 0;
    float inv_beta = 1.0f;

    float threshold = 20.0f;
};

} // namespace FusedGdnGating

#endif // FUSED_GDN_GATING_TILING_DATA_H
