/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_fused_topk_arch35_tiling_def.h
 * \brief Ascend950 tiling data for MoeFusedTopk.
 */

#ifndef ASCENDC_MOE_FUSED_TOPK_ARCH35_TILING_DEF_H_
#define ASCENDC_MOE_FUSED_TOPK_ARCH35_TILING_DEF_H_

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct MoeFusedTopkArch35TilingData {
    uint32_t firstDimSize{0};
    uint32_t secondDimSize{0};
    uint32_t addNumDimSize{0};
    uint32_t groupNum{0};
    uint32_t groupTopk{0};
    uint32_t topN{0};
    uint32_t topK{0};

    uint32_t activateType{0};
    uint32_t isNorm{0};
    float scale{1.0f};
    uint32_t groupEles{0};
    uint32_t blockNum{0};
    uint32_t ubFactorElement{0};
    uint32_t batchPerCore{0};
    uint32_t tailBatch{0};

    uint32_t expertNum{0};
    uint32_t tableDim{0};
    uint32_t topkMaxValue{0};
    uint32_t topkMinValue{0};
    uint32_t reserved{0};
    uint64_t workspacePerCore{0};
    TopkTiling topkTilingData;
};

#endif // ASCENDC_MOE_FUSED_TOPK_ARCH35_TILING_DEF_H_
