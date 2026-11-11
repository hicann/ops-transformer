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
 * \file matmul_tiling_data.h
 * \brief Serialized tiling data passed from host launcher to kernel.
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) MatmulTilingData {
    // Original problem shape.
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    uint32_t mL1{0};
    uint32_t nL1{0};
    uint32_t kL1{0};
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t baseK{0};

    uint32_t skSingleCoreK{0};
    // Tail split factors used in the final scheduling round.
    uint32_t mTailCnt{0};
    uint32_t nTailCnt{0};
    uint32_t mBaseTailSplitCnt{1};
    uint32_t nBaseTailSplitCnt{1};
    uint32_t mTailMain{0};
    uint32_t nTailMain{0};

    // Launch-time AIC count and buffering parameters.
    // Number of AICs launched for this kernel instance.
    uint32_t usedCoreNum{0};
    bool isHf32{0};
    uint8_t l1BufferNum{0};
    uint8_t l0cDB{1};
};
#pragma pack(pop)
