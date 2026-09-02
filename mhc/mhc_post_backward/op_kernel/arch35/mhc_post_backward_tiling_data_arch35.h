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
 * \file mhc_post_backward_tiling_data.h
 * \brief mhc_post_backward
 */

#ifndef MHC_POST_BACKWARD_TILING_DATA_ARCH35_H
#define MHC_POST_BACKWARD_TILING_DATA_ARCH35_H

class MhcPostBackwardTilingDataArch35 {
public:
    uint32_t totalItems{0};
    uint32_t itemsPerCore{0};
    uint32_t remainderItems{0};
    uint32_t usedCores{0};
    uint32_t S{0};
    uint32_t n{0};
    uint32_t D{0};
    uint32_t tileD{0};
    uint32_t nTilesD{0};
    uint32_t alignedD{0};
    uint32_t lastTileD{0};
    uint32_t alignedN{0};  // n aligned to 8 for float32 vector ops
    uint32_t alignedNN{0}; // n*n aligned to 8 for float32 vector ops
    uint32_t itemsPerAic{0};
    uint32_t remainderItemsAic{0};
    TCubeTiling matmulTiling{};
};

#endif // MHC_POST_BACKWARD_TILING_DATA_ARCH35_H
