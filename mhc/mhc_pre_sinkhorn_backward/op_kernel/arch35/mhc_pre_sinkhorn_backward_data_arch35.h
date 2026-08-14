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
 * \file mhc_pre_sinkhorn_backward_data_arch35.h
 * \brief tiling data struct
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_DATA_ARCH35_H
#define MHC_PRE_SINKHORN_BACKWARD_DATA_ARCH35_H

#include <cstdint>

struct MhcPreSinkhornBackwardArch35TilingData {
    int64_t batchSize = 0;
    int64_t seqLength = 0;
    int64_t c = 0;
    int64_t n = 0;
    int64_t c0 = 0;
    int64_t c1 = 0;
    int64_t cTail = 0;
    int64_t aivNum = 0;
    int64_t tileGradY = 0;
    int64_t tileHHat2 = 0;
    int64_t tileSize = 0;
    int64_t skIterCount = 0;
    int64_t ubSize = 0;
    float eps = 0;
    TCubeTiling mm1TilingData;
    TCubeTiling mm2TilingData;
};

struct MhcPreSinkhornBackwardArch35DeterminiticTilingData {
    int64_t bs = 0;
    int64_t c = 0;
    int64_t n = 0;
    int64_t usedAivNum = 0;
    int64_t aicNum = 0;
    int64_t cLoopDataLen = 0;
    int64_t bsLoopDataLen = 0;
    int64_t skIterCount = 0;
    int64_t bsTaskCount = 0;
    int64_t tailBsTaskCount = 0;
    // reset后
    int64_t cBlockLoops = 0;         // B*S*N*C，切核，总循环次数
    int64_t cBlockTailLoops = 0;     // B*S*N*C，切核，尾核循环次数
    int64_t cBlockCount = 0;         // B*S*N*C，切核，普通核尾循环处理元素个数
    int64_t cBlockTailCount = 0;     // B*S*N*C，切核，普通核尾核循环次数
    int64_t cBlockTailTailCount = 0; // B*S*N*C，切核，尾核尾循环处理个数
    int64_t finalUsedAivNum = 0;
    float eps = 0;
    TCubeTiling mm1TilingData;
    TCubeTiling mm2TilingData;
};

#endif // MHC_PRE_SINKHORN_BACKWARD_DATA_ARCH35_H
