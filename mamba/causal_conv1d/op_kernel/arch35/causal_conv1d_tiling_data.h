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
 * \file causal_conv1d_tiling_data.h
 * \brief Tiling data struct — shape, flags, and partitioning parameters from host to kernel.
 */

#ifndef CAUSAL_CONV1D_TILING_DATA_H_
#define CAUSAL_CONV1D_TILING_DATA_H_

#include <cstdint>

struct CausalConv1dTilingData {
    int64_t dim;
    int64_t cuSeqlen;
    int64_t seqLen;
    int64_t batch;
    int64_t kernelWidth;
    int64_t stateLen;
    int64_t numCacheLines;

    int64_t activationMode;
    int64_t nullBlockId;

    int32_t baseDim;
    int64_t tokensPerBlock;
    int64_t tokenBlockCnt;

    bool hasNumAcceptedTokens;
    bool hasCacheIndices;
    bool hasInitialStateMode;
    bool hasInitStateWorkspace;

    int32_t coBatch;
};
#endif // CAUSAL_CONV1D_TILING_DATA_H_
