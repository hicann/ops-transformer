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
 * \file moe_gating_top_k_softmax_tiling_def.h
 * \brief tiling data struct
 */

#ifndef __MOE_GATING_TOP_K_SOFTMAX_TILING_DATA_H__
#define __MOE_GATING_TOP_K_SOFTMAX_TILING_DATA_H__

struct MoeGatingTopKSoftmaxRegbaseTilingData {
    int64_t rows;
    int64_t perCoreRowCount;
    int64_t lastCoreRowCount;
    int64_t perCoreLoopCount;
    int64_t lastCoreLoopCount;
    int64_t perCorePerLoopRowCount;
    int64_t perCoreLastLoopRowCount;
    int64_t lastCorePerLoopRowCount;
    int64_t lastCoreLastLoopRowCount;
    int64_t expertCount;
    int64_t k;
    int64_t expertCountAlign;
    int64_t kAlign;
    int64_t padNegInfCount;
    int64_t sortRepeatTimes;
};

#endif
