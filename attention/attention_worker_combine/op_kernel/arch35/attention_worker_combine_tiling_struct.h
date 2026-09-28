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
 * \file attention_worker_combine_tiling_struct.h
 * \brief Host/kernel ABI for Ascend950 AttentionWorkerCombine BS/K/H tiling.
 */

#ifndef ATTENTION_WORKER_COMBINE_TILING_STRUCT_H_
#define ATTENTION_WORKER_COMBINE_TILING_STRUCT_H_

namespace optiling {

struct AttentionWorkerCombineRegbaseTilingData {
    int64_t usedCoreNum{0}; // 实际使用的 AIV 核数

    // Logical dimensions and scheduling mode.
    int64_t BS{0};           // Number of output token rows R.
    int64_t K{0};            // Routed experts for nonquant; all K+1 rows, including shared expert, for MXFP.
    int64_t H{0};            // Valid hidden elements in each token row.
    int64_t needSchedule{0}; // Whether the kernel waits for and clears readiness flags.

    // BS partition: token rows assigned to AIV cores.
    int64_t BsSplitFactor{0};     // Reserved BS granularity; currently one token row.
    int64_t BsSplitCoreNum{0};    // Number of core groups along BS.
    int64_t mainCoreBsLoopNum{0}; // Token rows handled by each main BS core group.
    int64_t tailCoreBsLoopNum{0}; // Token rows handled by the final BS core group.

    // H partition, in logical hidden elements.
    int64_t HSplitFactor{0};     // Hidden elements processed by one H loop.
    int64_t HSplitTailFactor{0}; // Valid elements in the final H tile; zero means a full tile.
    int64_t HSplitCoreNum{0};    // Cores cooperating on one token along H.
    int64_t mainCoreHLoopNum{0}; // H loops handled by each main H core.
    int64_t tailCoreHLoopNum{0}; // H loops handled by the final H core.

    // K partition, in token/expert rows.
    int64_t KSplitFactor{0};     // Rows loaded in one K loop.
    int64_t KSplitTailFactor{0}; // Rows in the final K loop; zero means no tail loop.
    int64_t KSplitLoopNum{0};    // Number of full K loops before an optional tail.
};

} // namespace optiling

#endif // ATTENTION_WORKER_COMBINE_TILING_STRUCT_H_
