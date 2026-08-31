/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_EP_COMBINE_EPILOGUE_TILING_H
#define MOE_EP_COMBINE_EPILOGUE_TILING_H

#include "../../common/op_kernel/moe_ep_exception_dump_defs.h"

struct MoeEpCommonTilingData {
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t numExperts;
    uint32_t numLocalExperts;
    uint32_t numTokens;
    uint32_t hidden;
    uint32_t topK;
    uint32_t numMaxTokensPerRank;
    uint32_t scalesBytes;
    uint32_t perSlotBytes;
    uint32_t expertAlignment;
};

struct MoeEpCombineEpilogueInfo {
    MoeEpCommonTilingData cfg;
    MoeEpDumpMetadata dumpMetadata;
    uint32_t hasTopkWeights = 0;
    uint32_t aivNum = 0;
    uint64_t totalUbSize = 0;
    uint64_t combineStateWinOffset = 0;
    uint64_t combineDataWinOffset = 0;
};

struct MoeEpCombineEpilogueTilingData {
    MoeEpCombineEpilogueInfo moeEpCombineEpilogueInfo;
};

#endif // MOE_EP_COMBINE_EPILOGUE_TILING_H
