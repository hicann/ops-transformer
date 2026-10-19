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
 * \file moe_init_routing_quant_v2_arch35_tiling_def.h
 * \brief
 */

#ifndef MOE_INIT_ROUTING_QUANT_V2_ARCH35_TILING_DEF_H
#define MOE_INIT_ROUTING_QUANT_V2_ARCH35_TILING_DEF_H

struct MoeV2Arch35VBSComputeTilingData {
    int64_t needCoreNum{0};
    int64_t perCoreElements{0};
    int64_t perCoreLoops{0};
    int64_t perCorePerLoopElements{0};
    int64_t perCoreLastLoopElements{0};
    int64_t lastCoreElements{0};
    int64_t lastCoreLoops{0};
    int64_t lastCorePerLoopElements{0};
    int64_t lastCoreLastLoopElements{0};
    int64_t oneLoopMaxElements{0};
};

struct MoeV2Arch35VMSMiddleComputeTilingData {
    int64_t needCoreNum{0};
};

struct MoeV2Arch35SortOutComputeTilingData {
    int64_t oneLoopMaxElements{0};
};

struct MoeV2Arch35GatherOutComputeTilingData {
    int64_t needCoreNum{0};
    int64_t activateRows{0};
    int64_t perCoreRows{0};
    int64_t perCorePerLoopRows{0};
    int64_t perCoreLastLoopRows{0};
    int64_t lastCoreRows{0};
    int64_t lastCorePerLoopRows{0};
    int64_t lastCoreLastLoopRows{0};
    int64_t perCoreLoops{0};
    int64_t lastCoreLoops{0};
    int64_t perLoopCols{0};
    int64_t lastLoopCols{0};
    int64_t colLoops{0};
};

struct MoeInitRoutingQuantV2Arch35TilingData {
    int64_t coreNum{0};
    int64_t n{0};
    int64_t cols{0};
    int64_t k{0};
    int64_t expertCapacity{0};
    int64_t expertNum{0};
    int64_t dropPadMode{0};
    int64_t expertTokensCountOrCumsumFlag{0};
    int64_t expertTokensBeforeCapacityFlag{0};
    int64_t smoothType{0};
    int64_t histWithRegBase{0};
    int64_t tmpData{0};
    MoeV2Arch35VBSComputeTilingData vbsComputeParamsOp;
    MoeV2Arch35VMSMiddleComputeTilingData vmsMiddleComputeParamsOp;
    MoeV2Arch35SortOutComputeTilingData sortOutComputeParamsOp;
    MoeV2Arch35GatherOutComputeTilingData srcToDstComputeParamsOp;
    MoeV2Arch35GatherOutComputeTilingData srcToDstCapacityComputeParamsOp;
    MoeV2Arch35GatherOutComputeTilingData gatherOutComputeParamsOp;
};
#endif // MOE_INIT_ROUTING_QUANT_V2_ARCH35_TILING_DEF_H
