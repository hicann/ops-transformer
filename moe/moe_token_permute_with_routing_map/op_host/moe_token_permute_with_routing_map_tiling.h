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
 * \file moe_token_permute_with_routing_map_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_H
#include <cmath>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(PermuteVBSComputeRMTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopElements);
TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreWSindex);
TILING_DATA_FIELD_DEF(int64_t, frontcoreTask);
TILING_DATA_FIELD_DEF(int64_t, tailcoreTask);
TILING_DATA_FIELD_DEF(int64_t, frontCoreNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PermuteVBSComputeRMTilingDataOp, PermuteVBSComputeRMTilingData)

BEGIN_TILING_DATA_DEF(PermuteVMSMiddleComputeRMTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PermuteVMSMiddleComputeRMTilingDataOp, PermuteVMSMiddleComputeRMTilingData)

BEGIN_TILING_DATA_DEF(PermuteSortOutComputeRMTilingData)
TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PermuteSortOutComputeRMTilingDataOp, PermuteSortOutComputeRMTilingData)

BEGIN_TILING_DATA_DEF(IndexCopyComputeRMTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, frontCoreNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreNum);
TILING_DATA_FIELD_DEF(int64_t, coreCalcNum);
TILING_DATA_FIELD_DEF(int64_t, coreCalcTail);
TILING_DATA_FIELD_DEF(int64_t, oneTokenBtypeSize);
TILING_DATA_FIELD_DEF(int64_t, onceIndicesTokenMoveTimes);
TILING_DATA_FIELD_DEF(int64_t, onceUbTokenNums);
TILING_DATA_FIELD_DEF(int64_t, onceIndicesTokenNums);
TILING_DATA_FIELD_DEF(int64_t, onceIndices);
TILING_DATA_FIELD_DEF(int64_t, oneTokenlastMove);
TILING_DATA_FIELD_DEF(int64_t, oneTokenOnceMove);
TILING_DATA_FIELD_DEF(int64_t, oneTokenMoveTimes);
TILING_DATA_FIELD_DEF(int64_t, frontCoreLoop);
TILING_DATA_FIELD_DEF(int64_t, frontCoreLastTokenNums);
TILING_DATA_FIELD_DEF(int64_t, tailCoreLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreLastTokenNums);
TILING_DATA_FIELD_DEF(int64_t, tailLastonceIndicesTokenMoveTimes);
TILING_DATA_FIELD_DEF(int64_t, tailLastIndicesLastTokenNums);
TILING_DATA_FIELD_DEF(int64_t, frontLastonceIndicesTokenMoveTimes);
TILING_DATA_FIELD_DEF(int64_t, frontLastIndicesLastTokenNums);
TILING_DATA_FIELD_DEF(int64_t, numOutTokens);
TILING_DATA_FIELD_DEF(int64_t, tokenUB);
TILING_DATA_FIELD_DEF(int64_t, indicesUB);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IndexCopyComputeRMTilingDataOp, IndexCopyComputeRMTilingData)

BEGIN_TILING_DATA_DEF(MaskedSelectRMTilingData)
TILING_DATA_FIELD_DEF(uint64_t, formerNum);
TILING_DATA_FIELD_DEF(uint64_t, formerLength);
TILING_DATA_FIELD_DEF(uint64_t, formertileNum);
TILING_DATA_FIELD_DEF(uint64_t, formertileLength);
TILING_DATA_FIELD_DEF(uint64_t, formerlasttileLength);
TILING_DATA_FIELD_DEF(uint64_t, tailNum);
TILING_DATA_FIELD_DEF(uint64_t, tailLength);
TILING_DATA_FIELD_DEF(uint64_t, tailtileNum);
TILING_DATA_FIELD_DEF(uint64_t, tailtileLength);
TILING_DATA_FIELD_DEF(uint64_t, taillasttileLength);
TILING_DATA_FIELD_DEF(uint64_t, tokenNum);
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MaskedSelectRMTilingDataOp, MaskedSelectRMTilingData)
BEGIN_TILING_DATA_DEF(MoeTokenPermuteWithRoutingMapTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, colsAlign);
TILING_DATA_FIELD_DEF(int64_t, topK);
TILING_DATA_FIELD_DEF(int64_t, expertNum);
TILING_DATA_FIELD_DEF(int64_t, capacity);
TILING_DATA_FIELD_DEF(int64_t, hasProb);
TILING_DATA_FIELD_DEF_STRUCT(PermuteVBSComputeRMTilingData, vbsComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(PermuteVMSMiddleComputeRMTilingData, vmsMiddleComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(PermuteSortOutComputeRMTilingData, sortOutComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(IndexCopyComputeRMTilingData, indexCopyComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(MaskedSelectRMTilingData, maskedSelectParamsOp);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeTokenPermuteWithRoutingMap, MoeTokenPermuteWithRoutingMapTilingData)
struct MoeTokenPermuteWithRoutingMapCompileInfo {
    uint64_t aivNum = 0;
    uint64_t ubSize = 0;
    uint64_t workSpaceSize = 0;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_H
