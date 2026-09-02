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
 * \file moe_token_permute_tiling_base.h
 * \brief
 */
#ifndef MOE_TOKEN_PERMUTE_WITH_EP_MOE_TOKEN_PERMUTE_H_
#define MOE_TOKEN_PERMUTE_WITH_EP_MOE_TOKEN_PERMUTE_H_
#include <cmath>
#include "tiling/tiling_api.h"
#include "register/op_impl_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PermuteVBSComputeTilingEPData)
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
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PermuteVBSComputeTilingEPDataOp, PermuteVBSComputeTilingEPData)

BEGIN_TILING_DATA_DEF(PermuteVMSMiddleComputeTilingEPData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PermuteVMSMiddleComputeTilingEPDataOp, PermuteVMSMiddleComputeTilingEPData)

BEGIN_TILING_DATA_DEF(PermuteSortOutComputeTilingEPData)
TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PermuteSortOutComputeTilingEPDataOp, PermuteSortOutComputeTilingEPData)

BEGIN_TILING_DATA_DEF(IndexCopyComputeTilingEPData)
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
REGISTER_TILING_DATA_CLASS(IndexCopyComputeTilingEPDataOp, IndexCopyComputeTilingEPData)
} // namespace optiling
#endif // MOE_TOKEN_PERMUTE_WITH_EP_MOE_TOKEN_PERMUTE_H_
