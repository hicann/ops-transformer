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
 * \file moe_token_permute_with_ep_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_PERMUTE_WITH_EP_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_TOKEN_PERMUTE_WITH_EP_H_

#include <cmath>
#include "moe_token_permute_tiling_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/tiling_api.h"
#include "util/shape_util.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexMixCopyComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, frontCoreNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreNum);
TILING_DATA_FIELD_DEF(int64_t, coreCalcNum);
TILING_DATA_FIELD_DEF(int64_t, coreCalcTail);
TILING_DATA_FIELD_DEF(int64_t, oneTokenBtypeSize);
TILING_DATA_FIELD_DEF(int64_t, oneProbBtypeSize);
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
TILING_DATA_FIELD_DEF(int64_t, start);
TILING_DATA_FIELD_DEF(int64_t, end);
TILING_DATA_FIELD_DEF(int64_t, tokenUB);
TILING_DATA_FIELD_DEF(int64_t, indicesUB);
TILING_DATA_FIELD_DEF(int64_t, probsUB);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IndexMixCopyComputeTilingDataOp, IndexMixCopyComputeTilingData)

BEGIN_TILING_DATA_DEF(MoeTokenPermuteWithEpTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, colsAlign);

TILING_DATA_FIELD_DEF(int64_t, topK);
TILING_DATA_FIELD_DEF_STRUCT(PermuteVBSComputeTilingEPData, vbsComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(PermuteVMSMiddleComputeTilingEPData, vmsMiddleComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(PermuteSortOutComputeTilingEPData, sortOutComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(IndexMixCopyComputeTilingData, indexCopyComputeParamsOp);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeTokenPermuteWithEp, MoeTokenPermuteWithEpTilingData)
struct MoeTokenPermuteWithEpCompileInfo {};
} // namespace optiling
#endif // MOE_TOKEN_PERMUTE
