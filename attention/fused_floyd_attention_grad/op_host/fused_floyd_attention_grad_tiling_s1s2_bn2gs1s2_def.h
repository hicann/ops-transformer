/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

#include <vector>
#include <register/op_impl_registry.h>

namespace optiling {
namespace FFAG {
/////////////////////////////////////////////////////////////////////////
// S1S2_BNGS1S2
/////////////////////////////////////////////////////////////////////////
BEGIN_TILING_DATA_DEF(FusedFloydAttentionGradS1S2BNGS1S2BaseParams)
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, n2);
TILING_DATA_FIELD_DEF(int64_t, g);
TILING_DATA_FIELD_DEF(int64_t, s1);
TILING_DATA_FIELD_DEF(int64_t, s2);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(float, keepProb);
TILING_DATA_FIELD_DEF(int64_t, s1Token); // pre_tokens
TILING_DATA_FIELD_DEF(int64_t, s2Token); // next_tokens
TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
TILING_DATA_FIELD_DEF(uint32_t, isSparse);
TILING_DATA_FIELD_DEF(int64_t, attenMaskS2Size);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskCompressMode);
TILING_DATA_FIELD_DEF(int64_t, qStartIdx);
TILING_DATA_FIELD_DEF(int64_t, kvStartIdx);
TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS1);
TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS2);
TILING_DATA_FIELD_DEF(uint32_t, pseType);
TILING_DATA_FIELD_DEF(uint32_t, pseOptional);
TILING_DATA_FIELD_DEF(uint32_t, pseShapeType);
TILING_DATA_FIELD_DEF(uint32_t, pseDtype);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskOptional);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskDtype);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskShapeType);
TILING_DATA_FIELD_DEF(uint32_t, pad);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FusedFloydAttentionGradS1S2BNGS1S2BaseParamsOp, FusedFloydAttentionGradS1S2BNGS1S2BaseParams)

BEGIN_TILING_DATA_DEF(FusedFloydAttentionGradS1S2BNGS1S2SplitCoreParams)
TILING_DATA_FIELD_DEF(int64_t, s1Outer);
TILING_DATA_FIELD_DEF(uint32_t, s1CvRatio);
TILING_DATA_FIELD_DEF(uint32_t, s1Inner);
TILING_DATA_FIELD_DEF(uint32_t, s1CvInner);
TILING_DATA_FIELD_DEF(uint32_t, s1Tail);
TILING_DATA_FIELD_DEF(uint32_t, s1CvTail);
TILING_DATA_FIELD_DEF(int64_t, s2Outer);
TILING_DATA_FIELD_DEF(uint32_t, s2CvRatio);
TILING_DATA_FIELD_DEF(uint32_t, s2Inner);
TILING_DATA_FIELD_DEF(uint32_t, s2Tail);
TILING_DATA_FIELD_DEF(uint32_t, baseMN);
TILING_DATA_FIELD_DEF(uint32_t, sfmgdOuter);
TILING_DATA_FIELD_DEF(uint32_t, sfmgdFactor);
TILING_DATA_FIELD_DEF(uint32_t, sfmgdTail);
TILING_DATA_FIELD_DEF(uint32_t, blockOuter);
TILING_DATA_FIELD_DEF(int64_t, bandIdx);
TILING_DATA_FIELD_DEF(uint32_t, gTail);
TILING_DATA_FIELD_DEF(uint32_t, gInner);
TILING_DATA_FIELD_DEF(int64_t, gOuter);
TILING_DATA_FIELD_DEF(int64_t, bmmS1base);
END_TILING_DATA_DEF;
// 固定写法不能换行，会失败
REGISTER_TILING_DATA_CLASS(FusedFloydAttentionGradS1S2BNGS1S2SplitCoreParamsOp,
                           FusedFloydAttentionGradS1S2BNGS1S2SplitCoreParams)

BEGIN_TILING_DATA_DEF(BlockNumListParams)
TILING_DATA_FIELD_DEF_ARR(int64_t, 50, blockStarts);
TILING_DATA_FIELD_DEF_ARR(int64_t, 50, blockEnds);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BlockNumListParamsOp, BlockNumListParams)

BEGIN_TILING_DATA_DEF(FusedFloydAttentionGradTilingDataS1s2Bn2gs1s2)
TILING_DATA_FIELD_DEF_STRUCT(FusedFloydAttentionGradS1S2BNGS1S2BaseParams, s1s2BNGS1S2BaseParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedFloydAttentionGradS1S2BNGS1S2SplitCoreParams, s1s2BNGS1S2SplitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm3TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm3TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
TILING_DATA_FIELD_DEF_STRUCT(BlockNumListParams, s1s2BNGS1S2BlockNumList);
TILING_DATA_FIELD_DEF_STRUCT(FFAGPreParams, preTilingData);
TILING_DATA_FIELD_DEF_STRUCT(FFAGPostParams, postTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedFloydAttentionGrad, FusedFloydAttentionGradTilingDataS1s2Bn2gs1s2)
} // namespace FFAG
} // namespace optiling
