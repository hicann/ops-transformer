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
 * \file sparse_flash_attention_grad_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "sparse_flash_attention_grad_tiling_common.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SparseFlashAttentionGradSplitCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, s2OuterOuter);
TILING_DATA_FIELD_DEF(uint32_t, singleM);
TILING_DATA_FIELD_DEF(uint32_t, singleN);
TILING_DATA_FIELD_DEF(uint32_t, sftBaseM);
TILING_DATA_FIELD_DEF(uint32_t, sftBaseN);
TILING_DATA_FIELD_DEF(uint32_t, rsv);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGradSplitCoreParamsOp, SparseFlashAttentionGradSplitCoreParams)

BEGIN_TILING_DATA_DEF(SfagPostParams)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, postUbBaseSize);
TILING_DATA_FIELD_DEF(uint32_t, postMaxDataSize);
TILING_DATA_FIELD_DEF(uint32_t, nzReservedSize);
TILING_DATA_FIELD_DEF(uint32_t, qPostBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, qPostBlockTotal);
TILING_DATA_FIELD_DEF(uint64_t, qPostBaseNum);
TILING_DATA_FIELD_DEF(uint64_t, qPostTailNum);
TILING_DATA_FIELD_DEF(uint64_t, kPostBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, kPostBlockTotal);
TILING_DATA_FIELD_DEF(uint64_t, kPostBaseNum);
TILING_DATA_FIELD_DEF(uint64_t, kPostTailNum);
TILING_DATA_FIELD_DEF(uint64_t, vPostBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, vPostBlockTotal);
TILING_DATA_FIELD_DEF(uint64_t, vPostBaseNum);
TILING_DATA_FIELD_DEF(uint64_t, vPostTailNum);
TILING_DATA_FIELD_DEF(uint64_t, qSizeAlign);
TILING_DATA_FIELD_DEF(uint64_t, kSizeAlign);
TILING_DATA_FIELD_DEF(uint64_t, vSizeAlign);
TILING_DATA_FIELD_DEF(uint64_t, dqWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dkWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dvWorkSpaceOffset);
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, n2);
TILING_DATA_FIELD_DEF(int64_t, g);
TILING_DATA_FIELD_DEF(int64_t, s1);
TILING_DATA_FIELD_DEF(int64_t, s2);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, d2);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SfagPostParamsOp, SfagPostParams)

BEGIN_TILING_DATA_DEF(SparseFlashAttentionGradEmptyTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashAttentionGradSplitCoreParams, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(SfagPostParams, postTilingData);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(SparseFlashAttentionGradBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreProcessNNum);
TILING_DATA_FIELD_DEF(uint32_t, remainCoreProcessNNum);
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, N2);
TILING_DATA_FIELD_DEF(int64_t, S1);
TILING_DATA_FIELD_DEF(int64_t, S2);
TILING_DATA_FIELD_DEF(int64_t, G);
TILING_DATA_FIELD_DEF(int64_t, D);  // qk_dim
TILING_DATA_FIELD_DEF(int64_t, D2); // v_dim
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, layout);
TILING_DATA_FIELD_DEF(uint32_t, selectedKWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, selectedVWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, scatterDkWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, scatterDvWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm1WorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm2WorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm4InputWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm3InputWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, castUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, dqWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dkWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dvWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockCount);
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGradBaseParamsOp, SparseFlashAttentionGradBaseParams)

//======================== basic template tiling data def start ========================//
BEGIN_TILING_DATA_DEF(SparseFlashAttentionGradBaseBasicParams)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreProcessNNum);
TILING_DATA_FIELD_DEF(uint32_t, remainCoreProcessNNum);
TILING_DATA_FIELD_DEF(int64_t, B);
TILING_DATA_FIELD_DEF(int64_t, N2);
TILING_DATA_FIELD_DEF(int64_t, S1);
TILING_DATA_FIELD_DEF(int64_t, S2);
TILING_DATA_FIELD_DEF(int64_t, G);
TILING_DATA_FIELD_DEF(int64_t, D);     // qk_dim
TILING_DATA_FIELD_DEF(int64_t, D2);    // v_dim
TILING_DATA_FIELD_DEF(int64_t, ropeD); // rope_dim
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, layout);
TILING_DATA_FIELD_DEF(uint64_t, selectedKWorkspaceLen);
TILING_DATA_FIELD_DEF(uint64_t, selectedVWorkspaceLen);
TILING_DATA_FIELD_DEF(uint64_t, mm12WorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, castUsedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, kvMerge);
TILING_DATA_FIELD_DEF(int64_t, dqWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dkWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dvWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockCount);
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockSize);
TILING_DATA_FIELD_DEF(bool, deterministic);
TILING_DATA_FIELD_DEF(bool, enableOptimizedScatter);
TILING_DATA_FIELD_DEF_ARR(bool, 6, rsv);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGradBaseBasicParamsOp, SparseFlashAttentionGradBaseBasicParams)

BEGIN_TILING_DATA_DEF(SparseFlashAttentionGradBasicTilingData)
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashAttentionGradBaseBasicParams, opInfo);
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashAttentionGradSplitCoreParams, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(SfagPostParams, postTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF;
//======================== basic template tiling data def end ========================//

REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad, SparseFlashAttentionGradEmptyTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_100000, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_100001, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_100010, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_100011, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_100100, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_100101, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_100110, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_100111, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_101000, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_101001, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_101010, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_101011, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_101100, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_101101, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_101110, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_101111, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_110000, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_110001, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_110010, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_110011, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_110100, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_110101, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_110110, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_110111, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_111000, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_111001, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_111010, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_111011, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_111100, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_111101, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_111110, SparseFlashAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionGrad_111111, SparseFlashAttentionGradBasicTilingData)
} // namespace optiling
