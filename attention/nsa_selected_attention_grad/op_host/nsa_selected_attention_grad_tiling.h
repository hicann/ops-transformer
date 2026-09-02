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
 * \file nsa_selected_attention_grad_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "nsa_selected_attention_grad_tiling_common.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(NsaSelectedAttentionGradSplitCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, s2OuterOuter);
TILING_DATA_FIELD_DEF(uint32_t, singleM);
TILING_DATA_FIELD_DEF(uint32_t, singleN);
TILING_DATA_FIELD_DEF(uint32_t, sftBaseM);
TILING_DATA_FIELD_DEF(uint32_t, sftBaseN);
TILING_DATA_FIELD_DEF(uint32_t, rsv);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NsaSelectedAttentionGradSplitCoreParamsOp, NsaSelectedAttentionGradSplitCoreParams)

BEGIN_TILING_DATA_DEF(NsaPostParams)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, postUbBaseSize);
TILING_DATA_FIELD_DEF(uint32_t, postMaxDataSize);
TILING_DATA_FIELD_DEF(uint32_t, nzReservedSize);
TILING_DATA_FIELD_DEF(uint32_t, qPostBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, qPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, qPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, qPostTailNum);
TILING_DATA_FIELD_DEF(uint32_t, kPostBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, kPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, kPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, kPostTailNum);
TILING_DATA_FIELD_DEF(uint32_t, vPostBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, vPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, vPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, vPostTailNum);
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
REGISTER_TILING_DATA_CLASS(NsaPostParamsOp, NsaPostParams)

BEGIN_TILING_DATA_DEF(NsaSelectedAttentionGradEmptyTilingData)
TILING_DATA_FIELD_DEF_STRUCT(NsaSelectedAttentionGradSplitCoreParams, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(NsaPostParams, postTilingData);
END_TILING_DATA_DEF;

//======================== norm template tiling data def start ========================//
BEGIN_TILING_DATA_DEF(NsaSelectedAttentionGradBaseParams)
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
REGISTER_TILING_DATA_CLASS(NsaSelectedAttentionGradBaseParamsOp, NsaSelectedAttentionGradBaseParams)

BEGIN_TILING_DATA_DEF(NsaSelectedAttentionGradTilingData)
TILING_DATA_FIELD_DEF_STRUCT(NsaSelectedAttentionGradBaseParams, opInfo);
TILING_DATA_FIELD_DEF_STRUCT(NsaSelectedAttentionGradSplitCoreParams, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(NsaPostParams, postTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm31TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm4TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm5TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF;
//======================== norm template tiling data def end ========================//

//======================== basic template tiling data def start ========================//
BEGIN_TILING_DATA_DEF(NsaSelectedAttentionGradBaseBasicParams)
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
TILING_DATA_FIELD_DEF(uint32_t, mm12WorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, castUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, dqWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dkWorkspaceLen);
TILING_DATA_FIELD_DEF(int64_t, dvWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockCount);
TILING_DATA_FIELD_DEF(uint32_t, selectedBlockSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NsaSelectedAttentionGradBaseBasicParamsOp, NsaSelectedAttentionGradBaseBasicParams)

BEGIN_TILING_DATA_DEF(NsaSelectedAttentionGradBasicTilingData)
TILING_DATA_FIELD_DEF_STRUCT(NsaSelectedAttentionGradBaseBasicParams, opInfo);
TILING_DATA_FIELD_DEF_STRUCT(NsaSelectedAttentionGradSplitCoreParams, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(NsaPostParams, postTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF;
//======================== basic template tiling data def end ========================//

REGISTER_TILING_DATA_CLASS(NsaSelectedAttentionGrad, NsaSelectedAttentionGradEmptyTilingData)

REGISTER_TILING_DATA_CLASS(NsaSelectedAttentionGrad_0, NsaSelectedAttentionGradTilingData)
REGISTER_TILING_DATA_CLASS(NsaSelectedAttentionGrad_1, NsaSelectedAttentionGradTilingData)

REGISTER_TILING_DATA_CLASS(NsaSelectedAttentionGrad_10, NsaSelectedAttentionGradBasicTilingData)
REGISTER_TILING_DATA_CLASS(NsaSelectedAttentionGrad_11, NsaSelectedAttentionGradBasicTilingData)
} // namespace optiling
