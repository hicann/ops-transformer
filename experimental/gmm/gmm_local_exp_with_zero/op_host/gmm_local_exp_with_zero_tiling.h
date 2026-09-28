/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GMM_LOCAL_EXP_TILING_H_
#define GMM_LOCAL_EXP_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GmmLocalExpWithZeroBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, K);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, trans_b);
TILING_DATA_FIELD_DEF(uint32_t, singleM);
TILING_DATA_FIELD_DEF(uint32_t, singleN);
TILING_DATA_FIELD_DEF(uint32_t, expStartIdx);
TILING_DATA_FIELD_DEF(uint32_t, expEndIdx);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GmmLocalExpWithZeroBaseParamsOp, GmmLocalExpWithZeroBaseParams)

BEGIN_TILING_DATA_DEF(GmmLocalExpWithZeroTilingData)
TILING_DATA_FIELD_DEF_STRUCT(GmmLocalExpWithZeroBaseParams, gmmBaseParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
TILING_DATA_FIELD_DEF(uint32_t, totalEleNum);
TILING_DATA_FIELD_DEF(uint32_t, eleNumPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GmmLocalExpWithZero, GmmLocalExpWithZeroTilingData)
} // namespace optiling
#endif
