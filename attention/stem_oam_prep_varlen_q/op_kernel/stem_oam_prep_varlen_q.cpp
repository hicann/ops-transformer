/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "arch35/stem_oam_prep_varlen_q_arch35.h"
#include "arch35/stem_oam_prep_varlen_q_tiling_data.h"

extern "C" __global__ __aicore__ void stem_oam_prep_varlen_q(GM_ADDR q, GM_ADDR qSeqLens, GM_ADDR cuSeqLensQ,
                                                             GM_ADDR qScale, GM_ADDR qFlat, GM_ADDR workspace,
                                                             GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(StemPrepQTilingData);
    GET_TILING_DATA_WITH_STRUCT(StemPrepQTilingData, tilingData, tiling);

    StemOamPrepVarlenQ op;
    op.Init(q, qScale, cuSeqLensQ, qFlat, &tilingData);
    op.Process();
}
