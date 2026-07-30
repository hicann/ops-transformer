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
 * \file stem_oam_prep_paged_kv_apt.cpp
 * \brief StemOamPrepPagedKV arch35 kernel entry point
 */
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "arch35/stem_oam_prep_paged_kv_tiling_key.h"
#include "arch35/stem_oam_prep_paged_kv_simd.h"

template <bool TILINGKEY>
__global__ __aicore__ void stem_oam_prep_paged_kv(GM_ADDR kCache, GM_ADDR vCache, GM_ADDR kvIndices, GM_ADDR kvSeqLens,
                                                  GM_ADDR kScaleCache, GM_ADDR vScale, GM_ADDR kFlat, GM_ADDR vBias,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(StemOamPrepPagedKvTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(StemOamPrepPagedKvTilingData, tilingData, tiling);

    TPipe pipe;
    StemOamPrepPagedKvSimd op;
    op.Init(kCache, vCache, kvIndices, kvSeqLens, kScaleCache, vScale, kFlat, vBias, workspace, &tilingData, &pipe);
    op.Process();
}
