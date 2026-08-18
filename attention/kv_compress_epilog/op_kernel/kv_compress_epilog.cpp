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
 * \file kv_compress_epilog.cpp
 * \brief KV compress epilog kernel implementation
 */

#include "kv_compress_epilog_kernel.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void kv_compress_epilog(
    GM_ADDR cache,
    GM_ADDR x,
    GM_ADDR slot_mapping,
    GM_ADDR cache_out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userspace = GetUserWorkspace(workspace);
    if (userspace == nullptr) {
        return;
    }

    TPipe pipe;

    // Get tiling data
    GET_TILING_DATA_WITH_STRUCT(KvCompressEpilogTilingData, tilingDataIn, tiling);
    const KvCompressEpilogTilingData* __restrict__ tilingData = &tilingDataIn;

    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    if (TILING_KEY_IS(0)) {
        KvCompressEpilogOps::KvCompressEpilog<DTYPE_X, DTYPE_SLOT_MAPPING, DTYPE_CACHE> op(&pipe);
        op.Init(x, slot_mapping, cache, tilingData);
        op.Process();
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}