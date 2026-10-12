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
 * \file rope_with_sin_cos_cache.cpp
 * \brief rope_with_sin_cos_cache.cpp
 */

#include "kernel_operator.h"
#include "rope_with_sin_cos_cache_fp32.h"
#include "rope_with_sin_cos_cache_f_bf16.h"

using namespace AscendC;
using namespace RopeWithSinCosCache;

extern "C" __global__ __aicore__ void rope_with_sin_cos_cache(GM_ADDR position_id, GM_ADDR query_in, GM_ADDR key_in,
                                                              GM_ADDR cos_sin_cache, GM_ADDR query_out, GM_ADDR key_out,
                                                              GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    TPipe pipe;
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)) && ORIG_DTYPE_QUERYIN == DT_BF16
    if (TILING_KEY_IS(20)) {
        TPipe* ptr = &pipe;
        if (ptr != nullptr) {
            RopeWithSinCosCacheFP16<bfloat16_t> op;
            op.Init(position_id, query_in, key_in, cos_sin_cache, query_out, key_out, tilingData, ptr);
            op.Process();
        }
    }
#elif ORIG_DTYPE_QUERYIN == DT_FLOAT16
    if (TILING_KEY_IS(21)) {
        TPipe* ptr = &pipe;
        if (ptr != nullptr) {
            RopeWithSinCosCacheFP16<half> op;
            op.Init(position_id, query_in, key_in, cos_sin_cache, query_out, key_out, tilingData, ptr);
            op.Process();
        }
    }
#elif ORIG_DTYPE_QUERYIN == DT_FLOAT
    if (TILING_KEY_IS(22)) {
        TPipe* ptr = &pipe;
        if (ptr != nullptr) {
            RopeWithSinCosCacheF32<float> op;
            op.Init(position_id, query_in, key_in, cos_sin_cache, query_out, key_out, tilingData, ptr);
            op.Process();
        }
    }
#endif
}
