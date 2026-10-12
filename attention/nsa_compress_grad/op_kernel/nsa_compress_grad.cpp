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
 * \file nsa_compress_grad.cpp
 * \brief nsa_compress_grad kernal
 */
#include "nsa_compress_grad.h"

using namespace AscendC;

using namespace NsaCompressGrad;

extern "C" __global__ __aicore__ void nsa_compress_grad(GM_ADDR outputGrad, GM_ADDR input, GM_ADDR weight,
                                                        GM_ADDR actSeqLenOptional, GM_ADDR inputGrad,
                                                        GM_ADDR weightGrad, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }

    TPipe tPipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
#if (ORIG_DTYPE_OUTPUTGRAD == DT_FLOAT16)
    if (TILING_KEY_IS(0)) {
        NsaCompressGradND<half> op;
        op.Init(outputGrad, input, weight, actSeqLenOptional, inputGrad, weightGrad, userWs, &tilingData, &tPipe);
        op.Process();
        op.MoveResult();
        tPipe.Destroy();
    }
#endif
#if (ORIG_DTYPE_OUTPUTGRAD == DT_BF16)
    if (TILING_KEY_IS(0)) {
        NsaCompressGradND<bfloat16_t> op;
        op.Init(outputGrad, input, weight, actSeqLenOptional, inputGrad, weightGrad, userWs, &tilingData, &tPipe);
        op.Process();
        op.MoveResult();
        tPipe.Destroy();
    }
#endif
}
