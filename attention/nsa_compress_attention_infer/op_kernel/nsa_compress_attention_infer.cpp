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
 * \file nsa_compress_attention_infer.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "nsa_compress_attention_infer_kernel.h"

using namespace AscendC;
using namespace NSA_COMPRESS_ATTENTION_INFER;

extern "C" __global__ __aicore__ void 
nsa_compress_attention_infer(
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attentionMaskOptional, GM_ADDR blockTableOptional, 
        GM_ADDR actualQSeqLenOptional, GM_ADDR actualCmpKvSeqLenOptional, GM_ADDR actualSelKvSeqLenOptional,
        GM_ADDR topkMaskOptional, GM_ADDR output, GM_ADDR topkIndicesOut, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR user = GetUserWorkspace(workspace);
    if (user == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tiling);
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    if (TILING_KEY_IS(0)) {
        NsaCompressAttentionInfer<NCAIType<half, half, half>> op;
        op.Run(query, key, value, blockTableOptional, actualQSeqLenOptional, actualCmpKvSeqLenOptional, 
                actualSelKvSeqLenOptional, output, topkIndicesOut, user, &tilingData);
    } else if (TILING_KEY_IS(1)) {
        NsaCompressAttentionInfer<NCAIType<half, half, half, LAYOUT::BSND>> op;
        op.Run(query, key, value, blockTableOptional, actualQSeqLenOptional, actualCmpKvSeqLenOptional, 
                actualSelKvSeqLenOptional, output, topkIndicesOut, user, &tilingData);
    }
#endif
#if (ORIG_DTYPE_QUERY == DT_BF16)
    if (TILING_KEY_IS(0)) {
        NsaCompressAttentionInfer<NCAIType<bfloat16_t, bfloat16_t, bfloat16_t>> op;
        op.Run(query, key, value, blockTableOptional, actualQSeqLenOptional, actualCmpKvSeqLenOptional, 
                actualSelKvSeqLenOptional, output, topkIndicesOut, user, &tilingData);
    } else if (TILING_KEY_IS(1)) {
        NsaCompressAttentionInfer<NCAIType<bfloat16_t, bfloat16_t, bfloat16_t, LAYOUT::BSND>> op;
        op.Run(query, key, value, blockTableOptional, actualQSeqLenOptional, actualCmpKvSeqLenOptional, 
                actualSelKvSeqLenOptional, output, topkIndicesOut, user, &tilingData);
    }
#endif
}