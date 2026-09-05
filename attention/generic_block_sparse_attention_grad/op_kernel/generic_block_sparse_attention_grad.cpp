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
#include "kernel_operator_list_tensor_intf.h"
#include "arc35/generic_block_sparse_attention_grad.h"

// ============================================================================
// Kernel Entry Point — GenericBlockSparseAttentionGrad (design §1 / IR)
// Inputs: query,key,value,dout,out,lse,rsvd_block_idx,rsvd_block_count,metadata,
//         atten_mask?, cu_seq_lengths?, cu_seq_lengths_kv?, seqused_q?, seqused_kv?
// Outputs: dQuery, dKey, dValue
// ============================================================================

extern "C" __global__ __aicore__ void generic_block_sparse_attention_grad(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dout, __gm__ uint8_t *out,
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *rsvdBlockIdx, __gm__ uint8_t *rsvdBlockCount, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *cuSeqLengthsQ, __gm__ uint8_t *cuSeqLengthsKv, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedKv, __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(GenericBlockSparseAttentionGradTilingDataArch35, tiling_data_in, tiling);
    const GenericBlockSparseAttentionGradTilingDataArch35 *__restrict tilingDataPtr = &tiling_data_in;
    using ARC35_TILING_CLASS = GenericBlockSparseAttentionGradTilingDataArch35;
    TPipe tPipe;

    // Tiling keys: 1000 BF16 BSND, 1001 FP16 BSND, 1002 BF16 BNSD, 1003 FP16 BNSD, 1004 BF16 TND, 1005 FP16 TND
    if (TILING_KEY_IS(1000)) {
        using gsag_type = GSAG_ARC35::GSAG_TYPE<bfloat16_t, GSAG_ARC35::BSND, ARC35_TILING_CLASS, false>;
        GSAG_ARC35::GenericBlockSparseAttentionGradArch35<gsag_type> op;
        op.Process(query, key, value, dout, out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata, attenMask,
                   cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, dq, dk, dv, user, tilingDataPtr, &tPipe);
    } else if (TILING_KEY_IS(1001)) {
        using gsag_type = GSAG_ARC35::GSAG_TYPE<half, GSAG_ARC35::BSND, ARC35_TILING_CLASS, false>;
        GSAG_ARC35::GenericBlockSparseAttentionGradArch35<gsag_type> op;
        op.Process(query, key, value, dout, out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata, attenMask,
                   cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, dq, dk, dv, user, tilingDataPtr, &tPipe);
    } else if (TILING_KEY_IS(1002)) {
        using gsag_type = GSAG_ARC35::GSAG_TYPE<bfloat16_t, GSAG_ARC35::BNSD, ARC35_TILING_CLASS, false>;
        GSAG_ARC35::GenericBlockSparseAttentionGradArch35<gsag_type> op;
        op.Process(query, key, value, dout, out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata, attenMask,
                   cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, dq, dk, dv, user, tilingDataPtr, &tPipe);
    } else if (TILING_KEY_IS(1003)) {
        using gsag_type = GSAG_ARC35::GSAG_TYPE<half, GSAG_ARC35::BNSD, ARC35_TILING_CLASS, false>;
        GSAG_ARC35::GenericBlockSparseAttentionGradArch35<gsag_type> op;
        op.Process(query, key, value, dout, out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata, attenMask,
                   cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, dq, dk, dv, user, tilingDataPtr, &tPipe);
    } else if (TILING_KEY_IS(1004)) {
        using gsag_type = GSAG_ARC35::GSAG_TYPE<bfloat16_t, GSAG_ARC35::TND, ARC35_TILING_CLASS, false>;
        GSAG_ARC35::GenericBlockSparseAttentionGradArch35<gsag_type> op;
        op.Process(query, key, value, dout, out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata, attenMask,
                   cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, dq, dk, dv, user, tilingDataPtr, &tPipe);
    } else if (TILING_KEY_IS(1005)) {
        using gsag_type = GSAG_ARC35::GSAG_TYPE<half, GSAG_ARC35::TND, ARC35_TILING_CLASS, false>;
        GSAG_ARC35::GenericBlockSparseAttentionGradArch35<gsag_type> op;
        op.Process(query, key, value, dout, out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata, attenMask,
                   cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, dq, dk, dv, user, tilingDataPtr, &tPipe);
    }
}
