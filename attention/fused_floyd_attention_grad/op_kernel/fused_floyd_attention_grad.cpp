/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifdef KFC_L1_RESERVER_SIZE
#undef KFC_L1_RESERVER_SIZE
#define KFC_L1_RESERVER_SIZE 0 // only support Gm in and Gm out
#else
#define KFC_L1_RESERVER_SIZE 0 // only support Gm in and Gm out
#endif

#include "kernel_operator.h"
#include "fused_floyd_attention_grad_post.h"
#include "fused_floyd_attention_grad_s1s2_bn2gs1s2.h"
#include "fused_floyd_attention_grad_pre.h"

constexpr MatmulConfig MM_CFG_EXCEED = GetNormalConfig(true);
constexpr MatmulConfig MM_CFG_NORMAL = GetNormalConfig(false);
constexpr CubeFormat MM_NZ_OUT_FORMAT = CubeFormat::NZ;
constexpr CubeFormat MM_ND_OUT_FORMAT = CubeFormat::ND_ALIGN;
constexpr CubeFormat MM_ND_OUT_NOALIGN = CubeFormat::ND;
constexpr uint64_t INPUT_NONE = 0;
constexpr uint64_t INPUT_EXIST = 1;
constexpr uint32_t INPUT_DISABLE = 0;
constexpr uint32_t INPUT_ENABLE = 1;

constexpr static uint32_t ND = 0;
constexpr static uint32_t NZ = 1;

constexpr static const uint32_t BNGSD = 0;
constexpr static const uint32_t SBNGD = 1;
constexpr static const uint32_t BSNGD = 2;
constexpr static const uint32_t TND = 3;

#define INVOKE_FFAG_GENERAL_S1S2_BN2GS1S2_IMPL(INPUT_TYPE, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, \
                                               INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(FusedFloydAttentionGradTilingDataS1s2Bn2gs1s2, tiling_data_in, tiling_data); \
        const FusedFloydAttentionGradTilingDataS1s2Bn2gs1s2* __restrict tilingData = &tiling_data_in; \
        const TCubeTiling* __restrict mm1tiling = &(tilingData->mm1TilingData); \
        const TCubeTiling* __restrict bmm1tiling = &(tilingData->bmm1TilingData); \
        const TCubeTiling* __restrict mm2tiling = &(tilingData->mm2TilingData); \
        const TCubeTiling* __restrict bmm2tiling = &(tilingData->bmm2TilingData); \
        const TCubeTiling* __restrict mm3tiling = &(tilingData->mm3TilingData); \
        const TCubeTiling* __restrict bmm3tiling = &(tilingData->bmm3TilingData); \
        FusedFloydAttentionGradPre<INPUT_TYPE, float, FusedFloydAttentionGradTilingDataS1s2Bn2gs1s2, true> opPre; \
        opPre.Init(dq, dk, dv, dk_1, dv_1, user, tilingData, &pipeIn); \
        opPre.Process(); \
        opPre.SyncALLCores(); \
        pipeIn.Destroy(); \
\
        TPipe pipeBase; \
        FusedFloydAttentionGradS1s2Bn2gs1s2<INPUT_TYPE, float, IS_ATTEN_MASK, IS_PSE, IS_DROP, MM_OUT_FORMAT, \
                                            INPUT_LAYOUT, MM2_OUT_FORMAT, TND_S1_PP> \
            op; \
        REGIST_MATMUL_OBJ(&pipeBase, GetSysWorkSpacePtr(), op.mm1, mm1tiling, op.mm3, mm3tiling, op.bmm1k1, \
                          bmm1tiling, op.bmm2k1, bmm2tiling, op.bmm3k1v1, bmm3tiling); \
        op.Init(key, value, key_1, value_1, dy, query, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv, \
                dk_1, dv_1, user, tilingData, &pipeBase); \
        op.Process(); \
        op.SyncALLCores(); \
        pipeBase.Destroy(); \
        TPipe pipePost; \
        constexpr static uint32_t input_format = (MM2_OUT_FORMAT == MM_NZ_OUT_FORMAT) ? NZ : ND; \
        FusedFloydAttentionGradPost<INPUT_TYPE, FusedFloydAttentionGradTilingDataS1s2Bn2gs1s2, true, INPUT_LAYOUT, \
                                    input_format> \
            opPost; \
        opPost.Init(dq, dk, dv, dk_1, dv_1, user, tilingData, &pipePost); \
        opPost.Process(); \
    } while (0)

// implementation of kernel function
extern "C" __global__ __aicore__ void fused_floyd_attention_grad(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* key_1, __gm__ uint8_t* value_1,
    __gm__ uint8_t* dy, __gm__ uint8_t* atten_mask, __gm__ uint8_t* softmax_max, __gm__ uint8_t* softmax_sum,
    __gm__ uint8_t* attention_in, __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dk_1,
    __gm__ uint8_t* dv_1, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling_data)
{
    TPipe pipeIn;
    AscendC::SetMaskNorm();
    __gm__ uint8_t* user = GetUserWorkspace(workspace);

// --------------------------------------------bfloat16 tilingkey------------------------------------------------------
#if (ORIG_DTYPE_QUERY == DT_BF16)
    if (TILING_KEY_IS(10000000000000022434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FFAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE,
                                               MM_ND_OUT_NOALIGN, BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000000000100022434UL)) {
        //  attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FFAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE,
                                               MM_ND_OUT_NOALIGN, BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    }
    // -----------------------4.1 end---------------------------------
#endif

// --------------------------------------------float16 tilingkey------------------------------------------------------
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    if (TILING_KEY_IS(10000000000000023434UL)) {
        //  attention_mask:0, pse:0, drop:0, mm_out:nd
        INVOKE_FFAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_DISABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                               BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    } else if (TILING_KEY_IS(10000000000100023434UL)) {
        //  attention_mask:1, pse:0, drop:0, mm_out:nd
        INVOKE_FFAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, INPUT_ENABLE, INPUT_DISABLE, INPUT_DISABLE, MM_ND_OUT_NOALIGN,
                                               BNGSD, MM_ND_OUT_NOALIGN, INPUT_DISABLE);
        return;
    }
    // -----------------------4.1 end---------------------------------
#endif
}
