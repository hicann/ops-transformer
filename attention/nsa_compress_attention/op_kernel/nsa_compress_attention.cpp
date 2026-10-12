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
 * \file nsa_compress_attention.cpp
 * \brief
 */

#ifdef KFC_L1_RESERVER_SIZE
#undef KFC_L1_RESERVER_SIZE
#define KFC_L1_RESERVER_SIZE 0
#else
#define KFC_L1_RESERVER_SIZE 0
#endif

#include "kernel_operator.h"
#include "nsa_compress_attention_s1s2_bn2gs1_sab.h"
#include "nsa_compress_attention_var_len_score_sab.h"

using namespace AscendC;

#ifdef __DAV_C220_CUBE__ // CUBE 实现

#define COPY_TILING_DATA(tiling) \
    GET_TILING_DATA_MEMBER(NsaCompressAttentionGeneralTilingData, bmm1TilingData, bmm1TilingDataVar, tiling); \
    GET_TILING_DATA_MEMBER(NsaCompressAttentionGeneralTilingData, bmm2TilingData, bmm2TilingDataVar, tiling); \
    const NsaCompressAttentionGeneralTilingData* __restrict tilingData = nullptr; \
    const TCubeTiling* __restrict bmm1tiling = &bmm1TilingDataVar; \
    const TCubeTiling* __restrict bmm2tiling = &bmm2TilingDataVar;

#define INVOKE_NSA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(templateClass, ...) \
    do { \
        templateClass<__VA_ARGS__> op; \
        COPY_TILING_DATA(tiling); \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling); \
    } while (0)

#else // VECTOR 实现

#define COPY_TILING_DATA(tiling) \
    GET_TILING_DATA_WITH_STRUCT(NsaCompressAttentionGeneralTilingData, tilingDataIn, tiling); \
    const NsaCompressAttentionGeneralTilingData* __restrict tilingData = &tilingDataIn; \
    const TCubeTiling* __restrict bmm1tiling = &(tilingData->bmm1TilingData); \
    const TCubeTiling* __restrict bmm2tiling = &(tilingData->bmm2TilingData);

#define INVOKE_NSA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(templateClass, ...) \
    do { \
        COPY_TILING_DATA(tiling); \
        __gm__ uint8_t* user = GetUserWorkspace(workspace); \
        templateClass<__VA_ARGS__> op; \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling); \
        op.UnpackInit(query, key, value, attenMask, actualSeqLengths, actualSeqLengthsCmpKv, actualSeqLengthsSelKv, \
                      topkMask, softmaxMax, softmaxSum, attentionOut, topkIndicesOut, user, tilingData, &tPipe); \
        op.Process(); \
    } while (0)

#endif

extern "C" __global__ __aicore__ void nsa_compress_attention(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* attenMask,
    __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsCmpKv, __gm__ uint8_t* actualSeqLengthsSelKv,
    __gm__ uint8_t* topkMask, __gm__ uint8_t* softmaxMax, __gm__ uint8_t* softmaxSum, __gm__ uint8_t* attentionOut,
    __gm__ uint8_t* topkIndicesOut, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    TPipe tPipe;

#if (ORIG_DTYPE_QUERY == DT_FLOAT16)             // 0
    if (TILING_KEY_IS(10000000000000000024UL)) { // VarLen No-AttenMask No-TopkMask
        INVOKE_NSA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(NsaCompressAttentionVarLenScoreSameAB, LayOutTypeEnum::LAYOUT_TND,
                                                  false, false, half, float);
        return;
    } else if (TILING_KEY_IS(10000000000000001024UL)) { // VarLen No-AttenMask Has-TopkMask
        INVOKE_NSA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(NsaCompressAttentionVarLenScoreSameAB, LayOutTypeEnum::LAYOUT_TND,
                                                  false, true, half, float);
        return;
    } else if (TILING_KEY_IS(10000000000000000124UL)) { // VarLen Has-AttenMask No-TopkMask
        INVOKE_NSA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(NsaCompressAttentionVarLenScoreSameAB, LayOutTypeEnum::LAYOUT_TND,
                                                  true, false, half, float);
        return;
    } else if (TILING_KEY_IS(10000000000000001124UL)) { // VarLen Has-AttenMask Has-TopkMask
        INVOKE_NSA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(NsaCompressAttentionVarLenScoreSameAB, LayOutTypeEnum::LAYOUT_TND,
                                                  true, true, half, float);
        return;
    }
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16)                // 2
    if (TILING_KEY_IS(10000000000000000024UL)) { // VarLen No-AttenMask No-TopkMask
        INVOKE_NSA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(NsaCompressAttentionVarLenScoreSameAB, LayOutTypeEnum::LAYOUT_TND,
                                                  false, false, bfloat16_t, float);
        return;
    } else if (TILING_KEY_IS(10000000000000001024UL)) { // VarLen No-AttenMask Has-TopkMask
        INVOKE_NSA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(NsaCompressAttentionVarLenScoreSameAB, LayOutTypeEnum::LAYOUT_TND,
                                                  false, true, bfloat16_t, float);
        return;
    } else if (TILING_KEY_IS(10000000000000000124UL)) { // VarLen Has-AttenMask No-TopkMask
        INVOKE_NSA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(NsaCompressAttentionVarLenScoreSameAB, LayOutTypeEnum::LAYOUT_TND,
                                                  true, false, bfloat16_t, float);
        return;
    } else if (TILING_KEY_IS(10000000000000001124UL)) { // VarLen Has-AttenMask Has-TopkMask
        INVOKE_NSA_GENERAL_OP_IMPL_VAR_LEN_SAMEAB(NsaCompressAttentionVarLenScoreSameAB, LayOutTypeEnum::LAYOUT_TND,
                                                  true, true, bfloat16_t, float);
        return;
    }
#endif
}
