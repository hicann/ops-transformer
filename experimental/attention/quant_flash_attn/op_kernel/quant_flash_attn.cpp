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
 * \file quant_flash_attn.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "arch35/quant_flash_attn_template_tiling_key.h"
#include "arch35/quant_flash_attn_kernel_dn.h"
#include "arch35/quant_flash_attn_block_cube_dn.h"
#include "arch35/quant_flash_attn_block_vector_dn.h"

using namespace optiling;
using namespace AscendC;
using namespace QFA_KERNEL;

template <uint8_t Q_OUT_LAYOUT_T>
__aicore__ inline constexpr QFA_LAYOUT GetQueryLayout()
{
    static_assert((Q_OUT_LAYOUT_T == LAYOUT_ENUM_BSND) || (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD) ||
                      (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD_BSND) || (Q_OUT_LAYOUT_T == LAYOUT_ENUM_TND),
                  "Get Query Layout fail, Q_OUT_LAYOUT_T is incorrect");
    if constexpr (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BSND) {
        return QFA_LAYOUT::BSND;
    } else if constexpr (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD || Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD_BSND) {
        return QFA_LAYOUT::BNSD;
    } else if constexpr (Q_OUT_LAYOUT_T == LAYOUT_ENUM_TND) {
        return QFA_LAYOUT::TND;
    }
}

template <uint8_t Q_OUT_LAYOUT_T>
__aicore__ inline constexpr QFA_LAYOUT GetOutLayout()
{
    static_assert((Q_OUT_LAYOUT_T == LAYOUT_ENUM_BSND) || (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD) ||
                      (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD_BSND) || (Q_OUT_LAYOUT_T == LAYOUT_ENUM_TND),
                  "Get AttnOut Layout fail, Q_OUT_LAYOUT_T is incorrect");
    if constexpr (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BSND || Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD_BSND) {
        return QFA_LAYOUT::BSND;
    } else if constexpr (Q_OUT_LAYOUT_T == LAYOUT_ENUM_BNSD) {
        return QFA_LAYOUT::BNSD;
    } else if constexpr (Q_OUT_LAYOUT_T == LAYOUT_ENUM_TND) {
        return QFA_LAYOUT::TND;
    }
}

template <uint8_t Q_OUT_LAYOUT_T, uint8_t KV_STORAGE_MODE>
__aicore__ inline constexpr QFA_LAYOUT GetKvLayout()
{
    static_assert((KV_STORAGE_MODE == KV_STORAGE_MODE_CONTINUE) || (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BSND) ||
                      (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BNSD),
                  "Get Key/Value Layout fail, KV_STORAGE_MODE is incorrect");
    if constexpr (KV_STORAGE_MODE == KV_STORAGE_MODE_CONTINUE) {
        return GetQueryLayout<Q_OUT_LAYOUT_T>();
    } else if constexpr (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BSND) {
        return QFA_LAYOUT::BSND; // block内的格式类似于BSND
    } else if constexpr (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BNSD) {
        return QFA_LAYOUT::BNSD; // block内的格式类似于BNSD
    }
}

template <uint8_t KV_STORAGE_MODE>
__aicore__ inline constexpr bool IsPageAttention()
{
    static_assert((KV_STORAGE_MODE == KV_STORAGE_MODE_CONTINUE) || (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BSND) ||
                      (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BNSD),
                  "Get PAGE_ATTENTION flag fail, KV_STORAGE_MODE is incorrect");
    return (KV_STORAGE_MODE != KV_STORAGE_MODE_CONTINUE);
}

template <uint8_t Q_OUT_LAYOUT_T, uint8_t KV_STORAGE_MODE, bool HAS_MASK>
__global__ __aicore__ void
quant_flash_attn(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *q_descale,
                 __gm__ uint8_t *k_descale, __gm__ uint8_t *v_descale, __gm__ uint8_t *blockTable,
                 __gm__ uint8_t *pScale, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensKv,
                 __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *learnableSink,
                 __gm__ uint8_t *attnMask, __gm__ uint8_t *metadata, __gm__ uint8_t *attnOut,
                 __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(QuantFlashAttnTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    (void)pScale;

#if (ORIG_DTYPE_Q == DT_FLOAT4_E2M1) && (ORIG_DTYPE_Q_DESCALE == DT_FLOAT8_E8M0) && (ORIG_DTYPE_ATTN_OUT == DT_BF16)
    InitSocState();
    GET_TILING_DATA_MEMBER(QuantFlashAttnTilingData, baseTiling, baseTilingIn, tiling);
    const FlashAttnTilingData *__restrict tilingData = &baseTilingIn;

    using QFA_T = QFAType<fp4x2_e2m1_t, fp8_e8m0_t, bfloat16_t, IsPageAttention<KV_STORAGE_MODE>(),
                          GetQueryLayout<Q_OUT_LAYOUT_T>(), GetKvLayout<Q_OUT_LAYOUT_T, KV_STORAGE_MODE>(),
                          GetOutLayout<Q_OUT_LAYOUT_T>(), HAS_MASK>;
    using CubBlock = QuantFlashAttnBlockCubeDn<QFA_T>;
    using VectorBlock = QuantFlashAttnBlockVectorDn<QFA_T>;
    QuantFlashAttnKernelDn<QFA_T, CubBlock, VectorBlock> op;
    op.Init(query, key, value, q_descale, k_descale, v_descale, blockTable, cuSeqlensQ, cuSeqlensKv, sequsedQ,
            sequsedKv, attnMask, learnableSink, softmaxLse, attnOut, workspace, metadata, tilingData);
    op.Process();
    PipeBarrier<PIPE_ALL>();
#endif
}
