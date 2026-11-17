/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_MLA_ENTRY_ARCH22_H_
#define QUANT_FLASH_MLA_ENTRY_ARCH22_H_

#include "../../../quant_flash_mla_metadata/op_kernel_aicpu/qfmla_metadata_arch22_layout.h"
#include "qfmla_ifa_tiling_data_mla_gen.h"
class IncreFlashAttentionTilingDataV2;
#include "qfmla_ifa_preload_mla.h"

using namespace AscendC;
using namespace matmul;

// ─────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────
constexpr uint64_t QFMLA_ARCH22_KEY_QUANT_MODE_SHIFT = 0;
constexpr uint64_t QFMLA_ARCH22_KEY_IN_OUT_LAYOUT_SHIFT = 4;
constexpr uint64_t QFMLA_ARCH22_KEY_KV_LAYOUT_SHIFT = 8;
constexpr uint64_t QFMLA_ARCH22_KEY_HAS_ATTEN_MASK_SHIFT = 12;

constexpr uint64_t QFMLA_ARCH22_QUANT_MODE_INT8 = 2;
constexpr uint64_t QFMLA_ARCH22_INOUT_LAYOUT_BSH = 0;
constexpr uint64_t QFMLA_ARCH22_INOUT_LAYOUT_TND = 1;
constexpr uint64_t QFMLA_ARCH22_KV_LAYOUT_NZ = 1;

// ─────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────
template <LAYOUT IN_LAYOUT_T, LAYOUT KV_LAYOUT_T>
using QfmlaMlaIfaType = IFAType<int8_t, int8_t, bfloat16_t, bfloat16_t, true, true, IN_LAYOUT_T, 0, false, KV_LAYOUT_T,
                                AMLAMODE::NORMAL, true, false, IncreFlashAttentionTilingDataMla>;

template <typename IFAT>
__aicore__ inline void QfmlaMlaInt8RunKernel(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* qDescale,
                                             __gm__ uint8_t* kDescale, __gm__ uint8_t* blockTable,
                                             __gm__ uint8_t* cacheSeqLens, __gm__ uint8_t* actualSeqLensQ,
                                             __gm__ uint8_t* attnMask, __gm__ uint8_t* metadata, __gm__ uint8_t* qRope,
                                             __gm__ uint8_t* kRope, __gm__ uint8_t* attnOut,
                                             __gm__ uint8_t* userWorkspace,
                                             const IncreFlashAttentionTilingDataMla* tilingData, __gm__ uint8_t* tiling)
{
    TPipe tPipe;
    IncreFlashAttentionAttenPreloadMla<IFAT, true> op;
    op.Init(query, key, key, nullptr, attnMask, actualSeqLensQ, cacheSeqLens, blockTable, nullptr, qRope, kRope,
            attnOut, nullptr, userWorkspace, tilingData, tiling, &tPipe, metadata);
    op.InitQuant(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, kDescale /*keyAntiquantScale*/, nullptr,
                 kDescale /*valueAntiquantScale*/, nullptr, nullptr, qDescale /*dequantScaleQuery*/, userWorkspace);
    op.Process();
}

template <uint8_t quantMode, uint8_t inOutLayout, uint8_t kvLayout, bool hasAttenMask>
__global__ __aicore__ void quant_flash_mla(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* qDescale,
                                           __gm__ uint8_t* kDescale, __gm__ uint8_t* blockTable,
                                           __gm__ uint8_t* cacheSeqLens, __gm__ uint8_t* cuSeqLensQ,
                                           __gm__ uint8_t* sequsedQ, __gm__ uint8_t* attnMask, __gm__ uint8_t* metadata,
                                           __gm__ uint8_t* qRope, __gm__ uint8_t* kRope, __gm__ uint8_t* attnOut,
                                           __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                           __gm__ uint8_t* tiling)
{
    REGISTER_TILING_DEFAULT(IncreFlashAttentionTilingDataMla);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    (void)hasAttenMask;
    (void)kvLayout;
    (void)softmaxLse;
    GET_TILING_DATA_WITH_STRUCT(IncreFlashAttentionTilingDataMla, tilingDataIn, tiling);
#if (ORIG_DTYPE_Q == DT_INT8)
    if constexpr (quantMode == QFMLA_ARCH22_QUANT_MODE_INT8) {
        __gm__ uint8_t* actualSeqLensQ = nullptr;
        if constexpr (inOutLayout == QFMLA_ARCH22_INOUT_LAYOUT_TND) {
            actualSeqLensQ = cuSeqLensQ;
        } else {
            actualSeqLensQ = sequsedQ;
        }
        __gm__ uint8_t* user = GetUserWorkspace(workspace);
        if constexpr (inOutLayout == QFMLA_ARCH22_INOUT_LAYOUT_TND) {
            QfmlaMlaInt8RunKernel<QfmlaMlaIfaType<LAYOUT::TND, LAYOUT::NZ>>(
                query, key, qDescale, kDescale, blockTable, cacheSeqLens, actualSeqLensQ, attnMask, metadata, qRope,
                kRope, attnOut, user, &tilingDataIn, tiling);
        } else {
            QfmlaMlaInt8RunKernel<QfmlaMlaIfaType<LAYOUT::BSH, LAYOUT::NZ>>(
                query, key, qDescale, kDescale, blockTable, cacheSeqLens, actualSeqLensQ, attnMask, metadata, qRope,
                kRope, attnOut, user, &tilingDataIn, tiling);
        }
    }
#else
    (void)query;
    (void)key;
    (void)qDescale;
    (void)kDescale;
    (void)blockTable;
    (void)cacheSeqLens;
    (void)cuSeqLensQ;
    (void)sequsedQ;
    (void)attnMask;
    (void)metadata;
    (void)qRope;
    (void)kRope;
    (void)attnOut;
    (void)workspace;
    (void)tiling;
#endif
}

#endif
