/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_ENTRY_ARCH22_H_
#define QUANT_FLASH_ATTN_ENTRY_ARCH22_H_

#define QFA_A3_COMPILE_JOINT_ONLY 1

#include "quant_flash_attn_a8c8_cube.h"

using namespace matmul;

template <uint16_t rawConfig>
__aicore__ inline void quant_flash_attn_decode(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* qDescale,
    __gm__ uint8_t* kDescale, __gm__ uint8_t* vDescale, __gm__ uint8_t* blockTable, __gm__ uint8_t* pScale,
    __gm__ uint8_t* cuSeqLensQ, __gm__ uint8_t* cuSeqLensKv, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedKv,
    __gm__ uint8_t* sinks, __gm__ uint8_t* attnMask, __gm__ uint8_t* metadata, __gm__ uint8_t* vTail,
    __gm__ uint8_t* blockTableTail, __gm__ uint8_t* sequsedVTail, __gm__ uint8_t* attnOut, __gm__ uint8_t* softmaxLse,
    __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    (void)cuSeqLensKv;
    (void)sinks;
    (void)attnMask;
    (void)vTail;
    (void)blockTableTail;
    (void)sequsedVTail;
    GET_TILING_DATA_WITH_STRUCT(optiling::QuantFlashAttnA8C8TilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    uint32_t coreIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    uint32_t coreNum = static_cast<uint32_t>(tilingData.coreNum);
    if ASCEND_IS_AIV {
        const uint32_t subBlockNum = static_cast<uint32_t>(AscendC::GetSubBlockNum());
        coreIdx /= subBlockNum;
    }
    __gm__ uint8_t* userWorkspace = AscendC::GetUserWorkspace(workspace);

    if constexpr (rawConfig == 3U) {
        if ASCEND_IS_AIV {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BNSD>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        } else if (tilingData.kvLayout == 0U) {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BSH>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        } else if (tilingData.kvLayout == 1U) {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BNSD>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        } else {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::NZ>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        }
    }
    if constexpr (rawConfig == 4U) {
        if ASCEND_IS_AIV {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BNSD, true, 16U, true>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        } else if (tilingData.kvLayout == 0U) {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BSH, true, 16U, true>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        } else if (tilingData.kvLayout == 1U) {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BNSD, true, 16U, true>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        } else {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::NZ, true, 16U, true>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        }
    }
    if constexpr (rawConfig == 5U) {
        if ASCEND_IS_AIV {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BNSD, true, 8U, false>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        } else if (tilingData.kvLayout == 0U) {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BSH, true, 8U, false>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        } else if (tilingData.kvLayout == 1U) {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BNSD, true, 8U, false>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        } else {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::NZ, true, 8U, false>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        }
    }
    if constexpr (rawConfig == 6U) {
        if ASCEND_IS_AIV {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BNSD, true, 16U, false>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        } else if (tilingData.kvLayout == 0U) {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BSH, true, 16U, false>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        } else if (tilingData.kvLayout == 1U) {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BNSD, true, 16U, false>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        } else {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::NZ, true, 16U, false>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        }
    }
    if constexpr (rawConfig == 7U) {
        QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BSH, true, 0U, false, true>(
            query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
            attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
    }
    if constexpr (rawConfig == 8U) {
        QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BSH, true, 0U, false, true, 64U>(
            query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
            attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
    }
    if constexpr (rawConfig == 11U) {
        if ASCEND_IS_AIV {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BNSD, true, 16U, false, false, 32U, true>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        } else if (tilingData.kvLayout == 0U) {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BSH, true, 16U, false, false, 32U, true>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        } else if (tilingData.kvLayout == 1U) {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::BNSD, true, 16U, false, false, 32U, true>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        } else {
            QuantFlashAttnA8C8::QfaRunHostPlan<AttentionCommon::FIA_LAYOUT::NZ, true, 16U, false, false, 32U, true>(
                query, key, value, qDescale, kDescale, vDescale, blockTable, pScale, cuSeqLensQ, sequsedQ, sequsedKv,
                attnOut, softmaxLse, userWorkspace, &tilingData, tiling, &pipe, metadata);
        }
    }
}

#endif
