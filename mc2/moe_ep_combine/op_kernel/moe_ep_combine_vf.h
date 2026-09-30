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
 * \file moe_ep_combine_vf.h
 * \brief Metadata-to-byte-offset vector functions for MoE EP Combine.
 */
#ifndef MOE_EP_COMBINE_VF_H
#define MOE_EP_COMBINE_VF_H

#include "moe_ep_combine_base.h"

#if defined(ENABLE_MOE_EP_COMBINE_KERNEL)
#include "basic_api/kernel_basic_intf.h"
#include "basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h"
#include "basic_api/reg_compute/kernel_reg_compute_vec_binary_intf.h"
#include "basic_api/reg_compute/kernel_reg_compute_vec_binary_scalar_intf.h"
#include "basic_api/reg_compute/kernel_reg_compute_vec_duplicate_intf.h"
#include "basic_api/reg_compute/kernel_reg_compute_maskreg_intf.h"

namespace MoeEpCombineVf {

using namespace AscendC;
using namespace MoeEpCombineLayout;

constexpr uint32_t TOKENS_PER_ROUND = 64U;
constexpr uint32_t B32_ELEMENTS_PER_VECTOR = 64U;
static_assert(TOKENS_PER_ROUND == B32_ELEMENTS_PER_VECTOR);
static_assert(METADATA_BATCH_TOKENS % TOKENS_PER_ROUND == 0U);

// Shared arithmetic for full and partial rounds. Keep the high words of token-dependent offsets.
template <uint32_t HasTopkWeight>
__simd_callee__ inline void ComputeAndStoreOffsets(__ubuf__ uint32_t* dstOffUb, __ubuf__ uint32_t* xOffUb,
                                                   __ubuf__ uint32_t* wOffUb, Reg::RegTensor<uint32_t>& tok,
                                                   Reg::RegTensor<uint32_t>& topk, Reg::RegTensor<uint32_t>& rx,
                                                   Reg::RegTensor<uint32_t>& dstStepReg,
                                                   Reg::RegTensor<uint32_t>& xStepReg, Reg::RegTensor<uint32_t>& zero,
                                                   uint32_t slotStep, Reg::MaskReg& mask)
{
    Reg::RegTensor<uint32_t> offsetLo, offsetHi, topkBytes;
    Reg::MaskReg carryLo, carryHi;
    // dst = tokenIdx * (topK * slotStep) + topkIdx * slotStep, without truncating dstSlot to u32.
    Reg::Mull(offsetLo, offsetHi, tok, dstStepReg, mask);
    Reg::Muls(topkBytes, topk, slotStep, mask);
    Reg::Add(carryLo, offsetLo, offsetLo, topkBytes, mask);
    Reg::AddC(carryHi, offsetHi, offsetHi, zero, carryLo, mask);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(dstOffUb, offsetLo, offsetHi, mask);
    Reg::Mull(offsetLo, offsetHi, rx, xStepReg, mask);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(xOffUb, offsetLo, offsetHi, mask);
    if constexpr (HasTopkWeight == 1) {
        Reg::ShiftLefts(offsetLo, rx, static_cast<int16_t>(2), mask);
        Reg::ShiftRights(offsetHi, rx, static_cast<int16_t>(30), mask);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_INTLV_B32>(wOffUb, offsetLo, offsetHi, mask);
    }
}

/**
 * Build uint64 byte offsets in the SoA layout defined by MoeEpCombineLayout.
 * metaUb contains batchCount valid rows; token/topK/recv-X indices are nonnegative uint32 values.
 * batchCount is in [1, METADATA_BATCH_TOKENS]; FullBatch requires exactly METADATA_BATCH_TOKENS.
 * slotStep, dstTokenStep = topK * slotStep and xStep = axisH * sizeof(XType) fit uint32.
 * Each topK index is less than topK, so its product with slotStep also fits uint32.
 * addrUb is 32-byte aligned with two full regions, or three when HasTopkWeight == 1.
 * Interleaved stores can write a whole 64-token round even with a tail mask. Padded regions keep
 * those stores in bounds; only the first batchCount offsets in each region are valid outputs.
 * The caller waits for MTE2 before invoking this VF and for V_S before scalar consumption or
 * metadata reuse. Address-buffer reuse must also follow completion of the scalar reads.
 */
template <uint32_t HasTopkWeight, bool FullBatch>
__simd_vf__ __aicore__ inline void BuildBatchAddrs(__ubuf__ uint32_t* metaUb, __ubuf__ uint64_t* addrUb,
                                                   uint32_t batchCount, uint32_t slotStep, uint32_t dstTokenStep,
                                                   uint32_t xStep)
{
    __ubuf__ uint32_t* dstOffUb = reinterpret_cast<__ubuf__ uint32_t*>(addrUb);
    __ubuf__ uint32_t* xOffUb = reinterpret_cast<__ubuf__ uint32_t*>(addrUb + X_OFFSET_REGION);
    __ubuf__ uint32_t* wOffUb = dstOffUb;
    if constexpr (HasTopkWeight == 1) {
        wOffUb = reinterpret_cast<__ubuf__ uint32_t*>(addrUb + WEIGHT_OFFSET_REGION);
    }
    Reg::RegTensor<int32_t> lane;
    Reg::RegTensor<uint32_t> tokIndex, topkIndex, rxIndex;
    Reg::RegTensor<uint32_t> tok, topk, rx;
    Reg::RegTensor<uint32_t> dstStepReg, xStepReg, zero;
    const uint16_t fullRounds =
        FullBatch ? (METADATA_BATCH_TOKENS / TOKENS_PER_ROUND) : static_cast<uint16_t>(batchCount / TOKENS_PER_ROUND);

    // Initialize indices, broadcasts and the full mask once per batch, not once per round.
    Reg::Arange(lane, 0);
    uint32_t allCnt = B32_ELEMENTS_PER_VECTOR;
    Reg::MaskReg mAll = Reg::UpdateMask<uint32_t>(allCnt);
    Reg::Muls(tokIndex, reinterpret_cast<Reg::RegTensor<uint32_t>&>(lane), RECV_META_FIELDS, mAll);
    Reg::Adds(topkIndex, tokIndex, META_TOPK_IDX_OFFSET, mAll);
    Reg::Adds(rxIndex, tokIndex, META_RECV_X_IDX_OFFSET, mAll);
    Reg::Adds(tokIndex, tokIndex, META_TOKEN_IDX_OFFSET, mAll);
    Reg::Duplicate(dstStepReg, dstTokenStep, mAll);
    Reg::Duplicate(xStepReg, xStep, mAll);
    Reg::Duplicate(zero, 0U, mAll);

    // FullBatch has four fixed rounds and no tail path. Full rounds do not clamp Gather indices.
    for (uint16_t r = 0; r < fullRounds; ++r) {
        const uint16_t done = static_cast<uint16_t>(r * TOKENS_PER_ROUND);
        Reg::Gather<uint32_t, uint32_t, uint32_t>(tok, metaUb, tokIndex, mAll);
        Reg::Gather<uint32_t, uint32_t, uint32_t>(topk, metaUb, topkIndex, mAll);
        Reg::Gather<uint32_t, uint32_t, uint32_t>(rx, metaUb, rxIndex, mAll);
        ComputeAndStoreOffsets<HasTopkWeight>(dstOffUb + done * 2U, xOffUb + done * 2U, wOffUb + done * 2U, tok, topk,
                                              rx, dstStepReg, xStepReg, zero, slotStep, mAll);
        metaUb += TOKENS_PER_ROUND * RECV_META_FIELDS;
    }
    if constexpr (!FullBatch) {
        const uint16_t done = fullRounds * TOKENS_PER_ROUND;
        uint32_t tailCount = batchCount - done;
        if (tailCount != 0U) {
            // UpdateMask consumes tailCount, so derive the clamp before updating the mask.
            const uint32_t clamp = tailCount * RECV_META_FIELDS - 1U;
            Reg::MaskReg mTail = Reg::UpdateMask<uint32_t>(tailCount);
            Reg::RegTensor<uint32_t> gTok, gTopk, gRx;
            Reg::Mins(gTok, tokIndex, clamp, mAll);
            Reg::Mins(gTopk, topkIndex, clamp, mAll);
            Reg::Mins(gRx, rxIndex, clamp, mAll);
            Reg::Gather<uint32_t, uint32_t, uint32_t>(tok, metaUb, gTok, mTail);
            Reg::Gather<uint32_t, uint32_t, uint32_t>(topk, metaUb, gTopk, mTail);
            Reg::Gather<uint32_t, uint32_t, uint32_t>(rx, metaUb, gRx, mTail);
            ComputeAndStoreOffsets<HasTopkWeight>(dstOffUb + done * 2U, xOffUb + done * 2U, wOffUb + done * 2U, tok,
                                                  topk, rx, dstStepReg, xStepReg, zero, slotStep, mTail);
        }
    }
}

} // namespace MoeEpCombineVf
#endif // ENABLE_MOE_EP_COMBINE_KERNEL

#endif // MOE_EP_COMBINE_VF_H
