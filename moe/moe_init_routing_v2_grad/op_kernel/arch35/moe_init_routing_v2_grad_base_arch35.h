/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_v2_grad_base_arch35.h
 * \brief
 */
#ifndef MOE_INIT_ROUTING_V2_GRAD_BASE_ARCH35_H
#define MOE_INIT_ROUTING_V2_GRAD_BASE_ARCH35_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace MoeInitRoutingV2Grad {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

constexpr int64_t DROP_PAD_MODE = 1;
constexpr int64_t ACTIVE_MODE = 2;
constexpr uint32_t VL_F32 = Ops::Base::GetVRegSize() / sizeof(float);
constexpr int64_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t SCALE_COEF_TWO = 2;
constexpr int64_t SCALE_COEF_FOUR = 4;
constexpr int64_t SCALE_COEF_EIGHT = 8;

constexpr uint16_t ROW_ZERO = 0;
constexpr uint16_t ROW_ONE = 1;
constexpr uint16_t ROW_TWO = 2;
constexpr uint16_t ROW_THREE = 3;
constexpr uint16_t ROW_FOUR = 4;
constexpr uint16_t ROW_FIVE = 5;
constexpr uint16_t ROW_SIX = 6;
constexpr uint16_t ROW_SEVEN = 7;

constexpr uint32_t ROW_TWO_OFFSET = 2;
constexpr uint32_t ROW_THREE_OFFSET = 3;
constexpr uint32_t ROW_FOUR_OFFSET = 4;
constexpr uint32_t ROW_FIVE_OFFSET = 5;
constexpr uint32_t ROW_SIX_OFFSET = 6;
constexpr uint32_t ROW_SEVEN_OFFSET = 7;

constexpr int64_t MAX_BINARY_ADD_BUFFER_CNT = 64;
constexpr int64_t BINARY_ADD_COEF = 2;
constexpr uint16_t BINARY_ADD_LOOP_COUNT = 4;

constexpr int64_t UNROLL_COEF = 4;

template <int64_t Mode>
__aicore__ inline bool IsInvalidRowIdx(int32_t rowIdx, int64_t rowUpperBound, int64_t activeNum)
{
    if constexpr (Mode == ACTIVE_MODE) {
        return rowIdx < 0 || rowIdx >= activeNum;
    } else if constexpr (Mode == DROP_PAD_MODE) {
        return rowIdx < 0;
    } else {
        return rowIdx < 0 || rowIdx >= rowUpperBound;
    }
}

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

__aicore__ inline void BinaryAddVF(__ubuf__ float *binaryAddTmpAddr, uint32_t rLoopStride, uint32_t offset,
                                   uint16_t binaryAddKLoop, uint16_t binaryAddInnerLoop, uint16_t binaryAddLastLoop,
                                   MaskReg &pregLoop, RegTensor<float> &x1, RegTensor<float> &x2, RegTensor<float> &x3,
                                   RegTensor<float> &x4)
{
    uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
    for (uint16_t i = 0; i < binaryAddKLoop; i++) {
        curBinaryAddInnerLoop = curBinaryAddInnerLoop / ROW_FOUR_OFFSET;
        for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
            LoadAlign(x1, ((__ubuf__ float *)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET) * rLoopStride + offset));
            LoadAlign(x2, ((__ubuf__ float *)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + 1) * rLoopStride + offset));
            Add(x1, x1, x2, pregLoop);
            LoadAlign(x3, ((__ubuf__ float *)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + ROW_TWO_OFFSET) * rLoopStride +
                           offset));
            LoadAlign(x4, ((__ubuf__ float *)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + ROW_THREE_OFFSET) * rLoopStride +
                           offset));
            Add(x3, x3, x4, pregLoop);
            Add(x1, x1, x3, pregLoop);
            StoreAlign(((__ubuf__ float *)binaryAddTmpAddr + j * rLoopStride + offset), x1, pregLoop);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t i = 0; i < binaryAddLastLoop; i++) {
        LoadAlign(x1, ((__ubuf__ float *)binaryAddTmpAddr + offset));
        LoadAlign(x2, ((__ubuf__ float *)binaryAddTmpAddr + rLoopStride + offset));
        Add(x1, x1, x2, pregLoop);
        StoreAlign(((__ubuf__ float *)binaryAddTmpAddr + offset), x1, pregLoop);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

template <typename T>
__aicore__ inline void LoadOneTensorForDtypeT(__ubuf__ T *input, RegTensor<float> &dst, MaskReg &preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16, ((__ubuf__ half *)(input) + (offset)));
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16, ((__ubuf__ bfloat16_t *)(input) + (offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        LoadAlign(dst, ((__ubuf__ float *)(input) + (offset)));
    }
}

template <typename T>
__aicore__ inline void StoreOneTensorForDtypeT(__ubuf__ T *output, RegTensor<float> &src, MaskReg &preg,
                                               uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> yFp16;
        Cast<half, float, castTraitB322B16>(yFp16, src, preg);
        StoreAlign<half, StoreDist::DIST_PACK_B32>(((__ubuf__ half *)output + offset), yFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(((__ubuf__ bfloat16_t *)output + offset), xBf16, preg);
    } else {
        StoreAlign(((__ubuf__ float *)output + offset), src, preg);
    }
}

template <typename T>
__aicore__ inline void TwoRowAddWithTail(RegTensor<float> &dst, __ubuf__ T *input, MaskReg &preg, uint32_t offset1,
                                         uint32_t offset2, uint32_t offset3, uint32_t offset4, RegTensor<float> &rem,
                                         RegTensor<float> &nextRow, RegTensor<float> &remNextRow)
{
    LoadOneTensorForDtypeT(input, dst, preg, offset1);
    LoadOneTensorForDtypeT(input, rem, preg, offset2);
    Add(dst, dst, rem, preg);
    LoadOneTensorForDtypeT(input, nextRow, preg, offset3);
    LoadOneTensorForDtypeT(input, remNextRow, preg, offset4);
    Add(nextRow, nextRow, remNextRow, preg);
    Add(dst, dst, nextRow, preg);
}

template <typename T>
__aicore__ inline void TwoRowAdd(RegTensor<float> &dst, __ubuf__ T *input, MaskReg &preg, uint32_t offset1,
                                 uint32_t offset2, RegTensor<float> &nextRow)
{
    LoadOneTensorForDtypeT(input, dst, preg, offset1);
    LoadOneTensorForDtypeT(input, nextRow, preg, offset2);
    Add(dst, dst, nextRow, preg);
}

template <typename T>
__aicore__ inline void SixteenRowAddWithTail(RegTensor<float> &dst, __ubuf__ T *input, MaskReg &pregLoop,
                                             uint32_t quotOffset, uint32_t remOffset, uint32_t stride,
                                             RegTensor<float> &rem, RegTensor<float> &nextRow,
                                             RegTensor<float> &remNextRow, RegTensor<float> &x2, RegTensor<float> &x3,
                                             RegTensor<float> &x4)
{
    TwoRowAddWithTail(dst, input, pregLoop, quotOffset, remOffset, quotOffset + stride, remOffset + stride, rem,
                      nextRow, remNextRow);
    TwoRowAddWithTail(x2, input, pregLoop, quotOffset + ROW_TWO_OFFSET * stride, remOffset + ROW_TWO_OFFSET * stride,
                      quotOffset + ROW_THREE_OFFSET * stride, remOffset + ROW_THREE_OFFSET * stride, rem, nextRow,
                      remNextRow);
    Add(dst, dst, x2, pregLoop);
    TwoRowAddWithTail(x3, input, pregLoop, quotOffset + ROW_FOUR_OFFSET * stride, remOffset + ROW_FOUR_OFFSET * stride,
                      quotOffset + ROW_FIVE_OFFSET * stride, remOffset + ROW_FIVE_OFFSET * stride, rem, nextRow,
                      remNextRow);
    TwoRowAddWithTail(x4, input, pregLoop, quotOffset + ROW_SIX_OFFSET * stride, remOffset + ROW_SIX_OFFSET * stride,
                      quotOffset + ROW_SEVEN_OFFSET * stride, remOffset + ROW_SEVEN_OFFSET * stride, rem, nextRow,
                      remNextRow);
    Add(x3, x3, x4, pregLoop);
    Add(dst, dst, x3, pregLoop);
}

template <typename T, typename U, bool withInit>
__aicore__ inline void SequenceReduceSum(__ubuf__ T *inputAddr, __ubuf__ U *outputAddr, int64_t currentN,
                                         int64_t currentK, int64_t currentH, int64_t currentHAlign,
                                         int64_t currentKStride)
{
    int64_t hQuotCount = currentH / VL_F32 / UNROLL_COEF;
    int64_t hRem = currentH - hQuotCount * VL_F32 * UNROLL_COEF;

    uint16_t hQuotLoopCount = static_cast<uint16_t>(hQuotCount);
    uint32_t hRemCount = static_cast<uint32_t>(hRem);
    uint16_t hRemLoopCount = Ops::Base::CeilDiv(hRemCount, VL_F32);
    uint16_t nLoopCount = static_cast<uint16_t>(currentN);
    uint16_t kLoopCount = static_cast<uint16_t>(currentK);

    uint32_t nInputStride = currentKStride * currentHAlign;
    uint32_t nOutputStride = currentHAlign;
    uint32_t hRemOffset = hQuotCount * VL_F32 * UNROLL_COEF;

    __VEC_SCOPE__
    {
        // 4路H展开 × 4路K并行 = 16个输入寄存器
        RegTensor<float> x0_0, x0_1, x0_2, x0_3;
        RegTensor<float> x1_0, x1_1, x1_2, x1_3;
        RegTensor<float> x2_0, x2_1, x2_2, x2_3;
        RegTensor<float> x3_0, x3_1, x3_2, x3_3;
        // 累加器（4路H）
        RegTensor<float> sum0, sum1, sum2, sum3;

        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();

        for (uint16_t hIdx = 0; hIdx < hQuotLoopCount; hIdx++) {
            uint32_t hOffset = hIdx * VL_F32 * UNROLL_COEF;
            for (uint16_t nIdx = 0; nIdx < nLoopCount; nIdx++) {
                uint32_t nInputOffset = nIdx * nInputStride;
                if constexpr (withInit) {
                    Duplicate(sum0, 0.0, pregMain);
                    Duplicate(sum1, 0.0, pregMain);
                    Duplicate(sum2, 0.0, pregMain);
                    Duplicate(sum3, 0.0, pregMain);
                } else {
                    LoadOneTensorForDtypeT(outputAddr, sum0, pregMain, nIdx * nOutputStride + hOffset);
                    LoadOneTensorForDtypeT(outputAddr, sum1, pregMain, nIdx * nOutputStride + hOffset + VL_F32);
                    LoadOneTensorForDtypeT(outputAddr, sum2, pregMain,
                                           nIdx * nOutputStride + hOffset + VL_F32 * ROW_TWO_OFFSET);
                    LoadOneTensorForDtypeT(outputAddr, sum3, pregMain,
                                           nIdx * nOutputStride + hOffset + VL_F32 * ROW_THREE_OFFSET);
                }
                // K维度4路分块处理
                uint16_t kBlock = 0;
                while (kBlock + 3 < kLoopCount) {
                    LoadOneTensorForDtypeT(inputAddr, x0_0, pregMain,
                                           nInputOffset + hOffset + (kBlock + 0) * nOutputStride);
                    LoadOneTensorForDtypeT(inputAddr, x0_1, pregMain,
                                           nInputOffset + hOffset + (kBlock + 1) * nOutputStride);
                    LoadOneTensorForDtypeT(inputAddr, x0_2, pregMain,
                                           nInputOffset + hOffset + (kBlock + 2) * nOutputStride);
                    LoadOneTensorForDtypeT(inputAddr, x0_3, pregMain,
                                           nInputOffset + hOffset + (kBlock + 3) * nOutputStride);

                    LoadOneTensorForDtypeT(inputAddr, x1_0, pregMain,
                                           nInputOffset + hOffset + VL_F32 + (kBlock + 0) * nOutputStride);
                    LoadOneTensorForDtypeT(inputAddr, x1_1, pregMain,
                                           nInputOffset + hOffset + VL_F32 + (kBlock + 1) * nOutputStride);
                    LoadOneTensorForDtypeT(inputAddr, x1_2, pregMain,
                                           nInputOffset + hOffset + VL_F32 + (kBlock + 2) * nOutputStride);
                    LoadOneTensorForDtypeT(inputAddr, x1_3, pregMain,
                                           nInputOffset + hOffset + VL_F32 + (kBlock + 3) * nOutputStride);

                    LoadOneTensorForDtypeT(
                        inputAddr, x2_0, pregMain,
                        nInputOffset + hOffset + VL_F32 * ROW_TWO_OFFSET + (kBlock + 0) * nOutputStride);
                    LoadOneTensorForDtypeT(
                        inputAddr, x2_1, pregMain,
                        nInputOffset + hOffset + VL_F32 * ROW_TWO_OFFSET + (kBlock + 1) * nOutputStride);
                    LoadOneTensorForDtypeT(
                        inputAddr, x2_2, pregMain,
                        nInputOffset + hOffset + VL_F32 * ROW_TWO_OFFSET + (kBlock + 2) * nOutputStride);
                    LoadOneTensorForDtypeT(
                        inputAddr, x2_3, pregMain,
                        nInputOffset + hOffset + VL_F32 * ROW_TWO_OFFSET + (kBlock + 3) * nOutputStride);

                    LoadOneTensorForDtypeT(
                        inputAddr, x3_0, pregMain,
                        nInputOffset + hOffset + VL_F32 * ROW_THREE_OFFSET + (kBlock + 0) * nOutputStride);
                    LoadOneTensorForDtypeT(
                        inputAddr, x3_1, pregMain,
                        nInputOffset + hOffset + VL_F32 * ROW_THREE_OFFSET + (kBlock + 1) * nOutputStride);
                    LoadOneTensorForDtypeT(
                        inputAddr, x3_2, pregMain,
                        nInputOffset + hOffset + VL_F32 * ROW_THREE_OFFSET + (kBlock + 2) * nOutputStride);
                    LoadOneTensorForDtypeT(
                        inputAddr, x3_3, pregMain,
                        nInputOffset + hOffset + VL_F32 * ROW_THREE_OFFSET + (kBlock + 3) * nOutputStride);

                    Add(x0_0, x0_0, x0_1, pregMain);
                    Add(x0_2, x0_2, x0_3, pregMain);
                    Add(x1_0, x1_0, x1_1, pregMain);
                    Add(x1_2, x1_2, x1_3, pregMain);
                    Add(x2_0, x2_0, x2_1, pregMain);
                    Add(x2_2, x2_2, x2_3, pregMain);
                    Add(x3_0, x3_0, x3_1, pregMain);
                    Add(x3_2, x3_2, x3_3, pregMain);

                    Add(x0_0, x0_0, x0_2, pregMain);
                    Add(x1_0, x1_0, x1_2, pregMain);
                    Add(x2_0, x2_0, x2_2, pregMain);
                    Add(x3_0, x3_0, x3_2, pregMain);

                    Add(sum0, sum0, x0_0, pregMain);
                    Add(sum1, sum1, x1_0, pregMain);
                    Add(sum2, sum2, x2_0, pregMain);
                    Add(sum3, sum3, x3_0, pregMain);
                    kBlock += 4;
                }
                // 处理剩余不足4个的K位置
                while (kBlock < kLoopCount) {
                    RegTensor<float> x0, x1, x2, x3;
                    LoadOneTensorForDtypeT(inputAddr, x0, pregMain, nInputOffset + hOffset + kBlock * nOutputStride);
                    LoadOneTensorForDtypeT(inputAddr, x1, pregMain,
                                           nInputOffset + hOffset + kBlock * nOutputStride + VL_F32);
                    LoadOneTensorForDtypeT(inputAddr, x2, pregMain,
                                           nInputOffset + hOffset + kBlock * nOutputStride + VL_F32 * ROW_TWO_OFFSET);
                    LoadOneTensorForDtypeT(inputAddr, x3, pregMain,
                                           nInputOffset + hOffset + kBlock * nOutputStride + VL_F32 * ROW_THREE_OFFSET);
                    Add(sum0, sum0, x0, pregMain);
                    Add(sum1, sum1, x1, pregMain);
                    Add(sum2, sum2, x2, pregMain);
                    Add(sum3, sum3, x3, pregMain);
                    kBlock += 1;
                }
                StoreOneTensorForDtypeT(outputAddr, sum0, pregMain, nIdx * nOutputStride + hOffset);
                StoreOneTensorForDtypeT(outputAddr, sum1, pregMain, nIdx * nOutputStride + hOffset + VL_F32);
                StoreOneTensorForDtypeT(outputAddr, sum2, pregMain,
                                        nIdx * nOutputStride + hOffset + VL_F32 * ROW_TWO_OFFSET);
                StoreOneTensorForDtypeT(outputAddr, sum3, pregMain,
                                        nIdx * nOutputStride + hOffset + VL_F32 * ROW_THREE_OFFSET);
            }
        }
        // 尾部处理：非对齐H维度
        uint32_t remainingH = hRemCount;
        for (uint16_t hIdxTail = 0; hIdxTail < hRemLoopCount; hIdxTail++) {
            uint32_t elemsInBlock = (remainingH < VL_F32) ? remainingH : VL_F32;
            MaskReg pregLoop = UpdateMask<float>(elemsInBlock);
            uint32_t hOffset = hIdxTail * VL_F32 + hRemOffset;
            for (uint16_t nIdxTail = 0; nIdxTail < nLoopCount; nIdxTail++) {
                uint32_t nInputOffset = nIdxTail * nInputStride;
                if constexpr (withInit) {
                    Duplicate(sum0, 0.0, pregMain);
                } else {
                    LoadOneTensorForDtypeT(outputAddr, sum0, pregLoop, nIdxTail * nOutputStride + hOffset);
                }
                uint16_t kBlockTail = 0;
                while (kBlockTail + 3 < kLoopCount) {
                    RegTensor<float> x0, x1, x2, x3;
                    LoadOneTensorForDtypeT(inputAddr, x0, pregLoop,
                                           nInputOffset + hOffset + (kBlockTail + 0) * nOutputStride);
                    LoadOneTensorForDtypeT(inputAddr, x1, pregLoop,
                                           nInputOffset + hOffset + (kBlockTail + 1) * nOutputStride);
                    LoadOneTensorForDtypeT(inputAddr, x2, pregLoop,
                                           nInputOffset + hOffset + (kBlockTail + 2) * nOutputStride);
                    LoadOneTensorForDtypeT(inputAddr, x3, pregLoop,
                                           nInputOffset + hOffset + (kBlockTail + 3) * nOutputStride);
                    Add(x0, x0, x1, pregLoop);
                    Add(x2, x2, x3, pregLoop);
                    Add(x0, x0, x2, pregLoop);
                    Add(sum0, sum0, x0, pregLoop);
                    kBlockTail += 4;
                }
                while (kBlockTail < kLoopCount) {
                    RegTensor<float> x;
                    LoadOneTensorForDtypeT(inputAddr, x, pregLoop, nInputOffset + hOffset + kBlockTail * nOutputStride);
                    Add(sum0, sum0, x, pregLoop);
                    kBlockTail += 1;
                }
                StoreOneTensorForDtypeT(outputAddr, sum0, pregLoop, nIdxTail * nOutputStride + hOffset);
            }
            remainingH = (remainingH > VL_F32) ? (remainingH - VL_F32) : 0;
        }
    }
}
} // namespace MoeInitRoutingV2Grad
#endif // MOE_INIT_ROUTING_V2_GRAD_BASE_ARCH35_H
