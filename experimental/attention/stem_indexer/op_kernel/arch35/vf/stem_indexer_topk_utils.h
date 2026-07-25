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
 * \file stem_indexer_topk_utils.h
 * \brief
 */
#ifndef STEM_INDEXER_TOPK_UTILS_H
#define STEM_INDEXER_TOPK_UTILS_H

#include "kernel_operator.h"
#include "../stem_indexer_common.h"

namespace SIKernel {
namespace TopkUtils {

template <typename SCORE_T>
__simd_callee__ inline void PadMrgRowRangeVFCore(__ubuf__ SCORE_T *rowBuf, const uint32_t padStart,
                                                 const uint32_t padEnd)
{
    constexpr uint32_t VF_ELEMS = 256U / sizeof(SCORE_T);
    // Arange only supports signed index tensors; later casts reuse the same lane bits for SCORE_T comparisons.
    using IDX_T = std::conditional_t<std::is_same_v<SCORE_T, uint16_t>, int16_t, int32_t>;
    const uint32_t blockStart = padStart & ~(VF_ELEMS - 1U);
    const uint32_t blockEnd = (padEnd + VF_ELEMS - 1U) & ~(VF_ELEMS - 1U);

    MicroAPI::MaskReg maskAll = AscendC::MicroAPI::CreateMask<SCORE_T, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskGeStart;
    MicroAPI::MaskReg maskLtEnd;
    MicroAPI::MaskReg maskPad;
    MicroAPI::RegTensor<IDX_T> regIdx;
    MicroAPI::RegTensor<SCORE_T> regOldValue;
    MicroAPI::RegTensor<SCORE_T> regZero;
    MicroAPI::RegTensor<SCORE_T> regOutValue;
    MicroAPI::Duplicate(regZero, (SCORE_T)0, maskAll);

    for (uint32_t blockBase = blockStart; blockBase < blockEnd; blockBase += VF_ELEMS) {
        MicroAPI::Arange(regIdx, static_cast<IDX_T>(blockBase));
        MicroAPI::LoadAlign<SCORE_T, MicroAPI::LoadDist::DIST_NORM>(regOldValue, rowBuf + blockBase);
        MicroAPI::Compares<SCORE_T, CMPMODE::GE>(maskGeStart, (MicroAPI::RegTensor<SCORE_T> &)regIdx, (SCORE_T)padStart,
                                                 maskAll);
        MicroAPI::Compares<SCORE_T, CMPMODE::LT>(maskLtEnd, (MicroAPI::RegTensor<SCORE_T> &)regIdx, (SCORE_T)padEnd,
                                                 maskAll);
        MicroAPI::And(maskPad, maskGeStart, maskLtEnd, maskAll);
        MicroAPI::Select(regOutValue, regZero, regOldValue, maskPad);
        MicroAPI::StoreAlign<SCORE_T, MicroAPI::StoreDist::DIST_NORM>(rowBuf + blockBase, regOutValue, maskAll);
    }
}

__simd_callee__ inline uint32_t PadMrgAlignUp(const uint32_t value, const uint32_t align)
{
    // 当前调用方的align均为2的幂。
    return (value + align - 1U) & ~(align - 1U);
}

template <typename SCORE_T>
__simd_callee__ inline void PadMrgOneRowFirstS2RangeVFCore(__ubuf__ SCORE_T *rowBuf, const uint32_t topkSelectNum,
                                                           const uint32_t s2BlockValidLen,
                                                           const uint32_t s2BlockValidLenAlign)
{
    const uint32_t topkSelectNumAlign = PadMrgAlignUp(topkSelectNum, 256U);
    PadMrgRowRangeVFCore(rowBuf, topkSelectNumAlign + s2BlockValidLen, topkSelectNumAlign + s2BlockValidLenAlign);
}

template <typename SCORE_T>
__simd_callee__ inline void PadMrgOneRowNotFirstS2RangeVFCore(__ubuf__ SCORE_T *rowBuf, const uint32_t topkSelectNum,
                                                              const uint32_t s2BlockValidLen,
                                                              const uint32_t s2BlockValidLenAlign)
{
    const uint32_t topkSelectNumAlign = PadMrgAlignUp(topkSelectNum, 256U);
    PadMrgRowRangeVFCore(rowBuf, topkSelectNum, topkSelectNumAlign);
    PadMrgRowRangeVFCore(rowBuf, topkSelectNumAlign + s2BlockValidLen, topkSelectNumAlign + s2BlockValidLenAlign);
}

template <typename SCORE_T>
__simd_vf__ inline void
PadMrgFourRowsFirstS2RangeVFImpl(__ubuf__ SCORE_T *mrgValueBuf, const uint32_t rowStride, const uint32_t rowIdx0,
                                 const uint32_t rowIdx1, const uint32_t rowIdx2, const uint32_t rowIdx3,
                                 const uint32_t topkNum0, const uint32_t topkNum1, const uint32_t topkNum2,
                                 const uint32_t topkNum3, const uint32_t s2ValidLen0, const uint32_t s2ValidLen1,
                                 const uint32_t s2ValidLen2, const uint32_t s2ValidLen3)
{
    PadMrgOneRowFirstS2RangeVFCore(mrgValueBuf + rowIdx0 * rowStride, topkNum0, s2ValidLen0,
                                   PadMrgAlignUp(s2ValidLen0, SICommon::TRUNK_LEN_256));
    PadMrgOneRowFirstS2RangeVFCore(mrgValueBuf + rowIdx1 * rowStride, topkNum1, s2ValidLen1,
                                   PadMrgAlignUp(s2ValidLen1, SICommon::TRUNK_LEN_256));
    PadMrgOneRowFirstS2RangeVFCore(mrgValueBuf + rowIdx2 * rowStride, topkNum2, s2ValidLen2,
                                   PadMrgAlignUp(s2ValidLen2, SICommon::TRUNK_LEN_256));
    PadMrgOneRowFirstS2RangeVFCore(mrgValueBuf + rowIdx3 * rowStride, topkNum3, s2ValidLen3,
                                   PadMrgAlignUp(s2ValidLen3, SICommon::TRUNK_LEN_256));
}

template <typename SCORE_T>
__simd_vf__ inline void
PadMrgFourRowsNotFirstS2RangeVFImpl(__ubuf__ SCORE_T *mrgValueBuf, const uint32_t rowStride, const uint32_t rowIdx0,
                                    const uint32_t rowIdx1, const uint32_t rowIdx2, const uint32_t rowIdx3,
                                    const uint32_t topkNum0, const uint32_t topkNum1, const uint32_t topkNum2,
                                    const uint32_t topkNum3, const uint32_t s2ValidLen0, const uint32_t s2ValidLen1,
                                    const uint32_t s2ValidLen2, const uint32_t s2ValidLen3)
{
    PadMrgOneRowNotFirstS2RangeVFCore(mrgValueBuf + rowIdx0 * rowStride, topkNum0, s2ValidLen0,
                                      SICommon::TRUNK_LEN_256);
    PadMrgOneRowNotFirstS2RangeVFCore(mrgValueBuf + rowIdx1 * rowStride, topkNum1, s2ValidLen1,
                                      SICommon::TRUNK_LEN_256);
    PadMrgOneRowNotFirstS2RangeVFCore(mrgValueBuf + rowIdx2 * rowStride, topkNum2, s2ValidLen2,
                                      SICommon::TRUNK_LEN_256);
    PadMrgOneRowNotFirstS2RangeVFCore(mrgValueBuf + rowIdx3 * rowStride, topkNum3, s2ValidLen3,
                                      SICommon::TRUNK_LEN_256);
}

template <typename SCORE_T>
__simd_vf__ inline void PadMrgRowsInitBlockVFImpl(__ubuf__ SCORE_T *mrgValueBuf, const uint32_t rowStride,
                                                  const uint32_t rowNum, const uint32_t topkSelectNumAlign,
                                                  const uint32_t initialBlocks)
{
    using IDX_T = std::conditional_t<std::is_same_v<SCORE_T, uint16_t>, int16_t, int32_t>;
    MicroAPI::MaskReg maskAll = AscendC::MicroAPI::CreateMask<SCORE_T, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskInit;
    MicroAPI::RegTensor<IDX_T> regIdx;
    MicroAPI::RegTensor<SCORE_T> regZero;
    MicroAPI::Duplicate(regZero, (SCORE_T)0, maskAll);
    // Arange only supports signed index tensors; later casts reuse the same lane bits for SCORE_T comparisons.
    MicroAPI::Arange(regIdx, static_cast<IDX_T>(0));
    MicroAPI::Compares<SCORE_T, CMPMODE::LT>(maskInit, (MicroAPI::RegTensor<SCORE_T> &)regIdx, (SCORE_T)initialBlocks,
                                             maskAll);

    for (uint32_t rowIdx = 0; rowIdx < rowNum; rowIdx++) {
        __ubuf__ SCORE_T *rowBuf = mrgValueBuf + rowIdx * rowStride + topkSelectNumAlign;
        MicroAPI::StoreAlign<SCORE_T, MicroAPI::StoreDist::DIST_NORM>(rowBuf, regZero, maskInit);
    }
}

template <typename SCORE_T>
__aicore__ inline void PadMrgRowsInitBlockVF(LocalTensor<SCORE_T> &mrgValueLocal, const uint32_t rowStride,
                                             const uint32_t rowNum, const uint32_t topkSelectNumAlign,
                                             const uint32_t initialBlocks)
{
    __ubuf__ SCORE_T *mrgValueBuf = (__ubuf__ SCORE_T *)mrgValueLocal.GetPhyAddr();
    PadMrgRowsInitBlockVFImpl<SCORE_T>(mrgValueBuf, rowStride, rowNum, topkSelectNumAlign, initialBlocks);
}

template <typename SCORE_T>
__aicore__ inline void PadMrgFourRowsFirstS2RangeVF(const LocalTensor<SCORE_T> &mrgValueLocal,
                                                    const SICommon::RowIdx4 &rowIdx4,
                                                    const SICommon::TopkNum4 &topkNum4,
                                                    const SICommon::S2ValidLen4 &s2ValidLen4, const uint32_t rowStride)
{
    __ubuf__ SCORE_T *mrgValueBuf = (__ubuf__ SCORE_T *)mrgValueLocal.GetPhyAddr();
    // The caller duplicates the last valid row into tail lanes, so VF always processes four rows without branches.
    PadMrgFourRowsFirstS2RangeVFImpl<SCORE_T>(mrgValueBuf, rowStride, rowIdx4.v0, rowIdx4.v1, rowIdx4.v2, rowIdx4.v3,
                                              topkNum4.v0, topkNum4.v1, topkNum4.v2, topkNum4.v3, s2ValidLen4.v0,
                                              s2ValidLen4.v1, s2ValidLen4.v2, s2ValidLen4.v3);
}

template <typename SCORE_T>
__aicore__ inline void
PadMrgFourRowsNotFirstS2RangeVF(const LocalTensor<SCORE_T> &mrgValueLocal, const SICommon::RowIdx4 &rowIdx4,
                                const SICommon::TopkNum4 &topkNum4, const SICommon::S2ValidLen4 &s2ValidLen4,
                                const uint32_t rowStride)
{
    __ubuf__ SCORE_T *mrgValueBuf = (__ubuf__ SCORE_T *)mrgValueLocal.GetPhyAddr();
    // The caller duplicates the last valid row into tail lanes, so VF always processes four rows without branches.
    PadMrgFourRowsNotFirstS2RangeVFImpl<SCORE_T>(mrgValueBuf, rowStride, rowIdx4.v0, rowIdx4.v1, rowIdx4.v2, rowIdx4.v3,
                                                 topkNum4.v0, topkNum4.v1, topkNum4.v2, topkNum4.v3, s2ValidLen4.v0,
                                                 s2ValidLen4.v1, s2ValidLen4.v2, s2ValidLen4.v3);
}

__simd_vf__ inline void PackSparseIndicesU32VFImpl(__ubuf__ uint32_t *outBuf, __ubuf__ uint32_t *topkBuf,
                                                   const uint32_t packLen, const uint32_t initialBlocks,
                                                   const uint32_t topkSelectNum, const uint32_t windowSize,
                                                   const uint32_t windowBase)
{
    // 每行整理长度由最大topK数量、initBlocks和windowSize共同决定，单轮处理64个uint32_t。
    constexpr uint32_t VF_ELEMS = 64U;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    // Arange只支持有符号类型，生成后按相同bit位解释为uint32_t索引。
    MicroAPI::RegTensor<int32_t> regIdx;
    // regOut默认填-1，再依次用init、topK和window三个区段覆盖有效位置。
    MicroAPI::RegTensor<uint32_t> regOut;
    // topK输出位置减去initialBlocks后，得到topkBuf中的相对下标。
    MicroAPI::RegTensor<uint32_t> regTopkIdx;
    MicroAPI::RegTensor<uint32_t> regTopkValue;
    // window区段直接根据输出位置生成连续索引。
    MicroAPI::RegTensor<uint32_t> regWindowValue;
    MicroAPI::RegTensor<uint32_t> regInitialBlocks;
    MicroAPI::RegTensor<uint32_t> regTopkEnd;
    auto &regIdxU32 = (MicroAPI::RegTensor<uint32_t> &)regIdx;

    // 最终布局：[0, initialBlocks) | topK | window | -1填充。
    const uint32_t topkEnd = initialBlocks + topkSelectNum;
    const uint32_t windowEnd = topkEnd + windowSize;
    MicroAPI::Duplicate(regInitialBlocks, initialBlocks, maskAll);
    MicroAPI::Duplicate(regTopkEnd, topkEnd, maskAll);

    const uint32_t vfLoop = (packLen + VF_ELEMS - 1U) >> 6U;
    for (uint32_t loopIdx = 0U; loopIdx < vfLoop; loopIdx++) {
        const uint32_t blockBase = loopIdx * VF_ELEMS;
        // 生成当前64个输出位置，并先将整组初始化为无效索引-1。
        MicroAPI::Arange(regIdx, static_cast<int32_t>(blockBase));
        MicroAPI::Duplicate(regOut, static_cast<uint32_t>(-1), maskAll);

        // init区段直接输出位置本身，即0、1、2、...、initialBlocks-1。
        MicroAPI::MaskReg maskInit;
        MicroAPI::Compares<uint32_t, CMPMODE::LT>(maskInit, regIdxU32, initialBlocks, maskAll);
        MicroAPI::Select(regOut, regIdxU32, regOut, maskInit);

        // topK区段通过Gather读取已经计算完成的global index。
        MicroAPI::MaskReg maskTopkStart;
        MicroAPI::MaskReg maskTopkEnd;
        MicroAPI::MaskReg maskTopk;
        MicroAPI::Compares<uint32_t, CMPMODE::GE>(maskTopkStart, regIdxU32, initialBlocks, maskAll);
        MicroAPI::Compares<uint32_t, CMPMODE::LT>(maskTopkEnd, regIdxU32, topkEnd, maskAll);
        MicroAPI::And(maskTopk, maskTopkStart, maskTopkEnd, maskAll);
        MicroAPI::Sub(regTopkIdx, regIdxU32, regInitialBlocks, maskTopk);
        MicroAPI::Gather(regTopkValue, topkBuf, regTopkIdx, maskTopk);
        MicroAPI::Select(regOut, regTopkValue, regOut, maskTopk);

        // window区段生成windowBase开始的连续索引，并拼在topK区段之后。
        MicroAPI::MaskReg maskWindowStart;
        MicroAPI::MaskReg maskWindowEnd;
        MicroAPI::MaskReg maskWindow;
        MicroAPI::Compares<uint32_t, CMPMODE::GE>(maskWindowStart, regIdxU32, topkEnd, maskAll);
        MicroAPI::Compares<uint32_t, CMPMODE::LT>(maskWindowEnd, regIdxU32, windowEnd, maskAll);
        MicroAPI::And(maskWindow, maskWindowStart, maskWindowEnd, maskAll);
        MicroAPI::Sub(regWindowValue, regIdxU32, regTopkEnd, maskWindow);
        MicroAPI::Adds(regWindowValue, regWindowValue, windowBase, maskWindow);
        MicroAPI::Select(regOut, regWindowValue, regOut, maskWindow);

        // 最后一轮仅写packLen以内的lane，避免覆盖当前行固定输出区间之外的数据。
        MicroAPI::MaskReg maskPack;
        MicroAPI::Compares<uint32_t, CMPMODE::LT>(maskPack, regIdxU32, packLen, maskAll);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outBuf + blockBase, regOut, maskPack);
    }
}

__aicore__ inline void PackSparseIndicesU32VF(const LocalTensor<uint32_t> &outLocal,
                                              const LocalTensor<uint32_t> &topkLocal, const uint32_t packLen,
                                              const uint32_t initialBlocks, const uint32_t topkSelectNum,
                                              const uint32_t windowSize, const uint32_t windowBase)
{
    __ubuf__ uint32_t *outBuf = (__ubuf__ uint32_t *)outLocal.GetPhyAddr();
    __ubuf__ uint32_t *topkBuf = (__ubuf__ uint32_t *)topkLocal.GetPhyAddr();
    PackSparseIndicesU32VFImpl(outBuf, topkBuf, packLen, initialBlocks, topkSelectNum, windowSize, windowBase);
}

} // namespace TopkUtils
} // namespace SIKernel

#endif
