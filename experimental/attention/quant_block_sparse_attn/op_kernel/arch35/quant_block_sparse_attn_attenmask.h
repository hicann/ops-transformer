/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_block_sparse_attn_attenmask.h
 * \brief
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_ATTENMASK_H_
#define QUANT_BLOCK_SPARSE_ATTN_ATTENMASK_H_

#include "common/util_regbase.h"
#include "quant_block_sparse_attn_common_arch35.h"

using namespace AscendC;
using namespace AscendC::MicroAPI;
namespace regbaseutil {
enum class AttenMaskCompressMode {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE = 1,
    RIGHT_DOWN_CAUSAL_MODE = 2,
    BAND_MODE = 3,
    PREFIX_MODE = 4,
    RIGHT_DOWN_CAUSAL_BAND_MODE = 5,
    BAND_LEFT_UP_CAUSAL_MODE = 6
};

enum class AttenMaskComputeMode {
    NORMAL_MODE = 0,
    CAUSAL_OR_NEXT_ONLY_MODE,
    PRE_ONLY_MODE,
    PRE_AND_NEXT_MODE,
    NO_NEED_COMPUTE_MODE,
    PREFIX_COMPUTE_MODE,
    PREFIX_N_COMPUTE_MODE
};

struct AttenMaskInfo {
    int64_t preTokens;
    int64_t nextTokens;
    uint8_t compressMode;
    int64_t attenMaskS2Size;
    int64_t bandIndex;
    GM_ADDR prefixNAddr;
    int64_t attenMaskOffsetPre;
    AttenMaskComputeMode computeMode = AttenMaskComputeMode::NORMAL_MODE;
    int64_t attenMaskS1Offset;
};

__aicore__ inline void BoolCopyInRegbase(const LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
                                         int64_t srcOffset, uint32_t s1Size, uint32_t s2Size, int64_t totalS2Size,
                                         int64_t s2BaseSize, ConstInfo &constInfo)
{
    if (s1Size == 0 || s2Size == 0) {
        return;
    }

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s1Size;
    dataCopyParams.blockLen = CeilDiv(s2Size, blockBytes);
    dataCopyParams.dstStride = CeilDiv(s2BaseSize, blockBytes) - dataCopyParams.blockLen;
    if (totalS2Size % blockBytes == 0 && s2Size % blockBytes == 0) {
        dataCopyParams.srcStride = (totalS2Size - s2Size) / blockBytes;
        DataCopy(dstTensor, srcTensor[srcOffset], dataCopyParams);
    } else {
        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = s1Size;
        dataCopyExtParams.dstStride = CeilDiv(s2BaseSize, blockBytes) - CeilDiv(s2Size, blockBytes);
        dataCopyExtParams.blockLen = s2Size;
        dataCopyExtParams.srcStride = totalS2Size - s2Size;
        DataCopyPadExtParams<uint8_t> dataCopyPadParams;
        DataCopyPad(dstTensor, srcTensor[srcOffset], dataCopyExtParams, dataCopyPadParams);
    }
}

template <bool hasAtten>
__aicore__ inline void GetAttenMaskComputeMode(int64_t deltaCausalOrNext, int64_t deltaPre, int64_t s1Offset,
                                               const RunInfo &runInfo, ConstInfo &constInfo,
                                               AttenMaskInfo &attenMaskInfo)
{
    if constexpr (hasAtten) {
        uint32_t s2BaseSize = constInfo.s2BaseSize;
        int64_t causalOrNextFactor = deltaCausalOrNext - s2BaseSize;
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE) ||
            attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            if (causalOrNextFactor >= 0) {
                attenMaskInfo.computeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
            } else {
                attenMaskInfo.computeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
            }
            return;
        }
        if (((attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) ||
             (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_BAND_MODE) &&
              runInfo.boIdx == attenMaskInfo.bandIndex) ||
             (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_LEFT_UP_CAUSAL_MODE) &&
              runInfo.boIdx == attenMaskInfo.bandIndex))) {
            int64_t preFactor = deltaPre + 1 + constInfo.s1BaseSize;
            if (causalOrNextFactor >= 0 && preFactor <= 0) {
                attenMaskInfo.computeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
            } else if (causalOrNextFactor < 0 && preFactor <= 0) {
                attenMaskInfo.computeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
            } else if (causalOrNextFactor >= 0 && preFactor > 0) {
                attenMaskInfo.computeMode = AttenMaskComputeMode::PRE_ONLY_MODE;
            } else {
                attenMaskInfo.computeMode = AttenMaskComputeMode::PRE_AND_NEXT_MODE;
            }
        }
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) {
            int64_t preFactor = deltaPre - constInfo.s2BaseSize;
            // Triangular part and rectangular part have one is not counted, then the whole is not counted,
            // otherwise it needs to be calculated
            if (causalOrNextFactor >= 0 || preFactor >= 0 || deltaPre > runInfo.actualS2Size) {
                // attenmask value is all 0, no need to compute
                attenMaskInfo.computeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
            } else {
                int64_t intersectionX = runInfo.actualS1Size - runInfo.actualS2Size +
                                        ((__gm__ int64_t *)attenMaskInfo.prefixNAddr)[runInfo.boIdx];
                if (s1Offset >= intersectionX) {
                    attenMaskInfo.computeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
                } else if (s1Offset + constInfo.s1BaseSize <= intersectionX) {
                    attenMaskInfo.computeMode = AttenMaskComputeMode::PREFIX_N_COMPUTE_MODE;
                } else {
                    attenMaskInfo.computeMode = AttenMaskComputeMode::PREFIX_COMPUTE_MODE;
                }
            }
        }
        return;
    }
}

__aicore__ inline int64_t ComputeOffsetForCausal(const int64_t &delta, const uint32_t &s1BaseSize,
                                                 const uint32_t &s2BaseSize, const uint32_t &attenMaskS2Size,
                                                 const int64_t &vecCoreOffset, const bool useDn = false)
{
    if (useDn) {
        if (delta >= 0) {
            return Min(delta, s2BaseSize) + vecCoreOffset;
        }
        // 非对齐 RDC：deltaN(=actualS1Size-actualS2Size) 非 0、非 128 倍数时，diagonal-crossing 的
        // sparse block 走 delta<0 分支。原 "+1" 在 DN(转置) 读取下把对角线右移一列，使该块多保留
        // 一列 kv(causal 过宽，c<=r+D+1 而非 c<=r+D)，导致 555/666 等非对齐 shape 在需要 partial mask
        // 的 q block 上精度大面积越界(对齐场景 deltaN=0、对角块走 delta==0 的 >=0 分支，不触发此 +1，故不受影响)。
        return (Min(-1 * delta, s1BaseSize)) * attenMaskS2Size + vecCoreOffset;
    }
    if (delta <= 0) {
        return Min(-1 * delta, s1BaseSize) + vecCoreOffset * attenMaskS2Size;
    }
    return (Min(delta, s2BaseSize) + vecCoreOffset) * attenMaskS2Size;
}


#ifndef __CCE_KT_TEST__
// 128bit 寄存器每次加载 4 个 uint32_t，偏移量 = 128bit/8 = 16 字节，即 64 个字节时为 4 个 128bit 元素
constexpr uint32_t VREG_STRIDE_BYTES = 64;
// XOR 掩码: 对每个字节取反低 8bit，用于 mask 合并
constexpr uint32_t XOR_MASK = 0x1010101;

__simd_vf__ inline void MergeBandVF(const uint64_t maskPreUb, const uint64_t maskNextUb, const uint16_t loopCount)
{
    RegTensor<uint32_t> vreg_pre;
    RegTensor<uint32_t> vreg_next;
    RegTensor<uint32_t> vreg_xor;
    RegTensor<uint32_t> vreg_not;
    RegTensor<uint32_t> vreg_or;
    MaskReg preg_all = CreateMask<uint32_t, MaskPattern::ALL>();
    Duplicate(vreg_xor, XOR_MASK);

    for (uint16_t i = 0; i < loopCount; ++i) {
        LoadAlign(vreg_pre, (__ubuf__ uint32_t *&)maskPreUb + i * VREG_STRIDE_BYTES);
        LoadAlign(vreg_next, (__ubuf__ uint32_t *&)maskNextUb + i * VREG_STRIDE_BYTES);
        Xor(vreg_not, vreg_pre, vreg_xor, preg_all);
        Or(vreg_or, vreg_not, vreg_next, preg_all);
        StoreAlign((__ubuf__ uint32_t *&)maskNextUb + i * VREG_STRIDE_BYTES, vreg_or, preg_all);
    }
}

template <bool hasAtten>
__aicore__ inline void MergeBandModeMask(LocalTensor<uint8_t> &maskPre, LocalTensor<uint8_t> &maskNext,
                                         int32_t &halfS1RealSize, int64_t s2BaseSize)
{
    uint64_t maskPreUb = maskPre.GetPhyAddr();
    uint64_t maskNextUb = maskNext.GetPhyAddr();
    uint16_t rowNumEachLoop;
    uint64_t rowNumTimesEachLoop;
    if (s2BaseSize > regBytes) {
        rowNumEachLoop = 1;
        rowNumTimesEachLoop = static_cast<uint16_t>(s2BaseSize) / regBytes;
    } else {
        rowNumEachLoop = regBytes / static_cast<uint16_t>(s2BaseSize);
        rowNumTimesEachLoop = 1;
    }
    uint16_t halfS1RealSizeLoop = static_cast<uint16_t>(halfS1RealSize) + 1;
    uint16_t loopCount = (halfS1RealSizeLoop / rowNumEachLoop) * rowNumTimesEachLoop;

    MergeBandVF(maskPreUb, maskNextUb, loopCount);
}

__simd_vf__ inline void MergePrefixVF(const uint64_t maskPreUb, const uint64_t maskNextUb, const uint16_t loopCount)
{
    RegTensor<uint32_t> vreg_pre;
    RegTensor<uint32_t> vreg_next;
    RegTensor<uint32_t> vreg_and;
    MaskReg preg_all = CreateMask<uint32_t, MaskPattern::ALL>();

    for (uint16_t i = 0; i < loopCount; ++i) {
        LoadAlign(vreg_pre, (__ubuf__ uint32_t *&)maskPreUb + i * VREG_STRIDE_BYTES);
        LoadAlign(vreg_next, (__ubuf__ uint32_t *&)maskNextUb + i * VREG_STRIDE_BYTES);
        And(vreg_and, vreg_pre, vreg_next, preg_all);
        StoreAlign((__ubuf__ uint32_t *&)maskNextUb + i * VREG_STRIDE_BYTES, vreg_and, preg_all);
    }
}

template <bool hasAtten>
__aicore__ inline void MergePrefixModeMask(LocalTensor<uint8_t> &maskPre, LocalTensor<uint8_t> &maskNext,
                                           int32_t &halfS1RealSize, int64_t s2BaseSize)
{
    uint64_t maskPreUb = maskPre.GetPhyAddr();
    uint64_t maskNextUb = maskNext.GetPhyAddr();
    uint16_t rowNumEachLoop;
    uint64_t rowNumTimesEachLoop;
    if (s2BaseSize > regBytes) {
        rowNumEachLoop = 1;
        rowNumTimesEachLoop = static_cast<uint16_t>(s2BaseSize) / regBytes;
    } else {
        rowNumEachLoop = regBytes / static_cast<uint16_t>(s2BaseSize);
        rowNumTimesEachLoop = 1;
    }
    uint16_t halfS1RealSizeLoop = static_cast<uint16_t>(halfS1RealSize) + 1;
    uint16_t loopCount = (halfS1RealSizeLoop / rowNumEachLoop) * rowNumTimesEachLoop;

    MergePrefixVF(maskPreUb, maskNextUb, loopCount);
}
#else
template <bool hasAtten>
__aicore__ inline void MergeBandModeMask(LocalTensor<uint8_t> &maskPre, LocalTensor<uint8_t> &maskNext,
                                         int32_t &halfS1RealSize, int64_t s2BaseSize)
{}

template <bool hasAtten>
__aicore__ inline void MergePrefixModeMask(LocalTensor<uint8_t> &maskPre, LocalTensor<uint8_t> &maskNext,
                                           int32_t &halfS1RealSize, int64_t s2BaseSize)
{}
#endif

template <bool hasAtten, DTemplateType dTemplateType = DTemplateType::Aligned128>
__aicore__ inline int64_t ComputeAttenMaskInnerOffset(const RunInfo &runInfo, ConstInfo &constInfo,
                                                      AttenMaskInfo &attenMaskInfo, bool first_block = true,
                                                      const bool useDn = false, const int32_t subLoop = 0)
{
    if constexpr (hasAtten) {
        // compress mode, 推理场景的TND和TND的offset计算相同，因为mask被padding到了最大的s2Size
        int64_t deltaCausalOrNext = 0;
        int64_t deltaPre = 0;
        int64_t deltaN = runInfo.actualS1Size - runInfo.actualS2Size;
        int64_t s1Offset = runInfo.s1oIdx * constInfo.s1BaseSize;
        int64_t s2Offset = 0;
        if (unlikely(first_block)) {
            s2Offset = runInfo.sparseBlkIdx1 * constInfo.kvSparseBlockSize;
        } else {
            s2Offset = runInfo.sparseBlkIdx2 * constInfo.kvSparseBlockSize;
        }

        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            deltaCausalOrNext = s1Offset - s2Offset - deltaN;
        }
        GetAttenMaskComputeMode<hasAtten>(deltaCausalOrNext, deltaPre, s1Offset, runInfo, constInfo, attenMaskInfo);
        int64_t ret = 0;

        uint32_t s2BaseSize = constInfo.s2BaseSize;
        ret = ComputeOffsetForCausal(deltaCausalOrNext, constInfo.s1BaseSize, s2BaseSize, attenMaskInfo.attenMaskS2Size,
                                     runInfo.vecCoreOffset, useDn);
        return ret;
    }
}

template <bool hasAtten, DTemplateType dTemplateType = DTemplateType::Aligned128>
__aicore__ inline int64_t ComputeAttenMaskOffset(const RunInfo &runInfo, ConstInfo &constInfo,
                                                 AttenMaskInfo &attenMaskInfo, const bool first_block = false,
                                                 const bool useDn = true, const int32_t subLoop = 0)
{
    auto result = ComputeAttenMaskInnerOffset<hasAtten, dTemplateType>(runInfo, constInfo, attenMaskInfo, first_block,
                                                                       useDn, subLoop);
    return result;
}

__aicore__ inline void GenerateRightDownCausalMaskDn(LocalTensor<uint8_t> &dstTensor, RunInfo &runInfo,
                                                     ConstInfo &constInfo, int64_t sparseBlkIdx,
                                                     uint32_t sparseBlkRealSize)
{
    uint32_t halfS1BaseSize = constInfo.s1BaseSize >> 1;
    int64_t qBase = runInfo.s1oIdx * constInfo.s1BaseSize + runInfo.vecCoreOffset;
    int64_t kvBase = sparseBlkIdx * constInfo.kvSparseBlockSize;
    int64_t causalShift = runInfo.actualS2Size - runInfo.actualS1Size;
    for (uint32_t s2Idx = 0; s2Idx < sparseBlkRealSize; ++s2Idx) {
        int64_t kvPos = kvBase + s2Idx;
        for (uint32_t s1Idx = 0; s1Idx < halfS1BaseSize; ++s1Idx) {
            int64_t qPos = qBase + s1Idx;
            dstTensor.SetValue(s2Idx * halfS1BaseSize + s1Idx, kvPos <= qPos + causalShift ? 1 : 0);
        }
    }
}

template <bool hasAtten>
__aicore__ inline void AttenMaskCopyIn(TQue<QuePosition::VECIN, 1> &attenMaskInQue, GlobalTensor<uint8_t> &srcTensor,
                                       RunInfo &runInfo, ConstInfo &constInfo, AttenMaskInfo &attenMaskInfo,
                                       bool isPre = false)
{
    if constexpr (hasAtten) {
        LocalTensor<uint8_t> attenMaskUb = attenMaskInQue.template AllocTensor<uint8_t>();
        int64_t maskOffset = ComputeAttenMaskOffset<hasAtten>(runInfo, constInfo, attenMaskInfo);
        BoolCopyInRegbase(attenMaskUb, srcTensor, maskOffset, runInfo.halfS1RealSize, runInfo.s2RealSize,
                          attenMaskInfo.attenMaskS2Size, constInfo.s2BaseSize, constInfo);
        attenMaskInQue.template EnQue(attenMaskUb);
        return;
    }
}

template <bool hasAtten>
__aicore__ inline void AttenMaskCopyInDnFun(LocalTensor<uint8_t> attenMaskUb, GlobalTensor<uint8_t> &srcTensor,
                                            RunInfo &runInfo, ConstInfo &constInfo, AttenMaskInfo &attenMaskInfo,
                                            bool needAtten, uint64_t offset, int32_t sparseBlkIdx, uint32_t s2Len)
{
    if constexpr (hasAtten) {
        int64_t maskOffset = 0;
        if (sparseBlkIdx == -1) {
            // 丢弃块：s2Len(=s2SparseBlkRealSize)为0，必须按整块宽度(kvSparseBlockSize)清零该段，
            // 否则该段 mask 未初始化，vf 读到未初始化 mask + cube 未写过的 stale/NaN bmm1 区域 → NaN/随时序漂移。
            Duplicate<uint8_t>(attenMaskUb[offset], 0, constInfo.kvSparseBlockSize * (constInfo.s1BaseSize >> 1));
        } else {
            if (needAtten) {
                maskOffset = ComputeAttenMaskOffset<hasAtten>(runInfo, constInfo, attenMaskInfo,
                                                              sparseBlkIdx == runInfo.sparseBlkIdx1);
                BoolCopyInRegbase(attenMaskUb[offset], srcTensor, maskOffset, s2Len, constInfo.s1BaseSize >> 1,
                                  attenMaskInfo.attenMaskS2Size, constInfo.s1BaseSize >> 1, constInfo);
            } else {
                Duplicate<uint8_t>(attenMaskUb[offset], 1, s2Len * (constInfo.s1BaseSize >> 1));
            }
        }
    }
}

template <bool hasAtten>
__aicore__ inline void AttenMaskCopyInDn(TQue<QuePosition::VECIN, 1> &attenMaskInQue, GlobalTensor<uint8_t> &srcTensor,
                                         RunInfo &runInfo, ConstInfo &constInfo, AttenMaskInfo &attenMaskInfo,
                                         bool needAtten1, bool needAtten2)
{
    LocalTensor<uint8_t> attenMaskUb = attenMaskInQue.template AllocTensor<uint8_t>();
    AttenMaskCopyInDnFun<hasAtten>(attenMaskUb, srcTensor, runInfo, constInfo, attenMaskInfo, needAtten1, 0,
                                   runInfo.sparseBlkIdx1, runInfo.s2SparseBlk1RealSize);
    AttenMaskCopyInDnFun<hasAtten>(attenMaskUb, srcTensor, runInfo, constInfo, attenMaskInfo, needAtten2,
                                   (constInfo.s1BaseSize >> 1) * runInfo.s2SparseBlk1RealSize, runInfo.sparseBlkIdx2,
                                   runInfo.s2SparseBlk2RealSize);
    attenMaskInQue.template EnQue(attenMaskUb);
}

} // namespace regbaseutil

#endif // QUANT_BLOCK_SPARSE_ATTN_ATTENMASK_H_
