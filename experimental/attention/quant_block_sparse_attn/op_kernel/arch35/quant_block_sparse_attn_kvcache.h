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
 * \file quant_block_sparse_attn_kvcache.h
 * \brief
 */
#ifndef INFER_FLASH_ATTENTION_KVCACHE_H
#define INFER_FLASH_ATTENTION_KVCACHE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "quant_block_sparse_attn_common_arch35.h"
#include "quant_block_sparse_attn_attenmask.h"
using namespace matmul;

TEMPLATE_INTF
__aicore__ inline int64_t CalculateActualS1Size(RunParamStr &, const ConstInfo &, int32_t bIdx,
                                                __gm__ int32_t *cuSeqQlenAddr)
{
    int64_t actualS1Size = 0;
    if constexpr (layout == QBSALayout::TND || layout == QBSALayout::NTD) {
        actualS1Size = cuSeqQlenAddr[bIdx + 1] - cuSeqQlenAddr[bIdx];
    }

    return actualS1Size;
}

TEMPLATE_INTF
__aicore__ inline int64_t CalculateActualS2Size(RunParamStr &, const ConstInfo &, int32_t bIdx,
                                                __gm__ int32_t *seqUsedKvlenAddr)
{
    int64_t actualS2Size = 0;
    if constexpr (layout == QBSALayout::TND || layout == QBSALayout::NTD) {
        actualS2Size = seqUsedKvlenAddr[bIdx];
    }
    return actualS2Size;
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamBatch(RunParamStr &runParam, const ConstInfo &constInfo,
                                         const AttenMaskInfo &attenMaskInfo, GlobalTensor<INPUT_T> &keyGm,
                                         __gm__ int32_t *cuSeqQlenAddr, __gm__ int32_t *seqUsedKvlenAddr)
{
    // 计算实际序列长度
    runParam.actualS1Size =
        CalculateActualS1Size<TEMPLATE_INTF_ARGS>(runParam, constInfo, runParam.boIdx, cuSeqQlenAddr);
    runParam.actualS2Size =
        CalculateActualS2Size<TEMPLATE_INTF_ARGS>(runParam, constInfo, runParam.boIdx, seqUsedKvlenAddr);

    if constexpr (hasAtten) {
        runParam.preTokensPerBatch = attenMaskInfo.preTokens;
        runParam.nextTokensPerBatch = attenMaskInfo.nextTokens;
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            runParam.preTokensPerBatch = SPARSE_MODE_INT_DEFAULT;
            if constexpr (layout == QBSALayout::TND || layout == QBSALayout::NTD) {
                runParam.nextTokensPerBatch = runParam.actualS2Size - runParam.actualS1Size;
            }
        }
    }
    // 计算query偏移量
    if constexpr (layout == QBSALayout::TND) {
        runParam.qBScalarOffset = cuSeqQlenAddr[runParam.boIdx] * constInfo.n2G;
        runParam.qBOffset = runParam.qBScalarOffset * constInfo.dSize;
    } else if constexpr (layout == QBSALayout::NTD) {
        runParam.qBScalarOffset =
            constInfo.t1Size * (runParam.n2oIdx * constInfo.gSize + runParam.goIdx) + cuSeqQlenAddr[runParam.boIdx];
        runParam.qBOffset = runParam.qBScalarOffset * constInfo.dSize;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeS1LoopInfo(RunParamStr &runParam, const ConstInfo &constInfo, uint32_t s1StartIdx,
                                         uint32_t s1EndIdx, bool firstBN, bool lastBN)
{
    uint32_t s1OuterSize = CeilDiv(runParam.actualS1Size, constInfo.s1BaseSize);
    runParam.s1LoopStart = firstBN ? s1StartIdx : 0;
    runParam.s1LoopEnd = lastBN ? (s1EndIdx == 0 ? s1OuterSize : s1EndIdx) : s1OuterSize;

    if constexpr (hasAtten) {
        if (runParam.nextTokensPerBatch < 0) {
            int64_t nextTokensOuterSize = (-runParam.nextTokensPerBatch) / constInfo.s1BaseSize; // 向下取整
            runParam.s1LoopStart = Max(runParam.s1LoopStart, nextTokensOuterSize);
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeSouterParam(RunParamStr &runParam, const ConstInfo &constInfo, uint32_t sOuterLoopIdx)
{
    int64_t cubeSOuterOffset = sOuterLoopIdx * constInfo.s1BaseSize;
    if (runParam.actualS1Size == 0) {
        runParam.s1RealSize = 0;
    } else {
        runParam.s1RealSize = Min(constInfo.s1BaseSize, runParam.actualS1Size - cubeSOuterOffset);
    }
    if constexpr (useDn) {
        runParam.s1RealSizeAlign32 = ((runParam.s1RealSize + 31) >> 5 << 5);
        runParam.halfS1RealSize = runParam.s1RealSize <= 16 ? runParam.s1RealSize : runParam.s1RealSizeAlign32 >> 1;
    } else {
        runParam.halfS1RealSize = (runParam.s1RealSize + 1) >> 1;
    }
    runParam.firstHalfS1RealSize = runParam.halfS1RealSize;
    if (constInfo.subBlockIdx == 1) {
        runParam.halfS1RealSize = runParam.s1RealSize - runParam.halfS1RealSize;
        runParam.sOuterOffset = cubeSOuterOffset + runParam.firstHalfS1RealSize;
    } else {
        runParam.sOuterOffset = cubeSOuterOffset;
    }
    runParam.cubeSOuterOffset = cubeSOuterOffset;
}

TEMPLATE_INTF
__aicore__ inline void LoopSOuterOffsetInit(RunParamStr &runParam, const ConstInfo &constInfo, int32_t bIdx,
                                            __gm__ int32_t *cuSeqQlenAddr)
{
    int64_t seqOffset = cuSeqQlenAddr[bIdx];

    if ASCEND_IS_AIC {
        if constexpr (layout == QBSALayout::TND) {
            runParam.tensorQOffset = runParam.qBOffset + runParam.cubeSOuterOffset * constInfo.n2GD +
                                     runParam.n2oIdx * constInfo.gD + runParam.goIdx * constInfo.dSize;
        } else if constexpr (layout == QBSALayout::NTD) {
            runParam.tensorQOffset = runParam.n2oIdx * constInfo.gSize * constInfo.t1Size * constInfo.dSize +
                                     runParam.goIdx * constInfo.t1Size * constInfo.dSize +
                                     (seqOffset + runParam.cubeSOuterOffset) * constInfo.dSize;
        }
    } else {
        if constexpr (layout == QBSALayout::TND || layout == QBSALayout::NTD) {
            // 起始地址  需要stride偏移
            runParam.attentionOutOffset = seqOffset * constInfo.n2GDv + runParam.sOuterOffset * constInfo.n2GDv +
                                          runParam.n2oIdx * constInfo.gDv + runParam.goIdx * constInfo.dSizeV;
        }
        if constexpr (layout == QBSALayout::TND || layout == QBSALayout::NTD) {
            runParam.softmaxLseOffset = runParam.n1oIdx * constInfo.t1Size + seqOffset + runParam.sOuterOffset;
        }
    }
}

TEMPLATE_INTF
__aicore__ inline int64_t ClipSInnerTokenCube(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamS1(RunParamStr &runParam, const ConstInfo &constInfo, uint32_t sOuterLoopIdx,
                                      __gm__ int32_t *cuSeqQlenAddr)
{
    // 后续的函数依赖 sOuterOffset
    ComputeSouterParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, sOuterLoopIdx);
    LoopSOuterOffsetInit<TEMPLATE_INTF_ARGS>(runParam, constInfo, runParam.boIdx, cuSeqQlenAddr);
}

TEMPLATE_INTF
__aicore__ inline bool ComputeS2LoopInfo(RunParamStr &runParam, const ConstInfo &constInfo,
                                         GlobalTensor<int32_t> &sparseSeqLenGm, int64_t s1Index)
{
    int64_t sparseSeqLenFlatIdx = (runParam.boIdx * constInfo.n1Size + runParam.n1oIdx) * constInfo.maxQb + s1Index;
    runParam.actSparseLen = sparseSeqLenGm.GetValue(sparseSeqLenFlatIdx);
    runParam.s2LoopEndIdx = (runParam.actSparseLen + 2 - 1) >> 1; // sparse块为128  基本块为256
    return runParam.actSparseLen <= 0;
}

TEMPLATE_INTF
__aicore__ inline void InitTaskParamByRun(const RunParamStr &runParam, RunInfo &runInfo)
{
    runInfo.boIdx = runParam.boIdx;
    runInfo.preTokensPerBatch = runParam.preTokensPerBatch;
    runInfo.nextTokensPerBatch = runParam.nextTokensPerBatch;
    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.softmaxLseOffset = runParam.softmaxLseOffset;
}

__aicore__ inline void UpdateLoopFlagsBasedOnS1Index(int32_t extras1, bool &notLast, bool &notLastTwoLoop,
                                                     bool &notLastThreeLoop)
{
    switch (extras1) {
        case -1:
            break;
        case 0:
            notLastThreeLoop = false;
            break;
        case 1:
            notLastThreeLoop = false;
            notLastTwoLoop = false;
            break;
        case 2:
            notLast = false;
            notLastThreeLoop = false;
            notLastTwoLoop = false;
            break;
        default:
            break;
    }
}
__aicore__ inline void AttentionMaskFullProcessingOrRequired(int64_t s1Offset, int64_t s2Offset, int64_t delta,
                                                             uint32_t realSize, int64_t &sparseBlkIdx,
                                                             bool &sparseBlkPartialMask)
{
    int64_t offsetDiff = s1Offset - s2Offset + delta;
    if (offsetDiff + realSize - 1 < 0) {
        // 情景三：该 sparse block 全部被 mask 掉。DN 流水仍会用 sparseBlkIdx 做 PA GM 搬运，
        // 因此不能写成 -1；保留原 index，由 mask 阶段把该块全部置无效。
        sparseBlkPartialMask = true;
        return;
    }
    if (offsetDiff - realSize + 1 < 0) {
        // 情景二
        sparseBlkPartialMask = true;
    } else {
        // 情景一
        sparseBlkPartialMask = false;
    }
}

template <typename T>
__aicore__ inline void SwapValues(T &numValue1, T &numValue2)
{
    T tempValue = numValue1;
    numValue1 = numValue2;
    numValue2 = tempValue;
}

__aicore__ inline void SwapSparseBlockSlots(RunInfo &runInfo)
{
    SwapValues(runInfo.sparseBlkIdx1, runInfo.sparseBlkIdx2);
    SwapValues(runInfo.s2SparseBlk1RealSize, runInfo.s2SparseBlk2RealSize);
    SwapValues(runInfo.s2SparseBlk1RealAlignedSize, runInfo.s2SparseBlk2RealAlignedSize);
    SwapValues(runInfo.sparseBlk1PartialMask, runInfo.sparseBlk2PartialMask);
}

__aicore__ inline void IdxSortBySparseIdx(RunInfo &runInfo)
{
    if (unlikely(runInfo.sparseBlkIdx1 == -1 && runInfo.sparseBlkIdx2 == -1)) {
        return;
    }
    if (unlikely(runInfo.sparseBlkIdx1 == -1)) {
        SwapSparseBlockSlots(runInfo);
        return;
    }
    if (unlikely(runInfo.sparseBlkIdx2 == -1)) {
        return;
    }
    if (runInfo.sparseBlkIdx1 > runInfo.sparseBlkIdx2) {
        SwapSparseBlockSlots(runInfo);
    }
}

#endif // INFER_FLASH_ATTENTION_KVCACHE_H
