/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Phase2 combine epilogue for prefill (Arch35) is the BlockEpilogue specialization
 * of EpilogueRescaleOSplitKvArch35 in this header.
 */

#ifndef BLOCK_EPILOGUE_HPP
#define BLOCK_EPILOGUE_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "kernel_operator.h"

using namespace AscendC;
using namespace AscendC::Reg;

#include "../../../../../common/op_kernel/arch35/vf/vf_flash_decode.h"
// Local copy of the library ReduceFinalRes_VF family with the per-row regbase Div
// embedded (this operator's Phase1 leaves accumOut unnormalized). FP32-only path
// (<T, Tacc> with Tacc=float; the bf16 O_partial path uses the separate header
// below, do NOT add bf16 branches here — keeps the verified fp32 binary identical).
// Must follow the vf_flash_decode.h include above (relies on FLT_ZERO / MicroAPI).
#include "../../../arch35/vector_api/vf_flash_decode_msa.h"

namespace NpuArch::Epilogue::Block {

template <class DispatchPolicy, class... Args>
class BlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

} // namespace NpuArch::Epilogue::Block

#include "../../../attn_infra/epilogue/block/block_epilogue_online_softmax_arch35_reg_low_prec_bf16.hpp"
#include "../../../attn_infra/epilogue/block/block_epilogue_online_softmax_arch35_reg_high_prec.hpp"

namespace NpuArch::Epilogue::Block {
template <class OutDtype, class Tws, class ElementKV>
class BlockEpilogue<EpilogueRescaleOSplitKvArch35, OutDtype, Tws, ElementKV> {
public:
    using DispatchPolicy = EpilogueRescaleOSplitKvArch35;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = OutDtype;
    using T = float; // lse / rowSum / dst / cast compute (always fp32)
    // O_partial (workspace accumOut) dtype: float (fp32 path) or bfloat16_t (innerPrecise==1
    // bf16 path: PV fixpipe F322BF16 writes bf16, Phase2 MTE2 reads bf16, regbase-cast to fp32).
    using TwsO = Tws;

    static constexpr uint32_t FP32_ONE_BLOCK_SIZE = 8U;
    static constexpr uint32_t BYTE_BLOCK = 32U;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768U;
    static constexpr uint32_t BUF_32K_FLOATS = BUFFER_SIZE_BYTE_32K / sizeof(T);
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192U;
    static constexpr uint32_t BUF_8K_FLOATS = BUFFER_SIZE_BYTE_8K / sizeof(T);
    static constexpr float NEG_INF_LSE = -3.4028235e38f;

    // accumOut ping-pong (mirrors the kernel ubSTensor[ubSBufId] style): a 2-deep array of
    // UB tensors indexed by bufId = g % ACCUM_OUT_STAGES (one ping-pong slot per buf-group,
    // each holding splitsPerBuf_ contiguous splits), each a strided stage, with event
    // ids derived as ACCUM_OUT_FLAG_BASE + bufId for both MTE2_V and V_MTE2 (distinct
    // HardEvent types share an id safely, per the IFA fdMm2ResBuf1/2 pattern).
    static constexpr uint32_t ACCUM_OUT_STAGES = 2U;
    static constexpr uint32_t ACCUM_OUT_FLAG_BASE = 3U; // MTE2_V/V_MTE2 id = base + bufId

    __aicore__ inline BlockEpilogue() {}

    __aicore__ inline void InitFDBuffers(AscendC::LocalTensor<T> &ubBase, uint32_t D, uint32_t groupSize, uint32_t topK,
                                         uint32_t kvHeads, uint32_t softmaxLseFlag, uint32_t layoutType,
                                         uint32_t qSeqLen, uint32_t numHeads)
    {
        D_ = D;
        groupSize_ = groupSize;
        topK_ = topK;
        kvHeads_ = kvHeads;
        softmaxLseFlag_ = softmaxLseFlag;
        layoutType_ = layoutType;
        qSeqLen_ = qSeqLen;
        numHeads_ = numHeads;
        headDimAlignFp32_ = RoundUp(D_, FP32_ONE_BLOCK_SIZE);

        // E: batched accumOut read. One 32K-float UB slot holds cap splits
        // (split = groupSize * headDimAlignFp32 floats). Supported range
        // (groupSize<=16, D<=512 => split<=8192 floats) keeps cap>=4. Aim for ~2
        // buf-groups so the 2-stage ping-pong still overlaps MTE2(g+1) || V(g) at
        // coarser grain (was per-split). Capped by cap, floored at 1.
        uint32_t cap = BUF_32K_FLOATS / (groupSize_ * headDimAlignFp32_);
        if (cap == 0U) {
            cap = 1U;
        }
        uint32_t half = (topK_ + 1U) / 2U;
        splitsPerBuf_ = (half < 1U) ? 1U : ((cap < half) ? cap : half);

        // UB layout: contiguous, no holes. lse buffers capped at 8K (the [topK, dealRowCount, 8]
        // broadcast layout fits in 8KiB for topK, groupSize <= 16); the rest are 32K slots.
        lseMaxBuf_ = ubBase;                // 8K
        lseSumBuf_ = ubBase[BUF_8K_FLOATS]; // 8K
        uint32_t off = BUF_8K_FLOATS * 2U;
        // dst slot carved as a 32K-float (32KB) stride; for the bf16 path TwsO=bf16, so the
        // raw float carve is reinterpreted (bf16 dst uses <=4KB of the 32KB — slack is free).
        dstBuf_ = ubBase[off].template ReinterpretCast<TwsO>();
        off += BUF_32K_FLOATS; // 32K
        broadCastTmpBuf_ = ubBase[off];
        off += BUF_32K_FLOATS; // 32K (Broadcast scratch + lse staging)
        // Separate bf16 cast-output buffer so CopyFinalResOut's Cast is out-of-place.
        castOutBuf_ = ubBase[off];
        off += BUF_32K_FLOATS; // 32K
        // accumOut ping-pong (kernel ubSTensor style): 2 stages, uniform 32K-slot stride.
        // For the bf16 path the stage is ReinterpretCast<TwsO> over the same 32K-float carve
        // (bf16 stage holds <=16KB of data, the 32KB carve leaves slack — UB only gets looser).
        for (uint32_t i = 0U; i < ACCUM_OUT_STAGES; ++i) {
            accumOutBuf[i] = ubBase[off + BUF_32K_FLOATS * i].template ReinterpretCast<TwsO>();
        }
        // Dedicated LSE staging, after existing 176KB layout so flag-off offsets stay unchanged.
        lseOutBuf_ = ubBase[off + BUF_32K_FLOATS * ACCUM_OUT_STAGES];
    }

    __aicore__ inline void FlashDecodeCompute(uint32_t tmpBlockIdx, uint32_t totalTaskNumP2,
                                              AscendC::GlobalTensor<ElementO> &gmO,
                                              AscendC::GlobalTensor<TwsO> &gmAccumOut,
                                              AscendC::GlobalTensor<T> &gmSoftmaxMax,
                                              AscendC::GlobalTensor<T> &gmSoftmaxSum,
                                              AscendC::GlobalTensor<T> &gmSoftmaxLse, uint32_t numHeads,
                                              uint32_t kvHeads)
    {
        if (tmpBlockIdx >= totalTaskNumP2) {
            return;
        }
        (void)numHeads;

        uint32_t qToken = tmpBlockIdx / kvHeads;
        uint32_t kvHeadIdx = tmpBlockIdx % kvHeads;
        actualCombineLoopSize_ = topK_;
        CombineSplitKVRes(qToken, kvHeadIdx, gmO, gmAccumOut, gmSoftmaxMax, gmSoftmaxSum, gmSoftmaxLse);
    }

private:
    AscendC::LocalTensor<T> lseMaxBuf_;
    AscendC::LocalTensor<T> lseSumBuf_;
    // dst (combined O) dtype = TwsO: float for the fp32 path, bfloat16_t for the
    // innerPrecise==1 bf16 path (regbase-fused reduce stores bf16 directly here;
    // CopyFinalResOut then plain-copies bf16 to GM O, no cast).
    AscendC::LocalTensor<TwsO> dstBuf_;
    // O_partial ping-pong slot dtype = TwsO (float for fp32 path, bf16 for innerPrecise==1).
    AscendC::LocalTensor<TwsO> accumOutBuf[ACCUM_OUT_STAGES];
    AscendC::LocalTensor<T> broadCastTmpBuf_;
    AscendC::LocalTensor<T> castOutBuf_;
    AscendC::LocalTensor<T> lseOutBuf_;

    uint32_t D_ = 0;
    uint32_t groupSize_ = 0;
    uint32_t topK_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t softmaxLseFlag_ = 0;
    uint32_t layoutType_ = 0;
    uint32_t qSeqLen_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t headDimAlignFp32_ = 0;
    uint32_t actualCombineLoopSize_ = 0;
    uint32_t splitsPerBuf_ = 0; // E: splits read per ping-pong slot (batched contiguous read)

    __aicore__ inline uint64_t QHeadGmOffset(uint32_t qToken, uint32_t qHead) const
    {
        if (layoutType_ == 1U && qSeqLen_ > 0U) { // LAYOUT_BNSD: heads of one token are S apart.
            uint32_t b = qToken / qSeqLen_;
            uint32_t t = qToken - b * qSeqLen_;
            return (static_cast<uint64_t>(b) * numHeads_ + qHead) * qSeqLen_ * D_ + static_cast<uint64_t>(t) * D_;
        }
        return static_cast<uint64_t>(qToken) * numHeads_ * D_ + static_cast<uint64_t>(qHead) * D_;
    }

    __aicore__ inline uint64_t LseGmOffset(uint32_t qToken, uint32_t qHead) const
    {
        if (layoutType_ == 1U && qSeqLen_ > 0U) { // LAYOUT_BNSD: heads of one token are S apart.
            uint32_t b = qToken / qSeqLen_;
            uint32_t t = qToken - b * qSeqLen_;
            return (static_cast<uint64_t>(b) * numHeads_ + qHead) * qSeqLen_ + static_cast<uint64_t>(t);
        }
        return static_cast<uint64_t>(qToken) * numHeads_ + qHead;
    }

    __aicore__ inline static uint32_t RoundUp(uint32_t a, uint32_t b)
    {
        return (a + b - 1U) / b * b;
    }

    __aicore__ inline uint64_t TaskStatBase(uint32_t qToken, uint32_t kvHeadIdx) const
    {
        return (static_cast<uint64_t>(qToken) * kvHeads_ + kvHeadIdx) * topK_ * groupSize_;
    }

    __aicore__ inline uint64_t TaskAccumBase(uint32_t qToken, uint32_t kvHeadIdx) const
    {
        return (static_cast<uint64_t>(qToken) * kvHeads_ + kvHeadIdx) * topK_ * groupSize_ * D_;
    }

    // GM max/sum compact [topK, groupSize] -> UB IFA layout [topK, dealRow, 8].
    __aicore__ inline void CopyLseIn(uint32_t qToken, uint32_t kvHeadIdx, uint32_t dealRowCount,
                                     AscendC::GlobalTensor<T> &gmSoftmaxMax, AscendC::GlobalTensor<T> &gmSoftmaxSum,
                                     AscendC::LocalTensor<T> &lseMaxLocal, AscendC::LocalTensor<T> &lseSumLocal)
    {
        uint32_t ubRowStride = dealRowCount * FP32_ONE_BLOCK_SIZE;
        auto lseCount = dealRowCount * topK_;
        auto alignedLseCount = RoundUp(lseCount, FP32_ONE_BLOCK_SIZE);

        AscendC::LocalTensor<T> lseMaxTmpBuf = broadCastTmpBuf_;
        AscendC::LocalTensor<T> lseSumTmpBuf = broadCastTmpBuf_[alignedLseCount];
        AscendC::LocalTensor<uint8_t> broadCastTmpBuf =
            broadCastTmpBuf_[alignedLseCount * 2U].template ReinterpretCast<uint8_t>();
        uint64_t taskBase = TaskStatBase(qToken, kvHeadIdx);
        DataCopyParams copyParams{
            1,                                  // blockCount
            uint16_t(lseCount * sizeof(float)), // blockLen
            0,                                  // srcStride
            0,                                  // dstStride
        };
        DataCopyPadParams copyPadParams{
            false, // isPad
            0,     // leftPadding
            0,     // rightPadding
            0      // paddingValue
        };
        AscendC::DataCopyPad(lseMaxTmpBuf, gmSoftmaxMax[taskBase], copyParams, copyPadParams);
        AscendC::DataCopyPad(lseSumTmpBuf, gmSoftmaxSum[taskBase], copyParams, copyPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        const uint32_t srcShape[2] = {lseCount, 1};
        const uint32_t dstShape[2] = {lseCount, FP32_ONE_BLOCK_SIZE};
        AscendC::Broadcast<T, 2, 1>(lseMaxLocal, lseMaxTmpBuf, dstShape, srcShape, broadCastTmpBuf);
        AscendC::Broadcast<T, 2, 1>(lseSumLocal, lseSumTmpBuf, dstShape, srcShape, broadCastTmpBuf);
        // Div->Mul: precompute 1/rowSum in place on the compact lseSumTmpBuf. The 8x
        // broadcast lseSumBuf_ (read by ComputeScaleValue for the per-split scale) was
        // just written above and is a SEPARATE buffer, so the original sum values feeding
        // scale stay intact; only the compact per-row rowSum that ReduceFinalRes_VF reads
        // as rowSumLocal becomes its reciprocal, turning the per-(row,split,z) Div into a
        // Mul. Broadcast (V) reads lseSumTmpBuf before this in-place Reciprocal (same V
        // pipe, in-order). Invalid slots have rowSum==0 (WS_ROWSUM_INIT) -> reciprocal=+inf,
        // never used (combineDone breaks before the reduce reaches them).
        AscendC::Reciprocal<T>(lseSumTmpBuf, lseSumTmpBuf, static_cast<int32_t>(alignedLseCount));
        if constexpr (AscendC::IsSameType<ElementKV, fp8_e4m3fn_t>::value) {
            AscendC::Muls<T>(lseSumTmpBuf, lseSumTmpBuf, 1.0f / 448.0f, static_cast<int32_t>(alignedLseCount));
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeScaleValue(AscendC::LocalTensor<T> &lseMaxLocal, AscendC::LocalTensor<T> &lseSumLocal,
                                             uint32_t dealRowCount, uint32_t qToken, uint32_t kvHeadIdx,
                                             AscendC::GlobalTensor<T> &gmSoftmaxLse)
    {
        AscendC::LocalTensor<bfloat16_t> tmpSinkUb;
        bool learnableSinkFlag = false;
        if (softmaxLseFlag_ == 1U) {
            FaVectorApi::ComputeScaleValue_VF(tmpSinkUb, lseMaxLocal, lseSumLocal, lseOutBuf_, dealRowCount,
                                              actualCombineLoopSize_, true, learnableSinkFlag);
            AscendC::PipeBarrier<PIPE_V>();
            CopySoftmaxLseOut(qToken, kvHeadIdx, dealRowCount, gmSoftmaxLse, lseOutBuf_);
        } else {
            // Flag-off: same dummy lseOutputUb as the original verified path.
            FaVectorApi::ComputeScaleValue_VF(tmpSinkUb, lseMaxLocal, lseSumLocal,
                                              accumOutBuf[0].template ReinterpretCast<T>(), dealRowCount,
                                              actualCombineLoopSize_, false, learnableSinkFlag);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // TND softmax LSE GM layout [T, N, 1]: one fp32 per (qToken, qHead).
    // UB from ComputeScaleValue_VF is [dealRowCount, 8] (32B-aligned rows).
    // DataCopyPad blockLen=4, srcStride=0 copies the first float of each 32B row
    // (same as IFA FlashDecode / DataCopySoftmaxLseTNDArch35).
    __aicore__ inline void CopySoftmaxLseOut(uint32_t qToken, uint32_t kvHeadIdx, uint32_t dealRowCount,
                                             AscendC::GlobalTensor<T> &gmSoftmaxLse,
                                             AscendC::LocalTensor<T> &lseOutputUb)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        if (layoutType_ == 1U && qSeqLen_ > 0U) { // LAYOUT_BNSD: heads of one token are S apart.
            // BNSD [B, N, S, 1]: consecutive GQA heads of one token are S apart.
            for (uint32_t gh = 0U; gh < dealRowCount; ++gh) {
                AscendC::DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = sizeof(float);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                uint64_t dst = LseGmOffset(qToken, kvHeadIdx * groupSize_ + gh);
                AscendC::DataCopyPad(gmSoftmaxLse[dst], lseOutputUb[gh * FP32_ONE_BLOCK_SIZE], copyParams);
            }
            return;
        }
        uint64_t lseOffset = LseGmOffset(qToken, kvHeadIdx * groupSize_);
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = dealRowCount;
        copyParams.blockLen = sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(gmSoftmaxLse[lseOffset], lseOutputUb, copyParams);
    }

    __aicore__ inline void CopyAccumOutIn(uint32_t qToken, uint32_t kvHeadIdx, uint32_t jStart, uint32_t nSplits,
                                          uint32_t dealRowCount, AscendC::GlobalTensor<TwsO> &gmAccumOut,
                                          AscendC::LocalTensor<TwsO> &accumOutLocal)
    {
        // E: batched contiguous read of nSplits splits in one DataCopyPad (was one
        // 8KB DataCopyPad per split). GM layout [totalQ, kvHead, topK, groupSize, D]
        // is contiguous over [jStart..jStart+nSplits) (topK slowest, D fastest,
        // srcStride 0 => rows back-to-back), so the whole span is one transaction.
        // O_partial dtype = TwsO (float fp32 path, bf16 innerPrecise==1 path). blockElems
        // is the 32B-block element count (8 fp32 = FP32_ONE_BLOCK_SIZE, 16 bf16); for the
        // fp32 path blockElems==FP32_ONE_BLOCK_SIZE so the params are byte-identical to prior.
        constexpr uint32_t blockElems = BYTE_BLOCK / sizeof(TwsO);
        AscendC::DataCopyExtParams copyInParams;
        AscendC::DataCopyPadExtParams<TwsO> copyInPadParams;
        copyInParams.blockCount = nSplits * dealRowCount;
        copyInParams.blockLen = static_cast<uint32_t>(D_) * sizeof(TwsO);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (headDimAlignFp32_ - D_) / blockElems;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (headDimAlignFp32_ - D_) % blockElems;
        copyInPadParams.paddingValue = 0;

        uint64_t oOffset = TaskAccumBase(qToken, kvHeadIdx) + static_cast<uint64_t>(jStart) * groupSize_ * D_;
        AscendC::DataCopyPad(accumOutLocal, gmAccumOut[oOffset], copyInParams, copyInPadParams);
    }

    __aicore__ inline void ReduceFinalRes(uint32_t qToken, uint32_t kvHeadIdx, AscendC::LocalTensor<TwsO> &dst,
                                          AscendC::LocalTensor<T> &scaleLocal, uint32_t dealRowCount,
                                          AscendC::GlobalTensor<TwsO> &gmAccumOut)
    {
        // Slot validity check from UB (no GM round-trip). After CopyLseIn +
        // ComputeScaleValue, lseMaxBuf_ holds the per-slot rowMax in broadcast
        // layout [topK, dealRowCount, FP32_ONE_BLOCK_SIZE] (topK outer, each
        // scalar replicated 8x). ComputeScaleValue_8_VF reads lseMax but does not
        // overwrite it, so the values are intact. InitWorkspaceStats prefills
        // -FLT_MAX (NEG_INF_LSE) for slots Phase1 never wrote; a valid slot's
        // rowMax is a finite score-derived value, so firstRowMax <= NEG_INF_LSE
        // discriminates the whole slot. Valid slots are left-contiguous in the
        // workspace (Phase1 writes slotK = 0..valid_k-1 per qToken/kvHead, since
        // build_k2q_csr assigns slot_flat = arange(topk) only to valid entries),
        // so the first invalid slot means every later slot is invalid too: break
        // once instead of continue. Skipping CopyAccumOutIn + ReduceFinalRes_VF
        // for invalid splits avoids Divs-by-zero (rowSum==0) and stale-UB
        // accumulation; scale[j]==0 from ComputeScaleValue makes them 0 anyway.
        // The per-row rowSum for the regbase Div is reused from the compact rowSum
        // already staged by CopyLseIn into broadCastTmp_ (lseSumTmpBuf =
        // broadCastTmpBuf_[alignedLseCount], layout [topK, dealRowCount]); no GM
        // re-read or SetValue here.
        uint32_t lseCount = dealRowCount * topK_;
        uint32_t alignedLseCount = RoundUp(lseCount, FP32_ONE_BLOCK_SIZE);

        // Double-buffered (ping-pong) accumOut, managed like the kernel's ubSTensor[ubSBufId]:
        // a 2-deep array accumOutBuf[curId], curId = g % ACCUM_OUT_STAGES, with event ids
        // ACCUM_OUT_FLAG_BASE + curId for both MTE2_V (forward) and V_MTE2 (reverse) — distinct
        // HardEvent types share an id safely (IFA fdMm2ResBuf1/2 does the same).
        //
        // E: the ping-pong grain moved from per-split j to per buf-group g. Each group
        // loads splitsPerBuf_ (or the tail) splits in ONE contiguous DataCopyPad
        // (CopyAccumOutIn), then the V reduce walks those splits from UB. The MTE2 load
        // of group g+1 overlaps the V reduce of group g (still 2-stage, still overlapped);
        // only the transaction size grew (8KB -> ~splitsPerBuf_*8KB) so per-transaction
        // overhead drops ~splitsPerBuf_-fold and effective MTE2 BW rises. For topK=1 this
        // degenerates to a single load (no overlap, nothing to overlap with).
        //
        // Validity is checked in the inner split loop (one scalar GetValue per split,
        // issued alongside the V reduce so it hides under the V pipe, not pacing it).
        // Valid slots are left-contiguous (Phase1 writes slotK = 0..valid_k-1; slots
        // beyond are prefilled -FLT_MAX = NEG_INF_LSE by InitWorkspaceStats, whose
        // WS_ROWSUM_INIT=0 makes an unreduced invalid split a Div-by-zero NaN risk), so
        // the first invalid slot means every later slot is invalid too: break out of the
        // whole combine there. The batch load already pulled this group's splits (incl.
        // the rare invalid tail straddling valid_k); we simply don't reduce past the first
        // invalid. The extra copied data is <= one partial group and only when valid_k <
        // topK (causal head / sparse tail) — rare in prefill, accepted.

        // Prime each stage's V_MTE2: a stage's first reuse has no prior V reduce to drain.
        for (uint32_t s = 0U; s < ACCUM_OUT_STAGES; ++s) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + s);
        }

        uint32_t numGroups = (topK_ + splitsPerBuf_ - 1U) / splitsPerBuf_;
        bool combineDone = false;
        for (uint32_t g = 0U; (g < numGroups) && !combineDone; ++g) {
            uint32_t jStart = g * splitsPerBuf_;
            uint32_t nSplits = splitsPerBuf_;
            uint32_t remain = topK_ - jStart;
            if (nSplits > remain) {
                nSplits = remain;
            }
            uint32_t curId = g % ACCUM_OUT_STAGES;
            // stage drained (reduce g-2, or primed) -> MTE2 may overwrite it.
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + curId);
            CopyAccumOutIn(qToken, kvHeadIdx, jStart, nSplits, dealRowCount, gmAccumOut, accumOutBuf[curId]);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ACCUM_OUT_FLAG_BASE + curId);  // load g || V reduce g-1
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ACCUM_OUT_FLAG_BASE + curId); // load g done
            // V reduce each split in this group: accumOut base is the in-buf offset
            // (j-jStart)*groupSize*headDimAlignFp32; j stays the global split index so
            // lse (lseSumBuf_) and rowSum (broadCastTmpBuf_[alignedLseCount+j*dealRowCount])
            // and the _0/_Rest (j==0 init vs load prior dst) dispatches are unchanged.
            // First invalid (left-contiguous) breaks the whole combine.
            for (uint32_t j = jStart; j < jStart + nSplits; ++j) {
                uint32_t maxOff = j * dealRowCount * FP32_ONE_BLOCK_SIZE;
                if (lseMaxBuf_[maxOff].GetValue(0) <= NEG_INF_LSE) {
                    combineDone = true;
                    break;
                }
                uint64_t localOff = static_cast<uint64_t>(j - jStart) * groupSize_ * headDimAlignFp32_;
                uint64_t rowSumOff = static_cast<uint64_t>(alignedLseCount) + static_cast<uint64_t>(j) * dealRowCount;
                AscendC::LocalTensor<TwsO> accumSplit = accumOutBuf[curId][localOff];
                AscendC::LocalTensor<T> rowSumLocal = broadCastTmpBuf_[rowSumOff];
                // bf16 path: regbase-fused reduce (vf_flash_decode_msa.h::ReduceFinalRes_VF_BF16).
                // Reads bf16 accumSplit straight into a reg, fuses Cast bf16->fp32 (ZERO/ONE),
                // fp32 Mul/Mul/Add, Cast back to bf16 (ZERO/ONE), Or-merge, vsstb store bf16
                // directly to dst (TwsO=bf16). No vector Cast staging pass / no castOutBuf_ staging
                // here (castOutBuf_ stays free for the fp32 path's CopyFinalResOut). dst is bf16 so
                // the cross-split accumulation is bf16-rounded per split (innerPrecise==1 INNER_LOW).
                // fp32 path calls the fp32 reduce directly (byte-identical to the verified binary).
                if constexpr (AscendC::IsSameType<TwsO, bfloat16_t>::value) {
                    FaVectorApiSplitKv::ReduceFinalRes_VF_BF16<TwsO>(dst, scaleLocal, accumSplit, rowSumLocal,
                                                                     dealRowCount,
                                                                     static_cast<uint64_t>(headDimAlignFp32_), j);
                } else {
                    FaVectorApiSplitKv::ReduceFinalRes_VF<T>(dst, scaleLocal, accumSplit, rowSumLocal, dealRowCount,
                                                             static_cast<uint64_t>(headDimAlignFp32_), j);
                }
            }
            // V done reading accumOutBuf[curId] (and dst store committed; SetFlag<V_MTE2> is a V
            // completion barrier, ordering dst V-store->V-load across iterations) -> stage reusable.
            // ALWAYS executed, even when the inner loop broke: this group already did
            // Wait<V_MTE2> above, so it must replenish with a Set or the trailing drain
            // would underflow (the broken group's Wait consumed a primed/prior Set).
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + curId);
        }
        // Trailing drain: consume the dangling V_MTE2 per stage (last reduce's set + the primed
        // twin that no later load reached) so it doesn't stale-signal the next chunk/task. MTE2_V
        // is set+waited 1:1 within each iter, so no MTE2_V trailing drain.
        for (uint32_t s = 0U; s < ACCUM_OUT_STAGES; ++s) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + s);
        }
    }

    __aicore__ inline void CopyFinalResOut(uint32_t qToken, uint32_t kvHeadIdx, AscendC::GlobalTensor<ElementO> &gmO,
                                           AscendC::LocalTensor<TwsO> &accumOutLocal, uint32_t dealRowCount)
    {
        AscendC::PipeBarrier<PIPE_V>();
        const bool isBnsd = (layoutType_ == 1U && qSeqLen_ > 0U); // LAYOUT_BNSD only; BSND is contiguous like TND.
        if constexpr (AscendC::IsSameType<TwsO, bfloat16_t>::value) {
            uint32_t shapeArray[] = {dealRowCount, D_};
            accumOutLocal.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray, AscendC::DataFormat::ND));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            if (isBnsd) {
                for (uint32_t gh = 0U; gh < dealRowCount; ++gh) {
                    AscendC::DataCopyExtParams dataCopyParams;
                    dataCopyParams.blockCount = 1;
                    dataCopyParams.blockLen = static_cast<uint32_t>(D_) * sizeof(ElementO);
                    dataCopyParams.srcStride = 0;
                    dataCopyParams.dstStride = 0;
                    uint64_t off = QHeadGmOffset(qToken, kvHeadIdx * groupSize_ + gh);
                    AscendC::DataCopyPad(gmO[off], accumOutLocal[gh * headDimAlignFp32_], dataCopyParams);
                }
            } else {
                uint64_t attenOutOffset = QHeadGmOffset(qToken, kvHeadIdx * groupSize_);
                AscendC::DataCopyExtParams dataCopyParams;
                dataCopyParams.blockCount = dealRowCount;
                dataCopyParams.blockLen = static_cast<uint32_t>(D_) * sizeof(ElementO);
                dataCopyParams.srcStride = (headDimAlignFp32_ - D_) / (BYTE_BLOCK / sizeof(ElementO));
                dataCopyParams.dstStride = 0;
                AscendC::DataCopyPad(gmO[attenOutOffset], accumOutLocal, dataCopyParams);
            }
        } else {
            AscendC::LocalTensor<ElementO> castBuf = castOutBuf_.template ReinterpretCast<ElementO>();
            uint32_t shapeArray[] = {dealRowCount, D_};
            castBuf.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray, AscendC::DataFormat::ND));
            AscendC::Cast(castBuf, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * headDimAlignFp32_);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            if (isBnsd) {
                for (uint32_t gh = 0U; gh < dealRowCount; ++gh) {
                    AscendC::DataCopyExtParams dataCopyParams;
                    dataCopyParams.blockCount = 1;
                    dataCopyParams.blockLen = static_cast<uint32_t>(D_) * sizeof(ElementO);
                    dataCopyParams.srcStride = 0;
                    dataCopyParams.dstStride = 0;
                    uint64_t off = QHeadGmOffset(qToken, kvHeadIdx * groupSize_ + gh);
                    AscendC::DataCopyPad(gmO[off], castBuf[gh * headDimAlignFp32_], dataCopyParams);
                }
            } else {
                uint64_t attenOutOffset = QHeadGmOffset(qToken, kvHeadIdx * groupSize_);
                AscendC::DataCopyExtParams dataCopyParams;
                dataCopyParams.blockCount = dealRowCount;
                dataCopyParams.blockLen = static_cast<uint32_t>(D_) * sizeof(ElementO);
                dataCopyParams.srcStride = (headDimAlignFp32_ - D_) / (BYTE_BLOCK / sizeof(ElementO));
                dataCopyParams.dstStride = 0;
                AscendC::DataCopyPad(gmO[attenOutOffset], castBuf, dataCopyParams);
            }
        }
    }

    __aicore__ inline void CombineSplitKVRes(uint32_t qToken, uint32_t kvHeadIdx, AscendC::GlobalTensor<ElementO> &gmO,
                                             AscendC::GlobalTensor<TwsO> &gmAccumOut,
                                             AscendC::GlobalTensor<T> &gmSoftmaxMax,
                                             AscendC::GlobalTensor<T> &gmSoftmaxSum,
                                             AscendC::GlobalTensor<T> &gmSoftmaxLse)
    {
        // groupSize <= 16 fits every UB buffer (lse 8K cap, dst/accumOut/castOut 32K slots)
        // for topK <= 16, D <= 512, so a single chunk covers the whole group.
        CopyLseIn(qToken, kvHeadIdx, groupSize_, gmSoftmaxMax, gmSoftmaxSum, lseMaxBuf_, lseSumBuf_);
        ComputeScaleValue(lseMaxBuf_, lseSumBuf_, groupSize_, qToken, kvHeadIdx, gmSoftmaxLse);
        AscendC::LocalTensor<TwsO> dst = dstBuf_;
        ReduceFinalRes(qToken, kvHeadIdx, dst, lseSumBuf_, groupSize_, gmAccumOut);
        CopyFinalResOut(qToken, kvHeadIdx, gmO, dst, groupSize_);
    }
};

} // namespace NpuArch::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP
