/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Prefill PV matmul (KV-centric). L0C fixpipe once to GM O_partial (no dual-subcore split).
 */
#ifndef GEMM_BLOCK_MMAD_PV_SPLIT_KV_ARCH35_HPP
#define GEMM_BLOCK_MMAD_PV_SPLIT_KV_ARCH35_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/arch/cross_core_sync.hpp"
#include "../../../attn_infra/coord.hpp"
#include "../../../attn_infra/gemm/dispatch_policy.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_mmad.hpp"
#include "../../../attn_infra/layout/layout.hpp"
#include "../../../tla/layout.hpp"
#include "../../../tla/tensor.hpp"

namespace NpuArch::Gemm::Block {

template <class ElementP_, class ElementV_, class ElementO_>
struct BlockMmadPVSplitKvArch35 {
    using DispatchPolicy = MmadAtlasA5SplitKvPV;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementP = ElementP_;
    using ElementV = ElementV_;
    using ElementO = ElementO_;
    using ElementAccumulator = float;

    using LayoutTagP = layout::zN;
    using LayoutTagV = layout::RowMajor;
    using LayoutTagO = layout::RowMajor;

    using TileCopy =
        Gemm::Tile::PackedTileCopyTla<ArchTag, ElementP, LayoutTagP, ElementV, LayoutTagV, ElementO, LayoutTagO, void>;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    using TileMmad = Gemm::Tile::TileMmadTla<ArchTag, ElementP, LayoutTagL1A>;

    static constexpr uint32_t L0_STAGES = 2;
    static constexpr uint32_t L0_TILE_M = 128;
    static constexpr uint32_t L0_TILE_N = 128;
    static constexpr uint32_t L0_TILE_K = 128;
    static constexpr uint32_t L0A_BUF_SIZE = ArchTag::L0A_SIZE / L0_STAGES;
    static constexpr uint32_t L0B_BUF_SIZE = ArchTag::L0B_SIZE / L0_STAGES;
    static constexpr uint32_t L0C_HALF_SIZE = ArchTag::L0C_SIZE / 2;
    static constexpr uint32_t L0C_BUF_SIZE = L0C_HALF_SIZE / L0_STAGES;
    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16;
    static constexpr uint32_t COPY_GRANULARITY = 2;
    static constexpr uint32_t C0_SIZE = 16;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementP);

    AscendC::LocalTensor<ElementV> l1VTensor_;
    AscendC::LocalTensor<ElementP> l1PTensor_;
    AscendC::LocalTensor<ElementP> l0ATensor_[L0_STAGES];
    AscendC::LocalTensor<ElementV> l0BTensor_[L0_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor_[L0_STAGES];

    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    TileMmad tileMmad;

    Arch::Resource<ArchTag> *resourcePtr_;
    uint32_t l1BaseAddr_;
    uint32_t vBufBytes_;
    uint32_t l1PStageBytes_;
    // K dim (block valid size) of the V resident loaded by LoadVResident. operator() must build
    // the L1 V layout with THIS K, not the gemm's causalValidLen (which is <= block valid size and
    // only scopes the subtile). zN's N-fractal stride = RoundUp(K,16)*16 depends on this K; if the
    // layout uses causalValidLen while the L1 data was laid out for block valid size, the strides
    // diverge and the L0B transpose copy reads V with a wrong N-fractal stride -> N>=16 of
    // O_partial corrupt (first N-C0 fractal correct, rest wrong). Matches QK where resident K (D)
    // equals the gemm K, so QK never hit this.
    uint32_t residentValidSize_ = 0;

    __aicore__ inline BlockMmadPVSplitKvArch35()
        : resourcePtr_(nullptr),
          l1BaseAddr_(0),
          vBufBytes_(0),
          l1PStageBytes_(0)
    {}

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    __aicore__ inline void Init(Arch::Resource<ArchTag> &resource, uint32_t l1StartAddr, uint32_t blockSize, uint32_t D,
                                uint32_t groupSize)
    {
        vBufBytes_ = blockSize * D * sizeof(ElementV);
        l1BaseAddr_ = l1StartAddr;
        l1PStageBytes_ = RoundUp(groupSize, 16U) * RoundUp(blockSize, ELE_NUM_PER_C0) * sizeof(ElementP);
        resourcePtr_ = &resource;
        l1VTensor_ = resource.l1Buf.template GetBufferByByte<ElementV>(l1StartAddr);
        l1PTensor_ = resource.l1Buf.template GetBufferByByte<ElementP>(l1StartAddr + vBufBytes_);

        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor_[i] = resource.l0ABuf.template GetBufferByByte<ElementP>(L0A_BUF_SIZE * i);
            l0BTensor_[i] = resource.l0BBuf.template GetBufferByByte<ElementV>(L0B_BUF_SIZE * i);
            l0CTensor_[i] =
                resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_HALF_SIZE + L0C_BUF_SIZE * i);
        }
    }

    __aicore__ inline void SetL1PBuf(uint32_t l1PBufId)
    {
        l1PTensor_ = resourcePtr_->l1Buf.template GetBufferByByte<ElementP>(l1BaseAddr_ + vBufBytes_ +
                                                                            l1PStageBytes_ * l1PBufId);
    }

    template <class TensorV>
    __aicore__ inline void LoadVResident(TensorV &gmVTensor, uint32_t validSize, uint32_t D)
    {
        residentValidSize_ = validSize;
        using CopyGmToL1V = typename TileCopy::template CopyGmToL1B<TensorV>;
        CopyGmToL1V copyGmToL1V;

        auto l1VLayout = tla::MakeLayout<ElementV, LayoutTagL1B>(validSize, D);
        auto l1VTensorTla = tla::MakeTensor(l1VTensor_, l1VLayout, Arch::PositionL1{});
        auto l1VTile = GetTile(l1VTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(validSize, D));
        auto gmVTile = GetTile(gmVTensor, tla::MakeCoord(0, 0), tla::MakeShape(validSize, D));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        copyGmToL1V(l1VTile, gmVTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);

        // V resident in L0B[1] (stage 1, flag 3): copy L1->L0B once per task, co-located with
        // the GM->L1 load (above). De-aliased from QK's K (L0B[0], flag 2). Wait<M_MTE1>(3):
        // L0B[1] free from the prev task's last PV batch drain (InitSyncFlags primes first
        // task). Set<MTE1_M>(3): L0B[1] ready; operator()'s first PV batch (bDe==0) Wait<MTE1_M>(3)
        // consumes it before MMA; the last PV batch's Set<M_MTE1>(3) drains for the next task.
        auto l0BVLayout = tla::MakeLayout<ElementV, LayoutTagL0B>(validSize, D);
        auto l0BVTensor = tla::MakeTensor(l0BTensor_[1], l0BVLayout, Arch::PositionL0B{});
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3U);
        copyL1ToL0B(l0BVTensor, l1VTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(3U);
    }

    // Batched PV: M = groupCount * groupRows (up to 128 = 8 groups), K = validSize (the
    // block's valid size, shared by every qToken in the batch), N=D. P is rectangular
    // [batchM, validSize] with per-group causal tails [cvl[g], validSize) == 0 (softmax
    // pre-zeros them), so reducing over the uniform K=validSize yields valid exp + zero tail.
    // K equals residentValidSize_ (the V resident load's K), so PV reads it directly -- no
    // separate K param, and P/V K strides are consistent by construction. One matmul produces
    // O_partial[batchM, D] in L0C; then it is scattered to per-qToken GM workspace slots
    // [groupRows, D] (each batched qToken has its own (qToken, slotK)). wsOOff is computed on
    // demand from the batch's qToken/slotK ring row (== SlotOOffset(qToken,kvHeadIdx,slotK) =
    // qToken*perQTokenStride + kvHeadBase + slotK*slotOElems); the kernel precomputes
    // perQTokenStride (=kvHeads*topK*slotOElems) and kvHeadBase (=kvHeadIdx*topK*slotOElems),
    // constant within a kvHead, so no wsOOff array is marshalled. M is derived from
    // groupCount*groupRows (groupRows runtime constant), matching BlockMmadQK.
    template <class TensorO>
    __aicore__ inline void operator()(TensorO &gAccumOut, const uint32_t *qTokens, const uint32_t *slotKs,
                                      uint32_t groupCount, uint32_t groupRows, uint32_t D, uint64_t perQTokenStride,
                                      uint64_t kvHeadBase, uint64_t slotOElems, uint32_t numBatches, uint32_t bDe,
                                      Arch::CrossCoreFlag &smToMm2Flag)
    {
        uint32_t M = groupCount * groupRows;
        uint32_t N = D;
        uint32_t K = residentValidSize_;
        WaitCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);

        auto gmOLayout = tla::MakeLayout<ElementO, LayoutTagO>(groupRows, N);
        auto gmOSlotEx = tla::MakeTensor(gAccumOut, gmOLayout, Arch::PositionGM{});
        using CopyL0CToGm = typename TileCopy::template CopyL0CToDst<decltype(gmOSlotEx)>;
        CopyL0CToGm copyL0CToGm;
        (void)gmOSlotEx;

        auto l1PLayout = tla::MakeLayout<ElementP, LayoutTagL1A>(M, K);
        auto l1PTensorTla = tla::MakeTensor(l1PTensor_, l1PLayout, Arch::PositionL1{});
        // V L1 layout must describe the resident (K = block valid size). The gemm K now equals
        // residentValidSize_ (P was zero-padded to validSize by softmax), so P and V share the
        // same K; zN N-fractal strides are consistent by construction (else srcStride in the
        // L0B transpose copy is off -> N>=16 bad).
        auto l1VLayout = tla::MakeLayout<ElementV, LayoutTagL1B>(residentValidSize_, N);
        auto l1VTensorTla = tla::MakeTensor(l1VTensor_, l1VLayout, Arch::PositionL1{});

        uint32_t nL0Num = CeilDiv(N, L0_TILE_N);
        uint32_t mL0Num = CeilDiv(M, L0_TILE_M);
        uint32_t kL0Num = CeilDiv(K, L0_TILE_K);
        for (uint32_t nL0Itr = 0; nL0Itr < nL0Num; nL0Itr++) {
            uint32_t nAct = (nL0Itr == nL0Num - 1) ? (N - nL0Itr * L0_TILE_N) : L0_TILE_N;
            // Plan B: key L0C stage by bDe (nL0Num=1 always since N=D=128=L0_TILE_N). Old
            // nL0Itr%2 keying pinned one stage -> PV fixpipe(bi) blocked MMA(bi+1) on the same
            // upper-half L0C buffer. With bDe%2 the two upper-half L0C stages ping-pong across
            // batches so PV's O-partial fixpipe(bi) overlaps PV-MMA(bi+1); same-stage reuse
            // (bDe vs bDe+2) stays serialized by the Wait<FIX_M>(bDe%2+2) at MMA entry, keeping
            // the GM O-partial scatter dst safe. Flag lifecycle unchanged: InitSyncFlags primes
            // FIX_M(2,3) once at kernel start, ReleaseSyncFlags drains once at end, and the
            // per-batch Wait<FIX_M>@252 / Set<FIX_M>@294 chain across batches/tasks.
            uint32_t l0CBufId = bDe % L0_STAGES;
            uint32_t l0CEventId = l0CBufId + 2;

            auto l0CLayout = tla::MakeLayoutL0C(M, nAct);
            auto l0CTensorTla = tla::MakeTensor(l0CTensor_[l0CBufId], l0CLayout, Arch::PositionL0C{});
            for (uint32_t mL0Itr = 0; mL0Itr < mL0Num; mL0Itr++) {
                uint32_t mAct = (mL0Itr == mL0Num - 1) ? (M - mL0Itr * L0_TILE_M) : L0_TILE_M;
                auto l0CTile = GetTile(l0CTensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, 0), tla::MakeShape(mAct, nAct));
                for (uint32_t kL0Itr = 0; kL0Itr < kL0Num; kL0Itr++) {
                    uint32_t kAct = (kL0Itr == kL0Num - 1) ? (K - kL0Itr * L0_TILE_K) : L0_TILE_K;
                    uint32_t l0ABufId = kL0Itr % L0_STAGES;
                    // V resident in L0B[1] (stage 1, offset 32KB, flag 3): de-aliased from
                    // QK's K (L0B[0], flag 2). Fixed stage 1 -- not kL0Itr%2 (kL0Num=1 anyway).
                    uint32_t l0BBufId = 1U;
                    uint32_t l0AEventId = l0ABufId;
                    uint32_t l0BEventId = l0BBufId + 2;

                    auto l0BLayout = tla::MakeLayout<ElementV, LayoutTagL0B>(kAct, nAct);
                    auto l0BTensorTla = tla::MakeTensor(l0BTensor_[l0BBufId], l0BLayout, Arch::PositionL0B{});
                    // V resident in L0B[1] (stage 1, flag 3): the L1->L0B copy lives in
                    // LoadVResident (once per task, co-located with GM->L1). De-aliased from QK's
                    // K (L0B[0], flag 2). Only the MMA-side gate + drain remain here:
                    // firstPVBatch (bDe==0) Wait<MTE1_M>(3) gates MMA on the load-done Set<MTE1_M>(3)
                    // from LoadVResident; lastPVBatch (bDe+1>=numBatches) Set<M_MTE1>(3) drains
                    // L0B[1] free for the next task's LoadVResident. Middle batches read resident.
                    bool firstPVBatch = (bDe == 0U);
                    bool lastPVBatch = (bDe + 1U >= numBatches);

                    auto l1PSubTile = GetTile(l1PTensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, kL0Itr * L0_TILE_K),
                                              tla::MakeShape(mAct, kAct));
                    auto l0ALayout = tla::MakeLayout<ElementP, LayoutTagL0A>(mAct, kAct);
                    auto l0ATensorTla = tla::MakeTensor(l0ATensor_[l0ABufId], l0ALayout, Arch::PositionL0A{});
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    copyL1ToL0A(l0ATensorTla, l1PSubTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);

                    if ((mL0Itr == mL0Num - 1U) && (nL0Itr == nL0Num - 1U) && (kL0Itr == kL0Num - 1U)) {
                        SetCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
                    }

                    bool initMmad = (kL0Itr == 0);
                    uint32_t mAligned = RoundUp(mAct, 16);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    if (firstPVBatch) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                    }
                    if (mL0Itr == 0 && kL0Itr == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                    }
                    tileMmad(l0CTile, l0ATensorTla, l0BTensorTla, mAligned, nAct, kAct, initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    if (lastPVBatch) {
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    }
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            uint32_t mFixPAligned8Grp = RoundUp(groupRows, 8U);
            uint32_t nFixPAligned8 = RoundUp(nAct, 8U);
            // Scatter L0C[batchM, nAct] to per-qToken GM slots [groupRows, nAct]. wsOOff is
            // computed on demand from the batch's qToken/slotK (see operator() comment).
            // GM O_partial layout per slot is [groupRows, D]; only the valid groupRows rows are
            // written (not mFixPAligned8Grp) to avoid fixpipe overflow into the next slot.
            for (uint32_t g = 0; g < groupCount; g += COPY_GRANULARITY) {
                uint64_t wsOOff = static_cast<uint64_t>(qTokens[g]) * perQTokenStride + kvHeadBase +
                                  static_cast<uint64_t>(slotKs[g]) * slotOElems;
                uint16_t ndNum = 1;
                uint16_t srcNdStride = 1;
                uint32_t dstNdStride = 1;
                if (g + 1 < groupCount) {
                    uint32_t nextQ = qTokens[g + 1];
                    uint32_t nextSlot = slotKs[g + 1];
                    if (nextQ > qTokens[g]) {
                        dstNdStride = static_cast<int32_t>(nextQ - qTokens[g]) * perQTokenStride +
                                      static_cast<int32_t>(nextSlot - slotKs[g]) * static_cast<int32_t>(slotOElems);
                    } else {
                        dstNdStride = static_cast<int32_t>(qTokens[g] - nextQ) * perQTokenStride +
                                      static_cast<int32_t>(slotKs[g] - nextSlot) * static_cast<int32_t>(slotOElems);
                        wsOOff = static_cast<uint64_t>(nextQ) * perQTokenStride + kvHeadBase +
                                 static_cast<uint64_t>(nextSlot) * slotOElems;
                    }
                    srcNdStride = groupRows;
                    ndNum = COPY_GRANULARITY;
                }
                auto gmOSlot = tla::MakeTensor(gAccumOut[wsOOff], gmOLayout, Arch::PositionGM{});
                auto oTile =
                    GetTile(gmOSlot, tla::MakeCoord(0, nL0Itr * L0_TILE_N), tla::MakeShape(groupRows, nFixPAligned8));
                auto l0CTile =
                    GetTile(l0CTensorTla, tla::MakeCoord(g * groupRows, 0), tla::MakeShape(mFixPAligned8Grp, nAct));
                copyL0CToGm(oTile, l0CTile, ndNum, srcNdStride, dstNdStride);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }
    }
};

} // namespace NpuArch::Gemm::Block

#endif // GEMM_BLOCK_MMAD_PV_SPLIT_KV_ARCH35_HPP
