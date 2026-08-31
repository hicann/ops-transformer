/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Prefill QK matmul (KV-centric). FixPipe / cross-core sync: S is split to two AIV UB buffers.
 */
#ifndef GEMM_BLOCK_MMAD_QK_SPLIT_KV_ARCH35_HPP
#define GEMM_BLOCK_MMAD_QK_SPLIT_KV_ARCH35_HPP

#include <type_traits>
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

template <class ElementQ_, class ElementK_, class ElementS_>
struct BlockMmadQKSplitKvArch35 {
    using DispatchPolicy = MmadAtlasA5SplitKvQK;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementQ = ElementQ_;
    using ElementK = ElementK_;
    using ElementS = ElementS_;
    using ElementAccumulator = float;

    using LayoutTagQ = layout::RowMajor;
    using LayoutTagK = layout::ColumnMajor;
    using LayoutTagS = layout::RowMajor;

    using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutTagQ, ElementK, LayoutTagK, ElementS,
                                                       LayoutTagS, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    using TileMmad = Gemm::Tile::TileMmadTla<ArchTag, ElementQ, LayoutTagL1A>;

    static constexpr uint32_t L0_STAGES = 2;
    static constexpr uint32_t L0_TILE_M = 128;
    static constexpr uint32_t L0_TILE_N = 128;
    static constexpr uint32_t L0_TILE_K = 128;
    static constexpr uint32_t L0A_BUF_SIZE = ArchTag::L0A_SIZE / L0_STAGES;
    static constexpr uint32_t L0B_BUF_SIZE = ArchTag::L0B_SIZE / L0_STAGES;
    static constexpr uint32_t L0C_HALF_SIZE = ArchTag::L0C_SIZE / 2;
    static constexpr uint32_t L0C_BUF_SIZE = L0C_HALF_SIZE / L0_STAGES;
    static constexpr uint32_t L1_Q_STAGES = 1;
    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16;
    static constexpr uint32_t COPY_GRANULARITY = 2;
    static constexpr uint32_t C0_SIZE = BYTE_PER_C0 / sizeof(ElementQ);

    AscendC::LocalTensor<ElementK> l1KTensor_;
    AscendC::LocalTensor<ElementQ> l1QTensor_[L1_Q_STAGES];
    AscendC::LocalTensor<ElementQ> l0ATensor_[L0_STAGES];
    AscendC::LocalTensor<ElementK> l0BTensor_[L0_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor_[L0_STAGES];

    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    TileMmad tileMmad;

    uint32_t l1KBufBytes_;

    __aicore__ inline BlockMmadQKSplitKvArch35() {}

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

    __aicore__ inline void Init(Arch::Resource<ArchTag> &resource, uint32_t blockSize, uint32_t D)
    {
        uint32_t kBufBytes = blockSize * D * sizeof(ElementK);
        uint32_t qBufBytes = L0_TILE_M * D * sizeof(ElementQ);
        l1KBufBytes_ = kBufBytes;

        l1KTensor_ = resource.l1Buf.template GetBufferByByte<ElementK>(0);
        for (uint32_t i = 0; i < L1_Q_STAGES; i++) {
            l1QTensor_[i] = resource.l1Buf.template GetBufferByByte<ElementQ>(kBufBytes + qBufBytes * i);
        }
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor_[i] = resource.l0ABuf.template GetBufferByByte<ElementQ>(L0A_BUF_SIZE * i);
            l0BTensor_[i] = resource.l0BBuf.template GetBufferByByte<ElementK>(L0B_BUF_SIZE * i);
            l0CTensor_[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_BUF_SIZE * i);
        }
    }

    template <class TensorK>
    __aicore__ inline void LoadKResident(TensorK &gmKTensor, uint32_t validSize, uint32_t D)
    {
        using CopyGmToL1K = typename TileCopy::template CopyGmToL1B<TensorK>;
        CopyGmToL1K copyGmToL1K;

        auto l1KLayout = tla::MakeLayout<ElementK, LayoutTagL1B>(D, validSize);
        auto l1KTensorTla = tla::MakeTensor(l1KTensor_, l1KLayout, Arch::PositionL1{});
        auto l1KTile = GetTile(l1KTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(D, validSize));
        auto gmKTile = GetTile(gmKTensor, tla::MakeCoord(0, 0), tla::MakeShape(D, validSize));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        copyGmToL1K(l1KTile, gmKTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

        // K resident in L0B[0] (stage 0, flag 2): copy L1->L0B once per task, co-located with
        // the GM->L1 load (above). De-aliased from PV's V (L0B[1], flag 3). Wait<M_MTE1>(2):
        // L0B[0] free from the prev task's last QK batch drain (InitSyncFlags primes first
        // task). Set<MTE1_M>(2): L0B[0] ready; operator()'s first QK batch (bi==0) Wait<MTE1_M>(2)
        // consumes it before MMA; the last QK batch's Set<M_MTE1>(2) drains for the next task.
        auto l0BKLayout = tla::MakeLayout<ElementK, LayoutTagL0B>(D, validSize);
        auto l0BKTensor = tla::MakeTensor(l0BTensor_[0], l0BKLayout, Arch::PositionL0B{});
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2U);
        copyL1ToL0B(l0BKTensor, l1KTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(2U);
    }

    // Batched QK: M = groupCount * groupRows (up to 128 = 8 groups). groupCount scattered
    // gmQ slabs [groupRows, K] are gathered into contiguous L1 Q rows [g*groupRows].
    // qBases[g] is the GM offset of query[qToken, qHeadStart, :] (TND or BNSD).
    // qGmRowStride is the GM distance between consecutive Q heads of the same token:
    // TND/PA/BSND = D, BNSD = S * D. Layout shape uses that stride so Nd2Nz srcDValue gathers
    // the GQA group in one copy.
    template <class TensorQ, class TensorS>
    __aicore__ inline void operator()(TensorQ &gQ, const uint64_t *qBases, uint32_t groupCount, uint32_t groupRows,
                                      TensorS &ubSTensor, uint32_t validSize, uint32_t D, uint32_t qGmRowStride,
                                      uint32_t numBatches, uint32_t bi, Arch::CrossCoreFlag &mm1ToSmFlag)
    {
        uint32_t M = groupCount * groupRows;
        uint32_t N = validSize;
        uint32_t K = D;

        auto gmQLayout = tla::MakeLayout<ElementQ, LayoutTagQ>(groupRows, qGmRowStride);
        auto gmQEx = tla::MakeTensor(gQ, gmQLayout, Arch::PositionGM{});
        using CopyGmToL1Q = typename TileCopy::template CopyGmToL1A<decltype(gmQEx)>;
        CopyGmToL1Q copyGmToL1Q;
        (void)gmQEx;
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorS>;
        CopyL0CToDst copyL0CToDstSub0;
        CopyL0CToDst copyL0CToDstSub1;

        auto l1QLayout = tla::MakeLayout<ElementQ, LayoutTagL1A>(M, K);
        auto l1QTensorTla = tla::MakeTensor(l1QTensor_[0], l1QLayout, Arch::PositionL1{});
        auto l1KLayout = tla::MakeLayout<ElementK, LayoutTagL1B>(K, N);
        auto l1KTensorTla = tla::MakeTensor(l1KTensor_, l1KLayout, Arch::PositionL1{});

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        for (uint32_t g = 0; g < groupCount; g += COPY_GRANULARITY) {
            uint64_t qBase = qBases[g];
            uint64_t srcNdMatrixStride = 0;
            uint64_t dstNzMatrixStride = 0;
            uint32_t ndNum = 1;
            if (g + 1 < groupCount) {
                uint64_t qBaseNext = qBases[g + 1];
                // 950平台，srcNdMatrixStride不限制小于65536
                if (qBaseNext > qBase) {
                    srcNdMatrixStride = qBaseNext - qBase;
                } else {
                    srcNdMatrixStride = qBase - qBaseNext;
                    qBase = qBaseNext;
                }
                dstNzMatrixStride = groupRows * C0_SIZE;
                ndNum = COPY_GRANULARITY;
            }
            auto gmQTensor = tla::MakeTensor(gQ[qBase], gmQLayout, Arch::PositionGM{});
            auto gmQTile = GetTile(gmQTensor, tla::MakeCoord(0, 0), tla::MakeShape(groupRows, K));
            auto l1QTile = GetTile(l1QTensorTla, tla::MakeCoord(g * groupRows, 0), tla::MakeShape(groupRows, K));
            copyGmToL1Q(l1QTile, gmQTile, ndNum, srcNdMatrixStride, dstNzMatrixStride);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);

        uint32_t nL0Num = CeilDiv(N, L0_TILE_N);
        uint32_t mL0Num = CeilDiv(M, L0_TILE_M);
        uint32_t kL0Num = CeilDiv(K, L0_TILE_K);

        for (uint32_t nL0Itr = 0; nL0Itr < nL0Num; nL0Itr++) {
            uint32_t nAct = (nL0Itr == nL0Num - 1) ? (N - nL0Itr * L0_TILE_N) : L0_TILE_N;
            // Plan B: key L0C stage by bi (nL0Num=1 always since N=validSize<=blockSize=128=
            // L0_TILE_N, so nL0Itr is always 0 -> the old nL0Itr%2 keying pinned a single L0C
            // stage and forced fixpipe(bi) to block MMA(bi+1) on the same buffer). With bi%2 the
            // two L0C stages ping-pong across batches: MMA(bi)->L0C[bi%2] runs in parallel with
            // fixpipe(bi-1) draining L0C[(bi-1)%2]. Same-stage reuse (bi vs bi+2) is still
            // serialized by the Wait<FIX_M>(bi%2) at MMA entry, so the UB S fixpipe dst is safe.
            // Flag lifecycle is unchanged: InitSyncFlags primes FIX_M(0..3) once at kernel start
            // (so bi=0,1 don't deadlock), ReleaseSyncFlags drains once at kernel end, and the
            // per-batch Wait<FIX_M>@254 / Set<FIX_M>@288 chain fixpipe done->next same-stage MMA
            // across batches and tasks.
            uint32_t l0CBufId = bi % L0_STAGES;
            uint32_t l0CEventId = l0CBufId;

            auto l0CLayout = tla::MakeLayoutL0C(M, nAct);
            auto l0CTensorTla = tla::MakeTensor(l0CTensor_[l0CBufId], l0CLayout, Arch::PositionL0C{});

            for (uint32_t mL0Itr = 0; mL0Itr < mL0Num; mL0Itr++) {
                uint32_t mAct = (mL0Itr == mL0Num - 1) ? (M - mL0Itr * L0_TILE_M) : L0_TILE_M;
                auto l0CTile = GetTile(l0CTensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, 0), tla::MakeShape(mAct, nAct));

                for (uint32_t kL0Itr = 0; kL0Itr < kL0Num; kL0Itr++) {
                    uint32_t kAct = (kL0Itr == kL0Num - 1) ? (K - kL0Itr * L0_TILE_K) : L0_TILE_K;
                    uint32_t l0ABufId = kL0Itr % L0_STAGES;
                    uint32_t l0BBufId = kL0Itr % L0_STAGES;
                    uint32_t l0AEventId = l0ABufId;
                    uint32_t l0BEventId = l0BBufId + 2;

                    auto l1QSubTile = GetTile(l1QTensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, kL0Itr * L0_TILE_K),
                                              tla::MakeShape(mAct, kAct));
                    auto l0ALayout = tla::MakeLayout<ElementQ, LayoutTagL0A>(mAct, kAct);
                    auto l0ATensorTla = tla::MakeTensor(l0ATensor_[l0ABufId], l0ALayout, Arch::PositionL0A{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    copyL1ToL0A(l0ATensorTla, l1QSubTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);

                    auto l0BLayout = tla::MakeLayout<ElementK, LayoutTagL0B>(kAct, nAct);
                    auto l0BTensorTla = tla::MakeTensor(l0BTensor_[l0BBufId], l0BLayout, Arch::PositionL0B{});
                    // K resident in L0B[0] (stage 0, flag 2): the L1->L0B copy lives in
                    // LoadKResident (once per task, co-located with GM->L1). De-aliased from PV's
                    // V (L0B[1], flag 3) -> QK<->PV L0B[0] serialization gone; QK<->PV now by L0A
                    // flag 0 (Q/P aliased per-batch). Only the MMA-side gate + drain remain here:
                    // firstQKBatch (bi==0) Wait<MTE1_M>(2) gates MMA on the load-done Set<MTE1_M>(2)
                    // from LoadKResident; lastQKBatch (bi+1>=numBatches) Set<M_MTE1>(2) drains
                    // L0B[0] free for the next task's LoadKResident. Middle batches read resident.
                    bool firstQKBatch = (bi == 0U);
                    bool lastQKBatch = (bi + 1U >= numBatches);

                    bool initMmad = (kL0Itr == 0);
                    uint32_t mAligned = RoundUp(mAct, 16);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    if (firstQKBatch) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                    }
                    if (mL0Itr == 0 && kL0Itr == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                    }
                    tileMmad(l0CTile, l0ATensorTla, l0BTensorTla, mAligned, nAct, kAct, initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    if (lastQKBatch) {
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    }
                }
            }

            if (nL0Itr == 0) {
                WaitCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
            }
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);

            uint32_t nFixPAligned16 = RoundUp(nAct, 16U);
            // Split S across the 2 AIVs by GROUP (not M/2) so no group straddles a sub-core
            // (matches softmax/ScatterBatchStats group-dim split). groupRows-aligned; asymmetric
            // for odd groupCount: AIV0 gets ceil(groupCount/2) groups, AIV1 the rest (0 if only 1).
            uint32_t gSplit = CeilDiv(groupCount, 2U);
            uint32_t mSub0 = gSplit * groupRows;
            uint32_t mSub1 = (groupCount - gSplit) * groupRows;
            if constexpr (std::is_same<ElementS, float>::value) {
                // A5 NoQuant L0C→UB: dualDstCtl=1 deadlocks with CrossCore (Fixpipe waits
                // for both AIVs while they wait for mm1ToSmFlag after Fixpipe). Sequential
                // subBlockId=1 is the original 20003 MTE exception. Write the full S tile
                // to AIV0 only; high-prec softmax / ScatterBatchStats are AIV0-only.
                auto ubSTile =
                    GetTile(ubSTensor, tla::MakeCoord(0, nL0Itr * L0_TILE_N), tla::MakeShape(M, nFixPAligned16));
                auto l0CTile = GetTile(l0CTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(M, nAct));
                copyL0CToDstSub0(ubSTile, l0CTile, false);
                (void)mSub0;
                (void)mSub1;
            } else {
                auto ubSTileSub0 =
                    GetTile(ubSTensor, tla::MakeCoord(0, nL0Itr * L0_TILE_N), tla::MakeShape(mSub0, nFixPAligned16));
                auto l0CTileSub0 = GetTile(l0CTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(mSub0, nAct));
                copyL0CToDstSub0(ubSTileSub0, l0CTileSub0, false);
                if (mSub1 > 0U) {
                    auto ubSTileSub1 = GetTile(ubSTensor, tla::MakeCoord(0, nL0Itr * L0_TILE_N),
                                               tla::MakeShape(mSub1, nFixPAligned16));
                    auto l0CTileSub1 = GetTile(l0CTensorTla, tla::MakeCoord(mSub0, 0), tla::MakeShape(mSub1, nAct));
                    copyL0CToDstSub1(ubSTileSub1, l0CTileSub1, true);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }
        SetCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    }
};

} // namespace NpuArch::Gemm::Block

#endif // GEMM_BLOCK_MMAD_QK_SPLIT_KV_ARCH35_HPP
