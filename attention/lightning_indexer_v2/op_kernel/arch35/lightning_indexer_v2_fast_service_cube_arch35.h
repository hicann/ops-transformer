/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIGHTNING_INDEXER_FAST_SERVICE_CUBE_ARCH35_H
#define LIGHTNING_INDEXER_FAST_SERVICE_CUBE_ARCH35_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lightning_indexer_v2_common_arch35.h"
#include "lightning_indexer_v2_fast_common_arch35.h"

namespace LIV2Kernel {

template <typename LIT>
class LightningIndexerV2FastServiceCube {
public:
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;

    __aicore__ inline void InitBuffers(TPipe* pipe)
    {
        pipe->InitBuffer(bufUB_, LIV2Fast::kQkBankPlaneBytes);
        pipe->InitBuffer(bufQL1_, LIV2Fast::kDoubleBuffer * LIV2Fast::kQl1Bytes);
        pipe->InitBuffer(bufKeyL1_, LIV2Fast::kDoubleBuffer * LIV2Fast::kKl1Bytes);
        pipe->InitBuffer(bufQL0_, LIV2Fast::kDoubleBuffer * LIV2Fast::kQl1Bytes);
        pipe->InitBuffer(bufKeyL0_, LIV2Fast::kDoubleBuffer * LIV2Fast::kL0bBytes);
        pipe->InitBuffer(bufL0C_, LIV2Fast::kDoubleBuffer * LIV2Fast::kL0cBytes);
        queryL1_ = bufQL1_.Get<Q_T>();
        keyL1_ = bufKeyL1_.Get<K_T>();
        queryL0_ = bufQL0_.Get<Q_T>();
        keyL0_ = bufKeyL0_.Get<K_T>();
        cL0_ = bufL0C_.Get<float>();
        mm1ResUB_ = bufUB_.Get<float>();
    }

    __aicore__ inline void InitParams(const LIV2Fast::FastConstInfo& constInfo)
    {
        constInfo_ = constInfo;
    }

    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<Q_T>& queryGm, const GlobalTensor<K_T>& keyGm)
    {
        queryGm_ = queryGm;
        keyGm_ = keyGm;
    }

    __aicore__ inline void InitPageAttention(const LIV2TilingData* tiling, __gm__ uint8_t* blockTable)
    {
        if constexpr (LIT::pageAttention) {
            blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
            blockSize_ = tiling->blockSize;
            maxBlocks_ = tiling->maxBlockNumPerBatch;
            keyStride_ = tiling->keyStride0;
        }
    }

    __aicore__ inline void SetBatchIndex(uint32_t batch)
    {
        batchIndex_ = batch;
    }

    __aicore__ inline void Process()
    {
        auto* queryAddress = queryGm_.GetPhyAddr(0);
        auto* keyAddress = keyGm_.GetPhyAddr(0);
        const auto& info = constInfo_;
        const uint32_t s1 = info.s1Size;
        // TPipe publishes these events for its matmul lifetime. Borrow them
        // before starting the existing per-tile event protocol.
        WaitFlag<HardEvent::M_MTE1>(LIV2Fast::FastEvent::QUERY_SLOT_0);
        WaitFlag<HardEvent::M_MTE1>(LIV2Fast::FastEvent::QUERY_SLOT_1);
        WaitFlag<HardEvent::M_MTE1>(LIV2Fast::FastEvent::KEY_SLOT_0);
        AllocEventID();
        const uint32_t tileEnd = info.queryTileEnd;
        for (uint32_t tile = info.firstQueryTile; tile < tileEnd; ++tile) {
            auto* queryTileAddress =
                queryAddress + static_cast<uint64_t>(tile) * LIV2Fast::kTileS1 * info.headCount * LIV2Fast::kHeadDim;
            validQueryRows_ = min(static_cast<uint32_t>(LIV2Fast::kTileS1), s1 - tile * LIV2Fast::kTileS1);
            // Each Q8 tile uses two Q4 slots: each query contributes G matrix rows.
            // Full tiles pack G tightly; tail tiles reserve 32 rows per query.
            queryResultRows_ = LIV2Fast::kQuerySlotRows * LIV2Fast::GetQueryHeadStride(info.headCount, validQueryRows_);
            // Round up to the 16-row NZ matrix unit. Padding adds no valid queries.
            queryMatrixRows_ = (queryResultRows_ + (LIV2Fast::kNzBlockElements - 1U)) / LIV2Fast::kNzBlockElements *
                               LIV2Fast::kNzBlockElements;
            ComputeMm1(queryTileAddress, keyAddress, LIV2Fast::GetQueryTileScanSize(info, tile), tile + 1U >= tileEnd);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(LIV2Fast::FastEvent::KEY_SLOT_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(LIV2Fast::FastEvent::KEY_SLOT_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(LIV2Fast::FastEvent::QUERY_SLOT_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(LIV2Fast::FastEvent::QUERY_SLOT_1);
        // Return the borrowed events for TPipe destruction.
        SetFlag<HardEvent::M_MTE1>(LIV2Fast::FastEvent::QUERY_SLOT_0);
        SetFlag<HardEvent::M_MTE1>(LIV2Fast::FastEvent::QUERY_SLOT_1);
        SetFlag<HardEvent::M_MTE1>(LIV2Fast::FastEvent::KEY_SLOT_0);
    }

private:
    __aicore__ inline void AllocEventID()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(LIV2Fast::FastEvent::QUERY_SLOT_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(LIV2Fast::FastEvent::QUERY_SLOT_1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(LIV2Fast::FastEvent::KEY_SLOT_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(LIV2Fast::FastEvent::KEY_SLOT_1);
    }

    __aicore__ inline void AllocTileEventID()
    {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(LIV2Fast::FastEvent::QUERY_SLOT_0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(LIV2Fast::FastEvent::QUERY_SLOT_1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(LIV2Fast::FastEvent::KEY_SLOT_0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(LIV2Fast::FastEvent::KEY_SLOT_1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(LIV2Fast::FastEvent::ACCUMULATOR_SLOT_0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(LIV2Fast::FastEvent::ACCUMULATOR_SLOT_1);
    }

    template <int32_t SLOT>
    __aicore__ inline void LoadQueryToL0a(__gm__ Q_T* q)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SLOT);
        GlobalTensor<Q_T> queryTile;
        queryTile.SetGlobalBuffer(q);
        Nd2NzParams copyParams;
        copyParams.ndNum = 1;
        copyParams.nValue = queryResultRows_;
        copyParams.dValue = LIV2Fast::kHeadDim;
        copyParams.srcDValue = LIV2Fast::kHeadDim;
        copyParams.dstNzC0Stride = queryMatrixRows_;
        copyParams.dstNzNStride = 1;
        if (validQueryRows_ == LIV2Fast::kTileS1) {
            DataCopy(queryL1_[SLOT * LIV2Fast::kQl1Bytes / sizeof(Q_T)],
                     queryTile[static_cast<uint64_t>(SLOT) * queryResultRows_ * LIV2Fast::kHeadDim], copyParams);
        } else {
            // Reserve a 32-head NZ group for each query, including G < 32. Keep one group per
            // query so Fixpipe and the vector bank-plane layout stay shared.
            copyParams.nValue = constInfo_.headCount;
            for (uint32_t row = 0; row < LIV2Fast::kQuerySlotRows; ++row) {
                DataCopy(queryL1_[SLOT * LIV2Fast::kQl1Bytes / sizeof(Q_T) +
                                  row * LIV2Fast::kHeads * LIV2Fast::kNzBlockElements],
                         queryTile[((static_cast<uint64_t>(SLOT) * LIV2Fast::kQuerySlotRows + row < validQueryRows_) ?
                                        static_cast<uint64_t>(SLOT) * LIV2Fast::kQuerySlotRows + row :
                                        0U) *
                                   constInfo_.headCount * LIV2Fast::kHeadDim],
                         copyParams);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(SLOT);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(SLOT);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SLOT);
        LoadData2DParamsV2 loadParams;
        loadParams.mStep = queryMatrixRows_ / LIV2Fast::kNzBlockElements;
        loadParams.kStep = LIV2Fast::kHeadDim / LIV2Fast::kNzBlockElements;
        loadParams.srcStride = queryMatrixRows_ / LIV2Fast::kNzBlockElements;
        loadParams.dstStride = queryMatrixRows_ / LIV2Fast::kNzBlockElements;
        LoadData(queryL0_[SLOT * LIV2Fast::kQl1Bytes / sizeof(Q_T)], queryL1_[SLOT * LIV2Fast::kQl1Bytes / sizeof(Q_T)],
                 loadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SLOT);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(SLOT);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(SLOT);
    }

    template <int32_t SLOT>
    __aicore__ inline void KeyNd2Nz(__gm__ K_T* k, uint64_t keyOffset)
    {
        constexpr int32_t flag = LIV2Fast::FastEvent::KEY_SLOT_0 + SLOT;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(flag);
        GlobalTensor<K_T> keyTile;
        keyTile.SetGlobalBuffer(k);
        Nd2NzParams copyParams;
        copyParams.ndNum = 1;
        copyParams.nValue =
            min(static_cast<uint32_t>(LIV2Fast::kKTile), static_cast<uint32_t>(constInfo_.s2Size - keyOffset));
        copyParams.dValue = LIV2Fast::kHeadDim;
        copyParams.srcDValue = LIV2Fast::kHeadDim;
        copyParams.dstNzC0Stride = LIV2Fast::kKTile;
        copyParams.dstNzNStride = 1;
        if constexpr (LIT::pageAttention) {
            const uint32_t validKeys = copyParams.nValue;
            uint32_t copied = 0;
            while (copied < validKeys) {
                const uint64_t logicalKey = keyOffset + copied;
                const uint64_t logicalBlock = logicalKey / blockSize_;
                const uint64_t blockOffset = logicalKey % blockSize_;
                const uint32_t physicalBlock =
                    blockTableGm_.GetValue(static_cast<uint64_t>(batchIndex_) * maxBlocks_ + logicalBlock);
                copyParams.nValue = min(validKeys - copied, static_cast<uint32_t>(blockSize_ - blockOffset));
                // NZ C0 has 16 elements; copied counts key rows, not bytes.
                DataCopy(keyL1_[SLOT * LIV2Fast::kKl1Bytes / sizeof(K_T) + copied * LIV2Fast::kNzBlockElements],
                         keyTile[static_cast<uint64_t>(physicalBlock) * keyStride_ +
                                 static_cast<uint64_t>(blockOffset) * LIV2Fast::kHeadDim],
                         copyParams);
                copied += copyParams.nValue;
            }
        } else {
            DataCopy(keyL1_[SLOT * LIV2Fast::kKl1Bytes / sizeof(K_T)],
                     keyTile[static_cast<uint64_t>(keyOffset) * LIV2Fast::kHeadDim], copyParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(flag);
    }

    template <int32_t B_SLOT, int32_t Q_SLOT, int32_t C_SLOT, int32_t N_BLOCK, int32_t K_SLOT>
    __aicore__ inline void ComputeL0c()
    {
        constexpr int32_t bflag = LIV2Fast::FastEvent::KEY_SLOT_0 + B_SLOT;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bflag);
        LoadData2DParamsV2 loadParams;
        loadParams.mStep = LIV2Fast::kKSubtile / LIV2Fast::kNzBlockElements;
        loadParams.kStep = LIV2Fast::kHeadDim / LIV2Fast::kNzBlockElements;
        loadParams.srcStride = LIV2Fast::kKTile / LIV2Fast::kNzBlockElements;
        loadParams.dstStride = LIV2Fast::kKSubtile / LIV2Fast::kNzBlockElements;
        const uint64_t keyOffset = static_cast<uint64_t>(K_SLOT) * LIV2Fast::kKl1Bytes / sizeof(K_T) +
                                   (N_BLOCK * LIV2Fast::kKSubtile / LIV2Fast::kNzBlockElements) *
                                       LIV2Fast::kNzBlockElements * LIV2Fast::kNzBlockElements;
        LoadData(keyL0_[B_SLOT * LIV2Fast::kL0bBytes / sizeof(K_T)], keyL1_[keyOffset], loadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(bflag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(bflag);
        MmadParams params;
        params.m = queryMatrixRows_;
        params.n = LIV2Fast::kKSubtile;
        params.k = LIV2Fast::kHeadDim;
        params.disableGemv = true;
        params.cmatrixInitVal = true;
        const uint64_t resultOffset = static_cast<uint64_t>(C_SLOT) * LIV2Fast::kL0cBytes / sizeof(float) +
                                      (N_BLOCK & 1) * queryMatrixRows_ * LIV2Fast::kKSubtile;
        Mmad(cL0_[resultOffset], queryL0_[Q_SLOT * LIV2Fast::kQl1Bytes / sizeof(Q_T)],
             keyL0_[B_SLOT * LIV2Fast::kL0bBytes / sizeof(K_T)], params);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bflag);
    }

    template <int32_t OUTPUT_SLOT, int32_t C_SLOT, int32_t Q_SLOT, int32_t GROUP, int32_t K_SLOT>
    __aicore__ inline void Fixp()
    {
        // Both companion AIVs must have released this physical bank plane.
        AscendC::CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(static_cast<uint16_t>(OUTPUT_SLOT));
        AscendC::CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(
            static_cast<uint16_t>(OUTPUT_SLOT + LIV2Fast::FastEvent::COMPANION_AIV_OFFSET));
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(C_SLOT);

        ComputeL0c<0, Q_SLOT, C_SLOT, LIV2Fast::kKSubtilesPerFixTile * GROUP, K_SLOT>();
        ComputeL0c<1, Q_SLOT, C_SLOT, LIV2Fast::kKSubtilesPerFixTile * GROUP + 1, K_SLOT>();
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(C_SLOT);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(C_SLOT);

        FixpipeParamsC310<CO2Layout::ROW_MAJOR> params;
        // N width of each Fixpipe fragment: K256 is emitted as four K64 fragments.
        params.nSize = LIV2Fast::kFixpipeFragmentKeys;
        // Split complete queries between AIVs; exclude the matrix-end padding.
        params.mSize = queryResultRows_;
        params.srcStride = queryMatrixRows_;
        params.dstStride = LIV2Fast::kQHalfRows;
        params.dualDstCtl = 1;
        params.params.ndNum = LIV2Fast::kFixTile / LIV2Fast::kFixpipeFragmentKeys;
        params.params.srcNdStride = queryMatrixRows_ * LIV2Fast::kFixpipeSourceStrideFactor;
        params.params.dstNdStride = LIV2Fast::kFixpipeFragmentKeys * LIV2Fast::kQHalfRows;
        Fixpipe<float, float, LI_CFG_ROW_MAJOR_UB>(
            mm1ResUB_[OUTPUT_SLOT * LIV2Fast::kFixpipeFragmentKeys],
            cL0_[static_cast<uint64_t>(C_SLOT) * LIV2Fast::kL0cBytes / sizeof(float)], params);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(C_SLOT);
        AscendC::CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(static_cast<uint16_t>(OUTPUT_SLOT));
        AscendC::CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(
            static_cast<uint16_t>(OUTPUT_SLOT + LIV2Fast::FastEvent::COMPANION_AIV_OFFSET));
    }

    template <int32_t K_SLOT>
    __aicore__ inline void ProcessKeyTile()
    {
        constexpr int32_t flag = LIV2Fast::FastEvent::KEY_SLOT_0 + K_SLOT;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(flag);
        Fixp<0, 0, 0, 0, K_SLOT>();
        Fixp<1, 1, 1, 0, K_SLOT>();
        Fixp<0, 0, 0, 1, K_SLOT>();
        Fixp<1, 1, 1, 1, K_SLOT>();
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(flag);
    }

    __aicore__ inline void ComputeMm1(__gm__ Q_T* q, __gm__ K_T* k, uint32_t s2, bool drainOutput)
    {
        AscendC::SetMMLayoutTransform(true);
        AllocTileEventID();
        LoadQueryToL0a<0>(q);
        LoadQueryToL0a<1>(q);

        for (uint64_t n = 0; n < s2; n += LIV2Fast::kDoubleBuffer * LIV2Fast::kKTile) {
            KeyNd2Nz<0>(k, n);
            if (n + LIV2Fast::kKTile < s2) {
                KeyNd2Nz<1>(k, n + LIV2Fast::kKTile);
                ProcessKeyTile<0>();
                ProcessKeyTile<1>();
            } else {
                ProcessKeyTile<0>();
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(LIV2Fast::FastEvent::QUERY_SLOT_1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(LIV2Fast::FastEvent::QUERY_SLOT_0);
        if (drainOutput) {
            AscendC::CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(
                static_cast<uint16_t>(LIV2Fast::FastEvent::QK_SLOT_0));
            AscendC::CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(
                static_cast<uint16_t>(LIV2Fast::FastEvent::QK_SLOT_0 + LIV2Fast::FastEvent::COMPANION_AIV_OFFSET));
            AscendC::CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(
                static_cast<uint16_t>(LIV2Fast::FastEvent::QK_SLOT_1));
            AscendC::CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(
                static_cast<uint16_t>(LIV2Fast::FastEvent::QK_SLOT_1 + LIV2Fast::FastEvent::COMPANION_AIV_OFFSET));
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(LIV2Fast::FastEvent::ACCUMULATOR_SLOT_0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(LIV2Fast::FastEvent::ACCUMULATOR_SLOT_1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(LIV2Fast::FastEvent::KEY_SLOT_0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(LIV2Fast::FastEvent::KEY_SLOT_1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(LIV2Fast::FastEvent::QUERY_SLOT_0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(LIV2Fast::FastEvent::QUERY_SLOT_1);
        AscendC::SetMMLayoutTransform(false);
    }

    static constexpr FixpipeConfig LI_CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<int32_t> blockTableGm_;
    uint32_t blockSize_ = 0, maxBlocks_ = 0, batchIndex_ = 0;
    uint64_t keyStride_ = 0;
    LIV2Fast::FastConstInfo constInfo_{};
    uint32_t validQueryRows_ = LIV2Fast::kTileS1;
    uint32_t queryResultRows_ = LIV2Fast::kQHalfRows;
    uint32_t queryMatrixRows_ = LIV2Fast::kQHalfRows;
    TBuf<TPosition::A1> bufQL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    TBuf<TPosition::A2> bufQL0_;
    TBuf<TPosition::B2> bufKeyL0_;
    TBuf<TPosition::CO1> bufL0C_;
    TBuf<TPosition::VECCALC> bufUB_;
    LocalTensor<Q_T> queryL1_, queryL0_;
    LocalTensor<K_T> keyL1_, keyL0_;
    LocalTensor<float> cL0_, mm1ResUB_;
};

} // namespace LIV2Kernel
#endif
